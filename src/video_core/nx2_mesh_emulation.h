// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <optional>
#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/mesh_emulation.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"

// Host side of the task/mesh emulation (EDEN_NX2_MESH_EMU=1): what a native mesh draw looks
// like in Maxwell3D state, how big the scratch has to be, and the launch description that runs
// the rewritten programs through the ordinary compute path. The pieces are pure functions over
// engine state; the rasterizer sequence that uses them is:
//
//   1. PrepareDraw sees a native draw whose slot 1 program is a task or mesh program
//      (Describe() returns a value): skip the graphics pipeline lookup for it.
//   2. Ensure a guest-visible scratch of ScratchPlan::bytes and write Params at its head.
//   3. Task present: dispatch the rewritten task program with TaskLaunch(); then dispatch the
//      rewritten mesh program with MeshLaunch() (one workgroup per task workgroup * MeshPerTask,
//      the prologue exits the surplus). No task: MeshLaunch() alone.
//   4. Draw ReplayVertexCount() vertices as a triangle list with the passthrough vertex program
//      bound in slot 1 and the title's fragment program untouched; culled slots produce w = 0.
namespace VideoCommon::Nx2MeshEmulation {

using Maxwell3D = Tegra::Engines::Maxwell3D;
using Layout = Shader::Ampere::MeshEmulation::Layout;

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_MESH_EMU");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

// SET_MESH_CONTROL / SET_MESH_SHADER_A / SET_MESH_SHADER_B / SET_MESH_INIT_SHADER (methods
// 0x114c-0x1158), which Eden's register map leaves as padding.
struct MeshRegisters {
    static constexpr u32 ControlMethod = 0x114c / 4;
    static constexpr u32 ShaderAMethod = 0x1150 / 4;
    static constexpr u32 ShaderBMethod = 0x1154 / 4;
    static constexpr u32 InitShaderMethod = 0x1158 / 4;

    u32 control;
    u32 topology;         ///< 0 points, 1 lines, 2 triangles
    u32 max_vertices;     ///< per mesh workgroup
    u32 max_primitives;   ///< per mesh workgroup
    u32 mesh_shared_lines;
    u32 mesh_threads;
    u32 task_threads;
    u32 task_local_lines;
    u32 task_output_lines;

    static MeshRegisters Decode(const Maxwell3D::Regs& regs) {
        const u32 a = regs.reg_array[ShaderAMethod];
        const u32 b = regs.reg_array[ShaderBMethod];
        const u32 init = regs.reg_array[InitShaderMethod];
        return MeshRegisters{
            .control = regs.reg_array[ControlMethod],
            .topology = a & 0xf,
            .max_vertices = (a >> 4) & 0x1ff,
            .max_primitives = (a >> 13) & 0x3ff,
            .mesh_shared_lines = (b >> 8) & 0x1ff,
            .mesh_threads = (b >> 20) & 0x3f,
            .task_threads = init & 0x3f,
            .task_local_lines = (init >> 12) & 0x1ff,
            .task_output_lines = (init >> 22) & 0x1ff,
        };
    }
};

// A native mesh draw as NVN2 issues it: SET_DRAW_CONTROL_A topology POINTS and a vertex-array
// begin/end whose count is the number of task workgroups (or mesh workgroups without a task),
// the task-or-mesh program in slot 1 and, when a task exists, the mesh program in slot 3.
struct DrawDescription {
    GPUVAddr task_program;     ///< 0 without a task program
    GPUVAddr mesh_program;
    u32 workgroups;            ///< launch count: task workgroups, or mesh workgroups without a task
    MeshRegisters mesh;

    bool HasTask() const { return task_program != 0; }
    u32 TaskWorkgroups() const { return HasTask() ? workgroups : 0; }
    u32 MeshWorkgroups() const { return HasTask() ? workgroups * Layout::MeshPerTask : workgroups; }
};

inline GPUVAddr SlotProgramAddress(const Maxwell3D::Regs& regs, size_t slot) {
    return (GPUVAddr{regs.pipelines[slot].reserved[0] & 0xff} << 32) | regs.pipelines[slot].reserved[1];
}

// Whether the draw in flight is a mesh draw; the caller supplies what the shader cache learned
// about the slot 1 program (Analyze() of its native code), since the registers alone do not say.
inline std::optional<DrawDescription> Describe(const Maxwell3D& maxwell3d,
                                               Shader::Ampere::MeshEmulation::ProgramKind slot1) {
    using Shader::Ampere::MeshEmulation::ProgramKind;
    const auto& regs = maxwell3d.regs;
    if (slot1 == ProgramKind::None) return std::nullopt;
    DrawDescription description{
        .task_program = 0,
        .mesh_program = 0,
        .workgroups = maxwell3d.draw_manager.draw_state.vertex_buffer.count,
        .mesh = MeshRegisters::Decode(regs),
    };
    if (slot1 == ProgramKind::Task) {
        description.task_program = SlotProgramAddress(regs, 1);
        description.mesh_program = SlotProgramAddress(regs, 3);
        if (description.mesh_program == 0) return std::nullopt;
    } else {
        description.mesh_program = SlotProgramAddress(regs, 1);
    }
    if (description.workgroups == 0 || description.mesh_program == 0) return std::nullopt;
    return description;
}

// Scratch for one draw: Params, then the task regions, then the mesh regions.
struct ScratchPlan {
    static constexpr u64 ParamsBytes = 0x100;
    u64 bytes;
    u64 task_offset;
    u64 mesh_offset;
    u32 primitive_slots_shift; ///< log2 of the primitive slots replayed per mesh workgroup

    static ScratchPlan For(const DrawDescription& draw) {
        // The MAP region fits 340 triangles; the hardware limit is what SET_MESH_SHADER_A says.
        // The title's mesh programs emit two triangles per lane, so never replay fewer than 64
        // slots in case the register decode is off.
        const u32 primitives = std::clamp<u32>(std::max<u32>(draw.mesh.max_primitives, 64), 1,
                                               (Layout::MapBytes / 4 - 4) / 3);
        ScratchPlan plan{};
        plan.primitive_slots_shift = static_cast<u32>(std::bit_width(primitives - 1));
        plan.task_offset = ParamsBytes;
        plan.mesh_offset = plan.task_offset + u64{draw.TaskWorkgroups()} * Layout::TaskRegionBytes;
        plan.bytes = plan.mesh_offset + u64{draw.MeshWorkgroups()} * Layout::MeshRegionBytes;
        return plan;
    }

    Layout::Params Params(GPUVAddr scratch, const DrawDescription& draw) const {
        return Layout::Params{
            .task_base = scratch + task_offset,
            .task_bytes = mesh_offset - task_offset,
            .mesh_base = scratch + mesh_offset,
            .mesh_bytes = bytes - mesh_offset,
            .primitive_slots_shift = primitive_slots_shift,
            // Per-primitive attribute groups follow the vertex groups: max_vertices rounded up
            // to 32-lane groups, each the size of one attribute group set. The generator only
            // replays them when the GS-slot header (per-primitive output map) is available.
            .primitive_region = ((std::max<u32>(draw.mesh.max_vertices, 1) + 31) / 32) * 128,
            .reserved = {},
        };
    }

    u32 ReplayVertexCount(const DrawDescription& draw) const {
        return draw.MeshWorkgroups() * (1U << primitive_slots_shift) * 3;
    }
};

// A launch description for the ordinary compute path: 32 threads per workgroup, the slot's
// constant buffers 0-7 as the title bound them for the graphics stage, and the scratch Params
// block as bank Layout::params_bank. `program_start` is relative to KeplerCompute code_loc.
using LaunchParams = Tegra::Engines::KeplerCompute::LaunchParams;
using ConstBuffers = std::array<Tegra::Engines::ConstBufferInfo, Maxwell3D::Regs::MaxConstBuffers>;

inline LaunchParams Launch(u32 program_start, u32 workgroups, const ConstBuffers& bound,
                           GPUVAddr params_address, const Layout& layout) {
    LaunchParams qmd{};
    qmd.program_start = program_start;
    qmd.grid_dim_x.Assign(workgroups);
    qmd.grid_dim_y.Assign(1);
    qmd.grid_dim_z.Assign(1);
    qmd.block_dim_x.Assign(32);
    qmd.block_dim_y.Assign(1);
    qmd.block_dim_z.Assign(1);
    u32 enable_mask = 0;
    for (size_t index = 0; index < qmd.const_buffer_config.size(); ++index) {
        GPUVAddr address = 0;
        u32 size = 0;
        if (index == layout.params_bank) {
            address = params_address;
            size = ScratchPlan::ParamsBytes;
        } else if (bound[index].enabled && bound[index].size != 0) {
            address = bound[index].address;
            size = bound[index].size;
        } else {
            continue;
        }
        auto& config = qmd.const_buffer_config[index];
        config.address_low = static_cast<u32>(address);
        config.address_high.Assign(static_cast<u32>(address >> 32));
        config.size.Assign(std::min<u32>(size, 0x1ffff));
        enable_mask |= 1U << index;
    }
    qmd.const_buffer_enable_mask.Assign(enable_mask);
    return qmd;
}

inline LaunchParams TaskLaunch(u32 program_start, const DrawDescription& draw, const ConstBuffers& bound,
                               GPUVAddr params_address, const Layout& layout) {
    return Launch(program_start, draw.TaskWorkgroups(), bound, params_address, layout);
}

inline LaunchParams MeshLaunch(u32 program_start, const DrawDescription& draw, const ConstBuffers& bound,
                               GPUVAddr params_address, const Layout& layout) {
    return Launch(program_start, draw.MeshWorkgroups(), bound, params_address, layout);
}

} // namespace VideoCommon::Nx2MeshEmulation
