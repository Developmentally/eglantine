// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Field layout reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

// Task/mesh (NV-style, Turing+) programs on a host without a mesh pipeline, behind
// EDEN_NX2_MESH_EMU=1. A task or mesh program is a 32-lane workgroup that talks to the hardware
// through the internal stage buffer (ISBE): ISBERD reads it, ISBEWR writes it. This header turns
// such a program into an ordinary compute program whose ISBE lives in a guest-visible scratch
// buffer (one region per workgroup), and generates the vertex program that replays the mesh
// output from that scratch as a plain triangle-list draw. Nothing here touches the host; see
// video_core/nx2_mesh_emulation.h for the draw sequence.
//
// ISBE addressing (hardware, matched against NAK's lowering and the title's programs):
//   ISBERD.ATTR.SKEW [0]            workgroup index (task, or mesh without a task)
//   ISBERD.ATTR      [0]            mesh workgroup index within its task launch (mesh with a task)
//   ISBERD.ATTR      [a]  a >= 4    task payload / launch parameters written by the task program
//   ISBEWR.O.ATTR    [4]            mesh workgroup count (task), [8..0x10] x/y/z, [0x10..] payload
//   ISBEWR.O.MAP     [3]            primitive count (mesh)
//   ISBEWR.O.MAP.U8  [4 + 3p + i]   vertex index i of primitive p (mesh, triangles)
//   ISBEWR.O.ATTR.SKEW [g*G + k*128 + l*4]   component k of vertex 32g+l; G = 128 * attribute
//                                   components in the program header's output map; per-primitive
//                                   attributes follow the vertex groups the same way
namespace Shader::Ampere::MeshEmulation {

enum class ProgramKind { None, Task, Mesh };

struct Analysis {
    ProgramKind kind = ProgramKind::None;
    bool has_task = false;            ///< a mesh program fed by a task program (reads the payload)
    bool writes_launch_count = false; ///< task: ISBEWR.O.ATTR [4]
    bool writes_primitive_count = false;
    bool writes_indices = false;
    u32 isbe_instructions = 0;
    u32 max_output_attribute = 0;     ///< highest ISBEWR.O.ATTR.SKEW immediate (+4)
    u32 max_payload = 0;              ///< highest ISBEWR.O.ATTR non-skew immediate (+4)
    u32 max_gpr = 0;                  ///< highest general register the program names
    u32 max_uniform = 0;              ///< highest uniform register the program names
    std::string_view error;           ///< an ISBE form the emulation cannot express
    explicit operator bool() const { return error.empty() && kind != ProgramKind::None; }
};

// Scratch layout shared with the host. Every task workgroup owns TaskRegionBytes at
// task_base + t * TaskRegionBytes; every mesh workgroup owns MeshRegionBytes at
// mesh_base + m * MeshRegionBytes. A mesh fed by a task is workgroup m = t * MeshPerTask + i
// (i < the count the task wrote), so no prefix sum is needed; tasks emitting more than
// MeshPerTask workgroups lose the rest (the title's tasks emit at most one per lane).
struct Layout {
    static constexpr u32 MapOffset = 0;        ///< MAP region: one dword per hardware byte address
    static constexpr u32 MapBytes = 0x1000;    ///< 1024 MAP bytes: count + indices of 340 triangles
    static constexpr u32 AttrOffset = 0x1000;  ///< ATTR region, hardware byte addresses
    static constexpr u32 MeshRegionBytes = 0x4000;
    static constexpr u32 TaskRegionBytes = 0x1000;
    static constexpr u32 MeshPerTask = 32;
    static constexpr u32 MeshRegionShift = std::countr_zero(MeshRegionBytes);
    static constexpr u32 TaskRegionShift = std::countr_zero(TaskRegionBytes);
    static constexpr u32 MeshPerTaskShift = std::countr_zero(MeshPerTask);
    static_assert(std::has_single_bit(MeshRegionBytes) && std::has_single_bit(TaskRegionBytes) &&
                  std::has_single_bit(MeshPerTask));

    // Parameter block the host binds as a constant buffer (bank/offset below) for the rewritten
    // compute programs and for the passthrough vertex program.
    // Each base is followed by its byte size: the buffer cache sizes a storage buffer from the
    // qword after its pointer when that qword looks like a size.
    struct Params {
        u64 task_base;             ///< 0x00  first task region
        u64 task_bytes;            ///< 0x08  extent of the task regions
        u64 mesh_base;             ///< 0x10  first mesh region
        u64 mesh_bytes;            ///< 0x18  extent of the mesh regions
        u32 primitive_slots_shift; ///< 0x20  log2 of primitives per workgroup the replay draws
        u32 primitive_region;      ///< 0x24  ATTR byte offset of the per-primitive attribute groups
        u32 reserved[2];
    };
    static_assert(sizeof(Params) == 0x30);
    static constexpr u32 TaskBaseOffset = 0x00;
    static constexpr u32 MeshBaseOffset = 0x10;
    static constexpr u32 SlotsShiftOffset = 0x20;
    static constexpr u32 PrimitiveRegionOffset = 0x24;
    u32 params_bank = 6;   ///< constant buffer bank holding Params (compute QMDs stop at 7)
    u32 params_offset = 0;
};

namespace Detail {
struct Isbe {
    bool write, output, skew;
    u32 access; ///< 0 MAP, 1 PATCH, 2 PRIM, 3 ATTR
    u32 size;   ///< 0 U8, 1 U16, 2 B32
    u32 base;   ///< offset register (255 = none)
    u32 imm;    ///< immediate byte address
    u32 reg;    ///< data (ISBEWR) or destination (ISBERD) register
    u32 predicate;
};
inline Isbe DecodeIsbe(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    return Isbe{
        .write = opcode == 0x927,
        .output = insn.Bits(79, 1) != 0,
        .skew = insn.Bits(78, 1) != 0,
        .access = insn.Bits(76, 2),
        .size = insn.Bits(74, 2),
        .base = insn.Bits(24, 8),
        .imm = insn.Bits(40, 16),
        .reg = opcode == 0x927 ? insn.Bits(32, 8) : insn.Bits(16, 8),
        .predicate = insn.Bits(12, 4),
    };
}
inline bool IsIsbe(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    return opcode == 0x923 || opcode == 0x927;
}
// Native encodings, copied from the title's own programs so the lowering accepts them unchanged.
inline Instruction S2R(u32 predicate, u32 dest, u32 special) {
    return {0x00007919U | (predicate << 12) | (dest << 16), 0, special << 8, 0x000e2400};
}
inline Instruction IMadImm(u32 predicate, u32 dest, u32 a, u32 imm, u32 c) {
    return {0x00007824U | (predicate << 12) | (dest << 16) | (a << 24), imm, 0x078e0200U | c, 0x000fca00};
}
inline Instruction IAdd3(u32 predicate, u32 dest, u32 a, u32 b) {
    return {0x00007210U | (predicate << 12) | (dest << 16) | (a << 24), b, 0x07ffe0ffU, 0x000fc800};
}
inline Instruction Lea(u32 predicate, u32 dest, u32 a, u32 b, u32 shift) {
    return {0x00007211U | (predicate << 12) | (dest << 16) | (a << 24), b, 0x078e00ffU | (shift << 11), 0x000fc800};
}
inline Instruction Lop3AndImm(u32 predicate, u32 dest, u32 a, u32 imm) {
    return {0x00007812U | (predicate << 12) | (dest << 16) | (a << 24), imm, 0x078ec0ffU, 0x000fc800};
}
inline Instruction ShfRightU32(u32 predicate, u32 dest, u32 source, u32 shift) {
    return {0x00007819U | (predicate << 12) | (dest << 16) | (255U << 24), shift, 0x00011600U | source, 0x000fc800};
}
inline Instruction UldcU64(u32 dest, u32 bank, u32 offset) {
    return {0x00007ab9U | (dest << 16), (offset << 6) | (bank << 22), 0x00000a00, 0x000fe200};
}
// STG.E / LDG.E through a uniform 64-bit base plus a 32-bit register offset and a 24-bit immediate.
inline Instruction StgUniform(u32 predicate, u32 uniform, u32 offset_reg, u32 imm, u32 data, u32 size) {
    return {0x00007986U | (predicate << 12) | (offset_reg << 24), data | (imm << 8),
            0x0810a100U | (size << 9) | uniform, 0x0003e800};
}
inline Instruction LdgUniform(u32 predicate, u32 uniform, u32 offset_reg, u32 imm, u32 dest, u32 size) {
    return {0x00007981U | (predicate << 12) | (dest << 16) | (offset_reg << 24), uniform | (imm << 8),
            0x081ea100U | (size << 9), 0x000ea200};
}
inline Instruction ISetpGeU32(u32 target, u32 a, u32 b) {
    return {0x0000720cU | (target << 16) | (a << 24), b, 0x03f06070, 0x000fda00};
}
inline Instruction Exit(u32 predicate) {
    return {0x0000094dU | (predicate << 12), 0, 0x03800000, 0x000fea00};
}
} // namespace Detail

inline Analysis Analyze(std::span<const Instruction> code) {
    using namespace Detail;
    Analysis result;
    const auto fail = [&](std::string_view reason) {
        result.error = reason;
        return result;
    };
    for (const auto& insn : code) {
        if (insn.Bits(12, 4) == 15) continue;
        const u32 opcode = insn.Bits(0, 12);
        // General registers: Rd at 16 and Ra at 24 everywhere outside the uniform datapath, Rb at
        // 32 in the register form (and the data of STG/ISBEWR), Rc at 64 for three-source ALU
        // forms; compares, selects, conversions and two-source float ops keep flags there.
        {
            const u32 base = opcode & 0x1ff, form = opcode >> 9;
            const bool uniform_op = (base >= 0x80 && base < 0x100) || base == 0x1c2 || base == 0x1c3;
            static constexpr std::array<u32, 14> no_rc{0x02, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
                                                        0x0a, 0x0b, 0x0c, 0x0d, 0x17, 0x1c, 0x20};
            const bool alu = base < 0x80 && !uniform_op;
            const bool has_rc = alu && base != 0x21 && std::ranges::find(no_rc, base) == no_rc.end();
            const auto note = [&](u32 bit) {
                const u32 reg = insn.Bits(bit, 8);
                if (reg != 255 && reg < 216) result.max_gpr = std::max(result.max_gpr, reg);
            };
            if (!uniform_op) { note(16); note(24); }
            if ((form == 1 && !uniform_op) || opcode == 0x927 || opcode == 0x986 || opcode == 0x386) note(32);
            if (has_rc) note(64);
        }
        // Uniform registers: the uniform datapath (0x280-0x2bf, 0x880-0x8bf) names them in every
        // operand field; R2UR/VOTEU/S2UR/ULDC write one; uniform-operand ALU forms (0xcxx, 0xdxx)
        // and uniform-pointer LDG/STG read one. Over-approximating is harmless.
        const auto note_uniform = [&](u32 bit, u32 extra = 0) {
            const u32 reg = insn.Bits(bit, 6);
            if (reg != 63) result.max_uniform = std::max(result.max_uniform, reg + extra);
        };
        const u32 low = opcode & 0xff, high = opcode & 0xf00;
        if ((high == 0x200 || high == 0x800) && low >= 0x80 && low < 0xc0) {
            note_uniform(16); note_uniform(24); note_uniform(64);
            if (high == 0x200) note_uniform(32); // the immediate forms carry imm32 there
        } else if (opcode == 0x3c2 || opcode == 0x3c4 || opcode == 0x886 || opcode == 0x9c3) {
            note_uniform(16);
        } else if (opcode == 0xab9 || opcode == 0xabb) {
            note_uniform(16, insn.Bits(73, 3) == 5 ? 1 : 0); // ULDC.64 names its pair by the even register
            if (opcode == 0xabb) note_uniform(24);
        } else if (high == 0xc00 || high == 0xd00 || high == 0xe00) {
            note_uniform(32);
        } else if (opcode == 0x981) {
            note_uniform(32, 1);
        } else if (opcode == 0x986) {
            note_uniform(64, 1);
        }
        if (!IsIsbe(insn)) continue;
        ++result.isbe_instructions;
        const Isbe isbe = DecodeIsbe(insn);
        if (isbe.access == 1 || isbe.access == 2) return fail("native stage buffer patch/primitive access");
        if (isbe.size > 2) return fail("native stage buffer access width");
        if (isbe.write) {
            if (!isbe.output) return fail("native stage buffer input write");
            if (isbe.access == 0) {
                result.kind = ProgramKind::Mesh;
                if (isbe.base == 255 && isbe.imm == 3) result.writes_primitive_count = true;
                else result.writes_indices = true;
            } else if (isbe.skew) {
                result.kind = ProgramKind::Mesh;
                result.max_output_attribute = std::max(result.max_output_attribute, isbe.imm + 4);
            } else {
                if (isbe.base == 255 && isbe.imm == 4) result.writes_launch_count = true;
                result.max_payload = std::max(result.max_payload, isbe.imm + 4);
            }
        } else if (isbe.output) {
            if (isbe.skew) return fail("native stage buffer skewed output read");
        } else {
            if (isbe.access != 3) return fail("native stage buffer input map read");
            if (isbe.skew) {
                if (isbe.base != 255 || isbe.imm != 0) return fail("native stage buffer skewed input address");
            } else if (isbe.base != 255 || isbe.imm != 0) {
                result.has_task = true; // payload or launch parameters of the task
            } else {
                result.has_task = true; // mesh workgroup index within its task launch
            }
        }
    }
    if (result.kind == ProgramKind::None && result.isbe_instructions) {
        // Only ATTR writes without a MAP: the task program of a task + mesh pair.
        result.kind = ProgramKind::Task;
        if (result.has_task) return fail("native task program reads a payload");
    }
    if (result.kind == ProgramKind::Task && !result.writes_launch_count)
        return fail("native task program never writes its mesh count");
    return result;
}

struct Rewritten {
    std::vector<Instruction> code;
    std::string_view error;
    explicit operator bool() const { return error.empty(); }
};

// Turn a task or mesh program into a compute program over the scratch layout: one 32-thread
// workgroup per task/mesh workgroup, SR_CTAID.X as the workgroup index. Relative branches are
// re-targeted around the inserted instructions.
inline Rewritten RewriteToCompute(std::span<const Instruction> code, const Analysis& analysis,
                                  const Layout& layout) {
    using namespace Detail;
    Rewritten result;
    const auto fail = [&](std::string_view reason) {
        result.code.clear();
        result.error = reason;
        return result;
    };
    if (!analysis) return fail(analysis.error.empty() ? "not a native task or mesh program" : analysis.error);
    if (analysis.max_gpr + 4 > 200) return fail("native task/mesh program leaves no scratch registers");
    const u32 uniform_base = (analysis.max_uniform + 2) & ~1U; // pair for the program's own base
    const u32 uniform_task = uniform_base + 2;                 // pair for the task base (mesh with task)
    if ((analysis.has_task ? uniform_task + 1 : uniform_base + 1) > 27)
        return fail("native task/mesh program leaves no scratch uniform registers");
    const u32 region_reg = analysis.max_gpr + 1;  // byte offset of this workgroup's region
    const u32 task_reg = analysis.max_gpr + 2;    // byte offset of the feeding task's region
    const u32 scratch_reg = analysis.max_gpr + 3;
    const bool mesh = analysis.kind == ProgramKind::Mesh;
    const u32 params = layout.params_offset;

    std::vector<u32> position(code.size() + 1); // original index -> rewritten index
    auto& out = result.code;
    // Prologue: region offset of this workgroup and the scratch base(s).
    out.push_back(S2R(7, region_reg, 37));
    out.push_back(UldcU64(uniform_base, layout.params_bank,
                          params + (mesh ? Layout::MeshBaseOffset : Layout::TaskBaseOffset)));
    if (analysis.has_task) {
        out.push_back(UldcU64(uniform_task, layout.params_bank, params + Layout::TaskBaseOffset));
        out.push_back(ShfRightU32(7, task_reg, region_reg, Layout::MeshPerTaskShift));
        out.push_back(IMadImm(7, task_reg, task_reg, Layout::TaskRegionBytes, 255));
        // Exit when this mesh workgroup is beyond the count its task launched, leaving a zero
        // primitive count in its region (the scratch is reused, the replay reads every region).
        out.push_back(Lop3AndImm(7, scratch_reg, region_reg, Layout::MeshPerTask - 1));
        out.push_back(LdgUniform(7, uniform_task, task_reg, 4, region_reg, 4));
        out.push_back(ISetpGeU32(0, scratch_reg, region_reg));
        out.push_back(S2R(7, region_reg, 37));
        out.push_back(IMadImm(0, region_reg, region_reg, Layout::MeshRegionBytes, 255));
        out.push_back(StgUniform(0, uniform_base, region_reg, Layout::MapOffset + 3 * 4, 255, 4));
        out.push_back(Exit(0));
    }
    out.push_back(IMadImm(7, region_reg, region_reg, mesh ? Layout::MeshRegionBytes : Layout::TaskRegionBytes, 255));

    for (u32 index = 0; index < code.size(); ++index) {
        position[index] = static_cast<u32>(out.size());
        const auto& insn = code[index];
        const u32 opcode = insn.Bits(0, 12);
        if (insn.Bits(12, 4) == 15 || !IsIsbe(insn)) {
            if (opcode == 0x944 || opcode == 0x949 || opcode == 0x958)
                return fail("native task/mesh program with calls or jump tables");
            out.push_back(insn);
            continue;
        }
        const Isbe isbe = DecodeIsbe(insn);
        const u32 p = isbe.predicate;
        // Byte address in the scratch region: MAP bytes become dwords.
        const auto address = [&](u32 uniform, u32 region, u32 region_offset) {
            u32 offset_reg = region;
            u32 imm = region_offset;
            if (isbe.access == 0) {
                imm += Layout::MapOffset + isbe.imm * 4;
                if (isbe.base != 255) { out.push_back(Lea(p, scratch_reg, isbe.base, region, 2)); offset_reg = scratch_reg; }
            } else {
                imm += isbe.imm;
                if (isbe.base != 255) { out.push_back(IAdd3(p, scratch_reg, isbe.base, region)); offset_reg = scratch_reg; }
            }
            return std::pair{uniform, std::pair{offset_reg, imm}};
        };
        const u32 width = isbe.access == 0 ? 4U : isbe.size == 2 ? 4U : isbe.size == 1 ? 2U : 0U;
        if (isbe.write) {
            const auto [uniform, where] = address(uniform_base, region_reg, isbe.access == 0 ? 0 : Layout::AttrOffset * mesh);
            out.push_back(StgUniform(p, uniform, where.first, where.second, isbe.reg, width));
        } else if (isbe.output) {
            const auto [uniform, where] = address(uniform_base, region_reg, isbe.access == 0 ? 0 : Layout::AttrOffset * mesh);
            out.push_back(LdgUniform(p, uniform, where.first, where.second, isbe.reg, width));
        } else if (isbe.skew || (!analysis.has_task && isbe.base == 255 && isbe.imm == 0)) {
            out.push_back(S2R(p, isbe.reg, 37));
        } else if (isbe.base == 255 && isbe.imm == 0) {
            out.push_back(S2R(p, isbe.reg, 37));
            out.push_back(Lop3AndImm(p, isbe.reg, isbe.reg, Layout::MeshPerTask - 1));
        } else {
            const auto [uniform, where] = address(uniform_task, task_reg, 0);
            out.push_back(LdgUniform(p, uniform, where.first, where.second, isbe.reg, width));
        }
    }
    position[code.size()] = static_cast<u32>(out.size());
    // Re-target relative branches (BRA, BSSY) whose distance changed.
    for (u32 index = 0; index < code.size(); ++index) {
        const auto& insn = code[index];
        const u32 opcode = insn.Bits(0, 12);
        if (insn.Bits(12, 4) == 15 || (opcode != 0x947 && opcode != 0x945)) continue;
        const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
        const s64 relative = static_cast<s64>(raw << 14) >> 14;
        const s64 target = s64{index} * 16 + 16 + relative;
        if (target < 0 || target % 16 || target / 16 > static_cast<s64>(code.size()))
            return fail("native task/mesh branch leaves the program");
        const s64 rewritten = (s64{position[static_cast<size_t>(target / 16)]} - s64{position[index]} - 1) * 16;
        auto& moved = out[position[index]];
        moved.words[1] = static_cast<u32>(rewritten);
        moved.words[2] = (moved.words[2] & ~0x3ffffU) | (static_cast<u32>(rewritten >> 32) & 0x3ffff);
    }
    return result;
}

// Output attribute components of a program header (Maxwell SPH layout, which the native 128-byte
// header shares): byte addresses in ascending order, the order of the skewed ISBE groups.
inline std::vector<u32> OutputAttributes(std::span<const u8> header) {
    std::vector<u32> attributes;
    if (header.size() < 80) return attributes;
    const auto bit = [&](u32 byte, u32 index) { return (header[byte + index / 8] >> (index % 8)) & 1; };
    for (u32 i = 0; i < 32; ++i) if (bit(50, i)) attributes.push_back(i * 4);            // system values A/B
    for (u32 i = 0; i < 128; ++i) if (bit(54, i)) attributes.push_back(0x80 + i * 4);    // generic 0-31
    for (u32 i = 0; i < 16; ++i) if (bit(70, i)) attributes.push_back(0x280 + i * 4);    // colors
    for (u32 i = 0; i < 16; ++i) if (bit(72, i)) attributes.push_back(0x2c0 + i * 4);    // system values C
    for (u32 i = 0; i < 40; ++i) if (bit(74, i)) attributes.push_back(0x300 + i * 4);    // fixed-function textures
    return attributes;
}

struct PassthroughLayout {
    std::vector<u32> vertex_attributes;    ///< OutputAttributes of the mesh program header
    std::vector<u32> primitive_attributes; ///< OutputAttributes of the per-primitive (GS slot) header
    Layout layout;
};

// Maxwell vertex program replaying one mesh workgroup's output per Params::primitive_slots_shift
// primitives: vertex id v -> primitive p = v / 3, workgroup m = p >> shift, slot p & mask; slots at
// or past the workgroup's primitive count draw a clipped triangle. Emits the position and every
// attribute the mesh program declares, plus the per-primitive attributes as flat vertex outputs.
inline LoweredProgram GeneratePassthroughVertexProgram(const PassthroughLayout& in) {
    LoweredProgram result;
    auto& code = result.instructions;
    constexpr u64 P = u64{7} << 16; // always
    const auto emit = [&](u64 word) { code.push_back(word); };
    const auto mov_imm = [&](u32 reg, u32 value) { emit(0x0100000000000000ULL | P | reg | (u64{15} << 12) | (u64{value} << 20)); };
    const auto mov_cbuf = [&](u32 reg, u32 bank, u32 offset) { emit(0x4c98000000000000ULL | P | reg | (u64{15} << 39) | (u64{offset / 4} << 20) | (u64{bank} << 34)); };
    const auto iadd = [&](u32 dest, u32 a, u32 b, bool negate_b = false, bool carry_out = false, bool carry_in = false) {
        emit(0x5c10000000000000ULL | P | dest | (u64{a} << 8) | (u64{b} << 20) | (u64{negate_b} << 48) | (u64{carry_out} << 47) | (u64{carry_in} << 43));
    };
    const auto iadd_imm = [&](u32 dest, u32 a, u32 imm) { emit(0x1c00000000000000ULL | P | dest | (u64{a} << 8) | (u64{imm & 0xffffffff} << 20)); };
    const auto shl_imm = [&](u32 dest, u32 a, u32 shift) { emit(0x3848000000000000ULL | P | dest | (u64{a} << 8) | (u64{shift} << 20)); };
    const auto shr_imm = [&](u32 dest, u32 a, u32 shift) { emit(0x3828000000000000ULL | P | dest | (u64{a} << 8) | (u64{shift} << 20)); };
    const auto shl_reg = [&](u32 dest, u32 a, u32 b) { emit(0x5c48000000000000ULL | P | dest | (u64{a} << 8) | (u64{b} << 20)); };
    const auto shr_reg = [&](u32 dest, u32 a, u32 b) { emit(0x5c28000000000000ULL | P | dest | (u64{a} << 8) | (u64{b} << 20)); };
    const auto and_imm = [&](u32 dest, u32 a, u32 imm) { emit(0x0400000000000000ULL | P | dest | (u64{a} << 8) | (u64{imm} << 20)); };
    const auto and_reg = [&](u32 dest, u32 a, u32 b) { emit(0x5c40000000000000ULL | P | dest | (u64{a} << 8) | (u64{b} << 20)); };
    const auto iscadd = [&](u32 dest, u32 a, u32 b, u32 shift) { emit(0x5c18000000000000ULL | P | dest | (u64{a} << 8) | (u64{b} << 20) | (u64{shift} << 39)); };
    const auto ldg = [&](u32 dest, u32 address, u32 offset, u32 size) { emit(0xeed0000000000000ULL | P | dest | (u64{address} << 8) | (u64{offset & 0xffffff} << 20) | (u64{1} << 45) | (u64{size} << 48)); };
    const auto ald = [&](u32 dest, u32 attribute) { emit(0xefd8000000000000ULL | P | dest | (u64{255} << 8) | (u64{attribute} << 20) | (u64{255} << 39)); };
    const auto ast = [&](u32 source, u32 attribute, u32 size) { emit(0xeff0000000000000ULL | P | source | (u64{255} << 8) | (u64{attribute} << 20) | (u64{255} << 39) | (u64{size} << 47)); };
    const auto sel = [&](u32 dest, u32 a, u32 b, u32 predicate) { emit(0x5ca0000000000000ULL | P | dest | (u64{a} << 8) | (u64{b} << 20) | (u64{predicate} << 39)); };
    // ISETP.<cmp>.U32.AND Pd, PT, Ra, Rb, PT
    const auto isetp = [&](u32 predicate, u32 cmp, u32 a, u32 b) { emit(0x5b60000000000000ULL | P | predicate | (u64{7} << 3) | (u64{a} << 8) | (u64{b} << 20) | (u64{7} << 39) | (u64{cmp} << 49)); };
    // Multiply by a compile-time constant through shifts and adds (constants here have few bits).
    const auto mul_const = [&](u32 dest, u32 a, u32 constant, u32 temp) {
        bool first = true;
        for (u32 b = 0; b < 32; ++b) {
            if (!((constant >> b) & 1)) continue;
            if (first) { shl_imm(dest, a, b); first = false; }
            else { shl_imm(temp, a, b); iadd(dest, dest, temp); }
        }
        if (first) mov_imm(dest, 0);
    };
    const u32 bank = in.layout.params_bank, params = in.layout.params_offset;
    // 64-bit address pairs (Tmp0/1, Base, VAddr, PAddr) must start at an even register.
    enum : u32 { Vid = 0, Prim = 1, Corner = 2, Group = 3, Slot = 4, Count = 5, Tmp0 = 6, Tmp1 = 7,
                 BaseLo = 8, BaseHi = 9, VAddrLo = 10, VAddrHi = 11, PAddrLo = 12, PAddrHi = 13,
                 Idx = 14, Cull = 15, Zero = 16, Val = 18 };

    ald(Vid, 0x2fc);                                        // vertex id
    emit(0x5cb8000000000000ULL | P | Tmp0 | (u64{2} << 8) | (u64{2} << 10) | (u64{Vid} << 20)); // I2F.F32.U32
    emit(0x1e00000000000000ULL | P | Tmp0 | (u64{Tmp0} << 8) | (u64{0x3eaaaaab} << 20));    // FMUL32I 1/3
    emit(0x5cb0000000000000ULL | P | Prim | (u64{2} << 8) | (u64{2} << 10) | (u64{Tmp0} << 20) | (u64{1} << 39)); // F2I.U32.F32.FLOOR
    iscadd(Tmp0, Prim, Prim, 1);                            // 3p
    iadd(Corner, Vid, Tmp0, true);                          // corner i = v - 3p
    mov_cbuf(Tmp1, bank, params + Layout::SlotsShiftOffset); // primitive slots shift
    shr_reg(Group, Prim, Tmp1);                             // workgroup m
    mov_imm(Tmp0, 1);
    shl_reg(Tmp0, Tmp0, Tmp1);
    iadd_imm(Tmp0, Tmp0, 0xffffffff);                       // mask
    and_reg(Slot, Prim, Tmp0);                              // primitive slot within the workgroup
    mov_cbuf(BaseLo, bank, params + Layout::MeshBaseOffset);
    mov_cbuf(BaseHi, bank, params + Layout::MeshBaseOffset + 4);
    shl_imm(Tmp0, Group, Layout::MeshRegionShift);
    mov_imm(Zero, 0);
    iadd(BaseLo, BaseLo, Tmp0, false, true);                // region base = mesh_base + m * MeshRegionBytes
    iadd(BaseHi, BaseHi, Zero, false, false, true);
    ldg(Count, BaseLo, Layout::MapOffset + 3 * 4, 4);       // primitive count
    isetp(0, 1, Slot, Count);                               // P0 = slot < count
    // vertex index: MAP dword 4 + 3 * slot + corner (the MAP region stores a dword per byte)
    iscadd(Tmp0, Slot, Slot, 1);                            // 3 * slot
    iadd(Tmp0, Tmp0, Corner);                               // 3 * slot + corner
    iscadd(Tmp0, Tmp0, Zero, 2);                            // 4 * (3 slot + corner)
    iadd(Tmp0, Tmp0, BaseLo, false, true);
    iadd(Tmp1, BaseHi, Zero, false, false, true);
    ldg(Idx, Tmp0, Layout::MapOffset + 4 * 4, 4);
    and_imm(Idx, Idx, 0xff);
    // vertex address: Attr + (idx >> 5) * G + (idx & 31) * 4
    const u32 group_bytes = static_cast<u32>(in.vertex_attributes.size()) * 128;
    shr_imm(Tmp0, Idx, 5);
    mul_const(VAddrLo, Tmp0, group_bytes, Tmp1);
    and_imm(Tmp0, Idx, 31);
    iscadd(VAddrLo, Tmp0, VAddrLo, 2);
    iadd(VAddrLo, VAddrLo, BaseLo, false, true);
    iadd(VAddrHi, BaseHi, Zero, false, false, true);
    mov_imm(Cull, 0x7f7fffff);                              // clipped position: x = FLT_MAX, w = 0
    for (size_t k = 0; k < in.vertex_attributes.size(); ++k) {
        const u32 attribute = in.vertex_attributes[k];
        ldg(Val, VAddrLo, Layout::AttrOffset + static_cast<u32>(k) * 128, 4);
        sel(Val, Val, attribute == 0x70 ? Cull : Zero, 0);  // keep when P0, else the clipped position
        ast(Val, attribute, 0);
    }
    if (!in.primitive_attributes.empty()) {
        const u32 primitive_group_bytes = static_cast<u32>(in.primitive_attributes.size()) * 128;
        mov_cbuf(Tmp1, bank, params + Layout::PrimitiveRegionOffset); // primitive region offset
        shr_imm(Tmp0, Slot, 5);
        mul_const(PAddrLo, Tmp0, primitive_group_bytes, Idx);
        and_imm(Tmp0, Slot, 31);
        iscadd(PAddrLo, Tmp0, PAddrLo, 2);
        iadd(PAddrLo, PAddrLo, Tmp1);
        iadd(PAddrLo, PAddrLo, BaseLo, false, true);
        iadd(PAddrHi, BaseHi, Zero, false, false, true);
        for (size_t j = 0; j < in.primitive_attributes.size(); ++j) {
            ldg(Val, PAddrLo, Layout::AttrOffset + static_cast<u32>(j) * 128, 4);
            sel(Val, Val, Zero, 0);
            ast(Val, in.primitive_attributes[j], 0);
        }
    }
    emit(0xe30000000000000fULL | P);                        // EXIT
    result.source_instruction_offsets.assign(1, 0);
    result.source_size = sizeof(Instruction);
    return result;
}

} // namespace Shader::Ampere::MeshEmulation
