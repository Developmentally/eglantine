// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Offline check of native (Switch 2) shader programs captured with EDEN_NX2_SHADER_DUMP: every
// program is lowered to Maxwell, translated to IR and emitted as SPIR-V the way the runtime does,
// without a GPU or a console title running. Failures are grouped by reason so a translator change
// can be validated against a whole corpus in seconds.
//
//   nx2-shader-check [-l] [-q] [-v] [-d] [-b baseline] <file.bin | directory>...
//     -l           lowering only (skip IR translation and SPIR-V emission)
//     -q           print the summary only
//     -v           describe each program: header inputs/outputs, IPA forms, descriptors
//     -d           dump the IR of each program after translation
//     -b <file>    baseline: one program file name per line that is allowed to fail; the exit
//                  status then reports only new failures, and fixed programs are listed
//     -t <spec>    draw table: "<cb>:<hex off>[,<cb>:<hex off>...]@<cb>:<hex off>" patches the
//                  storage buffers at those cbuf slots per gl_DrawID through a table at the
//                  last slot (Environment::DrawTable), to check the batched-draw variant;
//                  "@<cb>:<off>#<n>" reads the draw index from generic attribute n instead
//     -f <n>       vertex program of a batched pipeline: forward gl_DrawID on generic n

#include <algorithm>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <typeinfo>
#include <vector>

#include <fmt/format.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ampere/mesh_emulation.h"
#include "shader_recompiler/frontend/ampere/program_address_map.h"
#include "shader_recompiler/frontend/ampere/program_dump.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/shader_info.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace {

namespace fs = std::filesystem;
bool verbose = false;
bool dump_ir = false;
fs::path spirv_output;
Shader::Environment::DrawTable draw_table_spec;
using Shader::Ampere::Instruction;
using Shader::Ampere::ProgramDumpInfo;

// Mirrors GenericEnvironment's native path: Maxwell code built from the lowered instructions,
// with the program header in front and the native address map behind.
class DumpEnvironment final : public Shader::Environment {
public:
    DumpEnvironment(const ProgramDumpInfo& info, const Shader::Ampere::LoweredProgram& lowered,
                    std::span<const Instruction> native) {
        stage = static_cast<Shader::Stage>(info.stage);
        gp_passthrough_mask = info.gp_passthrough_mask;
        draw_table = draw_table_spec;
        // Task and mesh programs are dumped from the vertex slot; the dump carries neither the
        // mesh registers nor the per-primitive header, so translate them with the shape the
        // title's programs have: 32 lanes, triangles, 4 vertices and 2 primitives per lane.
        if (const auto analysis = Shader::Ampere::MeshEmulation::Analyze(native);
            stage != Shader::Stage::Compute && analysis) {
            using Shader::Ampere::MeshEmulation::ProgramKind;
            stage = analysis.kind == ProgramKind::Task ? Shader::Stage::Task : Shader::Stage::Mesh;
            mesh_registers = {4U | (128U << 4) | (64U << 13), 32U << 20, 32U | (1U << 12) | (1U << 22)};
        }
        std::memcpy(&sph, info.header.data(), sizeof(sph));
        start_address = sizeof(sph);
        is_proprietary_driver = true;
        texture_bound = lowered.required_texture_buffer != ~0U ? lowered.required_texture_buffer
                                                                : info.texture_bound;
        workgroup_size = info.workgroup_size;
        shared_memory_size = info.shared_memory_size;
        local_memory_size = info.local_memory_size;

        code.resize(sizeof(sph) / sizeof(u64));
        std::memcpy(code.data(), &sph, sizeof(sph));
        for (const u64 instruction : lowered.instructions) {
            if ((code.size() * sizeof(u64)) % 32 == 0) {
                code.push_back(0);
            }
            code.push_back(instruction);
        }
        code.push_back(0);
        Shader::Ampere::AppendNativeAddressMap(code, lowered.source_instruction_offsets);
    }

    u64 ReadInstruction(u32 address) override {
        const size_t index = address / sizeof(u64);
        if (index >= code.size()) {
            throw Shader::RuntimeError("Instruction read outside the program at {:#x}", address);
        }
        return code[index];
    }
    u32 TranslateProgramAddress(u32 address) const override {
        if (const auto translated = Shader::Ampere::NativeProgramAddress(code, address)) {
            return *translated;
        }
        throw Shader::RuntimeError("Invalid native indirect branch address {:#x}", address);
    }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bound; }
    u32 LocalMemorySize() const override { return local_memory_size; }
    u32 SharedMemorySize() const override { return shared_memory_size; }
    std::array<u32, 3> WorkgroupSize() const override { return workgroup_size; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::vector<u64> code;
    std::array<u32, 3> workgroup_size{};
    u32 texture_bound{};
    u32 shared_memory_size{};
    u32 local_memory_size{};
};

// A permissive Vulkan-class host, close to what RADV on the Steam Deck reports.
Shader::HostTranslateInfo HostInfo() {
    Shader::HostTranslateInfo host{};
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_int64 = true;
    host.needs_demote_reorder = false;
    host.support_snorm_render_buffer = true;
    host.support_viewport_index_layer = true;
    host.support_viewport_mask = false;
    host.support_geometry_shader_passthrough = false;
    host.support_conditional_barrier = true;
    host.min_ssbo_alignment = 4;
    host.ApplyDescriptorLimitPolicy();
    return host;
}

Shader::Profile HostProfile() {
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.unified_descriptor_binding = true;
    profile.support_descriptor_aliasing = true;
    profile.support_int8 = true;
    profile.support_uniform_and_storage_buffer_8bit = true;
    profile.support_storage_buffer_8bit = true;
    profile.support_int16 = true;
    profile.support_uniform_and_storage_buffer_16bit = true;
    profile.support_storage_buffer_16bit = true;
    profile.support_int64 = true;
    profile.support_float_controls = true;
    profile.support_separate_denorm_behavior = true;
    profile.support_separate_rounding_mode = true;
    profile.support_fp16_denorm_preserve = true;
    profile.support_fp32_denorm_preserve = true;
    profile.support_fp16_denorm_flush = true;
    profile.support_fp32_denorm_flush = true;
    profile.support_fp16_signed_zero_nan_preserve = true;
    profile.support_fp32_signed_zero_nan_preserve = true;
    profile.support_fp64_signed_zero_nan_preserve = true;
    profile.support_explicit_workgroup_layout = true;
    profile.support_workgroup_layout_8bit_access = true;
    profile.support_workgroup_layout_16bit_access = true;
    profile.support_quad_shuffles = true;
    profile.support_vote = true;
    profile.support_shuffle_relative = true;
    profile.support_viewport_index_layer_non_geometry = true;
    profile.support_typeless_image_loads = true;
    profile.support_demote_to_helper_invocation = true;
    profile.support_int64_atomics = true;
    profile.support_shared_int64_atomics = true;
    profile.support_derivative_control = true;
    profile.support_multi_viewport = true;
    profile.support_geometry_streams = true;
    profile.support_sampled_image_array_nonuniform_indexing = true;
    profile.support_storage_image_array_nonuniform_indexing = true;
    profile.support_uniform_texel_buffer_array_nonuniform_indexing = true;
    profile.support_storage_texel_buffer_array_nonuniform_indexing = true;
    return profile;
}

struct Outcome {
    std::string phase;   // "lower", "translate", "spirv" or empty on success
    std::string message; // the reason, with program-specific numbers kept
    std::string key;     // the reason with instruction words stripped, for grouping
    bool skipped_spirv = false;
};

std::string StripInstructionWords(std::string message) {
    // "... [op 305 00027305 00000002 0021e800 002fe200]" -> "... [op 305]"
    if (const auto open = message.rfind(" [op "); open != std::string::npos) {
        const auto space = message.find(' ', open + 5);
        if (space != std::string::npos) {
            message = message.substr(0, space) + "]";
        }
    }
    return message;
}

// -v: what the header made of the program (interpolation, inputs and outputs), to compare
// against what the shader's code evidently expects.
void DescribeProgram(const ProgramDumpInfo& info, const Shader::IR::Program& program,
                     std::span<const Instruction> native) {
    const auto& sph = *reinterpret_cast<const Shader::ProgramHeader*>(info.header.data());
    fmt::print("  stage {} header: imap_systemb {:#04x} omap target {:#x} local {:#x}\n",
               static_cast<u32>(program.stage), static_cast<u32>(sph.ps.imap_systemb.raw),
               sph.ps.omap.target, sph.LocalMemorySize());
    std::string inputs, outputs, colors;
    for (size_t index = 0; index < 32; ++index) {
        if (program.info.loads.Generic(index)) {
            const char* mode = program.info.interpolation[index] == Shader::Interpolation::Flat ? "flat"
                : program.info.interpolation[index] == Shader::Interpolation::NoPerspective ? "noperspective" : "smooth";
            inputs += fmt::format(" {}:{}", index, mode);
            if (program.stage == Shader::Stage::Fragment) {
                // Per-component SPH bits (2 each: 0 unused, 1 constant, 2 perspective, 3 linear).
                inputs += fmt::format("/{:02x}", sph.ps.imap_generic_vector[index].raw);
            }
        }
        if (program.info.stores.Generic(index)) outputs += fmt::format(" {}", index);
    }
    for (size_t index = 0; index < 8; ++index) {
        if (program.info.stores_frag_color[index]) colors += fmt::format(" {}", index);
    }
    fmt::print("  inputs:{}\n  outputs:{}\n  frag colors:{} depth {}\n", inputs, outputs, colors,
               program.info.stores_frag_depth);
    if (program.stage == Shader::Stage::Fragment) {
        // Native IPA forms: attribute byte address, sample mode (bits 76-77), constant (78).
        std::map<u32, std::set<std::string>> ipa;
        for (const auto& insn : native) {
            const u32 op = insn.Bits(0, 12);
            if (op != 0x326 && op != 0x526) continue;
            const u32 attribute = insn.Bits(64, 8) * 4;
            ipa[attribute].insert(fmt::format("{}{}{}", insn.Bits(78, 1) ? "const" : "interp",
                                              insn.Bits(76, 2) == 1 ? ".centroid" : insn.Bits(76, 2) ? ".offset" : "",
                                              op == 0x526 ? ".idx" : ""));
        }
        std::string text;
        for (const auto& [attribute, modes] : ipa) {
            text += fmt::format(" a[{:#x}]", attribute);
            for (const auto& mode : modes) text += ":" + mode;
        }
        fmt::print("  ipa:{}\n", text);
    }
    std::string textures;
    for (const auto& desc : program.info.texture_descriptors) {
        textures += fmt::format(" [cb{}+{:#x}{}{} type {} count {}]", desc.cbuf_index, desc.cbuf_offset,
                                desc.has_secondary ? fmt::format(" | cb{}+{:#x}", desc.secondary_cbuf_index,
                                                                 desc.secondary_cbuf_offset) : "",
                                desc.shift_left ? fmt::format(" <<{}", desc.shift_left) : "",
                                static_cast<u32>(desc.type), desc.count);
    }
    std::string buffers;
    for (const auto& desc : program.info.storage_buffers_descriptors) {
        buffers += fmt::format(" [cb{}+{:#x}{}]", desc.cbuf_index, desc.cbuf_offset,
                               desc.is_written ? " w" : "");
    }
    fmt::print("  textures:{}\n  ssbos:{} global {} indexed-attrs {}\n", textures, buffers,
               program.info.uses_global_memory, program.info.loads_indexed_attributes);
}

Outcome Check(const fs::path& path, bool lowering_only) {
    Outcome outcome;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<Instruction> native(size / sizeof(Instruction));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(native.data()), native.size() * sizeof(Instruction));

    const auto lowered = Shader::Ampere::LowerToMaxwell(native);
    if (!lowered) {
        const size_t failed = lowered.error_offset / sizeof(Instruction);
        const auto words = failed < native.size() ? native[failed].words : std::array<u32, 4>{};
        outcome.phase = "lower";
        outcome.message = fmt::format("instruction {:#x}: {} [op {:03x} {:08x} {:08x} {:08x} {:08x}]",
                                      lowered.error_offset, lowered.error, words[0] & 0xfff,
                                      words[0], words[1], words[2], words[3]);
        outcome.key = fmt::format("{} [op {:03x}]", lowered.error, words[0] & 0xfff);
        return outcome;
    }
    if (lowering_only) {
        return outcome;
    }

    ProgramDumpInfo info{};
    const fs::path meta_path = path.string() + ".meta";
    if (std::ifstream meta(meta_path, std::ios::binary); meta) {
        meta.read(reinterpret_cast<char*>(&info), sizeof(info));
        if (!meta || info.magic != ProgramDumpInfo::Magic || info.version != ProgramDumpInfo::Version) {
            outcome.phase = "meta";
            outcome.message = outcome.key = "unreadable .meta sidecar";
            return outcome;
        }
    } else {
        // Dumps from before the sidecar existed: compute programs can still be compiled with
        // defaults, graphics programs need their header and stage.
        const bool compute = path.filename().string().find("-cs-") != std::string::npos;
        if (!compute) {
            outcome.skipped_spirv = true;
            return outcome;
        }
        info.stage = static_cast<u32>(Shader::Stage::Compute);
        info.workgroup_size = {64, 1, 1};
        info.shared_memory_size = 48 * 1024;
    }

    try {
        Shader::ObjectPool<Shader::IR::Inst> inst_pool;
        Shader::ObjectPool<Shader::IR::Block> block_pool;
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        DumpEnvironment env(info, lowered, native);
        const auto host = HostInfo();
        outcome.phase = "translate";
        Shader::Maxwell::Flow::CFG cfg(env, flow_pool, env.StartAddress());
        auto program = Shader::Maxwell::TranslateProgram(inst_pool, block_pool, env, cfg, host);
        if (verbose) {
            DescribeProgram(info, program, native);
        }
        if (dump_ir) {
            fmt::print("{}\n", Shader::IR::DumpProgram(program));
        }
        outcome.phase = "spirv";
        Shader::RuntimeInfo runtime_info{};
        // No previous stage is known offline: assume every varying is written so the emitted
        // module keeps its inputs instead of folding them to constants.
        runtime_info.previous_stage_stores.mask.set();
        if (program.stage == Shader::Stage::Geometry) {
            runtime_info.input_topology = Shader::InputTopology::Triangles;
        }
        Shader::Backend::Bindings bindings;
        const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(HostProfile(), runtime_info, program, bindings);
        if (spirv.empty()) {
            throw Shader::RuntimeError("empty SPIR-V module");
        }
        if (!spirv_output.empty()) {
            std::ofstream out(spirv_output, std::ios::binary);
            out.write(reinterpret_cast<const char*>(spirv.data()), spirv.size() * sizeof(u32));
        }
        outcome.phase.clear();
    } catch (const Shader::Exception& error) {
        outcome.message = outcome.key = StripInstructionWords(error.what());
    } catch (const std::exception& error) {
        outcome.message = outcome.key = fmt::format("{} ({})", error.what(), typeid(error).name());
    }
    return outcome;
}

std::set<std::string> ReadBaseline(const fs::path& path) {
    std::set<std::string> names;
    std::ifstream file(path);
    for (std::string line; std::getline(file, line);) {
        if (const auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (!line.empty()) names.insert(fs::path(line).filename().string());
    }
    return names;
}

} // Anonymous namespace

int main(int argc, char** argv) {
    bool lowering_only = false, quiet = false;
    fs::path baseline_path;
    std::vector<fs::path> inputs;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "-l") lowering_only = true;
        else if (arg == "-q") quiet = true;
        else if (arg == "-v") verbose = true;
        else if (arg == "-d") dump_ir = verbose = true;
        else if (arg == "-b" && i + 1 < argc) baseline_path = argv[++i];
        else if (arg == "-s" && i + 1 < argc) spirv_output = argv[++i];
        else if (arg == "-t" && i + 1 < argc) {
            const std::string spec{argv[++i]};
            const auto parse_slot = [](std::string_view text) {
                const size_t colon = text.find(':');
                return Shader::Environment::DrawTable::Slot{
                    static_cast<u32>(std::stoul(std::string{text.substr(0, colon)})),
                    static_cast<u32>(std::stoul(std::string{text.substr(colon + 1)}, nullptr, 16))};
            };
            const size_t at = spec.find('@');
            std::string_view slots{spec};
            slots = slots.substr(0, at);
            while (!slots.empty() && draw_table_spec.num_slots < draw_table_spec.slots.size()) {
                const size_t comma = slots.find(',');
                draw_table_spec.slots[draw_table_spec.num_slots++] = parse_slot(slots.substr(0, comma));
                slots = comma == std::string_view::npos ? std::string_view{} : slots.substr(comma + 1);
            }
            std::string_view table{std::string_view{spec}.substr(at + 1)};
            if (const size_t hash = table.find('#'); hash != std::string_view::npos) {
                // "#<n>": the patched stage reads the draw index from generic attribute n.
                draw_table_spec.draw_id_attribute = static_cast<u32>(std::stoul(std::string{table.substr(hash + 1)}));
                table = table.substr(0, hash);
            }
            draw_table_spec.table_slot = parse_slot(table);
            draw_table_spec.stride = 16;
        }
        else if (arg == "-f" && i + 1 < argc) {
            // Vertex program of a batched pipeline: forward gl_DrawID on generic attribute n.
            draw_table_spec.draw_id_attribute = static_cast<u32>(std::stoul(argv[++i]));
            draw_table_spec.forward_draw_id = true;
        }
        else inputs.emplace_back(arg);
    }
    if (inputs.empty()) {
        std::fputs("usage: nx2-shader-check [-l] [-q] [-v] [-d] [-s out.spv] [-b baseline] <file.bin | directory>...\n", stderr);
        return 2;
    }
    std::vector<fs::path> files;
    for (const auto& input : inputs) {
        if (fs::is_directory(input)) {
            for (const auto& entry : fs::recursive_directory_iterator(input)) {
                if (entry.is_regular_file() && entry.path().extension() == ".bin") files.push_back(entry.path());
            }
        } else {
            files.push_back(input);
        }
    }
    std::sort(files.begin(), files.end());

    const auto baseline = baseline_path.empty() ? std::set<std::string>{} : ReadBaseline(baseline_path);
    std::map<std::string, std::vector<std::string>> failures_by_reason;
    std::vector<std::string> new_failures, fixed, failed_names;
    size_t passed = 0, skipped = 0;
    for (const auto& file : files) {
        if (verbose) fmt::print("{}\n", file.string());
        const auto outcome = Check(file, lowering_only);
        const auto name = file.filename().string();
        if (outcome.phase.empty()) {
            if (outcome.skipped_spirv) ++skipped; else ++passed;
            if (baseline.contains(name)) fixed.push_back(name);
            continue;
        }
        failed_names.push_back(name);
        failures_by_reason[outcome.phase + ": " + outcome.key].push_back(name);
        if (!baseline.contains(name)) new_failures.push_back(name);
        if (!quiet) fmt::print("{}: {}: {}\n", file.string(), outcome.phase, outcome.message);
    }

    fmt::print("\n{} programs: {} passed, {} failed, {} lowered only (no .meta)\n", files.size(), passed,
               failed_names.size(), skipped);
    std::vector<std::pair<size_t, std::string>> ranked;
    for (const auto& [reason, names] : failures_by_reason) ranked.emplace_back(names.size(), reason);
    std::sort(ranked.rbegin(), ranked.rend());
    for (const auto& [count, reason] : ranked) fmt::print("  {:5}  {}\n", count, reason);
    if (!baseline_path.empty()) {
        fmt::print("baseline {}: {} new failures, {} fixed\n", baseline_path.string(), new_failures.size(), fixed.size());
        for (const auto& name : new_failures) fmt::print("  new:   {}\n", name);
        for (const auto& name : fixed) fmt::print("  fixed: {}\n", name);
        return new_failures.empty() ? 0 : 1;
    }
    return failed_names.empty() ? 0 : 1;
}
