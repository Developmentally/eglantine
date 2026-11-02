// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <optional>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
bool IsStorageAccess(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadStorageU8:
    case IR::Opcode::LoadStorageS8:
    case IR::Opcode::LoadStorageU16:
    case IR::Opcode::LoadStorageS16:
    case IR::Opcode::LoadStorage32:
    case IR::Opcode::LoadStorage64:
    case IR::Opcode::LoadStorage128:
    case IR::Opcode::WriteStorageU8:
    case IR::Opcode::WriteStorageS8:
    case IR::Opcode::WriteStorageU16:
    case IR::Opcode::WriteStorageS16:
    case IR::Opcode::WriteStorage32:
    case IR::Opcode::WriteStorage64:
    case IR::Opcode::WriteStorage128:
    case IR::Opcode::StorageAtomicIAdd32:
    case IR::Opcode::StorageAtomicSMin32:
    case IR::Opcode::StorageAtomicUMin32:
    case IR::Opcode::StorageAtomicSMax32:
    case IR::Opcode::StorageAtomicUMax32:
    case IR::Opcode::StorageAtomicInc32:
    case IR::Opcode::StorageAtomicDec32:
    case IR::Opcode::StorageAtomicAnd32:
    case IR::Opcode::StorageAtomicOr32:
    case IR::Opcode::StorageAtomicXor32:
    case IR::Opcode::StorageAtomicExchange32:
    case IR::Opcode::StorageAtomicIAdd64:
    case IR::Opcode::StorageAtomicSMin64:
    case IR::Opcode::StorageAtomicUMin64:
    case IR::Opcode::StorageAtomicSMax64:
    case IR::Opcode::StorageAtomicUMax64:
    case IR::Opcode::StorageAtomicAnd64:
    case IR::Opcode::StorageAtomicOr64:
    case IR::Opcode::StorageAtomicXor64:
    case IR::Opcode::StorageAtomicExchange64:
    case IR::Opcode::StorageAtomicIAdd32x2:
    case IR::Opcode::StorageAtomicSMin32x2:
    case IR::Opcode::StorageAtomicUMin32x2:
    case IR::Opcode::StorageAtomicSMax32x2:
    case IR::Opcode::StorageAtomicUMax32x2:
    case IR::Opcode::StorageAtomicAnd32x2:
    case IR::Opcode::StorageAtomicOr32x2:
    case IR::Opcode::StorageAtomicXor32x2:
    case IR::Opcode::StorageAtomicExchange32x2:
    case IR::Opcode::StorageAtomicAddF32:
    case IR::Opcode::StorageAtomicAddF16x2:
    case IR::Opcode::StorageAtomicAddF32x2:
    case IR::Opcode::StorageAtomicMinF16x2:
    case IR::Opcode::StorageAtomicMinF32x2:
    case IR::Opcode::StorageAtomicMaxF16x2:
    case IR::Opcode::StorageAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

IR::Attribute DrawIdAttribute(u32 generic) {
    return static_cast<IR::Attribute>(static_cast<u32>(IR::Attribute::Generic0X) + generic * 4);
}
} // Anonymous namespace

void DrawTablePass(Environment& env, IR::Program& program) {
    const auto& table{env.GetDrawTable()};
    if (!table.Enabled()) {
        return;
    }
    if (table.forward_draw_id) {
        // Vertex program of a batched pipeline: gl_DrawID goes out as a varying for the patched
        // stage, as an integer-valued float (exact below 2^24) rather than its bit pattern, which
        // would be a denormal. Stored at the entry; outputs have no ordering.
        IR::Block* const entry{program.blocks.front()};
        IR::IREmitter ir{*entry, entry->begin()};
        const IR::U32 draw_id{ir.GetAttributeU32(IR::Attribute::DrawID)};
        ir.SetAttribute(DrawIdAttribute(table.draw_id_attribute),
                        IR::F32{ir.ConvertUToF(32, 32, draw_id)}, ir.Imm32(0));
    }
    if (table.num_slots == 0) {
        return;
    }
    auto& descriptors{program.info.storage_buffers_descriptors};
    // Storage buffer binding index -> table column, for the bindings whose pointer sits at one
    // of the patched slots. Bindings are numbered in descriptor order (one per descriptor).
    std::array<std::optional<u32>, 64> column_of_binding{};
    bool any{false};
    for (u32 binding = 0; binding < descriptors.size() && binding < column_of_binding.size(); ++binding) {
        const auto& desc{descriptors[binding]};
        for (u32 column = 0; column < table.num_slots; ++column) {
            const auto& slot{table.slots[column]};
            if (desc.cbuf_index == slot.cbuf_index && desc.cbuf_offset == slot.cbuf_offset) {
                column_of_binding[binding] = column;
                any = true;
            }
        }
    }
    if (!any) {
        return;
    }
    // The table itself is one more storage buffer, read-only, found through its own pointer.
    const u32 table_binding{static_cast<u32>(descriptors.size())};
    descriptors.push_back({
        .cbuf_index = table.table_slot.cbuf_index,
        .cbuf_offset = table.table_slot.cbuf_offset,
        .count = 1,
        .is_written = false,
    });
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsStorageAccess(inst.GetOpcode())) {
                continue;
            }
            const IR::Value binding_value{inst.Arg(0)};
            if (!binding_value.IsImmediate()) {
                continue;
            }
            const u32 binding{binding_value.U32()};
            if (binding >= column_of_binding.size() || !column_of_binding[binding]) {
                continue;
            }
            const u32 column{*column_of_binding[binding]};
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const IR::U32 draw_id{
                table.draw_id_attribute == ~0U
                    ? ir.GetAttributeU32(IR::Attribute::DrawID)
                    : IR::U32{ir.ConvertFToU(32, ir.GetAttribute(DrawIdAttribute(table.draw_id_attribute),
                                                                ir.Imm32(0)))}};
            const IR::U32 row{ir.IMul(draw_id, ir.Imm32(table.stride))};
            const IR::U32 delta_offset{ir.IAdd(row, ir.Imm32(column * 4))};
            const IR::Value delta{&*block->PrependNewInst(
                IR::Block::InstructionList::s_iterator_to(inst), IR::Opcode::LoadStorage32,
                {IR::Value{table_binding}, IR::Value{delta_offset}})};
            const IR::U32 offset{ir.IAdd(IR::U32{inst.Arg(1)}, IR::U32{delta})};
            inst.SetArg(1, offset);
        }
    }
}

} // namespace Shader::Optimization
