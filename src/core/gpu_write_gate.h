// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
//
// eglantine test 3 - shared CPU-write gate (testability seam).
// Extracted VERBATIM from Memory::Impl::HandleRasterizerWrite (src/core/memory.cpp),
// sequencing preserved exactly:
//   - the seed (last_address = subaddress) happens BEFORE collect, as the original
//   - a FALSE on_cpu_write returns before seeding and before collect
//   - collect is unconditional once the page is seeded (the skip path)
// The DiagTrace emissions are additive evidence capture, isolated from the
// gated logic: Emit is noexcept with all observer work inside one try and the
// catch limited to a noexcept counter increment, so a fault can never skip the
// seed or collect below (faults are counted in observer_errors). All of this
// is disabled by default (EDEN_NX2_TRACE=1 enables). No allocation, retry, or
// write-handling behavior change to the gated logic itself. Tests call this same function with the
// real BufferCache/GPUDirtyMemoryManager components behind the callables.

#pragma once

#include "common/common_types.h"
#include "common/diag_trace.h"

namespace Core {

inline void HandleRasterizerWriteGate(PAddr& last_address, PAddr subaddress, DAddr address,
                                      size_t size, u32 core, auto on_cpu_write, auto collect) {
    if (last_address != subaddress) {
        // Short circuit the need to check for address/size
        auto const do_collection = (address != 0 && size != 0) && on_cpu_write(address, size);
        if (do_collection) {
            last_address = subaddress;
            Common::DiagTrace::Emit(Common::DiagTrace::Ev::HrwSeed, Common::DiagTrace::CurrentFrame(),
                                    core, subaddress, address, size);
        } else {
            return;
        }
    } else {
        Common::DiagTrace::Emit(Common::DiagTrace::Ev::HrwSkip, Common::DiagTrace::CurrentFrame(),
                                core, subaddress, address, size);
    }
    Common::DiagTrace::Emit(Common::DiagTrace::Ev::HrwCollect, Common::DiagTrace::CurrentFrame(),
                                core, address, size, 0);
    collect(address, size);
}

} // namespace Core
