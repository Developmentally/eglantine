// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Regression tests for the deferred CPU-write collection path (eglantine test 3
// diagnostic, acceptance criterion 2). These run the REAL
// Core::GPUDirtyMemoryManager Collect/Gather state machine. They do NOT model
// the memory.cpp shortcut gate or BufferCache::OnCPUWrite; that interplay is
// covered by runtime trace lines in the diagnostic build, not by this test.

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "core/gpu_dirty_memory_manager.h"

namespace {

using Range = std::pair<PAddr, size_t>;

std::vector<Range> GatherAll(Core::GPUDirtyMemoryManager& manager) {
    std::vector<Range> out;
    std::function<void(PAddr, size_t)> cb = [&out](PAddr address, size_t size) {
        out.emplace_back(address, size);
    };
    manager.Gather(cb);
    return out;
}

bool HasRange(const std::vector<Range>& ranges, PAddr address, size_t size) {
    return std::find(ranges.begin(), ranges.end(), Range{address, size}) != ranges.end();
}

// The manager works on 2048-byte units of 64-byte granules (DEVICE_PAGEBITS-1,
// align_bits=6). Addresses below are aligned accordingly.
constexpr PAddr kPageA = 0x0000'0400'0000; // manager page boundary (multiple of 2048)
constexpr PAddr kPageB = 0x0000'0400'0800; // next manager page

} // namespace

TEST_CASE("GPUDirtyMemoryManager: second same-page write is delivered", "[gpu_dirty]") {
    Core::GPUDirtyMemoryManager manager;
    // First write to the page (in the live path this one also ran OnCPUWrite).
    manager.Collect(kPageA, 64);
    // Second write to the SAME page (in the live path the shortcut skips
    // OnCPUWrite here; the deferred record is the only trace of this write).
    manager.Collect(kPageA + 64, 64);
    const auto ranges = GatherAll(manager);
    // Gather merges consecutive 64-byte granule bits into ONE contiguous range:
    // the two adjacent collects yield a single 128-byte range, not two 64-byte ones.
    REQUIRE(ranges.size() == 1);
    REQUIRE(HasRange(ranges, kPageA, 128));
}

TEST_CASE("GPUDirtyMemoryManager: re-collect after drain is delivered again", "[gpu_dirty]") {
    Core::GPUDirtyMemoryManager manager;
    manager.Collect(kPageA, 64);
    const auto first = GatherAll(manager);
    REQUIRE(first.size() == 1);
    // After a drain, the same range written again must not be swallowed by
    // leftover state (no sticky mask, no seeded transform).
    manager.Collect(kPageA, 64);
    const auto second = GatherAll(manager);
    REQUIRE(second.size() == 1);
    REQUIRE(HasRange(second, kPageA, 64));
}

TEST_CASE("GPUDirtyMemoryManager: writes to two pages are both delivered", "[gpu_dirty]") {
    Core::GPUDirtyMemoryManager manager;
    manager.Collect(kPageA, 128);
    manager.Collect(kPageB, 64);
    const auto ranges = GatherAll(manager);
    REQUIRE(ranges.size() == 2);
    REQUIRE(HasRange(ranges, kPageA, 128));
    REQUIRE(HasRange(ranges, kPageB, 64));
}

TEST_CASE("GPUDirtyMemoryManager: duplicate collect is not double-reported", "[gpu_dirty]") {
    Core::GPUDirtyMemoryManager manager;
    manager.Collect(kPageA, 64);
    manager.Collect(kPageA, 64); // identical range: mask subset, early return
    const auto ranges = GatherAll(manager);
    REQUIRE(ranges.size() == 1);
    REQUIRE(HasRange(ranges, kPageA, 64));
}

TEST_CASE("GPUDirtyMemoryManager: empty gather reports nothing", "[gpu_dirty]") {
    Core::GPUDirtyMemoryManager manager;
    const auto ranges = GatherAll(manager);
    REQUIRE(ranges.empty());
}
