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

// ============================================================================
// Criterion 2 chain fixture (eglantine test 3): does the second same-page CPU
// write - the one the rasterizer-write gate's shortcut skips - still reach the
// buffer-cache consumer through the REAL deferred Collect/Gather path?
//
// REAL components exercised:
//   - Core::HandleRasterizerWriteGate (src/core/gpu_write_gate.h): the verbatim
//     production shortcut seam from Memory::Impl::HandleRasterizerWrite.
//   - Core::GPUDirtyMemoryManager Collect/Gather: the production deferred
//     record + drain.
//   - VideoCommon::BufferCache<TestParams>::WriteMemory: the production mark
//     applied by the REAL drain consumer (RasterizerVulkan::OnCacheInvalidation
//     calls exactly this under the buffer-cache mutex).
//   - VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>: the
//     production tracker class (real WordManagers), over a REAL manager
//     instance with no process registered (unmapped pages read as zeros via
//     SparseLargeVector; UpdatePagesCachedCount no-ops).
//
// FAITHFUL MODELING NOTES (verified against source at this commit):
//   - The gate's on_cpu_write callable models RasterizerVulkan::OnCPUWrite
//     (vk_rasterizer.cpp) -> BufferCache<P>::OnCPUWrite (buffer_cache.h):
//       * unregistered range            -> false (no seed, no collect)
//       * registered + GPU-modified      -> TRUE, and crucially WITHOUT calling
//         WriteMemory: the CPU write is DEFERRED ENTIRELY; its only record is
//         the gate's Collect. The fixture models this precondition by marking
//         the region GPU-modified in the mirror tracker.
//       * registered + not GPU-modified  -> WriteMemory immediately, false.
//   - The gate seeds per YUZU (4 KiB) page; production rasterizer_write_areas
//     is value-initialized (memory.cpp:778), so last_address starts at 0 and a
//     first write to guest page 0 takes the skip path - tested below.
//   - The tracker's CPU-modified granularity is one DEVICE page (4 KiB)
//     (word_manager.h: BYTES_PER_PAGE = Core::DEVICE_PAGESIZE), so within one
//     gate page the cache's IsRegionCpuModified CANNOT distinguish a second
//     write from the first. The second-write assertions therefore rest on the
//     mirror tracker's clean-baseline re-registration (the real unmark in
//     ForEachUploadRange), not on a sub-page cache query.
//
// PARTIAL BOUNDARY - criterion 2 is NOT claimed passed from this fixture:
//   - The cache's OWN tracker is private; a standalone MemoryTrackerBase
//     mirrors the same marks for range assertions. The cache stays CPU-marked
//     after the fixture's syncs because the real consume (SynchronizeBuffer)
//     is not run - asserted and labeled, not hidden.
//   - The real SynchronizeBuffer -> UploadMemory data movement is NOT
//     exercised: DeviceMemoryManager::Map requires a registered Core::Memory
//     process (registered_processes[asid]); RegisterProcess is public
//     (device_memory_manager.h:121) but Core::Memory::Memory's ctor requires
//     Core::System& (core/memory.h:64) and this harness has no System fixture.
//     The fixture copies over exactly the ranges the REAL tracker yields.
//   - SynchronizeBuffer's gpu_modified_ranges exclusion is not exercised.

#include <array>
#include <cstring>

#include "core/device_memory.h"
#include "core/gpu_write_gate.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/host1x/gpu_device_memory_manager.h"

namespace {

class ChainTestRuntime {
public:
    bool CanReportMemoryUsage() const {
        return false;
    }
    // Only reachable in dead code after the CanReportMemoryUsage early return;
    // still compiled, so it must exist.
    u64 GetDeviceLocalMemory() const {
        return 8ULL << 30;
    }
};

class ChainTestBuffer : public VideoCommon::BufferBase {
public:
    ChainTestBuffer(ChainTestRuntime&, VideoCommon::NullBufferParams params) : BufferBase(params) {}
    ChainTestBuffer(ChainTestRuntime&, VAddr cpu_addr_, u64 size_bytes_)
        : BufferBase(cpu_addr_, size_bytes_) {
        host.assign(size_bytes_, 0);
    }

    // Host-observable backing; stands in for the mapped GPU buffer the real
    // upload path would copy into.
    std::vector<u8> host;
};

struct ChainTestParams {
    using Runtime = ChainTestRuntime;
    using Buffer = ChainTestBuffer;
    using Async_Buffer = int; // unused on the exercised path; signatures only
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using ChainTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

// Copies the ranges the REAL tracker yields from the guest shadow into the
// buffer backing, standing in for MappedUploadMemory (see boundary note).
std::vector<Range> SyncUploads(ChainTracker& tracker, VAddr base, u64 size,
                               const std::array<u8, 4096>& shadow, ChainTestBuffer& buffer) {
    std::vector<Range> ranges;
    tracker.ForEachUploadRange(base, size, [&](u64 addr, u64 range_size) {
        ranges.emplace_back(addr, range_size);
        std::memcpy(buffer.host.data() + (addr - base), shadow.data() + (addr - base), range_size);
    });
    return ranges;
}

bool CoversRange(const std::vector<Range>& ranges, u64 addr, u64 size) {
    for (const auto& [start, len] : ranges) {
        if (start <= addr && start + len >= addr + size) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("Criterion2 chain: deferred second same-page write re-registers after a clean "
          "baseline",
          "[gpu_dirty]") {
    // One gate (4 KiB) page; GPUDirtyMemoryManager pages (2048 B) nest inside.
    constexpr PAddr kBase = 0x0000'0400'0000; // 4 KiB and 2048-byte aligned
    constexpr u64 kWindow = 4096;
    static_assert(kBase % Core::Memory::YUZU_PAGESIZE == 0);

    Core::DeviceMemory backing;
    Tegra::MaxwellDeviceMemoryManager gmmu{backing};
    ChainTestRuntime runtime;
    VideoCommon::BufferCache<ChainTestParams> cache{gmmu, runtime};
    ChainTracker mirror{gmmu};
    Core::GPUDirtyMemoryManager manager;

    std::array<u8, 4096> shadow{};
    ChainTestBuffer buffer{runtime, kBase, kWindow};

    // Production precondition for OnCPUWrite's TRUE (fully deferred) branch:
    // the region is registered and GPU-modified.
    mirror.MarkRegionAsGpuModified(kBase, kWindow);

    PAddr gate_last_address = 0; // matches value-initialized rasterizer_write_areas
    unsigned on_cpu_write_calls = 0;
    unsigned collect_calls = 0;
    unsigned drain_writes = 0;

    // Faithful model of RasterizerVulkan::OnCPUWrite's true branch
    // (registered + GPU-modified): returns true and does NOT mark the buffer
    // cache; the write is deferred entirely.
    const auto on_cpu_write = [&](DAddr, u64) {
        ++on_cpu_write_calls;
        return true;
    };
    const auto collect = [&](DAddr addr, u64 size) {
        ++collect_calls;
        manager.Collect(addr, size);
    };
    const auto drive_gate = [&](PAddr paddr, u64 size) {
        Core::HandleRasterizerWriteGate(gate_last_address, paddr >> Core::Memory::YUZU_PAGEBITS,
                                        paddr, size, 0, on_cpu_write, collect);
    };
    // The real drain consumer (GPU::InvalidateGPUCache ->
    // RasterizerVulkan::OnCacheInvalidation -> buffer_cache.WriteMemory;
    // texture/pipeline legs are out of scope here).
    const auto drain = [&] {
        std::function<void(PAddr, size_t)> cb = [&](PAddr addr, size_t size) {
            ++drain_writes;
            cache.WriteMemory(addr, static_cast<u64>(size));
            mirror.MarkRegionAsCpuModified(addr, static_cast<u64>(size));
        };
        manager.Gather(cb);
    };

    // Frame 1: first CPU write runs the full gate path. OnCPUWrite accepts
    // (true) WITHOUT marking; verbatim production sequencing still records the
    // write in the deferred manager.
    std::memset(shadow.data(), 0xAA, 64); // value1 at [kBase, kBase+64)
    drive_gate(kBase, 64);
    REQUIRE(on_cpu_write_calls == 1);
    REQUIRE(collect_calls == 1);
    drain();
    REQUIRE(drain_writes == 1);
    {
        const auto ranges = SyncUploads(mirror, kBase, kWindow, shadow, buffer);
        REQUIRE(CoversRange(ranges, kBase, 64));
        REQUIRE(std::all_of(buffer.host.begin(), buffer.host.begin() + 64,
                            [](u8 b) { return b == 0xAA; }));
        // The real ForEachUploadRange unmarked what it yielded: clean baseline.
        REQUIRE(!mirror.IsRegionCpuModified(kBase, kWindow));
        // The cache's own tracker remains marked: its real consume
        // (SynchronizeBuffer) is not run in this fixture. Labeled boundary, and
        // per the granularity note this mark cannot distinguish write 2 anyway.
        REQUIRE(cache.IsRegionCpuModified(kBase, 64));
    }

    // Frame 2: second CPU write, SAME gate page AND same range (value1 is
    // replaced by value2). The gate shortcut skips OnCPUWrite entirely; the
    // deferred record is the ONLY path to the consumer.
    std::memset(shadow.data(), 0xBB, 64); // value2 replaces value1 at [kBase, kBase+64)
    drive_gate(kBase, 64);
    REQUIRE(on_cpu_write_calls == 1); // unchanged: the shortcut skipped OnCPUWrite
    REQUIRE(collect_calls == 2);
    drain();
    REQUIRE(drain_writes == 2);
    {
        const auto ranges = SyncUploads(mirror, kBase, kWindow, shadow, buffer);
        // From a genuinely clean baseline (the real unmark in sync 1), the
        // second write re-registered and is yielded again: this is the
        // deferred-delivery evidence.
        REQUIRE(!ranges.empty());
        REQUIRE(CoversRange(ranges, kBase, 64));
        // The buffer now holds value2 at that range...
        REQUIRE(std::all_of(buffer.host.begin(), buffer.host.begin() + 64,
                            [](u8 b) { return b == 0xBB; }));
        // ...and untouched offsets stay zero (control).
        REQUIRE(std::all_of(buffer.host.begin() + 64, buffer.host.end(),
                            [](u8 b) { return b == 0x00; }));
        REQUIRE(!mirror.IsRegionCpuModified(kBase, kWindow));
    }
}

TEST_CASE("Criterion2 chain: write to a NEW gate page reopens the full path", "[gpu_dirty]") {
    constexpr PAddr kBase = 0x0000'0400'0000;
    constexpr PAddr kOtherPage = kBase + Core::Memory::YUZU_PAGESIZE;

    Core::GPUDirtyMemoryManager manager;
    PAddr gate_last_address = 0;
    unsigned on_cpu_write_calls = 0;
    unsigned collect_calls = 0;
    const auto on_cpu_write = [&](DAddr, u64) {
        ++on_cpu_write_calls;
        return true;
    };
    const auto collect = [&](DAddr addr, u64 size) {
        ++collect_calls;
        manager.Collect(addr, size);
    };
    const auto drive_gate = [&](PAddr paddr, u64 size) {
        Core::HandleRasterizerWriteGate(gate_last_address, paddr >> Core::Memory::YUZU_PAGEBITS,
                                        paddr, size, 0, on_cpu_write, collect);
    };

    drive_gate(kBase, 64);
    drive_gate(kBase, 64);       // same page: skip
    drive_gate(kOtherPage, 64);  // new page: full path again
    REQUIRE(on_cpu_write_calls == 2);
    REQUIRE(collect_calls == 3);
    REQUIRE(GatherAll(manager).size() == 2);
}

TEST_CASE("Criterion2 chain: rejected OnCPUWrite neither seeds nor collects, and a "
          "rejection on another page does not unseed",
          "[gpu_dirty]") {
    constexpr PAddr kBase = 0x0000'0400'0000;
    constexpr PAddr kOtherPage = kBase + Core::Memory::YUZU_PAGESIZE;
    Core::GPUDirtyMemoryManager manager;
    PAddr gate_last_address = 0;
    unsigned collect_calls = 0;
    unsigned full_calls = 0;
    bool accept = true;
    const auto on_cpu_write = [&](DAddr, u64) {
        ++full_calls;
        return accept;
    };
    const auto collect = [&](DAddr addr, u64 size) {
        ++collect_calls;
        manager.Collect(addr, size);
    };
    const auto drive_gate = [&](PAddr paddr, u64 size) {
        Core::HandleRasterizerWriteGate(gate_last_address, paddr >> Core::Memory::YUZU_PAGEBITS,
                                        paddr, size, 0, on_cpu_write, collect);
    };

    // First write REJECTED (unregistered range in production): no seed, no
    // collect; the page is not latched.
    accept = false;
    drive_gate(kBase, 64);
    REQUIRE(full_calls == 1);
    REQUIRE(collect_calls == 0);
    REQUIRE(GatherAll(manager).empty());

    // Same page, now accepted: treated as a FIRST write (full path again),
    // because the rejection did not seed it.
    accept = true;
    drive_gate(kBase, 64);
    REQUIRE(full_calls == 2);
    REQUIRE(collect_calls == 1);

    // Same page again: skip path.
    drive_gate(kBase, 64);
    REQUIRE(full_calls == 2);
    REQUIRE(collect_calls == 2);

    // A REJECTED write to ANOTHER page must not unseed the first page:
    // the rejection returns before touching last_address.
    accept = false;
    drive_gate(kOtherPage, 64);
    REQUIRE(full_calls == 3);
    REQUIRE(collect_calls == 2);
    accept = true;
    drive_gate(kBase, 64); // still the seeded page: skip, no full path
    REQUIRE(full_calls == 3);
    REQUIRE(collect_calls == 3);
    REQUIRE(GatherAll(manager).size() == 1);
}

TEST_CASE("Criterion2 chain: first write to guest page 0 takes the skip path "
          "(production initial state)",
          "[gpu_dirty]") {
    // rasterizer_write_areas is value-initialized (memory.cpp:778), so
    // last_address starts at 0 == page 0's subaddress. A first write to guest
    // page 0 therefore looks "already seeded" and skips OnCPUWrite. This test
    // documents that production edge, it does not bless it.
    Core::GPUDirtyMemoryManager manager;
    PAddr gate_last_address = 0;
    unsigned full_calls = 0;
    unsigned collect_calls = 0;
    const auto on_cpu_write = [&](DAddr, u64) {
        ++full_calls;
        return true;
    };
    const auto collect = [&](DAddr addr, u64 size) {
        ++collect_calls;
        manager.Collect(addr, size);
    };

    Core::HandleRasterizerWriteGate(gate_last_address, PAddr{0}, PAddr{0}, 64, 0, on_cpu_write,
                                    collect);
    REQUIRE(full_calls == 0);  // OnCPUWrite skipped for the very first write to page 0
    REQUIRE(collect_calls == 1);
    REQUIRE(HasRange(GatherAll(manager), 0, 64));
}
