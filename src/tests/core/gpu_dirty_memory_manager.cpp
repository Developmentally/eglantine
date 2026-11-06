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
// Criterion 2 chain fixture (eglantine test 3): the second same-page CPU write
// must reach the consumer after revalidation through the REAL deferred path.
//
// REAL components exercised here:
//   - Core::HandleRasterizerWriteGate (src/core/gpu_write_gate.h, the verbatim
//     production shortcut seam from Memory::Impl::HandleRasterizerWrite),
//   - Core::GPUDirtyMemoryManager Collect/Gather (the production deferred
//     record + drain),
//   - VideoCommon::BufferCache<TestParams>::WriteMemory (the production
//     consumer entry behind RasterizerVulkan::OnCPUWrite / OnCacheInvalidation),
//   - VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager> (the
//     production tracker: WordManagers + UpdatePagesCachedCount notification
//     path against a REAL manager, no process registered, unmapped pages read
//     as zeros per SparseLargeVector semantics).
//
// PARTIAL BOUNDARY - criterion 2 is NOT claimed passed from this fixture:
//   - BufferCache's own memory_tracker member is private, so a standalone
//     MemoryTrackerBase instance mirrors the same marks for range assertions;
//     cache-side state is asserted through the public IsRegionCpuModified.
//   - The real SynchronizeBuffer -> UploadMemory -> MappedUploadMemory data
//     movement is NOT exercised: DeviceMemoryManager::Map requires a
//     registered Core::Memory process (registered_processes[asid]), which
//     needs a Core::System. The fixture performs the copy an upload would do
//     over the exact ranges the real tracker yields.
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

TEST_CASE("Criterion2 chain: second same-page write reaches consumer after revalidation",
          "[gpu_dirty]") {
    // Region under test: one 4 KiB guest page (one gate subaddress) that is also
    // inside one GPUDirtyMemoryManager 2048-byte page... use a 4 KiB window.
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

    PAddr gate_last_address = ~PAddr(0); // GPUDirtyState.last_address starts "no page"
    unsigned on_cpu_write_calls = 0;
    unsigned collect_calls = 0;

    const auto on_cpu_write = [&](DAddr addr, u64 size) {
        ++on_cpu_write_calls;
        // What RasterizerVulkan::OnCPUWrite -> buffer_cache.WriteMemory does.
        cache.WriteMemory(addr, size);
        mirror.MarkRegionAsCpuModified(addr, size);
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
    const auto drain = [&] {
        std::function<void(PAddr, size_t)> cb = [&](PAddr addr, size_t size) {
            // What GPU::InvalidateGPUCache -> OnCacheInvalidation ->
            // buffer_cache.WriteMemory does.
            cache.WriteMemory(addr, static_cast<u64>(size));
            mirror.MarkRegionAsCpuModified(addr, static_cast<u64>(size));
        };
        manager.Gather(cb);
    };

    // Frame 1: first CPU write hits the full path (gate seeds, OnCPUWrite runs,
    // and the deferred record ALSO collects - verbatim production sequencing).
    std::memset(shadow.data(), 0xAA, 64); // value1 at [kBase, kBase+64)
    drive_gate(kBase, 64);
    REQUIRE(on_cpu_write_calls == 1);
    REQUIRE(collect_calls == 1);
    drain();
    {
        const auto ranges = SyncUploads(mirror, kBase, kWindow, shadow, buffer);
        REQUIRE(CoversRange(ranges, kBase, 64));
        REQUIRE(std::all_of(buffer.host.begin(), buffer.host.begin() + 64,
                            [](u8 b) { return b == 0xAA; }));
        // First sync cleared the CPU-modified tracking (real ForEachUploadRange
        // unmarks): the next write must be registered from a clean slate.
        REQUIRE(!mirror.IsRegionCpuModified(kBase, kWindow));
        // The cache's own tracker still shows the mark (fixture does not run
        // the real upload that would consume it - boundary note).
        REQUIRE(cache.IsRegionCpuModified(kBase, 64));
    }

    // Frame 2: second write to the SAME page. The gate shortcut skips
    // OnCPUWrite; the deferred record is the only path to the consumer.
    std::memset(shadow.data() + 64, 0xBB, 64); // value2 at [kBase+64, kBase+128)
    drive_gate(kBase + 64, 64);
    REQUIRE(on_cpu_write_calls == 1); // unchanged: shortcut skipped OnCPUWrite
    REQUIRE(collect_calls == 2);
    drain();
    {
        // The drain callback delivered the second write to the REAL consumer
        // entry: the cache's own tracker now marks the second range CPU-modified.
        REQUIRE(cache.IsRegionCpuModified(kBase + 64, 64));
        const auto ranges = SyncUploads(mirror, kBase, kWindow, shadow, buffer);
        // The tracker registered the second write from a clean slate: at least
        // one range covers it, and nothing was left merged with frame 1 state.
        REQUIRE(!ranges.empty());
        REQUIRE(CoversRange(ranges, kBase + 64, 64));
        REQUIRE(std::all_of(buffer.host.begin() + 64, buffer.host.begin() + 128,
                            [](u8 b) { return b == 0xBB; }));
        // Value1 from frame 1 is still intact in the buffer.
        REQUIRE(std::all_of(buffer.host.begin(), buffer.host.begin() + 64,
                            [](u8 b) { return b == 0xAA; }));
        REQUIRE(!mirror.IsRegionCpuModified(kBase, kWindow));
    }
}

TEST_CASE("Criterion2 chain: write to a NEW page reopens the full path", "[gpu_dirty]") {
    constexpr PAddr kBase = 0x0000'0400'0000;
    constexpr PAddr kOtherPage = kBase + Core::Memory::YUZU_PAGESIZE; // different gate subaddress

    Core::DeviceMemory backing;
    Tegra::MaxwellDeviceMemoryManager gmmu{backing};
    ChainTestRuntime runtime;
    VideoCommon::BufferCache<ChainTestParams> cache{gmmu, runtime};
    Core::GPUDirtyMemoryManager manager;

    PAddr gate_last_address = ~PAddr(0);
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
    drive_gate(kBase, 64); // same page: skip
    drive_gate(kOtherPage, 64); // new page: full path again
    REQUIRE(on_cpu_write_calls == 2);
    REQUIRE(collect_calls == 3);

    std::function<void(PAddr, size_t)> cb = [](PAddr, size_t) {};
    manager.Gather(cb);
}

TEST_CASE("Criterion2 chain: rejected OnCPUWrite neither seeds nor collects", "[gpu_dirty]") {
    constexpr PAddr kBase = 0x0000'0400'0000;
    Core::GPUDirtyMemoryManager manager;
    PAddr gate_last_address = ~PAddr(0);
    unsigned collect_calls = 0;
    const auto reject = [](DAddr, u64) { return false; };
    const auto collect = [&](DAddr addr, u64 size) {
        ++collect_calls;
        manager.Collect(addr, size);
    };

    Core::HandleRasterizerWriteGate(gate_last_address, kBase >> Core::Memory::YUZU_PAGEBITS,
                                    kBase, 64, 0, reject, collect);
    // FALSE return: no seed, no collect (verbatim production order).
    REQUIRE(collect_calls == 0);
    REQUIRE(GatherAll(manager).empty());

    // The page was not seeded, so the next write is treated as a FIRST write:
    // the full path runs again.
    const auto accept = [](DAddr, u64) { return true; };
    Core::HandleRasterizerWriteGate(gate_last_address, kBase >> Core::Memory::YUZU_PAGEBITS,
                                    kBase, 64, 0, accept, collect);
    REQUIRE(collect_calls == 1);
    REQUIRE(HasRange(GatherAll(manager), kBase, 64));
}
