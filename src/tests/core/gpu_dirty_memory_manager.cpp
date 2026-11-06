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

TEST_CASE("DiagTrace names every event class", "[diagtrace]") {
    // An enum insert without a matching kClsNames insert silently mislabels
    // every later class; the static_assert keeps the count honest and this
    // keeps every entry named and unique.
    for (u32 i = 0; i < static_cast<u32>(Common::DiagTrace::Ev::Count); ++i) {
        REQUIRE(std::string(Common::DiagTrace::ClsName(i)) != "Unknown");
        for (u32 j = i + 1; j < static_cast<u32>(Common::DiagTrace::Ev::Count); ++j) {
            REQUIRE(std::string(Common::DiagTrace::ClsName(i)) !=
                    std::string(Common::DiagTrace::ClsName(j)));
        }
    }
}

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

class ChainTestBuffer; // fwd for the runtime surface

// Staging-buffer stand-in for the real mapped upload path: MappedUploadMemory
// fills mapped_span via the REAL DeviceMemoryManager::ReadBlockUnsafe, then
// calls runtime.CopyBuffer. Capturing these bytes IS the delivery evidence.
struct ChainStaging {
    std::vector<u8> data;
};
struct ChainUpload {
    ChainStaging* buffer;
    u64 offset;
    std::span<u8> mapped_span;
};

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
    u64 CurrentTick() {
        return ++tick;
    }
    ChainUpload UploadStagingBuffer(u64 total_size_bytes) {
        ++staging_calls;
        staging.data.assign(static_cast<size_t>(total_size_bytes), 0);
        return {&staging, 0, std::span<u8>(staging.data)};
    }
    bool CanReorderUpload(ChainTestBuffer&, std::span<VideoCommon::BufferCopy>) {
        return false;
    }
    bool IsFree(u64) {
        return true;
    }
    void Wait(u64) {}
    // Real memcpy delivery of the staged bytes into the buffer's host backing.
    void CopyBuffer(ChainTestBuffer& dst, ChainStaging* src,
                    std::span<VideoCommon::BufferCopy> copies, bool, bool);
    // Buffer-to-buffer copy (used by SplitBuffer paths); same memcpy model.
    template <typename Copies>
    void CopyBuffer(ChainTestBuffer& dst, ChainTestBuffer& src, const Copies& copies, bool);
    void ClearBuffer(ChainTestBuffer& dst, u64 offset, u64 size, u32 value);

    u64 tick = 0;
    u64 staging_calls = 0;
    ChainStaging staging;
};

class ChainTestBuffer : public VideoCommon::BufferBase {
public:
    ChainTestBuffer(ChainTestRuntime&, VideoCommon::NullBufferParams params) : BufferBase(params) {}
    ChainTestBuffer(ChainTestRuntime&, VAddr cpu_addr_, u64 size_bytes_)
        : BufferBase(cpu_addr_, size_bytes_) {
        host.assign(size_bytes_, 0);
        cpu_addr_cached = cpu_addr_;
    }

    // Host-observable backing; stands in for the mapped GPU buffer the real
    // upload path would copy into.
    std::vector<u8> host;

    ChainTestBuffer(ChainTestRuntime& r, VAddr cpu_addr_, u32 size_, bool)
        : ChainTestBuffer(r, cpu_addr_, static_cast<u64>(size_)) {
        cpu_addr_cached = cpu_addr_;
    }

    void setWriteTick(u64 t) {
        write_tick = t;
    }
    u64 getWriteTick() const {
        return write_tick;
    }
    void MarkUsage(u64, u64) {}
    u64 write_tick = 0;
    VAddr cpu_addr_cached = 0; // a field in the real Buffer, not a method
};

inline void ChainTestRuntime::CopyBuffer(ChainTestBuffer& dst, ChainStaging* src,
                                         std::span<VideoCommon::BufferCopy> copies, bool, bool) {
    for (const auto& c : copies) {
        std::memcpy(dst.host.data() + c.dst_offset, src->data.data() + c.src_offset, c.size);
    }
}

template <typename Copies>
inline void ChainTestRuntime::CopyBuffer(ChainTestBuffer& dst, ChainTestBuffer& src,
                                         const Copies& copies, bool) {
    for (const auto& c : copies) {
        std::memcpy(dst.host.data() + c.dst_offset, src.host.data() + c.src_offset, c.size);
    }
}

inline void ChainTestRuntime::ClearBuffer(ChainTestBuffer& dst, u64 offset, u64 size, u32 value) {
    std::memset(dst.host.data() + offset, static_cast<int>(value), static_cast<size_t>(size));
}

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

// ---------------------------------------------------------------------------
// DiagTrace observer-failure isolation (R1-R6/G1-G4 review items).
// These tests exercise the ACTUAL production header via compile-only hooks
// (EDEN_NX2_TRACE_TEST_HOOKS, defined for the fixture build only). The
// catch-all paths they exercise are always compiled.
// ReadEnvOnce latches per process, so the build script runs three tagged
// batches: [trace-init-fail] (own process, init-fail env), [trace-on]
// (EDEN_NX2_TRACE=1), [trace-off] with [gpu_dirty] (no trace env).
// ---------------------------------------------------------------------------

// T5 (compile time): the isolation surface is noexcept by contract.
static_assert(noexcept(Common::DiagTrace::Emit(Common::DiagTrace::Ev::Config, 0, 0, 0, 0,0,"")));
static_assert(noexcept(Common::DiagTrace::Flush(0, "t")));
static_assert(noexcept(Common::DiagTrace::TickFlush(0)));
static_assert(noexcept(Common::DiagTrace::Enabled()));
static_assert(noexcept(Common::DiagTrace::BeginDraw(0, std::declval<u64&>())));
static_assert(std::is_nothrow_default_constructible_v<Common::DiagTrace::State>);

namespace {
using Common::DiagTrace::Ev;
namespace DT = Common::DiagTrace;

struct GateProbe {
    u64 last = 0; // production zero-init (memory.cpp GPUDirtyState value-init)
    unsigned oncpu = 0;
    unsigned collects = 0;
    auto OnCpu() { return [this](u64, u64) { ++oncpu; return true; }; }
    auto Collect() { return [this](u64, u64) { ++collects; }; }
};
constexpr u64 kTracePage = 0x400000; // nonzero page, well above page 0
} // namespace

TEST_CASE("DiagTrace isolation T1: emit fault before collect is contained (fresh page, seed kept)",
          "[trace-on]") {
    GateProbe g;
    const u64 err0 = DT::S().observer_errors.load();
    DT::TestArmFailNext(1); // next enabled emit (HrwSeed) throws
    Core::HandleRasterizerWriteGate(g.last, kTracePage, kTracePage, 64, 0, g.OnCpu(), g.Collect());
    REQUIRE(g.last == kTracePage); // seed happened despite the throwing emit
    REQUIRE(g.oncpu == 1);
    REQUIRE(g.collects == 1); // collect still ran exactly once
    REQUIRE(DT::S().observer_errors.load() - err0 == 1); // contained + counted
}

TEST_CASE("DiagTrace isolation T2: skip-path emit fault is contained (pre-seeded page)",
          "[trace-on]") {
    GateProbe g;
    g.last = kTracePage; // pre-seeded
    const u64 err0 = DT::S().observer_errors.load();
    DT::TestArmFailNext(1); // HrwSkip throws
    Core::HandleRasterizerWriteGate(g.last, kTracePage, kTracePage, 64, 0, g.OnCpu(), g.Collect());
    REQUIRE(g.oncpu == 0); // skip path never calls OnCPUWrite
    REQUIRE(g.collects == 1); // collect still ran
    REQUIRE(DT::S().observer_errors.load() - err0 == 1);
}

TEST_CASE("DiagTrace isolation T3: collect-report emit fault is contained (pre-seeded page)",
          "[trace-on]") {
    GateProbe g;
    g.last = kTracePage;
    const u64 err0 = DT::S().observer_errors.load();
    DT::TestArmFailNext(1, 1); // let HrwSkip pass; HrwCollect throws (it precedes collect())
    Core::HandleRasterizerWriteGate(g.last, kTracePage, kTracePage, 64, 0, g.OnCpu(), g.Collect());
    REQUIRE(g.collects == 1); // collect still ran exactly once
    REQUIRE(DT::S().observer_errors.load() - err0 == 1);
}

TEST_CASE("DiagTrace isolation T4: both emits faulting still collect exactly once", "[trace-on]") {
    GateProbe g;
    const u64 err0 = DT::S().observer_errors.load();
    DT::TestArmFailNext(2); // HrwSeed and HrwCollect both throw
    Core::HandleRasterizerWriteGate(g.last, kTracePage, kTracePage, 64, 0, g.OnCpu(), g.Collect());
    REQUIRE(g.last == kTracePage);
    REQUIRE(g.collects == 1);
    REQUIRE(DT::S().observer_errors.load() - err0 == 2);
}

TEST_CASE("DiagTrace isolation T8: flush fault is counted and latches the fail-safe",
          "[trace-on]") {
    const u64 err0 = DT::S().observer_errors.load();
    REQUIRE(!DT::S().flush_broken.load());
    DT::TestArmFailNext(1);
    DT::Flush(DT::CurrentFrame(), "t8");
    REQUIRE(DT::S().observer_errors.load() - err0 == 1);
    REQUIRE(DT::S().flush_broken.load()); // fail-safe: no more attempts
    DT::Flush(DT::CurrentFrame(), "t8b"); // early-returns: no throw, no new error
    REQUIRE(DT::S().observer_errors.load() - err0 == 1);
    DT::S().flush_broken.store(false); // leave the process clean for later tests
}

TEST_CASE("DiagTrace isolation T9: post-halt arrivals count late_after_halt only", "[trace-on]") {
    const u64 late0 = DT::S().late_after_halt.load();
    const u64 em0 = DT::S().emitted[static_cast<size_t>(Ev::HrwSkip)].load();
    DT::S().capture_halted.store(true);
    for (int k = 0; k < 3; ++k) {
        DT::Emit(Ev::HrwSkip, DT::CurrentFrame(), 0, kTracePage, kTracePage, 64);
    }
    REQUIRE(DT::S().late_after_halt.load() - late0 == 3);
    REQUIRE(DT::S().emitted[static_cast<size_t>(Ev::HrwSkip)].load() == em0); // unchanged
    DT::S().capture_halted.store(false);
}

TEST_CASE("DiagTrace isolation T10: Observe() contains call-site faults; outside state advances",
          "[trace-on]") {
    const u64 err0 = DT::S().observer_errors.load();
    unsigned adjacent = 0;
    DT::Observe([&] { throw std::bad_alloc{}; });
    ++adjacent; // state OUTSIDE the lambda still advances
    REQUIRE(adjacent == 1);
    REQUIRE(DT::S().observer_errors.load() - err0 == 1);
}

TEST_CASE("DiagTrace isolation T7: init fault degrades to disabled, latches, never retries",
          "[trace-init-fail]") {
    // Ran in its own process with EDEN_NX2_TRACE=1 and
    // EDEN_NX2_TRACE_TEST_FAIL_INIT=1 (build script sets both).
    REQUIRE(!DT::Enabled()); // the init fault forced disabled
    REQUIRE(DT::S().init_runs.load() == 1); // init body ran exactly once
    REQUIRE(DT::S().observer_errors.load() >= 1);
    DT::Emit(Ev::Config, 0, 0, 0, 0, 0, "t7");
    DT::Emit(Ev::Config, 0, 0, 0, 0, 0, "t7");
    REQUIRE(DT::S().init_runs.load() == 1); // no retry storm after the latch
}

TEST_CASE("DiagTrace isolation T6: disabled trace never consults the hook", "[trace-off]") {
    GateProbe g;
    DT::TestArmFailNext(1);
    Core::HandleRasterizerWriteGate(g.last, kTracePage, kTracePage, 64, 0, g.OnCpu(), g.Collect());
    REQUIRE(g.collects == 1); // gate behaves exactly as without the hook
    REQUIRE(DT::S().test_fail_next.load() == 1); // hook armed but never consulted
    DT::S().test_fail_next.store(0);
}

TEST_CASE("GPUDirtyMemoryManager gate T11: size==0 on a fresh nonzero page neither seeds nor "
          "collects", "[trace-off]") {
    GateProbe g;
    Core::HandleRasterizerWriteGate(g.last, kTracePage, kTracePage, 0, 0, g.OnCpu(), g.Collect());
    REQUIRE(g.oncpu == 0); // (address != 0 && size != 0) short-circuits first
    REQUIRE(g.collects == 0);
    REQUIRE(g.last == 0); // unseeded
}

// ---------------------------------------------------------------------------
// Criterion-2 real-path seam (CPU lane v3, editor-steered): the production
// diff is the single friend declaration above; the accessor below reproduces
// DeviceMemoryManager::Map's simple base_dev==0 branch
// (device_memory_manager.inc:206-246) EXACTLY, with the
// process_memory->GetPointerSilent call replaced by a test resolver.
// OUT OF SCOPE (explicitly not exercised): Map itself, the multi-device
// branch, track=true / TrackContinuityImpl, Unmap, and reverse-map readers.
// Mapping-loop equivalence is NOT claimed beyond this branch.
// ---------------------------------------------------------------------------
// Defined in namespace Core to match the friend declaration.
struct Core::DeviceMemoryManagerTestAccess {
    template <typename Traits>
    static void MapForTest(Core::DeviceMemoryManager<Traits>& m, DAddr address,
                           VAddr virtual_address, size_t size, Core::Asid asid,
                           u8* (*resolve)(void*, VAddr), void* ctx) {
        const size_t start_page_d = address >> Core::Memory::YUZU_PAGEBITS;
        const size_t num_pages =
            Common::AlignUp(size, Core::Memory::YUZU_PAGESIZE) >> Core::Memory::YUZU_PAGEBITS;
        std::scoped_lock lk(m.mapping_guard);
        m.tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages);
        for (size_t i = 0; i < num_pages; i++) {
            const VAddr new_vaddress = virtual_address + i * Core::Memory::YUZU_PAGESIZE;
            u8* ptr = resolve(ctx, new_vaddress);
            if (ptr == nullptr) {
                m.tracked_entries.GetUnchecked(start_page_d + i).compressed_physical_ptr = 0;
                continue;
            }
            const auto phys_addr =
                static_cast<u32>(m.GetRawPhysicalAddr(ptr) >> Core::Memory::YUZU_PAGEBITS) + 1U;
            m.tracked_entries.GetUnchecked(start_page_d + i).compressed_physical_ptr = phys_addr;
            m.InsertCPUBacking(start_page_d + i, new_vaddress, asid);
            const u32 base_dev = m.compressed_device_addr[phys_addr - 1U];
            REQUIRE(base_dev == 0); // multi-device branch is out of test scope
            m.compressed_device_addr.GetAndFault(phys_addr - 1U) =
                static_cast<u32>(start_page_d + i);
        }
        m.translation_generation.fetch_add(1, std::memory_order_release);
        // Test-only rigor: the packed cpu_backing_address must round-trip
        // (guards the guest_mask/ASID packing, which ReadBlockUnsafe never
        // inspects).
        for (size_t i = 0; i < num_pages; i++) {
            const auto [pk_asid, pk_vaddr] = m.ExtractCPUBacking(start_page_d + i);
            REQUIRE(pk_asid.id == asid.id);
            REQUIRE(pk_vaddr == virtual_address + i * Core::Memory::YUZU_PAGESIZE);
        }
    }

    // Test-only TLS isolation: t_cache is a static thread_local shared across
    // cases in one process; distinct DAddr spans key it apart, but reset it
    // explicitly so repeated same-case runs cannot reuse a stale host pointer.
    template <typename Traits>
    static void ResetTranslationCacheForTest(Core::DeviceMemoryManager<Traits>&) {
        Core::DeviceMemoryManager<Traits>::t_cache =
            typename Core::DeviceMemoryManager<Traits>::TranslationCache{};
    }

    // Cached-count branch precondition: exactly one registered slot, and it is
    // the nullptr process (the abort stubs trip on any deref).
    template <typename Traits>
    static void RequireSingleNullProcessForTest(Core::DeviceMemoryManager<Traits>& m) {
        REQUIRE(m.registered_processes.size() == 1);
        REQUIRE(m.registered_processes[0] == nullptr);
    }
};

namespace {
constexpr DAddr kSeamDev = 0x0000'0004'0000; // device address of the mapped span
constexpr VAddr kSeamGuest = 0x0000'0000'4000'0000; // 2^30 < 2^39 (guest_mask) // < 2^39 (guest_mask), nonzero, page-aligned, unique per case
constexpr PAddr kSeamPhys = 0x0000'0020'0000; // offset inside the DeviceMemory backing
constexpr u64 kSeamWindow = 4096; // one page
} // namespace

TEST_CASE("Criterion2 seam: MapForTest round-trips through real ReadBlockUnsafe", "[gpu_dirty]") {
    Core::DeviceMemory backing;
    Tegra::MaxwellDeviceMemoryManager gmmu{backing};
    gmmu.RegisterProcess(nullptr); // one slot; the abort stubs trip on any deref
    const Core::Asid asid{.id = 0};

    const auto resolver = +[](void* ctx, VAddr v) -> u8* {
        return static_cast<Core::DeviceMemory*>(ctx)->GetPointerFromRaw<u8>(kSeamPhys +
                                                                            (v - kSeamGuest));
    };
    Core::DeviceMemoryManagerTestAccess::MapForTest(gmmu, kSeamDev, kSeamGuest, kSeamWindow, asid,
                                              resolver, &backing);

    // Bytes written through the guest resolver land in the backing...
    std::memset(backing.GetPointerFromRaw<u8>(kSeamPhys), 0x5A, 128);
    // ...and the REAL upload read path (MappedUploadMemory -> ReadBlockUnsafe
    // -> tracked_entries.compressed_physical_ptr) returns exactly those bytes.
    std::array<u8, 128> out{};
    gmmu.ReadBlockUnsafe(kSeamDev, out.data(), out.size());
    REQUIRE(std::all_of(out.begin(), out.end(), [](u8 b) { return b == 0x5A; }));
    // Writes through the real WriteBlockUnsafe land at the same backing bytes.
    const std::array<u8, 64> in{[] { std::array<u8, 64> a{}; a.fill(0xC3); return a; }()};
    gmmu.WriteBlockUnsafe(kSeamDev + 128, in.data(), in.size());
    REQUIRE(std::memcmp(backing.GetPointerFromRaw<u8>(kSeamPhys + 128), in.data(),
                        in.size()) == 0);
}

namespace {
// Shared driver for the real consumer leg. When skip_collect_on_hit is true the
// hit-branch Collect is suppressed (the sensitive negative control).
// Each case MUST get a distinct DAddr span: TLS Translations() is keyed by
// (owner pointer, generation, DAddr guest page), and a fresh gmmu reuses the
// same stack address and generation across cases - same-DAddr cases would read
// the previous case's stale host pointer (observed: control run first poisoned
// the main case's step-3 upload). Distinct DAddr pages = distinct TLS keys.
template <typename F>
void RunConsumerCase(bool skip_collect_on_hit, DAddr kConsDev, VAddr kConsGuest,
                     F&& final_check) {
    constexpr PAddr kConsPhys = 0x0000'0030'0000;
    constexpr u64 kSpan = 65536;
    constexpr u64 kRegion = 64;

    Core::DeviceMemory backing;
    Tegra::MaxwellDeviceMemoryManager gmmu{backing};
    gmmu.RegisterProcess(nullptr); // one slot; abort stubs trip on any deref
    const Core::Asid asid{.id = 0};
    struct ResolverCtx {
        Core::DeviceMemory* mem;
        VAddr guest_base;
    } rctx{&backing, kConsGuest};
    const auto resolver = +[](void* ctx, VAddr v) -> u8* {
        auto* rc = static_cast<ResolverCtx*>(ctx);
        return rc->mem->template GetPointerFromRaw<u8>(kConsPhys + (v - rc->guest_base));
    };
    Core::DeviceMemoryManagerTestAccess::MapForTest(gmmu, kConsDev, kConsGuest, kSpan, asid,
                                                    resolver, &rctx);
    Core::DeviceMemoryManagerTestAccess::ResetTranslationCacheForTest(gmmu);
    Core::DeviceMemoryManagerTestAccess::RequireSingleNullProcessForTest(gmmu);

    ChainTestRuntime runtime;
    VideoCommon::BufferCache<ChainTestParams> cache{gmmu, runtime};
    Core::GPUDirtyMemoryManager manager;
    u8* const host_page = backing.GetPointerFromRaw<u8>(kConsPhys);

    PAddr gate_last_address = 0; // matches value-initialized rasterizer_write_areas
    unsigned on_cpu_write_calls = 0;
    unsigned collect_calls = 0;
    std::vector<std::pair<PAddr, size_t>> drains;

    // REAL cache.OnCPUWrite through the gate (production wiring), with a
    // counting wrapper so the hit-branch skip is directly observable.
    const auto on_cpu_write = [&](DAddr a, u64 s) {
        ++on_cpu_write_calls;
        return cache.OnCPUWrite(a, s);
    };
    const auto collect = [&](DAddr a, u64 s) {
        ++collect_calls;
        manager.Collect(a, s);
    };
    const auto skip_collect = [&](DAddr, u64) {};
    unsigned gate_calls = 0;
    const auto drive_gate = [&](PAddr paddr, u64 size) {
        std::function<bool(DAddr, u64)> cb_write = on_cpu_write;
        // The control suppresses Collect ONLY on the hit-path write (the
        // second gate touch); the first miss-path write must still collect,
        // or value2 never drains and the control stops being sensitive.
        const bool suppress = skip_collect_on_hit && gate_calls > 0;
        std::function<void(DAddr, u64)> cb_collect =
            suppress ? std::function<void(DAddr, u64)>(skip_collect)
                     : std::function<void(DAddr, u64)>(collect);
        ++gate_calls;
        Core::HandleRasterizerWriteGate(gate_last_address,
                                        paddr >> Core::Memory::YUZU_PAGEBITS, paddr, size, 0,
                                        cb_write, cb_collect);
    };
    // REAL drain consumer: GPU::InvalidateGPUCache -> BufferCache::WriteMemory.
    const auto gather = [&] {
        std::function<void(PAddr, size_t)> drain = [&](PAddr a, size_t s) {
            drains.emplace_back(a, s);
            cache.WriteMemory(a, static_cast<u64>(s));
        };
        manager.Gather(drain);
    };
    // Initial obtain syncs AND GPU-marks: the OnCPUWrite TRUE-branch
    // precondition (registered + GPU-modified) for the first deferred write.
    const auto sync_and_mark = [&] {
        auto [buf, off] =
            cache.ObtainCPUBuffer(kConsDev, kRegion,
                                  VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                                  VideoCommon::ObtainBufferOperation::MarkAsWritten);
        REQUIRE(off == 0);
        return static_cast<ChainTestBuffer*>(buf);
    };
    // Later obtains sync ONLY: no re-mark. The clean-between-writes state is
    // asserted, not recreated.
    const auto sync_only = [&] {
        auto [buf, off] =
            cache.ObtainCPUBuffer(kConsDev, kRegion,
                                  VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                                  VideoCommon::ObtainBufferOperation::DoNothing);
        REQUIRE(off == 0);
        return static_cast<ChainTestBuffer*>(buf);
    };

    // Step 1: initial value1, CPU-marked, delivered by the real upload path,
    // then GPU-marked (the OnCPUWrite TRUE-branch precondition).
    std::memset(host_page, 0x11, kRegion);
    cache.WriteMemory(kConsDev, kRegion);
    auto* buf = sync_and_mark();
    REQUIRE(buf->host.size() >= kRegion);
    REQUIRE(std::all_of(buf->host.begin(), buf->host.begin() + kRegion,
                        [](u8 b) { return b == 0x11; }));
    REQUIRE(cache.IsRegionGpuModified(kConsDev, kRegion));

    // Step 2: first write value2 through the gate -> real OnCPUWrite TRUE
    // branch, no cache marks (deferred), seeded + collected.
    std::memset(host_page, 0x22, kRegion);
    drive_gate(kConsDev, kRegion);
    REQUIRE(on_cpu_write_calls == 1);
    REQUIRE(collect_calls == 1);

    // Step 3: real Gather -> real WriteMemory -> real full sync delivers
    // value2 and leaves the range CLEAN - CPU clean AND GPU clean, and NOT
    // re-marked before the skipped second write (the sensitive precondition).
    gather();
    buf = sync_only();
    REQUIRE(std::all_of(buf->host.begin(), buf->host.begin() + kRegion,
                        [](u8 b) { return b == 0x22; }));
    REQUIRE(!cache.IsRegionCpuModified(kConsDev, kRegion));
    REQUIRE(!cache.IsRegionGpuModified(kConsDev, kRegion));
    const u64 staging_after_clean = runtime.staging_calls;

    // Step 4: second write value3 on the SAME bytes/page -> gate hit branch
    // skips OnCPUWrite; only Collect records it.
    const unsigned calls_before = on_cpu_write_calls;
    std::memset(host_page, 0x33, kRegion);
    drive_gate(kConsDev, kRegion);
    REQUIRE(on_cpu_write_calls == calls_before); // unchanged: the skip fired

    // Step 5: fresh collect/drain -> full sync (still no re-mark). With the
    // real chain, value3 MUST be delivered; with Collect suppressed it MUST
    // NOT.
    gather();
    buf = sync_only();
    final_check(*buf, drains, collect_calls, runtime.staging_calls, staging_after_clean, cache);
}
} // namespace

TEST_CASE("Criterion2 consumer: deferred second same-page write reaches the real buffer",
          "[gpu_dirty]") {
    RunConsumerCase(false, 0x0000'0000'0100'0000, 0x0000'0002'0000'0000,
                    [](ChainTestBuffer& buf, const std::vector<std::pair<PAddr, size_t>>& drains,
                       unsigned collect_calls, u64 staging_calls, u64,
                       VideoCommon::BufferCache<ChainTestParams>&) {
        // Exactly two collects (steps 2 and 4), two drains, and three real
        // staged uploads (value1, value2, value3).
        REQUIRE(collect_calls == 2);
        REQUIRE(staging_calls == 3);
        // Exactly two drains: one per collected write (steps 2 and 4).
        REQUIRE(drains.size() == 2);
        // The second drain covers the written 64 bytes.
        REQUIRE(drains[1].first <= 0x0000'0000'0100'0000);
        REQUIRE(drains[1].first + drains[1].second >= 0x0000'0000'0100'0000 + 64);
        REQUIRE(std::all_of(buf.host.begin(), buf.host.begin() + 64,
                            [](u8 b) { return b == 0x33; }));
    });
}

TEST_CASE("Criterion2 consumer control: without the skip-path Collect the write is lost",
          "[gpu_dirty]") {
    RunConsumerCase(true, 0x0000'0000'0200'0000, 0x0000'0003'0000'0000,
                    [](ChainTestBuffer& buf, const std::vector<std::pair<PAddr, size_t>>& drains,
                       unsigned collect_calls, u64 staging_calls, u64 staging_after_clean,
                       VideoCommon::BufferCache<ChainTestParams>& cache) {
        // The first (miss-path) write collected and drained exactly once...
        REQUIRE(collect_calls == 1);
        REQUIRE(drains.size() == 1);
        // ...the suppressed hit-path Collect marked nothing new...
        REQUIRE(!cache.IsRegionCpuModified(0x0000'0000'0200'0000, 64));
        // ...and no upload re-ran (staging untouched since the clean state)...
        REQUIRE(staging_calls == staging_after_clean);
        // ...so the buffer still holds value2 and value3 was lost.
        REQUIRE(std::all_of(buf.host.begin(), buf.host.begin() + 64,
                            [](u8 b) { return b == 0x22; }));
    });
}
