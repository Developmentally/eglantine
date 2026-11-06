// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
//
// eglantine test 3 - diagnostic trace (evidence capture only).
// Additive instrumentation: no allocation, retry, or write-handling behavior
// changes. Events go to a bounded ring and are formatted at flush time.
// Flush cadence: once per emulated frame from TextureCache::TickFrame ONLY.
// There is NO shutdown/exit flush: events still buffered after the final frame
// tick are lost (harness-measured tail), and stdio buffering, blocking, an
// unfinished frame, or thread interleaving can lose more. The last printed
// flush before exit is the final evidence; no final print is guaranteed.
//
// Cost accounting (overhead measurement remains UNTESTED): the hot path takes
// a mutex and appends to the ring; flush formats and logs under the same
// mutex; the texture-cache identity tables are std::unordered_map and their
// inserts can allocate (and therefore throw) on emulator paths.
//
// A contained observer fault inside Observe() can consume an identity id
// before its map insert: identity gaps are possible and observable via
// observer_errors, never silent corruption of the emulator state.
//
// Counter semantics: emitted[X] = ring writes; dropped[X] = ring-overflow
// victims charged to the VICTIM's class plus budget-halt discards (exact at
// all times); observer_errors = contained observer faults (init, emit, flush
// and Observe() catches), kept SEPARATE so victim attribution stays pure;
// late_after_halt = arrivals after the capture boundary (monotonic while
// running, never claimed final; printed only at the next flush, if any).
//
// One capture clock: the texture-cache frame tick, published via SetFrame.
// Core-side (HRW) events read CurrentFrame(); before the first tick they are
// frame 0. Window checks apply to every event uniformly.
//
// Env knobs (read once, std::call_once):
// Budget scope: max_lines counts formatted EVENT lines; control lines (init,
// flush summaries, halt notice) are bounded separately (frames + 2).
//   EDEN_NX2_TRACE=1              master switch; capture is OFF without it
//   EDEN_NX2_TRACE_DRAWS=N        fully capture the next N recorded draws that
//                                 START INSIDE the window (all final
//                                 descriptors, RT ids, buffer epochs, copy deps)
//   EDEN_NX2_TRACE_FROM_FRAME=F   capture window start (default 0)
//   EDEN_NX2_TRACE_TO_FRAME=F     capture window end (default UINT64_MAX)
//   EDEN_NX2_TRACE_FLUSH_EVERY=N  intended flush cadence in frames (default 1)
//   EDEN_NX2_TRACE_MAX_LINES=N    formatted line budget (default 200000); when
//                                 exhausted, buffered events are counted into
//                                 dropped per class, capture halts, ONE
//                                 terminal notice is logged, counters continue.

#pragma once

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace Common::DiagTrace {

enum class Ev : u8 {
    ImageCreate = 0, // a=trace_obj b=slot_index c=gpu_addr d=byte_size tag=format/type/levels/layers
    ImageErase,      // a=trace_obj b=slot_index c=gpu_addr d=created_frame
                     // tag: as<manager-id+1> c=<cause> r=<disposition AT SLOT FREE>:
                     // retained/immediate/sentenced is NOT the backing-release time
                     // (sentenced-ring pop and retained-download completion emit no
                     // event) - never infer VRAM release from this line
    ImageRevive,     // a=trace_obj(SAME object) b=slot_index c=gpu_addr d=generation tag=lifetime
    RelatedEvict,    // a=old_trace_obj b=new_trace_obj c=old_gpu_addr tag=relation (eviction
                     // relation, NOT proven same semantic object)
    ViewCreate,      // a=view_slot b=image_trace_obj c=view_slot_generation d=view_type
                     // (generation is in c here, but in d on ViewErase - read the right field)
    ViewCacheHit,    // a=view_slot b=image_trace_obj c=view_slot_generation
    ViewErase,       // a=view_slot d=erased generation (0 = unknown/cap-skipped, not "gen 0")
    Draw,            // a=draw_seq b=vertex_count c=instance_count d=indexed (in record lambda)
    DrawSkip,        // a=draw_seq tag=reason
    DescFinal,       // a=draw_seq b=ORIGINAL image_trace_obj (kept on substitution) c=view_slot d=final VkImageView handle; tag f=final kind k/n/o
                     // tag=stage/ordinal/type/substitution-reason
    DescRoute,       // a=draw_seq b=route(0=descbuf 1=push 2=updset 255=null-pipeline)
                     // c=num_entries d=reused_allocation (inside record lambda)
    RTBind,          // a=draw_seq b=image_trace_obj c=rt_index d=is_depth
                     // tag: v<view slot> g<view gen> (view lineage of the bound RT)
                     // CHUNK-BATCH GAP: with EDEN_NX2_CHUNK_BATCH=1, batched draws skip
                     // ConfigureDraw entirely - no Draw events (silent consumed seqs) and
                     // the batch GetFramebuffer fires at flush under a DIFFERENT draw's
                     // seq, misattributing RTBind. Diagnostic captures REQUIRE the
                     // default (unset) mode; the run manifest must record the actual env
    CopyDep,         // a=src_trace_obj b=dst_trace_obj c=copy-region count/layers
                     // (0 = no count at that site; NEVER a byte claim) tag=route.
                     // Emitted AT the actual issue route, never on a request:
                     // copy-direct/copy-emulated/copy-reinterpret/copy-convert
                     // (inside the CopyImage wrapper, after the astc_opaque no-copy
                     // return; the tag names the route, not the caller - covers
                     // join-alias, SynchronizeAliases and all other wrapper
                     // callers), join-resize/join-shrink[-msaa] (direct runtime
                     // calls in the join), blit (TextureCache::BlitImage issue
                     // point). CopyImageMSAA outside the join is NOT covered
    BufEpoch,        // a=write_epoch b=device_addr c=size
    HrwSeed,         // a=core b=page c=address d=size
    HrwSkip,         // a=core b=page c=address d=size (shortcut hit, OnCPUWrite not called)
    HrwCollect,      // a=core b=address c=size
    HrwBufAccept,    // a=address b=size (registered+GPU-modified -> true)
    HrwBufReject,    // a=address b=size c=reason(0=unregistered 1=clean)
    HrwGather,       // tag=trigger site
    HrwDrain,        // a=address b=size
    HrwCacheInval,   // a=address b=size (OnCacheInvalidation invoked - invalidation, not use)
    HrwUpload,       // a=buffer_cpu_addr b=device_addr c=size (actual upload copy)
    Config,          // a=kind b/c/d=value; tag=name
    IdentitySnap,    // a=trace_obj b=slot c=gpu_addr d=state(0=live 1=deleted 2=dormant) tag=gen
                     // taken at the FIRST in-window TickFrame (not necessarily the exact
                     // window opening); deleted pre-window records are counted, not listed,
                     // and their erase events are OUTSIDE the capture - the snapshot plus
                     // the window does NOT give full lineage. Skip counters are cumulative
                     // SINCE INIT, not window-only. COMPLETENESS TEST (all required):
                     // the identity-snapshot-states Config line is present AND
                     // dropped[IdentitySnap]==0 in the NEXT successful flush summary
                     // AND no flush_broken/observer_errors. Marker presence alone is
                     // NOT sufficient (reproduced: marker printed with 4858 snapshot
                     // drops under concurrent emitters). Backend losses (log filter,
                     // file cap) are uncounted. Chunk flushes bound but never
                     // guarantee against loss; losses are counted, not prevented
    ImageDormant,    // a=trace_obj b=slot c=gpu_addr d=created_frame tag=encoded asid
    Count
};

struct Event {
    u64 seq;
    u64 frame;
    u32 cls;
    u64 a, b, c, d;
    char tag[48];
};

struct State {
    std::mutex mutex;
    std::array<Event, 16384> ring;
    u64 head = 0;
    u64 tail = 0;
    std::array<std::atomic<u64>, static_cast<size_t>(Ev::Count)> emitted{};
    std::array<std::atomic<u64>, static_cast<size_t>(Ev::Count)> dropped{};
    std::atomic<u64> attempted{0};
    std::atomic<u64> skipped_window{0};
    std::atomic<u64> tag_truncated{0}; // descriptor tags that did not fit their field
    std::atomic<u64> identity_skipped{0}; // image-identity records dropped at the capacity bound
    std::atomic<u64> view_gen_skipped{0};  // view-gen entries dropped at the capacity bound
    std::atomic<u64> observer_errors{0}; // contained observer faults (NOT overflow victims)
    std::atomic<u64> late_after_halt{0}; // arrivals after the capture boundary
    std::atomic<u64> init_runs{0}; // times the init lambda body ran (must stay 1)
    std::atomic<u64> next_obj_id{1};
    std::atomic<u64> draw_seq{0};
    std::atomic<s64> selected_draws_remaining{0};
    std::atomic<u64> lines_used{0};
    std::atomic<u64> current_frame{0}; // the one capture clock (texture-cache tick)
    std::atomic<u64> last_flush_frame{0};
    std::atomic<bool> enabled{false}; // EDEN_NX2_TRACE=1 master switch; OFF by default
    std::atomic<bool> capture_halted{false};
    std::atomic<bool> terminal_notice_done{false};
    std::atomic<bool> init_broken{false}; // call_once mechanics failed; never retry init
    std::atomic<bool> flush_broken{false}; // a flush faulted; stop future flush attempts
#if defined(EDEN_NX2_TRACE_TEST_HOOKS)
    // Compile-time test-only fault injection. Never defined in shipping builds;
    // zero hook code exists there. The catch-alls are always compiled - the
    // hooks merely exercise them.
    std::atomic<u64> test_fail_next{0}; // throw std::bad_alloc on the next N enabled emits/flushes
    std::atomic<u64> test_fail_skip{0}; // ...after letting this many enabled emits/flushes pass
#endif
    // Set exactly once via std::call_once before any use:
    u64 from_frame = 0;
    u64 to_frame = ~u64{0};
    u64 flush_every = 1;
    u64 max_lines = 200000;
};

inline State& S() {
    static State s;
    return s;
}
inline std::once_flag& InitFlag() {
    static std::once_flag f;
    return f;
}

// Isolation catches run on a fully noexcept path: every State member is a
// mutex, scalar aggregate, or atomic with a non-throwing default constructor,
// so magic-static completion of S() cannot throw a C++ exception.
static_assert(std::is_nothrow_default_constructible_v<State>);

inline u64 EnvU64(const char* name, u64 fallback) {
    if (const char* v = std::getenv(name); v != nullptr && *v != '\0') {
        return std::strtoull(v, nullptr, 0);
    }
    return fallback;
}

// Two-tier noexcept init. Tier 1 (inside the lambda): any fault - the only
// thrower is the guarded LOG_INFO/fmt, EnvU64 (getenv/strtoull) never throws -
// degrades to disabled, is counted, and the lambda COMPLETES so the once-flag
// latches: no throw+retry per Emit. Tier 2 (outside): a std::system_error from
// the call_once mechanics itself leaves the once-flag unset, so a permanent
// init_broken latch makes every later call return immediately, disabled.
inline void ReadEnvOnce() noexcept {
    State& s = S();
    if (s.init_broken.load(std::memory_order_relaxed)) {
        return;
    }
    try {
        std::call_once(InitFlag(), [&s] {
            s.init_runs.fetch_add(1, std::memory_order_relaxed);
            try {
                s.selected_draws_remaining.store(
                    static_cast<s64>(EnvU64("EDEN_NX2_TRACE_DRAWS", 0)),
                    std::memory_order_relaxed);
                s.from_frame = EnvU64("EDEN_NX2_TRACE_FROM_FRAME", 0);
                s.to_frame = EnvU64("EDEN_NX2_TRACE_TO_FRAME", ~u64{0});
                s.flush_every = (std::max)(u64{1}, EnvU64("EDEN_NX2_TRACE_FLUSH_EVERY", 1));
                s.max_lines = EnvU64("EDEN_NX2_TRACE_MAX_LINES", 200000);
                s.enabled.store(EnvU64("EDEN_NX2_TRACE", 0) != 0, std::memory_order_relaxed);
#if defined(EDEN_NX2_TRACE_TEST_HOOKS)
                if (std::getenv("EDEN_NX2_TRACE_TEST_FAIL_INIT") != nullptr) {
                    throw std::bad_alloc{}; // synthetic: exercises the init catch
                }
#endif
                if (s.enabled.load(std::memory_order_relaxed)) {
                    LOG_INFO(HW_GPU,
                             "T3TRACE init draws={} window=[{},{}] flush_every={} max_lines={}",
                             s.selected_draws_remaining.load(), s.from_frame, s.to_frame,
                             s.flush_every, s.max_lines);
                }
            } catch (...) {
                s.enabled.store(false, std::memory_order_relaxed);
                s.observer_errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    } catch (...) {
        s.init_broken.store(true, std::memory_order_relaxed);
        s.enabled.store(false, std::memory_order_relaxed);
        s.observer_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

inline bool InWindow(u64 frame) {
    const State& s = S();
    return frame >= s.from_frame && frame <= s.to_frame;
}

inline u64 CurrentFrame() {
    return S().current_frame.load(std::memory_order_relaxed);
}

// Master-switch read for guarding pre-Emit diagnostic work (identity maps,
// tag formatting, generation maps): when OFF those sites skip ALL observer
// work, not just the Emit call.
inline bool Enabled() noexcept {
    ReadEnvOnce();
    return S().enabled.load(std::memory_order_relaxed);
}

// Scoped observer-only containment for diagnostic work OUTSIDE Emit (identity
// map inserts, tag formatting, generation lookups at call sites). Encloses
// ONLY diagnostic work - the original callbacks/pipeline logic stay outside.
template <class F>
inline void Observe(F&& f) noexcept {
    if (!Enabled()) {
        return;
    }
    try {
        f();
    } catch (...) {
        S().observer_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

#if defined(EDEN_NX2_TRACE_TEST_HOOKS)
// Test-only: arm synthetic std::bad_alloc faults on the next `n` enabled
// emits/flushes, letting `skip` of them pass first. Never in shipping builds.
inline void TestArmFailNext(u64 n, u64 skip = 0) noexcept {
    S().test_fail_skip.store(skip, std::memory_order_relaxed);
    S().test_fail_next.store(n, std::memory_order_relaxed);
}
#endif

inline thread_local u64 tl_draw_seq = 0;
inline thread_local bool tl_selected_draw = false;

// Begin a recorded draw/dispatch. Selected-draw budget is consumed ONLY for
// draws starting inside the capture window.
inline bool BeginDrawImpl(u64 frame, u64& out_draw_seq) {
    ReadEnvOnce();
    State& s = S();
    if (!s.enabled.load(std::memory_order_relaxed)) {
        out_draw_seq = 0;
        tl_draw_seq = 0;
        tl_selected_draw = false;
        return false;
    }
    out_draw_seq = s.draw_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    bool selected = false;
    if (InWindow(frame)) {
        s64 remaining = s.selected_draws_remaining.load(std::memory_order_relaxed);
        while (remaining > 0 &&
               !s.selected_draws_remaining.compare_exchange_weak(remaining, remaining - 1,
                                                               std::memory_order_relaxed)) {
        }
        selected = remaining > 0;
    }
    tl_draw_seq = out_draw_seq;
    tl_selected_draw = selected;
    return selected;
}

inline void EndDraw() {
    tl_draw_seq = 0;
    tl_selected_draw = false;
}

inline bool BeginDraw(u64 frame, u64& out_draw_seq) noexcept {
    try {
        return BeginDrawImpl(frame, out_draw_seq);
    } catch (...) {
        // Contained: no half-initialized draw tracking, and the fault is
        // counted rather than silent.
        out_draw_seq = 0;
        tl_draw_seq = 0;
        tl_selected_draw = false;
        S().observer_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

inline void EmitImpl(Ev cls, u64 frame, u64 a, u64 b, u64 c, u64 d, const char* tag);

// Observer failure isolation: tracing must never change the behavior it
// observes. The fast path is the noexcept Enabled() check; EVERYTHING else
// (halt checks, counters, lock, ring write, snprintf) sits inside one try, and
// the catch performs a single noexcept atomic increment - no terminate path
// exists, and a throw can never skip a gated Collect, a draw record, or a
// cache mark. Faults charge observer_errors, NEVER dropped[] (which stays pure
// ring-overflow victim attribution).
inline void Emit(Ev cls, u64 frame, u64 a, u64 b, u64 c, u64 d, const char* tag = "") noexcept {
    if (!Enabled()) {
        return;
    }
    try {
        EmitImpl(cls, frame, a, b, c, d, tag);
    } catch (...) {
        S().observer_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// Precondition: the caller (Emit) already established Enabled(). Called only
// inside Emit's try region.
inline void EmitImpl(Ev cls, u64 frame, u64 a, u64 b, u64 c, u64 d, const char* tag) {
    State& s = S();
#if defined(EDEN_NX2_TRACE_TEST_HOOKS)
    if (s.test_fail_next.load(std::memory_order_relaxed) != 0) {
        if (s.test_fail_skip.load(std::memory_order_relaxed) != 0) {
            s.test_fail_skip.fetch_sub(1, std::memory_order_relaxed);
        } else {
            s.test_fail_next.fetch_sub(1, std::memory_order_relaxed);
            throw std::bad_alloc{}; // synthetic: exercises the isolation catch
        }
    }
#endif
    // Exact capture boundary: once the halt is observed the event is neither
    // captured nor per-class counted; it accumulates in late_after_halt.
    if (s.capture_halted.load(std::memory_order_relaxed)) {
        s.late_after_halt.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const size_t i = cls < Ev::Count ? static_cast<size_t>(cls) : 0;
    s.attempted.fetch_add(1, std::memory_order_relaxed);
    if (!InWindow(frame)) {
        s.skipped_window.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard lock{s.mutex};
    if (s.capture_halted.load(std::memory_order_relaxed)) {
        // Boundary race: this writer passed the pre-halt fast check but lost the
        // lock race against the halting flush. Counted in late_after_halt only;
        // the race window is bounded by the flush duration and acknowledged in
        // the terminal notice.
        s.late_after_halt.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (s.head - s.tail == s.ring.size()) {
        // Ring full: the OLDEST event is overwritten; charge the VICTIM's class.
        const Event& victim = s.ring[s.tail % s.ring.size()];
        const size_t vi = victim.cls < static_cast<u32>(Ev::Count) ? victim.cls : 0;
        s.dropped[vi].fetch_add(1, std::memory_order_relaxed);
        ++s.tail;
    }
    Event& e = s.ring[s.head % s.ring.size()];
    e.seq = s.head;
    e.frame = frame;
    e.cls = static_cast<u32>(cls);
    e.a = a;
    e.b = b;
    e.c = c;
    e.d = d;
    std::snprintf(e.tag, sizeof(e.tag), "%s", tag);
    ++s.head;
    s.emitted[i].fetch_add(1, std::memory_order_relaxed);
}

inline const char* ClsName(u32 cls) {
    static constexpr std::array names{
        "ImageCreate", "ImageErase",  "ImageRevive", "RelatedEvict", "ViewCreate",
        "ViewCacheHit", "ViewErase",  "Draw",        "DrawSkip",     "DescFinal",
        "DescRoute",   "RTBind",      "CopyDep",     "BufEpoch",     "HrwSeed",
        "HrwSkip",     "HrwCollect",  "HrwBufAccept","HrwBufReject", "HrwGather",
        "HrwDrain",    "HrwCacheInval","HrwUpload",  "Config",       "IdentitySnap",
        "ImageDormant",
    };
    return cls < names.size() ? names[cls] : "Unknown";
}

inline std::string CounterSummary() {
    const State& s = S();
    std::string out;
    for (size_t i = 0; i < static_cast<size_t>(Ev::Count); ++i) {
        const u64 em = s.emitted[i].load(std::memory_order_relaxed);
        const u64 dr = s.dropped[i].load(std::memory_order_relaxed);
        if (em != 0 || dr != 0) {
            out += fmt::format(" {}={}/drop{}", ClsName(static_cast<u32>(i)), em, dr);
        }
    }
    // identity/view-gen skip counters repeat in EVERY flush summary (D7), not
    // just in the one-time snapshot Config event.
    out += fmt::format(
        " attempted={} skipped_window={} tag_truncated={} observer_errors={} late_after_halt={}"
        " identity_skipped={} view_gen_skipped={}",
        S().attempted.load(std::memory_order_relaxed),
        S().skipped_window.load(std::memory_order_relaxed),
        S().tag_truncated.load(std::memory_order_relaxed),
        S().observer_errors.load(std::memory_order_relaxed),
        S().late_after_halt.load(std::memory_order_relaxed),
        S().identity_skipped.load(std::memory_order_relaxed),
        S().view_gen_skipped.load(std::memory_order_relaxed));
    if (S().init_broken.load(std::memory_order_relaxed)) {
        out += " init_broken=1";
    }
    if (S().flush_broken.load(std::memory_order_relaxed)) {
        out += " flush_broken=1";
    }
    return out;
}

// Formats and logs every buffered event under the ring mutex (the logger has
// its own lock and never calls back into DiagTrace, so no lock cycle). A
// summary line is printed only when this flush emitted at least one event.
// Precondition: the caller (Flush) already established Enabled(). Called only
// inside Flush's try region.
inline void FlushImpl(u64 frame, const char* why) {
    State& s = S();
    if (s.flush_broken.load(std::memory_order_relaxed)) {
        return; // a previous flush faulted; events stay in the ring, capture continues
    }
#if defined(EDEN_NX2_TRACE_TEST_HOOKS)
    if (s.test_fail_next.load(std::memory_order_relaxed) != 0) {
        if (s.test_fail_skip.load(std::memory_order_relaxed) != 0) {
            s.test_fail_skip.fetch_sub(1, std::memory_order_relaxed);
        } else {
            s.test_fail_next.fetch_sub(1, std::memory_order_relaxed);
            throw std::bad_alloc{}; // synthetic: exercises the isolation catch
        }
    }
#endif
    std::lock_guard lock{s.mutex};
    if (s.capture_halted.load(std::memory_order_relaxed)) {
        s.tail = s.head;
        if (!s.terminal_notice_done.exchange(true, std::memory_order_relaxed)) {
            LOG_CRITICAL(HW_GPU,
                         "T3TRACE halted: line budget {} exhausted. Counters are exact as of "
                         "the capture boundary; later arrivals are neither captured nor "
                         "per-class counted but accumulate in late_after_halt, printed at "
                         "the next flush if one happens (no final print is guaranteed - "
                         "there is no shutdown flush):{}", s.max_lines, CounterSummary());
        }
        return;
    }
    bool flushed_any = false;
    while (s.tail < s.head) {
        if (s.lines_used.load(std::memory_order_relaxed) >= s.max_lines) {
            // Budget exhausted: count every discarded buffered event into its class.
            for (u64 p = s.tail; p < s.head; ++p) {
                const Event& e = s.ring[p % s.ring.size()];
                s.dropped[e.cls < static_cast<u32>(Ev::Count) ? e.cls : 0].fetch_add(
                    1, std::memory_order_relaxed);
            }
            s.tail = s.head;
            s.capture_halted.store(true, std::memory_order_relaxed);
            s.terminal_notice_done.store(true, std::memory_order_relaxed);
            LOG_CRITICAL(HW_GPU,
                         "T3TRACE capture halted: event-line budget {} exhausted; buffered "
                         "events discarded and counted as dropped. Counters are exact as of "
                         "this halt; later arrivals are neither captured nor per-class "
                         "counted but accumulate in late_after_halt (a writer racing this "
                         "flush can add one such arrival, bounded by the flush duration), "
                         "printed at the next flush if one happens (no final print is "
                         "guaranteed - there is no shutdown flush):{}",
                         s.max_lines, CounterSummary());
            return;
        }
        const Event& e = s.ring[s.tail % s.ring.size()];
        LOG_INFO(HW_GPU, "T3TRACE seq={} cls={} frame={} a={} b={} c={:#x} d={} tag={}", e.seq,
                 ClsName(e.cls), e.frame, e.a, e.b, e.c, e.d, e.tag);
        s.lines_used.fetch_add(1, std::memory_order_relaxed);
        ++s.tail;
        flushed_any = true;
    }
    if (flushed_any) {
        LOG_INFO(HW_GPU, "T3TRACE flush frame={} why={} counters:{}", frame, why,
                 CounterSummary());
    }
}

// noexcept wrapper: a throwing flush (format/allocation when enabled) is
// contained and counted, and the fail-safe latch stops a deterministically
// faulting flush from throwing every frame tick. Buffered events stay in the
// ring; capture continues with ring-overflow accounting in dropped[].
inline void Flush(u64 frame, const char* why) noexcept {
    if (!Enabled()) {
        return;
    }
    try {
        FlushImpl(frame, why);
    } catch (...) {
        State& s = S();
        s.observer_errors.fetch_add(1, std::memory_order_relaxed);
        if (!s.flush_broken.exchange(true, std::memory_order_relaxed)) {
            // Best-effort one-time notice: without it a first-flush fault was
            // silent forever (flush_broken and the drops only lived in
            // counters no successful flush would print). The flush path just
            // faulted, so logging itself can fail - contained, never counted.
            try {
                LOG_CRITICAL(HW_GPU,
                             "T3TRACE flush faulted; flush_broken latched, future flushes "
                             "skipped, buffered events remain in the ring");
            } catch (...) {
            }
        }
    }
}

// Intended-cadence tick: publishes the capture clock and flushes when the
// frame advanced past the cadence.
inline void TickFlush(u64 frame) noexcept {
    ReadEnvOnce();
    State& s = S();
    if (!s.enabled.load(std::memory_order_relaxed)) {
        return;
    }
    s.current_frame.store(frame, std::memory_order_relaxed);
    const u64 last = s.last_flush_frame.load(std::memory_order_relaxed);
    if (last == 0 || frame - last >= s.flush_every) {
        s.last_flush_frame.store(frame, std::memory_order_relaxed);
        Flush(frame, "tick");
    }
}

} // namespace Common::DiagTrace
