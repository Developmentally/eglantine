// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
//
// eglantine test 3 - diagnostic trace (evidence capture only).
// Additive instrumentation: no allocation, retry, or write-handling behavior
// changes. Events go to a bounded ring and are formatted at flush time.
// Intended flush cadence: once per emulated frame from TextureCache::TickFrame
// plus an explicit shutdown flush. This is an INTENDED cadence, NOT a proven
// crash-loss bound (stdout buffering, blocking, an unfinished frame, or thread
// interleaving can still lose buffered events).
//
// Cost accounting (overhead measurement remains UNTESTED): the hot path takes
// a mutex and appends to the ring; flush formats and logs under the same
// mutex; the texture-cache identity tables are std::unordered_map and their
// inserts can allocate (and therefore throw) on emulator paths.
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
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace Common::DiagTrace {

enum class Ev : u8 {
    ImageCreate = 0, // a=trace_obj b=slot_index c=gpu_addr d=byte_size tag=format/type/levels/layers
    ImageErase,      // a=trace_obj b=slot_index c=gpu_addr d=created_frame
    ImageRevive,     // a=trace_obj(SAME object) b=slot_index c=gpu_addr d=generation tag=lifetime
    RelatedEvict,    // a=old_trace_obj b=new_trace_obj c=old_gpu_addr tag=relation (eviction
                     // relation, NOT proven same semantic object)
    ViewCreate,      // a=view_slot b=image_trace_obj c=view_slot_generation d=view_type
    ViewCacheHit,    // a=view_slot b=image_trace_obj c=view_slot_generation
    ViewErase,       // a=view_slot
    Draw,            // a=draw_seq b=vertex_count c=instance_count d=indexed (in record lambda)
    DrawSkip,        // a=draw_seq tag=reason
    DescFinal,       // a=draw_seq b=ORIGINAL image_trace_obj (kept on substitution) c=view_slot d=final VkImageView handle; tag f=final kind k/n/o
                     // tag=stage/ordinal/type/substitution-reason
    DescRoute,       // a=draw_seq b=route(0=descbuf 1=push 2=updset 255=null-pipeline)
                     // c=num_entries d=reused_allocation (inside record lambda)
    RTBind,          // a=draw_seq b=image_trace_obj c=rt_index d=is_depth
    CopyDep,         // a=frame b=src_trace_obj c=dst_trace_obj d=size
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
    IdentitySnap,    // a=trace_obj b=slot c=gpu_addr d=state(0=live 1=deleted) tag=gen
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
    std::atomic<u64> next_obj_id{1};
    std::atomic<u64> draw_seq{0};
    std::atomic<s64> selected_draws_remaining{0};
    std::atomic<u64> lines_used{0};
    std::atomic<u64> current_frame{0}; // the one capture clock (texture-cache tick)
    std::atomic<u64> last_flush_frame{0};
    std::atomic<bool> enabled{false}; // EDEN_NX2_TRACE=1 master switch; OFF by default
    std::atomic<bool> capture_halted{false};
    std::atomic<bool> terminal_notice_done{false};
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

inline u64 EnvU64(const char* name, u64 fallback) {
    if (const char* v = std::getenv(name); v != nullptr && *v != '\0') {
        return std::strtoull(v, nullptr, 0);
    }
    return fallback;
}

inline void ReadEnvOnce() {
    std::call_once(InitFlag(), [] {
        State& s = S();
        s.selected_draws_remaining.store(
            static_cast<s64>(EnvU64("EDEN_NX2_TRACE_DRAWS", 0)), std::memory_order_relaxed);
        s.from_frame = EnvU64("EDEN_NX2_TRACE_FROM_FRAME", 0);
        s.to_frame = EnvU64("EDEN_NX2_TRACE_TO_FRAME", ~u64{0});
        s.flush_every = (std::max)(u64{1}, EnvU64("EDEN_NX2_TRACE_FLUSH_EVERY", 1));
        s.max_lines = EnvU64("EDEN_NX2_TRACE_MAX_LINES", 200000);
        s.enabled.store(EnvU64("EDEN_NX2_TRACE", 0) != 0, std::memory_order_relaxed);
        if (s.enabled.load(std::memory_order_relaxed)) {
            LOG_INFO(HW_GPU,
                     "T3TRACE init draws={} window=[{},{}] flush_every={} max_lines={}",
                     s.selected_draws_remaining.load(), s.from_frame, s.to_frame, s.flush_every,
                     s.max_lines);
        }
    });
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
inline bool Enabled() {
    ReadEnvOnce();
    return S().enabled.load(std::memory_order_relaxed);
}

inline thread_local u64 tl_draw_seq = 0;
inline thread_local bool tl_selected_draw = false;

// Begin a recorded draw/dispatch. Selected-draw budget is consumed ONLY for
// draws starting inside the capture window.
inline bool BeginDraw(u64 frame, u64& out_draw_seq) {
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

inline void Emit(Ev cls, u64 frame, u64 a, u64 b, u64 c, u64 d, const char* tag = "") {
    ReadEnvOnce();
    State& s = S();
    if (!s.enabled.load(std::memory_order_relaxed)) {
        return; // master switch off: one atomic load of overhead, no counters
    }
    // Exact capture boundary: once the halt is observed the event is neither
    // captured nor counted, so the counters printed at halt are final and
    // exact for everything before the boundary.
    if (s.capture_halted.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t i = static_cast<size_t>(cls);
    s.attempted.fetch_add(1, std::memory_order_relaxed);
    if (!InWindow(frame)) {
        s.skipped_window.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard lock{s.mutex};
    if (s.capture_halted.load(std::memory_order_relaxed)) {
        // Boundary race: this writer passed the pre-halt fast check but lost the
        // lock race against the halting flush. Dropped without counting; the
        // race window is bounded by the flush duration and acknowledged in the
        // terminal notice.
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
    out += fmt::format(" attempted={} skipped_window={} tag_truncated={}",
                       S().attempted.load(std::memory_order_relaxed),
                       S().skipped_window.load(std::memory_order_relaxed),
                       S().tag_truncated.load(std::memory_order_relaxed));
    return out;
}

// Formats and logs every buffered event under the ring mutex (the logger has
// its own lock and never calls back into DiagTrace, so no lock cycle). A
// summary line is printed only when this flush emitted at least one event.
inline void Flush(u64 frame, const char* why) {
    ReadEnvOnce();
    State& s = S();
    if (!s.enabled.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard lock{s.mutex};
    if (s.capture_halted.load(std::memory_order_relaxed)) {
        s.tail = s.head;
        if (!s.terminal_notice_done.exchange(true, std::memory_order_relaxed)) {
            LOG_CRITICAL(HW_GPU,
                         "T3TRACE halted: line budget {} exhausted. Events after the halt are "
                         "neither captured nor counted; counters are final and exact as of "
                         "the capture boundary:{}", s.max_lines, CounterSummary());
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
                         "events discarded and counted as dropped. Counters are final and "
                         "exact as of this halt; events after it are neither captured nor "
                         "counted (a writer racing this flush can lose one event uncounted, "
                         "bounded by the flush duration):{}",
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

// Intended-cadence tick: publishes the capture clock and flushes when the
// frame advanced past the cadence.
inline void TickFlush(u64 frame) {
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
