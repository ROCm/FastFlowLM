// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_profile.hpp
/// \brief Per-stage timings for the vision tower, behind
///        `-DMINICPM_V_VISION_PROFILE` (`make VISION_PROFILE=1`).
///
/// Ported from Qwen3.8-27B's tower. The engine has `DECODE_PROFILE`; this is the
/// tower's, and it exists for the same reason. Once both matmuls and the
/// attention are on bitstreams the tower is **glue-bound** — `npu/README.md`
/// puts ~1 s of the remaining ~1.7 s on the host — and "the host half" is not
/// an optimisation target until it is broken into stages that can be ranked.
///
/// Three groups of counter:
///
///   * the tower's own host math (`vision_tower`, `vision_glue`);
///   * the three host stages *inside* a `vision_mm` dispatch — the fused
///     fp32→bf16 narrow and bfp16 pre-round of x, the BO syncs, and the widen
///     of y back to fp32 — which live in `npu_vision_mm_op::run` and are
///     charged to it rather than to whichever of the 111 call sites was
///     running;
///   * the two device dispatches.
///
/// Like the other two profile flags this is a `-D` with no make dependency, so
/// **`make clean` first** (memory: make-does-not-track-debug-level).
///
/// The counters are process-global and not thread safe. Every one of them is
/// written from the serial driver thread — the `#pragma omp parallel for`s in
/// the stages are *inside* a bracket, never around one — so that is sound here
/// and would stop being sound the moment two images encode concurrently.
#ifndef __MINICPM_V_VISION_PROFILE_HPP__
#define __MINICPM_V_VISION_PROFILE_HPP__
#pragma once

#ifdef MINICPM_V_VISION_PROFILE
#include "utils/utils.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#endif

namespace mcv {
namespace vision {

#ifdef MINICPM_V_VISION_PROFILE

/// One encode's worth of microseconds, by stage.
struct prof {
    // ---- the tower's host math --------------------------------------------
    static inline double norm_us   = 0;   ///< layer_norm, 2 per block + merger
    static inline double split_us  = 0;   ///< qkv [L][3D] -> three [L][D]
    static inline double wattn_us  = 0;   ///< the vit_merger's 2x2 windowed attention
    static inline double gather_us = 0;   ///< 2x2 regroups + mean, both merges
    static inline double resid_us  = 0;   ///< add_inplace, 2 per block
    static inline double pos_us    = 0;   ///< the bucketized position lookup

    // ---- inside npu_vision_mm_op::run --------------------------------------
    static inline double mm_x_us   = 0;   ///< fp32 -> bf16 narrow + bfp16
                                          ///  pre-round of x, one fused pass
    static inline double mm_sync_us= 0;   ///< sync_to_device + sync_from_device
    static inline double mm_y_us   = 0;   ///< bf16 -> fp32 widen (+ erf gelu)

    // ---- device -------------------------------------------------------------
    static inline double mm_dev_us   = 0; ///< vision_mm dispatches
    static inline double attn_dev_us = 0; ///< vision_attn dispatches
    static inline double attn_host_us= 0; ///< its own marshalling

    static inline double wait_us     = 0; ///< streamed path: host blocked on the device

    static inline long   mm_calls    = 0;

    // ---- timeline (FLM_MINICPM_V_VISION_TRACE=<file>): every VPROF_EXPR / VTRACE span, appended per encode ----
    struct span { const char* tag; long long t0, t1; };
    static inline std::mutex        trace_mu;
    static inline std::vector<span> trace;
    static const char* trace_path() {
        static const char* p = std::getenv("FLM_MINICPM_V_VISION_TRACE");
        return p;
    }
    static long long now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static void add_span(const char* tag, long long t0, long long t1) {
        if (!trace_path()) return;
        std::lock_guard<std::mutex> g(trace_mu);
        trace.push_back({tag, t0, t1});
    }
    static void dump_trace(double total_ms) {
        if (!trace_path()) return;
        std::lock_guard<std::mutex> g(trace_mu);
        if (FILE* f = std::fopen(trace_path(), "a")) {
            std::fprintf(f, "encode %.3f %zu\n", total_ms, trace.size());
            for (const span& e : trace) std::fprintf(f, "%s %lld %lld\n", e.tag, e.t0, e.t1);
            std::fclose(f);
        }
        trace.clear();
    }

    static void reset() {
        norm_us = split_us = wattn_us = gather_us = resid_us = pos_us = 0;
        mm_x_us = mm_sync_us = mm_y_us = 0;
        mm_dev_us = attn_dev_us = attn_host_us = wait_us = 0;
        mm_calls = 0;
        std::lock_guard<std::mutex> g(trace_mu);
        trace.clear();
    }

    /// One table, after an encode. `total_ms` is the caller's wall clock, so
    /// the residual line says how much is in stages nobody brackets.
    static void report(double total_ms) {
        dump_trace(total_ms);
        const double dev  = mm_dev_us + attn_dev_us;
        const double host = norm_us + split_us + wattn_us + gather_us + resid_us + pos_us +
                            mm_x_us + mm_sync_us + mm_y_us +
                            attn_host_us;
        auto row = [&](const char* name, double us) {
            std::printf("   %-26s %8.1f ms  %5.1f%%\n", name, us / 1000.0,
                        total_ms > 0 ? 100.0 * us / 1000.0 / total_ms : 0.0);
        };
        std::printf("\n-- vision encode, %.1f ms, %ld mm dispatches --\n",
                    total_ms, mm_calls);
        row("device, vision_mm",      mm_dev_us);
        row("device, vision_attn",    attn_dev_us);
        row("host, x narrow+preround", mm_x_us);
        row("host, BO syncs",         mm_sync_us);
        row("host, y bf16->fp32",     mm_y_us);
        row("host, layer_norm",       norm_us);
        row("host, qkv split",        split_us);
        row("host, window attention",  wattn_us);
        row("host, 2x2 regroup + mean", gather_us);
        row("host, residual add",     resid_us);
        row("host, pos embed lookup", pos_us);
        row("host, attention glue",   attn_host_us);
        row("host waiting on device (streamed)", wait_us);
        std::printf("   %-26s %8.1f ms  %5.1f%%\n", "-- device", dev / 1000.0,
                    100.0 * dev / 1000.0 / total_ms);
        std::printf("   %-26s %8.1f ms  %5.1f%%\n", "-- host, bracketed",
                    host / 1000.0, 100.0 * host / 1000.0 / total_ms);
        std::printf("   %-26s %8.1f ms  %5.1f%%\n", "-- unbracketed",
                    total_ms - (dev + host) / 1000.0,
                    100.0 * (total_ms - (dev + host) / 1000.0) / total_ms);
    }
};

/// Adds its own lifetime to one counter. A scope guard rather than the
/// open/close mark pairs module 1 uses: the stages here nest (a mm dispatch's
/// four host brackets sit inside no other bracket, but the attention op's do
/// sit inside the tower's), and a guard cannot be left open on an early return.
struct prof_scope {
    double* dst;
    time_utils::time_point t0;
    explicit prof_scope(double* d) : dst(d), t0(time_utils::now()) {}
    /// Nanoseconds, scaled: `duration_us` truncates to whole microseconds and
    /// this is accumulated over ~500 brackets an encode, several of which are
    /// genuinely sub-microsecond.
    ~prof_scope() {
        *dst += time_utils::duration_ns(t0, time_utils::now()).first / 1000.0;
    }
};

/// One span of the timeline, for the rest of the enclosing scope.
struct trace_scope {
    const char* tag;
    long long   t0;
    explicit trace_scope(const char* t) : tag(t), t0(prof::trace_path() ? prof::now_ns() : 0) {}
    ~trace_scope() { if (prof::trace_path()) prof::add_span(tag, t0, prof::now_ns()); }
};
#define VTRACE(tag) mcv::vision::trace_scope VPROF_CAT(__vtrace_, __LINE__)(tag)
#define VPROF_CAT2(a, b) a##b
#define VPROF_CAT(a, b)  VPROF_CAT2(a, b)
/// Charge the rest of the enclosing scope to `prof::<field>`.
#define VPROF(field) \
    mcv::vision::prof_scope VPROF_CAT(__vprof_, __LINE__)( \
        &mcv::vision::prof::field); \
    VTRACE(#field)
/// Charge one expression, and yield its value.
#define VPROF_EXPR(field, expr) \
    [&] { VPROF(field); return (expr); }()
#define VPROF_COUNT(field) (mcv::vision::prof::field++)

#else   // ------------------------------------------------------------------

/// The same two entry points, so a driver can call them unconditionally.
struct prof {
    static void reset() {}
    static void report(double) {}
};

#define VPROF(field)            do {} while (0)
#define VPROF_EXPR(field, expr) (expr)
#define VTRACE(tag)             do {} while (0)
#define VPROF_COUNT(field)      do {} while (0)

#endif

}  // namespace vision
}  // namespace mcv

#endif  // __MINICPM_V_VISION_PROFILE_HPP__
