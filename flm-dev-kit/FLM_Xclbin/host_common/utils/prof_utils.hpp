// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __PROF_UTILS_HPP__
#define __PROF_UTILS_HPP__

#include "utils/utils.hpp"

// ---------------------------------------------------------------------------
// Lightweight stage profiling, shared across execution contexts.
//
// Enable by compiling with -DFLM_PROFILE (all macros below compile to nothing
// otherwise, so they are zero-cost when disabled).
//
// Unlike a print-on-every-tick profiler, PROF_TICK *accumulates* elapsed time
// into a process-wide registry keyed by the stage name. A stage that runs once
// per loop iteration (e.g. the 12 identical vision layers) is summed under a
// single name instead of flooding the log with one line per iteration. Call
// PROF_REPORT() once at the end to print a clean table: per stage the call
// count, total, and average.
//
//   PROF_INIT();               // once at the start of a scope (local cursor)
//   ... work ...
//   PROF_TICK("stage_a");      // add time since previous tick to "stage_a"
//   ... work ...
//   PROF_TICK("stage_b");
//   ...
//   PROF_REPORT();             // print aggregated table, then
//   PROF_RESET();              // clear counters (e.g. between images)
//
// PROF_INIT declares a local cursor, so nested scopes each keep their own and
// timings do not bleed across function boundaries.
//
// PROF_TICK measures from the *previous tick*, so it cannot time a region that
// contains ticks of its own (the inner ticks move the cursor, leaving the outer
// tick with only the tail). Use a span for that -- it carries its own cursor:
//
//   PROF_SPAN_BEGIN(loop);
//   for (...) { ...; PROF_TICK("loop:stage_a"); ...; PROF_TICK("loop:stage_b"); }
//   PROF_SPAN_END(loop, "loop");   // whole loop, ticks above unaffected
// ---------------------------------------------------------------------------
#ifdef FLM_PROFILE

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace prof_detail {

// time_utils::duration_us truncates to whole microseconds, which silently eats
// up to 1 us per tick -- significant for the fine-grained per-stage ticks that
// fire thousands of times. Measure in ns and convert in floating point.
inline double elapsed_us(time_utils::time_point start, time_utils::time_point stop) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() / 1000.0;
}

// One accumulated stage: total microseconds and how many ticks contributed.
struct stage_stat {
    double total_us = 0.0;
    uint64_t count  = 0;
    uint64_t order  = 0;   // first-seen index, for stable report ordering
};

// Process-wide registry. Header-only + inline so every translation unit shares
// one instance without a .cpp. Not thread-safe: profile from the driver thread,
// not from inside an omp parallel region.
class registry {
public:
    static registry& instance() {
        static registry inst;
        return inst;
    }

    void add(const std::string& name, double us) {
        auto it = stats_.find(name);
        if (it == stats_.end()) {
            stage_stat s;
            s.total_us = us;
            s.count    = 1;
            s.order    = next_order_++;
            stats_.emplace(name, s);
        } else {
            it->second.total_us += us;
            it->second.count    += 1;
        }
    }

    void report() const {
        std::vector<const std::pair<const std::string, stage_stat>*> rows;
        rows.reserve(stats_.size());
        for (auto& kv : stats_) rows.push_back(&kv);
        std::sort(rows.begin(), rows.end(),
                  [](auto* a, auto* b) { return a->second.order < b->second.order; });

        // measure the widest name so every ':' lines up.
        size_t name_w = 0;
        for (auto* r : rows) name_w = std::max(name_w, r->first.size());

        auto fmt_time = [](double us) {
            time_utils::time_with_unit t =
                time_utils::re_unit(time_utils::time_with_unit((float)us, "us"));
            std::ostringstream o;
            o << std::fixed << std::setprecision(3) << std::setw(9) << std::right
              << t.first << ' ' << std::setw(2) << std::left << t.second;
            return o.str();
        };

        for (auto* r : rows) {
            const stage_stat& s = r->second;
            std::ostringstream line;
            line << std::left << std::setw((int)name_w) << r->first << "  "
                 << std::right << std::setw(4) << s.count << "x"
                 << "  total " << fmt_time(s.total_us)
                 << "  avg " << fmt_time(s.total_us / (double)s.count);
            header_print("prof", line.str());
        }
    }

    void reset() { stats_.clear(); next_order_ = 0; }

private:
    std::unordered_map<std::string, stage_stat> stats_;
    uint64_t next_order_ = 0;
};

}  // namespace prof_detail

#define PROF_INIT() time_utils::time_point __prof_t = time_utils::now()
#define PROF_TICK(name)                                                        \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        double __prof_us = prof_detail::elapsed_us(__prof_t, __prof_now);      \
        prof_detail::registry::instance().add((name), __prof_us);              \
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PROF_LAP() (__prof_t = time_utils::now())

// Span: an independent cursor for timing a region that itself contains ticks.
// Nesting a span around PROF_TICKs is safe -- the ticks never touch __prof_span_##tag.
#define PROF_SPAN_BEGIN(tag)                                                   \
    time_utils::time_point __prof_span_##tag = time_utils::now()
#define PROF_SPAN_END(tag, name)                                               \
    do {                                                                       \
        double __prof_us =                                                     \
            prof_detail::elapsed_us(__prof_span_##tag, time_utils::now());     \
        prof_detail::registry::instance().add((name), __prof_us);              \
    } while (0)

#define PROF_REPORT() prof_detail::registry::instance().report()
#define PROF_RESET() prof_detail::registry::instance().reset()

// Legacy accumulate-into-a-named-local API. Kept so existing call sites still
// compile; prefer PROF_TICK's built-in accumulation for new code.
#define PROF_ACC_DECL(var) float __prof_acc_##var = 0.0f
#define PROF_ACC(var)                                                          \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        __prof_acc_##var += (float)prof_detail::elapsed_us(__prof_t, __prof_now);\
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PROF_ACC_PRINT(name, var)                                              \
    do {                                                                       \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::time_with_unit(__prof_acc_##var, "us"));\
        header_print("prof", name << " (sum): " << __prof_d.first              \
                                  << " " << __prof_d.second);                  \
    } while (0)

#else
#define PROF_INIT() do {} while (0)
#define PROF_TICK(name) do {} while (0)
#define PROF_LAP() do {} while (0)
#define PROF_SPAN_BEGIN(tag) do {} while (0)
#define PROF_SPAN_END(tag, name) do {} while (0)
#define PROF_REPORT() do {} while (0)
#define PROF_RESET() do {} while (0)
#define PROF_ACC_DECL(var)
#define PROF_ACC(var) do {} while (0)
#define PROF_ACC_PRINT(name, var) do {} while (0)
#endif

#endif // __PROF_UTILS_HPP__
