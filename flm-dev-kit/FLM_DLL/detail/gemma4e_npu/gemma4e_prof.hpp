// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __GEMMA4E_PROF_HPP__
#define __GEMMA4E_PROF_HPP__
// Stage timers for the gemma4e prefill. Compiled in only with -DGEMMA4E_PROF; otherwise every macro is empty.
//
//   PROF_BEGIN();        starts a stopwatch in the current scope
//   PROF_TICK("name");   adds the time since the previous tick (or begin) to the row "name"
//   PROF_SKIP();         restarts the stopwatch without recording (the time was counted by finer ticks)
//   PROF_REPORT();       prints the table of one prefill call and clears it
//
// Rows are summed over all layers of the call, and the group ("full" or "tail") is part of the row name.
#ifdef GEMMA4E_PROF
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace gemma4e_prof {
    inline std::map<std::string, double>& rows() { static std::map<std::string, double> t; return t; }
    inline const char*& group() { static const char* g = "full"; return g; }
    inline std::chrono::steady_clock::time_point now() { return std::chrono::steady_clock::now(); }
    inline void add(const char* name, double seconds) { rows()[std::string(group()) + "/" + name] += seconds; }
    inline void report(double wall) {
        std::vector<std::pair<std::string, double>> sorted(rows().begin(), rows().end());
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
        double sum = 0;
        for (auto& r : sorted) sum += r.second;
        std::fprintf(stderr, "[prefill-prof] wall %.3f s, timed %.3f s\n", wall, sum);
        for (auto& r : sorted) std::fprintf(stderr, "[prefill-prof]   %-24s %8.3f s %5.1f%%\n", r.first.c_str(), r.second, 100.0 * r.second / wall);
        rows().clear();
    }
}
#define PROF_BEGIN() auto prof_last__ = gemma4e_prof::now()
#define PROF_TICK(name) do { auto prof_now__ = gemma4e_prof::now(); \
    gemma4e_prof::add(name, std::chrono::duration<double>(prof_now__ - prof_last__).count()); prof_last__ = prof_now__; } while (0)
#define PROF_SKIP() (prof_last__ = gemma4e_prof::now())  /* drop time already counted by finer ticks */
#define PROF_GROUP(g) (gemma4e_prof::group() = (g))
#define PROF_REPORT(wall) gemma4e_prof::report(wall)
#else
#define PROF_BEGIN() do {} while (0)
#define PROF_TICK(name) do {} while (0)
#define PROF_SKIP() do {} while (0)
#define PROF_GROUP(g) do {} while (0)
#define PROF_REPORT(wall) do {} while (0)
#endif
#endif
