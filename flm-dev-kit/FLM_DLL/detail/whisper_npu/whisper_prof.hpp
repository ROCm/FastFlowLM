// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
// Optional per-stage timer for the Whisper encoder. Off unless WHISPER_PROFILE=1.
// PROF_LAP("name") adds the time since the previous lap to "name"; the table is
// printed to stderr once per encode (PROF_REPORT).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

struct WhisperProf {
    bool on = std::getenv("WHISPER_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point last;
    std::vector<std::pair<std::string, double>> stages; // insertion order

    void start() {
        if (!on) return;
        stages.clear();
        last = std::chrono::steady_clock::now();
    }
    void lap(const char* name) {
        if (!on) return;
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        for (auto& s : stages) {
            if (s.first == name) { s.second += ms; return; }
        }
        stages.emplace_back(name, ms);
    }
    void report(const char* tag) {
        if (!on) return;
        double total = 0;
        for (auto& s : stages) total += s.second;
        std::fprintf(stderr, "[whisper-prof] %s total %.1f ms |", tag, total);
        for (auto& s : stages) std::fprintf(stderr, " %s=%.1f", s.first.c_str(), s.second);
        std::fprintf(stderr, "\n");
    }
};
inline WhisperProf g_prof;
#define PROF_START() g_prof.start()
#define PROF_LAP(name) g_prof.lap(name)
#define PROF_REPORT(tag) g_prof.report(tag)
