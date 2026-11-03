#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct vision_timer_t {
    using clock = std::chrono::steady_clock;

    struct stage_t {
        std::string name;
        std::vector<double> samples_us;
    };

    std::vector<stage_t> stages;
    clock::time_point last;
    clock::time_point begin;

    static bool probe_enabled() {
        const char* value = std::getenv("VISION_PROBE");
        return value != nullptr && value[0] == '1';
    }

    void start() {
        stages.clear();
        begin = clock::now();
        last = begin;
    }

    void mark(const char* name) {
        clock::time_point now = clock::now();
        double us = std::chrono::duration<double, std::micro>(now - last).count();
        last = now;
        stage_t* stage = find_or_add(name);
        stage->samples_us.push_back(us);
    }

    stage_t* find_or_add(const char* name) {
        for (auto& stage : stages) {
            if (stage.name == name) {
                return &stage;
            }
        }
        stages.push_back({name, {}});
        return &stages.back();
    }

    static double median(std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    void print(int seq_len, int seq_len_padded) {
        double total_us = std::chrono::duration<double, std::micro>(clock::now() - begin).count();
        std::fprintf(stderr, "VISION L %d (padded %d) probe %d: encode %.2f ms\n",
                     seq_len, seq_len_padded, probe_enabled() ? 1 : 0, total_us / 1000.0);
        std::fprintf(stderr, "VISION %-16s %6s %10s %10s %10s %10s\n", "stage", "calls", "total_ms", "median_us", "min_us", "max_us");
        for (const auto& stage : stages) {
            double sum = 0;
            for (double v : stage.samples_us) {
                sum += v;
            }
            double lo = *std::min_element(stage.samples_us.begin(), stage.samples_us.end());
            double hi = *std::max_element(stage.samples_us.begin(), stage.samples_us.end());
            std::fprintf(stderr, "VISION %-16s %6zu %10.2f %10.1f %10.1f %10.1f\n",
                         stage.name.c_str(), stage.samples_us.size(), sum / 1000.0,
                         median(stage.samples_us), lo, hi);
        }
    }
};
