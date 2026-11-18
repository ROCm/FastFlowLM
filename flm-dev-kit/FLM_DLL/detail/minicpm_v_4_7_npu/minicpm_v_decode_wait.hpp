// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
// Completion wait for a decode submission: nap in short slices until shortly before the command is due, then poll its state for a bounded time,
// then fall back to the ordinary blocking wait (which also reports errors/timeouts exactly as before).
//
// A blocking wait puts the CPU thread to sleep in the driver; the NPU's completion interrupt then has to wake a CPU that may have dropped into a
// deep C-state. Polling the command state (a memory read, no ioctl) near the expected end avoids that wake-up. The poll is bounded
// (max_spin_ms) and yields, so it cannot starve other work; nothing is pinned.
// Linux + XRT only: Windows timers can turn a 200 us sleep into ~15 ms, which would delay every token, and the HRX runlist has no state().
// FLM_MCV_NAPWAIT=<slice in us> sets the nap slice (default 200 on Linux); 0 = plain blocking wait.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include "npu_utils/flm_runtime.hpp"
#include "npu_utils/runtime_wait.hpp"

namespace minicpm_wait {

struct Policy {
#if defined(__linux__) && !defined(FLM_USE_HRX)
    int nap_us = 200;        // 0 = off (plain blocking wait)
#else
    int nap_us = 0;
#endif
    double lead_ms = 1.0;    // stop napping this long before the shortest recent duration
    double max_spin_ms = 2.0;
};
inline const Policy& policy() {
    static const Policy p = [] {
        Policy q;
#if defined(__linux__) && !defined(FLM_USE_HRX)
        if (const char* e = std::getenv("FLM_MCV_NAPWAIT")) q.nap_us = std::max(0, std::atoi(e));
#endif
        if (const char* e = std::getenv("FLM_MCV_NAPLEAD")) q.lead_ms = std::atof(e);
        if (const char* e = std::getenv("FLM_MCV_SPINMAX")) q.max_spin_ms = std::atof(e);
        return q;
    }();
    return p;
}

class Waiter {
    using clk = std::chrono::steady_clock;
    static constexpr int kHist = 16;
    double hist_[kHist];
    int n_ = 0;
    clk::time_point t0_;
public:
    void start() { t0_ = clk::now(); }
    // `cmd` is an xrt::run or xrt::runlist (anything with state() and wait())
    template <class Cmd>
    void wait(Cmd& cmd, std::string_view what = "minicpm_v command") {
#if defined(__linux__) && !defined(FLM_USE_HRX)
        const Policy& p = policy();
        if (p.nap_us > 0 && n_ >= 4) {
            const double best = *std::min_element(hist_, hist_ + std::min(n_, kHist));
            const double nap_until = best - p.lead_ms;
            auto el = [&] { return std::chrono::duration<double, std::milli>(clk::now() - t0_).count(); };
            while (el() < nap_until) std::this_thread::sleep_for(std::chrono::microseconds(p.nap_us));
            const double spin_end = std::max(el(), best) + p.max_spin_ms;
            for (;;) {
                const ert_cmd_state s = cmd.state();
                if (s != ERT_CMD_STATE_NEW && s != ERT_CMD_STATE_QUEUED && s != ERT_CMD_STATE_RUNNING && s != ERT_CMD_STATE_SUBMITTED) break;
                if (el() > spin_end) break;
                std::this_thread::yield();
            }
        }
#endif
        // wait() returns at once if the nap already observed a terminal state.
        // XRT runlist::wait() is void and throws on failure in current XRT.
        // HRX wait() returns the state and does not throw. Check either one.
        flm::npu::wait_for_completion(cmd, what);
        const double d = std::chrono::duration<double, std::milli>(clk::now() - t0_).count();
        hist_[n_ % kHist] = d;
        n_++;
    }
};

}  // namespace minicpm_wait
