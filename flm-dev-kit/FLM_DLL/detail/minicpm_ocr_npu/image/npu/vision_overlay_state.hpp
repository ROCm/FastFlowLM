// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_overlay_state.hpp
/// \brief Which op vision_overlay.xclbin is set up for right now, and the run-again-on-failure rule.
///
/// Shared by every op that runs on the overlay (design-A matmuls and the attention), owned by
/// the factory. Each run picks its prefix from it and updates it; a run on another xclbin
/// (design B) or a failed run resets it, so the next overlay run switches fully.
/// See FLM_Xclbin/minicpm_ocr/vision_overlay/README.md and ../README.md, "Run failures".
#ifndef __MINICPM_OCR_NPU_VISION_OVERLAY_STATE_HPP__
#define __MINICPM_OCR_NPU_VISION_OVERLAY_STATE_HPP__

#include "image/npu/vision_overlay.hpp"

#include "npu_utils/npu_utils.hpp"
#include "utils/debug_utils.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <set>
#include <sstream>
#include <string>
#include <thread>

namespace mco_npu {

struct overlay_state {
    int current = vision_overlay::OP_NONE;

    overlay_state() {
        _read_run_list("FLM_MINICPM_OCR_VISION_FAKE_FAIL_AT", fake_fail_at_);
        _read_run_list("FLM_MINICPM_OCR_VISION_IDLE_BEFORE", idle_before_);
    }

    /// `first` is the run with its usual prefix. If it does not complete, the driver ended it and
    /// reset the array, so the host-written setup is gone: forget it and run `again` (the same op
    /// from a full switch). Right after a timeout the driver may still be restarting the context
    /// and the next run fails at once, so a fast failure is tried again after a pause; a second
    /// timeout is not. Returns the state of the last run; the caller throws on failure.
    template <typename First, typename Again>
    ert_cmd_state run_checked(const std::string& what, First first, Again again) {
        runs_++;
        if (idle_before_.count(runs_)) {
            header_print("WARNING", "FLM_MINICPM_OCR_VISION_IDLE_BEFORE: 7 s idle before overlay run " << runs_
                                    << " (" << what << ")");
            std::this_thread::sleep_for(std::chrono::seconds(7));
        }
        ert_cmd_state st = _guarded(what, first);
        if (fake_fail_at_.count(runs_)) {
            header_print("WARNING", "FLM_MINICPM_OCR_VISION_FAKE_FAIL_AT: overlay run " << runs_
                                    << " (" << what << ") treated as failed");
            st = ERT_CMD_STATE_TIMEOUT;
        }
        if (st == ERT_CMD_STATE_COMPLETED) return st;

        current = vision_overlay::OP_NONE;
        retries_++;
        static constexpr int PAUSE_MS[] = {0, 100, 1000};
        for (int pause_ms : PAUSE_MS) {
            if (st == ERT_CMD_STATE_TIMEOUT && pause_ms > 0) break;
            header_print("WARNING", "vision overlay run " << what << " ended with state " << (int)st
                                    << ", running it again from a full switch after " << pause_ms << " ms");
            std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
            st = _guarded(what, again);
            if (st == ERT_CMD_STATE_COMPLETED) break;
        }
        return st;
    }

    long runs() const { return runs_; }
    long retries() const { return retries_; }

private:
    template <typename Run>
    static ert_cmd_state _guarded(const std::string& what, Run run) {
        try {
            return run();
        } catch (const std::exception& e) {
            header_print("WARNING", "vision overlay run " << what << " threw: " << e.what());
            return ERT_CMD_STATE_ERROR;
        }
    }

    /// Test hooks: a comma-separated list of overlay run numbers (1 = the first run).
    static void _read_run_list(const char* name, std::set<long>& runs) {
        const char* env = std::getenv(name);
        if (!env) return;
        std::istringstream in(env);
        std::string item;
        while (std::getline(in, item, ',')) {
            if (!item.empty()) runs.insert(std::atol(item.c_str()));
        }
    }

    long           runs_    = 0;
    long           retries_ = 0;
    std::set<long> fake_fail_at_;
    std::set<long> idle_before_;
};

} // namespace mco_npu

#endif  // __MINICPM_OCR_NPU_VISION_OVERLAY_STATE_HPP__
