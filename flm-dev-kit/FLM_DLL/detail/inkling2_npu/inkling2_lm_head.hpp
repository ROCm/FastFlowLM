// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_lm_head.hpp
/// \brief lm_head on the NPU (lm_head.xclbin, FLM_Xclbin/Inkling2/lm_head): one matrix-vector product over the whole
///        array, 8 columns x 4 cores, two shim weight streams per column
/// \note The Q8_0 rows are padded to whole rounds of 1024 (ROWS, zero blocks) and laid out per round, per column, as
///       the two streams' halves back to back (see reorder_cpy_lm_head of the Qwen3.5 engine). x is the final-normed
///       token (bf16; the kernel's norm is off); y comes back bf16, the soft cap runs on the host.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "inkling2_host.hpp"
#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

// The decode and lm_head programs load control-packet switches as ELF modules (xrt::elf / xrt::module / xrt::ext::kernel), which the
// HRX shim does not provide (as d3's switches): on HRX decode and lm_head stay on the host.
#if !defined(FLM_USE_HRX)
class inkling2_lm_head {
public:
    static constexpr int M_PER_ROUND = 1024;
    static constexpr int ROWS = (inkling2_host::VOCAB_PAD + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;   // 201728

    inkling2_lm_head(npu_xclbin_manager* npu, const std::string& xclbin, const inkling2_host::weights& w);
    /// \brief a guest in another app's hardware context (control-packet switches, see inkling2_launch.hpp): runs made
    ///        by `mgr` (the base xclbin's manager), x read at byte offset x_off of `xsrc` (2 D bf16; the second half
    ///        is not used without the kernel's norm)
    inkling2_lm_head(npu_xclbin_manager* npu, npu_app_manager* mgr, const inkling2_host::weights& w,
                     buffer<uint8_t>* xsrc, size_t x_off);
    /// \brief a guest's run for a window (not started)
    xrt::run guest_run() { return this->app.create_run(this->y, this->wb, *this->xsrc); }
    /// \brief after a guest's run completed: logits[VOCAB] = softcap(y)
    void finish(float* logits);
    /// \brief logits[VOCAB] = softcap(lm_head . x) of a final-normed token x (bf16[D]); false if the run did not
    ///        complete (the device is suspect then)
    bool run(const uint16_t* x, float* logits);
    /// \brief the raw products (bf16[ROWS]) of the last run
    const uint16_t* y_out() const { return this->y.data(); }
    double t_run = 0, t_post = 0;   // ms: start to completion and y sync; soft cap
    double t_dev = 0, t_sync = 0;   // of t_run: start to completion; y sync
    double t_fin = 0;               // finish: y sync and soft cap
    int n_run = 0;
    // rounds per shim descriptor (INKLING2_LMH_RPB): 1 issues every round's transfers from the sequence
    int rpb = ROWS / M_PER_ROUND;

private:
    void init(npu_xclbin_manager* npu, const inkling2_host::weights& w);
    void gen_seq();
    buffer<uint8_t>* xsrc = nullptr;
    size_t x_off = 0;
    npu_app_manager* mgr = nullptr;
    npu_app app;
    xrt::run run_;
    buffer<uint16_t> y, x;
    buffer<uint8_t> wb;
};
#endif   // !FLM_USE_HRX
