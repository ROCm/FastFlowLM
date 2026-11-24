// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_decode.hpp
/// \brief a decode token's routed experts on the NPU (decode.xclbin, FLM_Xclbin/Inkling2/decode), beside the host
/// \note The overlay streams Q4_K weights from DDR through 16 shim channels (shim MM2S 0 and 1 of every column) into
///       16 GEMV cores (decode_core.cc). A channel's work is a header task (opcode, groups, column blocks, then x as
///       bf16) followed by weight tasks of whole block groups (96 rows x nc column blocks, the grouped layout the
///       prefill buffers hold); the core writes 96 fp32 outputs per group. A dispatch of n experts (n <= 4) spreads
///       their n x 24 groups over the channels in (expert, group) order; the experts are sub-buffers of the layer
///       buffer bound per call (arguments 1 .. 4), so the instruction sequence depends on n only.
///       The NPU streams ~3 experts in about the time the host takes for the other 5, so a token's experts are split.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

class inkling2_decode {
public:
    static constexpr int MAX_N = 4;   // experts per dispatch (weight arguments)

    /// \param xclbin path of decode.xclbin
    inkling2_decode(npu_xclbin_manager* npu, const std::string& xclbin);

    /// \brief start experts ex[0 .. n) of a layer: kind 0 gate_up of x [D] (w at the layer's gate_up), kind 1 down of
    ///        a [n][FF_PAD] (w at the layer's down)
    void start(int kind, bytes& w, size_t base, const int* ex, int n, const float* in);
    /// \brief wait for start(): kind 0 out [n][2 FF] (gate rows, then up rows), kind 1 out [n][D]
    void wait(float* out);

    // DIAG: seconds in start / wait and calls
    double t_start = 0, t_wait = 0;
    long calls = 0;

private:
    struct seg {
        int slot, g0, n;
    };
    struct plan {
        npu_app app;
        std::vector<std::vector<seg>> segs;   // per channel
        std::vector<size_t> ybase;            // per channel, bytes into io
        size_t io_bytes = 0;
    };
    npu_app_manager* mgr;
    std::map<int, plan> plans;   // kind * 16 + n
    buffer<uint8_t> io[2];       // per kind: headers, then y
    xrt::run run;
    int kind = -1, n = 0;
    plan& get(int kind, int n);
};
