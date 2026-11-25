// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_prefill_npu.hpp
/// \brief The prompt prefill's matrix products on the NPU (prefill.xclbin, built from FLM_Xclbin/dequant_mm_mix_bfp16).
///
/// The kernel reads the decode weights in place: per layer one buffer of Q4_1 (k, q, o, up, gate) and Q8_0 (v, down)
/// units, K|V|Q, o, up|gate (interleaved 512 rows at a time), down (qwen3_asr_desc). A layer is four calls: K, V and Q in
/// one, o, up and gate in one, down. The host packs X as block float (8 values share an exponent, rounded to nearest),
/// unpacks the bf16 results, and computes silu(gate) * up between the last two calls. The CPU decoder keeps the rest of the
/// layer (norms, RoPE, attention, the KV cache), so the streaming and background prefill paths are unchanged.
/// Calls are serialized: the engine prefills in the foreground and in the background.
#pragma once
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "../decoder/qwen3_asr_cpu_decoder.hpp"
#include "qwen3_asr_prefill_seq.hpp"

class qwen3_asr_prefill_npu : public qwen3_asr_cpu::projections {
public:
    struct layout {
        int D = 0, DQ = 0, DK = 0, I = 0;      // hidden, q heads x head_dim, kv heads x head_dim, intermediate
        size_t k = 0, v = 0, q = 0, o = 0, up = 0, down = 0;   // byte offsets of the weights in a layer's buffer
    };
    /// \param mgr the app manager of prefill.xclbin
    /// \param weights the decode layer's weight buffers, one per layer (read in place)
    qwen3_asr_prefill_npu(npu_app_manager* mgr, const layout& l, std::vector<buffer<u8>>& weights);
    int min_rows() const override { return min_rows_; }
    void qkv(int layer, const float* x, int n, float* q, float* k, float* v) override;
    void o(int layer, const float* att, int n, float* out) override;
    void mlp(int layer, const float* x, int n, float* out) override;

private:
    enum call_id { KVQ, O, UP_GATE, DOWN, CALLS };
    struct call_t {
        uint32_t din = 0, dout = 0;
        std::vector<qwen3_asr_prefill_seq::mix_band> bands;
    };
    /// one NPU call: rows [r0, r0 + rows) of the prompt, input x [rows][din]; unpack reads their bf16 Y [rows][dout]
    struct job {
        call_id c;
        int r0, rows;
        const float* x;
        std::function<void(const uint16_t* y, int r0, int rows)> unpack;
        bool needs_prev = false;   // x is written by the unpack of the job just before
    };
    npu_app& app_for(call_id c, uint32_t rows_padded);
    /// the jobs of `layer` in order, in two buffer slots: while the NPU runs job j the host packs job j + 1 and unpacks job j - 1
    /// (a job's x must be written by the unpack of a job at least two earlier, or of the job just before with needs_prev)
    void run(int layer, const std::vector<job>& jobs);
    /// rows per job of an n-row prompt: the kernel reads the weights once per 512 rows anyway, so 512-row jobs cost no more NPU
    /// time than one, and the host packs and unpacks them while the NPU runs the next
    static int step(int n) { return std::min(n, 512); }
    /// QWEN3_ASR_PROF: print the sums of a prompt of n rows and reset them
    void report(int n);

    npu_app_manager* mgr_;
    layout l_;
    std::vector<buffer<u8>>& weights_;
    call_t calls_[CALLS];
    int min_rows_ = 4;
    std::map<std::pair<int, uint32_t>, std::unique_ptr<npu_app>> apps_;
    npu_app io_app_;   // owns no sequence: creates the X and Y buffers
    buffer<u8> x_[2];
    buffer<u16> y_[2];
    std::vector<float> act_;   // silu(gate) * up, [n][I]
    std::mutex mu_;
    // QWEN3_ASR_PROF: per call, ms of packing, NPU and unpacking, and its multiply-adds, summed over the layers
    bool prof_ = false;
    double prof_ms_[CALLS][3] = {}, prof_flop_[CALLS] = {};
};
