// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_audio_gemm_npu.hpp
/// \brief The audio encoder's matrix products on the NPU (audio_gemm.xclbin, built from FLM_Xclbin/bfp16_gemm).
///
/// Weights are packed to bfp16 once, at load, one BO each (K padded to 128, or 512 when the weights stream; N padded
/// to 1024). A product packs A into the X BO, runs as many calls as its rows need (a call takes at most 31 rounds of
/// 256 rows, and 256 / KC rounds when the weights stream) and unpacks the bf16 result. The bias is added on the cores;
/// the erf GELU runs on the host while unpacking (the cores' tanh form costs too much accuracy). K is not cut into
/// parts added on the host: the conv stack's partial sums over kernel positions nearly cancel on quiet input, and
/// rounding each to bf16 costs accuracy there.
///
/// bfp16 keeps about 7 bits of an activation, less for values well below the largest of their block of 8. For a product
/// added with exact_a, the rest of A (scaled up) is packed as more rows of the same call and the host adds the two
/// results: A then keeps its bf16 precision. For a weight added with exact_w, the rest of W (scaled up) is packed as N
/// more output columns, so the same call also returns A times the rest; where N is at most 512 those columns are padding
/// anyway (conv2 and conv3). Both keep the encoder as close to its bf16 reference as the host path; without them the
/// conv stack drifts on quiet input, where its output is a small signal over the bias.
/// Calls are serialized: the engine encodes in the foreground and in the background.
#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "qwen3_asr_audio_encoder.hpp"

class qwen3_asr_audio_gemm_npu : public qwen3_asr_audio_gemm {
public:
    /// \param mgr the app manager of audio_gemm.xclbin
    explicit qwen3_asr_audio_gemm_npu(npu_app_manager* mgr);
    int add_weight(const float* W, int N, int K, const float* bias, bool exact_a, bool exact_w) override;
    void run(int weight, const rows_fn& A, int M, float* C, bool gelu) override;
    /// encodes each input pixel once (cin values = cin / 8 blocks, so a block never spans two pixels), then gathers the blocks
    /// of each im2col row: the same bytes as run() on the rows
    void run_conv(int weight, const float* in, int count, int H, int W, int cin, int K, float* C, bool gelu) override;

private:
    struct weight_t {
        buffer<u8> w;   // [W; rest of W] when split_w
        int N = 0, K = 0, n_pad = 0, k_pad = 0;
        bool split = false, split_w = false;
        std::vector<float> bias;   // bf16 values, what the cores add
    };
    using shape_t = std::tuple<uint32_t, uint32_t, uint32_t>;   // rounds, kc, nb
    npu_app& app_for(uint32_t rounds, uint32_t kc, uint32_t nb);
    void grow(size_t x_bytes, size_t y_elems, int k);
    /// packs rows [r0, r0 + rows) of A into hi (and their rest into lo when lo is given), KC chunks of 128 k
    using pack_fn = std::function<void(uint8_t* hi, uint8_t* lo, int r0, int rows, int KC)>;
    void run_packed(weight_t& w, int M, float* C, bool gelu, const pack_fn& pack);

    npu_app_manager* mgr_;
    std::vector<std::unique_ptr<weight_t>> weights_;   // never moved: a copied XRT buffer does not own its BO
    std::map<shape_t, std::unique_ptr<npu_app>> apps_;
    npu_app io_app_;   // owns no sequence: creates the X and Y buffers
    buffer<u8> x_;    // A rounded to bfp16, then its rest (split weights)
    buffer<u16> y_;
    std::vector<float> tmp_;   // 16 rows of A per thread
    std::vector<uint8_t> px_hi_, px_lo_;   // run_conv: the input pixels as blocks, [pixel][cin / 8][9 B]
    std::mutex mu_;
};
