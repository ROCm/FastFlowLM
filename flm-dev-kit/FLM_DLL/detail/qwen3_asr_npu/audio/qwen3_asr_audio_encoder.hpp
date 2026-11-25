// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_asr_audio_encoder.hpp
/// \brief Qwen3-ASR / Confucius4-R2T2 audio encoder, host (AVX-512 + OpenMP) reference implementation.
///
/// mel [frames][128] -> 3x Conv2d(3x3, s2, p1)+GELU -> conv_out -> +sinusoid pos -> 24 pre-LN layers with
/// 104-token window attention -> ln_post -> proj1+GELU -> proj2 -> [n_tokens][2048] bf16.
///
/// Source of truth: qwen_asr/core/transformers_backend/modeling_qwen3_asr.py (Qwen3ASRAudioEncoder.forward) with the
/// window mask applied (run/r2t2_patch.py: the stock HF code passes attention_mask=None, i.e. full attention).
///
/// Numerics follow the BF16 reference: every op output is rounded to bf16 (round-to-nearest-even), accumulation,
/// layer norm statistics, softmax and erf-GELU run in fp32. Activations are held as fp32 values that are exactly
/// bf16-representable; only the result of encode() is converted to bf16 bits.
///
/// This file has no NPU / XRT dependency so it can be built and tested standalone. The matrix products go to a
/// qwen3_asr_audio_gemm when one is given (the NPU in the engine); layer norm, attention and conv1 stay here.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/// bf16 bit pattern (same storage as biovault::bfloat16_t, kept as raw bits so the file stays self-contained)
typedef uint16_t asr_bf16_bits;

struct qwen3_asr_audio_config {
    int d_model = 1024;
    int heads = 16;
    int ffn_dim = 4096;
    int layers = 24;
    int num_mel_bins = 128;
    int n_window = 50;            ///< conv chunk = 2 * n_window = 100 mel frames
    int n_window_infer = 800;     ///< attention window = 800 mel frames = 8 chunks
    int downsample_hidden = 480;  ///< conv channels
    int output_dim = 2048;
    int max_source_positions = 1500;
    float ln_eps = 1e-5f;

    int chunk_frames() const { return 2 * n_window; }                       ///< 100
    int chunk_tokens() const { return 13; }                                   ///< tokens of a full chunk (3 stride-2 convs)
    int window_tokens() const { return chunk_tokens() * (n_window_infer / chunk_frames()); }  ///< 104
};

/// number of audio tokens for `frames` mel frames (= _get_feat_extract_output_lengths)
int qwen3_asr_audio_tokens(int frames);

/// Matrix products of the encoder on another device (the engine passes the NPU's, qwen3_asr_audio_gemm_npu.hpp):
/// C[M][N] = bf16(A[M][K] W[N][K]^T + bias), then bf16(erf GELU) when asked. A and C hold bf16 values in fp32; C is row-major.
/// The device may round the operands further (the NPU: block float, 8 values share an exponent).
class qwen3_asr_audio_gemm {
public:
    /// row r of A (K values): a pointer into the caller's data, or `tmp` (K floats) after filling it. Called from several threads.
    using rows_fn = std::function<const float*(int r, float* tmp)>;
    virtual ~qwen3_asr_audio_gemm() = default;
    /// W row-major [N][K] (bf16 values), bias [N] or nullptr; `exact_a` / `exact_w`: the device keeps A / W at bf16 precision
    /// (the NPU then also runs the product on the rest of A rounded to its format / on the rest of W). Returns the id run()
    /// takes. Called while loading only.
    virtual int add_weight(const float* W, int N, int K, const float* bias, bool exact_a, bool exact_w) = 0;
    virtual void run(int weight, const rows_fn& A, int M, float* C, bool gelu) = 0;
    /// Conv2d(3x3, stride 2, pad 1) of `count` NHWC images in[count][H][W][cin]: A = their im2col rows, (image, ho, wo) order, K
    /// values each in (kh, kw, cin) order (K >= 9 cin, zero past 9 cin), as the weight was added. The default builds the rows.
    virtual void run_conv(int weight, const float* in, int count, int H, int W, int cin, int K, float* C, bool gelu);
};

class qwen3_asr_audio_encoder {
public:
    /// \param audio_weight_q4nx path to audio_weight.q4nx (BF16, matrices in the audio MM tile layout)
    /// \param gemm runs the projections, conv2, conv3 and conv_out when given (the host keeps no copy of those weights)
    qwen3_asr_audio_encoder(const std::string& audio_weight_q4nx, const qwen3_asr_audio_config& cfg = {},
                            qwen3_asr_audio_gemm* gemm = nullptr);
    ~qwen3_asr_audio_encoder();

    /// \param mel row-major [frames][num_mel_bins] fp32 (the feature extractor output, time-major)
    /// \return [qwen3_asr_audio_tokens(frames)][output_dim] bf16 bits, row-major
    /// Streaming: the conv front-end output (position embedding added) of the complete 100-frame chunks of a growing recording. The
    /// front end of a chunk sees only that chunk's mel, so its output is reused while that mel is the same (it is not always: the
    /// feature extractor clamps at the clip's max - 8, which moves when louder audio arrives; then that chunk and the rest are redone).
    struct conv_cache {
        std::vector<float> rows;   ///< [chunks * chunk_tokens][d_model]
        std::vector<float> mel;    ///< [chunks * chunk_frames][num_mel_bins] the input of those rows
        int chunks = 0;
        /// keep only the first n chunks
        void truncate(int n, int row_floats, int mel_floats) {
            if (n >= chunks) return;
            rows.resize((size_t)n * row_floats);
            mel.resize((size_t)n * mel_floats);
            chunks = n;
        }
        /// drop the first n chunks (the caller's first frame moved forward by n chunks)
        void drop_front(int n, int row_floats, int mel_floats) {
            n = n < chunks ? n : chunks;
            rows.erase(rows.begin(), rows.begin() + (size_t)n * row_floats);
            mel.erase(mel.begin(), mel.begin() + (size_t)n * mel_floats);
            chunks -= n;
        }
    };
    std::vector<asr_bf16_bits> encode(const float* mel, int frames, conv_cache* cache = nullptr);

    // ------------------------------------------------------------------------------------------------------------
    // stage seams (all activations are [rows][cols] row-major fp32 holding bf16 values)
    // ------------------------------------------------------------------------------------------------------------
    /// One 100-frame chunk through conv1..3 and conv_out. `mel_chunk` is [valid_frames][128]; missing frames up to 100 are
    /// zero (the reference pads the tail chunk to 100 and masks afterwards, so the conv sees zeros, not "end of input").
    /// \return [valid_tokens][d_model] (conv_out output, before the position embedding)
    void conv_front(const float* mel_chunk, int valid_frames, float* out_tokens);
    /// x[t] += pos[t % 13] for the tokens of one chunk (positions restart at 0 in every chunk)
    void add_pos(float* chunk_tokens, int tokens);
    /// One encoder layer over all tokens; attention is confined to windows of window_tokens().
    void encoder_layer(int layer, float* x, int n_tokens);
    /// ln_post, proj1+GELU, proj2 -> [n_tokens][output_dim] fp32 (bf16 values)
    void post(const float* x, int n_tokens, float* out);

    const qwen3_asr_audio_config& config() const { return cfg_; }

private:
    struct Impl;
    Impl* impl_;
    qwen3_asr_audio_config cfg_;
};
