// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file conv2d.hpp
/// \brief AVX-512 BF16 audio patch-embedding: 4× Conv2d(3×3, stride=2, pad=1) + GELU.
///
/// All four conv layers share the same geometry:
///   kernel_size = 3×3,  stride = 2×2,  padding = 1×1 (zero, symmetric),
///   dilation = 1×1,  groups = 1
///   H_out = (H_in + 1) / 2,  W_out = (W_in + 1) / 2  (ceiling division)
///
/// Per-layer tensor layouts
/// ────────────────────────
///   Weight (after one-time load-time pre-transposition):
///     PyTorch original : [C_out, C_in, 3, 3]
///     Stored as        : [C_in, 9, C_out]   ← C_out is the fast/contiguous dim
///     Reason: the innermost AXPY loop strides over C_out=480 elements,
///     so making C_out contiguous gives 30 back-to-back AVX-512 FMA instructions.
///
///   Per-layer I/O (NCHW, one block at a time):
///     input  : [C_in,  H_in,  W_in]  row-major
///     output : [C_out, H_out, W_out] row-major
///     bias   : [C_out] (f32, optional)
///
/// Concrete dimensions (C = downsample_hidden_size = 480, chunk_size = 100):
///   layer   C_in  C_out  H_in  W_in   H_out  W_out
///   -----   ----  -----  ----  ----   -----  -----
///   conv1   1     480    128   100    64     50
///   conv2   480   480    64    50     32     25
///   conv3   480   480    32    25     16     13
///   conv4   480   480    16    13     8      7
///     (H_in for conv1 = num_mel_bins; W_in for conv1 = chunk_size)
///
/// Final output of audio_conv_downsample
/// ──────────────────────────────────────
///   out4: [num_blocks, W4, C*H4]  i.e. [b, t, c*f]
///
///   The NCHW result of conv4 ([C, H4, W4]) is transposed in-place so that
///   time steps become the token axis.  This matches Python's padded_embed_permuted:
///     padded_embed.permute(0,3,1,2).contiguous().view(b, t, c*f)
///   where b=num_blocks, t=W4, c*f=C*H4.
///   No further permutation is needed before feeding into conv_out (nn.Linear).
#pragma once
#include <immintrin.h>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include "typedef.hpp"
#include "../avx512_util.hpp"

// ---------------------------------------------------------------------------
// conv_weight_pretranspose
//
//  Rearrange weight from PyTorch layout [C_out, C_in, 3, 3]
//  to cache-friendly layout [C_in, 3, 3, C_out] (C_out contiguous).
//
//  Call once at model load time.  The transposed buffer must be pre-allocated
//  to C_out * C_in * 9 elements.
// ---------------------------------------------------------------------------
inline void conv_weight_pretranspose(
    const bf16* src,   // [C_out, C_in, 9]  PyTorch layout
    bf16*       dst,   // [C_in, 9, C_out]  transposed
    int C_in, int C_out)
{
    // src[co, ci, k] = src[co * C_in*9 + ci * 9 + k]
    // dst[ci, k, co] = dst[(ci*9 + k) * C_out + co]
    for (int co = 0; co < C_out; ++co) {
        const bf16* src_co = src + (size_t)co * C_in * 9;
        for (int ci = 0; ci < C_in; ++ci) {
            const bf16* src_ci = src_co + ci * 9;
            for (int k = 0; k < 9; ++k) {
                dst[(size_t)(ci * 9 + k) * C_out + co] = src_ci[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// conv2d_3x3_s2p1_bf16_fast
//
//  Single-layer Conv2d(3×3, stride=2, pad=1) in BF16 with optional fused GELU.
//
//  Inputs
//    weight_T : [C_in, 9, C_out]  pre-transposed (C_out contiguous, see file header)
//    bias     : [C_out] as float32, or nullptr
//    input    : [C_in, H_in, W_in]
//    output   : [C_out, H_out, W_out]  (NCHW, caller-allocated)
//    acc      : [W_out * C_out] float32 scratch, caller-owned (enables per-thread reuse)
//
//  Loop order (oh parallelised by caller via omp for):
//    for oh in [oh_begin, oh_end):           ← one output row per OMP iteration
//      acc[W_out][C_out] = bias broadcast    ← f32 accumulator; fits in L2
//      for ow, ci, ky, kx:                  ← accumulate 3×3 kernel contributions
//        acc[ow][co] += input[ci,ih,iw] * weight_T[ci, ky*3+kx, co]   (AXPY)
//      flush: output[co, oh, ow] = GELU(acc[ow][co])  for all co, ow
//
//  The innermost AXPY over C_out=480 is 30 back-to-back AVX-512 FMA instructions
//  (C_out must be a multiple of 16 for full vectorisation; scalar tail otherwise).
//
//  Accumulator L2 footprint per output row:
//    conv1: 50  × 480 × 4B = 96 KB
//    conv2: 25  × 480 × 4B = 48 KB
//    conv3: 13  × 480 × 4B = 25 KB
//    conv4:  7  × 480 × 4B = 13 KB   (all fit in a typical 256–512 KB L2)
// ---------------------------------------------------------------------------

// Core kernel — processes a subset of output rows [oh_begin, oh_end).
// Caller supplies acc scratch [W_out * C_out] floats (no alloc, per-thread).
inline void conv2d_3x3_s2p1_bf16_fast(
    const bf16* __restrict input,    // [C_in, H_in, W_in]
    const bf16* __restrict weight_T, // [C_in, 9, C_out]  pre-transposed
    const float* __restrict bias,    // [C_out] as float32, or nullptr
    bf16*        __restrict output,  // [C_out, H_out, W_out]
    float*       __restrict acc,     // [W_out * C_out] scratch, caller-owned
    int C_in, int C_out,
    int H_in, int W_in,
    int oh_begin, int oh_end,          // row range [oh_begin, oh_end)
    bool apply_gelu = true)
{
    const int H_out = (H_in + 1) / 2;
    const int W_out = (W_in + 1) / 2;

    // Number of full AVX-512 lanes covering C_out (C_out must be multiple of 16
    // for full vectorisation; tail handled scalar).
    const int C_vec = (C_out / 16) * 16;

    for (int oh = oh_begin; oh < oh_end; ++oh) {
        const int ih_base = oh * 2 - 1;

        // --- Initialise accumulator with bias ---
        if (bias) {
            // Broadcast bias[co] into every acc[ow, co] column.
            for (int ow = 0; ow < W_out; ++ow) {
                float* row = acc + (size_t)ow * C_out;
                int co = 0;
                for (; co <= C_vec - 16; co += 16)
                    _mm512_storeu_ps(row + co, _mm512_loadu_ps(bias + co));
                for (; co < C_out; ++co)
                    row[co] = bias[co];
            }
        } else {
            memset(acc, 0, (size_t)W_out * C_out * sizeof(float));
        }

        // --- Accumulate kernel contributions ---
        for (int ow = 0; ow < W_out; ++ow) {
            float* acc_ow = acc + (size_t)ow * C_out;
            const int iw_mid = ow * 2;

            for (int ci = 0; ci < C_in; ++ci) {
                const bf16* in_ci = input + (size_t)ci * H_in * W_in;

                for (int ky = 0; ky < 3; ++ky) {
                    const int ih = ih_base + ky;
                    if (ih < 0 || ih >= H_in) continue;
                    const bf16* in_row = in_ci + (size_t)ih * W_in;

                    // weight row for (ci, ky*3+kx): pointer into [C_in,9,C_out]
                    // weight_T[(ci*9 + ky*3 + kx) * C_out + co]
                    const int k_base = (ci * 9 + ky * 3) * C_out;

                    for (int kx = 0; kx < 3; ++kx) {
                        const int iw = iw_mid + kx - 1;
                        if (iw < 0 || iw >= W_in) continue;

                        const float in_val = bf16_to_f32(in_row + iw);
                        const __m512 v_in  = _mm512_set1_ps(in_val);
                        const bf16*  w_row = weight_T + k_base + kx * C_out;

                        // Prefetch next weight row (one step ahead in kx/ky/ci)
                        _mm_prefetch((const char*)(w_row + C_out), _MM_HINT_T1);

                        // AXPY: acc_ow[co] += in_val * w_row[co]
                        int co = 0;
                        for (; co <= C_vec - 16; co += 16) {
                            __m512 vacc = _mm512_loadu_ps(acc_ow + co);
                            __m512 vw   = load_bfloat16_to_m512(w_row + co);
                            vacc = _mm512_fmadd_ps(v_in, vw, vacc);
                            _mm512_storeu_ps(acc_ow + co, vacc);
                        }
                        for (; co < C_out; ++co)
                            acc_ow[co] += in_val * bf16_to_f32(w_row + co);
                    }
                }
            }
        }

        // --- Flush accumulator → output [C_out, oh, W_out], optionally GELU ---
        // output[co, oh, ow] = acc[ow, co]
        // We write one output channel row at a time (contiguous in output).
        for (int co = 0; co < C_out; ++co) {
            bf16* out_row = output + ((size_t)co * H_out + oh) * W_out;
            int ow = 0;
            if (apply_gelu) {
                for (; ow <= W_out - 16; ow += 16) {
                    // Gather acc[ow..ow+15][co] — stride C_out, not contiguous.
                    // Scalar loop is cleaner here; W_out is small (<=50).
                    alignas(64) float tmp[16];
                    for (int j = 0; j < 16; ++j) tmp[j] = acc[(size_t)(ow + j) * C_out + co];
                    __m512 v = _mm512_load_ps(tmp);
                    v = gelu_tanh_avx512_simd(v);
                    store_m512_to_bfloat16(out_row + ow, v);
                }
                for (; ow < W_out; ++ow) {
                    // scalar tanh-based GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
                    float x = acc[(size_t)ow * C_out + co];
                    float x3 = x * x * x;
                    float inner = 0.7978845608f * (x + 0.044715f * x3);
                    f32_to_bf16(out_row + ow, 0.5f * x * (1.0f + tanhf(inner)));
                }
            } else {
                for (; ow <= W_out - 16; ow += 16) {
                    alignas(64) float tmp[16];
                    for (int j = 0; j < 16; ++j) tmp[j] = acc[(size_t)(ow + j) * C_out + co];
                    __m512 v = _mm512_load_ps(tmp);
                    store_m512_to_bfloat16(out_row + ow, v);
                }
                for (; ow < W_out; ++ow)
                    f32_to_bf16(out_row + ow, acc[(size_t)ow * C_out + co]);
            }
        }
    }
}

// Overload: caller-supplied acc, processes all rows.
inline void conv2d_3x3_s2p1_bf16_fast(
    const bf16* __restrict input,
    const bf16* __restrict weight_T,
    const float* __restrict bias,
    bf16*        __restrict output,
    float*       __restrict acc,
    int C_in, int C_out, int H_in, int W_in,
    bool apply_gelu = true)
{
    const int H_out = (H_in + 1) / 2;
    conv2d_3x3_s2p1_bf16_fast(input, weight_T, bias, output, acc,
                               C_in, C_out, H_in, W_in, 0, H_out, apply_gelu);
}

// Overload: allocates acc internally. Use only outside the hot path.
inline void conv2d_3x3_s2p1_bf16_fast(
    const bf16* __restrict input,
    const bf16* __restrict weight_T,
    const float* __restrict bias,
    bf16*        __restrict output,
    int C_in, int C_out, int H_in, int W_in,
    bool apply_gelu = true)
{
    const int W_out = (W_in + 1) / 2;
    std::vector<float> acc_buf((size_t)W_out * C_out);
    conv2d_3x3_s2p1_bf16_fast(input, weight_T, bias, output, acc_buf.data(),
                               C_in, C_out, H_in, W_in, apply_gelu);
}

// ---------------------------------------------------------------------------
// audio_conv_weights
//
//  Pre-transposed weight + bias buffers for the 4-layer conv downsampler.
//  Call prepare() once at load time; pass the struct to audio_conv_downsample.
// ---------------------------------------------------------------------------
struct audio_conv_weights {
    std::vector<float> bias1, bias2, bias3, bias4;  // float32 for fast broadcast
    std::vector<bf16>  w1T, w2T, w3T, w4T;          // transposed [C_in,9,C_out]

    void prepare(
        const bf16* w1, const bf16* b1,
        const bf16* w2, const bf16* b2,
        const bf16* w3, const bf16* b3,
        const bf16* w4, const bf16* b4,
        int C)
    {
        // conv1: C_in=1, conv2-4: C_in=C
        auto to_f32_bias = [](const bf16* src, int n, std::vector<float>& dst) {
            dst.resize(n);
            for (int i = 0; i < n; ++i) dst[i] = bf16_to_f32(src + i);
        };

        w1T.resize(1 * 9 * C);   conv_weight_pretranspose(w1, w1T.data(), 1, C);
        w2T.resize((size_t)C * 9 * C);   conv_weight_pretranspose(w2, w2T.data(), C, C);
        w3T.resize((size_t)C * 9 * C);   conv_weight_pretranspose(w3, w3T.data(), C, C);
        w4T.resize((size_t)C * 9 * C);   conv_weight_pretranspose(w4, w4T.data(), C, C);

        to_f32_bias(b1, C, bias1);
        to_f32_bias(b2, C, bias2);
        to_f32_bias(b3, C, bias3);
        to_f32_bias(b4, C, bias4);
    }
};

// ---------------------------------------------------------------------------
// audio_conv_downsample
//
//  Runs the 4-layer Conv2d downsampler over num_blocks input chunks.
//  audio_conv_weights::prepare() must be called before the first invocation.
//
//  Input
//    input      : [num_blocks, num_mel_bins, chunk_size]  (the chunked mel spectrogram)
//    C          : downsample_hidden_size (e.g. 480)
//
//  Output
//    out4       : [num_blocks, W4, C*H4]  i.e. [b, t, c*f]
//    out_b      : set to num_blocks (batch dimension of out4)
//    out_t      : set to W4 (time/token dimension of out4; derived from chunk_size)
//
//      The NCHW result of conv4 ([C, H4, W4] per block) is transposed so that
//      time steps (W4) become the token/sequence axis.  Index mapping:
//        out4[b, t, co*H4 + fh]  ←  conv4_nchw[co, fh, t]
//
//      This is identical to Python's padded_embed_permuted:
//        padded_embed.permute(0,3,1,2).contiguous().view(b, t, c*f)
//      No further permutation is needed before comparing or feeding conv_out.
//

// ---------------------------------------------------------------------------
inline void audio_conv_downsample(
    const bf16* input,
    int num_blocks,
    int num_mel_bins,
    int chunk_size,
    const audio_conv_weights& wt,

    std::vector<bf16>& out4,

    int C,
    int & out_b,
    int &out_t)
{
    assert(num_blocks > 0 && num_mel_bins > 0 && chunk_size > 0 && C > 0);

    const int H1 = (num_mel_bins + 1) / 2, W1 = (chunk_size + 1) / 2;
    const int H2 = (H1           + 1) / 2, W2 = (W1         + 1) / 2;
    const int H3 = (H2           + 1) / 2, W3 = (W2         + 1) / 2;
    const int H4 = (H3           + 1) / 2, W4 = (W3         + 1) / 2;

    const size_t block_in_elems   = (size_t)num_mel_bins * chunk_size;
    const size_t block_nchw_elems = (size_t)C * H4 * W4;   // conv4 scratch: [C, H4, W4]
    const size_t block_out_elems  = block_nchw_elems;       // out4 per block: [W4, C*H4] (same count, different stride)

    out4.resize((size_t)num_blocks * block_out_elems);
    //assert(out4.size() == (size_t)num_blocks * block_out_elems);

    out_b = num_blocks;
    out_t = W4;

    // Per-block intermediate buffers (shared across layers, sequential within a block).
    std::vector<bf16> t1((size_t)C * H1 * W1);
    std::vector<bf16> t2((size_t)C * H2 * W2);
    std::vector<bf16> t3((size_t)C * H3 * W3);
    // Scratch buffer for conv4 NCHW output before transposing to [t, c*f].
    std::vector<bf16> t4_nchw(block_nchw_elems);

    // Per-thread acc scratch: largest row accumulator is conv1's W1*C floats.
    // Allocated once per thread; index by tid.
    const size_t acc_elems = (size_t)W1 * C;
    std::vector<std::vector<float>> thread_acc(max_prefill_thread,
                                               std::vector<float>(acc_elems));

    // Helper: run one conv layer in parallel over oh rows, each thread using its own acc.
    auto par_conv = [&](const bf16* in, const bf16* wT, const float* bias,
                        bf16* out, int C_in, int H_in, int W_in, bool gelu) {
        const int H_out = (H_in + 1) / 2;
        #pragma omp parallel num_threads(max_prefill_thread)
        {
            const int tid  = omp_get_thread_num();
            float*    acc  = thread_acc[tid].data();
            #pragma omp for schedule(static)
            for (int oh = 0; oh < H_out; ++oh)
                conv2d_3x3_s2p1_bf16_fast(in, wT, bias, out, acc,
                                          C_in, C, H_in, W_in, oh, oh + 1, gelu);
        }
    };

    for (int b = 0; b < num_blocks; ++b) {
        const bf16* blk_in  = input + (size_t)b * block_in_elems;
        bf16*       blk_out = out4.data() + (size_t)b * block_out_elems;

        par_conv(blk_in,      wt.w1T.data(), wt.bias1.data(), t1.data(),      1, num_mel_bins, chunk_size, true);
        par_conv(t1.data(),   wt.w2T.data(), wt.bias2.data(), t2.data(),      C, H1, W1, true);
        par_conv(t2.data(),   wt.w3T.data(), wt.bias3.data(), t3.data(),      C, H2, W2, true);
        par_conv(t3.data(),   wt.w4T.data(), wt.bias4.data(), t4_nchw.data(), C, H3, W3, true);

        // Transpose t4_nchw from [C, H4, W4] to blk_out in [W4, C*H4] layout.
        // Index mapping:  blk_out[t * C*H4 + co*H4 + fh]  =  t4_nchw[co * H4*W4 + fh*W4 + t]
        // Parallelised over co: each thread owns a disjoint co-stripe of blk_out,
        // no atomics needed. t and fh are not parallelised — they are too small
        // (typically W4=7, H4=8) to amortise OMP thread-launch overhead.
        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int co = 0; co < C; ++co) {
            const bf16* src_ch = t4_nchw.data() + (size_t)co * H4 * W4;
            for (int t = 0; t < W4; ++t) {
                bf16* dst_tok = blk_out + (size_t)t * C * H4 + (size_t)co * H4;
                for (int fh = 0; fh < H4; ++fh)
                    dst_tok[fh] = src_ch[fh * W4 + t];
            }
        }
    }
}
