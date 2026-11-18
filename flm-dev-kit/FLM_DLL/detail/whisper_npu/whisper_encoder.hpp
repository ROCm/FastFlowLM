// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "whisper_params.hpp"
#include "tensor_2d.hpp"

class WhisperEncoder {
private:
    /// @brief constexprs

    static constexpr int hidden_size = WHISPER_D;
    static constexpr int num_attention_heads = WHISPER_ENCODER_HEADS;
    static constexpr int head_dim = 64;
    static constexpr int num_key_value_heads = WHISPER_ENCODER_HEADS;
    static constexpr int intermediate_size = WHISPER_ENCODER_FFN_D;
    static constexpr int L = 1536;
    static constexpr int D = hidden_size;
    static constexpr int DQ = head_dim * num_attention_heads;
    static constexpr int DK = head_dim * num_key_value_heads;
    static constexpr int DV = head_dim * num_key_value_heads;

    static constexpr int num_hidden_layers = 32;//  WHISPER_ENCODER_LAYERS; debug

    static constexpr size_t total_proj_weights =
          (num_attention_heads * head_dim) * hidden_size
        + (num_key_value_heads * head_dim) * hidden_size * 2
        + hidden_size * hidden_size
        + intermediate_size * hidden_size * 2;
    static constexpr size_t weight_q_k_v_size = D * (DQ + DK + DV);
    static constexpr size_t weight_o_size = D * DQ;
    static constexpr size_t weight_up_size = D * intermediate_size;
    static constexpr size_t weight_down_size = intermediate_size * D;
    static constexpr size_t total_proj_weights_size = total_proj_weights * 5 / 8;
    static constexpr size_t kv_cache_size = L * num_key_value_heads * head_dim * 2;

    static constexpr int proj_y_group_id = 3;
    static constexpr int proj_x_group_id = 4;
    static constexpr int proj_w_group_id = 5;
    static constexpr int dequant_w_group_id = 3;
    static constexpr int dequant_qw_group_id = 4;
    static constexpr int mha_y_group_id = 3;
    static constexpr int mha_q_group_id = 4;
    static constexpr int mha_kv_group_id = 5;

    /// @brief MM kernel RTP offset (emulated build).
    static constexpr uint32_t CT_rtp_address = 61440;  // pinned in encoder_MM/iron/mm_aie.py

    static constexpr float MY_PI = 3.14159265358979323846;

    /**
    * @brief Converts a vector of 16 floats (__m512) to 16 bfloat16s (__m256i).
    * Uses the dedicated AVX512_BF16 instruction for efficient conversion.
    */
    inline __m256i fp32_to_bf16_avx512(__m512 v_fp32) {
        __m256bh v_bf16 = _mm512_cvtneps_pbh(v_fp32);
        __m256i result;
        std::memcpy(&result, &v_bf16, sizeof(result));
        return result;
    }

    inline float hsum256(__m256 v) {
        __m128 low  = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    // Load 16 bf16 -> 16 fp32 (always AVX-512F)
    static inline __m512 loadu_bf16x16_to_ps(const bf16* p) {
        __m256i b16 = _mm256_loadu_si256((const __m256i*)p);
        __m512i i32 = _mm512_cvtepu16_epi32(b16);   // u16 -> i32 lanes
        i32 = _mm512_slli_epi32(i32, 16);           // place in fp32 high 16
        return _mm512_castsi512_ps(i32);            // bitcast to fp32
    }

    // Store 16 fp32 -> 16 bf16
    #if defined(__AVX512BF16__)
    // Native BF16 fast path
    using bf16pack = __m256bh;

    static inline bf16pack pack_ps_to_bf16x16(__m512 f) {
        return _mm512_cvtneps_pbh(f);              // 16x fp32 -> 16x bf16
    }
    static inline void storeu_bf16x16(bf16* dst, bf16pack v) {
        _mm256_storeu_si256((__m256i*)dst, (__m256i)v);
    }
    #else
    // Portable fallback (software narrowing); still AVX-512F for rest
    using bf16pack = __m256i;

    static inline bf16pack pack_ps_to_bf16x16(__m512 f) {
        alignas(64) float tmp[16];
        alignas(32) bf16  out[16];
        _mm512_store_ps(tmp, f);
        for (int i = 0; i < 16; ++i) {
            uint32_t u; std::memcpy(&u, &tmp[i], 4);
            u += 0x8000u;               // RN
            out[i] = bf16(int((u >> 16)));
        }
        return _mm256_load_si256((const __m256i*)out);
    }
    static inline void storeu_bf16x16(bf16* dst, bf16pack v) {
        _mm256_storeu_si256((__m256i*)dst, v);
    }
    #endif

    // ===================================================================================
    //
    //              Conv1d2: BLOCKED DIRECT CONVOLUTION
    //
    // ===================================================================================

    /**
    * @brief Applies 1D convolution with stride 2, kernel 3 using a direct, blocked AVX-512 implementation.
    *
    * This version introduces two key optimizations:
    * 1. AVX-512 Intrinsics: Processes 16 output elements simultaneously, doubling the throughput of the AVX2 version.
    * The `_mm512_*` intrinsics are used for all vector operations.
    *
    * 2. Cache Blocking (Tiling): The outer loop is restructured to compute a block of output channels (OC_BLOCK)
    * at once. This significantly improves data reuse. For a given spatial window, the corresponding input data
    * is loaded once from the de-interleaved streams and then reused for all output channels within the block.
    * This drastically reduces the number of times we have to go to main memory for the input tensor, leading
    * to much better cache performance.
    *
    * @param dest   Destination buffer (bf16)
    * @param src    Source buffer (bf16)
    * @param weight Weight buffer (bf16)
    * @param bias   Bias buffer (bf16)
    */
    void apply_conv1d2_s2_k3(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& weight, buffer<bf16>& bias, int frames = 3000) {
        const int L_in = frames;     // frames in src (conv1 output), channel-major with stride L_in
        const int IC = 1280;
        const int K  = 3;
        const int OC = weight.size() / (IC * K);
        const int stride = 2;
        const int padding = 1;
        const int L_out = (L_in + 2 * padding - K) / stride + 1;

        // --- AVX-512 specific constants ---
        const int simd_width = 16;
        // --- Cache blocking parameter ---
        // Process this many output channels at a time to improve input data reuse.
        // Must be chosen to balance register pressure and cache benefits.
        const int OC_BLOCK = 4;

        // Assertions to ensure dimensions are valid
        assert(dest.size() >= (size_t)OC * L_out);
        assert((int)bias.size() == OC);
        assert(weight.size() == OC * IC * K);

        // --- Pointers to raw data ---
        bf16* src_ptr  = src.data();
        bf16* w_ptr    = weight.data();
        bf16* dst_ptr  = dest.data();
        bf16* bias_ptr = bias.data();

        auto IDX_SRC = [&](int ic, int t) -> bf16& { return src_ptr[ic * L_in + t]; };
        std::vector<float> weight_f(weight.size());
        for (size_t idx = 0; idx < weight.size(); ++idx) {
            weight_f[idx] = (float)w_ptr[idx];
        }

        auto IDX_W   = [&](int oc, int ic, int k) -> float { return weight_f[(oc * IC + ic) * K + k]; };
        auto IDX_DST = [&](int oc, int t)  -> bf16& { return dst_ptr[oc * L_out + t]; };

        // -----------------------------------------------------------------
        // STEP 1: De-interleave source into even/odd fp32 streams for contiguous vector loads.
        // This pre-processing step is kept from the original implementation as it simplifies
        // the main vectorized loop by ensuring memory accesses are contiguous.
        // -----------------------------------------------------------------
        const int L_even = (L_in + 1) / 2;
        const int L_odd  = L_in / 2;
        std::vector<float> even(IC * L_even);
        std::vector<float> odd (IC * L_odd );

        #pragma omp parallel for schedule(static)
        for (int ic = 0; ic < IC; ++ic) {
            for (int e = 0; e < L_even; ++e) even[ic * L_even + e] = (2 * e < L_in) ? (float)IDX_SRC(ic, 2 * e) : 0.0f;
            for (int o = 0; o < L_odd; ++o) odd[ic * L_odd + o] = (2 * o + 1 < L_in) ? (float)IDX_SRC(ic, 2 * o + 1) : 0.0f;
        }
        auto EVEN = [&](int ic, int e) -> float& { return even[ic * L_even + e]; };
        auto ODD  = [&](int ic, int o) -> float& { return odd[ic * L_odd + o]; };

        // Pre-convert biases to fp32
        std::vector<float> bias_f(OC);
        for(int i = 0; i < OC; ++i) bias_f[i] = (float)bias_ptr[i];

        // -----------------------------------------------------------------
        // STEP 2: Main computation loops with cache blocking
        // Loop order: OC Blocks -> Spatial (t) -> IC
        // -----------------------------------------------------------------
        // Output-channel blocks are independent, so they run on all cores.
        const int n_oc_blocks = OC / OC_BLOCK;
        #pragma omp parallel for schedule(static)
        for (int blk = 0; blk < n_oc_blocks; ++blk) {
            const int oc = blk * OC_BLOCK;
            // ---- Head (t = 0): Scalar computation due to left padding ----
            {
                const int t = 0;
                float acc[OC_BLOCK];
                for(int i=0; i<OC_BLOCK; ++i) acc[i] = bias_f[oc+i];

                for (int ic = 0; ic < IC; ++ic) {
                    float v1 = EVEN(ic, 0);
                    float v2 = (L_odd > 0) ? ODD(ic, 0) : 0.0f;
                    for(int i=0; i<OC_BLOCK; ++i) {
                        acc[i] += v1 * IDX_W(oc + i, ic, 1);
                        if (L_odd > 0) acc[i] += v2 * IDX_W(oc + i, ic, 2);
                    }
                }
                for(int i=0; i<OC_BLOCK; ++i) IDX_DST(oc + i, t) = (bf16)acc[i];
            }

            // ---- Vector body: Process 16 spatial locations (t) at a time ----
            int t = 1;
            int t_end_vec = std::min({L_even, L_odd, L_odd + 1}) - simd_width;

            for (; t <= t_end_vec; t += simd_width) {
                __m512 acc[OC_BLOCK];
                for(int i=0; i<OC_BLOCK; ++i) acc[i] = _mm512_set1_ps(bias_f[oc+i]);

                for (int ic = 0; ic < IC; ++ic) {
                    // Load input data ONCE. This is the key cache benefit.
                    // This data will be reused for all OC_BLOCK channels.
                    const __m512 x0 = _mm512_loadu_ps(&ODD(ic, t - 1));
                    const __m512 x1 = _mm512_loadu_ps(&EVEN(ic, t));
                    const __m512 x2 = _mm512_loadu_ps(&ODD(ic, t));

                    // Perform FMA for each output channel in the block
                    for(int i=0; i<OC_BLOCK; ++i) {
                        const __m512 w0 = _mm512_set1_ps(IDX_W(oc + i, ic, 0));
                        const __m512 w1 = _mm512_set1_ps(IDX_W(oc + i, ic, 1));
                        const __m512 w2 = _mm512_set1_ps(IDX_W(oc + i, ic, 2));

                        acc[i] = _mm512_fmadd_ps(x0, w0, acc[i]);
                        acc[i] = _mm512_fmadd_ps(x1, w1, acc[i]);
                        acc[i] = _mm512_fmadd_ps(x2, w2, acc[i]);
                    }
                }

                // Store results for the block
                for(int i=0; i<OC_BLOCK; ++i) {
                    __m256i y_bf16 = fp32_to_bf16_avx512(acc[i]);
                    _mm256_storeu_si256((__m256i*)&IDX_DST(oc + i, t), y_bf16);
                }
            }

            // ---- Tail (remaining t): Scalar computation for the rest of the row ----
            for (; t < L_out; ++t) {
                float acc[OC_BLOCK];
                for(int i=0; i<OC_BLOCK; ++i) acc[i] = bias_f[oc+i];

                for (int ic = 0; ic < IC; ++ic) {
                    float v0 = (t > 0 && t - 1 < L_odd) ? ODD(ic, t - 1) : 0.0f;
                    float v1 = (t < L_even) ? EVEN(ic, t) : 0.0f;
                    float v2 = (t < L_odd) ? ODD(ic, t) : 0.0f;
                    for(int i=0; i<OC_BLOCK; ++i) {
                        acc[i] += v0 * IDX_W(oc + i, ic, 0);
                        acc[i] += v1 * IDX_W(oc + i, ic, 1);
                        acc[i] += v2 * IDX_W(oc + i, ic, 2);
                    }
                }
                for(int i=0; i<OC_BLOCK; ++i) IDX_DST(oc + i, t) = (bf16)acc[i];
            }
        }

        // --- Final tail for any remaining output channels not fitting in a full block ---
        for (int oc = n_oc_blocks * OC_BLOCK; oc < OC; ++oc) {
            for (int t = 0; t < L_out; ++t) {
                float acc = bias_f[oc];
                for (int ic = 0; ic < IC; ++ic) {
                    int base = t * stride - padding;
                    for (int k=0; k<K; ++k) {
                        int idx = base + k;
                        if (idx >= 0 && idx < L_in) {
                            acc += (float)IDX_SRC(ic, idx) * IDX_W(oc, ic, k);
                        }
                    }
                }
                IDX_DST(oc, t) = (bf16)acc;
            }
        }
    }

    void apply_layernorm_avx(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& weight, buffer<bf16>& bias, float eps) {
        int total = src.size();
        assert(weight.size() == bias.size());

        const int D = weight.size();
        const int L = total / D;
        const int simd_width = 8;

        // Preload weight and bias in fp32
        std::vector<__m256> gamma(D / simd_width), beta(D / simd_width);
        for (int i = 0; i < D; i += simd_width) {
            __m128i w_bf16 = _mm_loadu_si128((__m128i*)&weight[i]);
            __m128i b_bf16 = _mm_loadu_si128((__m128i*)&bias[i]);
            gamma[i / simd_width] = bf16o_fp32(w_bf16);
            beta[i / simd_width] = bf16o_fp32(b_bf16);
        }

        for (int l = 0; l < L; ++l) {
            bf16* x_ptr = src.data() + l * D;
            bf16* y_ptr = dest.data() + l * D;

            // Step 1: compute mean and variance
            __m256 sum = _mm256_setzero_ps();
            __m256 sumsq = _mm256_setzero_ps();

            // Process full SIMD chunks
            for (int i = 0; i < D - (D % simd_width); i += simd_width) {
                __m128i x_bf16 = _mm_loadu_si128((__m128i*)&x_ptr[i]);
                __m256 x_fp32 = bf16o_fp32(x_bf16);
                sum = _mm256_add_ps(sum, x_fp32);
                sumsq = _mm256_fmadd_ps(x_fp32, x_fp32, sumsq);
            }

            // Handle remaining elements
            for (int i = D - (D % simd_width); i < D; ++i) {
                float x_val = (float)x_ptr[i];
                sum = _mm256_add_ps(sum, _mm256_set1_ps(x_val));
                sumsq = _mm256_fmadd_ps(_mm256_set1_ps(x_val), _mm256_set1_ps(x_val), sumsq);
            }

            float mean = hsum256(sum) / D;
            // Correct variance calculation: Var(X) = E[X²] - E[X]²
            float var = hsum256(sumsq) / D - mean * mean;
            float inv_std = 1.0f / sqrtf(var + eps);
            __m256 mean_v = _mm256_set1_ps(mean);
            __m256 inv_std_v = _mm256_set1_ps(inv_std);

            // Step 2: normalize and apply affine
            // Process full SIMD chunks
            for (int i = 0; i < D - (D % simd_width); i += simd_width) {
                __m128i x_bf16 = _mm_loadu_si128((__m128i*)&x_ptr[i]);
                __m256 x_fp32 = bf16o_fp32(x_bf16);
                __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_fp32, mean_v), inv_std_v);
                __m256 y_fp32 = _mm256_fmadd_ps(norm, gamma[i / simd_width], beta[i / simd_width]);
                __m128i y_bf16 = f32o_bf16(y_fp32);
                _mm_storeu_si128((__m128i*)&y_ptr[i], y_bf16);
            }

            // Handle remaining elements
            for (int i = D - (D % simd_width); i < D; ++i) {
                float x_val = (float)x_ptr[i];
                float norm = (x_val - mean) * inv_std;
                float y_val = norm * (float)weight[i] + (float)bias[i];
                y_ptr[i] = bf16(y_val);
            }
        }
    }

    void apply_conv1d1_gemm(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& weight, buffer<bf16>& bias, int frames = 3000) {
        const int src_stride = 3000;   // mel frames per channel in src
        const int Length = frames;     // frames to convolve (zero padding after the last one)
        const int kernel_size = 3;
        const int padding = 1;
        const int IC = 128;                                   // input channels (mel bins)
        const int OC = weight.size() / (IC * kernel_size);     // output channels
        const int L_out = Length + 2 * padding - kernel_size + 1;
        const int simd_width = 8;

        // ------------------------------------------------------------------
        // Step 1: Preload weight to fp32, row-major [OC x (IC*K)]
        // ------------------------------------------------------------------
        const int kernel_dim = IC * kernel_size;
        std::vector<__m256> W_fp32(OC * kernel_dim / simd_width);
        for (int oc = 0; oc < OC; ++oc) {
            for (int d = 0; d < kernel_dim; d += simd_width) {
                __m128i w_bf16 = _mm_loadu_si128((const __m128i*)&weight[oc * kernel_dim + d]);
                W_fp32[(oc * kernel_dim + d) / simd_width] = bf16o_fp32(w_bf16);
            }
        }

        // ------------------------------------------------------------------
        // Step 2: im2col transform of src → fp32 matrix
        // Store row-major: [L_out x kernel_dim]
        // Each row is a window of size kernel_dim (contiguous in memory)
        // ------------------------------------------------------------------
        std::vector<float> im2col((size_t)L_out * kernel_dim);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < L_out; ++t) {
            float* row = &im2col[t * kernel_dim];
            for (int ic = 0; ic < IC; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int idx = t + k - padding;
                    float val = 0.0f;
                    if (idx >= 0 && idx < Length) {
                        bf16 x_bf16 = src[ic * src_stride + idx];
                        val = float(x_bf16);
                    }
                    row[ic * kernel_size + k] = val;
                }
            }
        }

        // ------------------------------------------------------------------
        // Step 3: GEMM (OC x kernel_dim) * (kernel_dim x L_out) = OC x L_out
        // ------------------------------------------------------------------
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC; ++oc) {
            float bias_f = (float)bias[oc];

            for (int t = 0; t < L_out; ++t) {
                __m256 acc = _mm256_setzero_ps();
                const float* row = &im2col[t * kernel_dim];

                int d = 0;
                for (; d <= kernel_dim - simd_width; d += simd_width) {
                    // Load 8 input values
                    __m256 x_fp32 = _mm256_loadu_ps(&row[d]);

                    // Load 8 weights (preconverted)
                    __m256 w_fp32 = W_fp32[(oc * kernel_dim + d) / simd_width];

                    // acc += x*w
                    acc = _mm256_fmadd_ps(x_fp32, w_fp32, acc);
                }

                // Horizontal sum of acc
                float sum = bias_f + hsum256(acc);

                // Handle tail (if kernel_dim not multiple of 8)
                for (; d < kernel_dim; ++d) {
                    float x = row[d];
                    float w = (float)weight[oc * kernel_dim + d];
                    sum += x * w;
                }

                // Store back to dest (fp32 → bf16)
                __m256 y = _mm256_set1_ps(sum);
                __m128i y_bf16 = f32o_bf16(y);
                dest[oc * L_out + t] = ((bf16*)&y_bf16)[0];
            }
        }
    }

    static inline __m256 _mm256_exp_ps(__m256 x) {
        // clamp x to avoid overflow
        const __m256 max_val = _mm256_set1_ps(88.3762626647949f);
        const __m256 min_val = _mm256_set1_ps(-88.3762626647949f);
        x = _mm256_min_ps(x, max_val);
        x = _mm256_max_ps(x, min_val);

        // express exp(x) = 2^(x / ln2)
        const __m256 ln2_inv = _mm256_set1_ps(1.44269504088896341f);  // 1/ln(2)
        __m256 fx = _mm256_mul_ps(x, ln2_inv);
        // round to nearest int
        fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
        __m256i emm0 = _mm256_cvttps_epi32(fx);
        fx = _mm256_cvtepi32_ps(emm0);

        // rebuild x' = x - fx*ln2
        const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
        const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
        __m256 tmp = _mm256_mul_ps(fx, ln2_hi);
        __m256 z   = _mm256_mul_ps(fx, ln2_lo);
        x = _mm256_sub_ps(x, tmp);
        x = _mm256_sub_ps(x, z);

        // polynomial approximation on reduced x
        const __m256 c0 = _mm256_set1_ps(1.9875691500E-4f);
        const __m256 c1 = _mm256_set1_ps(1.3981999507E-3f);
        const __m256 c2 = _mm256_set1_ps(8.3334519073E-3f);
        const __m256 c3 = _mm256_set1_ps(4.1665795894E-2f);
        const __m256 c4 = _mm256_set1_ps(1.6666665459E-1f);
        const __m256 c5 = _mm256_set1_ps(5.0000001201E-1f);

        __m256 y = c0;
        y = _mm256_fmadd_ps(y, x, c1);
        y = _mm256_fmadd_ps(y, x, c2);
        y = _mm256_fmadd_ps(y, x, c3);
        y = _mm256_fmadd_ps(y, x, c4);
        y = _mm256_fmadd_ps(y, x, c5);
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.0f));

        // build 2^n from the integer part
        emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7F));
        emm0 = _mm256_slli_epi32(emm0, 23);
        __m256 pow2n = _mm256_castsi256_ps(emm0);

        // combine
        return _mm256_mul_ps(y, pow2n);
    }

    inline __m512 _mm512_exp_ps_corrected(__m512 x) {
        // clamp x to a reasonable range to avoid overflow/underflow
        const __m512 max_val = _mm512_set1_ps(88.0f);
        const __m512 min_val = _mm512_set1_ps(-88.0f);
        x = _mm512_min_ps(x, max_val);
        x = _mm512_max_ps(x, min_val);

        // express exp(x) = 2^n * exp(r), where n is integer and |r| <= 0.5*ln(2)
        const __m512 ln2_inv = _mm512_set1_ps(1.44269504088896341f);  // 1/ln(2)
        const __m512 ln2_hi = _mm512_set1_ps(0.693147182f);
        const __m512 ln2_lo = _mm512_set1_ps(-1.902275E-9f);

        // n = round(x / ln(2))
        __m512 fx = _mm512_mul_ps(x, ln2_inv);

        // Use proper rounding for all float values.
        fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // The integer part n
        __m512i emm0 = _mm512_cvttps_epi32(fx);

        // The reduced argument r = x - n * ln(2)
        __m512 n_ps = _mm512_cvtepi32_ps(emm0);
        __m512 r_hi = _mm512_fmsub_ps(n_ps, ln2_hi, x); // r_hi = n*ln2_hi - x
        __m512 r_lo = _mm512_mul_ps(n_ps, ln2_lo);
        __m512 r = _mm512_sub_ps(r_lo, r_hi); // r = -(n*ln2_hi - x) + n*ln2_lo = x - n*ln2

        // Polynomial approximation for exp(r) on the reduced range
        const __m512 c1 = _mm512_set1_ps(1.0f);
        const __m512 c2 = _mm512_set1_ps(0.5f);
        const __m512 c3 = _mm512_set1_ps(0.1666666667f);
        const __m512 c4 = _mm512_set1_ps(0.0416666667f);
        const __m512 c5 = _mm512_set1_ps(0.0083333333f);

        // Correctly evaluate the polynomial using Horner's method
        __m512 y = _mm512_fmadd_ps(c5, r, c4);
        y = _mm512_fmadd_ps(y, r, c3);
        y = _mm512_fmadd_ps(y, r, c2);
        y = _mm512_fmadd_ps(y, r, c1);
        y = _mm512_fmadd_ps(y, r, c1); // Note: c1 is 1.0f

        // Build 2^n from the integer part n
        __m512i pow2n_i = _mm512_add_epi32(emm0, _mm512_set1_epi32(127));
        pow2n_i = _mm512_slli_epi32(pow2n_i, 23);
        __m512 pow2n = _mm512_castsi512_ps(pow2n_i);

        // Combine: exp(x) = 2^n * exp(r)
        return _mm512_mul_ps(y, pow2n);
    }

    inline __m512 gelu_avx512(__m512 x) {
        // Exact GELU, 0.5 x (1 + erf(x / sqrt(2))), as Whisper was trained with.
        // The tanh approximation is several percent off in the negative tail,
        // where whole MLP layers of the decoder can sit. erfc is Abramowitz &
        // Stegun 7.1.26 (|error| <= 1.5e-7), arranged so the negative tail keeps
        // its relative accuracy: x < 0 gives 0.5 x erfc(|x| / sqrt(2)) and
        // x >= 0 gives x minus that.
        const __m512 inv_sqrt2 = _mm512_set1_ps(0.70710678118654752f);
        const __m512 p  = _mm512_set1_ps(0.3275911f);
        const __m512 a1 = _mm512_set1_ps(0.254829592f);
        const __m512 a2 = _mm512_set1_ps(-0.284496736f);
        const __m512 a3 = _mm512_set1_ps(1.421413741f);
        const __m512 a4 = _mm512_set1_ps(-1.453152027f);
        const __m512 a5 = _mm512_set1_ps(1.061405429f);
        const __m512 one = _mm512_set1_ps(1.0f);
        const __m512 half = _mm512_set1_ps(0.5f);

        __m512 z = _mm512_mul_ps(_mm512_abs_ps(x), inv_sqrt2);
        __m512 t = _mm512_div_ps(one, _mm512_fmadd_ps(p, z, one));
        __m512 poly = _mm512_fmadd_ps(a5, t, a4);
        poly = _mm512_fmadd_ps(poly, t, a3);
        poly = _mm512_fmadd_ps(poly, t, a2);
        poly = _mm512_fmadd_ps(poly, t, a1);
        poly = _mm512_mul_ps(poly, t);
        __m512 erfc = _mm512_mul_ps(poly, _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(z, z))));

        __m512 tail = _mm512_mul_ps(half, _mm512_mul_ps(x, erfc));
        __mmask16 positive = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GE_OQ);
        return _mm512_mask_sub_ps(tail, positive, x, tail);
    }

    void apply_gelu_avx512(buffer<bf16>& dest, const buffer<bf16>& src, size_t n = (size_t)-1) {
        const size_t simd_width = 16; // AVX512 processes 16 floats at once
        const size_t N = (n == (size_t)-1) ? src.size() : n;

        // Process vectorized chunks
        // Elements are independent, so split the vector chunks across cores.
        const long n_chunks = (long)(N / simd_width);
        #pragma omp parallel for schedule(static) if (n_chunks > 4096)
        for (long c = 0; c < n_chunks; ++c) {
            const size_t i = (size_t)c * simd_width;
            // Load 16 bfloat16 values and convert to float32
            __m512 src_fp32 = loadu_bf16x16_to_ps(&src[i]);

            // Apply GELU activation
            __m512 gelu_result = gelu_avx512(src_fp32);

            // Convert back to bfloat16 and store
            __m256i result_bf16 = fp32_to_bf16_avx512(gelu_result);
            _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
        }
        size_t i = (size_t)n_chunks * simd_width;

        // Handle remaining elements (scalar processing)
        for (; i < N; i++) {
            float val_fp32 = float(src[i]);
            float gelu_val = 0.5f * val_fp32 * std::erfc(-val_fp32 * 0.70710678118654752f);
            dest[i] = (bf16)gelu_val;
        }
    }

    // Vectorized version with AVX512 for better performance
    void transpose_zeropad_add_positional_embedding_avx512_vectorized(
        buffer<bf16>& output_data,           // Output: (1536, 1280) with positional embeddings added
        const buffer<bf16>& input_data,      // Input: (1280, 1500)
        const buffer<bf16>& positional_embedding, // Positional embeddings: (1536, 1280)
        const int input_height,              // 1280
        const int input_width,               // 1500
        const int output_height,             // 1536 (padded height)
        const int output_width) {            // 1280

        const size_t simd_width = 16; // AVX512 processes 16 floats at once
        const size_t input_size = (size_t)input_height * input_width;
        const size_t output_size = (size_t)output_height * output_width;

        // Initialize output with zeros (zero-padding)
        std::fill(output_data.begin(), output_data.end(), bf16(0.0f));

        // Process in vectorized blocks
        const int VECTOR_BLOCK_SIZE = 64; // Process 64 elements at a time

        // Only process the transposed data (1500 rows), leave the rest zero-padded
        const int transposed_height = input_width; // 1500 (after transpose)

        for (int i = 0; i < transposed_height; ++i) {
            for (int j_start = 0; j_start < input_height; j_start += VECTOR_BLOCK_SIZE) {
                int j_end = std::min(j_start + VECTOR_BLOCK_SIZE, input_height);

                // Process vectorized chunks
                int j = j_start;
                for (; j + simd_width <= j_end; j += simd_width) {
                    // Load 16 input values (transposed)
                    alignas(64) float input_vals[16];
                    alignas(64) float embed_vals[16];

                    for (int k = 0; k < 16; ++k) {
                        const size_t input_idx = (size_t)(j + k) * input_width + i;
                        const size_t embed_idx = (size_t)i * output_width + (j + k);

                        input_vals[k] = float(input_data[input_idx]);
                        embed_vals[k] = float(positional_embedding[embed_idx]);
                    }

                    // Load into AVX512 registers
                    __m512 input_vec = _mm512_loadu_ps(input_vals);
                    __m512 embed_vec = _mm512_loadu_ps(embed_vals);

                    // Add: result = input + embedding
                    __m512 result_vec = _mm512_add_ps(input_vec, embed_vec);

                    // Store result (transposed back)
                    alignas(64) float result_vals[16];
                    _mm512_storeu_ps(result_vals, result_vec);

                    for (int k = 0; k < 16; ++k) {
                        const size_t output_idx = (size_t)i * output_width + (j + k);
                        output_data[output_idx] = (bf16)result_vals[k];
                    }
                }

                // Handle remaining elements
                for (; j < j_end; ++j) {
                    const size_t input_idx = (size_t)j * input_width + i;
                    const size_t output_idx = (size_t)i * output_width + j;
                    const size_t embed_idx = (size_t)i * output_width + j; // FIXED: Use padded_width

                    float input_val = float(input_data[input_idx]);
                    float embed_val = float(positional_embedding[embed_idx]);
                    float result_val = input_val + embed_val;
                    output_data[output_idx] = (bf16)result_val;
                }
            }
        }
    }

    // Optimized version specifically for your dimensions (1280, 1500) -> (1536, 1280)
    void transpose_zeropad_add_positional_embedding_1280x1500_avx512(
        buffer<bf16>& output_data,           // Output: (out_rows, 1280) with positional embeddings added
        const buffer<bf16>& input_data,      // Input: (1280, in_width)
        const buffer<bf16>& positional_embedding, // Positional embeddings: (1536, 1280)
        int in_width = 1500,
        int out_rows = 1536) {

        const int input_height = 1280;
        const int input_width = in_width;
        const int output_height = out_rows;
        const int output_width = 1280;

        // Use the vectorized version for optimal performance
        transpose_zeropad_add_positional_embedding_avx512_vectorized(
            output_data, input_data, positional_embedding,
            input_height, input_width, output_height, output_width
        );
    }

    /**
    * @brief Highly optimized AVX-512 implementation for adding bias with advanced cache optimization.
    *
    * This is an enhanced version with additional optimizations:
    * 1. Row-wise blocking for better cache utilization
    * 2. Memory prefetching with multiple prefetch streams
    * 3. Loop unrolling for reduced branch overhead
    * 4. Optimized memory access patterns
    * 5. Better register utilization
    *
    * @param dest   Output buffer (bf16) - same size as src
    * @param src    Input q projection matrix (bf16) - [1536, 1280]
    * @param bias   Bias vector (bf16) - [1, 1280]
    */
    void add_bias_optimized(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& bias, int rows = 1536) {
        const int ROWS = rows;
        const int COLS = bias.size();
        const int SIMD_WIDTH = 16;
        const int ROW_BLOCK_SIZE = 8; // Process 8 rows at a time
        const int COL_BLOCK_SIZE = 128; // Process 128 columns at a time
        const int PREFETCH_DISTANCE = 256;

        // Validate input dimensions
        assert(src.size() >= (size_t)ROWS * COLS);
        assert(bias.size() == COLS);
        assert(dest.size() >= (size_t)ROWS * COLS);

        // Get raw pointers
        bf16* src_ptr = src.data();
        bf16* bias_ptr = bias.data();
        bf16* dest_ptr = dest.data();

        // Pre-convert bias to fp32 with alignment
        std::vector<float> bias_vec_f32(COLS);
        float* bias_fp32 = bias_vec_f32.data();
        for (int j = 0; j < COLS; ++j) {
            bias_fp32[j] = float(bias_ptr[j]);
        }

        // Process in row blocks for better cache locality; blocks are independent
        #pragma omp parallel for schedule(static)
        for (int i_start = 0; i_start < ROWS; i_start += ROW_BLOCK_SIZE) {
            int i_end = std::min(i_start + ROW_BLOCK_SIZE, ROWS);

            // Process each column block
            for (int j_start = 0; j_start < COLS; j_start += COL_BLOCK_SIZE) {
                int j_end = std::min(j_start + COL_BLOCK_SIZE, COLS);

                // Process each row in the current block
                for (int i = i_start; i < i_end; ++i) {
                    bf16* src_row = src_ptr + i * COLS;
                    bf16* dest_row = dest_ptr + i * COLS;

                    // Vectorized processing with unrolling
                    int j = j_start;
                    for (; j + 2 * SIMD_WIDTH <= j_end; j += 2 * SIMD_WIDTH) {
                        // Process two SIMD blocks at once
                        __m512 src_vec1 = loadu_bf16x16_to_ps(&src_row[j]);
                        __m512 src_vec2 = loadu_bf16x16_to_ps(&src_row[j + SIMD_WIDTH]);

                        __m512 bias_vec1 = _mm512_loadu_ps(&bias_fp32[j]);
                        __m512 bias_vec2 = _mm512_loadu_ps(&bias_fp32[j + SIMD_WIDTH]);

                        __m512 result_vec1 = _mm512_add_ps(src_vec1, bias_vec1);
                        __m512 result_vec2 = _mm512_add_ps(src_vec2, bias_vec2);

                        __m256i result_bf16_1 = fp32_to_bf16_avx512(result_vec1);
                        __m256i result_bf16_2 = fp32_to_bf16_avx512(result_vec2);

                        _mm256_storeu_si256((__m256i*)&dest_row[j], result_bf16_1);
                        _mm256_storeu_si256((__m256i*)&dest_row[j + SIMD_WIDTH], result_bf16_2);
                    }

                    // Handle remaining SIMD block
                    for (; j + SIMD_WIDTH <= j_end; j += SIMD_WIDTH) {
                        __m512 src_vec = loadu_bf16x16_to_ps(&src_row[j]);
                        __m512 bias_vec = _mm512_loadu_ps(&bias_fp32[j]);
                        __m512 result_vec = _mm512_add_ps(src_vec, bias_vec);
                        __m256i result_bf16 = fp32_to_bf16_avx512(result_vec);
                        _mm256_storeu_si256((__m256i*)&dest_row[j], result_bf16);
                    }

                    // Handle remaining elements
                    for (; j < j_end; ++j) {
                        float src_val = float(src_row[j]);
                        float bias_val = bias_fp32[j];
                        float result_val = src_val + bias_val;
                        dest_row[j] = (bf16)result_val;
                    }
                }
            }
        }
    }


    /// dst[i] = src[i] for n elements (n a multiple of 16)
    void bf16_to_f32(float* dst, const bf16* src, size_t n){
        #pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < n; i += 16){
            _mm512_storeu_ps(dst + i, loadu_bf16x16_to_ps(src + i));
        }
    }

    /// acc[i] += x[i] for n elements (n a multiple of 16), in fp32
    void add_bf16_to_f32(float* acc, const bf16* x, size_t n){
        #pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < n; i += 16){
            _mm512_storeu_ps(acc + i, _mm512_add_ps(_mm512_loadu_ps(acc + i), loadu_bf16x16_to_ps(x + i)));
        }
    }

    // Round 16 floats to bf16 and back, as storing to a bf16 buffer and reloading would.
    inline __m512 round_bf16_ps(__m512 v) {
        __m512i wide = _mm512_cvtepu16_epi32(fp32_to_bf16_avx512(v));
        return _mm512_castsi512_ps(_mm512_slli_epi32(wide, 16));
    }

    /// acc[r][c] += bf16(x[r][c] + bias[c]) over rows x bias.size(): the bias add and the residual
    /// add in one pass. x is only read, so the NPU buffer it lives in needs no flush afterwards.
    void add_biased_bf16_to_f32(float* acc, const bf16* x, const float* bias, int rows, int cols){
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++){
            const bf16* xr = x + (size_t)r * cols;
            float* ar = acc + (size_t)r * cols;
            for (int c = 0; c < cols; c += 16){
                __m512 biased = round_bf16_ps(_mm512_add_ps(loadu_bf16x16_to_ps(xr + c), _mm512_loadu_ps(bias + c)));
                _mm512_storeu_ps(ar + c, _mm512_add_ps(_mm512_loadu_ps(ar + c), biased));
            }
        }
    }

    /// buf[r][c] = gelu(bf16(buf[r][c] + bias[c])) over rows x bias.size(): the fc1 bias add and the
    /// GELU in one pass, rounding where the two separate passes did.
    /// dst[r][c] = gelu(bf16(src[r][c] + bias[c])) over rows x cols
    void add_bias_gelu_to(bf16* dst, const bf16* src, const float* bias, int rows, int cols){
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++){
            const bf16* in = src + (size_t)r * cols;
            bf16* out = dst + (size_t)r * cols;
            for (int c = 0; c < cols; c += 16){
                __m512 biased = round_bf16_ps(_mm512_add_ps(loadu_bf16x16_to_ps(in + c), _mm512_loadu_ps(bias + c)));
                _mm256_storeu_si256((__m256i*)(out + c), fp32_to_bf16_avx512(gelu_avx512(biased)));
            }
        }
    }
    void add_bias_gelu(bf16* buf, const float* bias, int rows, int cols){
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++){
            bf16* row = buf + (size_t)r * cols;
            for (int c = 0; c < cols; c += 16){
                __m512 biased = round_bf16_ps(_mm512_add_ps(loadu_bf16x16_to_ps(row + c), _mm512_loadu_ps(bias + c)));
                _mm256_storeu_si256((__m256i*)(row + c), fp32_to_bf16_avx512(gelu_avx512(biased)));
            }
        }
    }

    /// LayerNorm of `rows` rows of an fp32 residual stream into bf16. The residual
    /// stays fp32 because it carries large outlier channels: in bf16, adding a
    /// layer's small update to them rounds most of the update away. The variance
    /// is two-pass for the same reason.
    void apply_layernorm_f32(buffer<bf16>& dest, const float* src, size_t rows, buffer<bf16>& weight, buffer<bf16>& bias, float eps){
        apply_layernorm_f32(dest.data(), src, rows, weight, bias, eps);
    }
    void apply_layernorm_f32(bf16* dest, const float* src, size_t rows, buffer<bf16>& weight, buffer<bf16>& bias, float eps){
        const int D = weight.size();
        #pragma omp parallel for schedule(static) if (rows > 1)
        for (size_t r = 0; r < rows; r++){
            const float* x = src + r * D;
            bf16* y = dest + r * D;
            __m512 s = _mm512_setzero_ps();
            for (int i = 0; i < D; i += 16) s = _mm512_add_ps(s, _mm512_loadu_ps(x + i));
            const __m512 mean = _mm512_set1_ps(_mm512_reduce_add_ps(s) / D);
            __m512 v = _mm512_setzero_ps();
            for (int i = 0; i < D; i += 16){
                __m512 d = _mm512_sub_ps(_mm512_loadu_ps(x + i), mean);
                v = _mm512_fmadd_ps(d, d, v);
            }
            const __m512 inv_std = _mm512_set1_ps(1.0f / std::sqrt(_mm512_reduce_add_ps(v) / D + eps));
            for (int i = 0; i < D; i += 16){
                __m512 n = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + i), mean), inv_std);
                __m512 out = _mm512_fmadd_ps(n, loadu_bf16x16_to_ps(weight.data() + i), loadu_bf16x16_to_ps(bias.data() + i));
                _mm256_storeu_si256((__m256i*)(y + i), fp32_to_bf16_avx512(out));
            }
        }
    }

    void residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual){
        int n = src.size();
        const int simd_width = 8;

        for (int i = 0; i < n; i += simd_width){
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[i]);
            __m128i bf16_vals_r = _mm_loadu_si128((__m128i*)&residual[i]);
            __m256 fp32_vals = bf16o_fp32(bf16_vals);
            __m256 fp32_vals_r = bf16o_fp32(bf16_vals_r);
            __m256 result_vec = _mm256_add_ps(fp32_vals, fp32_vals_r);
            __m128i result_bf16 = f32o_bf16(result_vec);
            _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
        }
    }

    void load_all_proj_weights(buffer<u8>& w, Q4NX& q4nx, int layer_idx){
        const int DQ = 1280;
        const int DK = 1280;
        const int DV = 1280;
        const int D = 1280;
        const int INTERMEDIATE_SIZE = 5120;
        u8* w_ptr = w.data();
        u8* w_ptr_end = w.end();
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_q, "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
            q4nx.load_weights(w_k, "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
            q4nx.load_weights(w_v, "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            memcpy(w_ptr, w_q.data(), DQ * D * 5 / 8);
            w_ptr += DQ * D * 5 / 8;
            memcpy(w_ptr, w_k.data(), DK * D * 5 / 8);
            w_ptr += DK * D * 5 / 8;
            memcpy(w_ptr, w_v.data(), DV * D * 5 / 8);
            w_ptr += DV * D * 5 / 8;
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading o weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_o;
            q4nx.load_weights(w_o, "model.encoder.layers." + std::to_string(layer_idx) + ".self_attn.out_proj.weight");
            memcpy(w_ptr, w_o.data(), D * D * 5 / 8);
            w_ptr += D * D * 5 / 8;
        }
        // read up weights
        {
            LOG_VERBOSE(1, "Loading up weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_up;
            q4nx.load_weights(w_up, "model.encoder.layers." + std::to_string(layer_idx) + ".fc1.weight");
            memcpy(w_ptr, w_up.data(), INTERMEDIATE_SIZE * D * 5 / 8);
            w_ptr += INTERMEDIATE_SIZE * D * 5 / 8;
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading down weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_down;
            q4nx.load_weights(w_down, "model.encoder.layers." + std::to_string(layer_idx) + ".fc2.weight");
            memcpy(w_ptr, w_down.data(), INTERMEDIATE_SIZE * D * 5 / 8);
            w_ptr += INTERMEDIATE_SIZE * D * 5 / 8;
        }
        if (w_ptr != w_ptr_end){
            std::cout << "Warning: w_ptr != w_ptr_end" << std::endl;
            exit(1);
        }
    }

    Whisper_Config config;
    npu_xclbin_manager *npu;

    npu_app_manager* mm_xclbin;
    npu_app_manager* attention_xclbin;
    npu_app_manager* dequant_xclbin;

    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app up_proj;
    npu_app down_proj;
    // the same three matmuls on one half of the rows each (index 0: rows 0..767, 1: rows 768..1535)
    npu_app o_half[2];
    npu_app up_half[2];
    npu_app down_half[2];

    npu_app mha_engine;

    npu_app dequant_qkv;
    npu_app dequant_o;
    npu_app dequant_up;
    npu_app dequant_down;

    npu_app dequant_kv_cache;

    npu_app k_cache_proj;
    npu_app v_cache_proj;

    std::vector<buffer<u8>> proj_weights;
    buffer<bf16> weights_qkv;
    buffer<bf16> weights_o;
    buffer<bf16> weights_up;
    buffer<bf16> weights_down;
    // Per-layer cache of the dequantised bf16 weights (about 1.25 GB for 32
    // layers), on by default; WHISPER_CACHE_DEQUANT=0 turns it off. Filled on the
    // first encode; later encodes skip the dequant xclbin, which also saves two
    // xclbin switches per layer.
    bool cache_dequant = [] {
        const char* v = std::getenv("WHISPER_CACHE_DEQUANT");
        return !(v && std::string(v) == "0");
    }();
    std::vector<bool> dequant_cached = std::vector<bool>(num_hidden_layers, false);
    std::vector<buffer<bf16>> cache_qkv = std::vector<buffer<bf16>>(num_hidden_layers);
    std::vector<buffer<bf16>> cache_o = std::vector<buffer<bf16>>(num_hidden_layers);
    std::vector<buffer<bf16>> cache_up = std::vector<buffer<bf16>>(num_hidden_layers);
    std::vector<buffer<bf16>> cache_down = std::vector<buffer<bf16>>(num_hidden_layers);
    std::vector<bool> kv_dequant_cached = std::vector<bool>(WHISPER_DECODER_LAYERS, false);
    std::vector<buffer<bf16>> cache_kv = std::vector<buffer<bf16>>(WHISPER_DECODER_LAYERS);

    buffer<bf16> buf_q_o_d; // shared by q_proj, o_proj, down_proj
    buffer<bf16> buf_attn_output;
    buffer<bf16> buf_up;
    buffer<bf16> kv_cache;
    buffer<bf16> buf_x;

    buffer<bf16> conv1d_1_weight;
    buffer<bf16> conv1d_1_bias;
    buffer<bf16> conv1d_2_weight;
    buffer<bf16> conv1d_2_bias;

    buffer<bf16> residual_buffer;
    std::vector<float> residual_f32;  // the residual stream, kept in fp32

    buffer<bf16> positional_embedding;

    buffer<bf16> encoder_out;

    std::vector<buffer<bf16>> attn_norm_weights;
    std::vector<buffer<bf16>> attn_norm_biases;
    std::vector<buffer<bf16>> mlp_norm_weights;
    std::vector<buffer<bf16>> mlp_norm_biases;

    std::vector<buffer<bf16>> q_biases;
    std::vector<buffer<bf16>> v_biases;
    std::vector<buffer<bf16>> o_biases;
    std::vector<buffer<bf16>> fc1_biases;
    std::vector<buffer<bf16>> fc2_biases;

    std::vector<buffer<u8>> cross_attn_kv_proj_weights;
    std::vector<buffer<bf16>> cross_attn_v_proj_biases;

    buffer<bf16> buf_kv_cache_proj_weight;

    buffer<bf16> encoder_layer_norm_weights;
    buffer<bf16> encoder_layer_norm_biases;

    buffer<bf16> buf_k_cache;
    buffer<bf16> buf_v_cache;
    buffer<bf16> KV_CACHE;

    // sequence functions
    // y_base / x_base: element offsets of the output / input inside their buffers (k and v write into the
    // two halves of kv_cache; the row-half matmuls read and write one half of the rows)
    void prepare_mm(npu_sequence* seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t m, const uint32_t k, const uint32_t n, const uint32_t weight_offset, const uint32_t y_base = 0, const uint32_t x_base = 0, const uint32_t x_row_stride = 0);
    void post_attention_split(int layer_idx, buffer<bf16>& w_o, buffer<bf16>& w_up, buffer<bf16>& w_down);
    bool ln1_first_half_done = false;   // post_attention_split already wrote the next ln1's first half
    void prepare_dequant(npu_sequence* seq_ptr, const u32 D_in, const u32 D_out, const u32 weight_offset);
    void prepare_attention(npu_sequence* seq, const uint32_t Lq, const uint32_t Lk, const uint32_t valid_keys);

    // fp32 biases for the fused bias passes; o's has the V bias folded in (fold_v_bias)
    std::vector<std::vector<float>> o_bias_f32, fc1_bias_f32, fc2_bias_f32;
    void fold_v_bias(int layer_idx, buffer<bf16>& w_o);
    void prepare_sequences();

    // conv front end on the NPU (see conv_front_end)
    static constexpr int CONV1_K = 4 * 128;          // 4 frames x 128 mel bins (4th frame: zero weights)
    static constexpr int CONV2_K = 3 * 1280;         // 3 frames x 1280 channels
    static constexpr int CONV1_ROWS = 3072 + 4;      // conv1 rows read + padding rows
    npu_app conv1_mm, conv2_mm;
    buffer<bf16> conv1_x, conv1_w, conv2_x, conv2_w;
    std::vector<float> conv1_bias_f32, conv2_bias_f32;
    void pack_conv_weights();
    void conv_front_end(const buffer<bf16>& mel);

public:

    WhisperEncoder(Whisper_Config config, npu_xclbin_manager *npu);
    ~WhisperEncoder();

    void load_weights(Q4NX& q4nx);

    buffer<bf16> forward(buffer<bf16>& mel_feature);
};
