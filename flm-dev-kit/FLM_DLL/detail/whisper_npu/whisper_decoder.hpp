// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "whisper_params.hpp"
#include "tensor_2d.hpp"
#include "modules/embedding.hpp"

class WhisperDecoder {
private:
    /// @brief constexprs

    static constexpr int hidden_size = WHISPER_D;
    static constexpr int num_attention_heads = WHISPER_DECODER_HEADS;
    static constexpr int head_dim = 64;
    static constexpr int num_key_value_heads = WHISPER_DECODER_HEADS;
    static constexpr int intermediate_size = WHISPER_DECODER_FFN_D;
    static constexpr int CROSS_ATTN_L = 1536;
    static constexpr int SELF_ATTN_L = 448;
    static constexpr int D = hidden_size;
    static constexpr int DQ = head_dim * num_attention_heads;
    static constexpr int DK = head_dim * num_key_value_heads;
    static constexpr int DV = head_dim * num_key_value_heads;
    static constexpr int num_hidden_layers = WHISPER_DECODER_LAYERS;
    static constexpr float SCALING = 0.125f;

    static constexpr size_t total_proj_weights =
        (num_attention_heads * head_dim) * hidden_size * 2
    +   (num_key_value_heads * head_dim) * hidden_size * 2
    +   hidden_size * hidden_size * 2
    +   intermediate_size * hidden_size * 2;

    static constexpr size_t total_proj_weights_size = total_proj_weights * 5 / 8;
    static constexpr size_t weight_qkv_size = D * (DQ + DK + DV) * 5 / 8;
    static constexpr size_t weight_o_size = D * DQ * 5 / 8;
    static constexpr size_t weight_q_size = D * DQ * 5 / 8;
    static constexpr size_t weight_up_size = intermediate_size * D * 5 / 8;
    static constexpr size_t weight_down_size = D * intermediate_size * 5 / 8;

    static constexpr int proj_y_group_id = 3;
    static constexpr int proj_x_group_id = 4;
    static constexpr int proj_w_group_id = 5;

    // MM kernel RTP offsets: one set for the projection kernel, one shared by
    // the QK^T and SV attention kernels.
    static constexpr uint32_t proj_M_div_m_rtp_address = 61440;  // pinned in decoder_mv_dequant_mv/iron/mv_aie.py
    static constexpr uint32_t proj_CT_rtp_address = 62464;
    static constexpr uint32_t attn_M_div_m_rtp_address = 61440;
    static constexpr uint32_t attn_CT_rtp_address = 62464;

    static constexpr float MY_PI = 3.14159265358979323846;

    // ===================================================================================
    //
    //                        UTILITY FUNCTIONS
    //
    // ===================================================================================

    /**
    * @brief Converts a bfloat16 value (stored as uint16_t) to a float.
    * This is a scalar conversion.
    */
    inline float bf16_to_fp32(bf16 val) {
        uint32_t val_32 = static_cast<uint32_t>(val) << 16;
        return *reinterpret_cast<float*>(&val_32);
    }

    /**
    * @brief Converts a float value to a bfloat16 (stored as uint16_t).
    * This is a scalar conversion that truncates the mantissa.
    */
    inline bf16 fp32_to_bf16(float val) {
        uint32_t val_32 = *reinterpret_cast<uint32_t*>(&val);
        return static_cast<bf16>(static_cast<u16>(val_32 >> 16));
    }

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
        __m256i result;
        std::memcpy(&result, &v, sizeof(result));
        _mm256_storeu_si256((__m256i*)dst, result);
    }
    #else
    // Portable fallback (software narrowing); still AVX-512F for rest
    using bf16pack = __m256i;

    static inline bf16pack pack_ps_to_bf16x16(__m512 f) {
        alignas(64) float tmp[16];
        alignas(32) bf16  out[16];
        uint16_t *out_ptr = reinterpret_cast<uint16_t*>(out);
        _mm512_store_ps(tmp, f);
        for (int i = 0; i < 16; ++i) {
            uint32_t u; std::memcpy(&u, &tmp[i], 4);
            u += 0x8000u;               // RN
            out_ptr[i] = uint16_t(u >> 16);
        }
        return _mm256_load_si256((const __m256i*)out);
    }
    static inline void storeu_bf16x16(bf16* dst, bf16pack v) {
        _mm256_storeu_si256((__m256i*)dst, v);
    }
    #endif

    inline void fp32_to_bf16_store(uint16_t* dst, __m512 val) {
        __m256bh bf16_val = _mm512_cvtneps_pbh(val); // round to nearest even
        __m256i result;
        std::memcpy(&result, &bf16_val, sizeof(result));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), result);
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

    // AVX-512 GELU tanh-based, now using the corrected exp function
    inline __m512 gelu_tanh_avx512_simd(__m512 gate_vec_fp32) {
    // ---- GELU(gate) with tanh approximation: 0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))  ----
        const __m512 half = _mm512_set1_ps(0.5f);
        const __m512 one = _mm512_set1_ps(1.0f);
        const __m512 sqrt_2_pi = _mm512_set1_ps(0.7978845608f); // sqrt(2/pi)
        const __m512 coeff = _mm512_set1_ps(0.044715f);

        // Compute gate^3
        __m512 gate_squared = _mm512_mul_ps(gate_vec_fp32, gate_vec_fp32);
        __m512 gate_cubed = _mm512_mul_ps(gate_squared, gate_vec_fp32);

        // Compute gate + 0.044715 * gate^3
        __m512 inner_term = _mm512_fmadd_ps(coeff, gate_cubed, gate_vec_fp32);

        // Compute sqrt(2/pi) * (gate + 0.044715 * gate^3)
        __m512 scaled_term = _mm512_mul_ps(sqrt_2_pi, inner_term);

        // Compute tanh using the corrected exp function: tanh(x) ≈ (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        __m512 exp_pos = _mm512_exp_ps_corrected(scaled_term);
        __m512 exp_neg = _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), scaled_term));

        __m512 numerator = _mm512_sub_ps(exp_pos, exp_neg);
        __m512 denominator = _mm512_add_ps(exp_pos, exp_neg);
        __m512 tanh_approx = _mm512_div_ps(numerator, denominator);

        // Compute 1 + tanh(...)
        __m512 one_plus_tanh = _mm512_add_ps(one, tanh_approx);

        // Compute 0.5 * gate * (1 + tanh(...))
        __m512 gelu = _mm512_mul_ps(half, _mm512_mul_ps(gate_vec_fp32, one_plus_tanh));
        return gelu;
    }

    // GELU activation function with tanh approximation (same as PytorchGELUTanh)
    inline float gelu_tanh(float x) {
        return 0.5f * x * std::erfc(-x * 0.70710678118654752f);
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

    // Apply GELU activation to a buffer of bfloat16 values using AVX512
    // Processes 16 elements at a time for maximum efficiency
    void apply_gelu_avx512(buffer<bf16>& dest, const buffer<bf16>& src) {
        const size_t simd_width = 16; // AVX512 processes 16 floats at once
        const size_t N = src.size();

        // Process vectorized chunks
        size_t i = 0;
        for (; i + simd_width <= N; i += simd_width) {
            // Load 16 bfloat16 values and convert to float32
            __m512 src_fp32 = loadu_bf16x16_to_ps(&src[i]);

            // Apply GELU activation
            __m512 gelu_result = gelu_avx512(src_fp32);

            // Convert back to bfloat16 and store
            __m256i result_bf16 = fp32_to_bf16_avx512(gelu_result);
            _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
        }

        // Handle remaining elements (scalar processing)
        for (; i < N; i++) {
            float val_fp32 = bf16_to_fp32(src[i]);
            float gelu_val = 0.5f * val_fp32 * std::erfc(-val_fp32 * 0.70710678118654752f);
            dest[i] = fp32_to_bf16(gelu_val);
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

    /// LayerNorm of `rows` rows of an fp32 residual stream into bf16. The residual
    /// stays fp32 because it carries large outlier channels: in bf16, adding a
    /// layer's small update to them rounds most of the update away. The variance
    /// is two-pass for the same reason.
    void apply_layernorm_f32(buffer<bf16>& dest, const float* src, size_t rows, buffer<bf16>& weight, buffer<bf16>& bias, float eps){
        const int D = weight.size();
        #pragma omp parallel for schedule(static) if (rows > 1)
        for (size_t r = 0; r < rows; r++){
            const float* x = src + r * D;
            bf16* y = dest.data() + r * D;
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

    inline void softmax_bf16_avx512(bf16* output, const bf16* input, int num_heads, int L_padded, int L_MAX) {
        constexpr int simd_width = 16;
        const float neg_inf = std::numeric_limits<float>::lowest();

        for (int head = 0; head < num_heads; ++head) {
            const bf16* src = input + head * L_MAX;
            bf16* dst = output + head * L_MAX;

            // --- Pass 1: Compute max for numerical stability ---
            __m512 max_vec = _mm512_set1_ps(neg_inf);
            int idx = 0;
            for (; idx + simd_width <= L_padded; idx += simd_width) {
                __m512 vals = loadu_bf16x16_to_ps(src + idx);
                max_vec = _mm512_max_ps(max_vec, vals);
            }
            float max_val = _mm512_reduce_max_ps(max_vec);
            for (; idx < L_padded; ++idx) {
                max_val = std::max(max_val, bf16_to_fp32(src[idx]));
            }

            // --- Pass 2: Compute sum of exp(x - max) ---
            double sum = 0.0;
            idx = 0;
            const __m512 max_broadcast = _mm512_set1_ps(max_val);
            for (; idx + simd_width <= L_padded; idx += simd_width) {
                __m512 vals = loadu_bf16x16_to_ps(src + idx);
                __m512 shifted = _mm512_sub_ps(vals, max_broadcast);
                __m512 exp_vals = _mm512_exp_ps_corrected(shifted);
                sum += _mm512_reduce_add_ps(exp_vals);
            }
            for (; idx < L_padded; ++idx) {
                sum += std::exp(bf16_to_fp32(src[idx]) - max_val);
            }

            // Avoid divide by zero / nan
            if (!std::isfinite(sum) || sum <= 0.0) {
                sum = 1.0;
            }

            const float inv_sum = 1.0f / sum;
            const __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);

            // --- Pass 3: Write out normalized softmax values ---
            idx = 0;
            for (; idx + simd_width <= L_padded; idx += simd_width) {
                __m512 vals = loadu_bf16x16_to_ps(src + idx);
                __m512 shifted = _mm512_sub_ps(vals, max_broadcast);
                __m512 exp_vals = _mm512_exp_ps_corrected(shifted);
                __m512 normalized = _mm512_mul_ps(exp_vals, inv_sum_vec);
                bf16pack packed = pack_ps_to_bf16x16(normalized);
                storeu_bf16x16(dst + idx, packed);
            }
            for (; idx < L_padded; ++idx) {
                float val = std::exp(bf16_to_fp32(src[idx]) - max_val) * inv_sum;
                dst[idx] = fp32_to_bf16(val);
            }
        }
    }

    ///@brief Scale and mask bf16 scores
    ///@note scores_bf16: (num_heads, 1, L_MAX)
    ///@note scaling_bf16: scaling factor
    ///@note num_heads: number of attention heads
    ///@note L_MAX: L_MAX, the dimension of k cache
    ///@note context_length: context length
    ///@param scores_bf16: scores_bf16
    ///@param scaling_bf16: scaling factor
    void scale_and_mask_bf16_avx512(bf16* dest, bf16* scores_bf16, float scaling_bf16, int num_heads, int L_MAX, int context_length, int L_padded) {
        constexpr int stride = 16;
        const float neg_inf = -std::numeric_limits<float>::infinity();
        assert(L_MAX % stride == 0);
        alignas(64) float mask_buf[16];

        // Convert scalar bf16 to fp32 - FIXED: use scaling_bf16 instead of scores_bf16
        __m512 vscale = _mm512_set1_ps(scaling_bf16);  // Use the scaling factor
        __m512 vone   = _mm512_set1_ps(1.0f);
        __m512 vinf   = _mm512_set1_ps(neg_inf);

        for (int h = 0; h < num_heads; ++h) {
            bf16* head_bf16 = scores_bf16 + h * L_MAX;
            bf16* head_bf16_dest = dest + h * L_MAX;
            for (int i = 0; i < L_padded; i += stride) {
                // Load 16 bf16 values → fp32
                __m512 x = loadu_bf16x16_to_ps(head_bf16 + i);

                // x *= scaling
                x = _mm512_mul_ps(x, vscale);

                // Generate mask vector: +1.0 for valid, -inf for padded
                for (int j = 0; j < stride; ++j)
                    mask_buf[j] = (i + j) <= context_length ? 1.0f : neg_inf;

                __m512 vmask = _mm512_load_ps(mask_buf);

                // Add mask
                x = _mm512_add_ps(x, vmask);

                // Convert back to bf16 and store
                fp32_to_bf16_store(reinterpret_cast<uint16_t*>(head_bf16_dest + i), x);
            }
        }
    }

    // Copy 1 * DV -> DV * L_MAX array
    void move_v(bf16* dst, const bf16* src, int context_length, int DV, int L_MAX) {
        bf16* dst_ptr = dst + context_length;

        for (int i = 0; i < DV; i++) {
            *dst_ptr = src[i];
            dst_ptr += L_MAX;
        }
    }

    /// dst[j][i] = src[i][j] for i < L, j < DV (dst row stride L_MAX). Done in 64x64 blocks so both
    /// sides stay in cache; the caller runs it from several threads (one slice of rows each).
    void transpose_v_cache(bf16* dst, const bf16* src, int L, int DV, int L_MAX, int i_begin = 0, int i_end = -1) {
        constexpr int B = 64;
        if (i_end < 0) i_end = L;
        for (int i0 = i_begin; i0 < i_end; i0 += B) {
            const int i1 = std::min(i0 + B, i_end);
            for (int j0 = 0; j0 < DV; j0 += B) {
                const int j1 = std::min(j0 + B, DV);
                for (int i = i0; i < i1; i++) {
                    for (int j = j0; j < j1; j++) {
                        dst[(size_t)j * L_MAX + i] = src[(size_t)i * DV + j];
                    }
                }
            }
        }
    }

    void load_all_decoder_proj_weights(buffer<u8>& w, Q4NX& q4nx, int layer_idx){
        u8* w_ptr = w.data();
        u8* w_ptr_end = w.end();
        {
            LOG_VERBOSE(1, "Loading self attn qkv weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_q, "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
            q4nx.load_weights(w_k, "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
            q4nx.load_weights(w_v, "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");
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
            LOG_VERBOSE(1, "Loading self attn o weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_o;
            q4nx.load_weights(w_o, "model.decoder.layers." + std::to_string(layer_idx) + ".self_attn.out_proj.weight");
            memcpy(w_ptr, w_o.data(), D * D * 5 / 8);
            w_ptr += D * D * 5 / 8;
        }
        // read encoder attn q weights
        {
            LOG_VERBOSE(1, "Loading encoder attn q weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_encoder_attn_q;
            q4nx.load_weights(w_encoder_attn_q, "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.q_proj.weight");
            LOG_VERBOSE(1, "Copying q weights to buffer");
            memcpy(w_ptr, w_encoder_attn_q.data(), DQ * D * 5 / 8);
            w_ptr += DQ * D * 5 / 8;
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading encoder attn o weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_encoder_attn_o;
            q4nx.load_weights(w_encoder_attn_o, "model.decoder.layers." + std::to_string(layer_idx) + ".encoder_attn.out_proj.weight");
            memcpy(w_ptr, w_encoder_attn_o.data(), D * D * 5 / 8);
            w_ptr += D * D * 5 / 8;
        }

        // read up weights
        {
            LOG_VERBOSE(1, "Loading up weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_up;
            q4nx.load_weights(w_up, "model.decoder.layers." + std::to_string(layer_idx) + ".fc1.weight");
            memcpy(w_ptr, w_up.data(), intermediate_size * D * 5 / 8);
            w_ptr += intermediate_size * D * 5 / 8;
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading down weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_down;
            q4nx.load_weights(w_down, "model.decoder.layers." + std::to_string(layer_idx) + ".fc2.weight");
            memcpy(w_ptr, w_down.data(), intermediate_size * D * 5 / 8);
            w_ptr += intermediate_size * D * 5 / 8;
        }

        if (w_ptr != w_ptr_end){
            std::cout << "Warning: w_ptr != w_ptr_end" << std::endl;
            exit(1);
        }
    }

    void prepare_proj(npu_sequence* seq, const uint32_t D_in, const uint32_t D_out, const uint32_t weight_offset);
    void prepare_qkt(npu_sequence* seq, const uint32_t L_padded, const uint32_t DK, const uint32_t L_MAX, const uint32_t m, const uint32_t k, const uint32_t batch_size, const uint32_t weight_offset);
    void prepare_sv(npu_sequence* seq, const uint32_t DV, const uint32_t L_MAX, const uint32_t L_padded, const uint32_t m, const uint32_t k, const uint32_t batch_size, const uint32_t weight_offset);
    inline int _ceil_to_multiple(int value, int multiple) {
        assert(multiple > 0 && "multiple must be positive");
        return ((value + multiple - 1) / multiple) * multiple;
    }

    Whisper_Config config;

    int context_length;

    std::unique_ptr<Embedding> embedding;

    npu_xclbin_manager *npu;
    npu_app_manager* layer_app_manager;
    npu_app proj_self_attn_qkv;
    npu_app proj_self_attn_o;
    npu_app proj_cross_attn_q;
    npu_app proj_cross_attn_o;
    npu_app self_attn_qkT;
    npu_app self_attn_sv;
    npu_app cross_attn_qkT;
    npu_app cross_attn_sv;
    npu_app proj_up;
    npu_app proj_down;

    buffer<bf16> positional_embedding;
    buffer<bf16> decoder_out;
    buffer<bf16> residual_buffer;
    std::vector<float> residual_f32;  // the residual stream, kept in fp32
    buffer<bf16> hidden_states_buffer;

    buffer<bf16> buf_q_o_d; // shared by q_proj, o_proj, down_proj
    buffer<bf16> buf_qkv;
    buffer<bf16> buf_q;
    buffer<bf16> buf_k;
    buffer<bf16> buf_v;
    buffer<bf16> buf_self_attn_score;
    buffer<bf16> buf_cross_attn_score;
    buffer<bf16> buf_up;
    buffer<bf16> buf_x;
    std::vector<buffer<bf16>> self_attn_k_cache;
    std::vector<buffer<bf16>> self_attn_v_cache;
    std::vector<buffer<bf16>> cross_attn_k_cache;
    std::vector<buffer<bf16>> cross_attn_v_cache;

    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> self_attn_norm_weights;
    std::vector<buffer<bf16>> self_attn_norm_biases;
    std::vector<buffer<bf16>> cross_attn_norm_weights;
    std::vector<buffer<bf16>> cross_attn_norm_biases;
    std::vector<buffer<bf16>> mlp_norm_weights;
    std::vector<buffer<bf16>> mlp_norm_biases;
    std::vector<buffer<bf16>> self_attn_q_biases;
    std::vector<buffer<bf16>> self_attn_v_biases;
    std::vector<buffer<bf16>> self_attn_o_biases;
    std::vector<buffer<bf16>> cross_attn_q_biases;
    std::vector<buffer<bf16>> cross_attn_o_biases;
    std::vector<buffer<bf16>> fc1_biases;
    std::vector<buffer<bf16>> fc2_biases;

    buffer<bf16> decoder_layer_norm_weights;
    buffer<bf16> decoder_layer_norm_biases;

    tensor_2d<bf16> positional_embedding_tensor;

public:
    WhisperDecoder(Whisper_Config config, npu_xclbin_manager *npu);
    ~WhisperDecoder();

    void load_weights(Q4NX& q4nx);
    void load_cross_attn_kv_cache(buffer<bf16>& kv_cache);
    buffer<bf16> forward(int ids);

    /// final LayerNorm of the fp32 residual left by forward()
    void final_layernorm(buffer<bf16>& dest, buffer<bf16>& weight, buffer<bf16>& bias){
        apply_layernorm_f32(dest, residual_f32.data(), 1, weight, bias, 1e-5f);
    }

    void set_context_length(int context_length) {
        this->context_length = context_length;
        for (int i = 0; i < this->self_attn_k_cache.size(); i++){
            this->self_attn_k_cache[i].memset(bf16(0));
            this->self_attn_v_cache[i].memset(bf16(0));
        }
    }

    void apply_layernorm_avx(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& weight, buffer<bf16>& bias, float eps) {
        assert(weight.size() == bias.size());

        const int D = weight.size();
        const int simd_width = 8;

        bf16* x_ptr = src.data();
        bf16* y_ptr = dest.data();

        // Step 1: compute mean and variance
        __m256 sum = _mm256_setzero_ps();
        __m256 sumsq = _mm256_setzero_ps();
        for (int i = 0; i < D; i += simd_width) {
            __m128i x_bf16 = _mm_loadu_si128((__m128i*)&x_ptr[i]);
            __m256 x_fp32 = bf16o_fp32(x_bf16);
            sum = _mm256_add_ps(sum, x_fp32);
            sumsq = _mm256_fmadd_ps(x_fp32, x_fp32, sumsq);
        }

        float mean = hsum256(sum) / D;
        float var = hsum256(sumsq) / D - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        __m256 mean_v = _mm256_set1_ps(mean);
        __m256 inv_std_v = _mm256_set1_ps(inv_std);

        // Step 2: normalize and apply affine
        for (int i = 0; i < D; i += simd_width) {
            __m128i x_bf16 = _mm_loadu_si128((__m128i*)&x_ptr[i]);
            __m128i w_bf16 = _mm_loadu_si128((__m128i*)&weight[i]);
            __m128i b_bf16 = _mm_loadu_si128((__m128i*)&bias[i]);
            __m256 w_fp32 = bf16o_fp32(w_bf16);
            __m256 b_fp32 = bf16o_fp32(b_bf16);
            __m256 x_fp32 = bf16o_fp32(x_bf16);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_fp32, mean_v), inv_std_v);
            __m256 y_fp32 = _mm256_fmadd_ps(norm, w_fp32, b_fp32);
            __m128i y_bf16 = f32o_bf16(y_fp32);
            _mm_storeu_si128((__m128i*)&y_ptr[i], y_bf16);
        }
    }
};
