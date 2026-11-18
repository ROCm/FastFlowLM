// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file cpu_functions.hpp
/// \brief host reference kernels for the hunyuan-dense engine
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \note Two of these deliberately disagree with the equivalent kernels in the
///       other engines, because the architecture does:
///         * _rms_norm casts x * inv_rms back to bf16 *before* multiplying the
///           weight, which is what HunYuanRMSNorm does;
///         * _rope_then_qk_rms_batch rotates first and normalizes second, the
///           opposite of every QK-norm model in the tree.
///       The q4_k / q8_0 readers below invert hunyuan_desc::reorder_cpy, so the host
///       matmul consumes exactly the bytes the array would have consumed.
#ifndef __HUNYUAN_CPU_FUNCTIONS_HPP__
#define __HUNYUAN_CPU_FUNCTIONS_HPP__

#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "weight_desc.hpp"

static constexpr int max_prefill_thread = 6;

namespace cpu_func {

    // -----------------------------------------------------------------------
    // elementwise / norm
    // -----------------------------------------------------------------------

    void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual, size_t length){
        bf16*       dest_base = dest.data();
        const bf16* src_base  = src.data();
        const bf16* res_base  = residual.data();

        const size_t simd_width = 16;
        const size_t simd_end = length - (length % simd_width);

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int i = 0; i < (int)simd_end; i += simd_width) {
            __m256i bf16_src = _mm256_loadu_si256((const __m256i*)(src_base + i));
            __m256i bf16_res = _mm256_loadu_si256((const __m256i*)(res_base + i));
            __m512 result    = _mm512_add_ps(bf16o_fp32_512(bf16_src), bf16o_fp32_512(bf16_res));
            _mm256_storeu_si256((__m256i*)(dest_base + i), f32o_bf16_512(result));
        }

        for (size_t i = simd_end; i < length; i++) {
            dest_base[i] = static_cast<bf16>(static_cast<float>(src_base[i]) + static_cast<float>(res_base[i]));
        }
    }

    /// @brief 1 / rms of one row of _rms_norm
    inline f32 _rms_inv(const bf16* x_row, int D, f32 eps){
        const int simd_width = 16;
        __m512 sum_xx_vec = _mm512_setzero_ps();
        int i = 0;
        for (; i + simd_width <= D; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x_row + i));
            __m512 fp32_vals  = bf16o_fp32_512(bf16_vals);
            sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
        }
        f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
        for (; i < D; i++) { f32 v = static_cast<f32>(x_row[i]); sum_xx += v * v; }

        return 1.0f / sqrtf(sum_xx / (f32)D + eps);
    }

    /// @brief n columns of one row of _rms_norm (x_row, w_ptr, y_row point at the first one)
    inline void _rms_norm_cols(bf16* y_row, const bf16* x_row, const bf16* w_ptr, f32 inv_rms_x, int n){
        const int simd_width = 16;
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms_x);
        int i = 0;
        for (; i + simd_width <= n; i += simd_width) {
            __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + i));
            __m256i bf16_w = _mm256_loadu_si256((const __m256i*)(w_ptr + i));
            // round to bf16 first, then apply the weight
            __m512 normed  = _mm512_mul_ps(bf16o_fp32_512(bf16_x), inv_rms_vec);
            __m512 rounded = bf16o_fp32_512(f32o_bf16_512(normed));
            __m512 y_vec   = _mm512_mul_ps(bf16o_fp32_512(bf16_w), rounded);
            _mm256_storeu_si256((__m256i*)(y_row + i), f32o_bf16_512(y_vec));
        }
        for (; i < n; i++) {
            bf16 normed = static_cast<bf16>(static_cast<f32>(x_row[i]) * inv_rms_x);
            y_row[i] = static_cast<bf16>(static_cast<f32>(w_ptr[i]) * static_cast<f32>(normed));
        }
    }

    /// @brief RMSNorm, HunYuan flavour
    /// @note  out = w * (bf16)(x * rsqrt(mean(x^2) + eps)). The cast back to
    ///        bf16 happens *before* the weight multiply -- see
    ///        models_simple/HUNYUAN_V1_DENSE.md, HunYuanRMSNorm.forward.
    void _rms_norm(bf16* y, const bf16* x, const bf16* w_ptr, int D, int L, int L_offset_dest, int L_offset_input, f32 eps){
        const bf16* x_base = x + (size_t)L_offset_input * D;
        bf16*       y_base = y + (size_t)L_offset_dest  * D;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* x_row = x_base + (size_t)l * D;
            _rms_norm_cols(y_base + (size_t)l * D, x_row, w_ptr, _rms_inv(x_row, D, eps), D);
        }
    }

    void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, const bf16* w_ptr, int D, int L, int L_offset_dest, int L_offset_input, f32 eps){
        assert((size_t)(L_offset_input + L) * D <= x.size() && "x buffer overflow in _rms_norm");
        assert((size_t)(L_offset_dest  + L) * D <= y.size() && "y buffer overflow in _rms_norm");
        _rms_norm(y.data(), x.data(), w_ptr, D, L, L_offset_dest, L_offset_input, eps);
    }

    void _elementwise_mul(bf16* dst, bf16* src_0, bf16* src_1, int length){
        const int simd_width = 16;
        int i = 0;
        for (; i + simd_width <= length; i += simd_width){
            __m256i a_vals = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i b_vals = _mm256_loadu_si256((const __m256i*)(src_1 + i));
            __m512 result_vec = _mm512_mul_ps(bf16o_fp32_512(a_vals), bf16o_fp32_512(b_vals));
            _mm256_storeu_si256((__m256i*)(dst + i), f32o_bf16_512(result_vec));
        }
        for (; i < length; i++){
            dst[i] = static_cast<bf16>(static_cast<float>(src_0[i]) * static_cast<float>(src_1[i]));
        }
    }

    void _f32_to_bf16(bf16* dst, const float* src, int length){
        const int simd_width = 16;
        int i = 0;
        for (; i + simd_width <= length; i += simd_width){
            __m512 a_vals = _mm512_loadu_ps(src + i);
            _mm256_storeu_si256((__m256i*)(dst + i), f32o_bf16_512(a_vals));
        }
        for (; i < length; i++){
            dst[i] = static_cast<bf16>(src[i]);
        }
    }

    /// @brief fused SwiGLU where up and gate are interleaved along an inner axis
    /// @note  up_gate layout: (L, intermediate_size / slice_size, 2, slice_size),
    ///        the "2" axis is 0 = up, 1 = gate; dst is (L, intermediate_size).
    ///        slice_size must match hunyuan_desc::UP_GATE_OUTDIM_REORDEER, which
    ///        is how load_layer_weights interleaved the two projections.
    void _exp_glu(bf16* dst, const bf16* up_gate, int L, int intermediate_size, int slice_size) {
        const int simd_width = 16;
        const int num_slices = intermediate_size / slice_size;
        const int slice_stride = 2 * slice_size;

        const __m512 v_one = _mm512_set1_ps(1.0f);

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* ug_row  = up_gate + (size_t)l * intermediate_size * 2;
            bf16*       dst_row = dst     + (size_t)l * intermediate_size;

            for (int s = 0; s < num_slices; s++) {
                const bf16* up_slice   = ug_row + (size_t)s * slice_stride;
                const bf16* gate_slice = up_slice + slice_size;
                bf16*       out_slice  = dst_row + (size_t)s * slice_size;

                int i = 0;
                for (; i + simd_width <= slice_size; i += simd_width) {
                    __m512 up_v   = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(up_slice   + i)));
                    __m512 gate_v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(gate_slice + i)));

                    // silu(gate) = gate / (1 + exp(-gate))
                    __m512 neg_gate = _mm512_sub_ps(_mm512_setzero_ps(), gate_v);
                    __m512 denom    = _mm512_add_ps(v_one, _mm512_exp_ps_corrected(neg_gate));
                    __m512 silu     = _mm512_div_ps(gate_v, denom);

                    __m512 result = _mm512_mul_ps(up_v, silu);
                    _mm256_storeu_si256((__m256i*)(out_slice + i), f32o_bf16_512(result));
                }
                for (; i < slice_size; i++) {
                    float up_val   = static_cast<float>(up_slice[i]);
                    float gate_val = static_cast<float>(gate_slice[i]);
                    float silu     = gate_val / (1.0f + expf(-gate_val));
                    out_slice[i]   = static_cast<bf16>(up_val * silu);
                }
            }
        }
    }

    /// @brief output columns [col0, col0 + n) of one row of _glu_mul; the n columns must
    ///        lie inside one slice
    inline void _glu_mul_cols(bf16* out, const bf16* ug_row, int col0, int n, int slice_size) {
        const int simd_width = 16;
        const bf16* up_slice   = ug_row + (size_t)(col0 / slice_size) * 2 * slice_size + col0 % slice_size;
        const bf16* gate_slice = up_slice + slice_size;

        int i = 0;
        for (; i + simd_width <= n; i += simd_width) {
            __m512 up_v   = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(up_slice   + i)));
            __m512 gate_v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(gate_slice + i)));
            _mm256_storeu_si256((__m256i*)(out + i),
                                f32o_bf16_512(_mm512_mul_ps(up_v, gate_v)));
        }
        for (; i < n; i++) {
            out[i] = static_cast<bf16>(static_cast<float>(up_slice[i]) *
                                       static_cast<float>(gate_slice[i]));
        }
    }

    /// @brief the tail of _exp_glu for when the array already applied the silu
    /// @note  Same up_gate layout and same output as _exp_glu; the only difference
    ///        is that the gate slabs are read as-is instead of being passed through
    ///        silu first. Pair this with the array's up|gate run, whose right half
    ///        (the gate slabs) gets the silu -- see hunyuan_mlp_prefill_context::forward.
    ///
    ///        Note the array computes silu in bf16 and rounds the result before
    ///        this multiply, where _exp_glu keeps the whole expression in fp32, so
    ///        the two are close but not bit-identical.
    void _glu_mul(bf16* dst, const bf16* up_gate, int L, int intermediate_size, int slice_size) {
        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* ug_row  = up_gate + (size_t)l * intermediate_size * 2;
            bf16*       dst_row = dst     + (size_t)l * intermediate_size;
            for (int s = 0; s < intermediate_size / slice_size; s++) {
                _glu_mul_cols(dst_row + (size_t)s * slice_size, ug_row, s * slice_size, slice_size, slice_size);
            }
        }
    }

    // -----------------------------------------------------------------------
    // rope + qk norm
    // -----------------------------------------------------------------------

    /// @brief RoPE over a whole head, then the per-head Q/K RMS norm
    /// @param tensor  L x D activations, D = num_heads * DH
    /// @param pos     absolute position of each of the L tokens
    /// @param norm_w  the DH-element q_norm or k_norm weight
    /// @note  Order matters: HunYuanAttention applies rotary embeddings and only
    ///        then query_layernorm / key_layernorm. Rotation is the full head
    ///        (rotate_half over DH/2 + DH/2), not a partial-rope prefix.
    void _rope_then_qk_rms_batch(buffer<bf16>& tensor, int D, const std::vector<int>& pos, int L,
                                 int num_heads, const bf16* norm_w, int DH,
                                 const f32* inv_freq, f32 eps){
        const int half = DH / 2;
        const int simd_width = 16;

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int ll = 0; ll < L; ll++){
            // cos/sin are computed in fp32: bf16 sin/cos at long positions loses
            // meaningful precision.
            std::vector<float> local_cos(half);
            std::vector<float> local_sin(half);
            const float p = (float)pos[ll];
            for (int j = 0; j < half; j++){
                float angle = inv_freq[j] * p;
                local_cos[j] = cosf(angle);
                local_sin[j] = sinf(angle);
            }

            bf16* x_token = tensor.data() + (size_t)ll * D;
            for (int h = 0; h < num_heads; h++){
                bf16* x_head = x_token + (size_t)h * DH;

                // --- rope: q = q * cos + rotate_half(q) * sin ---------------
                bf16* x_left  = x_head;
                bf16* x_right = x_head + half;
                int i = 0;
                for (; i + simd_width <= half; i += simd_width) {
                    __m512 Lv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_left  + i)));
                    __m512 Rv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_right + i)));
                    __m512 C  = _mm512_loadu_ps(local_cos.data() + i);
                    __m512 S  = _mm512_loadu_ps(local_sin.data() + i);
                    __m512 newL = _mm512_sub_ps(_mm512_mul_ps(Lv, C), _mm512_mul_ps(Rv, S));
                    __m512 newR = _mm512_add_ps(_mm512_mul_ps(Lv, S), _mm512_mul_ps(Rv, C));
                    _mm256_storeu_si256((__m256i*)(x_left  + i), f32o_bf16_512(newL));
                    _mm256_storeu_si256((__m256i*)(x_right + i), f32o_bf16_512(newR));
                }
                for (; i < half; i++) {
                    float lv = static_cast<float>(x_left[i]);
                    float rv = static_cast<float>(x_right[i]);
                    x_left[i]  = static_cast<bf16>(lv * local_cos[i] - rv * local_sin[i]);
                    x_right[i] = static_cast<bf16>(lv * local_sin[i] + rv * local_cos[i]);
                }

                // --- then the per-head rms norm -----------------------------
                __m512 sum_xx_vec = _mm512_setzero_ps();
                i = 0;
                for (; i + simd_width <= DH; i += simd_width) {
                    __m512 fp32_vals = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_head + i)));
                    sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
                }
                f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
                for (; i < DH; i++) { f32 v = static_cast<f32>(x_head[i]); sum_xx += v * v; }
                f32 inv_rms = 1.0f / sqrtf(sum_xx / (f32)DH + eps);

                __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);
                i = 0;
                for (; i + simd_width <= DH; i += simd_width) {
                    __m512 xv      = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_head + i)));
                    __m512 wv      = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(norm_w + i)));
                    __m512 rounded = bf16o_fp32_512(f32o_bf16_512(_mm512_mul_ps(xv, inv_rms_vec)));
                    _mm256_storeu_si256((__m256i*)(x_head + i), f32o_bf16_512(_mm512_mul_ps(wv, rounded)));
                }
                for (; i < DH; i++) {
                    bf16 normed = static_cast<bf16>(static_cast<f32>(x_head[i]) * inv_rms);
                    x_head[i] = static_cast<bf16>(static_cast<f32>(norm_w[i]) * static_cast<f32>(normed));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // weight readers
    //
    // A q4nx q4_k chunk covers 32 output rows x 256 input columns in 4736 B,
    // column major (FLM_Q4NX_Converter _pack_q4k):
    //   [0, 256)     uint8 s[8 groups][32 rows]
    //   [256, 512)   uint8 m[8 groups][32 rows]
    //   [512, 4608)  nibbles, byte col * 16 + row / 2, low = even row, unsigned
    //   [4608, 4672) bf16 S[32 rows]
    //   [4672, 4736) bf16 M[32 rows], stored negated
    //   w = S s_g q + M m_g
    // The lm_head is Q8_0, and each 32 x 256 chunk is stored as a pair of
    // 32 x 128 halves, each padded to one q4_k chunk (see
    // hunyuan_desc::reorder_cpy_lm_head): [128 bf16 scales][4096 int8], w = s q.
    // Its interleave runs over the halves (half h of column block c is half
    // block 2c + h), so a band pair reads [b0 h0][b1 h0][b0 h1][b1 h1].
    // Chunks arrive row-band major, then hunyuan_desc::reorder_cpy permutes them
    // for the array's shared weight streams. _block_index() below inverts
    // that permutation.
    // -----------------------------------------------------------------------

    static constexpr int Q4K_CHUNK_BYTES = 4736;
    static constexpr int Q80_PAIR_CHUNK_BYTES = 2 * Q4K_CHUNK_BYTES;
    enum class wq_kind { q4k, q80_pair };

    /// @brief where reorder_cpy() put the chunk for (row band, column block)
    inline size_t _block_index(int band, int cblk, int blocks_per_row, int V) {
        return (size_t)(band / V) * V * blocks_per_row + (size_t)cblk * V + (band % V);
    }

    /// @brief dequantize one q4_k chunk into a 32 x stride float tile at column cblk*256
    inline void _q4k_dequant_chunk(float* dst, size_t stride, const u8* chunk, int cblk) {
        constexpr int R = QXNX_ROW_BLOCK_SIZE;
        constexpr int G = QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
        const u8*   s8     = chunk;
        const u8*   m8     = chunk + G * R;
        const u8*   quants = chunk + 2 * G * R;
        const bf16* S      = (const bf16*)(quants + QXNX_COL_BLOCK_SIZE * R / 2);
        const bf16* M      = S + R;
        float* dst_base = dst + (size_t)cblk * QXNX_COL_BLOCK_SIZE;

        float scale[G][R];
        float minv[G][R];
        for (int g = 0; g < G; g++) {
            for (int r = 0; r < R; r++) {
                scale[g][r] = static_cast<float>(S[r]) * (float)s8[g * R + r];
                minv[g][r]  = static_cast<float>(M[r]) * (float)m8[g * R + r];
            }
        }

        for (int c = 0; c < QXNX_COL_BLOCK_SIZE; c++) {
            const int g = c / GGML_GROUP_SIZE;
            const u8* col = quants + (size_t)c * (R / 2);
            for (int r = 0; r < R; r += 2) {
                const u8 b = col[r >> 1];
                dst_base[(size_t)r       * stride + c] = scale[g][r]     * (float)(b & 0x0F) + minv[g][r];
                dst_base[(size_t)(r + 1) * stride + c] = scale[g][r + 1] * (float)(b >> 4)   + minv[g][r + 1];
            }
        }
    }

    /// @brief dequantize one lm_head chunk (two padded Q8_0 halves) into a 32 x stride float tile
    inline void _q80_pair_dequant_chunk(float* dst, size_t stride, const u8* const halves[2], int cblk) {
        constexpr int R = QXNX_ROW_BLOCK_SIZE;
        constexpr int HALF_COLS = QXNX_COL_BLOCK_SIZE / 2;
        float* dst_base = dst + (size_t)cblk * QXNX_COL_BLOCK_SIZE;

        for (int half = 0; half < 2; half++) {
            const u8*     base   = halves[half];
            const bf16*   scales = (const bf16*)base;
            const int8_t* q      = (const int8_t*)(base + (HALF_COLS / GGML_GROUP_SIZE) * R * sizeof(bf16));
            for (int c = 0; c < HALF_COLS; c++) {
                const bf16* s = scales + (size_t)(c / GGML_GROUP_SIZE) * R;
                for (int r = 0; r < R; r++) {
                    dst_base[(size_t)r * stride + half * HALF_COLS + c] =
                        static_cast<float>(s[r]) * (float)q[(size_t)c * R + r];
                }
            }
        }
    }

    /// @brief y[L, Dout] = x[L, Din] * W^T, W read straight out of the reordered
    ///        slab the array would have read.
    /// @param slice_out  the output width each independently reordered slab
    ///                   covers. Dout for a plain projection; 512 for the
    ///                   interleaved up/gate slab, where slab j alternates up
    ///                   and gate and lands at output columns j*512..(j+1)*512.
    /// @param V          the reorder_cpy vertical_blocks the slab was written with
    void _dequant_mm(bf16* y, const u8* w_base, const bf16* x, int L, int Din, int Dout,
                     int slice_out, int V, wq_kind kind = wq_kind::q4k) {
        assert(Din % QXNX_COL_BLOCK_SIZE == 0);
        assert(Dout % QXNX_ROW_BLOCK_SIZE == 0);
        assert(slice_out % QXNX_ROW_BLOCK_SIZE == 0);
        assert(Dout % slice_out == 0);

        const size_t chunk_bytes  = kind == wq_kind::q4k ? Q4K_CHUNK_BYTES : Q80_PAIR_CHUNK_BYTES;
        const int blocks_per_row  = Din / QXNX_COL_BLOCK_SIZE;
        const int bands_per_slice = slice_out / QXNX_ROW_BLOCK_SIZE;
        const int bands           = Dout / QXNX_ROW_BLOCK_SIZE;
        const size_t slice_bytes  = (size_t)bands_per_slice * blocks_per_row * chunk_bytes;

        #pragma omp parallel num_threads(max_prefill_thread)
        {
            std::vector<float> wbuf((size_t)QXNX_ROW_BLOCK_SIZE * Din);

            #pragma omp for schedule(static)
            for (int band = 0; band < bands; band++) {
                const int slice       = band / bands_per_slice;
                const int local_band  = band % bands_per_slice;
                const u8* slab        = w_base + (size_t)slice * slice_bytes;

                for (int cblk = 0; cblk < blocks_per_row; cblk++) {
                    if (kind == wq_kind::q4k) {
                        const u8* chunk = slab + _block_index(local_band, cblk, blocks_per_row, V) * chunk_bytes;
                        _q4k_dequant_chunk(wbuf.data(), Din, chunk, cblk);
                    } else {
                        const u8* halves[2];
                        for (int half = 0; half < 2; half++) {
                            const size_t hi = _block_index(local_band, 2 * cblk + half, 2 * blocks_per_row, V);
                            halves[half] = slab + hi * Q4K_CHUNK_BYTES;
                        }
                        _q80_pair_dequant_chunk(wbuf.data(), Din, halves, cblk);
                    }
                }

                // 32 output columns at a time, four dot products in flight
                const int n_base = band * QXNX_ROW_BLOCK_SIZE;
                for (int l = 0; l < L; l++) {
                    const bf16* x_row = x + (size_t)l * Din;
                    bf16*       y_row = y + (size_t)l * Dout + n_base;
                    for (int r = 0; r < QXNX_ROW_BLOCK_SIZE; r += 4) {
                        const float* w0 = wbuf.data() + (size_t)(r + 0) * Din;
                        const float* w1 = wbuf.data() + (size_t)(r + 1) * Din;
                        const float* w2 = wbuf.data() + (size_t)(r + 2) * Din;
                        const float* w3 = wbuf.data() + (size_t)(r + 3) * Din;
                        __m512 acc0 = _mm512_setzero_ps();
                        __m512 acc1 = _mm512_setzero_ps();
                        __m512 acc2 = _mm512_setzero_ps();
                        __m512 acc3 = _mm512_setzero_ps();
                        for (int k = 0; k < Din; k += 16) {
                            __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + k)));
                            acc0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w0 + k), acc0);
                            acc1 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w1 + k), acc1);
                            acc2 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w2 + k), acc2);
                            acc3 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w3 + k), acc3);
                        }
                        y_row[r + 0] = static_cast<bf16>(_mm512_reduce_add_ps(acc0));
                        y_row[r + 1] = static_cast<bf16>(_mm512_reduce_add_ps(acc1));
                        y_row[r + 2] = static_cast<bf16>(_mm512_reduce_add_ps(acc2));
                        y_row[r + 3] = static_cast<bf16>(_mm512_reduce_add_ps(acc3));
                    }
                }
            }
        }
    }

    /// @brief plain projection: one reordered slab covering all of Dout
    inline void _dequant_mm(bf16* y, const u8* w_base, const bf16* x, int L, int Din, int Dout) {
        _dequant_mm(y, w_base, x, L, Din, Dout, Dout, 2);
    }

    /// @brief the lm_head: Q8_0 half pairs, reordered like a plain projection
    inline void _dequant_mm_lm_head(bf16* y, const u8* w_base, const bf16* x, int L, int Din, int Dout) {
        _dequant_mm(y, w_base, x, L, Din, Dout, Dout, 2, wq_kind::q80_pair);
    }

    /// @brief the interleaved up/gate slab; y is L x (2 * intermediate_size),
    ///        laid out exactly as _exp_glu expects.
    inline void _dequant_mm_upgate(bf16* y, const u8* w_base, const bf16* x, int L, int Din,
                                   int intermediate_size, int slice_out) {
        _dequant_mm(y, w_base, x, L, Din, intermediate_size * 2, slice_out, 2);
    }

    // -----------------------------------------------------------------------
    // attention
    // -----------------------------------------------------------------------

    /// @brief causal GQA attention against a flat kv cache
    /// @param out         L x qo_stride
    /// @param q           L x qo_stride, already roped and normed
    /// @param k_cache     kv_len x k_stride, position major
    /// @param v_cache     kv_len x v_stride
    /// @param L_begin     absolute position of the first of the L query tokens
    /// @param qo_stride   row stride of q and out; 0 means num_heads * DH.
    ///                    Pass the full DQ when calling this once per half of a
    ///                    split kv cache, so that q keeps its whole-row stride
    ///                    while num_heads counts only the half's heads.
    /// @note  Query l sees kv positions 0 .. L_begin + l inclusive. There is no
    ///        sliding window and no per-layer mask variation in this
    ///        architecture. Scores and softmax run in fp32.
    void _attention(bf16* out, const bf16* q, const bf16* k_cache, const bf16* v_cache,
                    int L, int L_begin, int num_heads, int num_kv_heads, int DH,
                    int k_stride, int v_stride, float scale, int qo_stride = 0) {
        const int num_kv_groups = num_heads / num_kv_heads;
        const int DQ = (qo_stride > 0) ? qo_stride : num_heads * DH;
        const int kv_max = L_begin + L;

        #pragma omp parallel num_threads(max_prefill_thread)
        {
            std::vector<float> scores(kv_max);
            std::vector<float> acc(DH);

            #pragma omp for schedule(static) collapse(2)
            for (int l = 0; l < L; l++) {
                for (int h = 0; h < num_heads; h++) {
                    const int kv_len = L_begin + l + 1;
                    const int kvh    = h / num_kv_groups;
                    const bf16* q_head = q + (size_t)l * DQ + (size_t)h * DH;

                    float max_score = -INFINITY;
                    for (int t = 0; t < kv_len; t++) {
                        const bf16* k_row = k_cache + (size_t)t * k_stride + (size_t)kvh * DH;
                        __m512 accv = _mm512_setzero_ps();
                        for (int d = 0; d < DH; d += 16) {
                            __m512 qv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(q_head + d)));
                            __m512 kv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(k_row  + d)));
                            accv = _mm512_fmadd_ps(qv, kv, accv);
                        }
                        float s = _mm512_reduce_add_ps(accv) * scale;
                        scores[t] = s;
                        max_score = std::max(max_score, s);
                    }

                    float denom = 0.0f;
                    for (int t = 0; t < kv_len; t++) {
                        scores[t] = expf(scores[t] - max_score);
                        denom += scores[t];
                    }
                    const float inv_denom = 1.0f / denom;

                    std::fill(acc.begin(), acc.end(), 0.0f);
                    for (int t = 0; t < kv_len; t++) {
                        const float p = scores[t] * inv_denom;
                        if (p == 0.0f) continue;
                        const bf16* v_row = v_cache + (size_t)t * v_stride + (size_t)kvh * DH;
                        __m512 pv = _mm512_set1_ps(p);
                        for (int d = 0; d < DH; d += 16) {
                            __m512 vv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(v_row + d)));
                            __m512 av = _mm512_loadu_ps(acc.data() + d);
                            _mm512_storeu_ps(acc.data() + d, _mm512_fmadd_ps(pv, vv, av));
                        }
                    }

                    _f32_to_bf16(out + (size_t)l * DQ + (size_t)h * DH, acc.data(), DH);
                }
            }
        }
    }
}

#endif // __HUNYUAN_CPU_FUNCTIONS_HPP__
