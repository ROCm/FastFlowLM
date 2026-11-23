// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __CPU__FUNCTIONS__HPP__
#define __CPU__FUNCTIONS__HPP__

#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "avx512_util.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/// @brief CPU reference kernels for the SmolVLA towers.
///
/// Every kernel is named with a leading underscore and takes its shapes as
/// explicit arguments; nothing here knows about SmolVLA's specific dimensions.
/// All of them accumulate in fp32 and round back to bf16 on store.
namespace cpu_func {

    /// number of fp32 lanes in one AVX-512 vector; every kernel blocks by this.
    static constexpr int simd_width = 16;

    /// OpenMP team size used by the prefill-shaped kernels: up to 8, but never
    /// more than the OpenMP runtime offers (OMP_NUM_THREADS / the CPU affinity
    /// mask), so a 2-core cap runs 2 threads instead of 8 time-sliced ones.
    inline const int max_prefill_thread = std::min(8, omp_get_max_threads());

    /// _cpu_mm's blocked fast path emits this many output columns per pass.
    static constexpr int mm_unroll_n = 4;

    /// Row-strided residual add: dest[l, 0:D] = src[l, 0:D] + residual[l, 0:D] for
    /// L rows, each operand carrying its own row stride (>= D); 0 means D. Columns
    /// past D are left untouched in dest, so a padded destination keeps its
    /// padding. Use the flat overload below when all three strides are D.
    inline void _residual_add(bf16* dest, const bf16* src, const bf16* residual,
                              int L, int D, int dest_stride = 0,
                              int src_stride = 0, int res_stride = 0){
        const int DS = (dest_stride > 0) ? dest_stride : D;
        const int SS = (src_stride  > 0) ? src_stride  : D;
        const int RS = (res_stride  > 0) ? res_stride  : D;
        const int simd_end = D - (D % simd_width);

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            bf16*       d_row = dest     + (size_t)l * DS;
            const bf16* s_row = src      + (size_t)l * SS;
            const bf16* r_row = residual + (size_t)l * RS;
            int i = 0;
            for (; i < simd_end; i += simd_width) {
                __m256i bf16_src = _mm256_loadu_si256((const __m256i*)(s_row + i));
                __m256i bf16_res = _mm256_loadu_si256((const __m256i*)(r_row + i));
                __m512 result    = _mm512_add_ps(bf16o_fp32_512(bf16_src), bf16o_fp32_512(bf16_res));
                _mm256_storeu_si256((__m256i*)(d_row + i), f32o_bf16_512(result));
            }
            for (; i < D; i++) {
                d_row[i] = static_cast<bf16>(static_cast<float>(s_row[i]) + static_cast<float>(r_row[i]));
            }
        }
    }

    inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual, size_t length){
        bf16*       dest_base = dest.data()    ;
        const bf16* src_base  = src.data()     ;
        const bf16* res_base  = residual.data();

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

    // Load `simd_width` weights starting at wp into a __m512, converting from bf16 if needed.
    template <typename WT>
    static inline __m512 _load_w16(const WT* wp) {
        if constexpr (std::is_same_v<WT, float>) {
            return _mm512_loadu_ps(wp);
        } else {
            return bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)wp));
        }
    }

    // Biased LayerNorm over the last dim: for each of L rows of length D,
    //   y = (x - mean) / sqrt(var + eps) * gamma + beta
    // where mean/var are the per-row mean and (biased) variance over D. gamma
    // and beta are shared [D] vectors. x and y are [L, D] bf16 (may alias).
    inline void _layer_norm_biased(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& gamma,
                           buffer<bf16>& beta, int L, int D, float eps){
        bf16*       y_base = y.data();
        const bf16* x_base = x.data();
        const bf16* g_base = gamma.data();
        const bf16* b_base = beta.data();

        // gamma/beta are shared across all L rows, so widen them bf16->fp32 once
        // here instead of re-converting both [D] vectors on every row (was ~L
        // redundant conversions of 2*D elements). pass 2 then loads them as fp32.
        std::vector<float> g_f(D), b_f(D);
        {
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                _mm512_storeu_ps(&g_f[i], bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(g_base + i))));
                _mm512_storeu_ps(&b_f[i], bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(b_base + i))));
            }
            for (; i < D; i++) { g_f[i] = static_cast<float>(g_base[i]); b_f[i] = static_cast<float>(b_base[i]); }
        }
        const float* gf = g_f.data();
        const float* bf = b_f.data();

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* x_row = x_base + (size_t)l * D;
            bf16*       y_row = y_base + (size_t)l * D;

            // pass 1: sum(x) and sum(x^2) in fp32 -> mean, var.
            __m512 sum_x_vec  = _mm512_setzero_ps();
            __m512 sum_xx_vec = _mm512_setzero_ps();
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + i)));
                sum_x_vec  = _mm512_add_ps(sum_x_vec, xv);
                sum_xx_vec = _mm512_fmadd_ps(xv, xv, sum_xx_vec);
            }
            float sum_x  = _mm512_reduce_add_ps(sum_x_vec);
            float sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
            for (; i < D; i++) {
                float v = static_cast<float>(x_row[i]);
                sum_x  += v;
                sum_xx += v * v;
            }
            float mean = sum_x / (float)D;
            float var  = sum_xx / (float)D - mean * mean;
            float inv  = 1.0f / sqrtf(var + eps);

            // pass 2: y = (x - mean) * inv * gamma + beta. fold the centering
            // into a single fma: (x - mean)*inv == fma(x, inv, -mean*inv).
            __m512 inv_vec      = _mm512_set1_ps(inv);
            __m512 neg_mean_inv = _mm512_set1_ps(-mean * inv);
            i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + i)));
                __m512 gv = _mm512_loadu_ps(gf + i);
                __m512 bv = _mm512_loadu_ps(bf + i);
                __m512 norm = _mm512_fmadd_ps(xv, inv_vec, neg_mean_inv);
                __m512 res  = _mm512_fmadd_ps(norm, gv, bv);
                _mm256_storeu_si256((__m256i*)(y_row + i), f32o_bf16_512(res));
            }
            for (; i < D; i++) {
                float v = (static_cast<float>(x_row[i]) - mean) * inv;
                y_row[i] = static_cast<bf16>(v * gf[i] + bf[i]);
            }
        }
    }

    // Non-biased layer norm: y = (x - mean) / sqrt(var + eps) * gamma.
    // Same as _layer_norm_biased but with no beta term.
    inline void _layer_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& gamma,
                           int L, int D, float eps){
        bf16*       y_base = y.data();
        const bf16* x_base = x.data();
        const bf16* g_base = gamma.data();

        // gamma is shared across all L rows, so widen it bf16->fp32 once here
        // instead of re-converting the [D] vector on every row.
        std::vector<float> g_f(D);
        {
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                _mm512_storeu_ps(&g_f[i], bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(g_base + i))));
            }
            for (; i < D; i++) g_f[i] = static_cast<float>(g_base[i]);
        }
        const float* gf = g_f.data();

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* x_row = x_base + (size_t)l * D;
            bf16*       y_row = y_base + (size_t)l * D;

            // pass 1: sum(x) and sum(x^2) in fp32 -> mean, var.
            __m512 sum_x_vec  = _mm512_setzero_ps();
            __m512 sum_xx_vec = _mm512_setzero_ps();
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + i)));
                sum_x_vec  = _mm512_add_ps(sum_x_vec, xv);
                sum_xx_vec = _mm512_fmadd_ps(xv, xv, sum_xx_vec);
            }
            float sum_x  = _mm512_reduce_add_ps(sum_x_vec);
            float sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
            for (; i < D; i++) {
                float v = static_cast<float>(x_row[i]);
                sum_x  += v;
                sum_xx += v * v;
            }
            float mean = sum_x / (float)D;
            float var  = sum_xx / (float)D - mean * mean;
            float inv  = 1.0f / sqrtf(var + eps);

            // pass 2: y = (x - mean) * inv * gamma.
            __m512 inv_vec      = _mm512_set1_ps(inv);
            __m512 neg_mean_inv = _mm512_set1_ps(-mean * inv);
            i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + i)));
                __m512 gv = _mm512_loadu_ps(gf + i);
                __m512 norm = _mm512_fmadd_ps(xv, inv_vec, neg_mean_inv);
                __m512 res  = _mm512_mul_ps(norm, gv);
                _mm256_storeu_si256((__m256i*)(y_row + i), f32o_bf16_512(res));
            }
            for (; i < D; i++) {
                float v = (static_cast<float>(x_row[i]) - mean) * inv;
                y_row[i] = static_cast<bf16>(v * gf[i]);
            }
        }
    }

    // RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma. No mean subtraction (unlike
    // _layer_norm above). x, y, gamma all bf16; in-place safe. Used by the LLaMA-
    // style towers (SmolVLM text model + action expert).
    // y_stride/x_stride are the row strides of y and x (>= D); 0 means D. Only
    // the first D columns of each row are read/written, so a padded destination
    // keeps whatever its trailing columns already held (zeros, if the buffer was
    // cleared once at allocation).
    inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& gamma,
                         int L, int D, float eps, int y_stride = 0, int x_stride = 0){
        bf16*       y_base = y.data();
        const bf16* x_base = x.data();
        const bf16* g_base = gamma.data();
        const int   YS = (y_stride > 0) ? y_stride : D;
        const int   XS = (x_stride > 0) ? x_stride : D;

        std::vector<float> g_f(D);
        {
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                _mm512_storeu_ps(&g_f[i], bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(g_base + i))));
            }
            for (; i < D; i++) g_f[i] = static_cast<float>(g_base[i]);
        }
        const float* gf = g_f.data();

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* x_row = x_base + (size_t)l * XS;
            bf16*       y_row = y_base + (size_t)l * YS;

            __m512 sum_xx_vec = _mm512_setzero_ps();
            int i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + i)));
                sum_xx_vec = _mm512_fmadd_ps(xv, xv, sum_xx_vec);
            }
            float sum_xx = _mm512_reduce_add_ps(sum_xx_vec);
            for (; i < D; i++) {
                float v = static_cast<float>(x_row[i]);
                sum_xx += v * v;
            }
            float inv = 1.0f / sqrtf(sum_xx / (float)D + eps);

            __m512 inv_vec = _mm512_set1_ps(inv);
            i = 0;
            for (; i + simd_width <= D; i += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + i)));
                __m512 gv = _mm512_loadu_ps(gf + i);
                __m512 res = _mm512_mul_ps(_mm512_mul_ps(xv, inv_vec), gv);
                _mm256_storeu_si256((__m256i*)(y_row + i), f32o_bf16_512(res));
            }
            for (; i < D; i++) {
                y_row[i] = static_cast<bf16>(static_cast<float>(x_row[i]) * inv * gf[i]);
            }
        }
    }

    /// @brief single-row biased linear kept in fp32: y[o] = dot(x[0:in], w[o, 0:in])
    ///        (+ bias[o]), for o in [0, out).
    ///
    /// The fp32 output is the point: this exists to fold a broadcast row of a
    /// wider matmul into the fp32 bias of the remaining GEMM, so the contribution
    /// never rounds to bf16 on the way. w is row-major [out, w_stride]; only the
    /// first `in` columns of each row are read, and w_stride defaults to `in`.
    inline void _linear_biased_row_f32(float* y, const bf16* x, const bf16* w,
                                       const bf16* bias, int in, int out,
                                       int w_stride = 0){
        const int WS = (w_stride > 0) ? w_stride : in;
        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int o = 0; o < out; o++) {
            const bf16* w_row = w + (size_t)o * WS;
            __m512 acc = _mm512_setzero_ps();
            int k = 0;
            for (; k + simd_width <= in; k += simd_width) {
                __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x + k)));
                __m512 wv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(w_row + k)));
                acc = _mm512_fmadd_ps(xv, wv, acc);
            }
            float s = _mm512_reduce_add_ps(acc);
            for (; k < in; k++)
                s += static_cast<float>(x[k]) * static_cast<float>(w_row[k]);
            if (bias) s += static_cast<float>(bias[o]);
            y[o] = s;
        }
    }

#ifdef __AVX512BF16__
    /// M rows kept live by one micro-kernel pass. Every weight vector loaded is
    /// reused across all mm_block_m rows, so the weight matrix is streamed once
    /// per block of rows instead of once per row.
    static constexpr int mm_block_m = 4;

    /// bf16 elements consumed by one VDPBF16PS (vs simd_width for an fp32 fma).
    static constexpr int mm_bf16_step = 32;

    /// @brief y[M,N] = x[M,K] * w[N,K]^T for bf16 weights.
    ///
    /// Two things separate this from the generic path below:
    ///  - VDPBF16PS multiplies bf16 pairs directly with fp32 accumulation, so it
    ///    covers mm_bf16_step elements per instruction and needs no bf16->fp32
    ///    widening of either operand.
    ///  - the mm_block_m x mm_unroll_n accumulator tile means each weight load
    ///    feeds mm_block_m fmas. Without it the kernel re-reads the whole weight
    ///    matrix for every output row, which is the actual bottleneck at the
    ///    shapes here (the connector's w is 23.6 MB against a 64 MB L3).
    ///
    /// Accumulation order differs from the generic path, so results can differ
    /// in the last bf16 ulp; both accumulate in fp32 and measure the same error
    /// against an fp64 reference.
    ///
    /// The `#pragma GCC unroll` on every acc[][] loop is load-bearing, not a
    /// hint: left rolled, gcc keeps the tile on the stack and reloads/spills an
    /// accumulator around each dpbf16 (2 vdpbf16ps emitted instead of 16), which
    /// gives back the whole speedup. Check with `objdump -d | grep -c vdpbf16ps`
    /// after touching this loop nest.
    /// @param WS weight row stride (>= K); only the first K columns are read.
    /// @param XS activation row stride (>= K); only the first K columns are read.
    /// @param NS output row stride (>= N); only the first N columns are written.
    /// @param bias optional [N] fp32 bias, folded in before the bf16 store so the
    ///             sum never round-trips through bf16. nullptr to skip.
    inline void _cpu_mm_bf16(bf16* y, const bf16* w_ptr, const bf16* x,
                             int M, int K, int N, int WS, int XS, int NS,
                             const float* bias = nullptr) {
        const int m_blocks = M / mm_block_m;
        const int n_groups = N / mm_unroll_n;
        const int k_main   = K - (K % mm_bf16_step);

        // One (mb, ng) tile. Written once and called from both schedules below
        // so the two only differ in which loop carries the omp pragma.
        auto tile = [&](int mb, int ng) {
            const int m0 = mb * mm_block_m;
            const int n0 = ng * mm_unroll_n;
            const bf16* wp[mm_unroll_n];
            const bf16* xp[mm_block_m];
            #pragma GCC unroll 4
            for (int j = 0; j < mm_unroll_n; j++) wp[j] = w_ptr + (size_t)(n0 + j) * WS;
            #pragma GCC unroll 4
            for (int i = 0; i < mm_block_m; i++) xp[i] = x + (size_t)(m0 + i) * XS;

            __m512 acc[mm_block_m][mm_unroll_n];
            #pragma GCC unroll 4
            for (int i = 0; i < mm_block_m; i++)
                #pragma GCC unroll 4
                for (int j = 0; j < mm_unroll_n; j++) acc[i][j] = _mm512_setzero_ps();

            for (int k = 0; k < k_main; k += mm_bf16_step) {
                __m512bh wv[mm_unroll_n];
                #pragma GCC unroll 4
                for (int j = 0; j < mm_unroll_n; j++)
                    wv[j] = (__m512bh)_mm512_loadu_si512((const void*)(wp[j] + k));
                #pragma GCC unroll 4
                for (int i = 0; i < mm_block_m; i++) {
                    __m512bh xv = (__m512bh)_mm512_loadu_si512((const void*)(xp[i] + k));
                    #pragma GCC unroll 4
                    for (int j = 0; j < mm_unroll_n; j++)
                        acc[i][j] = _mm512_dpbf16_ps(acc[i][j], xv, wv[j]);
                }
            }

            #pragma GCC unroll 4
            for (int i = 0; i < mm_block_m; i++) {
                bf16* y_row = y + (size_t)(m0 + i) * NS;
                #pragma GCC unroll 4
                for (int j = 0; j < mm_unroll_n; j++) {
                    float s = _mm512_reduce_add_ps(acc[i][j]);
                    for (int k = k_main; k < K; k++)
                        s += static_cast<float>(xp[i][k]) * static_cast<float>(wp[j][k]);
                    if (bias) s += bias[n0 + j];
                    y_row[n0 + j] = static_cast<bf16>(s);
                }
            }
        };

        if (m_blocks >= max_prefill_thread) {
            // Enough row blocks to fill the team. Prefer this: each thread walks
            // the full weight matrix once for its own rows, so the n loop stays
            // sequential and w streams in order.
            #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
            for (int mb = 0; mb < m_blocks; mb++)
                for (int ng = 0; ng < n_groups; ng++) tile(mb, ng);
        } else {
            // Too few row blocks (short prompts / single-step expert calls would
            // leave most threads idle), so split the n dimension as well.
            #pragma omp parallel for collapse(2) num_threads(max_prefill_thread) schedule(static)
            for (int mb = 0; mb < m_blocks; mb++)
                for (int ng = 0; ng < n_groups; ng++) tile(mb, ng);
        }

        // rows left over when M is not a multiple of mm_block_m (at most
        // mm_block_m - 1): same inner loop, one row at a time.
        for (int m = m_blocks * mm_block_m; m < M; m++) {
            const bf16* x_row = x + (size_t)m * XS;
            bf16*       y_row = y + (size_t)m * NS;
            #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
            for (int ng = 0; ng < n_groups; ng++) {
                const int n0 = ng * mm_unroll_n;
                const bf16* wp[mm_unroll_n];
                __m512 acc[mm_unroll_n];
                #pragma GCC unroll 4
                for (int j = 0; j < mm_unroll_n; j++) {
                    wp[j]  = w_ptr + (size_t)(n0 + j) * WS;
                    acc[j] = _mm512_setzero_ps();
                }
                for (int k = 0; k < k_main; k += mm_bf16_step) {
                    __m512bh xv = (__m512bh)_mm512_loadu_si512((const void*)(x_row + k));
                    #pragma GCC unroll 4
                    for (int j = 0; j < mm_unroll_n; j++)
                        acc[j] = _mm512_dpbf16_ps(acc[j], xv,
                                     (__m512bh)_mm512_loadu_si512((const void*)(wp[j] + k)));
                }
                #pragma GCC unroll 4
                for (int j = 0; j < mm_unroll_n; j++) {
                    float s = _mm512_reduce_add_ps(acc[j]);
                    for (int k = k_main; k < K; k++)
                        s += static_cast<float>(x_row[k]) * static_cast<float>(wp[j][k]);
                    if (bias) s += bias[n0 + j];
                    y_row[n0 + j] = static_cast<bf16>(s);
                }
            }
        }
    }
#endif // __AVX512BF16__

    template <typename WT>
    inline void _cpu_mm(bf16* y, WT* w_ptr, bf16* x, int M, int K, int N, int w_stride = 0,
                        const float* bias = nullptr, int x_stride = 0, int y_stride = 0){
        // y[m][n] = dot(x[m,:K], weight[n,:K]) (+ bias[n])
        // x      : M x x_row_stride, row-major, bf16; only the first K columns of
        //          each row are read. x_stride lets the activation rows be wider
        //          than K (the expert keeps every activation at its padded dim,
        //          zeros to the right): when > 0 it is the row stride. Defaults
        //          to K.
        // weight : N x w_row_stride, row-major, WT (bf16 or float); only the first
        //          K columns of each weight row are accumulated. w_stride lets the
        //          weight rows be wider than K (padded storage): when > 0 it is the
        //          weight row stride and only K columns are read. Defaults to K.
        // bias   : optional [N] fp32 bias added before the bf16 store, so a biased
        //          matmul keeps the whole accumulation in fp32 instead of rounding
        //          the product to bf16 and adding afterwards. nullptr to skip.
        // y      : M x y_row_stride, row-major, bf16 (dot products accumulate in
        //          fp32); only the first N columns of each row are written, so a
        //          padded destination keeps whatever its trailing columns held.
        //          y_stride > 0 is that row stride; defaults to N.
        static_assert(mm_unroll_n == 4, "the unrolled path below keeps 4 accumulators");
        const int WS = (w_stride > 0) ? w_stride : K;
        const int XS = (x_stride > 0) ? x_stride : K;
        const int NS = (y_stride > 0) ? y_stride : N;

#ifdef __AVX512BF16__
        // bf16 weights get the native-bf16 register-blocked kernel. fp32 weights
        // fall through: VDPBF16PS has no fp32 operand form, and narrowing them
        // would change the result.
        if constexpr (std::is_same_v<WT, bf16>) {
            if (N % mm_unroll_n == 0) {
                _cpu_mm_bf16(y, w_ptr, x, M, K, N, WS, XS, NS, bias);
                return;
            }
        }
#endif

        // Fast path for output widths that divide evenly by mm_unroll_n: emit
        // mm_unroll_n weight rows per pass so each x load feeds that many fmas
        // instead of one.
        if (N % mm_unroll_n == 0) {
            const int n_groups = N / mm_unroll_n;
            #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
            for (int m = 0; m < M; m++) {
                const bf16* x_row = x + (size_t)m * XS;
                bf16*        y_row = y + (size_t)m * NS;

                for (int ng = 0; ng < n_groups; ng++) {
                    const int    n  = ng * mm_unroll_n;
                    const WT* w0 = w_ptr + (size_t)(n + 0) * WS;
                    const WT* w1 = w_ptr + (size_t)(n + 1) * WS;
                    const WT* w2 = w_ptr + (size_t)(n + 2) * WS;
                    const WT* w3 = w_ptr + (size_t)(n + 3) * WS;

                    __m512 acc0 = _mm512_setzero_ps();
                    __m512 acc1 = _mm512_setzero_ps();
                    __m512 acc2 = _mm512_setzero_ps();
                    __m512 acc3 = _mm512_setzero_ps();

                    int k = 0;
                    for (; k <= K - simd_width; k += simd_width) {
                        __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + k));
                        __m512 xv = bf16o_fp32_512(bf16_x);
                        acc0 = _mm512_fmadd_ps(xv, _load_w16(w0 + k), acc0);
                        acc1 = _mm512_fmadd_ps(xv, _load_w16(w1 + k), acc1);
                        acc2 = _mm512_fmadd_ps(xv, _load_w16(w2 + k), acc2);
                        acc3 = _mm512_fmadd_ps(xv, _load_w16(w3 + k), acc3);
                    }
                    float s0 = _mm512_reduce_add_ps(acc0);
                    float s1 = _mm512_reduce_add_ps(acc1);
                    float s2 = _mm512_reduce_add_ps(acc2);
                    float s3 = _mm512_reduce_add_ps(acc3);
                    for (; k < K; k++) {
                        float xk = static_cast<float>(x_row[k]);
                        s0 += xk * static_cast<float>(w0[k]);
                        s1 += xk * static_cast<float>(w1[k]);
                        s2 += xk * static_cast<float>(w2[k]);
                        s3 += xk * static_cast<float>(w3[k]);
                    }
                    if (bias) { s0 += bias[n+0]; s1 += bias[n+1]; s2 += bias[n+2]; s3 += bias[n+3]; }
                    y_row[n + 0] = static_cast<bf16>(s0);
                    y_row[n + 1] = static_cast<bf16>(s1);
                    y_row[n + 2] = static_cast<bf16>(s2);
                    y_row[n + 3] = static_cast<bf16>(s3);
                }
            }
        } else {
            // General fallback: one dot product per (m, n), vectorised over K
            #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
            for (int m = 0; m < M; m++) {
                const bf16* x_row = x + (size_t)m * XS;
                bf16*        y_row = y + (size_t)m * NS;

                for (int n = 0; n < N; n++) {
                    const WT* w_row = w_ptr + (size_t)n * WS;
                    __m512 acc = _mm512_setzero_ps();
                    int k = 0;
                    for (; k <= K - simd_width; k += simd_width) {
                        __m256i bf16_x = _mm256_loadu_si256((const __m256i*)(x_row + k));
                        __m512 xv = bf16o_fp32_512(bf16_x);
                        __m512 wv = _load_w16(w_row + k);
                        acc = _mm512_fmadd_ps(xv, wv, acc);
                    }
                    float s = _mm512_reduce_add_ps(acc);
                    for (; k < K; k++)
                        s += static_cast<float>(x_row[k]) * static_cast<float>(w_row[k]);
                    if (bias) s += bias[n];
                    y_row[n] = static_cast<bf16>(s);
                }
            }
        }
    }

    // Biased linear: y = x @ W^T + bias, with W stored row-major [out, in]
    // (the safetensor layout) and an optional [out] bias. x is [L, in], y is
    // [L, out], all bf16. Accumulates each dot product in fp32 over the in dim.
    //
    // This is just _cpu_mm with the bf16 bias widened to the fp32 vector that
    // kernel wants: same maths, but it picks up the VDPBF16PS register-blocked
    // path whenever `out` divides by mm_unroll_n (every call site here does).
    // Widening rather than adding afterwards also keeps the bias inside the fp32
    // accumulation, so the sum no longer round-trips through bf16.
    inline void _linear_biased(bf16* y, const bf16* x, const bf16* w,
                               const bf16* bias, int L, int in, int out){
        std::vector<float> bias_f;
        if (bias) {
            bias_f.resize(out);
            for (int o = 0; o < out; o++) bias_f[o] = static_cast<float>(bias[o]);
        }
        _cpu_mm<bf16>(y, const_cast<bf16*>(w), const_cast<bf16*>(x), L, in, out,
                      /*w_stride=*/0, bias ? bias_f.data() : nullptr);
    }

    inline void _elementwise_mul(bf16* dst, bf16* src_0, bf16* src_1, int length);

    inline void _elementwise_mul(bf16* dst, bf16* src_0, bf16* src_1, int length){
        const int unroll = simd_width * 4;

        int i = 0;
        for (; i + unroll <= length; i += unroll){
            __m256i a_vals_0 = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i a_vals_1 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width));
            __m256i a_vals_2 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 2));
            __m256i a_vals_3 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 3));

            __m256i b_vals_0 = _mm256_loadu_si256((const __m256i*)(src_1 + i));
            __m256i b_vals_1 = _mm256_loadu_si256((const __m256i*)(src_1 + i + simd_width));
            __m256i b_vals_2 = _mm256_loadu_si256((const __m256i*)(src_1 + i + simd_width * 2));
            __m256i b_vals_3 = _mm256_loadu_si256((const __m256i*)(src_1 + i + simd_width * 3));

            __m512 result_vec0 = _mm512_mul_ps(bf16o_fp32_512(a_vals_0), bf16o_fp32_512(b_vals_0));
            __m512 result_vec1 = _mm512_mul_ps(bf16o_fp32_512(a_vals_1), bf16o_fp32_512(b_vals_1));
            __m512 result_vec2 = _mm512_mul_ps(bf16o_fp32_512(a_vals_2), bf16o_fp32_512(b_vals_2));
            __m512 result_vec3 = _mm512_mul_ps(bf16o_fp32_512(a_vals_3), bf16o_fp32_512(b_vals_3));

            _mm256_storeu_si256((__m256i*)(dst + i), f32o_bf16_512(result_vec0));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width), f32o_bf16_512(result_vec1));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width * 2), f32o_bf16_512(result_vec2));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width * 3), f32o_bf16_512(result_vec3));
        }

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
    
    inline void _elementwise_mul(float* dst, bf16* src_0, float src_1, int length){
        const int unroll = simd_width * 4;
        const __m512 b_vals = _mm512_set1_ps(src_1);

        int i = 0;
        for (; i + unroll <= length; i += unroll){
            __m256i a_vals_0 = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i a_vals_1 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width));
            __m256i a_vals_2 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 2));
            __m256i a_vals_3 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 3));

            __m512 result_vec0 = _mm512_mul_ps(bf16o_fp32_512(a_vals_0), b_vals);
            __m512 result_vec1 = _mm512_mul_ps(bf16o_fp32_512(a_vals_1), b_vals);
            __m512 result_vec2 = _mm512_mul_ps(bf16o_fp32_512(a_vals_2), b_vals);
            __m512 result_vec3 = _mm512_mul_ps(bf16o_fp32_512(a_vals_3), b_vals);

            _mm512_storeu_ps(dst + i, result_vec0);
            _mm512_storeu_ps(dst + i + simd_width, result_vec1);
            _mm512_storeu_ps(dst + i + simd_width * 2, result_vec2);
            _mm512_storeu_ps(dst + i + simd_width * 3, result_vec3);
        }

        for (; i + simd_width <= length; i += simd_width){
            __m256i a_vals = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m512 result_vec = _mm512_mul_ps(bf16o_fp32_512(a_vals), b_vals);
            _mm512_storeu_ps(dst + i, result_vec);
        }

        for (; i < length; i++){
            dst[i] = static_cast<float>(src_0[i]) * src_1;
        }
    }

    // Scale a bf16 buffer by a scalar, in place / bf16 output: dst = src_0 * src_1.
    inline void _elementwise_mul(bf16* dst, bf16* src_0, float src_1, int length){
        const int unroll = simd_width * 4;
        const __m512 b_vals = _mm512_set1_ps(src_1);

        int i = 0;
        for (; i + unroll <= length; i += unroll){
            __m256i a_vals_0 = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            __m256i a_vals_1 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width));
            __m256i a_vals_2 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 2));
            __m256i a_vals_3 = _mm256_loadu_si256((const __m256i*)(src_0 + i + simd_width * 3));

            _mm256_storeu_si256((__m256i*)(dst + i),                  f32o_bf16_512(_mm512_mul_ps(bf16o_fp32_512(a_vals_0), b_vals)));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width),     f32o_bf16_512(_mm512_mul_ps(bf16o_fp32_512(a_vals_1), b_vals)));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width * 2), f32o_bf16_512(_mm512_mul_ps(bf16o_fp32_512(a_vals_2), b_vals)));
            _mm256_storeu_si256((__m256i*)(dst + i + simd_width * 3), f32o_bf16_512(_mm512_mul_ps(bf16o_fp32_512(a_vals_3), b_vals)));
        }

        for (; i + simd_width <= length; i += simd_width){
            __m256i a_vals = _mm256_loadu_si256((const __m256i*)(src_0 + i));
            _mm256_storeu_si256((__m256i*)(dst + i), f32o_bf16_512(_mm512_mul_ps(bf16o_fp32_512(a_vals), b_vals)));
        }

        for (; i < length; i++){
            dst[i] = static_cast<bf16>(static_cast<float>(src_0[i]) * src_1);
        }
    }

    inline void _sigmoid(bf16* y, const bf16* x, int length) {
        const __m512 v_one     = _mm512_set1_ps(1.0f);
        const __m512 v_neg_one = _mm512_set1_ps(-1.0f);

        int i = 0;
        for (; i + simd_width <= length; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((const __m256i*)(x + i));
            __m512  fp32_x    = bf16o_fp32_512(bf16_vals);

            __m512 neg_x      = _mm512_mul_ps(fp32_x, v_neg_one);
            __m512 exp_neg_x  = _mm512_exp_ps_corrected(neg_x);
            __m512 denom      = _mm512_add_ps(v_one, exp_neg_x);
            __m512 result     = _mm512_div_ps(v_one, denom);

            _mm256_storeu_si256((__m256i*)(y + i), f32o_bf16_512(result));
        }
        for (; i < length; i++) {
            float val = static_cast<float>(x[i]);
            y[i]  = static_cast<bf16>(1.0f / (1.0f + expf(-val)));
        }
    }

    // tanh-approximate GELU (gelu_pytorch_tanh), in place or out of place:
    //   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Since 0.5 * (1 + tanh(z)) == sigmoid(2z), this collapses to
    //   y = x * sigmoid(2*sqrt(2/pi) * (x + 0.044715 * x^3))
    // which needs a single exp instead of the two a tanh would cost.
    /// exp(x) for the GELU: range reduction with roundscale + scalef (no int round trip), degree-5 polynomial.
    /// Relative error ~2e-7 over [-87, 87], far below the bf16 store.
    static inline __m512 _exp_scalef(__m512 x) {
        x = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(87.0f)), _mm512_set1_ps(-87.0f));
        const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.6931471824645996f), x);
        r = _mm512_fnmadd_ps(n, _mm512_set1_ps(1.9082149292705877e-10f), r);
        __m512 p = _mm512_fmadd_ps(_mm512_set1_ps(0.008333333333333333f), r, _mm512_set1_ps(0.041666666666666664f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.16666666666666666f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
        return _mm512_scalef_ps(p, n);
    }

    /// Faster _gelu_tanh: scalef exp and rcp14 + one Newton step instead of the divide (SMOLVLA_GELU_FAST=0 selects
    /// the original path below).
    inline void _gelu_tanh_fast(bf16* y, const bf16* x, int length) {
        const __m512 v_one   = _mm512_set1_ps(1.0f);
        const __m512 v_two   = _mm512_set1_ps(2.0f);
        const __m512 v_scale = _mm512_set1_ps(1.5957691216057308f);
        const __m512 v_coeff = _mm512_set1_ps(0.044715f);
        const int simd_end = length - (length % simd_width);

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int i = 0; i < simd_end; i += simd_width) {
            __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x + i)));
            __m512 x3    = _mm512_mul_ps(_mm512_mul_ps(xv, xv), xv);
            __m512 inner = _mm512_fmadd_ps(v_coeff, x3, xv);
            __m512 u     = _mm512_mul_ps(v_scale, inner);
            __m512 denom = _mm512_add_ps(v_one, _exp_scalef(_mm512_sub_ps(_mm512_setzero_ps(), u)));
            __m512 rcp   = _mm512_rcp14_ps(denom);
            rcp = _mm512_mul_ps(rcp, _mm512_fnmadd_ps(denom, rcp, v_two));   // Newton: r * (2 - d*r)
            _mm256_storeu_si256((__m256i*)(y + i), f32o_bf16_512(_mm512_mul_ps(xv, rcp)));
        }
        for (int i = simd_end; i < length; i++) {
            float v = static_cast<float>(x[i]);
            float u = 1.5957691216057308f * (v + 0.044715f * v * v * v);
            y[i] = static_cast<bf16>(v / (1.0f + expf(-u)));
        }
    }

    inline void _gelu_tanh(bf16* y, const bf16* x, int length) {
        static const bool fast = std::getenv("SMOLVLA_GELU_FAST") == nullptr || std::getenv("SMOLVLA_GELU_FAST")[0] != '0';
        if (fast) { _gelu_tanh_fast(y, x, length); return; }
        const __m512 v_one   = _mm512_set1_ps(1.0f);
        const __m512 v_scale = _mm512_set1_ps(1.5957691216057308f);  // 2*sqrt(2/pi)
        const __m512 v_coeff = _mm512_set1_ps(0.044715f);
        const int simd_end = length - (length % simd_width);

        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int i = 0; i < simd_end; i += simd_width) {
            __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x + i)));

            // u = 2*sqrt(2/pi) * (x + 0.044715 * x^3)
            __m512 x3    = _mm512_mul_ps(_mm512_mul_ps(xv, xv), xv);
            __m512 inner = _mm512_fmadd_ps(v_coeff, x3, xv);
            __m512 u     = _mm512_mul_ps(v_scale, inner);

            // y = x * sigmoid(u) = x / (1 + exp(-u))
            __m512 denom  = _mm512_add_ps(v_one, _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), u)));
            __m512 result = _mm512_div_ps(xv, denom);

            _mm256_storeu_si256((__m256i*)(y + i), f32o_bf16_512(result));
        }
        for (int i = simd_end; i < length; i++) {
            float v = static_cast<float>(x[i]);
            float u = 1.5957691216057308f * (v + 0.044715f * v * v * v);
            y[i] = static_cast<bf16>(v / (1.0f + expf(-u)));
        }
    }

    // LLaMA-style RoPE inverse frequencies: inv_freq[j] = theta^(-2j/head_dim),
    // head_dim/2 entries. Both towers build their table through this instead of
    // carrying a hand-transcribed constant array.
    inline std::vector<float> _rope_inv_freq(int head_dim, float theta){
        std::vector<float> inv_freq(head_dim / 2);
        for (int j = 0; j < head_dim / 2; j++) {
            inv_freq[j] = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
        }

        return inv_freq;
    }

    // Plain rotary position embedding (rotate_half / LLaMA convention), no
    // rms-norm. x is [L, num_heads*head_dim] bf16, updated in place. Positions
    // are 0..L-1; inv_freq has head_dim/2 entries.
    // pos_offset shifts the rotary position of row 0 to pos_offset (rows use
    // pos_offset..pos_offset+L-1). Used by the action expert, whose action-token
    // chunk sits after the VLM prefix (positions start at the prefix length).
    // row_stride is x's row stride (>= num_heads*head_dim); 0 means exactly that.
    // Heads sit at h*head_dim from the row start either way, so a wider stride
    // only moves the rows apart and leaves the trailing padding untouched.
    inline void _rope(bf16* x, int L, int num_heads, int head_dim, const float* inv_freq,
                      int pos_offset = 0, int row_stride = 0){
        const int D = (row_stride > 0) ? row_stride : num_heads * head_dim;
        const int half = head_dim / 2;
        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
        for (int pos = 0; pos < L; pos++){
            std::vector<float> cs(half), sn(half);
            for (int j = 0; j < half; j++){
                float angle = inv_freq[j] * (float)(pos + pos_offset);
                cs[j] = cosf(angle);
                sn[j] = sinf(angle);
            }
            bf16* x_row = x + (size_t)pos * D;
            for (int h = 0; h < num_heads; h++){
                bf16* xh = x_row + h * head_dim;
                for (int j = 0; j < half; j++){
                    float l = (float)xh[j];
                    float r = (float)xh[j + half];
                    xh[j]        = (bf16)(l * cs[j] - r * sn[j]);
                    xh[j + half] = (bf16)(l * sn[j] + r * cs[j]);
                }
            }
        }
    }

    // Grouped-query multi-head self-attention. q is [L_q, num_q_heads*head_dim],
    // k/v are [L_kv, num_kv_heads*head_dim]; each group of num_q_heads/num_kv_heads
    // query heads shares one kv head. out is [L_q, num_q_heads*head_dim] bf16.
    // Softmax over keys is computed per (head, query) in fp32. L_q may be smaller
    // than L_kv (e.g. attend just the last query rows over the full key prefix);
    // q_offset is the absolute position of query row 0 within the key sequence,
    // used only for causal masking (query i sees keys [0, q_offset + i]).
    // q_stride/kv_stride/out_stride are the row strides of q, k+v, and out
    // (>= num_q_heads*head_dim / num_kv_heads*head_dim respectively); 0 means the
    // packed width. Heads always sit at h*head_dim from the row start, so a wider
    // stride only spaces the rows out and the trailing padding is never touched.
    inline void _gqa_mha(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                         int L_q, int L_kv, int num_q_heads, int num_kv_heads,
                         int head_dim, bool causal, int q_offset = 0,
                         int q_stride = 0, int kv_stride = 0, int out_stride = 0){
        const int QS  = (q_stride   > 0) ? q_stride   : num_q_heads  * head_dim;
        const int DKV = (kv_stride  > 0) ? kv_stride  : num_kv_heads * head_dim;
        const int OS  = (out_stride > 0) ? out_stride : num_q_heads  * head_dim;
        const int group = num_q_heads / num_kv_heads;
        const float scale = 1.0f / sqrtf((float)head_dim);
        #pragma omp parallel for num_threads(max_prefill_thread) schedule(static) collapse(2)
        for (int h = 0; h < num_q_heads; h++) {
            for (int i = 0; i < L_q; i++) {
                const int kvh = h / group;
                const bf16* qi = q + (size_t)i * QS + h * head_dim;
                const int jmax = causal ? (q_offset + i + 1) : L_kv;
                std::vector<float> scores(jmax);
                float mx = -1e30f;
                for (int j = 0; j < jmax; j++) {
                    const bf16* kj = k + (size_t)j * DKV + kvh * head_dim;
                    float s = 0.0f;
                    for (int e = 0; e < head_dim; e++)
                        s += (float)qi[e] * (float)kj[e];
                    s *= scale;
                    scores[j] = s;
                    if (s > mx) mx = s;
                }
                float denom = 0.0f;
                for (int j = 0; j < jmax; j++) {
                    scores[j] = expf(scores[j] - mx);
                    denom += scores[j];
                }
                float inv = 1.0f / denom;
                std::vector<float> oi(head_dim, 0.0f);
                for (int j = 0; j < jmax; j++) {
                    float wgt = scores[j] * inv;
                    const bf16* vj = v + (size_t)j * DKV + kvh * head_dim;
                    for (int e = 0; e < head_dim; e++)
                        oi[e] += wgt * (float)vj[e];
                }
                bf16* oiptr = out + (size_t)i * OS + h * head_dim;
                for (int e = 0; e < head_dim; e++) oiptr[e] = (bf16)oi[e];
            }
        }
    }
}

#endif // __CPU__FUNCTIONS__HPP__