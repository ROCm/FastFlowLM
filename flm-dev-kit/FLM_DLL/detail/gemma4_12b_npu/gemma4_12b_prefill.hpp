// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __GEMMA4_12B_PREFILL_HPP__
#define __GEMMA4_12B_PREFILL_HPP__

#include "npu_utils/npu_utils.hpp"
#include "gemma4_12b_npu_sequence.hpp"
#include "gemma4_12b_npu_def.hpp"
#include "gemma4_12b_prefill_xref.hpp"
#include "avx512_util.hpp"
#include "cpu_functions.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GEMMA4_12B_PREFILL_PROFILE
#define PREFILL_PROF_INIT() time_utils::time_point __prof_t = time_utils::now()
#define PREFILL_PROF_TICK(name)                                                \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::duration_us(__prof_t, __prof_now));\
        header_print("prefill-prof", name << ": " << __prof_d.first << " "     \
                                          << __prof_d.second);                 \
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PREFILL_PROF_LAP() (__prof_t = time_utils::now())
#define PREFILL_PROF_ACC_DECL(var) float __prof_acc_##var = 0.0f
#define PREFILL_PROF_ACC(var)                                                  \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        __prof_acc_##var += time_utils::duration_us(__prof_t, __prof_now).first;\
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PREFILL_PROF_ACC_PRINT(name, var)                                      \
    do {                                                                       \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::time_with_unit(__prof_acc_##var, "us"));\
        header_print("prefill-prof", name << " (sum): " << __prof_d.first      \
                                          << " " << __prof_d.second);          \
    } while (0)
#else
#define PREFILL_PROF_INIT() do {} while (0)
#define PREFILL_PROF_TICK(name) do {} while (0)
#define PREFILL_PROF_LAP() do {} while (0)
#define PREFILL_PROF_ACC_DECL(var)
#define PREFILL_PROF_ACC(var) do {} while (0)
#define PREFILL_PROF_ACC_PRINT(name, var) do {} while (0)
#endif


// ===========================================================================
//  Backend selection
//
//  Two stages of the batched prefill can run either on the host or on the NPU:
//
//    the dequantizing matmuls  -- q/k/v/o and up/gate/down, dequant_mm.xclbin
//    the attention             -- the causal / sliding window softmax
//
//  Pick them independently at compile time. From the Makefile:
//
//    make PREFILL_MM=cpu      # -DFLM_PREFILL_MM_ON_CPU
//    make PREFILL_ATTN=cpu    # -DFLM_PREFILL_ATTN_ON_CPU
//
//  Both default to the NPU. With FLM_PREFILL_MM_ON_CPU *and* FLM_PREFILL_ATTN_ON_CPU
//  the prefill touches no xclbin at all -- nothing is registered and the activation
//  scratch is plain host memory -- which is what makes the CPU build runnable without
//  a device.
//
//  The attention selection is per layer kind, because the two kinds are two separate
//  overlays with their own bitstream, RTP map and floorplan:
//
//    global  (full attention, DH 512)  -> attention_DH_512_prefill, attn_global.xclbin
//    sliding (windowed,       DH 256)  -> attention_DH_256_prefill, attn_sliding.xclbin
//
//  Neither of those two knows about vision blocks -- the mask they can express is
//  "causal, optionally windowed" -- so the bidirectional rows of a §4.4 image block have
//  to be recomputed and overwritten afterwards, see _stage_vision_blocks() /
//  _patch_vision_blocks(). That recompute has a third overlay of its own:
//
//    global image (block mask, DH 512) -> attention_global_image, attn_global_image.xclbin
//
//  which is the DH_512 array with the causal term deleted: every query row it is handed
//  attends kv rows [0, L). For a *global* layer that is exactly the block rule, because
//  the block's own kv range is the same [0, block_end) for all of its query rows. The
//  sliding layers keep the host patch -- their per row window makes the range row
//  dependent, which this overlay cannot express.
//
//    make PREFILL_IMAGE_ATTN=cpu   # -DFLM_PREFILL_IMAGE_ATTN_ON_CPU
//
//  keeps the global block patch on the host. It is also the automatic fallback whenever
//  attn_global_image.xclbin is missing or a block does not fit the overlay, so the host
//  kernel stays live and reachable in every build.
// ===========================================================================
#ifdef FLM_PREFILL_MM_ON_CPU
#define GEMMA4_12B_MM_ON_NPU 0
#else
#define GEMMA4_12B_MM_ON_NPU 1
#endif

#ifdef FLM_PREFILL_ATTN_ON_CPU
#define GEMMA4_12B_ATTN_GLOBAL_ON_NPU 0
#else
#define GEMMA4_12B_ATTN_GLOBAL_ON_NPU 1
#endif

#ifdef FLM_PREFILL_ATTN_ON_CPU
#define GEMMA4_12B_ATTN_SWA_ON_NPU 0
#else
#define GEMMA4_12B_ATTN_SWA_ON_NPU 1
#endif

//!< The image block patch of the global layers. It rides on the global overlay's own
//!< switch -- the block rows are a correction *to* the array's output, so running them on
//!< the device while the pass they correct is on the host would be a pointless round trip.
#if GEMMA4_12B_ATTN_GLOBAL_ON_NPU && !defined(FLM_PREFILL_IMAGE_ATTN_ON_CPU)
#define GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU 1
#else
#define GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU 0
#endif

#define GEMMA4_12B_ATTN_ON_NPU (GEMMA4_12B_ATTN_GLOBAL_ON_NPU || GEMMA4_12B_ATTN_SWA_ON_NPU)

//!< The activation scratch has to be a device BO as soon as *either* stage runs on the
//!< array; when both run on the host it is plain memory and must never be synced.
#define GEMMA4_12B_SCRATCH_ON_DEVICE (GEMMA4_12B_MM_ON_NPU || GEMMA4_12B_ATTN_ON_NPU)

/// \brief Host kernels of the batched (matmul) prefill path: the q4_0 dequantizing
///        matmul and the causal / sliding window attention. The signatures mirror
///        the NPU graph below (same buffers, same shapes, same call order), which is
///        what lets FLM_PREFILL_MM_ON_CPU swap one for the other.
///
/// The q4_0 block layout is the one the decoding kernel reads, see
/// FLM_Xclbin/Gemma4_12B_QAT/decoding/kernels/q4_core.h::_Q40_x_bf16. A block covers
/// 16 rows x 256 columns in 2304 bytes: 128 bf16 scales indexed `g * 16 + r`
/// (g = 32-column group, r = row in block), then 2048 bytes of packed signed 4 bit
/// quants where nibble `n` is (g, c, r) = (n / 512, (n / 16) % 32, n % 16), low
/// nibble first, giving `w[row][col] = v(g, c, r) * scale[g * 16 + r]`.
/// gemma4_12b_desc::reorder_cpy then interleaves pairs of 16-row bands per column
/// block, which is the block_index() below.

namespace gemma4_12b_cpu {

    // ---- q4_0 block geometry, mirrors gemma4_12b_desc ----
    static constexpr int Q4_M           = 16;    //!< rows per block
    static constexpr int Q4_K           = 256;   //!< columns per block
    static constexpr int Q4_GROUP       = 32;    //!< columns sharing one scale
    static constexpr int Q4_GROUPS      = Q4_K / Q4_GROUP;
    static constexpr int Q4_SCALE_BYTES = Q4_M * Q4_GROUPS * (int)sizeof(bf16); // 256
    static constexpr int Q4_BLOCK_BYTES = Q4_SCALE_BYTES + Q4_M * Q4_K / 2;     // 2304
    static constexpr int Q4_VBLOCKS     = 2;     //!< 16-row bands interleaved by reorder_cpy

    /// \brief Thread count for the host prefill kernels.
    /// \note Deliberately *not* Impl::max_prefill_thread. These two kernels are
    ///       the whole cost of a CPU prefill (a 1k token prompt is ~2e13 flops),
    ///       so they get the whole machine. Override with FLM_PREFILL_THREADS.
    inline int prefill_threads() {
        static const int nt = [] {
#ifdef _OPENMP
            int n = omp_get_max_threads();
#else
            int n = 1;
#endif
            if (const char* env = std::getenv("FLM_PREFILL_THREADS")) {
                int v = std::atoi(env);
                if (v > 0) n = v;
            }
            return n < 1 ? 1 : n;
        }();
        return nt;
    }

    /// \brief Byte offset of the block covering rows [rb*16, rb*16+16) of column block cb.
    /// \param rb 16-row band index
    /// \param cb 256-column block index
    /// \param blocks_per_row number of column blocks, i.e. K / 256
    inline size_t block_offset(int rb, int cb, int blocks_per_row) {
        const size_t pair = (size_t)(rb / Q4_VBLOCKS);
        const size_t half = (size_t)(rb % Q4_VBLOCKS);
        return (((pair * (size_t)blocks_per_row + (size_t)cb) * Q4_VBLOCKS) + half) * Q4_BLOCK_BYTES;
    }

    /// \brief Dequantize one 16-row band of a q4_0 weight into a dense fp32 tile.
    /// \param dst destination, 16 * K floats, row major with leading dimension K
    /// \param w   base of the reordered q4_0 region
    /// \param rb  16-row band index
    /// \param K   input dimension of the weight, a multiple of 256
    inline void dequant_row_block(float* dst, const u8* w, int rb, int K) {
        const int blocks_per_row = K / Q4_K;
        for (int cb = 0; cb < blocks_per_row; cb++) {
            const u8*   blk = w + block_offset(rb, cb, blocks_per_row);
            const bf16* sc  = reinterpret_cast<const bf16*>(blk);
            const u8*   qs  = blk + Q4_SCALE_BYTES;
            const int   col0 = cb * Q4_K;

            for (int g = 0; g < Q4_GROUPS; g++) {
                float s[Q4_M];
                for (int r = 0; r < Q4_M; r++) s[r] = static_cast<float>(sc[g * Q4_M + r]);

                const u8* gp = qs + (size_t)g * (Q4_GROUP * Q4_M / 2);
                for (int c = 0; c < Q4_GROUP; c++) {
                    const u8* p   = gp + (size_t)c * (Q4_M / 2);
                    const int col = col0 + g * Q4_GROUP + c;
                    for (int rr = 0; rr < Q4_M / 2; rr++) {
                        const int b  = (int)p[rr];
                        // low nibble first, both nibbles are signed 4 bit
                        const int lo = (b & 0x0F) - ((b & 0x08) ? 16 : 0);
                        const int hi = (b >> 4)   - ((b & 0x80) ? 16 : 0);
                        dst[(size_t)(2 * rr)     * K + col] = (float)lo * s[2 * rr];
                        dst[(size_t)(2 * rr + 1) * K + col] = (float)hi * s[2 * rr + 1];
                    }
                }
            }
        }
    }

    /// \brief y[l][0..15] = x[l] . w[0..15], for a dense 16 x K fp32 tile. Serial.
    inline void mm_row_block(bf16* y, int ldy, const float* w, const bf16* x, int L, int K) {
        for (int l = 0; l < L; l++) {
            const bf16* x_row = x + (size_t)l * K;
            bf16*       y_row = y + (size_t)l * ldy;
            for (int r0 = 0; r0 < Q4_M; r0 += 4) {
                const float* w0 = w + (size_t)(r0 + 0) * K;
                const float* w1 = w + (size_t)(r0 + 1) * K;
                const float* w2 = w + (size_t)(r0 + 2) * K;
                const float* w3 = w + (size_t)(r0 + 3) * K;

                __m512 a0 = _mm512_setzero_ps();
                __m512 a1 = _mm512_setzero_ps();
                __m512 a2 = _mm512_setzero_ps();
                __m512 a3 = _mm512_setzero_ps();

                int k = 0;
                for (; k + 16 <= K; k += 16) {
                    __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x_row + k)));
                    a0 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w0 + k), a0);
                    a1 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w1 + k), a1);
                    a2 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w2 + k), a2);
                    a3 = _mm512_fmadd_ps(xv, _mm512_loadu_ps(w3 + k), a3);
                }
                float t0 = _mm512_reduce_add_ps(a0);
                float t1 = _mm512_reduce_add_ps(a1);
                float t2 = _mm512_reduce_add_ps(a2);
                float t3 = _mm512_reduce_add_ps(a3);
                for (; k < K; k++) {   // K is always a multiple of 256, kept for safety
                    const float xv = static_cast<float>(x_row[k]);
                    t0 += xv * w0[k]; t1 += xv * w1[k]; t2 += xv * w2[k]; t3 += xv * w3[k];
                }
                y_row[r0 + 0] = static_cast<bf16>(t0);
                y_row[r0 + 1] = static_cast<bf16>(t1);
                y_row[r0 + 2] = static_cast<bf16>(t2);
                y_row[r0 + 3] = static_cast<bf16>(t3);
            }
        }
    }

    /// \brief y[L][N] = x[L][K] * W^T, with W an N x K weight in reordered q4_0.
    /// \param y destination, L * N bf16, row major
    /// \param w base of the reordered q4_0 region (proj_weights.data() + desc.offset)
    /// \param x input activations, L * K bf16, row major
    /// \note The equivalent of one gen_dequant_mm_512() run. Weights are walked once
    ///       per 16-row band and reused across all L rows, so the memory traffic is
    ///       the weight size, not L times it.
    inline void dequant_mm(bf16* y, const u8* w, const bf16* x, int L, int K, int N) {
        assert(K % Q4_K == 0 && "q4_0 needs a 256 element input dimension");
        assert(N % (Q4_M * Q4_VBLOCKS) == 0 && "q4_0 needs a 32 element output dimension");
        if (L <= 0) return;

        const int row_blocks = N / Q4_M;

        #pragma omp parallel num_threads(prefill_threads())
        {
            std::vector<float> tile((size_t)Q4_M * K);
            #pragma omp for schedule(static)
            for (int rb = 0; rb < row_blocks; rb++) {
                dequant_row_block(tile.data(), w, rb, K);
                mm_row_block(y + (size_t)rb * Q4_M, N, tile.data(), x, L, K);
            }
        }
    }

    /// \brief fp32 dot product of two bf16 vectors.
    inline float dot_bf16(const bf16* a, const bf16* b, int n) {
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            acc = _mm512_fmadd_ps(bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(a + i))),
                                  bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(b + i))), acc);
        }
        float s = _mm512_reduce_add_ps(acc);
        for (; i < n; i++) s += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        return s;
    }

    /// \brief acc[0..n) += weight * v[0..n), bf16 source into an fp32 accumulator.
    inline void axpy_bf16(float* acc, const bf16* v, float weight, int n) {
        const __m512 wv = _mm512_set1_ps(weight);
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 a = _mm512_loadu_ps(acc + i);
            a = _mm512_fmadd_ps(wv, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(v + i))), a);
            _mm512_storeu_ps(acc + i, a);
        }
        for (; i < n; i++) acc[i] += weight * static_cast<float>(v[i]);
    }

    /// \brief Grouped query attention over a contiguous key/value range.
    ///
    /// \param out         [L_q, num_heads * DH] bf16, head `h` at column h * DH.
    /// \param q           [L_q, num_heads * DH] bf16, already q-normed and roped.
    /// \param k           [L_kv, k_stride] bf16, kv head `hk` at column hk * DH.
    /// \param v           [L_kv, v_stride] bf16, same head layout as k.
    /// \param q_pos_begin absolute position of q row 0.
    /// \param kv_pos_begin absolute position of k/v row 0.
    /// \param window      sliding window size, or 0 for plain causal attention.
    ///
    /// \param blk_begin   absolute start of the vision block the queries sit in
    /// \param blk_end     absolute end (exclusive) of it; blk_end <= blk_begin disables it
    ///
    /// \note There is no 1/sqrt(head_dim) factor: gemma4-12b folds it into
    ///       q_norm.weight, exactly like SimpleGemma4Attention.
    /// \note The output column order is the q_proj head order. The reference
    ///       rearranges "B (h R) L d -> B h L R d" before the einsum and back with
    ///       "B h R L d -> B L (h R d)" afterwards, and those two cancel.
    inline void attention(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                          int L_q, int L_kv, int num_heads, int num_kv_heads, int DH,
                          int q_pos_begin, int kv_pos_begin, int window,
                          int k_stride, int v_stride,
                          int blk_begin = 0, int blk_end = 0) {
        assert(num_kv_heads > 0 && num_heads % num_kv_heads == 0);
        const int groups   = num_heads / num_kv_heads;
        const int q_stride = num_heads * DH;

        #pragma omp parallel num_threads(prefill_threads())
        {
            std::vector<float> scores((size_t)(L_kv > 0 ? L_kv : 1));
            std::vector<float> acc((size_t)DH);

            #pragma omp for collapse(2) schedule(static)
            for (int l = 0; l < L_q; l++) {
                for (int h = 0; h < num_heads; h++) {
                    bf16* o = out + (size_t)l * q_stride + (size_t)h * DH;

                    const int p  = q_pos_begin + l;
                    int       j0 = 0;
                    int       j1 = p - kv_pos_begin;
                    if (window > 0) {
                        const int lo = p - window + 1 - kv_pos_begin;
                        if (lo > j0) j0 = lo;
                    }
                    if (p >= blk_begin && p < blk_end) {
                        // bidirectional inside the block: open the upper bound to its end.
                        // The lower bound only ever moves for a block wider than the
                        // window, which cannot happen (280 < 1024), but costs nothing.
                        const int be = blk_end - 1 - kv_pos_begin;
                        if (be > j1) j1 = be;
                        const int bs = blk_begin - kv_pos_begin;
                        if (bs < j0) j0 = bs;
                    }
                    if (j0 < 0) j0 = 0;
                    if (j1 > L_kv - 1) j1 = L_kv - 1;
                    if (j1 < j0) {   // fully masked, cannot happen for a causal prompt
                        memset(o, 0, (size_t)DH * sizeof(bf16));
                        continue;
                    }

                    const int   hk = h / groups;
                    const bf16* qh = q + (size_t)l * q_stride + (size_t)h * DH;

                    float m = -INFINITY;
                    for (int j = j0; j <= j1; j++) {
                        const float s = dot_bf16(qh, k + (size_t)j * k_stride + (size_t)hk * DH, DH);
                        scores[j] = s;
                        if (s > m) m = s;
                    }
                    float sum = 0.0f;
                    for (int j = j0; j <= j1; j++) {
                        const float e = expf(scores[j] - m);
                        scores[j] = e;
                        sum += e;
                    }
                    const float inv = 1.0f / sum;

                    std::fill(acc.begin(), acc.end(), 0.0f);
                    for (int j = j0; j <= j1; j++) {
                        axpy_bf16(acc.data(), v + (size_t)j * v_stride + (size_t)hk * DH,
                                  scores[j] * inv, DH);
                    }
                    for (int d = 0; d < DH; d++) o[d] = static_cast<bf16>(acc[d]);
                }
            }
        }
    }

} // namespace gemma4_12b_cpu

namespace gemma4_12b_cpu {

    /// \brief The host equivalent of the fused up/gate projection pair.
    ///
    /// The NPU runs that region twice, once per slab parity, and the two runs scatter
    /// into a de-interleaved [ up | gate ] row with the GeLU already fused into the gate
    /// half. One host matmul produces the *interleaved* (L, IS / slab, 2, slab) layout
    /// the weights are stored in, so undo the interleave and apply the GeLU here: what
    /// this leaves in `y` is what the two runs leave behind, which is what lets
    /// everything downstream -- cpu_func::_glu_mul and the reference checks alike -- be
    /// shared between the two backends.
    ///
    /// \param y    [L, 2 * IS] bf16, [ up | gate ] per row, gate already GeLU'd
    /// \param w    base of the reordered q4_0 up/gate region
    /// \param x    [L, K] bf16 activations
    /// \param slab gemma4_12b_desc::UP_GATE_OUTDIM_REORDER
    inline void dequant_mm_upgate(bf16* y, const u8* w, const bf16* x,
                                  int L, int K, int IS, int slab) {
        if (L <= 0) return;
        std::vector<bf16> fused((size_t)L * IS * 2);
        dequant_mm(fused.data(), w, x, L, K, IS * 2);

        const int slices = IS / slab;
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* src   = fused.data() + (size_t)l * IS * 2;
            bf16*       up    = y + (size_t)l * IS * 2;
            bf16*       gate  = up + IS;
            for (int s = 0; s < slices; s++) {
                const bf16* p = src + (size_t)s * 2 * slab;
                memcpy(up   + (size_t)s * slab, p,        (size_t)slab * sizeof(bf16));
                // gelu_tanh, the same approximation DEQUANT_MM_ACT_GELU fuses on the NPU
                // and cpu_func::_gelu_glu uses on the host
                for (int i = 0; i < slab; i++) {
                    const float g     = static_cast<float>(p[slab + i]);
                    const float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
                    gate[(size_t)s * slab + i] = static_cast<bf16>(0.5f * g * (1.0f + tanhf(inner)));
                }
            }
        }
    }

} // namespace gemma4_12b_cpu

/// \brief One projection that has been issued and not yet collected.
///
/// On the NPU backend it holds the xrt::run, so the caller can do host work while the
/// array is busy and pull the result back in wait(). On the CPU backend the matmul has
/// already run by the time the handle exists and wait() has nothing to do, so the same
/// call sites work on both, just without the overlap.
struct gemma4_12b_mm_run {
#if GEMMA4_12B_MM_ON_NPU
    flm_rt::run      run;
    buffer<bf16>* dst = nullptr;  //!< null once collected, or when nothing was issued

#ifdef ENABLE_DEBUG
    //!< everything gemma4_12b_cpu::dequant_mm() needs to recompute this projection on the
    //!< host, captured at issue time and replayed in wait(). See gemma4_12b_prefill_xref.
    gemma4_12b_prefill_xref* xref  = nullptr;
    const char*              name  = nullptr;
    int                      layer = -1;
    const u8*                w     = nullptr;
    const bf16*              x     = nullptr;
    int                      L = 0, K = 0, N = 0;

    /// \brief Replay this projection on the host and score the array against it.
    /// \note Only the first xref->rows() rows are recomputed: matmul rows are independent,
    ///       so a prefix costs a fraction of the full replay and still exercises every
    ///       weight block the output columns touch.
    void _cross_reference() {
        if (xref == nullptr || !xref->enabled(layer)) return;
        const int rows = xref->rows(L);
        if (rows <= 0) return;

        std::vector<bf16> want((size_t)rows * N);
        gemma4_12b_cpu::dequant_mm(want.data(), w, x, rows, K, N);
        xref->check(name, layer, dst->data(), want.data(), N, N, rows);
    }
#endif

    void wait() {
        if (dst == nullptr) return;
        flm::npu::wait_for_completion(run, "gemma4_12b_mm_run");
        dst->sync_from_device();
#ifdef ENABLE_DEBUG
        this->_cross_reference();
#endif
        dst = nullptr;
    }
#else
    void wait() {}
#endif
};

/// \brief Allocate `n` elements of activation scratch for the compiled in backend: a
///        device BO when a stage runs on the NPU, plain host memory otherwise.
inline buffer<bf16> gemma4_12b_scratch_alloc(npu_app_manager* mgr, size_t n) {
#if GEMMA4_12B_SCRATCH_ON_DEVICE
    return mgr->create_bo_buffer<bf16>(n);
#else
    (void)mgr;
    return buffer<bf16>(n);
#endif
}

/// \brief Push host side scratch to the device, a no-op when the scratch is host memory.
/// \note  buffer<T>::sync_to_device() asserts on a non BO buffer, so this cannot just be
///        called unconditionally.
inline void gemma4_12b_scratch_push(buffer<bf16>& b) {
#if GEMMA4_12B_SCRATCH_ON_DEVICE
    b.sync_to_device();
#else
    (void)b;
#endif
}

/// \brief Pull device side scratch back to the host, a no-op when the scratch is host memory.
///
/// buffer<T>::data() is the bo's own host mapping, so these two are a cache invalidate and a
/// cache flush of one allocation rather than copies between two. xrt gives no ordering
/// beyond that, which makes the pairing rule strict and symmetric:
///
///   - pull() before a host stage reads anything an npu run wrote -- rope, the rms norms,
///     the glu multiply, the kv cache unroll. Without it the host can read lines the
///     array's write never invalidated.
///   - push() immediately after *any* host write to a bo, whether or not an npu run is
///     about to read it. An unflushed host write leaves dirty lines that evict at an
///     arbitrary later time, landing on top of whatever the array has written into that
///     same allocation since. "Immediately" is the whole rule: a push that is correct but
///     late still leaves a window in which an intervening run can be clobbered.
///
/// \note buffer<T>::sync_from_device() asserts on a non BO buffer, so like push() this
///       cannot just be called unconditionally: PREFILL_MM=cpu allocates plain host memory.
inline void gemma4_12b_scratch_pull(buffer<bf16>& b) {
#if GEMMA4_12B_SCRATCH_ON_DEVICE
    b.sync_from_device();
#else
    (void)b;
#endif
}

/// \brief Geometry of the one pass the whole prompt makes over the layer stack.
struct gemma4_12b_prefill_mm_shape {
    static constexpr int L_ALIGN      = 512; //!< gen_dequant_mm_512 needs L % 512 == 0
    static constexpr int L_ATTN_ALIGN = 128; //!< one query round of the attention overlay

    int L        = 0;  //!< real rows of the prompt
    int L_begin  = 0;  //!< absolute position of row 0
    int L_end    = 0;  //!< L_begin + L
    int L_padded = 0;  //!< L rounded up to L_ALIGN, the row count every matmul run sees
    int L_attn   = 0;  //!< L rounded up to L_ATTN_ALIGN, the rows the attention run covers

    void update(int L_begin, int L) {
        this->L        = L;
        this->L_begin  = L_begin;
        this->L_end    = L_begin + L;
        this->L_padded = (L + L_ALIGN - 1) / L_ALIGN * L_ALIGN;
        //!< always <= L_padded, since L_ALIGN is a multiple of L_ATTN_ALIGN: the attention
        //!< can cover fewer rows than the matmuls without running off the scratch
        this->L_attn   = (L + L_ATTN_ALIGN - 1) / L_ATTN_ALIGN * L_ATTN_ALIGN;
    }
};

/// \brief One maximal run of image-or-video soft tokens, in absolute positions.
///
/// Queries inside a block see the whole block, in both directions; everything else stays
/// causal. A block is exactly
/// one image (or one video frame) and never spans two: the processor always emits the
/// boi/eoi markers around each item, and those are plain text, which breaks the run.
struct gemma4_12b_vision_block {
    int begin = 0;  //!< absolute position of the first soft token
    int end   = 0;  //!< one past the last, so the width is end - begin
};

/// \brief The vision blocks of one prefill, in left to right order.
///
/// Built once per prompt from the token ids alone -- upstream derives its
/// mm_token_type_ids by the same plain equality test, so there is no hidden state to
/// carry. A text only or audio only prompt leaves this empty and the attention runs
/// exactly as it did before.
///
/// \note Audio soft tokens are deliberately NOT blocks. Upstream groups image and video
///       only; audio is masked exactly like text. This is the most surprising rule of the
///       whole masking scheme, so it is called out here rather than left implicit.
struct gemma4_12b_prefill_blocks {
    std::vector<gemma4_12b_vision_block> blocks;

    bool   empty() const { return this->blocks.empty(); }
    size_t size()  const { return this->blocks.size(); }
    void   clear()       { this->blocks.clear(); }

    /// \brief Find the blocks of `tokens`, whose row 0 sits at absolute position `L_begin`.
    /// \param image_soft_id  the <|image|> id (258880 on the released checkpoint)
    /// \param video_soft_id  the <|video|> id (258884); pass image_soft_id again to ignore
    /// \note A block that started in an earlier prefill chunk is not recoverable from this
    ///       chunk's ids alone. The processor never splits an item across calls, so it
    ///       cannot happen in practice; a block flush against row 0 is reported instead of
    ///       being silently truncated.
    void build(const int* tokens, int L, int L_begin, int image_soft_id, int video_soft_id) {
        this->blocks.clear();
        int run_begin = -1;
        for (int i = 0; i < L; i++) {
            const bool vision = (tokens[i] == image_soft_id || tokens[i] == video_soft_id);
            if (vision) {
                if (run_begin < 0) run_begin = i;
            }
            else if (run_begin >= 0) {
                this->blocks.push_back({L_begin + run_begin, L_begin + i});
                run_begin = -1;
            }
        }
        if (run_begin >= 0) {
            this->blocks.push_back({L_begin + run_begin, L_begin + L});
        }

        DEBUG_BLOCK(1,
        if (!this->blocks.empty()) {
            if (this->blocks.front().begin == L_begin && L_begin > 0) {
                header_print_r("warning", "prefill: a vision block is flush against the start of"
                               " this chunk; if it was split from an earlier one its mask is"
                               " truncated");
            }
            std::string line = "vision blocks:";
            for (const gemma4_12b_vision_block& b : this->blocks) {
                line += " [" + std::to_string(b.begin) + ", " + std::to_string(b.end) + ")";
            }
            header_print("prefill-media", line);
        }
        )
    }
};

/// \brief The activation scratch whose width the two layer kinds agree on.
struct gemma4_12b_mm_common_buffers {
    buffer<bf16> x;       //!< normed hidden state, and the output of o_proj / down_proj
    buffer<bf16> upgate;  //!< [ up | gate ], the gate half with the GeLU already applied
    buffer<bf16> hid;     //!< gelu_tanh(gate) * up, i.e. the input of down_proj

    int L_alloc = 0;      //!< rows the current allocation was cut for

    /// \brief Cut the scratch to exactly `L` rows, a no-op when it already matches.
    /// \return true when something was allocated
    bool reserve(npu_app_manager* mgr, gemma4_12b_desc* desc, int L) {
        if (L == L_alloc) {
            //!< The padded rows past the real tokens still hold what the previous prompt of this size left, and the array reads
            //!< them: a prompt that follows a restored state (same padded size as the request before it) would not see the zeros
            //!< a cold prompt gets from a fresh allocation. Start every prompt from zeros.
            this->x.memset((bf16)0);
            this->upgate.memset((bf16)0);
            this->hid.memset((bf16)0);
            gemma4_12b_scratch_push(this->x);
            gemma4_12b_scratch_push(this->upgate);
            gemma4_12b_scratch_push(this->hid);
            return false;
        }

        this->x      = gemma4_12b_scratch_alloc(mgr, (size_t)L * desc->D);
        this->upgate = gemma4_12b_scratch_alloc(mgr, (size_t)L * desc->intermediate_size * 2);
        this->hid    = gemma4_12b_scratch_alloc(mgr, (size_t)L * desc->intermediate_size);

        //!< buffer::data() is the bo's own host mapping, so sync_to_device() is a cache
        //!< flush of this very memory rather than a copy into a separate device buffer.
        //!< A memset that is never flushed leaves the whole allocation dirty in cache, and
        //!< those lines evict to dram at arbitrary later times -- on top of whatever the
        //!< array wrote in the meantime. Every buffer touched on the host here therefore
        //!< has to be pushed, including the ones only the array ever writes.
        this->x.memset((bf16)0);
        this->upgate.memset((bf16)0);
        this->hid.memset((bf16)0);
        gemma4_12b_scratch_push(this->x);
        gemma4_12b_scratch_push(this->upgate);
        gemma4_12b_scratch_push(this->hid);

        this->L_alloc = L;
        return true;
    }
};

/// \brief The q / k / v / attention scratch of *one* layer kind.
///
/// The sliding and global layers disagree on head dim and kv head count, so their
/// DQ / DK / DV disagree too. One instance lives in each gemma4_12b_mm_attn_block,
/// sized for that block's own dims: neither kind pays for the other being wider.
///
/// \see gemma4_12b_mm_common_buffers for the allocation and padding rules, which
///      are the same here.
struct gemma4_12b_mm_attn_buffers {
    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> attn;    //!< attention output, i.e. the input of o_proj

    //!< The sliding window keys/values the prompt can see, unrolled from the decoding ring
    //!< into linear order: [swa_rows][_DK] then [swa_rows][_DV], which is the layout the
    //!< DH_256 overlay reads and the same K-half / V-half shape the global kind's decoding
    //!< cache already has. Allocated from the prefill scratch (device bo on an array build)
    //!< and left empty on the global kind.
    //!<
    //!< \note One allocation rather than two because the overlay takes a *single* kv run
    //!<       argument and finds V at half of it. This is prefill staging only -- the
    //!<       decoding ring in gemma4_12b_npu_detail::kv_caches is untouched, and nothing
    //!<       outside _attend_sliding() ever looks at these rows.
    buffer<bf16> swa_kv;
    int swa_rows = 0;     //!< row capacity of each half of swa_kv

    //!< the two halves of swa_kv. Equal sized, because SWA_DV == SWA_DK.
    bf16* swa_k_ptr() { return swa_kv.data(); }
    bf16* swa_v_ptr() { return swa_kv.data() + (size_t)swa_rows * _DK; }

    // dims of the layer kind that owns this, set by init()
    int _DQ = 0, _DK = 0, _DV = 0;
    int sliding = 0;      //!< sliding window of this kind, 0 on the global kind
    int L_alloc = 0;      //!< rows the current allocation was cut for

    /// \param sliding window length, or 0 for a global kind that needs no unrolled ring
    void init(int DQ, int DK, int DV, int sliding) {
        this->_DQ     = DQ;
        this->_DK     = DK;
        this->_DV     = DV;
        this->sliding = sliding;
    }

    /// \brief Cut the scratch to exactly `L` rows, a no-op when it already matches.
    /// \return true when something was allocated
    bool reserve(npu_app_manager* mgr, int L) {
        if (L == L_alloc) {
            //!< see gemma4_12b_mm_common_buffers::reserve(): a reused scratch starts from zeros too
            this->q.memset((bf16)0);
            this->k.memset((bf16)0);
            this->v.memset((bf16)0);
            this->attn.memset((bf16)0);
            gemma4_12b_scratch_push(this->q);
            gemma4_12b_scratch_push(this->k);
            gemma4_12b_scratch_push(this->v);
            gemma4_12b_scratch_push(this->attn);
            return false;
        }

        this->q    = gemma4_12b_scratch_alloc(mgr, (size_t)L * _DQ);
        this->k    = gemma4_12b_scratch_alloc(mgr, (size_t)L * _DK);
        this->v    = gemma4_12b_scratch_alloc(mgr, (size_t)L * _DV);
        this->attn = gemma4_12b_scratch_alloc(mgr, (size_t)L * _DQ);

        //!< all four for the reason spelled out in gemma4_12b_mm_common_buffers::reserve()
        this->q.memset((bf16)0);
        this->k.memset((bf16)0);
        this->v.memset((bf16)0);
        this->attn.memset((bf16)0);
        gemma4_12b_scratch_push(this->q);
        gemma4_12b_scratch_push(this->k);
        gemma4_12b_scratch_push(this->v);
        gemma4_12b_scratch_push(this->attn);

        if (sliding > 0) {
            //!< (sliding - 1) history rows plus the new ones, and then one more query round
            //!< on top: the DH_256 overlay streams keys to the end of the 128 row block that
            //!< holds its last query row, which is past the last real key whenever the chunk
            //!< is not a whole number of rounds. Those rows are masked off but still read, so
            //!< they have to be inside the allocation. See _attend_sliding_npu_start().
            this->swa_rows = sliding + L + gemma4_12b_prefill_mm_shape::L_ATTN_ALIGN;
            //!< the unrolled ring is written by the host and pushed like the rest of the
            //!< scratch, so it has to come from the same allocator: a plain host buffer
            //!< here trips the bo_ assert in sync_to_device() on an array build.
            this->swa_kv = gemma4_12b_scratch_alloc(mgr, (size_t)swa_rows * (_DK + _DV));
            this->swa_kv.memset((bf16)0);
            gemma4_12b_scratch_push(this->swa_kv);
        }

        this->L_alloc = L;
        return true;
    }
};

/// \brief Handle on a standard attention run that has been issued to the array but not
///        waited on yet.
///
/// The array's attention and the host's vision block fix-up work on disjoint buffers --
/// _stage_vision_blocks() copies everything the fix-up needs out of q/k/v first, exactly
/// so that they do -- which leaves the recompute free to run while the array does. The
/// issuing side hands back one of these, does the host work, then joins here; only after
/// the join may anything touch bufs.attn.
///
/// `active == false` means nothing was issued and the caller owes the host kernel a run.
struct gemma4_12b_attn_run {
#if GEMMA4_12B_ATTN_ON_NPU
    flm_rt::run run;
#endif
    bool active = false;

    void wait() {
#if GEMMA4_12B_ATTN_ON_NPU
        if (active) {
            flm::npu::wait_for_completion(run, "gemma4_12b_attn_run");
        }
#endif
    }
};

/// \brief q/k/v projections, their norms, rope, attention and o projection of one
///        layer kind.
struct gemma4_12b_mm_attn_block {
    gemma4_12b_desc*              desc    = nullptr;
    gemma4_12b_npu_sequence*      seq_gen = nullptr;
    npu_app_manager*              mm_app_manager = nullptr;
    //!< the attention overlay of this layer kind, null when the kind runs on the host
    npu_app_manager*              attn_app_manager = nullptr;
    //!< where the activation scratch is allocated from. Any manager will do -- a bo is
    //!< created on the *device*, not on a hw context, so one allocation can be bound to
    //!< runs on either bitstream -- but it must not be null, and PREFILL_MM=cpu leaves
    //!< mm_app_manager null while the attention still needs real bos.
    npu_app_manager*              bo_manager = nullptr;
    gemma4_12b_mm_common_buffers* common  = nullptr;  //!< x, shared with the mlp block
    gemma4_12b_layer_type_t       layer_type = gemma4_12b_swa_layer;

    //!< q / k / v / attn of this layer kind only, sized for the dims below
    gemma4_12b_mm_attn_buffers bufs;

    //!< the prompt's image/video blocks, owned by gemma4_12b_prefill_mm_context and
    //!< rebuilt per prefill. Null or empty means "plain causal", which is every text only
    //!< prompt, and then nothing below this line ever runs.
    const gemma4_12b_prefill_blocks* blocks = nullptr;

    //!< Plain host staging for the block fix-up, never a device BO even when the standard
    //!< attention runs on the array. The fix-up is a host kernel reading q/k/v that the
    //!< array has just written and writing rows the array has just written: pointing it at
    //!< the BO mappings would interleave host and accelerator traffic on the same memory
    //!< banks for the whole of the recompute. One bounded copy in and one out is cheaper,
    //!< and it keeps the fix-up independent of where the standard attention ran.
    std::vector<bf16> stage_q;
    std::vector<bf16> stage_k;
    std::vector<bf16> stage_v;
    std::vector<bf16> stage_out;

    //!< True when this prefill's block fix-up goes to attn_global_image instead of the host
    //!< kernel. Decided once per prefill by _image_route(), *before* _stage_vision_blocks()
    //!< runs, because the two paths stage different things: the device path never fills
    //!< stage_k / stage_v (it reads the decoding cache in place) and so has no way back to
    //!< the host kernel afterwards. Everything that can disqualify the device -- a missing
    //!< bitstream, a block the overlay cannot shape, a cache too short -- is therefore
    //!< checked up front, and any one of them leaves this false and the host path intact.
    bool image_npu = false;

    //!< one block's slice of the staging buffers above, resolved by _stage_vision_blocks()
    struct vision_stage {
        int    blk_begin = 0, blk_end = 0;  //!< the block, in absolute positions
        int    q_lo = 0, q_hi = 0;          //!< query rows of it this chunk holds
        int    kv_lo = 0, kv_hi = 0;        //!< key rows the recompute needs for them
        size_t q_off = 0;                   //!< first row in stage_q / stage_out
        size_t kv_off = 0;                  //!< first row in stage_k / stage_v
    };
    std::vector<vision_stage> stages;

    uint32_t SLIDING_LENGTH = 0;

    int L_seq = 0;  //!< rows the projection sequences are generated for

    //!< The attention sequence bakes in the absolute position of query row 0 as well as the
    //!< row count -- unlike the projections, which only see a shape -- so it is regenerated
    //!< whenever any of these four change. All of them are per prefill, not per layer, so a
    //!< 24 layer kind pays for one regeneration and then reuses it.
    int    L_attn_seq  = 0;   //!< query rows, and -1 for "no sequence generated yet"
    int    L_begin_seq = -1;
    int    kv_rows_seq = 0;
    size_t kv_elems_seq = 0;

    //!< The same guard for the image overlay's sequence. It takes no absolute query
    //!< position -- the mask does not depend on one -- so the key is (q rows, visible kv
    //!< rows, cache geometry). One image is one regeneration for all 24 global layers;
    //!< several blocks in one prompt cost one regeneration each, per distinct triple.
    int    img_q_rows_seq  = -1;
    int    img_kv_valid_seq = 0;
    int    img_kv_rows_seq  = 0;
    size_t img_kv_elems_seq = 0;

    // layer kind specific dims, resolved once in build()
    int _DH = 0, _DQ = 0, _DK = 0, _DV = 0;
    int num_attn_heads = 0, num_kv_heads = 0;
    bool is_global   = false;
    bool has_v_proj  = false;

    //!< the weight offset, D_in and D_out are baked into each control sequence, so the
    //!< only run argument of a projection app is the packed weight buffer of the layer
    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;  //!< left unbuilt when the layer kind has no v projection
    npu_app o_proj;

    //!< the prefill attention of this layer kind, left unbuilt when it runs on the host.
    //!< Its run arguments are (attn, q, kv_cache); everything else rides the RTPs and the
    //!< control sequence, see gemma4_12b_npu_sequence::gen_attn_prefill_global().
    npu_app attn;

    //!< The block-masked variant, global layers only, and only when the bitstream was
    //!< found -- see gemma4_12b_prefill_mm_context::set_vision_blocks(), which registers it
    //!< lazily on the first prompt that actually carries an image. Null manager means the
    //!< block fix-up stays on the host, which is also the state of every sliding block.
    npu_app_manager* image_app_manager = nullptr;
    npu_app          image_attn;

    //!< Query rows and output rows of one image run. Separate from bufs.q / bufs.attn on
    //!< purpose: the overlay works in whole 128 row chunks, so a block of, say, 260 rows is
    //!< run as 384 and the 124 rows of tail come back computed against the block mask,
    //!< which is the wrong answer for whatever follows the block. Writing that straight
    //!< into bufs.attn would corrupt rows the array had already got right. Here it lands in
    //!< scratch that only _commit_vision_blocks() reads, row by row, within the block.
    //!<
    //!< \note The padded q rows are zeroed rather than left dirty. A zero query gives a
    //!<       uniform softmax over the visible keys -- finite, sum > 0, no NaN -- whereas
    //!<       stale bf16 can carry an inf into the running max and poison the whole tile.
    buffer<bf16> img_q;
    buffer<bf16> img_out;
    int          img_rows = 0;   //!< row capacity of img_q / img_out

#ifdef ENABLE_DEBUG
    //!< the cpu/npu cross reference, owned by gemma4_12b_prefill_mm_context. Never null in
    //!< a debug build; whether it scores anything is up to xref->enabled(layer_idx).
    gemma4_12b_prefill_xref* xref = nullptr;
    //!< index of the layer currently in forward(), which is what selects the scored layers.
    //!< Set per layer by gemma4_12b_prefill_mm_context::forward().
    int layer_idx = -1;
#endif

    /// \param attn_app_manager the attention overlay of this layer kind, or null to keep
    ///                         the attention on the host
    /// \param bo_manager       what the activation scratch is allocated from; see the
    ///                         member of the same name
    void build(gemma4_12b_desc*              desc,
               gemma4_12b_npu_sequence*      seq_gen,
               gemma4_12b_mm_common_buffers* common,
               gemma4_12b_layer_type_t       layer_type,
               npu_app_manager*              mm_app_manager,
               npu_app_manager*              attn_app_manager,
               npu_app_manager*              bo_manager) {
        this->desc             = desc;
        this->seq_gen          = seq_gen;
        this->mm_app_manager   = mm_app_manager;
        this->attn_app_manager = attn_app_manager;
        this->bo_manager       = bo_manager;
        this->common         = common;
        this->layer_type     = layer_type;
        this->SLIDING_LENGTH = desc->sliding_window;
        this->is_global      = is_global_layer(layer_type);

        gemma4_12b_layer_weight_def& w = desc->weight_desc(layer_type);
        this->has_v_proj = w.has_v_proj;

        this->_DH = is_global ? (int)desc->DH : (int)desc->SWA_DH;
        this->_DQ = is_global ? (int)desc->DQ : (int)desc->SWA_DQ;
        this->_DK = is_global ? (int)desc->DK : (int)desc->SWA_DK;
        this->_DV = is_global ? (int)desc->DV : (int)desc->SWA_DV;
        this->num_attn_heads = _DQ / _DH;
        this->num_kv_heads   = _DK / _DH;

        // a global layer attends the whole prefix straight out of the decoding cache and
        // needs no unrolled ring, which is what the 0 stands for
        this->bufs.init(_DQ, _DK, _DV, is_global ? 0 : (int)SLIDING_LENGTH);

#if GEMMA4_12B_MM_ON_NPU
        this->q_proj = mm_app_manager->create_app();
        this->k_proj = mm_app_manager->create_app();
        this->o_proj = mm_app_manager->create_app();
        if (has_v_proj) {
            this->v_proj = mm_app_manager->create_app();
        }
        // the control sequences are generated for a row count, which is only known once
        // a prompt arrives; see reserve()
#endif
        if (attn_app_manager != nullptr) {
            this->attn = attn_app_manager->create_app();
        }
    }

    /// \brief Hand the block the image overlay, once its bitstream is known to exist.
    ///
    /// Called at most once, from set_vision_blocks() on the first prompt that carries a
    /// vision block. Kept out of build() because build() runs for every model load and the
    /// overlay is a fourth hw context that a text-only session should never pay for.
    void bind_image_overlay(npu_app_manager* mgr) {
        if (mgr == nullptr || this->image_app_manager != nullptr) return;
        this->image_app_manager = mgr;
        this->image_attn        = mgr->create_app();
    }

    /// \brief Make the block ready for a prompt of `L_padded` rows.
    ///
    /// The projection sequences bake the row count in, so they are regenerated whenever
    /// it changes, and the scratch is re-cut to match.
    ///
    /// \note The attention sequence is *not* built here. It encodes the kv cache geometry
    ///       as well as the shape, and the cache is a per layer argument this call has no
    ///       access to, so _attend_global() regenerates it on demand instead. The guard
    ///       there makes that one regeneration per prefill, shared by the kind's 24 layers.
    /// \return true when something was allocated
    bool reserve(int L_padded) {
        const bool reallocated = this->bufs.reserve(bo_manager, L_padded);

#if GEMMA4_12B_MM_ON_NPU
        if (L_padded != L_seq) {
            gemma4_12b_layer_weight_def& w = desc->weight_desc(layer_type);
            const int no_act = gemma4_12b_npu_sequence::DEQUANT_MM_ACT_NONE;

            seq_gen->gen_dequant_mm_512(this->q_proj.seq(), L_padded, desc->D, _DQ, w.attn_q.offset,      0, no_act);
            seq_gen->gen_dequant_mm_512(this->k_proj.seq(), L_padded, desc->D, _DK, w.attn_k.offset,      0, no_act);
            seq_gen->gen_dequant_mm_512(this->o_proj.seq(), L_padded, _DQ, desc->D, w.attn_output.offset, 0, no_act);
            if (has_v_proj) {
                seq_gen->gen_dequant_mm_512(this->v_proj.seq(), L_padded, desc->D, _DV, w.attn_v.offset,  0, no_act);
            }
            L_seq = L_padded;
        }
#endif

        return reallocated;
    }

    /// \brief Issue `y = x * W^T` for one projection on the compiled in backend.
    ///
    /// The returned handle is only meaningful on the NPU backend, where the run is still
    /// in flight and the caller may overlap host work with it; see gemma4_12b_mm_run.
    /// Either way `y` holds the result on the host once the handle has been waited on.
    ///
    /// \param name     what this projection is called in the cross reference report
    /// \param app      the projection app, unused on the CPU backend
    /// \param w_offset byte offset of the weight region inside `weights`
    gemma4_12b_mm_run _mm_start(const char* name, npu_app& app,
                                buffer<bf16>& y, buffer<bf16>& x,
                                buffer<u8>& weights, size_t w_offset,
                                int L, int K, int N) {
#if GEMMA4_12B_MM_ON_NPU
        (void)w_offset; (void)L; (void)K; (void)N; //!< all three are baked into the sequence
        gemma4_12b_mm_run h;
        h.run = app.create_run(y, x, weights);
        h.dst = &y;
        h.run.start();
#ifdef ENABLE_DEBUG
        //!< the host kernel is the reference, so the replay is deferred to wait(): starting
        //!< it here would serialise the overlap the caller asked for by using _mm_start()
        h.xref     = xref;
        h.name     = name;
        h.layer    = layer_idx;
        h.w        = weights.data() + w_offset;
        h.x        = x.data();
        h.L        = L;
        h.K        = K;
        h.N        = N;
#else
        (void)name;
#endif
        return h;
#else
        (void)app; (void)name;
        gemma4_12b_cpu::dequant_mm(y.data(), weights.data() + w_offset, x.data(), L, K, N);
        return gemma4_12b_mm_run{};
#endif
    }

    /// \brief _mm_start() followed immediately by wait(), for the projections nothing
    ///        overlaps with.
    void _mm(const char* name, npu_app& app, buffer<bf16>& y, buffer<bf16>& x,
             buffer<u8>& weights, size_t w_offset, int L, int K, int N) {
        _mm_start(name, app, y, x, weights, w_offset, L, K, N).wait();
    }

    /// \param shape          prompt geometry, see gemma4_12b_prefill_mm_shape::update()
    /// \param proj_weights   packed projections of this layer
    /// \param rope_weights   q_norm / k_norm (and the cos/sin table the decode path uses)
    /// \param decoding_cache kv cache of this layer as the decode path sees it
    /// \note On entry common->x holds input_layernorm(residual) and is already on the
    ///       device; on return it holds the attention output, and is on the host.
    void forward(const gemma4_12b_prefill_mm_shape& shape,
                 buffer<u8>&   proj_weights,
                 buffer<bf16>& rope_weights,
                 buffer<bf16>& decoding_cache) {
        PREFILL_PROF_INIT();
        const bf16* q_norm_w = rope_weights.data() + desc->get_q_norm_offset(layer_type);
        const bf16* k_norm_w = rope_weights.data() + desc->get_k_norm_offset(layer_type);
        const f32*  inv_freq = desc->inv_freq(layer_type).data();

        // ---- q / k / v projections -----------------------------------------
        // q_norm / k_norm are per head rms norms with a weight, applied before rope.
        // The 1/sqrt(head_dim) attention scale is folded into q_norm.weight. Each norm
        // is issued against the projection that follows it rather than after all three,
        // so the host stage runs while the NPU works on a tensor it does not need.
        gemma4_12b_layer_weight_def& w = desc->weight_desc(layer_type);
        const int D = (int)desc->D;

        _mm("q_proj", q_proj, bufs.q, common->x, proj_weights, w.attn_q.offset,
            shape.L, D, _DQ);

        gemma4_12b_mm_run k_run = _mm_start("k_proj", k_proj, bufs.k, common->x, proj_weights,
                                            w.attn_k.offset, shape.L, D, _DK);
        
        cpu_func::_rope_rms_batch(bufs.q.data(), _DQ, 0, shape.L_begin, shape.L,
                                  q_norm_w, _DH, inv_freq);
        gemma4_12b_scratch_push(bufs.q); //!< for the attention backend, not for k_proj
        k_run.wait();

        if (has_v_proj) {
            gemma4_12b_mm_run v_run = _mm_start("v_proj", v_proj, bufs.v, common->x, proj_weights,
                                                w.attn_v.offset, shape.L, D, _DV);
            cpu_func::_rope_rms_batch(bufs.k.data(), _DK, 0, shape.L_begin, shape.L,
                                      k_norm_w, _DH, inv_freq);
            gemma4_12b_scratch_push(bufs.k);
            v_run.wait();
            gemma4_12b_scratch_pull(bufs.v);
        }
        else {
            // attention_k_eq_v: the global layers store no v_proj and take V from the
            // *raw* k_proj output, before k_norm and before rope. Copy it out now.
            assert(_DK == _DV && "attention_k_eq_v requires DK == DV");
            memcpy(bufs.v.data(), bufs.k.data(), (size_t)shape.L * _DV * sizeof(bf16));
            cpu_func::_rope_rms_batch(bufs.k.data(), _DK, 0, shape.L_begin, shape.L,
                                      k_norm_w, _DH, inv_freq);
            gemma4_12b_scratch_push(bufs.k);
        }

        // value_norm is an rms norm *without* a scale (SimpleRMSNorm with_scale=False)
        cpu_func::_rms_norm_batch(bufs.v.data(), bufs.v.data(), nullptr, _DH, _DV,
                                  shape.L, 0, 0);
        gemma4_12b_scratch_push(bufs.v); //!< both branches above land here, v_proj result
                                         //!< and raw k_proj copy alike
        //!< no cross reference on q_norm / rope / v_norm: _rope_rms_batch and
        //!< _rms_norm_batch are the same cpu_func calls in both builds
        PREFILL_PROF_TICK("attn:qkv_proj");

        // ---- kv cache + attention ------------------------------------------
        // the cache syncs live in the two _attend_* below: only they know which layout
        // the cache has and which rows of it they touch
        //!< each _attend_* leaves bufs.attn ready for o_proj, on the device on an array
        //!< build. Pushing it here instead would be wrong on the npu attention path: the
        //!< array wrote those lines, and one bo is shared by all 24 layers of a kind, so a
        //!< blanket flush would drop this layer's dirty host lines -- the vision patch of
        //!< the *previous* layer -- on top of what the array just produced.
        if (is_global) _attend_global(shape, decoding_cache);
        else           _attend_sliding(shape, decoding_cache);
        PREFILL_PROF_TICK("attn:cache+attention");

        // ---- output projection ---------------------------------------------
        // o_proj overwrites common->x, which q/k/v are long done reading
        _mm("o_proj", o_proj, common->x, bufs.attn, proj_weights, w.attn_output.offset,
            shape.L, _DQ, D);
        PREFILL_PROF_TICK("attn:o_proj");
    }

    /// \brief Recompute, on the host, the query rows the vision blocks make bidirectional,
    ///        and overwrite what the standard attention produced for them.
    ///
    /// The standard attention -- host today, an xclbin tomorrow -- runs the plain causal /
    /// sliding mask and knows nothing about images. That is wrong for exactly the query
    /// rows that sit inside an image or video block: those rows must also see the *later*
    /// tokens of their own block. Rather than teach a fixed-function attention an
    /// arbitrary mask, the wrong rows are simply recomputed here and written over. There
    /// are at most 280 of them per image against a whole prompt, so the recompute is a
    /// rounding error next to the pass it corrects, and the standard path keeps a mask
    /// shape an accelerator can actually implement.
    ///
    /// \param k_src        keys as the standard attention saw them: the decoding cache for
    ///                     a global layer, the unrolled ring for a sliding one
    /// \param kv_pos_begin absolute position of k_src / v_src row 0
    /// \param L_kv         rows in k_src / v_src
    /// \param window       sliding window, or 0 for a global layer
    ///
    /// \note Every buffer this reads is pulled back from the device first, and the rows it
    ///       writes are pushed after, so the function is already correct once
    ///       GEMMA4_12B_ATTN_ON_NPU has an xclbin behind it. On a host build every one of
    ///       those syncs is a no-op.
    /// \brief Copy everything the host block fix-up will need out of q / k / v and into
    ///        plain host memory, and record which rows each block owns.
    ///
    /// Must run *before* the standard attention is launched. On an array build q, k and v
    /// are device BOs the accelerator is about to stream and attn is a device BO it is
    /// about to write; extracting first means the recompute never shares a buffer with the
    /// accelerator. That removes the bank contention between the host kernel and the array,
    /// and leaves the recompute free to overlap the array run.
    ///
    /// \brief Rows of img_q / img_out one stage occupies: its query rows rounded up to the
    ///        128 row chunk the overlay works in.
    static int _img_q_pad(const vision_stage& st) {
        return (st.q_hi - st.q_lo + 127) / 128 * 128;
    }

    /// \brief Decide whether this prefill's block fix-up can go to attn_global_image.
    ///
    /// Called from _stage_vision_blocks() once `stages` is resolved but before a single
    /// byte has been copied, because the answer changes what gets copied. Every condition
    /// is checked here and none is re-checked later: past this point the path is committed.
    ///
    /// \param cache_rows row capacity of the decoding cache the run would read, or 0 to
    ///                   force the host path (which is what the sliding kind passes)
    bool _image_route(int cache_rows) {
#if GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU
        if (cache_rows <= 0 || !is_global) return false;
        if (this->image_app_manager == nullptr) return false;  //!< no bitstream on this box

        for (const vision_stage& st : this->stages) {
            //!< The overlay's mask is "keys [0, L)", counted from row 0 of the buffer it is
            //!< handed. A global stage always has kv_lo == kv_pos_begin == 0, so the two
            //!< origins agree; anything else would need a shifted mask the kernel has no
            //!< RTP for.
            if (st.kv_lo != 0) return false;
            //!< the stream is rounded up to whole 128 row blocks, and every row of it is
            //!< read whether or not the mask keeps it
            const int kv_read = (st.kv_hi + 127) / 128 * 128;
            if (kv_read > cache_rows) return false;
            if (kv_read / 128 > 1024) return false;  //!< one BD sweep is 1024 iterations
        }
        return !this->stages.empty();
#else
        (void)cache_rows;
        return false;
#endif
    }

    /// \brief Grow img_q / img_out to cover the tallest stage, a no-op once they do.
    void _image_reserve(int rows) {
#if GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU
        if (rows <= this->img_rows) return;
        this->img_q   = gemma4_12b_scratch_alloc(bo_manager, (size_t)rows * _DQ);
        this->img_out = gemma4_12b_scratch_alloc(bo_manager, (size_t)rows * _DQ);
        //!< see gemma4_12b_mm_common_buffers::reserve(): a fresh bo's host mapping is dirty
        //!< in the cache, so it is written and flushed once before the array ever reads it
        this->img_q.memset((bf16)0);
        this->img_out.memset((bf16)0);
        gemma4_12b_scratch_push(this->img_q);
        gemma4_12b_scratch_push(this->img_out);
        this->img_rows = rows;
#else
        (void)rows;
#endif
    }

    /// \param k_src,v_src host readable keys/values for [kv_pos_begin, kv_pos_begin+L_kv)
    /// \param window      sliding window, or 0 for a global layer
    /// \param image_cache_rows row capacity of the decoding cache, when the fix-up may run
    ///                         on attn_global_image; 0 pins it to the host, see _image_route()
    void _stage_vision_blocks(const gemma4_12b_prefill_mm_shape& shape,
                              const bf16* k_src, const bf16* v_src,
                              int kv_pos_begin, int L_kv, int window,
                              int image_cache_rows = 0) {
        this->stages.clear();
        this->image_npu = false;
        if (this->blocks == nullptr || this->blocks->empty()) return;

        //!< q was written by the host (rope) and pushed to the device; pull it back so the
        //!< extract below sees exactly the rows the accelerator is given
        gemma4_12b_scratch_pull(bufs.q);

        const int kv_end_abs = kv_pos_begin + L_kv;
        size_t q_rows_total = 0, kv_rows_total = 0;

        for (const gemma4_12b_vision_block& b : this->blocks->blocks) {
            vision_stage st;
            st.blk_begin = b.begin;
            st.blk_end   = b.end;
            // the block clipped to the rows this prefill actually holds queries for
            st.q_lo = std::max(b.begin, shape.L_begin);
            st.q_hi = std::min(b.end,   shape.L_end);
            if (st.q_hi <= st.q_lo) continue;

            // Keys the recompute needs: the causal / windowed range of the *first* query
            // row, extended up to the end of the block. Everything above the block end is
            // masked for every query in it, and everything below the window is masked too.
            st.kv_lo = (window > 0) ? std::max(kv_pos_begin, st.q_lo - window + 1) : kv_pos_begin;
            st.kv_hi = std::min(b.end, kv_end_abs);
            if (st.kv_hi <= st.kv_lo) continue;

            st.q_off  = q_rows_total;
            st.kv_off = kv_rows_total;
            q_rows_total  += (size_t)(st.q_hi - st.q_lo);
            kv_rows_total += (size_t)(st.kv_hi - st.kv_lo);
            this->stages.push_back(st);
        }
        if (this->stages.empty()) return;

        this->image_npu = _image_route(image_cache_rows);

        //!< stage_out is the meeting point of both paths -- whoever computes the rows leaves
        //!< them here and _commit_vision_blocks() writes them back -- so it is sized either way
        this->stage_out.resize(q_rows_total * (size_t)_DQ);

        if (this->image_npu) {
            int img_rows_needed = 0;
            for (const vision_stage& st : this->stages) {
                img_rows_needed = std::max(img_rows_needed, _img_q_pad(st));
            }
            _image_reserve(img_rows_needed);
        }

        //!< The device path reads q out of bufs.q and the keys out of the decoding cache in
        //!< place, so none of the three input copies below are needed. They are the bulk of
        //!< the staging cost -- kv_rows_total is the whole prefix up to the block end, not
        //!< the block -- which is most of what moving the fix-up onto the array buys.
        bool host_inputs = !this->image_npu;
#if defined(ENABLE_DEBUG)
        //!< except when the cross reference is going to score this layer: then the host
        //!< kernel has to run too, on the same inputs, to be the reference
        if (xref != nullptr && xref->enabled(layer_idx)) host_inputs = true;
#endif
        if (!host_inputs) {
            this->stage_q.clear();
            this->stage_k.clear();
            this->stage_v.clear();
            return;
        }

        this->stage_q.resize(q_rows_total * (size_t)_DQ);
        this->stage_k.resize(kv_rows_total * (size_t)_DK);
        this->stage_v.resize(kv_rows_total * (size_t)_DV);

        for (const vision_stage& st : this->stages) {
            memcpy(this->stage_q.data() + st.q_off * (size_t)_DQ,
                   bufs.q.data() + (size_t)(st.q_lo - shape.L_begin) * _DQ,
                   (size_t)(st.q_hi - st.q_lo) * (size_t)_DQ * sizeof(bf16));
            memcpy(this->stage_k.data() + st.kv_off * (size_t)_DK,
                   k_src + (size_t)(st.kv_lo - kv_pos_begin) * _DK,
                   (size_t)(st.kv_hi - st.kv_lo) * (size_t)_DK * sizeof(bf16));
            memcpy(this->stage_v.data() + st.kv_off * (size_t)_DV,
                   v_src + (size_t)(st.kv_lo - kv_pos_begin) * _DV,
                   (size_t)(st.kv_hi - st.kv_lo) * (size_t)_DV * sizeof(bf16));
        }
    }

    /// \brief Recompute the staged block rows with the block term, into stage_out.
    ///
    /// Touches host memory only -- the staged copies _stage_vision_blocks() took and
    /// stage_out -- and nothing the standard attention reads or writes. That is what lets
    /// it run *while* the array is attending, see _attend_sliding(); the result is parked
    /// in stage_out until _commit_vision_blocks() writes it over bufs.attn.
    void _recompute_vision_blocks(int window) {
        for (const vision_stage& st : this->stages) {
            gemma4_12b_cpu::attention(this->stage_out.data() + st.q_off * (size_t)_DQ,
                                      this->stage_q.data()   + st.q_off * (size_t)_DQ,
                                      this->stage_k.data()   + st.kv_off * (size_t)_DK,
                                      this->stage_v.data()   + st.kv_off * (size_t)_DV,
                                      st.q_hi - st.q_lo, st.kv_hi - st.kv_lo,
                                      num_attn_heads, num_kv_heads, _DH,
                                      st.q_lo, st.kv_lo, window, _DK, _DV,
                                      st.blk_begin, st.blk_end);
        }
    }

    /// \brief Recompute the staged block rows on attn_global_image, into stage_out.
    ///
    /// The device counterpart of _recompute_vision_blocks(), and the reason the overlay
    /// exists. What it computes is the same thing the host kernel does for a *global*
    /// block: every query row of the block attends kv rows [0, block_end). The host kernel
    /// arrives there by unioning the causal term with the block term; the overlay has no
    /// causal term at all and is simply told where the keys stop, which for these rows is
    /// the same mask.
    ///
    /// The kv argument is the layer's decoding cache, bound exactly as _attend_global_npu()
    /// binds it -- same buffer, same halves, V found at cache.size()/2 -- so there is no
    /// second kv layout anywhere in this path and nothing to keep in step. Only q and o
    /// move, through img_q / img_out, and only because the overlay rounds the row count up
    /// to 128 and the rows of that padding must not be allowed near bufs.attn.
    ///
    /// \pre  `cache` is coherent on the device and the standard attention has finished
    ///       reading it; _attend_global() syncs it and joins before calling here
    /// \post stage_out holds the block rows, exactly as the host path leaves it
    void _recompute_vision_blocks_npu(const gemma4_12b_prefill_mm_shape& shape,
                                      buffer<bf16>& cache) {
#if GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU
        const size_t kv_elems = cache.size();
        const int    kv_rows  = (int)(kv_elems / 2 / (size_t)_DK);

        for (const vision_stage& st : this->stages) {
            const int q_rows = st.q_hi - st.q_lo;
            const int q_pad  = _img_q_pad(st);

            //!< the block's query rows, then zeros up to the chunk boundary. A zero query
            //!< row scores 0 against every key, so its softmax is uniform and finite -- the
            //!< padding computes a harmless average that _commit_vision_blocks() never reads.
            memcpy(img_q.data(),
                   bufs.q.data() + (size_t)(st.q_lo - shape.L_begin) * _DQ,
                   (size_t)q_rows * (size_t)_DQ * sizeof(bf16));
            if (q_pad > q_rows) {
                memset(img_q.data() + (size_t)q_rows * _DQ, 0,
                       (size_t)(q_pad - q_rows) * (size_t)_DQ * sizeof(bf16));
            }
            gemma4_12b_scratch_push(img_q);

            //!< the mask rides the RTPs, which ride the control sequence, so a differently
            //!< shaped block means a new sequence. One image is one regeneration for the
            //!< whole 24 layer kind; the layers after the first fall straight through.
            if (q_pad != img_q_rows_seq || st.kv_hi != img_kv_valid_seq ||
                kv_rows != img_kv_rows_seq || kv_elems != img_kv_elems_seq) {
                seq_gen->gen_attn_prefill_global_image(this->image_attn.seq(),
                                                       (uint32_t)q_pad, (uint32_t)st.kv_hi,
                                                       (uint32_t)kv_rows, kv_elems);
                this->img_q_rows_seq   = q_pad;
                this->img_kv_valid_seq = st.kv_hi;
                this->img_kv_rows_seq  = kv_rows;
                this->img_kv_elems_seq = kv_elems;
            }

            flm_rt::run run = this->image_attn.create_run(img_out, img_q, cache);
            run.start();
            flm::npu::wait_for_completion(run, "image_attn");

            //!< the array wrote img_out; the memcpy below is a host read of it
            gemma4_12b_scratch_pull(img_out);
            memcpy(this->stage_out.data() + st.q_off * (size_t)_DQ, img_out.data(),
                   (size_t)q_rows * (size_t)_DQ * sizeof(bf16));

#ifdef ENABLE_DEBUG
            //!< _stage_vision_blocks() kept the host inputs alive for exactly this
            if (xref != nullptr && xref->enabled(layer_idx)) {
                std::vector<bf16> want((size_t)q_rows * _DQ);
                gemma4_12b_cpu::attention(want.data(),
                                          this->stage_q.data() + st.q_off * (size_t)_DQ,
                                          this->stage_k.data() + st.kv_off * (size_t)_DK,
                                          this->stage_v.data() + st.kv_off * (size_t)_DV,
                                          q_rows, st.kv_hi - st.kv_lo,
                                          num_attn_heads, num_kv_heads, _DH,
                                          st.q_lo, st.kv_lo, /*window*/ 0, _DK, _DV,
                                          st.blk_begin, st.blk_end);
                xref->check("attention_out_image", layer_idx,
                            this->stage_out.data() + st.q_off * (size_t)_DQ, want.data(),
                            _DQ, _DQ, q_rows);
            }
#endif
        }
#else
        (void)shape; (void)cache;
#endif
    }

    /// \brief Overwrite the rows the standard attention got wrong with what
    ///        _recompute_vision_blocks() produced.
    ///
    /// The one bounded write back into bufs.attn, and the only part of the fix-up that
    /// touches a buffer the accelerator owns: it must run after the standard attention
    /// has finished writing attn.
    void _commit_vision_blocks(const gemma4_12b_prefill_mm_shape& shape) {
        //!< the standard attention wrote attn; make the untouched rows coherent on the
        //!< host before patching, or the push below would flush stale lines over them
        gemma4_12b_scratch_pull(bufs.attn);

        int patched_rows = 0;
        for (const vision_stage& st : this->stages) {
            memcpy(bufs.attn.data() + (size_t)(st.q_lo - shape.L_begin) * _DQ,
                   this->stage_out.data() + st.q_off * (size_t)_DQ,
                   (size_t)(st.q_hi - st.q_lo) * (size_t)_DQ * sizeof(bf16));
            patched_rows += st.q_hi - st.q_lo;
        }

        //!< o_proj reads attn next, from the device on an array build
        gemma4_12b_scratch_push(bufs.attn);

        DEBUG_BLOCK(2,
        header_print("info", "vision block fix-up: recomputed " + std::to_string(patched_rows)
                     + " query row(s) over " + std::to_string(this->stages.size()) + " block(s)");
        )
    }

    /// \brief Recompute the staged block rows and overwrite the rows the standard
    ///        attention got wrong, back to back.
    ///
    /// Pairs with _stage_vision_blocks(), and must run after the standard attention has
    /// finished writing bufs.attn. Callers that can overlap the recompute with the array
    /// use the two halves directly instead.
    ///
    /// \param cache the layer's decoding cache when the fix-up may run on the array, else
    ///              null. Which of the two it is was settled in _stage_vision_blocks();
    ///              `image_npu` only reads back that decision, it does not re-take it.
    void _patch_vision_blocks(const gemma4_12b_prefill_mm_shape& shape, int window,
                              buffer<bf16>* cache = nullptr) {
        if (this->stages.empty()) return;
        if (this->image_npu && cache != nullptr) _recompute_vision_blocks_npu(shape, *cache);
        else                                     _recompute_vision_blocks(window);
        _commit_vision_blocks(shape);
    }

    /// \brief Run the global prefill attention on the array, if it is compiled in and the
    ///        cache geometry allows it.
    ///
    /// \return false when nothing ran and the caller must fall back to the host kernel
    /// \post   on true, bufs.attn holds the standard (causal, image-unaware) attention
    ///         output on the *device*; the host mapping is stale
    bool _attend_global_npu(const gemma4_12b_prefill_mm_shape& shape, buffer<bf16>& cache) {
#if GEMMA4_12B_ATTN_GLOBAL_ON_NPU
        const size_t kv_elems = cache.size();
        //!< the real row capacity of the cache, i.e. MAX_L, and deliberately not
        //!< shape.L_padded: the overlay's testbench passes its own padded length there
        //!< because its kv buffer is exactly that tall, which is not true here
        const int    kv_rows  = (int)(kv_elems / 2 / (size_t)_DK);

        //!< the last query round streams keys to the end of the 128 row block holding its
        //!< last query row, which sits past L_end whenever L is not a multiple of 128.
        //!< The causal mask drops every one of those rows, but the dma still reads them,
        //!< so they have to be inside the cache.
        const int kv_end = (shape.L_begin + shape.L_attn + 127) / 128 * 128;
        if (kv_end > kv_rows) {
            //!< only reachable within 128 rows of the context cap; the host kernel reads
            //!< exactly L_end rows and has no such tail
            DEBUG_BLOCK(1,
            header_print("info", "global attention: kv tail read to " + std::to_string(kv_end)
                         + " exceeds the " + std::to_string(kv_rows)
                         + " row cache, falling back to the host");
            )
            return false;
        }

        //!< L_begin rides the control sequence, not a run argument, so a new prompt
        //!< position means a new sequence. Same shape and same cache means no work, which
        //!< is what all 24 layers after the first hit.
        if (shape.L_begin != L_begin_seq || shape.L_attn != L_attn_seq ||
            kv_rows != kv_rows_seq || kv_elems != kv_elems_seq) {
            seq_gen->gen_attn_prefill_global(this->attn.seq(),
                                             (uint32_t)shape.L_begin, (uint32_t)shape.L_attn,
                                             (uint32_t)kv_rows, kv_elems);
            this->L_begin_seq  = shape.L_begin;
            this->L_attn_seq   = shape.L_attn;
            this->kv_rows_seq  = kv_rows;
            this->kv_elems_seq = kv_elems;
        }

        //!< q was pushed in forward(), the cache by the append above; attn is written by
        //!< the array and read straight from the device by o_proj, so it needs no sync
        flm_rt::run run = this->attn.create_run(bufs.attn, bufs.q, cache);
        run.start();
        flm::npu::wait_for_completion(run, "attn");

#ifdef ENABLE_DEBUG
        //!< the host kernel is the reference. Only pay for the pull and the replay on the
        //!< layers xref actually scores -- attn is L x DQ, which is megabytes per layer.
        if (xref != nullptr && xref->enabled(layer_idx)) {
            const int rows = xref->rows(shape.L);
            std::vector<bf16> want((size_t)rows * _DQ);
            gemma4_12b_cpu::attention(want.data(), bufs.q.data(),
                                      cache.data(), cache.data() + kv_elems / 2,
                                      rows, shape.L_end, num_attn_heads, num_kv_heads, _DH,
                                      shape.L_begin, /*kv_pos_begin*/ 0, /*window*/ 0,
                                      _DK, _DV);
            gemma4_12b_scratch_pull(bufs.attn);
            xref->check("attention_out", layer_idx, bufs.attn.data(), want.data(),
                        _DQ, _DQ, rows);
        }
#endif
        return true;
#else
        (void)shape; (void)cache;
        return false;
#endif
    }

    /// \brief Global layer: the cache is [MAX_L * DK | MAX_L * DV], indexed by absolute
    ///        position, so the new rows are appended and the whole prefix is attended.
    void _attend_global(const gemma4_12b_prefill_mm_shape& shape, buffer<bf16>& cache) {
        cache.sync_from_device();
        bf16* k_cache = cache.data();
        //!< the halves are equal sized (DV == DK), so the V base is the midpoint of
        //!< whatever the layer currently owns -- never a MAX_L captured at build time,
        //!< which update_max_length() would leave stale
        bf16* v_cache = cache.data() + cache.size() / 2;
        memcpy(k_cache + (size_t)shape.L_begin * _DK, bufs.k.data(),
               (size_t)shape.L * _DK * sizeof(bf16));
        memcpy(v_cache + (size_t)shape.L_begin * _DV, bufs.v.data(),
               (size_t)shape.L * _DV * sizeof(bf16));

        cache.sync_to_device();

        //!< extract the block fix-up's inputs before the standard attention runs. It reads
        //!< the cache through the host mapping, which the sync above has just made
        //!< coherent, and the attention never writes k or v.
        //!<
        //!< The row count is what decides whether the fix-up goes to attn_global_image: the
        //!< overlay reads this very cache, so its capacity is the constraint. It is derived
        //!< from what the layer owns right now, like the V base above, and never from a
        //!< MAX_L captured at build time.
        const int cache_rows = (int)(cache.size() / 2 / (size_t)_DK);
        _stage_vision_blocks(shape, k_cache, v_cache, 0, shape.L_end, /*window*/ 0, cache_rows);

        if (!_attend_global_npu(shape, cache)) {
            gemma4_12b_cpu::attention(bufs.attn.data(), bufs.q.data(), k_cache, v_cache,
                                      shape.L, shape.L_end, num_attn_heads, num_kv_heads, _DH,
                                      shape.L_begin, /*kv_pos_begin*/ 0, /*window*/ 0,
                                      _DK, _DV);
            //!< the host wrote attn; publish it for o_proj, which reads it from the device
            gemma4_12b_scratch_push(bufs.attn);
        }

        //!< the standard attention has finished with the cache, so the image run is free to
        //!< read it -- the same buffer, unmodified since the sync above
        _patch_vision_blocks(shape, /*window*/ 0, &cache);
    }

    /// \brief Run the sliding window prefill attention on the array, if it is compiled in
    ///        and the unrolled window is tall enough.
    ///
    /// The overlay masks with the same counter it indexes kv rows by, so it cannot read the
    /// decoding ring: (position % sliding) is not monotonic in position. It reads the
    /// unrolled window instead, whose row 0 is absolute position `win_begin`, and every
    /// position it is told about is measured from there -- which is what `L_begin - win_begin`
    /// below is. Both mask terms are position *differences*, so the shift cancels.
    ///
    /// The run is only *issued* here. It writes bufs.attn, which nothing else in this
    /// layer reads until o_proj, so the caller is free to do host work before joining --
    /// which is what _attend_sliding() spends on the vision block recompute.
    ///
    /// \param win_begin absolute position of row 0 of the unrolled window
    /// \param win_len   rows of it holding real keys; the tail past this is read and masked
    /// \return an inactive handle when nothing was issued and the caller must fall back to
    ///         the host kernel
    /// \post   on an active handle, bufs.attn holds the standard (windowed, image-unaware)
    ///         attention output on the *device* once waited on; the host mapping is stale
    gemma4_12b_attn_run _attend_sliding_npu_start(const gemma4_12b_prefill_mm_shape& shape,
                                                  int win_begin, int win_len) {
#if GEMMA4_12B_ATTN_SWA_ON_NPU
        const int sliding = (int)SLIDING_LENGTH;
        //!< min(L_begin, sliding - 1) -- it stops growing once the window has filled, which
        //!< is what keeps the kv stream bounded no matter how long the prompt gets
        const int L_begin_rel = shape.L_begin - win_begin;
        const int    kv_rows  = bufs.swa_rows;
        const size_t kv_elems = bufs.swa_kv.size();

        //!< the last round streams keys to the end of the 128 row block holding its last
        //!< query row, at most 127 rows past it. bufs.reserve() sizes the window for exactly
        //!< this, so the guard is a belt on top of a brace rather than a live path.
        const int kv_tail = L_begin_rel + shape.L_attn + gemma4_12b_prefill_mm_shape::L_ATTN_ALIGN - 1;
        if (kv_tail > kv_rows) {
            DEBUG_BLOCK(1,
            header_print("info", "sliding attention: kv tail read to " + std::to_string(kv_tail)
                         + " exceeds the " + std::to_string(kv_rows)
                         + " row unrolled window, falling back to the host");
            )
            return gemma4_12b_attn_run{};
        }

        //!< the relative L_begin rides the control sequence, not a run argument. It is
        //!< constant across the kind's 24 layers, so only the first one regenerates.
        if (L_begin_rel != L_begin_seq || shape.L_attn != L_attn_seq ||
            kv_rows != kv_rows_seq || kv_elems != kv_elems_seq) {
            seq_gen->gen_attn_prefill_swa(this->attn.seq(), (uint32_t)L_begin_rel,
                                          (uint32_t)shape.L_attn, (uint32_t)kv_rows,
                                          (uint32_t)sliding, kv_elems);
            this->L_begin_seq  = L_begin_rel;
            this->L_attn_seq   = shape.L_attn;
            this->kv_rows_seq  = kv_rows;
            this->kv_elems_seq = kv_elems;
        }

        //!< q was pushed in forward(), the unrolled window by the caller; attn is written by
        //!< the array and read straight from the device by o_proj, so it needs no sync
        gemma4_12b_attn_run h;
        h.run    = this->attn.create_run(bufs.attn, bufs.q, bufs.swa_kv);
        h.active = true;
        h.run.start();
        (void)win_len;
        return h;
#else
        (void)shape; (void)win_begin; (void)win_len;
        return gemma4_12b_attn_run{};
#endif
    }

    /// \brief Score the sliding array run against the host kernel, after it has been
    ///        waited on and before the vision block fix-up overwrites any of it.
    /// \note  Compiled out entirely without ENABLE_DEBUG, and even then it only costs
    ///        anything on the layers xref is asked to score.
    void _attend_sliding_npu_xref(const gemma4_12b_prefill_mm_shape& shape,
                                  int win_begin, int win_len) {
#if GEMMA4_12B_ATTN_SWA_ON_NPU && defined(ENABLE_DEBUG)
        //!< the host kernel is the reference. Only pay for the pull and the replay on the
        //!< layers xref actually scores -- attn is L x DQ, which is megabytes per layer.
        if (xref != nullptr && xref->enabled(layer_idx)) {
            const int rows = xref->rows(shape.L);
            std::vector<bf16> want((size_t)rows * _DQ);
            gemma4_12b_cpu::attention(want.data(), bufs.q.data(),
                                      bufs.swa_k_ptr(), bufs.swa_v_ptr(),
                                      rows, win_len, num_attn_heads, num_kv_heads, _DH,
                                      shape.L_begin, win_begin, /*window*/ (int)SLIDING_LENGTH,
                                      _DK, _DV);
            gemma4_12b_scratch_pull(bufs.attn);
            xref->check("attention_out", layer_idx, bufs.attn.data(), want.data(),
                        _DQ, _DQ, rows);
        }
#else
        (void)shape; (void)win_begin; (void)win_len;
#endif
    }

    /// \brief Sliding layer: the cache is [SLIDING_LENGTH * DK | SLIDING_LENGTH * DV], a
    ///        ring indexed by (position % SLIDING_LENGTH).
    ///
    /// Unroll the part of the ring the prompt can still see into linear order, append
    /// the new rows, attend, and publish the new rows back into the ring for the
    /// decoding graph.
    ///
    /// \see _attend_sliding_npu_start() for the array path and why it reads the unrolled window
    ///      rather than the ring.
    ///
    void _attend_sliding(const gemma4_12b_prefill_mm_shape& shape, buffer<bf16>& cache) {
        cache.sync_from_device();
        const int sliding = (int)SLIDING_LENGTH;
        bf16* ring_k = cache.data();
        bf16* ring_v = cache.data() + cache.size() / 2;  //!< == sliding * _DK

        //!< the oldest position any query row of this chunk can still see. Row 0 of the
        //!< unrolled window, and the origin every relative position below is measured from.
        const int win_begin = std::max(0, shape.L_begin - sliding + 1);
        const int win_len   = shape.L_end - win_begin;

        bf16* swa_k = bufs.swa_k_ptr();
        bf16* swa_v = bufs.swa_v_ptr();

        for (int pos = win_begin; pos < shape.L_begin; pos++) {
            const int ring = pos % sliding;
            memcpy(swa_k + (size_t)(pos - win_begin) * _DK,
                   ring_k + (size_t)ring * _DK, (size_t)_DK * sizeof(bf16));
            memcpy(swa_v + (size_t)(pos - win_begin) * _DV,
                   ring_v + (size_t)ring * _DV, (size_t)_DV * sizeof(bf16));
        }
        for (int i = 0; i < shape.L; i++) {
            const int pos  = shape.L_begin + i;
            const int ring = pos % sliding;
            memcpy(swa_k + (size_t)(pos - win_begin) * _DK,
                   bufs.k.data() + (size_t)i * _DK, (size_t)_DK * sizeof(bf16));
            memcpy(swa_v + (size_t)(pos - win_begin) * _DV,
                   bufs.v.data() + (size_t)i * _DV, (size_t)_DV * sizeof(bf16));
            memcpy(ring_k + (size_t)ring * _DK,
                   bufs.k.data() + (size_t)i * _DK, (size_t)_DK * sizeof(bf16));
            memcpy(ring_v + (size_t)ring * _DV,
                   bufs.v.data() + (size_t)i * _DV, (size_t)_DV * sizeof(bf16));
        }

        //!< the array reads the rows past the last real key (masked off, but read): whatever an earlier call left there
        //!< must not reach the result, or a prefill that continues live and one that follows a restored state disagree
        if (bufs.swa_rows > win_len) {
            memset(swa_k + (size_t)win_len * _DK, 0, (size_t)(bufs.swa_rows - win_len) * _DK * sizeof(bf16));
            memset(swa_v + (size_t)win_len * _DV, 0, (size_t)(bufs.swa_rows - win_len) * _DV * sizeof(bf16));
        }

        cache.sync_to_device();

        //!< the unrolled window was just written by the host; publish it before anything
        //!< reads it from the device
        gemma4_12b_scratch_push(bufs.swa_kv);

        //!< extract the block fix-up's inputs before the standard attention touches them
        _stage_vision_blocks(shape, swa_k, swa_v, win_begin, win_len, /*window*/ sliding);

        //!< issue the array's standard attention and keep the host busy with the block
        //!< recompute while it runs: the two share no buffer, so the only ordering that
        //!< matters is the join before attn is touched
        gemma4_12b_attn_run attn_run = _attend_sliding_npu_start(shape, win_begin, win_len);

        if (attn_run.active) {
            _recompute_vision_blocks(/*window*/ sliding);  //!< no-op without vision blocks
            attn_run.wait();
            _attend_sliding_npu_xref(shape, win_begin, win_len);
            //!< the array is done writing attn; only now may the fix-up read and patch it
            if (!this->stages.empty()) _commit_vision_blocks(shape);
        }
        else {
            gemma4_12b_cpu::attention(bufs.attn.data(), bufs.q.data(), swa_k, swa_v,
                                      shape.L, win_len, num_attn_heads, num_kv_heads, _DH,
                                      shape.L_begin, win_begin, /*window*/ sliding,
                                      _DK, _DV);
            //!< the host wrote attn; publish it for o_proj, which reads it from the device
            gemma4_12b_scratch_push(bufs.attn);

            _patch_vision_blocks(shape, /*window*/ sliding);
        }
    }
};

/// \brief Fused up/gate projection, GeLU GLU and down projection of one layer kind.
struct gemma4_12b_mm_mlp_block {
    gemma4_12b_desc*              desc    = nullptr;
    gemma4_12b_npu_sequence*      seq_gen = nullptr;
    //!< x / upgate / hid: nothing here is layer kind specific, so both kinds run on the
    //!< same scratch and the largest tensors of the graph exist once
    gemma4_12b_mm_common_buffers* common  = nullptr;
    gemma4_12b_layer_type_t       layer_type = gemma4_12b_swa_layer;

    int _INTERMEDIATE_SIZE = 0;
    int L_seq = 0;  //!< rows the projection sequences are generated for

    //!< the fused up/gate region takes two apps, one per slab parity, so the gate run
    //!< can fuse the GeLU into its write back while the up run stays linear
    npu_app up_proj;
    npu_app gate_proj;
    npu_app down_proj;

#ifdef ENABLE_DEBUG
    //!< the cpu/npu cross reference, owned by gemma4_12b_prefill_mm_context. Never null in
    //!< a debug build; whether it scores anything is up to xref->enabled(layer_idx).
    gemma4_12b_prefill_xref* xref = nullptr;
    //!< index of the layer currently in forward(), which is what selects the scored layers.
    //!< Set per layer by gemma4_12b_prefill_mm_context::forward().
    int layer_idx = -1;
#endif

    void build(gemma4_12b_desc*              desc,
               gemma4_12b_npu_sequence*      seq_gen,
               gemma4_12b_mm_common_buffers* common,
               gemma4_12b_layer_type_t       layer_type,
               npu_app_manager*              mm_app_manager) {
        this->desc       = desc;
        this->seq_gen    = seq_gen;
        this->common     = common;
        this->layer_type = layer_type;
        this->_INTERMEDIATE_SIZE = (int)desc->intermediate_size;

#if GEMMA4_12B_MM_ON_NPU
        this->up_proj   = mm_app_manager->create_app();
        this->gate_proj = mm_app_manager->create_app();
        this->down_proj = mm_app_manager->create_app();
        // the control sequences are generated for a row count, which is only known once
        // a prompt arrives; see reserve()
#else
        (void)mm_app_manager;
#endif
    }

    /// \brief Make the block ready for a prompt of `L_padded` rows.
    /// \note The scratch the projections write lives in gemma4_12b_mm_common_buffers, so
    ///       the row count is all there is to do here.
    void reserve(int L_padded) {
        if (L_padded == L_seq) return;

#if GEMMA4_12B_MM_ON_NPU
        gemma4_12b_layer_weight_def& w = desc->weight_desc(layer_type);
        const int no_act = gemma4_12b_npu_sequence::DEQUANT_MM_ACT_NONE;

        // One weight region, two runs: up and gate are stored as alternating
        // UP_GATE_OUTDIM_REORDER row slabs, and the activation RTP reaches every core of
        // the array, so the only way to GeLU the gate slabs alone is to run them alone.
        // Both scatter into the same [L, 2 * IS] buffer, de-interleaved as [ up | gate ],
        // which leaves cpu_func::_glu_mul with nothing but the multiply.
        seq_gen->gen_dequant_mm_512(this->up_proj.seq(), L_padded, desc->D, _INTERMEDIATE_SIZE * 2,
                                    w.ffn_up.offset, 0, no_act,
                                    gemma4_12b_desc::UP_GATE_OUTDIM_REORDER,
                                    gemma4_12b_npu_sequence::UP_MATRIX);
        seq_gen->gen_dequant_mm_512(this->gate_proj.seq(), L_padded, desc->D, _INTERMEDIATE_SIZE * 2,
                                    w.ffn_up.offset, 0,
                                    gemma4_12b_npu_sequence::DEQUANT_MM_ACT_GELU,
                                    gemma4_12b_desc::UP_GATE_OUTDIM_REORDER,
                                    gemma4_12b_npu_sequence::GATE_MATRIX);
        seq_gen->gen_dequant_mm_512(this->down_proj.seq(), L_padded, _INTERMEDIATE_SIZE, desc->D,
                                    w.ffn_down.offset, 0, no_act);
#endif
        L_seq = L_padded;
    }

    /// \note On entry common->x holds pre_feedforward_layernorm(residual) and is already
    ///       on the device; on return it holds the mlp output, and is on the host.
    void forward(const gemma4_12b_prefill_mm_shape& shape, buffer<u8>& proj_weights) {
        PREFILL_PROF_INIT();
        gemma4_12b_layer_weight_def& w = desc->weight_desc(layer_type);

#if GEMMA4_12B_MM_ON_NPU
        up_proj(common->upgate, common->x, proj_weights);
        gate_proj(common->upgate, common->x, proj_weights);
        gemma4_12b_scratch_pull(common->upgate);
#else
        //!< one pass over the interleaved weight region, then the same de-interleave and
        //!< fused GeLU the two NPU runs leave behind, so everything downstream is shared
        gemma4_12b_cpu::dequant_mm_upgate(common->upgate.data(),
                                          proj_weights.data() + w.ffn_up.offset,
                                          common->x.data(), shape.L, (int)desc->D,
                                          _INTERMEDIATE_SIZE,
                                          gemma4_12b_desc::UP_GATE_OUTDIM_REORDER);
#endif
#if GEMMA4_12B_MM_ON_NPU && defined(ENABLE_DEBUG)
        // dequant_mm_upgate() produces exactly what the two runs above leave behind, so one
        // host call covers both. The halves live in one [L, 2 * IS] row and are scored
        // separately at that row stride, which keeps a fused activation bug (`gelu`)
        // distinguishable from a matmul bug (`up_proj`).
        if (xref->enabled(layer_idx)) {
            const int rows = xref->rows(shape.L);
            if (rows > 0) {
                std::vector<bf16> want((size_t)rows * _INTERMEDIATE_SIZE * 2);
                gemma4_12b_cpu::dequant_mm_upgate(want.data(),
                                                  proj_weights.data() + w.ffn_up.offset,
                                                  common->x.data(), rows, (int)desc->D,
                                                  _INTERMEDIATE_SIZE,
                                                  gemma4_12b_desc::UP_GATE_OUTDIM_REORDER);
                const int stride = _INTERMEDIATE_SIZE * 2;

                //!< check() indexes `got` with got_stride but `want` *packed*, so the two
                //!< halves of the [L, 2 * IS] reference row have to be compacted out into
                //!< [rows][IS] first. Handing it want.data() at stride reads every row
                //!< after the first from the wrong offset, which scores as a plausible
                //!< looking low cosine rather than as an obvious failure -- exactly the
                //!< reading that made up_proj look broken while down_proj passed.
                std::vector<bf16> want_up((size_t)rows * _INTERMEDIATE_SIZE);
                std::vector<bf16> want_gate((size_t)rows * _INTERMEDIATE_SIZE);
                for (int r = 0; r < rows; r++) {
                    const bf16* src = want.data() + (size_t)r * stride;
                    memcpy(want_up.data()   + (size_t)r * _INTERMEDIATE_SIZE, src,
                           (size_t)_INTERMEDIATE_SIZE * sizeof(bf16));
                    memcpy(want_gate.data() + (size_t)r * _INTERMEDIATE_SIZE,
                           src + _INTERMEDIATE_SIZE,
                           (size_t)_INTERMEDIATE_SIZE * sizeof(bf16));
                }

                xref->check("up_proj", layer_idx, common->upgate.data(),
                            want_up.data(), _INTERMEDIATE_SIZE, stride, rows);
                xref->check("gelu", layer_idx, common->upgate.data() + _INTERMEDIATE_SIZE,
                            want_gate.data(), _INTERMEDIATE_SIZE, stride, rows);
            }
        }
#endif
        PREFILL_PROF_TICK("mlp:upgate_proj");

        gemma4_12b_scratch_pull(common->hid);
        cpu_func::_glu_mul(common->hid.data(), common->upgate.data(), shape.L, _INTERMEDIATE_SIZE);
        gemma4_12b_scratch_push(common->hid);
        //!< no cross reference on the glu multiply: cpu_func::_glu_mul is the same call in
        //!< both builds, so there is no second implementation to compare it against
        PREFILL_PROF_TICK("mlp:glu_mul");

#if GEMMA4_12B_MM_ON_NPU
        down_proj(common->x, common->hid, proj_weights);
        gemma4_12b_scratch_pull(common->x);
#ifdef ENABLE_DEBUG
        if (xref->enabled(layer_idx)) {
            const int rows = xref->rows(shape.L);
            if (rows > 0) {
                std::vector<bf16> want((size_t)rows * desc->D);
                gemma4_12b_cpu::dequant_mm(want.data(), proj_weights.data() + w.ffn_down.offset,
                                           common->hid.data(), rows, _INTERMEDIATE_SIZE,
                                           (int)desc->D);
                xref->check("down_proj", layer_idx, common->x.data(), want.data(),
                            (int)desc->D, (int)desc->D, rows);
            }
        }
#endif
#else
        gemma4_12b_cpu::dequant_mm(common->x.data(), proj_weights.data() + w.ffn_down.offset,
                                   common->hid.data(), shape.L, _INTERMEDIATE_SIZE,
                                   (int)desc->D);
#endif
        PREFILL_PROF_TICK("mlp:down_proj");
    }
};

/// \brief The whole batched prefill path: one xclbin, one block pair per layer kind,
///        and the kind independent part of the activation scratch.
struct gemma4_12b_prefill_mm_context {
    gemma4_12b_desc*         desc    = nullptr;
    gemma4_12b_npu_sequence* seq_gen = nullptr;

    npu_app_manager* mm_app_manager = nullptr;
    //!< the two prefill attention overlays, one per layer kind. They are separate
    //!< bitstreams with separate RTP maps and floorplans, so neither app nor sequence is
    //!< shared; a null one keeps that kind's attention on the host.
    npu_app_manager* global_attn_app_manager = nullptr;
    npu_app_manager* swa_attn_app_manager    = nullptr;
    //!< the block masked global overlay, registered lazily by set_vision_blocks() on the
    //!< first prompt that carries an image. A text only session never loads it -- it is a
    //!< fourth hw context, and models shipped without the bitstream must still run -- and a
    //!< null one leaves the block fix-up on the host, which is always a correct answer.
    npu_app_manager* global_image_app_manager = nullptr;
    //!< whichever of the two is available, and what every activation bo is created from.
    //!< A bo belongs to the device rather than to a hw context, so one allocation can be
    //!< bound to runs on either bitstream; this only has to be non null.
    npu_app_manager* bo_manager = nullptr;

    //!< the D and intermediate_size wide tensors, which both kinds and both blocks run
    //!< on. The q/k/v/attn scratch is not here: it is kind specific and lives in each
    //!< gemma4_12b_mm_attn_block.
    gemma4_12b_mm_common_buffers common;

    // one instance per layer kind: the two kinds differ in head dim, kv head count,
    // rope frequencies and weight layout, so neither their sequences nor their
    // attention scratch can be shared.
    gemma4_12b_mm_attn_block swa_attn;
    gemma4_12b_mm_attn_block global_attn;
    gemma4_12b_mm_mlp_block  swa_mlp;
    gemma4_12b_mm_mlp_block  global_mlp;

    //!< the prompt's image/video blocks. Rebuilt once per prefill by set_vision_blocks();
    //!< both attn blocks hold a pointer to it, so a rebuild reaches all 48 layers.
    gemma4_12b_prefill_blocks vision_blocks;

    //!< what the lazy registration above needs, kept from the constructor's arguments
    npu_xclbin_manager* xclbin_mgr = nullptr;
    std::string         image_attn_xclbin;
    bool                image_attn_tried = false;  //!< registration is attempted once

#ifdef ENABLE_DEBUG
    //!< the cpu/npu cross reference. Owned here because both blocks score through it, and
    //!< because its layer selection is the same for the whole stack.
    gemma4_12b_prefill_xref xref;
#endif

    gemma4_12b_mm_attn_block& attn_block(gemma4_12b_layer_type_t type) {
        return is_global_layer(type) ? global_attn : swa_attn;
    }
    gemma4_12b_mm_mlp_block& mlp_block(gemma4_12b_layer_type_t type) {
        return is_global_layer(type) ? global_mlp : swa_mlp;
    }

    /// \note Registering dequant_mm.xclbin in the gemma4_12b_npu::Impl constructor would
    ///       make every decode only session pay for it, and would make models shipped
    ///       without it fail to load at all, so this is called from the first prefill.
    ///
    /// \note Nothing here depends on the prompt length: the scratch and the control
    ///       sequences are sized by reserve(), which the first prefill calls once it
    ///       knows how many rows it has.
    gemma4_12b_prefill_mm_context(npu_xclbin_manager*      npu,
                                  gemma4_12b_desc*         model_desc,
                                  gemma4_12b_npu_sequence* seq_gen,
                                  LM_Config&               config)
        : desc(model_desc), seq_gen(seq_gen)
#ifdef ENABLE_DEBUG
        //!< layer 0 is the first SWA layer, sliding_window_pattern - 1 the first global
        //!< one, and the last layer the one carrying whatever drift the other 47 added
        , xref((int)model_desc->sliding_window_pattern - 1,
               (int)model_desc->num_hidden_layers - 1)
#endif
    {
#if GEMMA4_12B_MM_ON_NPU
        this->mm_app_manager = npu->register_xclbin(
            utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant_mm.xclbin"));
#endif
#if GEMMA4_12B_ATTN_GLOBAL_ON_NPU
        this->global_attn_app_manager = npu->register_xclbin(
            utils::path_join(config.exec_path, "xclbins", config.model_name, "attn_global.xclbin"));
#endif
#if GEMMA4_12B_ATTN_SWA_ON_NPU
        this->swa_attn_app_manager = npu->register_xclbin(
            utils::path_join(config.exec_path, "xclbins", config.model_name, "attn_sliding.xclbin"));
#endif
#if GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU
        //!< resolved now, registered later: see set_vision_blocks()
        this->xclbin_mgr        = npu;
        this->image_attn_xclbin = utils::path_join(config.exec_path, "xclbins",
                                                   config.model_name, "attn_global_image.xclbin");
#endif
#if !GEMMA4_12B_MM_ON_NPU && !GEMMA4_12B_ATTN_ON_NPU
        //!< nothing runs on the array, so do not make a decode only session -- or a model
        //!< shipped without the xclbins -- pay for bitstreams it will never use
        (void)npu; (void)config;
#endif
        this->bo_manager = (mm_app_manager != nullptr)          ? mm_app_manager
                         : (global_attn_app_manager != nullptr) ? global_attn_app_manager
                                                                : swa_attn_app_manager;

        for (int t = 0; t < gemma4_12b_total_layer_types; t++) {
            const gemma4_12b_layer_type_t type = static_cast<gemma4_12b_layer_type_t>(t);
            //!< the overlay of this kind, or null to keep its attention on the host
            npu_app_manager* attn_mgr = is_global_layer(type) ? global_attn_app_manager
                                                              : swa_attn_app_manager;
            attn_block(type).build(desc, seq_gen, &common, type,
                                   mm_app_manager, attn_mgr, bo_manager);
            attn_block(type).blocks = &this->vision_blocks;
            mlp_block(type).build(desc, seq_gen, &common, type, mm_app_manager);
        }

        DEBUG_BLOCK(1,
        header_print("info", std::string("prefill path ready: matmuls on ")
                     + (GEMMA4_12B_MM_ON_NPU ? "npu" : "cpu")
                     + ", global attention on "
                     + (GEMMA4_12B_ATTN_GLOBAL_ON_NPU ? "npu" : "cpu")
                     + ", sliding attention on "
                     + (GEMMA4_12B_ATTN_SWA_ON_NPU ? "npu" : "cpu")
                     + ", global image blocks on "
                     + (GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU ? "npu (if the bitstream is there)"
                                                            : "cpu"));
        )
    }

    /// \brief Register attn_global_image.xclbin and hand it to the global attention block.
    ///
    /// Runs at most once per context, on the first prompt that actually carries a vision
    /// block. Registration is where a missing or unloadable bitstream shows up, and the
    /// answer to that is not to fail the prompt: the host kernel is still there and still
    /// correct, so this warns once and leaves image_app_manager null, which routes every
    /// block fix-up back to _recompute_vision_blocks().
    void _ensure_image_overlay() {
#if GEMMA4_12B_ATTN_GLOBAL_IMAGE_ON_NPU
        if (this->image_attn_tried || this->xclbin_mgr == nullptr) return;
        this->image_attn_tried = true;
        try {
            this->global_image_app_manager = this->xclbin_mgr->register_xclbin(this->image_attn_xclbin);
        } catch (const std::exception& e) {
            header_print("warn", std::string("prefill: attn_global_image.xclbin unavailable (")
                         + e.what() + "); image block attention stays on the cpu");
            this->global_image_app_manager = nullptr;
            return;
        }
        //!< global layers only: a sliding layer's block range is per query row, which this
        //!< overlay's single "keys stop at L" mask cannot express
        global_attn.bind_image_overlay(this->global_image_app_manager);
#endif
    }

    /// \brief Find this prompt's image/video blocks, which is what makes the attention
    ///        bidirectional inside each image. Call once per prefill, before the layer
    ///        loop; a text only prompt clears the list and every layer runs plain causal.
    /// \see   gemma4_12b_mm_attn_block::_attend_vision_blocks()
    void set_vision_blocks(const int* tokens, int L, int L_begin,
                           int image_soft_id, int video_soft_id) {
        this->vision_blocks.build(tokens, L, L_begin, image_soft_id, video_soft_id);
        //!< only now is it known that this session will ever need the image overlay
        if (!this->vision_blocks.empty()) _ensure_image_overlay();
    }

    /// \brief Size the activation scratch and the control sequences for `shape`.
    ///
    /// The whole prompt goes through the stack in one pass, so this is what the row
    /// count of every NPU run comes from. Both layer kinds are prepared, because a
    /// prompt of any length reaches both. Called once per prefill; a second prompt of
    /// the same padded length costs nothing.
    void reserve(const gemma4_12b_prefill_mm_shape& shape) {
        bool reallocated = this->common.reserve(mm_app_manager, desc, shape.L_padded);

        for (int t = 0; t < gemma4_12b_total_layer_types; t++) {
            const gemma4_12b_layer_type_t type = static_cast<gemma4_12b_layer_type_t>(t);
            reallocated |= attn_block(type).reserve(shape.L_padded);
            mlp_block(type).reserve(shape.L_padded);
        }

        DEBUG_BLOCK(1,
        header_print("info", "prefill graph sized for " + std::to_string(shape.L_padded)
                     + " rows (" + std::to_string(shape.L) + " tokens)"
                     + (reallocated ? ", scratch reallocated" : ", scratch reused"));
        )
    }

    /// \param layer_idx    index into the layer stack, picks the layer kind
    /// \param residual     L_padded x D hidden states, updated in place
    /// \param shape        prompt geometry, see gemma4_12b_prefill_mm_shape::update()
    /// \pre   reserve(shape) has been called for this shape
    /// \param proj_weights packed projections of this layer
    /// \param rms_weights  the four layer norms of this layer, laid out in the order
    ///                     given by gemma4_12b_layer_weight_def
    /// \param rope_weights q_norm / k_norm (and the decode cos/sin table)
    /// \param kv_cache     kv cache of this layer as the decode path sees it
    /// \param layer_scale  layer_scalar, the per layer output scale returned by
    ///                     gemma4_12b_desc::load_layer_weights()
    /// \return `residual`, so that the caller can chain layers
    buffer<bf16>& forward(int                                layer_idx,
                          buffer<bf16>&                      residual,
                          const gemma4_12b_prefill_mm_shape& shape,
                          buffer<u8>&                        proj_weights,
                          buffer<bf16>&                      rms_weights,
                          buffer<bf16>&                      rope_weights,
                          buffer<bf16>&                      kv_cache,
                          bf16                               layer_scale) {
        PREFILL_PROF_INIT();
        const gemma4_12b_layer_type_t type = desc->layer_type(layer_idx);
        gemma4_12b_layer_weight_def&  w    = desc->weight_desc(type);
        const int D = (int)desc->D;

        const bf16* input_norm_w     = rms_weights.data() + w.input_layer_norm.offset      / sizeof(bf16);
        const bf16* post_attn_norm_w = rms_weights.data() + w.post_attention_norm.offset   / sizeof(bf16);
        const bf16* pre_ffn_norm_w   = rms_weights.data() + w.pre_feedforward_norm.offset  / sizeof(bf16);
        const bf16* post_ffn_norm_w  = rms_weights.data() + w.post_feedforward_norm.offset / sizeof(bf16);

#ifdef ENABLE_DEBUG
        //!< the blocks decide per matmul whether to score; all they need from here is which
        //!< layer they are running as
        attn_block(type).xref      = &xref;
        attn_block(type).layer_idx = layer_idx;
        mlp_block(type).xref       = &xref;
        mlp_block(type).layer_idx  = layer_idx;
#endif

        // ---- input_layernorm -> self attention -> post_attention_layernorm ----
        cpu_func::_rms_norm_batch(common.x.data(), residual.data(), input_norm_w,
                                  D, D, shape.L, 0, 0);
        gemma4_12b_scratch_push(common.x);
        PREFILL_PROF_TICK("layer:input_layernorm");

        attn_block(type).forward(shape, proj_weights, rope_weights, kv_cache);
        PREFILL_PROF_TICK("layer:attn_block_total");

        gemma4_12b_scratch_pull(common.x);
        cpu_func::_rms_norm_batch(common.x.data(), common.x.data(), post_attn_norm_w,
                                  D, D, shape.L, 0, 0);
        //!< an in place rewrite of what o_proj / down_proj left in the bo. The next full
        //!< overwrite of x would flush these lines eventually, but "eventually" is exactly
        //!< the window in which they can evict on top of an array write.
        gemma4_12b_scratch_push(common.x);
        cpu_func::_residual_add_batch(residual.data(), common.x.data(), residual.data(),
                                      D, shape.L, 0, 0, 0);

        // ---- pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm ----
        cpu_func::_rms_norm_batch(common.x.data(), residual.data(), pre_ffn_norm_w,
                                  D, D, shape.L, 0, 0);
        gemma4_12b_scratch_push(common.x);
        PREFILL_PROF_TICK("layer:pre_ffn_layernorm");

        mlp_block(type).forward(shape, proj_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        gemma4_12b_scratch_pull(common.x);
        cpu_func::_rms_norm_batch(common.x.data(), common.x.data(), post_ffn_norm_w,
                                  D, D, shape.L, 0, 0);
        //!< an in place rewrite of what o_proj / down_proj left in the bo. The next full
        //!< overwrite of x would flush these lines eventually, but "eventually" is exactly
        //!< the window in which they can evict on top of an array write.
        gemma4_12b_scratch_push(common.x);
        cpu_func::_residual_add_batch(residual.data(), common.x.data(), residual.data(),
                                      D, shape.L, 0, 0, 0);

        // ---- per layer output scale ------------------------------------------
        cpu_func::_elementwise_scale_batch(residual.data(), (float)layer_scale,
                                           D, shape.L, 0);
        PREFILL_PROF_TICK("layer:output_scale");

        return residual;
    }
};

#endif
