// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file minicpm_v_prefill_cpu.hpp
/// \brief Backend switches, plus the host implementation of every array stage of the
///        MiniCPM-V-4.7-1B text tower.
///
/// This is the "complete host model" of ADDING_A_NEW_MODEL.md S5 step 1. Every stage the
/// array can run has a host kernel here that reads the *same* device-layout bytes the array
/// reads (reorder_cpy'd q4_k / q8_0 blocks, the conv/recurrent state in the kv-cache slot,
/// the [K | V] attention cache) and leaves behind *exactly* what the array leaves behind, so
/// any single stage can move between host and array without touching its neighbours.
///
/// It lives apart from minicpm_v_prefill.hpp only to keep the CPU section (S3) and the NPU
/// prefill section (S5) off each other's files; it is the one deviation from the qwen3_5vl
/// layout.
#ifndef __MINICPM_V_PREFILL_CPU_HPP__
#define __MINICPM_V_PREFILL_CPU_HPP__

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "typedef.hpp"
#include "buffer.hpp"
#include "weight_desc.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "image/npu/vision_bo.hpp"

// ---------------------------------------------------------------------------
// Backend selection.
//
// One macro per overlay, one owning section per macro. All default to the array; the
// make flags are opt-OUT and move one stage back to the host:
//
//   make PREFILL_MM=cpu    -> -DFLM_PREFILL_MM_ON_CPU   -> MINICPM_V_MM_ON_NPU     = 0  (S5, mm.xclbin)
//   make PREFILL_ATTN=cpu  -> -DFLM_PREFILL_ATTN_ON_CPU -> MINICPM_V_ATTN_ON_NPU   = 0  (S5, attn.xclbin)
//   make PREFILL_GDN=cpu   -> -DFLM_PREFILL_GDN_ON_CPU  -> MINICPM_V_GDN_ON_NPU    = 0  (S5, GateDeltaNet_prefill.xclbin)
//   make PREFILL_CONV=cpu  -> -DFLM_PREFILL_CONV_ON_CPU -> MINICPM_V_CONV_ON_NPU   = 0  (S5, conv.xclbin)
//   make DECODE=cpu        -> -DFLM_DECODE_ON_CPU       -> MINICPM_V_DECODE_ON_NPU = 0  (S4, layer.xclbin + lm_head.xclbin)
//
// Every array path is host-fallback-clean: its _*_npu() helper returns false when its
// macro is 0 (or when it declines at run time) and the caller runs the host kernel, so
// setting any single macro to 0 by hand is a valid configuration needing no other change.
// Flip them one at a time and the layer where the answer moves names the stage that broke.
//
// DECODE covers the lm_head too: both bitstreams are registered by the decode path, and
// with DECODE=cpu a token is the prefill graph run at L = 1 followed by a host lm_head.
// ---------------------------------------------------------------------------
#ifndef MINICPM_V_MM_ON_NPU
#ifdef FLM_PREFILL_MM_ON_CPU
#define MINICPM_V_MM_ON_NPU 0
#else
#define MINICPM_V_MM_ON_NPU 1
#endif
#endif

#ifndef MINICPM_V_ATTN_ON_NPU
#ifdef FLM_PREFILL_ATTN_ON_CPU
#define MINICPM_V_ATTN_ON_NPU 0
#else
#define MINICPM_V_ATTN_ON_NPU 1
#endif
#endif

/// The array attention reads q, K and V^T as bfp16 tiles made on the host (attn_bfp16.xclbin, the 2B kernel
/// FLM_Xclbin/Qwen3_5/gated_attention_prefill: 8 q / 2 kv heads of 256, same shape as this model's attention
/// layers). On HRX it relies on the exact-range syncs (mcv_npu::sync_*_range -> sync_range_*_exact), not whole buffers.
#ifndef MINICPM_V_ATTN_BFP16
#if MINICPM_V_ATTN_ON_NPU
#define MINICPM_V_ATTN_BFP16 1
#else
#define MINICPM_V_ATTN_BFP16 0
#endif
#endif

#ifndef MINICPM_V_GDN_ON_NPU
#ifdef FLM_PREFILL_GDN_ON_CPU
#define MINICPM_V_GDN_ON_NPU 0
#else
#define MINICPM_V_GDN_ON_NPU 1
#endif
#endif

#ifndef MINICPM_V_CONV_ON_NPU
#ifdef FLM_PREFILL_CONV_ON_CPU
#define MINICPM_V_CONV_ON_NPU 0
#else
#define MINICPM_V_CONV_ON_NPU 1
#endif
#endif

#ifndef MINICPM_V_DECODE_ON_NPU
#ifdef FLM_DECODE_ON_CPU
#define MINICPM_V_DECODE_ON_NPU 0
#else
#define MINICPM_V_DECODE_ON_NPU 1
#endif
#endif

/// Per-stage timing inside the array decode loop (`make DECODE_PROFILE=1`), printed to
/// stderr every 32 tokens. Off by default and compiled out entirely; it exists because
/// "decode is 3% short of the bar" is unanswerable without knowing how the ~22 ms splits
/// between the 24 layer runs, the lm_head run, and the host work between them.
#ifndef MINICPM_V_DECODE_PROFILE
#ifdef FLM_DECODE_PROFILE
#define MINICPM_V_DECODE_PROFILE 1
#else
#define MINICPM_V_DECODE_PROFILE 0
#endif
#endif

/// Whether the prefill scratch must be device BOs. When 0 it is plain host memory and
/// sync_to/from_device() must never be called on it (buffer<T>::sync_* asserts on a
/// non-BO), which is what scratch_push()/scratch_pull() below are for.
#define MINICPM_V_SCRATCH_ON_DEVICE \
    (MINICPM_V_MM_ON_NPU || MINICPM_V_ATTN_ON_NPU || MINICPM_V_GDN_ON_NPU || MINICPM_V_CONV_ON_NPU)

namespace minicpm_v_cpu {

    //! 16-column value slabs one gated_delta_net work unit carries (independent FMA chains)
#ifndef GDN_SLABS
#define GDN_SLABS 4
#endif

    // ---------------------------------------------------------------------------------
    // Scratch discipline (ADDING_A_NEW_MODEL.md S2.5).
    //
    // The invariant every stage keeps: when a stage returns, its output is valid on both
    // sides. An array stage pulls what it wrote; a host stage pushes what it wrote. Then
    // any mix of backends is coherent without the caller knowing which ran.
    // ---------------------------------------------------------------------------------

    /// \brief allocate prefill scratch: a device BO if any prefill stage is on the array,
    ///        host memory otherwise.
    /// \note  Allocated through the xclbin manager rather than one app manager, so that
    ///        e.g. PREFILL_MM=cpu PREFILL_ATTN=npu still gets real BOs. A BO belongs to the
    ///        device, not to a hw_context, so one allocation serves runs on any bitstream.
    template <typename T>
    inline buffer<T> scratch_alloc(npu_xclbin_manager* npu, size_t n) {
#if MINICPM_V_SCRATCH_ON_DEVICE
        return npu->create_bo_buffer<T>(n);
#else
        (void)npu;
        return buffer<T>(n);
#endif
    }

    /// \brief flush a host write to scratch; a no-op when scratch is host memory
    template <typename T>
    inline void scratch_push(buffer<T>& b) {
#if MINICPM_V_SCRATCH_ON_DEVICE
        b.sync_to_device();
#else
        (void)b;
#endif
    }

    /// \brief invalidate before the host reads what an array run wrote; a no-op when
    ///        scratch is host memory
    template <typename T>
    inline void scratch_pull(buffer<T>& b) {
#if MINICPM_V_SCRATCH_ON_DEVICE
        b.sync_from_device();
#else
        (void)b;
#endif
    }

    /// \brief scratch_push / scratch_pull over rows [row0, row0 + rows) of a row-major buffer with
    ///        `ld` elements per row. The prefill scratch is sized for L_padded (a multiple of 512)
    ///        rows but only the first L rows carry data, and the rows past L are never read by the
    ///        host (every row is independent in the projections, and the attention and the cache
    ///        writes only use the first L), so a stage syncs what it produced, not the padding.
    ///        HRX has no ranged sync and keeps the whole-buffer one.
    template <typename T>
    inline void scratch_push_rows(buffer<T>& b, size_t row0, size_t rows, size_t ld) {
#if MINICPM_V_SCRATCH_ON_DEVICE
        mcv_npu::sync_to_device_range(b, row0 * ld, rows * ld);
#else
        (void)b; (void)row0; (void)rows; (void)ld;
#endif
    }
    template <typename T>
    inline void scratch_pull_rows(buffer<T>& b, size_t row0, size_t rows, size_t ld) {
#if MINICPM_V_SCRATCH_ON_DEVICE
        mcv_npu::sync_from_device_range(b, row0 * ld, rows * ld);
#else
        (void)b; (void)row0; (void)rows; (void)ld;
#endif
    }

    /// \brief Thread count for the host kernels of the prefill: cpu_functions.hpp's host_team_threads(),
    ///        so every region of a prefill runs on the same OpenMP team (see there).
    /// \note The cap is measured, not a guess: on the 16-core / 32-thread Ryzen AI MAX+ 395 under a
    ///       load of ~10, a 1 k prompt took 3.1 s at 32 threads against 1.2 s at 16, and an L = 1
    ///       decode 1.7 s/token against 56 ms. Override with FLM_PREFILL_THREADS (8 was fastest for
    ///       decode on that loaded box).
    inline int prefill_threads() { return host_team_threads(); }

    // ---------------------------------------------------------------------------------
    // The on-device block format. This is the executable specification of it.
    //
    // A quantized weight of N output rows x K input columns is a grid of 32 x 256 chunks
    // (row block rb = row / 32, column block cb = col / 256), each a fixed-size block:
    //
    //   flm_q4k, 4736 B   u8   s[8][32]        per-group scale code   (g = col%256 / 32)
    //                     u8   m[8][32]        per-group min code
    //                     u8   q[256][16]      column-major nibbles: byte (c, j) holds rows
    //                                          2j (low nibble) and 2j+1 (high), unsigned
    //                     bf16 S[32]           per-row super scale
    //                     bf16 negM[32]        per-row super min, already negated
    //       w[r][c] = S[r] * s[g][r] * q + negM[r] * m[g][r]
    //
    //   flm_q80, 8704 B   bf16 s[8][32]
    //                     i8   q[256][32]      column-major
    //       w[r][c] = s[g][r] * q
    //
    // (Same decode as FLM_Q4NX_Converter/verify_minicpm_v_q4nx.py::dequant_q4nx, which
    // round-trips every tensor of this checkpoint against gguf.dequantize().)
    //
    // minicpm_v_desc::reorder_cpy then interleaves the blocks of `vblocks` consecutive row
    // blocks per column block; reorder_cpy_lm_head additionally splits each band into
    // `streams` contiguous halves, one per shim MM2S channel. block_offset() inverts both.
    // ---------------------------------------------------------------------------------

    static constexpr int QB_ROWS   = QXNX_ROW_BLOCK_SIZE;  //!< 32 output rows per block
    static constexpr int QB_COLS   = QXNX_COL_BLOCK_SIZE;  //!< 256 input columns per block
    static constexpr int QB_GROUP  = 32;                   //!< columns sharing one scale
    static constexpr int QB_GROUPS = QB_COLS / QB_GROUP;   //!< 8

    /// \brief how reorder_cpy laid a weight's blocks out in the device buffer
    struct weight_layout {
        flm_dtype_t dtype;
        int vblocks;   //!< row blocks interleaved per band
        int streams;   //!< contiguous halves per band (1 = plain interleave)
    };

    /// every projection in proj_weights: minicpm_v_desc::reorder_cpy(..., vertical_blocks = 2)
    static constexpr weight_layout BODY_LAYOUT    = {flm_q4k, 2, 1};
    /// lm_head_weights: minicpm_v_desc::reorder_cpy_lm_head(..., flm_q80, 4)
    static constexpr weight_layout LM_HEAD_LAYOUT = {flm_q80, 4, 2};

    inline size_t block_bytes(flm_dtype_t dtype) {
        return get_quantization_byte_size((size_t)QB_ROWS * QB_COLS, dtype);
    }

    /// \brief Byte offset of the block covering rows [rb*32, rb*32+32) of column block cb.
    /// \param blocks_per_row number of column blocks, K / 256
    inline size_t block_offset(int rb, int cb, int blocks_per_row, const weight_layout& lay) {
        const int rps = lay.vblocks / lay.streams;   // row blocks per stream
        const size_t idx = (size_t)(rb / lay.vblocks) * lay.vblocks * blocks_per_row
                         + (size_t)((rb % lay.vblocks) / rps) * rps * blocks_per_row
                         + (size_t)cb * rps
                         + (size_t)(rb % rps);
        return idx * block_bytes(lay.dtype);
    }

    /// \brief Dequantize the 32-row band rb of an N x K weight to fp32.
    /// \param dst [K][32] floats: the 32 row values of input column k are contiguous. This is
    ///            the block's own column-major order, and the order dequant_mm() wants (it
    ///            vectorizes across the 32 rows).
    /// \param w   base of the reordered region (proj_weights.data() + desc.offset)
    inline void dequant_row_block(float* dst, const u8* w, int rb, int K, const weight_layout& lay) {
        const int blocks_per_row = K / QB_COLS;
        if (lay.dtype == flm_q4k) {
            // rows 0..15 come from bytes 0..7 and rows 16..31 from bytes 8..15, low nibble
            // first, so interleave the low/high nibble vectors pairwise
            const __m512i idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            const __m512i idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            const __m512i nib    = _mm512_set1_epi32(0x0F);
            for (int cb = 0; cb < blocks_per_row; cb++) {
                const u8*   blk  = w + block_offset(rb, cb, blocks_per_row, lay);
                const u8*   s8   = blk;
                const u8*   m8   = blk + QB_GROUPS * QB_ROWS;
                const u8*   qs   = blk + 2 * QB_GROUPS * QB_ROWS;
                const bf16* S    = reinterpret_cast<const bf16*>(qs + QB_COLS * QB_ROWS / 2);
                const bf16* negM = S + QB_ROWS;
                const __m512 S0 = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)S));
                const __m512 S1 = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(S + 16)));
                const __m512 M0 = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)negM));
                const __m512 M1 = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(negM + 16)));
                for (int g = 0; g < QB_GROUPS; g++) {
                    const u8* sg = s8 + g * QB_ROWS;
                    const u8* mg = m8 + g * QB_ROWS;
                    const __m512 sc0 = _mm512_mul_ps(S0, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)sg))));
                    const __m512 sc1 = _mm512_mul_ps(S1, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sg + 16)))));
                    const __m512 mn0 = _mm512_mul_ps(M0, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)mg))));
                    const __m512 mn1 = _mm512_mul_ps(M1, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(mg + 16)))));
                    for (int c = 0; c < QB_GROUP; c++) {
                        const int col = cb * QB_COLS + g * QB_GROUP + c;
                        const __m512i b  = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(qs + (size_t)(g * QB_GROUP + c) * (QB_ROWS / 2))));
                        const __m512  lo = _mm512_cvtepi32_ps(_mm512_and_si512(b, nib));
                        const __m512  hi = _mm512_cvtepi32_ps(_mm512_srli_epi32(b, 4));
                        const __m512  q0 = _mm512_permutex2var_ps(lo, idx_lo, hi);
                        const __m512  q1 = _mm512_permutex2var_ps(lo, idx_hi, hi);
                        _mm512_storeu_ps(dst + (size_t)col * QB_ROWS,      _mm512_fmadd_ps(q0, sc0, mn0));
                        _mm512_storeu_ps(dst + (size_t)col * QB_ROWS + 16, _mm512_fmadd_ps(q1, sc1, mn1));
                    }
                }
            }
        }
        else if (lay.dtype == flm_q80) {
            for (int cb = 0; cb < blocks_per_row; cb++) {
                const u8*   blk = w + block_offset(rb, cb, blocks_per_row, lay);
                const bf16* sc  = reinterpret_cast<const bf16*>(blk);
                const i8*   qs  = reinterpret_cast<const i8*>(blk + QB_GROUPS * QB_ROWS * sizeof(bf16));
                for (int g = 0; g < QB_GROUPS; g++) {
                    const __m512 sc0 = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(sc + g * QB_ROWS)));
                    const __m512 sc1 = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(sc + g * QB_ROWS + 16)));
                    for (int c = 0; c < QB_GROUP; c++) {
                        const int col = cb * QB_COLS + g * QB_GROUP + c;
                        const i8* qc  = qs + (size_t)(g * QB_GROUP + c) * QB_ROWS;
                        const __m512 q0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)qc)));
                        const __m512 q1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(qc + 16))));
                        _mm512_storeu_ps(dst + (size_t)col * QB_ROWS,      _mm512_mul_ps(q0, sc0));
                        _mm512_storeu_ps(dst + (size_t)col * QB_ROWS + 16, _mm512_mul_ps(q1, sc1));
                    }
                }
            }
        }
        else {
            // q4_0 / q4_1 are not used by this checkpoint and have no decode here yet
            assert(false && "minicpm_v_cpu::dequant_row_block: unsupported dtype");
        }
    }

    /// \brief y[L][N] = x[L][K] * W^T, W an N x K weight in the device layout `lay`.
    /// \param y   destination, row l at y + l * ldy, bf16 (what the array writes)
    /// \param w   base of the reordered region
    /// \param x   activations, row l at x + l * ldx, bf16
    /// \param N   output rows actually stored; the band covering a ragged tail is still
    ///            dequantized whole (the lm_head's 248094 rows sit in 7753 bands)
    /// \note The host equivalent of one gen_dequant_mm_512() run. Each band is dequantized
    ///       once and reused across all L rows, so weight traffic is the weight size, not
    ///       L times it. fp32 accumulation, rounded to bf16 once at the end.
    inline void dequant_mm(bf16* y, int ldy, const u8* w, const bf16* x, int ldx,
                           int L, int K, int N, const weight_layout& lay) {
        assert(K % QB_COLS == 0 && "q4nx needs a 256 element input dimension");
        if (L <= 0 || N <= 0) return;
        const int row_blocks = (N + QB_ROWS - 1) / QB_ROWS;

        // activations to fp32 once, shared by every band
        std::vector<float> xf((size_t)L * K);
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* xr = x + (size_t)l * ldx;
            float*      xo = xf.data() + (size_t)l * K;
            for (int k = 0; k < K; k += 16) {
                _mm512_storeu_ps(xo + k, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(xr + k))));
            }
        }

        #pragma omp parallel num_threads(prefill_threads())
        {
            std::vector<float> tile((size_t)K * QB_ROWS);
            #pragma omp for schedule(dynamic, 4)
            for (int rb = 0; rb < row_blocks; rb++) {
                dequant_row_block(tile.data(), w, rb, K, lay);
                const int  n0     = rb * QB_ROWS;
                const bool ragged = n0 + QB_ROWS > N;
                for (int l0 = 0; l0 < L; l0 += 4) {
                    const int nl = (L - l0) < 4 ? (L - l0) : 4;
                    __m512 acc[4][2];
                    for (int i = 0; i < 4; i++) { acc[i][0] = _mm512_setzero_ps(); acc[i][1] = _mm512_setzero_ps(); }
                    const float* x0 = xf.data() + (size_t)l0 * K;
                    for (int k = 0; k < K; k++) {
                        const __m512 t0 = _mm512_loadu_ps(tile.data() + (size_t)k * QB_ROWS);
                        const __m512 t1 = _mm512_loadu_ps(tile.data() + (size_t)k * QB_ROWS + 16);
                        for (int i = 0; i < nl; i++) {
                            const __m512 xv = _mm512_set1_ps(x0[(size_t)i * K + k]);
                            acc[i][0] = _mm512_fmadd_ps(xv, t0, acc[i][0]);
                            acc[i][1] = _mm512_fmadd_ps(xv, t1, acc[i][1]);
                        }
                    }
                    for (int i = 0; i < nl; i++) {
                        bf16* yr = y + (size_t)(l0 + i) * ldy + n0;
                        if (!ragged) {
                            _mm256_storeu_si256((__m256i*)yr,        cpu_func::f32_bf16_rne_512(acc[i][0]));
                            _mm256_storeu_si256((__m256i*)(yr + 16), cpu_func::f32_bf16_rne_512(acc[i][1]));
                        }
                        else {
                            alignas(64) float tmp[QB_ROWS];
                            _mm512_store_ps(tmp, acc[i][0]);
                            _mm512_store_ps(tmp + 16, acc[i][1]);
                            for (int r = 0; r < N - n0; r++) yr[r] = static_cast<bf16>(tmp[r]);
                        }
                    }
                }
            }
        }
    }

    /// \brief silu in place, fp32 math, bf16 storage
    inline void silu_inplace(bf16* p, size_t n) {
        const __m512 one = _mm512_set1_ps(1.0f);
        const long long nn = (long long)n;
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (long long i = 0; i < nn - nn % 16; i += 16) {
            const __m512 v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(p + i)));
            const __m512 d = _mm512_add_ps(one, _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), v)));
            _mm256_storeu_si256((__m256i*)(p + i), cpu_func::f32_bf16_rne_512(_mm512_div_ps(v, d)));
        }
        for (long long i = nn - nn % 16; i < nn; i++) {
            const float v = static_cast<float>(p[i]);
            p[i] = static_cast<bf16>(v / (1.0f + expf(-v)));
        }
    }

    /// \brief The host equivalent of the fused up/gate projection.
    ///
    /// load_*_weights() store up and gate interleaved in slabs of `slab` output rows
    /// (up[0:512], gate[0:512], up[512:1024], ...), so one matmul over the fused region
    /// yields the interleaved (L, IS / slab, 2, slab) layout. This undoes the interleave and
    /// applies the activation, leaving the de-interleaved [ up | silu(gate) ] row: the
    /// layout the reference dumps (mlp_up, mlp_gate) use, and what glu_mul() consumes.
    ///
    /// \param y    [L, 2 * IS] bf16, [ up | silu(gate) ] per row
    /// \param slab UP_GATE_OUTDIM_REORDEER in minicpm_v_desc::load_*_weights (512)
    inline void dequant_mm_upgate(bf16* y, const u8* w, const bf16* x, int L, int K, int IS, int slab) {
        if (L <= 0) return;
        std::vector<bf16> fused((size_t)L * IS * 2);
        dequant_mm(fused.data(), IS * 2, w, x, K, L, K, IS * 2, BODY_LAYOUT);

        const int slices = IS / slab;
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* src  = fused.data() + (size_t)l * IS * 2;
            bf16*       up   = y + (size_t)l * IS * 2;
            bf16*       gate = up + IS;
            for (int s = 0; s < slices; s++) {
                const bf16* p = src + (size_t)s * 2 * slab;
                memcpy(up + (size_t)s * slab, p, (size_t)slab * sizeof(bf16));
                for (int i = 0; i < slab; i++) {
                    const float g = static_cast<float>(p[slab + i]);
                    gate[(size_t)s * slab + i] = static_cast<bf16>(g / (1.0f + expf(-g)));
                }
            }
        }
    }

    /// \brief hid[L][IS] = up * gate, from dequant_mm_upgate()'s [ up | gate ] rows
    inline void glu_mul(bf16* hid, const bf16* upgate, int L, int IS) {
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            const bf16* up   = upgate + (size_t)l * IS * 2;
            const bf16* gate = up + IS;
            bf16*       h    = hid + (size_t)l * IS;
            int i = 0;
            for (; i + 16 <= IS; i += 16) {
                const __m512 u = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(up + i)));
                const __m512 g = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(gate + i)));
                _mm256_storeu_si256((__m256i*)(h + i), cpu_func::f32_bf16_rne_512(_mm512_mul_ps(u, g)));
            }
            for (; i < IS; i++) h[i] = static_cast<bf16>(static_cast<float>(up[i]) * static_cast<float>(gate[i]));
        }
    }

    /// \brief fp32 dot product of two bf16 vectors
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

    /// \brief acc[0..n) += weight * v[0..n), bf16 source into an fp32 accumulator
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

    /// \brief Causal grouped-query attention over a contiguous key/value range.
    ///
    /// \param out          [L_q, num_heads * DH] bf16, head h at column h * DH
    /// \param q            [L_q, num_heads * DH] bf16, already q-normed and roped
    /// \param k            [L_kv, k_stride] bf16, kv head hk at column hk * DH
    /// \param v            [L_kv, v_stride] bf16, same head layout as k
    /// \param q_pos_begin  absolute (cache) position of q row 0
    /// \param kv_pos_begin absolute (cache) position of k/v row 0
    /// \param scale        softmax scale; 1/sqrt(DH) for this model (q_norm does not fold it)
    /// \note q head h reads kv head h / (num_heads / num_kv_heads), which is the reference's
    ///       "B (h R) L d -> B h L R d" grouping.
    inline void attention(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                          int L_q, int L_kv, int num_heads, int num_kv_heads, int DH,
                          int q_pos_begin, int kv_pos_begin,
                          int k_stride, int v_stride, float scale) {
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
                    bf16* o  = out + (size_t)l * q_stride + (size_t)h * DH;
                    int   j1 = q_pos_begin + l - kv_pos_begin;
                    if (j1 > L_kv - 1) j1 = L_kv - 1;
                    if (j1 < 0) {   // nothing visible; cannot happen for a causal prompt
                        memset(o, 0, (size_t)DH * sizeof(bf16));
                        continue;
                    }

                    const int   hk = h / groups;
                    const bf16* qh = q + (size_t)l * q_stride + (size_t)h * DH;

                    float m = -INFINITY;
                    for (int j = 0; j <= j1; j++) {
                        const float s = dot_bf16(qh, k + (size_t)j * k_stride + (size_t)hk * DH, DH) * scale;
                        scores[j] = s;
                        if (s > m) m = s;
                    }
                    float sum = 0.0f;
                    for (int j = 0; j <= j1; j++) {
                        const float e = expf(scores[j] - m);
                        scores[j] = e;
                        sum += e;
                    }
                    const float inv = 1.0f / sum;

                    std::fill(acc.begin(), acc.end(), 0.0f);
                    for (int j = 0; j <= j1; j++) {
                        axpy_bf16(acc.data(), v + (size_t)j * v_stride + (size_t)hk * DH, scores[j] * inv, DH);
                    }
                    for (int d = 0; d < DH; d++) o[d] = static_cast<bf16>(acc[d]);
                }
            }
        }
    }

    /// \brief Causal depthwise conv1d + SiLU, the first half of the conv overlay.
    /// \param y [L, C] bf16
    /// \param x [L + KS - 1, C] bf16: the KS - 1 carried rows (the conv state), then the L
    ///          new rows. Output row l reads input rows l .. l + KS - 1.
    /// \param w [KS, C] bf16, tap-major, exactly as model.q4nx stores ssm_conv1d.weight and
    ///          load_linear_weights() copies it into the ssm buffer
    inline void conv1d_silu(bf16* y, const bf16* x, const bf16* w, int L, int C, int KS) {
        const __m512 one = _mm512_set1_ps(1.0f);
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            int c = 0;
            for (; c + 16 <= C; c += 16) {
                __m512 acc = _mm512_setzero_ps();
                for (int t = 0; t < KS; t++) {
                    const __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x + (size_t)(l + t) * C + c)));
                    const __m512 wv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(w + (size_t)t * C + c)));
                    acc = _mm512_fmadd_ps(xv, wv, acc);
                }
                const __m512 d = _mm512_add_ps(one, _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), acc)));
                _mm256_storeu_si256((__m256i*)(y + (size_t)l * C + c), cpu_func::f32_bf16_rne_512(_mm512_div_ps(acc, d)));
            }
            for (; c < C; c++) {
                float acc = 0.0f;
                for (int t = 0; t < KS; t++) acc += static_cast<float>(x[(size_t)(l + t) * C + c]) * static_cast<float>(w[(size_t)t * C + c]);
                y[(size_t)l * C + c] = static_cast<bf16>(acc / (1.0f + expf(-acc)));
            }
        }
    }

    /// \brief In-place L2 normalisation of `heads` consecutive `dh`-wide heads per row,
    ///        x * rsqrt(sum(x^2) + eps). The second half of the conv overlay, which runs it
    ///        over the q and k heads (conv.cc l2_norm()), not over v.
    inline void l2norm_heads(bf16* y, int ld, int L, int heads, int dh, float eps = 1e-6f) {
        #pragma omp parallel for collapse(2) num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            for (int h = 0; h < heads; h++) {
                bf16* p = y + (size_t)l * ld + (size_t)h * dh;
                const float inv = 1.0f / sqrtf(dot_bf16(p, p, dh) + eps);
                for (int i = 0; i < dh; i++) p[i] = static_cast<bf16>(static_cast<float>(p[i]) * inv);
            }
        }
    }

    /// \brief conv1d_silu followed by l2norm_heads over the first `qk_heads` heads, in one pass
    ///        per row: the head's 128 outputs are rounded to bf16 (as conv1d_silu stores them),
    ///        their sum of squares is taken the way dot_bf16 does, and the normalised head is
    ///        stored, so the result is bitwise that of the two kernels run back to back, without
    ///        the second parallel region and the re-read of the whole conv output.
    /// \param C channels (heads * dh), a multiple of dh; dh a multiple of 16
    inline void conv1d_silu_l2norm(bf16* y, const bf16* x, const bf16* w, int L, int C, int KS,
                                   int qk_heads, int dh, float eps = 1e-6f) {
        assert(dh % 16 == 0 && C % dh == 0 && dh <= 256);
        const __m512 one = _mm512_set1_ps(1.0f);
        const int heads = C / dh;
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int l = 0; l < L; l++) {
            for (int h = 0; h < heads; h++) {
                __m512 v[16];   // dh / 16 <= 16
                __m512 sq = _mm512_setzero_ps();
                for (int j = 0; j < dh / 16; j++) {
                    const int c = h * dh + 16 * j;
                    __m512 acc = _mm512_setzero_ps();
                    for (int t = 0; t < KS; t++) {
                        const __m512 xv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(x + (size_t)(l + t) * C + c)));
                        const __m512 wv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(w + (size_t)t * C + c)));
                        acc = _mm512_fmadd_ps(xv, wv, acc);
                    }
                    const __m512 d = _mm512_add_ps(one, _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), acc)));
                    const __m256i r = cpu_func::f32_bf16_rne_512(_mm512_div_ps(acc, d));
                    if (h < qk_heads) {
                        v[j] = bf16o_fp32_512(r);
                        sq = _mm512_fmadd_ps(v[j], v[j], sq);
                    } else {
                        _mm256_storeu_si256((__m256i*)(y + (size_t)l * C + c), r);
                    }
                }
                if (h < qk_heads) {
                    const __m512 inv = _mm512_set1_ps(1.0f / sqrtf(_mm512_reduce_add_ps(sq) + eps));
                    for (int j = 0; j < dh / 16; j++) {
                        _mm256_storeu_si256((__m256i*)(y + (size_t)l * C + h * dh + 16 * j),
                                            cpu_func::f32_bf16_rne_512(_mm512_mul_ps(v[j], inv)));
                    }
                }
            }
        }
    }

    /// \brief the two small projections of a linear-attention block and their activations,
    ///        one pass per row: alpha[l][h] = exp(A[h] * softplus(x_l . wa[h] + dt_bias[h])),
    ///        beta[l][h] = sigmoid(x_l . wb[h]).
    /// \note  Bitwise what cpu_func::_cpu_mm (one 16-lane FMA chain per output, reduced once) followed by
    ///        _activate_a / _activate_b produces, but x is converted once per row instead of once per
    ///        output, the weights are the fp32 copies of the bf16 rows, and the activations run on
    ///        the row while it is in registers.
    /// \param wa, wb [Hv][K] fp32 (Hv == 16: the activations are one vector per row)
    inline void alpha_beta_proj(float* alpha, float* beta, const float* wa, const float* wb,
                               const bf16* x, int L, int K, int Hv, const float* A, const float* dt_bias) {
        assert(Hv == 16 && K % 16 == 0 && K <= 4096);
        const __m512 one = _mm512_set1_ps(1.0f);
        const __m512 Av = _mm512_loadu_ps(A);
        const __m512 dtv = _mm512_loadu_ps(dt_bias);
        #pragma omp parallel num_threads(prefill_threads())
        {
            alignas(64) float xf[4096];
            #pragma omp for schedule(static)
            for (int l = 0; l < L; l++) {
                const bf16* xr = x + (size_t)l * K;
                for (int k = 0; k < K; k += 16) _mm512_store_ps(xf + k, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(xr + k))));
                alignas(64) float ra[16], rb[16];
                for (int n0 = 0; n0 < 2 * Hv; n0 += 8) {
                    const float* wr[8];
                    for (int j = 0; j < 8; j++) { const int n = n0 + j; wr[j] = (n < Hv ? wa + (size_t)n * K : wb + (size_t)(n - Hv) * K); }
                    __m512 acc[8];
                    for (int j = 0; j < 8; j++) acc[j] = _mm512_setzero_ps();
                    for (int k = 0; k < K; k += 16) {
                        const __m512 xv = _mm512_load_ps(xf + k);
                        for (int j = 0; j < 8; j++) acc[j] = _mm512_fmadd_ps(xv, _mm512_loadu_ps(wr[j] + k), acc[j]);
                    }
                    for (int j = 0; j < 8; j++) { const int n = n0 + j; (n < Hv ? ra : rb)[n % Hv] = _mm512_reduce_add_ps(acc[j]); }
                }
                {   // _activate_a
                    const __m512 sum_vec = _mm512_add_ps(_mm512_load_ps(ra), dtv);
                    const __m512 exp_sum = _mm512_exp_ps_corrected(sum_vec);
                    const __m512 softplus_vec = _mm512_log_ps_approx(_mm512_add_ps(one, exp_sum));
                    _mm512_storeu_ps(alpha + (size_t)l * Hv, _mm512_exp_ps_corrected(_mm512_mul_ps(Av, softplus_vec)));
                }
                {   // _activate_b
                    const __m512 neg_y = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_load_ps(rb));
                    const __m512 den_vec = _mm512_add_ps(one, _mm512_exp_ps_corrected(neg_y));
                    _mm512_storeu_ps(beta + (size_t)l * Hv, _mm512_div_ps(one, den_vec));
                }
            }
        }
    }

    /// \brief Token-serial gated delta rule, the GateDeltaNet_prefill overlay.
    ///
    /// Per token t, per value head:  S <- exp(g_t) S ;  S <- S + k_t (beta_t (v_t - S^T k_t))^T ;
    /// o_t = S^T q_t / sqrt(DK). This is HF's torch_recurrent_gated_delta_rule, i.e. what
    /// the array runs; S2's dumps come from the chunked form and agree with it to ~1e-6.
    ///
    /// \param out   [L, Hv * DV] bf16, head-major (the reference's linear_core)
    /// \param S     [Hv][DK][DV] fp32, the recurrent state in the kv-cache slot, updated in
    ///              place. [DK][DV] is the array's layout (S_buffer in
    ///              FLM_Xclbin/Qwen3_5/gate_delta_net_prefill host.cpp) and HF's.
    /// \param qkv   conv output, row l at qkv + l * ld; q / k already L2-normalised (the
    ///              conv overlay does that), q not yet scaled
    /// \param alpha [L, Hv] fp32, exp(g) -- the decay, NOT the log-space g
    /// \param beta  [L, Hv] fp32, sigmoid(b)
    /// \note Parallel over (head, 16-wide value slab): a slab of S is DK x 16 floats, stays
    ///       in L1 across tokens, and the update is independent per value column.
    inline void gated_delta_net(bf16* out, float* S, const bf16* qkv, int ld,
                                int q_off, int k_off, int v_off,
                                const float* alpha, const float* beta,
                                int L, int Hqk, int Hv, int DK, int DV) {
        assert(Hv % Hqk == 0 && DV % 16 == 0);
        const int   groups = Hv / Hqk;
        const int   slabs  = DV / 16;
        const float scale  = 1.0f / sqrtf((float)DK);

        // A unit is one value head and GDN_SLABS adjacent 16-column slabs. The per-token chain
        // over the DK rows of S is latency bound (one FMA accumulator per slab), so the unit
        // keeps GDN_SLABS independent chains in flight; each slab's arithmetic and summation
        // order are exactly the one-slab-at-a-time loop's, so the result is bitwise the same.
        // Token t's rank-1 update and token t+1's decay share one sweep over S (one load and
        // one store per row instead of two of each); the state is stored after the update, and
        // the last token's "next decay" is 1.0f, which is exact.
        constexpr int SG = GDN_SLABS;
        assert(slabs % SG == 0 && DK % 16 == 0);
        const int units = Hv * (slabs / SG);

        #pragma omp parallel num_threads(prefill_threads())
        {
            std::vector<float> qk_buf((size_t)DK * 4 + 16);
            float* qf  = (float*)(((uintptr_t)qk_buf.data() + 63) & ~(uintptr_t)63);
            float* kf  = qf + DK;
            float* kf2 = kf + DK;   //!< next token's k
            #pragma omp for schedule(static)
            for (int u = 0; u < units; u++) {
                const int hv = u / (slabs / SG);
                const int vs0 = (u % (slabs / SG)) * SG;
                const int hq = hv / groups;
                float* Sh = S + (size_t)hv * DK * DV + (size_t)vs0 * 16;
                auto load_qk = [&](int l, float* qdst, float* kdst) {
                    const bf16* row = qkv + (size_t)l * ld;
                    const bf16* qp  = row + q_off + (size_t)hq * DK;
                    const bf16* kp  = row + k_off + (size_t)hq * DK;
                    const __m512 sc = _mm512_set1_ps(scale);
                    for (int i = 0; i < DK; i += 16) {
                        if (qdst) _mm512_storeu_ps(qdst + i, _mm512_mul_ps(bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(qp + i))), sc));
                        _mm512_storeu_ps(kdst + i, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(kp + i))));
                    }
                };
                // prologue: decay of token 0 and its kv_mem
                load_qk(0, qf, kf);
                __m512 mem[SG];
                {
                    const __m512 a = _mm512_set1_ps(alpha[hv]);
                    for (int j = 0; j < SG; j++) mem[j] = _mm512_setzero_ps();
                    for (int i = 0; i < DK; i++) {
                        const __m512 kk = _mm512_set1_ps(kf[i]);
                        float* sp = Sh + (size_t)i * DV;
                        for (int j = 0; j < SG; j++) {
                            const __m512 s = _mm512_mul_ps(_mm512_loadu_ps(sp + 16 * j), a);
                            _mm512_storeu_ps(sp + 16 * j, s);
                            mem[j] = _mm512_fmadd_ps(s, kk, mem[j]);
                        }
                    }
                }
                for (int l = 0; l < L; l++) {
                    const bf16* row = qkv + (size_t)l * ld;
                    const bf16* vp  = row + v_off + (size_t)hv * DV + (size_t)vs0 * 16;
                    const bool  more = l + 1 < L;
                    const __m512 b = _mm512_set1_ps(beta[(size_t)l * Hv + hv]);
                    const __m512 a_next = _mm512_set1_ps(more ? alpha[(size_t)(l + 1) * Hv + hv] : 1.0f);
                    __m512 delta[SG];
                    for (int j = 0; j < SG; j++) {
                        const __m512 vv = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(vp + 16 * j)));
                        delta[j] = _mm512_mul_ps(_mm512_sub_ps(vv, mem[j]), b);
                    }
                    if (more) load_qk(l + 1, nullptr, kf2);
                    else for (int i = 0; i < DK; i++) kf2[i] = 0.0f;
                    __m512 o[SG];
                    for (int j = 0; j < SG; j++) { o[j] = _mm512_setzero_ps(); mem[j] = _mm512_setzero_ps(); }
                    for (int i = 0; i < DK; i++) {
                        const __m512 kk = _mm512_set1_ps(kf[i]);
                        const __m512 qq = _mm512_set1_ps(qf[i]);
                        const __m512 kn = _mm512_set1_ps(kf2[i]);
                        float* sp = Sh + (size_t)i * DV;
                        for (int j = 0; j < SG; j++) {
                            const __m512 s = _mm512_fmadd_ps(kk, delta[j], _mm512_loadu_ps(sp + 16 * j));
                            o[j] = _mm512_fmadd_ps(s, qq, o[j]);
                            const __m512 sn = _mm512_mul_ps(s, a_next);
                            _mm512_storeu_ps(sp + 16 * j, sn);
                            mem[j] = _mm512_fmadd_ps(sn, kn, mem[j]);
                        }
                    }
                    for (int j = 0; j < SG; j++) {
                        _mm256_storeu_si256((__m256i*)(out + (size_t)l * Hv * DV + (size_t)hv * DV + (size_t)(vs0 + j) * 16),
                                            cpu_func::f32_bf16_rne_512(o[j]));
                    }
                    if (more) {
                        // next token: its q and k, k is already in kf2
                        load_qk(l + 1, qf, kf);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Tensor dumps, for scoring the engine against models_simple/dumps/minicpm_v/.
    //
    // Compiled in at DEBUG_LEVEL >= 1 only (MINICPM_V_DUMP below expands to nothing
    // otherwise), and inert unless FLM_DUMP_DIR is set. Files follow
    // REFERENCE_MANIFEST.md's names and layout -- <dir>/layers/NN/<name>.npy, float32 --
    // so host_ref/compare_engine.py can diff the two trees directly.
    // ---------------------------------------------------------------------------------
    struct dumper_t {
        std::string dir;
        bool active = false;   //!< set by the engine around a prefill; decode never dumps
        int  layer  = -1;      //!< -1 = top level

        dumper_t() {
            if (const char* e = std::getenv("FLM_DUMP_DIR")) dir = e;
        }
        bool on() const { return active && !dir.empty(); }

        template <typename T>
        void operator()(const char* name, const T* p, size_t rows, size_t cols, size_t ld = 0) {
            if (!on()) return;
            if (ld == 0) ld = cols;
            std::string sub = dir;
            if (layer >= 0) {
                char nn[32];
                std::snprintf(nn, sizeof(nn), "/layers/%02d", layer);
                sub += nn;
            }
            std::filesystem::create_directories(sub);
            std::vector<float> f(rows * cols);
            for (size_t r = 0; r < rows; r++)
                for (size_t c = 0; c < cols; c++)
                    f[r * cols + c] = static_cast<float>(p[r * ld + c]);
            _write_npy(sub + "/" + name + ".npy", f.data(), {rows, cols});
        }

        /// \brief 3-D float dump (the recurrent state)
        void dump3(const char* name, const float* p, size_t a, size_t b, size_t c) {
            if (!on()) return;
            char nn[32];
            std::snprintf(nn, sizeof(nn), "/layers/%02d", layer);
            std::string sub = dir + (layer >= 0 ? std::string(nn) : std::string());
            std::filesystem::create_directories(sub);
            _write_npy(sub + "/" + name + ".npy", p, {a, b, c});
        }

        static void _write_npy(const std::string& path, const float* p, std::vector<size_t> shape) {
            std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
            size_t n = 1;
            for (size_t i = 0; i < shape.size(); i++) {
                dict += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
                n *= shape[i];
            }
            dict += "), }";
            // magic(6) + version(2) + header_len(2) + dict + padding + '\n', 64-byte aligned
            size_t total = 10 + dict.size() + 1;
            dict.append((64 - total % 64) % 64, ' ');
            dict += '\n';
            std::ofstream f(path, std::ios::binary);
            const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
            f.write(magic, 8);
            const uint16_t hl = (uint16_t)dict.size();
            f.write(reinterpret_cast<const char*>(&hl), 2);
            f.write(dict.data(), dict.size());
            f.write(reinterpret_cast<const char*>(p), n * sizeof(float));
        }
    };

    inline dumper_t& dumper() {
        static dumper_t d;
        return d;
    }

} // namespace minicpm_v_cpu

#if DEBUG_LEVEL >= 1
#define MINICPM_V_DUMP(...) minicpm_v_cpu::dumper()(__VA_ARGS__)
#define MINICPM_V_DUMP3(...) minicpm_v_cpu::dumper().dump3(__VA_ARGS__)
#else
#define MINICPM_V_DUMP(...) do {} while (0)
#define MINICPM_V_DUMP3(...) do {} while (0)
#endif

#endif // __MINICPM_V_PREFILL_CPU_HPP__
