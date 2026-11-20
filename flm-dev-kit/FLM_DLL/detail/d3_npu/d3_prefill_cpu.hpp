// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file d3_prefill_cpu.hpp
/// \brief Backend switch plus the host kernels of the d3 decision encoder.
///
/// The host kernels come from minicpm_v_4_7_npu (FLM_DLL/ADDING_A_NEW_MODEL.md S5 step 1: the
/// complete host model). They read the same device-layout bytes the array reads (reorder_cpy'd
/// q4_k / q8_0 blocks) and leave behind exactly what an array stage leaves, so the GEMMs can
/// move between host and array without touching their neighbours. Changes for d3: attention
/// takes a `causal` flag (d3 runs it bidirectionally), and rig_dump() writes named checkpoints
/// to compare with the PyTorch reference (not shipped with the dev kit).
#ifndef __D3_PREFILL_CPU_HPP__
#define __D3_PREFILL_CPU_HPP__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _OPENMP
#endif
#include "typedef.hpp"
#include "buffer.hpp"
#include "weight_desc.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"

// ---------------------------------------------------------------------------
// Backend selection.
//
// d3 registers one bitstream: the prefill GEMM (mm.xclbin). conv1d, the gated delta rule and
// attention always run on the host here -- token-serial stages pay a bitstream switch they do
// not amortize, and
// the bidirectional attention d3 needs has no array kernel yet.
//
//   make PREFILL_MM=cpu    -> -DFLM_PREFILL_MM_ON_CPU   -> D3_MM_ON_NPU = 0: no xclbin at all
// ---------------------------------------------------------------------------
#ifndef D3_MM_ON_NPU
#ifdef FLM_PREFILL_MM_ON_CPU
#define D3_MM_ON_NPU 0
#else
#define D3_MM_ON_NPU 1
#endif
#endif

/// Whether the prefill scratch must be device BOs. When 0 it is plain host memory and
/// sync_to/from_device() must never be called on it (buffer<T>::sync_* asserts on a
/// non-BO), which is what scratch_push() below is for.
#define D3_SCRATCH_ON_DEVICE D3_MM_ON_NPU

namespace d3_cpu {

    // ---------------------------------------------------------------------------------
    // Scratch discipline (FLM_DLL/ADDING_A_NEW_MODEL.md S2.5).
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
#if D3_SCRATCH_ON_DEVICE
        return npu->create_bo_buffer<T>(n);
#else
        (void)npu;
        return buffer<T>(n);
#endif
    }

    /// \brief flush a host write to scratch; a no-op when scratch is host memory
    template <typename T>
    inline void scratch_push(buffer<T>& b) {
#if D3_SCRATCH_ON_DEVICE
        b.sync_to_device();
#else
        (void)b;
#endif
    }

    /// \brief Thread count for the host kernels in this file: cpu_functions.hpp's d3_host_threads(),
    ///        one thread per physical core. The cap is measured, not a guess: on the 16-core /
    ///        32-thread Ryzen AI MAX+ 395 under a load of ~10, a 1 k prompt took 3.1 s at 32 threads
    ///        against 1.2 s at 16. SMT siblings add no FMA throughput, and with every hardware thread
    ///        claimed any other load stalls each parallel region's barrier for a scheduler tick.
    ///        Override with FLM_PREFILL_THREADS.
    inline int prefill_threads() { return d3_host_threads(); }

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
    // (Checked during bring-up against gguf.dequantize() on every tensor of the d3-lite
    // checkpoint.)
    //
    // d3_desc::reorder_cpy then interleaves the blocks of `vblocks` consecutive row
    // blocks per column block (a layout may also split each band into `streams`
    // contiguous halves, one per shim MM2S channel). block_offset() inverts both.
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

    /// every projection in proj_weights: d3_desc::reorder_cpy(..., vertical_blocks = 2)
    static constexpr weight_layout BODY_LAYOUT    = {flm_q4k, 2, 1};

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
            assert(false && "d3_cpu::dequant_row_block: unsupported dtype");
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
        d3_for(L, [&](int l) {
            const bf16* xr = x + (size_t)l * ldx;
            float*      xo = xf.data() + (size_t)l * K;
            for (int k = 0; k < K; k += 16) {
                _mm512_storeu_ps(xo + k, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(xr + k))));
            }
        });

        // row blocks in chunks of 4 as threads come free; each thread keeps one K x QB_ROWS tile, sized on first use
        std::vector<std::vector<float>> tiles(prefill_threads());
        d3_for_dynamic(row_blocks, 4, [&](int tid, long long rb64) {
                const int rb = (int)rb64;
                std::vector<float>& tile = tiles[tid];
                if (tile.empty()) tile.resize((size_t)K * QB_ROWS);
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
        });
    }

    /// \brief Round GEMM activations to values the array's bf16 MAC represents exactly.
    ///
    /// The npu2 `aie::mmul<8,8,8,bf16>` runs with -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16: each
    /// 8-element block along K is cast to bfp16 (shared exponent, 8-bit signed mantissa) by
    /// *truncation*, which loses the low bits of every value smaller than its block's maximum and
    /// biases them toward zero. Rounding to that grid on the host first makes the cast exact, so the
    /// activation side carries round-to-nearest error instead of truncation error (half the size,
    /// unbiased). Gated inputs (o_proj, down_proj) have the most to gain. Per row, in place.
    inline void bfp16_round_rows(bf16* p, int ld, int rows, int cols) {
        // 16 lanes = two 8-element blocks. Per block: the max |x| (3 butterfly steps inside each
        // 8-lane half), its float exponent field E, step = 2^(E - 127 - 6) (so the max lands in
        // [64, 128): 7 magnitude bits + sign), x rounded to a multiple of step and clamped to
        // [-127, 127] steps. Every result has <= 8 significant bits, so the bf16 store is exact.
        const __m512i swap1 = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
        const __m512i swap2 = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
        const __m512i swap4 = _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
        // symmetric: -128 steps would make the block's max |x| exactly 2^E and move the shared
        // exponent up one when the array re-derives it, re-quantizing the block more coarsely
        const __m512  lo = _mm512_set1_ps(-127.0f), hi = _mm512_set1_ps(127.0f);
        const __m512i min_field = _mm512_set1_epi32(7);
        d3_for(rows, [&](int r) {
            bf16* row = p + (size_t)r * ld;
            int c = 0;
            for (; c + 16 <= cols; c += 16) {
                const __m512 v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(row + c)));
                __m512 m = _mm512_abs_ps(v);
                m = _mm512_max_ps(m, _mm512_permutexvar_ps(swap1, m));
                m = _mm512_max_ps(m, _mm512_permutexvar_ps(swap2, m));
                m = _mm512_max_ps(m, _mm512_permutexvar_ps(swap4, m));
                __m512i field = _mm512_srli_epi32(_mm512_castps_si512(m), 23);
                field = _mm512_max_epi32(field, min_field);              // all-zero / tiny blocks stay finite
                const __m512 step = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(field, _mm512_set1_epi32(6)), 23));
                const __m512 inv  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(254 + 6), field), 23));
                __m512 q = _mm512_roundscale_ps(_mm512_mul_ps(v, inv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                q = _mm512_min_ps(_mm512_max_ps(q, lo), hi);
                _mm256_storeu_si256((__m256i*)(row + c), cpu_func::f32_bf16_rne_512(_mm512_mul_ps(q, step)));
            }
            for (; c + 8 <= cols; c += 8) {   // a trailing 8-block (cols % 16 == 8)
                float v[8], m = 0.0f;
                for (int k = 0; k < 8; k++) { v[k] = static_cast<float>(row[c + k]); m = std::max(m, std::fabs(v[k])); }
                if (m == 0.0f) continue;
                int e;
                std::frexp(m, &e);
                const float step = std::ldexp(1.0f, e - 7);
                for (int k = 0; k < 8; k++)
                    row[c + k] = static_cast<bf16>(std::min(127.0f, std::max(-127.0f, std::nearbyint(v[k] / step))) * step);
            }
        });
    }

    /// \brief silu in place, fp32 math, bf16 storage
    inline void silu_inplace(bf16* p, size_t n) {
        const __m512 one = _mm512_set1_ps(1.0f);
        const long long nn = (long long)n;
        d3_for((nn - nn % 16) / 16, [&](long long blk) {
            const long long i = blk * 16;
            const __m512 v = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(p + i)));
            const __m512 d = _mm512_add_ps(one, _mm512_exp_ps_corrected(_mm512_sub_ps(_mm512_setzero_ps(), v)));
            _mm256_storeu_si256((__m256i*)(p + i), cpu_func::f32_bf16_rne_512(_mm512_div_ps(v, d)));
        });
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
    /// \param slab UP_GATE_OUTDIM_REORDEER in d3_desc::load_*_weights (512)
    inline void dequant_mm_upgate(bf16* y, const u8* w, const bf16* x, int L, int K, int IS, int slab) {
        if (L <= 0) return;
        std::vector<bf16> fused((size_t)L * IS * 2);
        dequant_mm(fused.data(), IS * 2, w, x, K, L, K, IS * 2, BODY_LAYOUT);

        const int slices = IS / slab;
        d3_for(L, [&](int l) {
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
        });
    }

    /// \brief hid[L][IS] = up * gate, from dequant_mm_upgate()'s [ up | gate ] rows
    inline void glu_mul(bf16* hid, const bf16* upgate, int L, int IS) {
        d3_for(L, [&](int l) {
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
        });
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

    /// \brief Grouped-query attention over a contiguous key/value range, causal or bidirectional.
    ///
    /// \param out          [L_q, num_heads * DH] bf16, head h at column h * DH
    /// \param q            [L_q, num_heads * DH] bf16, already q-normed and roped
    /// \param k            [L_kv, k_stride] bf16, kv head hk at column hk * DH
    /// \param v            [L_kv, v_stride] bf16, same head layout as k
    /// \param q_pos_begin  absolute (cache) position of q row 0
    /// \param kv_pos_begin absolute (cache) position of k/v row 0
    /// \param scale        softmax scale; 1/sqrt(DH) for this model (q_norm does not fold it)
    /// \param kt_buf       the caller's scratch (K^T and every thread's scores); it only grows.
    ///                     Not allocated per call: a fresh large buffer is an mmap/munmap per call,
    ///                     and the unmaps slowed the array GEMMs around it (330 tokens: +25 ms).
    /// \note q head h reads kv head h / (num_heads / num_kv_heads), which is the reference's
    ///       "B (h R) L d -> B h L R d" grouping.
    /// \note Blocked: one task is ATT_NV query vectors of one kv head (a few query rows times the
    ///       q heads that share it), so every K and V row it reads serves ATT_NV queries. K is
    ///       transposed to fp32 [DH][keys] once per call, so q.k is 16 keys per FMA with no
    ///       horizontal sums. fp32 throughout, bf16 (round to nearest even) on store. A query's
    ///       arithmetic depends only on its own row and its segment's keys, never on how rows
    ///       were packed or scheduled, so packing and threading stay bit-exact.
    static constexpr int ATT_NV = 16;

    /// One thread's per-task scratch for the attention kernels: ATT_NV query vectors of DH, their limits and (row, head).
    struct AttScratch {
        std::vector<float> qf;
        std::vector<int> lim, vrow, vhead;   // lim: last visible key per vector, -1 none
        void init(int nv, int dh) {
            if (!qf.empty()) return;
            qf.resize((size_t)nv * dh);
            lim.resize(nv); vrow.resize(nv); vhead.resize(nv);
        }
    };

    inline void attention(bf16* out, const bf16* q, const bf16* k, const bf16* v,
                          int L_q, int L_kv, int num_heads, int num_kv_heads, int DH,
                          int q_pos_begin, int kv_pos_begin,
                          int k_stride, int v_stride, float scale, bool causal,
                          std::vector<float>& kt_buf) {
        // hard checks (release builds define NDEBUG: an assert here would let a bad shape through)
        if (num_kv_heads <= 0 || num_heads <= 0 || num_heads % num_kv_heads != 0 || DH <= 0 || DH % 16 != 0)
            throw std::invalid_argument("d3_cpu::attention: num_heads must be a multiple of num_kv_heads and head_dim of 16");
        if (L_q <= 0) return;
        if (L_kv <= 0) throw std::invalid_argument("d3_cpu::attention: no keys");
        const int groups   = num_heads / num_kv_heads;
        const int q_stride = num_heads * DH;
        const int Lp       = (L_kv + 15) / 16 * 16;           // keys padded to whole 16-key tiles
        // A task is ATT_NV consecutive (row, head) query vectors of one kv head, row-major then
        // head, whatever the group size: for GQA 4 that is 4 rows, for GQA 6 two rows and two thirds
        // of the next (no idle lanes except in the last task). A query's arithmetic does not depend
        // on which task it lands in.
        const long long total_vecs = (long long)L_q * groups;
        const int row_tasks = (int)((total_vecs + ATT_NV - 1) / ATT_NV);

        // K^T in fp32, [kv head][DH][Lp], padding keys zero (their scores are masked below),
        // then each thread's scores [ATT_NV][Lp]
        const int nthreads = prefill_threads();
        const size_t kt_size = (size_t)num_kv_heads * DH * Lp;
        const size_t need = kt_size + (size_t)nthreads * ATT_NV * Lp;
        if (kt_buf.size() < need) kt_buf.resize(need);
        float* const kt = kt_buf.data();

        const int j_tiles = Lp / 16;
        d3_for((long long)num_kv_heads * j_tiles, [&](long long task) {
                    const int hk = (int)(task / j_tiles);
                    const int j0 = (int)(task % j_tiles) * 16;
                    float tile[16][16];
                    for (int d0 = 0; d0 < DH; d0 += 16) {
                        for (int jj = 0; jj < 16; jj++) {
                            const int j = j0 + jj;
                            if (j < L_kv) {
                                _mm512_storeu_ps(tile[jj], bf16o_fp32_512(_mm256_loadu_si256(
                                    (const __m256i*)(k + (size_t)j * k_stride + (size_t)hk * DH + d0))));
                            } else {
                                _mm512_storeu_ps(tile[jj], _mm512_setzero_ps());
                            }
                        }
                        float* dst = kt + ((size_t)hk * DH + d0) * Lp + j0;
                        for (int dd = 0; dd < 16; dd++)
                            for (int jj = 0; jj < 16; jj++) dst[(size_t)dd * Lp + jj] = tile[jj][dd];
                    }
        });

        // then (kv head, row task) pairs as threads come free; each thread has its own scratch
        std::vector<AttScratch> scratch_of(nthreads);
        d3_for_dynamic((long long)num_kv_heads * row_tasks, 1, [&](int tid, long long task) {
                    const int hk = (int)(task / row_tasks);
                    const int rt = (int)(task % row_tasks);
                    AttScratch& sc_ = scratch_of[tid];
                    sc_.init(ATT_NV, DH);
                    std::vector<float>& qf = sc_.qf;
                    std::vector<int>& lim = sc_.lim;
                    std::vector<int>& vrow = sc_.vrow;
                    std::vector<int>& vhead = sc_.vhead;
                    float* const p = kt + kt_size + (size_t)tid * ATT_NV * Lp;
                    {
                        // this task's vectors; unused slots repeat the first one and are not stored
                        int nv = 0;
                        for (long long t = (long long)rt * ATT_NV; t < total_vecs && nv < ATT_NV; t++) {
                            vrow[nv] = (int)(t / groups);
                            vhead[nv] = hk * groups + (int)(t % groups);
                            nv++;
                        }
                        if (nv == 0) return;
                        for (int n = 0; n < ATT_NV; n++) {
                            const int src = n < nv ? n : 0;
                            const bf16* qh = q + (size_t)vrow[src] * q_stride + (size_t)vhead[src] * DH;
                            for (int d = 0; d < DH; d += 16)
                                _mm512_storeu_ps(qf.data() + (size_t)n * DH + d, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(qh + d))));
                            int j1 = causal ? q_pos_begin + vrow[src] - kv_pos_begin : L_kv - 1;
                            lim[n] = std::min(j1, L_kv - 1);
                        }

                        // scores = (q . k) * scale, 16 keys at a time for all ATT_NV vectors
                        const float* K = kt + (size_t)hk * DH * Lp;
                        for (int j0 = 0; j0 < Lp; j0 += 16) {
                            __m512 acc[ATT_NV];
                            for (int n = 0; n < ATT_NV; n++) acc[n] = _mm512_setzero_ps();
                            for (int d = 0; d < DH; d++) {
                                const __m512 kv = _mm512_loadu_ps(K + (size_t)d * Lp + j0);
                                for (int n = 0; n < ATT_NV; n++)
                                    acc[n] = _mm512_fmadd_ps(_mm512_set1_ps(qf[(size_t)n * DH + d]), kv, acc[n]);
                            }
                            const __m512 sc = _mm512_set1_ps(scale);
                            for (int n = 0; n < ATT_NV; n++)
                                _mm512_storeu_ps(p + (size_t)n * Lp + j0, _mm512_mul_ps(acc[n], sc));
                        }

                        // softmax over each vector's visible keys; invisible keys get weight 0
                        for (int n = 0; n < ATT_NV; n++) {
                            float* pn = p + (size_t)n * Lp;
                            const int j1 = lim[n];
                            if (j1 < 0) {   // nothing visible; cannot happen for a causal prompt
                                std::fill(pn, pn + Lp, 0.0f);
                                continue;
                            }
                            float m = -INFINITY;
                            for (int j = 0; j <= j1; j++) m = std::max(m, pn[j]);
                            float sum = 0.0f;
                            for (int j = 0; j <= j1; j++) {
                                const float e = expf(pn[j] - m);
                                pn[j] = e;
                                sum += e;
                            }
                            const float inv = 1.0f / sum;
                            for (int j = 0; j <= j1; j++) pn[j] *= inv;
                            std::fill(pn + j1 + 1, pn + Lp, 0.0f);
                        }

                        // out = p . V, 16 output dims at a time for all ATT_NV vectors
                        const int jmax = *std::max_element(lim.begin(), lim.end());
                        for (int d0 = 0; d0 < DH; d0 += 16) {
                            __m512 acc[ATT_NV];
                            for (int n = 0; n < ATT_NV; n++) acc[n] = _mm512_setzero_ps();
                            for (int j = 0; j <= jmax; j++) {
                                const __m512 vv = bf16o_fp32_512(_mm256_loadu_si256(
                                    (const __m256i*)(v + (size_t)j * v_stride + (size_t)hk * DH + d0)));
                                for (int n = 0; n < ATT_NV; n++)
                                    acc[n] = _mm512_fmadd_ps(_mm512_set1_ps(p[(size_t)n * Lp + j]), vv, acc[n]);
                            }
                            for (int n = 0; n < nv; n++)
                                _mm256_storeu_si256((__m256i*)(out + (size_t)vrow[n] * q_stride + (size_t)vhead[n] * DH + d0),
                                                    cpu_func::f32_bf16_rne_512(acc[n]));
                        }
                    }
        });
    }

    /// \brief floats of K^T scratch transpose_keys() writes for L keys
    inline size_t keys_t_floats(int L, int num_kv_heads, int DH) {
        return (size_t)num_kv_heads * DH * (size_t)((L + 15) / 16 * 16);
    }

    /// \brief K^T in fp32, [kv head][DH][Lp] with Lp = L rounded up to 16 and the padding keys zero:
    ///        the layout attention() and attention_prefix() score against (16 keys per FMA)
    /// \param k [L, k_stride] bf16, kv head hk at column hk * DH
    inline void transpose_keys(float* kt, const bf16* k, int L, int k_stride, int num_kv_heads, int DH) {
        const int Lp = (L + 15) / 16 * 16;
        const int j_tiles = Lp / 16;
        d3_for((long long)num_kv_heads * j_tiles, [&](long long task) {
                const int hk = (int)(task / j_tiles);
                const int j0 = (int)(task % j_tiles) * 16;
                float tile[16][16];
                for (int d0 = 0; d0 < DH; d0 += 16) {
                    for (int jj = 0; jj < 16; jj++) {
                        const int j = j0 + jj;
                        if (j < L) {
                            _mm512_storeu_ps(tile[jj], bf16o_fp32_512(_mm256_loadu_si256(
                                (const __m256i*)(k + (size_t)j * k_stride + (size_t)hk * DH + d0))));
                        } else {
                            _mm512_storeu_ps(tile[jj], _mm512_setzero_ps());
                        }
                    }
                    float* dst = kt + ((size_t)hk * DH + d0) * Lp + j0;
                    for (int dd = 0; dd < 16; dd++)
                        for (int jj = 0; jj < 16; jj++) dst[(size_t)dd * Lp + jj] = tile[jj][dd];
                }
        });
    }

    /// \brief Bidirectional attention of a question block over TWO key ranges: S cached state keys
    ///        followed by the block's own keys (state-once, `[state | question]` without recomputing the state).
    ///
    /// Query row i sees every state key and every own key (no mask), exactly the question rows of
    /// attention(out, q, [state_k; own_k], [state_v; own_v], L_q, S + L_q, ..., causal = false), and with
    /// the SAME arithmetic: a key's score is one fp32 dot over DH (independent of its tile), the softmax
    /// runs sequentially over the keys in the order [state..., own...], and p . V accumulates sequentially in
    /// that order. The result is therefore bit-identical to the per-row path in which the state rows and the
    /// question rows are computed jointly, and does not depend on how many other blocks share the state
    /// or on the task shape.
    ///
    /// \param out        [L_q, num_heads * DH] bf16
    /// \param q          [L_q, num_heads * DH] bf16, q-normed and roped
    /// \param own_k      [L_q, k_stride] bf16 (post k_norm + RoPE), own_v [L_q, v_stride] bf16
    /// \param state_kt   the state's keys transposed by transpose_keys(S keys), shared by every block of
    ///                   the state (built once per layer and pass); may be null when S == 0
    /// \param state_v    [S, state_v_stride] bf16 (kv head hk at column hk * DH)
    /// \param scratch    the caller's buffer (own K^T and every thread's scores); it only grows
    inline void attention_prefix(bf16* out, const bf16* q, const bf16* own_k, const bf16* own_v, int L_q,
                                 const float* state_kt, const bf16* state_v, int S,
                                 int num_heads, int num_kv_heads, int DH,
                                 int k_stride, int v_stride, int state_v_stride, float scale,
                                 std::vector<float>& scratch) {
        if (num_kv_heads <= 0 || num_heads <= 0 || num_heads % num_kv_heads != 0 || DH <= 0 || DH % 16 != 0)
            throw std::invalid_argument("d3_cpu::attention_prefix: num_heads must be a multiple of num_kv_heads and head_dim of 16");
        if (L_q <= 0) return;
        if (S < 0 || (S > 0 && (!state_kt || !state_v))) throw std::invalid_argument("d3_cpu::attention_prefix: bad state range");
        const int groups   = num_heads / num_kv_heads;
        const int q_stride = num_heads * DH;
        const int Sp = (S + 15) / 16 * 16, Lp = (L_q + 15) / 16 * 16;
        const int J  = S + L_q;                                   // keys per query
        const int T  = S + Lp;                                    // score row stride: [state keys | own keys], tiles may overrun J
        const long long total_vecs = (long long)L_q * groups;
        const int row_tasks = (int)((total_vecs + ATT_NV - 1) / ATT_NV);

        const int nthreads = prefill_threads();
        const size_t kt_size = keys_t_floats(L_q, num_kv_heads, DH);
        const size_t need = kt_size + (size_t)nthreads * ATT_NV * T;
        if (scratch.size() < need) scratch.resize(need);
        float* const kt = scratch.data();
        transpose_keys(kt, own_k, L_q, k_stride, num_kv_heads, DH);

        std::vector<AttScratch> scratch_of(nthreads);
        d3_for_dynamic((long long)num_kv_heads * row_tasks, 1, [&](int tid, long long task) {
                    const int hk = (int)(task / row_tasks);
                    const int rt = (int)(task % row_tasks);
                    AttScratch& sc_ = scratch_of[tid];
                    sc_.init(ATT_NV, DH);
                    std::vector<float>& qf = sc_.qf;
                    std::vector<int>& vrow = sc_.vrow;
                    std::vector<int>& vhead = sc_.vhead;
                    float* const p = kt + kt_size + (size_t)tid * ATT_NV * T;
                    int nv = 0;
                    for (long long t = (long long)rt * ATT_NV; t < total_vecs && nv < ATT_NV; t++) {
                        vrow[nv] = (int)(t / groups);
                        vhead[nv] = hk * groups + (int)(t % groups);
                        nv++;
                    }
                    if (nv == 0) return;
                    for (int n = 0; n < ATT_NV; n++) {
                        const int src = n < nv ? n : 0;
                        const bf16* qh = q + (size_t)vrow[src] * q_stride + (size_t)vhead[src] * DH;
                        for (int d = 0; d < DH; d += 16)
                            _mm512_storeu_ps(qf.data() + (size_t)n * DH + d, bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)(qh + d))));
                    }

                    // scores = (q . k) * scale: the state tiles, then the own tiles
                    const __m512 scv = _mm512_set1_ps(scale);
                    for (int range = 0; range < 2; range++) {
                        const int    len = range == 0 ? S : L_q;
                        const int    ldp = range == 0 ? Sp : Lp;
                        const float* K = (range == 0 ? state_kt : kt) + (size_t)hk * DH * ldp;
                        float*       P = p + (range == 0 ? 0 : S);   // own scores directly after the S state scores
                        for (int j0 = 0; j0 < len; j0 += 16) {
                            __m512 acc[ATT_NV];
                            for (int n = 0; n < ATT_NV; n++) acc[n] = _mm512_setzero_ps();
                            for (int d = 0; d < DH; d++) {
                                const __m512 kv = _mm512_loadu_ps(K + (size_t)d * ldp + j0);
                                for (int n = 0; n < ATT_NV; n++)
                                    acc[n] = _mm512_fmadd_ps(_mm512_set1_ps(qf[(size_t)n * DH + d]), kv, acc[n]);
                            }
                            for (int n = 0; n < ATT_NV; n++)
                                _mm512_storeu_ps(P + (size_t)n * T + j0, _mm512_mul_ps(acc[n], scv));
                        }
                    }

                    // softmax over the joint key list [state..., own...]: the SAME loops as attention() over a
                    // contiguous score row (-ffast-math may vectorize the sum; the same source in the same
                    // function body keeps its order identical)
                    for (int n = 0; n < ATT_NV; n++) {
                        float* pn = p + (size_t)n * T;
                        float m = -INFINITY;
                        for (int j = 0; j < J; j++) m = std::max(m, pn[j]);
                        float sum = 0.0f;
                        for (int j = 0; j < J; j++) {
                            const float e = expf(pn[j] - m);
                            pn[j] = e;
                            sum += e;
                        }
                        const float inv = 1.0f / sum;
                        for (int j = 0; j < J; j++) pn[j] *= inv;
                    }

                    // out = p . V, 16 output dims at a time for all ATT_NV vectors
                    for (int d0 = 0; d0 < DH; d0 += 16) {
                        __m512 acc[ATT_NV];
                        for (int n = 0; n < ATT_NV; n++) acc[n] = _mm512_setzero_ps();
                        for (int j = 0; j < S; j++) {
                            const __m512 vv = bf16o_fp32_512(_mm256_loadu_si256(
                                (const __m256i*)(state_v + (size_t)j * state_v_stride + (size_t)hk * DH + d0)));
                            for (int n = 0; n < ATT_NV; n++)
                                acc[n] = _mm512_fmadd_ps(_mm512_set1_ps(p[(size_t)n * T + j]), vv, acc[n]);
                        }
                        for (int j = 0; j < L_q; j++) {
                            const __m512 vv = bf16o_fp32_512(_mm256_loadu_si256(
                                (const __m256i*)(own_v + (size_t)j * v_stride + (size_t)hk * DH + d0)));
                            for (int n = 0; n < ATT_NV; n++)
                                acc[n] = _mm512_fmadd_ps(_mm512_set1_ps(p[(size_t)n * T + S + j]), vv, acc[n]);
                        }
                        for (int n = 0; n < nv; n++)
                            _mm256_storeu_si256((__m256i*)(out + (size_t)vrow[n] * q_stride + (size_t)vhead[n] * DH + d0),
                                                cpu_func::f32_bf16_rne_512(acc[n]));
                    }
        });
    }

    /// \brief Causal depthwise conv1d + SiLU, the first half of the conv overlay.
    /// \param y [L, C] bf16
    /// \param x [L + KS - 1, C] bf16: the KS - 1 carried rows (the conv state), then the L
    ///          new rows. Output row l reads input rows l .. l + KS - 1.
    /// \param w [KS, C] bf16, tap-major, exactly as model.q4nx stores ssm_conv1d.weight and
    ///          load_linear_weights() copies it into the ssm buffer
    inline void conv1d_silu(bf16* y, const bf16* x, const bf16* w, int L, int C, int KS) {
        const __m512 one = _mm512_set1_ps(1.0f);
        d3_for(L, [&](int l) {
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
        });
    }

    /// \brief In-place L2 normalisation of `heads` consecutive `dh`-wide heads per row,
    ///        x * rsqrt(sum(x^2) + eps). The second half of the conv overlay, which runs it
    ///        over the q and k heads (conv.cc l2_norm()), not over v.
    inline void l2norm_heads(bf16* y, int ld, int L, int heads, int dh, float eps = 1e-6f) {
        d3_for((long long)L * heads, [&](long long task) {
            const int l = (int)(task / heads);
            const int h = (int)(task % heads);
            bf16* p = y + (size_t)l * ld + (size_t)h * dh;
            const float inv = 1.0f / sqrtf(dot_bf16(p, p, dh) + eps);
            for (int i = 0; i < dh; i++) p[i] = static_cast<bf16>(static_cast<float>(p[i]) * inv);
        });
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
        if (Hqk <= 0 || Hv % Hqk != 0 || DV % 16 != 0)
            throw std::invalid_argument("d3_cpu::gated_delta_net: value heads must be a multiple of key heads, value head dim of 16");
        const int   groups = Hv / Hqk;   // value heads per key head (HF order: value head hv reads key head hv / groups)
        const int   slabs  = DV / 16;
        const float scale  = 1.0f / sqrtf((float)DK);

        d3_for((long long)Hv * slabs, [&](long long task) {
            const int hv = (int)(task / slabs);
            const int vs = (int)(task % slabs);
            std::vector<float> qf((size_t)DK), kf((size_t)DK);
            {
                {
                    float*    Sh = S + (size_t)hv * DK * DV + (size_t)vs * 16;
                    const int hq = hv / groups;
                    for (int l = 0; l < L; l++) {
                        const bf16* row = qkv + (size_t)l * ld;
                        const bf16* qp  = row + q_off + (size_t)hq * DK;
                        const bf16* kp  = row + k_off + (size_t)hq * DK;
                        const bf16* vp  = row + v_off + (size_t)hv * DV + (size_t)vs * 16;
                        for (int i = 0; i < DK; i++) {
                            qf[i] = static_cast<float>(qp[i]) * scale;
                            kf[i] = static_cast<float>(kp[i]);
                        }
                        const __m512 a = _mm512_set1_ps(alpha[(size_t)l * Hv + hv]);
                        const __m512 b = _mm512_set1_ps(beta[(size_t)l * Hv + hv]);

                        // decay, and kv_mem = S^T k against the decayed state
                        __m512 mem = _mm512_setzero_ps();
                        for (int i = 0; i < DK; i++) {
                            float* sp = Sh + (size_t)i * DV;
                            const __m512 s = _mm512_mul_ps(_mm512_loadu_ps(sp), a);
                            _mm512_storeu_ps(sp, s);
                            mem = _mm512_fmadd_ps(s, _mm512_set1_ps(kf[i]), mem);
                        }
                        const __m512 vv    = bf16o_fp32_512(_mm256_loadu_si256((const __m256i*)vp));
                        const __m512 delta = _mm512_mul_ps(_mm512_sub_ps(vv, mem), b);

                        // rank-1 update, and the output against the new state
                        __m512 o = _mm512_setzero_ps();
                        for (int i = 0; i < DK; i++) {
                            float* sp = Sh + (size_t)i * DV;
                            const __m512 s = _mm512_fmadd_ps(_mm512_set1_ps(kf[i]), delta, _mm512_loadu_ps(sp));
                            _mm512_storeu_ps(sp, s);
                            o = _mm512_fmadd_ps(s, _mm512_set1_ps(qf[i]), o);
                        }
                        _mm256_storeu_si256((__m256i*)(out + (size_t)l * Hv * DV + (size_t)hv * DV + (size_t)vs * 16),
                                            cpu_func::f32_bf16_rne_512(o));
                    }
                }
            }
        });
    }

    /// \brief conv1d_silu of one segment's rows continuing from a conv state (state-once).
    ///
    /// The conv state is the KS - 1 input rows (the in_proj_qkv outputs BEFORE the conv) that precede
    /// the segment: `tail`, or zeros when null (a segment that starts a sequence). Output row l reads
    /// rows l .. l + KS - 1 of [tail | rows], so a question block that continues from the state's tail
    /// equals the same rows computed after the state in one pass, bit for bit.
    /// \param y        [L, C] bf16
    /// \param rows     [L, C] bf16 (the segment's own conv inputs)
    /// \param tail     [KS - 1, C] bf16 or nullptr
    /// \param tail_out [KS - 1, C] bf16 or nullptr: receives the last KS - 1 rows of [tail | rows], the
    ///                 conv state for whoever continues after this segment (rows before the start: zeros)
    /// \param conv_in  scratch, grows
    inline void conv1d_silu_carry(bf16* y, const bf16* rows, const bf16* tail, const bf16* w, int L, int C, int KS,
                                  std::vector<bf16>& conv_in, bf16* tail_out = nullptr) {
        conv_in.resize((size_t)(L + KS - 1) * C);
        if (tail) memcpy(conv_in.data(), tail, (size_t)(KS - 1) * C * sizeof(bf16));
        else std::fill(conv_in.begin(), conv_in.begin() + (size_t)(KS - 1) * C, bf16(0.0f));
        memcpy(conv_in.data() + (size_t)(KS - 1) * C, rows, (size_t)L * C * sizeof(bf16));
        conv1d_silu(y, conv_in.data(), w, L, C, KS);
        if (tail_out) memcpy(tail_out, conv_in.data() + (size_t)L * C, (size_t)(KS - 1) * C * sizeof(bf16));
    }

    /// \brief gated_delta_net() of a question block that continues from a cached recurrent state:
    ///        S_work <- S_cache, then the rows update S_work in place; S_cache is read only, so any
    ///        number of blocks can continue from it. S_cache null: from zero.
    /// \param S_work [Hv][DK][DV] fp32 scratch of the caller
    inline void gated_delta_net_carry(bf16* out, float* S_work, const float* S_cache, const bf16* qkv, int ld,
                                      int q_off, int k_off, int v_off, const float* alpha, const float* beta,
                                      int L, int Hqk, int Hv, int DK, int DV) {
        const size_t n = (size_t)Hv * DK * DV;
        if (S_cache) memcpy(S_work, S_cache, n * sizeof(float));
        else std::fill(S_work, S_work + n, 0.0f);
        gated_delta_net(out, S_work, qkv, ld, q_off, k_off, v_off, alpha, beta, L, Hqk, Hv, DK, DV);
    }

    /// \brief write a float32 .npy file (the rig dumps below)
    inline void write_npy(const std::string& path, const float* p, std::vector<size_t> shape) {
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

    /// \brief named checkpoint dump (the names the PyTorch reference saves): <$D3_RIG_DUMP>/<name>.npy, float32 [rows, cols]
    /// \note  compiled in only with -DD3_RIG (d3_prefill.hpp's D3_RIG_DUMP macro); the engine
    ///        turns `rig_active` on around a single-prompt encode.
    inline bool& rig_active() { static bool on = false; return on; }
    template <typename T>
    inline void rig_dump(const std::string& name, const T* p, size_t rows, size_t cols, size_t ld) {
        const char* dir = std::getenv("D3_RIG_DUMP");   // read per call: d3_engine_run --rig-dump changes it per prompt
        if (!dir || !*dir || !rig_active()) return;
        // D3_RIG_LAYERS_ONLY=1: only the layer outputs l<i>.out, final_norm.last and vision.feat
        // (D3_RIG_LAYERS_ONLY=2: vision.feat only)
        static const char* lo = std::getenv("D3_RIG_LAYERS_ONLY");
        static const bool layers_only = lo != nullptr;
        if (lo && lo[0] == '2' && name != "vision.feat") return;
        if (layers_only && std::count(name.begin(), name.end(), '.') != 1) return;
        if (layers_only && name.size() > 4 && name[0] == 'l' && name.compare(name.size() - 4, 4, ".out") != 0 && name != "vision.feat" && name != "final_norm.last") return;
        std::filesystem::create_directories(dir);
        std::vector<float> f(rows * cols);
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                f[r * cols + c] = static_cast<float>(p[r * ld + c]);
        write_npy(std::string(dir) + "/" + name + ".npy", f.data(), {rows, cols});
    }

} // namespace d3_cpu

#endif // __D3_PREFILL_CPU_HPP__
