// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __GEMMA4E_BFP_GEMM_HPP__
#define __GEMMA4E_BFP_GEMM_HPP__
/// \file gemma4e_bfp_gemm.hpp
/// \brief Prefill projections on the 8-column Q4_1 block-float kernel (FLM_Xclbin/dequant_mm_Q4_1_bfp16).
///
/// One xclbin replaces the dequant + bf16 gemm pair for q, k, v, o, up, gate and down: it reads the packed Q4_1 weights of the
/// layer buffer as they are, and multiplies native block float.  Its input X has to arrive as block-float tiles in the order the
/// cores eat them; `pack_x` makes that from a bf16 matrix.  See the README of the kernel directory for the layouts.
///
/// The kernel works on whole 512 x 512 passes: the row count of a call, and Din and Dout of every projection, are multiples of 512.

#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "npu_utils/npu_utils.hpp"
#include "gemma4e_npu_def.hpp"

// GCC / clang need the AVX-512 feature set named on each function that uses its intrinsics and have a builtin to test the CPU at run time;
// MSVC accepts the intrinsics anywhere and needs cpuid / xgetbv for the test.
#if defined(_MSC_VER) && !defined(__clang__)
#define GEMMA4E_AVX512_FN
#define GEMMA4E_AVX512_VBMI_FN
/// true when the CPU and the OS support AVX-512 F + BW + VL + VBMI
inline bool gemma4e_cpu_has_avx512_vbmi() {
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    if (!(r[2] & (1 << 27))) return false;                          // OS has not enabled xsave
    if ((_xgetbv(0) & 0xE6) != 0xE6) return false;                  // OS does not save the zmm / opmask state
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 16)) && (r[1] & (1 << 30)) && (r[1] & (1 << 31)) && (r[2] & (1 << 1));   // F, BW, VL, VBMI
}
#else
#define GEMMA4E_AVX512_FN __attribute__((target("avx512f,avx512bw,avx512vl")))
#define GEMMA4E_AVX512_VBMI_FN __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
inline bool gemma4e_cpu_has_avx512_vbmi() { return __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"); }
#endif

namespace gemma4e_bfp {

/// rows of one core row, and the four core rows of a pass
constexpr uint32_t ROWS_PER_CORE = 128;
constexpr uint32_t CORE_ROWS     = 4;
constexpr uint32_t ROWS_PER_PASS = ROWS_PER_CORE * CORE_ROWS;   // 512
/// outputs of one core, and the eight columns of a pass
constexpr uint32_t OUT_PER_CORE  = 64;
constexpr uint32_t COLUMNS       = 8;
constexpr uint32_t OUT_PER_PASS  = OUT_PER_CORE * COLUMNS;      // 512
/// width of the k slab that is staged in a memtile at once
constexpr uint32_t K_PER_SLAB    = 512;
/// bytes of one slab of X: 128 rows x 512 k at 9 bytes per 8 values
constexpr size_t   X_SLAB_BYTES  = (size_t)ROWS_PER_CORE * K_PER_SLAB * 9 / 8;

/// Core-local addresses of the runtime parameters (K_div_k, activation) of the even and odd core of a pair.  They are assigned by
/// the compiler and move whenever a core's buffers change: after a rebuild of the xclbin re-read them from
/// FLM_Xclbin/dequant_mm_Q4_1_bfp16/build/<type>/bitstream/from_iron/dequant_mm.mlir.prj/input_with_addresses.mlir.
/// A stale address shows up as a hang, not as wrong numbers.
constexpr uint32_t CT_RTP_ADDRESS_EVEN_ROW  = 53760;
constexpr uint32_t ACT_RTP_ADDRESS_EVEN_ROW = 53824;
constexpr uint32_t CT_RTP_ADDRESS_ODD_ROW   = 61952;
constexpr uint32_t ACT_RTP_ADDRESS_ODD_ROW  = 62016;

/// threads that pack the activations (the other host passes of the prefill use four as well)
constexpr int PACK_THREADS = 8;

/// activations the kernel can apply to its output tile
enum activation_t { ACT_NONE = 0, ACT_SILU = 1, ACT_GELU = 2 };

/// bytes of the packed X for `rows` rows of width `k` (rows is a multiple of 512, k of 512)
inline size_t x_bytes(size_t rows, size_t k) { return rows * k * 9 / 8; }

/// \brief One block of 8 values as the NPU stores it: a shared exponent byte, then 8 two's complement mantissa bytes.
/// \note  The core's own conversion truncates, so this does too.  Only the top 7 bits of a bf16 mantissa exist, which keeps the
///        arithmetic in integers.
inline void bfp16_block8_from_bf16(const uint16_t* v, uint8_t* out9) {
    unsigned max_exp = 0;
    for (int e = 0; e < 8; e++) {
        const unsigned exp = (v[e] >> 7) & 0xFFu;
        if (exp > max_exp) max_exp = exp;
    }
    out9[0] = (uint8_t)max_exp;
    for (int e = 0; e < 8; e++) {
        const unsigned bits = v[e];
        const unsigned exp  = (bits >> 7) & 0xFFu;
        const unsigned mant = bits & 0x7Fu;
        if (exp == 255) { out9[1 + e] = 0; continue; }                      // inf / nan: left out, as the host converter does
        // 24 bit mantissa >> 17: the implicit one plus the top 6 bits of the stored mantissa
        int magnitude = (int)((exp ? (0x80u | mant) : mant) >> 1);
        const int dropped = (int)(mant & 1u);                                // truncation rounds a negative value down
        int q = (bits & 0x8000u) ? -(magnitude + dropped) : magnitude;
        q = (int8_t)q;
        const unsigned shift = max_exp - exp;
        q = (shift >= 8) ? (q < 0 ? -1 : 0) : (q >> shift);
        out9[1 + e] = (uint8_t)q;
    }
}

/// four rows of one k tile: 8 bf16 values from each of `r0`..`r3`, one row per 128 bit lane
GEMMA4E_AVX512_FN
inline __m512i load_4_rows(const uint16_t* r0, const uint16_t* r1, const uint16_t* r2, const uint16_t* r3) {
    __m512i v = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)r0));
    v = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i*)r1), 1);
    v = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i*)r2), 2);
    return _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i*)r3), 3);
}

/// the elementwise product of two such vectors, rounded to bf16 the way the host's multiply pass does (truncation)
GEMMA4E_AVX512_FN
inline __m512i product_of_4_rows(__m512i a, __m512i b) {
    auto widen = [](__m256i half) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(half), 16)); };
    auto narrow = [](__m512 f) { return _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(f), 16)); };
    const __m256i lo = narrow(_mm512_mul_ps(widen(_mm512_castsi512_si256(a)), widen(_mm512_castsi512_si256(b))));
    const __m256i hi = narrow(_mm512_mul_ps(widen(_mm512_extracti64x4_epi64(a, 1)), widen(_mm512_extracti64x4_epi64(b, 1))));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/// four rows of one k tile (`v`: 8 bf16 per row, one row per 128 bit lane) -> four blocks of 9 bytes, back to back at `out36`.
/// Same arithmetic as bfp16_block8_from_bf16, on 16 bit lanes: the shared exponent is the largest exponent of a row, every
/// mantissa is shifted right by its distance from it, negative values round down.
GEMMA4E_AVX512_VBMI_FN
inline void blocks_from_vector_avx512(__m512i v, uint8_t* out36) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i exponent = _mm512_and_si512(_mm512_srli_epi16(v, 7), _mm512_set1_epi16(0xFF));
    const __m512i mantissa = _mm512_and_si512(v, _mm512_set1_epi16(0x7F));

    // largest exponent of every row (a row is one 128 bit lane of 8 values), copied to all 8 values of the row
    __m512i row_max = _mm512_max_epu16(exponent, _mm512_srli_epi32(exponent, 16));
    row_max = _mm512_max_epu16(row_max, _mm512_shuffle_epi32(row_max, (_MM_PERM_ENUM)0xB1));
    row_max = _mm512_max_epu16(row_max, _mm512_shuffle_epi32(row_max, (_MM_PERM_ENUM)0x4E));
    const __m512i first_value = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
    row_max = _mm512_shuffle_epi8(row_max, first_value);

    // magnitude: the stored mantissa with its implicit one, without the lowest bit; that bit is what truncation drops
    const __mmask32 normal = _mm512_test_epi16_mask(exponent, exponent);
    const __m512i with_one = _mm512_or_si512(mantissa, _mm512_maskz_mov_epi16(normal, _mm512_set1_epi16(0x80)));
    const __m512i magnitude = _mm512_srli_epi16(with_one, 1);
    const __m512i dropped = _mm512_and_si512(mantissa, _mm512_set1_epi16(1));
    const __m512i negative = _mm512_sub_epi16(zero, _mm512_add_epi16(magnitude, dropped));
    __m512i q = _mm512_mask_blend_epi16(_mm512_movepi16_mask(v), magnitude, negative);
    q = _mm512_mask_mov_epi16(q, _mm512_cmpeq_epi16_mask(exponent, _mm512_set1_epi16(0xFF)), zero);   // inf / nan: left out

    // align to the shared exponent (a distance of 15 or more leaves only the sign)
    const __m512i distance = _mm512_min_epu16(_mm512_sub_epi16(row_max, exponent), _mm512_set1_epi16(15));
    q = _mm512_srav_epi16(q, distance);

    // 32 mantissa bytes and the 4 shared exponents -> [exp, 8 mantissas] x 4
    static const uint8_t pick[64] = {
         64,  0,  1,  2,  3,  4,  5,  6,  7,   72,  8,  9, 10, 11, 12, 13, 14, 15,
         80, 16, 17, 18, 19, 20, 21, 22, 23,   88, 24, 25, 26, 27, 28, 29, 30, 31,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    const __m512i bytes     = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(q));
    const __m512i exponents = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(row_max));
    const __m512i blocks = _mm512_permutex2var_epi8(bytes, _mm512_loadu_si512(pick), exponents);
    _mm512_mask_storeu_epi8(out36, (__mmask64)0xFFFFFFFFFULL, blocks);
}

/// \brief Packs a matrix into the tile order the kernel eats: the bf16 matrix `a` [rows, k] (row pitch `ld` elements), or, with
///        GLU, the elementwise product `a * b` of two such matrices (the mlp's gate * up, rounded to bf16 like the host multiply).
///
/// For every 128-row block and every 512-wide k slab: 1024 tiles of 8 rows x 8 k, 72 bytes each, ordered
///     [k slice of 256 (2)] [16-row group (8)] [k half of 128 (2)] [k tile (16)] [row tile in the group (2)]
/// Rows outside [row_begin, row_end) are written as zeros.  Blocks of 128 rows x one slab are independent and run in parallel.
/// \param use_avx512  false forces the scalar reference path (the vector one is bit-identical)
template <bool GLU>
inline void pack_tiles(uint8_t* dst, const uint16_t* a, const uint16_t* b, size_t row_begin, size_t row_end, size_t rows,
                       size_t k, size_t ld, bool use_avx512) {
    const size_t slabs = k / K_PER_SLAB;
    static const bool has_avx512 = gemma4e_cpu_has_avx512_vbmi();
    const bool vector_path = use_avx512 && has_avx512;
    static const uint16_t zero_row[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma omp parallel for schedule(static) num_threads(PACK_THREADS)
    for (long item = 0; item < (long)((rows / ROWS_PER_CORE) * slabs); item++) {
        const size_t blk = item / slabs, slab = item % slabs;
        uint8_t* slab_out = dst + ((size_t)blk * slabs + slab) * X_SLAB_BYTES;
        // For one 8-row tile row (k slice j, 16-row group i, row tile z) the 32 tiles t = (k half, k tile) read 8 rows front to back,
        // 8 sequential streams of 512 bytes, which the prefetcher handles; the output tiles sit two apart ([t][z] interleave) and the
        // other row tile fills the gaps right after, while they are still in the cache.
        for (int j = 0; j < 2; j++) for (int i = 0; i < 8; i++) for (int z = 0; z < 2; z++) {
            const size_t row0 = blk * ROWS_PER_CORE + i * 16 + z * 8;
            const size_t col0 = slab * K_PER_SLAB + j * 256;
            const uint16_t* ra[8];
            const uint16_t* rb[8];
            for (int r = 0; r < 8; r++) {
                const bool live = row0 + r >= row_begin && row0 + r < row_end;
                ra[r] = live ? a + (row0 + r) * ld + col0 : zero_row;
                if constexpr (GLU) rb[r] = live ? b + (row0 + r) * ld + col0 : zero_row;
            }
            uint8_t* out = slab_out + (((size_t)(j * 8 + i) * 32) * 2 + z) * 72;
            for (int t = 0; t < 32; t++, out += 144) {
                const uint16_t* pa[8];
                const uint16_t* pb[8];
                for (int r = 0; r < 8; r++) {
                    pa[r] = (ra[r] == zero_row) ? zero_row : ra[r] + (size_t)t * 8;
                    if constexpr (GLU) pb[r] = (rb[r] == zero_row) ? zero_row : rb[r] + (size_t)t * 8;
                }
                if (vector_path) {
                    for (int half = 0; half < 2; half++) {
                        __m512i v4 = load_4_rows(pa[4 * half], pa[4 * half + 1], pa[4 * half + 2], pa[4 * half + 3]);
                        if constexpr (GLU)
                            v4 = product_of_4_rows(v4, load_4_rows(pb[4 * half], pb[4 * half + 1], pb[4 * half + 2], pb[4 * half + 3]));
                        blocks_from_vector_avx512(v4, out + 36 * half);
                    }
                }
                else {
                    for (int r = 0; r < 8; r++) {
                        uint16_t vals[8];
                        for (int e = 0; e < 8; e++) {
                            if constexpr (GLU) {
                                const uint32_t fa = (uint32_t)pa[r][e] << 16, fb = (uint32_t)pb[r][e] << 16;
                                float x, y; memcpy(&x, &fa, 4); memcpy(&y, &fb, 4);
                                const float p = x * y; uint32_t pbits; memcpy(&pbits, &p, 4);
                                vals[e] = (uint16_t)(pbits >> 16);
                            }
                            else vals[e] = pa[r][e];
                        }
                        bfp16_block8_from_bf16(vals, out + 9 * r);
                    }
                }
            }
        }
    }
}

/// bf16 matrix [rows, k] (row pitch `ld`) -> tile order; rows at or after `rows_valid` become zeros
inline void pack_x(uint8_t* dst, const uint16_t* src, size_t rows_valid, size_t rows, size_t k, size_t ld, bool use_avx512 = true) {
    pack_tiles<false>(dst, src, nullptr, 0, rows_valid, rows, k, ld, use_avx512);
}

/// gate * up (both [rows, k]) -> tile order, in one pass; rows outside [row_begin, row_end) become zeros
inline void pack_glu(uint8_t* dst, const uint16_t* gate, const uint16_t* up, size_t row_begin, size_t row_end, size_t rows,
                     size_t k, bool use_avx512 = true) {
    pack_tiles<true>(dst, gate, up, row_begin, row_end, rows, k, k, use_avx512);
}

/// \brief The control sequence of one projection Y[L, Dout] = act(X[L, Din] W^T) on the kernel.
/// \param weight_offset         byte offset of the matrix inside the weight buffer (kernel argument 2)
/// \param weight_band_stride    bytes between the 512-row bands of the matrix that are used (the matrix is contiguous: the
///                              bytes of 512 rows; the up/gate region alternates up and gate bands: twice that)
/// \note  X is kernel argument 1, Y argument 0.
inline void generate_sequence(npu_sequence* seq, uint32_t L, uint32_t Din, uint32_t Dout,
                              size_t weight_offset, size_t weight_band_stride, int activation) {
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    assert(L % ROWS_PER_PASS == 0 && Din % K_PER_SLAB == 0 && Dout % OUT_PER_PASS == 0);

    const uint32_t k_slabs     = Din / K_PER_SLAB;
    const uint32_t row_passes  = L / ROWS_PER_PASS;
    const uint32_t out_passes  = Dout / OUT_PER_PASS;
    const uint32_t x_block_bytes = (uint32_t)(k_slabs * X_SLAB_BYTES);   // all slabs of one 128-row block
    const uint32_t w_group_bytes = (uint32_t)get_quantization_byte_size((size_t)OUT_PER_CORE * Din, flm_q41);  // 64 output rows

    seq->clear_cmds();

    // both cores of every pair learn the number of k slabs and the activation
    for (int row = 2; row < 6; row += 2) {
        for (uint32_t col = 0; col < COLUMNS; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_RTP_ADDRESS_EVEN_ROW, k_slabs);
            seq->rtp_write(tile, ACT_RTP_ADDRESS_EVEN_ROW, activation);
        }
    }
    for (int row = 3; row < 6; row += 2) {
        for (uint32_t col = 0; col < COLUMNS; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_RTP_ADDRESS_ODD_ROW, k_slabs);
            seq->rtp_write(tile, ACT_RTP_ADDRESS_ODD_ROW, activation);
        }
    }

    for (uint32_t i = 0; i < row_passes; i++) {
        for (uint32_t j = 0; j < out_passes; j++) {
            const uint32_t round  = i * out_passes + j;
            const uint32_t bd_set = (round % 2) * 8;               // two sets of descriptors: one fills while the other drains
            const size_t y_offset = (size_t)i * ROWS_PER_PASS * Dout + (size_t)j * OUT_PER_PASS;
            const size_t w_band   = weight_offset + (size_t)j * weight_band_stride;

            const npu_bd_id y_bd_lo = npu_bd_id(0 + bd_set);
            const npu_bd_id y_bd_hi = npu_bd_id(3 + bd_set);
            const npu_bd_id x_bd    = npu_bd_id(1 + bd_set);
            const npu_bd_id w_bd_lo = npu_bd_id(2 + bd_set);
            const npu_bd_id w_bd_hi = npu_bd_id(4 + bd_set);

            for (uint32_t c = 0; c < 4; c++) {
                // weights: the 64 output rows of column c, and those of column c + 4
                seq->npu_dma_memcpy_nd(sizeof(char), 2, MM2S, IT[c], w_bd_lo, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_band + (size_t)c * w_group_bytes)},
                    {1, 1, 1, w_group_bytes}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(sizeof(char), 2, MM2S, IT[c + 4], w_bd_hi, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_band + (size_t)(c + 4) * w_group_bytes)},
                    {1, 1, 1, w_group_bytes}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                // y: 128 rows x 256 columns for each of the two column groups of core row c
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[c], y_bd_lo, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + (size_t)c * ROWS_PER_CORE * Dout)},
                    {1, 1, ROWS_PER_CORE, OUT_PER_CORE * 4}, {0, 0, Dout, 1}, -1, 0, true);
                seq->npu_dma_memcpy_nd(2, 0, S2MM, IT[c + 4], y_bd_hi, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + (size_t)c * ROWS_PER_CORE * Dout + OUT_PER_CORE * 4)},
                    {1, 1, ROWS_PER_CORE, OUT_PER_CORE * 4}, {0, 0, Dout, 1}, -1, 0, true);
                // x: the packed 128-row block of core row c, all its slabs in one linear copy
                seq->npu_dma_memcpy_nd(1, 1, MM2S, IT[2 * c], x_bd, it_channel_0,
                    {0, 0, 0, (uint32_t)((size_t)(i * CORE_ROWS + c) * x_block_bytes)},
                    {1, 1, 1, x_block_bytes}, {0, 0, 0, 1}, -1, 0, false);
            }
            if (round > 0) {
                for (int r = 0; r < 4; r++) {
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < 4; r++) {
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}


/// \brief The projections of the layers that own a kv cache, on the block-float kernel.
///
/// One app per (layer kind, projection); the sequence of an app depends on the row count of the batch, so `setup` rebuilds them
/// when it changes.  The packed activations live here too: one buffer for each input width (hidden size, attention width, mlp width).
class gemma4e_bfp_block {
public:
    enum kind_t { Q = 0, K, V, O, UP, GATE, DOWN, NUM_KINDS };

    gemma4e_bfp_block(gemma4e_desc* desc, npu_xclbin_manager* npu, npu_app_manager* manager) : desc(desc), npu(npu) {
        for (int slot = 0; slot < 2; slot++)
            for (int kind = 0; kind < NUM_KINDS; kind++)
                apps[slot][kind] = manager->create_app();
    }

    /// Sizes the packed buffers and rebuilds the sequences for a batch of `rows` rows (a multiple of 512).
    void setup(int rows) {
        if (rows == rows_) return;
        assert(rows % ROWS_PER_PASS == 0);
        const size_t D = desc->D;
        x_hidden   = npu->create_bo_buffer<uint8_t>(x_bytes(rows, D));
        x_attn     = npu->create_bo_buffer<uint8_t>(x_bytes(rows, desc->DQ > desc->SWA_DQ ? desc->DQ : desc->SWA_DQ));
        x_mlp      = npu->create_bo_buffer<uint8_t>(x_bytes(rows, desc->INTERMEDIATE_SIZE));

        for (int slot = 0; slot < 2; slot++) {
            const gemma4e_layer_type_t type = slot ? e_gemma4e_global_layer : e_gemma4e_swa_layer;
            gemma4e_layer_weight_def& W = desc->weight_desc(type);
            const size_t I  = desc->get_intermediate_size(type);
            const size_t DQ = desc->get_DQ(type);
            // the bytes of 512 output rows of a matrix with `din` inputs: the step from one pass over the outputs to the next
            auto band = [](size_t din) { return get_quantization_byte_size((size_t)OUT_PER_PASS * din, flm_q41); };

            auto make = [&](kind_t kind, size_t din, size_t dout, size_t offset, size_t stride, int act) {
                generate_sequence(apps[slot][kind].seq(), rows, din, dout, offset, stride, act);
            };
            make(Q,    D,  DQ,                  W.attn_q.offset,      band(D),     ACT_NONE);
            make(K,    D,  desc->get_DK(type),  W.attn_k.offset,      band(D),     ACT_NONE);
            make(V,    D,  desc->get_DV(type),  W.attn_v.offset,      band(D),     ACT_NONE);
            make(O,    DQ, D,                   W.attn_output.offset, band(DQ),    ACT_NONE);
            // up and gate alternate in 512-row chunks: [up 0][gate 0][up 1][gate 1] ...
            make(UP,   D,  I,                   W.ffn_up_gate.offset,              2 * band(D), ACT_NONE);
            make(GATE, D,  I,                   W.ffn_up_gate.offset + band(D),    2 * band(D), ACT_GELU);
            make(DOWN, I,  D,                   W.ffn_down.offset,    band(I),     ACT_NONE);
        }
        rows_ = rows;
    }

    /// the app of one projection for a layer of this type (sliding or global)
    npu_app& app(kind_t kind, gemma4e_layer_type_t type) { return apps[is_global_layer(type) ? 1 : 0][kind]; }

    /// Packs `rows` rows of a bf16 matrix [rows, width] into one of the packed buffers and makes it visible to the NPU.
    void pack(buffer<uint8_t>& dst, const buffer<bf16>& src, int rows, size_t width) {
        pack_x(dst.data(), reinterpret_cast<const uint16_t*>(src.data()), rows, rows, width, width);
        dst.sync_to_device();
    }

    /// Packs gate * up (rows [row_begin, row_end) of two bf16 matrices [rows, width]) into the mlp input buffer in one pass.
    void pack_gate_up(const buffer<bf16>& gate, const buffer<bf16>& up, int row_begin, int row_end, int rows, size_t width) {
        pack_glu(x_mlp.data(), reinterpret_cast<const uint16_t*>(gate.data()), reinterpret_cast<const uint16_t*>(up.data()),
                 row_begin, row_end, rows, width);
        x_mlp.sync_to_device();
    }

    buffer<uint8_t> x_hidden;   ///< packed input of q, k, v, up and gate: width = hidden size
    buffer<uint8_t> x_attn;     ///< packed input of o: width = the layer kind's attention width
    buffer<uint8_t> x_mlp;      ///< packed input of down: width = the mlp width

private:
    gemma4e_desc* desc;
    npu_xclbin_manager* npu;
    npu_app apps[2][NUM_KINDS];   ///< [sliding, global][projection]
    int rows_ = -1;
};

}  // namespace gemma4e_bfp

#endif  // __GEMMA4E_BFP_GEMM_HPP__
