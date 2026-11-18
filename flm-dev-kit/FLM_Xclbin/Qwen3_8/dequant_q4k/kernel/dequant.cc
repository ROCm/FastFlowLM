// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "aie_kernel_utils.h"

// Q4_K dequant, 32x256 block, parallel 32.
//
//     w[r][c] = S[r] * s8[c / 32][r] * q[c][r]  +  M[r] * m8[c / 32][r]
//
// M is stored negated by the converter (_pack_q4k in
// FLM_Q4NX_Converter/q4nx/model_converter.py), so both terms are added here --
// GGUF's Q4_K subtracts its min, this format does not.
#define NPU_M_TILE 32
#define NPU_K_TILE 256
#define NPU_GROUP  32

constexpr int GROUPS_PER_BLOCK = NPU_K_TILE / NPU_GROUP; // 8

/// @brief One 32x256 q4_k block as it sits on disk, 4736 B.
///
/// Everything is column major over the block: a column of quants is exactly
/// NPU_M_TILE nibbles (row 2b in the low nibble of byte b, row 2b+1 in the
/// high), a group of scales/mins is NPU_M_TILE uint8, and S/M carry one bf16
/// per row because a 256-column block is exactly one super-block.
typedef struct {
    uint8 scales[GROUPS_PER_BLOCK * NPU_M_TILE];  // 256 B
    uint8 mins  [GROUPS_PER_BLOCK * NPU_M_TILE];  // 256 B
    uint4 qs    [NPU_M_TILE * NPU_K_TILE / 2];    // 4096 B, column major
    bf16  S     [NPU_M_TILE];                     // 64 B
    bf16  M     [NPU_M_TILE];                     // 64 B, already negated
} q4k_block;

static_assert(sizeof(q4k_block) == 4736, "q4k block must match the on-disk stride");

constexpr int qw_prod_lock = 0;
constexpr int qw_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;

/// @brief Dequantize one block into column-pair interleaved bf16.
///
/// The output is *not* row major: it is w[128 column pairs][32 rows][2], so
/// every contiguous run is 2 bf16 = one 32-bit word.  That is the granularity
/// the receiving memtile BD addresses at, so the transpose to row major costs
/// nothing there (ct_to_mt_dimension in iron/q4nx_dequant.py) while a transpose
/// here would cost ~1024 narrow stores per block instead of these 256.
///
/// @param w_ptr    output, NPU_M_TILE * NPU_K_TILE bf16
/// @param blk      the quantized block
void dequant_core_q4k(
    bf16 * __restrict w_ptr,
    q4k_block * __restrict blk
) {
    constexpr int pr = NPU_M_TILE;
    // one 512-bit uint4 load is 128 nibbles, i.e. this many whole columns
    constexpr int NIBBLES_PER_LOAD = 128;
    constexpr int COLS_PER_LOAD    = NIBBLES_PER_LOAD / pr; // 4
    constexpr int LOADS_PER_GROUP  = NPU_GROUP / COLS_PER_LOAD; // 8

    uint4 * it_quant = blk->qs;
    uint8 * it_scale = blk->scales;
    uint8 * it_min   = blk->mins;

    // S and M are per row and shared by all 8 groups of the block.
    aie::vector<bf16, pr> v_S = aie::load_v<pr>(blk->S);
    aie::vector<bf16, pr> v_M = aie::load_v<pr>(blk->M);

    AIE_LOOP_RANGE(GROUPS_PER_BLOCK, GROUPS_PER_BLOCK)
    for (int g = 0; g < GROUPS_PER_BLOCK; g++)
    {
        // uint8 -> bf16 is exact (bf16 holds every integer up to 256), so the
        // only rounding on the metadata path is the S*s8 product below.
        aie::accum<accfloat, pr> s_acc;
        s_acc.from_vector(aie::to_float(aie::load_v<pr>(it_scale), 0));
        aie::accum<accfloat, pr> m_acc;
        m_acc.from_vector(aie::to_float(aie::load_v<pr>(it_min), 0));
        it_scale += pr;
        it_min   += pr;

        aie::vector<bf16, pr> v_sc =
            aie::mul(v_S, s_acc.template to_vector<bf16>()).template to_vector<bf16>();
        aie::vector<bf16, pr> v_mn =
            aie::mul(v_M, m_acc.template to_vector<bf16>()).template to_vector<bf16>();

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(LOADS_PER_GROUP, LOADS_PER_GROUP)
        for (int l = 0; l < LOADS_PER_GROUP; l++) {
            aie::vector<uint4, NIBBLES_PER_LOAD> q_raw =
                aie::load_v<NIBBLES_PER_LOAD>(it_quant);
            it_quant += NIBBLES_PER_LOAD / 2; // uint4* steps two nibbles

            // The quants are unsigned 0..15; sign 0 keeps them that way.
            aie::accum<accfloat, NIBBLES_PER_LOAD> q_acc;
            q_acc.from_vector(aie::to_float(q_raw, 0));
            aie::vector<bf16, NIBBLES_PER_LOAD> q_bf16 =
                q_acc.template to_vector<bf16>();

            aie::vector<bf16, pr> v_out[COLS_PER_LOAD];

            AIE_LOOP_UNROLL_FULL
            for (int k = 0; k < COLS_PER_LOAD; k++) {
                // One extracted column is pr *rows* at a single column, and
                // v_sc / v_mn are per row -- a plain elementwise MAC, no
                // broadcast and no transpose.
                aie::vector<bf16, pr> a_col = q_bf16.template extract<pr>(k);
                aie::accum<accfloat, pr> acc;
                acc.from_vector(v_mn);
                acc = aie::mac(acc, v_sc, a_col);
                v_out[k] = acc.template to_vector<bf16>();
            }

            AIE_LOOP_UNROLL_FULL
            for (int p = 0; p < COLS_PER_LOAD / 2; p++) {
                // [r0c0, r0c1, r1c0, r1c1, ...]: rows stay in order, the two
                // columns of the pair land in one word.
                auto zz = aie::interleave_zip(v_out[2 * p], v_out[2 * p + 1], 1);
                aie::store_v(w_ptr, zz.first);
                aie::store_v(w_ptr + pr, zz.second);
                w_ptr += 2 * pr;
            }
        }
    }
}

extern "C" {
void dequant(
    bf16 * w_ping,
    bf16 * w_pong,
    q4k_block * qw_ping,
    q4k_block * qw_pong
)
{
    AIE_PERSIST_ZERO static bool is_ping = false;
    AIE_PERSIST_ZERO static bool is_qw_ping = false;
    _lock_acquire(w_prod_lock, 1);
    _lock_acquire(qw_cons_lock, 1);

    is_ping = !is_ping;
    bf16 * w_ptr = is_ping ? w_ping : w_pong;

    is_qw_ping = !is_qw_ping;
    q4k_block * blk = is_qw_ping ? qw_ping : qw_pong;

    dequant_core_q4k(w_ptr, blk);

    _lock_release(qw_prod_lock, 1);
    _lock_release(w_cons_lock, 1);
}
}
