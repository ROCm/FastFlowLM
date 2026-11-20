// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "flm_rsqrt.h"

#ifdef HAS_QK_NORM
///@brief pseduo rms norm: y = x * w
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
///@param overwrite_x_buf: if true, overwrite x_buf with x
void qk_norm(
    const bf16 *restrict x,
    const bf16 *restrict w
){
    const float epsilon = 1e-6;

    constexpr int vector_size = 16;
    constexpr float one_over_D = 1.0f / (float)DH;

    bf16 *it_x_0 = const_cast<bf16*>(x);
    bf16 *it_x_1 = const_cast<bf16*>(x) + vector_size;
    bf16 *it_w_0 = const_cast<bf16*>(w);
    bf16 *it_w_1 = const_cast<bf16*>(w) + vector_size;

    aie::accum<accfloat, vector_size> sum_squares_0 = aie::zeros<accfloat>();
    aie::accum<accfloat, vector_size> sum_squares_1 = aie::zeros<accfloat>();
    aie::vector<float, vector_size> mx_vec;

    const int F = DH / (2 * vector_size);
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        auto x_vec_0 = aie::load_v<vector_size>(it_x_0);
        auto x_vec_1 = aie::load_v<vector_size>(it_x_1);
        sum_squares_0 = aie::mac_square(sum_squares_0, x_vec_0);
        sum_squares_1 = aie::mac_square(sum_squares_1, x_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
    }

    it_x_0 = const_cast<bf16*>(x);
    it_x_1 = const_cast<bf16*>(x) + vector_size;

    mx_vec = aie::add(sum_squares_0.template to_vector<float>(),
                      sum_squares_1.template to_vector<float>());
    float sum = aie::reduce_add(mx_vec);
#if defined(__chess__)
    sum = sum * one_over_D;
    sum = sum + epsilon;
#else
    // Same math as the chess branch, rounded like chess (see AIE_FADD).
    sum = AIE_FMUL(sum, one_over_D);
    sum = AIE_FADD(sum, epsilon);
#endif
    const float divrms = flm_rsqrt(sum);

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bf16, vector_size> x_vec_0 = aie::load_v<vector_size>(it_x_0);
        aie::vector<bf16, vector_size> x_vec_1 = aie::load_v<vector_size>(it_x_1);
        aie::vector<bf16, vector_size> w_vec_0 = aie::load_v<vector_size>(it_w_0);
        aie::vector<bf16, vector_size> w_vec_1 = aie::load_v<vector_size>(it_w_1);
        aie::vector<float, vector_size> wx_vec_0 = aie::mul(x_vec_0, w_vec_0);
        aie::vector<float, vector_size> wx_vec_1 = aie::mul(x_vec_1, w_vec_1);
        aie::vector<bf16, vector_size> o_vec_0 = aie::mul(wx_vec_0, divrms);
        aie::vector<bf16, vector_size> o_vec_1 = aie::mul(wx_vec_1, divrms);
        aie::store_v(it_x_0, o_vec_0);
        aie::store_v(it_x_1, o_vec_1);
        it_x_0 += 2 * vector_size;
        it_x_1 += 2 * vector_size;
        it_w_0 += 2 * vector_size;
        it_w_1 += 2 * vector_size;
    }
}
#endif

void apply_rope(
    bf16 *restrict y,
    bf16 *restrict x,
    bf16 *restrict cos_val,
    bf16 *restrict sin_val
) {
    constexpr int vector_size = 16;
    const int DH_2 = ROPE_DIM / 2;
    constexpr int F = DH_2 / vector_size;
    constexpr int remaining_F = (ROPE_BY_PASS) / vector_size;
    bf16 *it_y_p1 = y;
    bf16 *it_y_p2 = y + DH_2;
    bf16 *it_x_p1 = x;
    bf16 *it_x_p2 = x + DH_2;
    bf16 *it_cos = cos_val;
    bf16 *it_sin = sin_val;
    // |x_p1      |x_p2      |Bypass                |
    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bf16, vector_size> x1_vec = aie::load_v<vector_size>(it_x_p1);
        aie::vector<bf16, vector_size> x2_vec = aie::load_v<vector_size>(it_x_p2);
        aie::vector<bf16, vector_size> cos_vec = aie::load_v<vector_size>(it_cos);
        aie::vector<bf16, vector_size> sin_vec = aie::load_v<vector_size>(it_sin);
        aie::vector<bf16, vector_size> neg_sin_vec = aie::neg(sin_vec);
        aie::accum<accfloat, vector_size> C = aie::mul(x1_vec, cos_vec);
        C = aie::mac(C, x2_vec, neg_sin_vec);
        aie::accum<accfloat, vector_size> D = aie::mul(x1_vec, sin_vec);
        D = aie::mac(D, x2_vec, cos_vec);
        aie::store_v(it_y_p1, C.template to_vector<bf16>());
        aie::store_v(it_y_p2, D.template to_vector<bf16>());
        it_y_p1 += vector_size;
        it_y_p2 += vector_size;
        it_x_p1 += vector_size;
        it_x_p2 += vector_size;
        it_cos += vector_size;
        it_sin += vector_size;
    }
    for (int i = 0; i < remaining_F; i++) // bypass
    chess_prepare_for_pipelining chess_loop_range(8, ) {
        aie::vector<bf16, vector_size> x2_vec = aie::load_v<vector_size>(it_x_p2);
        aie::store_v(it_y_p2, x2_vec);
        it_y_p2 += vector_size;
        it_x_p2 += vector_size;
    }
}
///@brief k/v for one token: k_norm + rotate on k, v copied through.
///@param k: rotated k out
///@param v: v out
///@param qkv: the token's | k | v | half, as staged
///@param rope_w: the weight block, see rope() below
///@param t: which token -- selects the cos/sin pair
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope_kv(
    bf16 *restrict k,
    bf16 *restrict v,
    bf16 * restrict qkv,
    bf16 * restrict rope_w,
    int t
) {
    bf16 *it_k = k;
    bf16 *it_v = v;
    bf16 *it_qkv = qkv;
    bf16 *it_rope = rope_w + t * ROPE_DIM;
    #ifdef HAS_QK_NORM
    bf16 *q_norm_weight = rope_w + L_TOKENS * ROPE_DIM;
    bf16 *k_norm_weight = rope_w + L_TOKENS * ROPE_DIM + DH;
    #endif
    const int DH_2 = ROPE_DIM / 2;

    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + ROPE_DIM / 2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        #ifdef HAS_QK_NORM
        qk_norm(it_qkv, k_norm_weight);
        #endif
        apply_rope(it_k, it_qkv, cos_val, sin_val);
        it_k += DH;
        it_qkv += DH;
    }

    // move v from qkv to v, N
    for (int i = 0; i < DV / 32; i++) {
        aie::vector<bf16, 32> v_vec = aie::load_v<32>(it_qkv);
        aie::store_v(it_v, v_vec);
        it_v += 32;
        it_qkv += 32;
    }
}

///@brief q for one token: q_norm + rotate, per head, with the GQA padding zeroed.
///@param q: rotated, padded q out
///@param qkv: the token's q half, as staged
///@param rope_w: the weight block, see rope() below
///@param t: which token -- selects the cos/sin pair
///@note: C = A cos - B sin, D = A sin + B cos
void pseduo_rope_q(
    bf16 *restrict q,
    bf16 * restrict qkv,
    bf16 * restrict rope_w,
    int t
) {
    bf16 *it_q = q;
    bf16 *it_qkv = qkv;
    bf16 *it_rope = rope_w + t * ROPE_DIM;
    #ifdef HAS_QK_NORM
    bf16 *q_norm_weight = rope_w + L_TOKENS * ROPE_DIM;
    bf16 *k_norm_weight = rope_w + L_TOKENS * ROPE_DIM + DH;
    #endif
    const int DH_2 = ROPE_DIM / 2;

    bf16* cos_val = it_rope;
    bf16* sin_val = it_rope + ROPE_DIM / 2;

    for (int h = 0; h < NUM_KV_HEADS; h++){
        for (int g = 0; g < Q_HEADS_PER_GROUP; g++){
            #ifdef HAS_QK_NORM
            qk_norm(it_qkv, q_norm_weight);
            #endif
            apply_rope(it_q, it_qkv, cos_val, sin_val);
            it_q += DH;
            it_qkv += DH;
        }
        // The GQA padding: ATTN_GROUPS_PADDING whole *heads* of DH, zeroed, so
        // each kv group occupies Q_HEADS_PER_GROUP_PADDED * DH and the attention
        // CU can index every group with one stride.
        //
        // Two bugs lived here, and the second hid the first.
        //
        // `if constexpr`, not `#if`. ATTN_GROUPS_PADDING is a constexpr int in
        // model_spec.h, not a macro, so the preprocessor never saw a value: it
        // substituted 0 for the unknown identifier, `0 > 0` was false, and this
        // whole block was dropped from every build. Nothing warns -- that
        // substitution is what the standard asks for.
        //
        // And the body advanced by DH_2 twice per padding head, which reads as
        // "two halves of a head" but is not: DH_2 is ROPE_DIM / 2 = 32, the
        // rotation's half-width, not DH / 2 = 128. So it would have zeroed and
        // skipped 64 elements per padding head instead of 256.
        //
        // Between them, q head 6 onwards of every token was written 384
        // elements early, over the tail of the group before it, and the padding
        // it should have left behind kept whatever the previous token put there.
        // Measured at L = 8: cosine 0.386 against the golden, with |padding| up
        // to 9.94 where it should be 0.
        if constexpr (ATTN_GROUPS_PADDING > 0) {
            constexpr int pad_vec = 32;
            aie::vector<bf16, pad_vec> zero = aie::zeros<bf16, pad_vec>();
            for (int i = 0; i < ATTN_GROUPS_PADDING * DH / pad_vec; i++){
                aie::store_v(it_q, zero);
                it_q += pad_vec;
            }
        }
    }
}

///@brief q/k RMSNorm and partial rotary, L_TOKENS tokens per invocation.
///
/// One token at a time, in arrival order. The staging memtile upstream holds the
/// whole L x (DQ + DK + DV) activation and hands this tile one token per BD,
/// rotated to | k | v | q | -- so k and v are on their way to the KV cache while
/// q is still being normed, rather than six q heads behind it.
///
/// The weight block is acquired once, outside the loop, because it is the same
/// for every token except for which cos/sin pair is read out of it:
///
///     rope_w[t * ROPE_DIM              .. + ROPE_DIM/2)   cos for token t
///     rope_w[t * ROPE_DIM + ROPE_DIM/2 .. + ROPE_DIM/2)   sin for token t
///     rope_w[L_TOKENS * ROPE_DIM       .. + DH)           q_norm.weight
///     rope_w[L_TOKENS * ROPE_DIM + DH  .. + DH)           k_norm.weight
///
/// The L = 1 version had one cos/sin pair and put the norm weights at a fixed
/// +ROPE_DIM; the table grew in front of them, so both offsets moved. The host
/// builds it in iron/RoPE.py's documented order -- the two files have to agree
/// and nothing checks them against each other.
///
/// Qwen3.8 norms before rotating, per 256-dim head, and rotates only dims
/// [0, 64). v is untouched: it is here because it shares the fused row.
extern "C" {

void rope(
    bf16* restrict q,
    bf16* restrict k,
    bf16* restrict v,
    bf16* restrict q_in,
    bf16* restrict kv_in,
    bf16* restrict rope_w
)
{
    constexpr int qkv_prod_lock = 0;
    constexpr int qkv_cons_lock = 1;
    constexpr int q_prod_lock = 2;
    constexpr int q_cons_lock = 3;
    constexpr int k_prod_lock = 4;
    constexpr int k_cons_lock = 5;
    constexpr int v_prod_lock = 6;
    constexpr int v_cons_lock = 7;
    constexpr int rope_prod_lock = 8;
    constexpr int rope_cons_lock = 9;

    _lock_acquire(rope_cons_lock);

    for (int t = 0; t < L_TOKENS; t++) {
        _lock_acquire(qkv_cons_lock);
        _lock_acquire(k_prod_lock, 1);
        _lock_acquire(v_prod_lock, 1);
        pseduo_rope_kv(k, v, kv_in, rope_w, t);
        _lock_release(qkv_prod_lock);
        _lock_release(v_cons_lock, 1);
        _lock_release(k_cons_lock, 1);

        _lock_acquire(qkv_cons_lock);
        _lock_acquire(q_prod_lock);
        pseduo_rope_q(q, q_in, rope_w, t);
        _lock_release(q_cons_lock);
        _lock_release(qkv_prod_lock);
    }

    _lock_release(rope_prod_lock);
}


}