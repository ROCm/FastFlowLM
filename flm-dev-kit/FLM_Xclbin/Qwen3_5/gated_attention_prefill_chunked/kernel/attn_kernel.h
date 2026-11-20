// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef ATTN_KERNEL_H
#define ATTN_KERNEL_H
#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

void apply_softmax(
    bf16*__restrict pS,
    bf16*__restrict new_m_local
){
    const bf16*__restrict p_in = pS;
    bf16*__restrict p_out = pS;
#if !defined(__chess__)
    // Peano: a key block's 16 rows x 16 keys are 256 values in a row, so do 4 rows per 64-lane vector
    // (same per-lane math as the 16-lane loop below).
    for (int g = 0; g < lq / 4; g++){
        const aie::vector<bf16, 64> m4 = aie::concat(aie::broadcast<bf16, 16>(new_m_local[4 * g]),
                                                     aie::broadcast<bf16, 16>(new_m_local[4 * g + 1]),
                                                     aie::broadcast<bf16, 16>(new_m_local[4 * g + 2]),
                                                     aie::broadcast<bf16, 16>(new_m_local[4 * g + 3]));
        for (int b = 0; b < 8; b++){
            aie::vector<bf16, 64> s_vec = aie::load_v<64>(p_in + b * lq * lk + g * 4 * lk);
            aie::accum<accfloat, 64> x = aie::mac(aie::mul(m4, neg_exp_scale), s_vec, exp_scale);
            aie::store_v(p_out + b * lq * lk + g * 4 * lk, aie::exp2<bf16>(x.template to_vector<float>()));
        }
    }
    return;
#endif
    for (int q = 0; q < lq; q++){
        aie::accum<accfloat, 16> minus_m = aie::mul(aie::broadcast<bf16, 16>(new_m_local[q]), neg_exp_scale);
        for (int b = 0; b < 8; b++)
        chess_prepare_for_pipelining chess_loop_range(8, 8)
        {
            aie::vector<bf16, 16> s_vec = aie::load_v<16>(p_in + b * lq * lk + q * lk);
            aie::accum<accfloat, 16> x = aie::mac(minus_m, s_vec, exp_scale);
            aie::store_v(p_out + b * lq * lk + q * lk, aie::exp2<bf16>(x.template to_vector<float>()));
        }
    }
}

alignas(aie::vector_decl_align) int idx_array[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
void apply_mask_and_get_max(
    bf16*__restrict pS, 
    bf16*__restrict m,
    int pointer_block_inner_k, 
    int pointer_block_inner_q, 
    int pointer_block_inner_k_current
){
    aie::vector<int, 16> idx_vec = aie::load_v<16>(idx_array);
    aie::vector<int, lk> l_vec = aie::broadcast<int, lk>(pointer_block_inner_k);
    l_vec = aie::add(l_vec, idx_vec);
    aie::vector<int, lk> r_vec = aie::broadcast<int, lk>(pointer_block_inner_q);
    r_vec = aie::add(r_vec, idx_vec);

    const bool fully_visible = pointer_block_inner_k_current + lk - 1 <= pointer_block_inner_q
                            && pointer_block_inner_k_current > pointer_block_inner_k + lk - 1;
    if (fully_visible){
#if !defined(__chess__)
        // Peano: the 16 x 16 block and running max are 256 values in a row, 64 lanes at a time
        for (int i = 0; i < lq * lk / 64; i++){
            aie::vector<bf16, 64> s_vec = aie::load_v<64>(pS + i * 64);
            aie::vector<bf16, 64> m_vec = aie::load_v<64>(m + i * 64);
            aie::store_v(m + i * 64, aie::max(m_vec, s_vec));
        }
        return;
#endif
        for (int i = 0; i < lq; i++){
            aie::vector<bf16, lk> s_vec = aie::load_v<lk>(pS);
            aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m);
            aie::store_v(m, aie::max(m_vec, s_vec));
            pS += lk;
            m += lk;
        }
        return;
    }

    for (int i = 0; i < lq; i++){
        int current_idx = pointer_block_inner_k_current + 15 - i;
        aie::mask<lk> mask_l = aie::gt(current_idx, l_vec);
        aie::mask<lk> mask_r = aie::le(current_idx, r_vec);
        aie::vector<bf16, lk> In_vec = aie::load_v<lk>(pS);
        aie::vector<bf16, lk> s_vec = aie::select(bf16(-0x1.FEp127f), In_vec, mask_r);
        s_vec = aie::select(bf16(-0x1.FEp127f), s_vec, mask_l);

        aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m);
        aie::mask<lk> mask_m = aie::gt(s_vec, m_vec);
        aie::vector<bf16, lk> m_vec_new = aie::select(m_vec, s_vec, mask_m);
        aie::store_v(pS, s_vec);
        aie::store_v(m, m_vec_new);
        pS += lk;
        m += lk;
    }
}

// c = exp(prev_m - new_m) per row; exactly 1 where the running max was kept.
// Returns true if any row needs the y / l rescale.
bool calculate_c(float* c, bf16* prev_m_local, bf16* new_m_local) {
    aie::vector<bf16, lq> prev_m = aie::load_v<lq>(prev_m_local);
    aie::vector<bf16, lq> new_m = aie::load_v<lq>(new_m_local);
    aie::mask<lq> moved = aie::neq(prev_m, new_m);
    aie::vector<bf16, lq> diff = aie::sub(prev_m, new_m);
    aie::accum<accfloat, lq> diff_acc = aie::mul(diff, exp_scale);
    aie::vector<bf16, lq> correct = aie::exp2<bf16>(diff_acc.template to_vector<float>());
    aie::accum<accfloat, lq> correct_acc;
    correct_acc.from_vector(correct);
    aie::vector<float, lq> c_vec = aie::select(aie::broadcast<float, lq>(1.0f), correct_acc.template to_vector<float>(), moved);
    aie::store_v(c, c_vec);
#if !defined(__chess__)
    // Peano: one vector compare instead of a scalar loop (same test: any c not exactly 1.0f)
    aie::vector<int32, lq> c_int = aie::load_v<lq>((const int32*)c);
    return !aie::eq(c_int, aie::broadcast<int32, lq>(0x3F800000)).full();
#else
    const int* c_bits = (const int*)c;
    int moved_rows = 0;
    for (int i = 0; i < lq; i++){
        moved_rows += (c_bits[i] != 0x3F800000);
    }
    return moved_rows > 0;
#endif
}

void reorder_s(bf16*__restrict pS) {
    for (int b = 0; b < 8; b++){
        bf16 *pS1 = pS + b * 256;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        aie::vector<bf16, 32> S00 = aie::filter_even(S0, 8);
        aie::vector<bf16, 32> S01 = aie::filter_odd(S0, 8);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
        aie::vector<bf16, 32> S10 = aie::filter_even(S1, 8);
        aie::vector<bf16, 32> S11 = aie::filter_odd(S1, 8);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
        aie::vector<bf16, 32> S20 = aie::filter_even(S2, 8);
        aie::vector<bf16, 32> S21 = aie::filter_odd(S2, 8);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);
        aie::vector<bf16, 32> S30 = aie::filter_even(S3, 8);
        aie::vector<bf16, 32> S31 = aie::filter_odd(S3, 8);

        S0 = aie::concat(S00, S10);
        S1 = aie::concat(S01, S11);
        S2 = aie::concat(S20, S30);
        S3 = aie::concat(S21, S31);

        aie::store_v(pS1, S0);
        aie::store_v(pS1 + 64, S1);
        aie::store_v(pS1 + 128, S2);
        aie::store_v(pS1 + 192, S3);
    }
}

void calculate_l(float* l, float*__restrict c, bf16*__restrict pS) {
    // calculate l
    using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;

    // calculte l = correct * l_left + sum(f)
    aie::vector<bf16, 64> Ones = aie::broadcast<bf16, 64>(1.0f);
    aie::vector<bf16, MMUL::size_C> acc_C0 = aie::zeros<bf16, MMUL::size_C>(); 
    aie::vector<bf16, MMUL::size_C> acc_C1 = aie::zeros<bf16, MMUL::size_C>(); 
    MMUL C0(acc_C0);
    MMUL C1(acc_C1);

#if !defined(__chess__)
    // Peano: what the emulated mmul::mac does (A and transpose(B) to bfp16, native mac), with the constant B
    // converted once instead of on every mac
    using ACC = aie::accum<accfloat, 64>;
    const auto ones_bfp = ::to_v64bfp16ebs8(ACC(aie::transpose(Ones, 8, 8)));
    ACC acc0(acc_C0);
    ACC acc1(acc_C1);
    for (int b = 0; b < 8; b++){
        bf16 *pS1 = pS + b * 256;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);

        acc0 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(ACC(S0)), ones_bfp, acc0, 0, 0, 0);
        acc1 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(ACC(S2)), ones_bfp, acc1, 0, 0, 0);

        acc0 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(ACC(S1)), ones_bfp, acc0, 0, 0, 0);
        acc1 = ::mac_8x8_8x8T_conf(::to_v64bfp16ebs8(ACC(S3)), ones_bfp, acc1, 0, 0, 0);
    }
    C0 = MMUL(acc0.template to_vector<float>());
    C1 = MMUL(acc1.template to_vector<float>());
#else
    for (int b = 0; b < 8; b++){
        bf16 *pS1 = pS + b * 256;
        aie::vector<bf16, 64> S0 = aie::load_v<64>(pS1);
        aie::vector<bf16, 64> S1 = aie::load_v<64>(pS1 + 64);
        aie::vector<bf16, 64> S2 = aie::load_v<64>(pS1 + 128);
        aie::vector<bf16, 64> S3 = aie::load_v<64>(pS1 + 192);

        C0.mac(S0, Ones);
        C1.mac(S2, Ones);

        C0.mac(S1, Ones);
        C1.mac(S3, Ones);
    }
#endif

    auto sum0 = aie::filter_even(C0.template to_vector<float>(), 4);
    auto sum00 = aie::filter_even(sum0, 2);
    auto sum01 = aie::filter_even(sum00, 1);
    auto sum1 = aie::filter_even(C1.template to_vector<float>(), 4);
    auto sum10 = aie::filter_even(sum1, 2);
    auto sum11 = aie::filter_even(sum10, 1);
    auto sum2 = aie::concat(sum01, sum11);
    aie::accum<accfloat, 16> sum;
    sum.from_vector(sum2);

    aie::vector<float, 16> c_float32 = aie::load_v<16>(c);
    aie::vector<float, 16> l_float32 = aie::load_v<16>(l);

    aie::accum<accfloat, 16> l_out;
    l_out = mac_elem_16_accuracy_safe(l_float32, c_float32, sum, 0, 0, 0);
    aie::store_v(l, l_out.template to_vector<float>());
}

void calculate_y(float* y, float* c) {
    // calculate y
    for (int i = 0; i < lq / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        aie::vector<float, 8> c0 = aie::broadcast<float, 8>(c[i * 8]);
        aie::vector<float, 8> c1 = aie::broadcast<float, 8>(c[i * 8 + 1]);
        aie::vector<float, 8> c2 = aie::broadcast<float, 8>(c[i * 8 + 2]);
        aie::vector<float, 8> c3 = aie::broadcast<float, 8>(c[i * 8 + 3]);  
        aie::vector<float, 8> c4 = aie::broadcast<float, 8>(c[i * 8 + 4]);  
        aie::vector<float, 8> c5 = aie::broadcast<float, 8>(c[i * 8 + 5]);  
        aie::vector<float, 8> c6 = aie::broadcast<float, 8>(c[i * 8 + 6]);  
        aie::vector<float, 8> c7 = aie::broadcast<float, 8>(c[i * 8 + 7]);  
        aie::vector<float, 64> CORRECT = aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

        float* pY = y + i * 8 * dh;
        for (unsigned j = 0; j < dh / 8; j += 1){
            aie::vector<float, 64> Y = aie::load_v<64>(pY);
            aie::accum<accfloat, 64> ACC_Y = aie::mul(CORRECT, Y);
            aie::store_v(pY, ACC_Y.template to_vector<float>());
            pY += 64;
        }
    }
}

// Q (16 rows) and K (16 keys) arrive as bfp16 tiles [dh/8][2][72 B]: a tile is 8 rows (keys) x 8 dh, each row one
// block. Same math as aie::mmul<8, 8, 8, bf16, bf16> built with AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16, minus its
// conversions.
void _attn_qk_bfp16(bf16 *__restrict pS, const bfp16ebs8 *__restrict pQ, const bfp16ebs8 *__restrict pK) {
    using ACC = aie::accum<accfloat, 64>;
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sQ(pQ);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sK(pK);

    ACC C00(aie::zeros<bf16, 64>());
    ACC C01(aie::zeros<bf16, 64>());
    ACC C10(aie::zeros<bf16, 64>());
    ACC C11(aie::zeros<bf16, 64>());

    // two k steps per iteration: Peano emits 8 MMULs in 10 bundles this way (4 in 8 for one step)
    for (unsigned i = 0; i < dh / 8; i += 2)
    chess_prepare_for_pipelining chess_loop_range(dh / 16, dh / 16) {
        aie::block_vector<bfp16ebs8, 64> Q0a = sQ.pop();
        aie::block_vector<bfp16ebs8, 64> Q1a = sQ.pop();
        aie::block_vector<bfp16ebs8, 64> K0a = sK.pop();
        aie::block_vector<bfp16ebs8, 64> K1a = sK.pop();
        aie::block_vector<bfp16ebs8, 64> Q0b = sQ.pop();
        aie::block_vector<bfp16ebs8, 64> Q1b = sQ.pop();
        aie::block_vector<bfp16ebs8, 64> K0b = sK.pop();
        aie::block_vector<bfp16ebs8, 64> K1b = sK.pop();

        C00 = mac_8x8_8x8T(Q0a, K0a, C00);
        C01 = mac_8x8_8x8T(Q0a, K1a, C01);
        C10 = mac_8x8_8x8T(Q1a, K0a, C10);
        C11 = mac_8x8_8x8T(Q1a, K1a, C11);
        C00 = mac_8x8_8x8T(Q0b, K0b, C00);
        C01 = mac_8x8_8x8T(Q0b, K1b, C01);
        C10 = mac_8x8_8x8T(Q1b, K0b, C10);
        C11 = mac_8x8_8x8T(Q1b, K1b, C11);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(), C01.template to_vector<bf16>(), 8);
    auto mout1 = aie::interleave_zip(C10.template to_vector<bf16>(), C11.template to_vector<bf16>(), 8);

    aie::store_v(pS, aie::mul(mout0.first, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 64, aie::mul(mout0.second, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 128, aie::mul(mout1.first, (bf16)scale).template to_vector<bf16>(0));
    aie::store_v(pS + 192, aie::mul(mout1.second, (bf16)scale).template to_vector<bf16>(0));
}

// y += p v for one 16-key tile. p: the 16 x 16 softmax tile after reorder_s (four 8 x 8 tiles), turned into bfp16
// here; V arrives transposed as bfp16 tiles [dh/8][2][72 B] (a tile: 8 dh x 8 keys, each dh one block of 8 keys).
void _attn_fv_bfp16(float *__restrict pY, const bf16 *__restrict pS, const bfp16ebs8 *__restrict pV) {
    using ACC = aie::accum<accfloat, 64>;
    constexpr unsigned size_C = 64;
    aie::block_vector<bfp16ebs8, 64> P0 = ACC(aie::load_v<64>(pS)).template to_vector<bfp16ebs8>();
    aie::block_vector<bfp16ebs8, 64> P1 = ACC(aie::load_v<64>(pS + 64)).template to_vector<bfp16ebs8>();
    aie::block_vector<bfp16ebs8, 64> P2 = ACC(aie::load_v<64>(pS + 128)).template to_vector<bfp16ebs8>();
    aie::block_vector<bfp16ebs8, 64> P3 = ACC(aie::load_v<64>(pS + 192)).template to_vector<bfp16ebs8>();
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sV(pV);

    float *__restrict pY1 = pY;
    float *__restrict pY2 = pY + (dh / 8) * size_C;

    for (unsigned j = 0; j < dh / 8; j += 2)
    chess_prepare_for_pipelining chess_loop_range(dh / 16, dh / 16) {
        aie::block_vector<bfp16ebs8, 64> V00 = sV.pop();
        aie::block_vector<bfp16ebs8, 64> V01 = sV.pop();
        aie::block_vector<bfp16ebs8, 64> V10 = sV.pop();
        aie::block_vector<bfp16ebs8, 64> V11 = sV.pop();

        ACC Y00(aie::load_v<size_C>(pY1));
        ACC Y01(aie::load_v<size_C>(pY1 + size_C));
        ACC Y10(aie::load_v<size_C>(pY2));
        ACC Y11(aie::load_v<size_C>(pY2 + size_C));

        Y00 = mac_8x8_8x8T(P0, V00, Y00);
        Y01 = mac_8x8_8x8T(P0, V10, Y01);
        Y10 = mac_8x8_8x8T(P2, V00, Y10);
        Y11 = mac_8x8_8x8T(P2, V10, Y11);

        Y00 = mac_8x8_8x8T(P1, V01, Y00);
        Y01 = mac_8x8_8x8T(P1, V11, Y01);
        Y10 = mac_8x8_8x8T(P3, V01, Y10);
        Y11 = mac_8x8_8x8T(P3, V11, Y11);

        aie::store_v(pY1, Y00.template to_vector<float>());
        aie::store_v(pY1 + size_C, Y01.template to_vector<float>());
        aie::store_v(pY2, Y10.template to_vector<float>());
        aie::store_v(pY2 + size_C, Y11.template to_vector<float>());
        pY1 += 2 * size_C;
        pY2 += 2 * size_C;
    }
}


// The softmax tiles of a whole 128-key block (both halves) as bfp16 blocks, in place at the start of s: [half][key tile
// 0..7][P0 P1 P2 P3][72 B] (what _attn_fv_bfp16 converted tile by tile; the same conversion, once per block).
// Writes never pass the bytes still to be read: 288 B per tile out, 512 B per tile in.
void convert_p_bfp16(bf16 *__restrict s) {
    using ACC = aie::accum<accfloat, 64>;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8*)s);
    for (int hh = 0; hh < halves; hh++){
        for (int kt = 0; kt < 8; kt++){
            const bf16 *p = s + hh * lq * lk_block + kt * 256;
            aie::block_vector<bfp16ebs8, 64> P0 = ACC(aie::load_v<64>(p)).template to_vector<bfp16ebs8>();
            aie::block_vector<bfp16ebs8, 64> P1 = ACC(aie::load_v<64>(p + 64)).template to_vector<bfp16ebs8>();
            aie::block_vector<bfp16ebs8, 64> P2 = ACC(aie::load_v<64>(p + 128)).template to_vector<bfp16ebs8>();
            aie::block_vector<bfp16ebs8, 64> P3 = ACC(aie::load_v<64>(p + 192)).template to_vector<bfp16ebs8>();
            out << P0;
            out << P1;
            out << P2;
            out << P3;
        }
    }
}

// y += P V for one chunk of V^T: dh blocks [4 c, 4 c + 4) of all 128 keys of the block, laid out [pair of dh blocks 0..1]
// [key tile 0..7][V00 V01 V10 V11][72 B] (V00: dh block j, keys 0-7 of the tile; V01: dh block j, keys 8-15; V10 / V11: dh
// block j + 1). Per tile the 8 MMULs are the ones _attn_fv_bfp16 did; the four y tiles of a (half, pair) stay in the
// accumulators over the 8 key tiles, so y is loaded and stored once per block instead of once per key tile.
void _attn_pv_chunk(float *__restrict y, const bfp16ebs8 *__restrict pb, const bfp16ebs8 *__restrict v, int c) {
    using ACC = aie::accum<accfloat, 64>;
    constexpr unsigned size_C = 64;
    for (int dp = 0; dp < 2; dp++){
        const unsigned jj = c * 4 + dp * 2;
        for (int hh = 0; hh < halves; hh++){
            float *__restrict pY1 = y + hh * lq * dh + jj * size_C;
            float *__restrict pY2 = pY1 + (dh / 8) * size_C;
            ACC Y00(aie::load_v<size_C>(pY1));
            ACC Y01(aie::load_v<size_C>(pY1 + size_C));
            ACC Y10(aie::load_v<size_C>(pY2));
            ACC Y11(aie::load_v<size_C>(pY2 + size_C));
            aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sV((const bfp16ebs8*)((const uint8_t*)v + dp * 2304));
            aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sP((const bfp16ebs8*)((const uint8_t*)pb + hh * 2304));
            for (int kt = 0; kt < 8; kt++)
            chess_prepare_for_pipelining chess_loop_range(8, 8) {
                aie::block_vector<bfp16ebs8, 64> V00 = sV.pop();
                aie::block_vector<bfp16ebs8, 64> V01 = sV.pop();
                aie::block_vector<bfp16ebs8, 64> V10 = sV.pop();
                aie::block_vector<bfp16ebs8, 64> V11 = sV.pop();
                aie::block_vector<bfp16ebs8, 64> P0 = sP.pop();
                aie::block_vector<bfp16ebs8, 64> P1 = sP.pop();
                aie::block_vector<bfp16ebs8, 64> P2 = sP.pop();
                aie::block_vector<bfp16ebs8, 64> P3 = sP.pop();

                Y00 = mac_8x8_8x8T(P0, V00, Y00);
                Y01 = mac_8x8_8x8T(P0, V10, Y01);
                Y10 = mac_8x8_8x8T(P2, V00, Y10);
                Y11 = mac_8x8_8x8T(P2, V10, Y11);

                Y00 = mac_8x8_8x8T(P1, V01, Y00);
                Y01 = mac_8x8_8x8T(P1, V11, Y01);
                Y10 = mac_8x8_8x8T(P3, V01, Y10);
                Y11 = mac_8x8_8x8T(P3, V11, Y11);
            }
            aie::store_v(pY1, Y00.template to_vector<float>());
            aie::store_v(pY1 + size_C, Y01.template to_vector<float>());
            aie::store_v(pY2, Y10.template to_vector<float>());
            aie::store_v(pY2 + size_C, Y11.template to_vector<float>());
        }
    }
}

void scale_div_aie(bf16* o, bf16 *l, float *y) {
    constexpr int vec_factor = 64;

    // organize l
    aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l[0]);
    aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l[1]);
    aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l[2]);
    aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l[3]);
    aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l[4]);
    aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l[5]);
    aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l[6]);
    aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l[7]);

    auto LL00 = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

    aie::vector<float, vec_factor> Y00 = aie::load_v<vec_factor>(y);
    aie::accum<accfloat, vec_factor> Y00_acc;
    Y00_acc.from_vector(Y00);
    aie::accum<accfloat, vec_factor> AL00 = aie::mul(Y00_acc.template to_vector<bf16>(), LL00);
    aie::store_v(o, AL00.template to_vector<bf16>());
}
#endif