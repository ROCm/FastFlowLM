// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include <math.h>
#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void mm_aie(T_out *__restrict pY, const T_in *__restrict pX, const T_in *__restrict pW) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pY1 = pY + (z * colB) * MMUL::size_C;
        T_out *__restrict pY2 = pY + ((z + 1) * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
        chess_prepare_for_pipelining chess_loop_range(2, ) {
            const T_in *__restrict pX1 = pX + z * MMUL::size_A;
            const T_in *__restrict pX2 = pX + (z + 1) * MMUL::size_A;
            const T_in *__restrict pW1 = pW + j * MMUL::size_B;
            const T_in *__restrict pW2 = pW + (j + 1) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> X0 = aie::load_v<MMUL::size_A>(pX1);
            pX1 += MMUL::size_A * rowA;
            aie::vector<T_in, MMUL::size_A> X1 = aie::load_v<MMUL::size_A>(pX2);
            pX2 += MMUL::size_A * rowA;
            aie::vector<T_in, MMUL::size_B> W00 = aie::load_v<MMUL::size_B>(pW1);
            aie::vector<T_in, MMUL::size_B> W0 = aie::transpose(W00, 4, 8);
            pW1 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> W10 = aie::load_v<MMUL::size_B>(pW2);
            aie::vector<T_in, MMUL::size_B> W1 = aie::transpose(W10, 4, 8);
            pW2 += MMUL::size_B * colB;

            aie::vector<T_out, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1); 
            aie::vector<T_out, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
            aie::vector<T_out, MMUL::size_C> acc_Y11 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);

            MMUL Y00(acc_Y00);
            MMUL Y01(acc_Y01);
            MMUL Y10(acc_Y10);
            MMUL Y11(acc_Y11);

            Y00.mac(X0, W0);
            Y01.mac(X0, W1);
            Y10.mac(X1, W0);
            Y11.mac(X1, W1);

            for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, 7) {
                X0 = aie::load_v<MMUL::size_A>(pX1);
                pX1 += MMUL::size_A * rowA;
                X1 = aie::load_v<MMUL::size_A>(pX2);
                pX2 += MMUL::size_A * rowA;
                W00 = aie::load_v<MMUL::size_B>(pW1);
                W0 = aie::transpose(W00, 4, 8);
                pW1 += MMUL::size_B * colB;
                W10 = aie::load_v<MMUL::size_B>(pW2);
                W1 = aie::transpose(W10, 4, 8);
                pW2 += MMUL::size_B * colB;

                Y00.mac(X0, W0);
                Y01.mac(X0, W1);
                Y10.mac(X1, W0);
                Y11.mac(X1, W1);
            }
            
            aie::store_v(pY1, Y00.template to_vector<T_out>());
            pY1 += MMUL::size_C;
            aie::store_v(pY1, Y01.template to_vector<T_out>());
            pY1 += MMUL::size_C;
            aie::store_v(pY2, Y10.template to_vector<T_out>());
            pY2 += MMUL::size_C;
            aie::store_v(pY2, Y11.template to_vector<T_out>());
            pY2 += MMUL::size_C;
        }
    }
}

void silu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){

    bf16 *x_it = x;

    constexpr int vector_size = 16;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSiluBf16_nonLUT(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void gelu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){

    bf16 *x_it = x;

    constexpr int vector_size = 64;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++){
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getGeluBf16_nonLUT(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

template <typename T_in, typename T_out, int vec_factor>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    aie::vector<T_out, vec_factor> Out;
    aie::accum<accfloat, vec_factor> c_acc_in;
    c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
    aie::store_v(out, c_acc_in.template to_vector<T_out>());
}

void passbias_aie(float * bias, const bf16 * w){
    aie::vector<bf16, 8> b_vec_0 = aie::load_v<8>(w);
    aie::vector<bf16, 8> b_vec_1 = aie::load_v<8>(w + 8 * k_slice);
    aie::vector<bf16, 8> b_vec_2 = aie::load_v<8>(w + 8 * k_slice * 2);
    aie::vector<bf16, 8> b_vec_3 = aie::load_v<8>(w + 8 * k_slice * 3);
    aie::vector<bf16, 8> b_vec_4 = aie::load_v<8>(w + 8 * k_slice * 4);
    aie::vector<bf16, 8> b_vec_5 = aie::load_v<8>(w + 8 * k_slice * 5);
    aie::vector<bf16, 8> b_vec_6 = aie::load_v<8>(w + 8 * k_slice * 6);
    aie::vector<bf16, 8> b_vec_7 = aie::load_v<8>(w + 8 * k_slice * 7);

    aie::vector<bf16, 64> vec = aie::concat(b_vec_0, b_vec_1, 
        b_vec_2, b_vec_3, 
        b_vec_4, b_vec_5, 
        b_vec_6, b_vec_7);
    aie::accum<accfloat, 64> acc;
    acc.from_vector(vec);
    aie::store_v(bias, acc.template to_vector<float>());
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int l_prod_lock = 6;

extern "C" {
    void mm(
        const bf16 *__restrict x_ping,
        const bf16 *__restrict x_pong,
        const bf16 *__restrict w_ping,
        const bf16 *__restrict w_pong,
        bf16 *__restrict y_ping,
        bf16 *__restrict y_pong,
        float *__restrict y_acc,
        float *__restrict bias,
        int * rtp_value

    ){
        AIE_PERSIST_ZERO static bool is_w_ping = false;
        AIE_PERSIST_ZERO static bool is_y_ping = false;
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int n_div_n_slice = NPU_N_TILE / n_slice; //rowB
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA

        _lock_acquire(l_prod_lock);
        int num_n = rtp_value[0];
        int add_bias = rtp_value[1];
        int num_m = rtp_value[2];
        int num_k = rtp_value[3];
        int add_act = rtp_value[4];
        for (int n = 0; n < num_n; n++){
            if (add_bias == 1){
                _lock_acquire(w_cons_lock);
                is_w_ping = !is_w_ping;
                const bf16 *w = is_w_ping ? w_ping : w_pong;
                passbias_aie(bias, w);
                _lock_release(w_prod_lock);
                
                for (int j = 1; j < k_div_k_slice; j++){
                    _lock_acquire(w_cons_lock);
                    is_w_ping = !is_w_ping;
                    const bf16 *w = is_w_ping ? w_ping : w_pong;
                    _lock_release(w_prod_lock);
                }
            }

            for (int m = 0; m < num_m; m++){
                if (add_bias == 1){ 
                    for (int ym = 0; ym < NPU_M_TILE / 8; ym++){
                        for (int in = 0; in < 8; in++){
                            passThrough_aie<float, float, 4>(bias, y_acc + ym * 8 * NPU_N_TILE + in * 4);
                            passThrough_aie<float, float, 4>(bias + 4, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32);
                            passThrough_aie<float, float, 4>(bias + 8, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 2);
                            passThrough_aie<float, float, 4>(bias + 12, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 3);
                            passThrough_aie<float, float, 4>(bias + 16, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 4);
                            passThrough_aie<float, float, 4>(bias + 20, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 5);
                            passThrough_aie<float, float, 4>(bias + 24, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 6);
                            passThrough_aie<float, float, 4>(bias + 28, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 7);
                            passThrough_aie<float, float, 4>(bias + 32, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 8);
                            passThrough_aie<float, float, 4>(bias + 36, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 9);
                            passThrough_aie<float, float, 4>(bias + 40, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 10);
                            passThrough_aie<float, float, 4>(bias + 44, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 11);
                            passThrough_aie<float, float, 4>(bias + 48, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 12);
                            passThrough_aie<float, float, 4>(bias + 52, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 13);
                            passThrough_aie<float, float, 4>(bias + 56, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 14);
                            passThrough_aie<float, float, 4>(bias + 60, y_acc + ym * 8 * NPU_N_TILE + in * 4 + 32 * 15);
                        }
                    }
                }
                else{
                    zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
                }

                int k = 0; // IT to MT
                do {
                    for (int j = 0; j < k_div_k_slice; j++){
                        _lock_acquire(w_cons_lock);
                        is_w_ping = !is_w_ping;
                        const bf16 *w = is_w_ping ? w_ping : w_pong;
                        for (int i = 0; i < m_div_m_slice; i++){
                            _lock_acquire(x_cons_lock);
                            is_x_ping = !is_x_ping;
                            const bf16 *x = is_x_ping ? x_ping : x_pong;
                            mm_aie<bf16, float, m_slice / 8, k_slice / 8, n_slice / 4, 8, 8, 4>(y_acc + i * m_slice * n_slice, x, w);
                            _lock_release(x_prod_lock);
                        }
                        _lock_release(w_prod_lock);
                    }
                    k += 1;
                } while (k < num_k);

                for (int i = 0; i < y_num; i++){
                    _lock_acquire(y_prod_lock);
                    is_y_ping = !is_y_ping;
                    bf16 *y = is_y_ping ? y_ping : y_pong;
                    passThrough_aie<float, bf16, 32>(y_acc+i*y_slice, y);
                    if (add_act == 1){
                        silu_aie(y, y, y_slice);
                    }
                    else if (add_act == 2){
                        gelu_aie(y, y, y_slice);
                    }
                    _lock_release(y_cons_lock);
                }
            } // end m loop
        }// end n loop
    }
}