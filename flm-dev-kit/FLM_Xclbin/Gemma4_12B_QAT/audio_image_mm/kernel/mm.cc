// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include <math.h>
#include "aie_kernel_utils.h"
#include "typedef.hpp"

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void mm_aie(T_out *__restrict pY, const T_in *__restrict pX, const T_in *__restrict pW) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
#if defined(__chess__) || !defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
    using MMUL_C = MMUL;
#else
    using MMUL_C = aie_mmul_bf16_888;
#endif

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
            aie::vector<T_in, MMUL::size_B> W0 = aie::transpose(W00, 8, 8);
            pW1 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> W10 = aie::load_v<MMUL::size_B>(pW2);
            aie::vector<T_in, MMUL::size_B> W1 = aie::transpose(W10, 8, 8);
            pW2 += MMUL::size_B * colB;

            aie::vector<T_out, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1); 
            aie::vector<T_out, MMUL::size_C> acc_Y01 = aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);
            aie::vector<T_out, MMUL::size_C> acc_Y11 = aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C);

            MMUL_C Y00(acc_Y00);
            MMUL_C Y01(acc_Y01);
            MMUL_C Y10(acc_Y10);
            MMUL_C Y11(acc_Y11);

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
                W0 = aie::transpose(W00, 8, 8);
                pW1 += MMUL::size_B * colB;
                W10 = aie::load_v<MMUL::size_B>(pW2);
                W1 = aie::transpose(W10, 8, 8);
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

template <typename T_in, typename T_out>
void passThrough_aie(T_in *restrict in0, T_out *restrict out) {
    const int vec_factor = 64;

    aie::vector<T_out, vec_factor> Out;
    aie::accum<accfloat, vec_factor> c_acc_in;
    c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
    aie::store_v(out, c_acc_in.template to_vector<T_out>());
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
        bf16 *__restrict x_ping,
        bf16 *__restrict x_pong,
        const bf16 *__restrict w_ping,
        const bf16 *__restrict w_pong,
        bf16 *__restrict y_ping,
        bf16 *__restrict y_pong,
        float *__restrict y_acc
    ){
        static AIE_PING_FLAG is_w_ping = false;
        static AIE_PING_FLAG is_y_ping = false;
        static AIE_PING_FLAG is_x_ping = false;

        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int n_div_n_slice = NPU_N_TILE / n_slice; //rowB
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA
        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);

        for (int j = 0; j < k_div_k_slice; j++){
            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            const bf16 *w = is_w_ping ? w_ping : w_pong;
            for (int i = 0; i < m_div_m_slice; i++){
                _lock_acquire(x_cons_lock);
                is_x_ping = !is_x_ping;
                const bf16 *x = is_x_ping ? x_ping : x_pong;
                mm_aie<bf16, float, m_slice / 8, k_slice / 8, n_slice / 8, 8, 8, 8>(y_acc + i * m_slice * n_slice, x, w);
                _lock_release(x_prod_lock);
            }
            _lock_release(w_prod_lock);
        }

        for (int i = 0; i < y_num; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            passThrough_aie<float, bf16>(y_acc+i*y_slice, y);
            _lock_release(y_cons_lock);
        }
    }
}