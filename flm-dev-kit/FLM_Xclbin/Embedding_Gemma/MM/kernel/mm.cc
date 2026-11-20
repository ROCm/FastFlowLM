// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void mm_aie(T_out *__restrict pY, const T_in *__restrict pX, const T_in *__restrict pW) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    static_assert(r == 8 && s == 8 && t == 8, "W blocks are 8x8");
    using MMUL_C = mmul_wt_t<MMUL, T_in>;

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pY1 = pY + (z * colB) * MMUL::size_C;
        T_out *__restrict pY2 = pY + ((z + 1) * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
        chess_prepare_for_pipelining chess_loop_range(4, ) {
            const T_in *__restrict pX1 = pX + z * MMUL::size_A;
            const T_in *__restrict pX2 = pX + (z + 1) * MMUL::size_A;
            const T_in *__restrict pW1 = pW + j * MMUL::size_B;
            const T_in *__restrict pW2 = pW + (j + 1) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> X0 = aie::load_v<MMUL::size_A>(pX1);
            pX1 += MMUL::size_A * rowA;
            aie::vector<T_in, MMUL::size_A> X1 = aie::load_v<MMUL::size_A>(pX2);
            pX2 += MMUL::size_A * rowA;
            aie::vector<T_in, MMUL::size_B> W0 = aie::load_v<MMUL::size_B>(pW1);
            // aie::vector<T_in, MMUL::size_B> W0 = aie::transpose(W00, 8, 8);
            pW1 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> W1 = aie::load_v<MMUL::size_B>(pW2);
            // aie::vector<T_in, MMUL::size_B> W1 = aie::transpose(W10, 8, 8);
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
            chess_prepare_for_pipelining chess_loop_range(7, ) {
                X0 = aie::load_v<MMUL::size_A>(pX1);
                pX1 += MMUL::size_A * rowA;
                X1 = aie::load_v<MMUL::size_A>(pX2);
                pX2 += MMUL::size_A * rowA;
                W0 = aie::load_v<MMUL::size_B>(pW1);
                // W0 = aie::transpose(W00, 8, 8);
                pW1 += MMUL::size_B * colB;
                W1 = aie::load_v<MMUL::size_B>(pW2);
                // W1 = aie::transpose(W10, 8, 8);
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

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

extern "C" {
    void mm(
        const bf16 *__restrict x_ping,
        const bf16 *__restrict x_pong,
        const bf16 *__restrict w_ping,
        const bf16 *__restrict w_pong,
        bf16 *__restrict y_ping,
        bf16 *__restrict y_pong,
        float *__restrict y_acc,
        volatile int * K_div_k // K / k, written by the runtime (RTP) before each command

    ){
        // Must start false and keep their value across calls; see AIE_PERSIST_ZERO.
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_w_ping = false;
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_y_ping = false;
        AIE_PERSIST_ZERO static AIE_PING_FLAG is_x_ping = false;

        // The core comes out of reset rounding toward -inf. That biases every
        // conversion to bf16 or bfp16 (mmul operands, accumulator stores) low,
        // and on mostly-positive inputs (GELU, softmax) the bias adds up.
        aie::set_rounding(aie::rounding_mode::conv_even);

        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA
        int k = 0; // IT to MT
        do {

            for (int j = 0; j < k_div_k_slice; j++){
                _lock_acquire(w_cons_lock);
                is_w_ping = !is_w_ping;
                const bf16 *w = reinterpret_cast<const bf16 *> (is_w_ping ? w_ping : w_pong);
                for (int i = 0; i < m_div_m_slice; i++){
                    _lock_acquire(x_cons_lock);
                    is_x_ping = !is_x_ping;
                    const bf16 *x = is_x_ping ? x_ping : x_pong;
                    mm_aie<bf16, float, m_slice / 8, k_slice / 8, n_slice / 8, 8, 8, 8>(y_acc+i*m_slice*n_slice, x, w);
                    _lock_release(x_prod_lock);
                }
                _lock_release(w_prod_lock);
            }
            k += 1;
        } while (k < K_div_k[0]);

        for (int i = 0; i < y_num; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            passThrough_aie<float, bf16, y_slice>(y_acc+i*y_slice, y);
            _lock_release(y_cons_lock);
        }
    }
}