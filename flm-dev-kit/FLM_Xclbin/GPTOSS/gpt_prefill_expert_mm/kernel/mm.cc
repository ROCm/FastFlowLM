// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include <math.h>

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void mm_aie(T_out *__restrict pY, const T_in *__restrict pX, const T_in *__restrict pW) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pY1 = pY + (z * colB) * MMUL::size_C;
        T_out *__restrict pY2 = pY + ((z + 1) * colB) * MMUL::size_C;
        const T_in *__restrict pX1 = pX + z * MMUL::size_A;
        const T_in *__restrict pX2 = pX + (z + 1) * MMUL::size_A;

        const T_in *__restrict pW1 = pW;

        aie::vector<T_in, MMUL::size_A> X0 = aie::load_v<MMUL::size_A>(pX1);
        pX1 += MMUL::size_A * rowA;
        aie::vector<T_in, MMUL::size_A> X1 = aie::load_v<MMUL::size_A>(pX2);
        pX2 += MMUL::size_A * rowA;
        aie::vector<T_in, MMUL::size_B> W00 = aie::load_v<MMUL::size_B>(pW1);
        aie::vector<T_in, MMUL::size_B> W0 = aie::transpose(W00, 4, 8);
        pW1 += MMUL::size_B * colB;

        aie::vector<T_out, MMUL::size_C> acc_Y00 = aie::load_v<MMUL::size_C>(pY1); 
        aie::vector<T_out, MMUL::size_C> acc_Y10 = aie::load_v<MMUL::size_C>(pY2);

        MMUL Y00(acc_Y00);
        MMUL Y10(acc_Y10);

        Y00.mac(X0, W0);
        Y10.mac(X1, W0);

        for (unsigned i = 1; i < colA; ++i)
        {
            X0 = aie::load_v<MMUL::size_A>(pX1);
            pX1 += MMUL::size_A * rowA;
            X1 = aie::load_v<MMUL::size_A>(pX2);
            pX2 += MMUL::size_A * rowA;
            W00 = aie::load_v<MMUL::size_B>(pW1);
            W0 = aie::transpose(W00, 4, 8);
            pW1 += MMUL::size_B * colB;

            Y00.mac(X0, W0);
            Y10.mac(X1, W0);
        }
        
        aie::store_v(pY1, Y00.template to_vector<T_out>());
        pY1 += MMUL::size_C;
        aie::store_v(pY2, Y10.template to_vector<T_out>());
        pY2 += MMUL::size_C;
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
        int * K_div_k // K is the size in DDR, k is 256 (MT)

    ){
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        AIE_PERSIST_ZERO static bool is_w_ping = false;
        AIE_PERSIST_ZERO static bool is_y_ping = false;

        zero_vectorized<float, m * n>(y_acc);
        int k_div_k_slice = k / k_slice; //colA
        int m_div_m_slice = m / m_slice; //rowA
        int k = 0; // IT to MT
        do {

            for (int j = 0; j < k_div_k_slice; j++){
                _lock_acquire(w_cons_lock);
                is_w_ping = !is_w_ping;
                const bf16 *w = reinterpret_cast<const bf16 *> (is_w_ping ? w_ping : w_pong);
                for (int i = 0; i < m_div_m_slice; i++){
                    _lock_acquire(x_cons_lock);
                    is_x_ping = !is_x_ping;
                    const bf16 *x = reinterpret_cast<const bf16 *> (is_x_ping ? x_ping : x_pong);
                    mm_aie<bf16, float, m_slice / 8, k_slice / 8, n_slice / 4, 8, 8, 4>(y_acc+i*m_slice*n_slice, x, w);
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