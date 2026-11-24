// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include "common.h"

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void MM_aie(const T_in *__restrict pA, const T_in *__restrict pB, T_out *__restrict pC) {

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        T_out *__restrict pC1 = pC + (z + 0) * colB * MMUL::size_C;
        T_out *__restrict pC2 = pC + (z + 1) * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2)
        chess_flatten_loop{
            const T_in *__restrict pA1 = pA + (z + 0) * colA * MMUL::size_A;
            const T_in *__restrict pA2 = pA + (z + 1) * colA * MMUL::size_A;
            const T_in *__restrict pB1 = pB + (j + 0) * MMUL::size_B;
            const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B * colB;

            B0 = aie::transpose(B0, 8, 8);
            B1 = aie::transpose(B1, 8, 8);

            aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC1);
            aie::vector<T_out, MMUL::size_C> acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
            aie::vector<T_out, MMUL::size_C> acc_C10 = aie::load_v<MMUL::size_C>(pC2);
            aie::vector<T_out, MMUL::size_C> acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

            MMUL C00(acc_C00);
            MMUL C01(acc_C01);
            MMUL C10(acc_C10);
            MMUL C11(acc_C11);

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);

            for (unsigned i = 1; i < colA; ++i)
            chess_flatten_loop{
                A0 = aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                A1 = aie::load_v<MMUL::size_A>(pA2);
                pA2 += MMUL::size_A;
                B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;
                B1 = aie::load_v<MMUL::size_B>(pB2);
                pB2 += MMUL::size_B * colB;

                B0 = aie::transpose(B0, 8, 8);
                B1 = aie::transpose(B1, 8, 8);

                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
            }

            aie::store_v(pC1, C00.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC1, C01.template to_vector<T_out>());
            pC1 += MMUL::size_C;
            aie::store_v(pC2, C10.template to_vector<T_out>());
            pC2 += MMUL::size_C;
            aie::store_v(pC2, C11.template to_vector<T_out>());
            pC2 += MMUL::size_C;
        }
    }

    event1();
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

extern "C" {
    void mm(
        const bf16 * x_ping,
        const bf16 * x_pong,
        const bf16 * w_ping,
        const bf16 * w_pong,
        float * y_acc,
        bf16 * y,
        int* K_div_k
    ){
#if defined(__chess__)
        static bool is_wx_ping = false;
#else
        // The buffers can fill this core's data memory, so under Peano the selector is kept
        // in the last two words of y_acc (core-private, read here before it is zeroed and
        // rewritten after copy_y). The marker word tells the first call after load apart.
        constexpr float sel_marker = -0x1.5a5a5ap+77f;
        float *sel = y_acc + NPU_M_TILE * NPU_N_TILE - 2;
        bool is_wx_ping = sel[0] == sel_marker && sel[1] != 0.0f;
#endif

        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
        int k = 0;
        do {
            _lock_acquire(w_cons_lock); 
            _lock_acquire(x_cons_lock);
            is_wx_ping = !is_wx_ping;
            const bf16 *w = reinterpret_cast<const bf16 *>(is_wx_ping ? w_ping : w_pong);
            const bf16 *x = reinterpret_cast<const bf16 *>(is_wx_ping ? x_ping : x_pong);
            MM_aie<bf16, float, NPU_M_TILE/8, NPU_K_TILE/8, NPU_N_TILE/8, 8, 8, 8>(x, w, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            k += 1;
        } while (k < K_div_k[0]);
        
        _lock_acquire(y_prod_lock);
        copy_y<NPU_M_TILE * NPU_N_TILE>(y, y_acc);
        _lock_release(y_cons_lock);
#if !defined(__chess__)
        sel[0] = sel_marker;
        sel[1] = is_wx_ping ? 1.0f : 0.0f;
#endif
    }
}