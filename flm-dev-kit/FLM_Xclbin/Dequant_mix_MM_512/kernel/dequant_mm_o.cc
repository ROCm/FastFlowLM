// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_kernel_utils.h"
#include "q4_core.h"
#include "q8_core.h"
#include "lut_based_ops.h"
#include "common.hpp"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int qw_prod_lock = 2;
constexpr int qw_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int sync_lock = 6;

AIE_PERSIST_ZERO bool is_y_ping = false;
AIE_PERSIST_ZERO bool is_w_ping = false;
AIE_PERSIST_ZERO bool is_x_ping = false;

template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void mm_aie(T_out *__restrict pY, const T_in *__restrict pX, const T_in *__restrict pW) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pY1 = pY + (z * colB) * MMUL::size_C;
        T_out *__restrict pY2 = pY + ((z + 1) * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2){
            const T_in *__restrict pX1 = pX + z * MMUL::size_A;
            const T_in *__restrict pX2 = pX + (z + 1) * MMUL::size_A;
            const T_in *__restrict pW1 = pW + j * colA * MMUL::size_B;
            const T_in *__restrict pW2 = pW + (j + 1) * colA * MMUL::size_B;

            aie::vector<T_in, MMUL::size_A> X0 = aie::load_v<MMUL::size_A>(pX1);
            pX1 += MMUL::size_A * rowA;
            aie::vector<T_in, MMUL::size_A> X1 = aie::load_v<MMUL::size_A>(pX2);
            pX2 += MMUL::size_A * rowA;
            aie::vector<T_in, MMUL::size_B> W0 = aie::load_v<MMUL::size_B>(pW1);
            pW1 += MMUL::size_B;
            aie::vector<T_in, MMUL::size_B> W1 = aie::load_v<MMUL::size_B>(pW2);
            pW2 += MMUL::size_B;

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
            chess_prepare_for_pipelining chess_loop_range(colA - 1, colA - 1) {
                X0 = aie::load_v<MMUL::size_A>(pX1);
                pX1 += MMUL::size_A * rowA;
                X1 = aie::load_v<MMUL::size_A>(pX2);
                pX2 += MMUL::size_A * rowA;
                W0 = aie::load_v<MMUL::size_B>(pW1);
                pW1 += MMUL::size_B;
                W1 = aie::load_v<MMUL::size_B>(pW2);
                pW2 += MMUL::size_B;

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
        } // end for colB
    } // end for row
}

void silu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){

    bf16 *x_it = x;

    constexpr int vector_size = 64;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    for (int i = 0; i < F; i++){
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSiluBf16_nonLUT<vector_size>(x_vec);
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
        o_vec = getGeluBf16_nonLUT<vector_size>(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

template <typename T_in, typename T_out>
void passThrough_aie(const T_in *restrict in, T_out *restrict out) {
    const int vec_factor = 64;

    aie::vector<T_out, vec_factor> Out;
    aie::accum<accfloat, vec_factor> c_acc_in;
    c_acc_in.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, c_acc_in.template to_vector<T_out>());
}

extern "C" {

    void dequant_mm_o(
        bf16 * y_ping,
        bf16 * y_pong,
        bf16 * x_ping,
        bf16 * x_pong,
        uint8 * qw_ping,
        uint8 * qw_pong,
        uint8 * w8_buffer,
        bf16 * w_buffer_e,
        bf16 * w_buffer_o,
        bf16 * scale_buffer,
        bf16 * zp_buffer,
        float * y_acc,
        int * K_div_k,
        int * rtp_activate,
        int * rtp_q4,
        int * rtp_zp
    ) {
        // Both cores of the (even, odd) row pair dequantize their half of the
        // slices into the odd tile's buffer, rendezvous on sync_lock, then each
        // multiplies against the completed tile.  The even tile's own buffer is
        // unused for now.
        (void)w_buffer_e;
        bf16 * w_buffer = w_buffer_o;

        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
        aie::store_v(zp_buffer, aie::zeros<bf16, QXNX_ROW_BLOCK_SIZE>());
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA
        int n_div_n_slice = NPU_N_TILE / n_slice; //colB

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                for (int p = 0; p < n_div_n_slice; p++){
                    _lock_acquire(x_cons_lock);
                    if (rtp_q4[0] == 1){
                        _Q4_x_bf16<false>(
                            w_buffer,
                            scale_buffer,
                            zp_buffer,
                            qw_ping,
                            qw_pong,
                            w8_buffer,
                            is_w_ping,
                            qw_prod_lock,
                            qw_cons_lock,
                            rtp_zp[0]
                        );
                    }
                    else{
                        _Q8_x_bf16<false>(
                            w_buffer,
                            scale_buffer,
                            zp_buffer,
                            (int8 *)qw_ping,
                            (int8 *)qw_pong,
                            (int8 *)w8_buffer,
                            is_w_ping,
                            qw_prod_lock,
                            qw_cons_lock,
                            rtp_zp[0]
                        );
                    }

                    _lock_release(sync_lock);
                    _down_lock_acquire(sync_lock);

                    is_x_ping = !is_x_ping;
                    const bf16 *x = is_x_ping ? x_ping : x_pong;
                    mm_aie<bf16, float, m_slice / 8, k_slice / 8, n_slice / 8, 8, 8, 8>(y_acc + p * NPU_M_TILE * n_slice, x, w_buffer);
                    _lock_release(x_prod_lock);
                    
                    for (int i = 1; i < m_div_m_slice; i++){
                        _lock_acquire(x_cons_lock);
                        is_x_ping = !is_x_ping;
                        const bf16 *x = is_x_ping ? x_ping : x_pong;
                        mm_aie<bf16, float, m_slice / 8, k_slice / 8, n_slice / 8, 8, 8, 8>(y_acc + i * m_slice * n_slice + p * NPU_M_TILE * n_slice, x, w_buffer);
                        _lock_release(x_prod_lock);
                    }
                }
            }
            k += 1;
        } while (k < K_div_k[0]);
        int use_activation = rtp_activate[0];
        for (int i = 0; i < y_num; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            
            passThrough_aie<float, bf16>(y_acc + i * y_slice, y);
            if (use_activation == 1){
                silu_aie(y, y, y_slice);
            }
            else if (use_activation == 2){
                gelu_aie(y, y, y_slice);
            }
            _lock_release(y_cons_lock);
        }
    }
}