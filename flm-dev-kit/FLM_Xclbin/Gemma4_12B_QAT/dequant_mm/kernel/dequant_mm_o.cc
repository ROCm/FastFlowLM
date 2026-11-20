// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "aie_kernel_utils.h"
#include "q4_core.h"
#include "lut_based_ops.h"
#include "common.hpp"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int qw_prod_lock = 2;
constexpr int qw_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int sync_lock = 6;

AIE_PING_FLAG is_y_ping = false;
AIE_PING_FLAG is_qw_ping = false;
AIE_PING_FLAG is_x_ping = false;

// colA is the number of k-tiles consumed per call; ldB is the number of k-tiles
// per n-tile row of pW.  They are only equal when the whole k_slice is consumed
// in one call -- with the k_slice split into two 128-wide passes they are not.
// ldC is the same idea on the output side: the number of n-tiles per m-tile row
// of pY.  This call writes colB of them, but pY is a slice of a wider y_acc, so
// the row stride is ldC and not colB.
template <typename T_in, typename T_out, unsigned rowA, unsigned colA, unsigned colB, unsigned ldB, unsigned ldC, unsigned r, unsigned s, unsigned t>
void mm_aie(T_out *__restrict pY, const T_in *__restrict pX, const T_in *__restrict pW) {
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
#if defined(__chess__) || !defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
    using MMUL_C = MMUL;
#else
    using MMUL_C = aie_mmul_bf16_888;
#endif

    // both loops step two tiles at a time and neither has a remainder path
    static_assert(rowA % 2 == 0, "rowA must be even: the z loop emits two m-tiles per step");
    static_assert(colB % 2 == 0, "colB must be even: the j loop emits two n-tiles per step");
    static_assert(ldC >= colB, "pY row stride cannot be narrower than the slice written");

#if !defined(__chess__)
    // Peano: no peeled first k step, which makes Peano spill the X tiles, and
    // a fully unrolled j loop. Same macs in the same order as the chess branch.
    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pY1 = pY + (z * ldC) * MMUL::size_C;
        T_out *__restrict pY2 = pY + ((z + 1) * ldC) * MMUL::size_C;
        AIE_LOOP_UNROLL_FULL
        for (unsigned j = 0; j < colB; j += 2){
            const T_in *__restrict pX1 = pX + z * MMUL::size_A;
            const T_in *__restrict pX2 = pX + (z + 1) * MMUL::size_A;
            const T_in *__restrict pW1 = pW + j * ldB * MMUL::size_B;
            const T_in *__restrict pW2 = pW + (j + 1) * ldB * MMUL::size_B;
            MMUL_C Y00(aie::load_v<MMUL::size_C>(pY1));
            MMUL_C Y01(aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C));
            MMUL_C Y10(aie::load_v<MMUL::size_C>(pY2));
            MMUL_C Y11(aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C));
            for (unsigned i = 0; i < colA; ++i) {
                aie::vector<T_in, MMUL::size_A> X0 = aie::load_v<MMUL::size_A>(pX1);
                pX1 += MMUL::size_A * rowA;
                aie::vector<T_in, MMUL::size_A> X1 = aie::load_v<MMUL::size_A>(pX2);
                pX2 += MMUL::size_A * rowA;
                aie::vector<T_in, MMUL::size_B> W0 = aie::load_v<MMUL::size_B>(pW1);
                pW1 += MMUL::size_B;
                aie::vector<T_in, MMUL::size_B> W1 = aie::load_v<MMUL::size_B>(pW2);
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
        }
    }
#else
    for (unsigned z = 0; z < rowA; z += 2){
        T_out *__restrict pY1 = pY + (z * ldC) * MMUL::size_C;
        T_out *__restrict pY2 = pY + ((z + 1) * ldC) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 2){
            const T_in *__restrict pX1 = pX + z * MMUL::size_A;
            const T_in *__restrict pX2 = pX + (z + 1) * MMUL::size_A;
            const T_in *__restrict pW1 = pW + j * ldB * MMUL::size_B;
            const T_in *__restrict pW2 = pW + (j + 1) * ldB * MMUL::size_B;

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

            MMUL_C Y00(acc_Y00);
            MMUL_C Y01(acc_Y01);
            MMUL_C Y10(acc_Y10);
            MMUL_C Y11(acc_Y11);

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
#endif
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

// One call drains exactly one y_slice of y_acc into the bf16 buffer the DMA is
// about to ship, so the width has to track y_slice rather than repeat it: the
// activation helpers below are told y_slice explicitly, and a mismatch would leave
// part of the buffer stale instead of failing.
template <typename T_in, typename T_out>
void passThrough_aie(const T_in *restrict in, T_out *restrict out) {
    constexpr int vec_factor = y_slice;

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
        int8 * qw_ping,
        int8 * qw_pong,
        bf16 * w_buffer,
        float * y_acc,
        int * K_div_k,
        int * rtp_activate
    ) {
        aie::set_rounding(aie::rounding_mode::conv_even);

        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA
        int n_div_n_slice = NPU_N_TILE / n_slice; //colB

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                for (int p = 0; p < n_div_n_slice; p++){
                    _down_lock_acquire(qw_cons_lock, 1);
                    is_qw_ping = !is_qw_ping;
                    bf16* scale_buffer0 = is_qw_ping ? (bf16*) qw_ping : (bf16*) qw_pong;
                    int8_t * data_ptr0 = ((int8_t *)scale_buffer0) + QXNX_ROW_BLOCK_SIZE * 8 * sizeof(bf16); // after scales and mins
                    bf16 * scale_buffer1 = (bf16 *)(data_ptr0 + QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE / 2);
                    int8_t * data_ptr1 = ((int8_t *)scale_buffer1) + QXNX_ROW_BLOCK_SIZE * 8 * sizeof(bf16);
                    _Q4_0_bf16(
                        w_buffer,
                        data_ptr1,
                        scale_buffer1,
                        16
                    );

                    // Barrier A: my half of w_buffer is written, wait for the
                    // even core's half.  Release before acquiring so the two
                    // cores cannot deadlock on each other.
                    _down_lock_release(sync_lock, 1);
                    _lock_acquire(sync_lock, 1);
                    _down_lock_release(qw_prod_lock, 1);

                    for (int i = 0; i < m_div_m_slice; i++){
                        // v must be the inner loop: the memtile BD dimensions
                        // [(m/32, 16*k), (2, 128), (16, k), (128, 1)] iterate
                        // outermost first, so the two 128-wide halves of this
                        // k_slice arrive back to back for each 16-row group.
                        // Both accumulate into the same y_acc tile.
                        for (int v = 0; v < 2; v++){
                            _lock_acquire(x_cons_lock);
                            is_x_ping = !is_x_ping;
                            const bf16 *x = is_x_ping ? x_ping : x_pong;
                            mm_aie<bf16, float, m_slice / 8, 128 / 8, n_slice / 8, k_slice / 8, NPU_N_TILE / 8, 8, 8, 8>(y_acc + i * m_slice * NPU_N_TILE + p * n_slice * 8, x, w_buffer + v * 8 * 128);
                            _lock_release(x_prod_lock);
                        }
                    }
                    // Barrier B: done reading w_buffer, so the even core may
                    // overwrite rows 0-15 for the next slice.
                    _down_lock_release(sync_lock, 1);
                    _lock_acquire(sync_lock, 1);
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