// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_kernel_utils.h"
#include "q4_core.h"
#include "lut_based_ops.h"
#include "common.hpp"
#include "glu_epilogue.h"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int qw_prod_lock = 2;
constexpr int qw_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int sync_lock = 6;

bool is_y_ping AIE_PERSIST_ZERO = false;
bool is_qw_ping AIE_PERSIST_ZERO = false;
bool is_x_ping AIE_PERSIST_ZERO = false;

template <unsigned rowA, unsigned colA, unsigned colB>
void mm_aie_bfp16(float *__restrict pY, const bfp16ebs8 *__restrict pX, const bfp16ebs8 *__restrict pW) {
    using ACC = aie::accum<accfloat, 64>;
    constexpr unsigned size_C = 64;
    static_assert(rowA == 2);

    float *__restrict pY1 = pY;
    float *__restrict pY2 = pY + colB * size_C;

    for (unsigned j = 0; j < colB; j += 2){
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sX(pX);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sW(pW);
        sW.seek(j * colA);

        ACC Y00(aie::load_v<size_C>(pY1));
        ACC Y01(aie::load_v<size_C>(pY1 + size_C));
        ACC Y10(aie::load_v<size_C>(pY2));
        ACC Y11(aie::load_v<size_C>(pY2 + size_C));

        for (unsigned i = 0; i < colA; ++i)
        chess_prepare_for_pipelining chess_loop_range(colA, colA) {
            aie::block_vector<bfp16ebs8, 64> X0 = sX.pop();
            aie::block_vector<bfp16ebs8, 64> X1 = sX.pop();
            aie::block_vector<bfp16ebs8, 64> W0 = sW.pop();
            aie::block_vector<bfp16ebs8, 64> W1 = sW.pop();

            Y00 = mac_8x8_8x8T(X0, W0, Y00);
            Y01 = mac_8x8_8x8T(X0, W1, Y01);
            Y10 = mac_8x8_8x8T(X1, W0, Y10);
            Y11 = mac_8x8_8x8T(X1, W1, Y11);
        }
        aie::store_v(pY1, Y00.template to_vector<float>());
        pY1 += size_C;
        aie::store_v(pY1, Y01.template to_vector<float>());
        pY1 += size_C;
        aie::store_v(pY2, Y10.template to_vector<float>());
        pY2 += size_C;
        aie::store_v(pY2, Y11.template to_vector<float>());
        pY2 += size_C;
    }
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

template <typename T_in, typename T_out>
void passThrough_aie(const T_in *restrict in, T_out *restrict out) {
    const int vec_factor = 64;

    aie::vector<T_out, vec_factor> Out;
    aie::accum<accfloat, vec_factor> c_acc_in;
    c_acc_in.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, c_acc_in.template to_vector<T_out>());
}

extern "C" {

    void dequant_mm_e(
        bf16 * y_ping,
        bf16 * y_pong,
        bf16 * x_ping,
        bf16 * x_pong,
        uint8 * qw_ping,
        uint8 * qw_pong,
        bf16 * w_buffer,
        float * y_acc,
        int * K_div_k,
        int * rtp_activate
    ) {
        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
        bfp16ebs8* w_bfp = (bfp16ebs8*)w_buffer;
        constexpr int k_div_k_slice = NPU_K_TILE / k_slice;
        constexpr int m_div_m_slice = NPU_M_TILE / m_slice;
        constexpr int n_div_n_slice = NPU_N_TILE / n_slice;

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                uint8* qw_blocks[n_div_n_slice];
                for (int p = 0; p < n_div_n_slice; p++){
                    if (p > 0){
                        _up_lock_release(sync_lock, 1);
                        _lock_acquire(sync_lock, 1);
                    }
                    _lock_acquire(qw_cons_lock, 1);
                    is_qw_ping = !is_qw_ping;
                    qw_blocks[p] = is_qw_ping ? qw_ping : qw_pong;
                }
                for (int v = 0; v < passes_per_k_slice; v++){
                    for (int p = 0; p < n_div_n_slice; p++){
                        _Q4_K_bfp16<0>(
                            w_bfp,
                            qw_blocks[p],
                            v * groups_per_pass,
                            groups_per_pass,
                            p * (n_slice / 8),
                            tiles_per_n_block
                        );
                    }
                    _up_lock_release(sync_lock, 1);
                    _lock_acquire(sync_lock, 1);
                    if (v == passes_per_k_slice - 1){
                        for (int p = 0; p < n_div_n_slice; p++){
                            _lock_release(qw_prod_lock, 1);
                        }
                    }

                    for (int i = 0; i < m_div_m_slice; i++){
                        _lock_acquire(x_cons_lock);
                        is_x_ping = !is_x_ping;
                        // x arrives from the host already as bfp16 tiles (16 rows x 256 k)
                        const bfp16ebs8 *x_bfp = (const bfp16ebs8*)(is_x_ping ? x_ping : x_pong);
                        mm_aie_bfp16<m_slice / 8, k_pass / 8, NPU_N_TILE / 8>(y_acc + i * m_slice * NPU_N_TILE, x_bfp, w_bfp);
                        _lock_release(x_prod_lock);
                    }

                    _up_lock_release(sync_lock, 1);
                    _lock_acquire(sync_lock, 1);
                }
            }
            k += 1;
        } while (k < K_div_k[0]);
        int use_activation = rtp_activate[0];
        alignas(64) uint8 glu_group0[glu_group_bytes];
        glu_stream glu_out = glu_stream_begin(glu_group0);
        if (use_activation == act_glu){
            glu_tiles(y_acc, glu_group0);
        }
        for (int i = 0; i < y_num; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            
            if (use_activation == act_glu){
                glu_slot_chunk(y, y_acc, glu_out);
            }
            else {
                for (int t = 0; t < y_slice / 64; t++){
                    passThrough_aie<float, bf16>(y_acc + i * y_slice + t * 64, y + t * 64);
                }
                if (use_activation == 1){
                    silu_aie(y, y, y_slice);
                }
            }
            _lock_release(y_cons_lock);
        }
    }
}