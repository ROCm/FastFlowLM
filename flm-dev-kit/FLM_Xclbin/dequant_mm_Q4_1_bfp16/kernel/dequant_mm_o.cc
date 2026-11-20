// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "aie_kernel_utils.h"
#include "q4_core.h"
#include "lut_based_ops.h"
#include "common.hpp"
#include "mm_bfp.h"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int qw_prod_lock = 2;
constexpr int qw_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int sync_lock = 6;
// tiles of one 32 x 256 n slice of w_buffer: 4 n tiles x 32 k tiles
constexpr int W_SLICE_TILES = (n_slice / 8) * (k_slice / 8);

bool is_y_ping = false;
bool is_qw_ping = false;
bool is_x_ping = false;

void silu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){

    bf16 *x_it = x;

    constexpr int vector_size = 64;

    aie::vector<bf16, vector_size> x_vec;
    aie::vector<bf16, vector_size> o_vec;

    const int F = D / vector_size;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < F; i++){
        x_vec = aie::load_v<vector_size>(x_it);
        o_vec = getSiluBf16_nonLUT<vector_size>(x_vec);
        aie::store_v(o, o_vec);
        x_it += vector_size;
        o += vector_size;
    }
}

void gelu_aie(bf16 *__restrict x, bf16 *__restrict o, int D){
    constexpr int vector_size = 64;
    // two independent vectors per step: the activation is a long dependent chain, this lets the compiler overlap two of them
    const int F = D / (2 * vector_size);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < F; i++){
        aie::vector<bf16, vector_size> a = aie::load_v<vector_size>(x);
        aie::vector<bf16, vector_size> b = aie::load_v<vector_size>(x + vector_size);
        aie::vector<bf16, vector_size> ga = getGeluBf16_nonLUT<vector_size>(a);
        aie::vector<bf16, vector_size> gb = getGeluBf16_nonLUT<vector_size>(b);
        aie::store_v(o, ga);
        aie::store_v(o + vector_size, gb);
        x += 2 * vector_size;
        o += 2 * vector_size;
    }
}

/// converts `count` fp32 accumulators to bf16 (count is a multiple of 64)
template <typename T_in, typename T_out>
void passThrough_aie(const T_in *restrict in, T_out *restrict out, int count) {
    const int vec_factor = 64;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < count / vec_factor; i++) {
        aie::accum<accfloat, vec_factor> c_acc_in;
        c_acc_in.from_vector(aie::load_v<vec_factor>(in + i * vec_factor));
        aie::store_v(out + i * vec_factor, c_acc_in.template to_vector<T_out>());
    }
}

extern "C" {

    void dequant_mm_o(
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
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA
        int n_div_n_slice = NPU_N_TILE / n_slice; //colB

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                // Both n slices of this k slice are dequantized first, each into its own half of w_buffer, so that
                // every X tile that arrives can be multiplied against both: X is the stream that limits the core.
                for (int p = 0; p < n_div_n_slice; p++){
                    _down_lock_acquire(qw_cons_lock, 1);
                    is_qw_ping = !is_qw_ping;
                    uint8* qw_using = is_qw_ping ? qw_ping : qw_pong;
                    _Q4_1_bfp16(
                        (bfp16ebs8*)w_buffer,
                        qw_using,
                        PARALLEL_ROWS,
                        p * W_SLICE_TILES
                    );
                }

                // Barrier A: my halves of w_buffer are written, wait for the partner core's halves.  Release
                // before acquiring so the two cores cannot deadlock on each other.
                _down_lock_release(sync_lock, 1);
                _lock_acquire(sync_lock, 1);
                for (int p = 0; p < n_div_n_slice; p++){
                    _down_lock_release(qw_prod_lock, 1);
                }

                for (int i = 0; i < m_div_m_slice; i++){
                    // v must be the inner loop: the memtile BD dimensions [(m/16, 16*k), (2, 128), (16, k), (128, 1)]
                    // iterate outermost first, so the two 128-wide halves of this k_slice arrive back to back for
                    // each 16-row group.  Both accumulate into the same y_acc tile.
                    for (int v = 0; v < 2; v++){
                        _lock_acquire(x_cons_lock);
                        is_x_ping = !is_x_ping;
                        const bf16 *x = is_x_ping ? x_ping : x_pong;
                        for (int p = 0; p < n_div_n_slice; p++){
                            mm_bfp_aie<m_slice / 8, 128 / 8, n_slice / 8, k_slice / 8>(
                                y_acc + i * m_slice * NPU_N_TILE + p * n_slice * 8, (const bfp16ebs8*)x,
                                (const bfp16ebs8*)w_buffer, p * W_SLICE_TILES + v * 32);
                        }
                        _lock_release(x_prod_lock);
                    }
                }
                // Barrier B: done reading w_buffer, so the partner may overwrite its rows for the next k slice.
                _down_lock_release(sync_lock, 1);
                _lock_acquire(sync_lock, 1);
            }
            k += 1;
        } while (k < K_div_k[0]);
        int use_activation = rtp_activate[0];
        for (int i = 0; i < y_num; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            
            passThrough_aie<float, bf16>(y_acc + i * y_slice, y, y_slice);
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