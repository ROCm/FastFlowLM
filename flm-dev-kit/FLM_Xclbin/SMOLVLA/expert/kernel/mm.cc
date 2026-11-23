// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include <math.h>
#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "common.h"

// Expert matmul with the weights packed as bfp16 (bfp16ebs8: 9 B per 8 consecutive k of one n row: exponent byte
// + 8 int8 mantissas). W reaches L1 as ready tiles, x arrives as bf16 [k8][16 rows][8 k] and is converted once per
// chunk (vision_mm_c does the same). DDR carries 9/16 of the weight bytes of the bf16 design.
//   w chunk (n_slice x k_slice): [n16 group][k8][B0 = n 0-7, B1 = n 8-15]
// Round to nearest even for every narrowing in the call (power-on default is floor).
struct KernelRneScope {
    aie::rounding_mode saved;
    inline __attribute__((always_inline)) KernelRneScope() { saved = aie::swap_rounding(aie::rounding_mode::conv_even); }
    inline __attribute__((always_inline)) ~KernelRneScope() { aie::set_rounding(saved); }
};
#define KERNEL_RNE_SCOPE KernelRneScope _kernel_rne_scope;

static constexpr int TILE_B   = 72;
static constexpr int K8       = k_slice / 8;
static constexpr int N_BLK    = n_slice / 8;
static_assert(N_BLK % 2 == 0, "n_slice must hold whole pairs of n8 blocks");
static_assert(m_slice == 16, "the multiply loop is written for 16-row x chunks");

static inline __attribute__((always_inline)) void convert_x(const bf16 *__restrict px, uint8 *__restrict xc) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> xo((bfp16ebs8 *)xc);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(K8, K8)
    for (int k = 0; k < K8; k++) {
        auto a0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(px)));
        auto a1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(px + 64)));
        px += 128;
        xo << a0;
        xo << a1;
    }
}

// one 16-row x chunk against every n pair of the w chunk; y_group is the chunk's 2 x N_BLK C blocks
static inline __attribute__((always_inline)) void mac_chunk(float *__restrict y_group, const uint8 *__restrict xc,
                                                            const uint8 *__restrict wc) {
    for (int p = 0; p < N_BLK / 2; p++) {
        float *y0 = y_group + (2 * p) * 64;
        float *y1 = y_group + (N_BLK + 2 * p) * 64;
        aie::accum<accfloat, 64> c00(aie::load_v<64>(y0)), c01(aie::load_v<64>(y0 + 64));
        aie::accum<accfloat, 64> c10(aie::load_v<64>(y1)), c11(aie::load_v<64>(y1 + 64));
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> xs((const bfp16ebs8 *)xc);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ws((const bfp16ebs8 *)(wc + p * K8 * 2 * TILE_B));
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(K8, K8)
        AIE_TRY_INITIATION_INTERVAL(4)
        for (int k = 0; k < K8; k++) {
            auto a0 = xs.pop();
            auto a1 = xs.pop();
            auto b0 = ws.pop();
            auto b1 = ws.pop();
            c00 = mac_8x8_8x8T(a0, b0, c00);
            c01 = mac_8x8_8x8T(a0, b1, c01);
            c10 = mac_8x8_8x8T(a1, b0, c10);
            c11 = mac_8x8_8x8T(a1, b1, c11);
        }
        aie::store_v(y0, c00.template to_vector<float>());
        aie::store_v(y0 + 64, c01.template to_vector<float>());
        aie::store_v(y1, c10.template to_vector<float>());
        aie::store_v(y1 + 64, c11.template to_vector<float>());
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
        o_vec = getSiluBf16(x_vec);
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
        const uint8 *__restrict w_ping,
        const uint8 *__restrict w_pong,
        bf16 *__restrict y_ping,
        bf16 *__restrict y_pong,
        float *__restrict y_acc,
        uint8 *__restrict xc,
        bf16 *__restrict g_keep,
        int * rtp_value

    ){
        KERNEL_RNE_SCOPE
        AIE_PERSIST_ZERO static bool is_w_ping = false;
        AIE_PERSIST_ZERO static bool is_y_ping = false;
        AIE_PERSIST_ZERO static bool is_x_ping = false;
        // act 3 = SwiGLU pair: the host streams the gate strip, then the up strip, of every n block through this
        // tile. The gate half keeps silu(gate) in g_keep and emits nothing; the up half emits bf16(g_keep * up).
        // Every fused launch runs whole gate/up pairs, so the toggle is back at "gate" when the launch ends.
        AIE_PERSIST_ZERO static bool is_up = false;
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA

        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);

        int k = 0; // IT to MT
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                _lock_acquire(w_cons_lock);
                is_w_ping = !is_w_ping;
                const uint8 *w = is_w_ping ? w_ping : w_pong;
                for (int i = 0; i < m_div_m_slice; i++){
                    _lock_acquire(x_cons_lock);
                    is_x_ping = !is_x_ping;
                    const bf16 *x = is_x_ping ? x_ping : x_pong;
                    convert_x(x, xc);
                    mac_chunk(y_acc + i * m_slice * n_slice, xc, w);
                    _lock_release(x_prod_lock);
                }
                _lock_release(w_prod_lock);
            }
            k += 1;
        } while (k < rtp_value[0]);

        if (rtp_value[1] == 3) {
            if (!is_up) {
                for (int i = 0; i < y_num; i++){
                    bf16 *g = g_keep + i * y_slice;
                    passThrough_aie<float, bf16, 64>(y_acc + i * y_slice, g);
                    silu_aie(g, g, y_slice);
                }
            } else {
                for (int i = 0; i < y_num; i++){
                    _lock_acquire(y_prod_lock);
                    is_y_ping = !is_y_ping;
                    bf16 *y = is_y_ping ? y_ping : y_pong;
                    passThrough_aie<float, bf16, 64>(y_acc + i * y_slice, y);
                    const bf16 *g = g_keep + i * y_slice;
                    for (int v = 0; v < y_slice; v += 32) {
                        aie::vector<bf16, 32> u = aie::load_v<32>(y + v);
                        aie::vector<bf16, 32> gv = aie::load_v<32>(g + v);
                        aie::store_v(y + v, aie::mul(u, gv).template to_vector<bf16>());
                    }
                    _lock_release(y_cons_lock);
                }
            }
            is_up = !is_up;
            return;
        }

        for (int i = 0; i < y_num; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            passThrough_aie<float, bf16, 64>(y_acc+i*y_slice, y);
            if (rtp_value[1] == 1){
                silu_aie(y, y, y_slice);
            }
            _lock_release(y_cons_lock);
        }
    }
}
