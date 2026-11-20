// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <aie_api/aie.hpp>
#include <math.h>
#include "lut_based_ops.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

// Core rounding mode: its power-on default is floor (toward minus infinity), which biases every narrowing of an accumulator
// or fp32 value to bf16 / bfp16 downwards. Select round-to-nearest-even for the duration of the call, restore on return.
struct KernelRneScope {
    aie::rounding_mode saved;
    inline __attribute__((always_inline)) KernelRneScope() { saved = aie::swap_rounding(aie::rounding_mode::conv_even); }
    inline __attribute__((always_inline)) ~KernelRneScope() { aie::set_rounding(saved); }
};
#define KERNEL_RNE_SCOPE KernelRneScope _kernel_rne_scope;

// ---- bfp16 fast path ---------------------------------------------------------
// The operands reach L1 in the layouts the old kernel's MAC loop used: x as
// [k8][16 rows][8 k] and w as [k8][n][8 k] (bf16), so every 64-element load is
// already an 8x8 A tile (rows x 8 k) or a B tile in the "T" form (n x 8 k).
// Each arriving chunk is converted ONCE into bfp16 tiles (9 B per 8 values:
// exponent + 8 mantissas), then the multiply loop streams those tiles with no
// per-use conversion and no transposes. The host pre-rounds both operands onto
// the bfp16 grid, so the conversion is exact and the products are the ones the
// emulated bf16 mmul formed: same accumulation order, bit-identical results.
static constexpr int TILE_B   = 72;
static constexpr int K8       = k_slice / 8;
static constexpr int N_BLK    = n_slice / 8;
static constexpr int XC_BYTES = K8 * 2 * TILE_B;        // [k8][A0, A1]
static constexpr int WC_BYTES = N_BLK * K8 * TILE_B;    // [n pair][k8][B0, B1] (N_BLK even)
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

static inline __attribute__((always_inline)) void convert_w(const bf16 *__restrict pw, uint8 *__restrict wc) {
    for (int p = 0; p < N_BLK / 2; p++) {
        aie::block_vector_output_buffer_stream<bfp16ebs8, 64> wo((bfp16ebs8 *)(wc + p * K8 * 2 * TILE_B));
        const bf16 *q = pw + p * 128;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(K8, K8)
        for (int k = 0; k < K8; k++) {
            auto b0 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q)));
            auto b1 = ::to_v64bfp16ebs8(aie::accum<accfloat, 64>(aie::load_v<64>(q + 64)));
            q += n_slice * 8;
            wo << b0;
            wo << b1;
        }
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

/// Gather the part of the bias that arrives in w chunk `chunk`.
///
/// The host puts the N bias values in front of the weights, so in the memtile's
/// (n_slice, k) buffer they are row 0, columns 0..n_slice-1. The memtile forwards
/// that buffer to the core one k_slice-wide column band at a time:
///     [(k/k_slice, k_slice), (n_slice, k), (k_slice, 1)]
/// so chunk j carries row-0 columns [j*k_slice, (j+1)*k_slice). Bias element c
/// therefore rides chunk c/k_slice - **not always chunk 0**. When n_slice is
/// wider than k_slice the bias is split across several chunks, which is the case
/// for design B (n_slice 128 > k_slice 64) and not for design A (48 <= 64).
///
/// Within its chunk, element c is at stream position s = c - chunk*k_slice
/// (row 0, so no n_slice term), and the core's S2MM BD
///     [(n_slice, 8), (k_slice/8, n_slice*8), (8, 1)]
/// scatters stream position s to (s/8)*(n_slice*8) + s%8. Eight consecutive c
/// share a destination run, hence the vector gather.
///
/// Getting this wrong is quiet: the weights are still correct, so the result
/// stays well correlated and only the bias term is damaged.
void passbias_aie(float * bias, const bf16 * w, int chunk){
    const int c_begin = chunk * k_slice;
    if (c_begin >= n_slice) return;
    int c_end = c_begin + k_slice;
    if (c_end > n_slice) c_end = n_slice;

    for (int c = c_begin; c < c_end; c += 8){
        const int s = c - c_begin;
        aie::vector<bf16, 8> b_vec = aie::load_v<8>(w + (s / 8) * n_slice * 8 + (s % 8));
        aie::accum<accfloat, 8> acc;
        acc.from_vector(b_vec);
        aie::store_v(bias + c, acc.template to_vector<float>());
    }
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
        uint8 *__restrict xc,
        uint8 *__restrict wc,
        int * rtp_value

    ){
        KERNEL_RNE_SCOPE
#if defined(__chess__)
        static bool is_w_ping = false;
        static bool is_y_ping = false;
        static bool is_x_ping = false;
#else
        // Each call flips every selector an even number of times, so all three are false
        // on entry. Locals under Peano, so the kernel needs no data memory of its own.
        static_assert((NPU_K_TILE / k_slice) % 2 == 0 && y_num % 2 == 0,
                      "ping/pong selectors must return to false at the end of a call");
        bool is_w_ping = false;
        bool is_y_ping = false;
        bool is_x_ping = false;
#endif
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        int n_div_n_slice = NPU_N_TILE / n_slice; //rowB
        int m_div_m_slice = NPU_M_TILE / m_slice; //rowA

        // Bias prologue. The bias occupies row 0 of the (n_slice, k) weight block,
        // and the memtile hands that row over one k_slice-wide band per chunk, so
        // when n_slice > k_slice the bias is spread over several chunks and every
        // one of them has to be read. passbias_aie() returns immediately for the
        // chunks that carry no bias, so this also drains the remaining chunks.
        for (int j = 0; j < k_div_k_slice; j++){
            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            const bf16 *w = is_w_ping ? w_ping : w_pong;
            passbias_aie(bias, w, j);
            _lock_release(w_prod_lock);
        }

        for (int m = 0; m < rtp_value[0]; m++){
            // Seed the accumulator with the bias. y_acc is in mmul C-operand order,
            // so the 8x8 block (ym, nb) occupies ym*8*N + nb*64 and element (in, j)
            // of it sits at +in*8 + j. Bias column nb*8+j therefore replicates down
            // the 8 rows of that block. The old code unrolled nb 0..7, which assumed
            // NPU_N_TILE == 64; both vision designs have a different N.
            for (int ym = 0; ym < NPU_M_TILE / 8; ym++){
                for (int in = 0; in < 8; in++){
                    for (int nb = 0; nb < NPU_N_TILE / 8; nb++){
                        passThrough_aie<float, float, 8>(
                            bias + nb * 8,
                            y_acc + ym * 8 * NPU_N_TILE + in * 8 + nb * 64);
                    }
                }
            }

            int k = 0; // IT to MT
            do {
                for (int j = 0; j < k_div_k_slice; j++){
                    _lock_acquire(w_cons_lock);
                    is_w_ping = !is_w_ping;
                    const bf16 *w = is_w_ping ? w_ping : w_pong;
                    convert_w(w, wc);
                    _lock_release(w_prod_lock);
                    for (int i = 0; i < m_div_m_slice; i++){
                        _lock_acquire(x_cons_lock);
                        is_x_ping = !is_x_ping;
                        const bf16 *x = is_x_ping ? x_ping : x_pong;
                        convert_x(x, xc);
                        _lock_release(x_prod_lock);
                        mac_chunk(y_acc + i * m_slice * n_slice, xc, wc);
                    }
                }
                k += 1;
            } while (k < rtp_value[1]);

            for (int i = 0; i < y_num; i++){
                _lock_acquire(y_prod_lock);
                is_y_ping = !is_y_ping;
                bf16 *y = is_y_ping ? y_ping : y_pong;
                passThrough_aie<float, bf16, 64>(y_acc+i*y_slice, y);
                if (rtp_value[2] == 1){
                    silu_aie(y, y, y_slice);
                }
                else if (rtp_value[2] == 2){
                    gelu_aie(y, y, y_slice);
                }
                _lock_release(y_cons_lock);
            }
        } // end m loop
    }
}