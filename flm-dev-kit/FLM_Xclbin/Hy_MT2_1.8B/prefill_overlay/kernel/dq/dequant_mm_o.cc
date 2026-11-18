// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

bool is_y_ping = false;
bool is_qw_ping = false;

// One 8 x 8 B tile (k major in the w tile) as bfp16: the transpose and conversion the
// emulated bf16 mmul does for its B operand (aie_api mmul_bf16_bf16.hpp), so the
// products are bit-identical to the bf16 path.
static inline __attribute__((always_inline)) aie::block_vector<bfp16ebs8, 64> b_tile_bfp16(
    aie::vector<bf16, 64> tile
) {
    aie::accum<accfloat, 64> acc(aie::transpose(tile, 8, 8));
    return acc.template to_vector<bfp16ebs8>();
}

// x arrives as bfp16 tiles the host made, [k / 8][2 row blocks][72 B], and is read with
// one block stream (its pops are the fast way to fill a bfp16 register); w is the bf16
// tile from _Q4_K_bf16. Same tiles, same order of the four accumulators as before.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned ldB, unsigned ldC>
void mm_aie(float *__restrict pY, const bfp16ebs8 *__restrict pX, const bf16 *__restrict pW) {
    using MMUL = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8, accauto>;

    static_assert(rowA == 2, "one x stream pops the two 8-row blocks of each k step");
    static_assert(colB % 2 == 0, "colB must be even: the j loop emits two n-tiles per step");
    static_assert(ldC >= colB, "pY row stride cannot be narrower than the slice written");

    float *__restrict pY1 = pY;
    float *__restrict pY2 = pY + ldC * MMUL::size_C;

    for (unsigned j = 0; j < colB; j += 2){
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sX(pX);
        const bf16 *__restrict pW1 = pW + j * ldB * 64;
        const bf16 *__restrict pW2 = pW + (j + 1) * ldB * 64;

        MMUL Y00(aie::load_v<MMUL::size_C>(pY1));
        MMUL Y01(aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C));
        MMUL Y10(aie::load_v<MMUL::size_C>(pY2));
        MMUL Y11(aie::load_v<MMUL::size_C>(pY2 + MMUL::size_C));

        for (unsigned i = 0; i < colA; ++i)
        chess_prepare_for_pipelining chess_loop_range(colA, colA) {
            aie::block_vector<bfp16ebs8, 64> X0 = sX.pop();
            aie::block_vector<bfp16ebs8, 64> X1 = sX.pop();
            aie::block_vector<bfp16ebs8, 64> W0 = b_tile_bfp16(aie::load_v<64>(pW1));
            pW1 += 64;
            aie::block_vector<bfp16ebs8, 64> W1 = b_tile_bfp16(aie::load_v<64>(pW2));
            pW2 += 64;

            // the instruction is A B^T (mac_8x8_8x8T); W0 / W1 already are B^T tiles
            Y00.mac(X0, aie::op_transpose(W0));
            Y01.mac(X0, aie::op_transpose(W1));
            Y10.mac(X1, aie::op_transpose(W0));
            Y11.mac(X1, aie::op_transpose(W1));
        }
        aie::store_v(pY1, Y00.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY1, Y01.template to_vector<float>());
        pY1 += MMUL::size_C;
        aie::store_v(pY2, Y10.template to_vector<float>());
        pY2 += MMUL::size_C;
        aie::store_v(pY2, Y11.template to_vector<float>());
        pY2 += MMUL::size_C;
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
        uint8 * qw_ping,
        uint8 * qw_pong,
        bf16 * w_buffer,
        float * y_acc,
        int * K_div_k,
        int * rtp_activate,
        int * rtp_m16
    ) {
        // The core comes out of reset rounding toward -inf, which biases every
        // accumulator -> bf16 conversion (dequant, mmul operands, passThrough) low.
        aie::set_rounding(aie::rounding_mode::conv_even);
        zero_vectorized<float, NPU_M_TILE * NPU_N_TILE>(y_acc);
        int k_div_k_slice = NPU_K_TILE / k_slice; //colA
        // m16: 16 tokens per core (the 32- and 64-row plans), x0 / x1 then hold
        // consecutive k slices instead of the two token halves of one k slice
        const bool m16 = rtp_m16[0] != 0;
        bool x_flip = false;
        int n_div_n_slice = NPU_N_TILE / n_slice; //colB

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                for (int p = 0; p < n_div_n_slice; p++){
                    _down_lock_acquire(qw_cons_lock, 1);
                    is_qw_ping = !is_qw_ping;
                    uint8* qw = is_qw_ping ? qw_ping : qw_pong;
                    // one 32x256 q4_k block; this core owns input columns 128-255 of it, core `e` takes 0-127
                    _Q4_K_bf16(w_buffer, qw, 1);

                    _down_lock_release(sync_lock, 1);
                    _lock_acquire(sync_lock, 1);
                    _down_lock_release(qw_prod_lock, 1);

                    // one x buffer (16 tokens, bfp16 tiles) is a whole k_slice wide, so a single mm_aie call
                    // sweeps the full depth of the dequantised tile, and is replayed
                    // for every weight band p: only p == 0 waits on the DMA. The
                    // second token half is written out (not a loop over a runtime
                    // count) so the 32-token path compiles as before.
                    if (p == 0) _lock_acquire(x_cons_lock, 1);
                    mm_aie<m_slice / 8, k_slice / 8, n_slice / 8, k_slice / 8, NPU_N_TILE / 8>(y_acc + p * n_slice * 8, (const bfp16ebs8 *)(x_flip ? x_pong : x_ping), w_buffer);
                    if (!m16) {
                        if (p == 0) _lock_acquire(x_cons_lock, 1);
                        mm_aie<m_slice / 8, k_slice / 8, n_slice / 8, k_slice / 8, NPU_N_TILE / 8>(y_acc + m_slice * NPU_N_TILE + p * n_slice * 8, (const bfp16ebs8 *)x_pong, w_buffer);
                    }
                    _down_lock_release(sync_lock, 1);
                    _lock_acquire(sync_lock, 1);
                }
                // the last band has now read both halves, so hand them back together
                _lock_release(x_prod_lock, m16 ? 1 : 2);
                if (m16) x_flip = !x_flip;
            }
            k += 1;
        } while (k < K_div_k[0]);
        int use_activation = rtp_activate[0];
        const int y_sends = m16 ? y_num / 2 : y_num;
        for (int i = 0; i < y_sends; i++){
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