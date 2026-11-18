// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// PLE stage for 8 token rows (plan S6.8), gate tile: g = gelu(W_gate . h) * pli for 8 tokens.
//
// W_gate (256 x 2560, bf16) streams once per dispatch as 8 row blocks x 10 k blocks of [256 k][32 rows] (kernels/bf16_proj.h layout, the
// same bytes the decode design reads).  For every (row block, k block) the 8 tokens' x slices [8][256] arrive in a 3-buffer ring; each token
// runs the decode's _mvm_bf16_bf16<32,256> on its own accumulator row, so the per token arithmetic and accumulation order are those of the
// 1-token PLE gate.
//
// Per dispatch the x stream carries 81 transfers of [8][256] bf16: the host pli rows first (copied away at once), then 8 x 10 h slices
// ([rb][kb][t][256], h re-sent for every row block).  81 = 3 x 27 so the 3-buffer ring starts at buffer 0 in every dispatch.
//
//   locks (own tile): 0 x_prod (init 3)   1 x_cons    2 w_prod (init 2)   3 w_cons    4 u_prod (init 1)   5 u_cons
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "bf16_proj.h"
#include "ple8_mvm.h"
#include "lut_based_ops.h"

constexpr int PLE8_T = 8;
constexpr int PLE8_M = BF16_PROJ_M_BLOCK;   // 32 rows per block
constexpr int PLE8_K = BF16_PROJ_K_BLOCK;   // 256 columns per block
static_assert(PLI_D % PLE8_M == 0 && MODEL_DIM % PLE8_K == 0, "ple8 gate blocking");

extern "C" {

void ple8_gate(
    bf16 * x0, bf16 * x1, bf16 * x2,       // x slice ring, [8][256] each
    bf16 * w_ping, bf16 * w_pong,          // W_gate blocks, [256 k][32 rows]
    bf16 * pli,                            // host pli rows [8][256], kept for the whole dispatch
    bf16 * u                               // out: u = gelu(gate) * pli, [8][256] (also the gate accumulation target)
){
    constexpr int x_prod_lock = 0, x_cons_lock = 1, w_prod_lock = 2, w_cons_lock = 3, u_prod_lock = 4, u_cons_lock = 5;
    bf16* xr[3] = {x0, x1, x2};
    alignas(aie::vector_decl_align) float y_acc[PLE8_T * PLE8_M];

    _lock_acquire(u_prod_lock);

    // transfer 0 of the x stream: pli rows
    _lock_acquire(x_cons_lock);
    for (int i = 0; i < PLE8_T * PLI_D / 32; i++) {
        aie::store_v(pli + i * 32, aie::load_v<32>(x0 + i * 32));
    }
    _lock_release(x_prod_lock);

    int xi = 1;
    bool w_is_ping = false;
    for (int rb = 0; rb < PLI_D / PLE8_M; rb++) {
        zero_vectorized<float, PLE8_T * PLE8_M>(y_acc);
        for (int kb = 0; kb < MODEL_DIM / PLE8_K; kb++) {
            w_is_ping = !w_is_ping;
            bf16* w = w_is_ping ? w_ping : w_pong;
            bf16* x = xr[xi];
            xi = (xi == 2) ? 0 : xi + 1;
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
#if defined(PLE8_SKIP_COMPUTE)
#elif defined(PLE8_MVM_SERIAL)
            for (int t = 0; t < PLE8_T; t++) {
                _mvm_bf16_bf16<PLE8_M, PLE8_K>(w, x + t * PLE8_K, y_acc + t * PLE8_M);
            }
#else
            ple8_mvm_block<PLE8_K>(w, x, y_acc);
#endif
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        for (int t = 0; t < PLE8_T; t++) {
            aie::vector<float, PLE8_M> y_vec = aie::load_v<PLE8_M>(y_acc + t * PLE8_M);
            aie::accum<accfloat, PLE8_M> c_acc;
            c_acc.from_vector(y_vec);
            aie::store_v(u + t * PLI_D + rb * PLE8_M, c_acc.template to_vector<bf16>());
        }
    }
    // u = gelu(g) * pli, in place
    for (int t = 0; t < PLE8_T; t++) {
        for (int i = 0; i < PLI_D / 16; i++) {
            aie::vector<bf16, 16> g = aie::load_v<16>(u + t * PLI_D + i * 16);
            aie::vector<bf16, 16> a = getActivationBf16(g);
            aie::vector<bf16, 16> p = aie::load_v<16>(pli + t * PLI_D + i * 16);
            aie::vector<bf16, 16> o = aie::mul(a, p);
            aie::store_v(u + t * PLI_D + i * 16, o);
        }
    }
    _lock_release(u_cons_lock);
}

}
