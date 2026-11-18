// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// PLE stage for 8 token rows (plan S6.8), up tile: up = W_up . u for 8 tokens, streamed as 80 row blocks.
//
// W_up (2560 x 256, bf16) is 80 row blocks of one [256 k][32 rows] block each; u = gelu(gate) * pli arrives once as [8][256].  Every block
// gives y[8][32] (one 32-row slice of the 8 tokens' up vectors), sent on at once as the 512 byte packet [t][32]; the memtile stage
// collects the 80 packets as [rb][t][32] and hands them to the norm tile token-major.
//
//   locks (own tile): 0 u_prod (init 1)   1 u_cons   2 w_prod (init 2)   3 w_cons   4 y_prod (init 2)   5 y_cons
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"
#include "bf16_proj.h"
#include "ple8_mvm.h"

constexpr int PLE8_T = 8;
constexpr int PLE8_M = BF16_PROJ_M_BLOCK;
constexpr int PLE8_K = BF16_PROJ_K_BLOCK;
static_assert(PLI_D == PLE8_K, "W_up has a single k block");
static_assert(MODEL_DIM % PLE8_M == 0 && (MODEL_DIM / PLE8_M) % 2 == 0, "W_up row blocks must be even (ping/pong toggles restart every dispatch)");

extern "C" {

void ple8_up(
    bf16 * u,                              // [8][256]
    bf16 * w_ping, bf16 * w_pong,          // W_up blocks
    bf16 * y_ping, bf16 * y_pong           // [8][32] packets
){
    constexpr int u_prod_lock = 0, u_cons_lock = 1, w_prod_lock = 2, w_cons_lock = 3, y_prod_lock = 4, y_cons_lock = 5;
    alignas(aie::vector_decl_align) float y_acc[PLE8_T * PLE8_M];

    _lock_acquire(u_cons_lock);
    bool w_is_ping = false;
    bool y_is_ping = false;
    for (int rb = 0; rb < MODEL_DIM / PLE8_M; rb++) {
        zero_vectorized<float, PLE8_T * PLE8_M>(y_acc);
        w_is_ping = !w_is_ping;
        bf16* w = w_is_ping ? w_ping : w_pong;
        _lock_acquire(w_cons_lock);
#if defined(PLE8_SKIP_COMPUTE)
#elif defined(PLE8_MVM_SERIAL)
        for (int t = 0; t < PLE8_T; t++) {
            _mvm_bf16_bf16<PLE8_M, PLE8_K>(w, u + t * PLE8_K, y_acc + t * PLE8_M);
        }
#else
        ple8_mvm_block<PLE8_K>(w, u, y_acc);
#endif
        _lock_release(w_prod_lock);

        y_is_ping = !y_is_ping;
        bf16* y = y_is_ping ? y_ping : y_pong;
        _lock_acquire(y_prod_lock);
        for (int t = 0; t < PLE8_T; t++) {
            aie::vector<float, PLE8_M> y_vec = aie::load_v<PLE8_M>(y_acc + t * PLE8_M);
            aie::accum<accfloat, PLE8_M> c_acc;
            c_acc.from_vector(y_vec);
            aie::store_v(y + t * PLE8_M, c_acc.template to_vector<bf16>());
        }
        _lock_release(y_cons_lock);
    }
    _lock_release(u_prod_lock);
}

}
