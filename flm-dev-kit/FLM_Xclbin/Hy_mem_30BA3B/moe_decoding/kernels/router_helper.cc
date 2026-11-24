// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "router_block.hpp"

// In the router CT (the tile above), reached with _up_lock_*.
constexpr int router_helper_x_ready_lock = 13;     // the router copied x
constexpr int router_helper_finished_lock = 14;    // set when this tile's logits are in

// This tile's own weight blocks (IT5 channel 1 -> MT5 -> this tile's S2MM 0).
constexpr int w_prod_lock = 0;
constexpr int w_cons_lock = 1;

extern "C" {
// Router logits of experts NUM_EXPERTS/2 .. NUM_EXPERTS-1, written straight into
// the router CT's logit buffer. The router does the first half from its own
// stream at the same time, so the router weight comes in over 2 streams.
void router_helper(
    bf16* AIE_DMA_RESTRICT x_in,                  // the router's copy of x (router memory)
    bfloat16* AIE_DMA_RESTRICT w_block_ping,      // this tile's weight blocks
    bfloat16* AIE_DMA_RESTRICT w_block_pong,
    float* logits                         // NUM_EXPERTS fp32 (router memory)
)
{
    static bool is_ping = true;

    _up_lock_acquire(router_helper_x_ready_lock);
    for (AIE_LOCK_LOOP_CTR i = NUM_EXPERTS / 2; i < NUM_EXPERTS; i += router_weight_row_block) {
        for (AIE_LOCK_LOOP_CTR j = 0; j < MODEL_DIM; j += router_weight_col_block) {
            _lock_acquire(w_cons_lock);
            router_block_matvec(is_ping ? w_block_ping : w_block_pong, x_in + j, logits + i);
            _lock_release(w_prod_lock);
            is_ping = !is_ping;
        }
    }
    _up_lock_release(router_helper_finished_lock);
}
}
