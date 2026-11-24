// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// 8-row GLU of the verify layer (plan S6.7, D17).  Built only for L_TOKENS > 1; the L_TOKENS = 1 layer keeps kernels/glu.cc.
//
// The up|gate projection emits 512-row rounds; load_weights() weaves the weights so that round 2p is all `up`, round 2p+1 all `gate`,
// both covering intermediate columns [512 p, 512 (p + 1)).  The memtile stage in front of this tile (iron/design_rmsglu.py glu_stage_mt)
// hands every round over in the order [half][t][256] (half = column 256 * half of the round), so that
//   x      = [ up round (8 x 512 as [2][8][256]) | gate round (same) ]      one BD of 2 * GLU_ROUND elements
//   y      = gelu_tanh(gate) * up                                           GLU_ROUND elements, same [2][8][256] order
// is a flat elementwise pass.  A y round is 4096 consecutive elements of the down projection's x buffer [slice j][t][256]
// (j = 2 p + half), which is the layout the replay read BD of the down phase expects (iron/design_rmsglu.py).
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"

#if L_TOKENS > 1
constexpr int GLU_ROUND = L_TOKENS * M_PER_ROUND;          // elements per round
constexpr int GLU_PAIRS = INTERMEDIATE_SIZE / M_PER_ROUND; // up|gate pairs = hidden rounds

static_assert(INTERMEDIATE_SIZE % M_PER_ROUND == 0, "a round has to be all up or all gate");
static_assert(2 * GLU_PAIRS == UP_GATE_REPEATS, "the pair loop must consume exactly the rounds the up|gate projection emits");
static_assert(GLU_PAIRS % 2 == 0, "ping/pong parity must survive the call boundary");

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;

extern "C" {
AIE_PERSIST_ZERO static bool is_x_ping = false;
AIE_PERSIST_ZERO static bool is_y_ping = false;
void glu_l8(bf16 *y_ping, bf16 *y_pong, const bf16 *x_ping, const bf16 *x_pong) {
    for (int p = 0; p < GLU_PAIRS; p++) {
        is_x_ping = !is_x_ping;
        const bf16 *up = is_x_ping ? x_ping : x_pong;
        const bf16 *gate = up + GLU_ROUND;
        _lock_acquire(x_cons_lock);

        is_y_ping = !is_y_ping;
        bf16 *y = is_y_ping ? y_ping : y_pong;
        _lock_acquire(y_prod_lock);

        for (int i = 0; i < GLU_ROUND; i += 16) {
            aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate + i);
            aie::vector<bf16, 16> up_vec = aie::load_v<16>(up + i);
            gate_vec = getActivationBf16(gate_vec);
            aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, up_vec);
            aie::store_v(y + i, y_vec);
        }

        _lock_release(x_prod_lock);
        _lock_release(y_cons_lock);
    }
}
}
#endif // L_TOKENS > 1
