// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"

// Query rows per forward pass, set by the top-level Makefile. Repeated here so
// the file stands alone in an IDE.
#ifndef L_TOKENS
#define L_TOKENS 8
#endif

// ---------------------------------------------------------------------------
// SwiGLU, one projection *round* at a time.
//
// The engine emits its y a round at a time: 16 cores x QXNX_ROW_BLOCK_SIZE rows
// = M_PER_ROUND output columns for every one of the L_TOKENS token rows, so an
// assembled packet is GLU_ROUND words laid out [core][token][row] -- permuted,
// but the permutation is the *same* for every round.
//
// M_PER_ROUND is 512 and load_weights() weaves the up|gate weight rows 512 at a
// time, so a round is either all `up` or all `gate` and the two alternate:
// round 2p is up, round 2p+1 is gate, both covering intermediate columns
// [p * M_PER_ROUND, (p+1) * M_PER_ROUND). Two rounds in, one round out, and
// because both operands carry the same permutation the whole thing is a flat
// elementwise pass -- no reshuffling here, and the un-permute happens once, in
// the shim BD on the way to DDR.
//
// That is also why this is not the L=1 kernel it replaces: at L_TOKENS = 8 the
// hidden result is 8 * INTERMEDIATE_SIZE bf16 = 278 KB, four times the tile, so
// hid cannot be accumulated and replayed into the down projection. It leaves a
// round at a time instead.
// ---------------------------------------------------------------------------
constexpr int GLU_ROUND = L_TOKENS * M_PER_ROUND;
constexpr int GLU_PAIRS = INTERMEDIATE_SIZE / M_PER_ROUND;

template<int N>
void swiglu_round(bf16 * y, const bf16 * up, const bf16 * gate) {
    for (int i = 0; i < N; i += 16) {
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate + i);
        aie::vector<bf16, 16> up_vec = aie::load_v<16>(up + i);
        gate_vec = getActivationBf16(gate_vec);

        aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, up_vec);
        aie::store_v(y + i, y_vec);
    }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;

extern "C" {
// Ping/pong parity has to survive the call boundary: the DMA chains keep
// walking, so the next invocation resumes on whichever buffer the last one left
// off at. Both counts here are even -- 2 * GLU_PAIRS rounds in, GLU_PAIRS out --
// so the statics come back to false and the two sides stay in step.
// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
AIE_PERSIST_ZERO static bool is_x_ping = false;
AIE_PERSIST_ZERO static bool is_y_ping = false;
void glu(
    bf16 * y_ping,
    bf16 * y_pong,
    const bf16 * x_ping,
    const bf16 * x_pong
)
{
    static_assert(INTERMEDIATE_SIZE % M_PER_ROUND == 0,
                  "M_PER_ROUND must divide INTERMEDIATE_SIZE: a round has to be all up or all gate");
    static_assert(2 * GLU_PAIRS == UP_GATE_REPEATS,
                  "the pair loop must consume exactly the rounds the up|gate mvm emits");

    for (int p = 0; p < GLU_PAIRS; p++) {
        // up first, then gate -- the weave order load_weights() lays down.
        is_x_ping = !is_x_ping;
        const bf16 * up = is_x_ping ? x_ping : x_pong;
        _lock_acquire(x_cons_lock);

        is_x_ping = !is_x_ping;
        const bf16 * gate = is_x_ping ? x_ping : x_pong;
        _lock_acquire(x_cons_lock);

        is_y_ping = !is_y_ping;
        bf16 * y = is_y_ping ? y_ping : y_pong;
        _lock_acquire(y_prod_lock);

        swiglu_round<GLU_ROUND>(y, up, gate);

        _lock_release(x_prod_lock, 2);
        _lock_release(y_cons_lock);
    }
}


}
