// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"


template<int L>
void pseduo_glu(bf16 * y, const bf16 * x) {
    bf16* gate_ptr = const_cast<bf16*>(x) + (L / 2);
    bf16* hid_ptr = const_cast<bf16*>(x);
    bf16* y_ptr = y;
    for (int i = 0; i < L / 2; i += 16) {
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr + i);
        aie::vector<bf16, 16> hid_vec = aie::load_v<16>(hid_ptr + i);
        gate_vec = getActivationBf16(gate_vec);

        aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, hid_vec);
        aie::store_v(y_ptr + i, y_vec);
    }
}

// These ids are the ABI with iron/glu.py -- keep them in step.
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;

extern "C" {
// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
AIE_PERSIST_ZERO static bool is_x_ping = false;

/// @brief The MLP's SwiGLU: hid = silu(gate) * up, a round pair at a time.
///
/// The projection engine assembles y one *round* at a time -- 16 cores x
/// QXNX_ROW_BLOCK_SIZE rows = M_PER_ROUND output columns -- and the host weaves
/// the up and gate weight rows in blocks of exactly M_PER_ROUND, so a round is
/// either all up or all gate and the two alternate: round 2p is up, round 2p+1
/// is gate, both covering intermediate columns
/// [p * M_PER_ROUND, (p+1) * M_PER_ROUND).
///
/// GLU_SLICE is two rounds, so **one x buffer is exactly one up round followed
/// by one gate round** and pseduo_glu's split at L/2 lands on the boundary by
/// construction. That is the whole reason the weave exists: concatenating up
/// and gate instead would put gate's matching column 17408 rows away, which no
/// buffer on this tile can span.
///
/// Both operands arrive under the identical [core][row] permutation, so this is
/// a flat elementwise pass -- at L = 1 a round comes back in natural order and
/// there is nothing to un-permute.
///
/// Ping/pong parity: is_x_ping is static across invocations, so the chunk count
/// has to stay even. INTERMEDIATE_SIZE / M_PER_ROUND = 34 pairs, 34 chunks.
void glu(
    bf16 * y,
    const bf16 * x_ping,
    const bf16 * x_pong
)
{
    static_assert((2 * INTERMEDIATE_SIZE) % GLU_SLICE == 0, "GLU_SLICE must divide 2 * INTERMEDIATE_SIZE");
    // One chunk per (up, gate) round pair.
    constexpr int chunks = 2 * INTERMEDIATE_SIZE / GLU_SLICE;
    static_assert(chunks % 2 == 0, "chunk count must stay even or is_x_ping desyncs");

    // Take every outstanding read credit before touching y: the whole
    // INTERMEDIATE_SIZE row is built here across `chunks` passes and must not be
    // sent until the last one has landed.
    _lock_acquire(y_prod_lock, GLU_OUT_REPLAYS);

    bf16* y_it = y;

    for (int i = 0; i < chunks; i++) {
        is_x_ping = !is_x_ping;
        const bf16* x_using = is_x_ping ? x_ping : x_pong;
        _lock_acquire(x_cons_lock);
        pseduo_glu<GLU_SLICE>(y_it, x_using);
        y_it += GLU_SLICE / 2;
        _lock_release(x_prod_lock);
    }

    _lock_release(y_cons_lock, GLU_OUT_REPLAYS);
}


}