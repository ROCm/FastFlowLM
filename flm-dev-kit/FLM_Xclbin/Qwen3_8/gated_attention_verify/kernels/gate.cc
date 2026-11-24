// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.


#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"

// Half a row of the attention output gate.
//
// There are two of these tiles, one per attention column, and each owns the
// half of the row its column produced: column 4 computes q heads 0..11, which
// is dims [0, DQ/2), and column 5 computes heads 12..23.
//
// **Why halves rather than one gate tile.** attn * sigmoid(z) is elementwise,
// so a half-row of attention only ever needs the matching half of z -- there is
// nothing to merge. Merging them first would need three inbound streams (two
// attention halves and z) on a tile that has two S2MM channels, so it would
// cost an extra memtile hop to reassemble a row that is then immediately split
// again by the arithmetic. Splitting the *gate* instead gives each tile exactly
// two inbound streams, which is what a compute tile has, and the two halves run
// concurrently.
//
// The split is free on the z side too: DQ/2 = 3072 is a whole number of the
// projection engine's 32-row blocks, so "which half" is just where the staging
// memtile's outbound BD starts.
constexpr int GATE_WIDTH = DQ / 2;
static_assert(DQ % 2 == 0, "the two attention columns split the row evenly");
static_assert(GATE_WIDTH % 32 == 0, "apply_gate steps 32 lanes at a time");

template<int L>
void activate_gate(bf16* gate){
    bf16* gate_ptr = gate;
    for (int i = 0; i < L; i += 16) {
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr);
        gate_vec = getSigmoidBf16_nonLUT<16>(gate_vec);

        aie::store_v(gate_ptr, gate_vec);
        gate_ptr += 16;
    }
}

template<int L>
void apply_gate(bf16 * y, const bf16 * gate, const bf16 * attn_out){
    bf16* gate_ptr_0 = const_cast<bf16*>(gate);
    bf16* gate_ptr_1 = const_cast<bf16*>(gate) + 16;
    bf16* attn_out_ptr_0 = const_cast<bf16*>(attn_out);
    bf16* attn_out_ptr_1 = const_cast<bf16*>(attn_out) + 16;
    bf16* y_ptr_0 = y;
    bf16* y_ptr_1 = y + 16;
    for (int i = 0; i < L; i += 32) {
        aie::vector<bf16, 16> gate_vec_0 = aie::load_v<16>(gate_ptr_0);
        aie::vector<bf16, 16> gate_vec_1 = aie::load_v<16>(gate_ptr_1);
        aie::vector<bf16, 16> attn_out_vec_0 = aie::load_v<16>(attn_out_ptr_0);
        aie::vector<bf16, 16> attn_out_vec_1 = aie::load_v<16>(attn_out_ptr_1);

        aie::vector<bf16, 16> y_vec_0 = aie::mul(gate_vec_0, attn_out_vec_0);
        aie::vector<bf16, 16> y_vec_1 = aie::mul(gate_vec_1, attn_out_vec_1);

        aie::store_v(y_ptr_0, y_vec_0);
        aie::store_v(y_ptr_1, y_vec_1);
        gate_ptr_0 += 32;
        gate_ptr_1 += 32;
        attn_out_ptr_0 += 32;
        attn_out_ptr_1 += 32;
        y_ptr_0 += 32;
        y_ptr_1 += 32;
    }
}

constexpr int gate_prod_lock = 0;
constexpr int gate_cons_lock = 1;
constexpr int attn_out_prod_lock = 2;
constexpr int attn_out_cons_lock = 3;
constexpr int gated_attn_out_prod_lock = 4;
constexpr int gated_attn_out_cons_lock = 5;

extern "C" {

///@brief One token's half-row of the output gate, into the shared row.
///
/// Two adjacent tiles in one column run this and split the row. **`y_row` is the
/// whole DQ row and lives on the emitter**: rejoining the halves here, through
/// the neighbour's memory window, costs no DMA and hands the consumer a single
/// producer. Two flows into one port would not express it -- two producers on
/// one S2MM channel must be one merged packet flow, and the arrival order would
/// be a timing race.
///
/// `is_emitter` only picks the lock window: the emitter uses its own locks
/// (+48), the helper reaches north for the same ones (+32), since the buffer is
/// there. Separate from `half` because which half a core computes is fixed by
/// its attention column and which core emits is fixed by placement.
///
///@param y_row the shared DQ row; this core writes [half * GATE_WIDTH, +GATE_WIDTH)
///@param gate this core's half of z, overwritten in place by its sigmoid
///@param attn_out this core's attention column's half-row
///@param half 0 for attention column 4 (q heads 0..11), 1 for column 5
///@param is_emitter non-zero if y_row is this core's own buffer
void gate(
    bf16 * y_row,
    bf16 * gate,
    bf16 * attn_out,
    int half,
    int is_emitter
)
{
    // One invocation is one token. Must match gate_tile.repeats in iron/gate.py.
    const int repeats = 1;

    // **Acquire z first, and not only for latency.** z is parked in MT[3] before
    // attention starts, while the attention half has to cross q/k norm, RoPE,
    // the cache append and the whole softmax column. Taking z first runs the
    // sigmoid -- the expensive part, a tanh per 16 lanes -- while the array is
    // still busy, leaving only the multiply.
    //
    // Both are in-order token streams, so token t always meets token t; there is
    // no tag here and nothing would catch a slip.
    _lock_acquire(gate_cons_lock);
    activate_gate<GATE_WIDTH>(gate);

    _lock_acquire(attn_out_cons_lock);

    // **One credit per core**, so the halves are unordered: each takes one,
    // writes its own disjoint range, releases one, and the outbound BD waits for
    // both. Deliberate -- the attention columns do not finish together.
    bf16 *y = y_row + half * GATE_WIDTH;
    if (is_emitter) {
        _lock_acquire(gated_attn_out_prod_lock, repeats);
        apply_gate<GATE_WIDTH>(y, gate, attn_out);
        _lock_release(gated_attn_out_cons_lock, repeats);
    } else {
        _up_lock_acquire(gated_attn_out_prod_lock, repeats);
        apply_gate<GATE_WIDTH>(y, gate, attn_out);
        _up_lock_release(gated_attn_out_cons_lock, repeats);
    }

    _lock_release(gate_prod_lock);
    _lock_release(attn_out_prod_lock);
}
}
