// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;

// The MTP fusion's fc: 2 * MODEL_DIM -> MODEL_DIM. Its K is the only one in
// the head that is not MODEL_DIM, DQ or INTERMEDIATE_SIZE, which is why it
// needs a constant of its own rather than reusing O_DOWN_REPEATS -- the row
// count matches (both emit MODEL_DIM rows), but the x it reduces over is
// twice as wide, so the slice count per row block differs.
constexpr int FC_IN = 2 * MODEL_DIM;
constexpr int FC_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

// q|k|v only, i.e. without the z half of q_proj.
constexpr int QKV_ONLY_REPEATS = (DQ + DK + DV) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

// z, the output gate: the second DQ rows of self_attn.q_proj.weight, which the
// q4nx converter has already de-interleaved into a clean second half.
//
// It is a separate mvm() from qkv but reduces over the *same* xn, so the rms
// tile has to keep that one norm on the wire for both -- which is why
// rms_residual.cc releases QKV_REPEATS credits rather than QKV_ONLY_REPEATS.
constexpr int Z_REPEATS   = DQ / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int QKV_REPEATS = QKV_ONLY_REPEATS + Z_REPEATS;

constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / QXNX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / QXNX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = QXNX_ROW_BLOCK_SIZE * MVM_CORES;

// How many times the GLU tile emits its SwiGLU result.
//
// down_proj is a streaming mvm(): it re-reads the whole INTERMEDIATE_SIZE-wide
// hidden row once per row block, and it owns O_DOWN_REPEATS = 10 of them. The
// DDR tap that used to be the only consumer is gone -- the row now leaves as a
// packet into the projection engine's x FIFO -- so this replay *is* down's M/m
// loop, the same way gate_tile.repeats is o_proj's.
//
// Must stay equal to `replays` in iron/glu.py: the core takes that many credits
// before writing hid and releases that many after, and the outbound BD takes
// one per send. 10 is comfortably under the lock's 63 ceiling, unlike up|gate's
// 68 -- which is why that one had to go resident and this one does not.
constexpr int GLU_OUT_REPLAYS = O_DOWN_REPEATS;

// The two ping/pong chains in proj_main.cc are static across invocations, so
// what has to stay even is the *sum* over every mvm the kernel runs, not each
// term. Get this wrong and the toggles start the next token on the wrong
// buffer -- a hang whose first few tokens are correct.
//
// These two sums must list exactly the phases proj_main() runs, which is now
// the whole layer: fc, qkv, z, o, up|gate and down.
//
// fc and up|gate are *resident* (mvm_resident): they drain their activation
// once, K/k slices, rather than once per row block, so their x term is a flat
// K/k. qkv, z, o and down are streaming mvm()s and consume a fresh slice per
// (row block, column block) pair, so theirs are per-row-block products. o's
// reduction is DQ wide and down's is INTERMEDIATE_SIZE wide, not MODEL_DIM --
// they reduce over the gated attention row and the SwiGLU row respectively.
//
// up|gate is resident for a reason beyond traffic: streaming it would need the
// rms tile to replay its post-attention norm UP_GATE_REPEATS = 68 times, and a
// lock saturates at 63. Latching the norm on the tile sidesteps the ceiling
// entirely rather than splitting the credit across a relay chain. down is
// streaming because its replay count is only O_DOWN_REPEATS = 10, well under
// the ceiling, and its x is 34 KiB -- too big to latch on a core that already
// holds a 20 KiB x_full.
constexpr int TOTAL_ROW_BLOCKS = FC_REPEATS + QKV_ONLY_REPEATS + Z_REPEATS
                               + O_DOWN_REPEATS + UP_GATE_REPEATS
                               + O_DOWN_REPEATS;
constexpr int TOTAL_X_SLICES   = FC_IN / QXNX_COL_BLOCK_SIZE
                               + (QKV_ONLY_REPEATS + Z_REPEATS)
                                 * (MODEL_DIM / QXNX_COL_BLOCK_SIZE)
                               + O_DOWN_REPEATS * (DQ / QXNX_COL_BLOCK_SIZE)
                               + MODEL_DIM / QXNX_COL_BLOCK_SIZE
                               + O_DOWN_REPEATS
                                 * (INTERMEDIATE_SIZE / QXNX_COL_BLOCK_SIZE);

// One replay of the SwiGLU row is exactly one row block's worth of down's x.
// If this ever stopped holding, GLU_OUT_REPLAYS would no longer be down's M/m
// loop and the two would drift apart silently.
static_assert(INTERMEDIATE_SIZE % QXNX_COL_BLOCK_SIZE == 0,
              "down's x must be a whole number of slices");

// A round is all up or all gate, and the GLU tile pairs two of them into one
// GLU_SLICE-wide buffer. Both have to divide evenly or the pairing slips.
static_assert(GLU_SLICE == 2 * M_PER_ROUND,
              "the GLU tile's x buffer is exactly one up round plus one gate round");
static_assert(INTERMEDIATE_SIZE % M_PER_ROUND == 0,
              "M_PER_ROUND must divide INTERMEDIATE_SIZE: a round has to be all up or all gate");

static_assert(TOTAL_ROW_BLOCKS % 2 == 0,
              "total row-block count must stay even or is_y_ping desyncs");
static_assert(TOTAL_X_SLICES % 2 == 0,
              "total x slice count must stay even or is_x_ping desyncs");

// One-hot, and they must agree with iron/constants.py: a switchbox demuxes
// several logical streams off one physical MM2S by masking on these, so two
// ids sharing a bit would both match.
//
// fc_out *and* down, both north to the rms tile. One id, because they share a
// destination channel and the tile's ping/pong chain orders them by arrival --
// see the note in iron/constants.py for why down cannot keep an id of its own
// now that its route ends at CT[0][2] instead of IT[2].
constexpr int pkt_id_to_rms      = 2;   // fc_out, then down
constexpr int pkt_id_to_rope     = 4;   // q|k|v
// z *and* o. Both land on MT[2] and the memtile's BD chain tells them apart by
// arrival order, so they share one id and one rule per switchbox.
constexpr int pkt_id_to_aux      = 8;
constexpr int pkt_id_to_glu      = 16;  // up|gate

// The rms tile's own y ids. Two destination sets on one master port: west to
// the projection engine, south to IT[2].
constexpr int pkt_id_rms_to_proj = 0;   // phases 1 and 2, xn and yn -> the engine
constexpr int pkt_id_rms_to_out  = 4;   // phase 3, h2 -> IT[2], the layer output

// The passthrough tile that stages the fc input. Both halves go out under the
// same id -- the proj side tells them apart by arrival order, not by id.
constexpr int pkt_id_pass_to_proj = 0;
constexpr int pkt_id_pass_to_it   = 4;


constexpr bfloat16 exp_scale = 1.44269504089;
constexpr int W_SLICE_SIZE = 512;
constexpr int Q4_WEIGHTS_COUNT = W_SLICE_SIZE * 2;
constexpr int Q8_WEIGHTS_COUNT = W_SLICE_SIZE;
// The MVM cores hold a whole weight block's rows at once: one column of a block
// is exactly PARALLEL_ROWS elements, so this is a layout constant tied to the
// block shape, not a tuning knob.
constexpr int PARALLEL_ROWS = QXNX_ROW_BLOCK_SIZE;
// proj related

constexpr int Q4_COL_PER_SLICE = W_SLICE_SIZE / PARALLEL_ROWS * 2;
constexpr int Q8_COL_PER_SLICE = W_SLICE_SIZE / PARALLEL_ROWS;

#endif // __AIE_ARRAY_LAYOUT_H__