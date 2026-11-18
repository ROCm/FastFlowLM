// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __AIE_ARRAY_LAYOUT_H__
#define __AIE_ARRAY_LAYOUT_H__

#include "typedef.hpp"
#include "model_spec.h"

// calculate the number of repeats of x;
// Each round, the MVM engine computes QKV_ROW_BLOCK_SIZE * MVM_CORES of rows for the output;
constexpr int MVM_CORES = 16;
constexpr int QKV_REPEATS = (DQ + DK + DV) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS = (2 * INTERMEDIATE_SIZE) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = 32 * MVM_CORES;

// Packet buffer layout, shared with iron/proj_tile.py and iron/rms_residual.py
// through iron/constants.py -- change both or neither.
//
// A packet buffer is [pad][one uint32 stream packet header][payload].  The
// header word has to sit immediately before the payload, because the DMA sends
// the two as one contiguous run (see the dma_bd offset/len in those two files).
//
// The payload offset is what pins the alignment of every vector access the
// kernels make into a packet.  aie_kernel_utils.h widens those to 512 bits,
// which aie_api requires to be 64-byte aligned on AIE2P
// (aie_api/detail/ld_st.hpp, __AIE_ARCH__ == 21); a 512-bit access to a
// 32-byte-aligned address is *silently* truncated down to the 64-byte boundary
// by the hardware, so getting this wrong reads and writes the wrong data
// without any diagnostic.  Buffer bases are 64-byte aligned, so the payload
// offset must be a multiple of 32 bf16 elements.
//
// 32 rather than 16 also makes both packet buffers a whole number of 64-byte
// units long, which keeps the buffer allocated after them 64-byte aligned too;
// the RMS tile's norm-weight buffer used to land at a 32-byte offset for
// exactly that reason.
constexpr int PKT_PAYLOAD_OFFSET = 32;                     // bf16 elements
constexpr int PKT_ID_OFFSET      = PKT_PAYLOAD_OFFSET - 2; // the uint32 header

static_assert(PKT_PAYLOAD_OFFSET % 32 == 0,
              "packet payload must start 64-byte aligned; see aie_kernel_utils.h");

constexpr int pkt_id_to_rope     = 1;
constexpr int pkt_id_to_rms_norm = 4;
constexpr int pkt_id_to_glu      = 8;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it   = 1;

// PARALLEL_ROWS -- how many rows of a weight block one vector operand covers --
// is the packing stride of the model's weights and comes from models/<model>.h.
// A block's rows then take this many passes. See the layout note in
// ../models/all_models.h.
constexpr int ROW_PASSES = Q4NX_ROW_BLOCK_SIZE / PARALLEL_ROWS;

static_assert(Q4NX_ROW_BLOCK_SIZE % PARALLEL_ROWS == 0,
              "PARALLEL_ROWS must divide the row block");

// Q4_K and Q8_0 are only ever packed plain column major over the whole row span
// (FLM_Q4NX_Converter's _pack_q4k / _pack_q8nx), and their kernels index the
// per-group metadata assuming one full-width pass.
#if QUANT_TYPE == __Q4_K__ || QUANT_TYPE == __Q8_0__
static_assert(PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE,
              "Q4_K / Q8_0 require PARALLEL_ROWS == Q4NX_ROW_BLOCK_SIZE");
#endif

#endif // __AIE_ARRAY_LAYOUT_H__
