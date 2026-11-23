// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// Mirror of qwen3_5vl_npu_sequence::gen_dequant_mm_512 in
// FLM_DLL/detail/qwen3_5vl_npu/qwen3_5vl_npu_sequence.cpp. Keep the two in sync:
// argument order is (L, Din, Dout, ...), and the quantization type used for every
// byte offset must be the one this xclbin's kernel decodes (q4_k here).
// Core-local buffer addresses assigned by aie-assign-buffer-addresses; they move
// whenever the kernel's data layout changes. These are the qwen3_5vl runtime's
// values -- after a bitstream rebuild here, re-read them from
//   build/bitstream/from_iron/dequant_mm.mlir.prj/input_with_addresses.mlir
// (grep for K_div_k_2_7 / rtp_act_2_7 and K_div_k_3_0 / rtp_act_3_0).
static constexpr uint32_t CT_rtp_address_even_row  = 54784;
static constexpr uint32_t act_rtp_address_even_row = 54848;
static constexpr uint32_t CT_rtp_address_odd_row   = 61696;
static constexpr uint32_t act_rtp_address_odd_row  = 61760;


// ---------------------------------------------------------------------------------------------
// Runtime-m variant (this design's xclbin only).  The array works on blocks of 4 core rows x m rows;
// the original kernel is built for m = 128 (512-row blocks).  Here m is chosen per dispatch,
// 16 <= m <= 128 in steps of 16 (the x tile of the core DMA), so a short prompt pays for
// ceil(rows / 64) x 64 rows instead of 512.  Three things carry m:
//   * RTP word 1 of every core (K_div_k[1]): m / 16, read by the kernel,
//   * the shim x / y descriptors (row count),
//   * the lengths of the memtile DMA descriptors, which the design programs for m = 128 (see
//     check_memtile_bds.py, which verifies the ids and values below in the built CDO).
// Rows [rows, covered) of a dispatch are computed on whatever the x buffer holds there and written
// to y: both buffers must span `covered` rows (<= the 512-row padded size of the caller).
// ---------------------------------------------------------------------------------------------
struct rtm_geometry {
    uint32_t blocks;   //!< 4 x m row blocks
    uint32_t m;        //!< rows per core row, multiple of 16
    uint32_t covered;  //!< blocks * 4 * m
};

inline rtm_geometry rtm_pick_geometry(uint32_t rows){
    assert(rows > 0);
    rtm_geometry g;
    g.blocks = (rows + 511) / 512;
    g.m = 16 * ((rows + g.blocks * 64 - 1) / (g.blocks * 64));
    g.covered = g.blocks * 4 * g.m;
    // A reduced-row dispatch pays ~50 us of memtile reconfiguration (before and after); below ~128 rows saved
    // that is a loss (measured: 400 rows -> 448 covered was 4% slower), so keep the zero-overhead full blocks.
    if (g.blocks * 512 - g.covered < 128) { g.m = 128; g.covered = g.blocks * 512; }
    return g;
}

// Memtile (row 1) DMA registers.  The channel rings are parked on buffer descriptors the DMA has
// already fetched, so a descriptor rewritten under a parked channel is not seen: the channels of the
// x / y paths are reset, their locks put back to the initial values (a reset does not cancel an acquire
// that is pending in the lock module; writing the maximum value grants it first), the descriptors are
// rewritten and the channels are started again exactly as the design's CDO starts them (a start-queue
// push per channel; the CDO's trailing "enable" write has mask 0 on this architecture, i.e. does nothing).
static constexpr uint32_t MT_BD_BASE = 0xA0000, MT_BD_STRIDE = 0x20, MT_LOCK_BASE = 0xC0000, MT_LOCK_STRIDE = 0x10;
struct rtm_channel { uint32_t ctrl; uint32_t start_bd; };
static constexpr rtm_channel MT_X_CH[] = {{0xA0600, 0}, {0xA0630, 2}};                                   // S2MM0, MM2S0
static constexpr rtm_channel MT_Y_CH[] = {{0xA0608, 24}, {0xA0610, 6}, {0xA0618, 26}, {0xA0620, 8}, {0xA0638, 28}};  // S2MM1-4, MM2S1
static constexpr uint32_t MT_X_S2MM[]  = {0, 1};
static constexpr uint32_t MT_X_MM2S[]  = {2, 3, 4, 5};
static constexpr uint32_t MT_Y_S2MM[]  = {6, 7, 8, 9, 24, 25, 26, 27};
static constexpr uint32_t MT_Y_MM2S[]  = {28, 29};
static constexpr uint32_t MT_X_LOCKS[] = {0, 1, 2, 3, 4};        // x_prod = 2, x_cons* = 0
static constexpr uint32_t MT_Y_LOCKS[] = {5, 6, 7, 8};        // y_prod_0 = 2, y_prod_1..3 / y_cons = 0
static constexpr uint32_t MT_LOCK_INIT[] = {2, 0, 0, 0, 0};          // x locks
static constexpr uint32_t MT_Y_LOCK_INIT[] = {4, 4, 0, 0};        // y join: 2 x 4 producer credits, 2 consumer counters

// GLU mode (rtp_act == 3): the kernel epilogue emits silu(gate) * up of the two n slices of a core, so a core
// sends 32 columns per row instead of 64. The memtile y descriptors differ from the design's in three
// words: S2MM D2 wrap (the 8x8 n tiles of a row group: 8 -> 4) and the MM2S D0 wrap (row length in words:
// 32 -> 16); buffer lengths halve. Strides and base addresses (the core's region keeps its 64-column pitch)
// are unchanged. 0 = the design's configuration, 1 = GLU, 2 = back to the design's after a GLU dispatch.
enum rtm_y_mode : uint32_t { RTM_Y_PLAIN = 0, RTM_Y_GLU = 1, RTM_Y_RESTORE = 2 };
static constexpr uint32_t MT_Y_S2MM_D2_PLAIN = (8u << 17) | 3u, MT_Y_S2MM_D2_GLU = (4u << 17) | 3u;   // word 4
static constexpr uint32_t MT_Y_MM2S_D0_PLAIN = 32u << 17,       MT_Y_MM2S_D0_GLU = 16u << 17;          // word 2

// put the memtile x / y paths on m rows per core row (m = 128 restores the design's configuration)
inline void rtm_rearm(npu_sequence* seq, uint32_t m, uint32_t ymode = RTM_Y_PLAIN){
    static constexpr uint32_t columns = 8;
    auto bd_word = [](uint32_t bd, uint32_t word){ return MT_BD_BASE + MT_BD_STRIDE * bd + 4 * word; };
    for (uint32_t col = 0; col < columns; col++){
        npu_tiles mt = get_tile(1, col);
        const bool has_x = (col % 2) == 0;     // the x of core row r enters through memtile 2r
        if (has_x) for (auto& c : MT_X_CH) seq->rtp_write(mt, c.ctrl, 2);
        for (auto& c : MT_Y_CH) seq->rtp_write(mt, c.ctrl, 2);
        if (has_x) for (uint32_t i = 0; i < 5; i++){ seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_X_LOCKS[i], 63); seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_X_LOCKS[i], MT_LOCK_INIT[i]); }
        for (uint32_t i = 0; i < 4; i++){ seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_Y_LOCKS[i], 63); seq->rtp_write(mt, MT_LOCK_BASE + MT_LOCK_STRIDE * MT_Y_LOCKS[i], MT_Y_LOCK_INIT[i]); }
        if (has_x){
            for (uint32_t bd : MT_X_S2MM) seq->rtp_write(mt, bd_word(bd, 0), m * 256);
            for (uint32_t bd : MT_X_MM2S) seq->rtp_write(mt, bd_word(bd, 0), m * 128);
        }
        const bool glu = ymode == RTM_Y_GLU;
        for (uint32_t bd : MT_Y_S2MM) {
            seq->rtp_write(mt, bd_word(bd, 0), glu ? m * 16 : m * 32);
            if (ymode != RTM_Y_PLAIN) seq->rtp_write(mt, bd_word(bd, 4), glu ? MT_Y_S2MM_D2_GLU : MT_Y_S2MM_D2_PLAIN);
        }
        for (uint32_t bd : MT_Y_MM2S) {
            seq->rtp_write(mt, bd_word(bd, 0), glu ? m * 64 : m * 128);
            if (ymode != RTM_Y_PLAIN) seq->rtp_write(mt, bd_word(bd, 2), glu ? MT_Y_MM2S_D0_GLU : MT_Y_MM2S_D0_PLAIN);
            seq->rtp_write(mt, bd_word(bd, 4), (m << 17) | 0x1F);
        }
        if (has_x) for (auto& c : MT_X_CH){ seq->rtp_write(mt, c.ctrl, 0); seq->rtp_write(mt, c.ctrl + 4, c.start_bd); }
        for (auto& c : MT_Y_CH){ seq->rtp_write(mt, c.ctrl, 0); seq->rtp_write(mt, c.ctrl + 4, c.start_bd); }
    }
}

// RTP word 1 of every core: rows per core row / 16 (read by the kernel)
inline void rtm_set_core_rows(npu_sequence* seq, uint32_t m){
    for (int row = 2; row < 6; row++){
        const uint32_t ctp = (row % 2 == 0) ? CT_rtp_address_even_row : CT_rtp_address_odd_row;
        for (uint32_t col = 0; col < 8; col++) seq->rtp_write(get_tile(row, col), ctp + 4, m / 16);
    }
}


// ---------------------------------------------------------------------------------------------
// Gate/up weights for the fused GLU dispatch (rtp_act 3).  The model stores the projection as slabs of
// 1024 output rows [up 512 | gate 512] (up[0:512], gate[0:512], up[512:1024], ...), 32-row blocks,
// pairs of consecutive row blocks interleaved along the column blocks (reorder_cpy, vertical_blocks = 2).
// A core of the array owns one such pair (64 rows = its two n slices).  The fused kernel needs the up
// channels and the matching gate channels in the SAME core: pair P = 8 * (2 * slab + half) + c holds
// [up row block 32 * slab + 8 * half + c | gate row block 32 * slab + 16 + 8 * half + c], so a pass of 512
// weight rows yields 256 hidden channels, core c channels 32 c .. 32 c + 31 of the pass.
// dst / src: device-layout weight regions of Dout x Din (Dout a multiple of 1024), may not overlap.
// ---------------------------------------------------------------------------------------------
inline void glu_repack(uint8_t* dst, const uint8_t* src, uint32_t Din, uint32_t Dout){
    assert(Dout % 1024 == 0 && Din % QXNX_COL_BLOCK_SIZE == 0);
    const size_t bb = get_quantization_byte_size((size_t)QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
    const uint32_t bpr = Din / QXNX_COL_BLOCK_SIZE;
    auto off = [&](uint32_t rb, uint32_t cb){ return ((size_t)(rb / 2) * 2 * bpr + (size_t)cb * 2 + (rb % 2)) * bb; };
    const uint32_t slabs = Dout / 1024;
    for (uint32_t s = 0; s < slabs; s++)
        for (uint32_t h = 0; h < 2; h++)
            for (uint32_t c = 0; c < 8; c++){
                const uint32_t P = (2 * s + h) * 8 + c;
                const uint32_t rb_up = 32 * s + 8 * h + c, rb_gate = 32 * s + 16 + 8 * h + c;
                for (uint32_t cb = 0; cb < bpr; cb++){
                    memcpy(dst + off(2 * P, cb), src + off(rb_up, cb), bb);
                    memcpy(dst + off(2 * P + 1, cb), src + off(rb_gate, cb), bb);
                }
            }
}

void gen_dequant_mm_rtm(npu_sequence* seq, const uint32_t rows, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const uint32_t row_base = 0){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr int qm = QXNX_ROW_BLOCK_SIZE;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;
    int K_div_k = Din / desired_k;
    const uint32_t columns = 8;
    assert(Dout / qm / columns > 0);
    assert(Dout % (qm * columns) == 0);

    const rtm_geometry g = rtm_pick_geometry(rows);
    const uint32_t m = g.m;
    const int M_div_m = g.blocks;
    int N_div_n = Dout / desired_n / columns;
    // rtp_act 3: gate/up weights re-packed in 64-row core groups of [up block | gate block] (see glu_repack),
    // y = silu(gate) * up, Dout / 2 columns wide
    const bool glu = rtp_act == 3;
    const uint32_t Dy = glu ? Dout / 2 : Dout;             // row pitch of y
    const uint32_t pass_cols = glu ? desired_n * columns / 2 : desired_n * columns;   // output columns per pass
    const uint32_t half_cols = pass_cols / 2;              // columns served by one of the two memtile groups

    seq->clear_cmds();
#ifdef DIAG_BUS_EN
    // DIAG: let the cores read their own module registers (Timer_Low) with read_tm: Core_Processor_Bus.enable
    for (int row = 2; row < 6; row++) for (uint32_t col = 0; col < columns; col++) seq->rtp_write(get_tile(row, col), 0x32038, 1);
#endif

    for (int row = 2; row < 6; row++){
        const bool even = (row % 2) == 0;
        const uint32_t ctp = even ? CT_rtp_address_even_row : CT_rtp_address_odd_row;
        const uint32_t atp = even ? act_rtp_address_even_row : act_rtp_address_odd_row;
        for (uint32_t col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, ctp, K_div_k);
            seq->rtp_write(tile, atp, rtp_act);
        }
    }
    rtm_set_core_rows(seq, m);
    if (glu) rtm_rearm(seq, m, RTM_Y_GLU); else if (m != 128) rtm_rearm(seq, m);

    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            int round = i * N_div_n + j;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            uint32_t y_offset = (row_base + i * m * 4) * Dy + j * pass_cols;
            uint32_t x_offset = (row_base + i * m * 4) * Din;
            int w_offset = get_quantization_byte_size((size_t)j * desired_n * Din * columns, flm_q4k) + weight_offset;

            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                seq->npu_dma_memcpy_nd(
                    sizeof(char), 2, MM2S, IT[col], w_bd_0, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(
                    sizeof(char), 2, MM2S, IT[col + 4], w_bd_1, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k) + get_quantization_byte_size(4 * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0, S2MM, IT[col], y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * m * Dy + out_offset)},
                    {1, 1, m, half_cols}, {0, 0, Dy, 1}, -1, 0, true);
                seq->npu_dma_memcpy_nd(
                    2, 0, S2MM, IT[col + 4], y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * m * Dy + half_cols + out_offset)},
                    {1, 1, m, half_cols}, {0, 0, Dy, 1}, -1, 0, true);
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1, MM2S, IT[2 * col], x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * m * Din},
                    {1, (uint32_t)K_div_k, m, desired_k}, {0, desired_k, Din, 1}, -1, 0, false);
            }
            if (round > 0){
                for (int r = 0; r < 4; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < 4; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    if (glu){ rtm_rearm(seq, 128, RTM_Y_RESTORE); rtm_set_core_rows(seq, 128); }
    else if (m != 128){ rtm_rearm(seq, 128); rtm_set_core_rows(seq, 128); }   // leave the design's configuration behind
    seq->cmds2seq();
}

void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int m = QXNX_ROW_BLOCK_SIZE;
    static constexpr int k = QXNX_COL_BLOCK_SIZE;
    static constexpr uint32_t elems = m * k;
    static constexpr uint32_t desired_m = 128;
    static constexpr uint32_t desired_k = 512;
    static constexpr uint32_t desired_n = 64;
    //  %K_div_k_2_7 = aie.buffer(%tile_7_2) {address = 54784 : i32, sym_name = "K_div_k_2_7"} : memref<16xi32>
    // %rtp_act_2_7 = aie.buffer(%tile_7_2) {address = 54848 : i32, sym_name = "rtp_act_2_7"} : memref<16xi32>
    // %K_div_k_3_0 = aie.buffer(%tile_0_3) {address = 61696 : i32, sym_name = "K_div_k_3_0"} : memref<16xi32>
    // %rtp_act_3_0 = aie.buffer(%tile_0_3) {address = 61760 : i32, sym_name = "rtp_act_3_0"} : memref<16xi32>

    uint32_t a_block_size = get_quantization_byte_size(m * k, flm_q4k);

    const uint32_t blocks_per_row = Din / k;
    int K_div_k = Din / desired_k;

    const uint32_t columns = 8;
    assert(Dout / m / columns > 0);
    assert(Dout % (m * columns) == 0);

    seq->clear_cmds();

    int M_div_m = L / (desired_m * 4); // 4 row-blocks processed per i iteration (IT[col], col=0..3)
    int N_div_n = Dout / desired_n / columns;
    assert(L % (desired_m * 4) == 0);

    for (int row = 2; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_even_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_even_row, rtp_act);
        }
    }
    for (int row = 3; row < 6; row+=2){
        for (int col = 0; col < columns; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, CT_rtp_address_odd_row, K_div_k);
            seq->rtp_write(tile, act_rtp_address_odd_row, rtp_act);
        }
    }

    for (int i = 0; i < M_div_m; i++){
        for (int j = 0; j < N_div_n; j++){
            int round = i * N_div_n + j;
            int is_pong = round % 2;
            int bd_offset = is_pong * 8;
            uint32_t y_offset = i * desired_m * 4 * Dout + j * desired_n * columns;
            uint32_t x_offset = i * desired_m * 4 * Din;
            int w_offset = get_quantization_byte_size((size_t)j * desired_n * Din * columns, flm_q4k) + weight_offset;

            npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
            npu_bd_id y_bd_1 = npu_bd_id(3 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd_0 = npu_bd_id(2 + bd_offset);
            npu_bd_id w_bd_1 = npu_bd_id(4 + bd_offset);
            for (uint32_t col = 0; col < 4; col++){
                seq->npu_dma_memcpy_nd(
                    sizeof(char),
                    2,
                    MM2S,
                    IT[col],
                    w_bd_0,
                    it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                seq->npu_dma_memcpy_nd(
                    sizeof(char),
                    2,
                    MM2S,
                    IT[col + 4],
                    w_bd_1,
                    it_channel_1,
                    {0, 0, 0, (uint32_t)(w_offset + get_quantization_byte_size((size_t)col * desired_n * Din, flm_q4k) + get_quantization_byte_size(4 * desired_n * Din, flm_q4k))},
                    {1, 1, 1, (uint32_t)get_quantization_byte_size(desired_n * Din, flm_q4k)},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    y_bd_0, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + out_offset)},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, (uint32_t)(y_offset + col * desired_m * Dout + 4 * desired_n + out_offset)},
                    {1, 1, desired_m, desired_n * 4},
                    {0, 0, Dout, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[2 * col],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * desired_m * Din},
                    {1, (uint32_t)K_div_k, desired_m, desired_k},
                    {0, desired_k, Din, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < 4; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
            }
        }// end j
    } // end i
    for (int r = 0; r < 4; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

#endif
