// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lfm3_npu_sequence.hpp"
#include <cstring>

lfm3_npu_sequence::lfm3_npu_sequence(const lfm3_desc& d, uint32_t max_l) : desc(&d) {
    D = d.D; DH = d.DH; DK = d.DK; DV = d.DK; DQ = d.DQ; I = d.intermediate_size;
    vocab_padded = pad_vocab(d.vocab_size, d.D);
    conv_hist_rows = d.conv_kernel - 1;
    set_max_length(max_l);
}

void lfm3_npu_sequence::set_max_length(uint32_t max_l) {
    MAX_L = max_l;
    const size_t quarter = (size_t)max_l * (DK / 2);     // one band: MAX_L rows of DK/2 elements
    K03_offset = 0;
    K47_offset = quarter;
    V03_offset = 2 * quarter;
    V47_offset = 3 * quarter;
}

void lfm3_npu_sequence::_send_x(npu_sequence* seq) {
    seq->npu_dma_memcpy_nd(sizeof(bf16), xy_arg_idx, MM2S, xyr_tile, bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1}, 0, 0, false, aggressive_cache);
}

// The rms tile holds all four norm weights of the layer (4 x D); one DMA fills the buffer.
void lfm3_npu_sequence::_send_rms_weights(npu_sequence* seq) {
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_rms_arg_idx, MM2S, xyr_tile, bd_1, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D * NORMS_PER_LAYER}, {0, 0, 0, 1}, 0, 0, false, aggressive_cache);
}

void lfm3_npu_sequence::_send_rope_weights(npu_sequence* seq) {
    // conv layer: the K taps [K][D] (the rope tile's buffer is read by the conv core)
    seq->npu_dma_memcpy_nd(sizeof(bf16), w_rope_arg_idx, MM2S, xyr_tile, bd_2, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)desc->conv_kernel * D}, {0, 0, 0, 1}, 1, 0, false, aggressive_cache);
}

// The conv history lives in DDR as a ring of R = K-1 rows. The token at position t (0-based, t = L-1) overwrites
// slot t % R (which holds token t-R, the oldest). The kernel wants the history oldest to newest, i.e. slots p..R-1
// then 0..p-1 with p = t % R.
void lfm3_npu_sequence::_send_conv_cache(npu_sequence* seq, const uint32_t L) {
    static constexpr int conv_cache_idx = kv_cache_arg_idx;
    const uint32_t R = conv_hist_rows;
    const uint32_t p = (L - 1) % R;
    if (p == 0) {
        seq->npu_dma_memcpy_nd(sizeof(bf16), conv_cache_idx, MM2S, xyr_tile, bd_14, it_channel_1,
            {0, 0, 0, 0}, {1, 1, 1, R * D}, {0, 0, 0, 1}, 8, 0, true, aggressive_cache);
    } else {
        seq->npu_dma_memcpy_nd(sizeof(bf16), conv_cache_idx, MM2S, xyr_tile, bd_14, it_channel_1,
            {0, 0, 0, p * D}, {1, 1, 1, (R - p) * D}, {0, 0, 0, 1}, 8, 0, false, aggressive_cache);
        seq->npu_dma_memcpy_nd(sizeof(bf16), conv_cache_idx, MM2S, xyr_tile, bd_15, it_channel_1,
            {0, 0, 0, 0}, {1, 1, 1, p * D}, {0, 0, 0, 1}, 8, 0, true, aggressive_cache);
    }
    seq->npu_dma_wait(xyr_tile, MM2S, it_channel_1);
}

void lfm3_npu_sequence::_receive_conv_cache(npu_sequence* seq, const uint32_t L) {
    static constexpr int conv_cache_idx = kv_cache_arg_idx;
    const uint32_t slot = (L - 1) % conv_hist_rows;
    seq->npu_dma_memcpy_nd(sizeof(bf16), conv_cache_idx, S2MM, xyr_tile, bd_13, it_channel_0,
        {0, 0, 0, slot * D}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1}, -1, 0, false);
}

void lfm3_npu_sequence::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t byte_offset) {
    const uint32_t w_offset = (uint32_t)(byte_offset / sizeof(bf16));
    const uint32_t columns = 4;
    const uint32_t a_block_size = (uint32_t)QB_ELEMS;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout / m / cores % 2 == 0);
    for (size_t round = 0; round < Dout / m / cores; round++) {
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++) {
            seq->npu_dma_memcpy_nd(sizeof(bf16), w_proj_arg_idx, MM2S, mvm_tiles[col], npu_bd_id(bd_1 + bd_offset), it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + w_offset)},
                {1, 1, 1, blocks_per_row * 2 * a_block_size}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
            seq->npu_dma_memcpy_nd(sizeof(bf16), w_proj_arg_idx, MM2S, mvm_tiles[col], npu_bd_id(bd_2 + bd_offset), it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + w_offset)},
                {1, 1, 1, blocks_per_row * 2 * a_block_size}, {0, 0, 0, 1}, -1, 0, true, aggressive_cache);
        }
        if (round > 0) {
            for (uint32_t col = 0; col < columns; col++) {
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++) {
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}

void lfm3_npu_sequence::gen_conv_layer_seq(npu_sequence* seq, const uint32_t L) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    const int rtp_lock_id = 6;
    assert(L > 0);
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(proj_tiles[i], proj_addr, 0);                                      // IS_ATTN = 0: conv layer
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * rtp_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_addr, 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rtp_lock_id, 1);
    seq->rtp_write(rope_tile, rope_addr, 0);

    _send_x(seq);
    _send_rms_weights(seq);
    _send_conv_cache(seq, L);
    _send_rope_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(sizeof(bf16), xy_arg_idx, S2MM, xyr_tile, bd_10, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1}, -1, 0, true);

    _move_weights(seq, 3 * (size_t)D, D, desc->slot0_off());          // in_proj
    _receive_conv_cache(seq, L);
    _move_weights(seq, D, D, desc->slot1_off());                       // out_proj
    _move_weights(seq, 2 * (size_t)I, D, desc->upgate_off());          // up|gate band
    _move_weights(seq, D, I, desc->down_off());

    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);
    seq->npu_preemption(0);
    seq->cmds2seq();
}

void lfm3_npu_sequence::gen_lm_head_seq(npu_sequence* seq) {
    static constexpr int lm_head_y_arg_idx = 0;
    static constexpr int lm_head_w_arg_idx = 1;
    static constexpr int lm_head_x_arg_idx = 2;
    static constexpr int lm_head_rms_arg_idx = 3;
    (void)lm_head_w_arg_idx;
    constexpr size_t CT_lock_address_base = 0x000001F000;
    const int rtp_lock_id = 6;
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(proj_tiles[i], proj_addr, 2);                                      // IS_ATTN = 2: lm_head
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * rtp_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_addr, 2);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rtp_lock_id, 1);

    seq->npu_dma_memcpy_nd(sizeof(bf16), lm_head_x_arg_idx, MM2S, xyr_tile, bd_0, it_channel_0,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1}, 0, 0, false, aggressive_cache);
    // the rms tile's weight BD is 4 x D words long: the final norm weight is padded to that
    seq->npu_dma_memcpy_nd(sizeof(bf16), lm_head_rms_arg_idx, MM2S, xyr_tile, bd_1, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D * NORMS_PER_LAYER}, {0, 0, 0, 1}, 0, 0, false, aggressive_cache);
    // logits: the padded vocabulary (the kernel produces whole D-wide rounds)
    seq->npu_dma_memcpy_nd(sizeof(bf16), lm_head_y_arg_idx, S2MM, IT0, bd_15, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, vocab_padded}, {0, 0, 0, 1}, -1, 0, true);

    _move_weights(seq, vocab_padded, D, 0);

    seq->npu_dma_wait(IT0, S2MM, it_channel_1);
}


// ---------------------------------------------------------------------------------------------------------------
// attention layer
// ---------------------------------------------------------------------------------------------------------------

// The new token's K and V rows go to position L-1 of the cache; the rope tile emits K (first half of the kv heads), V
// (first half), K (second half), V (second half) as packets 14 (to IT0) and 15 (to IT7). The q|k|v projection weights are
// streamed in between (they produce what the rope tile consumes).
void lfm3_npu_sequence::_receive_kv_cache(npu_sequence* seq, const uint32_t L) {
    const uint32_t row = DK / 2;                          // elements per token in one band
    const uint32_t off = (L - 1) * row;
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, IT0, bd_0, it_channel_0,
        {0, 0, 0, off}, {1, 1, 1, row}, {0, 0, 0, 1}, -1, 0, true);
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, IT7, bd_0, it_channel_0,
        {0, 0, 0, off + (uint32_t)V03_offset}, {1, 1, 1, row}, {0, 0, 0, 1}, -1, 0, true);
    _move_weights(seq, DQ + DK + DV, D, desc->q_off());   // q|k|v band at the front of slot 0
    seq->npu_dma_wait(IT0, S2MM, it_channel_0);
    seq->npu_dma_wait(IT7, S2MM, it_channel_0);
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, IT0, bd_0, it_channel_0,
        {0, 0, 0, off + (uint32_t)K47_offset}, {1, 1, 1, row}, {0, 0, 0, 1}, -1, 0, true);
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, IT7, bd_0, it_channel_0,
        {0, 0, 0, off + (uint32_t)V47_offset}, {1, 1, 1, row}, {0, 0, 0, 1}, -1, 0, true);
    seq->npu_dma_wait(IT0, S2MM, it_channel_0);
    seq->npu_dma_wait(IT7, S2MM, it_channel_0);
}

// Stream `count` tokens starting at position `start` of every band to the two memtiles (K to the QK cores, V to the KV
// cores). The attention cores work on chunks of 16 tokens and mask the tail via their RTP length, so the move is padded to a
// multiple of 16; the padding rows are masked.
void lfm3_npu_sequence::_move_kv_cache(npu_sequence* seq, const uint32_t start, const uint32_t count) {
    const uint32_t row = DK / 2;
    const uint32_t padded = (count + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = padded * row;
    const uint32_t s = start * row;
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, MM2S, kv_tiles[0], bd_0, it_channel_0,
        {0, 0, 0, (uint32_t)K03_offset + s}, {1, 1, 1, data2move}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, MM2S, kv_tiles[1], bd_0, it_channel_0,
        {0, 0, 0, (uint32_t)K47_offset + s}, {1, 1, 1, data2move}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, MM2S, kv_tiles[0], bd_1, it_channel_1,
        {0, 0, 0, (uint32_t)V03_offset + s}, {1, 1, 1, data2move}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
    seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, MM2S, kv_tiles[1], bd_1, it_channel_1,
        {0, 0, 0, (uint32_t)V47_offset + s}, {1, 1, 1, data2move}, {0, 0, 0, 1}, -1, 0, false, aggressive_cache);
}

void lfm3_npu_sequence::gen_attn_layer_seq(npu_sequence* seq, const uint32_t L, const uint32_t window) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    const int rtp_lock_id = 6;
    assert(L > 0 && L <= MAX_L);
    const uint32_t count = (window && L > window) ? window : L;     // tokens the query attends to
    const uint32_t start = L - count;
    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int i = 0; i < 4; i++) {
        seq->rtp_write(attn_qk_tiles[i], l_qk_address, count);
        seq->rtp_write(attn_kv_tiles[i], l_kv_address, count);
    }
    for (int i = 0; i < 16; i++) {
        seq->rtp_write(proj_tiles[i], proj_addr, 1);                                      // IS_ATTN = 1
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * rtp_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_addr, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rtp_lock_id, 1);
    seq->rtp_write(rope_tile, rope_addr, 1);

    _send_x(seq);
    _send_rms_weights(seq);
    _send_rope_weights(seq);

    seq->npu_dma_memcpy_nd(sizeof(bf16), xy_arg_idx, S2MM, xyr_tile, bd_10, it_channel_1,
        {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1}, -1, 0, true);

    _receive_kv_cache(seq, L);                                          // also streams the q|k|v weights
    _move_kv_cache(seq, start, count);
    _move_weights(seq, D, DQ, desc->slot1_off());                       // o_proj
    _move_weights(seq, 2 * (size_t)I, D, desc->upgate_off());
    _move_weights(seq, D, I, desc->down_off());

    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);
    seq->npu_preemption(0);
    seq->cmds2seq();
}
