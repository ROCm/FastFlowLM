// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_npu_sequence.cpp
/// \brief instruction generation for the hunyuan-dense engine
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \note Two overlays are driven from here; see hunyuan_prefill.hpp for the
///       HUNYUAN_*_ON_NPU switches.
///
///       gen_layer_seq and gen_lm_head_seq target the fused *decode* overlay
///       built from FLM_Xclbin/Hy_MT2_1.8B/decoding with
///       MODEL_TYPE=HUNYUAN_DENSE_1_8B. Its floorplan is model-independent
///       (iron/layer_aie.py); what this model changes is only the rope tile's
///       object file (rope_hunyuan.o, QK norm after RoPE) and the projection
///       quantization (Q4_K, 4736-byte blocks; Q8_0 lm_head as padded half pairs).
///
///       The prefill runs on prefill_overlay.xclbin; its instruction sequences
///       are in hunyuan_overlay.hpp (gen_dq_run, gen_attn_run).
#include "hunyuan_npu_sequence.hpp"
#include <cassert>

/// @brief cache the shapes, the weight offsets and the kv-cache quadrants
hunyuan_npu_sequence::hunyuan_npu_sequence(hunyuan_desc& desc, LM_Config config, uint32_t MAX_L){
    (void)config;
    this->desc = &desc;

    // weight_desc_t::offset is a *byte* offset into the proj_weights BO; the DMA
    // descriptors below address it in bf16 elements.

    // q, k and v are adjacent in that order and every one of DQ, DK, DV is a
    // multiple of the 512-row round, so the three matrices stream as one move.
    assert(desc.layer_desc.attn_v.offset ==
           desc.layer_desc.attn_k.offset + get_quantization_byte_size((size_t)desc.DK * desc.D, desc.layer_desc.attn_k.dtype));
    assert(desc.layer_desc.attn_q.offset ==
           desc.layer_desc.attn_v.offset + get_quantization_byte_size((size_t)desc.DK * desc.D, desc.layer_desc.attn_k.dtype));

    this->set_max_length(MAX_L);
}

hunyuan_npu_sequence::~hunyuan_npu_sequence(){}

void hunyuan_npu_sequence::set_max_length(const uint32_t MAX_L){
    this->MAX_L = MAX_L;
    this->kv_cache_size = (uint32_t)this->desc->get_kv_cache_size(MAX_L);
    // _move_kv_cache reads ceil(L / L_CHUNK) * L_CHUNK rows out of a quadrant,
    // so a MAX_L that is not a whole number of chunks would read past its end.
    assert(MAX_L % L_CHUNK == 0);
    this->K03_offset = 0;
    this->K47_offset = this->kv_cache_size / 4;
    this->V03_offset = this->kv_cache_size / 2;
    this->V47_offset = 3 * (size_t)this->kv_cache_size / 4;
    const size_t vb_block = (size_t)(MAX_L / L_CHUNK) * 2 * v_bfp16_head_chunk_bytes() / sizeof(bf16);
    this->Kb03_offset = this->kv_cache_size;
    this->Kb47_offset = this->kv_cache_size + vb_block;
    this->Vb03_offset = this->kv_cache_size + 2 * vb_block;
    this->Vb47_offset = this->kv_cache_size + 3 * vb_block;
    this->kv_buffer_size = this->kv_cache_size + 4 * vb_block;
}

// ---------------------------------------------------------------------------
// decode sequence building blocks
// ---------------------------------------------------------------------------

/// @brief the single hidden-state row into the rms tile
void hunyuan_npu_sequence::_send_x(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// @brief input_layernorm at 0 and post_attention_layernorm at +D, in one go
void hunyuan_npu_sequence::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D * 2},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// @brief cos[DH/2] | sin[DH/2] | q_norm[DH] | k_norm[DH] into the rope tile
/// @note  3 * DH, not DH: this architecture has per-head Q/K norms, which the
///        rope tile applies *after* the rotation (kernels/rope_hunyuan.cc).
void hunyuan_npu_sequence::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->DH * 3},
        {0, 0, 0, 1},
        1, 0, false, aggressive_cache
    );
}

/// @brief write the token's new K and V rows back into the four cache quadrants
/// @note  The rope tile emits K on one shim and V on the other, each split in
///        two halves of num_kv_heads/2 heads. The qkv weight move is issued
///        between the two halves so the array is never idle waiting on it.
void hunyuan_npu_sequence::_receive_kv_cache(npu_sequence* seq, const int L){
    const uint32_t heads_per_tile = desc->num_kv_heads / 2;
    const uint32_t row = heads_per_tile * desc->DH;   // = DK / 2
    const uint32_t kv_cache_offset = (uint32_t)(L - 1) * row;

    // K heads 0..1 and V heads 0..1
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), kv_cache_arg_idx, S2MM, attn_tiles[0], bd_0, it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)K03_offset},
        {1, 1, 1, row},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), kv_cache_arg_idx, S2MM, attn_tiles[1], bd_0, it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V03_offset},
        {1, 1, 1, row},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // K heads 2..3 and V heads 2..3
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), kv_cache_arg_idx, S2MM, attn_tiles[0], bd_1, it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)K47_offset},
        {1, 1, 1, row},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), kv_cache_arg_idx, S2MM, attn_tiles[1], bd_1, it_channel_0,
        {0, 0, 0, kv_cache_offset + (uint32_t)V47_offset},
        {1, 1, 1, row},
        {0, 0, 0, 1},
        -1, 0, true
    );


    // the band is packed K|V|Q, so attn_k carries the base offset of all three
    _move_weights(seq, desc->DQ + desc->DK + desc->DV, desc->D, desc->layer_desc.attn_k);

    seq->npu_dma_wait(attn_tiles[0], S2MM, it_channel_0);
    seq->npu_dma_wait(attn_tiles[1], S2MM, it_channel_0);

    seq->npu_dma_wait(attn_tiles[0], S2MM, it_channel_0);
    seq->npu_dma_wait(attn_tiles[1], S2MM, it_channel_0);
}

/// @brief stream a Dout x Din quantized matrix out to the 16 projection cores
/// @param w_offset bf16-element offset of the matrix inside proj_weights
/// @note  Each round feeds all 16 cores one 32-row band each, two cores per
///        shim descriptor, so a round covers 512 output rows. The two channels
///        per shim are double buffered: the wait for round r-1 is issued only
///        after round r has been posted.
void hunyuan_npu_sequence::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight, const size_t chunk_bytes){
    // the descriptor's offset is in bytes; every DMA below counts in bf16 elements
    const uint32_t w_offset = (uint32_t)(weight.offset / sizeof(bf16));
    constexpr uint32_t columns = 4;
    constexpr uint32_t cores = columns * 4;
    const uint32_t elems = m * k;
    // one 32x256 chunk in bf16 words: 4736 B for q4_k; the lm_head passes its
    // padded Q8_0 half pair (hunyuan_desc::lm_head_chunk_bytes)
    const size_t bytes = chunk_bytes ? chunk_bytes : get_quantization_byte_size(elems, weight.dtype);
    const uint32_t a_block_size = (uint32_t)(bytes / sizeof(bf16));
    const uint32_t blocks_per_row = (uint32_t)Din / k;

    assert(Din % k == 0);
    assert(Dout % (m * cores) == 0);
    assert(Dout / m / cores > 0);

    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), w_proj_arg_idx, MM2S, mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset), it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), w_proj_arg_idx, MM2S, mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset), it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
                seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}

/// @brief stream the committed part of the kv cache into the attention tiles
/// @note  The finished chunks of K and V as bfp16 (the copy the host keeps), then the
///        chunk holding the new row as bf16 through shim 5: its K rows, then its V rows.
///        Rows of that chunk past L are zero (every cache buffer is zeroed on allocation
///        and on clear_context()) and masked by the Q·K core.
void hunyuan_npu_sequence::_move_kv_cache(npu_sequence* seq, const size_t L){
    const uint32_t heads_per_tile = desc->num_kv_heads / 2;
    const uint32_t row = desc->DH * heads_per_tile;
    const uint32_t n_bfp16 = v_bfp16_chunks((uint32_t)L);

    const size_t kb_src[2] = {Kb03_offset, Kb47_offset};
    const size_t vb_src[2] = {Vb03_offset, Vb47_offset};
    const size_t k_src[2]  = {K03_offset, K47_offset};
    const size_t v_src[2]  = {V03_offset, V47_offset};
    const npu_bd_id k_tail_bds[2] = {bd_0, bd_2};
    const npu_bd_id v_tail_bds[2] = {bd_1, bd_3};
    const npu_it_channel tail_chans[2] = {it_channel_0, it_channel_1};

    if (n_bfp16 > 0){
        const uint32_t b_elems = (size_t)n_bfp16 * heads_per_tile * v_bfp16_head_chunk_bytes() / sizeof(bf16);
        for (int col = 0; col < 2; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_cache_arg_idx, MM2S, attn_tiles[col], bd_0, it_channel_0,
                {0, 0, 0, (uint32_t)kb_src[col]},
                {1, 1, 1, b_elems},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16), kv_cache_arg_idx, MM2S, attn_tiles[col], bd_1, it_channel_1,
                {0, 0, 0, (uint32_t)vb_src[col]},
                {1, 1, 1, b_elems},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
        }
    }
    const size_t tail_row = (size_t)n_bfp16 * L_CHUNK * row;
    for (int col = 0; col < 2; col++){
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_idx, MM2S, v_tail_tile, k_tail_bds[col], tail_chans[col],
            {0, 0, 0, (uint32_t)(k_src[col] + tail_row)},
            {1, 1, 1, (uint32_t)L_CHUNK * row},
            {0, 0, 0, 1},
            -1, 0, false, aggressive_cache
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_idx, MM2S, v_tail_tile, v_tail_bds[col], tail_chans[col],
            {0, 0, 0, (uint32_t)(v_src[col] + tail_row)},
            {1, 1, 1, (uint32_t)L_CHUNK * row},
            {0, 0, 0, 1},
            -1, 0, false, aggressive_cache
        );
    }
}

// ---------------------------------------------------------------------------
// generators
// ---------------------------------------------------------------------------

/// @brief one decode step of one layer: rms -> qkv -> rope+qk norm -> mha ->
///        o proj -> rms -> glu -> down proj, with the result written back over
///        x[0, D).
/// @param L number of kv rows the attention tiles must attend to, i.e. the
///          context length including the token being decoded.
void hunyuan_npu_sequence::gen_layer_seq(npu_sequence* seq, const uint32_t L){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    seq->clear_cmds();

    _send_x(seq);
    _send_rms_weights(seq);
    _send_rope_rms_weights(seq);

    // post the descriptor that takes the layer output back into x[0, D) before
    // any of the weight traffic, so the array can retire it as soon as it lands
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        xyr_tile,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)desc->D},
        {0, 0, 0, 1},
        -1, 0, true
    );

    _receive_kv_cache(seq, (int)L);   // also moves the fused q|k|v weights
    _move_kv_cache(seq, L);
    _move_weights(seq, desc->D, desc->DQ, desc->layer_desc.attn_output);
    // up and gate are interleaved into one 2 * intermediate_size band based at ffn_up
    _move_weights(seq, 2 * (size_t)desc->intermediate_size, desc->D, desc->layer_desc.ffn_up);
    _move_weights(seq, desc->D, desc->intermediate_size, desc->layer_desc.ffn_down);

    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);

    seq->cmds2seq();
}

/// @brief On decoding, these parameters are never changed so only update once
/// @param L number of kv rows the attention tiles must attend to, i.e. the
///          context length including the token being decoded.
void hunyuan_npu_sequence::gen_layer_rtp_seq(npu_sequence* seq, const uint32_t L){
    assert(L <= MAX_L + 1);
    assert(L > 0);
    seq->clear_cmds();
    seq->npu_preemption(0);

    for (int i = 0; i < 4; i++){
        seq->rtp_write(attn_qk_tiles[i], l_qk_address, L);
        seq->rtp_write(attn_qk_tiles[i], l_qk_address + 4, v_bfp16_chunks(L));
        seq->rtp_write(attn_kv_tiles[i], l_kv_address, L);
        seq->rtp_write(attn_kv_tiles[i], l_kv_address + 4, v_bfp16_chunks(L));
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * rtp_available_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_addr, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rtp_available_lock_id, 1);

    seq->cmds2seq();
}

/// @brief the lm_head matvec, on the *same* layer.xclbin as the decode layer
/// @note  The overlay has no separate lm_head bitstream: writing 0 instead of 1
///        to the proj/rms `is_attn` RTPs turns the 16 proj cores plus the rms
///        core into a final-norm + [VOCAB_SIZE x D] matvec. Buffer arguments
///        are rebound accordingly -- arg0 = logits out, arg1 = lm_head weights,
///        arg2 = x, arg3 = the model.norm weight -- so the rms core does the
///        final norm on the array and the host must hand it the *un*-normalized
///        hidden state. See llama_npu / nanbeige_npu, which do the same.
void hunyuan_npu_sequence::gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D){
    static constexpr int lm_head_y_arg_idx   = 0;
    static constexpr int lm_head_x_arg_idx   = 2;
    static constexpr int lm_head_rms_arg_idx = 3;

    // the weight stream still comes in on w_proj_arg_idx (1); the lm_head weights
    // were reordered with V=2 like the proj weights, in 9472 B half-pair chunks.
    assert((size_t)VOCAB_SIZE % (m * 16) == 0);
    assert((size_t)D % k == 0);

    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_addr, 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * rtp_available_lock_id, 1);
    }
    seq->rtp_write(rms_tile, rms_addr, 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rtp_available_lock_id, 1);

    // logits out
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), lm_head_y_arg_idx,
        S2MM, logits_out_tile, bd_15, it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)VOCAB_SIZE},
        {0, 0, 0, 1},
        -1, 0, true
    );
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), lm_head_x_arg_idx,
        MM2S, xyr_tile, bd_0, it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
    // send the final rms weight
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), lm_head_rms_arg_idx,
        MM2S, xyr_tile, bd_1, it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );

    this->_move_weights(seq, (size_t)VOCAB_SIZE, (size_t)D, desc->lm_head, desc->lm_head_chunk_bytes());

    seq->npu_dma_wait(logits_out_tile, S2MM, it_channel_1);
    seq->cmds2seq();
}
