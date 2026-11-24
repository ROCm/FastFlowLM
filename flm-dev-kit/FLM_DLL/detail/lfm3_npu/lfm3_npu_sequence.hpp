// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once
/// \file lfm3_npu_sequence.hpp
/// \brief Instruction sequences of the LFM3 decode layer (FLM_Xclbin/LiquidAI/lfm3_npu_bin, MODEL_TYPE LFM3_1_7B).
///
/// One xclbin runs one transformer layer for one token; the host rebuilds the DMA sequence per token. Same floorplan and
/// scheme as the LFM2 engine (lfm2_npu_sequence), with these differences:
///   - four norm weights per layer (operator, post_operator, ffn, post_ffn): the rms tile holds all four, one DMA of 4 x D
///   - conv width 7: the conv taps are 7 x D, the history is a ring of 6 rows
///   - lm_head vocab padded to a multiple of D (kernel loops over whole D-wide rounds)
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "lfm3_model_def.hpp"

struct lfm3_npu_sequence {
    static constexpr npu_tiles kv_tiles[] = {IT0, IT7};
    static constexpr npu_tiles xyr_tile = IT1;
    static constexpr npu_tiles mvm_tiles[] = {IT2, IT3, IT4, IT5};
    static constexpr npu_tiles attn_qk_tiles[] = {CT00, CT20, CT07, CT27};
    static constexpr npu_tiles attn_kv_tiles[] = {CT10, CT30, CT17, CT37};
    static constexpr npu_tiles proj_tiles[] = {CT02, CT12, CT22, CT32, CT03, CT13, CT23, CT33,
                                               CT04, CT14, CT24, CT34, CT05, CT15, CT25, CT35};
    static constexpr npu_tiles rms_tile = CT01;
    static constexpr npu_tiles rope_tile = CT21;
    // argument order of the layer kernel: x/y, projection weights, rms weights, rope (conv taps), kv cache / conv ring
    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;

    static constexpr int m = 32;
    static constexpr int k = 256;
    static constexpr int L_CHUNK = 16;
    static constexpr int NORMS_PER_LAYER = 4;

    // NOTE: rms_addr depends on the size of the rms tile's weight buffer (4 x D here).
    // RTP addresses of this xclbin: read them from the build (tools/rtp.py on bitstream/from_iron/flowArch.mlir.prj/input_with_addresses.mlir)
    static constexpr uint32_t l_qk_address = 5120;
    static constexpr uint32_t proj_addr = 33280;
    static constexpr uint32_t l_kv_address = 34816;
    static constexpr uint32_t rms_addr = 5120;
    static constexpr uint32_t rope_addr = 46080;

    static constexpr size_t QB_ELEMS = (size_t)lfm3_cpu::QB_BYTES / sizeof(bf16);   ///< bf16 elements per 32x256 block

    uint32_t D, DH, DK, DV, DQ, I, vocab_padded, conv_hist_rows;
    uint32_t MAX_L = 0;
    size_t K03_offset = 0, K47_offset = 0, V03_offset = 0, V47_offset = 0;   ///< element offsets of the four KV bands
    void set_max_length(uint32_t max_l);
    const lfm3_desc* desc = nullptr;

    lfm3_npu_sequence(const lfm3_desc& d, uint32_t max_l);

    /// padded vocabulary of the lm_head (multiple of D)
    static uint32_t pad_vocab(uint32_t vocab, uint32_t D) { return (vocab + D - 1) / D * D; }

    void gen_conv_layer_seq(npu_sequence* seq, const uint32_t L);   ///< L = tokens in the context including the new one
    /// Attention layer for the token at position L-1. `window` = 0: attend to all L tokens, else only the last `window`.
    /// Nothing in the sequence depends on the layer: the sigmoid gates of the query heads travel in the rope buffer.
    void gen_attn_layer_seq(npu_sequence* seq, const uint32_t L, const uint32_t window);
    void gen_lm_head_seq(npu_sequence* seq);

private:
    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const uint32_t L);
    void _move_kv_cache(npu_sequence* seq, const uint32_t start, const uint32_t count);
    void _send_conv_cache(npu_sequence* seq, const uint32_t L);
    void _receive_conv_cache(npu_sequence* seq, const uint32_t L);
    /// stream the Q4 weights of a [Dout x Din] matrix that starts `byte_offset` bytes into the proj buffer through the 16 matvec cores
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t byte_offset);
};
