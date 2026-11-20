// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma_text_dequant_detail.hpp"

// constructors
GemmaTextDequant::Impl::Impl(LM_Config& config) : config(config){
    // Initialize any dequant-specific configuration here
}

GemmaTextDequant::Impl::~Impl() = default;

// methods
/// @brief generate the dequant sequence
/// @param seq: the sequence
/// @param D_in: input dimension
/// @param D_out: output dimension
/// @param weight_offset: the weight offset
void GemmaTextDequant::Impl::generate_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset){
    // N is din, M is dout
    const int rounds = D_out / 16 / total_cols / 2;;
    seq->clear_cmds();
    seq->npu_preemption(0);
    uint32_t qw_size = 16 * 128 * 5 / 8 / 4;
    for (int round = 0; round < rounds; round++){
        int bd_offset = (round % 2) * 8;
        npu_bd_id w_out_bd = npu_bd_id(0 + bd_offset);
        npu_bd_id wq_in_up_bd = npu_bd_id(1 + bd_offset);
        npu_bd_id wq_in_down_bd = npu_bd_id(2 + bd_offset);

        for (int c = 0; c < total_cols; c++){
            uint32_t qw_offset = (round * 8 + c) * qw_size * D_in / 128 * 2 + weight_offset / 4;
            uint32_t q_offset = (round * 8 + c) * 32 * D_in;

            seq->npu_dma_memcpy_nd(
                2, 0,
                S2MM, IT[c],
                w_out_bd, it_channel_0,
                {0, 0, 0, q_offset},
                {1, 1, 1, 32 * D_in},
                {0, 0, 0, 1},
                -1, 0, true
            );

            seq->npu_dma_memcpy_nd(
                4, 1,
                MM2S, IT[c],
                wq_in_up_bd, it_channel_0,
                {0, 0, 0, qw_offset},
                {1, 1, 1, qw_size * D_in / 128},
                {0, 0, 0, 1},
                -1, 0, false
            );

            seq->npu_dma_memcpy_nd(
                4, 1,
                MM2S, IT[c],
                wq_in_down_bd, it_channel_1,
                {0, 0, 0, qw_offset + qw_size * D_in / 128},
                {1, 1, 1, qw_size * D_in / 128},
                {0, 0, 0, 1},
                -1, 0, false
            );

            if (round > 0){
                seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
            }
        }
    }// end round
    for (int c = 0; c < total_cols; c++){
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

// wrappers
GemmaTextDequant::GemmaTextDequant(LM_Config& config) : _impl(new Impl(config)){}
GemmaTextDequant::~GemmaTextDequant(){
    delete _impl;
}

void GemmaTextDequant::generate_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset){
    _impl->generate_seq(seq, D_in, D_out, weight_offset);
}
