// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "dequant.hpp"

struct Dequant::Impl{
private:
    static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr u32 total_cols = 8;
    static constexpr u32 total_rows = 4;

    static constexpr int w_out_arg_idx = 0;
    static constexpr int qw_in_arg_idx = 1;

    static constexpr int m_tile_q4 = 32;
    static constexpr int k_tile_q4 = 256;

    static constexpr uint32_t block_size_in_byte_q4 = m_tile_q4 * k_tile_q4 * 5 / 8;


    static constexpr int m_tile_q8 = 32;
    static constexpr int k_tile_q8 = 128;
    static constexpr uint32_t block_size_in_byte_q8 = m_tile_q8 * k_tile_q8 * (10)/8;


    static constexpr int quant_block_col_stride = 2;

    static constexpr int desired_k_dequant = 512;
    static constexpr int desired_m_dequant = 128;

    static constexpr int glu_slice = 1024;
    static constexpr int gate_up_m_interleave_size = glu_slice / 2;
    
public:
    /// @brief dequant output mode, as the quantized weight of UP and GATE are interleaved in memory, now we want to seperate them.
    /// @note NORMAL_DEQUANT: normal dequant output
    /// @note UP_MATRIX: up projection matrix output
    /// @note GATE_MATRIX: gate projection matrix output
    typedef enum: int{
        NORMAL_DEQUANT = 0,
        UP_MATRIX      = 1,
        GATE_MATRIX    = 2
    } dequant_output_mode_t;

    Impl(){}
    Impl(LM_Config& config);
    ~Impl();
    /// @brief generate the dequant sequence
    /// @param seq: the sequence
    /// @param D_in: input dimension of the projection weight
    /// @param D_out: output dimension of the projection weight
    /// @param weight_offset: the weight offset in byte
    /// @param mode: dequant output mode
    void generate_dequant_q4_1_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset, dequant_output_mode_t mode);
    void generate_dequant_q80_packed_in_q4nx_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset, dequant_output_mode_t mode);
    LM_Config config;
};
