#include "dequant_detail.hpp"

// constructors
Dequant::Impl::Impl(LM_Config& config) : config(config){
    // Initialize any dequant-specific configuration here
}

Dequant::Impl::~Impl() = default;

// methods
/// @brief generate the dequant sequence
/// @param seq: the sequence
/// @param D_in: input dimension of the projection weight
/// @param D_out: output dimension of the projection weight
/// @param weight_offset: the weight offset in byte
/// @param mode: dequant output mode
void Dequant::Impl::generate_dequant_q80_packed_in_q4nx_seq(npu_sequence* seq_ptr, const u32 D_in, const u32 D_out, const u32 weight_offset, dequant_output_mode_t output_mode){
    std::cout << "generate_dequant_q80_packed_in_q4nx_seq, D_in: " << D_in << ", D_out: " << D_out << ", weight_offset: " << weight_offset << std::endl;
    if (D_in % k_tile_q4 != 0) {
        std::cerr << "D_in % k_tile_q4 != 0" << std::endl;
        exit(1);
    }

    int bd_wait_counter[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // although each data block is in mxk block, but the data block could be reorder in col-stride on block view
    /*
        For example, quant_block_col_stride = 2 means

        //This is the logical view of the data block, each block of m_tile_q4 x k_tile_q4
        [block0, block1, ...... blockD,
        blockD+1, blockD+2, ......
        ]

        But in memory order, the data block is arrange as block0, blockD+1, block1, blockD+2 ....

    */

    if(D_in % desired_k_dequant != 0){
        std::cerr << "D_in % desired_k_dequant != 0" << std::endl;
        exit(1);
    }

    const uint32_t blocks_per_row = D_in / k_tile_q4  * 2;
    std::cout << "blocks_per_row: " << blocks_per_row << std::endl;

    if(D_out % desired_m_dequant != 0 ){
        std::cerr << "D_out % desired_m_dequant != 0" << std::endl;
        exit(1);
    }

    const int quant_in_per_column =  (desired_m_dequant / m_tile_q4) * blocks_per_row * block_size_in_byte_q4_1;
    const int total_column_rounds = D_out / (desired_m_dequant);

    const int row_per_round = desired_m_dequant * total_cols;
    // down rounds, go though D_out
    const int down_rounds = (D_out + row_per_round - 1) / row_per_round;

    npu_sequence& seq = *seq_ptr;
    seq.clear_cmds();

    uint32_t input_offset = weight_offset;

    if(output_mode == dequant_output_mode_t::GATE_MATRIX){
        input_offset += (gate_up_m_interleave_size / m_tile_q4) * blocks_per_row * block_size_in_byte_q4_1;
    }
    uint32_t gate_up_interleave_counter= 0;

    // first, the dequant of down
    for(int i = 0; i < down_rounds; i++){
        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t round_offset = i * 8 + col;
            if(round_offset < total_column_rounds){
                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S,
                    IT[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0, 0, 0, input_offset},
                    //NOTE: this for now only work if desired_m_dequant == quant_block_col_stride*m_tile_q4
                    {
                        blocks_per_row,
                        (desired_m_dequant / m_tile_q4) / quant_block_col_stride,
                        quant_block_col_stride * block_size_in_byte_q4_1 / 512,
                        512
                    },
                    {
                        quant_block_col_stride * block_size_in_byte_q4_1,
                        quant_block_col_stride * block_size_in_byte_q4_1 * blocks_per_row,
                        512,
                        1
                    },
                    -1 ,0, false
                );

                if(output_mode == dequant_output_mode_t::NORMAL_DEQUANT){
                    std::cout << "Use normal output!" << std::endl;
                    input_offset += quant_in_per_column;
                }
                else{
                    gate_up_interleave_counter++;
                    input_offset += quant_in_per_column;
                    if(gate_up_interleave_counter == (gate_up_m_interleave_size / desired_m_dequant) ){
                        gate_up_interleave_counter = 0;
                        input_offset += (gate_up_m_interleave_size / m_tile_q4) * blocks_per_row * block_size_in_byte_q4_1;
                    }
                }

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                uint32_t output_offset_0 = round_offset * desired_m_dequant  * D_in;

                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),//bf16 outpout
                    w_out_arg_idx,
                    S2MM,
                    IT[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0, 0, 0, output_offset_0},
                    {
                        (uint32_t)D_in/desired_k_dequant,
                        desired_k_dequant/k_tile_q4,
                        desired_m_dequant,
                        k_tile_q4
                    },
                    {
                        desired_m_dequant * desired_k_dequant,
                        k_tile_q4,
                        desired_k_dequant,
                        1
                    },
                    -1, 0, true,
                    aggressive_cache
                );
                bd_wait_counter[col]++;
            }
        }
        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
            bd_wait_counter[col]--;
        }
    }
    seq.cmds2seq();
}

/// @brief generate the dequant sequence
/// @param seq: the sequence
/// @param D_in: input dimension of the projection weight
/// @param D_out: output dimension of the projection weight
/// @param weight_offset: the weight offset in byte
/// @param mode: dequant output mode
void Dequant::Impl::generate_dequant_q4_1_seq(npu_sequence* seq_ptr, const u32 D_in, const u32 D_out, const u32 weight_offset, dequant_output_mode_t output_mode){
    if (D_in % k_tile_q4 != 0) {
        std::cerr << "D_in % k_tile_q4 != 0" << std::endl;
        exit(1);
    }

    int bd_wait_counter[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // although each data block is in mxk block, but the data block could be reorder in col-stride on block view
    /*
        For example, quant_block_col_stride = 2 means

        //This is the logical view of the data block, each block of m_tile_q4 x k_tile_q4
        [block0, block1, ...... blockD,
        blockD+1, blockD+2, ......
        ]

        But in memory order, the data block is arrange as block0, blockD+1, block1, blockD+2 ....

    */

    if(D_in % desired_k_dequant != 0){
        std::cerr << "D_in % desired_k_dequant != 0" << std::endl;
        exit(1);
    }

    const uint32_t blocks_per_row = D_in / k_tile_q4;

    if(D_out % desired_m_dequant != 0 ){
        std::cerr << "D_out % desired_m_dequant != 0" << std::endl;
        exit(1);
    }

    const int quant_in_per_column =  (desired_m_dequant / m_tile_q4) * blocks_per_row * block_size_in_byte_q4_1;
    const int total_column_rounds = D_out / (desired_m_dequant);

    const int row_per_round = desired_m_dequant * total_cols;
    // down rounds, go though D_out
    const int down_rounds = (D_out + row_per_round - 1) / row_per_round;

    npu_sequence& seq = *seq_ptr;
    seq.clear_cmds();

    uint32_t input_offset = weight_offset;

    if(output_mode == dequant_output_mode_t::GATE_MATRIX){
        input_offset += (gate_up_m_interleave_size / m_tile_q4) * blocks_per_row * block_size_in_byte_q4_1;
    }
    uint32_t gate_up_interleave_counter= 0;

    // first, the dequant of down
    for(int i = 0; i < down_rounds; i++){
        for(int col = 0; col < 8; col++){
            uint32_t bd_offset = (i % 2) * 8;
            uint32_t round_offset = i * 8 + col;
            if(round_offset < total_column_rounds){

                seq.npu_dma_memcpy_nd(
                    sizeof(char),
                    qw_in_arg_idx,
                    MM2S,
                    IT[col],
                    (npu_bd_id)(0+bd_offset),
                    it_channel_0,
                    {0, 0, 0, input_offset},
                    //NOTE: this for now only work if desired_m_dequant == quant_block_col_stride*m_tile_q4
                    {
                        blocks_per_row,
                        (desired_m_dequant / m_tile_q4) / quant_block_col_stride,
                        quant_block_col_stride * block_size_in_byte_q4_1 / 512,
                        512
                    },
                    {
                        quant_block_col_stride * block_size_in_byte_q4_1,
                        quant_block_col_stride * block_size_in_byte_q4_1 * blocks_per_row,
                        512,
                        1
                    },
                    -1 ,0, false
                );

                if(output_mode == dequant_output_mode_t::NORMAL_DEQUANT){
                    input_offset += quant_in_per_column;
                }
                else{
                    gate_up_interleave_counter++;
                    input_offset += quant_in_per_column;
                    if(gate_up_interleave_counter == (gate_up_m_interleave_size / desired_m_dequant) ){
                        gate_up_interleave_counter = 0;
                        input_offset += (gate_up_m_interleave_size / m_tile_q4) * blocks_per_row * block_size_in_byte_q4_1;
                    }
                }

                // Each port receive 2*Q4NX_ROWx D_Q4NX_BLOCK_PER_ROW*Q4NX_COL
                uint32_t output_offset_0 = round_offset * desired_m_dequant  * D_in;

                seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),//bf16 outpout
                    w_out_arg_idx,
                    S2MM,
                    IT[col],
                    (npu_bd_id)(1+bd_offset),
                    it_channel_0,
                    {0, 0, 0, output_offset_0},
                    {
                        (uint32_t)D_in/desired_k_dequant,
                        desired_k_dequant/k_tile_q4,
                        desired_m_dequant,
                        k_tile_q4
                    },
                    {
                        desired_m_dequant * desired_k_dequant,
                        k_tile_q4,
                        desired_k_dequant,
                        1
                    },
                    -1, 0, true,
                    aggressive_cache
                );
                bd_wait_counter[col]++;
            }
        }
        // note: for now
        for(int col = 0; col < 8; col++){
            if(bd_wait_counter[col] == 2){
                seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
                bd_wait_counter[col]--;
            }
        }
    }

    for(int col = 0; col < 8; col++){
        while(bd_wait_counter[col] != 0){
            seq.npu_dma_wait(IT[col], S2MM, it_channel_0);
            bd_wait_counter[col]--;
        }
    }
    seq.cmds2seq();
}

// wrappers
Dequant::Dequant(LM_Config& config) : _impl(new Impl(config)){}
Dequant::~Dequant(){
    delete _impl;
}
void Dequant::generate_dequant_q80_packed_in_q4nx_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset, int mode){
    _impl->generate_dequant_q80_packed_in_q4nx_seq(seq, D_in, D_out, weight_offset, (Dequant::Impl::dequant_output_mode_t)mode);
}

void Dequant::generate_dequant_q4_1_seq(npu_sequence* seq, const u32 D_in, const u32 D_out, const u32 weight_offset, int mode){
    _impl->generate_dequant_q4_1_seq(seq, D_in, D_out, weight_offset, (Dequant::Impl::dequant_output_mode_t)mode);
}

void Dequant::reorder_cpy(
    u8 *dst, buffer<u8> &src,
    quant_block_t quant_block_type,
    const int quant_matrix_row,
    const int quant_matrix_col,
    const int vertical_blocks ,
    const int vetrical_block_interleave_byte_size

){

    _impl->reorder_cpy(
        dst, src, quant_block_type, quant_matrix_row, quant_matrix_col,
        vertical_blocks, vetrical_block_interleave_byte_size
    );
}
