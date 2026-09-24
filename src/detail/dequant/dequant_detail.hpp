#pragma once
#include "modules/dequant.hpp"

struct Dequant::Impl{
private:
    static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr u32 total_cols = 8;
    static constexpr u32 total_rows = 4;

    static constexpr int w_out_arg_idx = 0;
    static constexpr int qw_in_arg_idx = 1;

    static constexpr int m_tile_q4 = 32;
    static constexpr int k_tile_q4 = 256;

    static constexpr uint32_t block_size_in_byte_q4_0 = ((m_tile_q4 * k_tile_q4 * 4.5) / 8.0);
    static constexpr uint32_t block_size_in_byte_q4_1 = m_tile_q4 * k_tile_q4 * 5 / 8;

    static constexpr int m_tile_q8 = 32;
    static constexpr int k_tile_q8 = 256;

    static constexpr uint32_t block_size_in_byte_q8_0 = (m_tile_q8 * k_tile_q8 * (8.5) )/8.0;
    static constexpr uint32_t block_size_in_byte_q8_1 = (m_tile_q8 * k_tile_q8 * (9) )/8.0;
    static constexpr int quant_block_col_stride = 2;
    static constexpr int quant_block_interleave_byte_size = 512;

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

    void reorder_cpy(u8 *dst, buffer<u8> &src,
        Dequant::quant_block_t quant_block_type,
        const int quant_matrix_row,
        const int quant_matrix_col,
        const int vertical_blocks,
        const int vetrical_block_interleave_byte_size)
    {
        int a_block_size = 0;
        int block_col_size = 0;
        int block_row_size = 0;
        switch(quant_block_type){
            case Q4_1:
                a_block_size = block_size_in_byte_q4_1;
                block_col_size = k_tile_q4;
                block_row_size = m_tile_q4;
                break;
            case Q8_0:
                a_block_size= block_size_in_byte_q8_0;
                block_col_size = k_tile_q8;
                block_row_size = m_tile_q8;
                break;
            default:
                std::cerr << "Unsupport type for now";
                exit(-1);
                break;
        }

        assert( quant_matrix_col % block_col_size== 0);
        const int blocks_per_row = quant_matrix_col / block_col_size;

        assert(quant_matrix_row%(block_row_size* vertical_blocks) == 0 );

        const int rows = src.size() / a_block_size / blocks_per_row;

        u8 *dst_ptr = dst;
        std::vector<u8 *> src_ptr(vertical_blocks);
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
        }
        for (int r = 0; r < rows; r += vertical_blocks)
        {
            for (int c = 0; c < blocks_per_row; c++)
            {
                for (int i = 0; i < vertical_blocks; i++)
                {
                    memcpy(dst_ptr, src_ptr[i], a_block_size);
                    dst_ptr += a_block_size;
                    src_ptr[i] += a_block_size;
                }
            }
            for (int i = 0; i < vertical_blocks; i++)
            {
                src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
                if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
                {
                    src_ptr[i] = src.data(); // useless padding
                }
            }
        }

        // At this step, the blocks are now reorder with vertical blocks

        // For example
        // IF previous are row-block order
        /*
            [A, B, C
            D, E, F]
            Where blocks are ordered as A, B, C, D, E, F

            With the vertical_blocks =2,
            blocks are reorder as A, D, B, E, C, F

        */

        if(vetrical_block_interleave_byte_size <=0){
            return ; // no need this step
        }
        // Apply vetrical_block_interleave_byte_size reorder

        // From example above, now A, D blocks are continousy in memory at block level

        // However, we want to do a byte-block level mixing
        /**
            For example, If A, D block are Block size of 2K and vetrical_block_interleave_byte_size = 1024

            In memory, the data are layout as A(1-1204) A(1025-2048), D(1-1024), D(1025-2048)

            After the block level reorder, we have
            A(1-1204), D(1-1024), A(1025-2048), D(1025-2048)

        */
        size_t num_data_block = (quant_matrix_row/block_row_size) * (quant_matrix_col/block_col_size);
        std::vector<uint8_t> temp_buffer(vertical_blocks*a_block_size );

        size_t num_byte_data_chunk  =  (a_block_size) / vetrical_block_interleave_byte_size;
        assert(a_block_size % vetrical_block_interleave_byte_size == 0);

        for(int i = 0; i < num_data_block; i+= vertical_blocks){
            uint8_t* cur_ptr = dst + i*a_block_size;
            memcpy( temp_buffer.data(), cur_ptr, temp_buffer.size() );

            uint8_t* chunk_dst_ptr = cur_ptr;

            for(int byte_chunk_idx = 0; byte_chunk_idx <num_byte_data_chunk; byte_chunk_idx++ ){

                for(int b_idx = 0; b_idx < vertical_blocks; b_idx++){

                    uint8_t* chunk_src_ptr = temp_buffer.data() +\
                    b_idx*a_block_size+  byte_chunk_idx*vetrical_block_interleave_byte_size;

                    memcpy(chunk_dst_ptr,chunk_src_ptr, vetrical_block_interleave_byte_size );
                    chunk_dst_ptr+= vetrical_block_interleave_byte_size;
                }
            }
        }
    }
};
