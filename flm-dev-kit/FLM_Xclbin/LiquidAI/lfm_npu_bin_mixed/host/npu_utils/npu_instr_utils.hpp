// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __NPU_INSTR_UTILS_HPP__
#define __NPU_INSTR_UTILS_HPP__

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include "buffer.hpp"
#include "utils/debug_utils.hpp"
#include "xrt/xrt_bo.h"
#include "instr_utils/npu_cmd.hpp"
#include "instr_utils/npu_cmd_write.hpp"
#include "instr_utils/npu_cmd_ddr.hpp"
#include "instr_utils/npu_cmd_write_dma.hpp"
#include "instr_utils/npu_cmd_issue_token.hpp"
#include "instr_utils/npu_cmd_wait.hpp"
#include "instr_utils/npu_cmd_maskwrite.hpp"

// This function is used to interperate instructions
// Useful files:
// mlir-aie/lib/Dialect/AIEX/IR/AIEXDialect.cpp
// mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp
// mlir-aie/lib/Targets/AIETargetNPU.cpp

// found in AIETargetNPU.cpp

///@brief npu device
///@note This is an enum to specify the npu device
///@warning npu1 is not supported yet
typedef enum{
    device_npu1,
    device_npu2
} npu_device;

///@brief npu tiles
///@note This is an enum to specify the npu tiles
///@note The low 4 bit is col, the high 4 bit is row
///@warning Though npu1 only has 4 cols, the enum defines 8 cols for convenience
typedef enum: uint32_t{
    IT0 = 0x00,  IT1 = 0x01,  IT2 = 0x02,  IT3 = 0x03,
    IT4 = 0x04,  IT5 = 0x05,  IT6 = 0x06,  IT7 = 0x07,
    MT0 = 0x10,  MT1 = 0x11,  MT2 = 0x12,  MT3 = 0x13,
    MT4 = 0x14,  MT5 = 0x15,  MT6 = 0x16,  MT7 = 0x17,
    CT00 = 0x20, CT01 = 0x21, CT02 = 0x22, CT03 = 0x23,
    CT04 = 0x24, CT05 = 0x25, CT06 = 0x26, CT07 = 0x27,
    CT10 = 0x30, CT11 = 0x31, CT12 = 0x32, CT13 = 0x33,
    CT14 = 0x34, CT15 = 0x35, CT16 = 0x36, CT17 = 0x37,
    CT20 = 0x40, CT21 = 0x41, CT22 = 0x42, CT23 = 0x43,
    CT24 = 0x44, CT25 = 0x45, CT26 = 0x46, CT27 = 0x47,
    CT30 = 0x50, CT31 = 0x51, CT32 = 0x52, CT33 = 0x53,
    CT34 = 0x54, CT35 = 0x55, CT36 = 0x56, CT37 = 0x57,
} npu_tiles;

///@brief get the tile
///@param row row of the tile
///@param col column of the tile
///@return the tile
///@note This is a helper function to get the tile from the row and column
inline npu_tiles get_tile(uint32_t row, uint32_t col){
    assert(col < 8 && row < 6);
    return static_cast<npu_tiles>((row << 4) | col);
}

///@brief npu it channel
///@note This is an enum to specify the npu it channel
typedef enum: uint32_t{
    it_channel_0,
    it_channel_1
} npu_it_channel;

///@brief npu ct channel
///@note This is an enum to specify the npu ct channel
typedef enum: uint32_t{
    ct_channel_0,
    ct_channel_1
} npu_ct_channel;

///@brief npu mt channel
///@note This is an enum to specify the npu mt channel
typedef enum: uint32_t{
    mt_channel_0,
    mt_channel_1,
    mt_channel_2,
    mt_channel_3,
    mt_channel_4,
    mt_channel_5
} npu_mt_channel;

///@brief npu bd id
///@note This is an enum to specify the npu bd id, there are 16 bd ids
typedef enum: uint32_t{
    bd_0,
    bd_1,
    bd_2,
    bd_3,
    bd_4,
    bd_5,
    bd_6,
    bd_7,
    bd_8,
    bd_9,
    bd_10,
    bd_11,
    bd_12,
    bd_13,
    bd_14,
    bd_15
} npu_bd_id;

///@brief npu sequence
///@note This is a class to store, create and process the npu sequence
class npu_sequence{
    public:
        npu_sequence() {
            this->npu_seq = nullptr;
            this->belonging_device = nullptr;
            this->belonging_kernel = nullptr;
            this->is_valid = false;
        }; // default constructor, not allowed
        npu_sequence(npu_device device_gen, xrt::device* device, xrt::kernel* kernel, std::string app_name); // for construct from xclbin and app_name
        void from_file(std::string filename, bool is_binary = true); // read from file
        void write_out_sequence(std::string filename); // write out the sequence
        void interpret(); // print the sequence
        void name_instr(std::string instr_name);
        void setup_device(npu_device device);
        void seq2cmds(); // parse the sequence
        void cmds2seq(); // parse the sequence
        void rtp_write(npu_tiles tile, uint32_t addr, uint32_t value);
        void npu_dma_memcpy_nd(
            int elem_size,
            int arg_idx,
            dma_direction channel_direction,
            npu_tiles tile,
            npu_bd_id bd_id,
            npu_it_channel it_channel,
            const std::vector<uint32_t>& _offset,
            const std::vector<uint32_t>& _size,
            const std::vector<uint32_t>& _stride,
            int packet_id = -1,
            int packet_type = 0,
            bool issue_token = false,
            cache_flag_t cache_flag = normal_cache,
            int shm_control_packed_id=15,
            bool submit_to_queue=true
        );
        void npu_dma_wait(npu_tiles tile, dma_direction channel_direction, npu_it_channel it_channel);
        void npu_maskwrite(npu_tiles tile, uint32_t addr, uint32_t value, uint32_t mask);
        xrt::bo& bo();
        void clear_cmds();
        void sync() {this->npu_seq->sync_to_device();}
        size_t size() {return this->npu_seq->size() * sizeof(uint32_t);}
        buffer<uint32_t> dump();
        std::string name() {return this->instr_name;}
        npu_device device_gen;
    private:
        constexpr static uint32_t dev_n_row_shift = 24;
        constexpr static uint32_t dev_gen_shift = 16;
        constexpr static uint32_t dev_minor_shift = 8;
        constexpr static uint32_t dev_major_shift = 0;
        constexpr static uint32_t dev_mem_tile_rows_shift = 8;
        constexpr static uint32_t dev_num_cols_shift = 0;
        constexpr static uint32_t dev_n_row_mask = 0xFF;
        constexpr static uint32_t dev_gen_mask = 0xFF;
        constexpr static uint32_t dev_minor_mask = 0xFF;
        constexpr static uint32_t dev_major_mask = 0xFF;
        constexpr static uint32_t dev_mem_tile_rows_mask = 0xFF;
        constexpr static uint32_t dev_num_cols_mask = 0xFF;  
        uint32_t npu_rows;
        uint32_t npu_cols;
        uint32_t npu_dev_gen;
        uint32_t npu_mem_tile_rows;
        uint32_t npu_minor;
        uint32_t npu_major;
        uint32_t instruction_counts;
        uint32_t instruction_lines;

        std::string instr_name;
        std::vector<std::unique_ptr<npu_cmd>> cmds;
        xrt::device* belonging_device;
        xrt::kernel* belonging_kernel;
        std::unique_ptr<buffer<uint32_t>> npu_seq;
        bool is_valid;

        bool _check_bo_valid(){

            if (this->belonging_device == nullptr){
                return false;
            }
            if (this->belonging_kernel == nullptr){
                return false;
            }
            return true;
        }
};

#endif
