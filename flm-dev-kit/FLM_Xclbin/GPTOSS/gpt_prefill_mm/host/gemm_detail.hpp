// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// #ifndef __gemm_detail__
// #define __gemm_detail__
// #include "npu_utils.hpp"
// #include "npu_instr_utils.hpp"
// /// \brief gemm class
// /// \note This is a class for the gemm layer
// class Gemm
// {
// public:
//     Gemm() {}

//     /// \brief Constructor
//     /// \param config the configuration
//     /// \param xclbin_name the xclbin name
//     /// \param npu the npu manager
//     ~Gemm();

//     /// \brief Generate the sequence
//     void generate_seq(npu_sequence *seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset);

//     /// \brief get y_group_id
//     /// \return the y group id
//     int y_id();
//     /// \brief get x_group_id
//     /// \return the x group id
//     int x_id();
//     /// \brief get w_group_id
//     /// \return the w group id
//     int w_id();

// private:
//     struct Impl;
//     Impl *_impl;
// };

// struct Gemm::Impl
// {
// private:
//     static constexpr int mm_y_group_id = 3;
//     static constexpr int mm_x_group_id = 4;
//     static constexpr int mm_w_group_id = 5;

//     static constexpr npu_tiles IT[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

//     static constexpr uint32_t total_cols = 8;
//     static constexpr uint32_t total_rows = 4;
//     static constexpr uint32_t m = 128; //TODO:
//     static constexpr uint32_t k = 512;
//     static constexpr uint32_t n = 64;

    
//     static constexpr uint32_t kk_address =  4096; // for 128

// public:
//     // Impl();

//     ~Impl();

//     inline int y_id() { return mm_y_group_id; };
//     inline int x_id() { return mm_x_group_id; };
//     inline int w_id() { return mm_w_group_id; };
//     void generate_seq(npu_sequence *seq, const uint32_t M, const uint32_t K, const uint32_t N, const uint32_t weight_offset);
//     void generate_runtime_sequence(
//         uint32_t Arg_A, uint32_t Arg_B, uint32_t Arg_C,
//         uint32_t M_size, uint32_t N_size, uint32_t K_size,
//         std::vector<int> &list_shim_queue,
//         uint32_t weight_offset,
//         std::map<int, int>& valid_A_MT_shimtile_index,
//         npu_sequence* seq,
//         std::vector<npu_tiles> &shim_tiles);

//     void generate_one_mega_block_sequence(
//         uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
//         uint32_t M_size, uint32_t N_size, uint32_t K_size,
//         uint32_t Arg_A, uint32_t  Arg_B, uint32_t Arg_C,
//         std::vector<int> &list_shim_queue,
//         uint32_t weight_offset,
//         std::map<int, int>& valid_A_MT_shimtile_index,
//         npu_sequence* seq,
//         std::vector<npu_tiles> &shim_tiles);

//     void generate_shimtile_sequence_per_mega_block(
//         uint32_t shim_index,
//         uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
//         uint32_t M_size, uint32_t K_size, uint32_t N_size,
//         uint32_t Arg_A, uint32_t  Arg_B, uint32_t Arg_C,
//         int &shim_wait_queue,
//         uint32_t weight_offset, 
//         std::map<int, int>& valid_A_MT_shimtile_index,
//         npu_sequence* seq,
//         npu_tiles& cur_shimtile );
// };
// #endif