#ifndef __gemm_detail__
#define __gemm_detail__
#include "modules/gemm.hpp"

struct Gemm::Impl
{
private:
    static constexpr int Arg_C = 0;
    static constexpr int Arg_A = 1;
    static constexpr int Arg_B = 2;
    static constexpr int Arg_Bias = 3;
    static constexpr int CT_lock_address_base = 0x000001F000;

    static constexpr int mm_y_group_id = 3;
    static constexpr int mm_x_group_id = 4;
    static constexpr int mm_w_group_id = 5;
    static constexpr npu_tiles shim_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr uint32_t total_cols = 8;
    static constexpr uint32_t total_rows = 4;

    static constexpr uint32_t shimtile_size = total_cols > total_rows ? total_cols : total_rows; // max of the two

    static constexpr uint32_t CT_rtp_address = 4096; // for 128

    static constexpr int CT_rtp_sync_lock_id = 10; // for now hard coded

    std::map<int, int> valid_A_MT_shimtile_index;

public:
    static constexpr uint32_t m = 64;
    static constexpr uint32_t k = 512;
    static constexpr uint32_t n = 128; // for now
    static constexpr uint32_t r = 8;
    static constexpr uint32_t s = 8;
    static constexpr uint32_t t = 8;
    bool IS_B_ROW_MAJOR = false; // for language model, B is always in col-major order
    bool B_in_K_N_block_col_major_order = true; // for language model, B is default in kxn block col-major order
    bool ENABLE_AXI4 = true; // for language model, B is always in kxn block col-major order

    Impl();
    ~Impl();

    /// \brief Generate the sequence
    /// \param seq the npu sequence
    /// \param M the M dimension
    /// \param K the K dimension
    /// \param N the N dimension
    /// \param weight_offset the weight offset
    /// \param ADD_BIAS whether to add bias
    /// \param OUTPUT_MODE the output activation mode
    /// \param bias_offset the bias offset
    void generate_seq(
        npu_sequence *seq,
        uint32_t M,
        uint32_t K,
        uint32_t N,
        const uint32_t weight_offset,
        bool ADD_BIAS,
        Activation_Type_t OUTPUT_MODE,
        const uint32_t bias_offset,
        uint32_t C_const_offset
    );

    template <typename T_in, typename T_out>
    void generate_runtime_sequence(
        npu_sequence* seq,
        uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset, uint32_t Bias_const_offset,
        uint32_t M_size, uint32_t N_size, uint32_t K_size,
        std::vector<int> &list_A_shim_queue,
        std::vector<int> &list_B_shim_queue,
        std::vector<int> &list_C_shim_queue,
        bool IS_B_ROW_MAJOR,
        bool ENABLE_AXI4,
        bool B_in_K_N_block_col_major_order,
        bool ADD_BIAS
    );

    template <typename T_in, typename T_out>
    void generate_shimtile_sequence_per_k_block(
        npu_sequence*seq,
        uint32_t shim_index,
        uint32_t mega_block_row_idx, uint32_t mega_block_col_idx,
        uint32_t M_size, uint32_t K_size, uint32_t N_size,
        uint32_t A_const_offset, uint32_t B_const_offset,  uint32_t C_const_offset, uint32_t Bias_const_offset,
        std::vector<int> &list_A_shim_queue, std::vector<int> &list_B_shim_queue, std::vector<int> &list_C_shim_queue,
        std::vector<int> &list_A_bd_pingpong_flag, std::vector<int> &list_B_bd_pingpong_flag, std::vector<int> &list_C_bd_pingpong_flag,
        bool IS_B_ROW_MAJOR, bool ENABLE_AXI4,
        bool B_in_K_N_block_col_major_order,
        bool VALID_COLUMN,
        bool ADD_BIAS, bool SEND_BIAS
    );
};
#endif
