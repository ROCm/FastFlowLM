/// \file lm_head.hpp
/// \brief lm_head class
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.29
/// \note This is a header file for the prefill stage mha
#pragma once
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "npu_utils/npu_utils.hpp"

typedef enum : uint32_t {
    mha_d64_q4, // dh = 64, 4 q head to 1 kv head
    mha_d128_q2, // dh = 128, 2 q head to 1 kv head
    mha_d128_q3, // dh = 128, 3 q head to 1 kv head
    mha_d128_q4, // dh = 128, 4 q head to 1 kv head
    mha_d256_q2, // dh = 256, 2 q head to 1 kv head
    mha_d256_q4,  // dh = 256, 4 q head to 1 kv head
    /// dh = 128, 4 q head to 1 kv head, but only one 2-column compute unit, on
    /// physical columns 6-7.  This is the attention half of a fused prefill
    /// overlay whose other six columns are a dequant+mm array; it trades 4x the
    /// head rounds for never having to swap the array between the two.
    mha_d128_q4_1cu,
    mha_not_supported
} mha_type_t;


/// \brief MHA class
/// \note This is a class for the mha layer
class MHA{
public:
    MHA(){}

    /// \brief Constructor
    /// \param mha_ty the type of mha
    /// \param num_kv_heads the number of key-value heads
    MHA(mha_type_t& mha_ty, int num_kv_heads);
    ~MHA();

    void generate_mha_sequence(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE = -1);
    
    int get_chunk_size();

private:
    struct Impl;
    Impl* _impl;

};
