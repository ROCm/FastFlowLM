// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "modules/mha.hpp"

struct MHA::Impl{
private:
    static inline constexpr uint32_t _get_head_dim(mha_type_t type) {
        switch(type) {
            case mha_d64_q4:   return 64;
            case mha_d128_q2:  return 128;
            case mha_d128_q3:  return 128;
            case mha_d128_q4:  return 128;
            case mha_d256_q2:  return 256;
            case mha_d256_q4:  return 256;
            case mha_d128_q4_1cu: return 128;
            default:           return 0;
        }
    }

    static inline constexpr uint32_t _get_l_begin_addr(mha_type_t type) {
        switch(type) {
            case mha_d64_q4:   return 59392;
            case mha_d128_q2:  return 15360;
            case mha_d128_q3:  return 15360;
            case mha_d128_q4:  return 15360;
            case mha_d256_q2:  return 61568;
            case mha_d256_q4:  return 61568;
            case mha_d128_q4_1cu: return 15360;
            default:           return 0;
        }
    }

    static inline constexpr uint32_t _get_l_end_addr(mha_type_t type) {
        switch(type) {
            case mha_d64_q4:   return 12416;
            case mha_d128_q2:  return 25088;
            case mha_d128_q3:  return 25088;
            case mha_d128_q4:  return 25088;
            case mha_d256_q2:  return 11904;
            case mha_d256_q4:  return 11904;
            case mha_d128_q4_1cu: return 25088;
            default:           return 0;
        }
    }

    static inline constexpr uint32_t _get_window_size_addr(mha_type_t type) {
        switch(type) {
            case mha_d64_q4:   return 45184;
            case mha_d128_q2:  return 36992;
            case mha_d128_q3:  return 36992;
            case mha_d128_q4:  return 36992;
            case mha_d256_q2:  return 61600;
            case mha_d256_q4:  return 61600;
            case mha_d128_q4_1cu: return 36992;
            default:           return 0;
        }
    }

    static inline constexpr int _get_ct_chunk_size(mha_type_t type) {
        switch(type) {
            case mha_d64_q4:   return 32;
            case mha_d128_q2:  return 16;
            case mha_d128_q3:  return 16;
            case mha_d128_q4:  return 16;
            case mha_d256_q2:  return 16;
            case mha_d256_q4:  return 16;
            case mha_d128_q4_1cu: return 16;
            default:           return 0;
        }
    }

    static constexpr npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};

    mha_type_t mha_ty;
    uint32_t l_begin_mha_address;
    uint32_t l_end_mha_address;
    uint32_t window_size_address;

    uint32_t head_dim;
    int num_kv_heads;
    int num_q_heads;
    // how many heads are continuous on memory
    // in this MHA implementation, KV are seperatred into 4 parts
    // K fisrt half
    // V first half
    // K second half
    // V second half
    int num_continuous_heads;

    int ct_chunk_size;
    int chunk_size;

    void _gen_mha_seq_d64_q4 (npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);
    void _gen_mha_seq_d128_q2(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);
    void _gen_mha_seq_d128_q3(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);
    void _gen_mha_seq_d128_q4(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);
    void _gen_mha_seq_d256_q2(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);
    void _gen_mha_seq_d256_q4(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);
    void _gen_mha_seq_d128_q4_1cu(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);

    int _get_num_q_heads(mha_type_t type) {
        switch(type) {
            case mha_d64_q4:   return this->num_kv_heads * 4;
            case mha_d128_q2:  return this->num_kv_heads * 2;
            case mha_d128_q3:  return this->num_kv_heads * 3;
            case mha_d128_q4:  return this->num_kv_heads * 4;
            case mha_d256_q2:  return this->num_kv_heads * 2;
            case mha_d256_q4:  return this->num_kv_heads * 4;
            case mha_d128_q4_1cu: return this->num_kv_heads * 4;
            default:           return 0;
        }
    }

    bool _parameter_check(const uint32_t L_begin, const uint32_t L_end, const bool is_sliding_window, const int WINDOW_SIZE);
public:
    Impl(){}
    Impl(mha_type_t& mha_ty, int num_kv_heads);
    ~Impl();

    void generate_mha_sequence(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_end, const uint32_t MAX_L, const bool is_sliding_window, const int WINDOW_SIZE);

    int get_chunk_size(){ return chunk_size; }
};
