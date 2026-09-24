#pragma once
#include "modules/lm_head.hpp"

struct LMHead::Impl{
public:
    // Impl(){}
    Impl(LM_Config config, npu_xclbin_manager *npu);
    ~Impl();
    void load_weights(Q4NX& q4nx);
    void execute();
    buffer<bf16> wait();
    buffer<bf16> x_exposed;

    static constexpr npu_tiles lm_head_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr int lm_head_y_arg_idx = 0;
    static constexpr int lm_head_w_arg_idx = 1;
    static constexpr int lm_head_x_arg_idx = 2;

    static constexpr uint32_t columns = 8;
    static constexpr uint32_t m = 32;
    static constexpr int split_apps = 4;
    static constexpr int padding_size = 4096;
    static constexpr uint32_t a_block_size = 32 * 256 * 5 / 8 / 4;
    static constexpr uint32_t cores = columns * 4;
    LM_Config config;
    npu_xclbin_manager *npu;
    npu_app_manager* lm_head_app_manager;

    std::vector<npu_app> apps;
    flm_rt::run final_run;
    flm_rt::runlist lm_head_run;
    std::vector<buffer<u8>> lm_head_weights;
    buffer<bf16> lm_head_y;
    buffer<bf16> lm_head_x;
    uint32_t blocks_per_row;

    uint32_t vocab_size;
    uint32_t vocab_size_padded;
    uint32_t hidden_size;
    uint32_t chunk_size;
    buffer<bf16> logits;

    void _generate_seq();
};
