// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "utils/utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "npu_utils/npu_utils.hpp"

struct WhisperLMHead {
public:
    WhisperLMHead(){}
    WhisperLMHead(Whisper_Config config, npu_xclbin_manager *npu);
    ~WhisperLMHead();
    void load_weights(Q4NX& q4nx);
    void sync_from_device();
    void execute();
    buffer<bf16> wait();

    static constexpr int lm_head_y_group_id = 3;
    static constexpr int lm_head_w_group_id = 4;
    static constexpr int lm_head_x_group_id = 5;

    static constexpr int lm_head_y_arg_idx = 0;
    static constexpr int lm_head_w_arg_idx = 1;
    static constexpr int lm_head_x_arg_idx = 2;
    static constexpr npu_tiles lm_head_tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    static constexpr uint32_t columns = 8;
    static constexpr uint32_t m = 32;
    static constexpr int split_apps = 2;
    static constexpr int padding_size = 4096;
    static constexpr uint32_t a_block_size = 32 * 128 * 5 / 8 / 4;
    static constexpr uint32_t cores = columns * 4;
    Whisper_Config config;
    npu_xclbin_manager *npu;
    npu_app_manager* lm_head_app_manager;
    bool enable_preemption;

    npu_app app;
    flm_rt::run final_run;
    flm_rt::runlist lm_head_run;
    buffer<u8> lm_head_weights;
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

WhisperLMHead::WhisperLMHead(Whisper_Config config, npu_xclbin_manager *npu) : config(config), npu(npu){
    vocab_size = config.get("vocab_size");
    hidden_size = config.get("hidden_size");
    vocab_size_padded = (vocab_size + padding_size - 1) / padding_size * padding_size;
    chunk_size = vocab_size_padded / split_apps;
    if (hidden_size % 128 != 0){
        hidden_size = (hidden_size / 128 + 1) * 128;
    }
    blocks_per_row = hidden_size / 128;
    assert(blocks_per_row <= 64);
    // assert(a_block_size * blocks_per_row <= 1024);
    lm_head_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "whisper_head.xclbin"));

    app = lm_head_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        lm_head_run = lm_head_app_manager->create_runlist();
    }

    lm_head_x = app.create_bo_buffer<bf16>(hidden_size);
    lm_head_y = app.create_bo_buffer<bf16>(vocab_size_padded);
    lm_head_y.memset((bf16)0.0f);
    lm_head_y.sync_to_device();

    lm_head_weights = app.create_bo_buffer<u8>(vocab_size_padded * hidden_size * 5 / 8);

    _generate_seq();

    logits = buffer<bf16>(vocab_size);
}

void WhisperLMHead::_generate_seq(){
    npu_sequence& seq = *(this->app.seq());
    seq.clear_cmds();
    seq.npu_preemption(0);
    for (int i = 0; i < split_apps; i++){
        assert(chunk_size / m / cores > 0);
        assert(chunk_size % (m * cores)  == 0);
        assert(chunk_size % (columns * 4 * m) == 0);
        // x_in
        seq.npu_dma_memcpy_nd(
            sizeof(bf16), lm_head_x_arg_idx,
            MM2S, lm_head_tiles[0],  bd_0, it_channel_0,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)hidden_size},
            {0, 0, 0, 1},
            -1, 0, false
        );

        // y_out
        for (uint32_t col = 0; col < columns; col++){
            uint32_t y_offset = 4 * m * col;
            seq.npu_dma_memcpy_nd(
                sizeof(bf16), lm_head_y_arg_idx,
                S2MM, lm_head_tiles[col],  bd_2, it_channel_0,
                {0, 0, 0, y_offset + i * chunk_size},
                {1, 1, chunk_size / columns / 4 / m, 4 * m},
                {0, 0, columns * 4 * m, 1},
                -1, 0, true
            );
        }

        for (uint32_t round = 0; round < chunk_size / m / cores; round++){
            uint32_t bd_offset = (round % 2) * 8;
            for (uint32_t col = 0; col < columns; col++){
                seq.npu_dma_memcpy_nd(
                    sizeof(uint32_t),
                    lm_head_w_arg_idx,
                    MM2S,
                    lm_head_tiles[col],
                    npu_bd_id(bd_1 + bd_offset),
                    it_channel_1,
                    {0, 0, 0, (round * cores + col * 4) * a_block_size * blocks_per_row + i * chunk_size * hidden_size * 5 / 8 / 4},
                    {blocks_per_row, 4, 8, a_block_size / 8},
                    {a_block_size, a_block_size * blocks_per_row, a_block_size / 8, 1},
                    -1, 0, true
                );
            }
            if (round > 0){
                for (uint32_t col = 0; col < columns; col++){
                    seq.npu_dma_wait(
                        lm_head_tiles[col],
                        MM2S,
                        it_channel_1
                    );
                }
            }
        }
        for (uint32_t col = 0; col < columns; col++){
            seq.npu_dma_wait(
                lm_head_tiles[col],
                MM2S,
                it_channel_1
            );
        }
        for (uint32_t col = 0; col < columns; col++){
            seq.npu_dma_wait(
                lm_head_tiles[col],
                S2MM,
                it_channel_0
            );
        }
    }
    seq.cmds2seq();
}

void WhisperLMHead::load_weights(Q4NX& q4nx){
    buffer<u8> w_lm_head_w;
    q4nx.load_weights(w_lm_head_w, "lm_head.weight");
    memcpy(lm_head_weights.data(), w_lm_head_w.data(), w_lm_head_w.size());
    memset(lm_head_weights.data() + w_lm_head_w.size(), 0, lm_head_weights.size() - w_lm_head_w.size());
    lm_head_weights.sync_to_device();
}

void WhisperLMHead::execute(){
    lm_head_weights.sync_to_device();
    app(lm_head_y, lm_head_weights, lm_head_x);
    this->lm_head_y.sync_from_device();
    memcpy(logits.data(), lm_head_y.data(), vocab_size * sizeof(bf16));
}

buffer<bf16> WhisperLMHead::wait(){

    return this->logits;
}

void WhisperLMHead::sync_from_device(){
}

WhisperLMHead::~WhisperLMHead() = default;
