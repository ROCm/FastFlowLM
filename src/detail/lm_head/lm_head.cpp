#include "lm_head_detail.hpp"
#include "utils/utils.hpp"

LMHead::Impl::Impl(LM_Config config, npu_xclbin_manager *npu) : config(config), npu(npu){
    vocab_size = config.get("vocab_size");
    hidden_size = config.get("hidden_size");
    vocab_size_padded = (vocab_size + padding_size - 1) / padding_size * padding_size;
    chunk_size = vocab_size_padded / split_apps;
    if (hidden_size % 256 != 0){
        hidden_size = (hidden_size / 256 + 1) * 256;
    }
    blocks_per_row = hidden_size / 256;

    lm_head_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "lm_head.xclbin"));
    assert(blocks_per_row <= 64);
    // assert(a_block_size * blocks_per_row <= 1024);
    apps.resize(split_apps);
    for (int i = 0; i < split_apps; i++){
        apps[i] = lm_head_app_manager->create_app();
    }
    if (!this->npu->is_preemption_enabled()){
        lm_head_run = lm_head_app_manager->create_runlist();
    }

    lm_head_x = apps[0].create_bo_buffer<bf16>(hidden_size);
    lm_head_y = apps[0].create_bo_buffer<bf16>(vocab_size_padded);
    lm_head_y.memset((bf16)0.0f);
    lm_head_y.sync_to_device();

    lm_head_weights.resize(split_apps);
    for (int i = 0; i < split_apps; i++){
        lm_head_weights[i] = apps[i].create_bo_buffer<u8>(chunk_size * hidden_size * 5 / 8);
    }

    _generate_seq();
    if (!this->npu->is_preemption_enabled()){
        for (int i = 0; i < split_apps; i++){
            lm_head_run.add(apps[i].create_run(lm_head_y, lm_head_weights[i], lm_head_x));
        }
    }

    logits = buffer<bf16>(lm_head_y.data(), vocab_size);
    x_exposed = buffer<bf16>(this->lm_head_x.data(), hidden_size);
}

void LMHead::Impl::_generate_seq(){
    for (int i = 0; i < split_apps; i++){
        npu_sequence& seq = *this->apps[i].seq();

        seq.npu_preemption(0);
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
                    {0, 0, 0, (round * cores + col * 4) * a_block_size * blocks_per_row},
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
        seq.cmds2seq();
        apps[i].update_ctrl_seq();
    }
}

void LMHead::Impl::load_weights(Q4NX& q4nx){
    buffer<u8> w_lm_head_w;
    q4nx.load_weights(w_lm_head_w, "lm_head.weight");
    size_t w_size = lm_head_weights[0].size();
    u8* ptr = w_lm_head_w.data();
    for (int i = 0; i < split_apps - 1; i++){
        memcpy(lm_head_weights[i].data(), ptr, w_size);
        ptr += w_size;
    }
    lm_head_weights[split_apps - 1].memset(0);
    size_t remaining_size = w_lm_head_w.size() - w_size * (split_apps - 1);
    memcpy(lm_head_weights[split_apps - 1].data(), ptr, remaining_size);

    for (int i = 0; i < split_apps; i++){
        lm_head_weights[i].sync_to_device();
    }
}

void LMHead::Impl::execute(){
    this->lm_head_x.sync_to_device();
    if (!this->npu->is_preemption_enabled()){
        this->lm_head_run.execute();
    }
    else{
        for (int i = 0; i < split_apps - 1; i++){
            apps[i](lm_head_y, lm_head_weights[i], lm_head_x);
        }
        this->final_run = apps[split_apps - 1].create_run(lm_head_y, lm_head_weights[split_apps - 1], lm_head_x);
        this->final_run.start();
    }
}

buffer<bf16> LMHead::Impl::wait(){
    if (!this->npu->is_preemption_enabled()){
        this->lm_head_run.wait();
    }
    else{
        this->final_run.wait();
    }
    this->lm_head_y.sync_from_device();
    return this->logits;
}

LMHead::Impl::~Impl() = default;

// ===============================================
// Externals for LMHead
// ===============================================
LMHead::LMHead(LM_Config config, npu_xclbin_manager *npu){
    this->_impl = new Impl(config, npu);
}

void LMHead::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void LMHead::execute(){
    this->_impl->execute();
}

buffer<bf16> LMHead::wait(){
    return this->_impl->wait();
}

buffer<bf16> LMHead::x_exposed(){
    return this->_impl->x_exposed;
}
LMHead::~LMHead(){
    delete this->_impl;
}
