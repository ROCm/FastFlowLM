// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gemma_embedding_detail.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include "metrices.hpp"

// A dispatch that did not complete leaves its output BO holding stale data;
// pooling that would return a plausible-looking but wrong vector with no error.
static void require_completed(ert_cmd_state st, const char* what, int layer_idx) {
    if (st == ERT_CMD_STATE_COMPLETED) return;
    throw std::runtime_error(
        std::string("gemma_embedding::embed: ") + what + " dispatch state " +
        std::to_string((int)st) + " at layer " + std::to_string(layer_idx));
}

gemma_embedding::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance)
    : config(config), npu(npu_instance){

    this->model_path = config.model_path;
    this->embedding = std::make_unique<Embedding>(config.get("vocab_size"), config.get("hidden_size"));
    gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
    full_attn_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn_full_mask.xclbin"));
    window_attn_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "sliding_attn.xclbin"));
    mv_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mv.xclbin"));

    // create all apps

    this->q_proj = gemm_app_manager->create_app();
    this->k_proj = gemm_app_manager->create_app();
    this->v_proj = gemm_app_manager->create_app();
    this->o_proj = gemm_app_manager->create_app();
    this->up_gate_proj = gemm_app_manager->create_app();
    this->down_proj = gemm_app_manager->create_app();

    this->full_attn = full_attn_app_manager->create_app();
    this->window_attn = window_attn_app_manager->create_app();

    this->dense_1 = mv_app_manager->create_app();
    this->dense_2 = mv_app_manager->create_app();

    // create all buffers
    this->input_layernorm_weights.resize(num_hidden_layers);
    this->post_attention_layernorm_weights.resize(num_hidden_layers);
    this->pre_feedforward_layernorm_weights.resize(num_hidden_layers);
    this->post_feedforward_layernorm_weights.resize(num_hidden_layers);
    this->q_norm_weights.resize(num_hidden_layers);
    this->k_norm_weights.resize(num_hidden_layers);
    this->proj_weights.resize(num_hidden_layers);
    this->kv_caches.resize(num_hidden_layers);
    this->norm_weights = buffer<bf16>(D);

    for (int i = 0; i < num_hidden_layers; i++){
        this->input_layernorm_weights[i] = buffer<bf16>(D);
        this->post_attention_layernorm_weights[i] = buffer<bf16>(D);
        this->pre_feedforward_layernorm_weights[i] = buffer<bf16>(D);
        this->post_feedforward_layernorm_weights[i] = buffer<bf16>(D);
        this->q_norm_weights[i] = buffer<bf16>(DH);
        this->k_norm_weights[i] = buffer<bf16>(DH);
    }
    this->final_tokens_buffer = buffer<bf16>(D);
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_caches[i] = window_attn_app_manager->create_bo_buffer<bf16>(kv_cache_size);
        } else {
            this->kv_caches[i] = full_attn_app_manager->create_bo_buffer<bf16>(kv_cache_size);
        }

        this->proj_weights[i] = gemm_app_manager->create_bo_buffer<bf16>(total_proj_weights);
    }

    // computing buffers
    this->buf_q = gemm_app_manager->create_bo_buffer<bf16>(MAX_L * DQ);
    this->buf_k = gemm_app_manager->create_bo_buffer<bf16>(MAX_L * DK);
    this->buf_v = gemm_app_manager->create_bo_buffer<bf16>(MAX_L * DV);
    this->buf_up_gate = gemm_app_manager->create_bo_buffer<bf16>(2 *MAX_L * intermediate_size);
    this->buf_down = gemm_app_manager->create_bo_buffer<bf16>(MAX_L * intermediate_size);
    this->buf_x = gemm_app_manager->create_bo_buffer<bf16>(MAX_L * D);
    this->buf_attn_out = gemm_app_manager->create_bo_buffer<bf16>(MAX_L * DQ);
    this->buf_pooling_out = dense_1.create_bo_buffer<bf16>(D);
    this->dense_1_weights = dense_1.create_bo_buffer<bf16>(dense_intermediate_size * D);
    this->dense_2_weights = dense_2.create_bo_buffer<bf16>(dense_intermediate_size * D);
    this->buf_dense_1_out = dense_1.create_bo_buffer<bf16>(dense_intermediate_size);

    this->residual_buffer = buffer<bf16>(MAX_L * D);
    this->hidden_states_buffer = buffer<bf16>(MAX_L * D);

    this->tensor_residual.assign(residual_buffer, D, 0);
    this->tensor_hidden_states.assign(hidden_states_buffer, D, 0);
    this->tensor_q.assign(buf_q, DQ, 0);
    this->tensor_k.assign(buf_k, DK, 0);
    this->tensor_v.assign(buf_v, DV, 0);
    this->tensor_up_gate.assign(buf_up_gate, intermediate_size, 0);
    this->tensor_down.assign(buf_down, intermediate_size, 0);
    this->tensor_x.assign(buf_x, D, 0);
    this->tensor_attn_out.assign(buf_attn_out, DQ, 0);
}

void gemma_embedding::Impl::load_weights(Q4NX& q4nx){
    // read qkv weights
    LOG_VERBOSE(1, "Loading embedding weights");
    this->embedding->init_weights(&q4nx, "embed_tokens");

    q4nx.load_weights(this->norm_weights, "norm.weight");

    q4nx.load_weights(this->dense_1_weights, "dense_2.weight");
    q4nx.load_weights(this->dense_2_weights, "dense_3.weight");

    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        load_all_proj_weights(this->proj_weights[layer_idx], q4nx, layer_idx);
        q4nx.load_weights(this->input_layernorm_weights[layer_idx], "layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
        q4nx.load_weights(this->post_attention_layernorm_weights[layer_idx], "layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
        q4nx.load_weights(this->pre_feedforward_layernorm_weights[layer_idx], "layers." + std::to_string(layer_idx) + ".pre_feedforward_layernorm.weight");
        q4nx.load_weights(this->post_feedforward_layernorm_weights[layer_idx], "layers." + std::to_string(layer_idx) + ".post_feedforward_layernorm.weight");
        q4nx.load_weights(this->q_norm_weights[layer_idx], "layers." + std::to_string(layer_idx) + ".self_attn.q_norm.weight");
        q4nx.load_weights(this->k_norm_weights[layer_idx], "layers." + std::to_string(layer_idx) + ".self_attn.k_norm.weight");
        this->proj_weights[layer_idx].sync_to_device();
    }
}

buffer<bf16> gemma_embedding::Impl::embed(std::vector<int>& tokens){
    int L = tokens.size();

    if (L > MAX_L) {
        header_print("Error", "Sequence length is too long!");
    }

    int LQ_padded = (L + 128 - 1) / 128 * 128;
    int LK_padded = (L + 16 - 1) / 16 * 16;
    // generate sequence
    gen_mm_seq(q_proj.seq(), LQ_padded, D, DQ, 0);
    gen_mm_seq(k_proj.seq(), LQ_padded, D, DK, DQ * D);
    gen_mm_seq(v_proj.seq(), LQ_padded, D, DV, (DQ + DK) * D);
    gen_mm_seq(o_proj.seq(), LQ_padded, D, D, (DQ + DK + DV) * D);
    gen_mm_seq(up_gate_proj.seq(), LQ_padded, D, 2 * intermediate_size, (DQ + DK + DV + D) * D);
    gen_mm_seq(down_proj.seq(), LQ_padded, intermediate_size, D, D * (DQ + DK + DV + D + intermediate_size * 2));

    gen_sliding_mha_engine_seq(window_attn.seq(), L);
    gen_mha_engine_seq(full_attn.seq(), LK_padded, LQ_padded, L);

    gen_mvm_batch_engine_seq(dense_1.seq(), dense_intermediate_size, D, 32, 256, 1, 0);
    gen_mvm_batch_engine_seq(dense_2.seq(), D, dense_intermediate_size, 32, 256, 1, 0);
    // initial embedding
    for (int l = 0; l < L; l++){
        this->embedding->forward(tokens[l], this->tensor_residual[l]);
    }

    // run sequence
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        // input layer norm
        this->buf_x.sync_to_device();
        for (int l = 0; l < L; l++){
            apply_rms_norm(this->tensor_x[l], this->tensor_residual[l], this->input_layernorm_weights[layer_idx]);
        }

        // qkv projection
        this->buf_x.sync_to_device();
        this->buf_q.sync_to_device();
        // this->proj_weights[layer_idx].sync_to_device();
        require_completed(this->q_proj(this->buf_q, this->buf_x, this->proj_weights[layer_idx]), "q_proj", layer_idx);
        this->buf_q.sync_from_device();
        auto k_run = k_proj.create_run(this->buf_k, this->buf_x, this->proj_weights[layer_idx]);
        auto v_run = v_proj.create_run(this->buf_v, this->buf_x, this->proj_weights[layer_idx]);
        this->buf_x.sync_to_device();
        this->buf_k.sync_to_device();
        k_run.start();
        for (int i = 0; i < L; i++){
            rope_rms(this->tensor_q[i], i, this->q_norm_weights[layer_idx], layer_idx);
        }
        this->buf_q.sync_to_device();
        require_completed(k_run.wait(), "k_proj", layer_idx);
        this->buf_k.sync_from_device();

        // this->proj_weights[layer_idx].sync_to_device();
        this->buf_x.sync_to_device();
        this->buf_v.sync_to_device();
        v_run.start();
        for (int i = 0; i < L; i++){
            rope_rms(this->tensor_k[i], i, this->k_norm_weights[layer_idx], layer_idx);
        }
        this->buf_k.sync_to_device();
        require_completed(v_run.wait(), "v_proj", layer_idx);
        this->buf_v.sync_from_device();

        this->buf_attn_out.sync_from_device();
        migrate_kv_cache(layer_idx, L);
        this->buf_q.sync_to_device();
        if (is_sliding_window(layer_idx)){
            require_completed(window_attn(this->buf_attn_out, this->buf_q, this->kv_caches[layer_idx]), "window_attn", layer_idx);
        }
        else{
            require_completed(full_attn(this->buf_attn_out, this->buf_q, this->kv_caches[layer_idx]), "full_attn", layer_idx);
        }
        this->buf_attn_out.sync_from_device();
        memcpy(this->buf_x.data(), this->buf_attn_out.data(), L * D * sizeof(bf16));
        this->buf_x.sync_to_device();
        // this->proj_weights[layer_idx].sync_to_device();
        this->buf_q.sync_to_device();
        require_completed(o_proj(this->buf_q, this->buf_x, this->proj_weights[layer_idx]), "o_proj", layer_idx);

        this->buf_q.sync_from_device();

        for (int i = 0; i < L; i++){
            apply_rms_norm(this->tensor_hidden_states[i], this->tensor_q[i], this->post_attention_layernorm_weights[layer_idx]);
            residual_add(this->tensor_residual[i], this->tensor_hidden_states[i], this->tensor_residual[i]);
            apply_rms_norm(this->tensor_x[i], this->tensor_residual[i], this->pre_feedforward_layernorm_weights[layer_idx]);
        }

        this->buf_x.sync_to_device();
        this->buf_up_gate.sync_to_device();

        require_completed(this->up_gate_proj(this->buf_up_gate, this->buf_x, this->proj_weights[layer_idx]), "up_gate_proj", layer_idx);
        this->buf_up_gate.sync_from_device();
        this->buf_down.sync_from_device();
        element_wise(this->buf_down, this->buf_up_gate, L);
        this->buf_down.sync_to_device();
        this->buf_q.sync_from_device();

        require_completed(this->down_proj(this->buf_q, this->buf_down, this->proj_weights[layer_idx]), "down_proj", layer_idx);
        this->buf_q.sync_from_device();
        this->proj_weights[layer_idx].sync_from_device();
        for (int i = 0; i < L; i++){
            apply_rms_norm(this->tensor_hidden_states[i], this->tensor_q[i], this->post_feedforward_layernorm_weights[layer_idx]);
            residual_add(this->tensor_residual[i], this->tensor_hidden_states[i], this->tensor_residual[i]);
        }
    }
    for (int i = 0; i < L; i++){
        apply_rms_norm(this->tensor_residual[i], this->tensor_residual[i], this->norm_weights);
    }
    pooling_mean(buf_pooling_out, residual_buffer, L, D);
    buf_pooling_out.sync_to_device();
    this->dense_1_weights.sync_to_device();
    require_completed(dense_1(this->buf_dense_1_out, this->dense_1_weights, this->buf_pooling_out), "dense_1", num_hidden_layers);
    this->buf_dense_1_out.sync_from_device();
    this->dense_1_weights.sync_from_device();
    this->dense_2_weights.sync_to_device();
    require_completed(dense_2(this->buf_pooling_out, this->dense_2_weights, this->buf_dense_1_out), "dense_2", num_hidden_layers);
    this->buf_pooling_out.sync_from_device();
    this->dense_2_weights.sync_from_device();
    l2_norm(this->final_tokens_buffer, this->buf_pooling_out);

    return final_tokens_buffer;
}

///@brief destructor of gemma_embedding
gemma_embedding::Impl::~Impl() = default;

// ===============================================
// Externals for gemma_embedding
// ===============================================
gemma_embedding::gemma_embedding(LM_Config config, npu_xclbin_manager *npu_instance){
    this->_impl = new Impl(config, npu_instance);
}

void gemma_embedding::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

buffer<bf16> gemma_embedding::embed(std::vector<int>& tokens){
    return this->_impl->embed(tokens);
}

gemma_embedding::~gemma_embedding() = default;
