// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "decoding_layer.hpp"
#include <fstream>

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int layer_idx, int context_length)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), MAX_L(context_length){
    header_print("info", "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();
    this->lm_head = this->layer_app_mgr->create_app();
    this->current_context_length = 0;
    num_hidden_layers = lm_config.num_hidden_layers;
    this->layer_idx = layer_idx;
    this->num_kv_per_round = lm_config.num_key_value_heads;
    this->num_kv_shared_layers = lm_config.num_kv_shared_layers;
    if (lm_config.hidden_size == 1536) {
        full_attn_period = 5;
    }
    else if (lm_config.hidden_size == 2560) {
        full_attn_period = 6;
    }
    else {
        throw std::runtime_error("Unsupported model config: hidden size " + std::to_string(lm_config.hidden_size) + " is not supported.");
     }
    this->is_swa_layer = is_sliding_window_layer(layer_idx);
    this->shall_skip_kv = skip_kv(layer_idx);
    header_print("info", "This layer is " + std::string(is_swa_layer? "a sliding window attention layer." : "a full attention layer."));
    header_print("info", "This layer will " + std::string(shall_skip_kv? "skip kv cache." : "not skip kv cache."));
    this->kv_caches_required = this->num_hidden_layers - this->num_kv_shared_layers;
    this->final_kv_cache_layer = last_kv_cache_layer();

    
    DH = is_swa_layer ? lm_config.head_dim : lm_config.global_head_dim;
    DQ = DH * lm_config.num_attention_heads;
    DK = DH * lm_config.num_key_value_heads;
    DV = DH * lm_config.num_key_value_heads;

    hidden_size_per_layer_input = lm_config.hidden_size_per_layer_input;

    num_kv_heads = lm_config.num_key_value_heads;
    intermediate_size = lm_config.intermediate_size;

    use_double_wide_mlp = lm_config._json_config["use_double_wide_mlp"].get<bool>();

    header_print("info", "Double wide mlp is " + std::string(use_double_wide_mlp? "enabled." : "disabled."));

    if (shall_skip_kv && use_double_wide_mlp) {
        intermediate_size *= 2;
    }
    
    D = lm_config.hidden_size;

    kv_cache_size = (size_t)(is_swa_layer ? SWA_WINDOW : MAX_L) * (DK + DV);
    w_qkv_offset = 0;
    w_o_offset = w_qkv_offset + (shall_skip_kv? DQ: (DQ + DK + DV)) * D * 5 / 8 / sizeof(bf16);
    w_up_offset = w_o_offset + D * DQ * 5 / 8 / sizeof(bf16);
    w_down_offset = w_up_offset + (2 * intermediate_size) * D * 5 / 8 / sizeof(bf16);
    w_per_layer_input_proj_offset = w_down_offset + intermediate_size * D * 5 / 8 / sizeof(bf16);
    w_inp_gate_offset = w_per_layer_input_proj_offset + (size_t)hidden_size_per_layer_input * D;
    w_inp_up_offset = w_inp_gate_offset + (size_t)hidden_size_per_layer_input * D;

    if (!s_token_embed) {
        s_token_embed = std::make_shared<embedding_module>((size_t)lm_config.vocab_size, D);
        s_pli_embed = std::make_shared<embedding_module>((size_t)lm_config.vocab_size, lm_config.num_hidden_layers * hidden_size_per_layer_input);
    }
    this->token_embed = s_token_embed;
    this->pli_embed = s_pli_embed;


    header_print("info", "Decoding layer config: ");
    header_print("info", "  D : " << D);
    header_print("info", "  DQ: " << DQ);
    header_print("info", "  DK: " << DK);
    header_print("info", "  DV: " << DV);
    header_print("info", "  Intermediate Size: " << intermediate_size);
    header_print("info", "  Num shared kv layers: " << num_kv_shared_layers);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    header_print("info", "  Num KV Per Round: " << this->num_kv_per_round);
    header_print("info", "  Final KV Cache Layer: " << this->final_kv_cache_layer);
    header_print("info", "  Hidden Size Per Layer Input: " << hidden_size_per_layer_input);
    header_print("info", "  Has QK Norm: " << (has_qk_norm ? "Yes" : "No"));
    header_print("info", "  Skip KV    : " << (shall_skip_kv? "Yes" : "No"));
    header_print("info", "  Is SWA Layer: " << (is_swa_layer? "Yes" : "No"));
    header_print("info", "  QKV offset: " << w_qkv_offset);
    header_print("info", "  O offset: " << w_o_offset);
    header_print("info", "  Up offset: " << w_up_offset);
    header_print("info", "  Down offset: " << w_down_offset);
    header_print("info", "  Per layer input proj offset: " << w_per_layer_input_proj_offset);
    header_print("info", "  Input gate offset: " << w_inp_gate_offset);
    header_print("info", "  Input up offset: " << w_inp_up_offset);
    if (this->shall_skip_kv) {
        tb_size = 0;
        header_print("info", "TB_SIZE is set to " + std::to_string(tb_size) + " since this layer will skip kv cache.");
    }
    else {
        tb_size = 0;
        header_print("info", "TB_SIZE is set to " + std::to_string(tb_size) + " since this layer will not skip kv cache.");
    }

    if (D == 1536) { // E2B
        l_qk_address = 57344;
        l_kv_address = 14976;
        swa_l_qk_address = 9216;
        swa_l_kv_address = 40960;
        proj_swa_address = 33280;
        proj_skip_address = 49664;
        rms_swa_address = 52224;
        rms_skip_address = 25600;
        rope_skip_kv_address = 33792;
        swa_rope_skip_kv_address = 33280;
        glu_skip_address = 34816;
    }
    else if (D == 2560) { // E4B
        l_qk_address = 57344;
        l_kv_address = 14976;
        swa_l_qk_address = 53248;
        swa_l_kv_address = 57664;
        proj_swa_address = 33280;
        proj_skip_address = 49664;
        rms_swa_address = 55328;
        rms_skip_address = 55392;
        rope_skip_kv_address = 34816;
        swa_rope_skip_kv_address = 33792;
        glu_skip_address = 30720;
    }
    else {
        throw std::runtime_error("Unsupported model config with D = " + std::to_string(D));
    }


    _load_rtp_file();

    this->rope_scaling = model_list::get_rope_scaling(this->model_type);

    size_t total_proj_weights = 
          (shall_skip_kv? DQ: (DQ + DK + DV)) * D
        + DQ * D
        + 3 * D * intermediate_size;
    size_t per_layer_input_weights = (size_t)hidden_size_per_layer_input * D * 3; // bf16
    size_t total_proj_weights_byte_size = total_proj_weights * 5 / 8 + per_layer_input_weights * 2;
    size_t kv_cache_size = this->kv_cache_size;

    LOG_VERBOSE(1, "Proj buffer size in bytes: " + std::to_string(total_proj_weights_byte_size));

    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(3 * D + tb_size);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(total_proj_weights_byte_size);
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(4 * D);
    if (this->has_qk_norm) {
        // rope: DH, q norm: DH, k norm: DH, per_layer_embedding: hidden_size_per_layer_input, per_layer_embedding norm: hidden_size_per_layer_input, D last norm
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(DH + DH + DH + hidden_size_per_layer_input * 2 + D + 32);
    }
    else {
        assert(0 && "QK norm is required for this model, but the code does not support non-qk-norm path anymore");
    }

    rope_rms_buffer.sync_to_device();
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();

}

void decoding_layer::_load_rtp_file(){
    if (rtp_file.empty()) return;
    std::ifstream f(rtp_file);
    if (!f) { header_print("warn", "rtp file not readable, using compiled-in RTP addresses: " + rtp_file); return; }
    std::string name; uint32_t addr; int n = 0;
    while (f >> name >> addr) {
        auto is = [&](const char* p){ return name.rfind(p, 0) == 0; };
        if (name == "RTP_L_attn_qk_core_2_2") l_qk_address = addr;
        else if (name == "RTP_L_attn_kv_core_3_2") l_kv_address = addr;
        else if (name == "RTP_L_swa_attn_qk_core_4_2") swa_l_qk_address = addr;
        else if (name == "RTP_L_swa_attn_kv_core_5_2") swa_l_kv_address = addr;
        else if (name == "RTP_RMS_IS_SWA_BUFFER") rms_swa_address = addr;
        else if (name == "RTP_RMS_SKIP_BUFFER") rms_skip_address = addr;
        else if (name == "RTP_RoPE_SKIP_KV_4_3") rope_skip_kv_address = addr;
        else if (name == "RTP_SWA_RoPE_SKIP_KV_5_3") swa_rope_skip_kv_address = addr;
        else if (name == "RTP_GLU_IS_SKIP_3_3") glu_skip_address = addr;
        else if (is("RTP_PROJ_IS_SWA_BUFFER_")) proj_swa_address = addr;
        else if (is("RTP_PROJ_SKIP_KV_BUFFER_")) proj_skip_address = addr;
        else continue;
        n++;
    }
    header_print("info", "RTP addresses read from " + rtp_file + " (" + std::to_string(n) + " entries)");
}

buffer<bf16> decoding_layer::kv_row(int plane, int pos){
    const size_t row = (size_t)DK;
    bf16* base = this->kv_cache.data() + (plane ? this->kv_cache_size / 2 : 0) + (size_t)ring_slot(pos) * row;
    buffer<bf16> r(DK);
    memcpy(r.data(), base, row * sizeof(bf16));
    return r;
}

void decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 32 * 256 * 5 / 8;
    const int blocks_per_row = col / 256;

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
}

void decoding_layer::load_weights(Q4NX& q4nx) {

    if (!s_embed_loaded) {
        this->token_embed->init_weights(q4nx, "model.embed_tokens");
        this->pli_embed->init_weights(q4nx, "model.per_layer_token_embd");
        s_embed_loaded = true;
    }

    u8* w_ptr = this->all_proj_weight_buffer.data();
    u8* w_ptr_end = this->all_proj_weight_buffer.end();
    {
        LOG_VERBOSE(1, "Loading q weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_q;
        q4nx.load_weights(w_q, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
        reorder_cpy(w_ptr, w_q, D);
        w_ptr += DQ * D * 5 / 8;
        if (!shall_skip_kv) {
            LOG_VERBOSE(1, "Loading kv weights");
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_k, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
            q4nx.load_weights(w_v, "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");
            reorder_cpy(w_ptr, w_k, D);
            w_ptr += DK * D * 5 / 8;
            reorder_cpy(w_ptr, w_v, D);
            w_ptr += DV * D * 5 / 8;
        }
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_o;
        q4nx.load_weights(w_o, "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight");
        reorder_cpy(w_ptr, w_o, DQ);
        w_ptr += DQ * D * 5 / 8;
    }
    // read up weights
    {
        LOG_VERBOSE(1, "Loading up weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up, "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight");
        q4nx.load_weights(w_gate, "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight");
        // For every D rows, half is up, half is gate
        const int blocks = 1024 / 2 / 32 * (D / 256);
        const int chunk_size = blocks * 32 * 256 * 5 / 8;
        const int total_size = intermediate_size * D * 5 / 8;
        LOG_VERBOSE(1, "Total up/gate weights size in bytes: " + std::to_string(total_size));
        const int phases = total_size / chunk_size;
        tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
        for (int i = 0; i < phases; i++)
        {
            assert(w_ptr < w_ptr_end);
            LOG_VERBOSE(1, "Copying up and gate weights to buffer, phase " << i + 1 << "/" << phases);
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
    }
    // read down weights
    {
        LOG_VERBOSE(1, "Loading down weights");
        assert(w_ptr < w_ptr_end);
        buffer<u8> w_down;
        q4nx.load_weights(w_down, "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");
        LOG_VERBOSE(1, "Copying down weights to buffer " + std::to_string(w_down.size()) + " bytes");
        reorder_cpy(w_ptr, w_down, intermediate_size);
        w_ptr += intermediate_size * D * 5 / 8;
    }
    // read input projection weights
    {
        LOG_VERBOSE(1, "Loading input projection weights");
        assert(w_ptr < w_ptr_end);
        buffer<bf16> w_inp_gate;
        q4nx.load_weights(w_inp_gate, "model.per_layer_model_proj.weight_layer" + std::to_string(layer_idx));
        memcpy(w_ptr, w_inp_gate.data(), (size_t)hidden_size_per_layer_input * D * sizeof(bf16));
        w_ptr += (size_t)hidden_size_per_layer_input * D * sizeof(bf16);
    }
    {
        LOG_VERBOSE(1, "Loading input projection weights");
        assert(w_ptr < w_ptr_end);
        buffer<bf16> w_inp_gate;
        q4nx.load_weights(w_inp_gate, "model.layers." + std::to_string(layer_idx) + ".inp_gate.weight");
        memcpy(w_ptr, w_inp_gate.data(), (size_t)hidden_size_per_layer_input * D * sizeof(bf16));
        w_ptr += (size_t)hidden_size_per_layer_input * D * sizeof(bf16);
    }
    {
        LOG_VERBOSE(1, "Loading input projection weights");
        assert(w_ptr < w_ptr_end);
        buffer<bf16> w_input_proj;
        q4nx.load_weights(w_input_proj, "model.layers." + std::to_string(layer_idx) + ".per_layer_projection.weight");
        memcpy(w_ptr, w_input_proj.data(), (size_t)hidden_size_per_layer_input * D * sizeof(bf16));
        w_ptr += (size_t)hidden_size_per_layer_input * D * sizeof(bf16);
    }
    assert(w_ptr == w_ptr_end);
    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post_attn;
        buffer<bf16> w_pre_feedforward_layernorm;
        buffer<bf16> w_post_feedforward_layernorm;

        bf16* w_rms_ptr = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
        q4nx.load_weights(w_rms_post_attn, "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
        q4nx.load_weights(w_pre_feedforward_layernorm, "model.layers." + std::to_string(layer_idx) + ".pre_feedforward_layernorm.weight");
        q4nx.load_weights(w_post_feedforward_layernorm, "model.layers." + std::to_string(layer_idx) + ".post_feedforward_layernorm.weight");
        header_print("info", "Input layer_norm:");
        utils::print_matrix(w_rms_pre, D);

        // w_rms_post.memset(1);
        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;      
        memcpy((unsigned char*)w_rms_ptr, w_rms_post_attn.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char*)w_rms_ptr, w_pre_feedforward_layernorm.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char*)w_rms_ptr, w_post_feedforward_layernorm.data(), D * sizeof(bf16));
        w_rms_ptr += D;
    }
    if (this->has_qk_norm)
    {
        LOG_VERBOSE(1, "Loading attention norm weights");
        buffer<bf16> q_attn_norm;
        buffer<bf16> k_attn_norm;
        q4nx.load_weights(q_attn_norm, "model.layers." + std::to_string(layer_idx) + ".self_attn.q_norm.weight");
        q4nx.load_weights(k_attn_norm, "model.layers." + std::to_string(layer_idx) + ".self_attn.k_norm.weight");
        memcpy(this->rope_rms_buffer.data() + DH, q_attn_norm.data(), DH * sizeof(bf16));
        memcpy(this->rope_rms_buffer.data() + DH + DH, k_attn_norm.data(), DH * sizeof(bf16));
    }
    {
        LOG_VERBOSE(1, "Loading per layer norm weights");
        buffer<bf16> per_layer_norm;
        q4nx.load_weights(per_layer_norm, "model.per_layer_proj_norm.weight");
        memcpy(this->rope_rms_buffer.data() + DH + DH + DH + hidden_size_per_layer_input, per_layer_norm.data(), hidden_size_per_layer_input * sizeof(bf16));
    }

    {
        buffer<bf16> layer_scale_buffer;
        q4nx.load_weights(layer_scale_buffer, "model.layers." + std::to_string(layer_idx) + ".layer_output_scale.weight");
        bf16* layer_scale_ptr = (this->rope_rms_buffer.data() + DH * 3 + hidden_size_per_layer_input * 2 + D);
        *layer_scale_ptr = layer_scale_buffer[0];
        std::cout << "Layer output scale: " << (float)(*layer_scale_ptr) << std::endl;
    }
    {
        LOG_VERBOSE(1, "Loading per layer input projection norm weights");
        buffer<bf16> per_layer_input_proj_norm;
        q4nx.load_weights(per_layer_input_proj_norm, "model.layers." + std::to_string(layer_idx) + ".post_layernorm.weight");
        memcpy(this->rope_rms_buffer.data() + DH + DH + DH + hidden_size_per_layer_input * 2, per_layer_input_proj_norm.data(), D * sizeof(bf16));
    }
  
    this->_set_rope_weights(0);
    this->kv_cache.memset((bf16)0.0f);

    this->all_proj_weight_buffer.sync_to_device();
    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    this->kv_cache.sync_to_device();
    header_print("info", "Weights loaded!");
}

buffer<bf16> decoding_layer::embedding(int idx) {
    if (idx < 0) {
        throw std::runtime_error("Invalid token index: " + std::to_string(idx));
    }
    if (idx >= 262144) {
        throw std::runtime_error("Token index out of bounds: " + std::to_string(idx) + ", max is 262143");
    }
    buffer<bf16> result = this->token_embed->forward(idx);
    return result;
}

buffer<bf16> decoding_layer::operator()(buffer<bf16>& x, int idx) {
    buffer<bf16> token_embd = this->token_embed->forward(idx);
    memcpy(this->hidden_states_buffer.data(), x.data(), D * sizeof(bf16));
    memcpy(this->hidden_states_buffer.data() + D * 2, token_embd.data(), D * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    buffer<bf16> per_layer_embedding = this->pli_embed->forward(idx);
    bf16* per_layer_input_embd = per_layer_embedding.data() + layer_idx * hidden_size_per_layer_input;
    memcpy(this->rope_rms_buffer.data() + DH * 3, per_layer_input_embd, hidden_size_per_layer_input * sizeof(bf16));
    this->rope_rms_buffer.sync_to_device();

    {
        auto t0 = std::chrono::steady_clock::now();
        xrt::run run = this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
        run.start();
        ert_cmd_state st = run.wait(std::chrono::milliseconds(dispatch_timeout_ms));
        last_dispatch_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        last_state = st;
        if (st != ERT_CMD_STATE_COMPLETED) {
            throw std::runtime_error("layer dispatch did not complete, state " + std::to_string((int)st) + " (layer " + std::to_string(layer_idx) + ", ctx " + std::to_string(current_context_length) + ")");
        }
    }
    this->hidden_states_buffer.sync_from_device();
    this->set_context_length(this->current_context_length + 1);
    return this->hidden_states_buffer;
}


float decoding_layer::bench(buffer<bf16>& input) {
    this->set_context_length(8);
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < num_hidden_layers; i++){
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache));
    }
    time_utils::time_point start = time_utils::now();
    rl.execute();
    rl.wait();
    this->hidden_states_buffer.sync_from_device();
    time_utils::time_point end = time_utils::now();
    time_utils::time_with_unit total_ms = time_utils::duration_ms(start, end);
    return 1000.0 / total_ms.first;
}


void decoding_layer::_set_rope_weights(int idx){
    static float inv_freq_global[256] = {
        1.0000e+00, 9.4746e-01, 8.9769e-01, 8.5053e-01, 8.0584e-01, 7.6351e-01, 7.2339e-01, 6.8539e-01,
        6.4938e-01, 6.1527e-01, 5.8294e-01, 5.5232e-01, 5.2330e-01, 4.9581e-01, 4.6976e-01, 4.4508e-01,
        4.2170e-01, 3.9954e-01, 3.7855e-01, 3.5866e-01, 3.3982e-01, 3.2197e-01, 3.0505e-01, 2.8903e-01,
        2.7384e-01, 2.5946e-01, 2.4582e-01, 2.3291e-01, 2.2067e-01, 2.0908e-01, 1.9810e-01, 1.8769e-01,
        1.7783e-01, 1.6849e-01, 1.5963e-01, 1.5125e-01, 1.4330e-01, 1.3577e-01, 1.2864e-01, 1.2188e-01,
        1.1548e-01, 1.0941e-01, 1.0366e-01, 9.8217e-02, 9.3057e-02, 8.8168e-02, 8.3536e-02, 7.9148e-02,
        7.4989e-02, 7.1050e-02, 6.7317e-02, 6.3780e-02, 6.0430e-02, 5.7255e-02, 5.4247e-02, 5.1397e-02,
        4.8697e-02, 4.6138e-02, 4.3714e-02, 4.1418e-02, 3.9242e-02, 3.7180e-02, 3.5227e-02, 3.3376e-02,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00
    };
    static float inv_freq_swa[128] = {
        1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01, 6.4938e-01, 6.0430e-01,
        5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01, 4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01,
        3.1623e-01, 2.9427e-01, 2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01,
        1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01, 1.1548e-01, 1.0746e-01,
        1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02, 7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02,
        5.6234e-02, 5.2330e-02, 4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
        3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02, 2.0535e-02, 1.9110e-02,
        1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02, 1.3335e-02, 1.2409e-02, 1.1548e-02, 1.0746e-02,
        1.0000e-02, 9.3057e-03, 8.6596e-03, 8.0584e-03, 7.4989e-03, 6.9783e-03, 6.4938e-03, 6.0430e-03,
        5.6234e-03, 5.2330e-03, 4.8697e-03, 4.5316e-03, 4.2170e-03, 3.9242e-03, 3.6517e-03, 3.3982e-03,
        3.1623e-03, 2.9427e-03, 2.7384e-03, 2.5483e-03, 2.3714e-03, 2.2067e-03, 2.0535e-03, 1.9110e-03,
        1.7783e-03, 1.6548e-03, 1.5399e-03, 1.4330e-03, 1.3335e-03, 1.2409e-03, 1.1548e-03, 1.0746e-03,
        1.0000e-03, 9.3057e-04, 8.6596e-04, 8.0584e-04, 7.4989e-04, 6.9783e-04, 6.4938e-04, 6.0430e-04,
        5.6234e-04, 5.2330e-04, 4.8697e-04, 4.5316e-04, 4.2170e-04, 3.9242e-04, 3.6517e-04, 3.3982e-04,
        3.1623e-04, 2.9427e-04, 2.7384e-04, 2.5483e-04, 2.3714e-04, 2.2067e-04, 2.0535e-04, 1.9110e-04,
        1.7783e-04, 1.6548e-04, 1.5399e-04, 1.4330e-04, 1.3335e-04, 1.2409e-04, 1.1548e-04, 1.0746e-04    
    };
    bf16* w_rope_ptr = this->rope_rms_buffer.data();
    float* inv_freq;
    if (this->is_swa_layer){
        inv_freq = inv_freq_swa;
    }
    else {
        inv_freq = inv_freq_global;
    }
    
    for (uint32_t i = 0; i < DH / 2; i++){
        w_rope_ptr[i] = bf16(std::cos(inv_freq[i] * idx) * this->rope_scaling);
        w_rope_ptr[i + DH / 2] = bf16(std::sin(inv_freq[i] * idx) * this->rope_scaling);
    }
    this->rope_rms_buffer.sync_to_device();
}


void decoding_layer::set_context_length(int L){

    this->current_context_length = L;

    this->_gen_sequence(this->layer_app.seq(), L + 1);

    this->_set_rope_weights(L);
}

void decoding_layer::_gen_sequence(npu_sequence* seq, int L) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    int glu_rtp_lock_id = 6;
    seq->clear_cmds();

    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], proj_swa_address, is_swa_layer? 1 : 0);
        seq->rtp_write(proj_tiles[i], proj_skip_address, shall_skip_kv? 1 : 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }
    const int L_local = (is_swa_layer && L > SWA_WINDOW) ? SWA_WINDOW : L;   // the kernel only sees the window length (ring replay below)
    seq->rtp_write(attn_qk_tile, l_qk_address, L_local);
    seq->rtp_write(attn_kv_tile, l_kv_address, L_local);
    seq->rtp_write(swa_attn_qk_tile, swa_l_qk_address, L_local);
    seq->rtp_write(swa_attn_kv_tile, swa_l_kv_address, L_local);
    seq->rtp_write(rms_tile, rms_swa_address, is_swa_layer? 1 : 0);
    seq->rtp_write(rms_tile, rms_skip_address, shall_skip_kv? 1 : 0);
    seq->rtp_write(rope_ct, rope_skip_kv_address, shall_skip_kv? 1 : 0);
    seq->rtp_write(swa_rope_ct, swa_rope_skip_kv_address, shall_skip_kv? 1 : 0);
    seq->rtp_write(glu_tile, glu_skip_address, (shall_skip_kv && use_double_wide_mlp)? 1 : 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(glu_tile, CT_lock_address_base + 16 * glu_rtp_lock_id, 1); // set lock to 1

    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_rope_rms_weights(seq);

    // receive y
    
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        IT3,
        bd_10,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(D + 0)},
        {0, 0, 0, 1},
        -1, 0, true
    );
    if (this->shall_skip_kv){
        _move_weights(seq, DQ, D, w_qkv_offset);
    }
    else {
        _move_weights(seq, DQ + DK + DV, D, w_qkv_offset);
    }
    
    if (!this->shall_skip_kv){
        _receive_kv_cache(seq, L); // also move qkv weights
    }

    _per_layer_input_path(seq);
    _move_kv_cache(seq, L);
    _move_weights(seq, D, DQ, w_o_offset);
    _move_weights(seq, 2 * intermediate_size, D, w_up_offset);
    _move_weights(seq, D, intermediate_size, w_down_offset);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT5,
        bd_14,
        it_channel_0,
        {0, 0, 0, (uint32_t)(w_inp_gate_offset)},
        {1, 1, 1, (uint32_t)(hidden_size_per_layer_input * D)},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT5,
        bd_15,
        it_channel_1,
        {0, 0, 0, (uint32_t)(w_inp_up_offset)},
        {1, 1, 1, (uint32_t)(hidden_size_per_layer_input * D)},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );

    // wait for receiving y
    seq->npu_dma_wait(
        IT3,
        S2MM,
        it_channel_0
    );
    seq->cmds2seq();
}

void decoding_layer::_per_layer_input_path(npu_sequence* seq){

    // move per layer input weights
    // send x, x, w
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_rms_arg_id,
        MM2S,
        IT4,
        bd_11,
        it_channel_0,
        {0, 0, 0, (uint32_t)(DH * 3)},
        {1, 1, 1, (uint32_t)(hidden_size_per_layer_input * 2 + D + 32)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT4,
        MM2S,
        it_channel_0
    );


    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        IT4,
        bd_12,
        it_channel_0,
        {0, 0, 0, (uint32_t)D * 2},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT4,
        bd_13,
        it_channel_1,
        {0, 0, 0, (uint32_t)w_per_layer_input_proj_offset},
        {1, 1, 1, (uint32_t)(hidden_size_per_layer_input * D)},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        IT4,
        MM2S,
        it_channel_1
    );



}

void decoding_layer::_send_hidden_states(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        xyr_tile,
        MM2S,
        it_channel_0
    );
}

void decoding_layer::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 4},
        {0, 0, 0, 1},
        0, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        xyr_tile,
        MM2S,
        it_channel_1
    );
}

void decoding_layer::_send_rope_rms_weights(npu_sequence* seq){
    if (is_swa_layer) {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            rope_rms_arg_id,
            MM2S,
            xyr_tile,
            bd_4,
            it_channel_1,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)(this->has_qk_norm? (DH + DH * 2): DH)},
            {0, 0, 0, 1},
            1, 0, true, no_cache
        );
        seq->npu_dma_wait(
            xyr_tile,
            MM2S,
            it_channel_1
        );
    }
    else {
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            rope_rms_arg_id,
            MM2S,
            xyr_tile,
            bd_4,
            it_channel_0,
            {0, 0, 0, 0},
            {1, 1, 1, (uint32_t)(this->has_qk_norm? (DH + DH * 2): DH)},
            {0, 0, 0, 1},
            1, 0, true, no_cache
        );
        seq->npu_dma_wait(
            xyr_tile,
            MM2S,
            it_channel_0
        );
    }
}

void decoding_layer::_receive_kv_cache(npu_sequence* seq, const int L){
    uint32_t data_per_chunk = this->num_kv_heads * this->DH;
    uint32_t v_offset = this->kv_cache_size / 2;
    uint32_t L_offset = (uint32_t)(is_swa_layer ? ((L - 1) % SWA_WINDOW) : (L - 1)) * data_per_chunk;
    
    npu_it_channel receiving_channel = is_swa_layer? it_channel_1 : it_channel_0;

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        attn_tile,
        bd_0,
        receiving_channel,
        {0, 0, 0, (uint32_t)L_offset},
        {1, 1, 1, data_per_chunk},
        {0, 0, 0, 1},
        -1, 0, true
    );

    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        receiving_channel
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        attn_tile,
        bd_1,
        receiving_channel,
        {0, 0, 0, (uint32_t)(L_offset + v_offset)},
        {1, 1, 1, data_per_chunk},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        receiving_channel
    );
}

void decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 5 / 8 / 2;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                proj_arg_id,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset)},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
        }
        if (round > 0){
            for (uint32_t col = 0; col < columns; col++){
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_0
                );
                seq->npu_dma_wait(
                    mvm_tiles[col],
                    MM2S,
                    it_channel_1
                );
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++){
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_0
        );
        seq->npu_dma_wait(
            mvm_tiles[col],
            MM2S,
            it_channel_1
        );
    }
}

void decoding_layer::_move_kv_cache(npu_sequence* seq, const size_t L){
    if (is_swa_layer && (L % SWA_WINDOW == 0 || L > (size_t)SWA_WINDOW)) {
        // same replay as the engine (gemma4e_npu_sequence::_move_kv_cache): corner = whole ring; wrapped = two phases
        const uint32_t v_off = this->kv_cache_size / 2;
        const uint32_t DKv = this->num_kv_heads * this->DH;
        auto mv = [&](npu_bd_id bd, npu_it_channel ch, uint32_t off, uint32_t n){
            seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_id, MM2S, attn_tile, bd, ch,
                {0, 0, 0, off}, {1, 1, 1, n}, {0, 0, 0, 1}, 13, 0, false, aggressive_cache);
        };
        if (L % SWA_WINDOW == 0) {
            mv(bd_8, it_channel_0, 0, SWA_WINDOW * DKv);
            mv(bd_9, it_channel_1, v_off, SWA_WINDOW * DKv);
        } else {
            uint32_t L_begin = L % SWA_WINDOW;
            uint32_t p1 = (SWA_WINDOW - L_begin) * DKv, p2 = L_begin * DKv, off = L_begin * DKv;
            mv(bd_8, it_channel_0, off, p1);
            mv(bd_9, it_channel_1, v_off + off, p1);
            mv(bd_10, it_channel_0, 0, p2);
            mv(bd_11, it_channel_1, v_off, p2);
        }
        return;
    }
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH * this->num_kv_heads;
    const uint32_t v_offset = this->kv_cache_size / 2;
    int pkt_id = is_swa_layer? 13 : 12;

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        attn_tile,
        bd_8,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        pkt_id, 0, true, aggressive_cache
    );

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        attn_tile,
        bd_9,
        it_channel_1,
        {0, 0, 0, (uint32_t)(v_offset)},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        pkt_id, 0, true, aggressive_cache
    );

    seq->npu_dma_wait(
        attn_tile,
        MM2S,
        it_channel_0
    );
    seq->npu_dma_wait(
        attn_tile,
        MM2S,
        it_channel_1
    );
}

buffer<bf16> decoding_layer::get_k_cache(int idx){
    this->kv_cache.sync_from_device();
    size_t data2copy = (size_t)this->DH * this->num_kv_heads * sizeof(bf16);
    bf16* k_cache_ptr = this->kv_cache.data() + idx * this->DH * this->num_kv_heads;
    buffer<bf16> k_cache(DK);
    bf16* dst_ptr = k_cache.data();
    memcpy(dst_ptr, k_cache_ptr, data2copy);
    return k_cache;
}

buffer<bf16> decoding_layer::get_v_cache(int idx){
    this->kv_cache.sync_from_device();
    size_t offset_per_group = (size_t)this->num_kv_heads * this->DH * this->MAX_L;
    size_t data2copy = (size_t)this->DH * this->num_kv_heads * sizeof(bf16);
    bf16* v_cache_ptr = this->kv_cache.data() + this->kv_cache_size / 2 + idx * this->DH * this->num_kv_heads;
    buffer<bf16> v_cache(DV);
    bf16* dst_ptr = v_cache.data();
    memcpy(dst_ptr, v_cache_ptr, data2copy);
    return v_cache;
}


void decoding_layer::load_kv_cache(buffer<bf16>& k_cache, buffer<bf16>& v_cache, int L){
    size_t data2copy = (size_t)L * DK * sizeof(bf16);
    this->kv_cache.sync_from_device();
    bf16* k_cache_ptr = this->kv_cache.data();
    bf16* v_cache_ptr = this->kv_cache.data() + this->kv_cache_size / 2;
    memcpy(k_cache_ptr, k_cache.data(), data2copy);
    memcpy(v_cache_ptr, v_cache.data(), data2copy);
    this->kv_cache.sync_to_device();
}