// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "linear_decoding_layer.hpp"

linear_decoding_layer::linear_decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int tb_size)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), tb_size(tb_size) {
    header_print("info", "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();
    this->lm_head = this->layer_app_mgr->create_app();
    this->current_context_length = 0;

    num_hidden_layers = lm_config.num_hidden_layers;
    linear_qk_head_dim = lm_config.linear_key_head_dim;
    linear_v_head_dim = lm_config.linear_value_head_dim;
    linear_num_qk_heads = lm_config.linear_num_key_heads;
    linear_num_v_heads = lm_config.linear_num_value_heads;

    linear_num_v_heads_padded = (linear_num_v_heads + 31) / 32 * 32;

    conv_kernel_size = lm_config.linear_conv_kernel_dim;
    Linear_DQ = linear_qk_head_dim * linear_num_qk_heads;
    Linear_DK = linear_qk_head_dim * linear_num_qk_heads;
    Linear_DV = linear_v_head_dim * linear_num_v_heads;
    intermediate_size = lm_config.intermediate_size;
    D = lm_config.hidden_size;

    l_qk_address = lm_config.addr_qk;
    l_kv_address = lm_config.addr_kv;
    l_begin_mha_address = lm_config.addr_l_begin_mha;
    l_end_mha_address = lm_config.addr_l_end_mha;
    kk_address = lm_config.addr_kk;
    rms_address = model_list::get_rms_addr(this->model_type);
    JSON_GET(this->num_experts, lm_config._json_config, "num_experts", 0, int);
    JSON_GET(this->num_activate_experts, lm_config._json_config, "num_experts_per_tok", 0, int);

    header_print("info", "Decoding layer config: ");
    header_print("info", "  D : " << D);
    header_print("info", "  Linear_DQ: " << Linear_DQ);
    header_print("info", "  Linear_DK: " << Linear_DK);
    header_print("info", "  Linear_DV: " << Linear_DV);
    header_print("info", "  Conv Kernel Size: " << conv_kernel_size);
    header_print("info", "  Intermediate Size: " << intermediate_size);
    has_qk_norm = model_list::has_qk_norm(this->model_type);

    conv_w_size = (size_t)conv_kernel_size * (Linear_DQ + Linear_DK + Linear_DV);
    linear_kv_cache_size = (conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV) + linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * 2; // conv cache + alpha beta cache
    s_offset = (size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV);
    
    // weight_desc_t linear_attn_qkv;
    // weight_desc_t attn_gate;
    // weight_desc_t ssm_a;
    // weight_desc_t ssm_alpha;
    // weight_desc_t ssm_beta;
    // weight_desc_t ssm_conv1d;
    // weight_desc_t ssm_dt;
    // weight_desc_t ssm_norm;
    // weight_desc_t ssm_out;
    // weight_desc_t input_layer_norm;
    // weight_desc_t post_attntion_norm;
    // weight_desc_t ffn_down_exps;
    // weight_desc_t ffn_down_shexp;
    // weight_desc_t ffn_gate_exps;
    // weight_desc_t ffn_gate_shexp;
    // weight_desc_t ffn_up_exps;
    // weight_desc_t ffn_up_shexp;
    // weight_desc_t ffn_gate_inp;
    // weight_desc_t ffn_gate_inp_shexp;
    _weight_desc.linear_attn_qkv = weight_desc_t(flm_q80, {D, Linear_DQ + Linear_DK + Linear_DV}, "model.layer.%d.linear_attn.qkv_proj.weight");
    _weight_desc.attn_gate = weight_desc_t(flm_q80, {D, Linear_DV}, "model.layer.%d.self_attn.gate_proj.weight");
    _weight_desc.ssm_a = weight_desc_t(flm_f32, {linear_num_v_heads}, "model.layer.%d.linear_attn.ssm_a");
    _weight_desc.ssm_alpha = weight_desc_t(flm_bf16, {D, linear_num_v_heads}, "model.layer.%d.linear_attn.ssm_alpha_proj.weight");
    _weight_desc.ssm_beta = weight_desc_t(flm_bf16, {D, linear_num_v_heads}, "model.layer.%d.linear_attn.ssm_beta_proj.weight");
    _weight_desc.ssm_conv1d = weight_desc_t(flm_bf16, {Linear_DV + Linear_DQ + Linear_DK, conv_kernel_size}, "model.layer.%d.linear_attn.ssm_conv1d.weight");
    _weight_desc.ssm_dt = weight_desc_t(flm_f32, {linear_num_v_heads}, "model.layer.%d.linear_attn.ssm_dt.bias");
    _weight_desc.ssm_norm = weight_desc_t(flm_bf16, {linear_v_head_dim}, "model.layer.%d.linear_attn.ssm_norm.weight");
    _weight_desc.ssm_out = weight_desc_t(flm_q80, {Linear_DV, D}, "model.layer.%d.linear_attn.ssm_out_proj.weight");
    _weight_desc.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.post_attention_layernorm.weight");
    _weight_desc.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.input_layernorm.weight");
    _weight_desc.ffn_down_exps = weight_desc_t(flm_q4k, {intermediate_size, D, num_experts}, "model.layer.%d.mlp.down_exps_proj.weight");
    _weight_desc.ffn_down_shexp = weight_desc_t(flm_q80, {intermediate_size, D}, "model.layer.%d.mlp.share_down_exps_proj.weight");
    _weight_desc.ffn_up_exps = weight_desc_t(flm_q4k, {D, intermediate_size, num_experts}, "model.layer.%d.mlp.up_exps_proj.weight");
    _weight_desc.ffn_up_shexp = weight_desc_t(flm_q80, {D, intermediate_size}, "model.layer.%d.mlp.share_up_exps_proj.weight");
    _weight_desc.ffn_gate_exps = weight_desc_t(flm_q4k, {D, intermediate_size, num_experts}, "model.layer.%d.mlp.gate_exps_proj.weight");
    _weight_desc.ffn_gate_shexp = weight_desc_t(flm_q80, {D, intermediate_size}, "model.layer.%d.mlp.share_gate_exps_proj.weight");
    _weight_desc.ffn_gate_inp = weight_desc_t(flm_bf16, {D, num_experts}, "model.layer.%d.moe_router.weight");
    _weight_desc.ffn_gate_inp_shexp = weight_desc_t(flm_bf16, {D}, "model.layer.%d.shared_expert_gate.weight");

    proj_weights_helper.add_weight(_weight_desc.ffn_up_exps);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate_exps);
    proj_weights_helper.add_weight(_weight_desc.ffn_down_exps);    
    proj_weights_helper.add_weight(_weight_desc.linear_attn_qkv);
    proj_weights_helper.add_weight(_weight_desc.attn_gate);
    proj_weights_helper.add_weight(_weight_desc.ssm_out);
    proj_weights_helper.add_weight(_weight_desc.ffn_up_shexp);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate_shexp);
    proj_weights_helper.add_weight(_weight_desc.ffn_down_shexp);

    


    // std::cout << proj_weights_helper.get_size() / 1024 / 1024 << std::endl;
    // assert(proj_weights_helper.get_size() < 512ull * 1024 * 1024);  // check to avoid 512 M problem
    
    norm_exp_helper.add_weight(_weight_desc.input_layer_norm);
    norm_exp_helper.add_weight(_weight_desc.post_attntion_norm);
    norm_exp_helper.add_weight(_weight_desc.ffn_gate_inp_shexp);    
    norm_exp_helper.add_weight(_weight_desc.ffn_gate_inp);


    rope_ssm_weights_helper.add_weight(_weight_desc.ssm_conv1d);
    rope_ssm_weights_helper.add_weight(_weight_desc.ssm_norm);
    rope_ssm_weights_helper.add_weight(_weight_desc.ssm_a);
    rope_ssm_weights_helper.add_weight(_weight_desc.ssm_dt);
    rope_ssm_weights_helper.add_weight(_weight_desc.ssm_alpha);
    rope_ssm_weights_helper.add_weight(_weight_desc.ssm_beta);
    for (weight_desc_t* w : _weight_desc.all()) {
        std::cout << "[offset] " << w->name << " = " << w->offset << std::endl;
    }
    
   

    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(D + tb_size);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(proj_weights_helper.get_size());
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<u8>(norm_exp_helper.get_size());
    this->rope_rms_buffer = this->layer_app.create_bo_buffer<u8>(rope_ssm_weights_helper.get_size());
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(linear_kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();
    this->_gen_linear_sequence(this->layer_app.seq());

}

void linear_decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks, int block_for_512_reorder)
{
    if (block_for_512_reorder == -1) block_for_512_reorder = vertical_blocks;
    assert(is_quantize(dtype));
    assert(vertical_blocks % block_for_512_reorder == 0);

    const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);

    const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
    const int rows = src.size() / a_block_size / blocks_per_row;

    // Step 1: vertical block reorder at a_block_size granularity
    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;

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
                src_ptr[i] = src.data(); // useless padding
        }
    }

    if (vertical_blocks == 1)
        return;

    // Step 2: within each group of vertical_blocks consecutive blocks in the output,
    // apply 512-byte interleave within sub-groups of block_for_512_reorder blocks.
    // e.g. vertical_blocks=4, block_for_512_reorder=2:
    //   before: [A_full, B_full, C_full, D_full]
    //   after:  [A[0:512],B[0:512], A[512:1024],B[512:1024], ..., C[0:512],D[0:512], ...]
    const int slice_size = 512;
    assert(a_block_size % slice_size == 0);
    const int slices = a_block_size / slice_size;

    const int num_vblocks_groups = (rows / vertical_blocks) * blocks_per_row;
    std::vector<uint8_t> temp((size_t)vertical_blocks * a_block_size);

    u8 *blk_ptr = dst;
    for (int g = 0; g < num_vblocks_groups; g++)
    {
        memcpy(temp.data(), blk_ptr, temp.size());
        u8 *out = blk_ptr;

        for (int sub = 0; sub < vertical_blocks; sub += block_for_512_reorder)
        {
            for (int s = 0; s < slices; s++)
            {
                for (int i = sub; i < sub + block_for_512_reorder; i++)
                {
                    memcpy(out, temp.data() + i * a_block_size + s * slice_size, slice_size);
                    out += slice_size;
                }
            }
        }
        blk_ptr += vertical_blocks * a_block_size;
    }
}

void linear_decoding_layer::load_weights(Q4NX& q4nx, int layer_idx) {
    constexpr int GLU_SLICE = 1024;
    constexpr int UP_GATE_OUTDIM_REORDEER = 512;
    constexpr int EXP_UP_GATE_OUTDIM_REORDER= 128;    
    //constexpr int GLU_OUTDIM = 512;

    u8* w_ptr_begin = this->all_proj_weight_buffer.data();


    // read up expert weights
    {
        LOG_VERBOSE(1, "Loading up/gate expert weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up_exps;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate_exps;
        buffer<u8> w_up_exps;
        buffer<u8> w_gate_exps;
        q4nx.load_weights(w_up_exps, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate_exps, desc_gate.format_name(layer_idx));

        u8* w_ptr = w_ptr_begin + desc_up.offset;

        size_t size_per_expert = desc_up.get_size() / num_experts;
        assert(size_per_expert == desc_gate.get_size() / num_experts);

        int slices = desc_up.shape[1] / EXP_UP_GATE_OUTDIM_REORDER;
        size_t slice_size = size_per_expert / slices;

        for(int e = 0; e < num_experts; e++) {
            buffer<u8> w_up_exp = buffer<u8>(w_up_exps.data() + e * size_per_expert, size_per_expert);
            buffer<u8> w_gate_exp = buffer<u8>(w_gate_exps.data() + e * size_per_expert, size_per_expert);
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up_exp.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate_exp.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, D, desc_up.dtype,4,2);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, D, desc_gate.dtype,4,2);
                w_ptr += slice_size;
            }
        }
        desc_up.load();
        desc_gate.load();
    }
    // read down expert weights
    {
        LOG_VERBOSE(1, "Loading down expert weights");
        weight_desc_t& desc = _weight_desc.ffn_down_exps;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w_down_exps;
        q4nx.load_weights(w_down_exps, desc.format_name(layer_idx));
        reorder_cpy(w_ptr, w_down_exps, intermediate_size, desc.dtype,4,2);
        desc.load();
    }



    {
        LOG_VERBOSE(1, "Loading qkv weights");
        weight_desc_t& desc = _weight_desc.linear_attn_qkv;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w;
        q4nx.load_weights(w, desc.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        reorder_cpy(w_ptr, w, D, desc.dtype);
        desc.load();
    }
    // read gate weights
    {
        LOG_VERBOSE(1, "Loading gate weights");
        weight_desc_t& desc = _weight_desc.attn_gate;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w;
        q4nx.load_weights(w, desc.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        reorder_cpy(w_ptr, w, D, desc.dtype);
        desc.load();
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        weight_desc_t& desc = _weight_desc.ssm_out;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w;
        q4nx.load_weights(w, desc.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        reorder_cpy(w_ptr, w, Linear_DV, desc.dtype);
        desc.load();
    }
    // read up weights
    // read shared expert up weights
    {
        LOG_VERBOSE(1, "Loading shared up/gate weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up_shexp;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate_shexp;
        buffer<u8> w_up_shexp;
        buffer<u8> w_gate_shexp;
        q4nx.load_weights(w_up_shexp, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate_shexp, desc_gate.format_name(layer_idx));

        u8* w_ptr = w_ptr_begin + desc_up.offset;

        assert(desc_up.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);
        assert(desc_gate.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);

        int slices = desc_up.shape[1] / UP_GATE_OUTDIM_REORDEER;
        size_t slice_size = desc_up.get_size() / slices;
        for (int i = 0; i < slices; i++) {
            buffer<u8> up_slice = buffer<u8>(w_up_shexp.data() + i * slice_size, slice_size);
            buffer<u8> gate_slice = buffer<u8>(w_gate_shexp.data() + i * slice_size, slice_size);
            reorder_cpy(w_ptr, up_slice, D, _weight_desc.ffn_up_shexp.dtype);
            w_ptr += slice_size;
            reorder_cpy(w_ptr, gate_slice, D, _weight_desc.ffn_gate_shexp.dtype);
            w_ptr += slice_size;
        }
        desc_up.load();
        desc_gate.load();
    }
    // read shared expert down weights
    {
        LOG_VERBOSE(1, "Loading down weights");
        weight_desc_t& desc = _weight_desc.ffn_down_shexp;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w;
        q4nx.load_weights(w, desc.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        reorder_cpy(w_ptr, w, intermediate_size, desc.dtype);
        desc.load();
    }


    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;

        u8* rms_weights_begin = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, _weight_desc.input_layer_norm.format_name(layer_idx));
        q4nx.load_weights(w_rms_post, _weight_desc.post_attntion_norm.format_name(layer_idx));
        header_print("info", "Input layer_norm:");
        utils::print_matrix(w_rms_pre, D);

        // w_rms_post.memset(1);
        memcpy(rms_weights_begin + _weight_desc.input_layer_norm.offset, w_rms_pre.data(), D * sizeof(bf16));
        memcpy(rms_weights_begin + _weight_desc.post_attntion_norm.offset, w_rms_post.data(), D * sizeof(bf16));
        _weight_desc.input_layer_norm.load();
        _weight_desc.post_attntion_norm.load();

        // load ffn_gate_inp, ffn_gate_inp_shexp
        buffer<bf16> w_gate_inp;
        buffer<bf16> w_gate_inp_shexp;
        q4nx.load_weights(w_gate_inp, _weight_desc.ffn_gate_inp.format_name(layer_idx));
        q4nx.load_weights(w_gate_inp_shexp, _weight_desc.ffn_gate_inp_shexp.format_name(layer_idx));
        memcpy(rms_weights_begin + _weight_desc.ffn_gate_inp_shexp.offset, w_gate_inp_shexp.data(), _weight_desc.ffn_gate_inp_shexp.get_size());
        memcpy(rms_weights_begin + _weight_desc.ffn_gate_inp.offset, w_gate_inp.data(), _weight_desc.ffn_gate_inp.get_size());

        _weight_desc.ffn_gate_inp.load();
        _weight_desc.ffn_gate_inp_shexp.load();
    }

    u8* other_weights_begin = rope_rms_buffer.data();
    {
        // load conv_weights
        LOG_VERBOSE(1, "Loading ssm conv weights");
        weight_desc_t& desc = _weight_desc.ssm_conv1d;
        u8* w_ptr = other_weights_begin + desc.offset;
        buffer<bf16> conv_weights;
        q4nx.load_weights(conv_weights, desc.format_name(layer_idx));
        memcpy(w_ptr, conv_weights.data(), conv_weights.size() * sizeof(bf16));
        desc.load();
    }

    {
        // load ssm_norm_weights
        LOG_VERBOSE(1, "Loading ssm norm weights");
        weight_desc_t& desc = _weight_desc.ssm_norm;
        u8* w_ptr = other_weights_begin + desc.offset;
        buffer<bf16> norm_weights;
        q4nx.load_weights(norm_weights, desc.format_name(layer_idx));
        memcpy(w_ptr, norm_weights.data(), linear_v_head_dim * sizeof(bf16));
        desc.load();
    }
    {
        // load ssm a, dt
        LOG_VERBOSE(1, "Loading ssm a, dt weights");
        weight_desc_t& desc_a = _weight_desc.ssm_a;
        weight_desc_t& desc_dt = _weight_desc.ssm_dt;
        buffer<float> a;
        buffer<float> dt;

        q4nx.load_weights(a, desc_a.format_name(layer_idx));
        q4nx.load_weights(dt, desc_dt.format_name(layer_idx));
        memcpy(other_weights_begin + desc_a.offset, a.data(), linear_num_v_heads * sizeof(float));
        memcpy(other_weights_begin + desc_dt.offset, dt.data(), linear_num_v_heads * sizeof(float));
        
        desc_a.load();
        desc_dt.load();
    }
    {
        // load ssm alpha, beta
        LOG_VERBOSE(1, "Loading ssm alpha, beta weights");
        weight_desc_t& desc_alpha = _weight_desc.ssm_alpha;
        weight_desc_t& desc_beta = _weight_desc.ssm_beta;
        buffer<bf16> alpha;
        buffer<bf16> beta;

        q4nx.load_weights(alpha, desc_alpha.format_name(layer_idx));
        q4nx.load_weights(beta, desc_beta.format_name(layer_idx));
      
        memcpy(other_weights_begin + desc_alpha.offset, alpha.data(), desc_alpha.get_size());
        memcpy(other_weights_begin + desc_beta.offset, beta.data(), desc_beta.get_size());
        
        desc_alpha.load();
        desc_beta.load();
    }

    this->kv_cache.memset((bf16)0.0f);

    this->all_proj_weight_buffer.sync_to_device();
    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    this->kv_cache.sync_to_device();
    
    bool all_ready = true;
    for (weight_desc_t* w : _weight_desc.all()) {
        if (!w->ready()) {
            header_print("error", "Weight not ready: " << w->name);
            all_ready = false;
        }
    }
    if (all_ready) {
        header_print("info", "Weights ready!");
    } else {
        header_print("error", "Some weights were not ready!");
    }
}

void rms_norm(buffer<bf16>&y, buffer<bf16>&x, bf16* w) {
    int D = y.size();
    const float epsilon = 1e-6f;

    float sum_squares = 0.0f;
    for (int i = 0; i < D; i++) {
        float xi = (float)x[i];
        sum_squares += xi * xi;
    }

    float divrms = 1.0f / std::sqrt(sum_squares / (float)D + epsilon);
    for (int i = 0; i < D; i++) {
        y[i] = (bf16)((float)x[i] * divrms * (float)w[i]);
    }
}

void mvm(buffer<bf16>&y, buffer<bf16>&x, bf16* w){
    int D_in = x.size();
    int D_out = y.size();

    for (int j = 0; j < D_out; j++) {
        float acc = 0.0f;
        for (int i = 0; i < D_in; i++) {
            acc += (float)x[i] * (float)w[j * D_in + i];
        }
        y[j] = (bf16)acc;
    }
}

buffer<bf16> linear_decoding_layer::operator()(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    // mvm(alpha, y, (bf16*)(this->rope_rms_buffer.data() + _weight_desc.ssm_alpha.offset));
    this->hidden_states_buffer.sync_to_device();
    this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
    this->hidden_states_buffer.sync_from_device();
    return this->hidden_states_buffer;
}

float linear_decoding_layer::bench(buffer<bf16>& input) {
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

void linear_decoding_layer::_gen_linear_sequence(npu_sequence* seq) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    int gate_rtp_lock_id = 6;
    int AB_proj_rtp_lock_id = 14;
    int router_shimtile_configure_ready_lock = 10; 
    seq->clear_cmds();

    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], 17408, 0);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }



    seq->rtp_write(rms_tile, rms_address, 0);
    seq->rtp_write(gate_tile, 49152, 0);
    seq->rtp_write(ab_proj_tile, 14592, 0);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(gate_tile, CT_lock_address_base + 16 * gate_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(ab_proj_tile, CT_lock_address_base + 16 * AB_proj_rtp_lock_id, 1);
    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_linear_conv_weights(seq);

    // // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        IT2,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );  
    
    // //receive tb
    // if (tb_size > 0)
    //     std::cout << "TB enabled!" << std::endl;
    //     seq->npu_dma_memcpy_nd(
    //         sizeof(bf16),
    //         xy_arg_idx,
    //         S2MM,
    //         IT5,
    //         bd_10,
    //         it_channel_0,
    //         {0, 0, 0, (uint32_t)D},
    //         {1, 1, 1, (uint32_t)tb_size},
    //         {0, 0, 0, 1},
    //         -1, 0, true
    //     );

    _move_linear_kv_cache(seq);
    _move_alpha_beta_weights(seq, linear_num_v_heads_padded * 2, D, _weight_desc.ssm_alpha.offset);
    _move_weights_q80(seq, Linear_DQ + Linear_DK + Linear_DV, D, _weight_desc.linear_attn_qkv.offset);
    _receive_linear_kv_cache(seq); // also move qkv weights
    _move_s(seq);
    _move_weights_q80(seq, Linear_DV, D, _weight_desc.attn_gate.offset);
    _move_weights_q80(seq, D, Linear_DV, _weight_desc.ssm_out.offset);
    


    for(int i = 0; i < 4; i++){
        setup_expert_up_gate_q4k(
            seq, i,
            intermediate_size, D,
            4, 5,
            _weight_desc.ffn_up_exps.offset);

        setup_expert_down_gate_q4k(
            seq, i,
            intermediate_size, D,
            6, 7,
            _weight_desc.ffn_up_exps.offset);

    }

    for(int i = 0; i < 4; i++){
        setup_expert_up_gate_q4k(
            seq, i,
            intermediate_size, D,
            12, 13,
            _weight_desc.ffn_up_exps.offset);

        setup_expert_down_gate_q4k(
            seq, i,
            intermediate_size, D,
            14, 15,
            _weight_desc.ffn_up_exps.offset);

    }





    auto router_CT = get_tile(4, 5);
    seq->npu_maskwrite(router_CT, 0x32038, 0x1, 0x1);
    seq->rtp_write(router_CT, CT_lock_address_base + 16 * router_shimtile_configure_ready_lock, 1); // set lock to 1





    _send_router_w_and_share_exp_gate(seq,num_experts, D, _weight_desc.ffn_gate_inp.offset);


    _move_weights_q80(seq, 2 * intermediate_size, D, _weight_desc.ffn_up_shexp.offset);
    _move_weights_q80(seq, D, intermediate_size, _weight_desc.ffn_down_shexp.offset);




            

    // // wait for receiving y
    seq->npu_dma_wait(
        IT2,
        S2MM,
        it_channel_1
    );
    // if (tb_size > 0) { seq->npu_dma_wait(
    //         IT5,
    //         S2MM,
    //         it_channel_0
    //     );
    // }
    seq->cmds2seq();
}



void linear_decoding_layer::_send_hidden_states(npu_sequence* seq){
    // send x
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void linear_decoding_layer::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 3},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void linear_decoding_layer::_send_linear_conv_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)conv_w_size + linear_v_head_dim + linear_num_v_heads_padded * 2 * 2},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

void linear_decoding_layer::_receive_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        S2MM,
        it_channel_1
    );
}

void linear_decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 / 2;
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
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
    

void linear_decoding_layer::_move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = 8704 / sizeof(bf16);
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    for (size_t round = 0; round < Dout / m / cores; round++){
        uint32_t bd_offset = (round % 2) * 8;
        for (uint32_t col = 0; col < columns; col++){
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_1 + bd_offset),
                it_channel_0,
                {0, 0, 0, (uint32_t)((round * cores + col * 4) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
                {1, 1, 1, 2 * blocks_per_row * a_block_size},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_proj_arg_idx,
                MM2S,
                mvm_tiles[col],
                npu_bd_id(bd_2 + bd_offset),
                it_channel_1,
                {0, 0, 0, (uint32_t)((round * cores + col * 4 + 2) * a_block_size * blocks_per_row + (uint32_t)w_offset / sizeof(bf16))},
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

void linear_decoding_layer::_move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    // currently, only on DMA channel 0 of IT5
    size_t router_weight_row_block = 32;
    size_t router_weight_col_block = 256;

    assert(Dout %2 == 0);

    // send alpha
    const size_t alpha_beta_Dout = Dout/2;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_13),
        it_channel_0,
        {0, 0, 0, (uint32_t)w_offset / sizeof(bf16)},
        {       alpha_beta_Dout/router_weight_row_block                      ,Din/router_weight_col_block           ,router_weight_col_block    ,router_weight_row_block},
        {       router_weight_row_block,                            alpha_beta_Dout*router_weight_col_block          ,alpha_beta_Dout                 ,1},
        3, 0, false, aggressive_cache
    );

    // send beta
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_14),
        it_channel_0,
        {0, 0, 0, (uint32_t)w_offset / sizeof(bf16) +   alpha_beta_Dout*Din },
        {       alpha_beta_Dout/router_weight_row_block                      ,Din/router_weight_col_block           ,router_weight_col_block    ,router_weight_row_block},
        {       router_weight_row_block,                            alpha_beta_Dout*router_weight_col_block          ,alpha_beta_Dout                 ,1},
        3, 0, false, aggressive_cache
    );

    // seq->npu_dma_wait(
    //     IT5,
    //     MM2S,
    //     it_channel_0
    // );
}

void linear_decoding_layer::_move_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

void linear_decoding_layer::_move_s(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        IT4,
        bd_2,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)(linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * 2)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        IT4,
        bd_3,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)(linear_num_v_heads * linear_v_head_dim * linear_qk_head_dim * 2)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
    // seq->npu_dma_wait(
    //     IT4,
    //     MM2S,
    //     it_channel_0
    // );

    // seq->npu_dma_wait(
    //     IT4,
    //     S2MM,
    //     it_channel_0
    // );
    // for (int h = 0; h < 32; h++){
    //     seq->npu_dma_memcpy_nd(
    //         sizeof(bf16),
    //         kv_cache_arg_idx,
    //         MM2S,
    //         IT4,
    //         bd_2,
    //         it_channel_0,
    //         {0, 0, 0, (uint32_t)s_offset + h * (linear_v_head_dim * linear_qk_head_dim * 2)},
    //         {1, 1, 1, (uint32_t)(linear_v_head_dim * linear_qk_head_dim * 2)},
    //         {0, 0, 0, 1},
    //         -1, 0, true, no_cache
    //     );
    //     seq->npu_dma_wait(
    //         IT4,
    //         MM2S,
    //         it_channel_0
    //     );
    //     seq->npu_dma_memcpy_nd(
    //         sizeof(bf16),
    //         kv_cache_arg_idx,
    //         S2MM,
    //         IT4,
    //         bd_3,
    //         it_channel_0,
    //         {0, 0, 0, (uint32_t)s_offset + h * (linear_v_head_dim * linear_qk_head_dim * 2)},
    //         {1, 1, 1, (uint32_t)(linear_v_head_dim * linear_qk_head_dim * 2)},
    //         {0, 0, 0, 1},
    //         -1, 0, true, no_cache
    //     );
    //     seq->npu_dma_wait(
    //         IT4,
    //         S2MM,
    //         it_channel_0
    //     );
    // }
}





void linear_decoding_layer::_send_router_w_and_share_exp_gate(npu_sequence* seq, const size_t NUM_expert, const size_t D, const size_t W_offset_in_byte){
    
    //TODO: have this aligned with the MT parametesr
    // router weights is NUM_expert x D in row major

    // but transfer router_weight in block sof router_weight_row_block x router_wieght_col_block
    // each block is column major order
    // but the blocks are row-block major order

    //TODO: need to share same file with decode for sanity 
    size_t router_weight_row_block = 32;
    size_t router_weight_col_block = 256;


    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_12),
        it_channel_0,
        {0, 0, 0, (uint32_t)((W_offset_in_byte/2)  )},
        {       NUM_expert/router_weight_row_block                      ,D/router_weight_col_block           ,router_weight_col_block    ,router_weight_row_block},
        {       router_weight_row_block,                            NUM_expert*router_weight_col_block          ,NUM_expert                 ,1},
        10, 0, false, aggressive_cache
    );
 
    // seq->npu_dma_memcpy_nd(
    //     sizeof(float),
    //     w_rms_arg_idx,
    //     MM2S,
    //     IT5,
    //     npu_bd_id(bd_13),
    //     it_channel_0,
    //     {0, 0, 0, (uint32_t)W_offset_in_fp32   + (D* NUM_expert)},
    //     {1, 1, 1, (uint32_t)D},
    //     {0, 0, 0, 1},
    //     10, 0, false, aggressive_cache
    // );
  

    

}








void linear_decoding_layer::send_manual_expert_up_gate_q4k(
    npu_sequence*seq, 
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int expert_id, size_t const_w_byte_offset
){


    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    size_t expert_up_gate_block =  (size_t)((2*intermediate)/(m)) * ( D/k);
    //size_t expert_down_block = (D/m)*(intermediate/k);

    size_t cur_expert_w_offset_in_byte = const_w_byte_offset + expert_id* size_t(a_block_size* expert_up_gate_block );

    size_t D_Q4nx_BLOCK_PER_ROW = D/k;
    size_t gate_up_col_number = intermediate/ (num_core_per_column*m);

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_4),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,true
    );// TODO: change later, not submit to queue by default

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_5), //TODO: different number for down?
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte +(2*a_block_size) },
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,true
    );




}






void linear_decoding_layer::send_manual_expert_down_gate_q4k(

    npu_sequence*seq, 
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int expert_id, size_t const_w_byte_offset
){

    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    //size_t expert_up_gate_block =  ((2*intermediate)/(m)) * ( D/k);
    size_t expert_down_block = (size_t)(D/m)*(intermediate/k);

    size_t cur_expert_w_offset_in_byte = const_w_byte_offset + expert_id*a_block_size * (expert_down_block );

    size_t INT_Q4nx_BLOCK_PER_ROW = intermediate/k;
    size_t down_col_number = D/ (num_core_per_column*m);


    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_6),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
        -1,0,true
    );// TODO: change l




    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_7),
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte + 2*a_block_size},
        {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
        -1,0,true
    );// TODO: change l

}




void linear_decoding_layer::setup_expert_down_gate_q4k(

    npu_sequence*seq, 
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int bd_id_channel_0, int bd_id_channel_1,
    size_t const_w_byte_offset
){

    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    //size_t expert_up_gate_block =  ((2*intermediate)/(m)) * ( D/k);
    size_t expert_down_block = (size_t)(D/m)*(intermediate/k);

    size_t cur_expert_w_offset_in_byte = const_w_byte_offset;// + expert_id*a_block_size * (expert_down_block );

    size_t INT_Q4nx_BLOCK_PER_ROW = intermediate/k;
    size_t down_col_number = D/ (num_core_per_column*m);
    //TODO: sync with the kernel through share hpp
    constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_0),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false         
    );// TODO: change l




    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_1),
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte },
        {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false         
    );// TODO: change l

}


void linear_decoding_layer::setup_expert_up_gate_q4k(
    npu_sequence*seq, 
    int col, // the shimtile for this expert, 0-3
    int intermediate, int D,
    int bd_id_channel_0, int bd_id_channel_1,    
    size_t const_w_byte_offset
){


    const int m = 32;
    const int k = 256;
    const int num_core_per_column=4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 37 / 64 ;

    //TODO: parameterized it
    size_t expert_up_gate_block =  (size_t)((2*intermediate)/(m)) * ( D/k);
    //size_t expert_down_block = (D/m)*(intermediate/k);

    size_t cur_expert_w_offset_in_byte = const_w_byte_offset;// + expert_id* size_t(a_block_size* expert_up_gate_block );

    size_t D_Q4nx_BLOCK_PER_ROW = D/k;
    size_t gate_up_col_number = intermediate/ (num_core_per_column*m);
    //TODO: sync with the kernel through share hpp
    constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;
    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_0),
        it_channel_0,
        {0,0,0,cur_expert_w_offset_in_byte},
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false
    );// TODO: change later, not submit to queue by default

    seq->npu_dma_memcpy_nd(
        sizeof(uint8_t),
        w_proj_arg_idx,
        MM2S,
        mvm_tiles[col],
        npu_bd_id(bd_id_channel_1), //TODO: different number for down?
        it_channel_1,
        {0,0,0,cur_expert_w_offset_in_byte },
        {gate_up_col_number*2,                        D_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
        {4*a_block_size*D_Q4nx_BLOCK_PER_ROW, 4*a_block_size, 256,1},

        -1,0,false,
        aggressive_cache,
        SHIM_6_CONTROL_RES_PKT, false        
    );

}


buffer<bf16> linear_decoding_layer::get_conv_cache(){
    buffer<bf16> conv_cache_buffer = buffer<bf16>((size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV));
    this->kv_cache.sync_from_device();
    memcpy(conv_cache_buffer.data(), this->kv_cache.data(), (size_t)(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV) * sizeof(bf16));
    return conv_cache_buffer;
}


buffer<bf16> linear_decoding_layer::get_S_cache(int head){
    buffer<bf16> s_cache_buffer = buffer<bf16>((size_t)linear_v_head_dim * linear_qk_head_dim);
    this->kv_cache.sync_from_device();
    float * s_begin = (float*)(this->kv_cache.data() + s_offset + head * linear_v_head_dim * linear_qk_head_dim * 2);
    for (int i = 0; i < linear_v_head_dim * linear_qk_head_dim; i++) {
        s_cache_buffer[i] = (bf16)s_begin[i];
    }
    return s_cache_buffer;
}

void linear_decoding_layer::clear_cache(){
    this->kv_cache.sync_from_device();
    this->kv_cache.memset((bf16)0.0f);
    this->kv_cache.sync_to_device();
}