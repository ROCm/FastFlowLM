// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "decoding_layer.hpp"

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int context_length)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), MAX_L(context_length) {
    header_print("info", "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();

    this->current_context_length = 0;

    num_hidden_layers = lm_config.num_hidden_layers;
    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    num_kv_heads = lm_config.num_key_value_heads;
    this->num_kv_per_round = 2;
    intermediate_size = lm_config.intermediate_size;
    D = lm_config.hidden_size;
    kv_cache_size = (size_t)MAX_L * (DK + DV);

    const float rope_partial_factor = 0.25f; // use 25% of head_dim for RoPE, the rest for QK norm if applicable

    RoPE_DIM = int(DH * rope_partial_factor);

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
    header_print("info", "  DQ: " << DQ);
    header_print("info", "  DK: " << DK);
    header_print("info", "  DV: " << DV);
    header_print("info", "  RoPE Dim: " << RoPE_DIM);
    header_print("info", "  Intermediate Size: " << intermediate_size);
    header_print("info", "  Experts: " << num_activate_experts << "/" << num_experts);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    header_print("info", "  Has QK Norm: " << (has_qk_norm ? "Yes" : "No"));
    header_print("info", "  Num KV Per Round: " << this->num_kv_per_round);
    this->rope_scaling = model_list::get_rope_scaling(this->model_type);

    _weight_desc.attn_k = weight_desc_t(flm_q80, {D, DK}, "model.layer.%d.self_attn.k_proj.weight");
    _weight_desc.attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.k_norm.weight");
    _weight_desc.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.input_layernorm.weight");
    _weight_desc.attn_q = weight_desc_t(flm_q80, {D, DQ}, "model.layer.%d.self_attn.q_proj.weight");
    _weight_desc.attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.q_norm.weight");
    _weight_desc.attn_v = weight_desc_t(flm_q80, {D, DV}, "model.layer.%d.self_attn.v_proj.weight");
    _weight_desc.ffn_down_exps = weight_desc_t(flm_q4k, {intermediate_size, D, num_experts}, "model.layer.%d.mlp.down_exps_proj.weight");
    _weight_desc.ffn_down_shexp = weight_desc_t(flm_q80, {intermediate_size, D}, "model.layer.%d.mlp.share_down_exps_proj.weight");
    _weight_desc.ffn_up_exps = weight_desc_t(flm_q4k, {D, intermediate_size, num_experts}, "model.layer.%d.mlp.up_exps_proj.weight");
    _weight_desc.ffn_up_shexp = weight_desc_t(flm_q80, {D, intermediate_size}, "model.layer.%d.mlp.share_up_exps_proj.weight");
    _weight_desc.ffn_gate_exps = weight_desc_t(flm_q4k, {D, intermediate_size, num_experts}, "model.layer.%d.mlp.gate_exps_proj.weight");
    _weight_desc.ffn_gate_shexp = weight_desc_t(flm_q80, {D, intermediate_size}, "model.layer.%d.mlp.share_gate_exps_proj.weight");
    _weight_desc.ffn_gate_inp = weight_desc_t(flm_bf16, {D, num_experts}, "model.layer.%d.moe_router.weight");
    _weight_desc.ffn_gate_inp_shexp = weight_desc_t(flm_bf16, {D}, "model.layer.%d.shared_expert_gate.weight");
    _weight_desc.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.post_attention_layernorm.weight");
    _weight_desc.attn_output = weight_desc_t(flm_q80, {DQ, D}, "model.layer.%d.self_attn.o_proj.weight");
    _weight_desc.attn_gate = weight_desc_t(flm_q80, {D, DQ}, "model.layer.%d.self_attn.q_proj.weight"); // special actully merged with q



    proj_weights_helper.add_weight(_weight_desc.ffn_up_exps);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate_exps);
    proj_weights_helper.add_weight(_weight_desc.ffn_down_exps);    
    proj_weights_helper.add_weight(_weight_desc.attn_q);
    proj_weights_helper.add_weight(_weight_desc.attn_k);
    proj_weights_helper.add_weight(_weight_desc.attn_v);
    proj_weights_helper.add_weight(_weight_desc.attn_gate);
    proj_weights_helper.add_weight(_weight_desc.attn_output);
    proj_weights_helper.add_weight(_weight_desc.ffn_up_shexp);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate_shexp);
    proj_weights_helper.add_weight(_weight_desc.ffn_down_shexp);


    assert(proj_weights_helper.get_size() < 512ull * 1024 * 1024);  // check to avoid 512 M problem

    norm_exp_helper.add_weight(_weight_desc.input_layer_norm);
    norm_exp_helper.add_weight(_weight_desc.post_attntion_norm);
    norm_exp_helper.add_weight(_weight_desc.ffn_gate_inp_shexp);    
    norm_exp_helper.add_weight(_weight_desc.ffn_gate_inp);


    // self managed
    _weight_desc.attn_q_norm.indp();
    _weight_desc.attn_k_norm.indp();



    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(8192);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(proj_weights_helper.get_size());
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(norm_exp_helper.get_size() / sizeof(bf16));
    if (this->has_qk_norm) {
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(RoPE_DIM + DH * 2);
    }
    else {
        this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(RoPE_DIM);
    }


    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();

}

void decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks, int block_for_512_reorder)
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


void decoding_layer::load_weights(Q4NX& q4nx,   int layer_idx) {
   
    constexpr int GLU_SLICE = 1024;
    constexpr int UP_GATE_OUTDIM_REORDEER = 512;
    constexpr int EXP_UP_GATE_OUTDIM_REORDER= 128;
    u8* w_ptr = this->all_proj_weight_buffer.data();
    u8* w_ptr_begin= w_ptr;
    u8* w_ptr_end = this->all_proj_weight_buffer.end();

    {
        LOG_VERBOSE(1, "Loading up/gate expert weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up_exps;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate_exps;
        u8* w_ptr = w_ptr_begin + desc_up.offset;
        buffer<u8> w_up_exps;
        buffer<u8> w_gate_exps;
        q4nx.load_weights(w_up_exps, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate_exps, desc_gate.format_name(layer_idx));

        size_t size_per_expert = _weight_desc.ffn_up_exps.get_size() / num_experts;
        assert(size_per_expert == _weight_desc.ffn_gate_exps.get_size() / num_experts);
        std::cout << "size_per_expert " << size_per_expert<< std::endl;
        int slices = _weight_desc.ffn_up_exps.shape[1] / EXP_UP_GATE_OUTDIM_REORDER;
        std::cout << "Slices size is " << slices << std::endl;
        size_t slice_size = size_per_expert / slices;
        std::cout <<"slice size for debug " << slice_size << std::endl;
        for(int e = 0; e < num_experts; e++) {
            buffer<u8> w_up_exp = buffer<u8>(w_up_exps.data() + e * size_per_expert, size_per_expert);
            buffer<u8> w_gate_exp = buffer<u8>(w_gate_exps.data() + e * size_per_expert, size_per_expert);
            for (int i = 0; i < slices; i++) {
                buffer<u8> up_slice = buffer<u8>(w_up_exp.data() + i * slice_size, slice_size);
                buffer<u8> gate_slice = buffer<u8>(w_gate_exp.data() + i * slice_size, slice_size);
                reorder_cpy(w_ptr, up_slice, D, _weight_desc.ffn_up_exps.dtype, 4, 2);
                w_ptr += slice_size;
                reorder_cpy(w_ptr, gate_slice, D, _weight_desc.ffn_gate_exps.dtype, 4, 2);
                w_ptr += slice_size;
            }
        }
        _weight_desc.ffn_up_exps.load();
        _weight_desc.ffn_gate_exps.load();
    }
    // read down expert weights
    {
        LOG_VERBOSE(1, "Loading down expert weights");
        weight_desc_t& desc = _weight_desc.ffn_down_exps;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w_down_exps;
        q4nx.load_weights(w_down_exps, _weight_desc.ffn_down_exps.format_name(layer_idx));

        size_t size_per_expert = desc.get_size() / num_experts;
        int slices = desc.shape[1] / EXP_UP_GATE_OUTDIM_REORDER;
        size_t slice_size = size_per_expert / slices;



        for (int e = 0; e < num_experts; e++) {
            buffer<u8> w_down_exp = buffer<u8>(w_down_exps.data() + e * size_per_expert, size_per_expert);
            for (int i = 0; i < slices; i++) {
                buffer<u8> down_slice = buffer<u8>(w_down_exp.data() + i * slice_size, slice_size);
                assert(intermediate_size == 512);
                reorder_cpy(w_ptr, down_slice, intermediate_size, desc.dtype, 4, 2);
                // reorder_cpy(w_ptr, down_slice, intermediate_size, desc.dtype, 4, 2);
                // reorder_cpy(w_ptr + , down_slice, intermediate_size, desc.dtype, 4, 2);


                w_ptr += slice_size ;
            }
        }
        _weight_desc.ffn_down_exps.load();
    }    
    {
        LOG_VERBOSE(1, "Loading qkv weights");
        weight_desc_t& desc_q = _weight_desc.attn_q;
        weight_desc_t& desc_k = _weight_desc.attn_k;
        weight_desc_t& desc_v = _weight_desc.attn_v;
        weight_desc_t& desc_gate = _weight_desc.attn_gate;
        u8* w_ptr = w_ptr_begin + desc_q.offset;
        buffer<u8> w_q;
        buffer<u8> w_k;
        buffer<u8> w_v;
        q4nx.load_weights(w_q, desc_q.format_name(layer_idx));
        q4nx.load_weights(w_k, desc_k.format_name(layer_idx));
        q4nx.load_weights(w_v, desc_v.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        size_t w_q_gate_size = w_q.size();
        buffer<u8> w_q_q(w_q.data(),w_q_gate_size / 2);
        buffer<u8> w_gate(w_q.data() + w_q_gate_size / 2, w_q_gate_size / 2);
        reorder_cpy(w_ptr, w_q_q, D, desc_q.dtype);
        w_ptr += desc_q.get_size();
        reorder_cpy(w_ptr, w_k, D, desc_k.dtype);
        w_ptr += desc_k.get_size();
        reorder_cpy(w_ptr, w_v, D, desc_v.dtype);
        w_ptr += desc_v.get_size();
        reorder_cpy(w_ptr, w_gate, D, desc_gate.dtype);
        desc_q.load();
        desc_k.load();
        desc_v.load();
        desc_gate.load();
    }
    // read o weights
    {
        LOG_VERBOSE(1, "Loading o weights");
        weight_desc_t& desc = _weight_desc.attn_output;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w_o;
        q4nx.load_weights(w_o, desc.format_name(layer_idx));
        reorder_cpy(w_ptr, w_o, DQ, desc.dtype);
        desc.load();
    }
    // read shared expert up weights
    {
        LOG_VERBOSE(1, "Loading shared up/gate weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up_shexp;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate_shexp;
        u8* w_ptr = w_ptr_begin + desc_up.offset;
        buffer<u8> w_up_shexp;
        buffer<u8> w_gate_shexp;
        q4nx.load_weights(w_up_shexp, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate_shexp, desc_gate.format_name(layer_idx));

        assert(_weight_desc.ffn_up_shexp.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);
        assert(_weight_desc.ffn_gate_shexp.shape[1] % UP_GATE_OUTDIM_REORDEER == 0);

        int slices = _weight_desc.ffn_up_shexp.shape[1] / UP_GATE_OUTDIM_REORDEER;

        size_t slice_size = _weight_desc.ffn_up_shexp.get_size() / slices;
        for (int i = 0; i < slices; i++) {
            buffer<u8> up_slice = buffer<u8>(w_up_shexp.data() + i * slice_size, slice_size);
            buffer<u8> gate_slice = buffer<u8>(w_gate_shexp.data() + i * slice_size, slice_size);
            reorder_cpy(w_ptr, up_slice, D, desc_up.dtype);
            w_ptr += slice_size;
            reorder_cpy(w_ptr, gate_slice, D, desc_gate.dtype);
            w_ptr += slice_size;
        }
        desc_up.load();
        desc_gate.load();
    }
    // read shared expert down weights
    {
        LOG_VERBOSE(1, "Loading shared down weights");
        weight_desc_t& desc = _weight_desc.ffn_down_shexp;
        u8* w_ptr = w_ptr_begin + desc.offset;
        buffer<u8> w_down_shexp;
        q4nx.load_weights(w_down_shexp, desc.format_name(layer_idx));
        reorder_cpy(w_ptr, w_down_shexp, intermediate_size, desc.dtype);
        desc.load();
    }
    // read up expert weights

    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;

        bf16* w_rms_ptr = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, _weight_desc.input_layer_norm.format_name(layer_idx));
        q4nx.load_weights(w_rms_post, _weight_desc.post_attntion_norm.format_name(layer_idx));
        header_print("info", "Input layer_norm:");
        utils::print_matrix(w_rms_pre, D);

        // w_rms_post.memset(1);
        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        _weight_desc.input_layer_norm.load();
        _weight_desc.post_attntion_norm.load();

        // load ffn_gate_inp, ffn_gate_inp_shexp
        buffer<bf16> w_gate_inp;
        buffer<bf16> w_gate_inp_shexp;
        q4nx.load_weights(w_gate_inp, _weight_desc.ffn_gate_inp.format_name(layer_idx));
        q4nx.load_weights(w_gate_inp_shexp, _weight_desc.ffn_gate_inp_shexp.format_name(layer_idx));

        memcpy((unsigned char*)w_rms_ptr, w_gate_inp_shexp.data(), _weight_desc.ffn_gate_inp_shexp.get_size());
        w_rms_ptr += _weight_desc.ffn_gate_inp_shexp.get_size() / sizeof(bf16);

        memcpy((unsigned char*)w_rms_ptr, w_gate_inp.data(), _weight_desc.ffn_gate_inp.get_size());
        w_rms_ptr += _weight_desc.ffn_gate_inp.get_size() / sizeof(bf16);

        _weight_desc.ffn_gate_inp.load();
        _weight_desc.ffn_gate_inp_shexp.load();

        assert(w_rms_ptr == this->rms_weights_buffer.end());
    }

    if (this->has_qk_norm)
    {
        LOG_VERBOSE(1, "Loading attention norm weights");
        buffer<bf16> q_attn_norm;
        buffer<bf16> k_attn_norm;
        q4nx.load_weights(q_attn_norm, _weight_desc.attn_q_norm.format_name(layer_idx));
        q4nx.load_weights(k_attn_norm, _weight_desc.attn_k_norm.format_name(layer_idx));
        memcpy(this->rope_rms_buffer.data() + RoPE_DIM, q_attn_norm.data(), DH * sizeof(bf16));
        memcpy(this->rope_rms_buffer.data() + RoPE_DIM + DH, k_attn_norm.data(), DH * sizeof(bf16));
        _weight_desc.attn_q_norm.load();
        _weight_desc.attn_k_norm.load();
    }

    this->_set_rope_weights(0);
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

buffer<bf16> decoding_layer::operator()(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    this->all_proj_weight_buffer.sync_to_device();
    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    this->kv_cache.sync_to_device();
    this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
    this->hidden_states_buffer.sync_from_device();
    this->kv_cache.sync_from_device();
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
    bf16* w_rope_ptr = this->rope_rms_buffer.data();
    if (rope_weights.size() == 0) {
        header_print("warning", "No RoPE weights, skipping setting rope weights");
        for (uint32_t i = 0; i < RoPE_DIM / 2; i++){
            w_rope_ptr[i] = bf16(1.0f);
        }
        return; // no RoPE, no need to set weights
    }
    for (uint32_t i = 0; i < RoPE_DIM / 2; i++){
        w_rope_ptr[i] = bf16(std::cos(this->rope_weights[i] * idx) * this->rope_scaling);
        w_rope_ptr[i + RoPE_DIM / 2] = bf16(std::sin(this->rope_weights[i] * idx) * this->rope_scaling);
    }
    this->rope_rms_buffer.sync_to_device();
}

void decoding_layer::set_rope_frequency(buffer<float>& rope_freqs){
    this->rope_weights = buffer<float>(rope_freqs.size());
    header_print("info", "RoPE frequencies set, size: " << rope_weights.size());
    memcpy(this->rope_weights.data(), rope_freqs.data(), rope_freqs.size() * sizeof(float));
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
    int gate_rtp_lock_id = 6;
    int AB_proj_rtp_lock_id = 14;    
    int router_shimtile_configure_ready_lock = 10;
    seq->clear_cmds();

    for (int i = 0; i < 2; i++) {
        seq->rtp_write(attn_qk_tiles[i], l_qk_address, L);
        seq->rtp_write(attn_kv_tiles[i], l_kv_address, L);
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], 17408, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }



    seq->rtp_write(rms_tile, rms_address, 1);
    seq->rtp_write(gate_tile, 49152, 1);
    seq->rtp_write(ab_proj_tile, 14592, 1);    
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * rms_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(gate_tile, CT_lock_address_base + 16 * gate_rtp_lock_id, 1); // set lock to 1
    seq->rtp_write(ab_proj_tile, CT_lock_address_base + 16 * AB_proj_rtp_lock_id, 1);
    _send_hidden_states(seq);
    _send_rms_weights(seq, 0);
    _send_rope_rms_weights(seq);

    // receive y
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        xyr_tile,
        bd_10,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D},
        {0, 0, 0, 1},
        -1, 0, true
    );

    // //receive tb
    // seq->npu_dma_memcpy_nd(
    //     sizeof(bf16),
    //     xy_arg_idx,
    //     S2MM,
    //     IT5,
    //     bd_10,
    //     it_channel_0,
    //     {0, 0, 0, D},
    //     {1, 1, 1, (uint32_t)DQ},
    //     {0, 0, 0, 1},
    //     -1, 0, true
    // );


    // //receive tb from router
    // seq->npu_dma_memcpy_nd(
    //     sizeof(bf16),
    //     xy_arg_idx,
    //     S2MM,
    //     IT5,
    //     bd_10,
    //     it_channel_0,
    //     {0, 0, 0, D},
    //     {1, 1, 1, (uint32_t)D},
    //     {0, 0, 0, 1},
    //     -1, 0, true
    // );


    _move_weights_q80(seq, DQ + DK + DV, D, _weight_desc.attn_q.offset);
    _move_weights_q80(seq, DQ, D, _weight_desc.attn_gate.offset);
    _receive_kv_cache(seq, L); // also move qkv weights
    _move_kv_cache(seq, L);
    _move_weights_q80(seq, D, DQ, _weight_desc.attn_output.offset);



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
    


    //TODO: consider send router weights and share_expert vector here

    // send share-expert up-gate weights
    _move_weights_q80(seq, 2 * intermediate_size, D, _weight_desc.ffn_up_shexp.offset);
    
    // send share-expert down weights
    _move_weights_q80(seq, D, intermediate_size, _weight_desc.ffn_down_shexp.offset);


   

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );
    // seq->npu_dma_wait(
    //     IT5,
    //     S2MM,
    //     it_channel_0
    // );

    seq->cmds2seq();
}



void decoding_layer::_send_hidden_states(npu_sequence* seq){
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

void decoding_layer::_send_rms_weights(npu_sequence* seq, size_t byte_offset){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, byte_offset/sizeof(bf16)},
        {1, 1, 1, (uint32_t)D*3 },
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
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rope_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_2,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(this->has_qk_norm? (RoPE_DIM + DH * 2): RoPE_DIM)},
        {0, 0, 0, 1},
        1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        xyr_tile,
        MM2S,
        it_channel_0
    );
    // seq->npu_dma_memcpy_nd(
    //     sizeof(bf16),
    //     w_rms_arg_idx,
    //     MM2S,
    //     xyr_tile,
    //     bd_3,
    //     it_channel_0,
    //     {0, 0, 0, (uint32_t)D * 2},
    //     {1, 1, 1, (uint32_t)DQ + DK + DV},
    //     {0, 0, 0, 1},
    //     1, 0, false, no_cache
    // );
}

void decoding_layer::_receive_kv_cache(npu_sequence* seq, const int L){
    uint32_t data_per_chunk = this->num_kv_heads * this->DH;
    uint32_t v_offset = this->kv_cache_size / 2;
    uint32_t L_offset = (L - 1) * data_per_chunk;

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        attn_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, (uint32_t)L_offset},
        {1, 1, 1, data_per_chunk},
        {0, 0, 0, 1},
        -1, 0, true
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        S2MM,
        attn_tile,
        bd_1,
        it_channel_0,
        {0, 0, 0, (uint32_t)(L_offset + v_offset)},
        {1, 1, 1, data_per_chunk},
        {0, 0, 0, 1},
        -1, 0, true
    );
}

void decoding_layer::_send_router_w_and_share_exp_gate(npu_sequence* seq, const size_t NUM_expert, const size_t D, const size_t W_offset_in_byte){
    
    //TODO: have this aligned with the MT parametesr
    // router weights is NUM_expert x D in row major

    // but transfer router_weight in block sof router_weight_row_block x router_wieght_col_block
    // each block is column major order
    // but the blocks are row-block major order

    //TODO: need to share same file with decode for sanity 
    size_t router_weight_row_block = 32;
    size_t router_weight_col_block = 256;


    // assume the shared_expert_gate is right next to the router_weight
    //assert(_weight_desc.ffn_gate_inp_shexp.offset == _weight_desc.ffn_gate_inp.offset+ sizeof(float) * D* NUM_expert);


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

    //Chunk the D dimension into 512-element slices to stay within BD transfer size limits.
    //Each call sends 512 cols x router_weight_row_block rows, preserving column-major-within-block order.
    //  for(int i = 0; i < (int)(NUM_expert/router_weight_row_block); i++){
        // for(int c = 0; c < (int)(D/512); c++){
            // seq->npu_dma_memcpy_nd(
                // sizeof(float),
                // w_rms_arg_idx,
                // MM2S,
                // IT5,
                // npu_bd_id(bd_12),
                // it_channel_0,
                // {0, 0, 0, (uint32_t)(W_offset_in_fp32 + i*(D*router_weight_row_block) + c*512)},
                // {1,   512, router_weight_row_block, 1},
                // {0,     1, D,                       1},
                // 10, 0, true, aggressive_cache
            // );
            // seq->npu_dma_wait(
                // IT5,
                // MM2S,
                // it_channel_0
            // );
        // }
    // }

    // int bd_queue = 0;
    // int is_bd_ping = true;
    // for(int i = 0; i < (int)(NUM_expert/router_weight_row_block); i++){

    //     if(bd_queue==2){
    //         seq->npu_dma_wait(
    //             IT5,
    //             MM2S,
    //             it_channel_0
    //         );
    //         bd_queue--;
    //     }


    //     seq->npu_dma_memcpy_nd(
    //         sizeof(float),
    //         w_rms_arg_idx,
    //         MM2S,
    //         IT5,
    //         is_bd_ping? npu_bd_id(bd_12): npu_bd_id(bd_13),
    //         it_channel_0,
    //         {0, 0, 0, (uint32_t)(W_offset_in_fp32 + i*(D*router_weight_row_block) )},
    //         {D/512,   512, router_weight_row_block, 1},
    //         {512,     1, D,                       1},
    //         10, 0, true, aggressive_cache
    //     );
    //     is_bd_ping = !is_bd_ping;
    //     bd_queue++;
    
    // }    

    // while(bd_queue> 0){
    //     seq->npu_dma_wait(
    //         IT5,
    //         MM2S,
    //         it_channel_0
    //     );
    //     bd_queue--;
    // }
 
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


void decoding_layer::send_manual_expert_down_gate_q4k(

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



    // seq->npu_dma_memcpy_nd(
    //     sizeof(uint8_t),
    //     w_proj_arg_idx,
    //     MM2S,
    //     mvm_tiles[col],
    //     npu_bd_id(bd_7),
    //     it_channel_1,
    //     {0,0,0,cur_expert_w_offset_in_byte + 2*a_block_size},
    //     {down_col_number,                        INT_Q4nx_BLOCK_PER_ROW,    (2*a_block_size)/256 ,256},
    //     {4*a_block_size*INT_Q4nx_BLOCK_PER_ROW,       4*a_block_size,             256,1},
    //     -1,0,true
    // );// TODO: change l

}



void decoding_layer::setup_expert_down_gate_q4k(

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


void decoding_layer::setup_expert_up_gate_q4k(
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

void decoding_layer::send_manual_expert_up_gate_q4k(
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

void decoding_layer::_move_weights_q4k(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems*37 / 64 / sizeof(bf16);
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

void decoding_layer::_move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
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
void decoding_layer::_move_kv_cache(npu_sequence* seq, const size_t L){
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t data2move = L_padded * DH * this->num_kv_heads;
    const uint32_t v_offset = this->kv_cache_size / 2;


    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        attn_tile,
        bd_1,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    seq->npu_dma_wait(
        attn_tile,
        S2MM,
        it_channel_0
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_idx,
        MM2S,
        attn_tile,
        bd_2,
        it_channel_1,
        {0, 0, 0, (uint32_t)(v_offset)},
        {1, 1, 1, data2move},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
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


// void decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks)
// {
//     assert(is_quantize(dtype));
//     const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);
//     const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
//     const int slice_size = 512;
//     const int slices = a_block_size / slice_size;
//     const int rows = src.size() / a_block_size / blocks_per_row;

//     u8 *dst_ptr = dst;
//     std::vector<u8 *> src_ptr(vertical_blocks);
//     for (int i = 0; i < vertical_blocks; i++)
//     {
//         src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
//     }
//     for (int r = 0; r < rows; r += vertical_blocks)
//     {
//         for (int c = 0; c < blocks_per_row; c++)
//         {
//             for (int s = 0; s < slices; s++) {
//                 for (int i = 0; i < vertical_blocks; i++)
//                 {
//                     memcpy(dst_ptr, src_ptr[i], slice_size);
//                     dst_ptr += slice_size;
//                     src_ptr[i] += slice_size;
//                 }
//             }
//         }
//         for (int i = 0; i < vertical_blocks; i++)
//         {
//             src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
//             if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
//             {
//                 src_ptr[i] = src.data(); // useless padding
//             }
//         }
//     }
// }
