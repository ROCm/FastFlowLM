// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "decoding_layer.hpp"

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int context_length)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), MAX_L(context_length) {
    LOG_VERBOSE(1, "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();

    this->current_context_length = 0;

    num_hidden_layers = lm_config.num_hidden_layers;
    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    num_kv_heads = lm_config.num_key_value_heads;
    D = lm_config.hidden_size;
    // the dense MLP width must match the kernels (INTERMEDIATE_SIZE in the model header)
    intermediate_size = model_list::get_intermediate_size(model_type);
    int config_intermediate_size = 0;
    JSON_GET(config_intermediate_size, lm_config._json_config, "intermediate_size", 0, int);
    assert(config_intermediate_size == intermediate_size);

    assert(num_kv_heads == attn_cus); // one KV head per attention CU
    // _move_kv_cache reads whole 16-token chunks, so MAX_L must be whole chunks
    assert(MAX_L % L_CHUNK == 0);
    kv_heads_per_col = num_kv_heads / attn_cols;
    kv_col_row = (size_t)kv_heads_per_col * DH;
    kv_col_half = (size_t)MAX_L * kv_col_row;
    kv_cache_size = (size_t)MAX_L * (DK + DV);

    RoPE_DIM = int(DH * model_list::get_rope_partial_factor(model_type));

    _read_rtp_addresses(lm_config.layer_xclbin_name);

    // Debug: HY_STOP_AFTER=<1..5> ends the run sequence early, to find where a
    // hang is (see _gen_sequence). Unset or 0 runs the whole layer.
    const char* stop_env = std::getenv("HY_STOP_AFTER");
    debug_stop_after = stop_env ? std::atoi(stop_env) : 0;
    if (debug_stop_after != 0) {
        header_print("warning", "HY_STOP_AFTER=" << debug_stop_after << ": the run sequence ends early (debug)");
    }

    // the config, only with VERBOSE >= 1 (make host VERBOSE=1)
    LOG_VERBOSE(1, "Decoding layer config: D " << D << ", DQ " << DQ << ", DK " << DK << ", DV " << DV
                   << ", RoPE dim " << RoPE_DIM << ", MLP width " << intermediate_size);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    LOG_VERBOSE(1, "  QK norm: " << (has_qk_norm ? "yes" : "no"));
    this->rope_scaling = model_list::get_rope_scaling(this->model_type);

    _weight_desc.attn_k = weight_desc_t(flm_q4k, {D, DK}, "model.layer.%d.self_attn.k_proj.weight");
    _weight_desc.attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.k_norm.weight");
    _weight_desc.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.input_layernorm.weight");
    _weight_desc.attn_q = weight_desc_t(flm_q4k, {D, DQ}, "model.layer.%d.self_attn.q_proj.weight");
    _weight_desc.attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.q_norm.weight");
    _weight_desc.attn_v = weight_desc_t(flm_q4k, {D, DV}, "model.layer.%d.self_attn.v_proj.weight");
    _weight_desc.ffn_up = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layer.%d.mlp.up_proj.weight");
    _weight_desc.ffn_gate = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layer.%d.mlp.gate_proj.weight");
    _weight_desc.ffn_down = weight_desc_t(flm_q4k, {intermediate_size, D}, "model.layer.%d.mlp.down_proj.weight");
    _weight_desc.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.post_attention_layernorm.weight");
    _weight_desc.attn_output = weight_desc_t(flm_q4k, {DQ, D}, "model.layer.%d.self_attn.o_proj.weight");

    // k, v, q back to back: the proj engine runs them as one matrix and RoPE
    // takes its K/V block (kv_in) first, then Q (q_in)
    proj_weights_helper.add_weight(_weight_desc.attn_k);
    proj_weights_helper.add_weight(_weight_desc.attn_v);
    proj_weights_helper.add_weight(_weight_desc.attn_q);
    proj_weights_helper.add_weight(_weight_desc.attn_output);
    // up and gate are one region, interleaved per round (see _load_mlp_weights)
    proj_weights_helper.add_weight(_weight_desc.ffn_up);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate);
    proj_weights_helper.add_weight(_weight_desc.ffn_down);


    assert(proj_weights_helper.get_size() < 512ull * 1024 * 1024);  // check to avoid 512 M problem

    // [input_layernorm | post_attention_layernorm]
    norm_helper.add_weight(_weight_desc.input_layer_norm);
    norm_helper.add_weight(_weight_desc.post_attntion_norm);


    // self managed
    _weight_desc.attn_q_norm.indp();
    _weight_desc.attn_k_norm.indp();



    // [0, D): layer input, then layer output
    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(D);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(proj_weights_helper.get_size());
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(norm_helper.get_size() / sizeof(bf16));
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

// The xclbin build writes the address of every RTP buffer to
// rtp_addresses.txt next to layer.xclbin, one "RTP_<name> <address>" per line.
// The attention tiles' context length lives in RTP_L_attn_{qk,kv}_core_<row>_<col>,
// the proj / rms IS_ATTN flag in RTP_PROJ_IS_ATTN_BUFFER_<row>_<col> / RTP_RMS_IS_ATTN_BUFFER.
void decoding_layer::_read_rtp_addresses(const std::string& xclbin_path){
    std::filesystem::path rtp_path = std::filesystem::path(xclbin_path).parent_path() / "rtp_addresses.txt";
    std::ifstream file(rtp_path);
    if (!file.is_open()) {
        header_print("error", "Cannot open " << rtp_path.string());
        exit(1);
    }
    std::map<std::string, uint32_t> addr;
    std::string name;
    uint32_t value;
    while (file >> name >> value) {
        addr[name] = value;
    }
    auto lookup = [&](const std::string& key) {
        if (addr.count(key) == 0) {
            header_print("error", key << " not in " << rtp_path.string());
            exit(1);
        }
        return addr[key];
    };
    auto rc_name = [](const int rc[2]) {
        return std::to_string(rc[0]) + "_" + std::to_string(rc[1]);
    };
    for (int i = 0; i < attn_cus; i++) {
        l_qk_address[i] = lookup("RTP_L_attn_qk_core_" + rc_name(attn_qk_rc[i]));
        l_kv_address[i] = lookup("RTP_L_attn_kv_core_" + rc_name(attn_kv_rc[i]));
    }
    for (int i = 0; i < 16; i++) {
        proj_is_attn_address[i] = lookup("RTP_PROJ_IS_ATTN_BUFFER_" + rc_name(proj_rc[i]));
    }
    rms_is_attn_address = lookup("RTP_RMS_IS_ATTN_BUFFER");
}

/// Lays a quantized weight out in the order the proj engine reads it: rows of
/// blocks are taken vertical_blocks at a time and interleaved block by block,
/// so one shim BD streams the blocks of vertical_blocks proj cores together.
/// The blocks themselves stay in the q4nx byte order: this overlay's Q4_K
/// kernel (kernels/q4_core.h) reads q4k_block_t as stored, unlike
/// moe_decoding's int8-mmul kernel, so there is no q4k_relayout_for_mmul here.
void decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, flm_dtype_t dtype, const int vertical_blocks)
{
    assert(is_quantize(dtype));

    const int a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, dtype);

    const int blocks_per_row = col / QXNX_COL_BLOCK_SIZE;
    const int rows = src.size() / a_block_size / blocks_per_row;

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
}


void decoding_layer::load_weights(Q4NX& q4nx,   int layer_idx) {
    // this overlay only runs the dense layers (layer 0)
    assert(layer_idx >= 0 && layer_idx < num_hidden_layers);
    assert(model_list::is_dense_layer(model_type, layer_idx));
    u8* w_ptr_begin = this->all_proj_weight_buffer.data();

    _load_mlp_weights(q4nx, layer_idx);

    {
        LOG_VERBOSE(1, "Loading qkv weights");
        weight_desc_t& desc_q = _weight_desc.attn_q;
        weight_desc_t& desc_k = _weight_desc.attn_k;
        weight_desc_t& desc_v = _weight_desc.attn_v;
        u8* w_ptr = w_ptr_begin + desc_k.offset;
        buffer<u8> w_q;
        buffer<u8> w_k;
        buffer<u8> w_v;
        q4nx.load_weights(w_q, desc_q.format_name(layer_idx));
        q4nx.load_weights(w_k, desc_k.format_name(layer_idx));
        q4nx.load_weights(w_v, desc_v.format_name(layer_idx));
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        // k, v, q back to back (see the constructor)
        assert(desc_v.offset == desc_k.offset + desc_k.get_size());
        assert(desc_q.offset == desc_v.offset + desc_v.get_size());
        reorder_cpy(w_ptr, w_k, D, desc_k.dtype);
        w_ptr += desc_k.get_size();
        reorder_cpy(w_ptr, w_v, D, desc_v.dtype);
        w_ptr += desc_v.get_size();
        reorder_cpy(w_ptr, w_q, D, desc_q.dtype);
        desc_q.load();
        desc_k.load();
        desc_v.load();
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
    // read rms weights
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;

        bf16* w_rms_ptr = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, _weight_desc.input_layer_norm.format_name(layer_idx));
        q4nx.load_weights(w_rms_post, _weight_desc.post_attntion_norm.format_name(layer_idx));

        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        _weight_desc.input_layer_norm.load();
        _weight_desc.post_attntion_norm.load();

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
        LOG_VERBOSE(1, "Weights ready");
    } else {
        header_print("error", "Some weights were not ready!");
    }
}

/// Dense MLP weights, in the order the proj engine and the GLU read them (see
/// models/hy-memory-30b-a3b-dense.h). A Q4_K tensor in model.q4nx is row
/// blocks of 32 rows, each row block's 256-column blocks back to back, so
/// "rows a..b" of a tensor is one contiguous byte range.
void decoding_layer::_load_mlp_weights(Q4NX& q4nx, int layer_idx) {
    // GLU_SLICE = 512: one proj round (16 cores x 32 rows) is [up 256 | gate 256]
    const int rows_per_round = model_list::get_up_gate_rows_per_round(model_type);
    const size_t a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
    u8* w_ptr_begin = this->all_proj_weight_buffer.data();

    // bytes of `rows` rows of a matrix with `in_dim` columns
    auto row_bytes = [&](size_t rows, size_t in_dim) {
        return rows / QXNX_ROW_BLOCK_SIZE * (in_dim / QXNX_COL_BLOCK_SIZE) * a_block_size;
    };
    // [up rows 0..n | gate rows 0..n | up rows n..2n | gate rows n..2n | ...], n = rows_per_piece
    auto interleave_up_gate = [&](buffer<u8>& out, const u8* up, const u8* gate, size_t rows, size_t rows_per_piece) {
        const size_t piece = row_bytes(rows_per_piece, D);
        u8* dst = out.data();
        for (size_t r = 0; r < rows; r += rows_per_piece) {
            memcpy(dst, up, piece);    dst += piece; up += piece;
            memcpy(dst, gate, piece);  dst += piece; gate += piece;
        }
    };

    // up/gate: 2 * 6912 / 512 = 27 rounds of [up 256 | gate 256]
    {
        LOG_VERBOSE(1, "Loading up/gate weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate;
        assert(desc_gate.offset == desc_up.offset + desc_up.get_size());
        assert(intermediate_size % rows_per_round == 0);
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate, desc_gate.format_name(layer_idx));
        assert(w_up.size() == desc_up.get_size() && w_gate.size() == desc_gate.get_size());

        buffer<u8> both(desc_up.get_size() + desc_gate.get_size());
        interleave_up_gate(both, w_up.data(), w_gate.data(), intermediate_size, rows_per_round);
        reorder_cpy(w_ptr_begin + desc_up.offset, both, D, flm_q4k);
        desc_up.load();
        desc_gate.load();
    }
    // down: plain row order
    {
        LOG_VERBOSE(1, "Loading down weights");
        weight_desc_t& desc = _weight_desc.ffn_down;
        buffer<u8> w_down;
        q4nx.load_weights(w_down, desc.format_name(layer_idx));
        assert(w_down.size() == desc.get_size());
        reorder_cpy(w_ptr_begin + desc.offset, w_down, intermediate_size, flm_q4k);
        // DEBUG (HY_ZERO_MLP=1): all-zero Q4_K blocks (S' = M' = 0) make the MLP
        // output 0, so the layer output is the residual (x + o_proj): isolates
        // attention + o_proj from the MLP
        if (std::getenv("HY_ZERO_MLP")) {
            memset(w_ptr_begin + desc.offset, 0, desc.get_size());
        }
        desc.load();
    }
}

buffer<bf16> decoding_layer::operator()(buffer<bf16>& input) {
    // the KV cache is full: the new token has no row to go to
    assert(this->current_context_length < MAX_L);
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    this->all_proj_weight_buffer.sync_to_device();
    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    this->kv_cache.sync_to_device();
    // the call's end state: anything but completed (e.g. a timeout) means the
    // output is not this call's result (the host test counts these)
    ert_cmd_state state = this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
    this->last_call_completed = (state == ERT_CMD_STATE_COMPLETED);
    this->hidden_states_buffer.sync_from_device();
    this->kv_cache.sync_from_device();
    this->set_context_length(this->current_context_length + 1);
    return this->hidden_states_buffer;
}


/// NPU time per layer call in ms: `calls` runs back to back in one runlist, no
/// host syncs between them, at context length ctx (overwrites the KV state; the
/// KV values do not matter for timing).
float decoding_layer::bench(buffer<bf16>& input, int calls, int ctx) {
    this->set_context_length(ctx);
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < calls; i++){
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache));
    }
    time_utils::time_point start = time_utils::now();
    rl.execute();
    rl.wait();
    this->hidden_states_buffer.sync_from_device();
    time_utils::time_point end = time_utils::now();
    time_utils::time_with_unit total_ms = time_utils::duration_ms(start, end);
    return total_ms.first / calls;
}


void decoding_layer::_set_rope_weights(int idx){
    bf16* w_rope_ptr = this->rope_rms_buffer.data();
    if (rope_weights.size() == 0) {
        // normal at load time: the frequencies come later (set_rope_frequency)
        LOG_VERBOSE(1, "No RoPE frequencies yet, RoPE table left at 1");
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
    LOG_VERBOSE(1, "RoPE frequencies set, size: " << rope_weights.size());
    memcpy(this->rope_weights.data(), rope_freqs.data(), rope_freqs.size() * sizeof(float));
}

void decoding_layer::set_context_length(int L){

    this->current_context_length = L;

    // A full cache (L == MAX_L tokens) has no next call to prepare: that call
    // would write row L, past the cache. operator() refuses to run then.
    if (L >= MAX_L) return;

    this->_gen_sequence(this->layer_app.seq(), L + 1);

    this->_set_rope_weights(L);
}

void decoding_layer::_gen_sequence(npu_sequence* seq, int L) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int is_attn_rtp_lock_id = 6;   // proj and rms wait on it for IS_ATTN
    assert(L > 0 && L <= MAX_L);
    seq->clear_cmds();

    for (int i = 0; i < attn_cus; i++) {
        seq->rtp_write(get_tile(attn_qk_rc[i][0], attn_qk_rc[i][1]), l_qk_address[i], L);
        seq->rtp_write(get_tile(attn_kv_rc[i][0], attn_kv_rc[i][1]), l_kv_address[i], L);
    }
    // IS_ATTN = 1: decoder layer (0 would be the LM head, not used: lm_head is Q8_0)
    for (int i = 0; i < 16; i++){
        npu_tiles tile = get_tile(proj_rc[i][0], proj_rc[i][1]);
        seq->rtp_write(tile, proj_is_attn_address[i], 1);
        seq->rtp_write(tile, CT_lock_address_base + 16 * is_attn_rtp_lock_id, 1); // set lock to 1
    }
    npu_tiles rms_tile = get_tile(rms_rc[0], rms_rc[1]);
    seq->rtp_write(rms_tile, rms_is_attn_address, 1);
    seq->rtp_write(rms_tile, CT_lock_address_base + 16 * is_attn_rtp_lock_id, 1); // set lock to 1

    _send_hidden_states(seq);
    _send_rms_weights(seq);
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

    // attention
    // the new K/V row's receive BDs go first, as in the production sequence
    _receive_kv_cache(seq, L);
    _move_weights_q4k(seq, DK + DV + DQ, D, _weight_desc.attn_k.offset);  // k, v, q
    if (debug_stop_after == 1) { seq->cmds2seq(); return; }
    _move_kv_cache(seq, L);
    if (debug_stop_after == 3) { seq->cmds2seq(); return; }
    _move_weights_q4k(seq, D, DQ, _weight_desc.attn_output.offset);
    if (debug_stop_after == 4) { seq->cmds2seq(); return; }

    // MLP: up/gate (27 rounds into the GLU), then down
    _move_weights_q4k(seq, 2 * intermediate_size, D, _weight_desc.ffn_up.offset);
    _move_weights_q4k(seq, D, intermediate_size, _weight_desc.ffn_down.offset);
    if (debug_stop_after == 5) { seq->cmds2seq(); return; }

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}

void decoding_layer::_send_hidden_states(npu_sequence* seq){
    // send x (packet 0: to the rms tile)
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

void decoding_layer::_send_rms_weights(npu_sequence* seq){
    // input_layernorm then post_attention_layernorm. Not waited on: the rms
    // tile holds one D-long weight at a time (iron/rms_residual.py), so the
    // second half only lands after attention, long after the weights below.
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void decoding_layer::_send_rope_rms_weights(npu_sequence* seq){
    // cos | sin | q_norm | k_norm (packet 1: to the RoPE tile, which holds all of it)
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
}

void decoding_layer::_receive_kv_cache(npu_sequence* seq, const int L){
    // RoPE sends the new token's K (all KV heads) through shim 3 and its V
    // through shim 4, each as two halves (heads 0, 1 then heads 2, 3). Write
    // each half into its column's K / V block, row L - 1: BD 0 = col 0, BD 1 = col 1.
    const uint32_t row_offset = (uint32_t)((L - 1) * kv_col_row);
    for (int kv = 0; kv < 2; kv++) {               // 0: K, 1: V
        for (int col = 0; col < attn_cols; col++) {
            uint32_t block_offset = (uint32_t)(kv_block(col, kv) * kv_col_half);
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                S2MM,
                kv_write_shim[kv],
                npu_bd_id(col),
                it_channel_0,
                {0, 0, 0, block_offset + row_offset},
                {1, 1, 1, (uint32_t)kv_col_row},
                {0, 0, 0, 1},
                -1, 0, true
            );
        }
    }
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
            _wait_proj_weights(seq);
        }
    }
    _wait_proj_weights(seq);
}

/// Wait for one weight task on both channels of each proj shim (0, 1, 6, 7).
void decoding_layer::_wait_proj_weights(npu_sequence* seq){
    for (uint32_t col = 0; col < 4; col++){
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}

void decoding_layer::_move_kv_cache(npu_sequence* seq, const size_t L){
    // Each column's shim reads its own K block (channel 0) and V block
    // (channel 1). K reads start once both K writes of the new row are done
    // (shim 3), V reads once both V writes are (shim 4).
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    assert(L_padded <= MAX_L);
    const uint32_t data2move = (uint32_t)(L_padded * kv_col_row);
    const npu_bd_id read_bd[2] = {bd_4, bd_5};    // shims 3, 4 use BDs 0, 1 to write

    for (int kv = 0; kv < 2; kv++) {               // 0: K, 1: V
        for (int col = 0; col < attn_cols; col++) {
            seq->npu_dma_wait(kv_write_shim[kv], S2MM, it_channel_0);
        }
        for (int col = 0; col < attn_cols; col++) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                attn_shims[col],
                read_bd[kv],
                kv == 0 ? it_channel_0 : it_channel_1,
                {0, 0, 0, (uint32_t)(kv_block(col, kv) * kv_col_half)},
                {1, 1, 1, data2move},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
        }
    }
}

// K (or V) of token idx, all KV heads in head order
static buffer<bf16> gather_kv_row(buffer<bf16>& kv_cache, int kv, int idx, int cols, size_t col_row, size_t col_half) {
    buffer<bf16> row(cols * col_row);
    for (int col = 0; col < cols; col++) {
        const bf16* src = kv_cache.data() + decoding_layer::kv_block(col, kv) * col_half + idx * col_row;
        memcpy(row.data() + col * col_row, src, col_row * sizeof(bf16));
    }
    return row;
}

void decoding_layer::fill_kv(const bf16* k_rows, const bf16* v_rows, int src_rows, int n_rows){
    assert(n_rows <= MAX_L);
    const size_t row = (size_t)num_kv_heads * DH;   // one source row: all KV heads
    for (int j = 0; j < n_rows; j++) {
        const size_t src = (size_t)(j % src_rows) * row;
        // a column holds its own KV heads: values [col * kv_col_row, (col + 1) * kv_col_row) of a row
        for (int col = 0; col < attn_cols; col++) {
            bf16* k_dst = kv_cache.data() + kv_block(col, 0) * kv_col_half + (size_t)j * kv_col_row;
            bf16* v_dst = kv_cache.data() + kv_block(col, 1) * kv_col_half + (size_t)j * kv_col_row;
            memcpy(k_dst, k_rows + src + col * kv_col_row, kv_col_row * sizeof(bf16));
            memcpy(v_dst, v_rows + src + col * kv_col_row, kv_col_row * sizeof(bf16));
        }
    }
    kv_cache.sync_to_device();
}

buffer<bf16> decoding_layer::get_k_cache(int idx){
    this->kv_cache.sync_from_device();
    return gather_kv_row(this->kv_cache, 0, idx, attn_cols, kv_col_row, kv_col_half);
}

buffer<bf16> decoding_layer::get_v_cache(int idx){
    this->kv_cache.sync_from_device();
    return gather_kv_row(this->kv_cache, 1, idx, attn_cols, kv_col_row, kv_col_half);
}

void decoding_layer::report_debug_tap(){
    if (debug_stop_after != 0) {
        // did RoPE's new K/V row reach the cache? (proves QKV was computed)
        buffer<bf16> k = get_k_cache(current_context_length - 1);
        buffer<bf16> v = get_v_cache(current_context_length - 1);
        size_t k_nz = 0, v_nz = 0;
        for (size_t i = 0; i < k.size(); i++) k_nz += (float)k[i] != 0.0f;
        for (size_t i = 0; i < v.size(); i++) v_nz += (float)v[i] != 0.0f;
        header_print("debug", "new K/V row: " << k_nz << " / " << k.size() << " K values and "
            << v_nz << " / " << v.size() << " V values non-zero");
    }
}
