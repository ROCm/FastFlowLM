// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "decoding_layer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int context_length, bool attn_tap)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), MAX_L(context_length), attn_tap(attn_tap) {
    LOG_VERBOSE(1, "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();

    this->current_context_length = 0;

    num_hidden_layers = lm_config.num_hidden_layers;
    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    num_kv_heads = lm_config.num_key_value_heads;
    // config.json's intermediate_size is the dense layer's; the MoE width must
    // match the kernels (see model_list::get_moe_intermediate_size)
    intermediate_size = model_list::get_moe_intermediate_size(model_type);
    D = lm_config.hidden_size;

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
    const char* tap_env = std::getenv("HY_DEBUG_TAP");
    debug_tap = tap_env ? tap_env : "";
    if (debug_stop_after != 0) {
        header_print("warning", "HY_STOP_AFTER=" << debug_stop_after << ": the run sequence ends early (debug)");
    }

    JSON_GET(this->num_experts, lm_config._json_config, "num_experts", 0, int);

    JSON_GET(this->num_activate_experts, lm_config._json_config, "num_experts_per_tok", 0, int);
    // the config, only with VERBOSE >= 1 (make host VERBOSE=1)
    LOG_VERBOSE(1, "Decoding layer config: D " << D << ", DQ " << DQ << ", DK " << DK << ", DV " << DV
                   << ", RoPE dim " << RoPE_DIM << ", expert width " << intermediate_size
                   << ", experts " << num_activate_experts << "/" << num_experts);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    LOG_VERBOSE(1, "  QK norm: " << (has_qk_norm ? "yes" : "no"));
    LOG_VERBOSE(1, "  attention tap: " << (attn_tap ? "yes" : "no"));
    this->rope_scaling = model_list::get_rope_scaling(this->model_type);

    _weight_desc.attn_k = weight_desc_t(flm_q4k, {D, DK}, "model.layer.%d.self_attn.k_proj.weight");
    _weight_desc.attn_k_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.k_norm.weight");
    _weight_desc.input_layer_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.input_layernorm.weight");
    _weight_desc.attn_q = weight_desc_t(flm_q4k, {D, DQ}, "model.layer.%d.self_attn.q_proj.weight");
    _weight_desc.attn_q_norm = weight_desc_t(flm_bf16, {DH}, "model.layer.%d.self_attn.q_norm.weight");
    _weight_desc.attn_v = weight_desc_t(flm_q4k, {D, DV}, "model.layer.%d.self_attn.v_proj.weight");
    _weight_desc.ffn_down_exps = weight_desc_t(flm_q4k, {intermediate_size, D, num_experts}, "model.layer.%d.mlp.down_exps_proj.weight");
    _weight_desc.ffn_down_shexp = weight_desc_t(flm_q4k, {intermediate_size, D}, "model.layer.%d.mlp.share_down_exps_proj.weight");
    _weight_desc.ffn_up_exps = weight_desc_t(flm_q4k, {D, intermediate_size, num_experts}, "model.layer.%d.mlp.up_exps_proj.weight");
    _weight_desc.ffn_up_shexp = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layer.%d.mlp.share_up_exps_proj.weight");
    _weight_desc.ffn_gate_exps = weight_desc_t(flm_q4k, {D, intermediate_size, num_experts}, "model.layer.%d.mlp.gate_exps_proj.weight");
    _weight_desc.ffn_gate_shexp = weight_desc_t(flm_q4k, {D, intermediate_size}, "model.layer.%d.mlp.share_gate_exps_proj.weight");
    _weight_desc.ffn_gate_inp = weight_desc_t(flm_bf16, {D, num_experts}, "model.layer.%d.moe_router.weight");
    _weight_desc.post_attntion_norm = weight_desc_t(flm_bf16, {D}, "model.layer.%d.post_attention_layernorm.weight");
    _weight_desc.attn_output = weight_desc_t(flm_q4k, {DQ, D}, "model.layer.%d.self_attn.o_proj.weight");

    proj_weights_helper.add_weight(_weight_desc.ffn_up_exps);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate_exps);
    proj_weights_helper.add_weight(_weight_desc.ffn_down_exps);
    // k, v, q back to back: the proj engine runs them as one matrix and RoPE
    // takes its K/V block (kv_in) first, then Q (q_in)
    proj_weights_helper.add_weight(_weight_desc.attn_k);
    proj_weights_helper.add_weight(_weight_desc.attn_v);
    proj_weights_helper.add_weight(_weight_desc.attn_q);
    proj_weights_helper.add_weight(_weight_desc.attn_output);
    proj_weights_helper.add_weight(_weight_desc.ffn_up_shexp);
    proj_weights_helper.add_weight(_weight_desc.ffn_gate_shexp);
    proj_weights_helper.add_weight(_weight_desc.ffn_down_shexp);


    assert(proj_weights_helper.get_size() < 512ull * 1024 * 1024);  // check to avoid 512 M problem

    // [input_layernorm | post_attention_layernorm | router weight]
    norm_exp_helper.add_weight(_weight_desc.input_layer_norm);
    norm_exp_helper.add_weight(_weight_desc.post_attntion_norm);
    norm_exp_helper.add_weight(_weight_desc.ffn_gate_inp);


    // self managed
    _weight_desc.attn_q_norm.indp();
    _weight_desc.attn_k_norm.indp();



    // [0, D): layer input, then layer output; [D, D + DQ): attention tap
    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(8192);
    assert(D + DQ <= 8192);
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

// The xclbin build writes the address of every RTP buffer to
// rtp_addresses.txt next to layer.xclbin, one "RTP_<name> <address>" per line.
// The attention tiles' context length lives in RTP_L_attn_{qk,kv}_core_<row>_<col>.
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
    auto lookup = [&](const char* kind, const int rc[2]) {
        std::string key = std::string("RTP_L_attn_") + kind + "_core_" + std::to_string(rc[0]) + "_" + std::to_string(rc[1]);
        if (addr.count(key) == 0) {
            header_print("error", key << " not in " << rtp_path.string());
            exit(1);
        }
        return addr[key];
    };
    for (int i = 0; i < attn_cus; i++) {
        l_qk_address[i] = lookup("qk", attn_qk_rc[i]);
        l_kv_address[i] = lookup("kv", attn_kv_rc[i]);
    }
    if (addr.count("RTP_router_expert_bias") == 0) {
        header_print("error", "RTP_router_expert_bias not in " << rtp_path.string());
        exit(1);
    }
    router_bias_address = addr["RTP_router_expert_bias"];
    if (addr.count("RTP_router_layer") == 0) {
        header_print("error", "RTP_router_layer not in " << rtp_path.string());
        exit(1);
    }
    router_layer_address = addr["RTP_router_layer"];
}

/// Lays a quantized weight out in the order the proj engine reads it: rows of
/// blocks are taken vertical_blocks at a time and interleaved block by block,
/// so one shim BD streams the blocks of vertical_blocks proj cores together.
/// Q4_K blocks (32 rows x 256 columns, 4736 B) from the q4nx order into the
/// order of the int8-mmul kernel (_Q4K_x_int8 in kernels/q4_core.h), in place:
///   qs     [256 columns][16 B]           -> [k step 16][half t 2][16 columns][8 B]
///          (byte b of a column holds rows 2b, 2b + 1; half t = rows 16t .. 16t+15)
///   scales [group 8][32 rows]            -> [h 2][t 2][a 4][16 rows], group = 4h + a
///   mins, S, M unchanged.
/// One B matrix of the mmul (16 columns x 16 rows) is then 128 contiguous bytes.
void decoding_layer::q4k_relayout_for_mmul(u8* data, size_t bytes)
{
    constexpr size_t BLOCK = 4736, SCALES = 0, QS = 512;
    assert(bytes % BLOCK == 0);
    std::vector<u8> old(BLOCK);
    for (size_t off = 0; off < bytes; off += BLOCK) {
        u8* blk = data + off;
        memcpy(old.data(), blk, BLOCK);
        for (int k = 0; k < 16; k++)                     // 16 columns per k step
            for (int t = 0; t < 2; t++)                  // rows 16t .. 16t+15
                for (int c = 0; c < 16; c++)
                    memcpy(blk + QS + (k * 2 + t) * 128 + c * 8,
                           old.data() + QS + (k * 16 + c) * 16 + t * 8, 8);
        for (int h = 0; h < 2; h++)
            for (int t = 0; t < 2; t++)
                for (int a = 0; a < 4; a++)
                    memcpy(blk + SCALES + ((h * 2 + t) * 4 + a) * 16,
                           old.data() + SCALES + (4 * h + a) * 32 + 16 * t, 16);
    }
}

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
    // the router keeps one weight base address per layer, by this number
    // (MAX_LAYERS in the model header = num_hidden_layers = 48)
    assert(layer_idx >= 0 && layer_idx < num_hidden_layers);
    this->layer_slot = layer_idx;
    u8* w_ptr_begin = this->all_proj_weight_buffer.data();

    _load_moe_weights(q4nx, layer_idx);

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
    // read rms weights and the router weight (same buffer, see the constructor)
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

        // Router weight, stored in the exact order the router reads it, so its
        // DMA is one straight 512 KB read (_send_router_weights). model.q4nx holds
        // it as [D][E]; the router takes 32-expert x 256-input blocks, expert-block
        // major, each block input-major with the 32 experts adjacent. Read in
        // place from [D][E] that was 8192 separate 64-byte pieces per call.
        buffer<bf16> w_gate_inp;
        q4nx.load_weights(w_gate_inp, _weight_desc.ffn_gate_inp.format_name(layer_idx));
        assert(w_gate_inp.size() == (size_t)D * num_experts);
        const int row_block = 32;     // router_weight_row_block (experts per block)
        const int col_block = 256;    // router_weight_col_block (inputs per block)
        bf16* dst = w_rms_ptr;
        for (int eb = 0; eb < num_experts; eb += row_block) {
            for (int db = 0; db < D; db += col_block) {
                for (int d = db; d < db + col_block; d++) {
                    memcpy(dst, w_gate_inp.data() + (size_t)d * num_experts + eb, row_block * sizeof(bf16));
                    dst += row_block;
                }
            }
        }
        w_rms_ptr += _weight_desc.ffn_gate_inp.get_size() / sizeof(bf16);
        assert(dst == w_rms_ptr);
        _weight_desc.ffn_gate_inp.load();

        assert(w_rms_ptr == this->rms_weights_buffer.end());
    }
    // router expert bias, fp32; goes to the router CT with RTP writes (see _gen_sequence)
    {
        buffer<u8> bias_bytes;
        q4nx.load_weights(bias_bytes, "model.layer." + std::to_string(layer_idx) + ".moe_expert_bias");
        assert(bias_bytes.size() == num_experts * sizeof(float));
        this->router_bias = buffer<float>(num_experts);
        memcpy(this->router_bias.data(), bias_bytes.data(), bias_bytes.size());
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

    // every proj weight block into the int8-mmul kernel's order (q4_core.h)
    if (Q4K_INT8_MMUL) {
        q4k_relayout_for_mmul(this->all_proj_weight_buffer.data(), this->all_proj_weight_buffer.size());
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

/// Expert and shared-expert weights, in the order the proj engine and the GLU
/// read them (see models/hy-memory-30b-a3b.h). A Q4_K tensor in model.q4nx is
/// row blocks of 32 rows, each row block's 256-column blocks back to back, so
/// "rows a..b" of a tensor is one contiguous byte range.
void decoding_layer::_load_moe_weights(Q4NX& q4nx, int layer_idx) {
    constexpr int SHARED_ROWS_PER_ROUND = 256;   // UP_GATE_OUTDIM_REORDEER: shared [up 256 | gate 256] per round
    constexpr int EXPERT_ROWS_PER_ROUND = 64;    // EXP_UP_GATE_OUTDIM_REORDER: expert [up 64 | gate 64] per column per round
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

    // routed experts: one expert = [channel 0 stream | channel 1 stream], expert e
    // at e * (its gate/up + down bytes) from the up region's start (the router
    // uses the same offsets). Each channel stream is that channel's gate/up blocks
    // then its down blocks, in the order the proj cores read them, so the router
    // starts one linear BD per channel per expert.
    //   gate/up rows: the 4 cores of a column take 4 row blocks per round; with
    //     64-row pieces cores 0, 1 make up and cores 2, 3 make gate, so one round
    //     of a column is [up 64 | gate 64]
    //   down rows: in order
    //   channels: reorder_cpy(..., 4) puts the 4 blocks of a group next to each
    //     other; channel 0 feeds cores 0, 1 (the first 2), channel 1 cores 2, 3
    {
        LOG_VERBOSE(1, "Loading routed expert weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up_exps;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate_exps;
        weight_desc_t& desc_down = _weight_desc.ffn_down_exps;
        // up, gate, down are one region, re-cut into experts
        assert(desc_gate.offset == desc_up.offset + desc_up.get_size());
        assert(desc_down.offset == desc_gate.offset + desc_gate.get_size());
        buffer<u8> w_up_exps;
        buffer<u8> w_gate_exps;
        buffer<u8> w_down_exps;
        q4nx.load_weights(w_up_exps, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate_exps, desc_gate.format_name(layer_idx));
        q4nx.load_weights(w_down_exps, desc_down.format_name(layer_idx));
        assert(w_up_exps.size() == desc_up.get_size() && w_gate_exps.size() == desc_gate.get_size());
        assert(w_down_exps.size() == desc_down.get_size());

        const size_t up_per_expert = desc_up.get_size() / num_experts;
        const size_t down_per_expert = desc_down.get_size() / num_experts;
        assert(up_per_expert == row_bytes(intermediate_size, D));
        const size_t gate_up_bytes = 2 * up_per_expert;
        const size_t expert_bytes = gate_up_bytes + down_per_expert;
        const size_t channel_bytes = expert_bytes / 2;
        const size_t group = 4 * a_block_size;          // the 4 blocks of one group

        // ch0 takes the first 2 blocks of every group of `src`, ch1 the last 2
        auto split_channels = [&](u8*& ch0, u8*& ch1, const u8* src, size_t bytes) {
            assert(bytes % group == 0);
            for (size_t g = 0; g < bytes; g += group) {
                memcpy(ch0, src + g, group / 2);              ch0 += group / 2;
                memcpy(ch1, src + g + group / 2, group / 2);  ch1 += group / 2;
            }
        };

        buffer<u8> both(gate_up_bytes);
        buffer<u8> gate_up(gate_up_bytes);
        buffer<u8> down(down_per_expert);
        for (int e = 0; e < num_experts; e++) {
            interleave_up_gate(both, w_up_exps.data() + e * up_per_expert, w_gate_exps.data() + e * up_per_expert,
                               intermediate_size, EXPERT_ROWS_PER_ROUND);
            reorder_cpy(gate_up.data(), both, D, flm_q4k, 4);
            buffer<u8> w_down_exp(w_down_exps.data() + e * down_per_expert, down_per_expert);
            reorder_cpy(down.data(), w_down_exp, intermediate_size, flm_q4k, 4);

            u8* ch0 = w_ptr_begin + desc_up.offset + e * expert_bytes;
            u8* ch1 = ch0 + channel_bytes;
            split_channels(ch0, ch1, gate_up.data(), gate_up_bytes);
            split_channels(ch0, ch1, down.data(), down_per_expert);
            assert(ch0 == w_ptr_begin + desc_up.offset + e * expert_bytes + channel_bytes);
        }
        desc_up.load();
        desc_gate.load();
        desc_down.load();
    }
    // shared expert, up/gate: one proj round (16 cores x 32 rows) is [up 256 | gate 256]
    {
        LOG_VERBOSE(1, "Loading shared up/gate weights");
        weight_desc_t& desc_up = _weight_desc.ffn_up_shexp;
        weight_desc_t& desc_gate = _weight_desc.ffn_gate_shexp;
        assert(desc_gate.offset == desc_up.offset + desc_up.get_size());
        buffer<u8> w_up_shexp;
        buffer<u8> w_gate_shexp;
        q4nx.load_weights(w_up_shexp, desc_up.format_name(layer_idx));
        q4nx.load_weights(w_gate_shexp, desc_gate.format_name(layer_idx));
        assert(w_up_shexp.size() == desc_up.get_size() && w_gate_shexp.size() == desc_gate.get_size());

        buffer<u8> both(desc_up.get_size() + desc_gate.get_size());
        interleave_up_gate(both, w_up_shexp.data(), w_gate_shexp.data(), intermediate_size, SHARED_ROWS_PER_ROUND);
        reorder_cpy(w_ptr_begin + desc_up.offset, both, D, flm_q4k);
        desc_up.load();
        desc_gate.load();
    }
    // shared expert, down: plain row order
    {
        LOG_VERBOSE(1, "Loading shared down weights");
        weight_desc_t& desc = _weight_desc.ffn_down_shexp;
        buffer<u8> w_down_shexp;
        q4nx.load_weights(w_down_shexp, desc.format_name(layer_idx));
        assert(w_down_shexp.size() == desc.get_size());
        reorder_cpy(w_ptr_begin + desc.offset, w_down_shexp, intermediate_size, flm_q4k);
        desc.load();
    }
}

buffer<bf16> decoding_layer::operator()(buffer<bf16>& input) {
    // the KV cache is full: the new token has no row to go to
    assert(this->current_context_length < MAX_L);
    // HY_TIME_HOST=1: print where one call's host time goes (microseconds)
    static const bool time_host = std::getenv("HY_TIME_HOST") != nullptr;
    using clk = std::chrono::steady_clock;
    auto t = [](clk::time_point a, clk::time_point b) { return (long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };
    clk::time_point t0 = clk::now();

    // Only what changes per call goes to / comes from the NPU: the layer input and
    // the output. The weights and norms were copied once in load_weights(); the
    // RoPE table is copied by _set_rope_weights(); the KV cache is written by the
    // NPU only (get_k_cache / get_v_cache copy it back when asked). Copying them
    // every call cost ~4 ms of host time (weights 2.9 ms, KV 2 x 0.5 ms).
    memcpy(this->hidden_states_buffer.data(), input.data(), D * sizeof(bf16));
    if (debug_tap == "x") {
        // fill the tap area with a marker (0xFFFF, a bf16 NaN) to count what arrives
        const size_t tap_len = 10 * D;
        memset(this->kv_cache.data() + kv_cache_size - tap_len, 0xFF, tap_len * sizeof(bf16));
        this->kv_cache.sync_to_device();
    }
    this->hidden_states_buffer.sync_to_device();
    clk::time_point t1 = clk::now();
    // the call's end state: anything but completed (e.g. a timeout) means the
    // output is not this call's result (the host test counts these)
    ert_cmd_state state = this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, rope_rms_buffer, kv_cache);
    this->last_call_completed = (state == ERT_CMD_STATE_COMPLETED);
    clk::time_point t2 = clk::now();
    this->hidden_states_buffer.sync_from_device();
    if (debug_tap == "x") {
        this->kv_cache.sync_from_device();    // report_debug_tap reads the tap area
    }
    clk::time_point t3 = clk::now();
    this->set_context_length(this->current_context_length + 1);
    clk::time_point t4 = clk::now();
    if (time_host) {
        std::cout << "host us: x in " << t(t0, t1) << ", NPU run (+ setup) " << t(t1, t2)
                  << ", y out " << t(t2, t3) << ", new sequence " << t(t3, t4) << std::endl;
    }
    return this->hidden_states_buffer;
}


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
    int proj_rtp_lock_id = 6;
    int router_shimtile_configure_ready_lock = 10;   // expert BDs written, safe to push expert tasks
    assert(L > 0 && L <= MAX_L);
    seq->clear_cmds();

    for (int i = 0; i < attn_cus; i++) {
        seq->rtp_write(get_tile(attn_qk_rc[i][0], attn_qk_rc[i][1]), l_qk_address[i], L);
        seq->rtp_write(get_tile(attn_kv_rc[i][0], attn_kv_rc[i][1]), l_kv_address[i], L);
    }
    for (int i = 0; i < 16; i++){
        seq->rtp_write(proj_tiles[i], 17408, 1);
        seq->rtp_write(proj_tiles[i], CT_lock_address_base + 16 * proj_rtp_lock_id, 1); // set lock to 1
    }

    // which layer this is: the router picks this layer's saved weight base address
    seq->rtp_write(get_tile(router_rc[0], router_rc[1]), router_layer_address, (uint32_t)layer_slot);

    // router expert bias: one RTP write per expert (fp32 bit pattern). The router
    // reads it only after its logits, long after these writes; the previous call
    // has finished with it (a call ends after the MoE output arrives).
    // TODO(MoE step 2): 128 writes per call; a DMA path would be cheaper.
    assert(router_bias.size() == (size_t)num_experts);
    for (int e = 0; e < num_experts; e++) {
        uint32_t bits;
        memcpy(&bits, &router_bias[e], sizeof(bits));
        seq->rtp_write(get_tile(router_rc[0], router_rc[1]), router_bias_address + 4 * e, bits);
    }

    _send_hidden_states(seq);
    _send_rms_weights(seq, 0);
    if (debug_stop_after == 11) { seq->cmds2seq(); return; }
    _send_rope_rms_weights(seq);
    if (debug_stop_after == 12) { seq->cmds2seq(); return; }

    // Router weight first: MT5 holds all of it (512 KB), so IT5 fills it while
    // attention runs, and the router + helper start on it as soon as x is there.
    _send_router_weights(seq, num_experts, D, _weight_desc.ffn_gate_inp.offset);

    // DEBUG (HY_DEBUG_TAP=x with a DEBUG_TAP=x build): catch the copy of rms's
    // x for the proj engine in the unused tail of the KV buffer. Not waited on.
    if (debug_tap == "x") {
        const uint32_t rms_len = 10 * D;              // rms output, 10 sends (QKV rounds)
        const uint32_t base = (uint32_t)(kv_cache_size - rms_len);
        seq->npu_dma_memcpy_nd(sizeof(bf16), kv_cache_arg_idx, S2MM, IT4, bd_8, it_channel_0,
            {0, 0, 0, base}, {1, 1, 1, rms_len}, {0, 0, 0, 1}, -1, 0, false);
    }

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
    if (attn_tap) {
        _receive_attn_tap(seq);
    }
    // the new K/V row's receive BDs go first, as in the production sequence
    _receive_kv_cache(seq, L);
    _move_weights_q4k(seq, DK + DV + DQ, D, _weight_desc.attn_k.offset);  // k, v, q
    if (debug_stop_after == 1) { seq->cmds2seq(); return; }
    if (debug_stop_after == 2) {
        for (int i = 0; i < 2 * attn_cols; i++) {
            seq->npu_dma_wait(attn_shims[0], S2MM, it_channel_0);
        }
        seq->cmds2seq();
        return;
    }
    _move_kv_cache(seq, L);
    if (debug_stop_after == 3) {
        if (attn_tap) {
            seq->npu_dma_wait(attn_shims[0], S2MM, it_channel_1);
        }
        seq->cmds2seq();
        return;
    }
    _move_weights_q4k(seq, D, DQ, _weight_desc.attn_output.offset);
    if (debug_stop_after == 4) { seq->cmds2seq(); return; }

    // Routed expert weight BDs on shim 0, 1, 6, 7: one linear BD per channel,
    // 4 / 5 for round 0 and 12 / 13 for round 1 (SHIM_X_expert_BD_ID in the
    // kernels). Written here but not started; the router CT patches in the chosen
    // expert's address and starts them with control packets.
    for (int i = 0; i < 4; i++) {
        setup_expert_q4k(seq, i, 4, _weight_desc.ffn_up_exps.offset);
        setup_expert_q4k(seq, i, 12, _weight_desc.ffn_up_exps.offset);
    }

    if (debug_stop_after == 45) { seq->cmds2seq(); return; }

    // shared expert up-gate, then down (down's last round pushed but not yet waited for)
    _move_weights_q4k(seq, 2 * intermediate_size, D, _weight_desc.ffn_up_shexp.offset);
    _move_weights_q4k(seq, D, intermediate_size, _weight_desc.ffn_down_shexp.offset, false);

    // Tell the router the expert BDs are ready. Only now: the router pushes the
    // routed-expert weight tasks onto the same shim queues (0, 1, 6, 7) as the
    // shared-expert ones above, and a router that got here first queued them
    // in front of the last shared-down weights (wrong output from the 2nd call
    // on, seen once the router weights were sent early).
    // (Setting this earlier and having the router read the BD base address
    // while the shared expert streamed hung every call; MoE step 2, T3.)
    auto router_CT = get_tile(router_rc[0], router_rc[1]);
    seq->npu_maskwrite(router_CT, 0x32038, 0x1, 0x1);
    seq->rtp_write(router_CT, CT_lock_address_base + 16 * router_shimtile_configure_ready_lock, 1); // set lock to 1
    _wait_proj_weights(seq);
    if (debug_stop_after == 5) { seq->cmds2seq(); return; }

    // wait for receiving y
    seq->npu_dma_wait(
        xyr_tile,
        S2MM,
        it_channel_1
    );
    if (attn_tap) {
        seq->npu_dma_wait(attn_shims[0], S2MM, it_channel_1);
    }

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
        {1, 1, 1, (uint32_t)D*2 },
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
}

void decoding_layer::_receive_kv_cache(npu_sequence* seq, const int L){
    // RoPE sends the new token's K (all KV heads) then V (all KV heads) through
    // IT3. Write each column's heads into that column's K / V block, row L - 1:
    // BDs 0..3 = K col 0, K col 1, V col 0, V col 1, in stream order.
    const uint32_t row_offset = (uint32_t)((L - 1) * kv_col_row);
    for (int kv = 0; kv < 2; kv++) {               // 0: K, 1: V
        for (int col = 0; col < attn_cols; col++) {
            uint32_t block_offset = (uint32_t)(kv_block(col, kv) * kv_col_half);
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                S2MM,
                attn_shims[0],
                npu_bd_id(kv * attn_cols + col),
                it_channel_0,
                {0, 0, 0, block_offset + row_offset},
                {1, 1, 1, (uint32_t)kv_col_row},
                {0, 0, 0, 1},
                -1, 0, true
            );
        }
    }
}

void decoding_layer::_receive_attn_tap(npu_sequence* seq){
    // attention output (before o_proj), from MT3 through IT3 S2MM 1
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        xy_arg_idx,
        S2MM,
        attn_shims[0],
        bd_6,
        it_channel_1,
        {0, 0, 0, (uint32_t)D},
        {1, 1, 1, (uint32_t)DQ},
        {0, 0, 0, 1},
        -1, 0, true
    );
}

void decoding_layer::_send_router_weights(npu_sequence* seq, const size_t NUM_expert, const size_t D, const size_t W_offset_in_byte){
    // The host stored the router weight in read order (32-expert x 256-input
    // blocks, see load_weights), so this is one straight read of NUM_expert x D.
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        w_rms_arg_idx,
        MM2S,
        IT5,
        npu_bd_id(bd_12),
        it_channel_0,
        {0, 0, 0, (uint32_t)((W_offset_in_byte/2)  )},
        {1, 1, 1, (uint32_t)(NUM_expert * D)},
        {0, 0, 0, 1},
        10, 0, false, aggressive_cache
    );
}


/// One routed expert's weight BDs on proj shim `col` (0-3): channel 0 gets BD
/// bd_id, channel 1 BD bd_id + 1, each one linear read of half an expert (see
/// _load_moe_weights). Points at expert 0; the router patches in the chosen
/// expert's address and pushes the BDs onto the queues.
void decoding_layer::setup_expert_q4k(npu_sequence* seq, int col, int bd_id, size_t const_w_byte_offset){
    const uint32_t a_block_size = get_quantization_byte_size(QXNX_ROW_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE, flm_q4k);
    const size_t gate_up_bytes = 2 * (size_t)(intermediate_size / QXNX_ROW_BLOCK_SIZE) * (D / QXNX_COL_BLOCK_SIZE) * a_block_size;
    const size_t down_bytes = (size_t)(D / QXNX_ROW_BLOCK_SIZE) * (intermediate_size / QXNX_COL_BLOCK_SIZE) * a_block_size;
    const uint32_t channel_bytes = (uint32_t)((gate_up_bytes + down_bytes) / 2);
    //TODO: sync with the kernel through share hpp
    constexpr uint32_t SHIM_6_CONTROL_RES_PKT = 30;
    for (int ch = 0; ch < 2; ch++) {
        seq->npu_dma_memcpy_nd(
            sizeof(uint8_t),
            w_proj_arg_idx,
            MM2S,
            mvm_tiles[col],
            npu_bd_id(bd_id + ch),
            ch == 0 ? it_channel_0 : it_channel_1,
            {0, 0, 0, (uint32_t)(const_w_byte_offset + (size_t)ch * channel_bytes)},
            {1, 1, 1, channel_bytes},
            {0, 0, 0, 1},
            -1, 0, false,
            aggressive_cache,
            SHIM_6_CONTROL_RES_PKT, false
        );
    }
}

void decoding_layer::_move_weights_q4k(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset, bool wait_last_round){
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
    if (wait_last_round) {
        _wait_proj_weights(seq);
    }
}

/// Wait for one weight task on both channels of each proj shim (0, 1, 6, 7).
void decoding_layer::_wait_proj_weights(npu_sequence* seq){
    for (uint32_t col = 0; col < 4; col++){
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_0);
        seq->npu_dma_wait(mvm_tiles[col], MM2S, it_channel_1);
    }
}


void decoding_layer::_move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
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
    // Each column's shim reads its own K block (channel 0) and V block
    // (channel 1). K reads start once both K writes of the new row are done,
    // V reads once both V writes are.
    const int L_padded = (L + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    assert(L_padded <= MAX_L);
    const uint32_t data2move = (uint32_t)(L_padded * kv_col_row);
    const npu_bd_id read_bd[2][attn_cols] = {{bd_4, bd_0}, {bd_5, bd_1}};  // IT3 uses BDs 0..3 to write

    for (int kv = 0; kv < 2; kv++) {               // 0: K, 1: V
        for (int col = 0; col < attn_cols; col++) {
            seq->npu_dma_wait(attn_shims[0], S2MM, it_channel_0);
        }
        for (int col = 0; col < attn_cols; col++) {
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                kv_cache_arg_idx,
                MM2S,
                attn_shims[col],
                read_bd[kv][col],
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
    // operator() does not sync the cache (only the NPU touches it), so push it here
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

buffer<bf16> decoding_layer::get_attn_tap(){
    assert(attn_tap);
    buffer<bf16> tap(DQ);
    memcpy(tap.data(), this->hidden_states_buffer.data() + D, DQ * sizeof(bf16));
    return tap;
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
    if (debug_tap != "x") return;
    const size_t rms_len = 10 * D;
    const uint16_t* base = reinterpret_cast<const uint16_t*>(this->kv_cache.data() + kv_cache_size - rms_len);
    auto arrived = [](const uint16_t* p, size_t n) {
        size_t count = 0;
        for (size_t i = 0; i < n; i++) count += (p[i] != 0xFFFF);
        return count;
    };
    header_print("debug", "x tap: rms output " << arrived(base, rms_len) << " / " << rms_len);
}
