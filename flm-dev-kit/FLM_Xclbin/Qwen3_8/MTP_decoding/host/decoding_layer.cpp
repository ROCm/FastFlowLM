// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "decoding_layer.hpp"
#include "quant_size.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

namespace {
/// rope_theta from the Qwen3.8-27B config. The draft head shares the main
/// model's, and cpu_ref_common.hpp's q38::mtp::ROPE_THETA is the same number.
constexpr double mtp_rope_theta = 1e7;
}

void decoding_layer::seed_kv_cache(const std::vector<float>& k,
                                   const std::vector<float>& v, int n) {
    assert(n >= 0 && n <= MAX_L);
    assert((int)k.size() >= n * DK && (int)v.size() >= n * DV);
    // The caller keeps two planes and so does the BO, so this is a straight
    // copy per row -- but it goes through k_at/v_at rather than assuming the
    // planes, so a layout flip carries it. See decoding_layer.hpp::kv_cache_size.
    bf16* base = this->kv_cache.data();
    for (int t = 0; t < n; t++) {
        bf16* kp = base + this->k_at(t);
        bf16* vp = base + this->v_at(t);
        for (int j = 0; j < DK; j++) kp[j] = (bf16)k[(size_t)t * DK + j];
        for (int j = 0; j < DV; j++) vp[j] = (bf16)v[(size_t)t * DV + j];
    }
    this->kv_cache.sync_to_device();
}

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr, int context_length)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr), MAX_L(context_length) {
    header_print("info", "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();
    this->lm_head = this->layer_app_mgr->create_app();
    this->current_context_length = 0;

    num_hidden_layers = lm_config.num_hidden_layers;
    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    num_kv_heads = lm_config.num_key_value_heads;
    this->num_kv_per_round = 2;
    // 4 kv heads / 1 per CU / 2 columns = 2 CUs per column, so 2 heads each.
    // Must agree with iron/model_config.py, which derives the same numbers for
    // the array side.
    this->kv_heads_per_column = num_kv_heads / this->num_attn_columns;
    assert(this->kv_heads_per_column * this->num_attn_columns == num_kv_heads &&
           "the kv heads do not divide evenly across the attention columns");
    intermediate_size = lm_config.intermediate_size;
    D = lm_config.hidden_size;
    // A deeper cache than the bench needs, to exercise the append's addressing.
    // The pass still drives one row at `current_context_length`, so the scores
    // do not depend on this -- what changes is where in DDR the v plane starts,
    // v_at(0) being a whole k plane in. A run at 16384 that still scores k and
    // v is the evidence that splitting them onto two shim channels removed the
    // ceiling rather than moving it: under the old descriptor the k -> v stride
    // masked to 0 at that depth, dropping every v row on top of k, and only the
    // v readback would have noticed. Same knob and same purpose as
    // ../gated_attention_verify.
    if (const char* e = std::getenv("FLM_MAX_L")) {
        const int n = std::atoi(e);
        if (n > 0) {
            MAX_L = n;
            header_print("info", "MAX_L overridden to " << MAX_L << " (FLM_MAX_L)");
        }
    }
    // Both halves of the cap in decoding_layer.hpp::MAX_L_LIMIT. Neither of
    // these faults on hardware if it is violated -- an oversized 10-bit BD size
    // is masked after a printed warning, and a short cache is simply read past
    // -- so they are checked here, where MAX_L arrives, rather than left to the
    // descriptor that would quietly absorb them.
    assert(MAX_L % L_CHUNK == 0 &&
           "MAX_L must be a whole number of L_CHUNKs -- the replay rounds the "
           "key count up to a chunk boundary and reads the rows it lands on");
    assert(MAX_L <= MAX_L_LIMIT &&
           "MAX_L is past what the replay's 10-bit chunk count can describe "
           "(decoding_layer.hpp::MAX_L_LIMIT)");
    kv_cache_size = (size_t)MAX_L * (DK + DV);
    // In bf16 elements, to match _move_weights()'s element-sized DMA.
    w_fc_offset = 0;
    w_qkv_offset = q4k_bytes((size_t)(2 * D) * D) / sizeof(bf16);
    w_gate_offset = w_qkv_offset + q4k_bytes((size_t)(DQ + DK + DV) * D) / sizeof(bf16);
    w_o_offset = w_gate_offset + q4k_bytes((size_t)DQ * D) / sizeof(bf16);
    w_up_offset = w_o_offset + q4k_bytes((size_t)D * DQ) / sizeof(bf16);
    w_down_offset = w_up_offset + q4k_bytes((size_t)(2 * intermediate_size) * D) / sizeof(bf16);

    const float rope_partial_factor = 0.25f; // use 25% of head_dim for RoPE, the rest for QK norm if applicable

    RoPE_DIM = int(DH * rope_partial_factor);

    l_begin_mha_address = lm_config.addr_l_begin_mha;
    l_end_mha_address = lm_config.addr_l_end_mha;
    kk_address = lm_config.addr_kk;
    rms_address = model_list::get_rms_addr(this->model_type);

    header_print("info", "Decoding layer config: ");
    header_print("info", "  D : " << D);
    header_print("info", "  DQ: " << DQ);
    header_print("info", "  DK: " << DK);
    header_print("info", "  DV: " << DV);
    header_print("info", "  RoPE Dim: " << RoPE_DIM);
    header_print("info", "  Intermediate Size: " << intermediate_size);
    has_qk_norm = model_list::has_qk_norm(this->model_type);
    header_print("info", "  Has QK Norm: " << (has_qk_norm ? "Yes" : "No"));
    header_print("info", "  Num KV Per Round: " << this->num_kv_per_round);
    // No rope_scaling member: get_rope_scaling() returns 1.0f for this model,
    // and set_rope_position() builds the pair straight from rope_theta.

    size_t total_proj_weights = 
        2 * D * D + (DQ + DK + DV) * D
        + DQ * D
        + DQ * D
        + 3 * D * intermediate_size;

    // q4k is 4.625 bits/weight, not 5: the 5/8 shorthand over-counts a 32x256
    // block as 5120 B when it is 4736.
    size_t total_proj_weights_byte_size = q4k_bytes(total_proj_weights);

    // The 2 * D fusion input, then the layer output behind it. The tap sits
    // behind the input rather than over it: the cores are still reading x when
    // the first row blocks come back.
    this->layer_out_offset = (size_t)2 * D;
    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(
        this->layer_out_offset + (size_t)D);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(total_proj_weights_byte_size);
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(4 * D);
    
    this->rope_w_offset = (size_t)RoPE_DIM;
    this->rope_rms_buffer = this->layer_app.create_bo_buffer<bf16>(
        this->has_qk_norm ? this->rope_w_offset + 2 * (size_t)DH
                          : this->rope_w_offset);
    this->rope_rms_buffer.memset((bf16)0);
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();

}

void decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
    const int a_block_size = q4k_bytes(32 * 256);
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
    u8* w_ptr = this->all_proj_weight_buffer.data();
    u8* w_ptr_end = this->all_proj_weight_buffer.end();

    {
        LOG_VERBOSE(1, "Loading MTP fc weight");
        const size_t FC_IN = 2 * (size_t)this->D;
        const size_t fc_bytes = q4k_bytes((size_t)this->D * FC_IN);

        buffer<u8> w_fc;
        q4nx.load_weights(w_fc, "model.mtp.fc.weight");
        if (w_fc.size() != fc_bytes) {
            header_print("error", "model.mtp.fc.weight is " + std::to_string(w_fc.size()) +
                                      " B, expected " + std::to_string(fc_bytes));
            assert(false && "fc weight size mismatch");
        }
        assert(w_ptr + fc_bytes <= w_ptr_end);
        this->w_fc_offset = 0;
        reorder_cpy(w_ptr, w_fc, (int)FC_IN);
        w_ptr += fc_bytes;

        LOG_VERBOSE(1, "Loading MTP pre-fc norm weights");
        buffer<bf16> w_norm_embed;
        buffer<bf16> w_norm_hidden;
        q4nx.load_weights(w_norm_embed,  "model.mtp.pre_fc_norm_embedding.weight");
        q4nx.load_weights(w_norm_hidden, "model.mtp.pre_fc_norm_hidden.weight");
        assert((int)w_norm_embed.size()  == this->D);
        assert((int)w_norm_hidden.size() == this->D);
        memcpy(this->rms_weights_buffer.data(),
               w_norm_embed.data(),  (size_t)this->D * sizeof(bf16));
        memcpy(this->rms_weights_buffer.data() + this->D,
               w_norm_hidden.data(), (size_t)this->D * sizeof(bf16));
    }

    {
        LOG_VERBOSE(1, "Loading MTP draft-layer qkv weights");
        buffer<u8> w_q;
        buffer<u8> w_k;
        buffer<u8> w_v;
        q4nx.load_weights(w_q, "model.mtp.layers.0.self_attn.q_proj.weight");
        q4nx.load_weights(w_k, "model.mtp.layers.0.self_attn.k_proj.weight");
        q4nx.load_weights(w_v, "model.mtp.layers.0.self_attn.v_proj.weight");

        const size_t q_bytes = q4k_bytes((size_t)DQ * D);
        const size_t k_bytes = q4k_bytes((size_t)DK * D);
        const size_t v_bytes = q4k_bytes((size_t)DV * D);
        assert(w_q.size() == 2 * q_bytes && "q_proj holds q and the output gate");
        assert(w_k.size() == k_bytes && w_v.size() == v_bytes);

        buffer<u8> w_q_only(w_q.data(), q_bytes);

        assert(w_ptr + k_bytes + v_bytes + q_bytes <= w_ptr_end);
        assert((size_t)(w_ptr - this->all_proj_weight_buffer.data())
               == this->w_qkv_offset * sizeof(bf16) &&
               "w_qkv_offset must name where the walk actually put qkv");
        assert(DK % (32 * 16) == 0 && DV % (32 * 16) == 0 && DQ % (32 * 16) == 0 &&
               "each region must be a whole number of engine rounds or the "
               "k|v|q split lands mid-round and the RoPE tile is fed garbage");
        reorder_cpy(w_ptr, w_k, D);
        w_ptr += k_bytes;
        reorder_cpy(w_ptr, w_v, D);
        w_ptr += v_bytes;
        reorder_cpy(w_ptr, w_q_only, D);
        w_ptr += q_bytes;

        LOG_VERBOSE(1, "Loading MTP draft-layer output-gate weights");
        buffer<u8> w_gate(w_q.data() + q_bytes, q_bytes);
        assert((size_t)(w_ptr - this->all_proj_weight_buffer.data())
               == this->w_gate_offset * sizeof(bf16) &&
               "w_gate_offset must name where the walk actually put the gate");
        assert(w_ptr + q_bytes <= w_ptr_end);
        reorder_cpy(w_ptr, w_gate, D);
        w_ptr += q_bytes;

        LOG_VERBOSE(1, "Loading MTP input_layernorm weight");
        buffer<bf16> w_in_ln;
        q4nx.load_weights(w_in_ln, "model.mtp.layers.0.input_layernorm.weight");
        assert((int)w_in_ln.size() == this->D);
        memcpy(this->rms_weights_buffer.data() + 2 * this->D,
               w_in_ln.data(), (size_t)this->D * sizeof(bf16));

        LOG_VERBOSE(1, "Loading MTP post_attention_layernorm weight");
        buffer<bf16> w_post_ln;
        q4nx.load_weights(w_post_ln, "model.mtp.layers.0.post_attention_layernorm.weight");
        assert((int)w_post_ln.size() == this->D);
        memcpy(this->rms_weights_buffer.data() + 3 * this->D,
               w_post_ln.data(), (size_t)this->D * sizeof(bf16));
    }

    {
        LOG_VERBOSE(1, "Loading MTP draft-layer o_proj weights");
        buffer<u8> w_o;
        q4nx.load_weights(w_o, "model.mtp.layers.0.self_attn.o_proj.weight");
        const size_t o_bytes = q4k_bytes((size_t)this->D * this->DQ);
        assert(w_o.size() == o_bytes && "o_proj is [D][DQ]");
        assert((size_t)(w_ptr - this->all_proj_weight_buffer.data())
               == this->w_o_offset * sizeof(bf16) &&
               "w_o_offset must name where the walk actually put o_proj");
        assert(w_ptr + o_bytes <= w_ptr_end);
        reorder_cpy(w_ptr, w_o, this->DQ);
        w_ptr += o_bytes;
    }
    {
        LOG_VERBOSE(1, "Loading MTP draft-layer up/gate weights");
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up,   "model.mtp.layers.0.mlp.up_proj.weight");
        q4nx.load_weights(w_gate, "model.mtp.layers.0.mlp.gate_proj.weight");

        const int m_per_round = 32 * 16;
        const size_t chunk_size = q4k_bytes((size_t)m_per_round * D);
        const size_t total_size = q4k_bytes((size_t)intermediate_size * D);
        assert(w_up.size() == total_size && w_gate.size() == total_size &&
               "up/gate are [intermediate_size][D]");
        assert(total_size % chunk_size == 0 &&
               "a round has to be all up or all gate");
        const int phases = (int)(total_size / chunk_size);

        assert((size_t)(w_ptr - this->all_proj_weight_buffer.data())
               == this->w_up_offset * sizeof(bf16) &&
               "w_up_offset must name where the walk actually put up|gate");
        assert(w_ptr + 2 * total_size <= w_ptr_end);

        tensor_2d<u8> w_up_tensor(w_up, (uint32_t)chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, (uint32_t)chunk_size, 0);
        for (int i = 0; i < phases; i++) {
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
    }

    {
        LOG_VERBOSE(1, "Loading MTP draft-layer down weights");
        buffer<u8> w_down;
        q4nx.load_weights(w_down, "model.mtp.layers.0.mlp.down_proj.weight");
        const size_t down_bytes = q4k_bytes((size_t)this->D * this->intermediate_size);
        assert(w_down.size() == down_bytes && "down_proj is [D][intermediate_size]");
        assert((size_t)(w_ptr - this->all_proj_weight_buffer.data())
               == this->w_down_offset * sizeof(bf16) &&
               "w_down_offset must name where the walk actually put down_proj");
        assert(w_ptr + down_bytes <= w_ptr_end);
        reorder_cpy(w_ptr, w_down, this->intermediate_size);
        w_ptr += down_bytes;
    }

    if (this->has_qk_norm) {
        LOG_VERBOSE(1, "Loading MTP q/k norm weights");
        buffer<bf16> w_q_norm;
        buffer<bf16> w_k_norm;
        q4nx.load_weights(w_q_norm, "model.mtp.layers.0.self_attn.q_norm.weight");
        q4nx.load_weights(w_k_norm, "model.mtp.layers.0.self_attn.k_norm.weight");
        assert((int)w_q_norm.size() == this->DH && (int)w_k_norm.size() == this->DH);
        memcpy(this->rope_rms_buffer.data() + this->rope_w_offset,
               w_q_norm.data(), (size_t)this->DH * sizeof(bf16));
        memcpy(this->rope_rms_buffer.data() + this->rope_w_offset + this->DH,
               w_k_norm.data(), (size_t)this->DH * sizeof(bf16));
    }

    this->all_proj_weight_buffer.sync_to_device();
    this->rms_weights_buffer.sync_to_device();
    this->rope_rms_buffer.sync_to_device();
    header_print("info", "Weights loaded!");
}

buffer<bf16> decoding_layer::operator()(buffer<bf16>& input) {
    const size_t FC_IN = 2 * (size_t)this->D;
    const size_t QKV = (size_t)(this->DQ + this->DK + this->DV);
    (void)QKV;
    assert(input.size() >= FC_IN &&
           "the draft head's input is the fusion's two D-wide halves");
    assert(this->hidden_states_buffer.size()
               >= this->layer_out_offset + (size_t)this->D &&
           "hidden_states_buffer must hold the input halves and the output tap");

    memcpy(this->hidden_states_buffer.data(), input.data(), FC_IN * sizeof(bf16));

    // FLM_MTP_STALL: localise a hung pass without touching the design. Zero
    // every region the array is supposed to write, so that after a timeout
    // "still zero" means the pass never reached that stage. The three regions
    // are in pipeline order -- k|v append, then the layer output tap -- so the
    // pattern says whether the stall is upstream or downstream of attention.
    const bool stall_probe = getenv("FLM_MTP_STALL") != nullptr;
    const int probe_slot = this->current_context_length;
    if (stall_probe) {
        memset(this->hidden_states_buffer.data() + this->layer_out_offset, 0,
               (size_t)this->D * sizeof(bf16));
        memset(this->kv_cache.data() + this->k_at(probe_slot), 0,
               (size_t)this->DK * sizeof(bf16));
        memset(this->kv_cache.data() + this->v_at(probe_slot), 0,
               (size_t)this->DV * sizeof(bf16));
        this->kv_cache.sync_to_device();
    }
    this->hidden_states_buffer.sync_to_device();

    this->_gen_sequence(this->layer_app.seq(), this->current_context_length + 1);
    // The dispatch state is CHECKED. A timeout leaves hidden_states_buffer
    // holding the previous pass's output, so an unchecked call reads back a
    // plausible row and reports success -- which is how the repeat probe below
    // "proved" 33 dispatches that had in fact all timed out after the first.
    const ert_cmd_state st =
        this->layer_app(this->hidden_states_buffer, this->all_proj_weight_buffer,
                        this->rms_weights_buffer, this->rope_rms_buffer,
                        this->kv_cache);
    if (st != ERT_CMD_STATE_COMPLETED) {
        header_print("error", "layer dispatch state " << (int)st
                       << " at context length " << this->current_context_length);
        if (stall_probe) {
            try {
                this->kv_cache.sync_from_device();
                this->hidden_states_buffer.sync_from_device();
                auto nz = [](const bf16* p, size_t n) {
                    size_t c = 0;
                    for (size_t i = 0; i < n; i++) if ((float)p[i] != 0.0f) c++;
                    return c;
                };
                header_print("info", "  stall probe, nonzero words written:"
                    << " k " << nz(this->kv_cache.data() + this->k_at(probe_slot),
                                   (size_t)this->DK) << "/" << this->DK
                    << ", v " << nz(this->kv_cache.data() + this->v_at(probe_slot),
                                    (size_t)this->DV) << "/" << this->DV
                    << ", layer_out "
                    << nz(this->hidden_states_buffer.data() + this->layer_out_offset,
                          (size_t)this->D) << "/" << this->D);
            } catch (const std::exception& e) {
                header_print("error", std::string("  stall probe readback threw: ")
                                          + e.what());
            }
        }
        exit(1);
    }
    this->hidden_states_buffer.sync_from_device();

    buffer<bf16> out((size_t)this->D);
    memcpy(out.data(), this->hidden_states_buffer.data() + this->layer_out_offset,
           (size_t)this->D * sizeof(bf16));
    return out;
}

void decoding_layer::set_rope_position(int position) {
    const int half = this->RoPE_DIM / 2;
    bf16* p = this->rope_rms_buffer.data();
    for (int j = 0; j < half; j++) {
        const double inv = std::pow((double)mtp_rope_theta,
                                    -(double)(2 * j) / (double)this->RoPE_DIM);
        const double ang = (double)position * inv;
        p[j]        = (bf16)(float)std::cos(ang);
        p[half + j] = (bf16)(float)std::sin(ang);
    }
    this->rope_rms_buffer.sync_to_device();
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

/// \param L the number of keys already in the head's cache, i.e. the slot the
///          next drafted token will occupy and the position it rotates at.
void decoding_layer::set_context_length(int L){
    this->current_context_length = L;

    this->_gen_sequence(this->layer_app.seq(), L + 1);

    this->set_rope_position(L);
}

/// \param L context length; unused until attention is wired.
void decoding_layer::_gen_sequence(npu_sequence* seq, int L) {
    constexpr size_t CT_lock_address_base = 0x000001F000;
    int proj_rtp_lock_id = 6;
    int rms_rtp_lock_id = 6;
    int gate_rtp_lock_id = 6;
    (void)L;
    (void)CT_lock_address_base;
    (void)proj_rtp_lock_id; (void)rms_rtp_lock_id; (void)gate_rtp_lock_id;
    seq->clear_cmds();

    // FLM_MTP_TRUNC=n: emit only the first n phases and stop. Every
    // _move_weights() ends in a DMA wait for the round it just posted, so a
    // truncated sequence still blocks whenever the mvm meant to consume that
    // round is stalled -- which turns the dispatch state into a bisection for
    // *where* a hung pass is stuck. Unset (or 0) is the normal full sequence.
    //   1 fc  2 qkv  3 z  4 append wait + replay  5 o  6 up|gate  7 down
    // A truncated pass leaves the array mid-flight, so only the dispatch it is
    // used on is meaningful; nothing after it is.
    const int trunc = getenv("FLM_MTP_TRUNC") ? atoi(getenv("FLM_MTP_TRUNC")) : 0;
    auto stop_after = [&](int phase) {
        if (trunc > 0 && phase >= trunc) { seq->cmds2seq(); return true; }
        return false;
    };

    _write_attn_L(seq);

    // --- phase 1: the fusion, 2 * D -> D -----------------------------------
    _send_fc_norm_weights(seq);
    _send_fc_input(seq);

    // --- phase 2: input_layernorm, then qkv --------------------------------
    _send_rms_weights(seq);

    // --- phase 3: q/k norm and the rotation --------------------------------
    _send_rope_rms_weights(seq);

    _receive_kv_cache(seq, this->current_context_length);
    _receive_layer_out(seq);

    _move_weights(seq, D, 2 * (size_t)D, w_fc_offset);
    if (stop_after(1)) return;
    _move_weights(seq, (DQ + DK + DV), D, w_qkv_offset);
    if (stop_after(2)) return;
    _move_weights(seq, DQ, D, w_gate_offset);
    if (stop_after(3)) return;

    // --- phase 4: attention -------------------------------------------------
    // The append is two transfers per column now, k on channel 0 and v on
    // channel 1, so both have to be drained before the replay reads the row
    // back. A wait is per shim *and* per channel: waiting only on channel 0
    // would let the replay start with v still in flight, and the damage would
    // be a plausible-looking output rather than a hang.
    for (int c = 0; c < num_attn_columns; c++) {
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_0);   // k
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_1);   // v
    }

    assert(L == this->current_context_length + 1 &&
           "the replay and the attention L RTP must describe the same context");
    _move_kv_cache(seq);
    if (stop_after(4)) return;

    // --- phase 5: o_proj ----------------------------------------------------
    _move_weights(seq, D, DQ, w_o_offset);
    if (stop_after(5)) return;

    // --- phase 6: the MLP's up | gate ---------------------------------------
    _move_weights(seq, 2 * intermediate_size, D, w_up_offset);
    if (stop_after(6)) return;

    // --- phase 7: down, then the MLP residual -------------------------------
    _move_weights(seq, D, intermediate_size, w_down_offset);
    if (stop_after(7)) return;

    // The output tap: h2, the whole layer.
    seq->npu_dma_wait(layer_out_tap_tile, S2MM, it_channel_1);

    seq->cmds2seq();

    if (const char* d = getenv("FLM_MTP_SEQDUMP")) {   // TEMPORARY
        auto pr = seq->dump();
        std::string fn = std::string(d) + "/scaffold_L" + std::to_string(L) + ".txt";
        std::ofstream f(fn);
        for (size_t i = 0; i < pr.second; i++) f << std::hex << pr.first[i] << "\n";
    }
}

void decoding_layer::_send_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, (uint32_t)(2 * this->D)},
        {1, 1, 1, (uint32_t)(2 * this->D)},
        {0, 0, 0, 1},
        rms_w_pkt_id, 0, false, aggressive_cache
    );
}

void decoding_layer::_send_rope_rms_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_rms_arg_id,
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

void decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = q4k_bytes(elems) / sizeof(bf16);
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
    
void decoding_layer::_move_weights_q80(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 128;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = elems * 10 / 8 / 2;
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

void decoding_layer::_send_fc_norm_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), rms_arg_id, MM2S, pass_tile, bd_1, it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)this->D},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), rms_arg_id, MM2S, pass_tile, bd_2, it_channel_1,
        {0, 0, 0, (uint32_t)this->D},
        {1, 1, 1, (uint32_t)this->D},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

void decoding_layer::_send_fc_input(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), x_arg_id, MM2S, pass_tile, bd_0, it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)this->D},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), x_arg_id, MM2S, pass_tile, bd_3, it_channel_0,
        {0, 0, 0, (uint32_t)this->D},
        {1, 1, 1, (uint32_t)this->D},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );
}

void decoding_layer::_write_attn_L(npu_sequence* seq){
    const uint32_t keys = (uint32_t)(this->current_context_length + 1);
    const int cus = num_attn_columns * cus_per_column;
    for (int i = 0; i < cus; i++) {
        seq->rtp_write(attn_qk_tiles[i], 9216, keys);
        seq->rtp_write(attn_kv_tiles[i], 36864, keys);
    }
}

/// The drafted token's k and v into the DDR cache, two descriptors per column.
///
/// The RoPE tile stamps k and v with different packet ids and they land on
/// different S2MM channels of the column's shim (iron/RoPE.py, and
/// iron/MTP_layer_aie.py for the routing), so each plane gets its own BD:
///
///     k(t) at t * DK        v(t) at MAX_L * DK + t * DV
///
/// Only one token is appended per pass -- T = 1 is the whole point of this
/// design -- so each descriptor is a single contiguous run of the column's two
/// heads. Neither carries a stride at all.
///
/// **This is what removed the MAX_L ceiling.** k and v used to arrive
/// interleaved on one channel, which forced one BD to describe both planes:
/// the column's heads on one axis and "k or v" on another, whose stride was the
/// whole k plane. A shim BD's stride field is 20 bits and bf16 strides are
/// halved on lowering, so MAX_L * DK / 2 <= 1 << 20 capped MAX_L at 2048 -- and
/// it did not fault. npu_instr_utils prints "stride_2 out of range", masks the
/// value into 20 bits and carries on, so every v row landed at a wrapped
/// address and only the *next* pass saw the damage. Splitting the producer
/// removed the hop rather than raising it. What bounds the context now is the
/// replay's chunk count; see _move_kv_cache.
///
/// \param slot the cache row to write, i.e. the drafted token's position
void decoding_layer::_receive_kv_cache(npu_sequence* seq, int slot){
    const uint32_t width = (uint32_t)(this->kv_heads_per_column * this->DH);
    assert(this->v_at(slot) + (size_t)this->DV <= this->kv_cache_size &&
           "cache slot is past the end of the cache");
    assert(width <= 1023 * 2 && "a BD dimension is 10 bits, and bf16 sizes are halved");

    for (int c = 0; c < num_attn_columns; c++) {
        const size_t head_off = (size_t)c * width;
        // k on channel 0, v on channel 1 -- the same two channels the ids are
        // routed to. Each plane's base is a full-width byte offset patched into
        // the descriptor, not a BD field, so it does not overflow.
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_id, S2MM, attn_col_tiles[c], bd_11, it_channel_0,
            {0, 0, 0, (uint32_t)(this->k_at(slot) + head_off)},
            {1, 1, 1, width},
            {0, 0, 0, 1},
            -1, 0, true
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_id, S2MM, attn_col_tiles[c], bd_12, it_channel_1,
            {0, 0, 0, (uint32_t)(this->v_at(slot) + head_off)},
            {1, 1, 1, width},
            {0, 0, 0, 1},
            -1, 0, true
        );
    }
}

/// The whole context out of the DDR cache and into the two attention columns,
/// in chunks of L_CHUNK keys.
///
/// Each column takes the two kv heads it owns, out of each plane's rows:
///
///   dim1 (chunks)   successive L_CHUNKs of context, stride L_CHUNK * row stride
///   dim2 (L_CHUNK)  one chunk's rows, stride one row
///   dim3 (width)    the column's two heads, contiguous
///
/// Each plane is walked on its own descriptor, so the row stride is just DK (or
/// DV) and nothing here depends on MAX_L. Only the base offsets differ --
/// k_at(0) is 0 and v_at(0) is a whole k plane in. The append is symmetric now:
/// it too puts k and v on separate channels, so no descriptor anywhere carries
/// a MAX_L-scaled stride; see _receive_kv_cache.
///
/// **What does bound the context is right here.** dim1 counts whole L_CHUNKs
/// and a BD size field is 10 bits, so one pass replays at most 1024 chunks --
/// 16384 keys. That is the cap the constructor checks MAX_L against, and an
/// oversized 10-bit size masks exactly as quietly as the old stride did.
void decoding_layer::_move_kv_cache(npu_sequence* seq){
    const uint32_t width  = (uint32_t)(this->kv_heads_per_column * this->DH);
    const uint32_t keys   = (uint32_t)(this->current_context_length + 1);
    const uint32_t rows   = (keys + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    const uint32_t chunks = rows / (uint32_t)L_CHUNK;
    const uint32_t k_row_stride   = (uint32_t)this->k_row_stride();
    const uint32_t v_row_stride   = (uint32_t)this->v_row_stride();
    const uint32_t k_chunk_stride = (uint32_t)(L_CHUNK * this->k_row_stride());
    const uint32_t v_chunk_stride = (uint32_t)(L_CHUNK * this->v_row_stride());
    assert(rows <= (uint32_t)MAX_L && "the replay walks past the end of the cache");
    assert(width <= 1023 * 2 && "a BD dimension is 10 bits, and bf16 sizes are halved");
    assert(chunks <= 1024 && "dim1 is 10 bits");

    for (int c = 0; c < num_attn_columns; c++) {
        const uint32_t head_off = (uint32_t)(c * (int)width);
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_id, MM2S, attn_col_tiles[c], bd_0, it_channel_0,
            {0, 0, 0, (uint32_t)this->k_at(0) + head_off},
            {1, chunks,         (uint32_t)L_CHUNK, width},
            {0, k_chunk_stride, k_row_stride,      1},
            -1, 0, false, no_cache
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16), kv_cache_arg_id, MM2S, attn_col_tiles[c], bd_1, it_channel_1,
            {0, 0, 0, (uint32_t)this->v_at(0) + head_off},
            {1, chunks,         (uint32_t)L_CHUNK, width},
            {0, v_chunk_stride, v_row_stride,      1},
            -1, 0, false, no_cache
        );
    }
}

void decoding_layer::_receive_layer_out(npu_sequence* seq){
    const uint32_t outer = 4;
    const uint32_t inner = (uint32_t)this->D / outer;
    assert(this->D % (int)outer == 0 && inner <= 2048 &&
           "a BD dimension is 10 bits, and bf16 sizes are halved");
    seq->npu_dma_memcpy_nd(
        sizeof(bf16), x_arg_id, S2MM, layer_out_tap_tile, bd_10, it_channel_1,
        {0, 0, 0, (uint32_t)this->layer_out_offset},
        {1, 1, outer, inner},
        {0, 0, inner, 1},
        -1, 0, true
    );
}

buffer<bf16> decoding_layer::get_k_cache(int idx){
    assert(idx >= 0 && idx < MAX_L && "cache row is outside the cache");
    this->kv_cache.sync_from_device();
    buffer<bf16> k_cache(DK);
    memcpy(k_cache.data(), this->kv_cache.data() + this->k_at(idx),
           (size_t)this->DK * sizeof(bf16));
    return k_cache;
}

buffer<bf16> decoding_layer::get_v_cache(int idx){
    assert(idx >= 0 && idx < MAX_L && "cache row is outside the cache");
    this->kv_cache.sync_from_device();
    buffer<bf16> v_cache(DV);
    memcpy(v_cache.data(), this->kv_cache.data() + this->v_at(idx),
           (size_t)this->DV * sizeof(bf16));
    return v_cache;
}
