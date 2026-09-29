#include "flm_override.hpp"
#include "gemma4e_prefill.hpp"
#include "metrices.hpp"

// ---------------------------------------------------------------------------
// attention block
// ---------------------------------------------------------------------------

void gemma4e_attn_block_prefill_context::_forward_swa(
    const gemma4e_prefill_shape& s,
    gemma4e_layer_type_t type,
    buffer<bf16>& qkv_weights,
    buffer<bf16>& o_weights,
    buffer<bf16>& kv_cache,
    buffer<bf16>& q_norm,
    buffer<bf16>& k_norm,
    SafeTensors* reference
){
    const bool is_skip = is_skip_layer(type);
    const int D = desc->D;
    const int SWA_DQ = desc->SWA_DQ;
    const int SWA_DK = desc->SWA_DK;
    const int SWA_DV = desc->SWA_DV;

    FLM_OVERRIDE(q_swa_proj, q_swa_proj(bufs->q_buffer, bufs->hidden_state_buffer, qkv_weights), this->desc, type, s);
    bufs->q_buffer.sync_from_device();

    DEBUG_BLOCK(2,
        buffer<bf16> q_ref;
        reference->load_weights(q_ref, "q_proj");
        buffer<bf16> q_valid = buffer<bf16>(bufs->q_buffer.data(), s.L_effective * SWA_DQ);
        buffer<bf16> q_valid_ref = buffer<bf16>(q_ref.data(), s.L_effective * SWA_DQ);
        print_error_metrics(get_error_metrics(q_valid, q_valid_ref), "SWA Q Projection Error: ");
    )
    if (is_skip){
        // a skip layer carries no k/v projection: it reuses the cache the last
        // non-skip layer of its kind filled.
        gemma4e_cpu_func::_rope_rms_batch(bufs->q_buffer.data(), SWA_DQ, s.L_offset, s.L_begin, s.L_effective, q_norm.data(), type, desc->get_DH(type));
        bufs->q_buffer.sync_to_device();
    }
    else {
        auto run_k = FLM_OVERRIDE(k_swa_proj,
            k_swa_proj.create_run(bufs->k_buffer, bufs->hidden_state_buffer, qkv_weights), this->desc, type, s);
        run_k.start();

        gemma4e_cpu_func::_rope_rms_batch(bufs->q_buffer.data(), SWA_DQ, s.L_offset, s.L_begin, s.L_effective, q_norm.data(), type, desc->get_DH(type));
        bufs->q_buffer.sync_to_device();

        run_k.wait();
        bufs->k_buffer.sync_from_device();
        auto run_v = FLM_OVERRIDE(v_swa_proj,
            v_swa_proj.create_run(bufs->v_buffer, bufs->hidden_state_buffer, qkv_weights), this->desc, type, s);
        run_v.start();
        gemma4e_cpu_func::_rope_rms_batch(bufs->k_buffer.data(), SWA_DK, s.L_offset, s.L_begin, s.L_effective, k_norm.data(), type, desc->get_DH(type));
        bufs->k_buffer.sync_to_device();
        run_v.wait();

        bufs->v_buffer.sync_from_device();
        gemma4e_cpu_func::_rms_norm_batch(bufs->v_buffer.data(), bufs->v_buffer.data(), nullptr, SWA_DV, SWA_DV, s.L_effective, s.L_offset, s.L_offset);
        bufs->v_buffer.sync_to_device();
    }

    DEBUG_BLOCK(2,
        buffer<bf16> q_ref;
        reference->load_weights(q_ref, "q_embed");
        buffer<bf16> q_valid = buffer<bf16>(bufs->q_buffer.data(), s.L_effective * SWA_DQ);
        buffer<bf16> q_valid_ref = buffer<bf16>(q_ref.data(), s.L_effective * SWA_DQ);
        print_error_metrics(get_error_metrics(q_valid, q_valid_ref), "SWA Q Projection Error: ");
        if (!is_skip){
            buffer<bf16> k_ref;
            reference->load_weights(k_ref, "k_embed");
            buffer<bf16> k_valid = buffer<bf16>(bufs->k_buffer.data(), s.L_effective * SWA_DK);
            buffer<bf16> k_valid_ref = buffer<bf16>(k_ref.data(), s.L_effective * SWA_DK);
            print_error_metrics(get_error_metrics(k_valid, k_valid_ref), "SWA K Projection Error: ");

            buffer<bf16> v_ref;
            reference->load_weights(v_ref, "v_norm");
            buffer<bf16> v_valid = buffer<bf16>(bufs->v_buffer.data(), s.L_effective * SWA_DV);
            buffer<bf16> v_valid_ref = buffer<bf16>(v_ref.data(), s.L_effective * SWA_DV);
            print_error_metrics(get_error_metrics(v_valid, v_valid_ref), "SWA V Projection Error: ");
        }
    )
    if (!is_skip){
        sync_sliding_kv_cache(bufs->k_buffer, bufs->v_buffer, kv_cache_sliding_prefill, kv_cache, s.L_offset, s.L_begin, s.L_effective, s.sliding_l_begin, s.L_end_chunked, SWA_DK, SWA_DV);
    }
    else{
        kv_cache_sliding_prefill.sync_from_device();
        kv_cache_sliding_prefill.sync_to_device();
    }
    FLM_OVERRIDE(swa_attn_core,
        this->swa_engine(bufs->attn_out_buffer, bufs->q_buffer, kv_cache_sliding_prefill), type, s, this->MAX_L);
    bufs->attn_out_buffer.sync_from_device();
    DEBUG_BLOCK(2,
        buffer<bf16> attn_out_ref;
        reference->load_weights(attn_out_ref, "attention_output");
        buffer<bf16> attn_out_valid = buffer<bf16>(bufs->attn_out_buffer.data(), s.L_effective * SWA_DQ);
        buffer<bf16> attn_out_valid_ref = buffer<bf16>(attn_out_ref.data(), s.L_effective * SWA_DQ);
        print_error_metrics(get_error_metrics(attn_out_valid, attn_out_valid_ref), "SWA Attention Output Error: ");
    )

    FLM_OVERRIDE(o_swa_proj,
        this->o_swa_proj(bufs->hidden_state_buffer, bufs->attn_out_buffer, o_weights), this->desc, type, s);
    bufs->hidden_state_buffer.sync_from_device();

    DEBUG_BLOCK(2,
        buffer<bf16> o_proj_ref;
        reference->load_weights(o_proj_ref, "after_o_proj");
        buffer<bf16> o_proj_valid = buffer<bf16>(bufs->hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> o_proj_valid_ref = buffer<bf16>(o_proj_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(o_proj_valid, o_proj_valid_ref), "SWA O Projection Error: ");
    )
}

void gemma4e_attn_block_prefill_context::_forward_global(
    const gemma4e_prefill_shape& s,
    gemma4e_layer_type_t type,
    buffer<bf16>& qkv_weights,
    buffer<bf16>& o_weights,
    buffer<bf16>& kv_cache,
    buffer<bf16>& q_norm,
    buffer<bf16>& k_norm,
    SafeTensors* reference
){
    const bool is_skip = is_skip_layer(type);
    const int D = desc->D;
    const int DQ = desc->DQ;
    const int DK = desc->DK;
    const int DV = desc->DV;

    FLM_OVERRIDE(q_global_proj, q_global_proj(bufs->q_buffer, bufs->hidden_state_buffer, qkv_weights), this->desc, type, s);
    bufs->q_buffer.sync_from_device();

    DEBUG_BLOCK(2,
        buffer<bf16> q_ref;
        reference->load_weights(q_ref, "q_proj");
        buffer<bf16> q_valid = buffer<bf16>(bufs->q_buffer.data(), s.L_effective * DQ);
        buffer<bf16> q_valid_ref = buffer<bf16>(q_ref.data(), s.L_effective * DQ);
        print_error_metrics(get_error_metrics(q_valid, q_valid_ref), "Q Projection Error: ");
    )
    if (is_skip){
        gemma4e_cpu_func::_rope_rms_batch(bufs->q_buffer.data(), DQ, s.L_offset, s.L_begin, s.L_effective, q_norm.data(), type, desc->get_DH(type));
        bufs->q_buffer.sync_to_device();
    }
    else {
        auto run_k = FLM_OVERRIDE(k_global_proj,
            k_global_proj.create_run(bufs->k_buffer, bufs->hidden_state_buffer, qkv_weights), this->desc, type, s);
        run_k.start();
        gemma4e_cpu_func::_rope_rms_batch(bufs->q_buffer.data(), DQ, s.L_offset, s.L_begin, s.L_effective, q_norm.data(), type, desc->get_DH(type));
        bufs->q_buffer.sync_to_device();
        run_k.wait();
        bufs->k_buffer.sync_from_device();

        auto run_v = FLM_OVERRIDE(v_global_proj,
            v_global_proj.create_run(bufs->v_buffer, bufs->hidden_state_buffer, qkv_weights), this->desc, type, s);
        run_v.start();
        gemma4e_cpu_func::_rope_rms_batch(bufs->k_buffer.data(), DK, s.L_offset, s.L_begin, s.L_effective, k_norm.data(), type, desc->get_DH(type));
        bufs->k_buffer.sync_to_device();
        run_v.wait();

        bufs->v_buffer.sync_from_device();
        gemma4e_cpu_func::_rms_norm_batch(bufs->v_buffer.data(), bufs->v_buffer.data(), nullptr, DV, DV, s.L_effective, s.L_offset, s.L_offset);
        bufs->v_buffer.sync_to_device();
    }

    DEBUG_BLOCK(2,
        buffer<bf16> q_ref;
        reference->load_weights(q_ref, "q_embed");
        buffer<bf16> q_valid = buffer<bf16>(bufs->q_buffer.data(), s.L_effective * DQ);
        buffer<bf16> q_valid_ref = buffer<bf16>(q_ref.data(), s.L_effective * DQ);
        print_error_metrics(get_error_metrics(q_valid, q_valid_ref), "Q Projection Error: ");

        buffer<bf16> k_ref;
        reference->load_weights(k_ref, "k_embed");
        buffer<bf16> k_valid = buffer<bf16>(bufs->k_buffer.data(), s.L_effective * DK);
        buffer<bf16> k_valid_ref = buffer<bf16>(k_ref.data(), s.L_effective * DK);
        print_error_metrics(get_error_metrics(k_valid, k_valid_ref), "K Projection Error: ");

        buffer<bf16> v_ref;
        reference->load_weights(v_ref, "v_norm");
        buffer<bf16> v_valid = buffer<bf16>(bufs->v_buffer.data(), s.L_effective * DV);
        buffer<bf16> v_valid_ref = buffer<bf16>(v_ref.data(), s.L_effective * DV);
        print_error_metrics(get_error_metrics(v_valid, v_valid_ref), "V Projection Error: ");
    )
    if (!is_skip){
        sync_kv_cache(bufs->k_buffer, bufs->v_buffer, kv_cache_global_prefill, kv_cache, s.L_offset, s.L_begin, s.L_effective, DK, DV);
    }
    else {
        kv_cache_global_prefill.sync_from_device();
        kv_cache_global_prefill.sync_to_device();
    }
    FLM_OVERRIDE(global_attn_core,
        this->mha_engine(bufs->attn_out_buffer, bufs->q_buffer, kv_cache_global_prefill), type, s, this->MAX_L);
    DEBUG_BLOCK(2,
        header_print("info", "Attn done!");
    )
    bufs->attn_out_buffer.sync_from_device();
    DEBUG_BLOCK(2,
        buffer<bf16> attn_out_ref;
        reference->load_weights(attn_out_ref, "attention_output");
        buffer<bf16> attn_out_valid = buffer<bf16>(bufs->attn_out_buffer.data(), s.L_effective * DQ);
        buffer<bf16> attn_out_valid_ref = buffer<bf16>(attn_out_ref.data(), s.L_effective * DQ);
        print_error_metrics(get_error_metrics(attn_out_valid, attn_out_valid_ref), "Attention Output Error: ");
    )

    FLM_OVERRIDE(o_global_proj,
        this->o_global_proj(bufs->hidden_state_buffer, bufs->attn_out_buffer, o_weights), this->desc, type, s);
    bufs->hidden_state_buffer.sync_from_device();

    DEBUG_BLOCK(2,
        buffer<bf16> o_proj_ref;
        reference->load_weights(o_proj_ref, "after_o_proj");
        buffer<bf16> o_proj_valid = buffer<bf16>(bufs->hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> o_proj_valid_ref = buffer<bf16>(o_proj_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(o_proj_valid, o_proj_valid_ref), "O Projection Error: ");
    )
}

void gemma4e_attn_block_prefill_context::sync_kv_cache(
    buffer<bf16>& buffer_k,
    buffer<bf16>& buffer_v,
    buffer<bf16>& prefill_cache,
    buffer<bf16>& decoding_cache,
    int L_offset,
    int L_begin,
    int L_effective,
    int DK, int DV
){
    bf16* prefill_k_cache_ptr = prefill_cache.data();
    bf16* prefill_v_cache_ptr = prefill_cache.data() + (size_t)MAX_L * DK;
    bf16* decoding_k_cache_ptr = decoding_cache.data();
    bf16* decoding_v_cache_ptr = decoding_cache.data() + (size_t)MAX_L * DK;
    // copy the existing cache up to L_begin from decoding_cache to prefill_cache, and then copy the new k cache from buffer_k and v cache from buffer_v to prefill_cache at the appropriate location, then sync the prefill_cache to device for the attention run
    decoding_cache.sync_from_device();
    prefill_cache.sync_from_device();
    memcpy(prefill_k_cache_ptr, decoding_k_cache_ptr, L_begin * DK * sizeof(bf16)); // copy the existing cache up to L_begin
    memcpy(prefill_v_cache_ptr, decoding_v_cache_ptr, L_begin * DV * sizeof(bf16)); // copy the existing cache up to L_begin

    prefill_k_cache_ptr += L_begin * DK;
    prefill_v_cache_ptr += L_begin * DV;
    decoding_k_cache_ptr += L_begin * DK;
    decoding_v_cache_ptr += L_begin * DV;

    // copy the new k cache and v cache to the appropriate location in prefill_cache and decoding_cache
    bf16* new_k_cache_ptr = buffer_k.data() + L_offset * DK;
    bf16* new_v_cache_ptr = buffer_v.data() + L_offset * DV;

    memcpy(prefill_k_cache_ptr, new_k_cache_ptr, L_effective * DK * sizeof(bf16));
    memcpy(prefill_v_cache_ptr, new_v_cache_ptr, L_effective * DV * sizeof(bf16));
    memcpy(decoding_k_cache_ptr, new_k_cache_ptr, L_effective * DK * sizeof(bf16));
    memcpy(decoding_v_cache_ptr, new_v_cache_ptr, L_effective * DV * sizeof(bf16));

    decoding_cache.sync_to_device();
    prefill_cache.sync_to_device();
}

void gemma4e_attn_block_prefill_context::sync_sliding_kv_cache(
    buffer<bf16>& buffer_k,
    buffer<bf16>& buffer_v,
    buffer<bf16>& prefill_cache,
    buffer<bf16>& decoding_cache,
    int L_offset,
    int L_begin,
    int L_effective,
    int sliding_l_begin,
    int L_end_chunked,
    int DK, int DV
){
    const int SLIDING_LENGTH = desc->SLIDING_LENGTH;
    auto linear2ring = [sliding = SLIDING_LENGTH](int l) { return l % sliding; };
    int in_memory = (L_begin - SLIDING_LENGTH) > 0 ? L_begin - SLIDING_LENGTH : 0;
    int l_idx = in_memory % SLIDING_LENGTH;
    int local_offset = in_memory - sliding_l_begin;
    int local_length = L_end_chunked - sliding_l_begin;
    DEBUG_BLOCK(2,
        std::cout << "DEBUG: L_begin: " << L_begin << ", L_end_chunked: " << L_end_chunked << ", sliding_l_begin: " << sliding_l_begin << std::endl;
        std::cout << "DEBUG: in_memory: " << in_memory << std::endl;
        std::cout << "DEBUG: local_offset: " << local_offset << ", local_length: " << local_length << std::endl;
    )
    // | sliding_l_begin -> in_memory | in_memory -> L_begin | L_begin -> L_end | L_end -> L_end_chunked |
    // | Part 1                       | Part 2               | Part 3           | Part 4                 |

    int part_1_length = in_memory - sliding_l_begin;
    int part_2_length = L_begin - in_memory;
    int part_3_length = L_effective;
    int part_4_length = L_end_chunked - L_begin - L_effective;

    bf16* prefill_k_cache_ptr = prefill_cache.data() + sliding_l_begin * DK;
    bf16* prefill_v_cache_ptr = prefill_cache.data() + sliding_l_begin * DV + (size_t)MAX_L * DK;
    bf16* decoding_k_cache_ptr = decoding_cache.data();
    bf16* decoding_v_cache_ptr = decoding_cache.data() + SLIDING_LENGTH * DK;

    // copy the existing cache up to L_begin from decoding_cache to prefill_cache, and then copy the new k cache from buffer_k and v cache from buffer_v to prefill_cache at the appropriate location, then sync the prefill_cache to device for the attention run
    decoding_cache.sync_from_device();
    prefill_cache.sync_from_device();

    // part 1: padded, no actual data exists
    if (part_1_length > 0){
        memset(prefill_k_cache_ptr, 0, part_1_length * DK * sizeof(bf16));
        memset(prefill_v_cache_ptr, 0, part_1_length * DV * sizeof(bf16));
        prefill_k_cache_ptr += part_1_length * DK;
        prefill_v_cache_ptr += part_1_length * DV;
    }

    // part 2: copy from the existing cache in decoding_cache, but we need to copy in the order of the ring buffer
    for (int i = 0; i < part_2_length; i++){
        int ring_idx = linear2ring(in_memory + i);
        memcpy(prefill_k_cache_ptr, decoding_k_cache_ptr + ring_idx * DK, DK * sizeof(bf16));
        memcpy(prefill_v_cache_ptr, decoding_v_cache_ptr + ring_idx * DV, DV * sizeof(bf16));
        prefill_k_cache_ptr += DK;
        prefill_v_cache_ptr += DV;
    }

    // copy the new k cache and v cache to the appropriate location in prefill_cache and decoding_cache
    bf16* new_k_cache_ptr = buffer_k.data() + L_offset * DK;
    bf16* new_v_cache_ptr = buffer_v.data() + L_offset * DV;
    //part 3: copy the new k cache and v cache from buffer_k and buffer_v to prefill_cache and decoding_cache, but we also need to copy in the order of the ring buffer
    for (int i = 0; i < part_3_length; i++){
        int ring_idx = linear2ring(L_begin + i);
        memcpy(prefill_k_cache_ptr, new_k_cache_ptr + i * DK, DK * sizeof(bf16));
        memcpy(prefill_v_cache_ptr, new_v_cache_ptr + i * DV, DV * sizeof(bf16));
        memcpy(decoding_k_cache_ptr + ring_idx * DK, new_k_cache_ptr + i * DK, DK * sizeof(bf16));
        memcpy(decoding_v_cache_ptr + ring_idx * DV, new_v_cache_ptr + i * DV, DV * sizeof(bf16));
        prefill_k_cache_ptr += DK;
        prefill_v_cache_ptr += DV;
    }

    // // part 4: padded, no actual data exists
    // memset(prefill_k_cache_ptr, 0, part_4_length * DK * sizeof(bf16));
    // memset(prefill_v_cache_ptr, 0, part_4_length * DV * sizeof(bf16));
    decoding_cache.sync_to_device();
    prefill_cache.sync_to_device();
}

// ---------------------------------------------------------------------------
// mlp block
// ---------------------------------------------------------------------------

void gemma4e_mlp_prefill_context::forward(
    const gemma4e_prefill_shape& s,
    bool double_wide,
    buffer<bf16>& gate_weights,
    buffer<bf16>& up_weights,
    buffer<bf16>& down_weights,
    SafeTensors* reference
){
    // a double-wide layer runs the same three gemms over twice the mlp width
    npu_app& gate = double_wide ? this->gate_skip_proj : this->gate_proj;
    npu_app& up   = double_wide ? this->up_skip_proj   : this->up_proj;
    npu_app& down = double_wide ? this->down_skip_proj : this->down_proj;
    const int I = double_wide ? desc->INTERMEDIATE_SIZE * 2 : desc->INTERMEDIATE_SIZE;

    bufs->hidden_state_buffer.sync_to_device();
    FLM_OVERRIDE(gate_proj, gate(bufs->gate_buffer, bufs->hidden_state_buffer, gate_weights), this->desc, double_wide, s);
    bufs->gate_buffer.sync_from_device();
    FLM_OVERRIDE(up_proj, up(bufs->up_buffer, bufs->hidden_state_buffer, up_weights), this->desc, double_wide, s);
    bufs->up_buffer.sync_from_device();

    gemma4e_cpu_func::_elementwise_mul_batch(bufs->hid_buffer.data(), bufs->gate_buffer.data(), bufs->up_buffer.data(), I, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    bufs->hid_buffer.sync_to_device();

    FLM_OVERRIDE(down_proj, down(bufs->hidden_state_buffer, bufs->hid_buffer, down_weights), this->desc, double_wide, s);
    bufs->hidden_state_buffer.sync_from_device();

    DEBUG_BLOCK(2,
        buffer<bf16> down_ref;
        reference->load_weights(down_ref, "mlp_output");
        buffer<bf16> down_valid = buffer<bf16>(bufs->hidden_state_buffer.data(), s.L_effective * desc->D);
        buffer<bf16> down_valid_ref = buffer<bf16>(down_ref.data(), s.L_effective * desc->D);
        print_error_metrics(get_error_metrics(down_valid, down_valid_ref), "MLP Error: ");
    )
}

// ---------------------------------------------------------------------------
// per layer input path
// ---------------------------------------------------------------------------

void gemma4e_pli_prefill_context::setup(const gemma4e_prefill_shape& s){
    if (L_padded_512_old == s.L_padded_512) {
        return;
    }
    const int D = desc->D;
    const int PLI_D = desc->PLI_D;
    const int num_hidden_layers = desc->num_hidden_layers;
    Gemma4e_ImageEncoder* enc = this->image_encoder;

    generate_mm_sequence<bf16, bf16>(*this->pli_down_proj.seq(),
            s.L_padded_512, D, num_hidden_layers * PLI_D,
            enc->MM_tile_M, enc->MM_tile_K, enc->MM_tile_N,
            8,8,8,
            enc->rtp_address, enc->rtp_sync_lock_id,
            enc->MM_ROW_SIZE, enc->MM_COL_SIZE,
            0,0,0,
            enc->IS_B_ROW_MAJOR, enc->ENABLE_AXI4, true,
            false, 0,// no activation
            0,  -10000.0, 1000000.0, // do not clamp on output
            false, 0 // no need to reorder it
    );

    generate_mm_sequence<bf16, bf16>(*this->pli_gate_proj.seq(),
            s.L_padded_512, D, PLI_D,
            enc->MM_tile_M, enc->MM_tile_K, enc->MM_tile_N,
            8,8,8,
            enc->rtp_address, enc->rtp_sync_lock_id,
            enc->MM_ROW_SIZE, enc->MM_COL_SIZE,
            0,0,0,
            enc->IS_B_ROW_MAJOR, enc->ENABLE_AXI4, true,
            false, 1,// no activation
            0,  -10000.0, 1000000.0, // do not clamp on output
            false, 0 // no need to reorder it
    );

    generate_mm_sequence<bf16, bf16>(*this->pli_up_proj.seq(),
            s.L_padded_512, PLI_D, D,
            enc->MM_tile_M, enc->MM_tile_K, enc->MM_tile_N,
            8,8,8,
            enc->rtp_address, enc->rtp_sync_lock_id,
            enc->MM_ROW_SIZE, enc->MM_COL_SIZE,
            0,PLI_D * D,0,
            enc->IS_B_ROW_MAJOR, enc->ENABLE_AXI4, true,
            false, 0,// no activation
            0,  -10000.0, 1000000.0, // do not clamp on output
            false, 0 // no need to reorder it
    );
    L_padded_512_old = s.L_padded_512;
}

void gemma4e_pli_prefill_context::pre_pass(
    const gemma4e_prefill_shape& s,
    buffer<bf16>& pli_down_weights,
    buffer<bf16>& pli_input_norm
){
    const int D = desc->D;
    const int PLI_D = desc->PLI_D;
    const int num_hidden_layers = desc->num_hidden_layers;

    // the down projection reads the token embeddings, which start life in the residual
    memcpy(bufs->hidden_state_buffer.data() + (size_t)s.L_offset * D, bufs->residual_buffer.data() + (size_t)s.L_offset * D, (size_t)s.L_effective * D * sizeof(bf16));
    bufs->hidden_state_buffer.sync_to_device();
    FLM_OVERRIDE(pli_down_proj, this->pli_down_proj(bufs->hidden_state_buffer, pli_down_weights, bufs->pli_down_buffer), this->desc, s);
    bufs->pli_down_buffer.sync_from_device();

    gemma4e_cpu_func::_elementwise_scale_batch(bufs->pli_down_buffer.data(), 1.0 / sqrtf((float)D), PLI_D * num_hidden_layers, s.L_effective, s.L_offset);
    gemma4e_cpu_func::_rms_norm_batch(bufs->pli_down_buffer.data(), bufs->pli_down_buffer.data(), pli_input_norm.data(), PLI_D, num_hidden_layers * PLI_D, s.L_effective, s.L_offset, s.L_offset);
    gemma4e_cpu_func::_residual_add_batch(bufs->pli_embed_buffer.data(), bufs->pli_down_buffer.data(), bufs->pli_embed_buffer.data(), num_hidden_layers * PLI_D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    gemma4e_cpu_func::_elementwise_scale_batch(bufs->pli_embed_buffer.data(), 1.0 / sqrtf((float)2.0), PLI_D * num_hidden_layers, s.L_effective, s.L_offset);
}

void gemma4e_pli_prefill_context::layer_pass(
    const gemma4e_prefill_shape& s,
    int layer_idx,
    buffer<bf16>& pli_gate_up_weights,
    buffer<bf16>& pli_final_norm
){
    const int D = desc->D;
    const int PLI_D = desc->PLI_D;
    const int num_hidden_layers = desc->num_hidden_layers;

    FLM_OVERRIDE(pli_gate_proj, this->pli_gate_proj(bufs->hidden_state_buffer, pli_gate_up_weights, bufs->pli_gate_buffer), this->desc, layer_idx, s);
    bufs->pli_gate_buffer.sync_from_device();

    // this layer's slice of the per-layer embeddings gates the projection
    gemma4e_cpu_func::_elementwise_mul_batch(bufs->pli_hid_buffer.data(), bufs->pli_gate_buffer.data(), bufs->pli_embed_buffer.data() + (size_t)layer_idx * PLI_D, PLI_D, num_hidden_layers * PLI_D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    bufs->pli_hid_buffer.sync_to_device();
    FLM_OVERRIDE(pli_up_proj, this->pli_up_proj(bufs->pli_hid_buffer, pli_gate_up_weights, bufs->hidden_state_buffer), this->desc, layer_idx, s);
    bufs->hidden_state_buffer.sync_from_device();

    gemma4e_cpu_func::_rms_norm_batch(bufs->hidden_state_buffer.data(), bufs->hidden_state_buffer.data(), pli_final_norm.data(), D, D, s.L_effective, s.L_offset, s.L_offset);
    bufs->hidden_state_buffer.sync_to_device();
}

// ---------------------------------------------------------------------------
// one decoder layer
// ---------------------------------------------------------------------------

void gemma4e_prefill_context::forward(
    int layer_idx,
    gemma4e_layer_type_t type,
    const gemma4e_prefill_shape& s,
    buffer<u8>& proj_weights,
    buffer<bf16>& rms_weights,
    buffer<bf16>& rope_rms_weights,
    buffer<bf16>& pli_gate_up_weights,
    buffer<bf16>& kv_cache,
    float layer_scale,
    SafeTensors* reference
){
    const int D = desc->D;
    const bool is_skip = is_skip_layer(type);

    FLM_OVERRIDE(prefill_layer_begin, (void)0, this->desc, layer_idx, type, s);

    this->dequant_block->run(type, proj_weights);

    DEBUG_BLOCK(1,
    header_print_r("info", "Running layer " + std::to_string(layer_idx) + " of type " + std::to_string(type));
    )

    // every norm weight of this layer, located through the descriptor
    buffer<bf16> input_layernorm_weight(rms_weights.data(), D);
    buffer<bf16> post_attention_layernorm_weight(rms_weights.data() + D, D);
    buffer<bf16> pre_feedforward_layernorm_weight(rms_weights.data() + D * 2, D);
    buffer<bf16> post_feedforward_layernorm_weight(rms_weights.data() + D * 3, D);
    buffer<bf16> pli_final_norm(rope_rms_weights.data() + desc->get_post_pli_norm_offset(type), D);
    buffer<bf16> q_norm(rope_rms_weights.data() + desc->get_q_norm_offset(type), desc->get_DH(type));
    buffer<bf16> k_norm(rope_rms_weights.data() + desc->get_k_norm_offset(type), desc->get_DH(type));

    // input layernorm
    gemma4e_cpu_func::_rms_norm_batch(bufs.hidden_state_buffer.data(), bufs.residual_buffer.data(), input_layernorm_weight.data(), D, D, s.L_effective, s.L_offset, s.L_offset);
    bufs.hidden_state_buffer.sync_to_device();

    DEBUG_BLOCK(2,
        buffer<bf16> norm_ref;
        reference->load_weights(norm_ref, "input_layernorm_output");
        buffer<bf16> norm_valid = buffer<bf16>(bufs.hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> norm_valid_ref = buffer<bf16>(norm_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(norm_valid, norm_valid_ref), "Input LayerNorm Error: ");
    )

    this->attn_block->forward(s, type,
        this->dequant_block->qkv_weights, this->dequant_block->o_weights,
        kv_cache, q_norm, k_norm, reference);

    gemma4e_cpu_func::_rms_norm_batch(bufs.hidden_state_buffer.data(), bufs.hidden_state_buffer.data(), post_attention_layernorm_weight.data(), D, D, s.L_effective, s.L_offset, s.L_offset);
    bufs.attn_out_buffer.sync_to_device();

    DEBUG_BLOCK(2,
        buffer<bf16> post_attn_norm_ref;
        reference->load_weights(post_attn_norm_ref, "post_attention_norm_output");
        buffer<bf16> post_attn_norm_valid = buffer<bf16>(bufs.hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> post_attn_norm_valid_ref = buffer<bf16>(post_attn_norm_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(post_attn_norm_valid, post_attn_norm_valid_ref), "Post Attention LayerNorm Error: ");
    )

    gemma4e_cpu_func::_residual_add_batch(bufs.residual_buffer.data(), bufs.hidden_state_buffer.data(), bufs.residual_buffer.data(), D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    gemma4e_cpu_func::_rms_norm_batch(bufs.hidden_state_buffer.data(), bufs.residual_buffer.data(), pre_feedforward_layernorm_weight.data(), D, D, s.L_effective, s.L_offset, s.L_offset);
    bufs.hidden_state_buffer.sync_to_device();

    DEBUG_BLOCK(2,
        buffer<bf16> pre_ffn_norm_ref;
        reference->load_weights(pre_ffn_norm_ref, "pre_ffn_norm_output");
        buffer<bf16> pre_ffn_norm_valid = buffer<bf16>(bufs.hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> pre_ffn_norm_valid_ref = buffer<bf16>(pre_ffn_norm_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(pre_ffn_norm_valid, pre_ffn_norm_valid_ref), "Pre-FFN LayerNorm Error: ");
    )

    this->mlp->forward(s, is_skip && desc->enable_double_wide_mlp,
        this->dequant_block->gate_weights, this->dequant_block->up_weights,
        this->dequant_block->down_weights, reference);

    gemma4e_cpu_func::_rms_norm_batch(bufs.hidden_state_buffer.data(), bufs.hidden_state_buffer.data(), post_feedforward_layernorm_weight.data(), D, D, s.L_effective, s.L_offset, s.L_offset);
    bufs.hidden_state_buffer.sync_to_device();

    DEBUG_BLOCK(2,
        buffer<bf16> post_ffn_norm_ref;
        reference->load_weights(post_ffn_norm_ref, "post_ffn_norm_output");
        buffer<bf16> post_ffn_norm_valid = buffer<bf16>(bufs.hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> post_ffn_norm_valid_ref = buffer<bf16>(post_ffn_norm_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(post_ffn_norm_valid, post_ffn_norm_valid_ref), "Post-FFN LayerNorm Error: ");
    )

    gemma4e_cpu_func::_residual_add_batch(bufs.residual_buffer.data(), bufs.hidden_state_buffer.data(), bufs.residual_buffer.data(), D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    // per layer input: the gate reads the layer output, which lives in the residual
    memcpy(bufs.hidden_state_buffer.data() + (size_t)s.L_offset * D, bufs.residual_buffer.data() + (size_t)s.L_offset * D, (size_t)s.L_effective * D * sizeof(bf16));
    bufs.hidden_state_buffer.sync_to_device();

    this->pli->layer_pass(s, layer_idx, pli_gate_up_weights, pli_final_norm);

    gemma4e_cpu_func::_residual_add_batch(bufs.residual_buffer.data(), bufs.hidden_state_buffer.data(), bufs.residual_buffer.data(), D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    gemma4e_cpu_func::_elementwise_scale_batch(bufs.residual_buffer.data(), layer_scale, D, s.L_effective, s.L_offset);

    DEBUG_BLOCK(2,
        buffer<bf16> output_ref;
        reference->load_weights(output_ref, "layer_" + std::to_string(layer_idx));
        buffer<bf16> output_valid = buffer<bf16>(bufs.residual_buffer.data(), s.L_effective * D);
        buffer<bf16> output_valid_ref = buffer<bf16>(output_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(output_valid, output_valid_ref), "Main path Error: ");
    )
}
