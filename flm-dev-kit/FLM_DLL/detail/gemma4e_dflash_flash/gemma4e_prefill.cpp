// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "flm_override.hpp"
#include "gemma4e_prefill.hpp"
#include "metrices.hpp"


// ---------------------------------------------------------------------------
// attention block
// ---------------------------------------------------------------------------

void gemma4e_attn_block_prefill_context::_forward_swa(
    const gemma4e_prefill_shape& s,
    gemma4e_layer_type_t type,
    buffer<u8>& proj_weights,
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

    PREFILL_PROF_INIT();
    q_proj[type](bufs->q_buffer, bufs->hidden_state_buffer, proj_weights);
    bufs->q_buffer.sync_from_device();
    PREFILL_PROF_TICK("attn:q_proj");

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
        auto run_k = k_proj[type].create_run(bufs->k_buffer, bufs->hidden_state_buffer, proj_weights);
        run_k.start();

        gemma4e_cpu_func::_rope_rms_batch(bufs->q_buffer.data(), SWA_DQ, s.L_offset, s.L_begin, s.L_effective, q_norm.data(), type, desc->get_DH(type));
        bufs->q_buffer.sync_to_device();

        flm::npu::wait_for_completion(run_k, "run_k");
        bufs->k_buffer.sync_from_device();
        auto run_v = v_proj[type].create_run(bufs->v_buffer, bufs->hidden_state_buffer, proj_weights);
        run_v.start();
        gemma4e_cpu_func::_rope_rms_batch(bufs->k_buffer.data(), SWA_DK, s.L_offset, s.L_begin, s.L_effective, k_norm.data(), type, desc->get_DH(type));
        bufs->k_buffer.sync_to_device();
        flm::npu::wait_for_completion(run_v, "run_v");

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
        sync_sliding_kv_cache(bufs->k_buffer, bufs->v_buffer, kv_cache_sliding_prefill, kv_cache, s.L_offset, s.L_begin, s.L_effective, s.sliding_l_begin, s.L_end_chunked, SWA_DK, SWA_DV, s.swa_base_old, s.swa_base_new, s.swa_zero_hi);
    }
    else{
        kv_cache_sliding_prefill.sync_from_device();
        kv_cache_sliding_prefill.sync_to_device();
    }
    PREFILL_PROF_TICK("attn:kv_proj+rope+cache_sync");
    this->swa_engine(bufs->attn_out_buffer, bufs->q_buffer, kv_cache_sliding_prefill);
    bufs->attn_out_buffer.sync_from_device();
    PREFILL_PROF_TICK("attn:swa_engine");
    DEBUG_BLOCK(2,
        buffer<bf16> attn_out_ref;
        reference->load_weights(attn_out_ref, "attention_output");
        buffer<bf16> attn_out_valid = buffer<bf16>(bufs->attn_out_buffer.data(), s.L_effective * SWA_DQ);
        buffer<bf16> attn_out_valid_ref = buffer<bf16>(attn_out_ref.data(), s.L_effective * SWA_DQ);
        print_error_metrics(get_error_metrics(attn_out_valid, attn_out_valid_ref), "SWA Attention Output Error: ");
    )

    this->o_proj[type](bufs->hidden_state_buffer, bufs->attn_out_buffer, proj_weights);
    bufs->hidden_state_buffer.sync_from_device();
    PREFILL_PROF_TICK("attn:o_proj");

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
    buffer<u8>& proj_weights,
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

    PREFILL_PROF_INIT();
    q_proj[type](bufs->q_buffer, bufs->hidden_state_buffer, proj_weights);
    bufs->q_buffer.sync_from_device();
    PREFILL_PROF_TICK("attn:q_proj");

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
        auto run_k = k_proj[type].create_run(bufs->k_buffer, bufs->hidden_state_buffer, proj_weights);
        run_k.start();
        gemma4e_cpu_func::_rope_rms_batch(bufs->q_buffer.data(), DQ, s.L_offset, s.L_begin, s.L_effective, q_norm.data(), type, desc->get_DH(type));
        bufs->q_buffer.sync_to_device();
        flm::npu::wait_for_completion(run_k, "run_k");
        bufs->k_buffer.sync_from_device();

        auto run_v = v_proj[type].create_run(bufs->v_buffer, bufs->hidden_state_buffer, proj_weights);
        run_v.start();
        gemma4e_cpu_func::_rope_rms_batch(bufs->k_buffer.data(), DK, s.L_offset, s.L_begin, s.L_effective, k_norm.data(), type, desc->get_DH(type));
        bufs->k_buffer.sync_to_device();
        flm::npu::wait_for_completion(run_v, "run_v");

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
    PREFILL_PROF_TICK("attn:kv_proj+rope+cache_sync");

    this->mha_engine(bufs->attn_out_buffer, bufs->q_buffer, kv_cache_global_prefill);
    DEBUG_BLOCK(2,
        header_print("info", "Attn done!");
    )
    bufs->attn_out_buffer.sync_from_device();
    PREFILL_PROF_TICK("attn:mha_engine");
    DEBUG_BLOCK(2,
        buffer<bf16> attn_out_ref;
        reference->load_weights(attn_out_ref, "attention_output");
        buffer<bf16> attn_out_valid = buffer<bf16>(bufs->attn_out_buffer.data(), s.L_effective * DQ);
        buffer<bf16> attn_out_valid_ref = buffer<bf16>(attn_out_ref.data(), s.L_effective * DQ);
        print_error_metrics(get_error_metrics(attn_out_valid, attn_out_valid_ref), "Attention Output Error: ");
    )

    this->o_proj[type](bufs->hidden_state_buffer, bufs->attn_out_buffer, proj_weights);
    bufs->hidden_state_buffer.sync_from_device();
    PREFILL_PROF_TICK("attn:o_proj");

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
    memcpy(prefill_k_cache_ptr, decoding_k_cache_ptr, (size_t)L_begin * DK * sizeof(bf16)); // copy the existing cache up to L_begin
    memcpy(prefill_v_cache_ptr, decoding_v_cache_ptr, (size_t)L_begin * DV * sizeof(bf16)); // copy the existing cache up to L_begin

    prefill_k_cache_ptr += L_begin * DK;
    prefill_v_cache_ptr += L_begin * DV;
    decoding_k_cache_ptr += L_begin * DK;
    decoding_v_cache_ptr += L_begin * DV;

    // copy the new k cache and v cache to the appropriate location in prefill_cache and decoding_cache
    bf16* new_k_cache_ptr = buffer_k.data() + L_offset * DK;
    bf16* new_v_cache_ptr = buffer_v.data() + L_offset * DV;

    memcpy(prefill_k_cache_ptr, new_k_cache_ptr, (size_t)L_effective * DK * sizeof(bf16));
    memcpy(prefill_v_cache_ptr, new_v_cache_ptr, (size_t)L_effective * DV * sizeof(bf16));
    memcpy(decoding_k_cache_ptr, new_k_cache_ptr, (size_t)L_effective * DK * sizeof(bf16));
    memcpy(decoding_v_cache_ptr, new_v_cache_ptr, (size_t)L_effective * DV * sizeof(bf16));
    global_scratch_rows = std::max(global_scratch_rows, static_cast<size_t>(L_begin + L_effective));

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
    int DK, int DV,
    int swa_base_old, int swa_base_new, int swa_zero_hi
){
    const int SLIDING_LENGTH = desc->SLIDING_LENGTH;
    int in_memory = (L_begin - SLIDING_LENGTH) > 0 ? L_begin - SLIDING_LENGTH : 0;
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

    bf16* prefill_k_cache_ptr = prefill_cache.data() + sliding_l_begin * DK;
    bf16* prefill_v_cache_ptr = prefill_cache.data() + sliding_l_begin * DV + (size_t)MAX_L * DK;
    // the decode buffer is a linear window: the row of position p is p - base
    bf16* decoding_k_cache_ptr = decoding_cache.data();
    bf16* decoding_v_cache_ptr = decoding_cache.data() + (size_t)kSwaBufferRows * DK;

    // copy the existing cache up to L_begin from decoding_cache to prefill_cache, and then copy the new k cache from buffer_k and v cache from buffer_v to prefill_cache at the appropriate location, then sync the prefill_cache to device for the attention run
    decoding_cache.sync_from_device();
    prefill_cache.sync_from_device();

    // part 1: padded, no actual data exists
    if (part_1_length > 0){
        memset(prefill_k_cache_ptr, 0, (size_t)part_1_length * DK * sizeof(bf16));
        memset(prefill_v_cache_ptr, 0, (size_t)part_1_length * DV * sizeof(bf16));
        prefill_k_cache_ptr += part_1_length * DK;
        prefill_v_cache_ptr += part_1_length * DV;
    }

    // part 2: the rows already in the decode buffer (positions in_memory .. L_begin), read at the OLD base, in position order
    if (part_2_length > 0){
        const int first_row = in_memory - swa_base_old;
        memcpy(prefill_k_cache_ptr, decoding_k_cache_ptr + (size_t)first_row * DK, (size_t)part_2_length * DK * sizeof(bf16));
        memcpy(prefill_v_cache_ptr, decoding_v_cache_ptr + (size_t)first_row * DV, (size_t)part_2_length * DV * sizeof(bf16));
        prefill_k_cache_ptr += (size_t)part_2_length * DK;
        prefill_v_cache_ptr += (size_t)part_2_length * DV;
    }

    // the decode buffer moves its base forward: keep the rows of positions [new base, L_begin) at the front (they are part of the
    // window this call leaves behind), the stale rows behind the new context are zeroed after part 3
    if (swa_base_new > swa_base_old && L_begin > swa_base_new){
        const size_t move_rows = (size_t)(L_begin - swa_base_new);
        const size_t from_row = (size_t)(swa_base_new - swa_base_old);
        memmove(decoding_k_cache_ptr, decoding_k_cache_ptr + from_row * DK, move_rows * DK * sizeof(bf16));
        memmove(decoding_v_cache_ptr, decoding_v_cache_ptr + from_row * DV, move_rows * DV * sizeof(bf16));
    }

    // copy the new k cache and v cache to the appropriate location in prefill_cache and decoding_cache
    bf16* new_k_cache_ptr = buffer_k.data() + L_offset * DK;
    bf16* new_v_cache_ptr = buffer_v.data() + L_offset * DV;
    //part 3: the new rows go to the prefill cache (absolute positions) and to the decode buffer (row = position - new base; positions before the new base are not kept)
    for (int i = 0; i < part_3_length; i++){
        memcpy(prefill_k_cache_ptr, new_k_cache_ptr + (size_t)i * DK, DK * sizeof(bf16));
        memcpy(prefill_v_cache_ptr, new_v_cache_ptr + (size_t)i * DV, DV * sizeof(bf16));
        const int pos = L_begin + i;
        if (pos >= swa_base_new){
            const size_t row = (size_t)(pos - swa_base_new);
            memcpy(decoding_k_cache_ptr + row * DK, new_k_cache_ptr + (size_t)i * DK, DK * sizeof(bf16));
            memcpy(decoding_v_cache_ptr + row * DV, new_v_cache_ptr + (size_t)i * DV, DV * sizeof(bf16));
        }
        prefill_k_cache_ptr += DK;
        prefill_v_cache_ptr += DV;
    }
    // rows behind the new context must be zero (see gemma4e_dflash_flash_swa.hpp)
    {
        const int first = (L_begin + L_effective) - swa_base_new;
        const int hi = std::min(swa_zero_hi, kSwaBufferRows);
        if (hi > first && first >= 0){
            memset(decoding_k_cache_ptr + (size_t)first * DK, 0, (size_t)(hi - first) * DK * sizeof(bf16));
            memset(decoding_v_cache_ptr + (size_t)first * DV, 0, (size_t)(hi - first) * DV * sizeof(bf16));
        }
    }

    sliding_scratch_rows = std::max(sliding_scratch_rows, static_cast<size_t>(L_begin + L_effective));
    decoding_cache.sync_to_device();
    prefill_cache.sync_to_device();
}

// ---------------------------------------------------------------------------
// mlp block
// ---------------------------------------------------------------------------

void gemma4e_mlp_prefill_context::forward(
    const gemma4e_prefill_shape& s,
    gemma4e_layer_type_t type,
    buffer<u8>& proj_weights,
    SafeTensors* reference
){
    // a double-wide layer is the same three dispatches over twice the mlp width
    const int t = int(type);
    const int I = (int)desc->get_intermediate_size(type);

    PREFILL_PROF_INIT();
    bufs->hidden_state_buffer.sync_to_device();
    // up first, then gate: the two write disjoint 512-column slabs of the same
    // band, and the array has already applied gelu to the gate slabs.
    this->upgate_proj[t][0](bufs->upgate_buffer, bufs->hidden_state_buffer, proj_weights);
    this->upgate_proj[t][1](bufs->upgate_buffer, bufs->hidden_state_buffer, proj_weights);
    bufs->upgate_buffer.sync_from_device();
    PREFILL_PROF_TICK("mlp:up+gate_proj");

    static_assert(gemma4e_desc::UP_GATE_OUTDIM_REORDER == 512,
        "one fused-gemm j step covers 512 output columns; the up/gate interleave "
        "period must match or the strided split lands mid-slab");
    // over all L_padded rows, not L_in: the real tokens start at row L_offset,
    // and the padding rows still feed the down gemm.
    gemma4e_cpu_func::_glu_mul(bufs->hid_buffer.data(), bufs->upgate_buffer.data(),
        s.L_padded, I, gemma4e_desc::UP_GATE_OUTDIM_REORDER);
    bufs->hid_buffer.sync_to_device();
    PREFILL_PROF_TICK("mlp:glu_mul");

    this->down_proj[t](bufs->hidden_state_buffer, bufs->hid_buffer, proj_weights);
    bufs->hidden_state_buffer.sync_from_device();
    PREFILL_PROF_TICK("mlp:down_proj");

    DEBUG_BLOCK(2,
        buffer<bf16> down_ref;
        reference->load_weights(down_ref, "mlp_output");
        buffer<bf16> down_valid = buffer<bf16>(bufs->hidden_state_buffer.data(), (size_t)s.L_effective * desc->D);
        buffer<bf16> down_valid_ref = buffer<bf16>(down_ref.data(), (size_t)s.L_effective * desc->D);
        print_error_metrics(get_error_metrics(down_valid, down_valid_ref), "MLP Error: ");
    )
}

// ---------------------------------------------------------------------------
// per layer input path
// ---------------------------------------------------------------------------

void gemma4e_pli_prefill_context::setup(const gemma4e_prefill_shape& s){
    if (L_padded_old == s.L_padded) {
        return;
    }
    const uint32_t D = desc->D;
    const uint32_t PLI_D = desc->PLI_D;
    const uint32_t PLI = PLI_D * desc->num_hidden_layers;

    // All three run on fused_prefill's bf16 passthrough: the q_type RTP tells
    // the mm cores to copy the weight tile into w_buffer instead of unpacking
    // q4_1 nibbles, so no overlay switch is needed to reach a bf16 gemm.
    seq_gen->gen_dequant_mm_512(this->pli_down_proj.seq(), s.L_padded, D, PLI,
        /*weight_offset=*/0, /*out_offset=*/0, /*rtp_act=*/0,
        /*n_stride=*/1, /*n_first=*/0, flm_bf16);

    // gelu, matching the image encoder's OUTPUT_MODE 1 that this replaced
    seq_gen->gen_dequant_mm_512(this->pli_gate_proj.seq(), s.L_padded, D, PLI_D,
        /*weight_offset=*/0, /*out_offset=*/0, /*rtp_act=*/2,
        /*n_stride=*/1, /*n_first=*/0, flm_bf16);

    // K is PLI_D outright: the fused mm stages one 256-column k_slice per
    // memtile load, so this needs no padding.  The up half of the packed
    // weight sits behind the gate half.
    seq_gen->gen_dequant_mm_512(this->pli_up_proj.seq(), s.L_padded, PLI_D, D,
        /*weight_offset=*/desc->pli_gate_packed_elems() * sizeof(bf16), /*out_offset=*/0,
        /*rtp_act=*/0, /*n_stride=*/1, /*n_first=*/0, flm_bf16);

    L_padded_old = s.L_padded;
}

void gemma4e_pli_prefill_context::pre_pass(
    const gemma4e_prefill_shape& s,
    buffer<bf16>& pli_down_weights,
    buffer<bf16>& pli_input_norm
){
    const int D = desc->D;
    const int PLI_D = desc->PLI_D;
    const int num_hidden_layers = desc->num_hidden_layers;

    PREFILL_PROF_INIT();
    // the down projection reads the token embeddings, which start life in the residual
    memcpy(bufs->hidden_state_buffer.data() + (size_t)s.L_offset * D, bufs->residual_buffer.data() + (size_t)s.L_offset * D, (size_t)s.L_effective * D * sizeof(bf16));
    bufs->hidden_state_buffer.sync_to_device();
    this->pli_down_proj(bufs->pli_down_buffer, bufs->hidden_state_buffer, pli_down_weights);
    bufs->pli_down_buffer.sync_from_device();
    PREFILL_PROF_TICK("pli:down_proj");

    gemma4e_cpu_func::_elementwise_scale_batch(bufs->pli_down_buffer.data(), 1.0 / sqrtf((float)D), PLI_D * num_hidden_layers, s.L_effective, s.L_offset);
    gemma4e_cpu_func::_rms_norm_batch(bufs->pli_down_buffer.data(), bufs->pli_down_buffer.data(), pli_input_norm.data(), PLI_D, num_hidden_layers * PLI_D, s.L_effective, s.L_offset, s.L_offset);
    gemma4e_cpu_func::_residual_add_batch(bufs->pli_embed_buffer.data(), bufs->pli_down_buffer.data(), bufs->pli_embed_buffer.data(), num_hidden_layers * PLI_D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    gemma4e_cpu_func::_elementwise_scale_batch(bufs->pli_embed_buffer.data(), 1.0 / sqrtf((float)2.0), PLI_D * num_hidden_layers, s.L_effective, s.L_offset);
    PREFILL_PROF_TICK("pli:norm+residual");
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

    PREFILL_PROF_INIT();
    this->pli_gate_proj(bufs->pli_gate_buffer, bufs->hidden_state_buffer, pli_gate_up_weights);
    bufs->pli_gate_buffer.sync_from_device();
    PREFILL_PROF_TICK("pli:gate_proj");

    // this layer's slice of the per-layer embeddings gates the projection
    gemma4e_cpu_func::_elementwise_mul_batch(bufs->pli_hid_buffer.data(), bufs->pli_gate_buffer.data(), bufs->pli_embed_buffer.data() + (size_t)layer_idx * PLI_D, PLI_D, num_hidden_layers * PLI_D, s.L_effective, s.L_offset, s.L_offset, s.L_offset);

    bufs->pli_hid_buffer.sync_to_device();
    this->pli_up_proj(bufs->hidden_state_buffer, bufs->pli_hid_buffer, pli_gate_up_weights);
    bufs->hidden_state_buffer.sync_from_device();
    PREFILL_PROF_TICK("pli:gate_mul+up_proj");

    gemma4e_cpu_func::_rms_norm_batch(bufs->hidden_state_buffer.data(), bufs->hidden_state_buffer.data(), pli_final_norm.data(), D, D, s.L_effective, s.L_offset, s.L_offset);
    bufs->hidden_state_buffer.sync_to_device();
    PREFILL_PROF_TICK("pli:post_pli_norm");
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

    PREFILL_PROF_INIT();

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
    PREFILL_PROF_TICK("layer:input_rms_norm");

    DEBUG_BLOCK(2,
        buffer<bf16> norm_ref;
        reference->load_weights(norm_ref, "input_layernorm_output");
        buffer<bf16> norm_valid = buffer<bf16>(bufs.hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> norm_valid_ref = buffer<bf16>(norm_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(norm_valid, norm_valid_ref), "Input LayerNorm Error: ");
    )

    this->attn_block->forward(s, type, proj_weights, kv_cache, q_norm, k_norm, reference);
    PREFILL_PROF_TICK("layer:attn_block_total");

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
    PREFILL_PROF_TICK("layer:post_attn_norm+residual+pre_ffn_norm");

    DEBUG_BLOCK(2,
        buffer<bf16> pre_ffn_norm_ref;
        reference->load_weights(pre_ffn_norm_ref, "pre_ffn_norm_output");
        buffer<bf16> pre_ffn_norm_valid = buffer<bf16>(bufs.hidden_state_buffer.data(), s.L_effective * D);
        buffer<bf16> pre_ffn_norm_valid_ref = buffer<bf16>(pre_ffn_norm_ref.data(), s.L_effective * D);
        print_error_metrics(get_error_metrics(pre_ffn_norm_valid, pre_ffn_norm_valid_ref), "Pre-FFN LayerNorm Error: ");
    )

    this->mlp->forward(s, type, proj_weights, reference);
    PREFILL_PROF_TICK("layer:mlp_block_total");

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

    PREFILL_PROF_TICK("layer:post_ffn_norm+residual");

    this->pli->layer_pass(s, layer_idx, pli_gate_up_weights, pli_final_norm);
    PREFILL_PROF_TICK("layer:pli_layer_pass");

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
