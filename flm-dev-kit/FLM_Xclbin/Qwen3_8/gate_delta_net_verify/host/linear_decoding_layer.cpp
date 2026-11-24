// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "linear_decoding_layer.hpp"

linear_decoding_layer::linear_decoding_layer(model_list::model_t model_type, LM_Config& lm_config, npu_app_manager* layer_app_mgr)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr) {
    header_print("info", "Initializing decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();

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
  
    linear_w_qkv_offset = 0;
    linear_w_gate_offset = linear_w_qkv_offset + q4k_bytes((size_t)(Linear_DQ + Linear_DK + Linear_DV) * D) / sizeof(bf16);
    linear_w_o_offset = linear_w_gate_offset + q4k_bytes((size_t)Linear_DV * D) / sizeof(bf16);
    linear_w_upgate_offset = linear_w_o_offset + q4k_bytes((size_t)D * Linear_DV) / sizeof(bf16);
    linear_w_down_offset = linear_w_upgate_offset + q4k_bytes((size_t)2 * intermediate_size * D) / sizeof(bf16);
    // The a/b tail: one bias block, then alpha and beta. The bias is a weight
    // block as far as the tile is concerned -- its BD chain on that channel is a
    // uniform ping/pong now, with no prologue to land it separately -- so it has
    // to be exactly AB_W_BLOCK_SIZE and it has to sit immediately in front of
    // the q8_0 stream, where one descriptor can carry both.
    ab_w_block_words = q80_bytes(32 * 256) / sizeof(bf16);
    w_ab_bias_offset = linear_w_down_offset + q4k_bytes((size_t)intermediate_size * D) / sizeof(bf16);
    w_ab_offset = w_ab_bias_offset + ab_w_block_words;

    header_print("info", "Decoding layer config: ");
    header_print("info", "  D : " << D);
    header_print("info", "  Linear_DQ: " << Linear_DQ);
    header_print("info", "  Linear_DK: " << Linear_DK);
    header_print("info", "  Linear_DV: " << Linear_DV);
    header_print("info", "  Conv Kernel Size: " << conv_kernel_size);
    header_print("info", "  Intermediate Size: " << intermediate_size);

    size_t total_q4_k_proj_weights =
          (Linear_DQ + Linear_DK + Linear_DV) * D  // qkv
        + Linear_DV * D                            // attn gate
        + D * Linear_DV                            // ssm_out
        + 3 * D * intermediate_size;               // up, gate, down

    size_t total_q8_0_proj_weights =
        (size_t)2 * (linear_num_v_heads_padded) * D; // alpha and beta q8

    size_t total_proj_weights_byte_size = q4k_bytes(total_q4_k_proj_weights) +
                                          q80_bytes(total_q8_0_proj_weights) +
                                          ab_w_block_words * sizeof(bf16); // the a/b bias block
    // | conv in | S in | conv out |. S is f32, hence the * 2 in bf16 words; the
    // conv window is bf16 pre-conv rows, so it is not doubled.
    //
    // There is no S-out region. The state crosses DDR one way now: the host
    // writes S_0 here, the cores read it and keep S resident for all L_TOKENS
    // steps, and the state after the accepted prefix is rebuilt by
    // get_all_S_at() from the triplet arena rather than read back off the array.
    // That drops a 3 MB write-back per pass and 128 KiB of MT[4], and it costs
    // no information: the device could only ever report the accept-*all* state,
    // which is the one case speculative decode cannot rely on.
    //
    // The conv window in and out are *not* the same size, for that same reason.
    // The pass reads one window in but writes every accept length's window out,
    // because at dispatch time nobody knows how many drafted tokens will
    // survive verification. See get_conv_out() for the slicing.
    conv_words = static_cast<size_t>(conv_kernel_size - 1) * (Linear_DQ + Linear_DK + Linear_DV);
    conv_out_words = static_cast<size_t>(L_TOKENS + conv_kernel_size - 2) *
                     (Linear_DQ + Linear_DK + Linear_DV);
    s_offset = conv_words;
    s_words = static_cast<size_t>(linear_num_v_heads) * linear_v_head_dim * linear_qk_head_dim * 2;
    conv_out_offset = s_offset + s_words;
    linear_kv_cache_size = conv_out_offset + conv_out_words;
    conv_w_size = (size_t)conv_kernel_size * (Linear_DQ + Linear_DK + Linear_DV);

    header_print("info", "  CONV_W_SIZE: " << conv_w_size);

    // Rows each projection emits per token. Both are weight-stream shapes now,
    // not DDR shapes: neither qkv nor z ever reaches the host.
    qkv_out   = Linear_DQ + Linear_DK + Linear_DV;
    z_out     = Linear_DV;
    // alpha then beta, f32, padded to the 32-row quantisation block.
    ab_out    = 2 * linear_num_v_heads_padded;

    // The weight offsets above are in the order the cores consume them:
    // qkv | z | o | up|gate | down | alpha | beta. alpha and beta move in one
    // _move_alpha_beta_weights() call, so they stay adjacent and in that order.

    // The triplet arena. Every offset in it has to be an even bf16 word count:
    // npu_dma_memcpy_nd folds bf16 into 32-bit words, halving the offset and
    // every stride, so an odd one would silently truncate. 128 + 2, 3 * 130 and
    // arena_offset are all even, so the arena is aligned by construction -- and
    // the two-word alpha / beta holes are 4-byte aligned, which is what lets the
    // a/b tile write f32 into them.
    triplet_stride = linear_v_head_dim + 2;
    arena_block    = 3 * triplet_stride;
    arena_heads    = linear_num_v_heads_padded;

    // One buffer, because the sequence addresses every region through the same
    // argument id:
    //
    //   [0, L*D)                        x, the rows going in
    //   [arena_offset, +heads*L*block)  the head-major (k|a|v|b|q) triplets
    //   [layer_out_offset, +L*D)        h2, the layer's output
    //
    // The recurrent state, the other output, lives in kv_cache. The arena is not
    // a readback -- it is the one round trip the array itself depends on, the
    // token-major -> head-major transpose no tile on that path can do. alpha and
    // beta have no region of their own: they live in the arena's holes.
    arena_offset = static_cast<size_t>(L_TOKENS) * D;
    layer_out_offset = arena_offset + static_cast<size_t>(arena_heads) * L_TOKENS * arena_block;
    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(
        layer_out_offset + static_cast<size_t>(L_TOKENS) * D);
    // Nothing writes the two pad words after q, and the head slots above
    // linear_num_v_heads only ever get alpha / beta. Zero once so the cores'
    // padding lanes are not NaN.
    this->hidden_states_buffer.memset((bf16)0);
    // Both norm weights in one transfer: the rms tile's w BD is self-chained, so
    // it reloads w_buffer between the input norm and the post-attention norm.
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(2 * D);

    // qkv, z, o, up|gate, down and a/b -- ~183 MB, of which down is ~49 MB.
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(total_proj_weights_byte_size);
    // The layer's small resident weights:
    //   | conv_w (conv_w_size) | ssm_norm (linear_v_head_dim) |
    // conv_w and ssm_norm being adjacent is what lets one lengthened transfer
    // on IT5 MM2S ch1 feed MT[5]'s conv-tap BD and its ssm_norm BD back to
    // back -- see _send_linear_conv_weights(). The a/b activation bias used to
    // follow them; it lives in all_proj_weight_buffer now, at w_ab_bias_offset.
    this->aux_w_buffer = this->layer_app.create_bo_buffer<bf16>(
        conv_w_size + linear_v_head_dim);
    // The conv window and the recurrent state, each an in / out pair. Zeroed to
    // start, which is what cpu_ref::delta_net_layer::reset() starts from, so a
    // single L_TOKENS pass compares exactly -- and advance_state() carries both
    // out regions back over their in regions so a second pass does too.
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(linear_kv_cache_size);

    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();
    this->_gen_linear_sequence(this->layer_app.seq());

}

void linear_decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
    // The memtile fills w_buffer[0 : 2*W_BLOCK_SIZE] contiguously and hands half to
    // each of the two row-blocks, so DDR must alternate whole blocks between them.
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


void linear_decoding_layer::load_weights(Q4NX& q4nx) {
    int layer_idx = 0;

    // Both layernorms, qkv_proj, the z gate proj, ssm_out_proj, the fused
    // up | gate, mlp.down_proj and the a/b projections. The blocks land in the
    // order _gen_linear_sequence() streams them.
    {
        LOG_VERBOSE(1, "Loading qkv weights");
        buffer<u8> w_qkv;
        q4nx.load_weights(w_qkv, "model.layers." + std::to_string(layer_idx) + ".linear_attn.qkv_proj.weight");
        LOG_VERBOSE(1, "Copying qkv weights to buffer");
        reorder_cpy(this->all_proj_weight_buffer.data(), w_qkv, D);
    }
    {
        // z, the silu gate of the delta-net's gated RMSNorm. The checkpoint
        // calls it self_attn.gate_proj even on a linear-attention layer.
        LOG_VERBOSE(1, "Loading z (gate) weights");
        buffer<u8> w_z;
        q4nx.load_weights(w_z, "model.layers." + std::to_string(layer_idx) + ".self_attn.gate_proj.weight");
        LOG_VERBOSE(1, "Copying z weights to buffer");
        reorder_cpy(this->all_proj_weight_buffer.data() + q4k_bytes(static_cast<size_t>(qkv_out) * D),
                    w_z, D);
    }
    {
        // ssm_out_proj, D x Linear_DV -- the only projection so far whose K is
        // not D, hence the Linear_DV column count handed to reorder_cpy.
        LOG_VERBOSE(1, "Loading o weights");
        buffer<u8> w_o;
        q4nx.load_weights(w_o, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_out_proj.weight");
        LOG_VERBOSE(1, "Copying o weights to buffer");
        reorder_cpy(this->all_proj_weight_buffer.data() + linear_w_o_offset * sizeof(bf16),
                    w_o, Linear_DV);
    }
    {
        // mlp.up_proj and mlp.gate_proj, woven 512 output rows at a time rather
        // than concatenated: the GLU tile takes one engine round at a time and
        // needs a round to be all up or all gate. 512 is even in row blocks, so
        // this is byte for byte what one reorder_cpy() over the woven matrix
        // would produce; host.cpp's golden weaves the reference the same way.
        LOG_VERBOSE(1, "Loading up/gate weights");
        buffer<u8> w_up;
        buffer<u8> w_gate;
        q4nx.load_weights(w_up,   "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight");
        q4nx.load_weights(w_gate, "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight");

        const int blocks     = 1024 / 2 / 32 * (D / 256);
        const int chunk_size = blocks * q4k_bytes(32 * 256);
        const int total_size = q4k_bytes(static_cast<size_t>(intermediate_size) * D);
        const int phases     = total_size / chunk_size;
        assert(static_cast<size_t>(phases) * chunk_size == static_cast<size_t>(total_size));
        tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
        u8* w_ptr = this->all_proj_weight_buffer.data() + linear_w_upgate_offset * sizeof(bf16);
        for (int i = 0; i < phases; i++)
        {
            LOG_VERBOSE(1, "Copying up and gate weights to buffer, phase " << i + 1 << "/" << phases);
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
    }
    {
        // mlp.down_proj, D x intermediate_size.
        LOG_VERBOSE(1, "Loading down weights");
        buffer<u8> w_down;
        q4nx.load_weights(w_down, "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");
        LOG_VERBOSE(1, "Copying down weights to buffer");
        reorder_cpy(this->all_proj_weight_buffer.data() + linear_w_down_offset * sizeof(bf16),
                    w_down, intermediate_size);
    }
    {
        // alpha and beta, q8_0, linear_num_v_heads_padded x D each. A plain
        // memcpy, not reorder_cpy: the a/b kernel walks its blocks row-block
        // major (all of a row block's column blocks, then the next row block),
        // which is already the q4nx file order. The two-block interleave the
        // projection engine needs comes from *its* memtile handing half a fill
        // to each of two row blocks, and the a/b tile has no such split.
        LOG_VERBOSE(1, "Loading alpha and beta weights");
        buffer<u8> w_alpha;
        buffer<u8> w_beta;
        q4nx.load_weights(w_alpha, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_alpha_proj.weight");
        q4nx.load_weights(w_beta,  "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_beta_proj.weight");

        const size_t half = q80_bytes(static_cast<size_t>(linear_num_v_heads_padded) * D);
        assert(w_alpha.size() == half && w_beta.size() == half);
        u8* ab_ptr = this->all_proj_weight_buffer.data() + w_ab_offset * sizeof(bf16);
        memcpy(ab_ptr,        w_alpha.data(), half);
        memcpy(ab_ptr + half, w_beta.data(),  half);
    }
    {
        // The a/b activation bias, laid out exactly as activate_a() reads it:
        // A = A_BIAS[0 .. padded), BIAS = A_BIAS[padded .. 2 * padded), f32.
        // ssm_a and ssm_dt.bias only have linear_num_v_heads entries, so the
        // tail lanes are zeroed -- their alpha comes out exp(0 * softplus(0))
        // = 1 rather than a garbage exponential.
        //
        // It sits in the *projection* BO, one AB_W_BLOCK_SIZE block ahead of
        // alpha, because the a/b tile receives it as an ordinary weight block
        // and ab_proj() copies it off the top of each token's stream. Whole
        // block zeroed first: everything past the 2 * padded floats is padding
        // the kernel does not read, but it crosses DDR either way and NaNs in a
        // q8_0 block are not worth the risk of some later reader finding them.
        LOG_VERBOSE(1, "Loading alpha/beta activation bias");
        buffer<float> a;
        buffer<float> dt;
        q4nx.load_weights(a,  "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_a");
        q4nx.load_weights(dt, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_dt.bias");

        u8* blk = this->all_proj_weight_buffer.data() + w_ab_bias_offset * sizeof(bf16);
        memset(blk, 0, ab_w_block_words * sizeof(bf16));
        float* bias_ptr = reinterpret_cast<float*>(blk);
        memcpy(bias_ptr, a.data(), linear_num_v_heads * sizeof(float));
        memcpy(bias_ptr + linear_num_v_heads_padded, dt.data(), linear_num_v_heads * sizeof(float));
    }
    this->all_proj_weight_buffer.sync_to_device();
    {
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre;
        buffer<bf16> w_rms_post;

        bf16* w_rms_ptr = this->rms_weights_buffer.data();
        q4nx.load_weights(w_rms_pre, "model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
        q4nx.load_weights(w_rms_post, "model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
        memcpy((unsigned char*)w_rms_ptr, w_rms_pre.data(), D * sizeof(bf16));
        w_rms_ptr += D;
        memcpy((unsigned char*)w_rms_ptr, w_rms_post.data(), D * sizeof(bf16));
        w_rms_ptr += D;
    }
    this->rms_weights_buffer.sync_to_device();
    this->aux_w_buffer.memset((bf16)0.0f);
    {
        // The depthwise conv1d taps. The .q4nx already stores them tap-major,
        // [conv_kernel_size][Linear_DQ + Linear_DK + Linear_DV], which is
        // exactly the memtile's conv_w_buffer layout -- hence a plain memcpy
        // and not reorder_cpy. cpu_ref::delta_net_weights::load reads the same
        // tensor with the same assumption.
        LOG_VERBOSE(1, "Loading conv1d weights");
        buffer<bf16> conv_weights;
        q4nx.load_weights(conv_weights, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_conv1d.weight");
        assert(conv_weights.size() == static_cast<size_t>(conv_w_size));
        memcpy(this->aux_w_buffer.data(), conv_weights.data(), conv_w_size * sizeof(bf16));
    }
    {
        // The gated RMSNorm's weight. One head's worth, linear_v_head_dim, and
        // the same vector for all linear_num_v_heads heads -- which is why
        // MT[5] can just replay it and delta_net_norm.cc can loop the heads
        // against a single w.
        LOG_VERBOSE(1, "Loading ssm_norm weights");
        buffer<bf16> norm_weights;
        q4nx.load_weights(norm_weights, "model.layers." + std::to_string(layer_idx) + ".linear_attn.ssm_norm.weight");
        assert(norm_weights.size() >= static_cast<size_t>(linear_v_head_dim));
        memcpy(this->aux_w_buffer.data() + conv_w_size, norm_weights.data(),
               linear_v_head_dim * sizeof(bf16));
    }
    this->aux_w_buffer.sync_to_device();
    header_print("info", "Weights loaded!");
}


/// One pass over L_TOKENS rows. `input` must hold at least L_TOKENS * D
/// elements; the return is h2 = h1 + mlp.down_proj(hid), the layer's output,
/// L_TOKENS x D in row-major token order.
///
/// The rms tile emits whole D-wide rows, so unlike the projections' output this
/// needs no un-permute on the way in. The pass also fills the triplet arena,
/// which the array reads back itself.
buffer<bf16> linear_decoding_layer::operator()(buffer<bf16>& input) {
    const size_t in_size = static_cast<size_t>(L_TOKENS) * D;
    assert(input.size() >= in_size);
    memcpy(this->hidden_states_buffer.data(), input.data(), in_size * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    this->layer_app(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, aux_w_buffer, kv_cache);
    this->hidden_states_buffer.sync_from_device();

    buffer<bf16> output(static_cast<size_t>(L_TOKENS) * D);
    memcpy(output.data(), this->hidden_states_buffer.data() + layer_out_offset,
           output.size() * sizeof(bf16));
    return output;
}

bool linear_decoding_layer::probe_pass(buffer<bf16>& input, const std::string& label) {
    // 0xAA in both bytes of a bf16 is -3.0316488e-13: a value no projection
    // produces, and one that survives a sync round trip unchanged.
    constexpr u8 POISON = 0xAA;
    const size_t in_size   = static_cast<size_t>(L_TOKENS) * D;
    const size_t arena_len = this->layer_out_offset - this->arena_offset;
    const size_t out_len   = in_size;

    // Snapshot of whatever the previous pass left, so the run below can be
    // scored against it word for word. That is the real invariant -- nothing a
    // pass runs changes one of its own inputs, so pass n + 1 has to reproduce
    // pass n exactly -- and it is stricter than counting writes.
    std::vector<bf16> prev_arena(this->hidden_states_buffer.data() + this->arena_offset,
                                 this->hidden_states_buffer.data() + this->arena_offset + arena_len);
    std::vector<bf16> prev_conv(this->kv_cache.data() + this->conv_out_offset,
                                this->kv_cache.data() + this->conv_out_offset + this->conv_out_words);
    std::vector<bf16> prev_h2(this->hidden_states_buffer.data() + this->layer_out_offset,
                              this->hidden_states_buffer.data() + this->layer_out_offset + out_len);

    memcpy(this->hidden_states_buffer.data(), input.data(), in_size * sizeof(bf16));
    memset(this->hidden_states_buffer.data() + this->arena_offset, POISON,
           arena_len * sizeof(bf16));
    memset(this->hidden_states_buffer.data() + this->layer_out_offset, POISON,
           out_len * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();

    memset(this->kv_cache.data() + this->conv_out_offset, POISON,
           this->conv_out_words * sizeof(bf16));
    this->kv_cache.sync_to_device();

    time_utils::time_point t0 = time_utils::now();
    xrt::run run = this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer,
                                              rms_weights_buffer, aux_w_buffer, kv_cache);
    run.start();
    ert_cmd_state state = run.wait();
    const double ms = time_utils::duration_ms(t0, time_utils::now()).first;

    this->hidden_states_buffer.sync_from_device();
    this->kv_cache.sync_from_device();

    // Count words still holding the poison. A partially written region is the
    // interesting case: it says the stall is inside that stage, not before it.
    auto untouched = [&](const bf16* p, size_t n) {
        const u8* b = reinterpret_cast<const u8*>(p);
        size_t k = 0;
        for (size_t i = 0; i < n; i++)
            if (b[2 * i] == POISON && b[2 * i + 1] == POISON) k++;
        return k;
    };
    const size_t a_left = untouched(this->hidden_states_buffer.data() + this->arena_offset, arena_len);
    const size_t c_left = untouched(this->kv_cache.data() + this->conv_out_offset, this->conv_out_words);
    const size_t h_left = untouched(this->hidden_states_buffer.data() + this->layer_out_offset, out_len);

    // Words the pass wrote, against what the previous pass left there. Only the
    // written ones: the arena is structurally sparse -- it is indexed by the
    // padded v-head count, and the 16 pad heads plus the tail of each 130-word
    // triplet slot are never written by anyone -- so "100% written" is not the
    // bar for it and a_left == 0 would fail every healthy pass.
    auto differs = [&](const bf16* now, const std::vector<bf16>& was, size_t n) {
        const u8* b = reinterpret_cast<const u8*>(now);
        size_t k = 0;
        for (size_t i = 0; i < n; i++) {
            if (b[2 * i] == POISON && b[2 * i + 1] == POISON) continue;   // untouched
            if (memcmp(now + i, was.data() + i, sizeof(bf16)) != 0) k++;
        }
        return k;
    };
    const size_t a_diff = differs(this->hidden_states_buffer.data() + this->arena_offset,
                                  prev_arena, arena_len);
    const size_t c_diff = differs(this->kv_cache.data() + this->conv_out_offset,
                                  prev_conv, this->conv_out_words);
    const size_t h_diff = differs(this->hidden_states_buffer.data() + this->layer_out_offset,
                                  prev_h2, out_len);

    auto pct = [](size_t left, size_t n) {
        return std::to_string(100.0 * (double)(n - left) / (double)n) + "% written";
    };
    header_print("info", label + ": ert state " + std::to_string((int)state) + ", " +
                             std::to_string(ms) + " ms");
    header_print("info", "  arena    " + pct(a_left, arena_len) + ", " +
                             std::to_string(a_diff) + " words differ from the pass before");
    header_print("info", "  conv out " + pct(c_left, this->conv_out_words) + ", " +
                             std::to_string(c_diff) + " words differ");
    header_print("info", "  h2       " + pct(h_left, out_len) + ", " +
                             std::to_string(h_diff) + " words differ");

    // conv out and h2 are dense, so those two do have to come back whole.
    return state == ERT_CMD_STATE_COMPLETED && c_left == 0 && h_left == 0 &&
           a_diff == 0 && c_diff == 0 && h_diff == 0;
}

bool linear_decoding_layer::probe_prefix(buffer<bf16>& input, int stages) {
    this->_gen_linear_sequence(this->layer_app.seq(), stages);
    const size_t in_size = static_cast<size_t>(L_TOKENS) * D;
    memcpy(this->hidden_states_buffer.data(), input.data(), in_size * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();

    time_utils::time_point t0 = time_utils::now();
    xrt::run run = this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer,
                                              rms_weights_buffer, aux_w_buffer, kv_cache);
    run.start();
    ert_cmd_state state = run.wait();
    const double ms = time_utils::duration_ms(t0, time_utils::now()).first;
    header_print("info", "prefix " << stages << ": ert state " << (int)state << ", "
                                   << ms << " ms");
    return state == ERT_CMD_STATE_COMPLETED;
}

bool linear_decoding_layer::probe_runlist(buffer<bf16>& input, int n) {
    const size_t in_size = static_cast<size_t>(L_TOKENS) * D;
    memcpy(this->hidden_states_buffer.data(), input.data(), in_size * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();

    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < n; i++)
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer,
                                          rms_weights_buffer, aux_w_buffer, kv_cache));
    // xrt::runlist::wait() returns void here and reports a failed command by
    // throwing, so the state has to be read back off the list rather than
    // returned.
    bool ok = true;
    try {
        rl.execute();
        rl.wait();
    } catch (const std::exception& e) {
        header_print("error", "  runlist threw: " << e.what());
        ok = false;
    }
    this->hidden_states_buffer.sync_from_device();
    return ok;
}

/// The same pass operator() runs, split so the host keeps its thread.
///
/// operator() is exactly start_pass() + finish_pass(); the only reason to use
/// the pair is to run get_S_at() -- or any other host work -- between them.
/// Everything the kernel reads is synced before the run starts, so the host is
/// free the moment this returns.
xrt::run linear_decoding_layer::start_pass(buffer<bf16>& input) {
    const size_t in_size = static_cast<size_t>(L_TOKENS) * D;
    assert(input.size() >= in_size);
    memcpy(this->hidden_states_buffer.data(), input.data(), in_size * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    xrt::run run = this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer,
                                              rms_weights_buffer, aux_w_buffer, kv_cache);
    run.start();
    return run;
}

buffer<bf16> linear_decoding_layer::finish_pass(xrt::run& run) {
    // An unchecked state reads as a pass: a timeout leaves the output BO
    // holding the previous run's rows, which score fine against a reference
    // that has not changed either.
    ert_cmd_state state = run.wait();
    if (state != ERT_CMD_STATE_COMPLETED) {
        header_print("error", "delta-net pass did not complete, ert state " << state);
        throw std::runtime_error("linear_decoding_layer: NPU pass did not complete");
    }
    this->hidden_states_buffer.sync_from_device();

    buffer<bf16> output(static_cast<size_t>(L_TOKENS) * D);
    memcpy(output.data(), this->hidden_states_buffer.data() + layer_out_offset,
           output.size() * sizeof(bf16));
    return output;
}

/// The conv window after the pass, from its own write-back region -- the layer's
/// third output, and the other half of what step t+1 consumes.
///
/// Stays bf16: these are the pre-conv qkv rows as the projection engine emitted
/// them, and MT[5] un-permuted them on the way in, so this is already
/// [tap][channel] with the oldest tap first.
/// One head's state after `accept_len` of the L_TOKENS drafted tokens, replayed
/// on the host from the arena. See the header for why this is not a readback.
///
/// Everything it needs is already in DDR and already synced: operator() syncs
/// hidden_states_buffer back, and the arena block for (head, token) is
///
///   [ k(dk) | alpha(f32) | v(dv) | beta(f32) | q(dq) pad ]
///
/// three triplet_stride-word sub-blocks at (head * L_TOKENS + t) * arena_block.
/// These are the bf16 values the cores actually read.
///
/// That this reproduces the core's own arithmetic was established against the
/// device's write-back while that write-back still existed: replaying to
/// accept_len == L_TOKENS matched it at cos 1.0 / NMSE 6e-11. The write-back is
/// gone now, so that cross-check cannot be re-run here; what still covers this
/// path is h2, which depends on S through o.
buffer<float> linear_decoding_layer::get_S_at(int head, int accept_len){
    assert(accept_len >= 1 && accept_len <= L_TOKENS);
    assert(head >= 0 && head < linear_num_v_heads);
    const int dk = linear_qk_head_dim;
    const int dv = linear_v_head_dim;
    const size_t count = static_cast<size_t>(dk) * dv;

    this->kv_cache.sync_from_device();
    buffer<float> S(count);
    // S_0, the state this pass started from, in the same [dk][dv] layout.
    memcpy(S.data(),
           reinterpret_cast<const float*>(this->kv_cache.data() + s_offset +
                                          static_cast<size_t>(head) * count * 2),
           count * sizeof(float));

    const bf16* arena = this->hidden_states_buffer.data() + arena_offset;
    std::vector<float> u(dv);
    for (int t = 0; t < accept_len; t++) {
        const bf16* blk = arena + (static_cast<size_t>(head) * L_TOKENS + t) * arena_block;
        const bf16* k_v = blk;
        const bf16* v_v = blk + triplet_stride;
        float alpha, beta;
        memcpy(&alpha, blk + dk, sizeof(float));
        memcpy(&beta,  v_v + dv, sizeof(float));

        // u = beta * (v - alpha * (S @ k)). Accumulated against the *incoming*
        // S, before any of the update below touches it -- same as the core,
        // which holds S_src fixed for the whole step.
        for (int j = 0; j < dv; j++) u[j] = 0.0f;
        for (int i = 0; i < dk; i++) {
            const float kk = static_cast<float>(k_v[i]);
            if (kk == 0.0f) continue;
            const float* Sr = S.data() + static_cast<size_t>(i) * dv;
            for (int j = 0; j < dv; j++) u[j] += Sr[j] * kk;
        }
        for (int j = 0; j < dv; j++)
            u[j] = beta * (static_cast<float>(v_v[j]) - alpha * u[j]);

        // S = alpha * S + k (x) u. beta is already inside u; multiplying here
        // as well would square it.
        for (int i = 0; i < dk; i++) {
            const float kk = static_cast<float>(k_v[i]);
            float* Sr = S.data() + static_cast<size_t>(i) * dv;
            for (int j = 0; j < dv; j++) Sr[j] = alpha * Sr[j] + kk * u[j];
        }
    }
    return S;
}

/// Every head's state at `accept_len`, one sweep, threads over heads.
///
/// Same recurrence as get_S_at(), hoisted so the BO sync and the parallel
/// region are paid once instead of per head. The heads are independent -- each
/// owns a disjoint dk*dv slice and its own arena blocks -- so this is a plain
/// static partition with no reduction.
void linear_decoding_layer::get_all_S_at(int accept_len, buffer<float>& out){
    assert(accept_len >= 1 && accept_len <= L_TOKENS);
    const int dk = linear_qk_head_dim;
    const int dv = linear_v_head_dim;
    const size_t count = static_cast<size_t>(dk) * dv;
    const size_t total = static_cast<size_t>(linear_num_v_heads) * count;
    if (out.size() < total) out = buffer<float>(total);

    this->kv_cache.sync_from_device();

    // Stage the inputs into ordinary heap first. An XRT BO is mapped for the
    // device, not for the CPU, and the scan reads S_0 dk times per token -- off
    // a BO that costs ~100x what the same arithmetic costs off normal memory
    // (9.5 ms vs 0.09 ms for this sweep, measured; the BO sync itself is only
    // 0.05 ms, so it is the reads, not the transfer). The copy is one linear
    // pass over 3.5 MB and does not care.
    const size_t arena_words = static_cast<size_t>(arena_heads) * L_TOKENS * arena_block;
    if (s_stage.size() < total) s_stage.assign(total, 0.0f);
    if (arena_stage.size() < arena_words) arena_stage.assign(arena_words, (bf16)0);
    memcpy(s_stage.data(), this->kv_cache.data() + s_offset, total * sizeof(float));
    memcpy(arena_stage.data(), this->hidden_states_buffer.data() + arena_offset,
           arena_words * sizeof(bf16));

    const bf16* arena = arena_stage.data();
    const float* S_in = s_stage.data();

#pragma omp parallel
    {
        std::vector<float> u(dv);
#pragma omp for schedule(static)
        for (int h = 0; h < linear_num_v_heads; h++) {
            float* S = out.data() + static_cast<size_t>(h) * count;
            memcpy(S, S_in + static_cast<size_t>(h) * count, count * sizeof(float));
            for (int t = 0; t < accept_len; t++) {
                const bf16* blk = arena + (static_cast<size_t>(h) * L_TOKENS + t) * arena_block;
                const bf16* k_v = blk;
                const bf16* v_v = blk + triplet_stride;
                float alpha, beta;
                memcpy(&alpha, blk + dk, sizeof(float));
                memcpy(&beta,  v_v + dv, sizeof(float));

                for (int j = 0; j < dv; j++) u[j] = 0.0f;
                for (int i = 0; i < dk; i++) {
                    const float kk = static_cast<float>(k_v[i]);
                    if (kk == 0.0f) continue;
                    const float* Sr = S + static_cast<size_t>(i) * dv;
                    for (int j = 0; j < dv; j++) u[j] += Sr[j] * kk;
                }
                for (int j = 0; j < dv; j++)
                    u[j] = beta * (static_cast<float>(v_v[j]) - alpha * u[j]);
                for (int i = 0; i < dk; i++) {
                    const float kk = static_cast<float>(k_v[i]);
                    float* Sr = S + static_cast<size_t>(i) * dv;
                    for (int j = 0; j < dv; j++) Sr[j] = alpha * Sr[j] + kk * u[j];
                }
            }
        }
    }
}

void linear_decoding_layer::sync_state_in(){
    this->kv_cache.sync_from_device();
}

/// Install a host-reconstructed state as the next pass's S input.
void linear_decoding_layer::set_S_in(const buffer<float>& S){
    const size_t total = static_cast<size_t>(linear_num_v_heads) *
                         linear_qk_head_dim * linear_v_head_dim;
    assert(S.size() >= total);
    this->kv_cache.sync_from_device();
    memcpy(this->kv_cache.data() + s_offset, S.data(), total * sizeof(float));
    this->kv_cache.sync_to_device();
}

/// The region holds every accept length's window at once. MT[5]'s conv_x_buffer
/// is the K - 1 cached rows followed by the L new ones, and accept-j wants the
/// K - 1 rows ending at new-row j -- buffer rows j .. j + K - 2. Over
/// j = 1 .. L that is buffer rows 1 .. L + K - 2, which the device dumps as one
/// contiguous run (row 0 is the only row no accept length can want). So a
/// window is a slice starting at row j - 1 of what came back, and j = L --
/// accept everything -- is the last one, which is what this used to return.
buffer<bf16> linear_decoding_layer::get_conv_out(int accept_len){
    assert(accept_len >= 1 && accept_len <= L_TOKENS);
    this->kv_cache.sync_from_device();
    const size_t row = static_cast<size_t>(accept_len - 1) * qkv_out;
    buffer<bf16> out(conv_words);
    memcpy(out.data(), this->kv_cache.data() + conv_out_offset + row,
           conv_words * sizeof(bf16));
    return out;
}

/// Carry both states forward so the next pass continues from the accepted
/// prefix. Both halves honour accept_len now.
///
/// The conv half is a slice of the emitted window block. The S half is a replay:
/// the device no longer writes a state back, so S_j is rebuilt from the arena
/// and written straight over the s_offset region the next pass reads.
///
/// This used to memcpy the device's write-back and silently ignore accept_len
/// for S, which was only correct when every drafted token was accepted -- the
/// caller had to know to rebuild the state itself first. It no longer can get
/// that wrong.
void linear_decoding_layer::advance_state(int accept_len){
    assert(accept_len >= 1 && accept_len <= L_TOKENS);

    // Replay before the sync_from_device() below: get_all_S_at() does its own,
    // and it reads the arena out of hidden_states_buffer, which operator() has
    // already brought back.
    buffer<float> S(static_cast<size_t>(linear_num_v_heads) * linear_qk_head_dim *
                    linear_v_head_dim);
    get_all_S_at(accept_len, S);

    this->kv_cache.sync_from_device();
    bf16* base = this->kv_cache.data();
    memcpy(base + s_offset, S.data(), s_words * sizeof(bf16));
    memcpy(base, base + conv_out_offset + static_cast<size_t>(accept_len - 1) * qkv_out,
           conv_words * sizeof(bf16));
    this->kv_cache.sync_to_device();
}

float linear_decoding_layer::bench(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(), static_cast<size_t>(L_TOKENS) * D * sizeof(bf16));
    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < num_hidden_layers; i++){
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer, rms_weights_buffer, aux_w_buffer, kv_cache));
    }
    time_utils::time_point start = time_utils::now();
    rl.execute();
    rl.wait();
    this->hidden_states_buffer.sync_from_device();
    time_utils::time_point end = time_utils::now();
    time_utils::time_with_unit total_ms = time_utils::duration_ms(start, end);
    return 1000.0 / total_ms.first;
}

void linear_decoding_layer::_gen_linear_sequence(npu_sequence* seq, int stages) {
    seq->clear_cmds();

    // `stages` < 0 builds the whole pass; otherwise the sequence stops after
    // that many of the numbered steps below. Truncation is a debugging lever,
    // not a mode: a short sequence leaves the array mid-pass, so a process that
    // runs one is good for exactly one measurement afterwards. What it buys is
    // the only way to find *which* transfer a wedged array stops on -- dispatch
    // the full pass once, rebuild at a prefix, dispatch again, and the shortest
    // prefix that times out names the step.
    int step = 0;
    auto go = [&](void) { return stages < 0 || ++step <= stages; };

    // Shim channel maps:
    //
    //   IT3  MM2S ch0  x, to the rms tile (phase 1 only; o comes from MT[3])
    //        MM2S ch1  the input and post-attention layernorm weights
    //        S2MM ch1  h2 out, the end of the pass
    //   IT4  MM2S ch0  the recurrent state in (one way: nothing comes back)
    //        MM2S ch1  the arena read-back
    //   IT5  MM2S ch0  the q8_0 a/b weights, bias block first, once per token
    //        MM2S ch1  conv taps + ssm_norm, then the conv cache
    //        S2MM ch1  alpha | beta, scattered into the arena
    //   IT6  S2MM ch0  the conv tile's k|v|q triplets, into the arena
    //
    // Three ordering rules, all of them deadlock rather than wrong answers:
    //
    //   * _move_weights() calls are positional against proj_main()'s mvms --
    //     qkv, z, o, up|gate, down, in that order and all present. An mvm with
    //     no weights behind it stalls the cores; weights with no mvm to consume
    //     them stall the DMA wait inside _move_weights().
    //   * every read is posted before any weights move: the cores stall on
    //     their y locks until the shim is draining.
    //   * the a/b weights go before the qkv weights and are re-sent once per
    //     token (the a/b tile holds one x row and streams a whole 128 x D set
    //     past it). The rms tile cannot finish row t until the a/b tile has
    //     taken it, and the engine will not drain a qkv weight until all
    //     L_TOKENS rows are in MT[1].
    //
    // The two mid-sequence waits are the arena round trip: the delta cores need
    // (k, v, q, alpha, beta) head-major while the conv tile emits token-major,
    // and no tile on that path can transpose. The conv scatters k/v/q on IT6
    // S2MM ch0, the a/b tile scatters alpha/beta into the holes on IT5 S2MM
    // ch1, and IT4 MM2S ch1 reads the whole thing back linearly, so the
    // read-back cannot be issued until both writers have drained.
    //
    // The same dependency chain pins the o and up|gate weights below: o's x is
    // the gate tile's output, which needs the delta scan, and up|gate's x is
    // the post-attention norm, which needs o.
    if (!go()) { seq->cmds2seq(); return; } _send_hidden_states(seq);
    if (!go()) { seq->cmds2seq(); return; } _send_rms_weights(seq);
    if (!go()) { seq->cmds2seq(); return; } _receive_ab(seq);
    if (!go()) { seq->cmds2seq(); return; } _receive_arena(seq);
    if (!go()) { seq->cmds2seq(); return; } _receive_layer_out(seq);
    // The conv window out. Like S, independent of the arena, so it is posted
    // with the other reads; MT[5] holds it back until the L token rows land.
    if (!go()) { seq->cmds2seq(); return; } _receive_linear_kv_cache(seq);
    // S in. Independent of the arena, so it can be posted now: MT[4] fills its
    // four state buffers straight away and the cores park on their k/v/q locks
    // until the read-back starts. The cores still need S_0 streamed in every
    // pass -- it is only the write-back that is gone.
    if (!go()) { seq->cmds2seq(); return; } _move_s(seq);
    // Prime MT[5] before the projection engine starts pushing qkv at it. Both
    // are IT5 MM2S ch1 and both block on their own DMA wait, so the order here
    // has to match the memtile's BD chain: conv taps first, then the cache.
    if (!go()) { seq->cmds2seq(); return; } _send_linear_conv_weights(seq);
    if (!go()) { seq->cmds2seq(); return; } _move_linear_kv_cache(seq);
    for (int t = 0; t < L_TOKENS; t++) {
        if (!go()) { seq->cmds2seq(); return; }
        _move_alpha_beta_weights(seq, ab_out, D, w_ab_bias_offset);
    }
    if (!go()) { seq->cmds2seq(); return; } _move_weights(seq, qkv_out, D, linear_w_qkv_offset);
    if (!go()) { seq->cmds2seq(); return; } _move_weights(seq, z_out,   D, linear_w_gate_offset);
    // The arena is complete once the conv dump and the a/b scatter have both
    // drained. Everything that feeds them has been issued above, so neither
    // wait can deadlock.
    if (!go()) { seq->cmds2seq(); return; }
    seq->npu_dma_wait(
        IT6,
        S2MM,
        it_channel_0
    );
    if (!go()) { seq->cmds2seq(); return; }
    seq->npu_dma_wait(
        IT5,
        S2MM,
        it_channel_1
    );
    if (!go()) { seq->cmds2seq(); return; }
    _send_arena_to_delta(seq);
    // ssm_out_proj. Third _move_weights(), so third mvm() in proj_main.cc -- the
    // cores consume the weight stream positionally. K is Linear_DV here, not D:
    // this projection reduces over the gated rows.
    if (!go()) { seq->cmds2seq(); return; } _move_weights(seq, D, Linear_DV, linear_w_o_offset);
    // The conv window. There used to be a wait on IT4 S2MM ch0 above this one
    // -- the recurrent state coming back -- and it was the point at which the
    // scan was known to have finished. Nothing writes that channel now, so
    // waiting on it would block forever; it is deleted rather than moved.
    //
    // Losing it costs no ordering. This wait is the same barrier: MT[5] does
    // not release the conv window until all L sliding windows have been taken,
    // which needs the same L_TOKENS steps the scan does, and h2 at the end of
    // the sequence is downstream of the scan through o anyway.
    if (!go()) { seq->cmds2seq(); return; }
    seq->npu_dma_wait(
        IT5,
        S2MM,
        it_channel_0
    );

    // The MLP: up | gate off the post-attention norm, then down off the SwiGLU.
    // Fourth and fifth _move_weights(), so fourth and fifth mvm().
    if (!go()) { seq->cmds2seq(); return; } _move_weights(seq, 2 * intermediate_size, D, linear_w_upgate_offset);
    if (!go()) { seq->cmds2seq(); return; } _move_weights(seq, D, intermediate_size, linear_w_down_offset);

    // h2, the end of the pass.
    if (!go()) { seq->cmds2seq(); return; }
    seq->npu_dma_wait(
        IT3,
        S2MM,
        it_channel_1
    );

    seq->cmds2seq();
}



void linear_decoding_layer::_send_hidden_states(npu_sequence* seq){
    // All L_TOKENS rows as one contiguous transfer. The rms tile's S2MM chain
    // splits it back into D-element rows across its ping/pong buffers, so the
    // shim does not need to know about the row boundaries.
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        xyr_tile,
        bd_0,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(L_TOKENS * D)},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

void linear_decoding_layer::_send_rms_weights(npu_sequence* seq){
    // Both norms, input_layernorm then post_attention_layernorm, as one
    // transfer. The rms tile's weight BD is self-chained on a single w_buffer
    // and the kernel acquires the weight lock once per phase, so the second D
    // words simply overwrite the first between the two phases -- which is the
    // only reason a tile with four banks can do two different norms.
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rms_arg_id,
        MM2S,
        xyr_tile,
        bd_1,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(2 * D)},
        {0, 0, 0, 1},
        0, 0, false, aggressive_cache
    );
}

/// Scatter the a / b tile's output into the arena's alpha and beta holes.
///
/// The tile emits, per token, 64 alpha floats then 64 beta floats; only the BD's
/// access pattern makes an arena out of them. That is why the L=1 design's
/// a/b -> conv handoff could go: the scalars meet k and v in DDR instead of on
/// the array, and the shim's address generator does it for free.
///
///   iter (8)   token     stride arena_block  -- one token per replay
///   dim2 (2)   alpha|beta stride triplet_stride -- alpha sits one sub-block
///                                                 before beta
///   dim1 (64)  head       stride L*arena_block
///   dim0 (2)   the two bf16 words of one f32
///
/// starting at +linear_v_head_dim, i.e. right after k in each block.
void linear_decoding_layer::_receive_ab(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        IT5,
        bd_11,
        it_channel_1,
        {0, 0, 0, (uint32_t)(arena_offset + linear_v_head_dim)},
        {(uint32_t)L_TOKENS, 2, (uint32_t)arena_heads, 2},
        {(uint32_t)arena_block, (uint32_t)triplet_stride, (uint32_t)(L_TOKENS * arena_block), 1},
        -1, 0, true
    );
}

/// Scatter the conv tile's packed triplets into the arena.
///
/// The tile emits, per token, 48 dense 384-word (k | v | q) packets in head
/// order; this spreads each packet's three 128-word runs across the block's
/// three triplet_stride-word sub-blocks, leaving the alpha and beta holes for
/// _receive_ab() to fill.
///
///   iter (8)          token           stride arena_block
///   dim2 (v heads)    head            stride L*arena_block
///   dim1 (3)          k | v | q       stride triplet_stride
///   dim0 (head dim)   channel         stride 1
///
/// IT[6] only ever sends q4_k weights, so both of its S2MM channels are idle.
void linear_decoding_layer::_receive_arena(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        IT6,
        bd_4,
        it_channel_0,
        {0, 0, 0, (uint32_t)arena_offset},
        {(uint32_t)L_TOKENS, (uint32_t)linear_num_v_heads, 3, (uint32_t)linear_v_head_dim},
        {(uint32_t)arena_block, (uint32_t)(L_TOKENS * arena_block), (uint32_t)triplet_stride, 1},
        -1, 0, true
    );
}

/// Broadcast the finished arena to the four delta cores.
///
/// Linear, because the arena is head-major and the cores want it head-major:
/// heads 0 .. linear_num_v_heads are one contiguous run, and inside it every
/// triplet_stride-word sub-block is exactly one of the cores' 130-word k, v or
/// q BDs. The padded head slots above linear_num_v_heads carry only alpha and
/// beta and are simply not read.
void linear_decoding_layer::_send_arena_to_delta(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        IT4,
        bd_4,
        it_channel_1,
        {0, 0, 0, (uint32_t)arena_offset},
        {1, 1, 1, (uint32_t)(static_cast<size_t>(linear_num_v_heads) * L_TOKENS * arena_block)},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

/// h2 = h1 + mlp.down_proj(hid), the layer's output and the next layer's input.
///
/// Flat: this comes off the rms tile, one packet per token in token order, not
/// off the engine's y tree, so there is no row-block permute to undo -- MT[3]
/// already did that for down's y on its way in. Split 5 x 1024 only because a BD
/// dimension caps at 1024 after the bf16 fold.
///
/// Its own channel rather than a second BD behind the norm's flow, which MT[2]
/// takes as up|gate's x. The rms tile stamps a distinct packet id per phase, so
/// the router keeps the two apart; see rms_pkt_id in iron/constants.py.
void linear_decoding_layer::_receive_layer_out(npu_sequence* seq){
    static_assert(L_TOKENS < 64, "iter_size is 6 bits");
    assert(D % 1024 == 0);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        IT3,
        bd_9,
        it_channel_1,
        {0, 0, 0, (uint32_t)layer_out_offset},
        {(uint32_t)L_TOKENS, 1, (uint32_t)(D / 1024), 1024},
        {(uint32_t)D, 0, 1024, 1},
        -1, 0, true
    );
}

/// The conv1d taps and the gated norm's weight, in one transfer.
///
/// First on IT5 MM2S ch1: the memtile's BD chain there is conv taps, then
/// ssm_norm, then the conv cache, cycling. Because aux_w_buffer stores
/// conv_w and ssm_norm adjacently, the two memtile BDs can be fed by a single
/// contiguous conv_w_size + linear_v_head_dim send -- no second shim channel
/// and no second helper. MT[5] then replays the 128 ssm_norm words to the norm
/// tile once per token, which is what that kernel expects (it re-acquires its
/// weight lock every invocation).
void linear_decoding_layer::_send_linear_conv_weights(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        aux_w_arg_id,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)(conv_w_size + linear_v_head_dim)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

/// Every accept length's conv window for the next pass, off MT[5]'s MM2S ch4.
///
/// Pre-conv qkv rows 1 .. L + CONV_SIZE - 2 of the tile's staging buffer, one
/// contiguous run. Each accept length's window is a CONV_SIZE - 1 row slice of
/// it, in the same layout _move_linear_kv_cache() sends the cache back in with;
/// get_conv_out() does the indexing.
///
/// IT5 S2MM ch0 was the one free channel on this shim: ch1 is the a/b scatter
/// into the arena. bd_14 because BD ids are a single per-shim pool across both
/// directions, and IT5 already spends 2, 11, 12 and 13 on the MM2S side.
///
/// Posted with the other reads, before any weights move -- the memtile will not
/// release the window until the L token rows are in, and a read issued late
/// would stall the tile on a full channel.
void linear_decoding_layer::_receive_linear_kv_cache(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        S2MM,
        IT5,
        bd_14,
        it_channel_0,
        {0, 0, 0, (uint32_t)conv_out_offset},
        {1, 1, 1, (uint32_t)conv_out_words},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
}

void linear_decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    const int m = 32;
    const int k = 256;
    const uint32_t columns = 4;
    const uint32_t elems = m * k;
    const uint32_t a_block_size = q4k_bytes(elems) / sizeof(bf16); // W_BLOCK_SIZE
    const uint32_t blocks_per_row = Din / k;
    const uint32_t cores = columns * 4;
    assert(Dout / m / cores > 0);
    assert(Dout % (m * cores) == 0);
    const size_t rounds = Dout / m / cores;
    for (size_t round = 0; round < rounds; round++){
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
    

/// One token's whole a/b weight set, bias block first.
///
/// `w_offset` is w_ab_bias_offset, not w_ab_offset: the leading block carries
/// the activation bias and the tile takes it through the same ping/pong BDs as
/// everything else. That is what makes the design re-dispatchable -- the chain
/// on this channel has no prologue BD any more, so every pass puts exactly
/// 1 + total_blocks blocks on it and the ping/pong parity never drifts. A
/// sequence that sent the bias once, ahead of the loop, worked for exactly one
/// dispatch per xclbin load.
void linear_decoding_layer::_move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
    // currently, only on DMA channel 0 of IT5
    const int m = 32;
    const int k = 256;
    constexpr uint32_t block_size = q80_bytes(m * k) / sizeof(bf16); // AB_W_BLOCK_SIZE
    const uint32_t total_blocks = 1 + (uint32_t)((Dout / m) * (Din / k));
    const uint32_t data2move = block_size * total_blocks;
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        proj_arg_id,
        MM2S,
        IT5,
        npu_bd_id(bd_13),
        it_channel_0,
        {0, 0, 0, (uint32_t)w_offset},
        {1, 1, 1, (uint32_t)data2move},
        {0, 0, 0, 1},
        -1, 0, true, aggressive_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_0
    );
}

void linear_decoding_layer::_move_linear_kv_cache(npu_sequence* seq){
    const uint32_t rows = (uint32_t)(conv_kernel_size - 1);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        IT5,
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {1, 1, 1, rows * (uint32_t)(Linear_DQ + Linear_DK + Linear_DV)},
        {0, 0, 0, 1},
        -1, 0, true, no_cache
    );
    seq->npu_dma_wait(
        IT5,
        MM2S,
        it_channel_1
    );
}

/// The recurrent state in, one linear 3 MB run. There is no run back.
///
/// DDR holds it [head][dk][dv] f32, which is both MT[4]'s per-head block layout
/// and cpu_ref's, so neither side needs an access pattern.
///
/// The write-back used to go to its own s_out region on IT4 S2MM ch0. It is
/// gone: only the accept-all state was ever recoverable that way, and
/// speculation needs the state at whichever prefix verification accepts, which
/// the host replays from the arena instead. No token is issued and nothing
/// waits on this leg -- the barrier that used to sit on the S2MM half is now
/// the h2 wait at the end of the sequence.
void linear_decoding_layer::_move_s(npu_sequence* seq){
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        kv_cache_arg_id,
        MM2S,
        IT4,
        bd_2,
        it_channel_0,
        {0, 0, 0, (uint32_t)s_offset},
        {1, 1, 1, (uint32_t)s_words},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

