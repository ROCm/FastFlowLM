// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "decoding_layer.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

void decoding_layer::_load_rtp_addresses() {
    const std::string path = "build/QWEN3_8_27B/xclbins/rtp_addresses.txt";
    std::ifstream f(path);
    if (!f) {
        header_print("error", "cannot open " << path
                     << " -- run `make bitstream` first; the attention cores' L "
                        "addresses come from it and are per-build");
        std::abort();
    }

    std::string name;
    uint32_t addr;
    int n_qk = 0, n_kv = 0;
    while (f >> name >> addr) {
        uint32_t* slot = nullptr;
        if (name.rfind("RTP_L_attn_qk_core_", 0) == 0) { slot = &l_qk_address; n_qk++; }
        else if (name.rfind("RTP_L_attn_kv_core_", 0) == 0) { slot = &l_kv_address; n_kv++; }
        else continue;

        if (*slot != 0 && *slot != addr) {
            header_print("error", name << " is at " << addr << " but a sibling core is at "
                         << *slot << " -- the cores are no longer one tile layout, so a "
                            "single address per kind is wrong");
            std::abort();
        }
        *slot = addr;
    }

    if (n_qk != 4 || n_kv != 4 || l_qk_address == 0 || l_kv_address == 0) {
        header_print("error", "found " << n_qk << " qk and " << n_kv
                     << " attention L buffers in " << path << ", expected 4 and 4");
        std::abort();
    }
    header_print("info", "  attention L RTP: qk @ " << l_qk_address
                 << ", kv @ " << l_kv_address);
}

void decoding_layer::set_context_offset(int n) {
    assert(n >= 0 && n + L_TOKENS <= MAX_L);
    // Any offset works. Only the *replay* is chunked: it always starts at row 0
    // and rounds its length up to a whole number of L_CHUNKs, so a ragged
    // context just means the final chunk is partly padding. The append writes at
    // the exact offset, and the kernels count keys absolutely from row 0, so
    // nothing downstream cares whether that offset is aligned.
    context_offset = n;
    // The sequence bakes in both the key counts and the replay length, so it
    // has to be rebuilt whenever the context length changes.
    this->_gen_sequence(this->layer_app.seq());
}

void decoding_layer::seed_kv_cache(const std::vector<float>& k,
                                       const std::vector<float>& v, int n) {
    assert((int)k.size() >= n * DK && (int)v.size() >= n * DV);
    assert(n >= 0 && n <= MAX_L);
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

decoding_layer::decoding_layer(model_list::model_t model_type, LM_Config& lm_config,
                                       npu_app_manager* layer_app_mgr)
    : model_type(model_type), lm_config(lm_config), layer_app_mgr(layer_app_mgr) {
    header_print("info", "Initializing qkv decoding layer");

    this->layer_app = this->layer_app_mgr->create_app();

    num_hidden_layers = lm_config.num_hidden_layers;
    D  = lm_config.hidden_size;
    DH = lm_config.head_dim;
    DQ = lm_config.head_dim * lm_config.num_attention_heads;
    DK = lm_config.head_dim * lm_config.num_key_value_heads;
    DV = lm_config.head_dim * lm_config.num_key_value_heads;
    num_kv_heads = lm_config.num_key_value_heads;
    qkv_out = DQ + DK + DV;
    intermediate_size = lm_config.intermediate_size;
    // The GLU tile takes one engine round at a time, and a round has to be all
    // up or all gate -- which is what the host's weave below assumes.
    assert(intermediate_size % 512 == 0 &&
           "a projection round is 512 output rows; up|gate is woven at that grain");

    // What the RoPE tile actually emits. 24 q heads over 4 kv heads is 6 per
    // group, padded to 8 for the 1x8x1 attention CU, so q comes out a third
    // wider than DQ with the tail of each group zeroed. Must match DQ_PADDED in
    // iron/RoPE.py, which computes it the same way from q_heads_padding.
    q_heads_per_group = lm_config.num_attention_heads / num_kv_heads;
    const int q_heads_padding = 8;
    DQ_PADDED = (q_heads_per_group + q_heads_padding - 1) / q_heads_padding
                * q_heads_padding * num_kv_heads * DH;

    // head_dim * rope_partial_factor. 0.25 for this model, so 64 of 256 dims
    // rotate and the rest pass through.
    ROPE_DIM_ = DH / 4;

    header_print("info", "qkv layer config: ");
    header_print("info", "  D  : " << D);
    header_print("info", "  DQ : " << DQ << " (padded " << DQ_PADDED << ")");
    header_print("info", "  DK : " << DK);
    header_print("info", "  DV : " << DV);
    header_print("info", "  qkv_out : " << qkv_out);
    header_print("info", "  ROPE_DIM: " << ROPE_DIM_);
    header_print("info", "  layer   : " << layer_idx);

    // The engine partitions output rows 512 at a time (16 cores x 32 rows), and
    // the weight DDR layout pairs adjacent row blocks. q, k and v are each a
    // whole number of both, so concatenating three separately-reordered
    // matrices is byte for byte one reorder of the fused matrix -- which is
    // what lets load_weights() below treat them as three tensors and the array
    // treat them as one projection.
    assert(DQ % 512 == 0 && DK % 512 == 0 && DV % 512 == 0);

    // How the attention is partitioned: one kv head per CU, two CUs per column,
    // two columns. Has to agree with iron/model_config.py's kv_heads_per_cu and
    // cus_per_column, which is what iron/attn.py builds the BDs from.
    cus_per_column     = 2;
    kv_heads_per_column = cus_per_column;                                // 1 head per CU
    attn_col_width     = cus_per_column * q_heads_per_group * DH;        // half a row
    assert(num_kv_heads == 4 && "two columns x two CUs x one kv head");
    assert(2 * attn_col_width == DQ);

    // | x | o | down | h2 |, one buffer: the sequence reaches every region
    // through x_arg_id. h2 is what the layer produces; o and down are the two
    // diagnostics still read back, each the direct control for the stage after
    // it. All three are D wide.
    o_offset         = static_cast<size_t>(L_TOKENS) * D;
    // Adjacent to o and in that order: both rounds pass through the same
    // memtile buffer and leave on one channel, so _receive_o() takes them as a
    // single contiguous transfer.
    down_offset      = o_offset + static_cast<size_t>(L_TOKENS) * D;
    h2_offset        = down_offset + static_cast<size_t>(L_TOKENS) * D;
    this->hidden_states_buffer = this->layer_app.create_bo_buffer<bf16>(
        h2_offset + static_cast<size_t>(L_TOKENS) * D);
    this->hidden_states_buffer.memset((bf16)0);

    this->_load_rtp_addresses();

    // Two norms, so two weight vectors: input_layernorm then
    // post_attention_layernorm. The rms tile has one D-sized w buffer and its
    // self-chained BD reloads it between the two phases, which is the only
    // reason a four-bank tile can run both. The length has to track the live
    // phase count exactly -- too few and phase 2 parks on the weight acquire,
    // too many and the surplus desynchronises the next invocation.
    this->rms_weights_buffer = this->layer_app.create_bo_buffer<bf16>(2 * D);

    // The fused q | k | v matrix, the output gate z, o_proj, the fused
    // up | gate and down. Five rounds, streamed in this order because that is
    // the order proj_main() consumes them.
    //
    // o_proj is D x DQ, not DQ x D: it is the one projection in the layer whose
    // K is not the model dimension. The byte count is the same either way, but
    // the reorder and the descriptor below both need the right one.
    w_z_offset     = q4k_bytes(static_cast<size_t>(qkv_out) * D);
    w_o_offset     = w_z_offset + q4k_bytes(static_cast<size_t>(DQ) * D);
    w_upgate_offset = w_o_offset + q4k_bytes(static_cast<size_t>(D) * DQ);
    w_down_offset  = w_upgate_offset
                   + q4k_bytes(2 * static_cast<size_t>(intermediate_size) * D);
    this->all_proj_weight_buffer = this->layer_app.create_bo_buffer<u8>(
        w_down_offset + q4k_bytes(static_cast<size_t>(D) * intermediate_size));

    // The RoPE weight block: one cos|sin pair per token, then the two per-head
    // norm weights. The layout is the ABI between _set_rope_weights below,
    // iron/RoPE.py's buffer type and kernels/rope.cc's pointer arithmetic --
    // three files, no cross-check.
    //
    //   [t * ROPE_DIM              .. + ROPE_DIM/2)   cos for token t
    //   [t * ROPE_DIM + ROPE_DIM/2 .. + ROPE_DIM/2)   sin for token t
    //   [L_TOKENS * ROPE_DIM       .. + DH)           q_norm.weight
    //   [L_TOKENS * ROPE_DIM + DH  .. + DH)           k_norm.weight
    rope_w_offset = static_cast<size_t>(L_TOKENS) * ROPE_DIM_;
    this->rope_weights_buffer = this->layer_app.create_bo_buffer<bf16>(
        rope_w_offset + 2 * static_cast<size_t>(DH));
    this->rope_weights_buffer.memset((bf16)0);

    // **MAX_L is a test knob here, not a model constant.** The bench drives 8
    // rows at a small offset, so 256 is plenty for scoring; what raising it
    // buys is coverage of the cache's *addressing*. The append used to cross
    // between the planes inside one BD with a stride of MAX_L * DK, which is a
    // 20-bit field that masked rather than faulted above 2048 -- so a run at
    // 4096 or 16384 that still scores is the evidence that splitting k and v
    // onto two shim channels removed that ceiling instead of moving it. Under
    // the old descriptor the v plane would land at a wrapped address and the v
    // readback would collapse, while h2 stayed clean.
    if (const char* e = std::getenv("FLM_MAX_L")) {
        const int n = std::atoi(e);
        if (n > 0) {
            MAX_L = n;
            header_print("info", "MAX_L overridden to " << MAX_L << " (FLM_MAX_L)");
        }
    }
    // The one limit the layout still has: the replay walks `chunks` on a 10-bit
    // BD dimension. Unlike the old append stride, exceeding this is asserted
    // rather than masked -- see _replay_kv_cache.
    assert(MAX_L <= 1024 * L_CHUNK && "the replay's chunk count is 10 bits");

    // Two contiguous planes: [MAX_L][DK] token-major, then [MAX_L][DV]. See
    // decoding_layer.hpp::kv_cache_size. This used to carry a MAX_L <= 2048
    // ceiling, from the append crossing between the planes inside one BD; k and
    // v land on separate shim channels now and it does not. The array side does
    // not care either way -- the memtile's k_order/v_order see two separate
    // streams regardless, and only the shim descriptors know where in DDR those
    // streams come from.
    kv_cache_size = static_cast<size_t>(MAX_L) * (DK + DV);
    this->kv_cache = this->layer_app.create_bo_buffer<bf16>(kv_cache_size);
    this->kv_cache.memset((bf16)0);
    this->kv_cache.sync_to_device();

    this->_gen_sequence(this->layer_app.seq());
}

/// The per-token rotary table, from the reference dump rather than from
/// cos(inv_freq * pos) on the host.
///
/// The dumped rows are ROPE_DIM wide with the two halves identical -- HF's
/// rotate_half pairs dim j with dim j + ROPE_DIM/2 and gives both the same
/// angle -- and the kernel wants the angle once, so only the first half of each
/// row is taken. Building them from `inv_freq` and an absolute position would
/// work too and is what the L = 1 driver did; taking them from the dump removes
/// the position bookkeeping and the rope_scaling lookup from the comparison.
void decoding_layer::set_rope_table(const std::vector<float>& cos,
                                        const std::vector<float>& sin) {
    const size_t half = static_cast<size_t>(ROPE_DIM_) / 2;
    const size_t need = static_cast<size_t>(L_TOKENS) * ROPE_DIM_;
    assert(cos.size() >= need && sin.size() >= need);

    bf16* p = this->rope_weights_buffer.data();
    for (int t = 0; t < L_TOKENS; t++) {
        const float* c = cos.data() + static_cast<size_t>(t) * ROPE_DIM_;
        const float* s = sin.data() + static_cast<size_t>(t) * ROPE_DIM_;
        bf16* dst = p + static_cast<size_t>(t) * ROPE_DIM_;
        for (size_t j = 0; j < half; j++) {
            dst[j]        = (bf16)c[j];
            dst[half + j] = (bf16)s[j];
        }
    }
    this->rope_weights_buffer.sync_to_device();
}

void decoding_layer::reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
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

void decoding_layer::load_weights(Q4NX& q4nx) {
    const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    {
        // q, then k, then v, laid down contiguously as the one fused matrix the
        // engine reduces over.
        //
        // self_attn.q_proj.weight is 2 * DQ rows, not DQ: rows [0, DQ) are q and
        // rows [DQ, 2 * DQ) are the output gate z. HF interleaves the two per
        // head and the q4nx converter de-interleaves them, which is what lets
        // the array split the matrix between two mvms. Because q4nx blocks are
        // row-block major, the split is a plain byte offset: q is the first
        // q4k_bytes(DQ * D) bytes of the tensor and z is the second.
        //
        // They are laid down as two separate rounds -- q | k | v, then z --
        // because that is the order proj_main() runs its mvms and
        // _move_weights() streams them positionally. z is *not* contiguous with
        // q here, even though it is in the checkpoint.
        LOG_VERBOSE(1, "Loading q/k/v and output-gate weights");
        buffer<u8> w_q, w_k, w_v;
        q4nx.load_weights(w_q, prefix + "self_attn.q_proj.weight");
        q4nx.load_weights(w_k, prefix + "self_attn.k_proj.weight");
        q4nx.load_weights(w_v, prefix + "self_attn.v_proj.weight");

        const size_t q_bytes = q4k_bytes(static_cast<size_t>(DQ) * D);
        const size_t k_bytes = q4k_bytes(static_cast<size_t>(DK) * D);
        const size_t v_bytes = q4k_bytes(static_cast<size_t>(DV) * D);
        assert(w_q.size() == 2 * q_bytes);
        assert(w_k.size() == k_bytes);
        assert(w_v.size() == v_bytes);
        assert(w_z_offset == q_bytes + k_bytes + v_bytes);

        buffer<u8> w_q_only(w_q.data(), q_bytes);
        buffer<u8> w_z_only(w_q.data() + q_bytes, q_bytes);
        u8* dst = this->all_proj_weight_buffer.data();
        LOG_VERBOSE(1, "Copying q/k/v and output-gate weights to buffer");
        reorder_cpy(dst, w_q_only, D);
        reorder_cpy(dst + q_bytes, w_k, D);
        reorder_cpy(dst + q_bytes + k_bytes, w_v, D);
        reorder_cpy(dst + w_z_offset, w_z_only, D);
    }
    {
        // o_proj, the third round. **The reorder's column count is DQ, not D**:
        // it walks the matrix in 32 x 256 blocks and has to know how many of
        // them make a row, and o_proj reduces over the gated rows rather than
        // over the hidden state. Passing D here would stride the wrong way
        // through a matrix that is the same number of bytes, so nothing would
        // assert -- it would just dequantise into the wrong rows.
        LOG_VERBOSE(1, "Loading o_proj weights");
        buffer<u8> w_o;
        q4nx.load_weights(w_o, prefix + "self_attn.o_proj.weight");
        assert(w_o.size() == q4k_bytes(static_cast<size_t>(D) * DQ));
        reorder_cpy(this->all_proj_weight_buffer.data() + w_o_offset, w_o, DQ);
    }
    {
        // mlp.up_proj and mlp.gate_proj, **woven M_PER_ROUND output rows at a
        // time rather than concatenated**. The GLU tile takes one engine round
        // at a time and needs a round to be all up or all gate, so the two
        // alternate: round 2p is up, round 2p+1 is gate, both covering
        // intermediate columns [p * 512, (p+1) * 512).
        //
        // 512 rows is an even number of the engine's 32-row blocks, so this is
        // byte for byte what one reorder_cpy() over the woven matrix would
        // produce -- the weave and the row-block pairing do not interfere.
        // host.cpp's golden weaves the reference the same way.
        LOG_VERBOSE(1, "Loading up/gate weights");
        buffer<u8> w_up, w_gate;
        q4nx.load_weights(w_up,   prefix + "mlp.up_proj.weight");
        q4nx.load_weights(w_gate, prefix + "mlp.gate_proj.weight");

        const int blocks     = 1024 / 2 / 32 * (D / 256);
        const int chunk_size = blocks * q4k_bytes(32 * 256);
        const int total_size = q4k_bytes(static_cast<size_t>(intermediate_size) * D);
        const int phases     = total_size / chunk_size;
        assert(static_cast<size_t>(phases) * chunk_size == static_cast<size_t>(total_size));
        assert(w_up.size() == static_cast<size_t>(total_size));
        assert(w_gate.size() == static_cast<size_t>(total_size));
        tensor_2d<u8> w_up_tensor(w_up, chunk_size, 0);
        tensor_2d<u8> w_gate_tensor(w_gate, chunk_size, 0);
        u8* w_ptr = this->all_proj_weight_buffer.data() + w_upgate_offset;
        for (int i = 0; i < phases; i++) {
            reorder_cpy(w_ptr, w_up_tensor[i], D);
            w_ptr += chunk_size;
            reorder_cpy(w_ptr, w_gate_tensor[i], D);
            w_ptr += chunk_size;
        }
    }
    {
        // mlp.down_proj, D x intermediate_size. Like o_proj, its K is not D --
        // hence the intermediate_size column count handed to reorder_cpy.
        LOG_VERBOSE(1, "Loading down weights");
        buffer<u8> w_down;
        q4nx.load_weights(w_down, prefix + "mlp.down_proj.weight");
        assert(w_down.size() == q4k_bytes(static_cast<size_t>(D) * intermediate_size));
        reorder_cpy(this->all_proj_weight_buffer.data() + w_down_offset,
                    w_down, intermediate_size);
    }
    this->all_proj_weight_buffer.sync_to_device();
    {
        // Both norms, in the order the tile consumes them -- the single w
        // buffer is reloaded between the two phases, so this is a stream and
        // not two addressable regions.
        LOG_VERBOSE(1, "Loading rms weights");
        buffer<bf16> w_rms_pre, w_rms_post;
        q4nx.load_weights(w_rms_pre, prefix + "input_layernorm.weight");
        q4nx.load_weights(w_rms_post, prefix + "post_attention_layernorm.weight");
        assert(w_rms_pre.size() >= static_cast<size_t>(D));
        assert(w_rms_post.size() >= static_cast<size_t>(D));
        memcpy(this->rms_weights_buffer.data(), w_rms_pre.data(), D * sizeof(bf16));
        memcpy(this->rms_weights_buffer.data() + D, w_rms_post.data(), D * sizeof(bf16));
    }
    this->rms_weights_buffer.sync_to_device();
    {
        // The two per-head norm weights, into the tail of the RoPE block. The
        // cos/sin half in front of them is set separately, per pass, by
        // set_rope_table() -- these do not change between passes.
        LOG_VERBOSE(1, "Loading q/k norm weights");
        buffer<bf16> q_attn_norm, k_attn_norm;
        q4nx.load_weights(q_attn_norm, prefix + "self_attn.q_norm.weight");
        q4nx.load_weights(k_attn_norm, prefix + "self_attn.k_norm.weight");
        assert(q_attn_norm.size() >= static_cast<size_t>(DH));
        assert(k_attn_norm.size() >= static_cast<size_t>(DH));
        memcpy(this->rope_weights_buffer.data() + rope_w_offset,
               q_attn_norm.data(), DH * sizeof(bf16));
        memcpy(this->rope_weights_buffer.data() + rope_w_offset + DH,
               k_attn_norm.data(), DH * sizeof(bf16));
    }
    this->rope_weights_buffer.sync_to_device();
    header_print("info", "Weights loaded!");
}

/// One pass over L_TOKENS rows. `input` must hold at least L_TOKENS * D
/// elements; the return is h2, the layer output, L_TOKENS x D in token order.
/// k and v land in the KV cache -- see get_k() / get_v() -- and nothing else
/// leaves the array.
buffer<bf16> decoding_layer::operator()(buffer<bf16>& input) {
    const size_t in_size = static_cast<size_t>(L_TOKENS) * D;
    assert(input.size() >= in_size);
    memcpy(this->hidden_states_buffer.data(), input.data(), in_size * sizeof(bf16));
    this->hidden_states_buffer.sync_to_device();
    // The dispatch state is CHECKED. On a timeout hidden_states_buffer still
    // holds the previous pass's output, so an unchecked call reads back a
    // plausible row -- and the repeat probe's "bit-identical" test passes
    // BECAUSE nothing was written. A timeout has to be an error, not a pass.
    const ert_cmd_state st =
        this->layer_app(hidden_states_buffer, all_proj_weight_buffer,
                        rms_weights_buffer, rope_weights_buffer, kv_cache);
    if (st != ERT_CMD_STATE_COMPLETED) {
        header_print("error", "layer dispatch state " << (int)st
                     << " at context offset " << this->context_offset);
        exit(1);
    }
    this->hidden_states_buffer.sync_from_device();

    buffer<bf16> output(static_cast<size_t>(L_TOKENS) * D);
    memcpy(output.data(), this->hidden_states_buffer.data() + h2_offset,
           output.size() * sizeof(bf16));
    return output;
}

/// o_proj's output for the pass just run, L_TOKENS x D, token-major. The last
/// diagnostic before the residual. Valid after operator().
buffer<bf16> decoding_layer::get_o() {
    buffer<bf16> out(static_cast<size_t>(L_TOKENS) * D);
    memcpy(out.data(), this->hidden_states_buffer.data() + o_offset,
           out.size() * sizeof(bf16));
    return out;
}

/// down_proj's output for the pass just run, L_TOKENS x D, token-major. The last
/// diagnostic before the MLP residual. Valid after operator().
buffer<bf16> decoding_layer::get_down() {
    buffer<bf16> out(static_cast<size_t>(L_TOKENS) * D);
    memcpy(out.data(), this->hidden_states_buffer.data() + down_offset,
           out.size() * sizeof(bf16));
    return out;
}

/// h2 = h1 + down_proj(SwiGLU), L_TOKENS x D, token-major: the layer's output
/// and the next layer's input -- the end of what this xclbin computes. Valid
/// after operator().
buffer<bf16> decoding_layer::get_h2() {
    buffer<bf16> out(static_cast<size_t>(L_TOKENS) * D);
    memcpy(out.data(), this->hidden_states_buffer.data() + h2_offset,
           out.size() * sizeof(bf16));
    return out;
}

/// This pass's L_TOKENS rows of k, token-major and dense -- the plane shape the
/// CPU reference works in.
///
/// Written as a per-row copy rather than one memcpy. Under the two-plane layout
/// the rows *are* contiguous and the two forms agree, but the loop is what
/// survives a flip to an interleaved cache, where consecutive rows sit DK + DV
/// apart and a single memcpy would quietly return v bytes as k.
buffer<bf16> decoding_layer::get_k() {
    this->kv_cache.sync_from_device();
    buffer<bf16> out(static_cast<size_t>(L_TOKENS) * DK);
    for (int t = 0; t < L_TOKENS; t++)
        memcpy(out.data() + static_cast<size_t>(t) * DK,
               this->kv_cache.data() + this->k_at(context_offset + t),
               static_cast<size_t>(DK) * sizeof(bf16));
    return out;
}

/// As get_k, from the v plane. See its note on why this is a loop.
buffer<bf16> decoding_layer::get_v() {
    this->kv_cache.sync_from_device();
    buffer<bf16> out(static_cast<size_t>(L_TOKENS) * DV);
    for (int t = 0; t < L_TOKENS; t++)
        memcpy(out.data() + static_cast<size_t>(t) * DV,
               this->kv_cache.data() + this->v_at(context_offset + t),
               static_cast<size_t>(DV) * sizeof(bf16));
    return out;
}

float decoding_layer::bench(buffer<bf16>& input) {
    memcpy(this->hidden_states_buffer.data(), input.data(),
           static_cast<size_t>(L_TOKENS) * D * sizeof(bf16));
    xrt::runlist rl = this->layer_app_mgr->create_runlist();
    for (int i = 0; i < num_hidden_layers; i++){
        rl.add(this->layer_app.create_run(hidden_states_buffer, all_proj_weight_buffer,
                                          rms_weights_buffer, rope_weights_buffer, kv_cache));
    }
    time_utils::time_point start = time_utils::now();
    rl.execute();
    rl.wait();
    this->hidden_states_buffer.sync_from_device();
    time_utils::time_point end = time_utils::now();
    time_utils::time_with_unit total_ms = time_utils::duration_ms(start, end);
    return 1000.0 / total_ms.first;
}

void decoding_layer::_gen_sequence(npu_sequence* seq) {
    seq->clear_cmds();
    
    _write_attn_L(seq);

    _send_hidden_states(seq);
    _send_rms_weights(seq);
    _send_rope_weights(seq);
    _receive_kv_cache(seq);
    // _receive_o(seq);
    _receive_h2(seq);
    _move_weights(seq, qkv_out, D, 0);
    // **In bf16 elements, not bytes.** _move_weights builds its descriptor with
    // an element size of sizeof(bf16) and its own a_block_size is likewise
    // divided down, so the offset has to be in the same unit as everything else
    // in that arithmetic. Passing the byte offset addresses twice as far into
    // the buffer and streams whatever is past the end -- which the cores
    // dequantise into NaN rather than rejecting.
    _move_weights(seq, DQ, D, w_z_offset / sizeof(bf16));

    // The append, waited on per column **and per plane**. This only orders the
    // replay because both are on these same two shims -- npu_dma_wait does not
    // reach across shim tiles, which is why the k | v sink was moved here from
    // IT3. Everything after this point may assume the cache holds
    // context_offset + L_TOKENS valid rows.
    //
    // Two channels each now: k lands on S2MM ch0 and v on ch1 (iron/RoPE.py
    // stamps them with different packet ids). Waiting only on ch0 would order
    // the replay against the k plane alone and leave the v plane racing -- the
    // same class of bug as waiting on the wrong shim, and just as quiet, since
    // the CUs use the v they computed rather than the v they read back.
    for (int c = 0; c < 2; c++) {
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_0);   // k
        seq->npu_dma_wait(attn_col_tiles[c], S2MM, it_channel_1);   // v
    }

    _replay_kv_cache(seq);

    // o_proj's weights, and **not with the other two rounds**. _move_weights
    // waits on its own transfers between rounds, and the cores cannot touch
    // o's weights until the gate has filled o's replay phase -- which needs the
    // attention output, which needs the replay above. Issuing this earlier is a
    // cycle: the host would block on a weight round the array cannot consume
    // yet, before posting the replay the array is waiting for.
    //
    // **The offset is in bf16 elements, not bytes** -- see the note on z's
    // round above. Dout is D and Din is DQ: K is the gated row here.
    _move_weights(seq, D, DQ, w_o_offset / sizeof(bf16));

    // The MLP: up | gate off the post-attention norm, then down off the SwiGLU.
    // Fourth and fifth rounds, so fourth and fifth mvm(). Both are behind
    // o_proj's round for the same reason it was behind the replay -- the x each
    // reduces over does not exist until the stage before it has run.
    _move_weights(seq, 2 * intermediate_size, D, w_upgate_offset / sizeof(bf16));
    _move_weights(seq, D, intermediate_size, w_down_offset / sizeof(bf16));

   
    seq->npu_dma_wait(xyr_tile, S2MM, it_channel_1);

    seq->cmds2seq();
}

void decoding_layer::_send_hidden_states(npu_sequence* seq){
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

void decoding_layer::_send_rms_weights(npu_sequence* seq){
    // Both norm weights as one stream: input_layernorm, then
    // post_attention_layernorm overwriting it in the tile's single w buffer.
    // The kernel acquires the weight lock once per *norm*, outside each row
    // loop, so the self-chained w BD fires exactly twice per invocation -- this
    // length has to track the live phase count, or the surplus sits in the
    // channel unconsumed and desynchronises the next pass.
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

/// The RoPE weight block: one cos|sin pair per token, then q_norm and k_norm.
///
/// Once per invocation, not once per token -- the kernel acquires it outside its
/// token loop and indexes the pair it wants. Linear: the host already built it in
/// the order the kernel reads. `no_cache`, because the cos/sin half is rewritten
/// between passes whenever the positions change.
void decoding_layer::_send_rope_weights(npu_sequence* seq){
    const uint32_t len = (uint32_t)(rope_w_offset + 2 * (size_t)DH);
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        rope_arg_id,
        MM2S,
        attn_tile,
        bd_3,
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, len},
        {0, 0, 0, 1},
        -1, 0, false, no_cache
    );
}

/// The pass's key count, to all eight attention cores. One word each.
///
/// Token t attends to context_offset + t + 1 keys: everything already in the
/// cache, plus the tokens of this pass up to and including itself. That is the
/// causal mask, in the only form the cores can see it -- but it is an
/// arithmetic sequence, so **the last token's count is the whole of it**. The
/// cores keep their own token counter (attn_qk.cc) and take
/// keys = L_max - (L_TOKENS - 1) + t. This used to be an L_TOKENS-entry RTP
/// burst per core, 64 writes a pass, of which 56 carried no information.
///
/// The same word is the chunk count both cores walk, ceil(L_max / L_CHUNK), and
/// it is what _replay_kv_cache() sizes its transfers from -- see the assert
/// there. If the two disagree the run does not mis-compare, it corrupts: the
/// surplus chunks stay queued in the memtile for the next token.
void decoding_layer::_write_attn_L(npu_sequence* seq){
    const uint32_t l_max = (uint32_t)(context_offset + L_TOKENS);
    for (int i = 0; i < 4; i++) {
        seq->rtp_write(attn_qk_tiles[i], l_qk_address, l_max);
        seq->rtp_write(attn_kv_tiles[i], l_kv_address, l_max);
    }
}

/// The whole context out of the DDR cache and into the two attention columns,
/// in chunks of L_CHUNK keys.
///
/// Each column takes the two kv heads it owns, and its memtile splits those
/// again per CU. The cache is token-major over all four heads:
///
///     k[t * DK + head * DH .. + DH)
///
/// so a column's slice is a strided walk -- kv_heads_per_column * DH out of
/// every DK -- with column c starting at c * kv_heads_per_column * DH.
///
/// Every token re-reads the whole context, and **that repeat belongs in the
/// iteration dimension** -- size[0], the first of the four. It is the axis the
/// hardware provides for exactly this, a BD replayed from the same address, and
/// its counter is 6 bits, so the repeat count has to stay under 64. L_TOKENS = 8
/// is comfortably inside that.
///
/// It cannot be an ordinary axis with a zero stride. npu_dma_memcpy_nd reads a
/// zero stride as "this axis does not advance" and collapses it
/// (`iter_size = (stride[0] == 0) ? 1 : size[0]`), and the iteration count is
/// deliberately left out of buffer_length, so writing the repeat as size L with
/// stride 0 on dim1 sends the context once and the cores wait forever for the
/// rest. That is what hung the first version of this: the run sat until XRT
/// gave up after 63 s -- against 6 ms for the same design without attention --
/// and the output region was never written at all.
///
/// The stride that goes with it is 0: every replay starts from the same place.
///
/// Each pass is sized to the *longest* token, L[L_TOKENS - 1], which is why the
/// repeat is per token rather than one long stream. **Every token therefore
/// receives the same number of chunks, and both kernels consume that same
/// number** -- they derive it from L[L_TOKENS - 1] exactly as `chunks` is
/// derived here, and mask the keys past their own L[t] rather than stopping
/// early. A kernel that stopped early would leave its surplus chunks queued in
/// the memtile for the next token to eat; that was a real bug, and the assert
/// below is the contract iron/attn.py's attn_memtile refers to.
///
///   dim0 (L_TOKENS) the repeat, carried by the queue push
///   dim1 (chunks)   successive L_CHUNKs of context, stride L_CHUNK * row stride
///   dim2 (L_CHUNK)  one chunk's context rows, stride one row
///   dim3 (width)    the column's two heads, contiguous
///
/// Each plane is walked on its own descriptor, so the row stride is just DK (or
/// DV); only the base offsets differ -- k_at(0) is 0 and v_at(0) is a whole k
/// plane in. The append is now the same shape (see _receive_kv_cache), which is
/// what removed the old MAX_L <= 2048 ceiling.
///
/// **So this function holds the remaining one.** dim1 carries `chunks` and is a
/// 10-bit field, which bounds the cache at 1024 * L_CHUNK = 16384 positions.
/// Unlike the stride that used to cap the append, this one is asserted below
/// and would be a loud failure rather than a masked address.
void decoding_layer::_replay_kv_cache(npu_sequence* seq){
    const uint32_t width  = (uint32_t)(kv_heads_per_column * DH);
    const uint32_t rows   = (uint32_t)(((context_offset + L_TOKENS) + L_CHUNK - 1)
                                       / L_CHUNK * L_CHUNK);
    const uint32_t chunks = rows / (uint32_t)L_CHUNK;
    const uint32_t k_row_stride   = (uint32_t)this->k_row_stride();
    const uint32_t v_row_stride   = (uint32_t)this->v_row_stride();
    const uint32_t k_chunk_stride = (uint32_t)(L_CHUNK * this->k_row_stride());
    const uint32_t v_chunk_stride = (uint32_t)(L_CHUNK * this->v_row_stride());
    assert(rows <= (uint32_t)MAX_L);
    assert(chunks <= 1024);   // dim1 is 10 bits
    assert(width <= 1023 * 2);   // 10-bit dims, doubled for bf16

    // **One pass over the whole context per token; the repeat covers the
    // tokens.** The memtile's inbound BD holds one L_CHUNK x kv_row buffer, so
    // it consumes a chunk per firing -- but the *descriptor* still has to walk
    // every chunk, which is what dim1 does. Describing only one chunk and
    // folding the chunk count into the repeat sends chunk 0 over and over and
    // never reaches row 16, because the repeat replays a BD from the same
    // address. That is invisible at context_offset = 0, where L_TOKENS = 8 rows
    // round up to exactly one chunk and the two forms coincide.
    //
    // The earlier failure this comment used to describe -- whole context in the
    // descriptor *and* the chunk count in the repeat -- was the other half of
    // the same confusion: it sent chunks times too much.
    assert((uint32_t)L_TOKENS < 64 && "the queue's repeat count is 6 bits");
    assert(width * L_CHUNK <= 1023 * 2 * 1024 && "dim sizes are 10 bits");

    // The count the kernels will walk: both attn_qk.cc and attn_kv.cc take
    // ceil(L[0] / L_CHUNK), and _write_attn_L writes L[0] = context_offset +
    // L_TOKENS. Sent and consumed have to be equal -- one chunk either way and
    // the memtile's queue and the cores' ping/pong parity are out of step for
    // the rest of the pass.
    const uint32_t kernel_chunks =
        ((uint32_t)(context_offset + L_TOKENS) + L_CHUNK - 1) / L_CHUNK;
    assert(chunks == kernel_chunks &&
           "the replay's chunk count must equal the kernels' -- see attn_qk.cc");
    (void)kernel_chunks;

    for (int c = 0; c < 2; c++) {
        const uint32_t head_off = (uint32_t)(c * kv_heads_per_column * DH);
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            MM2S,
            attn_col_tiles[c],
            bd_0,
            it_channel_0,
            {0, 0, 0, (uint32_t)this->k_at(0) + head_off},
            {(uint32_t)L_TOKENS, chunks,         (uint32_t)L_CHUNK, width},
            {0,                  k_chunk_stride, k_row_stride,      1},
            // no_cache: these are rows the append wrote earlier in this same
            // sequence, so a cached read can serve the pre-append contents.
            -1, 0, false, no_cache
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            MM2S,
            attn_col_tiles[c],
            bd_1,
            it_channel_1,
            {0, 0, 0, (uint32_t)this->v_at(0) + head_off},
            {(uint32_t)L_TOKENS, chunks,         (uint32_t)L_CHUNK, width},
            {0,                  v_chunk_stride, v_row_stride,      1},
            -1, 0, false, no_cache
        );
    }
}

// /// o_proj's output **and down_proj's**, 2 x L_TOKENS x D, token-major.
// ///
// /// Both pass through the same staging buffer on MT[3] and leave on the same
// /// channel, o first and down second, so this is one descriptor covering the two
// /// firings rather than two BDs -- which would be wrong as well as wasteful, since
// /// BDs on a channel consume the stream in sequence and the first would swallow
// /// part of o rather than all of it.
// ///
// /// The memtile un-permutes the engine's row-block order on the way in and sends
// /// each whole L x D buffer as a single BD, so what arrives here is already dense
// /// token-major. D = 5120 does not fit one 10-bit dimension once bf16 sizes are
// /// halved on lowering (the ceiling is 2048), so the row is split across two axes
// /// of this one descriptor, and the two rounds ride the outermost axis.
// ///
// /// o_offset and down_offset are adjacent and in that order, which is what lets
// /// the round axis be a plain stride of L_TOKENS * D.
// ///
// /// On IT3 and not IT2: the staging memtile is in column 3, and IT2's two S2MM
// /// channels carry the gated row and h2.
// void decoding_layer::_receive_o(npu_sequence* seq){
//     // 5120 = 4 x 1280; the outer of the two is contiguous with the inner, so
//     // together they are just "the whole row".
//     const uint32_t inner = (uint32_t)(D / 4);
//     const uint32_t rows  = (uint32_t)(L_TOKENS * 4);
//     assert((uint32_t)D % 4 == 0 && inner <= 2048 &&
//            "a BD dimension is 10 bits, and bf16 sizes are halved");
//     assert(down_offset == o_offset + (size_t)L_TOKENS * D &&
//            "the two rounds are one contiguous transfer, in arrival order");
//     assert(rows <= 1024);

//     // One flat run: o's L x D then down's, contiguous in both the stream and
//     // the buffer, so the token and round axes collapse into a single count.
//     seq->npu_dma_memcpy_nd(
//         sizeof(bf16),
//         x_arg_id,
//         S2MM,
//         attn_tile,
//         bd_5,
//         it_channel_1,
//         {0, 0, 0, (uint32_t)o_offset},
//         {1, 2, rows,  inner},
//         {0, (uint32_t)((size_t)L_TOKENS * D), inner, 1},
//         -1, 0, true
//     );
// }

/// The layer's output, h2 = h1 + down_proj(SwiGLU), L_TOKENS x D.
///
/// This is the rms tile's *third* phase, not its second: the post-attention norm
/// now goes west to the projection engine as up|gate's x and has no DDR tap of
/// its own, because rms_core.Y_out reaches at most two destination sets and both
/// norms share the westward one.
///
/// Same shape as _receive_o and for the same reason: one producer, one whole
/// token-major row per packet, and D = 5120 split across two axes of one
/// descriptor because a 10-bit dimension tops out at 2048 once bf16 sizes are
/// halved on lowering. Never two BDs -- the first would swallow whole tokens.
void decoding_layer::_receive_h2(npu_sequence* seq){
    const uint32_t inner = (uint32_t)(D / 4);
    assert((uint32_t)D % 4 == 0 && inner <= 2048 &&
           "a BD dimension is 10 bits, and bf16 sizes are halved");
    assert(L_TOKENS <= 1024);

    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        S2MM,
        xyr_tile,
        bd_6,
        it_channel_1,
        {0, 0, 0, (uint32_t)h2_offset},
        {1, (uint32_t)L_TOKENS, 4,     inner},
        {0, (uint32_t)D,        inner, 1},
        -1, 0, true
    );
}

/// k and v into the two-plane DDR cache: **a descriptor per column per plane.**
///
/// The RoPE tile emits k then v for each token, alternating, and stamps each
/// with its own packet id, so the two arrive on two S2MM channels of the same
/// shim rather than interleaved on one:
///
///     ch0   k(0) k(1) ... k(L-1)      -> k(t) at t * DK
///     ch1   v(0) v(1) ... v(L-1)      -> v(t) at plane_rows * DK + t * DV
///
/// Each is then a plain strided write, one row per token:
///
///   dim2 (L_TOKENS)  token   stride DK (or DV) -- advance a row of the plane
///   dim3 (width)     element stride 1
///
/// **This used to be one descriptor, and it was the cache's context ceiling.**
/// Catching the interleaved stream meant describing both planes at once, with
/// an extra axis of size 2 whose stride was the whole k plane. Shim BD strides
/// are a 20-bit field and bf16 halves them on lowering, so `MAX_L * DK / 2 <=
/// 1 << 20` put the cache at MAX_L <= 2048 -- and npu_instr_utils only *prints*
/// on overflow before masking the value, so anything above it landed every v
/// row at a wrapped address, invisibly, until the next pass read it back.
///
/// Splitting the arrival across two channels removes the axis rather than
/// widening it. The only strides left are one row of one plane, DK and DV,
/// which do not depend on MAX_L at all; the plane base is an offset the host
/// patches into the BD's address at full width (npu_dma_memcpy_nd folds
/// offset[] into a byte `arg_offset`, not into a strided field). What bounds
/// MAX_L now is the replay's 10-bit chunk count, 16384 -- see
/// _replay_kv_cache and decoding_layer.hpp::kv_cache_size.
///
/// The token axis stays in dims 1..3 with the iteration dimension at 1. The
/// iteration count is *not* part of the transfer length -- npu_dma_memcpy_nd()
/// builds buffer_length from dims 1..3 only ("iteration size does not count"),
/// and the hardware derives dim2_size from that length -- so putting the tokens
/// there would silently transfer one token's worth and stop. That is exactly the
/// bug the old _receive_qkv() hit, and it costs nothing to avoid: L_TOKENS = 8
/// and DK = 1024 both fit the 10-bit dim size field.
void decoding_layer::_receive_kv_cache(npu_sequence* seq){
    const uint32_t k_row_stride = (uint32_t)this->k_row_stride();
    const uint32_t v_row_stride = (uint32_t)this->v_row_stride();
    const uint32_t width = (uint32_t)(kv_heads_per_column * DH);
    assert(this->v_at(context_offset + L_TOKENS - 1) + (size_t)DV <= kv_cache_size &&
           "the append walks past the end of the cache");
    assert(L_TOKENS <= 1023 && width <= 1023 * 2);   // 10-bit dim, doubled for bf16
    // The only strides left are one row of one plane. Both are constants of the
    // model, not of MAX_L, so there is nothing here to overflow however deep the
    // cache gets -- which is the entire point of the two channels.
    assert(k_row_stride / 2 <= (1u << 20) && v_row_stride / 2 <= (1u << 20));

    // Two BDs per attention column, on that column's own shim: k on S2MM ch0
    // and v on ch1. The RoPE tile splits k and v by kv head AND by plane, and
    // sends each (column, plane) as its own packet (see iron/RoPE.py); column
    // c's heads sit at [c * width, + width) of every cache row.
    //
    // It is on the attention shims and not IT3 because npu_dma_wait only orders
    // transfers on one shim tile. The replay reads from these same shims, so
    // this is what lets the sequence wait for the append before replaying --
    // an append on IT3 would be unordered against the replay and the CUs would
    // attend to stale rows. Both channels now have to be waited on; see
    // _gen_sequence.
    for (int c = 0; c < 2; c++) {
        const size_t head_off = (size_t)c * width;
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            S2MM,
            attn_col_tiles[c],
            bd_3,
            it_channel_0,
            {0, 0, 0, (uint32_t)(this->k_at(context_offset) + head_off)},
            {1, 1, (uint32_t)L_TOKENS, width},
            {0, 0, k_row_stride,       1},
            -1, 0, true
        );
        seq->npu_dma_memcpy_nd(
            sizeof(bf16),
            kv_cache_arg_id,
            S2MM,
            attn_col_tiles[c],
            bd_4,
            it_channel_1,
            {0, 0, 0, (uint32_t)(this->v_at(context_offset) + head_off)},
            {1, 1, (uint32_t)L_TOKENS, width},
            {0, 0, v_row_stride,       1},
            -1, 0, true
        );
    }
}

void decoding_layer::_move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset){
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
