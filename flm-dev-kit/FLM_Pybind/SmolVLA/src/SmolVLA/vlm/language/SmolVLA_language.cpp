// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "SmolVLA_language.hpp"

#include "tensor_utils/safe_tensors.hpp"
#include "npu_utils/npu_utils.hpp"
#include "cpu_functions.hpp"
#include "utils/prof_utils.hpp"

void SmolVLA_language::build(uint32_t num_layers) {
    this->def.build(num_layers);
    this->exec = SmolVLA_language_exec(this->xclbin_mgr, &this->def);
}

void SmolVLA_language::alloc_weights() {
    // buffer<T> has no move ctor (its copy ctor is shallow), so resize to
    // empty buffers first, then move-assign each bo-backed buffer into place.
    this->layer_weights.resize(this->def.num_layers);
    this->norm_weights.resize(this->def.num_layers);
    for (uint32_t l = 0; l < this->def.num_layers; l++) {
        this->norm_weights[l] = buffer<bf16>(this->def.norm_byte_size / sizeof(bf16));
        this->layer_weights[l] =
            this->xclbin_mgr->create_bo_buffer<bf16>(this->def.layer_byte_size / sizeof(bf16));
    }
    this->non_layer_weights = this->xclbin_mgr->create_bo_buffer<u8>(this->def.non_layer_byte_size);

    // kv cache: (2, num_layers, MAX_L, DK). DK == DV so the key/value halves are
    // equal-sized. Zero-initialized so unused (padded) rows read as 0.
    const size_t kv_elems = (size_t)2 * this->def.num_layers * MAX_L * this->def.DK;
    this->kv_cache = buffer<bf16>(kv_elems);
    memset(this->kv_cache.data(), 0, kv_elems * sizeof(bf16));
}

void SmolVLA_language::load_weights(SafeTensors& st) {
    this->def.load(st, this->layer_weights, this->norm_weights, this->non_layer_weights);
}

buffer<bf16> SmolVLA_language::forward(buffer<bf16>& embeds, buffer<bf16>& state, std::vector<int>& language_tokens, int num_images, int L) {
    // vision already reported and is done with the registry; start a fresh set
    // of counters so the language table stands on its own.
    PROF_RESET();
    PROF_INIT();
    // pre-process
    const int text_begin = num_images * smolvla_cfg::VISION_TOKENS_PER_IMAGE;
    buffer<bf16> embed_tokens = this->def.embed_tokens.locate_myself<bf16>(non_layer_weights);
    // copy the real (non-pad) text-token embeddings; stop at the first pad so
    // the state token follows immediately after the last real token.
    constexpr int pad_token_id = smolvla_cfg::PAD_TOKEN_ID;
    int num_text = 0;
    for (; num_text < (int)language_tokens.size(); num_text++){
        if (language_tokens[num_text] == pad_token_id) {
            break;
        }
        memcpy(embeds.data() + (size_t)(text_begin + num_text) * def.D,
               embed_tokens.data() + (size_t)language_tokens[num_text] * def.D,
               def.D * sizeof(bf16));
    }

    // SmolVLA normalizes both the image (post-connector) and text embeddings by
    // sqrt(hidden_size) before they enter the language model (see modeling_
    // smolvla.py: img_emb *= sqrt(dim), lang_emb *= sqrt(dim)). The state
    // projection output is NOT scaled, so scale only the image + real-text rows
    // [0, text_begin + num_text) in one contiguous vectorized pass, before the
    // (unscaled) state token is written into the next row.
    const int scaled_elems = (text_begin + num_text) * (int)def.D;
    cpu_func::_elementwise_mul(embeds.data(), embeds.data(), def.embed_scale, scaled_elems);

    // append the projected state token right after the last real text token.
    memcpy(embeds.data() + (size_t)(text_begin + num_text) * def.D, state.data(), def.D * sizeof(bf16));

    // exact non-padded context length: image tokens + real text tokens + 1 state.
    L = text_begin + num_text + 1;
    this->valid_prefix_len = L;
    // rows [L, L_padded) are read by the kernels (they work in whole 64-row blocks); clear what an earlier,
    // longer prefix left there so the output does not depend on the call history.
    const int L_padded = (L + 63) / 64 * 64;
    memset(embeds.data() + (size_t)L * def.D, 0, (size_t)(L_padded - L) * def.D * sizeof(bf16));
    PROF_TICK("Token embedding");

    // run every layer, writing each layer's pre-rope k/v into the shared cache.
    // cache layout is (2, num_layers, MAX_L, DK): the key half spans the first
    // num_layers*MAX_L*DK elements, the value half follows it.
    const size_t layer_stride = (size_t)MAX_L * def.DK;      // DK == DV
    const size_t value_base   = (size_t)def.num_layers * layer_stride;
    //
    // The prefill's only output is the kv cache -- the final hidden state is
    // discarded (this function returns kv_cache, and the action expert reads
    // nothing else). So on the last layer everything after the k/v writes is
    // dead: o_proj, mha, both residual adds, layer_norm2 and the entire mlp
    // sub-block. early_stop drops that tail.
    for (uint32_t layer_id = 0; layer_id < def.num_layers; layer_id++){
        bf16* k_dst = kv_cache.data() + (size_t)layer_id * layer_stride;
        bf16* v_dst = kv_cache.data() + value_base + (size_t)layer_id * layer_stride;
        const bool early_stop = (layer_id == def.num_layers - 1);
        this->exec.forward(embeds, norm_weights[layer_id], layer_weights[layer_id], L, k_dst, v_dst,
                           early_stop);
    }
    PROF_TICK("All layers");

    PROF_REPORT();

    // return the full kv cache; the action expert cross-attends over it. This is
    // a shallow (non-owning) view — ownership stays in this->kv_cache so the
    // expert can keep reading it after forward() returns.
    return kv_cache;
}
