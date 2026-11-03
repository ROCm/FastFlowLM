#include <limits>
#include <stdexcept>
#include <algorithm>
#include <omp.h>
#include "qwen3_5vl_npu_detail.hpp"
#include "metrices.hpp"
#include "utils/error_measure.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
qwen3_5vl_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, qwen3_5vl_npu* parent_ptr, int MAX_L )
    : config(config), npu(npu_instance), parent_ptr(parent_ptr){
    is_vlm = config.get<bool>("is_vlm", false);
    current_context_length = 0;
    // build the weight-shape definition (dimensions + per-layer weight descriptors)
    desc.build(config);
    // an attention layer's cache is K rows then V rows, each half MAX_L rows long; the slot copies
    // and update_max_length find the V half at the middle of the buffer, which needs DK == DV
    if (desc.DK != desc.DV) {
        throw std::runtime_error("qwen3_5vl_npu: DK != DV, the K/V cache layout assumed by checkpoints does not hold");
    }

    vocab_size = desc.vocab_size;
    vocab_size_padded = desc.vocab_size_padded;

    this->MAX_L = std::max(MAX_L, 4096);
    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->sequence = std::make_unique<qwen3_5vl_npu_sequence>(this->desc, config, MAX_L);
    this->V_offset = this->sequence->get_V_offset();

    if (is_vlm){
        this->qwen3_5vl_image_encoder = std::make_unique<Qwen3_5VLImageEncoder>(config, npu, this->parent_ptr);
    }

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));

    this->layer = layer_app_manager->create_app();
    this->linear_layer = layer_app_manager->create_app();
    this->first_linear_layer = layer_app_manager->create_app();
    this->set_layer_rtp = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }

    // lm_head runs on layer.xclbin's projection engine (mode RTPs), so a token never switches xclbins
    this->lm_head = layer_app_manager->create_app();

    // allocate all buffers
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);
    alpha_proj_cpu.resize(desc.num_hidden_layers);
    beta_proj_cpu.resize(desc.num_hidden_layers);

    size_t attn_total_proj_weights_byte_size = desc.attn_total_proj_weights_byte_size;
    size_t linear_total_proj_weights_byte_size = desc.linear_total_proj_weights_byte_size;
    size_t attn_rms_weights_byte_size = desc.attn_rms_weights_byte_size;

    size_t linear_rms_weights_byte_size = desc.linear_rms_weights_byte_size;
    size_t linear_ssm_weights_byte_size = desc.linear_ssm_weights_byte_size;
    size_t kv_cache_size = desc.get_kv_cache_size(this->MAX_L);

    size_t linear_kv_cache_byte_size = desc.get_linear_cache_size();
    size_t conv_w_byte_size = desc.get_conv_w_size();

    for (size_t i = 0; i < desc.num_hidden_layers; i++){
        if (is_attn_layer(i)){
            this->proj_weights[i] = this->layer.create_bo_buffer<u8>(attn_total_proj_weights_byte_size);
            this->rms_weights[i] = this->layer.create_bo_buffer<bf16>(attn_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = this->layer.create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
            this->kv_caches[i] = this->layer.create_bo_buffer<bf16>(kv_cache_size);
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
        }
        else{
            this->proj_weights[i] = this->linear_layer.create_bo_buffer<u8>(linear_total_proj_weights_byte_size);
            this->rms_weights[i] = this->linear_layer.create_bo_buffer<bf16>(linear_rms_weights_byte_size / sizeof(bf16));
            this->rope_rms_weights[i] = this->linear_layer.create_bo_buffer<bf16>(linear_ssm_weights_byte_size / sizeof(bf16));
            this->kv_caches[i] = this->linear_layer.create_bo_buffer<bf16>(linear_kv_cache_byte_size / sizeof(bf16));
            this->kv_caches[i].memset((bf16)0);
            this->kv_caches[i].sync_to_device();
            this->alpha_proj_cpu[i] = buffer<bf16>(desc.D * desc.linear_num_v_heads);
            this->beta_proj_cpu[i] = buffer<bf16>(desc.D * desc.linear_num_v_heads);
        }
    }

    x = layer.create_bo_buffer<bf16>(desc.D * 2);
    lm_head_weights = lm_head.create_bo_buffer<u8>(desc.get_lm_head_in_layer_w_size());
    lm_head_norm = lm_head.create_bo_buffer<bf16>(desc.D);
    logits = lm_head.create_bo_buffer<bf16>(desc.get_lm_head_in_layer_rows());
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    this->clear_context();
    // the embedding table is created in load_weights, once we know whether the
    // checkpoint stores model.embed_tokens or only the tied q8_0 lm_head.

    pctx = std::make_unique<qwen3_5vl_prefill_context>(npu, &desc, sequence.get(), config);

    this->sequence->gen_lm_head_in_layer_seq(lm_head.seq());
    this->lm_head_run = lm_head.create_run(this->x, this->lm_head_weights, this->lm_head_norm, this->lm_head_norm, this->logits);

    this->sequence->gen_layer_seq(this->linear_layer.seq(), 0, false, false);
    this->sequence->gen_layer_seq(this->first_linear_layer.seq(), 0, false, true);

    set_context_length(0, 0);
}

void qwen3_5vl_npu::Impl::load_weights(Q4NX& q4nx){

    // read qkv weights
    if(is_vlm){
        SafeTensors vision_weights(config.get<std::string>("vision_model_weight", ""));
        this->qwen3_5vl_image_encoder->init_weights(vision_weights);
    }
    // Newer converters tie the word embeddings and emit only the quantized
    // lm_head, so fall back to dequantizing the embedding out of it.
    if (q4nx.has_tensor("model.embed_tokens.weight")){
        this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);
        this->embedding->init_weights(&q4nx, "model.embed_tokens");
    }
    else{
        this->embedding_q8 = std::make_unique<embedding_q8_0>(desc.vocab_size, desc.D);
        this->embedding_q8->init_weights(q4nx, "lm_head.weight");
    }

    for (int i = 0; i < desc.num_hidden_layers; i++){
        if (desc.is_attn_layer(i)){
            desc.load_attn_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i]);
        }
        else{
            desc.load_linear_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->alpha_proj_cpu[i], this->beta_proj_cpu[i]);
        }
    }

    {
        buffer<bf16> final_rms_norm_weight;
        q4nx.load_weights(final_rms_norm_weight, "model.norm.weight");
        memcpy(this->x.data() + desc.D, final_rms_norm_weight.data(), desc.D * sizeof(bf16));
        this->x.sync_to_device();
        memcpy(this->lm_head_norm.data(), final_rms_norm_weight.data(), desc.D * sizeof(bf16));
        this->lm_head_norm.sync_to_device();
    }
    {
        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, "lm_head.weight");

        desc.reorder_cpy_lm_head_in_layer(this->lm_head_weights.data(), lm_head_w);
        this->lm_head_weights.sync_to_device();
    }

    this->clear_context();
}

void qwen3_5vl_npu::Impl::_set_rope_rms_weights(int idx){
    int RoPE_DIM = desc.RoPE_DIM;
    buffer<bf16> cos_buf(RoPE_DIM / 2);
    buffer<bf16> sin_buf(RoPE_DIM / 2);

    for (int j = 0; j < RoPE_DIM / 2; j++){
        cos_buf[j] = (bf16)cos(inv_freq[j] * idx);
        sin_buf[j] = (bf16)sin(inv_freq[j] * idx);
    }

    for (int i = 0; i < desc.num_hidden_layers; i++){
        if (is_attn_layer(i)){
            bf16* w_rope_ptr = this->rope_rms_weights[i].data();
            memcpy(w_rope_ptr, cos_buf.data(), RoPE_DIM * sizeof(bf16) / 2);
            memcpy(w_rope_ptr + RoPE_DIM / 2, sin_buf.data(), RoPE_DIM * sizeof(bf16) / 2);
            this->rope_rms_weights[i].sync_to_device();
        }
    }
}

void qwen3_5vl_npu::Impl::set_context_length(int L, int new_general_token_idx){
    this->current_context_length = L;
    this->general_token_idx = new_general_token_idx;

    this->sequence->gen_set_context(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, true);
    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();

        this->layers_run.add(this->set_layer_rtp.create_run());
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
            }
            else {
                if (is_first_linear_layer(i)) {
                    this->layers_run.add(this->first_linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                }
                else  {
                    this->layers_run.add(this->linear_layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
                }
            }
        }
    }

    this->_set_rope_rms_weights(new_general_token_idx);
}

#ifdef QWEN35_DECODE_PROFILE
#include <chrono>
#include <cstdlib>
#include <iostream>
namespace {
using prof_clock = std::chrono::steady_clock;

struct decode_profile_t {
    static constexpr int N = 8;
    const char* names[N] = {"embed", "x to NPU", "24 layers", "x sync", "lm_head start + next run list",
                            "lm_head wait", "logits from NPU", "between calls (host)"};
    double sum_us[N] = {};
    int calls = 0;
    int between_count = 0;
    prof_clock::time_point last_end{};
    bool lm_head_alone = std::getenv("FLM_PROF_LMHEAD_ALONE") != nullptr;

    static double us(prof_clock::time_point a, prof_clock::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    }

    void report() {
        double total = 0;
        for (int i = 0; i < N - 1; i++) {
            total += sum_us[i];
        }
        std::cerr << "[decode profile] " << calls << " tokens" << (lm_head_alone ? " (lm_head not overlapped)" : "") << "\n";
        for (int i = 0; i < N; i++) {
            int n = (i == N - 1) ? std::max(between_count, 1) : calls;
            std::cerr << "    " << names[i] << ": " << sum_us[i] / n << " us\n";
        }
        std::cerr << "    forward() total: " << total / calls << " us" << std::endl;
        std::fill(sum_us, sum_us + N, 0.0);
        calls = 0;
        between_count = 0;
    }
};

decode_profile_t g_profile;
}  // namespace
#endif

buffer<bf16> qwen3_5vl_npu::Impl::forward(int ids){
#ifdef QWEN35_DECODE_PROFILE
    prof_clock::time_point t[8];
    t[0] = prof_clock::now();
    if (g_profile.last_end != prof_clock::time_point{} && decode_profile_t::us(g_profile.last_end, t[0]) < 1e6) {
        g_profile.sum_us[7] += decode_profile_t::us(g_profile.last_end, t[0]);
        g_profile.between_count++;
    }
#define PROF_MARK(i) t[i] = prof_clock::now()
#else
#define PROF_MARK(i)
#endif
    this->embed_forward(ids, this->x);
    PROF_MARK(1);
    this->x.sync_to_device();
    PROF_MARK(2);

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        this->layers_run.wait();
    }
    else{
        set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else if (is_first_linear_layer(i)) {
                this->first_linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else  {
                this->linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
        }
    }
    PROF_MARK(3);

    this->x.sync_to_device();
    PROF_MARK(4);
#ifdef QWEN35_DECODE_PROFILE
    if (g_profile.lm_head_alone) {
        auto a = prof_clock::now();
        this->lm_head_run.start();
        this->lm_head_run.wait();
        auto b = prof_clock::now();
        this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
        auto c = prof_clock::now();
        t[5] = t[4] + (c - b);
        t[6] = t[5] + (b - a);
    } else
#endif
    {
        this->lm_head_run.start();
        this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
        PROF_MARK(5);
        this->lm_head_run.wait();
        PROF_MARK(6);
    }
    this->logits.sync_from_device();
    PROF_MARK(7);
#ifdef QWEN35_DECODE_PROFILE
    for (int i = 0; i < 7; i++) {
        g_profile.sum_us[i] += decode_profile_t::us(t[i], t[i + 1]);
    }
    g_profile.calls++;
    g_profile.last_end = prof_clock::now();
    if (g_profile.calls == 64) {
        g_profile.report();
    }
#endif
#undef PROF_MARK
    return this->logits_valid;
}

buffer<bf16> qwen3_5vl_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    return this->_prefill_with_mm(ids, payload);
#endif
}

buffer<bf16> qwen3_5vl_npu::Impl::_prefill_with_mv(std::vector<int>& ids, void* payload){
    for (int i = 0; i < (int)ids.size(); i++){
        this->embed_forward(ids[i], this->x);
        this->x.sync_to_device();
        set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            if (is_attn_layer(i)){
                this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else if (is_first_linear_layer(i)) {
                this->first_linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
            else  {
                this->linear_layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
            }
        }
        this->set_context_length(this->current_context_length + 1, this->general_token_idx + 1);
        std::cout << "Token " << i << " done!" << std::endl;
    }
    this->lm_head_run.start();
    this->lm_head_run.wait();
    this->logits.sync_from_device();

    utils::print_matrix(logits_valid, desc.vocab_size);

    return logits_valid;
}

buffer<bf16> qwen3_5vl_npu::Impl::_prefill_with_mm(std::vector<int>& ids, void* payload){
#ifdef ENABLE_DEBUG
    SafeTensors aref(utils::path_join(config.model_path, "gated_attention_0.8B.safetensors"));
    buffer<int> input_ids; aref.load_weights(input_ids, "input_ids");
#endif
    int L_in = ids.size();

#ifdef ENABLE_DEBUG
    std::cout << std::endl;
    header_print("DEBUG", "DEBUG ENABLED!");
    L_in = input_ids.size();
    ids.resize(L_in);
    memcpy(ids.data(), input_ids.data(), L_in * sizeof(int));
    header_print("DEBUG", "Debuging input ids:" << L_in);
#endif

    buffer<bf16> hidden_states(desc.D * L_in);
    tensor_2d<bf16> hidden_states_2d(hidden_states, desc.D);

    std::vector<std::vector<bf16>> image_embedding;

    if(is_vlm && payload != nullptr){
        image_embedding = this->qwen3_5vl_image_encoder->encode(payload);
    }

    bf16 * vision_token_ptr = (payload != nullptr)? image_embedding[0].data() : nullptr;
    std::vector<bool> vision_mask(L_in);
    std::vector<int> input_idx = this->get_input_idx(vision_mask, ids, (qwen3_5vl_image_payload_t*)payload);

    for (int l = 0; l < L_in; l++) {
        if (vision_mask[l] && ((payload != nullptr))){
            memcpy(hidden_states_2d[l].data(), vision_token_ptr, desc.D * sizeof(bf16));
            vision_token_ptr += desc.D;
        }
        else {
            this->embed_forward(ids[l], hidden_states_2d[l]);
        }
    }

#ifdef ENABLE_DEBUG
    buffer<bf16> embed_ref;
    aref.load_weights(embed_ref, "embeddings");
    print_error_metrics(
        hidden_states.data(), embed_ref.data(),
        1,
        L_in, desc.D,
        L_in, desc.D
    );
#endif

    time_utils::time_point start = time_utils::now();
    for (int layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++) {
        auto ret = pctx->forward(layer_idx, hidden_states, L_in, proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx], kv_caches[layer_idx], input_idx, alpha_proj_cpu[layer_idx], beta_proj_cpu[layer_idx], this->current_context_length);

        // buffer<bf16> ref;
        // aref.load_weights(ref, "layer_output");
        // print_error_metrics(
        //     ret.data(), ref.data(),
        //     1,
        //     L_in, desc.D,
        //     L_in, desc.D
        // );
        // exit(0);
    }

    buffer<bf16> predict = hidden_states_2d[L_in - 1];
    this->lm_head_weights.sync_to_device();
    get_logits(predict);
    this->set_context_length(this->current_context_length + L_in, input_idx[L_in * 3 - 1] + 1);

    return logits_valid;
}

std::vector<int> qwen3_5vl_npu::Impl::get_input_idx(std::vector<bool>& vision_mask, std::vector<int>& tokens, qwen3_5vl_image_payload_t* image_payload){
    int l = tokens.size();
    bool in_image = false;
    int image_counter = 0;

    // Optimization: Cached width and counters to avoid expensive division/modulo operations
    int current_w = 0;
    int cur_row = 0;
    int cur_col = 0;

    std::vector<int> input_idx(l * 3);
    for (int i = 0; i < tokens.size(); i++){
        vision_mask[i] = (tokens[i] == image_soft_token_id);
        if (vision_mask[i]){
            if (!in_image){
                in_image = true;
                // Update cached width for the new image block
                current_w = image_payload->images[image_counter].grid_w / 2;
                cur_row = 0;
                cur_col = 0;
            }

            input_idx[i * 3 + 0] = general_token_idx;
            input_idx[i * 3 + 1] = general_token_idx + cur_row;
            input_idx[i * 3 + 2] = general_token_idx + cur_col;

            // Advance counters
            cur_col++;
            if (cur_col == current_w) {
                cur_col = 0;
                cur_row++;
            }
        }
        else {
            if (in_image){
                in_image = false;
                general_token_idx += std::max(image_payload->images[image_counter].grid_h,
                                           image_payload->images[image_counter].grid_w) / 2;
                image_counter++;
            }
            input_idx[i * 3 + 0] = general_token_idx;
            input_idx[i * 3 + 1] = general_token_idx;
            input_idx[i * 3 + 2] = general_token_idx;
            general_token_idx += 1;
        }
    }

    return input_idx;
}

buffer<bf16> qwen3_5vl_npu::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DK;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<void*>(k_cache_ptr + offset), desc.DK * sizeof(bf16));

    return k_cache;
}

buffer<bf16> qwen3_5vl_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DV;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<void*>(v_cache_ptr + offset), desc.DV * sizeof(bf16));
    return v_cache;
}

buffer<bf16> qwen3_5vl_npu::Impl::get_logits(buffer<bf16>& predict){
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    this->lm_head_run.wait();
    this->logits.sync_from_device();
    return this->logits_valid;
}

size_t qwen3_5vl_npu::Impl::state_row_bytes() {
    size_t attention_layers = 0;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        attention_layers += is_attn_layer(i) ? 1 : 0;
    }
    return attention_layers * (static_cast<size_t>(desc.DK) + desc.DV) * sizeof(bf16);
}

size_t qwen3_5vl_npu::Impl::state_fixed_bytes() {
    size_t bytes = 0;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        if (!is_attn_layer(i)) {
            bytes += this->kv_caches[i].size() * sizeof(bf16);
        }
    }
    return bytes;
}

std::unique_ptr<hybrid_engine_state> qwen3_5vl_npu::Impl::new_state() {
    return std::make_unique<hybrid_engine_state>(state_fixed_bytes(), state_row_bytes());
}

int qwen3_5vl_npu::Impl::read_state(hybrid_engine_state& base, int kv_from) {
    auto* st = &base;
    if (st == nullptr) {
        return -1;
    }
    const int length = this->current_context_length;
    kv_from = std::clamp(kv_from, 0, length);
    st->state.resize(desc.num_hidden_layers);
    st->kv_k.resize(desc.num_hidden_layers);
    st->kv_v.resize(desc.num_hidden_layers);
    header_print_r("FLM", "Reading state at context length " + std::to_string(length) + ", K/V rows from " + std::to_string(kv_from));
    const size_t rows = static_cast<size_t>(length - kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        this->kv_caches[i].sync_from_device();
        if (!is_attn_layer(i)) {
            if (st->state[i].size() != this->kv_caches[i].size()) st->state[i] = buffer<bf16>(this->kv_caches[i].size());
            memcpy(st->state[i].data(), this->kv_caches[i].data(), this->kv_caches[i].size() * sizeof(bf16));
        }
        else if (rows > 0) {
            // K rows then V rows, one DK / DV row per position; the V half starts at the middle of the buffer
            if (st->kv_k[i].size() != rows * desc.DK) st->kv_k[i] = buffer<bf16>(rows * desc.DK);
            if (st->kv_v[i].size() != rows * desc.DV) st->kv_v[i] = buffer<bf16>(rows * desc.DV);
            memcpy(st->kv_k[i].data(), this->kv_caches[i].data() + static_cast<size_t>(kv_from) * desc.DK, rows * desc.DK * sizeof(bf16));
            memcpy(st->kv_v[i].data(), this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(kv_from) * desc.DV, rows * desc.DV * sizeof(bf16));
        }
        else {
            // nothing to keep: a reused state must not hold rows of its previous contents
            st->kv_k[i] = buffer<bf16>();
            st->kv_v[i] = buffer<bf16>();
        }
        if (!is_attn_layer(i)) {
            this->kv_caches[i].sync_to_device();
        }
    }
    st->filled = true;
    st->context_length = length;
    st->general_token_idx = this->general_token_idx;
    st->kv_from = kv_from;
    return length;
}

/// @brief whether `st` holds what this engine's write_state needs (a state of another engine would not)
static bool qwen3_5vl_state_fits(const hybrid_engine_state* st, const std::vector<buffer<bf16>>& kv_caches, const qwen3_5vl_desc& desc, int full_attention_interval) {
    if (st == nullptr || !st->filled || st->state.size() != desc.num_hidden_layers) {
        return false;
    }
    const size_t rows = static_cast<size_t>(st->context_length - st->kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        const bool attn = (i % full_attention_interval) == (full_attention_interval - 1);
        if (attn ? (rows > 0 && (st->kv_k[i].size() != rows * desc.DK || st->kv_v[i].size() != rows * desc.DV))
                 : st->state[i].size() != kv_caches[i].size()) {
            return false;
        }
    }
    return true;
}

int qwen3_5vl_npu::Impl::write_state_kv(const hybrid_engine_state& base) {
    const auto* st = &base;
    if (!qwen3_5vl_state_fits(st, this->kv_caches, desc, full_attention_interval)) {
        return -1;
    }
    const size_t rows = static_cast<size_t>(st->context_length - st->kv_from);
    if (rows == 0) {
        return st->context_length;
    }
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (is_attn_layer(i)) {
            this->kv_caches[i].sync_from_device();
            memcpy(this->kv_caches[i].data() + static_cast<size_t>(st->kv_from) * desc.DK, st->kv_k[i].data(), rows * desc.DK * sizeof(bf16));
            memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st->kv_from) * desc.DV, st->kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    return st->context_length;
}

int qwen3_5vl_npu::Impl::write_state(const hybrid_engine_state& base) {
    const auto* st = &base;
    if (!qwen3_5vl_state_fits(st, this->kv_caches, desc, full_attention_interval)) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(st->context_length));
    const size_t rows = static_cast<size_t>(st->context_length - st->kv_from);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        this->kv_caches[i].sync_from_device();
        if (!is_attn_layer(i)) {
            memcpy(this->kv_caches[i].data(), st->state[i].data(), st->state[i].size() * sizeof(bf16));
        }
        else {
            if (rows > 0) {
                memcpy(this->kv_caches[i].data() + static_cast<size_t>(st->kv_from) * desc.DK, st->kv_k[i].data(), rows * desc.DK * sizeof(bf16));
                memcpy(this->kv_caches[i].data() + (this->kv_caches[i].size() / 2) + static_cast<size_t>(st->kv_from) * desc.DV, st->kv_v[i].data(), rows * desc.DV * sizeof(bf16));
            }
            // Rows past the restored length hold whatever the requests since the
            // state was read left there. The kernels do not rely on the length alone to
            // keep them out of attention, so put back the zeros a fresh context has.
            const size_t from = static_cast<size_t>(st->context_length);
            const size_t half = this->kv_caches[i].size() / 2;
            memset(this->kv_caches[i].data() + from * desc.DK, 0, (half - from * desc.DK) * sizeof(bf16));
            memset(this->kv_caches[i].data() + half + from * desc.DV, 0, (half - from * desc.DV) * sizeof(bf16));
        }
        this->kv_caches[i].sync_to_device();
    }
    this->current_context_length = st->context_length;
    this->general_token_idx = st->general_token_idx;
    this->set_context_length(this->current_context_length, this->general_token_idx);
    return this->current_context_length;
}

void qwen3_5vl_npu::Impl::clear_context(){
    // Checkpoint slots survive a clear: each holds its own copy of the recurrent state and of
    // the K/V rows it needs, so zeroing the live cache does not touch them. A request that
    // shares nothing with the cache (another conversation's side call) can clear it and the
    // next request still resumes from the snapshots saved before. The caller must only
    // restore a slot that keeps rows from position 0 (kv_from 0), or put the rows before its
    // kv_from back first (restore_slot_kv of the slot that holds them).
    this->set_context_length(0, 0);
    for (int i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
}

int qwen3_5vl_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void qwen3_5vl_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = MAX_L;
    const size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);
    this->sequence->set_max_length(MAX_L);
    this->V_offset = this->sequence->get_V_offset();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        if (!is_attn_layer(i)){
            continue; // nothing to do with linear layer
        }
        // sizes come from this layer's own buffer: layer 0 is a linear layer with a differently sized state,
        // and the old V half starts at the middle of the old buffer
        this->kv_caches[i].sync_from_device();
        const size_t old_half = this->kv_caches[i].size() / 2;
        buffer<bf16> kv_cache_temp(this->kv_caches[i].size());
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].memset((bf16)0);
        const size_t new_half = this->kv_caches[i].size() / 2;
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(kv_cache_temp.data()), old_half * sizeof(bf16));
        memcpy(static_cast<void*>(this->kv_caches[i].data() + new_half), static_cast<void*>(kv_cache_temp.data() + old_half), old_half * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    this->set_context_length(this->current_context_length, this->general_token_idx); // regenerate sequence
    this->legacy_checkpoint.reset(); // the legacy copy was taken against the old buffers
}
///@brief destructor of qwen3vl_npu
qwen3_5vl_npu::Impl::~Impl() = default;

// ===============================================
// Externals for qwen3vl_npu
// ===============================================
qwen3_5vl_npu::qwen3_5vl_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){

    this->load_vision_preprocess_parameters(config);
    this->_impl = new Impl(config, npu_instance, this,   MAX_L);
}

buffer<bf16> qwen3_5vl_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> qwen3_5vl_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen3_5vl_npu::set_context_length(int L){
    std::cout << "Setting context length is not supported for qwen3vl_npu!" << std::endl;
}

void qwen3_5vl_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void qwen3_5vl_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void qwen3_5vl_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> qwen3_5vl_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> qwen3_5vl_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int qwen3_5vl_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int qwen3_5vl_npu::checkpoint(){
    // the single legacy copy keeps no K/V rows: its caller continues from the same cache
    if (!this->_impl->legacy_checkpoint) {
        this->_impl->legacy_checkpoint = std::make_unique<hybrid_engine_state>(this->_impl->state_fixed_bytes(), this->_impl->state_row_bytes());
    }
    return this->_impl->read_state(*this->_impl->legacy_checkpoint, std::numeric_limits<int>::max());
}

int qwen3_5vl_npu::restore(){
    if (!this->_impl->legacy_checkpoint) {
        return -1;
    }
    return this->_impl->write_state(*this->_impl->legacy_checkpoint);
}

std::unique_ptr<hybrid_engine_state> qwen3_5vl_npu::new_state(){
    return this->_impl->new_state();
}

int qwen3_5vl_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int qwen3_5vl_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int qwen3_5vl_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

qwen3_5vl_npu::~qwen3_5vl_npu(){
    delete this->_impl;
}

extern "C" const flm_state_api* qwen3_5vl_state_api(void* engine){
    return hybrid_state_api<qwen3_5vl_npu>(engine);
}
