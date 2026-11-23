// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "SmolVLA.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include "tensor_utils/safe_tensors.hpp"
#include "npu_utils/npu_utils.hpp"
#include "cpu_functions.hpp"
#include "utils/prof_utils.hpp"

std::vector<weight_desc_t*> SmolVLA::adapter_all() {
    return {
        &action_in_proj, &action_in_proj_bias,
        &action_out_proj, &action_out_proj_bias,
        &action_time_mlp_in, &action_time_mlp_in_bias,
        &action_time_mlp_out, &action_time_mlp_out_bias,
        &state_proj, &state_proj_bias,
    };
}

void SmolVLA::from_pretrained(const std::string& model_path) {
    this->model_path = model_path;
    std::ifstream file(model_path + "/config.json");
    if (!file.is_open()) {
        std::cerr << "Failed to open config: " << model_path << "/config.json" << std::endl;
        exit(1);
    }
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    nlohmann::json jcfg = nlohmann::json::parse(json_str);
    
    this->build(jcfg);
}

void SmolVLA::build(nlohmann::json& jcfg) {
    assert(jcfg.contains("vision_config") && "config.json missing 'vision_config'");
    assert(jcfg.contains("expert_config") && "config.json missing 'expert_config'");

    // the text tower's layer count is the top-level "num_vlm_layers"
    const uint32_t num_vlm_layers = jcfg.at("num_vlm_layers").get<uint32_t>();

    this->vlm->build(jcfg["vision_config"], num_vlm_layers);
    this->expert->build(jcfg["expert_config"]);
    // top-level F32 adapters. shapes are {in, out} (transpose of the stored
    // [out, in] safetensor layout). names are model.* paths.
    const uint32_t D_expert  = this->expert->def.D;      // 720
    const uint32_t D_text    = this->vlm->def.D_text;    // 960
    const uint32_t act_dim   = jcfg.at("max_action_dim").get<uint32_t>();  // 32
    const uint32_t state_dim = jcfg.at("max_state_dim").get<uint32_t>();   // 32

    out.resize(smolvla_cfg::MAX_ACTION_CHUNK * D_expert);
    action_in_proj           = weight_desc_t(flm_bf16, {act_dim, D_expert},     "model.action_in_proj.weight");
    action_in_proj_bias      = weight_desc_t(flm_bf16, {D_expert},              "model.action_in_proj.bias");
    action_out_proj          = weight_desc_t(flm_bf16, {D_expert, act_dim},     "model.action_out_proj.weight");
    action_out_proj_bias     = weight_desc_t(flm_bf16, {act_dim},               "model.action_out_proj.bias");
    action_time_mlp_in       = weight_desc_t(flm_bf16, {2 * D_expert, D_expert}, "model.action_time_mlp_in.weight");
    action_time_mlp_in_bias  = weight_desc_t(flm_bf16, {D_expert},              "model.action_time_mlp_in.bias");
    action_time_mlp_out      = weight_desc_t(flm_bf16, {D_expert, D_expert},    "model.action_time_mlp_out.weight");
    action_time_mlp_out_bias = weight_desc_t(flm_bf16, {D_expert},              "model.action_time_mlp_out.bias");
    state_proj               = weight_desc_t(flm_bf16, {state_dim, D_text},     "model.state_proj.weight");
    state_proj_bias          = weight_desc_t(flm_bf16, {D_text},                "model.state_proj.bias");

    weight_container helper;
    for (weight_desc_t* w : adapter_all()) {
        helper.add_weight(*w);
    }
    this->adapter_byte_size = helper.get_size();
    state_in.resize(this->vlm->language->def.D);
}

void SmolVLA::load_weights() {
    // allocate device buffers, then stream everything from the single model file.
    this->adapter_weights = this->xclbin_mgr->create_bo_buffer<u8>(this->adapter_byte_size);
    this->vlm->alloc_weights();
    this->expert->alloc_weights();

    SafeTensors st(this->model_path + "/model.safetensors");

    for (weight_desc_t* w : adapter_all()) {
        st.load_weights(this->adapter_weights, w->name, w->offset);
        w->load();
    }
    this->vlm->load_weights(st);
    this->expert->load_weights(st);
}

buffer<bf16> SmolVLA::predict(smolvla_input_t& model_input) {
    // project the robot state [state_dim] -> [D_text] on CPU (state_proj is
    // stored [out=D_text, in=state_dim], biased), producing the single state
    // token that gets appended to the language input.
    const int D_text    = (int)this->vlm->language->def.D;
    const int state_dim = (int)this->state_proj.shape[0];   // {state_dim, D_text}
    // the vlm resets the shared registry, so hold the state-projection time in a
    // local and print it alongside the vlm's own tower breakdown.
    PROF_INIT();
    PROF_ACC_DECL(state_proj);
    buffer<bf16> state_padded(state_dim);
    memset(state_padded.data(), 0, state_padded.size() * sizeof(bf16));
    memcpy(state_padded.data(), model_input.states.data(), model_input.states.size() * sizeof(bf16));
    buffer<bf16> sp_w = this->state_proj.locate_myself<bf16>(this->adapter_weights);
    buffer<bf16> sp_b = this->state_proj_bias.locate_myself<bf16>(this->adapter_weights);
    cpu_func::_linear_biased(state_in.data(), state_padded.data(),
                             sp_w.data(), sp_b.data(), 1, state_dim, D_text);
    PROF_ACC(state_proj);

    buffer<bf16> kv_cache = this->vlm->forward(model_input, state_in);
    PROF_ACC_PRINT("vlm.state_proj", state_proj);
    return kv_cache;

}

buffer<bf16> SmolVLA::embed_actions(buffer<bf16>& x_t, int L, float timestep) {
    const int D_expert = (int)this->expert->def.D;   // 720
    const int act_dim  = (int)this->action_in_proj.shape[0];  // {action_dim, D_expert}

    auto A = [&](weight_desc_t& d) { return d.locate_myself<bf16>(this->adapter_weights); };
    buffer<bf16> aip_w = A(action_in_proj),      aip_b = A(action_in_proj_bias);
    buffer<bf16> mi_w  = A(action_time_mlp_in),  mi_b  = A(action_time_mlp_in_bias);
    buffer<bf16> mo_w  = A(action_time_mlp_out), mo_b  = A(action_time_mlp_out_bias);

    static const bool fold = [] { const char* e = std::getenv("SMOLVLA_EMBED_FOLD"); return !(e && e[0] == '0'); }();
    if (fold) {
        constexpr float min_period = smolvla_cfg::TIME_MIN_PERIOD;
        constexpr float max_period = smolvla_cfg::TIME_MAX_PERIOD;
        const int half = D_expert / 2;
        if (embed_wc.empty()) {
            embed_wc.assign((size_t)D_expert * act_dim, 0.f);
            embed_bias0.assign(D_expert, 0.f);
            #pragma omp parallel for num_threads(2) schedule(static)
            for (int o = 0; o < D_expert; o++) {
                const bf16* w1 = mi_w.data() + (size_t)o * 2 * D_expert;
                float* wc = embed_wc.data() + (size_t)o * act_dim;
                float b0 = 0.f;
                for (int j = 0; j < D_expert; j++) {
                    const float w = (float)w1[j];
                    b0 += w * (float)aip_b.data()[j];
                    const bf16* wa = aip_w.data() + (size_t)j * act_dim;
                    for (int k = 0; k < act_dim; k++) wc[k] += w * (float)wa[k];
                }
                embed_bias0[o] = b0;
            }
        }
        auto tb = embed_time_bias.find(timestep);
        if (tb == embed_time_bias.end()) {
            std::vector<bf16> time_emb(D_expert);
            for (int i = 0; i < half; i++) {
                float frac    = (half > 1) ? (float)i / (float)(half - 1) : 0.0f;
                float period  = min_period * std::pow(max_period / min_period, frac);
                float scaling = 1.0f / period * 2.0f * (float)M_PI;
                float s       = scaling * timestep;
                time_emb[i]        = (bf16)std::sin(s);
                time_emb[i + half] = (bf16)std::cos(s);
            }
            std::vector<float> b(D_expert);
            cpu_func::_linear_biased_row_f32(b.data(), time_emb.data(), mi_w.data() + D_expert, mi_b.data(),
                                             D_expert, D_expert, /*w_stride=*/2 * D_expert);
            for (int o = 0; o < D_expert; o++) b[o] += embed_bias0[o];
            tb = embed_time_bias.emplace(timestep, std::move(b)).first;
        }
        buffer<bf16> h((size_t)L * D_expert);
        cpu_func::_cpu_mm<float>(h.data(), embed_wc.data(), x_t.data(), L, act_dim, D_expert, 0, tb->second.data());
        buffer<bf16> sig((size_t)L * D_expert);
        cpu_func::_sigmoid(sig.data(), h.data(), L * D_expert);
        cpu_func::_elementwise_mul(h.data(), h.data(), sig.data(), L * D_expert);
        cpu_func::_linear_biased(out.data(), h.data(), mo_w.data(), mo_b.data(), L, D_expert, D_expert);
        return out;
    }

    // action_in_proj: [L, action_dim] -> [L, D_expert].
    buffer<bf16> action_emb((size_t)L * D_expert);
    cpu_func::_linear_biased(action_emb.data(), x_t.data(), aip_w.data(), aip_b.data(),
                             L, act_dim, D_expert);

    // sinusoidal time embedding (shared across the L rows), [D_expert].
    // period = min_period * (max_period/min_period)^(i/(D/2-1)); scaling = 2*pi/period.
    // emb = [sin(scaling*t) ; cos(scaling*t)].
    constexpr float min_period = smolvla_cfg::TIME_MIN_PERIOD;
    constexpr float max_period = smolvla_cfg::TIME_MAX_PERIOD;
    const int half = D_expert / 2;
    std::vector<bf16> time_emb(D_expert);
    for (int i = 0; i < half; i++) {
        float frac    = (half > 1) ? (float)i / (float)(half - 1) : 0.0f;
        float period  = min_period * std::pow(max_period / min_period, frac);
        float scaling = 1.0f / period * 2.0f * (float)M_PI;
        float s       = scaling * timestep;
        time_emb[i]        = (bf16)std::sin(s);
        time_emb[i + half] = (bf16)std::cos(s);
    }

    // mlp_in over concat([action_emb, time_emb], dim=-1), without materialising
    // the concat. action_time_mlp_in is stored [out=D, in=2*D], so columns
    // 0..D of each weight row hit the action half and D..2D the time half:
    //
    //   h = action_emb @ W[:, :D]^T + (time_emb @ W[:, D:]^T + b)
    //
    // The time half is one row broadcast over all L, so it collapses to a single
    // GEMV whose fp32 result becomes the bias of the remaining [L,D]x[D,D] GEMM.
    // That halves the MACs and keeps the time contribution in fp32 throughout.
    std::vector<float> mi_bias(D_expert);
    cpu_func::_linear_biased_row_f32(mi_bias.data(), time_emb.data(),
                                     mi_w.data() + D_expert, mi_b.data(),
                                     D_expert, D_expert, /*w_stride=*/2 * D_expert);

    // mlp_in -> silu -> mlp_out.
    buffer<bf16> h((size_t)L * D_expert);
    cpu_func::_cpu_mm<bf16>(h.data(), mi_w.data(), action_emb.data(),
                            L, D_expert, D_expert, /*w_stride=*/2 * D_expert,
                            mi_bias.data());
    // silu(h) = h * sigmoid(h).
    buffer<bf16> sig((size_t)L * D_expert);
    cpu_func::_sigmoid(sig.data(), h.data(), L * D_expert);
    cpu_func::_elementwise_mul(h.data(), h.data(), sig.data(), L * D_expert);

    cpu_func::_linear_biased(out.data(), h.data(), mo_w.data(), mo_b.data(),
                             L, D_expert, D_expert);
    return out;
}

void SmolVLA::prepare_expert_prefix_kv() {
    const int P              = this->vlm->language->valid_prefix_len;
    const int num_vlm_layers = (int)this->vlm->language->def.num_layers;
    const int MAX_L          = SmolVLA_language::MAX_L;
    this->expert->prepare_prefix_kv(this->vlm->language->kv_cache,
                                    P, num_vlm_layers, MAX_L);
}

buffer<bf16> SmolVLA::expert_core(buffer<bf16>& chunk, int L) {
    const int P = this->vlm->language->valid_prefix_len;
    return this->expert->forward(chunk, L, P);
}

buffer<bf16> SmolVLA::read_velocity(buffer<bf16>& hidden, int L) {
    const int D_expert = (int)this->expert->def.D;
    const int act_dim  = (int)this->action_out_proj.shape[1];  // {D_expert, action_dim}
    auto A = [&](weight_desc_t& d) { return d.locate_myself<bf16>(this->adapter_weights); };
    buffer<bf16> aop_w = A(action_out_proj), aop_b = A(action_out_proj_bias);
    buffer<bf16> v((size_t)L * act_dim);
    cpu_func::_linear_biased(v.data(), hidden.data(), aop_w.data(), aop_b.data(),
                             L, D_expert, act_dim);
    return v;
}

buffer<bf16> SmolVLA::forward(smolvla_input_t& model_input, buffer<bf16>* noise,
                             bool skip_prefill) {
    const int L       = (int)this->expert->def.chunk_size;
    const int act_dim = (int)this->action_in_proj.shape[0];
    constexpr int   num_steps = smolvla_cfg::NUM_DENOISE_STEPS;
    constexpr float dt        = smolvla_cfg::DENOISE_DT;

    PROF_RESET();
    PROF_INIT();
    // 1. vlm prefill: vision -> connector -> language, filling the kv cache.
    //    Skipped when the caller has already populated (or injected) the cache.
    if (!skip_prefill) {
        this->predict(model_input);
    }
    PROF_TICK("e2e_image_prefill");

    // 2. flow-matching denoise: seed x with the captured noise if provided,
    //    otherwise sample it (Box-Muller from rand(), deterministic under srand).
    buffer<bf16> x((size_t)L * act_dim);
    if (noise != nullptr) {
        memcpy(x.data(), noise->data(), (size_t)L * act_dim * sizeof(bf16));
    } else {
        for (int i = 0; i < L * act_dim; i++) {
            float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
            x[i] = (bf16)(std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2));
        }
    }
    PROF_TICK("e2e_noise_gen");

    // 2b. hoist every expert layer's view of the VLM prefix out of the denoise
    //     loop. Cross layers re-project the same prefix through the same frozen
    //     weights on every step; self layers re-copy the same prefix rows into
    //     the head of their [prefix ; chunk] K/V. Neither changes across steps,
    //     so both are done once here and read from device memory by all
    //     num_steps steps. Has to sit after the prefill branch above: it consumes
    //     the kv cache that branch produces (or that a caller injected under
    //     skip_prefill).
    this->prepare_expert_prefix_kv();
    PROF_TICK("e2e_expert_prefix_kv");

    PROF_SPAN_BEGIN(denoise);
    // 3. integrate num_steps Euler steps over the expert. Every stage tick below
    //    accumulates over all num_steps iterations, so the report shows one
    //    aggregated row per stage (10x) rather than ten copies of the table.
    for (int step = 0; step < num_steps; step++) {
        float t = 1.0f + step * dt;
        buffer<bf16> h = this->embed_actions(x, L, t);
        PROF_TICK("denoise:embed_actions");
        h = this->expert_core(h, L);
        PROF_TICK("denoise:expert_core");
        buffer<bf16> v = this->read_velocity(h, L);
        for (int i = 0; i < L * act_dim; i++)
            x[i] = (bf16)((float)x[i] + dt * (float)v[i]);
        PROF_TICK("denoise:read_velocity+euler");
    }
    PROF_SPAN_END(denoise, "e2e_denoise");
    PROF_REPORT();
    return x;
}
