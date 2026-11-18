// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "models.h"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

// One decoded token compared with one reference row.
struct check_t {
    std::string name;           // what is compared, e.g. "output (layer output)"
    std::string ref_key;        // tensor in the reference file
    int width;                  // elements per token
    buffer<bf16> ref_bf16;      // reference with the original bf16 weights
    buffer<bf16> ref_q4k;       // reference with the q4nx Q4_K weights (may be empty)
    std::vector<error_metrics> vs_bf16;
    std::vector<error_metrics> vs_q4k;
};

// LM_Config::from_pretrained wants an FLM config.json (it asserts addr_qk /
// addr_kv). Hy's folder has the HF config.json, and the RTP addresses come from
// the build's rtp_addresses.txt, so fill in only the fields this test uses.
static LM_Config load_hf_config(const std::string& dir) {
    std::ifstream file(dir + "/config.json");
    if (!file.is_open()) {
        header_print("error", "Cannot open " << dir << "/config.json");
        exit(1);
    }
    LM_Config cfg;
    cfg.model_path = dir;
    cfg.model_name = dir;
    cfg._json_config = nlohmann::json::parse(file);
    JSON_GET(cfg.model_type, cfg._json_config, "model_type", "", std::string);
    JSON_GET(cfg.head_dim, cfg._json_config, "head_dim", 0, u32);
    JSON_GET(cfg.hidden_size, cfg._json_config, "hidden_size", 0, u32);
    JSON_GET(cfg.num_attention_heads, cfg._json_config, "num_attention_heads", 0, u32);
    JSON_GET(cfg.num_key_value_heads, cfg._json_config, "num_key_value_heads", 0, u32);
    JSON_GET(cfg.num_hidden_layers, cfg._json_config, "num_hidden_layers", 0, u32);
    assert(cfg.head_dim > 0 && cfg.hidden_size > 0 && cfg.num_attention_heads > 0);
    assert(cfg.num_key_value_heads > 0 && cfg.num_hidden_layers > 0);
    return cfg;
}

// One comparison (e.g. output vs the Q4_K reference) over all tokens.
struct summary_t {
    float mean_cos = 0.0f, min_cos = 1.0f, max_l2 = 0.0f;
    int min_token = 0;      // the token with the lowest cos
    int below = 0;          // tokens with cos < LOW_COS
};
constexpr float LOW_COS = 0.999f;

static summary_t summarize(const std::vector<error_metrics>& m) {
    summary_t r;
    for (size_t i = 0; i < m.size(); i++) {
        r.mean_cos += m[i].CosineSimilarity / m.size();
        r.max_l2 = std::max(r.max_l2, m[i].RelativeL2);
        if (m[i].CosineSimilarity < r.min_cos) {
            r.min_cos = m[i].CosineSimilarity;
            r.min_token = (int)i;
        }
        r.below += m[i].CosineSimilarity < LOW_COS;
    }
    return r;
}

// "  output vs Q4_K ref: mean cos 0.999803  min 0.999147 (token 69)  max rel L2 0.0413  [0 tokens < 0.999]"
static void print_summary_line(const std::string& name, const char* against, const std::vector<error_metrics>& m) {
    if (m.empty()) return;
    const summary_t r = summarize(m);
    std::cout << "  " << std::left << std::setw(16) << name << " vs " << std::setw(4) << against << " ref: "
              << std::right << std::fixed << std::setprecision(6) << "mean cos " << r.mean_cos
              << "  min " << r.min_cos << " (token " << r.min_token << ")"
              << "  max rel L2 " << std::setprecision(4) << r.max_l2
              << "  [" << r.below << " tokens < " << std::setprecision(3) << LOW_COS << "]" << std::endl;
}

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("HY_MEMORY_30B_A3B_DENSE"), "Model type");
    desc.add_options()("len,l", arg_utils::po::value<int>()->default_value(100), "Number of tokens to decode");
    desc.add_options()("layer,L", arg_utils::po::value<int>()->default_value(0), "Layer to test (a dense layer: 0)");
    desc.add_options()("ctx", arg_utils::po::value<int>()->default_value(131072), "KV cache length (a multiple of 16; 131072 = 128K)");
    desc.add_options()("xclbin", arg_utils::po::value<std::string>()->default_value(""), "Use this layer.xclbin instead of the default build");
    desc.add_options()("bench", arg_utils::po::value<int>()->default_value(0), "After the check, time the NPU only: this many rounds of 480 back-to-back calls");
    desc.add_options()("bench_ctx", arg_utils::po::value<int>()->default_value(8), "Context length for --bench (must be below --ctx)");
    desc.add_options()("per_step", arg_utils::po::bool_switch()->default_value(false), "Also print every token's cos / rel L2");
    desc.add_options()("kv_fill", arg_utils::po::value<int>()->default_value(0), "Long-context check: fill KV rows 0..N-1 with the Q4_K reference's K/V rows (repeated) and decode at positions N..; no reference checks, use --dump and models_simple/check_hy_long_kv.py");
    desc.add_options()("dump", arg_utils::po::value<std::string>()->default_value(""), "Write the layer outputs and the new K/V rows to <prefix>.out.bin / .k.bin / .v.bin (raw bf16)");

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();
    int max_len = vm["len"].as<int>();
    int layer_idx = vm["layer"].as<int>();
    int context_length = vm["ctx"].as<int>();

    model_list::model_t model = model_list::get_model_type(model_type);
    if (!model_list::is_dense_layer(model, layer_idx)) {
        header_print("error", "Layer " << layer_idx << " is not a dense layer (only layer 0 is; the MoE layers run on ../moe_decoding)");
        return 1;
    }
    LM_Config lm_config = load_hf_config(model_list::get_config_path(model));
    int D = lm_config.hidden_size;
    int DK = lm_config.num_key_value_heads * lm_config.head_dim;
    int DH = lm_config.head_dim;

    lm_config.layer_xclbin_name = vm["xclbin"].as<std::string>().empty()
        ? model_list::get_layer_xclbin_path(model) : vm["xclbin"].as<std::string>();

    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_list::get_q4nx_path(model));

    // ---- references ----
    std::string reference_path = model_list::get_reference_path(model);
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

    // sanity check: embedding lookup from model.q4nx vs the reference embeddings
    buffer<int32_t> ids;
    references->load_weights(ids, "input.input_ids");
    buffer<bf16> ref_embed;
    references->load_weights(ref_embed, "embeddings");
    float embed_cos = 0.0f;   // model.q4nx embeddings vs the reference's: should be 1
    {
        buffer<bf16> embed;
        q4nx_model->load_weights(embed, "model.embed_tokens.weight");
        buffer<bf16> embeddings(ref_embed.size());
        for (size_t i = 0; i < ids.size(); i++) {
            memcpy(embeddings.data() + i * D, embed.data() + (size_t)ids[i] * D, D * sizeof(bf16));
        }
        embed_cos = get_error_metrics(embeddings, ref_embed).CosineSimilarity;
    }

    // ---- NPU ----
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);

    // RoPE frequencies in fp32, as transformers does: theta^(-2i / DH).
    // (The reference file stores inv_freq as bf16, which is too coarse:
    // ~0.4 rad of angle error by token 100.)
    const double theta = model_list::get_rope_theta(model);
    const int rope_dim = int(DH * model_list::get_rope_partial_factor(model));
    buffer<float> rope_freqs(rope_dim / 2);
    for (int i = 0; i < rope_dim / 2; i++) {
        rope_freqs[i] = (float)(1.0 / std::pow(theta, (2.0 * i) / rope_dim));
    }

    // the layer's reference input / outputs
    const std::string layer_prefix = "layer_" + std::to_string(layer_idx) + ".";
    buffer<bf16> input_ref;
    references->load_weights(input_ref, layer_prefix + "input");

    std::string q4k_reference_path = model_list::get_q4k_reference_path(model, layer_idx);
    std::unique_ptr<SafeTensors> q4k_references;
    if (std::filesystem::exists(q4k_reference_path)) {
        q4k_references = std::make_unique<SafeTensors>(q4k_reference_path);
    }
    else {
        header_print("warning", "No Q4_K reference at " << q4k_reference_path
            << " (run models_simple/ref_hy_mem_layer_q4k.py --layer " << layer_idx << "); comparing with the bf16 one only");
    }
    // layer output = residual + MLP; K / V: the new KV-cache row (K after
    // q/k norm and RoPE), read back from the cache, so a wrong KV layout shows on its own
    std::vector<check_t> checks;
    checks.push_back({"output (layer output)", "output", D});
    checks.push_back({"K cache row (after RoPE)", "k_rope", DK});
    checks.push_back({"V cache row", "v_proj", DK});
    if (std::getenv("HY_ZERO_MLP")) {
        // DEBUG: MLP zeroed on the NPU (see decoding_layer.cpp), so its output is x + o_proj
        checks.push_back({"residual (MLP zeroed)", "residual", D});
    }
    for (auto& c : checks) {
        references->load_weights(c.ref_bf16, layer_prefix + c.ref_key);
        if (q4k_references) {
            q4k_references->load_weights(c.ref_q4k, c.ref_key);
        }
    }

    decoding_layer layer(model, lm_config, layer_app_mgr, context_length);
    layer.load_weights(*q4nx_model, layer_idx);
    layer.set_rope_frequency(rope_freqs);
    layer.set_context_length(0);

    // ---- decode ----
    // Decode token i of a layer equals row i of the prefill reference (causal
    // attention), so feed the reference layer input row by row and compare.
    const uint32_t steps = std::min<uint32_t>(max_len, ids.size());

    // Long-context check (--kv_fill N): the first N cache rows hold the Q4_K
    // reference's K/V rows, repeated, and the tokens decode at positions N.. .
    // The reference rows assume positions 0.., so there is nothing to check here;
    // models_simple/check_hy_long_kv.py recomputes the layer from --dump.
    const int kv_fill = vm["kv_fill"].as<int>();
    if (kv_fill > 0) {
        if (!q4k_references) {
            header_print("error", "--kv_fill needs the Q4_K reference " << q4k_reference_path);
            return 1;
        }
        if (kv_fill + (int)steps > context_length) {
            header_print("error", "--kv_fill " << kv_fill << " + " << steps << " tokens is more than --ctx " << context_length);
            return 1;
        }
        buffer<bf16> k_rows, v_rows;
        q4k_references->load_weights(k_rows, "k_rope");
        q4k_references->load_weights(v_rows, "v_proj");
        layer.fill_kv(k_rows.data(), v_rows.data(), (int)(k_rows.size() / DK), kv_fill);
        layer.set_context_length(kv_fill);
        checks.clear();
        header_print("info", "KV rows 0.." << kv_fill - 1 << " filled with the Q4_K reference's K/V rows (repeated)");
    }
    // --dump: the outputs and the new K/V rows, for check_hy_long_kv.py
    const std::string dump_prefix = vm["dump"].as<std::string>();
    std::ofstream dump_out, dump_k, dump_v;
    if (!dump_prefix.empty()) {
        dump_out.open(dump_prefix + ".out.bin", std::ios::binary | std::ios::trunc);
        dump_k.open(dump_prefix + ".k.bin", std::ios::binary | std::ios::trunc);
        dump_v.open(dump_prefix + ".v.bin", std::ios::binary | std::ios::trunc);
    }
    header_print("info", "Running layer " << layer_idx << " for " << steps << " tokens ("
                 << lm_config.layer_xclbin_name << ", KV cache " << context_length << ")");
    buffer<bf16> token_in(D);
    int completed = 0;   // calls that ended in COMPLETED (a timeout leaves the last output in place)
    float total_ms = 0.0f;
    // DEBUG: HY_DUMP_OUT=<file> appends every token's layer output (D raw bf16) to <file>
    const char* dump_path = std::getenv("HY_DUMP_OUT");
    std::ofstream dump_file;
    if (dump_path) {
        dump_file.open(dump_path, std::ios::binary | std::ios::trunc);
    }
    for (uint32_t i = 0; i < steps; i++) {
        memcpy(token_in.data(), input_ref.data() + (size_t)i * D, D * sizeof(bf16));
        time_utils::time_point start = time_utils::now();
        buffer<bf16> out = layer(token_in);
        completed += layer.last_call_completed;
        layer.report_debug_tap();
        time_utils::time_point end = time_utils::now();
        total_ms += time_utils::duration_ms(start, end).first;
        if (dump_file.is_open()) {
            dump_file.write(reinterpret_cast<const char*>(out.data()), D * sizeof(bf16));
        }
        if (dump_out.is_open()) {
            buffer<bf16> k_row = layer.get_k_cache(kv_fill + i);
            buffer<bf16> v_row = layer.get_v_cache(kv_fill + i);
            dump_out.write(reinterpret_cast<const char*>(out.data()), D * sizeof(bf16));
            dump_k.write(reinterpret_cast<const char*>(k_row.data()), DK * sizeof(bf16));
            dump_v.write(reinterpret_cast<const char*>(v_row.data()), DK * sizeof(bf16));
        }

        for (auto& c : checks) {
            buffer<bf16> got(c.width);
            if (c.ref_key == "k_rope") {
                got = layer.get_k_cache(i);
            }
            else if (c.ref_key == "v_proj") {
                got = layer.get_v_cache(i);
            }
            else {
                memcpy(got.data(), out.data(), D * sizeof(bf16));
            }
            buffer<bf16> ref_row(c.ref_bf16.data() + (size_t)i * c.width, c.width);
            c.vs_bf16.push_back(get_error_metrics(got, ref_row));
            if (c.ref_q4k.size() > 0) {
                buffer<bf16> q4k_row(c.ref_q4k.data() + (size_t)i * c.width, c.width);
                c.vs_q4k.push_back(get_error_metrics(got, q4k_row));
            }
        }
    }

    // ---- result ----
    // per-token table, only on request (--per_step)
    if (vm["per_step"].as<bool>() && !checks.empty()) {
        for (uint32_t i = 0; i < steps; i++) {
            std::cout << "\tLayer " << layer_idx << " step " << i;
            for (const auto& c : checks) {
                const error_metrics& e = c.vs_q4k.empty() ? c.vs_bf16[i] : c.vs_q4k[i];
                std::cout << "  " << c.ref_key << " cos " << std::fixed << std::setprecision(5) << e.CosineSimilarity
                          << " L2 " << e.RelativeL2;
            }
            std::cout << (checks[0].vs_q4k.empty() ? "  (vs bf16 ref)" : "  (vs Q4_K ref)") << std::endl;
        }
    }
    const bool ok = completed == (int)steps;
    std::cout << std::endl << "==== RESULT: layer " << layer_idx << " (dense), " << steps << " tokens ====" << std::endl;
    std::cout << "  calls completed  : " << completed << " / " << steps
              << (ok ? "" : "   <-- NOT ALL CALLS COMPLETED: outputs below are not valid") << std::endl;
    std::cout << "  time per token   : " << std::fixed << std::setprecision(2) << total_ms / steps
              << " ms (host + NPU; the NPU alone: --bench)" << std::endl;
    std::cout << "  embedding check  : cos " << std::setprecision(6) << embed_cos << std::endl;
    for (const auto& c : checks) {
        print_summary_line(c.ref_key, "Q4_K", c.vs_q4k);
        print_summary_line(c.ref_key, "bf16", c.vs_bf16);
    }

    // NPU-only timing. The per-token latency above includes the host syncing
    // every buffer (the 35 MB weight buffer too) each call; bench() runs `calls`
    // calls in one runlist with no syncs, at context length bench_ctx. 480 calls
    // per round, as the timer counts whole ms. Best round reported.
    const int bench_rounds = vm["bench"].as<int>();
    if (bench_rounds > 0) {
        const int calls = 480;
        float best_ms = 1e9f;
        for (int r = 0; r < bench_rounds; r++) {
            const float ms_per_call = layer.bench(token_in, calls, vm["bench_ctx"].as<int>());
            best_ms = std::min(best_ms, ms_per_call);
        }
        std::cout << "  NPU time per call: " << std::fixed << std::setprecision(3) << best_ms << " ms (context "
                  << vm["bench_ctx"].as<int>() << ", best of " << bench_rounds << " x " << calls << " calls)" << std::endl;
    }
    std::cout << "=================================================" << std::endl;
    return ok ? 0 : 1;
}
