// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file host.cpp
/// \brief Qwen3.5-2B decode speed test: all 24 layers at a random 10K context, timed the way the engine runs them
/// \note See README.md in this folder.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <boost/program_options.hpp>
#include "npu_utils/npu_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "qwen3_5vl_model_def.hpp"
#include "qwen3_5vl_npu_sequence.hpp"
#include "cpu_ref.hpp"

namespace po = boost::program_options;
using steady = std::chrono::steady_clock;

struct bench_args_t {
    std::string model_path;
    std::string xclbin_path;
    int context_len;
    int iterations;
    int warmup;
    bool per_layer;
    bool check;
    float kv_scale;
    std::string prompt_ids;
    std::string kv_quant;
    std::vector<cpu_ref::kv_quant_t> kv_quants;
    int repeat;
    int gap_ms;
    std::string lm_head_xclbin;
    bool lm_head_in_layer;
    bool no_proj_mode;
    uint32_t seed;
};

struct layer_buffers_t {
    buffer<u8> proj;
    buffer<bf16> rms;
    buffer<bf16> rope_rms;
    buffer<bf16> cache;
};

struct stats_t {
    double mean = 0, std = 0, min = 0, max = 0;
};

static bool parse_args(int argc, char* argv[], bench_args_t& args) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show help")
        ("model-path,p", po::value<std::string>(&args.model_path)->default_value("/path/to/models/Qwen3.5-2B-NPU2"), "Model folder (model.q4nx, config.json)")
        ("xclbin,x", po::value<std::string>(&args.xclbin_path)->default_value("build/QWEN3_5_2B/xclbins/layer.xclbin"), "Decode layer xclbin")
        ("len,l", po::value<int>(&args.context_len)->default_value(10240), "Tokens already in the cache; the test decodes token len + 1")
        ("iters,i", po::value<int>(&args.iterations)->default_value(20), "Measured decode steps")
        ("warmup,w", po::value<int>(&args.warmup)->default_value(3), "Unmeasured decode steps first")
        ("per-layer", po::bool_switch(&args.per_layer), "Also time every layer as its own run")
        ("check", po::bool_switch(&args.check), "Check every layer against the fp32 CPU reference instead of timing")
        ("kv-scale", po::value<float>(&args.kv_scale)->default_value(1.0f), "Range of the random K cache (larger = peakier attention)")
        ("prompt-ids", po::value<std::string>(&args.prompt_ids)->default_value(""), "Token-id file: build a real context by decoding the first len tokens")
        ("kv-quant", po::value<std::string>(&args.kv_quant)->default_value("none"), "With --check: also simulate an 8-bit KV cache on the CPU: none, int8, fp8, all")
        ("repeat", po::value<int>(&args.repeat)->default_value(0), "Determinism test: run the same decode step N times from one saved state, both run styles")
        ("lm-head", po::value<std::string>(&args.lm_head_xclbin)->default_value(""), "Also time lm_head.xclbin: alone, and alternating with the layers like the engine")
        ("lm-head-in-layer", po::bool_switch(&args.lm_head_in_layer), "lm_head as a 25th run in layer.xclbin: check logits vs CPU, then time it")
        ("no-proj-mode", po::bool_switch(&args.no_proj_mode), "Do not write the projection mode RTP (for xclbins built before it existed)")
        ("gap-ms", po::value<int>(&args.gap_ms)->default_value(0), "Determinism test: pause between per-layer runs")
        ("seed,s", po::value<uint32_t>(&args.seed)->default_value(1234), "Random seed for cache, state and input");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return false;
    }
    if (args.kv_quant == "int8" || args.kv_quant == "all") {
        args.kv_quants.push_back(cpu_ref::kv_quant_t::int8);
    }
    if (args.kv_quant == "fp8" || args.kv_quant == "all") {
        args.kv_quants.push_back(cpu_ref::kv_quant_t::fp8);
    }
    return true;
}

static LM_Config make_config(const std::string& model_path) {
    LM_Config config;
    config.model_path = model_path;
    config.model_name = std::filesystem::path(model_path).filename().string();
    std::ifstream file(std::filesystem::path(model_path) / "config.json");
    config._json_config = nlohmann::json::parse(file);
    config._json_config["is_vlm"] = false;
    config.flm_version = config._json_config.value("flm_version", std::string("0.0.0"));
    return config;
}

static int round_up_pow2(int value) {
    int result = 4096;
    while (result < value) {
        result *= 2;
    }
    return result;
}

static void fill_random_bf16(bf16* dst, size_t count, float range, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < count; i++) {
        dst[i] = (bf16)dist(rng);
    }
}

static void fill_random_f32(float* dst, size_t count, float range, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < count; i++) {
        dst[i] = dist(rng);
    }
}

static void fill_attention_cache(layer_buffers_t& layer, const qwen3_5vl_desc& desc, size_t V_offset, int context_len, float k_range, std::mt19937& rng) {
    bf16* k_rows = layer.cache.data();
    bf16* v_rows = layer.cache.data() + V_offset;
    fill_random_bf16(k_rows, (size_t)context_len * desc.DK, k_range, rng);
    fill_random_bf16(v_rows, (size_t)context_len * desc.DV, 1.0f, rng);
    layer.cache.sync_to_device();
}

static void fill_linear_state(layer_buffers_t& layer, qwen3_5vl_desc& desc, std::mt19937& rng) {
    size_t conv_count = desc.get_s_offset();
    size_t state_count = (size_t)desc.linear_num_v_heads * desc.linear_v_head_dim * desc.linear_qk_head_dim;
    fill_random_bf16(layer.cache.data(), conv_count, 0.5f, rng);
    float* state = reinterpret_cast<float*>(layer.cache.data() + conv_count);
    fill_random_f32(state, state_count, 0.05f, rng);
    layer.cache.sync_to_device();
}

static bool is_first_linear_layer(const qwen3_5vl_desc& desc, int layer_idx) {
    return layer_idx % desc.full_attention_interval == 0;
}

static void set_rope_position(std::vector<layer_buffers_t>& layers, qwen3_5vl_desc& desc, int position) {
    int half = desc.RoPE_DIM / 2;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!desc.is_attn_layer(i)) {
            continue;
        }
        bf16* rope = layers[i].rope_rms.data();
        for (int j = 0; j < half; j++) {
            rope[j] = (bf16)std::cos(desc.inv_freq[j] * position);
            rope[half + j] = (bf16)std::sin(desc.inv_freq[j] * position);
        }
        layers[i].rope_rms.sync_to_device();
    }
}

struct compare_t {
    double cos = 0, rel = 0, max_abs = 0;
};

static std::vector<float> to_float(const bf16* src, size_t count) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; i++) {
        out[i] = (float)src[i];
    }
    return out;
}

static compare_t compare(const std::vector<float>& npu, const std::vector<float>& ref) {
    double dot = 0, nn = 0, rr = 0, dd = 0;
    compare_t c;
    for (size_t i = 0; i < ref.size(); i++) {
        double diff = (double)npu[i] - ref[i];
        dot += (double)npu[i] * ref[i];
        nn += (double)npu[i] * npu[i];
        rr += (double)ref[i] * ref[i];
        dd += diff * diff;
        c.max_abs = std::max(c.max_abs, std::fabs(diff));
    }
    c.cos = dot / (std::sqrt(nn * rr) + 1e-30);
    c.rel = std::sqrt(dd / (rr + 1e-30));
    return c;
}

static void print_compare(int layer, const char* kind, const char* what, const compare_t& c, double& worst_cos) {
    worst_cos = std::min(worst_cos, c.cos);
    std::cout << std::setw(5) << layer << "  " << std::left << std::setw(8) << kind << std::setw(10) << what << std::right
              << std::scientific << std::setprecision(3)
              << "  cos " << std::fixed << std::setprecision(6) << c.cos
              << "  rel " << std::scientific << std::setprecision(3) << c.rel
              << "  max_abs " << c.max_abs << (c.cos < 0.999 ? "   <-- low" : "") << std::defaultfloat << std::endl;
}

struct npu_apps_t {
    npu_app& attn_layer;
    npu_app& linear_layer;
    npu_app& first_linear_layer;
    npu_app& set_layer_rtp;
};

static npu_app& app_for_layer(qwen3_5vl_desc& desc, npu_apps_t& apps, int layer) {
    if (desc.is_attn_layer(layer)) {
        return apps.attn_layer;
    }
    return is_first_linear_layer(desc, layer) ? apps.first_linear_layer : apps.linear_layer;
}

static void run_layer_alone(qwen3_5vl_desc& desc, npu_apps_t& apps, std::vector<layer_buffers_t>& layers,
                            buffer<bf16>& x, int layer) {
    xrt::run run = app_for_layer(desc, apps, layer).create_run(x, layers[layer].proj, layers[layer].rms,
                                                               layers[layer].rope_rms, layers[layer].cache);
    run.start();
    run.wait();
}

static std::vector<int> read_ids(const std::string& path) {
    std::ifstream file(path);
    std::vector<int> ids;
    int id;
    while (file >> id) {
        ids.push_back(id);
    }
    return ids;
}

static void set_x_to_token(buffer<bf16>& x, const buffer<u8>& lm_head, int token, int D) {
    std::vector<float> row = cpu_ref::embed_token(lm_head, token, D);
    for (int i = 0; i < D; i++) {
        x[i] = (bf16)row[i];
    }
    x.sync_to_device();
}

static void prefill_with_decode(qwen3_5vl_desc& desc, qwen3_5vl_npu_sequence& sequence, npu_app_manager* manager,
                                npu_apps_t& apps, std::vector<layer_buffers_t>& layers, buffer<bf16>& x,
                                const buffer<u8>& lm_head, const std::vector<int>& ids, int count) {
    for (auto& layer : layers) {
        layer.cache.memset((bf16)0);
        layer.cache.sync_to_device();
    }
    auto t0 = steady::now();
    for (int t = 0; t < count; t++) {
        set_x_to_token(x, lm_head, ids[t], desc.D);
        sequence.gen_set_context(apps.set_layer_rtp.seq(), t + 1);
        sequence.gen_layer_seq(apps.attn_layer.seq(), t + 1, true);
        set_rope_position(layers, desc, t);
        xrt::runlist run = manager->create_runlist();
        run.add(apps.set_layer_rtp.create_run());
        for (int i = 0; i < desc.num_hidden_layers; i++) {
            run.add(app_for_layer(desc, apps, i).create_run(x, layers[i].proj, layers[i].rms, layers[i].rope_rms, layers[i].cache));
        }
        run.execute();
        run.wait();
        if ((t + 1) % 2048 == 0 || t + 1 == count) {
            header_print("info", "prefill by decode: " << t + 1 << " / " << count << " tokens, "
                         << std::fixed << std::setprecision(1) << std::chrono::duration<double>(steady::now() - t0).count() << " s");
        }
    }
    for (auto& layer : layers) {
        layer.cache.sync_from_device();
    }
}

struct saved_state_t {
    std::vector<std::vector<bf16>> caches;
    std::vector<bf16> x;
};

static saved_state_t save_state(std::vector<layer_buffers_t>& layers, buffer<bf16>& x) {
    saved_state_t s;
    for (auto& layer : layers) {
        layer.cache.sync_from_device();
        s.caches.emplace_back(layer.cache.data(), layer.cache.data() + layer.cache.size());
    }
    x.sync_from_device();
    s.x.assign(x.data(), x.data() + x.size());
    return s;
}

static void restore_state(const saved_state_t& s, std::vector<layer_buffers_t>& layers, buffer<bf16>& x) {
    for (size_t i = 0; i < layers.size(); i++) {
        memcpy(layers[i].cache.data(), s.caches[i].data(), s.caches[i].size() * sizeof(bf16));
        layers[i].cache.sync_to_device();
    }
    memcpy(x.data(), s.x.data(), s.x.size() * sizeof(bf16));
    x.sync_to_device();
}

static uint64_t state_checksum(const saved_state_t& s, int D) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const bf16* p, size_t n) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n * sizeof(bf16); i++) {
            h = (h ^ b[i]) * 1099511628211ull;
        }
    };
    for (auto& c : s.caches) {
        mix(c.data(), c.size());
    }
    mix(s.x.data(), D);
    return h;
}

static size_t count_diff(const bf16* a, const bf16* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) {
        d += memcmp(&a[i], &b[i], sizeof(bf16)) != 0;
    }
    return d;
}

static void run_step(qwen3_5vl_desc& desc, npu_app_manager* manager, npu_apps_t& apps,
                     std::vector<layer_buffers_t>& layers, buffer<bf16>& x, bool per_layer, int gap_ms) {
    if (per_layer) {
        xrt::run rtp = apps.set_layer_rtp.create_run();
        rtp.start();
        rtp.wait();
        for (int i = 0; i < desc.num_hidden_layers; i++) {
            run_layer_alone(desc, apps, layers, x, i);
            std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        }
        return;
    }
    xrt::runlist run = manager->create_runlist();
    run.add(apps.set_layer_rtp.create_run());
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        run.add(app_for_layer(desc, apps, i).create_run(x, layers[i].proj, layers[i].rms, layers[i].rope_rms, layers[i].cache));
    }
    run.execute();
    run.wait();
}

static int run_determinism(const bench_args_t& args, qwen3_5vl_desc& desc, npu_app_manager* manager, npu_apps_t& apps,
                           std::vector<layer_buffers_t>& layers, buffer<bf16>& x) {
    saved_state_t start = save_state(layers, x);
    int bad_runs = 0;
    for (bool per_layer : {false, true}) {
        std::cout << "\n" << (per_layer ? "one run per layer" : "one run list per token (engine style)")
                  << ": " << args.repeat << " repeats of the same step\n";
        saved_state_t first;
        for (int r = 0; r < args.repeat; r++) {
            restore_state(start, layers, x);
            run_step(desc, manager, apps, layers, x, per_layer, args.gap_ms);
            saved_state_t now = save_state(layers, x);
            if (r == 0) {
                first = now;
                std::cout << "  repeat 0 checksum of all caches + x: " << std::hex << state_checksum(now, desc.D) << std::dec << std::endl;
                continue;
            }
            std::string report;
            for (size_t i = 0; i < layers.size(); i++) {
                size_t d = count_diff(now.caches[i].data(), first.caches[i].data(), now.caches[i].size());
                if (d) {
                    report += " L" + std::to_string(i) + ":" + std::to_string(d);
                }
            }
            size_t dx = count_diff(now.x.data(), first.x.data(), desc.D);
            if (dx) {
                report += " x:" + std::to_string(dx);
            }
            bad_runs += !report.empty();
            std::cout << "  repeat " << r << (report.empty() ? ": identical to repeat 0" : ": DIFFERS (values per layer)" + report) << std::endl;
        }
    }
    std::cout << "\n" << bad_runs << " repeats differed" << std::endl;
    return bad_runs;
}

static stats_t compute_stats(const std::vector<double>& values);

static double time_run_us(xrt::run& run) {
    auto t0 = steady::now();
    run.start();
    run.wait();
    return std::chrono::duration<double, std::micro>(steady::now() - t0).count();
}

static double time_runlist_us(xrt::runlist& run) {
    auto t0 = steady::now();
    run.execute();
    run.wait();
    return std::chrono::duration<double, std::micro>(steady::now() - t0).count();
}

static void bench_lm_head(const bench_args_t& args, qwen3_5vl_desc& desc, qwen3_5vl_npu_sequence& sequence,
                          npu_xclbin_manager& npu, npu_app& owner, xrt::runlist& layers_run, buffer<bf16>& x, Q4NX& q4nx) {
    npu_app_manager* lm_manager = npu.register_xclbin(args.lm_head_xclbin);
    npu_app lm_head = lm_manager->create_app();
    buffer<u8> lm_weights = owner.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    buffer<bf16> logits = owner.create_bo_buffer<bf16>(desc.vocab_size_padded);
    buffer<u8> raw;
    q4nx.load_weights(raw, "lm_head.weight");
    desc.reorder_cpy_lm_head(lm_weights.data(), raw, desc.D, flm_q80, 4);
    lm_weights.sync_to_device();
    sequence.gen_lm_head_seq(lm_head.seq(), desc.vocab_size_padded, desc.D);
    xrt::run lm_run = lm_head.create_run(logits, lm_weights, x);

    std::vector<double> lm_alone, layers_alone, layers_alt, lm_alt;
    for (int i = 0; i < args.warmup + args.iterations; i++) {
        double us = time_run_us(lm_run);
        if (i >= args.warmup) lm_alone.push_back(us);
    }
    for (int i = 0; i < args.warmup + args.iterations; i++) {
        double us = time_runlist_us(layers_run);
        if (i >= args.warmup) layers_alone.push_back(us);
    }
    for (int i = 0; i < args.warmup + args.iterations; i++) {
        double a = time_runlist_us(layers_run);
        double b = time_run_us(lm_run);
        if (i >= args.warmup) {
            layers_alt.push_back(a);
            lm_alt.push_back(b);
        }
    }
    stats_t la = compute_stats(lm_alone), ya = compute_stats(layers_alone);
    stats_t yb = compute_stats(layers_alt), lb = compute_stats(lm_alt);
    std::cout << std::fixed << std::setprecision(3)
              << "\nlm_head (" << args.lm_head_xclbin << "), ms per call, mean +- std:\n"
              << "  lm_head alone, back to back      " << la.mean / 1000 << " +- " << la.std / 1000 << "\n"
              << "  24 layers alone, back to back    " << ya.mean / 1000 << " +- " << ya.std / 1000 << "\n"
              << "  alternating like the engine:\n"
              << "    24 layers (after lm_head)      " << yb.mean / 1000 << " +- " << yb.std / 1000
              << "   -> switch into layer.xclbin ~" << (yb.mean - ya.mean) / 1000 << "\n"
              << "    lm_head (after layers)         " << lb.mean / 1000 << " +- " << lb.std / 1000
              << "   -> switch into lm_head.xclbin ~" << (lb.mean - la.mean) / 1000 << "\n"
              << "    token total                    " << (yb.mean + lb.mean) / 1000 << std::endl;
}

static void add_layer_runs(xrt::runlist& run, qwen3_5vl_desc& desc, npu_apps_t& apps,
                           std::vector<layer_buffers_t>& layers, buffer<bf16>& x) {
    run.add(apps.set_layer_rtp.create_run());
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        run.add(app_for_layer(desc, apps, i).create_run(x, layers[i].proj, layers[i].rms, layers[i].rope_rms, layers[i].cache));
    }
}

static std::vector<int> top_k(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        idx[i] = (int)i;
    }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&v](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

static int run_lm_head_in_layer(const bench_args_t& args, qwen3_5vl_desc& desc, qwen3_5vl_npu_sequence& sequence,
                                npu_app_manager* manager, npu_apps_t& apps, npu_app& lm_app,
                                std::vector<layer_buffers_t>& layers, buffer<bf16>& x, const buffer<bf16>& x_input,
                                Q4NX& q4nx, const std::vector<int>& ids) {
    const int D = desc.D;
    const int rows = desc.get_lm_head_in_layer_rows();
    buffer<u8> weights = lm_app.create_bo_buffer<u8>(desc.get_lm_head_in_layer_w_size());
    buffer<bf16> norm = lm_app.create_bo_buffer<bf16>(D);
    buffer<bf16> logits = lm_app.create_bo_buffer<bf16>(rows);

    auto t0 = steady::now();
    buffer<u8> raw;
    q4nx.load_weights(raw, "lm_head.weight");
    desc.reorder_cpy_lm_head_in_layer(weights.data(), raw);
    weights.sync_to_device();
    buffer<bf16> norm_raw;
    q4nx.load_weights(norm_raw, "model.norm.weight");
    memcpy(norm.data(), norm_raw.data(), D * sizeof(bf16));
    norm.sync_to_device();
    header_print("info", "lm_head re-laid out for layer.xclbin: " << weights.size() / 1e6 << " MB, " << rows << " rows, "
                 << std::fixed << std::setprecision(2) << std::chrono::duration<double>(steady::now() - t0).count() << " s");
    sequence.gen_lm_head_in_layer_seq(lm_app.seq());

    auto reset_x = [&]() {
        memcpy(x.data(), x_input.data(), D * sizeof(bf16));
        x.sync_to_device();
    };

    // correctness: one token, layers then lm_head, logits vs the CPU on the NPU's own final x
    reset_x();
    xrt::runlist layers_run = manager->create_runlist();
    add_layer_runs(layers_run, desc, apps, layers, x);
    layers_run.execute();
    layers_run.wait();
    x.sync_from_device();
    std::vector<float> x_final = to_float(x.data(), D);
    xrt::run lm_run = lm_app.create_run(x, weights, norm, norm, logits);
    lm_run.start();
    lm_run.wait();
    logits.sync_from_device();

    const int vocab = desc.vocab_size;
    std::vector<float> npu = to_float(logits.data(), vocab);
    std::vector<float> ref = cpu_ref::lm_head_logits(raw, x_final, to_float(norm.data(), D), vocab);
    compare_t c = compare(npu, ref);
    std::vector<int> npu_top = top_k(npu, 5);
    std::vector<int> ref_top = top_k(ref, 5);
    int overlap = 0;
    for (int a : npu_top) {
        overlap += std::count(ref_top.begin(), ref_top.end(), a);
    }
    bool finite = std::all_of(npu.begin(), npu.end(), [](float v) { return std::isfinite(v); });
    std::cout << "\nlm_head in layer.xclbin vs fp32 CPU (Q8_0 lm_head on the NPU's final x):\n"
              << "  cos " << std::fixed << std::setprecision(6) << c.cos
              << "  rel " << std::scientific << std::setprecision(3) << c.rel << "  max_abs " << c.max_abs << std::defaultfloat
              << (finite ? "" : "   NOT FINITE") << "\n"
              << "  top-1 NPU " << npu_top[0] << " (" << npu[npu_top[0]] << ")  CPU " << ref_top[0] << " (" << ref[ref_top[0]] << ")"
              << (npu_top[0] == ref_top[0] ? "  same" : "  DIFFERENT") << "\n"
              << "  top-5 overlap " << overlap << " / 5";
    if ((int)ids.size() > args.context_len + 1) {
        std::cout << "\n  next token in the text: " << ids[args.context_len + 1];
    }
    std::cout << std::endl;

    // the next token's layers must not see anything left behind by the lm_head run
    saved_state_t start = save_state(layers, x);
    run_step(desc, manager, apps, layers, x, false, 0);
    uint64_t plain = state_checksum(save_state(layers, x), D);
    restore_state(start, layers, x);
    lm_run.start();
    lm_run.wait();
    run_step(desc, manager, apps, layers, x, false, 0);
    uint64_t after_lm = state_checksum(save_state(layers, x), D);
    restore_state(start, layers, x);
    std::cout << "  next step's layers: checksum " << std::hex << plain << " alone, " << after_lm << " right after lm_head"
              << std::dec << (plain == after_lm ? "  same" : "  DIFFERENT") << std::endl;

    // speed: the same run list with and without the lm_head run, and lm_head alone
    xrt::runlist with_lm = manager->create_runlist();
    add_layer_runs(with_lm, desc, apps, layers, x);
    with_lm.add(lm_app.create_run(x, weights, norm, norm, logits));
    xrt::runlist layers_only = manager->create_runlist();
    add_layer_runs(layers_only, desc, apps, layers, x);
    xrt::run lm_alone = lm_app.create_run(x, weights, norm, norm, logits);

    std::vector<double> layers_us, with_lm_us, lm_us;
    for (int i = 0; i < args.warmup + args.iterations; i++) {
        reset_x();
        double a = time_runlist_us(layers_only);
        reset_x();
        double b = time_runlist_us(with_lm);
        double l = time_run_us(lm_alone);
        if (i >= args.warmup) {
            layers_us.push_back(a);
            with_lm_us.push_back(b);
            lm_us.push_back(l);
        }
    }
    stats_t sl = compute_stats(layers_us), sw = compute_stats(with_lm_us), sa = compute_stats(lm_us);
    std::cout << std::fixed << std::setprecision(3)
              << "\nms per call, mean +- std (" << args.iterations << " iters):\n"
              << "  24 layers, one run list                " << sl.mean / 1000 << " +- " << sl.std / 1000 << "\n"
              << "  24 layers + lm_head, one run list      " << sw.mean / 1000 << " +- " << sw.std / 1000
              << "  ->  " << std::setprecision(2) << 1e6 / sw.mean << " tok/s (no host work)\n" << std::setprecision(3)
              << "  lm_head alone, back to back            " << sa.mean / 1000 << " +- " << sa.std / 1000
              << "  (" << std::setprecision(1) << weights.size() / sa.mean / 1e3 << " GB/s)" << std::endl;
    return (finite && c.cos > 0.99 && plain == after_lm) ? 0 : 1;
}

static double run_check(const bench_args_t& args, qwen3_5vl_desc& desc, size_t V_offset, npu_apps_t& apps,
                        std::vector<layer_buffers_t>& layers, buffer<bf16>& x, Q4NX& q4nx) {
    int L = args.context_len;
    int D = desc.D;
    size_t conv_count = desc.get_s_offset();
    size_t state_count = (size_t)desc.linear_num_v_heads * desc.linear_v_head_dim * desc.linear_qk_head_dim;

    std::vector<cpu_ref::gdn_state_t> states(desc.num_hidden_layers);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (!desc.is_attn_layer(i)) {
            const float* S = reinterpret_cast<const float*>(layers[i].cache.data() + conv_count);
            states[i].conv = to_float(layers[i].cache.data(), conv_count);
            states[i].S.assign(S, S + state_count);
        }
    }
    x.sync_to_device();
    std::vector<float> x_cpu = to_float(x.data(), D);
    std::vector<float> x_block = x_cpu;

    xrt::run rtp_run = apps.set_layer_rtp.create_run();
    rtp_run.start();
    rtp_run.wait();

    std::vector<std::vector<float>> x_after(desc.num_hidden_layers);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        run_layer_alone(desc, apps, layers, x, i);
        if (desc.is_attn_layer(i)) {
            x.sync_from_device();
            x_after[i] = to_float(x.data(), D);
        }
    }
    for (auto& layer : layers) {
        layer.cache.sync_from_device();
    }

    std::cout << "\nlayer  kind  checked    (NPU vs fp32 CPU reference; CPU restarts from the NPU's x at each block start)\n";
    double worst_cos = 1.0;
    double worst_quant_cos = 1.0;
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (is_first_linear_layer(desc, i)) {
            x_cpu = x_block;
        }
        if (desc.is_attn_layer(i)) {
            cpu_ref::attn_weights_t w = cpu_ref::load_attn_weights(q4nx, desc, i);
            cpu_ref::attn_step_t step;
            const bf16* k_cache = layers[i].cache.data();
            const bf16* v_cache = layers[i].cache.data() + V_offset;
            std::vector<float> k_rows = to_float(k_cache, (size_t)L * desc.DK);
            std::vector<float> v_rows = to_float(v_cache, (size_t)L * desc.DV);
            std::vector<float> x_in = x_cpu;
            x_cpu = cpu_ref::attn_layer(desc, w, k_rows.data(), v_rows.data(), L, L, x_in, step);
            x_block = x_after[i];
            print_compare(i, "attn", "new K", compare(to_float(k_cache + (size_t)L * desc.DK, desc.DK), step.k_new), worst_cos);
            print_compare(i, "attn", "new V", compare(to_float(v_cache + (size_t)L * desc.DV, desc.DV), step.v_new), worst_cos);
            print_compare(i, "attn", "x out", compare(x_block, x_cpu), worst_cos);
            for (cpu_ref::kv_quant_t q : args.kv_quants) {
                std::vector<float> k_q = k_rows;
                std::vector<float> v_q = v_rows;
                cpu_ref::quantize_rows(k_q.data(), L, desc.num_kv_heads, desc.DH, q);
                cpu_ref::quantize_rows(v_q.data(), L, desc.num_kv_heads, desc.DH, q);
                cpu_ref::attn_step_t q_step;
                std::vector<float> x_q = cpu_ref::attn_layer(desc, w, k_q.data(), v_q.data(), L, L, x_in, q_step, q);
                std::string label = std::string(cpu_ref::kv_quant_name(q)) + " KV";
                print_compare(i, label.c_str(), "attn core", compare(q_step.core_out, step.core_out), worst_quant_cos);
                print_compare(i, label.c_str(), "x out", compare(x_q, x_cpu), worst_quant_cos);
            }
        } else {
            cpu_ref::gdn_weights_t w = cpu_ref::load_gdn_weights(q4nx, desc, i);
            x_cpu = cpu_ref::gdn_layer(desc, w, states[i], x_cpu);
            const float* S = reinterpret_cast<const float*>(layers[i].cache.data() + conv_count);
            print_compare(i, "gdn", "state S", compare(std::vector<float>(S, S + state_count), states[i].S), worst_cos);
            print_compare(i, "gdn", "conv", compare(to_float(layers[i].cache.data(), conv_count), states[i].conv), worst_cos);
        }
    }
    std::cout << "\nworst cosine NPU vs CPU " << std::fixed << std::setprecision(6) << worst_cos << std::endl;
    if (!args.kv_quants.empty()) {
        std::cout << "worst cosine 8-bit KV vs bf16 KV (CPU) " << worst_quant_cos << std::endl;
    }
    return worst_cos;
}

static stats_t compute_stats(const std::vector<double>& values) {
    stats_t s;
    if (values.empty()) {
        return s;
    }
    s.min = *std::min_element(values.begin(), values.end());
    s.max = *std::max_element(values.begin(), values.end());
    for (double v : values) {
        s.mean += v;
    }
    s.mean /= values.size();
    for (double v : values) {
        s.std += (v - s.mean) * (v - s.mean);
    }
    s.std = std::sqrt(s.std / values.size());
    return s;
}

static bool output_is_sane(buffer<bf16>& x, int D) {
    x.sync_from_device();
    double sum_abs = 0;
    for (int i = 0; i < D; i++) {
        float v = (float)x[i];
        if (!std::isfinite(v)) {
            return false;
        }
        sum_abs += std::fabs(v);
    }
    return sum_abs > 0;
}

int main(int argc, char* argv[]) {
    bench_args_t args;
    if (!parse_args(argc, argv, args)) {
        return 0;
    }

    LM_Config config = make_config(args.model_path);
    qwen3_5vl_desc desc;
    desc.build(config);
    int MAX_L = round_up_pow2(args.context_len + 1);
    qwen3_5vl_npu_sequence sequence(desc, config, MAX_L);
    sequence.write_proj_mode = !args.no_proj_mode;
    size_t V_offset = sequence.get_V_offset();

    header_print("info", "Model:   " << args.model_path);
    header_print("info", "Xclbin:  " << std::filesystem::absolute(args.xclbin_path).string());
    header_print("info", "Context: " << args.context_len << " cached tokens, decoding token " << args.context_len + 1
                 << " | MAX_L " << MAX_L << " | iters " << args.iterations << " | warmup " << args.warmup);

    xrt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* layer_app_manager = npu.register_xclbin(args.xclbin_path);
    npu_app attn_layer = layer_app_manager->create_app();
    npu_app linear_layer = layer_app_manager->create_app();
    npu_app first_linear_layer = layer_app_manager->create_app();
    npu_app set_layer_rtp = layer_app_manager->create_app();

    std::vector<layer_buffers_t> layers(desc.num_hidden_layers);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        npu_app& owner = desc.is_attn_layer(i) ? attn_layer : linear_layer;
        if (desc.is_attn_layer(i)) {
            layers[i].proj = owner.create_bo_buffer<u8>(desc.attn_total_proj_weights_byte_size);
            layers[i].rms = owner.create_bo_buffer<bf16>(desc.attn_rms_weights_byte_size / sizeof(bf16));
            layers[i].rope_rms = owner.create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
            layers[i].cache = owner.create_bo_buffer<bf16>(desc.get_kv_cache_size(MAX_L));
        } else {
            layers[i].proj = owner.create_bo_buffer<u8>(desc.linear_total_proj_weights_byte_size);
            layers[i].rms = owner.create_bo_buffer<bf16>(desc.linear_rms_weights_byte_size / sizeof(bf16));
            layers[i].rope_rms = owner.create_bo_buffer<bf16>(desc.linear_ssm_weights_byte_size / sizeof(bf16));
            layers[i].cache = owner.create_bo_buffer<bf16>(desc.get_linear_cache_size() / sizeof(bf16));
        }
        layers[i].cache.memset((bf16)0);
    }
    buffer<bf16> x = attn_layer.create_bo_buffer<bf16>(desc.D * 2);

    auto load_start = steady::now();
    Q4NX q4nx(args.model_path);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (desc.is_attn_layer(i)) {
            desc.load_attn_weights(i, q4nx, layers[i].proj, layers[i].rms, layers[i].rope_rms);
        } else {
            buffer<bf16> alpha_cpu((size_t)desc.D * desc.linear_num_v_heads);
            buffer<bf16> beta_cpu((size_t)desc.D * desc.linear_num_v_heads);
            desc.load_linear_weights(i, q4nx, layers[i].proj, layers[i].rms, layers[i].rope_rms, alpha_cpu, beta_cpu);
        }
    }
    header_print("info", "Weights loaded in " << std::fixed << std::setprecision(2)
                 << std::chrono::duration<double>(steady::now() - load_start).count() << " s");

    std::mt19937 rng(args.seed);
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        if (desc.is_attn_layer(i)) {
            fill_attention_cache(layers[i], desc, V_offset, args.context_len, args.kv_scale, rng);
        } else {
            fill_linear_state(layers[i], desc, rng);
        }
    }
    fill_random_bf16(x.data(), desc.D, 0.05f, rng);
    std::fill(x.data() + desc.D, x.data() + 2 * desc.D, (bf16)1.0f);
    sequence.gen_layer_seq(linear_layer.seq(), 0, false, false);
    sequence.gen_layer_seq(first_linear_layer.seq(), 0, false, true);
    if (!args.prompt_ids.empty()) {
        std::vector<int> ids = read_ids(args.prompt_ids);
        if ((int)ids.size() <= args.context_len) {
            header_print("ERROR", "--prompt-ids has " << ids.size() << " tokens, need more than --len " << args.context_len);
            return 1;
        }
        buffer<u8> lm_head;
        q4nx.load_weights(lm_head, "lm_head.weight");
        npu_apps_t apps{attn_layer, linear_layer, first_linear_layer, set_layer_rtp};
        prefill_with_decode(desc, sequence, layer_app_manager, apps, layers, x, lm_head, ids, args.context_len);
        set_x_to_token(x, lm_head, ids[args.context_len], desc.D);
    }
    buffer<bf16> x_input(desc.D);
    memcpy(x_input.data(), x.data(), desc.D * sizeof(bf16));

    sequence.gen_set_context(set_layer_rtp.seq(), args.context_len + 1);
    sequence.gen_layer_seq(attn_layer.seq(), args.context_len + 1, true);
    set_rope_position(layers, desc, args.context_len);

    if (args.repeat > 0) {
        npu_apps_t apps{attn_layer, linear_layer, first_linear_layer, set_layer_rtp};
        x.sync_to_device();
        return run_determinism(args, desc, layer_app_manager, apps, layers, x) == 0 ? 0 : 1;
    }
    if (args.check) {
        npu_apps_t apps{attn_layer, linear_layer, first_linear_layer, set_layer_rtp};
        double worst = run_check(args, desc, V_offset, apps, layers, x, q4nx);
        return std::isfinite(worst) ? 0 : 1;
    }

    xrt::runlist layers_run = layer_app_manager->create_runlist();
    layers_run.add(set_layer_rtp.create_run());
    for (int i = 0; i < desc.num_hidden_layers; i++) {
        npu_app& app = desc.is_attn_layer(i) ? attn_layer : (is_first_linear_layer(desc, i) ? first_linear_layer : linear_layer);
        layers_run.add(app.create_run(x, layers[i].proj, layers[i].rms, layers[i].rope_rms, layers[i].cache));
    }

    if (args.lm_head_in_layer) {
        npu_app lm_app = layer_app_manager->create_app();
        npu_apps_t apps{attn_layer, linear_layer, first_linear_layer, set_layer_rtp};
        std::vector<int> ids = args.prompt_ids.empty() ? std::vector<int>() : read_ids(args.prompt_ids);
        return run_lm_head_in_layer(args, desc, sequence, layer_app_manager, apps, lm_app, layers, x, x_input, q4nx, ids);
    }
    if (!args.lm_head_xclbin.empty()) {
        bench_lm_head(args, desc, sequence, npu, attn_layer, layers_run, x, q4nx);
        return 0;
    }

    std::vector<double> step_ms;
    for (int step = 0; step < args.warmup + args.iterations; step++) {
        memcpy(x.data(), x_input.data(), desc.D * sizeof(bf16));
        x.sync_to_device();
        auto t0 = steady::now();
        layers_run.execute();
        layers_run.wait();
        double ms = std::chrono::duration<double, std::milli>(steady::now() - t0).count();
        if (step >= args.warmup) {
            step_ms.push_back(ms);
        }
    }
    bool sane = output_is_sane(x, desc.D);

    stats_t s = compute_stats(step_ms);
    std::cout << std::fixed << std::setprecision(3)
              << "\n24 layers, one run list per token (as the engine runs them):\n"
              << "  ms/token " << s.mean << " +- " << s.std << " [" << s.min << ", " << s.max << "]"
              << "  ->  " << std::setprecision(2) << 1000.0 / s.mean << " tok/s (layers only, no lm_head)\n"
              << "  output " << (sane ? "finite and non-zero" : "NOT SANE (NaN/Inf or all zero)") << std::endl;

    if (args.per_layer) {
        std::vector<double> attn_us, linear_us;
        xrt::run rtp_run = set_layer_rtp.create_run();
        for (int step = 0; step < args.iterations; step++) {
            rtp_run.start();
            rtp_run.wait();
            for (int i = 0; i < desc.num_hidden_layers; i++) {
                npu_app& app = desc.is_attn_layer(i) ? attn_layer : (is_first_linear_layer(desc, i) ? first_linear_layer : linear_layer);
                xrt::run run = app.create_run(x, layers[i].proj, layers[i].rms, layers[i].rope_rms, layers[i].cache);
                auto t0 = steady::now();
                run.start();
                run.wait();
                double us = std::chrono::duration<double, std::micro>(steady::now() - t0).count();
                if (desc.is_attn_layer(i)) {
                    attn_us.push_back(us);
                } else if (!is_first_linear_layer(desc, i)) {
                    linear_us.push_back(us);
                }
            }
        }
        stats_t a = compute_stats(attn_us);
        stats_t l = compute_stats(linear_us);
        std::cout << std::fixed << std::setprecision(1)
                  << "\nEach layer as its own run (includes one host wait per layer):\n"
                  << "  attention layer  " << a.mean << " +- " << a.std << " us  (x6 = " << a.mean * 6 / 1000 << " ms)\n"
                  << "  GDN layer        " << l.mean << " +- " << l.std << " us  (x18 = " << l.mean * 18 / 1000 << " ms)\n";
    }
    return sane ? 0 : 1;
}
