// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file bench_decode.cpp
/// \brief NPU-only decode timing for Hy-MT2-1.8B on this folder's layer.xclbin
/// \note Uses the engine's own hunyuan_desc + hunyuan_npu_sequence, so the
///       sequences are exactly what FLM_DLL/detail/hunyuan_npu runs. See README.md.

#include "hunyuan_model_def.hpp"
#include "hunyuan_npu_sequence.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/vm_args.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

static std::vector<int> parse_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        values.push_back(std::stoi(item));
    }
    return values;
}

static LM_Config load_config(const std::string& model_dir) {
    LM_Config cfg;
    cfg.model_path = model_dir;
    cfg.model_name = std::filesystem::path(model_dir).filename().string();
    std::ifstream f(std::filesystem::path(model_dir) / "config.json");
    cfg._json_config = nlohmann::json::parse(f);
    return cfg;
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

/// best (lowest) ms of `reps` executions of a runlist
template <typename RL>
static double time_runlist(RL& rl, int reps) {
    rl.execute();
    rl.wait();
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t = std::chrono::steady_clock::now();
        rl.execute();
        rl.wait();
        best = std::min(best, ms_since(t));
    }
    return best;
}

/// best ms of `reps` tokens run as the engine does today: layers, then lm_head as a second submission
template <typename RL>
static double time_separate(RL& layers_rl, RL& lm_head_rl, int reps) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t = std::chrono::steady_clock::now();
        layers_rl.execute();
        layers_rl.wait();
        lm_head_rl.execute();
        lm_head_rl.wait();
        best = std::min(best, ms_since(t));
    }
    return best;
}

static void fill_pseudo_random(bf16* dst, int n, float scale, uint32_t seed) {
    for (int j = 0; j < n; j++) {
        seed = seed * 1664525u + 1013904223u;
        const float u = (float)(seed >> 8) / (float)(1u << 24) - 0.5f;
        dst[j] = bf16(scale * u);
    }
}

/// cosine, max |a - b| and max |b| of two bf16 vectors
static void print_match(const char* name, const bf16* a, const bf16* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0, max_err = 0.0, max_ref = 0.0;
    for (int i = 0; i < n; i++) {
        const double va = (float)a[i];
        const double vb = (float)b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
        max_err = std::max(max_err, std::abs(va - vb));
        max_ref = std::max(max_ref, std::abs(vb));
    }
    std::cout << "  " << std::setw(8) << name << " cos " << std::setprecision(6) << dot / std::sqrt(na * nb)
              << ", max |err| " << max_err << " (max |ref| " << max_ref << ")" << std::endl;
}

/// One NPU decode layer at context 1 vs the engine's CPU functions. With one key
/// the softmax is exactly 1, so each Q head's attention output is its KV head's V
/// row and Q drops out: this checks the K, V, O, up/gate and down projections (all
/// Q4_K) without a CPU attention. K is checked as the cache row (QK norm applied;
/// the bench's RoPE table is cos 1, sin 0).
template <typename MGR>
static void check_layer(hunyuan_desc& desc, hunyuan_npu_sequence& sequence, MGR* mgr, npu_app& rtp_app,
                        npu_app& layer_app, buffer<bf16>& x, buffer<u8>& proj, buffer<bf16>& rms,
                        buffer<bf16>& rope, buffer<bf16>& kv, int layer) {
    const int D = (int)desc.D;
    const int DH = (int)desc.DH;
    const int DK = (int)desc.DK;
    const int DQ = (int)desc.DQ;
    const int I = (int)desc.intermediate_size;
    const int q_per_kv = (int)(desc.num_heads / desc.num_kv_heads);
    const float eps = desc.rms_norm_eps;
    std::vector<bf16> saved_x(x.data(), x.data() + D);

    fill_pseudo_random(x.data(), D, 0.02f, 777u + (uint32_t)layer);
    std::vector<bf16> x_in(x.data(), x.data() + D);
    x.sync_to_device();
    kv.memset(bf16(0.0f));
    kv.sync_to_device();

    sequence.gen_layer_rtp_seq(rtp_app.seq(), 1);
    sequence.gen_layer_seq(layer_app.seq(), 1);
    auto rl = mgr->create_runlist();
    rl.add(rtp_app.create_run());
    rl.add(layer_app.create_run(x, proj, rms, rope, kv));
    rl.execute();
    rl.wait();
    x.sync_from_device();
    kv.sync_from_device();

    const u8* w = proj.data();
    std::vector<bf16> xn(D), k(DK), v(DK), att(DQ), o(D), h1(D), xn2(D), ug(2 * I), hid(I), dn(D), out(D);
    cpu_func::_rms_norm(xn.data(), x_in.data(), rms.data(), D, 1, 0, 0, eps);
    cpu_func::_dequant_mm(k.data(), w + desc.get_attn_k_offset(), xn.data(), 1, D, DK);
    cpu_func::_dequant_mm(v.data(), w + desc.get_attn_v_offset(), xn.data(), 1, D, DK);
    for (int h = 0; h < DK / DH; h++) {
        cpu_func::_rms_norm(k.data() + h * DH, k.data() + h * DH, rope.data() + desc.RoPE_DIM + DH, DH, 1, 0, 0, eps);
    }
    for (int h = 0; h < DQ / DH; h++) {
        std::copy(v.begin() + (size_t)(h / q_per_kv) * DH, v.begin() + (size_t)(h / q_per_kv + 1) * DH, att.begin() + (size_t)h * DH);
    }
    cpu_func::_dequant_mm(o.data(), w + desc.get_attn_o_offset(), att.data(), 1, DQ, D);
    for (int j = 0; j < D; j++) h1[j] = bf16((float)x_in[j] + (float)o[j]);
    cpu_func::_rms_norm(xn2.data(), h1.data(), rms.data() + D, D, 1, 0, 0, eps);
    cpu_func::_dequant_mm_upgate(ug.data(), w + desc.get_upgate_offset(), xn2.data(), 1, D, I,
                                 hunyuan_desc::UP_GATE_OUTDIM_REORDEER);
    cpu_func::_exp_glu(hid.data(), ug.data(), 1, I, hunyuan_desc::UP_GATE_OUTDIM_REORDEER);
    cpu_func::_dequant_mm(dn.data(), w + desc.get_down_offset(), hid.data(), 1, I, D);
    for (int j = 0; j < D; j++) out[j] = bf16((float)h1[j] + (float)dn[j]);

    const int half = DK / 2;
    std::vector<bf16> k_npu(DK), v_npu(DK);
    std::copy(kv.data() + sequence.K03_offset, kv.data() + sequence.K03_offset + half, k_npu.begin());
    std::copy(kv.data() + sequence.K47_offset, kv.data() + sequence.K47_offset + half, k_npu.begin() + half);
    std::copy(kv.data() + sequence.V03_offset, kv.data() + sequence.V03_offset + half, v_npu.begin());
    std::copy(kv.data() + sequence.V47_offset, kv.data() + sequence.V47_offset + half, v_npu.begin() + half);

    std::cout << "layer " << layer << " check (ctx 1):" << std::endl;
    print_match("V", v_npu.data(), v.data(), DK);
    print_match("K", k_npu.data(), k.data(), DK);
    print_match("x out", x.data(), out.data(), D);
    std::vector<bf16> delta_npu(D), delta_cpu(D);
    for (int j = 0; j < D; j++) {
        delta_npu[j] = bf16((float)x[j] - (float)x_in[j]);
        delta_cpu[j] = bf16((float)out[j] - (float)x_in[j]);
    }
    print_match("out - in", delta_npu.data(), delta_cpu.data(), D);

    kv.memset(bf16(0.0f));
    kv.sync_to_device();
    std::copy(saved_x.begin(), saved_x.end(), x.data());
    x.sync_to_device();
}

/// NPU lm_head (final norm + Q8_0 matvec) on a pseudo-random x vs the engine's CPU path
template <typename RL>
static void check_lm_head(hunyuan_desc& desc, RL& lm_head_rl, buffer<bf16>& logits, buffer<u8>& lm_head_w,
                          buffer<bf16>& x, buffer<bf16>& final_rms) {
    const int D = (int)desc.D;
    const int V = (int)desc.vocab_size_padded;
    const int vocab = (int)desc.vocab_size;
    std::vector<bf16> saved_x(x.data(), x.data() + D);

    fill_pseudo_random(x.data(), D, 4.0f, 12345u);
    x.sync_to_device();
    lm_head_rl.execute();
    lm_head_rl.wait();
    logits.sync_from_device();

    std::vector<bf16> xn(D);
    std::vector<bf16> ref(V);
    cpu_func::_rms_norm(xn.data(), x.data(), final_rms.data(), D, 1, 0, 0, desc.rms_norm_eps);
    cpu_func::_dequant_mm_lm_head(ref.data(), lm_head_w.data(), xn.data(), 1, D, V);

    double dot = 0.0, nn = 0.0, nr = 0.0, max_err = 0.0, max_ref = 0.0;
    int top_npu = 0, top_ref = 0;
    for (int i = 0; i < vocab; i++) {
        const double a = (float)logits[i];
        const double b = (float)ref[i];
        dot += a * b;
        nn += a * a;
        nr += b * b;
        max_err = std::max(max_err, std::abs(a - b));
        max_ref = std::max(max_ref, std::abs(b));
        if (a > (float)logits[top_npu]) top_npu = i;
        if (b > (float)ref[top_ref]) top_ref = i;
    }
    std::cout << "lm_head check: cos " << std::setprecision(6) << dot / std::sqrt(nn * nr)
              << ", max |err| " << max_err << " (max |ref| " << max_ref << "), top-1 npu "
              << top_npu << " cpu " << top_ref << std::endl << std::endl;

    std::copy(saved_x.begin(), saved_x.end(), x.data());
    x.sync_to_device();
}

int main(int argc, char** argv) {
    arg_utils::po::options_description desc_opts("Allowed options");
    arg_utils::po::variables_map vm;
    desc_opts.add_options()("model", arg_utils::po::value<std::string>()->default_value("/path/to/models/Hy-MT2-1.8B-NPU2"), "Model folder (config.json + model.q4nx)");
    desc_opts.add_options()("xclbin", arg_utils::po::value<std::string>()->default_value("build/HUNYUAN_DENSE_1_8B/xclbins/layer.xclbin"), "layer.xclbin to time");
    desc_opts.add_options()("ctx", arg_utils::po::value<std::string>()->default_value("8,1024,4096,16384,32768"), "Context lengths");
    desc_opts.add_options()("reps", arg_utils::po::value<int>()->default_value(20), "Timed runs per point (best is reported)");
    desc_opts.add_options()("check", "Also check the NPU lm_head logits against the engine's CPU reader");
    desc_opts.add_options()("help,h", "Show this help");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc_opts), vm);
    if (vm.count("help")) {
        std::cout << desc_opts << std::endl;
        return 0;
    }

    const std::string model_dir = vm["model"].as<std::string>();
    const std::vector<int> contexts = parse_list(vm["ctx"].as<std::string>());
    const int reps = vm["reps"].as<int>();
    const int longest = *std::max_element(contexts.begin(), contexts.end());
    const uint32_t kv_rows = (uint32_t)((longest + 1 + 127) / 128 * 128);

    LM_Config cfg = load_config(model_dir);
    hunyuan_desc desc;
    desc.build(cfg);
    hunyuan_npu_sequence sequence(desc, cfg, kv_rows);
    const int layers = (int)desc.num_hidden_layers;

    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* mgr = npu.register_xclbin(vm["xclbin"].as<std::string>());
    npu_app layer_app = mgr->create_app();
    npu_app rtp_app = mgr->create_app();
    npu_app lm_head_app = mgr->create_app();

    buffer<bf16> x = npu.create_bo_buffer<bf16>(desc.D);
    std::vector<buffer<u8>> proj(layers);
    std::vector<buffer<bf16>> rms(layers);
    std::vector<buffer<bf16>> rope(layers);
    std::vector<buffer<bf16>> kv(layers);

    Q4NX q4nx(model_dir);
    for (int i = 0; i < layers; i++) {
        proj[i] = npu.create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        rms[i] = npu.create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        rope[i] = npu.create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
        kv[i] = npu.create_bo_buffer<bf16>(sequence.kv_buffer_size);
        desc.load_layer_weights(i, q4nx, proj[i], rms[i], rope[i]);
        for (uint32_t j = 0; j < desc.RoPE_DIM; j++) {
            rope[i][j] = bf16(j < desc.RoPE_DIM / 2 ? 1.0f : 0.0f);
        }
        kv[i].memset(bf16(0.0f));
        proj[i].sync_to_device();
        rms[i].sync_to_device();
        rope[i].sync_to_device();
        kv[i].sync_to_device();
    }
    for (uint32_t j = 0; j < desc.D; j++) {
        x[j] = bf16(0.01f * (float)((j % 17) - 8));
    }
    x.sync_to_device();

    buffer<u8> lm_head_w = npu.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    memset(lm_head_w.data(), 0, lm_head_w.size());
    {
        buffer<u8> w;
        q4nx.load_weights(w, desc.lm_head.name);
        desc.reorder_cpy_lm_head(lm_head_w.data(), w);
    }
    lm_head_w.sync_to_device();
    buffer<bf16> final_rms = npu.create_bo_buffer<bf16>(desc.D);
    final_rms.memset(bf16(1.0f));
    final_rms.sync_to_device();
    buffer<bf16> logits = npu.create_bo_buffer<bf16>(desc.vocab_size_padded);

    const double proj_mb = desc.total_proj_weights_byte_size / 1e6;
    const double lm_head_mb = desc.get_lm_head_w_size() / 1e6;
    std::cout << std::endl << "weights per layer " << std::fixed << std::setprecision(2) << proj_mb
              << " MB, lm_head " << lm_head_mb << " MB, KV " << (desc.DK + desc.DV) * 2
              << " B per token per layer" << std::endl;

    sequence.gen_lm_head_seq(lm_head_app.seq(), desc.vocab_size_padded, desc.D);
    auto lm_head_rl = mgr->create_runlist();
    lm_head_rl.add(lm_head_app.create_run(logits, lm_head_w, x, final_rms));
    const double lm_head_ms = time_runlist(lm_head_rl, reps);
    std::cout << "lm_head: " << std::setprecision(3) << lm_head_ms << " ms, "
              << std::setprecision(1) << lm_head_mb / lm_head_ms << " GB/s" << std::endl << std::endl;
    if (vm.count("check")) {
        check_lm_head(desc, lm_head_rl, logits, lm_head_w, x, final_rms);
        for (int layer : {0, layers / 2, layers - 1}) {
            check_layer(desc, sequence, mgr, rtp_app, layer_app, x, proj[layer], rms[layer], rope[layer], kv[layer], layer);
        }
        std::cout << std::endl;
    }

    std::cout << std::setw(8) << "context" << std::setw(12) << "32 layers"
              << std::setw(12) << "per layer" << std::setw(12) << "KV MB" << std::setw(14) << "all bytes"
              << std::setw(10) << "GB/s" << std::setw(14) << "token+lm_head"
              << std::setw(12) << "separate" << std::setw(12) << "one list" << std::endl;
    std::cout << std::setw(8) << "" << std::setw(12) << "(ms)" << std::setw(12) << "(us)" << std::setw(12) << "/token"
              << std::setw(14) << "(MB)" << std::setw(10) << "" << std::setw(14) << "(tok/s)"
              << std::setw(12) << "(ms)" << std::setw(12) << "(ms)" << std::endl;

    for (int L : contexts) {
        sequence.gen_layer_rtp_seq(rtp_app.seq(), L + 1);
        sequence.gen_layer_seq(layer_app.seq(), L + 1);
        auto rl = mgr->create_runlist();
        rl.add(rtp_app.create_run());
        for (int i = 0; i < layers; i++) {
            rl.add(layer_app.create_run(x, proj[i], rms[i], rope[i], kv[i]));
        }
        const double ms = time_runlist(rl, reps);

        auto fused_rl = mgr->create_runlist();
        fused_rl.add(rtp_app.create_run());
        for (int i = 0; i < layers; i++) {
            fused_rl.add(layer_app.create_run(x, proj[i], rms[i], rope[i], kv[i]));
        }
        fused_rl.add(lm_head_app.create_run(logits, lm_head_w, x, final_rms));
        const double fused_ms = time_runlist(fused_rl, reps);
        const double separate_ms = time_separate(rl, lm_head_rl, reps);

        const size_t kv_rows_read = (size_t)(L + 1 + 15) / 16 * 16;
        const size_t v_bfp16_rows = (size_t)hunyuan_npu_sequence::v_bfp16_chunks(L + 1) * 16;
        const double kv_bytes = v_bfp16_rows * (desc.DK + desc.DV) * 9 / 8
                              + (kv_rows_read - v_bfp16_rows) * (desc.DK + desc.DV) * sizeof(bf16);
        const double kv_mb = kv_bytes * layers / 1e6;
        const double all_mb = proj_mb * layers + kv_mb;
        std::cout << std::setw(8) << L << std::setw(12) << std::setprecision(3) << ms
                  << std::setw(12) << std::setprecision(1) << ms * 1000.0 / layers
                  << std::setw(12) << kv_mb << std::setw(14) << all_mb
                  << std::setw(10) << all_mb / ms << std::setw(14) << 1000.0 / (ms + lm_head_ms)
                  << std::setw(12) << std::setprecision(3) << separate_ms << std::setw(12) << fused_ms << std::endl;
    }
    std::cout << std::endl;
    return 0;
}
