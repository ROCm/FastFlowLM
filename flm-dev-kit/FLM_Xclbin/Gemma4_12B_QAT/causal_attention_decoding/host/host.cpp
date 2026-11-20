// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "tensor_utils/safe_tensors.hpp"
#include "attn_ref.hpp"
#include <random>

namespace po = boost::program_options;

int round_up_to_multiple(int x, int multiple){
    if (multiple == 0) {
        return x; // Cannot divide by zero
    }

    return ((x + multiple - 1) / multiple) * multiple;
}
int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(3094), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(500), "Iteration");
    desc.add_options()("Window,w", po::value<int>()->default_value(0),
                       "Sliding attention window (0 = full causal)");
    desc.add_options()("Random,r", po::value<bool>()->default_value(true), "Use random q/k/v + CPU reference instead of the safetensors dump");
    const char* env_ref = std::getenv("FLM_REFERENCE_DIR");
    desc.add_options()("reference", po::value<std::string>()->default_value(
        env_ref ? std::string(env_ref) + "/gemma4_12b_qat_reference.safetensors" : ""),
        "With -r false: the safetensors dump to read q/k/v and the expected output from "
        "(default: $FLM_REFERENCE_DIR/gemma4_12b_qat_reference.safetensors; not shipped)");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    const bool RANDOM = vm["Random"].as<bool>();
    const std::string reference_path = vm["reference"].as<std::string>();
    if (!RANDOM && reference_path.empty()) {
        header_print("error", "No reference: -r false needs --reference or FLM_REFERENCE_DIR (references are not shipped with the dev kit)");
        return 1;
    }

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    // layer_0 of gemma-4-12B is a sliding attention layer: mask.local in the
    // reference only lets row r see rows (r - 1023 .. r), and rope for this
    // layer comes from rope.*_sliding_attention.  The kernel has no local mask,
    // so the window has to be applied here by handing it just the tail of the
    // kv cache -- otherwise every head also attends to the 2070 rows outside
    // the band, which flattens the softmax and shrinks the output.
    // WINDOW <= 0 keeps the old full-causal behaviour (layer_5 style).
    const int WINDOW = vm["Window"].as<int>();
    const int L_total = L_begin + L_wanted;
    const int kv_start = (WINDOW > 0 && L_total > WINDOW) ? L_total - WINDOW : 0;
    int L_effective = L_total - kv_start;
    std::cout << "L_effective: " << L_effective << " (kv rows " << kv_start
              << ".." << (L_total - 1) << ")" << std::endl;
    int L_padded = round_up_to_multiple(L_effective, 16);
    std::cout << "L_padded: " << L_padded << std::endl;
    int DH = 512;
    int HEADS = 16;
    int KV_HEADS = 1;
    int lc = 16;
    // Mirrors kernel/model_spec.h for the active ATTN_IMPL (4x2x1).  The kernel
    // headers are not on the host include path, so keep this in sync by hand --
    // it only affects how the per-head error table is grouped, not the run.
    int KV_HEADS_PER_CU = 4;

    int DQ = DH * HEADS;
    int DK = DH * KV_HEADS;

    buffer<bf16> q_in;
    buffer<bf16> k_in;
    buffer<bf16> v_in;
    buffer<bf16> aux_out;
    if (RANDOM) {
        header_print("info", "Generating random q/k/v, seed " << SEED);
        std::mt19937 rng(SEED);
        auto fill_random = [&rng](buffer<bf16> &b, size_t n, float stddev) {
            std::normal_distribution<float> dist(0.0f, stddev);
            b.resize(n);
            for (size_t i = 0; i < n; i++) b[i] = (bf16)dist(rng);
        };
        fill_random(q_in, (size_t)L_wanted * DQ, 0.3f);
        fill_random(k_in, (size_t)L_total * DK, 0.3f);
        fill_random(v_in, (size_t)L_total * DK, 1.0f);
    } else {
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
        references->load_weights(q_in, "layer_5.q_embed");
        references->load_weights(k_in, "layer_5.k_cache");
        references->load_weights(v_in, "layer_5.v_norm");
        references->load_weights(aux_out, "layer_5.attention_out");
    }
    std::cout << "q in:" << q_in.size() / DQ << std::endl;
    std::cout << "k in:" << k_in.size() / DK << std::endl;
    std::cout << "v in:" << v_in.size() / DK << std::endl;

    const int Q_VOLUME = HEADS * DH;
    const int K_VOLUME = L_padded * DK;
    const int V_VOLUME = L_padded * DK;
    const int O_VOLUME = HEADS * DH;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME);

    q_0.memset((bf16)0.0f);
    q_0.copy_from(q_in.data() + (L_wanted - 1) * DQ, Q_VOLUME);
    q_0.sync_to_device();

    buffer<bf16> ones(L_padded * DH * KV_HEADS);
    for (int i = 0; i < L_padded * DH * KV_HEADS; i++){
        ones[i] = 1.0f;
    }
    size_t offset = L_padded * DH * KV_HEADS;
    void * k_src_ptr = static_cast<void*>(k_in.data() + (size_t)kv_start * DK);
    void * v_src_ptr = static_cast<void*>(v_in.data() + (size_t)kv_start * DK);
    void * k_ptr = static_cast<void*>(kv_0.data());
    void * v_ptr = static_cast<void*>(kv_0.data() + offset);

    kv_0.memset((bf16)0.0f);
    size_t data_block_size_in_bytes = DH * KV_HEADS * sizeof(bf16);
    // move kv_cache
    for (int l = 0; l < L_effective; l++){
        memcpy(k_ptr, k_src_ptr, data_block_size_in_bytes);
        k_src_ptr += data_block_size_in_bytes;
        k_ptr += data_block_size_in_bytes;
        memcpy(v_ptr, v_src_ptr, data_block_size_in_bytes);
        v_src_ptr += data_block_size_in_bytes;
        v_ptr += data_block_size_in_bytes;
    }
    kv_0.sync_to_device();

    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_seq(app_0.seq(), L_padded, L_effective, lc, DH, HEADS, KV_HEADS);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, q_0, kv_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    buffer<bf16> o_ref(O_VOLUME);
    if (RANDOM) {
        attn_ref_config cfg{HEADS, KV_HEADS, DH};
        attn_cpu_ref(q_0.data(), 1, kv_0.data(), kv_0.data() + offset, L_effective, cfg, o_ref.data());
    } else {
        o_ref.copy_from(aux_out.data() + (L_wanted - 1) * DQ, O_VOLUME);
    }
    header_print("info", "Aux out");
    utils::print_matrix(o_ref, DH, 16);
    header_print("info", "O out");
    utils::print_matrix(o_0, DH, 16);

    error_metrics metrics = get_error_metrics(o_0, o_ref);
    print_error_metrics(metrics, RANDOM ? "NPU vs CPU reference" : "NPU vs reference dump");
    const bool pass = metrics.CosineSimilarity > 0.97f && metrics.NormalizedMSE < 6e-2f;

    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");
    if (RANDOM) {
        header_print("info", (pass ? "PASSED " : "FAILED!"));
        return pass ? 0 : 1;
    }
    return 0;
}