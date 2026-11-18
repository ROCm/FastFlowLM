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
    desc.add_options()("Window,w", po::value<int>()->default_value(1024),
                       "Sliding attention window (0 = full causal)");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    // load reference data
    std::string reference_path = "/scratch/Gemma4-12B-QAT-NPU2/gemma4_12b_qat_reference.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    buffer<bf16> q_in;
    references->load_weights(q_in, "layer_0.q_embed");
    buffer<bf16> k_in;
    references->load_weights(k_in, "layer_0.k_cache");
    buffer<bf16> v_in;
    references->load_weights(v_in, "layer_0.v_norm");
    buffer<bf16> aux_out;
    references->load_weights(aux_out, "layer_0.attention_out");

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
    int DH = 256;
    int HEADS = 16;
    int KV_HEADS = 8;
    int lc = 16;
    // Mirrors kernel/model_spec.h for the active ATTN_IMPL (4x2x1).  The kernel
    // headers are not on the host include path, so keep this in sync by hand --
    // it only affects how the per-head error table is grouped, not the run.
    int KV_HEADS_PER_CU = 4;

    int DQ = DH * HEADS;
    int DK = DH * KV_HEADS;

    std::cout << "q in:" << q_in.size() / DH << std::endl;
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
    o_ref.copy_from(aux_out.data() + (L_wanted - 1) * DQ, O_VOLUME);
    header_print("info", "Aux out");
    utils::print_matrix(o_ref, DH, 16);
    header_print("info", "O out");
    utils::print_matrix(o_0, DH, 16);

    error_metrics metrics = get_error_metrics(o_0, o_ref);
    print_error_metrics(metrics);

    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");
    return 0;
}