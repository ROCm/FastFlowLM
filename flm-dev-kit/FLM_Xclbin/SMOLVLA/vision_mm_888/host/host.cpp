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

namespace po = boost::program_options;

void Silu(int dim, bf16* x, bf16* o) {
    for (int i = 0; i < dim; i++) {
        bf16 xi = x[i];
        o[i] = (bf16)(xi / (1.0f + std::exp(-xi)));
    }
}
int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("Row,r", po::value<int>()->default_value(0), "Row of B to use as input x");
    desc.add_options()("Trace,t", po::value<int>()->default_value(131072), "Trace size");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("EnableTrace,e", po::value<bool>()->default_value(false), "Enable trace");

    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int ROW = vm["Row"].as<int>();
    const int TRACE_SIZE = vm["Trace"].as<int>();
    const bool enable_trace = vm["EnableTrace"].as<bool>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const size_t SEQ = 1024;  // real sequence length in the reference
    const size_t m = 128;
    const size_t k = 256;
    const size_t n = 64;

    int bias = 1;
    bool pass = true;

    std::string reference_path = "/path/to/models/SmolVLA/ref.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    std::string weight_path = "/path/to/models/SmolVLA/model.safetensors";
    std::unique_ptr<SafeTensors> weights = std::make_unique<SafeTensors>(weight_path);

    buffer<bf16> x_in;
    buffer<bf16> w_in;
    buffer<bf16> b_in;
    buffer<bf16> y_ref;
    int M;
    int K;
    int N;

    if (bias == 1){
        references->load_weights(x_in, "vision.camera1.layer.2.layer_norm2");
        weights->load_weights(w_in, "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.mlp.fc1.weight");
        weights->load_weights(b_in, "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.2.mlp.fc1.bias");
        references->load_weights(y_ref, "vision.camera1.layer.2.mlp_fc1_act");
        M = 1024;    // padded rows for the kernel tile
        K = 768;
        N = 3072;
    }
    else{

        references->load_weights(x_in, "layer_0.input_layernorm");
        weights->load_weights(w_in, "layer_0.qkv_proj.weight");
        references->load_weights(y_ref, "layer_0.qkv_proj");
        M = 64;    // padded rows for the kernel tile
        K = 768;
        N = 1792;
    }

    const int Y_VOLUME = M * N;
    const int X_VOLUME = M * K;
    int bias_offset = (bias == 1)? N : 0;
    const int W_VOLUME = N * K + bias_offset;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/mm.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

    npu_app app_0 = mm_app_mgr->create_app();

    buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME);

    x_0.memset((bf16)(0.0f));  // zero-pad rows [SEQ, M)
    memcpy(x_0.data(), x_in.data(), SEQ * K * sizeof(bf16));

    if(bias == 1){
        memcpy(w_0.data(), b_in.data(), N * sizeof(bf16));
    }
    memcpy(w_0.data() + bias_offset, w_in.data(), N * K * sizeof(bf16));

    y_0.memset((bf16)(-1.0f));
    w_0.sync_to_device();
    x_0.sync_to_device();
    y_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        int Iterations = 1000;
        gen_mm_seq(app_0.seq(), M, K, N, m, k, n, 0, 2);
        for (int i = 0; i < Iterations; i++) {
            time_utils::time_point start = time_utils::now();
            app_0(y_0, x_0, w_0);
            time_utils::time_point stop = time_utils::now();
            npu_time.first += time_utils::duration_us(start, stop).first;
        }
        npu_time.first /= Iterations;
        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
        MSG_BONDLINE(60);

        y_0.sync_from_device(); 

        buffer<bf16> y_wanted(SEQ * N);
        memcpy(y_wanted.data(), y_0.data(), SEQ * N * sizeof(bf16));
        header_print("info", "NPU result:");
        utils::print_matrix(y_wanted, N);
        header_print("info", "Ref result:");
        utils::print_matrix(y_ref, N);

        error_metrics metrics = get_error_metrics(y_wanted, y_ref);
        print_error_metrics(metrics);

        header_print("info", "Finished running kernel");
        header_print("info", "Running kernel with bare call (run based).");
        utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
    }

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}