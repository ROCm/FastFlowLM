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
    const char* env_ref = std::getenv("FLM_REFERENCE_DIR");
    desc.add_options()("reference", po::value<std::string>()->default_value(
        env_ref ? std::string(env_ref) + "/matmul_test.safetensors" : ""),
        "The x / w / y safetensors to check against (default: $FLM_REFERENCE_DIR/matmul_test.safetensors; not shipped)");

    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int ROW = vm["Row"].as<int>();
    const int TRACE_SIZE = vm["Trace"].as<int>();
    const bool enable_trace = vm["EnableTrace"].as<bool>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const size_t M = 4096;  
    const size_t K = 2048;
    const size_t N = 4096;

    const size_t m = 64;
    const size_t k = 512;
    const size_t n = 64;
    const int Y_VOLUME = M * N;
    const int X_VOLUME = M * K;
    const int W_VOLUME = N * K;

    bool pass = true;

    const std::string reference_path = vm["reference"].as<std::string>();
    if (reference_path.empty()) {
        header_print("error", "No reference: pass --reference or set FLM_REFERENCE_DIR (references are not shipped with the dev kit)");
        return 1;
    }
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    buffer<bf16> x_in;
    references->load_weights(x_in, "x");
    buffer<bf16> w_in;
    references->load_weights(w_in, "w");
    buffer<bf16> y_ref;
    references->load_weights(y_ref, "y");

    header_print("info", "ref result:");
    utils::print_matrix(y_ref, N);
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

    memcpy(x_0.data(), x_in.data(), X_VOLUME * sizeof(bf16));
    memcpy(w_0.data(), w_in.data(), W_VOLUME * sizeof(bf16));

    y_0.memset((bf16)(-1.0f));
    w_0.sync_to_device();
    x_0.sync_to_device();
    y_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    unsigned long data_size = M * K * N / n / 8 + K * N * M / m / 4;
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        int Iterations = 500;
        gen_mm_seq(app_0.seq(), M, K, N, m, k, n, 0, 0);
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
        header_print("info", "NPU result:");
        utils::print_matrix(y_0, N);

        // error_metrics metrics = get_error_metrics(y_0, y_ref);
        // print_error_metrics(metrics);

        header_print("info", "Finished running kernel");
        header_print("info", "Running kernel with bare call (run based).");
        utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
        header_print("info", "Data size: " << std::to_string(data_size));
        header_print("info", "Bandwidth: " + std::to_string(data_size * 2/ npu_time.first * 1e6 / 1e9) + " GB/s");
    }

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}