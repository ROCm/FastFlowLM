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

/// \brief fp32 CPU reference of the kernel: y[r][n] = sum_k x[r][k] * w[n][k] (+ b[n]).
void mm_cpu_ref(const bf16* x, const bf16* w, const bf16* b, bf16* y, int rows, int K, int N) {
    for (int r = 0; r < rows; r++) {
        for (int n = 0; n < N; n++) {
            float acc = b ? (float)b[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += (float)x[(size_t)r * K + k] * (float)w[(size_t)n * K + k];
            }
            y[(size_t)r * N + n] = (bf16)acc;
        }
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
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Random", po::value<bool>()->default_value(true), "Use random x/w + CPU reference instead of the safetensors dump");
    const char* env_ref = std::getenv("FLM_REFERENCE_DIR");
    const std::string ref_dir = env_ref ? env_ref : "";
    desc.add_options()("reference", po::value<std::string>()->default_value(ref_dir.empty() ? "" : ref_dir + "/expert_reference.safetensors"),
        "With --Random false: activations dump (default: $FLM_REFERENCE_DIR/expert_reference.safetensors; not shipped)");
    desc.add_options()("weights", po::value<std::string>()->default_value(ref_dir.empty() ? "" : ref_dir + "/expert_weights_pad256.safetensors"),
        "With --Random false: the matching weights (default: $FLM_REFERENCE_DIR/expert_weights_pad256.safetensors; not shipped)");

    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int ROW = vm["Row"].as<int>();
    const int TRACE_SIZE = vm["Trace"].as<int>();
    const bool enable_trace = vm["EnableTrace"].as<bool>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const size_t SEQ = 50;  // real sequence length in the reference
    const size_t m = 16;
    const size_t k = 256;
    const size_t n = 64;

    int bias = 0;
    bool pass = true;

    const bool RANDOM = vm["Random"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    const std::string reference_path = vm["reference"].as<std::string>();
    const std::string weight_path = vm["weights"].as<std::string>();
    if (!RANDOM && (reference_path.empty() || weight_path.empty())) {
        header_print("error", "No reference: --Random false needs --reference and --weights, or FLM_REFERENCE_DIR (references are not shipped with the dev kit)");
        return 1;
    }

    buffer<bf16> x_in;
    buffer<bf16> w_in;
    buffer<bf16> b_in;
    buffer<bf16> y_ref;
    int M = 64;    // padded rows for the kernel tile
    int K = 768;
    int N = (bias == 1) ? 768 : 1792;

    if (RANDOM) {
        header_print("info", "Generating random x/w, seed " << SEED);
        std::mt19937 rng(SEED);
        auto fill_random = [&rng](buffer<bf16> &b, size_t n, float stddev) {
            std::normal_distribution<float> dist(0.0f, stddev);
            b.resize(n);
            for (size_t i = 0; i < n; i++) b[i] = (bf16)dist(rng);
        };
        fill_random(x_in, SEQ * K, 1.0f);
        fill_random(w_in, (size_t)N * K, 1.0f / std::sqrt((float)K));
        if (bias == 1) fill_random(b_in, N, 1.0f);
        y_ref.resize(SEQ * N);
        mm_cpu_ref(x_in.data(), w_in.data(), bias == 1 ? b_in.data() : nullptr, y_ref.data(), SEQ, K, N);
    } else {
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
        std::unique_ptr<SafeTensors> weights = std::make_unique<SafeTensors>(weight_path);
        if (bias == 1) {
            references->load_weights(x_in, "suffix.action_time");
            weights->load_weights(w_in, "action_time_mlp_out.weight");
            weights->load_weights(b_in, "action_time_mlp_out.bias");
            references->load_weights(y_ref, "suffix.embs");
        } else {
            references->load_weights(x_in, "layer_0.input_layernorm");
            weights->load_weights(w_in, "layer_0.qkv_proj.weight");
            references->load_weights(y_ref, "layer_0.qkv_proj");
        }
    }

    const int Y_VOLUME = M * N;
    const int X_VOLUME = M * K;
    int bias_offset = (bias == 1)? N : 0;
    const int W_VOLUME = N * K + bias_offset;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/expert.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

    npu_app app_0 = mm_app_mgr->create_app();

    buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME);

    x_0.memset((bf16)(0.0f));  // zero-pad rows [SEQ, M)
    memcpy(x_0.data(), x_in.data(), SEQ * K * sizeof(bf16));

    memcpy(w_0.data(), w_in.data(), N * K * sizeof(bf16));

    y_0.memset((bf16)(-1.0f));
    w_0.sync_to_device();
    x_0.sync_to_device();
    y_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    unsigned long data_size = M * K * N / n / 4 + K * N * M / m / 4;
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        int Iterations = 1000;
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

        buffer<bf16> y_wanted(SEQ * N);
        memcpy(y_wanted.data(), y_0.data(), SEQ * N * sizeof(bf16));
        header_print("info", "NPU result:");
        utils::print_matrix(y_wanted, N);
        header_print("info", "Ref result:");
        utils::print_matrix(y_ref, N);

        error_metrics metrics = get_error_metrics(y_wanted, y_ref);
        print_error_metrics(metrics, RANDOM ? "NPU vs CPU reference" : "NPU vs reference dump");
        if (RANDOM) {
            pass = metrics.CosineSimilarity > 0.97f && metrics.NormalizedMSE < 6e-2f;
        }

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
    return pass ? 0 : 1;
}