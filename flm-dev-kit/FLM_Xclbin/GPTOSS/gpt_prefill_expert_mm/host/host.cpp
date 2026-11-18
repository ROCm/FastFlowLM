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
#include "llama/llama_npu_sequence.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"

#include "tensor_utils/q4_npu_eXpress.hpp"
#include "sequence_gen.hpp"

namespace po = boost::program_options;

float get_normalized_l2_error(buffer<bf16>& y, buffer<bf16>& y_ref){
    float sum_error = 0;
    float sum_ref = 0;
    for (int i = 0; i < y.size(); i++){
        sum_error += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
        sum_ref += y_ref[i] * y_ref[i];
    }
    return sqrt(sum_error / sum_ref);
}

void linear(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w, int M, int K, int N){
    assert((M * K) == x.size());
    assert((K * N) == w.size());
    assert((M * N) == y.size());
    for (int row = 0; row < M; row++){
        for (int col = 0; col < N; col++){
            float sum = 0;
            for (int k = 0; k < K; k++){
                sum = sum + (x[row * K + k] * w[k * N + col]);
            }
            y[row * N + col] = (bf16)sum;
        }
    }
}

void load_data(buffer<bf16>& y, std::string filename){
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }
    // get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    y.resize(file_size / sizeof(float));
    buffer<float> y_float(y.size());
    file.read(reinterpret_cast<char*>(y_float.data()), file_size);
    file.close();
    y.resize(y_float.size());
    for (int i = 0; i < y.size(); i++){
        y[i] = (bf16)y_float[i];
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
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("L_end,e", po::value<int>()->default_value(4096), "L_end");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(32), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    const int L_end = vm["L_end"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int M = L_end;
    int K = 2944;
    int N = 32;
    int m = 64;
    int k = 128;
    int n = 4;

    int X_VOLUME = M * K;
    int W_VOLUME = K * N;
    int Y_VOLUME = M * N;
    // initialize npu

    npu_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "build/xclbins/mm.xclbin";
    npu_app_desc accel_desc_0;
    accel_desc_0.xclbin_name = xclbin_name;
    accel_desc_0.app_name = "mm_compiled";

    npu_app app_0 = npu_instance.create_app(accel_desc_0);

    // define cache buffers
    buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME, 3);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME, 4);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME, 5);
    
    for (int i = 0; i < X_VOLUME; i++){
        x_0[i] = utils::getRand(-4, 4);
    }
    x_0.sync_to_device();

    for (int i = 0; i < W_VOLUME; i++){
        w_0[i] = utils::getRand(-4, 4);
    }
    w_0.sync_to_device();

    buffer<bf16> w_ref(W_VOLUME);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < K; j++){
            w_ref[j * N + i] = w_0[i * K + j];
        }
    }

    y_0.memset((bf16)-1.0f);
    y_0.sync_to_device();

    // compute y_ref
    buffer<bf16> y_ref(Y_VOLUME);
    linear(y_ref, x_0, w_ref, M, K, N);
    

    time_utils::time_with_unit npu_time = {0.0, "us"};

    generate_mm_sequence(app_0.instr_seq, M, K, N, m, k, n, 0);

    header_print("info", "Warm up.");
    app_0(y_0, x_0, w_0);
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(y_0, x_0, w_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    y_0.sync_from_device();
    
    header_print("info", "y_0");
    utils::print_matrix(y_0, N);

    header_print("info", "y_ref");
    utils::print_matrix(y_ref, N);

    buffer<float> cosine_similarity_table = buffer<float>(M * N);
    for (int l = 0; l < M; l++){
        for (int i = 0; i < N; i++){
            buffer<bf16> y_wanted_block(y_0.data() + l * N, N);
            buffer<bf16> y_ref_block(y_ref.data() + l * N, N);
            auto error_metrics = get_error_metrics(y_wanted_block, y_ref_block);
            cosine_similarity_table[l * N + i] = error_metrics.CosineSimilarity;
            // cosine_similarity_table[l * DQ / d + i] = error_metrics.RelativeL2;
            // cosine_similarity_table[l * DQ / d + i] = float(aux_out_wanted_block[0]);
        }
    }
    header_print("info", "Cosine similarity table, L x (DQ / DH)");
    utils::print_matrix(cosine_similarity_table, n, 16);

    error_metrics metrics = get_error_metrics(y_0, y_ref);
    print_error_metrics(metrics);
    return 0;
}
