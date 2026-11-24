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

namespace po = boost::program_options;

void linear(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w, int M, int K, int N);
float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref);
float get_relativeL1(buffer<bf16>& y, buffer<bf16>& y_ref);
float get_relativeL2(buffer<bf16>& y, buffer<bf16>& y_ref);
float get_cosine_similarity(buffer<bf16>& y, buffer<bf16>& y_ref);

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
    buffer<float> y_float(file_size / sizeof(float));
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
    desc.add_options()("Row,r", po::value<int>()->default_value(0), "Row of B to use as input x");
    desc.add_options()("Trace,t", po::value<int>()->default_value(131072), "Trace size");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("EnableTrace,e", po::value<bool>()->default_value(false), "Enable trace");

    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int ROW = vm["Row"].as<int>();
    const int TRACE_SIZE = vm["Trace"].as<int>();
    const bool enable_trace = vm["EnableTrace"].as<bool>();
    const bool VERIFY= vm["Verify"].as<bool>();
    const size_t M = 1280;  
    const size_t K = 1280;
    const size_t N = 1;

    const size_t m = 32;
    const size_t k = 128;
    const size_t W_BLOCK_SIZE = M * K * 5 / 8;
    const size_t n = 1;
    const int Y_VOLUME = M * N;
    const int W_VOLUME = W_BLOCK_SIZE;
    const int X_VOLUME = N * K;

    bool pass = true;
    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/mv.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mv_dequant_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

    npu_app app_0 = mv_dequant_app_mgr->create_app();

    buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME, 3);
    buffer<u8> w_0 = app_0.create_bo_buffer<u8>(W_VOLUME, 4);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME, 5);
    std::string model_path = "../../../models/whisper-large-v3-turbo-NPU2";

    header_print("info", "Loading weights...");
    Q4NX model_weights(model_path);

    header_print("info", "Loading x...");
    model_weights.load_weights(w_0, "model.decoder.layers.0.self_attn.q_proj.weight");

    buffer<bf16> x_0_f32(X_VOLUME);
    load_data(x_0_f32, "./x.bin");
    utils::print_matrix(x_0_f32, 1280);
    x_0.copy_from(x_0_f32.data(), X_VOLUME);

    // calculate the reference
    buffer<bf16> y_ref_0_f32(Y_VOLUME);
    load_data(y_ref_0_f32, "./q_proj.bin");
    
    y_0.memset((bf16)(-1.0f));
    w_0.sync_to_device();
    x_0.sync_to_device();
    y_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    unsigned long data_size = M * K * N / n / 4 + K * N * M / m / 4;
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        int Iterations = 200;
        gen_mv_dequant_engine_seq(app_0.seq(), K, M, 0);
        for (int i = 0; i < Iterations; i++) {
            time_utils::time_point start = time_utils::now();
            app_0(y_0, w_0, x_0);
            time_utils::time_point stop = time_utils::now();
            npu_time.first += time_utils::duration_us(start, stop).first;
        }
        npu_time.first /= Iterations;
        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
        MSG_BONDLINE(60);

        y_0.sync_from_device(); 
        header_print("info", "f32 reference:");
        utils::print_matrix(y_ref_0_f32, 128);
        header_print("info", "NPU result:");
        utils::print_matrix(y_0, 128);

        if (utils::compare_vectors(y_0, y_ref_0_f32, 8, 1, 99) > 0){
            pass = false;
        }

        float rmse = get_rmse(y_0, y_ref_0_f32);
        float relativeL1 = get_relativeL1(y_0, y_ref_0_f32);
        float relativeL2 = get_relativeL2(y_0, y_ref_0_f32);
        float cosine_similarity = get_cosine_similarity(y_0, y_ref_0_f32);

        header_print("info", "Relative L1: " << relativeL1);
        header_print("info", "Relative L2: " << relativeL2);
        header_print("info", "Cosine similarity: " << cosine_similarity);
        header_print("info", "RMSE: " << rmse);
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

float get_relativeL2(buffer<bf16>& y, buffer<bf16>& y_ref){
    float rmse = 0;
    float ref_sum = 0;
    for (int i = 0; i < y.size(); i++){
        ref_sum += y_ref[i] * y_ref[i];
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
}

float get_relativeL1(buffer<bf16>& y, buffer<bf16>& y_ref){
    float l1 = 0;
    float ref_sum = 0;
    for (int i = 0; i < y.size(); i++){
        ref_sum += abs(y_ref[i]);
        l1 += abs(y[i] - y_ref[i]);
    }
    return l1 / ref_sum;
}

float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref){
    float rmse = 0;
    for (int i = 0; i < y.size(); i++){
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size());
}

float get_cosine_similarity(buffer<bf16>& y, buffer<bf16>& y_ref){
    float dot_product = 0;
    float norm_y = 0;
    float norm_y_ref = 0;
    for (int i = 0; i < y.size(); i++){
        dot_product += y[i] * y_ref[i];
        norm_y += y[i] * y[i];
        norm_y_ref += y_ref[i] * y_ref[i];
    }
    return dot_product / (sqrt(norm_y) * sqrt(norm_y_ref));
}

// #include <bits/stdc++.h>
// #include <boost/program_options.hpp>
// #include <chrono>
// #include <cstdint>
// #include <cstdlib>
// #include <ctime>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <sstream>
// #include <stdfloat>
// #include "utils/debug_utils.hpp"

// #include "typedef.hpp"
// #include "npu_utils/npu_utils.hpp"
// #include "utils/utils.hpp"
// #include "utils/vm_args.hpp"
// #include "llama/llama_npu_sequence.hpp"
// #include "lm_config.hpp"
// #include <math.h>
// #include "metrices.hpp"
// #include "seq_gen.hpp"
// #include "tensor_utils/q4_npu_eXpress.hpp"

// namespace po = boost::program_options;

// void linear(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w, int M, int K, int N);
// float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref);
// float get_relativeL1(buffer<bf16>& y, buffer<bf16>& y_ref);
// float get_relativeL2(buffer<bf16>& y, buffer<bf16>& y_ref);
// float get_cosine_similarity(buffer<bf16>& y, buffer<bf16>& y_ref);

// int main(int argc, char **argv) {
//     // Fix the seed to ensure reproducibility in CI.
//     srand(0);
//     // Program arguments parsing
//     po::options_description desc("Allowed options");
//     po::variables_map vm;
//     arg_utils::add_default_options(desc);

//     // Add custom options
//     desc.add_options()("Row,r", po::value<int>()->default_value(0), "Row of B to use as input x");
//     desc.add_options()("Trace,t", po::value<int>()->default_value(131072), "Trace size");
//     desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
//     desc.add_options()("EnableTrace,e", po::value<bool>()->default_value(false), "Enable trace");

//     arg_utils::parse_options(argc, argv, desc, vm);

//     // User logic
//     const int ROW = vm["Row"].as<int>();
//     const int TRACE_SIZE = vm["Trace"].as<int>();
//     const bool enable_trace = vm["EnableTrace"].as<bool>();
//     const bool VERIFY= vm["Verify"].as<bool>();
//     const size_t M = 1280;  
//     const size_t K = 1280;
//     const size_t N = 1;

//     const size_t m = 16;
//     const size_t k = 64;
//     const size_t n = 1;
//     const int Y_VOLUME = M * N;
//     const int W_VOLUME = M * K;
//     const int X_VOLUME = N * K;

//     bool pass = true;
//     // NPU instance
//     npu_manager npu_instance(npu_device::device_npu2);

//     std::string xclbin_name = "build/xclbins/mv.xclbin";
//     npu_app_desc accel_desc_0;
//     accel_desc_0.xclbin_name = xclbin_name;
//     accel_desc_0.app_name = "mv_compiled";

//     header_print("info", "Matrix size " << M << "x" << K << "x" << N);

//     npu_app app_0 = npu_instance.create_app(accel_desc_0);

//     buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME, 3);
//     buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME, 4);
//     buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME, 5);

//     for (int i = 0; i < W_VOLUME; i++){
//         w_0[i] = utils::getRand(-4, 4) * 0.01f;
//         // w_0[i] = 1.0f;
//     }

//     for (int i = 0; i < X_VOLUME; i++){
//         x_0[i] = utils::getRand(-4, 4);
//         // x_0[i] = 0.01f;
//     }

//     buffer<bf16> y_ref_0_f32(Y_VOLUME);
//     linear(y_ref_0_f32, w_0, x_0, M, K, N);

//     y_0.memset(-1.0f);
//     w_0.sync_to_device();
//     x_0.sync_to_device();
//     y_0.sync_to_device();

//     time_utils::time_with_unit npu_time = {0.0, "us"};
//     unsigned long data_size = M * K * N / n / 8 + K * N * M / m / 4;
//     if (VERIFY){
//         header_print("info", "Running runtime test.");
//         header_print("info", "Running kernel with bare call (direct based).");
//         int Iterations = 1;
//         auto runlist = npu_instance.create_runlist(app_0);
//         gen_mv_engine_seq(app_0.instr_seq, M, K, N, m, k, n, 0);
//         for (int i = 0; i < Iterations; i++) {
//             auto run = app_0.create_run(y_0, w_0, x_0);
//             runlist.add(run);
//         }
//         time_utils::time_point start = time_utils::now();
//         runlist.execute();
//         runlist.wait();
//         time_utils::time_point stop = time_utils::now();
//         npu_time.first += time_utils::duration_us(start, stop).first;
//         npu_time.first /= Iterations;
//         MSG_BONDLINE(60);
//         MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
//         MSG_BONDLINE(60);

//         y_0.sync_from_device(); 
//         header_print("info", "f32 reference:");
//         utils::print_matrix(y_ref_0_f32, 64);
//         header_print("info", "NPU result:");
//         utils::print_matrix(y_0, 64);

//         if (utils::compare_vectors(y_0, y_ref_0_f32, 8, 1, 99) > 0){
//             pass = false;
//         }

//         float rmse = get_rmse(y_0, y_ref_0_f32);
//         float relativeL1 = get_relativeL1(y_0, y_ref_0_f32);
//         float relativeL2 = get_relativeL2(y_0, y_ref_0_f32);
//         float cosine_similarity = get_cosine_similarity(y_0, y_ref_0_f32);

//         header_print("info", "Relative L1: " << relativeL1);
//         header_print("info", "Relative L2: " << relativeL2);
//         header_print("info", "Cosine similarity: " << cosine_similarity);
//         header_print("info", "RMSE: " << rmse);
//         header_print("info", "Finished running kernel");
//         header_print("info", "Running kernel with bare call (run based).");
//         utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
//         header_print("info", "Data size: " << std::to_string(data_size));
//         header_print("info", "Bandwidth: " + std::to_string(data_size * 2/ npu_time.first * 1e6 / 1e9) + " GB/s");
//     }

//     if (pass){
//         header_print("info", "PASSED ");
//     } else {
//         header_print("info", "FAILED!");
//     }
//     return 0;
// }

// void linear(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w, int M, int K, int N){
//     assert((M * K) == x.size());
//     assert((K * N) == w.size());
//     assert((M * N) == y.size());
//     for (int row = 0; row < M; row++){
//         for (int col = 0; col < N; col++){
//             float sum = 0;
//             for (int k = 0; k < K; k++){
//                 sum = sum + (x[row * K + k] * w[k * N + col]);
//             }
//             y[row * N + col] = (bf16)sum;
//         }
//     }
// }

// float get_relativeL2(buffer<bf16>& y, buffer<bf16>& y_ref){
//     float rmse = 0;
//     float ref_sum = 0;
//     for (int i = 0; i < y.size(); i++){
//         ref_sum += y_ref[i] * y_ref[i];
//         rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
//     }
//     return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
// }

// float get_relativeL1(buffer<bf16>& y, buffer<bf16>& y_ref){
//     float l1 = 0;
//     float ref_sum = 0;
//     for (int i = 0; i < y.size(); i++){
//         ref_sum += abs(y_ref[i]);
//         l1 += abs(y[i] - y_ref[i]);
//     }
//     return l1 / ref_sum;
// }

// float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref){
//     float rmse = 0;
//     for (int i = 0; i < y.size(); i++){
//         rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
//     }
//     return sqrt(rmse / y.size());
// }

// float get_cosine_similarity(buffer<bf16>& y, buffer<bf16>& y_ref){
//     float dot_product = 0;
//     float norm_y = 0;
//     float norm_y_ref = 0;
//     for (int i = 0; i < y.size(); i++){
//         dot_product += y[i] * y_ref[i];
//         norm_y += y[i] * y[i];
//         norm_y_ref += y_ref[i] * y_ref[i];
//     }
//     return dot_product / (sqrt(norm_y) * sqrt(norm_y_ref));
// }