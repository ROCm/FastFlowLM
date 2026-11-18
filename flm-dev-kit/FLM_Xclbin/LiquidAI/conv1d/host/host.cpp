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
#include "seq_gen.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

namespace po = boost::program_options;
// round up to a multiple
int round_up_to_multiple(int x, int multiple){
    if (multiple == 0) {
        return x; // Cannot divide by zero
    }

    return ((x + multiple - 1) / multiple) * multiple;
}
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
    buffer<bf16> y_float(file_size / sizeof(bf16));
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
    
    const size_t CONV_PAD_SIZE = 2;        
    const size_t L = 2048 + CONV_PAD_SIZE;  // For simulation L is the actual, +2 from cache/padding
    const size_t L_PAD_REQUIREMENT =1;
    const size_t CONV_KERNEL_SIZE = 3;
    const size_t L_padded =  round_up_to_multiple(L, L_PAD_REQUIREMENT);
    const size_t D = 2048;
    const size_t  L_out = (L_padded- CONV_KERNEL_SIZE + 1);



    bool pass = true;
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "build/xclbins/conv.xclbin";
    npu_app_desc accel_desc_0;
    accel_desc_0.xclbin_name = xclbin_name;
    accel_desc_0.app_name = "conv1d_compiled";

    npu_app app_0 = npu_instance.create_app(accel_desc_0);

    header_print("info", "Matrix size " << L << "x" << D);

    buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>( L_out * D, 3);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16> (3 * D, 4);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>((L_padded) * D, 5);

    header_print("info", "load data");
    buffer<bf16> ref_score( (L- CONV_KERNEL_SIZE + 1) * D);
    load_data(ref_score, "../data/conv_output.bin");
    std::cout << "Assing" << std::endl;


    std::cout << "finished loading reference " << std::endl;
    buffer<bf16> w_ref(3 * D);
    buffer<bf16> x_ref((L) * D);
    load_data(w_ref, "../data/conv_weight.bin");
    load_data(x_ref, "../data/conv_input.bin");

    header_print("info", "load w");
    memcpy(w_0.data(), w_ref.data(), sizeof(bf16)* 3*D);
    memcpy(  x_0.data(),x_ref.data(), sizeof(bf16)* L* D );
    

    assert(y_0.size() == ref_score.size()); // when L_PAD_REQUIREMENT== 1


    y_0.memset(-1.0f);
    w_0.sync_to_device();
    x_0.sync_to_device();
    y_0.sync_to_device();

    // auto runlist = npu_instance.create_runlist(app_0);
    uint32_t iter = 100;
    gen_conv1d_engine_seq(app_0.instr_seq, L_padded, D, L_out, L_PAD_REQUIREMENT, 0);

    auto runlist = npu_instance.create_runlist(app_0);
    time_utils::time_with_unit npu_time = {0.0, "us"};
    for (int i = 0; i < iter; i++) {
        auto run = app_0.create_run(y_0, w_0, x_0);
        runlist.add(run);
    }
    time_utils::time_point start = time_utils::now();  
    runlist.execute();
    runlist.wait();
    time_utils::time_point stop = time_utils::now();
    //

    npu_time.first += time_utils::duration_us(start, stop).first;
    npu_time.first /= iter;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    y_0.sync_from_device();

    header_print("info", "y");
    utils::print_matrix(y_0, D);
    header_print("info", "ref");
    utils::print_matrix(ref_score, D);

    float rmse = get_rmse(y_0, ref_score);
    float relativeL1 = get_relativeL1(y_0, ref_score);
    float relativeL2 = get_relativeL2(y_0, ref_score);
    float cosine_similarity = get_cosine_similarity(y_0, ref_score);
    header_print("info", "Relative L1: " << relativeL1);
    header_print("info", "Relative L2: " << relativeL2);
    header_print("info", "Cosine similarity: " << cosine_similarity);
    header_print("info", "RMSE: " << rmse);

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
    int min_size = std::min(y.size(), y_ref.size());      
    float rmse = 0;
    float ref_sum = 0;
    for (int i = 0; i < min_size; i++){
        ref_sum += y_ref[i] * y_ref[i];
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
}

float get_relativeL1(buffer<bf16>& y, buffer<bf16>& y_ref){
    int min_size = std::min(y.size(), y_ref.size());    
    float l1 = 0;
    float ref_sum = 0;
    for (int i = 0; i < min_size; i++){
        ref_sum += abs(y_ref[i]);
        l1 += abs(y[i] - y_ref[i]);
    }
    return l1 / ref_sum;
}

float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref){
    int min_size = std::min(y.size(), y_ref.size());
    float rmse = 0;
    for (int i = 0; i < min_size; i++){
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size());
}

float get_cosine_similarity(buffer<bf16>& y, buffer<bf16>& y_ref){

    int min_size  = std::min(y.size(), y_ref.size());
    float dot_product = 0;
    float norm_y = 0;
    float norm_y_ref = 0;
    for (int i = 0; i <min_size; i++){
        dot_product += y[i] * y_ref[i];
        norm_y += y[i] * y[i];
        norm_y_ref += y_ref[i] * y_ref[i];
    }
    return dot_product / (sqrt(norm_y) * sqrt(norm_y_ref));
}