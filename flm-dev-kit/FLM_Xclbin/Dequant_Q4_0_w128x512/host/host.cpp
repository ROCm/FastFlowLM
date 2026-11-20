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
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "modules/embedding.hpp"
#include "lm_config.hpp"
#include "modules/dequant.hpp"

#include "tensor_utils/q4_npu_eXpress.hpp"

namespace po = boost::program_options;

template<typename T>
void write_bin(buffer<T>& data, std::string filename){
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
    file.close();
}

void linear(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& w, int M, int K, int N);
float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref);


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

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("Model,m", po::value<std::string>()->default_value("../../../models"), "Model path");
    desc.add_options()("ModelType,t", po::value<std::string>()->default_value("GPT"), "Model type");
    desc.add_options()("Iterations,i", po::value<int>()->default_value(128), "Iterations");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");

    arg_utils::parse_options(argc, const_cast<char **>(argv), desc, vm);

    // User logic
    const std::string MODEL_PATH = vm["Model"].as<std::string>();
    const std::string MODEL_TYPE = vm["ModelType"].as<std::string>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const int ITERATIONS = vm["Iterations"].as<int>();
    std::string model_path = MODEL_PATH + "/";

    if (MODEL_TYPE == "GPT"){
        model_path += "GPT-OSS-20B-NPU2";
    }
    else{
        std::cerr << "Invalid model type: " << MODEL_TYPE << std::endl;
        return 1;
    }

    LM_Config config;
    config.from_pretrained(model_path);

    Dequant dequant_seq(config);

    Q4NX weights(model_path);

    const int D_in = 3072;
    const int D_out = 512;

    const int W_VOLUME = D_in * D_out;
    const int WQ_VOLUME = D_in * D_out * 5 / 8;

    bool pass = true;
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "build/xclbins/dequant.xclbin";
    npu_app_desc accel_desc_0;
    accel_desc_0.xclbin_name = xclbin_name;
    accel_desc_0.app_name = "dequant_q4k_bf16_compiled";

    header_print("info", "Matrix size " << D_in << "x" << D_out);

    npu_app app_0 = npu_instance.create_app(accel_desc_0);

    dequant_seq.generate_seq(app_0.instr_seq, D_in, D_out, 0);


    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME, 3);
    buffer<bf16> w_ref = buffer<bf16>(W_VOLUME);
    buffer<u8> wq_0 = app_0.create_bo_buffer<u8>(WQ_VOLUME, 4);
    buffer<bf16> w_ref_T(W_VOLUME);
    buffer<bf16> x_ref(2048 * D_in);
    for (int i = 0; i < 2048; i++){
        for (int j = 0; j < D_in; j++){
            x_ref[i * D_in + j] = utils::getRand(-1.0, 1.0);
        }
    }
   
    weights.load_weights(wq_0, "model.layers.0.self_attn.v_proj.weight");


    // buffer<bf16> w_ref_reordered(w_ref.size());
    // bf16* p = w_ref.data();
    
    // // transpose every 8x8
    // bf16 p_temp[8][8];
    // for (int i = 0; i < D_out / 8; i++){
    //     for (int j = 0; j < D_in / 8; j++){
    //         for (int k = 0; k < 8; k++){
    //             for (int l = 0; l < 8; l++){
    //                 p_temp[k][l] = p[(i * 8 + k) * D_in + (j * 8 + l)];
    //             }
    //         }
            
    //         // write transpose_ptemp to original p
    //         for (int k = 0; k < 8; k++){
    //             for (int l = 0; l < 8; l++){
    //                 p[(i * 8 + k) * D_in + (j * 8 + l)] = p_temp[l][k];
    //             }
    //         }
    //     }
    // }

    
    // // reorder
    // p = w_ref_reordered.data();
    // for (int r = 0; r < D_out / 64; r++){
    //     for(int c = 0; c < D_in / 512; c++){
    //         for (int i = 0; i < 64; i++){
    //             for (int j = 0; j < 512; j++){
    //                 *p++ = w_ref[(r * 64 + i) * D_in + (c * 512 + j)];
    //             }
    //         }
    //     }
    // }
    // header_print("info", "w_ref_reordered!");

    wq_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    unsigned long data_size_rd = D_in * D_out * 5 / 8;
    unsigned long data_size_wr = D_in * D_out * 2;
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        auto run_list = npu_instance.create_runlist(app_0);
        for (int i = 0; i < ITERATIONS; i++){
            auto run = app_0.create_run(w_0, wq_0);
            run_list.add(run);
        }

        time_utils::time_point start = time_utils::now();
        run_list.execute();
        run_list.wait();
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
        npu_time.first /= ITERATIONS;
        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
        MSG_BONDLINE(60);

        w_0.sync_from_device(); 
        // header_print("info", "w_ref:");
        // utils::print_matrix(w_ref_reordered, D_in, 16, 16);
        header_print("info", "NPU result:");
        utils::print_matrix(w_0, D_in, 256, 32);

        // if (utils::compare_vectors(w_0, w_ref_reordered, 8, 1, 99) > 0){
        //     pass = false;
        // }

        // float rmse = get_rmse(w_0, w_ref_reordered);
        // write_bin(w_0, "w_reordered.bin");
        // header_print("info", "RMSE: " << rmse);

        header_print("info", "Finished running kernel");
        header_print("info", "Running kernel with bare call (run based).");
        utils::print_npu_profile(npu_time, 2.0 * float(D_in) * float(D_out));
        header_print("info", "U_mem^rd: " + std::to_string(data_size_rd / npu_time.first * 1e6 / 1e9) + " GB/s");
        header_print("info", "U_mem^wr: " + std::to_string(data_size_wr / npu_time.first * 1e6 / 1e9) + " GB/s");
    }

    header_print("info", "trace");
   
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

float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref){
    float rmse = 0;
    float ref_sum = 0;
    for (int i = 0; i < y.size(); i++){
        ref_sum += y_ref[i] * y_ref[i];
        rmse += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
    }
    return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
}
