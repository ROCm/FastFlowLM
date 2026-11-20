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
#include "sequnce_gen.hpp"
#include "tensor_2d.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "gpt_oss_npu/gpt_oss_gemm.hpp"
#include "metrices.hpp"

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

    Q4NX weights(model_path);
    SafeTensors ref_tensors(model_path + "/prefill_data.safetensors");

    const int D_in = 3072 - 128;
    const int D_out = 512;

    const int W_VOLUME = D_in * D_out;
    const int WQ_VOLUME = D_in * D_out * 5 / 8;

    bool pass = true;
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = model_path + "/dequant_q4_1.xclbin";
    npu_app_desc dequant_accel_desc;
    npu_app_desc gemm_accel_desc;
    dequant_accel_desc.xclbin_name = xclbin_name;
    dequant_accel_desc.app_name = "dequant_q4k_bf16_compiled";
    gemm_accel_desc.xclbin_name = model_path + "/mm.xclbin";
    gemm_accel_desc.app_name = "qmm_q4k_bf16_compiled";

    header_print("info", "Matrix size " << D_in << "x" << D_out);

    npu_app dequant_app = npu_instance.create_app(dequant_accel_desc);
    npu_app gemm_app = npu_instance.create_app(gemm_accel_desc);

    GPT_OSS_Gemm gemm = GPT_OSS_Gemm();

    generate_dequant_seq(dequant_app.instr_seq, D_in, D_out, 0);


    buffer<bf16> w_0 = gemm_app.create_bo_buffer<bf16>(W_VOLUME, 3);
    buffer<u8> wq_0 = gemm_app.create_bo_buffer<u8>(WQ_VOLUME, 4);
   
    weights.load_weights(wq_0, "model.layers.0.self_attn.k_proj.weight");
    buffer<bf16> x_input;
    ref_tensors.load_weights(x_input, "layer_0_after_input_layernorm");

    header_print("info", "x_input:");
    utils::print_matrix(x_input, 2880);

    int L = x_input.size() / 2880;
    int L_padded = (L + 255) / 256 * 256;
    const int Y_VOLUME = L_padded * D_out;
    const int X_VOLUME = L_padded * D_in;
    buffer<bf16> y_0 = gemm_app.create_bo_buffer<bf16>(Y_VOLUME, 3);
    buffer<bf16> x_0 = gemm_app.create_bo_buffer<bf16>(X_VOLUME, 4);
    buffer<bf16> w_mm = gemm_app.create_bo_buffer<bf16>(W_VOLUME, 5);
    
    gemm.generate_seq(gemm_app.instr_seq, L_padded, D_in, D_out, 0);
    header_print("info", "L: " << L);

    {
        tensor_2d<bf16> x_input_tensor(x_input, 2880);
        tensor_2d<bf16> x_padded_tensor(x_0, 2944);
        for (int i = 0; i < L; i++){
            memcpy(x_padded_tensor[i].data(), x_input_tensor[i].data(), 2880 * sizeof(bf16));
        }
        for (int i = L; i < L_padded; i++){
            memset(x_padded_tensor[i].data(), 0, 2944 * sizeof(bf16));
        }
    }

    buffer<bf16> y_ref;
    ref_tensors.load_weights(y_ref, "layer_0_k_proj_no_bias");

    assert(y_ref.size() == L * D_out);

    wq_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    unsigned long data_size_rd = D_in * D_out * 5 / 8;
    unsigned long data_size_wr = D_in * D_out * 2;

    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        auto run_list = npu_instance.create_runlist(dequant_app);
        for (int i = 0; i < ITERATIONS; i++){
            auto run = dequant_app.create_run(w_0, wq_0);
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
        
        buffer<bf16> w_ref_reordered(w_0.size());

        bf16* p = w_0.data();
        
        for (int r = 0; r < D_out / 64; r++){
            for(int c = 0; c < D_in / 128; c++){
                for (int j = 0; j < 128; j++){
                    for (int i = 0; i < 64; i++){
                        int w_ref_row = r * 64 + i;
                        int w_ref_col = c * 128 + j;
                        w_ref_reordered[w_ref_row * D_in + w_ref_col] = *p++;
                    }
                }
            }
        }

        // reorder
        buffer<bf16> w_ref_reordered_useful = buffer<bf16>(D_out * 2880);
        tensor_2d<bf16> w_ref_reordered_tensor(w_ref_reordered, D_in);
        tensor_2d<bf16> w_ref_reordered_useful_tensor(w_ref_reordered_useful, 2880);
        for (int i = 0; i < D_out; i++){
            memcpy(w_ref_reordered_useful_tensor[i].data(), w_ref_reordered_tensor[i].data(), 2880 * sizeof(bf16));
        }

        buffer<bf16> w_ref(D_out * 2880);
        // w_ref.from_file("w_ref.bin");
        // header_print("info", "w_ref_reordered!");
        // header_print("info", "NPU result:");
        // utils::print_matrix(w_ref_reordered_useful, 2880);

        // print_error_metrics(get_error_metrics(w_ref, w_ref_reordered_useful));

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

        // now run the gemm

        w_mm.copy_from(w_0);
        w_mm.sync_to_device();
        x_0.sync_to_device();
        
        auto mm_run_list = npu_instance.create_runlist(gemm_app);

        for (int i = 0; i < ITERATIONS; i++){
            auto run = gemm_app.create_run(y_0, x_0, w_mm);
            mm_run_list.add(run);
        }

        start = time_utils::now();
        mm_run_list.execute();
        mm_run_list.wait();
        stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
        npu_time.first /= ITERATIONS;
        MSG_BONDLINE(60);
        MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
        MSG_BONDLINE(60);

        y_0.sync_from_device();
        buffer<bf16> y_useful = buffer<bf16>(y_0.data(), L * D_out);
        header_print("info", "y_0:");
        utils::print_matrix(y_useful, D_out);

        buffer<bf16> y_ref(L * D_out);
        ref_tensors.load_weights(y_ref, "layer_0_k_proj_no_bias");
        header_print("info", "y_ref:");
        utils::print_matrix(y_ref, D_out);

        print_error_metrics(get_error_metrics(y_useful, y_ref));
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
