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

#include "tensor_2d.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "gpt_oss_npu/gpt_oss_gemm.hpp"
#include "metrices.hpp"
#include "gptossParam.hpp"
#include "Mxfp4GptossDequant.hpp"
#include <stdfloat>

#include "error_measure.hpp"


namespace po = boost::program_options;


// Transpose A [N x K] -> B [K x N], both row-major
void transpose_bf16(
    const std::bfloat16_t* A,
    std::bfloat16_t* B,
    int N, int K
) {
    assert(A != nullptr && B != nullptr);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            // A[i, j] = A[i*K + j]
            // B[j, i] = B[j*N + i]
            B[j * N + i] = A[i * K + j];
        }
    }
}


// Matrix multiplication with padding in A
// A: [M x K_padded] row-major, but only first K columns valid
// B: [K x N] row-major
// C: [M x N] row-major
void matmul_bf16_paddedA(
    const std::bfloat16_t* A,
    const std::bfloat16_t* B,
    std::bfloat16_t* C,
    int M, int K, int N, int K_padded
) {
    assert(A != nullptr && B != nullptr && C != nullptr);
    assert(K_padded >= K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f; // accumulate in float for accuracy
            for (int k = 0; k < K; k++) {
                // A is padded, stride is K_padded
                sum += static_cast<float>(A[i * K_padded + k]) *
                       static_cast<float>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<std::bfloat16_t>(sum);
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



    bool pass = true;
    // NPU instance
    npu_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "build/xclbins/mxfp4_dequant.xclbin";
    npu_app_desc dequant_accel_desc;
    npu_app_desc gemm_accel_desc;
    dequant_accel_desc.xclbin_name = xclbin_name;
    dequant_accel_desc.app_name = "dequant_mxfp4_bf16_compiled";
    gemm_accel_desc.xclbin_name = model_path + "/mm.xclbin";
    gemm_accel_desc.app_name = "qmm_mxfp4_bf16_compiled";



    npu_app dequant_app = npu_instance.create_app(dequant_accel_desc);
    npu_app gemm_app = npu_instance.create_app(gemm_accel_desc);

    GPT_OSS_Gemm gemm = GPT_OSS_Gemm();


    constexpr uint32_t INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8 = ((((INTERMEDIATE_PADDED_FOUR_Q4NX_ROW) + ((8*64)-1) ) / (8*64)) *(8* 64));
    constexpr uint32_t D_PADDED_FOUR_Q4NX_ROW_ROW_to_64_mul_8 = ((((D_PADDED_FOUR_Q4NX_ROW) + ((8*64)-1) ) / (8*64)) *(8* 64));
    
    constexpr uint32_t mat_size_gate_up_pad = INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8*D_PADDED_Q4NX_COL;
    constexpr uint32_t mat_size_down_pad = D_PADDED_FOUR_Q4NX_ROW_ROW_to_64_mul_8* INTERMEDIATE_PADDED_Q4NX_COL;

    uint32_t L=  512; // for Testing purpose
    

    // for debug, only generate expert 0 for now
    generate_mxfp4_gpotss_dequant_sequence_single_expert
    <Q4NX_ROW, Q4NX_COL, Q4NX_BLOCK_BYTE_SIZE,
    INTERMEDIATE_PADDED_FOUR_Q4NX_ROW, D_PADDED_Q4NX_COL,
    D_PADDED_FOUR_Q4NX_ROW, INTERMEDIATE_PADDED_Q4NX_COL
    >
    (
        *dequant_app.instr_seq, 0, mat_size_gate_up_pad, mat_size_down_pad
    );

    // load all expert, but only dequant the first 1
    buffer<char> layer_1_mlp_quant_weight = dequant_app.create_bo_buffer<char>(GATE_UP_DOWN_WEIGHT_BYTE , 3);
    buffer<std::bfloat16_t> layer_0_mlp_dequant_weight = dequant_app.create_bo_buffer<std::bfloat16_t>(
        NUM_OF_EXPERT *(2*mat_size_gate_up_pad + mat_size_down_pad), 4);


    weights.load_weights(layer_1_mlp_quant_weight, "model.layers.1.ffn_gate_up_down_exps.weight");
    buffer<std::bfloat16_t> ref_gate_weight;
    ref_tensors.load_weights(ref_gate_weight, "layer_1_gate_weight");

    // first, dequant it  
    auto start_dequant_time = time_utils::now();
    layer_1_mlp_quant_weight.sync_to_device();
    auto run_list = npu_instance.create_runlist(dequant_app);
    auto run = dequant_app.create_run(layer_1_mlp_quant_weight, layer_0_mlp_dequant_weight);
    run_list.add(run);
    run_list.execute();
    layer_0_mlp_dequant_weight.sync_from_device();
    auto end_dequant_time = time_utils::now();
    auto dequant_duration = time_utils::duration_ms(start_dequant_time, end_dequant_time);
    std::cout << "Dequant time: " << dequant_duration.first << " "<< dequant_duration.second << std::endl;




    buffer<std::bfloat16_t> x_padded_input = buffer<std::bfloat16_t>(L * D_PADDED_Q4NX_COL);
    for(int i = 0; i < L; i++){
        for(int k = 0; k < D_PADDED_Q4NX_COL; k++){
            if (k < D){
                x_padded_input[i*D_PADDED_Q4NX_COL + k] = (std::bfloat16_t)( (float)(rand()) / RAND_MAX * 2 - 1); //rand -1 to 1
            }
            else{
                x_padded_input[i*D_PADDED_Q4NX_COL + k] = (std::bfloat16_t)0.0f;
            }
        }
    }

    buffer<std::bfloat16_t> gate_exp_0_ref = buffer<std::bfloat16_t>( GPTOSS_INTERMEDIATE_SIZE*D );
    buffer<std::bfloat16_t> y_ref = buffer<std::bfloat16_t>(L * INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8);
    // Note: Only first GPTOSS_INTERMEDIATE_SIZE elements per row are valid

    {   

        transpose_bf16(
            ref_gate_weight.data(),
            gate_exp_0_ref.data(),
            GPTOSS_INTERMEDIATE_SIZE, D
        );

        // First compute into a temporary buffer with correct size
        buffer<std::bfloat16_t> y_ref_temp = buffer<std::bfloat16_t>(L * GPTOSS_INTERMEDIATE_SIZE);
        matmul_bf16_paddedA(
            x_padded_input.data(),
            gate_exp_0_ref.data(),
            y_ref_temp.data(),
            L, D, GPTOSS_INTERMEDIATE_SIZE, D_PADDED_Q4NX_COL
        );
        
        // Copy to padded reference buffer with padding
        for(int i = 0; i < L; i++) {
            for(int j = 0; j < INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8; j++) {
                if(j < GPTOSS_INTERMEDIATE_SIZE) {
                    y_ref[i * INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8 + j] = y_ref_temp[i * GPTOSS_INTERMEDIATE_SIZE + j];
                } else {
                    y_ref[i * INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8 + j] = static_cast<std::bfloat16_t>(0.0f);
                }
            }
        }

    }


    buffer<std::bfloat16_t> gemm_y = gemm_app.create_bo_buffer<std::bfloat16_t>(L * INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8,3);
    buffer<std::bfloat16_t> gemm_x = gemm_app.create_bo_buffer<std::bfloat16_t>(L * D_PADDED_Q4NX_COL,4);
    buffer<std::bfloat16_t> gemm_w = gemm_app.create_bo_buffer<std::bfloat16_t>( mat_size_gate_up_pad, 5);   
    // memset(gemm_w.data(), 0, sizeof(std::bfloat16_t)* mat_size_gate_up_pad);


    memcpy(gemm_x.data(), x_padded_input.data(), sizeof(std::bfloat16_t)*  L*D_PADDED_Q4NX_COL);

    memcpy(gemm_w.data(), layer_0_mlp_dequant_weight.data(), sizeof(std::bfloat16_t)*mat_size_gate_up_pad);
    gemm.generate_seq(gemm_app.instr_seq, L, D_PADDED_Q4NX_COL, INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8, 0);


    gemm_x.sync_to_device();
    gemm_w.sync_to_device();
    auto mm_run_list = npu_instance.create_runlist(gemm_app);
    auto gemm_run = gemm_app.create_run(gemm_y, gemm_x, gemm_w);
    mm_run_list.add(gemm_run);

    auto mm_start = time_utils::now();
    mm_run_list.execute();
    mm_run_list.wait();
    auto mm_stop = time_utils::now();
    auto mm_duration = time_utils::duration_ms(mm_start, mm_stop);
    std::cout << "GEMM time: " << mm_duration.first << " "<< mm_duration.second << std::endl;
    gemm_y.sync_from_device();



   
    error_measure::print_error_metrics<std::bfloat16_t, std::bfloat16_t>(
        gemm_y.data(), y_ref.data(),
        1,
        L,     INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8,
        L, INTERMEDIATE_PADDED_FOUR_Q4NX_ROW_to_64_mul_8

    );
    
    return 0;
}

