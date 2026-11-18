// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "qwen/qwen_npu.hpp"
#include "lm_config.hpp"
#include "tokenizer/tokenizer.hpp"
#include "modules/sampler.hpp"
#include "modules/lm_head.hpp"
#include "modules/embedding.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "whisper_lm_head.hpp"

void layer_norm(buffer<bf16> y, buffer<bf16> weight, buffer<bf16> bias, buffer<bf16> x){

    int n = x.size();
    const int simd_width = 8;

    __m256 sum_x_vec = _mm256_setzero_ps();
    __m256 sum_xx_vec = _mm256_setzero_ps();
    
    // Step 1: Compute mean and variance using AVX
    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);

        // Convert BF16 to FP32
        __m256 fp32_vals = bf16o_fp32(bf16_vals);
        sum_x_vec = _mm256_add_ps(sum_x_vec, fp32_vals);  // sum_x += x[i]
        sum_xx_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i]
    }

    // Handle remaining elements
    for (; i < n; i++) {
        f32 val = x[i].as_float();
        sum_x_vec = _mm256_add_ps(sum_x_vec, _mm256_set1_ps(val));
        sum_xx_vec = _mm256_fmadd_ps(_mm256_set1_ps(val), _mm256_set1_ps(val), sum_xx_vec);
    }

    // Horizontal sum of the AVX registers
    f32 sum_x_arr[simd_width], sum_xx_arr[simd_width];
    _mm256_storeu_ps(sum_x_arr, sum_x_vec);
    _mm256_storeu_ps(sum_xx_arr, sum_xx_vec);
    
    f32 sum_x = sum_x_arr[0] + sum_x_arr[1] + sum_x_arr[2] + sum_x_arr[3] +
                     sum_x_arr[4] + sum_x_arr[5] + sum_x_arr[6] + sum_x_arr[7];
    f32 sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3] +
                      sum_xx_arr[4] + sum_xx_arr[5] + sum_xx_arr[6] + sum_xx_arr[7];

    f32 mean_x = sum_x / (f32)n;
    f32 var_x = (sum_xx / (f32)n) - (mean_x * mean_x);
    f32 std_x = sqrtf(var_x + 1e-5f);

    // Step 2: Compute element-wise operation y[i] = ((x[i] - mean_x) / std_x) * weight[i] + bias[i]
    __m256 mean_x_vec = _mm256_set1_ps(mean_x);
    __m256 std_x_vec = _mm256_set1_ps(std_x);
    
    for (i = 0; i <= n - simd_width; i += simd_width) {
        __m128i x_bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m128i weight_bf16_vals = _mm_loadu_si128((__m128i*)&weight[i]);
        __m128i bias_bf16_vals = _mm_loadu_si128((__m128i*)&bias[i]);
        
        // Convert BF16 to FP32
        __m256 x_fp32_vals = bf16o_fp32(x_bf16_vals);
        __m256 weight_fp32_vals = bf16o_fp32(weight_bf16_vals);
        __m256 bias_fp32_vals = bf16o_fp32(bias_bf16_vals);
        
        // y[i] = ((x[i] - mean_x) / std_x) * weight[i] + bias[i]
        __m256 normalized = _mm256_div_ps(_mm256_sub_ps(x_fp32_vals, mean_x_vec), std_x_vec);
        __m256 y_vec = _mm256_fmadd_ps(normalized, weight_fp32_vals, bias_fp32_vals);
        
        __m128i result_bf16 = f32o_bf16(y_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        f32 x_val = x[i].as_float();
        f32 weight_val = weight[i].as_float();
        f32 bias_val = bias[i].as_float();
        f32 normalized = (x_val - mean_x) / std_x;
        f32 y_val = normalized * weight_val + bias_val;
        y[i] = bf16(y_val);
    }
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("../../../models"), "Model file");
    desc.add_options()("testing_idx,i", arg_utils::po::value<int>()->default_value(0), "Testing index");
    desc.add_options()("model_type,t", arg_utils::po::value<std::string>()->default_value("WHISPER"), "Model type");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_path = vm["model"].as<std::string>();
    int testing_idx = vm["testing_idx"].as<int>();
    if (testing_idx < 0 || testing_idx >= 51872){
        std::cout << "Testing index out of range: " << testing_idx << std::endl;
        return 1;
    }
    std::string model_type = vm["model_type"].as<std::string>();
    std::string xclbin_path = "build/" + model_type + "/xclbins/qmm.xclbin";

    if (model_type == "WHISPER"){
        model_path = model_path + "/Whisper-V3-Turbo-NPU2";
    }
    else{
        std::cout << "Model type not supported: " << model_type << std::endl;
        return 1;
    }

    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Model type: " << model_type << std::endl;
    std::cout << "XCLBIN: " << xclbin_path << std::endl;  
    Q4NX weights(model_path);
    Whisper_Config config;
    config.from_pretrained(model_path);
    config.lm_head_xclbin_name = xclbin_path;
    std::cout << config._str() << std::endl;
    npu_manager npu(npu_device::device_npu2, 0);
    std::unique_ptr<WhisperLMHead> lm_head;
    std::unique_ptr<Embedding> embedding;
    sampler_config sampler_config;
    sampler_config.temperature = 0.7;
    sampler_config.top_k = 1;
    sampler_config.top_p = 0.9;
    sampler_config.rep_penalty = 0.0;
    sampler_config.freq_penalty = 0.0;
    std::unique_ptr<Sampler> sampler = std::make_unique<Sampler>(config.vocab_size, sampler_config);
    lm_head = std::make_unique<WhisperLMHead>(config, &npu);
    embedding = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    embedding->init_weights(&weights, "model.decoder.embed_tokens");
    lm_head->load_weights(weights);

    if (!utils::check_file_exists(model_path + "/model.q4nx")) {
        std::cout << "Model file does not exist: " << model_path << std::endl;
        return 1;
    }

  


    std::cout << "Model loaded!" << std::endl;
    buffer<bf16> x_lm_head_exposed = lm_head->x_exposed;
    embedding->forward(testing_idx, x_lm_head_exposed);
    utils::print_matrix(x_lm_head_exposed, 320);
    lm_head->execute();
    buffer<bf16> y = lm_head->wait();
    std::cout << "y size: " << y.size() << std::endl;
    utils::print_matrix(y, 16);
    int new_token = sampler->sample(y);
    std::cout << "New token: " << new_token << std::endl;
    buffer<bf16> norm_weight;
    buffer<bf16> norm_bias;
    weights.load_weights(norm_weight, "model.decoder.layer_norm.weight");
    weights.load_weights(norm_bias, "model.decoder.layer_norm.bias");

    // std::ofstream output_file("output.txt");

    // for (int i = 0; i < 1000; i++){
    //     embedding->forward(i, x_lm_head_exposed);
    //     layer_norm(x_lm_head_exposed, norm_weight, norm_bias, x_lm_head_exposed);
    //     lm_head->execute();
    //     buffer<bf16> y = lm_head->wait();
    //     int new_token = sampler->sample(y);
    //     if (i != new_token){
    //         output_file << i << " " << new_token << std::endl;
    //     }
    // }
    // output_file.close();

    buffer<float> y_float(y.size());
    for (int i = 0; i < y_float.size(); i++){
        y_float[i] = y[i].as_float();
    }
    float max = y_float[0];
    int max_index = 0;
    for (int i = 1; i < y_float.size(); i++){
        if (y_float[i] > max){
            max = y_float[i];
            max_index = i;
        }
    }
    std::cout << "Max: " << max << std::endl;
    std::cout << "Max index: " << max_index << std::endl;
    return 0;
}