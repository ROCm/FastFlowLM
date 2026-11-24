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
#include <omp.h>

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

inline float hsum256(__m256 v) {
    __m128 low  = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void apply_layernorm_avx(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& weight, buffer<bf16>& bias, float eps) {
    int total = src.size();
    assert(weight.size() == bias.size());

    const int D = weight.size();
    const int L = total / D;
    const int simd_width = 8;

    // Preload weight and bias in fp32
    std::vector<__m256> gamma(D / simd_width), beta(D / simd_width);
    for (int i = 0; i < D; i += simd_width) {
        __m128i w_bf16 = _mm_loadu_si128((__m128i*)&weight[i]);
        __m128i b_bf16 = _mm_loadu_si128((__m128i*)&bias[i]);
        gamma[i / simd_width] = bf16o_fp32(w_bf16);
        beta[i / simd_width] = bf16o_fp32(b_bf16);
    }

    #pragma omp parallel for
    for (int l = 0; l < L; ++l) {
        bf16* x_ptr = src.data() + l * D;
        bf16* y_ptr = dest.data() + l * D;

        // Step 1: compute mean and variance
        __m256 sum = _mm256_setzero_ps();
        __m256 sumsq = _mm256_setzero_ps();
        for (int i = 0; i < D; i += simd_width) {
            __m128i x_bf16 = _mm_loadu_si128((__m128i*)&x_ptr[i]);
            __m256 x_fp32 = bf16o_fp32(x_bf16);
            sum = _mm256_add_ps(sum, x_fp32);
            sumsq = _mm256_fmadd_ps(x_fp32, x_fp32, sumsq);
        }

        float mean = hsum256(sum) / D;
        float var = hsum256(sumsq) / D - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        __m256 mean_v = _mm256_set1_ps(mean);
        __m256 inv_std_v = _mm256_set1_ps(inv_std);

        // Step 2: normalize and apply affine
        for (int i = 0; i < D; i += simd_width) {
            __m128i x_bf16 = _mm_loadu_si128((__m128i*)&x_ptr[i]);
            __m256 x_fp32 = bf16o_fp32(x_bf16);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_fp32, mean_v), inv_std_v);
            __m256 y_fp32 = _mm256_fmadd_ps(norm, gamma[i / simd_width], beta[i / simd_width]);
            __m128i y_bf16 = f32o_bf16(y_fp32);
            _mm_storeu_si128((__m128i*)&y_ptr[i], y_bf16);
        }
    }
}

float get_normalized_l2_error(buffer<bf16>& y, buffer<bf16>& y_ref){
    float sum_error = 0;
    float sum_ref = 0;
    for (int i = 0; i < y.size(); i++){
        sum_error += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
        sum_ref += y_ref[i] * y_ref[i];
    }
    return sqrt(sum_error / sum_ref);
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

int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("../../models"), "Model file");
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Model_type,t", po::value<std::string>()->default_value("GEMMA_1B"), "Model type");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(1500), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(512), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    std::string model_path = vm["model"].as<std::string>();
    const std::string model_type = vm["Model_type"].as<std::string>();
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int D = 1280;

    buffer<bf16> input_cache;
    buffer<bf16> weight_cache;
    buffer<bf16> bias_cache;
    buffer<bf16> output_cache;

    load_data(input_cache, "data/layernorm_input.bin");
    load_data(weight_cache, "data/layernorm_weight.bin");
    load_data(bias_cache, "data/layernorm_bias.bin");
    load_data(output_cache, "data/layernorm_output.bin");

    header_print("info", "Finished loading data");
    header_print("info", "Input shape: " << input_cache.size() / D << "x" << D);
    header_print("info", "Weight shape: " << weight_cache.size() / D << "x" << D);
    header_print("info", "Bias shape: " << bias_cache.size() / D << "x" << D);
    header_print("info", "Output shape: " << output_cache.size() / D << "x" << D);

    int MAX_L = input_cache.size() / D; 
    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_end = L_begin + L_wanted;

    buffer<bf16> layernorm_output(L_end * D);
    layernorm_output.copy_from(output_cache.data() + L_begin * D, L_wanted * D);

    const int In_VOLUME = L_end * D;
    const int W_VOLUME = 1 * D;
    const int B_VOLUME = 1 * D;
    const int O_VOLUME = L_end * D;

    bool pass = true;

    //////////////////////////// NPU instance ////////////////////////////    
    // // NPU instance
    // npu_manager npu_instance(npu_device::device_npu2);

    // std::string xclbin_name = "build/xclbins/layernorm.xclbin";
    // npu_app_desc accel_desc_0;
    // accel_desc_0.xclbin_name = xclbin_name;
    // accel_desc_0.app_name = "layernorm_compiled";

    // header_print("info", "Matrix size " << L_end << "x" << D << "x" << L_end);

    // npu_app app_0 = npu_instance.create_app(accel_desc_0);

    // buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME, 3);
    // buffer<bf16> in_0 = app_0.create_bo_buffer<bf16>(In_VOLUME, 4);
    // buffer<bf16> wb_0 = app_0.create_bo_buffer<bf16>(W_VOLUME + B_VOLUME, 5);

    // // move q_cache
    // header_print("info", "Moving input cache");
    // in_0.copy_from(input_cache.data() + L_begin * D, L_wanted * D);
    // in_0.sync_to_device();
    
    // // move k_cache and v_cache
    // header_print("info", "Moving weight and bias cache");
    // void * w_src_ptr = static_cast<void*>(weight_cache.data());
    // void * b_src_ptr = static_cast<void*>(bias_cache.data());
    // size_t w_chunk_size = 1 * D;
    // void * w_ptr = static_cast<void*>(wb_0.data());
    // void * b_ptr = static_cast<void*>(wb_0.data() + w_chunk_size);

    // size_t data_block_size_in_bytes = D * sizeof(bf16);
    // // weight
    // memcpy(w_ptr, w_src_ptr, data_block_size_in_bytes);
    // w_src_ptr += data_block_size_in_bytes;
    // w_ptr += data_block_size_in_bytes;

    // // bias
    // memcpy(b_ptr, b_src_ptr, data_block_size_in_bytes);
    // b_src_ptr += data_block_size_in_bytes;
    // b_ptr += data_block_size_in_bytes;
   
    // wb_0.sync_to_device();

    // header_print("info", "Memset output cache");
    // o_0.memset(-1.0f);
    // o_0.sync_to_device();

    // time_utils::time_with_unit npu_time = {0.0, "us"};

    // header_print("info", "Running runtime test.");
    // header_print("info", "Running kernel with bare call (direct based).");
    // for (int i = 0; i < NT; i++) {
    //     gen_layernorm_engine_seq(app_0.instr_seq, L_end);
    //     time_utils::time_point start = time_utils::now();
    //     app_0(o_0, in_0, wb_0);
    //     time_utils::time_point stop = time_utils::now();
    //     npu_time.first += time_utils::duration_us(start, stop).first;
    // }
    // o_0.sync_from_device();
    // MSG_BONDLINE(60);
    // MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    // MSG_BONDLINE(60);

    //////////////////////////// CPU instance ////////////////////////////    
    // for cpu reference
    time_utils::time_with_unit npu_time = {0.0, "us"};
    buffer<bf16> o_wanted(L_end * D);
    for (int i = 0; i < 512; i++) {
        time_utils::time_point start = time_utils::now();
        apply_layernorm_avx(o_wanted, input_cache, weight_cache, bias_cache, 1e-5f);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= 512;

    //////////////////////////// Compare Result ////////////////////////////    

    header_print("info", "Aux out");
    utils::print_matrix(layernorm_output, D);
    header_print("info", "O out");
    utils::print_matrix(o_wanted, D);

    error_metrics metrics = get_error_metrics(o_wanted, layernorm_output);
    print_error_metrics(metrics);
    
    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU/CPU time: " << npu_time.first << " us");
    return 0;
}