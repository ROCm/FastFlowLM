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
#include "gemma_text/gemma_text_gemm.hpp"
#include "modules/embedding.hpp"
#include "lm_config.hpp"
#include "gemma_text/gemma_text_dequant.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "metrices.hpp"

namespace po = boost::program_options;

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
    desc.add_options()("L,l", po::value<int>()->default_value(2048), "Sequence length");
    desc.add_options()("Model,m", po::value<std::string>()->default_value("../../../models"), "Model path");
    desc.add_options()("ModelType,t", po::value<std::string>()->default_value("Gemma3_1B"), "Model type");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");

    arg_utils::parse_options(argc, const_cast<char **>(argv), desc, vm);

    // User logic
    const int L = vm["L"].as<int>();
    const std::string MODEL_PATH = vm["Model"].as<std::string>();
    const std::string MODEL_TYPE = vm["ModelType"].as<std::string>();
    const bool VERIFY = vm["Verify"].as<bool>();
    if (L > 8192){
        std::cerr << "L is too large: " << L << std::endl;
        return 1;
    }

    std::string model_path = MODEL_PATH + "/";

    if (MODEL_TYPE == "Gemma3_1B"){
        model_path += "Gemma3-1B-NPU2";
    } 
    else{
        std::cerr << "Invalid model type: " << MODEL_TYPE << std::endl;
        return 1;
    }

    LM_Config config;
    config.from_pretrained(model_path);

    GemmaTextGemm gemm(config);
    GemmaTextDequant dequant(config);

    Q4NX weights(model_path);
    Embedding embedding(config.vocab_size, config.hidden_size);
    embedding.init_weights(&weights, "model.embed_tokens");
    const int D_in = config.hidden_size;
    const int DK = config.head_dim * config.num_key_value_heads;
    const int D_out = config.head_dim * config.num_key_value_heads * 2;

    const int X_VOLUME = L * D_in;
    const int W_VOLUME = D_in * D_out;
    const int Y_VOLUME = L * D_out;

    bool pass = true;
    // NPU instance
    npu_xclbin_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "/path/to/models/Gemma3-1B-NPU2/mm.xclbin";
    npu_app_desc accel_desc_0;
    accel_desc_0.xclbin_name = xclbin_name;
    accel_desc_0.app_name = "qmm_q4k_bf16_compiled";

    npu_app_desc accel_desc_1;
    accel_desc_1.xclbin_name = "/path/to/models/Gemma3-1B-NPU2/dequant.xclbin";
    accel_desc_1.app_name = "dequanter";
    header_print("info", "Matrix size " << L << "x" << D_in << "x" << D_out);

    npu_app app_0 = npu_instance.create_app(accel_desc_0);
    npu_app dequant_app = npu_instance.create_app(accel_desc_1);

    // app_0.instr_seq->from_file("build/insts/mm.txt");
    gemm.generate_seq(app_0.instr_seq, L, D_in, D_out, 0);
    dequant.generate_seq(dequant_app.instr_seq, D_in, D_out, 0);

    buffer<bf16> y_0 = app_0.create_bo_buffer<bf16>(Y_VOLUME, 3);
    buffer<bf16> x_0 = app_0.create_bo_buffer<bf16>(X_VOLUME, 4);
    buffer<bf16> w_0 = app_0.create_bo_buffer<bf16>(W_VOLUME, 5);
    buffer<bf16> w_0_dequant = dequant_app.create_bo_buffer<bf16>(W_VOLUME, 3);

    // buffer<bf16> w_dequant = dequant_app.create_bo_buffer<bf16>(W_VOLUME, 3);
    buffer<u8>   qw = dequant_app.create_bo_buffer<u8>(D_in * D_out * 5 / 8, 4);
   
    time_utils::time_with_unit npu_time = {0.0, "us"};

    buffer<bf16> x_norm(D_in);
    load_data(x_norm, "input_layernorm_output.bin");
    x_0.copy_from(x_norm);

    header_print("info", "x_0:");
    utils::print_matrix(x_0, D_in);

    // weights.load_weights(qw, "model.layers.0.mlp.up_proj.weight");
    buffer<u8> qw_k(D_in * DK * 5 / 8);
    buffer<u8> qw_v(D_in * DK * 5 / 8);
    weights.load_weights(qw_k, "model.layers.0.self_attn.k_proj.weight");
    weights.load_weights(qw_v, "model.layers.0.self_attn.v_proj.weight");
    memcpy(qw.data(), qw_k.data(), D_in * DK * 5 / 8);
    memcpy(qw.data() + D_in * DK * 5 / 8, qw_v.data(), D_in * DK * 5 / 8);

    buffer<bf16> y_ref((size_t)L * D_out);

    qw.sync_to_device();

    time_utils::time_point start = time_utils::now();

    dequant_app(w_0_dequant, qw);

    time_utils::time_point stop = time_utils::now();
    npu_time.first += time_utils::duration_us(start, stop).first;
    header_print("info", "Dequantization time: " << npu_time.first << " us");
    npu_time.first = 0;

    w_0_dequant.sync_from_device();
    header_print("info", "w_0_dequant:");
    utils::print_matrix(w_0_dequant, D_in);
    // buffer<bf16> w_ref(D_out * D_in);
    // load_data(w_ref, "v_weight.bin");
    // buffer<bf16> w_ref_trans(D_in * D_out);
    // bf16* p = w_ref_trans.data();
    // for (int rc = 0; rc < D_out / 32; rc++){
    //     for (int cc = 0; cc < D_in / 128; cc++){
    //         int r_offset = rc * 32;
    //         int c_offset = cc * 128;
    //         for (int r = 0; r < 32; r++){
    //             for (int c = 0; c < 128; c++){
    //                 int r_idx = r_offset + r;
    //                 int c_idx = c_offset + c;
    //                 int idx = r_idx * D_in + c_idx;
    //                 *p++ = w_ref[idx];
    //             }
    //         }
    //     }
    // }

    // header_print("info", "w_ref_trans:");
    // utils::print_matrix(w_ref_trans, D_in);
    // error_metrics metrics = get_error_metrics(w_0_dequant, w_ref_trans);
    // print_error_metrics(metrics);
    return 0;
    w_0.copy_from(w_0_dequant);

    w_0.sync_to_device();
    x_0.sync_to_device();

    unsigned long data_size = (size_t(L) * size_t(D_in) * size_t(D_out) / 4 / 64) + (size_t(D_in) * size_t(D_out) * size_t(L)/ 8 / 64);
    if (VERIFY){
        header_print("info", "Running runtime test.");
        header_print("info", "Running kernel with bare call (direct based).");
        int Iterations = 1;
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
        header_print("info", "y_0:");
        utils::print_matrix(y_0, D_out);

        buffer<bf16> y_ref(L * D_out);
        load_data(y_ref, "v_cache.bin");
        header_print("info", "y_ref:");
        utils::print_matrix(y_ref, D_out);

        error_metrics metrics = get_error_metrics(y_0, y_ref);
        print_error_metrics(metrics);

        header_print("info", "Finished running kernel");
        header_print("info", "Running kernel with bare call (run based).");
        utils::print_npu_profile(npu_time, 2.0 * float(L) * float(D_in) * float(D_out));
        header_print("info", "Data size: " << std::to_string(data_size));
        header_print("info", "Bandwidth: " + std::to_string(data_size * 2/ npu_time.first * 1e6 / 1e9) + " GB/s");
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
    assert((M*K) == x.size());
    assert((K*N) == w.size());
    assert((M*N) == y.size());

    const int simd = 8;
    bf16* y_ptr = y.data();
    bf16* x_ptr = x.data();
    bf16* w_ptr = w.data();

    for(int row = 0; row < M; ++row) {
      bf16*  x_row = x_ptr + row*K;
      bf16*  y_row = y_ptr + row*N;

      // process columns in groups of 8
      int col = 0;
      for(; col + simd <= N; col += simd) {
        __m256 acc = _mm256_setzero_ps();

        // dot product: sum_k x[row,k] * w[k,col + 0..7]
        for(int k = 0; k < K; ++k) {
          float  xv = x_row[k].as_float();
          __m256 xb = _mm256_set1_ps(xv);

          // load 8 BF16 weights for this k
          __m128i w16 = _mm_loadu_si128((__m128i*)(w_ptr + k*N + col));
          __m256 w32 = bf16o_fp32(w16);

          acc = _mm256_fmadd_ps(xb, w32, acc);
        }

        // pack and store 8 results
        __m128i out16 = f32o_bf16(acc);
        _mm_storeu_si128((__m128i*)(y_row + col), out16);
      }

      // tail columns
      for(; col < N; ++col) {
        float sum = 0;
        for(int k = 0; k < K; ++k)
          sum += x_row[k].as_float() * w_ptr[k*N + col].as_float();
        y_row[col] = bf16(sum);
      }
    }
}

float get_rmse(buffer<bf16>& y, buffer<bf16>& y_ref){
    float rmse = 0;
    float ref_sum = 0;
    for (int i = 0; i < y.size(); i++){
        ref_sum += y_ref[i].as_float() * y_ref[i].as_float();
        rmse += (y[i].as_float() - y_ref[i].as_float()) * (y[i].as_float() - y_ref[i].as_float());
    }
    return sqrt(rmse / y.size()) / sqrt(ref_sum / y.size());
}
