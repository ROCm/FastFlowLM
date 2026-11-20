// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <iostream>
#include <iomanip>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "nlohmann/json.hpp"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "gpt_oss/gpt_oss_npu.hpp"
#include "metrices.hpp"

void comparing_print(buffer<bf16>& y, buffer<bf16>& y_ref, int col_size, int n_printable_rows = 10, int n_printable_cols = 10){
    tensor_2d<bf16> y_tensor(y, col_size);
    tensor_2d<bf16> y_ref_tensor(y_ref, col_size);
    for (int i = 0; i < y.size() / col_size; i++){
        std::cout << "\033[31m" << std::flush;
        utils::print_matrix(y_tensor[i], col_size, 1, n_printable_cols);
        std::cout << "\033[0m" << std::flush;
        std::cout << "\033[32m" << std::flush;
        utils::print_matrix(y_ref_tensor[i], col_size, 1, n_printable_cols);
        std::cout << "\033[0m" << std::flush;
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

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("/path/to/models/GPT-OSS-20B-NPU2"), "Model PATH");
    desc.add_options()("active_layers,n", arg_utils::po::value<int>()->default_value(1), "Number of active layers");    
    desc.add_options()("l_begin,b", arg_utils::po::value<int>()->default_value(0), "Start length"); 
    desc.add_options()("target_l,l", arg_utils::po::value<int>()->default_value(130), "Target length");
    desc.add_options()("cascade,c", arg_utils::po::value<bool>()->default_value(false), "Cascade");
    arg_utils::parse_options(argc, argv, desc, vm);

    constexpr int DQ = 4096;
    constexpr int DK = 512;
    constexpr int DV = 512;
    constexpr int D = 2880;

    constexpr int D_USING = D;


    int active_layers = vm["active_layers"].as<int>();
    std::string model_path = vm["model"].as<std::string>();
    int l_begin = vm["l_begin"].as<int>();
    int target_l = vm["target_l"].as<int>();
    bool cascade = vm["cascade"].as<bool>();
    if (active_layers < 1) {
        std::cerr << "Active layers must be greater than 0" << std::endl;
        return 1;
    }
    header_print("info", "Initializing chat model...");
    // std::string xclbin_path = "build/GPTOSS_20B/xclbins/flowArch.xclbin";

    // SafeTensors ref_tensors(model_path + "/prefill_data.safetensors");

    buffer<int32_t> sample_token_id(135);
    sample_token_id.from_file("../../data/sample_token_id.bin");

    buffer<bf16> y_ref;
    SafeTensors ref_tensors("../../data/prefill_data_debug.safetensors");
    ref_tensors.load_weights(y_ref, "layer_" + std::to_string(active_layers - 1) + "_after_residual_2");
    // ref_tensors.load_weights(y_ref, "layer_" + std::to_string(active_layers - 1) + "_v_proj");
    std::cout << y_ref.size() << std::endl;
    assert(y_ref.size() % D_USING == 0);
    // buffer<bf16> dequant_weight_ref(2944 * 3072 * 3);
    // dequant_weight_ref.from_file("../../data/layer_0_mlp_dequant_weight.bin");

    tensor_2d<bf16> y_ref_tensor(y_ref, D_USING);

    LM_Config config;
    config.from_pretrained(model_path);
    config.num_hidden_layers = active_layers; // override for debugging
    std::cout << config._str() << std::endl;
    npu_xclbin_manager npu_instance(npu_device::device_npu2, 0);

    Q4NX weights(model_path);

    gpt_oss_npu gpt_oss_npu(config, &npu_instance, 4096);
    gpt_oss_npu.load_weights(weights);

    std::vector<int> ids;
    for (int i = 0; i < target_l; i++){
        ids.push_back(sample_token_id[i + l_begin]);
    }
    std::cout << "Start!" << std::endl;
    for (int i = 0; i < l_begin; i++){
        gpt_oss_npu.forward(sample_token_id[i]);
    }
    buffer<bf16> y = gpt_oss_npu.prefill(ids);


    // print_error_metrics(get_error_metrics(y, dequant_weight_ref));
    // return 0;
    assert(y.size() % D_USING == 0);
    tensor_2d<bf16> y_tensor = tensor_2d(y, D_USING);


    header_print("info", "y:");
    utils::print_matrix(y, D_USING, 10, 20);
    header_print("info", "y_ref:");
    buffer<bf16> y_used = buffer<bf16>(y_ref.data() + l_begin * D_USING, target_l * D_USING);
    utils::print_matrix(y_used, D_USING, 10, 20);

    
    std::vector<error_metrics> metrics;
    for (int i = 0; i < target_l; i++) {
        metrics.push_back(get_error_metrics(y_tensor[i], y_ref_tensor[i + l_begin]));
        error_metrics error_metrics;
        // float square_sum = 0;
        // float error_sum = 0;
        // float error_square_sum = 0;
        // float ref_sum = 0;
        // float dot_product = 0;
        // for (int j = 0; j < D_USING; j++) {
        //     square_sum += (y_ref_tensor[i + l_begin][j]) * (y_ref_tensor[i + l_begin][j]);
        //     error_sum += (y_tensor[i][j] - y_ref_tensor[i + l_begin][j]);
        //     error_square_sum += (y_tensor[i][j] - y_ref_tensor[i + l_begin][j]) * (y_tensor[i][j] - y_ref_tensor[i + l_begin][j]);
        //     ref_sum += y_ref_tensor[i + l_begin][j];
        //     dot_product += y_tensor[i][j] * y_ref_tensor[i + l_begin][j];
        // }
        // error_metrics.RelativeL2 = sqrt(square_sum / ref_sum);
        // error_metrics.CosineSimilarity = dot_product / square_sum;
        // error_metrics.RMSE = sqrt(square_sum / D_USING);
        // error_metrics.RelativeL1 = error_sum / ref_sum;
        // metrics.push_back(error_metrics);
    }
    std::ofstream error_file("error.txt");
    
    // Write error metrics to file instead of console
    error_file << std::setw(12) << "Time index" << "| " 
               << std::setw(10) << "RelativeL2 " << "| " 
               << std::setw(17) << "CosineSimilarity" << " | " 
               << std::setw(10) << "RMSE" << " | " 
               << std::setw(10) << "RelativeL1" << std::endl;
    for (int i = 0; i < target_l; i++) {
        error_file << std::dec << std::setw(12) << i << "| " 
                   << std::fixed << std::setprecision(6) << std::setw(10) << metrics[i].RelativeL2 << " | " 
                   << std::fixed << std::setprecision(6) << std::setw(17) << metrics[i].CosineSimilarity << " | " 
                   << std::fixed << std::setprecision(6) << std::setw(10) << metrics[i].RMSE << " | " 
                   << std::fixed << std::setprecision(6) << std::setw(10) << metrics[i].RelativeL1 << std::endl;
    }
    error_file.close();
    // error_file << "-----------------------------------------------------------------" << std::endl;
    // buffer<bf16> y_ref_used = buffer<bf16>(y_ref.data(), target_l * 2880);
    // error_metrics error_metrics_summary = get_error_metrics(y_whole, y_ref_used);
    // error_file << std::dec << std::setw(12) << "summary" << "| " 
    //            << std::fixed << std::setprecision(6) << std::setw(10) << error_metrics_summary.RelativeL2 << " | " 
    //            << std::fixed << std::setprecision(6) << std::setw(17) << error_metrics_summary.CosineSimilarity << " | " 
    //            << std::fixed << std::setprecision(6) << std::setw(10) << error_metrics_summary.RMSE << " | " 
    //            << std::fixed << std::setprecision(6) << std::setw(10) << error_metrics_summary.RelativeL1 << std::endl;
    // header_print("info", "k_cache: ");
    // utils::print_matrix(k_cache, 128);
    // header_print("info", "k_cache_ref: ");
    // utils::print_matrix(k_cache_ref_tensor[target_l - 1], 128);
    // header_print("info", "v_cache: ");
    // utils::print_matrix(v_cache, 128);
    // header_print("info", "v_cache_ref: ");
    // utils::print_matrix(v_cache_ref_tensor[target_l - 1], 128);

    // error_metrics error_metrics_k_cache = get_error_metrics(k_cache, k_cache_ref_tensor[target_l - 1]);
    // error_metrics error_metrics_v_cache = get_error_metrics(v_cache, v_cache_ref_tensor[target_l - 1]);
    // header_print("info", "k_cache error: ");
    // print_error_metrics(error_metrics_k_cache);
    // header_print("info", "v_cache error: ");
    // print_error_metrics(error_metrics_v_cache);

   

    return 0;
}