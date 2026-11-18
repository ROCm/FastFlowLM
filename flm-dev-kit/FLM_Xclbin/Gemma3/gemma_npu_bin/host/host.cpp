// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "llama/llama_npu.hpp"
#include "modules/lm_head.hpp"
#include "gemma/gemma_npu.hpp"
#include "lm_config.hpp"
#include "tokenizer/tokenizer.hpp"
#include "modules/sampler.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "metrices.hpp"

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("model_type,t", arg_utils::po::value<std::string>()->required(), "Model type");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_path = vm["model"].as<std::string>();
    std::string model_type = vm["model_type"].as<std::string>();
    std::string xclbin_path = "build/" + model_type + "/xclbins/flowArch.xclbin";

    if (model_type == "GEMMA3_4B"){
        model_path = model_path + "/Gemma3-4B-NPU2";
    } 
    else{
        std::cout << "Model type not supported: " << model_type << std::endl;
        return 1;
    }

    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Model type: " << model_type << std::endl;
    std::cout << "XCLBIN: " << xclbin_path << std::endl;

    if (!utils::check_file_exists(model_path + "/model.q4nx")) {
        std::cout << "Model file does not exist: " << model_path << std::endl;
        return 1;
    }

    Q4NX weights(model_path);
    LM_Config config;
    config.from_pretrained(model_path);
    Tokenizer tokenizer(model_path);
    // override the default config
    config.layer_xclbin_name = xclbin_path;
    std::cout << "Layer xclbin name: " << config.layer_xclbin_name << std::endl;
    config.num_hidden_layers = 2;
    int target_l = 129;
    std::cout << config._str() << std::endl;
    npu_xclbin_manager npu_instance(npu_device::device_npu2, 0);
    sampler_config sampler_config;
    sampler_config.temperature = 0.7;
    sampler_config.top_k = 5;
    sampler_config.top_p = 0.9;
    sampler_config.rep_penalty = 0.0;
    sampler_config.freq_penalty = 0.0;
    std::unique_ptr<Sampler> sampler = std::make_unique<Sampler>(config.vocab_size, sampler_config);
    gemma_npu gemma_npu(config, &npu_instance, 4096);
    std::unique_ptr<Embedding> embedding = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    embedding->init_weights(&weights, "model.embed_tokens");
    header_print("info", "Loading weights...");
    gemma_npu.load_weights(weights);

    std::cout << "Model loaded!" << std::endl;


    gemma_npu.set_context_length(0);

    buffer<uint32_t> tokens(2048);
    tokens.from_file("token_id.bin");
    
    std::cout << "Tokens: ";
    for (int i = 0; i < 5; i++) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;
    buffer<bf16> y;
    gemma_npu.set_context_length(0);
    for (int i = 0; i < target_l; i++){
        y = gemma_npu.forward(tokens[i]);
    }
    buffer<bf16> y_ref(2048 * 2560);
    y_ref.from_file("y_ref.bin");
    tensor_2d<bf16> y_ref_tensor(y_ref, 2560);

    buffer<bf16> k_cache = gemma_npu.get_k_cache(0, target_l - 1);
    buffer<bf16> v_cache = gemma_npu.get_v_cache(0, target_l - 1);
    buffer<bf16> k_cache_ref(1024 * 2048);
    k_cache_ref.from_file("k_cache.bin");
    buffer<bf16> v_cache_ref(1024 * 2048);
    v_cache_ref.from_file("v_cache.bin");
    tensor_2d<bf16> k_cache_ref_tensor(k_cache_ref, 1024);
    tensor_2d<bf16> v_cache_ref_tensor(v_cache_ref, 1024);
    buffer<bf16> k_cache_ref_idx = k_cache_ref_tensor[target_l - 1];
    buffer<bf16> v_cache_ref_idx = v_cache_ref_tensor[target_l - 1];
    header_print("info", "k_cache: ");
    utils::print_matrix(k_cache, 128);
    header_print("info", "k_cache_ref: ");
    utils::print_matrix(k_cache_ref_idx, 128);
    print_error_metrics(get_error_metrics(k_cache, k_cache_ref_idx));
    header_print("info", "v_cache: ");
    utils::print_matrix(v_cache, 128);
    header_print("info", "v_cache_ref: ");
    utils::print_matrix(v_cache_ref_idx, 128);
    print_error_metrics(get_error_metrics(v_cache, v_cache_ref_idx));


    header_print("info", "y: " << y.size());
    buffer<bf16> y_npu(y.data(), 2560);
    utils::print_matrix(y_npu, 512, 16);
    header_print("info", "y_ref: ");
    buffer<bf16> y_cpu = y_ref_tensor[target_l - 1];
    utils::print_matrix(y_cpu, 512, 16);
    header_print("info", "y error: ");
    print_error_metrics(get_error_metrics(y_npu, y_cpu));
   
    std::cout << std::endl;




    return 0;
}