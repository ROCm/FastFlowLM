// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "llama/llama_npu.hpp"
#include "modules/lm_head.hpp"
#include "gemma/gemma_npu.hpp"
#include "gemma_text/gemma_text_npu.hpp"
#include "lm_config.hpp"
#include "tokenizer/tokenizer.hpp"
#include "modules/sampler.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "metrices.hpp"

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
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("model_type,t", arg_utils::po::value<std::string>()->required(), "Model type");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_path = vm["model"].as<std::string>();
    std::string model_type = vm["model_type"].as<std::string>();
    std::string xclbin_path = "build/" + model_type + "/xclbins/flowArch.xclbin";

    if (model_type == "GEMMA3_4B"){
        model_path = model_path + "/Gemma3-4B-NPU2";
    } 
    else if (model_type == "GEMMA3_1B"){
        model_path = model_path + "/Gemma3-1B-NPU2";
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
    // config.num_hidden_layers = 1;
    std::cout << config._str() << std::endl;
    npu_xclbin_manager npu_instance(npu_device::device_npu2, 0);
    sampler_config sampler_config;
    sampler_config.temperature = 0.7;
    sampler_config.top_k = 5;
    sampler_config.top_p = 0.9;
    sampler_config.rep_penalty = 0.0;
    sampler_config.freq_penalty = 0.0;
    std::unique_ptr<Sampler> sampler = std::make_unique<Sampler>(config.vocab_size, sampler_config);
    gemma_text_npu gemma_text_npu(config, &npu_instance, 16384);
    std::unique_ptr<Embedding> embedding = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    embedding->init_weights(&weights, "model.embed_tokens");
    header_print("info", "Loading weights...");
    gemma_text_npu.load_weights(weights);

    std::cout << "Model loaded!" << std::endl;

    std::cout << "start decoder" << std::endl;
    int32_t tokens[] = {2,    105,   2364,    107,   3689,    563,    506,   7606,    528,
        21038, 236881,    106,    107,    105,   4368,    107,   1509,    563,
        17635, 236761,    106,    107,    105,   2364,    107,   3689,   1719,
          563,    625, 236881,    106,    107,    105,   4368,    107};
    int32_t tokens_len = sizeof(tokens) / sizeof(int32_t);

    int target_l = tokens_len;
    std::cout << "Tokens length: " << tokens_len << std::endl;
    std::cout << "Tokens: ";
    for (int i = 0; i < tokens_len; i++) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;
    buffer<bf16> y;
    header_print("info", "start forward");
    gemma_text_npu.set_context_length(0);
    buffer<bf16> y_ref_attn;
    load_data(y_ref_attn, "y_ref.bin");
    for (int i = 0; i < target_l; i++){
        std::cout << i + 1 << "th step" << std::endl;
        y = gemma_text_npu.forward(tokens[i]);
        buffer<bf16> y_ref_attn_valid(y_ref_attn.data() + 1152 * (i), 1152);
        print_error_metrics(get_error_metrics(y, y_ref_attn_valid));
    }
    utils::print_matrix(y, 128);
    header_print("info", "forward done");
    buffer<bf16> k_cache_0 = gemma_text_npu.get_k_cache(0, target_l - 1);
    buffer<bf16> v_cache_0 = gemma_text_npu.get_v_cache(0, target_l - 1);
    std::cout << "k_cache_0: " << std::endl;
    utils::print_matrix(k_cache_0, 128);
    std::cout << "v_cache_0: " << std::endl;
    utils::print_matrix(v_cache_0, 128);

    return 0;
}