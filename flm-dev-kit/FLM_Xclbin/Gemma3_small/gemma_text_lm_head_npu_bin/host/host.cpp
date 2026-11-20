// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <iostream>
#include <cmath>
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "gemma_text/gemma_text_npu.hpp"
#include "gemma_text/gemma_text_lm_head.hpp"
#include "lm_config.hpp"
#include "tokenizer/tokenizer.hpp"
#include "modules/sampler.hpp"
#include "modules/embedding.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("model_type,t", arg_utils::po::value<std::string>()->required(), "Model type");
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_path = vm["model"].as<std::string>();
    std::string model_type = vm["model_type"].as<std::string>();
    std::string xclbin_path = "build/" + model_type + "/xclbins/qmm.xclbin";

    if (model_type == "GEMMA3_1B"){
        model_path = model_path + "/Gemma3-1B-NPU2";
    }
    else{
        std::cout << "Model type not supported: " << model_type << std::endl;
        return 1;
    }

    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Model type: " << model_type << std::endl;
    std::cout << "XCLBIN: " << xclbin_path << std::endl;  
    Q4NX weights(model_path);
    LM_Config config;
    config.from_pretrained(model_path);
    config.lm_head_xclbin_name = xclbin_path;
    std::cout << config._str() << std::endl;
    npu_xclbin_manager npu(npu_device::device_npu2, 0);
    std::unique_ptr<GemmaTextLMHead> lm_head;
    std::unique_ptr<Embedding> embedding;
    sampler_config sampler_config;
    sampler_config.temperature = 0.7;
    sampler_config.top_k = 5;
    sampler_config.top_p = 0.9;
    sampler_config.rep_penalty = 0.0;
    sampler_config.freq_penalty = 0.0;
    std::unique_ptr<Sampler> sampler = std::make_unique<Sampler>(config.vocab_size, sampler_config);
    lm_head = std::make_unique<GemmaTextLMHead>(config, &npu);
    embedding = std::make_unique<Embedding>(config.vocab_size, config.hidden_size);
    embedding->init_weights(&weights, "model.embed_tokens");
    lm_head->load_weights(weights);

    if (!utils::check_file_exists(model_path + "/model.q4nx")) {
        std::cout << "Model file does not exist: " << model_path << std::endl;
        return 1;
    }

    std::cout << "Model loaded!" << std::endl;
    buffer<bf16> x_lm_head_exposed = lm_head->x_exposed();
    embedding->forward(142451, x_lm_head_exposed);
    utils::print_matrix(x_lm_head_exposed, 1152);
    lm_head->execute();
    buffer<bf16> y = lm_head->wait();
    std::cout << "y size: " << y.size() << std::endl;
    utils::print_matrix(y, 4);
    int new_token = sampler->sample(y);
    std::cout << "New token: " << new_token << std::endl;
    utils::print_matrix(y, 4);

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