// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "models.h"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"

#include <iostream>

#include "test_swa_non_skip.hpp"
#include "test_swa_skip.hpp"
#include "test_global_non_skip.hpp"
#include "test_global_skip.hpp"
#include "test_swa_non_skip_detail.hpp"

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("GEMMA4_E2B"), "Model type, e.g., GEMMA4_E2B");
    desc.add_options()("layer,l", arg_utils::po::value<int>()->default_value(0), "Layer number to test, 0 - 35");
    desc.add_options()("test_seq_len,s", arg_utils::po::value<int>()->default_value(10), "Test sequence length for global skip layer");

    const int max_len = 1;

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();
    int layer_idx = vm["layer"].as<int>();
    int test_seq_len = vm["test_seq_len"].as<int>();


    header_print("info", "Testing model: " + model_type);

    model_list::model_t model  = model_list::get_model_type(model_type);
    std::string model_path     = model_list::get_model_path(model);
    std::string reference_path = model_list::get_reference_path(model);
    header_print("info", "Model path: " + model_path);
    LM_Config lm_config;
    lm_config.from_pretrained(model_path);

    int D = lm_config.hidden_size;
    int DH = lm_config.head_dim;
    int DQ = lm_config.num_attention_heads * lm_config.head_dim;
    int DK = lm_config.num_key_value_heads * lm_config.head_dim;
    int DV = lm_config.num_key_value_heads * lm_config.head_dim;
    // overide the layer xclbin

    lm_config.layer_xclbin_name = model_list::get_layer_xclbin_path(model);
    header_print("info", "Layer xclbin path: " + lm_config.layer_xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(D));


    // std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    // buffer<bf16> embed;

    // q4nx_model->load_weights(embed, "model.embed_tokens.weight");

    // header_print("info", "Embedding weights:");
    // utils::print_matrix(embed, D, 10);
    header_print("info", "Testing decoding layer...");
    header_print("info", "Initializing NPU manager...");
    
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);
    // std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

    std::string ref_path = model_list::get_reference_path(model);

    std::unique_ptr<Q4NX> q4nx = std::make_unique<Q4NX>(model_path);
    test_swa_non_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path);
    test_global_non_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path);
    test_swa_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path);
    test_global_skip_layer(model, layer_app_mgr, lm_config, *q4nx, ref_path, test_seq_len);
    q4nx.reset();
    return 0;
}
    