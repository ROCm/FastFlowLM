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

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("LLAMA_3_2_1B"), "Model type, e.g., LLAMA_3_2_1B");

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();

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


    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    buffer<bf16> embed;

    q4nx_model->load_weights(embed, "model.embed_tokens.weight");

    header_print("info", "Embedding weights:");
    utils::print_matrix(embed, D, 10);

    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

    buffer<bf16> ref_embed;
    references->load_weights(ref_embed, "embed");

    buffer<int32_t> ids;
    references->load_weights(ids, "input_ids");

    // buffer<f32> rope_short;
    // references->load_weights(rope_short, "inv_freq");

    // for (int i = 0; i < rope_short.size(); i++) {
    //     std::cout <<  rope_short[i] << ", ";
    //     if (i % 8 == 7) std::cout << std::endl;
    // }

    header_print("info", "input ids:");
    utils::print_matrix(ids, ids.size());

    header_print("info", "Comparing embeddings...");
    buffer<bf16> embeddings(ref_embed.size());

    for (int i = 0; i < ids.size(); i++) {
        int32_t token_id = ids[i];
        memcpy(embeddings.data() + i * D,
               embed.data() + token_id * D,
               D * sizeof(bf16)
        );
    }
    print_error_metrics(get_error_metrics(embeddings, ref_embed), "Model Embedding Error Metrics");

    header_print("info", "Embedding test completed.");

    header_print("info", "Testing decoding layer...");
    header_print("info", "Initializing NPU manager...");
    
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);

    int context_length = ids.size() < 4096? 4096 : ids.size();
    
    // ids.resize(2);
    // make 2^n context length
    context_length = 1 << static_cast<int>(std::ceil(std::log2(context_length)));
    header_print("info", "Max context length: " + std::to_string(context_length));

    decoding_layer layer(model, lm_config, layer_app_mgr, context_length);

    layer.load_weights(*q4nx_model);
    layer.set_context_length(0);

    tensor_2d<bf16> inputs(ref_embed, D);
    buffer<bf16> output_buffer(ref_embed.size());
    tensor_2d<bf16> outputs(output_buffer, D);
    buffer<bf16> ref_output_buffer;
    references->load_weights(ref_output_buffer, "layer_0_output");
    tensor_2d<bf16> ref_outputs(ref_output_buffer, D);

    buffer<bf16> output;
    std::vector<error_metrics> all_metrics(ids.size());
    header_print("info", "Running decoding layer...");
    time_utils::time_with_unit total_time = {0.0f, "ms"};
    for (uint32_t i = 0; i < ids.size(); i++) {
        time_utils::time_point start = time_utils::now();
        output = layer(inputs[i]);
        time_utils::time_point end = time_utils::now();
        total_time.first += time_utils::duration_ms(start, end).first;
        memcpy(outputs[i].data(), output.data(), output.size() * sizeof(bf16));
        all_metrics[i] = get_error_metrics(outputs[i], ref_outputs[i]);
    }

    header_print("info", "Decoding layer run completed.");
    header_print("info", "Latency:" + std::to_string(total_time.first / ids.size()) + " " + total_time.second);


    for (uint32_t i = 0; i < ids.size(); i++) {
        std::cout << "\tStep " << i << " Cosim:" << all_metrics[i].CosineSimilarity << std::endl;
    }


    int k_idx = 0;
    int v_idx = 0;
    buffer<bf16> k_cache = layer.get_k_cache(k_idx);
    buffer<bf16> v_cache = layer.get_v_cache(v_idx);

    buffer<bf16> ref_k_cache;
    buffer<bf16> ref_v_cache;
    references->load_weights(ref_k_cache, "k_cache");
    references->load_weights(ref_v_cache, "v_cache");
    tensor_2d<bf16> ref_k_cache_tensor(ref_k_cache, DK);
    tensor_2d<bf16> ref_v_cache_tensor(ref_v_cache, DV);

    header_print("info", "K cache:");
    utils::print_matrix(k_cache, DK);
    header_print("info", "ref K cache:");
    utils::print_matrix(ref_k_cache_tensor[k_idx], DK);
    header_print("info", "Comparing K cache...");
    print_error_metrics(get_error_metrics(k_cache, ref_k_cache_tensor[k_idx]), "K Cache Error Metrics");
    
    {
        tensor_2d<bf16> k_cache_head_tensor(k_cache, DH);
        tensor_2d<bf16> ref_k_cache_head_tensor(ref_k_cache_tensor[k_idx], DH);
        for (int i = 0; i < 2; i++) {
            print_error_metrics(get_error_metrics(k_cache_head_tensor[i], ref_k_cache_head_tensor[i]),
                                                 "K Cache Head " + std::to_string(i) + " Error Metrics");
        }
    }
    header_print("info", "V cache:");
    utils::print_matrix(v_cache, DV);
    header_print("info", "ref V cache:");
    utils::print_matrix(ref_v_cache_tensor[v_idx], DV);
    header_print("info", "Comparing V cache...");
    print_error_metrics(get_error_metrics(v_cache, ref_v_cache_tensor[v_idx]), "V Cache Error Metrics");
    {
        tensor_2d<bf16> v_cache_head_tensor(v_cache, DH);
        tensor_2d<bf16> ref_v_cache_head_tensor(ref_v_cache_tensor[v_idx], DH);
        for (int i = 0; i < 2; i++) {
            print_error_metrics(get_error_metrics(v_cache_head_tensor[i], ref_v_cache_head_tensor[i]),
                                                 "V Cache Head " + std::to_string(i) + " Error Metrics");
        }
    }

    // test lm_head

    buffer<bf16> x_input = buffer<bf16>(D);
    memcpy(x_input.data(), embed.data() + D * (ids[ids.size() - 1]), D * sizeof(bf16));

    buffer<bf16> lm_logits = layer.get_logits(x_input);

    header_print("info", "LM Head logits:");
    // utils::print_matrix(lm_logits, 64, 8);

    header_print("info", "Max logits index and value:");
    int max_index = 0;
    float max_value = -1e10f;
    for (int i = 0; i < lm_logits.size(); i++) {
        float val = static_cast<float>(lm_logits[i]);
        if (val > max_value) {
            max_value = val;
            max_index = i;
        }
    }
    header_print("info", "Index: " + std::to_string(max_index) + ", Value: " + std::to_string(max_value));
    header_print("info", "Input ids: " << std::to_string(ids[ids.size() -1]));

    // approximate tps
    float tps = layer.bench(inputs[0]);
    header_print("info", "Approximate TPS: " + std::to_string(tps));

    // long ctx test

    layer.set_context_length(context_length - 128);
    {
        time_utils::time_point start = time_utils::now();
        for (int i = 0; i < 100; i++) {
            layer(inputs[0]);
        }
        time_utils::time_point end = time_utils::now();
        time_utils::time_with_unit duration = time_utils::duration_ms(start, end);
        header_print("info", "Time taken for 100 iterations: " + std::to_string(duration.first) + " " + duration.second);
    }

    return 0;
}
