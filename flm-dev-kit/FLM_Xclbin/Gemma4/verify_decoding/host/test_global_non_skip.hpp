// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __TEST_GLOBAL_NON_SKIP_HPP__
#define __TEST_GLOBAL_NON_SKIP_HPP__

#include "decoding_layer.hpp"
#include "metrices.hpp"


bool test_global_non_skip_layer(
    model_list::model_t model,
    npu_app_manager* layer_app_mgr,
    LM_Config& lm_config,
    Q4NX& q4nx,
    std::string reference_path
){
    int layer_idx;
    if (model == model_list::GEMMA4_E2B ){
        layer_idx = 4;
    }
    else if (model == model_list::GEMMA4_E4B ){
        layer_idx = 5;
    }
    else {
        throw std::runtime_error("Unsupported model type for test_global_non_skip_layer: " + std::to_string(model));
    }
    int max_length = 256;
    int test_seq_len = 10;
    std::unique_ptr<decoding_layer> layer = std::make_unique<decoding_layer>(model, lm_config, layer_app_mgr, layer_idx, max_length);
    header_print("info", "Loading weights...");
    layer->load_weights(q4nx);

    header_print("info", "Load references...");
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    
    header_print("info", "Check embeddings...");
    buffer<int32_t> input_ids;
    references->load_weights(input_ids, "input_ids");

    int ref_size = input_ids.size();

    buffer<bf16> input_embd((size_t)ref_size * layer->D);
    buffer<bf16> input_embd_ref((size_t)ref_size * layer->D);
    tensor_2d<bf16> input_embd_tensor(input_embd_ref, layer->D);
    references->load_weights(input_embd_ref, "layer_" + std::to_string(layer_idx - 1));

    header_print("info", "Input:");
    layer->rms_weights_buffer.sync_to_device();
    buffer<bf16> output;
    // layer->set_context_length(0);
    // buffer<bf16> output = (*layer)(input_embd_tensor[0], input_ids[0]);
    layer->set_context_length(0);
    time_utils::time_point start_time = time_utils::now();
    buffer<bf16> output_ref;
    references->load_weights(output_ref, "layer_" + std::to_string(layer_idx));
    tensor_2d<bf16> output_ref_tensor(output_ref, layer->D);
    float* cosimilarity_sum = new float[test_seq_len];
    for (int i = 0; i < test_seq_len; i++) {
        output = (*layer)(input_embd_tensor[i], input_ids[i]);
        buffer<bf16> output_valid = buffer<bf16>(output.data(), layer->D);
        error_metrics metrics = get_error_metrics(output_valid, output_ref_tensor[i]);
        cosimilarity_sum[i] = metrics.CosineSimilarity;
    }
    for (int i = 0; i < test_seq_len; i++) {
        header_print("info", "Cosine similarity for token " + std::to_string(i) + ": " + std::to_string(cosimilarity_sum[i]));
    }
    delete[] cosimilarity_sum;

    time_utils::time_point end_time = time_utils::now();
    time_utils::time_with_unit duration = time_utils::duration_us(start_time, end_time);
    header_print("info", "Time taken for forward pass: " + std::to_string(duration.first) + " " + duration.second);

    buffer<bf16> output_valid = buffer<bf16>(output.data(), layer->D);
    header_print("info", "Input:");
    utils::print_matrix(input_embd_tensor[test_seq_len - 1], 256, 10);
    header_print("info", "Output:");
    utils::print_matrix(output_valid, 8, 10);

    // check output correctness
    {
        
        header_print("info", "Output ref:");
        utils::print_matrix(output_ref_tensor[test_seq_len - 1], 8, 10);
        print_error_metrics(get_error_metrics(output_valid, output_ref_tensor[test_seq_len - 1]), "output");
    }

    // check kv cache
    if (1){
        buffer<bf16> k_cache = layer->get_k_cache(test_seq_len - 1);
        buffer<bf16> v_cache = layer->get_v_cache(test_seq_len - 1);
        buffer<bf16> k_cache_ref;
        buffer<bf16> v_cache_ref;
        references->load_weights(k_cache_ref, "layer_" + std::to_string(layer_idx) + "_k");
        references->load_weights(v_cache_ref, "layer_" + std::to_string(layer_idx) + "_v");
        tensor_2d<bf16> k_cache_tensor(k_cache_ref, layer->DK);
        tensor_2d<bf16> v_cache_tensor(v_cache_ref, layer->DV);
        print_error_metrics(get_error_metrics(k_cache, k_cache_tensor[test_seq_len - 1]), "k cache");
        print_error_metrics(get_error_metrics(v_cache, v_cache_tensor[test_seq_len - 1]), "v cache");
        header_print("info", "K cache:");
        utils::print_matrix(k_cache, 512, 10);
        header_print("info", "K cache ref:");
        utils::print_matrix(k_cache_tensor[test_seq_len - 1], 512, 10);
        header_print("info", "V cache:");
        utils::print_matrix(v_cache, 512, 10);
        header_print("info", "V cache ref:");
        utils::print_matrix(v_cache_tensor[test_seq_len - 1], 512, 10);
    }
    header_print("info", "Test completed.");

    layer.reset();
    header_print("info", "Layer released.");
    return true;
}


#endif