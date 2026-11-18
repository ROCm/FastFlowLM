// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
#include "linear_decoding_layer.hpp"
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

    const int max_len = 3;

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
    header_print("info", "Testing decoding layer...");
    header_print("info", "Initializing NPU manager...");
    
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* layer_app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);


    if (1)
    {
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);

        buffer<bf16> ref_embed;
        references->load_weights(ref_embed, "embeddings");

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


        int context_length = ids.size() < 4096? 4096 : ids.size();
        
        // ids.resize(2);
        // make 2^n context length
        context_length = 1 << static_cast<int>(std::ceil(std::log2(context_length)));
        header_print("info", "Max context length: " + std::to_string(context_length));

        decoding_layer layer(model, lm_config, layer_app_mgr, context_length);

        layer.load_weights(*q4nx_model);
        layer.set_context_length(0);

        buffer<float> rope_freqs;
        references->load_weights(rope_freqs, "inv_freq");

        layer.set_rope_frequency(rope_freqs);


        tensor_2d<bf16> inputs(ref_embed, D);
        buffer<bf16> output_buffer(ref_embed.size());
        tensor_2d<bf16> outputs(output_buffer, D);
        buffer<bf16> ref_output_buffer;
        references->load_weights(ref_output_buffer, "layer_output");
        tensor_2d<bf16> ref_outputs(ref_output_buffer, D);

        buffer<bf16> output;
        std::vector<error_metrics> all_metrics(ids.size());
        header_print("info", "Running decoding layer...");
        time_utils::time_with_unit total_time = {0.0f, "ms"};
        // buffer<bf16> Q(DQ);
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            time_utils::time_point start = time_utils::now();
            output = layer(inputs[i]);
            time_utils::time_point end = time_utils::now();
            total_time.first += time_utils::duration_ms(start, end).first;
            memcpy(outputs[i].data(), output.data(), D * sizeof(bf16));
            // memcpy(Q.data(), output.data() + D, DQ * sizeof(bf16));
            all_metrics[i] = get_error_metrics(outputs[i], ref_outputs[i]);
        }

        header_print("info", "Decoding layer run completed.");
        header_print("info", "Latency:" + std::to_string(total_time.first / (ids.size() > max_len? max_len : ids.size())) + " " + total_time.second);
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            std::cout << "\tStep " << i << " Cosim:" << all_metrics[i].CosineSimilarity << std::endl;
        }
        // header_print("info", "QQQ");
        // buffer<bf16> q_cache;
        // references->load_weights(q_cache, "attention_output_gate");
        // buffer<bf16> q_cache_ref = buffer<bf16>(q_cache.data(), DQ);
        // header_print("info", "Q ref:");
        // utils::print_matrix(q_cache_ref, DH);

        // header_print("info", "Q cache:");
        // utils::print_matrix(Q, DH);
        // header_print("info", "Comparing Q cache...");
        // print_error_metrics(get_error_metrics(Q, q_cache_ref), "Q Cache Error Metrics");

        int k_idx = 0;
        int v_idx = 0;
        buffer<bf16> k_cache = layer.get_k_cache(k_idx);
        buffer<bf16> v_cache = layer.get_v_cache(v_idx);

        buffer<bf16> ref_k_cache;
        buffer<bf16> ref_v_cache;
        references->load_weights(ref_k_cache, "k_embed");
        references->load_weights(ref_v_cache, "v_proj");
        tensor_2d<bf16> ref_k_cache_tensor(ref_k_cache, DK);
        tensor_2d<bf16> ref_v_cache_tensor(ref_v_cache, DV);

        header_print("info", "K cache:");
        utils::print_matrix(k_cache, DH);
        header_print("info", "ref K cache:");
        utils::print_matrix(ref_k_cache_tensor[k_idx], DH);
        header_print("info", "Comparing K cache...");
        print_error_metrics(get_error_metrics(k_cache, ref_k_cache_tensor[k_idx]), "K Cache Error Metrics");
        
        {
            tensor_2d<bf16> k_cache_head_tensor(k_cache, DH);
            tensor_2d<bf16> ref_k_cache_head_tensor(ref_k_cache_tensor[k_idx], DH);
            for (int i = 0; i < lm_config.num_key_value_heads; i++) {
                print_error_metrics(get_error_metrics(k_cache_head_tensor[i], ref_k_cache_head_tensor[i]),
                                                    "K Cache Head " + std::to_string(i) + " Error Metrics");
            }
        }
        header_print("info", "V cache:");
        utils::print_matrix(v_cache, DH);
        header_print("info", "ref V cache:");
        utils::print_matrix(ref_v_cache_tensor[v_idx], DH);
        header_print("info", "Comparing V cache...");
        print_error_metrics(get_error_metrics(v_cache, ref_v_cache_tensor[v_idx]), "V Cache Error Metrics");
        {
            tensor_2d<bf16> v_cache_head_tensor(v_cache, DH);
            tensor_2d<bf16> ref_v_cache_head_tensor(ref_v_cache_tensor[v_idx], DH);
            for (int i = 0; i < lm_config.num_key_value_heads; i++) {
                print_error_metrics(get_error_metrics(v_cache_head_tensor[i], ref_v_cache_head_tensor[i]),
                                                    "V Cache Head " + std::to_string(i) + " Error Metrics");
            }
        }

    }
    if (1){
        std::string ref_path = model_list::get_model_path(model);
        ref_path = ref_path + "/gated_deltanet_0.8B.safetensors";
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(ref_path);

        buffer<bf16> ref_embed;
        references->load_weights(ref_embed, "embeddings");

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

        tensor_2d<bf16> inputs(ref_embed, D);
        std::string tb_data_name = "alpha";
        int time_steps = 3;
        const int tb_offset = D;
        const int tb_size = 0;
        linear_decoding_layer linear_layer(model, lm_config, layer_app_mgr, tb_size);
        linear_layer.load_weights(*q4nx_model);
        buffer<bf16> lout;
        buffer<bf16> output_buffer(ref_embed.size());
        tensor_2d<bf16> outputs(output_buffer, D);
        buffer<bf16> ref_output_buffer;
        references->load_weights(ref_output_buffer, "layer_output");
        tensor_2d<bf16> ref_outputs(ref_output_buffer, D);
       
        std::vector<error_metrics> all_metrics(ids.size());
        header_print("info", "Running decoding layer...");
        time_utils::time_with_unit total_time = {0.0f, "ms"};
        buffer<bf16> output;
        // buffer<bf16> Q(DQ);
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            time_utils::time_point start = time_utils::now();
            output = linear_layer(inputs[i]);
            time_utils::time_point end = time_utils::now();
            total_time.first += time_utils::duration_ms(start, end).first;
            memcpy(outputs[i].data(), output.data(), D * sizeof(bf16));
            // memcpy(Q.data(), output.data() + D, DQ * sizeof(bf16));
            all_metrics[i] = get_error_metrics(outputs[i], ref_outputs[i]);
        }

        header_print("info", "Decoding layer run completed.");
        header_print("info", "Latency:" + std::to_string(total_time.first / (ids.size() > max_len? max_len : ids.size())) + " " + total_time.second);
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            std::cout << "\tStep " << i << " Cosim:" << all_metrics[i].CosineSimilarity << std::endl;
        }

        // for (int i = 0; i < time_steps; i++) {
        //     lout = linear_layer(inputs[i]);
        // }
        
        // buffer<bf16> l_tb = buffer<bf16>(lout.data() + D, 16);
        // buffer<bf16> l1_tb = buffer<bf16>(lout.data() + D + 32, 16);


        // buffer<bf16> all_data;
        // buffer<bf16> tb_data_all;
        // buffer<float> alpha;

        // references->load_weights(all_data, "layer_output");
        // // references->load_weights(tb_data_all, tb_data_name);
        // references->load_weights(alpha, "alpha");
        // buffer<bf16> alpha_bf16(alpha.size());
        // tensor_2d<bf16> alpha_tensor(alpha_bf16, 16);
        // for (int i = 0; i < alpha.size(); i++) {
        //     alpha_bf16[i] = static_cast<bf16>(alpha[i]);
        // }
        // header_print("info", "Alpha ref:");
        // utils::print_matrix(alpha_tensor[time_steps - 1], 8);
        // header_print("info", "Alpha:");
        // utils::print_matrix(l_tb, 8);

        // references->load_weights(alpha, "beta");
        // for (int i = 0; i < alpha.size(); i++) {
        //     alpha_bf16[i] = static_cast<bf16>(alpha[i]);
        // }


        // header_print("info", "Beta ref:");
        // utils::print_matrix(alpha_tensor[time_steps - 1], 8);
        // header_print("info", "Beta:");
        // utils::print_matrix(l1_tb, 8);


        // buffer<bf16> out = buffer<bf16>(lout.data(), D);
        // buffer<bf16> ref = buffer<bf16>(all_data.data() + D * (time_steps - 1), D);
        // buffer<bf16> tb_ref = buffer<bf16>(tb_data_all.data() + tb_size * (time_steps - 1), tb_size);

        // header_print("info", "Out:");
        // utils::print_matrix(out, 32);
        // header_print("info", "Ref:");
        // utils::print_matrix(ref, 32);
        // print_error_metrics(get_error_metrics(out, ref), "Out");

        // header_print("info", "TB:");
        // utils::print_matrix(l_tb, 8);
        // header_print("info", "Ref:");
        // utils::print_matrix(tb_ref, 8);
        // print_error_metrics(get_error_metrics(l_tb, tb_ref), "TB");

        // // {
        // //     buffer<bf16> q = buffer<bf16>(l_tb.data(), 2048);
        // //     buffer<bf16> q_ref = buffer<bf16>(tb_ref.data(), 2048);
        // //     header_print("info", "Q:");
        // //     utils::print_matrix(q, 128);
        // //     header_print("info", "Q Ref:");
        // //     utils::print_matrix(q_ref, 128);
        // //     print_error_metrics(get_error_metrics(q, q_ref), "Q");
        // //     buffer<bf16> k = buffer<bf16>(l_tb.data() + 2048, 2048);
        // //     buffer<bf16> k_ref = buffer<bf16>(tb_ref.data() + 2048, 2048);
        // //     header_print("info", "K:");
        // //     utils::print_matrix(k, 128);
        // //     header_print("info", "K Ref:");
        // //     utils::print_matrix(k_ref, 128);
        // //     print_error_metrics(get_error_metrics(k, k_ref), "K");
        // //     buffer<bf16> v = buffer<bf16>(l_tb.data() + 4096, 4096);
        // //     buffer<bf16> v_ref = buffer<bf16>(tb_ref.data() + 4096, 4096);
        // //     // {
        // //     //     for (int i = 0; i < v_ref.size(); i++) {
        // //     //         float vv = v_ref[i];
        // //     //         vv = vv / (1 + exp(-vv));
        // //     //         v_ref[i] = vv;

        // //     //     }
        // //     // }
        // //     header_print("info", "V:");
        // //     utils::print_matrix(v, 128);
        // //     header_print("info", "V Ref:");
        // //     utils::print_matrix(v_ref, 128);
        // //     print_error_metrics(get_error_metrics(v, v_ref), "V");

        // //     // {
        // //     //     tensor_2d<bf16> v_tensor(v, 128);
        // //     //     tensor_2d<bf16> v_ref_tensor(v_ref, 128);
        // //     //     for (int i = 0; i < 32; i++) {
        // //     //         print_error_metrics(get_error_metrics(v_tensor[i], v_ref_tensor[i]),
        // //     //                                             "V Head " + std::to_string(i) + " Error Metrics");
        // //     //     }
        // //     // }
        // // }
     
        // buffer<bf16> conv_cache = linear_layer.get_conv_cache();
        // header_print("info", "Conv cache:");
        // utils::print_matrix(conv_cache, 6144);
        // buffer<bf16> s_cache = linear_layer.get_S_cache(31);
        // header_print("info", "S:");
        // utils::print_matrix(s_cache, 128);

        // buffer<bf16> v_out = buffer<bf16>(conv_cache.data() + 6144 * (3 - time_steps) + 6144, 6144);
        // buffer<bf16> v_ref_all;
        // references->load_weights(v_ref_all, "qkv_proj");
        // buffer<bf16> v_ref = buffer<bf16>(v_ref_all.data() + 6144, 6144);
        // print_error_metrics(get_error_metrics(v_out, v_ref), "linear v " );
        // header_print("info", "V out:");
        // utils::print_matrix(v_out, 128);
        // header_print("info", "V Ref:");
        // utils::print_matrix(v_ref, 128);

    }
    return 0;
}
