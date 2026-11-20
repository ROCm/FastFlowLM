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
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("QWEN3_6_35B_A3B"), "Model type, e.g., QWEN3_6_35B_A3B");
    desc.add_options()("len,l", arg_utils::po::value<int>()->default_value(1), "Length to test");
    desc.add_options()("type,t", arg_utils::po::value<std::string>()->default_value("linear"), "Layer type to test: linear, attn, both");
    
    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();
    int max_len = vm["len"].as<int>();
    std::string test_type = vm["type"].as<std::string>();
    bool test_linear_layer = false;
    bool test_attn_layer = false;
    if (test_type == "linear") {
        test_linear_layer = true;
    }
    else if (test_type == "attn") {
        test_attn_layer = true;
    }
    else if (test_type == "both") {
        test_linear_layer = true;
        test_attn_layer = true;
    }
    else {
        std::cout << "Error: Unknown layer type, must be linear,attn or both" << std::endl;
        exit(1);
    }

    header_print("info", "Testing model: " + model_type);

    model_list::model_t model  = model_list::get_model_type(model_type);
    std::string model_path     = model_list::get_model_path(model);
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


    if (test_attn_layer)
    {   


        int layer_idx = 31;
        // std::string reference_path = model_list::get_detail_reference_path(model, false);
        std::string reference_path = "/path/to/models/Qwen3.6-35B-A3B-NPU2/activations.safetensors";
        std::cout << "loading reference from  " << reference_path << " " << std::endl;
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
        buffer<bf16> ref_embed;
        references->load_weights(ref_embed, "embeddings");

        buffer<int32_t> ids;
        references->load_weights(ids, "input_ids");

        header_print("info", "input ids:");
        utils::print_matrix(ids, ids.size());
 
        header_print("info", "Comparing embeddings...");
        buffer<bf16> embeddings(ref_embed.size());
        size_t vocab_size = embed.size() / D;

        for (int i = 0; i < ids.size(); i++) {
            int32_t token_id = ids[i];
            memcpy(embeddings.data() + i * D,
                embed.data() + token_id * D,
                D * sizeof(bf16)
            );
        }
        print_error_metrics(get_error_metrics(embeddings, ref_embed), "Model Embedding Error Metrics");

        header_print("info", "Embedding test completed.");


        int context_length = ids.size() < 32768? 32768 : ids.size();
        
        // ids.resize(2);
        // make 2^n context length
        context_length = 1 << static_cast<int>(std::ceil(std::log2(context_length)));
        header_print("info", "Max context length: " + std::to_string(context_length));

        decoding_layer layer(model, lm_config, layer_app_mgr, context_length);

        layer.load_weights(*q4nx_model, layer_idx);

        buffer<float> rope_freqs;
        references->load_weights(rope_freqs, "inv_freq");

        layer.set_rope_frequency(rope_freqs);
        layer.set_context_length(0);


        tensor_2d<bf16> inputs(ref_embed, D);
        buffer<bf16> output_buffer(ref_embed.size());
        tensor_2d<bf16> outputs(output_buffer, D);

        buffer<bf16> decode_output;
        std::vector<error_metrics> all_metrics(ids.size());
        header_print("info", "Running decoding layer...");
        time_utils::time_with_unit total_time = {0.0f, "ms"};
        
        buffer<bf16> layer_input_ref;
        buffer<bf16> layer_shared_expert_output_before_sigmoid;        
        buffer<bf16> layer_shared_expert_output_after_sigmoid;
        buffer<bf16> final_hidden_ref_all;
        buffer<bf16> layer_post_attention_layernorm;
        buffer<int32_t> layer_selected_experts;
        buffer<bf16> layer_share_down;
        buffer<bf16> layer_share_glu_act;
        buffer<bf16> layer_share_gate;
        buffer<bf16> layer_share_up;
        buffer<bf16> layer_mlp_output;

        buffer<bf16> layer_activated;
        buffer<bf16> layer_down;

        

        references->load_weights(layer_input_ref, "layer_"+ std::to_string(layer_idx)+"_input");
        references->load_weights(final_hidden_ref_all, "layer_"+ std::to_string(layer_idx)+"_output");
        references->load_weights(layer_shared_expert_output_before_sigmoid, "layer_"+ std::to_string(layer_idx)+"_shared_expert_output_before_sigmoid");        
        references->load_weights(layer_shared_expert_output_after_sigmoid, "layer_"+ std::to_string(layer_idx)+"_shared_expert_output_after_sigmoid");
        references->load_weights(layer_selected_experts, "layer_"+ std::to_string(layer_idx)+"_selected_experts");
        references->load_weights(layer_post_attention_layernorm, "layer_" + std::to_string(layer_idx)+"_post_attention_layernorm");
        references->load_weights(layer_share_down, "layer_" +  std::to_string(layer_idx)+ "_share_down");
        references->load_weights(layer_share_glu_act, "layer_" +  std::to_string(layer_idx)+ "_share_glu_act");
        references->load_weights(layer_share_gate, "layer_" +  std::to_string(layer_idx)+ "_share_gate");
        references->load_weights(layer_share_up, "layer_" +  std::to_string(layer_idx)+ "_share_up");
        references->load_weights(layer_mlp_output, "layer_" +  std::to_string(layer_idx)+"_mlp_output");
        references->load_weights(layer_activated, "layer_" + std::to_string(layer_idx)+"_activated");
        references->load_weights(layer_down, "layer_" + std::to_string(layer_idx)+"_down");




        tensor_2d<bf16> layer_input_ref_2d(layer_input_ref, D);
        tensor_2d<bf16> final_hidden_ref_all_2d(final_hidden_ref_all, D);
        tensor_2d<bf16>layer_shared_expert_output_after_sigmoid_2d(layer_shared_expert_output_after_sigmoid, D);
        tensor_2d<bf16>layer_shared_expert_output_before_sigmoid_2d(layer_shared_expert_output_before_sigmoid, D);
        tensor_2d<bf16>layer_post_attention_layernorm_2d(layer_post_attention_layernorm, D);
        tensor_2d<int32_t> layer_selected_experts_2d(layer_selected_experts, 8);
        tensor_2d<bf16>layer_share_down_2d(layer_share_down, D);
        tensor_2d<bf16>layer_share_glu_act_2d(layer_share_glu_act, 512);
        tensor_2d<bf16> layer_share_gate_2d(layer_share_gate, 512);
        tensor_2d<bf16> layer_share_up_2d(layer_share_up, 512);
        tensor_2d<bf16>layer_mlp_output_2d(layer_mlp_output, D);
        tensor_2d<bf16> layer_activated_2d(layer_activated,  8*512);// times 8 for 8 expert
        tensor_2d<bf16> layer_down_2d(layer_down, 8*D);


        
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            //std::cout << "Test token " << std::to_string(i) <<  "  " << std::endl;
            time_utils::time_point start = time_utils::now();


            memcpy(inputs[i].data(), layer_input_ref_2d[i].data(), D*sizeof(bf16));
            // debug:   
            // //TODO: FIXME: REMOE IT LATER AFTER DEBUG
            // memcpy( layer.rms_weights_buffer.data()+D, 
            // layer_post_attention_layernorm_2d[i].data(),
            // D*sizeof(bf16));    



            decode_output = layer(inputs[i]);
            time_utils::time_point end = time_utils::now();
            total_time.first += time_utils::duration_ms(start, end).first;
            memcpy(outputs[i].data(), decode_output.data(), D * sizeof(bf16));

            

            // for(int k = 0; k < 16; k++){
            //      std::cout << std::fixed << std::setprecision(8) << *(outputs[i].data()+k) << std::endl;
            // }
            
            // {
            //     // receive 4 expert's glu, aka 4x 512
            //     // however, they are in the order of 128 interleaved, 
                

            //     buffer<bf16> ref_compare_buf;

            //     buffer<bf16> reorder_glu_expert;
            //     reorder_glu_expert.resize(4* 512);
            //     ref_compare_buf.resize(4*512);
            //     //NOTE: the  exp_glu_res has data format in the following order of
            //     // exp0-glu[0:127] exp1-glu[0:127] exp2-glu[0:127] exp3-glu[0:127],
            //     // exp0-glu[128:255]...... fp;;pwomg 
                
            //     for(int chunk = 0; chunk < 512; chunk += 256){
            //         for(int exp = 0; exp< 4; exp++){
            //             memcpy( reorder_glu_expert.data() + exp*512 + chunk,
            //                 outputs[i].data() + chunk*4 + exp*256,  256*sizeof(bf16)  );
            //         }
            //     }

            //     memcpy(
            //         ref_compare_buf.data(), 
            //         layer_activated_2d[i].data() + 4*512,
            //         sizeof(bf16)* 4* 512
            //     );

            //     all_metrics[i] = get_error_metrics(reorder_glu_expert, ref_compare_buf);
                

            // }

            // {
            //     buffer<bf16> ref_exp_down;
            //     ref_exp_down.resize(D);

            //     memcpy(
            //         ref_exp_down.data(),  layer_down_2d[i].data() + sizeof(bf16)*0, sizeof(bf16)*D
            //     );




            //     all_metrics[i] = get_error_metrics(outputs[i], ref_exp_down);


            //     for(int k = 0; k < D; k++){
            //         std::cout << "at k " << k <<" reference is " << *(ref_exp_down.data() + k) << " output is " << *(outputs[i].data() + k) << std::endl;
            //     }

            // }



             //all_metrics[i] = get_error_metrics(outputs[i], layer_post_attention_layernorm_2d[i]);
            //all_metrics[i] = get_error_metrics(outputs[i], layer_mlp_output_2d[i]);
                //all_metrics[i] = get_error_metrics(outputs[i], layer_shared_expert_output_after_sigmoid_2d[i]);
             //all_metrics[i] = get_error_metrics(outputs[i], layer_share_down_2d[i]);
            // {
     
            //     buffer<bf16> intermediate_out;
            //     intermediate_out.resize(512);
            //     memcpy(intermediate_out.data(), outputs[i].data(), 512*sizeof(bf16));
            //     //all_metrics[i] = get_error_metrics(intermediate_out, layer_share_up_2d[i]);                 
            //     // all_metrics[i] = get_error_metrics(intermediate_out, layer_share_gate_2d[i]); 
            //     all_metrics[i] = get_error_metrics(intermediate_out, layer_share_glu_act_2d[i]); 
            // }
                         
            //all_metrics[i] = get_error_metrics(outputs[i], layer_shared_expert_output_before_sigmoid_2d[i]);
             all_metrics[i] = get_error_metrics(outputs[i], final_hidden_ref_all_2d[i]);
            
            
            // header_print("info", "post_layernorm_out Out:");
            // // utils::print_matrix(post_layernorm_out, 256);
            // for(int i = 0; i < 20; i++){
            //     std::cout << std::fixed << std::setprecision(4) << post_layernorm_out[i] << std::endl;
            // }
            // header_print("info", "ref post_layernorm_out:");
            // utils::print_matrix(post_layernorm_ref_all_2d[i], 256);
            // auto e = get_error_metrics(post_layernorm_out, post_layernorm_ref_all_2d[i]);
            // header_print("info", "post_layernorm_out Cosine similarity: " << e.CosineSimilarity);


            
            
            // header_print("info", "shared_expert_output Out:");
            // utils::print_matrix(outputs[i], 32);
            // header_print("info", "ref Out:");
            // utils::print_matrix(shared_expert_final_output_ref_all_2d[i], 32);
            // auto e2 = get_error_metrics(outputs[i], shared_expert_final_output_ref_all_2d[i]);
            // header_print("info", "shared_expert_output Cosine similarity: " << e2.CosineSimilarity);
            // header_print("info", "shared_expert_output L2 Error" << e2.RelativeL2);

            // // it has 2048 bf16 elements, I want to write into a .bin file
            // bf16* shared_expert_output_ptr = outputs[i];
            // {
            //     std::string filename = "shared_expert_output_" + std::to_string(i) + ".bin";
            //     std::ofstream file(filename, std::ios::binary);
            //     file.write(reinterpret_cast<const char*>(shared_expert_output_ptr), 2048 * sizeof(bf16));
            // }

        //     header_print("info", "final_x_out:");
        //     utils::print_matrix(outputs[i], 256);
        //     header_print("info", "ref final x out:");
        //     utils::print_matrix(final_hidden_ref_all_2d[i], 256);
        //     auto e_final =  get_error_metrics(outputs[i], final_hidden_ref_all_2d[i]);
        //     header_print("info", "final Cosine similarity: " <<  e_final.CosineSimilarity);
        //     header_print("info", "final L2 similarity: " <<  e_final.RelativeL2);
        //    header_print("info", "final L1 similarity: " <<  e_final.RelativeL1);
            // for(int k = 0; k < 8; k++){
            //     std::cout << std::fixed << std::setprecision(4) <<outputs[i][k] << std::endl;
            // }
            // header_print("info", "final_x_out:");
            // utils::print_matrix(outputs[i], 256);
            // header_print("info", "ref final x out:");
            // utils::print_matrix(final_hidden_ref_all_2d[i], 256);
            // auto e_final =  get_error_metrics(outputs[i], final_hidden_ref_all_2d[i]);
            // header_print("info", "final Cosine similarity: " <<  e_final.CosineSimilarity);
            // header_print("info", "final L2 similarity: " <<  e_final.RelativeL2);

        }


        header_print("info", "Decoding layer run completed.");
        header_print("info", "Latency:" + std::to_string(total_time.first / (ids.size() > max_len? max_len : ids.size())) + " " + total_time.second);
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            std::cout << "\tStep " << i << " Cosim:" << std::fixed << std::setprecision(4) << all_metrics[i].CosineSimilarity << std::endl;
            std::cout << "\tStep " << i << " L2:" << all_metrics[i].RelativeL2 << std::endl;
        }

    

        int k_idx = max_len - 1;
        int v_idx = max_len - 1;
        buffer<bf16> k_cache = layer.get_k_cache(k_idx);
        buffer<bf16> v_cache = layer.get_v_cache(v_idx);

        // buffer<bf16> ref_k_cache;
        // buffer<bf16> ref_v_cache;
        // references->load_weights(ref_k_cache, "k_embed");
        // references->load_weights(ref_v_cache, "v_proj");
        // tensor_2d<bf16> ref_k_cache_tensor(ref_k_cache, DK);
        // tensor_2d<bf16> ref_v_cache_tensor(ref_v_cache, DV);

        // header_print("info", "Comparing K cache...");
        // header_print("info", "K cache:");
        // utils::print_matrix(k_cache, DH);
        // header_print("info", "ref K cache:");
        // utils::print_matrix(ref_k_cache_tensor[k_idx], DH);
        // print_error_metrics(get_error_metrics(k_cache, ref_k_cache_tensor[k_idx]), "K Cache Error Metrics");
        
        // {
        //     tensor_2d<bf16> k_cache_head_tensor(k_cache, DH);
        //     tensor_2d<bf16> ref_k_cache_head_tensor(ref_k_cache_tensor[k_idx], DH);
        //     for (int i = 0; i < lm_config.num_key_value_heads; i++) {
        //         print_error_metrics(get_error_metrics(k_cache_head_tensor[i], ref_k_cache_head_tensor[i]),
        //                                             "K Cache Head " + std::to_string(i) + " Error Metrics");
        //     }
        // }
        // header_print("info", "Comparing V cache...");
        // header_print("info", "V cache:");
        // utils::print_matrix(v_cache, DH);
        // header_print("info", "ref V cache:");
        // utils::print_matrix(ref_v_cache_tensor[v_idx], DH);
        // print_error_metrics(get_error_metrics(v_cache, ref_v_cache_tensor[v_idx]), "V Cache Error Metrics");
        // {
        //     tensor_2d<bf16> v_cache_head_tensor(v_cache, DH);
        //     tensor_2d<bf16> ref_v_cache_head_tensor(ref_v_cache_tensor[v_idx], DH);
        //     for (int i = 0; i < lm_config.num_key_value_heads; i++) {
        //         print_error_metrics(get_error_metrics(v_cache_head_tensor[i], ref_v_cache_head_tensor[i]),
        //                                             "V Cache Head " + std::to_string(i) + " Error Metrics");
        //     }
        // }

    }
    if (test_linear_layer){



        int layer_idx = 30;        
        // std::string reference_path = model_list::get_detail_reference_path(model, true);

        std::string reference_path = "/path/to/models/Qwen3.6-35B-A3B-NPU2/activations.safetensors";        
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
      std::cout << "loading reference from  " << reference_path << " " << std::endl;
        buffer<bf16> ref_embed;
        references->load_weights(ref_embed, "embeddings");

        buffer<int32_t> ids;
        references->load_weights(ids, "input_ids");

        header_print("info", "input ids:");
        utils::print_matrix(ids, ids.size());

        
        buffer<bf16> embeddings(ref_embed.size());
        size_t vocab_size = embed.size() / D;
        assert(embed.size() % D == 0 && "embed size is not a multiple of D");
        assert(ref_embed.size() == (size_t)ids.size() * D && "ref_embed size does not match ids.size() * D");

        for (int i = 0; i < ids.size(); i++) {
            int32_t token_id = ids[i];
            assert(token_id >= 0 && (size_t)token_id < vocab_size &&
                "token_id out of embedding table bounds");
            memcpy(embeddings.data() + i * D,
                embed.data() + token_id * D,
                D * sizeof(bf16)
            );
        }
        print_error_metrics(get_error_metrics(embeddings, ref_embed), "Model Embedding Error Metrics");

        header_print("info", "Embedding test completed.");

        tensor_2d<bf16> inputs(ref_embed, D);
        buffer<bf16> output_buffer(ref_embed.size());
        tensor_2d<bf16> outputs(output_buffer, D);
        buffer<bf16> layer_shared_expert_output_before_sigmoid;        
        buffer<bf16> layer_shared_expert_output_after_sigmoid;
        buffer<bf16> layer_input_ref;
        buffer<bf16> final_hidden_ref_all;
        buffer<bf16> layer_post_attention_layernorm_ref;
        buffer<bf16> mixed_qkv_ref;
        buffer<bf16> layer_residual_1;
        references->load_weights(layer_input_ref, "layer_" + std::to_string(layer_idx) + "_input");
        references->load_weights(final_hidden_ref_all, "layer_" + std::to_string(layer_idx) + "_output");
        references->load_weights(layer_shared_expert_output_before_sigmoid, "layer_"+ std::to_string(layer_idx)+"_shared_expert_output_before_sigmoid");        
        references->load_weights(layer_shared_expert_output_after_sigmoid, "layer_"+ std::to_string(layer_idx)+"_shared_expert_output_after_sigmoid");
        references->load_weights(layer_post_attention_layernorm_ref, "layer_"+ std::to_string(layer_idx)+"_post_attention_layernorm");
        references->load_weights(mixed_qkv_ref, "layer_"+ std::to_string(layer_idx) + "_mixed_qkv");
        references->load_weights(layer_residual_1, "layer_"+ std::to_string(layer_idx) + "_residual_1");


        
        tensor_2d<bf16> layer_input_ref_2d(layer_input_ref, D);
        tensor_2d<bf16> final_hidden_ref_all_2d(final_hidden_ref_all, D);
        tensor_2d<bf16>layer_shared_expert_output_after_sigmoid_2d(layer_shared_expert_output_after_sigmoid, D);
        tensor_2d<bf16>layer_shared_expert_output_before_sigmoid_2d(layer_shared_expert_output_before_sigmoid, D);
        tensor_2d<bf16> layer_post_attention_layernorm_2d(layer_post_attention_layernorm_ref, D);
        tensor_2d<bf16> mixed_qkv_2d(mixed_qkv_ref, 8192);
        tensor_2d<bf16> layer_residual_1_2d(layer_residual_1, D);
        linear_decoding_layer linear_layer(model, lm_config, layer_app_mgr, D);
        linear_layer.load_weights(*q4nx_model, layer_idx);

        buffer<bf16> decode_output;
        std::vector<error_metrics> all_metrics(ids.size());
        header_print("info", "Running decoding layer...");
        time_utils::time_with_unit total_time = {0.0f, "ms"};

        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            //std::cout << "Test token " << std::to_string(i) << "  " << std::endl;
            time_utils::time_point start = time_utils::now();

            memcpy(inputs[i].data(), layer_input_ref_2d[i].data(), D * sizeof(bf16));

            decode_output = linear_layer(inputs[i]);
            time_utils::time_point end = time_utils::now();
            total_time.first += time_utils::duration_ms(start, end).first;
            memcpy(outputs[i].data(), decode_output.data(), D * sizeof(bf16));

            all_metrics[i] = get_error_metrics(outputs[i], final_hidden_ref_all_2d[i]);

            // header_print("info", "final_x_out:");
            // utils::print_matrix(outputs[i], 256);
            // header_print("info", "ref final x out:");
            // utils::print_matrix(final_hidden_ref_all_2d[i], 256);




            // // comparing the Conv1d cache result
            // if(i >= 3){

            //     buffer<bf16> conv_cache = linear_layer.get_conv_cache();
            //     assert(conv_cache.size() == 3*8192);

            //     // mixed_qkv_2d[ i-3].data()
            //     // conv_cache .data()

            //     buffer<bf16> ref_cache_size(mixed_qkv_2d[ i-2].data(),  3* 8192 );

            //     auto e_conv_cache = get_error_metrics(conv_cache, ref_cache_size);
            // header_print("info", "e_conv_cache Cosine similarity: " << e_conv_cache.CosineSimilarity);
            // header_print("info", "e_conv_cache L2 similarity: " << e_conv_cache.RelativeL2);
            // header_print("info", "e_conv_cache L1 similarity: " << e_conv_cache.RelativeL1);
            //     // buffer<bf16> v_out = buffer<bf16>(conv_cache.data() + 8192 * (3 - time_steps), 8192);
            //     // buffer<bf16> v_ref_all;
            //     // references->load_weights(v_ref_all, "qkv_proj");
            //     // buffer<bf16> v_ref = buffer<bf16>(v_ref_all.data(), 8192);
            //     // print_error_metrics(get_error_metrics(v_out, v_ref), "linear v " );
            //     // header_print("info", "V out:");
            //     // utils::print_matrix(v_out, 128);
            //     // header_print("info", "V Ref:");
            //     // utils::print_matrix(v_ref, 128);                      
            // }
      



        }

        header_print("info", "Decoding layer run completed.");
        header_print("info", "Latency:" + std::to_string(total_time.first / (ids.size() > max_len? max_len : ids.size())) + " " + total_time.second);
        for (uint32_t i = 0; i < (ids.size() > max_len? max_len : ids.size()); i++) {
            std::cout << "\tStep " << i << " Cosim:" << std::fixed << std::setprecision(4) << all_metrics[i].CosineSimilarity << std::endl;
            std::cout << "\tStep " << i << " L2:" << std::fixed << std::setprecision(4) << all_metrics[i].RelativeL2 << std::endl;
        }

        // for (int i = 0; i < time_steps; i++) {
        //     lout = linear_layer(inputs[i]);
        // }
        
        // buffer<bf16> l_tb = buffer<bf16>(tb.data(), 32);
        // buffer<bf16> l1_tb = buffer<bf16>(tb.data() + 32, 32);


        // buffer<float> alpha;

        // references->load_weights(alpha, "alpha");
        // buffer<bf16> alpha_bf16(alpha.size());
        // tensor_2d<bf16> alpha_tensor(alpha_bf16, 32);
        // for (int i = 0; i < 32; i++) {
        //     alpha_bf16[i] = static_cast<bf16>(alpha[i]);
        // }
        // header_print("info", "Alpha ref:");
        // utils::print_matrix(alpha_tensor[time_steps - 1], 8);
        // header_print("info", "Alpha:");
        // utils::print_matrix(l_tb, 8);
        // {
        //     auto e = get_error_metrics(l_tb, alpha_tensor[time_steps - 1]);
        //     header_print("info", "Alpha L2: " << e.RelativeL2);
        // }

        // references->load_weights(alpha, "beta");
        // for (int i = 0; i < alpha.size(); i++) {
        //     alpha_bf16[i] = static_cast<bf16>(alpha[i]);
        // }


        // header_print("info", "Beta ref:");
        // utils::print_matrix(alpha_tensor[time_steps - 1], 8);
        // header_print("info", "Beta:");
        // utils::print_matrix(l1_tb, 8);
        // {
        //     auto e = get_error_metrics(l1_tb, alpha_tensor[time_steps - 1]);
        //     header_print("info", "Beta L2: " << e.RelativeL2);
        // }


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

     
        // buffer<bf16> conv_cache = linear_layer.get_conv_cache();
        // header_print("info", "Conv cache:");
        // utils::print_matrix(conv_cache, 8192);
        // // buffer<bf16> s_cache = linear_layer.get_S_cache(31);
        // // header_print("info", "S:");
        // // utils::print_matrix(s_cache, 128);


    }
    return 0;
}
