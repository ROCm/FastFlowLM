// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_image_embed.cpp
/// \brief qwen3_5_omni image embedder implementation — simple stub.
/// \author FastFlowLM Team
#include "qwen3_5_omni_image_embed.hpp"
#include "utils/error_measure.hpp"
#include "npu_sequences/mmRuntimeSequence.hpp"
#include "vision/conv3dPatchEmbedding.hpp" // same with 3vl
#include "npu_sequences/image_attention_sequence.hpp"
#include "image_seq_gen.hpp"
#include "vision/rot_pos_emb.hpp"  // same with 3vl
#include "vision/pos_embed_interpolate.hpp"
#include "vision/vision_prefill_helper.hpp"
#include "vision/norm.hpp"
#define DEBUG_PRINT_ENCODE_TIME_DETAIL (DEBUG_LEVEL >= 1)
#define DEBUG_PRINT_ENCODE_ERROR_METRICS (DEBUG_LEVEL >= 1)

qwen3_5_omni_image_embed::qwen3_5_omni_image_embed(LM_Config config, npu_xclbin_manager *npu_instance)
    : config(config), npu(npu_instance)
{
    desc.build(config);

    padded_vision_hidden_size = round_up_to_multiple(
        desc.hidden_size, desc.vision_mm_tile_k
    );
    padded_vision_mlp_intermediate_size = round_up_to_multiple(
        desc.intermediate_size, desc.vision_mm_tile_k
    );
    padded_merger_hidden_size = round_up_to_multiple(
        desc.merge_hidden_size, desc.vision_mm_tile_k
    );
    padded_vision_out_hidden_size = round_up_to_multiple(
        desc.out_hidden_size, desc.vision_mm_tile_k
    );
    seq_len_pad_requirement_for_MM = MM_ROW_SIZE* desc.vision_mm_tile_m;

    this->fla = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_attn.xclbin"));
    this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));

    this->flash_attention_app = this->fla->create_app();
    this->qkv_projection_app = this->proj->create_app();
    this->o_projection_app = this->proj->create_app();
    this->mlp_fc1_app = this->proj->create_app();
    this->mlp_fc2_app = this->proj->create_app();
    this->merger_fc1_app = this->proj->create_app();
    this->merger_fc2_app = this->proj->create_app();

    // resize the vectors to hold all layers weights
    norm_1_weight.resize(desc.depth);
    norm_1_bias.resize(desc.depth);
    norm_2_weight.resize(desc.depth);
    norm_2_bias.resize(desc.depth);
    attn_qkv_weight.resize(desc.depth);
    o_proj_weight.resize(desc.depth);
    mlp_fc1_weight.resize(desc.depth);
    mlp_fc2_weight.resize(desc.depth);
}

qwen3_5_omni_image_embed::~qwen3_5_omni_image_embed() = default;

buffer<bf16> qwen3_5_omni_image_embed::forward(qwen3_5_omni_image_payload_t& payload)
{
    // TODO: for now

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        SafeTensors reference_tensor(utils::path_join(config.model_path, "vision_data_debug.safetensors"));
        std::cout << "debugging qwen3.5 omni vision" << std::endl;
    #endif

    std::vector<std::vector<int32_t>> image_grid_thw;
    std::vector<float> cos_emb;
    std::vector<float> sin_emb;
    int seq_len = 0;
    int seq_len_of_last_image = 0;
    for(int i = 0; i < payload.num_images; i++ ){
        auto grid_h = payload.image_grid_h_w[i][0];
        auto grid_w = payload.image_grid_h_w[i][1];
        int t = grid_h * grid_w;
        seq_len += t;
        seq_len_of_last_image = t;
        image_grid_thw.push_back({1, grid_h, grid_w});

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "grid_h " << grid_h << " grid_w " << grid_w <<std::endl;
        #endif
    }

    int seq_len_padded = round_up_to_multiple(seq_len_of_last_image, vision_L_padded_requirement_for_attention) - seq_len_of_last_image;
    seq_len_padded += seq_len;
    seq_len_padded = round_up_to_multiple(seq_len_padded, seq_len_pad_requirement_for_MM);

    int merger_seq_len = (seq_len*desc.hidden_size) / (desc.merge_hidden_size); //NOTE: use the unpadded version for calculation
    int merger_seq_len_padded = round_up_to_multiple(merger_seq_len, seq_len_pad_requirement_for_MM);

    hidden_state.resize((size_t)seq_len_padded * padded_vision_hidden_size);
    memset(hidden_state.data(), 0, hidden_state.size() * sizeof(bf16)); // clear the buffer to avoid reading uninitialized data in case of partial tiles

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        buffer<bf16> inital_input;
        reference_tensor.load_weights(inital_input, "initial_hidden_states");
        print_error_metrics<bf16, bf16>(
        payload._processed_pixel_values.data(), inital_input.data(),
        1,
        1, inital_input.size(),
        1, inital_input.size()
        );

        // for debug to avoid error accumulation purpose
        memcpy(
            payload._processed_pixel_values.data(),
            inital_input.data(),
            inital_input.size()*sizeof(bf16)
        );
    #endif

    {
        bool ADD_BIAS= true;

        // pre-generate the mm runtime sequence
        generate_mm_sequence<bf16, bf16>(*this->qkv_projection_app.seq(),
                seq_len_padded, padded_vision_hidden_size,    3*padded_vision_hidden_size,
                desc.vision_mm_tile_m,desc.vision_mm_tile_k,desc.vision_mm_tile_n,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                ENABLE_QKV_REORDER, desc.head_dim
        );

        generate_mm_sequence<bf16, bf16>(*this->o_projection_app.seq(),
                seq_len_padded, padded_vision_hidden_size,  padded_vision_hidden_size,
                desc.vision_mm_tile_m,desc.vision_mm_tile_k,desc.vision_mm_tile_n,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );
        generate_mm_sequence<bf16, bf16>(*this->mlp_fc1_app.seq(),
                seq_len_padded, padded_vision_hidden_size,  padded_vision_mlp_intermediate_size,
                desc.vision_mm_tile_m,desc.vision_mm_tile_k,desc.vision_mm_tile_n,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 1,
                false,0
        );

        generate_mm_sequence<bf16, bf16>(*this->mlp_fc2_app.seq(),
                seq_len_padded,  padded_vision_mlp_intermediate_size,  padded_vision_hidden_size,
                desc.vision_mm_tile_m,desc.vision_mm_tile_k,desc.vision_mm_tile_n,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );

        generate_mm_sequence<bf16, bf16>(*this->merger_fc1_app.seq(),
                merger_seq_len_padded, padded_merger_hidden_size,  padded_merger_hidden_size,
                desc.vision_mm_tile_m,desc.vision_mm_tile_k,desc.vision_mm_tile_n,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 1, // gelu
                false,0
        );
        generate_mm_sequence<bf16, bf16>(*this->merger_fc2_app.seq(),
                merger_seq_len_padded, padded_merger_hidden_size,  padded_vision_out_hidden_size,
                desc.vision_mm_tile_m,desc.vision_mm_tile_k,desc.vision_mm_tile_n,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );

        gen_mha_vision_attention(
            this->flash_attention_app.seq(),
            image_grid_thw,
            vision_L_padded_requirement_for_attention,
            vision_S_padded_requirement_for_attention,
            vision_num_of_columns,
            vision_num_of_rows,
            vision_CU_mode,
            vision_LQ_per_CT,
            vision_LK_per_CT,
            vision_LQ_internal,
            vision_LK_internal,
            ENABLE_QKV_REORDER,

            this
        );
    }

    {
        // initialize the remainig buffer
        mm_qkv_input = qkv_projection_app.create_bo_buffer<bf16>((size_t)seq_len_padded * padded_vision_hidden_size);
        mm_qkv_output_mha_input = qkv_projection_app.create_bo_buffer<bf16>((size_t)seq_len_padded*padded_vision_hidden_size*3);
        mha_qkv_output_mm_o_input = o_projection_app.create_bo_buffer<bf16>((size_t)seq_len_padded * padded_vision_hidden_size);
        mm_o_output_fc1_proj_input = o_projection_app.create_bo_buffer<bf16>((size_t)seq_len_padded * padded_vision_hidden_size);
        mm_fc1_output_fc2_input = mlp_fc1_app.create_bo_buffer<bf16>((size_t)seq_len_padded * padded_vision_mlp_intermediate_size);
        mm_fc2_output = mlp_fc2_app.create_bo_buffer<bf16>((size_t)seq_len_padded * padded_vision_hidden_size);

        merger_fc1_input = merger_fc1_app.create_bo_buffer<bf16>((size_t)merger_seq_len_padded*padded_merger_hidden_size);
        merger_fc1_output_fc2_input = merger_fc1_app.create_bo_buffer<bf16>( (size_t)merger_seq_len_padded*padded_merger_hidden_size);
        merger_fc2_output = merger_fc2_app.create_bo_buffer<bf16>( (size_t)merger_seq_len_padded*padded_vision_out_hidden_size);

        memset(mm_qkv_input.data(), 0, mm_qkv_input.size() * sizeof(bf16));
        mm_qkv_input.sync_to_device();
        memset(mm_o_output_fc1_proj_input.data(), 0, mm_o_output_fc1_proj_input.size() * sizeof(bf16));
        mm_o_output_fc1_proj_input.sync_to_device();
        memset(mm_fc1_output_fc2_input.data(), 0, mm_fc1_output_fc2_input.size() * sizeof(bf16));
        mm_fc1_output_fc2_input.sync_to_device();
        memset(mm_fc2_output.data(), 0, mm_fc2_output.size() * sizeof(bf16));
        mm_fc2_output.sync_to_device();

        memset(mm_qkv_output_mha_input.data(), 0, mm_qkv_output_mha_input.size() * sizeof(bf16));
        mm_qkv_output_mha_input.sync_to_device();
        memset(mha_qkv_output_mm_o_input.data(), 0, mha_qkv_output_mm_o_input.size() * sizeof(bf16));
        mha_qkv_output_mm_o_input.sync_to_device();
    }

    DEBUG_BLOCK(
        1,
        std::cout << "seq_len: " << seq_len << std::endl;
        std::cout << "seq_len_padded: " << seq_len_padded << std::endl;
        std::cout << "merger_seq_len: " << merger_seq_len << std::endl;
        std::cout << "merger_seq_len_padded: " << merger_seq_len_padded << std::endl;
    )

    std::vector<bf16> patch_embed = conv3d_patch_embed_avx512_max_performance(
        payload._processed_pixel_values.data(),
        patch_embed_proj_weight.data(),
        patch_embed_proj_bias.data(),
        seq_len,
        3, // r, g, b
        desc.hidden_size,
        desc.temporal_patch_size,
        desc.patch_size
    );

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error after patch embedding comparison:" << std::endl;
        buffer<bf16> after_patch_embed;
        reference_tensor.load_weights(
            after_patch_embed,
            "after_patch_embed"
        );
        print_error_metrics<bf16, bf16>(
            patch_embed.data(), after_patch_embed.data(),
                1,
                seq_len, desc.hidden_size,
                seq_len, desc.hidden_size
        );

    #endif
    std::vector<bf16>  pos_embeds =  fast_pos_embed_interpolate(
        image_grid_thw,
        pos_embed_weight.data(),
        std::sqrt(desc.num_position_embeddings),
        desc.hidden_size,
        desc.spatial_merge_size
    );

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        std::cout << "Error after pos embed comparison:" << std::endl;
        buffer<bf16> after_pos_embed;
        reference_tensor.load_weights(
            after_pos_embed,
            "pos_embeds"
        );
        print_error_metrics<bf16, bf16>(
            pos_embeds.data(), after_pos_embed.data(),
            1,
            seq_len, desc.hidden_size,
            seq_len, desc.hidden_size
        );
    #endif

    simd_add(
        patch_embed.data(), pos_embeds.data(),
        hidden_state.data(), (size_t)seq_len *desc.hidden_size
    );

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        std::cout << "Error after hidden_states_plus_pos_embeds:" << std::endl;
        buffer<bf16> after_pos_embed_plus_patch_embed;
        reference_tensor.load_weights(
            after_pos_embed_plus_patch_embed,
            "hidden_states_plus_pos_embeds"
        );
        print_error_metrics<bf16, bf16>(
            hidden_state.data(), after_pos_embed_plus_patch_embed.data(),
            1,
            seq_len, desc.hidden_size,
            seq_len_padded, desc.hidden_size
        );
    #endif

    std::vector<float> rotary_pos_emb = rot_pos_emb_float(
        image_grid_thw,
        desc.spatial_merge_size,           // spatial_merge_size = 2
        desc.head_dim/ 2           // rot_dim = 64 / 2 = 32
    );
   cos_sine_pose_emb(rotary_pos_emb, cos_emb, sin_emb, desc.head_dim);
   #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        buffer<float> position_embeddings_cos_ptr;
        reference_tensor.load_weights(
            position_embeddings_cos_ptr,
            "rope_cos"
        );
        buffer<float> position_embeddings_sin_ptr;
        reference_tensor.load_weights(
            position_embeddings_sin_ptr,
            "rope_sin"
        );
        std::cout << "Error after position_embeddings_cos comparison:" << std::endl;
        print_error_metrics<float, float>(
            cos_emb.data(), position_embeddings_cos_ptr.data(),
            1,
            seq_len, desc.head_dim,
            seq_len, desc.head_dim
        );

        std::cout << "Error after position_embeddings_sin comparison:" << std::endl;
        print_error_metrics<float, float>(
            sin_emb.data(), position_embeddings_sin_ptr.data(),
            1,
            seq_len, desc.head_dim,
            seq_len, desc.head_dim
        );

    #endif

    residual_buffer.resize(hidden_state.size());
    memset(residual_buffer.data(), 0, sizeof(bf16)*residual_buffer.size());

    for(int layer_idx = 0; layer_idx < desc.depth; layer_idx++){

        // if(layer_idx != 0){
        //     // use previous reference to avoid accumulation error
        //     buffer<bf16> after_mlp_residual_add_ptr;
        //     reference_tensor.load_weights(
        //         after_mlp_residual_add_ptr,
        //         "layer_" + std::to_string(layer_idx-1)
        //     );
        //     assert(padded_vision_hidden_size == desc.hidden_size);
        //     memcpy(hidden_state.data(), after_mlp_residual_add_ptr.data(), sizeof(bf16)* seq_len* padded_vision_hidden_size);

        // }

        memcpy(residual_buffer.data(), hidden_state.data(), sizeof(bf16)* seq_len* padded_vision_hidden_size);

        // first, do the norm_1 output
        auto time_start_layernorm1 = std::chrono::high_resolution_clock::now();
        layernorm_parallel(
            seq_len,
            desc.hidden_size,
            padded_vision_hidden_size,
            hidden_state.data(),
            norm_1_weight[layer_idx].data(),
            norm_1_bias[layer_idx].data(),
            desc.layer_norm_eps,
            mm_qkv_input.data()  // Use separate output buffer!
        );
        auto time_end_layernorm1 = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_layernorm1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_layernorm1 - time_start_layernorm1).count();
            std::cout << "Layer norm 1 time for layer " << layer_idx << ": " << duration_layernorm1 << " ms" << std::endl;
        #endif

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        //     // compare with the reference
        //     std::cout << "Error after layernorm 1 comparison:" << std::endl;
        //     buffer<bf16> after_layernorm1;
        //     reference_tensor.load_weights(
        //         after_layernorm1,
        //           "vision_block_normed_1_" + std::to_string(layer_idx)
        //     );
        //     std::cout << "Error after norm1 layer " << layer_idx << " comparison:" << std::endl;
        //     print_error_metrics<bf16,bf16>(
        //         mm_qkv_input.data(), after_layernorm1.data(),  // Compare mm_qkv buffer which has norm output
        //         1,
        //         seq_len, desc.hidden_size,
        //         seq_len_padded, padded_vision_hidden_size
        //     );

        // #endif

        auto time_start_qkv_projection = std::chrono::high_resolution_clock::now();
        mm_qkv_input.sync_to_device();
        attn_qkv_weight[layer_idx].sync_to_device();
        qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
        mm_qkv_output_mha_input.sync_from_device();
        auto time_end_qkv_projection = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_qkv_projection = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_qkv_projection - time_start_qkv_projection).count();
            std::cout << "QKV projection time for layer " << layer_idx << ": " << duration_qkv_projection << " ms" << std::endl;
        #endif
        auto time_start_add_qkv_bias = std::chrono::high_resolution_clock::now();

        auto time_end_add_qkv_bias = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_add_qkv_bias = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_add_qkv_bias - time_start_add_qkv_bias).count();
            std::cout << "Add QKV bias time for layer " << layer_idx << ": " << duration_add_qkv_bias << " ms" << std::endl;
        #endif

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        //     // compare with the reference

        //     buffer<bf16> after_qkv_projection;
        //     reference_tensor.load_weights(
        //         after_qkv_projection,
        //         "vision_attention_qkv_mm_with_bias_" + std::to_string(layer_idx)
        //     );
        //     std::cout << "Error after qkv mm layer " << layer_idx << " comparison:" << std::endl;
        //     print_error_metrics<bf16, bf16>(
        //         mm_qkv_output_mha_input.data(), after_qkv_projection.data(),
        //         1,
        //         seq_len, 3*desc.hidden_size,
        //         seq_len_padded, 3*padded_vision_hidden_size
        //     );

        // #endif

        // now apply rope to q, k
        auto time_start_rotary_pos_emb_application = std::chrono::high_resolution_clock::now();

            apply_rotary_pos_emb_vision_inplace(
                mm_qkv_output_mha_input.data(),
                cos_emb.data(),
                sin_emb.data(),
                seq_len,
                seq_len_padded,
                desc.hidden_size,
                3*padded_vision_hidden_size,
                desc.num_heads
            );

        auto time_end_rotary_pos_emb_application = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_rotary_pos_emb_application = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_rotary_pos_emb_application - time_start_rotary_pos_emb_application).count();
            std::cout << "Rotary position embedding application time for layer " << layer_idx << ": " << duration_rotary_pos_emb_application << " ms" << std::endl;
        #endif

        // now mha
        auto time_start_mha = std::chrono::high_resolution_clock::now();
        mm_qkv_output_mha_input.sync_to_device();
        mha_qkv_output_mm_o_input.sync_to_device();
        flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
        mha_qkv_output_mm_o_input.sync_from_device();
        auto time_end_mha = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mha = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mha - time_start_mha).count();
            std::cout << "MHA time for layer " << layer_idx << ": " << duration_mha << " ms" << std::endl;
        #endif

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        //     // compare with the reference
        //     buffer<bf16> after_attention_output_ptr;
        //     reference_tensor.load_weights(
        //         after_attention_output_ptr,
        //         "vision_attention_output" + std::to_string(layer_idx)
        //     );
        //     std::cout << "Error after attention output layer " << layer_idx << " comparison:" << std::endl;
        //     print_error_metrics<bf16, bf16>(
        //         mha_qkv_output_mm_o_input.data(), after_attention_output_ptr.data(),
        //         1,
        //         seq_len, desc.hidden_size,
        //         seq_len_padded, padded_vision_hidden_size
        //     );
        // #endif

        // now the o projection
        auto time_start_o_projection = std::chrono::high_resolution_clock::now();
        mha_qkv_output_mm_o_input.sync_to_device();
        o_proj_weight[layer_idx].sync_to_device();
        o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
        mm_o_output_fc1_proj_input.sync_from_device();
        auto time_end_o_projection = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_o_projection = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_o_projection - time_start_o_projection).count();
            std::cout << "O projection time for layer " << layer_idx << ": " << duration_o_projection << " ms" << std::endl;
        #endif
        // add the bias
        auto time_start_add_o_proj_bias = std::chrono::high_resolution_clock::now();

        auto time_end_add_o_proj_bias = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_add_o_proj_bias = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_add_o_proj_bias - time_start_add_o_proj_bias).count();
            std::cout << "Add o projection bias time for layer " << layer_idx << ": " << duration_add_o_proj_bias << " ms" << std::endl;
        #endif

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        //     // compare with the reference
        //     buffer<bf16> after_o_project_ptr;
        //     reference_tensor.load_weights(
        //         after_o_project_ptr,
        //         "vision_attention_output_proj_" + std::to_string(layer_idx)
        //     );
        //     std::cout << "Error after o projection layer " << layer_idx << " comparison:" << std::endl;
        //     print_error_metrics<bf16, bf16>(
        //         mm_o_output_fc1_proj_input.data(), after_o_project_ptr.data(),
        //         1,
        //         seq_len, desc.hidden_size,
        //         seq_len_padded, padded_vision_hidden_size
        //     );
        // #endif

        auto time_start_residual_add_1 = std::chrono::high_resolution_clock::now();
        simd_add(
            mm_o_output_fc1_proj_input.data(),
            residual_buffer.data(),
            hidden_state.data(),
            (size_t)seq_len * padded_vision_hidden_size
        );
        auto time_end_residual_add_1 = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_residual_add_1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_residual_add_1 - time_start_residual_add_1).count();
            std::cout << "Residual add 1 time for layer " << layer_idx << ": " << duration_residual_add_1 << " ms" << std::endl;
        #endif
        memcpy(
            residual_buffer.data(),
            hidden_state.data(),
            sizeof(bf16) * seq_len * padded_vision_hidden_size
        );

        // norm2
        auto time_start_layernorm2 = std::chrono::high_resolution_clock::now();
        layernorm_parallel(
            seq_len,
            desc.hidden_size,
            padded_vision_hidden_size,
            hidden_state.data(),
            norm_2_weight[layer_idx].data(),
            norm_2_bias[layer_idx].data(),
            desc.layer_norm_eps,
            mm_o_output_fc1_proj_input.data()
        );
        auto time_end_layernorm2 = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_layernorm2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_layernorm2 - time_start_layernorm2).count();
            std::cout << "Layer norm 2 time for layer " << layer_idx << ": " << duration_layernorm2 << " ms" << std::endl;
        #endif

        // now to fc1
        auto time_start_fc1 = std::chrono::high_resolution_clock::now();
        mm_o_output_fc1_proj_input.sync_to_device();
        mlp_fc1_weight[layer_idx].sync_to_device();
        mlp_fc2_weight[layer_idx].sync_to_device();
        mlp_fc1_app( mm_o_output_fc1_proj_input,  mlp_fc1_weight[layer_idx],   mm_fc1_output_fc2_input );
        // mm_fc1_output_fc2_input.sync_from_device();
        auto time_end_fc1 = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_fc1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_fc1 - time_start_fc1).count();
            std::cout << "MLP FC1 time for layer " << layer_idx << ": " << duration_fc1 << " ms" << std::endl;
        #endif

        // Combined bias addition and GELU activation using optimized AVX-512
        auto time_start_mlp_fc1_gelu = std::chrono::high_resolution_clock::now();

        auto time_end_mlp_fc1_gelu = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc1_gelu = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc1_gelu - time_start_mlp_fc1_gelu).count();
            std::cout << "MLP FC1 GELU time for layer " << layer_idx << ": " << duration_mlp_fc1_gelu << " ms" << std::endl;
        #endif

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        //     // compare with the reference
        //     buffer<bf16> after_mlp_fc1_gelu_ptr;
        //     reference_tensor.load_weights(
        //         after_mlp_fc1_gelu_ptr,
        //         "vision_mlp_activated_output_" + std::to_string(layer_idx)
        //     );
        //     std::cout << "Error after mlp fc1 gelu layer " << layer_idx << " comparison:" << std::endl;
        //     print_error_metrics<bf16, bf16>(
        //         mm_fc1_output_fc2_input.data(), after_mlp_fc1_gelu_ptr.data(),
        //         1,
        //         seq_len, desc.intermediate_size,
        //         seq_len_padded, padded_vision_mlp_intermediate_size
        //     );
        // #endif

        // now do the fc2
        auto time_start_mlp_fc2 = std::chrono::high_resolution_clock::now();
        mlp_fc2_app( mm_fc1_output_fc2_input,  mlp_fc2_weight[layer_idx],   mm_fc2_output );
        mm_fc2_output.sync_from_device();
        auto time_end_mlp_fc2 = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc2 - time_start_mlp_fc2).count();
            std::cout << "MLP FC2 time for layer " << layer_idx << ": " << duration_mlp_fc2 << " ms" << std::endl;
        #endif

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        //     //compare with the reference
        //     buffer<bf16> after_mlp_fc2_ptr;
        //     reference_tensor.load_weights(
        //         after_mlp_fc2_ptr,
        //         "vision_mlp_final_output_" + std::to_string(layer_idx)
        //     );
        //     std::cout << "Error after mlp fc2 layer " << layer_idx << " comparison:" << std::endl;
        //     print_error_metrics<bf16, bf16>(
        //         mm_fc2_output.data(), after_mlp_fc2_ptr.data(),
        //         1,
        //         seq_len, desc.hidden_size,
        //         seq_len_padded, padded_vision_hidden_size
        //     );
        // #endif

        auto time_start_mlp_residual_add = std::chrono::high_resolution_clock::now();
        simd_add(
            mm_fc2_output.data(),
            residual_buffer.data(),
            hidden_state.data(),
            (size_t)seq_len * padded_vision_hidden_size
        );
        auto time_end_mlp_residual_add = std::chrono::high_resolution_clock::now();
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_residual_add = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_residual_add - time_start_mlp_residual_add).count();
            std::cout << "MLP residual add time for layer " << layer_idx << ": " << duration_mlp_residual_add << " ms" << std::endl;
        #endif
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare the final
            buffer<bf16> after_mlp_residual_add_ptr;
            reference_tensor.load_weights(
                after_mlp_residual_add_ptr,
                "layer_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp residual add layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                hidden_state.data(), after_mlp_residual_add_ptr.data(),
                1,
                seq_len, desc.hidden_size,
                seq_len_padded, padded_vision_hidden_size
            );

        #endif
    }

    int spatial_merge_size_sq =desc.spatial_merge_size*desc.spatial_merge_size;
    for(int s = 0; s < seq_len; s++){
        int merged_row = s / spatial_merge_size_sq;
        int sub_idx = s % spatial_merge_size_sq;
        layernorm_high_precision(
            desc.hidden_size,
            hidden_state.data() + s * padded_vision_hidden_size,
            merger_norm_weight.data(),
            merger_norm_bias.data(),
            desc.layer_norm_eps,
            merger_fc1_input.data() + merged_row * padded_merger_hidden_size + sub_idx * desc.hidden_size
        );
    }

    merger_fc1_input.sync_to_device();
    merger_fc1_weight.sync_to_device();
    // merger_fc1_output_fc2_input.sync_to_device();
    merger_fc2_weight.sync_to_device();
    merger_fc1_app( merger_fc1_input,  merger_fc1_weight,   merger_fc1_output_fc2_input );

    //fc2

    merger_fc2_app( merger_fc1_output_fc2_input,  merger_fc2_weight,   merger_fc2_output );
    merger_fc2_output.sync_from_device();

    buffer<bf16> final_res;
    final_res.resize(merger_fc2_output.size());
    if(padded_vision_out_hidden_size == desc.out_hidden_size){
        // simple memcyp
        memcpy(final_res.data(), merger_fc2_output.data(), (size_t)merger_seq_len*padded_vision_out_hidden_size*sizeof(bf16));
    }else{
        // need to use a for loop
        // should not even happened here, so
        std::cerr << "unexpected condition for 3.5omni" <<std::endl;
        exit(-1);
    }

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        buffer<bf16> deepstack_output_ptr_ref;
        reference_tensor.load_weights(
            deepstack_output_ptr_ref,
            "hidden_states_after_merger"
        );
        std::cout << "Error after final patch merger comparison:" << std::endl;
        print_error_metrics<bf16, bf16>(
            final_res.data(), deepstack_output_ptr_ref.data(),
            1,
            merger_seq_len, desc.out_hidden_size,
            merger_seq_len_padded, padded_vision_out_hidden_size
        );

    #endif

    return final_res;
}

void qwen3_5_omni_image_embed::load_weights(SafeTensors &q4nx) {

    q4nx.load_weights(patch_embed_proj_weight, "model.visual.patch_embed.proj.weight");
    q4nx.load_weights(patch_embed_proj_bias,"model.visual.patch_embed.proj.bias");
    q4nx.load_weights(pos_embed_weight,  "model.visual.pos_embed.weight");

    //load each layer weights
    for(int layer_idx = 0; layer_idx < desc.depth; layer_idx++){
        q4nx.load_weights(
            this->norm_1_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm1.weight"
        );
        q4nx.load_weights(
            this->norm_1_bias[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm1.bias"
        );
        q4nx.load_weights(
            this->norm_2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm2.weight"
        );
        q4nx.load_weights(
            this->norm_2_bias[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm2.bias"
        );

        this->attn_qkv_weight[layer_idx] = qkv_projection_app.create_bo_buffer<bf16>(padded_vision_hidden_size*(3*padded_vision_hidden_size) + 3*padded_vision_hidden_size);
        this->o_proj_weight[layer_idx] = o_projection_app.create_bo_buffer<bf16>(padded_vision_hidden_size*(padded_vision_hidden_size) + padded_vision_hidden_size );

        q4nx.load_weights(this->attn_qkv_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.qkv.bias",
            0
        );
        q4nx.load_weights(this->o_proj_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.proj.bias",
            0
        );

        q4nx.load_weights(this->attn_qkv_weight[layer_idx],
             "model.visual.blocks." + std::to_string(layer_idx) + ".attn.qkv.weight",
             sizeof(bf16) * 3*padded_vision_hidden_size
        );
        q4nx.load_weights(this->o_proj_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.proj.weight",
            sizeof(bf16) * padded_vision_hidden_size
        );
        this->attn_qkv_weight[layer_idx].sync_to_device();
        this->o_proj_weight[layer_idx].sync_to_device();

        // mlo
        this->mlp_fc1_weight[layer_idx] = mlp_fc1_app.create_bo_buffer<bf16>(padded_vision_hidden_size*padded_vision_mlp_intermediate_size
             + padded_vision_mlp_intermediate_size);
        this->mlp_fc2_weight[layer_idx] = mlp_fc2_app.create_bo_buffer<bf16>(padded_vision_mlp_intermediate_size*padded_vision_hidden_size
            + padded_vision_hidden_size);

        q4nx.load_weights(this->mlp_fc1_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc1.bias",
            0
        );
        q4nx.load_weights(this->mlp_fc2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc2.bias",
            0
        );

        q4nx.load_weights(this->mlp_fc1_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc1.weight",
            sizeof(bf16) * padded_vision_mlp_intermediate_size
        );
        q4nx.load_weights(this->mlp_fc2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc2.weight",
            sizeof(bf16) * padded_vision_hidden_size
        );
        this->mlp_fc1_weight[layer_idx].sync_to_device();
        this->mlp_fc2_weight[layer_idx].sync_to_device();
    }
    q4nx.load_weights(
        this->merger_norm_weight,
        "model.visual.merger.norm.weight"
    );
    q4nx.load_weights(
        this->merger_norm_bias,
        "model.visual.merger.norm.bias"
    );

    this->merger_fc1_weight = merger_fc1_app.create_bo_buffer<bf16>(
        padded_merger_hidden_size*padded_merger_hidden_size+padded_merger_hidden_size

    );
    this->merger_fc2_weight = merger_fc2_app.create_bo_buffer<bf16>(
        padded_merger_hidden_size* padded_merger_hidden_size + padded_vision_out_hidden_size
    );
    q4nx.load_weights(this->merger_fc1_weight,
        "model.visual.merger.linear_fc1.bias",
        0
    );
    q4nx.load_weights(this->merger_fc2_weight,
        "model.visual.merger.linear_fc2.bias",
        0
    );
    q4nx.load_weights(this->merger_fc1_weight,
        "model.visual.merger.linear_fc1.weight",
        sizeof(bf16) * padded_merger_hidden_size
    );
    q4nx.load_weights(this->merger_fc2_weight,
        "model.visual.merger.linear_fc2.weight",
        sizeof(bf16) * padded_vision_out_hidden_size
    );
}
