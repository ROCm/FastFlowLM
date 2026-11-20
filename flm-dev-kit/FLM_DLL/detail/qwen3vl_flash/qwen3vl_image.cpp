// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "qwen3vl_image.hpp"

#include <thread>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif

#include "utils/error_measure.hpp"
#include "vision/pos_embed_interpolate.hpp"
#include "vision/vision_prefill_helper.hpp"
#include "rot_pos_emb.hpp"
#include "norm.hpp"
#include "npu_sequences/mmRuntimeSequence.hpp"
#include "seq_gen.hpp"

// #define DEBUG_PRINT_ENCODE_TIME_DETAIL 1
// #define DEBUG_PRINT_ENCODE_ERROR_METRICS 1

Qwen3VLImageEncoder::~Qwen3VLImageEncoder()
{
}

Qwen3VLImageEncoder::Qwen3VLImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance)
    : config(config), npu(npu_instance), model_path(config.model_path)
{

    this->fla = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_attn.xclbin"));
    this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));

    this->flash_attention_app = this->fla->create_app();
    this->qkv_projection_app = this->proj->create_app();
    this->o_projection_app = this->proj->create_app();
    this->mlp_fc1_app = this->proj->create_app();
    this->mlp_fc2_app = this->proj->create_app();
    this->deepstack_fc1_app = this->proj->create_app();
    this->deepstack_fc2_app = this->proj->create_app();

    // resize the vectors to hold all layers weights
    norm_1_weight.resize(QWEN3_VISION_NUM_LAYERS);
    norm_1_bias.resize(QWEN3_VISION_NUM_LAYERS);
    norm_2_weight.resize(QWEN3_VISION_NUM_LAYERS);
    norm_2_bias.resize(QWEN3_VISION_NUM_LAYERS);
    attn_qkv_weight.resize(QWEN3_VISION_NUM_LAYERS);
    //attn_qkv_bias.resize(QWEN3_VISION_NUM_LAYERS);
    o_proj_weight.resize(QWEN3_VISION_NUM_LAYERS);
    //o_proj_bias.resize(QWEN3_VISION_NUM_LAYERS);
    mlp_fc1_weight.resize(QWEN3_VISION_NUM_LAYERS);
    //mlp_fc1_bias.resize(QWEN3_VISION_NUM_LAYERS);
    mlp_fc2_weight.resize(QWEN3_VISION_NUM_LAYERS);
    //mlp_fc2_bias.resize(QWEN3_VISION_NUM_LAYERS);

    deepstack_fc1_weight.resize(QWEN3_VISION_DEEPSTACK_NUM_LAYERS); // only have 4 deepstack in QWEN3VL
    //deepstack_fc1_bias.resize(QWEN3_VISION_DEEPSTACK_NUM_LAYERS);
    deepstack_fc2_weight.resize(QWEN3_VISION_DEEPSTACK_NUM_LAYERS);
    //deepstack_fc2_bias.resize(QWEN3_VISION_DEEPSTACK_NUM_LAYERS);
    deepstack_norm_weight.resize(QWEN3_VISION_DEEPSTACK_NUM_LAYERS);
    deepstack_norm_bias.resize(QWEN3_VISION_DEEPSTACK_NUM_LAYERS);
}
void Qwen3VLImageEncoder::init_weights(SafeTensors &q4nx){
    // std::cout << "Loading Qwen3VL Image Encoder weights from " << this->config.get<std::string>("vision_model_weight", "") << " ..." <<std::endl;
    q4nx.load_weights(patch_embed_proj_weight, "model.visual.patch_embed.proj.weight");
    q4nx.load_weights(patch_embed_proj_bias,"model.visual.patch_embed.proj.bias");
    q4nx.load_weights(pos_embed_weight,  "model.visual.pos_embed.weight");

    //load each layer weights
    for(int layer_idx = 0; layer_idx < QWEN3_VISION_NUM_LAYERS; layer_idx++){
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

        //attention
        // q4nx.load_weights(this->attn_qkv_bias[layer_idx],
        //     "model.visual.blocks." + std::to_string(layer_idx) + ".attn.qkv.bias"
        // );
        // q4nx.load_weights(this->o_proj_bias[layer_idx],
        //     "model.visual.blocks." + std::to_string(layer_idx) + ".attn.proj.bias"
        // );

        this->attn_qkv_weight[layer_idx] = qkv_projection_app.create_bo_buffer<bf16>(QWEN3_VISION_HIDDEN_SIZE*(3*QWEN3_VISION_HIDDEN_SIZE) + 3*QWEN3_VISION_HIDDEN_SIZE);
        this->o_proj_weight[layer_idx] = o_projection_app.create_bo_buffer<bf16>(QWEN3_VISION_HIDDEN_SIZE*(QWEN3_VISION_HIDDEN_SIZE) + QWEN3_VISION_HIDDEN_SIZE );

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
             sizeof(bf16) * 3*QWEN3_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->o_proj_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.proj.weight",
            sizeof(bf16) * QWEN3_VISION_HIDDEN_SIZE
        );
        this->attn_qkv_weight[layer_idx].sync_to_device();
        this->o_proj_weight[layer_idx].sync_to_device();

        // mlo
        this->mlp_fc1_weight[layer_idx] = mlp_fc1_app.create_bo_buffer<bf16>(QWEN3_VISION_HIDDEN_SIZE*QWEN3_VISION_MLP_INTERMEDIATE_SIZE + QWEN3_VISION_MLP_INTERMEDIATE_SIZE);
        this->mlp_fc2_weight[layer_idx] = mlp_fc2_app.create_bo_buffer<bf16>(QWEN3_VISION_MLP_INTERMEDIATE_SIZE*QWEN3_VISION_HIDDEN_SIZE + QWEN3_VISION_HIDDEN_SIZE);

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
            sizeof(bf16) * QWEN3_VISION_MLP_INTERMEDIATE_SIZE
        );
        q4nx.load_weights(this->mlp_fc2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc2.weight",
            sizeof(bf16) * QWEN3_VISION_HIDDEN_SIZE
        );
        this->mlp_fc1_weight[layer_idx].sync_to_device();
        this->mlp_fc2_weight[layer_idx].sync_to_device();

        // q4nx.load_weights(this->mlp_fc1_bias[layer_idx],
        //     "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc1.bias"
        // );
        // q4nx.load_weights(this->mlp_fc2_bias[layer_idx],
        //     "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc2.bias"
        // );
    }

    for(int deep_stack_indx = 0; deep_stack_indx < QWEN3_VISION_DEEPSTACK_NUM_LAYERS-1; deep_stack_indx++){
        q4nx.load_weights(
            this->deepstack_norm_weight[deep_stack_indx],
            "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".norm.weight"
        );
        q4nx.load_weights(
            this->deepstack_norm_bias[deep_stack_indx],
            "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".norm.bias"
        );

        // mlp
        this->deepstack_fc1_weight[deep_stack_indx] = deepstack_fc1_app.create_bo_buffer<bf16>(QWEN3_DEEPSTACK_HIDDEN_SIZE*QWEN3_DEEPSTACK_HIDDEN_SIZE + QWEN3_DEEPSTACK_HIDDEN_SIZE);
        this->deepstack_fc2_weight[deep_stack_indx] = deepstack_fc2_app.create_bo_buffer<bf16>(QWEN3_DEEPSTACK_HIDDEN_SIZE*QWEN3_DEEPSTACK_OUTPUT_SIZE + QWEN3_DEEPSTACK_OUTPUT_SIZE);

        q4nx.load_weights(this->deepstack_fc1_weight[deep_stack_indx],
          "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".linear_fc1.bias",
          0
        );
        q4nx.load_weights(this->deepstack_fc2_weight[deep_stack_indx],
          "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".linear_fc2.bias",
          0
        );

        q4nx.load_weights(this->deepstack_fc1_weight[deep_stack_indx],
          "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".linear_fc1.weight",
          sizeof(bf16) * QWEN3_DEEPSTACK_HIDDEN_SIZE
        );
        q4nx.load_weights(this->deepstack_fc2_weight[deep_stack_indx],
          "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".linear_fc2.weight",
          sizeof(bf16) * QWEN3_DEEPSTACK_OUTPUT_SIZE
        );
        this->deepstack_fc1_weight[deep_stack_indx].sync_to_device();
        this->deepstack_fc2_weight[deep_stack_indx].sync_to_device();
        // q4nx.load_weights(this->deepstack_fc1_bias[deep_stack_indx],
        //   "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".linear_fc1.bias"
        // );
        // q4nx.load_weights(this->deepstack_fc2_bias[deep_stack_indx],
        //   "model.visual.deepstack_merger_list." + std::to_string(deep_stack_indx) + ".linear_fc2.bias"
        // );
    }
    // weights for last deepstack name is a bit different
    int deep_stack_indx = QWEN3_VISION_DEEPSTACK_NUM_LAYERS - 1;
    q4nx.load_weights(
        this->deepstack_norm_weight[deep_stack_indx],
        "model.visual.merger.norm.weight"
    );
    q4nx.load_weights(
        this->deepstack_norm_bias[deep_stack_indx],
        "model.visual.merger.norm.bias"
    );
    // mlp
    this->deepstack_fc1_weight[deep_stack_indx] =  deepstack_fc1_app.create_bo_buffer<bf16>(QWEN3_DEEPSTACK_HIDDEN_SIZE*QWEN3_DEEPSTACK_HIDDEN_SIZE +QWEN3_DEEPSTACK_HIDDEN_SIZE );
    this->deepstack_fc2_weight[deep_stack_indx] =  deepstack_fc2_app.create_bo_buffer<bf16>(QWEN3_DEEPSTACK_HIDDEN_SIZE*QWEN3_DEEPSTACK_OUTPUT_SIZE + QWEN3_DEEPSTACK_OUTPUT_SIZE);

    q4nx.load_weights(this->deepstack_fc1_weight[deep_stack_indx],
      "model.visual.merger.linear_fc1.bias",
      0
    );
    q4nx.load_weights(this->deepstack_fc2_weight[deep_stack_indx],
      "model.visual.merger.linear_fc2.bias",
      0
    );
    q4nx.load_weights(this->deepstack_fc1_weight[deep_stack_indx],
      "model.visual.merger.linear_fc1.weight",
      sizeof(bf16) * QWEN3_DEEPSTACK_HIDDEN_SIZE
    );
    q4nx.load_weights(this->deepstack_fc2_weight[deep_stack_indx],
      "model.visual.merger.linear_fc2.weight",
        sizeof(bf16) * QWEN3_DEEPSTACK_OUTPUT_SIZE
    );
    this->deepstack_fc1_weight[deep_stack_indx].sync_to_device();
    this->deepstack_fc2_weight[deep_stack_indx].sync_to_device();
    // q4nx.load_weights(this->deepstack_fc1_bias[deep_stack_indx],
    //   "model.visual.merger.linear_fc1.bias"
    // );
    // q4nx.load_weights(this->deepstack_fc2_bias[deep_stack_indx],
    //   "model.visual.merger.linear_fc2.bias"
    // );
}

/// @brief Computes the padded geometry of an image batch and readies the NPU for it.
///
/// Split out of encode() so the caller can hoist it: it is pure host work
/// (sequence generation plus buffer cuts) and touches no device queue, so a
/// caller may run another engine's setup on a second thread alongside encode().
qwen3vl_vision_shape Qwen3VLImageEncoder::setup(void* image_payload_ptr)
{
    auto round_up_to_multiple_lambda = [](int x, int multiple) -> int {
            return ((x + multiple - 1) / multiple) * multiple;
        };

    qwen3vl_image_payload_t* image_payload = (qwen3vl_image_payload_t*)image_payload_ptr;

    qwen3vl_vision_shape shape;
    std::vector<std::vector<int32_t>>& image_grid_thw = shape.image_grid_thw;
    int seq_len = 0;
    int seq_len_of_last_image = 0;
    for(auto img: image_payload->images ){
        int t = img.grid_h * img.grid_w;
        seq_len += t;
        seq_len_of_last_image = t;
        image_grid_thw.push_back({1, img.grid_h, img.grid_w});
    }

    int seq_len_padded = round_up_to_multiple_lambda(seq_len_of_last_image, vision_L_padded_requirement_for_attention) - seq_len_of_last_image;
    seq_len_padded += seq_len;
    seq_len_padded = round_up_to_multiple_lambda(seq_len_padded, seq_len_pad_requirement_for_MM);

    int deepstack_seq_len = (seq_len*QWEN3_VISION_HIDDEN_SIZE) / (QWEN3_DEEPSTACK_HIDDEN_SIZE);
    int deepstack_seq_len_padded = round_up_to_multiple_lambda(deepstack_seq_len, seq_len_pad_requirement_for_MM);

    #if DEBUG_PRINT_ENCODE_TIME_DETAIL
        std::cout <<  "Total seq len: " << seq_len << std::endl;
        std::cout <<  "Total seq len padded: " << seq_len_padded << std::endl;
    #endif
    hidden_state.resize(seq_len_padded * QWEN3_VISION_HIDDEN_SIZE);

    // regenerating a sequence bumps its version, which forces the whole kernel to
    // be re-elaborated on the next dispatch; skip it when the shape is unchanged
    const bool mm_shape_changed = (seq_len_padded != this->cached_seq_len_padded);
    const bool deepstack_shape_changed = (deepstack_seq_len_padded != this->cached_deepstack_seq_len_padded);
    const bool grid_changed = (image_grid_thw != this->cached_image_grid_thw);

    if (mm_shape_changed)
    {
        bool ADD_BIAS= true;

        // pre-generate the mm runtime sequence
        generate_mm_sequence<bf16, bf16>(*this->qkv_projection_app.seq(),
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE,    3*QWEN3_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                ENABLE_QKV_REORDER, QWEN3_VISION_HEAD_DIM
        );

        generate_mm_sequence<bf16, bf16>(*this->o_projection_app.seq(),
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE,  QWEN3_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );
        generate_mm_sequence<bf16, bf16>(*this->mlp_fc1_app.seq(),
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE,  QWEN3_VISION_MLP_INTERMEDIATE_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 1,
                false,0
        );

        generate_mm_sequence<bf16, bf16>(*this->mlp_fc2_app.seq(),
                seq_len_padded,  QWEN3_VISION_MLP_INTERMEDIATE_SIZE,  QWEN3_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );
    }

    if (deepstack_shape_changed)
    {
        bool ADD_BIAS= true;
        generate_mm_sequence<bf16, bf16>(*this->deepstack_fc1_app.seq(),
                deepstack_seq_len_padded, QWEN3_DEEPSTACK_HIDDEN_SIZE,  QWEN3_DEEPSTACK_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 1,
                false,0
        );
        generate_mm_sequence<bf16, bf16>(*this->deepstack_fc2_app.seq(),
                deepstack_seq_len_padded, QWEN3_DEEPSTACK_HIDDEN_SIZE,  QWEN3_DEEPSTACK_OUTPUT_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );
    }

    if (grid_changed)
    {
        //mha
        // gen_mha_main(
        //     this->flash_attention_app.seq(),
        //     image_grid_thw,
        //     vision_L_padded_requirement_for_attention,
        //     QWEN3_VISION_HIDDEN_SIZE
        // );
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
            QWEN3_VISION_HIDDEN_SIZE,
            ENABLE_QKV_REORDER
        );
    }

    if (mm_shape_changed)
    {
        // initialize the remainig buffer
        mm_qkv_input = qkv_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN3_VISION_HIDDEN_SIZE);
        mm_qkv_output_mha_input = qkv_projection_app.create_bo_buffer<bf16>(seq_len_padded*QWEN3_VISION_HIDDEN_SIZE*3);
        mha_qkv_output_mm_o_input = o_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN3_VISION_HIDDEN_SIZE);
        mm_o_output_fc1_proj_input = o_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN3_VISION_HIDDEN_SIZE);
        mm_fc1_output_fc2_input = mlp_fc1_app.create_bo_buffer<bf16>(seq_len_padded * QWEN3_VISION_MLP_INTERMEDIATE_SIZE);
        mm_fc2_output = mlp_fc2_app.create_bo_buffer<bf16>(seq_len_padded * QWEN3_VISION_HIDDEN_SIZE);
    }

    if (deepstack_shape_changed)
    {
        deepstack_fc1_input = deepstack_fc1_app.create_bo_buffer<bf16>(deepstack_seq_len_padded*QWEN3_DEEPSTACK_HIDDEN_SIZE);
        deepstack_fc1_output_fc2_input = deepstack_fc1_app.create_bo_buffer<bf16>( deepstack_seq_len_padded*QWEN3_DEEPSTACK_HIDDEN_SIZE);
        deepstack_fc2_output = deepstack_fc2_app.create_bo_buffer<bf16>( deepstack_seq_len_padded*QWEN3_DEEPSTACK_OUTPUT_SIZE);
    }

    this->cached_seq_len_padded = seq_len_padded;
    this->cached_deepstack_seq_len_padded = deepstack_seq_len_padded;
    this->cached_image_grid_thw = image_grid_thw;

    shape.seq_len = seq_len;
    shape.seq_len_padded = seq_len_padded;
    shape.deepstack_seq_len = deepstack_seq_len;
    shape.deepstack_seq_len_padded = deepstack_seq_len_padded;
    return shape;
}

std::vector<std::vector<bf16>> Qwen3VLImageEncoder::encode( void* image_payload_ptr)
{
    return this->encode(image_payload_ptr, this->setup(image_payload_ptr));
}

std::vector<std::vector<bf16>> Qwen3VLImageEncoder::encode( void* image_payload_ptr, const qwen3vl_vision_shape& shape)
{
    //DEBUG
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    SafeTensors reference_tensor(this->model_path + "/visual_tensors_step_0.safetensors");
    #endif
    auto encoder_start_time = std::chrono::high_resolution_clock::now();

    qwen3vl_image_payload_t* image_payload = (qwen3vl_image_payload_t*)image_payload_ptr;
    std::vector<std::vector<bf16>> final_res;

    std::vector<float> cos_emb;
    std::vector<float> sin_emb;

    // setup() already sized every buffer and regenerated every sequence for this shape
    const std::vector<std::vector<int32_t>>& image_grid_thw = shape.image_grid_thw;
    const int seq_len = shape.seq_len;
    const int seq_len_padded = shape.seq_len_padded;
    const int deepstack_seq_len = shape.deepstack_seq_len;
    const int deepstack_seq_len_padded = shape.deepstack_seq_len_padded;

    auto time_start_patch_embed = std::chrono::high_resolution_clock::now();
    std::vector<bf16> patch_embed = conv3d_patch_embed_avx512_max_performance(
        image_payload->_data__processed.data(),
        patch_embed_proj_weight.data(),
        patch_embed_proj_bias.data(),
        seq_len,
        3, // r, g, b
        QWEN3_VISION_EMBED_DIM,
        QWEN3_TEMPORAL_PATCH_SIZE,
        QWEN3_PATCH_SIZE
    );
    auto time_end_patch_embed = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_patch_embed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_patch_embed - time_start_patch_embed).count();
        std::cout << "Patch embedding time: " << duration_patch_embed << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error after patch embedding comparison:" << std::endl;
        buffer<bf16> after_patch_embed;
        reference_tensor.load_weights(
            after_patch_embed,
            "after_patch_embed"
        );
        print_error_metrics<bf16, bf16>(
            patch_embed.data(), after_patch_embed.data(),
                1,
                seq_len, QWEN3_VISION_EMBED_DIM,
                seq_len, QWEN3_VISION_EMBED_DIM
        );

    #endif

    auto time_start_pos_emb = std::chrono::high_resolution_clock::now();
    std::vector<bf16>  pos_embeds =  fast_pos_embed_interpolate(
        image_grid_thw,
        pos_embed_weight.data(),
        std::sqrt(QWEN3_VISION_NUM_POSITION_EMBEDDINGS),
        QWEN3_VISION_EMBED_DIM,
        QWEN3_SPATIAL_MERGE_SIZE
    );
    auto time_end_pos_emb = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_pos_emb = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_pos_emb - time_start_pos_emb).count();
        std::cout << "Position embedding time: " << duration_pos_emb << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
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
            seq_len, QWEN3_VISION_HIDDEN_SIZE,
            seq_len, QWEN3_VISION_HIDDEN_SIZE
        );
    #endif

    auto time_start_pose_embed_plus_patch_embed = std::chrono::high_resolution_clock::now();
    simd_add(
        patch_embed.data(), pos_embeds.data(),
        hidden_state.data(), seq_len *QWEN3_VISION_HIDDEN_SIZE
    );
    auto time_end_pose_embed_plus_patch_embed = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_pos_embed_plus_patch_embed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_pose_embed_plus_patch_embed - time_start_pose_embed_plus_patch_embed).count();
        std::cout << "Position embedding plus patch embedding time: " << duration_pos_embed_plus_patch_embed << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        std::cout << "Error after hidden_states_plus_pos_embeds:" << std::endl;
        buffer<bf16> after_pos_embed_plus_patch_embed;
        reference_tensor.load_weights(
            after_pos_embed_plus_patch_embed,
            "after_hidden_states_plus_pos_embeds"
        );
        print_error_metrics<bf16, bf16>(
            hidden_state.data(), after_pos_embed_plus_patch_embed.data(),
            1,
            seq_len, QWEN3_VISION_HIDDEN_SIZE,
            seq_len_padded, QWEN3_VISION_HIDDEN_SIZE
        );
    #endif

    auto time_Start_rotary_pos_emb = std::chrono::high_resolution_clock::now();
    std::vector<float> rotary_pos_emb = rot_pos_emb_float(
        image_grid_thw,
        QWEN3_SPATIAL_MERGE_SIZE,           // spatial_merge_size = 2
        QWEN3_VISION_HEAD_DIM / 2           // rot_dim = 64 / 2 = 32
    );
    auto time_end_rotary_pos_emb = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_rotary_pos_emb = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_rotary_pos_emb - time_Start_rotary_pos_emb).count();
        std::cout << "Rotary position embedding time: " << duration_rotary_pos_emb << " ms" << std::endl;
    #endif

    auto time_start_cos_sine_pose_emb = std::chrono::high_resolution_clock::now();
    cos_sine_pose_emb(rotary_pos_emb, cos_emb, sin_emb);
    auto time_end_cos_sine_pose_emb = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_cos_sine_pose_emb = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_cos_sine_pose_emb - time_start_cos_sine_pose_emb).count();
        std::cout << "Cosine and Sine embedding computation time: " << duration_cos_sine_pose_emb << " ms" << std::endl;
    #endif
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        buffer<float> position_embeddings_cos_ptr;
        reference_tensor.load_weights(
            position_embeddings_cos_ptr,
            "position_embeddings_cos"
        );
        buffer<float> position_embeddings_sin_ptr;
        reference_tensor.load_weights(
            position_embeddings_sin_ptr,
            "position_embeddings_sin"
        );
        std::cout << "Error after position_embeddings_cos comparison:" << std::endl;
        print_error_metrics<float, float>(
            cos_emb.data(), position_embeddings_cos_ptr.data(),
            1,
            seq_len, QWEN3_VISION_HEAD_DIM,
            seq_len, QWEN3_VISION_HEAD_DIM
        );

        std::cout << "Error after position_embeddings_sin comparison:" << std::endl;
        print_error_metrics<float, float>(
            sin_emb.data(), position_embeddings_sin_ptr.data(),
            1,
            seq_len, QWEN3_VISION_HEAD_DIM,
            seq_len, QWEN3_VISION_HEAD_DIM
        );

    #endif
    // the MM runs over seq_len_padded rows, so the padding tail must be defined
    memset(hidden_state.data() + (size_t)seq_len * QWEN3_VISION_HIDDEN_SIZE, 0,
           sizeof(bf16) * (size_t)(seq_len_padded - seq_len) * QWEN3_VISION_HIDDEN_SIZE);
    for(int layer_idx = 0; layer_idx < QWEN3_VISION_NUM_LAYERS; layer_idx++){

        // first, do the norm_1 output
        auto time_start_layernorm1 = std::chrono::high_resolution_clock::now();
        layernorm_parallel(
            seq_len,
            QWEN3_VISION_HIDDEN_SIZE,
            hidden_state.data(),
            norm_1_weight[layer_idx].data(),
            norm_1_bias[layer_idx].data(),
            QWEN3_VISION_LAYER_NORM_EPSILON,
            mm_qkv_input.data()  // Use separate output buffer!
        );
        auto time_end_layernorm1 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_layernorm1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_layernorm1 - time_start_layernorm1).count();
            std::cout << "Layer norm 1 time for layer " << layer_idx << ": " << duration_layernorm1 << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            std::cout << "Error after layernorm 1 comparison:" << std::endl;
            buffer<bf16> after_layernorm1;
            reference_tensor.load_weights(
                after_layernorm1,
                  "vision_block_normed_1_" + std::to_string(layer_idx)
            );
            std::cout << "Error after norm1 layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16,bf16>(
                mm_qkv_input.data(), after_layernorm1.data(),  // Compare mm_qkv buffer which has norm output
                1,
                seq_len, QWEN3_VISION_HIDDEN_SIZE,
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE
            );

        #endif

        auto time_start_qkv_projection = std::chrono::high_resolution_clock::now();
        mm_qkv_input.sync_to_device();
        qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
        mm_qkv_output_mha_input.sync_from_device();
        auto time_end_qkv_projection = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_qkv_projection = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_qkv_projection - time_start_qkv_projection).count();
            std::cout << "QKV projection time for layer " << layer_idx << ": " << duration_qkv_projection << " ms" << std::endl;
        #endif
        auto time_start_add_qkv_bias = std::chrono::high_resolution_clock::now();
        // perform the add bias of q,k,v bias - ONLY process seq_len, not seq_len_padded!
        // for(int l = 0;  l < seq_len; l++){
        //     simd_add(
        //         mm_qkv_output_mha_input.data() + l * (3*QWEN3_VISION_HIDDEN_SIZE),
        //         attn_qkv_weight[layer_idx].data(),
        //         mm_qkv_output_mha_input.data() + l * (3*QWEN3_VISION_HIDDEN_SIZE),
        //         3*QWEN3_VISION_HIDDEN_SIZE
        //     );
        // }
        auto time_end_add_qkv_bias = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_add_qkv_bias = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_add_qkv_bias - time_start_add_qkv_bias).count();
            std::cout << "Add QKV bias time for layer " << layer_idx << ": " << duration_add_qkv_bias << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference

            buffer<bf16> after_qkv_projection;
            reference_tensor.load_weights(
                after_qkv_projection,
                "vision_attention_qkv_mm_with_bias_" + std::to_string(layer_idx)
            );
            std::cout << "Error after qkv mm layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_qkv_output_mha_input.data(), after_qkv_projection.data(),
                1,
                seq_len, 3*QWEN3_VISION_HIDDEN_SIZE,
                seq_len_padded, 3*QWEN3_VISION_HIDDEN_SIZE
            );

        #endif

        // now apply rope to q, k
        auto time_start_rotary_pos_emb_application = std::chrono::high_resolution_clock::now();
        if(ENABLE_QKV_REORDER){
            apply_rotary_pos_emb_vision_inplace_reordered(
                mm_qkv_output_mha_input.data(),
                cos_emb.data(),
                sin_emb.data(),
                seq_len,
                seq_len_padded,
                QWEN3_VISION_HIDDEN_SIZE,
                QWEN3_VISION_NUM_HEADS
            );
        }else{
            apply_rotary_pos_emb_vision_inplace(
                mm_qkv_output_mha_input.data(),
                cos_emb.data(),
                sin_emb.data(),
                seq_len,
                seq_len_padded,
                QWEN3_VISION_HIDDEN_SIZE,
                QWEN3_VISION_NUM_HEADS
            );
        }

        auto time_end_rotary_pos_emb_application = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_rotary_pos_emb_application = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_rotary_pos_emb_application - time_start_rotary_pos_emb_application).count();
            std::cout << "Rotary position embedding application time for layer " << layer_idx << ": " << duration_rotary_pos_emb_application << " ms" << std::endl;
        #endif

        // now mha
        auto time_start_mha = std::chrono::high_resolution_clock::now();
        mm_qkv_output_mha_input.sync_to_device();
        // mha_qkv_output_mm_o_input is cut from o_projection_app but written here by
        // flash_attention_app, and the two run out of different xclbins. Handing the
        // device buffer straight over looks like it should work and measurably hurt
        // image accuracy, so the attention result goes back through host memory.
        mha_qkv_output_mm_o_input.sync_to_device();
        flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
        mha_qkv_output_mm_o_input.sync_from_device();
        auto time_end_mha = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mha = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mha - time_start_mha).count();
            std::cout << "MHA time for layer " << layer_idx << ": " << duration_mha << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_attention_output_ptr;
            reference_tensor.load_weights(
                after_attention_output_ptr,
                "vision_attention_output" + std::to_string(layer_idx)
            );
            std::cout << "Error after attention output layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mha_qkv_output_mm_o_input.data(), after_attention_output_ptr.data(),
                1,
                seq_len, QWEN3_VISION_HIDDEN_SIZE,
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE
            );
        #endif

        // now the o projection
        auto time_start_o_projection = std::chrono::high_resolution_clock::now();
        mha_qkv_output_mm_o_input.sync_to_device();
        o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
        mm_o_output_fc1_proj_input.sync_from_device();
        auto time_end_o_projection = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_o_projection = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_o_projection - time_start_o_projection).count();
            std::cout << "O projection time for layer " << layer_idx << ": " << duration_o_projection << " ms" << std::endl;
        #endif
        // add the bias
        auto time_start_add_o_proj_bias = std::chrono::high_resolution_clock::now();
        // for(int l = 0;  l < seq_len; l++){
        //     simd_add(
        //         mm_o_output_fc1_proj_input.data() + l * (QWEN3_VISION_HIDDEN_SIZE),
        //         o_proj_weight[layer_idx].data(),
        //         mm_o_output_fc1_proj_input.data() + l * (QWEN3_VISION_HIDDEN_SIZE),
        //         QWEN3_VISION_HIDDEN_SIZE
        //     );
        // }
        auto time_end_add_o_proj_bias = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_add_o_proj_bias = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_add_o_proj_bias - time_start_add_o_proj_bias).count();
            std::cout << "Add o projection bias time for layer " << layer_idx << ": " << duration_add_o_proj_bias << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_o_project_ptr;
            reference_tensor.load_weights(
                after_o_project_ptr,
                "vision_attention_output_proj_" + std::to_string(layer_idx)
            );
            std::cout << "Error after o projection layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_o_output_fc1_proj_input.data(), after_o_project_ptr.data(),
                1,
                seq_len, QWEN3_VISION_HIDDEN_SIZE,
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE
            );
        #endif

        auto time_start_residual_add_1 = std::chrono::high_resolution_clock::now();
        // hidden_state is the residual: accumulate in place
        simd_add(
            mm_o_output_fc1_proj_input.data(),
            hidden_state.data(),
            hidden_state.data(),
            seq_len * QWEN3_VISION_HIDDEN_SIZE
        );
        auto time_end_residual_add_1 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_residual_add_1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_residual_add_1 - time_start_residual_add_1).count();
            std::cout << "Residual add 1 time for layer " << layer_idx << ": " << duration_residual_add_1 << " ms" << std::endl;
        #endif

        // norm2
        auto time_start_layernorm2 = std::chrono::high_resolution_clock::now();
        layernorm_parallel(
            seq_len,
            QWEN3_VISION_HIDDEN_SIZE,
            hidden_state.data(),
            norm_2_weight[layer_idx].data(),
            norm_2_bias[layer_idx].data(),
            QWEN3_VISION_LAYER_NORM_EPSILON,
            mm_o_output_fc1_proj_input.data()
        );
        auto time_end_layernorm2 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_layernorm2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_layernorm2 - time_start_layernorm2).count();
            std::cout << "Layer norm 2 time for layer " << layer_idx << ": " << duration_layernorm2 << " ms" << std::endl;
        #endif

        // now to fc1
        auto time_start_fc1 = std::chrono::high_resolution_clock::now();
        mm_o_output_fc1_proj_input.sync_to_device();
        mlp_fc1_app( mm_o_output_fc1_proj_input,  mlp_fc1_weight[layer_idx],   mm_fc1_output_fc2_input );
        // mm_fc1_output_fc2_input.sync_from_device();
        auto time_end_fc1 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_fc1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_fc1 - time_start_fc1).count();
            std::cout << "MLP FC1 time for layer " << layer_idx << ": " << duration_fc1 << " ms" << std::endl;
        #endif

        // Combined bias addition and GELU activation using optimized AVX-512
        auto time_start_mlp_fc1_gelu = std::chrono::high_resolution_clock::now();
        // simd_bias_add_gelu(
        //     mm_fc1_output_fc2_input.data(),
        //     mlp_fc1_weight[layer_idx].data(),
        //     mm_fc1_output_fc2_input.data(),
        //     seq_len * QWEN3_VISION_MLP_INTERMEDIATE_SIZE,
        //     QWEN3_VISION_MLP_INTERMEDIATE_SIZE
        // );
        auto time_end_mlp_fc1_gelu = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc1_gelu = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc1_gelu - time_start_mlp_fc1_gelu).count();
            std::cout << "MLP FC1 GELU time for layer " << layer_idx << ": " << duration_mlp_fc1_gelu << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_mlp_fc1_gelu_ptr;
            reference_tensor.load_weights(
                after_mlp_fc1_gelu_ptr,
                "vision_mlp_activated_output_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp fc1 gelu layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_fc1_output_fc2_input.data(), after_mlp_fc1_gelu_ptr.data(),
                1,
                seq_len, QWEN3_VISION_MLP_INTERMEDIATE_SIZE,
                seq_len_padded, QWEN3_VISION_MLP_INTERMEDIATE_SIZE
            );
        #endif

        // now do the fc2
        auto time_start_mlp_fc2 = std::chrono::high_resolution_clock::now();
        mlp_fc2_app( mm_fc1_output_fc2_input,  mlp_fc2_weight[layer_idx],   mm_fc2_output );
        mm_fc2_output.sync_from_device();
        auto time_end_mlp_fc2 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc2 - time_start_mlp_fc2).count();
            std::cout << "MLP FC2 time for layer " << layer_idx << ": " << duration_mlp_fc2 << " ms" << std::endl;
        #endif

        // add bias
        auto time_start_mlp_fc2_bias_add = std::chrono::high_resolution_clock::now();
        // for(int l = 0;  l < seq_len; l++){
        //     simd_add(
        //         mm_fc2_output.data() + l * (QWEN3_VISION_HIDDEN_SIZE),
        //         mlp_fc2_weight[layer_idx].data(),
        //         mm_fc2_output.data() + l * (QWEN3_VISION_HIDDEN_SIZE),
        //         QWEN3_VISION_HIDDEN_SIZE
        //     );
        // }
        auto time_end_mlp_fc2_bias_add = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc2_bias_add = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc2_bias_add - time_start_mlp_fc2_bias_add).count();
            std::cout << "MLP FC2 bias add time for layer " << layer_idx << ": " << duration_mlp_fc2_bias_add << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            //compare with the reference
            buffer<bf16> after_mlp_fc2_ptr;
            reference_tensor.load_weights(
                after_mlp_fc2_ptr,
                "vision_mlp_final_output_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp fc2 layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_fc2_output.data(), after_mlp_fc2_ptr.data(),
                1,
                seq_len, QWEN3_VISION_HIDDEN_SIZE,
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE
            );
        #endif

        auto time_start_mlp_residual_add = std::chrono::high_resolution_clock::now();
        simd_add(
            mm_fc2_output.data(),
            hidden_state.data(),
            hidden_state.data(),
            seq_len * QWEN3_VISION_HIDDEN_SIZE
        );
        auto time_end_mlp_residual_add = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_residual_add = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_residual_add - time_start_mlp_residual_add).count();
            std::cout << "MLP residual add time for layer " << layer_idx << ": " << duration_mlp_residual_add << " ms" << std::endl;
        #endif
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare the final
            buffer<bf16> after_mlp_residual_add_ptr;
            reference_tensor.load_weights(
                after_mlp_residual_add_ptr,
                "vision_block_post_mlp_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp residual add layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                hidden_state.data(), after_mlp_residual_add_ptr.data(),
                1,
                seq_len, QWEN3_VISION_HIDDEN_SIZE,
                seq_len_padded, QWEN3_VISION_HIDDEN_SIZE
            );

        #endif

        // now, handle the mlp merger stuff
        if(layer_idx == 5 || layer_idx == 11 || layer_idx == 17){
            int deep_stack_indx = final_res.size();

            std::vector<bf16> deepstack_hidden_vector(deepstack_seq_len_padded * QWEN3_DEEPSTACK_OUTPUT_SIZE);

            layernorm_parallel(
                deepstack_seq_len,
                QWEN3_DEEPSTACK_HIDDEN_SIZE,
                hidden_state.data(),
                deepstack_norm_weight[deep_stack_indx].data(),
                deepstack_norm_bias[deep_stack_indx].data(),
                QWEN3_VISION_LAYER_NORM_EPSILON,
                deepstack_fc1_input.data()
            );

            deepstack_fc1_input.sync_to_device();
            deepstack_fc1_app( deepstack_fc1_input,  deepstack_fc1_weight[deep_stack_indx],   deepstack_fc1_output_fc2_input );
            // deepstack_fc1_output_fc2_input.sync_from_device();

            // Combined bias addition and GELU activation using optimized AVX-512
            // simd_bias_add_gelu(
            //     deepstack_fc1_output_fc2_input.data(),
            //     deepstack_fc1_weight[deep_stack_indx].data(),
            //     deepstack_fc1_output_fc2_input.data(),
            //     deepstack_seq_len * QWEN3_DEEPSTACK_HIDDEN_SIZE,
            //     QWEN3_DEEPSTACK_HIDDEN_SIZE
            // );

            deepstack_fc2_app( deepstack_fc1_output_fc2_input,  deepstack_fc2_weight[deep_stack_indx],   deepstack_fc2_output );
            deepstack_fc2_output.sync_from_device();
            // // add bias
            // for(int l = 0;  l < deepstack_seq_len; l++){
            //     simd_add(
            //         deepstack_fc2_output.data() + l * (QWEN3_DEEPSTACK_OUTPUT_SIZE),
            //         deepstack_fc2_weight[deep_stack_indx].data(),
            //         deepstack_hidden_vector.data() + l * (QWEN3_DEEPSTACK_OUTPUT_SIZE),
            //         QWEN3_DEEPSTACK_OUTPUT_SIZE
            //     );
            // }
            memcpy(deepstack_hidden_vector.data(), deepstack_fc2_output.data(), sizeof(bf16) *deepstack_seq_len * QWEN3_DEEPSTACK_OUTPUT_SIZE);
            // push back
            final_res.push_back(std::move(deepstack_hidden_vector));
            #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
                // compare with the reference
                buffer<bf16> deepstack_output_ptr_ref;
                reference_tensor.load_weights(
                    deepstack_output_ptr_ref,
                    "deepstack_feature_" + std::to_string(layer_idx)
                );
                std::cout << "Error after deepstack block " << deep_stack_indx << " comparison:" << std::endl;
                print_error_metrics<bf16, bf16>(
                    final_res.back().data(), deepstack_output_ptr_ref.data(),
                    1,
                    deepstack_seq_len, QWEN3_DEEPSTACK_OUTPUT_SIZE,
                    deepstack_seq_len_padded, QWEN3_DEEPSTACK_OUTPUT_SIZE
                );
            #endif
        }
    }

    // do it for last layer deepstack

    std::vector<bf16> final_deepstack_hidden_vector(deepstack_seq_len_padded * QWEN3_DEEPSTACK_OUTPUT_SIZE);
    int deep_stack_indx = final_res.size();
    for(int s = 0; s < seq_len; s++){
        layernorm_high_precision(
            QWEN3_VISION_HIDDEN_SIZE,
            hidden_state.data() + s * QWEN3_VISION_HIDDEN_SIZE,
            deepstack_norm_weight[deep_stack_indx].data(),
            deepstack_norm_bias[deep_stack_indx].data(),
            QWEN3_VISION_LAYER_NORM_EPSILON,
            deepstack_fc1_input.data() + s * QWEN3_VISION_HIDDEN_SIZE
        );
    }

    deepstack_fc1_input.sync_to_device();
    deepstack_fc1_app( deepstack_fc1_input,  deepstack_fc1_weight[deep_stack_indx],   deepstack_fc1_output_fc2_input );

    // // Combined bias addition and GELU activation using optimized AVX-512
    // simd_bias_add_gelu(
    //     deepstack_fc1_output_fc2_input.data(),
    //     deepstack_fc1_weight[deep_stack_indx].data(),
    //     deepstack_fc1_output_fc2_input.data(),
    //     deepstack_seq_len * QWEN3_DEEPSTACK_HIDDEN_SIZE,
    //     QWEN3_DEEPSTACK_HIDDEN_SIZE
    // );

    //fc2

    deepstack_fc2_app( deepstack_fc1_output_fc2_input,  deepstack_fc2_weight[deep_stack_indx],   deepstack_fc2_output );
    deepstack_fc2_output.sync_from_device();
    // for(int l = 0;  l < deepstack_seq_len; l++){
    //     simd_add(
    //         deepstack_fc2_output.data() + l * (QWEN3_DEEPSTACK_OUTPUT_SIZE),
    //         deepstack_fc2_weight[deep_stack_indx].data(),
    //         final_deepstack_hidden_vector.data() + l * (QWEN3_DEEPSTACK_OUTPUT_SIZE),
    //         QWEN3_DEEPSTACK_OUTPUT_SIZE
    //     );
    // }
    memcpy(final_deepstack_hidden_vector.data(), deepstack_fc2_output.data(), sizeof(bf16) *deepstack_seq_len * QWEN3_DEEPSTACK_OUTPUT_SIZE);
    final_res.push_back(std::move(final_deepstack_hidden_vector));
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        buffer<bf16> deepstack_output_ptr_ref;
        reference_tensor.load_weights(
            deepstack_output_ptr_ref,
            "final_hidden_states_after_merger"
        );
        std::cout << "Error after final patch merger comparison:" << std::endl;
        print_error_metrics<bf16, bf16>(
            final_res.back().data(), deepstack_output_ptr_ref.data(),
            1,
            deepstack_seq_len, QWEN3_DEEPSTACK_OUTPUT_SIZE,
            deepstack_seq_len_padded, QWEN3_DEEPSTACK_OUTPUT_SIZE
        );

    #endif

    auto encode_time_end = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto total_encode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encode_time_end - encoder_start_time).count();
        std::cout << "Total encoding time: " << total_encode_duration << " ms" << std::endl;
    #endif
    return final_res;
}
