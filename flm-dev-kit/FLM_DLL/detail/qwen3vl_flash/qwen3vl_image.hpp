// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "typedef.hpp"
#include <string>
#include <vector>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen3vl_flash/qwen3vl_flash.hpp"
#include "vision/conv3dPatchEmbedding.hpp"
#include "rot_pos_emb.hpp"
#include "norm.hpp"

constexpr unsigned int QWEN3_VISION_EMBED_DIM = 1024;
constexpr unsigned int QWEN3_VISION_NUM_HEADS = 16;
constexpr unsigned int QWEN3_VISION_HEAD_DIM = QWEN3_VISION_EMBED_DIM / QWEN3_VISION_NUM_HEADS;
constexpr unsigned int QWEN3_VISION_HIDDEN_SIZE = QWEN3_VISION_EMBED_DIM;
constexpr unsigned int QWEN3_VISION_MLP_INTERMEDIATE_SIZE = 4096;
constexpr unsigned int QWEN3_VISION_NUM_POSITION_EMBEDDINGS = 2304;
constexpr unsigned int QWEN3_VISION_NUM_LAYERS = 24;
constexpr float QWEN3_VISION_LAYER_NORM_EPSILON = 1e-6f;
constexpr unsigned int QWEN3_DEEPSTACK_OUTPUT_SIZE = 2560;
constexpr unsigned int QWEN3_DEEPSTACK_HIDDEN_SIZE = QWEN3_VISION_HIDDEN_SIZE * (QWEN3_SPATIAL_MERGE_SIZE*QWEN3_SPATIAL_MERGE_SIZE);
constexpr unsigned int QWEN3_VISION_DEEPSTACK_NUM_LAYERS = 4;

/// @brief The padded row geometry of one image batch, produced by setup().
struct qwen3vl_vision_shape {
    int seq_len = 0;
    int seq_len_padded = 0;
    int deepstack_seq_len = 0;
    int deepstack_seq_len_padded = 0;
    std::vector<std::vector<int32_t>> image_grid_thw;
};

class Qwen3VLImageEncoder{

    public:

        ~Qwen3VLImageEncoder();
        void init_weights(SafeTensors &q4nx);
        Qwen3VLImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance);

        /// @brief Readies the sequences and buffers for an image batch; see the .cpp.
        qwen3vl_vision_shape setup(void* image_payload_ptr);

        std::vector<std::vector<bf16>> encode(  void* image_payload_ptr);
        std::vector<std::vector<bf16>> encode(  void* image_payload_ptr, const qwen3vl_vision_shape& shape);

        LM_Config config;
        npu_xclbin_manager *npu;

        uint32_t MM_tile_M = 128;
        uint32_t MM_tile_K = 512;
        uint32_t MM_tile_N = 64;
        uint32_t MM_ROW_SIZE = 4;
        uint32_t MM_COL_SIZE = 8;

        uint32_t seq_len_pad_requirement_for_MM = MM_ROW_SIZE*MM_tile_M;

        // parameter for vision attention kernel
        uint32_t vision_num_of_columns = 8;
        uint32_t vision_num_of_rows = 4;
        uint32_t vision_CU_mode = 2;
        uint32_t vision_LQ_per_CT = 32;
        uint32_t vision_LK_per_CT = 512;
        uint32_t vision_LQ_internal=32;
        uint32_t vision_LK_internal=32;
        uint32_t vision_L_padded_requirement_for_attention = 512;  // padding for L_seq in vision attention
        uint32_t vision_S_padded_requirement_for_attention = vision_LK_per_CT; // padding for S_seq in vision attention
        //uint32_t vision_L_padded_requirement_for_attention = 32;

        // IF true, reorder qkv from L_Seq x (3*QWEN3_VISION_NUM_HEADS*QWEN3_VISION_HEAD_DIM) row major ->
        // [ 3, QWEN3_VISION_NUM_HEADS, L_Seq ,QWEN3_VISION_HEAD_DIM]
        bool ENABLE_QKV_REORDER = false; // don't use True, have bug in multiple image
        // for MM runtime sequence
        uint32_t rtp_address = 4096; // offset right after stack size
        uint32_t rtp_sync_lock_id = 10; // the rtp sync lock
        bool ENABLE_AXI4 = true;
        bool IS_B_ROW_MAJOR = false;

        // debug ptr for now
        std::string model_path;

        npu_app_manager* fla;
        npu_app_manager* proj;

        // define the necessary bitstream no
        npu_app flash_attention_app;
        npu_app qkv_projection_app;
        npu_app o_projection_app;
        npu_app mlp_fc1_app;
        npu_app mlp_fc2_app;

        npu_app deepstack_fc1_app;
        npu_app deepstack_fc2_app;

        // The weights
        buffer<bf16> patch_embed_proj_weight;
        buffer<bf16> patch_embed_proj_bias;
        buffer<bf16> pos_embed_weight;

        std::vector<buffer<bf16>> norm_1_weight;
        std::vector<buffer<bf16>> norm_1_bias;
        std::vector<buffer<bf16>> norm_2_weight;
        std::vector<buffer<bf16>> norm_2_bias;
        std::vector<buffer<bf16>> attn_qkv_weight;
        // std::vector<buffer<bf16>> attn_qkv_bias;
        std::vector<buffer<bf16>> o_proj_weight;
        // std::vector<buffer<bf16>> o_proj_bias;
        std::vector<buffer<bf16>> mlp_fc1_weight;
        // std::vector<buffer<bf16>> mlp_fc1_bias;
        std::vector<buffer<bf16>> mlp_fc2_weight;
        //std::vector<buffer<bf16>> mlp_fc2_bias;

        std::vector<buffer<bf16>> deepstack_fc1_weight;
        //std::vector<buffer<bf16>> deepstack_fc1_bias;
        std::vector<buffer<bf16>> deepstack_fc2_weight;
        //std::vector<buffer<bf16>> deepstack_fc2_bias;
        std::vector<buffer<bf16>> deepstack_norm_weight;
        std::vector<buffer<bf16>> deepstack_norm_bias;

        // encode() rebuilds the runtime sequences and the device buffers only when
        // the padded shapes actually change, so a repeated image size costs nothing
        int cached_seq_len_padded = -1;
        int cached_deepstack_seq_len_padded = -1;
        std::vector<std::vector<int32_t>> cached_image_grid_thw;

        // the vectors
        // hidden_state doubles as the residual: both adds accumulate in place
        buffer<bf16> hidden_state;

        buffer<bf16> mm_qkv_input;
        buffer<bf16> mm_qkv_output_mha_input;
        buffer<bf16> mha_qkv_output_mm_o_input;
        buffer<bf16> mm_o_output_fc1_proj_input;
        buffer<bf16> mm_fc1_output_fc2_input;
        buffer<bf16> mm_fc2_output;

        buffer<bf16> deepstack_fc1_input;
        buffer<bf16> deepstack_fc1_output_fc2_input;
        buffer<bf16> deepstack_fc2_output;
};
