// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "typedef.hpp"
#include <string>
#include <vector>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen2vl/qwen2vl_npu.hpp"

constexpr unsigned int QWEN2_VISION_HIDDEN_SIZE = 1280;
constexpr unsigned int QWEN2_VISION_NUM_HEADS = 16;
constexpr unsigned int QWEN2_VISION_HEAD_DIM = QWEN2_VISION_HIDDEN_SIZE / QWEN2_VISION_NUM_HEADS;
constexpr unsigned int QWEN2_VISION_MLP_INTERMEDIATE_SIZE = 3420; // NOTE: need padding
constexpr float  QWEN2_VISION_RMS_NORM_EPSILON = 1e-6f;
constexpr unsigned int QWEN2_NUM_LAYERS = 32;
constexpr unsigned int QWEN2_VISION_OUT_HIDDEN_SIZE = 2048;
constexpr unsigned int QWEN2_MERGER_HIDDEN_SIZE = QWEN2_VISION_HIDDEN_SIZE * (QWEN2_SPATIAL_MERGE_SIZE*QWEN2_SPATIAL_MERGE_SIZE);
constexpr unsigned int QWEN2_MERGER_MLP_INTERMEDIATE_SIZE = QWEN2_MERGER_HIDDEN_SIZE;

class Qwen2VLImageEncoder{

    public:

        ~Qwen2VLImageEncoder();
        void init_weights(SafeTensors &q4nx);
        Qwen2VLImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance);
        void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight);
        std::vector<int> RotaryEmbedGeneration(
            std::vector<std::vector<int32_t>>   image_grid_thw,
            int seq_len,
            int seq_len_padded,
            std::vector<float>& cos_emb,
            std::vector<float>& sin_emb
        );
        void simd_add(
            bf16* input1,
            bf16* input2,
            bf16* output,
            size_t size
        );
        void simd_mul(
            bf16* input1,
            bf16* input2,
            bf16* output,
            size_t size
        );

        std::vector<bf16> encode(  void* image_payload_ptr);

        LM_Config config;
        npu_xclbin_manager *npu;

        uint32_t MM_tile_M = 128;
        uint32_t MM_tile_K = 256;
        uint32_t MM_tile_N = 64;
        uint32_t MM_ROW_SIZE = 4;
        uint32_t MM_COL_SIZE = 8;

        uint32_t seq_len_pad_requirement_for_MM = MM_ROW_SIZE*MM_tile_M;

        // parameter for vision attention kernel for full window, not windowed attention
        uint32_t vision_num_of_columns = 8;
        uint32_t vision_num_of_rows = 4;

        uint32_t vision_intermediate_size_padded = (QWEN2_VISION_MLP_INTERMEDIATE_SIZE + MM_tile_K - 1) / MM_tile_K * MM_tile_K; // round up to multiple of MM_tile_K

        uint16_t vision_CU_mode = 2;
        uint16_t vision_LQ_per_CT = 32;
        uint16_t vision_LK_per_CT = 512;
        uint16_t vision_LQ_internal=32;
        uint16_t vision_LK_internal=32;
        uint32_t vision_L_padded_requirement_for_attention = 512;  // padding for L_seq in vision attention
        uint32_t vision_S_padded_requirement_for_attention = vision_LK_per_CT; // padding for S_seq in vision attention
        // assert(vision_L_padded_requirement_for_attention% vision_S_padded_requirement_for_attention == 0); // ensure that the padding requirement for L is multiple of that for S, otherwise the kernel design needs to be changed

        bool ENABLE_QKV_REORDER = false;

        // config for mm
        uint32_t rtp_address = 4096;// for mm
        uint32_t rtp_sync_lock_id = 10; // the rtp sync lock
        bool ENABLE_AXI4 = true;
        bool IS_B_ROW_MAJOR = false;

        // config for window attention
        uint32_t window_LQ = 64;
        uint32_t window_LK = 64;
        uint32_t window_L_padded_requirement_for_attention = (vision_num_of_rows *window_LQ);

        std::string model_path;

        npu_app_manager* vision_flash_attention;
        npu_app_manager* vision_proj;
        npu_app_manager* vision_window_attention;

        npu_app vision_flash_attention_app;
        npu_app vision_window_attention_app;
        npu_app vision_q_projection_app;
        npu_app vision_k_projection_app;
        npu_app vision_v_projection_app;
        npu_app vision_o_projection_app;

        npu_app vision_mlp_gate_proj_app;
        npu_app vision_mlp_up_proj_app;
        npu_app vision_mlp_down_proj_app;

        npu_app merger_mlp_fc1_app;
        npu_app merger_mlp_fc2_app;

        // The weights
        //NOTE: for vision, all bias are load into the weight buffer
        buffer<bf16> patch_embed_proj_weight;

        std::vector<buffer<bf16>> norm_1_weight;
        std::vector<buffer<bf16>> norm_2_weight;

        std::vector<buffer<bf16>> attn_q_weight;
        std::vector<buffer<bf16>> attn_k_weight;
        std::vector<buffer<bf16>> attn_v_weight;
        std::vector<buffer<bf16>> attn_o_weight;

        std::vector<buffer<bf16>> mlp_gate_proj_weight;
        std::vector<buffer<bf16>> mlp_up_proj_weight;
        std::vector<buffer<bf16>> mlp_down_proj_weight;

        buffer<bf16> post_norm_weight;

        buffer<bf16> merger_mlp_fc1_weight;
        buffer<bf16> merger_mlp_fc1_input;
        buffer<bf16> merger_mlp_fc1_out_fc2_in;
        buffer<bf16> merger_mlp_fc2_out;
        buffer<bf16> merger_mlp_fc2_weight;

        buffer<bf16> mm_qkv_input;
        buffer<bf16> mm_q_output;
        buffer<bf16> mm_k_output;
        buffer<bf16> mm_v_output;
        buffer<bf16> window_attention_o_output;
        buffer<bf16> mm_o_output_fc1_input;
        buffer<bf16> mm_gate_output;
        buffer<bf16> mm_up_output;

        buffer<bf16> hidden_state;
        buffer<bf16> residual_buffer;
};
