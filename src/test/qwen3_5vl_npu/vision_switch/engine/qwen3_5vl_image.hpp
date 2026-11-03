#pragma once
#include "typedef.hpp"
#include <string>
#include <vector>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/qwen3_5vl/qwen3_5vl_npu.hpp"
#include "vision/conv3dPatchEmbedding.hpp"
#include "vision/rot_pos_emb.hpp"
#include "vision/norm.hpp"

class Qwen3_5VLImageEncoder{

    public:

        ~Qwen3_5VLImageEncoder();
        void init_weights(SafeTensors &q4nx);
        Qwen3_5VLImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance, qwen3_5vl_npu* parent_npu_ptr);

        std::vector<std::vector<bf16>> encode(  void* image_payload_ptr);

        LM_Config config;
        npu_xclbin_manager *npu;
        qwen3_5vl_npu* parent_npu_ptr;

        unsigned int QWEN3_5_VISION_EMBED_DIM;
        unsigned int QWEN3_5_VISION_NUM_HEADS;
        unsigned int QWEN3_5_VISION_HEAD_DIM;
        unsigned int _QWEN3_5_VISION_HIDDEN_SIZE;
        unsigned int _QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE;
        unsigned int QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS;
        unsigned int QWEN3_5_VISION_NUM_LAYERS;
        float QWEN3_5_VISION_LAYER_NORM_EPSILON;
        unsigned int _QWEN3_5_MERGER_HIDDEN_SIZE;
        unsigned int _QWEN3_5_VISION_OUT_HIDDEN_SIZE;

        unsigned int Padded_QWEN3_5_VISION_HIDDEN_SIZE;
        unsigned int Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE;
        unsigned int Padded_QWEN3_5_MERGER_HIDDEN_SIZE;
        unsigned int Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE;

        uint32_t MM_tile_M;
        uint32_t MM_tile_K;
        uint32_t MM_tile_N;

        uint32_t seq_len_pad_requirement_for_MM;
        uint32_t MM_ROW_SIZE = 4;
        uint32_t MM_COL_SIZE = 8;

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

        // IF true, reorder qkv from L_Seq x (3*QWEN3_5_VISION_NUM_HEADS*QWEN3_VISION_HEAD_DIM) row major ->
        // [ 3, QWEN3_5_VISION_NUM_HEADS, L_Seq ,QWEN3_VISION_HEAD_DIM]
        bool ENABLE_QKV_REORDER = false;
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
        npu_app qkv_first_projection_app;
        bool use_overlay = false;
        npu_app o_projection_app;
        npu_app mlp_fc1_app;
        npu_app mlp_fc2_app;

        npu_app merger_fc1_app;
        npu_app merger_fc2_app;

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

        buffer<bf16> merger_fc1_weight;
        buffer<bf16> merger_fc2_weight;
        buffer<bf16> merger_norm_weight;
        buffer<bf16> merger_norm_bias;

        // the vectors
        buffer<bf16> hidden_state;
        buffer<bf16> residual_buffer;

        buffer<bf16> mm_qkv_input;
        buffer<bf16> mm_qkv_output_mha_input;
        buffer<bf16> mha_qkv_output_mm_o_input;
        buffer<bf16> mm_o_output_fc1_proj_input;
        buffer<bf16> mm_fc1_output_fc2_input;
        buffer<bf16> mm_fc2_output;

        buffer<bf16> merger_fc1_input;
        buffer<bf16> merger_fc1_output_fc2_input;
        buffer<bf16> merger_fc2_output;

        inline int round_up_to_multiple (int x, int multiple)
        {
            return ((x + multiple - 1) / multiple) * multiple;
        };
};
