#pragma once
#include "typedef.hpp"
#include <string>
#include <vector>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma4e/gemma4e_npu.hpp"

#include "vision/norm.hpp"

class Gemma4e_ImageEncoder{

    public:

        ~Gemma4e_ImageEncoder();
        void init_weights(SafeTensors &q4nx);
        Gemma4e_ImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_npu* parent_npu_ptr);

        std::vector<bf16> encode(  void* image_payload_ptr);

        LM_Config config;
        npu_xclbin_manager *npu;
        gemma4e_npu* parent_npu_ptr;

        unsigned int Padded_GEMMA4E_VISION_HIDDEN_SIZE;
        unsigned int Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE;
        unsigned int Padded_GEMMA4E_VISION_OUT_HIDDEN_SIZE;

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
        npu_app_manager* proj_high_precision;

        // define the necessary bitstream no
        npu_app flash_attention_app;
        npu_app patch_embedder_posisiton_embedding_dim_0_app;
        npu_app patch_embedder_posisiton_embedding_dim_1_app;
        npu_app patch_embedding_app;
        npu_app q_proj_app;
        npu_app k_proj_app;
        npu_app v_proj_app;
        npu_app o_proj_app;
        npu_app gate_proj_app;
        npu_app up_proj_app;
        npu_app down_proj_app;

        npu_app vision_to_language_input_projection_app;

        // The weights

        buffer<bf16> patch_embedder_position_embedding_table;
        buffer<bf16> patch_embd_weight;
        buffer<bf16> vision_to_language_input_projection_weight; // project into language hidden space,

        std::vector<buffer<bf16>> q_proj_weight;
        std::vector<buffer<bf16>> k_proj_weight;
        std::vector<buffer<bf16>> v_proj_weight;
        std::vector<buffer<bf16>> o_proj_weight;
        std::vector<buffer<bf16>> gate_proj_weight;
        std::vector<buffer<bf16>> up_proj_weight;
        std::vector<buffer<bf16>> down_proj_weight;

        std::vector<bf16> hidden_state;
        std::vector<bf16> residual_buffer;
        std::vector<buffer<bf16>> q_norm_weight;
        std::vector<buffer<bf16>> k_norm_weight;
        std::vector<buffer<bf16>> post_o_norm_weight;
        std::vector<buffer<bf16>> post_ffn_norm_weight;
        std::vector<buffer<bf16>> layer_norm_1_weight;
        std::vector<buffer<bf16>> layer_norm_2_weight;

        std::vector<bf16> input_q_max;
        std::vector<bf16> input_q_min;
        std::vector<bf16> input_k_max;
        std::vector<bf16> input_k_min;
        std::vector<bf16> input_v_max;
        std::vector<bf16> input_v_min;
        std::vector<bf16> input_o_max;
        std::vector<bf16> input_o_min;
        std::vector<bf16> input_gate_max;
        std::vector<bf16> input_gate_min;
        std::vector<bf16> input_up_max;
        std::vector<bf16> input_up_min;
        std::vector<bf16> input_down_max;
        std::vector<bf16> input_down_min;

        std::vector<bf16> output_q_max;
        std::vector<bf16> output_q_min;
        std::vector<bf16> output_k_max;
        std::vector<bf16> output_k_min;
        std::vector<bf16> output_v_max;
        std::vector<bf16> output_v_min;
        std::vector<bf16> output_o_max;
        std::vector<bf16> output_o_min;
        std::vector<bf16> output_gate_max;
        std::vector<bf16> output_gate_min;
        std::vector<bf16> output_up_max;
        std::vector<bf16> output_up_min;
        std::vector<bf16> output_down_max;
        std::vector<bf16> output_down_min;

        inline int round_up_to_multiple (int x, int multiple)
        {
            return ((x + multiple - 1) / multiple) * multiple;
        };
};
