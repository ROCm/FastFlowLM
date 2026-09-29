#pragma once
#include "typedef.hpp"
#include <string>
#include <vector>
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma4e/gemma4e_npu.hpp"

#include "vision/norm.hpp"

class Gemma4e_AudioEncoder{
    public:
        ~Gemma4e_AudioEncoder();
        void init_weights(SafeTensors &q4nx);
        Gemma4e_AudioEncoder(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_npu* parent_npu_ptr);

        void ffn_layer(

            buffer<bf16> &ffn_up_proj_input,
            buffer<bf16> &ffn_up_proj_output_down_input,
            buffer<bf16> &ffn_down_proj_output,

            buffer<bf16> &cur_ffn_norm_weight,  // ffn_norm or ffn_norm_1
            buffer<bf16> &cur_ffn_post_norm_weight,// ffn_post_norm_weight or ffn_post_norm_1_weight
            buffer<bf16> &cur_ffn_up_weight, // audio_ffn_up_weight or audio_ffn_up_1_weight
            buffer<bf16> &cur_ffn_down_weight, // audio_ffn_down_weight
            int seq_len, int seq_len_padded,

            bf16 cur_audio_ffn_up_input_min, bf16 cur_audio_ffn_up_input_max,
            bf16 cur_audio_ffn_up_output_min, bf16 cur_audio_ffn_up_output_max,
            bf16 cur_audio_ffn_down_input_min, bf16 cur_audio_ffn_down_input_max,
            bf16 cur_audio_ffn_down_output_min, bf16 cur_audio_ffn_down_output_max

        );
        void conv1d_layer(
            int layer_idx,
            int seq_len, int seq_len_padded,
            std::vector<int> &seq_len_per_audio, std::vector<int> &start_seq_len_index_per_audio,
            buffer<bf16> &conv1d_start_proj_input, buffer<bf16> &conv1d_start_proj_output,
            buffer<bf16> &conv1d_input, buffer<bf16> &conv1d_output,
            buffer<bf16> &conv1d_end_proj_input, buffer<bf16> &conv1d_end_proj_output,

            bf16 conv1d_start_input_min, bf16 conv1d_start_input_max,
            bf16 conv1d_start_output_min, bf16 conv1d_start_output_max,
            bf16 conv1d_end_input_min, bf16 conv1d_end_input_max,
            bf16 conv1d_end_output_min, bf16 conv1d_end_output_max,

            gemma4e_audio_payload_t* audio_payload,
            SafeTensors *reference_safetensor
        );

        std::vector<bf16> encode(  void* audio_payload_ptr);

        LM_Config config;
        npu_xclbin_manager *npu;
        gemma4e_npu* parent_npu_ptr;

        uint32_t MM_tile_M;
        uint32_t MM_tile_K;
        uint32_t MM_tile_N;

        float Gemma4E_Audio_residual_weight;
        float Gemma4E_Audio_q_scale;
        float Gemma4E_Audio_k_scale;
        int Gemma4E_Audio_attention_head_dim;
        int Gemma4E_Audio_padded_requirement_for_conv1d;

        uint32_t seq_len_pad_requirement_for_MM;
        uint32_t MM_ROW_SIZE = 4;
        uint32_t MM_COL_SIZE = 8;

        unsigned int Padded_GEMMA4E_Audio_HIDDEN_SIZE;
        unsigned int Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE;
        unsigned int Padded_Gemma4E_Audio_Multimodal_Output_SIZE;
        unsigned int Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE;
        unsigned int Padded_Gemma4E_Audio_num_attention_heads;

        inline int round_up_to_multiple (int x, int multiple)
        {
            return ((x + multiple - 1) / multiple) * multiple;
        };

        // no reorder for MM
        bool ENABLE_QKV_REORDER = false;

        // for MM runtime sequence
        uint32_t rtp_address = 4096; // offset right after stack size
        uint32_t rtp_sync_lock_id = 10; // the rtp sync lock
        bool ENABLE_AXI4 = true;
        bool IS_B_ROW_MAJOR = false;

        std::string model_path;
        npu_app_manager *proj;
        npu_app_manager *proj_high_precision;
        npu_app_manager *conv1d;

        npu_app conv1d_app;

        npu_app sub_sampleConvProjection_app;
        npu_app q_proj_app;
        npu_app k_proj_app;
        npu_app k_relative_proj_app;
        npu_app v_proj_app;
        npu_app o_proj_app;

        npu_app ffn_down_proj_app; // share for both first and second FFN in the layer
        npu_app ffn_up_proj_app;
        npu_app conv1d_start_proj_app;
        npu_app conv1d_end_proj_app;
        npu_app audio_pre_encode_proj_app;
        npu_app audio_to_language_proj_app;

        std::vector<bf16> residual;
        std::vector<bf16> hidden_state;

        std::vector<bf16> q_blocks;
        std::vector<bf16> k_blocks;
        std::vector<bf16> v_blocks;

        // the weights
        buffer<bf16> audio_embedding_projection_weight;

        buffer<bf16> audio_subsample_conv2d_weight_0;
        buffer<bf16> audio_subsample_conv2d_norm_weight_0;
        buffer<bf16> audio_subsample_conv2d_weight_1;
        buffer<bf16> audio_subsample_conv2d_norm_weight_1;

        buffer<bf16> audio_pre_encode_weight;
        buffer<bf16> audio_to_language_projection_weight;
        // weights for each layer
        std::vector<buffer<bf16>> audio_attn_k_weight;
        std::vector<buffer<bf16>> audio_attn_k_rel_weight;
        std::vector<buffer<bf16>> audio_attn_v_weight;
        std::vector<buffer<bf16>> audio_attn_q_weight;
        std::vector<buffer<bf16>> audio_attn_o_weight;
        std::vector<buffer<bf16>> audio_conv1d_weight;

        std::vector<buffer<bf16>> audio_conv_pw_1_weight;
        std::vector<buffer<bf16>> audio_conv_pw_2_weight;

        std::vector<buffer<bf16>> audio_ffn_down_weight;
        std::vector<buffer<bf16>> audio_ffn_up_weight;
        std::vector<buffer<bf16>> audio_ffn_down_1_weight;
        std::vector<buffer<bf16>> audio_ffn_up_1_weight;

        std::vector<buffer<bf16>> attn_post_norm_weight;
        std::vector<buffer<bf16>> attn_pre_norm_weight;
        std::vector<buffer<bf16>> conv_norm_weight;
        std::vector<buffer<bf16>> norm_conv_weight;
        std::vector<buffer<bf16>> ffn_norm_weight;
        std::vector<buffer<bf16>> ffn_norm_1_weight;
        std::vector<buffer<bf16>> ffn_post_norm_weight;
        std::vector<buffer<bf16>> ffn_post_norm_1_weight;
        std::vector<buffer<bf16>> norm2_weight;
        std::vector<buffer<bf16>> per_dim_scale_with_softplus_weight;

        std::vector<bf16> audio_k_input_min;
        std::vector<bf16> audio_k_input_max;
        std::vector<bf16> audio_v_input_min;
        std::vector<bf16> audio_v_input_max;
        std::vector<bf16> audio_q_input_min;
        std::vector<bf16> audio_q_input_max;
        std::vector<bf16> audio_o_input_min;
        std::vector<bf16> audio_o_input_max;
        std::vector<bf16> audio_conv_pw1_input_min;
        std::vector<bf16> audio_conv_pw1_input_max;
        std::vector<bf16> audio_conv_pw2_input_min;
        std::vector<bf16> audio_conv_pw2_input_max;
        std::vector<bf16> audio_ffn_down_input_min;
        std::vector<bf16> audio_ffn_down_input_max;
        std::vector<bf16> audio_ffn_up_input_min;
        std::vector<bf16> audio_ffn_up_input_max;
        std::vector<bf16> audio_ffn_down_1_input_min;
        std::vector<bf16> audio_ffn_down_1_input_max;
        std::vector<bf16> audio_ffn_up_1_input_min;
        std::vector<bf16> audio_ffn_up_1_input_max;

        std::vector<bf16> audio_k_output_min;
        std::vector<bf16> audio_k_output_max;
        std::vector<bf16> audio_v_output_min;
        std::vector<bf16> audio_v_output_max;
        std::vector<bf16> audio_q_output_min;
        std::vector<bf16> audio_q_output_max;
        std::vector<bf16> audio_o_output_min;
        std::vector<bf16> audio_o_output_max;
        std::vector<bf16> audio_conv1d_output_min;
        std::vector<bf16> audio_conv1d_output_max;
        std::vector<bf16> audio_conv_pw1_output_min;
        std::vector<bf16> audio_conv_pw1_output_max;
        std::vector<bf16> audio_conv_pw2_output_min;
        std::vector<bf16> audio_conv_pw2_output_max;
        std::vector<bf16> audio_ffn_down_output_min;
        std::vector<bf16> audio_ffn_down_output_max;
        std::vector<bf16> audio_ffn_up_output_min;
        std::vector<bf16> audio_ffn_up_output_max;
        std::vector<bf16> audio_ffn_down_1_output_min;
        std::vector<bf16> audio_ffn_down_1_output_max;
        std::vector<bf16> audio_ffn_up_1_output_min;
        std::vector<bf16> audio_ffn_up_1_output_max;
};
