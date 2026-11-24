// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

#include <string>
#include "biovault_bfloat16.h"
#include "typedef.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma/gemma_npu.hpp"
#include <vector>
#include  "gemma_image_param.hpp"
// #define GEMMA_DEBUGGING 1

class GemmaImageEncoder {
    public:

        ~GemmaImageEncoder();

        template <bool print_time_detail = false, bool verify_value = false>
        buffer<bf16> encode(buffer<bf16> pixel_values, SafeTensors *reference = nullptr);

        void init_weights(SafeTensors &q4nx);

        GemmaImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance);

        LM_Config config;
        npu_xclbin_manager *npu;

    //   // num_of_image The batchsize for gemma3 vision model

        // calculate value

        uint32_t mm_m = 128;
        uint32_t mm_n = 64;
        uint32_t mm_k = 256;
        uint32_t rtp_address=4096;
        uint32_t rtp_sync_lock_id = 10;
        uint32_t MM_ROW_SIZE = 4;
        uint32_t MM_COL_SIZE = 8;
        uint32_t IS_B_ROW_MAJOR = false;
        uint32_t ENABLE_AXI4 = true;
        uint32_t ADD_BIAS = true;

        // for image attention
        uint32_t LQ_per_CT = 32;
        uint32_t LK_per_CT = 512;
        uint32_t LQ_internal = 32;
        uint32_t LK_internal = 32;
        uint32_t NUM_OF_COLUMN = 8;
        uint32_t NUM_OF_CT_PER_COLUMN = 4;
        uint32_t CU_MODE = 1; // 1-8 broadcast

        uint32_t IMAGE_HIDDEN_SIZE_PADDED;

        uint32_t IMAGE_INTERMEDIATE_SIZE_PADDED;
        uint32_t IMAGE_SEQ_LEN_AVERAGE_POOLING_PADDED;
        uint32_t sequence_length_per_image;
        uint32_t multi_model_proj_seq_len_padded;

        // debug ptr for now
        std::string model_path;

    private:

        void processConv2D_with_Bias(
            buffer< biovault::bfloat16_t> &conv2dOutput,
            const  biovault::bfloat16_t *image_ptr, int num_of_image, int C_in, int H_in, int W_in,
            int conv2d_stride,
            bool verify_value,
            bool print_time_details,
            SafeTensors *reference
        );

        void processEmbedding(
             biovault::bfloat16_t * __restrict data, //[batch_size, seq_len, hidden_size_padded_to_512]

            const int batch_size,
            const int sequence_length,
            const int hidden_size,
            const int hidden_size_padded,
            bool verify_value,
            bool print_time_details,
            SafeTensors *reference
        );

        void processEncoderLayer(
             biovault::bfloat16_t * __restrict data, //[batch_size, seq_len_padded, hidden_size_padded]
            const int batch_size, const int sequence_length, const int hidden_size,
            const int sequence_length_padded, // this is the padded sequence length
            const int hidden_size_padded, // this is the padded hidden size
            const int layer_id,
            const size_t num_heads,
            const size_t head_dim, const size_t intermediate_size,
            const size_t padded_intermediate_size,
            const float layernorm_eps,
            bool verify_value,
            bool print_time_details,
            SafeTensors *reference
        );

        void multi_head_attention(
             biovault::bfloat16_t* hidden_states,
            size_t layer_id,
            const size_t num_heads,
            const size_t head_dim,
            const size_t batch_size,
            const size_t sequence_len,
            const size_t sequence_len_padded,
            const size_t hidden_size_padded,
            bool print_details,
            SafeTensors *reference
        );
        void mlp_layer_avx512(
            biovault::bfloat16_t* hidden_space,             // [batch_size * seq_len, padded_hidden_size]
            size_t layer_id,
            int batch_size,
            int seq_len,
            int hidden_size,
            int intermediate_size,
            int padded_hidden_size,
            int padded_intermediate_size,
            bool print_time_details,
            SafeTensors *reference
        ) ;

        // The weights
        buffer<bf16> conv2d_kernels_weights;
        buffer<bf16> conv2d_bias;

        buffer<bf16> image_embedding_weight;

        std::vector<buffer<bf16>> encoder_layer_norm_1_weights;
        std::vector<buffer<bf16>> encoder_layer_norm_1_bias;

        std::vector<buffer<bf16>> encoder_layer_norm_2_weights;
        std::vector<buffer<bf16>> encoder_layer_norm_2_bias;

        // Attention part
        std::vector<buffer< biovault::bfloat16_t>> attn_q_weights;
        std::vector<buffer< biovault::bfloat16_t>> attn_k_weights;
        std::vector<buffer< biovault::bfloat16_t>> attn_v_weights;
        std::vector<buffer< biovault::bfloat16_t>> attn_output_weights;

        //MLP
        // First INTERMEDIATE_PADDED for bias, then INTERMEDIATE_PADDED x HIDDEN_SIZE_PADDED for weights
        std::vector<buffer< biovault::bfloat16_t>> fc1_weights;
        std::vector<buffer< biovault::bfloat16_t>> fc2_weights;

        // post layernorm
        buffer<bf16> post_layer_norm_weight;
        buffer<bf16> post_layer_norm_bias;

        //gemma rms norm
        buffer<bf16> gemma_rms_norm_weight;

        //MM projection
        buffer<bf16> multi_model_projection_weight;

        npu_app_manager* fla;
        npu_app_manager* proj;

        // define the necessary bitstream no
        npu_app flash_attention;
        npu_app q_projection;
        npu_app k_projection;
        npu_app v_projection;
        npu_app output_projection;

        flm_rt::runlist qkv_projs;

        // turn this into a list instead

        npu_app fc1;
        npu_app fc2;
        npu_app multi_model_proj;

        // preallocate buffers, so don't allocate buffer at runtime

        // preallocate buffer for the attention
        buffer< biovault::bfloat16_t> q_hidden_in_bo;
        buffer<bf16> q_projected_padded;

        // buffer< biovault::bfloat16_t> k_hidden_in_bo;
        buffer<bf16> k_projected_padded;

        // buffer< biovault::bfloat16_t> v_hidden_in_bo;
        buffer<bf16> v_projected_padded;

        buffer<bf16> flash_attent_concatenated_out;

        buffer<bf16> projected_output;

        // preallocate buffer for mlp layer
        buffer<bf16> fc1_hidden;
        buffer<bf16> fc1_res;
        buffer<bf16> fc2_res;

        buffer<bf16> multi_model_proj_hidden;
        buffer<bf16> multi_model_proj_res;
        // residaul
        std::vector<bf16> residual_1;
        std::vector<bf16> residual_2;
};
