// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_audio_embed.hpp
/// \brief qwen3_5_omni audio embedder — simple stub.
/// \author FastFlowLM Team
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "models/qwen3_5_omni/qwen3_5_omni.hpp"
#include "qwen3_5_omni_audio_embed_def.hpp"
#include "conv2d.hpp"
#include "sinusoids_position_embedding.hpp"
#include "npu_sequences/window_attention_sequence.hpp"
class qwen3_5_omni_audio_embed {
public:
    qwen3_5_omni_audio_embed(LM_Config config, npu_xclbin_manager *npu_instance);
    ~qwen3_5_omni_audio_embed();

    buffer<bf16> forward(qwen3_5_omni_audio_payload_t& payload);
    void load_weights(SafeTensors& q4nx);

    unsigned int padded_audio_hidden_size;
    unsigned int padded_audio_mlp_intermediate_size;
    unsigned int padded_audio_out_hidden_size;
    unsigned int padded_audio_conv_out_k_size;

    uint32_t MM_tile_M;
    uint32_t MM_tile_K;
    uint32_t MM_tile_N;

    uint32_t seq_len_pad_requirement_for_MM;
    uint32_t MM_ROW_SIZE = 4;
    uint32_t MM_COL_SIZE = 8;

    // no reorder for MM
    bool ENABLE_QKV_REORDER = false;

    // for MM runtime sequence
    uint32_t rtp_address = 4096; // offset right after stack size
    uint32_t rtp_sync_lock_id = 10; // the rtp sync lock
    bool ENABLE_AXI4 = true;
    bool IS_B_ROW_MAJOR = false;

    // parameter for audio attention kernel
    uint32_t vision_num_of_columns = 8;
    uint32_t vision_num_of_rows = 4;

    npu_app_manager* window_fla;
    npu_app_manager* proj;
    npu_app_manager* proj_high;

    npu_app flash_window_attention_app;
    npu_app conv_out_proj;
    npu_app q_proj_app;
    npu_app k_proj_app;
    npu_app v_proj_app;
    npu_app o_proj_app;
    npu_app mlp_fc1_app;
    npu_app mlp_fc2_app;

    npu_app merger_fc1_app;
    npu_app merger_fc2_app;

    // Pre-transposed conv weights (prepared once in load_weights).
    audio_conv_weights conv_wt;

    SinusoidsPositionEmbedding pos_emb;

    buffer<bf16> conv2d_1_weight;
    buffer<bf16> conv2d_1_bias;
    buffer<bf16> conv2d_2_weight;
    buffer<bf16> conv2d_2_bias;
    buffer<bf16> conv2d_3_weight;
    buffer<bf16> conv2d_3_bias;
    buffer<bf16> conv2d_4_weight;
    buffer<bf16> conv2d_4_bias;

    buffer<bf16> audio_conv_out_weight;

    std::vector<buffer<bf16>> norm_1_weight;
    std::vector<buffer<bf16>> norm_1_bias;
    std::vector<buffer<bf16>> norm_2_weight;
    std::vector<buffer<bf16>> norm_2_bias;

    std::vector<buffer<bf16>> q_weight;
    std::vector<buffer<bf16>> k_weight;
    std::vector<buffer<bf16>> v_weight;
    std::vector<buffer<bf16>> o_weight;
    std::vector<buffer<bf16>> mlp_fc1_weight;
    std::vector<buffer<bf16>> mlp_fc2_weight;

    buffer<bf16> merger_fc1_weight;
    buffer<bf16> merger_fc2_weight;

    buffer<bf16> merger_norm_weight;
    buffer<bf16> merger_norm_bias;

    buffer<bf16> hidden_states;
    buffer<bf16> residual_buffer;

    /// @brief audio encoder config (thinker_config.audio_config)
    qwen3_5_omni_audio_desc desc;

    inline int round_up_to_multiple (int x, int multiple)
    {
        return ((x + multiple - 1) / multiple) * multiple;
    };

private:
    LM_Config config;
    npu_xclbin_manager *npu;
};
