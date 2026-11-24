// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_talker_def.hpp
/// \brief qwen3_5_omni talker head + text backbone config descriptor.
/// \author FastFlowLM Team
/// \note Owns the config-derived parameters for the talker part of the model
///       and fills them via build(LM_Config), mirroring qwen3_6_moe_desc.
#pragma once
#include "lm_config.hpp"

/// @brief talker head (talker_config) plus its text backbone (talker_config.text_config)
struct qwen3_5_omni_talker_desc {
    // talker_config
    uint32_t accept_hidden_layer;
    uint32_t num_code_groups;
    uint32_t thinker_hidden_size;
    uint32_t spatial_merge_size;
    int codec_bos_id;
    int codec_eos_token_id;
    int codec_pad_id;

    // talker_config.text_config
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t moe_intermediate_size;
    uint32_t head_dim;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t num_hidden_layers;
    uint32_t vocab_size;
    uint32_t text_vocab_size;
    uint32_t full_attention_interval;
    uint32_t linear_conv_kernel_dim;
    uint32_t linear_key_head_dim;
    uint32_t linear_value_head_dim;
    uint32_t linear_num_key_heads;
    uint32_t linear_num_value_heads;
    uint32_t max_position_embeddings;
    float rms_norm_eps;

    qwen3_5_omni_talker_desc() {}
    qwen3_5_omni_talker_desc(LM_Config config) { build(config); }

    inline void build(LM_Config& config) {
        const nlohmann::json &talker_config = config.sub("talker_config");
        const nlohmann::json &text_config = cfg_sub(talker_config, "text_config");

        JSON_GET(accept_hidden_layer,     talker_config, "accept_hidden_layer",     0, uint32_t);
        JSON_GET(num_code_groups,         talker_config, "num_code_groups",         0, uint32_t);
        JSON_GET(thinker_hidden_size,     talker_config, "thinker_hidden_size",     0, uint32_t);
        JSON_GET(spatial_merge_size,      talker_config, "spatial_merge_size",      0, uint32_t);
        JSON_GET(codec_bos_id,            talker_config, "codec_bos_id",            0, int);
        JSON_GET(codec_eos_token_id,      talker_config, "codec_eos_token_id",      0, int);
        JSON_GET(codec_pad_id,            talker_config, "codec_pad_id",            0, int);

        JSON_GET(hidden_size,             text_config, "hidden_size",             0, uint32_t);
        JSON_GET(intermediate_size,       text_config, "intermediate_size",       0, uint32_t);
        JSON_GET(moe_intermediate_size,   text_config, "moe_intermediate_size",   0, uint32_t);
        JSON_GET(head_dim,                text_config, "head_dim",                0, uint32_t);
        JSON_GET(num_attention_heads,     text_config, "num_attention_heads",     0, uint32_t);
        JSON_GET(num_key_value_heads,     text_config, "num_key_value_heads",     0, uint32_t);
        JSON_GET(num_hidden_layers,       text_config, "num_hidden_layers",       0, uint32_t);
        JSON_GET(vocab_size,              text_config, "vocab_size",              0, uint32_t);
        JSON_GET(text_vocab_size,         text_config, "text_vocab_size",         0, uint32_t);
        JSON_GET(full_attention_interval, text_config, "full_attention_interval", 0, uint32_t);
        JSON_GET(linear_conv_kernel_dim,  text_config, "linear_conv_kernel_dim",  0, uint32_t);
        JSON_GET(linear_key_head_dim,     text_config, "linear_key_head_dim",     0, uint32_t);
        JSON_GET(linear_value_head_dim,   text_config, "linear_value_head_dim",   0, uint32_t);
        JSON_GET(linear_num_key_heads,    text_config, "linear_num_key_heads",    0, uint32_t);
        JSON_GET(linear_num_value_heads,  text_config, "linear_num_value_heads",  0, uint32_t);
        JSON_GET(max_position_embeddings, text_config, "max_position_embeddings", 0, uint32_t);
        JSON_GET(rms_norm_eps,            text_config, "rms_norm_eps",            0.0f, float);

        DEBUG_BLOCK(1,
        std::cout << "================ qwen3_5_omni_talker_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "accept_hidden_layer"     << " = " << accept_hidden_layer << std::endl;
        std::cout << std::left << std::setw(28) << "num_code_groups"         << " = " << num_code_groups << std::endl;
        std::cout << std::left << std::setw(28) << "thinker_hidden_size"     << " = " << thinker_hidden_size << std::endl;
        std::cout << std::left << std::setw(28) << "spatial_merge_size"      << " = " << spatial_merge_size << std::endl;
        std::cout << std::left << std::setw(28) << "codec_bos_id"            << " = " << codec_bos_id << std::endl;
        std::cout << std::left << std::setw(28) << "codec_eos_token_id"      << " = " << codec_eos_token_id << std::endl;
        std::cout << std::left << std::setw(28) << "codec_pad_id"            << " = " << codec_pad_id << std::endl;
        std::cout << std::left << std::setw(28) << "hidden_size"             << " = " << hidden_size << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size"       << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "moe_intermediate_size"   << " = " << moe_intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "head_dim"                << " = " << head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "num_attention_heads"     << " = " << num_attention_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_key_value_heads"     << " = " << num_key_value_heads << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"       << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "vocab_size"              << " = " << vocab_size << std::endl;
        std::cout << std::left << std::setw(28) << "text_vocab_size"         << " = " << text_vocab_size << std::endl;
        std::cout << std::left << std::setw(28) << "full_attention_interval" << " = " << full_attention_interval << std::endl;
        std::cout << std::left << std::setw(28) << "linear_conv_kernel_dim"  << " = " << linear_conv_kernel_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_key_head_dim"     << " = " << linear_key_head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_value_head_dim"   << " = " << linear_value_head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_key_heads"    << " = " << linear_num_key_heads << std::endl;
        std::cout << std::left << std::setw(28) << "linear_num_value_heads"  << " = " << linear_num_value_heads << std::endl;
        std::cout << std::left << std::setw(28) << "max_position_embeddings" << " = " << max_position_embeddings << std::endl;
        std::cout << std::left << std::setw(28) << "rms_norm_eps"            << " = " << rms_norm_eps << std::endl;
        std::cout << "==========================================================" << std::endl;
        )
    }
};
