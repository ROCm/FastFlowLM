// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_audio_embed_def.hpp
/// \brief qwen3_5_omni thinker audio encoder config descriptor.
/// \author FastFlowLM Team
/// \note Owns the config-derived parameters for the audio encoder and fills
///       them via build(LM_Config), mirroring qwen3_6_moe_desc.
#pragma once
#include "lm_config.hpp"

/// @brief thinker audio encoder (thinker_config.audio_config)
struct qwen3_5_omni_audio_desc {
    uint32_t d_model;
    uint32_t encoder_attention_heads;
    uint32_t encoder_ffn_dim;
    uint32_t encoder_layers;
    uint32_t num_hidden_layers;
    uint32_t num_mel_bins;
    uint32_t output_dim;
    uint32_t downsample_hidden_size;
    uint32_t max_source_positions;
    uint32_t n_window;
    uint32_t n_window_infer;
    uint32_t conv_chunksize;
    uint32_t head_dim;
    int conv_out_k;
    float layer_norm_eps;

    uint32_t audio_mm_tile_m;
    uint32_t audio_mm_tile_k;
    uint32_t audio_mm_tile_n;

    // audio_input_config (FLM front-end parameters)
    uint32_t sampling_rate;
    uint32_t downsample_time;
    uint32_t downsample_chunk_size;
    uint32_t audio_timestamp_interval;
    uint32_t audio_hop_length;
    uint32_t audio_window_attention_length;

    std::string activation_function;

    qwen3_5_omni_audio_desc() {}
    qwen3_5_omni_audio_desc(LM_Config config) { build(config); }

    inline void build(LM_Config& config) {
        const nlohmann::json &audio_config = cfg_sub(config.sub("thinker_config"), "audio_config");
        const nlohmann::json &customize_audio_config = config.sub("audio_input_config");
        JSON_GET(d_model,                 audio_config, "d_model",                 0, uint32_t);
        JSON_GET(encoder_attention_heads, audio_config, "encoder_attention_heads", 0, uint32_t);
        JSON_GET(encoder_ffn_dim,         audio_config, "encoder_ffn_dim",         0, uint32_t);
        JSON_GET(encoder_layers,          audio_config, "encoder_layers",          0, uint32_t);
        JSON_GET(num_hidden_layers,       audio_config, "num_hidden_layers",       0, uint32_t);
        JSON_GET(num_mel_bins,            audio_config, "num_mel_bins",            0, uint32_t);
        JSON_GET(output_dim,              audio_config, "output_dim",              0, uint32_t);
        JSON_GET(downsample_hidden_size,  audio_config, "downsample_hidden_size",  0, uint32_t);
        JSON_GET(max_source_positions,    audio_config, "max_source_positions",    0, uint32_t);
        JSON_GET(n_window,                audio_config, "n_window",                0, uint32_t);
        JSON_GET(n_window_infer,          audio_config, "n_window_infer",          0, uint32_t);
        JSON_GET(conv_chunksize,          audio_config, "conv_chunksize",          0, uint32_t);
        JSON_GET(activation_function,     audio_config, "activation_function",     std::string(""), std::string);
        JSON_GET(layer_norm_eps,          audio_config, "layer_norm_eps",          1e-5f, float);

        JSON_GET(audio_mm_tile_m,         customize_audio_config, "AUDIO_MM_TILE_M",         0, uint32_t);
        JSON_GET(audio_mm_tile_k,         customize_audio_config, "AUDIO_MM_TILE_K",         0, uint32_t);
        JSON_GET(audio_mm_tile_n,         customize_audio_config, "AUDIO_MM_TILE_N",         0, uint32_t);
        JSON_GET(sampling_rate,                 customize_audio_config, "sampling_rate",                 0, uint32_t);
        JSON_GET(downsample_time,               customize_audio_config, "downsample_time",               0, uint32_t);
        JSON_GET(downsample_chunk_size,         customize_audio_config, "downsample_chunk_size",         0, uint32_t);
        JSON_GET(audio_timestamp_interval,      customize_audio_config, "audio_timestamp_interval",      0, uint32_t);
        JSON_GET(audio_hop_length,              customize_audio_config, "audio_hop_length",              0, uint32_t);
        JSON_GET(audio_window_attention_length, customize_audio_config, "audio_window_attention_length", 0, uint32_t);

        conv_out_k = downsample_hidden_size * (((((num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2 + 1) / 2);

        head_dim = d_model/ encoder_attention_heads;

        DEBUG_BLOCK(1,
        std::cout << "=============== qwen3_5_omni_audio_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "d_model"                 << " = " << d_model << std::endl;
        std::cout << std::left << std::setw(28) << "encoder_attention_heads" << " = " << encoder_attention_heads << std::endl;
        std::cout << std::left << std::setw(28) << "encoder_ffn_dim"         << " = " << encoder_ffn_dim << std::endl;
        std::cout << std::left << std::setw(28) << "encoder_layers"          << " = " << encoder_layers << std::endl;
        std::cout << std::left << std::setw(28) << "num_hidden_layers"       << " = " << num_hidden_layers << std::endl;
        std::cout << std::left << std::setw(28) << "num_mel_bins"            << " = " << num_mel_bins << std::endl;
        std::cout << std::left << std::setw(28) << "output_dim"              << " = " << output_dim << std::endl;
        std::cout << std::left << std::setw(28) << "downsample_hidden_size"  << " = " << downsample_hidden_size << std::endl;
        std::cout << std::left << std::setw(28) << "max_source_positions"    << " = " << max_source_positions << std::endl;
        std::cout << std::left << std::setw(28) << "n_window"                << " = " << n_window << std::endl;
        std::cout << std::left << std::setw(28) << "n_window_infer"          << " = " << n_window_infer << std::endl;
        std::cout << std::left << std::setw(28) << "conv_chunksize"          << " = " << conv_chunksize << std::endl;
        std::cout << std::left << std::setw(28) << "activation_function"     << " = " << activation_function << std::endl;
        std::cout << std::left << std::setw(28) << "layer_norm_eps"          << " = " << layer_norm_eps << std::endl;
        std::cout << std::left << std::setw(28) << "sampling_rate"           << " = " << sampling_rate << std::endl;
        std::cout << std::left << std::setw(28) << "downsample_time"         << " = " << downsample_time << std::endl;
        std::cout << std::left << std::setw(28) << "downsample_chunk_size"   << " = " << downsample_chunk_size << std::endl;
        std::cout << std::left << std::setw(28) << "audio_timestamp_interval"<< " = " << audio_timestamp_interval << std::endl;
        std::cout << std::left << std::setw(28) << "audio_hop_length"        << " = " << audio_hop_length << std::endl;
        std::cout << std::left << std::setw(28) << "audio_window_attention_length" << " = " << audio_window_attention_length << std::endl;
        std::cout << "=========================================================" << std::endl;
        )
    }
};
