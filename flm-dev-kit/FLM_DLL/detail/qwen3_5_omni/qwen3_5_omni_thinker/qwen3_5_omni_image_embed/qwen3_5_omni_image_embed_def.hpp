// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_image_embed_def.hpp
/// \brief qwen3_5_omni thinker vision encoder config descriptor.
/// \author FastFlowLM Team
/// \note Owns the config-derived parameters for the vision encoder and fills
///       them via build(LM_Config), mirroring qwen3_6_moe_desc.
#pragma once
#include "lm_config.hpp"

/// @brief thinker vision encoder (thinker_config.vision_config)
struct qwen3_5_omni_image_desc {
    uint32_t depth;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t out_hidden_size;
    uint32_t in_channels;
    uint32_t patch_size;
    uint32_t temporal_patch_size;
    uint32_t spatial_merge_size;
    uint32_t num_position_embeddings;
    uint32_t merge_hidden_size;
    uint32_t head_dim;
    float layer_norm_eps;
    std::string hidden_act;

    uint32_t vision_mm_tile_m;
    uint32_t vision_mm_tile_k;
    uint32_t vision_mm_tile_n;

    qwen3_5_omni_image_desc() {}
    qwen3_5_omni_image_desc(LM_Config config) { build(config); }

    inline void build(LM_Config& config) {
        const nlohmann::json &vision_config = cfg_sub(config.sub("thinker_config"), "vision_config");
        const nlohmann::json &vision_extra_config = config.sub("vision_config");

        JSON_GET(depth,                   vision_config, "depth",                   0, uint32_t);
        JSON_GET(hidden_size,             vision_config, "hidden_size",             0, uint32_t);
        JSON_GET(intermediate_size,       vision_config, "intermediate_size",       0, uint32_t);
        JSON_GET(num_heads,               vision_config, "num_heads",               0, uint32_t);
        JSON_GET(out_hidden_size,         vision_config, "out_hidden_size",         0, uint32_t);
        JSON_GET(in_channels,             vision_config, "in_channels",             0, uint32_t);
        JSON_GET(patch_size,              vision_config, "patch_size",              0, uint32_t);
        JSON_GET(temporal_patch_size,     vision_config, "temporal_patch_size",     0, uint32_t);
        JSON_GET(spatial_merge_size,      vision_config, "spatial_merge_size",      0, uint32_t);
        JSON_GET(num_position_embeddings, vision_config, "num_position_embeddings", 0, uint32_t);
        JSON_GET(hidden_act,              vision_config, "hidden_act",              std::string(""), std::string);
        JSON_GET(layer_norm_eps,          vision_config, "layer_norm_eps",          1e-6f, float);
        JSON_GET(vision_mm_tile_m,        vision_extra_config, "VISION_MM_TILE_M", 0, uint32_t);
        JSON_GET(vision_mm_tile_k,        vision_extra_config, "VISION_MM_TILE_K", 0, uint32_t);
        JSON_GET(vision_mm_tile_n,        vision_extra_config, "VISION_MM_TILE_N", 0, uint32_t);

        merge_hidden_size = spatial_merge_size*spatial_merge_size* hidden_size;
        head_dim = hidden_size/num_heads;

        DEBUG_BLOCK(1,
        std::cout << "================ qwen3_5_omni_image_desc ================" << std::endl;
        std::cout << std::left << std::setw(28) << "depth"                   << " = " << depth << std::endl;
        std::cout << std::left << std::setw(28) << "hidden_size"             << " = " << hidden_size << std::endl;
        std::cout << std::left << std::setw(28) << "intermediate_size"       << " = " << intermediate_size << std::endl;
        std::cout << std::left << std::setw(28) << "num_heads"               << " = " << num_heads << std::endl;
        std::cout << std::left << std::setw(28) << "out_hidden_size"         << " = " << out_hidden_size << std::endl;
        std::cout << std::left << std::setw(28) << "in_channels"             << " = " << in_channels << std::endl;
        std::cout << std::left << std::setw(28) << "patch_size"              << " = " << patch_size << std::endl;
        std::cout << std::left << std::setw(28) << "temporal_patch_size"     << " = " << temporal_patch_size << std::endl;
        std::cout << std::left << std::setw(28) << "spatial_merge_size"      << " = " << spatial_merge_size << std::endl;
        std::cout << std::left << std::setw(28) << "num_position_embeddings" << " = " << num_position_embeddings << std::endl;
        std::cout << std::left << std::setw(28) << "hidden_act"              << " = " << hidden_act << std::endl;
        std::cout << std::left << std::setw(28) << "head_dim"              << " = " << head_dim << std::endl;
        std::cout << std::left << std::setw(28) << "layer_norm_eps"          << " = " << layer_norm_eps << std::endl;
        std::cout << "=========================================================" << std::endl;
        )
    }
};
