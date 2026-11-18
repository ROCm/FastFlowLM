// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include <cstdlib>
#include "typedef.hpp"

namespace model_list {
    enum model_t {
        HY_MEMORY_30B_A3B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "HY_MEMORY_30B_A3B") return HY_MEMORY_30B_A3B;
        else return UNKNOWN_MODEL;
    }

    /// \brief folder with the HF config.json
    inline std::string get_config_path(model_t model) {
        const char* user = std::getenv("USER");
        static const std::string base_path = "/scratch/" + std::string(user ? user : "") + "/models/";
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return base_path + "Hy-Memory-30B-A3B";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief folder with model.q4nx
    inline std::string get_q4nx_path(model_t model) {
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return get_config_path(model) + "/q4nx";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief per-layer activations (bf16 reference, generated outside the dev kit), in ref_dir
    inline std::string get_reference_path(model_t model, const std::string& ref_dir) {
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return ref_dir + "/hy_mem_30a3b_reference.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief the same layers recomputed with the Q4_K weights from model.q4nx, attention
    ///        and MoE (generated outside the dev kit), one file per layer, in ref_dir
    inline std::string get_q4k_reference_path(model_t model, int layer_idx, const std::string& ref_dir) {
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return ref_dir + "/hy_mem_layer_q4k_layer_" + std::to_string(layer_idx) + ".safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief must match BUILD_DIR in the Makefile
    inline std::string get_layer_xclbin_path(model_t model, bool attn_tap) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        const std::string tap_suffix = attn_tap ? "_attn_tap" : "";
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return base_path + "HY_MEMORY_30B_A3B" + tap_suffix + layer_suffix;

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return true;

            default:
                return false;
        }
    }

    /// \brief share of head_dim that RoPE rotates
    inline f32 get_rope_partial_factor(model_t model) {
        return 1.0f;
    }

    inline f32 get_rope_scaling(model_t model) {
        return 1.0f;
    }

    /// \brief config.json keeps it under rope_parameters, which LM_Config does not read
    inline double get_rope_theta(model_t model) {
        switch (model) {
            case HY_MEMORY_30B_A3B:
                return 11158840.0;
            default:
                return 10000.0;
        }
    }

    /// \brief expert width, must match INTERMEDIATE_SIZE in the model header.
    ///        Hy's experts are expert_hidden_dim = 768 wide; config.json's
    ///        moe_intermediate_size (1536) is not what the weights use.
    inline int get_moe_intermediate_size(model_t model) {
        return 768;
    }

} // namespace model_list

// Proj weight blocks in the int8-mmul kernel's order. Must match Q4K_INT8_MMUL in
// kernels/q4_core.h (a mismatch gives wrong outputs, not an error).
#define Q4K_INT8_MMUL 1
#endif
