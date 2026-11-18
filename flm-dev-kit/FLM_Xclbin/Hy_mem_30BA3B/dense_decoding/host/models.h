// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include <cstdlib>
#include "typedef.hpp"

namespace model_list {
    enum model_t {
        HY_MEMORY_30B_A3B_DENSE,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "HY_MEMORY_30B_A3B_DENSE") return HY_MEMORY_30B_A3B_DENSE;
        else return UNKNOWN_MODEL;
    }

    /// \brief folder with the HF config.json
    inline std::string get_config_path(model_t model) {
        const char* user = std::getenv("USER");
        static const std::string base_path = "/scratch/" + std::string(user ? user : "") + "/models/";
        switch (model) {
            case HY_MEMORY_30B_A3B_DENSE:
                return base_path + "Hy-Memory-30B-A3B";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief folder with model.q4nx
    inline std::string get_q4nx_path(model_t model) {
        switch (model) {
            case HY_MEMORY_30B_A3B_DENSE:
                return get_config_path(model) + "/q4nx";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief per-layer activations (bf16 reference, generated outside the dev kit), in ref_dir
    inline std::string get_reference_path(model_t model, const std::string& ref_dir) {
        switch (model) {
            case HY_MEMORY_30B_A3B_DENSE:
                return ref_dir + "/hy_mem_30a3b_reference.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief the same layer recomputed with the Q4_K weights from model.q4nx
    ///        (generated outside the dev kit), one file per layer, in ref_dir
    inline std::string get_q4k_reference_path(model_t model, int layer_idx, const std::string& ref_dir) {
        switch (model) {
            case HY_MEMORY_30B_A3B_DENSE:
                return ref_dir + "/hy_mem_layer_q4k_layer_" + std::to_string(layer_idx) + ".safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    /// \brief must match MODEL_TYPE in the Makefile
    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        switch (model) {
            case HY_MEMORY_30B_A3B_DENSE:
                return base_path + "HY_MEMORY_30B_A3B_DENSE" + layer_suffix;

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        switch (model) {
            case HY_MEMORY_30B_A3B_DENSE:
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
            case HY_MEMORY_30B_A3B_DENSE:
                return 11158840.0;
            default:
                return 10000.0;
        }
    }

    /// \brief dense MLP width, must match INTERMEDIATE_SIZE in the model header
    ///        (config.json intermediate_size; checked against it at load)
    inline int get_intermediate_size(model_t model) {
        return 6912;
    }

    /// \brief up (or gate) rows per proj round: one round (16 cores x 32 rows)
    ///        is one GLU slice [up 256 | gate 256], GLU_SLICE = 512 in the model header
    inline int get_up_gate_rows_per_round(model_t model) {
        return 256;
    }

    /// \brief the dense layers (config mlp_layer_types); only layer 0
    inline bool is_dense_layer(model_t model, int layer_idx) {
        return layer_idx == 0;
    }

} // namespace model_list

#endif
