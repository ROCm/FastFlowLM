// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include <cstdlib>
#include "rope_weights.h"

namespace model_list {
    enum model_t {
        QWEN3_6_35B_A3B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "QWEN3_6_35B_A3B") return QWEN3_6_35B_A3B;
        else return UNKNOWN_MODEL;
    }

    inline std::string get_model_path(model_t model) {
        const char* user = std::getenv("USER");
        static const std::string base_path = "/scratch/" + std::string(user ? user : "") + "/models/";
        switch (model) {
            case QWEN3_6_35B_A3B:
                return base_path + "Qwen3.6-35B-A3B-NPU2";

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_detail_reference_path(model_t model, bool is_linear_layer) {
        std::string model_path = get_model_path(model);
        std::string base_path = model_path + "/";
        switch (model) {
            case QWEN3_6_35B_A3B:
                if (is_linear_layer)
                    return base_path + "gated_deltanet_35B_A3B.safetensors";
                else
                    return base_path + "gated_attention_35B_A3B.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_reference_path(model_t model) {
        std::string model_path = get_model_path(model);
        std::string base_path = model_path + "/";
        switch (model) {
            case QWEN3_6_35B_A3B:
                return base_path + "qwen3_6_35B_A3B_ref.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        switch (model) {
            case QWEN3_6_35B_A3B:
                return base_path + "QWEN3_6_35B_A3B" + layer_suffix;

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        switch (model) {
            case QWEN3_6_35B_A3B:
                return true;

            default:
                return false;
        }
    }

    inline f32 get_rope_scaling(model_t model) {
        return 1.0f;
    }

    inline uint32_t get_rms_addr(model_t model) {
        switch (model) {
            case QWEN3_6_35B_A3B:
                return 53248;
            default:
                return 0;
        }
    }

} // namespace model_list
#endif
