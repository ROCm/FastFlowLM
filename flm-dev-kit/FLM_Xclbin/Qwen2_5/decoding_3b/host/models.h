// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include "rope_weights.h"

namespace model_list {
    enum model_t {
        QWEHN2_3B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "QWEN2_3B") return QWEHN2_3B;
        else return UNKNOWN_MODEL;
    }

    inline std::string get_model_path(model_t model) {
        static const std::string base_path = "q4nx_files/";
        switch (model) {
            case QWEHN2_3B:
                return base_path + "Qwen2.5-3B-NPU2";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_reference_path(model_t model) {
        static const std::string base_path = "./";
        switch (model) {
            case QWEHN2_3B:
                return base_path + "QWEN2_3B_ref.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        switch (model) {
            case QWEHN2_3B:
                return base_path + "QWEN2_3B" + layer_suffix;
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        switch (model) {
            case QWEHN2_3B:
                return false;
            default:
                return false;
        }
    }

    inline f32* get_model_rotary_frequencies(model_t model) {
        switch (model) {
            case QWEHN2_3B:
                return &(qwen2_3b_rope[0]);
            default:
                return nullptr;
        }
    }

    inline f32 get_rope_scaling(model_t model) {
        return 1.0f;
    }
    inline uint32_t get_rms_addr(model_t model) {
        return 53248;
    }

} // namespace model_list
#endif