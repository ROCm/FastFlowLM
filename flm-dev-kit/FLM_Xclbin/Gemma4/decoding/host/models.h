// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include "rope_weights.h"

namespace model_list {
    enum model_t {
        GEMMA4_E2B,
        GEMMA4_E4B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "GEMMA4_E2B") return GEMMA4_E2B;
        else if (model_name == "GEMMA4_E4B") return GEMMA4_E4B;
        else return UNKNOWN_MODEL;
    }

    inline std::string get_model_path(model_t model) {
        static const std::string base_path = "q4nx_files/";
        switch (model) {
            case GEMMA4_E2B:
                return base_path + "Gemma4-E2B-IT-NPU2";
            case GEMMA4_E4B:
                return base_path + "Gemma4-E4B-IT-NPU2";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_reference_path(model_t model) {
        static const std::string base_path = "references/";
        switch (model) {
            case GEMMA4_E2B:
                return base_path + "gemma4_e2b_ref.safetensors";
            case GEMMA4_E4B:
                return base_path + "gemma4_e4b_ref.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        switch (model) {
            case GEMMA4_E2B:
                return base_path + "GEMMA4_E2B" + layer_suffix;
            case GEMMA4_E4B:
                return base_path + "GEMMA4_E4B" + layer_suffix;
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        return true;
    }

    inline f32 get_rope_scaling(model_t model) {
        return 1.0f;
    }

    inline uint32_t get_rms_addr(model_t model) {
        switch (model) {
            case GEMMA4_E2B:
                return 36864;
            case GEMMA4_E4B:
                return 37888;
            default:
                return 0;
        }
    }

} // namespace model_list
#endif