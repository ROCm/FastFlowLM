// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include "rope_weights.h"

namespace model_list {
    enum model_t {
        QWEN3_5_2B,
        QWEN3_5_4B,
        QWEN3_5_9B,
        QWEN3_5_08B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "QWEN3_5_2B") return QWEN3_5_2B;
        else if (model_name == "QWEN3_5_4B") return QWEN3_5_4B;
        else if (model_name == "QWEN3_5_9B") return QWEN3_5_9B;
        else if (model_name == "QWEN3_5_08B") return QWEN3_5_08B;
        else return UNKNOWN_MODEL;
    }

    inline std::string get_model_path(model_t model) {
        static const std::string base_path = "/path/to/models/";
        switch (model) {
            case QWEN3_5_2B:
                return base_path + "Qwen3.5-2B-NPU2";
            case QWEN3_5_4B:
                return base_path + "Qwen3.5-4B-NPU2";
            case QWEN3_5_9B:
                return base_path + "Qwen3.5-9B-NPU2";
            case QWEN3_5_08B:
                return base_path + "Qwen3.5-0.8B-NPU2";

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_reference_path(model_t model) {
        std::string base_path = get_model_path(model);
        switch (model) {
            case QWEN3_5_2B:
                return base_path + "/gated_attention_2B.safetensors";
            case QWEN3_5_4B:
                return base_path + "/QWEN3_5_4B_ref.safetensors";
            case QWEN3_5_08B:
                return base_path + "/gated_attention_0.8B.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        switch (model) {
            case QWEN3_5_2B:
                return base_path + "QWEN3_5_2B" + layer_suffix;
            case QWEN3_5_4B:
                return base_path + "QWEN3_5_4B" + layer_suffix;
            case QWEN3_5_08B:
                return base_path + "QWEN3_5_08B" + layer_suffix;

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        switch (model) {
            case QWEN3_5_2B:
                return true;
            case QWEN3_5_4B:
                return true;
            case QWEN3_5_9B:
                return true;
            case QWEN3_5_08B:
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
            case QWEN3_5_2B:
                return 36864;
            case QWEN3_5_4B:
                return 37888;
            case QWEN3_5_9B:
                return 57344;
            case QWEN3_5_08B:
                return 34816;
            default:
                return 0;
        }
    }

} // namespace model_list
#endif
