// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>

namespace model_list {
    enum model_t {
        QWEN3_8_27B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "QWEN3_8_27B") return QWEN3_8_27B;
        else return UNKNOWN_MODEL;
    }

    inline std::string get_model_path(model_t model) {
        static const std::string base_path = "/path/to/models/";
        switch (model) {
            case QWEN3_8_27B:
                return base_path + "Qwen3.8-27B-NPU2";

            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_reference_path(model_t model) {
        std::string base_path = get_model_path(model);
        switch (model) {
            case QWEN3_8_27B:
                return base_path + "/ref.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/deltanet_layer.xclbin";
        switch (model) {
            case QWEN3_8_27B:
                return base_path + "QWEN3_8_27B" + layer_suffix;

            default:
                return "UNKNOWN_MODEL";
        }
    }

} // namespace model_list
#endif
