// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MODELS_H__
#define __MODELS_H__
#include <string>
#include "rope_weights.h"

namespace model_list {
    enum model_t {
        LLAMA_3_2_1B,
        LLAMA_3_2_3B,
        LLAMA_3_1_8B,
        QWEN3_0_6B,
        QWEN3_4B,
        PHI4_4B,
        NANBEIGE_3B,
        UNKNOWN_MODEL
    };

    inline model_t get_model_type(const std::string& model_name) {
        if (model_name == "LLAMA_3_2_1B") return LLAMA_3_2_1B;
        else if (model_name == "LLAMA_3_2_3B") return LLAMA_3_2_3B;
        else if (model_name == "LLAMA_3_1_8B") return LLAMA_3_1_8B;
        else if (model_name == "QWEN3_0_6B") return QWEN3_0_6B;
        else if (model_name == "QWEN3_4B") return QWEN3_4B;
        else if (model_name == "PHI4_4B") return PHI4_4B;
        else if (model_name == "NANBEIGE_3B") return NANBEIGE_3B;
        else return UNKNOWN_MODEL;
    }

    inline std::string get_model_path(model_t model) {
        static const std::string base_path = "q4nx_files/";
        switch (model) {
            case LLAMA_3_2_1B:
                return base_path + "Llama-3.2-1B-NPU2";
            case LLAMA_3_2_3B:
                return base_path + "Llama-3.2-3B-NPU2";
            case LLAMA_3_1_8B:
                return base_path + "Llama-3.1-8B-NPU2";
            case PHI4_4B:
                return base_path + "Phi4-mini-Instruct-NPU2";
            case QWEN3_0_6B:
                return base_path + "Qwen3-0.6B-NPU2";
            case QWEN3_4B:
                return base_path + "Qwen3-4B-NPU2";
            case NANBEIGE_3B:
                return base_path + "Nanbeige4.1-3B-NPU2";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_reference_path(model_t model) {
        static const std::string base_path = "references/";
        switch (model) {
            case LLAMA_3_2_1B:
                return base_path + "LLAMA_3_2_1B_ref.safetensors";
            case LLAMA_3_2_3B:
                return base_path + "LLAMA_3_2_3B_ref.safetensors";
            case LLAMA_3_1_8B:
                return base_path + "LLAMA_3_1_8B_ref.safetensors";
            case PHI4_4B:
                return base_path + "PHI4_4B_ref.safetensors";
            case QWEN3_0_6B:
                return base_path + "QWEN3_0_6B_ref.safetensors";
            case QWEN3_4B:  
                return base_path + "QWEN3_4B_ref.safetensors";
            case NANBEIGE_3B:
                return base_path + "NANBEIGE_3B_ref.safetensors";
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline std::string get_layer_xclbin_path(model_t model) {
        static const std::string base_path = "build/";
        static const std::string layer_suffix = "/xclbins/layer.xclbin";
        switch (model) {
            case LLAMA_3_2_1B:
                return base_path + "LLAMA_3_2_1B" + layer_suffix;
            case LLAMA_3_2_3B:
                return base_path + "LLAMA_3_2_3B" + layer_suffix;
            case PHI4_4B:
                return base_path + "PHI4_4B" + layer_suffix;
            case QWEN3_0_6B:
                return base_path + "QWEN3_0_6B" + layer_suffix;
            case QWEN3_4B:
                return base_path + "QWEN3_4B" + layer_suffix;
            case NANBEIGE_3B:
                return base_path + "NANBEIGE_3B" + layer_suffix;
            default:
                return "UNKNOWN_MODEL";
        }
    }

    inline bool has_qk_norm(model_t model) {
        switch (model) {
            case LLAMA_3_2_1B:
                return false;
            case LLAMA_3_2_3B:
                return false;
            case LLAMA_3_1_8B:
                return false;
            case PHI4_4B:
                return false;
            case QWEN3_0_6B:
                return true;
            case QWEN3_4B:
                return true;
            case NANBEIGE_3B:
                return false;   
            default:
                return false;
        }
    }

    inline f32* get_model_rotary_frequencies(model_t model) {
        switch (model) {
            case LLAMA_3_2_1B:
                return &(llama_1b_rope[0]);
            case LLAMA_3_2_3B:
            case LLAMA_3_1_8B:
                return &(llama_3b_8b_rope[0]);
            case NANBEIGE_3B:
                return &(nanbeige_3b_rope[0]);
            case PHI4_4B:
                return &(phi4_rope[0]);
            case QWEN3_0_6B:
            case QWEN3_4B:
                return &(qwen3_rope[0]);
            default:
                return nullptr;
        }
    }

    inline f32 get_rope_scaling(model_t model) {
        switch (model) {
            case LLAMA_3_2_1B:
            case LLAMA_3_2_3B:
            case LLAMA_3_1_8B:
                return 1.0f;
            case NANBEIGE_3B:
                return 1.0f;
            case PHI4_4B:
                return 1.0f; //1.1902380714238083f;
            case QWEN3_0_6B:
            case QWEN3_4B:
                return 1.0f;
            default:
                return 1.0f;
        }
    }
    inline uint32_t get_rms_addr(model_t model) {
        switch (model) {
            case LLAMA_3_2_1B:
                return 53248;
            case LLAMA_3_2_3B:
                return 10272;
            case PHI4_4B:
                return 10272;
            case QWEN3_0_6B:
                return 51200;
            case QWEN3_4B:
                return 54272;
            case NANBEIGE_3B:
                return 54272;
            default:
                return 0;
        }
    }

} // namespace model_list
#endif