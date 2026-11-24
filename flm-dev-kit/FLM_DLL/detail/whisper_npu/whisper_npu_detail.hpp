// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "models/whisper/whisper_npu.hpp"
#include "whisper_encoder.hpp"
#include "whisper_decoder.hpp"
#include "whisper_lm_head.hpp"

struct whisper_npu::Impl{
    /// @brief constexprs

    /// @brief  initialize the whisper_npu
    /// @param config
    /// @param npu_instance
    Impl(Whisper_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 448);
    ~Impl();

    buffer<bf16> encode_audio(buffer<bf16>& mel_feature);
    /// @brief forward the whisper_npu
    buffer<bf16> decode_audio(int ids);

    void load_weights(Q4NX& q4nx);
    void set_context_length(int L);
    void clear_context();

    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();

    Whisper_Config config;
    npu_xclbin_manager *npu;
    int current_context_length;
    int MAX_L;

    std::unique_ptr<WhisperLMHead> lm_head;
    std::unique_ptr<WhisperDecoder> decoder;
    std::unique_ptr<WhisperEncoder> encoder;

    buffer<bf16> final_layer_norm_weights;
    buffer<bf16> final_layer_norm_biases;
};
