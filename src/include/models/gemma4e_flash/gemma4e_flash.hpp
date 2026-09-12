/// \file gemma4e_flash.hpp
/// \brief gemma4e_flash class
/// \author FastFlowLM Team
/// \date 2026-09-11
/// \version 0.9.28
/// \note This is a header file for the gemma4e_flash class
///
/// gemma4e_flash is a second engine for the same Gemma4-E checkpoint as
/// gemma4e_npu. It starts life as a byte-for-byte copy of that engine and is
/// the place to tune prefill for short input prompts; gemma4e_npu remains the
/// general-purpose engine.
///
/// The layer-type enum, the image/audio payload types and the GEMMA4E_*
/// preprocessing constants are shared with gemma4e_npu, so this header pulls
/// them in rather than redefining them: the application builds one
/// gemma4e_image_payload_t and hands it to either engine.
#pragma once
#include "models/gemma4e/gemma4e_npu.hpp"

class gemma4e_flash : public causal_lm{
public:
    /// \brief  initialize the gemma4e_flash
    /// \param config the configuration
    /// \param npu_instance the npu instance
    gemma4e_flash(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma4e_flash();

    /// \brief forward the gemma4e_flash
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief update the max length
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;


    int checkpoint() override;
    int restore() override;

    // parameters for vision preprocessing in Gemma4e
    unsigned int GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS;
    unsigned int GEMMA4E_VISION_NUM_HIDDEN_LAYERS;
    unsigned int GEMMA4E_VISION_NUM_ATTENTION_HEADS;
    unsigned int GEMMA4E_VISION_HIDDEN_SIZE;
    unsigned int GEMMA4E_VISION_INTERMEDIATE_SIZE;
    unsigned int GEMMA4E_VISION_HEAD_DIM;
    unsigned int GEMMA4E_VISION_PATCH_SIZE;
    float GEMMA4E_ROPE_THETA;
    unsigned int GEMMA4E_POOLING_KERNEL_SIZE;
    unsigned int GEMMA4E_POSITION_EMBEDDING_SIZE;
    unsigned int GEMMA4E_VISION_IMAGE_OUTPUT_SIZE;
    float GEMMA4E_VISION_RESCALE_FACTOR;
    float GEMMA4E_VISION_IMAGE_MEAN;
    float GEMMA4E_VISION_IMAGE_STD;

    // parameters for audio preprocessing in Gemma4e
    unsigned int Audio_MM_TILE_M;
    unsigned int Audio_MM_TILE_K;
    unsigned int Audio_MM_TILE_N;
    int Gemma4E_Audio_resample_rate;
    float Gemma4E_Audio_gradient_clipping;
    unsigned int Gemma4E_Audio_Multimodal_Output_SIZE;
    unsigned int Gemma4E_Audio_language_projection_output_size;
    unsigned int Gemma4E_Audio_HIDDEN_SIZE;
    unsigned int Gemma4E_Audio_INTERMEDIATE_SIZE;
    unsigned int Gemma4E_Audio_attention_chunk_size;
    unsigned int Gemma4E_Audio_attention_context_left;
    unsigned int Gemma4E_Audio_attention_context_right;
    unsigned int Gemma4E_Audio_num_attention_heads;
    unsigned int Gemma4E_Audio_num_attention_layers;
    unsigned int Gemma4E_Audio_conv1d_kernel_size;
    unsigned int Gemma4E_Audio_conv1d_stride;
    unsigned int Gemma4E_Audio_conv2d_kernel_size;
    unsigned int Gemma4E_Audio_conv2d_Stride;
    unsigned int Gemma4e_Audio_conv2d_Padding;
    unsigned int Gemma4E_Audio_subsampling_conv_channels_0;
    unsigned int Gemma4E_Audio_subsampling_conv_channels_1;
    float Gemma4E_Audio_attention_softcap;


    inline void load_vision_preprocess_parameters(LM_Config& config){
        // Note: this should be called by Impl:: constructor
        GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS = config.sub("vision_config").value("GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS", -1);
        GEMMA4E_VISION_NUM_HIDDEN_LAYERS   = config.sub("vision_config").value("GEMMA4E_VISION_NUM_HIDDEN_LAYERS", -1);
        GEMMA4E_VISION_NUM_ATTENTION_HEADS = config.sub("vision_config").value("GEMMA4E_VISION_NUM_ATTENTION_HEADS", -1);
        GEMMA4E_VISION_HIDDEN_SIZE         = config.sub("vision_config").value("GEMMA4E_VISION_HIDDEN_SIZE", -1);
        GEMMA4E_VISION_INTERMEDIATE_SIZE   = config.sub("vision_config").value("GEMMA4E_VISION_INTERMEDIATE_SIZE", -1);
        GEMMA4E_VISION_HEAD_DIM            = config.sub("vision_config").value("GEMMA4E_VISION_HEAD_DIM", -1);
        GEMMA4E_VISION_PATCH_SIZE          = config.sub("vision_config").value("GEMMA4E_VISION_PATCH_SIZE", -1);
        GEMMA4E_ROPE_THETA                 = config.sub("vision_config").value("GEMMA4E_ROPE_THETA", -1.0f);
        GEMMA4E_POOLING_KERNEL_SIZE        = config.sub("vision_config").value("GEMMA4E_POOLING_KERNEL_SIZE", -1);
        GEMMA4E_POSITION_EMBEDDING_SIZE    = config.sub("vision_config").value("GEMMA4E_POSITION_EMBEDDING_SIZE", -1);
        GEMMA4E_VISION_IMAGE_OUTPUT_SIZE   = config.sub("vision_config").value("GEMMA4E_VISION_IMAGE_OUTPUT_SIZE", -1);
        GEMMA4E_VISION_RESCALE_FACTOR      = config.sub("vision_config").value("GEMMA4E_VISION_RESCALE_FACTOR", -1.0f);
        GEMMA4E_VISION_IMAGE_MEAN          = config.sub("vision_config").value("GEMMA4E_VISION_IMAGE_MEAN", -1.0f);
        GEMMA4E_VISION_IMAGE_STD           = config.sub("vision_config").value("GEMMA4E_VISION_IMAGE_STD", -1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        Audio_MM_TILE_M = config.sub("audio_config").value("Audio_MM_TILE_M", 128);
        Audio_MM_TILE_K = config.sub("audio_config").value("Audio_MM_TILE_K", 512);
        Audio_MM_TILE_N = config.sub("audio_config").value("Audio_MM_TILE_N", 64);
        Gemma4E_Audio_resample_rate = config.sub("audio_config").value("Gemma4E_Audio_audio_resample_rate", -1);
        Gemma4E_Audio_gradient_clipping = config.sub("audio_config").value("Gemma4E_Audio_gradient_clipping", -1.0f);
        Gemma4E_Audio_Multimodal_Output_SIZE = config.sub("audio_config").value("Gemma4E_Audio_Multimodal_Output_SIZE", -1);
        Gemma4E_Audio_language_projection_output_size = config.sub("audio_config").value("Gemma4E_Audio_language_projection_output_size", -1);
        Gemma4E_Audio_HIDDEN_SIZE = config.sub("audio_config").value("Gemma4E_Audio_HIDDEN_SIZE", -1);
        Gemma4E_Audio_INTERMEDIATE_SIZE = config.sub("audio_config").value("Gemma4E_Audio_INTERMEDIATE_SIZE", -1);
        Gemma4E_Audio_attention_chunk_size = config.sub("audio_config").value("Gemma4E_Audio_attention_chunk_size", -1);
        Gemma4E_Audio_attention_context_left = config.sub("audio_config").value("Gemma4E_Audio_attention_context_left", -1);
        Gemma4E_Audio_attention_context_right = config.sub("audio_config").value("Gemma4E_Audio_attention_context_right", -1);
        Gemma4E_Audio_num_attention_heads = config.sub("audio_config").value("Gemma4E_Audio_num_attention_heads", -1);
        Gemma4E_Audio_num_attention_layers = config.sub("audio_config").value("Gemma4E_Audio_num_attention_layers", -1);
        Gemma4E_Audio_conv1d_kernel_size = config.sub("audio_config").value("Gemma4E_Audio_conv1d_kernel_size", -1);
        Gemma4E_Audio_conv1d_stride = config.sub("audio_config").value("Gemma4E_Audio_conv1d_stride", -1);
        Gemma4E_Audio_conv2d_kernel_size = config.sub("audio_config").value("Gemma4E_conv2d_kernel_size", -1);
        Gemma4E_Audio_conv2d_Stride = config.sub("audio_config").value("Gemma4E_conv2d_Stride", -1);
        Gemma4e_Audio_conv2d_Padding = config.sub("audio_config").value("Gemma4e_conv2d_Padding", -1);
        Gemma4E_Audio_subsampling_conv_channels_0 = config.sub("audio_config").value("Gemma4E_Audio_subsampling_conv_channels_0", -1);
        Gemma4E_Audio_subsampling_conv_channels_1 = config.sub("audio_config").value("Gemma4E_Audio_subsampling_conv_channels_1", -1);
        Gemma4E_Audio_attention_softcap = config.sub("audio_config").value("Gemma4E_Audio_attention_softcap", -1.0f);
    }

private:
    struct Impl;
    Impl* _impl;
};

