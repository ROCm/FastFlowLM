/// \file gemma4e_dflash.hpp
/// \brief gemma4e_dflash class
/// \author FastFlowLM Team
/// \date 2026-10-28
/// \note Engine for Nemotron-3.5-Bolt-E4B with DFlash speculative decoding.
///
/// gemma4e_dflash starts as a byte-for-byte fork of gemma4e_npu (same Gemma4-E4B checkpoint).
/// The layer-type enum, the image/audio payload types and the GEMMA4E_* preprocessing constants are
/// shared with gemma4e_npu, so this header pulls them in rather than redefining them.
#pragma once
#include "models/gemma4e/flm/aie2p/gemma4e_npu.hpp"

class gemma4e_dflash : public causal_lm{
public:
    /// \brief  initialize the qwen3vl_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    gemma4e_dflash(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma4e_dflash();

    /// \brief forward the qwen3vl_npu
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

    // ---- saved states (see flm_state_api.h) --------------------------------
    // A state holds the sliding-window ring of every sliding layer (valid at exactly the length it was read at: a
    // ring cannot be rewound) and the K/V rows of the global layers. The layers that share another layer's cache
    // hold no state. read_state() copies them out of the live buffers, write_state() puts a state back (and sets
    // the context length), write_state_kv() puts back only the global rows (an ancestor's head, before a state
    // taken further on is written).
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);

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
        const nlohmann::json& vc = config.sub("vision_config");
        GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS = vc.value("GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS", -1);
        GEMMA4E_VISION_NUM_HIDDEN_LAYERS   = vc.value("GEMMA4E_VISION_NUM_HIDDEN_LAYERS", -1);
        GEMMA4E_VISION_NUM_ATTENTION_HEADS = vc.value("GEMMA4E_VISION_NUM_ATTENTION_HEADS", -1);
        GEMMA4E_VISION_HIDDEN_SIZE         = vc.value("GEMMA4E_VISION_HIDDEN_SIZE", -1);
        GEMMA4E_VISION_INTERMEDIATE_SIZE   = vc.value("GEMMA4E_VISION_INTERMEDIATE_SIZE", -1);
        GEMMA4E_VISION_HEAD_DIM            = vc.value("GEMMA4E_VISION_HEAD_DIM", -1);
        GEMMA4E_VISION_PATCH_SIZE          = vc.value("GEMMA4E_VISION_PATCH_SIZE", -1);
        GEMMA4E_ROPE_THETA                 = vc.value("GEMMA4E_ROPE_THETA", -1.0f);
        GEMMA4E_POOLING_KERNEL_SIZE        = vc.value("GEMMA4E_POOLING_KERNEL_SIZE", -1);
        GEMMA4E_POSITION_EMBEDDING_SIZE    = vc.value("GEMMA4E_POSITION_EMBEDDING_SIZE", -1);
        GEMMA4E_VISION_IMAGE_OUTPUT_SIZE   = vc.value("GEMMA4E_VISION_IMAGE_OUTPUT_SIZE", -1);
        GEMMA4E_VISION_RESCALE_FACTOR      = vc.value("GEMMA4E_VISION_RESCALE_FACTOR", -1.0f);
        GEMMA4E_VISION_IMAGE_MEAN          = vc.value("GEMMA4E_VISION_IMAGE_MEAN", -1.0f);
        GEMMA4E_VISION_IMAGE_STD           = vc.value("GEMMA4E_VISION_IMAGE_STD", -1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        const nlohmann::json& ac = config.sub("audio_config");
        Audio_MM_TILE_M = ac.value("Audio_MM_TILE_M", 128);
        Audio_MM_TILE_K = ac.value("Audio_MM_TILE_K", 512);
        Audio_MM_TILE_N = ac.value("Audio_MM_TILE_N", 64);
        Gemma4E_Audio_resample_rate = ac.value("Gemma4E_Audio_audio_resample_rate", -1);
        Gemma4E_Audio_gradient_clipping = ac.value("Gemma4E_Audio_gradient_clipping", -1.0f);
        Gemma4E_Audio_Multimodal_Output_SIZE = ac.value("Gemma4E_Audio_Multimodal_Output_SIZE", -1);
        Gemma4E_Audio_language_projection_output_size = ac.value("Gemma4E_Audio_language_projection_output_size", -1);
        Gemma4E_Audio_HIDDEN_SIZE = ac.value("Gemma4E_Audio_HIDDEN_SIZE", -1);
        Gemma4E_Audio_INTERMEDIATE_SIZE = ac.value("Gemma4E_Audio_INTERMEDIATE_SIZE", -1);
        Gemma4E_Audio_attention_chunk_size = ac.value("Gemma4E_Audio_attention_chunk_size", -1);
        Gemma4E_Audio_attention_context_left = ac.value("Gemma4E_Audio_attention_context_left", -1);
        Gemma4E_Audio_attention_context_right = ac.value("Gemma4E_Audio_attention_context_right", -1);
        Gemma4E_Audio_num_attention_heads = ac.value("Gemma4E_Audio_num_attention_heads", -1);
        Gemma4E_Audio_num_attention_layers = ac.value("Gemma4E_Audio_num_attention_layers", -1);
        Gemma4E_Audio_conv1d_kernel_size = ac.value("Gemma4E_Audio_conv1d_kernel_size", -1);
        Gemma4E_Audio_conv1d_stride = ac.value("Gemma4E_Audio_conv1d_stride", -1);
        Gemma4E_Audio_conv2d_kernel_size = ac.value("Gemma4E_conv2d_kernel_size", -1);
        Gemma4E_Audio_conv2d_Stride = ac.value("Gemma4E_conv2d_Stride", -1);
        Gemma4e_Audio_conv2d_Padding = ac.value("Gemma4e_conv2d_Padding", -1);
        Gemma4E_Audio_subsampling_conv_channels_0 = ac.value("Gemma4E_Audio_subsampling_conv_channels_0", -1);
        Gemma4E_Audio_subsampling_conv_channels_1 = ac.value("Gemma4E_Audio_subsampling_conv_channels_1", -1);
        Gemma4E_Audio_attention_softcap = ac.value("Gemma4E_Audio_attention_softcap", -1.0f);
    }

private:
    struct Impl;
    Impl* _impl;
};

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not a gemma4e_dflash
extern "C" const flm_state_api* gemma4e_dflash_state_api(void* engine);

