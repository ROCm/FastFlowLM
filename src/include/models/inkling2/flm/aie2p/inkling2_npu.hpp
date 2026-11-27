/// \file inkling2_npu.hpp
/// \brief inkling2_npu class
/// \author FastFlowLM Team
/// \date 2026-11-21
/// \note Portions of this file consist of AI-generated content.
/// \note The engine for Inkling-2-Mini (inkling2_loop), text only: 24 weight layers, layers
///       6-17 run three times (48 layer applications), 128-expert top-8 MoE in every layer,
///       relative-position attention (no RoPE), sliding window 1024 except every 6th layer.
///       Q4_K body, Q4_1 expert down projections, Q8_0 lm_head.
#pragma once
#include <cstdint>
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#include "flm_state_api.h"


/// \brief one image for the engine's own vision tower
/// \note rgb is height x width x 3 bytes (RGB, row-major, no padding), exactly as decoded: the tower pads it to whole
///       40 x 40 pixel cells and normalizes it itself. Borrowed; only read during the call.
struct inkling2_image_t {
    const uint8_t* rgb = nullptr;
    int            height = 0;
    int            width = 0;
};

/// \brief the optional payload of prefill(): the images the prompt's image placeholder tokens (one per cell,
///        inkling2_npu::image_token_id) stand for, in prompt order
/// \note The engine counts the placeholders of the window and throws if they differ from the cells of the images: a
///       mismatch would put one image's rows into another's places. The payload goes with the first window of a prompt,
///       which must hold all the placeholders of its images.
struct inkling2_payload_t {
    const inkling2_image_t* images = nullptr;
    int                     num_images = 0;
};

class inkling2_npu : public causal_lm{
public:
    /// \brief  initialize the inkling2_npu
    /// \param config the configuration
    /// \param npu_instance the npu instance
    /// \param MAX_L context length in tokens (the global layers' K/V grow with it)
    inkling2_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 65536);
    ~inkling2_npu();

    /// \brief decode one token
    /// \param ids the token
    /// \return the logits (vocab_size)
    buffer<bf16> forward(int ids) override;

    /// \brief run the prompt into the caches
    /// \return the logits of the last prompt token
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \note The convolution state and the sliding-window rows only hold the latest
    ///       position: going back needs restore() or write_state() first.
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief clear the context
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the cache slot (0-47: layers 0-23, then layers 6-17 of loop passes 2 and 3)
    /// \param idx the token index
    /// \return K of that token, all KV heads in head order
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the cache slot (as get_k_cache)
    /// \param idx the token index
    /// \return V of that token, all KV heads in head order
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief resize the caches, keeping the conversation
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    /// \brief the placeholder token the prompt repeats once per image cell
    static constexpr int image_token_id = 200054;

    /// \brief whether vision_weight.q4nx is in the model folder and the image tower is enabled (INKLING2_VISION=0 turns it off)
    bool has_vision_tower() const;

    /// \brief the image tokens (cells of 40 x 40 pixels) an image of this size takes
    int image_tokens_for(int height, int width) const;

private:
    struct Impl;
    Impl* _impl;
};

/// \brief the saved-state table of this engine (flm_state_api.h); nullptr if `engine` is not an inkling2_npu
extern "C" const flm_state_api* inkling2_npu_state_api(void* engine);
