// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__CONFIG__HPP__
#define __SMOLVLA__CONFIG__HPP__

#include <cstdint>

/// @brief Shape and sizing constants that config.json does not carry.
///
/// SmolVLA's config.json describes the weights but not the runtime shapes the
/// host has to size scratch buffers against (image count, token counts, chunk
/// length, ...). Those used to be spelled as bare literals in a dozen places;
/// they live here instead so a single edit re-sizes every buffer consistently.
namespace smolvla_cfg {

// ---------------------------------------------------------------- vision ---
/// patch grid is 32x32 = 1024 tokens per image, at 512x512 with a 16x16 patch.
constexpr int IMAGE_SIZE          = 512;
constexpr int PATCH_SIZE          = 16;
constexpr int PATCH_CIN           = 3;
constexpr int PATCH_GRID          = IMAGE_SIZE / PATCH_SIZE;            // 32
constexpr int TOKENS_PER_IMAGE    = PATCH_GRID * PATCH_GRID;            // 1024
constexpr int NUM_POSITIONS       = TOKENS_PER_IMAGE;

/// observation carries at most this many camera views.
constexpr int MAX_IMAGES          = 3;
constexpr int MAX_VISION_TOKENS   = TOKENS_PER_IMAGE * MAX_IMAGES;      // 3072

/// pixel shuffle folds a PATCH_GRID x PATCH_GRID token grid down by SHUFFLE_FACTOR
/// in each axis, so each image contributes VISION_TOKENS_PER_IMAGE tokens of
/// width D_vision * SHUFFLE_FACTOR^2 to the connector.
constexpr int SHUFFLE_FACTOR      = 4;
constexpr int SHUFFLED_GRID       = PATCH_GRID / SHUFFLE_FACTOR;        // 8
constexpr int VISION_TOKENS_PER_IMAGE = SHUFFLED_GRID * SHUFFLED_GRID;  // 64

/// flattened vision feature width feeding the vision->text connector. Not in
/// config.json; it is D_vision(768) * SHUFFLE_FACTOR^2 = 12288.
constexpr int CONNECTOR_IN        = 12288;

// -------------------------------------------------------------- language ---
/// prefix budget: MAX_IMAGES worth of shuffled vision tokens plus the text
/// tokens and the single state token. Sizes the kv cache and every language
/// scratch buffer.
constexpr int MAX_PREFIX_LEN      = VISION_TOKENS_PER_IMAGE * 4;        // 256

/// padding token of the SmolVLM2 tokenizer, used to fill the unused prefix.
constexpr int PAD_TOKEN_ID        = 2;

// ---------------------------------------------------------------- expert ---
/// action chunk length the expert runs over in one denoise step.
constexpr int MAX_ACTION_CHUNK    = 64;

/// the expert alternates self- and cross-attention layers; layer l is a
/// self-attention layer when l % SELF_ATTN_EVERY_N_LAYERS == 0.
constexpr int SELF_ATTN_EVERY_N_LAYERS = 2;

/// self-attention scratch spans the chunk plus the VLM prefix it concatenates.
constexpr int MAX_EXPERT_KV       = MAX_ACTION_CHUNK + MAX_PREFIX_LEN;

/// weight rows/cols are padded up to this many elements for the NPU tiling.
constexpr int PAD_ALIGN           = 256;

// --------------------------------------------------------- flow matching ---
/// Euler integration of the flow-matching ODE from t=1 down to t=0.
constexpr int   NUM_DENOISE_STEPS = 10;
constexpr float DENOISE_DT        = -1.0f / (float)NUM_DENOISE_STEPS;

/// sinusoidal time-embedding period range (matches the reference policy).
constexpr float TIME_MIN_PERIOD   = 0.004f;
constexpr float TIME_MAX_PERIOD   = 4.0f;

// --------------------------------------------------------------- numeric ---
/// normalization epsilons. The three towers do not agree on this: SigLIP's
/// LayerNorm uses 1e-6, the text tower and the expert's RMSNorm use 1e-5.
constexpr float VISION_NORM_EPS   = 1e-6f;
constexpr float TEXT_NORM_EPS     = 1e-5f;
constexpr float EXPERT_NORM_EPS   = 1e-5f;

/// RoPE base frequency (LLaMA-style rotate_half), shared by text and expert.
constexpr float ROPE_THETA        = 1e4f;

/// round `v` up to the next multiple of `align`.
constexpr uint32_t pad_to(uint32_t v, uint32_t align = PAD_ALIGN) {
    return ((v + align - 1) / align) * align;
}

}  // namespace smolvla_cfg

#endif // __SMOLVLA__CONFIG__HPP__
