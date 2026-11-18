/// \file gemma4e_payload.hpp
/// \brief Frontend-to-engine multimodal payloads for the Gemma 4 family
/// \warning This is a private Gemma implementation detail, not part of the
///          stable model-extension ABI. A provider that builds this Gemma
///          engine owns both the producer and consumer of these C++ types.
#pragma once

#include "typedef.hpp"

#include <cstdint>
#include <utility>
#include <vector>

struct gemma4e_image_payload_t {
    /// Per image: patch count and elements per patch.
    std::vector<std::pair<int, int>> image_patch__element_per_patch;
    std::vector<std::uint32_t> valid_patch_size_per_image;
    std::vector<std::vector<bf16>> pixel_values;
    /// Flattened x/y position pairs for each image.
    std::vector<std::vector<int>> image_grid_pairs_per_image;
    std::vector<unsigned int> num_soft_tokens_per_image;
    unsigned int num_images = 0;
};

struct gemma4e_audio_payload_t {
    /// Per-audio row-major frames-by-bins mel spectrogram.
    std::vector<std::vector<bf16>> mel_spectrograms;
    std::vector<int> mel_spectrogram_frames_per_audio;
    std::vector<int> mel_spectrogram_bins_per_audio;
    unsigned int num_audios = 0;
    std::vector<unsigned int> num_soft_tokens_per_audio;
};

struct gemma4e_multi_modal_payload_t {
    gemma4e_image_payload_t image_payload;
    gemma4e_audio_payload_t audio_payload;
};
