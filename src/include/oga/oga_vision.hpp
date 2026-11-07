// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file oga_vision.hpp
/// \brief The images a prefill hands an OGA engine
/// \note Standard C++ only: the Qwen3.5 frontend builds these in every build,
///       with or without the OGA backend.
#pragma once

#include <bit>
#include <cstdint>
#include <vector>

namespace flm::oga {

/// \brief the most patches the images of one prompt may have together
/// \note OGA runs every image of a prompt in one pass of the vision model. On
///       hip-ep 0.5.2 (OGA 0.17.0) that pass gives wrong image features when it
///       has many more patches than the first pass the model ran, or when that
///       first pass had only a few hundred. The engine therefore runs one pass of
///       this size before its first real image, and no prompt may have more.
///       8100 patches is a 90x90 grid (1440x1440 pixels), 2025 image tokens.
inline constexpr std::int64_t kMaxImagePatches = 90 * 90;

/// \brief one image, preprocessed for the vision model
struct OgaImage {
    std::int64_t grid_t = 1;
    std::int64_t grid_h = 0;
    std::int64_t grid_w = 0;
    /// \brief float16 bits, one row of channels * temporal patch * patch * patch
    ///        values per patch, in the processor's patch order
    std::vector<std::uint16_t> pixels;
};

/// \brief the images of one prefill, in the order of their image token runs
struct OgaVisionInput {
    std::vector<OgaImage> images;
};

/// \brief the float16 nearest to `value`, ties to even
inline std::uint16_t FloatToHalf(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    bits &= 0x7fffffffu;
    if (bits >= 0x47800000u)  // 65536 or more, infinity, NaN
        return static_cast<std::uint16_t>(sign | (bits > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (bits < 0x38800000u) {  // under 2^-14: a float16 subnormal or zero
        // Adding 0.5 lines the 10 subnormal bits up at the bottom of the
        // mantissa, and the addition rounds them to nearest even.
        const float aligned = std::bit_cast<float>(bits) + 0.5f;
        return static_cast<std::uint16_t>(sign | (std::bit_cast<std::uint32_t>(aligned) - 0x3f000000u));
    }
    const std::uint32_t odd = (bits >> 13) & 1u;
    bits += 0xc8000fffu + odd;  // rebias the exponent from 127 to 15 and round
    return static_cast<std::uint16_t>(sign | (bits >> 13));
}

}  // namespace flm::oga
