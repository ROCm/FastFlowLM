#pragma once

#include <cstdint>

namespace flm::phi4 {
inline constexpr std::int64_t kLayerCount = 32;
inline constexpr std::int64_t kHiddenSize = 3072;
inline constexpr std::int64_t kIntermediateSize = 8192;
inline constexpr std::int64_t kQueryHeadCount = 24;
inline constexpr std::int64_t kKvHeadCount = 8;
inline constexpr std::int64_t kHeadSize = 128;
inline constexpr std::int64_t kQueryDimension = 3072;
inline constexpr std::int64_t kKvDimension = 1024;
inline constexpr std::int64_t kVocabularySize = 200064;
inline constexpr std::int64_t kRopeDimension = 96;
inline constexpr std::int64_t kMaxSequenceLength = 4096;
inline constexpr std::int64_t kMaxDecodeWindow = 4095;
inline constexpr std::uint32_t kRequantizedGroupSize = 64;
inline constexpr float kRmsEpsilon = 1.0e-5f;
}  // namespace flm::phi4
