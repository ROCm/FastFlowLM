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
inline constexpr std::int64_t kModelContextLength = 131072;
inline constexpr std::int64_t kMaxDecodeWindow = 4095;
inline constexpr std::uint32_t kRequantizedGroupSize = 64;
/// Intra-packer threads for the Q8_0 requantizing creates. corelib treats 0 as
/// ONE deliberately; this path is compute-bound and scales with the hint.
///
/// This is the per-create hint, NOT concurrent creates. corelib documents that
/// loading a model with 8 CONCURRENT creates on this entry point failed 2 of 10
/// with all-zero output, against 0 of 10 serialized, with attribution open. The
/// creates therefore stay serialized.
inline constexpr std::uint32_t kRequantizeThreads = 8;
inline constexpr float kRmsEpsilon = 1.0e-5f;
}  // namespace flm::phi4
