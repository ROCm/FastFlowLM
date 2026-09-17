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
/// Intra-packer thread hint for one Q8_0 requantizing create. corelib treats 0
/// as ONE deliberately. Packing many weights at once is the bigger lever and
/// belongs to the caller, so the parallelism is taken below as concurrent
/// creates instead; asking for both would oversubscribe the machine.
inline constexpr std::uint32_t kRequantizeThreads = 0;

/// How many weight creates run at once. The 161 creates are independent -- each
/// reads its own mapped range of the GGUF and produces its own object -- so
/// this is the parallelism that actually shortens load. Real threads rather
/// than a packer hint, so it is not subject to whatever thread limits the
/// surrounding environment imposes on the packer.
inline constexpr std::size_t kWeightCreateConcurrency = 8;
inline constexpr float kRmsEpsilon = 1.0e-5f;
}  // namespace flm::phi4
