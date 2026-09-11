#pragma once

#include "models/phi4/phi4_corelib_gguf.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace flm::phi4 {

struct RopeTables {
    std::vector<float> cosine;
    std::vector<float> sine;
};

std::vector<float> DecodeEmbeddingRowsQ8(
    const TensorView& embedding,
    std::span<const int> token_ids);

std::vector<std::uint16_t> ConvertF32ToBf16(std::span<const float> values);

RopeTables BuildShortRopeTables(
    const GgufPhi4Metadata& metadata,
    std::optional<FloatTensorView> short_factors);

}  // namespace flm::phi4
