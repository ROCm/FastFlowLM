/// \file host_ops.hpp
/// \brief The little math a rai engine does on the host
/// \note Plain CPU and no corelib types, so it is testable without hardware.
#pragma once

#include "rai/gguf_file.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace flm::rai {

/// \brief dequantize the rows of a [vocab, width] Q8_0 table for these ids
/// \return ids.size() * width floats, row-major
std::vector<float> DecodeQ8Rows(const GgufTensor& table, std::span<const int> ids);

/// \brief scale * x / sqrt(mean(x^2) + epsilon), row by row
void RmsNorm(std::span<const float> input, std::span<const float> scale,
             std::int64_t rows, std::int64_t width, float epsilon,
             std::span<float> output);

/// \brief round-to-nearest-even F32 -> BF16, NaN kept quiet
std::vector<std::uint16_t> F32ToBf16(std::span<const float> values);

/// \brief the same conversion into a caller's buffer of values.size() elements
void F32ToBf16(std::span<const float> values, std::span<std::uint16_t> output);

}  // namespace flm::rai
