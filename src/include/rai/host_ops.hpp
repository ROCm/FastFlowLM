/// \file host_ops.hpp
/// \brief The little math a rai engine does on the host
/// \note Plain CPU and no corelib types, so it is testable without hardware.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace flm::rai {

/// \brief round-to-nearest-even F32 -> BF16, NaN kept quiet
std::vector<std::uint16_t> F32ToBf16(std::span<const float> values);

/// \brief the same conversion into a caller's buffer of values.size() elements
void F32ToBf16(std::span<const float> values, std::span<std::uint16_t> output);

}  // namespace flm::rai
