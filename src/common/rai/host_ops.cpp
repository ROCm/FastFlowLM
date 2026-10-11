#include "rai/host_ops.hpp"

#include <bit>
#include <stdexcept>

namespace flm::rai {

std::vector<std::uint16_t> F32ToBf16(std::span<const float> values) {
    std::vector<std::uint16_t> result(values.size());
    F32ToBf16(values, result);
    return result;
}

void F32ToBf16(std::span<const float> values, std::span<std::uint16_t> output) {
    if (output.size() != values.size())
        throw std::invalid_argument("F32ToBf16 output size does not match its input");
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits = std::bit_cast<std::uint32_t>(values[i]);
        if ((bits & 0x7fffffffU) > 0x7f800000U) {
            bits |= 0x00400000U;
        } else {
            bits += 0x7fffU + ((bits >> 16) & 1U);
        }
        output[i] = static_cast<std::uint16_t>(bits >> 16);
    }
}

}  // namespace flm::rai
