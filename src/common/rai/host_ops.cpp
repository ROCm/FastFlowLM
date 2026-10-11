#include "rai/host_ops.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace flm::rai {
namespace {

float HalfToFloat(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1f;
    std::uint32_t fraction = half & 0x03ff;
    std::uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((fraction & 0x0400) == 0) {
                fraction <<= 1;
                ++shift;
            }
            fraction &= 0x03ff;
            bits = sign | (static_cast<std::uint32_t>(127 - 14 - shift) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | (fraction << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    return std::bit_cast<float>(bits);
}

}  // namespace

std::vector<float> DecodeQ8Rows(const GgufTensor& table, std::span<const int> ids) {
    if (table.ggml_type != kGgmlTypeQ8_0 || table.shape.size() != 2 ||
        table.shape[0] <= 0 || table.shape[1] <= 0 || table.shape[1] % 32 != 0)
        throw std::runtime_error("embedding must be a two-dimensional Q8_0 tensor with block-aligned rows");
    const auto rows = static_cast<std::size_t>(table.shape[0]);
    const auto width = static_cast<std::size_t>(table.shape[1]);
    const auto blocks_per_row = width / 32;
    const auto row_bytes = blocks_per_row * 34;
    if (rows > std::numeric_limits<std::size_t>::max() / row_bytes ||
        table.bytes.size() != rows * row_bytes)
        throw std::runtime_error("embedding Q8_0 byte length does not match its logical shape");

    std::vector<float> result;
    result.reserve(ids.size() * width);
    for (const int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= rows)
            throw std::out_of_range("embedding token id is outside the vocabulary");
        const std::byte* row = table.bytes.data() + static_cast<std::size_t>(id) * row_bytes;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const std::byte* encoded = row + block * 34;
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, encoded, sizeof(scale_bits));
            const float scale = HalfToFloat(scale_bits);
            for (std::size_t element = 0; element < 32; ++element) {
                const auto code = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(encoded[2 + element]));
                result.push_back(scale * static_cast<float>(code));
            }
        }
    }
    return result;
}

void RmsNorm(std::span<const float> input, std::span<const float> scale,
             std::int64_t rows, std::int64_t width, float epsilon,
             std::span<float> output) {
    if (rows <= 0 || width <= 0) throw std::invalid_argument("RMSNorm rows and width must be positive");
    const auto row_count = static_cast<std::size_t>(rows);
    const auto row_width = static_cast<std::size_t>(width);
    if (row_count > std::numeric_limits<std::size_t>::max() / row_width)
        throw std::invalid_argument("RMSNorm shape overflow");
    const auto elements = row_count * row_width;
    if (input.size() != elements || output.size() != elements || scale.size() != row_width)
        throw std::invalid_argument("RMSNorm shape mismatch");
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("RMSNorm epsilon must be finite and nonnegative");

    for (std::size_t row = 0; row < row_count; ++row) {
        const auto base = row * row_width;
        double sum_of_squares = 0.0;
        for (std::size_t column = 0; column < row_width; ++column) {
            const double value = input[base + column];
            sum_of_squares += value * value;
        }
        const float mean_square = static_cast<float>(sum_of_squares / static_cast<double>(width));
        const float denominator = std::sqrt(mean_square + epsilon);
        for (std::size_t column = 0; column < row_width; ++column)
            output[base + column] = (input[base + column] / denominator) * scale[column];
    }
}

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
