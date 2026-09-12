#include "models/phi4/phi4_corelib_host.hpp"

#include "models/phi4/phi4_corelib_constants.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace flm::phi4 {
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
            bits = sign | (static_cast<std::uint32_t>(127 - 14 - shift) << 23) |
                   (fraction << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | (fraction << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    return std::bit_cast<float>(bits);
}

}  // namespace

std::vector<float> DecodeEmbeddingRowsQ8(
    const TensorView& embedding, std::span<const int> token_ids) {
    if (embedding.ggml_type != 8 || embedding.logical_shape.size() != 2 ||
        embedding.logical_shape[0] <= 0 || embedding.logical_shape[1] <= 0 ||
        embedding.logical_shape[1] % 32 != 0) {
        throw std::runtime_error("embedding must be a two-dimensional Q8_0 tensor with block-aligned rows");
    }
    const auto rows = static_cast<std::size_t>(embedding.logical_shape[0]);
    const auto width = static_cast<std::size_t>(embedding.logical_shape[1]);
    const auto blocks_per_row = width / 32;
    const auto row_bytes = blocks_per_row * 34;
    if (rows > std::numeric_limits<std::size_t>::max() / row_bytes ||
        embedding.bytes.size() != rows * row_bytes) {
        throw std::runtime_error("embedding Q8_0 byte length does not match its logical shape");
    }

    std::vector<float> result;
    result.reserve(token_ids.size() * width);
    for (const int token_id : token_ids) {
        if (token_id < 0 || static_cast<std::size_t>(token_id) >= rows) {
            throw std::out_of_range("embedding token id is outside the vocabulary");
        }
        const std::byte* row = embedding.bytes.data() +
                               static_cast<std::size_t>(token_id) * row_bytes;
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const std::byte* encoded = row + block * 34;
            std::uint16_t scale_bits;
            std::memcpy(&scale_bits, encoded, sizeof(scale_bits));
            const float scale = HalfToFloat(scale_bits);
            for (std::size_t element = 0; element < 32; ++element) {
                const auto code = static_cast<std::int8_t>(
                    std::to_integer<std::uint8_t>(encoded[2 + element]));
                result.push_back(scale * static_cast<float>(code));
            }
        }
    }
    return result;
}

void HostRmsNorm(
    std::span<const float> input,
    std::span<const float> scale,
    std::int64_t rows,
    std::int64_t width,
    float epsilon,
    std::span<float> output) {
    if (rows <= 0 || width <= 0)
        throw std::invalid_argument("Phi-4 RMSNorm rows and width must be positive");
    const auto row_count = static_cast<std::size_t>(rows);
    const auto row_width = static_cast<std::size_t>(width);
    if (row_count > std::numeric_limits<std::size_t>::max() / row_width)
        throw std::invalid_argument("Phi-4 RMSNorm shape overflow");
    const auto elements = row_count * row_width;
    if (input.size() != elements || output.size() != elements ||
        scale.size() != row_width)
        throw std::invalid_argument("Phi-4 RMSNorm shape mismatch");
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("Phi-4 RMSNorm epsilon must be finite and nonnegative");

    for (std::size_t row = 0; row < row_count; ++row) {
        const auto base = row * row_width;
        double sum_of_squares = 0.0;
        for (std::size_t column = 0; column < row_width; ++column) {
            const double value = input[base + column];
            sum_of_squares += value * value;
        }
        const float mean_square = static_cast<float>(
            sum_of_squares / static_cast<double>(width));
        const float denominator = std::sqrt(mean_square + epsilon);
        for (std::size_t column = 0; column < row_width; ++column)
            output[base + column] =
                (input[base + column] / denominator) * scale[column];
    }
}

std::vector<std::uint16_t> ConvertF32ToBf16(std::span<const float> values) {
    std::vector<std::uint16_t> result;
    result.reserve(values.size());
    for (const float value : values) {
        std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        if ((bits & 0x7fffffffU) > 0x7f800000U) {
            bits |= 0x00400000U;
        } else {
            bits += 0x7fffU + ((bits >> 16) & 1U);
        }
        result.push_back(static_cast<std::uint16_t>(bits >> 16));
    }
    return result;
}

RopeTables BuildShortRopeTables(
    const GgufPhi4Metadata& metadata,
    std::optional<FloatTensorView> short_factors) {
    if (metadata.context_length < static_cast<std::uint64_t>(kMaxSequenceLength) ||
        metadata.rope_original_context_length != static_cast<std::uint64_t>(kMaxSequenceLength) ||
        metadata.rope_dimension_count != static_cast<std::uint64_t>(kRopeDimension) ||
        !std::isfinite(metadata.rope_frequency_base) || metadata.rope_frequency_base <= 0 ||
        !std::isfinite(metadata.rope_attention_factor)) {
        throw std::runtime_error("invalid Phi-4 RoPE metadata");
    }

    std::array<double, kRopeDimension / 2> factors{};
    factors.fill(1.0);
    if (short_factors) {
        if (short_factors->logical_shape != std::vector<std::int64_t>{kRopeDimension / 2} ||
            short_factors->values.size() != factors.size()) {
            throw std::runtime_error("rope_factors_short.weight must have shape [48]");
        }
        for (std::size_t i = 0; i < factors.size(); ++i) {
            factors[i] = short_factors->values[i];
            if (!std::isfinite(factors[i]) || factors[i] <= 0)
                throw std::runtime_error("rope_factors_short.weight must contain finite positive values");
        }
    }

    RopeTables tables;
    tables.cosine.resize(kMaxSequenceLength * factors.size());
    tables.sine.resize(kMaxSequenceLength * factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        const double inv_freq = 1.0 /
            (std::pow(metadata.rope_frequency_base, (2.0 * i) / 96.0) * factors[i]);
        for (std::size_t position = 0; position < kMaxSequenceLength; ++position) {
            const double angle = static_cast<double>(position) * inv_freq;
            const auto index = position * factors.size() + i;
            tables.cosine[index] = static_cast<float>(
                std::cos(angle) * metadata.rope_attention_factor);
            tables.sine[index] = static_cast<float>(
                std::sin(angle) * metadata.rope_attention_factor);
        }
    }
    return tables;
}

}  // namespace flm::phi4
