#include "models/phi4/phi4_corelib_host.hpp"
#include "test_support.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#define NOMINMAX
#include <windows.h>

namespace {
using namespace flm::phi4;

void PutHalf(std::vector<std::byte>& bytes, std::size_t offset, std::uint16_t bits) {
    bytes[offset] = static_cast<std::byte>(bits & 0xff);
    bytes[offset + 1] = static_cast<std::byte>(bits >> 8);
}

TensorView ThreeRows() {
    static std::vector<std::byte> bytes(3 * 34, std::byte{0x7f});
    std::fill(bytes.begin(), bytes.end(), std::byte{0x7f});
    for (std::size_t row = 0; row < 3; ++row) {
        const std::size_t base = row * 34;
        PutHalf(bytes, base, row == 0 ? 0x3800 : row == 1 ? 0x3c00 : 0x4000);
        for (std::size_t column = 0; column < 32; ++column) {
            const auto value = static_cast<std::int8_t>(row == 1 ? -static_cast<int>(column) :
                                                        static_cast<int>(row + column));
            bytes[base + 2 + column] = static_cast<std::byte>(value);
        }
    }
    return {"token_embd.weight", bytes, {3, 32}, 8};
}

GgufPhi4Metadata Metadata(double attention = 1.0) {
    return {"phi3", 32, 3072, 8192, 24, 8, 4096, 96, 10000.0,
            attention, 4096, 200064, false};
}

void TestLazyEmbeddingDecodesOnlyRequestedRows() {
    constexpr std::size_t width = 65536;
    constexpr std::size_t row_bytes = width / 32 * 34; // 17 Windows pages.
    auto* mapping = static_cast<std::byte*>(VirtualAlloc(
        nullptr, 3 * row_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    TEST_REQUIRE(mapping != nullptr);
    for (std::size_t block = 0; block < width / 32; ++block) {
        const std::uint16_t scale = 0x3c00;
        std::memcpy(mapping + row_bytes + block * 34, &scale, sizeof(scale));
        std::fill_n(mapping + row_bytes + block * 34 + 2, 32, std::byte{0xff});
    }
    DWORD old_protection{};
    TEST_REQUIRE(VirtualProtect(mapping, row_bytes, PAGE_NOACCESS, &old_protection));
    TEST_REQUIRE(VirtualProtect(mapping + 2 * row_bytes, row_bytes,
                                PAGE_NOACCESS, &old_protection));
    const TensorView embedding{"token_embd.weight", {mapping, 3 * row_bytes},
                               {3, static_cast<std::int64_t>(width)}, 8};
    const std::array ids{1};
    const auto decoded = DecodeEmbeddingRowsQ8(embedding, ids);
    TEST_REQUIRE(decoded.size() == width);
    TEST_REQUIRE(decoded.front() == -1.0f && decoded.back() == -1.0f);
    VirtualFree(mapping, 0, MEM_RELEASE);
}

void TestLazyEmbeddingPreservesRequestOrderAndDuplicates() {
    const auto embedding = ThreeRows();
    const std::array ids{2, 0, 2};
    const auto decoded = DecodeEmbeddingRowsQ8(embedding, ids);
    TEST_REQUIRE(decoded.size() == 96);
    TEST_REQUIRE(decoded[0] == 4.0f);
    TEST_REQUIRE(decoded[32] == 0.0f);
    TEST_REQUIRE(decoded[64] == 4.0f);
    TEST_REQUIRE(decoded[95] == 66.0f);
}

void TestLazyEmbeddingRejectsNegativeAndOutOfRangeIds() {
    const auto embedding = ThreeRows();
    std::array negative{-1};
    std::array too_large{3};
    RequireContains(RequireThrows([&] { DecodeEmbeddingRowsQ8(embedding, negative); }),
                    "token id");
    RequireContains(RequireThrows([&] { DecodeEmbeddingRowsQ8(embedding, too_large); }),
                    "token id");
    auto malformed = embedding;
    malformed.bytes = malformed.bytes.first(malformed.bytes.size() - 1);
    std::array valid{0};
    RequireContains(RequireThrows([&] { DecodeEmbeddingRowsQ8(malformed, valid); }),
                    "Q8_0");
}

void TestF32ToBf16UsesRoundToNearestEven() {
    const std::array values{
        std::bit_cast<float>(std::uint32_t{0x3f808000}),
        std::bit_cast<float>(std::uint32_t{0x3f818000}),
        -2.5f,
        std::numeric_limits<float>::infinity()};
    const auto result = ConvertF32ToBf16(values);
    TEST_REQUIRE(result == std::vector<std::uint16_t>({0x3f80, 0x3f82, 0xc020, 0x7f80}));
}

void TestRopeTablesUseFloat64IntermediatesAndFloat32Outputs() {
    const auto tables = BuildShortRopeTables(Metadata(), std::nullopt);
    constexpr std::size_t i = 47;
    constexpr std::size_t p = 4095;
    const double inv = 1.0 / std::pow(10000.0, (2.0 * i) / 96.0);
    const float expected = static_cast<float>(std::cos(p * inv));
    TEST_REQUIRE(tables.cosine[p * 48 + i] == expected);
}

void TestRopeTablesApplyShortFactorsAndAttentionFactor() {
    std::array<float, 48> factors{};
    factors.fill(2.0f);
    FloatTensorView factor_view{"rope_factors_short.weight", factors, {48}};
    const auto tables = BuildShortRopeTables(Metadata(1.5), factor_view);
    const double inv = 1.0 / (std::pow(10000.0, 2.0 / 96.0) * 2.0);
    TEST_REQUIRE(tables.cosine[48 + 1] == static_cast<float>(std::cos(inv) * 1.5));
    TEST_REQUIRE(tables.sine[48 + 1] == static_cast<float>(std::sin(inv) * 1.5));
}

void TestRopeTablesHaveShape4096By48() {
    const auto tables = BuildShortRopeTables(Metadata(), std::nullopt);
    TEST_REQUIRE(tables.cosine.size() == 4096 * 48);
    TEST_REQUIRE(tables.sine.size() == 4096 * 48);
    TEST_REQUIRE(tables.cosine[0] == 1.0f);
    TEST_REQUIRE(tables.sine[0] == 0.0f);
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestLazyEmbeddingDecodesOnlyRequestedRows);
    RUN_TEST(TestLazyEmbeddingPreservesRequestOrderAndDuplicates);
    RUN_TEST(TestLazyEmbeddingRejectsNegativeAndOutOfRangeIds);
    RUN_TEST(TestF32ToBf16UsesRoundToNearestEven);
    RUN_TEST(TestRopeTablesUseFloat64IntermediatesAndFloat32Outputs);
    RUN_TEST(TestRopeTablesApplyShortFactorsAndAttentionFactor);
    RUN_TEST(TestRopeTablesHaveShape4096By48);
#undef RUN_TEST
}
