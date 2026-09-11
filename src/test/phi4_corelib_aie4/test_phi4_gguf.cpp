#include "gguf_fixture.hpp"
#include "fake_corelib.hpp"
#include "corelib/corelib_api.hpp"
#include "models/phi4/phi4_corelib_constants.hpp"
#include "models/phi4/phi4_corelib_gguf.hpp"
#include "test_support.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {
using flm::phi4::Phi4GgufPackage;
using gguf_fixture::Builder;
using gguf_fixture::Mutation;

std::shared_ptr<Phi4GgufPackage> Open(Builder builder,
                                      gguf_fixture::TempFile& file,
                                      std::string_view label) {
    file = builder.Write(label);
    return Phi4GgufPackage::Open(file.path);
}

std::string OpenFailure(Builder builder, Mutation mutation,
                        std::string_view label) {
    auto file = builder.Apply(mutation).Write(label);
    return RequireThrows([&] { Phi4GgufPackage::Open(file.path); });
}

void RequireMismatch(std::string_view error, std::string_view field,
                     std::string_view actual, std::string_view expected) {
    RequireContains(error, field);
    RequireContains(error, "actual " + std::string(actual));
    RequireContains(error, "expected " + std::string(expected));
}

std::string ShapeText(const std::vector<std::uint64_t>& shape) {
    std::string result = "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) result += ',';
        result += std::to_string(shape[index]);
    }
    return result + ']';
}

struct TensorRole {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint32_t type;
};

std::vector<TensorRole> RequiredTensorRoles() {
    std::vector<TensorRole> roles = {
        {"token_embd.weight", {200064, 3072}, gguf_fixture::kQ8_0},
        {"output_norm.weight", {3072}, gguf_fixture::kF32}};
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const auto prefix = "blk." + std::to_string(layer);
        roles.push_back({prefix + ".attn_norm.weight", {3072}, gguf_fixture::kF32});
        roles.push_back({prefix + ".ffn_norm.weight", {3072}, gguf_fixture::kF32});
        roles.push_back({prefix + ".attn_qkv.weight", {5120, 3072}, gguf_fixture::kQ8_0});
        roles.push_back({prefix + ".attn_output.weight", {3072, 3072}, gguf_fixture::kQ8_0});
        roles.push_back({prefix + ".ffn_up.weight", {16384, 3072}, gguf_fixture::kQ8_0});
        roles.push_back({prefix + ".ffn_down.weight", {3072, 8192}, gguf_fixture::kQ8_0});
    }
    return roles;
}

Builder SplitFixture() {
    Builder builder;
    builder.AddTensor("blk.0.attn_qkv.weight", {5120, 3072}, gguf_fixture::kQ8_0)
           .AddTensor("blk.0.ffn_up.weight", {16384, 3072}, gguf_fixture::kQ8_0)
           .AddTensor("f32", {48}, gguf_fixture::kF32);
    return builder;
}

void TestValidV3HeaderMetadataDirectoryAndAlignment() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture(), file, "valid");
    const auto metadata = package->Metadata();
    TEST_REQUIRE(metadata.architecture == "phi3");
    TEST_REQUIRE(metadata.layer_count == 32);
    TEST_REQUIRE(metadata.tokenizer_vocabulary_size == 200064);
    TEST_REQUIRE(!metadata.add_bos_token);
}

void TestEveryMetadataScalarStringAndArrayEncodingCanBeSkippedSafely() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture().AddEverySkippableMetadataType(), file,
                        "metadata-types");
    TEST_REQUIRE(package->Metadata().hidden_size == 3072);
}

void TestTruncatedHeaderMetadataStringArrayAndTensorDirectoryFail() {
    const auto truncated_header = std::filesystem::temp_directory_path() / "flm_phi4_short_header.gguf";
    { std::ofstream out(truncated_header, std::ios::binary | std::ios::trunc); out << "GG"; }
    RequireContains(RequireThrows([&] { Phi4GgufPackage::Open(truncated_header); }), "header");
    std::error_code ignored; std::filesystem::remove(truncated_header, ignored);
    RequireContains(OpenFailure(SplitFixture(), Mutation::TruncatedString, "truncated-string"), "string");
    RequireContains(OpenFailure(SplitFixture(), Mutation::TruncatedDirectory, "truncated-directory"), "tensor");

    Builder array;
    array.RemoveMetadata("tokenizer.ggml.tokens")
         .AddMetadata("tokenizer.ggml.tokens", gguf_fixture::ArrayValue{
             8, std::numeric_limits<std::uint64_t>::max(), {}})
         .AddTensor("x", {32}, gguf_fixture::kQ8_0);
    auto file = array.Write("truncated-array");
    RequireContains(RequireThrows([&] { Phi4GgufPackage::Open(file.path); }), "array");
}

void TestCountProductAlignmentAndOffsetOverflowFail() {
    RequireContains(OpenFailure(SplitFixture(), Mutation::CountOverflow, "count-overflow"), "count");
    RequireContains(OpenFailure(SplitFixture(), Mutation::ProductOverflow, "product-overflow"), "overflow");
    RequireContains(OpenFailure(SplitFixture(), Mutation::OffsetOverflow, "offset-overflow"), "overflow");
}

void TestZeroAndNonPowerOfTwoAlignmentFail() {
    RequireContains(OpenFailure(SplitFixture(), Mutation::ZeroAlignment, "zero-align"), "alignment");
    RequireContains(OpenFailure(SplitFixture(), Mutation::NonPowerOfTwoAlignment, "bad-align"), "alignment");
}

void TestDuplicateTensorNamesFail() {
    RequireContains(OpenFailure(SplitFixture(), Mutation::DuplicateName, "duplicate"), "duplicate");
}

void TestOutOfFileAndOverlappingTensorRangesFail() {
    RequireContains(OpenFailure(SplitFixture(), Mutation::OutOfFileRange, "outside"), "range");
    RequireContains(OpenFailure(SplitFixture(), Mutation::OverlappingRanges, "overlap"), "overlap");
    RequireContains(OpenFailure(SplitFixture(), Mutation::PayloadLengthMismatch, "short-payload"), "range");
}

void TestUnsupportedUnskippableMetadataTypeFails() {
    RequireContains(OpenFailure(SplitFixture(), Mutation::UnsupportedMetadataType, "unsupported"), "metadata type");
}

void TestRequireQ8AndRequireF32ReportNameActualAndExpected() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture(), file, "requires");
    auto error = RequireThrows([&] { package->RequireQ8("f32", std::array<std::int64_t, 1>{48}); });
    RequireMismatch(error, "f32", "F32", "Q8_0");
    error = RequireThrows([&] { package->RequireF32("f32", std::array<std::int64_t, 1>{47}); });
    RequireMismatch(error, "f32", "[48]", "[47]");
}

void TestAttentionQkvReturnsThreeZeroCopyWholeRowViews() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture(), file, "qkv");
    const auto fused = package->RequireQ8("blk.0.attn_qkv.weight", std::array<std::int64_t, 2>{5120, 3072});
    const auto views = package->AttentionQkv(0);
    const std::size_t row_bytes = 3072 / 32 * 34;
    TEST_REQUIRE(views.count == 3);
    TEST_REQUIRE(views.values[0].bytes.data() == fused.bytes.data());
    TEST_REQUIRE(views.values[1].bytes.data() == fused.bytes.data() + 3072 * row_bytes);
    TEST_REQUIRE(views.values[2].bytes.data() == fused.bytes.data() + 4096 * row_bytes);
    TEST_REQUIRE(views.values[0].logical_shape == std::vector<std::int64_t>({3072, 3072}));
    TEST_REQUIRE(views.values[1].logical_shape == std::vector<std::int64_t>({1024, 3072}));
    TEST_REQUIRE(views.values[2].logical_shape == std::vector<std::int64_t>({1024, 3072}));
}

void TestGateUpReturnsTwoZeroCopyWholeRowViews() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture(), file, "gate-up");
    const auto fused = package->RequireQ8("blk.0.ffn_up.weight", std::array<std::int64_t, 2>{16384, 3072});
    const auto views = package->GateUp(0);
    const std::size_t row_bytes = 3072 / 32 * 34;
    TEST_REQUIRE(views.count == 2);
    TEST_REQUIRE(views.values[0].bytes.data() == fused.bytes.data());
    TEST_REQUIRE(views.values[1].bytes.data() == fused.bytes.data() + 8192 * row_bytes);
    TEST_REQUIRE(views.values[0].logical_shape == std::vector<std::int64_t>({8192, 3072}));
    TEST_REQUIRE(views.values[1].logical_shape == std::vector<std::int64_t>({8192, 3072}));
}

void TestSplitRejectsNonIntegralQ8RowBoundary() {
    Builder builder;
    builder.AddTensor("blk.0.attn_qkv.weight", {5120, 3073}, gguf_fixture::kQ8_0);
    auto file = builder.Write("bad-row");
    auto package = Phi4GgufPackage::Open(file.path);
    RequireMismatch(RequireThrows([&] { package->AttentionQkv(0); }),
                    "blk.0.attn_qkv.weight", "3073",
                    "Q8_0 whole-row width divisible by 32");
}

void TestViewsPointIntoTheReadOnlyMapping() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture(), file, "mapping");
    const auto first = package->RequireF32("f32", std::array<std::int64_t, 1>{48});
    const auto second = package->RequireF32("f32", std::array<std::int64_t, 1>{48});
    TEST_REQUIRE(first.values.data() == second.values.data());
    TEST_REQUIRE(first.values.size() == 48);

    auto misaligned_file = Builder().AddTensor("misaligned-f32", {48}, gguf_fixture::kF32)
                               .Apply(Mutation::MisalignedF32).Write("misaligned-f32");
    auto misaligned = Phi4GgufPackage::Open(misaligned_file.path);
    const auto error = RequireThrows([&] {
        misaligned->RequireF32("misaligned-f32", std::array<std::int64_t, 1>{48});
    });
    RequireMismatch(error, "misaligned-f32", "address", "alignment 4");
}

struct ContractFixture {
    gguf_fixture::TempFile file;
    std::shared_ptr<Phi4GgufPackage> package;
    ContractFixture() {
        package = Open(Builder().AddFullContractTensors(), file, "contract");
    }
};

void TestAcceptsExactPhi3Phi4Contract() {
    ContractFixture fixture;
    fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(),
        gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig());
}

void TestRejectsWrongArchitectureAndEveryDimension() {
    struct Case {
        std::string field;
        gguf_fixture::MetadataValue value;
        std::string actual;
        std::string expected;
    };
    const std::vector<Case> cases = {
        {"general.architecture", std::string("llama"), "llama", "phi3"},
        {"phi3.block_count", std::uint32_t{31}, "31", "32"},
        {"phi3.context_length", std::uint32_t{4095}, "4095", "4096"},
        {"phi3.embedding_length", std::uint32_t{3071}, "3071", "3072"},
        {"phi3.feed_forward_length", std::uint32_t{8191}, "8191", "8192"},
        {"phi3.attention.head_count", std::uint32_t{23}, "23", "24"},
        {"phi3.attention.head_count_kv", std::uint32_t{7}, "7", "8"},
        {"phi3.rope.dimension_count", std::uint32_t{95}, "95", "96"},
        {"tokenizer.ggml.tokens",
         gguf_fixture::ArrayValue{0, 200063, std::vector<std::byte>(200063)},
         "200063", "200064"}};
    for (const auto& test_case : cases) {
        auto file = Builder().SetMetadata(test_case.field, test_case.value)
                        .AddFullContractTensors().Write("wrong-field");
        auto package = Phi4GgufPackage::Open(file.path);
        const auto error = RequireThrows([&] { package->ValidatePhi4Contract(
            gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
        RequireMismatch(error, test_case.field, test_case.actual, test_case.expected);
    }
}

void TestRejectsMissingWrongTypeWrongShapeAndWrongLengthForEveryTensorRole() {
    ContractFixture valid;
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const auto prefix = "blk." + std::to_string(layer);
        valid.package->RequireF32(prefix + ".attn_norm.weight", std::array<std::int64_t, 1>{3072});
        valid.package->RequireF32(prefix + ".ffn_norm.weight", std::array<std::int64_t, 1>{3072});
        valid.package->RequireQ8(prefix + ".attn_qkv.weight", std::array<std::int64_t, 2>{5120, 3072});
        valid.package->RequireQ8(prefix + ".attn_output.weight", std::array<std::int64_t, 2>{3072, 3072});
        valid.package->RequireQ8(prefix + ".ffn_up.weight", std::array<std::int64_t, 2>{16384, 3072});
        valid.package->RequireQ8(prefix + ".ffn_down.weight", std::array<std::int64_t, 2>{3072, 8192});
    }
    const nlohmann::json unused;
    for (const auto& role : RequiredTensorRoles()) {
        auto missing_file = Builder().AddFullContractTensors().RemoveTensor(role.name).Write("missing-role");
        auto missing = Phi4GgufPackage::Open(missing_file.path);
        RequireMismatch(RequireThrows([&] { missing->ValidatePhi4Contract(unused, unused, unused); }),
                        role.name, "missing", "present tensor");

        const auto wrong_type = role.type == gguf_fixture::kQ8_0
                                    ? gguf_fixture::kF32 : gguf_fixture::kQ8_0;
        auto type_file = Builder().AddFullContractTensors()
                             .MutateTensor(role.name, wrong_type, role.shape).Write("wrong-type");
        auto type_package = Phi4GgufPackage::Open(type_file.path);
        RequireMismatch(RequireThrows([&] { type_package->ValidatePhi4Contract(unused, unused, unused); }),
                        role.name, wrong_type == gguf_fixture::kF32 ? "F32" : "Q8_0",
                        role.type == gguf_fixture::kF32 ? "F32" : "Q8_0");

        auto wrong_shape = role.shape;
        --wrong_shape.front();
        auto shape_file = Builder().AddFullContractTensors()
                              .MutateTensor(role.name, role.type, wrong_shape).Write("wrong-shape");
        auto shape_package = Phi4GgufPackage::Open(shape_file.path);
        const auto shape_error = RequireThrows([&] {
            shape_package->ValidatePhi4Contract(unused, unused, unused);
        });
        RequireMismatch(shape_error, role.name, ShapeText(wrong_shape), ShapeText(role.shape));

        auto length_file = Builder().AddFullContractTensors()
                               .TruncateTensorPayload(role.name).Write("wrong-length");
        const auto length_error = RequireThrows([&] { Phi4GgufPackage::Open(length_file.path); });
        RequireMismatch(length_error, role.name + " range", "out-of-file range",
                        "range within mapped file");
    }
}

void TestRejectsMixedQuantizationAndOutputWeightPresence() {
    auto mixed_file = Builder().AddFullContractTensors().MutateTensor("token_embd.weight", gguf_fixture::kF32, {200064,3072}).Write("mixed");
    auto mixed = Phi4GgufPackage::Open(mixed_file.path);
    RequireMismatch(RequireThrows([&] { mixed->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }),
                    "token_embd.weight", "F32", "Q8_0");
    auto output_file = Builder().AddFullContractTensors().AddTensor("output.weight", {200064,3072}, gguf_fixture::kQ8_0).Write("output-weight");
    auto output = Phi4GgufPackage::Open(output_file.path);
    const auto error = RequireThrows([&] { output->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
    RequireMismatch(error, "output.weight", "present", "absent (tied token_embd.weight)");
}

void TestRequiresTiedQ8TokenEmbeddingAsLmHead() {
    auto file = Builder().AddFullContractTensors().RemoveTensor("token_embd.weight").Write("untied");
    auto package = Phi4GgufPackage::Open(file.path);
    RequireMismatch(RequireThrows([&] { package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }),
                    "token_embd.weight", "missing", "present tensor");
}

void TestRequiresOriginal4096WindowAndRejectsLongRopeBranch() {
    auto wrong_file = Builder().SetMetadata("phi3.rope.scaling.original_context_length", std::uint32_t{8192}).AddFullContractTensors().Write("long-window");
    auto wrong = Phi4GgufPackage::Open(wrong_file.path);
    RequireMismatch(RequireThrows([&] { wrong->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }),
                    "phi3.rope.scaling.original_context_length", "8192", "4096");
    auto long_file = Builder().AddFullContractTensors().AddTensor("rope_factors_long.weight", {48}, gguf_fixture::kF32).Write("long-rope");
    auto long_rope = Phi4GgufPackage::Open(long_file.path);
    RequireMismatch(RequireThrows([&] { long_rope->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }),
                    "rope_factors_long.weight", "present", "absent");
}

void TestValidatesOptionalShortRopeFactorsAsF32Length48() {
    auto absent_file = Builder().AddFullContractTensors(false).Write("no-short-rope");
    auto absent = Phi4GgufPackage::Open(absent_file.path);
    absent->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig());
    auto wrong_file = Builder().AddFullContractTensors(false).AddTensor("rope_factors_short.weight", {47}, gguf_fixture::kF32).Write("wrong-short-rope");
    auto wrong = Phi4GgufPackage::Open(wrong_file.path);
    RequireMismatch(RequireThrows([&] { wrong->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }),
                    "rope_factors_short.weight", "[47]", "[48]");
}

void TestRejectsNonFiniteOrNonPositiveRopeValues() {
    for (const auto& field : {"phi3.rope.freq_base", "phi3.rope.scaling.attn_factor"}) {
        for (const float value : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto file = Builder().SetMetadata(field, value).AddFullContractTensors().Write("bad-rope-value");
            auto package = Phi4GgufPackage::Open(file.path);
            RequireMismatch(RequireThrows([&] { package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }),
                            field, std::to_string(static_cast<double>(value)),
                            "finite positive value");
        }
    }
}

void TestRejectsConfigDisagreement() {
    ContractFixture fixture;
    struct Case {
        std::string field;
        nlohmann::json value;
        std::string actual;
        std::string expected;
    };
    const std::vector<Case> cases = {
        {"model_type", "other", "other", "phi3"},
        {"num_hidden_layers", 31, "31", "32"},
        {"hidden_size", 3071, "3071", "3072"},
        {"intermediate_size", 8191, "8191", "8192"},
        {"num_attention_heads", 23, "23", "24"},
        {"num_key_value_heads", 7, "7", "8"},
        {"head_dim", 127, "127", "128"},
        {"vocab_size", 200063, "200063", "200064"},
        {"rms_norm_eps", 2.0e-5, "2e-05", "0.000010"},
        {"original_max_position_embeddings", 4095, "4095", "4096"},
        {"hidden_size", 3072.0, "3072.0", "integer 3072"},
        {"hidden_size", std::uint64_t{4294970368ULL}, "4294970368", "3072"},
        {"eos_token_id", 199999.0, "199999.0", "integer 199999"}};
    for (const auto& test_case : cases) {
        auto config = gguf_fixture::ValidConfig();
        config[test_case.field] = test_case.value;
        const auto error = RequireThrows([&] { fixture.package->ValidatePhi4Contract(
            config, gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
        RequireMismatch(error, test_case.field, test_case.actual, test_case.expected);
    }
}

void TestDerivesStopSetFromGgufConfigAndTokenizerIds() {
    ContractFixture fixture;
    fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(),
        gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig());

    auto gguf_file = Builder().SetMetadata("tokenizer.ggml.eos_token_id", std::uint32_t{1})
                         .AddFullContractTensors().Write("wrong-gguf-eos");
    auto gguf = Phi4GgufPackage::Open(gguf_file.path);
    RequireMismatch(RequireThrows([&] { gguf->ValidatePhi4Contract(
                        gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(),
                        gguf_fixture::ValidTokenizerConfig()); }),
                    "tokenizer.ggml.eos_token_id", "1", "200020");

    auto config = gguf_fixture::ValidConfig();
    config["eos_token_id"] = 1;
    RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                        config, gguf_fixture::ValidTokenizer(),
                        gguf_fixture::ValidTokenizerConfig()); }),
                    "eos_token_id", "1", "199999");

    for (const auto& [token, expected] : std::array{
             std::pair<std::string, int>{"<|end|>", 200020},
             std::pair<std::string, int>{"<|endoftext|>", 199999}}) {
        auto tokenizer = gguf_fixture::ValidTokenizer();
        tokenizer["model"]["vocab"][token] = 1;
        for (auto& added : tokenizer["added_tokens"])
            if (added["content"] == token) added["id"] = 1;
        RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                            gguf_fixture::ValidConfig(), tokenizer,
                            gguf_fixture::ValidTokenizerConfig()); }),
                        token, "1", std::to_string(expected));
    }
}

void TestRejectsTokenizerVocabularyEosBosAndMarkerDisagreement() {
    ContractFixture fixture;

    auto tokenizer = gguf_fixture::ValidTokenizer();
    tokenizer["model"]["vocab"].erase("t0");
    RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                        gguf_fixture::ValidConfig(), tokenizer,
                        gguf_fixture::ValidTokenizerConfig()); }),
                    "tokenizer.json distinct vocabulary ID count", "200063", "200064");

    for (const auto& [invalid_id, actual, expected] : std::array{
             std::tuple<nlohmann::json, std::string, std::string>{-1, "-1", "0..200063"},
             std::tuple<nlohmann::json, std::string, std::string>{200064, "200064", "0..200063"},
             std::tuple<nlohmann::json, std::string, std::string>{
                 std::numeric_limits<std::uint64_t>::max(), "18446744073709551615", "0..200063"},
             std::tuple<nlohmann::json, std::string, std::string>{0.0, "0.0", "integer in 0..200063"}}) {
        tokenizer = gguf_fixture::ValidTokenizer();
        tokenizer["model"]["vocab"]["t0"] = invalid_id;
        RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                            gguf_fixture::ValidConfig(), tokenizer,
                            gguf_fixture::ValidTokenizerConfig()); }),
                        "tokenizer.json token ID", actual, expected);
    }

    tokenizer = gguf_fixture::ValidTokenizer();
    tokenizer["added_tokens"].erase(tokenizer["added_tokens"].begin() + 1);
    RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                        gguf_fixture::ValidConfig(), tokenizer,
                        gguf_fixture::ValidTokenizerConfig()); }),
                    "tokenizer.json maximum vocabulary ID", "200062", "200063");

    auto bos_file = Builder().SetMetadata("tokenizer.ggml.add_bos_token", true)
                        .AddFullContractTensors().Write("wrong-gguf-bos");
    auto bos = Phi4GgufPackage::Open(bos_file.path);
    RequireMismatch(RequireThrows([&] { bos->ValidatePhi4Contract(
                        gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(),
                        gguf_fixture::ValidTokenizerConfig()); }),
                    "tokenizer.ggml.add_bos_token", "true", "false");

    auto tokenizer_config = gguf_fixture::ValidTokenizerConfig();
    tokenizer_config["add_bos_token"] = true;
    RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                        gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(),
                        tokenizer_config); }),
                    "add_bos_token", "true", "false");
    tokenizer_config = gguf_fixture::ValidTokenizerConfig();
    tokenizer_config["chat_template"] = "<|user|><|assistant|>";
    RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                        gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(),
                        tokenizer_config); }),
                    "<|end|>", "missing from chat_template", "present in chat_template");
}

void TestRejectsFiniteWrongRmsValue() {
    auto file = Builder().SetMetadata("phi3.attention.layer_norm_rms_epsilon", 2.0e-5f)
                    .AddFullContractTensors().Write("wrong-rms");
    auto package = Phi4GgufPackage::Open(file.path);
    const auto error = RequireThrows([&] { package->ValidatePhi4Contract(
        gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(),
        gguf_fixture::ValidTokenizerConfig()); });
    RequireMismatch(error, "phi3.attention.layer_norm_rms_epsilon",
                    "0.000020", "0.000010");
}

void TestValidationCreatesNoCorelibObjects() {
    fake_corelib::Reset();
    auto api = flm::corelib::CorelibApi::ResolveForTest(fake_corelib::Resolver());
    fake_corelib::GetState().call_counts.clear();

    ContractFixture fixture;
    auto config = gguf_fixture::ValidConfig();
    config["hidden_size"] = 1;
    RequireMismatch(RequireThrows([&] { fixture.package->ValidatePhi4Contract(
                        config, gguf_fixture::ValidTokenizer(),
                        gguf_fixture::ValidTokenizerConfig()); }),
                    "hidden_size", "1", "3072");

    for (const auto name : {"ryzenai_corelib_create_stream",
                            "ryzenai_corelib_create_device_tensor",
                            "ryzenai_corelib_create_tensor_window",
                            "ryzenai_corelib_matmul_bf16_weights_create_gguf_requantized",
                            "ryzenai_corelib_ssmlp_bf16_weights_create_gguf_requantized",
                            "ryzenai_corelib_rmsnorm_bf16_weights_create_scale"})
        TEST_REQUIRE(fake_corelib::GetState().call_counts[name] == 0);
    TEST_REQUIRE(fake_corelib::GetState().live_objects == 0);
    (void)api;
}

}  // namespace

int main() {
#define RUN(name) RunTest(name, #name)
    RUN(TestValidV3HeaderMetadataDirectoryAndAlignment);
    RUN(TestEveryMetadataScalarStringAndArrayEncodingCanBeSkippedSafely);
    RUN(TestTruncatedHeaderMetadataStringArrayAndTensorDirectoryFail);
    RUN(TestCountProductAlignmentAndOffsetOverflowFail);
    RUN(TestZeroAndNonPowerOfTwoAlignmentFail);
    RUN(TestDuplicateTensorNamesFail);
    RUN(TestOutOfFileAndOverlappingTensorRangesFail);
    RUN(TestUnsupportedUnskippableMetadataTypeFails);
    RUN(TestRequireQ8AndRequireF32ReportNameActualAndExpected);
    RUN(TestAttentionQkvReturnsThreeZeroCopyWholeRowViews);
    RUN(TestGateUpReturnsTwoZeroCopyWholeRowViews);
    RUN(TestSplitRejectsNonIntegralQ8RowBoundary);
    RUN(TestViewsPointIntoTheReadOnlyMapping);
    RUN(TestAcceptsExactPhi3Phi4Contract);
    RUN(TestRejectsWrongArchitectureAndEveryDimension);
    RUN(TestRejectsMissingWrongTypeWrongShapeAndWrongLengthForEveryTensorRole);
    RUN(TestRejectsMixedQuantizationAndOutputWeightPresence);
    RUN(TestRequiresTiedQ8TokenEmbeddingAsLmHead);
    RUN(TestRequiresOriginal4096WindowAndRejectsLongRopeBranch);
    RUN(TestValidatesOptionalShortRopeFactorsAsF32Length48);
    RUN(TestRejectsNonFiniteOrNonPositiveRopeValues);
    RUN(TestRejectsConfigDisagreement);
    RUN(TestRejectsFiniteWrongRmsValue);
    RUN(TestDerivesStopSetFromGgufConfigAndTokenizerIds);
    RUN(TestRejectsTokenizerVocabularyEosBosAndMarkerDisagreement);
    RUN(TestValidationCreatesNoCorelibObjects);
#undef RUN
    return 0;
}
