#include "gguf_fixture.hpp"
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
    RequireContains(error, "f32"); RequireContains(error, "actual F32"); RequireContains(error, "expected Q8_0");
    error = RequireThrows([&] { package->RequireF32("f32", std::array<std::int64_t, 1>{47}); });
    RequireContains(error, "f32"); RequireContains(error, "48"); RequireContains(error, "47");
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
    RequireContains(RequireThrows([&] { package->AttentionQkv(0); }), "row");
}

void TestViewsPointIntoTheReadOnlyMapping() {
    gguf_fixture::TempFile file;
    auto package = Open(SplitFixture(), file, "mapping");
    const auto first = package->RequireF32("f32", std::array<std::int64_t, 1>{48});
    const auto second = package->RequireF32("f32", std::array<std::int64_t, 1>{48});
    TEST_REQUIRE(first.values.data() == second.values.data());
    TEST_REQUIRE(first.values.size() == 48);
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
    const std::vector<std::pair<std::string, gguf_fixture::MetadataValue>> cases = {
        {"general.architecture", std::string("llama")}, {"phi3.block_count", std::uint32_t{31}},
        {"phi3.context_length", std::uint32_t{4095}}, {"phi3.embedding_length", std::uint32_t{3071}},
        {"phi3.feed_forward_length", std::uint32_t{8191}}, {"phi3.attention.head_count", std::uint32_t{23}},
        {"phi3.attention.head_count_kv", std::uint32_t{7}}, {"phi3.rope.dimension_count", std::uint32_t{95}},
        {"tokenizer.ggml.tokens", gguf_fixture::ArrayValue{0, 200063, std::vector<std::byte>(200063)}}};
    for (const auto& [field, value] : cases) {
        auto file = Builder().SetMetadata(field, value).AddFullContractTensors().Write("wrong-field");
        auto package = Phi4GgufPackage::Open(file.path);
        const auto error = RequireThrows([&] { package->ValidatePhi4Contract(
            gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
        RequireContains(error, field); RequireContains(error, "actual"); RequireContains(error, "expected");
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
    std::vector<std::string> required_names = {"token_embd.weight", "output_norm.weight"};
    for (std::size_t layer = 0; layer < 32; ++layer) {
        const auto prefix = "blk." + std::to_string(layer);
        required_names.push_back(prefix + ".attn_norm.weight");
        required_names.push_back(prefix + ".ffn_norm.weight");
        required_names.push_back(prefix + ".attn_qkv.weight");
        required_names.push_back(prefix + ".attn_output.weight");
        required_names.push_back(prefix + ".ffn_up.weight");
        required_names.push_back(prefix + ".ffn_down.weight");
    }
    const nlohmann::json unused;
    for (const auto& name : required_names) {
        auto file = Builder().AddFullContractTensors().RemoveTensor(name).Write("missing-role");
        auto package = Phi4GgufPackage::Open(file.path);
        const auto error = RequireThrows([&] {
            package->ValidatePhi4Contract(unused, unused, unused);
        });
        RequireContains(error, name); RequireContains(error, "actual missing"); RequireContains(error, "expected");
    }
    auto type_file = Builder().AddFullContractTensors().MutateTensor("blk.0.attn_output.weight", gguf_fixture::kF32, {3072,3072}).Write("wrong-type");
    auto type_package = Phi4GgufPackage::Open(type_file.path);
    auto error = RequireThrows([&] { type_package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
    RequireContains(error, "blk.0.attn_output.weight"); RequireContains(error, "actual F32"); RequireContains(error, "expected Q8_0");
    auto shape_file = Builder().AddFullContractTensors().MutateTensor("blk.0.ffn_down.weight", gguf_fixture::kQ8_0, {3072,8160}).Write("wrong-shape");
    auto shape_package = Phi4GgufPackage::Open(shape_file.path);
    error = RequireThrows([&] { shape_package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
    RequireContains(error, "blk.0.ffn_down.weight"); RequireContains(error, "actual"); RequireContains(error, "expected");
    RequireContains(OpenFailure(Builder().AddTensor("x", {32}, gguf_fixture::kQ8_0), Mutation::PayloadLengthMismatch, "wrong-length"), "range");
}

void TestRejectsMixedQuantizationAndOutputWeightPresence() {
    auto mixed_file = Builder().AddFullContractTensors().MutateTensor("token_embd.weight", gguf_fixture::kF32, {200064,3072}).Write("mixed");
    auto mixed = Phi4GgufPackage::Open(mixed_file.path);
    RequireContains(RequireThrows([&] { mixed->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), "token_embd.weight");
    auto output_file = Builder().AddFullContractTensors().AddTensor("output.weight", {200064,3072}, gguf_fixture::kQ8_0).Write("output-weight");
    auto output = Phi4GgufPackage::Open(output_file.path);
    const auto error = RequireThrows([&] { output->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
    RequireContains(error, "output.weight"); RequireContains(error, "actual present"); RequireContains(error, "expected absent");
}

void TestRequiresTiedQ8TokenEmbeddingAsLmHead() {
    auto file = Builder().AddFullContractTensors().RemoveTensor("token_embd.weight").Write("untied");
    auto package = Phi4GgufPackage::Open(file.path);
    RequireContains(RequireThrows([&] { package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), "token_embd.weight");
}

void TestRequiresOriginal4096WindowAndRejectsLongRopeBranch() {
    auto wrong_file = Builder().SetMetadata("phi3.rope.scaling.original_context_length", std::uint32_t{8192}).AddFullContractTensors().Write("long-window");
    auto wrong = Phi4GgufPackage::Open(wrong_file.path);
    RequireContains(RequireThrows([&] { wrong->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), "phi3.rope.scaling.original_context_length");
    auto long_file = Builder().AddFullContractTensors().AddTensor("rope_factors_long.weight", {48}, gguf_fixture::kF32).Write("long-rope");
    auto long_rope = Phi4GgufPackage::Open(long_file.path);
    RequireContains(RequireThrows([&] { long_rope->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), "rope_factors_long.weight");
}

void TestValidatesOptionalShortRopeFactorsAsF32Length48() {
    auto absent_file = Builder().AddFullContractTensors(false).Write("no-short-rope");
    auto absent = Phi4GgufPackage::Open(absent_file.path);
    absent->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig());
    auto wrong_file = Builder().AddFullContractTensors(false).AddTensor("rope_factors_short.weight", {47}, gguf_fixture::kF32).Write("wrong-short-rope");
    auto wrong = Phi4GgufPackage::Open(wrong_file.path);
    RequireContains(RequireThrows([&] { wrong->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), "rope_factors_short.weight");
}

void TestRejectsNonFiniteOrNonPositiveRopeValues() {
    for (const auto& field : {"phi3.rope.freq_base", "phi3.rope.scaling.attn_factor"}) {
        for (const float value : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto file = Builder().SetMetadata(field, value).AddFullContractTensors().Write("bad-rope-value");
            auto package = Phi4GgufPackage::Open(file.path);
            RequireContains(RequireThrows([&] { package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), field);
        }
    }
}

void TestRejectsConfigDisagreement() {
    ContractFixture fixture;
    const std::vector<std::pair<std::string, nlohmann::json>> cases = {
        {"model_type", "other"}, {"num_hidden_layers", 31}, {"hidden_size", 3071},
        {"intermediate_size", 8191}, {"num_attention_heads", 23}, {"num_key_value_heads", 7},
        {"head_dim", 127}, {"vocab_size", 200063}, {"rms_norm_eps", 2.0e-5},
        {"original_max_position_embeddings", 4095}};
    for (const auto& [field, value] : cases) {
        auto config = gguf_fixture::ValidConfig(); config[field] = value;
        const auto error = RequireThrows([&] { fixture.package->ValidatePhi4Contract(config, gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); });
        RequireContains(error, field); RequireContains(error, "actual"); RequireContains(error, "expected");
    }
}

void TestDerivesStopSetFromGgufConfigAndTokenizerIds() {
    ContractFixture fixture;
    fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig());
}

void TestRejectsTokenizerVocabularyEosBosAndMarkerDisagreement() {
    ContractFixture fixture;
    auto tokenizer = gguf_fixture::ValidTokenizer();
    tokenizer["model"]["vocab"]["<|end|>"] = 1;
    RequireContains(RequireThrows([&] { fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), tokenizer, gguf_fixture::ValidTokenizerConfig()); }), "<|end|>");
    auto config = gguf_fixture::ValidConfig(); config["eos_token_id"] = 1;
    RequireContains(RequireThrows([&] { fixture.package->ValidatePhi4Contract(config, gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig()); }), "eos_token_id");
    auto tokenizer_config = gguf_fixture::ValidTokenizerConfig(); tokenizer_config["add_bos_token"] = true;
    RequireContains(RequireThrows([&] { fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), tokenizer_config); }), "add_bos_token");
    tokenizer_config = gguf_fixture::ValidTokenizerConfig(); tokenizer_config["chat_template"] = "<|user|><|assistant|>";
    RequireContains(RequireThrows([&] { fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), tokenizer_config); }), "<|end|>");
}

void TestValidationCreatesNoCorelibObjects() {
    ContractFixture fixture;
    fixture.package->ValidatePhi4Contract(gguf_fixture::ValidConfig(), gguf_fixture::ValidTokenizer(), gguf_fixture::ValidTokenizerConfig());
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
    RUN(TestDerivesStopSetFromGgufConfigAndTokenizerIds);
    RUN(TestRejectsTokenizerVocabularyEosBosAndMarkerDisagreement);
    RUN(TestValidationCreatesNoCorelibObjects);
#undef RUN
    return 0;
}
