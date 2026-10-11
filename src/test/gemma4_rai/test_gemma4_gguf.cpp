#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "gemma4_gguf_fixture.hpp"
#include "rai/gguf_file.hpp"
#include "test_support.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using flm::rai::GgufFile;
using flm::gemma4::ConfigFromMetadata;
using flm::gemma4::Gemma4GgufPackage;

void TestConfigFromMetadataReadsE2b() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    const auto cfg = ConfigFromMetadata(*file);
    TEST_REQUIRE(cfg.row_name == "E2B");
    TEST_REQUIRE(cfg.layers == 35);
    TEST_REQUIRE(cfg.hidden == 1536);
    TEST_REQUIRE(cfg.kv_heads == 1);
    TEST_REQUIRE(cfg.kv_layers == 15);
    TEST_REQUIRE(cfg.sliding_window == 512);
    TEST_REQUIRE(cfg.context_length == 131072);
    TEST_REQUIRE(cfg.vocab == 262144);
    TEST_REQUIRE(cfg.rope_theta == 1000000.0);
    TEST_REQUIRE(cfg.rope_theta_swa == 10000.0);
    TEST_REQUIRE(cfg.logit_softcap == 30.0f);
    TEST_REQUIRE(cfg.eos_token_id == 106);
    TEST_REQUIRE(cfg.bos_token_id == 2);
    TEST_REQUIRE(cfg.add_bos);
    TEST_REQUIRE(cfg.group == 32 && cfg.head_group == 32 && cfg.ple_group == 32);
    TEST_REQUIRE(cfg.LayerHeadDim(0) == 256);   // sliding
    TEST_REQUIRE(cfg.LayerHeadDim(4) == 512);   // full
    TEST_REQUIRE(cfg.ple_dim == 256);
    // 9.999999974752427e-07 (i.e. 1e-6) -- NOT Phi-4's 1e-5.
    TEST_REQUIRE(std::abs(static_cast<double>(cfg.eps) - 1.0e-6) < 1e-9);
}

void TestConfigFromMetadataReadsE4b() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    const auto cfg = ConfigFromMetadata(*file);
    TEST_REQUIRE(cfg.row_name == "E4B");
    TEST_REQUIRE(cfg.layers == 42);
    TEST_REQUIRE(cfg.hidden == 2560);
    TEST_REQUIRE(cfg.kv_heads == 2);
    TEST_REQUIRE(cfg.kv_layers == 24);
    TEST_REQUIRE(cfg.LayerHeadDim(5) == 512);
    TEST_REQUIRE(cfg.ple_dim == 256);
}

void TestContextLengthIsReadFromTheFile() {
    auto builder = gemma4_fixture::MakeBuilder(gemma4_fixture::E2bOptions());
    builder.SetMetadata("gemma4.context_length", std::uint32_t{8192});
    auto file = GgufFile::Open(builder.Write());
    TEST_REQUIRE(ConfigFromMetadata(*file).context_length == 8192);
}

void TestAFileWithoutAContextLengthGetsOnePage() {
    auto builder = gemma4_fixture::MakeBuilder(gemma4_fixture::E2bOptions());
    builder.RemoveMetadata("gemma4.context_length");
    auto file = GgufFile::Open(builder.Write());
    TEST_REQUIRE(!file->HasMetadata("gemma4.context_length"));
    TEST_REQUIRE(ConfigFromMetadata(*file).context_length == 4096);
}

// THE ONE THAT MATTERS. gemma4.attention.shared_kv_layers is the file's
// SHARING count (20 on E2B, 18 on E4B), never the OWNING count. Pinning
// cfg.kv_layers against the real metadata values here means the
// subtraction (kv_layers = block_count - shared_kv_layers) cannot be
// quietly dropped -- a reader that read the raw key straight through would
// produce 20/18 here instead of 15/24, and this test would catch it even
// though DeriveCacheOwners's own unit tests (test_gemma4_derivations.cpp)
// would stay green, since they are handed the corrected count directly.
// See verified-gguf-facts.md's "correction that matters most".
void TestKvLayersIsBlockCountMinusTheFilesSharingCount() {
    auto e2b_file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    TEST_REQUIRE(e2b_file->Unsigned("gemma4.attention.shared_kv_layers") == 20);
    const auto e2b = ConfigFromMetadata(*e2b_file);
    TEST_REQUIRE(e2b.kv_layers == 15);   // 35 - 20, NOT the raw 20

    auto e4b_file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    TEST_REQUIRE(e4b_file->Unsigned("gemma4.attention.shared_kv_layers") == 18);
    const auto e4b = ConfigFromMetadata(*e4b_file);
    TEST_REQUIRE(e4b.kv_layers == 24);   // 42 - 18, NOT the raw 18
}

void TestHeadDimKeysAreNotReadTheObviousWayRound() {
    // The GGUF's names run opposite to the driver's: key_length is the FULL
    // layers' head size and key_length_swa the SLIDING one. Reading them the
    // obvious way round swaps 256 and 512 everywhere and is silent.
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    const auto cfg = ConfigFromMetadata(*file);
    TEST_REQUIRE(cfg.head_dim == 256);
    TEST_REQUIRE(cfg.global_head_dim == 512);
    TEST_REQUIRE(file->Unsigned("gemma4.attention.key_length") == 512);
    TEST_REQUIRE(file->Unsigned("gemma4.attention.key_length_swa") == 256);
}

void TestTensorCountIsSixPlusSeventeenPerLayer() {
    auto e2b = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    auto e4b = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    TEST_REQUIRE(e2b->TensorCount() == 601);
    TEST_REQUIRE(e4b->TensorCount() == 720);
    TEST_REQUIRE(ConfigFromMetadata(*e2b).layers == 35);
    TEST_REQUIRE(ConfigFromMetadata(*e4b).layers == 42);
}

// A tensor count that doesn't match 6 + layers*17 must be rejected, naming
// BOTH the actual and the expected count -- so whoever hits this can tell
// at a glance whether a tensor is missing/extra or the layer count itself
// disagrees with the file.
void TestRejectsWrongTensorCountNamingBothNumbers() {
    auto builder = gemma4_fixture::MakeBuilder(gemma4_fixture::E2bOptions());
    builder.AddTensor("blk.0.extra_bogus_tensor.weight", {1}, gemma4_fixture::kF32);
    auto file = GgufFile::Open(builder.Write());
    const auto error = RequireThrows([&] { ConfigFromMetadata(*file); });
    RequireContains(error, "602");
    RequireContains(error, "601");
}

void TestRejectsWrongArchitectureByName() {
    auto builder = gemma4_fixture::MakeBuilder(gemma4_fixture::E2bOptions());
    builder.SetMetadata("general.architecture", std::string("phi3"));
    auto file = GgufFile::Open(builder.Write());
    const auto error = RequireThrows([&] { ConfigFromMetadata(*file); });
    RequireContains(error, "phi3");
    RequireContains(error, "gemma4");
}

// lm_head is tied to token_embd on both real models -- output.weight never
// exists, and nothing here may require it.
void TestOutputWeightIsAbsentAndNotRequired() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    TEST_REQUIRE(!file->HasTensor("output.weight"));
    const auto cfg = ConfigFromMetadata(*file);
    TEST_REQUIRE(cfg.layers == 35);
}

void TestTensorCountIsSixPlusSeventeenPerLayerOnPackageOpen() {
    // Same contract, exercised through Gemma4GgufPackage::Open rather than
    // ConfigFromMetadata directly.
    const auto path = gemma4_fixture::Write(gemma4_fixture::E4bOptions());
    const auto package = Gemma4GgufPackage::Open(path);
    TEST_REQUIRE(package->Config().layers == 42);
    TEST_REQUIRE(package->Config().kv_layers == 24);
    TEST_REQUIRE(package->File().TensorCount() == 720);
    TEST_REQUIRE(package->Path() == path);
}

void TestContractDoesNotRejectE2bOverIntermediateSize() {
    // config.json says 6144, which is true of layers 0-14 and wrong about
    // 15-34. Comparing it to the derived widths would reject a GOOD package.
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{{"architectures", {"Gemma4ForConditionalGeneration"}},
                                {"intermediate_size", 6144},
                                {"num_hidden_layers", 35}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config{{"eos_token", "<turn|>"}};
    package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);  // must not throw
}

void TestContractRejectsWrongLayerCount() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{{"architectures", {"Gemma4ForConditionalGeneration"}},
                                {"num_hidden_layers", 999}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);
    });
    RequireContains(error, "999");
    RequireContains(error, "35");
}

void TestContractRejectsArchitectureNotNamingGemma4() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{{"architectures", {"Phi3ForCausalLM"}},
                                {"num_hidden_layers", 35}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);
    });
    RequireContains(error, "architecture");
}

// When the tokenizer really does carry a vocab, ValidateGemma4Contract
// cross-checks its size against the row's vocabulary -- exercising the
// branch TestContractDoesNotRejectE2bOverIntermediateSize's empty
// tokenizer.json never reaches.
void TestContractRejectsTokenizerVocabSizeMismatch() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{{"architectures", {"Gemma4ForConditionalGeneration"}},
                                {"num_hidden_layers", 35}};
    nlohmann::json vocab = nlohmann::json::object();
    vocab["<pad>"] = 0;
    vocab["<eos>"] = 1;
    const nlohmann::json tokenizer{{"model", {{"vocab", vocab}}}};
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);
    });
    RequireContains(error, "vocab");
}

// The real config.json is a MULTIMODAL config: architectures, model_type
// and eos_token_id sit at the top level, but num_hidden_layers, vocab_size
// and intermediate_size are nested one level down, under "text_config"
// (see verified-gguf-facts.md's "config.json is MULTIMODAL" section, read
// off the real gemma-4-E2B-it/config.json). An earlier version of this
// check read num_hidden_layers off the top level directly, which is
// ABSENT there in every real file -- it would have rejected every real
// Gemma 4 package. This is that exact shape, reproduced from the real
// file's structure, not a flat stand-in.
void TestContractAcceptsTheRealMultimodalConfigShape() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{
        {"architectures", {"Gemma4ForConditionalGeneration"}},
        {"model_type", "gemma4"},
        {"eos_token_id", 106},
        {"audio_config", nlohmann::json::object()},
        {"text_config", {{"num_hidden_layers", 35},
                        {"vocab_size", 262144},
                        {"intermediate_size", 6144},   // wrong about layers 15-34; must not be checked
                        {"sliding_window", 512},
                        {"rms_norm_eps", 1.0e-6}}}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);  // must not throw
}

// The catalog's pinned config.json revision states eos_token_id as [1, 106].
void TestContractAcceptsAnEosArrayOnlyWhenItHoldsTheGgufEos() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json empty = nlohmann::json::object();
    const auto config_with = [](nlohmann::json eos) {
        return nlohmann::json{{"architectures", {"Gemma4ForConditionalGeneration"}},
                              {"eos_token_id", std::move(eos)},
                              {"text_config", {{"num_hidden_layers", 35}}}};
    };
    package->ValidateGemma4Contract(config_with({1, 106}), empty, empty);  // must not throw
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config_with({1, 107}), empty, empty);
    });
    RequireContains(error, "eos_token_id");
    RequireContains(error, "106");
}

// A layer-count mismatch nested under text_config -- the shape the real
// package actually ships -- must still be caught. num_hidden_layers is the
// only cross-check field able to catch an E2B/E4B swap at all (vocab_size
// and eos_token_id are identical between the two rows), so this is the one
// that must never become silently optional.
void TestContractRejectsWrongLayerCountInTextConfig() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{
        {"architectures", {"Gemma4ForConditionalGeneration"}},
        {"text_config", {{"num_hidden_layers", 42}}}};   // E4B's count, wrong for this E2B file
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);
    });
    RequireContains(error, "42");
    RequireContains(error, "35");
}

// vocab_size's cross-check, nested under text_config as the real file has
// it, exercised directly: nothing above reached this branch.
void TestContractRejectsVocabSizeMismatchInTextConfig() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{
        {"architectures", {"Gemma4ForConditionalGeneration"}},
        {"text_config", {{"num_hidden_layers", 35}, {"vocab_size", 999}}}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);
    });
    RequireContains(error, "999");
    RequireContains(error, "262144");
}

// The BOS id is the mirror image of the EOS id and lives in the OTHER
// place: `eos_token_id` is a TOP-LEVEL field of the real multimodal
// config.json, while `bos_token_id` appears only inside `text_config` and
// is absent from the top level entirely. A cross-check that learned the eos
// rule and applied it to the bos would read a field no real file has -- a
// check that silently does nothing. This is the one that catches that.
void TestContractRejectsBosTokenIdMismatchInTextConfig() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    TEST_REQUIRE(package->Config().bos_token_id == 2);
    const nlohmann::json config{
        {"architectures", {"Gemma4ForConditionalGeneration"}},
        {"text_config", {{"num_hidden_layers", 35}, {"bos_token_id", 999}}}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    const auto error = RequireThrows([&] {
        package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);
    });
    RequireContains(error, "bos_token_id");
    RequireContains(error, "999");
    RequireContains(error, "2");
}

// And the inversion itself, asserted rather than assumed: a top-level
// bos_token_id is NOT the authority. Real files do not carry one at all, so
// a package that happens to have a stale one must not be rejected over it.
void TestContractReadsTheBosIdFromTextConfigNotTheTopLevel() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    auto package = Gemma4GgufPackage::Open(path);
    const nlohmann::json config{
        {"architectures", {"Gemma4ForConditionalGeneration"}},
        {"bos_token_id", 999},                       // ignored: wrong level
        {"text_config", {{"num_hidden_layers", 35}, {"bos_token_id", 2}}}};
    const nlohmann::json tokenizer = nlohmann::json::object();
    const nlohmann::json tokenizer_config = nlohmann::json::object();
    package->ValidateGemma4Contract(config, tokenizer, tokenizer_config);  // must not throw
}

}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestConfigFromMetadataReadsE2b);
    RUN_TEST(TestConfigFromMetadataReadsE4b);
    RUN_TEST(TestContextLengthIsReadFromTheFile);
    RUN_TEST(TestAFileWithoutAContextLengthGetsOnePage);
    RUN_TEST(TestKvLayersIsBlockCountMinusTheFilesSharingCount);
    RUN_TEST(TestHeadDimKeysAreNotReadTheObviousWayRound);
    RUN_TEST(TestTensorCountIsSixPlusSeventeenPerLayer);
    RUN_TEST(TestRejectsWrongTensorCountNamingBothNumbers);
    RUN_TEST(TestRejectsWrongArchitectureByName);
    RUN_TEST(TestOutputWeightIsAbsentAndNotRequired);
    RUN_TEST(TestTensorCountIsSixPlusSeventeenPerLayerOnPackageOpen);
    RUN_TEST(TestContractDoesNotRejectE2bOverIntermediateSize);
    RUN_TEST(TestContractRejectsWrongLayerCount);
    RUN_TEST(TestContractRejectsArchitectureNotNamingGemma4);
    RUN_TEST(TestContractRejectsTokenizerVocabSizeMismatch);
    RUN_TEST(TestContractAcceptsTheRealMultimodalConfigShape);
    RUN_TEST(TestContractAcceptsAnEosArrayOnlyWhenItHoldsTheGgufEos);
    RUN_TEST(TestContractRejectsWrongLayerCountInTextConfig);
    RUN_TEST(TestContractRejectsVocabSizeMismatchInTextConfig);
    RUN_TEST(TestContractRejectsBosTokenIdMismatchInTextConfig);
    RUN_TEST(TestContractReadsTheBosIdFromTextConfigNotTheTopLevel);
#undef RUN_TEST
}
