/// \file test_oga_hipep.cpp
/// \brief Host-only tests of the oga-hipep engine, package discovery and backend checks
/// \note The OGA C API is a fake defined here: it records every call and
///       answers GetLogits with values that describe the generator's own
///       token sequence, so a test can tell which state the logits came from.
#include "oga/oga_api.hpp"
#include "oga/oga_engine.hpp"
#include "oga/oga_hipep_backend.hpp"
#include "test_support.hpp"

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct OgaResult {
    std::string error;
};
struct OgaConfig {};
struct OgaModel {};
struct OgaGeneratorParams {
    double max_length = 0;
};
struct OgaGenerator {
    /// \brief the tokens the model state has seen, which the logits describe
    std::vector<std::int32_t> tokens;
    double max_length = 0;
};
struct OgaTensor {
    OgaElementType type = OgaElementType_float32;
    std::vector<std::int64_t> shape;
    std::vector<std::byte> data;
};
namespace {

constexpr std::uint32_t kVocab = 64;
constexpr int kMergedToken = 20;
constexpr int kSplitTokenOne = 21;
constexpr int kSplitTokenTwo = 22;

/// \brief what the fake OGA saw and how it answers next
struct FakeOga {
    int generators_created = 0;
    int generators_live = 0;
    int results_destroyed = 0;
    double last_max_length = 0;
    std::vector<std::vector<std::int32_t>> appends;
    std::optional<std::string> fail_next_append;
    OgaElementType logits_type = OgaElementType_float32;
    std::int64_t logits_rows = 0;  ///< 0: rank 2 [1, vocab]; n: rank 3 [1, n, vocab]
    std::int64_t logits_width = kVocab;
    /// \brief raw 16-bit values for the last row's first entries, for f16/bf16
    std::vector<std::uint16_t> raw_head;
    /// \brief handles destroyed, in the order they happened
    std::vector<std::string> events;
};

FakeOga g_fake;

void RecordEvent(std::string event) { g_fake.events.push_back(std::move(event)); }

/// \brief the logits the fake model gives a sequence: [count, last, sum, 0, ...]
std::vector<float> SequenceLogits(const std::vector<std::int32_t>& tokens) {
    std::vector<float> row(kVocab, 0.0f);
    row[0] = static_cast<float>(tokens.size());
    row[1] = tokens.empty() ? -1.0f : static_cast<float>(tokens.back());
    float sum = 0;
    for (const auto token : tokens) sum += static_cast<float>(token);
    row[2] = sum;
    return row;
}

OgaResult* Error(std::string text) { return new OgaResult{std::move(text)}; }

const char* OGA_API_CALL FakeResultGetError(const OgaResult* result) { return result->error.c_str(); }
void OGA_API_CALL FakeDestroyResult(OgaResult* result) {
    ++g_fake.results_destroyed;
    delete result;
}
void OGA_API_CALL FakeDestroyString(const char*) {}
OgaResult* OGA_API_CALL FakeCreateConfig(const char*, OgaConfig** out) {
    *out = new OgaConfig;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeConfigOverlay(OgaConfig*, const char*) { return nullptr; }
OgaResult* OGA_API_CALL FakeConfigSetProviderOption(OgaConfig*, const char*, const char*, const char*) {
    return nullptr;
}
void OGA_API_CALL FakeDestroyConfig(OgaConfig* config) { delete config; }
OgaResult* OGA_API_CALL FakeCreateModelFromConfig(const OgaConfig*, OgaModel** out) {
    *out = new OgaModel;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeModelGetDeviceType(const OgaModel*, const char** out) {
    *out = "AMDGPU";
    return nullptr;
}
void OGA_API_CALL FakeDestroyModel(OgaModel* model) {
    RecordEvent("destroy_model");
    delete model;
}
OgaResult* OGA_API_CALL FakeCreateGeneratorParams(const OgaModel*, OgaGeneratorParams** out) {
    *out = new OgaGeneratorParams;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeSetSearchNumber(OgaGeneratorParams* params, const char* name, double value) {
    if (std::string(name) != "max_length") return Error("unexpected search option");
    params->max_length = value;
    return nullptr;
}
void OGA_API_CALL FakeDestroyGeneratorParams(OgaGeneratorParams* params) { delete params; }
OgaResult* OGA_API_CALL FakeCreateGenerator(const OgaModel*, const OgaGeneratorParams* params,
                                            OgaGenerator** out) {
    ++g_fake.generators_created;
    ++g_fake.generators_live;
    g_fake.last_max_length = params->max_length;
    *out = new OgaGenerator{{}, params->max_length};
    return nullptr;
}
void OGA_API_CALL FakeDestroyGenerator(OgaGenerator* generator) {
    --g_fake.generators_live;
    RecordEvent("destroy_generator");
    delete generator;
}
OgaResult* OGA_API_CALL FakeAppendTokens(OgaGenerator* generator, const std::int32_t* ids, size_t count) {
    if (g_fake.fail_next_append) {
        auto text = *g_fake.fail_next_append;
        g_fake.fail_next_append.reset();
        return Error(text);
    }
    g_fake.appends.emplace_back(ids, ids + count);
    generator->tokens.insert(generator->tokens.end(), ids, ids + count);
    if (generator->tokens.size() > static_cast<std::size_t>(generator->max_length))
        return Error("past max_length");
    return nullptr;
}
OgaResult* OGA_API_CALL FakeGetLogits(OgaGenerator* generator, OgaTensor** out) {
    auto* tensor = new OgaTensor;
    tensor->type = g_fake.logits_type;
    const std::int64_t rows = g_fake.logits_rows ? g_fake.logits_rows : 1;
    tensor->shape = g_fake.logits_rows ? std::vector<std::int64_t>{1, rows, g_fake.logits_width}
                                       : std::vector<std::int64_t>{1, g_fake.logits_width};
    const auto count = static_cast<std::size_t>(rows * g_fake.logits_width);
    const auto last = SequenceLogits(generator->tokens);
    if (g_fake.logits_type == OgaElementType_float32) {
        std::vector<float> values(count, 99.0f);
        for (std::size_t i = 0; i < kVocab && i < static_cast<std::size_t>(g_fake.logits_width); ++i)
            values[count - g_fake.logits_width + i] = last[i];
        tensor->data.resize(count * sizeof(float));
        std::memcpy(tensor->data.data(), values.data(), tensor->data.size());
    } else {
        // Earlier rows hold 65504 (0x7bff), so reading the wrong row shows.
        std::vector<std::uint16_t> values(count, 0x7bff);
        std::fill(values.end() - g_fake.logits_width, values.end(), std::uint16_t{0});
        for (std::size_t i = 0; i < g_fake.raw_head.size(); ++i)
            values[count - g_fake.logits_width + i] = g_fake.raw_head[i];
        tensor->data.resize(count * sizeof(std::uint16_t));
        std::memcpy(tensor->data.data(), values.data(), tensor->data.size());
    }
    *out = tensor;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeTensorGetType(OgaTensor* tensor, OgaElementType* out) {
    *out = tensor->type;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeTensorGetShapeRank(OgaTensor* tensor, size_t* out) {
    *out = tensor->shape.size();
    return nullptr;
}
OgaResult* OGA_API_CALL FakeTensorGetShape(OgaTensor* tensor, std::int64_t* dims, size_t count) {
    if (count != tensor->shape.size()) return Error("rank mismatch");
    std::copy(tensor->shape.begin(), tensor->shape.end(), dims);
    return nullptr;
}
OgaResult* OGA_API_CALL FakeTensorGetData(OgaTensor* tensor, void** out) {
    *out = tensor->data.data();
    return nullptr;
}
void OGA_API_CALL FakeDestroyTensor(OgaTensor* tensor) { delete tensor; }
std::map<std::string, void*> FakeSymbols() {
    return {
        {"OgaResultGetError", reinterpret_cast<void*>(&FakeResultGetError)},
        {"OgaDestroyResult", reinterpret_cast<void*>(&FakeDestroyResult)},
        {"OgaDestroyString", reinterpret_cast<void*>(&FakeDestroyString)},
        {"OgaCreateConfig", reinterpret_cast<void*>(&FakeCreateConfig)},
        {"OgaConfigOverlay", reinterpret_cast<void*>(&FakeConfigOverlay)},
        {"OgaConfigSetProviderOption", reinterpret_cast<void*>(&FakeConfigSetProviderOption)},
        {"OgaDestroyConfig", reinterpret_cast<void*>(&FakeDestroyConfig)},
        {"OgaCreateModelFromConfig", reinterpret_cast<void*>(&FakeCreateModelFromConfig)},
        {"OgaModelGetDeviceType", reinterpret_cast<void*>(&FakeModelGetDeviceType)},
        {"OgaDestroyModel", reinterpret_cast<void*>(&FakeDestroyModel)},
        {"OgaCreateGeneratorParams", reinterpret_cast<void*>(&FakeCreateGeneratorParams)},
        {"OgaGeneratorParamsSetSearchNumber", reinterpret_cast<void*>(&FakeSetSearchNumber)},
        {"OgaDestroyGeneratorParams", reinterpret_cast<void*>(&FakeDestroyGeneratorParams)},
        {"OgaCreateGenerator", reinterpret_cast<void*>(&FakeCreateGenerator)},
        {"OgaDestroyGenerator", reinterpret_cast<void*>(&FakeDestroyGenerator)},
        {"OgaGenerator_AppendTokens", reinterpret_cast<void*>(&FakeAppendTokens)},
        {"OgaGenerator_GetLogits", reinterpret_cast<void*>(&FakeGetLogits)},
        {"OgaTensorGetType", reinterpret_cast<void*>(&FakeTensorGetType)},
        {"OgaTensorGetShapeRank", reinterpret_cast<void*>(&FakeTensorGetShapeRank)},
        {"OgaTensorGetShape", reinterpret_cast<void*>(&FakeTensorGetShape)},
        {"OgaTensorGetData", reinterpret_cast<void*>(&FakeTensorGetData)},
        {"OgaDestroyTensor", reinterpret_cast<void*>(&FakeDestroyTensor)},
    };
}

std::shared_ptr<const flm::oga::OgaApi> FakeApi(std::map<std::string, void*> symbols = FakeSymbols()) {
    return flm::oga::OgaApi::ResolveForTest([symbols](std::string_view name) -> void* {
        const auto it = symbols.find(std::string(name));
        return it == symbols.end() ? nullptr : it->second;
    });
}

/// \brief an engine over `api`, with a model that is a fresh fake
std::unique_ptr<flm::oga::OgaEngine> ConstructOn(std::shared_ptr<const flm::oga::OgaApi> api,
                                                 std::uint32_t max_length = 32,
                                                 std::uint32_t context_limit = 64) {
    flm::oga::OgaApi::Owned<OgaModel> model(new OgaModel, {&FakeDestroyModel});
    return std::make_unique<flm::oga::OgaEngine>(std::move(api), std::move(model), "Test", kVocab,
                                                 context_limit, max_length);
}

/// \brief an engine over the fake as `g_fake` is set up now
std::unique_ptr<flm::oga::OgaEngine> Construct(std::uint32_t max_length = 32,
                                               std::uint32_t context_limit = 64) {
    return ConstructOn(FakeApi(), max_length, context_limit);
}

/// \brief a fresh fake and an engine over it
std::unique_ptr<flm::oga::OgaEngine> MakeEngine(std::uint32_t max_length = 32,
                                                std::uint32_t context_limit = 64) {
    g_fake = FakeOga{};
    return Construct(max_length, context_limit);
}

float At(const buffer<bf16>& logits, std::size_t index) { return static_cast<float>(logits[index]); }

/// \brief the state the logits describe: {token count, last token, token sum}
void RequireState(const buffer<bf16>& logits, float count, float last, float sum) {
    TEST_REQUIRE(logits.size() == kVocab);
    TEST_REQUIRE(At(logits, 0) == count);
    TEST_REQUIRE(At(logits, 1) == last);
    TEST_REQUIRE(At(logits, 2) == sum);
}

using Tokens = std::vector<std::int32_t>;

void TestPrefillThenForwardAppendsOnlyTheNewToken() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1, 2, 3};
    RequireState(engine->prefill(prompt), 3, 3, 6);
    RequireState(engine->forward(4), 4, 4, 10);
    TEST_REQUIRE(g_fake.generators_created == 1);
    TEST_REQUIRE(g_fake.appends == (std::vector<Tokens>{{1, 2, 3}, {4}}));
    TEST_REQUIRE(g_fake.last_max_length == 32);
    TEST_REQUIRE(engine->get_current_context_length() == 4);
}

void TestRestoreIsBookkeepingAndAContinuationReusesTheGenerator() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    TEST_REQUIRE(engine->checkpoint() == 3);
    engine->forward(4);
    engine->forward(5);
    const auto appends = g_fake.appends.size();
    TEST_REQUIRE(engine->restore() == 3);
    TEST_REQUIRE(engine->get_current_context_length() == 3);
    TEST_REQUIRE(g_fake.appends.size() == appends);

    std::vector<int> turn{4, 5, 6};
    RequireState(engine->prefill(turn), 6, 6, 21);
    TEST_REQUIRE(g_fake.generators_created == 1);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{6}));
}

void TestADivergentPassReplaysEverythingInOneAppend() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    engine->checkpoint();
    engine->forward(4);
    engine->restore();
    std::vector<int> other{9};
    RequireState(engine->prefill(other), 4, 9, 15);
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.generators_live == 1);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2, 3, 9}));
}

void TestRetokenizedSequenceReplaysUsingTheNewTokenIds() {
    auto engine = MakeEngine();
    std::vector<int> first_encoding{1, 2, 3, kMergedToken};
    engine->prefill(first_encoding);
    engine->clear_context();

    // These ids can decode to equivalent text, but the token sequence changed.
    std::vector<int> retokenized{1, 2, 3, kSplitTokenOne, kSplitTokenTwo, 4};
    RequireState(engine->prefill(retokenized), 6, 4,
                 1 + 2 + 3 + kSplitTokenOne + kSplitTokenTwo + 4);
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.generators_live == 1);
    TEST_REQUIRE(g_fake.appends.back() ==
                 (Tokens{1, 2, 3, kSplitTokenOne, kSplitTokenTwo, 4}));
}

void TestTheSameSequenceAgainReadsTheLiveLogits() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    engine->clear_context();
    TEST_REQUIRE(engine->get_current_context_length() == 0);
    TEST_REQUIRE(engine->restore() == -1);
    RequireState(engine->prefill(prompt), 3, 3, 6);
    TEST_REQUIRE(g_fake.generators_created == 1);
    TEST_REQUIRE(g_fake.appends.size() == 1);
}

void TestSetContextLengthOnlyMovesTheLogicalEnd() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1, 2, 3, 4};
    engine->prefill(prompt);
    engine->set_context_length(2);
    TEST_REQUIRE(engine->get_current_context_length() == 2);
    TEST_REQUIRE(g_fake.appends.size() == 1);
    RequireContains(RequireThrows<std::runtime_error>([&] { engine->set_context_length(3); }),
                    "cannot rewind past");
    RequireThrows<std::out_of_range>([&] { engine->set_context_length(-1); });
    RequireState(engine->forward(7), 3, 7, 10);
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2, 7}));
}

void TestOverflowThrowsBeforeOgaIsCalled() {
    auto engine = MakeEngine(4);
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    engine->forward(4);
    const auto appends = g_fake.appends.size();
    RequireContains(RequireThrows<std::out_of_range>([&] { engine->forward(5); }),
                    "exceeds configured context capacity");
    TEST_REQUIRE(g_fake.appends.size() == appends);
    TEST_REQUIRE(engine->get_current_context_length() == 4);
}

void TestAnOgaErrorDropsTheGeneratorWithoutPoisoningTheEngine() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    g_fake.fail_next_append = "device lost";
    const auto text = RequireThrows<flm::oga::OgaError>([&] { engine->forward(4); });
    RequireContains(text, "OgaGenerator_AppendTokens failed: device lost");
    TEST_REQUIRE(g_fake.generators_live == 0);
    TEST_REQUIRE(g_fake.results_destroyed == 1);
    TEST_REQUIRE(engine->get_current_context_length() == 3);
    RequireState(engine->forward(4), 4, 4, 10);
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2, 3, 4}));
}

void TestGrowingPastCapacityDropsTheGenerator() {
    auto engine = MakeEngine(8, 64);
    std::vector<int> prompt{1, 2};
    engine->prefill(prompt);
    engine->update_max_length(4);
    TEST_REQUIRE(g_fake.generators_live == 1);
    engine->update_max_length(16);
    TEST_REQUIRE(g_fake.generators_live == 0);
    RequireState(engine->forward(3), 3, 3, 6);
    TEST_REQUIRE(g_fake.last_max_length == 16);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2, 3}));
    RequireThrows<std::out_of_range>([&] { engine->update_max_length(2); });
    RequireThrows<std::out_of_range>([&] { engine->update_max_length(65); });
    RequireThrows<std::out_of_range>([&] { engine->update_max_length(0); });
}

void TestACheckpointLongerThanANewMaximumIsForgotten() {
    auto engine = MakeEngine(8, 64);
    std::vector<int> prompt{1, 2, 3, 4, 5};
    engine->prefill(prompt);
    engine->checkpoint();
    engine->set_context_length(2);
    engine->update_max_length(4);
    TEST_REQUIRE(engine->restore() == -1);
}

void TestFloat16LogitsComeFromTheLastRow() {
    auto engine = MakeEngine();
    g_fake.logits_type = OgaElementType_float16;
    g_fake.logits_rows = 3;
    // 1.0, -2.0, the smallest subnormal, +infinity
    g_fake.raw_head = {0x3c00, 0xc000, 0x0001, 0x7c00};
    std::vector<int> prompt{1};
    const auto logits = engine->prefill(prompt);
    TEST_REQUIRE(At(logits, 0) == 1.0f);
    TEST_REQUIRE(At(logits, 1) == -2.0f);
    TEST_REQUIRE(At(logits, 2) == static_cast<float>(bf16(5.9604645e-8f)));
    TEST_REQUIRE(At(logits, 3) == std::numeric_limits<float>::infinity());
    TEST_REQUIRE(At(logits, 4) == 0.0f);
}

void TestBfloat16LogitsAreCopiedBitForBit() {
    auto engine = MakeEngine();
    g_fake.logits_type = OgaElementType_bfloat16;
    g_fake.raw_head = {0x3f80, 0x4049, 0xff80};
    std::vector<int> prompt{1};
    const auto logits = engine->prefill(prompt);
    TEST_REQUIRE(biovault::get_raw_bits(logits[0]) == 0x3f80);
    TEST_REQUIRE(biovault::get_raw_bits(logits[1]) == 0x4049);
    TEST_REQUIRE(biovault::get_raw_bits(logits[2]) == 0xff80);
}

void TestLogitsOfAnotherWidthOrTypeAreRejected() {
    auto engine = MakeEngine();
    g_fake.logits_width = kVocab + 1;
    std::vector<int> prompt{1};
    RequireContains(RequireThrows<std::runtime_error>([&] { engine->prefill(prompt); }),
                    "logits have shape [1, 65]");
    TEST_REQUIRE(g_fake.generators_live == 0);
    g_fake.logits_width = kVocab;
    g_fake.logits_type = OgaElementType_int32;
    RequireContains(RequireThrows<std::runtime_error>([&] { engine->prefill(prompt); }),
                    "not float32, float16 or bfloat16");
}

void TestCallsTheEngineDoesNotSupportThrow() {
    auto engine = MakeEngine();
    std::vector<int> prompt{1};
    int payload = 0;
    RequireContains(RequireThrows<std::invalid_argument>([&] { engine->prefill(prompt, &payload); }),
                    "vision payload");
    std::vector<int> empty;
    RequireThrows<std::invalid_argument>([&] { engine->prefill(empty); });
    std::vector<int> outside{static_cast<int>(kVocab)};
    RequireContains(RequireThrows<std::out_of_range>([&] { engine->prefill(outside); }),
                    "outside the vocabulary");
    RequireThrows<std::runtime_error>([&] { engine->get_k_cache(0, 0); });
    RequireThrows<std::runtime_error>([&] { engine->get_v_cache(0, 0); });
    TEST_REQUIRE(g_fake.generators_created == 0);
}

void TestAMissingSymbolNamesIt() {
    auto symbols = FakeSymbols();
    symbols.erase("OgaGenerator_GetLogits");
    RequireContains(RequireThrows<std::runtime_error>([&] { FakeApi(symbols); }),
                    "missing OGA symbol: OgaGenerator_GetLogits");
}

// ---- OGA lifetime: a new API after the last owner went ----------------------
// OgaShutdown() is never called: on hip-ep 0.5.2 a generator created after it
// crashes inside onnxruntime.dll (see OgaApi::GetOrCreate).

std::vector<std::string> Events() { return g_fake.events; }

std::size_t CountOf(const std::vector<std::string>& events, const std::string& event) {
    return static_cast<std::size_t>(std::count(events.begin(), events.end(), event));
}

void TestAFreshApiAfterFullReleaseWorksAgain() {
    g_fake = FakeOga{};
    int loads = 0;
    const auto load = [&loads] {
        ++loads;
        return FakeApi();
    };
    auto api = flm::oga::OgaApi::GetOrCreateWith(load);
    TEST_REQUIRE(loads == 1);
    // While one is held, the same API comes back and nothing loads again.
    TEST_REQUIRE(flm::oga::OgaApi::GetOrCreateWith(load).get() == api.get());
    TEST_REQUIRE(loads == 1);

    std::vector<int> prompt{1, 2, 3};
    auto engine = ConstructOn(api);
    RequireState(engine->prefill(prompt), 3, 3, 6);
    engine.reset();
    api.reset();
    // Every handle of the first engine is gone before the next API loads.
    TEST_REQUIRE(CountOf(Events(), "destroy_model") == 1);

    auto again = flm::oga::OgaApi::GetOrCreateWith(load);
    TEST_REQUIRE(loads == 2);
    auto second = ConstructOn(again);
    RequireState(second->prefill(prompt), 3, 3, 6);
    second.reset();
    again.reset();
    TEST_REQUIRE(CountOf(Events(), "destroy_model") == 2);
}

// ---- package discovery ----------------------------------------------------

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() /
               ("flm-oga-hipep-test-" + std::to_string(GetCurrentProcessId()) + "-" +
                std::to_string(counter()++));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    static int& counter() {
        static int value = 0;
        return value;
    }
};

void Touch(const std::filesystem::path& file, std::string_view text = "") {
    std::filesystem::create_directories(file.parent_path());
    std::ofstream(file, std::ios::binary) << text;
}

const char* kPackageFiles[] = {"onnxruntime-genai.dll", "onnxruntime.dll", "amdgpu-ep.dll",
                               "hip-backend.dll", "hipgpu.dll"};

std::string Reason(const flm::oga::PackageCandidate& candidate) {
    auto inspected = flm::oga::InspectPackage(candidate);
    TEST_REQUIRE(std::holds_alternative<std::string>(inspected));
    return std::get<std::string>(inspected);
}

void TestDiscoveryReportsWhyEachRootIsRejected() {
    TEST_REQUIRE(Reason({"AMDRT_PKG_ROOT", {}}) == "not set");

    TempDir missing;
    RequireContains(Reason({"AMDRT_PKG_ROOT", missing.path / "absent"}), "is not a directory");

    TempDir partial;
    Touch(partial.path / "bin" / "onnxruntime-genai.dll");
    RequireContains(Reason({"AMDRT_PKG_ROOT", partial.path}),
                    "has no onnxruntime.dll, amdgpu-ep.dll, hip-backend.dll, hipgpu.dll");

    TempDir unversioned;
    for (const char* file : kPackageFiles) Touch(unversioned.path / "bin" / file);
    RequireContains(Reason({"AMDRT_PKG_ROOT", unversioned.path}), "has no file version");
}

void TestDiscoveryAcceptsAVersionedPackageWithoutABinDirectory() {
    TempDir root;
    for (const char* file : kPackageFiles) Touch(root.path / file);
    // Any DLL with a version resource of 0.17 or later stands in for OGA here.
    std::filesystem::copy_file("C:\\Windows\\System32\\version.dll",
                               root.path / "onnxruntime-genai.dll",
                               std::filesystem::copy_options::overwrite_existing);
    auto inspected = flm::oga::InspectPackage({"test", root.path});
    TEST_REQUIRE(std::holds_alternative<flm::oga::Package>(inspected));
    const auto& package = std::get<flm::oga::Package>(inspected);
    TEST_REQUIRE(package.bin == root.path);
    TEST_REQUIRE(package.source == "test");
    TEST_REQUIRE(package.oga_version.major >= 1);
}

void TestFindPackageTakesTheFirstThatQualifiesAndListsEveryRejection() {
    TempDir good;
    for (const char* file : kPackageFiles) Touch(good.path / "bin" / file);
    std::filesystem::copy_file("C:\\Windows\\System32\\version.dll",
                               good.path / "bin" / "onnxruntime-genai.dll",
                               std::filesystem::copy_options::overwrite_existing);
    const auto found = flm::oga::FindPackage({{"first", {}}, {"second", good.path}});
    TEST_REQUIRE(found.source == "second");
    TEST_REQUIRE(found.bin == good.path / "bin");

    TempDir empty;
    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::FindPackage({{"AMDRT_PKG_ROOT", {}}, {"other", empty.path}}); });
    RequireContains(text, "AMDRT_PKG_ROOT: not set");
    RequireContains(text, "other: '" + (empty.path).string() + "' has no onnxruntime-genai.dll");
    RequireContains(text, "set AMDRT_PKG_ROOT");
}

/// \brief a complete hip-ep package under `root`, with a version resource standing in for OGA
void MakePackage(const std::filesystem::path& root) {
    for (const char* file : kPackageFiles) Touch(root / "bin" / file);
    std::filesystem::copy_file("C:\\Windows\\System32\\version.dll", root / "bin" / "onnxruntime-genai.dll",
                               std::filesystem::copy_options::overwrite_existing);
}

/// \brief an environment in which the variables discovery reads have these values; empty is unset
flm::oga::PackageEnvironment Environment(std::filesystem::path root, std::filesystem::path local_app_data,
                                         std::filesystem::path program_files) {
    return [root, local_app_data, program_files](const wchar_t* name) -> std::filesystem::path {
        const std::wstring_view variable(name);
        if (variable == L"AMDRT_PKG_ROOT") return root;
        if (variable == L"LOCALAPPDATA") return local_app_data;
        if (variable == L"ProgramFiles") return program_files;
        return {};
    };
}

void TestCandidatesComeFromAmdrtPkgRootOnly() {
    TempDir local;
    MakePackage(local.path / "hip-ep" / "0.5.2");
    const auto set = flm::oga::SearchPackages(Environment("D:\\somewhere\\hip-ep", local.path, {}));
    TEST_REQUIRE(set.candidates.size() == 1);
    TEST_REQUIRE(set.candidates[0].source == "AMDRT_PKG_ROOT");
    TEST_REQUIRE(set.candidates[0].root == std::filesystem::path("D:\\somewhere\\hip-ep"));
    TEST_REQUIRE(set.notes.empty());

    const auto unset = flm::oga::SearchPackages(Environment({}, local.path, {}));
    TEST_REQUIRE(unset.candidates.size() == 1);
    TEST_REQUIRE(unset.candidates[0].source == "%LOCALAPPDATA%\\hip-ep\\0.5.2");
}

void TestDefaultDiscoveryTriesTheNewestVersionFirst() {
    TempDir local;
    TempDir programs;
    MakePackage(local.path / "hip-ep" / "0.9.0");
    MakePackage(local.path / "hip-ep" / "0.10.0");
    MakePackage(programs.path / "hip-ep" / "v0.5.2");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    const auto found = flm::oga::FindPackage(search.candidates, search.notes);
    // 0.10.0 is newer than 0.9.0 by number, though it sorts first as text.
    TEST_REQUIRE(found.source == "%LOCALAPPDATA%\\hip-ep\\0.10.0");
    TEST_REQUIRE(found.bin == local.path / "hip-ep" / "0.10.0" / "bin");
}

void TestAFolderNameThatIsNotAVersionIsTriedLast() {
    TempDir local;
    MakePackage(local.path / "hip-ep" / "nightly");
    MakePackage(local.path / "hip-ep" / "0.5.2");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, {}));
    const auto found = flm::oga::FindPackage(search.candidates, search.notes);
    TEST_REQUIRE(found.source == "%LOCALAPPDATA%\\hip-ep\\0.5.2");
}

void TestTheNewestVersionWinsAcrossBothFolders() {
    TempDir local;
    TempDir programs;
    MakePackage(local.path / "hip-ep" / "0.5.1");
    MakePackage(programs.path / "hip-ep" / "0.5.2");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    const auto found = flm::oga::FindPackage(search.candidates, search.notes);
    TEST_REQUIRE(found.source == "%ProgramFiles%\\hip-ep\\0.5.2");
}

void TestAnEqualVersionInBothFoldersPrefersLocalAppData() {
    TempDir local;
    TempDir programs;
    MakePackage(local.path / "hip-ep" / "0.5.2");
    MakePackage(programs.path / "hip-ep" / "0.5.2");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    const auto found = flm::oga::FindPackage(search.candidates, search.notes);
    TEST_REQUIRE(found.source == "%LOCALAPPDATA%\\hip-ep\\0.5.2");
}

void TestARejectedVersionIsSkippedForTheNextOne() {
    TempDir local;
    TempDir programs;
    Touch(local.path / "hip-ep" / "0.5.2" / "bin" / "onnxruntime-genai.dll");
    MakePackage(programs.path / "hip-ep" / "0.5.1");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    const auto found = flm::oga::FindPackage(search.candidates, search.notes);
    TEST_REQUIRE(found.source == "%ProgramFiles%\\hip-ep\\0.5.1");
}

void TestWhenEveryVersionIsRejectedEachReasonIsListedInOrder() {
    TempDir local;
    TempDir programs;
    Touch(local.path / "hip-ep" / "0.5.2" / "bin" / "onnxruntime-genai.dll");
    Touch(programs.path / "hip-ep" / "0.5.1" / "bin" / "onnxruntime-genai.dll");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::FindPackage(search.candidates, search.notes); });
    const std::string local_bin = (local.path / "hip-ep" / "0.5.2" / "bin").string();
    const std::string programs_bin = (programs.path / "hip-ep" / "0.5.1" / "bin").string();
    RequireContains(text, "%LOCALAPPDATA%\\hip-ep\\0.5.2: '" + local_bin + "' has no onnxruntime.dll");
    RequireContains(text, "%ProgramFiles%\\hip-ep\\0.5.1: '" + programs_bin + "' has no onnxruntime.dll");
    TEST_REQUIRE(text.find("%LOCALAPPDATA%\\hip-ep\\0.5.2") < text.find("%ProgramFiles%\\hip-ep\\0.5.1"));
}

void TestAnExplicitRootThatIsRejectedDoesNotFallThrough() {
    TempDir local;
    TempDir programs;
    TempDir explicit_root;
    MakePackage(local.path / "hip-ep" / "0.5.2");
    MakePackage(programs.path / "hip-ep" / "0.5.2");
    const auto search =
        flm::oga::SearchPackages(Environment(explicit_root.path, local.path, programs.path));
    TEST_REQUIRE(search.candidates.size() == 1);
    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::FindPackage(search.candidates, search.notes); });
    RequireContains(text, "AMDRT_PKG_ROOT: '" + explicit_root.path.string() + "' has no onnxruntime-genai.dll");
    // The default folders are not named anywhere in the error, so none was tried.
    TEST_REQUIRE(text.find(local.path.string()) == std::string::npos);
    TEST_REQUIRE(text.find(programs.path.string()) == std::string::npos);
}

void TestWhenNoFolderHoldsAPackageTheErrorListsBothFolders() {
    TempDir local;
    TempDir programs;
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    TEST_REQUIRE(search.candidates.empty());
    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::FindPackage(search.candidates, search.notes); });
    RequireContains(text, "'" + (local.path / "hip-ep").string() + "' does not exist");
    RequireContains(text, "'" + (programs.path / "hip-ep").string() + "' does not exist");
    RequireContains(text, "Install hip-ep 0.5.2 or later");
    RequireContains(text, "Unzip it into %LOCALAPPDATA%\\hip-ep\\<version>");

    const auto unset = flm::oga::SearchPackages(Environment({}, {}, {}));
    const auto unset_text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::FindPackage(unset.candidates, unset.notes); });
    RequireContains(unset_text, "%LOCALAPPDATA% is not set");
    RequireContains(unset_text, "%ProgramFiles% is not set");
}

void TestProgramFilesIsUsedWhenLocalAppDataHasNothing() {
    TempDir local;
    TempDir programs;
    MakePackage(programs.path / "hip-ep" / "0.5.2");
    const auto search = flm::oga::SearchPackages(Environment({}, local.path, programs.path));
    const auto found = flm::oga::FindPackage(search.candidates, search.notes);
    TEST_REQUIRE(found.source == "%ProgramFiles%\\hip-ep\\0.5.2");
}

void TestAVersionInsideAFolderNameStillWinsByItsNumber() {
    // Extract All names the folder after the zip, so the version is only part of the name.
    TempDir extracted;
    MakePackage(extracted.path / "hip-ep" / "hipep_0.5.2_windows");
    MakePackage(extracted.path / "hip-ep" / "hipep_0.5.3_windows");
    const auto extracted_search = flm::oga::SearchPackages(Environment({}, extracted.path, {}));
    TEST_REQUIRE(flm::oga::FindPackage(extracted_search.candidates, extracted_search.notes).source ==
                 "%LOCALAPPDATA%\\hip-ep\\hipep_0.5.3_windows");

    TempDir decorated;
    MakePackage(decorated.path / "hip-ep" / "0.5.2");
    MakePackage(decorated.path / "hip-ep" / "hip-ep-0.5.3");
    const auto decorated_search = flm::oga::SearchPackages(Environment({}, decorated.path, {}));
    TEST_REQUIRE(flm::oga::FindPackage(decorated_search.candidates, decorated_search.notes).source ==
                 "%LOCALAPPDATA%\\hip-ep\\hip-ep-0.5.3");

    TempDir named;
    MakePackage(named.path / "hip-ep" / "v0.5.2");
    MakePackage(named.path / "hip-ep" / "0.5.3");
    MakePackage(named.path / "hip-ep" / "latest");
    const auto named_search = flm::oga::SearchPackages(Environment({}, named.path, {}));
    TEST_REQUIRE(named_search.candidates.size() == 3);
    TEST_REQUIRE(named_search.candidates.back().source == "%LOCALAPPDATA%\\hip-ep\\latest");
}

// ---- backend checks that run before hip-ep is loaded ------------------------

/// \brief a model directory with just the files the backend reads before loading hip-ep
struct PackageDir {
    TempDir dir;
    LM_Config config;
    explicit PackageDir(std::uint32_t vocab = kVocab, const char* eos_text = "<|im_end|>") {
        Touch(dir.path / "genai_config.json",
              R"({"model":{"type":"test_type","vocab_size":)" + std::to_string(kVocab) +
                  R"(,"context_length":128,"eos_token_id":[7]}})");
        Touch(dir.path / "tokenizer_config.json",
              std::string(R"({"bos_token":null,"eos_token":")") + eos_text + R"("})");
        Touch(dir.path / "tokenizer.json", R"({"added_tokens":[{"id":5,"content":"<|im_end|>"}]})");
        config._json_config = {{"vocab_size", vocab}};
    }
    flm::backend::BackendContext Context(std::uint32_t length = 64, bool preemption = false) const {
        flm::backend::BackendContext context;
        context.model_path = dir.path.string();
        context.config = &config;
        context.context_length = length;
        context.enable_preemption = preemption;
        return context;
    }
};

void TestTheBackendRejectsWhatThePackageCannotRun() {
    const auto factory = flm::oga::hipep_factory();
    PackageDir package;
    RequireContains(RequireThrows<std::invalid_argument>([&] { factory(package.Context(64, true)); }),
                    "does not support preemption");
    RequireContains(RequireThrows<std::out_of_range>([&] { factory(package.Context(0)); }),
                    "context length must be in 1..128");
    RequireThrows<std::out_of_range>([&] { factory(package.Context(129)); });

    PackageDir wider(kVocab + 1);
    RequireContains(RequireThrows<std::runtime_error>([&] { factory(wider.Context()); }),
                    "differs from genai_config.json vocab_size");

    PackageDir unknown_eos(kVocab, "<|unknown|>");
    RequireContains(RequireThrows<std::runtime_error>([&] { factory(unknown_eos.Context()); }),
                    "tokenizer.json has no id for eos_token <|unknown|>");
}

void TestTheBackendSaysHowToGetHipep() {
    // AMDRT_PKG_ROOT names a folder that does not exist, so the outcome does not depend on
    // what hip-ep this machine has installed, and the real environment is what is read.
    TempDir missing;
    const std::string absent = (missing.path / "absent").string();
    _putenv_s("AMDRT_PKG_ROOT", absent.c_str());
    PackageDir package;
    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::hipep_factory()(package.Context()); });
    _putenv_s("AMDRT_PKG_ROOT", "");
    RequireContains(text, "AMDRT_PKG_ROOT: '" + absent + "' is not a directory");
    RequireContains(text, "Install hip-ep 0.5.2 or later");
}

void TestConstructionDoesNotRunInference() {
    g_fake = FakeOga{};
    auto engine = Construct();
    TEST_REQUIRE(g_fake.generators_created == 0);
    TEST_REQUIRE(g_fake.appends.empty());
}

void TestBackendCapacityTracksContextUpdates() {
    g_fake = FakeOga{};
    auto api = flm::oga::OgaApi::GetOrCreateWith([] { return FakeApi(); });
    PackageDir package;
    auto backend = flm::oga::hipep_factory()(package.Context(32));
    TEST_REQUIRE(backend->max_decode_length() == 31);
    backend->engine().update_max_length(64);
    TEST_REQUIRE(backend->max_decode_length() == 63);
    backend->engine().update_max_length(16);
    TEST_REQUIRE(backend->max_decode_length() == 15);
}

}  // namespace

int main() {
#define RUN(name) RunTest(&name, #name)
    RUN(TestConstructionDoesNotRunInference);
    RUN(TestBackendCapacityTracksContextUpdates);
    RUN(TestPrefillThenForwardAppendsOnlyTheNewToken);
    RUN(TestRestoreIsBookkeepingAndAContinuationReusesTheGenerator);
    RUN(TestADivergentPassReplaysEverythingInOneAppend);
    RUN(TestRetokenizedSequenceReplaysUsingTheNewTokenIds);
    RUN(TestTheSameSequenceAgainReadsTheLiveLogits);
    RUN(TestSetContextLengthOnlyMovesTheLogicalEnd);
    RUN(TestOverflowThrowsBeforeOgaIsCalled);
    RUN(TestAnOgaErrorDropsTheGeneratorWithoutPoisoningTheEngine);
    RUN(TestGrowingPastCapacityDropsTheGenerator);
    RUN(TestACheckpointLongerThanANewMaximumIsForgotten);
    RUN(TestFloat16LogitsComeFromTheLastRow);
    RUN(TestBfloat16LogitsAreCopiedBitForBit);
    RUN(TestLogitsOfAnotherWidthOrTypeAreRejected);
    RUN(TestCallsTheEngineDoesNotSupportThrow);
    RUN(TestAMissingSymbolNamesIt);
    RUN(TestAFreshApiAfterFullReleaseWorksAgain);
    RUN(TestDiscoveryReportsWhyEachRootIsRejected);
    RUN(TestDiscoveryAcceptsAVersionedPackageWithoutABinDirectory);
    RUN(TestFindPackageTakesTheFirstThatQualifiesAndListsEveryRejection);
    RUN(TestCandidatesComeFromAmdrtPkgRootOnly);
    RUN(TestDefaultDiscoveryTriesTheNewestVersionFirst);
    RUN(TestAFolderNameThatIsNotAVersionIsTriedLast);
    RUN(TestTheNewestVersionWinsAcrossBothFolders);
    RUN(TestAnEqualVersionInBothFoldersPrefersLocalAppData);
    RUN(TestARejectedVersionIsSkippedForTheNextOne);
    RUN(TestWhenEveryVersionIsRejectedEachReasonIsListedInOrder);
    RUN(TestAnExplicitRootThatIsRejectedDoesNotFallThrough);
    RUN(TestWhenNoFolderHoldsAPackageTheErrorListsBothFolders);
    RUN(TestProgramFilesIsUsedWhenLocalAppDataHasNothing);
    RUN(TestAVersionInsideAFolderNameStillWinsByItsNumber);
    RUN(TestTheBackendRejectsWhatThePackageCannotRun);
    RUN(TestTheBackendSaysHowToGetHipep);
#undef RUN
    std::cout << "All oga-hipep tests passed\n";
    return 0;
}
