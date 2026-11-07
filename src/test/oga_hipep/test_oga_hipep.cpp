// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file test_oga_hipep.cpp
/// \brief Host-only tests of the oga-hipep engine, package discovery and backend checks
/// \note The OGA C API is a fake defined here: it records every call and
///       answers GetLogits with values that describe the generator's own
///       token sequence, so a test can tell which state the logits came from.
#include "oga/oga_api.hpp"
#include "oga/oga_engine.hpp"
#include "oga/oga_hipep_backend.hpp"
#include "test_support.hpp"
#include "../../common/oga/oga_runtime_install.hpp"

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
    /// \brief the length of the last OgaGenerator_SnapshotState
    std::optional<std::size_t> snapshot;
};
struct OgaTensor {
    OgaElementType type = OgaElementType_float32;
    std::vector<std::int64_t> shape;
    std::vector<std::byte> data;
};
struct OgaNamedTensors {
    std::map<std::string, OgaTensor> tensors;
};
namespace {

constexpr std::uint32_t kVocab = 64;
constexpr int kMergedToken = 20;
constexpr int kSplitTokenOne = 21;
constexpr int kSplitTokenTwo = 22;
constexpr int kImageToken = 40;
constexpr int kVisionStart = 41;
constexpr std::int64_t kPatchValues = 4;

/// \brief what the fake OGA saw and how it answers next
struct FakeOga {
    /// \brief what one OgaGenerator_SetInputs call was given
    struct Inputs {
        std::vector<std::int32_t> ids;
        std::vector<std::int64_t> pixel_shape;
        std::vector<std::uint16_t> pixels;
        std::vector<std::int64_t> grids;
        std::vector<std::int64_t> image_tokens;
    };

    int generators_created = 0;
    int generators_live = 0;
    int results_destroyed = 0;
    double last_max_length = 0;
    std::vector<std::vector<std::int32_t>> appends;
    std::optional<std::string> fail_next_append;
    std::vector<Inputs> set_inputs;
    int snapshots = 0;
    std::optional<std::string> fail_next_snapshot;
    /// \brief the length of each OgaGenerator_RewindTo call
    std::vector<std::size_t> rewinds;
    std::optional<std::string> fail_next_rewind;
    OgaElementType logits_type = OgaElementType_float32;
    std::int64_t logits_rows = 0;  ///< 0: rank 2 [1, vocab]; n: rank 3 [1, n, vocab]
    std::int64_t logits_width = kVocab;
    /// \brief raw 16-bit values for the last row's first entries, for f16/bf16
    std::vector<std::uint16_t> raw_head;
    /// \brief handles destroyed, in the order they happened
    std::vector<std::string> events;
    /// \brief the decoder's providers, which a config starts with from the package
    std::vector<std::string> providers{"CPU"};
    std::vector<std::string> overlays;
    std::string device = "AMDGPU";
    /// \brief each OgaRegisterExecutionProviderLibrary call, as name and path
    std::vector<std::pair<std::string, std::string>> registrations;
    std::optional<std::string> fail_registration;
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
void OGA_API_CALL FakeRegisterExecutionProviderLibrary(const char* name, const char* path) {
    if (g_fake.fail_registration) throw std::runtime_error(*g_fake.fail_registration);
    g_fake.registrations.emplace_back(name, path);
}
OgaResult* OGA_API_CALL FakeCreateConfig(const char*, OgaConfig** out) {
    *out = new OgaConfig;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeConfigClearProviders(OgaConfig*) {
    g_fake.providers.clear();
    return nullptr;
}
OgaResult* OGA_API_CALL FakeConfigAppendProvider(OgaConfig*, const char* provider) {
    g_fake.providers.emplace_back(provider);
    return nullptr;
}
OgaResult* OGA_API_CALL FakeConfigOverlay(OgaConfig*, const char* json) {
    g_fake.overlays.emplace_back(json);
    return nullptr;
}
void OGA_API_CALL FakeDestroyConfig(OgaConfig* config) { delete config; }
OgaResult* OGA_API_CALL FakeCreateModelFromConfig(const OgaConfig*, OgaModel** out) {
    *out = new OgaModel;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeModelGetDeviceType(const OgaModel*, const char** out) {
    *out = g_fake.device.c_str();
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

std::size_t ElementSize(OgaElementType type) {
    switch (type) {
    case OgaElementType_int32:
    case OgaElementType_float32: return 4;
    case OgaElementType_int64: return 8;
    case OgaElementType_float16:
    case OgaElementType_bfloat16: return 2;
    default: return 0;
    }
}
OgaResult* OGA_API_CALL FakeCreateTensorFromBuffer(void* data, const std::int64_t* shape, size_t rank,
                                                   OgaElementType type, OgaTensor** out) {
    // The engine has OGA own every tensor it hands over.
    if (data) return Error("the fake allocates every tensor itself");
    if (!ElementSize(type)) return Error("unexpected element type");
    auto* tensor = new OgaTensor;
    tensor->type = type;
    tensor->shape.assign(shape, shape + rank);
    std::size_t count = 1;
    for (std::size_t i = 0; i < rank; ++i) count *= static_cast<std::size_t>(shape[i]);
    tensor->data.resize(count * ElementSize(type));
    *out = tensor;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeCreateNamedTensors(OgaNamedTensors** out) {
    *out = new OgaNamedTensors;
    return nullptr;
}
OgaResult* OGA_API_CALL FakeNamedTensorsSet(OgaNamedTensors* named, const char* name, OgaTensor* tensor) {
    named->tensors[name] = *tensor;
    return nullptr;
}
void OGA_API_CALL FakeDestroyNamedTensors(OgaNamedTensors* named) { delete named; }

template <typename T>
std::vector<T> Values(const OgaTensor& tensor) {
    std::vector<T> values(tensor.data.size() / sizeof(T));
    std::memcpy(values.data(), tensor.data.data(), values.size() * sizeof(T));
    return values;
}
OgaResult* OGA_API_CALL FakeSetInputs(OgaGenerator* generator, const OgaNamedTensors* named) {
    // As in OGA, only a generator's first call can carry images.
    if (!generator->tokens.empty()) return Error("SetInputs on a generator that has run tokens");
    const auto& t = named->tensors;
    for (const char* name : {"input_ids", "pixel_values"})
        if (!t.count(name)) return Error(std::string("no ") + name);
    const OgaTensor& ids = t.at("input_ids");
    const OgaTensor& pixels = t.at("pixel_values");
    if (ids.type != OgaElementType_int32 || ids.shape.size() != 2 || ids.shape[0] != 1)
        return Error("input_ids is not int32 [1, n]");
    FakeOga::Inputs inputs;
    if (!t.count("image_grid_thw") && !t.count("num_image_tokens")) {
        // a text prompt's batch of one image without pixels
        if (pixels.type != OgaElementType_float16 || pixels.shape != std::vector<std::int64_t>{1, 0, 0})
            return Error("pixel_values is not float16 [1, 0, 0]");
    } else {
        for (const char* name : {"image_grid_thw", "num_image_tokens"})
            if (!t.count(name)) return Error(std::string("no ") + name);
        const OgaTensor& grids = t.at("image_grid_thw");
        const OgaTensor& counts = t.at("num_image_tokens");
        if (pixels.type != OgaElementType_float16 || pixels.shape.size() != 2)
            return Error("pixel_values is not float16 [patches, values]");
        if (grids.type != OgaElementType_int64 || grids.shape.size() != 2 || grids.shape[1] != 3)
            return Error("image_grid_thw is not int64 [images, 3]");
        if (counts.type != OgaElementType_int64 || counts.shape.size() != 1 || counts.shape[0] != grids.shape[0])
            return Error("num_image_tokens is not int64 [images]");
        inputs.grids = Values<std::int64_t>(grids);
        inputs.image_tokens = Values<std::int64_t>(counts);
    }
    inputs.ids = Values<std::int32_t>(ids);
    inputs.pixel_shape = pixels.shape;
    inputs.pixels = Values<std::uint16_t>(pixels);
    generator->tokens = inputs.ids;
    g_fake.set_inputs.push_back(std::move(inputs));
    if (generator->tokens.size() > static_cast<std::size_t>(generator->max_length))
        return Error("past max_length");
    return nullptr;
}
OgaResult* OGA_API_CALL FakeSnapshotState(OgaGenerator* generator) {
    if (g_fake.fail_next_snapshot) {
        auto text = *g_fake.fail_next_snapshot;
        g_fake.fail_next_snapshot.reset();
        return Error(text);
    }
    ++g_fake.snapshots;
    generator->snapshot = generator->tokens.size();
    return nullptr;
}
OgaResult* OGA_API_CALL FakeRewindTo(OgaGenerator* generator, size_t length) {
    g_fake.rewinds.push_back(length);
    if (g_fake.fail_next_rewind) {
        auto text = *g_fake.fail_next_rewind;
        g_fake.fail_next_rewind.reset();
        return Error(text);
    }
    // OGA 0.19 can bring a hybrid model's recurrent state back only from its snapshot.
    if (generator->snapshot != length || length > generator->tokens.size())
        return Error("no snapshot at length " + std::to_string(length));
    generator->tokens.resize(length);
    return nullptr;
}

std::map<std::string, void*> FakeSymbols() {
    return {
        {"OgaResultGetError", reinterpret_cast<void*>(&FakeResultGetError)},
        {"OgaDestroyResult", reinterpret_cast<void*>(&FakeDestroyResult)},
        {"OgaDestroyString", reinterpret_cast<void*>(&FakeDestroyString)},
        {"OgaRegisterExecutionProviderLibrary", reinterpret_cast<void*>(&FakeRegisterExecutionProviderLibrary)},
        {"OgaCreateConfig", reinterpret_cast<void*>(&FakeCreateConfig)},
        {"OgaConfigClearProviders", reinterpret_cast<void*>(&FakeConfigClearProviders)},
        {"OgaConfigAppendProvider", reinterpret_cast<void*>(&FakeConfigAppendProvider)},
        {"OgaConfigOverlay", reinterpret_cast<void*>(&FakeConfigOverlay)},
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
        {"OgaCreateTensorFromBuffer", reinterpret_cast<void*>(&FakeCreateTensorFromBuffer)},
        {"OgaCreateNamedTensors", reinterpret_cast<void*>(&FakeCreateNamedTensors)},
        {"OgaNamedTensorsSet", reinterpret_cast<void*>(&FakeNamedTensorsSet)},
        {"OgaDestroyNamedTensors", reinterpret_cast<void*>(&FakeDestroyNamedTensors)},
        {"OgaGenerator_SetInputs", reinterpret_cast<void*>(&FakeSetInputs)},
        {"OgaGenerator_SnapshotState", reinterpret_cast<void*>(&FakeSnapshotState)},
        {"OgaGenerator_RewindTo", reinterpret_cast<void*>(&FakeRewindTo)},
    };
}

std::shared_ptr<const flm::oga::OgaApi> FakeApi(std::map<std::string, void*> symbols = FakeSymbols(),
                                                flm::oga::Package package = {}) {
    return flm::oga::OgaApi::ResolveForTest(
        [symbols](std::string_view name) -> void* {
            const auto it = symbols.find(std::string(name));
            return it == symbols.end() ? nullptr : it->second;
        },
        std::move(package));
}

/// \brief the fake API of a package whose DLLs are in `bin`
std::shared_ptr<const flm::oga::OgaApi> FakeApiIn(std::filesystem::path bin) {
    return FakeApi(FakeSymbols(), {"test", std::move(bin), {0, 17, 0, 0}});
}

/// \brief an engine over `api`, with a model that is a fresh fake
std::unique_ptr<flm::oga::OgaEngine> ConstructOn(std::shared_ptr<const flm::oga::OgaApi> api,
                                                 std::uint32_t max_length = 32,
                                                 std::uint32_t context_limit = 64,
                                                 std::optional<flm::oga::OgaVisionConfig> vision = std::nullopt) {
    flm::oga::OgaApi::Owned<OgaModel> model(new OgaModel, {&FakeDestroyModel});
    return std::make_unique<flm::oga::OgaEngine>(std::move(api), std::move(model), "Test", kVocab,
                                                 context_limit, max_length, std::move(vision));
}

/// \brief a fresh fake and an engine over an OGA of `version`
std::unique_ptr<flm::oga::OgaEngine> MakeEngineOn(flm::oga::FileVersion version,
                                                  std::map<std::string, void*> symbols = FakeSymbols()) {
    g_fake = FakeOga{};
    return ConstructOn(FakeApi(std::move(symbols), {"test", {}, version}));
}

/// \brief 2x2 patches to an image token, 4 values to a patch, 64 patches to a prompt
flm::oga::OgaVisionConfig VisionConfig() {
    flm::oga::OgaVisionConfig config;
    config.image_token_id = kImageToken;
    config.vision_start_token_id = kVisionStart;
    config.spatial_merge_size = 2;
    config.patch_values = kPatchValues;
    config.max_patches = 64;
    return config;
}

/// \brief a fresh fake and an engine over it that runs images
std::unique_ptr<flm::oga::OgaEngine> MakeVisionEngine(flm::oga::FileVersion version = {0, 17, 0, 0}) {
    g_fake = FakeOga{};
    return ConstructOn(FakeApi(FakeSymbols(), {"test", {}, version}), 64, 128, VisionConfig());
}

/// \brief an image of `h` x `w` patches, every value `fill`
flm::oga::OgaImage Image(std::int64_t h, std::int64_t w, std::uint16_t fill = 0x3c00) {
    flm::oga::OgaImage image;
    image.grid_h = h;
    image.grid_w = w;
    image.pixels.assign(static_cast<std::size_t>(h * w * kPatchValues), fill);
    return image;
}

/// \brief `before`, then the vision start and image tokens of `image`, then `after`
std::vector<int> WithImage(std::vector<int> before, const flm::oga::OgaImage& image,
                           const std::vector<int>& after) {
    before.push_back(kVisionStart);
    before.insert(before.end(), static_cast<std::size_t>(image.grid_h * image.grid_w / 4), kImageToken);
    before.insert(before.end(), after.begin(), after.end());
    return before;
}

std::vector<int> Concat(std::vector<int> first, const std::vector<int>& second) {
    first.insert(first.end(), second.begin(), second.end());
    return first;
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

/// \brief the logits describe a generator that ran exactly `ids`
void RequireRan(const buffer<bf16>& logits, const std::vector<int>& ids) {
    float sum = 0;
    for (const int id : ids) sum += static_cast<float>(id);
    RequireState(logits, static_cast<float>(ids.size()), static_cast<float>(ids.back()),
                 static_cast<float>(bf16(sum)));
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

// ---- what one request runs, and what a later one runs again -----------------

void TestChunksAndDecodeStepsEachRunOnlyTheirOwnTokens() {
    auto engine = MakeEngine(64);
    std::vector<int> first{1, 2, 3, 4};
    std::vector<int> second{5, 6, 7};
    std::vector<int> third{8, 9};
    engine->prefill(first);
    engine->prefill(second);
    RequireState(engine->prefill(third), 9, 9, 45);
    engine->forward(10);
    RequireState(engine->forward(11), 11, 11, 66);
    TEST_REQUIRE(g_fake.generators_created == 1);
    TEST_REQUIRE(g_fake.appends == (std::vector<Tokens>{{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}, {11}}));
    TEST_REQUIRE(engine->replayed_tokens() == 0);
}

void TestOga017ReplaysTheHistoryAfterARestore() {
    g_fake = FakeOga{};
    auto engine = ConstructOn(FakeApiIn("C:\\hip-ep\\0.5.2"));
    TEST_REQUIRE(!engine->rewinds());
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    engine->checkpoint();
    engine->forward(4);
    engine->restore();
    std::vector<int> turn{5};
    RequireState(engine->prefill(turn), 4, 5, 11);
    TEST_REQUIRE(g_fake.snapshots == 0);
    TEST_REQUIRE(g_fake.rewinds.empty());
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2, 3, 5}));
    TEST_REQUIRE(engine->replayed_tokens() == 3);
}

void TestOga019RewindsToTheCheckpoint() {
    auto engine = MakeEngineOn({0, 19, 0, 0});
    TEST_REQUIRE(engine->rewinds());
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    TEST_REQUIRE(engine->checkpoint() == 3);
    TEST_REQUIRE(g_fake.snapshots == 1);
    engine->forward(4);
    engine->forward(5);
    engine->restore();
    std::vector<int> turn{6};
    RequireState(engine->prefill(turn), 4, 6, 12);
    TEST_REQUIRE(g_fake.rewinds == (std::vector<std::size_t>{3}));
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{6}));
    TEST_REQUIRE(g_fake.generators_created == 1);
    TEST_REQUIRE(engine->replayed_tokens() == 0);

    // the snapshot outlives the rewind, so the same checkpoint can be left again
    engine->forward(7);
    engine->restore();
    std::vector<int> again{8};
    RequireState(engine->prefill(again), 4, 8, 14);
    TEST_REQUIRE(g_fake.rewinds == (std::vector<std::size_t>{3, 3}));
    TEST_REQUIRE(g_fake.generators_created == 1);
    TEST_REQUIRE(engine->replayed_tokens() == 0);
}

void TestARewindNeedsTheHistoryToKeepTheCheckpoint() {
    auto engine = MakeEngineOn({0, 19, 0, 0});
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    engine->checkpoint();
    engine->forward(4);
    // a history that leaves the checkpoint's tokens replays
    engine->clear_context();
    std::vector<int> other{1, 9, 3, 4};
    RequireState(engine->prefill(other), 4, 4, 17);
    TEST_REQUIRE(g_fake.rewinds.empty());
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(engine->replayed_tokens() == 0);

    // so does one whose checkpoint could not be snapshotted
    g_fake.fail_next_snapshot = "out of memory";
    TEST_REQUIRE(engine->checkpoint() == 4);
    engine->forward(5);
    engine->restore();
    std::vector<int> turn{6};
    RequireState(engine->prefill(turn), 5, 6, 23);
    TEST_REQUIRE(g_fake.rewinds.empty());
    TEST_REQUIRE(g_fake.generators_created == 3);
    TEST_REQUIRE(engine->replayed_tokens() == 4);
}

void TestARefusedRewindReplaysInstead() {
    auto engine = MakeEngineOn({0, 19, 0, 0});
    std::vector<int> prompt{1, 2, 3};
    engine->prefill(prompt);
    engine->checkpoint();
    engine->forward(4);
    engine->restore();
    g_fake.fail_next_rewind = "state lost";
    std::vector<int> turn{5};
    RequireState(engine->prefill(turn), 4, 5, 11);
    TEST_REQUIRE(g_fake.rewinds == (std::vector<std::size_t>{3}));
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.generators_live == 1);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2, 3, 5}));
    TEST_REQUIRE(engine->replayed_tokens() == 3);
}

void TestTheRewindEntryPointsAreOptional() {
    auto symbols = FakeSymbols();
    symbols.erase("OgaGenerator_SnapshotState");
    symbols.erase("OgaGenerator_RewindTo");
    const auto api = FakeApi(symbols, {"test", {}, {0, 19, 0, 0}});
    TEST_REQUIRE(api->functions().generator_snapshot_state == nullptr);
    TEST_REQUIRE(api->functions().generator_rewind_to == nullptr);
    g_fake = FakeOga{};
    auto engine = ConstructOn(api);
    TEST_REQUIRE(!engine->rewinds());
    std::vector<int> prompt{1, 2};
    engine->prefill(prompt);
    engine->checkpoint();
    engine->forward(3);
    engine->restore();
    std::vector<int> turn{4};
    RequireState(engine->prefill(turn), 3, 4, 7);
    TEST_REQUIRE(g_fake.snapshots == 0);
    TEST_REQUIRE(g_fake.generators_created == 2);
}

void TestAnImagePromptRunsInOneSetInputsAfterAWarmUp() {
    auto engine = MakeVisionEngine();
    const auto image = Image(4, 4);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> prompt = WithImage({1}, image, {2});
    RequireRan(engine->prefill(prompt, &vision), prompt);
    TEST_REQUIRE(g_fake.set_inputs.size() == 2);
    // the warm-up: blank pixels, at the most patches a prompt may have
    const auto& warm = g_fake.set_inputs[0];
    TEST_REQUIRE(warm.ids.size() == 17);
    TEST_REQUIRE(warm.ids.front() == kVisionStart);
    TEST_REQUIRE(std::count(warm.ids.begin(), warm.ids.end(), kImageToken) == 16);
    TEST_REQUIRE(warm.pixel_shape == (std::vector<std::int64_t>{64, kPatchValues}));
    TEST_REQUIRE(std::all_of(warm.pixels.begin(), warm.pixels.end(), [](std::uint16_t v) { return v == 0; }));
    TEST_REQUIRE(warm.grids == (std::vector<std::int64_t>{1, 8, 8}));
    TEST_REQUIRE(warm.image_tokens == (std::vector<std::int64_t>{16}));
    const auto& run = g_fake.set_inputs[1];
    TEST_REQUIRE(run.ids == prompt);
    TEST_REQUIRE(run.pixel_shape == (std::vector<std::int64_t>{16, kPatchValues}));
    TEST_REQUIRE(run.pixels == image.pixels);
    TEST_REQUIRE(run.grids == (std::vector<std::int64_t>{1, 4, 4}));
    TEST_REQUIRE(run.image_tokens == (std::vector<std::int64_t>{4}));
    TEST_REQUIRE(g_fake.appends.empty());
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(g_fake.generators_live == 1);

    // what follows the image continues the same generator
    std::vector<int> more{3, 4};
    RequireRan(engine->prefill(more), Concat(prompt, more));
    RequireRan(engine->forward(5), Concat(prompt, {3, 4, 5}));
    TEST_REQUIRE(g_fake.appends == (std::vector<Tokens>{{3, 4}, {5}}));
    TEST_REQUIRE(g_fake.generators_created == 2);
    TEST_REQUIRE(engine->replayed_tokens() == 0);
}

void TestAnImageRunCanStartAChunk() {
    auto engine = MakeVisionEngine();
    const auto image = Image(2, 2);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> head{1, kVisionStart};
    std::vector<int> rest{kImageToken, 2};
    engine->prefill(head);
    RequireRan(engine->prefill(rest, &vision), Concat(head, rest));
    TEST_REQUIRE(g_fake.set_inputs.back().ids == Concat(head, rest));
    TEST_REQUIRE(engine->replayed_tokens() == 2);
}

void TestANewImageReplaysTheHistoryWithEveryImage() {
    auto engine = MakeVisionEngine();
    const auto first = Image(4, 4, 0x3c00);
    const auto second = Image(2, 4, 0x4000);
    flm::oga::OgaVisionInput one{{first}};
    flm::oga::OgaVisionInput two{{second}};
    std::vector<int> turn1 = WithImage({1}, first, {2});
    engine->prefill(turn1, &one);
    engine->forward(3);
    std::vector<int> turn2 = WithImage({4}, second, {5});
    const auto all = Concat(Concat(turn1, {3}), turn2);
    RequireRan(engine->prefill(turn2, &two), all);
    // the warm-up, the first turn and the replay: the vision model warms up once
    TEST_REQUIRE(g_fake.set_inputs.size() == 3);
    const auto& replay = g_fake.set_inputs.back();
    TEST_REQUIRE(replay.ids == all);
    TEST_REQUIRE(replay.grids == (std::vector<std::int64_t>{1, 4, 4, 1, 2, 4}));
    TEST_REQUIRE(replay.image_tokens == (std::vector<std::int64_t>{4, 2}));
    TEST_REQUIRE(replay.pixel_shape == (std::vector<std::int64_t>{24, kPatchValues}));
    auto pixels = first.pixels;
    pixels.insert(pixels.end(), second.pixels.begin(), second.pixels.end());
    TEST_REQUIRE(replay.pixels == pixels);
    TEST_REQUIRE(g_fake.generators_live == 1);
    TEST_REQUIRE(engine->replayed_tokens() == turn1.size() + 1);
}

void TestARestoredCheckpointBringsBackItsImages() {
    auto engine = MakeVisionEngine();
    const auto image = Image(4, 4);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> prompt = WithImage({1}, image, {2});
    engine->prefill(prompt, &vision);
    engine->checkpoint();
    engine->forward(3);
    engine->forward(4);
    engine->restore();
    std::vector<int> turn{5};
    const auto expected = Concat(prompt, turn);
    RequireRan(engine->prefill(turn), expected);
    TEST_REQUIRE(g_fake.set_inputs.size() == 3);
    TEST_REQUIRE(g_fake.set_inputs.back().ids == expected);
    TEST_REQUIRE(g_fake.set_inputs.back().pixels == image.pixels);
    TEST_REQUIRE(engine->replayed_tokens() == prompt.size());

    // a cleared context leaves no image to give a new generator
    engine->clear_context();
    std::vector<int> text{1, 2};
    RequireRan(engine->prefill(text), text);
    TEST_REQUIRE(g_fake.set_inputs.size() == 3);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2}));
}

void TestOga019DoesNotRewindPastANewImage() {
    auto engine = MakeVisionEngine({0, 19, 0, 0});
    std::vector<int> text{1, 2};
    engine->prefill(text);
    engine->checkpoint();
    TEST_REQUIRE(g_fake.snapshots == 1);
    engine->forward(3);
    engine->restore();
    const auto image = Image(2, 2);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> turn = WithImage({4}, image, {5});
    RequireRan(engine->prefill(turn, &vision), Concat(text, turn));
    TEST_REQUIRE(g_fake.rewinds.empty());
    TEST_REQUIRE(g_fake.set_inputs.back().ids == Concat(text, turn));
    TEST_REQUIRE(engine->replayed_tokens() == 2);
}

void TestAnotherImageOfTheSameSizeRebuildsTheGenerator() {
    auto engine = MakeVisionEngine();
    const auto first = Image(4, 4, 0x3c00);
    const auto second = Image(4, 4, 0x4000);
    flm::oga::OgaVisionInput one{{first}};
    flm::oga::OgaVisionInput two{{second}};
    std::vector<int> prompt = WithImage({1}, first, {2});
    engine->prefill(prompt, &one);
    engine->forward(3);
    // the same tokens, but not the image the generator ran
    engine->clear_context();
    std::vector<int> again = Concat(prompt, {3, 4});
    RequireRan(engine->prefill(again, &two), again);
    TEST_REQUIRE(g_fake.set_inputs.size() == 3);
    TEST_REQUIRE(g_fake.set_inputs.back().pixels == second.pixels);
    TEST_REQUIRE(g_fake.generators_live == 1);
    TEST_REQUIRE(engine->replayed_tokens() == 0);

    // equal pixels in another payload continue it
    engine->clear_context();
    flm::oga::OgaVisionInput copy{{second}};
    std::vector<int> more = Concat(again, {5});
    RequireRan(engine->prefill(more, &copy), more);
    TEST_REQUIRE(g_fake.set_inputs.size() == 3);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{5}));
}

void TestOga019DoesNotRewindToAnotherImageOfTheSameSize() {
    auto engine = MakeVisionEngine({0, 19, 0, 0});
    const auto first = Image(4, 4, 0x3c00);
    const auto second = Image(4, 4, 0x4000);
    flm::oga::OgaVisionInput one{{first}};
    flm::oga::OgaVisionInput two{{second}};
    std::vector<int> prompt = WithImage({1}, first, {2});
    engine->prefill(prompt, &one);
    engine->checkpoint();
    TEST_REQUIRE(g_fake.snapshots == 1);
    engine->forward(3);
    engine->clear_context();
    std::vector<int> turn = Concat(prompt, {4});
    RequireRan(engine->prefill(turn, &two), turn);
    TEST_REQUIRE(g_fake.rewinds.empty());
    TEST_REQUIRE(g_fake.set_inputs.back().ids == turn);
    TEST_REQUIRE(g_fake.set_inputs.back().pixels == second.pixels);
}

void TestImageRunsAndImagesMustAgree() {
    auto engine = MakeVisionEngine();
    const auto image = Image(4, 4);
    flm::oga::OgaVisionInput one{{image}};
    const std::vector<int> prompt = WithImage({1}, image, {2});
    const auto fails = [&](std::vector<int> ids, flm::oga::OgaVisionInput* vision) {
        return RequireThrows<std::invalid_argument>([&] { engine->prefill(ids, vision); });
    };
    RequireContains(fails(prompt, nullptr), "more runs of image tokens than its 0 image(s)");
    RequireContains(fails({1, 2}, &one), "has 0 runs of image tokens for 1 image(s)");
    RequireContains(fails({1, kImageToken, kImageToken, kImageToken, kImageToken, 2}, &one),
                    "image 1 does not follow a vision start token");
    RequireContains(fails(WithImage({1}, Image(2, 4), {2}), &one),
                    "image 1 has 4 image tokens, but its run in the prompt has 2");
    flm::oga::OgaVisionInput odd{{Image(3, 4)}};
    RequireContains(fails(prompt, &odd), "image 1 has an invalid grid");
    flm::oga::OgaVisionInput thin{{image}};
    thin.images[0].pixels.pop_back();
    RequireContains(fails(prompt, &thin), "image 1 has 63 pixel values, not 64");
    TEST_REQUIRE(g_fake.generators_created == 0);
    TEST_REQUIRE(engine->get_current_context_length() == 0);
}

void TestTheImagesOfAPromptShareThePatchBudget() {
    auto engine = MakeVisionEngine();
    const auto big = Image(6, 6);
    flm::oga::OgaVisionInput one{{big}};
    std::vector<int> turn1 = WithImage({1}, big, {2});
    engine->prefill(turn1, &one);
    // the earlier image counts too: a rebuilt generator runs both at once
    std::vector<int> turn2 = WithImage({3}, big, {4});
    RequireContains(RequireThrows<std::invalid_argument>([&] { engine->prefill(turn2, &one); }),
                    "prompt images have 72 patches, more than the 64");
    TEST_REQUIRE(engine->get_current_context_length() == static_cast<int>(turn1.size()));
    TEST_REQUIRE(g_fake.set_inputs.size() == 2);

    engine->clear_context();
    flm::oga::OgaVisionInput two{{Image(4, 4), Image(6, 6)}};
    std::vector<int> both = WithImage(WithImage({1}, two.images[0], {2}), two.images[1], {3});
    RequireRan(engine->prefill(both, &two), both);
    TEST_REQUIRE(g_fake.set_inputs.back().image_tokens == (std::vector<std::int64_t>{4, 9}));
}

void TestSetContextLengthKeepsWholeImagesOnly() {
    auto engine = MakeVisionEngine();
    const auto image = Image(4, 4);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> prompt = WithImage({1}, image, {2});
    engine->prefill(prompt, &vision);
    RequireContains(RequireThrows<std::invalid_argument>([&] { engine->set_context_length(4); }),
                    "cannot keep part of an image");
    TEST_REQUIRE(engine->get_current_context_length() == static_cast<int>(prompt.size()));
    engine->set_context_length(6);
    engine->set_context_length(2);
    std::vector<int> text{3};
    RequireRan(engine->prefill(text), {1, kVisionStart, 3});
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, kVisionStart, 3}));
}

void TestAPackageWithoutVisionTakesNoImages() {
    auto engine = MakeEngine();
    const auto image = Image(2, 2);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> prompt = WithImage({1}, image, {2});
    RequireContains(RequireThrows<std::invalid_argument>([&] { engine->prefill(prompt, &vision); }),
                    "does not run a vision payload");
    // without a vision model the image token is text like any other
    RequireRan(engine->prefill(prompt), prompt);
    TEST_REQUIRE(g_fake.set_inputs.empty());
}

void TestAnInvalidVisionConfigurationIsRefused() {
    g_fake = FakeOga{};
    auto config = VisionConfig();
    config.image_token_id = static_cast<int>(kVocab);
    RequireContains(RequireThrows<std::invalid_argument>([&] { ConstructOn(FakeApi(), 32, 64, config); }),
                    "vision configuration is invalid");
    config = VisionConfig();
    config.vision_start_token_id = config.image_token_id;
    RequireThrows<std::invalid_argument>([&] { ConstructOn(FakeApi(), 32, 64, config); });
    config = VisionConfig();
    config.max_patches = 3;
    RequireThrows<std::invalid_argument>([&] { ConstructOn(FakeApi(), 32, 64, config); });
}

void TestFloatToHalfRoundsToNearestEven() {
    using flm::oga::FloatToHalf;
    TEST_REQUIRE(FloatToHalf(1.0f) == 0x3c00);
    TEST_REQUIRE(FloatToHalf(-2.0f) == 0xc000);
    TEST_REQUIRE(FloatToHalf(-0.5f) == 0xb800);
    TEST_REQUIRE(FloatToHalf(0.0f) == 0x0000);
    TEST_REQUIRE(FloatToHalf(-0.0f) == 0x8000);
    TEST_REQUIRE(FloatToHalf(65504.0f) == 0x7bff);
    // halfway between the largest float16 and the next power of two
    TEST_REQUIRE(FloatToHalf(65520.0f) == 0x7c00);
    TEST_REQUIRE(FloatToHalf(std::numeric_limits<float>::infinity()) == 0x7c00);
    const auto nan = FloatToHalf(std::numeric_limits<float>::quiet_NaN());
    TEST_REQUIRE((nan & 0x7c00) == 0x7c00 && (nan & 0x03ff) != 0);
    // ties go to the even mantissa
    TEST_REQUIRE(FloatToHalf(1.0f + 0x1p-11f) == 0x3c00);
    TEST_REQUIRE(FloatToHalf(1.0f + 3 * 0x1p-11f) == 0x3c02);
    TEST_REQUIRE(FloatToHalf(0x1p-14f) == 0x0400);
    TEST_REQUIRE(FloatToHalf(0x1p-24f) == 0x0001);
    TEST_REQUIRE(FloatToHalf(0x1p-25f) == 0x0000);
    TEST_REQUIRE(FloatToHalf(3 * 0x1p-25f) == 0x0002);
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

// ---- the AMDGPU EP: registration and model creation -------------------------

void TestTheEpLibraryIsRegisteredUnderTheNameOgaLooksUp() {
    g_fake = FakeOga{};
    flm::oga::RegisterAmdGpuProvider(*FakeApiIn("C:\\hip-ep\\0.5.2"));
    TEST_REQUIRE(g_fake.registrations.size() == 1);
    TEST_REQUIRE(g_fake.registrations[0].first == "AMDGPUExecutionProvider");
    TEST_REQUIRE(g_fake.registrations[0].second == "C:\\hip-ep\\0.5.2\\hipgpu.dll");
}

void TestARefusedRegistrationNamesTheLibrary() {
    g_fake = FakeOga{};
    g_fake.fail_registration = "Error loading \"hipgpu.dll\" which depends on \"amdhip64_7.dll\"";
    const auto text = RequireThrows<std::runtime_error>(
        [] { flm::oga::RegisterAmdGpuProvider(*FakeApiIn("C:\\hip-ep\\0.5.2")); });
    RequireContains(text, "failed to register 'C:\\hip-ep\\0.5.2\\hipgpu.dll' as AMDGPUExecutionProvider");
    RequireContains(text, "depends on \"amdhip64_7.dll\"");
}

void TestANonAsciiLibraryPathIsRefusedBeforeOgaSeesIt() {
    // Under a UTF-8 code page OGA reads both of its conversions of the path alike.
    if (GetACP() == CP_UTF8) return;
    g_fake = FakeOga{};
    const auto text = RequireThrows<std::runtime_error>(
        [] { flm::oga::RegisterAmdGpuProvider(*FakeApiIn(std::filesystem::path(u8"C:\\用户\\hip-ep"))); });
    TEST_REQUIRE(g_fake.registrations.empty());
    RequireContains(text, "needs an ASCII path");
}

void TestOpenModelPutsTheDecoderOnTheRegisteredEpAlone() {
    g_fake = FakeOga{};
    const auto api = FakeApi();
    TEST_REQUIRE(flm::oga::OpenModel(*api, "C:\\model") != nullptr);
    TEST_REQUIRE(g_fake.providers == (std::vector<std::string>{"AMDGPUExecutionProvider"}));
    TEST_REQUIRE(g_fake.overlays.size() == 1);
    RequireContains(g_fake.overlays[0], R"("decoder":{"session_options":{"session.disable_cpu_ep_fallback":"1"}})");
}

void TestOpenModelRejectsAModelOnAnotherDevice() {
    g_fake = FakeOga{};
    g_fake.device = "CPU";
    const auto api = FakeApi();
    RequireContains(RequireThrows<std::runtime_error>([&] { flm::oga::OpenModel(*api, "C:\\model"); }),
                    "on 'CPU', not on AMDGPU");
    TEST_REQUIRE(CountOf(Events(), "destroy_model") == 1);
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

std::string ReadText(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

const char* kPackageFiles[] = {"onnxruntime-genai.dll", "onnxruntime.dll", "hipgpu.dll"};

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
    RequireContains(Reason({"AMDRT_PKG_ROOT", partial.path}), "has no onnxruntime.dll, hipgpu.dll");

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

/// \brief the package DLLs in `bin`, with a version resource standing in for OGA
void MakePackageFiles(const std::filesystem::path& bin) {
    for (const char* file : kPackageFiles) Touch(bin / file);
    std::filesystem::copy_file("C:\\Windows\\System32\\version.dll", bin / "onnxruntime-genai.dll",
                               std::filesystem::copy_options::overwrite_existing);
}

/// \brief a complete hip-ep package under `root`, with its DLLs in root\\bin
void MakePackage(const std::filesystem::path& root) { MakePackageFiles(root / "bin"); }

/// \brief what amdgpu.zip extracts to: no bin, the DLLs and hipBLASLt's kernels at the root
void MakeManagedRuntimeLayout(const std::filesystem::path& root) {
    MakePackageFiles(root);
    Touch(root / "amdhip64_7.dll");
    Touch(root / "hipblaslt.dll");
    std::filesystem::create_directories(root / "hipblaslt" / "library");
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

std::filesystem::path RealEnvironmentPath(const wchar_t* name) {
    const DWORD needed = GetEnvironmentVariableW(name, nullptr, 0);
    if (needed == 0) return {};
    std::wstring value(needed, L'\0');
    value.resize(GetEnvironmentVariableW(name, value.data(), needed));
    return value;
}

std::string Text(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

struct RuntimeInstallFixture {
    TempDir local;
    TempDir archive_root;
    std::filesystem::path archive_path = archive_root.path / "hipep-test.zip";
    flm::oga::HipEpRuntimeRelease release;
    int downloads = 0;

    RuntimeInstallFixture()
        : release({"9.9.9", "hipep-test.zip", "https://example.invalid/hipep-test.zip", 0, 128,
                   ""}) {
        Touch(archive_path, "tiny test archive");
        release.archive_size = std::filesystem::file_size(archive_path);
        release.sha256 = download_utils::calculate_file_sha256(archive_path.string());
    }

    std::filesystem::path final_dir() const {
        return local.path / "hip-ep" / ("flm-" + release.version);
    }

    std::filesystem::path staging_root() const { return local.path / "hip-ep-install" / "staging"; }

    std::filesystem::path quarantine_root() const { return local.path / "hip-ep-install" / "quarantine"; }

    flm::oga::PackageEnvironment environment(std::filesystem::path root = {}) const {
        return Environment(std::move(root), local.path, {});
    }

    flm::oga::HipEpInstallHooks Hooks(
        std::function<void(const std::filesystem::path&, const std::filesystem::path&)> extract = {}) {
        flm::oga::HipEpInstallHooks hooks;
        hooks.release = release;
        hooks.download = [this](const download_utils::DownloadRequest& request, std::function<void(double)>) {
            ++downloads;
            std::filesystem::create_directories(request.destination.parent_path());
            std::filesystem::copy_file(archive_path, request.destination,
                                       std::filesystem::copy_options::overwrite_existing);
            return true;
        };
        hooks.extract_archive = extract
                                    ? std::move(extract)
                                    : [](const std::filesystem::path&, const std::filesystem::path& staging) {
                                          MakeManagedRuntimeLayout(staging);
                                      };
        hooks.space = [](const std::filesystem::path&) {
            return std::filesystem::space_info{
                static_cast<std::uintmax_t>(1ULL << 30),
                static_cast<std::uintmax_t>(1ULL << 30),
                static_cast<std::uintmax_t>(1ULL << 30)};
        };
        hooks.lock_wait = std::chrono::seconds(5);
        hooks.lock_poll = std::chrono::milliseconds(10);
        return hooks;
    }
};

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

void TestAutoInstallCreatesAManagedRuntime() {
    RuntimeInstallFixture fixture;
    const auto package = flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks());
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(package.source == "%LOCALAPPDATA%\\hip-ep\\flm-9.9.9");
    TEST_REQUIRE(package.bin == fixture.final_dir());
    TEST_REQUIRE(std::filesystem::is_regular_file(fixture.final_dir() / ".flm-hipep-install.json"));
}

void TestAutoInstallReusesTheManagedRuntimeWithoutDownloadingAgain() {
    RuntimeInstallFixture fixture;
    TEST_REQUIRE(flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks()).bin ==
                 fixture.final_dir());
    auto hooks = fixture.Hooks();
    hooks.download = [&](const download_utils::DownloadRequest&, std::function<void(double)>) {
        ++fixture.downloads;
        return false;
    };
    const auto package = flm::oga::EnsureHipEpPackageForTest(fixture.environment(), hooks);
    TEST_REQUIRE(package.bin == fixture.final_dir());
    TEST_REQUIRE(fixture.downloads == 1);
}

void TestAnExplicitRootStillDisablesAutoInstall() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks();
    hooks.download = [&](const download_utils::DownloadRequest&, std::function<void(double)>) {
        ++fixture.downloads;
        return true;
    };
    const auto text = RequireThrows<std::runtime_error>([&] {
        flm::oga::EnsureHipEpPackageForTest(fixture.environment(fixture.local.path / "absent"), hooks);
    });
    TEST_REQUIRE(fixture.downloads == 0);
    RequireContains(text, "AMDRT_PKG_ROOT: '");
}

void TestAutoInstallDoesNotReplaceAnUnownedConflict() {
    RuntimeInstallFixture fixture;
    Touch(fixture.final_dir() / "onnxruntime-genai.dll");
    const auto text = RequireThrows<std::runtime_error>([&] {
        flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks());
    });
    TEST_REQUIRE(fixture.downloads == 0);
    RequireContains(text, "refusing to replace");
}

void TestAutoInstallRepairsABrokenManagedRuntime() {
    RuntimeInstallFixture fixture;
    TEST_REQUIRE(flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks()).bin ==
                 fixture.final_dir());
    std::filesystem::remove(fixture.final_dir() / "onnxruntime.dll");
    const auto repaired = flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks());
    TEST_REQUIRE(repaired.bin == fixture.final_dir());
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(std::filesystem::is_directory(fixture.quarantine_root()));
    bool quarantined = false;
    for (const auto& entry : std::filesystem::directory_iterator(fixture.quarantine_root())) {
        if (entry.is_directory()) quarantined = true;
    }
    TEST_REQUIRE(quarantined);
}

void TestAutoInstallFailsWhenTheDownloaderNeedsAnAliasThatDoesNotExist() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks();
    hooks.download_path = [](const std::filesystem::path&) -> std::optional<std::filesystem::path> {
        return std::nullopt;
    };
    const auto local = fixture.local.path / std::filesystem::path(u8"用户");
    std::filesystem::create_directories(local);
    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(Environment({}, local, {}), hooks); });
    TEST_REQUIRE(fixture.downloads == 0);
    RequireContains(text, "cannot be represented");
}

void TestAutoInstallStopsWhenDownloadReturnsFalse() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks();
    hooks.download = [&](const download_utils::DownloadRequest&, std::function<void(double)>) {
        ++fixture.downloads;
        return false;
    };

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), hooks); });
    RequireContains(text, "failed to download hip-ep");
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
}

void TestAutoInstallRejectsAnArchiveWithTheWrongHash() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks();
    hooks.release->sha256 = std::string(64, '0');

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), hooks); });
    RequireContains(text, "did not match the pinned SHA-256 after download");
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
}

void TestAutoInstallCleansFailedExtractionAndCanRetry() {
    RuntimeInstallFixture fixture;
    auto failing_hooks = fixture.Hooks([](const std::filesystem::path&, const std::filesystem::path& staging) {
        Touch(staging / "partial.txt", "incomplete extraction");
        throw std::runtime_error("extractor failed");
    });

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), failing_hooks); });
    RequireContains(text, "extractor failed");
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
    TEST_REQUIRE(std::filesystem::is_empty(fixture.staging_root()));

    const auto package = flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks());
    TEST_REQUIRE(package.bin == fixture.final_dir());
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(std::filesystem::is_empty(fixture.staging_root()));
}

void TestAutoInstallRejectsAnInvalidExtractedLayoutWithoutPromotingIt() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks([](const std::filesystem::path&, const std::filesystem::path& staging) {
        std::filesystem::create_directories(staging / "bin");
    });

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), hooks); });
    RequireContains(text, "extracted hip-ep archive is missing");
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
    TEST_REQUIRE(std::filesystem::is_empty(fixture.staging_root()));
}

void TestAutoInstallCleansStagingWhenTarRejectsAnInvalidArchive() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks();
    hooks.extract_archive = flm::oga::testing::ExtractHipEpArchiveWithTar;

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), hooks); });
    RequireContains(text, "tar.exe failed to extract");
    TEST_REQUIRE(fixture.downloads == 1);
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
    TEST_REQUIRE(std::filesystem::is_empty(fixture.staging_root()));
}

void TestAutoInstallChecksFreeSpaceBeforeDownloading() {
    RuntimeInstallFixture fixture;
    auto hooks = fixture.Hooks();
    hooks.space = [](const std::filesystem::path&) {
        return std::filesystem::space_info{0, 0, 0};
    };

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), hooks); });
    RequireContains(text, "not enough free space under %LOCALAPPDATA%");
    TEST_REQUIRE(fixture.downloads == 0);
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
}

void TestValidExternalPackagesBypassAutoInstall() {
    RuntimeInstallFixture fixture;
    TempDir explicit_root;
    TempDir programs;
    MakePackage(explicit_root.path);
    MakePackage(programs.path / "hip-ep" / "0.5.2");
    const auto hooks = fixture.Hooks();

    const auto explicit_package = flm::oga::EnsureHipEpPackageForTest(
        fixture.environment(explicit_root.path), hooks);
    TEST_REQUIRE(explicit_package.source == "AMDRT_PKG_ROOT");
    TEST_REQUIRE(explicit_package.bin == explicit_root.path / "bin");

    const auto discovered_package = flm::oga::EnsureHipEpPackageForTest(
        Environment({}, fixture.local.path, programs.path), hooks);
    TEST_REQUIRE(discovered_package.source == "%ProgramFiles%\\hip-ep\\0.5.2");
    TEST_REQUIRE(discovered_package.bin == programs.path / "hip-ep" / "0.5.2" / "bin");
    TEST_REQUIRE(fixture.downloads == 0);
    TEST_REQUIRE(!std::filesystem::exists(fixture.final_dir()));
}

void TestAutoInstallRepairsAnInstallFromAnotherArchive() {
    RuntimeInstallFixture fixture;
    TEST_REQUIRE(flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks()).bin ==
                 fixture.final_dir());
    const auto marker_path = fixture.final_dir() / ".flm-hipep-install.json";
    const std::string other_archive_marker = R"({"owner":"FastFlowLM","schema":"flm-hipep-install/v1","runtime":"hip-ep","version":")" +
                                             fixture.release.version +
                                             R"(","archive_sha256":"another-archive"})";
    Touch(marker_path, other_archive_marker);
    std::filesystem::remove(fixture.final_dir() / "onnxruntime.dll");

    const auto repaired = flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks());
    TEST_REQUIRE(repaired.bin == fixture.final_dir());
    TEST_REQUIRE(fixture.downloads == 1);
    RequireContains(ReadText(marker_path), fixture.release.sha256);
    std::vector<std::string> quarantined_markers;
    for (const auto& entry : std::filesystem::directory_iterator(fixture.quarantine_root())) {
        quarantined_markers.push_back(ReadText(entry.path() / ".flm-hipep-install.json"));
    }
    TEST_REQUIRE(quarantined_markers == std::vector<std::string>{other_archive_marker});
}

void TestAutoInstallDoesNotReplaceAnInstallForAnotherVersion() {
    RuntimeInstallFixture fixture;
    TEST_REQUIRE(flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks()).bin ==
                 fixture.final_dir());
    const auto marker_path = fixture.final_dir() / ".flm-hipep-install.json";
    const std::string other_version_marker =
        R"({"owner":"FastFlowLM","schema":"flm-hipep-install/v1","runtime":"hip-ep","version":"0.0.1","archive_sha256":")" +
        fixture.release.sha256 + R"("})";
    Touch(marker_path, other_version_marker);
    std::filesystem::remove(fixture.final_dir() / "onnxruntime.dll");

    const auto text = RequireThrows<std::runtime_error>(
        [&] { flm::oga::EnsureHipEpPackageForTest(fixture.environment(), fixture.Hooks()); });
    RequireContains(text, "refusing to replace");
    TEST_REQUIRE(ReadText(marker_path) == other_version_marker);
    TEST_REQUIRE(fixture.downloads == 1);
}

// ---- backend checks that run before hip-ep is loaded ------------------------

/// \brief a model directory with just the files the backend reads before loading hip-ep
struct PackageDir {
    TempDir dir;
    LM_Config config;
    /// \param model_extra more members of genai_config.json's model, each after a comma
    /// \param type genai_config.json's model type
    explicit PackageDir(std::uint32_t vocab = kVocab, const char* eos_text = "<|im_end|>",
                        const std::string& model_extra = "", const std::string& type = "qwen3_5") {
        Touch(dir.path / "genai_config.json",
              R"({"model":{"type":")" + type + R"(","vocab_size":)" + std::to_string(kVocab) +
                  R"(,"context_length":128,"eos_token_id":[7])" + model_extra + "}}");
        Touch(dir.path / "tokenizer_config.json",
              std::string(R"({"bos_token":null,"eos_token":")") + eos_text + R"("})");
        Touch(dir.path / "tokenizer.json", R"({"added_tokens":[{"id":5,"content":"<|im_end|>"}]})");
        // a patch row of in_channels * temporal_patch_size * patch_size^2 = kPatchValues
        config._json_config = {
            {"vocab_size", vocab},
            {"vision_config", {{"patch_size", 1}, {"temporal_patch_size", 2}, {"in_channels", 2}}}};
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

    PackageDir unnamed_tokens(kVocab, "<|im_end|>", R"(,"vision":{"spatial_merge_size":2})");
    RequireContains(RequireThrows<std::runtime_error>([&] { factory(unnamed_tokens.Context()); }),
                    "has a vision model but does not state image_token_id");
}

constexpr const char* kVisionModel =
    R"(,"image_token_id":40,"vision_start_token_id":41,"vision":{"spatial_merge_size":2})";

void TestTheBackendRunsThePackagesImages() {
    g_fake = FakeOga{};
    auto api = flm::oga::OgaApi::GetOrCreateWith([] { return FakeApi(); });
    PackageDir package(kVocab, "<|im_end|>", kVisionModel);
    auto backend = flm::oga::hipep_factory()(package.Context(64));
    // OGA 0.17 cannot rewind, so a restored checkpoint replays
    TEST_REQUIRE(!backend->keeps_state_on_restore());
    TEST_REQUIRE(backend->matches_token_prefixes());
    const auto image = Image(2, 2);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> prompt = WithImage({1}, image, {2});
    backend->engine().prefill(prompt, &vision);
    TEST_REQUIRE(g_fake.set_inputs.size() == 2);
    // the package's warm-up is the full budget: 90x90 patches
    TEST_REQUIRE(g_fake.set_inputs[0].grids == (std::vector<std::int64_t>{1, 90, 90}));
    TEST_REQUIRE(g_fake.set_inputs[0].pixel_shape == (std::vector<std::int64_t>{8100, 4}));
    TEST_REQUIRE(g_fake.set_inputs[1].ids == prompt);
    TEST_REQUIRE(g_fake.set_inputs[1].pixel_shape == (std::vector<std::int64_t>{4, 4}));
    TEST_REQUIRE(backend->replayed_tokens() == 0);
}

void TestTheBackendRunsTheTextOfAnotherTypesVisionPackage() {
    g_fake = FakeOga{};
    auto api = flm::oga::OgaApi::GetOrCreateWith([] { return FakeApi(); });
    // a Gemma 4 package names its image token, and neither a vision start nor a merge size
    PackageDir package(kVocab, "<|im_end|>",
                       R"(,"image_token_id":40,"vision":{"filename":"vision_encoder/model.onnx"})", "gemma4");
    auto backend = flm::oga::hipep_factory()(package.Context(64));
    const auto image = Image(2, 2);
    flm::oga::OgaVisionInput vision{{image}};
    std::vector<int> prompt = WithImage({1}, image, {2});
    RequireContains(RequireThrows<std::invalid_argument>([&] { backend->engine().prefill(prompt, &vision); }),
                    "does not run a vision payload");
    std::vector<int> text{1, 2};
    backend->engine().prefill(text);
    TEST_REQUIRE(g_fake.set_inputs.empty());
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{1, 2}));
}

void TestAGemma3PackageGivesItsTextAnEmptyImageBatch() {
    g_fake = FakeOga{};
    auto api = flm::oga::OgaApi::GetOrCreateWith([] { return FakeApi(); });
    PackageDir package(kVocab, "<|im_end|>",
                       R"(,"image_token_id":40,"vision":{"filename":"vision.onnx"})", "gemma3");
    auto backend = flm::oga::hipep_factory()(package.Context(64));
    std::vector<int> text{1, 2};
    backend->engine().prefill(text);
    TEST_REQUIRE(g_fake.set_inputs.size() == 1);
    TEST_REQUIRE(g_fake.set_inputs[0].ids == text);
    TEST_REQUIRE(g_fake.set_inputs[0].pixel_shape == (std::vector<std::int64_t>{1, 0, 0}));
    TEST_REQUIRE(g_fake.set_inputs[0].pixels.empty());
    // the generator runs on with tokens alone
    backend->engine().forward(3);
    TEST_REQUIRE(g_fake.set_inputs.size() == 1);
    TEST_REQUIRE(g_fake.appends.back() == (Tokens{3}));
    // a prompt that leaves the history gets a new generator, and the batch again
    backend->engine().clear_context();
    std::vector<int> other{1, 5};
    backend->engine().prefill(other);
    TEST_REQUIRE(g_fake.set_inputs.size() == 2);
    TEST_REQUIRE(g_fake.set_inputs[1].ids == other);
}

void TestABackendOnOga019KeepsItsStateOnRestore() {
    g_fake = FakeOga{};
    auto api = flm::oga::OgaApi::GetOrCreateWith(
        [] { return FakeApi(FakeSymbols(), {"test", {}, {0, 19, 0, 0}}); });
    PackageDir package;
    auto backend = flm::oga::hipep_factory()(package.Context(64));
    TEST_REQUIRE(backend->keeps_state_on_restore());
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

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--prepare-runtime") {
        try {
            const auto package = flm::oga::EnsureHipEpPackage(RealEnvironmentPath);
            std::cout << Text(package.bin) << "\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << "\n";
            return 1;
        }
    }
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
    RUN(TestChunksAndDecodeStepsEachRunOnlyTheirOwnTokens);
    RUN(TestOga017ReplaysTheHistoryAfterARestore);
    RUN(TestOga019RewindsToTheCheckpoint);
    RUN(TestARewindNeedsTheHistoryToKeepTheCheckpoint);
    RUN(TestARefusedRewindReplaysInstead);
    RUN(TestTheRewindEntryPointsAreOptional);
    RUN(TestAnImagePromptRunsInOneSetInputsAfterAWarmUp);
    RUN(TestAnImageRunCanStartAChunk);
    RUN(TestANewImageReplaysTheHistoryWithEveryImage);
    RUN(TestARestoredCheckpointBringsBackItsImages);
    RUN(TestOga019DoesNotRewindPastANewImage);
    RUN(TestAnotherImageOfTheSameSizeRebuildsTheGenerator);
    RUN(TestOga019DoesNotRewindToAnotherImageOfTheSameSize);
    RUN(TestImageRunsAndImagesMustAgree);
    RUN(TestTheImagesOfAPromptShareThePatchBudget);
    RUN(TestSetContextLengthKeepsWholeImagesOnly);
    RUN(TestAPackageWithoutVisionTakesNoImages);
    RUN(TestAnInvalidVisionConfigurationIsRefused);
    RUN(TestFloatToHalfRoundsToNearestEven);
    RUN(TestAFreshApiAfterFullReleaseWorksAgain);
    RUN(TestTheEpLibraryIsRegisteredUnderTheNameOgaLooksUp);
    RUN(TestARefusedRegistrationNamesTheLibrary);
    RUN(TestANonAsciiLibraryPathIsRefusedBeforeOgaSeesIt);
    RUN(TestOpenModelPutsTheDecoderOnTheRegisteredEpAlone);
    RUN(TestOpenModelRejectsAModelOnAnotherDevice);
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
    RUN(TestAutoInstallCreatesAManagedRuntime);
    RUN(TestAutoInstallReusesTheManagedRuntimeWithoutDownloadingAgain);
    RUN(TestAnExplicitRootStillDisablesAutoInstall);
    RUN(TestAutoInstallDoesNotReplaceAnUnownedConflict);
    RUN(TestAutoInstallRepairsABrokenManagedRuntime);
    RUN(TestAutoInstallFailsWhenTheDownloaderNeedsAnAliasThatDoesNotExist);
    RUN(TestAutoInstallStopsWhenDownloadReturnsFalse);
    RUN(TestAutoInstallRejectsAnArchiveWithTheWrongHash);
    RUN(TestAutoInstallCleansFailedExtractionAndCanRetry);
    RUN(TestAutoInstallRejectsAnInvalidExtractedLayoutWithoutPromotingIt);
    RUN(TestAutoInstallCleansStagingWhenTarRejectsAnInvalidArchive);
    RUN(TestAutoInstallChecksFreeSpaceBeforeDownloading);
    RUN(TestValidExternalPackagesBypassAutoInstall);
    RUN(TestAutoInstallRepairsAnInstallFromAnotherArchive);
    RUN(TestAutoInstallDoesNotReplaceAnInstallForAnotherVersion);
    RUN(TestTheBackendRejectsWhatThePackageCannotRun);
    RUN(TestTheBackendRunsThePackagesImages);
    RUN(TestTheBackendRunsTheTextOfAnotherTypesVisionPackage);
    RUN(TestAGemma3PackageGivesItsTextAnEmptyImageBatch);
    RUN(TestABackendOnOga019KeepsItsStateOnRestore);
    RUN(TestTheBackendSaysHowToGetHipep);
#undef RUN
    std::cout << "All oga-hipep tests passed\n";
    return 0;
}
