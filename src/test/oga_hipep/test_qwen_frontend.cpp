// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file test_qwen_frontend.cpp
/// \brief fake-backend regression coverage for the Qwen3.5VL OGA frontend

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_qwen3_5vl.hpp>
#include <oga/oga_vision.hpp>
#include <tensor_utils/safe_tensors.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

constexpr int kEos = 8;
constexpr int kImEnd = 248046;
constexpr int kVisionStart = 248053;
constexpr int kVisionEnd = 248054;
constexpr int kImagePad = 248056;
constexpr int kThinkStart = 248068;
constexpr int kThinkEnd = 248069;
std::vector<int> g_encoded_tokens;
/// \brief encodings of particular text; any other text encodes to g_encoded_tokens
std::map<std::string, std::vector<int>> g_encodings;
/// \brief the text of each token; when empty, decode() gives "decoded"
std::map<int, std::string> g_words;
std::vector<int> g_samples;
std::size_t g_sample_index{};
/// \brief the grid preprocess_image gives each image
int g_grid_h = 4;
int g_grid_w = 4;
/// \brief the max_pixels of each preprocess_image call
std::vector<unsigned> g_max_pixels;

class FakeEngine final : public causal_lm {
public:
    explicit FakeEngine(std::uint32_t length) : max_length(length) {}

    buffer<bf16> forward(int token) override {
        if (position + 1 > static_cast<int>(max_length)) {
            throw std::runtime_error("fake OGA engine decode limit exceeded");
        }
        ++position;
        forwarded.push_back(token);
        Run({token});
        return buffer<bf16>(1);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void* payload) override {
        if (position + static_cast<int>(tokens.size()) > static_cast<int>(max_length)) {
            throw std::runtime_error("fake OGA engine prefill limit exceeded");
        }
        position += static_cast<int>(tokens.size());
        prefills.push_back(tokens);
        if (payload) vision = *static_cast<const flm::oga::OgaVisionInput*>(payload);
        Run(tokens);
        return buffer<bf16>(1);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override {
        position = 0;
        executed.clear();
        forwarded.clear();
        saved.reset();
    }
    buffer<bf16> get_k_cache(int, int) override { return buffer<bf16>(1); }
    buffer<bf16> get_v_cache(int, int) override { return buffer<bf16>(1); }
    int get_current_context_length() override { return position; }
    int checkpoint() override {
        saved = position;
        return position;
    }
    int restore() override {
        if (!saved) return -1;
        ++restores;
        position = *saved;
        executed.resize(static_cast<std::size_t>(*saved));
        return position;
    }

    std::uint32_t max_length;
    int position{};
    /// \brief the logical sequence: every token since the last clear, less what restore() dropped
    std::vector<int> executed;
    std::vector<int> forwarded;
    std::vector<std::vector<int>> prefills;
    /// \brief the images of the last prefill that had any
    flm::oga::OgaVisionInput vision;
    std::optional<int> saved;
    int restores{};
    /// \brief whether, like OGA 0.19, it can go back to the checkpoint without running it again
    bool rewinds = false;
    /// \brief what the engine's generator has run, as OgaEngine keeps it
    std::vector<int> ran;
    /// \brief tokens run a second time, counted as OgaEngine::replayed_tokens counts them
    std::uint64_t replayed{};

private:
    void Run(const std::vector<int>& fresh) {
        const std::size_t before = executed.size();
        executed.insert(executed.end(), fresh.begin(), fresh.end());
        std::size_t reused = 0;
        if (ran.size() <= executed.size() && std::equal(ran.begin(), ran.end(), executed.begin())) {
            reused = ran.size();
        } else if (rewinds && saved && static_cast<std::size_t>(*saved) <= ran.size() &&
                   static_cast<std::size_t>(*saved) < executed.size() &&
                   std::equal(ran.begin(), ran.begin() + *saved, executed.begin())) {
            reused = static_cast<std::size_t>(*saved);
        }
        if (reused < before) replayed += before - reused;
        ran = executed;
    }
};

class FakeOgaBackend final : public flm::backend::ModelBackend {
public:
    explicit FakeOgaBackend(std::uint32_t max_length)
        : engine_(std::make_unique<FakeEngine>(max_length)) {}

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kOgaHipepBackendId; }
    std::uint32_t max_decode_length() const override { return engine_->max_length - 1; }
    bool forwards_past_eos() const override { return false; }
    bool keeps_state_on_restore() const override { return engine_->rewinds; }
    std::uint64_t replayed_tokens() const override { return engine_->replayed; }
    bool matches_token_prefixes() const override { return true; }
    FakeEngine& fake_engine() const { return *engine_; }

private:
    std::unique_ptr<FakeEngine> engine_;
};

class Harness final : public Qwen3_5VL {
public:
    explicit Harness(std::uint32_t backend_max_length = 64,
                     std::uint32_t frontend_max_length = 128)
        : Qwen3_5VL(nullptr) {
        auto backend = std::make_unique<FakeOgaBackend>(backend_max_length);
        engine_ = &backend->fake_engine();
        backend_ = std::move(backend);
        lm_engine = &backend_->engine();
        tokenizer = std::make_unique<Tokenizer>("");
        sampler_config config;
        sampler = std::make_unique<Sampler>(64, config);
        MAX_L = frontend_max_length;
        eos_token_ids = {kEos, kImEnd};
        bos_token_id = -1;
        has_bos_token = false;
        log_raw_output = false;
    }

    std::string apply_chat_template(nlohmann::ordered_json&,
                                    nlohmann::ordered_json,
                                    bool) override {
        return "fake rendered prompt";
    }

    const std::vector<int>& history() const { return token_history; }
    FakeEngine& engine() const { return *engine_; }
    void set_total_tokens(std::uint32_t value) { total_tokens = value; }
    void set_vision_geometry(const qwen3_5vl_vision_geometry_t& geometry) { vision_geometry = geometry; }

private:
    FakeEngine* engine_{};
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Reset(const std::vector<int>& encoded, const std::vector<int>& samples) {
    g_encoded_tokens = encoded;
    g_encodings.clear();
    g_words.clear();
    g_samples = samples;
    g_sample_index = 0;
    g_grid_h = 4;
    g_grid_w = 4;
    g_max_pixels.clear();
}

lm_uniform_input_t TextInput() {
    lm_uniform_input_t input;
    input.prompt = "hello";
    return input;
}

chat_meta_info_t Meta() {
    chat_meta_info_t meta;
    meta.max_prefill_len = 64;
    return meta;
}

void RequireHistoryMatchesEngine(const Harness& model) {
    Require(model.history() == model.engine().executed,
            "token history must equal the tokens committed to the fake engine");
}

void CapacityValidationIncludesTheRenderedPrompt(bool think) {
    Reset({10, 11, 12, 13, 14, 15, 16}, {7, 7});
    Harness model(8, 64);  // hard decode cap is 7; rendered prompt is 7 tokens
    Require(model.configure_parameter("enable_think", think), "enable_think setup failed");
    auto input = TextInput();
    input.requested_max_new_tokens = 1;
    auto meta = Meta();
    try {
        model.insert(meta, input);
    } catch (const ModelRequestError& error) {
        Require(error.http_code() == 400, "capacity rejection must be an HTTP 400 model request error");
        Require(model.engine().executed.empty(), "capacity rejection must happen before any engine call");
        return;
    }
    throw std::runtime_error("full rendered prompt plus requested output should exceed OGA capacity");
}

void DecodeStopsAtBackendLimitWithAlignedHistory(bool think) {
    Reset({10, 11, 12, 13, 14, 15, 16}, {7, 7, 7, 7, 7});
    Harness model;  // shrink the engine below MAX_L; its backend cap becomes 8
    model.set_max_length(9);
    Require(model.configure_parameter("enable_think", think), "enable_think setup failed");
    auto input = TextInput();
    auto meta = Meta();
    Require(model.insert(meta, input), "text prompt insert failed");
    std::ostringstream output;
    model.generate(meta, 0, output);

    Require(model.engine().position == 8, "generation must stop at the lowered backend context limit");
    Require(model.engine().forwarded.size() == (think ? 3 : 5),
            "only the preamble and one in-budget sampled token should be forwarded");
    RequireHistoryMatchesEngine(model);
}

void FirstSampleEosIsNeverForwarded() {
    Reset({10, 11, 12, 13, 14, 15}, {7, kEos});
    Harness model;
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "text prompt insert failed");
    std::ostringstream output;
    model.generate(meta, 0, output);

    Require(std::find(model.engine().forwarded.begin(), model.engine().forwarded.end(), kEos) ==
                model.engine().forwarded.end(),
            "the first sampled EOS must not be forwarded to OGA");
    RequireHistoryMatchesEngine(model);
}

void LoopEosIsNeverForwarded() {
    Reset({10, 11, 12, 13, 14, 15}, {7, 7, 7, kEos});
    Harness model;
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "text prompt insert failed");
    std::ostringstream output;
    model.generate(meta, 0, output);

    Require(std::find(model.engine().forwarded.begin(), model.engine().forwarded.end(), kEos) ==
                model.engine().forwarded.end(),
            "an EOS sampled inside the decode loop must not be forwarded to OGA");
    RequireHistoryMatchesEngine(model);
}

void OneTokenGenerationLimitKeepsOnlyExecutedHistory() {
    Reset({10, 11, 12, 13, 14, 15}, {7, 7, 7});
    Harness model;
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "text prompt insert failed");
    std::ostringstream output;
    model.generate(meta, 1, output);

    RequireHistoryMatchesEngine(model);
}

struct GenerationResult {
    std::string text;
    std::vector<int> executed;
    std::vector<int> forwarded;
};

GenerationResult GenerateExplicitly() {
    Reset({10, 11, 12, 13, 14, 15}, {7, 9, 10});
    Harness model;
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "text prompt insert failed");
    std::ostringstream output;
    model.generate(meta, 1, output);
    return {output.str(), model.engine().executed, model.engine().forwarded};
}

GenerationResult GenerateWithPrompt() {
    Reset({10, 11, 12, 13, 14, 15}, {7, 9, 10});
    Harness model;
    auto meta = Meta();
    auto input = TextInput();
    std::ostringstream output;
    const std::string text = model.generate_with_prompt(meta, input, 1, output);
    return {text, model.engine().executed, model.engine().forwarded};
}

void GenerateWithPromptMatchesInsertThenGenerate() {
    const auto explicit_result = GenerateExplicitly();
    const auto combined_result = GenerateWithPrompt();
    Require(combined_result.text == explicit_result.text,
            "generate_with_prompt must preserve the same sampled output as insert followed by generate");
    Require(combined_result.forwarded == explicit_result.forwarded,
            "generate_with_prompt must run the same OGA preamble forwards as generate");
    Require(combined_result.executed == explicit_result.executed,
            "generate_with_prompt must leave the same engine context as insert followed by generate");
}

// ---- what one request runs ---------------------------------------------------

void ChunkedPrefillAndDecodeRunEachTokenOnce() {
    std::vector<int> prompt(1200);
    for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1000 + static_cast<int>(i);
    std::vector<int> encoded = prompt;
    encoded.insert(encoded.end(), {kThinkStart, 271, kThinkEnd, 271});
    Reset(encoded, {7, 7, 7, kEos});
    Harness model(2048, 4096);
    auto meta = Meta();
    meta.max_prefill_len = 512;
    auto input = TextInput();
    Require(model.insert(meta, input), "text prompt insert failed");
    std::ostringstream output;
    model.generate(meta, 0, output);

    const FakeEngine& engine = model.engine();
    Require(engine.prefills.size() == 3 && engine.prefills[0].size() == 512 &&
                engine.prefills[1].size() == 512 && engine.prefills[2].size() == 176,
            "a long prompt must prefill in 512-token chunks");
    const std::vector<int> decoded{kThinkStart, 271, kThinkEnd, 271, 7, 7};
    Require(engine.forwarded == decoded, "decode must run one token a step");
    std::vector<int> expected = prompt;
    expected.insert(expected.end(), decoded.begin(), decoded.end());
    Require(engine.executed == expected, "every prompt and reply token must run once, in order");
    Require(engine.replayed == 0, "nothing may run twice within a request");
    RequireHistoryMatchesEngine(model);
}

// ---- the next turn of a conversation -----------------------------------------

void UseWords() {
    g_words = {{100, "Hello"}, {101, " world"}, {102, "!"}, {103, " there"}, {110, "Let me think"},
               {198, "\n"}, {271, "\n\n"}, {kThinkStart, "<think>"}, {kThinkEnd, "</think>"},
               {kImEnd, "<|im_end|>"}};
    g_encodings[" world"] = {101};
}

/// \brief turn 1: prompt 10..13, then the reply `samples` (after the insert's own sample), cut at `limit`
void FirstTurn(Harness& model, std::vector<int> samples, int limit, bool think) {
    std::vector<int> encoded{10, 11, 12, 13, kThinkStart};
    if (think) {
        encoded.push_back(198);
    } else {
        encoded.insert(encoded.end(), {271, kThinkEnd, 271});
    }
    samples.insert(samples.begin(), 7);
    Reset(encoded, samples);
    UseWords();
    Require(model.configure_parameter("enable_think", think), "enable_think setup failed");
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "first turn insert failed");
    std::ostringstream output;
    model.generate(meta, limit, output);
    RequireHistoryMatchesEngine(model);
}

/// \brief turn 2: the prompt carries `reply` as the last assistant message, then a new user turn
chat_meta_info_t SecondTurn(Harness& model, const std::vector<int>& reply, bool think) {
    std::vector<int> encoded{10, 11, 12, 13};
    encoded.insert(encoded.end(), reply.begin(), reply.end());
    encoded.insert(encoded.end(), {kImEnd, 198, 20, 21, kImEnd, 198, 30, kThinkStart});
    if (think) {
        encoded.push_back(198);
    } else {
        encoded.insert(encoded.end(), {271, kThinkEnd, 271});
    }
    g_encoded_tokens = encoded;
    auto meta = Meta();
    meta.restore_allowed = true;
    auto input = TextInput();
    Require(model.insert(meta, input), "second turn insert failed");
    RequireHistoryMatchesEngine(model);
    return meta;
}

const std::vector<int> kNewTurn{kImEnd, 198, 20, 21, kImEnd, 198, 30};

void TheSameReplyContinuesTheHistoryTheEngineRan() {
    Harness model;
    FirstTurn(model, {100, 101, 102, kImEnd}, 0, false);
    Require(model.history().size() == 11, "the first turn must leave prompt, preamble and reply");
    const auto meta = SecondTurn(model, {100, 101, 102}, false);
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 0, "a continued history must not restore the checkpoint");
    Require(engine.prefills.back() == kNewTurn, "only what follows the last reply may run");
    Require(engine.replayed == 0, "the continued history must not run again");
    Require(meta.cached_prompt_tokens == 11, "the whole history the engine kept must count as cached");
    Require(meta.prompt_tokens == 18, "the prompt is the history plus the new turn");
}

void AReplyCutAtTheLimitRunsItsLastSampledTokenNext() {
    Harness model;
    FirstTurn(model, {100, 101, 102}, 2, false);
    // the engine never ran " world", the last token the client received
    Require(model.history().back() == 100, "a reply cut at its limit keeps only the tokens the engine ran");
    const auto meta = SecondTurn(model, {100, 101}, false);
    std::vector<int> expected{101};
    expected.insert(expected.end(), kNewTurn.begin(), kNewTurn.end());
    Require(model.engine().prefills.back() == expected, "the missing tail of the reply must run before the new turn");
    Require(model.engine().replayed == 0, "the continued history must not run again");
    Require(meta.cached_prompt_tokens == 9, "the history the engine ran must count as cached");
}

void AReplyCutInsideItsReasoningContinuesToo() {
    Harness model;
    FirstTurn(model, {110, 110}, 2, true);
    const auto meta = SecondTurn(model, {}, true);
    std::vector<int> expected(kNewTurn.begin(), kNewTurn.end());
    Require(model.engine().prefills.back() == expected, "an empty reply after open reasoning must continue");
    Require(model.engine().restores == 0, "a continued history must not restore the checkpoint");
    Require(meta.cached_prompt_tokens == 7, "the history with its reasoning must count as cached");
}

void AnEditedReplyRestoresTheCheckpointAndReportsTheReplay() {
    Harness model;
    FirstTurn(model, {100, 101, 102, kImEnd}, 0, false);
    const auto meta = SecondTurn(model, {100, 103}, false);
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "a history the engine did not run must restore the checkpoint");
    std::vector<int> expected{100, 103};
    expected.insert(expected.end(), kNewTurn.begin(), kNewTurn.end());
    Require(engine.prefills.back() == expected, "the restored checkpoint must be followed by the edited reply");
    Require(engine.replayed == 4, "an engine that cannot rewind runs the checkpoint again");
    Require(meta.cached_prompt_tokens == 0, "tokens the engine ran again must not count as cached");
}

void AnEngineThatRewindsRestoresTheCheckpoint() {
    Harness model;
    model.engine().rewinds = true;
    FirstTurn(model, {100, 101, 102, kImEnd}, 0, false);
    const auto meta = SecondTurn(model, {100, 101, 102}, false);
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "an engine that keeps its state on restore must restore, as the NPU does");
    std::vector<int> expected{100, 101, 102};
    expected.insert(expected.end(), kNewTurn.begin(), kNewTurn.end());
    Require(engine.prefills.back() == expected, "the reply as sent must follow the checkpoint");
    Require(engine.replayed == 0, "a rewind runs nothing again");
    Require(meta.cached_prompt_tokens == 4, "the checkpoint must count as cached");
}

// ---- what the server leaves to the model ---------------------------------------

void TheOgaBackendLeavesThePrefixMatchToTheModel() {
    Harness model;
    Require(model.manages_prefix_cache(),
            "on OGA the server must let every prompt restore instead of clearing on a message-level miss");
}

void AFirstPromptTheServerLetsRestoreStillRuns() {
    Reset({10, 11, 12, 13, kThinkStart, 271, kThinkEnd, 271}, {7});
    Harness model;
    auto meta = Meta();
    meta.restore_allowed = true;
    auto input = TextInput();
    Require(model.insert(meta, input), "a prompt without a checkpoint to restore must still run");
    Require(model.engine().prefills.back() == (std::vector<int>{10, 11, 12, 13}), "the whole prompt must run");
    Require(meta.cached_prompt_tokens == 0 && meta.prompt_tokens == 4, "nothing of a first prompt is cached");
    RequireHistoryMatchesEngine(model);
}

/// \brief a request that continues its final assistant message, as the server sends it on OGA
chat_meta_info_t ContinueMessage(Harness& model, const std::vector<int>& encoded, int limit,
                                 lm_uniform_input_t input = TextInput()) {
    g_encoded_tokens = encoded;
    auto meta = Meta();
    meta.continue_final_message = true;
    meta.restore_allowed = true;
    Require(model.insert(meta, input), "continued message insert failed");
    std::ostringstream output;
    model.generate(meta, limit, output);
    RequireHistoryMatchesEngine(model);
    return meta;
}

void ASecondContinuationReusesTheFirst() {
    Reset({}, {7, 9, 7, 7});
    Harness model;
    const std::vector<int> first{10, 11, 12, 13};
    Require(ContinueMessage(model, first, 2).cached_prompt_tokens == 0, "nothing of a first prompt is cached");
    // the client sends back the message with the text it got
    std::vector<int> second = first;
    second.insert(second.end(), {7, 9});
    const auto meta = ContinueMessage(model, second, 2);
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "a continued message must restore the checkpoint of the one before");
    Require(engine.prefills.back() == (std::vector<int>{7, 9}), "only the text the first continuation added may run");
    Require(engine.replayed == 0, "the first continuation must not run again");
    Require(meta.cached_prompt_tokens == 4, "the first continuation's prompt must count as cached");
}

// ---- images ------------------------------------------------------------------

qwen3_5vl_vision_geometry_t Geometry() {
    qwen3_5vl_vision_geometry_t geometry;
    geometry.patch_size = 16;
    geometry.merge_size = 2;
    geometry.temporal_patch_size = 2;
    geometry.min_pixels = 65536;
    geometry.max_pixels = 16777216;
    geometry.rescale_factor = 1.0f / 255.0f;
    geometry.image_mean = 0.5f;
    geometry.image_std = 0.5f;
    return geometry;
}

lm_uniform_input_t ImageInput(std::size_t images) {
    lm_uniform_input_t input;
    nlohmann::ordered_json list = nlohmann::ordered_json::array();
    for (std::size_t i = 0; i < images; ++i) list.push_back("aW1hZ2U=");
    input.messages = nlohmann::ordered_json::array();
    input.messages.push_back({{"role", "user"}, {"content", "what is this"}, {"images", list}});
    return input;
}

void OgaImagesReachTheEngineAsFloat16() {
    Reset({10, kVisionStart, kImagePad, kVisionEnd, 11, kVisionStart, kImagePad, kVisionEnd, 12,
           kThinkStart, 271, kThinkEnd, 271},
          {7, 7, kEos});
    g_grid_h = 4;
    g_grid_w = 6;
    Harness model(4096, 8192);
    model.set_vision_geometry(Geometry());
    auto meta = Meta();
    auto input = ImageInput(2);
    Require(model.insert(meta, input), "image prompt insert failed");

    // OGA runs both in one vision pass of at most 8100 patches of 16x16 pixels
    Require(g_max_pixels == (std::vector<unsigned>{4050 * 256, 4050 * 256}),
            "the images of a prompt must share the vision model's patch budget");
    const FakeEngine& engine = model.engine();
    Require(engine.vision.images.size() == 2, "both images must reach the engine");
    const std::size_t values = 4 * 6 * 3 * 2 * 16 * 16;
    for (const auto& image : engine.vision.images) {
        Require(image.grid_t == 1 && image.grid_h == 4 && image.grid_w == 6, "each image keeps its grid");
        Require(image.pixels.size() == values, "each image keeps its own pixels");
    }
    Require(engine.vision.images[0].pixels.front() == 0x3c00 && engine.vision.images[1].pixels.back() == 0x4000,
            "pixel values must reach OGA as float16");
    std::vector<int> expected{10, kVisionStart};
    expected.insert(expected.end(), 6, kImagePad);
    expected.insert(expected.end(), {kVisionEnd, 11, kVisionStart});
    expected.insert(expected.end(), 6, kImagePad);
    expected.insert(expected.end(), {kVisionEnd, 12});
    Require(engine.prefills.size() == 1 && engine.prefills[0] == expected,
            "each placeholder must become one image token per 2x2 patches");
}

void ImagesAreRefusedWithoutImagePreprocessing() {
    Reset({10, kVisionStart, kImagePad, kVisionEnd, 12, kThinkStart, 271, kThinkEnd, 271}, {7});
    Harness model;
    auto meta = Meta();
    auto input = ImageInput(1);
    try {
        model.insert(meta, input);
    } catch (const ModelRequestError& error) {
        Require(error.http_code() == 400, "an image the engine cannot run must be an HTTP 400");
        Require(g_max_pixels.empty() && model.engine().prefills.empty(), "nothing may run for a refused image");
        return;
    }
    throw std::runtime_error("an engine without image preprocessing must refuse images");
}

void MorePlaceholdersThanImagesIsARequestError() {
    Reset({10, kVisionStart, kImagePad, kVisionEnd, kVisionStart, kImagePad, kVisionEnd, 12,
           kThinkStart, 271, kThinkEnd, 271},
          {7});
    Harness model(4096, 8192);
    model.set_vision_geometry(Geometry());
    auto meta = Meta();
    auto input = ImageInput(1);
    try {
        model.insert(meta, input);
    } catch (const ModelRequestError& error) {
        Require(error.http_code() == 400, "a prompt that spells an extra placeholder must be an HTTP 400");
        Require(model.engine().prefills.empty(), "nothing may run for that prompt");
        return;
    }
    throw std::runtime_error("more placeholders than images must be refused");
}

/// \brief what {10, vision start, image, vision end, 12} becomes with a 4x4 image, then `more`
std::vector<int> OneImagePrompt(const std::vector<int>& more) {
    std::vector<int> tokens{10, kVisionStart};
    tokens.insert(tokens.end(), 4, kImagePad);
    tokens.insert(tokens.end(), {kVisionEnd, 12});
    tokens.insert(tokens.end(), more.begin(), more.end());
    return tokens;
}

void AnotherImageOfTheSameSizeRunsTheWholePromptAgain() {
    const std::vector<int> head{10, kVisionStart, kImagePad, kVisionEnd, 12};
    Reset(head, {7, 9, 7});
    Harness model(4096, 8192);
    model.set_vision_geometry(Geometry());
    ContinueMessage(model, head, 1, ImageInput(1));
    const auto first_pixels = model.engine().vision.images.at(0).pixels;

    // the same picture again
    g_max_pixels.clear();
    std::vector<int> longer = head;
    longer.insert(longer.end(), {7, 9});
    const auto same = ContinueMessage(model, longer, 1, ImageInput(1));
    Require(model.engine().prefills.back() == (std::vector<int>{7, 9}), "a checkpoint with the same image must be reused");
    Require(same.cached_prompt_tokens == 8, "the checkpoint with the same image must count as cached");

    // another picture of the same size, which gives the same tokens
    longer.push_back(5);
    const auto other = ContinueMessage(model, longer, 1, ImageInput(1));
    const FakeEngine& engine = model.engine();
    Require(engine.prefills.back() == OneImagePrompt({7, 9, 5}), "a checkpoint with another image must not be reused");
    Require(engine.vision.images.size() == 1 && engine.vision.images[0].pixels != first_pixels,
            "the new image must reach the engine");
    Require(other.cached_prompt_tokens == 0, "nothing of a prompt with another image is cached");
}

void TheSameImagePromptAgainCarriesItsImage() {
    Reset({10, kVisionStart, kImagePad, kVisionEnd, 12, kThinkStart, 271, kThinkEnd, 271},
          {7, 7, kEos, 7, 7, kEos});
    Harness model(4096, 8192);
    model.set_vision_geometry(Geometry());
    const auto send = [&model] {
        g_max_pixels.clear();
        auto meta = Meta();
        meta.restore_allowed = true;
        auto input = ImageInput(1);
        Require(model.insert(meta, input), "image prompt insert failed");
        std::ostringstream output;
        model.generate(meta, 0, output);
        RequireHistoryMatchesEngine(model);
        return meta;
    };
    send();
    model.engine().vision = {};
    const auto again = send();
    const FakeEngine& engine = model.engine();
    Require(engine.prefills.back() == OneImagePrompt({}), "a prompt the engine holds whole must run again from the start");
    Require(engine.vision.images.size() == 1, "a prompt run again from the start must carry its image");
    Require(again.cached_prompt_tokens == 0, "a prompt run again from the start is not cached");
}


void ANewImageAfterACachedOneRunsAlone() {
    const std::vector<int> head{10, kVisionStart, kImagePad, kVisionEnd, 12};
    Reset(head, {7, 7, 7});
    Harness model(4096, 8192);
    model.set_vision_geometry(Geometry());
    ContinueMessage(model, head, 1, ImageInput(1));

    // the same first picture, then a second one: each preprocessed image has every value n, so the first
    // keeps its pixels and the second gets 2s
    g_max_pixels.clear();
    std::vector<int> longer = head;
    longer.insert(longer.end(), {7, 11, kVisionStart, kImagePad, kVisionEnd, 13});
    const auto meta = ContinueMessage(model, longer, 1, ImageInput(2));
    const FakeEngine& engine = model.engine();
    std::vector<int> suffix{7, 11, kVisionStart};
    suffix.insert(suffix.end(), 4, kImagePad);
    suffix.insert(suffix.end(), {kVisionEnd, 13});
    Require(engine.prefills.back() == suffix, "only what follows the cached image may run");
    Require(engine.vision.images.size() == 1, "only the new image may reach the engine");
    const auto& pixels = engine.vision.images[0].pixels;
    Require(!pixels.empty() && std::all_of(pixels.begin(), pixels.end(), [](std::uint16_t v) { return v == 0x4000; }),
            "the engine must get the second image's pixels, not the cached one's");
    Require(meta.cached_prompt_tokens == 8, "the checkpoint with the cached image must count as cached");
}

}  // namespace

Tokenizer::Tokenizer(const std::string&) { is_doubled_encoded = false; }
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string& text) {
    const auto it = g_encodings.find(text);
    return it != g_encodings.end() ? it->second : g_encoded_tokens;
}
std::string Tokenizer::decode(const std::vector<int>& tokens) {
    if (g_words.empty()) return "decoded";
    std::string text;
    for (const int token : tokens) {
        const auto it = g_words.find(token);
        if (it != g_words.end()) text += it->second;
    }
    return text;
}
std::string Tokenizer::run_time_decoder(int token) { return "t" + std::to_string(token); }

Sampler::Sampler(int features, sampler_config& config)
    : in_features(features), rep_penalty(config.rep_penalty), freq_penalty(config.freq_penalty),
      pre_penalty(config.pre_penalty), top_k(config.top_k), top_p(config.top_p),
      min_p(config.min_p), temperature(config.temperature), total_tokens(0),
      freq_penalty_window(config.freq_penalty_window),
      rep_penalty_window(config.rep_penalty_window), repeat_last_n(config.repeat_last_n),
      use_optimized_sampling(config.use_optimized_sampling) {
    logits.resize(1);
    counters.resize(1);
    token_positions.resize(1, -1);
}
void Sampler::reset_penalties() {}
void Sampler::set_seed(uint64_t) {}
void Sampler::set_repeat_last_n(size_t n) { repeat_last_n = n; }
int Sampler::sample(buffer<bf16>&, const std::vector<uint32_t>*, sample_logprobs_t*) {
    if (g_sample_index < g_samples.size()) return g_samples[g_sample_index++];
    return 7;
}

SafeTensors::~SafeTensors() = default;

namespace utils {
std::string get_executable_directory() { return std::filesystem::current_path().string(); }
}  // namespace utils

ImageMemoryPool::ImageMemoryPool(size_t max_cached_per_size)
    : max_cached_per_size_(max_cached_per_size) {}
ImageReader::ImageReader(size_t) {}
ImageReader::~ImageReader() = default;

qwen3_5vl_image_t Qwen3_5VL::load_image(const std::string&) {
    throw std::logic_error("regression test unexpectedly loaded an image file");
}
qwen3_5vl_image_t Qwen3_5VL::load_image_base64(const std::string&) {
    qwen3_5vl_image_t image{};
    image.width = 64;
    image.height = 64;
    return image;
}
/// \note The n-th image preprocessed has every value n.
void Qwen3_5VL::preprocess_image(qwen3_5vl_image_t& image, std::vector<bf16>& pixel_values, unsigned max_pixels) {
    g_max_pixels.push_back(max_pixels);
    image.grid_h = g_grid_h;
    image.grid_w = g_grid_w;
    const auto& geometry = this->vision_geometry;
    const std::size_t count = static_cast<std::size_t>(g_grid_h) * g_grid_w * 3 * geometry.temporal_patch_size *
                              geometry.patch_size * geometry.patch_size;
    pixel_values.insert(pixel_values.end(), count, bf16(static_cast<float>(g_max_pixels.size())));
}

extern "C" const flm_state_api* qwen3_5vl_state_api(void*) { return nullptr; }

int main() {
    int failures = 0;
    const auto run = [&failures](const char* name, const std::function<void()>& test) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    run("rendered capacity with thinking off", [] { CapacityValidationIncludesTheRenderedPrompt(false); });
    run("rendered capacity with thinking on", [] { CapacityValidationIncludesTheRenderedPrompt(true); });
    run("backend limit with thinking off", [] { DecodeStopsAtBackendLimitWithAlignedHistory(false); });
    run("backend limit with thinking on", [] { DecodeStopsAtBackendLimitWithAlignedHistory(true); });
    run("first EOS is not forwarded", FirstSampleEosIsNeverForwarded);
    run("loop EOS is not forwarded", LoopEosIsNeverForwarded);
    run("one-token limit history", OneTokenGenerationLimitKeepsOnlyExecutedHistory);
    run("generate_with_prompt parity", GenerateWithPromptMatchesInsertThenGenerate);
    run("chunked prefill and decode run each token once", ChunkedPrefillAndDecodeRunEachTokenOnce);
    run("the same reply continues the engine's history", TheSameReplyContinuesTheHistoryTheEngineRan);
    run("a reply cut at its limit runs its last token next", AReplyCutAtTheLimitRunsItsLastSampledTokenNext);
    run("a reply cut inside its reasoning continues", AReplyCutInsideItsReasoningContinuesToo);
    run("an edited reply restores and reports the replay", AnEditedReplyRestoresTheCheckpointAndReportsTheReplay);
    run("an engine that rewinds restores the checkpoint", AnEngineThatRewindsRestoresTheCheckpoint);
    run("OGA leaves the prefix match to the model", TheOgaBackendLeavesThePrefixMatchToTheModel);
    run("a first prompt the server lets restore still runs", AFirstPromptTheServerLetsRestoreStillRuns);
    run("a second continuation reuses the first", ASecondContinuationReusesTheFirst);
    run("OGA images reach the engine as float16", OgaImagesReachTheEngineAsFloat16);
    run("images refused without preprocessing", ImagesAreRefusedWithoutImagePreprocessing);
    run("more placeholders than images", MorePlaceholdersThanImagesIsARequestError);
    run("another image of the same size runs the prompt again", AnotherImageOfTheSameSizeRunsTheWholePromptAgain);
    run("the same image prompt again carries its image", TheSameImagePromptAgainCarriesItsImage);
    run("a new image after a cached one runs alone", ANewImageAfterACachedOneRunsAlone);
    if (failures == 0) std::cout << "Qwen frontend regression tests passed\n";
    return failures == 0 ? 0 : 1;
}
