// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file test_qwen_frontend.cpp
/// \brief fake-backend regression coverage for the Qwen3.5VL and Qwen2 OGA frontends

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_qwen2.hpp>
#include <AutoModel/modeling_qwen3_5vl.hpp>
#include <oga/oga_vision.hpp>
#include <tensor_utils/safe_tensors.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
/// \brief vocabulary width of the sampler; the fake engine returns this many logits
constexpr int kVocab = 64;
std::vector<int> g_encoded_tokens;
/// \brief encodings of particular text; any other text encodes to g_encoded_tokens
std::map<std::string, std::vector<int>> g_encodings;
/// \brief every text the tokenizer was asked to encode since the last Reset
std::vector<std::string> g_encoded_texts;
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
        return buffer<bf16>(kVocab);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void* payload) override {
        if (fail_prefill) {
            throw std::runtime_error("fake OGA engine injected prefill failure");
        }
        if (position + static_cast<int>(tokens.size()) > static_cast<int>(max_length)) {
            throw std::runtime_error("fake OGA engine prefill limit exceeded");
        }
        position += static_cast<int>(tokens.size());
        prefills.push_back(tokens);
        if (payload) {
            const auto& incoming = *static_cast<const flm::oga::OgaVisionInput*>(payload);
            bool pixels_changed = executed.empty() && !vision.images.empty() &&
                                  (incoming.images.size() != vision.images.size());
            if (!pixels_changed && executed.empty() && !vision.images.empty()) {
                for (std::size_t i = 0; i < incoming.images.size(); ++i) {
                    if (incoming.images[i].pixels != vision.images[i].pixels) {
                        pixels_changed = true;
                        break;
                    }
                }
            }
            if (pixels_changed) {
                ran.clear();
                snapshot.reset();
            }
            vision = incoming;
        }
        Run(tokens);
        return buffer<bf16>(kVocab);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override {
        position = 0;
        executed.clear();
        forwarded.clear();
        expected.clear();
        saved.reset();
    }
    buffer<bf16> get_k_cache(int, int) override { return buffer<bf16>(1); }
    buffer<bf16> get_v_cache(int, int) override { return buffer<bf16>(1); }
    int get_current_context_length() override { return position; }
    int checkpoint() override {
        ++checkpoints;
        saved = position;
        expected.clear();
        if (rewinds && !executed.empty() && ran == executed && !rewound) {
            snapshot = executed.size();
        }
        return position;
    }
    int restore() override {
        if (!saved) return -1;
        ++restores;
        position = *saved;
        executed.resize(static_cast<std::size_t>(*saved));
        return position;
    }
    void expect_prefill(const std::vector<int>& ids) {
        expected = executed;
        expected.insert(expected.end(), ids.begin(), ids.end());
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
    /// \brief slot-level prompt-end snapshot that outlives clear_context(), matching OgaEngine::Live::snapshot
    std::optional<std::size_t> snapshot;
    int checkpoints{};
    int restores{};
    int snapshot_rewinds{};
    /// \brief whether, like OGA 0.19, it can go back to the checkpoint without running it again
    bool rewinds = false;
    bool rewound = false;
    bool fail_next_rewind = false;
    bool fail_prefill = false;
    /// \brief what the engine's generator has run, as OgaEngine keeps it
    std::vector<int> ran;
    /// \brief token slices the fake engine actually had to execute (not served from slot)
    std::vector<std::vector<int>> engine_appends;
    /// \brief whole chunked prompt announced by expect_prefill()
    std::vector<int> expected;
    /// \brief tokens run a second time, counted as OgaEngine::replayed_tokens counts them
    std::uint64_t replayed{};
    /// \brief whether FakeOgaBackend exposes reused_tokens() to AutoModel (opt-in so pre-existing Qwen3_5VL tests keep origin/main expectations)
    bool reports_reused_tokens = false;
    /// \brief tokens served from the slot instead of re-prefilled, counted as OgaEngine::reused_tokens counts them
    std::uint64_t reused_total{};

private:
    void Run(const std::vector<int>& fresh) {
        const std::size_t before = executed.size();
        executed.insert(executed.end(), fresh.begin(), fresh.end());
        const std::vector<int>& target =
            expected.size() > executed.size() &&
                    std::equal(executed.begin(), executed.end(), expected.begin())
                ? expected
                : executed;
        if (&target == &executed) expected.clear();

        std::size_t reused = 0;
        if (ran.size() <= target.size() && std::equal(ran.begin(), ran.end(), target.begin())) {
            reused = ran.size();
        } else if (rewinds && snapshot && *snapshot <= ran.size() &&
                   *snapshot <= target.size() &&
                   std::equal(ran.begin(), ran.begin() + static_cast<std::ptrdiff_t>(*snapshot), target.begin())) {
            if (fail_next_rewind) {
                // Mirrors OgaEngine::Append() when OgaGenerator_RewindTo fails: disable rewinds and rebuild.
                fail_next_rewind = false;
                rewinds = false;
                snapshot.reset();
            } else {
                ran.resize(*snapshot);
                rewound = true;
                ++snapshot_rewinds;
                reused = *snapshot;
            }
        }

        const std::size_t covered = std::min(reused, executed.size());
        if (covered < before) {
            replayed += before - covered;
        } else {
            reused_total += covered - before;
        }
        if (reused < executed.size()) {
            rewound = false;
            engine_appends.emplace_back(executed.begin() + static_cast<std::ptrdiff_t>(covered),
                                        executed.end());
            ran = executed;
        }
    }
};

class FakeOgaBackend final : public flm::backend::ModelBackend {
public:
    explicit FakeOgaBackend(std::uint32_t max_length)
        : engine_(std::make_unique<FakeEngine>(max_length)) {}

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kOgaHipepBackendId; }
    std::uint32_t max_decode_length() const override {
        return engine_->max_length ? engine_->max_length - 1 : 0;
    }
    bool forwards_past_eos() const override { return false; }
    bool keeps_state_on_restore() const override { return engine_->rewinds; }
    std::uint64_t replayed_tokens() const override { return engine_->replayed; }
    std::uint64_t reused_tokens() const override {
        return engine_->reports_reused_tokens ? engine_->reused_total : 0;
    }
    void expect_prefill(const std::vector<int>& ids) override { engine_->expect_prefill(ids); }
    bool matches_token_prefixes() const override { return true; }
    FakeEngine& fake_engine() const { return *engine_; }

private:
    std::unique_ptr<FakeEngine> engine_;
};

class FakeFlmBackend final : public flm::backend::ModelBackend {
public:
    explicit FakeFlmBackend(std::uint32_t max_length)
        : engine_(std::make_unique<FakeEngine>(max_length)) {}

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kFlmBackendId; }
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
        sampler = std::make_unique<Sampler>(kVocab, config);
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

/// \brief chat_template.jinja of amd/Qwen2.5-14B-Instruct_gpuep_1.0_gpu_128 at efc1134c, as shipped
const char* const kQwen25ChatTemplate = R"jinja({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
)jinja";

/// \brief the Qwen2 frontend (qwen2.5-it, qwen2.5-coder-it) on the same fake OGA backend
class Qwen2Harness final : public Qwen2 {
public:
    explicit Qwen2Harness(std::uint32_t backend_max_length = 64,
                          std::uint32_t frontend_max_length = 128,
                          bool oga = true)
        : Qwen2(nullptr) {
        if (oga) {
            auto backend = std::make_unique<FakeOgaBackend>(backend_max_length);
            engine_ = &backend->fake_engine();
            engine_->reports_reused_tokens = true;
            backend_ = std::move(backend);
        } else {
            auto backend = std::make_unique<FakeFlmBackend>(backend_max_length);
            engine_ = &backend->fake_engine();
            backend_ = std::move(backend);
        }
        lm_engine = &backend_->engine();
        tokenizer = std::make_unique<Tokenizer>("");
        sampler_config config;
        sampler = std::make_unique<Sampler>(kVocab, config);
        MAX_L = frontend_max_length;
        eos_token_ids = {kEos, kImEnd};
        bos_token_id = -1;
        has_bos_token = false;
        log_raw_output = false;
    }

    /// \brief with the Qwen2.5 chat template, and what a catalog entry says about tools
    explicit Qwen2Harness(const nlohmann::ordered_json& catalog_entry) : Qwen2Harness() {
        chat_tmpl = std::make_unique<minja::chat_template>(kQwen25ChatTemplate, "", "<|im_end|>");
        extra_context = nlohmann::ordered_json::object();
        configure_tools(catalog_entry);
    }

    std::string apply_chat_template(nlohmann::ordered_json& messages,
                                    nlohmann::ordered_json tools,
                                    bool continue_final_message) override {
        if (!chat_tmpl) return "fake rendered prompt";
        return Qwen2::apply_chat_template(messages, std::move(tools), continue_final_message);
    }

    FakeEngine& engine() const { return *engine_; }

private:
    FakeEngine* engine_{};
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Reset(const std::vector<int>& encoded, const std::vector<int>& samples) {
    g_encoded_tokens = encoded;
    g_encodings.clear();
    g_encoded_texts.clear();
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

/// \brief a later turn: `rendered` is the conversation as the template renders it, before the generation prompt
chat_meta_info_t NextTurn(Harness& model, const std::vector<int>& rendered, bool think) {
    std::vector<int> encoded(rendered);
    encoded.push_back(kThinkStart);
    if (think) {
        encoded.push_back(198);
    } else {
        encoded.insert(encoded.end(), {271, kThinkEnd, 271});
    }
    g_encoded_tokens = encoded;
    auto meta = Meta();
    meta.restore_allowed = true;
    auto input = TextInput();
    Require(model.insert(meta, input), "next turn insert failed");
    RequireHistoryMatchesEngine(model);
    return meta;
}

void LaterTurnsContinueTheHistoryTheEngineRan(bool think) {
    // each reply as generate() samples it, reasoning first when thinking; the template renders its content alone
    const std::vector<std::vector<int>> contents{{100, 101, 102}, {103, 102}, {100, 103}};
    std::vector<int> samples;
    for (const auto& content : contents) {
        if (!samples.empty()) samples.push_back(7);  // the insert's own sample
        if (think) samples.insert(samples.end(), {110, kThinkEnd, 271});
        samples.insert(samples.end(), content.begin(), content.end());
        samples.push_back(kImEnd);
    }
    Harness model(128, 256);
    FirstTurn(model, samples, 0, think);
    std::vector<int> rendered{10, 11, 12, 13};
    for (std::size_t turn = 0; turn < contents.size(); ++turn) {
        const std::size_t ran = model.history().size();
        const int user = 40 + 10 * static_cast<int>(turn);
        const std::vector<int> new_turn{kImEnd, 198, user, user + 1, kImEnd, 198, 30};
        rendered.insert(rendered.end(), contents[turn].begin(), contents[turn].end());
        rendered.insert(rendered.end(), new_turn.begin(), new_turn.end());
        const FakeEngine& engine = model.engine();
        const int restores = engine.restores;
        const std::uint64_t replayed = engine.replayed;
        auto meta = NextTurn(model, rendered, think);
        const std::string at = "turn " + std::to_string(turn + 2) + ": ";
        if (think) {
            // continuing would keep the reply's reasoning, which the template leaves out
            Require(engine.restores == restores + 1, at + "a reply with reasoning must restore the checkpoint");
            Require(model.history() == rendered, at + "the prompt must run as rendered");
        } else {
            Require(engine.restores == restores, at + "a continued history must not restore the checkpoint");
            Require(engine.prefills.back() == new_turn, at + "only what follows the last reply may run");
            Require(engine.replayed == replayed, at + "the earlier turns must not run again");
            Require(meta.cached_prompt_tokens == static_cast<int>(ran),
                    at + "the whole history the engine kept must count as cached");
        }
        if (turn + 1 < contents.size()) {
            std::ostringstream output;
            model.generate(meta, 0, output);
            RequireHistoryMatchesEngine(model);
        }
    }
}

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

void AReplyCutInsideItsReasoningRestoresTheCheckpoint() {
    Harness model;
    FirstTurn(model, {110, 110}, 2, true);
    // the client got reasoning and no content, so the template renders the reply empty
    const auto meta = SecondTurn(model, {}, true);
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "a reply cut inside its reasoning must restore the checkpoint");
    Require(engine.prefills.back() == kNewTurn, "the prompt as rendered must follow the checkpoint");
    Require(meta.cached_prompt_tokens == 0, "tokens the engine ran again must not count as cached");
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

    // each image is sized on its own by the package's max_pixels: another image does not shrink it
    Require(g_max_pixels == (std::vector<unsigned>{16777216, 16777216}),
            "each image of a prompt must keep the package's max_pixels");
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

void OgaImagesPastTheContextAreRefusedBeforeAllArePreprocessed() {
    Reset({10, kVisionStart, kImagePad, kVisionEnd, 12, kThinkStart, 271, kThinkEnd, 271}, {7});
    g_grid_h = 4;
    g_grid_w = 6;  // 6 image tokens each
    Harness model(16, 64);  // decode cap 15
    model.set_vision_geometry(Geometry());
    auto meta = Meta();
    auto input = ImageInput(5);
    try {
        model.insert(meta, input);
    } catch (const ModelRequestError& error) {
        Require(error.http_code() == 400, "images past the context must be an HTTP 400");
        // 6, 12, then 18 tokens: the third image passes the cap, the last two are never preprocessed
        Require(g_max_pixels.size() == 3, "preprocessing must stop at the image that passes the context");
        Require(model.engine().prefills.empty(), "nothing may run for a refused prompt");
        return;
    }
    throw std::runtime_error("images past the context must be refused");
}

void OgaImagesAreRefusedAtTheImageThatReachesTheContext() {
    // 6 image tokens each. Decode cap 12: the second image reaches it. Cap 13: only the third passes it.
    for (const auto& [max_length, preprocessed] : {std::pair<std::uint32_t, std::size_t>{13, 2}, {14, 3}}) {
        Reset({10, kVisionStart, kImagePad, kVisionEnd, 12, kThinkStart, 271, kThinkEnd, 271}, {7});
        g_grid_h = 4;
        g_grid_w = 6;
        Harness model(max_length, 64);
        model.set_vision_geometry(Geometry());
        auto meta = Meta();
        auto input = ImageInput(3);
        bool refused = false;
        try {
            model.insert(meta, input);
        } catch (const ModelRequestError& error) {
            refused = error.http_code() == 400;
        }
        Require(refused, "images that reach the context must be an HTTP 400");
        Require(g_max_pixels.size() == preprocessed, "preprocessing must stop at the image that reaches the context");
    }
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

void Qwen2RunsTextOnOga() {
    Reset({10, 11, 12}, {7});
    Qwen2Harness model;
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "a text prompt must still run");
    Require(model.engine().prefills.back() == std::vector<int>({10, 11, 12}), "the whole prompt must run");
}

void RequireQwen2RefusesImage(lm_uniform_input_t input) {
    Reset({10, 11, 12}, {7});
    Qwen2Harness model;
    auto meta = Meta();
    try {
        model.insert(meta, input);
    } catch (const ModelRequestError& error) {
        Require(error.http_code() == 400, "an image sent to Qwen2 on OGA must be an HTTP 400");
        Require(model.engine().prefills.empty(), "nothing may run for a refused image");
        return;
    }
    throw std::runtime_error("Qwen2 on OGA must refuse images, not answer without them");
}

void Qwen2RefusesImagesOnOga() {
    RequireQwen2RefusesImage(ImageInput(1));
    auto cli = TextInput();
    cli.images = {"image.png"};
    RequireQwen2RefusesImage(cli);
}

/// \brief one size of a model in the catalog the server ships (src/model_list.json)
nlohmann::ordered_json CatalogEntry(const std::string& model, const std::string& size) {
    std::ifstream file(FLM_TEST_MODEL_LIST_PATH);
    if (!file) throw std::runtime_error(std::string("cannot open ") + FLM_TEST_MODEL_LIST_PATH);
    return nlohmann::ordered_json::parse(file).at("models").at(model).at(size);
}

/// \brief two functions as a request offers them; the first one's description is the argument
nlohmann::ordered_json OfferedTools(const std::string& shelf_description) {
    auto tools = nlohmann::ordered_json::parse(R"([
        {"type": "function", "function": {"name": "read_shelf", "description": "",
            "parameters": {"type": "object", "required": ["shelf"], "properties": {
                "shelf": {"type": "string"},
                "detail": {"type": "object", "properties": {"limit": {"type": "integer"}}}}}}},
        {"type": "function", "function": {"name": "count_boxes", "description": "Count the boxes on a pallet.",
            "parameters": {"type": "object", "properties": {"pallet": {"type": "number"}}}}}
    ])");
    tools[0]["function"]["description"] = shelf_description;
    return tools;
}

lm_uniform_input_t ChatInput(nlohmann::ordered_json tools) {
    lm_uniform_input_t input;
    input.messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "What is two plus four?"}}});
    input.tools = std::move(tools);
    return input;
}

/// \brief the prompt insert() handed to the tokenizer
std::string InsertedPrompt(Qwen2Harness& model, lm_uniform_input_t input) {
    Reset({10, 11, 12}, {7});
    auto meta = Meta();
    Require(model.insert(meta, input), "the chat prompt must run");
    for (const std::string& text : g_encoded_texts) {
        if (text.rfind("<|im_start|>system", 0) == 0) return text;
    }
    throw std::runtime_error("insert() did not encode a rendered chat prompt");
}

bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

void Qwen2ShowsOfferedToolsOnTheInstructOgaPackage() {
    Qwen2Harness model(CatalogEntry("qwen2.5-it-oga-hipep", "14b"));
    const std::string prompt = InsertedPrompt(model, ChatInput(OfferedTools("Shelf inventory.")));
    Require(Has(prompt, "<tools>") && Has(prompt, "</tools>"), "the prompt must carry the template's tool section");
    for (const char* part : {"read_shelf", "Shelf inventory.", "count_boxes", "Count the boxes on a pallet.",
                             "\"shelf\"", "\"string\"", "\"detail\"", "\"limit\"", "\"integer\"",
                             "\"pallet\"", "\"number\""}) {
        Require(Has(prompt, part), std::string("the prompt must show the model ") + part);
    }
    const std::string turn = "What is two plus four?<|im_end|>\n<|im_start|>assistant\n";
    Require(prompt.size() > turn.size() && prompt.compare(prompt.size() - turn.size(), turn.size(), turn) == 0 &&
                prompt.rfind("</tools>") < prompt.size() - turn.size(),
            "the question and the generation prompt must follow the tools");

    // the same text when the template is applied directly, as a caller outside insert() would
    auto input = ChatInput(OfferedTools("Shelf inventory."));
    Require(model.apply_chat_template(input.messages, input.tools, false) == prompt,
            "apply_chat_template must render the tools insert() renders");
}

void Qwen2ShowsNoToolsWhenNoneAreOffered() {
    Qwen2Harness model(CatalogEntry("qwen2.5-it-oga-hipep", "14b"));
    // tool_choice "none" reaches the model as an empty list; a request without tools as an empty list too
    const std::string none = InsertedPrompt(model, ChatInput(nlohmann::ordered_json::array()));
    Require(!Has(none, "<tools>") && !Has(none, "# Tools"), "no offered tool, no tool section");
    Require(none == InsertedPrompt(model, ChatInput(nlohmann::ordered_json())),
            "a request without a tools field must render like an empty list");
    auto input = ChatInput(nlohmann::ordered_json());
    Require(model.apply_chat_template(input.messages, nlohmann::ordered_json::object(), false) == none,
            "apply_chat_template without tools must render the plain prompt");
}

void Qwen2ChangedDefinitionChangesThePrompt() {
    Qwen2Harness model(CatalogEntry("qwen2.5-it-oga-hipep", "14b"));
    const std::string before = InsertedPrompt(model, ChatInput(OfferedTools("Shelf inventory.")));
    const std::string after = InsertedPrompt(model, ChatInput(OfferedTools("Shelf inventory, by aisle.")));
    Require(before != after, "a changed description must change the prompt, or the old prompt's cache would hit");
    Require(Has(after, "Shelf inventory, by aisle.") && !Has(before, "by aisle"),
            "each prompt must carry its own description");
    const std::size_t common = static_cast<std::size_t>(
        std::mismatch(before.begin(), before.end(), after.begin(), after.end()).first - before.begin());
    // the template's instructions name the tags too; the definitions are between the last pair
    Require(before.rfind("<tools>") < common && common < before.rfind("</tools>"),
            "the prompts must part inside the tool section");

    // a name or a parameter type is as much a part of the definition as its description
    auto renamed = OfferedTools("Shelf inventory.");
    renamed[0]["function"]["name"] = "read_rack";
    auto retyped = OfferedTools("Shelf inventory.");
    retyped[0]["function"]["parameters"]["properties"]["shelf"]["type"] = "integer";
    Require(InsertedPrompt(model, ChatInput(renamed)) != before, "a renamed function must change the prompt");
    Require(InsertedPrompt(model, ChatInput(retyped)) != before, "a changed parameter type must change the prompt");
}

void Qwen2PackagesWithoutTheCatalogFlagKeepTheirPrompt() {
    Qwen2Harness instruct(CatalogEntry("qwen2.5-it-oga-hipep", "14b"));
    const std::string plain = InsertedPrompt(instruct, ChatInput(nlohmann::ordered_json::array()));
    const std::vector<std::pair<std::string, std::string>> others = {
        {"qwen2.5-it", "3b"}, {"qwen2.5-coder-it-oga-hipep", "14b"}};
    for (const auto& [name, size] : others) {
        Qwen2Harness model(CatalogEntry(name, size));
        Require(InsertedPrompt(model, ChatInput(OfferedTools("Shelf inventory."))) == plain,
                name + " must keep the prompt it had: no tool section");
    }
}

void Qwen2ExactRetryRewindsToPromptCheckpoint() {
    Reset({10, 11, 12, 13}, {7, 100, 101, kImEnd, 7, 100, 101, kImEnd, 7, kImEnd});
    Qwen2Harness model;
    model.engine().rewinds = true;

    const auto run_turn = [&model] {
        auto meta = Meta();
        meta.restore_allowed = true;
        auto input = TextInput();
        Require(model.insert(meta, input), "Qwen2 OGA insert failed");
        std::ostringstream output;
        model.generate(meta, 0, output);
        return meta;
    };

    const auto first = run_turn();
    Require(first.prompt_tokens == 4 && first.cached_prompt_tokens == 0,
            "the first prompt must run cold");
    Require(model.engine().checkpoints == 1 && model.engine().snapshot == std::optional<std::size_t>{4},
            "a successful Qwen2 OGA prefill must establish a prompt-end checkpoint before decode");

    const auto retry1 = run_turn();
    Require(model.engine().snapshot_rewinds == 1,
            "retrying the identical prompt after decode must rewind to the prompt-end checkpoint");
    Require(retry1.prompt_tokens == 4 && retry1.cached_prompt_tokens == 4,
            "an exact retry must report the whole prompt as cached");
    Require(model.engine().replayed == 0, "an exact retry must replay zero tokens");

    // a second immediate retry still reuses the same prompt-end checkpoint
    const auto retry2 = run_turn();
    Require(retry2.prompt_tokens == 4 && retry2.cached_prompt_tokens == 4,
            "a repeated exact retry must keep reusing the prompt-end checkpoint");
    Require(model.engine().replayed == 0, "a repeated exact retry must replay zero tokens");
}

void Qwen2ChunkedExactRetryRewindsOnce() {
    std::vector<int> prompt(1200);
    for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 1000 + static_cast<int>(i);
    Reset(prompt, {7, 100, kImEnd, 7, 100, kImEnd});
    Qwen2Harness model(2048, 4096);
    model.engine().rewinds = true;

    const auto run_turn = [&model] {
        auto meta = Meta();
        meta.max_prefill_len = 512;
        meta.restore_allowed = true;
        auto input = TextInput();
        Require(model.insert(meta, input), "Qwen2 chunked OGA insert failed");
        std::ostringstream output;
        model.generate(meta, 0, output);
        return meta;
    };

    const auto first = run_turn();
    Require(first.prompt_tokens == 1200 && first.cached_prompt_tokens == 0,
            "the cold chunked prompt must run all 1,200 tokens");
    Require(model.engine().snapshot == std::optional<std::size_t>{1200},
            "the prompt-end checkpoint must cover the whole chunked prompt");

    const auto retry = run_turn();
    Require(model.engine().snapshot_rewinds == 1,
            "retrying a multi-chunk prompt must rewind once");
    Require(retry.prompt_tokens == 1200 && retry.cached_prompt_tokens == 1200,
            "retrying a multi-chunk prompt must cache every prompt token");
    Require(model.engine().replayed == 0, "a chunked exact retry must replay zero tokens");
}

void Qwen2ExactRetryReplaysHonestlyWhenEngineCannotRewind() {
    // 1. Engine without rewind after decode advanced past prompt end: must replay full prompt and report 0 cached.
    {
        Reset({10, 11, 12, 13}, {7, 100, kImEnd, 7, 100, kImEnd});
        Qwen2Harness model;
        model.engine().rewinds = false;

        const auto run_turn = [&model] {
            auto meta = Meta();
            meta.restore_allowed = true;
            auto input = TextInput();
            Require(model.insert(meta, input), "Qwen2 insert failed");
            std::ostringstream output;
            model.generate(meta, 0, output);
            return meta;
        };

        run_turn();
        Require(model.engine().checkpoints == 1 && !model.engine().snapshot,
                "an engine that cannot rewind must not record a rewindable snapshot");
        const auto appends_before_retry = model.engine().engine_appends.size();
        const auto retry = run_turn();
        Require(model.engine().prefills.back() == (std::vector<int>{10, 11, 12, 13}),
                "non-rewinding retry must send the full prompt to the engine");
        Require(model.engine().engine_appends.size() > appends_before_retry &&
                    model.engine().engine_appends[appends_before_retry] == (std::vector<int>{10, 11, 12, 13}),
                "non-rewinding retry after decode must execute the full prompt again");
        Require(retry.prompt_tokens == 4 && retry.cached_prompt_tokens == 0,
                "when the engine cannot rewind after decode, retry must report zero cached tokens");
    }

    // 2. Engine without rewind when first sampled token is EOS (generator stays at prompt end):
    //    matches OgaEngine::Run() counting live-slot reuse without needing a rewind.
    {
        Reset({10, 11, 12, 13}, {kImEnd, kImEnd});
        Qwen2Harness model;
        model.engine().rewinds = false;
        auto meta1 = Meta();
        auto input = TextInput();
        Require(model.insert(meta1, input), "first immediate-EOS insert failed");
        std::ostringstream out1;
        model.generate(meta1, 0, out1);
        const auto appends_after_first = model.engine().engine_appends.size();

        auto meta2 = Meta();
        Require(model.insert(meta2, input), "second immediate-EOS insert failed");
        Require(model.engine().snapshot_rewinds == 0, "live-slot hit at prompt end does not rewind");
        Require(model.engine().engine_appends.size() == appends_after_first,
                "live-slot hit at prompt end must not re-execute the prompt");
        Require(meta2.prompt_tokens == 4 && meta2.cached_prompt_tokens == 4,
                "live-slot hit at prompt end must report full prompt reuse even when rewinds is false");
    }

    // 3. Snapshot was established, decode advanced, then provider refuses RewindTo on retry:
    //    must drop the snapshot, replay the whole prompt, and report zero cached tokens.
    {
        Reset({10, 11, 12, 13}, {7, 100, kImEnd, 7, 100, kImEnd});
        Qwen2Harness model;
        model.engine().rewinds = true;

        auto first = Meta();
        auto input = TextInput();
        Require(model.insert(first, input), "cold insert before refused rewind failed");
        std::ostringstream out1;
        model.generate(first, 0, out1);
        Require(model.engine().snapshot == std::optional<std::size_t>{4},
                "cold insert must establish the prompt-end snapshot");

        model.engine().fail_next_rewind = true;
        const auto appends_before_retry = model.engine().engine_appends.size();
        auto retry = Meta();
        Require(model.insert(retry, input), "retry insert after refused rewind failed");
        std::ostringstream out2;
        model.generate(retry, 0, out2);
        Require(model.engine().snapshot_rewinds == 0 && !model.engine().rewinds && !model.engine().snapshot,
                "a refused rewind must disable further rewinds and clear the snapshot");
        Require(model.engine().engine_appends.size() > appends_before_retry &&
                    model.engine().engine_appends[appends_before_retry] == (std::vector<int>{10, 11, 12, 13}),
                "a refused rewind must replay the whole prompt");
        Require(retry.prompt_tokens == 4 && retry.cached_prompt_tokens == 0,
                "a refused rewind must report zero cached tokens instead of a false hit");
    }
}

void Qwen2FailedOrCancelledPrefillDoesNotCheckpoint() {
    // 1. cancelled before prefill
    Reset({10, 11, 12, 13}, {7});
    {
        Qwen2Harness model;
        model.engine().rewinds = true;
        auto meta = Meta();
        auto input = TextInput();
        Require(!model.insert(meta, input, [] { return true; }),
                "a cancelled prefill must return false");
        Require(model.engine().checkpoints == 0 && !model.engine().snapshot,
                "a cancelled prefill must not establish a checkpoint");
    }
    // 2. cancelled mid-chunk after the first 512-token chunk ran
    {
        std::vector<int> prompt(1024);
        for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = 100 + static_cast<int>(i);
        Reset(prompt, {7});
        Qwen2Harness model(2048, 4096);
        model.engine().rewinds = true;
        auto meta = Meta();
        meta.max_prefill_len = 512;
        auto input = TextInput();
        int checks = 0;
        Require(!model.insert(meta, input, [&checks] { return ++checks >= 2; }),
                "a mid-chunk cancellation must return false");
        Require(model.engine().checkpoints == 0 && !model.engine().snapshot,
                "a half-prefilled prompt must not establish a checkpoint");
    }
    // 3. prompt exceeding context capacity (throws HTTP 400 in _shared_validate_capacity)
    {
        Reset({10, 11, 12, 13, 14}, {7});
        Qwen2Harness model(64, 4);
        model.engine().rewinds = true;
        auto meta = Meta();
        auto input = TextInput();
        bool threw = false;
        try {
            model.insert(meta, input);
        } catch (const ModelRequestError& error) {
            threw = error.http_code() == 400;
        }
        Require(threw, "an oversized prompt must throw HTTP 400");
        Require(model.engine().checkpoints == 0 && !model.engine().snapshot,
                "a refused oversized prompt must not establish a checkpoint");
    }
    // 4. refused image request
    {
        Reset({10, 11, 12}, {7});
        Qwen2Harness model;
        model.engine().rewinds = true;
        auto meta = Meta();
        bool threw = false;
        try {
            auto img = ImageInput(1);
            model.insert(meta, img);
        } catch (const ModelRequestError& error) {
            threw = error.http_code() == 400;
        }
        Require(threw, "an image on Qwen2 OGA must throw HTTP 400");
        Require(model.engine().checkpoints == 0 && !model.engine().snapshot,
                "a refused image request must not establish a checkpoint");
    }
    // 5. engine prefill throws mid-request
    {
        Reset({10, 11, 12, 13}, {7});
        Qwen2Harness model;
        model.engine().rewinds = true;
        model.engine().fail_prefill = true;
        auto meta = Meta();
        auto input = TextInput();
        bool threw = false;
        try {
            model.insert(meta, input);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        Require(threw, "a failing engine prefill must propagate");
        Require(model.engine().checkpoints == 0 && !model.engine().snapshot,
                "a failed engine prefill must not establish a checkpoint");
    }
}

void Qwen2ChangedEndingDoesNotFalseHit() {
    Reset({10, 11, 12, 13}, {7, 100, kImEnd, 7, 100, kImEnd, 7, 100, kImEnd});
    Qwen2Harness model;
    model.engine().rewinds = true;

    auto first_meta = Meta();
    first_meta.restore_allowed = true;
    auto input = TextInput();
    Require(model.insert(first_meta, input), "first Qwen2 insert failed");
    std::ostringstream first_out;
    model.generate(first_meta, 0, first_out);

    // same prefix [10, 11, 12], changed final token 99
    g_encoded_tokens = {10, 11, 12, 99};
    const auto appends_before_changed = model.engine().engine_appends.size();
    auto changed_meta = Meta();
    changed_meta.restore_allowed = true;
    Require(model.insert(changed_meta, input), "changed-ending Qwen2 insert failed");
    Require(model.engine().snapshot_rewinds == 0,
            "a prompt whose ending changed must not rewind to the previous prompt-end snapshot");
    Require(model.engine().engine_appends.size() > appends_before_changed &&
                model.engine().engine_appends[appends_before_changed] == (std::vector<int>{10, 11, 12, 99}),
            "a prompt whose ending changed must execute the new prompt from scratch");
    Require(changed_meta.prompt_tokens == 4 && changed_meta.cached_prompt_tokens == 0,
            "a prompt whose ending changed must not report a false cache hit");

    // positive control: decoding the changed prompt and retrying it identically DOES hit its new checkpoint
    std::ostringstream changed_out;
    model.generate(changed_meta, 0, changed_out);
    const auto appends_before_retry = model.engine().engine_appends.size();
    auto retry_meta = Meta();
    retry_meta.restore_allowed = true;
    Require(model.insert(retry_meta, input), "retry of changed prompt failed");
    Require(model.engine().snapshot_rewinds == 1,
            "retrying the changed prompt after decode must rewind to its own prompt-end snapshot");
    Require(model.engine().engine_appends.size() == appends_before_retry,
            "retrying the changed prompt must not re-execute prompt tokens");
    Require(retry_meta.prompt_tokens == 4 && retry_meta.cached_prompt_tokens == 4,
            "retrying the changed prompt must report full cache reuse");
}

void Qwen2NonOgaLegacyInsertDoesNotCheckpoint() {
    Reset({10, 11, 12, 13}, {7});
    Qwen2Harness model(64, 128, /*oga=*/false);
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "non-OGA Qwen2 insert must still succeed");
    Require(model.engine().checkpoints == 0,
            "the OGA prompt-end checkpoint must not run on non-OGA backends");
}

}  // namespace

Tokenizer::Tokenizer(const std::string&) { is_doubled_encoded = false; }
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string& text) {
    g_encoded_texts.push_back(text);
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
      repeat_last_n(config.repeat_last_n),
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
// Speculative verify rows (AutoModel::_sample_row) sample the same scripted way.
int Sampler::sample(const float*, const std::vector<uint32_t>*, sample_logprobs_t*) {
    if (g_sample_index < g_samples.size()) return g_samples[g_sample_index++];
    return 7;
}
int Sampler::argmax(const float* x, int n, const std::vector<uint32_t>* allowed) {
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (allowed && !((*allowed)[i / 32] >> (i % 32) & 1u)) continue;
        if (best < 0 || x[i] > x[best]) best = i;
    }
    return best;
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
extern "C" const flm_state_api* qwen2_npu_state_api(void*) { return nullptr; }

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
    run("every later turn continues, thinking off", [] { LaterTurnsContinueTheHistoryTheEngineRan(false); });
    run("every later turn runs as rendered, thinking on", [] { LaterTurnsContinueTheHistoryTheEngineRan(true); });
    run("a reply cut at its limit runs its last token next", AReplyCutAtTheLimitRunsItsLastSampledTokenNext);
    run("a reply cut inside its reasoning restores", AReplyCutInsideItsReasoningRestoresTheCheckpoint);
    run("an edited reply restores and reports the replay", AnEditedReplyRestoresTheCheckpointAndReportsTheReplay);
    run("an engine that rewinds restores the checkpoint", AnEngineThatRewindsRestoresTheCheckpoint);
    run("OGA leaves the prefix match to the model", TheOgaBackendLeavesThePrefixMatchToTheModel);
    run("a first prompt the server lets restore still runs", AFirstPromptTheServerLetsRestoreStillRuns);
    run("a second continuation reuses the first", ASecondContinuationReusesTheFirst);
    run("OGA images reach the engine as float16", OgaImagesReachTheEngineAsFloat16);
    run("images refused without preprocessing", ImagesAreRefusedWithoutImagePreprocessing);
    run("OGA images past the context are refused early", OgaImagesPastTheContextAreRefusedBeforeAllArePreprocessed);
    run("OGA images are refused at the image that reaches the context", OgaImagesAreRefusedAtTheImageThatReachesTheContext);
    run("more placeholders than images", MorePlaceholdersThanImagesIsARequestError);
    run("another image of the same size runs the prompt again", AnotherImageOfTheSameSizeRunsTheWholePromptAgain);
    run("the same image prompt again carries its image", TheSameImagePromptAgainCarriesItsImage);
    run("a new image after a cached one runs alone", ANewImageAfterACachedOneRunsAlone);
    run("Qwen2 runs text on OGA", Qwen2RunsTextOnOga);
    run("Qwen2 refuses images on OGA", Qwen2RefusesImagesOnOga);
    run("Qwen2.5 Instruct on OGA is shown the offered tools", Qwen2ShowsOfferedToolsOnTheInstructOgaPackage);
    run("Qwen2.5 Instruct on OGA is shown no tools when none are offered", Qwen2ShowsNoToolsWhenNoneAreOffered);
    run("a changed tool definition changes the Qwen2.5 prompt", Qwen2ChangedDefinitionChangesThePrompt);
    run("Qwen2 packages without the catalog flag keep their prompt", Qwen2PackagesWithoutTheCatalogFlagKeepTheirPrompt);
    run("Qwen2 exact retry rewinds to the prompt-end checkpoint", Qwen2ExactRetryRewindsToPromptCheckpoint);
    run("Qwen2 chunked exact retry rewinds once", Qwen2ChunkedExactRetryRewindsOnce);
    run("Qwen2 exact retry replays honestly when engine cannot rewind", Qwen2ExactRetryReplaysHonestlyWhenEngineCannotRewind);
    run("Qwen2 failed or cancelled prefill does not checkpoint", Qwen2FailedOrCancelledPrefillDoesNotCheckpoint);
    run("Qwen2 changed ending does not false hit", Qwen2ChangedEndingDoesNotFalseHit);
    run("Qwen2 non-OGA legacy insert does not checkpoint", Qwen2NonOgaLegacyInsertDoesNotCheckpoint);
    if (failures == 0) std::cout << "Qwen frontend regression tests passed\n";
    return failures == 0 ? 0 : 1;
}
