// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file test_gpt_oss_frontend.cpp
/// \brief fake-backend regression coverage for the GPT-OSS frontend on OGA

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_gpt_oss.hpp>
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

// the o200k_harmony ids of the special tokens; the words are made up
constexpr int kEndOfText = 199999;
constexpr int kReturn = 200002;
constexpr int kChannel = 200005;
constexpr int kStart = 200006;
constexpr int kEnd = 200007;
constexpr int kMessage = 200008;
constexpr int kCall = 200012;
constexpr int kFinal = 120;
constexpr int kAnalysis = 121;
constexpr int kAssistant = 122;
constexpr int kUser = 123;
std::vector<int> g_encoded_tokens;
/// \brief encodings of particular text; any other text encodes to g_encoded_tokens
std::map<std::string, std::vector<int>> g_encodings;
/// \brief the text of each token; when empty, decode() gives "decoded"
std::map<int, std::string> g_words;
std::vector<int> g_samples;
std::size_t g_sample_index{};

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
    buffer<bf16> prefill(std::vector<int>& tokens, void*) override {
        if (position + static_cast<int>(tokens.size()) > static_cast<int>(max_length)) {
            throw std::runtime_error("fake OGA engine prefill limit exceeded");
        }
        position += static_cast<int>(tokens.size());
        prefills.push_back(tokens);
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

class Harness final : public GPT_OSS {
public:
    explicit Harness(std::uint32_t backend_max_length = 64,
                     std::uint32_t frontend_max_length = 128)
        : GPT_OSS(nullptr) {
        auto backend = std::make_unique<FakeOgaBackend>(backend_max_length);
        engine_ = &backend->fake_engine();
        backend_ = std::move(backend);
        lm_engine = &backend_->engine();
        tokenizer = std::make_unique<Tokenizer>("");
        sampler_config config;
        sampler = std::make_unique<Sampler>(64, config);
        MAX_L = frontend_max_length;
        eos_token_ids = {kReturn, kEndOfText, kCall};
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

// ---- the next turn of a conversation -----------------------------------------

/// \brief the first prompt, which ends with the generation prompt "<|start|>assistant"
const std::vector<int> kPrompt{10, 11, kStart, kAssistant};
/// \brief what the template renders after the last reply: its <|end|>, then a new user turn
const std::vector<int> kNewTurn{kEnd, kStart, kUser, kMessage, 20, kEnd, kStart, kAssistant};

void UseWords() {
    g_words = {{100, "Hello"}, {101, " world"}, {103, " there"}, {110, "Let me think"},
               {kFinal, "final"}, {kAnalysis, "analysis"}, {kAssistant, "assistant"}, {kUser, "user"},
               {kChannel, "<|channel|>"}, {kStart, "<|start|>"}, {kEnd, "<|end|>"}, {kMessage, "<|message|>"}};
    g_encodings = {{"<|channel|>", {kChannel}}, {"<|message|>", {kMessage}}, {"<|end|>", {kEnd}}, {" world", {101}}};
}

/// \brief a reply as gpt-oss writes it: an analysis message, then `final` as the final message
std::vector<int> Reply(const std::vector<int>& final) {
    std::vector<int> reply{kChannel, kAnalysis, kMessage, 110, kEnd, kStart, kAssistant, kChannel, kFinal, kMessage};
    reply.insert(reply.end(), final.begin(), final.end());
    return reply;
}

/// \brief turn 1: kPrompt, then `reply` (the insert's own sample first) and <|return|>, cut at `limit`
void FirstTurn(Harness& model, std::vector<int> reply, int limit) {
    reply.push_back(kReturn);
    Reset(kPrompt, reply);
    UseWords();
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "first turn insert failed");
    std::ostringstream output;
    model.generate(meta, limit, output);
    RequireHistoryMatchesEngine(model);
}

/// \brief turn 2: the prompt carries `final` as the last reply, as the template renders it, then a new user turn
chat_meta_info_t SecondTurn(Harness& model, const std::vector<int>& final) {
    std::vector<int> encoded = kPrompt;
    encoded.insert(encoded.end(), {kChannel, kFinal, kMessage});
    encoded.insert(encoded.end(), final.begin(), final.end());
    encoded.insert(encoded.end(), kNewTurn.begin(), kNewTurn.end());
    g_encoded_tokens = encoded;
    auto meta = Meta();
    meta.restore_allowed = true;
    auto input = TextInput();
    Require(model.insert(meta, input), "second turn insert failed");
    RequireHistoryMatchesEngine(model);
    return meta;
}

void AReplyEndsWithoutRunningItsReturn() {
    Harness model;
    FirstTurn(model, Reply({100, 101}), 0);
    const FakeEngine& engine = model.engine();
    Require(std::find(engine.forwarded.begin(), engine.forwarded.end(), kReturn) == engine.forwarded.end(),
            "<|return|> must not be forwarded to a backend that opts out of the forward");
    Require(model.history().size() == kPrompt.size() + Reply({100, 101}).size(),
            "the history must end with the reply's last token");
}

void AFirstPromptTheServerLetsRestoreStillRuns() {
    Reset(kPrompt, {kChannel});
    Harness model;
    auto meta = Meta();
    meta.restore_allowed = true;
    auto input = TextInput();
    Require(model.insert(meta, input), "a prompt without a checkpoint to restore must still run");
    Require(model.engine().restores == 0, "there is no checkpoint to restore yet");
    Require(model.engine().prefills.back() == kPrompt, "the whole prompt must run");
    Require(meta.cached_prompt_tokens == 0 && meta.prompt_tokens == 4, "nothing of a first prompt is cached");
    RequireHistoryMatchesEngine(model);
}

void TheSameReplyContinuesTheHistoryTheEngineRan() {
    Harness model;
    FirstTurn(model, Reply({100, 101}), 0);
    const auto meta = SecondTurn(model, {100, 101});
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 0, "a continued history must not restore the checkpoint");
    Require(engine.prefills.back() == kNewTurn, "only what follows the last reply may run");
    Require(engine.replayed == 0, "the continued history must not run again");
    Require(meta.cached_prompt_tokens == 16, "the whole history the engine kept, analysis included, must count as cached");
    Require(meta.prompt_tokens == 24, "the prompt is the history plus the new turn");
}

void AReplyCutAtTheLimitRunsItsLastSampledTokenNext() {
    Harness model;
    FirstTurn(model, Reply({100, 101}), 12);
    // the engine never ran " world", the last token the client received
    Require(model.history().back() == 100, "a reply cut at its limit keeps only the tokens the engine ran");
    const auto meta = SecondTurn(model, {100, 101});
    std::vector<int> expected{101};
    expected.insert(expected.end(), kNewTurn.begin(), kNewTurn.end());
    Require(model.engine().prefills.back() == expected, "the missing tail of the reply must run before the new turn");
    Require(model.engine().replayed == 0, "the continued history must not run again");
    Require(meta.cached_prompt_tokens == 15, "the history the engine ran must count as cached");
}

void AReplyCutInsideItsAnalysisRestoresTheCheckpoint() {
    Harness model;
    FirstTurn(model, Reply({100, 101}), 3);
    // the client got reasoning and no content, and sends some of the reasoning back as the content
    SecondTurn(model, {110});
    Require(model.engine().restores == 1, "a reply without a final message must restore the checkpoint");
}

void AnEditedReplyRestoresTheCheckpointAndReportsTheReplay() {
    Harness model;
    FirstTurn(model, Reply({100, 101}), 0);
    const auto meta = SecondTurn(model, {100, 103});
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "a history the engine did not run must restore the checkpoint");
    std::vector<int> expected{kChannel, kFinal, kMessage, 100, 103};
    expected.insert(expected.end(), kNewTurn.begin(), kNewTurn.end());
    Require(engine.prefills.back() == expected, "the restored checkpoint must be followed by the edited reply");
    Require(engine.replayed == 4, "an engine that cannot rewind runs the checkpoint again");
    Require(meta.cached_prompt_tokens == 0, "tokens the engine ran again must not count as cached");
}

void AnEngineThatRewindsRestoresTheCheckpoint() {
    Harness model;
    model.engine().rewinds = true;
    FirstTurn(model, Reply({100, 101}), 0);
    const auto meta = SecondTurn(model, {100, 101});
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "an engine that keeps its state on restore must restore, as the NPU does");
    std::vector<int> expected{kChannel, kFinal, kMessage, 100, 101};
    expected.insert(expected.end(), kNewTurn.begin(), kNewTurn.end());
    Require(engine.prefills.back() == expected, "the reply as sent must follow the checkpoint");
    Require(engine.replayed == 0, "a rewind runs nothing again");
    Require(meta.cached_prompt_tokens == 4, "the checkpoint must count as cached");
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

extern "C" const flm_state_api* gpt_oss_npu_state_api(void*) { return nullptr; }

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
    run("a reply ends without running its <|return|>", AReplyEndsWithoutRunningItsReturn);
    run("a first prompt the server lets restore still runs", AFirstPromptTheServerLetsRestoreStillRuns);
    run("the same reply continues the engine's history", TheSameReplyContinuesTheHistoryTheEngineRan);
    run("a reply cut at its limit runs its last token next", AReplyCutAtTheLimitRunsItsLastSampledTokenNext);
    run("a reply cut inside its analysis restores", AReplyCutInsideItsAnalysisRestoresTheCheckpoint);
    run("an edited reply restores and reports the replay", AnEditedReplyRestoresTheCheckpointAndReportsTheReplay);
    run("an engine that rewinds restores the checkpoint", AnEngineThatRewindsRestoresTheCheckpoint);
    if (failures == 0) std::cout << "GPT-OSS frontend regression tests passed\n";
    return failures == 0 ? 0 : 1;
}
