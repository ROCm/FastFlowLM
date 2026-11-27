// Copyright (c) 2026 Advanced Micro Devices, Inc.

/// \file test_template_chat_frontend.cpp
/// \brief fake-backend regression coverage for the TemplateChat frontends on OGA

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_template_chat.hpp>
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

// made-up ids
constexpr int kEos = 1;
constexpr int kImEnd = 2;
constexpr int kAssistant = 3;
constexpr int kUser = 4;
constexpr int kThink = 5;
constexpr int kNewline = 6;
/// \brief vocabulary width of the sampler; the fake engine returns this many logits
constexpr int kVocab = 64;
std::vector<int> g_encoded_tokens;
/// \brief encodings of particular text; any other text encodes to g_encoded_tokens
std::map<std::string, std::vector<int>> g_encodings;
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
        Run({token});
        return buffer<bf16>(kVocab);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void*) override {
        if (position + static_cast<int>(tokens.size()) > static_cast<int>(max_length)) {
            throw std::runtime_error("fake OGA engine prefill limit exceeded");
        }
        position += static_cast<int>(tokens.size());
        prefills.push_back(tokens);
        Run(tokens);
        return buffer<bf16>(kVocab);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override {
        position = 0;
        executed.clear();
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
    std::uint64_t replayed_tokens() const override { return engine_->replayed; }
    bool matches_token_prefixes() const override { return true; }
    FakeEngine& fake_engine() const { return *engine_; }

private:
    std::unique_ptr<FakeEngine> engine_;
};

template <typename Frontend>
class Harness final : public Frontend {
public:
    Harness() : Frontend(nullptr) {
        auto backend = std::make_unique<FakeOgaBackend>(64);
        engine_ = &backend->fake_engine();
        this->backend_ = std::move(backend);
        this->lm_engine = &this->backend_->engine();
        this->tokenizer = std::make_unique<Tokenizer>("");
        sampler_config config;
        this->sampler = std::make_unique<Sampler>(kVocab, config);
        this->MAX_L = 128;
        this->eos_token_ids = {kEos};
        this->bos_token_id = -1;
        this->has_bos_token = false;
        this->log_raw_output = false;
    }

    std::string apply_chat_template(nlohmann::ordered_json&, nlohmann::ordered_json, bool) override {
        return "fake rendered prompt";
    }

    const std::vector<int>& history() const { return this->token_history; }
    FakeEngine& engine() const { return *engine_; }

private:
    FakeEngine* engine_{};
};

using StepModel = Harness<Step3_5>;
using LagunaModel = Harness<Laguna>;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Model>
void RequireHistoryMatchesEngine(const Model& model) {
    Require(model.history() == model.engine().executed,
            "token history must equal the tokens committed to the fake engine");
}

/// \brief the first prompt's turns, up to the reply header "<|im_start|>assistant\n"
const std::vector<int> kHeader{10, 11, kImEnd, kAssistant};
/// \brief the reasoning block Step's template opens every reply with
const std::vector<int> kOpening{kThink, kNewline};
/// \brief what the template renders after the reply header on the next turn: the reply's
///        content without its reasoning, then a new user turn and its reply header
const std::vector<int> kNextTurn{30, kImEnd, kUser, 20, kImEnd, kAssistant};

std::vector<int> Concat(std::initializer_list<std::vector<int>> parts) {
    std::vector<int> all;
    for (const auto& part : parts) all.insert(all.end(), part.begin(), part.end());
    return all;
}

chat_meta_info_t Meta() {
    chat_meta_info_t meta;
    meta.max_prefill_len = 64;
    meta.restore_allowed = true;
    return meta;
}

lm_uniform_input_t TextInput() {
    lm_uniform_input_t input;
    input.prompt = "hello";
    return input;
}

void UseOpening() {
    g_encodings = {{"<think>\n", kOpening}};
}

/// \brief inserts `encoded` and generates reasoning, an answer and the end of turn
template <typename Model>
chat_meta_info_t Turn(Model& model, const std::vector<int>& encoded, bool generate = true) {
    g_encoded_tokens = encoded;
    g_samples = {40, 41, kEos};
    g_sample_index = 0;
    auto meta = Meta();
    auto input = TextInput();
    Require(model.insert(meta, input), "insert failed");
    if (generate) {
        std::ostringstream output;
        model.generate(meta, 0, output);
    }
    RequireHistoryMatchesEngine(model);
    return meta;
}

void StepCheckpointsAtTheReplyHeader() {
    UseOpening();
    StepModel model;
    const auto meta = Turn(model, Concat({kHeader, kOpening}), false);
    const FakeEngine& engine = model.engine();
    Require(engine.prefills.size() == 2 && engine.prefills[0] == kHeader && engine.prefills[1] == kOpening,
            "the prompt must run up to the reply header, then the reasoning opening");
    Require(engine.saved == static_cast<int>(kHeader.size()), "the checkpoint must sit at the reply header");
    Require(engine.restores == 0, "there is no checkpoint to restore yet");
    Require(meta.prompt_tokens == 6 && meta.cached_prompt_tokens == 0, "nothing of a first prompt is cached");
}

void StepNextTurnRewindsToTheReplyHeader() {
    UseOpening();
    StepModel model;
    model.engine().rewinds = true;
    Turn(model, Concat({kHeader, kOpening}));
    const auto meta = Turn(model, Concat({kHeader, kNextTurn, kOpening}), false);
    const FakeEngine& engine = model.engine();
    Require(engine.restores == 1, "the next turn must restore the checkpoint");
    Require(engine.prefills[engine.prefills.size() - 2] == kNextTurn && engine.prefills.back() == kOpening,
            "only what follows the reply header may run, then the reasoning opening");
    Require(engine.replayed == 0, "a rewind runs nothing again");
    Require(meta.cached_prompt_tokens == static_cast<int>(kHeader.size()), "the reply header must count as cached");
    Require(meta.prompt_tokens == 12, "the prompt is the header, the next turn and the opening");
    Require(engine.saved == static_cast<int>(kHeader.size() + kNextTurn.size()),
            "the checkpoint must move to the new reply header");
}

void StepNextTurnReportsAReplay() {
    UseOpening();
    StepModel model;
    Turn(model, Concat({kHeader, kOpening}));
    const auto meta = Turn(model, Concat({kHeader, kNextTurn, kOpening}), false);
    Require(model.engine().replayed == kHeader.size(), "an engine that cannot rewind runs the header again");
    Require(meta.cached_prompt_tokens == 0, "tokens the engine ran again must not count as cached");
}

void StepRetryRunsOnlyTheOpening() {
    UseOpening();
    StepModel model;
    model.engine().rewinds = true;
    Turn(model, Concat({kHeader, kOpening}));
    const auto prefills_before = model.engine().prefills.size();
    const auto meta = Turn(model, Concat({kHeader, kOpening}), false);
    const FakeEngine& engine = model.engine();
    Require(engine.prefills.size() == prefills_before + 1 && engine.prefills.back() == kOpening,
            "a retry must run only the reasoning opening after the restored header");
    Require(meta.cached_prompt_tokens == static_cast<int>(kHeader.size()) && meta.prompt_tokens == 6,
            "the reply header must count as cached");
}

void StepContinuedMessageIsNotSplit() {
    UseOpening();
    StepModel model;
    g_encoded_tokens = Concat({kHeader, kOpening});
    g_samples = {40};
    g_sample_index = 0;
    auto meta = Meta();
    meta.continue_final_message = true;
    auto input = TextInput();
    Require(model.insert(meta, input), "insert failed");
    const FakeEngine& engine = model.engine();
    Require(engine.prefills.size() == 1 && engine.prefills[0] == g_encoded_tokens,
            "a continued message must run whole");
    Require(engine.saved == static_cast<int>(g_encoded_tokens.size()), "the checkpoint must sit at its end");
}

void LagunaNeitherCheckpointsNorRestores() {
    UseOpening();
    LagunaModel model;
    model.engine().rewinds = true;
    Turn(model, Concat({kHeader, kOpening}));
    Require(model.engine().prefills.size() == 1, "Laguna must run its prompt in one piece");
    Turn(model, Concat({kHeader, kNextTurn, kOpening}), false);
    const FakeEngine& engine = model.engine();
    Require(!engine.saved && engine.restores == 0, "Laguna must neither checkpoint nor restore");
    Require(engine.prefills.back() == Concat({kHeader, kNextTurn, kOpening}),
            "a history Laguna did not run must run whole");
}

}  // namespace

Tokenizer::Tokenizer(const std::string&) { is_doubled_encoded = false; }
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string& text) {
    const auto it = g_encodings.find(text);
    return it != g_encodings.end() ? it->second : g_encoded_tokens;
}
std::string Tokenizer::decode(const std::vector<int>&) { return "decoded"; }
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
    return kEos;
}
int Sampler::sample(const float*, const std::vector<uint32_t>*, sample_logprobs_t*) {
    if (g_sample_index < g_samples.size()) return g_samples[g_sample_index++];
    return kEos;
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
    run("Step checkpoints at the reply header", StepCheckpointsAtTheReplyHeader);
    run("Step's next turn rewinds to the reply header", StepNextTurnRewindsToTheReplyHeader);
    run("Step's next turn reports a replay", StepNextTurnReportsAReplay);
    run("a Step retry runs only the opening", StepRetryRunsOnlyTheOpening);
    run("a continued Step message is not split", StepContinuedMessageIsNotSplit);
    run("Laguna neither checkpoints nor restores", LagunaNeitherCheckpointsNorRestores);
    if (failures == 0) std::cout << "TemplateChat frontend regression tests passed\n";
    return failures == 0 ? 0 : 1;
}
