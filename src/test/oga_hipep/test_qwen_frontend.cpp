/// \file test_qwen_frontend.cpp
/// \brief fake-backend regression coverage for the Qwen3.5VL OGA frontend

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_qwen3_5vl.hpp>
#include <tensor_utils/safe_tensors.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
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
std::vector<int> g_encoded_tokens;
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
        executed.push_back(token);
        return buffer<bf16>(1);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void*) override {
        if (position + static_cast<int>(tokens.size()) > static_cast<int>(max_length)) {
            throw std::runtime_error("fake OGA engine prefill limit exceeded");
        }
        position += static_cast<int>(tokens.size());
        executed.insert(executed.end(), tokens.begin(), tokens.end());
        return buffer<bf16>(1);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override { position = 0; executed.clear(); forwarded.clear(); }
    buffer<bf16> get_k_cache(int, int) override { return buffer<bf16>(1); }
    buffer<bf16> get_v_cache(int, int) override { return buffer<bf16>(1); }
    int get_current_context_length() override { return position; }
    int checkpoint() override { return position; }
    int restore() override { return position; }

    std::uint32_t max_length;
    int position{};
    std::vector<int> executed;
    std::vector<int> forwarded;
};

class FakeOgaBackend final : public flm::backend::ModelBackend {
public:
    explicit FakeOgaBackend(std::uint32_t max_length)
        : engine_(std::make_unique<FakeEngine>(max_length)) {}

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kOgaHipepBackendId; }
    std::uint32_t max_decode_length() const override { return engine_->max_length - 1; }
    bool forwards_past_eos() const override { return false; }
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
        eos_token_ids = {kEos};
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

private:
    FakeEngine* engine_{};
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Reset(const std::vector<int>& encoded, const std::vector<int>& samples) {
    g_encoded_tokens = encoded;
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

}  // namespace

Tokenizer::Tokenizer(const std::string&) { is_doubled_encoded = false; }
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string&) { return g_encoded_tokens; }
std::string Tokenizer::decode(const std::vector<int>&) { return "decoded"; }
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
    throw std::logic_error("text-only regression test unexpectedly loaded an image");
}
qwen3_5vl_image_t Qwen3_5VL::load_image_base64(const std::string&) {
    throw std::logic_error("text-only regression test unexpectedly decoded an image");
}
void Qwen3_5VL::preprocess_image(qwen3_5vl_image_t&, std::vector<bf16>&) {
    throw std::logic_error("text-only regression test unexpectedly preprocessed an image");
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
    if (failures == 0) std::cout << "Qwen frontend regression tests passed\n";
    return failures == 0 ? 0 : 1;
}
