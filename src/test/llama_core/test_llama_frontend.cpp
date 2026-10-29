/// \file test_llama_frontend.cpp
/// \brief the Llama3 frontend over a stub of the core backend: the budget check,
///        the 500 that clears the conversation, the poisoned engine, and the
///        stop ids the backend states
#include "test_support.hpp"

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_llama3.hpp>
#include <models/llama/core/dev04/llama_core_backend.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/// \brief the builtin set, emptied
/// \note The real one lives in builtin_backends.cpp and names every engine, and
///       with them the prebuilt libraries this suite deliberately does not link.
///       Each test registers the backend it needs instead.
namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

/// \brief what the pinned config.json and tokenizer_config.json list as eos_token_id
const std::vector<int> kPackageEosIds = {128001, 128008, 128009};
/// \brief what llama_core_backend.cpp states
const std::vector<int> kCoreEosIds = {128009, 128008, 128001};
constexpr int kBosId = 128000;

std::vector<int> g_encoded_tokens;
std::vector<int> g_samples;
std::size_t g_sample_index{};

class FakeEngine final : public causal_lm {
public:
    explicit FakeEngine(std::uint32_t limit) : max_length(limit) {}

    buffer<bf16> forward(int token) override {
        ++forward_calls;
        forwarded.push_back(token);
        if (fail_forward) Fail();
        ++position;
        return buffer<bf16>(1);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void*) override {
        ++prefill_calls;
        prefill_sizes.push_back(static_cast<int>(tokens.size()));
        if (fail_prefill) Fail();
        position += static_cast<int>(tokens.size());
        return buffer<bf16>(1);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override {
        if (poisoned_state) throw std::runtime_error("poisoned");
        ++clear_calls;
        position = 0;
    }
    buffer<bf16> get_k_cache(int, int) override { return buffer<bf16>(1); }
    buffer<bf16> get_v_cache(int, int) override { return buffer<bf16>(1); }
    int get_current_context_length() override { return position; }
    int checkpoint() override { return position; }
    int restore() override { return position; }

    std::uint32_t max_length;
    int position{};
    int prefill_calls{};
    int forward_calls{};
    int clear_calls{};
    std::vector<int> prefill_sizes;
    std::vector<int> forwarded;
    bool fail_prefill{};
    bool fail_forward{};
    /// \brief whether a failure leaves the engine needing a reload, as a
    ///        corelib failure after submit does
    bool poison_on_failure{true};
    bool poisoned_state{};

private:
    void Fail() {
        if (poison_on_failure) poisoned_state = true;
        throw std::runtime_error("submitted inference failed");
    }
};

struct FactoryState {
    int calls{};
    FakeEngine* engine{};
} g_factory;

/// \brief the core backend's policy around FakeEngine
/// \note Mirrors CorelibBackend in corelib_model_backend.cpp: it decodes up to
///       one short of its context, declines the forward after an EOS, aligns
///       prefill chunks, and states the stop ids llama_core_backend.cpp gives it.
class StubCoreBackend final : public flm::backend::ModelBackend {
public:
    explicit StubCoreBackend(std::uint32_t context_length)
        : context_length_(context_length),
          engine_(std::make_unique<FakeEngine>(context_length)) {
        g_factory.engine = engine_.get();
    }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kCoreBackendId; }
    std::uint32_t max_decode_length() const override { return context_length_ - 1; }
    bool supports_preemption() const override { return false; }
    bool forwards_past_eos() const override { return false; }
    bool aligns_prefill_chunks() const override { return true; }
    bool poisoned() const noexcept override { return engine_->poisoned_state; }
    std::optional<std::vector<int>> forced_eos_ids() const override { return kCoreEosIds; }

private:
    std::uint32_t context_length_;
    std::unique_ptr<FakeEngine> engine_;
};

std::unique_ptr<flm::backend::ModelBackend> MakeCoreStub(
    const flm::backend::BackendContext& context) {
    ++g_factory.calls;
    return std::make_unique<StubCoreBackend>(context.context_length);
}

/// \brief a Llama 3 package directory in the shape the catalog pins
/// \param eos_token_id what tokenizer_config.json and config.json list
/// \note No tokenizer.json or GGUF: the tokenizer and the backend are stubs and
///       open neither.
class TempPackage final {
public:
    explicit TempPackage(std::vector<int> eos_token_id = kPackageEosIds) {
        static std::uint64_t serial{};
        path_ = std::filesystem::temp_directory_path() /
            ("flm-llama-frontend-" + std::to_string(++serial));
        std::filesystem::create_directories(path_);
        Write(path_ / "config.json",
              {{"model_type", "llama"}, {"vocab_size", 128256},
               {"bos_token_id", kBosId}, {"eos_token_id", eos_token_id}});
        Write(path_ / "tokenizer_config.json",
              {{"bos_token", "<|begin_of_text|>"}, {"bos_token_id", kBosId},
               {"eos_token", "<|eot_id|>"}, {"eos_token_id", eos_token_id},
               {"chat_template",
                "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}"}});
    }
    ~TempPackage() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    static void Write(const std::filesystem::path& path, const nlohmann::json& value) {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write package fixture");
        out << value.dump();
    }
    std::filesystem::path path_;
};

nlohmann::ordered_json ModelInfo() {
    return {{"default_context_length", 4096},
            {"details", {{"family", "llama3"}}}};
}

chat_meta_info_t Meta() {
    chat_meta_info_t value;
    value.max_prefill_len = 64;
    return value;
}

lm_uniform_input_t Input(std::optional<int> budget = std::nullopt) {
    lm_uniform_input_t value;
    value.prompt = "prompt";
    value.requested_max_new_tokens = budget;
    return value;
}

template <class F>
void ExpectRequestError(F&& action, int code, bool cleared, std::string_view text) {
    try { action(); }
    catch (const ModelRequestError& error) {
        TEST_REQUIRE(error.http_code() == code);
        TEST_REQUIRE(error.session_cleared() == cleared);
        RequireContains(error.what(), text);
        return;
    }
    throw std::runtime_error("expected ModelRequestError");
}

}  // namespace

Tokenizer::Tokenizer(const std::string&) { is_doubled_encoded = false; }
Tokenizer::~Tokenizer() = default;
std::vector<int> Tokenizer::encode(const std::string&) { return g_encoded_tokens; }
std::string Tokenizer::decode(const std::vector<int>&) { return "decoded"; }
std::string Tokenizer::run_time_decoder(int token) { return "t" + std::to_string(token); }
SafeTensors::~SafeTensors() = default;

Sampler::Sampler(int features, sampler_config& config)
    : in_features(features), rep_penalty(config.rep_penalty),
      freq_penalty(config.freq_penalty), pre_penalty(config.pre_penalty),
      top_k(config.top_k), top_p(config.top_p), min_p(config.min_p),
      temperature(config.temperature), total_tokens(0),
      freq_penalty_window(config.freq_penalty_window),
      rep_penalty_window(config.rep_penalty_window),
      repeat_last_n(config.repeat_last_n),
      use_optimized_sampling(config.use_optimized_sampling) {
    logits.resize(1); counters.resize(1); token_positions.resize(1, -1);
}
void Sampler::reset_penalties() {}
void Sampler::set_seed(uint64_t) {}
void Sampler::set_repeat_last_n(size_t n) { repeat_last_n = n; }
int Sampler::sample(buffer<bf16>&, const std::vector<uint32_t>*, sample_logprobs_t*) {
    if (g_sample_index < g_samples.size()) return g_samples[g_sample_index++];
    return 7;
}

// No engine library is linked; a null table leaves the prefix cache off.
extern "C" const flm_state_api* llama_npu_state_api(void*) { return nullptr; }

namespace {

/// \brief point the llama3 family's core backend at the stub for one test
/// \note replace_backend, not register_backend: the registry is process-wide
///       and every test re-arms it.
struct FactoryScope {
    FactoryScope() {
        g_factory = {};
        g_encoded_tokens.clear();
        g_samples.clear();
        g_sample_index = 0;
        flm::backend::BackendRegistry::instance().replace_backend(
            "llama3", flm::backend::kCoreBackendId, MakeCoreStub,
            flm::llama::core_traits());
    }
};

std::unique_ptr<Llama3> ReadyCore(const TempPackage& package) {
    auto model = std::make_unique<Llama3>(nullptr);
    model->load_model(package.path().string(), ModelInfo(), -1, false,
                      flm::backend::kCoreBackendId);
    return model;
}

void TestRenderedPromptPlusExplicitBudgetMayEqual4095() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens.assign(4000, 1);
    auto meta = Meta(); auto input = Input(95);
    TEST_REQUIRE(model->insert(meta, input));
}

void TestRenderedPromptPlusExplicitBudgetAbove4095Is400() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens.assign(4000, 1);
    auto meta = Meta(); auto input = Input(96);
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 400, false, "4095");
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
}

void TestAFailedPrefillReturns500AndClearsTheConversation() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1, 2}; g_samples = {11, 12};
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    (void)model->generate(meta, 1, output);
    TEST_REQUIRE(!model->get_history().second.empty());

    g_factory.engine->poison_on_failure = false;
    g_factory.engine->fail_prefill = true;
    g_encoded_tokens = {1, 2, 11, 3};
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 500, true,
                       "the current conversation was cleared (submitted inference failed)");
    TEST_REQUIRE(model->get_current_context_length() == 0);
    TEST_REQUIRE(model->get_history().second.empty());
    TEST_REQUIRE(g_factory.engine->position == 0);

    // Nothing is left to match against, so the next turn prefills all of it.
    g_factory.engine->fail_prefill = false;
    g_factory.engine->prefill_sizes.clear();
    auto next = Meta();
    TEST_REQUIRE(model->insert(next, input));
    TEST_REQUIRE((g_factory.engine->prefill_sizes == std::vector<int>{4}));
}

void TestAFailedDecodeReturns500AndLeavesTheModelPoisoned() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1}; g_samples = {11};
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true,
                       "unload/reload is required because the model is poisoned "
                       "(submitted inference failed)");
    TEST_REQUIRE(model->get_current_context_length() == 0);
    TEST_REQUIRE(model->get_history().second.empty());
}

void TestPoisonedModelReturns500UntilReload() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1}; g_samples = {11};
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");

    // Refused before the engine is touched again.
    const int prefills = g_factory.engine->prefill_calls;
    const int forwards = g_factory.engine->forward_calls;
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 500, true, "unload/reload");
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    TEST_REQUIRE(g_factory.engine->prefill_calls == prefills);
    TEST_REQUIRE(g_factory.engine->forward_calls == forwards);

    auto reloaded = ReadyCore(package);
    TEST_REQUIRE(reloaded->insert(meta, input));
}

void TestGenerateWithPromptGetsTheSameFailureHandlingAsGenerate() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1}; g_samples = {11};
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate_with_prompt(meta, input, 3, output); },
                       500, true, "unload/reload");
    TEST_REQUIRE(model->get_current_context_length() == 0);
    TEST_REQUIRE(model->get_history().second.empty());
}

void TestEveryStopIdTheBackendStatesEndsTheTurn() {
    // The package lists only <|eot_id|>, so <|eom_id|> and <|end_of_text|> can
    // only end the turn because the backend states them.
    for (const int eos : kCoreEosIds) {
        TempPackage package({128009}); FactoryScope scope; auto model = ReadyCore(package);
        g_encoded_tokens = {1, 2}; g_samples = {11, eos};
        auto meta = Meta(); auto input = Input(); std::ostringstream output;
        TEST_REQUIRE(model->insert(meta, input));
        const auto text = model->generate(meta, 10, output);
        TEST_REQUIRE(meta.stop_reason == EOT_DETECTED);
        TEST_REQUIRE(text == "t11");
        // The core backend declines the forward after an EOS.
        TEST_REQUIRE((g_factory.engine->forwarded == std::vector<int>{11}));
    }
}

}  // namespace

int main() {
#define RUN(name) RunTest(&name, #name)
    RUN(TestRenderedPromptPlusExplicitBudgetMayEqual4095);
    RUN(TestRenderedPromptPlusExplicitBudgetAbove4095Is400);
    RUN(TestAFailedPrefillReturns500AndClearsTheConversation);
    RUN(TestAFailedDecodeReturns500AndLeavesTheModelPoisoned);
    RUN(TestPoisonedModelReturns500UntilReload);
    RUN(TestGenerateWithPromptGetsTheSameFailureHandlingAsGenerate);
    RUN(TestEveryStopIdTheBackendStatesEndsTheTurn);
#undef RUN
    std::cout << "test_llama_frontend: PASS\n";
    return 0;
}
