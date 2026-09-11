#include "test_support.hpp"
#include "gguf_fixture.hpp"
#include "utils/file_access.hpp"

#include <AutoModel/modeling_phi4.hpp>
#include "server.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int> g_encoded_tokens;
std::vector<int> g_samples;
std::vector<std::filesystem::path> g_opened_paths;
std::size_t g_sample_index{};

class FakeEngine final : public causal_lm {
public:
    explicit FakeEngine(std::uint32_t limit) : max_length(limit) {}

    buffer<bf16> forward(int token) override {
        ++forward_calls;
        forwarded.push_back(token);
        if (fail_forward) {
            poisoned_state = true;
            throw std::runtime_error("submitted inference failed");
        }
        ++position;
        return buffer<bf16>(1);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void*) override {
        ++prefill_calls;
        if (fail_prefill) {
            poisoned_state = true;
            throw std::runtime_error("submitted inference failed");
        }
        position += static_cast<int>(tokens.size());
        return buffer<bf16>(1);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override {
        if (poisoned_state) throw std::runtime_error("poisoned");
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
    bool fail_prefill{};
    bool fail_forward{};
    bool poisoned_state{};
    std::vector<int> forwarded;
};

struct FactoryState {
    int legacy_calls{};
    int aie4_calls{};
    bool throw_for_aie4{};
    FakeEngine* engine{};
} g_factory;

class TempPackage final {
public:
    explicit TempPackage(bool valid = true) {
        static std::uint64_t serial{};
        path_ = std::filesystem::temp_directory_path() /
            ("flm-task4-" + std::to_string(++serial));
        std::filesystem::create_directories(path_);
        Write(path_ / "config.json", gguf_fixture::ValidConfig());
        Write(path_ / "tokenizer.json", gguf_fixture::ValidTokenizer());
        auto tokenizer_config = gguf_fixture::ValidTokenizerConfig();
        tokenizer_config["eos_token"] = "<legacy-eos>";
        tokenizer_config["eos_token_id"] = nlohmann::json::array({200020, 199999});
        Write(path_ / "tokenizer_config.json", tokenizer_config);
        auto gguf = gguf_fixture::Builder().AddFullContractTensors(false).Write("frontend");
        std::filesystem::rename(gguf.path, path_ / "Phi-4-mini-instruct.Q8_0.gguf");
        if (!valid) {
            auto config = gguf_fixture::ValidConfig();
            config["hidden_size"] = 1;
            Write(path_ / "config.json", config);
        }
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

nlohmann::ordered_json ModelInfo(std::optional<nlohmann::ordered_json> backend = std::nullopt) {
    nlohmann::ordered_json details = {{"family", "phi4"}};
    if (backend) details["execution_backend"] = *backend;
    return {{"default_context_length", 4096}, {"details", details}};
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

} // namespace

Tokenizer::Tokenizer(const std::string& model_path) {
    flm::file_access::ObserveOpen(
        std::filesystem::path(model_path) / "tokenizer.json");
    is_doubled_encoded = false;
}
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
int Sampler::sample(buffer<bf16>&) {
    if (g_sample_index < g_samples.size()) return g_samples[g_sample_index++];
    return 7;
}

namespace utils {
std::string get_executable_directory() { return std::filesystem::current_path().string(); }
}

namespace flm::phi4::testing {
class Phi4FrontendTestAccess final {
public:
    static void InstallFactory() {
        g_factory = {};
        g_opened_paths.clear();
        flm::file_access::SetOpenObserver([](const auto& path) {
            g_opened_paths.push_back(path);
        });
        Phi4::engine_factory_for_testing_ =
            [](bool aie4, const LM_Config&, npu_xclbin_manager*,
               const std::filesystem::path&, std::uint32_t limit) {
                if (aie4) {
                    ++g_factory.aie4_calls;
                    if (g_factory.throw_for_aie4) throw std::runtime_error("missing corelib");
                } else {
                    ++g_factory.legacy_calls;
                }
                auto result = std::make_unique<FakeEngine>(limit);
                g_factory.engine = result.get();
                return std::unique_ptr<causal_lm>(std::move(result));
            };
        Phi4::engine_poisoned_for_testing_ = [](const causal_lm* engine) {
            return static_cast<const FakeEngine*>(engine)->poisoned_state;
        };
    }
    static void RemoveFactory() {
        Phi4::engine_factory_for_testing_ = {};
        Phi4::engine_poisoned_for_testing_ = {};
        flm::file_access::SetOpenObserver({});
    }
    static bool HasLegacyNpu(const Phi4& model) { return model.npu != nullptr; }
    static const std::string& EosToken(const Phi4& model) { return model.eos_token; }
    static const std::vector<int>& EosTokenIds(const Phi4& model) { return model.eos_token_ids; }
    static bool HasBosToken(const Phi4& model) { return model.has_bos_token; }
};
} // namespace flm::phi4::testing

namespace {
using flm::phi4::testing::Phi4FrontendTestAccess;

struct FactoryScope {
    FactoryScope() { Phi4FrontendTestAccess::InstallFactory(); }
    ~FactoryScope() { Phi4FrontendTestAccess::RemoveFactory(); }
};

std::unique_ptr<Phi4> Load(const TempPackage& package,
                           nlohmann::ordered_json info,
                           int context = -1,
                           bool preemption = false,
                           flm_rt::device* device = reinterpret_cast<flm_rt::device*>(1)) {
    auto model = std::make_unique<Phi4>(device);
    model->load_model(package.path().string(), std::move(info), context, preemption);
    return model;
}

void TestAbsentBackendStillBuildsQ4nxPhi4Npu() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo());
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(g_factory.aie4_calls == 0);
    TEST_REQUIRE(!model->uses_corelib_aie4());
    TEST_REQUIRE(Phi4FrontendTestAccess::HasLegacyNpu(*model));
}

void TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo());
    g_encoded_tokens = {1};
    g_samples = {7};
    g_sample_index = 0;
    auto meta = Meta();
    auto input = Input(1);
    std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    (void)model->generate(meta, 1, output);
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(g_factory.aie4_calls == 0);
}

void TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_aie4 = true;
    auto model = Load(package, ModelInfo());
    g_encoded_tokens = {1};
    auto meta = Meta();
    auto input = Input(1);
    TEST_REQUIRE(model->insert(meta, input));
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(g_factory.aie4_calls == 0);
}

void TestCorelibAie4GgufBuildsOnlyTheCorelibEngine() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
    TEST_REQUIRE(g_factory.aie4_calls == 1);
    TEST_REQUIRE(model->uses_corelib_aie4());
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasLegacyNpu(*model));
}

void TestNoManifestOnnxConvertedWeightOrCachePathIsOpened() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr);
    TEST_REQUIRE(model->uses_corelib_aie4());
    std::vector<std::string> names;
    for (const auto& path : g_opened_paths) {
        const auto text = path.generic_string();
        TEST_REQUIRE(text.find("manifest") == std::string::npos);
        TEST_REQUIRE(text.find("onnx") == std::string::npos);
        TEST_REQUIRE(text.find("converted") == std::string::npos);
        TEST_REQUIRE(text.find("cache") == std::string::npos);
        names.push_back(path.filename().string());
    }
    std::sort(names.begin(), names.end());
    TEST_REQUIRE(names == std::vector<std::string>({
        "Phi-4-mini-instruct.Q8_0.gguf", "config.json", "config.json",
        "tokenizer.json", "tokenizer.json", "tokenizer_config.json"}));
}

void TestUnknownAndNonStringBackendAreErrors() {
    TempPackage package;
    FactoryScope scope;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo("other")); }), "Unknown");
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo(7)); }), "string");
    TEST_REQUIRE(g_factory.legacy_calls == 0 && g_factory.aie4_calls == 0);
}

void TestFeatureOffRejectsAie4TagWithoutIncludingCorelibHeaders() {
#if !defined(FLM_ENABLE_CORELIB_AIE4)
    TempPackage package;
    FactoryScope scope;
    RequireContains(RequireThrows([&] {
        (void)Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr);
    }), "This binary was built without Phi-4 AIE4 corelib support");
    TEST_REQUIRE(g_factory.legacy_calls == 0 && g_factory.aie4_calls == 0);
#endif
}

void TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation() {
    TempPackage package(false);
    FactoryScope scope;
    RequireThrows([&] { (void)Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr); });
    TEST_REQUIRE(g_factory.aie4_calls == 0);
}

void TestMissingCorelibFailsOnlyWhenAie4ModelLoads() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_aie4 = true;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr); }), "missing corelib");
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestAie4SelectionWithMissingDllFailsWithoutChangingBackend() {
    TempPackage package;
    FactoryScope scope;
    Phi4 model(nullptr);
    g_factory.throw_for_aie4 = true;
    RequireContains(RequireThrows([&] {
        model.load_model(package.path().string(), ModelInfo("corelib_aie4_gguf"));
    }), "missing corelib");
    TEST_REQUIRE(!model.uses_corelib_aie4());
    TEST_REQUIRE(g_factory.aie4_calls == 1);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestAie4SelectionCannotReachQ4nxPhi4NpuOrCpuFallback() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_aie4 = true;
    (void)RequireThrows([&] {
        (void)Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr);
    });
    TEST_REQUIRE(g_factory.aie4_calls == 1);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestOrdinaryModelLoadsAfterAnAie4RuntimeLoadFailure() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_aie4 = true;
    RequireThrows([&] { (void)Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr); });
    g_factory.throw_for_aie4 = false;
    auto ordinary = Load(package, ModelInfo());
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(!ordinary->uses_corelib_aie4());
}

void TestPreemptionIsRejectedForTheAie4Route() {
    TempPackage package;
    FactoryScope scope;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo("corelib_aie4_gguf"), -1, true, nullptr); }), "preemption");
    TEST_REQUIRE(g_factory.aie4_calls == 0);
}

std::unique_ptr<Phi4> ReadyAie4(const TempPackage& package) {
    return Load(package, ModelInfo("corelib_aie4_gguf"), -1, false, nullptr);
}

void TestRenderedPromptPlusExplicitBudgetMayEqual4095() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens.assign(4000, 1);
    auto meta = Meta(); auto input = Input(95);
    TEST_REQUIRE(model->insert(meta, input));
}

void TestRenderedPromptPlusExplicitBudgetAbove4095Is400() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens.assign(4000, 1);
    auto meta = Meta(); auto input = Input(96);
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 400, false, "4095");
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
}

void TestOmittedZeroAndNegativeSentinelBudgetsCapAtRemainingWindow() {
    for (const auto requested : {std::optional<int>{}, std::optional<int>{0}, std::optional<int>{-1}}) {
        TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
        g_encoded_tokens.assign(4093, 1); g_samples = {11, 12, 13}; g_sample_index = 0;
        auto meta = Meta(); auto input = Input(requested); std::ostringstream output;
        TEST_REQUIRE(model->insert(meta, input));
        (void)model->generate(meta, 4096, output);
        TEST_REQUIRE(meta.generated_tokens == 2);
        TEST_REQUIRE(meta.stop_reason == MAX_LENGTH_REACHED);
    }

    // /api/chat uses generate_with_prompt and retains 4096 only as the legacy
    // loop default; omission must not become an explicit AIE4 budget.
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens.assign(4093, 1); g_samples = {11, 12, 13}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    (void)model->generate_with_prompt(meta, input, 4096, output);
    TEST_REQUIRE(meta.generated_tokens == 2);
}

void TestCancellationBeforePrefillSubmitsNothing() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens = {1, 2}; auto meta = Meta(); auto input = Input();
    int checks = 0;
    TEST_REQUIRE(!model->insert(meta, input, [&] { return ++checks >= 2; }));
    TEST_REQUIRE(checks >= 2);
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
}

void TestCancellationBetweenDecodeStepsStopsWithCancelReason() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens = {1}; g_samples = {11, 12}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    int checks = 0;
    (void)model->generate(meta, 10, output, [&] { return checks++ == 2; });
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
    TEST_REQUIRE(g_factory.engine->forward_calls == 1);
}

void TestCancellationReturnsOnlyAfterSynchronize() {
    // Fake calls are synchronous by construction: observing one completed call
    // before cancellation proves no work remains outstanding at return.
    TestCancellationBetweenDecodeStepsStopsWithCancelReason();
}

void TestNonStreamingChatGenerateWithPromptForwardsCancellation() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens = {1, 2}; auto meta = Meta(); auto input = Input();
    std::ostringstream output;
    int checks = 0;
    AutoModel* endpoint_model = model.get();
    const auto response = endpoint_model->generate_with_prompt(
        meta, input, 4096, output, [&] { return ++checks >= 2; });
    TEST_REQUIRE(response.empty());
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
}

void TestLegacyTokenizerContractIsPreserved() {
    TempPackage package; FactoryScope scope; auto legacy = Load(package, ModelInfo());
    // Main's legacy Phi-4 frontend intentionally did not pass the textual EOS
    // token into minja and retained an empty eos_token string.
    TEST_REQUIRE(Phi4FrontendTestAccess::EosToken(*legacy).empty());
    TEST_REQUIRE(Phi4FrontendTestAccess::EosTokenIds(*legacy) ==
                 std::vector<int>({200020, 199999}));

    auto aie4 = ReadyAie4(package);
    TEST_REQUIRE(Phi4FrontendTestAccess::EosTokenIds(*aie4) ==
                 std::vector<int>({200020, 199999}));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasBosToken(*aie4));
}

void TestSamePathBackendSwitchForcesLegacyInitialization() {
    TempPackage package; FactoryScope scope;
    Phi4 model(reinterpret_cast<flm_rt::device*>(1));
    model.load_model(package.path().string(), ModelInfo("corelib_aie4_gguf"));
    TEST_REQUIRE(model.uses_corelib_aie4());
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasLegacyNpu(model));
    model.load_model(package.path().string(), ModelInfo());
    TEST_REQUIRE(!model.uses_corelib_aie4());
    TEST_REQUIRE(Phi4FrontendTestAccess::HasLegacyNpu(model));
    TEST_REQUIRE(g_factory.legacy_calls == 1);
}

void TestPostSubmitErrorReturns500ClearsConversationAndLeavesModelPoisoned() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens = {1}; g_samples = {11}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    TEST_REQUIRE(model->get_current_context_length() == 0);
}

void TestPoisonedModelReturns500UntilReload() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens = {1}; g_samples = {11}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 500, true, "unload/reload");
    auto reloaded = ReadyAie4(package);
    TEST_REQUIRE(reloaded->insert(meta, input));
}

void TestEosSelfTerminatesWithoutAnExtraDecode() {
    TempPackage package; FactoryScope scope; auto model = ReadyAie4(package);
    g_encoded_tokens = {1}; g_samples = {200020}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    (void)model->generate(meta, 10, output);
    TEST_REQUIRE(g_factory.engine->forward_calls == 0);
    TEST_REQUIRE(meta.stop_reason == EOT_DETECTED);
}

void TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics() {
    for (const auto raw : {std::optional<int>{}, std::optional<int>{0}, std::optional<int>{-2}, std::optional<int>{17}}) {
        const auto expected = raw && *raw > 0 ? raw : std::nullopt;
        for (int source = 0; source < 5; ++source)
            TEST_REQUIRE(normalize_requested_max_new_tokens(raw) == expected);
    }
}

void TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint() {
    TEST_REQUIRE(requires_npu_access("POST", "/v1/completions"));
    for (int path = 0; path < 5; ++path) {
        int releases = 0;
        {
            NPURequestCompletionGuard guard([&] { ++releases; });
            if (path == 0) guard.complete();
            else if (path == 1) { guard.complete(); guard.complete(); }
            else if (path == 2) { NPURequestCompletionGuard moved(std::move(guard)); }
            else if (path == 3) { try { throw std::runtime_error("model"); } catch (...) {} }
            else { try { throw 1; } catch (...) {} }
        }
        TEST_REQUIRE(releases == 1);
    }
}

void TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable() {
    TempPackage package;
    FactoryScope scope;
    auto model = ReadyAie4(package);
    NPURequestCoordinator coordinator(3);
    bool cancelled = false;
    bool capacity_failed = false;
    bool queued_request_ran = false;
    int completion_callbacks = 0;
    int accelerator_releases = 0;

    TEST_REQUIRE(coordinator.try_enqueue([&] {
        auto meta = Meta();
        auto input = Input(1);
        g_encoded_tokens = {1};
        cancelled = !model->insert(meta, input, [] { return true; });
    }));
    TEST_REQUIRE(coordinator.try_enqueue([&] {
        auto meta = Meta();
        auto input = Input(1);
        g_encoded_tokens.assign(4095, 1);
        try { (void)model->insert(meta, input); }
        catch (const ModelRequestError& error) {
            capacity_failed = error.http_code() == 400;
        }
    }));
    TEST_REQUIRE(coordinator.try_enqueue([&] {
        auto meta = Meta();
        auto input = Input(1);
        g_encoded_tokens = {1};
        queued_request_ran = model->insert(meta, input);
    }));
    TEST_REQUIRE(!coordinator.try_enqueue([] {}));

    std::function<void(std::function<void()>)> execute;
    const auto complete = [&] {
        ++completion_callbacks;
        coordinator.complete_current(execute, [&] { ++accelerator_releases; },
                                     std::chrono::milliseconds(0));
    };
    execute = [&](std::function<void()> task) {
        NPURequestCompletionGuard completion(complete);
        task();
        completion.complete();
        completion.complete();
    };
    {
        NPURequestCompletionGuard active_request_completion(complete);
        active_request_completion.complete();
        active_request_completion.complete();
    }

    TEST_REQUIRE(cancelled);
    TEST_REQUIRE(capacity_failed);
    TEST_REQUIRE(queued_request_ran);
    TEST_REQUIRE(coordinator.empty());
    TEST_REQUIRE(completion_callbacks == 4);
    TEST_REQUIRE(accelerator_releases == 1);
}

} // namespace

int main() {
#if defined(FLM_ENABLE_CORELIB_AIE4)
    RunTest(TestAbsentBackendStillBuildsQ4nxPhi4Npu, "TestAbsentBackendStillBuildsQ4nxPhi4Npu");
    RunTest(TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing, "TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing");
    RunTest(TestCorelibAie4GgufBuildsOnlyTheCorelibEngine, "TestCorelibAie4GgufBuildsOnlyTheCorelibEngine");
    RunTest(TestNoManifestOnnxConvertedWeightOrCachePathIsOpened, "TestNoManifestOnnxConvertedWeightOrCachePathIsOpened");
    RunTest(TestUnknownAndNonStringBackendAreErrors, "TestUnknownAndNonStringBackendAreErrors");
    RunTest(TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation, "TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation");
    RunTest(TestMissingCorelibFailsOnlyWhenAie4ModelLoads, "TestMissingCorelibFailsOnlyWhenAie4ModelLoads");
    RunTest(TestAie4SelectionWithMissingDllFailsWithoutChangingBackend, "TestAie4SelectionWithMissingDllFailsWithoutChangingBackend");
    RunTest(TestAie4SelectionCannotReachQ4nxPhi4NpuOrCpuFallback, "TestAie4SelectionCannotReachQ4nxPhi4NpuOrCpuFallback");
    RunTest(TestOrdinaryModelLoadsAfterAnAie4RuntimeLoadFailure, "TestOrdinaryModelLoadsAfterAnAie4RuntimeLoadFailure");
    RunTest(TestPreemptionIsRejectedForTheAie4Route, "TestPreemptionIsRejectedForTheAie4Route");
    RunTest(TestRenderedPromptPlusExplicitBudgetMayEqual4095, "TestRenderedPromptPlusExplicitBudgetMayEqual4095");
    RunTest(TestRenderedPromptPlusExplicitBudgetAbove4095Is400, "TestRenderedPromptPlusExplicitBudgetAbove4095Is400");
    RunTest(TestOmittedZeroAndNegativeSentinelBudgetsCapAtRemainingWindow, "TestOmittedZeroAndNegativeSentinelBudgetsCapAtRemainingWindow");
    RunTest(TestCancellationBeforePrefillSubmitsNothing, "TestCancellationBeforePrefillSubmitsNothing");
    RunTest(TestCancellationBetweenDecodeStepsStopsWithCancelReason, "TestCancellationBetweenDecodeStepsStopsWithCancelReason");
    RunTest(TestCancellationReturnsOnlyAfterSynchronize, "TestCancellationReturnsOnlyAfterSynchronize");
    RunTest(TestNonStreamingChatGenerateWithPromptForwardsCancellation, "TestNonStreamingChatGenerateWithPromptForwardsCancellation");
    RunTest(TestLegacyTokenizerContractIsPreserved, "TestLegacyTokenizerContractIsPreserved");
    RunTest(TestSamePathBackendSwitchForcesLegacyInitialization, "TestSamePathBackendSwitchForcesLegacyInitialization");
    RunTest(TestPostSubmitErrorReturns500ClearsConversationAndLeavesModelPoisoned, "TestPostSubmitErrorReturns500ClearsConversationAndLeavesModelPoisoned");
    RunTest(TestPoisonedModelReturns500UntilReload, "TestPoisonedModelReturns500UntilReload");
    RunTest(TestEosSelfTerminatesWithoutAnExtraDecode, "TestEosSelfTerminatesWithoutAnExtraDecode");
    RunTest(TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics, "TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics");
    RunTest(TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint, "TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint");
    RunTest(TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable, "TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable");
#else
    RunTest(TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib, "TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib");
    RunTest(TestFeatureOffRejectsAie4TagWithoutIncludingCorelibHeaders, "TestFeatureOffRejectsAie4TagWithoutIncludingCorelibHeaders");
#endif
    std::cout << "test_phi4_frontend: PASS\n";
}
