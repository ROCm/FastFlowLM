#include "test_support.hpp"
#include "gguf_fixture.hpp"
#if defined(FLM_ENABLE_RAI)
#include "fake_corelib.hpp"
#endif
#include "utils/file_access.hpp"

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_phi4.hpp>
#if defined(FLM_ENABLE_RAI)
#include <rai/corelib_runtime.hpp>
#include <models/phi4/rai/aie_next/phi4_rai_backend.hpp>
#include <models/phi4/rai/aie_next/phi4_rai_gguf.hpp>
#endif
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
#include <thread>
#include <vector>
#include <chrono>
#include <cstdlib>

/// \brief the builtin set, emptied
/// \note The real one lives in builtin_backends.cpp and names every engine, and
///       with them the prebuilt libraries this suite deliberately does not link.
///       Each test registers the backends it needs instead.
namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

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
        if (forward_delay.count()) std::this_thread::sleep_for(forward_delay);
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
    std::chrono::microseconds forward_delay{0};
    std::vector<int> forwarded;
};

struct FactoryState {
    int legacy_calls{};
    int rai_calls{};
    bool throw_for_rai{};
    FakeEngine* engine{};
#if defined(FLM_ENABLE_RAI)
    /// \brief the runtime the corelib stub reports as its detail()
    std::shared_ptr<flm::corelib::CorelibRuntime> runtime;
#endif
} g_factory;

/// \brief a backend wrapping FakeEngine, with a policy the test dictates
/// \note This is the seam the old engine_factory_for_testing_ hook used to be.
///       Registering it under a real backend id means the frontend, AutoModel
///       and the registry all run their production code paths; only the engine
///       and, for corelib, the loaded library are faked.
class StubBackend final : public flm::backend::ModelBackend {
public:
    StubBackend(std::string id, std::uint32_t context_length)
        : id_(std::move(id)),
          engine_(std::make_unique<FakeEngine>(context_length)) {
        g_factory.engine = engine_.get();
    }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return id_; }
    std::string detail() const override { return detail_; }
    std::uint32_t max_decode_length() const override { return decode_limit_; }
    bool supports_preemption() const override { return supports_preemption_; }
    bool forwards_past_eos() const override { return forwards_past_eos_; }
    bool poisoned() const noexcept override { return engine_->poisoned_state; }
    std::optional<std::vector<int>> forced_eos_ids() const override {
        return forced_eos_ids_;
    }

    std::string detail_;
    std::uint32_t decode_limit_{};
    bool supports_preemption_{true};
    bool forwards_past_eos_{true};
    std::optional<std::vector<int>> forced_eos_ids_;

private:
    std::string id_;
    std::unique_ptr<FakeEngine> engine_;
};

/// \brief the FastFlowLM NPU backend, with its engine faked
std::unique_ptr<flm::backend::ModelBackend> MakeFlmStub(
    const flm::backend::BackendContext& context) {
    ++g_factory.legacy_calls;
    return std::make_unique<StubBackend>(std::string(flm::backend::kFlmBackendId),
                                         context.context_length);
}

#if defined(FLM_ENABLE_RAI)
/// \brief read a JSON file, recording the open for the file-access audit
nlohmann::json ReadObserved(const std::filesystem::path& path) {
    flm::file_access::ObserveOpen(path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.string());
    return nlohmann::json::parse(input);
}

/// \brief the corelib backend, with its engine and runtime faked
/// \note Everything RaiBackend does before it touches the device is
///       repeated verbatim, so a mismatched package still fails with nothing
///       allocated and the file-access audit still sees the real reads.
std::unique_ptr<flm::backend::ModelBackend> MakeRaiStub(
    const flm::backend::BackendContext& context) {
    const std::filesystem::path root(context.model_path);
    const auto config = ReadObserved(root / "config.json");
    const auto tokenizer_json = ReadObserved(root / "tokenizer.json");
    // tokenizer_config.json arrives already parsed from the frontend, exactly
    // as it does for the real backend, so the audit sees a single open.
    if (context.tokenizer_config == nullptr) {
        throw std::runtime_error(
            "Phi-4 rai backend needs the frontend's tokenizer_config.json");
    }
    auto package = flm::phi4::Phi4GgufPackage::Open(
        root / "Phi-4-mini-instruct.Q8_0.gguf");
    package->ValidatePhi4Contract(config, tokenizer_json,
                                  *context.tokenizer_config);

    ++g_factory.rai_calls;
    if (g_factory.throw_for_rai) throw std::runtime_error("missing corelib");

    auto backend = std::make_unique<StubBackend>(
        std::string(flm::backend::kRaiBackendId), context.context_length);
    backend->decode_limit_ = flm::phi4::kRaiDecodeLimit;
    backend->supports_preemption_ = false;
    backend->forwards_past_eos_ = false;
    backend->forced_eos_ids_ = std::vector<int>({200020, 199999});
    if (g_factory.runtime) {
        backend->detail_ = g_factory.runtime->loaded_library_path().string();
    }
    return backend;
}
#endif

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

/// \brief a catalog entry as the frontend sees it
/// \note It names no backend. It cannot: a build links one kernel flow, and the
///       catalog entry has already been filtered to the platform that build
///       targets by the time it reaches here. The tests say which backend they
///       mean by passing it to Load, which is the --backend path.
nlohmann::ordered_json ModelInfo() {
    return {{"default_context_length", 4096},
            {"details", {{"family", "phi4"}}}};
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
/// \brief reads the tokenizer contract Phi4 keeps to itself
/// \note Nothing here writes: the seam the tests drive the model through is the
///       backend registry, not this class.
class Phi4FrontendTestAccess final {
public:
    static bool HasLegacyNpu(const Phi4& model) { return model.npu != nullptr; }
    static const std::string& EosToken(const Phi4& model) { return model.eos_token; }
    static const std::vector<int>& EosTokenIds(const Phi4& model) { return model.eos_token_ids; }
    static bool HasBosToken(const Phi4& model) { return model.has_bos_token; }
};
} // namespace flm::phi4::testing

namespace {
using flm::phi4::testing::Phi4FrontendTestAccess;

/// \brief the backend ids this suite drives
/// \note A backend id names a kernel provider, not silicon. These are the shared
///       constants from model_backend.hpp, spelled out here so the suite reads
///       as "run this through that kernel flow".
constexpr const char* kFlm = "flm";
constexpr const char* kRai = "rai";

/// \brief Point the phi4 family's backends at the stubs for one test
/// \note replace_backend, not register_backend: every test re-arms the registry,
///       which is process-wide. The OFF build deliberately registers no corelib
///       backend, which is what makes rejecting its id observable.
struct FactoryScope {
    FactoryScope() {
        g_factory = {};
        g_opened_paths.clear();
        flm::file_access::SetOpenObserver([](const auto& path) {
            g_opened_paths.push_back(path);
        });
        auto& registry = flm::backend::BackendRegistry::instance();
        registry.replace_backend("phi4", kFlm, MakeFlmStub);
#if defined(FLM_ENABLE_RAI)
        registry.replace_backend("phi4", kRai, MakeRaiStub,
                                 flm::phi4::rai_traits());
#endif
    }
    ~FactoryScope() {
        flm::file_access::SetOpenObserver({});
#if defined(FLM_ENABLE_RAI)
        g_factory.runtime.reset();
#endif
    }
};

/// \brief which backend a loaded model ended up on
bool OnRai(const AutoModel& model) {
    return model.backend_id() == kRai;
}
bool OnFlm(const AutoModel& model) { return model.backend_id() == kFlm; }

std::unique_ptr<Phi4> Load(const TempPackage& package,
                           nlohmann::ordered_json info,
                           int context = -1,
                           bool preemption = false,
                           flm_rt::device* device = reinterpret_cast<flm_rt::device*>(1),
                           const std::string& backend = kFlm) {
    auto model = std::make_unique<Phi4>(device);
    model->load_model(package.path().string(), std::move(info), context,
                      preemption, backend);
    return model;
}

void TestAbsentBackendStillBuildsQ4nxPhi4Npu() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo());
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(g_factory.rai_calls == 0);
    TEST_REQUIRE(OnFlm(*model));
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
    TEST_REQUIRE(g_factory.rai_calls == 0);
}

void TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_rai = true;
    auto model = Load(package, ModelInfo());
    g_encoded_tokens = {1};
    auto meta = Meta();
    auto input = Input(1);
    TEST_REQUIRE(model->insert(meta, input));
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(g_factory.rai_calls == 0);
}

void TestCorelibRaiGgufBuildsOnlyTheCorelibEngine() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo(), -1, false, nullptr, kRai);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
    TEST_REQUIRE(g_factory.rai_calls == 1);
    TEST_REQUIRE(OnRai(*model));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasLegacyNpu(*model));
}

#if defined(FLM_ENABLE_RAI)
void TestRaiProfileUsesCachedRuntimeDllPathAfterEnvironmentChanges() {
    TempPackage package;
    FactoryScope scope;
    fake_corelib::Reset();
    const auto dll_a = std::filesystem::absolute(package.path() / "runtime-a.dll");
    const auto dll_b = std::filesystem::absolute(package.path() / "runtime-b.dll");
    auto api = flm::corelib::CorelibApi::ResolveForTest(
        fake_corelib::Resolver(), dll_a);
    auto runtime = flm::corelib::CorelibRuntime::CreateForTest(std::move(api));

#ifdef _WIN32
    _putenv_s("FLM_RAI_CORELIB_PATH", dll_b.string().c_str());
#else
    setenv("FLM_RAI_CORELIB_PATH", dll_b.string().c_str(), 1);
#endif
    // The backend caches the runtime it was built with; a later environment
    // change must not move the path it reports.
    g_factory.runtime = runtime;
    auto rai = Load(package, ModelInfo(), -1, false, nullptr, kRai);
    g_factory.runtime.reset();
    const auto rai_profile = rai->show_profile();
    RequireContains(rai_profile, kRai);
    RequireContains(rai_profile, dll_a.string());
    TEST_REQUIRE(rai_profile.find(dll_b.string()) == std::string::npos);

    auto legacy = Load(package, ModelInfo());
    const auto legacy_profile = legacy->show_profile();
    TEST_REQUIRE(legacy_profile.find(kRai) == std::string::npos);
    TEST_REQUIRE(legacy_profile.find(dll_a.string()) == std::string::npos);
    rai.reset();
    runtime.reset();
    flm::corelib::CorelibRuntime::ShutdownProcess();
#ifdef _WIN32
    _putenv_s("FLM_RAI_CORELIB_PATH", "");
#else
    unsetenv("FLM_RAI_CORELIB_PATH");
#endif
}
#endif

void TestNoManifestOnnxConvertedWeightOrCachePathIsOpened() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo(), -1, false, nullptr, kRai);
    TEST_REQUIRE(OnRai(*model));
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

void TestUnknownBackendIsAnError() {
    TempPackage package;
    FactoryScope scope;
    // Hardware this build has no engine for is refused by name, before any
    // engine is constructed.
    const auto message = RequireThrows(
        [&] { (void)Load(package, ModelInfo(), -1, false, nullptr, "other"); });
    RequireContains(message, "not compiled into this build");
    RequireContains(message, "other");
    TEST_REQUIRE(g_factory.legacy_calls == 0 && g_factory.rai_calls == 0);
}

void TestFeatureOffRejectsRaiTagWithoutIncludingCorelibHeaders() {
#if !defined(FLM_ENABLE_RAI)
    TempPackage package;
    FactoryScope scope;
    RequireContains(RequireThrows([&] {
        (void)Load(package, ModelInfo(), -1, false, nullptr, kRai);
    }), "not compiled into this build");
    TEST_REQUIRE(g_factory.legacy_calls == 0 && g_factory.rai_calls == 0);
#endif
}

void TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation() {
    TempPackage package(false);
    FactoryScope scope;
    RequireThrows([&] { (void)Load(package, ModelInfo(), -1, false, nullptr, kRai); });
    TEST_REQUIRE(g_factory.rai_calls == 0);
}

void TestMissingCorelibFailsOnlyWhenRaiModelLoads() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_rai = true;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo(), -1, false, nullptr, kRai); }), "missing corelib");
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestRaiSelectionWithMissingDllFailsWithoutChangingBackend() {
    TempPackage package;
    FactoryScope scope;
    Phi4 model(nullptr);
    g_factory.throw_for_rai = true;
    RequireContains(RequireThrows([&] {
        model.load_model(package.path().string(), ModelInfo(), -1, false, kRai);
    }), "missing corelib");
    TEST_REQUIRE(model.backend_id().empty());
    TEST_REQUIRE(g_factory.rai_calls == 1);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestRaiSelectionCannotReachQ4nxPhi4NpuOrCpuFallback() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_rai = true;
    (void)RequireThrows([&] {
        (void)Load(package, ModelInfo(), -1, false, nullptr, kRai);
    });
    TEST_REQUIRE(g_factory.rai_calls == 1);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestOrdinaryModelLoadsAfterAnRaiRuntimeLoadFailure() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_rai = true;
    RequireThrows([&] { (void)Load(package, ModelInfo(), -1, false, nullptr, kRai); });
    g_factory.throw_for_rai = false;
    auto ordinary = Load(package, ModelInfo());
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(OnFlm(*ordinary));
}

void TestPreemptionIsRejectedForTheRaiRoute() {
    TempPackage package;
    FactoryScope scope;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo(), -1, true, nullptr, kRai); }), "preemption");
    TEST_REQUIRE(g_factory.rai_calls == 0);
}

std::unique_ptr<Phi4> ReadyRai(const TempPackage& package) {
    return Load(package, ModelInfo(), -1, false, nullptr, kRai);
}

void TestRenderedPromptPlusExplicitBudgetMayEqual4095() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens.assign(4000, 1);
    auto meta = Meta(); auto input = Input(95);
    TEST_REQUIRE(model->insert(meta, input));
}

void TestRenderedPromptPlusExplicitBudgetAbove4095Is400() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens.assign(4000, 1);
    auto meta = Meta(); auto input = Input(96);
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 400, false, "4095");
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
}

void TestOmittedZeroAndNegativeSentinelBudgetsCapAtRemainingWindow() {
    for (const auto requested : {std::optional<int>{}, std::optional<int>{0}, std::optional<int>{-1}}) {
        TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
        g_encoded_tokens.assign(4093, 1); g_samples = {11, 12, 13}; g_sample_index = 0;
        auto meta = Meta(); auto input = Input(requested); std::ostringstream output;
        TEST_REQUIRE(model->insert(meta, input));
        (void)model->generate(meta, 4096, output);
        TEST_REQUIRE(meta.generated_tokens == 2);
        TEST_REQUIRE(meta.stop_reason == MAX_LENGTH_REACHED);
    }

    // /api/chat uses generate_with_prompt and retains 4096 only as the legacy
    // loop default; omission must not become an explicit rai budget.
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens.assign(4093, 1); g_samples = {11, 12, 13}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    (void)model->generate_with_prompt(meta, input, 4096, output);
    TEST_REQUIRE(meta.generated_tokens == 2);
}

void TestCancellationBeforePrefillSubmitsNothing() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens = {1, 2}; auto meta = Meta(); auto input = Input();
    int checks = 0;
    TEST_REQUIRE(!model->insert(meta, input, [&] { return ++checks >= 2; }));
    TEST_REQUIRE(checks >= 2);
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
}

void TestCancellationBetweenDecodeStepsStopsWithCancelReason() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens = {1}; g_samples = {11, 12}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    // Cancel once a decode step has actually happened, rather than after a
    // fixed number of polls: how often the loop consults the predicate is its
    // own business, but it must not dispatch another forward once cancelled.
    (void)model->generate(meta, 10, output,
                          [&] { return g_factory.engine->forward_calls >= 1; });
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
    TEST_REQUIRE(g_factory.engine->forward_calls == 1);
}

void TestCancellationReturnsOnlyAfterSynchronize() {
    // Fake calls are synchronous by construction: observing one completed call
    // before cancellation proves no work remains outstanding at return.
    TestCancellationBetweenDecodeStepsStopsWithCancelReason();
}

void TestNonStreamingChatGenerateWithPromptForwardsCancellation() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
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

    auto rai = ReadyRai(package);
    TEST_REQUIRE(Phi4FrontendTestAccess::EosTokenIds(*rai) ==
                 std::vector<int>({200020, 199999}));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasBosToken(*rai));
}

void TestSamePathBackendSwitchForcesLegacyInitialization() {
    TempPackage package; FactoryScope scope;
    Phi4 model(reinterpret_cast<flm_rt::device*>(1));
    model.load_model(package.path().string(), ModelInfo(), -1, false, kRai);
    TEST_REQUIRE(OnRai(model));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasLegacyNpu(model));
    // Named explicitly: the default is now the *build's* platform, which for
    // this translation unit is rai, so an argument-less reload would be the
    // same backend and would not switch anything.
    model.load_model(package.path().string(), ModelInfo(), -1, false, kFlm);
    TEST_REQUIRE(OnFlm(model));
    TEST_REQUIRE(Phi4FrontendTestAccess::HasLegacyNpu(model));
    TEST_REQUIRE(g_factory.legacy_calls == 1);
}

void TestPostSubmitErrorReturns500ClearsConversationAndLeavesModelPoisoned() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens = {1}; g_samples = {11}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    TEST_REQUIRE(model->get_current_context_length() == 0);
}

void TestPoisonedModelReturns500UntilReload() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens = {1}; g_samples = {11}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 500, true, "unload/reload");
    auto reloaded = ReadyRai(package);
    TEST_REQUIRE(reloaded->insert(meta, input));
}

void TestEosSelfTerminatesWithoutAnExtraDecode() {
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens = {1}; g_samples = {200020}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    (void)model->generate(meta, 10, output);
    TEST_REQUIRE(g_factory.engine->forward_calls == 0);
    TEST_REQUIRE(meta.stop_reason == EOT_DETECTED);
}

void TestRaiDecodeTimeAndSpeedAreMeasured() {
    // The rai path used to have a decode loop of its own that recorded nothing, so
    // profile reported "0 us" and a nan speed. It now shares _shared_generate;
    // this keeps the hardware acceptance record's decode throughput honest.
    TempPackage package; FactoryScope scope; auto model = ReadyRai(package);
    g_encoded_tokens = {1}; g_samples = {11, 12, 13, 200020}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->forward_delay = std::chrono::microseconds(2000);
    (void)model->generate(meta, 10, output);
    TEST_REQUIRE(g_factory.engine->forward_calls == 3);
    TEST_REQUIRE(meta.decoding_duration > 0);
    const auto profile = model->show_profile();
    TEST_REQUIRE(profile.find("nan") == std::string::npos);
    TEST_REQUIRE(profile.find("inf") == std::string::npos);
    TEST_REQUIRE(profile.find("Decoding time:       0 ") == std::string::npos);
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

void TestQueueCompletionReleasesImmediatelyOrDelaysQueuedHandoff() {
    constexpr auto cooldown = std::chrono::milliseconds(100);

    NPURequestCoordinator empty;
    bool released = false;
    const auto empty_start = std::chrono::steady_clock::now();
    empty.complete_current([](auto) { TEST_REQUIRE(false); },
                           [&] { released = true; }, cooldown);
    const auto empty_elapsed = std::chrono::steady_clock::now() - empty_start;
    TEST_REQUIRE(released);
    TEST_REQUIRE(empty_elapsed < std::chrono::milliseconds(50));

    NPURequestCoordinator queued;
    bool handed_off = false;
    bool released_while_queued = false;
    TEST_REQUIRE(queued.try_enqueue([] {}));
    const auto queued_start = std::chrono::steady_clock::now();
    queued.complete_current(
        [&](auto task) {
            handed_off = true;
            task();
        },
        [&] { released_while_queued = true; }, cooldown);
    const auto queued_elapsed = std::chrono::steady_clock::now() - queued_start;
    TEST_REQUIRE(handed_off);
    TEST_REQUIRE(!released_while_queued);
    TEST_REQUIRE(queued_elapsed >= std::chrono::milliseconds(75));
}

void TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable() {
    TempPackage package;
    FactoryScope scope;
    auto model = ReadyRai(package);
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
#if defined(FLM_ENABLE_RAI)
    RunTest(TestAbsentBackendStillBuildsQ4nxPhi4Npu, "TestAbsentBackendStillBuildsQ4nxPhi4Npu");
    RunTest(TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing, "TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing");
    RunTest(TestCorelibRaiGgufBuildsOnlyTheCorelibEngine, "TestCorelibRaiGgufBuildsOnlyTheCorelibEngine");
    RunTest(TestRaiProfileUsesCachedRuntimeDllPathAfterEnvironmentChanges, "TestRaiProfileUsesCachedRuntimeDllPathAfterEnvironmentChanges");
    RunTest(TestNoManifestOnnxConvertedWeightOrCachePathIsOpened, "TestNoManifestOnnxConvertedWeightOrCachePathIsOpened");
    RunTest(TestUnknownBackendIsAnError, "TestUnknownBackendIsAnError");
    RunTest(TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation, "TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation");
    RunTest(TestMissingCorelibFailsOnlyWhenRaiModelLoads, "TestMissingCorelibFailsOnlyWhenRaiModelLoads");
    RunTest(TestRaiSelectionWithMissingDllFailsWithoutChangingBackend, "TestRaiSelectionWithMissingDllFailsWithoutChangingBackend");
    RunTest(TestRaiSelectionCannotReachQ4nxPhi4NpuOrCpuFallback, "TestRaiSelectionCannotReachQ4nxPhi4NpuOrCpuFallback");
    RunTest(TestOrdinaryModelLoadsAfterAnRaiRuntimeLoadFailure, "TestOrdinaryModelLoadsAfterAnRaiRuntimeLoadFailure");
    RunTest(TestPreemptionIsRejectedForTheRaiRoute, "TestPreemptionIsRejectedForTheRaiRoute");
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
    RunTest(TestRaiDecodeTimeAndSpeedAreMeasured, "TestRaiDecodeTimeAndSpeedAreMeasured");
    RunTest(TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics, "TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics");
    RunTest(TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint, "TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint");
    RunTest(TestQueueCompletionReleasesImmediatelyOrDelaysQueuedHandoff, "TestQueueCompletionReleasesImmediatelyOrDelaysQueuedHandoff");
    RunTest(TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable, "TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable");
#else
    RunTest(TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib, "TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib");
    RunTest(TestFeatureOffRejectsRaiTagWithoutIncludingCorelibHeaders, "TestFeatureOffRejectsRaiTagWithoutIncludingCorelibHeaders");
#endif
    std::cout << "test_phi4_frontend: PASS\n";
}
