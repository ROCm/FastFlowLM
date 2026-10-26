#include "test_support.hpp"
#if defined(FLM_ENABLE_CORE)
#include "fake_corelib.hpp"
#endif
#include "utils/file_access.hpp"

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_phi4.hpp>
#if defined(FLM_ENABLE_CORE)
#include <core/corelib_runtime.hpp>
#include <models/phi4/core/dev04/phi4_core_backend.hpp>
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

constexpr const char* kGguf = "Phi-4-mini-instruct-Q4_G128.gguf";

std::vector<int> g_encoded_tokens;
std::vector<int> g_samples;
std::vector<std::filesystem::path> g_opened_paths;
std::size_t g_sample_index{};
int g_unowned_samples{};

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
        prefill_sizes.push_back(static_cast<int>(tokens.size()));
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
    std::vector<int> prefill_sizes;
    int forward_calls{};
    bool fail_prefill{};
    bool fail_forward{};
    bool poisoned_state{};
    std::chrono::microseconds forward_delay{0};
    std::vector<int> forwarded;
};

struct FactoryState {
    int legacy_calls{};
    int core_calls{};
    bool throw_for_core{};
    bool aligns_prefill_chunks{};
    FakeEngine* engine{};
#if defined(FLM_ENABLE_CORE)
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
    bool aligns_prefill_chunks() const override { return g_factory.aligns_prefill_chunks; }
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

#if defined(FLM_ENABLE_CORE)
/// \brief the corelib backend, with its engine and runtime faked
/// \note The real backend states its EOS ids, so before it touches the device
///       it reads no file and only requires the GGUF corelib will read.
std::unique_ptr<flm::backend::ModelBackend> MakeCoreStub(
    const flm::backend::BackendContext& context) {
    const std::filesystem::path root(context.model_path);
    if (!std::filesystem::exists(root / kGguf)) {
        throw std::runtime_error("Phi-4 core: " + (root / kGguf).string() + " does not exist");
    }

    ++g_factory.core_calls;
    if (g_factory.throw_for_core) throw std::runtime_error("missing corelib");

    auto backend = std::make_unique<StubBackend>(
        std::string(flm::backend::kCoreBackendId), context.context_length);
    backend->decode_limit_ = flm::phi4::kCoreDecodeLimit;
    backend->supports_preemption_ = false;
    backend->forwards_past_eos_ = false;
    backend->forced_eos_ids_ = std::vector<int>({200020, 199999});
    if (g_factory.runtime) {
        backend->detail_ = g_factory.runtime->loaded_library_path().string();
    }
    return backend;
}
#endif

nlohmann::json PackageConfig() {
    return {{"model_type", "phi3"}, {"num_hidden_layers", 32},
            {"hidden_size", 3072}, {"intermediate_size", 8192},
            {"num_attention_heads", 24}, {"num_key_value_heads", 8},
            {"head_dim", 128}, {"vocab_size", 200064},
            {"rms_norm_eps", 1.0e-5}, {"original_max_position_embeddings", 4096},
            {"eos_token_id", 199999}};
}

nlohmann::json PackageTokenizer() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int id = 0; id < 200019; ++id) vocab["t" + std::to_string(id)] = id;
    vocab["<|endoftext|>"] = 199999;
    vocab["<|end|>"] = 200020;
    return {{"model", {{"vocab", std::move(vocab)}}},
            {"added_tokens", nlohmann::json::array({
                {{"id", 200019}, {"content", "<|assistant|>"}},
                {{"id", 200020}, {"content", "<|end|>"}},
                {{"id", 200021}, {"content", "<|user|>"}},
                {{"id", 200022}, {"content", "<|system|>"}},
                {{"id", 200023}, {"content", "<|tool|>"}},
                {{"id", 200024}, {"content", "<|/tool|>"}},
                {{"id", 200025}, {"content", "<|tool_call|>"}},
                {{"id", 200026}, {"content", "<|/tool_call|>"}},
                {{"id", 200027}, {"content", "<|tool_response|>"}},
                {{"id", 200028}, {"content", "<|tag|>"}},
                {{"id", 200018}, {"content", "<|endofprompt|>"}},
                {{"id", 199999}, {"content", "<|endoftext|>"}}})}};
}

/// \brief a Phi-4 package directory; an invalid one has no GGUF
class TempPackage final {
public:
    explicit TempPackage(bool valid = true) {
        static std::uint64_t serial{};
        path_ = std::filesystem::temp_directory_path() /
            ("flm-task4-" + std::to_string(++serial));
        std::filesystem::create_directories(path_);
        Write(path_ / "config.json", PackageConfig());
        Write(path_ / "tokenizer.json", PackageTokenizer());
        nlohmann::json tokenizer_config = {
            {"add_bos_token", false},
            {"chat_template", "<|user|>{{ message }}<|end|><|assistant|>"},
            {"eos_token", "<legacy-eos>"},
            {"eos_token_id", nlohmann::json::array({200020, 199999})}};
        Write(path_ / "tokenizer_config.json", tokenizer_config);
        if (valid) {
            std::ofstream(path_ / kGguf, std::ios::binary) << "GGUF";
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
int Sampler::sample(buffer<bf16>& logits, const std::vector<uint32_t>*, sample_logprobs_t*) {
    // FakeEngine hands back logits it allocated, as the core engines do. buffer's
    // copy is shallow and non-owning, so a buffer that no longer owns them by the
    // time they are sampled is reading memory its owner has already freed.
    if (!logits.is_owner()) ++g_unowned_samples;
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
constexpr const char* kCore = "core";

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
#if defined(FLM_ENABLE_CORE)
        registry.replace_backend("phi4", kCore, MakeCoreStub,
                                 flm::phi4::core_traits());
#endif
    }
    ~FactoryScope() {
        flm::file_access::SetOpenObserver({});
#if defined(FLM_ENABLE_CORE)
        g_factory.runtime.reset();
#endif
    }
};

/// \brief which backend a loaded model ended up on
bool OnCore(const AutoModel& model) {
    return model.backend_id() == kCore;
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
    TEST_REQUIRE(g_factory.core_calls == 0);
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
    TEST_REQUIRE(g_factory.core_calls == 0);
}

void TestPrefillLogitsAreStillOwnedWhenSampled() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo());
    g_encoded_tokens = {1, 2, 3};
    g_samples = {7};
    g_sample_index = 0;
    g_unowned_samples = 0;
    auto meta = Meta();
    // A prefill limit of 512 or more takes the chunked path, where the logits
    // are handed out of the last chunk.
    meta.max_prefill_len = 4096;
    auto input = Input(1);
    TEST_REQUIRE(model->insert(meta, input));
    TEST_REQUIRE(g_factory.engine->prefill_calls == 1);
    TEST_REQUIRE(g_unowned_samples == 0);
}

/// \brief the chunk sizes a turn adding `fresh` tokens is prefilled in, after a
///        first turn left `held` tokens in the engine
std::vector<int> ContinuationChunks(bool aligned, int held, int fresh) {
    TempPackage package;
    FactoryScope scope;
    g_factory.aligns_prefill_chunks = aligned;
    auto model = Load(package, ModelInfo());
    auto meta = Meta();
    meta.max_prefill_len = 512;
    auto input = Input(1);
    g_encoded_tokens.clear();
    for (int i = 0; i < held; ++i) g_encoded_tokens.push_back(i + 1);
    TEST_REQUIRE(model->insert(meta, input));
    TEST_REQUIRE(g_factory.engine->position == held);
    for (int i = 0; i < fresh; ++i) g_encoded_tokens.push_back(held + i + 1);
    g_factory.engine->prefill_sizes.clear();
    TEST_REQUIRE(model->insert(meta, input));
    return g_factory.engine->prefill_sizes;
}

void TestPrefillChunksEndOnMultiplesFromPositionZeroWhenTheBackendAsks() {
    TEST_REQUIRE((ContinuationChunks(false, 100, 900) == std::vector<int>{512, 388}));
    TEST_REQUIRE((ContinuationChunks(true, 100, 900) == std::vector<int>{412, 488}));
    TEST_REQUIRE((ContinuationChunks(true, 512, 600) == std::vector<int>{512, 88}));
    TEST_REQUIRE((ContinuationChunks(true, 100, 300) == std::vector<int>{300}));
}

void TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_core = true;
    auto model = Load(package, ModelInfo());
    g_encoded_tokens = {1};
    auto meta = Meta();
    auto input = Input(1);
    TEST_REQUIRE(model->insert(meta, input));
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(g_factory.core_calls == 0);
}

void TestCorelibCoreGgufBuildsOnlyTheCorelibEngine() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo(), -1, false, nullptr, kCore);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
    TEST_REQUIRE(g_factory.core_calls == 1);
    TEST_REQUIRE(OnCore(*model));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasLegacyNpu(*model));
}

#if defined(FLM_ENABLE_CORE)
void TestCoreProfileUsesCachedRuntimeDllPathAfterEnvironmentChanges() {
    TempPackage package;
    FactoryScope scope;
    fake_corelib::Reset();
    const auto dll_a = std::filesystem::absolute(package.path() / "runtime-a.dll");
    const auto dll_b = std::filesystem::absolute(package.path() / "runtime-b.dll");
    auto api = flm::corelib::CorelibApi::ResolveForTest(
        fake_corelib::Resolver(), dll_a);
    auto runtime = flm::corelib::CorelibRuntime::CreateForTest(std::move(api));

#ifdef _WIN32
    _putenv_s("FLM_CORE_CORELIB_PATH", dll_b.string().c_str());
#else
    setenv("FLM_CORE_CORELIB_PATH", dll_b.string().c_str(), 1);
#endif
    // The backend caches the runtime it was built with; a later environment
    // change must not move the path it reports.
    g_factory.runtime = runtime;
    auto core = Load(package, ModelInfo(), -1, false, nullptr, kCore);
    g_factory.runtime.reset();
    const auto core_profile = core->show_profile();
    RequireContains(core_profile, kCore);
    RequireContains(core_profile, dll_a.string());
    TEST_REQUIRE(core_profile.find(dll_b.string()) == std::string::npos);

    auto legacy = Load(package, ModelInfo());
    const auto legacy_profile = legacy->show_profile();
    TEST_REQUIRE(legacy_profile.find(kCore) == std::string::npos);
    TEST_REQUIRE(legacy_profile.find(dll_a.string()) == std::string::npos);
    core.reset();
    runtime.reset();
    flm::corelib::CorelibRuntime::ShutdownProcess();
#ifdef _WIN32
    _putenv_s("FLM_CORE_CORELIB_PATH", "");
#else
    unsetenv("FLM_CORE_CORELIB_PATH");
#endif
}
#endif

void TestNoManifestOnnxConvertedWeightOrCachePathIsOpened() {
    TempPackage package;
    FactoryScope scope;
    auto model = Load(package, ModelInfo(), -1, false, nullptr, kCore);
    TEST_REQUIRE(OnCore(*model));
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
        "config.json", "tokenizer.json", "tokenizer_config.json"}));
}

void TestUnknownBackendIsAnError() {
    TempPackage package;
    FactoryScope scope;
    // Hardware this build has no engine for is refused by name, before any
    // engine is constructed.
    const auto message = RequireThrows(
        [&] { (void)Load(package, ModelInfo(), -1, false, nullptr, "other"); });
    RequireContains(message, "is not available for model family 'phi4'");
    RequireContains(message, "other");
    TEST_REQUIRE(g_factory.legacy_calls == 0 && g_factory.core_calls == 0);
}

void TestFeatureOffRejectsCoreTagWithoutIncludingCorelibHeaders() {
#if !defined(FLM_ENABLE_CORE)
    TempPackage package;
    FactoryScope scope;
    const auto message = RequireThrows([&] {
        (void)Load(package, ModelInfo(), -1, false, nullptr, kCore);
    });
    RequireContains(message, "is not available for model family 'phi4'");
    RequireContains(message, "It provides: flm");
    TEST_REQUIRE(g_factory.legacy_calls == 0 && g_factory.core_calls == 0);
#endif
}

void TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation() {
    TempPackage package(false);
    FactoryScope scope;
    RequireThrows([&] { (void)Load(package, ModelInfo(), -1, false, nullptr, kCore); });
    TEST_REQUIRE(g_factory.core_calls == 0);
}

void TestMissingCorelibFailsOnlyWhenCoreModelLoads() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_core = true;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo(), -1, false, nullptr, kCore); }), "missing corelib");
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestCoreSelectionWithMissingDllFailsWithoutChangingBackend() {
    TempPackage package;
    FactoryScope scope;
    Phi4 model(nullptr);
    g_factory.throw_for_core = true;
    RequireContains(RequireThrows([&] {
        model.load_model(package.path().string(), ModelInfo(), -1, false, kCore);
    }), "missing corelib");
    TEST_REQUIRE(model.backend_id().empty());
    TEST_REQUIRE(g_factory.core_calls == 1);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestCoreSelectionCannotReachQ4nxPhi4NpuOrCpuFallback() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_core = true;
    (void)RequireThrows([&] {
        (void)Load(package, ModelInfo(), -1, false, nullptr, kCore);
    });
    TEST_REQUIRE(g_factory.core_calls == 1);
    TEST_REQUIRE(g_factory.legacy_calls == 0);
}

void TestOrdinaryModelLoadsAfterACoreRuntimeLoadFailure() {
    TempPackage package;
    FactoryScope scope;
    g_factory.throw_for_core = true;
    RequireThrows([&] { (void)Load(package, ModelInfo(), -1, false, nullptr, kCore); });
    g_factory.throw_for_core = false;
    auto ordinary = Load(package, ModelInfo());
    TEST_REQUIRE(g_factory.legacy_calls == 1);
    TEST_REQUIRE(OnFlm(*ordinary));
}

void TestPreemptionIsRejectedForTheCoreRoute() {
    TempPackage package;
    FactoryScope scope;
    RequireContains(RequireThrows([&] { (void)Load(package, ModelInfo(), -1, true, nullptr, kCore); }), "preemption");
    TEST_REQUIRE(g_factory.core_calls == 0);
}

std::unique_ptr<Phi4> ReadyCore(const TempPackage& package) {
    return Load(package, ModelInfo(), -1, false, nullptr, kCore);
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

void TestOmittedZeroAndNegativeSentinelBudgetsCapAtRemainingWindow() {
    for (const auto requested : {std::optional<int>{}, std::optional<int>{0}, std::optional<int>{-1}}) {
        TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
        g_encoded_tokens.assign(4093, 1); g_samples = {11, 12, 13}; g_sample_index = 0;
        auto meta = Meta(); auto input = Input(requested); std::ostringstream output;
        TEST_REQUIRE(model->insert(meta, input));
        (void)model->generate(meta, 4096, output);
        TEST_REQUIRE(meta.generated_tokens == 2);
        TEST_REQUIRE(meta.stop_reason == MAX_LENGTH_REACHED);
    }

    // /api/chat uses generate_with_prompt and retains 4096 only as the legacy
    // loop default; omission must not become an explicit core budget.
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens.assign(4093, 1); g_samples = {11, 12, 13}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    (void)model->generate_with_prompt(meta, input, 4096, output);
    TEST_REQUIRE(meta.generated_tokens == 2);
}

void TestCancellationBeforePrefillSubmitsNothing() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1, 2}; auto meta = Meta(); auto input = Input();
    int checks = 0;
    TEST_REQUIRE(!model->insert(meta, input, [&] { return ++checks >= 2; }));
    TEST_REQUIRE(checks >= 2);
    TEST_REQUIRE(g_factory.engine->prefill_calls == 0);
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
}

void TestCancellationBetweenDecodeStepsStopsWithCancelReason() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
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
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
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

    auto core = ReadyCore(package);
    TEST_REQUIRE(Phi4FrontendTestAccess::EosTokenIds(*core) ==
                 std::vector<int>({200020, 199999}));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasBosToken(*core));
}

void TestSamePathBackendSwitchForcesLegacyInitialization() {
    TempPackage package; FactoryScope scope;
    Phi4 model(reinterpret_cast<flm_rt::device*>(1));
    model.load_model(package.path().string(), ModelInfo(), -1, false, kCore);
    TEST_REQUIRE(OnCore(model));
    TEST_REQUIRE(!Phi4FrontendTestAccess::HasLegacyNpu(model));
    // Named explicitly: the default is now the *build's* platform, which for
    // this translation unit is core, so an argument-less reload would be the
    // same backend and would not switch anything.
    model.load_model(package.path().string(), ModelInfo(), -1, false, kFlm);
    TEST_REQUIRE(OnFlm(model));
    TEST_REQUIRE(Phi4FrontendTestAccess::HasLegacyNpu(model));
    TEST_REQUIRE(g_factory.legacy_calls == 1);
}

void TestPostSubmitErrorReturns500ClearsConversationAndLeavesModelPoisoned() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1}; g_samples = {11}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    TEST_REQUIRE(model->get_current_context_length() == 0);
}

void TestPoisonedModelReturns500UntilReload() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1}; g_samples = {11}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    g_factory.engine->fail_forward = true;
    ExpectRequestError([&] { (void)model->generate(meta, 3, output); }, 500, true, "unload/reload");
    ExpectRequestError([&] { (void)model->insert(meta, input); }, 500, true, "unload/reload");
    auto reloaded = ReadyCore(package);
    TEST_REQUIRE(reloaded->insert(meta, input));
}

void TestEosSelfTerminatesWithoutAnExtraDecode() {
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
    g_encoded_tokens = {1}; g_samples = {200020}; g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model->insert(meta, input));
    (void)model->generate(meta, 10, output);
    TEST_REQUIRE(g_factory.engine->forward_calls == 0);
    TEST_REQUIRE(meta.stop_reason == EOT_DETECTED);
}

struct SecondTurn {
    std::vector<int> forwarded;
    std::vector<int> prefill_sizes;
};

/// \brief a first turn that ends in EOS, then a second that resends it, its EOS and one token
SecondTurn RunSecondTurn(Phi4& model, std::vector<int> first_samples) {
    g_encoded_tokens = {1, 2}; g_samples = std::move(first_samples); g_sample_index = 0;
    auto meta = Meta(); auto input = Input(); std::ostringstream output;
    TEST_REQUIRE(model.insert(meta, input));
    (void)model.generate(meta, 10, output);
    TEST_REQUIRE(meta.stop_reason == EOT_DETECTED);
    SecondTurn turn{g_factory.engine->forwarded, {}};
    g_encoded_tokens = {1, 2};
    for (std::size_t i = 0; i + 1 < g_samples.size(); ++i) g_encoded_tokens.push_back(g_samples[i]);
    g_encoded_tokens.insert(g_encoded_tokens.end(), {200020, 3});
    g_factory.engine->prefill_sizes.clear();
    TEST_REQUIRE(model.insert(meta, input));
    turn.prefill_sizes = g_factory.engine->prefill_sizes;
    return turn;
}

void TestAnEosTheEngineNeverRanIsPrefilledWithTheNextTurn() {
    // The core backend declines the post-EOS forward, so its engine holds the
    // reply without its EOS and the next turn has to carry it: here 200020, 3.
    {
        TempPackage package; FactoryScope scope; auto core = ReadyCore(package);
        const auto turn = RunSecondTurn(*core, {11, 200020});
        TEST_REQUIRE((turn.forwarded == std::vector<int>{11}));
        TEST_REQUIRE((turn.prefill_sizes == std::vector<int>{2}));
    }
    // An EOS sampled straight from the prefill is never run either.
    {
        TempPackage package; FactoryScope scope; auto core = ReadyCore(package);
        const auto turn = RunSecondTurn(*core, {200020});
        TEST_REQUIRE(turn.forwarded.empty());
        TEST_REQUIRE((turn.prefill_sizes == std::vector<int>{2}));
    }
    // The flm backend forwards the EOS, so only the new token is left.
    {
        TempPackage package; FactoryScope scope; auto flm = Load(package, ModelInfo());
        const auto turn = RunSecondTurn(*flm, {11, 200020});
        TEST_REQUIRE((turn.forwarded == std::vector<int>{11, 200020}));
        TEST_REQUIRE((turn.prefill_sizes == std::vector<int>{1}));
    }
}

void TestCoreDecodeTimeAndSpeedAreMeasured() {
    // The core path used to have a decode loop of its own that recorded nothing, so
    // profile reported "0 us" and a nan speed. It now shares _shared_generate;
    // this keeps the hardware acceptance record's decode throughput honest.
    TempPackage package; FactoryScope scope; auto model = ReadyCore(package);
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
    auto model = ReadyCore(package);
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
#if defined(FLM_ENABLE_CORE)
    RunTest(TestAbsentBackendStillBuildsQ4nxPhi4Npu, "TestAbsentBackendStillBuildsQ4nxPhi4Npu");
    RunTest(TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing, "TestEnabledBuildStartsAndRunsLegacyPhi4WhenCorelibDllIsMissing");
    RunTest(TestCorelibCoreGgufBuildsOnlyTheCorelibEngine, "TestCorelibCoreGgufBuildsOnlyTheCorelibEngine");
    RunTest(TestCoreProfileUsesCachedRuntimeDllPathAfterEnvironmentChanges, "TestCoreProfileUsesCachedRuntimeDllPathAfterEnvironmentChanges");
    RunTest(TestNoManifestOnnxConvertedWeightOrCachePathIsOpened, "TestNoManifestOnnxConvertedWeightOrCachePathIsOpened");
    RunTest(TestUnknownBackendIsAnError, "TestUnknownBackendIsAnError");
    RunTest(TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation, "TestInvalidPackageFailsBeforeRuntimeAndDeviceCreation");
    RunTest(TestMissingCorelibFailsOnlyWhenCoreModelLoads, "TestMissingCorelibFailsOnlyWhenCoreModelLoads");
    RunTest(TestCoreSelectionWithMissingDllFailsWithoutChangingBackend, "TestCoreSelectionWithMissingDllFailsWithoutChangingBackend");
    RunTest(TestCoreSelectionCannotReachQ4nxPhi4NpuOrCpuFallback, "TestCoreSelectionCannotReachQ4nxPhi4NpuOrCpuFallback");
    RunTest(TestOrdinaryModelLoadsAfterACoreRuntimeLoadFailure, "TestOrdinaryModelLoadsAfterACoreRuntimeLoadFailure");
    RunTest(TestPreemptionIsRejectedForTheCoreRoute, "TestPreemptionIsRejectedForTheCoreRoute");
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
    RunTest(TestAnEosTheEngineNeverRanIsPrefilledWithTheNextTurn, "TestAnEosTheEngineNeverRanIsPrefilledWithTheNextTurn");
    RunTest(TestCoreDecodeTimeAndSpeedAreMeasured, "TestCoreDecodeTimeAndSpeedAreMeasured");
    RunTest(TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics, "TestCliAndAllFourGenerationEndpointsPassTheSameBudgetSemantics");
    RunTest(TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint, "TestQueueCompletionIsExactlyOnceAndIncludesCompletionsEndpoint");
    RunTest(TestQueueCompletionReleasesImmediatelyOrDelaysQueuedHandoff, "TestQueueCompletionReleasesImmediatelyOrDelaysQueuedHandoff");
    RunTest(TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable, "TestCancellationAndCapacityErrorsLeaveTheServerQueueUsable");
    RunTest(TestPrefillChunksEndOnMultiplesFromPositionZeroWhenTheBackendAsks, "TestPrefillChunksEndOnMultiplesFromPositionZeroWhenTheBackendAsks");
#else
    RunTest(TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib, "TestDefaultBuildCanConstructAndRunLegacyPhi4WithoutCorelib");
    RunTest(TestPrefillLogitsAreStillOwnedWhenSampled, "TestPrefillLogitsAreStillOwnedWhenSampled");
    RunTest(TestPrefillChunksEndOnMultiplesFromPositionZeroWhenTheBackendAsks, "TestPrefillChunksEndOnMultiplesFromPositionZeroWhenTheBackendAsks");
    RunTest(TestFeatureOffRejectsCoreTagWithoutIncludingCorelibHeaders, "TestFeatureOffRejectsCoreTagWithoutIncludingCorelibHeaders");
#endif
    std::cout << "test_phi4_frontend: PASS\n";
}
