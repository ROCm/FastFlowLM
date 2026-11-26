#include "test_support.hpp"

#include <AutoModel/model_backend.hpp>
#include <AutoModel/modeling_gemma4e.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/// \brief the builtin set, emptied
/// \note The real one lives in builtin_backends.cpp and names every engine, and
///       with them the ~20 prebuilt engine libraries under src/lib/<runtime>
///       this suite deliberately does not link (see src/test/phi4_core's
///       test_phi4_frontend.cpp for the same pattern). This test registers the
///       one backend it needs instead.
namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

std::vector<int> g_encoded_tokens;
std::vector<int> g_samples;
std::size_t g_sample_index{};

/// \brief a causal_lm that is neither gemma4e_npu nor gemma4e_flash
/// \note This is exactly the shape of engine the core backend introduces for
///       the gemma4e family: a real causal_lm subclass that read_gemma4e_
///       engine_config<EngineT>'s dynamic_cast cannot resolve to EngineT.
class FakeEngine final : public causal_lm {
public:
    explicit FakeEngine(std::uint32_t limit) : max_length(limit) {}

    buffer<bf16> forward(int token) override {
        ++forward_calls;
        forwarded.push_back(token);
        ++position;
        return buffer<bf16>(1);
    }
    buffer<bf16> prefill(std::vector<int>& tokens, void*) override {
        ++prefill_calls;
        position += static_cast<int>(tokens.size());
        return buffer<bf16>(1);
    }
    void set_context_length(int value) override { position = value; }
    void load_weights(Q4NX&) override {}
    void update_max_length(std::uint32_t value) override { max_length = value; }
    void clear_context() override { position = 0; }
    buffer<bf16> get_k_cache(int, int) override { return buffer<bf16>(1); }
    buffer<bf16> get_v_cache(int, int) override { return buffer<bf16>(1); }
    int get_current_context_length() override { return position; }
    int checkpoint() override { return position; }
    int restore() override { return position; }

    std::uint32_t max_length;
    int position{};
    int prefill_calls{};
    int forward_calls{};
    std::vector<int> forwarded;
};

struct FactoryState {
    int calls{};
    FakeEngine* engine{};
} g_factory;

/// \brief the ids Google's real Gemma 4 package proves but does not state
///
/// Both shipped rows put `bos_token`/`eos_token` in `tokenizer_config.json`
/// and NEITHER `bos_token_id` NOR `eos_token_id`. The values below are the
/// ones three independent sources agree on -- the GGUF's
/// `tokenizer.ggml.{bos,eos}_token_id`, `tokenizer.json`'s vocab, and
/// `config.json` (bos in `text_config`, eos at the TOP level: the two live in
/// opposite places, which is the single easiest thing to get wrong here).
constexpr int kRealBosId = 2;
constexpr int kRealEosId = 106;
constexpr const char* kRealBosToken = "<bos>";
constexpr const char* kRealEosToken = "<turn|>";

/// \brief what the stub backend claims to have proven from its own package
/// \note Defaults to the real Gemma 4 values. A test that wants the
///       "backend knows nothing" case -- i.e. every family that does not
///       override these -- clears them.
struct ForcedIds {
    std::optional<int> bos = kRealBosId;
    std::optional<std::vector<int>> eos = std::vector<int>{kRealEosId};
} g_forced;

/// \brief a backend wrapping FakeEngine
/// \note Registered under the family's normal flm id via replace_backend, so
///       the frontend and registry both run their production code paths; only
///       the engine is faked. needs_npu_xclbin is turned off so load_model
///       never has to construct a real npu_xclbin_manager.
class StubBackend final : public flm::backend::ModelBackend {
public:
    explicit StubBackend(std::string id, std::uint32_t context_length)
        : id_(std::move(id)), engine_(std::make_unique<FakeEngine>(context_length)) {
        g_factory.engine = engine_.get();
    }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return id_; }

    /// \note Stands in for what the core backend derives from its GGUF. The
    ///       real one cross-validates three sources before answering; what
    ///       matters to THIS target is only that AutoModel asks the backend
    ///       and uses the answer.
    std::optional<int> forced_bos_id() const override { return g_forced.bos; }
    std::optional<std::vector<int>> forced_eos_ids() const override {
        return g_forced.eos;
    }

private:
    std::string id_;
    std::unique_ptr<FakeEngine> engine_;
};

std::unique_ptr<flm::backend::ModelBackend> MakeGemma4eStub(
    const flm::backend::BackendContext& context) {
    ++g_factory.calls;
    return std::make_unique<StubBackend>(std::string(flm::backend::kFlmBackendId),
                                         context.context_length);
}

/// rief the same stub under the oga-hipep id (gemma4-it-oga-hipep:26b-a4b runs on Gemma4e)
std::unique_ptr<flm::backend::ModelBackend> MakeGemma4eOgaStub(
    const flm::backend::BackendContext& context) {
    ++g_factory.calls;
    return std::make_unique<StubBackend>(std::string(flm::backend::kOgaHipepBackendId),
                                         context.context_length);
}

/// \brief what tokenizer_config.json states about its own ids
struct PackageOptions {
    /// \brief write a `bos_token_id`, which the real package does NOT
    std::optional<int> bos_token_id;
    /// \brief write an `eos_token_id` array, which the real package does NOT
    std::optional<int> eos_token_id;
    /// rief write `vocab_size` to config.json so the sampler has a vocabulary; unset keeps it 0
    std::optional<int> vocab_size;
};

/// \brief a temporary on-disk package with just enough files for Gemma4e::
///        load_model to succeed: config.json (LM_Config only needs it to
///        parse), and tokenizer_config.json (bos/eos tokens and a chat
///        template minja can render). tokenizer.json is never opened by any
///        code this test links, so it is not written.
///
/// THE tokenizer_config.json HERE IS GOOGLE'S OWN SHAPE, AND THAT IS THE
/// POINT. It used to write `bos_token = null`, which is the ONE value that
/// makes AutoModel::_shared_setup_tokenizer skip its bos check -- so this
/// target was green over a package no user will ever have, while the real
/// `gemma-4-E{2,4}B-it` package (`bos_token` = "<bos>", no `bos_token_id`,
/// no `eos_token_id`) hit `exit(1)` on the line AFTER the backend had loaded.
/// A fixture that carries the one value that skips the check is not covering
/// the frontend. See the C-1 finding in review-C10.md.
class TempPackage final {
public:
    explicit TempPackage(PackageOptions options = {}) {
        static std::uint64_t serial{};
        path_ = std::filesystem::temp_directory_path() /
            ("flm-taskA3-" + std::to_string(++serial));
        std::filesystem::create_directories(path_);
        nlohmann::json config = nlohmann::json::object();
        if (options.vocab_size) { config["vocab_size"] = *options.vocab_size; }
        Write(path_ / "config.json", config);
        nlohmann::json tokenizer_config;
        tokenizer_config["bos_token"] = kRealBosToken;
        tokenizer_config["eos_token"] = kRealEosToken;
        if (options.bos_token_id)
            tokenizer_config["bos_token_id"] = *options.bos_token_id;
        if (options.eos_token_id)
            tokenizer_config["eos_token_id"] =
                nlohmann::json::array({*options.eos_token_id});
        // Enough Jinja for minja to render: emit the BOS the real template
        // emits, then walk messages and their content blocks and concatenate
        // the text. Roles/special tokens are irrelevant here since the stubbed
        // Tokenizer::encode ignores the rendered text and returns whatever the
        // test set g_encoded_tokens to -- but `{{ bos_token }}` is NOT
        // irrelevant: the real chat template opens with exactly that, so it is
        // what a fix that switched the BOS off entirely would silently break.
        tokenizer_config["chat_template"] =
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% for item in message.content %}{{ item.text }}{% endfor %}"
            "{% endfor %}";
        Write(path_ / "tokenizer_config.json", tokenizer_config);
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
nlohmann::ordered_json ModelInfo() {
    return {{"default_context_length", 4096},
            {"details", {{"family", "gemma4e"}}},
            {"size", 100000000}};
}

chat_meta_info_t Meta() {
    chat_meta_info_t value;
    value.max_prefill_len = 64;
    return value;
}

lm_uniform_input_t TextInput(const std::string& prompt) {
    lm_uniform_input_t value;
    value.prompt = prompt;
    return value;
}

}  // namespace

// ---------------------------------------------------------------------------
// Stand-ins for the classes Gemma4e is built from, which this target does not
// otherwise link:
//   - Tokenizer/Sampler: the same shared classes test_phi4_frontend.cpp stubs,
//     for the same reason (no tokenizers-cpp / real sampling math needed).
//   - SafeTensors, ImageMemoryPool/ImageReader, AudioReader: value members
//     Gemma4e (or the headers it drags in) constructs and destroys as part of
//     an ordinary object lifetime. Their real implementations open weight
//     files or decode media (libjpeg-turbo/libopus/...); a text-only test never calls into them,
//     so only construction/destruction needs to link.
//   - Gemma4e's own load_image/load_image_base64/preprocess_image/load_audio/
//     load_audio_base64/clip_audio_length/extract_spectrogram: implemented in
//     modeling_gemma4e_image.cpp and modeling_gemma4e_audio.cpp, which this
//     target deliberately does not compile (task A3 is scoped to the
//     dynamic_cast helper in modeling_gemma4e.cpp only, and pulling those two
///    files in would pull in the image/audio codec libraries). insert() still *calls* these on the
//     image/audio branches, so the symbols must exist; a text-only run never
//     takes those branches, so the bodies are never exercised.
// ---------------------------------------------------------------------------

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
      repeat_last_n(config.repeat_last_n),
      use_optimized_sampling(config.use_optimized_sampling) {
    logits.resize(1); counters.resize(1); token_positions.resize(1, -1);
}
void Sampler::reset_penalties() {}
void Sampler::set_seed(uint64_t) {}
void Sampler::set_repeat_last_n(size_t n) { repeat_last_n = n; }

// No engine library is linked; a null table leaves the prefix cache off. The
// count says whether load_model asked the NPU engine for its table at all.
int g_state_api_calls = 0;
extern "C" const flm_state_api* gemma4e_npu_state_api(void*) {
    ++g_state_api_calls;
    return nullptr;
}
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

ImageMemoryPool::ImageMemoryPool(size_t max_cached_per_size)
    : max_cached_per_size_(max_cached_per_size) {}
ImageReader::ImageReader(size_t) {}
ImageReader::~ImageReader() = default;
AudioReader::AudioReader() = default;
AudioReader::~AudioReader() = default;

gemma4e_image_t Gemma4e::load_image(const std::string&) { return gemma4e_image_t{}; }
gemma4e_image_t Gemma4e::load_image_base64(const std::string&) { return gemma4e_image_t{}; }
void Gemma4e::preprocess_image(gemma4e_image_t&, std::pair<int, int>&, uint32_t&,
                                std::vector<bf16>&, std::vector<int>&, uint32_t&) {}
audio_data_t Gemma4e::load_audio(const std::string&, int, MonoDownmixMode) { return audio_data_t{}; }
audio_data_t Gemma4e::load_audio_base64(const std::string&, int, MonoDownmixMode) { return audio_data_t{}; }
std::vector<audio_data_t> Gemma4e::clip_audio_length(audio_data_t&, double) { return {}; }
void Gemma4e::extract_spectrogram(std::vector<audio_data_t>&, gemma4e_audio_payload_t&) {}

namespace {

/// \brief which family/id this suite drives -- the real gemma4e family, on the
///        id every build registers (flm), with the engine swapped for a fake
struct FactoryScope {
    FactoryScope() {
        g_factory = {};
        flm::backend::BackendTraits traits;
        traits.needs_npu_xclbin = false;
        flm::backend::BackendRegistry::instance().replace_backend(
            "gemma4e", flm::backend::kFlmBackendId, MakeGemma4eStub, traits);
        flm::backend::BackendTraits oga;
        oga.needs_npu_xclbin = false;
        oga.needs_npu_device = false;
        flm::backend::BackendRegistry::instance().replace_backend(
            "gemma4e", flm::backend::kOgaHipepBackendId, MakeGemma4eOgaStub, oga);
    }
};

/// \brief reaches the protected, virtual engine_config() the way test_phi4_
///        frontend.cpp reaches Phi4's own test-only surface: a subclass, not a
///        widened production interface. Overriding it (public, here) also
///        gives the "text path never asks for it" test something concrete to
///        assert on, instead of trusting that no exception means no call.
class Gemma4eFrontendTestAccess final : public Gemma4e {
public:
    using Gemma4e::Gemma4e;

    gemma4e_engine_config_t engine_config() const override {
        ++engine_config_calls;
        return Gemma4e::engine_config();
    }

    // AutoModel keeps these protected. Reading them through a subclass is the
    // same trick this file already uses for engine_config(): a test-only view,
    // not a widened production interface.
    bool has_bos() const { return this->has_bos_token; }
    int bos_id() const { return this->bos_token_id; }
    const std::vector<int>& eos_ids() const { return this->eos_token_ids; }
    bool conversational() const { return this->conversational_mode; }
    int image_budget() const { return this->image_softtoken_budget; }
    int sampler_top_k() const { return this->sampler->top_k; }
    int sampler_vocab() const { return this->sampler->in_features; }

    mutable int engine_config_calls = 0;
};

std::unique_ptr<Gemma4eFrontendTestAccess> MakeGemma4eWithStubEngineForTest(
    const TempPackage& package, const std::string& backend = "",
    const nlohmann::ordered_json& model_info = ModelInfo()) {
    auto model = std::make_unique<Gemma4eFrontendTestAccess>(nullptr);
    model->load_model(package.path().string(), model_info, -1, false, backend);
    return model;
}

/// \brief what the chat template renders for a one-message conversation
std::string RenderForTest(Gemma4eFrontendTestAccess& model) {
    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    messages.push_back(
        {{"role", "user"},
         {"content", nlohmann::ordered_json::array(
                         {{{"type", "text"}, {"text", "hi"}}})}});
    return model.apply_chat_template(messages);
}

void RunTextOnlyGenerationForTest(Gemma4eFrontendTestAccess& model, const std::string& prompt) {
    g_encoded_tokens = {11, 12, 13};
    g_samples = {21};
    g_sample_index = 0;
    auto meta = Meta();
    auto input = TextInput(prompt);
    TEST_REQUIRE(model.insert(meta, input));
    std::ostringstream output;
    (void)model.generate(meta, 1, output);
}

/// \brief restores the forced-id defaults so one test cannot leak into another
struct ForcedIdScope {
    explicit ForcedIdScope(ForcedIds forced) { g_forced = forced; }
    ForcedIdScope(const ForcedIdScope&) = delete;
    ForcedIdScope& operator=(const ForcedIdScope&) = delete;
    ~ForcedIdScope() { g_forced = ForcedIds{}; }
};

void TestTheRealPackageShapeLoadsWhenTheBackendKnowsTheIds() {
    // THE C-1 REGRESSION TEST, DRIVEN THROUGH THE PRODUCTION Gemma4e::
    // load_model. The fixture is Google's own tokenizer_config.json shape:
    // `bos_token` is "<bos>" and there is NO `bos_token_id` and NO
    // `eos_token_id`. Before the fix _shared_setup_tokenizer reached
    //
    //   [ERROR] bos_token is set in tokenizer_config.json but bos_token_id
    //           is missing or not an integer
    //
    // and called exit(1) -- on the line AFTER _shared_load_backend had
    // already loaded the model. There is no way to catch that, so the
    // failure mode of this test is the whole binary dying with exit 1.
    TempPackage package;
    FactoryScope scope;
    ForcedIdScope forced{ForcedIds{}};
    auto model = MakeGemma4eWithStubEngineForTest(package);

    TEST_REQUIRE(model->bos_id() == kRealBosId);
    TEST_REQUIRE(model->eos_ids() == std::vector<int>{kRealEosId});

    // AND THE BOS IS STILL ON. Phi-4 solved its own version of this by
    // setting `has_bos_token = false; bos_token_id = -1;`, which is right for
    // Phi-4 and would be wrong here: Gemma 4's chat template opens with
    // `{{ bos_token }}` and its GGUF says `add_bos_token: true`, so a fix
    // that switched the BOS off would silently drop one token from every
    // prompt -- a different prompt, with no error anywhere.
    TEST_REQUIRE(model->has_bos());
    const auto rendered = RenderForTest(*model);
    TEST_REQUIRE(rendered.rfind(kRealBosToken, 0) == 0);
}

void TestAPackageStatingItsOwnIdsIsUnaffectedWhenTheBackendForcesNothing() {
    // THE REGRESSION GUARD FOR EVERY OTHER FAMILY. _shared_setup_tokenizer is
    // shared by most frontends in the tree, and the backends they run on
    // return nullopt from both forced accessors. With nothing forced the file
    // is still the only source, exactly as before.
    TempPackage package(PackageOptions{/*bos_token_id=*/5, /*eos_token_id=*/9});
    FactoryScope scope;
    ForcedIdScope forced{ForcedIds{std::nullopt, std::nullopt}};
    auto model = MakeGemma4eWithStubEngineForTest(package);
    TEST_REQUIRE(model->bos_id() == 5);
    TEST_REQUIRE(model->eos_ids() == std::vector<int>{9});
    TEST_REQUIRE(model->has_bos());
}

void TestABackendThatProvedItsIdsOverridesTheFile() {
    // Same precedence forced_eos_ids() already has, and for the same reason:
    // the backend cross-validated three sources of its own package, and
    // tokenizer_config.json is one unverified source. A package that states
    // something else is a package that disagrees with its own weights.
    TempPackage package(PackageOptions{/*bos_token_id=*/5, /*eos_token_id=*/9});
    FactoryScope scope;
    ForcedIdScope forced{ForcedIds{}};
    auto model = MakeGemma4eWithStubEngineForTest(package);
    TEST_REQUIRE(model->bos_id() == kRealBosId);
    TEST_REQUIRE(model->bos_id() != 5);
    TEST_REQUIRE(model->eos_ids() == std::vector<int>{kRealEosId});
}

void TestEngineConfigOnAMismatchedEngineThrowsRatherThanReadingNull() {
    // A Release build compiles the assert out, so this is the difference
    // between a clear error and reading through a null pointer.
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    const auto error = RequireThrows([&] { (void)model->engine_config(); });
    RequireContains(error, "engine_config");
    RequireContains(error, "core");
}

void TestTextGenerationNeverAsksForEngineConfig() {
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    // The text path must not touch it -- that is why this has been latent.
    RunTextOnlyGenerationForTest(*model, "hello");
    TEST_REQUIRE(model->engine_config_calls == 0);
}

/// \brief one user message carrying `key` ("images" or "audios"), as the server hands it over
lm_uniform_input_t MessageWith(const char* key) {
    lm_uniform_input_t value;
    value.messages = nlohmann::ordered_json::array(
        {{{"role", "user"}, {"content", "hi"}, {key, nlohmann::ordered_json::array({"AAAA"})}}});
    return value;
}

/// \brief the message of the HTTP 400 insert() refuses `input` with
std::string RequireRefused(Gemma4eFrontendTestAccess& model, lm_uniform_input_t input) {
    auto meta = Meta();
    try {
        (void)model.insert(meta, input);
    } catch (const ModelRequestError& error) {
        TEST_REQUIRE(error.http_code() == 400);
        return error.what();
    }
    throw std::runtime_error("insert() took a request it should have refused");
}

void TestATextOnlyEntryRefusesImagesBeforeTheEngine() {
    // An entry without "vlm" may run an engine with no vision tower, such as an
    // extension engine behind this frontend, which cannot read the image payload.
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    RequireContains(RequireRefused(*model, MessageWith("images")), "does not accept image input");
    TEST_REQUIRE(model->engine_config_calls == 0);
}

void TestATextOnlyEntryRefusesAudioBeforeTheEngine() {
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    RequireContains(RequireRefused(*model, MessageWith("audios")), "does not accept audio input");
    TEST_REQUIRE(model->engine_config_calls == 0);
}

/// \brief ModelInfo() with `key` ("vlm" or "asr") set, as the built-in Gemma 4 entries set both
nlohmann::ordered_json ModelInfoWith(const char* key) {
    auto info = ModelInfo();
    info[key] = true;
    return info;
}

void TestAVisionEntryStillTakesImages() {
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package, "", ModelInfoWith("vlm"));
    // The stubbed decoder returns an empty image, which insert() skips, so the
    // request prefills its text: it got past the refusal.
    g_encoded_tokens = {11, 12, 13};
    auto meta = Meta();
    auto input = MessageWith("images");
    TEST_REQUIRE(model->insert(meta, input));
}

void TestAnAudioEntryStillHandsAudioToTheEngine() {
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package, "", ModelInfoWith("asr"));
    // The audio path asks the engine for its resample rate first; the fake
    // engine is not gemma4e_npu, so that throws, but not the 400 refusal.
    auto meta = Meta();
    auto input = MessageWith("audios");
    try {
        (void)model->insert(meta, input);
    } catch (const ModelRequestError& error) {
        throw std::runtime_error(std::string("an \"asr\" entry refused audio: ") + error.what());
    } catch (const std::exception&) {
    }
    TEST_REQUIRE(model->engine_config_calls == 1);
}

void TestAnOgaLoadDoesNotCallTheNpuEngine() {
    // WP-26 (PR #1066 review): the Windows OGA build delay-loads the NPU engine
    // DLLs. Calling gemma4e_npu_state_api would load gemma4e_npu.dll, which
    // needs XRT, and on a host without the NPU driver the process ends there.
    TempPackage package;
    FactoryScope scope;
    g_state_api_calls = 0;
    auto model = MakeGemma4eWithStubEngineForTest(package, flm::backend::kOgaHipepBackendId);
    TEST_REQUIRE(model->backend_id() == flm::backend::kOgaHipepBackendId);
    TEST_REQUIRE(g_state_api_calls == 0);
}

void TestAnNpuLoadStillAsksForTheStateTable() {
    TempPackage package;
    FactoryScope scope;
    g_state_api_calls = 0;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    TEST_REQUIRE(model->backend_id() == flm::backend::kFlmBackendId);
    TEST_REQUIRE(g_state_api_calls == 1);
}

void TestRequestSettingsInModelFieldsFallBackToTheLoadedValues() {
    // WP-04: conversational_mode and image_max_tokens live in Gemma4e's own fields, so the base
    // class's sampler/think reset does not reach them; the model's save/restore hooks do.
    TempPackage package;
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    model->save_request_settings();
    const bool loaded_conversational = model->conversational();
    const int loaded_budget = model->image_budget();

    TEST_REQUIRE(model->configure_parameter("conversational_mode", !loaded_conversational));
    TEST_REQUIRE(model->configure_parameter("image_max_tokens", 1120));
    TEST_REQUIRE(model->conversational() != loaded_conversational);
    TEST_REQUIRE(model->image_budget() == 1120);

    model->restore_request_settings();
    TEST_REQUIRE(model->conversational() == loaded_conversational);
    TEST_REQUIRE(model->image_budget() == loaded_budget);
}

void TestTopKOfZeroOrMinusOneTurnsTopKOff() {
    TempPackage package(PackageOptions{/*bos_token_id=*/std::nullopt, /*eos_token_id=*/std::nullopt, /*vocab_size=*/32});
    FactoryScope scope;
    auto model = MakeGemma4eWithStubEngineForTest(package);
    const int vocab = model->sampler_vocab();
    TEST_REQUIRE(vocab > 1);  // else "off" and "greedy" would be the same number
    model->set_topk(1);
    TEST_REQUIRE(model->sampler_top_k() == 1);
    model->set_topk(0);
    TEST_REQUIRE(model->sampler_top_k() == vocab);
    model->set_topk(1);
    model->set_topk(-1);
    TEST_REQUIRE(model->sampler_top_k() == vocab);
    model->set_topk(vocab + 100);
    TEST_REQUIRE(model->sampler_top_k() == vocab);
}

}  // namespace

int main() {
    RunTest(TestTheRealPackageShapeLoadsWhenTheBackendKnowsTheIds,
             "TestTheRealPackageShapeLoadsWhenTheBackendKnowsTheIds");
    RunTest(TestAPackageStatingItsOwnIdsIsUnaffectedWhenTheBackendForcesNothing,
             "TestAPackageStatingItsOwnIdsIsUnaffectedWhenTheBackendForcesNothing");
    RunTest(TestABackendThatProvedItsIdsOverridesTheFile,
             "TestABackendThatProvedItsIdsOverridesTheFile");
    RunTest(TestEngineConfigOnAMismatchedEngineThrowsRatherThanReadingNull,
             "TestEngineConfigOnAMismatchedEngineThrowsRatherThanReadingNull");
    RunTest(TestTextGenerationNeverAsksForEngineConfig,
             "TestTextGenerationNeverAsksForEngineConfig");
    RunTest(TestRequestSettingsInModelFieldsFallBackToTheLoadedValues,
             "TestRequestSettingsInModelFieldsFallBackToTheLoadedValues");
    RunTest(TestTopKOfZeroOrMinusOneTurnsTopKOff, "TestTopKOfZeroOrMinusOneTurnsTopKOff");
    RunTest(TestATextOnlyEntryRefusesImagesBeforeTheEngine,
             "TestATextOnlyEntryRefusesImagesBeforeTheEngine");
    RunTest(TestATextOnlyEntryRefusesAudioBeforeTheEngine,
             "TestATextOnlyEntryRefusesAudioBeforeTheEngine");
    RunTest(TestAVisionEntryStillTakesImages, "TestAVisionEntryStillTakesImages");
    RunTest(TestAnAudioEntryStillHandsAudioToTheEngine,
             "TestAnAudioEntryStillHandsAudioToTheEngine");
    RunTest(TestAnOgaLoadDoesNotCallTheNpuEngine, "TestAnOgaLoadDoesNotCallTheNpuEngine");
    RunTest(TestAnNpuLoadStillAsksForTheStateTable, "TestAnNpuLoadStillAsksForTheStateTable");
    std::cout << "test_gemma4_frontend: PASS\n";
}
