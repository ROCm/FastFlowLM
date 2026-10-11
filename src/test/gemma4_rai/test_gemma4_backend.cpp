/// \file test_gemma4_backend.cpp
/// \brief Task C10 -- the Gemma 4 rai backend, its traits, and registration.
///
/// THIS TARGET DRIVES THE REAL BACKEND, NOT A STUB. src/test/phi4_rai's
/// test_phi4_frontend.cpp has to stand a stub in for its corelib backend
/// because it is a FRONTEND test: it links modeling_phi4.cpp and cannot also
/// link an engine. This one links `gemma4_rai_backend.cpp` itself and swaps
/// only the thing underneath it -- corelib -- for `fake_corelib`. What makes
/// that possible is that `CorelibRuntime::GetOrCreate` returns the
/// process-wide runtime when one already exists, so a test that installs a
/// fake runtime FIRST hands the production code path a fake device without
/// the production code path knowing. So "validation happens before the
/// runtime is acquired" and "a bad package creates no device object" are
/// assertions about the backend rather than about a copy of it.
///
/// What it still cannot see: `fake_corelib` computes nothing (see its header's
/// own "what it cannot catch" list), so nothing here says the model is right.
/// This binary is about POLICY -- the six overrides, the order of
/// construction, which files are opened and when, and what the registry
/// answers for family `gemma4e`.
///
/// The E2B and E4B synthetic fixtures deliberately disagree about ONE value:
/// the E2B fixture carries the real EOS id 106 and the E4B fixture carries
/// 4242. See kE4bEosId.
#include "models/gemma4/rai/aie_next/gemma4_rai.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_backend.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "rai/corelib_runtime.hpp"
#include "utils/file_access.hpp"

#include "AutoModel/model_backend.hpp"
#include "model_list.hpp"

#include "fake_corelib.hpp"
#include "gemma4_gguf_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

/// \brief the builtin set, emptied
/// \note The real one lives in builtin_backends.cpp and names every engine
///       type, and with them the ~20 prebuilt libraries under
///       src/lib/<runtime> this suite deliberately does not link. This binary
///       makes the ONE registration call C10 adds to that file, verbatim --
///       see RegisterAsBuiltinBackendsDoes below.
namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

/// \brief stands in for the real one, which lives in utils.cpp and drags in
///        the NPU runtime
/// \note Never consulted: every test installs a fake process runtime before
///       the backend runs, and `CorelibRuntime::GetOrCreate` returns that one
///       without looking at the directory. It has to LINK, not to work.
namespace utils {
std::string get_executable_directory() {
    return std::filesystem::current_path().string();
}
}  // namespace utils

namespace {

using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::gemma4::Gemma4GgufPackage;

constexpr const char* kFamily = "gemma4e";
constexpr const char* kRai = "rai";

/// \brief the EOS id every real Gemma 4 package states, from three sources
/// \note `tokenizer.ggml.eos_token_id` in the GGUF, top-level `eos_token_id`
///       in config.json, and `tokenizer_config.json`'s `eos_token` resolved
///       through tokenizer.json's vocab. Measured on both shipped rows:
///       `<turn|>` is 106 and the three agree. NOT what the backend returns --
///       the backend reads the file. This constant is what the TEST expects
///       the file to have said.
constexpr int kRealEosId = 106;
constexpr const char* kRealEosToken = "<turn|>";

/// \brief the other stop id real packages list in config.json (`<eos>`)
constexpr int kRealConfigEosId = 1;

/// \brief the BOS id every real Gemma 4 package states, from three sources
/// \note `tokenizer.ggml.bos_token_id` in the GGUF, `text_config.bos_token_id`
///       in config.json -- NOT the top level, which is where the EOS lives and
///       where no real file carries a BOS at all -- and `tokenizer_config
///       .json`'s `bos_token` resolved through tokenizer.json's vocab. All
///       three say 2 (`<bos>`) on both shipped rows.
/// \note The package states `bos_token` and NO `bos_token_id`, which is the
///       whole reason forced_bos_id() exists: AutoModel::_shared_setup_
///       tokenizer used to exit(1) on exactly that shape.
constexpr int kRealBosId = 2;
constexpr const char* kRealBosToken = "<bos>";

/// \brief the EOS id the E4B SYNTHETIC fixture carries instead of 106
///
/// THE WHOLE POINT OF THE COLUMN. `forced_eos_ids()` exists because three
/// sources agreeing beats any one of them -- and a literal agrees with
/// nothing. A suite whose every fixture says 106 cannot tell a backend that
/// READ 106 from one that RETURNED 106, so it would stay green if someone
/// replaced the derivation with `return {106};`. This fixture says something
/// else, and TestForcedEosIdsAreReadFromTheFileNotHardcoded is the test that
/// goes red for that edit. The real E4B file does say 106, and the opt-in
/// real-package tests below assert exactly that against it.
constexpr int kE4bEosId = 4242;

/// \brief the BOS id the E4B SYNTHETIC fixture carries instead of 2
/// \note Same argument as kE4bEosId, for the other id. Every real package
///       says 2, so a fixture set that only ever said 2 could not tell
///       `return 2;` from a derivation.
constexpr int kE4bBosId = 4343;

/// \brief the path the fake corelib reports as its loaded library
/// \note `detail()` is the provenance line in `flm show`, so it has to be the
///       path the runtime actually loaded rather than anything recomputed.
const std::filesystem::path& FakeDllPath() {
    static const std::filesystem::path path =
        std::filesystem::absolute(std::filesystem::temp_directory_path() /
                                  "flm-gemma4-backend-fake-corelib.dll");
    return path;
}

// ---------------------------------------------------------------------------
// The two synthetic fixtures, written ONCE each.
//
// A full E2B fixture is ~5 GB and an E4B one ~8.6 GB, so they are PINNED
// (see Builder::Write) and every package below HARD LINKS to one of them
// rather than copying: a package directory needs the GGUF to sit inside it
// under the name the catalog entry gives, and there are a dozen packages in
// this file. A hard link is instant and costs no space, and deleting one
// leaves the fixture alone.

struct Fixture {
    gemma4_fixture::Builder builder;
    std::filesystem::path path;
    std::int64_t layers;
    int eos_id;
    int bos_id;
};

const Fixture& E2bFixture() {
    static const Fixture fixture = [] {
        const auto options = gemma4_fixture::E2bOptions();
        auto builder = gemma4_fixture::MakeBuilder(options);
        // Left at the fixture's own defaults, which are the real values.
        builder.SetMetadata("tokenizer.ggml.eos_token_id",
                            static_cast<std::uint32_t>(kRealEosId));
        builder.SetMetadata("tokenizer.ggml.bos_token_id",
                            static_cast<std::uint32_t>(kRealBosId));
        auto path = builder.Write("backend-e2b", /*pinned=*/true);
        return Fixture{std::move(builder), std::move(path), options.layers,
                       kRealEosId, kRealBosId};
    }();
    return fixture;
}

const Fixture& E4bFixture() {
    static const Fixture fixture = [] {
        const auto options = gemma4_fixture::E4bOptions();
        auto builder = gemma4_fixture::MakeBuilder(options);
        builder.SetMetadata("tokenizer.ggml.eos_token_id",
                            static_cast<std::uint32_t>(kE4bEosId));
        builder.SetMetadata("tokenizer.ggml.bos_token_id",
                            static_cast<std::uint32_t>(kE4bBosId));
        auto path = builder.Write("backend-e4b", /*pinned=*/true);
        return Fixture{std::move(builder), std::move(path), options.layers,
                       kE4bEosId, kE4bBosId};
    }();
    return fixture;
}

// ---------------------------------------------------------------------------
// A package on disk: the GGUF the catalog names, plus the three JSON files
// the backend cross-validates it against.

/// \brief write tokenizer.json's `model.vocab`, all 262144 entries of it
///
/// Written as TEXT rather than assembled as a json object: the contract check
/// compares `model.vocab`'s size against the GGUF's vocabulary, so a small
/// vocab is not an option -- it would be rejected, and a fixture that omits
/// `model.vocab` entirely would silently skip the eos cross-check that is the
/// whole subject of this file.
void WriteTokenizerJson(const std::filesystem::path& path, int eos_id,
                        const std::string& eos_token, int bos_id,
                        const std::string& bos_token) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    if (eos_id == bos_id)
        throw std::runtime_error("fixture: eos and bos cannot share an id");
    out << "{\"model\":{\"vocab\":{";
    bool first = true;
    for (std::int64_t id = 0; id < flm::gemma4::kVocabularySize; ++id) {
        // Their slots are taken by the two named tokens below.
        if (id == eos_id || id == bos_id) continue;
        if (!first) out << ',';
        first = false;
        out << "\"t" << id << "\":" << id;
    }
    out << ",\"" << eos_token << "\":" << eos_id
        << ",\"" << bos_token << "\":" << bos_id << "}}}";
    if (!out) throw std::runtime_error("write failed for " + path.string());
}

void WriteJson(const std::filesystem::path& path, const nlohmann::json& value) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << value.dump();
    if (!out) throw std::runtime_error("write failed for " + path.string());
}

/// \brief what to write into the package, and what to write WRONG
struct PackageOptions {
    /// the name the GGUF gets inside the package -- this is what tells E2B
    /// and E4B apart on disk, and it is why the backend cannot hold a literal
    std::string gguf_name;
    /// what config.json's top-level `eos_token_id` says; defaults to the
    /// fixture's own
    std::optional<int> config_eos;
    /// what tokenizer.json maps `eos_token` to; defaults to the fixture's own
    std::optional<int> tokenizer_eos;
    /// what config.json's `text_config.num_hidden_layers` says; defaults to
    /// the fixture's own. THE ONE FIELD THAT CAN CATCH AN E2B/E4B SWAP.
    std::optional<std::int64_t> config_layers;
    /// drop config.json entirely
    bool omit_config = false;
    /// write no .gguf into the directory at all
    bool omit_gguf = false;
    /// what config.json's `text_config.bos_token_id` says -- the BOS lives
    /// THERE, not at the top level where the EOS lives; defaults to the
    /// fixture's own
    std::optional<int> config_bos;
    /// what tokenizer.json maps `bos_token` to; defaults to the fixture's own
    std::optional<int> tokenizer_bos;
};

class TempPackage final {
public:
    TempPackage(const Fixture& fixture, PackageOptions options)
        : fixture_(&fixture) {
        static std::uint64_t serial{};
        path_ = std::filesystem::temp_directory_path() /
                ("flm-gemma4-c10-" + std::to_string(++serial));
        std::filesystem::create_directories(path_);
        gguf_name_ = options.gguf_name;

        if (!options.omit_gguf) {
            std::filesystem::create_hard_link(fixture.path, path_ / gguf_name_);
        }

        const int eos = fixture.eos_id;
        const int bos = fixture.bos_id;
        if (!options.omit_config) {
            nlohmann::json config;
            config["architectures"] =
                nlohmann::json::array({"Gemma4ForConditionalGeneration"});
            config["model_type"] = "gemma4";
            config["eos_token_id"] = options.config_eos.value_or(eos);
            // Mirrors the real files, INCLUDING THE INVERSION: the eos the
            // generation path stops on is the TOP-LEVEL one written above,
            // while text_config carries its OWN eos (1, `<eos>`) that is not
            // it -- and the bos is the other way round, stated only inside
            // text_config and absent from the top level. Pinning both here is
            // what keeps the contract check honest about which field it
            // reads for which id.
            config["text_config"] = {
                {"num_hidden_layers", options.config_layers.value_or(fixture.layers)},
                {"vocab_size", flm::gemma4::kVocabularySize},
                {"eos_token_id", 1},
                {"bos_token_id", options.config_bos.value_or(bos)}};
            WriteJson(path_ / "config.json", config);
        }

        WriteTokenizerJson(path_ / "tokenizer.json",
                           options.tokenizer_eos.value_or(eos), kRealEosToken,
                           options.tokenizer_bos.value_or(bos), kRealBosToken);
        // GOOGLE'S OWN SHAPE: it names both tokens and states NEITHER id.
        WriteJson(path_ / "tokenizer_config.json",
                  nlohmann::json{{"eos_token", kRealEosToken},
                                 {"bos_token", kRealBosToken}});
    }

    TempPackage(const TempPackage&) = delete;
    TempPackage& operator=(const TempPackage&) = delete;

    ~TempPackage() {
        std::error_code ignored;
        // Removes the hard link, never the pinned fixture behind it.
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path& path() const { return path_; }
    const Fixture& fixture() const { return *fixture_; }

    /// \brief the catalog entry, as AutoModel hands it to the factory
    nlohmann::ordered_json model_info() const {
        nlohmann::ordered_json info;
        info["details"] = {{"family", kFamily}};
        info["default_context_length"] = 4096;
        info["files"] = nlohmann::ordered_json::array(
            {gguf_name_, "tokenizer.json", "tokenizer_config.json",
             "config.json"});
        return info;
    }

private:
    const Fixture* fixture_;
    std::filesystem::path path_;
    std::string gguf_name_;
};

/// \brief a BackendContext pointing at a package, with the frontend's defaults
struct Request {
    LM_Config config;
    nlohmann::ordered_json info;
    flm::backend::BackendContext context;

    explicit Request(const TempPackage& package,
                     std::uint32_t context_length = 4096)
        : info(package.model_info()) {
        context.model_path = package.path().string();
        context.model_info = info;
        context.config = &config;
        context.context_length = context_length;
        // Gemma4e::load_model calls _shared_load_backend BEFORE
        // setup_tokenizer, so the frontend has nothing parsed to hand down and
        // this really is null on the production path. See the backend's own
        // comment on why it reads the file itself.
        context.tokenizer_config = nullptr;
    }

    /// \brief the same request against a real on-disk package directory
    Request(const std::filesystem::path& root, const std::string& gguf_name,
            std::uint32_t context_length = 4096) {
        info["details"] = {{"family", kFamily}};
        info["default_context_length"] = 4096;
        info["files"] = nlohmann::ordered_json::array(
            {gguf_name, "tokenizer.json", "tokenizer_config.json",
             "config.json"});
        context.model_path = root.string();
        context.model_info = info;
        context.config = &config;
        context.context_length = context_length;
        context.tokenizer_config = nullptr;
    }
};

/// \brief the fake corelib, installed as the PROCESS runtime
///
/// Declare it FIRST in a test's scope so it is destroyed LAST: the backend
/// holds corelib objects, and `ShutdownProcess` THROWS while any are live.
/// A destructor that swallowed that would leave the process runtime in place
/// and every later test would report "corelib runtime already exists" instead
/// of its own result -- one genuine failure followed by a dozen bogus ones.
struct FakeRuntimeScope {
    std::shared_ptr<CorelibRuntime> runtime;

    FakeRuntimeScope() {
        fake_corelib::Reset();
        runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver(), FakeDllPath()));
        baseline_objects = live_objects();
    }
    FakeRuntimeScope(const FakeRuntimeScope&) = delete;
    FakeRuntimeScope& operator=(const FakeRuntimeScope&) = delete;
    ~FakeRuntimeScope() {
        try {
            runtime.reset();
            CorelibRuntime::ShutdownProcess();
        } catch (...) {
        }
    }

    std::size_t live_objects() const {
        return runtime->api()->live_object_count();
    }
    /// \brief true when nothing has been created on the device since this
    ///        runtime was installed -- no stream, no weight, no tensor
    bool nothing_created() const { return live_objects() == baseline_objects; }

    std::size_t baseline_objects{};
};

/// \brief records every file the production code opens, for one scope
struct OpenAudit {
    OpenAudit() {
        flm::file_access::SetOpenObserver(
            [this](const std::filesystem::path& path) {
                names.push_back(path.filename().string());
            });
    }
    OpenAudit(const OpenAudit&) = delete;
    OpenAudit& operator=(const OpenAudit&) = delete;
    ~OpenAudit() { flm::file_access::SetOpenObserver({}); }

    std::vector<std::string> sorted() const {
        auto copy = names;
        std::sort(copy.begin(), copy.end());
        return copy;
    }

    std::vector<std::string> names;
};

std::unique_ptr<flm::backend::ModelBackend> Build(
    const flm::backend::BackendContext& context) {
    return flm::backend::BackendRegistry::instance().create(kFamily, kRai,
                                                            context);
}

// ---------------------------------------------------------------------------
// 1. Traits -- readable with no engine anywhere.

void TestTraitsAreReadableWithoutConstructingAnEngine() {
    const auto traits = flm::gemma4::rai_traits();
    TEST_REQUIRE(!traits.needs_npu_xclbin);
    TEST_REQUIRE(!traits.supports_preemption);
    TEST_REQUIRE(traits.max_context_length == 16384);
    // Nothing above touched corelib, a GGUF or the filesystem, which is the
    // property that lets the frontend consult it while it is still ASSEMBLING
    // the BackendContext -- before any engine exists to ask.
}

void TestTraitsComeBackOutOfTheRegistryUnchanged() {
    const auto traits =
        flm::backend::BackendRegistry::instance().traits(kFamily, kRai);
    TEST_REQUIRE(!traits.needs_npu_xclbin);
    TEST_REQUIRE(!traits.supports_preemption);
    TEST_REQUIRE(traits.max_context_length ==
                 flm::gemma4::rai_traits().max_context_length);
}

void TestTheTraitsCeilingIsTheEnginesOwnLimit() {
    // The frontend rejects an over-long context against the TRAITS. The
    // engine accepts the same ceiling: 1..4096 unpaged, or whole pages up to
    // kMaxPagedContext.
    TEST_REQUIRE(flm::gemma4::kRaiContextLimit ==
                 static_cast<std::uint32_t>(flm::gemma4::kMaxPagedContext));
    TEST_REQUIRE(flm::gemma4::kRaiDecodeLimit + 1 ==
                 flm::gemma4::kRaiContextLimit);
}

void TestContextFitsStopsAtTheValidated16384() {
    TEST_REQUIRE(flm::gemma4::ContextFits(4096));
    TEST_REQUIRE(flm::gemma4::ContextFits(8192));
    TEST_REQUIRE(flm::gemma4::ContextFits(16384));
    TEST_REQUIRE(!flm::gemma4::ContextFits(10000));
    TEST_REQUIRE(!flm::gemma4::ContextFits(20480));
    TEST_REQUIRE(!flm::gemma4::ContextFits(131072));
}

// ---------------------------------------------------------------------------
// 2. Everything is refused before anything is opened or allocated.

void TestPreemptionIsRejectedBeforeAnythingIsOpened() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    OpenAudit audit;
    Request request(package);
    request.context.enable_preemption = true;
    RequireContains(RequireThrows([&] { (void)Build(request.context); }),
                    "preemption");
    // Not merely "no device object": no FILE either. A backend that read the
    // package first and refused afterwards would still pass a
    // live-object-count check.
    TEST_REQUIRE(audit.names.empty());
    TEST_REQUIRE(runtime.nothing_created());
}

void TestAnOverLongContextIsRejectedBeforeAnythingIsOpened() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    OpenAudit audit;
    Request request(package, /*context_length=*/4097);
    const auto error = RequireThrows([&] { (void)Build(request.context); });
    RequireContains(error, "4096");
    TEST_REQUIRE(audit.names.empty());
    TEST_REQUIRE(runtime.nothing_created());
}

void TestAContextPastThePagedCeilingIsRejectedBeforeAnythingIsOpened() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    OpenAudit audit;
    Request request(package, /*context_length=*/20480);
    RequireContains(RequireThrows([&] { (void)Build(request.context); }),
                    "16384");
    TEST_REQUIRE(audit.names.empty());
    TEST_REQUIRE(runtime.nothing_created());
}

void TestAZeroContextIsRejectedBeforeAnythingIsOpened() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    OpenAudit audit;
    Request request(package, /*context_length=*/0);
    RequireContains(RequireThrows([&] { (void)Build(request.context); }),
                    "4096");
    TEST_REQUIRE(audit.names.empty());
    TEST_REQUIRE(runtime.nothing_created());
}

void TestAMismatchedConfigJsonIsRejectedAndNoDeviceObjectIsCreated() {
    // num_hidden_layers is the one cross-checked field that can catch an
    // E2B/E4B swap, so this is the shape of the mismatch that matters.
    TempPackage package(E2bFixture(),
                        {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt, std::nullopt,
                         /*config_layers=*/42});
    FakeRuntimeScope runtime;
    Request request(package);
    const auto error = RequireThrows([&] { (void)Build(request.context); });
    RequireContains(error, "num_hidden_layers");
    // THE ASSERTION THE BRIEF ASKS FOR: the package was diagnosed at load
    // with nothing allocated. `live_object_count` counts every corelib object
    // this process holds -- the stream included -- so this says "no device
    // object", not merely "no weights".
    TEST_REQUIRE(runtime.nothing_created());
}

/// \brief sets an environment variable for one scope and puts it back
struct ScopedEnv {
    ScopedEnv(const char* name, const std::string& value) : name_(name) {
        if (const char* existing = std::getenv(name)) {
            previous_ = existing;
            had_previous_ = true;
        }
        Set(value);
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
    ~ScopedEnv() { Set(had_previous_ ? previous_ : std::string()); }

private:
    void Set(const std::string& value) {
#ifdef _WIN32
        _putenv_s(name_, value.c_str());
#else
        if (value.empty()) unsetenv(name_);
        else setenv(name_, value.c_str(), 1);
#endif
    }
    const char* name_;
    std::string previous_;
    bool had_previous_{};
};

void TestABadPackageIsDiagnosedBeforeTheRuntimeIsAcquired() {
    // THE ORDERING TEST, AND IT DELIBERATELY INSTALLS NO FAKE RUNTIME.
    //
    // With one installed, `CorelibRuntime::GetOrCreate` hands back the
    // existing process runtime and creates nothing, so a backend that called
    // it FIRST would leave no trace and every "nothing was created" assertion
    // would still hold. Without one, GetOrCreate has to load a DLL -- and
    // FLM_RAI_CORELIB_PATH below guarantees it cannot. So the two orders
    // produce two DIFFERENT error messages, and which one comes back says
    // which ran first.
    const auto missing_dll =
        std::filesystem::absolute(std::filesystem::temp_directory_path() /
                                  "flm-gemma4-no-such-corelib.dll");
    ScopedEnv corelib_path("FLM_RAI_CORELIB_PATH", missing_dll.string());
    TempPackage package(E2bFixture(),
                        {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt, std::nullopt,
                         /*config_layers=*/42});
    Request request(package);
    const auto error = RequireThrows([&] { (void)Build(request.context); });
    RequireContains(error, "num_hidden_layers");
    TEST_REQUIRE(error.find(missing_dll.string()) == std::string::npos);
    TEST_REQUIRE(error.find("corelib") == std::string::npos);

    // And no process-wide runtime was left behind: CreateForTest throws
    // "corelib runtime already exists" if one is installed, so its succeeding
    // is the proof.
    FakeRuntimeScope runtime;
    TEST_REQUIRE(runtime.nothing_created());
}

void TestAMissingConfigJsonIsNamed() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt,
                                       std::nullopt, std::nullopt,
                                       /*omit_config=*/true});
    FakeRuntimeScope runtime;
    Request request(package);
    RequireContains(RequireThrows([&] { (void)Build(request.context); }),
                    "config.json");
    TEST_REQUIRE(runtime.nothing_created());
}

void TestANullConfigIsRejectedBeforeAnythingIsOpened() {
    // MUT9 in review-C10.md: deleting `if (!context.config)` turned nothing
    // red, because no test ever passed a null one. The frontend always
    // supplies an LM_Config, so the guard is a second line for a caller that
    // reaches the factory directly -- and without it the null is carried all
    // the way to `*context.config` at the engine constructor, where it is a
    // dereference of nullptr rather than a diagnosis.
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    OpenAudit audit;
    Request request(package);
    request.context.config = nullptr;
    RequireContains(RequireThrows([&] { (void)Build(request.context); }),
                    "LM_Config");
    TEST_REQUIRE(audit.names.empty());
    TEST_REQUIRE(runtime.nothing_created());
}

void TestAPackageWithNoGgufIsNamedRatherThanGuessedAt() {
    TempPackage package(E2bFixture(),
                        {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt, std::nullopt,
                         std::nullopt, /*omit_config=*/false,
                         /*omit_gguf=*/true});
    FakeRuntimeScope runtime;
    Request request(package);
    const auto error = RequireThrows([&] { (void)Build(request.context); });
    RequireContains(error, "gemma-4-E2B-it-Q8_0.gguf");
    TEST_REQUIRE(runtime.nothing_created());
}

// ---------------------------------------------------------------------------
// 3. The six overrides.

/// \brief assert the whole policy of one loaded backend
/// \note Seven now, not six: forced_bos_id() joined the set when C-1 showed
///       that a package stating `bos_token` and no `bos_token_id` -- which is
///       every real Gemma 4 package -- killed the process in the frontend.
void RequireTheSevenOverrides(flm::backend::ModelBackend& backend,
                              const std::vector<int>& expected_eos,
                              int expected_bos) {
    TEST_REQUIRE(backend.id() == std::string(flm::backend::kRaiBackendId));
    TEST_REQUIRE(backend.id() == std::string("rai"));
    // The provenance line in `flm show`: the library the runtime ACTUALLY
    // loaded, not a path recomputed from the environment afterwards.
    TEST_REQUIRE(backend.detail() == FakeDllPath().string());
    TEST_REQUIRE(backend.max_decode_length() == 4095);
    TEST_REQUIRE(!backend.supports_preemption());
    // corelib refuses a decode past its own window, so the extra forward()
    // the FastFlowLM engines want after an EOS token would fail here.
    TEST_REQUIRE(!backend.forwards_past_eos());
    TEST_REQUIRE(!backend.poisoned());
    const auto eos = backend.forced_eos_ids();
    TEST_REQUIRE(eos.has_value());
    TEST_REQUIRE(*eos == expected_eos);
    // The seventh. Without it AutoModel::_shared_setup_tokenizer exits the
    // process on any package that names `bos_token` without stating an id --
    // which is what both shipped Google packages do.
    const auto bos = backend.forced_bos_id();
    TEST_REQUIRE(bos.has_value());
    TEST_REQUIRE(*bos == expected_bos);
}

void TestTheSevenOverridesOnASyntheticE2bPackage() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    RequireTheSevenOverrides(*backend, {kRealEosId}, kRealBosId);
    TEST_REQUIRE(runtime.live_objects() > runtime.baseline_objects);
}

void TestTheSevenOverridesOnASyntheticE4bPackage() {
    // E4B is not a second copy of E2B: 42 layers, hidden 2560, two KV heads,
    // a different full-attention period and a different shared-cache pair --
    // and, in this binary, a different EOS id and a different BOS id.
    TempPackage package(E4bFixture(), {"gemma-4-E4B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    RequireTheSevenOverrides(*backend, {kE4bEosId}, kE4bBosId);
}

void TestTheBackendOwnsTheEngineItReportsOn() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    // clear_context() ran in the constructor, so the engine is at position 0
    // and usable without the caller doing anything first.
    TEST_REQUIRE(backend->engine().get_current_context_length() == 0);
    TEST_REQUIRE(&backend->engine() == &backend->engine());
    TEST_REQUIRE(dynamic_cast<flm::gemma4::gemma4_rai*>(&backend->engine()) !=
                 nullptr);
}

// ---------------------------------------------------------------------------
// 4. forced_eos_ids() -- derived, and cross-validated across three sources.

void TestForcedEosIdsAreReadFromTheFileNotHardcoded() {
    // The E4B fixture says 4242 and all three of its sources agree on 4242.
    // A backend that returned the literal 106 -- which is what every real
    // package says, and therefore what a suite of 106-only fixtures cannot
    // distinguish from a derivation -- fails here and only here.
    TempPackage package(E4bFixture(), {"gemma-4-E4B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    const auto eos = backend->forced_eos_ids();
    TEST_REQUIRE(eos.has_value());
    TEST_REQUIRE(*eos == std::vector<int>{kE4bEosId});
    TEST_REQUIRE((*eos)[0] != kRealEosId);
}

void TestConfigJsonDisagreeingAboutEosRejectsThePackage() {
    // Source 2 of 3. If this did not reject, "cross-validated" would be a
    // word rather than a check, and forced_eos_ids() would rest on the GGUF
    // alone.
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf",
                                       /*config_eos=*/4242});
    FakeRuntimeScope runtime;
    const auto error =
        RequireThrows([&] { (void)Build(Request(package).context); });
    RequireContains(error, "eos_token_id");
    RequireContains(error, "4242");
    TEST_REQUIRE(runtime.nothing_created());
}

void TestTokenizerDisagreeingAboutEosRejectsThePackage() {
    // Source 3 of 3: tokenizer_config.json names the token, tokenizer.json
    // says what id that token has. Moving the id there alone is enough.
    TempPackage package(E2bFixture(),
                        {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt,
                         /*tokenizer_eos=*/4242});
    FakeRuntimeScope runtime;
    const auto error =
        RequireThrows([&] { (void)Build(Request(package).context); });
    RequireContains(error, "eos_token");
    TEST_REQUIRE(runtime.nothing_created());
}

// ---------------------------------------------------------------------------
// 4b. forced_bos_id() -- the same three sources, for the id the real package
//     does not state at all.

void TestForcedBosIdIsReadFromTheFileNotHardcoded() {
    // The E4B fixture says 4343 and all three of its sources agree on 4343.
    // A backend that returned the literal 2 -- which is what every real
    // package says, and therefore what a suite of 2-only fixtures cannot
    // distinguish from a derivation -- fails here and only here.
    TempPackage package(E4bFixture(), {"gemma-4-E4B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    const auto bos = backend->forced_bos_id();
    TEST_REQUIRE(bos.has_value());
    TEST_REQUIRE(*bos == kE4bBosId);
    TEST_REQUIRE(*bos != kRealBosId);
}

void TestConfigJsonDisagreeingAboutBosRejectsThePackage() {
    // Source 2 of 3, read out of `text_config` -- NOT the top level, which is
    // where the eos lives and where no real file states a bos at all. A check
    // that looked at the top level would find nothing and pass silently,
    // which is the failure this test exists to make impossible.
    TempPackage package(E2bFixture(),
                        {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt, std::nullopt,
                         std::nullopt, /*omit_config=*/false,
                         /*omit_gguf=*/false, /*config_bos=*/4343});
    FakeRuntimeScope runtime;
    const auto error =
        RequireThrows([&] { (void)Build(Request(package).context); });
    RequireContains(error, "bos_token_id");
    RequireContains(error, "4343");
    TEST_REQUIRE(runtime.nothing_created());
}

void TestTokenizerDisagreeingAboutBosRejectsThePackage() {
    // Source 3 of 3: tokenizer_config.json names `<bos>`, tokenizer.json says
    // what id that name has.
    TempPackage package(E2bFixture(),
                        {"gemma-4-E2B-it-Q8_0.gguf", std::nullopt, std::nullopt,
                         std::nullopt, /*omit_config=*/false,
                         /*omit_gguf=*/false, /*config_bos=*/std::nullopt,
                         /*tokenizer_bos=*/4343});
    FakeRuntimeScope runtime;
    const auto error =
        RequireThrows([&] { (void)Build(Request(package).context); });
    RequireContains(error, "bos_token");
    TEST_REQUIRE(runtime.nothing_created());
}

// ---------------------------------------------------------------------------
// 5. Which files are opened, and where the GGUF's name comes from.

void TestTheBackendOpensExactlyTheFourPackageFiles() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    OpenAudit audit;
    auto backend = Build(Request(package).context);
    TEST_REQUIRE(audit.sorted() ==
                 std::vector<std::string>({"config.json",
                                           "gemma-4-E2B-it-Q8_0.gguf",
                                           "tokenizer.json",
                                           "tokenizer_config.json"}));
    // No manifest, no ONNX, no converted-weight directory, no cache: this
    // path loads a GGUF package and nothing else.
    for (const auto& name : audit.names) {
        TEST_REQUIRE(name.find("manifest") == std::string::npos);
        TEST_REQUIRE(name.find("onnx") == std::string::npos);
        TEST_REQUIRE(name.find("cache") == std::string::npos);
    }
}

void TestTheFrontendsTokenizerConfigIsUsedWhenItIsSupplied() {
    // Gemma4e does not supply one today (it sets its tokenizer up AFTER the
    // backend is built), which is why the backend reads the file. If a
    // frontend ever does supply one, the file must not be opened twice.
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    const nlohmann::json tokenizer_config{{"eos_token", kRealEosToken},
                                          {"bos_token", "<bos>"}};
    OpenAudit audit;
    Request request(package);
    request.context.tokenizer_config = &tokenizer_config;
    auto backend = Build(request.context);
    TEST_REQUIRE(audit.sorted() ==
                 std::vector<std::string>({"config.json",
                                           "gemma-4-E2B-it-Q8_0.gguf",
                                           "tokenizer.json"}));
    const auto eos = backend->forced_eos_ids();
    TEST_REQUIRE(eos.has_value() && *eos == std::vector<int>{kRealEosId});
}

void TestTheGgufNameComesFromTheCatalogEntryNotAConstant() {
    // Phi-4 can hold a literal filename because one file serves the whole
    // family. Gemma 4 cannot: E2B and E4B ship DIFFERENT names under one
    // family, so a literal loads exactly one of the two. Both names below are
    // resolved from the same code, against the same fixture, so the only
    // thing under test is where the name came from.
    for (const char* name :
         {"gemma-4-E2B-it-Q8_0.gguf", "some-other-conversion.Q8_0.gguf"}) {
        TempPackage package(E2bFixture(), {name});
        FakeRuntimeScope runtime;
        auto backend = Build(Request(package).context);
        TEST_REQUIRE(backend->id() == std::string(kRai));
    }
}

void TestACatalogEntryNamingNoGgufFallsBackToTheDirectory() {
    // A model directory assembled by hand has no catalog "files" array. One
    // GGUF in the directory is unambiguous; the scan is what covers that.
    // The file is deliberately named nothing a backend could have guessed.
    TempPackage package(E2bFixture(), {"hand-assembled.Q8_0.gguf"});
    FakeRuntimeScope runtime;
    Request request(package);
    request.context.model_info.erase("files");
    auto backend = Build(request.context);
    TEST_REQUIRE(backend->forced_eos_ids().has_value());
}

// ---------------------------------------------------------------------------
// 6. poisoned() forwards.

void TestPoisonedForwardsFromTheEngine() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    TEST_REQUIRE(!backend->poisoned());

    // A dispatch refused MID-CHAIN, i.e. after one was already accepted --
    // C9's definition of the only condition that poisons. Index 20 is well
    // inside layer 1, not the first dispatch of the pass.
    fake_corelib::GetState().fail_dispatch_at = 20;
    std::vector<int> ids{1, 2, 3};
    (void)RequireThrows([&] { (void)backend->engine().prefill(ids); });

    // FORWARDED, not recomputed: the engine is the thing that knows, and
    // AutoModel::_shared_guard_poisoned asks the BACKEND. A backend that left
    // poisoned() at ModelBackend's `false` default would turn a model that
    // needs a reload into one that keeps answering 200 with garbage.
    auto* engine = dynamic_cast<flm::gemma4::gemma4_rai*>(&backend->engine());
    TEST_REQUIRE(engine != nullptr);
    TEST_REQUIRE(engine->poisoned());
    TEST_REQUIRE(backend->poisoned());
}

void TestAFirstDispatchRefusalDoesNotPoison() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    auto backend = Build(Request(package).context);
    fake_corelib::GetState().fail_dispatch_at = 0;
    std::vector<int> ids{1, 2, 3};
    (void)RequireThrows([&] { (void)backend->engine().prefill(ids); });
    TEST_REQUIRE(!backend->poisoned());
}

// ---------------------------------------------------------------------------
// 7. Teardown order -- structural, because the fault it prevents is silent.

/// \brief the source with its comments removed
///
/// A STRUCTURAL GATE THAT A COMMENT CAN TRIP IS A GATE THAT GETS REWORDED
/// RATHER THAN OBEYED. The destructor's own comment explains why there is no
/// enumerated teardown and has to be able to say `runtime_.reset()` while
/// doing the opposite; searching the raw text made that comment fail the
/// test, which would have taught the next reader to delete the explanation.
/// Tracks double-quoted strings so a literal containing a slash pair does not
/// swallow the rest of the file.
std::string StripComments(const std::string& source) {
    std::string out;
    out.reserve(source.size());
    enum class In { Code, Line, Block, Text } state = In::Code;
    for (std::size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        const char next = i + 1 < source.size() ? source[i + 1] : '\0';
        switch (state) {
            case In::Code:
                if (c == '/' && next == '/') { state = In::Line; ++i; }
                else if (c == '/' && next == '*') { state = In::Block; ++i; }
                else { if (c == '"') state = In::Text; out.push_back(c); }
                break;
            case In::Line:
                if (c == '\n') { state = In::Code; out.push_back(c); }
                break;
            case In::Block:
                if (c == '*' && next == '/') { state = In::Code; ++i; }
                else if (c == '\n') out.push_back(c);
                break;
            case In::Text:
                out.push_back(c);
                if (c == '\\') { if (next) { out.push_back(next); ++i; } }
                else if (c == '"') state = In::Code;
                break;
        }
    }
    return out;
}

std::string ReadSource(const char* path) {
    std::ifstream input(path, std::ios::binary);
    // A structural gate that SKIPS when it cannot find its subject is worse
    // than none: it reports green over nothing.
    if (!input) {
        throw std::runtime_error(
            "cannot open " + std::string(path) +
            " -- a structural gate reads this source and cannot run without "
            "it");
    }
    std::ostringstream text;
    text << input.rdbuf();
    return StripComments(text.str());
}

std::string ReadBackendSource() {
    return ReadSource(FLM_GEMMA4_RAI_BACKEND_SOURCE);
}

/// \brief one free function's body, from its signature to its closing brace
///
/// A gate that searched from a signature to the END OF THE FILE would find
/// any other function's use of the same call and report green over a
/// function that does not make it. This one stops at the first `}` in column
/// 1 after the signature, which is where a top-level definition ends in this
/// tree's style. THROWS rather than returning the tail when it cannot find
/// either end, for the same reason ReadSource throws: a structural gate that
/// quietly degrades is worse than no gate.
std::string FunctionBody(const std::string& source, const std::string& signature) {
    const auto start = source.find(signature);
    if (start == std::string::npos) {
        throw std::runtime_error("no definition of " + signature +
                                 " -- this gate cannot run over nothing");
    }
    const auto end = source.find("\n}", start);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated body for " + signature);
    }
    return source.substr(start, end - start);
}

void TestFunctionBodyStopsAtTheFunctionItNames() {
    // The gate below is only as good as this: a body extractor that ran on
    // to the end of the file would make the gate pass over the WRONG
    // function. Both halves are checked -- that it stops, and that it throws
    // rather than degrading when its subject is absent.
    const std::string source =
        "void a() {\n    wanted();\n}\n\nvoid b() {\n    unwanted();\n}\n";
    const auto body = FunctionBody(source, "void a()");
    TEST_REQUIRE(body.find("wanted()") != std::string::npos);
    TEST_REQUIRE(body.find("unwanted()") == std::string::npos);
    RequireContains(RequireThrows([&] { (void)FunctionBody(source, "void c()"); }),
                    "void c()");
}

void TestStripCommentsDoesNotEatCode() {
    // The gate below is only as good as this. A stripper that dropped a
    // declaration would make both structural tests pass over nothing.
    TEST_REQUIRE(StripComments("int a; // gone\nint b;") == "int a; \nint b;");
    TEST_REQUIRE(StripComments("int a; /* gone */ int b;") == "int a;  int b;");
    TEST_REQUIRE(StripComments("auto s = \"// not a comment\"; int b;") ==
                 "auto s = \"// not a comment\"; int b;");
    // And it really does remove the destructor comment's own mention.
    TEST_REQUIRE(
        StripComments("// runtime_.reset()\nengine_.reset();").find(
            "runtime_.reset()") == std::string::npos);
}

void TestTheRuntimeMemberIsDeclaredBeforeTheEngineMember() {
    const auto source = ReadBackendSource();
    const auto runtime_decl =
        source.find("std::shared_ptr<corelib::CorelibRuntime> runtime_;");
    const auto engine_decl =
        source.find("std::unique_ptr<gemma4_rai> engine_;");
    TEST_REQUIRE(runtime_decl != std::string::npos);
    TEST_REQUIRE(engine_decl != std::string::npos);
    // Members are destroyed in REVERSE declaration order, so declaring the
    // runtime first is what destroys the engine first. corelib
    // access-violates when a packed object outlives the API it came from, and
    // the fault is intermittent and raises nothing -- 7 of 10 measured runs,
    // process still exiting 0 -- so no behavioural test can catch a reorder.
    TEST_REQUIRE(runtime_decl < engine_decl);
}

void TestTheConstructorClearsTheEngineBeforeItPublishesIt() {
    // MUT5 in review-C10.md: deleting `engine->clear_context()` from the
    // constructor turned NOTHING red, because today's engine constructor
    // already leaves position 0 and no checkpoint (gemma4_rai.cpp's Impl
    // zero-initialises both). So the contract below is true either way --
    // it is asserted anyway, because it is what callers rely on and what
    // would start being false first...
    {
        TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
        FakeRuntimeScope runtime;
        auto backend = Build(Request(package).context);
        TEST_REQUIRE(backend->engine().get_current_context_length() == 0);
        // clear_context() also drops any saved checkpoint, and restore()
        // answering -1 is the only way to see that from outside.
        TEST_REQUIRE(backend->engine().restore() == -1);
    }

    // ...and the CALL is pinned structurally, for the same reason C9 pinned
    // the teardown that way: the property is "a future engine that stopped
    // zero-initialising must not silently hand out a dirty context", and no
    // behavioural assertion against TODAY's engine can see it. Ordering is
    // part of it -- clearing after publishing would leave a window where
    // engine_ is reachable and unclear.
    const auto source = ReadBackendSource();
    const auto constructed = source.find("std::make_unique<gemma4_rai>");
    const auto cleared = source.find("engine->clear_context();");
    const auto published = source.find("engine_ = std::move(engine);");
    TEST_REQUIRE(constructed != std::string::npos);
    TEST_REQUIRE(cleared != std::string::npos);
    TEST_REQUIRE(published != std::string::npos);
    TEST_REQUIRE(constructed < cleared);
    TEST_REQUIRE(cleared < published);
}

void TestTheDestructorDoesNotEnumerateTheTeardown() {
    const auto source = ReadBackendSource();
    // C9's rule: declaration order, not a hand-written release list. A
    // destructor BODY runs before member destruction, so an explicit
    // `runtime_.reset()` there would release the runtime AHEAD of anything
    // the list forgot -- exactly the order corelib faults on.
    TEST_REQUIRE(source.find("runtime_.reset()") == std::string::npos);
}

void TestDestroyingTheBackendReleasesEveryCorelibObject() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    {
        auto backend = Build(Request(package).context);
        TEST_REQUIRE(runtime.live_objects() > runtime.baseline_objects);
    }
    TEST_REQUIRE(runtime.nothing_created());
    // And the stream went LAST among them, which is the order corelib
    // requires. The engine's own C9 gate proves it for the engine; this
    // proves the backend does not disturb it.
    const auto& order = fake_corelib::GetState().release_order;
    const auto stream = std::find(order.begin(), order.end(), "stream");
    TEST_REQUIRE(stream != order.end());
    TEST_REQUIRE(std::find(stream, order.end(), "matmul_weights") == order.end());
    TEST_REQUIRE(std::find(stream, order.end(), "ple_weights") == order.end());
}

// ---------------------------------------------------------------------------
// 8. Registration, exactly as builtin_backends.cpp makes it.

void TestGemma4eHasARaiBackendRegistered() {
    auto& registry = flm::backend::BackendRegistry::instance();
    TEST_REQUIRE(registry.has(kFamily, kRai));
    const auto ids = registry.available(kFamily);
    TEST_REQUIRE(std::find(ids.begin(), ids.end(), std::string(kRai)) !=
                 ids.end());
}

void TestResolveBackendIdPicksRaiForGemma4e() {
    std::string source;
    TEST_REQUIRE(flm::backend::resolve_backend_id(
                     kFamily, flm::backend::kRaiBackendId, "", &source) == kRai);
    TEST_REQUIRE(flm::backend::resolve_backend_id(kFamily, "flm", kRai) == kRai);
}

void TestAnUnknownBackendIdForGemma4eNamesTheRealOne() {
    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    Request request(package);
    const auto error = RequireThrows([&] {
        (void)flm::backend::BackendRegistry::instance().create(
            kFamily, "bogus", request.context);
    });
    RequireContains(error, "bogus");
    RequireContains(error, kRai);
    TEST_REQUIRE(runtime.nothing_created());
}

// ---------------------------------------------------------------------------
// 9. The two tripwires D1 left for this task.

/// \brief the shipped catalog as a build that links both kernel flows sees it
model_list OpenCatalog(const std::string& platform) {
    std::string path = FLM_TEST_MODEL_LIST_PATH;
    std::string exe_dir = ".";
    return model_list(path, exe_dir, platform, {"flm", "rai"});
}

void TestTripwireOneTheAieNextFallbackIsGemma4AndItsFamilyCanLoad() {
    // D1 flagged this and said to re-check it once a rai backend existed for
    // gemma4e. fallback_model() returns the first entry in SORTED KEY order
    // (nlohmann::json's object type is std::map), llama3.2 is pruned on
    // aie_next, and "gemma4-it-rai" < "phi4-mini-it-rai". So an unresolvable
    // tag lands on gemma4-it-rai:e2b.
    auto catalog = OpenCatalog("aie_next");
    const auto [tag, info] = catalog.get_model_info("bogus:9b");
    TEST_REQUIRE(tag == "gemma4-it-rai:e2b");
    TEST_REQUIRE(std::string("gemma4-it-rai") < std::string("phi4-mini-it-rai"));

    // THE PART THAT WAS FALSE BEFORE THIS TASK. D1's report: "BackendRegistry
    // ::create(\"gemma4e\", ...) has no rai factory yet, so that load would
    // throw naming the family's available ids (currently none)."
    const auto family = info.at("details").at("family").get<std::string>();
    TEST_REQUIRE(family == kFamily);
    auto& registry = flm::backend::BackendRegistry::instance();
    TEST_REQUIRE(registry.has(family, kRai));
    const auto traits = registry.traits(family, kRai);
    const auto default_context =
        info.at("default_context_length").get<std::uint32_t>();
    TEST_REQUIRE(default_context <= traits.max_context_length);
    TEST_REQUIRE(flm::gemma4::ContextFits(default_context));
}

void TestTripwireOneAnUnresolvableTagStillLoadsRatherThanThrowing() {
    // Not merely "a factory is registered": the entry the fallback returns is
    // actually taken through that factory, with the context length the
    // catalog gives it, and a backend comes out.
    auto catalog = OpenCatalog("aie_next");
    const auto [tag, info] = catalog.get_model_info("bogus:9b");
    const auto family = info.at("details").at("family").get<std::string>();
    const auto context_length =
        info.at("default_context_length").get<std::uint32_t>();

    TempPackage package(E2bFixture(), {"gemma-4-E2B-it-Q8_0.gguf"});
    FakeRuntimeScope runtime;
    Request request(package, context_length);
    // The catalog's own file list for this entry, rather than the package
    // helper's: the fallback path has to work with what model_list hands it.
    request.context.model_info["files"] = info.at("files");
    request.context.model_info["details"] = info.at("details");
    auto backend = flm::backend::BackendRegistry::instance().create(
        family, kRai, request.context);
    TEST_REQUIRE(backend->id() == std::string(kRai));
    TEST_REQUIRE(backend->max_decode_length() == context_length - 1);
}

void TestTripwireOneStxIsUnaffected() {
    // The same lookup on aie2p (Strix and its siblings) still reaches
    // llama3.2:1b, so nothing about the shipped default moved for the
    // platform that has one.
    auto catalog = OpenCatalog("aie2p");
    const auto [tag, info] = catalog.get_model_info("bogus:9b");
    TEST_REQUIRE(tag == "llama3.2:1b");
    // AND THE FAMILY IS THE ONE THE DELETED LITERAL ASSUMED. get_auto_model
    // used to answer an unsupported tag with the hardcoded pair
    // ("llama3.2:1b", Llama3); it now takes both from this resolution
    // instead. On aie2p that yields llama3.2:1b and family llama3 -- the same
    // pair -- which is the whole claim that C-2's fix moved nothing for the
    // platform every other family runs on.
    TEST_REQUIRE(info.at("details").at("family") == "llama3");
}

void TestAPrefixedTagResolvesToItsOwnFamilyRatherThanTheDefault() {
    // The behaviour change C-2 DOES make on aie2p, stated rather than
    // discovered later. "Ollama/qwen3:4b" is not itself a catalog tag, so it
    // missed get_auto_model's is_model_supported check and was answered with
    // the hardcoded ("llama3.2:1b", Llama3) pair -- the wrong model, served
    // successfully, with only a log line to say so. cut_tag exists precisely
    // to accept this spelling, and get_model_info has always resolved it
    // correctly; it was only the frontend choice that ignored the result.
    auto catalog = OpenCatalog("aie2p");
    TEST_REQUIRE(!catalog.is_model_supported("Ollama/qwen3:4b"));
    const auto [tag, info] = catalog.get_model_info("Ollama/qwen3:4b");
    TEST_REQUIRE(tag == "qwen3:4b");
    TEST_REQUIRE(info.at("details").at("family") == "qwen3");
}

void TestTripwireTwoTheServerFallbackTagIsItselfUnsupportedOnAieNext() {
    // The second tripwire. D1 looked at RestHandler and concluded the path
    // "looks unreachable today" because main.cpp gates an unsupported CLI tag
    // first. Re-checked now: the gate that matters is get_auto_model's, and
    // what it does on a miss is return the LITERAL "llama3.2:1b" -- the same
    // literal RestHandler's constructor substitutes. On aie_next that tag is
    // itself unsupported, so model_list::get_model_info falls through to
    // fallback_model() and hands back gemma4-it-rai:e2b's entry.
    auto catalog = OpenCatalog("aie_next");
    TEST_REQUIRE(!catalog.is_model_supported("llama3.2:1b"));
    const auto [tag, info] = catalog.get_model_info("llama3.2:1b");
    TEST_REQUIRE(tag == "gemma4-it-rai:e2b");
    // Which family that entry names is now a family with a rai backend, so
    // the registry lookup on this path resolves instead of throwing.
    TEST_REQUIRE(info.at("details").at("family") == kFamily);
    TEST_REQUIRE(flm::backend::BackendRegistry::instance().has(kFamily, kRai));
    TEST_REQUIRE(!catalog.is_model_supported("llama3.2"));
}

// ---------------------------------------------------------------------------
// 9b. C-2 -- an unknown model name must not reach that fallback chain at all.
//
// The chain above is only dangerous because arbitrary user input can enter
// it: /api/generate, /api/chat and rest_handler.cpp:1156 passed
// request["model"] straight to ensure_model_loaded with no check, and
// get_auto_model answered an unknown tag with the LITERAL "llama3.2:1b" plus
// a Llama3 frontend -- a tag that is itself pruned on aie_next, so the
// catalog then resolved it to gemma4-it-rai:e2b and paired a Llama-3 chat
// template with the Gemma 4 engine. Two things had to change and both are
// asserted here: the server refuses the name, and get_auto_model no longer
// picks a frontend from a literal it has not resolved.
//
// Neither production function can be LINKED here -- get_auto_model includes
// every frontend in the tree and ensure_model_loaded is the whole server --
// so what this binary can say about them is (a) the behaviour of the
// predicate the gate is built from, measured against the SHIPPED catalog on
// both platforms, and (b) structurally, that the production code is built
// from it. Both halves are needed: the predicate alone says nothing about
// where it is used, and the gate alone says nothing about what it admits.

/// \brief the predicate RestHandler::ensure_model_loaded gates on
/// \note Composed here from the same two public model_list calls the
///       production line composes. rectify_model_tag is the tolerant half:
///       it strips an "Ollama/" style prefix and fills in a default size,
///       exactly as get_model_info does, so the gate admits every spelling
///       that used to resolve and rejects only the ones that did not.
bool ServerWouldAccept(model_list& catalog, const std::string& tag) {
    return catalog.is_model_supported(catalog.rectify_model_tag(tag));
}

void TestTheServerGateAdmitsEveryShippedTagOnBothPlatforms() {
    // THE REGRESSION TEST FOR EVERY FAMILY, NOT JUST GEMMA 4. A gate that
    // rejected a tag the server used to serve would be a worse outcome than
    // the bug it closes, so this walks the whole catalog on both platforms.
    for (const char* platform : {"aie2p", "aie_next"}) {
        auto catalog = OpenCatalog(platform);
        TEST_REQUIRE(!catalog.all_tags.empty());
        for (const auto& tag : catalog.all_tags) {
            TEST_REQUIRE(ServerWouldAccept(catalog, tag));
            // The Ollama-style prefix cut_tag exists for. Before the gate
            // these reached get_auto_model, missed is_model_supported, and
            // were served as llama3.2:1b -- i.e. the WRONG MODEL, quietly.
            TEST_REQUIRE(ServerWouldAccept(catalog, "Ollama/" + tag));
        }
    }
}

void TestTheServerGateRejectsNamesThatUsedToBeSubstituted() {
    for (const char* platform : {"aie2p", "aie_next"}) {
        auto catalog = OpenCatalog(platform);
        // A typo, a real family with a size that does not exist, and the
        // empty string. Each one used to be answered with a substituted
        // model on aie2p and, after C10 registered a backend for gemma4e, with
        // a dead server process on aie_next.
        TEST_REQUIRE(!ServerWouldAccept(catalog, "typo"));
        TEST_REQUIRE(!ServerWouldAccept(catalog, "gemma4-it:e9z"));
        TEST_REQUIRE(!ServerWouldAccept(catalog, ""));
    }
    // And the one that mattered: on aie_next the historical default tag is
    // itself pruned, which is what turned a typo into a Gemma 4 load.
    auto aie_next = OpenCatalog("aie_next");
    TEST_REQUIRE(!ServerWouldAccept(aie_next, "llama3.2:1b"));
    auto aie2p = OpenCatalog("aie2p");
    TEST_REQUIRE(ServerWouldAccept(aie2p, "llama3.2:1b"));
}

void TestTheServerGatesAnUnknownModelNameBeforeItResolvesIt() {
    // Structural, for the reason given above: ensure_model_loaded cannot be
    // linked into this binary. What it pins is that the gate exists inside
    // that function, that it is the TOLERANT form tested above, and that it
    // runs BEFORE get_auto_model -- a gate placed after the resolution would
    // be decoration.
    const auto source = ReadSource(FLM_REST_HANDLER_SOURCE);
    // BOUNDED TO THE FUNCTION BODY ON PURPOSE. Searching from the signature
    // to the end of the file would have found handle_openai_completion's own
    // is_model_supported at :1468 and reported green over a function with no
    // gate at all -- which is exactly what the first cut of this test did.
    const auto body = FunctionBody(
        source, "bool RestHandler::ensure_model_loaded(");
    const auto gate = body.find("is_model_supported(");
    const auto rectify = body.find("rectify_model_tag(");
    const auto resolve = body.find("get_auto_model(");
    TEST_REQUIRE(gate != std::string::npos);
    TEST_REQUIRE(rectify != std::string::npos);
    TEST_REQUIRE(resolve != std::string::npos);
    TEST_REQUIRE(gate < resolve);
    TEST_REQUIRE(rectify < resolve);
}

void TestTheServerGateRefusesTheUnsupportedNameAndNotTheSupportedOne() {
    // THE POLARITY, WHICH THE TEST ABOVE CANNOT SEE. Everything it checks --
    // that the three calls are present and in the right order -- stays true
    // when the `!` is deleted from the condition. That single character
    // inverts the gate so `flm serve` refuses EVERY model it supports, a
    // total denial of service strictly worse than the bug C-2 closed, and
    // review-R4.md measured the result: 44 PASS / 0 FAIL. A structural check
    // that reads as behavioural is the shape this project keeps finding.
    //
    // Three assertions, because polarity is three facts and not one: the
    // predicate is NEGATED; the negation guards a REFUSAL; and the refusal
    // happens BEFORE the resolution rather than after it. Swap the sense of
    // the condition and the first fails; turn the refusal into a log line
    // and the second fails.
    const auto source = ReadSource(FLM_REST_HANDLER_SOURCE);
    const auto body = FunctionBody(
        source, "bool RestHandler::ensure_model_loaded(");
    const auto negated = body.find("!supported_models.is_model_supported(");
    const auto refusal = body.find("return false;");
    const auto resolve = body.find("get_auto_model(");
    TEST_REQUIRE(negated != std::string::npos);
    TEST_REQUIRE(refusal != std::string::npos);
    TEST_REQUIRE(resolve != std::string::npos);
    // The FIRST `return false;` in this body is the gate's own -- the others
    // are the download-incompatible and load-failure paths, both of which
    // come after get_auto_model.
    TEST_REQUIRE(negated < refusal);
    TEST_REQUIRE(refusal < resolve);
}

void TestTheUnsupportedTagBranchDoesNotReturnBeforeTheCatalogResolvesIt() {
    // The other half. get_auto_model used to answer an unsupported tag with
    //     return std::make_pair("llama3.2:1b", std::make_unique<Llama3>(...));
    // choosing a FRONTEND from a tag the catalog had not resolved yet. The
    // caller then resolved that literal independently and got a different
    // entry, so the frontend and the package came from two different
    // answers. Letting get_model_info resolve first and taking the frontend
    // from the resolved entry's details.family makes one answer out of two.
    //
    // THE NAME OF THIS TEST USED TO CLAIM MORE THAN IT CHECKED. It was
    // "NeverPairsALiteralTagWithAFrontend" over a grep for the exact
    // substring `std::make_pair("llama3.2:1b"` -- so re-adding the identical
    // substitution spelled `std::make_pair(std::string("llama3.2:1b"), ...)`
    // passed, and, worse, the `default:` branch ~90 lines below does exactly
    // what the old name forbids, today, on purpose (see the comment at that
    // branch). A test whose name is false of the file it reads is a trap for
    // the next reader.
    //
    // So it now asserts the property that is actually true and actually
    // load-bearing: THE UNSUPPORTED-TAG BRANCH DOES NOT RETURN. Any spelling
    // of an early return between the gate and the resolution goes red, not
    // just the one that was there.
    const auto source = ReadSource(FLM_ALL_MODELS_SOURCE);
    const auto body = FunctionBody(
        source,
        "inline std::pair<std::string, std::unique_ptr<AutoModel>> "
        "get_auto_model(");
    const auto gate = body.find("is_model_supported(model_tag) == false");
    const auto resolve = body.find("available_models.get_model_info(model_tag)");
    const auto dispatch = body.find("modelFamilyMap.at(");
    TEST_REQUIRE(gate != std::string::npos);
    TEST_REQUIRE(resolve != std::string::npos);
    TEST_REQUIRE(dispatch != std::string::npos);
    TEST_REQUIRE(gate < resolve);
    // And the resolution really is the thing the frontend is chosen from.
    TEST_REQUIRE(resolve < dispatch);
    // The general form of the old assertion: nothing returns out of that
    // branch, however it is spelled.
    TEST_REQUIRE(body.find("return", gate) > resolve);
}

// ---------------------------------------------------------------------------
// 10. The real packages. OPT-IN -- see main().

std::filesystem::path g_e2b_gguf;
std::filesystem::path g_e4b_gguf;

void RequireRealPackageLoads(const std::filesystem::path& gguf,
                             std::size_t expected_slots) {
    FakeRuntimeScope runtime;
    Request request(gguf.parent_path(), gguf.filename().string());
    auto backend = Build(request.context);
    // Every real Gemma 4 package this port supports states 106 and 2, from
    // three sources each, and the backend read them rather than assuming
    // them. The BOS half is the one that matters most here: Google's
    // tokenizer_config.json does not state it, so this is the assertion that
    // says the value the frontend will use really did come off these files.
    // config.json lists eos_token_id [1, 106], so <eos> stops generation too.
    RequireTheSevenOverrides(*backend, {kRealEosId, kRealConfigEosId}, kRealBosId);
    auto* engine = dynamic_cast<flm::gemma4::gemma4_rai*>(&backend->engine());
    TEST_REQUIRE(engine != nullptr);
    TEST_REQUIRE(engine->weight_slot_count() == expected_slots);
    TEST_REQUIRE(!engine->loaded_from_cache());
}

void TestTheRealE2bPackageLoadsThroughTheBackend() {
    // 1 standalone norm + 15 owning layers x 10 + 20 sharing layers x 6 + 1
    RequireRealPackageLoads(g_e2b_gguf, 272);
}

void TestTheRealE4bPackageLoadsThroughTheBackend() {
    // 1 + 24 x 10 + 18 x 6 + 1
    RequireRealPackageLoads(g_e4b_gguf, 350);
}

}  // namespace

int main() {
    // BEFORE ANYTHING ELSE. The weight cache defaults to the MODEL DIRECTORY,
    // and two of the tests below point at read-only packages under
    // C:/Users/chiz/work/models. A cache write there would modify files this
    // task is forbidden to touch -- and would also make every later load BIND
    // instead of PACK, which silently empties most of the assertions.
#ifdef _WIN32
    _putenv_s("FLM_RAI_WEIGHT_CACHE", "0");
#else
    setenv("FLM_RAI_WEIGHT_CACHE", "0", 1);
#endif

    // THE ONE LINE C10 ADDS TO builtin_backends.cpp, made here verbatim
    // because this binary does not link that file (it names every engine type
    // and would drag in the prebuilt libraries).
    flm::backend::BackendRegistry::instance().register_backend(
        "gemma4e", flm::backend::kRaiBackendId, flm::gemma4::rai_factory(),
        flm::gemma4::rai_traits());

    const char* e2b = std::getenv("FLM_GEMMA4_E2B_GGUF");
    const char* e4b = std::getenv("FLM_GEMMA4_E4B_GGUF");
    const bool have_e2b = e2b && *e2b;
    const bool have_e4b = e4b && *e4b;
    if (have_e2b != have_e4b) {
        std::cerr << "FAIL: exactly one of FLM_GEMMA4_E2B_GGUF / "
                     "FLM_GEMMA4_E4B_GGUF is set. Set both or neither.\n";
        return 1;
    }
    const bool real_packages = have_e2b && have_e4b;
    if (real_packages) {
        g_e2b_gguf = std::filesystem::path(e2b);
        g_e4b_gguf = std::filesystem::path(e4b);
    }

#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestTraitsAreReadableWithoutConstructingAnEngine);
    RUN_TEST(TestTraitsComeBackOutOfTheRegistryUnchanged);
    RUN_TEST(TestTheTraitsCeilingIsTheEnginesOwnLimit);
    RUN_TEST(TestContextFitsStopsAtTheValidated16384);

    RUN_TEST(TestPreemptionIsRejectedBeforeAnythingIsOpened);
    RUN_TEST(TestAnOverLongContextIsRejectedBeforeAnythingIsOpened);
    RUN_TEST(TestAContextPastThePagedCeilingIsRejectedBeforeAnythingIsOpened);
    RUN_TEST(TestAZeroContextIsRejectedBeforeAnythingIsOpened);
    RUN_TEST(TestAMismatchedConfigJsonIsRejectedAndNoDeviceObjectIsCreated);
    RUN_TEST(TestABadPackageIsDiagnosedBeforeTheRuntimeIsAcquired);
    RUN_TEST(TestAMissingConfigJsonIsNamed);
    RUN_TEST(TestANullConfigIsRejectedBeforeAnythingIsOpened);
    RUN_TEST(TestAPackageWithNoGgufIsNamedRatherThanGuessedAt);

    RUN_TEST(TestTheSevenOverridesOnASyntheticE2bPackage);
    RUN_TEST(TestTheSevenOverridesOnASyntheticE4bPackage);
    RUN_TEST(TestTheBackendOwnsTheEngineItReportsOn);

    RUN_TEST(TestForcedEosIdsAreReadFromTheFileNotHardcoded);
    RUN_TEST(TestConfigJsonDisagreeingAboutEosRejectsThePackage);
    RUN_TEST(TestTokenizerDisagreeingAboutEosRejectsThePackage);

    RUN_TEST(TestForcedBosIdIsReadFromTheFileNotHardcoded);
    RUN_TEST(TestConfigJsonDisagreeingAboutBosRejectsThePackage);
    RUN_TEST(TestTokenizerDisagreeingAboutBosRejectsThePackage);

    RUN_TEST(TestTheBackendOpensExactlyTheFourPackageFiles);
    RUN_TEST(TestTheFrontendsTokenizerConfigIsUsedWhenItIsSupplied);
    RUN_TEST(TestTheGgufNameComesFromTheCatalogEntryNotAConstant);
    RUN_TEST(TestACatalogEntryNamingNoGgufFallsBackToTheDirectory);

    RUN_TEST(TestPoisonedForwardsFromTheEngine);
    RUN_TEST(TestAFirstDispatchRefusalDoesNotPoison);

    RUN_TEST(TestStripCommentsDoesNotEatCode);
    RUN_TEST(TestFunctionBodyStopsAtTheFunctionItNames);
    RUN_TEST(TestTheRuntimeMemberIsDeclaredBeforeTheEngineMember);
    RUN_TEST(TestTheConstructorClearsTheEngineBeforeItPublishesIt);
    RUN_TEST(TestTheDestructorDoesNotEnumerateTheTeardown);
    RUN_TEST(TestDestroyingTheBackendReleasesEveryCorelibObject);

    RUN_TEST(TestGemma4eHasARaiBackendRegistered);
    RUN_TEST(TestResolveBackendIdPicksRaiForGemma4e);
    RUN_TEST(TestAnUnknownBackendIdForGemma4eNamesTheRealOne);

    RUN_TEST(TestTripwireOneTheAieNextFallbackIsGemma4AndItsFamilyCanLoad);
    RUN_TEST(TestTripwireOneAnUnresolvableTagStillLoadsRatherThanThrowing);
    RUN_TEST(TestTripwireOneStxIsUnaffected);
    RUN_TEST(TestAPrefixedTagResolvesToItsOwnFamilyRatherThanTheDefault);
    RUN_TEST(TestTripwireTwoTheServerFallbackTagIsItselfUnsupportedOnAieNext);

    RUN_TEST(TestTheServerGateAdmitsEveryShippedTagOnBothPlatforms);
    RUN_TEST(TestTheServerGateRejectsNamesThatUsedToBeSubstituted);
    RUN_TEST(TestTheServerGatesAnUnknownModelNameBeforeItResolvesIt);
    RUN_TEST(TestTheServerGateRefusesTheUnsupportedNameAndNotTheSupportedOne);
    RUN_TEST(TestTheUnsupportedTagBranchDoesNotReturnBeforeTheCatalogResolvesIt);

    if (real_packages) {
        std::cout << "REAL PACKAGES: " << g_e2b_gguf.parent_path().string()
                  << "\n               "
                  << g_e4b_gguf.parent_path().string() << '\n';
        RUN_TEST(TestTheRealE2bPackageLoadsThroughTheBackend);
        RUN_TEST(TestTheRealE4bPackageLoadsThroughTheBackend);
    } else {
        // LOUD. A run without these covers the backend against synthetic
        // files only, and the one thing synthetic files cannot say is that
        // the EOS id a real Google package states is the one this backend
        // reports.
        std::cerr << "NOT COVERED: FLM_GEMMA4_E2B_GGUF / FLM_GEMMA4_E4B_GGUF "
                     "are unset, so the real-package tests did not run.\n";
    }
#undef RUN_TEST

    std::cout << "test_gemma4_backend: PASS\n";
}
