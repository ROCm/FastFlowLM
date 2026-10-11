/// \file test_gemma4_registry.cpp
/// \brief Gemma 4's rai backend as AutoModel and the server see it: its
///        traits, its registration under gemma4e, and the shipped catalog
#include "models/gemma4/rai/aie_next/gemma4_rai_backend.hpp"
#include "rai/corelib_runtime.hpp"

#include "AutoModel/model_backend.hpp"
#include "model_list.hpp"

#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace utils {
std::string get_executable_directory() { return std::filesystem::current_path().string(); }
}  // namespace utils

namespace {

using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;

constexpr const char* kFamily = "gemma4e";
constexpr const char* kRai = "rai";

/// \brief a Gemma 4 package as the rai backend reads it: the token files, and
///        a GGUF the fake corelib never opens
class TempPackage final {
public:
    explicit TempPackage(std::string gguf_name = "gemma-4-E2B-it-Q8_0.gguf")
        : gguf_name_(std::move(gguf_name)) {
        path_ = std::filesystem::temp_directory_path() /
                ("flm-gemma4-registry-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path_);
        std::ofstream(path_ / gguf_name_, std::ios::binary) << "GGUF";
        std::ofstream(path_ / "config.json") << R"({"eos_token_id": [1, 106]})";
        std::ofstream(path_ / "tokenizer_config.json")
            << R"({"eos_token": "<eos>", "bos_token": "<bos>"})";
        std::ofstream(path_ / "tokenizer.json")
            << R"({"added_tokens": [{"id": 1, "content": "<eos>"},)"
               R"( {"id": 2, "content": "<bos>"}, {"id": 106, "content": "<turn|>"}]})";
    }
    ~TempPackage() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TempPackage(const TempPackage&) = delete;
    TempPackage& operator=(const TempPackage&) = delete;

    const std::filesystem::path& path() const { return path_; }
    nlohmann::ordered_json model_info() const {
        nlohmann::ordered_json info;
        info["details"] = {{"family", kFamily}};
        info["default_context_length"] = 4096;
        info["files"] = nlohmann::ordered_json::array(
            {gguf_name_, "tokenizer.json", "tokenizer_config.json", "config.json"});
        return info;
    }

private:
    std::filesystem::path path_;
    std::string gguf_name_;
};

/// \brief a BackendContext pointing at a package, with the frontend's defaults
struct Request {
    LM_Config config;
    nlohmann::ordered_json info;
    flm::backend::BackendContext context;

    explicit Request(const TempPackage& package, std::uint32_t context_length = 4096)
        : info(package.model_info()) {
        context.model_path = package.path().string();
        context.model_info = info;
        context.config = &config;
        context.context_length = context_length;
    }
};

/// \brief the fake corelib as the process runtime; declare it first in a scope
struct FakeRuntimeScope {
    std::shared_ptr<CorelibRuntime> runtime;

    FakeRuntimeScope() {
        fake_corelib::Reset();
        runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
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

    std::size_t live_objects() const { return runtime->api()->live_object_count(); }
    bool nothing_created() const { return live_objects() == baseline_objects; }

    std::size_t baseline_objects{};
};

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
    // kRaiContextLimit.
    TEST_REQUIRE(flm::gemma4::kRaiContextLimit == 16384);
    TEST_REQUIRE(flm::gemma4::kRaiDecodeLimit + 1 == flm::gemma4::kRaiContextLimit);
}

void TestContextFitsStopsAtTheValidated16384() {
    TEST_REQUIRE(flm::gemma4::ContextFits(4096));
    TEST_REQUIRE(flm::gemma4::ContextFits(8192));
    TEST_REQUIRE(flm::gemma4::ContextFits(16384));
    TEST_REQUIRE(!flm::gemma4::ContextFits(10000));
    TEST_REQUIRE(!flm::gemma4::ContextFits(20480));
    TEST_REQUIRE(!flm::gemma4::ContextFits(131072));
}

void TestTheFactoryReadsTheShippedPackagesTokensAndContextRule() {
    TempPackage package;
    FakeRuntimeScope runtime;
    {
        Request request(package, 8192);
        auto backend = flm::gemma4::rai_factory()(request.context);
        auto eos = backend->forced_eos_ids().value_or(std::vector<int>{});
        std::sort(eos.begin(), eos.end());
        TEST_REQUIRE((eos == std::vector<int>{1, 106}));
        TEST_REQUIRE(backend->forced_bos_id() == 2);
        TEST_REQUIRE(backend->max_decode_length() == 8192 - 1);
    }
    TEST_REQUIRE(runtime.nothing_created());
    Request refused(package, 5000);
    RequireContains(RequireThrows<std::out_of_range>(
                        [&] { (void)flm::gemma4::rai_factory()(refused.context); }),
                    "multiple of 4096");
    TEST_REQUIRE(runtime.nothing_created());
}

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
    TempPackage package;
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

    TempPackage package;
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

}  // namespace

int main() {
    flm::backend::BackendRegistry::instance().register_backend(
        "gemma4e", flm::backend::kRaiBackendId, flm::gemma4::rai_factory(),
        flm::gemma4::rai_traits());
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestTraitsAreReadableWithoutConstructingAnEngine);
    RUN_TEST(TestTraitsComeBackOutOfTheRegistryUnchanged);
    RUN_TEST(TestTheTraitsCeilingIsTheEnginesOwnLimit);
    RUN_TEST(TestContextFitsStopsAtTheValidated16384);
    RUN_TEST(TestTheFactoryReadsTheShippedPackagesTokensAndContextRule);

    RUN_TEST(TestFunctionBodyStopsAtTheFunctionItNames);
    RUN_TEST(TestStripCommentsDoesNotEatCode);

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
#undef RUN_TEST
    std::cout << "All Gemma 4 registry tests passed\n";
    return 0;
}
