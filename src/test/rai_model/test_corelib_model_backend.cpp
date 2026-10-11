/// \file test_corelib_model_backend.cpp
/// \brief rai::Factory against the fake corelib: what it refuses before the
///        weights are opened, which GGUF it loads, and what it reports
#include "rai/corelib_model_backend.hpp"
#include "../phi4_rai/fake_corelib.hpp"
#include "../phi4_rai/test_support.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/// \brief stands in for the one in utils.cpp, which drags in the NPU runtime
/// \note Never consulted: each test installs the fake process runtime first,
///       and CorelibRuntime::GetOrCreate returns that one.
namespace utils {
std::string get_executable_directory() { return std::filesystem::current_path().string(); }
}  // namespace utils

namespace {
namespace fs = std::filesystem;
using flm::backend::BackendContext;
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::rai::BackendSpec;
using flm::rai::Continuation;
using flm::rai::Rewind;

/// \brief a scratch model directory, removed on the way out
struct Package {
    fs::path root;

    explicit Package(const std::vector<std::string>& files) {
        root = fs::temp_directory_path() /
               ("flm-rai-backend-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        for (const auto& name : files) std::ofstream(root / name) << "x";
    }
    ~Package() {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    void Write(const std::string& name, const std::string& text) const {
        std::ofstream(root / name) << text;
    }
};

/// \brief the fake corelib, installed as the process runtime for one scope
struct Runtime {
    std::shared_ptr<CorelibRuntime> runtime;

    Runtime() {
        fake_corelib::Reset();
        fake_corelib::GetState().model_vocab = 300000;
        runtime = CorelibRuntime::CreateForTest(CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    }
    ~Runtime() {
        try {
            runtime.reset();
            CorelibRuntime::ShutdownProcess();
        } catch (...) {
        }
    }
};

BackendSpec Spec() {
    return {.model = {"qwen3", "Qwen3", 4096, Continuation::kFromStart, Rewind::kSeek},
            .context_rule = "1..4096",
            .eos_ids = {151645, 151643}};
}

BackendContext Context(const Package& package, std::uint32_t length,
                       nlohmann::ordered_json files = nlohmann::ordered_json::array({"m.gguf"})) {
    BackendContext context;
    context.model_path = package.root.string();
    context.model_info = {{"files", files}};
    context.context_length = length;
    return context;
}

int WeightLoads() {
    return fake_corelib::GetState().call_counts["ryzenai_corelib_model_weights_create_path"];
}

void TestPreemptionIsRefusedBeforeTheWeightsAreOpened() {
    Runtime r;
    Package p({"m.gguf"});
    auto context = Context(p, 4096);
    context.enable_preemption = true;
    (void)RequireThrows<std::invalid_argument>([&] { (void)flm::rai::Factory(Spec())(context); });
    TEST_REQUIRE(WeightLoads() == 0);
}

void TestAContextOutsideTheRuleIsRefusedBeforeTheWeightsAreOpened() {
    Runtime r;
    Package p({"m.gguf"});
    for (const std::uint32_t length : {0u, 4097u}) {
        RequireContains(RequireThrows<std::out_of_range>(
                            [&] { (void)flm::rai::Factory(Spec())(Context(p, length)); }),
                        "1..4096");
    }
    TEST_REQUIRE(WeightLoads() == 0);
}

void TestACustomContextRuleIsApplied() {
    Runtime r;
    Package p({"m.gguf"});
    auto spec = Spec();
    spec.model.context_limit = 16384;
    spec.context_fits = [](std::uint32_t n) { return n == 8192; };
    spec.context_rule = "8192 only";
    RequireContains(RequireThrows([&] { (void)flm::rai::Factory(spec)(Context(p, 4096)); }),
                    "8192 only");
    auto backend = flm::rai::Factory(spec)(Context(p, 8192));
    TEST_REQUIRE(fake_corelib::GetState().model_max_sequence_length == 8192);
}

void TestTheCatalogsGgufIsTheOneLoaded() {
    Runtime r;
    Package p({"a.gguf", "b.gguf"});
    auto backend = flm::rai::Factory(Spec())(
        Context(p, 4096, nlohmann::ordered_json::array({"b.gguf", "config.json"})));
    TEST_REQUIRE(fake_corelib::GetState().model_weights_path == (p.root / "b.gguf").string());
}

void TestACatalogGgufThatIsMissingIsNamed() {
    Runtime r;
    Package p({"a.gguf"});
    RequireContains(RequireThrows([&] {
                        (void)flm::rai::Factory(Spec())(
                            Context(p, 4096, nlohmann::ordered_json::array({"gone.gguf"})));
                    }),
                    "gone.gguf");
    TEST_REQUIRE(WeightLoads() == 0);
}

void TestACatalogNamingTwoGgufsIsRefused() {
    Runtime r;
    Package p({"a.gguf", "b.gguf"});
    RequireContains(RequireThrows([&] {
                        (void)flm::rai::Factory(Spec())(
                            Context(p, 4096, nlohmann::ordered_json::array({"a.gguf", "b.gguf"})));
                    }),
                    "exactly one");
}

void TestWithoutACatalogGgufTheDirectorysOnlyOneIsLoaded() {
    Runtime r;
    Package p({"only.gguf", "config.json"});
    auto backend = flm::rai::Factory(Spec())(
        Context(p, 4096, nlohmann::ordered_json::array({"config.json"})));
    TEST_REQUIRE(fake_corelib::GetState().model_weights_path == (p.root / "only.gguf").string());
}

void TestWithoutACatalogGgufTwoInTheDirectoryAreRefused() {
    Runtime r;
    Package p({"a.gguf", "b.gguf"});
    RequireContains(RequireThrows([&] {
                        (void)flm::rai::Factory(Spec())(Context(p, 4096, nlohmann::ordered_json::array()));
                    }),
                    "exactly one");
}

void TestTheBackendReportsWhatTheSpecStates() {
    Runtime r;
    Package p({"m.gguf"});
    auto backend = flm::rai::Factory(Spec())(Context(p, 2048));
    TEST_REQUIRE(backend->id() == flm::backend::kRaiBackendId);
    TEST_REQUIRE(backend->max_decode_length() == 2047);
    TEST_REQUIRE(!backend->supports_preemption());
    TEST_REQUIRE(!backend->forwards_past_eos());
    TEST_REQUIRE((backend->forced_eos_ids() == std::vector<int>{151645, 151643}));
    TEST_REQUIRE(!backend->forced_bos_id().has_value());
    TEST_REQUIRE(backend->engine().get_current_context_length() == 0);
    TEST_REQUIRE(fake_corelib::GetState().model_names.front() == "qwen3");
    TEST_REQUIRE(fake_corelib::GetState().model_max_sequence_length == 2048);
}

void TestWithoutSpecEosIdsThePackagesAreRead() {
    Runtime r;
    Package p({"m.gguf"});
    p.Write("tokenizer_config.json", R"({"eos_token": {"content": "<turn|>"}, "bos_token": "<bos>"})");
    p.Write("tokenizer.json",
            R"({"added_tokens": [{"id": 2, "content": "<bos>"}, {"id": 106, "content": "<turn|>"}]})");
    p.Write("config.json", R"({"eos_token_id": [1, 106]})");
    auto spec = Spec();
    spec.eos_ids.clear();
    spec.read_bos = true;
    auto backend = flm::rai::Factory(spec)(Context(p, 4096));
    TEST_REQUIRE((backend->forced_eos_ids() == std::vector<int>{106, 1}));
    TEST_REQUIRE(backend->forced_bos_id() == 2);
}

void TestAMissingTokenizerConfigIsNamedBeforeTheWeightsAreOpened() {
    Runtime r;
    Package p({"m.gguf"});
    auto spec = Spec();
    spec.eos_ids.clear();
    RequireContains(RequireThrows([&] { (void)flm::rai::Factory(spec)(Context(p, 4096)); }),
                    "tokenizer_config.json");
    TEST_REQUIRE(WeightLoads() == 0);
}

void TestAConfigJsonThatIsNotAnObjectIsNamed() {
    Runtime r;
    Package p({"m.gguf"});
    p.Write("tokenizer_config.json", R"({"eos_token": "<eos>"})");
    p.Write("tokenizer.json", R"({"added_tokens": [{"id": 1, "content": "<eos>"}]})");
    p.Write("config.json", "[1, 2]");
    auto spec = Spec();
    spec.eos_ids.clear();
    RequireContains(RequireThrows([&] { (void)flm::rai::Factory(spec)(Context(p, 4096)); }),
                    "config.json");
    TEST_REQUIRE(WeightLoads() == 0);
}

void TestDestroyingTheBackendReleasesEveryCorelibObject() {
    Runtime r;
    Package p({"m.gguf"});
    auto backend = flm::rai::Factory(Spec())(Context(p, 4096));
    TEST_REQUIRE(r.runtime->api()->live_object_count() > 0);
    backend.reset();
    TEST_REQUIRE(r.runtime->api()->live_object_count() == 0);
}

void TestPoisonedFollowsTheEngine() {
    Runtime r;
    Package p({"m.gguf"});
    auto backend = flm::rai::Factory(Spec())(Context(p, 4096));
    fake_corelib::GetState().statuses["ryzenai_corelib_model_forward"] =
        ryzenai_corelib_status_failure;
    std::vector<int> ids{1, 2};
    (void)RequireThrows([&] { (void)backend->engine().prefill(ids); });
    TEST_REQUIRE(backend->poisoned());
}

}  // namespace

int main() {
#define RUN(name) RunTest(&name, #name)
    RUN(TestPreemptionIsRefusedBeforeTheWeightsAreOpened);
    RUN(TestAContextOutsideTheRuleIsRefusedBeforeTheWeightsAreOpened);
    RUN(TestACustomContextRuleIsApplied);
    RUN(TestTheCatalogsGgufIsTheOneLoaded);
    RUN(TestACatalogGgufThatIsMissingIsNamed);
    RUN(TestACatalogNamingTwoGgufsIsRefused);
    RUN(TestWithoutACatalogGgufTheDirectorysOnlyOneIsLoaded);
    RUN(TestWithoutACatalogGgufTwoInTheDirectoryAreRefused);
    RUN(TestTheBackendReportsWhatTheSpecStates);
    RUN(TestWithoutSpecEosIdsThePackagesAreRead);
    RUN(TestAMissingTokenizerConfigIsNamedBeforeTheWeightsAreOpened);
    RUN(TestAConfigJsonThatIsNotAnObjectIsNamed);
    RUN(TestDestroyingTheBackendReleasesEveryCorelibObject);
    RUN(TestPoisonedFollowsTheEngine);
#undef RUN
    std::cout << "All rai backend tests passed\n";
    return 0;
}
