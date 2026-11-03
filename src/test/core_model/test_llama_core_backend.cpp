/// \file test_llama_core_backend.cpp
/// \brief flm::llama::core_factory against the fake corelib: the spec Llama 3 states
#include "models/llama/core/dev04/llama_core_backend.hpp"
#include "core/corelib_runtime.hpp"
#include "../phi4_core/fake_corelib.hpp"
#include "../phi4_core/test_support.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/// \brief stands in for the one in utils.cpp, which drags in the NPU runtime
/// \note Never consulted: each test installs the fake process runtime first.
namespace utils {
std::string get_executable_directory() { return std::filesystem::current_path().string(); }
}  // namespace utils

namespace {
namespace fs = std::filesystem;
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;

constexpr const char* kGguf = "Llama-3.2-1B-Instruct-Q8_0.gguf";

/// \brief a Llama 3 package directory holding only the GGUF, removed on the way out
/// \note No tokenizer files: a spec that states its stop ids must not read them.
struct Package {
    fs::path root = fs::temp_directory_path() /
                    ("flm-llama-core-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    Package() {
        fs::create_directories(root);
        std::ofstream(root / kGguf) << "x";
    }
    ~Package() {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
};

struct Runtime {
    std::shared_ptr<CorelibRuntime> runtime;

    Runtime() {
        fake_corelib::Reset();
        fake_corelib::GetState().model_vocab = 128256;
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

flm::backend::BackendContext Context(const Package& package, std::uint32_t length) {
    flm::backend::BackendContext context;
    context.model_path = package.root.string();
    context.model_info = {{"files", {kGguf, "tokenizer.json", "tokenizer_config.json", "config.json"}}};
    context.context_length = length;
    return context;
}

void TestTheCatalogsGgufRunsAsCorelibsLlama() {
    Runtime r;
    Package p;
    auto backend = flm::llama::core_factory()(Context(p, flm::llama::kCoreContextLimit));
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.model_names.front() == "llama");
    TEST_REQUIRE(state.model_path == (p.root / kGguf).string());
    TEST_REQUIRE(state.model_max_sequence_length == flm::llama::kCoreContextLimit);
    TEST_REQUIRE(backend->id() == flm::backend::kCoreBackendId);
    TEST_REQUIRE(backend->max_decode_length() == flm::llama::kCoreContextLimit - 1);
    TEST_REQUIRE(!backend->supports_preemption());
    TEST_REQUIRE(!backend->forwards_past_eos());
    // <|eot_id|>, <|eom_id|>, <|end_of_text|>
    TEST_REQUIRE((backend->forced_eos_ids() == std::vector<int>{128009, 128008, 128001}));
    TEST_REQUIRE(!backend->forced_bos_id().has_value());
}

void TestAContextPastTheLimitIsRefusedBeforeTheWeightsAreOpened() {
    Runtime r;
    Package p;
    RequireContains(RequireThrows<std::out_of_range>(
                        [&] { (void)flm::llama::core_factory()(Context(p, flm::llama::kCoreContextLimit + 1)); }),
                    "1..131072");
    TEST_REQUIRE(fake_corelib::GetState().call_counts["ryzenai_corelib_create_model_from_path"] == 0);
}

}  // namespace

int main() {
#define RUN(name) RunTest(&name, #name)
    RUN(TestTheCatalogsGgufRunsAsCorelibsLlama);
    RUN(TestAContextPastTheLimitIsRefusedBeforeTheWeightsAreOpened);
#undef RUN
    std::cout << "All Llama core backend tests passed\n";
    return 0;
}
