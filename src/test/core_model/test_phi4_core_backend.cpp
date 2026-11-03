/// \file test_phi4_core_backend.cpp
/// \brief flm::phi4::core_factory against the fake corelib: the spec Phi-4 states
#include "models/phi4/core/dev04/phi4_core_backend.hpp"
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

constexpr const char* kGguf = "Phi-4-mini-instruct-Q4_G128.gguf";

/// \brief a Phi-4 package directory holding the AMD GGUF, removed on the way out
struct Package {
    fs::path root = fs::temp_directory_path() /
                    ("flm-phi4-core-" +
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
        fake_corelib::GetState().model_vocab = 200064;
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

void TestTheCatalogsGgufRunsAsCorelibsPhi4mini() {
    Runtime r;
    Package p;
    auto backend = flm::phi4::core_factory()(Context(p, flm::phi4::kCoreContextLimit));
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.model_names.front() == "phi4mini");
    TEST_REQUIRE(state.model_path == (p.root / kGguf).string());
    TEST_REQUIRE(state.model_max_sequence_length == flm::phi4::kCoreContextLimit);
    TEST_REQUIRE(backend->id() == flm::backend::kCoreBackendId);
    TEST_REQUIRE(backend->max_decode_length() == flm::phi4::kCoreDecodeLimit);
    TEST_REQUIRE(!backend->supports_preemption());
    TEST_REQUIRE((backend->forced_eos_ids() == std::vector<int>{200020, 199999}));
    TEST_REQUIRE(!backend->forced_bos_id().has_value());
}

void TestAContextPastTheLimitIsRefusedBeforeTheWeightsAreOpened() {
    Runtime r;
    Package p;
    RequireContains(RequireThrows<std::out_of_range>(
                        [&] { (void)flm::phi4::core_factory()(Context(p, flm::phi4::kCoreContextLimit + 1)); }),
                    "1..131072");
    TEST_REQUIRE(fake_corelib::GetState().call_counts["ryzenai_corelib_create_model_from_path"] == 0);
}

}  // namespace

int main() {
#define RUN(name) RunTest(&name, #name)
    RUN(TestTheCatalogsGgufRunsAsCorelibsPhi4mini);
    RUN(TestAContextPastTheLimitIsRefusedBeforeTheWeightsAreOpened);
#undef RUN
    std::cout << "All Phi-4 core backend tests passed\n";
    return 0;
}
