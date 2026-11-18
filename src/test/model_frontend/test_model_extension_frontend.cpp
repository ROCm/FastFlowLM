/// \file test_model_extension_frontend.cpp
/// \brief Hardware-free tests that drive the production extension frontend
///        adapter against the complete C fixture
#include "AutoModel/automodel.hpp"
#include "AutoModel/model_backend.hpp"
#include "AutoModel/model_frontend.hpp"
#include "../phi4_core/test_support.hpp"

#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#ifndef FLM_TEST_COMPLETE_EXTENSION_PATH
#error FLM_TEST_COMPLETE_EXTENSION_PATH must name the complete fake extension
#endif

extern "C" int flm_test_event_count(int event);

// builtin_frontends.cpp names every shipped model, so the host-side pieces the
// registry and adapter need are supplied here.
namespace flm::frontend {
void register_builtin_frontends(FrontendRegistry&) {}

FrontendPtr own_frontend(std::unique_ptr<AutoModel> frontend) {
    return FrontendPtr(
        frontend.release(),
        FrontendDeleter{
            nullptr,
            [](void*, void* host_frontend) {
                delete static_cast<AutoModel*>(host_frontend);
            },
            nullptr});
}
}  // namespace flm::frontend

namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

// Complete providers tokenize themselves, so the host tokenizer must never run.
Tokenizer::Tokenizer(const std::string&) {
    throw std::logic_error("host tokenizer constructed");
}
Tokenizer::~Tokenizer() = default;
SafeTensors::~SafeTensors() = default;
std::vector<int> Tokenizer::encode(const std::string&) {
    throw std::logic_error("host tokenizer used");
}
std::string Tokenizer::decode(const std::vector<int>&) {
    throw std::logic_error("host tokenizer used");
}
std::string Tokenizer::run_time_decoder(int) {
    throw std::logic_error("host tokenizer used");
}

namespace {

using flm::frontend::FrontendPtr;
using flm::frontend::FrontendRegistry;

// Mirrors FLM_TEST_EVENT_* in fake_model_extension.c.
constexpr int kPreparedReleased = 0;
constexpr int kFrontendDestroyed = 1;

constexpr const char* kFamily = "complete-plugin-test";

FrontendRegistry& Registry() {
    static FrontendRegistry registry;
    static const bool loaded = [] {
        registry.load_extension(FLM_TEST_COMPLETE_EXTENSION_PATH, {});
        return true;
    }();
    (void)loaded;
    return registry;
}

nlohmann::json ModelInfo() {
    return {
        {"name", "Complete plugin model"},
        {"backend", "flm"},
        {"backend_provider", "plugin-provider"},
        {"frontend_provider", "plugin-provider"},
        {"default_context_length", 4096},
        {"max_prefill_len", 512},
        {"details", {{"family", kFamily}}},
    };
}

FrontendPtr LoadedFrontend(flm_rt::device& device) {
    auto frontend = Registry().create(kFamily, &device);
    frontend->load_model("fake-model", ModelInfo(), -1, false, "");
    // The fixture's logits favour its EOS id, so greedy decoding stops at once.
    sampler_config greedy;
    greedy.top_k = 1;
    frontend->set_sampler(greedy);
    return frontend;
}

lm_uniform_input_t Prompt(const std::string& text) {
    lm_uniform_input_t input;
    input.prompt = text;
    return input;
}

void test_adapter_loads_and_exposes_provider_traits() {
    flm_rt::device device;
    auto frontend = LoadedFrontend(device);
    TEST_REQUIRE(frontend->supports_logprobs());
    TEST_REQUIRE(!frontend->supports_continue_final_message());
    TEST_REQUIRE(frontend->encode_text("anything") == std::vector<int>({1, 2}));
    TEST_REQUIRE(frontend->token_text(2) == "2");
    nlohmann::ordered_json messages = nlohmann::ordered_json::array(
        {{{"role", "user"}, {"content", "hi"}}});
    TEST_REQUIRE(frontend->apply_chat_template(messages) == messages.dump());
}

void test_adapter_prepares_prefills_and_generates() {
    flm_rt::device device;
    auto frontend = LoadedFrontend(device);
    const int released = flm_test_event_count(kPreparedReleased);

    chat_meta_info_t meta;
    auto input = Prompt("hello");
    TEST_REQUIRE(frontend->insert(meta, input));
    TEST_REQUIRE(flm_test_event_count(kPreparedReleased) == released + 1);

    std::ostringstream output;
    (void)frontend->generate(meta, 8, output);
    TEST_REQUIRE(meta.stop_reason == EOT_DETECTED);
}

void test_adapter_reports_provider_cancellation() {
    flm_rt::device device;
    auto frontend = LoadedFrontend(device);

    chat_meta_info_t meta;
    auto input = Prompt("hello");
    TEST_REQUIRE(!frontend->insert(meta, input, [] { return true; }));
    TEST_REQUIRE(meta.stop_reason == CANCEL_DETECTED);
}

void test_adapter_rejects_requests_after_backend_poisoning() {
    flm_rt::device device;
    auto frontend = LoadedFrontend(device);
    const int released = flm_test_event_count(kPreparedReleased);

    chat_meta_info_t poisoning;
    auto poison = Prompt("poison-backend");
    RequireContains(
        RequireThrows<ModelRequestError>(
            [&] { (void)frontend->insert(poisoning, poison); }),
        "unload/reload is required");
    // The failed request's prepared payload is still released.
    TEST_REQUIRE(flm_test_event_count(kPreparedReleased) == released + 1);

    // A poisoned engine is never driven again: the next request is refused
    // before the provider prepares anything.
    chat_meta_info_t meta;
    auto input = Prompt("hello");
    RequireContains(
        RequireThrows<ModelRequestError>(
            [&] { (void)frontend->insert(meta, input); }),
        "unload/reload is required");
    TEST_REQUIRE(flm_test_event_count(kPreparedReleased) == released + 1);
    std::ostringstream output;
    RequireContains(
        RequireThrows<ModelRequestError>(
            [&] { (void)frontend->generate(meta, 8, output); }),
        "unload/reload is required");
}

void test_adapter_destroys_the_provider_frontend() {
    flm_rt::device device;
    const int destroyed = flm_test_event_count(kFrontendDestroyed);
    {
        auto frontend = Registry().create(kFamily, &device);
        TEST_REQUIRE(flm_test_event_count(kFrontendDestroyed) == destroyed);
    }
    TEST_REQUIRE(flm_test_event_count(kFrontendDestroyed) == destroyed + 1);
}

}  // namespace

int main() {
    RunTest(test_adapter_loads_and_exposes_provider_traits,
            "adapter loads and exposes provider traits");
    RunTest(test_adapter_prepares_prefills_and_generates,
            "adapter prepares, prefills and generates");
    RunTest(test_adapter_reports_provider_cancellation,
            "adapter reports provider cancellation");
    RunTest(test_adapter_rejects_requests_after_backend_poisoning,
            "adapter rejects requests after backend poisoning");
    RunTest(test_adapter_destroys_the_provider_frontend,
            "adapter destroys the provider frontend");
    std::cout << "All model extension frontend tests passed\n";
    return 0;
}
