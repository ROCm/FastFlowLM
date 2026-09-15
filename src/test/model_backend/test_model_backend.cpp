/// \file test_model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
/// \note  Deliberately free of NPU hardware and of any prebuilt engine library:
///        every backend here is a stub, so this builds and runs on Linux CI
///        where the phi4_corelib_aie4 suite cannot. register_builtin_backends
///        is stubbed out below for the same reason.
#include "AutoModel/model_backend.hpp"
#include "../phi4_corelib_aie4/test_support.hpp"

#include <cstdlib>
#include <memory>
#include <string>

using flm::backend::BackendContext;
using flm::backend::BackendRegistry;
using flm::backend::BackendTraits;
using flm::backend::kDefaultBackendId;
using flm::backend::ModelBackend;
using flm::backend::resolve_backend_id;
using flm::backend::supported_backends;

/// \brief the builtin set, emptied
/// \note The real one lives in builtin_backends.cpp and pulls in every engine
///       header, and with them the prebuilt libraries. These tests populate the
///       registry themselves, so an empty set is both enough and honest.
namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

/// \brief a backend that owns no engine
/// \note engine() is never called by these tests; nothing here has a causal_lm
///       to hand back, and building one needs an NPU.
class StubBackend final : public ModelBackend {
public:
    explicit StubBackend(std::string id) : id_(std::move(id)) {}
    causal_lm& engine() override {
        throw std::runtime_error("stub backend has no engine");
    }
    std::string id() const override { return id_; }

private:
    std::string id_;
};

flm::backend::BackendFactory StubFactory(std::string id) {
    return [id = std::move(id)](const BackendContext&) {
        return std::make_unique<StubBackend>(id);
    };
}

/// \brief a registry that is not the process-wide one
BackendRegistry MakeRegistry() { return BackendRegistry(); }

nlohmann::ordered_json Entry(nlohmann::ordered_json details = nlohmann::ordered_json::object()) {
    nlohmann::ordered_json info = nlohmann::ordered_json::object();
    info["details"] = std::move(details);
    return info;
}

/// \brief scoped setenv/unsetenv for FLM_BACKEND
class ScopedBackendEnv {
public:
    explicit ScopedBackendEnv(const char* value) {
#if defined(_WIN32)
        _putenv_s("FLM_BACKEND", value ? value : "");
#else
        if (value) ::setenv("FLM_BACKEND", value, 1);
        else ::unsetenv("FLM_BACKEND");
#endif
    }
    ~ScopedBackendEnv() {
#if defined(_WIN32)
        _putenv_s("FLM_BACKEND", "");
#else
        ::unsetenv("FLM_BACKEND");
#endif
    }
};

void test_register_and_create() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "flm_npu", StubFactory("flm_npu"));
    registry.register_backend("phi4", "corelib_aie4_gguf",
                              StubFactory("corelib_aie4_gguf"),
                              BackendTraits{false, false, 4096});

    TEST_REQUIRE(registry.has("phi4", "flm_npu"));
    TEST_REQUIRE(!registry.has("phi4", "bogus"));
    TEST_REQUIRE(!registry.has("llama3", "flm_npu"));

    // available() is sorted, which is what makes the error messages stable.
    const auto ids = registry.available("phi4");
    TEST_REQUIRE(ids.size() == 2);
    TEST_REQUIRE(ids[0] == "corelib_aie4_gguf");
    TEST_REQUIRE(ids[1] == "flm_npu");
    TEST_REQUIRE(registry.available("llama3").empty());

    BackendContext context;
    auto backend = registry.create("phi4", "corelib_aie4_gguf", context);
    TEST_REQUIRE(backend != nullptr);
    TEST_REQUIRE(backend->id() == "corelib_aie4_gguf");
}

void test_traits_are_kept_per_backend() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "flm_npu", StubFactory("flm_npu"));
    registry.register_backend("phi4", "corelib_aie4_gguf",
                              StubFactory("corelib_aie4_gguf"),
                              BackendTraits{false, false, 4096});

    // The defaults describe the FastFlowLM NPU engines.
    const auto npu = registry.traits("phi4", "flm_npu");
    TEST_REQUIRE(npu.needs_npu_xclbin);
    TEST_REQUIRE(npu.supports_preemption);
    TEST_REQUIRE(npu.max_context_length == 0);

    const auto corelib = registry.traits("phi4", "corelib_aie4_gguf");
    TEST_REQUIRE(!corelib.needs_npu_xclbin);
    TEST_REQUIRE(!corelib.supports_preemption);
    TEST_REQUIRE(corelib.max_context_length == 4096);
}

void test_backend_defaults() {
    StubBackend backend("flm_npu");
    TEST_REQUIRE(backend.detail().empty());
    TEST_REQUIRE(backend.max_decode_length() == 0);
    TEST_REQUIRE(backend.supports_preemption());
    TEST_REQUIRE(backend.forwards_past_eos());
    TEST_REQUIRE(!backend.poisoned());
    TEST_REQUIRE(!backend.forced_eos_ids().has_value());
}

void test_duplicate_registration_is_rejected() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "flm_npu", StubFactory("first"));
    const std::string message = RequireThrows([&] {
        registry.register_backend("phi4", "flm_npu", StubFactory("second"));
    });
    RequireContains(message, "already registered");

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", "flm_npu", context)->id() == "first");

    RequireThrows([&] { registry.register_backend("", "flm_npu", StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "", StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "x", nullptr); });
}

void test_replace_backend_is_the_test_seam() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "flm_npu", StubFactory("real"));
    registry.replace_backend("phi4", "flm_npu", StubFactory("stub"));

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", "flm_npu", context)->id() == "stub");
    TEST_REQUIRE(registry.available("phi4").size() == 1);

    // It also registers a backend that was not there before.
    registry.replace_backend("llama3", "flm_npu", StubFactory("fresh"));
    TEST_REQUIRE(registry.create("llama3", "flm_npu", context)->id() == "fresh");
}

void test_unknown_id_names_what_exists() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "flm_npu", StubFactory("flm_npu"));

    BackendContext context;
    const std::string message =
        RequireThrows([&] { registry.create("phi4", "bogus", context); });
    RequireContains(message, "bogus");
    RequireContains(message, "flm_npu");

    const std::string empty =
        RequireThrows([&] { registry.create("llama3", "flm_npu", context); });
    RequireContains(empty, "(none)");
}

void test_supported_backends_falls_back() {
    // No key at all: what the entry has always run on.
    TEST_REQUIRE(supported_backends(Entry()) ==
                 std::vector<std::string>{kDefaultBackendId});

    // execution_backend alone still narrows the entry to itself.
    const auto corelib = Entry({{"execution_backend", "corelib_aie4_gguf"}});
    TEST_REQUIRE(supported_backends(corelib) ==
                 std::vector<std::string>{"corelib_aie4_gguf"});

    // An explicit list wins over execution_backend.
    auto both = corelib;
    both["supported_backends"] = {"flm_npu", "corelib_aie4_gguf"};
    TEST_REQUIRE(supported_backends(both).size() == 2);

    auto malformed = Entry();
    malformed["supported_backends"] = nlohmann::ordered_json::array();
    RequireThrows([&] { supported_backends(malformed); });
    malformed["supported_backends"] = {1, 2};
    RequireThrows([&] { supported_backends(malformed); });
    RequireThrows([&] {
        supported_backends(Entry({{"execution_backend", 7}}));
    });
}

void test_resolution_precedence() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", "flm_npu", StubFactory("flm_npu"));
    registry.replace_backend("phi4", "corelib_aie4_gguf",
                             StubFactory("corelib_aie4_gguf"));

    auto info = Entry();
    info["supported_backends"] = {"flm_npu", "corelib_aie4_gguf"};

    std::string source;
    // 4. nothing says anything -> the default
    TEST_REQUIRE(resolve_backend_id("phi4", info, "", &source) == kDefaultBackendId);
    TEST_REQUIRE(source == "default");

    // 3. the catalog
    auto catalog = info;
    catalog["details"]["execution_backend"] = "corelib_aie4_gguf";
    TEST_REQUIRE(resolve_backend_id("phi4", catalog, "", &source) ==
                 "corelib_aie4_gguf");
    TEST_REQUIRE(source == "model catalog");

    {
        // 2. FLM_BACKEND beats the catalog
        ScopedBackendEnv env("flm_npu");
        TEST_REQUIRE(resolve_backend_id("phi4", catalog, "", &source) == "flm_npu");
        TEST_REQUIRE(source == "FLM_BACKEND");

        // 1. --backend beats both
        TEST_REQUIRE(resolve_backend_id("phi4", catalog, "corelib_aie4_gguf",
                                        &source) == "corelib_aie4_gguf");
        TEST_REQUIRE(source == "--backend");
    }

    // An empty FLM_BACKEND is the same as an unset one.
    ScopedBackendEnv empty("");
    TEST_REQUIRE(resolve_backend_id("phi4", info, "", &source) == kDefaultBackendId);
}

void test_resolution_rejects_with_a_readable_message() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", "flm_npu", StubFactory("flm_npu"));

    // Registered for the family, but this entry does not allow it.
    auto info = Entry();
    info["supported_backends"] = {"flm_npu"};
    const std::string not_allowed = RequireThrows(
        [&] { resolve_backend_id("phi4", info, "corelib_aie4_gguf"); });
    RequireContains(not_allowed, "--backend");
    RequireContains(not_allowed, "It supports: flm_npu");

    // Allowed by the entry, but this build does not have it.
    auto allowed = Entry();
    allowed["supported_backends"] = {"flm_npu", "not_built"};
    const std::string not_built =
        RequireThrows([&] { resolve_backend_id("phi4", allowed, "not_built"); });
    RequireContains(not_built, "not compiled into this build");
    RequireContains(not_built, "flm_npu");

    // The env var gets named in the message too, so the user can find it.
    ScopedBackendEnv env("bogus");
    const std::string from_env =
        RequireThrows([&] { resolve_backend_id("phi4", info, ""); });
    RequireContains(from_env, "FLM_BACKEND");
}

}  // namespace

int main() {
    RunTest(test_register_and_create, "register and create");
    RunTest(test_traits_are_kept_per_backend, "traits are kept per backend");
    RunTest(test_backend_defaults, "backend policy defaults");
    RunTest(test_duplicate_registration_is_rejected, "duplicate registration is rejected");
    RunTest(test_replace_backend_is_the_test_seam, "replace_backend is the test seam");
    RunTest(test_unknown_id_names_what_exists, "unknown id names what exists");
    RunTest(test_supported_backends_falls_back, "supported_backends falls back");
    RunTest(test_resolution_precedence, "resolution precedence");
    RunTest(test_resolution_rejects_with_a_readable_message,
            "resolution rejects with a readable message");
    std::cout << "All model backend tests passed\n";
    return 0;
}
