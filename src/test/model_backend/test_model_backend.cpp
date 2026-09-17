/// \file test_model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
/// \note  Deliberately free of NPU hardware and of any prebuilt engine library:
///        every backend here is a stub, so this builds and runs on Linux CI
///        where the phi4_aie4 suite cannot. register_builtin_backends
///        is stubbed out below for the same reason.
/// \note  A backend id is a hardware id: "aie2p", "aie4", and one day "gpu".
///        The stubs below use those names because the resolution rules are
///        about hardware, not about which engine happens to serve it.
#include "AutoModel/model_backend.hpp"
#include "../phi4_aie4/test_support.hpp"

#include <cstdlib>
#include <memory>
#include <string>

using flm::backend::BackendContext;
using flm::backend::BackendRegistry;
using flm::backend::BackendTraits;
using flm::backend::kAie2pBackendId;
using flm::backend::kAie4BackendId;
using flm::backend::ModelBackend;
using flm::backend::resolve_backend_id;

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
    registry.register_backend("phi4", kAie2pBackendId, StubFactory("aie2p"));
    registry.register_backend("phi4", kAie4BackendId, StubFactory("aie4"),
                              BackendTraits{false, false, 4096});

    TEST_REQUIRE(registry.has("phi4", "aie2p"));
    TEST_REQUIRE(!registry.has("phi4", "bogus"));
    TEST_REQUIRE(!registry.has("llama3", "aie2p"));

    // available() is sorted, which is what makes the error messages stable.
    const auto ids = registry.available("phi4");
    TEST_REQUIRE(ids.size() == 2);
    TEST_REQUIRE(ids[0] == "aie2p");
    TEST_REQUIRE(ids[1] == "aie4");
    TEST_REQUIRE(registry.available("llama3").empty());

    BackendContext context;
    auto backend = registry.create("phi4", "aie4", context);
    TEST_REQUIRE(backend != nullptr);
    TEST_REQUIRE(backend->id() == "aie4");
}

void test_traits_are_kept_per_backend() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kAie2pBackendId, StubFactory("aie2p"));
    registry.register_backend("phi4", kAie4BackendId, StubFactory("aie4"),
                              BackendTraits{false, false, 4096});

    // The defaults describe the FastFlowLM NPU engines, i.e. aie2p.
    const auto aie2p = registry.traits("phi4", kAie2pBackendId);
    TEST_REQUIRE(aie2p.needs_npu_xclbin);
    TEST_REQUIRE(aie2p.supports_preemption);
    TEST_REQUIRE(aie2p.max_context_length == 0);

    const auto aie4 = registry.traits("phi4", kAie4BackendId);
    TEST_REQUIRE(!aie4.needs_npu_xclbin);
    TEST_REQUIRE(!aie4.supports_preemption);
    TEST_REQUIRE(aie4.max_context_length == 4096);
}

void test_backend_defaults() {
    StubBackend backend(kAie2pBackendId);
    TEST_REQUIRE(backend.detail().empty());
    TEST_REQUIRE(backend.max_decode_length() == 0);
    TEST_REQUIRE(backend.supports_preemption());
    TEST_REQUIRE(backend.forwards_past_eos());
    TEST_REQUIRE(!backend.poisoned());
    TEST_REQUIRE(!backend.forced_eos_ids().has_value());
}

void test_duplicate_registration_is_rejected() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kAie2pBackendId, StubFactory("first"));
    const std::string message = RequireThrows([&] {
        registry.register_backend("phi4", kAie2pBackendId, StubFactory("second"));
    });
    RequireContains(message, "already registered");

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", kAie2pBackendId, context)->id() == "first");

    RequireThrows([&] { registry.register_backend("", kAie2pBackendId, StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "", StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "x", nullptr); });
}

void test_replace_backend_is_the_test_seam() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kAie2pBackendId, StubFactory("real"));
    registry.replace_backend("phi4", kAie2pBackendId, StubFactory("stub"));

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", kAie2pBackendId, context)->id() == "stub");
    TEST_REQUIRE(registry.available("phi4").size() == 1);

    // It also registers a backend that was not there before.
    registry.replace_backend("llama3", kAie2pBackendId, StubFactory("fresh"));
    TEST_REQUIRE(registry.create("llama3", kAie2pBackendId, context)->id() == "fresh");
}

void test_unknown_id_names_what_exists() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kAie2pBackendId, StubFactory("aie2p"));

    BackendContext context;
    const std::string message =
        RequireThrows([&] { registry.create("phi4", "bogus", context); });
    RequireContains(message, "bogus");
    RequireContains(message, "aie2p");

    const std::string empty =
        RequireThrows([&] { registry.create("llama3", kAie2pBackendId, context); });
    RequireContains(empty, "(none)");
}

void test_the_hardware_is_the_default() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", kAie2pBackendId, StubFactory("aie2p"));
    registry.replace_backend("phi4", kAie4BackendId, StubFactory("aie4"));

    std::string source;

    // Nothing overrides it, so the machine decides -- and it decides both ways,
    // which is the whole point of collapsing backend onto hardware.
    TEST_REQUIRE(resolve_backend_id("phi4", "aie2p", "", &source) == kAie2pBackendId);
    TEST_REQUIRE(source == "detected hardware");
    TEST_REQUIRE(resolve_backend_id("phi4", "aie4", "", &source) == kAie4BackendId);
    TEST_REQUIRE(source == "detected hardware");
}

void test_resolution_precedence() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", kAie2pBackendId, StubFactory("aie2p"));
    registry.replace_backend("phi4", kAie4BackendId, StubFactory("aie4"));

    std::string source;
    {
        // FLM_BACKEND beats the detected hardware.
        ScopedBackendEnv env(kAie4BackendId);
        TEST_REQUIRE(resolve_backend_id("phi4", "aie2p", "", &source) == kAie4BackendId);
        TEST_REQUIRE(source == "FLM_BACKEND");

        // --backend beats both.
        TEST_REQUIRE(resolve_backend_id("phi4", "aie2p", kAie2pBackendId, &source) ==
                     kAie2pBackendId);
        TEST_REQUIRE(source == "--backend");
    }

    // An empty FLM_BACKEND is the same as an unset one.
    ScopedBackendEnv empty("");
    TEST_REQUIRE(resolve_backend_id("phi4", "aie4", "", &source) == kAie4BackendId);
    TEST_REQUIRE(source == "detected hardware");
}

void test_resolution_rejects_with_a_readable_message() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("llama3", kAie2pBackendId, StubFactory("aie2p"));

    // A family with no engine for this hardware: the message has to name both
    // what was asked for and what the build does have.
    const std::string not_built = RequireThrows(
        [&] { resolve_backend_id("llama3", "aie2p", kAie4BackendId); });
    RequireContains(not_built, "--backend");
    RequireContains(not_built, "not compiled into this build");
    RequireContains(not_built, "aie2p");

    // Same for hardware nobody has an engine for yet.
    const std::string unknown_hardware =
        RequireThrows([&] { resolve_backend_id("llama3", "gpu"); });
    RequireContains(unknown_hardware, "detected hardware");
    RequireContains(unknown_hardware, "gpu");

    // The env var gets named in the message too, so the user can find it.
    ScopedBackendEnv env("bogus");
    const std::string from_env =
        RequireThrows([&] { resolve_backend_id("llama3", "aie2p"); });
    RequireContains(from_env, "FLM_BACKEND");

    // And a family that has nothing at all still says so rather than crashing.
    const std::string no_family =
        RequireThrows([&] { resolve_backend_id("nosuchfamily", "aie2p"); });
    RequireContains(no_family, "(none)");
}

}  // namespace

int main() {
    RunTest(test_register_and_create, "register and create");
    RunTest(test_traits_are_kept_per_backend, "traits are kept per backend");
    RunTest(test_backend_defaults, "backend policy defaults");
    RunTest(test_duplicate_registration_is_rejected, "duplicate registration is rejected");
    RunTest(test_replace_backend_is_the_test_seam, "replace_backend is the test seam");
    RunTest(test_unknown_id_names_what_exists, "unknown id names what exists");
    RunTest(test_the_hardware_is_the_default, "the hardware is the default");
    RunTest(test_resolution_precedence, "resolution precedence");
    RunTest(test_resolution_rejects_with_a_readable_message,
            "resolution rejects with a readable message");
    std::cout << "All model backend tests passed\n";
    return 0;
}
