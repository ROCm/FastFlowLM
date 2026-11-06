/// \file test_model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
/// \note  Deliberately free of NPU hardware and of any prebuilt engine library:
///        every backend here is a stub, so this builds and runs on Linux CI
///        where the phi4_core suite cannot. register_builtin_backends
///        is stubbed out below for the same reason.
/// \note  A backend id names an execution/artifact contract such as "flm" or
///        "core". A provider names one implementation of that contract. Neither
///        is a platform id; these rules do not choose which silicon to run on.
#include "AutoModel/model_backend.hpp"
#include "../phi4_core/test_support.hpp"

#include <cstdlib>
#include <memory>
#include <algorithm>
#include <string>

using flm::backend::BackendContext;
using flm::backend::BackendRegistry;
using flm::backend::BackendTraits;
using flm::backend::kFlmBackendId;
using flm::backend::kCoreBackendId;
using flm::backend::ModelBackend;
using flm::backend::resolve_backend_selection;

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
    registry.register_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.register_backend("phi4", kCoreBackendId, StubFactory("core"),
                              BackendTraits{false, false, 4096});

    TEST_REQUIRE(registry.has("phi4", "flm"));
    TEST_REQUIRE(!registry.has("phi4", "bogus"));
    TEST_REQUIRE(!registry.has("llama3", "flm"));

    // available() is sorted, which is what makes the error messages stable.
    const auto ids = registry.available("phi4");
    TEST_REQUIRE(ids.size() == 2);
    TEST_REQUIRE(ids[0] == "core");
    TEST_REQUIRE(ids[1] == "flm");
    TEST_REQUIRE(registry.available("llama3").empty());

    BackendContext context;
    auto backend = registry.create("phi4", "core", context);
    TEST_REQUIRE(backend != nullptr);
    TEST_REQUIRE(backend->id() == "core");
}

void test_traits_are_kept_per_backend() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.register_backend("phi4", kCoreBackendId, StubFactory("core"),
                              BackendTraits{false, false, 4096});

    // The defaults describe the FastFlowLM NPU engines, i.e. flm.
    const auto flm = registry.traits("phi4", kFlmBackendId);
    TEST_REQUIRE(flm.needs_npu_xclbin);
    TEST_REQUIRE(flm.supports_preemption);
    TEST_REQUIRE(flm.max_context_length == 0);

    const auto core = registry.traits("phi4", kCoreBackendId);
    TEST_REQUIRE(!core.needs_npu_xclbin);
    TEST_REQUIRE(!core.supports_preemption);
    TEST_REQUIRE(core.max_context_length == 4096);
}

void test_providers_are_kept_per_backend_contract() {
    auto registry = MakeRegistry();
    registry.register_backend(
        "gemma4e", kFlmBackendId, StubFactory("builtin"));
    registry.register_provider(
        "gemma4e", kFlmBackendId, "gemma4-open",
        StubFactory("gemma4-open"),
        BackendTraits{true, false, 32768});

    const auto providers =
        registry.available_providers("gemma4e", kFlmBackendId);
    TEST_REQUIRE(providers.size() == 2);
    TEST_REQUIRE(providers[0] == "builtin");
    TEST_REQUIRE(providers[1] == "gemma4-open");
    TEST_REQUIRE(registry.has_provider(
        "gemma4e", kFlmBackendId, "gemma4-open"));

    BackendContext context;
    auto backend = registry.create(
        "gemma4e", kFlmBackendId, "gemma4-open", context);
    TEST_REQUIRE(backend->id() == "gemma4-open");
    TEST_REQUIRE(
        !registry.traits(
            "gemma4e", kFlmBackendId, "gemma4-open")
             .supports_preemption);

    const auto inventory = registry.provider_inventory();
    TEST_REQUIRE(
        inventory.at("gemma4e").at(kFlmBackendId).size() == 2);
    RequireContains(
        RequireThrows([&] {
            registry.register_provider(
                "gemma4e", kFlmBackendId, "gemma4-open",
                StubFactory("duplicate"));
        }),
        "already registered");
}

void test_backend_defaults() {
    StubBackend backend(kFlmBackendId);
    TEST_REQUIRE(backend.detail().empty());
    TEST_REQUIRE(backend.max_decode_length() == 0);
    TEST_REQUIRE(backend.supports_preemption());
    TEST_REQUIRE(backend.forwards_past_eos());
    TEST_REQUIRE(!backend.aligns_prefill_chunks());
    TEST_REQUIRE(!backend.poisoned());
    TEST_REQUIRE(!backend.forced_eos_ids().has_value());
    TEST_REQUIRE(!backend.forced_bos_id().has_value());
}

void test_duplicate_registration_is_rejected() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kFlmBackendId, StubFactory("first"));
    const std::string message = RequireThrows([&] {
        registry.register_backend("phi4", kFlmBackendId, StubFactory("second"));
    });
    RequireContains(message, "already registered");

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", kFlmBackendId, context)->id() == "first");

    RequireThrows([&] { registry.register_backend("", kFlmBackendId, StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "", StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "x", nullptr); });
}

void test_replace_backend_is_the_test_seam() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kFlmBackendId, StubFactory("real"));
    registry.replace_backend("phi4", kFlmBackendId, StubFactory("stub"));

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", kFlmBackendId, context)->id() == "stub");
    TEST_REQUIRE(registry.available("phi4").size() == 1);

    // It also registers a backend that was not there before.
    registry.replace_backend("llama3", kFlmBackendId, StubFactory("fresh"));
    TEST_REQUIRE(registry.create("llama3", kFlmBackendId, context)->id() == "fresh");
}

void test_unknown_id_names_what_exists() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kFlmBackendId, StubFactory("flm"));

    BackendContext context;
    const std::string message =
        RequireThrows([&] { registry.create("phi4", "bogus", context); });
    RequireContains(message, "bogus");
    RequireContains(message, "flm");

    const std::string empty =
        RequireThrows([&] { registry.create("llama3", kFlmBackendId, context); });
    RequireContains(empty, "(none)");
}

void test_the_catalog_entry_is_the_default() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kCoreBackendId, StubFactory("core"));

    std::string source;
    auto selection = resolve_backend_selection(
        "phi4", kFlmBackendId, "builtin", "", &source);
    TEST_REQUIRE(selection.backend == kFlmBackendId);
    TEST_REQUIRE(selection.provider == "builtin");
    TEST_REQUIRE(source == "model catalog");
    selection = resolve_backend_selection(
        "phi4", kCoreBackendId, "builtin", "", &source);
    TEST_REQUIRE(selection.backend == kCoreBackendId);
    TEST_REQUIRE(selection.provider == "builtin");
    TEST_REQUIRE(source == "model catalog");
}

void test_catalog_selects_exact_backend_provider_pair() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend(
        "provider-test", kFlmBackendId, StubFactory("builtin"));
    if (!registry.has_provider(
            "provider-test", kFlmBackendId, "gemma4-open")) {
        registry.register_provider(
            "provider-test", kFlmBackendId, "gemma4-open",
            StubFactory("gemma4-open"));
    }

    const auto selected = resolve_backend_selection(
        "provider-test", kFlmBackendId, "gemma4-open");
    TEST_REQUIRE(selected.backend == kFlmBackendId);
    TEST_REQUIRE(selected.provider == "gemma4-open");
    RequireContains(
        RequireThrows([] {
            resolve_backend_selection(
                "provider-test", kFlmBackendId, "missing");
        }),
        "missing");

    registry.register_provider(
        "provider-switch-test", "iron", "worker",
        StubFactory("worker"));
    registry.register_provider(
        "provider-switch-test", "iron", "native",
        StubFactory("native"));
    TEST_REQUIRE(
        resolve_backend_selection(
            "provider-switch-test", "iron", "worker").provider == "worker");
    TEST_REQUIRE(
        resolve_backend_selection(
            "provider-switch-test", "iron", "native").provider == "native");
}

void test_resolution_precedence() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kCoreBackendId, StubFactory("core"));

    std::string source;
    {
        // FLM_BACKEND is a demand, but registration alone cannot make the
        // requested backend compatible with this model package.
        ScopedBackendEnv env(kCoreBackendId);
        const auto incompatible = RequireThrows([&] {
            resolve_backend_selection(
                "phi4", kFlmBackendId, "builtin", "", &source);
        });
        RequireContains(incompatible, "FLM_BACKEND");
        RequireContains(incompatible, "not compatible");

        // --backend beats both.
        TEST_REQUIRE(
            resolve_backend_selection(
                "phi4", kFlmBackendId, "builtin", kFlmBackendId, &source)
                .backend == kFlmBackendId);
        TEST_REQUIRE(source == "--backend");
    }

    // An empty FLM_BACKEND is the same as an unset one.
    ScopedBackendEnv empty("");
    TEST_REQUIRE(
        resolve_backend_selection(
            "phi4", kCoreBackendId, "builtin", "", &source)
            .backend == kCoreBackendId);
    TEST_REQUIRE(source == "model catalog");
}

void test_adding_core_leaves_the_flm_families_alone() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("llama3", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kCoreBackendId, StubFactory("core"));

    // This is the shape of a corelib build: core for the one family that has a
    // corelib engine, flm for that family too, and flm alone everywhere else.
    // Linking core must not disturb any of the families it says nothing about.
    std::string source;
    TEST_REQUIRE(
        resolve_backend_selection(
            "llama3", kFlmBackendId, "builtin", "", &source)
            .backend == kFlmBackendId);
    TEST_REQUIRE(source == "model catalog");
    TEST_REQUIRE(
        resolve_backend_selection(
            "phi4", kCoreBackendId, "builtin", "", &source)
            .backend == kCoreBackendId);
    TEST_REQUIRE(source == "model catalog");

    // A registered backend is not automatically compatible with every model
    // package for the family.
    RequireContains(
        RequireThrows([&] {
            resolve_backend_selection(
                "phi4", kFlmBackendId, "builtin", kCoreBackendId);
        }),
        "not compatible");

    // Every backend the build links, whichever family registered it.
    const auto ids = registry.backend_ids();
    TEST_REQUIRE(std::find(ids.begin(), ids.end(), kFlmBackendId) != ids.end());
    TEST_REQUIRE(std::find(ids.begin(), ids.end(), kCoreBackendId) != ids.end());
}

void test_resolution_rejects_with_a_readable_message() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("llama3", kFlmBackendId, StubFactory("flm"));

    const std::string incompatible = RequireThrows([&] {
        resolve_backend_selection(
            "llama3", kFlmBackendId, "builtin", kCoreBackendId);
    });
    RequireContains(incompatible, "--backend");
    RequireContains(incompatible, "not compatible");
    RequireContains(incompatible, "flm/builtin");

    // A catalog pair that this build does not provide fails explicitly.
    const std::string unknown_provider =
        RequireThrows([&] {
            resolve_backend_selection("llama3", kFlmBackendId, "missing");
        });
    RequireContains(unknown_provider, "model catalog");
    RequireContains(unknown_provider, "missing");

    // The env var gets named in the message too, so the user can find it.
    {
        ScopedBackendEnv env("bogus");
        const std::string from_env = RequireThrows([&] {
            resolve_backend_selection("llama3", kFlmBackendId, "builtin");
        });
        RequireContains(from_env, "FLM_BACKEND");
    }

    // And a family that has nothing at all still says so rather than crashing.
    const std::string no_family =
        RequireThrows([&] {
            resolve_backend_selection(
                "nosuchfamily", kFlmBackendId, "builtin");
        });
    RequireContains(no_family, "(none)");
}

}  // namespace

int main() {
    RunTest(test_register_and_create, "register and create");
    RunTest(test_traits_are_kept_per_backend, "traits are kept per backend");
    RunTest(test_providers_are_kept_per_backend_contract,
            "providers are kept per backend contract");
    RunTest(test_backend_defaults, "backend policy defaults");
    RunTest(test_duplicate_registration_is_rejected, "duplicate registration is rejected");
    RunTest(test_replace_backend_is_the_test_seam, "replace_backend is the test seam");
    RunTest(test_unknown_id_names_what_exists, "unknown id names what exists");
    RunTest(test_the_catalog_entry_is_the_default, "the catalog entry is the default");
    RunTest(test_catalog_selects_exact_backend_provider_pair,
            "catalog selects exact backend/provider pair");
    RunTest(test_resolution_precedence, "resolution precedence");
    RunTest(test_adding_core_leaves_the_flm_families_alone,
            "adding core leaves the flm families alone");
    RunTest(test_resolution_rejects_with_a_readable_message,
            "resolution rejects with a readable message");
    std::cout << "All model backend tests passed\n";
    return 0;
}
