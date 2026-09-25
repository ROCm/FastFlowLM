/// \file test_model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
/// \note  Deliberately free of NPU hardware and of any prebuilt engine library:
///        every backend here is a stub, so this builds and runs on Linux CI
///        where the phi4_rai suite cannot. register_builtin_backends
///        is stubbed out below for the same reason.
/// \note  A backend id names a kernel provider: "flm", "rai", and one day
///        whatever else grows an engine. It is not a platform id -- the
///        resolution rules below are about which provider a build links,
///        not about which silicon it runs on.
#include "AutoModel/model_backend.hpp"
#include "../phi4_rai/test_support.hpp"

#include <cstdlib>
#include <memory>
#include <algorithm>
#include <string>

using flm::backend::BackendContext;
using flm::backend::BackendRegistry;
using flm::backend::BackendTraits;
using flm::backend::kFlmBackendId;
using flm::backend::kRaiBackendId;
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
    registry.register_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.register_backend("phi4", kRaiBackendId, StubFactory("rai"),
                              BackendTraits{false, false, 4096});

    TEST_REQUIRE(registry.has("phi4", "flm"));
    TEST_REQUIRE(!registry.has("phi4", "bogus"));
    TEST_REQUIRE(!registry.has("llama3", "flm"));

    // available() is sorted, which is what makes the error messages stable.
    const auto ids = registry.available("phi4");
    TEST_REQUIRE(ids.size() == 2);
    TEST_REQUIRE(ids[0] == "flm");
    TEST_REQUIRE(ids[1] == "rai");
    TEST_REQUIRE(registry.available("llama3").empty());

    BackendContext context;
    auto backend = registry.create("phi4", "rai", context);
    TEST_REQUIRE(backend != nullptr);
    TEST_REQUIRE(backend->id() == "rai");
}

void test_traits_are_kept_per_backend() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.register_backend("phi4", kRaiBackendId, StubFactory("rai"),
                              BackendTraits{false, false, 4096});

    // The defaults describe the FastFlowLM NPU engines, i.e. flm.
    const auto flm = registry.traits("phi4", kFlmBackendId);
    TEST_REQUIRE(flm.needs_npu_xclbin);
    TEST_REQUIRE(flm.supports_preemption);
    TEST_REQUIRE(flm.max_context_length == 0);

    const auto rai = registry.traits("phi4", kRaiBackendId);
    TEST_REQUIRE(!rai.needs_npu_xclbin);
    TEST_REQUIRE(!rai.supports_preemption);
    TEST_REQUIRE(rai.max_context_length == 4096);
}

void test_backend_defaults() {
    StubBackend backend(kFlmBackendId);
    TEST_REQUIRE(backend.detail().empty());
    TEST_REQUIRE(backend.max_decode_length() == 0);
    TEST_REQUIRE(backend.supports_preemption());
    TEST_REQUIRE(backend.forwards_past_eos());
    TEST_REQUIRE(!backend.poisoned());
    TEST_REQUIRE(!backend.forced_eos_ids().has_value());
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
    registry.replace_backend("phi4", kRaiBackendId, StubFactory("rai"));

    std::string source;

    // Nothing overrides it, so the entry decides. Both flows are registered
    // for this family at once, which is the point: they are two ways to run
    // phi4 on one machine, not two machines.
    TEST_REQUIRE(resolve_backend_id("phi4", "flm", "", &source) == kFlmBackendId);
    TEST_REQUIRE(source == "model catalog");
    TEST_REQUIRE(resolve_backend_id("phi4", "rai", "", &source) == kRaiBackendId);
    TEST_REQUIRE(source == "model catalog");
}

void test_resolution_precedence() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kRaiBackendId, StubFactory("rai"));

    std::string source;
    {
        // FLM_BACKEND beats what the entry asked for.
        ScopedBackendEnv env(kRaiBackendId);
        TEST_REQUIRE(resolve_backend_id("phi4", "flm", "", &source) == kRaiBackendId);
        TEST_REQUIRE(source == "FLM_BACKEND");

        // --backend beats both.
        TEST_REQUIRE(resolve_backend_id("phi4", "flm", kFlmBackendId, &source) ==
                     kFlmBackendId);
        TEST_REQUIRE(source == "--backend");
    }

    // An empty FLM_BACKEND is the same as an unset one.
    ScopedBackendEnv empty("");
    TEST_REQUIRE(resolve_backend_id("phi4", "rai", "", &source) == kRaiBackendId);
    TEST_REQUIRE(source == "model catalog");
}

void test_adding_rai_leaves_the_flm_families_alone() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("llama3", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kFlmBackendId, StubFactory("flm"));
    registry.replace_backend("phi4", kRaiBackendId, StubFactory("rai"));

    // This is the shape of a corelib build: rai for the one family that has a
    // corelib engine, flm for that family too, and flm alone everywhere else.
    // Linking rai must not disturb any of the families it says nothing about.
    std::string source;
    TEST_REQUIRE(resolve_backend_id("llama3", kFlmBackendId, "", &source) ==
                 kFlmBackendId);
    TEST_REQUIRE(source == "model catalog");
    TEST_REQUIRE(resolve_backend_id("phi4", kRaiBackendId, "", &source) ==
                 kRaiBackendId);
    TEST_REQUIRE(source == "model catalog");

    // rai is registered, but not for llama3, and nothing quietly substitutes
    // another flow: asking for one backend and silently getting a different
    // one would be worse than an error. The catalog cannot ask for this in
    // practice -- model_list prunes an entry whose flow is missing -- so a
    // throw here means a real mismatch rather than a packaging gap.
    RequireThrows([&] { resolve_backend_id("llama3", kRaiBackendId); });
    RequireThrows(
        [&] { resolve_backend_id("llama3", kFlmBackendId, kRaiBackendId); });

    // Every backend the build links, whichever family registered it.
    const auto ids = registry.backend_ids();
    TEST_REQUIRE(std::find(ids.begin(), ids.end(), kFlmBackendId) != ids.end());
    TEST_REQUIRE(std::find(ids.begin(), ids.end(), kRaiBackendId) != ids.end());
}

void test_resolution_rejects_with_a_readable_message() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("llama3", kFlmBackendId, StubFactory("flm"));

    // A family with no engine for the requested provider: the message has to
    // name both what was asked for and what the build does have.
    const std::string not_built = RequireThrows(
        [&] { resolve_backend_id("llama3", "flm", kRaiBackendId); });
    RequireContains(not_built, "--backend");
    RequireContains(not_built, "is not available for model family");
    RequireContains(not_built, "flm");

    // Same for a provider nobody has an engine for yet.
    const std::string unknown_provider =
        RequireThrows([&] { resolve_backend_id("llama3", "gpu"); });
    RequireContains(unknown_provider, "model catalog");
    RequireContains(unknown_provider, "gpu");

    // The env var gets named in the message too, so the user can find it.
    ScopedBackendEnv env("bogus");
    const std::string from_env =
        RequireThrows([&] { resolve_backend_id("llama3", "flm"); });
    RequireContains(from_env, "FLM_BACKEND");

    // And a family that has nothing at all still says so rather than crashing.
    const std::string no_family =
        RequireThrows([&] { resolve_backend_id("nosuchfamily", "flm"); });
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
    RunTest(test_the_catalog_entry_is_the_default, "the catalog entry is the default");
    RunTest(test_resolution_precedence, "resolution precedence");
    RunTest(test_adding_rai_leaves_the_flm_families_alone,
            "adding rai leaves the flm families alone");
    RunTest(test_resolution_rejects_with_a_readable_message,
            "resolution rejects with a readable message");
    std::cout << "All model backend tests passed\n";
    return 0;
}
