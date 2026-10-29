/// \file test_model_frontend.cpp
/// \brief Hardware-free frontend registry and C loader tests
#include "AutoModel/model_backend.hpp"
#include "AutoModel/model_extension_frontend.hpp"
#include "AutoModel/model_frontend.hpp"
#include "../phi4_core/test_support.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#ifndef FLM_TEST_EXTENSION_PATH
#error FLM_TEST_EXTENSION_PATH must name the fake extension library
#endif
#ifndef FLM_TEST_INCOMPATIBLE_EXTENSION_PATH
#error FLM_TEST_INCOMPATIBLE_EXTENSION_PATH must name the incompatible library
#endif

namespace flm::frontend {
void register_builtin_frontends(FrontendRegistry&) {}

FrontendFactory make_extension_frontend_factory(
    ExtensionFrontendRegistration, std::string, std::string,
    std::shared_ptr<void> lifetime) {
    return [lifetime = std::move(lifetime)](void*) {
        auto* value = new std::uintptr_t(84);
        return FrontendPtr(
            reinterpret_cast<AutoModel*>(value),
            FrontendDeleter{
                nullptr,
                [](void*, void* object) {
                    delete static_cast<std::uintptr_t*>(object);
                },
                lifetime,
            });
    };
}
}  // namespace flm::frontend

namespace flm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace flm::backend

namespace {

using flm::frontend::FrontendDeleter;
using flm::frontend::FrontendPtr;
using flm::frontend::FrontendRegistry;
using flm::backend::BackendContext;
using flm::backend::BackendRegistry;

FrontendPtr MarkerFrontend(std::uintptr_t marker) {
    auto* value = new std::uintptr_t(marker);
    return FrontendPtr(
        reinterpret_cast<AutoModel*>(value),
        FrontendDeleter{
            nullptr,
            [](void*, void* object) {
                delete static_cast<std::uintptr_t*>(object);
            },
            nullptr,
        });
}

std::uintptr_t Marker(const FrontendPtr& frontend) {
    return *static_cast<std::uintptr_t*>(
        static_cast<void*>(frontend.get()));
}

std::filesystem::path WriteManifest(
    const std::filesystem::path& directory, const std::string& name,
    const nlohmann::json& manifest) {
    std::filesystem::create_directories(directory);
    const auto path = directory / name;
    std::ofstream stream(path);
    stream << manifest.dump(2);
    return path;
}

nlohmann::json ReadJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    TEST_REQUIRE(stream.is_open());
    return nlohmann::json::parse(stream);
}

void test_register_and_create() {
    FrontendRegistry registry;
    registry.register_frontend(
        "z-family", [](void*) { return MarkerFrontend(7); });
    registry.register_frontend(
        "a-family", [](void*) { return MarkerFrontend(9); });

    TEST_REQUIRE(registry.has("z-family"));
    TEST_REQUIRE(!registry.has("missing"));
    const auto families = registry.available();
    TEST_REQUIRE(families.size() == 2);
    TEST_REQUIRE(families[0] == "a-family");
    TEST_REQUIRE(families[1] == "z-family");

    auto frontend = registry.create("z-family", nullptr);
    TEST_REQUIRE(Marker(frontend) == 7);
}

void test_registration_validation() {
    FrontendRegistry registry;
    RequireThrows([&] {
        registry.register_frontend(
            "", [](void*) { return MarkerFrontend(1); });
    });
    RequireThrows([&] {
        registry.register_frontend("family", nullptr);
    });

    registry.register_frontend(
        "family", [](void*) { return MarkerFrontend(1); });
    RequireContains(
        RequireThrows([&] {
            registry.register_frontend(
                "family", [](void*) { return MarkerFrontend(2); });
        }),
        "already registered");
    RequireContains(
        RequireThrows([&] { registry.create("missing", nullptr); }),
        "no frontend");

    registry.register_frontend(
        "null-family", [](void*) { return FrontendPtr{}; });
    RequireContains(
        RequireThrows(
            [&] { registry.create("null-family", nullptr); }),
        "returned null");
}

void test_real_shared_library_loads_through_c_boundary() {
    FrontendRegistry registry;
    registry.register_frontend(
        "plugin-test", [](void*) { return MarkerFrontend(42); });
    registry.load_extension(FLM_TEST_EXTENSION_PATH, {});

    TEST_REQUIRE(registry.has("plugin-test"));
    const auto catalogs = registry.catalog_overlays();
    TEST_REQUIRE(catalogs.size() == 1);
    TEST_REQUIRE(
        std::filesystem::path(catalogs[0].path).filename() ==
        "fake-model-catalog.json");
    auto frontend = registry.create("plugin-test", nullptr);
    TEST_REQUIRE(Marker(frontend) == 42);
    auto& backends = BackendRegistry::instance();
    TEST_REQUIRE(backends.has_provider(
        "plugin-test", "flm", "plugin-provider"));
    TEST_REQUIRE(
        backends.traits("plugin-test", "flm", "plugin-provider")
            .max_context_length == 4096);

    BackendContext context;
    flm_rt::device device;
    context.model_path = "fake-model";
    context.device = &device;
    auto backend = backends.create(
        "plugin-test", "flm", "plugin-provider", context);
    TEST_REQUIRE(backend->id() == "flm");
    TEST_REQUIRE(backend->detail() == "fake stable C backend");
    TEST_REQUIRE(backend->supports_state());
    std::vector<int> tokens{1, 2, 3};
    TEST_REQUIRE(backend->engine().prefill(tokens).size() == 4);
    TEST_REQUIRE(backend->engine().get_current_context_length() == 3);
    TEST_REQUIRE(backend->engine().forward(4).size() == 4);
    TEST_REQUIRE(backend->engine().get_current_context_length() == 4);
    TEST_REQUIRE(backend->engine().checkpoint() == 4);
    backend->engine().set_context_length(1);
    TEST_REQUIRE(backend->engine().restore() == 4);
    TEST_REQUIRE(!backend->engine().supports_speculation());
    TEST_REQUIRE(!backend->poisoned());
    RequireContains(
        RequireThrows(
            [&] { (void)backend->engine().get_k_cache(0, 0); }),
        "does not support get_k_cache");
    RequireContains(
        RequireThrows([&] { (void)backend->engine().forward(-1); }),
        "fake forward failure");
    RequireContains(
        RequireThrows([&] {
            flm_prefill_payload_v1 invalid{
                sizeof(flm_prefill_payload_v1), {}, nullptr};
            (void)backend->engine().prefill(tokens, &invalid);
        }),
        "another frontend provider");
}

void test_npu_runtime_mismatch_is_rejected_without_registration() {
    FrontendRegistry registry;
    const std::string message = RequireThrows([&] {
        registry.load_extension(FLM_TEST_INCOMPATIBLE_EXTENSION_PATH, {});
    });
    RequireContains(message, "targets NPU runtime");
    TEST_REQUIRE(!BackendRegistry::instance().has_provider(
        "plugin-test", "flm", "plugin-provider"));
}

void test_registration_error_crosses_stable_c_boundary() {
    FrontendRegistry registry;
    RequireContains(
        RequireThrows([&] {
            registry.load_extension(
                FLM_TEST_FAILING_EXTENSION_PATH, {});
        }),
        "fake registration failure");
    TEST_REQUIRE(!BackendRegistry::instance().has_provider(
        "plugin-test", "flm", "plugin-provider"));
}

void test_backend_without_optional_capabilities() {
    FrontendRegistry registry;
    registry.register_frontend(
        "no-capability-plugin-test", [](void*) { return MarkerFrontend(43); });
    registry.load_extension(FLM_TEST_NO_CAPABILITY_EXTENSION_PATH, {});

    BackendContext context;
    flm_rt::device device;
    context.model_path = "fake-model";
    context.device = &device;
    auto backend = BackendRegistry::instance().create(
        "no-capability-plugin-test", "flm", "plugin-provider", context);

    TEST_REQUIRE(!backend->supports_state());
    TEST_REQUIRE(backend->engine().forward(1).size() == 4);
    RequireContains(
        RequireThrows([&] { (void)backend->engine().checkpoint(); }),
        "does not support checkpoint");
    RequireContains(
        RequireThrows(
            [&] { (void)backend->engine().get_k_cache(0, 0); }),
        "does not support get_k_cache");
    TEST_REQUIRE(!backend->engine().supports_speculation());
    TEST_REQUIRE(!backend->poisoned());
}

void test_complete_model_registration_and_opaque_payload() {
    FrontendRegistry registry;
    registry.load_extension(FLM_TEST_COMPLETE_EXTENSION_PATH, {});
    TEST_REQUIRE(registry.has("complete-plugin-test"));
    TEST_REQUIRE(Marker(
        registry.create("complete-plugin-test", nullptr)) == 84);

    BackendContext context;
    flm_rt::device device;
    context.model_path = "fake-model";
    context.frontend_provider = "plugin-provider";
    context.device = &device;
    auto backend = BackendRegistry::instance().create(
        "complete-plugin-test", "flm", "plugin-provider", context);

    int private_payload = 7;
    static constexpr char kProvider[] = "plugin-provider";
    flm_prefill_payload_v1 payload{
        sizeof(flm_prefill_payload_v1),
        {kProvider, sizeof(kProvider) - 1}, &private_payload};
    std::vector<int> tokens{1, 2};
    TEST_REQUIRE(
        backend->engine().prefill(tokens, &payload).size() == 4);
    flm_prefill_payload_v1 invalid{
        sizeof(flm_prefill_payload_v1), {}, nullptr};
    RequireContains(
        RequireThrows(
            [&] { (void)backend->engine().prefill(tokens, &invalid); }),
        "incomplete opaque payload");
}

void test_duplicate_extension_registration_is_atomic() {
    FrontendRegistry registry;
    registry.register_frontend(
        "plugin-test", [](void*) { return MarkerFrontend(42); });
    registry.load_extension(FLM_TEST_EXTENSION_PATH, {});

    RequireContains(
        RequireThrows([&] {
            registry.load_extension(FLM_TEST_EXTENSION_PATH, {});
        }),
        "duplicates model tag");
    TEST_REQUIRE(Marker(registry.create("plugin-test", nullptr)) == 42);
}

void test_catalog_validation_is_atomic() {
    const std::filesystem::path manifest_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_PATH);
    auto manifest = ReadJson(manifest_path);
    const auto directory = manifest_path.parent_path() / "invalid-catalog";
    std::filesystem::create_directories(directory);
    const auto library_name =
        std::filesystem::u8path(manifest.at("library").get<std::string>());
    std::filesystem::copy_file(
        manifest_path.parent_path() / library_name,
        directory / library_name,
        std::filesystem::copy_options::overwrite_existing);

    nlohmann::json catalog = {
        {"schema_version", 1},
        {"models",
         {{"plugin-model",
           {{"1b",
             {{"name", "Plugin model"},
              {"supported_platforms", {"aie2p"}},
              {"backend", "flm"},
              {"backend_provider", "missing-provider"},
              {"default_context_length", 4096},
              {"max_prefill_len", 512},
              {"details", {{"family", "plugin-test"}}}}}}}}}};
    const auto catalog_path = directory / "invalid-provider.json";
    std::ofstream(catalog_path) << catalog.dump(2);
    manifest["catalog"] = catalog_path.filename().generic_string();

    const auto cleanup_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_CLEANUP_PATH);
    std::filesystem::remove(cleanup_path);
    {
        FrontendRegistry registry;
        registry.register_frontend(
            "plugin-test", [](void*) { return MarkerFrontend(1); });
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    WriteManifest(
                        directory, "invalid-provider-manifest.json", manifest)
                        .string(),
                    {});
            }),
            "missing-provider");
        TEST_REQUIRE(!BackendRegistry::instance().has_provider(
            "plugin-test", "flm", "plugin-provider"));
    }
    TEST_REQUIRE(std::filesystem::exists(cleanup_path));

    std::filesystem::remove(cleanup_path);
    {
        FrontendRegistry registry;
        registry.register_frontend(
            "plugin-test", [](void*) { return MarkerFrontend(1); });
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    FLM_TEST_EXTENSION_PATH,
                    {"plugin-model:1b"});
            }),
            "duplicates");
        TEST_REQUIRE(!BackendRegistry::instance().has_provider(
            "plugin-test", "flm", "plugin-provider"));
    }
    TEST_REQUIRE(std::filesystem::exists(cleanup_path));

    catalog["models"]["plugin-model"]["1b"]["backend_provider"] =
        "plugin-provider";
    catalog["models"]["plugin-model"]["1b"]["supported_platforms"] =
        nlohmann::json::array();
    std::ofstream(catalog_path) << catalog.dump(2);
    std::filesystem::remove(cleanup_path);
    {
        FrontendRegistry registry;
        registry.register_frontend(
            "plugin-test", [](void*) { return MarkerFrontend(1); });
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    WriteManifest(
                        directory, "invalid-platform-manifest.json", manifest)
                        .string(),
                    {});
            }),
            "supported_platforms");
        TEST_REQUIRE(!BackendRegistry::instance().has_provider(
            "plugin-test", "flm", "plugin-provider"));
    }
    TEST_REQUIRE(!std::filesystem::exists(cleanup_path));
}

void test_extension_context_is_destroyed_after_registry() {
    const std::filesystem::path cleanup_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_CLEANUP_PATH);
    std::filesystem::remove(cleanup_path);

    {
        FrontendRegistry registry;
        registry.register_frontend(
            "plugin-test", [](void*) { return MarkerFrontend(1); });
        registry.load_extension(FLM_TEST_EXTENSION_PATH, {});
        TEST_REQUIRE(!std::filesystem::exists(cleanup_path));
    }
    TEST_REQUIRE(std::filesystem::exists(cleanup_path));
}

void test_explicit_unload_preserves_host_frontends() {
    const std::filesystem::path cleanup_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_CLEANUP_PATH);
    std::filesystem::remove(cleanup_path);

    FrontendRegistry registry;
    registry.register_frontend(
        "plugin-test", [](void*) { return MarkerFrontend(42); });
    registry.load_extension(FLM_TEST_EXTENSION_PATH, {});
    registry.unload_extensions();

    TEST_REQUIRE(
        Marker(registry.create("plugin-test", nullptr)) == 42);
    TEST_REQUIRE(!BackendRegistry::instance().has_provider(
        "plugin-test", "flm", "plugin-provider"));
    TEST_REQUIRE(std::filesystem::exists(cleanup_path));
}

void test_backend_keeps_provider_library_and_context_loaded() {
    const std::filesystem::path cleanup_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_CLEANUP_PATH);
    std::filesystem::remove(cleanup_path);

    std::unique_ptr<flm::backend::ModelBackend> backend;
    {
        FrontendRegistry registry;
        registry.register_frontend(
            "plugin-test", [](void*) { return MarkerFrontend(1); });
        registry.load_extension(FLM_TEST_EXTENSION_PATH, {});
        BackendContext context;
        flm_rt::device device;
        context.model_path = "fake-model";
        context.device = &device;
        backend = BackendRegistry::instance().create(
            "plugin-test", "flm", "plugin-provider", context);
        TEST_REQUIRE(!std::filesystem::exists(cleanup_path));
    }

    TEST_REQUIRE(!BackendRegistry::instance().has_provider(
        "plugin-test", "flm", "plugin-provider"));
    TEST_REQUIRE(backend->detail() == "fake stable C backend");
    TEST_REQUIRE(!std::filesystem::exists(cleanup_path));
    backend.reset();
    TEST_REQUIRE(std::filesystem::exists(cleanup_path));
}

void test_frontend_keeps_provider_library_and_context_loaded() {
    const std::filesystem::path cleanup_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_CLEANUP_PATH);
    std::filesystem::remove(cleanup_path);

    FrontendPtr frontend;
    {
        FrontendRegistry registry;
        registry.load_extension(FLM_TEST_COMPLETE_EXTENSION_PATH, {});
        frontend = registry.create("complete-plugin-test", nullptr);
        TEST_REQUIRE(!std::filesystem::exists(cleanup_path));
    }

    TEST_REQUIRE(!std::filesystem::exists(cleanup_path));
    TEST_REQUIRE(Marker(frontend) == 84);
    frontend.reset();
    TEST_REQUIRE(std::filesystem::exists(cleanup_path));
}

void test_unicode_bundle_path_and_private_dependency() {
    const auto source_manifest =
        std::filesystem::u8path(FLM_TEST_EXTENSION_PATH);
    auto manifest = ReadJson(source_manifest);
    const auto source_library =
        source_manifest.parent_path() /
        std::filesystem::u8path(
            manifest.at("library").get<std::string>());
    const std::filesystem::path source_dependency =
        std::filesystem::u8path(FLM_TEST_EXTENSION_DEPENDENCY_PATH);
    const auto directory =
        source_manifest.parent_path() /
        std::filesystem::path(
            std::u8string(u8"model-\u6269\u5c55"));
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(
        source_library, directory / source_library.filename(),
        std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(
        source_dependency, directory / source_dependency.filename(),
        std::filesystem::copy_options::overwrite_existing);
    if (manifest.contains("catalog")) {
        const auto catalog_name = std::filesystem::u8path(
            manifest.at("catalog").get<std::string>());
        std::filesystem::copy_file(
            source_manifest.parent_path() / catalog_name,
            directory / catalog_name,
            std::filesystem::copy_options::overwrite_existing);
    }
    const auto copied_manifest =
        WriteManifest(directory, "extension.json", manifest);
    const auto encoded = copied_manifest.generic_u8string();

    FrontendRegistry registry;
    registry.register_frontend(
        "plugin-test", [](void*) { return MarkerFrontend(1); });
    registry.load_extension(
        std::string(
            reinterpret_cast<const char*>(encoded.data()),
            encoded.size()),
        {});
    TEST_REQUIRE(BackendRegistry::instance().has_provider(
        "plugin-test", "flm", "plugin-provider"));
}

void test_invalid_manifests_are_rejected_before_registration() {
    const std::filesystem::path valid_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_PATH);
    nlohmann::json valid;
    {
        std::ifstream stream(valid_path);
        stream >> valid;
    }
    const auto directory = valid_path.parent_path() / "invalid-manifests";
    std::filesystem::create_directories(directory);
    const auto library_name =
        std::filesystem::u8path(valid["library"].get<std::string>());
    valid.erase("catalog");
    std::filesystem::copy_file(
        valid_path.parent_path() / library_name,
        directory / library_name,
        std::filesystem::copy_options::overwrite_existing);

    {
        const auto malformed = directory / "malformed.json";
        std::ofstream(malformed) << "{";
        FrontendRegistry registry;
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    malformed.string(), {});
            }),
            "Cannot parse");
        TEST_REQUIRE(!registry.has("plugin-test"));
    }
    {
        auto manifest = valid;
        manifest["library"] = "missing-library";
        FrontendRegistry registry;
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    WriteManifest(directory, "missing.json", manifest).string(),
                    {});
            }),
            "does not exist");
        TEST_REQUIRE(!registry.has("plugin-test"));
    }
    {
        auto manifest = valid;
        manifest["library"] =
            "../" + library_name.generic_string();
        FrontendRegistry registry;
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    WriteManifest(directory, "escaping.json", manifest).string(),
                    {});
            }),
            "escapes its bundle");
        TEST_REQUIRE(!registry.has("plugin-test"));
    }
    {
        auto manifest = valid;
        manifest["name"] = "wrong-provider-name";
        FrontendRegistry registry;
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    WriteManifest(directory, "identity.json", manifest).string(),
                    {});
            }),
            "identity does not match");
        TEST_REQUIRE(!registry.has("plugin-test"));
    }
}

void test_manifest_schema_fixtures_match_runtime_contract() {
    const auto schema = ReadJson(FLM_TEST_MANIFEST_SCHEMA);
    TEST_REQUIRE(
        schema.at("$id") ==
        "https://github.com/ROCm/FastFlowLM/schemas/"
        "model-extension-manifest-v1.schema.json");
    TEST_REQUIRE(schema.at("properties")
                     .at("schema_version")
                     .at("const") == 1);
    TEST_REQUIRE(schema.at("type") == "object");
    TEST_REQUIRE(schema.at("additionalProperties") == false);
    TEST_REQUIRE(
        schema.at("required") ==
        nlohmann::json::array(
            {"schema_version", "name", "version", "library"}));
    TEST_REQUIRE(
        schema.at("properties").at("library").at("oneOf").size() == 2);

    const std::filesystem::path generated_path =
        std::filesystem::u8path(FLM_TEST_EXTENSION_PATH);
    const auto generated = ReadJson(generated_path);

    auto valid = ReadJson(FLM_TEST_VALID_MANIFEST_FIXTURE);
    valid["library"] = generated.at("library");
    const auto valid_path = WriteManifest(
        generated_path.parent_path(), "valid-schema-fixture.json", valid);
    {
        FrontendRegistry registry;
        registry.register_frontend(
            "plugin-test", [](void*) { return MarkerFrontend(1); });
        registry.load_extension(valid_path.string(), {});
        TEST_REQUIRE(registry.has("plugin-test"));
    }

    auto invalid = ReadJson(FLM_TEST_INVALID_MANIFEST_FIXTURE);
    invalid["library"] = generated.at("library");
    const auto invalid_path = WriteManifest(
        generated_path.parent_path(), "invalid-schema-fixture.json", invalid);
    {
        FrontendRegistry registry;
        RequireContains(
            RequireThrows([&] {
                registry.load_extension(
                    invalid_path.string(), {});
            }),
            "unsupported field 'unsupported'");
        TEST_REQUIRE(!registry.has("plugin-test"));
    }

    std::filesystem::remove(valid_path);
    std::filesystem::remove(invalid_path);
}

}  // namespace

int main() {
    RunTest(test_register_and_create, "register and create");
    RunTest(test_registration_validation, "registration validation");
    RunTest(test_real_shared_library_loads_through_c_boundary,
            "real shared library loads through C boundary");
    RunTest(test_npu_runtime_mismatch_is_rejected_without_registration,
            "NPU runtime mismatch is rejected without registration");
    RunTest(test_registration_error_crosses_stable_c_boundary,
            "registration error crosses stable C boundary");
    RunTest(test_backend_without_optional_capabilities,
            "backend without optional capabilities");
    RunTest(test_complete_model_registration_and_opaque_payload,
            "complete model registration and opaque payload");
    RunTest(test_duplicate_extension_registration_is_atomic,
            "duplicate extension registration is atomic");
    RunTest(test_catalog_validation_is_atomic,
            "catalog validation is atomic");
    RunTest(test_extension_context_is_destroyed_after_registry,
            "extension context is destroyed after registry");
    RunTest(test_explicit_unload_preserves_host_frontends,
            "explicit extension unload");
    RunTest(test_backend_keeps_provider_library_and_context_loaded,
            "backend keeps provider library and context loaded");
    RunTest(test_frontend_keeps_provider_library_and_context_loaded,
            "frontend keeps provider library and context loaded");
    RunTest(test_unicode_bundle_path_and_private_dependency,
            "Unicode bundle path and private dependency");
    RunTest(test_invalid_manifests_are_rejected_before_registration,
            "invalid manifests are rejected before registration");
    RunTest(test_manifest_schema_fixtures_match_runtime_contract,
            "manifest schema fixtures match runtime contract");
    std::cout << "All model frontend tests passed\n";
    return 0;
}
