/// \file model_frontend.cpp
/// \brief Frontend registry and extension loader
#include "AutoModel/backend_contract.hpp"
#include "AutoModel/model_backend.hpp"
#include "AutoModel/model_frontend.hpp"
#include "AutoModel/model_extension_backend.hpp"
#include "AutoModel/model_extension_frontend.hpp"
#include "model_extension_detail.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace flm::frontend {
namespace {

std::string Join(const std::vector<std::string>& families) {
    if (families.empty()) return "(none)";
    std::ostringstream out;
    for (std::size_t i = 0; i < families.size(); ++i) {
        if (i) out << ", ";
        out << families[i];
    }
    return out.str();
}

std::string Utf8Path(const std::filesystem::path& path) {
    const auto encoded = path.generic_u8string();
    return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

std::filesystem::path NativePath(const std::string& encoded) {
    return std::filesystem::path(std::u8string(
        reinterpret_cast<const char8_t*>(encoded.data()),
        reinterpret_cast<const char8_t*>(encoded.data() + encoded.size())));
}

std::filesystem::path ResolveBundlePath(
    const std::filesystem::path& root, const nlohmann::json& manifest,
    const char* field, bool directory, bool required) {
    const auto value = manifest.find(field);
    if (value == manifest.end()) {
        if (!required) return {};
        throw std::runtime_error(
            "Model extension manifest is missing '" +
            std::string(field) + "'");
    }
    if (!value->is_string() || value->get_ref<const std::string&>().empty()) {
        throw std::runtime_error(
            "Model extension manifest field '" + std::string(field) +
            "' must be a non-empty relative path");
    }
    const std::filesystem::path relative =
        NativePath(value->get<std::string>());
    if (relative.is_absolute()) {
        throw std::runtime_error(
            "Model extension manifest field '" + std::string(field) +
            "' must be relative to the bundle");
    }

    std::error_code error;
    const auto canonical_root =
        std::filesystem::weakly_canonical(root, error);
    if (error) {
        throw std::runtime_error(
            "Cannot resolve model extension root '" + Utf8Path(root) + "'");
    }
    const auto resolved =
        std::filesystem::weakly_canonical(canonical_root / relative, error);
    if (error) {
        throw std::runtime_error(
            "Cannot resolve model extension path '" +
            Utf8Path(canonical_root / relative) + "'");
    }
    const auto within = resolved.lexically_relative(canonical_root);
    if (within.empty() || within.is_absolute() ||
        *within.begin() == "..") {
        throw std::runtime_error(
            "Model extension path escapes its bundle: " +
            Utf8Path(relative));
    }
    const bool valid = directory
        ? std::filesystem::is_directory(resolved, error)
        : std::filesystem::is_regular_file(resolved, error);
    if (error || !valid) {
        throw std::runtime_error(
            "Model extension " + std::string(directory ? "directory" : "file") +
            " does not exist: " + Utf8Path(resolved));
    }
    return resolved;
}

struct RegistrationContext {
    std::vector<ExtensionBackendRegistration> pending_backends;
    std::vector<ExtensionFrontendRegistration> pending_frontends;
    std::string error;
    void* extension_context = nullptr;
    void (*destroy_extension_context)(void*) = nullptr;

    ~RegistrationContext() {
        if (!destroy_extension_context || !extension_context) return;
        try {
            destroy_extension_context(extension_context);
        } catch (...) {
            // Extension callbacks must not throw across the C boundary.
        }
    }
};

struct RegistrationErrorCapture {
    std::string message;

    static void Set(void* context, const char* message,
                    std::size_t size) noexcept {
        try {
            auto& self =
                *static_cast<RegistrationErrorCapture*>(context);
            self.message.assign(message ? message : "",
                                message ? size : 0);
        } catch (...) {
        }
    }

    flm_error_sink_v1 sink() {
        return flm_error_sink_v1{this, Set};
    }
};

void DestroyFrontend(void (*destroy)(void*, void*),
                     void* context, void* object) noexcept {
    if (!destroy || !object) return;
    try {
        destroy(context, object);
    } catch (...) {
        // Extension callbacks must not throw across the C boundary. Destruction
        // has no error channel, so contain a provider violation here.
    }
}

int SetExtensionContext(
    void* opaque, void* extension_context,
    void (*destroy_extension_context)(void*)) noexcept {
    auto& context = *static_cast<RegistrationContext*>(opaque);
    try {
        if (!extension_context || !destroy_extension_context) {
            context.error =
                "extension context and destroy callback are required";
            return 1;
        }
        if (context.extension_context) {
            context.error = "extension context is already registered";
            return 1;
        }
        context.extension_context = extension_context;
        context.destroy_extension_context = destroy_extension_context;
        return 0;
    } catch (...) {
        context.error = "extension context registration failed";
        return 1;
    }
}

int RegisterBackendProvider(
    void* opaque, const char* family, const char* backend,
    const char* provider, const flm_backend_traits_v1* traits,
    flm_backend_create_v1 create, void* factory_context) noexcept {
    auto& context = *static_cast<RegistrationContext*>(opaque);
    try {
        if (!family || !*family || !backend || !*backend ||
            !provider || !*provider) {
            context.error =
                "extension backend family, backend, and provider are required";
            return 1;
        }
        if (std::string_view(provider) ==
            flm::backend::kBuiltinProviderId) {
            context.error =
                "extension backend provider id '" +
                std::string(flm::backend::kBuiltinProviderId) +
                "' is reserved";
            return 1;
        }
        if (!create) {
            context.error = "extension backend factory is null";
            return 1;
        }
        if (!traits ||
            traits->struct_size < FLM_BACKEND_TRAITS_V1_MIN_SIZE) {
            context.error = "extension backend traits are incomplete";
            return 1;
        }
        if (traits->forced_eos_ids_size != 0 &&
            !traits->forced_eos_ids) {
            context.error =
                "extension backend forced EOS ids have no data";
            return 1;
        }
        for (const auto& entry : context.pending_backends) {
            if (entry.family == family && entry.backend == backend &&
                entry.provider == provider) {
                context.error =
                    "extension backend provider '" + std::string(provider) +
                    "' is registered more than once for '" + family + "/" +
                    backend + "'";
                return 1;
            }
        }
        ExtensionBackendRegistration registration;
        registration.family = family;
        registration.backend = backend;
        registration.provider = provider;
        std::memcpy(
            &registration.traits, traits,
            std::min(traits->struct_size, sizeof(registration.traits)));
        if (traits->forced_eos_ids_size != 0) {
            registration.forced_eos_ids.assign(
                traits->forced_eos_ids,
                traits->forced_eos_ids + traits->forced_eos_ids_size);
        }
        registration.traits.forced_eos_ids =
            registration.forced_eos_ids.data();
        registration.create = create;
        registration.factory_context = factory_context;
        context.pending_backends.push_back(std::move(registration));
        return 0;
    } catch (const std::exception& error) {
        context.error = error.what();
        return 1;
    } catch (...) {
        context.error = "extension backend registration failed";
        return 1;
    }
}

int RegisterFrontendProvider(
    void* opaque, const char* family, const char* provider,
    flm_frontend_create_v1 create, void* factory_context) noexcept {
    auto& context = *static_cast<RegistrationContext*>(opaque);
    try {
        if (!family || !*family || !provider || !*provider) {
            context.error =
                "extension frontend family and provider are required";
            return 1;
        }
        if (std::string_view(provider) ==
            flm::backend::kBuiltinProviderId) {
            context.error =
                "extension frontend provider id '" +
                std::string(flm::backend::kBuiltinProviderId) +
                "' is reserved";
            return 1;
        }
        if (!create) {
            context.error = "extension frontend factory is null";
            return 1;
        }
        for (const auto& entry : context.pending_frontends) {
            if (entry.family == family) {
                context.error =
                    "extension frontend family '" + std::string(family) +
                    "' is registered more than once";
                return 1;
            }
        }
        context.pending_frontends.push_back(
            ExtensionFrontendRegistration{
                family, provider, create, factory_context});
        return 0;
    } catch (const std::exception& error) {
        context.error = error.what();
        return 1;
    } catch (...) {
        context.error = "extension frontend registration failed";
        return 1;
    }
}

}  // namespace

struct FrontendRegistry::LoadedExtension {
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    std::string name;
    std::string version;
    std::string root;
    std::string asset_root;
    std::optional<catalog::ModelCatalogOverlay> catalog;
    std::vector<ExtensionBackendKey> backend_keys;
    std::vector<std::string> frontend_families;
    void* extension_context = nullptr;
    void (*destroy_extension_context)(void*) = nullptr;

    ~LoadedExtension() {
        if (destroy_extension_context && extension_context) {
            try {
                destroy_extension_context(extension_context);
            } catch (...) {
                // Extension callbacks must not throw across the C boundary.
            }
        }
#if defined(_WIN32)
        if (handle) FreeLibrary(handle);
#else
        if (handle) dlclose(handle);
#endif
    }
};

void FrontendDeleter::operator()(AutoModel* frontend) noexcept {
    DestroyFrontend(destroy, context, static_cast<void*>(frontend));
    provider_lifetime.reset();
}

FrontendRegistry::FrontendRegistry() {
    // Keep the backend registry alive until after this registry has removed
    // every extension provider, including for non-singleton registry objects.
    (void)backend::BackendRegistry::instance();
}

FrontendRegistry::~FrontendRegistry() {
    unload_extensions();
    std::lock_guard lock(mutex_);
    factories_.clear();
    providers_.clear();
}

FrontendRegistry& FrontendRegistry::instance() {
    static FrontendRegistry registry;
    static std::once_flag once;
    // register_builtin_frontends must not call instance(), or this deadlocks.
    std::call_once(once, [] { register_builtin_frontends(registry); });
    return registry;
}

void FrontendRegistry::register_frontend(std::string family,
                                         FrontendFactory factory,
                                         std::string provider) {
    if (family.empty()) throw std::runtime_error("frontend family is empty");
    if (!factory) throw std::runtime_error("frontend factory is null");
    if (provider.empty()) throw std::runtime_error("frontend provider is empty");

    std::lock_guard lock(mutex_);
    if (factories_.count(family)) {
        throw std::runtime_error(
            "frontend family '" + family + "' is already registered");
    }
    const std::string key = family;
    providers_.emplace(key, std::move(provider));
    try {
        factories_.emplace(std::move(family), std::move(factory));
    } catch (...) {
        providers_.erase(key);
        throw;
    }
}

void FrontendRegistry::unload_extensions() noexcept {
    std::lock_guard lock(mutex_);
    auto& backends = backend::BackendRegistry::instance();
    for (auto extension = extensions_.rbegin();
         extension != extensions_.rend(); ++extension) {
        for (auto key = (*extension)->backend_keys.rbegin();
             key != (*extension)->backend_keys.rend(); ++key) {
            backends.unregister_provider(
                key->family, key->backend, key->provider);
        }
        (*extension)->backend_keys.clear();
        for (const auto& family : (*extension)->frontend_families) {
            factories_.erase(family);
            providers_.erase(family);
        }
    }
    extensions_.clear();
}

bool FrontendRegistry::has(const std::string& family) const {
    std::lock_guard lock(mutex_);
    return factories_.count(family) != 0;
}

std::vector<std::string> FrontendRegistry::available() const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> families;
    families.reserve(factories_.size());
    for (const auto& [family, factory] : factories_) {
        (void)factory;
        families.push_back(family);
    }
    return families;
}

std::vector<catalog::ModelCatalogOverlay>
FrontendRegistry::catalog_overlays() const {
    std::lock_guard lock(mutex_);
    std::vector<catalog::ModelCatalogOverlay> catalogs;
    for (const auto& extension : extensions_) {
        if (extension->catalog) {
            catalogs.push_back(*extension->catalog);
        }
    }
    return catalogs;
}

FrontendPtr FrontendRegistry::create(const std::string& family,
                                     void* device) const {
    FrontendFactory factory;
    {
        std::lock_guard lock(mutex_);
        const auto found = factories_.find(family);
        if (found == factories_.end()) {
            std::vector<std::string> families;
            families.reserve(factories_.size());
            for (const auto& [registered, entry] : factories_) {
                (void)entry;
                families.push_back(registered);
            }
            throw std::runtime_error(
                "Model family '" + family +
                "' has no frontend. This build provides: " +
                Join(families));
        }
        factory = found->second;
    }
    auto frontend = factory(device);
    if (!frontend) {
        throw std::runtime_error(
            "Frontend factory for model family '" + family +
            "' returned null");
    }
    return frontend;
}

void FrontendRegistry::load_extension(
    const std::string& manifest_path,
    const std::unordered_set<std::string>& reserved_model_tags) {
    const std::filesystem::path native_manifest =
        NativePath(manifest_path);
    std::ifstream stream(native_manifest);
    if (!stream) {
        throw std::runtime_error(
            "Cannot open model extension manifest '" + manifest_path + "'");
    }
    nlohmann::json manifest;
    try {
        manifest = nlohmann::json::parse(stream);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "Cannot parse model extension manifest '" + manifest_path +
            "': " + error.what());
    }
    if (!manifest.is_object() ||
        manifest.value("schema_version", 0) != 1) {
        throw std::runtime_error(
            "Model extension manifest '" + manifest_path +
            "' must use schema_version 1");
    }
    for (const auto& [field, value] : manifest.items()) {
        (void)value;
        if (field != "schema_version" && field != "name" &&
            field != "version" && field != "library" &&
            field != "asset_root" && field != "catalog") {
            throw std::runtime_error(
                "Model extension manifest '" + manifest_path +
                "' has unsupported field '" + field + "'");
        }
    }
    for (const char* field : {"name", "version"}) {
        const auto value = manifest.find(field);
        if (value == manifest.end() || !value->is_string() ||
            value->get_ref<const std::string&>().empty()) {
            throw std::runtime_error(
                "Model extension manifest '" + manifest_path +
                "' field '" + field + "' must be a non-empty string");
        }
    }
    const auto library = manifest.find("library");
    if (library == manifest.end()) {
        throw std::runtime_error(
            "Model extension manifest '" + manifest_path +
            "' is missing 'library'");
    }
    std::string relative_library;
    if (library->is_string()) {
        relative_library = library->get<std::string>();
    } else if (library->is_object()) {
#if defined(_WIN32)
        constexpr const char* kPlatform = "windows-x86_64";
#else
        constexpr const char* kPlatform = "linux-x86_64";
#endif
        for (const auto& [platform, value] : library->items()) {
            if ((platform != "linux-x86_64" &&
                 platform != "windows-x86_64") ||
                !value.is_string() ||
                value.get_ref<const std::string&>().empty()) {
                throw std::runtime_error(
                    "Model extension manifest '" + manifest_path +
                    "' has invalid library entry '" + platform + "'");
            }
        }
        const auto selected = library->find(kPlatform);
        if (selected == library->end() || !selected->is_string()) {
            throw std::runtime_error(
                "Model extension manifest '" + manifest_path +
                "' has no library for " + kPlatform);
        }
        relative_library = selected->get<std::string>();
    } else {
        throw std::runtime_error(
            "Model extension manifest '" + manifest_path +
            "' field 'library' must be a path or platform map");
    }

    const auto root = native_manifest.has_parent_path()
        ? native_manifest.parent_path()
        : std::filesystem::current_path();
    nlohmann::json selected_paths = manifest;
    selected_paths["_selected_library"] = relative_library;
    const auto native_library = ResolveBundlePath(
        root, selected_paths, "_selected_library", false, true);
    const auto native_assets = manifest.contains("asset_root")
        ? ResolveBundlePath(root, manifest, "asset_root", true, true)
        : std::filesystem::weakly_canonical(root);
    const auto native_catalog = ResolveBundlePath(
        root, manifest, "catalog", false, false);

    auto extension = std::make_shared<LoadedExtension>();
    extension->root = Utf8Path(std::filesystem::weakly_canonical(root));
    extension->asset_root = Utf8Path(native_assets);
    if (!native_catalog.empty()) {
        extension->catalog = catalog::load_model_catalog_overlay(
            Utf8Path(native_catalog));
    }
    const std::string library_path = Utf8Path(native_library);

#if defined(_WIN32)
    extension->handle = LoadLibraryExW(
        native_library.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!extension->handle) {
        throw std::runtime_error(
            "Cannot load model extension '" + library_path +
            "' (Windows error " + std::to_string(GetLastError()) + ")");
    }
    const auto query = reinterpret_cast<flm_model_extension_query_v1_fn>(
        GetProcAddress(extension->handle,
                       "flm_model_extension_query_v1"));
#else
    extension->handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!extension->handle) {
        const char* error = dlerror();
        throw std::runtime_error(
            "Cannot load model extension '" + library_path + "': " +
            (error ? std::string(error) : std::string("unknown error")));
    }
    dlerror();
    const auto query = reinterpret_cast<flm_model_extension_query_v1_fn>(
        dlsym(extension->handle, "flm_model_extension_query_v1"));
#endif
    if (!query) {
        throw std::runtime_error(
            "Model extension '" + library_path +
            "' does not export flm_model_extension_query_v1");
    }

    const flm_model_extension_v1* descriptor = nullptr;
    try {
        descriptor = query();
    } catch (...) {
        throw std::runtime_error(
            "Model extension '" + library_path +
            "' allowed an exception to cross its C query boundary");
    }
    if (!descriptor ||
        descriptor->struct_size < FLM_MODEL_EXTENSION_V1_MIN_SIZE) {
        throw std::runtime_error(
            "Model extension '" + library_path +
            "' returned an incomplete descriptor");
    }
    if (descriptor->abi_version != FLM_MODEL_EXTENSION_ABI_V1) {
        throw std::runtime_error(
            "Model extension '" + library_path +
            "' uses unsupported ABI " +
            std::to_string(descriptor->abi_version));
    }
    if (descriptor->npu_runtime !=
        flm::extension::detail::host_npu_runtime()) {
        throw std::runtime_error(
            "Model extension '" + library_path +
            "' targets NPU runtime " +
            std::to_string(descriptor->npu_runtime) +
            ", but this FastFlowLM build uses " +
            std::to_string(flm::extension::detail::host_npu_runtime()));
    }
    if (!descriptor->name || !*descriptor->name ||
        !descriptor->version || !*descriptor->version ||
        !descriptor->register_extension) {
        throw std::runtime_error(
            "Model extension '" + library_path +
            "' has invalid metadata");
    }
    if (manifest["name"].get_ref<const std::string&>() != descriptor->name ||
        manifest["version"].get_ref<const std::string&>() !=
            descriptor->version) {
        throw std::runtime_error(
            "Model extension manifest identity does not match provider '" +
            std::string(descriptor->name) + "' version '" +
            descriptor->version + "'");
    }

    RegistrationContext registration;
    const flm_model_extension_registrar_v1 registrar{
        sizeof(flm_model_extension_registrar_v1),
        &registration,
        SetExtensionContext,
        RegisterBackendProvider,
        RegisterFrontendProvider,
    };
    RegistrationErrorCapture error;
    auto error_sink = error.sink();
    int status = 0;
    try {
        status = descriptor->register_extension(&registrar, &error_sink);
    } catch (...) {
        throw std::runtime_error(
            "Model extension '" + std::string(descriptor->name) +
            "' allowed an exception to cross its C boundary");
    }
    if (status != 0 || !registration.error.empty()) {
        throw std::runtime_error(
            "Model extension '" + std::string(descriptor->name) +
            "' failed to register: " +
            (!registration.error.empty()
                 ? registration.error
                 : !error.message.empty()
                     ? error.message
                     : "provider returned " + std::to_string(status)));
    }
    if (registration.pending_backends.empty() &&
        registration.pending_frontends.empty()) {
        throw std::runtime_error(
            "Model extension '" + std::string(descriptor->name) +
            "' registered no frontend or backend providers");
    }

    extension->name = descriptor->name;
    extension->version = descriptor->version;
    extension->extension_context = registration.extension_context;
    extension->destroy_extension_context =
        registration.destroy_extension_context;
    registration.extension_context = nullptr;
    registration.destroy_extension_context = nullptr;

    std::lock_guard lock(mutex_);
    if (extension->catalog) {
        std::vector<std::string> frontends;
        frontends.reserve(factories_.size());
        for (const auto& [family, factory] : factories_) {
            (void)factory;
            frontends.push_back(family);
        }
        for (const auto& pending : registration.pending_frontends) {
            if (std::find(
                    frontends.begin(), frontends.end(),
                    pending.family) != frontends.end()) {
                throw std::runtime_error(
                    "frontend family '" + pending.family +
                    "' is already registered");
            }
            frontends.push_back(pending.family);
        }
        for (const auto& [model_type, sizes] :
             extension->catalog->document.at("models").items()) {
            for (const auto& [size, entry] : sizes.items()) {
                const std::string tag = model_type + ":" + size;
                const std::string family =
                    entry.at("details").at("family").get<std::string>();
                auto selected = providers_.find(family);
                std::string expected =
                    selected == providers_.end() ? std::string() : selected->second;
                bool requires_explicit = false;
                for (const auto& pending :
                     registration.pending_frontends) {
                    if (pending.family == family) {
                        expected = pending.provider;
                        requires_explicit = true;
                        break;
                    }
                }
                if (!expected.empty()) {
                    const auto provider = entry.find("frontend_provider");
                    if ((provider == entry.end() && requires_explicit) ||
                        (provider != entry.end() &&
                         (!provider->is_string() ||
                          provider->get<std::string>() != expected))) {
                        throw std::runtime_error(
                            "Model extension catalog entry '" + tag +
                            "' must select frontend provider '" + expected +
                            "' for family '" + family + "'");
                    }
                }
                const bool has_private_payload =
                    entry.value("vlm", false) ||
                    entry.value("asr", false);
                const std::string backend_provider =
                    entry.at(backend::kBackendProviderKey)
                        .get<std::string>();
                const auto frontend_provider =
                    entry.find("frontend_provider");
                if (has_private_payload &&
                    backend_provider != backend::kBuiltinProviderId &&
                    (frontend_provider == entry.end() ||
                     !frontend_provider->is_string() ||
                     frontend_provider->get<std::string>() !=
                         backend_provider)) {
                    throw std::runtime_error(
                        "Multimodal model extension catalog entry '" + tag +
                        "' must select matching frontend and backend provider '" +
                        backend_provider + "'");
                }
            }
        }

        auto providers =
            backend::BackendRegistry::instance().provider_inventory();
        for (const auto& pending : registration.pending_backends) {
            auto& registered =
                providers[pending.family][pending.backend];
            if (std::find(
                    registered.begin(), registered.end(),
                    pending.provider) == registered.end()) {
                registered.push_back(pending.provider);
            }
        }
        auto claimed_tags = reserved_model_tags;
        for (const auto& loaded : extensions_) {
            if (!loaded->catalog) continue;
            const auto tags = catalog::model_catalog_tags(
                loaded->catalog->document, loaded->catalog->path);
            claimed_tags.insert(tags.begin(), tags.end());
        }
        catalog::validate_model_catalog_overlay(
            *extension->catalog, frontends, providers, claimed_tags);
    }
    try {
        extensions_.reserve(extensions_.size() + 1);
        if (!registration.pending_backends.empty()) {
            extension->backend_keys = register_extension_backends(
                std::move(registration.pending_backends),
                extension->root, extension->asset_root,
                std::static_pointer_cast<void>(extension));
        }
        extension->frontend_families.reserve(
            registration.pending_frontends.size());
        for (auto& pending : registration.pending_frontends) {
            const std::string family = pending.family;
            const std::string frontend_provider = pending.provider;
            if (factories_.count(family)) {
                throw std::runtime_error(
                    "frontend family '" + family +
                    "' is already registered");
            }
            // Record the key first so every exception from factory construction
            // or map insertion is covered by the rollback below.
            extension->frontend_families.push_back(family);
            factories_.emplace(
                family, make_extension_frontend_factory(
                    std::move(pending), extension->root,
                    extension->asset_root,
                    std::static_pointer_cast<void>(extension)));
            providers_.emplace(family, frontend_provider);
        }
        extensions_.push_back(std::move(extension));
    } catch (...) {
        for (const auto& family : extension->frontend_families) {
            factories_.erase(family);
            providers_.erase(family);
        }
        extension->frontend_families.clear();
        for (auto key = extension->backend_keys.rbegin();
             key != extension->backend_keys.rend(); ++key) {
            backend::BackendRegistry::instance().unregister_provider(
                key->family, key->backend, key->provider);
        }
        extension->backend_keys.clear();
        throw;
    }
}

}  // namespace flm::frontend
