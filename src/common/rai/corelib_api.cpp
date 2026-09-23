#include "rai/corelib_api.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace flm::corelib {
namespace {
/// \brief the corelib version this build was compiled against
/// \note Taken from the vendor header's own macros rather than written out
///       again. corelib_api.hpp already refuses to compile against any other
///       version, so deriving the run-time check from the same macros is what
///       keeps the two from drifting -- they were separate constants once, and
///       bumping one without the other is a mismatch that only shows up on a
///       machine with the real DLL.
constexpr CorelibVersion kRequiredVersion{RYZENAI_CORELIB_VERSION_MAJOR,
                                          RYZENAI_CORELIB_VERSION_MINOR,
                                          RYZENAI_CORELIB_VERSION_PATCH};

std::string VersionText(CorelibVersion version) {
    return std::to_string(version.major) + "." + std::to_string(version.minor) +
           "." + std::to_string(version.patch);
}

std::string ErrorText(std::string_view call,
                      std::string_view status,
                      std::string_view detail) {
    std::string result(call);
    result += " failed: ";
    result += status;
    if (!detail.empty()) {
        result += ": ";
        result += detail;
    }
    return result;
}

/// \brief what a shared library is called on this platform
/// \note Only the spelling changes between platforms, so the extension and the
///       file name are named once here rather than at each of the three places
///       that used to write ".dll" inline. FLM_RAI_CORELIB_PATH is validated
///       against the same constants, so the diagnostic cannot describe a
///       different suffix from the one actually required.
constexpr std::string_view kSharedLibraryExtension =
#ifdef _WIN32
    ".dll";
#else
    ".so";
#endif

constexpr std::string_view kCorelibLibraryName =
#ifdef _WIN32
    "ryzenai_corelib.dll";
#else
    "libryzenai_corelib.so";
#endif

bool HasSharedLibraryExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension == kSharedLibraryExtension;
}
}  // namespace

CorelibError::CorelibError(ryzenai_corelib_status status,
                           std::string call,
                           std::string detail,
                           std::string status_text)
    : std::runtime_error(ErrorText(call, status_text, detail)),
      status_(status),
      call_(std::move(call)),
      detail_(std::move(detail)) {}

ryzenai_corelib_status CorelibError::status() const noexcept { return status_; }
const std::string& CorelibError::call() const noexcept { return call_; }
const std::string& CorelibError::detail() const noexcept { return detail_; }

CorelibApi::CorelibApi(Resolver resolver,
                       std::filesystem::path loaded_library_path)
    : resolver_(std::move(resolver)),
      loaded_library_path_(std::move(loaded_library_path)) {
    void* version_symbol = resolver_("ryzenai_corelib_get_version");
    if (!version_symbol) {
        throw std::runtime_error("missing corelib symbol: ryzenai_corelib_get_version");
    }
    functions_.get_version =
        reinterpret_cast<decltype(functions_.get_version)>(version_symbol);
    functions_.get_version(&runtime_version_.major, &runtime_version_.minor,
                           &runtime_version_.patch);
    if (runtime_version_.major != kRequiredVersion.major ||
        runtime_version_.minor != kRequiredVersion.minor ||
        runtime_version_.patch != kRequiredVersion.patch) {
        throw std::runtime_error("corelib ABI mismatch: runtime " +
                                 VersionText(runtime_version_) + ", required " +
                                 VersionText(kRequiredVersion));
    }

#define FLM_RESOLVE_CORELIB_FUNCTION(member, symbol)                               \
    if constexpr (std::string_view(#symbol) !=                                    \
                  std::string_view("ryzenai_corelib_get_version")) {              \
        void* address = resolver_(#symbol);                                        \
        if (!address) throw std::runtime_error("missing corelib symbol: " #symbol); \
        functions_.member = reinterpret_cast<decltype(functions_.member)>(address); \
    }
    FLM_CORELIB_FUNCTIONS(FLM_RESOLVE_CORELIB_FUNCTION)
#undef FLM_RESOLVE_CORELIB_FUNCTION
}

std::shared_ptr<CorelibApi> CorelibApi::ResolveForTest(
    Resolver resolver, std::filesystem::path loaded_library_path) {
    if (!resolver) throw std::invalid_argument("corelib resolver is empty");
    return std::shared_ptr<CorelibApi>(new CorelibApi(
        std::move(resolver), std::move(loaded_library_path)));
}

std::shared_ptr<CorelibApi> CorelibApi::Load(const std::filesystem::path& library) {
    const std::filesystem::path absolute_library = std::filesystem::absolute(library);
#ifdef _WIN32
    HMODULE raw_module = LoadLibraryExW(
        absolute_library.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!raw_module) {
        throw std::runtime_error("failed to load corelib library '" +
                                 absolute_library.string() + "' (Windows error " +
                                 std::to_string(GetLastError()) + ")");
    }
    auto module = std::shared_ptr<void>(raw_module, [](void* handle) {
        FreeLibrary(static_cast<HMODULE>(handle));
    });
    Resolver resolver = [module](std::string_view name) -> void* {
        const std::string terminated(name);
        return reinterpret_cast<void*>(
            GetProcAddress(static_cast<HMODULE>(module.get()), terminated.c_str()));
    };
#else
    // RTLD_LOCAL so corelib's symbols do not join the global namespace: this
    // process also links XRT directly, and a corelib built against a different
    // one must not be able to satisfy our XRT calls. RTLD_NOW because a missing
    // symbol has to surface here, next to the path that was loaded, rather than
    // at the first dispatch.
    void* raw_module = ::dlopen(absolute_library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!raw_module) {
        const char* reason = ::dlerror();
        throw std::runtime_error("failed to load corelib library '" +
                                 absolute_library.string() + "' (" +
                                 (reason ? reason : "unknown dlopen failure") + ")");
    }
    auto module = std::shared_ptr<void>(
        raw_module, [](void* handle) { ::dlclose(handle); });
    Resolver resolver = [module](std::string_view name) -> void* {
        const std::string terminated(name);
        return ::dlsym(module.get(), terminated.c_str());
    };
#endif
    return ResolveForTest(std::move(resolver), absolute_library);
}

#if defined(FLM_CORELIB_LINK_STATIC)
std::shared_ptr<CorelibApi> CorelibApi::LoadStatic() {
    const char* configured = std::getenv("FLM_RAI_CORELIB_PATH");
    if (configured && *configured) {
        std::cerr << "[FLM]  Ignoring FLM_RAI_CORELIB_PATH: corelib is linked "
                     "into this build, so there is no DLL to select."
                  << std::endl;
    }
    // Resolve through the same name-keyed resolver the DLL path uses, so the
    // version check and the missing-symbol diagnostics stay identical.
    Resolver resolver = [](std::string_view name) -> void* {
#define FLM_STATIC_CORELIB_SYMBOL(member, symbol)                                  \
        if (name == std::string_view(#symbol)) {                                   \
            return reinterpret_cast<void*>(&::symbol);                             \
        }
        FLM_CORELIB_FUNCTIONS(FLM_STATIC_CORELIB_SYMBOL)
#undef FLM_STATIC_CORELIB_SYMBOL
        return nullptr;
    };
    return ResolveForTest(std::move(resolver),
                          std::filesystem::path("<statically linked>"));
}
#endif

std::filesystem::path CorelibApi::ResolveLibraryPath(
    const std::filesystem::path& executable_dir) {
    const char* configured = std::getenv("FLM_RAI_CORELIB_PATH");
    if (configured && *configured) {
        const std::filesystem::path path(configured);
        if (!path.is_absolute()) {
            throw std::runtime_error(
                "FLM_RAI_CORELIB_PATH must be an absolute " +
                std::string(kSharedLibraryExtension) + " path");
        }
        if (!path.has_filename() || !HasSharedLibraryExtension(path)) {
            throw std::runtime_error(
                "FLM_RAI_CORELIB_PATH must name an absolute " +
                std::string(kSharedLibraryExtension) + " file");
        }
        return path;
    }
    return std::filesystem::absolute(executable_dir / "rai" /
                                     kCorelibLibraryName);
}

const CorelibFunctions& CorelibApi::functions() const noexcept { return functions_; }
CorelibVersion CorelibApi::runtime_version() const noexcept { return runtime_version_; }
const std::filesystem::path& CorelibApi::loaded_library_path() const noexcept {
    return loaded_library_path_;
}

void CorelibApi::Check(ryzenai_corelib_status status,
                       std::string_view call) const {
    if (status == ryzenai_corelib_status_success) return;
    const char* detail_pointer = functions_.get_last_error_message();
    const std::string detail = detail_pointer ? detail_pointer : "";
    const char* status_pointer = functions_.status_to_string(status);
    const std::string status_text = status_pointer ? status_pointer : "unknown";
    throw CorelibError(status, std::string(call), detail, status_text);
}

void CorelibApi::RegisterObject() const noexcept { ++live_object_count_; }

void CorelibApi::Release(void* object) const noexcept {
    if (!object) return;
    functions_.object_release(object);
    --live_object_count_;
}

std::size_t CorelibApi::live_object_count() const noexcept {
    return live_object_count_.load();
}

}  // namespace flm::corelib
