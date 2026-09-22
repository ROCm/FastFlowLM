/// \file flm_plugin.hpp
/// \brief Loading user supplied operator overrides from a shared library.
/// \note  A plugin is a shared library exporting flm_plugin_abi_version() and
///        flm_plugin_register(). Point FLM_PLUGIN at it and flm loads it right
///        after the NPU device is ready and before any model engine is
///        constructed, which is the window in which a plugin may register
///        overrides and allocate its own weights.
///
/// A minimal plugin:
/// \code
///   #include "flm_plugin.hpp"
///
///   static void register_overrides(const flm::plugin_context& ctx) {
///       ctx.npu->hooks.override_op<my_op_func_t>("mlp.up_proj",
///           [ctx](bytes& out, bytes& in, bytes& weights, int layer, int m) {
///               return flm::hook_result<ert_cmd_state>(...);
///           });
///   }
///   FLM_PLUGIN(register_overrides)
/// \endcode
///
/// \note A plugin and flm exchange C++ objects, so both must be built with the
///       same compiler and standard library. The ABI version guards against
///       loading a plugin built for a different override interface; it cannot
///       detect a toolchain mismatch.
#pragma once

#include <string>
#include <vector>

#include "npu_utils/npu_utils.hpp"
#include "utils/debug_utils.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace flm {

inline constexpr int plugin_abi_version = 1;

/// \brief What a plugin is given when it registers.
struct plugin_context {
    npu_xclbin_manager* npu;   ///< the one device manager; also holds the hook registry (npu->hooks)
    const char* model_path;    ///< directory holding the model's weights
    const char* xclbin_path;   ///< directory holding this model's xclbins, where a plugin's belong too
};

using plugin_abi_version_fn = int (*)();
using plugin_register_fn = void (*)(const plugin_context*);

#ifdef _WIN32
inline constexpr char plugin_path_separator = ';';
#else
inline constexpr char plugin_path_separator = ':';
#endif

// Load one plugin and let it register its overrides.
/// \throws std::runtime_error if the library cannot be loaded, lacks the entry
///         points, or was built against a different override interface.
inline void load_plugin(const plugin_context& ctx, const std::string& path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == nullptr) throw std::runtime_error("failed to load plugin: " + path);
    auto abi = reinterpret_cast<plugin_abi_version_fn>(GetProcAddress(handle, "flm_plugin_abi_version"));
    auto reg = reinterpret_cast<plugin_register_fn>(GetProcAddress(handle, "flm_plugin_register"));
#else
    // RTLD_GLOBAL so a plugin's typeinfo and the host's agree, which dynamic_cast
    // across the boundary depends on. The handle is deliberately never closed:
    // the overrides it registered outlive registration.
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) throw std::runtime_error("failed to load plugin " + path + ": " + dlerror());
    auto abi = reinterpret_cast<plugin_abi_version_fn>(dlsym(handle, "flm_plugin_abi_version"));
    auto reg = reinterpret_cast<plugin_register_fn>(dlsym(handle, "flm_plugin_register"));
#endif
    if (abi == nullptr || reg == nullptr) {
        throw std::runtime_error("plugin " + path + " does not export flm_plugin_abi_version/flm_plugin_register");
    }
    const int found = abi();
    if (found != plugin_abi_version) {
        throw std::runtime_error("plugin " + path + " targets override ABI " + std::to_string(found)
                                 + ", this build provides " + std::to_string(plugin_abi_version));
    }
    reg(&ctx);
}

// Load every plugin named in FLM_PLUGIN, separated by ':' (';' on Windows).
inline void load_plugins_from_env(const plugin_context& ctx) {
    const char* env = std::getenv("FLM_PLUGIN");
    if (env == nullptr || *env == '\0') return;
    const std::string spec(env);
    size_t begin = 0;
    while (begin <= spec.size()) {
        const size_t end = std::min(spec.find(plugin_path_separator, begin), spec.size());
        const std::string path = spec.substr(begin, end - begin);
        if (!path.empty()) {
            load_plugin(ctx, path);
            header_print_g("info", "Loaded operator plugin: " << path);
        }
        begin = end + 1;
    }
}

}  // namespace flm

// Defines a plugin's entry points around a registration function.
#define FLM_PLUGIN(register_function)                                            \
    extern "C" int flm_plugin_abi_version() { return flm::plugin_abi_version; }  \
    extern "C" void flm_plugin_register(const flm::plugin_context* ctx) {        \
        (register_function)(*ctx);                                               \
    }
