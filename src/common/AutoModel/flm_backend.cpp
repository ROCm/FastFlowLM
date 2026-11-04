/// \file flm_backend.cpp
/// \brief The aie_next engine libraries an aie2p build loads on dev04 silicon
#include "AutoModel/flm_backend.hpp"

#include "utils/npu_platform.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace flm::backend {
namespace {

/// \brief the entry point every aie_next engine library exports
/// \note The engine on the frontend's device, and the library's saved-state
///       table: the frontend's own `<engine>_state_api` is the linked aie2p
///       library's, which must never be handed this engine.
using CreateFn = causal_lm* (*)(const LM_Config* config, flm_rt::device* device,
                                int max_length, const flm_state_api** state_api);

}  // namespace

bool runs_next_engine(const char* library) {
#if defined(FLM_USE_HRX)
    // the aie_next engines take an XRT device
    (void)library;
    return false;
#else
    return library != nullptr &&
           utils::get_device() == utils::npu_platform::dev04;
#endif
}

NextEngine create_next_engine(const char* library, const LM_Config& config,
                              flm_rt::device* device, int max_length) {
    // Never unloaded: the engines it builds may outlive any one model.
#ifdef _WIN32
    const std::string file = std::string(library) + ".dll";
    HMODULE module = LoadLibraryA(file.c_str());
    if (!module) {
        throw std::runtime_error("failed to load " + file + " (Windows error " +
                                 std::to_string(GetLastError()) + ")");
    }
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(module, "flm_engine_create"));
#else
    const std::string file = "lib" + std::string(library) + ".so";
    void* module = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        const char* reason = dlerror();
        throw std::runtime_error("failed to load " + file + " (" +
                                 (reason ? reason : "unknown dlopen failure") + ")");
    }
    auto create = reinterpret_cast<CreateFn>(dlsym(module, "flm_engine_create"));
#endif
    if (!create) {
        throw std::runtime_error(file + " has no flm_engine_create");
    }
    NextEngine built;
    built.engine = create(&config, device, max_length, &built.state_api);
#ifndef _WIN32
    // The aie2p library of the same class is loaded already, and ELF lets its
    // vtable stand in for this library's unless it was linked -Bsymbolic. Such
    // an engine cannot even be destroyed safely, so it is left to leak.
    Dl_info own{}, vtable{};
    if (!dladdr(reinterpret_cast<void*>(create), &own) ||
        !dladdr(*reinterpret_cast<void* const*>(built.engine), &vtable) ||
        own.dli_fbase != vtable.dli_fbase) {
        throw std::runtime_error(file + " is bound to another library's engine class (link it -Bsymbolic)");
    }
#endif
    return built;
}

}  // namespace flm::backend
