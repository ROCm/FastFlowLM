/// \file model_extension_detail.hpp
/// \brief Private helpers shared by stable-C model-extension adapters.
#pragma once

#include "AutoModel/model_extension.h"
#include "device_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace flm::extension::detail {

inline flm_span_v1 span(std::string_view value) {
    return {value.data(), value.size()};
}

inline std::string copy_span(flm_span_v1 value) {
    return value.data
               ? std::string(static_cast<const char*>(value.data), value.size)
               : std::string();
}

class ErrorCapture {
public:
    static void set(void* context, const char* message,
                    std::size_t size) noexcept {
        try {
            static_cast<ErrorCapture*>(context)->message.assign(
                message ? message : "", message ? size : 0);
        } catch (...) {
        }
    }

    flm_error_sink_v1 sink() noexcept { return {this, set}; }

    std::string message;
};

inline void destroy_provider_object(
    void (*destroy)(void*), void* object) noexcept {
    if (!destroy || !object) return;
    try {
        destroy(object);
    } catch (...) {
        // Provider callbacks may not throw across the extension C boundary.
    }
}

constexpr std::uint32_t host_npu_runtime() {
#if defined(FLM_USE_HRX)
    return FLM_MODEL_EXTENSION_NPU_RUNTIME_HRX;
#else
    return FLM_MODEL_EXTENSION_NPU_RUNTIME_XRT;
#endif
}

inline void* native_runtime_device_handle(flm_rt::device* device) {
    if (!device) return nullptr;
#if defined(FLM_USE_HRX)
    // hrx_cpp::device is a facade over HRX's process-wide C runtime.
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(1));
#else
    return reinterpret_cast<void*>(static_cast<xclDeviceHandle>(*device));
#endif
}

}  // namespace flm::extension::detail
