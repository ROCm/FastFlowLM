/// \file llama_core_backend.hpp
/// \brief The ryzenai-corelib backend for Llama 3.2 1B and 3B, and Llama 3.1 8B
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::llama {

/// \brief the largest context this backend can hold
/// \note The most corelib pages Llama to, and the context of every size here.
inline constexpr std::uint32_t kCoreContextLimit = 131072;

/// \brief what the frontend must know before building this backend
/// \note Inline so the registry can reject a request before the engine exists.
inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCoreContextLimit;
    return traits;
}

/// \brief a factory building the core backend
flm::backend::BackendFactory core_factory();

}  // namespace flm::llama
