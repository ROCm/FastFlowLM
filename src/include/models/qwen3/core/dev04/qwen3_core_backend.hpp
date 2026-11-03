/// \file qwen3_core_backend.hpp
/// \brief The ryzenai-corelib backend for Qwen3 0.6B, 1.7B, 4B and 8B
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::qwen3 {

/// \brief the largest context this backend can hold
/// \note Qwen3's native context, the most corelib pages it to.
inline constexpr std::uint32_t kCoreContextLimit = 32768;

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

}  // namespace flm::qwen3
