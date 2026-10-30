/// \file phi4_core_backend.hpp
/// \brief The ryzenai-corelib backend for Phi-4-mini
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::phi4 {

/// \brief the largest context this backend can hold
/// \note The most corelib pages Phi-4-mini to; past one page it ropes every
///       position with LongRoPE's long factors.
inline constexpr std::uint32_t kCoreContextLimit = 131072;

/// \brief the largest number of tokens corelib will decode
inline constexpr std::uint32_t kCoreDecodeLimit = kCoreContextLimit - 1;

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

}  // namespace flm::phi4
