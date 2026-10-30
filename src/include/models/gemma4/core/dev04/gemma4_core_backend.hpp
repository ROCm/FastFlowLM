/// \file gemma4_core_backend.hpp
/// \brief The ryzenai-corelib backend for Gemma 4 (E2B and E4B)
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::gemma4 {

/// \note corelib pages Gemma 4 up to the model's own context, at most 131072.
inline constexpr std::uint32_t kCoreContextLimit = 131072;

/// \brief one below the longest context: the last position has to be readable
inline constexpr std::uint32_t kCoreDecodeLimit = kCoreContextLimit - 1;

/// \note Inline: AutoModel reads it while assembling the BackendContext,
///       before any engine exists.
inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCoreContextLimit;
    return traits;
}

flm::backend::BackendFactory core_factory();

}  // namespace flm::gemma4
