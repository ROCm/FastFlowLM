/// \file gemma4_rai_backend.hpp
/// \brief The ryzenai-corelib backend for Gemma 4 (E2B and E4B)
/// \note Compiled only when FLM_ENABLE_RAI is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::gemma4 {

/// \brief rows in one page of the paged KV cache, and the longest unpaged context
inline constexpr std::uint32_t kPageSize = 4096;

/// \brief the longest context validated on hardware
/// \note corelib pages Gemma 4 up to the model's own 131072 and refuses past a
///       GGUF's smaller gemma4.context_length; 16384 is what was validated.
inline constexpr std::uint32_t kRaiContextLimit = 16384;

/// \brief one below the longest context: the last position has to be readable
inline constexpr std::uint32_t kRaiDecodeLimit = kRaiContextLimit - 1;

/// \return true when `length` is 1..4096, or a multiple of kPageSize up to kRaiContextLimit
inline bool ContextFits(std::uint32_t length) {
    if (length < 1 || length > kRaiContextLimit) return false;
    return length <= kPageSize || length % kPageSize == 0;
}

/// \note Inline: AutoModel reads it while assembling the BackendContext,
///       before any engine exists.
inline flm::backend::BackendTraits rai_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kRaiContextLimit;
    return traits;
}

flm::backend::BackendFactory rai_factory();

}  // namespace flm::gemma4
