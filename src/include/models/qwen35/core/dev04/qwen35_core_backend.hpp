/// \file qwen35_core_backend.hpp
/// \brief The ryzenai-corelib backend for dense Qwen3.5
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::qwen35 {

/// \note Qwen3.5's native context, the most corelib pages it to; corelib
///       rounds a context up to whole 4096-row pages.
inline constexpr std::uint32_t kCoreContextLimit = 262144;

inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCoreContextLimit;
    return traits;
}

flm::backend::BackendFactory core_factory();

}  // namespace flm::qwen35
