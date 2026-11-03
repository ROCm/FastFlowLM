/// \file qwen36_core_backend.hpp
/// \brief The ryzenai-corelib backend for Qwen3.6-35B-A3B text
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::qwen36 {

/// \note The model's own context, the most corelib pages it to.
inline constexpr std::uint32_t kCoreContextLimit = 262144;

/// \brief the longest context it answers correctly at
/// \note Past it the NPU's paged attention loses what lies far back
///       (PAGED_DECODE_PAGE_DRIFT in corelib's docs/known-problems.md), so the
///       catalog defaults to it and a longer context loads with a warning.
inline constexpr std::uint32_t kReliableContext = 32768;

inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCoreContextLimit;
    return traits;
}

flm::backend::BackendFactory core_factory();

}  // namespace flm::qwen36
