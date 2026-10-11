/// \file qwen35_rai_backend.hpp
/// \brief The ryzenai-corelib backend for dense Qwen3.5
/// \note Compiled only when FLM_ENABLE_RAI is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::qwen35 {

inline constexpr std::uint32_t kRaiContextLimit = 4096;

inline flm::backend::BackendTraits rai_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kRaiContextLimit;
    return traits;
}

flm::backend::BackendFactory rai_factory();

}  // namespace flm::qwen35
