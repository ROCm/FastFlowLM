/// \file phi4_corelib_backend.hpp
/// \brief The ryzenai-corelib AIE4 backend for Phi-4
/// \note Compiled only when FLM_ENABLE_CORELIB_AIE4 is on. Everything that used
///       to be an aie4 special case inside the Phi4 frontend -- the 4095-token
///       decode limit, the no-preemption rule, the package-verified EOS ids, the
///       poisoning state -- is stated here instead.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::phi4 {

/// \brief the registered id of this backend
/// \note Matches details.execution_backend in model_list.json, so catalogs
///       written before the registry existed keep selecting it.
inline constexpr const char* kCorelibAie4BackendId = "corelib_aie4_gguf";

/// \brief the largest context this backend can hold
inline constexpr std::uint32_t kCorelibAie4ContextLimit = 4096;

/// \brief the largest number of tokens corelib will decode
inline constexpr std::uint32_t kCorelibAie4DecodeLimit = 4095;

/// \brief what the frontend must know before building this backend
/// \return no xclbin, no preemption, 4096-token context ceiling
/// \note Pure data, and inline on purpose: the registry needs it to reject a
///       request before the engine exists, and so does anything standing in for
///       the engine.
inline flm::backend::BackendTraits corelib_aie4_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCorelibAie4ContextLimit;
    return traits;
}

/// \brief a factory building the corelib AIE4 backend
/// \return a factory suitable for BackendRegistry::register_backend
flm::backend::BackendFactory corelib_aie4_factory();

}  // namespace flm::phi4
