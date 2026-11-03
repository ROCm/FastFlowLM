/// \file qwen36_core_backend.cpp
/// \brief Qwen3.6-35B-A3B text on corelib's "qwen36" model
#include "models/qwen36/core/dev04/qwen36_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::qwen36 {

flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"qwen36", "Qwen3.6", kCoreContextLimit, core::Continuation::kAnywhere,
                  core::Rewind::kSnapshot},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        .reliable_context = kReliableContext,
        .unreliable_reason = "past it the NPU's paged attention can return garbage "
                             "(corelib's PAGED_DECODE_PAGE_DRIFT)",
    });
}

}  // namespace flm::qwen36
