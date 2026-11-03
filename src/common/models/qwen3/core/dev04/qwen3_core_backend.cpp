/// \file qwen3_core_backend.cpp
/// \brief Qwen3 0.6B, 1.7B, 4B and 8B on corelib's "qwen3" model
#include "models/qwen3/core/dev04/qwen3_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::qwen3 {

/// \note corelib steps a pass that starts off a page boundary one row at a
///       time up to it, so a later prompt re-runs its own page's earlier tokens
///       with it and runs as whole pages.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"qwen3", "Qwen3", kCoreContextLimit, core::Continuation::kFromPage,
                  core::Rewind::kSeek},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        // <|im_end|>, <|endoftext|>
        .eos_ids = {151645, 151643},
    });
}

}  // namespace flm::qwen3
