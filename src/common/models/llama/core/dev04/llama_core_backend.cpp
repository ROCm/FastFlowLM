/// \file llama_core_backend.cpp
/// \brief Llama 3.2 1B and 3B, and Llama 3.1 8B, on corelib's "llama" model
#include "models/llama/core/dev04/llama_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::llama {

/// \note corelib steps a pass that starts off a page boundary one row at a
///       time up to it, so a later prompt re-runs its own page's earlier tokens
///       with it and runs as whole pages.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"llama", "Llama", kCoreContextLimit, core::Continuation::kFromPage,
                  core::Rewind::kSeek},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        // <|eot_id|>, <|eom_id|>, <|end_of_text|>: eos_token_id in the
        // catalog's config.json
        .eos_ids = {128009, 128008, 128001},
    });
}

}  // namespace flm::llama
