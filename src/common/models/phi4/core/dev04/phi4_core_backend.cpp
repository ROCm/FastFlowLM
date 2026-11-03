/// \file phi4_core_backend.cpp
/// \brief Phi-4-mini on corelib's "phi4mini" model
#include "models/phi4/core/dev04/phi4_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::phi4 {

/// \note corelib reads the catalog's GGUF, Q8_0 or Q4_G128, and checks its
///       shapes and metadata itself. It steps a pass that starts off a page
///       boundary one row at a time up to it, so a later prompt re-runs its own
///       page's earlier tokens with it and runs as whole pages.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"phi4mini", "Phi-4", kCoreContextLimit, core::Continuation::kFromPage,
                  core::Rewind::kSeek},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        // <|end|>, <|endoftext|>
        .eos_ids = {200020, 199999},
    });
}

}  // namespace flm::phi4
