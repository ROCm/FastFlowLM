/// \file gemma4_core_backend.cpp
/// \brief Gemma 4 E2B and E4B on corelib's "gemma4" model
#include "models/gemma4/core/dev04/gemma4_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::gemma4 {

/// \note corelib steps a pass that starts off a page boundary one row at a
///       time up to it, so a later prompt re-runs its own page's earlier tokens
///       with it and runs as whole pages.
/// \note The package states its stop and start tokens by name: <eos> and <bos>
///       in tokenizer_config.json, and config.json's eos_token_id [1, 106]
///       adds <turn|>.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"gemma4", "Gemma 4", kCoreContextLimit, core::Continuation::kFromPage,
                  core::Rewind::kSeek},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        .read_bos = true,
    });
}

}  // namespace flm::gemma4
