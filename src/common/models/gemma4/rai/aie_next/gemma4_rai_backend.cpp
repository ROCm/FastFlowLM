/// \file gemma4_rai_backend.cpp
/// \brief Gemma 4 E2B and E4B on corelib's "gemma4" model
#include "models/gemma4/rai/aie_next/gemma4_rai_backend.hpp"

#include "rai/corelib_model_backend.hpp"

namespace flm::gemma4 {

/// \note A multi-row pass starts at 0 or, past 4096, on a page boundary, so a
///       later prompt re-runs its own page's earlier tokens with it.
/// \note The package states its stop and start tokens by name: <eos> and <bos>
///       in tokenizer_config.json, and config.json's eos_token_id [1, 106]
///       adds <turn|>.
flm::backend::BackendFactory rai_factory() {
    return rai::Factory({
        .model = {"gemma4", "Gemma 4", kRaiContextLimit, rai::Continuation::kFromPage,
                  rai::Rewind::kSeek},
        .context_fits = ContextFits,
        .context_rule = "1..4096, or a multiple of 4096 up to 16384",
        .read_bos = true,
    });
}

}  // namespace flm::gemma4
