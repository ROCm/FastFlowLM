/// \file corelib_model_backend.hpp
/// \brief The core backend for a model corelib runs whole, one spec per family
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"
#include "core/corelib_model.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace flm::core {

struct BackendSpec {
    ModelSpec model;
    /// \brief whether a requested context is one the model runs; empty means
    ///        1..model.context_limit
    std::function<bool(std::uint32_t)> context_fits;
    /// \brief the contexts allowed, as a refusal states them, e.g. "1..4096"
    std::string context_rule;
    /// \brief the longest context the model is known to answer correctly at,
    ///        0 for its whole range; a longer one loads with a warning
    std::uint32_t reliable_context = 0;
    /// \brief what goes wrong past reliable_context, for that warning
    std::string unreliable_reason;
    /// \brief the ids generation stops on; empty reads them from the package:
    ///        tokenizer_config.json's eos_token, then config.json's eos_token_id
    std::vector<int> eos_ids;
    /// \brief report tokenizer_config.json's bos_token as the BOS id, for a
    ///        package that names the token but not its id (Gemma 4)
    bool read_bos = false;
};

/// \brief a factory building the core backend `spec` describes
/// \note Preemption, the context and the package's token ids are checked
///       before the runtime is acquired or the weights are opened.
flm::backend::BackendFactory Factory(BackendSpec spec);

}  // namespace flm::core
