/// \file qwen35_rai_backend.cpp
/// \brief Dense Qwen3.5 on corelib's "qwen35" model
#include "models/qwen35/rai/aie_next/qwen35_rai_backend.hpp"

#include "rai/corelib_model_backend.hpp"

namespace flm::qwen35 {

/// \note The linear-attention state can neither rewind nor skip and corelib
///       keeps it outside any state slot, so going back means replaying.
flm::backend::BackendFactory rai_factory() {
    return rai::Factory({
        .model = {"qwen35", "Qwen3.5", kRaiContextLimit, rai::Continuation::kAnywhere,
                  rai::Rewind::kReplay},
        .context_rule = "1..4096",
        .eos_ids = {248046, 248044},
    });
}

}  // namespace flm::qwen35
