/// \file qwen36_rai_backend.cpp
/// \brief Qwen3.6-35B-A3B text on corelib's "qwen36" model
#include "models/qwen36/rai/aie_next/qwen36_rai_backend.hpp"

#include "rai/corelib_model_backend.hpp"

#include <string>

namespace flm::qwen36 {

flm::backend::BackendFactory rai_factory() {
    return rai::Factory({
        .model = {"qwen36", "Qwen3.6", kRaiContextLimit, rai::Continuation::kAnywhere,
                  rai::Rewind::kSnapshot},
        .context_rule = "1.." + std::to_string(kRaiContextLimit),
    });
}

}  // namespace flm::qwen36
