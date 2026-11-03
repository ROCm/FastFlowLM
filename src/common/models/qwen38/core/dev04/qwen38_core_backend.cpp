/// \file qwen38_core_backend.cpp
/// \brief Qwen3.8-27B text on corelib's "qwen38" model
#include "models/qwen38/core/dev04/qwen38_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::qwen38 {

/// \note corelib runs it on the NPU only from the converted Q4_G128 export;
///       any other GGUF it takes runs on the GPU or the host.
/// \note The linear-attention state can neither rewind nor skip, so going back
///       is a snapshot restore at the checkpoint and a replay anywhere else.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"qwen38", "Qwen3.8", kCoreContextLimit, core::Continuation::kAnywhere,
                  core::Rewind::kSnapshot},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        // <|im_end|>, <|endoftext|>
        .eos_ids = {248046, 248044},
    });
}

}  // namespace flm::qwen38
