/// \file hybrid_state.hpp
/// \brief hybrid_engine_state + hybrid_state_api<Engine>: the saved context of a model that mixes recurrent and
///        full-attention layers, and the flm_state_api table that hands it to a caller as an opaque flm_state
/// \note The Qwen3.5 / Qwen3.6 family keeps, per linear-attention layer, one fixed-size recurrent state, and per
///       full-attention layer one K row and one V row per position, plus a position counter (`general_token_idx`)
///       that runs on its own once pictures are in the context. The engine fills and replays it; the caller never
///       sees this type, only the flm_state handle (flm_state_api.h). Nothing here knows the live cache layout.
#pragma once
#include <memory>
#include <vector>

#include "flm_state_api.h"
#include "causal_lm.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

class hybrid_engine_state {
public:
    /// \param fixed_bytes host bytes of the recurrent state, summed over the linear layers
    /// \param row_bytes   host bytes of one saved K/V row, summed over the attention layers (K and V)
    hybrid_engine_state(size_t fixed_bytes, size_t row_bytes) : fixed_bytes_(fixed_bytes), row_bytes_(row_bytes) {}

    /// \brief set by the engine once it has filled the buffers below
    bool filled = false;
    /// \brief context length the state was read at
    int context_length = 0;
    /// \brief first position whose attention K/V rows the state holds (0 = all of them)
    int kv_from = 0;
    /// \brief position counter of the rope (differs from context_length once images are in the context)
    int general_token_idx = 0;
    std::vector<buffer<bf16>> state; ///< indexed by layer; empty for attention layers
    std::vector<buffer<bf16>> kv_k;  ///< indexed by layer, (context_length - kv_from) rows of DK; empty for linear layers
    std::vector<buffer<bf16>> kv_v;  ///< same for V (rows of DV)

    /// \brief host bytes a state read at `length` with K/V rows from `from` on would hold
    size_t estimate_bytes(int length, int from) const {
        const size_t rows = length > from ? static_cast<size_t>(length - from) : 0;
        return fixed_bytes_ + rows * row_bytes_;
    }

private:
    size_t fixed_bytes_;
    size_t row_bytes_;
};

/// \brief the flm_state_api table of an engine class `Engine` (derived from causal_lm) that has
///        new_state() -> unique_ptr<hybrid_engine_state>, read_state(state&, kv_from), write_state(const state&)
///        and write_state_kv(const state&)
/// \param engine the causal_lm* the caller holds; nullptr unless it is an `Engine` (a lib answers only for its own)
template <class Engine>
const flm_state_api* hybrid_state_api(void* engine) {
    static const flm_state_api api = {
        FLM_STATE_API_VERSION, sizeof(flm_state_api),
        [](void* e) -> flm_state* {
            auto* eng = dynamic_cast<Engine*>(static_cast<causal_lm*>(e));
            return eng ? reinterpret_cast<flm_state*>(eng->new_state().release()) : nullptr;
        },
        [](flm_state* s) { delete reinterpret_cast<hybrid_engine_state*>(s); },
        [](void* e, flm_state* s, int kv_from) -> int {
            auto* eng = dynamic_cast<Engine*>(static_cast<causal_lm*>(e));
            return eng && s ? eng->read_state(*reinterpret_cast<hybrid_engine_state*>(s), kv_from) : -1;
        },
        [](void* e, const flm_state* s) -> int {
            auto* eng = dynamic_cast<Engine*>(static_cast<causal_lm*>(e));
            return eng && s ? eng->write_state(*reinterpret_cast<const hybrid_engine_state*>(s)) : -1;
        },
        [](void* e, const flm_state* s) -> int {
            auto* eng = dynamic_cast<Engine*>(static_cast<causal_lm*>(e));
            return eng && s ? eng->write_state_kv(*reinterpret_cast<const hybrid_engine_state*>(s)) : -1;
        },
        [](const flm_state* s) -> int { return reinterpret_cast<const hybrid_engine_state*>(s)->context_length; },
        [](const flm_state* s) -> int { return reinterpret_cast<const hybrid_engine_state*>(s)->kv_from; },
        [](const flm_state* s, int length, int from) -> size_t {
            return reinterpret_cast<const hybrid_engine_state*>(s)->estimate_bytes(length, from);
        },
    };
    return dynamic_cast<Engine*>(static_cast<causal_lm*>(engine)) ? &api : nullptr;
}
