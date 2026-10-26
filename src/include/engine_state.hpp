/// \file engine_state.hpp
/// \brief engine_state: a copy of everything a model keeps between two tokens, held outside the engine
/// \note A model that wants its context saved and brought back (prefix cache, multi-conversation serving) derives
///       one class from this with whatever it keeps: attention K/V rows, recurrent (linear-attention) state, the
///       context length, a position counter that differs from it. The engine only reads its live state into one
///       (causal_lm::read_state) and writes one back into itself (causal_lm::write_state). It never holds more
///       than the state of the context it is running: how many copies exist, which one is used and when one is
///       thrown away is the caller's business (PrefixSlotCache), never the engine's.
#pragma once
#include <cstddef>

class engine_state {
public:
    virtual ~engine_state() = default;

    /// \brief context length the state was read at; 0 until it holds something
    int context_length = 0;
    /// \brief first position whose attention K/V rows the state holds (0 = all of them)
    /// \note Rows before it are the caller's to have in place when the state is written back (write_state_kv of
    ///       a state that holds them).
    int kv_from = 0;

    /// \brief whether read_state() filled it
    virtual bool valid() const = 0;

    /// \brief host bytes the state holds now
    virtual size_t bytes() const = 0;

    /// \brief host bytes a state read at `length` with K/V rows from `from` on would hold
    /// \note Works on a state that was never filled, so a caller can budget before it pays for the copy.
    virtual size_t estimate_bytes(int length, int from) const = 0;
};
