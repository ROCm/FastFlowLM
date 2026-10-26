/// \file prefix_causal_lm.hpp
/// \brief hands a causal_lm to the PrefixSlotCache
#pragma once
#include "causal_lm.hpp"
#include "prefix_slot_cache.hpp"

/// \brief forwards the cache's four calls to the engine's state methods; owns nothing
/// \note Only for an engine built with those methods (the qwen3_5vl / qwen3_6_moe libs today): an older .so has a
///       shorter vtable and calling them would read past its end. The model wrapper decides which engine it hands over.
class prefix_causal_lm final : public prefix_state_io {
public:
    explicit prefix_causal_lm(causal_lm* engine) : engine_(engine) {}
    std::unique_ptr<engine_state> new_state() override { return engine_->new_state(); }
    int read_state(engine_state& state, int kv_from) override { return engine_->read_state(state, kv_from); }
    int write_state(const engine_state& state) override { return engine_->write_state(state); }
    int write_state_kv(const engine_state& state) override { return engine_->write_state_kv(state); }

private:
    causal_lm* engine_;
};
