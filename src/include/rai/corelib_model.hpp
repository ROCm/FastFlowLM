/// \file corelib_model.hpp
/// \brief A causal_lm over a model corelib runs whole (ryzenai_corelib_model_*)
#pragma once

#include "causal_lm.hpp"
#include "rai/corelib_runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace flm::rai {

/// \brief where a multi-row pass may start when the model is past position 0
enum class Continuation {
    /// corelib takes rows > 1 at any position
    kAnywhere,
    /// only at position 0: a continuation re-runs every earlier token with it
    kFromStart,
    /// at 0 or, on a paged cache, on a page boundary: a continuation re-runs
    /// the earlier tokens of its own page with it
    kFromPage,
};

/// \brief how the model goes back to an earlier position
enum class Rewind {
    /// attention only: each pass states its position and overwrites what lies past it
    kSeek,
    /// recurrent state in model slots: model_state_save/restore at the
    /// checkpoint, reset and replay anywhere else
    kSnapshot,
    /// recurrent state outside any slot: reset and replay
    kReplay,
};

/// \brief what tells one corelib model apart from another
struct ModelSpec {
    /// \brief corelib's name for the model, e.g. "qwen3"
    std::string corelib_name;
    /// \brief how messages name the model, e.g. "Qwen3"
    std::string display_name;
    /// \brief the longest max_length this engine accepts
    std::uint32_t context_limit;
    Continuation continuation;
    Rewind rewind;
};

/// \brief rows in one page of a paged KV cache
inline constexpr std::uint32_t kPageRows = 4096;

/// \brief one corelib model, its packed weights, and the tokens it holds
/// \note Text only. corelib packs the weights from the file at construction,
///       which is the slow part of a load; the KV cache is sized once, at the
///       capacity corelib rounds `max_length` up to.
class CorelibModel final : public causal_lm {
public:
    /// \param weights the file corelib packs the weights from
    /// \param max_length positions this engine may run: 1..spec.context_limit
    CorelibModel(ModelSpec spec, std::filesystem::path weights,
                 std::shared_ptr<corelib::CorelibRuntime> runtime,
                 std::uint32_t max_length);
    ~CorelibModel() override;

    buffer<bf16> forward(int id) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int length) override;
    void load_weights(Q4NX&) override;
    void update_max_length(std::uint32_t max_length) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer, int index) override;
    buffer<bf16> get_v_cache(int layer, int index) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    /// \brief whether a corelib call failed mid-sequence
    /// \note Nothing clears it; every other entry point throws while it is
    ///       set, and the only way out is unload and reload.
    bool poisoned() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flm::rai
