/// \file oga_engine.hpp
/// \brief A causal_lm over an OGA model, driven one AppendTokens at a time
#pragma once

#include "causal_lm.hpp"
#include "oga/oga_api.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace flm::oga {

/// \brief one OGA model and the tokens one generator has been given
/// \note Text only. Exact token prefixes continue the live generator. A
///       divergent history rebuilds it and replays the logical token sequence.
///       Checkpoint/restore only saves that sequence; it never relies on OGA
///       rewinding a hybrid model's recurrent state.
class OgaEngine final : public causal_lm {
public:
    /// \param model a model already checked to run on the GPU; the engine owns it
    /// \param vocab_size the logits width the package states
    /// \param context_limit the most positions the package supports
    /// \param max_length positions this engine may run: 1..context_limit
    OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
              std::string display_name, std::uint32_t vocab_size,
              std::uint32_t context_limit, std::uint32_t max_length);
    ~OgaEngine() override;

    std::uint32_t max_length() const { return max_length_; }

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

private:
    /// \brief a generator and what it has run
    struct Live {
        OgaApi::Owned<OgaGenerator> generator;
        /// \brief max_length the generator was created with
        std::uint32_t capacity = 0;
        /// \brief every token the generator has been given
        std::vector<int> ran;
    };

    /// \brief how a pass reached the end of `logical`
    enum class Reuse { kSameTokens, kRebuilt };

    /// \brief run `fresh` after `logical`, and keep it there
    buffer<bf16> Run(const std::vector<int>& fresh);
    /// \brief continue or rebuild the generator, then give it the rest of `logical`
    /// \return how, and how many tokens of `logical` were already run
    std::pair<Reuse, std::size_t> Append();
    /// \brief make `live` stand for a prefix of `logical`, or rebuild it
    Reuse Resume();
    /// \brief replace the generator with an empty one, which replays all of `logical`
    Reuse Rebuild();
    OgaApi::Owned<OgaGenerator> NewGenerator(std::uint32_t max_length) const;
    void AppendTokens(OgaGenerator* generator, const int* ids, std::size_t count) const;
    /// \brief the last position's logits, as bf16
    buffer<bf16> Logits(OgaGenerator* generator) const;
    void CheckIds(const std::vector<int>& ids) const;

    // Declared first, so it is destroyed last: the API, whose entry points the
    // destructors of every OGA object below call, outlives them.
    std::shared_ptr<const OgaApi> api_;
    std::string display_name_;
    std::uint32_t vocab_size_;
    std::uint32_t context_limit_;
    std::uint32_t max_length_;
    std::vector<int> logical_;
    std::optional<std::vector<int>> saved_;
    // The generator is destroyed before the model it reads.
    OgaApi::Owned<OgaModel> model_;
    std::optional<Live> live_;
};

}  // namespace flm::oga
