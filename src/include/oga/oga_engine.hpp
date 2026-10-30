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
/// \note Text only. `logical` is what the frontend sees; the live generator
///       holds what it ran. A pass continues the generator when what it ran
///       is a token prefix of `logical`. Otherwise, when OGA can restore a
///       snapshot, the generator goes back to the one taken at the last
///       checkpoint(), once, if `logical` still starts with what it held
///       there. A hybrid model's recurrent state cannot be stepped back any
///       other way, and onnxruntime-genai 0.17 accepts SnapshotState and
///       RewindTo without restoring anything, so a probe at construction
///       decides whether rewinding works at all. With Options::text_reuse
///       the generator also continues when what it ran decodes to a prefix
///       of `logical`'s text, as when generated text re-tokenizes into
///       another split. Any other pass rebuilds the generator and replays
///       every token from position 0. A snapshot or RewindTo that fails costs
///       that replay and never the request; a failed RewindTo also ends rewinding
///       for this engine.
class OgaEngine final : public causal_lm {
public:
    struct Options {
        /// \brief continue across another token split of the same text (experimental)
        bool text_reuse = false;
    };

    /// \param model a model already checked to run on the GPU; the engine owns it
    /// \param vocab_size the logits width the package states
    /// \param context_limit the most positions the package supports
    /// \param max_length positions this engine may run: 1..context_limit
    OgaEngine(std::shared_ptr<const OgaApi> api, OgaApi::Owned<OgaModel> model,
              std::string display_name, std::uint32_t vocab_size,
              std::uint32_t context_limit, std::uint32_t max_length, Options options);
    ~OgaEngine() override;

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
    enum class RewindSupport { kNone, kSnapshot };

    /// \brief a generator and what it has run
    struct Live {
        OgaApi::Owned<OgaGenerator> generator;
        /// \brief max_length the generator was created with
        std::uint32_t capacity = 0;
        /// \brief every token the generator has been given
        std::vector<int> ran;
        /// \brief the prefix of `logical` that `ran` equals, by tokens or by text
        std::vector<int> stands_for;
        /// \brief the length of `ran` the generator's snapshot holds
        std::optional<std::size_t> snapshot_at;
    };

    /// \brief how a pass reached the end of `logical`
    enum class Reuse { kSameTokens, kRewound, kSameText, kRebuilt };

    /// \brief run `fresh` after `logical`, and keep it there
    buffer<bf16> Run(const std::vector<int>& fresh);
    /// \brief continue or rebuild the generator, then give it the rest of `logical`
    /// \return how, and how many tokens of `logical` were already run
    std::pair<Reuse, std::size_t> Append();
    /// \brief make `live` stand for a prefix of `logical`, or rebuild it
    Reuse Resume();
    /// \brief replace the generator with an empty one, which replays all of `logical`
    Reuse Rebuild();
    /// \return how many leading tokens of `logical` decode to the text `live` ran
    std::optional<std::size_t> SameTextPrefix() const;
    std::string Decode(const int* ids, std::size_t count) const;
    OgaApi::Owned<OgaGenerator> NewGenerator(std::uint32_t max_length) const;
    void AppendTokens(OgaGenerator* generator, const int* ids, std::size_t count) const;
    /// \brief whether a snapshot this OGA takes is what RewindTo restores
    RewindSupport ProbeRewind() const;
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
    // The tokenizer and the generator read the model, so they are declared
    // after it and go first.
    OgaApi::Owned<OgaModel> model_;
    RewindSupport rewind_ = RewindSupport::kNone;
    /// \brief set only while text reuse is on
    OgaApi::Owned<OgaTokenizer> tokenizer_;
    std::optional<Live> live_;
};

}  // namespace flm::oga
