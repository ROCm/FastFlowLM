#pragma once

#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace flm::rai { class GgufFile; }

namespace flm::gemma4 {

/// \brief one row of the table: a shape to match, and what the file cannot say
///
/// The shape fields (`layers` through `global_head_dim`) are a CLAIM about a
/// model, matched against a real GGUF's own metadata and tensor shapes by
/// `RequireRowFor`. The four fields below them -- `vocab`, `group`,
/// `head_group`, `ple_group` -- are the answer no GGUF key gives: they are
/// facts about which ELFs DynamicDispatch ships at PDI pair (8, 17), read off
/// the ELF set itself, not guessed from the model's shape. A row existing is
/// not a claim that the model runs; it is a claim that *these particular
/// group sizes* are the ones its ELFs were built at.
///
/// Everything that differs between E2B and E4B beyond what is listed here --
/// FFN width pattern, the shared-cache pair, which layers are full-attention
/// -- is read from the file by later tasks (see gemma4_rai_gguf.cpp for FFN
/// width and SWA-layer derivation). A row does not carry it, and that a row
/// does not need to is the evidence the design is shape-driven rather than
/// E2B-shaped: nothing here bakes in one model's numbers as if they were
/// universal.
struct Gemma4Row {
    std::string_view name;
    std::int64_t layers;
    std::int64_t hidden;
    std::int64_t q_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;          // sliding-window layers
    std::int64_t global_head_dim;   // full-attention layers
    std::int64_t vocab;
    std::uint32_t group;
    std::uint32_t head_group;
    std::uint32_t ple_group;
};

/// \brief the rows that run -- currently E2B and E4B only
///
/// The other three Gemma 4 sizes (1B, 12B, 27B) are deliberately absent, not
/// merely un-added:
///   - they drop KV heads on their full-attention layers specifically, so
///     one scalar `kv_heads` field cannot describe them -- a row would need
///     a per-layer-kind KV head count, which is a different (larger) shape
///     this table does not model;
///   - they have no per-layer embeddings at all, so the whole PLE tensor
///     family (`per_layer_token_embd`, `per_layer_model_proj`, the per-layer
///     PLE trio) this port depends on does not exist in their GGUFs.
/// Adding a row for one of them would not be "one more entry" -- it would be
/// a different struct.
std::span<const Gemma4Row> Gemma4Rows();

/// \brief find the row matching a file's shape
/// \throws std::runtime_error naming every shape field, when none matches
///
/// Deliberately a lookup, not a formula: an unrecognized shape is refused
/// here, at load time, with a message that names the exact numbers and tells
/// the reader what to do next. Guessing a plausible-looking row for an
/// unknown shape would trade that clear, immediate rejection for a
/// dispatch failure far away in DynamicDispatch, over a shape whose group
/// sizes were never actually read off an ELF set -- a much harder failure
/// to trace back to "nobody added a row for this."
const Gemma4Row& RequireRowFor(std::int64_t layers, std::int64_t hidden,
                               std::int64_t q_heads, std::int64_t kv_heads,
                               std::int64_t head_dim,
                               std::int64_t global_head_dim);

/// \brief the MLP width of every layer, read off blk.N.ffn_gate.weight
///
/// \note NOT from `gemma4.feed_forward_length`. That metadata key is a
///       35-element ARRAY on E2B and a plain SCALAR 10240 on E4B -- its
///       *type* moves between conversions of one model family. A reader
///       that learned "scalar" truncates twenty of E2B's layers (E2B is
///       6144 for layers 0-14, then steps to 12288 from layer 15 on); one
///       that learned "array" fails to parse E4B at all. `config.json` is
///       no better: its `intermediate_size: 6144` is true of E2B's layers
///       0-14 and wrong about 15-34 -- exactly the class of error a paper
///       table reproduces faithfully.
///
///       The gate projection's own output shape is the only source that is
///       right about both files. In the FILE's own dims -- fastest-varying
///       first, so input width first -- `blk.N.ffn_gate.weight` is
///       `[1536, 6144]` / `[1536, 12288]` on E2B and `[2560, 10240]` on
///       E4B. `GgufFile::ShapeOf` REVERSES those, so what this function
///       reads is `ShapeOf[0]`, the OUTPUT width: `[6144, 1536]`,
///       `[12288, 1536]`, `[10240, 2560]`. Read `ShapeOf[1]` instead and
///       every layer of both models reports `hidden` -- a uniform,
///       plausible, wrong width that no shape check anywhere rejects.
///       Confirmed against the real files; see
///       test_gemma4_real_gguf.cpp, which asserts exactly these numbers
///       against gemma-4-E{2,4}B-it-Q8_0.gguf. Do not "simplify" this
///       back onto the metadata key; it is wrong on both models, in
///       opposite directions.
/// \throws std::runtime_error naming the tensor, if `blk.N.ffn_gate.weight`
///         is not 2-D.
std::vector<std::int64_t> DeriveFfnWidths(const flm::rai::GgufFile& file,
                                          std::int64_t layers);

/// \brief which layers use sliding-window attention, per layer
///
/// \note Read off `blk.N.attn_q.weight`'s own OUTPUT width -- `ShapeOf[0]`,
///       since `ShapeOf` is the reverse of the file's dims -- never off a
///       derived period. Sliding layers project to `q_heads * head_dim`; full-
///       attention layers project to `q_heads * global_head_dim`. At the
///       group sizes both shipped models use that is 2048 against 4096 --
///       the SAME two widths on E2B (hidden 1536) and E4B (hidden 2560),
///       which is exactly why width is the reliable discriminator and a
///       period is not: E2B's full-attention layers recur every 5th layer
///       (4, 9, 14, ... 34) and E4B's every 6th (5, 11, 17, ... 41). A
///       pattern generated from one model's period fits the other model's
///       weights nowhere -- it would produce a model that runs and is
///       silently wrong, computing attention with the wrong geometry on
///       most layers. Reading the width per layer, off the tensor that
///       actually carries it, needs no period at all.
/// \throws std::runtime_error naming the tensor, if `blk.N.attn_q.weight` is
///         not 2-D, or if its width matches neither the sliding nor the
///         full-attention geometry (both expected widths are named too --
///         this is a corrupt or unrecognized file, not a guess to round
///         toward the nearer one).
std::vector<bool> DeriveLayerIsSwa(const flm::rai::GgufFile& file,
                                   std::int64_t layers, std::int64_t q_heads,
                                   std::int64_t head_dim,
                                   std::int64_t global_head_dim);

/// \brief for each layer, the index of the layer owning the KV cache it reads
///
/// \param layer_is_swa which layers slide, from DeriveLayerIsSwa
/// \param kv_layers how many layers OWN a cache -- NOT
///        `gemma4.attention.shared_kv_layers` taken raw. That metadata key
///        counts the layers that SHARE a cache, the complement of what this
///        parameter means: `owning = block_count - shared_kv_layers` (20/18
///        shared gives 15/24 owning, on E2B/E4B -- see
///        verified-gguf-facts.md's "correction that matters most"). Passing
///        the raw key here would search the wrong prefix for "the last
///        own-cache layer of each kind": on E4B it would yield last-full=17
///        and last-sliding=16 instead of 23/22, and every shared layer would
///        bind the wrong cache -- a model that runs and is silently wrong.
///        Computing that subtraction belongs to whoever assembles the config
///        from the raw metadata, not to this function; this parameter takes
///        the count already corrected.
///
/// \note A layer that shares reads THE LAST OWN-CACHE LAYER OF ITS OWN KIND --
///       llama.cpp's rule (llama-model.cpp:2659-2667 through map_layer_ids in
///       llama-kv-cache.cpp:270). That gives 13 sliding / 14 full on E2B and
///       22 / 23 on E4B. Neither pair follows from the other, and neither
///       follows from a layer count, so a tabulated answer would be invisibly
///       wrong on one model even though it looked right on the other.
/// \note A layer below kv_layers owns its cache and maps to itself.
/// \throws std::runtime_error if kv_layers is outside 1..layer_is_swa.size(),
///         or if a shared layer's kind (sliding or full-attention) has no
///         owning layer at all -- naming the layer and the kind.
std::vector<std::int64_t> DeriveCacheOwners(const std::vector<bool>& layer_is_swa,
                                            std::int64_t kv_layers);

/// \brief everything about one loaded Gemma 4 GGUF, assembled from the three
///        derivations above plus the metadata keys that need no derivation
///
/// Every field either differs between E2B and E4B and is therefore read or
/// derived from the file (`layers`, `hidden`, `kv_heads`, `layer_*`,
/// `kv_layers`, ...), or is a fact about the ELF set no GGUF key states and
/// comes off the matched `Gemma4Row` instead (`vocab`, `group`,
/// `head_group`, `ple_group`).
struct Gemma4Config {
    std::string_view row_name;              // "E2B" or "E4B"
    std::int64_t layers;
    std::int64_t hidden;
    std::int64_t q_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;                  // sliding, 256
    std::int64_t global_head_dim;           // full, 512
    std::vector<std::int64_t> layer_intermediate;   // DeriveFfnWidths
    std::vector<bool> layer_is_swa;                 // DeriveLayerIsSwa
    std::vector<std::int64_t> layer_cache_owner;    // DeriveCacheOwners
    std::int64_t kv_layers;                 // OWNING count -- see DeriveCacheOwners
    std::int64_t sliding_window;            // 512 on both
    std::int64_t vocab;                     // from the row
    std::int64_t ple_dim;
    double rope_theta;                      // full-attention layers, 1e6
    double rope_theta_swa;                  // sliding layers, 1e4
    float eps;
    float logit_softcap;                    // 30.0
    std::int32_t eos_token_id;
    std::int32_t bos_token_id;
    bool add_bos;
    std::uint32_t group, head_group, ple_group;     // from the row
    std::int64_t context_length = kMaxPagedContext;  // gemma4.context_length

    /// \brief the head size a layer's attention runs at -- `head_dim` if it
    ///        slides, `global_head_dim` if it is full-attention
    std::int64_t LayerHeadDim(std::int64_t layer) const;
};

/// \brief assemble a Gemma4Config by reading `file`'s metadata and tensor
///        shapes, running the three derivations above, and matching the
///        result against `Gemma4Rows()`
///
/// \note `gemma4.attention.key_length` is the FULL-attention head size and
///       `gemma4.attention.key_length_swa` is the SLIDING one -- the GGUF's
///       names run OPPOSITE to this file's (`head_dim` is sliding,
///       `global_head_dim` is full). Reading them the obvious way round
///       swaps 256 and 512 through the whole model, silently. This is the
///       one place they are read; nowhere else needs to know.
///
/// \note `gemma4.attention.shared_kv_layers` counts the layers that SHARE a
///       cache, not the layers that OWN one -- `kv_layers` here is
///       `gemma4.block_count - gemma4.attention.shared_kv_layers`, never the
///       raw key. See DeriveCacheOwners's doc comment and
///       verified-gguf-facts.md's "correction that matters most": the raw
///       key is 20 on E2B and 18 on E4B; the owning counts this subtraction
///       produces are 15 and 24. Passing the raw key straight through would
///       search the wrong prefix for "the last own-cache layer of each
///       kind" and bind every shared layer to the wrong cache -- a model
///       that loads, runs, and is silently wrong.
///
/// \throws std::runtime_error if `general.architecture` is not "gemma4", if
///         the file's tensor count is not `6 + layers * 17` (names both the
///         actual and the expected count), if no `Gemma4Row` matches the
///         file's shape, or if any derivation above throws.
Gemma4Config ConfigFromMetadata(const flm::rai::GgufFile& file);

/// \brief a loaded Gemma 4 GGUF plus the config derived from it, with a
///        cross-check against the rest of the model package's JSON files
class Gemma4GgufPackage final {
public:
    static std::shared_ptr<Gemma4GgufPackage> Open(const std::filesystem::path& gguf_path);
    ~Gemma4GgufPackage();

    const Gemma4Config& Config() const noexcept;
    const flm::rai::GgufFile& File() const noexcept;
    const std::filesystem::path& Path() const noexcept;

    /// \brief cross-check the file against the package's JSON
    /// \note Runs before any device object exists, so a mismatched package
    ///       fails at load with an exact diagnostic rather than mid-generation.
    /// \note The real `config.json` is a MULTIMODAL config: `architectures`,
    ///       `model_type` and `eos_token_id` are top-level, but every field
    ///       describing the text model (`num_hidden_layers`, `vocab_size`,
    ///       `intermediate_size`, ...) is nested under `"text_config"`. This
    ///       reads those fields from `text_config`, falling back to the top
    ///       level only for a hypothetical flat config -- a check written
    ///       against a flat config rejects every real package, which is
    ///       exactly the failure mode this contract exists to prevent,
    ///       inverted. See verified-gguf-facts.md.
    /// \note `num_hidden_layers` is the only one of the cross-checked fields
    ///       that is FATAL if missing: it is the only one capable of
    ///       catching an E2B/E4B swap at all, since `vocab_size` (262144)
    ///       and `eos_token_id` (106) are identical between the two shipped
    ///       rows. Those two are checked when present and tolerated when
    ///       absent.
    /// \note Deliberately does NOT compare `config.json`'s `intermediate_size`
    ///       against anything: it is 6144 for E2B, which is true of layers
    ///       0-14 and wrong about 15-34 (which step to 12288). Comparing it
    ///       to the derived widths would reject a correct package -- see
    ///       DeriveFfnWidths's doc comment for the same trap in the GGUF's
    ///       own metadata.
    void ValidateGemma4Contract(const nlohmann::json& config,
                                const nlohmann::json& tokenizer,
                                const nlohmann::json& tokenizer_config) const;

private:
    struct Impl;
    explicit Gemma4GgufPackage(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// \brief the GGUF's eos first, then every other id config.json's top-level
///        `eos_token_id` array lists
std::vector<std::int64_t> ConfigEosIds(const nlohmann::json& config, std::int32_t gguf_eos);

}  // namespace flm::gemma4
