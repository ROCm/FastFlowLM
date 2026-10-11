#pragma once

#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "rai/corelib_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flm::gemma4 {

/// \brief one finished packed weight, and the named slot it was placed in
struct WeightPlacement {
    /// \brief the GGUF tensor this slot holds, e.g. "blk.7.attn_q_norm.weight";
    ///        "blk.7.ssmlp" and "blk.7.ple" for the two fused blocks
    std::string slot;
    const void* handle;
};

/// \brief the Gemma 4 engine on ryzenai-corelib: every packed weight the model
///        needs, held for the life of the object
///
/// Construction opens one stream on the (8, 17) PDI pair, builds the shape
/// plan, creates every packed weight (matmul, ssmlp, rmsnorm and ple),
/// allocates the activation buffers, the non-uniform KV caches and both
/// rotary table pairs. `forward` and `prefill` run the layer loop and return
/// softcapped logits for the last position.
///
/// A layer that owns its KV cache (`index < Gemma4Config::kv_layers`) has ten
/// weight objects; a layer that shares an earlier layer's cache has six -- it
/// never projects or norms K and V, although the file still carries those
/// tensors.
class gemma4_rai final : public causal_lm {
public:
    /// \param package the GGUF and the config derived from it
    /// \param runtime the corelib runtime this engine's objects belong to
    /// \param config the frontend's model configuration; not read for shape,
    ///        every dimension comes from the GGUF
    /// \param max_length the context this engine is built for: 1..4096, or a
    ///        multiple of 4096 up to kMaxPagedContext, and no longer than the
    ///        GGUF's own `gemma4.context_length`
    gemma4_rai(std::shared_ptr<Gemma4GgufPackage> package,
               std::shared_ptr<corelib::CorelibRuntime> runtime,
               const LM_Config& config,
               std::uint32_t max_length = 4096);

    /// \brief construct with an overridden shape, for tests only
    /// \note The shipped rows use one group size everywhere, so only an
    ///       overridden shape can tell `group`, `head_group` and `ple_group`
    ///       apart. `shape` is not validated against the file.
    static std::unique_ptr<gemma4_rai> CreateWithShapeForTest(
        std::shared_ptr<Gemma4GgufPackage> package,
        std::shared_ptr<corelib::CorelibRuntime> runtime,
        const LM_Config& config, Gemma4Config shape,
        std::uint32_t max_length = 4096);

    ~gemma4_rai() override;

    /// \brief one decode step: one token in, softcapped logits out
    /// \throws std::out_of_range if the token is outside the vocabulary, or
    ///         the request runs past the configured context
    buffer<bf16> forward(int id) override;
    /// \brief one multi-token pass: `ids` in, softcapped logits for the last
    ///        position out
    /// \note Same path as `forward`. A multi-row pass must start at position 0
    ///       or, on a paged cache, on a page boundary, so a multi-row request
    ///       at any other position re-runs the tokens behind it from the
    ///       nearest such start together with `ids`: from position 0 unpaged,
    ///       from the current page's first row paged.
    /// \throws std::invalid_argument for an empty request, or a continuation
    ///         whose earlier tokens this engine does not hold
    /// \throws std::out_of_range as `forward`
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int length) override;
    /// \brief unsupported; weights come from the GGUF package. Present because
    ///        causal_lm.hpp is a frozen ABI.
    /// \throws std::runtime_error always
    void load_weights(Q4NX&) override;
    void update_max_length(std::uint32_t max_length) override;
    void clear_context() override;
    /// \brief one position's K row, gathered across the KV heads
    /// \note Resolved through `Gemma4Config::layer_cache_owner`: a sharing
    ///       layer returns the cache it attends over. The width is
    ///       `kv_heads * head` of the owner, so it varies across layers.
    /// \throws std::out_of_range if the layer or the index is out of range
    buffer<bf16> get_k_cache(int layer, int index) override;
    /// \brief one position's V row; see get_k_cache
    buffer<bf16> get_v_cache(int layer, int index) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    /// \brief whether a forward pass failed after a dispatch was accepted
    /// \note Set when a pass throws after corelib accepted one of its
    ///       dispatches, or when its result is non-finite. Requests refused
    ///       before that (empty, bad token, chunked prefill, past capacity) do
    ///       not poison. Nothing clears it; every other entry point throws
    ///       while it is set, and the only exit is unload and reload. The
    ///       three const accessors below stay usable.
    bool poisoned() const noexcept;

    /// \brief whether this load bound packed weights from the on-disk cache
    bool loaded_from_cache() const noexcept;

    /// \brief how many packed weight objects this engine holds; also the
    ///        weight-cache key's slot count
    std::size_t weight_slot_count() const noexcept;

    /// \brief every packed weight paired with the slot it was placed in, for
    ///        tests only
    /// \note Reads the named members, not the slot table, so that it is an
    ///       independent statement of which tensor belongs in which member.
    ///       Size is `weight_slot_count()`.
    std::vector<WeightPlacement> WeightPlacementsForTest() const;

private:
    struct Impl;
    explicit gemma4_rai(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace flm::gemma4
