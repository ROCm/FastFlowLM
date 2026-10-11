#pragma once

#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "rai/corelib_api.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace flm::gemma4 {

/// \brief the padded row extent every op needs for one bucket, one geometry
struct Gemma4RowExtents {
    std::int64_t q_rows;
    std::int64_t kv_rows;
    std::int64_t output_rows;
    std::int64_t ssmlp_rows;
    /// \note Not queried through corelib -- see gemma4_rai_shape_plan.cpp's
    ///       Build() for why `ryzenai_corelib_rmsnorm_bf16_pad_rows` cannot
    ///       be trusted for this even though it exists.
    std::int64_t rmsnorm_rows;
    /// \note Not queried through corelib either -- there is no
    ///       `ryzenai_corelib_ple_bf16_pad_rows` in the pinned 0.5.0 header.
    ///       See Build()'s own comment.
    std::int64_t ple_rows;
    std::int64_t flat_mha_rows;
};

/// \brief asks corelib how it wants Gemma 4's operators padded, for BOTH
///        attention geometries at once, and holds the FOUR attention
///        descriptors a Gemma 4 layer loop can need
///
/// TWO AXES, AND THEY ARE NOT THE SAME KIND OF THING.
///
/// The first is GEOMETRY. Gemma 4 runs two in one stack: sliding-window layers
/// at head_dim 256, full-attention layers at head_dim 512. They resolve
/// DIFFERENT ELFs (`ryzenai_corelib_flat_mha_bf16_desc::head_size` is part of
/// what selects the kernel) and they genuinely have different SHAPES, so a
/// single row-extent table cannot answer for both -- `ForRows(rows, swa)`
/// keeps one per geometry.
///
/// The second is CACHE ROLE: whether a layer owns the KV cache it attends
/// over or reads one an earlier layer wrote (`Gemma4Config::layer_cache_owner`).
/// `kv_shared` selects the "_kvshare" artifacts, which are handed no K operand
/// at all and merge nothing into the cache (corelib.h:1772-1779). This is a
/// 2x2, not a second copy of the geometry split.
///
/// BUT THE ROW EXTENTS DO NOT SPLIT ON THE SECOND AXIS. The header states it
/// outright, as the very reason `flat_mha_bf16_elf_name` exists: the
/// descriptor's "window, kv_shared and scale do not change the shapes of
/// anything -- they change WHICH KERNEL RUNS, by changing the name DD
/// resolves" (corelib.h:1810-1812). So `flat_mha_bf16_pad_rows` cannot depend
/// on `kv_shared`, the owning and sharing families share one row grid per
/// geometry, and `sliding_rows_`/`full_rows_` are correct as two tables rather
/// than four.
///
/// WHY ALL FOUR DESCRIPTORS ARE MATERIALIZED ANYWAY. An earlier version of
/// this plan held two templates with `kv_shared = 0` and left the layer loop
/// to override the field per layer before dispatching. The reference driver
/// keys four descriptors on `(sliding, shares-a-cache)` instead
/// (`ryzenai-corelib/python/gemma4_driver.py:1526-1559`) and says why: "An
/// owning layer given a kvshare descriptor never writes its own cache; a
/// sharing layer given an owning one overwrites somebody else's. Neither
/// errors." A mutated shared template makes that a one-missed-assignment bug
/// with no failure mode; four immutable descriptors and a selecting accessor
/// make it unrepresentable.
class Gemma4ShapePlan final {
public:
    /// \brief interrogate corelib for the padded extent of every live row
    ///        count, for both geometries, plus one ssmlp query per distinct
    ///        FFN width the model has
    /// \param api the corelib binding
    /// \param stream the stream the plan will be dispatched on -- corelib
    ///        0.5.0 takes the stream on every padding helper, because the PDI
    ///        pair a stream was opened with selects the ELF set, and a shape
    ///        exists under one pair and not another
    /// \param config the model's shape, read off its GGUF (Task C5)
    /// \param context the engine's context. Above kPageSize the plan is paged:
    ///        attention descriptors carry that window and a kPageSize page.
    static Gemma4ShapePlan Build(
        const std::shared_ptr<const corelib::CorelibApi>& api,
        ryzenai_corelib_stream_ptr stream, const Gemma4Config& config,
        std::int64_t context = kMaxSequenceLength);

    bool paged() const noexcept { return paged_; }

    /// \param live_rows the live (unpadded) row count a caller wants extents
    ///        for, 1..4096
    /// \param swa true for a sliding-window layer, false for full attention
    /// \note Held PER GEOMETRY because head_dim 256 and 512 resolve DIFFERENT
    ///       ELFs. One table cannot answer for both.
    const Gemma4RowExtents& ForRows(std::size_t live_rows, bool swa) const;

    /// \param swa    true for a sliding-window layer, false for full attention
    /// \param shared true for a layer that READS another layer's KV cache
    /// \note SELECT, never mutate. The returned descriptor is one of four the
    ///       plan built once; assigning `kv_shared` on a copy held across a
    ///       layer loop is the bug this accessor exists to remove -- see the
    ///       class comment.
    /// \note THERE IS DELIBERATELY NO ONE-ARGUMENT OVERLOAD. An earlier draft
    ///       of this fix kept `attention_desc(swa)` meaning "the owning one",
    ///       which reintroduces the failure in a quieter form: a layer loop
    ///       that writes `attention_desc(layer_is_swa[i])` and forgets the
    ///       cache role compiles, runs, and hands every sharing layer an
    ///       owning descriptor. Requiring the second argument turns that into
    ///       a compile error. Callers that genuinely want the owning one say
    ///       `attention_desc(swa, /*shared=*/false)`.
    const ryzenai_corelib_flat_mha_bf16_desc& attention_desc(
        bool swa, bool shared) const noexcept;

    /// \note Valid only when paged() is true. Same (geometry, cache role)
    ///       index as attention_desc.
    const ryzenai_corelib_paged_mha_bf16_desc& paged_attention_desc(
        bool swa, bool shared) const noexcept;

    /// \brief the padded row extent lm_head's matmul runs at -- the leading
    ///        extent of the `[rows, vocab]` logits tensor a caller allocates
    /// \note ONE, on every shipped Gemma 4, and NOT a simplification. lm_head
    ///       is `hidden -> vocab` and ships at M == 1 alone at this model's
    ///       (8, 17) PDI pair: "there is no batched form to offer even if a
    ///       caller wanted every position's logits"
    ///       (`ryzenai-corelib/python/gemma4_driver.py:2449-2453`). It is
    ///       still asked of corelib rather than hardcoded, because the answer
    ///       is corelib's to give. See Build().
    std::int64_t lm_head_rows() const noexcept;

private:
    /// \brief index into `attention_descs_` for one (geometry, cache role)
    /// \note A free-standing constexpr rather than an inline expression at
    ///       each use, so the accessor and Build() cannot drift into
    ///       disagreeing about the ordering.
    static constexpr std::size_t DescIndex(bool swa, bool shared) noexcept {
        return (swa ? 0u : 2u) + (shared ? 1u : 0u);
    }

    std::vector<Gemma4RowExtents> sliding_rows_;
    std::vector<Gemma4RowExtents> full_rows_;
    /// \brief lm_head's padded row extent -- see lm_head_rows()
    /// \note NOT part of Gemma4RowExtents: those are per row bucket and per
    ///       geometry, and lm_head is neither. It has one shape.
    std::int64_t lm_head_rows_{0};
    /// \brief the four (geometry, cache role) descriptors, by DescIndex
    /// \note Materialized once in Build() and never written again. See the
    ///       class comment for why four and not two.
    std::array<ryzenai_corelib_flat_mha_bf16_desc, 4> attention_descs_{};
    std::array<ryzenai_corelib_paged_mha_bf16_desc, 4> paged_descs_{};
    bool paged_{false};
};

}  // namespace flm::gemma4
