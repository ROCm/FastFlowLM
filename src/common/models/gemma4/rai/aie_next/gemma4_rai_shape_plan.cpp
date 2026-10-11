#include "models/gemma4/rai/aie_next/gemma4_rai_shape_plan.hpp"

#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "rai/kernel_grid.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace flm::gemma4 {
namespace {

/// \brief the row buckets `ple_bf16` ships a kernel for, taken verbatim from
///        its own doc comment in the pinned corelib.h: "1 for decode, then
///        64, 128, 256, 512, 1024, 2048, 3072, 4096". This plan also walks
///        these same buckets to query matmul/ssmlp/flat_mha, which is safe
///        even though those ops' own real grids may be coarser subsets --
///        every corelib pad helper here rounds UP, so an extra query point
///        never produces a wrong answer, only a redundant one.
constexpr std::array<std::int64_t, 9> kExecutionRows{
    1, 64, 128, 256, 512, 1024, 2048, 3072, 4096};

/// \brief round `rows` up to the row count `ple_bf16` ships a kernel for
///
/// \note There is NO `ryzenai_corelib_ple_bf16_pad_rows` (or any other ple
///       padding entry point) in the pinned 0.5.0 corelib.h -- confirmed by
///       reading the header, not assumed from the task brief that expected
///       one. `ryzenai_corelib_ple_bf16`'s own doc comment says M "comes
///       from `x`'s own extent and SELECTS THE KERNEL rather than bounding
///       the work" and "a tensor that sizes itself grows to it
///       automatically" -- there is nothing to ask corelib for here, only a
///       fixed table to reproduce. `FLM_CORELIB_FUNCTIONS`
///       (src/include/rai/corelib_api.hpp) accordingly has no ple entry at
///       all, so there is no ABI slot to call even if one wanted to.
std::int64_t RoundToPleBucket(std::int64_t rows) {
    for (const auto bucket : kExecutionRows) {
        if (rows <= bucket) return bucket;
    }
    return kExecutionRows.back();
}

std::int64_t MatmulRows(const corelib::ShapeGrid& grid, std::int64_t rows,
                        std::int64_t logical_k, std::int64_t logical_n,
                        std::uint32_t group, const char* logical_name) {
    return corelib::CoveringRows(grid, rows, logical_k, logical_n, group, logical_name);
}

/// \brief the flat_mha_bf16_desc for one (geometry, cache role) pair
///
/// \param swa true for the sliding-window geometry, false for full attention
/// \param shared true for a layer that READS another layer's KV cache
/// \note Called four times, once per pair, and the result is never written
///       again. See gemma4_rai_shape_plan.hpp's class comment for why this is
///       a 2x2 rather than two templates the layer loop patches.
ryzenai_corelib_flat_mha_bf16_desc BuildAttentionDesc(const Gemma4Config& config,
                                                      bool swa, bool shared) {
    ryzenai_corelib_flat_mha_bf16_desc desc{};
    desc.num_heads = config.q_heads;      // Gemma4Config::q_heads
    desc.kv_num_heads = config.kv_heads;  // Gemma4Config::kv_heads
    // Sliding layers run at head_dim (256); full-attention layers run at
    // global_head_dim (512) -- THIS is the field that makes the two
    // geometries resolve DIFFERENT ELFs (see this file's own top comment and
    // gemma4_rai_shape_plan.hpp's class doc).
    desc.head_size = swa ? config.head_dim : config.global_head_dim;
    desc.max_seq = kMaxSequenceLength;  // gemma4_rai_constants.hpp, fixed context length
    // FULL ROTARY ON BOTH GEOMETRIES. Settled, not open: the reference
    // driver sets `rope_dim=head_size` for all four descriptors
    // (gemma4_driver.py:1537-1543), and that is what makes the shipped
    // `..._256_4096_lc512_scale1` keys -- which carry NO emb_dim segment --
    // resolve. corelib passes no emb_dim when rope_dim == head_size.
    //
    // `config.json`'s `partial_rotary_factor: 0.25` is NOT a shorter rotary,
    // and reading it as one is the trap this comment replaces. It is a
    // per-pair FREQUENCY DIVISOR carried in the `rope_freqs.weight` tensor --
    // which is why that tensor is [256] on both E2B and E4B: one entry per
    // rotary PAIR of a 512-wide head, not a rotary width of 256. llama.cpp
    // passes it as `freq_factors` on FULL-ATTENTION LAYERS ONLY, where it
    // divides the frequency (`theta /= freq_factors[i/2]`).
    //
    // That makes it a fact about the ROPE TABLES (the cos/sin caches, Task
    // C7's `_rope_tables` equivalent), which are a separate INPUT to
    // flat_mha, not about this field. Two table pairs per model -- base 1e4
    // over the sliding head width with no factors, base 1e6 over the full
    // head width with factors -- held for the life of the engine and never
    // rewritten in place, because corelib memoizes the derived table on
    // buffer identity. Nothing about them changes `rope_dim`.
    desc.rope_dim = desc.head_size;
    // Sliding layers window at sliding_window (512, from the config); full-
    // attention layers see the whole sequence -- 0 is corelib's own "no
    // window" (ryzenai_corelib_flat_mha_bf16_desc's doc comment). The window
    // is passed UNCONDITIONALLY on every sliding layer: DD decides whether it
    // is actually active (`include_lc = 0 < window < sequence`) and a second
    // rule here could only disagree with it (gemma4_driver.py:1545-1548).
    desc.window = swa ? config.sliding_window : 0;
    // kv_shared selects the "_kvshare" ELFs for a layer that reads ANOTHER
    // layer's cache (Gemma4Config::layer_cache_owner). It is NOT a cosmetic
    // flag: a kvshare kernel "is not handed K at all: it never ropes a K row
    // and never writes one" (corelib.h:1772-1779), so getting it wrong in
    // either direction runs a different computation over the same buffers and
    // returns plausible numbers. The driver's statement of the stakes: "An
    // owning layer given a kvshare descriptor never writes its own cache; a
    // sharing layer given an owning one overwrites somebody else's. Neither
    // errors." (gemma4_driver.py:1526-1559.)
    //
    // So this is a descriptor axis, materialized here, and NOT a field a
    // layer loop patches on a shared template -- which is what this plan used
    // to do. See gemma4_rai_shape_plan.hpp's class comment.
    desc.kv_shared = shared ? 1 : 0;
    // GEMMA 4'S ACTUAL SOFTMAX SCALE, WHICH REALLY IS 1.0. llama.cpp
    // hard-codes `f_attention_scale = 1.0f` for this architecture
    // (src/models/gemma4.cpp:11) and it reaches the softmax unmodified; the
    // reference driver states the same verbatim
    // (ryzenai-corelib/python/gemma4_driver.py:239-247). This is not a knob
    // and not a value to look up in the config -- it is the model's number,
    // and it happens to be the one corelib keys its `_scale1` artifacts on.
    //
    // THE ENGINE MUST NOT PRE-SCALE Q. corelib's field doc says the `_scale1`
    // kernels apply no 1/sqrt(head_size) because "the caller has already
    // scaled Q"; for Gemma 4 that caller is THE MODEL'S OWN WEIGHTS. The
    // 1/sqrt(head) is carried by the learned QK-Norm gammas (constant-valued
    // vectors whose product is 0.125 on a sliding layer and 0.0625 on a full
    // one, and it is the K gamma that carries it, not the Q one). THERE IS
    // NOTHING FOR THE HOST TO FOLD. A Q pre-scale added on the strength of
    // that header sentence would scale attention a second time -- 16x on
    // sliding layers, 32x on full ones -- and no test in this suite could see
    // it. Reading the header's general statement about the kernel family as a
    // statement about this model is precisely the error this comment replaces.
    //
    // 0.0f, which this field used to hold, means "unset" and selects the
    // default (unscaled) family. That family DOES NOT SHIP for Gemma 4 E2B's
    // 8/1/256 signature -- corelib.h:1750-1756 names this exact signature as
    // shipping ONLY `_scale1` artifacts -- so Build()'s own
    // flat_mha_bf16_pad_rows lookup would be rejected on hardware with "no
    // attention kernel ships as ...". Same literal on both geometries, which
    // is also what the driver does (gemma4_driver.py:1550-1558).
    desc.scale = 1.0f;
    return desc;
}

ryzenai_corelib_paged_mha_bf16_desc AsPaged(const ryzenai_corelib_flat_mha_bf16_desc& flat,
                                            std::int64_t context) {
    ryzenai_corelib_paged_mha_bf16_desc desc{};
    desc.num_heads = flat.num_heads;
    desc.kv_num_heads = flat.kv_num_heads;
    desc.head_size = flat.head_size;
    desc.max_seq = context;
    desc.page_size = kPageSize;
    desc.rope_dim = flat.rope_dim;
    desc.window = flat.window;
    desc.kv_shared = flat.kv_shared;
    desc.scale = flat.scale;
    return desc;
}

}  // namespace

Gemma4ShapePlan Gemma4ShapePlan::Build(
    const std::shared_ptr<const corelib::CorelibApi>& api,
    ryzenai_corelib_stream_ptr stream, const Gemma4Config& config,
    std::int64_t context) {
    if (!api) throw std::invalid_argument("Gemma4ShapePlan corelib API is null");
    if (!stream) throw std::invalid_argument("Gemma4ShapePlan stream is null");

    Gemma4ShapePlan plan;
    // All four (geometry, cache role) descriptors, up front. Nothing below
    // writes to any of them again.
    for (const bool swa : {true, false}) {
        for (const bool shared : {false, true}) {
            plan.attention_descs_[DescIndex(swa, shared)] =
                BuildAttentionDesc(config, swa, shared);
        }
    }
    plan.paged_ = context > kPageSize;
    if (plan.paged_) {
        if (context > kMaxPagedContext || context % kPageSize != 0) {
            throw std::invalid_argument(
                "Gemma 4 paged context must be a multiple of 4096 up to " +
                std::to_string(kMaxPagedContext));
        }
        for (std::size_t index = 0; index < plan.attention_descs_.size(); ++index) {
            plan.paged_descs_[index] = AsPaged(plan.attention_descs_[index], context);
        }
    }

    // ssmlp must be planned at EVERY DISTINCT width in
    // config.layer_intermediate, not just the first layer's -- E2B steps
    // from 6144 to 12288 at layer 15, so a plan built from layer 0 alone
    // would be wrong for layers 15-34. E4B has exactly one width; the loop
    // below still runs once for it.
    std::vector<std::int64_t> widths;
    for (const auto width : config.layer_intermediate) {
        if (std::find(widths.begin(), widths.end(), width) == widths.end()) {
            widths.push_back(width);
        }
    }
    if (widths.empty()) {
        throw std::invalid_argument("Gemma4ShapePlan: config has no FFN widths");
    }

    plan.sliding_rows_.reserve(static_cast<std::size_t>(kMaxSequenceLength));
    plan.full_rows_.reserve(static_cast<std::size_t>(kMaxSequenceLength));
    const auto matmul = corelib::ShapeGrid::Matmul(*api, stream);
    const auto ssmlp_grid = corelib::ShapeGrid::SsMlp(*api, stream, true);
    // A multi-row pass at position 0 is attention over the first page, and
    // corelib's paged_mha runs it on flat_mha's own prefill kernel at the
    // pass's rows; every later chunk is a whole page on the paged kernel. So
    // the row extents come from flat_mha's grid even on a paged cache, and
    // the paged grid only has to carry the one page-sized chunk.
    const auto sliding_mha =
        corelib::ShapeGrid::FlatMha(*api, stream, plan.attention_descs_[DescIndex(true, false)]);
    const auto full_mha =
        corelib::ShapeGrid::FlatMha(*api, stream, plan.attention_descs_[DescIndex(false, false)]);
    if (plan.paged_) {
        for (const bool swa : {true, false}) {
            const auto paged = corelib::ShapeGrid::PagedMha(
                *api, stream, plan.paged_descs_[DescIndex(swa, false)]);
            const char* what = swa ? "sliding paged_mha chunk" : "full paged_mha chunk";
            if (corelib::CoveringRows(paged, kPageSize, 0, 0, -1, what) != kPageSize) {
                throw std::runtime_error(std::string(what) + " has no kernel at exactly " +
                                         std::to_string(kPageSize) + " rows");
            }
        }
    }

    for (const auto rows : kExecutionRows) {
        // ssmlp: gelu (activation = 1) with the post-feedforward norm SET
        // (post_feedforward_layernorm = 1).
        //
        // `activation` is a DECLARATION, not a switch: it is baked into the
        // ELF, appears nowhere in the artifact's name, and on the no-bias
        // path the runtime parameter DD derives from it is never written to
        // the weights BO -- so setting it wrongly would not change what the
        // NPU computes, only make the descriptor lie about what it computes.
        // Set correctly anyway: it is the only place the block's activation
        // is written down. NEVER "debug" by flipping it.
        //
        // `post_feedforward_layernorm` genuinely changes what the block
        // computes and writes: set, the norm moves inside the residual and
        // the `normalized` output plane is never written (though corelib
        // still requires a distinct buffer for that port) -- this is the
        // field that keys the "gemma_fusion*" ELFs instead of
        // "fusion_mladf_ssmlp*". See verified-gguf-facts.md's "CONSTRAINT
        // FOR GEMMA 4'S ENGINE" for the full data-flow consequence (Task
        // C8's problem, not this one's -- this plan only asks how such a
        // descriptor pads, it never dispatches ssmlp).
        std::int64_t ssmlp_rows = rows;
        for (const auto width : widths) {
            ssmlp_rows = std::max(
                ssmlp_rows, corelib::CoveringRows(ssmlp_grid, rows, config.hidden, width,
                                                  config.group, "ssmlp"));
        }

        const auto ple_rows = RoundToPleBucket(rows);
        // `ryzenai_corelib_rmsnorm_bf16_pad_rows` exists in corelib.h, but is
        // documented as an IDENTITY FUNCTION that does NOT report the
        // covering extent -- "sizing a buffer from its return value is the
        // trap this paragraph exists to name" (corelib.h's own
        // ryzenai_corelib_rmsnorm_bf16 doc). It is also not part of
        // FLM_CORELIB_FUNCTIONS (src/include/rai/corelib_api.hpp), so there
        // is no binding to call it through even if its answer were
        // trustworthy. Every rmsnorm in a Gemma 4 layer runs over the same
        // per-token residual stream `ple_bf16` already sizes at this bucket,
        // so its already-safe covering extent is reused here rather than
        // inventing a second, uncheckable source of truth.
        const auto rmsnorm_rows = ple_rows;

        for (const bool swa : {true, false}) {
            // The OWNING descriptor of this geometry, and that choice is not
            // arbitrary: the row extents this loop fills are per GEOMETRY
            // only, never per (geometry, cache role). The header says so as
            // the stated reason flat_mha_bf16_elf_name exists at all -- the
            // descriptor's "window, kv_shared and scale do not change the
            // shapes of anything -- they change WHICH KERNEL RUNS, by
            // changing the name DD resolves" (corelib.h:1810-1812). So
            // flat_mha_bf16_pad_rows cannot answer differently for the
            // kvshare family, and sliding_rows_/full_rows_ are two tables,
            // not four. Do not "fix" this into a four-way split.
            const auto& desc = plan.attention_descs_[DescIndex(swa, /*shared=*/false)];
            // EVERY PROJECTION -- Q, K, V AND the output -- is packed at
            // `config.group`. Not a judgement call: the reference driver
            // declares the field as "group: int  # every projection"
            // (gemma4_driver.py:537-539), its `_matmul_weights` defaults
            // `group_size=self.cfg.group` (:1383-1387), and the ONLY call in
            // the whole driver that overrides that default is lm_head
            // (:1691-1694). Its enumeration of a prefill's dispatches puts
            // all three shapes at g32 == `cfg.group`: "Q/K/O at 1536->2048,
            // 1536->256 and 2048->1536 on a sliding layer and 1536->4096,
            // 1536->512 and 4096->1536 on a full one" (:294-298).
            //
            // `config.head_group` is lm_head's, and lm_head's alone -- see
            // the single query after this loop. It is a SEPARATE FIELD
            // because on other models the two genuinely differ; on both
            // shipped Gemma 4 rows they are both 32 (gemma4_rai_gguf.cpp),
            // which is exactly why a version of this code that queried Q and
            // K/V at `head_group` passed every test in the suite. Do not
            // reintroduce it here on the strength of "head"-shaped outputs:
            // `head_group` is named for lm_head, not for the attention heads.
            Gemma4RowExtents extents{};
            extents.q_rows = MatmulRows(matmul, rows, config.hidden,
                                        desc.num_heads * desc.head_size,
                                        config.group, "query");
            extents.kv_rows = MatmulRows(matmul, rows, config.hidden,
                                         desc.kv_num_heads * desc.head_size,
                                         config.group, "key/value");
            extents.output_rows = MatmulRows(matmul, rows,
                                             desc.num_heads * desc.head_size,
                                             config.hidden, config.group,
                                             "output");
            extents.ssmlp_rows = ssmlp_rows;
            extents.rmsnorm_rows = rmsnorm_rows;
            extents.ple_rows = ple_rows;

            const auto& mha_grid = swa ? sliding_mha : full_mha;
            extents.flat_mha_rows = corelib::CoveringRows(mha_grid, rows, 0, 0, -1, "flat_mha");

            auto& bucket_rows = swa ? plan.sliding_rows_ : plan.full_rows_;
            while (bucket_rows.size() < static_cast<std::size_t>(rows)) {
                bucket_rows.push_back(extents);
            }
        }
    }

    // LM_HEAD, AND IT IS THE ONE MATMUL OUTSIDE THE BUCKET LOOP ABOVE.
    //
    // Two things make it different from every other matmul in the model, and
    // both come from the reference driver rather than from this port:
    //
    //   - IT HAS EXACTLY ONE ROW SHAPE. "ONE ROW IS NOT A SIMPLIFICATION, IT
    //     IS THE ONLY SHAPE AVAILABLE. lm_head is 1536 -> 262144 and ships at
    //     M == 1 alone at this model's (8, 17) pair, so there is no batched
    //     form to offer even if a caller wanted every position's logits"
    //     (gemma4_driver.py:2449-2453). The driver says the same from the
    //     prefill side -- "lm_head runs at M=1, so neither is a prefill
    //     matmul" (:294-298). Walking it through `kExecutionRows` would
    //     therefore ask corelib about shapes that do not exist; at best the
    //     answers are meaningless, at worst the query itself fails on
    //     hardware.
    //   - IT IS THE ONE WEIGHT PACKED AT `head_group`. "TIED to token_embd --
    //     this conversion has no output.weight -- and at `head_group`, which
    //     on this model happens to equal `group`" (:1691-1694); "the only
    //     weight packed at `cfg.head_group`" (:2449-2453).
    //
    // It is asked of corelib rather than hardcoded to 1 because the answer is
    // corelib's to give, and because a fake or a future corelib that answers
    // otherwise should show up here rather than in C7's allocation.
    plan.lm_head_rows_ = MatmulRows(matmul, /*rows=*/1, config.hidden,
                                    config.vocab, config.head_group, "lm_head");

    return plan;
}

const Gemma4RowExtents& Gemma4ShapePlan::ForRows(std::size_t live_rows, bool swa) const {
    const auto& bucket_rows = swa ? sliding_rows_ : full_rows_;
    if (live_rows == 0 || live_rows > bucket_rows.size()) {
        throw std::out_of_range("Gemma 4 live rows must be in 1.." +
                                std::to_string(bucket_rows.size()));
    }
    return bucket_rows[live_rows - 1];
}

const ryzenai_corelib_flat_mha_bf16_desc&
Gemma4ShapePlan::attention_desc(bool swa, bool shared) const noexcept {
    return attention_descs_[DescIndex(swa, shared)];
}

const ryzenai_corelib_paged_mha_bf16_desc&
Gemma4ShapePlan::paged_attention_desc(bool swa, bool shared) const noexcept {
    return paged_descs_[DescIndex(swa, shared)];
}

std::int64_t Gemma4ShapePlan::lm_head_rows() const noexcept { return lm_head_rows_; }

}  // namespace flm::gemma4
