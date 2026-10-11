#include "models/gemma4/rai/aie_next/gemma4_rai.hpp"

#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_host.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_shape_plan.hpp"
#include "rai/corelib_object.hpp"
#include "rai/gguf_file.hpp"
#include "rai/weight_cache.hpp"
#include "rai/weight_source.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace flm::gemma4 {
namespace {
using namespace flm::corelib;
using namespace flm::rai;

std::string Blk(std::int64_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + suffix;
}

/// \brief `f32` as the BF16 the kernels read
std::vector<std::uint16_t> ToBf16(std::span<const float> values) {
    std::vector<std::uint16_t> out(values.size());
    FloatsToBf16(values, out);
    return out;
}

/// \brief `source`, a [rows, cols] row-major matrix, as a [cols, rows] one
/// \note THE ONE PLACE A GGUF MAPPING IS RE-LAID-OUT IN THIS ENGINE. Every
///       other corelib port takes the file's own memory order (see the
///       layout table in the review of C7/R1: matmul's `blocks`, ssmlp's
///       gate/up/down, both norm ports are all correct as mapped). `ple` is
///       the exception, and the transpose is materialised rather than
///       described because `ryzenai_corelib_ple_bf16_weights_create` takes a
///       dense FP32 source with no stride.
std::vector<float> Transposed(std::span<const float> source, std::int64_t rows,
                              std::int64_t cols) {
    const auto row_count = static_cast<std::size_t>(rows);
    const auto column_count = static_cast<std::size_t>(cols);
    std::vector<float> out(row_count * column_count);
    for (std::size_t r = 0; r < row_count; ++r) {
        const auto* in_row = source.data() + r * column_count;
        for (std::size_t c = 0; c < column_count; ++c)
            out[c * row_count + r] = in_row[c];
    }
    return out;
}

/// \brief a gamma of exactly 1.0, `k` wide
/// \note V's norm. The file carries NO learned weight for it
///       (llama.cpp src/models/gemma4.cpp:256), and `rmsnorm_bf16` ships no
///       weightless variant, so "normalize only" is spelled as a vector of
///       ones -- which is exact in BF16. The reference driver keeps exactly
///       this vector, for exactly this reason (`gemma4_driver.py`
///       `_ones_norm_weights`): "the cheap wrong implementation omits the
///       step entirely and still produces fluent text; on this model that is
///       a whole 1/sqrt(mean(v^2)) missing from every attended value."
std::vector<std::uint16_t> OnesBf16(std::int64_t k) {
    const std::vector<float> ones(static_cast<std::size_t>(k), 1.0f);
    return ToBf16(ones);
}

/// \brief everything besides the GGUF and corelib version that shapes the
///        packed bytes
std::string CacheLayout(const Gemma4Config& cfg) {
    return "gemma4 " + std::string(cfg.row_name) + " group=" + std::to_string(cfg.group) +
           " head_group=" + std::to_string(cfg.head_group) +
           " ple_group=" + std::to_string(cfg.ple_group) +
           " layout_revision=" + std::to_string(kPackedLayoutRevision);
}

enum class SlotKind { Matmul, SsMlp, RmsNorm, Ple };

/// \brief one packed weight: how to make it, where to put it, how to cache it
///
/// ONE TABLE, WALKED THE SAME WAY BOTH DIRECTIONS. The pack path runs
/// `pack()` for every slot across a thread pool; the cache path calls the
/// matching `..._weights_create` with the cached blob as its one `packed`
/// source and `assign()`s the handle. Because both walk the
/// same vector in the same order, a cache index entry always refers to the
/// weight it was written from -- which is the property Phi-4's engine gets
/// from a hand-written `slot % 5` scheme and Gemma 4 cannot, because its
/// per-layer slot count is not constant (ten on an owning layer, six on a
/// sharing one).
///
/// ASSIGNMENT IS BY SLOT, NEVER BY COMPLETION ORDER. `pack()` writes into
/// the one member it owns; the pool's workers finish in whatever order they
/// finish in and nothing appends to a shared list.
struct WeightSlot {
    SlotKind kind{};
    std::string label;
    /// \brief the layer this slot belongs to, or -1 for a model-level one
    /// \note Only ple uses it, to find its blob. Kept on every slot so the
    ///       labels and the blob index cannot drift apart.
    std::int64_t layer{-1};

    ryzenai_corelib_matmul_bf16_weights_desc matmul{};
    ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp{};
    ryzenai_corelib_rmsnorm_bf16_weights_desc rmsnorm{};
    ryzenai_corelib_ple_bf16_weights_desc ple{};

    /// \brief pack this weight from the GGUF and store it
    std::function<void()> pack;
    /// \brief adopt an already-created handle (the cache path)
    std::function<void(void*)> assign;
    /// \brief the live handle, for the cache write; null before packing
    std::function<void*()> handle;
};

}  // namespace

struct gemma4_rai::Impl {
    /// DECLARATION ORDER IS DESTRUCTION ORDER, REVERSED, AND IT IS LOAD-
    /// BEARING. Members are destroyed bottom-up, so the packed weights go
    /// first, then the stream, then the API and the package, and the runtime
    /// last of all. corelib objects must not outlive the API they came from,
    /// and A PACKED OBJECT MUST BE RELEASED BEFORE THE STREAM IT WAS PLANNED
    /// AGAINST.
    ///
    /// THAT LAST RULE IS MEASURED, NOT TIDINESS. corelib access-violates
    /// inside `ryzenai_corelib_object_release` when an object outlives its
    /// Stream: with two models in one process and the first torn down
    /// stream-first, the reference driver printed `Windows fatal exception:
    /// access violation` in 7 of 10 runs, up to 221 lines in one, against
    /// 0 of 10 for the packed-objects-first order. Box 43,
    /// `test_gemma4_weights.py`, twenty runs over the two orders, all of it
    /// in `t17r1d.log`.
    ///
    /// WHAT THE EXIT CODES SAY, STATED AS MEASURED. An earlier version of
    /// this comment said "AND THE PROCESS EXITED 0 EVERY TIME", and the
    /// source does not support that. What the corelib model notes record
    /// is that across ALL TWENTY runs -- both arms
    /// together -- nineteen sessions exited 0 with every test PASSED and THE
    /// TWENTIETH EXITED 1, with no violations at all. The document does not
    /// break exit codes down by arm, so nothing establishes what the
    /// stream-first ten did. "Nothing raised to the caller" is narrower
    /// still: what was measured is that a recorder in `Object.__del__`
    /// caught ZERO of the 833 violations those runs printed, and the source
    /// says in terms that a mechanism must not be re-derived from that zero.
    /// Every figure here is one shared box that takes no lock on the NPU,
    /// which is why the driver's own docstring declines to repeat any of
    /// them.
    ///
    /// THE CONCLUSION IS UNCHANGED, WHICH IS WHY THE MEASUREMENT IS WORTH
    /// GETTING RIGHT. At 19 of 20 an exit-code test asserts nothing either
    /// way, so the guard here has to be structural. Their driver met this
    /// only because a second model entered one process; FLM loads and
    /// unloads models routinely, so it is a shipping path here. The
    /// underlying defect is corelib's, is undiagnosed and unfixed, and is
    /// not touched here -- this engine simply does not reach it.
    ///
    /// THERE IS DELIBERATELY NO `~Impl()`, and that is the C9 decision rather
    /// than an omission. A destructor body runs BEFORE member destruction, so
    /// a hand-written `weights.clear(); ...; stream.reset();` list would
    /// release the stream before any member the list FORGOT -- which is
    /// precisely the fault, arriving silently the first time somebody adds a
    /// member and does not update the list. Declaration order covers every
    /// member automatically and has exactly one failure mode: a new
    /// corelib-owning member declared above `stream`. That one failure mode
    /// is what `TestEveryCorelibOwningMemberIsDeclaredAfterTheStream` in
    /// src/test/gemma4_rai/test_gemma4_engine.cpp reads this block to catch.
    /// A behavioural test cannot: the fault is intermittent and raises
    /// nothing, so one that waited for a crash would pass on a good day with
    /// bad code.
    ///
    /// IF YOU ADD A MEMBER: a corelib handle, or anything holding one, goes
    /// BELOW `stream`. Host memory corelib may have BOUND goes ABOVE the
    /// weights that bound it, and needs its own line in that test, because
    /// nothing in the type says which buffers those are.
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<Gemma4GgufPackage> package;
    std::shared_ptr<const CorelibApi> api;
    Gemma4Config shape;
    std::uint32_t max_length;
    /// \brief context longer than one flat_mha window: paged KV, PDI (8, 33)
    bool paged{};
    /// \brief rows the KV cache and rotary tables hold
    /// \note Unpaged this stays kMaxSequenceLength even when the caller asked
    ///       for less, which is the pitch flat_mha addresses. Paged it is the
    ///       constructed context, a whole number of pages.
    std::uint32_t cache_rows{};
    int position{};
    std::optional<int> saved;
    bool poisoned{};
    bool from_cache{};
    std::size_t slot_count{};

    UniqueStream stream;
    Gemma4ShapePlan plan;

    struct LayerWeights {
        /// Present on every layer.
        UniqueMatMulWeights q, o;
        UniqueRmsNormWeights q_norm, post_attn_norm;
        UniqueSsMlpWeights mlp;
        UniquePleWeights ple;
        /// Present ONLY on a layer that owns its KV cache. A sharing layer
        /// never projects or norms K and V -- the kvshare attention kernel
        /// "is not handed K at all" -- so these stay empty, and that is the
        /// config's decision, not the file's: the tensors exist on every
        /// layer of every shipped conversion.
        UniqueMatMulWeights k, v;
        UniqueRmsNormWeights k_norm, v_norm;
    };
    std::vector<LayerWeights> layers;

    /// \brief every activation buffer, allocated once at the WIDER geometry
    ///
    /// SIZED AT THE WIDEST HEAD THIS MODEL HAS, THEN WINDOWED PER LAYER ON THE
    /// COLUMN AXIS. Gemma 4 runs two geometries in one stack -- on E2B Q, O
    /// and the attention output are 4096 wide on a full layer and 2048 on a
    /// sliding one, K/V 512 and 256 -- and each op reads its row and column
    /// count off its input's shape, so a window is what says which kind of
    /// layer this is. One buffer at the maximum costs a few MB more than the
    /// largest single layer needs and removes any question of which kind wrote
    /// a buffer last.
    ///
    /// THAT IS THE COLUMN AXIS AND IT DOES NOT LICENSE THE ROW AXIS. A padded
    /// buffer is not a window onto a bigger one: every op here reads and
    /// writes its whole padded extent, so the row count a window declares is
    /// the row count the kernel runs, not a bound on it.
    ///
    /// FOUR HIDDEN-WIDTH BUFFERS AND NO SWAP, which is where this differs from
    /// Phi-4. Phi-4 ping-pongs `hidden`/`normalized` because its fused MLP
    /// writes the next layer's input into the buffer it just read; here `ple`
    /// produces BOTH the next residual and the next normed input, so four
    /// roles rotate through four fixed buffers and every op in a layer has
    /// four distinct operands:
    ///
    ///     hidden     the normed input, then o_proj's output, then the plane
    ///                ssmlp does NOT write, then ple's norm_out
    ///     skip_sum   post_attention_norm's output, ssmlp's input
    ///     ple_out    ssmlp's block output (plane 0), which is ple's `x`
    ///     residual   the residual stream, which ple overwrites with the next
    UniqueTensor hidden, residual, skip_sum, ple_out;
    /// \brief the projections' outputs, and QK-Norm's separate destinations
    /// \note `q_normed`/`k_normed` are NOT a buffer that could be saved by
    ///       normalizing in place. corelib.h measures in-place `rmsnorm_bf16`
    ///       CORRUPTING at M = 1024 and M = 2048 at k=128 -- on EXACT-kernel
    ///       dispatches, so "a kernel ships at this shape" is not the test --
    ///       with a failure shape ("a handful of consecutive rows out of a
    ///       thousand") that keeps correlation high and rarely moves an
    ///       argmax. A prefill Q-norm runs at `padded * q_heads` rows, which
    ///       is exactly that range.
    /// \note `v_buf` is the buffer Phi-4 does not need. Gemma 4 RMS-normalizes
    ///       V between the projection and the cache, and no norm can scatter,
    ///       so the projection lands here and THE NORM WRITES THE CACHE.
    UniqueTensor q_buf, k_buf, v_buf, q_normed, k_normed, attn_out;
    /// \brief lm_head's input and output, one row each
    /// \note Not a simplification: lm_head is `hidden -> vocab` and ships at
    ///       M == 1 alone at this model's (8, 17) PDI pair.
    UniqueTensor last_hidden, logits;
    /// \brief the per-layer embedding input planes, one tensor per layer
    /// \note One per layer rather than one [layers, rows, ple_dim] tensor so
    ///       that a layer's slice is a single write and a single bind.
    std::vector<UniqueTensor> ple_in;

    /// \brief the KV caches: `kv_layers` of them, NOT one per layer, and NOT
    ///        uniform in width
    ///
    /// CACHE `i` IS AS WIDE AS LAYER `i`'S HEAD. On E2B caches 4, 9 and 14 are
    /// 512 and the other twelve are 256; on E4B it is 5, 11, 17 and 23. Sizing
    /// them all at one head width "either wastes half the allocation or,
    /// worse, hands the full-attention layers a cache with the wrong row pitch
    /// -- which the kernel would address happily".
    ///
    /// PINNED TO `kMaxSequenceLength`, NEVER GROWN TO FIT, and that is
    /// corelib's rule rather than a convenience: "the kernel addresses them at
    /// that row pitch, so a cache sized to the current sequence changes stride
    /// every token and reads the wrong rows". The pitch it means is
    /// `flat_mha_bf16_desc::max_seq`, which this engine's shape plan sets to
    /// `kMaxSequenceLength` -- so these two numbers are one number, and
    /// allocating at the constructor's `max_length` instead would desynchronize
    /// them. `max_length` bounds what a caller may REQUEST, not what is
    /// allocated.
    std::vector<UniqueTensor> k_cache, v_cache;

    /// \brief the two rotary table pairs, sliding then full-attention, as host
    ///        views over engine-owned storage
    /// \note Two pairs rather than one rewritten per layer: corelib memoizes
    ///       the derived table on the view's identity. Written once at load and
    ///       never touched again. corelib copies nothing, so the storage is
    ///       declared first and outlives the views.
    std::vector<float> rope_cos_sliding, rope_sin_sliding, rope_cos_full, rope_sin_full;
    /// \brief the tokens behind `position`, so a multi-row continuation can be
    ///        re-run from position 0 (attention ships no chunked prefill)
    std::vector<int> history;
    UniqueHostView cos_sliding, sin_sliding, cos_full, sin_full;

    /// \brief the mapped tensors a forward pass reads on the HOST
    /// \note Views into the GGUF mapping, which `package` keeps alive for the
    ///       engine's life. `per_layer_token_embd` is 2.35 billion Q8_0
    ///       parameters on E2B -- materializing it is not an option at any
    ///       point, so it stays mapped and is gathered per token.
    GgufTensor embedding_table, ple_table;
    GgufBf16Tensor ple_projection;
    GgufFloatTensor ple_projection_norm;

    /// \brief layer 0's own attention norm, the model's ONE standalone
    ///        RMSNorm
    /// \note Every other input norm is the previous layer's `ple` second
    ///       output (`norm_out = RMSNorm(out, next_norm)`), which crosses the
    ///       block boundary -- so nothing between two layers applies one, and
    ///       layer 0 has no block before it to produce its own.
    UniqueRmsNormWeights input_norm;
    /// \brief lm_head, TIED to token_embd and the ONE weight at head_group
    UniqueMatMulWeights lm_head;

    Impl(std::shared_ptr<Gemma4GgufPackage> package_in,
         std::shared_ptr<CorelibRuntime> runtime_in, const LM_Config&,
         std::optional<Gemma4Config> shape_override, std::uint32_t maximum)
        : runtime(std::move(runtime_in)), package(std::move(package_in)),
          api(runtime ? runtime->api() : nullptr), max_length(maximum) {
        if (!package) throw std::invalid_argument("Gemma 4 GGUF package is null");
        if (!runtime || !api) throw std::invalid_argument("corelib runtime is null");
        if (!ContextFits(maximum)) {
            throw std::invalid_argument(
                "Gemma 4 maximum length must be in 1..4096, or a multiple of 4096 up to " +
                std::to_string(kMaxPagedContext));
        }
        paged = maximum > static_cast<std::uint32_t>(kPageSize);
        cache_rows = paged ? maximum : static_cast<std::uint32_t>(kMaxSequenceLength);
        shape = shape_override ? std::move(*shape_override) : package->Config();
        if (paged && static_cast<std::int64_t>(maximum) > shape.context_length) {
            throw std::invalid_argument(
                "Gemma 4 maximum length " + std::to_string(maximum) +
                " is past this GGUF's context of " + std::to_string(shape.context_length));
        }

        const auto& cfg = shape;
        const auto& file = package->File();
        const auto layer_count = static_cast<std::size_t>(cfg.layers);
        layers.resize(layer_count);

        // ---- host-side material, all of it resolved before any thread runs
        //
        // Every span below is either a view into the mapped GGUF (which the
        // package keeps alive) or a vector that lives until the pool joins.
        // Resolving them up front is what makes the creates independent of
        // each other: a worker touches its own slot and nothing else.
        const std::array<float, 1> epsilon_value{cfg.eps};
        const auto epsilon = ToBf16(epsilon_value);

        // ORIENTATION, AND IT IS [OUT, IN]. `GgufFile` reports a tensor's
        // shape REVERSED from the GGUF's own dims order, and the file stores
        // dims fastest-varying (input) first -- so `ShapeOf` answers
        // [output, input] and every expected shape written here is in that
        // convention. `blk.0.attn_q.weight` reads [1536, 2048] in the file
        // and is demanded as {2048, 1536} below. It is the same convention
        // phi4_rai.cpp uses for `token_embd.weight` ({vocab, hidden}), which
        // is the comparison that settles it: Phi-4 has run against a real
        // GGUF on hardware.
        //
        // These were all written the other way round until Task R1, because
        // gemma4_gguf_fixture.hpp was transposed and this file was checked
        // against the fixture. Every one of them is now asserted against the
        // real gemma-4-E{2,4}B-it-Q8_0.gguf by test_gemma4_real_gguf.cpp.
        // If one is ever wrong again, RequireQ8/RequireF32 throw naming the
        // tensor and both shapes; nothing here can pick up a
        // differently-shaped tensor and pack it quietly.
        //
        // The two parameters are named `out` and `in` rather than
        // `rows`/`cols` deliberately: "rows" is the axis whose meaning the
        // whole defect turned on.
        const auto q8 = [&](const std::string& name, std::int64_t out,
                            std::int64_t in) {
            const std::array<std::int64_t, 2> want{out, in};
            return file.RequireQ8(name, want);
        };
        const auto f32_1d = [&](const std::string& name, std::int64_t k) {
            const std::array<std::int64_t, 1> want{k};
            return file.RequireF32(name, want);
        };
        const auto f32_2d = [&](const std::string& name, std::int64_t out,
                                std::int64_t in) {
            const std::array<std::int64_t, 2> want{out, in};
            return file.RequireF32(name, want);
        };

        const auto embedding = q8("token_embd.weight", cfg.vocab, cfg.hidden);

        // THE THREE MAPPED TENSORS A FORWARD PASS READS ON THE HOST, resolved
        // here so a malformed file is named at load rather than at the first
        // prompt. `per_layer_model_proj.weight` is the ONE BF16 tensor in a
        // Gemma 4 conversion; the other two are Q8_0 and F32.
        embedding_table = embedding;
        ple_table = q8("per_layer_token_embd.weight", cfg.vocab,
                       cfg.layers * cfg.ple_dim);
        {
            const std::array<std::int64_t, 2> want{cfg.layers * cfg.ple_dim, cfg.hidden};
            ple_projection = file.RequireBf16("per_layer_model_proj.weight", want);
        }
        ple_projection_norm = f32_1d("per_layer_proj_norm.weight", cfg.ple_dim);

        // `rope_freqs.weight`, ONE ENTRY PER ROTARY PAIR OF THE FULL HEAD.
        // [256] on both shipped rows, which is `global_head_dim / 2` and NOT a
        // 256-wide rotary -- `config.json`'s `partial_rotary_factor: 0.25` is
        // a per-pair frequency SCALING, not a shorter rotary, and
        // `gemma4.rope.dimension_count` (512) / `..._swa` (256) confirm full
        // rotary on both geometries. Demanded at that exact length here, so a
        // conversion that ships another one is named at load by RequireF32
        // rather than read past its end; MakeRopeTables checks the same thing
        // again against the head it is building for. The driver checks it too.
        const auto rope_factors =
            f32_1d("rope_freqs.weight", cfg.global_head_dim / 2);

        // BF16 copies of every gamma that goes into a packer. Held in vectors
        // rather than passed as mapped floats because both packers take BF16
        // and the file stores F32.
        std::vector<std::vector<std::uint16_t>> ffn_norm(layer_count),
            post_ffw_norm(layer_count), post_attn_norm(layer_count),
            q_norm(layer_count), k_norm(layer_count);
        auto input_norm_bf16 = ToBf16(f32_1d("blk.0.attn_norm.weight", cfg.hidden).values);
        // One per DISTINCT head size, not one per layer: a sliding layer's
        // head is 256 wide and a full-attention layer's 512, and `k` selects
        // the rmsnorm kernel.
        auto ones_sliding = OnesBf16(cfg.head_dim);
        auto ones_full = OnesBf16(cfg.global_head_dim);

        std::vector<GgufTensor> q_weight(layer_count), k_weight(layer_count),
            v_weight(layer_count), o_weight(layer_count), gate_weight(layer_count),
            up_weight(layer_count), down_weight(layer_count);
        std::vector<GgufFloatTensor> inp_gate(layer_count), proj(layer_count),
            ple_post_norm(layer_count), ple_next_norm(layer_count);
        std::vector<float> layer_scale(layer_count);

        for (std::size_t index = 0; index < layer_count; ++index) {
            const auto layer = static_cast<std::int64_t>(index);
            const bool owns = layer < cfg.kv_layers;
            const auto head_dim = cfg.LayerHeadDim(layer);
            const auto q_dim = cfg.q_heads * head_dim;
            const auto kv_dim = cfg.kv_heads * head_dim;
            const auto intermediate = cfg.layer_intermediate[index];

            q_weight[index] = q8(Blk(layer, ".attn_q.weight"), q_dim, cfg.hidden);
            // THE TWO PROJECTIONS WHOSE INPUT IS NOT `hidden` -- attn_output
            // takes q_dim in, ffn_down takes `intermediate`.
            o_weight[index] = q8(Blk(layer, ".attn_output.weight"), cfg.hidden, q_dim);
            gate_weight[index] = q8(Blk(layer, ".ffn_gate.weight"), intermediate, cfg.hidden);
            up_weight[index] = q8(Blk(layer, ".ffn_up.weight"), intermediate, cfg.hidden);
            down_weight[index] = q8(Blk(layer, ".ffn_down.weight"), cfg.hidden, intermediate);
            if (owns) {
                k_weight[index] = q8(Blk(layer, ".attn_k.weight"), kv_dim, cfg.hidden);
                v_weight[index] = q8(Blk(layer, ".attn_v.weight"), kv_dim, cfg.hidden);
                k_norm[index] = ToBf16(f32_1d(Blk(layer, ".attn_k_norm.weight"), head_dim).values);
            }
            q_norm[index] = ToBf16(f32_1d(Blk(layer, ".attn_q_norm.weight"), head_dim).values);
            post_attn_norm[index] =
                ToBf16(f32_1d(Blk(layer, ".post_attention_norm.weight"), cfg.hidden).values);
            // ssmlp's TWO NORMS ARE BOTH THIS LAYER'S. With
            // `activation = 1, post_feedforward_layernorm = 1` the descriptor
            // keys the `ssmlp_gemma_fusion_no_rms1_*` family, whose topology
            // is NOT the silu one: the trailing norm is dropped and `norm1`
            // becomes the post-feedforward gamma applied INSIDE the residual.
            // So norm0 is `ffn_norm` and norm1 is `post_ffw_norm` and NEITHER
            // belongs to another layer -- unlike the silu family, which is
            // what Phi-4 is. Carrying Phi-4's assignment over here is silent.
            // (gemma4_driver.py:1441-1449.)
            ffn_norm[index] = ToBf16(f32_1d(Blk(layer, ".ffn_norm.weight"), cfg.hidden).values);
            post_ffw_norm[index] =
                ToBf16(f32_1d(Blk(layer, ".post_ffw_norm.weight"), cfg.hidden).values);

            // ple's two matrices, as mapped floats.
            //
            // TWO DIFFERENT CONVENTIONS MEET HERE AND IT IS WORTH BEING
            // EXPLICIT. corelib takes `gate` as FP32 [k, n] row-major and
            // `proj` as FP32 [n, k] row-major -- "BOTH MATRICES ARE GIVEN IN
            // [IN, OUT] ORDER, which is the orientation a model stores and
            // the transpose of what the kernel wants; the transpose happens
            // here" (corelib.h, ple section). `GgufFile::ShapeOf`, meanwhile,
            // is [OUT, IN], so the same two tensors are demanded below as
            // {ple_dim, hidden} and {hidden, ple_dim}.
            //
            // The shapes are what stop the two being swapped: they have the
            // SAME element count on this model (1536 x 256 against
            // 256 x 1536), so no length check anywhere would notice, and
            // "passing one the other way round packs silently and produces
            // noise".
            //
            // THE MAPPING IS THE TRANSPOSE OF WHAT THE PACKER WANTS, and the
            // pack lambda below is where that is undone. `inp_gate` maps as
            // ple_dim contiguous rows of hidden and corelib wants hidden rows
            // of ple_dim; `proj` maps as hidden rows of ple_dim and corelib
            // wants ple_dim rows of hidden. Both are the transpose of the
            // other side, and the element counts are IDENTICAL, so nothing
            // rejects the wrong one -- "passing one the other way round packs
            // silently and produces noise". The reference driver does exactly
            // the same thing at exactly this point
            // (`gemma4_driver.py:1517`: `np.ascontiguousarray(floats(...))`,
            // where `floats()` is `.reshape(reversed(dims)).T`).
            //
            // These two views are the MAPPING; they are never handed to the
            // packer's filling leg unmodified.
            inp_gate[index] = f32_2d(Blk(layer, ".inp_gate.weight"), cfg.ple_dim, cfg.hidden);
            proj[index] = f32_2d(Blk(layer, ".proj.weight"), cfg.hidden, cfg.ple_dim);
            ple_post_norm[index] = f32_1d(Blk(layer, ".post_norm.weight"), cfg.hidden);
            // THE NEXT LAYER'S ATTENTION NORM, AND THE LAST LAYER'S IS THE
            // MODEL'S FINAL NORM. `ple` fuses [gate -> proj -> norm ->
            // residual -> scale -> norm] and that last norm crosses the block
            // boundary: it is the NEXT block's input norm. There is no
            // `blk.{layers}.attn_norm`, so the last layer takes
            // `output_norm.weight`. The driver's rule verbatim
            // (`_next_norm_name`), and the header's statement of the stakes:
            // getting it off by one "is not a crash -- every layer is then
            // normalized by its neighbour's gamma, which still generates
            // text."
            ple_next_norm[index] = f32_1d(
                layer == cfg.layers - 1 ? std::string("output_norm.weight")
                                        : Blk(layer + 1, ".attn_norm.weight"),
                cfg.hidden);
            // `layer_output_scale` is a ONE-ELEMENT tensor -- a norm's shape
            // without being one -- and is the (x + o) multiplier baked into
            // the blob.
            layer_scale[index] = f32_1d(Blk(layer, ".layer_output_scale.weight"), 1).values[0];
        }

        // ---- the stream, then the plan, then the weights
        //
        // In that order and not another: corelib 0.5.0 takes the stream on
        // every padding helper (the PDI pair a stream was opened with selects
        // the ELF set, and a shape exists under one pair and not another), so
        // the plan cannot exist before a stream does.
        auto lease = runtime->AcquireExecution();
        const int token_pdi = paged ? kPagedTokenPdi : kTokenPdi;
        stream = corelib::OpenModelStream(api, kPrefillPdi, token_pdi, "gemma4");
        plan = Gemma4ShapePlan::Build(api, stream.get(), cfg,
                                      paged ? static_cast<std::int64_t>(maximum)
                                            : kMaxSequenceLength);

        // ---- the slot table
        std::vector<WeightSlot> slots;
        slots.reserve(layer_count * 10 + 2);

        const auto add_matmul = [&](std::string label, std::int64_t k, std::int64_t n,
                                    std::uint32_t group, const GgufTensor* blocks,
                                    UniqueMatMulWeights* destination) {
            WeightSlot slot;
            slot.kind = SlotKind::Matmul;
            slot.label = std::move(label);
            slot.matmul = {k, n, group, /*has_bias=*/false};
            const auto desc = slot.matmul;
            const auto name = slot.label;
            slot.pack = [this, desc, name, blocks, destination] {
                const ryzenai_corelib_weights_source sources[]{corelib::GgufQ8(
                    ryzenai_corelib_weights_role_qweight, blocks->bytes.data(),
                    blocks->bytes.size(), desc.n, desc.k)};
                const auto options = corelib::PackOptions(kRequantizeThreads);
                void* created = nullptr;
                api->Check(api->functions().matmul_weights_create(
                               &desc, sources, std::size(sources), &options, &created),
                           "ryzenai_corelib_matmul_bf16_weights_create " + name);
                *destination = UniqueMatMulWeights(api, created);
            };
            slot.assign = [this, destination](void* handle) {
                *destination = UniqueMatMulWeights(api, handle);
            };
            slot.handle = [destination] { return destination->get(); };
            slots.push_back(std::move(slot));
        };

        const auto add_rmsnorm = [&](std::string label, std::int64_t k,
                                     const std::vector<std::uint16_t>* gamma,
                                     UniqueRmsNormWeights* destination) {
            WeightSlot slot;
            slot.kind = SlotKind::RmsNorm;
            slot.label = std::move(label);
            // `epsilon` travels in the DESCRIPTOR here, not per dispatch: it
            // is baked into the packed blob. There is no group size and no
            // layout to choose -- the scale is BF16 and unquantized -- and
            // `k` must match a shipped artifact exactly, because this op
            // tiles the ROW count and cannot pad k.
            slot.rmsnorm = {k, cfg.eps};
            const auto desc = slot.rmsnorm;
            const auto name = slot.label;
            slot.pack = [this, desc, name, gamma, destination] {
                const ryzenai_corelib_weights_source sources[]{corelib::Bf16(
                    ryzenai_corelib_weights_role_scale, gamma->data(), gamma->size())};
                void* created = nullptr;
                api->Check(api->functions().rmsnorm_weights_create(
                               &desc, sources, std::size(sources), nullptr, &created),
                           "ryzenai_corelib_rmsnorm_bf16_weights_create " + name);
                *destination = UniqueRmsNormWeights(api, created);
            };
            slot.assign = [this, destination](void* handle) {
                *destination = UniqueRmsNormWeights(api, handle);
            };
            slot.handle = [destination] { return destination->get(); };
            slots.push_back(std::move(slot));
        };

        // LAYER 0'S STANDALONE NORM COMES FIRST, matching the reference
        // driver's own order (`_load_layers` packs it before the layers).
        add_rmsnorm("blk.0.attn_norm.weight", cfg.hidden, &input_norm_bf16, &input_norm);

        for (std::size_t index = 0; index < layer_count; ++index) {
            const auto layer = static_cast<std::int64_t>(index);
            const bool owns = layer < cfg.kv_layers;
            const auto head_dim = cfg.LayerHeadDim(layer);
            const auto q_dim = cfg.q_heads * head_dim;
            const auto kv_dim = cfg.kv_heads * head_dim;
            const auto intermediate = cfg.layer_intermediate[index];
            auto& weights = layers[index];

            // EVERY PROJECTION IS `cfg.group`. Not a judgement call: the
            // driver declares the field as "group: int  # every projection",
            // `_matmul_weights` defaults `group_size=self.cfg.group`, and the
            // ONLY call in the whole driver that overrides it is lm_head.
            // `cfg.head_group` is named for lm_head, NOT for the attention
            // heads -- using it here would requantize Q/K/V/O at the wrong
            // group, which is not a padding question but a wrong weight. It
            // happens to be the same number on both shipped rows, which is
            // exactly why it has to be written down.
            add_matmul(Blk(layer, ".attn_q.weight"), cfg.hidden, q_dim, cfg.group,
                       &q_weight[index], &weights.q);
            if (owns) {
                add_matmul(Blk(layer, ".attn_k.weight"), cfg.hidden, kv_dim, cfg.group,
                           &k_weight[index], &weights.k);
                add_matmul(Blk(layer, ".attn_v.weight"), cfg.hidden, kv_dim, cfg.group,
                           &v_weight[index], &weights.v);
            }
            add_rmsnorm(Blk(layer, ".attn_q_norm.weight"), head_dim, &q_norm[index],
                        &weights.q_norm);
            if (owns) {
                add_rmsnorm(Blk(layer, ".attn_k_norm.weight"), head_dim, &k_norm[index],
                            &weights.k_norm);
                // V's norm, which has NO learned weight in the file.
                add_rmsnorm(Blk(layer, ".attn_v_norm(ones)"), head_dim,
                            head_dim == cfg.head_dim ? &ones_sliding : &ones_full,
                            &weights.v_norm);
            }
            add_matmul(Blk(layer, ".attn_output.weight"), q_dim, cfg.hidden, cfg.group,
                       &o_weight[index], &weights.o);
            add_rmsnorm(Blk(layer, ".post_attention_norm.weight"), cfg.hidden,
                        &post_attn_norm[index], &weights.post_attn_norm);

            {
                WeightSlot slot;
                slot.kind = SlotKind::SsMlp;
                slot.label = Blk(layer, ".ssmlp");
                slot.layer = layer;
                slot.ssmlp = {cfg.hidden, intermediate, cfg.group,
                              /*activation=*/1, /*post_feedforward_layernorm=*/1};
                const auto desc = slot.ssmlp;
                const auto name = slot.label;
                const auto* norm0 = &ffn_norm[index];
                const auto* norm1 = &post_ffw_norm[index];
                const auto* gate = &gate_weight[index];
                const auto* up = &up_weight[index];
                const auto* down = &down_weight[index];
                const auto* eps = &epsilon;
                auto* destination = &weights.mlp;
                slot.pack = [this, desc, name, norm0, norm1, gate, up, down, eps,
                             destination] {
                    const ryzenai_corelib_weights_source sources[]{
                        corelib::Bf16(ryzenai_corelib_weights_role_epsilon, eps->data(), eps->size()),
                        corelib::Bf16(ryzenai_corelib_weights_role_norm0, norm0->data(), norm0->size()),
                        corelib::Bf16(ryzenai_corelib_weights_role_norm1, norm1->data(), norm1->size()),
                        corelib::GgufQ8(ryzenai_corelib_weights_role_gate_qweight,
                                        gate->bytes.data(), gate->bytes.size(), desc.n, desc.k),
                        corelib::GgufQ8(ryzenai_corelib_weights_role_up_qweight,
                                        up->bytes.data(), up->bytes.size(), desc.n, desc.k),
                        corelib::GgufQ8(ryzenai_corelib_weights_role_down_qweight,
                                        down->bytes.data(), down->bytes.size(), desc.k, desc.n)};
                    const auto options = corelib::PackOptions(kRequantizeThreads);
                    void* created = nullptr;
                    api->Check(api->functions().ssmlp_weights_create(
                                   &desc, sources, std::size(sources), &options, &created),
                               "ryzenai_corelib_ssmlp_bf16_weights_create " + name);
                    *destination = UniqueSsMlpWeights(api, created);
                };
                slot.assign = [this, destination](void* handle) {
                    *destination = UniqueSsMlpWeights(api, handle);
                };
                slot.handle = [destination] { return destination->get(); };
                slots.push_back(std::move(slot));
            }

            {
                WeightSlot slot;
                slot.kind = SlotKind::Ple;
                slot.label = Blk(layer, ".ple");
                slot.layer = layer;
                // `group_size` here is `ple_group`, NOT `group`: 32 is the
                // only value mladfple ships and corelib refuses anything else
                // by name. `epsilon` is ONE value added under the rsqrt of
                // BOTH RMSNorms, not one per norm.
                slot.ple = {cfg.hidden, cfg.ple_dim, cfg.ple_group, cfg.eps,
                            layer_scale[index]};
                const auto desc = slot.ple;
                const auto name = slot.label;
                const auto* gate = &inp_gate[index];
                const auto* projection = &proj[index];
                const auto* post_norm = &ple_post_norm[index];
                const auto* next_norm = &ple_next_norm[index];
                auto* destination = &weights.ple;
                slot.pack = [this, desc, name, gate, projection, post_norm, next_norm,
                             destination] {
                    // corelib takes gate as FP32 [k, n] and proj as [n, k].
                    // The GGUF mapping is the other way round, so transpose
                    // first. The packer allocates its own blob.
                    const auto gate_rows = Transposed(gate->values, desc.n, desc.k);
                    const auto proj_rows = Transposed(projection->values, desc.k, desc.n);
                    const ryzenai_corelib_weights_source sources[]{
                        corelib::Fp32(ryzenai_corelib_weights_role_gate, gate_rows.data(),
                                      gate_rows.size(), {desc.k, desc.n}),
                        corelib::Fp32(ryzenai_corelib_weights_role_proj, proj_rows.data(),
                                      proj_rows.size(), {desc.n, desc.k}),
                        corelib::Fp32(ryzenai_corelib_weights_role_post_norm,
                                      post_norm->values.data(), post_norm->values.size(),
                                      {desc.k}),
                        corelib::Fp32(ryzenai_corelib_weights_role_next_norm,
                                      next_norm->values.data(), next_norm->values.size(),
                                      {desc.k})};
                    void* created = nullptr;
                    api->Check(api->functions().ple_weights_create(
                                   &desc, sources, std::size(sources), nullptr, &created),
                               "ryzenai_corelib_ple_bf16_weights_create " + name);
                    *destination = UniquePleWeights(api, created);
                };
                slot.assign = [this, destination](void* handle) {
                    *destination = UniquePleWeights(api, handle);
                };
                slot.handle = [destination] { return destination->get(); };
                slots.push_back(std::move(slot));
            }
        }

        // LM_HEAD, LAST, AND THE ONE WEIGHT AT `head_group`. TIED to
        // token_embd -- this conversion has no `output.weight` on either row
        // -- and "the only weight packed at cfg.head_group"
        // (gemma4_driver.py:1691-1694, :2449-2453).
        add_matmul("token_embd.weight", cfg.hidden, cfg.vocab, cfg.head_group,
                   &embedding, &lm_head);
        slot_count = slots.size();

        // A miss, a stale key, an unreadable index or a failed bind all fall
        // back to packing.
        const auto cache = WeightCache::ForGguf(package->Path());
        const auto version = api->runtime_version();
        const auto cache_key = MakeWeightCacheKey(
            package->Path(), version.major, version.minor, version.patch,
            CacheLayout(cfg), slots.size());
        if (cache) {
            if (const auto spans = cache->ReadIndex(cache_key))
                from_cache = BindFromCache(cache->DataPath(), *spans, slots);
        }

        if (!from_cache) {
            std::cout << "[FLM]  Packing " << slots.size() << " Gemma 4 " << cfg.row_name
                      << " weights from Q8_0" << (cache ? "; cached for the next load" : "")
                      << std::endl;
            PackAll(slots);
        }

        if (cache && !from_cache) {
            // Whatever is there did not match, or it would have been used.
            (void)cache->Remove();
            WriteCache(*cache, cache_key, slots);
        }

        AllocateBuffers();
        AllocateRotaryTables(rope_factors);
    }

    /// \brief every activation buffer, the head's two, the per-layer embedding
    ///        planes and the KV caches -- once, at load
    /// \note See the member declarations for why each is shaped as it is. The
    ///       order here is the order the shapes are asserted in.
    void AllocateBuffers() {
        const auto& cfg = shape;
        const auto rows = kMaxSequenceLength;
        const auto widest = std::max(cfg.head_dim, cfg.global_head_dim);
        const auto q_dim = cfg.q_heads * widest;
        const auto kv_dim = cfg.kv_heads * widest;
        const auto bf16 = [&](std::vector<std::int64_t> dims, const char* label) {
            return Tensor(ryzenai_corelib_data_type_bf16, std::move(dims), label);
        };

        hidden = bf16({rows, cfg.hidden}, "hidden");
        residual = bf16({rows, cfg.hidden}, "residual");
        skip_sum = bf16({rows, cfg.hidden}, "skip_sum");
        ple_out = bf16({rows, cfg.hidden}, "ple_out");
        q_buf = bf16({rows, q_dim}, "q");
        k_buf = bf16({rows, kv_dim}, "k");
        v_buf = bf16({rows, kv_dim}, "v");
        q_normed = bf16({rows, q_dim}, "q normed");
        k_normed = bf16({rows, kv_dim}, "k normed");
        attn_out = bf16({rows, q_dim}, "attention");
        last_hidden = bf16({1, cfg.hidden}, "lm head input");
        logits = bf16({1, cfg.vocab}, "logits");

        ple_in.reserve(static_cast<std::size_t>(cfg.layers));
        for (std::int64_t layer = 0; layer < cfg.layers; ++layer)
            ple_in.push_back(bf16({rows, cfg.ple_dim}, "per-layer embedding plane"));

        k_cache.reserve(static_cast<std::size_t>(cfg.kv_layers));
        v_cache.reserve(static_cast<std::size_t>(cfg.kv_layers));
        for (std::int64_t cache = 0; cache < cfg.kv_layers; ++cache) {
            // THE WIDTH IS THE OWNING LAYER'S OWN HEAD. A layer below
            // `kv_layers` owns cache `index`, so this is layer `cache`'s head.
            const auto head = cfg.LayerHeadDim(cache);
            if (paged) {
                const auto pages = static_cast<std::int64_t>(cache_rows) / kPageSize;
                k_cache.push_back(bf16({pages, cfg.kv_heads, kPageSize, head}, "K cache"));
                v_cache.push_back(bf16({pages, cfg.kv_heads, kPageSize, head}, "V cache"));
            } else {
                k_cache.push_back(bf16({cfg.kv_heads, rows, head}, "K cache"));
                v_cache.push_back(bf16({cfg.kv_heads, rows, head}, "V cache"));
            }
        }
    }

    /// \brief create one device tensor, naming it in any failure
    UniqueTensor Tensor(ryzenai_corelib_data_type type,
                        std::vector<std::int64_t> dims, const char* label) {
        void* created = nullptr;
        api->Check(api->functions().create_device_tensor(type, dims.data(), dims.size(),
                                                         &created),
                   std::string("ryzenai_corelib_create_device_tensor ") + label);
        return UniqueTensor(api, created);
    }

    /// \brief both rotary pairs, built once and never written again
    /// \param factors `rope_freqs.weight`, the FULL-ATTENTION regime's
    ///        per-pair frequency divisors
    ///
    /// SLIDING LAYERS GET NO FACTORS AND FULL-ATTENTION LAYERS DO, which is
    /// llama.cpp's own split: `freq_factors` is passed on full-attention
    /// layers only. Applying them to both regimes, or to neither, produces a
    /// perfectly ordinary rotary table that is wrong from position 1 on and
    /// still generates fluent text.
    ///
    /// Sized at `kMaxSequenceLength` rather than at the prompt: corelib
    /// requires both tables to "reach at least `position + rows` rows", and
    /// they are indexed by ABSOLUTE position, so the table a decode step at
    /// position 4000 needs is the same table prefill wrote.
    void AllocateRotaryTables(const GgufFloatTensor& factors) {
        const auto& cfg = shape;
        const auto view = [&](UniqueHostView& destination, std::int64_t head_dim,
                              const std::vector<float>& values, const char* label) {
            const std::array<std::int64_t, 2> dims{static_cast<std::int64_t>(cache_rows),
                                                   head_dim / 2};
            void* created = nullptr;
            api->Check(api->functions().create_host_view(
                           ryzenai_corelib_data_type_fp32, dims.data(), dims.size(),
                           values.data(), &created),
                       std::string("ryzenai_corelib_create_host_view ") + label);
            destination = UniqueHostView(api, created);
        };
        // Base `rope_theta_swa` (1e4) over the sliding head, NO factors.
        const auto rope_rows = static_cast<std::int64_t>(cache_rows);
        auto sliding = MakeRopeTables(cfg.head_dim, cfg.rope_theta_swa, rope_rows);
        rope_cos_sliding = std::move(sliding.cos);
        rope_sin_sliding = std::move(sliding.sin);
        view(cos_sliding, cfg.head_dim, rope_cos_sliding, "sliding cos");
        view(sin_sliding, cfg.head_dim, rope_sin_sliding, "sliding sin");
        // Base `rope_theta` (1e6) over the full head, WITH `rope_freqs.weight`
        // as a per-pair frequency DIVISOR.
        auto full = MakeRopeTables(cfg.global_head_dim, cfg.rope_theta, rope_rows,
                                   factors.values);
        rope_cos_full = std::move(full.cos);
        rope_sin_full = std::move(full.sin);
        view(cos_full, cfg.global_head_dim, rope_cos_full, "full cos");
        view(sin_full, cfg.global_head_dim, rope_sin_full, "full sin");
    }

    /// \brief run every slot's `pack()` across a small pool
    /// \note TWO FORMS OF PARALLELISM THAT MULTIPLY. Each create is
    ///       submitted at `kRequantizeThreads` (0 -- corelib picks, 8 packer
    ///       threads at 0.25.0) while `kWeightCreateConcurrency` of them run
    ///       at once, so up to the product runs together. The reference
    ///       driver packs strictly serially and says why (an open,
    ///       unattributed corelib defect correlating concurrent creates with
    ///       all-zero device output); this follows Phi-4's shape instead,
    ///       which is what this tree already ships and runs.
    void PackAll(std::vector<WeightSlot>& slots) {
        std::atomic<std::size_t> next{0};
        std::mutex failure_mutex;
        std::exception_ptr first_failure;
        const auto run = [&] {
            for (auto index = next++; index < slots.size(); index = next++) {
                try {
                    slots[index].pack();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!first_failure) first_failure = std::current_exception();
                    // Stop the others: the first diagnostic is the useful
                    // one, and a package that fails one create fails the load.
                    next = slots.size();
                    return;
                }
            }
        };
        const auto workers = std::min(kWeightCreateConcurrency, slots.size());
        std::vector<std::thread> pool;
        pool.reserve(workers > 0 ? workers - 1 : 0);
        for (std::size_t worker = 1; worker < workers; ++worker) pool.emplace_back(run);
        run();
        for (auto& worker : pool) worker.join();
        if (first_failure) std::rethrow_exception(first_failure);
    }

    /// \brief bind every weight from the cache file instead of packing
    /// \return true when ALL of them bound; false leaves nothing bound
    /// \note A PARTIAL BIND RELEASES EVERYTHING. A half-cached model is not a
    ///       state worth having and the caller simply packs. corelib rejects
    ///       a slice whose length is not exactly what the descriptor packs
    ///       to, so a stale file is caught here rather than becoming
    ///       confident nonsense.
    bool BindFromCache(const std::filesystem::path& data_path,
                       const std::vector<CachedWeightSpan>& spans,
                       std::vector<WeightSlot>& slots) {
        if (spans.size() != slots.size()) return false;
        const auto path = data_path.string();
        for (std::size_t index = 0; index < slots.size(); ++index) {
            auto& slot = slots[index];
            void* handle = nullptr;
            auto status = ryzenai_corelib_status_failure;
            const auto packed =
                corelib::PackedFile(path.c_str(), spans[index].offset, spans[index].size);
            switch (slot.kind) {
                case SlotKind::Matmul:
                    status = api->functions().matmul_weights_create(
                        &slot.matmul, &packed, 1, nullptr, &handle);
                    break;
                case SlotKind::SsMlp:
                    status = api->functions().ssmlp_weights_create(
                        &slot.ssmlp, &packed, 1, nullptr, &handle);
                    break;
                case SlotKind::RmsNorm:
                    status = api->functions().rmsnorm_weights_create(
                        &slot.rmsnorm, &packed, 1, nullptr, &handle);
                    break;
                case SlotKind::Ple:
                    status = api->functions().ple_weights_create(
                        &slot.ple, &packed, 1, nullptr, &handle);
                    break;
            }
            if (status != ryzenai_corelib_status_success || handle == nullptr) {
                ReleaseAllWeights();
                return false;
            }
            slot.assign(handle);
        }
        return true;
    }

    void ReleaseAllWeights() {
        for (auto& layer : layers) layer = LayerWeights{};
        input_norm = {};
        lm_head = {};
    }

    /// \brief copy the packed bytes out and write them beside the model
    /// \note Best effort: a cache that cannot be written is not a load
    ///       failure, it only means the next launch packs again. The index is
    ///       written last, so a data file without a matching index is never
    ///       used.
    void WriteCache(const WeightCache& cache, const WeightCacheKey& key,
                    const std::vector<WeightSlot>& slots) {
        try {
            std::error_code error;
            const auto data_path = cache.DataPath();
            std::filesystem::create_directories(data_path.parent_path(), error);
            const auto temporary = data_path.string() + ".tmp";
            std::vector<CachedWeightSpan> spans;
            spans.reserve(slots.size());
            {
                std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
                if (!output) return;
                std::vector<char> buffer;
                std::uint64_t offset = 0;
                for (const auto& slot : slots) {
                    std::size_t size = 0;
                    if (api->functions().weights_copy_data(slot.handle(), nullptr, 0, &size) !=
                            ryzenai_corelib_status_success ||
                        size == 0)
                        return;
                    buffer.resize(size);
                    if (api->functions().weights_copy_data(
                            slot.handle(), buffer.data(), buffer.size(), &size) !=
                        ryzenai_corelib_status_success)
                        return;
                    if (size == 0) return;
                    output.write(buffer.data(), static_cast<std::streamsize>(size));
                    if (!output) return;
                    spans.push_back({offset, static_cast<std::uint64_t>(size)});
                    offset += size;
                }
            }
            std::filesystem::rename(temporary, data_path, error);
            if (error) { std::filesystem::remove(temporary, error); return; }
            if (!cache.WriteIndex(key, spans)) std::filesystem::remove(data_path, error);
        } catch (...) {
            // Caching is an optimisation; never let it fail a load.
        }
    }

    void usable() const {
        if (poisoned) throw std::runtime_error("Gemma 4 corelib engine is poisoned");
    }

    // ---- the forward pass -------------------------------------------------

    /// \brief create a window onto `parent`, naming it in any failure
    UniqueTensorWindow Window(void* parent, std::vector<std::int64_t> dims,
                              std::size_t offset, const char* label,
                              std::vector<std::int64_t> strides = {}) {
        void* created = nullptr;
        api->Check(api->functions().create_tensor_window(parent, dims.data(), dims.size(), offset,
                                                         strides.empty() ? nullptr : strides.data(),
                                                         strides.size(), &created),
                   std::string("ryzenai_corelib_create_tensor_window ") + label);
        return UniqueTensorWindow(api, created);
    }

    /// \brief the row count the kernels will actually run for `rows` live ones
    ///
    /// THE GRID STARTS AT 128 BECAUSE PRODUCTION'S DOES. A four-token prompt
    /// runs a 128-row kernel here where every other driver in this tree would
    /// run a 64-row one. That is not a mitigation for a broken artifact: 128
    /// is the floor hybrid-llm's full-fusion path supports at all, and below
    /// it upstream's own comment says a native buffer wedges the NPU for LLM
    /// matmul shapes. The reference driver's `PREFILL_GRID` is this table and
    /// `_pad_rows` is this function. The 124 rows of zeros are what staying on
    /// a supported shape costs.
    ///
    /// DECODE IS NOT PADDED: an M == 1 kernel ships for every operator here.
    std::int64_t PaddedRowsFor(std::int64_t rows) const {
        if (rows <= 1) return 1;
        // A paged chunk past page 0 is only correct at a full page.
        if (paged && position > 0 && rows <= kPageSize) return kPageSize;
        for (const std::int64_t bucket : {128, 256, 512, 1024, 2048, 3072, 4096}) {
            if (rows <= bucket) return bucket;
        }
        throw std::out_of_range("Gemma 4 cannot run " + std::to_string(rows) +
                                " rows: the largest shipped prefill kernel is 4096");
    }

    /// \brief byte-element offset of one KV row inside its cache
    std::size_t CacheOffset(std::int64_t kv_head, std::int64_t index,
                            std::int64_t head) const {
        if (!paged) {
            return static_cast<std::size_t>((kv_head * kMaxSequenceLength + index) * head);
        }
        const auto page = index / kPageSize;
        const auto row = index % kPageSize;
        return static_cast<std::size_t>(
            ((page * shape.kv_heads + kv_head) * kPageSize + row) * head);
    }

    /// \brief every operator agrees that `padded` is the extent it runs at
    ///
    /// ONE ROW EXTENT FOR THE WHOLE PASS, AND IT HAS TO BE CHECKED RATHER THAN
    /// ASSUMED. Every window this pass builds declares `padded` rows, and each
    /// operand "is checked against its OWN extent" by corelib -- so if any
    /// operator's own pad helper would round `padded` further, that operator
    /// reads and writes rows nobody allocated for it. The plan already asked
    /// corelib, for every bucket and both geometries, at load; this compares
    /// the answers.
    ///
    /// CHECKED ON THE HOST, BEFORE THE FIRST DISPATCH, for the reason every
    /// other check in `Run` is: a rejection part-way through an unbarriered
    /// chain aborts the process rather than raising.
    ///
    /// WHAT THIS DOES **NOT** COVER, AND THE REFERENCE DRIVER DOES.
    /// `gemma4_driver.py` carries a separate `_norm_rows` guard asserting that
    /// each RMSNorm runs at a row count an EXACT kernel ships for. It exists
    /// because an unshipped M does not fail -- it OVERSHOOTS, and for V's norm
    /// the destination is the KV CACHE, so the overshoot lands in another
    /// layer's cached values and comes back as attention over data no layer
    /// wrote. Nothing here reproduces that guard.
    ///
    /// It is latent today ONLY because
    /// `ryzenai_corelib_rmsnorm_bf16_pad_rows` is an identity function that
    /// does not report the covering extent (corelib.h says so outright) and is
    /// therefore not in `FLM_CORELIB_FUNCTIONS` at all -- so the rmsnorm rows
    /// this engine uses are `ple`'s bucket, which is a shipped one. That is a
    /// reason that can stop being true the day corelib implements the helper,
    /// and the failure it would unmask is silent. Carried as a pre-hardware
    /// debt item; see the SDD ledger's pre-hardware debt list.
    void RequireOneRowExtent(std::int64_t padded) const {
        const auto for_geometry = [&](bool swa) {
            const auto& kinds = shape.layer_is_swa;
            if (std::find(kinds.begin(), kinds.end(), swa) == kinds.end()) return;
            const auto& extents = plan.ForRows(static_cast<std::size_t>(padded), swa);
            const std::array<std::pair<const char*, std::int64_t>, 6> named{{
                {"query", extents.q_rows},
                {"key/value", extents.kv_rows},
                {"output", extents.output_rows},
                {"ssmlp", extents.ssmlp_rows},
                {"ple", extents.ple_rows},
                {"attention", extents.flat_mha_rows},
            }};
            for (const auto& [what, value] : named) {
                if (value == padded) continue;
                throw std::runtime_error(
                    std::string("Gemma 4 would run ") + std::to_string(padded) +
                    " rows, but corelib pads " + what + " to " + std::to_string(value) +
                    " on the " + (swa ? "sliding" : "full-attention") +
                    " geometry -- every buffer in this pass is windowed at " +
                    std::to_string(padded) + " rows and that op would run past it");
            }
        };
        for_geometry(true);
        for_geometry(false);
    }

    /// \brief the windows one geometry's layers dispatch through
    /// \note THE PER-HEAD VIEWS ARE WINDOWS OF THE SAME MEMORY, not copies.
    ///       Reading `[rows, heads * head]` as `[rows * heads, head]` IS the
    ///       per-head layout in row-major, because a token's heads are
    ///       adjacent -- which is why Q and K need no staging buffer and V
    ///       does: the CACHE is the other way round.
    struct Geometry {
        std::int64_t head{};
        UniqueTensorWindow q, k, v_dest;
        UniqueTensorWindow q_heads_in, k_heads_in, q_heads_out, k_heads_out;
        std::vector<UniqueTensorWindow> v_src;
        UniqueTensorWindow q_attn, k_attn, attn;
    };

    Geometry BuildGeometry(bool swa, std::int64_t padded) {
        const auto& cfg = shape;
        Geometry geometry;
        const auto head = swa ? cfg.head_dim : cfg.global_head_dim;
        geometry.head = head;
        const auto q_dim = cfg.q_heads * head;
        const auto kv_dim = cfg.kv_heads * head;
        geometry.q = Window(q_buf.get(), {padded, q_dim}, 0, "q");
        geometry.k = Window(k_buf.get(), {padded, kv_dim}, 0, "k");
        // WHERE V'S PROJECTION LANDS, AND IT IS HEAD-MAJOR BECAUSE THE CACHE
        // IS. A cache is [kv_heads, max_seq, head]; a projection's ordinary
        // output is [rows, kv_heads * head], token-major. Those are the same
        // bytes only when there is ONE KV head. With more, one head's rows sit
        // `kv_heads * head` apart in the token-major output, while V's norm
        // below reads each head as a dense [rows, head] plane at an offset --
        // so the staging buffer has to be written head-major in the first
        // place. Only the scattering kernel can do that, and a 3-D output, a
        // window strided [max_seq * head, head, 1], is what asks for it.
        //
        // AT EXACTLY ONE KV HEAD A PLAIN 2-D OUTPUT IS USED INSTEAD, and that
        // is not merely an optimization: with one head there is nothing to
        // scatter and the dense kernel writes byte-for-byte what the
        // scattering one would, while the gm artifact at one head and head
        // size 512 TAKES THE DEVICE DOWN (`gemma4_driver.py`'s
        // GM_AT_ONE_KV_HEAD). The rule is `cfg.kv_heads`, not a model.
        geometry.v_dest =
            cfg.kv_heads == 1
                ? Window(v_buf.get(), {padded, head}, 0, "v")
                : Window(v_buf.get(), {cfg.kv_heads, kMaxSequenceLength, head}, 0, "v",
                         {kMaxSequenceLength * head, head, 1});
        // QK-Norm normalizes ONE HEAD, so its row count is tokens x heads.
        geometry.q_heads_in =
            Window(q_buf.get(), {padded * cfg.q_heads, head}, 0, "q heads");
        geometry.k_heads_in =
            Window(k_buf.get(), {padded * cfg.kv_heads, head}, 0, "k heads");
        geometry.q_heads_out =
            Window(q_normed.get(), {padded * cfg.q_heads, head}, 0, "q heads normed");
        geometry.k_heads_out =
            Window(k_normed.get(), {padded * cfg.kv_heads, head}, 0, "k heads normed");
        // V's norm reads one head's plane at a time. The staging buffer
        // carries `max_seq` rows per head, exactly as the cache does, so head
        // `h` starts at the same offset in both.
        geometry.v_src.reserve(static_cast<std::size_t>(cfg.kv_heads));
        for (std::int64_t kv_head = 0; kv_head < cfg.kv_heads; ++kv_head) {
            geometry.v_src.push_back(Window(
                v_buf.get(), {padded, head},
                static_cast<std::size_t>(kv_head * kMaxSequenceLength * head),
                "v head"));
        }
        geometry.q_attn = Window(q_normed.get(), {padded, q_dim}, 0, "q attended");
        geometry.k_attn = Window(k_normed.get(), {padded, kv_dim}, 0, "k attended");
        geometry.attn = Window(attn_out.get(), {padded, q_dim}, 0, "attention");
        return geometry;
    }

    /// \brief one token's embedding row, scaled by `sqrt(hidden)`
    ///
    /// THE SCALE IS PART OF THE EMBEDDING, as it is in llama.cpp:
    /// `ggml_scale(inpL, sqrtf(n_embd))` runs immediately after the gather and
    /// before anything reads it. So the per-layer inputs must NOT scale again;
    /// a second `sqrt(1536)` there multiplies the projection by 39 and still
    /// produces finite, fluent-looking output.
    ///
    /// `token_embd.weight`'s file dims are `[hidden, vocab]` with the input
    /// width fastest-varying, so the memory is `vocab` rows of `hidden` and
    /// one token's row is contiguous -- the same layout question
    /// `GatherPerLayerEmbedding` answers for the per-layer table.
    std::vector<float> ScaledEmbeddingRows(std::span<const int> ids) const {
        const auto& cfg = shape;
        constexpr std::int64_t kBlockElements = 32;
        constexpr std::size_t kBlockBytes = 34;
        if (cfg.hidden % kBlockElements != 0) {
            throw std::runtime_error("hidden is " + std::to_string(cfg.hidden) +
                                     ", which is not a multiple of the Q8_0 block size");
        }
        const auto row_bytes =
            static_cast<std::size_t>(cfg.hidden / kBlockElements) * kBlockBytes;
        if (embedding_table.bytes.size() <
            static_cast<std::size_t>(cfg.vocab) * row_bytes) {
            throw std::runtime_error("token_embd.weight is shorter than " +
                                     std::to_string(cfg.vocab) + " rows of " +
                                     std::to_string(cfg.hidden));
        }
        const auto scale = static_cast<float>(std::sqrt(static_cast<double>(cfg.hidden)));
        std::vector<float> rows(ids.size() * static_cast<std::size_t>(cfg.hidden));
        for (std::size_t row = 0; row < ids.size(); ++row) {
            const auto span = std::span<float>(rows).subspan(
                row * static_cast<std::size_t>(cfg.hidden),
                static_cast<std::size_t>(cfg.hidden));
            DecodeQ8Row(embedding_table.bytes.subspan(
                            static_cast<std::size_t>(ids[row]) * row_bytes, row_bytes),
                        span);
            for (float& value : span) value *= scale;
        }
        return rows;
    }

    /// \brief one pass over every layer, then the head. Returns SOFTCAPPED
    ///        logits for the LAST position.
    ///
    /// NO LAYER BOUNDARY RETURNS TO THE HOST, which is the whole point of
    /// `ple_bf16`: its SECOND output is the next layer's input norm, so the
    /// chain closes on the device and the 306 dispatches of an E2B prefill
    /// carry nothing back here.
    ///
    /// ONE WAIT IN THE LOOP, and it is after the loop. There is NO BARRIER AT
    /// A ROTARY REGIME CHANGE: corelib holds one derived table per rotary key
    /// per synchronize interval and never rewrites one the interval has
    /// already bound, so "no barrier is needed at a change of base". The
    /// reference driver removed exactly that barrier and records what it cost:
    /// 13 extra round trips on an E2B.
    ///
    /// EVERY ARGUMENT IS CHECKED BEFORE THE FIRST DISPATCH, AND THAT IS A RULE
    /// OF THE WHOLE DISPATCH MODEL RATHER THAN A HOUSE STYLE. Any op rejection
    /// part-way through an unbarriered chain is PROCESS-FATAL, whatever the op
    /// and whatever the reason: the exception unwinds out of here without
    /// reaching `stream_synchronize`, so the runs already submitted are
    /// destructed while their commands are still in flight and XRT aborts --
    /// "xrt::run destructed while command is still in progress" -- with no
    /// catchable failure anywhere. There is no try/finally that fixes it
    /// either: synchronizing during the unwind waits on work whose operands
    /// are already being torn down. So anything this pass can refuse, it
    /// refuses HERE, before the stream has been handed anything.
    buffer<bf16> Run(std::span<const int> ids) {
        usable();
        const auto& cfg = shape;
        if (ids.empty())
            throw std::invalid_argument("Gemma 4 request contains no token IDs");
        for (const int id : ids) {
            if (id < 0 || id >= cfg.vocab) {
                throw std::out_of_range(
                    "Gemma 4 token id " + std::to_string(id) +
                    " is outside the vocabulary of " + std::to_string(cfg.vocab));
            }
        }
        const auto rows = static_cast<std::int64_t>(ids.size());
        if (rows > static_cast<std::int64_t>(max_length) ||
            position + rows > static_cast<std::int64_t>(max_length)) {
            throw std::out_of_range("Gemma 4 request exceeds configured context capacity");
        }
        const auto padded = PaddedRowsFor(rows);
        // The kernels run `padded` rows starting at `position`, and V's norm
        // writes that many cache rows -- so the bound is on `position +
        // padded`, not on `position`. Reported here, where the numbers are
        // still the caller's, rather than as a byte offset from inside a
        // window.
        //
        // UNREACHABLE TODAY, AND SAID SO RATHER THAN LEFT TO LOOK LIVE. The
        // check above bounds `position + rows` by `max_length`, which the
        // constructor bounds by `kMaxSequenceLength`; `padded` exceeds `rows`
        // only when `rows > 1`, which the chunked-prefill rule below pins to
        // `position == 0`, and `PaddedRowsFor` refuses anything over 4096. So
        // no argument reaches this branch while `max_length <=
        // kMaxSequenceLength` holds, and no test in this suite can reach it.
        // It is kept because it guards the CACHE and the one above guards the
        // CONFIGURATION: they are independent facts that happen to coincide,
        // the reference driver has this one and not the other, and a future
        // caller that decouples them would otherwise lose the only check that
        // is about the thing the kernel actually addresses.
        const auto cache_limit = static_cast<std::int64_t>(cache_rows);
        if (position < 0 || position + padded > cache_limit) {
            throw std::out_of_range(
                std::to_string(rows) + " rows at position " + std::to_string(position) +
                " run a " + std::to_string(padded) + "-row kernel, which reaches row " +
                std::to_string(position + padded) + " of a " +
                std::to_string(cache_limit) + "-row cache");
        }
        // Chunked prefill is paged only, and only on a page boundary.
        // flat_mha rejects a multi-row call at a non-zero position itself, but
        // as the sixth dispatch of a layer, which aborts the process.
        const bool page_chunk = paged && position > 0 && position % kPageSize == 0;
        if (rows > 1 && position != 0 && !page_chunk) {
            throw std::invalid_argument(
                std::to_string(rows) + " rows at position " + std::to_string(position) +
                ": a multi-row pass must start at position 0" +
                (paged ? " or on a 4096-row page boundary" : ""));
        }
        RequireOneRowExtent(padded);

        // ---- the host's share, all of it before anything is dispatched
        const auto embedded = ScaledEmbeddingRows(ids);
        const auto planes = BuildPerLayerInputs(
            ple_table.bytes, ple_projection.values, ple_projection_norm.values,
            embedded, ids, cfg.layers, cfg.ple_dim, cfg.hidden, cfg.vocab, cfg.eps);

        auto lease = runtime->AcquireExecution();
        // THE RESIDUAL STREAM ENTERING LAYER 0 IS THE SCALED EMBEDDING, and it
        // is also what layer 0's norm reads -- so ONE staged write, and the
        // norm below writes `hidden` entirely. FP32 into a BF16 tensor:
        // corelib converts on the way in.
        std::vector<float> staged(static_cast<std::size_t>(padded * cfg.hidden), 0.0f);
        std::copy(embedded.begin(), embedded.end(), staged.begin());
        api->Check(api->functions().tensor_write(residual.get(),
                                                 ryzenai_corelib_data_type_fp32,
                                                 staged.data(), staged.size(), 0),
                   "ryzenai_corelib_tensor_write residual");
        // The per-layer embedding planes, ONCE per forward, for EVERY layer.
        // This is the host touch that is allowed, because it happens before
        // anything is dispatched. The tail stays zero, which is what the
        // padded rows' PLE input should be.
        std::vector<float> plane(static_cast<std::size_t>(padded * cfg.ple_dim), 0.0f);
        for (std::size_t layer = 0; layer < planes.size(); ++layer) {
            std::fill(plane.begin(), plane.end(), 0.0f);
            std::copy(planes[layer].begin(), planes[layer].end(), plane.begin());
            api->Check(api->functions().tensor_write(ple_in[layer].get(),
                                                     ryzenai_corelib_data_type_fp32,
                                                     plane.data(), plane.size(), 0),
                       "ryzenai_corelib_tensor_write per-layer embedding plane");
        }

        // ---- every window, before the first dispatch
        const auto hidden_w = Window(hidden.get(), {padded, cfg.hidden}, 0, "hidden");
        const auto residual_w = Window(residual.get(), {padded, cfg.hidden}, 0, "residual");
        const auto skip_w = Window(skip_sum.get(), {padded, cfg.hidden}, 0, "skip_sum");
        const auto ple_x_w = Window(ple_out.get(), {padded, cfg.hidden}, 0, "ple x");
        std::vector<UniqueTensorWindow> ple_w;
        ple_w.reserve(static_cast<std::size_t>(cfg.layers));
        for (std::int64_t layer = 0; layer < cfg.layers; ++layer) {
            ple_w.push_back(Window(ple_in[static_cast<std::size_t>(layer)].get(),
                                   {padded, cfg.ple_dim}, 0, "per-layer embedding"));
        }
        std::array<std::optional<Geometry>, 2> geometry;
        const auto geometry_for = [&](bool swa) -> const Geometry& {
            auto& slot = geometry[swa ? 0 : 1];
            if (!slot) slot = BuildGeometry(swa, padded);
            return *slot;
        };
        for (std::int64_t layer = 0; layer < cfg.layers; ++layer)
            (void)geometry_for(cfg.layer_is_swa[static_cast<std::size_t>(layer)]);
        // Where V's norm writes, per owning layer and per head: the SAME
        // per-head arithmetic as `v_src`, plus `position`. A sharing layer
        // gets none -- it projects no V at all, and handing it a window would
        // be a way to overwrite somebody else's cache with nothing.
        std::vector<std::vector<UniqueTensorWindow>> v_cache_w(
            static_cast<std::size_t>(cfg.layers));
        for (std::int64_t layer = 0; layer < cfg.kv_layers; ++layer) {
            // THE HEAD SIZE COMES FROM THE LAYER'S OWN GEOMETRY, not from its
            // index. The two agree on any model this loads, but they agree by
            // construction rather than by definition, and this window is the
            // one thing in the pass that would then stride a cache at the
            // other geometry's pitch.
            const auto head =
                geometry_for(cfg.layer_is_swa[static_cast<std::size_t>(layer)]).head;
            auto& windows = v_cache_w[static_cast<std::size_t>(layer)];
            windows.reserve(static_cast<std::size_t>(cfg.kv_heads));
            for (std::int64_t kv_head = 0; kv_head < cfg.kv_heads; ++kv_head) {
                windows.push_back(Window(
                    v_cache[static_cast<std::size_t>(layer)].get(), {padded, head},
                    CacheOffset(kv_head, position, head), "V cache"));
            }
        }

        // THE ENTIRE DEFINITION OF POISONING IS THIS FLAG, so what sets it is
        // load-bearing rather than incidental. IT FLIPS ONLY ON AN ACCEPTED
        // CALL -- after `Check` has let the status through -- so it means
        // "corelib took a command, and that command is in flight or has run".
        // A REFUSED dispatch enqueued nothing: if it is the first of the
        // pass there is nothing in flight and the engine is exactly as it
        // was, and if it is not the first then the accepted calls before it
        // have already set the flag. Setting it BEFORE the `Check` instead --
        // which is what this did until C9's fix round -- makes the refusal of
        // the very first dispatch poison an engine nothing has touched, and
        // makes `if (submitted)` in the catch indistinguishable from
        // `if (true)`, which is how the guard came to have no test.
        bool submitted = false;
        try {
            const auto dispatch = [&](ryzenai_corelib_status status, const char* what,
                                      std::int64_t layer) {
                api->Check(status, std::string(what) + " layer " + std::to_string(layer));
                submitted = true;
            };
            // LAYER 0'S ATTN_NORM, THE ONE STANDALONE RMSNORM IN THE MODEL.
            // Every other input norm is the previous layer's `ple` second
            // output; this one reads the embedding with no block before it.
            api->Check(api->functions().rmsnorm(stream.get(), residual_w.get(),
                                                input_norm.get(), hidden_w.get()),
                       "ryzenai_corelib_rmsnorm_bf16 layer 0 input norm");
            submitted = true;

            for (std::int64_t index = 0; index < cfg.layers; ++index) {
                const auto slot = static_cast<std::size_t>(index);
                const auto& weights = layers[slot];
                const bool swa = cfg.layer_is_swa[slot];
                const bool owns = index < cfg.kv_layers;
                const auto& kind = geometry_for(swa);

                dispatch(api->functions().matmul(stream.get(), hidden_w.get(),
                                                 weights.q.get(), kind.q.get()),
                         "ryzenai_corelib_matmul_bf16 query", index);
                // K AND V ONLY ON A LAYER THAT OWNS A CACHE. A sharing layer
                // attends over a cache an earlier layer of its own kind wrote,
                // and its descriptor is a kvshare artifact that takes no K
                // operand at all.
                if (owns) {
                    dispatch(api->functions().matmul(stream.get(), hidden_w.get(),
                                                     weights.k.get(), kind.k.get()),
                             "ryzenai_corelib_matmul_bf16 key", index);
                    dispatch(api->functions().matmul(stream.get(), hidden_w.get(),
                                                     weights.v.get(), kind.v_dest.get()),
                             "ryzenai_corelib_matmul_bf16 value", index);
                    dispatch(api->functions().rmsnorm(stream.get(), kind.k_heads_in.get(),
                                                      weights.k_norm.get(),
                                                      kind.k_heads_out.get()),
                             "ryzenai_corelib_rmsnorm_bf16 K-norm", index);
                    // V'S NORM IS WHAT WRITES THE CACHE, once per KV head,
                    // because each head's plane is a separate contiguous run
                    // in both buffers. Its gamma is ALL ONES -- the file
                    // carries no learned weight -- and dropping the step
                    // because of that leaves a whole 1/sqrt(mean(v^2)) out of
                    // every attended value.
                    for (std::int64_t kv_head = 0; kv_head < cfg.kv_heads; ++kv_head) {
                        dispatch(api->functions().rmsnorm(
                                     stream.get(),
                                     kind.v_src[static_cast<std::size_t>(kv_head)].get(),
                                     weights.v_norm.get(),
                                     v_cache_w[slot][static_cast<std::size_t>(kv_head)].get()),
                                 "ryzenai_corelib_rmsnorm_bf16 V-norm", index);
                    }
                }
                // QK-NORM, BEFORE ATTENTION, because flat_mha applies the
                // rotary itself and Gemma normalizes ahead of it. THESE GAMMAS
                // ARE THE ATTENTION SCALE -- constant-valued vectors whose
                // product is 0.125 on a sliding layer and 0.0625 on a full one
                // -- which is why the descriptor's scale is genuinely 1.0 and
                // why skipping a norm whose gamma "looks constant" is an 8x or
                // 16x logit error. NOTHING HERE PRE-SCALES Q.
                dispatch(api->functions().rmsnorm(stream.get(), kind.q_heads_in.get(),
                                                  weights.q_norm.get(),
                                                  kind.q_heads_out.get()),
                         "ryzenai_corelib_rmsnorm_bf16 Q-norm", index);
                // TWO ROTARY PAIRS, AND `swa` SELECTS. One pair used
                // everywhere still converges to fluent text.
                const auto owner = static_cast<std::size_t>(cfg.layer_cache_owner[slot]);
                void* cos = (swa ? cos_sliding : cos_full).get();
                void* sin = (swa ? sin_sliding : sin_full).get();
                if (plan.paged()) {
                    dispatch(api->functions().paged_mha(
                                 stream.get(), &plan.paged_attention_desc(swa, /*shared=*/!owns),
                                 kind.q_attn.get(), kind.k_attn.get(), position, cos, sin,
                                 k_cache[owner].get(), v_cache[owner].get(), kind.attn.get()),
                             "ryzenai_corelib_paged_mha_bf16", index);
                } else {
                    dispatch(api->functions().flat_mha(
                                 stream.get(),
                                 // SELECT, NEVER MUTATE. The plan holds four
                                 // immutable descriptors keyed (geometry, cache
                                 // role); there is deliberately no one-argument
                                 // accessor, so a loop that forgot the cache role
                                 // would not compile.
                                 &plan.attention_desc(swa, /*shared=*/!owns),
                                 kind.q_attn.get(), kind.k_attn.get(), position, cos, sin,
                                 k_cache[owner].get(), v_cache[owner].get(), kind.attn.get()),
                             "ryzenai_corelib_flat_mha_bf16", index);
                }
                dispatch(api->functions().matmul(stream.get(), kind.attn.get(),
                                                 weights.o.get(), hidden_w.get()),
                         "ryzenai_corelib_matmul_bf16 output", index);
                dispatch(api->functions().rmsnorm(stream.get(), hidden_w.get(),
                                                  weights.post_attn_norm.get(),
                                                  skip_w.get()),
                         "ryzenai_corelib_rmsnorm_bf16 post-attention", index);
                // THE BLOCK'S OUTPUT IS `skip_sum` (PLANE 0), NOT `normalized`
                // (plane 1), and that is the opposite of the silu family.
                // Under the gemma post-feedforward topology the trailing norm
                // moves INSIDE the residual, DD declares a single output, and
                // plane 1 COMES BACK UNTOUCHED -- it is still bound, because
                // an unbound plane dispatches a null pointer. Measured with
                // the operands the other way round, `normalized` returns a
                // plane of EXACT ZEROS, which propagates through `ple` and out
                // of all 35 layers as a finite, correctly-shaped vector of
                // zeros. `hidden_w` is the plane nothing writes, and `ple`
                // overwrites it two lines down.
                dispatch(api->functions().ssmlp(stream.get(), skip_w.get(),
                                                residual_w.get(), weights.mlp.get(),
                                                ple_x_w.get(), hidden_w.get()),
                         "ryzenai_corelib_ssmlp_bf16", index);
                // TWO OUTPUTS: the new residual stream, and the NEXT layer's
                // normed input. The second is why no layer boundary touches
                // the host, and on the LAST layer its gamma is `output_norm`
                // rather than a 36th block's -- so `hidden` already carries
                // the model's final norm when the loop ends. DO NOT APPLY A
                // FINAL NORM AFTERWARDS.
                dispatch(api->functions().ple(stream.get(), ple_x_w.get(),
                                              ple_w[slot].get(), weights.ple.get(),
                                              residual_w.get(), hidden_w.get()),
                         "ryzenai_corelib_ple_bf16", index);
            }

            // THE ONLY WAIT IN THE LOOP, and it is here because the next line
            // reads on the host.
            api->Check(api->functions().stream_synchronize(stream.get()),
                       "ryzenai_corelib_stream_synchronize layer loop");
            // The LAST live row. Read back as FP32: the tensor is BF16, so
            // corelib widens on the way out and there is nothing to unpack.
            std::vector<float> last(static_cast<std::size_t>(cfg.hidden));
            api->Check(api->functions().tensor_read(
                           hidden.get(), ryzenai_corelib_data_type_fp32, last.data(),
                           last.size(),
                           static_cast<std::size_t>((rows - 1) * cfg.hidden)),
                       "ryzenai_corelib_tensor_read final hidden row");
            RequireFinite(last, "the forward pass");

            // ---- the head. lm_head runs at M == 1 alone at this model's
            // (8, 17) pair -- one row is not a simplification, it is the only
            // shape available -- so its input is a separate one-row tensor.
            api->Check(api->functions().tensor_write(last_hidden.get(),
                                                     ryzenai_corelib_data_type_fp32,
                                                     last.data(), last.size(), 0),
                       "ryzenai_corelib_tensor_write LM head input");
            api->Check(api->functions().matmul(stream.get(), last_hidden.get(),
                                               lm_head.get(), logits.get()),
                       "ryzenai_corelib_matmul_bf16 LM head");
            api->Check(api->functions().stream_synchronize(stream.get()),
                       "ryzenai_corelib_stream_synchronize logits");
            std::vector<float> raw(static_cast<std::size_t>(cfg.vocab));
            api->Check(api->functions().tensor_read(logits.get(),
                                                    ryzenai_corelib_data_type_fp32,
                                                    raw.data(), raw.size(), 0),
                       "ryzenai_corelib_tensor_read logits");
            RequireFinite(raw, "the head's logits");
            // THE SOFTCAP, ON THE HOST, AFTER THE READ-BACK: tanh(x/30)*30,
            // with 30 read from the file rather than written here. There is no
            // matmul epilogue that does it. Omitting it looks like nothing --
            // the raw logits are finite and well spread, the argmax usually
            // agrees, and the text stays fluent -- but the cap changes every
            // softmax it feeds. A model that states 0.0 means "no cap", which
            // `tanh(x/0)` would answer with a plane of NaNs.
            if (cfg.logit_softcap != 0.0f) SoftcapLogits(raw, cfg.logit_softcap);

            buffer<bf16> out(static_cast<std::size_t>(cfg.vocab));
            std::vector<std::uint16_t> as_bf16(raw.size());
            FloatsToBf16(raw, as_bf16);
            static_assert(sizeof(bf16) == sizeof(std::uint16_t),
                          "buffer<bf16> holds two bytes per logit");
            std::memcpy(out.data(), as_bf16.data(),
                        as_bf16.size() * sizeof(std::uint16_t));
            position += static_cast<int>(rows);
            return out;
        } catch (...) {
            // WAIT FOR WHAT WAS SUBMITTED BEFORE UNWINDING PAST THE OPERANDS.
            // This does not make a mid-chain corelib rejection survivable --
            // nothing does, which is why every check above is before the first
            // dispatch -- but a failure raised on the HOST side after a
            // successful submission (an allocation, the finite check) must not
            // destroy the windows while their commands are in flight.
            //
            // AND THE ENGINE IS POISONED FROM HERE ON. `submitted` is the
            // whole of the condition, and it means an ACCEPTED dispatch:
            // a command corelib took, which is in flight or has run, and
            // whose effect on the device this function cannot put back.
            //
            //   - a mid-chain corelib rejection is unrecoverable by
            //     construction (see this function's header: the chain is
            //     unbarriered, so the unwind passes operands whose commands
            //     are still in flight). The engine cannot know how much of
            //     the loop ran, and the KV caches now hold rows for
            //     positions it will not count.
            //   - a non-finite result is PERMANENT -- BUT NOT FOR THE REASON
            //     THIS COMMENT USED TO GIVE. It said "every later token
            //     would attend over it", and that is not established:
            //     `flat_mha` is handed `position`, and `position += rows`
            //     runs only on the success path, so a failed pass's NaN rows
            //     sit at [position, position + padded) -- outside the window
            //     any later pass attends over, and a retry at the same
            //     position overwrites them. The two reasons that DO hold:
            //       * THE CAUSE IS DETERMINISTIC -- weights, kernel
            //         selection, shape -- so a retry reproduces it. The flag
            //         costs a caller nothing that was actually available.
            //       * `RequireFinite` SEES LESS THAN THE CORRUPTION IT
            //         DETECTS: the final live row's hidden, and the logits,
            //         and nothing else. A NaN written into a cache row BELOW
            //         `position` by an earlier, apparently successful pass is
            //         exactly what it cannot see -- and `clear_context()`
            //         resets a POSITION, it does not zero a cache. The
            //         observable failure understates the damage rather than
            //         overstating it, which is an argument for poisoning and
            //         not against it.
            //
            // Anything refused before a dispatch was ACCEPTED leaves the
            // engine exactly as it was and must not poison it: the caller
            // errors (an empty request, a token outside the vocabulary, a
            // chunked prefill, a request past capacity), plus the refusal of
            // the pass's very first dispatch, which enqueued nothing. Turning
            // any of those into a mandatory model reload would be a worse bug
            // than the one this guards.
            //
            // THE TWO `tensor_write`s ABOVE ALSO WRITE DEVICE MEMORY, and
            // they sit before the `try`. So "a submitted dispatch is the only
            // thing here that writes device state" -- which this comment used
            // to say -- is false. `residual` and every `ple_in` plane are
            // rewritten IN FULL at the head of every pass, so a partial write
            // to either is recoverable; the true rule, and the one to apply
            // when adding a device write, is "the only thing that writes
            // state a later pass reads WITHOUT FIRST REWRITING IT".
            //
            // BOTH HALVES ARE PINNED, in test_gemma4_engine.cpp: refusing
            // dispatch 20 must poison and refusing dispatch 0 must not, and
            // a third, structural test keeps the host rejections outside this
            // `try` so that the second of those keeps meaning what it says.
            //
            // `position` and `saved` are cleared for the same reason Phi-4's
            // engine clears them: the surviving numbers describe a
            // continuation that no longer exists. Every accessor is behind
            // `usable()`, so this is defence rather than something a caller
            // can observe.
            //
            // ONE EXCEPTION. A paged prefill chunk past page 0 can come back
            // non-finite from an intermittent kernel defect: the chunk
            // rewrites only its own page, and `position` has not moved, so
            // RunPrompt may run that same chunk again. Unpaged passes and
            // page 0 still poison. The message text is RequireFinite's.
            bool retry = false;
            try {
                throw;
            } catch (const std::exception& error) {
                retry = paged && position > 0 &&
                        std::string(error.what()).find("non-finite") != std::string::npos;
            } catch (...) {
            }
            if (submitted) {
                (void)api->functions().stream_synchronize(stream.get());
                if (!retry) {
                    poisoned = true;
                    position = 0;
                    saved.reset();
                }
            }
            throw;
        }
    }

    /// \brief one prompt, one page at a time when the cache is paged
    /// \note Starts at `position`, which is 0 or a page boundary when `ids`
    ///       holds more than one row.
    buffer<bf16> RunPrompt(std::span<const int> ids) {
        if (!paged || ids.size() <= 1 ||
            (position == 0 && static_cast<std::int64_t>(ids.size()) <= kPageSize))
            return Run(ids);
        buffer<bf16> out;
        const auto step = static_cast<std::size_t>(kPageSize);
        for (std::size_t start = 0; start < ids.size(); start += step) {
            const auto count = std::min(step, ids.size() - start);
            const auto chunk = ids.subspan(start, count);
            if (position == 0) {
                out = Run(chunk);
                continue;
            }
            for (int attempt = 1;; ++attempt) {
                try {
                    out = Run(chunk);
                    break;
                } catch (const std::runtime_error& error) {
                    const bool again =
                        std::string(error.what()).find("non-finite") != std::string::npos &&
                        attempt < kPagedChunkAttempts;
                    if (!again) throw;
                    std::cout << "[FLM]  Gemma 4 paged prefill at position " << position
                              << " came back non-finite; running it again (" << attempt
                              << " of " << kPagedChunkAttempts << ")" << std::endl;
                }
            }
        }
        return out;
    }

    /// \brief refuse a non-finite result, naming where it came from
    /// \note CHECKED AT THE PASS'S OWN EXIT AND AGAIN AT THE HEAD, because
    ///       both are places a caller reads from. Without it an all-NaN result
    ///       reaches an argmax that answers 0 -- `<pad>`, which decodes to the
    ///       empty string -- for every step, so a CLI prints an empty answer
    ///       under a token count and a rate and exits 0.
    /// \note IT IS THE WEAKER HALF OF THE GUARD AND MUST NOT BE READ AS THE
    ///       WHOLE OF IT. A first pass can be finite and WRONG -- in range,
    ///       well spread, correctly bounded, and with a different argmax --
    ///       and nothing about such a vector says so.
    static void RequireFinite(std::span<const float> values, const char* what) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::isfinite(values[i])) continue;
            throw std::runtime_error(std::string("Gemma 4: ") + what +
                                     " returned a non-finite value at index " +
                                     std::to_string(i));
        }
    }

    /// \brief one position's K or V cache row, gathered across the KV heads
    buffer<bf16> ReadCache(bool is_k, int layer, int index) {
        usable();
        const auto& cfg = shape;
        if (layer < 0 || layer >= cfg.layers || index < 0 ||
            index >= static_cast<int>(cache_rows))
            throw std::out_of_range("Gemma 4 cache index is out of range");
        // THE CACHE A LAYER READS IS ITS OWNER'S, not its own index. Twenty of
        // E2B's layers own none at all.
        const auto owner = static_cast<std::size_t>(
            cfg.layer_cache_owner[static_cast<std::size_t>(layer)]);
        const auto head = cfg.LayerHeadDim(static_cast<std::int64_t>(owner));
        auto lease = runtime->AcquireExecution();
        api->Check(api->functions().stream_synchronize(stream.get()),
                   "ryzenai_corelib_stream_synchronize cache read");
        buffer<bf16> out(static_cast<std::size_t>(cfg.kv_heads * head));
        void* cache = (is_k ? k_cache : v_cache)[owner].get();
        for (std::int64_t kv_head = 0; kv_head < cfg.kv_heads; ++kv_head) {
            const auto offset = CacheOffset(kv_head, index, head);
            api->Check(api->functions().tensor_read(
                           cache, ryzenai_corelib_data_type_bf16,
                           out.data() + kv_head * head, static_cast<std::size_t>(head),
                           offset),
                       "ryzenai_corelib_tensor_read cache head " +
                           std::to_string(kv_head));
        }
        return out;
    }
};

gemma4_rai::gemma4_rai(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

gemma4_rai::gemma4_rai(std::shared_ptr<Gemma4GgufPackage> package,
                       std::shared_ptr<corelib::CorelibRuntime> runtime,
                       const LM_Config& config, std::uint32_t max_length)
    : impl_(std::make_unique<Impl>(std::move(package), std::move(runtime), config,
                                   std::nullopt, max_length)) {}

std::unique_ptr<gemma4_rai> gemma4_rai::CreateWithShapeForTest(
    std::shared_ptr<Gemma4GgufPackage> package,
    std::shared_ptr<corelib::CorelibRuntime> runtime, const LM_Config& config,
    Gemma4Config shape, std::uint32_t max_length) {
    return std::unique_ptr<gemma4_rai>(
        new gemma4_rai(std::make_unique<Impl>(std::move(package), std::move(runtime),
                                              config, std::move(shape), max_length)));
}

gemma4_rai::~gemma4_rai() = default;

buffer<bf16> gemma4_rai::forward(int id) {
    auto out = impl_->Run(std::span<const int>(&id, 1));
    impl_->history.push_back(id);
    return out;
}

// ONE PATH, NOT TWO, and that is the reference driver's shape rather than
// Phi-4's: `forward(hidden, rows, position, ple_in)` is a single function
// there, and the only thing that distinguishes a prefill from a decode step is
// the row count it is handed. The rules that would otherwise be
// "prefill-only" -- a multi-row pass must start at position 0, and every
// buffer is windowed at one padded extent -- are stated once, in Run, as
// properties of the row count and the position.
buffer<bf16> gemma4_rai::prefill(std::vector<int>& ids, void*) {
    auto& impl = *impl_;
    if (impl.position == 0 || ids.size() <= 1) {
        if (impl.position == 0) impl.history.clear();
        auto out = impl.RunPrompt(std::span<const int>(ids.data(), ids.size()));
        impl.history.insert(impl.history.end(), ids.begin(), ids.end());
        return out;
    }
    // A multi-row pass starts at position 0 or, on a paged cache, on a page
    // boundary. So a continuation re-runs the rows since the nearest such
    // start ahead of its own: the whole conversation unpaged, and on a paged
    // cache only the current page's rows, none at all on a boundary.
    impl.usable();
    if (impl.history.size() != static_cast<std::size_t>(impl.position))
        throw std::invalid_argument("Gemma 4 cannot continue from position " +
                                    std::to_string(impl.position) +
                                    " without the tokens behind it");
    const int resume = impl.position;
    const int from = impl.paged ? resume - resume % static_cast<int>(kPageSize) : 0;
    std::vector<int> tokens(impl.history.begin() + from, impl.history.end());
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    impl.position = from;
    try {
        auto out = impl.RunPrompt(std::span<const int>(tokens.data(), tokens.size()));
        impl.history.insert(impl.history.end(), ids.begin(), ids.end());
        return out;
    } catch (...) {
        if (!impl.poisoned) impl.position = resume;
        throw;
    }
}

void gemma4_rai::set_context_length(int length) {
    impl_->usable();
    if (length < 0 || static_cast<std::uint32_t>(length) > impl_->max_length)
        throw std::out_of_range("Gemma 4 context length is out of range");
    impl_->position = length;
    if (impl_->history.size() > static_cast<std::size_t>(length))
        impl_->history.resize(static_cast<std::size_t>(length));
}

// An ABI shim, not a capability. load_weights is pure virtual in causal_lm.hpp,
// which is frozen because the engine libraries in src/lib/<runtime> are
// prebuilt against it -- adding, removing or reordering a virtual there
// corrupts dispatch at runtime with no compiler error. Nothing calls this:
// FlmBackend loads weights through the concrete engine type, and this engine's
// weights come from the GGUF package it was constructed with. See
// AutoModel/model_backend.hpp.
void gemma4_rai::load_weights(Q4NX&) {
    impl_->usable();
    throw std::runtime_error("Gemma 4 rai weights are loaded only from GGUF");
}

void gemma4_rai::update_max_length(std::uint32_t max_length) {
    impl_->usable();
    if (!max_length || max_length > impl_->cache_rows ||
        max_length < static_cast<std::uint32_t>(impl_->position))
        throw std::out_of_range("Gemma 4 maximum length is invalid");
    impl_->max_length = max_length;
}

void gemma4_rai::clear_context() {
    impl_->usable();
    impl_->position = 0;
    impl_->saved.reset();
    impl_->history.clear();
}

buffer<bf16> gemma4_rai::get_k_cache(int layer, int index) {
    return impl_->ReadCache(/*is_k=*/true, layer, index);
}

buffer<bf16> gemma4_rai::get_v_cache(int layer, int index) {
    return impl_->ReadCache(/*is_k=*/false, layer, index);
}

int gemma4_rai::get_current_context_length() {
    impl_->usable();
    return impl_->position;
}

int gemma4_rai::checkpoint() {
    impl_->usable();
    impl_->saved = impl_->position;
    return impl_->position;
}

int gemma4_rai::restore() {
    impl_->usable();
    if (!impl_->saved) return -1;
    impl_->position = *impl_->saved;
    if (impl_->history.size() > static_cast<std::size_t>(impl_->position))
        impl_->history.resize(static_cast<std::size_t>(impl_->position));
    return impl_->position;
}

bool gemma4_rai::poisoned() const noexcept { return impl_ && impl_->poisoned; }

bool gemma4_rai::loaded_from_cache() const noexcept {
    return impl_ && impl_->from_cache;
}

std::size_t gemma4_rai::weight_slot_count() const noexcept {
    return impl_ ? impl_->slot_count : 0;
}

std::vector<WeightPlacement> gemma4_rai::WeightPlacementsForTest() const {
    std::vector<WeightPlacement> placements;
    if (!impl_) return placements;
    placements.reserve(impl_->slot_count);
    // NAME EACH MEMBER SEPARATELY, IN FULL. See the header: this function's
    // only value is that it is a SECOND statement of the tensor-to-member
    // mapping, written here rather than read off the slot table that
    // performed the assignment. A loop over `slots` would agree with the
    // assignment by construction and could never disagree with it.
    placements.push_back({"blk.0.attn_norm.weight", impl_->input_norm.get()});
    for (std::size_t index = 0; index < impl_->layers.size(); ++index) {
        const auto layer = static_cast<std::int64_t>(index);
        const auto& weights = impl_->layers[index];
        placements.push_back({Blk(layer, ".attn_q.weight"), weights.q.get()});
        if (weights.k) placements.push_back({Blk(layer, ".attn_k.weight"), weights.k.get()});
        if (weights.v) placements.push_back({Blk(layer, ".attn_v.weight"), weights.v.get()});
        placements.push_back({Blk(layer, ".attn_q_norm.weight"), weights.q_norm.get()});
        if (weights.k_norm)
            placements.push_back({Blk(layer, ".attn_k_norm.weight"), weights.k_norm.get()});
        if (weights.v_norm)
            placements.push_back({Blk(layer, ".attn_v_norm(ones)"), weights.v_norm.get()});
        placements.push_back({Blk(layer, ".attn_output.weight"), weights.o.get()});
        placements.push_back(
            {Blk(layer, ".post_attention_norm.weight"), weights.post_attn_norm.get()});
        placements.push_back({Blk(layer, ".ssmlp"), weights.mlp.get()});
        placements.push_back({Blk(layer, ".ple"), weights.ple.get()});
    }
    placements.push_back({"token_embd.weight", impl_->lm_head.get()});
    return placements;
}

}  // namespace flm::gemma4
