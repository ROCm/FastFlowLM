#include "models/gemma4/rai/aie_next/gemma4_rai_shape_plan.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "gemma4_gguf_fixture.hpp"
#include "rai/gguf_file.hpp"
#include "rai/weight_source.hpp"
#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using flm::corelib::CorelibApi;
using flm::gemma4::ConfigFromMetadata;
using flm::gemma4::Gemma4Config;
using flm::gemma4::Gemma4ShapePlan;
using flm::rai::GgufFile;

std::shared_ptr<CorelibApi> Api() {
    return CorelibApi::ResolveForTest(fake_corelib::Resolver());
}

/// \brief E2B's config, read through the same fixture/derivation path every
///        other gemma4_rai test uses (test_gemma4_gguf.cpp's
///        TestConfigFromMetadataReadsE2b) -- E2B is the model with TWO
///        distinct FFN widths (6144, then 12288 from layer 15), which is
///        exactly what TestSsmlpIsPlannedAtBothOfE2bsFfnWidths needs.
Gemma4Config E2bConfigForTest() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    return ConfigFromMetadata(*file);
}

/// \brief ship exactly the kernels a correct plan for `config` asks about
/// \note Every prefill bucket for Q, K/V and the output projection at
///       `group`, both geometries; the gemma_fusion ssmlp at every FFN width;
///       the owning, `_scale1` attention kernel of each geometry, keyed on
///       its window only where corelib would key it; and lm_head at one row
///       and `head_group`. A test then breaks ONE property and checks the
///       plan refuses to build -- which is what makes the query observable,
///       since the plan filters the reported set itself.
void ShipFullGrid(const Gemma4Config& config) {
    auto& state = fake_corelib::GetState();
    state.explicit_grid = true;
    constexpr std::int64_t kRows[] = {1, 64, 128, 256, 512, 1024, 2048, 3072, 4096};
    std::vector<std::int64_t> widths;
    for (const auto width : config.layer_intermediate)
        if (std::find(widths.begin(), widths.end(), width) == widths.end())
            widths.push_back(width);
    for (const auto rows : kRows) {
        for (const auto head : {config.head_dim, config.global_head_dim}) {
            state.shipped_matmul.push_back({rows, config.hidden, config.q_heads * head, config.group});
            state.shipped_matmul.push_back({rows, config.hidden, config.kv_heads * head, config.group});
            state.shipped_matmul.push_back({rows, config.q_heads * head, config.hidden, config.group});
        }
        for (const auto width : widths)
            state.shipped_ssmlp.push_back({"gemma_fusion", rows, config.hidden, width, config.group});
        for (const bool swa : {true, false}) {
            const auto window = swa ? config.sliding_window : 0;
            const auto keyed = window > 0 && window < rows ? window : 0;
            state.shipped_mha.push_back({config.q_heads, config.kv_heads, rows,
                                         swa ? config.head_dim : config.global_head_dim,
                                         flm::gemma4::kMaxSequenceLength, keyed, false, true});
        }
    }
    state.shipped_matmul.push_back({1, config.hidden, config.vocab, config.head_group});
}

/// \brief build a plan against the fake's current grid
/// \return empty on success, else the error the plan raised
std::string BuildError(const std::shared_ptr<CorelibApi>& api, ryzenai_corelib_stream_ptr stream,
                       const Gemma4Config& config) {
    try {
        (void)Gemma4ShapePlan::Build(api, stream, config);
        return {};
    } catch (const std::runtime_error& error) {
        return error.what();
    }
}

/// \brief the plan builds against the full grid, and refuses one with
///        `mutate` applied, naming `what`
template <typename Mutate>
void RequireGridBreaks(const Gemma4Config& config, Mutate mutate, const std::string& what) {
    fake_corelib::Reset();
    auto api = CorelibApi::ResolveForTest(fake_corelib::Resolver());
    auto* stream = fake_corelib::MakeStreamForTest();
    ShipFullGrid(config);
    TEST_REQUIRE(BuildError(api, stream, config).empty());
    mutate(fake_corelib::GetState());
    const auto error = BuildError(api, stream, config);
    TEST_REQUIRE(!error.empty());
    RequireContains(error, what + " has no kernel");
}

void TestPlanBuildsBothGeometriesWithDifferentHeadSizes() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto config = E2bConfigForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, config);

    // head_dim (sliding) is 256, global_head_dim (full) is 512 -- these
    // resolve DIFFERENT ELFs, which is the whole reason this plan holds two
    // descriptors instead of one.
    TEST_REQUIRE(plan.attention_desc(true, /*shared=*/false).head_size == 256);
    TEST_REQUIRE(plan.attention_desc(false, /*shared=*/false).head_size == 512);
    TEST_REQUIRE(plan.attention_desc(true, /*shared=*/false).window == 512);
    // Full-attention layers are not windowed; 0 is corelib's "no window".
    TEST_REQUIRE(plan.attention_desc(false, /*shared=*/false).window == 0);
    TEST_REQUIRE(plan.attention_desc(true, /*shared=*/false).num_heads == 8);
    TEST_REQUIRE(plan.attention_desc(true, /*shared=*/false).kv_num_heads == 1);
    TEST_REQUIRE(plan.attention_desc(true, /*shared=*/false).max_seq == 4096);
    // Both geometries share the same head counts and max_seq -- only
    // head_size and window differ between them.
    TEST_REQUIRE(plan.attention_desc(false, /*shared=*/false).num_heads == 8);
    TEST_REQUIRE(plan.attention_desc(false, /*shared=*/false).kv_num_heads == 1);
    TEST_REQUIRE(plan.attention_desc(false, /*shared=*/false).max_seq == 4096);
}

// Every kernel query names the stream, because which kernels ship is a
// property of the stream's PDI pair. There is no ple query at all: ple_bf16's
// own doc says M comes from the input tensor's extent, with nothing to ask --
// see gemma4_rai_shape_plan.cpp's Build() for the full reasoning.
void TestPlanPassesTheStreamToEveryKernelQuery() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    (void)Gemma4ShapePlan::Build(api, stream, E2bConfigForTest());
    const auto& calls = fake_corelib::GetState().enum_calls;
    const auto count = [&](std::string_view entry) {
        return std::count_if(calls.begin(), calls.end(),
                             [&](const auto& call) { return call.entry == entry; });
    };
    TEST_REQUIRE(count("ryzenai_corelib_matmul_bf16_enum_kernels") == 1);
    TEST_REQUIRE(count("ryzenai_corelib_ssmlp_bf16_enum_kernels") == 1);
    // One per geometry: the two descriptors resolve different kernels.
    TEST_REQUIRE(count("ryzenai_corelib_flat_mha_bf16_enum_kernels") == 2);
    for (const auto& call : calls) TEST_REQUIRE(call.stream == stream);
}

/// \brief the plan asks for the gemma_fusion ssmlp family at `group`
/// \note `activation` and `post_feedforward_layernorm` are not part of what a
///       kernel query sees: the post-norm IS the family (gemma_fusion rather
///       than fusion_mladf_ssmlp), so the family is what is asserted here.
///       The descriptor fields themselves travel with the weights, see
///       test_gemma4_engine's ssmlp tests.
void TestSsmlpIsQueriedInTheGemmaFamilyAtItsGroup() {
    const auto config = E2bConfigForTest();
    TEST_REQUIRE(config.group == 32);
    RequireGridBreaks(config, [](auto& state) {
        for (auto& shipped : state.shipped_ssmlp) shipped.family = "fusion_mladf_ssmlp";
    }, "ssmlp");
    RequireGridBreaks(config, [](auto& state) {
        for (auto& shipped : state.shipped_ssmlp) shipped.group = 64;
    }, "ssmlp");
}

void TestSsmlpIsPlannedAtBothOfE2bsFfnWidths() {
    const auto config = E2bConfigForTest();
    for (const std::int64_t width : {6144, 12288}) {
        RequireGridBreaks(config, [width](auto& state) {
            auto& shipped = state.shipped_ssmlp;
            shipped.erase(std::remove_if(shipped.begin(), shipped.end(),
                                         [width](const auto& s) { return s.n == width; }),
                          shipped.end());
        }, "ssmlp");
    }
}

/// \brief scale is 1.0f on every descriptor the plan holds AND on the
///        descriptors it queries the shipped kernel set with
///
/// The review that produced this test observed that a build with `scale`
/// fixed, or broken back to any other constant, passed all five of the
/// original tests unchanged, because nothing read the field off anything.
/// Both halves matter: the stored descriptors are what Task C8 dispatches
/// with, and the query is where a wrong `scale` fails on hardware, long
/// before the first dispatch.
void TestAttentionScaleIsExactlyOneEverywhere() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto config = E2bConfigForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, config);
    TEST_REQUIRE(plan.attention_desc(true, /*shared=*/false).scale == 1.0f);
    TEST_REQUIRE(plan.attention_desc(false, /*shared=*/false).scale == 1.0f);
    RequireGridBreaks(config, [](auto& state) {
        for (auto& shipped : state.shipped_mha) shipped.scale_one = false;
    }, "flat_mha");
}

/// \brief all four (geometry, cache-role) descriptors exist, and each one is
///        the descriptor a layer of that kind must actually be dispatched with
///
/// `kv_shared` selects a DIFFERENT artifact that is handed no K operand and
/// merges nothing into the cache (corelib.h:1772-1779). The reference driver
/// keys its descriptors on exactly this pair and states the stakes:
/// "An owning layer given a kvshare descriptor never writes its own cache; a
/// sharing layer given an owning one overwrites somebody else's. Neither
/// errors." (gemma4_driver.py:1526-1559.) So the plan must MATERIALIZE all
/// four and let the layer loop SELECT -- a single template whose `kv_shared`
/// the loop mutates per layer leaves a silent wrong-cache write one missed
/// assignment away.
void TestPlanHoldsFourDescriptorsKeyedByGeometryAndCacheRole() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, E2bConfigForTest());

    // The cache role is the ONLY thing kv_shared tracks: an owning layer gets
    // 0, a sharing layer gets 1, on both geometries alike.
    TEST_REQUIRE(plan.attention_desc(true, false).kv_shared == 0);
    TEST_REQUIRE(plan.attention_desc(true, true).kv_shared == 1);
    TEST_REQUIRE(plan.attention_desc(false, false).kv_shared == 0);
    TEST_REQUIRE(plan.attention_desc(false, true).kv_shared == 1);

    // head_size and window follow the GEOMETRY and nothing else -- the cache
    // role must not perturb either, or a sharing layer would attend at the
    // wrong width.
    for (const bool shared : {false, true}) {
        TEST_REQUIRE(plan.attention_desc(true, shared).head_size == 256);
        TEST_REQUIRE(plan.attention_desc(false, shared).head_size == 512);
        TEST_REQUIRE(plan.attention_desc(true, shared).window == 512);
        TEST_REQUIRE(plan.attention_desc(false, shared).window == 0);
    }

    // Everything that is a fact about the MODEL rather than about a layer is
    // equal across all four.
    for (const bool swa : {true, false}) {
        for (const bool shared : {false, true}) {
            const auto& desc = plan.attention_desc(swa, shared);
            TEST_REQUIRE(desc.num_heads == 8);
            TEST_REQUIRE(desc.kv_num_heads == 1);
            TEST_REQUIRE(desc.max_seq == 4096);
            TEST_REQUIRE(desc.scale == 1.0f);
        }
    }

    // The plan's own kernel queries run through the OWNING descriptor of each
    // geometry, which the header states is safe because kv_shared changes
    // which kernel runs, not any shape (corelib.h:1810-1812). A grid that
    // ships only the kvshare family must therefore leave the plan uncovered
    // -- a query made with a kvshare descriptor would be asking the wrong
    // artifact family a question it happens to answer identically, which is
    // right by luck rather than by the rule.
    RequireGridBreaks(E2bConfigForTest(), [](auto& state) {
        for (auto& shipped : state.shipped_mha) shipped.kv_shared = true;
    }, "flat_mha");
}

/// \brief rope_dim is the full head width on BOTH geometries
///
/// FULL ROTARY, resolved -- not an open question. `rope_dim == head_size` is
/// what the reference driver sets (gemma4_driver.py:1537-1543), and it is what
/// makes the shipped `..._256_4096_lc512_scale1` keys -- which carry no
/// emb_dim segment -- resolve. `config.json`'s `partial_rotary_factor: 0.25`
/// is NOT a shorter rotary: it is a per-pair frequency divisor carried in
/// `rope_freqs.weight`, which is a different input entirely (the rope TABLES,
/// Task C7) and does not touch this field.
void TestRopeDimIsFullRotaryOnBothGeometries() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, E2bConfigForTest());
    for (const bool shared : {false, true}) {
        TEST_REQUIRE(plan.attention_desc(true, shared).rope_dim == 256);
        TEST_REQUIRE(plan.attention_desc(false, shared).rope_dim == 512);
        TEST_REQUIRE(plan.attention_desc(true, shared).rope_dim ==
                     plan.attention_desc(true, shared).head_size);
        TEST_REQUIRE(plan.attention_desc(false, shared).rope_dim ==
                     plan.attention_desc(false, shared).head_size);
    }
}

/// \brief EVERY PROJECTION queries at `config.group`; `config.head_group`
///        belongs to lm_head and to nothing else
///
/// The direction here is the reference driver's, read off its own field
/// declaration rather than inferred from the ELF set:
///
///     group: int        # every projection
///     head_group: int   # lm_head, which on other models ships a DIFFERENT group
///     (gemma4_driver.py:537-539)
///
/// `_matmul_weights` (`gemma4_driver.py:1383-1387`) defaults
/// `group_size=self.cfg.group`, and the ONLY call in the whole driver that
/// overrides it is lm_head (`:1691-1694`, "at `head_group`, which on this
/// model happens to equal `group`"). `logits_for` says the same from the
/// other end: lm_head is "the only weight packed at `cfg.head_group`"
/// (`:2449-2453`). The driver's own enumeration of what a prefill dispatches
/// puts Q, K/V AND the output projection at g32 == `cfg.group` (`:294-298`).
///
/// An earlier version of this test asserted the opposite -- Q and K/V at
/// `head_group` -- and passed, because both shipped Gemma 4 rows set
/// `group == head_group == ple_group == 32`, so the two fields are
/// indistinguishable on any real config. Raising `head_group` to a value no
/// shipped row has is what makes the assertion observable at all; it is NOT a
/// claim that such a model exists, and nothing downstream of `Build()` is
/// exercised by it.
/// \brief every projection fails to plan when only its kernels move to the
///        other group
/// \note Q and K/V project OUT of the residual stream INTO head space, so
///       their K is the hidden size and their N is a head-shaped width
///       (q_heads or kv_heads times the geometry's head size). The output
///       projection runs the other way: N is the hidden size. lm_head is
///       hidden -> vocab, which is neither.
void RequireProjectionGroups(const Gemma4Config& config) {
    const auto regroup = [&config](auto pick) {
        return [&config, pick](auto& state) {
            for (auto& shipped : state.shipped_matmul) {
                if (!pick(shipped)) continue;
                shipped.group = shipped.group == config.group ? config.head_group : config.group;
            }
        };
    };
    const auto is_query = [&config](const auto& s) {
        return s.k == config.hidden && (s.n == config.q_heads * config.head_dim ||
                                        s.n == config.q_heads * config.global_head_dim);
    };
    const auto is_key_value = [&config](const auto& s) {
        return s.k == config.hidden && (s.n == config.kv_heads * config.head_dim ||
                                        s.n == config.kv_heads * config.global_head_dim);
    };
    const auto is_output = [&config](const auto& s) { return s.n == config.hidden; };
    const auto is_lm_head = [&config](const auto& s) { return s.n == config.vocab; };
    RequireGridBreaks(config, regroup(is_query), "query");
    RequireGridBreaks(config, regroup(is_key_value), "key/value");
    RequireGridBreaks(config, regroup(is_output), "output");
    RequireGridBreaks(config, regroup(is_lm_head), "lm_head");
}

void TestEveryProjectionQueriesAtGroupAndOnlyLmHeadAtHeadGroup() {
    auto config = E2bConfigForTest();
    config.head_group = 64;
    TEST_REQUIRE(config.head_group != config.group);
    RequireProjectionGroups(config);
}

/// \brief lm_head is planned ONCE, at one row, and at `head_group`
///
/// Three separate claims, all from the driver:
///   - it is `hidden -> vocab` and TIED to token_embd, so there is no
///     `output.weight` and no second shape (`gemma4_driver.py:1691-1694`);
///   - "ONE ROW IS NOT A SIMPLIFICATION, IT IS THE ONLY SHAPE AVAILABLE...
///     it ships at M == 1 alone at this model's (8, 17) pair, so there is no
///     batched form to offer even if a caller wanted every position's
///     logits" (`:2449-2453`). Walking it through this plan's row buckets
///     would therefore ask corelib for shapes that do not exist;
///   - it is "the only weight packed at `cfg.head_group`" (same place).
///
/// The driver states the same negatively when it enumerates what a prefill
/// dispatches: "lm_head runs at M=1, so neither is a prefill matmul"
/// (`:294-298`).
void TestLmHeadIsPlannedOnceAtOneRowAndAtHeadGroup() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    auto config = E2bConfigForTest();
    config.head_group = 64;  // see the test above for why they are pulled apart
    // The full grid ships lm_head at M == 1 and nowhere else, which is the
    // real kernel set: a plan that walked lm_head through the row buckets
    // would find no kernel for them.
    ShipFullGrid(config);
    const auto plan = Gemma4ShapePlan::Build(api, stream, config);
    // The logits tensor C7 allocates is [lm_head_rows(), vocab].
    TEST_REQUIRE(plan.lm_head_rows() == 1);
    // And it is queried at all: without its kernel the plan is uncovered.
    RequireGridBreaks(config, [&config](auto& state) {
        auto& shipped = state.shipped_matmul;
        shipped.erase(std::remove_if(shipped.begin(), shipped.end(),
                                     [&config](const auto& s) { return s.n == config.vocab; }),
                      shipped.end());
    }, "lm_head");
}

/// \brief the first dispatch Task C7 will make is ACCEPTED after Build()
///
/// This is the review's own probe, kept as a regression test. The fake
/// rejects a matmul whose operands carry a row count the matching
/// `"matmul-<n>"` pad helper never answered (`OnGrid`), and before this round
/// `Build()` never queried lm_head's width at all -- so a post-`Build()`
/// lm_head matmul came back `bad_argument`. That is a FALSE rejection of a
/// legitimate dispatch, not a caught bug, and it is the very first thing C7
/// does with a plan.
///
/// The test asserts the accept, not the reject, deliberately: the thing that
/// must hold for C7 is that a correct lm_head dispatch goes through.
void TestLmHeadDispatchAfterAFullPlanBuildIsAccepted() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto config = E2bConfigForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, config);
    const auto& fns = api->functions();

    const ryzenai_corelib_matmul_bf16_weights_desc weights_desc{
        config.hidden, config.vocab, config.head_group, false};
    const std::vector<std::byte> blocks(64, std::byte{0});
    const ryzenai_corelib_weights_source sources[]{flm::corelib::GgufQ8(
        ryzenai_corelib_weights_role_qweight, blocks.data(), blocks.size(), config.vocab,
        config.hidden)};
    ryzenai_corelib_weights_ptr weights = nullptr;
    TEST_REQUIRE(fns.matmul_weights_create(&weights_desc, sources, 1, nullptr, &weights) ==
                 ryzenai_corelib_status_success);

    auto make = [&](std::int64_t rows, std::int64_t columns) {
        const std::int64_t shape[] = {rows, columns};
        ryzenai_corelib_tensor_ptr tensor = nullptr;
        fns.create_device_tensor(ryzenai_corelib_data_type_bf16, shape, 2, &tensor);
        return tensor;
    };
    auto* input = make(plan.lm_head_rows(), config.hidden);
    auto* output = make(plan.lm_head_rows(), config.vocab);
    TEST_REQUIRE(fns.matmul(stream, input, weights, output) ==
                 ryzenai_corelib_status_success);

    for (auto* object : {input, output}) fns.object_release(object);
    fns.object_release(weights);
}

/// \brief a flat_mha / ssmlp dispatch record carries the descriptor it ran
///        with, so C7/C8 can assert on it
///
/// Round 1 made the WRONG descriptor impossible to construct (four immutable
/// descriptors and a two-argument accessor). That closes the construction
/// mistake; it does not close the HANDOVER mistake -- the right descriptor
/// built and the wrong one passed to the dispatch. The reference driver
/// closes that host-side, at load, by reading the descriptor's own field back
/// against `cfg.kv_layers`: "layer {index} was given an attention descriptor
/// with kv_shared=..., but this model's first {kv_layers} layers own a cache
/// and the rest share one" (`gemma4_driver.py:1670-1690`). That guard is only
/// writable if the record retains the descriptor.
///
/// ssmlp's dispatch takes a packed WEIGHTS object rather than a descriptor,
/// so what is recorded there is the descriptor the weights were CREATED with
/// -- which is the thing `activation` and `post_feedforward_layernorm` live
/// on and the thing a C8 assertion needs.
void TestDispatchRecordsCarryTheDescriptorTheyRanWith() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto config = E2bConfigForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, config);
    const auto& fns = api->functions();

    auto make = [&](std::int64_t rows, std::int64_t columns) {
        const std::int64_t shape[] = {rows, columns};
        ryzenai_corelib_tensor_ptr tensor = nullptr;
        fns.create_device_tensor(ryzenai_corelib_data_type_bf16, shape, 2, &tensor);
        return tensor;
    };

    // A SHARING sliding layer: the case the 2x2 exists for.
    const auto& sharing = plan.attention_desc(/*swa=*/true, /*shared=*/true);
    const auto mha_rows = plan.ForRows(64, /*swa=*/true).flat_mha_rows;
    auto* q = make(mha_rows, config.q_heads * config.head_dim);
    auto* k = make(mha_rows, config.kv_heads * config.head_dim);
    auto* out = make(mha_rows, config.q_heads * config.head_dim);
    const float rotary[1] = {};
    const std::int64_t rotary_shape[] = {1, 1};
    ryzenai_corelib_host_view_ptr cos_view = nullptr, sin_view = nullptr;
    fns.create_host_view(ryzenai_corelib_data_type_fp32, rotary_shape, 2, rotary, &cos_view);
    fns.create_host_view(ryzenai_corelib_data_type_fp32, rotary_shape, 2, rotary, &sin_view);
    TEST_REQUIRE(fns.flat_mha(stream, &sharing, q, k, 0, cos_view, sin_view,
                              nullptr, nullptr, out) ==
                 ryzenai_corelib_status_success);

    const ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp_desc{
        config.hidden, config.layer_intermediate[0], config.group, 1, 1};
    ryzenai_corelib_weights_ptr ssmlp_weights = nullptr;
    const std::vector<std::byte> ssmlp_blocks(64, std::byte{0});
    const ryzenai_corelib_weights_source ssmlp_sources[]{flm::corelib::GgufQ8(
        ryzenai_corelib_weights_role_gate_qweight, ssmlp_blocks.data(), ssmlp_blocks.size(),
        ssmlp_desc.n, ssmlp_desc.k)};
    TEST_REQUIRE(fns.ssmlp_weights_create(&ssmlp_desc, ssmlp_sources, 1, nullptr,
                                          &ssmlp_weights) == ryzenai_corelib_status_success);
    const auto ssmlp_rows = plan.ForRows(64, /*swa=*/true).ssmlp_rows;
    auto* input = make(ssmlp_rows, config.hidden);
    auto* residual = make(ssmlp_rows, config.hidden);
    auto* skip_sum = make(ssmlp_rows, config.hidden);
    auto* normalized = make(ssmlp_rows, config.hidden);
    TEST_REQUIRE(fns.ssmlp(stream, input, residual, ssmlp_weights, skip_sum,
                           normalized) == ryzenai_corelib_status_success);

    const auto& dispatches = fake_corelib::GetState().dispatches;
    TEST_REQUIRE(dispatches.size() == 2);

    TEST_REQUIRE(dispatches[0].kind == "mha");
    TEST_REQUIRE(dispatches[0].has_mha_desc);
    // THE ASSERTION C7 NEEDS: the cache role actually handed over, not the
    // one the plan holds.
    TEST_REQUIRE(dispatches[0].mha_desc.kv_shared == 1);
    TEST_REQUIRE(dispatches[0].mha_desc.head_size == config.head_dim);
    TEST_REQUIRE(dispatches[0].mha_desc.window == config.sliding_window);
    TEST_REQUIRE(dispatches[0].mha_desc.scale == 1.0f);
    TEST_REQUIRE(!dispatches[0].has_ssmlp_desc);

    TEST_REQUIRE(dispatches[1].kind == "ssmlp");
    TEST_REQUIRE(dispatches[1].has_ssmlp_desc);
    TEST_REQUIRE(dispatches[1].ssmlp_desc.n == config.layer_intermediate[0]);
    TEST_REQUIRE(dispatches[1].ssmlp_desc.activation == 1);
    TEST_REQUIRE(dispatches[1].ssmlp_desc.post_feedforward_layernorm == 1);
    TEST_REQUIRE(!dispatches[1].has_mha_desc);

    for (auto* object : {q, k, out, input, residual, skip_sum, normalized})
        fns.object_release(object);
    fns.object_release(ssmlp_weights);
}

/// \brief this round's new assertions, run against E4B as well as E2B
///
/// E4B is the model this suite has never built a plan for: hidden 2560, two
/// KV heads (so the K/V projection widths move from 256/512 to 512/1024), a
/// uniform 10240 FFN width, 42 layers. Every claim below is model-agnostic by
/// construction, which is exactly why it is worth running on the second model
/// -- "agnostic by construction" is the class of claim this project has twice
/// found to be wrong.
void TestE4bGroupDirectionAndLmHeadPlan() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    auto config = ConfigFromMetadata(*file);
    TEST_REQUIRE(config.hidden == 2560);
    TEST_REQUIRE(config.kv_heads == 2);
    config.head_group = 64;
    ShipFullGrid(config);
    const auto plan = Gemma4ShapePlan::Build(api, stream, config);
    TEST_REQUIRE(plan.lm_head_rows() == 1);
    RequireProjectionGroups(config);

    // E4B's K/V widths (512 and 1024) collide with nothing else E4B queries:
    // 2560 hidden, 2048/4096 Q, 262144 lm_head. On E2B they are 256 and 512.
    TEST_REQUIRE(config.kv_heads * config.head_dim == 512);
    TEST_REQUIRE(config.kv_heads * config.global_head_dim == 1024);
}

void TestPleRowsRoundUpToAShippedBucket() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto plan = Gemma4ShapePlan::Build(api, stream, E2bConfigForTest());
    // ple_bf16's M selects the kernel and the kernel then reads and writes
    // that many rows IN FULL, so every tensor must hold the padded extent, not
    // the live one -- a buffer sized to the live rows is run off the end.
    TEST_REQUIRE(plan.ForRows(1, true).ple_rows == 1);
    TEST_REQUIRE(plan.ForRows(2, true).ple_rows == 64);
    TEST_REQUIRE(plan.ForRows(65, true).ple_rows == 128);
    TEST_REQUIRE(plan.ForRows(4095, true).ple_rows == 4096);
    // The bucket table is geometry-independent (ple_bf16's row semantics do
    // not depend on which attention geometry a layer runs), so the
    // full-attention side must agree.
    TEST_REQUIRE(plan.ForRows(1, false).ple_rows == 1);
    TEST_REQUIRE(plan.ForRows(4095, false).ple_rows == 4096);
}
// ---------------------------------------------------------------------------
// The fake's own rejections
//
// These four do not exercise Gemma4ShapePlan at all -- they pin
// src/test/gemma4_rai/fake_corelib.{hpp,cpp}, which is about to become the
// thing test_gemma4_engine and test_gemma4_backend are tested against. Review
// round 1 found that fake recording calls and validating almost nothing while
// backing the SAME corelib ABI as src/test/phi4_rai's, which validates a great
// deal, and named that as the shape that let two Criticals through a fully
// green suite on the Phi-4 side.
//
// They live in this file rather than a new one because a new test .cpp needs a
// new target block in src/test/gemma4_rai/CMakeLists.txt, and that file is
// shared with concurrently-active tasks -- two agents editing it at once is a
// collision this project has already had. If a dedicated fake-self-test target
// is ever added, move them wholesale.
//
// Read fake_corelib.hpp's "what it still cannot catch" list before reading a
// green run here as proof of more than these four rejections.
// ---------------------------------------------------------------------------

/// \brief a matmul kernel at a different K or N does not cover the plan
///
/// corelib rounds nothing: a kernel is shipped at an exact (K, N), and one a
/// few columns wider is a different artifact. The plan must not take it as
/// covering the logical shape. The N side and the K side are broken
/// independently, so a fix that only compared one would not pass this.
/// \brief the paged kernels corelib 0.26's gemma4 bundle ships: the decode
///        step and the page-sized chunk, nothing in between
void ShipPagedChunkOnly(const Gemma4Config& config) {
    auto& state = fake_corelib::GetState();
    for (const auto head : {config.head_dim, config.global_head_dim}) {
        for (const std::int64_t rows : {std::int64_t{1}, flm::gemma4::kPageSize}) {
            state.shipped_paged.push_back(
                {config.q_heads, config.kv_heads, rows, head, flm::gemma4::kPageSize});
        }
    }
}

void TestAPagedPlanSizesAttentionFromFlatMhaAtPositionZero() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto config = E2bConfigForTest();
    ShipFullGrid(config);
    ShipPagedChunkOnly(config);
    const auto plan = Gemma4ShapePlan::Build(api, stream, config, 2 * flm::gemma4::kPageSize);
    TEST_REQUIRE(plan.ForRows(19, /*swa=*/true).flat_mha_rows == 64);
    TEST_REQUIRE(plan.ForRows(100, /*swa=*/false).flat_mha_rows == 128);
    TEST_REQUIRE(plan.ForRows(4096, /*swa=*/true).flat_mha_rows == 4096);
}

void TestAPagedPlanRefusesAGridWithoutThePageSizedChunk() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto config = E2bConfigForTest();
    ShipFullGrid(config);
    for (const auto head : {config.head_dim, config.global_head_dim}) {
        fake_corelib::GetState().shipped_paged.push_back(
            {config.q_heads, config.kv_heads, 1, head, flm::gemma4::kPageSize});
    }
    std::string error;
    try {
        (void)Gemma4ShapePlan::Build(api, stream, config, 2 * flm::gemma4::kPageSize);
    } catch (const std::runtime_error& e) {
        error = e.what();
    }
    RequireContains(error, "paged_mha chunk has no kernel");
}

void TestPlanRejectsAGridWithoutItsExactKAndN() {
    const auto config = E2bConfigForTest();
    RequireGridBreaks(config, [&config](auto& state) {
        for (auto& shipped : state.shipped_matmul)
            if (shipped.k == config.hidden && shipped.n == config.q_heads * config.head_dim)
                shipped.n += 64;
    }, "query");
    RequireGridBreaks(config, [&config](auto& state) {
        for (auto& shipped : state.shipped_matmul)
            if (shipped.n == config.hidden) shipped.k += 32;
    }, "output");
}

/// \brief a window that runs past the allocation it is carved from is refused
///
/// Gemma 4 needs this one specifically: its activation buffers are allocated
/// once at the WIDER geometry and windowed down per layer, so the arithmetic
/// that produces these offsets is real per-layer code, not a hypothetical.
void TestFakeRejectsAWindowThatDoesNotFitItsParent() {
    fake_corelib::Reset();
    auto api = Api();
    const auto& fns = api->functions();

    const std::int64_t parent_shape[] = {4, 8};  // 32 elements
    ryzenai_corelib_tensor_ptr parent = nullptr;
    TEST_REQUIRE(fns.create_device_tensor(ryzenai_corelib_data_type_bf16,
                                          parent_shape, 2, &parent) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(parent != nullptr);

    // 16 elements starting 16 in: exactly reaches the end, so it is legal.
    const std::int64_t window_shape[] = {2, 8};
    ryzenai_corelib_tensor_ptr fits = nullptr;
    TEST_REQUIRE(fns.create_tensor_window(parent, window_shape, 2, 16, nullptr, 0, &fits) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(fits != nullptr);

    // The same shape 8 elements further along runs 8 past the end.
    ryzenai_corelib_tensor_ptr overruns = nullptr;
    TEST_REQUIRE(fns.create_tensor_window(parent, window_shape, 2, 24, nullptr, 0, &overruns) !=
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(overruns == nullptr);

    fns.object_release(fits);
    fns.object_release(parent);
}

/// \brief tensor_write / tensor_read are bounded by the tensor's OWN extent
///
/// corelib.h, on tensor_get_byte_size: "The tensor's own size, which is what
/// write/read are bounded by. The underlying allocation is page-rounded and
/// may be larger." So the bound is the declared shape, not the allocation.
void TestFakeBoundsChecksTensorWriteAndRead() {
    fake_corelib::Reset();
    auto api = Api();
    const auto& fns = api->functions();

    const std::int64_t shape[] = {4, 8};  // 32 elements
    ryzenai_corelib_tensor_ptr tensor = nullptr;
    TEST_REQUIRE(fns.create_device_tensor(ryzenai_corelib_data_type_bf16, shape,
                                          2, &tensor) ==
                 ryzenai_corelib_status_success);

    std::vector<std::uint16_t> source(8, 0);
    TEST_REQUIRE(fns.tensor_write(tensor, ryzenai_corelib_data_type_bf16,
                                  source.data(), 8, 24) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(fns.tensor_write(tensor, ryzenai_corelib_data_type_bf16,
                                  source.data(), 8, 28) !=
                 ryzenai_corelib_status_success);

    std::vector<std::uint16_t> destination(8, 0);
    TEST_REQUIRE(fns.tensor_read(tensor, ryzenai_corelib_data_type_bf16,
                                 destination.data(), 8, 24) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(fns.tensor_read(tensor, ryzenai_corelib_data_type_bf16,
                                 destination.data(), 8, 28) !=
                 ryzenai_corelib_status_success);

    fns.object_release(tensor);
}

/// \brief ssmlp refuses two operands that are windows over ONE allocation
///
/// corelib.h states "all four are separate buffers" for this op, and on
/// Gemma 4 the requirement is MORE load-bearing than on Phi-4, not less: with
/// `post_feedforward_layernorm` set the kernel never writes the `normalized`
/// plane at all yet still requires a distinct buffer for that port -- a port
/// that is write-only in name is exactly the one a reader is tempted to alias.
///
/// The load-bearing half of this test is the `distinct handles` assertion: the
/// two aliasing operands are DIFFERENT `void*`s, so a check that compared the
/// handles would accept this dispatch. Only comparing backing storage catches
/// it.
void TestFakeRejectsSsmlpOperandsThatAliasOneAllocation() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto& fns = api->functions();

    // Teach the fake which row counts the ssmlp helper produces, the same way
    // the shape plan does, so these dispatches are on-grid and the only thing
    // left to reject is the aliasing.
    std::int64_t rows = 64;
    const ryzenai_corelib_ssmlp_bf16_weights_desc desc{1536, 6144, 32, 1, 1};
    TEST_REQUIRE(fake_corelib::NoteSsmlpRows(stream, &rows, &desc) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(rows == 64);

    const std::int64_t shape[] = {64, 1536};
    auto make = [&] {
        ryzenai_corelib_tensor_ptr tensor = nullptr;
        fns.create_device_tensor(ryzenai_corelib_data_type_bf16, shape, 2, &tensor);
        return tensor;
    };
    auto* input = make();
    auto* residual = make();
    auto* skip_sum = make();
    auto* normalized = make();

    TEST_REQUIRE(fns.ssmlp(stream, input, residual, nullptr, skip_sum,
                           normalized) == ryzenai_corelib_status_success);

    // One allocation, two windows onto it -- one at the front, one behind it.
    const std::int64_t parent_shape[] = {128, 1536};
    ryzenai_corelib_tensor_ptr parent = nullptr;
    TEST_REQUIRE(fns.create_device_tensor(ryzenai_corelib_data_type_bf16,
                                          parent_shape, 2, &parent) ==
                 ryzenai_corelib_status_success);
    ryzenai_corelib_tensor_ptr front = nullptr;
    ryzenai_corelib_tensor_ptr back = nullptr;
    TEST_REQUIRE(fns.create_tensor_window(parent, shape, 2, 0, nullptr, 0, &front) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(fns.create_tensor_window(parent, shape, 2, 64 * 1536, nullptr, 0, &back) ==
                 ryzenai_corelib_status_success);
    // THE POINT: distinct handles, one allocation.
    TEST_REQUIRE(front != back);

    TEST_REQUIRE(fns.ssmlp(stream, input, residual, nullptr, front, back) !=
                 ryzenai_corelib_status_success);

    for (auto* object : {input, residual, skip_sum, normalized, front, back, parent})
        fns.object_release(object);
}

/// \brief a dispatch presenting a row count no pad helper produced is refused
///
/// This is the per-operand check (`OnGrid`), not the operands-agree one: all
/// four operands below agree with each other and are still wrong, because 100
/// is not a value the ssmlp helper ever answered. See fake_corelib.hpp for the
/// bucket-level blindness this check does NOT remove.
void TestFakeRejectsADispatchAtARowCountNoHelperProduced() {
    fake_corelib::Reset();
    auto api = Api();
    auto* stream = fake_corelib::MakeStreamForTest();
    const auto& fns = api->functions();

    std::int64_t rows = 64;
    const ryzenai_corelib_ssmlp_bf16_weights_desc desc{1536, 6144, 32, 1, 1};
    TEST_REQUIRE(fake_corelib::NoteSsmlpRows(stream, &rows, &desc) ==
                 ryzenai_corelib_status_success);

    auto make = [&](std::int64_t m) {
        const std::int64_t shape[] = {m, 1536};
        ryzenai_corelib_tensor_ptr tensor = nullptr;
        fns.create_device_tensor(ryzenai_corelib_data_type_bf16, shape, 2, &tensor);
        return tensor;
    };
    auto* a = make(100);
    auto* b = make(100);
    auto* c = make(100);
    auto* d = make(100);
    TEST_REQUIRE(fns.ssmlp(stream, a, b, nullptr, c, d) !=
                 ryzenai_corelib_status_success);
    for (auto* object : {a, b, c, d}) fns.object_release(object);
}
// -- the ABI surface ------------------------------------------------------
//
// NOT ABOUT THE SHAPE PLAN. These two live here because this is the only
// gemma4_rai target that links the adapter (src/common/rai/corelib_api.cpp)
// and the fake, and because it needs no XRT or Boost, so it is the target
// that always builds. Their subject is "FLM_CORELIB_FUNCTIONS reaches the
// part of corelib 0.5.0 Gemma 4 actually uses" -- one reviewable claim,
// rather than six separate discoveries at six separate call sites.

/// \brief every entry point the adapter declares is resolvable
/// \note Generated FROM the macro, so a row added without a matching fake
///       function fails here rather than at the call site that needed it.
///       Deliberately NOT a hardcoded count: this target should not have to
///       be edited every time a row is added, and the phi4 suite already
///       pins the count itself (test_corelib_api.cpp).
void TestEveryDeclaredCorelibEntryPointResolves() {
    fake_corelib::Reset();
    const auto api = Api();
    const auto& functions = api->functions();
#define FLM_REQUIRE_RESOLVED(member, symbol) TEST_REQUIRE(functions.member != nullptr);
    FLM_CORELIB_FUNCTIONS(FLM_REQUIRE_RESOLVED)
#undef FLM_REQUIRE_RESOLVED
}

/// \brief the four symbols Gemma 4 needs and Phi-4 never did
///
/// BY NAME, not through the struct. `FLM_CORELIB_FUNCTIONS` is what decides
/// which symbols exist, so the test above cannot notice a missing ROW -- it
/// would simply iterate a shorter list and pass. Naming the four here is what
/// makes "the adapter binds ple and rmsnorm" a claim that can fail.
///
/// `ple_bf16_weights_create` and `rmsnorm_bf16_weights_create` both pack a
/// fresh weight and rebind a cached one; `ple_bf16` and `rmsnorm_bf16` are
/// the dispatches the layer loop runs. `add_rmsnorm_bf16` is deliberately
/// absent -- see corelib_api.hpp for why.
void TestTheEntryPointsGemma4AddedAreBoundByName() {
    auto resolver = fake_corelib::Resolver();
    for (const auto* symbol : {"ryzenai_corelib_ple_bf16_weights_create",
                               "ryzenai_corelib_ple_bf16",
                               "ryzenai_corelib_rmsnorm_bf16_weights_create",
                               "ryzenai_corelib_rmsnorm_bf16"}) {
        if (resolver(symbol) == nullptr)
            throw std::runtime_error(std::string("corelib entry point not bound: ") +
                                     symbol);
    }
    // And the one that is NOT bound, in the same test, so "we bound what we
    // needed" and "we did not bind what we do not use" cannot drift apart.
    for (const auto* symbol : {"ryzenai_corelib_add_rmsnorm_bf16"}) {
        if (resolver(symbol) != nullptr)
            throw std::runtime_error(std::string("corelib entry point bound with no "
                                                 "caller: ") + symbol);
    }
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestEveryDeclaredCorelibEntryPointResolves);
    RUN_TEST(TestTheEntryPointsGemma4AddedAreBoundByName);
    RUN_TEST(TestPlanBuildsBothGeometriesWithDifferentHeadSizes);
    RUN_TEST(TestPlanPassesTheStreamToEveryKernelQuery);
    RUN_TEST(TestSsmlpIsQueriedInTheGemmaFamilyAtItsGroup);
    RUN_TEST(TestSsmlpIsPlannedAtBothOfE2bsFfnWidths);
    RUN_TEST(TestAttentionScaleIsExactlyOneEverywhere);
    RUN_TEST(TestPlanHoldsFourDescriptorsKeyedByGeometryAndCacheRole);
    RUN_TEST(TestRopeDimIsFullRotaryOnBothGeometries);
    RUN_TEST(TestEveryProjectionQueriesAtGroupAndOnlyLmHeadAtHeadGroup);
    RUN_TEST(TestLmHeadIsPlannedOnceAtOneRowAndAtHeadGroup);
    RUN_TEST(TestLmHeadDispatchAfterAFullPlanBuildIsAccepted);
    RUN_TEST(TestDispatchRecordsCarryTheDescriptorTheyRanWith);
    RUN_TEST(TestE4bGroupDirectionAndLmHeadPlan);
    RUN_TEST(TestPleRowsRoundUpToAShippedBucket);
    RUN_TEST(TestAPagedPlanSizesAttentionFromFlatMhaAtPositionZero);
    RUN_TEST(TestAPagedPlanRefusesAGridWithoutThePageSizedChunk);
    RUN_TEST(TestPlanRejectsAGridWithoutItsExactKAndN);
    RUN_TEST(TestFakeRejectsAWindowThatDoesNotFitItsParent);
    RUN_TEST(TestFakeBoundsChecksTensorWriteAndRead);
    RUN_TEST(TestFakeRejectsSsmlpOperandsThatAliasOneAllocation);
    RUN_TEST(TestFakeRejectsADispatchAtARowCountNoHelperProduced);
#undef RUN_TEST
}
