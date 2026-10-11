#include "models/gemma4/rai/aie_next/gemma4_rai.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_host.hpp"
#include "rai/gguf_file.hpp"
#include "rai/weight_cache.hpp"
#include "rai/weight_source.hpp"

#include "fake_corelib.hpp"
#include "gemma4_gguf_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
using flm::corelib::CorelibApi;
using flm::corelib::CorelibRuntime;
using flm::gemma4::ConfigFromMetadata;
using flm::gemma4::Gemma4Config;
using flm::gemma4::Gemma4GgufPackage;
using flm::gemma4::gemma4_rai;
using flm::rai::GgufFile;

/// \brief a matmul source set: one Q8_0 qweight over `blocks`
std::vector<ryzenai_corelib_weights_source> MatmulSources(const std::vector<std::byte>& blocks) {
    return {flm::corelib::GgufQ8(ryzenai_corelib_weights_role_qweight, blocks.data(),
                                 blocks.size(), 1, 32)};
}

/// \brief an ssmlp source set whose six components all read `blocks`
std::vector<ryzenai_corelib_weights_source> SsMlpSources(const std::vector<std::byte>& blocks) {
    std::vector<ryzenai_corelib_weights_source> sources;
    for (const auto role : {ryzenai_corelib_weights_role_epsilon, ryzenai_corelib_weights_role_norm0,
                            ryzenai_corelib_weights_role_norm1})
        sources.push_back(flm::corelib::Bf16(role, blocks.data(), 1));
    for (const auto role : {ryzenai_corelib_weights_role_gate_qweight,
                            ryzenai_corelib_weights_role_up_qweight,
                            ryzenai_corelib_weights_role_down_qweight})
        sources.push_back(flm::corelib::GgufQ8(role, blocks.data(), blocks.size(), 1, 32));
    return sources;
}

// ---------------------------------------------------------------------------
// The two fixtures, written ONCE each.
//
// A full E2B fixture is ~5 GB and an E4B one ~8.6 GB, and this file opens one
// per test case. Writing a fresh file per case would cost minutes of wall
// clock and hundreds of gigabytes of churn, so these are PINNED (see
// Builder::Write's `pinned` parameter): exempt from the per-write sweep, but
// still removed when this process returns from main and by the next run's
// startup sweep if it does not. Two files, not one per test.
//
// The Builder is kept beside the path because it is also the ONLY source of
// the per-tensor signature constants the assertions below compare against --
// SignatureOf is a function of a tensor's position, so a Builder made from
// the same options agrees with the file even though it is not the same
// object.

struct Fixture {
    gemma4_fixture::Builder builder;
    std::filesystem::path path;
};

const Fixture& E2bFixture() {
    static const Fixture fixture = [] {
        auto builder = gemma4_fixture::MakeBuilder(gemma4_fixture::E2bOptions());
        auto path = builder.Write("engine-e2b", /*pinned=*/true);
        return Fixture{std::move(builder), std::move(path)};
    }();
    return fixture;
}

const Fixture& E4bFixture() {
    static const Fixture fixture = [] {
        auto builder = gemma4_fixture::MakeBuilder(gemma4_fixture::E4bOptions());
        auto path = builder.Write("engine-e4b", /*pinned=*/true);
        return Fixture{std::move(builder), std::move(path)};
    }();
    return fixture;
}

/// \brief scoped FLM_RAI_WEIGHT_CACHE, restored on the way out
struct ScopedWeightCache {
    std::string previous;
    bool had_previous{};
    explicit ScopedWeightCache(const std::string& value) {
        if (const char* existing = std::getenv("FLM_RAI_WEIGHT_CACHE")) {
            previous = existing;
            had_previous = true;
        }
        _putenv_s("FLM_RAI_WEIGHT_CACHE", value.c_str());
    }
    ~ScopedWeightCache() {
        if (had_previous) _putenv_s("FLM_RAI_WEIGHT_CACHE", previous.c_str());
        else _putenv_s("FLM_RAI_WEIGHT_CACHE", "");
    }
};

/// \brief tears the process-wide corelib runtime down however the scope ends
///
/// WITHOUT THIS, ONE FAILING TEST BREAKS EVERY LATER ONE. `TEST_REQUIRE`
/// throws, so a hand-written "engine.reset(); runtime.reset();
/// ShutdownProcess();" tail at the bottom of a test is skipped the moment an
/// assertion fires, and the next test reports "corelib runtime already
/// exists" instead of its own result. That was not hypothetical: it is what
/// mutation-testing this file produced -- one genuine failure followed by
/// seven bogus ones, which is precisely the shape that makes a red run
/// unreadable.
///
/// \note Declare it FIRST in its scope so it is destroyed LAST, after the
///       engine, package and runtime locals have released their handles.
struct RuntimeScope {
    RuntimeScope() = default;
    RuntimeScope(const RuntimeScope&) = delete;
    RuntimeScope& operator=(const RuntimeScope&) = delete;
    /// \note Swallows, because `ShutdownProcess` THROWS when corelib objects
    ///       are still live and this destructor can run while a failed
    ///       assertion is unwinding -- and a destructor that throws during
    ///       unwinding is std::terminate, which would replace a readable
    ///       FAIL line with a silent abort.
    ~RuntimeScope() {
        try {
            CorelibRuntime::ShutdownProcess();
        } catch (...) {
        }
    }
};

/// \brief one engine over one fixture, with the weight cache OFF
/// \note Off unless a test asks for it. A cache written by one test would
///       otherwise make every later one BIND instead of PACK, and every
///       assertion about descriptors -- which is nearly all of them -- would
///       silently stop testing anything.
struct Harness {
    ScopedWeightCache no_cache{"0"};
    std::shared_ptr<CorelibRuntime> runtime;
    std::shared_ptr<Gemma4GgufPackage> package;
    std::unique_ptr<gemma4_rai> engine;
    Gemma4Config shape;

    explicit Harness(const Fixture& fixture,
                     const std::function<void(Gemma4Config&)>& adjust = {},
                     std::uint32_t maximum = 4096) {
        fake_corelib::Reset();
        runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        try {
            package = Gemma4GgufPackage::Open(fixture.path);
            shape = package->Config();
            if (adjust) adjust(shape);
            engine = gemma4_rai::CreateWithShapeForTest(package, runtime,
                                                        LM_Config{}, shape, maximum);
        } catch (...) {
            // A CONSTRUCTOR THAT THROWS NEVER RUNS ITS DESTRUCTOR, so without
            // this the process-wide runtime survives the failed test and
            // every later one reports "corelib runtime already exists". See
            // RuntimeScope.
            Teardown();
            throw;
        }
    }
    ~Harness() { Teardown(); }

    /// \note Swallows for the reason RuntimeScope's destructor does.
    void Teardown() noexcept {
        try {
            engine.reset();
            package.reset();
            runtime.reset();
            CorelibRuntime::ShutdownProcess();
        } catch (...) {
        }
    }
};

std::string Blk(std::int64_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + suffix;
}

// ---------------------------------------------------------------------------
// The ABI shim, and the refusals.

void TestLoadWeightsThrowsAndNamesGguf() {
    Harness harness(E2bFixture());
    const auto error = RequireThrows([&] {
        Q4NX* nothing = nullptr;
        harness.engine->load_weights(*nothing);
    });
    RequireContains(error, "GGUF");
    // It is an ABI shim for a FROZEN base class, not a missing feature: the
    // message has to say where weights actually come from, or the next reader
    // implements it.
    RequireContains(error, "Gemma 4");
}

void TestStreamEnablesTheGemma4BundleBeforeAnyKernelQuery() {
    Harness h(E2bFixture());
    const auto& state = fake_corelib::GetState();
    TEST_REQUIRE(state.kernel_bundles.size() == 1);
    TEST_REQUIRE(state.kernel_bundles[0].model == "gemma4");
    TEST_REQUIRE(state.kernel_bundles[0].enum_calls_before == 0);
    TEST_REQUIRE(!state.enum_calls.empty());
}

void TestAFreshEngineIsNotPoisonedAndStartsAtPositionZero() {
    Harness harness(E2bFixture());
    TEST_REQUIRE(!harness.engine->poisoned());
    TEST_REQUIRE(harness.engine->get_current_context_length() == 0);
    // UNTIL TASK C8 THIS ASSERTED THAT forward AND prefill REFUSE, naming C8
    // in the message -- an engine that produced logits from no attention at
    // all would have been exactly the fluent-and-wrong failure this port
    // guards against. The layer loop exists now, so what is left to say here
    // is the state a fresh engine is in; what the loop DOES is asserted by the
    // dispatch-sequence tests further down.
    std::vector<int> ids{1, 2, 3};
    TEST_REQUIRE(harness.engine->prefill(ids).size() ==
                 static_cast<std::size_t>(harness.shape.vocab));
    TEST_REQUIRE(harness.engine->get_current_context_length() == 3);
    TEST_REQUIRE(!harness.engine->poisoned());
}

// ---------------------------------------------------------------------------
// The weight inventory.

/// \brief how many weight objects each row must hold, from the driver's rule
/// \note Ten on a layer that OWNS its KV cache, six on one that SHARES, plus
///       layer 0's standalone norm and lm_head. Written as the arithmetic
///       rather than as 272/350 so a reader can see WHICH numbers produce it.
std::size_t ExpectedSlots(const Gemma4Config& shape) {
    const auto owning = static_cast<std::size_t>(shape.kv_layers);
    const auto sharing = static_cast<std::size_t>(shape.layers) - owning;
    return 1 + owning * 10 + sharing * 6 + 1;
}

void RequireWeightInventory(const Fixture& fixture, std::size_t expected_slots,
                            std::size_t expected_matmuls) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    const auto& state = fake_corelib::GetState();

    TEST_REQUIRE(ExpectedSlots(shape) == expected_slots);
    TEST_REQUIRE(harness.engine->weight_slot_count() == expected_slots);
    TEST_REQUIRE(state.matmul_weights_creates.size() == expected_matmuls);
    // One ssmlp and one ple PER LAYER, on every layer -- sharing or not. What
    // a sharing layer drops is K and V (and their norms), never its MLP or
    // its per-layer embedding block.
    TEST_REQUIRE(state.ssmlp_weights_creates.size() ==
                 static_cast<std::size_t>(shape.layers));
    TEST_REQUIRE(state.ple_create_calls.size() ==
                 static_cast<std::size_t>(shape.layers));
    // 1 standalone + q_norm and post_attention_norm per layer + k_norm and
    // V's ones-norm per OWNING layer.
    TEST_REQUIRE(state.rmsnorm_weights_creates.size() ==
                 1 + 2 * static_cast<std::size_t>(shape.layers) +
                     2 * static_cast<std::size_t>(shape.kv_layers));
    // Everything packed, nothing bound from a file.
    for (const auto& call : state.matmul_weights_creates) TEST_REQUIRE(!call.from_file);
    for (const auto& call : state.ssmlp_weights_creates) TEST_REQUIRE(!call.from_file);
    for (const auto& call : state.rmsnorm_weights_creates) TEST_REQUIRE(!call.from_file);
    TEST_REQUIRE(!harness.engine->loaded_from_cache());
}

void TestE2bHoldsTenWeightsPerOwningLayerAndSixPerSharingOne() {
    // 1 + 15*10 + 20*6 + 1. The matmul count is q(35) + o(35) + k(15) + v(15)
    // + lm_head(1).
    RequireWeightInventory(E2bFixture(), 272, 101);
}

void TestE4bHoldsTenWeightsPerOwningLayerAndSixPerSharingOne() {
    // 1 + 24*10 + 18*6 + 1. NEITHER NUMBER FOLLOWS FROM E2B'S: E4B has more
    // layers AND more owning ones (24 of 42 against 15 of 35), so the split
    // moves in both directions at once.
    RequireWeightInventory(E4bFixture(), 350, 133);
}

// ---------------------------------------------------------------------------
// group vs head_group.

/// \brief every projection at `group`, lm_head alone at `head_group`
///
/// PINNED IN BOTH DIRECTIONS, and only possible because the harness can
/// override the shape. `group`, `head_group` and `ple_group` are all 32 on
/// both shipped rows, so against a real config this test would pass whichever
/// field the engine used. `head_group = 64` pulls them apart, and then:
///
///   - using `head_group` for a projection puts more than one create at 64,
///   - using `group` for lm_head puts none at 64,
///   - and swapping the two puts 100 at 64 and one at 32.
///
/// All three fail here. At C6 this distinction was inert -- it only decided
/// which padding query was made. Here it stops being inert: a weight
/// requantized at the wrong group size is a wrong weight, not a wrong
/// allocation.
void RequireGroupDirection(const Fixture& fixture) {
    Harness harness(fixture, [](Gemma4Config& shape) { shape.head_group = 64; });
    const auto& shape = harness.shape;
    TEST_REQUIRE(shape.head_group != shape.group);
    const auto& creates = fake_corelib::GetState().matmul_weights_creates;
    TEST_REQUIRE(!creates.empty());

    std::size_t at_head_group = 0;
    for (const auto& call : creates) {
        if (call.desc.group_size == shape.head_group) {
            ++at_head_group;
            // The one at head_group is lm_head and nothing else: hidden ->
            // vocab, TIED to token_embd (this conversion has no
            // output.weight on either row).
            TEST_REQUIRE(call.desc.k == shape.hidden);
            TEST_REQUIRE(call.desc.n == shape.vocab);
        } else {
            TEST_REQUIRE(call.desc.group_size == shape.group);
            TEST_REQUIRE(call.desc.n != shape.vocab);
        }
        // No projection in this model carries a bias.
        TEST_REQUIRE(!call.desc.has_bias);
    }
    TEST_REQUIRE(at_head_group == 1);

    // And the shapes themselves, since a group-size assertion says nothing
    // about whether the right tensor went in. Q is hidden -> q_heads*head,
    // K and V are hidden -> kv_heads*head, O is q_heads*head -> hidden, and
    // the head width is the LAYER'S own -- 256 sliding, 512 full.
    std::size_t q_creates = 0, o_creates = 0, kv_creates = 0;
    for (const auto& call : creates) {
        if (call.desc.n == shape.vocab) continue;
        const auto sliding_q = shape.q_heads * shape.head_dim;
        const auto full_q = shape.q_heads * shape.global_head_dim;
        const auto sliding_kv = shape.kv_heads * shape.head_dim;
        const auto full_kv = shape.kv_heads * shape.global_head_dim;
        if (call.desc.k == shape.hidden &&
            (call.desc.n == sliding_q || call.desc.n == full_q)) {
            ++q_creates;
        } else if (call.desc.k == shape.hidden &&
                   (call.desc.n == sliding_kv || call.desc.n == full_kv)) {
            ++kv_creates;
        } else if (call.desc.n == shape.hidden &&
                   (call.desc.k == sliding_q || call.desc.k == full_q)) {
            ++o_creates;
        } else {
            throw std::runtime_error("matmul create at an unrecognized shape: k=" +
                                     std::to_string(call.desc.k) + " n=" +
                                     std::to_string(call.desc.n));
        }
    }
    TEST_REQUIRE(q_creates == static_cast<std::size_t>(shape.layers));
    TEST_REQUIRE(o_creates == static_cast<std::size_t>(shape.layers));
    TEST_REQUIRE(kv_creates == 2 * static_cast<std::size_t>(shape.kv_layers));
}

void TestE2bPacksEveryProjectionAtGroupAndLmHeadAloneAtHeadGroup() {
    RequireGroupDirection(E2bFixture());
}

void TestE4bPacksEveryProjectionAtGroupAndLmHeadAloneAtHeadGroup() {
    RequireGroupDirection(E4bFixture());
}

// ---------------------------------------------------------------------------
// ssmlp.

/// \brief every ssmlp at `group`, gelu, and with the post-feedforward norm
///        SET -- and with THIS LAYER'S two norms
void RequireSsmlp(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    const auto& builder = fixture.builder;
    const auto& creates = fake_corelib::GetState().ssmlp_weights_creates;
    TEST_REQUIRE(creates.size() == static_cast<std::size_t>(shape.layers));

    for (const auto& call : creates) {
        TEST_REQUIRE(call.desc.k == shape.hidden);
        TEST_REQUIRE(call.desc.group_size == shape.group);
        // `activation = 1` is gelu. A DECLARATION, not a switch -- it is
        // baked into the ELF and appears nowhere in the artifact's name --
        // but it is the only place the block's activation is written down.
        TEST_REQUIRE(call.desc.activation == 1);
        // This one genuinely changes what the block computes: set, the norm
        // moves inside the residual and `normalized` is never written. It is
        // what keys the `ssmlp_gemma_fusion_no_rms1_*` ELFs.
        TEST_REQUIRE(call.desc.post_feedforward_layernorm == 1);
        TEST_REQUIRE(call.has_components);
    }

    // PER LAYER, NOT SPOT-CHECKED. norm0 is THIS layer's `ffn_norm` and
    // norm1 THIS layer's `post_ffw_norm` -- neither belongs to another
    // layer, unlike the silu family (which is what Phi-4 is) whose norm1 is
    // the next block's. Carrying Phi-4's assignment over is silent, so each
    // layer's pair is matched by the fixture's per-tensor signature rather
    // than by counting.
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const auto norm0 = builder.SignatureOfBf16(Blk(layer, ".ffn_norm.weight"));
        const auto norm1 = builder.SignatureOfBf16(Blk(layer, ".post_ffw_norm.weight"));
        const auto width = shape.layer_intermediate[index];
        const auto found = std::count_if(
            creates.begin(), creates.end(), [&](const auto& call) {
                return call.norm0_bf16 == norm0 && call.norm1_bf16 == norm1 &&
                       call.desc.n == width;
            });
        TEST_REQUIRE(found == 1);
    }

    // WHICH BLOCK STREAM REACHED WHICH PORT, PER LAYER.
    //
    // LEAD CASE FOR THIS WHOLE CLASS OF CHECK, and it is not hypothetical: a
    // gated MLP puts `gate` through the activation and `up` not, so
    // exchanging the two computes GELU(x @ up) * (x @ gate) -- a different
    // function. `ffn_gate` and `ffn_up` have the SAME shape on every Gemma 4
    // layer and travel to corelib as bare `const void*` block streams of the
    // same quant type, so neither the descriptor, nor a length check, nor
    // anything else in the ABI can tell them apart. Exchanging them was
    // measured to leave this suite 16/16 green.
    //
    // The fixture's per-tensor block tag is the only discriminator there is
    // (gemma4_gguf_fixture.hpp's BlockSignatureAt). Keyed on norm0, which
    // names the layer.
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const auto norm0 = builder.SignatureOfBf16(Blk(layer, ".ffn_norm.weight"));
        std::size_t matched = 0;
        for (const auto& call : creates) {
            if (call.norm0_bf16 != norm0) continue;
            ++matched;
            TEST_REQUIRE(call.gate_tag == builder.BlockSignatureOf(Blk(layer, ".ffn_gate.weight")));
            TEST_REQUIRE(call.up_tag == builder.BlockSignatureOf(Blk(layer, ".ffn_up.weight")));
            TEST_REQUIRE(call.down_tag == builder.BlockSignatureOf(Blk(layer, ".ffn_down.weight")));
        }
        TEST_REQUIRE(matched == 1);
    }

    // And the widths, which MOVE BETWEEN LAYERS on E2B (6144 for layers
    // 0-14, 12288 from 15 on) and do not on E4B. Read off ffn_gate's own
    // shape, never off gemma4.feed_forward_length.
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const auto norm0 = builder.SignatureOfBf16(Blk(layer, ".ffn_norm.weight"));
        for (const auto& call : creates) {
            if (call.norm0_bf16 != norm0) continue;
            TEST_REQUIRE(call.desc.n == shape.layer_intermediate[index]);
        }
    }
}

void TestE2bSsmlpDeclaresGeluAndKeepsBothOfItsOwnNorms() { RequireSsmlp(E2bFixture()); }
void TestE4bSsmlpDeclaresGeluAndKeepsBothOfItsOwnNorms() { RequireSsmlp(E4bFixture()); }

// ---------------------------------------------------------------------------
// ple: one create per layer, the group, and the NEXT layer's norm.

/// \brief find the pack calls belonging to one layer
/// \note KEYED ON `layer_scale`, not on call order. The engine packs across
///       eight threads, so the recorded order interleaves layers.
///       `layer_output_scale` is a one-element F32 tensor and the fixture
///       gives every F32 tensor a distinct constant, so it names the layer
///       exactly.
std::vector<std::size_t> PackCallsForLayer(float layer_scale) {
    std::vector<std::size_t> indices;
    const auto& calls = fake_corelib::GetState().ple_pack_calls;
    for (std::size_t i = 0; i < calls.size(); ++i)
        if (calls[i].desc.layer_scale == layer_scale) indices.push_back(i);
    return indices;
}

void RequirePle(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    const auto& builder = fixture.builder;
    const auto& state = fake_corelib::GetState();

    TEST_REQUIRE(state.ple_create_calls.size() ==
                 static_cast<std::size_t>(shape.layers));
    TEST_REQUIRE(state.ple_pack_calls.size() ==
                 static_cast<std::size_t>(shape.layers));

    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const auto scale =
            builder.SignatureOf(Blk(layer, ".layer_output_scale.weight"));
        const auto indices = PackCallsForLayer(scale);
        // ONE CREATE PER LAYER. corelib packs the four arrays and creates
        // the weights in a single call.
        TEST_REQUIRE(indices.size() == 1);
        const auto& full = state.ple_pack_calls[indices[0]];
        TEST_REQUIRE(!full.sizing_leg);
        TEST_REQUIRE(full.read_any_array);
        TEST_REQUIRE(full.reported_size == fake_corelib::PlePackedSize(full.desc));

        // The descriptor: k is hidden, n is the per-layer embedding width
        // (256 on both rows), and the group is `ple_group` -- NOT `group`.
        // 32 is the only value mladfple ships and the fake refuses anything
        // else, so an engine that passed `group` here would fail outright on
        // a row where they differ.
        TEST_REQUIRE(full.desc.k == shape.hidden);
        TEST_REQUIRE(full.desc.n == shape.ple_dim);
        TEST_REQUIRE(full.desc.group_size == shape.ple_group);
        TEST_REQUIRE(full.desc.epsilon == shape.eps);

        // WHICH MATRIX WENT INTO WHICH PORT. gate is [k, n] (inp_gate) and
        // proj is [n, k]; corelib.h says passing one the other way round
        // "packs silently and produces noise".
        const auto gate_name = Blk(layer, ".inp_gate.weight");
        const auto proj_name = Blk(layer, ".proj.weight");
        TEST_REQUIRE(full.gate_first == builder.SignatureOf(gate_name));
        TEST_REQUIRE(full.proj_first == builder.SignatureOf(proj_name));

        // AND IN WHICH ORIENTATION, which the two assertions above cannot
        // say: element 0 is element 0 under a transpose too.
        //
        // corelib wants `gate` as FP32 [k, n] ROW-MAJOR -- k = hidden
        // contiguous rows of n = ple_dim -- and `proj` as [n, k] row-major
        // (corelib.h's ple section). The GGUF maps each the other way round:
        // `inp_gate` is ShapeOf {ple_dim, hidden}, i.e. ple_dim rows of
        // hidden, and `proj` is {hidden, ple_dim}. So the engine must
        // materialise the TRANSPOSE of the mapping for both, exactly as the
        // reference driver does -- `floats()` returns
        // `reshape(reversed(dims)).T` and `_ple_weights` wraps both in
        // `np.ascontiguousarray` because that view "is not C-contiguous"
        // (gemma4_driver.py:1054-1081, :1494-1520).
        //
        // Element counts are IDENTICAL either way (1536x256 against
        // 256x1536), so nothing rejects the wrong one: corelib's own verdict
        // is that it "packs silently and produces noise". These four
        // assertions are the only thing in this suite that can see it.
        //
        // gate flat 1 is (in 0, out 1), which is the mapping's (out 1, in 0)
        // -- mapping element 1*hidden. gate flat n is (in 1, out 0), the
        // mapping's (out 0, in 1) -- mapping element 1.
        TEST_REQUIRE(full.gate_at_1 ==
                     builder.ElementAt(gate_name, static_cast<std::uint64_t>(shape.hidden)));
        TEST_REQUIRE(full.gate_at_n == builder.ElementAt(gate_name, 1));
        // proj flat 1 is (in 0, out 1) = the mapping's element 1*ple_dim;
        // proj flat k is (in 1, out 0) = the mapping's element 1.
        TEST_REQUIRE(full.proj_at_1 ==
                     builder.ElementAt(proj_name, static_cast<std::uint64_t>(shape.ple_dim)));
        TEST_REQUIRE(full.proj_at_k == builder.ElementAt(proj_name, 1));

        // The inner norm is THIS layer's post_norm -- the fifth per-layer
        // norm, whose name was guessed wrong once already.
        TEST_REQUIRE(full.post_norm.size() == static_cast<std::size_t>(shape.hidden));
        const auto post = builder.SignatureOf(Blk(layer, ".post_norm.weight"));
        TEST_REQUIRE(full.post_norm.front() == post);
        TEST_REQUIRE(full.post_norm.back() == post);

        // AND THE ONE THAT STILL GENERATES TEXT WHEN IT IS WRONG. next_norm
        // is the NEXT layer's attention-norm gamma, because ple's trailing
        // RMSNorm crosses the block boundary; the LAST layer's is
        // output_norm, the model's final norm, since there is no
        // blk.{layers}.attn_norm. Asserted for EVERY layer: an off-by-one
        // here normalizes every layer by its neighbour's gamma, which is
        // finite, plausible and wrong, and a spot check at layer 0 would not
        // see a rotation.
        const auto expected_next = builder.SignatureOf(
            layer == shape.layers - 1 ? std::string("output_norm.weight")
                                      : Blk(layer + 1, ".attn_norm.weight"));
        TEST_REQUIRE(full.next_norm.size() == static_cast<std::size_t>(shape.hidden));
        TEST_REQUIRE(full.next_norm.front() == expected_next);
        TEST_REQUIRE(full.next_norm.back() == expected_next);
        // Explicitly NOT this layer's own attn_norm, which is what an
        // off-by-one would produce on every layer but the first.
        TEST_REQUIRE(full.next_norm.front() !=
                     builder.SignatureOf(Blk(layer, ".attn_norm.weight")));
    }

    // Every blob handed to `create` is exactly what `pack` reported, which is
    // what corelib requires ("a truncated blob is still a plausible one").
    for (const auto& call : state.ple_create_calls) {
        TEST_REQUIRE(call.accepted);
        TEST_REQUIRE(call.packed_size == fake_corelib::PlePackedSize(call.desc));
    }
}

/// \brief ple is packed at `ple_group`, never at `group`
///
/// THE SAME PROBLEM AS head_group, WITH A DIFFERENT LEVER. All three group
/// fields are 32 on both shipped rows, so no descriptor assertion can tell
/// `ple_group` from `group`. But mladfple ships `_32` artifacts and NOTHING
/// ELSE, and corelib refuses any other value by name -- so the refusal
/// itself is the discriminator:
///
///   - with `ple_group = 64` and `group = 32`, an engine that passes
///     `ple_group` is REFUSED and one that passes `group` sails through;
///   - with `group = 64` and `ple_group = 32`, it is the other way round.
///
/// Both directions are asserted, so neither a confusion nor a swap passes.
void TestPleUsesPleGroupAndNotGroup() {
    const auto& fixture = E2bFixture();
    {
        const auto error = RequireThrows([&] {
            Harness harness(fixture,
                            [](Gemma4Config& shape) { shape.ple_group = 64; });
        });
        RequireContains(error, "ple_bf16_weights_create");
    }
    {
        // group 64 everywhere else; ple must still be packed at 32 and the
        // load must succeed.
        Harness harness(fixture, [](Gemma4Config& shape) { shape.group = 64; });
        const auto& calls = fake_corelib::GetState().ple_pack_calls;
        TEST_REQUIRE(!calls.empty());
        for (const auto& call : calls) TEST_REQUIRE(call.desc.group_size == 32);
    }
}

void TestE2bPacksOnePleBlockPerLayer() {
    RequirePle(E2bFixture());
}
void TestE4bPacksOnePleBlockPerLayer() {
    RequirePle(E4bFixture());
}

// ---------------------------------------------------------------------------
// The standalone norms.

void RequireRmsNorms(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    const auto& builder = fixture.builder;
    const auto& creates = fake_corelib::GetState().rmsnorm_weights_creates;

    const auto count_with = [&](std::int64_t k, std::uint16_t first) {
        return std::count_if(creates.begin(), creates.end(), [&](const auto& call) {
            return call.desc.k == k && !call.scale.empty() && call.scale.front() == first;
        });
    };

    for (const auto& call : creates) {
        TEST_REQUIRE(call.desc.epsilon == shape.eps);
        TEST_REQUIRE(!call.from_file);
        TEST_REQUIRE(call.scale.size() == static_cast<std::size_t>(call.desc.k));
    }

    // LAYER 0'S STANDALONE NORM. It has no producer: every other input norm
    // is the previous layer's ple second output, and layer 0 has no previous
    // layer. Exactly one create carries blk.0.attn_norm at the hidden width.
    TEST_REQUIRE(count_with(shape.hidden,
                            builder.SignatureOfBf16("blk.0.attn_norm.weight")) == 1);
    // And NO other layer's attn_norm is packed as a standalone norm -- they
    // reach the model through ple's `next_norm` instead.
    for (std::int64_t layer = 1; layer < shape.layers; ++layer) {
        TEST_REQUIRE(count_with(shape.hidden,
                                builder.SignatureOfBf16(Blk(layer, ".attn_norm.weight"))) == 0);
    }

    const auto ones_bf16 = static_cast<std::uint16_t>(0x3F80);   // 1.0f in BF16
    std::size_t ones_norms = 0;
    for (const auto& call : creates) {
        if (std::all_of(call.scale.begin(), call.scale.end(),
                        [&](std::uint16_t value) { return value == ones_bf16; }))
            ++ones_norms;
    }
    // V'S NORM, ONE PER OWNING LAYER. V has no learned gamma in the file and
    // rmsnorm_bf16 ships no weightless variant, so "normalize only" is a
    // vector of exactly 1.0. Skipping it entirely is the cheap wrong
    // implementation and still produces fluent text -- it just drops a whole
    // 1/sqrt(mean(v^2)) from every attended value.
    TEST_REQUIRE(ones_norms == static_cast<std::size_t>(shape.kv_layers));

    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const auto head_dim = shape.LayerHeadDim(layer);
        const bool owns = layer < shape.kv_layers;
        // THE QK NORMS ARE SIZED TO THE LAYER'S OWN HEAD, which is 256 on a
        // sliding layer and 512 on a full one -- `k` selects the kernel, so
        // one width for both would resolve the wrong artifact.
        TEST_REQUIRE(count_with(head_dim,
                                builder.SignatureOfBf16(Blk(layer, ".attn_q_norm.weight"))) == 1);
        TEST_REQUIRE(count_with(head_dim,
                                builder.SignatureOfBf16(Blk(layer, ".attn_k_norm.weight"))) ==
                     (owns ? 1 : 0));
        TEST_REQUIRE(count_with(shape.hidden,
                                builder.SignatureOfBf16(Blk(layer, ".post_attention_norm.weight"))) == 1);
    }
}

void TestE2bPacksTheStandaloneNormsIncludingVsVectorOfOnes() {
    RequireRmsNorms(E2bFixture());
}
void TestE4bPacksTheStandaloneNormsIncludingVsVectorOfOnes() {
    RequireRmsNorms(E4bFixture());
}

// ---------------------------------------------------------------------------
// The rotary tables.

/// \brief every FP32 host view this engine created, in creation order
std::vector<const fake_corelib::TensorCreateRecord*> Fp32Creates() {
    std::vector<const fake_corelib::TensorCreateRecord*> out;
    for (const auto& create : fake_corelib::GetState().host_view_creates) {
        if (create.data_type == ryzenai_corelib_data_type_fp32) out.push_back(&create);
    }
    return out;
}

/// \brief two rotary pairs, sliding then full, held for the engine's life
/// \note corelib memoizes the derived table on the view's identity, so one
///       pair shared by both regimes, or a sin aliased onto its cos, ropes
///       with the wrong table and still produces fluent text.
void RequireRotaryTables(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;

    const auto rotary = Fp32Creates();
    TEST_REQUIRE(rotary.size() == 4);
    TEST_REQUIRE(fake_corelib::GetState().host_view_creates.size() == 4);

    const auto expect_shape = [&](std::size_t index, std::int64_t head_dim) {
        const std::vector<std::int64_t> want{flm::gemma4::kMaxSequenceLength, head_dim / 2};
        TEST_REQUIRE(rotary[index]->shape == want);
    };
    // Sliding first, then full -- the order the engine builds them in. Both
    // pairs exist on both shipped rows; neither is conditional on the layer
    // stack, because both stacks contain both kinds.
    expect_shape(0, shape.head_dim);
    expect_shape(1, shape.head_dim);
    expect_shape(2, shape.global_head_dim);
    expect_shape(3, shape.global_head_dim);

    for (std::size_t i = 0; i < rotary.size(); ++i) {
        for (std::size_t j = i + 1; j < rotary.size(); ++j)
            TEST_REQUIRE(rotary[i]->object != rotary[j]->object);
    }
}

void TestE2bHoldsTwoDistinctRotaryPairsWrittenOnce() { RequireRotaryTables(E2bFixture()); }
void TestE4bHoldsTwoDistinctRotaryPairsWrittenOnce() { RequireRotaryTables(E4bFixture()); }

// ---------------------------------------------------------------------------
// The activation buffers and the KV caches.

/// \brief every BF16 device tensor this engine created, in creation order
std::vector<const fake_corelib::TensorCreateRecord*> Bf16Creates() {
    std::vector<const fake_corelib::TensorCreateRecord*> out;
    for (const auto& create : fake_corelib::GetState().tensor_creates) {
        if (create.data_type == ryzenai_corelib_data_type_bf16) out.push_back(&create);
    }
    return out;
}

/// \brief the buffers are at the WIDER geometry and the caches are NOT UNIFORM
///
/// TWO SEPARATE CLAIMS, AND THE SECOND IS THE ONE WITH A SILENT FAILURE MODE.
///
/// The activation buffers are allocated once at the widest head this model has
/// and windowed down per layer on the COLUMN axis, because each op reads its
/// row and column count off its input's shape. Sizing them at the sliding
/// geometry instead is caught by corelib (a window would not fit); sizing the
/// CACHES at one head width is not caught by anything, and the reference
/// driver names the consequence: it "hands the full-attention layers a cache
/// with the wrong row pitch -- which the kernel would address happily".
///
/// There are `kv_layers` caches, not one per layer, and cache `i` is as wide
/// as layer `i`'s own head. On E2B that is 512 at caches 4, 9 and 14 and 256
/// at the other twelve; on E4B 512 at 5, 11, 17 and 23. Neither pattern
/// follows from the other.
/// \param full_width_caches the cache indices that must be `global_head_dim`
///        wide, written out per row rather than re-derived -- see the bottom
///        of this function for why
void RequireBuffersAndCaches(const Fixture& fixture,
                             const std::vector<std::int64_t>& full_width_caches) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    const auto buffers = Bf16Creates();

    const auto widest = std::max(shape.head_dim, shape.global_head_dim);
    const auto q_dim = shape.q_heads * widest;
    const auto kv_dim = shape.kv_heads * widest;
    constexpr auto kRows = flm::gemma4::kMaxSequenceLength;

    // The ten activation buffers, the head's two, one per-layer-embedding
    // plane per layer, then K and V caches per OWNING layer.
    std::vector<std::vector<std::int64_t>> expected{
        {kRows, shape.hidden},   // hidden    -- normed input, o's output, ple's norm_out
        {kRows, shape.hidden},   // residual  -- the residual stream
        {kRows, shape.hidden},   // skip_sum  -- post_attention_norm's output
        {kRows, shape.hidden},   // ple_out   -- ssmlp's block output, ple's x
        {kRows, q_dim},          // q_buf
        {kRows, kv_dim},         // k_buf
        // THE V BUFFER PHI-4 DOES NOT HAVE. V is RMS-normalized between the
        // projection and the cache, and no norm can scatter, so the projection
        // lands here and the NORM writes the cache.
        {kRows, kv_dim},         // v_buf
        // QK-Norm's destinations. NOT an optimization to remove: corelib.h
        // measures in-place rmsnorm CORRUPTING at M = 1024 and 2048 on
        // exact-kernel dispatches, and a prefill Q-norm runs at
        // padded * q_heads rows.
        {kRows, q_dim},          // q_normed
        {kRows, kv_dim},         // k_normed
        {kRows, q_dim},          // attn_out
        {1, shape.hidden},       // last_hidden -- lm_head runs at M == 1 alone
        {1, shape.vocab},        // logits
    };
    for (std::int64_t layer = 0; layer < shape.layers; ++layer)
        expected.push_back({kRows, shape.ple_dim});
    for (std::int64_t cache = 0; cache < shape.kv_layers; ++cache) {
        const auto head = shape.LayerHeadDim(cache);
        expected.push_back({shape.kv_heads, kRows, head});   // K cache
        expected.push_back({shape.kv_heads, kRows, head});   // V cache
    }

    TEST_REQUIRE(buffers.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        TEST_REQUIRE(buffers[i]->shape == expected[i]);

    // EVERY ONE A SEPARATE ALLOCATION. ssmlp's four tensors and ple's three
    // both have distinctness rules corelib states outright, and two caches
    // aliased onto one would make a sharing layer read a cache nobody wrote.
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        for (std::size_t j = i + 1; j < buffers.size(); ++j)
            TEST_REQUIRE(buffers[i]->object != buffers[j]->object);
    }

    // AND THE CACHES REALLY ARE NON-UNIFORM, STATED INDEPENDENTLY.
    //
    // The `expected` table above reads `shape.LayerHeadDim`, which is the same
    // function the engine reads -- so on its own it would pass for an engine
    // that got the rule wrong in the same way the table does. What follows is
    // a SECOND statement of the answer, written out per row by the caller and
    // derived from nothing: the caches that must be `global_head_dim` wide,
    // measured off the real files in verified-gguf-facts.md.
    std::vector<std::int64_t> measured_full;
    for (std::int64_t cache = 0; cache < shape.kv_layers; ++cache) {
        const auto& created = buffers[12 + shape.layers + 2 * cache]->shape;
        TEST_REQUIRE(created.size() == 3);
        if (created[2] == shape.global_head_dim) measured_full.push_back(cache);
        else TEST_REQUIRE(created[2] == shape.head_dim);
    }
    TEST_REQUIRE(measured_full == full_width_caches);
    // Both widths really occur, so "non-uniform" is a property of this model
    // and not an empty claim about a stack that happens to be all one kind.
    TEST_REQUIRE(!measured_full.empty());
    TEST_REQUIRE(measured_full.size() < static_cast<std::size_t>(shape.kv_layers));
}

void TestE2bAllocatesTheWiderGeometryAndNonUniformCaches() {
    // E2B: 15 owning layers, full attention every 5th (4, 9, 14, ...), so
    // three of the fifteen caches are 512 wide and twelve are 256.
    RequireBuffersAndCaches(E2bFixture(), {4, 9, 14});
}
void TestE4bAllocatesTheWiderGeometryAndNonUniformCaches() {
    // E4B: 24 owning layers, full attention every 6th (5, 11, 17, 23, ...).
    // NEITHER SET FOLLOWS FROM THE OTHER -- different period, different owning
    // prefix -- which is why both are written out.
    RequireBuffersAndCaches(E4bFixture(), {5, 11, 17, 23});
}

// ---------------------------------------------------------------------------
// Which finished object reached which slot.

/// \brief every packed weight sits in the slot named for the tensor it was
///        built from
///
/// A THIRD AXIS, AND THE ONE NOTHING ABOVE CAN SEE. Every assertion before
/// this point asks either "was the right DESCRIPTOR built" or "did the right
/// BYTES reach a packer". Neither notices when the right object is built and
/// then stored in the wrong place: the destination is chosen at the call site
/// and never crosses the corelib ABI, so no record of any call can contain
/// it.
///
/// It was measured, not guessed. Exchanging the destination slots of one
/// layer's `q_norm` and `k_norm` -- identical descriptors (same `k`, same
/// epsilon), different gammas -- left the suite 16/16 green.
///
/// WHAT THAT PARTICULAR SWAP WOULD AND WOULD NOT DO ON THIS MODEL, because a
/// test whose stated reason is wrong gets deleted by the next person who
/// works the reasoning out. Gemma 4's Q and K gammas are NOT symmetric --
/// the K gamma carries the 1/sqrt(head_size) (0.12695 on blk.0) and the Q
/// gamma is ~1 -- but on this conversion both are CONSTANT-VALUED vectors,
/// and attention contracts Q against K: `Q.K = sum(nq*nk*gq*gk)`. Exchanging
/// two constants leaves that product unchanged. So the Q/K-norm swap is
/// mathematically NEUTRAL today, by a property of the weights, and this test
/// does not claim to have caught a bug. It claims that nothing anywhere
/// checks that property, that the neutrality is not by construction, and
/// that the SAME hole is not neutral elsewhere -- `ffn_gate` against
/// `ffn_up`, checked in RequireSsmlp, changes what the block computes.
///
/// Keyed on the fixture's per-tensor identity: a block tag for a quantized
/// tensor, a gamma signature for a norm, `layer_scale` for a ple block.
///
/// USED ON BOTH PATHS INTO A LOADED ENGINE, and written once so that it
/// cannot drift between them. The pack path creates every weight from GGUF
/// components; the CACHE path calls `..._weights_create_from_file` and
/// assigns the handles by walking the slot table IN INDEX ORDER, which is a
/// second and completely independent way to put the right weight in the wrong
/// member. R2's version of this function could not be pointed at the cache
/// path at all -- it joined on `blocks_tag` and `scale`, and both were empty
/// there by construction. They are not any more: see fake_corelib.cpp's
/// `identity_tag`, which puts what was packed into the packed bytes, so the
/// cache file carries the identity and the from_file leg reads it back.
///
/// WHAT IS STILL WEAKER ON THE CACHE PATH, stated rather than glossed:
///   - a norm is identified by ONE gamma element (`scale_tag`), not by the
///     whole vector, because corelib hands no gamma back through
///     `..._weights_create_from_file`. Two norms whose gammas agree in the
///     first element and differ later are indistinguishable there. They
///     cannot occur in this fixture (every F32 tensor is constant-filled) and
///     the full-vector check still runs wherever `scale` is populated.
///   - V's ones-norms are byte-identical between two owning layers of the
///     same head width on BOTH paths, so neither can tell one from the other.
///     Unchanged from R2; the confusion that matters is a ones-norm against a
///     learned gamma, and that is caught.
void RequireWeightPlacementOf(const gemma4_rai& engine, const Gemma4Config& shape,
                              const gemma4_fixture::Builder& builder) {
    const auto& state = fake_corelib::GetState();

    using Verifier = std::function<void(const void*)>;

    const auto matmul_from = [&](const std::string& tensor) -> Verifier {
        return [&, tensor](const void* handle) {
            const auto& creates = state.matmul_weights_creates;
            const auto found = std::find_if(
                creates.begin(), creates.end(),
                [&](const auto& call) { return call.object == handle; });
            TEST_REQUIRE(found != creates.end());
            TEST_REQUIRE(found->blocks_tag == builder.BlockSignatureOf(tensor));
        };
    };
    const auto norm_from = [&](const std::string& tensor) -> Verifier {
        return [&, tensor](const void* handle) {
            const auto& creates = state.rmsnorm_weights_creates;
            const auto found = std::find_if(
                creates.begin(), creates.end(),
                [&](const auto& call) { return call.object == handle; });
            TEST_REQUIRE(found != creates.end());
            // `scale_tag` is the ONE thing available on both legs. The whole
            // gamma is checked as well wherever it exists, which is the pack
            // leg -- the cache leg has none to check.
            TEST_REQUIRE(found->scale_tag == builder.SignatureOfBf16(tensor));
            if (!found->scale.empty())
                TEST_REQUIRE(found->scale.front() == builder.SignatureOfBf16(tensor));
        };
    };
    // V's norm has NO tensor in the file, so there is nothing to name it by
    // -- only its width and the fact that every element is exactly 1.0. That
    // is the honest limit here: two owning layers of the SAME head width hold
    // byte-identical ones-norms, so this cannot tell one from the other. It
    // can tell a ones-norm from a learned gamma, which is the confusion that
    // matters (skipping V's norm entirely still produces fluent text).
    const auto ones_norm = [&](std::int64_t k) -> Verifier {
        return [&, k](const void* handle) {
            const auto& creates = state.rmsnorm_weights_creates;
            const auto found = std::find_if(
                creates.begin(), creates.end(),
                [&](const auto& call) { return call.object == handle; });
            TEST_REQUIRE(found != creates.end());
            TEST_REQUIRE(found->desc.k == k);
            TEST_REQUIRE(found->scale_tag == 0x3F80);
            if (!found->scale.empty()) {
                TEST_REQUIRE(found->scale.size() == static_cast<std::size_t>(k));
                TEST_REQUIRE(std::all_of(found->scale.begin(), found->scale.end(),
                                         [](std::uint16_t value) { return value == 0x3F80; }));
            }
        };
    };
    const auto ssmlp_of = [&](std::int64_t layer) -> Verifier {
        return [&, layer](const void* handle) {
            const auto& creates = state.ssmlp_weights_creates;
            const auto found = std::find_if(
                creates.begin(), creates.end(),
                [&](const auto& call) { return call.object == handle; });
            TEST_REQUIRE(found != creates.end());
            TEST_REQUIRE(found->gate_tag ==
                         builder.BlockSignatureOf(Blk(layer, ".ffn_gate.weight")));
        };
    };
    const auto ple_of = [&](std::int64_t layer) -> Verifier {
        return [&, layer](const void* handle) {
            const auto& creates = state.ple_create_calls;
            const auto found = std::find_if(
                creates.begin(), creates.end(),
                [&](const auto& call) { return call.object == handle; });
            TEST_REQUIRE(found != creates.end());
            TEST_REQUIRE(found->desc.layer_scale ==
                         builder.SignatureOf(Blk(layer, ".layer_output_scale.weight")));
        };
    };

    // THE EXPECTED TABLE IS BUILT HERE, from the config, and not read off
    // anything the engine produced.
    std::map<std::string, Verifier> expected;
    expected["blk.0.attn_norm.weight"] = norm_from("blk.0.attn_norm.weight");
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const bool owns = layer < shape.kv_layers;
        expected[Blk(layer, ".attn_q.weight")] = matmul_from(Blk(layer, ".attn_q.weight"));
        expected[Blk(layer, ".attn_output.weight")] =
            matmul_from(Blk(layer, ".attn_output.weight"));
        expected[Blk(layer, ".attn_q_norm.weight")] =
            norm_from(Blk(layer, ".attn_q_norm.weight"));
        expected[Blk(layer, ".post_attention_norm.weight")] =
            norm_from(Blk(layer, ".post_attention_norm.weight"));
        expected[Blk(layer, ".ssmlp")] = ssmlp_of(layer);
        expected[Blk(layer, ".ple")] = ple_of(layer);
        if (owns) {
            expected[Blk(layer, ".attn_k.weight")] = matmul_from(Blk(layer, ".attn_k.weight"));
            expected[Blk(layer, ".attn_v.weight")] = matmul_from(Blk(layer, ".attn_v.weight"));
            expected[Blk(layer, ".attn_k_norm.weight")] =
                norm_from(Blk(layer, ".attn_k_norm.weight"));
            expected[Blk(layer, ".attn_v_norm(ones)")] = ones_norm(shape.LayerHeadDim(layer));
        }
    }
    expected["token_embd.weight"] = matmul_from("token_embd.weight");

    const auto placements = engine.WeightPlacementsForTest();
    TEST_REQUIRE(placements.size() == engine.weight_slot_count());
    TEST_REQUIRE(placements.size() == expected.size());
    for (const auto& placement : placements) {
        const auto found = expected.find(placement.slot);
        // An unexpected name, or a name appearing twice: both land here,
        // because a matched entry is erased.
        TEST_REQUIRE(found != expected.end());
        // A slot the engine left EMPTY is a missing weight, not a placement.
        TEST_REQUIRE(placement.handle != nullptr);
        found->second(placement.handle);
        expected.erase(found);
    }
    TEST_REQUIRE(expected.empty());
}

/// \brief RequireWeightPlacementOf against a freshly PACKED engine
void RequireWeightPlacement(const Fixture& fixture) {
    Harness harness(fixture);
    RequireWeightPlacementOf(*harness.engine, harness.shape, fixture.builder);
}

void TestE2bPutsEveryPackedWeightInTheSlotNamedForIt() {
    RequireWeightPlacement(E2bFixture());
}
void TestE4bPutsEveryPackedWeightInTheSlotNamedForIt() {
    RequireWeightPlacement(E4bFixture());
}

/// \brief a dispatch record carries the WEIGHTS OBJECT that crossed the ABI
///
/// FOR C8, AND IT HAS TO LAND BEFORE C8'S TESTS ARE WRITTEN. "The right
/// descriptor built and the wrong object handed over" becomes reachable the
/// moment there is a layer loop, and it then multiplies by 35 layers. Neither
/// this fake nor corelib can judge it from inside a dispatch -- both objects
/// are real and both were packed by the same entry point -- so the only
/// possible move is to RECORD the identity and let the caller assert on it.
/// The reference driver makes exactly this check host-side
/// (gemma4_driver.py:1670-1690).
///
/// EXERCISED DIRECTLY AGAINST THE FAKE, because the engine dispatches nothing
/// yet (`forward` and `prefill` refuse until C8). This is therefore a
/// fake-hardening test: it pins the fake's own contract so that C8's
/// assertions have something true to stand on. `flat_mha` is not covered
/// because it takes no weights object at all.
void TestADispatchRecordsTheWeightsObjectItRanWith() {
    RuntimeScope teardown;
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(
        CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    const auto& functions = runtime->api()->functions();
    auto* stream = fake_corelib::MakeStreamForTest();

    constexpr std::int64_t kK = 64;
    constexpr std::int64_t kN = 128;
    // Register a pad answer for this width first: the fake REFUSES a dispatch
    // whose operand rows no pad helper is known to have produced, and a
    // helper is keyed per output width. M == 1 is never rounded.
    std::int64_t rows = 1, padded_k = kK, padded_n = kN;
    ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp_desc{kK, kN, 32, 1, 1};
    std::int64_t ssmlp_rows = 1;
    TEST_REQUIRE(fake_corelib::NoteMatmulPad(stream, &rows, &padded_k, &padded_n, 32) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(fake_corelib::NoteSsmlpRows(stream, &ssmlp_rows, &ssmlp_desc) ==
                 ryzenai_corelib_status_success);

    std::vector<std::byte> blocks(64, std::byte{0});
    const auto tensor = [&](std::int64_t columns) {
        const std::int64_t shape[2] = {1, columns};
        void* object = nullptr;
        TEST_REQUIRE(functions.create_device_tensor(ryzenai_corelib_data_type_bf16,
                                                    shape, 2, &object) ==
                     ryzenai_corelib_status_success);
        return object;
    };

    ryzenai_corelib_matmul_bf16_weights_desc matmul_desc{kK, kN, 32, false};
    const auto matmul_sources = MatmulSources(blocks);
    void* matmul_weights = nullptr;
    TEST_REQUIRE(functions.matmul_weights_create(
                     &matmul_desc, matmul_sources.data(), matmul_sources.size(), nullptr,
                     &matmul_weights) == ryzenai_corelib_status_success);
    TEST_REQUIRE(functions.matmul(stream, tensor(kK), matmul_weights, tensor(kN)) ==
                 ryzenai_corelib_status_success);

    const auto ssmlp_sources = SsMlpSources(blocks);
    void* ssmlp_weights = nullptr;
    TEST_REQUIRE(functions.ssmlp_weights_create(
                     &ssmlp_desc, ssmlp_sources.data(), ssmlp_sources.size(), nullptr,
                     &ssmlp_weights) == ryzenai_corelib_status_success);
    TEST_REQUIRE(functions.ssmlp(stream, tensor(kK), tensor(kK), ssmlp_weights,
                                 tensor(kK), tensor(kK)) ==
                 ryzenai_corelib_status_success);

    const auto& dispatches = fake_corelib::GetState().dispatches;
    TEST_REQUIRE(dispatches.size() == 2);
    TEST_REQUIRE(dispatches[0].kind == "matmul");
    TEST_REQUIRE(dispatches[0].weights == matmul_weights);
    TEST_REQUIRE(dispatches[1].kind == "ssmlp");
    TEST_REQUIRE(dispatches[1].weights == ssmlp_weights);
    // AND THE JOIN C8 ACTUALLY NEEDS: the handle a dispatch ran with is the
    // handle some recorded create minted, so "this dispatch used the object
    // packed from THAT tensor" is one lookup away.
    const auto& creates = fake_corelib::GetState().matmul_weights_creates;
    TEST_REQUIRE(creates.size() == 1);
    TEST_REQUIRE(creates.front().object == dispatches[0].weights);
    TEST_REQUIRE(fake_corelib::GetState().ssmlp_weights_creates.front().object ==
                 dispatches[1].weights);
}

/// \brief `ple_bf16` and `rmsnorm_bf16` are recorded as dispatches, with
///        storage identity, their weights object, and IN ORDER
///
/// ABOUT HALF OF A GEMMA 4 LAYER WAS INVISIBLE. A layer is SEVEN dispatches
/// when it shares a cache and `10 + kv_heads` when it owns one -- 11 on E2B,
/// 12 on E4B, because V's norm runs once per KV head -- and these two
/// operators are 3 of 7 and 5 of 11 (6 of 12) of them. (This comment first
/// said "ten and six" and "two thirds": those are the WEIGHT-OBJECT counts
/// from gemma4_rai.hpp:40-47, correct for weights and wrong for dispatches.
/// Counted from gemma4_driver.py:2254-2336.)
/// Yet the fake minted records only for matmul, ssmlp and mha
/// and let everything else fall through its generic success branch. C8 cannot
/// assert anything about its own layer loop against an instrument that does
/// not see most of it.
///
/// LANDED BEFORE C8 ON PURPOSE. Four defects in this project got through
/// because the thing under test and the thing measuring it were written
/// together, by the same author, to the same assumption. The layer loop must
/// not build the instrument it is measured by.
///
/// THE SEQUENCE, NOT THE MULTISET. `State::dispatches` is in ABI call order
/// (push order under one lock, and unlike the weight CREATES nothing
/// dispatches across threads), so the assertion below is on the ordered list
/// of kinds. A permutation of a layer's dispatches is a different model;
/// a count of them is not.
///
/// EXERCISED DIRECTLY AGAINST THE FAKE, because the engine dispatches nothing
/// until C8 -- the same fake-hardening shape as
/// TestADispatchRecordsTheWeightsObjectItRanWith, and for the same reason.
void TestPleAndRmsNormAreRecordedAsDispatchesInSequence() {
    RuntimeScope teardown;
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(
        CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    const auto& functions = runtime->api()->functions();
    const auto& state = fake_corelib::GetState();
    auto* stream = fake_corelib::MakeStreamForTest();

    constexpr std::int64_t kHidden = 128;
    constexpr std::int64_t kPleDim = 64;

    const auto tensor = [&](std::int64_t rows, std::int64_t columns) {
        const std::int64_t shape[2] = {rows, columns};
        void* object = nullptr;
        TEST_REQUIRE(functions.create_device_tensor(ryzenai_corelib_data_type_bf16,
                                                    shape, 2, &object) ==
                     ryzenai_corelib_status_success);
        return object;
    };
    const auto window = [&](void* parent, std::int64_t rows, std::int64_t columns) {
        const std::int64_t shape[2] = {rows, columns};
        void* object = nullptr;
        TEST_REQUIRE(functions.create_tensor_window(parent, shape, 2, 0, nullptr, 0, &object) ==
                     ryzenai_corelib_status_success);
        return object;
    };

    // ---- the two weights objects, through the entry points the engine uses
    ryzenai_corelib_rmsnorm_bf16_weights_desc norm_desc{kHidden, 1.0e-6f};
    const std::vector<std::uint16_t> gamma(static_cast<std::size_t>(kHidden), 0x3F80);
    const ryzenai_corelib_weights_source norm_sources[]{flm::corelib::Bf16(
        ryzenai_corelib_weights_role_scale, gamma.data(), gamma.size())};
    void* norm_weights = nullptr;
    TEST_REQUIRE(functions.rmsnorm_weights_create(&norm_desc, norm_sources, 1, nullptr,
                                                  &norm_weights) ==
                 ryzenai_corelib_status_success);

    ryzenai_corelib_ple_bf16_weights_desc ple_desc{kHidden, kPleDim, 32, 1.0e-6f, 1.5f};
    const std::size_t packed = fake_corelib::PlePackedSize(ple_desc);
    std::vector<std::byte> blob(packed, std::byte{0});
    ryzenai_corelib_weights_source ple_source{};
    ple_source.data = blob.data();
    ple_source.role = ryzenai_corelib_weights_role_packed;
    ple_source.size = blob.size();
    ple_source.data_type = ryzenai_corelib_weights_data_type_native_packed;
    void* ple_weights = nullptr;
    TEST_REQUIRE(functions.ple_weights_create(&ple_desc, &ple_source, 1, nullptr,
                                              &ple_weights) ==
                 ryzenai_corelib_status_success);

    // ---- rmsnorm, IN PLACE, through two windows onto ONE allocation
    //
    // Accepted, and it has to be: corelib.h says input and output "may be the
    // SAME tensor FOR SMALL M -- the kernel reads a row before it writes it".
    // A distinctness rule here would be a false rejection of a legal decode
    // dispatch, so there is none. (The same paragraph says it CORRUPTS at
    // M = 1024/2048 at k=128 even where an exact kernel ships; that is a
    // property of the width and the row count, and nothing host-side can tell
    // the two cases apart, so it is not checkable here at all. C8 must use a
    // separate output unless the (M, k) it dispatches is one in-place has
    // been shown safe at.)
    void* residual = tensor(1, kHidden);
    void* norm_in = window(residual, 1, kHidden);
    void* norm_out = window(residual, 1, kHidden);
    TEST_REQUIRE(norm_in != norm_out);   // two OBJECTS...
    TEST_REQUIRE(functions.rmsnorm(stream, norm_in, norm_weights, norm_out) ==
                 ryzenai_corelib_status_success);
    // RECORDED AT ALL. Before this task both of these operators fell through
    // the fake's generic success branch and minted nothing.
    TEST_REQUIRE(state.dispatches.size() == 1);
    TEST_REQUIRE(state.dispatches[0].kind == "rmsnorm");
    TEST_REQUIRE(state.dispatches[0].weights == norm_weights);
    TEST_REQUIRE(state.dispatches[0].operands ==
                 std::vector<void*>({norm_in, norm_out}));
    // ...OVER ONE ALLOCATION. This is why an aliasing question must be asked
    // of `operand_storage` and never of `operands`: the handles differ above
    // and the storage does not.
    TEST_REQUIRE(state.dispatches[0].operand_storage[0] ==
                 state.dispatches[0].operand_storage[1]);
    TEST_REQUIRE(state.dispatches[0].operand_storage[0] != nullptr);

    // ---- ple, with its four row-bearing operands in the header's order
    void* x = tensor(1, kHidden);
    void* slice = tensor(1, kPleDim);
    void* out = tensor(1, kHidden);
    void* normed = tensor(1, kHidden);
    TEST_REQUIRE(functions.ple(stream, x, slice, ple_weights, out, normed) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(state.dispatches.size() == 2);
    TEST_REQUIRE(state.dispatches[1].kind == "ple");
    TEST_REQUIRE(state.dispatches[1].weights == ple_weights);
    TEST_REQUIRE(state.dispatches[1].operands ==
                 std::vector<void*>({x, slice, out, normed}));
    // The join C8 needs: the object a ple dispatch ran with is the object a
    // recorded create minted, and that create's descriptor carries
    // `layer_scale`, which names the layer.
    TEST_REQUIRE(state.ple_create_calls.size() == 1);
    TEST_REQUIRE(state.ple_create_calls.front().object == ple_weights);
    TEST_REQUIRE(state.ple_create_calls.front().desc.layer_scale == 1.5f);

    // corelib.h: "`out` and `norm_out` must be distinct buffers, from each
    // other and from `x`." Enforced, on STORAGE. `ple` -- the per-layer input
    // slice -- is deliberately not in that set; the header does not put it
    // there.
    TEST_REQUIRE(functions.ple(stream, x, slice, ple_weights, x, normed) ==
                 ryzenai_corelib_status_bad_argument);
    TEST_REQUIRE(functions.ple(stream, x, slice, ple_weights, out, out) ==
                 ryzenai_corelib_status_bad_argument);
    // And a row count no ple kernel ships for. corelib would ROUND UP and
    // read and write the whole padded tile off the end of a buffer sized to
    // the live rows -- "comes back with plausible numbers". The fake refuses
    // instead, so the overrun is a test failure rather than a plausible one.
    TEST_REQUIRE(functions.ple(stream, tensor(3, kHidden), tensor(3, kPleDim),
                               ple_weights, tensor(3, kHidden),
                               tensor(3, kHidden)) ==
                 ryzenai_corelib_status_bad_argument);
    // A refused dispatch records NOTHING -- still two.
    TEST_REQUIRE(state.dispatches.size() == 2);

    // ---- a matmul and an ssmlp between them, so the SEQUENCE is observable
    std::int64_t rows = 1, padded_k = kHidden, padded_n = kHidden;
    ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp_desc{kHidden, kPleDim, 32, 1, 1};
    std::int64_t ssmlp_rows = 1;
    TEST_REQUIRE(fake_corelib::NoteMatmulPad(stream, &rows, &padded_k, &padded_n, 32) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(fake_corelib::NoteSsmlpRows(stream, &ssmlp_rows, &ssmlp_desc) ==
                 ryzenai_corelib_status_success);
    std::vector<std::byte> blocks(64, std::byte{0});
    ryzenai_corelib_matmul_bf16_weights_desc matmul_desc{kHidden, kHidden, 32, false};
    const auto matmul_sources = MatmulSources(blocks);
    void* matmul_weights = nullptr;
    TEST_REQUIRE(functions.matmul_weights_create(
                     &matmul_desc, matmul_sources.data(), matmul_sources.size(), nullptr,
                     &matmul_weights) == ryzenai_corelib_status_success);
    TEST_REQUIRE(functions.matmul(stream, tensor(1, kHidden), matmul_weights,
                                  tensor(1, kHidden)) ==
                 ryzenai_corelib_status_success);
    const auto ssmlp_sources = SsMlpSources(blocks);
    void* ssmlp_weights = nullptr;
    TEST_REQUIRE(functions.ssmlp_weights_create(
                     &ssmlp_desc, ssmlp_sources.data(), ssmlp_sources.size(), nullptr,
                     &ssmlp_weights) == ryzenai_corelib_status_success);
    TEST_REQUIRE(functions.ssmlp(stream, tensor(1, kHidden), tensor(1, kHidden),
                                 ssmlp_weights, tensor(1, kHidden),
                                 tensor(1, kHidden)) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(functions.rmsnorm(stream, tensor(1, kHidden), norm_weights,
                                   tensor(1, kHidden)) ==
                 ryzenai_corelib_status_success);

    // THE ORDERED LIST, WHICH IS THE POINT. A multiset assertion would pass
    // under any permutation of a layer's dispatches, and a permuted layer is
    // a different model.
    const std::vector<std::string> expected{"rmsnorm", "ple", "matmul", "ssmlp",
                                            "rmsnorm"};
    TEST_REQUIRE(state.dispatches.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        TEST_REQUIRE(state.dispatches[i].kind == expected[i]);
    // Every dispatch names the stream it ran on.
    for (const auto& dispatch : state.dispatches)
        TEST_REQUIRE(dispatch.stream == stream);
}

/// \brief a `flat_mha` dispatch records the ROTARY TABLES and the CACHES it
///        was handed, not only q/k/out
///
/// THE INSTRUMENT DID NOT SEE EITHER, AND BOTH ARE SILENT WHEN WRONG.
///
/// - A layer handed the WRONG ROTARY PAIR ropes at the other regime's base.
///   "One pair used everywhere still converges to fluent text"
///   (`gemma4_driver.py` `_rope_tables`). Nothing about the output says so.
/// - A layer handed the WRONG CACHE attends over another layer's keys and
///   values. On E2B twenty layers read a cache they did not write, and which
///   one they read is `layer_cache_owner` -- "the last own-cache layer of its
///   own kind", which is 13/14 on E2B and 22/23 on E4B and follows from
///   neither the other pair nor from a layer count.
///
/// Both operands cross the ABI on every attention dispatch and were simply not
/// retained. R3's rule applies to them exactly as it does to `weights`: "the
/// right descriptor built and the wrong object handed over" is not detectable
/// from inside a dispatch, by this fake or by corelib, so the only possible
/// move is to record the identity and let the caller assert on it.
///
/// EXERCISED DIRECTLY AGAINST THE FAKE, the same fake-hardening shape as
/// TestPleAndRmsNormAreRecordedAsDispatchesInSequence.
///
/// IT NOW ALSO PINS `mha_position` AND `operand_shapes` HERE. Review C8
/// showed that `mha_position` was reachable only through the engine --
/// deleting it from the fake failed three tests and all three ran a full
/// forward pass -- which is exactly the arrangement this file's instrument
/// tests exist to prevent: a field that only the engine can exercise is a
/// field whose failure and the engine's failure are one observation.
void TestAnMhaDispatchRecordsItsRotaryTablesAndCaches() {
    RuntimeScope teardown;
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(
        CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    const auto& functions = runtime->api()->functions();
    const auto& state = fake_corelib::GetState();
    auto* stream = fake_corelib::MakeStreamForTest();

    ryzenai_corelib_flat_mha_bf16_desc desc{};
    desc.num_heads = 8;
    desc.kv_num_heads = 1;
    desc.head_size = 256;
    desc.max_seq = 4096;
    desc.rope_dim = 256;
    desc.window = 512;
    desc.kv_shared = 0;
    desc.scale = 1.0f;
    std::int64_t rows = 1;
    TEST_REQUIRE(fake_corelib::NoteMhaRows(stream, &rows, &desc) ==
                 ryzenai_corelib_status_success);

    const auto tensor = [&](std::vector<std::int64_t> shape,
                            ryzenai_corelib_data_type type) {
        void* object = nullptr;
        TEST_REQUIRE(functions.create_device_tensor(type, shape.data(), shape.size(),
                                                    &object) ==
                     ryzenai_corelib_status_success);
        return object;
    };
    void* q = tensor({1, 2048}, ryzenai_corelib_data_type_bf16);
    void* k = tensor({1, 256}, ryzenai_corelib_data_type_bf16);
    void* out = tensor({1, 2048}, ryzenai_corelib_data_type_bf16);
    const std::vector<float> rotary(4096 * 128, 0.0f);
    const auto view = [&]() {
        const std::int64_t shape[] = {4096, 128};
        void* object = nullptr;
        TEST_REQUIRE(functions.create_host_view(ryzenai_corelib_data_type_fp32, shape, 2,
                                                rotary.data(), &object) ==
                     ryzenai_corelib_status_success);
        return object;
    };
    void* cosine = view();
    void* sine = view();
    void* k_cache = tensor({1, 4096, 256}, ryzenai_corelib_data_type_bf16);
    void* v_cache = tensor({1, 4096, 256}, ryzenai_corelib_data_type_bf16);

    TEST_REQUIRE(functions.flat_mha(stream, &desc, q, k, 37, cosine, sine, k_cache,
                                    v_cache, out) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(state.dispatches.size() == 1);
    const auto& dispatch = state.dispatches.front();
    TEST_REQUIRE(dispatch.kind == "mha");
    TEST_REQUIRE(dispatch.operands == std::vector<void*>({q, k, out}));
    // THE WIDTHS, IN THE SAME ORDER. Retained by
    // `DispatchRecord::operand_shapes`, which exists because a whole geometry
    // can be wrong while every other field of this record is right. The three
    // tensors above were deliberately given DIFFERENT widths (2048, 256,
    // 2048), so a record that retained one shape for all three, or the rows
    // alone, fails here.
    TEST_REQUIRE(dispatch.operand_shapes ==
                 std::vector<std::vector<std::int64_t>>(
                     {{1, 2048}, {1, 256}, {1, 2048}}));
    // THE POSITION, ASSERTED DIRECTLY AGAINST THE FAKE. Review C8 found this
    // field reachable only through the engine: deleting it from the fake
    // failed three tests, all engine-level, so the claim that every
    // instrument extension is pinned by a direct test did not hold for it.
    // The position passed above is 37 rather than 0 ON PURPOSE -- the field's
    // default is -1, so 0 would discriminate against "the fake dropped it"
    // but NOT against "the fake, or the engine, hardcoded a zero", which is
    // the mutation this field exists to catch.
    TEST_REQUIRE(dispatch.mha_position == 37);
    // THE FOUR THAT WERE INVISIBLE. Each is the handle that crossed the ABI,
    // so a test can join it to the tensor the engine created for that role.
    TEST_REQUIRE(dispatch.mha_cos == cosine);
    TEST_REQUIRE(dispatch.mha_sin == sine);
    TEST_REQUIRE(dispatch.mha_k_cache == k_cache);
    TEST_REQUIRE(dispatch.mha_v_cache == v_cache);
    // And they are told apart from each other: a record that stored one handle
    // in two fields would satisfy every equality above if the caller happened
    // to pass the same tensor twice, so pass four distinct ones and say so.
    TEST_REQUIRE(cosine != sine && k_cache != v_cache);
}

/// \brief a `tensor_read` records WHICH ROW it read
///
/// READING THE WRONG ROW IS THE QUIETEST FAILURE IN THE WHOLE PASS. A forward
/// pass ends by reading the LAST live row of `hidden` -- element offset
/// `(rows - 1) * hidden` -- because that is the position whose logits the
/// caller wants. Reading row 0 instead hands back the FIRST prompt token's
/// hidden state: finite, correctly shaped, correctly bounded, and the logits
/// of the wrong position. Every later token then conditions on it, so the text
/// stays fluent and is answering a different prompt.
///
/// `tensor_write` has been recorded since C6 and `tensor_read` was not, so
/// nothing could ask the question. The (count, offset) pair is exactly what
/// the bug is made of.
///
/// AND WHAT THE READ PUTS IN THE BUFFER, which is asserted here for the first
/// time. The fake does not move device bytes -- it has none -- but it no
/// longer zeroes the destination either: it fills it with
/// `kSyntheticReadValue`, because zero is a FIXED POINT of the host-side
/// logit softcap and a read that returned zeros made "the engine applied the
/// cap" and "the engine dropped it" the same observation. Until now nothing
/// pinned that value directly against the fake: the only test that depended
/// on it ran through the engine, so "the fake stopped filling the buffer"
/// and "the engine stopped capping" were one failure.
void TestTensorReadsAreRecordedWithTheirOffset() {
    RuntimeScope teardown;
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(
        CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    const auto& functions = runtime->api()->functions();
    const auto& state = fake_corelib::GetState();

    const std::int64_t shape[2] = {4, 8};
    void* tensor = nullptr;
    TEST_REQUIRE(functions.create_device_tensor(ryzenai_corelib_data_type_bf16, shape,
                                                2, &tensor) ==
                 ryzenai_corelib_status_success);
    std::vector<float> destination(8, 1.0f);
    TEST_REQUIRE(functions.tensor_read(tensor, ryzenai_corelib_data_type_fp32,
                                       destination.data(), destination.size(), 24) ==
                 ryzenai_corelib_status_success);
    TEST_REQUIRE(state.tensor_reads.size() == 1);
    TEST_REQUIRE(state.tensor_reads[0].tensor == tensor);
    TEST_REQUIRE(state.tensor_reads[0].count == 8);
    TEST_REQUIRE(state.tensor_reads[0].offset == 24);
    TEST_REQUIRE(state.tensor_reads[0].destination_type ==
                 ryzenai_corelib_data_type_fp32);
    TEST_REQUIRE(state.tensor_reads[0].accepted);
    // AN ACCEPTED READ FILLS THE WHOLE BUFFER with the synthetic constant.
    // The buffer was pre-filled with 1.0f above, so this discriminates
    // against "nothing was written" as well as against "a different value
    // was".
    for (const float value : destination)
        TEST_REQUIRE(value == fake_corelib::kSyntheticReadValue);
    // AND THE CONSTANT IS OUTSIDE THE SOFTCAP'S NEAR-LINEAR REGION, which is
    // the whole reason it is not zero: 45 caps to about 27.15 at cap 30, so
    // an engine that dropped the cap is distinguishable from one that applied
    // it. A constant small enough to make the cap invisible would satisfy
    // TestTheReturnedLogitsAreSoftcapped's equality and defeat its purpose,
    // so state the premise where the constant is read rather than leave it to
    // the definition's comment.
    TEST_REQUIRE(fake_corelib::kSyntheticReadValue > 30.0f);

    // A REFUSED READ IS RECORDED AS REFUSED, not dropped: "read past the end"
    // and "did not read at all" are different bugs and a record that kept only
    // the accepted ones could not tell them apart.
    std::fill(destination.begin(), destination.end(), 1.0f);
    TEST_REQUIRE(functions.tensor_read(tensor, ryzenai_corelib_data_type_fp32,
                                       destination.data(), destination.size(), 28) ==
                 ryzenai_corelib_status_bad_argument);
    TEST_REQUIRE(state.tensor_reads.size() == 2);
    TEST_REQUIRE(!state.tensor_reads[1].accepted);
    // AND IT MOVED NOTHING. A refused read that filled the buffer anyway
    // would hand a caller that ignored the status a plausible answer.
    for (const float value : destination) TEST_REQUIRE(value == 1.0f);
}

// ---------------------------------------------------------------------------
// The layer loop.
//
// Everything below runs a real pass through the engine. The suite computes NO
// ATTENTION -- the fake moves no device bytes, and a read returns a synthetic
// constant that means nothing except "not zero" (fake_corelib.hpp's
// kSyntheticReadValue) -- so none of this is evidence about the numbers. The
// ONE thing a test may conclude from a read's contents is "this constant,
// transformed by the host-side arithmetic the engine claims to apply", which
// is what TestTheReturnedLogitsAreSoftcapped does and the only such test
// here. What all of this IS evidence about is the READ SEQUENCE:
// which operator ran, in what order, against which packed weight, over which
// buffer, with which descriptor, rotary pair and cache. Every characteristic
// failure of this task is a wrong entry in that list, and every one of them
// produces fluent, plausible, wrong text rather than an error.

/// \brief the packed-weight handle -> slot-name map, from the engine itself
/// \note `WeightPlacementsForTest` is an INDEPENDENTLY WRITTEN statement of
///       which tensor belongs in which member (see gemma4_rai.hpp), so joining
///       a dispatch's `weights` handle through it names the weight a dispatch
///       actually ran with rather than the one the loop meant to use.
std::map<const void*, std::string> SlotsByHandle(const gemma4_rai& engine) {
    std::map<const void*, std::string> slots;
    for (const auto& placement : engine.WeightPlacementsForTest())
        slots.emplace(placement.handle, placement.slot);
    return slots;
}

/// \brief the ordered (operator, weight slot) pairs one pass dispatched
/// \note `"-"` for `flat_mha`, which takes no weights object at all.
std::vector<std::pair<std::string, std::string>> DispatchedSequence(
    const gemma4_rai& engine) {
    const auto slots = SlotsByHandle(engine);
    std::vector<std::pair<std::string, std::string>> sequence;
    for (const auto& dispatch : fake_corelib::GetState().dispatches) {
        std::string slot = "-";
        if (dispatch.weights) {
            const auto found = slots.find(dispatch.weights);
            slot = found == slots.end() ? std::string("UNKNOWN") : found->second;
        }
        sequence.emplace_back(dispatch.kind, slot);
    }
    return sequence;
}

/// \brief the sequence the reference driver's `forward()` dispatches
///
/// TRANSCRIBED FROM `gemma4_driver.py:2254-2336`, NOT DERIVED. A layer is
/// SEVEN dispatches when it shares a cache and `10 + kv_heads` when it owns
/// one -- 11 on E2B, 12 on E4B -- and the difference is K, V, K's norm and ONE
/// V-norm PER KV HEAD.
///
/// A PERMUTATION OF THIS LIST IS A DIFFERENT MODEL AND A COUNT OF IT IS NOT,
/// which is why the assertion below is ordered. R3's M7 mutation proved the
/// instrument can tell the two apart: reversing dispatch order leaves the
/// multiset unchanged and fails only the ordered assertion.
std::vector<std::pair<std::string, std::string>> ExpectedSequence(
    const Gemma4Config& shape) {
    std::vector<std::pair<std::string, std::string>> expected;
    // LAYER 0'S STANDALONE NORM, the ONE in the model. Every other input norm
    // is the previous layer's `ple` second output; this one reads the
    // embedding with no block before it.
    expected.emplace_back("rmsnorm", "blk.0.attn_norm.weight");
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const bool owns = layer < shape.kv_layers;
        expected.emplace_back("matmul", Blk(layer, ".attn_q.weight"));
        // A SHARING LAYER PROJECTS NEITHER K NOR V AND NORMALIZES NEITHER. Its
        // descriptor is a kvshare artifact that takes no K operand at all.
        if (owns) {
            expected.emplace_back("matmul", Blk(layer, ".attn_k.weight"));
            expected.emplace_back("matmul", Blk(layer, ".attn_v.weight"));
            expected.emplace_back("rmsnorm", Blk(layer, ".attn_k_norm.weight"));
            // ONE V-NORM PER KV HEAD, because each head's plane is a separate
            // contiguous run in both the staging buffer and the cache.
            for (std::int64_t head = 0; head < shape.kv_heads; ++head)
                expected.emplace_back("rmsnorm", Blk(layer, ".attn_v_norm(ones)"));
        }
        // QK-NORM BEFORE ATTENTION: flat_mha applies the rotary itself and
        // Gemma normalizes ahead of it.
        expected.emplace_back("rmsnorm", Blk(layer, ".attn_q_norm.weight"));
        expected.emplace_back("mha", "-");
        expected.emplace_back("matmul", Blk(layer, ".attn_output.weight"));
        expected.emplace_back("rmsnorm", Blk(layer, ".post_attention_norm.weight"));
        expected.emplace_back("ssmlp", Blk(layer, ".ssmlp"));
        expected.emplace_back("ple", Blk(layer, ".ple"));
    }
    // lm_head, and NO FINAL NORM BEFORE IT. The last layer's `ple` is packed
    // with `output_norm.weight` as its `next_norm`, so `hidden` already
    // carries the model's final norm when the loop ends.
    expected.emplace_back("matmul", "token_embd.weight");
    return expected;
}

/// \brief the `create_tensor_window` record a handle came from, or null
///
/// SEARCHED IN REVERSE, AND THAT IS LOAD-BEARING ACROSS MORE THAN ONE PASS.
/// Every window `Run` builds is a local that is released when the pass
/// returns, so the allocator hands the SAME ADDRESS back to the next pass's
/// windows -- and `State::tensor_windows` keeps every record forever. A
/// forward search therefore answers a decode's question with the PREFILL's
/// window: same handle value, different offset. That is not hypothetical
/// bookkeeping, it is the difference between "V wrote cache row 3" and "V
/// wrote cache row 0" as seen by a test, and searching forwards would report
/// the prefill's offset for a correct decode -- a false failure -- while
/// still reporting it for an incorrect one.
///
/// Within ONE pass every window is alive at once, so no address is reused
/// there and the most recent record for a handle is that pass's own.
const fake_corelib::TensorWindowRecord* WindowFor(void* object) {
    const auto& windows = fake_corelib::GetState().tensor_windows;
    for (auto record = windows.rbegin(); record != windows.rend(); ++record) {
        if (record->object == object) return &*record;
    }
    return nullptr;
}

/// \brief the tensor a window was carved from, or null if `object` is not one
/// \note One level, which is all this engine uses: every window it builds is
///       taken directly off a device tensor.
void* ParentOf(void* object) {
    const auto* window = WindowFor(object);
    return window ? window->parent : nullptr;
}

/// \brief the offset a window was carved at, or `size_t(-1)`
std::size_t OffsetOf(void* object) {
    const auto* window = WindowFor(object);
    return window ? window->offset : static_cast<std::size_t>(-1);
}

/// \brief indices into `State::dispatches` of every dispatch of one kind
std::vector<std::size_t> IndicesOfKind(std::string_view kind) {
    std::vector<std::size_t> indices;
    const auto& dispatches = fake_corelib::GetState().dispatches;
    for (std::size_t i = 0; i < dispatches.size(); ++i) {
        if (dispatches[i].kind == kind) indices.push_back(i);
    }
    return indices;
}

/// \param expected_dispatches written out per row, not computed, so the two
///        models state their own answers
void RequireLayerLoop(const Fixture& fixture, std::size_t expected_dispatches) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    std::vector<int> ids{2, 5, 9};
    const auto logits = harness.engine->prefill(ids);
    TEST_REQUIRE(logits.size() == static_cast<std::size_t>(shape.vocab));

    const auto expected = ExpectedSequence(shape);
    const auto owning = static_cast<std::size_t>(shape.kv_layers);
    const auto sharing = static_cast<std::size_t>(shape.layers) - owning;
    // THE ARITHMETIC, SPELLED OUT. 10 + kv_heads on an owning layer and 7 on a
    // sharing one, plus layer 0's standalone norm and lm_head. An earlier
    // draft of this port's notes said "ten and six" -- those are the
    // WEIGHT-OBJECT counts, and trusting them here yields a MISSING DISPATCH
    // rather than a red test.
    TEST_REQUIRE(expected.size() ==
                 1 + owning * (10 + static_cast<std::size_t>(shape.kv_heads)) +
                     sharing * 7 + 1);
    TEST_REQUIRE(expected.size() == expected_dispatches);

    const auto actual = DispatchedSequence(*harness.engine);
    TEST_REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) TEST_REQUIRE(actual[i] == expected[i]);

    // THE PREFILL GRID'S FLOOR IS 128, NOT 64. Three live rows run a 128-row
    // kernel: 128 is the floor hybrid-llm's full-fusion path supports at all,
    // and below it upstream's own comment says a native buffer wedges the NPU
    // for LLM matmul shapes. The shape plan's own bucket table starts at 64 --
    // it is quoted verbatim from `ple_bf16`'s doc comment, which is about
    // which kernels EXIST -- so nothing else in this suite would notice a pass
    // that ran at 64, and the fake would accept it.
    for (const auto index : IndicesOfKind("ple"))
        TEST_REQUIRE(fake_corelib::GetState().dispatches[index].rows == 128);

    // ONE WAIT IN THE PASS, and it is the one before the read-back. Checked
    // here rather than assumed: the loop barriers NOWHERE, because corelib
    // holds one derived rotary buffer per key per interval and "no barrier is
    // needed at a change of base". A synchronize per geometry change cost the
    // reference driver 13 extra round trips on an E2B. (The head's own wait,
    // after lm_head, is the second -- `forward` returns logits, not the hidden
    // row the driver's `forward` returns.)
    TEST_REQUIRE(fake_corelib::GetState().tensor_reads.size() == 2);
}

void TestE2bPrefillDispatchesTheLayerSequence() {
    // 1 + 15*11 + 20*7 + 1.
    RequireLayerLoop(E2bFixture(), 307);
}
void TestE4bPrefillDispatchesTheLayerSequence() {
    // 1 + 24*12 + 18*7 + 1. NEITHER NUMBER FOLLOWS FROM E2B'S: more layers,
    // more owning ones, and a second KV head, so all three terms move.
    RequireLayerLoop(E4bFixture(), 416);
}

/// \brief the four places Phi-4's data flow is WRONG for Gemma 4
///
/// 1. `ssmlp`'s block output is PLANE 0 (`skip_sum`), not plane 1
///    (`normalized`) -- the opposite of the silu family. Plane 1 comes back
///    UNTOUCHED under `post_feedforward_layernorm`, but must still be bound,
///    because an unbound plane dispatches a null pointer. The driver measured
///    the operands the other way round: `normalized` returns a plane of EXACT
///    ZEROS, which propagates through `ple` and out of all 35 layers as a
///    finite, correctly-shaped vector of zeros.
/// 2. The next layer's input comes from `ple`'s SECOND output, not from
///    `ssmlp`. That is why no layer boundary touches the host.
/// 3. `ple`'s first output is the NEW RESIDUAL, written over the one `ssmlp`
///    just read.
/// 4. Layer 0's input comes from the ONE standalone `rmsnorm`.
///
/// Compared on `operand_storage` throughout, never on `operands`:
/// `create_tensor_window` mints a fresh handle every call, so two windows onto
/// one allocation are unequal by handle and equal by storage.
void RequireLayerDataFlow(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);
    const auto& dispatches = fake_corelib::GetState().dispatches;

    const auto ssmlp = IndicesOfKind("ssmlp");
    const auto ple = IndicesOfKind("ple");
    TEST_REQUIRE(ssmlp.size() == static_cast<std::size_t>(shape.layers));
    TEST_REQUIRE(ple.size() == static_cast<std::size_t>(shape.layers));

    // LAYER 0's Q PROJECTION READS THE STANDALONE NORM'S OUTPUT. Dispatch 0 is
    // that norm ({input, output}) and dispatch 1 is the Q matmul
    // ({input, output}).
    TEST_REQUIRE(dispatches[0].kind == "rmsnorm");
    TEST_REQUIRE(dispatches[1].kind == "matmul");
    TEST_REQUIRE(dispatches[1].operand_storage[0] == dispatches[0].operand_storage[1]);
    // ...and it normalized THE RESIDUAL STREAM, which layer 0's own ssmlp then
    // reads as its residual operand (index 1).
    TEST_REQUIRE(dispatches[0].operand_storage[0] ==
                 dispatches[ssmlp[0]].operand_storage[1]);

    // ---- EVERY HANDOFF INSIDE A LAYER, NOT ONLY THE MLP END OF IT
    //
    // ADDED BECAUSE A MUTATION SURVIVED. Making the output projection write
    // `skip_sum` instead of `hidden` left this whole file green: the
    // post-attention norm then normalizes the layer's own INPUT, the
    // attention output is discarded, and the model becomes attention-free --
    // finite, correctly shaped, fluent and wrong, which is the exact failure
    // class this task is about. The assertions above walked the chain from
    // `ssmlp` onward and simply did not reach back to attention.
    //
    // So: walk the dispatch list with a cursor that mirrors the driver's own
    // sequence, and require each operator's input to be the previous one's
    // output, ON STORAGE.
    std::size_t at = 1;   // dispatch 0 is the standalone norm, checked above
    // BOUNDS-CHECKED, so a loop that dispatches too FEW operators fails with
    // a readable line rather than walking off the end of the vector. A red run
    // that crashes is a red run nobody can read.
    const auto next = [&]() -> const fake_corelib::DispatchRecord& {
        TEST_REQUIRE(at < dispatches.size());
        return dispatches[at++];
    };
    const void* previous_layer_input = dispatches[0].operand_storage[1];
    const void* attention_key = nullptr;
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const bool owns = layer < shape.kv_layers;
        const auto& q_projection = next();
        TEST_REQUIRE(q_projection.kind == "matmul");
        // THE LAYER'S INPUT: the standalone norm on layer 0, the previous
        // layer's `ple` second output on every other.
        TEST_REQUIRE(q_projection.operand_storage[0] == previous_layer_input);

        const void* key_projection = nullptr;
        const void* value_projection = nullptr;
        if (owns) {
            const auto& k_projection = next();
            const auto& v_projection = next();
            TEST_REQUIRE(k_projection.kind == "matmul");
            TEST_REQUIRE(v_projection.kind == "matmul");
            // K and V read the same input Q does, not each other's output.
            TEST_REQUIRE(k_projection.operand_storage[0] == previous_layer_input);
            TEST_REQUIRE(v_projection.operand_storage[0] == previous_layer_input);
            key_projection = k_projection.operand_storage[1];
            value_projection = v_projection.operand_storage[1];
            TEST_REQUIRE(key_projection != value_projection);

            const auto& k_norm = next();
            TEST_REQUIRE(k_norm.kind == "rmsnorm");
            TEST_REQUIRE(k_norm.operand_storage[0] == key_projection);
            // NOT IN PLACE: corelib measures in-place rmsnorm corrupting at
            // the row counts a prefill QK-norm runs at.
            TEST_REQUIRE(k_norm.operand_storage[1] != key_projection);
            if (!attention_key) attention_key = k_norm.operand_storage[1];
            TEST_REQUIRE(k_norm.operand_storage[1] == attention_key);

            for (std::int64_t kv_head = 0; kv_head < shape.kv_heads; ++kv_head) {
                const auto& v_norm = next();
                TEST_REQUIRE(v_norm.kind == "rmsnorm");
                // V's norm reads the STAGING buffer the projection wrote...
                TEST_REQUIRE(v_norm.operand_storage[0] == value_projection);
                // ...and writes somewhere else entirely: the cache. Which
                // cache, and at what offset, is RequireVNormWritesTheCache.
                TEST_REQUIRE(v_norm.operand_storage[1] != value_projection);
            }
        }

        const auto& q_norm = next();
        TEST_REQUIRE(q_norm.kind == "rmsnorm");
        TEST_REQUIRE(q_norm.operand_storage[0] == q_projection.operand_storage[1]);
        TEST_REQUIRE(q_norm.operand_storage[1] != q_projection.operand_storage[1]);

        const auto& attention = next();
        TEST_REQUIRE(attention.kind == "mha");
        // ATTENTION READS THE NORMED Q AND K, not the raw projections. These
        // gammas ARE the attention scale -- skipping a norm whose gamma "looks
        // constant" is an 8x or 16x logit error -- so attending over the
        // unnormed buffer is the same bug as dropping the norm.
        TEST_REQUIRE(attention.operands.size() == 3);
        TEST_REQUIRE(attention.operand_storage[0] == q_norm.operand_storage[1]);
        // A SHARING LAYER STILL PASSES A K HANDLE: the kvshare ELFs are not
        // handed a K operand and do not look at it, but "a handle must still
        // be passed". It is the normed-K buffer on every layer.
        TEST_REQUIRE(attention.operand_storage[1] == attention_key);

        const auto& output_projection = next();
        TEST_REQUIRE(output_projection.kind == "matmul");
        // THE OUTPUT PROJECTION CONSUMES ATTENTION'S OUTPUT...
        TEST_REQUIRE(output_projection.operand_storage[0] ==
                     attention.operand_storage[2]);

        const auto& post_attention = next();
        TEST_REQUIRE(post_attention.kind == "rmsnorm");
        // ...AND THE POST-ATTENTION NORM CONSUMES THE PROJECTION'S. This is
        // the pair the surviving mutation broke: writing the projection
        // somewhere else leaves this norm reading the layer's own input, and
        // the attention output is simply dropped.
        TEST_REQUIRE(post_attention.operand_storage[0] ==
                     output_projection.operand_storage[1]);

        const auto& mlp = next();
        TEST_REQUIRE(mlp.kind == "ssmlp");
        // ssmlp's INPUT is the post-attention norm's output, and its RESIDUAL
        // is the stream -- not the other way round.
        TEST_REQUIRE(mlp.operand_storage[0] == post_attention.operand_storage[1]);
        TEST_REQUIRE(mlp.operand_storage[1] != post_attention.operand_storage[1]);

        const auto& block = next();
        TEST_REQUIRE(block.kind == "ple");
        TEST_REQUIRE(&mlp == &dispatches[ssmlp[static_cast<std::size_t>(layer)]]);
        TEST_REQUIRE(&block == &dispatches[ple[static_cast<std::size_t>(layer)]]);
        // THIS LAYER'S OWN PER-LAYER EMBEDDING PLANE, not some other layer's.
        // ADDED BECAUSE A MUTATION SURVIVED: binding plane 0 to every layer
        // left the whole suite green. `ple` is the one operand whose value is
        // per layer and whose SHAPE is identical on all of them, so nothing --
        // not corelib, not this fake's extent checks -- can tell them apart.
        // The planes are written in layer order at construction, so create
        // `12 + layer` is this layer's.
        TEST_REQUIRE(ParentOf(block.operands[1]) ==
                     Bf16Creates()[static_cast<std::size_t>(12 + layer)]->object);
        previous_layer_input = block.operand_storage[3];
    }
    // The cursor landed exactly on lm_head, so the walk consumed every
    // dispatch the loop made and no more.
    TEST_REQUIRE(at + 1 == dispatches.size());

    for (std::size_t layer = 0; layer < ple.size(); ++layer) {
        const auto& mlp = dispatches[ssmlp[layer]];
        const auto& block = dispatches[ple[layer]];
        // ssmlp is {input, residual, skip_sum, normalized};
        // ple is {x, ple, out, norm_out}.
        //
        // PLANE 0 IS WHAT PLE CONSUMES.
        TEST_REQUIRE(mlp.operand_storage[2] == block.operand_storage[0]);
        // PLANE 1 IS NOT. It is bound -- an unbound plane is a null pointer --
        // and nothing downstream reads it: the buffer it names is the one
        // `ple` immediately OVERWRITES with its own second output.
        TEST_REQUIRE(mlp.operand_storage[3] != block.operand_storage[0]);
        TEST_REQUIRE(mlp.operand_storage[3] == block.operand_storage[3]);
        // PLE'S FIRST OUTPUT IS THE NEW RESIDUAL, over the one ssmlp read.
        TEST_REQUIRE(block.operand_storage[2] == mlp.operand_storage[1]);
        // All four of ssmlp's tensors are separate allocations, and so are
        // ple's x/out/norm_out. The fake enforces both, but a dispatch that
        // was REFUSED records nothing, so stating it here is what makes a
        // silently-dropped layer visible as well.
        TEST_REQUIRE(mlp.operand_storage[0] != mlp.operand_storage[1]);
        TEST_REQUIRE(block.operand_storage[0] != block.operand_storage[2]);

        // THE NEXT LAYER'S INPUT IS PLE'S SECOND OUTPUT, NOT SSMLP'S. The
        // dispatch immediately after `ple` is the next layer's Q projection --
        // or, on the last layer, lm_head, which reads a separate one-row
        // buffer instead.
        const auto follows = ple[layer] + 1;
        TEST_REQUIRE(follows < dispatches.size());
        TEST_REQUIRE(dispatches[follows].kind == "matmul");
        if (layer + 1 < ple.size()) {
            TEST_REQUIRE(dispatches[follows].operand_storage[0] == block.operand_storage[3]);
        } else {
            // lm_head runs at M == 1 over its own input tensor, written from
            // the read-back -- so it must NOT be the loop's hidden buffer.
            TEST_REQUIRE(dispatches[follows].operand_storage[0] != block.operand_storage[3]);
        }
    }
}

void TestE2bLayerDataFlowTakesPlaneZeroAndPlesSecondOutput() {
    RequireLayerDataFlow(E2bFixture());
}
void TestE4bLayerDataFlowTakesPlaneZeroAndPlesSecondOutput() {
    RequireLayerDataFlow(E4bFixture());
}

/// \brief a shape, written the way corelib declares one
using Shape = std::vector<std::int64_t>;

std::string Describe(const Shape& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) text += ", ";
        text += std::to_string(shape[i]);
    }
    return text + "]";
}

/// \brief one operand of `dispatch` has shape `expected`, and NAMES THE LAYER
///        AND THE OPERAND when it does not
/// \note `TEST_REQUIRE` stringifies its condition, which for a shape
///       comparison inside a 35-iteration loop says nothing about WHICH layer
///       or WHICH operand disagreed, and prints neither shape. A geometry
///       failure is a wrong number, so the message has to carry the numbers
///       -- otherwise the first red run costs a debugging session to locate.
/// \note THE OPERAND INDEX IS BOUNDS-CHECKED HERE rather than at every call
///       site. `operand_shapes` is filled by the fake, and a fake that
///       stopped filling it would otherwise make this test read off the end
///       of a vector -- which is a crash, and a red run that crashes takes
///       every later test's result with it. That is not hypothetical: it is
///       what mutating the fake's retention away produced before this check
///       existed.
void RequireShape(const fake_corelib::DispatchRecord& dispatch, std::size_t operand,
                  const Shape& expected, std::int64_t layer, const char* what) {
    const auto& shapes = dispatch.operand_shapes;
    if (operand >= shapes.size()) {
        throw std::runtime_error("layer " + std::to_string(layer) + " " + what +
                                 ": dispatch '" + dispatch.kind + "' recorded only " +
                                 std::to_string(shapes.size()) + " operand shapes, " +
                                 "wanted index " + std::to_string(operand));
    }
    if (shapes[operand] == expected) return;
    throw std::runtime_error("layer " + std::to_string(layer) + " " + what + " is " +
                             Describe(shapes[operand]) + ", expected " +
                             Describe(expected));
}

/// \brief every layer's operands are bound at ITS OWN geometry's WIDTHS
///
/// ADDED BECAUSE A MUTATION SURVIVED, and it is the widest one this project
/// has found: replacing `geometry_for(swa)` with `geometry_for(false)` in the
/// layer loop -- so that every sliding layer binds full-attention-width
/// windows -- left the entire suite green. On E2B that is twenty of
/// thirty-five layers projecting Q, K and V into 4096/512-wide windows where
/// the weights were packed for 2048/256, normalizing at `k = 512` under a
/// 256-wide gamma, and reading the output projection through a window twice
/// as wide as its weight.
///
/// WHY NOTHING SAW IT. Gemma 4 runs two geometries over ONE set of
/// allocations, sized for the wider and windowed down per layer on the COLUMN
/// axis. So the wrong geometry's windows are the same tensors, at the same
/// offsets, over the same storage, with the same row counts:
///   - `operand_storage` is unchanged, so every data-flow assertion passes;
///   - `rows` is unchanged, so `RowsAgree` and `OnGrid` pass;
///   - the attention DESCRIPTOR is selected separately, by `swa`, so
///     `mha_desc.head_size == LayerHeadDim(layer)` still passes;
///   - and `DispatchRecord` retained no column extent at all.
/// The fifth of those is the one that was fixable, and
/// `DispatchRecord::operand_shapes` is the fix.
///
/// WHAT THIS DOES NOT COVER, STATED BECAUSE IT MATTERS MORE THAN USUAL HERE.
/// This test is, as far as anyone on this port has been able to establish,
/// THE ONLY THING that stands between that mutation and a wrong model.
/// corelib.h says a matmul takes K and N FROM THE WEIGHTS and "the weights
/// were packed for one (K, N) and cannot be dispatched at another"
/// (corelib.h:952) -- it does not say the bound tensor's column extent is
/// checked against them, and a too-WIDE output buffer is not obviously out of
/// bounds. So it is not known whether real corelib would refuse a
/// width-mismatched dispatch, and this test must not be read as evidence that
/// it would. What it asserts is narrower and is all a fake can honestly
/// assert: THE ENGINE BOUND THE WINDOWS ITS OWN GEOMETRY DEFINES. Whether
/// hardware objects to the alternative is a D3 question.
///
/// BOTH MODELS, because the two rows differ in `kv_heads` (1 and 2) and
/// therefore in the V destination's rank, and because a tabulated answer that
/// is right on one row and wrong on the other is this port's recurring
/// failure -- mutation M4 was E4B-only.
void RequireLayerGeometry(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);
    const auto& dispatches = fake_corelib::GetState().dispatches;

    // THE TWO GEOMETRIES HAVE TO DIFFER, OR EVERY ASSERTION BELOW IS VACUOUS
    // -- it would be comparing each layer against a width all of them share.
    // This is the fourth vacuous-assertion finding on this port; state the
    // premise rather than rely on it.
    TEST_REQUIRE(shape.head_dim != shape.global_head_dim);
    const auto& kinds = shape.layer_is_swa;
    TEST_REQUIRE(std::find(kinds.begin(), kinds.end(), true) != kinds.end());
    TEST_REQUIRE(std::find(kinds.begin(), kinds.end(), false) != kinds.end());

    // THREE LIVE ROWS RUN A 128-ROW KERNEL -- the prefill grid's floor, which
    // RequireLayerLoop asserts independently off the `ple` dispatches.
    const std::int64_t padded = 128;
    const auto hidden = shape.hidden;
    const auto ple_dim = shape.ple_dim;

    // LAYER 0'S STANDALONE NORM runs at the RESIDUAL width, which belongs to
    // no geometry: both geometries' layers read and write `hidden`.
    TEST_REQUIRE(!dispatches.empty());
    TEST_REQUIRE(dispatches[0].kind == "rmsnorm");
    RequireShape(dispatches[0], 0, Shape{padded, hidden}, 0, "standalone norm input");
    RequireShape(dispatches[0], 1, Shape{padded, hidden}, 0, "standalone norm output");

    std::size_t at = 1;
    // BOUNDS-CHECKED for the reason RequireLayerDataFlow's cursor is: a loop
    // that dispatches too few operators must fail with a readable line rather
    // than walk off the end.
    const auto next = [&]() -> const fake_corelib::DispatchRecord& {
        TEST_REQUIRE(at < dispatches.size());
        return dispatches[at++];
    };
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const bool owns = layer < shape.kv_layers;
        // THE LAYER'S OWN HEAD SIZE. `LayerHeadDim` is the engine's own rule
        // read back -- but it is also what the attention DESCRIPTOR is built
        // from, and the descriptor was already asserted against it. The point
        // of this test is that the WINDOWS agree with the descriptor: under
        // the mutation they did not, and only the descriptor was checked.
        const auto head = shape.LayerHeadDim(layer);
        const auto q_dim = shape.q_heads * head;
        const auto kv_dim = shape.kv_heads * head;

        const auto& q_projection = next();
        TEST_REQUIRE(q_projection.kind == "matmul");
        RequireShape(q_projection, 0, Shape{padded, hidden}, layer,
                     "Q projection input");
        RequireShape(q_projection, 1, Shape{padded, q_dim}, layer,
                     "Q projection output");

        if (owns) {
            const auto& k_projection = next();
            TEST_REQUIRE(k_projection.kind == "matmul");
            RequireShape(k_projection, 0, Shape{padded, hidden}, layer,
                         "K projection input");
            RequireShape(k_projection, 1, Shape{padded, kv_dim}, layer,
                         "K projection output");

            const auto& v_projection = next();
            TEST_REQUIRE(v_projection.kind == "matmul");
            RequireShape(v_projection, 0, Shape{padded, hidden}, layer,
                         "V projection input");
            // V'S DESTINATION IS THE ONE OPERAND WHOSE RANK DEPENDS ON THE
            // MODEL: a plain 2-D `[padded, head]` at exactly one KV head, and
            // the scattering kernel's 3-D `[kv_heads, max_seq, head]` above
            // it. Both end in the HEAD WIDTH, which is what this test is
            // about, and the head width is what the mutation moved.
            TEST_REQUIRE(v_projection.operand_shapes.size() == 2);
            const auto& destination = v_projection.operand_shapes[1];
            if (shape.kv_heads == 1) {
                RequireShape(v_projection, 1, Shape{padded, head}, layer,
                             "V projection destination");
            } else {
                TEST_REQUIRE(destination.size() == 3);
                TEST_REQUIRE(destination[0] == shape.kv_heads);
                // THE MIDDLE EXTENT IS DELIBERATELY NOT ASSERTED HERE. It is
                // the 3-D staging window's per-head pitch, which has its own
                // open finding (review-C8 I-4: changing it from
                // `kMaxSequenceLength` to `padded` desynchronizes the
                // scattering kernel from `v_src`'s offsets and no test sees
                // it). That is scheduled for the hardware phase; the
                // retention this test needed is what makes it writable, and
                // writing it is not this task's.
                TEST_REQUIRE(destination[2] == head);
            }

            const auto& k_norm = next();
            TEST_REQUIRE(k_norm.kind == "rmsnorm");
            // QK-NORM NORMALIZES ONE HEAD, so its rows are tokens x heads and
            // its WIDTH is the head size -- which is also the width the gamma
            // was packed at. A layer running this at the other geometry's
            // width normalizes under a gamma of the wrong length.
            RequireShape(k_norm, 0, Shape{padded * shape.kv_heads, head}, layer,
                         "K-norm input");
            RequireShape(k_norm, 1, Shape{padded * shape.kv_heads, head}, layer,
                         "K-norm output");

            for (std::int64_t kv_head = 0; kv_head < shape.kv_heads; ++kv_head) {
                const auto& v_norm = next();
                TEST_REQUIRE(v_norm.kind == "rmsnorm");
                // ONE HEAD'S PLANE, in and out -- the output being a window of
                // the KV cache, whose row pitch is this same head width.
                RequireShape(v_norm, 0, Shape{padded, head}, layer, "V-norm input");
                RequireShape(v_norm, 1, Shape{padded, head}, layer,
                             "V-norm cache window");
            }
        }

        const auto& q_norm = next();
        TEST_REQUIRE(q_norm.kind == "rmsnorm");
        RequireShape(q_norm, 0, Shape{padded * shape.q_heads, head}, layer,
                     "Q-norm input");
        RequireShape(q_norm, 1, Shape{padded * shape.q_heads, head}, layer,
                     "Q-norm output");

        const auto& attention = next();
        TEST_REQUIRE(attention.kind == "mha");
        // {q, k, out}. The descriptor's `head_size` is asserted elsewhere; what
        // is asserted HERE is that the windows handed to the same dispatch
        // agree with it, which is exactly what the mutation broke.
        RequireShape(attention, 0, Shape{padded, q_dim}, layer, "attention Q");
        RequireShape(attention, 1, Shape{padded, kv_dim}, layer, "attention K");
        RequireShape(attention, 2, Shape{padded, q_dim}, layer, "attention output");
        TEST_REQUIRE(attention.mha_desc.head_size == head);

        const auto& output_projection = next();
        TEST_REQUIRE(output_projection.kind == "matmul");
        // THE PAIR THAT MAKES THIS TEST TWO-SIDED: the output projection reads
        // at the geometry's width and writes at the residual width, so a
        // layer on the wrong geometry is wrong on its INPUT while its output
        // still matches. Both are stated.
        RequireShape(output_projection, 0, Shape{padded, q_dim}, layer,
                     "output projection input");
        RequireShape(output_projection, 1, Shape{padded, hidden}, layer,
                     "output projection output");

        const auto& post_attention = next();
        TEST_REQUIRE(post_attention.kind == "rmsnorm");
        RequireShape(post_attention, 0, Shape{padded, hidden}, layer,
                     "post-attention norm input");
        RequireShape(post_attention, 1, Shape{padded, hidden}, layer,
                     "post-attention norm output");

        const auto& mlp = next();
        TEST_REQUIRE(mlp.kind == "ssmlp");
        // {input, residual, skip_sum, normalized} -- all four at the residual
        // width, including the plane the gemma topology never writes.
        for (std::size_t operand = 0; operand < 4; ++operand) {
            RequireShape(mlp, operand, Shape{padded, hidden}, layer, "ssmlp operand");
        }

        const auto& block = next();
        TEST_REQUIRE(block.kind == "ple");
        // {x, ple, out, norm_out}. The per-layer embedding plane is the one
        // operand at `ple_dim`, and it is the SAME width on both geometries --
        // so this row is not a geometry check, it is the check that the
        // per-layer plane was not bound at the residual width.
        RequireShape(block, 0, Shape{padded, hidden}, layer, "ple x");
        RequireShape(block, 1, Shape{padded, ple_dim}, layer, "ple per-layer plane");
        RequireShape(block, 2, Shape{padded, hidden}, layer, "ple out");
        RequireShape(block, 3, Shape{padded, hidden}, layer, "ple norm_out");
    }

    // The cursor landed exactly on lm_head, which runs at M == 1 over its own
    // one-row buffer: `hidden -> vocab`, a width belonging to neither
    // geometry.
    TEST_REQUIRE(at + 1 == dispatches.size());
    const auto& head_matmul = dispatches[at];
    TEST_REQUIRE(head_matmul.kind == "matmul");
    RequireShape(head_matmul, 0, Shape{1, hidden}, shape.layers, "lm_head input");
    RequireShape(head_matmul, 1, Shape{1, shape.vocab}, shape.layers,
                 "lm_head output");
}

void TestE2bEveryLayerBindsItsOwnGeometrysWidths() {
    RequireLayerGeometry(E2bFixture());
}
void TestE4bEveryLayerBindsItsOwnGeometrysWidths() {
    RequireLayerGeometry(E4bFixture());
}

/// \brief V's norm writes THE CACHE, once per KV head, head-major
///
/// The one regression against every other driver on this stack. Elsewhere
/// v_proj's output tensor IS a window of the KV cache, and the scattering
/// kernel fills the cache directly. Gemma 4 RMS-normalizes V between the
/// projection and the cache, and no norm can scatter -- it takes its row count
/// from the tensor's byte size and writes contiguously -- so the projection
/// lands in a staging buffer and THE NORM IS WHAT WRITES THE CACHE, once per
/// head, at that head's own contiguous run.
/// AND IT WRITES AT `position`, WHICH ONLY A DECODE CAN SHOW.
///
/// ADDED BECAUSE A MUTATION SURVIVED, and this is the fourth vacuous test
/// found on this port. Deleting `+ position * head` from the cache window
/// left the whole suite green: every decode's V then lands on cache ROW 0,
/// no generated token's value vector ever reaches the cache, every layer
/// attends over a tail the prefill left behind, and the text stays fluent.
/// The assertion below was already here and already correct -- but the only
/// pass it was given was a PREFILL, where `position == 0` makes the term it
/// exists to pin contribute nothing. An assertion whose expected value
/// coincides with the degenerate case cannot fail on the degenerate case.
///
/// So the same assertion is now run at THREE positions -- 0, 3 and 4 -- and
/// the two non-zero ones are where it means something. Two decodes rather
/// than one on purpose: one would catch "the term is missing", two also catch
/// "the term is a constant".
void RequireVNormWritesTheCache(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    const auto& state = fake_corelib::GetState();
    const auto buffers = Bf16Creates();
    const auto slots = SlotsByHandle(*harness.engine);

    const auto cache_index = [&](std::int64_t cache, bool value_cache) {
        // The allocation order asserted by RequireBuffersAndCaches: ten
        // activation buffers, the head's two, one plane per layer, then K and
        // V per owning layer.
        return static_cast<std::size_t>(12 + shape.layers + 2 * cache + (value_cache ? 1 : 0));
    };

    /// \param from the first dispatch index of the pass to examine -- the
    ///        list is cumulative across passes, so a decode's V-norms are the
    ///        tail of it and asserting over the WHOLE list would re-assert
    ///        the prefill's at the decode's expected offset
    /// \param position the absolute row this pass's V rows start at
    const auto require_pass = [&](std::size_t from, std::int64_t position) {
        std::map<std::int64_t, std::vector<const fake_corelib::DispatchRecord*>> by_layer;
        for (std::size_t i = from; i < state.dispatches.size(); ++i) {
            const auto& dispatch = state.dispatches[i];
            if (dispatch.kind != "rmsnorm" || !dispatch.weights) continue;
            const auto found = slots.find(dispatch.weights);
            if (found == slots.end()) continue;
            for (std::int64_t layer = 0; layer < shape.kv_layers; ++layer) {
                if (found->second == Blk(layer, ".attn_v_norm(ones)"))
                    by_layer[layer].push_back(&dispatch);
            }
        }
        TEST_REQUIRE(by_layer.size() == static_cast<std::size_t>(shape.kv_layers));

        for (std::int64_t layer = 0; layer < shape.kv_layers; ++layer) {
            const auto& norms = by_layer[layer];
            // ONE PER KV HEAD. Not one per layer: with two KV heads the second
            // head's plane never reaches the cache, and the layers that share
            // this cache attend over whatever was there.
            TEST_REQUIRE(norms.size() == static_cast<std::size_t>(shape.kv_heads));
            const auto head = shape.LayerHeadDim(layer);
            for (std::int64_t kv_head = 0; kv_head < shape.kv_heads; ++kv_head) {
                const auto& norm = *norms[static_cast<std::size_t>(kv_head)];
                // {input, output}: the OUTPUT is a window of THIS layer's V
                // cache.
                void* destination = norm.operands[1];
                TEST_REQUIRE(ParentOf(destination) ==
                             buffers[cache_index(layer, /*value_cache=*/true)]->object);
                // HEAD-MAJOR, AT THIS PASS'S POSITION. The cache is
                // [kv_heads, max_seq, head], so head h starts at
                // h * max_seq * head and this pass's rows start `position`
                // further in.
                //
                // THE TWO TERMS CANNOT COINCIDE at the positions used here:
                // `position` is 0, 3 or 4 and max_seq is 4096, so
                // `position * head` is never a multiple of
                // `max_seq * head` except at position 0 -- a dropped
                // `position` term and a dropped `kv_head` term are distinct
                // failures and each is visible on its own.
                TEST_REQUIRE(
                    OffsetOf(destination) ==
                    static_cast<std::size_t>(
                        kv_head * flm::gemma4::kMaxSequenceLength * head + position * head));
                // And it is NOT the K cache: the two are the same shape, so a
                // mix-up binds cleanly and attends over the keys as values.
                TEST_REQUIRE(ParentOf(destination) !=
                             buffers[cache_index(layer, /*value_cache=*/false)]->object);
            }
        }
    };

    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);
    // THE PREFILL, at position 0 -- where the `position` term is invisible.
    // Kept because the head-major term and the cache identity are real here
    // and because a prefill is the pass that fills most of the cache.
    require_pass(0, 0);

    // THE DECODES, which is where the assertion becomes capable of failing.
    // Three prompt tokens were consumed, so the first generated token is at
    // row 3 and the second at row 4.
    std::size_t before = state.dispatches.size();
    (void)harness.engine->forward(7);
    require_pass(before, 3);

    before = state.dispatches.size();
    (void)harness.engine->forward(8);
    require_pass(before, 4);
}

void TestE2bVNormWritesTheCacheOncePerKvHead() { RequireVNormWritesTheCache(E2bFixture()); }
void TestE4bVNormWritesTheCacheOncePerKvHead() { RequireVNormWritesTheCache(E4bFixture()); }

void TestE4bVProjectionLandsThroughAStridedWindow() {
    Harness h(E4bFixture());
    (void)h.engine->forward(1);
    const auto& windows = fake_corelib::GetState().tensor_windows;
    const auto v = std::find_if(windows.begin(), windows.end(),
                                [](const auto& w) { return w.shape.size() == 3; });
    TEST_REQUIRE(v != windows.end());
    TEST_REQUIRE(v->strides.size() == 3 && v->strides[2] == 1);
    TEST_REQUIRE(v->strides[1] == v->shape[2]);
    TEST_REQUIRE(v->strides[0] == 4096 * v->strides[1]);
}

/// \brief every attention dispatch gets its own geometry's rotary pair, its
///        owner's caches, and the descriptor for its cache role
///
/// FOUR THINGS, ALL SILENT WHEN WRONG.
///
/// - THE ROTARY PAIR. One pair used everywhere still converges to fluent text.
/// - THE CACHE. A sharing layer reads "the last own-cache layer of its own
///   kind" -- 13 (sliding) and 14 (full) on E2B, 22 and 23 on E4B. Neither
///   pair follows from the other or from a layer count, which is what makes a
///   tabulated answer invisibly wrong on one model while right on the other.
/// - `kv_shared`. "An owning layer given a kvshare descriptor never writes its
///   own cache; a sharing layer given an owning one overwrites somebody
///   else's. Neither errors."
/// - `scale`. Exactly 1.0f, on all four descriptors. It is Gemma 4's actual
///   softmax scale -- the 1/sqrt(head) lives in the learned QK-norm gammas --
///   and 0.0f selects a kernel family that does not ship for this model.
void RequireAttentionBindings(const Fixture& fixture) {
    Harness harness(fixture);
    const auto& shape = harness.shape;
    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);
    const auto& dispatches = fake_corelib::GetState().dispatches;
    const auto rotary = Fp32Creates();
    const auto buffers = Bf16Creates();
    TEST_REQUIRE(rotary.size() == 4);

    const auto mha = IndicesOfKind("mha");
    TEST_REQUIRE(mha.size() == static_cast<std::size_t>(shape.layers));
    for (std::int64_t layer = 0; layer < shape.layers; ++layer) {
        const auto& attention = dispatches[mha[static_cast<std::size_t>(layer)]];
        const bool swa = shape.layer_is_swa[static_cast<std::size_t>(layer)];
        const auto owner = shape.layer_cache_owner[static_cast<std::size_t>(layer)];

        TEST_REQUIRE(attention.has_mha_desc);
        TEST_REQUIRE(attention.mha_desc.kv_shared == (layer < shape.kv_layers ? 0 : 1));
        TEST_REQUIRE(attention.mha_desc.head_size == shape.LayerHeadDim(layer));
        TEST_REQUIRE(attention.mha_desc.window == (swa ? shape.sliding_window : 0));
        TEST_REQUIRE(attention.mha_desc.scale == 1.0f);
        TEST_REQUIRE(attention.mha_desc.max_seq == flm::gemma4::kMaxSequenceLength);
        TEST_REQUIRE(attention.mha_desc.rope_dim == attention.mha_desc.head_size);
        // A PREFILL STARTS AT POSITION 0 -- see
        // TestEveryLayerIsToldTheSamePositionAndADecodeAdvancesIt for the
        // half of this that a prefill cannot show.
        TEST_REQUIRE(attention.mha_position == 0);

        // The rotary pair: sliding is creates 0/1, full is 2/3.
        TEST_REQUIRE(attention.mha_cos == rotary[swa ? 0 : 2]->object);
        TEST_REQUIRE(attention.mha_sin == rotary[swa ? 1 : 3]->object);

        // The caches, WHOLE, not windowed: "the caches are pinned to max_seq,
        // not grown to fit. The kernel addresses them at that row pitch."
        const auto k_index = static_cast<std::size_t>(12 + shape.layers + 2 * owner);
        TEST_REQUIRE(attention.mha_k_cache == buffers[k_index]->object);
        TEST_REQUIRE(attention.mha_v_cache == buffers[k_index + 1]->object);
    }
}

void TestE2bAttentionBindsItsOwnRotaryPairAndItsOwnersCaches() {
    RequireAttentionBindings(E2bFixture());
}
void TestE4bAttentionBindsItsOwnRotaryPairAndItsOwnersCaches() {
    RequireAttentionBindings(E4bFixture());
}

/// \brief every layer of one pass is told the SAME position, and a decode
///        advances it
///
/// ADDED BECAUSE A MUTATION SURVIVED: passing a literal 0 here instead of the
/// engine's position left the whole suite green. `position` is the one scalar
/// that makes a decode step different from a prefill -- "Q's shape is a
/// position range ... together they span [position, position + M), which is
/// what the rotary and the causal mask need" -- so an engine that always
/// passed 0 would rope every token as the first and mask every step to a
/// one-token context. Finite, plausible, fluent, wrong.
///
/// ONE POSITION PER SYNCHRONIZE INTERVAL IS ALSO A CORELIB RULE, not just an
/// arithmetic one: in the token phase `flat_mha_bf16` derives a scalar slot
/// holding `position + rows` into "a single per-stream buffer rewritten per
/// dispatch", which is safe only because every layer of one step presents the
/// same position. Two decodes at different positions inside one interval race
/// it. This engine synchronizes once per pass, so the rule holds -- and the
/// first half of that sentence is what the loop below asserts.
void TestEveryLayerIsToldTheSamePositionAndADecodeAdvancesIt() {
    Harness harness(E2bFixture());
    const auto& state = fake_corelib::GetState();
    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);

    const auto after_prefill = state.dispatches.size();
    (void)harness.engine->forward(7);
    (void)harness.engine->forward(8);

    std::vector<std::int64_t> positions;
    for (std::size_t i = after_prefill; i < state.dispatches.size(); ++i) {
        if (state.dispatches[i].kind == "mha")
            positions.push_back(state.dispatches[i].mha_position);
    }
    // 35 layers in each of the two decode steps.
    TEST_REQUIRE(positions.size() == 70);
    for (std::size_t i = 0; i < 35; ++i) TEST_REQUIRE(positions[i] == 3);
    for (std::size_t i = 35; i < 70; ++i) TEST_REQUIRE(positions[i] == 4);
}

/// \brief the pass reads the LAST live row, and the head's logits
void TestE2bReadsTheLastLiveRowAndNothingElse() {
    Harness harness(E2bFixture());
    const auto& shape = harness.shape;
    std::vector<int> ids{2, 5, 9, 11};
    (void)harness.engine->prefill(ids);
    const auto& reads = fake_corelib::GetState().tensor_reads;
    const auto buffers = Bf16Creates();

    // EXACTLY TWO READS. The layer loop touches the host nowhere: `ple`'s
    // second output is the next layer's normed input, so the chain closes on
    // the device.
    TEST_REQUIRE(reads.size() == 2);
    // The last live row of `hidden` -- buffer 0 -- as FP32. The tensor is
    // BF16; corelib widens on the way out.
    TEST_REQUIRE(reads[0].tensor == buffers[0]->object);
    TEST_REQUIRE(reads[0].destination_type == ryzenai_corelib_data_type_fp32);
    TEST_REQUIRE(reads[0].count == static_cast<std::size_t>(shape.hidden));
    TEST_REQUIRE(reads[0].offset ==
                 static_cast<std::size_t>((ids.size() - 1) * shape.hidden));
    TEST_REQUIRE(reads[0].offset != 0);   // ...and it is NOT row 0
    TEST_REQUIRE(reads[0].accepted);
    // Then the logits, one row, from the head's own output tensor.
    TEST_REQUIRE(reads[1].tensor == buffers[11]->object);
    TEST_REQUIRE(reads[1].destination_type == ryzenai_corelib_data_type_fp32);
    TEST_REQUIRE(reads[1].count == static_cast<std::size_t>(shape.vocab));
    TEST_REQUIRE(reads[1].offset == 0);
}

/// \brief the logits a pass returns have been SOFTCAPPED
///
/// ADDED BECAUSE A MUTATION SURVIVED: deleting the softcap call left the whole
/// suite green, because the fake's reads used to return zeros and
/// `tanh(0/30)*30` is zero. The fake now returns a synthetic constant well
/// outside the cap's near-linear region, so the transform is observable.
///
/// WHAT OMITTING THE CAP LOOKS LIKE OTHERWISE: nothing. The raw logits are
/// finite and well spread, the argmax usually agrees, and the text stays
/// fluent -- the cap is a monotonic squash, so it can only reorder logits
/// already far out in the tail. It changes every softmax it feeds, which is
/// why it is not optional and why it cannot be checked by looking at the
/// output's shape.
///
/// THE CAP IS READ FROM THE FILE (`gemma4.final_logit_softcapping`, 30.0 on
/// every shipped size), not written here, and a model that states 0.0 means
/// "no cap" -- which `tanh(x/0)` would answer with a plane of NaNs.
void TestTheReturnedLogitsAreSoftcapped() {
    Harness harness(E2bFixture());
    TEST_REQUIRE(harness.shape.logit_softcap == 30.0f);
    std::vector<int> ids{2, 5, 9};
    const auto logits = harness.engine->prefill(ids);

    const float cap = harness.shape.logit_softcap;
    const float expected = std::tanh(fake_corelib::kSyntheticReadValue / cap) * cap;
    std::vector<std::uint16_t> wanted(1);
    const std::vector<float> one{expected};
    flm::gemma4::FloatsToBf16(one, wanted);

    std::uint16_t actual = 0;
    std::memcpy(&actual, logits.data(), sizeof(actual));
    TEST_REQUIRE(actual == wanted[0]);
    // And it really is a TRANSFORM, not a copy: the uncapped value would have
    // come back as itself.
    std::vector<std::uint16_t> uncapped(1);
    const std::vector<float> raw{fake_corelib::kSyntheticReadValue};
    flm::gemma4::FloatsToBf16(raw, uncapped);
    TEST_REQUIRE(wanted[0] != uncapped[0]);
}

/// \brief a decode step runs the same layer structure at one row, and the
///        position advances by the number of tokens consumed
void TestDecodeRunsOneRowAndAdvancesThePosition() {
    Harness harness(E2bFixture());
    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 3);

    // NOT `fake_corelib::Reset()` HERE, and that cost twenty minutes the first
    // time: Reset clears `pad_answers_seen`, which the fake fills from the
    // engine's OWN pad-helper calls at construction. An engine that outlives a
    // Reset then has every dispatch refused by `OnGrid` for want of an answer
    // set -- a failure that looks exactly like a wrong row extent. Count the
    // delta instead.
    const auto& state = fake_corelib::GetState();
    const auto after_prefill = state.dispatches.size();
    const auto reads_after_prefill = state.tensor_reads.size();
    (void)harness.engine->forward(7);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 4);

    // THE SAME 307 DISPATCHES: a decode step is not a different model, it is
    // the same read sequence at one row.
    TEST_REQUIRE(state.dispatches.size() - after_prefill == 307);
    // AT ONE ROW, which is what selects the token PDI. Every rank-2 operand of
    // the decode's `ple` dispatches reports one row -- and `ple` is the op
    // whose M "rounds up to a row count the ELF set ships", so a decode that
    // ran at the prefill bucket would be reading 127 rows of somebody else's
    // residual stream.
    const auto ple = IndicesOfKind("ple");
    TEST_REQUIRE(ple.size() == 2 * 35);   // prefill's 35, then the decode's
    for (std::size_t i = 35; i < ple.size(); ++i) {
        TEST_REQUIRE(state.dispatches[ple[i]].rows == 1);
    }
    // Two more reads, and the hidden one is row 0 because there is one live
    // row: `(rows - 1) * hidden` is zero, not `position * hidden`.
    TEST_REQUIRE(state.tensor_reads.size() - reads_after_prefill == 2);
    TEST_REQUIRE(state.tensor_reads[reads_after_prefill].offset == 0);
}

/// \brief a cache read resolves through the OWNER, and its width is the
///        owner's head
///
/// `get_k_cache(layer, ...)` cannot index a cache by the layer: there are
/// `kv_layers` caches and not one per layer, and on E2B twenty layers own none
/// at all. A naive `k_cache[layer]` is out of bounds for exactly those twenty;
/// a `k_cache[min(layer, kv_layers-1)]` is in bounds and reads the wrong one.
/// Both are resolved here through `layer_cache_owner`.
///
/// AND THE WIDTH IS NOT CONSTANT ACROSS LAYERS. A sliding cache is 256 wide
/// and a full one 512, so a caller that assumed one width would read half a
/// row or run off the end of the other.
void TestACacheReadResolvesThroughTheOwner() {
    Harness harness(E2bFixture());
    const auto& shape = harness.shape;
    const auto& state = fake_corelib::GetState();
    const auto buffers = Bf16Creates();
    std::vector<int> ids{2, 5, 9};
    (void)harness.engine->prefill(ids);

    const auto require_read_of = [&](int layer, bool value_cache) {
        const auto before = state.tensor_reads.size();
        const auto row = value_cache ? harness.engine->get_v_cache(layer, 1)
                                     : harness.engine->get_k_cache(layer, 1);
        const auto owner = shape.layer_cache_owner[static_cast<std::size_t>(layer)];
        const auto head = shape.LayerHeadDim(owner);
        TEST_REQUIRE(row.size() == static_cast<std::size_t>(shape.kv_heads * head));
        // One read per KV head, against the OWNER's cache.
        TEST_REQUIRE(state.tensor_reads.size() - before ==
                     static_cast<std::size_t>(shape.kv_heads));
        const auto index =
            static_cast<std::size_t>(12 + shape.layers + 2 * owner + (value_cache ? 1 : 0));
        for (std::size_t i = before; i < state.tensor_reads.size(); ++i) {
            TEST_REQUIRE(state.tensor_reads[i].tensor == buffers[index]->object);
            TEST_REQUIRE(state.tensor_reads[i].count == static_cast<std::size_t>(head));
            TEST_REQUIRE(state.tensor_reads[i].accepted);
        }
        return owner;
    };

    // Layer 4 OWNS its cache and is full-attention: owner 4, head 512.
    TEST_REQUIRE(require_read_of(4, /*value_cache=*/false) == 4);
    TEST_REQUIRE(shape.LayerHeadDim(4) == shape.global_head_dim);
    // Layer 20 SHARES, and it is sliding: the last own-cache sliding layer is
    // 13, head 256. A cache indexed by the layer would be out of bounds.
    TEST_REQUIRE(require_read_of(20, /*value_cache=*/true) == 13);
    TEST_REQUIRE(shape.LayerHeadDim(13) == shape.head_dim);
    // Layer 34 SHARES and is full-attention: owner 14, head 512. The two
    // sharing answers differ, which is what says the kind is being honoured
    // and not just "the last owning layer".
    TEST_REQUIRE(require_read_of(34, /*value_cache=*/false) == 14);

    (void)RequireThrows([&] { (void)harness.engine->get_k_cache(shape.layers, 0); });
    (void)RequireThrows([&] { (void)harness.engine->get_k_cache(0, -1); });
}

/// \brief the host-side rejections fire BEFORE ANY DISPATCH
///
/// THIS IS THE DIFFERENCE BETWEEN AN EXCEPTION AND A CRASH, and it is a
/// property of the dispatch model rather than of any one operator. A rejection
/// part-way through an unbarriered chain unwinds out of the pass without
/// reaching `synchronize()`, so the runs already submitted are destructed
/// while their commands are still in flight and XRT ABORTS THE PROCESS --
/// "xrt::run destructed while command is still in progress" -- with no
/// catchable failure anywhere. There is no try/finally that fixes it either:
/// synchronizing during the unwind waits on work whose operands are already
/// being torn down.
///
/// The case it was first met on is chunked prefill, which `flat_mha_bf16`
/// refuses as the SIXTH dispatch of a layer. So the assertion is not merely
/// "it threw": it is that the dispatch list is EMPTY when it did.
void TestTheHostRejectionsFireBeforeAnyDispatch() {
    Harness harness(E2bFixture());

    // NO `fake_corelib::Reset()` ANYWHERE IN THIS TEST. Reset clears
    // `pad_answers_seen`, which the fake filled from this engine's own
    // construction-time pad-helper calls -- an engine that outlives a Reset
    // has every later dispatch refused by `OnGrid`, which would make each
    // assertion below pass for entirely the wrong reason. Deltas, not
    // emptiness.
    TEST_REQUIRE(fake_corelib::GetState().dispatches.empty());

    // A multi-row continuation needs the tokens behind it; set_context_length
    // moved the position without supplying them.
    harness.engine->set_context_length(5);
    std::vector<int> ids{2, 5, 9};
    const auto chunked = RequireThrows([&] { (void)harness.engine->prefill(ids); });
    RequireContains(chunked, "position");
    TEST_REQUIRE(fake_corelib::GetState().dispatches.empty());

    // A REQUEST PAST THE CONFIGURED CONTEXT. Note which of the two capacity
    // rules fires: `position + rows > max_length` is reached first, and with
    // `max_length == kMaxSequenceLength` the cache-extent rule below it
    // (`position + padded > 4096`) is UNREACHABLE -- see Run's comment. It is
    // kept because it guards the CACHE rather than the configuration, and the
    // two are independent facts, but no test here can reach it and this
    // comment is the disclosure rather than a pretence otherwise.
    harness.engine->set_context_length(flm::gemma4::kMaxSequenceLength);
    const auto before_past_end = fake_corelib::GetState().dispatches.size();
    const auto past_end = RequireThrows([&] { (void)harness.engine->forward(3); });
    RequireContains(past_end, "capacity");
    TEST_REQUIRE(fake_corelib::GetState().dispatches.size() == before_past_end);

    // A TOKEN OUTSIDE THE VOCABULARY. Caught on the host, before the gather,
    // so it is one exception rather than a read off the end of a 5 GB mapping.
    harness.engine->clear_context();
    const auto before_bad_token = fake_corelib::GetState().dispatches.size();
    const auto bad_token =
        RequireThrows([&] { (void)harness.engine->forward(flm::gemma4::kVocabularySize); });
    RequireContains(bad_token, "262144");
    TEST_REQUIRE(fake_corelib::GetState().dispatches.size() == before_bad_token);

    // AND AN EMPTY REQUEST, which would otherwise read row -1 at the end.
    const auto before_empty = fake_corelib::GetState().dispatches.size();
    std::vector<int> nothing;
    (void)RequireThrows([&] { (void)harness.engine->prefill(nothing); });
    TEST_REQUIRE(fake_corelib::GetState().dispatches.size() == before_empty);
}

/// \brief a multi-row continuation re-runs the conversation from position 0
/// \note What a chat's second turn does: FLM prefills only the new messages at
///       the cached position, and attention ships no chunked prefill.
void TestAMultiRowContinuationReplaysFromPositionZero() {
    Harness harness(E2bFixture());
    std::vector<int> first{2, 5, 9};
    (void)harness.engine->prefill(first);
    (void)harness.engine->forward(17);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 4);

    const auto before = fake_corelib::GetState().dispatches.size();
    std::vector<int> more{11, 13};
    (void)harness.engine->prefill(more);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 6);
    TEST_REQUIRE(!harness.engine->poisoned());

    const auto& dispatches = fake_corelib::GetState().dispatches;
    std::size_t attention = 0;
    for (std::size_t i = before; i < dispatches.size(); ++i) {
        if (dispatches[i].kind != "mha") continue;
        ++attention;
        TEST_REQUIRE(dispatches[i].mha_position == 0);
    }
    TEST_REQUIRE(attention == static_cast<std::size_t>(harness.shape.layers));

    // restore() rewinds the history with the position, so the next
    // continuation replays only what is still behind it.
    (void)harness.engine->checkpoint();
    (void)harness.engine->forward(19);
    TEST_REQUIRE(harness.engine->restore() == 6);
    std::vector<int> again{23, 29};
    (void)harness.engine->prefill(again);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 8);
}

/// \brief (position, rows) of every paged_mha dispatch from index `from` on
std::vector<std::pair<std::int64_t, std::int64_t>> PagedAttentionSince(std::size_t from) {
    const auto& dispatches = fake_corelib::GetState().dispatches;
    std::vector<std::pair<std::int64_t, std::int64_t>> passes;
    for (std::size_t i = from; i < dispatches.size(); ++i) {
        if (dispatches[i].kind == "paged_mha")
            passes.emplace_back(dispatches[i].mha_position, dispatches[i].rows);
    }
    return passes;
}

/// \brief a paged continuation on a page boundary runs only its own rows
/// \note The server sends a long prompt `max_prefill_len` (4096) tokens at a
///       time, so every call after the first lands on a page boundary.
///       Replaying from position 0 there made each call cost the whole
///       prompt so far.
void TestAPagedContinuationOnAPageBoundaryRunsOnlyItsOwnPage() {
    Harness harness(E2bFixture(), {}, 8192);
    const auto layers = static_cast<std::size_t>(harness.shape.layers);
    std::vector<int> first(4096, 5);
    (void)harness.engine->prefill(first);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 4096);
    const auto page0 = PagedAttentionSince(0);
    TEST_REQUIRE(page0.size() == layers);
    for (const auto& [position, rows] : page0) TEST_REQUIRE(position == 0 && rows == 4096);

    const auto before = fake_corelib::GetState().dispatches.size();
    std::vector<int> second(4096, 7);
    (void)harness.engine->prefill(second);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 8192);
    TEST_REQUIRE(!harness.engine->poisoned());
    const auto page1 = PagedAttentionSince(before);
    TEST_REQUIRE(page1.size() == layers);
    for (const auto& [position, rows] : page1) TEST_REQUIRE(position == 4096 && rows == 4096);
}

/// \brief off a page boundary a paged continuation re-runs its page, and
///        inside page 0 that is the whole conversation
void TestAPagedContinuationOffAPageBoundaryReplaysFromItsPage() {
    Harness harness(E2bFixture(), {}, 8192);
    const auto layers = static_cast<std::size_t>(harness.shape.layers);

    std::vector<int> short_prompt{2, 5, 9};
    (void)harness.engine->prefill(short_prompt);
    (void)harness.engine->forward(17);
    auto before = fake_corelib::GetState().dispatches.size();
    std::vector<int> more{11, 13};
    (void)harness.engine->prefill(more);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 6);
    const auto in_page0 = PagedAttentionSince(before);
    TEST_REQUIRE(in_page0.size() == layers);
    for (const auto& [position, rows] : in_page0) TEST_REQUIRE(position == 0 && rows == 128);

    harness.engine->clear_context();
    std::vector<int> long_prompt(4100, 5);
    (void)harness.engine->prefill(long_prompt);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 4100);
    before = fake_corelib::GetState().dispatches.size();
    (void)harness.engine->prefill(more);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 4102);
    TEST_REQUIRE(!harness.engine->poisoned());
    const auto in_page1 = PagedAttentionSince(before);
    TEST_REQUIRE(in_page1.size() == layers);
    for (const auto& [position, rows] : in_page1) TEST_REQUIRE(position == 4096 && rows == 4096);

    // And the rows it re-ran are the engine's own: one more continuation
    // still finds every token behind it.
    std::vector<int> last{19, 23, 29};
    (void)harness.engine->prefill(last);
    TEST_REQUIRE(harness.engine->get_current_context_length() == 4105);
}

// ---------------------------------------------------------------------------
// The weight cache.

/// \brief a scratch cache directory, removed on the way out
struct ScopedCacheDirectory {
    std::filesystem::path path;
    ScopedCacheDirectory() {
        path = std::filesystem::temp_directory_path() /
               ("flm_gemma4_cache_" +
                std::to_string(gemma4_fixture::detail::CurrentProcessId()) + "_" +
                std::to_string(gemma4_fixture::detail::NextSerial()));
        std::error_code ignored;
        std::filesystem::create_directories(path, ignored);
    }
    ~ScopedCacheDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

void RequireSecondLoadHitsTheCache(const Fixture& fixture) {
    ScopedCacheDirectory cache;
    ScopedWeightCache enabled(cache.path.string());

    std::size_t expected_slots = 0;
    {
        RuntimeScope teardown;
        fake_corelib::Reset();
        auto runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        auto package = Gemma4GgufPackage::Open(fixture.path);
        auto engine = std::make_unique<gemma4_rai>(package, runtime, LM_Config{});
        TEST_REQUIRE(!engine->loaded_from_cache());
        expected_slots = engine->weight_slot_count();
        // The first load PACKED: every ple weight went through the packer.
        TEST_REQUIRE(!fake_corelib::GetState().ple_pack_calls.empty());
    }

    {
        RuntimeScope teardown;
        fake_corelib::Reset();
        auto runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        auto package = Gemma4GgufPackage::Open(fixture.path);
        auto engine = std::make_unique<gemma4_rai>(package, runtime, LM_Config{});
        TEST_REQUIRE(engine->loaded_from_cache());
        TEST_REQUIRE(engine->weight_slot_count() == expected_slots);

        const auto& state = fake_corelib::GetState();
        // NOTHING WAS REQUANTIZED. Every weight, ple included, was created
        // from its packed slice of the file, so no packer ran.
        TEST_REQUIRE(state.ple_pack_calls.empty());
        TEST_REQUIRE(state.ple_create_calls.size() ==
                     static_cast<std::size_t>(package->Config().layers));
        TEST_REQUIRE(!state.matmul_weights_creates.empty());
        for (const auto& call : state.matmul_weights_creates) TEST_REQUIRE(call.from_file);
        for (const auto& call : state.ssmlp_weights_creates) TEST_REQUIRE(call.from_file);
        for (const auto& call : state.rmsnorm_weights_creates) TEST_REQUIRE(call.from_file);
        // Every bind is one slot: a cache whose entry count disagreed would
        // have been refused wholesale rather than bound half-way.
        TEST_REQUIRE(state.matmul_weights_creates.size() +
                         state.ssmlp_weights_creates.size() +
                         state.rmsnorm_weights_creates.size() +
                         state.ple_create_calls.size() ==
                     expected_slots);

    }
}

void TestE2bSecondLoadOfTheSameGgufBindsFromTheCache() {
    RequireSecondLoadHitsTheCache(E2bFixture());
}
void TestE4bSecondLoadOfTheSameGgufBindsFromTheCache() {
    RequireSecondLoadHitsTheCache(E4bFixture());
}

/// \brief a cache written for one model is never bound to the other
/// \note Both share one redirected cache directory; each cache is named after
///       its own GGUF.
void TestACacheWrittenForE2bIsNotBoundForE4b() {
    ScopedCacheDirectory cache;
    ScopedWeightCache enabled(cache.path.string());
    const auto load = [&](const Fixture& fixture) {
        RuntimeScope teardown;
        fake_corelib::Reset();
        auto runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        auto package = Gemma4GgufPackage::Open(fixture.path);
        auto engine = std::make_unique<gemma4_rai>(package, runtime, LM_Config{});
        return engine->loaded_from_cache();
    };
    TEST_REQUIRE(!load(E2bFixture()));   // cold
    TEST_REQUIRE(!load(E4bFixture()));   // E2B's cache must not answer for E4B
    TEST_REQUIRE(load(E4bFixture()));    // and E4B's own does
}

/// \brief a CACHE-BOUND engine puts every weight in the slot named for it
///
/// THE SECOND WAY A WEIGHT REACHES THE WRONG MEMBER, and until now the
/// untested one. The pack path assigns through each slot's own `destination`
/// pointer; the cache path calls `..._weights_create_from_file` and assigns
/// by walking `slots` IN INDEX ORDER, so the slot table's ordering is
/// load-bearing there and nowhere else. Nothing observed it: R2's placement
/// verifiers joined on `blocks_tag` and `scale`, and corelib gives back
/// neither when it is handed a slice of an opaque blob.
///
/// It is not enough that a mis-slotted bind would be refused on size. One
/// layer's Q-norm and K-norm have IDENTICAL descriptors -- same `k`, same
/// epsilon -- so they pack to identical lengths and exchanging their handles
/// binds cleanly and silently. Same for `ffn_gate`/`ffn_up`'s enclosing
/// ssmlp against another layer's of equal width.
///
/// E2B ONLY. The index walk under test is one loop with no row-dependent
/// branch, and E4B would cost a second full pack-and-bind of an 8.6 GB
/// fixture to exercise the same lines. The pack-path placement test runs on
/// both rows, so the per-row slot INVENTORY is still checked twice.
void TestE2bCacheBoundWeightsLandInTheSlotNamedForThem() {
    ScopedCacheDirectory cache;
    ScopedWeightCache enabled(cache.path.string());
    const auto& fixture = E2bFixture();

    {
        RuntimeScope teardown;
        fake_corelib::Reset();
        auto runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        auto package = Gemma4GgufPackage::Open(fixture.path);
        auto engine = std::make_unique<gemma4_rai>(package, runtime, LM_Config{});
        TEST_REQUIRE(!engine->loaded_from_cache());
    }
    {
        RuntimeScope teardown;
        fake_corelib::Reset();
        auto runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        auto package = Gemma4GgufPackage::Open(fixture.path);
        auto engine = std::make_unique<gemma4_rai>(package, runtime, LM_Config{});
        // Without this the whole test would silently degrade into a second
        // run of the pack-path one.
        TEST_REQUIRE(engine->loaded_from_cache());
        RequireWeightPlacementOf(*engine, package->Config(), fixture.builder);
    }
}

/// \brief an index written under packed-layout revision N is STALE under N+1
///
/// THE KEY HAS TO BE ABLE TO SAY "THE BYTES MEAN SOMETHING DIFFERENT NOW".
/// Task R2 changed which orientation of `inp_gate` and `proj` reaches
/// `ryzenai_corelib_ple_bf16_weights_pack` and moved NO field of the cache
/// key: model_id, GGUF size and mtime, corelib version, group sizes and slot
/// count were all identical before and after. A cache written before that fix
/// would have bound after it and served the untransposed blob on a load that
/// looks, from the outside, exactly like a hit -- which is this project's own
/// B2 lesson ("a cache that trusts its filename is how a stale blob is read
/// back as current") one level up.
///
/// TWO HALVES, AND THE FIRST IS THE ONE THAT PINS THE ENGINE. The first
/// assertion recomputes the key from `kPackedLayoutRevision` and requires the
/// index the engine actually wrote to read back under it: that is what says
/// the revision is IN the engine's key rather than merely defined next to it.
/// The second bumps the revision and requires a miss. Without the first, an
/// engine that ignored the revision entirely would still pass the second.
void TestACacheWrittenUnderOnePackedLayoutIsStaleUnderTheNext() {
    ScopedCacheDirectory cache;
    ScopedWeightCache enabled(cache.path.string());
    const auto& fixture = E2bFixture();

    Gemma4Config shape;
    std::uint64_t slots = 0;
    flm::corelib::CorelibVersion version{};
    {
        RuntimeScope teardown;
        fake_corelib::Reset();
        auto runtime = CorelibRuntime::CreateForTest(
            CorelibApi::ResolveForTest(fake_corelib::Resolver()));
        auto package = Gemma4GgufPackage::Open(fixture.path);
        auto engine = std::make_unique<gemma4_rai>(package, runtime, LM_Config{});
        TEST_REQUIRE(!engine->loaded_from_cache());
        shape = package->Config();
        slots = engine->weight_slot_count();
        version = runtime->api()->runtime_version();
    }

    // Rebuilt by hand from the config rather than read off the engine, so it
    // can disagree with the engine's own key.
    const auto key_at = [&](std::uint32_t revision) {
        const std::string layout =
            "gemma4 E2B group=" + std::to_string(shape.group) +
            " head_group=" + std::to_string(shape.head_group) +
            " ple_group=" + std::to_string(shape.ple_group) +
            " layout_revision=" + std::to_string(revision);
        return flm::rai::MakeWeightCacheKey(fixture.path, version.major, version.minor,
                                            version.patch, layout, slots);
    };
    const auto weight_cache = flm::rai::WeightCache::ForGguf(fixture.path);
    TEST_REQUIRE(weight_cache.has_value());
    TEST_REQUIRE(weight_cache->ReadIndex(key_at(flm::gemma4::kPackedLayoutRevision)).has_value());
    TEST_REQUIRE(!weight_cache->ReadIndex(key_at(flm::gemma4::kPackedLayoutRevision + 1)).has_value());
}

// ---------------------------------------------------------------------------
// Teardown order, and `poisoned()` (Task C9).
//
// *** NOTHING BELOW ASSERTS AN EXIT CODE, AND THAT IS THE POINT. ***
//
// The defect this section defends against was measured in corelib's own
// Python driver: with two models' fixtures in one process and the first torn
// down `stream.release()` FIRST -- its packed weights then left to the
// collector -- `Windows fatal exception: access violation` printed while the
// second model's tests ran, every one of them inside `Object.__del__ ->
// release`. Across 20 runs on box 43 the "Stream released first" order
// printed violations 7 of 10 times, up to 221 lines in a single run; the
// other order printed none in 10.
//
// THE EXIT CODES, AS MEASURED RATHER THAN AS PREVIOUSLY SUMMARIZED. This
// comment used to say "AND THE PROCESS EXITED 0 EVERY TIME". What
// the corelib model notes actually record is that NINETEEN OF
// THE TWENTY sessions -- both orders together -- exited 0 with every test
// PASSED, and THE TWENTIETH EXITED 1 with no violations at all. It does not
// break exit codes down by order, so what the stream-first ten did is not
// established. What is established about "nothing raised" is only that a
// recorder in `Object.__del__` caught ZERO of the 833 printed violations;
// the source says in terms not to re-derive a mechanism from that zero.
//
// So a behavioural test cannot catch it, and 19/20 is already enough to say
// so. "Tear two engines down and check the process survived" asserts NOTHING
// here: a bad ordering passes it 3 runs in 10, passes it 19 times in 20 on
// exit code alone, and passes it every time under a fake that has no device
// to violate. The
// tests below are structural instead -- they read the ORDER, from the
// engine's own source and from the ABI call sequence the fake records -- and
// that is the strongest statement available without hardware.
//
// WHAT THEY DO NOT SAY: no Gemma 4 teardown in this project has ever run
// against real corelib. The honest status of this defence is "ordered
// correctly by construction and pinned structurally", not "verified".

/// \brief every code line of `struct gemma4_rai::Impl`'s member block, in
///        declaration order, comments and blank lines removed
///
/// READS THE ENGINE'S SOURCE, exactly as the reference driver's own gate
/// does (`test_every_gemma4_model_fixture_tears_down_through_release` reads
/// its fixtures' source). DESTRUCTION ORDER IS REVERSE DECLARATION ORDER, so
/// the declarations ARE the teardown order, and they are the only form of it
/// a test can read: there is no destructor body to inspect, deliberately --
/// see gemma4_rai.cpp.
///
/// \note FAILS, NEVER SKIPS, if the source cannot be opened. A structural
///       gate that quietly passes when it cannot find its subject is worse
///       than no gate: it reports green over nothing. `FLM_GEMMA4_RAI_SOURCE`
///       is set by this target's CMakeLists.
std::vector<std::string> ImplMemberDeclarations() {
    std::ifstream source(FLM_GEMMA4_RAI_SOURCE);
    TEST_REQUIRE(source.is_open());
    std::vector<std::string> lines;
    std::string line;
    bool inside = false;
    bool reached_constructor = false;
    while (std::getline(source, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!inside) {
            if (line.find("struct gemma4_rai::Impl {") != std::string::npos) inside = true;
            continue;
        }
        // The constructor ends the member block. Everything after it is
        // function bodies, whose locals are not members and whose order says
        // nothing about destruction.
        if (line.rfind("    Impl(", 0) == 0) {
            reached_constructor = true;
            break;
        }
        const auto comment = line.find("//");
        auto code = line.substr(0, comment == std::string::npos ? line.size() : comment);
        if (code.find_first_not_of(" \t") == std::string::npos) continue;
        lines.push_back(std::move(code));
    }
    TEST_REQUIRE(inside);
    TEST_REQUIRE(reached_constructor);
    return lines;
}

/// \brief the index of the first declaration containing `needle`, or size()
std::size_t DeclarationIndex(const std::vector<std::string>& lines,
                             std::string_view needle) {
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].find(needle) != std::string::npos) return i;
    }
    return lines.size();
}

/// \brief every declaration index whose type owns a corelib handle
///
/// *** THIS LIST IS A DENYLIST OF KNOWN SPELLINGS. IT IS NOT A PROPERTY OF
/// THE TYPE. *** A test that reads source text cannot ask C++ what a type
/// owns; it can only recognize names somebody wrote down here. So this
/// function answers "which declarations are owning ones I have heard of",
/// which is a LOWER BOUND, and a corelib-owning member spelled any other way
/// is invisible to it. Review-C9's C3 demonstrated exactly that: the seven
/// entries below are all aliases for `UniqueObject<Tag>`, and
/// `flm::corelib::UniqueObject<flm::corelib::TensorTag> scratch;` -- the same
/// type, written out -- was declared above `stream` and the suite stayed
/// 57/57 green. `"UniqueObject"` is now in the list, which closes that
/// family; the NEXT spelling nobody thought of is still open, and no amount
/// of adding names changes that.
///
/// WHICH IS WHY THE GATE BELOW DOES NOT REST ON THIS LIST. `kAllowedAbove
/// TheStream` is the half that is closed under new spellings: it enumerates
/// what MAY precede `stream` and refuses everything else, so an unknown type
/// fails loudly instead of passing silently. Read the two together -- this
/// one states what is definitely owning, that one states what is definitely
/// harmless, and only the second has no blind spot.
///
/// \note `LayerWeights` is here because `std::vector<LayerWeights> layers` is
///       nine packed weights per row and the nested struct's own members are
///       declared inside the same block. Matching on the TYPE rather than on
///       a member name is what makes this survive a rename and catch an
///       ADDITION -- which is the edit the note in gemma4_rai.cpp warns
///       about and the one a reviewer would not otherwise see.
std::vector<std::size_t> CorelibOwningDeclarations(const std::vector<std::string>& lines) {
    static constexpr std::string_view kTypes[] = {
        "UniqueObject",         "UniqueStream",         "UniqueTensorWindow",
        "UniqueTensor",         "UniqueMatMulWeights",  "UniqueSsMlpWeights",
        "UniqueRmsNormWeights", "UniquePleWeights",     "LayerWeights",
    };
    std::vector<std::size_t> found;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        for (const auto type : kTypes) {
            if (lines[i].find(type) != std::string::npos) {
                found.push_back(i);
                break;
            }
        }
    }
    return found;
}

/// \brief a declaration with its leading indentation removed
std::string TrimmedDeclaration(const std::string& line) {
    const auto first = line.find_first_not_of(" \t");
    return first == std::string::npos ? std::string{} : line.substr(first);
}

/// \brief every type spelling permitted ABOVE `UniqueStream stream;`
///
/// *** DEFAULT-DENY, WHICH IS THE WHOLE POINT. *** Everything else in this
/// gate asks "is this declaration one of the dangerous kinds I know about" and
/// is therefore blind to a kind nobody listed. This asks the opposite
/// question -- "is this declaration one of the few I have already established
/// are safe here" -- and a member of ANY type nobody listed fails it. The
/// failure mode flips from silently blind to loudly conservative, and the
/// conservative direction is the correct one for a rule whose violation is an
/// intermittent access violation that raises nothing.
///
/// The region this governs is nine declarations long and has not changed
/// since the engine was written: the three shared_ptrs a corelib object may
/// depend on, and six POD members that hold no resource at all.
///
/// IF THIS TEST FAILS BECAUSE YOU ADDED A MEMBER, the fix is almost always to
/// move the member BELOW `stream`, not to add a line here. Add a line only
/// for something that genuinely must outlive the stream, and write down why
/// -- the entries below are load-bearing in that they are the only members
/// whose destruction after every corelib object has been reasoned about.
///
/// \note MATCHED ON THE TYPE PREFIX, not the whole line, so renaming a member
///       is free and changing its type is not. Still textual: this file
///       cannot see types, only the characters that spell them. What makes it
///       sound is not that the match is clever but that the DEFAULT is
///       refusal.
constexpr std::string_view kAllowedAboveTheStream[] = {
    "std::shared_ptr<CorelibRuntime> ",
    "std::shared_ptr<Gemma4GgufPackage> ",
    "std::shared_ptr<const CorelibApi> ",
    "Gemma4Config ",
    "std::uint32_t ",
    "int ",
    "std::optional<int> ",
    "bool ",
    "std::size_t ",
};

/// THE STRUCTURAL GATE. It fails if a future edit declares a corelib-owning
/// member above `stream` (which would release it AFTER the stream -- the
/// measured fault), above `package` (whose mapping a packed weight may still
/// be bound to: corelib.h says a create "retains, imports or copies" and the
/// adapter does not bind `weights_is_owned`, so the honest assumption is that
/// the pages are bound), or above `runtime`.
///
/// IT HAS TWO HALVES AND THEY FAIL DIFFERENTLY. The `kTypes` half names
/// owning types and is a denylist of spellings -- see
/// `CorelibOwningDeclarations`, which says so at length. The
/// `kAllowedAboveTheStream` half enumerates what may precede `stream` and
/// refuses everything else, so it is closed under new spellings and is what
/// actually stands behind the rule. Review-C9's C3 -- `UniqueObject<TensorTag>`
/// above `stream`, 57/57 green -- is now caught by both, for different
/// reasons; a type nobody has thought of yet is caught only by the second.
///
/// WHAT IT CANNOT CATCH, stated because the whole value of this test is that
/// it is read rather than trusted:
///   - a non-corelib member that must nonetheless outlive the weights, such
///     as a host buffer corelib binds in place. The engine holds none today:
///     every create copies its sources or maps the cache file itself. One
///     added later would have to be pinned here by hand, because the
///     default-deny half says nothing about the region below `stream`.
///   - anything about whether corelib actually faults. The fake's
///     `object_release` is a `delete`. See this section's header.
///   - an ordering error inside a nested type held by value elsewhere.
void TestEveryCorelibOwningMemberIsDeclaredAfterTheStream() {
    const auto lines = ImplMemberDeclarations();
    // A parse that matched nothing would pass every assertion below by
    // vacuity, which is the failure mode of every source-reading test.
    TEST_REQUIRE(lines.size() > 20);
    const auto owning = CorelibOwningDeclarations(lines);
    TEST_REQUIRE(owning.size() >= 10);

    // THE RUNTIME IS DECLARED FIRST, so it is destroyed LAST -- after every
    // corelib object and after the API they were created through.
    TEST_REQUIRE(lines.front().find("std::shared_ptr<CorelibRuntime> runtime;") !=
                 std::string::npos);

    const auto stream = DeclarationIndex(lines, "UniqueStream stream;");
    const auto api = DeclarationIndex(lines, "std::shared_ptr<const CorelibApi> api;");
    const auto package =
        DeclarationIndex(lines, "std::shared_ptr<Gemma4GgufPackage> package;");
    TEST_REQUIRE(stream < lines.size());
    TEST_REQUIRE(api < lines.size());
    TEST_REQUIRE(package < lines.size());
    // Exactly one stream, or "the first UniqueStream" is not the stream.
    TEST_REQUIRE(std::count_if(lines.begin(), lines.end(), [](const std::string& l) {
                     return l.find("UniqueStream") != std::string::npos;
                 }) == 1);

    // THE RULE, in the form corelib's measurement stated it: every packed
    // object is released BEFORE the Stream, so every one of them is declared
    // AFTER it. Stated over the DENYLIST, so it is a lower bound.
    TEST_REQUIRE(owning.front() == stream);
    for (const auto index : owning) TEST_REQUIRE(index >= stream);

    // THE SAME RULE STATED THE OTHER WAY ROUND, AND THIS IS THE HALF WITH NO
    // BLIND SPOT: nothing above `stream` may be of a type that is not already
    // on the short list of things established to be safe there. A member of
    // any other type -- corelib-owning or not, spelled any way at all --
    // fails here, which is the property `kTypes` cannot have.
    for (std::size_t i = 0; i < stream; ++i) {
        const auto declaration = TrimmedDeclaration(lines[i]);
        bool allowed = false;
        for (const auto type : kAllowedAboveTheStream) {
            if (declaration.rfind(type, 0) == 0) {
                allowed = true;
                break;
            }
        }
        if (!allowed) {
            throw std::runtime_error(
                "a member declared above `stream` has a type this gate has not "
                "established is safe there: '" + declaration +
                "'. Move it below `stream`, or add its type to "
                "kAllowedAboveTheStream with a reason it must outlive every "
                "corelib object.");
        }
    }

    // The API and the GGUF mapping outlive everything created through them.
    // The API one is belt-and-braces -- `UniqueObject` holds its own
    // `shared_ptr` to the API, so a handle cannot outlive it today -- and is
    // asserted anyway because a RAW corelib handle added as a member would
    // depend on it and nothing else would say so.
    TEST_REQUIRE(api < owning.front());
    TEST_REQUIRE(package < owning.front());
}

/// \brief the release kinds the fake recorded after `mark`
std::vector<std::string> ReleasesSince(std::size_t mark) {
    const auto& order = fake_corelib::GetState().release_order;
    TEST_REQUIRE(mark <= order.size());
    return {order.begin() + static_cast<std::ptrdiff_t>(mark), order.end()};
}

bool IsPackedWeight(const std::string& kind) {
    return kind == "matmul_weights" || kind == "ssmlp_weights" ||
           kind == "rmsnorm_weights" || kind == "ple_weights";
}

/// \brief one engine's teardown, read as an ABI call sequence
/// \note `expected_weights` is asserted EXACTLY, not as "> 0". A slice
///       holding one weight and the stream in the right order would satisfy
///       the ordering rule while saying nothing about the other 271.
void RequireTeardownSliceIsOrdered(const std::vector<std::string>& slice,
                                   std::size_t expected_weights,
                                   std::size_t expected_tensors) {
    TEST_REQUIRE(!slice.empty());
    // THE WHOLE RULE, IN ONE LINE: the Stream goes last.
    TEST_REQUIRE(slice.back() == "stream");
    TEST_REQUIRE(std::count(slice.begin(), slice.end(), "stream") == 1);
    TEST_REQUIRE(static_cast<std::size_t>(std::count_if(
                     slice.begin(), slice.end(), IsPackedWeight)) == expected_weights);
    TEST_REQUIRE(static_cast<std::size_t>(std::count(slice.begin(), slice.end(),
                                                     "tensor")) == expected_tensors);
}

void RequireTeardownReleasesEverythingBeforeTheStream(const Fixture& fixture) {
    Harness harness(fixture);
    // RUN A PASS FIRST. The measured fault happened to a model that had been
    // USED, and a pass is also what mints the per-dispatch windows -- which
    // are released during the pass, so the teardown slice below is not
    // diluted by them.
    std::vector<int> ids{1, 2, 3};
    (void)harness.engine->prefill(ids);
    const auto tensors = fake_corelib::GetState().tensor_creates.size();
    TEST_REQUIRE(tensors > 0);

    const auto mark = fake_corelib::GetState().release_order.size();
    harness.engine.reset();
    RequireTeardownSliceIsOrdered(ReleasesSince(mark), ExpectedSlots(harness.shape),
                                  tensors);
    // Nothing held back. `ShutdownProcess` would also throw on a non-zero
    // count, but Harness swallows that, so it is asserted here where it is
    // visible.
    TEST_REQUIRE(harness.runtime->api()->live_object_count() == 0);
}

void TestE2bTeardownReleasesEverythingBeforeTheStream() {
    RequireTeardownReleasesEverythingBeforeTheStream(E2bFixture());
}
void TestE4bTeardownReleasesEverythingBeforeTheStream() {
    RequireTeardownReleasesEverythingBeforeTheStream(E4bFixture());
}

/// THE RUNTIME GOES LAST OF ALL, and it is read off the SAME ABI sequence as
/// the releases. `ryzenai_corelib_cleanup` is the runtime's own teardown
/// call -- `CorelibRuntime::ShutdownProcess` makes it and then drops the
/// runtime -- so its position in the log is the runtime's position.
///
/// \note Whole-log, not a slice: the Harness resets the fake at construction,
///       so this log is exactly one engine's life from first create to
///       process-runtime shutdown.
void TestTheRuntimeIsCleanedUpAfterEveryCorelibObject() {
    {
        Harness harness(E2bFixture());
        std::vector<int> ids{1, 2, 3};
        (void)harness.engine->prefill(ids);
    }
    const auto& order = fake_corelib::GetState().release_order;
    TEST_REQUIRE(order.size() >= 2);
    TEST_REQUIRE(std::count(order.begin(), order.end(), "cleanup") == 1);
    TEST_REQUIRE(order.back() == "cleanup");
    // And the stream is the last corelib object before it -- i.e. the whole
    // sequence is "every object, then the stream, then the runtime".
    TEST_REQUIRE(order[order.size() - 2] == "stream");
}

/// TWO MODELS IN ONE PROCESS, WHICH IS THE CASE THAT FAULTED. The reference
/// driver met this only because a second model entered one process; FLM loads
/// and unloads models as a matter of course, so it is a shipping path here.
/// The survivor is driven AFTER the first is torn down, because that is
/// exactly when the driver printed its violations -- "while the second
/// model's tests ran".
///
/// \note Both orders, because the two models are not interchangeable: E2B has
///       35 layers, 15 owning, one KV head; E4B has 42, 24 owning, two. A
///       teardown bug that depends on which shape went first would otherwise
///       be a coin flip.
/// \note ONE runtime, not two. `CreateForTest` refuses a second while one
///       lives, and one shared runtime is also what FLM does.
void RequireTwoModelsTearDownInOneProcess(bool e2b_first) {
    ScopedWeightCache no_cache{"0"};
    RuntimeScope teardown;
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(
        CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    const auto api = runtime->api();
    auto e2b_package = Gemma4GgufPackage::Open(E2bFixture().path);
    auto e4b_package = Gemma4GgufPackage::Open(E4bFixture().path);
    auto e2b = std::make_unique<gemma4_rai>(e2b_package, runtime, LM_Config{});
    auto e4b = std::make_unique<gemma4_rai>(e4b_package, runtime, LM_Config{});

    auto& first = e2b_first ? e2b : e4b;
    auto& second = e2b_first ? e4b : e2b;
    const auto first_slots =
        ExpectedSlots((e2b_first ? e2b_package : e4b_package)->Config());
    const auto second_shape = (e2b_first ? e4b_package : e2b_package)->Config();

    std::vector<int> ids{1, 2, 3};
    (void)first->prefill(ids);
    (void)second->prefill(ids);

    // Tensors are created only at load here, so the two engines' totals
    // partition the fake's create log. Take the survivor's from what is left
    // after the first has gone rather than trying to attribute creates.
    const auto all_tensors = fake_corelib::GetState().tensor_creates.size();

    const auto first_mark = fake_corelib::GetState().release_order.size();
    first.reset();
    const auto first_slice = ReleasesSince(first_mark);
    const auto first_tensors =
        static_cast<std::size_t>(std::count(first_slice.begin(), first_slice.end(), "tensor"));
    RequireTeardownSliceIsOrdered(first_slice, first_slots, first_tensors);
    TEST_REQUIRE(first_tensors > 0);

    // THE SURVIVOR IS STILL DRIVEN, which is the driver's exact scenario.
    (void)second->forward(7);
    TEST_REQUIRE(second->get_current_context_length() == 4);

    const auto second_mark = fake_corelib::GetState().release_order.size();
    second.reset();
    const auto second_slice = ReleasesSince(second_mark);
    RequireTeardownSliceIsOrdered(second_slice, ExpectedSlots(second_shape),
                                  all_tensors - first_tensors);

    TEST_REQUIRE(api->live_object_count() == 0);
}

void TestTwoModelsTearDownInOneProcessE2bFirst() {
    RequireTwoModelsTearDownInOneProcess(/*e2b_first=*/true);
}
void TestTwoModelsTearDownInOneProcessE4bFirst() {
    RequireTwoModelsTearDownInOneProcess(/*e2b_first=*/false);
}

// ---------------------------------------------------------------------------
// `poisoned()`.
//
// WHAT POISONS THIS ENGINE: a forward pass that throws after one of its
// dispatches was ACCEPTED, and nothing else. Two reasons it is drawn there
// and not narrower:
//
//   - A MID-CHAIN CORELIB REJECTION IS UNRECOVERABLE BY CONSTRUCTION. The
//     chain is unbarriered, so an exception unwinds past operands whose
//     commands are still in flight. The engine cannot know how much of the
//     layer loop ran, and the KV caches hold rows for positions it will not
//     count.
//   - A NON-FINITE RESULT IS PERMANENT -- for two reasons that are NOT the
//     one an earlier version of this comment gave. It said "every later
//     token would attend over it", which is not established: `flat_mha` is
//     handed `position`, `position` advances only on success, so a failed
//     pass's NaN rows lie outside the window any later pass attends over and
//     a retry overwrites them. What does hold is (1) the cause is
//     DETERMINISTIC, so a retry reproduces it, and (2) `RequireFinite` sees
//     LESS than the corruption it detects -- the last live row's hidden and
//     the logits, never a cache row below `position` -- so the observable
//     failure understates the damage. `clear_context()` resets a POSITION;
//     it does not zero a cache.
//
// WHAT DOES NOT POISON: every rejection before a dispatch was accepted -- an
// empty request, a token outside the vocabulary, a chunked prefill, a request
// past capacity, and the refusal of the pass's very first dispatch, which
// enqueued nothing. None of those left work in flight, so the engine is
// exactly as it was, and poisoning on them would turn a caller error into a
// mandatory model reload.
//
// THE TWO HALVES ARE TESTED AS A PAIR AND THE PAIR IS THE POINT. Until C9's
// fix round only the poisoning half had a test, and replacing `if (submitted)`
// with `if (true)` in the engine's catch left the whole suite green -- the
// definition could not be told apart from its negation.
// `Require{ADispatchFailure,ARefusedFirstDispatch}...` are now the two sides
// of the same injection, at dispatch 20 and at dispatch 0, and
// `TestTheHostRejectionsAreRaisedOutsideTheTry` keeps the second one meaning
// what it says.
//
// THE FLAG IS A STATUS, NOT A RECOVERY MECHANISM. Nothing clears it. The only
// exit is unload and reload, which is what AutoModel's
// `_shared_guard_poisoned` already does with it.

void RequireADispatchFailurePoisonsTheEngine(const Fixture& fixture) {
    Harness harness(fixture);
    TEST_REQUIRE(!harness.engine->poisoned());

    // MID-CHAIN, NOT AT THE GATE. A prefill is one rmsnorm plus 11 (E2B) or
    // 12 (E4B) dispatches for each of the first two layers, which both own a
    // cache -- so the 21st accepted dispatch is inside the layer loop on
    // either model, after the caches have been written. Failing the FIRST
    // dispatch would exercise a different and much weaker claim.
    fake_corelib::GetState().fail_dispatch_at = 20;
    std::vector<int> ids{1, 2, 3};
    const auto error = RequireThrows([&] { (void)harness.engine->prefill(ids); });
    fake_corelib::GetState().fail_dispatch_at.reset();
    RequireContains(error, "ryzenai_corelib_");
    TEST_REQUIRE(harness.engine->poisoned());

    // AND IT DOES NOT CLEAR ITSELF once the injected failure is gone. This is
    // the assertion that says the flag is state rather than a re-read of the
    // fake.
    RequireContains(RequireThrows([&] { (void)harness.engine->prefill(ids); }), "poisoned");
    TEST_REQUIRE(harness.engine->poisoned());
}

void TestE2bADispatchFailurePoisonsTheEngine() {
    RequireADispatchFailurePoisonsTheEngine(E2bFixture());
}
void TestE4bADispatchFailurePoisonsTheEngine() {
    RequireADispatchFailurePoisonsTheEngine(E4bFixture());
}

/// THE OTHER SIDE OF THE SAME INJECTION, AND THE ONLY TEST IN THIS SUITE THE
/// `submitted` GUARD CHANGES THE ANSWER TO.
///
/// `RequireADispatchFailurePoisonsTheEngine` fails dispatch 20, which is
/// mid-chain: nineteen accepted commands are in flight, the engine poisons,
/// and it would poison just as happily if the guard read `if (true)`. This
/// one fails dispatch 0. Nothing has been accepted, nothing is in flight, and
/// the engine must come out usable -- so `if (submitted)` and `if (true)`
/// give OPPOSITE answers here, which is what the pair is for. Review-C9's C1
/// measured the gap: with the guard replaced by `if (true)` the suite stayed
/// 57/57 green, so the definition C9 rests on had nothing pinning it.
///
/// \note This depends on the engine setting `submitted` AFTER `Check`, not
///       before. Both spellings are defensible in the abstract; only the
///       second makes the claim in `poisoned()`'s own documentation --
///       "anything refused before a dispatch was accepted does not poison" --
///       a fact about the code rather than a description of a branch that
///       cannot be reached.
/// \note WHAT IT DOES NOT SAY: that real corelib can refuse a first dispatch
///       without having enqueued it. See fake_corelib.hpp's "cannot catch"
///       list. It says that IF a dispatch is refused with nothing in flight,
///       this engine does not force a model reload.
void RequireARefusedFirstDispatchDoesNotPoison(const Fixture& fixture) {
    Harness harness(fixture);
    TEST_REQUIRE(!harness.engine->poisoned());
    // A LOAD DISPATCHES NOTHING, so the pass's first dispatch is the fake's
    // dispatch 0. Asserted rather than assumed: if a future load ever
    // dispatched, index 0 would be one of ITS calls and this test would be
    // injecting somewhere else entirely while still looking green.
    TEST_REQUIRE(fake_corelib::GetState().dispatches.empty());

    fake_corelib::GetState().fail_dispatch_at = 0;
    std::vector<int> ids{1, 2, 3};
    const auto error = RequireThrows([&] { (void)harness.engine->prefill(ids); });
    fake_corelib::GetState().fail_dispatch_at.reset();
    // The refusal really did land on the first dispatch of the pass -- layer
    // 0's standalone input norm -- and not on some later one.
    RequireContains(error, "ryzenai_corelib_rmsnorm_bf16 layer 0 input norm");
    TEST_REQUIRE(fake_corelib::GetState().dispatches.empty());

    // THE ASSERTION THE GUARD TURNS ON.
    TEST_REQUIRE(!harness.engine->poisoned());
    TEST_REQUIRE(harness.engine->get_current_context_length() == 0);

    // AND THE ENGINE STILL RUNS. Without this the test would pass on an
    // engine that was left unpoisoned and broken, which is the failure the
    // poisoning half exists to prevent.
    TEST_REQUIRE(harness.engine->prefill(ids).size() ==
                 static_cast<std::size_t>(harness.shape.vocab));
    TEST_REQUIRE(harness.engine->get_current_context_length() == 3);
    TEST_REQUIRE(!harness.engine->poisoned());
}

void TestE2bARefusedFirstDispatchDoesNotPoison() {
    RequireARefusedFirstDispatchDoesNotPoison(E2bFixture());
}
void TestE4bARefusedFirstDispatchDoesNotPoison() {
    RequireARefusedFirstDispatchDoesNotPoison(E4bFixture());
}

/// \brief the code lines of `Run`'s body, from its signature to its `catch`
/// \note Same instrument and same limits as `ImplMemberDeclarations`: it
///       reads the engine's source and FAILS, never skips, if it cannot find
///       its subject.
std::vector<std::string> RunBodyLines() {
    std::ifstream source(FLM_GEMMA4_RAI_SOURCE);
    TEST_REQUIRE(source.is_open());
    std::vector<std::string> lines;
    std::string line;
    bool inside = false;
    bool reached_catch = false;
    while (std::getline(source, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!inside) {
            if (line.find("buffer<bf16> Run(std::span<const int> ids) {") !=
                std::string::npos)
                inside = true;
            continue;
        }
        if (line.rfind("        } catch (...) {", 0) == 0) {
            reached_catch = true;
            break;
        }
        const auto comment = line.find("//");
        auto code = line.substr(0, comment == std::string::npos ? line.size() : comment);
        if (code.find_first_not_of(" \t") == std::string::npos) continue;
        lines.push_back(std::move(code));
    }
    TEST_REQUIRE(inside);
    TEST_REQUIRE(reached_catch);
    return lines;
}

/// THE SECOND HALF OF THE DANGEROUS PAIR, AND IT IS STRUCTURAL FOR THE SAME
/// REASON THE TEARDOWN GATE IS.
///
/// `RequireARefusedFirstDispatchDoesNotPoison` pins the guard. It does not
/// pin the thing that makes the guard sufficient: that every host rejection
/// is raised OUTSIDE the `try`, so it can never reach the catch block at all.
/// Widening the `try` upwards is an ordinary-looking RAII refactor -- move
/// the lease or the staging vectors inside it and the validation follows --
/// and on its own it changes no test. Combined with a weakened guard it turns
/// every caller error into a mandatory model reload, which is precisely the
/// bug the engine says it refused to introduce. Each edit is individually
/// invisible; this test makes the first one visible.
///
/// \note IT IS A TEXTUAL READ OF A C++ FILE and nothing more. It knows where
///       the characters `try {` appear relative to other characters; it does
///       not know scope, and a `try` introduced by a macro or a helper would
///       defeat it. What earns its place is that the property it pins has no
///       behavioural symptom to test instead -- moving the `try` up is
///       silent until somebody also edits the guard.
/// \note `TestTheHostRejectionsFireBeforeAnyDispatch` is a DIFFERENT claim
///       and neither implies the other: that one asserts no dispatch was
///       recorded, which stays true however wide the `try` is.
void TestTheHostRejectionsAreRaisedOutsideTheTry() {
    const auto lines = RunBodyLines();
    TEST_REQUIRE(lines.size() > 40);

    const auto opens = DeclarationIndex(lines, "        try {");
    TEST_REQUIRE(opens < lines.size());
    // Exactly one, or "the first `try`" is not the one that guards dispatch.
    TEST_REQUIRE(std::count_if(lines.begin(), lines.end(), [](const std::string& l) {
                     return l.find("try {") != std::string::npos;
                 }) == 1);

    // EVERY REFUSAL THE ENGINE PROMISES IS NON-POISONING, each identified by
    // the message it throws rather than by a line number.
    static constexpr std::string_view kRejections[] = {
        "Gemma 4 request contains no token IDs",
        "outside the vocabulary of ",
        "Gemma 4 request exceeds configured context capacity",
        "a multi-row pass must start at position 0",
        "RequireOneRowExtent(padded);",
    };
    for (const auto rejection : kRejections) {
        const auto at = DeclarationIndex(lines, rejection);
        if (at >= opens) {
            throw std::runtime_error(
                "a host rejection that must not poison the engine is raised at or "
                "after the `try` that defines poisoning: '" +
                std::string(rejection) + "'");
        }
    }

    // And the poisoned-engine gate itself is the first thing the function
    // does, so a poisoned engine never reaches any of the above.
    TEST_REQUIRE(DeclarationIndex(lines, "usable();") < opens);
}

void RequireANonFiniteResultPoisonsTheEngine(const Fixture& fixture) {
    Harness harness(fixture);
    fake_corelib::GetState().read_returns_nonfinite = true;
    std::vector<int> ids{1, 2, 3};
    const auto error = RequireThrows([&] { (void)harness.engine->prefill(ids); });
    fake_corelib::GetState().read_returns_nonfinite = false;
    // Two claims, not one: the guard fired at all (it had no test before
    // C9), and firing it poisons. An engine that returned an all-NaN logit
    // vector would otherwise drive an argmax to 0 -- `<pad>`, the empty
    // string -- for every step, and print a plausible token rate over it.
    RequireContains(error, "Gemma 4");
    RequireContains(error, "non-finite");
    TEST_REQUIRE(harness.engine->poisoned());
}

void TestE2bANonFiniteResultPoisonsTheEngine() {
    RequireANonFiniteResultPoisonsTheEngine(E2bFixture());
}
void TestE4bANonFiniteResultPoisonsTheEngine() {
    RequireANonFiniteResultPoisonsTheEngine(E4bFixture());
}

/// THE ENGINE'S SECOND `RequireFinite` CALL SITE, WHICH UNTIL NOW WAS NEVER
/// REACHED BY ANYTHING.
///
/// The test above poisons every read, so the layer loop's check fires first
/// and the head's never runs. Review-C9's m-1 measured the consequence:
/// deleting `RequireFinite(raw, "the head's logits")` left the suite 57/57
/// green. So `State::nonfinite_read_index` narrows the injection to ONE read
/// and this test aims it at the logits.
///
/// WHY BOTH SITES MATTER SEPARATELY. The layer-loop check guards the hidden
/// state; the head's guards the vector a caller actually receives, and it is
/// the one standing between an all-NaN logit vector and an argmax that
/// answers 0 -- `<pad>`, the empty string -- for every step, under a
/// plausible token rate, exiting 0. They are also on opposite sides of the
/// `lm_head` matmul, so a NaN introduced by the head alone is invisible to
/// the first.
void RequireANonFiniteLogitVectorPoisonsTheEngine(const Fixture& fixture) {
    Harness harness(fixture);
    auto& state = fake_corelib::GetState();
    // A MARK RATHER THAN AN ASSUMPTION about the pass's first read being
    // index 0: if a load ever performs a read, the offset stays correct.
    const auto mark = state.tensor_reads.size();
    state.read_returns_nonfinite = true;
    state.nonfinite_read_index = mark + 1;

    std::vector<int> ids{1, 2, 3};
    const auto error = RequireThrows([&] { (void)harness.engine->prefill(ids); });
    state.read_returns_nonfinite = false;
    state.nonfinite_read_index.reset();

    // NAMED, not merely non-finite: "the head's logits" is the string the
    // second call site passes and the first one does not, so this assertion
    // cannot be satisfied by the guard that already had a test.
    RequireContains(error, "Gemma 4");
    RequireContains(error, "the head's logits");
    TEST_REQUIRE(harness.engine->poisoned());

    // AND THE INJECTION LANDED WHERE IT WAS AIMED. Two reads in the pass, and
    // the poisoned one is the vocabulary-wide read off `logits` -- so the
    // first read returned the synthetic constant, the layer-loop check passed
    // on it, and the head ran.
    const auto& reads = state.tensor_reads;
    TEST_REQUIRE(reads.size() == mark + 2);
    TEST_REQUIRE(reads[mark].count == static_cast<std::size_t>(harness.shape.hidden));
    TEST_REQUIRE(reads[mark + 1].count ==
                 static_cast<std::size_t>(harness.shape.vocab));
}

void TestE2bANonFiniteLogitVectorPoisonsTheEngine() {
    RequireANonFiniteLogitVectorPoisonsTheEngine(E2bFixture());
}
void TestE4bANonFiniteLogitVectorPoisonsTheEngine() {
    RequireANonFiniteLogitVectorPoisonsTheEngine(E4bFixture());
}

/// EVERY ENTRY POINT, NAMED ONE BY ONE. A poisoned engine that still answers
/// one of these is an engine that gets silently reused -- which is the whole
/// reason the flag exists.
void TestAPoisonedEngineRefusesEveryEntryPoint() {
    Harness harness(E2bFixture());
    fake_corelib::GetState().fail_dispatch_at = 20;
    std::vector<int> ids{1, 2, 3};
    (void)RequireThrows([&] { (void)harness.engine->prefill(ids); });
    fake_corelib::GetState().fail_dispatch_at.reset();
    TEST_REQUIRE(harness.engine->poisoned());

    auto& engine = *harness.engine;
    RequireContains(RequireThrows([&] { (void)engine.prefill(ids); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)engine.forward(1); }), "poisoned");
    RequireContains(RequireThrows([&] { engine.clear_context(); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)engine.get_current_context_length(); }),
                    "poisoned");
    RequireContains(RequireThrows([&] { engine.set_context_length(0); }), "poisoned");
    RequireContains(RequireThrows([&] { engine.update_max_length(64); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)engine.checkpoint(); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)engine.restore(); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)engine.get_k_cache(0, 0); }), "poisoned");
    RequireContains(RequireThrows([&] { (void)engine.get_v_cache(0, 0); }), "poisoned");
    RequireContains(RequireThrows([&] {
                        Q4NX* nothing = nullptr;
                        engine.load_weights(*nothing);
                    }),
                    "poisoned");

    // NOT GUARDED, AND DELIBERATELY SO: these three answer questions about
    // the LOAD, which happened and is still true. A caller unloading a
    // poisoned model still wants to log what it was.
    TEST_REQUIRE(engine.poisoned());
    TEST_REQUIRE(engine.weight_slot_count() == ExpectedSlots(harness.shape));
    TEST_REQUIRE(!engine.loaded_from_cache());
}

/// The other half of the definition, and the one that costs a user something
/// if it is wrong: a rejection BEFORE the first dispatch leaves the engine
/// usable. Poisoning on these would turn every bad request into a forced
/// model reload.
void TestARejectionBeforeAnyDispatchDoesNotPoison() {
    Harness harness(E2bFixture());
    auto& engine = *harness.engine;

    std::vector<int> nothing;
    (void)RequireThrows([&] { (void)engine.prefill(nothing); });
    TEST_REQUIRE(!engine.poisoned());

    (void)RequireThrows([&] { (void)engine.forward(harness.shape.vocab); });
    TEST_REQUIRE(!engine.poisoned());

    engine.set_context_length(5);
    std::vector<int> ids{2, 5, 9};
    (void)RequireThrows([&] { (void)engine.prefill(ids); });
    TEST_REQUIRE(!engine.poisoned());

    engine.set_context_length(flm::gemma4::kMaxSequenceLength);
    (void)RequireThrows([&] { (void)engine.forward(3); });
    TEST_REQUIRE(!engine.poisoned());

    // AND THE ENGINE STILL RUNS. Without this the test would pass on an
    // engine that was merely broken in some other way.
    engine.clear_context();
    TEST_REQUIRE(engine.prefill(ids).size() ==
                 static_cast<std::size_t>(harness.shape.vocab));
    TEST_REQUIRE(engine.get_current_context_length() == 3);
    TEST_REQUIRE(!engine.poisoned());
}

/// A POISONED ENGINE STILL TEARS DOWN IN THE RIGHT ORDER. The teardown that
/// matters most is the one after a failure -- that is when a caller unloads --
/// and it is the one reached through an exception unwind rather than a normal
/// return.
void TestAPoisonedEngineStillTearsDownInOrder() {
    Harness harness(E2bFixture());
    fake_corelib::GetState().fail_dispatch_at = 20;
    std::vector<int> ids{1, 2, 3};
    (void)RequireThrows([&] { (void)harness.engine->prefill(ids); });
    fake_corelib::GetState().fail_dispatch_at.reset();
    TEST_REQUIRE(harness.engine->poisoned());

    const auto tensors = fake_corelib::GetState().tensor_creates.size();
    const auto mark = fake_corelib::GetState().release_order.size();
    harness.engine.reset();
    RequireTeardownSliceIsOrdered(ReleasesSince(mark), ExpectedSlots(harness.shape),
                                  tensors);
    TEST_REQUIRE(harness.runtime->api()->live_object_count() == 0);
}

void TestAContextPastTheGgufsOwnIsRefused() {
    ScopedWeightCache no_cache{"0"};
    RuntimeScope teardown;
    fake_corelib::Reset();
    auto runtime = CorelibRuntime::CreateForTest(
        CorelibApi::ResolveForTest(fake_corelib::Resolver()));
    auto package = Gemma4GgufPackage::Open(E2bFixture().path);
    auto shape = package->Config();
    shape.context_length = 8192;
    const auto error = RequireThrows([&] {
        (void)gemma4_rai::CreateWithShapeForTest(package, runtime, LM_Config{}, shape, 12288);
    });
    RequireContains(error, "past this GGUF's context of 8192");
    package.reset();
    runtime.reset();
    CorelibRuntime::ShutdownProcess();
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestLoadWeightsThrowsAndNamesGguf);
    RUN_TEST(TestStreamEnablesTheGemma4BundleBeforeAnyKernelQuery);
    RUN_TEST(TestAFreshEngineIsNotPoisonedAndStartsAtPositionZero);
    RUN_TEST(TestE2bHoldsTenWeightsPerOwningLayerAndSixPerSharingOne);
    RUN_TEST(TestE4bHoldsTenWeightsPerOwningLayerAndSixPerSharingOne);
    RUN_TEST(TestE2bPacksEveryProjectionAtGroupAndLmHeadAloneAtHeadGroup);
    RUN_TEST(TestE4bPacksEveryProjectionAtGroupAndLmHeadAloneAtHeadGroup);
    RUN_TEST(TestE2bSsmlpDeclaresGeluAndKeepsBothOfItsOwnNorms);
    RUN_TEST(TestE4bSsmlpDeclaresGeluAndKeepsBothOfItsOwnNorms);
    RUN_TEST(TestPleUsesPleGroupAndNotGroup);
    RUN_TEST(TestE2bPacksOnePleBlockPerLayer);
    RUN_TEST(TestE4bPacksOnePleBlockPerLayer);
    RUN_TEST(TestE2bPacksTheStandaloneNormsIncludingVsVectorOfOnes);
    RUN_TEST(TestE4bPacksTheStandaloneNormsIncludingVsVectorOfOnes);
    RUN_TEST(TestE2bHoldsTwoDistinctRotaryPairsWrittenOnce);
    RUN_TEST(TestE4bHoldsTwoDistinctRotaryPairsWrittenOnce);
    RUN_TEST(TestE2bAllocatesTheWiderGeometryAndNonUniformCaches);
    RUN_TEST(TestE4bAllocatesTheWiderGeometryAndNonUniformCaches);
    RUN_TEST(TestE2bPutsEveryPackedWeightInTheSlotNamedForIt);
    RUN_TEST(TestE4bPutsEveryPackedWeightInTheSlotNamedForIt);
    RUN_TEST(TestADispatchRecordsTheWeightsObjectItRanWith);
    RUN_TEST(TestPleAndRmsNormAreRecordedAsDispatchesInSequence);
    RUN_TEST(TestAnMhaDispatchRecordsItsRotaryTablesAndCaches);
    RUN_TEST(TestTensorReadsAreRecordedWithTheirOffset);
    RUN_TEST(TestE2bPrefillDispatchesTheLayerSequence);
    RUN_TEST(TestE4bPrefillDispatchesTheLayerSequence);
    RUN_TEST(TestE2bLayerDataFlowTakesPlaneZeroAndPlesSecondOutput);
    RUN_TEST(TestE4bLayerDataFlowTakesPlaneZeroAndPlesSecondOutput);
    RUN_TEST(TestE2bEveryLayerBindsItsOwnGeometrysWidths);
    RUN_TEST(TestE4bEveryLayerBindsItsOwnGeometrysWidths);
    RUN_TEST(TestE2bVNormWritesTheCacheOncePerKvHead);
    RUN_TEST(TestE4bVNormWritesTheCacheOncePerKvHead);
    RUN_TEST(TestE4bVProjectionLandsThroughAStridedWindow);
    RUN_TEST(TestE2bAttentionBindsItsOwnRotaryPairAndItsOwnersCaches);
    RUN_TEST(TestE4bAttentionBindsItsOwnRotaryPairAndItsOwnersCaches);
    RUN_TEST(TestTheReturnedLogitsAreSoftcapped);
    RUN_TEST(TestEveryLayerIsToldTheSamePositionAndADecodeAdvancesIt);
    RUN_TEST(TestE2bReadsTheLastLiveRowAndNothingElse);
    RUN_TEST(TestDecodeRunsOneRowAndAdvancesThePosition);
    RUN_TEST(TestACacheReadResolvesThroughTheOwner);
    RUN_TEST(TestTheHostRejectionsFireBeforeAnyDispatch);
    RUN_TEST(TestAMultiRowContinuationReplaysFromPositionZero);
    RUN_TEST(TestAPagedContinuationOnAPageBoundaryRunsOnlyItsOwnPage);
    RUN_TEST(TestAPagedContinuationOffAPageBoundaryReplaysFromItsPage);
    RUN_TEST(TestE2bSecondLoadOfTheSameGgufBindsFromTheCache);
    RUN_TEST(TestE4bSecondLoadOfTheSameGgufBindsFromTheCache);
    RUN_TEST(TestACacheWrittenForE2bIsNotBoundForE4b);
    RUN_TEST(TestE2bCacheBoundWeightsLandInTheSlotNamedForThem);
    RUN_TEST(TestACacheWrittenUnderOnePackedLayoutIsStaleUnderTheNext);
    RUN_TEST(TestEveryCorelibOwningMemberIsDeclaredAfterTheStream);
    RUN_TEST(TestE2bTeardownReleasesEverythingBeforeTheStream);
    RUN_TEST(TestE4bTeardownReleasesEverythingBeforeTheStream);
    RUN_TEST(TestTheRuntimeIsCleanedUpAfterEveryCorelibObject);
    RUN_TEST(TestTwoModelsTearDownInOneProcessE2bFirst);
    RUN_TEST(TestTwoModelsTearDownInOneProcessE4bFirst);
    RUN_TEST(TestE2bADispatchFailurePoisonsTheEngine);
    RUN_TEST(TestE4bADispatchFailurePoisonsTheEngine);
    RUN_TEST(TestE2bANonFiniteResultPoisonsTheEngine);
    RUN_TEST(TestE4bANonFiniteResultPoisonsTheEngine);
    RUN_TEST(TestE2bANonFiniteLogitVectorPoisonsTheEngine);
    RUN_TEST(TestE4bANonFiniteLogitVectorPoisonsTheEngine);
    RUN_TEST(TestAPoisonedEngineRefusesEveryEntryPoint);
    RUN_TEST(TestARejectionBeforeAnyDispatchDoesNotPoison);
    RUN_TEST(TestE2bARefusedFirstDispatchDoesNotPoison);
    RUN_TEST(TestE4bARefusedFirstDispatchDoesNotPoison);
    RUN_TEST(TestTheHostRejectionsAreRaisedOutsideTheTry);
    RUN_TEST(TestAPoisonedEngineStillTearsDownInOrder);
    RUN_TEST(TestAContextPastTheGgufsOwnIsRefused);
#undef RUN_TEST
}
