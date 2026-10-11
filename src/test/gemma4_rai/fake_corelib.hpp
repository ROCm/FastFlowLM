#pragma once

// A corelib test double scoped to what src/test/gemma4_rai's targets need.
//
// This is a SEPARATE fake from src/test/phi4_rai/fake_corelib.{hpp,cpp} --
// deliberately, not out of preference for "one fake per model". Two concrete
// gaps rule out reusing or extending the Phi-4 fake for Gemma 4's shape-plan
// tests, and both are structural, not stylistic:
//
//   1. Phi-4's `RowsPadCall` records only `{helper, m, k, n, group_size}` --
//      it does not carry the ssmlp weights descriptor's `activation` or
//      `post_feedforward_layernorm` fields at all. Gemma 4's shape plan must
//      pin BOTH (see gemma4_rai_shape_plan.cpp), so a call record that drops
//      them cannot express the assertion.
//   2. None of Phi-4's `MatmulPadCall`/`RowsPadCall`/`MhaPadCall` records the
//      `stream` argument, even though the fake receives it -- there is
//      nothing for "the stream was passed to every pad helper" to compare
//      against.
//
// Editing src/test/phi4_rai/fake_corelib.{hpp,cpp} to add these is out of
// scope here: that fake is owned by another task and its own tests already
// pin its current shape. This file is therefore a full, independent
// implementation of the same `FLM_CORELIB_FUNCTIONS` ABI surface
// (corelib_api.hpp requires every entry to resolve).
//
// WHAT THIS FAKE VALIDATES, AND WHAT IT CANNOT.
//
// Round 1 of review found this file recording calls and validating almost
// nothing while backing the same ABI as the Phi-4 fake, which validates a
// great deal -- and named that gap as the shape that let two Criticals
// through a fully green suite on the Phi-4 side. The checks below were ported
// from that fake and adapted. They are:
//
//   - `create_tensor_window` REJECTS a window that does not fit inside the
//     allocation it is carved from, measured against the ROOT allocation that
//     every window in a chain shares.
//   - `tensor_write` / `tensor_read` REJECT a (count, offset) pair that runs
//     past the tensor's OWN declared extent -- which is what corelib.h says
//     they are bounded by, not the page-rounded allocation behind it.
//   - a dispatch's row-bearing operands must AGREE with each other, and each
//     must carry a row count the matching pad helper is KNOWN to have
//     produced (`State::pad_answers_seen`, via `OnGrid`).
//   - `ryzenai_corelib_ssmlp_bf16` REJECTS two operands that are windows over
//     the SAME backing allocation. Comparison is on backing STORAGE identity
//     (`DispatchRecord::operand_storage`), never on the `void*` handle, since
//     `create_tensor_window` mints a fresh handle every call and two windows
//     onto one allocation therefore compare unequal by handle.
//   - `matmul_pad_shape` can be made to perturb `k`/`n`
//     (`State::matmul_k_delta` / `matmul_n_delta`), so the shape plan's
//     "helper changed padded K/N" rejection is reachable from a test instead
//     of being dead code.
//   - THE PACKED BYTES NAME WHAT WAS PACKED INTO THEM. Every weights object
//     carries a four-byte `identity_tag` taken from its own components;
//     `weights_copy_data` stamps it at offset 0 of the blob it hands out, so
//     the weight cache's file contains it, and each
//     `..._weights_create_from_file` reads it back out of the slice it was
//     given. Without that the CACHE path carries no identity at all -- the
//     descriptor says "some weight of this shape", corelib returns no
//     components -- and `BindFromCache`'s assignment BY SLOT INDEX has
//     nothing watching it. It is also the more faithful model: real corelib's
//     packed bytes are a function of the weight, so two distinct tensors
//     never produce one blob, which the old descriptor-only size was
//     pretending otherwise.
//   - a `flat_mha` / `ssmlp` dispatch RETAINS THE DESCRIPTOR it ran with
//     (`DispatchRecord::mha_desc` / `ssmlp_desc`). Nothing here judges it --
//     see the `kv_shared` bullet below -- but retaining it is what makes an
//     assertion about the value that actually crossed the ABI writable at
//     all, which is what the reference driver's own per-layer `kv_shared`
//     guard needs (`gemma4_driver.py:1670-1690`).
//
// AND HERE IS WHAT IT STILL CANNOT CATCH, stated plainly because this file
// has twice taken a Critical for a comment that read as broader coverage than
// the code delivered:
//
//   - `pad_answers_seen` is filled ONLY by this engine's own calls into these
//     same pad helpers, and `Gemma4ShapePlan::Build()` enumerates EVERY row
//     bucket at construction. So for a helper the plan actually queried (and
//     only for such a helper -- see the `OnGrid` bullet further down), that
//     helper's answer set already holds every bucket's answer by the time any
//     dispatch runs. A dispatch
//     whose operands are UNIFORMLY sized to the wrong bucket -- decode handed
//     bucket 128's rows instead of bucket 1's -- passes both `RowsAgree` and
//     `OnGrid` and would fail on hardware. This is not a gap in the checks;
//     it is a gap the fake CANNOT close, because the live row count a
//     dispatch is meant to use is exactly the argument corelib 0.5.0 removed
//     from the dispatch ABI. Hardware is the only real check.
//   - NOTHING HERE CHECKS THE DESCRIPTOR AGAINST THE SHIPPED KERNEL SET.
//     `flat_mha_pad_rows` records the descriptor and always succeeds. A
//     `scale` or `window` that no artifact ships for -- the exact Critical
//     round 1 found -- is invisible to this fake and is caught only by a test
//     that reads the recorded field itself (see
//     TestAttentionScaleIsExactlyOneEverywhere) or by real hardware.
//   - `kv_shared` likewise. The fake cannot TELL an owning dispatch given a
//     kvshare descriptor from a correct one; neither can corelib -- both
//     buffers exist, both are the right shape, and the kernel that runs is
//     simply a different one. What the fake now does is RECORD the value that
//     crossed the ABI (`DispatchRecord::mha_desc`), which is a strictly
//     weaker thing and is the whole of the improvement: it moves the check
//     from impossible to writable, and leaves writing it to the caller. The
//     reference driver writes exactly that check host-side, against
//     `cfg.kv_layers`, and raises at load rather than at the first prompt
//     (`gemma4_driver.py:1670-1690`); C7 should port it. Until it does, the
//     only thing pinning `kv_shared` is the descriptor-level assertions in
//     test_gemma4_shape_plan.cpp.
//   - THE IDENTITY ON THE CACHE LEG IS FOUR BYTES, NOT THE WHOLE COMPONENT.
//     `identity_tag` is one block tag, or one BF16 gamma element, or (for
//     ssmlp) the gate stream's tag alone -- enough to say WHICH weight, never
//     enough to say the whole of it was right. Two norms agreeing in their
//     first element and differing later are one weight to a cache-bound
//     engine here; so are two ssmlp blobs whose gates agree and whose ups do
//     not. Neither can occur against this fixture (every F32 tensor is
//     constant-filled, every Q8_0 tensor's tag is unique) and the pack leg
//     still checks the full gamma and all three block streams. On a real
//     packed blob the content IS the weight, so the gap is the fake's, not
//     the design's.
//   - The row grid here is one flat "round up to 64, except M == 1" rule for
//     every op. Real corelib ships a different grid per op and per shape. A
//     bug that depends on two ops' grids genuinely differing cannot be seen
//     here.
//   - `OnGrid` REJECTS a matmul output width nothing ever queried a pad
//     answer for. That is only sound while the plan queries every width the
//     engine will dispatch: a width the plan misses is a FALSE REJECTION of a
//     correct dispatch, not a missing check. It has already happened once --
//     lm_head's `hidden -> vocab` was unqueried and C7's first dispatch was
//     refused -- which is why `Gemma4ShapePlan::Build()` now plans lm_head.
//     Read OnGrid's own comment in fake_corelib.cpp before adding a dispatch
//     at a width the plan does not query.
//   - The write/read bound above holds only for objects that HAVE a declared
//     shape. `WithinExtent` returns true when the shape is empty, and every
//     object minted by `matmul_weights_create_*`, `ssmlp_weights_create_*`,
//     `create_stream` and `MakeStreamForTest` has an empty shape -- so a
//     `tensor_write`/`tensor_read` aimed at one of those is UNBOUNDED here.
//     DEFERRED, not overlooked: corelib's weights and stream objects are not
//     tensors and no correct engine writes through them, so tightening it
//     would be guarding a path that only misuse reaches. Stated because the
//     bullet above reads, on its own, as an unqualified guarantee.
//   - `create_tensor_window` applies its fit check only when the parent has
//     storage. A window carved from a NULL parent has none, skips the check,
//     and then also slips past `AllDistinctStorage`, which skips
//     storage-less operands. Misuse rather than a live path; deferred for the
//     same reason.
//   - `MatmulHelperFor` keys on the output width N ALONE, so two matmuls with
//     the same N and different K share one grid entry. GEMMA 4 HAS SUCH A
//     PAIR, so this is not hypothetical: the output projection is
//     `q_heads*head_dim -> hidden` on a sliding layer and
//     `q_heads*global_head_dim -> hidden` on a full one -- 2048->1536 and
//     4096->1536 on E2B -- and both land on `"matmul-1536"`. Nothing else
//     collides on either shipped row (E2B queries N in {2048, 4096, 256, 512,
//     1536, 262144}; E4B in {2048, 4096, 512, 1024, 2560, 262144}).
//     DEFERRED: with one shared row rule here the two entries would hold the
//     same answers anyway, so keying on (K, N) would buy nothing today. It
//     would start to matter the moment this fake models per-shape grids.
//   - `release_order` RECORDS AN ORDER; IT DOES NOT MODEL THE FAULT. corelib
//     access-violates somewhere inside `ryzenai_corelib_object_release` when
//     an object outlives its Stream, and this fake's `object_release` is a
//     `delete` that cannot and does not reproduce that. So a green
//     release-order test says "this engine releases in the order that was
//     measured not to fault", NEVER "this engine does not fault". Nothing in
//     this project has torn a Gemma 4 engine down against real corelib.
//   - `fail_dispatch_at` INJECTS a status; it does not model a reason. The
//     engine's own comment says a mid-chain rejection against real corelib
//     is PROCESS-FATAL -- the unwind destructs runs whose commands are still
//     in flight and XRT aborts uncatchably. Here the unwind is clean, so a
//     test can observe `poisoned()` at all. What the test proves is that the
//     FLAG is set on that path, never that the path is survivable.
//     The same caveat covers `fail_dispatch_at = 0`, which C9's fix round
//     uses for the other half of the definition: it shows that the engine
//     does not POISON when its first dispatch is refused, and says nothing
//     about whether real corelib can refuse one without having enqueued it.
//   - `nonfinite_read_index` NARROWS the NaN injection to one read so the
//     engine's SECOND `RequireFinite` call site is reachable at all. It
//     chooses which guard fires; it does not make a NaN at that point
//     plausible, and no test may read anything into WHY the logits would be
//     non-finite while the hidden state was not.
//
// One further, load-bearing absence: there is no `ple_pad_calls` here, and
// there never will be. The pinned corelib.h has no
// `ryzenai_corelib_ple_bf16_pad_rows` (or any other ple padding entry point)
// at all -- ple_bf16's own doc comment says M "comes from `x`'s own extent
// and SELECTS THE KERNEL", implicitly, with no helper to ask. So the shape
// plan computes `ple_rows` (and, by the same reasoning, `rmsnorm_rows`) from
// a local, hardcoded bucket table taken verbatim from corelib.h's own
// ple_bf16 comment, never from a corelib call -- there is nothing here to
// fake.
//
// THE PLE AND RMSNORM *WEIGHTS* ENTRY POINTS ARE A DIFFERENT MATTER AND ARE
// FAKED FOR REAL. An earlier version of this paragraph said
// `FLM_CORELIB_FUNCTIONS` "has no ple or rmsnorm entries either", which was
// true when C6 wrote it and is not any more: C7 added
// `ple_bf16_weights_pack`/`_create`/`ple_bf16` and
// `rmsnorm_bf16_weights_create_reference`/`_from_file`/`rmsnorm_bf16`,
// because Gemma 4 dispatches both operators every layer. What this fake does
// with them:
//
//   - `ple_bf16_weights_pack` HONOURS THE TWO-CALL PROTOCOL. `out == NULL`
//     reports the size and dereferences none of the four arrays, and records
//     (`PlePackCall::read_any_array`) that it did not, so "the engine sized
//     its buffer before it held the data" is a checkable claim rather than a
//     comment. It also refuses `group_size != 32`, which is the only value
//     mladfple ships.
//   - it RETAINS `post_norm` and `next_norm` in full. `next_norm` is the
//     NEXT layer's attention-norm gamma and the off-by-one there still
//     generates text, so it has to be assertable per layer. The layer a pack
//     call belongs to is read off `desc.layer_scale`, which the fixture
//     makes unique per layer.
//   - `rmsnorm_bf16_weights_create_reference` retains the whole BF16 gamma,
//     so "V's norm is a vector of exactly 1.0" is checkable rather than
//     assumed.
//   - `ple_bf16` and `rmsnorm_bf16` ARE RECORDED AS DISPATCHES, which until
//     Task R3 they were not: only matmul / ssmlp / mha minted a
//     `DispatchRecord` and everything else fell through the generic success
//     branch. A Gemma 4 layer is SEVEN dispatches on a sharing layer and
//     `10 + kv_heads` on an owning one -- 11 on E2B, 12 on E4B -- of which
//     these two ops are 3 of 7 and 5 of 11 (6 of 12) respectively, so C8's
//     layer loop would have been measured by an instrument blind to roughly
//     HALF of it. (An earlier version of this comment said "ten and six":
//     those are the WEIGHT-OBJECT counts from gemma4_rai.hpp:40-47, correct
//     for weights and wrong for dispatches. Counted from gemma4_driver.py
//     :2254-2336.) Each
//     carries its operands in the header's own order, their backing STORAGE
//     identity, the weights object it ran with, and its position in
//     `State::dispatches`, which is ABI call order.
//   - `ple_bf16` enforces what its section of corelib.h states: `out` and
//     `norm_out` distinct from each other and from `x` (on storage, not on
//     handles), and a row count the ELF set actually ships -- see OnPleGrid,
//     which is stricter than corelib on purpose.
//   - `rmsnorm_bf16` enforces NEITHER of those, and the absences are real
//     rather than pending: in-place is explicitly legal for small M, and
//     `rmsnorm_bf16_pad_rows` is an identity function that is not even bound
//     in FLM_CORELIB_FUNCTIONS, so there is no grid to check against. See the
//     dispatch branch in fake_corelib.cpp for both, written out.
//   - `weights_copy_data` now reports a real (small, synthetic) size instead
//     of 0. It used to report 0 unconditionally, which silently disabled the
//     weight cache's whole write path.

#include "rai/corelib_api.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fake_corelib {

/// \brief one `ryzenai_corelib_matmul_bf16_pad_shape` call
/// \note `m`, `k` and `n` are recorded as they were PASSED IN, before any
///       padding or `matmul_*_delta` perturbation, so a test can classify a
///       call by its logical operand shape.
struct MatmulPadCall {
    ryzenai_corelib_stream_ptr stream;
    std::int64_t m;
    std::int64_t k;
    std::int64_t n;
    std::uint32_t group_size;
};

/// \brief one `ryzenai_corelib_ssmlp_bf16_pad_rows` call
///
/// Carries the WHOLE descriptor, not a projection of it -- `activation` and
/// `post_feedforward_layernorm` are exactly the fields Gemma 4's shape plan
/// must get right and Phi-4's fake never recorded (see this header's own
/// top comment).
struct RowsPadCall {
    ryzenai_corelib_stream_ptr stream;
    std::int64_t m;
    ryzenai_corelib_ssmlp_bf16_weights_desc desc;
};

/// \brief one `ryzenai_corelib_flat_mha_bf16_pad_rows` call
///
/// Carries the WHOLE descriptor for the same reason: `scale`, `kv_shared`
/// and `rope_dim` change WHICH KERNEL RUNS rather than any shape
/// (corelib.h:1810-1812), so a fake that dropped them would make the only
/// assertions capable of catching a wrong one impossible to write.
struct MhaPadCall {
    ryzenai_corelib_stream_ptr stream;
    std::int64_t m;
    ryzenai_corelib_flat_mha_bf16_desc desc;
};

/// \brief one `..._enum_kernels` call, and the stream it was asked about
struct EnumKernelsCall {
    std::string entry;
    ryzenai_corelib_stream_ptr stream;
};

/// \brief a matmul kernel an explicit grid reports
struct ShippedMatmul {
    std::int64_t m, k, n, group;
};

/// \brief a fused-MLP kernel an explicit grid reports
struct ShippedSsMlp {
    std::string family;
    std::int64_t m, k, n, group;
};

/// \brief an attention kernel an explicit grid reports
/// \note `window` is the window the kernel is KEYED on, which corelib only
///       writes when it is shorter than the rows the kernel runs.
struct ShippedMha {
    std::int64_t heads, kv_heads, m, head_size, max_seq, window;
    bool kv_shared, scale_one;
};

/// \brief one paged attention kernel `paged_mha_enum_kernels` reports
struct ShippedPaged {
    std::int64_t heads, kv_heads, m, head_size, page_size;
};

/// \brief one `ryzenai_corelib_matmul_bf16_weights_create_*` call
/// \note The WHOLE descriptor, because `group_size` is the field C7 must get
///       right and it is the one nothing else in this codebase can observe:
///       `group`, `head_group` and `ple_group` are all 32 on both shipped
///       Gemma 4 rows, so a weight requantized at the wrong one is a wrong
///       weight that every real config agrees with. A test forces them apart
///       (`head_group = 64`) and reads this back.
struct MatmulWeightsCreateCall {
    ryzenai_corelib_matmul_bf16_weights_desc desc;
    /// \brief true for the `..._from_file` leg, i.e. a cache bind rather than
    ///        a pack -- which is how "the second load hit the cache" is told
    ///        apart from "the second load packed again"
    bool from_file;
    /// \brief the four-byte tag at the head of the block stream, 0 on the
    ///        `..._from_file` leg
    /// \note WHICH QUANTIZED TENSOR WENT IN. The descriptor answers only
    ///       "some tensor of this shape at this group size", and Gemma 4 has
    ///       35 (E2B) layers' worth of identically-shaped `attn_q` weights.
    ///       See gemma4_gguf_fixture.hpp's BlockSignatureAt.
    std::uint32_t blocks_tag;
    /// \brief the weights object this call MINTED
    /// \note The bridge between "which bytes went in" and "which slot the
    ///       finished object ended up in" -- the two are different questions
    ///       and only the first was answerable before. See
    ///       DispatchRecord::weights and gemma4_rai's WeightPlacementsForTest.
    void* object;
};

/// \brief one `ryzenai_corelib_ssmlp_bf16_weights_create_*` call
/// \note Carries the FIRST ELEMENT of each norm, not the whole vector. That
///       is enough only because the fixture fills every F32 tensor with a
///       constant signature derived from its own name
///       (gemma4_gguf_fixture.hpp's `NormSignature`), so one element
///       identifies WHICH tensor was passed. Against a real GGUF it would
///       identify nothing, and this record would have to grow.
struct SsMlpWeightsCreateCall {
    ryzenai_corelib_ssmlp_bf16_weights_desc desc;
    bool from_file;
    bool has_components;
    std::uint16_t epsilon_bf16;
    std::uint16_t norm0_bf16;
    std::uint16_t norm1_bf16;
    /// \brief the four-byte tag at the head of each of the three block
    ///        streams, 0 on the `..._from_file` leg
    ///
    /// THE PORT ASSIGNMENT, WHICH NOTHING COULD SEE BEFORE. A gated MLP puts
    /// `gate` through the activation and `up` not, so exchanging the two
    /// computes `GELU(x @ up) * (x @ gate)` -- a different function, finite,
    /// plausible and wrong. corelib takes all three as bare `const void*`
    /// block streams of the same quant type, so nothing in the ABI, the
    /// descriptor or the shapes distinguishes them (`gate` and `up` are the
    /// SAME shape on every Gemma 4 layer). The fixture's per-tensor block
    /// tag is the only discriminator there is.
    std::uint32_t gate_tag;
    std::uint32_t up_tag;
    std::uint32_t down_tag;
    /// \brief the weights object this call minted -- see
    ///        MatmulWeightsCreateCall::object
    void* object;
};

/// \brief one `ryzenai_corelib_rmsnorm_bf16_weights_create_*` call
struct RmsNormWeightsCreateCall {
    /// \brief the PDI handed to the reference packer, or -1 on `..._from_file`
    int prefill_pdi;
    ryzenai_corelib_rmsnorm_bf16_weights_desc desc;
    bool from_file;
    /// \brief the BF16 gamma, `desc.k` entries, EMPTY on the from_file leg
    /// \note Retained in full rather than sampled: V's norm is a vector of
    ///       exactly 1.0 and the cheap wrong implementation is to skip it, so
    ///       a test has to be able to say "every element is one" rather than
    ///       "the first one is".
    std::vector<std::uint16_t> scale;
    /// \brief the gamma's first element, ON BOTH LEGS
    ///
    /// WHICH NORM THIS IS, ON THE ONE PATH WHERE `scale` CANNOT SAY. corelib
    /// hands no gamma back through `..._weights_create_from_file` -- it is
    /// given a slice of an opaque packed blob -- so a cache-bound engine
    /// carries no gamma anywhere and every verifier keyed on `scale` is blind
    /// to it. That matters because `BindFromCache` assigns handles by walking
    /// the slot table IN INDEX ORDER, which is a second, entirely independent
    /// way for a weight to land in the wrong member, and one layer's Q-norm
    /// and K-norm have IDENTICAL descriptors and therefore identical packed
    /// lengths: exchanging them binds cleanly.
    ///
    /// What closes it is that this fake's packed bytes now CARRY the identity
    /// of what was packed (see fake_corelib.cpp's `identity_tag`), so the
    /// from_file leg recovers this from the cache file itself. That is also
    /// the more faithful model: real corelib's packed bytes are a function of
    /// the weight, and two distinct gammas do not produce one blob.
    std::uint16_t scale_tag;
    /// \brief the weights object this call minted -- see
    ///        MatmulWeightsCreateCall::object
    /// \note Load-bearing for THIS record in particular. Gemma 4's Q-norm and
    ///       K-norm on one layer have identical descriptors (same `k`, same
    ///       epsilon) and differ only in their gamma, so "the object packed
    ///       from `attn_q_norm` is the one that reached the `q_norm` slot"
    ///       cannot be said without an object identity to join on.
    void* object;
};

/// \brief one `ryzenai_corelib_ple_bf16_weights_pack` call
///
/// BOTH LEGS OF THE TWO-CALL PROTOCOL LAND HERE, told apart by `sizing_leg`.
/// The sizing leg (`out == nullptr`) must report a size and READ NONE OF THE
/// ARRAYS; `read_any_array` records whether this fake dereferenced them, so
/// a test can prove the engine really made the sizing call first rather than
/// calling the full packer twice.
struct PlePackCall {
    ryzenai_corelib_ple_bf16_weights_desc desc;
    bool sizing_leg;
    bool read_any_array;
    std::size_t reported_size;
    /// \brief the first element of `gate` and of `proj`, 0 on the sizing leg
    /// \note WHICH TENSOR WENT INTO WHICH PORT. `gate` is [k, n] and `proj`
    ///       is [n, k]; on this model those are 1536x256 and 256x1536, the
    ///       same element count, so a swap is not a size error -- corelib.h
    ///       says passing one the other way round "packs silently and
    ///       produces noise". The fixture's per-tensor signature is what
    ///       makes the swap visible here.
    /// \note IDENTITY, NOT ORIENTATION. Element 0 is `M[0][0]` under either
    ///       orientation, so these two say WHICH tensor arrived and can say
    ///       nothing about whether it arrived transposed. `gate_at_*` and
    ///       `proj_at_*` below are the orientation probes.
    float gate_first;
    float proj_first;
    /// \brief two OFF-DIAGONAL probes of each matrix, 0 on the sizing leg
    ///
    /// THE PAIR THAT SWAPS UNDER A TRANSPOSE, WHICH IS THE WHOLE REASON THEY
    /// ARE HERE. Flat index 0 and flat index `k*n - 1` are both fixed points
    /// of a transpose -- first and last element of the buffer either way
    /// round -- so a record that probes only those is blind by construction,
    /// however the fixture is filled. Index 1 is `(row 0, col 1)` and index
    /// `n` (for `gate`, whose required layout is `[k, n]`) is
    /// `(row 1, col 0)`: transposing exchanges exactly these two. `proj`'s
    /// required layout is `[n, k]`, so its second probe is at index `k`.
    ///
    /// Reading them is only informative because `gemma4_gguf_fixture.hpp`
    /// gives 2-D F32 tensors a POSITIONAL fill -- against a constant-filled
    /// matrix every index reports the same number.
    float gate_at_1;
    float gate_at_n;
    float proj_at_1;
    float proj_at_k;
    /// \brief the inner RMSNorm's gamma, `desc.k` floats -- empty on the
    ///        sizing leg
    std::vector<float> post_norm;
    /// \brief THE NEXT LAYER'S attention-norm gamma, `desc.k` floats -- empty
    ///        on the sizing leg
    /// \note This is the off-by-one the header says "is not a crash -- every
    ///       layer is then normalized by its neighbour's gamma, which still
    ///       generates text". Retained per call so it can be asserted for
    ///       EVERY layer rather than spot-checked, keyed by
    ///       `desc.layer_scale` (which the fixture makes unique per layer).
    std::vector<float> next_norm;
};

/// \brief one `ryzenai_corelib_ple_bf16_weights_create` call
struct PleCreateCall {
    ryzenai_corelib_ple_bf16_weights_desc desc;
    std::size_t packed_size;
    /// \brief false when the blob's length was not exactly what this
    ///        descriptor packs to -- corelib.h: "a truncated blob is still a
    ///        plausible one", so this fake refuses the same way
    bool accepted;
    /// \brief the weights object this call minted, null when refused -- see
    ///        MatmulWeightsCreateCall::object
    void* object;
};

/// \brief one `ryzenai_corelib_create_device_tensor` call
struct TensorCreateRecord {
    ryzenai_corelib_data_type data_type;
    std::vector<std::int64_t> shape;
    void* object;
};

/// \brief one `ryzenai_corelib_create_tensor_window` call that SUCCEEDED
/// \note A rejected window records nothing -- there is no object to record.
struct TensorWindowRecord {
    void* parent;
    std::vector<std::int64_t> shape;
    std::size_t offset;
    void* object;
    /// \brief the strides the window was created with, empty for dense
    std::vector<std::int64_t> strides;
};

/// \brief one `ryzenai_corelib_tensor_write` call
struct TensorWriteRecord {
    void* tensor;
    ryzenai_corelib_data_type source_type;
    std::size_t count;
    std::size_t offset;
    bool accepted;
};

/// \brief one `ryzenai_corelib_tensor_read` call
/// \note WHICH ROW WAS READ, which is the whole content of the quietest bug a
///       forward pass has. A pass ends by reading the LAST live row of
///       `hidden`, at element offset `(rows - 1) * hidden`; reading row 0
///       instead returns the first prompt token's hidden state -- finite,
///       correctly shaped, correctly bounded, and the wrong position's logits.
///       Nothing downstream can tell.
/// \note Refused reads are recorded too (`accepted == false`): "read past the
///       end" and "did not read at all" are different bugs.
struct TensorReadRecord {
    void* tensor;
    ryzenai_corelib_data_type destination_type;
    std::size_t count;
    std::size_t offset;
    bool accepted;
};

/// \brief one accepted dispatch of matmul / ssmlp / rmsnorm / ple / flat_mha
/// \note A REFUSED DISPATCH RECORDS NOTHING. Every per-op check below returns
///       before the record is pushed, so `State::dispatches` is the list of
///       calls that would have reached the NPU.
struct DispatchRecord {
    /// \brief "matmul", "ssmlp", "rmsnorm", "ple" or "mha"
    std::string kind;
    void* stream;
    std::int64_t rows;
    /// \brief every operand of this dispatch corelib.h ties to a row count,
    ///        in a fixed per-kind order: matmul is {input, output}; ssmlp is
    ///        {input, residual, skip_sum, normalized}; rmsnorm is
    ///        {input, output}; ple is {x, ple, out, norm_out}; mha is
    ///        {q, k, out}
    /// \note THE ORDER IS PART OF THE CONTRACT, because it is how a caller
    ///       says "the thing that was normalized is the thing the block
    ///       wrote" rather than "two tensors were involved". It follows each
    ///       op's own argument order in corelib.h, minus the weights object
    ///       (which is `weights`) and minus operands with unrelated extent
    ///       rules (mha's cos/sin/caches).
    std::vector<void*> operands;
    /// \brief the underlying allocation identity behind each of `operands`,
    ///        in the same order
    /// \note An aliasing check MUST compare this, not `operands` itself.
    ///       `create_tensor_window` mints a fresh object every call, so two
    ///       DIFFERENT window objects can -- and, when a bug routes two
    ///       "must be distinct" operands through separate windows onto the
    ///       SAME backing, will -- point at the SAME storage. Entries here
    ///       would then be equal even though the corresponding `operands`
    ///       entries are not. See fake_corelib.cpp's StorageOf.
    std::vector<const void*> operand_storage;
    /// \brief the DECLARED SHAPE of each of `operands`, in the same order
    ///
    /// THE COLUMN EXTENT, WHICH NOTHING HERE RETAINED AND WHICH IS HALF OF
    /// EVERY GEOMETRY. `rows` is `operands.front()`'s leading extent and
    /// `RowsAgree`/`OnGrid` compare leading extents only, so until this field
    /// existed a `DispatchRecord` said nothing whatever about how WIDE its
    /// operands were. Gemma 4 runs TWO geometries -- sliding layers at
    /// `head_dim` and full-attention layers at `global_head_dim`, 256 and 512
    /// on both shipped rows -- over ONE set of allocations sized for the
    /// wider, windowed down per layer on the column axis. So a layer that
    /// binds the other geometry's windows produces operands that are the same
    /// tensors, at the same offsets, over the same storage, with the same row
    /// counts, and the wrong width. Every comparison this record could make
    /// before still passed: a mutation that bound the FULL geometry on every
    /// layer left the whole suite green.
    ///
    /// \note PURE RETENTION, like `mha_cos` and `weights`. Nothing in this
    ///       fake judges these shapes -- no width check was added to any
    ///       dispatch branch -- so no dispatch that was accepted before is
    ///       refused now, and the row checks keep the meaning they had. What
    ///       changes is only that "this layer ran at ITS OWN geometry's
    ///       widths" becomes a question a caller can ask.
    /// \note THE WHOLE SHAPE, NOT THE COLUMN COUNT. A rank-3 operand (the V
    ///       staging destination at more than one KV head) has no "column"
    ///       -- recording `shape[1]` for it would silently mean the middle
    ///       extent -- so the shape is carried as it was declared and the
    ///       reader says which extent it means.
    std::vector<std::vector<std::int64_t>> operand_shapes;
    /// \brief the flat_mha descriptor this dispatch was actually handed
    /// \note THE DESCRIPTOR, NOT THE PLAN'S COPY OF IT. Holding four
    ///       immutable descriptors makes the wrong one impossible to
    ///       CONSTRUCT; it does nothing about the right one being built and
    ///       the wrong one being HANDED OVER. The reference driver closes
    ///       that gap host-side by reading the field back per layer against
    ///       `cfg.kv_layers` and raising at load
    ///       (`gemma4_driver.py:1670-1690`); a port of that guard, and any
    ///       C7/C8 assertion of the same shape, needs the value that crossed
    ///       the ABI, which is this.
    /// \note Meaningless unless `has_mha_desc`. corelib passes the descriptor
    ///       by pointer and a null one is legal to attempt, so "all zeroes"
    ///       and "absent" are different states and a reader must be able to
    ///       tell them apart.
    ryzenai_corelib_flat_mha_bf16_desc mha_desc{};
    bool has_mha_desc{false};
    /// \brief `flat_mha`'s four remaining tensor operands: the rotary pair it
    ///        roped with and the two caches it attended over
    ///
    /// NOT IN `operands`, AND THAT IS DELIBERATE RATHER THAN AN OMISSION.
    /// `operands` is the row-bearing set -- the tensors `RowsAgree` and
    /// `OnGrid` are entitled to compare -- and these four have their own,
    /// different extent rules: the caches are rank 3 and pinned to
    /// `desc.max_seq`, and the rotary tables are FP32 `[positions,
    /// rope_dim/2]`. Putting them in `operands` would either weaken those
    /// checks or make them lie.
    ///
    /// RECORDED BECAUSE BOTH ARE SILENT WHEN WRONG, in exactly the way
    /// `DispatchRecord::weights` is. A layer handed the other regime's rotary
    /// pair ropes at the wrong base and "still converges to fluent text"; a
    /// layer handed the wrong cache attends over another layer's keys and
    /// values and nothing errors. Neither this fake nor corelib can tell a
    /// correct binding from an incorrect one -- both tensors are real, both
    /// are the right shape -- so the only possible move is to record the
    /// identity that crossed the ABI and let the caller join it to the tensor
    /// it created for that role (`State::tensor_creates`, or
    /// `State::tensor_windows` for a window onto one).
    ///
    /// \note Added by Task C8, which is also the first task to dispatch
    ///       attention. That is the arrangement R3 warned about -- the layer
    ///       loop extending the instrument it is measured by -- and it is
    ///       confined to RETAINING a handle: nothing here judges the value,
    ///       and no check anywhere else was relaxed to accommodate it.
    void* mha_cos{nullptr};
    void* mha_sin{nullptr};
    void* mha_k_cache{nullptr};
    void* mha_v_cache{nullptr};
    /// \brief the absolute position `flat_mha` was told these rows start at,
    ///        or -1 for a dispatch that is not attention
    ///
    /// "Q'S SHAPE IS A POSITION RANGE. Its rows are the rows this call ropes,
    /// and `position` is the absolute index of the first of them -- together
    /// they span [position, position + M), which is what the rotary and the
    /// causal mask need." It is therefore the one scalar in the pass that
    /// makes a decode step different from a prefill, and an engine that always
    /// passed 0 would rope every token as if it were the first and mask every
    /// step to a one-token context. That is finite, plausible, fluent and
    /// wrong -- and until this field existed no test could see it, which was
    /// demonstrated by a mutation that passed a literal 0 here and left the
    /// whole suite green.
    std::int64_t mha_position{-1};
    /// \brief the ssmlp descriptor this dispatch's WEIGHTS were created with
    /// \note `ryzenai_corelib_ssmlp_bf16` takes a packed weights object, not a
    ///       descriptor (corelib.h:1384-1388) -- there is no descriptor at
    ///       the dispatch to record. What is carried across instead is the
    ///       descriptor handed to `ssmlp_bf16_weights_create_*`, which is
    ///       where `activation` and `post_feedforward_layernorm` live and so
    ///       is what a C8 assertion about this block actually wants.
    /// \note Meaningless unless `has_ssmlp_desc` -- false when the dispatch
    ///       carried no weights object, or one this fake did not mint.
    ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp_desc{};
    bool has_ssmlp_desc{false};
    /// \brief the WEIGHTS OBJECT this dispatch was handed, or null for a kind
    ///        that takes none
    ///
    /// "THE RIGHT DESCRIPTOR BUILT AND THE WRONG OBJECT HANDED OVER" is a
    /// failure mode neither this fake nor corelib can detect from inside a
    /// dispatch -- both objects are real, both were packed by the same entry
    /// point, and the kernel simply multiplies by the wrong numbers. What is
    /// recorded here is the identity that crossed the ABI, which is what
    /// makes the assertion writable by the caller: joined against
    /// `MatmulWeightsCreateCall::object` (or the ssmlp / rmsnorm / ple
    /// equivalents) and against `gemma4_rai`'s WeightPlacementsForTest, a
    /// test can say "layer 7's attn_q dispatch ran with the object packed
    /// from blk.7.attn_q.weight".
    ///
    /// Null for `flat_mha`, which takes no weights object at all -- its
    /// operands are q/k/v, the caches and the rope tables
    /// (corelib.h:1384-1388 and the flat_mha section). That is an absence in
    /// the ABI, not an unrecorded field.
    ///
    /// SET FOR `rmsnorm` AND `ple` TOO, and that is what makes them
    /// identifiable at all: neither dispatch carries a descriptor, so "which
    /// layer's norm was this?" is answered only by joining this handle to
    /// `RmsNormWeightsCreateCall::object` (whose `scale` or `scale_tag` names
    /// the gamma) or to `PleCreateCall::object` (whose `desc.layer_scale`
    /// names the layer). No descriptor is duplicated onto this record for
    /// them: a second copy would be a second thing to keep in step with the
    /// first, and the join is already grounded in the object that crossed the
    /// ABI.
    ///
    /// \note C8, not C7, is where this becomes reachable from the engine:
    ///       nothing dispatches yet. It is landed now because C8's tests have
    ///       to be WRITTEN against it, and because the blind spot it closes
    ///       multiplies by 35 layers the moment the layer loop exists.
    void* weights{nullptr};
};

/// \brief one `ryzenai_corelib_stream_enable_kernel_bundle` call
struct KernelBundleRecord {
    void* stream{};
    std::string model;
    std::size_t enum_calls_before{};
};

/// \note EVERY MEMBER IS WRITTEN UNDER ONE MUTEX inside the fake, because
///       the engine packs its weights across a pool of threads
///       (`kWeightCreateConcurrency`). A reader -- i.e. a test -- runs after
///       the pool has joined and takes no lock. Before C7 nothing here was
///       called concurrently and there was no mutex at all; adding one is
///       not a style change, it is what stops eight threads pushing into the
///       same vector.
/// \note NOTHING HERE IS IN A MEANINGFUL ORDER once the pool is involved.
///       Assertions must key on a field that identifies the weight (the
///       descriptor, a norm's signature value) rather than on an index --
///       exactly as the engine assigns its results by slot rather than by
///       completion.
struct State {
    /// \brief every `..._enum_kernels` call, in order
    std::vector<EnumKernelsCall> enum_calls;
    std::vector<KernelBundleRecord> kernel_bundles;
    /// \brief report exactly the `shipped_*` kernels instead of "every M"
    /// \note What makes a plan's kernel QUERY observable: the plan filters
    ///       the reported set itself, so the only thing a test can see is
    ///       whether the set it shipped covers what the plan asked for.
    bool explicit_grid{false};
    std::vector<ShippedMatmul> shipped_matmul;
    std::vector<ShippedSsMlp> shipped_ssmlp;
    std::vector<ShippedMha> shipped_mha;
    /// \brief report exactly these paged kernels; empty reports "every M"
    std::vector<ShippedPaged> shipped_paged;
    std::vector<MatmulPadCall> matmul_pad_calls;
    std::vector<RowsPadCall> rows_pad_calls;
    std::vector<MhaPadCall> mha_pad_calls;
    std::vector<MatmulWeightsCreateCall> matmul_weights_creates;
    std::vector<SsMlpWeightsCreateCall> ssmlp_weights_creates;
    std::vector<RmsNormWeightsCreateCall> rmsnorm_weights_creates;
    std::vector<PlePackCall> ple_pack_calls;
    std::vector<PleCreateCall> ple_create_calls;
    std::vector<TensorCreateRecord> tensor_creates;
    /// \brief every `ryzenai_corelib_create_host_view` that succeeded
    std::vector<TensorCreateRecord> host_view_creates;
    std::vector<TensorWindowRecord> tensor_windows;
    std::vector<TensorWriteRecord> tensor_writes;
    std::vector<TensorReadRecord> tensor_reads;
    /// \brief every accepted dispatch, IN ABI CALL ORDER
    /// \note THE ONE VECTOR HERE WHOSE ORDER MEANS SOMETHING, and the
    ///       exception to the "nothing here is in a meaningful order" note
    ///       below. That note is about the weight CREATES, which run across
    ///       `kWeightCreateConcurrency` threads; a layer loop dispatches from
    ///       one thread, so push order under the state mutex is call order.
    ///       A Gemma 4 layer is SEVEN dispatches on a sharing layer and
    ///       `10 + kv_heads` on an owning one (11 on E2B, 12 on E4B; V's norm
    ///       runs once per KV head). A PERMUTATION of them is a different
    ///       model while a COUNT of them is not -- so C8 should assert the
    ///       sequence, not the multiset.
    std::vector<DispatchRecord> dispatches;

    /// \brief the row grid every pad helper here rounds through
    /// \note M == 1 is never rounded: corelib's flat_mha doc says decode
    ///       reports 1 because the token kernel runs a single row.
    std::int64_t pad_multiple{64};

    /// \brief what `matmul_pad_shape` adds to the caller's `k` and `n`
    /// \note Both are 0 by default, i.e. "the helper returns K and N
    ///       unchanged", which is what real corelib does for every shape this
    ///       model presents. A test sets one non-zero to reach
    ///       `MatmulRows`'s "helper changed padded K/N" rejection in
    ///       gemma4_rai_shape_plan.cpp, which is otherwise unreachable and
    ///       therefore untested.
    std::int64_t matmul_k_delta{0};
    std::int64_t matmul_n_delta{0};

    /// \brief every value a pad helper has actually returned, per helper name
    /// \note This is what lets a dispatch be checked against its OWN op
    ///       family's pad answers instead of merely against its co-operands.
    ///       Helper names are `"matmul-<n>"` (one per distinct output width,
    ///       so a Q-projection operand cannot be validated against the output
    ///       projection's grid), `"ssmlp"` and `"mha"`. Read the header's
    ///       "what it still cannot catch" list before treating a green
    ///       dispatch test as proof the extents were right.
    std::unordered_map<std::string, std::unordered_set<std::int64_t>> pad_answers_seen;

    // ---- Task C9's three fields. See the header's "what this fake still
    //      cannot catch" list: two of them make a FAILURE PATH reachable at
    //      all, and the third records an order the ABI already had. None of
    //      them changes what any other assertion in the suite observes.

    /// \brief every `ryzenai_corelib_object_release`, IN ABI CALL ORDER, by
    ///        the kind string the object was created with
    ///
    /// THE SECOND VECTOR HERE WHOSE ORDER MEANS SOMETHING, and it is the
    /// whole instrument behind C9. corelib access-violates when a packed
    /// object is released AFTER the `Stream` it was planned against -- 7 of
    /// 10 measured runs in the reference driver, up to 221 printed lines in
    /// one run. An earlier version of this note added "and the process
    /// exited 0 every time with nothing raised"; the source says nineteen of
    /// TWENTY runs -- both teardown orders together -- exited 0 and the
    /// twentieth exited 1, and does not break that down by order. Which is
    /// still enough: at 19 of 20 an exit code asserts nothing, so a test
    /// cannot wait for the fault and reads the ORDER instead.
    /// Releases run on the destroying thread and nothing else releases
    /// concurrently, so push order under the state mutex is call order.
    ///
    /// Kinds are the strings `NewObject` is called with: `"stream"`,
    /// `"tensor"`, `"window"`, `"matmul_weights"`, `"rmsnorm_weights"`,
    /// `"ple_weights"`, `"ssmlp_weights"` -- plus `"cleanup"`, which is NOT
    /// an object release but `ryzenai_corelib_cleanup`, the runtime's own
    /// teardown call. It is in this log so that "the runtime goes last of
    /// all" is one question about one ABI sequence rather than two questions
    /// about two instruments.
    ///
    /// \note NOT CLEARED BY ANYTHING BUT `Reset()`. A test that wants only
    ///       one teardown's worth takes `release_order.size()` as a mark
    ///       first and reads the slice appended after it.
    std::vector<std::string> release_order;

    /// \brief the 0-based index of the dispatch this fake REFUSES, if any
    ///
    /// WITHOUT IT THE ENGINE'S FAILURE PATH IS UNREACHABLE FROM A TEST and
    /// `poisoned()` could only ever be observed false. Counted over accepted
    /// dispatches in this run (`dispatches.size()`), so "fail the 20th" is a
    /// MID-CHAIN rejection -- the case the engine's catch exists for --
    /// rather than a refusal before anything was submitted.
    ///
    /// \note The refused dispatch records NOTHING. It never happened as far
    ///       as `dispatches` is concerned, which keeps the index stable:
    ///       setting it to 20 refuses whatever would have been the 21st
    ///       accepted call, not a moving target.
    /// \note This is an INJECTION, not a validation. It does not model any
    ///       reason corelib would reject a dispatch, and no test may read
    ///       anything into WHICH op it lands on beyond "it is not the first".
    std::optional<std::size_t> fail_dispatch_at;

    /// \brief make every accepted `tensor_read` return a quiet NaN
    /// \note The ONLY way to reach `RequireFinite` in gemma4_rai.cpp, which
    ///       is the engine's guard against an all-NaN result reaching an
    ///       argmax that then answers `<pad>` for every step -- a CLI that
    ///       prints an empty answer under a plausible token count and exits
    ///       0. Until C9 that guard had no test at all.
    /// \note Set AFTER the engine is constructed. A load performs no
    ///       `tensor_read`, but nothing here depends on that staying true.
    bool read_returns_nonfinite{false};

    /// \brief which accepted read the NaN lands on: an index into
    ///        `tensor_reads`, or every read when unset
    ///
    /// WITHOUT IT THE INJECTION IS GLOBAL AND ONLY THE FIRST `RequireFinite`
    /// CALL SITE IS EVER REACHED. gemma4_rai.cpp checks twice -- the layer
    /// loop's last live hidden row, and then the head's logits -- and with
    /// every read poisoned the first throws and the second is dead code as
    /// far as any test is concerned. Review-C9's m-1 measured exactly that:
    /// deleting the logits check left the suite green.
    ///
    /// \note NULLOPT IS THE OLD BEHAVIOUR, byte for byte: when this is unset
    ///       and `read_returns_nonfinite` is true, every accepted read
    ///       returns NaN as before. It narrows an existing injection and
    ///       adds no new one -- with `read_returns_nonfinite` false this
    ///       field does nothing at all.
    /// \note COUNTED OVER `tensor_reads`, WHICH INCLUDES REFUSED READS, and
    ///       the index is that of the read in progress at the time the value
    ///       is chosen. A test should take `tensor_reads.size()` as a mark
    ///       rather than assuming a pass's first read is index 0.
    /// \note Like `fail_dispatch_at`, this is an INJECTION and models no
    ///       reason. It proves which guard fires, never that a real device
    ///       would produce a NaN there.
    std::optional<std::size_t> nonfinite_read_index;
};

/// \brief what `tensor_read` puts in a caller's buffer
///
/// A SYNTHETIC CONSTANT, AND IT MEANS NOTHING EXCEPT "NOT ZERO". This fake
/// moves no bytes -- there is no device, and the contents it would have to
/// invent are the whole of what a real corelib computes -- so a read cannot
/// return the right answer. It used to return zeros, and that turned out to
/// hide a specific, named bug: Gemma 4 softcaps its logits on the HOST, after
/// the read-back, and `tanh(0/30)*30` is 0, so an engine that dropped the
/// softcap entirely was indistinguishable from one that applied it. A
/// mutation confirmed that: removing the call left the whole suite green.
///
/// Chosen well outside the cap's near-linear region, so a monotonic squash is
/// unmistakable: `tanh(45/30)*30` is about 27.15, not 45.
///
/// \note This is the ONLY thing a test may conclude from a read's contents.
///       "The value the engine returned is this constant, transformed by the
///       host-side arithmetic the engine claims to apply" is a fair question;
///       anything about what the NPU would have computed is not.
inline constexpr float kSyntheticReadValue = 45.0f;

State& GetState();
void Reset();
flm::corelib::CorelibApi::Resolver Resolver();
ryzenai_corelib_stream_ptr MakeStreamForTest();

/// \brief record a shipped matmul row count the way a kernel enumeration would
ryzenai_corelib_status NoteMatmulPad(ryzenai_corelib_stream_ptr stream,
                                     std::int64_t* rows,
                                     std::int64_t* k,
                                     std::int64_t* n,
                                     std::int64_t group);
/// \brief record a shipped ssmlp row count
ryzenai_corelib_status NoteSsmlpRows(ryzenai_corelib_stream_ptr stream,
                                     std::int64_t* rows,
                                     const ryzenai_corelib_ssmlp_bf16_weights_desc* desc);
/// \brief record a shipped attention row count
ryzenai_corelib_status NoteMhaRows(ryzenai_corelib_stream_ptr stream,
                                   std::int64_t* rows,
                                   const ryzenai_corelib_flat_mha_bf16_desc* desc);

/// \brief the byte count this fake's `ple_bf16_weights_pack` reports for a
///        descriptor, and the only length its `ple_bf16_weights_create`
///        accepts
/// \note A SMALL, DETERMINISTIC STAND-IN, not corelib's real packed size.
///       Real ple blobs are ~1 MB per layer and the weight-cache test writes
///       every packed weight to a file; using realistic sizes would make
///       that test write gigabytes to prove a bookkeeping property. What the
///       number has to be is (a) the same for the same descriptor and (b)
///       different for different ones, so that a cache entry written for one
///       weight cannot be bound to another. Exposed so a test can assert the
///       two-call protocol's reported size is what the second leg is then
///       given, rather than duplicating the formula.
std::size_t PlePackedSize(const ryzenai_corelib_ple_bf16_weights_desc& desc);

}  // namespace fake_corelib
