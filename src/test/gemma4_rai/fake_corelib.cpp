#include "fake_corelib.hpp"

#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

// See fake_corelib.hpp for why this is a separate, independent fake rather
// than a reuse of src/test/phi4_rai/fake_corelib.{hpp,cpp}, for the list of
// what it validates, and -- more importantly -- for the list of what it
// still cannot catch. This file implements every entry
// `FLM_CORELIB_FUNCTIONS` declares (corelib_api.hpp's constructor refuses to
// resolve if even one is missing).

namespace {

fake_corelib::State state;

/// \brief serializes every entry point in this file
/// \note NOT decoration. C7's engine packs its weights across a pool of
///       `kWeightCreateConcurrency` threads, so eight of these functions run
///       at once and every one of them pushes onto a `State` vector. Held for
///       the whole body of each entry point, the same way
///       src/test/phi4_rai/fake_corelib.cpp holds its own. Recursive because
///       nothing here promises one entry point never calls another.
std::recursive_mutex state_mutex;

/// \brief what `get_last_error_message` reports; set only by refusals that
///        carry corelib's own wording
std::string last_error;

/// \brief the backing allocation a tensor and all its windows share
///
/// Held by shared_ptr so that a window (and a window of a window) keeps the
/// root alive and, crucially, compares EQUAL to its siblings -- which is what
/// makes an aliasing check possible at all. See StorageOf.
struct FakeStorage {
    std::size_t byte_size{};
};

struct FakeObject {
    std::string kind;
    ryzenai_corelib_data_type data_type{ryzenai_corelib_data_type_bf16};
    std::vector<std::int64_t> shape;
    std::size_t byte_size{};
    /// \brief offset from the ROOT allocation, in elements, accumulated
    ///        through however many levels of window
    std::size_t window_offset{};
    /// \brief the strides a window was created with, empty for dense
    std::vector<std::int64_t> strides;
    std::shared_ptr<FakeStorage> storage;
    /// \brief the packed N a matmul weights object was created for
    /// \note What lets a matmul dispatch pick the SAME `"matmul-<n>"` helper
    ///       key `matmul_pad_shape_tag` used to compute its bucket answers.
    ///       Without it a dispatch has no way to know which grid its own
    ///       operands should be checked against.
    std::int64_t weight_n{-1};
    /// \brief how many bytes `weights_copy_data` reports for this object
    /// \note Zero for anything that is not a packed weight. See
    ///       PackedSizeFor for why the number is small and synthetic.
    std::size_t packed_size{0};
    /// \brief the ssmlp descriptor an ssmlp weights object was created with
    /// \note `ryzenai_corelib_ssmlp_bf16` is handed a packed weights object
    ///       and no descriptor, so this is the only place the descriptor's
    ///       fields still exist by the time the dispatch happens. Carried
    ///       onto DispatchRecord so C7/C8 can assert on them.
    ryzenai_corelib_ssmlp_bf16_weights_desc ssmlp_desc{};
    bool has_ssmlp_desc{false};
    /// \brief four bytes naming WHAT was packed into this weights object
    ///
    /// THE PACKED BYTES HAVE TO CARRY IT OR THE CACHE PATH CANNOT BE CHECKED
    /// AT ALL. `weights_copy_data` stamps this at offset 0 of the blob it
    /// hands out, so the weight cache's file contains it, and each
    /// `..._weights_create_from_file` reads it back out of the slice it was
    /// given. Without that, a cache-bound engine carries no identity anywhere
    /// -- the descriptor says "some weight of this shape" and corelib returns
    /// no components -- and `BindFromCache`'s assignment BY SLOT INDEX has
    /// nothing watching it.
    ///
    /// This also makes the fake MORE faithful, not less: real corelib's
    /// packed bytes are a function of the weight that was packed, so two
    /// distinct tensors never produce one blob. Before this the bytes were a
    /// function of the descriptor alone, which is a fidelity real corelib
    /// does not have and a blindness a test inherits.
    std::uint32_t identity_tag{0};
};

void* NewObject(std::string kind = "generic") {
    auto* object = new FakeObject;
    object->kind = std::move(kind);
    return object;
}

/// \brief the synthetic packed size for one weight kind and shape
/// \note SMALL AND SYNTHETIC ON PURPOSE -- see fake_corelib.hpp's
///       PlePackedSize. The weight cache writes every packed weight to a
///       file, and an E2B model has 272 of them; reporting realistic sizes
///       would make a bookkeeping test write gigabytes. The only properties
///       that matter are determinism and that two different descriptors
///       disagree, so a cache slice written for one weight is refused for
///       another -- which is the thing corelib really does
///       ("corelib rejects a slice that is not exactly what the descriptor
///       packs to").
std::size_t PackedSizeFor(std::uint32_t kind, std::int64_t k, std::int64_t n,
                          std::uint32_t group) {
    return static_cast<std::size_t>(64 + 11 * kind + (k % 251) + 3 * (n % 241) +
                                    7 * group);
}

std::size_t Elements(const std::vector<std::int64_t>& shape) {
    std::size_t result = 1;
    for (const auto dimension : shape) result *= static_cast<std::size_t>(dimension);
    return result;
}

std::size_t TypeBytes(ryzenai_corelib_data_type type) {
    return RYZENAI_CORELIB_DATA_TYPE_BITS(type) / 8;
}

/// \brief the fixture's four-byte tag at the head of a block stream, or 0
/// \note WHAT MAKES ONE QUANTIZED TENSOR TELLABLE FROM ANOTHER. corelib's
///       requantizing entry points take a bare `const void*`; the descriptor
///       carries only shape and group size, and Gemma 4 ships many tensors
///       that agree on both (every layer's `attn_q`, and `ffn_gate` against
///       `ffn_up` on the same layer). See gemma4_gguf_fixture.hpp's
///       BlockSignatureAt for what is actually in those four bytes and why it
///       is not a valid Q8_0 block.
std::uint32_t BlockTag(const void* blocks) {
    if (!blocks) return 0;
    std::uint32_t tag = 0;
    std::memcpy(&tag, blocks, sizeof(tag));
    return tag;
}

/// \brief the identity tag this fake stamped into a packed blob, read back
///        from the slice a `..._weights_create_from_file` was pointed at
/// \return 0 when the file cannot be opened or the slice is too short --
///         which is a MISS, not an error: every caller records it and the
///         assertion belongs to the test, exactly as on the pack leg where a
///         never-stamped payload also reads 0.
/// \note THE FAKE REALLY OPENS THE FILE. It has to: the whole point is that
///       the identity travelled through the weight cache's bytes on disk
///       rather than through a label this process kept in memory, which is
///       the same reason DispatchRecord::weights joins on the object that
///       crossed the ABI instead of on a name. Real corelib maps that slice;
///       reading four bytes of it is the cheapest honest imitation.
std::uint32_t IdentityTagInFile(const char* path, std::uint64_t offset,
                                std::uint64_t size) {
    if (!path || size < sizeof(std::uint32_t)) return 0;
    std::ifstream input(path, std::ios::binary);
    if (!input) return 0;
    input.seekg(static_cast<std::streamoff>(offset));
    std::uint32_t tag = 0;
    input.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    return input ? tag : 0;
}

/// \brief round `rows` up the way every corelib pad helper here rounds:
///        unrounded at the M=1 decode kernel, to the next multiple of
///        `State::pad_multiple` otherwise, recording the answer under
///        `helper` as it goes
/// \note This is a stand-in for corelib's OWN grid, which differs per op and
///       per shape (see corelib.h's own per-op comments). Nothing in
///       test_gemma4_shape_plan.cpp asserts an exact padded value for
///       matmul/ssmlp/flat_mha rows, so one simple consistent rule suffices
///       -- but see fake_corelib.hpp: a bug that only shows up when two ops'
///       real grids DIVERGE cannot be seen through a single shared rule.
std::int64_t PaddedRows(std::string_view helper, std::int64_t rows) {
    const auto multiple = state.pad_multiple;
    const std::int64_t result =
        (rows <= 1 || multiple <= 0) ? rows : (rows + multiple - 1) / multiple * multiple;
    state.pad_answers_seen[std::string(helper)].insert(result);
    return result;
}

/// \brief the `"matmul-<n>"` helper key an output width maps to
/// \note One key PER DISTINCT N, so a Q-projection operand is never checked
///       against the output projection's grid and vice versa. Keying every
///       matmul on one name would make OnGrid accept any matmul answer for
///       any matmul operand, which is most of what it is for.
std::string MatmulHelperFor(std::int64_t n) {
    return "matmul-" + std::to_string(n);
}

std::int64_t RowsOf(void* object) {
    if (!object) return -1;
    const auto& shape = static_cast<FakeObject*>(object)->shape;
    return shape.empty() ? -1 : shape[0];
}

std::size_t RankOf(void* object) {
    return object ? static_cast<FakeObject*>(object)->shape.size() : 0;
}

/// \brief `object`'s declared shape, empty for a null or unshaped object
/// \note What `DispatchRecord::operand_shapes` carries. See that field for
///       why the extent beyond the leading one has to be retained at all.
std::vector<std::int64_t> ShapeOf(void* object) {
    if (!object) return {};
    return static_cast<FakeObject*>(object)->shape;
}

/// \brief the underlying allocation `object` reads and writes through
/// \note Two DIFFERENT FakeObjects can share this: every
///       `create_tensor_window` call mints a fresh object, but windows over
///       the same parent (however many levels deep) share that parent's
///       `storage` shared_ptr. This is what
///       DispatchRecord::operand_storage compares -- pointer identity of
///       `operands` itself only tells two OBJECTS apart, never two VIEWS of
///       the same memory apart.
const void* StorageOf(void* object) {
    return object ? static_cast<FakeObject*>(object)->storage.get() : nullptr;
}

/// \brief every rank-2 operand among these reports the SAME row count
/// \note THIS IS NOT THE CHECK REAL CORELIB MAKES -- corelib checks each
///       operand against ITS OWN pad answer; it has no notion of "these
///       operands must agree with each other" as a rule in itself. Comparing
///       operands pairwise is necessary (a dispatch with disagreeing operands
///       cannot be consistent with any single pad answer) but NOT sufficient:
///       operands that agree with each other while being uniformly wrong sail
///       through it, and OnGrid is what is meant to narrow that -- see
///       fake_corelib.hpp for how far it actually gets. A rank != 2 operand
///       (a KV cache's [heads, seq, head_dim]) is skipped rather than
///       compared: its row-count concept is unrelated to the others'.
bool RowsAgree(std::initializer_list<void*> operands) {
    std::int64_t common = -1;
    for (void* object : operands) {
        if (!object || RankOf(object) != 2) continue;
        const auto rows = RowsOf(object);
        if (common == -1) common = rows;
        else if (common != rows) return false;
    }
    return true;
}

/// \brief `object`'s row count is a value `helper` is known to have produced
///        for SOME bucket
/// \return true (skipped, not failed) for a rank != 2 or null object --
///         nothing to check -- or for an EMPTY helper name, meaning the
///         caller could not identify a helper at all (e.g. a matmul
///         dispatched with no real weights object) and has nothing to check
///         against.
/// \return **false** for a NAMED helper this engine never queried a pad
///         answer for -- i.e. an unqueried helper is a REJECTION here, not a
///         skip.
///
/// THAT RULE IS ONLY AS GOOD AS THE PLAN'S COVERAGE, AND THE COVERAGE IS NOT
/// AUTOMATIC. An earlier version of this comment claimed "every helper name
/// in play is queried, and therefore recorded, before any dispatch can run,
/// because Gemma4ShapePlan::Build() enumerates every bucket at construction".
/// That is false and was demonstrated false by execution: `Build()`
/// enumerates every row BUCKET, but the matmul family is keyed per output
/// width (`MatmulHelperFor`), and a width `Build()` never queries has no
/// entry at all. Before this round `Build()` did not query lm_head, so C7's
/// very first dispatch -- 1 x hidden -> vocab -- came back `bad_argument`
/// from right here. A width the plan misses shows up as a FALSE REJECTION of
/// a legitimate dispatch, not as a missing check.
///
/// The rejection is kept anyway, and deliberately: a silent skip would make
/// `OnGrid` useless for exactly the ops it matters most for, and a false
/// rejection is loud, traceable to this function, and fixed by planning the
/// missing width -- which is where it belongs. `Build()` now queries
/// lm_head for that reason. `"ssmlp"` and `"mha"` are single constant keys
/// `Build()` always registers, so only the matmul family can be caught out
/// this way.
///
/// Conflating "unqueried" with the empty-name case above would hide both. An
/// equivalent fake-hardening test passed for the wrong reason on the Phi-4
/// side once already -- keep the two branches, and this distinction, intact.
bool OnGrid(std::string_view helper, void* object) {
    if (helper.empty() || !object || RankOf(object) != 2) return true;
    const auto seen = state.pad_answers_seen.find(std::string(helper));
    if (seen == state.pad_answers_seen.end()) return false;
    return seen->second.contains(RowsOf(object));
}

/// \brief `object`'s row count is one `ryzenai_corelib_ple_bf16` ships a
///        kernel for
/// \return true (skipped) for a null or rank != 2 object
///
/// THE ONE OP WITH A KNOWABLE GRID AND NO PAD HELPER TO ASK. There is no
/// `ryzenai_corelib_ple_bf16_pad_rows` in the pinned 0.5.0 header, so
/// `pad_answers_seen` can never hold a ple entry and `OnGrid` would reject
/// every ple dispatch outright. What the header does give is the table
/// itself, verbatim, in `ryzenai_corelib_ple_bf16`'s own comment: M "rounds
/// up to a row count the ELF set ships -- 1 for decode, then 64, 128, 256,
/// 512, 1024, 2048, 3072, 4096 -- and the kernel then reads and writes that
/// many rows in full. So EVERY tensor must hold the padded extent, not the
/// live one; a buffer sized to the live rows is run off the end and comes
/// back with plausible numbers."
///
/// STRICTER THAN CORELIB, DELIBERATELY. Corelib would accept a non-bucket M,
/// round it up and overrun. This refuses, so the overrun is a test failure
/// rather than a plausible one -- the same trade `OnGrid` makes for matmul.
///
/// A SECOND, INDEPENDENT COPY of the table in
/// gemma4_rai_shape_plan.cpp's `kExecutionRows`, and it stays a second copy:
/// a fake that read the plan's table could not disagree with the plan.
bool OnPleGrid(void* object) {
    if (!object || RankOf(object) != 2) return true;
    static constexpr std::int64_t kPleRows[] = {1,   64,   128,  256, 512,
                                                1024, 2048, 3072, 4096};
    const auto rows = RowsOf(object);
    for (const std::int64_t bucket : kPleRows) {
        if (rows == bucket) return true;
    }
    return false;
}

/// \brief true iff every one of `operands` that has storage at all has
///        DISTINCT storage from every other -- i.e. no two are windows
///        (however many levels deep) over the SAME backing allocation
/// \note corelib.h states this for `ssmlp_bf16`'s four tensors ("All four are
///       separate buffers; nothing is packed or interleaved"), and it is MORE
///       load-bearing on Gemma 4 than on Phi-4, not less: with
///       `post_feedforward_layernorm` set the kernel never writes the
///       `normalized` plane at all (corelib.h:1117-1128), yet still requires
///       a distinct buffer for that port. A port that is write-only in name
///       is exactly the one a reader is tempted to alias onto another.
///       Enforced HERE, at dispatch, not merely recorded for a caller to
///       notice afterward. Compares StorageOf, never the handle. A null or
///       storage-less operand is skipped rather than treated as "distinct
///       from everything", since there is nothing real to compare.
bool AllDistinctStorage(std::initializer_list<void*> operands) {
    std::vector<const void*> seen;
    for (void* object : operands) {
        const void* storage = StorageOf(object);
        if (!storage) continue;
        for (const void* other : seen) {
            if (other == storage) return false;
        }
        seen.push_back(storage);
    }
    return true;
}

/// \brief a (count, offset) element range lies inside `object`'s OWN extent
/// \note corelib.h: "The tensor's own size, which is what write/read are
///       bounded by. The underlying allocation is page-rounded and may be
///       larger." So this is checked against the declared shape, not against
///       the root allocation -- a window that reads past its own end but
///       stays inside the parent buffer is still a bug, and bounding against
///       the allocation would miss it.
bool WithinExtent(void* object, std::size_t count, std::size_t offset) {
    if (!object) return false;
    const auto* tensor = static_cast<FakeObject*>(object);
    if (tensor->shape.empty()) return true;  // an unshaped object bounds nothing
    return offset + count <= Elements(tensor->shape);
}

#define FLM_DEFINE_FAKE_TAG(member, symbol)                                       \
    struct member##_tag {                                                         \
        static constexpr std::string_view name = #symbol;                         \
    };
FLM_CORELIB_FUNCTIONS(FLM_DEFINE_FAKE_TAG)
#undef FLM_DEFINE_FAKE_TAG

// The pad helpers are gone from corelib. The branches below still name their
// tags; they are never selected, and defining the types keeps those branches
// from being ill-formed.
struct matmul_pad_shape_tag {};
struct ssmlp_pad_rows_tag {};
struct flat_mha_pad_rows_tag {};

/// \brief the source with `role` in a create's flat source array, or NULL
const ryzenai_corelib_weights_source* FindSource(const ryzenai_corelib_weights_source* sources,
                                                 std::size_t count,
                                                 ryzenai_corelib_weights_role role) {
    for (std::size_t i = 0; sources && i < count; ++i)
        if (sources[i].role == role) return &sources[i];
    return nullptr;
}

const void* SourceData(const ryzenai_corelib_weights_source* sources, std::size_t count,
                       ryzenai_corelib_weights_role role) {
    const auto* source = FindSource(sources, count, role);
    return source ? source->data : nullptr;
}

template <typename>
inline constexpr bool kAlwaysFalse = false;

template <typename Tag, typename Signature>
struct TypedFake;

template <typename Tag, typename Result, typename... Args>
struct TypedFake<Tag, Result (*)(Args...)> {
    static Result Invoke(Args... args) {
        // Held for the whole body. The engine's weight-create pool calls
        // eight of these at once and every branch below touches `state`.
        std::lock_guard<std::recursive_mutex> state_lock(state_mutex);
        auto arguments = std::forward_as_tuple(args...);

        if constexpr (std::is_same_v<Tag, get_version_tag>) {
            if (std::get<0>(arguments)) *std::get<0>(arguments) = RYZENAI_CORELIB_VERSION_MAJOR;
            if (std::get<1>(arguments)) *std::get<1>(arguments) = RYZENAI_CORELIB_VERSION_MINOR;
            if (std::get<2>(arguments)) *std::get<2>(arguments) = RYZENAI_CORELIB_VERSION_PATCH;
            return;
        } else if constexpr (std::is_same_v<Tag, status_to_string_tag>) {
            return "success";
        } else if constexpr (std::is_same_v<Tag, get_last_error_message_tag>) {
            return last_error.c_str();
        } else if constexpr (std::is_same_v<Tag, stream_get_kernels_root_tag>) {
            static const char kKernelsRoot[] = "";
            return kKernelsRoot;
        } else if constexpr (std::is_same_v<Tag, get_device_tag>) {
            static const int kFakeDevice = 0;
            return static_cast<const void*>(&kFakeDevice);
        } else if constexpr (std::is_same_v<Tag, object_release_tag>) {
            // RECORD THE KIND BEFORE THE DELETE -- the string lives in the
            // object being freed. See State::release_order for why the order
            // of these calls is the only observable this fault has.
            auto* object = static_cast<FakeObject*>(std::get<0>(arguments));
            if (object) state.release_order.push_back(object->kind);
            delete object;
            return;
        } else if constexpr (std::is_same_v<Tag, cleanup_tag>) {
            // RECORDED INTO THE SAME LOG AS THE RELEASES. `cleanup` is the
            // runtime's own teardown call -- `CorelibRuntime::ShutdownProcess`
            // makes it and then drops the runtime -- so "the runtime goes
            // last of all" becomes one question about one ABI sequence
            // rather than two questions about two instruments.
            state.release_order.push_back("cleanup");
            return;
        } else if constexpr (std::is_same_v<Tag, create_stream_tag>) {
            // 0.5.0: (prefill_pdi, token_pdi, out). The shape plan itself
            // never calls this -- MakeStreamForTest() hands back a stream
            // directly -- but corelib_api.hpp still requires the symbol to
            // resolve.
            auto* out = std::get<2>(arguments);
            if (out) *out = NewObject("stream");
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, create_device_tensor_tag>) {
            const auto type = std::get<0>(arguments);
            const auto* shape = std::get<1>(arguments);
            const auto shape_len = std::get<2>(arguments);
            auto* out = std::get<3>(arguments);
            if (out) *out = nullptr;
            if (!out || !shape) return ryzenai_corelib_status_bad_argument;
            auto* object = static_cast<FakeObject*>(NewObject("tensor"));
            object->data_type = type;
            object->shape.assign(shape, shape + shape_len);
            object->byte_size = Elements(object->shape) * TypeBytes(type);
            object->storage = std::make_shared<FakeStorage>();
            object->storage->byte_size = object->byte_size;
            *out = object;
            state.tensor_creates.push_back({type, object->shape, object});
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, create_host_view_tag>) {
            const auto type = std::get<0>(arguments);
            const auto* shape = std::get<1>(arguments);
            const auto shape_len = std::get<2>(arguments);
            const void* data = std::get<3>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (!out || !shape || !data) return ryzenai_corelib_status_bad_argument;
            auto* object = static_cast<FakeObject*>(NewObject("host_view"));
            object->data_type = type;
            object->shape.assign(shape, shape + shape_len);
            object->byte_size = Elements(object->shape) * TypeBytes(type);
            *out = object;
            state.host_view_creates.push_back({type, object->shape, object});
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, stream_enable_kernel_bundle_tag>) {
            const char* model = std::get<1>(arguments);
            state.kernel_bundles.push_back(
                {std::get<0>(arguments), model ? model : "", state.enum_calls.size()});
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, create_tensor_window_tag>) {
            void* parent = std::get<0>(arguments);
            const auto* shape = std::get<1>(arguments);
            const auto shape_len = std::get<2>(arguments);
            const auto offset = std::get<3>(arguments);
            auto* out = std::get<6>(arguments);
            if (out) *out = nullptr;
            if (!out || !shape) return ryzenai_corelib_status_bad_argument;
            ryzenai_corelib_data_type data_type = ryzenai_corelib_data_type_bf16;
            std::shared_ptr<FakeStorage> storage;
            std::size_t window_offset = offset;
            if (parent) {
                const auto* parent_object = static_cast<FakeObject*>(parent);
                data_type = parent_object->data_type;
                storage = parent_object->storage;
                window_offset = parent_object->window_offset + offset;
            }
            const std::vector<std::int64_t> window_shape(shape, shape + shape_len);
            // A WINDOW MUST FIT INSIDE THE ALLOCATION IT IS CARVED FROM.
            // Checked in ELEMENTS of the (possibly nested) parent's own
            // dtype, against the ROOT allocation's real byte size, which
            // every window in a chain shares via `storage` regardless of how
            // many levels deep it is. Gemma 4 needs this specifically: its
            // activation buffers are allocated once at the WIDER geometry and
            // windowed down per layer on the column axis, so an off-by-one in
            // that arithmetic is a live failure mode and not a hypothetical.
            const auto requested_elements = window_offset + Elements(window_shape);
            if (storage && requested_elements * TypeBytes(data_type) > storage->byte_size) {
                return ryzenai_corelib_status_bad_argument;
            }
            auto* object = static_cast<FakeObject*>(NewObject("window"));
            object->data_type = data_type;
            object->storage = storage;
            object->window_offset = window_offset;
            object->shape = window_shape;
            object->byte_size = Elements(object->shape) * TypeBytes(object->data_type);
            const auto* strides = std::get<4>(arguments);
            const auto strides_len = std::get<5>(arguments);
            if (strides) object->strides.assign(strides, strides + strides_len);
            *out = object;
            state.tensor_windows.push_back(
                {parent, object->shape, offset, object, object->strides});
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, tensor_get_byte_size_tag>) {
            auto* object = static_cast<FakeObject*>(std::get<0>(arguments));
            if (std::get<1>(arguments))
                *std::get<1>(arguments) = object ? object->byte_size : 0;
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, tensor_get_data_type_tag>) {
            auto* object = static_cast<FakeObject*>(std::get<0>(arguments));
            if (std::get<1>(arguments))
                *std::get<1>(arguments) =
                    object ? object->data_type : ryzenai_corelib_data_type_bf16;
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, tensor_write_tag>) {
            void* tensor = std::get<0>(arguments);
            const auto count = std::get<3>(arguments);
            const auto offset = std::get<4>(arguments);
            const bool accepted = WithinExtent(tensor, count, offset);
            state.tensor_writes.push_back(
                {tensor, std::get<1>(arguments), count, offset, accepted});
            // No bytes are moved: nothing in src/test/gemma4_rai reads a
            // tensor's contents back, and a fake that pretended to would be
            // claiming a fidelity it does not have. The BOUND is the part
            // that matters, and it is enforced.
            return accepted ? ryzenai_corelib_status_success
                            : ryzenai_corelib_status_bad_argument;
        } else if constexpr (std::is_same_v<Tag, tensor_read_tag>) {
            void* tensor = std::get<0>(arguments);
            const auto destination_type = std::get<1>(arguments);
            void* destination = std::get<2>(arguments);
            const auto count = std::get<3>(arguments);
            const auto offset = std::get<4>(arguments);
            const bool accepted = WithinExtent(tensor, count, offset);
            // RECORDED, INCLUDING WHEN REFUSED. See TensorReadRecord: the
            // (count, offset) pair is what "read the wrong row" is made of,
            // and it is the only part of a read this fake can honestly answer
            // for. NO DEVICE BYTES ARE MOVED -- there are none -- so an
            // ACCEPTED read fills the destination with a synthetic constant
            // instead (below), and a REFUSED one returns before writing
            // anything at all.
            state.tensor_reads.push_back(
                {tensor, destination_type, count, offset, accepted});
            if (!accepted) return ryzenai_corelib_status_bad_argument;
            // A SYNTHETIC CONSTANT, NOT ZERO. See fake_corelib.hpp's
            // kSyntheticReadValue: zeros are a fixed point of the host-side
            // logit softcap, so a read that returned them made "the engine
            // applied the cap" and "the engine dropped it" the same
            // observation. Written in the CALLER's dtype, because that is what
            // `count` is in.
            //
            // ... UNLESS A TEST ASKED FOR A NaN. See
            // State::read_returns_nonfinite: the engine's RequireFinite guard
            // is otherwise unreachable, and an unreachable guard is one
            // nothing says is still wired up.
            //
            // AND, OPTIONALLY, ONLY ON ONE READ. See
            // State::nonfinite_read_index: the engine checks TWICE, and a
            // global injection always trips the first check and never the
            // second. The record for the read in progress has just been
            // pushed, so its index is `tensor_reads.size() - 1`.
            const bool nonfinite =
                state.read_returns_nonfinite &&
                (!state.nonfinite_read_index ||
                 *state.nonfinite_read_index == state.tensor_reads.size() - 1);
            const float value = nonfinite
                                    ? std::numeric_limits<float>::quiet_NaN()
                                    : fake_corelib::kSyntheticReadValue;
            if (destination) {
                if (destination_type == ryzenai_corelib_data_type_fp32) {
                    auto* out = static_cast<float*>(destination);
                    for (std::size_t i = 0; i < count; ++i) out[i] = value;
                } else if (destination_type == ryzenai_corelib_data_type_bf16) {
                    const auto bits = std::bit_cast<std::uint32_t>(value);
                    const auto as_bf16 = static_cast<std::uint16_t>(bits >> 16);
                    auto* out = static_cast<std::uint16_t*>(destination);
                    for (std::size_t i = 0; i < count; ++i) out[i] = as_bf16;
                } else {
                    std::memset(destination, 0, count * TypeBytes(destination_type));
                }
            }
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, paged_mha_enum_kernels_tag>) {
            // A context above one page plans against this grid, not
            // flat_mha's, so an empty answer leaves every prefill bucket
            // without a kernel.
            auto fn = std::get<1>(arguments);
            if (fn && state.shipped_paged.empty()) {
                fn(std::get<2>(arguments), 0, 0, 0, -1, 0, 0, 0, 0);
            } else if (fn) {
                for (const auto& shipped : state.shipped_paged) {
                    fn(std::get<2>(arguments), 0, shipped.heads, shipped.kv_heads, shipped.m, 0,
                       shipped.head_size, shipped.page_size, 0);
                }
            }
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, matmul_pad_shape_tag>) {
            // 0.5.0: (stream, m*, k*, n*, group_size). Every padding helper
            // gained a leading stream in 0.5.0 -- recorded here so
            // "the stream was passed to every pad helper" has something to
            // check.
            auto stream = std::get<0>(arguments);
            auto* m = std::get<1>(arguments);
            auto* k = std::get<2>(arguments);
            auto* n = std::get<3>(arguments);
            const auto group = std::get<4>(arguments);
            // Recorded BEFORE any perturbation, so the record always states
            // the logical shape the caller asked about.
            state.matmul_pad_calls.push_back(
                {stream, m ? *m : -1, k ? *k : -1, n ? *n : -1, group});
            if (m) *m = PaddedRows(MatmulHelperFor(n ? *n : -1), *m);
            if (k) *k += state.matmul_k_delta;
            if (n) *n += state.matmul_n_delta;
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, matmul_weights_create_tag>) {
            // (desc, sources, count, options, out). A lone `packed` source is
            // a cache bind; anything else packs model components.
            const auto* desc = std::get<0>(arguments);
            const auto* sources = std::get<1>(arguments);
            const auto count = std::get<2>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (!desc || !out) return ryzenai_corelib_status_bad_argument;
            const auto packed = PackedSizeFor(1, desc->k, desc->n, desc->group_size);
            const auto* blob = FindSource(sources, count, ryzenai_corelib_weights_role_packed);
            const bool kFromFile = blob != nullptr;
            std::uint32_t blocks_tag = 0;
            if (kFromFile) {
                // corelib.h: the length must be exactly what this descriptor
                // packs to. A cache entry written for another weight is
                // refused here rather than bound and believed. That check is
                // blind to a swap between two weights of the SAME descriptor,
                // which is why the identity comes out of the bytes.
                const auto size = static_cast<std::size_t>(blob->size);
                if (count != 1 || size != packed) return ryzenai_corelib_status_bad_argument;
                blocks_tag = IdentityTagInFile(blob->path, blob->offset, size);
            } else {
                const void* qweight =
                    SourceData(sources, count, ryzenai_corelib_weights_role_qweight);
                if (!qweight) return ryzenai_corelib_status_bad_argument;
                blocks_tag = BlockTag(qweight);
            }
            *out = NewObject("matmul_weights");
            auto* object = static_cast<FakeObject*>(*out);
            object->weight_n = desc->n;
            object->packed_size = packed;
            static constexpr std::int64_t kBuckets[] = {
                1, 64, 128, 256, 512, 1024, 2048, 3072, 4096};
            for (const std::int64_t bucket : kBuckets)
                PaddedRows(MatmulHelperFor(desc->n), bucket);
            object->identity_tag = blocks_tag;
            state.matmul_weights_creates.push_back({*desc, kFromFile, blocks_tag, *out});
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, rmsnorm_weights_create_tag>) {
            // (desc, sources, count, options, out).
            const auto* desc = std::get<0>(arguments);
            const auto* sources = std::get<1>(arguments);
            const auto count = std::get<2>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (!desc || !out) return ryzenai_corelib_status_bad_argument;
            const auto packed = PackedSizeFor(3, desc->k, 0, 0);
            if (const auto* blob =
                    FindSource(sources, count, ryzenai_corelib_weights_role_packed)) {
                const auto size = static_cast<std::size_t>(blob->size);
                if (count != 1 || size != packed) return ryzenai_corelib_status_bad_argument;
                // WHICH GAMMA THESE BYTES WERE PACKED FROM. corelib hands no
                // components back on this leg, so `scale` stays empty and this
                // tag -- recovered from the slice itself -- is the only thing
                // that can tell one layer's Q-norm from its K-norm, whose
                // descriptors and packed lengths are identical.
                const auto tag = IdentityTagInFile(blob->path, blob->offset, size);
                *out = NewObject("rmsnorm_weights");
                static_cast<FakeObject*>(*out)->packed_size = packed;
                static_cast<FakeObject*>(*out)->identity_tag = tag;
                state.rmsnorm_weights_creates.push_back(
                    {-1, *desc, true, {}, static_cast<std::uint16_t>(tag), *out});
                return ryzenai_corelib_status_success;
            }
            const auto* scale = static_cast<const std::uint16_t*>(
                SourceData(sources, count, ryzenai_corelib_weights_role_scale));
            if (!scale) return ryzenai_corelib_status_bad_argument;
            fake_corelib::RmsNormWeightsCreateCall call{0, *desc, false, {}, 0, nullptr};
            call.scale.assign(scale, scale + static_cast<std::size_t>(desc->k));
            call.scale_tag = call.scale.empty() ? 0 : call.scale.front();
            *out = NewObject("rmsnorm_weights");
            static_cast<FakeObject*>(*out)->packed_size = packed;
            static_cast<FakeObject*>(*out)->identity_tag = call.scale_tag;
            call.object = *out;
            state.rmsnorm_weights_creates.push_back(std::move(call));
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, ple_weights_create_tag>) {
            // (desc, sources, count, options, out). A lone `packed` source is
            // a cache bind and is recorded as a create alone; model
            // components are packed AND created in this one call, so they
            // land in both records.
            const auto* desc = std::get<0>(arguments);
            const auto* sources = std::get<1>(arguments);
            const auto count = std::get<2>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (!desc || !out) return ryzenai_corelib_status_bad_argument;
            if (desc->k <= 0 || desc->n <= 0)
                return ryzenai_corelib_status_bad_argument;
            // "32 IS THE ONLY VALUE mladfple ships, and anything else is
            // refused by name" -- refused here too, so an engine that packed
            // at `group` instead of `ple_group` could not pass by accident.
            if (desc->group_size != 32) return ryzenai_corelib_status_unsupported;
            const auto size = fake_corelib::PlePackedSize(*desc);
            if (const auto* blob =
                    FindSource(sources, count, ryzenai_corelib_weights_role_packed)) {
                const auto blob_size = static_cast<std::size_t>(blob->size);
                // corelib.h: "The length must be exactly what this descriptor
                // packs to: a truncated blob is still a plausible one."
                if (count != 1 || blob_size != size) {
                    state.ple_create_calls.push_back({*desc, blob_size, false, nullptr});
                    return ryzenai_corelib_status_bad_argument;
                }
                *out = NewObject("ple_weights");
                static_cast<FakeObject*>(*out)->packed_size = size;
                state.ple_create_calls.push_back({*desc, blob_size, true, *out});
                return ryzenai_corelib_status_success;
            }
            const auto* gate = static_cast<const float*>(
                SourceData(sources, count, ryzenai_corelib_weights_role_gate));
            const auto* proj = static_cast<const float*>(
                SourceData(sources, count, ryzenai_corelib_weights_role_proj));
            const auto* post_norm = static_cast<const float*>(
                SourceData(sources, count, ryzenai_corelib_weights_role_post_norm));
            const auto* next_norm = static_cast<const float*>(
                SourceData(sources, count, ryzenai_corelib_weights_role_next_norm));
            if (!gate || !proj || !post_norm || !next_norm)
                return ryzenai_corelib_status_bad_argument;
            fake_corelib::PlePackCall call{*desc, false, true, size,
                                           0.0f,  0.0f,  0.0f, 0.0f,
                                           0.0f,  0.0f,  {},   {}};
            const auto k = static_cast<std::size_t>(desc->k);
            call.post_norm.assign(post_norm, post_norm + k);
            call.next_norm.assign(next_norm, next_norm + k);
            call.gate_first = gate[0];
            call.proj_first = proj[0];
            // THE ORIENTATION PROBES. gate is required as [k, n] row-major,
            // so its (row 1, col 0) is at flat index n; proj is required as
            // [n, k], so its (row 1, col 0) is at flat index k. Together with
            // index 1 -- (row 0, col 1) in both -- these are the two
            // positions a transpose exchanges. Index 0 and index k*n-1, the
            // only two this fake used to read, are FIXED POINTS of a
            // transpose and can never see one. See PlePackCall.
            const auto n = static_cast<std::size_t>(desc->n);
            call.gate_at_1 = gate[1];
            call.gate_at_n = gate[n];
            call.proj_at_1 = proj[1];
            call.proj_at_k = proj[k];
            // Touch the far end of both matrices so an engine that passed a
            // wrongly-sized buffer runs off it here rather than only on
            // hardware: gate is [k, n] and proj is [n, k], both k*n floats.
            const auto elements = k * n;
            volatile float sink = 0.0f;
            sink += gate[elements - 1];
            sink += proj[elements - 1];
            (void)sink;
            state.ple_pack_calls.push_back(std::move(call));
            *out = NewObject("ple_weights");
            static_cast<FakeObject*>(*out)->packed_size = size;
            state.ple_create_calls.push_back({*desc, size, true, *out});
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, matmul_enum_kernels_tag> ||
                             std::is_same_v<Tag, ssmlp_enum_kernels_tag> ||
                             std::is_same_v<Tag, flat_mha_enum_kernels_tag>) {
            auto* callback = std::get<1>(arguments);
            auto* ctx = std::get<2>(arguments);
            state.enum_calls.push_back({std::string(Tag::name), std::get<0>(arguments)});
            if (callback && state.explicit_grid) {
                if constexpr (std::is_same_v<Tag, matmul_enum_kernels_tag>) {
                    for (const auto& s : state.shipped_matmul)
                        callback(ctx, 0, s.m, s.k, s.n, s.group, false,
                                 static_cast<std::int64_t>(0), static_cast<std::int64_t>(0));
                } else if constexpr (std::is_same_v<Tag, ssmlp_enum_kernels_tag>) {
                    for (const auto& s : state.shipped_ssmlp)
                        callback(ctx, 0, s.family.c_str(), s.m, s.k, s.n, s.group);
                } else {
                    for (const auto& s : state.shipped_mha)
                        callback(ctx, 0, s.heads, s.kv_heads, s.m,
                                 static_cast<std::int64_t>(0), s.head_size, s.max_seq,
                                 static_cast<std::int64_t>(0), s.window, s.kv_shared,
                                 s.scale_one);
                }
            } else if (callback) {
                if constexpr (std::is_same_v<Tag, matmul_enum_kernels_tag>) {
                    callback(ctx, 0, static_cast<std::int64_t>(-1),
                             static_cast<std::int64_t>(0), static_cast<std::int64_t>(0),
                             static_cast<std::int64_t>(0), false,
                             static_cast<std::int64_t>(0), static_cast<std::int64_t>(0));
                } else if constexpr (std::is_same_v<Tag, ssmlp_enum_kernels_tag>) {
                    static constexpr std::int64_t kBuckets[] = {
                        1, 64, 128, 256, 512, 1024, 2048, 3072, 4096};
                    for (const std::int64_t bucket : kBuckets)
                        PaddedRows("ssmlp", bucket);
                    callback(ctx, 0, "gemma_fusion", static_cast<std::int64_t>(-1),
                             static_cast<std::int64_t>(0), static_cast<std::int64_t>(0),
                             static_cast<std::int64_t>(0));
                } else {
                    static constexpr std::int64_t kBuckets[] = {
                        1, 64, 128, 256, 512, 1024, 2048, 3072, 4096};
                    for (const std::int64_t bucket : kBuckets)
                        PaddedRows("mha", bucket);
                    callback(ctx, 0, static_cast<std::int64_t>(-1),
                             static_cast<std::int64_t>(0), static_cast<std::int64_t>(-1),
                             static_cast<std::int64_t>(0), static_cast<std::int64_t>(0),
                             static_cast<std::int64_t>(0), static_cast<std::int64_t>(0),
                             static_cast<std::int64_t>(0), false, false);
                }
            }
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, ssmlp_pad_rows_tag>) {
            // 0.5.0: (stream, m*, desc*). The WHOLE descriptor is recorded,
            // not a k/n/group_size projection of it -- see fake_corelib.hpp.
            auto stream = std::get<0>(arguments);
            auto* m = std::get<1>(arguments);
            const auto* desc = std::get<2>(arguments);
            state.rows_pad_calls.push_back(
                {stream, m ? *m : -1,
                 desc ? *desc : ryzenai_corelib_ssmlp_bf16_weights_desc{}});
            if (m) *m = PaddedRows("ssmlp", *m);
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, ssmlp_weights_create_tag>) {
            // (desc, sources, count, options, out).
            const auto* desc = std::get<0>(arguments);
            const auto* sources = std::get<1>(arguments);
            const auto count = std::get<2>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (!desc || !out) return ryzenai_corelib_status_bad_argument;
            const auto packed = PackedSizeFor(2, desc->k, desc->n, desc->group_size);
            const auto* blob = FindSource(sources, count, ryzenai_corelib_weights_role_packed);
            const bool kFromFile = blob != nullptr;
            fake_corelib::SsMlpWeightsCreateCall call{
                *desc, kFromFile, false, 0, 0, 0, 0, 0, 0, nullptr};
            if (kFromFile) {
                const auto size = static_cast<std::size_t>(blob->size);
                if (count != 1 || size != packed) return ryzenai_corelib_status_bad_argument;
                // ONLY `gate_tag`, and only because a packed blob has no
                // ports. On the cache leg there are no three block streams to
                // exchange -- corelib is handed one opaque slice -- so the
                // question is which LAYER'S ssmlp this is, not which stream
                // reached which port. `up_tag`/`down_tag` stay 0 here and the
                // port assignment stays the pack leg's business.
                call.gate_tag = IdentityTagInFile(blob->path, blob->offset, size);
            } else {
                // The two norms are the fields Gemma 4 must get right and
                // Phi-4's fake never recorded. norm0 is this layer's
                // `ffn_norm` and norm1 its `post_ffw_norm` -- BOTH this
                // layer's, unlike the silu family whose norm1 belongs to the
                // next one (gemma4_driver.py:1441-1449). One BF16 element of
                // each is enough to say WHICH tensor was passed, because the
                // fixture fills every F32 tensor with a constant signature
                // derived from its name.
                const auto data = [&](ryzenai_corelib_weights_role role) {
                    return SourceData(sources, count, role);
                };
                if (!data(ryzenai_corelib_weights_role_gate_qweight))
                    return ryzenai_corelib_status_bad_argument;
                call.has_components = true;
                const auto first = [](const void* p) -> std::uint16_t {
                    return p ? *static_cast<const std::uint16_t*>(p) : 0;
                };
                call.epsilon_bf16 = first(data(ryzenai_corelib_weights_role_epsilon));
                call.norm0_bf16 = first(data(ryzenai_corelib_weights_role_norm0));
                call.norm1_bf16 = first(data(ryzenai_corelib_weights_role_norm1));
                // AND WHICH BLOCK STREAM REACHED WHICH OF THE THREE PORTS.
                // `gate` and `up` have the same shape on every Gemma 4 layer,
                // so the descriptor cannot tell them apart and neither can a
                // length check. See SsMlpWeightsCreateCall::gate_tag.
                call.gate_tag = BlockTag(data(ryzenai_corelib_weights_role_gate_qweight));
                call.up_tag = BlockTag(data(ryzenai_corelib_weights_role_up_qweight));
                call.down_tag = BlockTag(data(ryzenai_corelib_weights_role_down_qweight));
            }
            *out = NewObject("ssmlp_weights");
            // The descriptor is kept HERE because the dispatch never sees
            // one -- see FakeObject::ssmlp_desc.
            auto* object = static_cast<FakeObject*>(*out);
            object->has_ssmlp_desc = true;
            object->ssmlp_desc = *desc;
            object->packed_size = packed;
            object->identity_tag = call.gate_tag;
            call.object = *out;
            state.ssmlp_weights_creates.push_back(call);
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, weights_copy_data_tag>) {
            // Two-call, like the real one: `out == NULL` reports the size.
            // It used to report 0 unconditionally, which made every caller
            // treat the copy as unavailable -- the weight cache's write path
            // in particular gave up silently and nothing noticed.
            auto* weights = std::get<0>(arguments);
            void* out = std::get<1>(arguments);
            const auto out_size = std::get<2>(arguments);
            auto* size = std::get<3>(arguments);
            if (!weights || !size) return ryzenai_corelib_status_bad_argument;
            const auto* object = static_cast<FakeObject*>(weights);
            const auto packed = object->packed_size;
            *size = packed;
            if (out) {
                if (out_size < packed) return ryzenai_corelib_status_bad_argument;
                std::memset(out, static_cast<int>(packed & 0xFF), packed);
                // THE BLOB NAMES WHAT WAS PACKED INTO IT, at offset 0, which
                // is what puts the identity into the weight cache's file and
                // therefore onto the cache path. See FakeObject::identity_tag.
                // Filler first, then the tag, so the tag is never overwritten.
                if (packed >= sizeof(std::uint32_t))
                    std::memcpy(out, &object->identity_tag, sizeof(std::uint32_t));
            }
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, flat_mha_pad_rows_tag>) {
            // 0.5.0: (stream, m*, desc*).
            auto stream = std::get<0>(arguments);
            auto* m = std::get<1>(arguments);
            auto* desc = std::get<2>(arguments);
            state.mha_pad_calls.push_back(
                {stream, m ? *m : -1,
                 desc ? *desc : ryzenai_corelib_flat_mha_bf16_desc{}});
            if (m) *m = PaddedRows("mha", *m);
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, matmul_tag> ||
                             std::is_same_v<Tag, ssmlp_tag> ||
                             std::is_same_v<Tag, rmsnorm_tag> ||
                             std::is_same_v<Tag, ple_tag> ||
                             std::is_same_v<Tag, flat_mha_tag> ||
                             std::is_same_v<Tag, paged_mha_tag>) {
            // THE INJECTED MID-CHAIN REFUSAL, BEFORE ANY VALIDATION AND
            // BEFORE THE RECORD. See State::fail_dispatch_at: this is the
            // only way a test can reach the engine's catch block, which is
            // where `poisoned` is set. Nothing is recorded for a refused
            // call, so the index a test sets stays fixed.
            if (state.fail_dispatch_at &&
                state.dispatches.size() == *state.fail_dispatch_at) {
                return ryzenai_corelib_status_failure;
            }
            // Which operands are row-bearing differs by op: matmul's are
            // input/output; ssmlp's are input/residual/skip_sum/normalized
            // (its weights are not a row-shaped tensor); rmsnorm's are
            // input/output; ple's are x/ple/out/norm_out; flat_mha's are
            // q/k/out (cos/sin/caches have their own, deliberately different,
            // extent rules and are not checked here).
            //
            // RowsAgree applies to every op, for a reason that is PARTIAL --
            // see RowsAgree and OnGrid above, and the "what it still cannot
            // catch" list in fake_corelib.hpp. The rest is per-op, because
            // corelib.h's requirements are: matmul/ssmlp/mha get OnGrid,
            // ple gets OnPleGrid instead (it has no pad helper to have
            // produced an answer), and rmsnorm gets NEITHER. ssmlp and ple
            // each get a distinctness rule their own section states; matmul,
            // mha and rmsnorm have none.
            std::vector<void*> operands;
            // Declared before the branches so each can record the descriptor
            // it was actually handed -- see DispatchRecord::mha_desc for why
            // the plan's own copy is not a substitute for it.
            fake_corelib::DispatchRecord record{};
            if constexpr (std::is_same_v<Tag, matmul_tag>) {
                void* input = std::get<1>(arguments);
                void* weights = std::get<2>(arguments);
                void* output = std::get<3>(arguments);
                operands = {input, output};
                // THE OBJECT, NOT JUST ITS SHAPE. See DispatchRecord::weights.
                record.weights = weights;
                const auto weight_n =
                    weights ? static_cast<FakeObject*>(weights)->weight_n : -1;
                // No weights object, or one created without an N: no helper
                // can be identified, so OnGrid is skipped rather than guessed
                // at. RowsAgree still applies.
                const std::string helper =
                    weight_n < 0 ? std::string() : MatmulHelperFor(weight_n);
                if (!RowsAgree({input, output}) || !OnGrid(helper, input) ||
                    !OnGrid(helper, output))
                    return ryzenai_corelib_status_bad_argument;
                // Like corelib: a 3-D output is the cache-writing form and
                // must be a strided window [k * head, head, 1] with k >= its
                // rows. A dense window over the whole cache is refused too.
                if (output && RankOf(output) == 3) {
                    const auto* object = static_cast<FakeObject*>(output);
                    const auto& strides = object->strides;
                    const auto head = object->shape[2];
                    if (head <= 0 || strides.size() != 3 || strides[2] != 1 || strides[1] != head ||
                        strides[0] % head != 0 || strides[0] / head < object->shape[1]) {
                        last_error = "a cache-writing 3-D output must be a window with "
                                     "strides [max_seq * head_size, head_size, 1]";
                        return ryzenai_corelib_status_bad_argument;
                    }
                }
            } else if constexpr (std::is_same_v<Tag, ssmlp_tag>) {
                void* input = std::get<1>(arguments);
                void* residual = std::get<2>(arguments);
                void* weights = std::get<3>(arguments);
                void* skip_sum = std::get<4>(arguments);
                void* normalized = std::get<5>(arguments);
                operands = {input, residual, skip_sum, normalized};
                // THE OBJECT, NOT JUST ITS SHAPE. See DispatchRecord::weights.
                record.weights = weights;
                // The descriptor travelled with the WEIGHTS, not with this
                // call -- corelib.h:1384-1388 passes no descriptor here.
                if (weights) {
                    const auto* object = static_cast<FakeObject*>(weights);
                    record.has_ssmlp_desc = object->has_ssmlp_desc;
                    record.ssmlp_desc = object->ssmlp_desc;
                }
                if (!RowsAgree({input, residual, skip_sum, normalized}) ||
                    !OnGrid("ssmlp", input) || !OnGrid("ssmlp", residual) ||
                    !OnGrid("ssmlp", skip_sum) || !OnGrid("ssmlp", normalized) ||
                    !AllDistinctStorage({input, residual, skip_sum, normalized}))
                    return ryzenai_corelib_status_bad_argument;
            } else if constexpr (std::is_same_v<Tag, rmsnorm_tag>) {
                // (stream, input, weights, output).
                void* input = std::get<1>(arguments);
                void* weights = std::get<2>(arguments);
                void* output = std::get<3>(arguments);
                operands = {input, output};
                record.weights = weights;
                // NO DISTINCTNESS RULE, AND THAT IS NOT AN OVERSIGHT.
                // corelib.h: input and output "may be the SAME tensor FOR
                // SMALL M -- the kernel reads a row before it writes it, so
                // normalizing in place is safe there and saves a buffer". A
                // rule here would be a false rejection of a legal decode
                // dispatch. The header also says in-place CORRUPTS at
                // M = 1024 / 2048 at k=128 even where an EXACT kernel ships,
                // which is a property of the width and the row count that
                // nothing host-side can distinguish from the safe case --
                // so it is not checkable here at all, by this fake or any
                // other. Hardware, or a separate output tensor.
                //
                // NO GRID CHECK EITHER. `rmsnorm_bf16_pad_rows` is "an
                // IDENTITY FUNCTION today and does not report the covering
                // extent", and it is not even bound in
                // FLM_CORELIB_FUNCTIONS, so there is no answer set to check
                // a row count against and no honest one to invent. What the
                // header warns of -- the smallest covering kernel writing
                // PAST the live rows -- is an allocation-extent question, and
                // the buffer a dispatch is handed here is already whatever
                // the caller allocated. Real hardware is the check.
                if (!RowsAgree({input, output}))
                    return ryzenai_corelib_status_bad_argument;
            } else if constexpr (std::is_same_v<Tag, ple_tag>) {
                // (stream, x, ple, weights, out, norm_out).
                void* x = std::get<1>(arguments);
                void* slice = std::get<2>(arguments);
                void* weights = std::get<3>(arguments);
                void* out = std::get<4>(arguments);
                void* norm_out = std::get<5>(arguments);
                operands = {x, slice, out, norm_out};
                record.weights = weights;
                // corelib.h: "`out` and `norm_out` must be distinct buffers,
                // from each other and from `x`." `ple` -- the per-layer input
                // slice -- is NOT in that set and is not added to it; the
                // header does not put it there and inventing a rule corelib
                // does not have is a false rejection waiting for C8.
                if (!RowsAgree({x, slice, out, norm_out}) || !OnPleGrid(x) ||
                    !AllDistinctStorage({x, out, norm_out}))
                    return ryzenai_corelib_status_bad_argument;
            } else {
                const auto* desc = std::get<1>(arguments);
                void* q = std::get<2>(arguments);
                void* k = std::get<3>(arguments);
                void* out = std::get<9>(arguments);
                operands = {q, k, out};
                // THE FIELD C7/C8 MUST BE ABLE TO ASSERT ON. Nothing here
                // judges it -- neither this fake nor corelib can tell an
                // owning dispatch handed a kvshare descriptor from a correct
                // one -- but recording it is what makes the reference
                // driver's load-time guard (gemma4_driver.py:1670-1690)
                // portable to a test.
                if constexpr (std::is_same_v<Tag, flat_mha_tag>) {
                    record.has_mha_desc = desc != nullptr;
                    if (desc) record.mha_desc = *desc;
                }
                // THE ROTARY PAIR AND THE TWO CACHES. Not row-bearing, so not
                // in `operands` and not checked here -- retained so that "this
                // layer roped with the sliding table and attended over cache
                // 13" is a question a caller can ask at all. See
                // DispatchRecord::mha_cos.
                record.mha_position = std::get<4>(arguments);
                record.mha_cos = std::get<5>(arguments);
                record.mha_sin = std::get<6>(arguments);
                record.mha_k_cache = std::get<7>(arguments);
                record.mha_v_cache = std::get<8>(arguments);
                // Like corelib: a rotary table that is not a host view is refused.
                for (void* table : {record.mha_cos, record.mha_sin}) {
                    if (!table || static_cast<FakeObject*>(table)->kind != "host_view")
                        return ryzenai_corelib_status_bad_argument;
                }
                // paged_mha has no pad helper, so there is no answer set to
                // put its rows on.
                if (!RowsAgree({q, k, out}))
                    return ryzenai_corelib_status_bad_argument;
                if constexpr (std::is_same_v<Tag, flat_mha_tag>) {
                    if (!OnGrid("mha", q) || !OnGrid("mha", k) || !OnGrid("mha", out))
                        return ryzenai_corelib_status_bad_argument;
                }
            }
            record.kind = std::is_same_v<Tag, matmul_tag>      ? "matmul"
                          : std::is_same_v<Tag, ssmlp_tag>     ? "ssmlp"
                          : std::is_same_v<Tag, rmsnorm_tag>   ? "rmsnorm"
                          : std::is_same_v<Tag, ple_tag>       ? "ple"
                          : std::is_same_v<Tag, paged_mha_tag> ? "paged_mha"
                                                               : "mha";
            record.stream = std::get<0>(arguments);
            record.operands = operands;
            record.operand_storage.reserve(operands.size());
            record.operand_shapes.reserve(operands.size());
            for (void* operand : operands) {
                record.operand_storage.push_back(StorageOf(operand));
                // THE WIDTH, NOT ONLY THE ROWS. See
                // DispatchRecord::operand_shapes: everything else this record
                // holds is invariant under binding the other geometry's
                // windows.
                record.operand_shapes.push_back(ShapeOf(operand));
            }
            // 0.5.0 removed the row count from every dispatch: M is the
            // leading extent of the operand that was bound.
            record.rows = operands.empty() ? -1 : RowsOf(operands.front());
            state.dispatches.push_back(std::move(record));
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Tag, stream_synchronize_tag>) {
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Result, ryzenai_corelib_status>) {
            return ryzenai_corelib_status_success;
        } else {
            static_assert(kAlwaysFalse<Tag>, "unhandled fake corelib ABI result");
        }
    }
};

#define FLM_ASSERT_FAKE_ABI(member, symbol)                                        \
    static_assert(std::is_same_v<                                                 \
                  decltype(&TypedFake<member##_tag, decltype(&::symbol)>::Invoke), \
                  decltype(&::symbol)>);
FLM_CORELIB_FUNCTIONS(FLM_ASSERT_FAKE_ABI)
#undef FLM_ASSERT_FAKE_ABI

void* FunctionFor(std::string_view name) {
#define FLM_MAP_FAKE_FUNCTION(member, symbol)                                      \
    if (name == #symbol) {                                                         \
        return reinterpret_cast<void*>(                                            \
            &TypedFake<member##_tag, decltype(&::symbol)>::Invoke);                \
    }
    FLM_CORELIB_FUNCTIONS(FLM_MAP_FAKE_FUNCTION)
#undef FLM_MAP_FAKE_FUNCTION
    return nullptr;
}

}  // namespace

namespace fake_corelib {

State& GetState() { return state; }

std::size_t PlePackedSize(const ryzenai_corelib_ple_bf16_weights_desc& desc) {
    return PackedSizeFor(4, desc.k, desc.n, desc.group_size);
}

ryzenai_corelib_status NoteMatmulPad(ryzenai_corelib_stream_ptr stream,
                                     std::int64_t* rows,
                                     std::int64_t* k,
                                     std::int64_t* n,
                                     std::int64_t group) {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    state.matmul_pad_calls.push_back(
        {stream, rows ? *rows : -1, k ? *k : -1, n ? *n : -1,
         static_cast<std::uint32_t>(group)});
    if (rows) *rows = PaddedRows(MatmulHelperFor(n ? *n : -1), *rows);
    if (k) *k += state.matmul_k_delta;
    if (n) *n += state.matmul_n_delta;
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status NoteSsmlpRows(
    ryzenai_corelib_stream_ptr stream, std::int64_t* rows,
    const ryzenai_corelib_ssmlp_bf16_weights_desc* desc) {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    state.rows_pad_calls.push_back(
        {stream, rows ? *rows : -1,
         desc ? *desc : ryzenai_corelib_ssmlp_bf16_weights_desc{}});
    if (rows) *rows = PaddedRows("ssmlp", *rows);
    return ryzenai_corelib_status_success;
}

ryzenai_corelib_status NoteMhaRows(ryzenai_corelib_stream_ptr stream,
                                   std::int64_t* rows,
                                   const ryzenai_corelib_flat_mha_bf16_desc* desc) {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    state.mha_pad_calls.push_back(
        {stream, rows ? *rows : -1,
         desc ? *desc : ryzenai_corelib_flat_mha_bf16_desc{}});
    if (rows) *rows = PaddedRows("mha", *rows);
    return ryzenai_corelib_status_success;
}

void Reset() {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    state.enum_calls.clear();
    state.kernel_bundles.clear();
    state.explicit_grid = false;
    state.shipped_matmul.clear();
    state.shipped_ssmlp.clear();
    state.shipped_mha.clear();
    state.shipped_paged.clear();
    state.matmul_pad_calls.clear();
    state.rows_pad_calls.clear();
    state.mha_pad_calls.clear();
    state.matmul_weights_creates.clear();
    state.ssmlp_weights_creates.clear();
    state.rmsnorm_weights_creates.clear();
    state.ple_pack_calls.clear();
    state.ple_create_calls.clear();
    state.tensor_creates.clear();
    state.host_view_creates.clear();
    state.tensor_windows.clear();
    state.tensor_writes.clear();
    state.tensor_reads.clear();
    state.dispatches.clear();
    state.pad_multiple = 64;
    state.matmul_k_delta = 0;
    state.matmul_n_delta = 0;
    state.pad_answers_seen.clear();
    state.release_order.clear();
    state.fail_dispatch_at.reset();
    state.read_returns_nonfinite = false;
    state.nonfinite_read_index.reset();
    last_error.clear();
}

flm::corelib::CorelibApi::Resolver Resolver() {
    return [](std::string_view name) -> void* { return FunctionFor(name); };
}

ryzenai_corelib_stream_ptr MakeStreamForTest() {
    return static_cast<ryzenai_corelib_stream_ptr>(NewObject("stream"));
}

}  // namespace fake_corelib
