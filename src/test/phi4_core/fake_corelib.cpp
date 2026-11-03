#include "fake_corelib.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {
fake_corelib::State state;
std::recursive_mutex state_mutex;
thread_local std::string current_detail;

struct FakeStorage {
    std::size_t byte_size{};
    std::unique_ptr<std::vector<std::byte>> bytes;
};

struct FakeObject {
    std::string kind;
    /// byte the fake's packed bytes are filled with, so a cached blob can be
    /// told apart from a freshly packed one and a round trip can be checked
    unsigned char fill{};
    ryzenai_corelib_data_type data_type{ryzenai_corelib_data_type_bf16};
    std::vector<std::int64_t> shape;
    std::size_t byte_size{};
    std::size_t window_offset{};
    std::vector<std::int64_t> strides;
    std::shared_ptr<FakeStorage> storage;
};

void* NewObject(std::string kind = "generic") {
    ++state.live_objects;
    auto* object = new FakeObject;
    object->kind = std::move(kind);
    return object;
}

bool IsKind(const void* object, std::string_view kind) {
    return object && static_cast<const FakeObject*>(object)->kind == kind;
}

/// \brief the size the fake claims a packed weight has
/// \note Only has to be deterministic and descriptor-derived: corelib rejects
///       a cached slice whose length is not exactly what the descriptor packs
///       to, and this is what lets a test exercise that.
std::size_t PackedSize(std::int64_t k, std::int64_t n) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(n) / 2 + 64;
}

/// \brief a byte that identifies which source range a weight was packed from
unsigned char FillFor(const void* source) {
    return static_cast<unsigned char>(
        (reinterpret_cast<std::uintptr_t>(source) >> 4) & 0xFF);
}

ryzenai_corelib_status Status(std::string_view name) {
    if (auto pending = state.fail_counts.find(std::string(name));
        pending != state.fail_counts.end() && pending->second > 0) {
        --pending->second;
        return ryzenai_corelib_status_failure;
    }
    const auto configured = state.statuses.find(std::string(name));
    return configured == state.statuses.end() ? state.default_status
                                               : configured->second;
}

std::size_t Elements(const std::vector<std::int64_t>& shape) {
    std::size_t result = 1;
    for (const auto dimension : shape) result *= static_cast<std::size_t>(dimension);
    return result;
}

std::size_t TypeBytes(ryzenai_corelib_data_type type) {
    return RYZENAI_CORELIB_DATA_TYPE_BITS(type) / 8;
}

std::int64_t PaddedRows(std::string_view helper, std::int64_t rows) {
    const auto helpers = state.pad_row_overrides.find(std::string(helper));
    if (helpers != state.pad_row_overrides.end()) {
        const auto found = helpers->second.find(rows);
        if (found != helpers->second.end()) return found->second;
    }
    if (rows == 1 || state.pad_multiple <= 0) return rows;
    return (rows + state.pad_multiple - 1) / state.pad_multiple * state.pad_multiple;
}

std::uint16_t Bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

float FloatFromBf16(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

void EnsureStorage(FakeObject& object) {
    if (!object.storage->bytes)
        object.storage->bytes = std::make_unique<std::vector<std::byte>>(
            object.storage->byte_size, std::byte{0});
}

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

void ObserveCreateConcurrency() {
    const int active = ++state.active_weight_creates;
    int maximum = state.maximum_active_weight_creates.load();
    while (active > maximum &&
           !state.maximum_active_weight_creates.compare_exchange_weak(maximum, active)) {}
}

/// \brief stand in for the packing a real create spends its time on
/// \param state_lock the fake state mutex, held by the caller
/// \note Every fake entry point holds the state mutex for its whole body, so
///       without releasing it here the creates would serialise no matter how
///       many threads the engine used, and the concurrency could not be
///       observed -- or exercised. Real packing touches only its own mapped
///       range, which is exactly what is modelled by stepping outside.
void SimulatePackingWork(std::unique_lock<std::recursive_mutex>& state_lock) {
    state_lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    state_lock.lock();
}

#define FLM_DEFINE_FAKE_TAG(member, symbol)                                        \
    struct member##_tag {                                                          \
        static constexpr std::string_view name = #symbol;                          \
    };
FLM_CORELIB_FUNCTIONS(FLM_DEFINE_FAKE_TAG)
#undef FLM_DEFINE_FAKE_TAG

template <typename>
inline constexpr bool kAlwaysFalse = false;

template <typename Tag, typename Signature>
struct TypedFake;

template <typename Tag, typename Result, typename... Args>
struct TypedFake<Tag, Result (*)(Args...)> {
    static Result Invoke(Args... args) {
        std::unique_lock<std::recursive_mutex> state_lock(state_mutex);
        ++state.call_counts[std::string(Tag::name)];
        state.call_log.emplace_back(Tag::name);
        auto arguments = std::forward_as_tuple(args...);

        if constexpr (std::is_same_v<Tag, get_version_tag>) {
            if (std::get<0>(arguments)) *std::get<0>(arguments) = state.version.major;
            if (std::get<1>(arguments)) *std::get<1>(arguments) = state.version.minor;
            if (std::get<2>(arguments)) *std::get<2>(arguments) = state.version.patch;
            return;
        } else if constexpr (std::is_same_v<Tag, status_to_string_tag>) {
            current_detail = "detail overwritten by status_to_string";
            return state.status_text.c_str();
        } else if constexpr (std::is_same_v<Tag, get_last_error_message_tag>) {
            current_detail = state.detail;
            return current_detail.c_str();
        } else if constexpr (std::is_same_v<Tag, selftest_dependencies_tag>) {
            return state.selftest_status;
        } else if constexpr (std::is_same_v<Tag, npu_usable_tag>) {
            return state.npu_usable;
        } else if constexpr (std::is_same_v<Tag, npu_arch_tag>) {
            return state.npu_arch.c_str();
        } else if constexpr (std::is_same_v<Tag, object_release_tag>) {
            void* object = std::get<0>(arguments);
            if (object) {
                delete static_cast<FakeObject*>(object);
                --state.live_objects;
                ++state.releases;
                state.lifetime_events.emplace_back("release");
            }
            return;
        } else if constexpr (std::is_same_v<Tag, cleanup_tag>) {
            ++state.cleanup_calls;
            state.lifetime_events.emplace_back("cleanup");
            return;
        } else if constexpr (std::is_same_v<Tag, create_stream_tag>) {
            const auto status = Status(Tag::name);
            auto* out = std::get<0>(arguments);
            if (out) *out = status == ryzenai_corelib_status_success ? NewObject("stream") : nullptr;
            return status;
        } else if constexpr (std::is_same_v<Tag, stream_enable_kernel_bundle_tag>) {
            const char* model = std::get<1>(arguments);
            state.kernel_bundles.emplace_back(model ? model : "");
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, create_model_from_path_tag>) {
            const auto status = Status(Tag::name);
            state.model_names.emplace_back(std::get<0>(arguments) ? std::get<0>(arguments) : "");
            state.model_path = std::get<1>(arguments) ? std::get<1>(arguments) : "";
            auto* out = std::get<2>(arguments);
            if (out) *out = status == ryzenai_corelib_status_success ? NewObject("model") : nullptr;
            return status;
        } else if constexpr (std::is_same_v<Tag, model_create_session_tag>) {
            // A model and a session are both an object_ptr to the compiler,
            // so the fake tells them apart where corelib would.
            if (!IsKind(std::get<0>(arguments), "model")) return ryzenai_corelib_status_bad_argument;
            const auto status = Status(Tag::name);
            const auto* options = std::get<1>(arguments);
            auto* out = std::get<2>(arguments);
            state.model_max_sequence_length = options ? options->max_sequence_length : 0;
            ++state.session_creates;
            if (out) *out = status == ryzenai_corelib_status_success ? NewObject("session") : nullptr;
            return status;
        } else if constexpr (std::is_same_v<Tag, model_get_metadata_tag>) {
            if (!IsKind(std::get<0>(arguments), "session")) return ryzenai_corelib_status_bad_argument;
            auto* out = std::get<1>(arguments);
            // corelib refuses a struct ending before image_token_id.
            if (out && out->size < sizeof(*out)) return ryzenai_corelib_status_bad_argument;
            if (out) {
                constexpr std::int64_t kPage = 4096;
                const auto wanted = std::max<std::int64_t>(state.model_max_sequence_length, 1);
                *out = {sizeof(*out), "fake", 16, state.model_vocab,
                        4, (wanted + kPage - 1) / kPage * kPage, -1};
            }
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, model_run_tag>) {
            if (!IsKind(std::get<0>(arguments), "session")) return ryzenai_corelib_status_bad_argument;
            const char* method = std::get<1>(arguments);
            const auto* inputs = std::get<2>(arguments);
            const auto input_count = std::get<3>(arguments);
            auto* outputs = std::get<4>(arguments);
            const auto output_count = std::get<5>(arguments);
            if (!method || std::string_view(method) != "forward")
                return ryzenai_corelib_status_bad_argument;
            const ryzenai_corelib_model_value* ids = nullptr;
            std::size_t position = fake_corelib::kPositionContinue;
            for (std::size_t i = 0; i < input_count; ++i) {
                const auto& value = inputs[i];
                if (value.size != sizeof(value) || !value.name)
                    return ryzenai_corelib_status_bad_argument;
                const std::string_view name(value.name);
                if (name == "input_ids" && value.data_type == ryzenai_corelib_data_type_int32 &&
                    value.rank == 1 && value.data) {
                    ids = &value;
                } else if (name == "position" &&
                           value.data_type == ryzenai_corelib_data_type_int64 &&
                           value.rank == 0 && value.data) {
                    position = static_cast<std::size_t>(*static_cast<const std::int64_t*>(value.data));
                } else {
                    return ryzenai_corelib_status_bad_argument;
                }
            }
            if (!ids) return ryzenai_corelib_status_bad_argument;
            float* out_logits = nullptr;
            std::size_t logits_capacity = 0;
            for (std::size_t i = 0; i < output_count; ++i) {
                auto& value = outputs[i];
                if (value.size != sizeof(value) || !value.name ||
                    std::string_view(value.name) != "logits" || !value.data)
                    return ryzenai_corelib_status_bad_argument;
                out_logits = static_cast<float*>(value.data);
                logits_capacity = value.capacity / sizeof(float);
            }
            state.model_forwards.push_back(
                {static_cast<std::size_t>(ids->dims[0]), position, out_logits != nullptr});
            if (out_logits)
                std::fill_n(out_logits,
                            std::min<std::size_t>(logits_capacity,
                                                  static_cast<std::size_t>(state.model_vocab)),
                            0.5f);
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, model_reset_tag>) {
            if (!IsKind(std::get<0>(arguments), "session")) return ryzenai_corelib_status_bad_argument;
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, model_state_save_tag>) {
            if (!IsKind(std::get<0>(arguments), "session")) return ryzenai_corelib_status_bad_argument;
            const auto status = Status(Tag::name);
            ++state.state_saves;
            auto* out = std::get<1>(arguments);
            if (out) *out = status == ryzenai_corelib_status_success ? NewObject("snapshot") : nullptr;
            return status;
        } else if constexpr (std::is_same_v<Tag, model_state_restore_tag>) {
            if (!IsKind(std::get<0>(arguments), "session")) return ryzenai_corelib_status_bad_argument;
            ++state.state_restores;
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, create_device_tensor_tag>) {
            const auto status = Status(Tag::name);
            const auto type = std::get<0>(arguments);
            const auto* shape = std::get<1>(arguments);
            const auto shape_len = std::get<2>(arguments);
            auto* out = std::get<3>(arguments);
            if (out) *out = nullptr;
            if (status == ryzenai_corelib_status_success && out && shape) {
                auto* object = static_cast<FakeObject*>(NewObject("tensor"));
                object->data_type = type;
                object->shape.assign(shape, shape + shape_len);
                object->byte_size = Elements(object->shape) * TypeBytes(type);
                object->storage = std::make_shared<FakeStorage>();
                object->storage->byte_size = object->byte_size;
                *out = object;
                state.tensor_creates.push_back({type, object->shape, object});
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, create_tensor_window_tag>) {
            const auto status = Status(Tag::name);
            void* parent = std::get<0>(arguments);
            const auto* shape = std::get<1>(arguments);
            const auto shape_len = std::get<2>(arguments);
            const auto offset = std::get<3>(arguments);
            auto* out = std::get<6>(arguments);
            if (out) *out = nullptr;
            if (status == ryzenai_corelib_status_success && out && shape) {
                auto* object = static_cast<FakeObject*>(NewObject("window"));
                if (parent) {
                    const auto* parent_object = static_cast<FakeObject*>(parent);
                    object->data_type = parent_object->data_type;
                    object->storage = parent_object->storage;
                    object->window_offset = parent_object->window_offset + offset;
                }
                object->shape.assign(shape, shape + shape_len);
                object->byte_size = Elements(object->shape) * TypeBytes(object->data_type);
                const auto* strides = std::get<4>(arguments);
                const auto strides_len = std::get<5>(arguments);
                if (strides) object->strides.assign(strides, strides + strides_len);
                *out = object;
                state.tensor_windows.push_back({parent, object->shape, offset, object, object->strides});
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, create_host_view_tag>) {
            const auto status = Status(Tag::name);
            const auto type = std::get<0>(arguments);
            const auto* shape = std::get<1>(arguments);
            const auto shape_len = std::get<2>(arguments);
            const void* data = std::get<3>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (status == ryzenai_corelib_status_success && out && shape && data) {
                auto* object = static_cast<FakeObject*>(NewObject("host_view"));
                object->data_type = type;
                object->shape.assign(shape, shape + shape_len);
                object->byte_size = Elements(object->shape) * TypeBytes(type);
                *out = object;
                state.host_view_creates.push_back({type, object->shape, object});
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, tensor_write_tag>) {
            const auto status = Status(Tag::name);
            const auto type = std::get<1>(arguments);
            const void* source = std::get<2>(arguments);
            const auto count = std::get<3>(arguments);
            const auto offset = std::get<4>(arguments);
            bool all_zero = true;
            if (source) {
                const auto* bytes = static_cast<const unsigned char*>(source);
                all_zero = std::all_of(bytes, bytes + count * TypeBytes(type),
                                       [](unsigned char value) { return value == 0; });
            }
            state.tensor_writes.push_back({std::get<0>(arguments), type, count, offset, all_zero});
            auto* object = static_cast<FakeObject*>(std::get<0>(arguments));
            if (status == ryzenai_corelib_status_success && object && source) {
                const auto target_offset = (object->window_offset + offset) *
                                           TypeBytes(object->data_type);
                if (!all_zero || object->storage->bytes) EnsureStorage(*object);
                if (object->storage->bytes) {
                    auto* target = object->storage->bytes->data() + target_offset;
                    if (object->data_type == type) {
                        std::memcpy(target, source, count * TypeBytes(type));
                    } else if (object->data_type == ryzenai_corelib_data_type_bf16 &&
                               type == ryzenai_corelib_data_type_fp32) {
                        const auto* values = static_cast<const float*>(source);
                        for (std::size_t i = 0; i < count; ++i) {
                            const auto converted = Bf16(values[i]);
                            std::memcpy(target + i * sizeof(converted), &converted,
                                        sizeof(converted));
                        }
                    }
                }
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, tensor_read_tag>) {
            const auto status = Status(Tag::name);
            auto* object = static_cast<FakeObject*>(std::get<0>(arguments));
            const auto destination_type = std::get<1>(arguments);
            void* destination = std::get<2>(arguments);
            const auto count = std::get<3>(arguments);
            const auto offset = std::get<4>(arguments);
            if (status == ryzenai_corelib_status_success && destination) {
                std::memset(destination, 0, count * TypeBytes(destination_type));
                if (object && object->storage && object->storage->bytes) {
                    const auto source_offset = (object->window_offset + offset) *
                                               TypeBytes(object->data_type);
                    const auto* source = object->storage->bytes->data() + source_offset;
                    if (object->data_type == destination_type) {
                        std::memcpy(destination, source,
                                    count * TypeBytes(destination_type));
                    } else if (object->data_type == ryzenai_corelib_data_type_bf16 &&
                               destination_type == ryzenai_corelib_data_type_fp32) {
                        auto* values = static_cast<float*>(destination);
                        for (std::size_t i = 0; i < count; ++i) {
                            std::uint16_t encoded;
                            std::memcpy(&encoded, source + i * sizeof(encoded),
                                        sizeof(encoded));
                            values[i] = FloatFromBf16(encoded);
                        }
                    }
                }
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, matmul_enum_kernels_tag> ||
                             std::is_same_v<Tag, ssmlp_enum_kernels_tag>) {
            // corelib 0.9 reports shipped kernels instead of padding a shape.
            // A negative M is the test double for "every requested M ships
            // exactly". Argument 1 is the callback,
            // argument 2 is its context, for both of these entry points.
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && !state.pad_row_overrides.empty()) {
                // A test that overrides a padded extent gets an explicit grid
                // of Phi-4's own shapes: each prefill bucket ships at its
                // override, and the shipped M never shrinks as rows grow.
                auto* callback = std::get<1>(arguments);
                auto* ctx = std::get<2>(arguments);
                const auto shipped = [](std::string_view helper) {
                    constexpr std::int64_t buckets[] = {1, 64, 128, 256, 512, 1024, 2048, 4096};
                    std::vector<std::int64_t> rows;
                    std::int64_t floor = 0;
                    for (const auto bucket : buckets) {
                        floor = std::max(floor, PaddedRows(helper, bucket));
                        rows.push_back(floor);
                    }
                    return rows;
                };
                if (callback) {
                    if constexpr (std::is_same_v<Tag, matmul_enum_kernels_tag>) {
                        constexpr std::int64_t outputs[] = {3072, 1024, 200064};
                        for (const auto n : outputs) {
                            for (const auto m : shipped(n == 1024 ? "matmul-1024" : "matmul-3072")) {
                                const ryzenai_corelib_matmul_bf16_kernel_info info{
                                    sizeof(info), 0, m, 3072, n, 64, false, 0, 0};
                                callback(ctx, &info);
                            }
                        }
                    } else {
                        for (const auto m : shipped("ssmlp")) {
                            const ryzenai_corelib_ssmlp_bf16_kernel_info info{
                                sizeof(info), 0, "fusion_mladf_ssmlp", m, 3072, 8192, 64};
                            callback(ctx, &info);
                        }
                    }
                }
            } else if (status == ryzenai_corelib_status_success) {
                auto* callback = std::get<1>(arguments);
                auto* ctx = std::get<2>(arguments);
                if (callback) {
                    if constexpr (std::is_same_v<Tag, matmul_enum_kernels_tag>) {
                        const ryzenai_corelib_matmul_bf16_kernel_info info{
                            sizeof(info), 0, -1, 0, 0, 0, false, 0, 0};
                        callback(ctx, &info);
                    } else {
                        const ryzenai_corelib_ssmlp_bf16_kernel_info info{
                            sizeof(info), 0, "", -1, 0, 0, 0};
                        callback(ctx, &info);
                    }
                }
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, matmul_weights_create_tag> ||
                             std::is_same_v<Tag, ssmlp_weights_create_tag>) {
            // One create per op: a lone `packed` source is a cache load, and
            // anything else is a pack from model components.
            constexpr bool matmul = std::is_same_v<Tag, matmul_weights_create_tag>;
            const auto status = Status(Tag::name);
            auto* desc = std::get<0>(arguments);
            const auto* sources = std::get<1>(arguments);
            const auto count = std::get<2>(arguments);
            const auto* options = std::get<3>(arguments);
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (const auto* packed =
                    FindSource(sources, count, ryzenai_corelib_weights_role_packed)) {
                state.weight_from_file.push_back({matmul ? "matmul" : "ssmlp",
                    packed->path ? packed->path : "", packed->offset, packed->size});
                // corelib rejects a slice that is not exactly what the
                // descriptor packs to, because a truncated blob is still a
                // plausible one.
                if (count != 1 || (desc && packed->size != PackedSize(desc->k, desc->n)))
                    return ryzenai_corelib_status_failure;
                if (status == ryzenai_corelib_status_success && out) {
                    *out = NewObject(matmul ? "matmul_weights" : "ssmlp_weights");
                    static_cast<FakeObject*>(*out)->byte_size =
                        static_cast<std::size_t>(packed->size);
                }
                return status;
            }
            ObserveCreateConcurrency();
            SimulatePackingWork(state_lock);
            const std::uint32_t threads = options ? options->fast_packer_threads : 0;
            const void* first = nullptr;
            if (desc) {
                fake_corelib::WeightCreateRecord record{matmul ? "matmul" : "ssmlp", desc->k,
                    desc->n, desc->group_size, threads, {}};
                const auto* qweight = FindSource(sources, count,
                    matmul ? ryzenai_corelib_weights_role_qweight
                           : ryzenai_corelib_weights_role_gate_qweight);
                if (qweight) record.qweight_type = qweight->data_type;
                if constexpr (matmul) {
                    first = SourceData(sources, count, ryzenai_corelib_weights_role_qweight);
                    record.pointers = {first};
                } else {
                    first = SourceData(sources, count, ryzenai_corelib_weights_role_gate_qweight);
                    record.pointers = {first,
                        SourceData(sources, count, ryzenai_corelib_weights_role_up_qweight),
                        SourceData(sources, count, ryzenai_corelib_weights_role_down_qweight)};
                    const auto bf16 = [&](ryzenai_corelib_weights_role role) {
                        return static_cast<const std::uint16_t*>(SourceData(sources, count, role));
                    };
                    if (const auto* epsilon = bf16(ryzenai_corelib_weights_role_epsilon))
                        record.epsilon = *epsilon;
                    if (const auto* norm0 = bf16(ryzenai_corelib_weights_role_norm0))
                        record.norm0.assign(norm0, norm0 + desc->k);
                    if (const auto* norm1 = bf16(ryzenai_corelib_weights_role_norm1))
                        record.norm1.assign(norm1, norm1 + desc->k);
                }
                state.weight_creates.push_back(std::move(record));
            }
            if (status == ryzenai_corelib_status_success && out) {
                *out = NewObject(matmul ? "matmul_weights" : "ssmlp_weights");
                auto* object = static_cast<FakeObject*>(*out);
                object->byte_size = PackedSize(desc ? desc->k : 0, desc ? desc->n : 0);
                object->fill = FillFor(first);
            }
            --state.active_weight_creates;
            return status;
        } else if constexpr (std::is_same_v<Tag, weights_copy_data_tag>) {
            // Two-call protocol: NULL out learns the size, then the caller
            // calls again with a buffer of at least that many bytes.
            auto* weights = static_cast<FakeObject*>(std::get<0>(arguments));
            auto* out = std::get<1>(arguments);
            const auto out_size = std::get<2>(arguments);
            auto* size = std::get<3>(arguments);
            const std::size_t packed = weights ? weights->byte_size : 0;
            if (size) *size = packed;
            const auto status = Status(Tag::name);
            if (status != ryzenai_corelib_status_success) return status;
            if (out == nullptr) return status;
            if (out_size < packed) return ryzenai_corelib_status_failure;
            std::memset(out, weights ? weights->fill : 0, packed);
            return status;
        } else if constexpr (std::is_same_v<Tag, stream_synchronize_tag>) {
            state.work_in_flight = false;
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, matmul_tag> ||
                             std::is_same_v<Tag, ssmlp_tag>) {
            if (state.statuses.contains("test_observe_dispatch_concurrency")) {
                const int active = ++state.active_leases;
                int maximum = state.maximum_active_leases.load();
                while (active > maximum &&
                       !state.maximum_active_leases.compare_exchange_weak(maximum, active)) {}
                state_lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                state_lock.lock();
                --state.active_leases;
            }
            const auto status = Status(Tag::name);
            if (status != ryzenai_corelib_status_success) return status;
            if constexpr (std::is_same_v<Tag, matmul_tag>) {
                // Like corelib: a 3-D output is the cache-writing form and must
                // be a strided window [k * head, head, 1] with k >= its rows.
                const auto* output = static_cast<FakeObject*>(std::get<3>(arguments));
                if (output && output->shape.size() == 3) {
                    const auto& strides = output->strides;
                    const auto head = output->shape[2];
                    if (head <= 0 || strides.size() != 3 || strides[2] != 1 || strides[1] != head ||
                        strides[0] % head != 0 || strides[0] / head < output->shape[1]) {
                        state.detail = "a cache-writing 3-D output must be a window with strides "
                                       "[max_seq * head_size, head_size, 1]";
                        return ryzenai_corelib_status_bad_argument;
                    }
                }
            }
            fake_corelib::DispatchRecord record{};
            record.thread_id = std::this_thread::get_id();
            record.kind = std::is_same_v<Tag, matmul_tag> ? "matmul" : "ssmlp";
            record.stream = std::get<0>(arguments);
            // 0.5.0 removed the row count from every dispatch: M is the leading
            // extent of the operand that was bound. Read it back off the input
            // so the recorded value still means what the tests assert about it.
            if constexpr (std::is_same_v<Tag, matmul_tag>) {
                record.input = std::get<1>(arguments);
                record.output = std::get<3>(arguments);
            } else {
                record.input = std::get<1>(arguments);
                record.output = std::get<5>(arguments);
            }
            if (record.input) {
                const auto& shape = static_cast<FakeObject*>(record.input)->shape;
                record.rows = shape.empty() ? 0 : shape.front();
            }
            if (record.output && static_cast<FakeObject*>(record.output)->kind == "window")
                record.window_offset = static_cast<FakeObject*>(record.output)->window_offset;
            state.dispatches.push_back(record);
            if constexpr (std::is_same_v<Tag, matmul_tag>) {
                auto* output = static_cast<FakeObject*>(record.output);
                if (output && output->shape == std::vector<std::int64_t>({1, 200064})) {
                    EnsureStorage(*output);
                    const auto value = Bf16(1.0f);
                    std::memcpy(output->storage->bytes->data() +
                                    output->window_offset * TypeBytes(output->data_type),
                                &value, sizeof(value));
                }
            }
            state.work_in_flight = true;
            if (state.fail_after_submit == Tag::name) return ryzenai_corelib_status_failure;
            return ryzenai_corelib_status_success;
        } else if constexpr (std::is_same_v<Result, ryzenai_corelib_status>) {
            return Status(Tag::name);
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

template <typename Result, typename... Args>
void CallAndCollect(Result (*function)(Args...),
                    std::vector<ryzenai_corelib_status>& statuses) {
    if constexpr (std::is_same_v<Result, ryzenai_corelib_status>) {
        statuses.push_back(function(Args{}...));
    } else {
        function(Args{}...);
    }
}
}  // namespace

namespace fake_corelib {

State& GetState() { return state; }

void Reset() {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    state.version = {RYZENAI_CORELIB_VERSION_MAJOR,
                     RYZENAI_CORELIB_VERSION_MINOR,
                     RYZENAI_CORELIB_VERSION_PATCH};
    state.selftest_status = ryzenai_corelib_status_success;
    state.default_status = ryzenai_corelib_status_success;
    state.npu_usable = true;
    state.npu_arch = "AIEM";
    state.detail.clear();
    state.status_text = "success";
    state.missing_symbol.clear();
    state.resolution_order.clear();
    state.resolution_counts.clear();
    state.call_counts.clear();
    state.statuses.clear();
    state.fail_counts.clear();
    state.lifetime_events.clear();
    state.live_objects = 0;
    state.releases = 0;
    state.cleanup_calls = 0;
    state.active_leases = 0;
    state.maximum_active_leases = 0;
    state.matmul_pad_calls.clear();
    state.rows_pad_calls.clear();
    state.pad_multiple = 64;
    state.matmul_k_delta = 0;
    state.matmul_n_delta = 0;
    state.pad_row_overrides.clear();
    state.tensor_creates.clear();
    state.host_view_creates.clear();
    state.tensor_windows.clear();
    state.weight_creates.clear();
    state.weight_from_file.clear();
    state.dispatches.clear();
    state.tensor_writes.clear();
    state.call_log.clear();
    state.kernel_bundles.clear();
    state.active_weight_creates = 0;
    state.maximum_active_weight_creates = 0;
    state.work_in_flight = false;
    state.fail_after_submit.clear();
    state.model_max_sequence_length = -1;
    state.model_vocab = 8;
    state.session_creates = 0;
    state.state_saves = 0;
    state.state_restores = 0;
    state.model_forwards.clear();
    state.model_names.clear();
    state.model_path.clear();
}

flm::corelib::CorelibApi::Resolver Resolver() {
    return [](std::string_view name) -> void* {
        std::lock_guard<std::recursive_mutex> lock(state_mutex);
        state.resolution_order.emplace_back(name);
        ++state.resolution_counts[std::string(name)];
        if (name == state.missing_symbol) return nullptr;
        return FunctionFor(name);
    };
}

std::vector<ryzenai_corelib_status> CallEveryResolvedFunction(
    const flm::corelib::CorelibFunctions& functions) {
    std::vector<ryzenai_corelib_status> statuses;
#define FLM_CALL_FAKE_FUNCTION(member, symbol) CallAndCollect(functions.member, statuses);
    FLM_CORELIB_FUNCTIONS(FLM_CALL_FAKE_FUNCTION)
#undef FLM_CALL_FAKE_FUNCTION
    return statuses;
}

void* MakeObject() {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    return NewObject();
}

void EnterLease() {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    const int active = ++state.active_leases;
    int maximum = state.maximum_active_leases.load();
    while (active > maximum &&
           !state.maximum_active_leases.compare_exchange_weak(maximum, active)) {}
}

void LeaveLease() {
    std::lock_guard<std::recursive_mutex> lock(state_mutex);
    --state.active_leases;
    state.lifetime_events.emplace_back("lease_leave");
}

}  // namespace fake_corelib
