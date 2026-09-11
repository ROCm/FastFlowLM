#include "fake_corelib.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {
fake_corelib::State state;
thread_local std::string current_detail;

struct FakeObject {
    std::string kind;
    ryzenai_corelib_data_type data_type{ryzenai_corelib_data_type_bf16};
    std::vector<std::int64_t> shape;
    std::size_t byte_size{};
    std::size_t window_offset{};
};

void* NewObject(std::string kind = "generic") {
    ++state.live_objects;
    auto* object = new FakeObject;
    object->kind = std::move(kind);
    return object;
}

ryzenai_corelib_status Status(std::string_view name) {
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

std::uint16_t Bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

void ObserveCreateConcurrency() {
    const int active = ++state.active_weight_creates;
    int maximum = state.maximum_active_weight_creates.load();
    while (active > maximum &&
           !state.maximum_active_weight_creates.compare_exchange_weak(maximum, active)) {}
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
        } else if constexpr (std::is_same_v<Tag, has_device_context_tag>) {
            return state.has_device_context;
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
            auto* out = std::get<4>(arguments);
            if (out) *out = nullptr;
            if (status == ryzenai_corelib_status_success && out && shape) {
                auto* object = static_cast<FakeObject*>(NewObject("window"));
                if (parent) object->data_type = static_cast<FakeObject*>(parent)->data_type;
                object->shape.assign(shape, shape + shape_len);
                object->byte_size = Elements(object->shape) * TypeBytes(object->data_type);
                object->window_offset = offset;
                *out = object;
                state.tensor_windows.push_back({parent, object->shape, offset, object});
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, tensor_get_byte_size_tag>) {
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && std::get<0>(arguments) && std::get<1>(arguments))
                *std::get<1>(arguments) = static_cast<FakeObject*>(std::get<0>(arguments))->byte_size;
            return status;
        } else if constexpr (std::is_same_v<Tag, tensor_get_data_type_tag>) {
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && std::get<0>(arguments) && std::get<1>(arguments))
                *std::get<1>(arguments) = static_cast<FakeObject*>(std::get<0>(arguments))->data_type;
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
            return status;
        } else if constexpr (std::is_same_v<Tag, tensor_read_tag>) {
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && std::get<2>(arguments)) {
                std::memset(std::get<2>(arguments), 0,
                            std::get<3>(arguments) * TypeBytes(std::get<1>(arguments)));
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, matmul_pad_shape_tag>) {
            auto* m = std::get<0>(arguments);
            auto* k = std::get<1>(arguments);
            auto* n = std::get<2>(arguments);
            const auto group = std::get<3>(arguments);
            state.matmul_pad_calls.push_back({m ? *m : -1, k ? *k : -1,
                                               n ? *n : -1, group});
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success) {
                if (m && state.pad_multiple > 0 && *m != 1)
                    *m = (*m + state.pad_multiple - 1) / state.pad_multiple * state.pad_multiple;
                if (k) *k += state.matmul_k_delta;
                if (n) *n += state.matmul_n_delta;
            }
            return status;
        } else if constexpr (std::is_same_v<Tag, ssmlp_pad_rows_tag>) {
            auto* m = std::get<0>(arguments);
            state.rows_pad_calls.push_back({"ssmlp", m ? *m : -1,
                std::get<1>(arguments), std::get<2>(arguments), std::get<3>(arguments)});
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && m && state.pad_multiple > 0 && *m != 1)
                *m = (*m + state.pad_multiple - 1) / state.pad_multiple * state.pad_multiple;
            return status;
        } else if constexpr (std::is_same_v<Tag, rmsnorm_pad_rows_tag>) {
            auto* m = std::get<0>(arguments);
            state.rows_pad_calls.push_back({"rmsnorm", m ? *m : -1,
                std::get<1>(arguments), 0, 0});
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && m && state.pad_multiple > 0 && *m != 1)
                *m = (*m + state.pad_multiple - 1) / state.pad_multiple * state.pad_multiple;
            return status;
        } else if constexpr (std::is_same_v<Tag, flat_mha_pad_rows_tag>) {
            auto* m = std::get<0>(arguments);
            auto* desc = std::get<1>(arguments);
            state.mha_pad_calls.push_back({m ? *m : -1, desc ? *desc : ryzenai_corelib_flat_mha_bf16_desc{}});
            const auto status = Status(Tag::name);
            if (status == ryzenai_corelib_status_success && m && state.pad_multiple > 0 && *m != 1)
                *m = (*m + state.pad_multiple - 1) / state.pad_multiple * state.pad_multiple;
            return status;
        } else if constexpr (std::is_same_v<Tag, matmul_weights_create_gguf_requantized_tag>) {
            const auto status = Status(Tag::name);
            auto* desc = std::get<0>(arguments);
            auto* components = std::get<1>(arguments);
            auto* out = std::get<3>(arguments);
            if (out) *out = nullptr;
            ObserveCreateConcurrency();
            if (desc && components) state.weight_creates.push_back({"matmul", desc->k, desc->n,
                desc->group_size, std::get<2>(arguments), {components->blocks}});
            if (status == ryzenai_corelib_status_success && out) *out = NewObject("matmul_weights");
            --state.active_weight_creates;
            return status;
        } else if constexpr (std::is_same_v<Tag, ssmlp_weights_create_gguf_requantized_tag>) {
            const auto status = Status(Tag::name);
            auto* desc = std::get<0>(arguments);
            auto* components = std::get<1>(arguments);
            auto* out = std::get<3>(arguments);
            if (out) *out = nullptr;
            ObserveCreateConcurrency();
            if (desc && components) {
                fake_corelib::WeightCreateRecord record{"ssmlp", desc->k, desc->n,
                    desc->group_size, std::get<2>(arguments),
                    {components->gate_blocks, components->up_blocks, components->down_blocks}};
                if (components->epsilon) record.epsilon = *static_cast<const std::uint16_t*>(components->epsilon);
                if (components->norm0) record.norm0.assign(static_cast<const std::uint16_t*>(components->norm0),
                                                           static_cast<const std::uint16_t*>(components->norm0) + desc->k);
                if (components->norm1) record.norm1.assign(static_cast<const std::uint16_t*>(components->norm1),
                                                           static_cast<const std::uint16_t*>(components->norm1) + desc->k);
                state.weight_creates.push_back(std::move(record));
            }
            if (status == ryzenai_corelib_status_success && out) *out = NewObject("ssmlp_weights");
            --state.active_weight_creates;
            return status;
        } else if constexpr (std::is_same_v<Tag, rmsnorm_weights_create_scale_tag>) {
            const auto status = Status(Tag::name);
            auto* desc = std::get<0>(arguments);
            auto* out = std::get<2>(arguments);
            if (out) *out = nullptr;
            ObserveCreateConcurrency();
            if (desc) {
                fake_corelib::WeightCreateRecord record{"rmsnorm", desc->k, 0, 0, 0, {}};
                record.epsilon = Bf16(desc->epsilon);
                if (std::get<1>(arguments))
                    record.norm0.assign(static_cast<const std::uint16_t*>(std::get<1>(arguments)),
                                        static_cast<const std::uint16_t*>(std::get<1>(arguments)) + desc->k);
                state.weight_creates.push_back(std::move(record));
            }
            if (status == ryzenai_corelib_status_success && out) *out = NewObject("rmsnorm_weights");
            --state.active_weight_creates;
            return status;
        } else if constexpr (std::is_same_v<Tag, stream_synchronize_tag>) {
            state.work_in_flight = false;
            return Status(Tag::name);
        } else if constexpr (std::is_same_v<Tag, matmul_tag> ||
                             std::is_same_v<Tag, ssmlp_tag> ||
                             std::is_same_v<Tag, rmsnorm_tag> ||
                             std::is_same_v<Tag, flat_mha_tag>) {
            const auto status = Status(Tag::name);
            if (status != ryzenai_corelib_status_success) return status;
            fake_corelib::DispatchRecord record{};
            record.kind = std::is_same_v<Tag, matmul_tag> ? "matmul" :
                          std::is_same_v<Tag, ssmlp_tag> ? "ssmlp" :
                          std::is_same_v<Tag, rmsnorm_tag> ? "rmsnorm" : "mha";
            record.stream = std::get<0>(arguments);
            if constexpr (std::is_same_v<Tag, matmul_tag>) {
                record.input = std::get<1>(arguments); record.rows = std::get<2>(arguments);
                record.output = std::get<4>(arguments);
            } else if constexpr (std::is_same_v<Tag, ssmlp_tag>) {
                record.input = std::get<1>(arguments); record.rows = std::get<3>(arguments);
                record.output = std::get<6>(arguments);
            } else if constexpr (std::is_same_v<Tag, rmsnorm_tag>) {
                record.input = std::get<1>(arguments); record.rows = std::get<2>(arguments);
                record.output = std::get<4>(arguments);
            } else {
                record.input = std::get<2>(arguments); record.rows = std::get<4>(arguments);
                record.position = std::get<5>(arguments); record.output = std::get<10>(arguments);
            }
            if (record.output && static_cast<FakeObject*>(record.output)->kind == "window")
                record.window_offset = static_cast<FakeObject*>(record.output)->window_offset;
            state.dispatches.push_back(record);
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
    state.version = {0, 3, 0};
    state.selftest_status = ryzenai_corelib_status_success;
    state.default_status = ryzenai_corelib_status_success;
    state.has_device_context = true;
    state.detail.clear();
    state.status_text = "success";
    state.missing_symbol.clear();
    state.resolution_order.clear();
    state.resolution_counts.clear();
    state.call_counts.clear();
    state.statuses.clear();
    state.lifetime_events.clear();
    state.live_objects = 0;
    state.releases = 0;
    state.cleanup_calls = 0;
    state.active_leases = 0;
    state.maximum_active_leases = 0;
    state.matmul_pad_calls.clear();
    state.rows_pad_calls.clear();
    state.mha_pad_calls.clear();
    state.pad_multiple = 64;
    state.matmul_k_delta = 0;
    state.matmul_n_delta = 0;
    state.tensor_creates.clear();
    state.tensor_windows.clear();
    state.weight_creates.clear();
    state.dispatches.clear();
    state.tensor_writes.clear();
    state.call_log.clear();
    state.active_weight_creates = 0;
    state.maximum_active_weight_creates = 0;
    state.work_in_flight = false;
    state.fail_after_submit.clear();
}

flm::corelib::CorelibApi::Resolver Resolver() {
    return [](std::string_view name) -> void* {
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

void* MakeObject() { return NewObject(); }

void EnterLease() {
    const int active = ++state.active_leases;
    int maximum = state.maximum_active_leases.load();
    while (active > maximum &&
           !state.maximum_active_leases.compare_exchange_weak(maximum, active)) {}
}

void LeaveLease() {
    --state.active_leases;
    state.lifetime_events.emplace_back("lease_leave");
}

}  // namespace fake_corelib
