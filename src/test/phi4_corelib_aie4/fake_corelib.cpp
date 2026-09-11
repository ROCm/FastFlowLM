#include "fake_corelib.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace {
fake_corelib::State state;
thread_local std::string current_detail;

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
                delete static_cast<int*>(object);
                --state.live_objects;
                ++state.releases;
                state.lifetime_events.emplace_back("release");
            }
            return;
        } else if constexpr (std::is_same_v<Tag, cleanup_tag>) {
            ++state.cleanup_calls;
            state.lifetime_events.emplace_back("cleanup");
            return;
        } else if constexpr (std::is_same_v<Result, ryzenai_corelib_status>) {
            const auto configured = state.statuses.find(std::string(Tag::name));
            return configured == state.statuses.end() ? state.default_status
                                                       : configured->second;
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

void* MakeObject() {
    ++state.live_objects;
    return new int(1);
}

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
