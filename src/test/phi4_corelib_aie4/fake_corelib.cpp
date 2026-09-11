#include "fake_corelib.hpp"

#include <algorithm>
#include <memory>

namespace {
fake_corelib::State state;
thread_local std::string current_detail;

void GetVersion(std::uint32_t* major, std::uint32_t* minor, std::uint32_t* patch) {
    if (major) *major = state.version.major;
    if (minor) *minor = state.version.minor;
    if (patch) *patch = state.version.patch;
}

const char* StatusToString(ryzenai_corelib_status) {
    current_detail = "detail overwritten by status_to_string";
    return state.status_text.c_str();
}

const char* GetLastErrorMessage() {
    current_detail = state.detail;
    return current_detail.c_str();
}

ryzenai_corelib_status SelftestDependencies() { return state.selftest_status; }
bool HasDeviceContext() { return state.has_device_context; }
void ObjectRelease(void* object) {
    if (object) {
        delete static_cast<int*>(object);
        --state.live_objects;
        ++state.releases;
        state.lifetime_events.emplace_back("release");
    }
}
void Cleanup() {
    ++state.cleanup_calls;
    state.lifetime_events.emplace_back("cleanup");
}
void UncalledSymbol() {}

void* FunctionFor(std::string_view name) {
    if (name == "ryzenai_corelib_get_version") return reinterpret_cast<void*>(&GetVersion);
    if (name == "ryzenai_corelib_status_to_string") return reinterpret_cast<void*>(&StatusToString);
    if (name == "ryzenai_corelib_get_last_error_message") return reinterpret_cast<void*>(&GetLastErrorMessage);
    if (name == "ryzenai_corelib_selftest_dependencies") return reinterpret_cast<void*>(&SelftestDependencies);
    if (name == "ryzenai_corelib_has_device_context") return reinterpret_cast<void*>(&HasDeviceContext);
    if (name == "ryzenai_corelib_object_release") return reinterpret_cast<void*>(&ObjectRelease);
    if (name == "ryzenai_corelib_cleanup") return reinterpret_cast<void*>(&Cleanup);
#define FLM_FAKE_CORELIB_SYMBOL(member, symbol)                                    \
    if (name == #symbol) return reinterpret_cast<void*>(&UncalledSymbol);
    FLM_CORELIB_FUNCTIONS(FLM_FAKE_CORELIB_SYMBOL)
#undef FLM_FAKE_CORELIB_SYMBOL
    return nullptr;
}
}  // namespace

namespace fake_corelib {

State& GetState() { return state; }

void Reset() {
    state.version = {0, 3, 0};
    state.selftest_status = ryzenai_corelib_status_success;
    state.has_device_context = true;
    state.detail.clear();
    state.status_text = "success";
    state.missing_symbol.clear();
    state.resolution_order.clear();
    state.resolution_counts.clear();
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
