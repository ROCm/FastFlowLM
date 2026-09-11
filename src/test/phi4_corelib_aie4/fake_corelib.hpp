#pragma once

#include "corelib/corelib_api.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fake_corelib {

struct State {
    flm::corelib::CorelibVersion version{0, 3, 0};
    ryzenai_corelib_status selftest_status{ryzenai_corelib_status_success};
    ryzenai_corelib_status default_status{ryzenai_corelib_status_success};
    bool has_device_context{true};
    std::string detail;
    std::string status_text{"success"};
    std::string missing_symbol;
    std::vector<std::string> resolution_order;
    std::unordered_map<std::string, int> resolution_counts;
    std::unordered_map<std::string, int> call_counts;
    std::unordered_map<std::string, ryzenai_corelib_status> statuses;
    std::vector<std::string> lifetime_events;
    std::atomic<int> live_objects{0};
    std::atomic<int> releases{0};
    std::atomic<int> cleanup_calls{0};
    std::atomic<int> active_leases{0};
    std::atomic<int> maximum_active_leases{0};
};

State& GetState();
void Reset();
flm::corelib::CorelibApi::Resolver Resolver();
std::vector<ryzenai_corelib_status> CallEveryResolvedFunction(
    const flm::corelib::CorelibFunctions& functions);
void* MakeObject();
void EnterLease();
void LeaveLease();

}  // namespace fake_corelib
