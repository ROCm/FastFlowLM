// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file runtime_wait.hpp
/// \brief Backend-neutral validation for NPU command completion states.
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace flm::npu {

// Stable ERT command-state values shared by the XRT and HRX adapters.
enum class ErtCmdState : int {
    New = 1,
    Queued = 2,
    Running = 3,
    Completed = 4,
    Error = 5,
    Abort = 6,
    Submitted = 7,
    Timeout = 8,
    NoResponse = 9,
    SkError = 10,
    SkCrashed = 11,
    Max = 12,
};

template <typename State>
constexpr ErtCmdState to_ert_cmd_state(State state) noexcept {
    return static_cast<ErtCmdState>(static_cast<int>(state));
}

constexpr std::string_view ert_cmd_state_name(ErtCmdState state) noexcept {
    switch (state) {
        case ErtCmdState::New: return "new";
        case ErtCmdState::Queued: return "queued";
        case ErtCmdState::Running: return "running";
        case ErtCmdState::Completed: return "completed";
        case ErtCmdState::Error: return "error";
        case ErtCmdState::Abort: return "abort";
        case ErtCmdState::Submitted: return "submitted";
        case ErtCmdState::Timeout: return "timeout";
        case ErtCmdState::NoResponse: return "noresponse";
        case ErtCmdState::SkError: return "skerror";
        case ErtCmdState::SkCrashed: return "skcrashed";
        case ErtCmdState::Max: return "max";
        default: return "unknown";
    }
}

// Set by the HRX wait path before a non-completed return. Empty on XRT.
// require_completed_ert_state appends it so the exception carries the status
// that hrx_report also prints to stderr. One thread-local for the process:
// every engine includes this header, and the symbol-clash allowlist keeps it.
inline std::string& ert_failure_detail() {
    static thread_local std::string detail;
    return detail;
}

inline void require_completed_ert_state(
    ErtCmdState state, std::string_view operation) {
    if (state == ErtCmdState::Completed) {
        ert_failure_detail().clear();
        return;
    }
    std::string message =
        std::string(operation) + " failed: ERT command ended in state " +
        std::to_string(static_cast<int>(state)) + " (" +
        std::string(ert_cmd_state_name(state)) + ")";
    if (!ert_failure_detail().empty()) {
        message += ": ";
        message += ert_failure_detail();
        ert_failure_detail().clear();
    }
    throw std::runtime_error(std::move(message));
}

template <typename Waitable>
auto wait_for_completion(Waitable& waitable, std::string_view operation) {
    if constexpr (std::is_void_v<decltype(waitable.wait())>) {
        // xrt::runlist::wait() returns void. On XRT 2.18 and later it checks
        // every chained command and throws command_error, so a normal return
        // means state() is COMPLETED and the check below does not observe a
        // failure. Older XRT returned normally after a terminal error state,
        // which is the case this check still covers.
        waitable.wait();
        auto state = waitable.state();
        require_completed_ert_state(to_ert_cmd_state(state), operation);
        return state;
    } else {
        auto state = waitable.wait();
        require_completed_ert_state(to_ert_cmd_state(state), operation);
        return state;
    }
}

}  // namespace flm::npu
