/// \file model_residency.hpp
/// \brief Admit an extra chat engine only with a known working set and shared host headroom.
#pragma once
#include <cstdint>

namespace flm::residency {
constexpr bool can_keep_peer(std::uint64_t available_mib, std::uint64_t incoming_peak_mib,
                             bool peer_calibrated) {
    // Keep room for the resident's default conversation slots plus the OS, separately
    // from the incoming package's load peak. Subtract only after checking for underflow.
    constexpr std::uint64_t reserve_mib = 4096 + 2048;
    return peer_calibrated && incoming_peak_mib && incoming_peak_mib <= available_mib &&
           available_mib - incoming_peak_mib >= reserve_mib;
}
}  // namespace flm::residency
