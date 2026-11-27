/// \file model_residency.hpp
/// \brief Cold-load decisions. The caller owns serialized accelerator admission.
#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include <string_view>

namespace flm::residency {
constexpr std::size_t slot_for(int limit, std::string_view device) {
    return limit == 2 && device == "gpu" ? 1 : 0;
}

// An unknown device/provider has no calibrated peak and must load alone.
constexpr const char* fallback_reason(std::uint64_t available_mib, std::uint64_t incoming_peak_mib,
                                      bool peer_calibrated) {
    if (!incoming_peak_mib || !peer_calibrated) return "uncalibrated model or settings";
    if (!available_mib) return "host headroom unavailable";
    constexpr std::uint64_t reserve_mib = 4096 + 2048; // resident slots and OS, respectively
    if (incoming_peak_mib > available_mib || available_mib - incoming_peak_mib < reserve_mib)
        return "insufficient host headroom";
    return nullptr;
}

// Only a typed allocation failure justifies evicting a healthy peer. The failing
// attempt's local frontend has already unwound before we enter the handler.
template<class Load, class Evict>
auto load_with_peer_retry(Load load, bool has_peer, Evict evict) {
    try { return load(); }
    catch (const std::bad_alloc&) {
        if (!has_peer) throw;
    }
    evict();
    return load();
}
}  // namespace flm::residency
