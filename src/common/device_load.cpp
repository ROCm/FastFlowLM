/// \file device_load.cpp
/// \brief Aggregate Windows scheduler counters by physical engine, independent of the OS API.
#include "device_load.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace device_load {
namespace {
bool take(std::wstring_view& s, std::wstring_view prefix) {
    if (!s.starts_with(prefix)) return false;
    s.remove_prefix(prefix.size());
    return true;
}

bool number(std::wstring_view& s, unsigned base, std::uint32_t& value) {
    const auto size = s.size();
    value = 0;
    while (!s.empty()) {
        const wchar_t c = s.front();
        const unsigned digit = c >= L'0' && c <= L'9' ? c - L'0' : c >= L'a' && c <= L'f' ? c - L'a' + 10 :
                               c >= L'A' && c <= L'F' ? c - L'A' + 10 : 16;
        if (digit >= base) break;
        if (value > (std::numeric_limits<std::uint32_t>::max() - digit) / base) return false;
        value = value * base + digit;
        s.remove_prefix(1);
    }
    return s.size() != size;
}

bool adapter_prefix(std::wstring_view& s, AdapterId& id) {
    return take(s, L"luid_0x") && number(s, 16, id.high) && take(s, L"_0x") && number(s, 16, id.low) &&
           take(s, L"_phys_") && number(s, 10, id.physical);
}
} // namespace

std::optional<AdapterId> parse_adapter(std::wstring_view s) {
    AdapterId id{};
    if (adapter_prefix(s, id) && s.empty()) return id;
    return std::nullopt;
}

void EngineUsage::add(std::wstring_view s, double value, bool valid) {
    std::uint32_t pid, engine;
    AdapterId id{};
    if (!take(s, L"pid_") || !number(s, 10, pid) || !take(s, L"_") || !adapter_prefix(s, id) ||
        !take(s, L"_eng_") || !number(s, 10, engine) || !take(s, L"_engtype_") || s.empty()) return;
    auto& sum = engines_[id][engine];
    if (!valid || !std::isfinite(value) || value < 0) {
        sum.valid = false;
        if (pid == pid_) sum.own_valid = false;
    } else {
        sum.measured = true;
        sum.total += value;
        if (pid == pid_) sum.own += value;
    }
}

Usage EngineUsage::get(AdapterId id) const {
    const auto it = engines_.find(id);
    if (it == engines_.end()) return {};
    Usage result{0, 0};
    bool measured = false;
    for (const auto& [engine, sum] : it->second) {
        measured |= sum.measured;
        if (!sum.valid) result.total.reset();
        else if (result.total) result.total = std::min(100., std::max(*result.total, sum.total));
        if (!sum.own_valid) result.own.reset();
        else if (result.own) result.own = std::min(100., std::max(*result.own, sum.own));
    }
    return measured ? result : Usage{};
}

#ifndef _WIN32
std::function<Snapshot()> make_reader() { return [] { return Snapshot{}; }; }
#endif
} // namespace device_load
