/// \file device_load.hpp
/// \brief A device snapshot contains only aggregate counters, never process identities.
#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace device_load {

struct AdapterId {
    std::uint32_t high, low, physical;
    auto operator<=>(const AdapterId&) const = default;
    bool same_luid(const AdapterId& o) const { return high == o.high && low == o.low; }
};

std::optional<AdapterId> parse_adapter(std::wstring_view instance);

/// nodes is PhysicalAdapterCount, or nullopt where the OS or driver does not report it.
struct Candidate { bool hardware; std::optional<unsigned> nodes; };
/// Index only when exactly one hardware adapter of this kind exists and it does not report more than
/// one physical node. An unreported count is accepted; EngineUsage::linked then catches a second node.
std::optional<std::size_t> single_adapter(std::span<const Candidate> candidates);
/// Why single_adapter found nothing, for the log.
const char* no_adapter_reason(std::span<const Candidate> candidates);

struct Usage { std::optional<double> total, own; };

class EngineUsage {
public:
    explicit EngineUsage(std::uint32_t pid) : pid_(pid) {}
    void add(std::wstring_view instance, double value, bool valid);
    Usage get(AdapterId adapter) const;
    /// Rows exist for another physical node of the same adapter, i.e. a linked adapter.
    bool linked(AdapterId adapter) const;
private:
    struct Sum { double total = 0, own = 0; bool valid = true, own_valid = true, measured = false; };
    std::uint32_t pid_;
    std::map<AdapterId, std::map<std::uint32_t, Sum>> engines_;
};

struct Device {
    std::optional<std::string> name;
    Usage usage;
    std::optional<std::uint64_t> shared, dedicated;
};
struct Snapshot {
    Device gpu, npu;
    std::optional<std::uint64_t> available;
    std::optional<std::string> sampled_at;
};

/// Owned by the GET handler. Reads never wait for hardware; an empty snapshot means unavailable.
std::function<Snapshot()> make_reader();

} // namespace device_load
