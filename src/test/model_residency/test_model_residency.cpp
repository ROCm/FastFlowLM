#include "model_residency.hpp"
#include <limits>
#include <stdexcept>

using namespace flm::residency;
static_assert(slot_for(1, "gpu") == 0 && slot_for(1, "npu") == 0);
static_assert(slot_for(2, "gpu") == 1 && slot_for(2, "npu") == 0 && slot_for(2, "") == 0);
static_assert(!fallback_reason(22528, 16384, true)); // exactly enough
static_assert(fallback_reason(22527, 16384, true) != nullptr);
static_assert(fallback_reason(0, 16384, true) != nullptr);
static_assert(fallback_reason(65536, 0, true) != nullptr);
static_assert(fallback_reason(65536, 8192, false) != nullptr);
static_assert(fallback_reason(8192, std::numeric_limits<std::uint64_t>::max(), true) != nullptr);

int main() {
    int tries = 0, evictions = 0;
    auto evict = [&] { ++evictions; };
    auto once = [&] { if (++tries == 1) throw std::bad_alloc(); return 7; };
    if (load_with_peer_retry(once, true, evict) != 7 || tries != 2 || evictions != 1) return 1;
    tries = evictions = 0;
    try { load_with_peer_retry(once, false, evict); return 2; } catch (const std::bad_alloc&) {}
    if (tries != 1 || evictions) return 3;
    tries = 0;
    bool failed = false;
    try { load_with_peer_retry([&]() -> int { ++tries; throw std::logic_error("bad package"); }, true, evict); }
    catch (const std::logic_error&) { failed = true; }
    if (!failed || tries != 1 || evictions) return 5;
    tries = 0;
    failed = false;
    try { load_with_peer_retry([&]() -> int { ++tries; throw std::bad_alloc(); }, true, evict); }
    catch (const std::bad_alloc&) { failed = true; }
    return failed && tries == 2 && evictions == 1 ? 0 : 7;
}
