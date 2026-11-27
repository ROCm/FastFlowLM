#include "model_residency.hpp"
#include <limits>

using flm::residency::can_keep_peer;
static_assert(can_keep_peer(24576, 16384, true));
static_assert(can_keep_peer(22528, 16384, true)); // 4 GiB future cache + 2 GiB host reserve
static_assert(!can_keep_peer(22527, 16384, true));
static_assert(!can_keep_peer(0, 16384, true));    // no reliable OS reading
static_assert(!can_keep_peer(65536, 0, true));    // unknown incoming working set
static_assert(!can_keep_peer(65536, 8192, false)); // peer configuration not calibrated
static_assert(!can_keep_peer(8192, std::numeric_limits<std::uint64_t>::max(), true));
int main() {}
