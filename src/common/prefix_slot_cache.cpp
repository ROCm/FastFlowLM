/// \file prefix_slot_cache.cpp
/// \brief see prefix_slot_cache.hpp
#include "prefix_slot_cache.hpp"

#include <cstring>

#include "utils/debug_utils.hpp"

namespace {
/// defaults of --prefix-cache-mb and --prefix-slots
constexpr size_t kDefaultPrefixCacheMiB = 4096;
constexpr size_t kDefaultPrefixSlots = 8;

size_t lcp_len(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

bool has_prefix(const std::vector<int>& longer, const std::vector<int>& shorter) {
    return shorter.size() <= longer.size() && std::equal(shorter.begin(), shorter.end(), longer.begin());
}
} // namespace

uint64_t prefix_hash_bytes(const void* data, size_t bytes, uint64_t h) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    for (; i < bytes; i++) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

size_t PrefixSlotCache::budget_bytes() const {
    const size_t mib = cache_mb_ > 0 ? static_cast<size_t>(cache_mb_) : kDefaultPrefixCacheMiB;
    return mib << 20;
}

size_t PrefixSlotCache::slot_limit() const {
    if (slots_ == 0) return 0; // --prefix-slots 0: off
    const size_t requested = slots_ > 0 ? static_cast<size_t>(slots_) : kDefaultPrefixSlots;
    return std::min(requested, static_cast<size_t>(kMaxSlots));
}

size_t PrefixSlotCache::count_image_runs(const std::vector<int>& tokens, size_t len) const {
    if (image_token_id < 0) return 0;
    size_t runs = 0;
    for (size_t i = 0; i < len; i++) {
        if (tokens[i] == image_token_id && (i == 0 || tokens[i - 1] != image_token_id)) {
            runs++;
        }
    }
    return runs;
}

void PrefixSlotCache::attach(void* engine, const flm_state_api* api) {
    forget_all();
    engine_ = api ? engine : nullptr;
    api_ = api;
    announced_ = false;
}

void PrefixSlotCache::forget_all() {
    slots_table_.clear(); // the states are ours: their memory goes with them
    live_tokens_.clear();
}

int PrefixSlotCache::end_kv_from(const std::vector<int>& tokens, size_t boundary) const {
    if (boundary == 0) return 0;
    for (const auto& slot : slots_table_) {
        if (slot.kv_from() == 0 &&slot.tokens.size() == boundary && has_prefix(tokens, slot.tokens)) {
            return static_cast<int>(boundary);
        }
    }
    return 0;
}

PrefixSlotCache::resume_t PrefixSlotCache::resume(const std::vector<int>& full_tokens, const std::vector<uint64_t>& image_hashes,
                                                  std::vector<int>& matched_tokens) {
    resume_t result;
    if (!engine_) return result;
    int best = -1;
    int best_ancestor = -1;  // slot that puts the head under `best` back, when the engine no longer holds it
    size_t match_len = 0;
    for (size_t i = 0; i < slots_table_.size(); i++) {
        const slot_t& slot = slots_table_[i];
        const size_t len = slot.tokens.size();
        // strictly shorter than the prompt: at least one token must be processed to get logits
        if (len == 0 || len >= full_tokens.size() || len <= match_len) continue;
        if (!has_prefix(full_tokens, slot.tokens)) continue;
        if (slot.image_hashes.size() > image_hashes.size() ||
            !std::equal(slot.image_hashes.begin(), slot.image_hashes.end(), image_hashes.begin())) continue;
        int ancestor = -1;
        const size_t head = static_cast<size_t>(slot.kv_from());
        if (head > 0 && lcp_len(live_tokens_, slot.tokens) < head) {
            // the rows before kv_from are not in the engine: a self-contained slot must supply them
            for (size_t j = 0; j < slots_table_.size(); j++) {
                const slot_t& other = slots_table_[j];
                if (other.kv_from() == 0 &&other.tokens.size() >= head &&
                    std::equal(slot.tokens.begin(), slot.tokens.begin() + head, other.tokens.begin())) {
                    ancestor = static_cast<int>(j);
                    break;
                }
            }
            if (ancestor < 0) continue;
        }
        best = static_cast<int>(i);
        best_ancestor = ancestor;
        match_len = len;
    }
    if (best < 0) {
        return result;
    }
    slot_t& slot = slots_table_[best];
    bool restored = true;
    if (best_ancestor >= 0) {
        slot_t& anchor = slots_table_[best_ancestor];
        restored = api_->state_write_kv(engine_, anchor.state->get()) == static_cast<int>(anchor.tokens.size());
        anchor.last_used = ++clock_;
    }
    restored = restored && api_->state_write(engine_, slot.state->get()) == static_cast<int>(match_len);
    if (!restored) {
        header_print("WARNING", "Prefix snapshot of " << match_len << " tokens could not be restored, prefilling from scratch");
        forget_all();
        return result;
    }
    slot.last_used = ++clock_;
    live_tokens_ = slot.tokens;
    matched_tokens = slot.tokens;
    header_print("FLM", "Prefix cache: resuming at " << match_len << " of " << full_tokens.size() << " tokens");
    result.resumed = true;
    result.match_len = match_len;
    return result;
}

void PrefixSlotCache::save(const std::vector<int>& tokens, const std::vector<uint64_t>& image_hashes, bool boundary, int kv_from) {
    if (!engine_ || tokens.empty() || slot_limit() == 0) {
        return;
    }
    for (auto& slot : slots_table_) {
        if (slot.tokens == tokens) { // same prompt, same state: nothing to save
            slot.last_used = ++clock_;
            slot.boundary = slot.boundary || boundary;
            return;
        }
    }
    if (!boundary) {
        // An end snapshot only a few tokens past a slot that already covers this prompt (a short side call on a
        // shared system prompt, the first turn of a chat) would save re-processing just those tokens, at the price
        // of a whole recurrent state, and could push out a snapshot that saves thousands. Resume from that slot.
        for (const auto& slot : slots_table_) {
            if (slot.tokens.size() < tokens.size() && tokens.size() - slot.tokens.size() < min_end_gain && has_prefix(tokens, slot.tokens) &&
                slot.image_hashes.size() <= image_hashes.size() &&
                std::equal(slot.image_hashes.begin(), slot.image_hashes.end(), image_hashes.begin())) {
                return;
            }
        }
    }
    // an empty state of the engine's kind: it tells what a snapshot costs before the engine pays for the copy
    flm_state* raw = api_->state_new(engine_);
    if (!raw) {
        return; // the engine cannot save its context
    }
    auto state = std::make_unique<saved_state>(api_, raw);
    kv_from = std::clamp(kv_from, 0, static_cast<int>(tokens.size()));
    if (!announced_) {
        announced_ = true;
        const size_t fixed = state->estimate_bytes(0, 0);
        header_print("FLM", "Prefix cache: up to " << slot_limit() << " snapshots in " << (budget_bytes() >> 20)
                     << " MiB (a saved row costs " << (state->estimate_bytes(1, 0) - fixed) / 1024 << " KiB, a snapshot's state " << (fixed >> 20) << " MiB)");
    }
    const size_t budget = budget_bytes();
    const size_t limit = slot_limit();
    auto cost = [](const saved_state& s, size_t length, int from) { return s.estimate_bytes(static_cast<int>(length), from); };
    const size_t new_cost = cost(*state, tokens.size(), kv_from);
    if (new_cost > budget) {
        // this prompt alone does not fit; say so once, since nothing would ever be reused
        if (!warned_too_big_) {
            warned_too_big_ = true;
            header_print("WARNING", "Prefix cache: a " << tokens.size() << "-token prompt needs " << (new_cost >> 20) << " MiB, more than --prefix-cache-mb ("
                         << (budget >> 20) << " MiB); it is not saved. Raise --prefix-cache-mb to reuse prompts this long.");
        }
        return;
    }

    // `e` keeps only the rows after kv_from; `b` is the self-contained slot that holds the ones before
    auto is_parent = [&](const slot_t& b, const slot_t& e) {
        return e.kv_from() > 0 && b.kv_from() == 0 && b.tokens.size() == static_cast<size_t>(e.kv_from()) && has_prefix(e.tokens, b.tokens);
    };
    // the slot that supplies the head rows of the slot being saved must survive the eviction below
    const saved_state* protected_state = nullptr;
    if (kv_from > 0) {
        for (const auto& slot : slots_table_) {
            if (slot.kv_from() == 0 && slot.tokens.size() == static_cast<size_t>(kv_from) && has_prefix(tokens, slot.tokens)) {
                protected_state = slot.state.get();
            }
        }
    }
    // `a` is superseded when a longer prompt of the same conversation was saved after it: that one can serve
    // every request `a` could except replaying `a`'s own prompt, so `a` is the cheapest thing to lose
    auto superseded_by = [&](const slot_t& a, const std::vector<int>& longer, int longer_kv_from, const std::vector<uint64_t>& longer_images) {
        return !a.boundary && longer.size() > a.tokens.size() && longer_kv_from <= a.kv_from() && has_prefix(longer, a.tokens) &&
               a.image_hashes.size() <= longer_images.size() && std::equal(a.image_hashes.begin(), a.image_hashes.end(), longer_images.begin());
    };

    // Make room, cheapest loss first:
    //   1. superseded snapshots (an older end snapshot whose newer, longer one exists), least recently used first;
    //   2. other end snapshots, least recently used first: they only hold the rows after their boundary snapshot,
    //      so losing one costs a conversation its latest history, not its whole system prompt;
    //   3. boundary snapshots, least recently used first. One is only reached once no end snapshot depends on it,
    //      so an end snapshot is never left without the boundary snapshot that holds its head rows.
    // The slot that supplies the head of the snapshot being saved is never chosen.
    auto evict_one = [&]() -> bool {
        auto drop_at = [&](size_t i, const char* why) {
            header_print("FLM", "Prefix cache: dropping " << why << " (" << slots_table_[i].tokens.size() << " tokens)");
            slots_table_.erase(slots_table_.begin() + i);
        };
        const size_t n = slots_table_.size();
        std::vector<bool> has_dependents(n, false);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                if (i != j && is_parent(slots_table_[i], slots_table_[j])) has_dependents[i] = true;
            }
        }
        int stale = -1, end_leaf = -1, boundary_leaf = -1;
        auto older = [&](int candidate, int current) { return current < 0 || slots_table_[candidate].last_used < slots_table_[current].last_used; };
        for (size_t i = 0; i < n; i++) {
            const auto& a = slots_table_[i];
            if (a.state.get() == protected_state || has_dependents[i]) continue;
            bool old = superseded_by(a, tokens, kv_from, image_hashes);
            for (size_t j = 0; j < n && !old; j++) {
                const auto& b = slots_table_[j];
                old = j != i && superseded_by(a, b.tokens, b.kv_from(), b.image_hashes);
            }
            if (old && older(static_cast<int>(i), stale)) stale = static_cast<int>(i);
            if (!a.boundary && older(static_cast<int>(i), end_leaf)) end_leaf = static_cast<int>(i);
            if (a.boundary && older(static_cast<int>(i), boundary_leaf)) boundary_leaf = static_cast<int>(i);
        }
        if (stale >= 0) {
            drop_at(static_cast<size_t>(stale), "a superseded snapshot");
        } else if (end_leaf >= 0) {
            drop_at(static_cast<size_t>(end_leaf), "the least recently used end snapshot");
        } else if (boundary_leaf >= 0) {
            drop_at(static_cast<size_t>(boundary_leaf), "the least recently used boundary snapshot");
        } else {
            return false;
        }
        return true;
    };
    auto total_cost = [&]() {
        size_t total = new_cost;
        for (const auto& slot : slots_table_) total += cost(*slot.state, slot.tokens.size(), slot.kv_from());
        return total;
    };
    while (!slots_table_.empty() && (total_cost() > budget || slots_table_.size() >= limit)) {
        if (!evict_one()) {
            return; // everything left is needed by this slot: do not save it
        }
    }
    if (slots_table_.size() >= limit) {
        return;
    }
    if (api_->state_read(engine_, state->get(), kv_from) != static_cast<int>(tokens.size())) {
        header_print("WARNING", "Prefix snapshot of " << tokens.size() << " tokens was read at an unexpected length, dropping it");
        return;
    }
    slot_t slot;
    slot.state = std::move(state);
    slot.tokens = tokens;
    const size_t runs = std::min(count_image_runs(tokens, tokens.size()), image_hashes.size());
    slot.image_hashes.assign(image_hashes.begin(), image_hashes.begin() + runs);
    slot.last_used = ++clock_;
    slot.boundary = boundary;
    slots_table_.push_back(std::move(slot));
}
