/// \file test_prefix_slot_cache.cpp
/// \brief PrefixSlotCache against a fake engine: which snapshot a prompt resumes from, what is evicted, and that the
///        snapshots live in the cache (the engine only reads one out and writes one back). No NPU needed.
#include "prefix_slot_cache.hpp"

#include <cstdio>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

/// \brief a state the fake engine fills; `alive` counts the ones that exist, to see that dropping a slot frees it
class fake_state final {
public:
    static inline int alive = 0;
    fake_state(size_t fixed, size_t row) : fixed_(fixed), row_(row) { alive++; }
    ~fake_state() { alive--; }
    bool filled = false;
    int context_length = 0;
    int kv_from = 0;
    size_t estimate_bytes(int length, int from) const { return fixed_ + static_cast<size_t>(std::max(length - from, 0)) * row_; }

private:
    size_t fixed_, row_;
};

/// \brief a model that keeps no state of its own: `length` plays its context length, `calls` records what the cache
///        asked of it. It has no slots; whatever it was read into is the cache's.
class fake_engine {
public:
    std::vector<std::string> calls;
    std::vector<int> kv_written;        // context length of each state write_state_kv put back
    int length = 0;
    size_t row_bytes = 1024;            // 1 KiB per row
    size_t state_bytes = 1u << 20;      // 1 MiB per snapshot
    bool fail_restore = false;

    fake_state* new_state() { return new fake_state(state_bytes, row_bytes); }
    int read_state(fake_state& s, int kv_from) {
        calls.push_back("read");
        s.filled = true;
        s.context_length = length;
        s.kv_from = std::clamp(kv_from, 0, length);
        return length;
    }
    int write_state(const fake_state& state) {
        calls.push_back("write");
        if (fail_restore || !state.filled) return -1;
        length = state.context_length;
        return length;
    }
    int write_state_kv(const fake_state& state) {
        calls.push_back("write_kv");
        kv_written.push_back(state.context_length);
        return state.filled ? state.context_length : -1;
    }
};

/// \brief the flm_state_api table of the fake engine, as a library would export it
static const flm_state_api* fake_api() {
    static const flm_state_api api = {
        FLM_STATE_API_VERSION, sizeof(flm_state_api),
        [](void* e) { return reinterpret_cast<flm_state*>(static_cast<fake_engine*>(e)->new_state()); },
        [](flm_state* s) { delete reinterpret_cast<fake_state*>(s); },
        [](void* e, flm_state* s, int kv_from) { return static_cast<fake_engine*>(e)->read_state(*reinterpret_cast<fake_state*>(s), kv_from); },
        [](void* e, const flm_state* s) { return static_cast<fake_engine*>(e)->write_state(*reinterpret_cast<const fake_state*>(s)); },
        [](void* e, const flm_state* s) { return static_cast<fake_engine*>(e)->write_state_kv(*reinterpret_cast<const fake_state*>(s)); },
        [](const flm_state* s) { return reinterpret_cast<const fake_state*>(s)->context_length; },
        [](const flm_state* s) { return reinterpret_cast<const fake_state*>(s)->kv_from; },
        [](const flm_state* s, int length, int from) { return reinterpret_cast<const fake_state*>(s)->estimate_bytes(length, from); },
    };
    return &api;
}

std::vector<int> seq(int from, int count) {
    std::vector<int> v(count);
    std::iota(v.begin(), v.end(), from);
    return v;
}
std::vector<int> cat(std::vector<int> a, const std::vector<int>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

/// \brief the calls AutoModel::_prefix_insert makes on the cache around a prefill, with the fake engine in place of the
///        model: the heads plan_heads() picks are saved after passes of their own, then the end of the prompt
/// \param first where the first user turn starts (0 = none)
/// \param last where the last user turn starts (0 = none)
/// \param match length of the slot the prompt resumed from (0 = none)
/// \param images content hashes of the pictures of the prompt
/// \param split false while the prompt has pictures still to read
void prefill(PrefixSlotCache& cache, fake_engine& e, const std::vector<int>& prompt, size_t first, size_t last = 0, size_t match = 0,
             const std::vector<uint64_t>& images = {}, bool split = true) {
    const PrefixSlotCache::heads_t heads = cache.plan_heads(prompt, {first, last}, match, split);
    auto save_head = [&](size_t length) {
        const std::vector<int> h(prompt.begin(), prompt.begin() + length);
        e.length = static_cast<int>(length);
        cache.set_live(h);
        cache.save(h, images, true, 0);
    };
    if (heads.save_boundary) save_head(heads.boundary);
    if (heads.save_history) save_head(heads.history);
    e.length = static_cast<int>(prompt.size());
    cache.set_live(prompt);
    cache.set_last_prompt(prompt);
    if (!cache.boundary_only) cache.save(prompt, images, false, cache.end_kv_from(prompt, heads));
}

size_t resume(PrefixSlotCache& cache, const std::vector<int>& prompt, const std::vector<uint64_t>& images = {}) {
    std::vector<int> matched;
    const auto r = cache.resume(prompt, images, matched);
    return r.resumed ? r.match_len : 0;
}

/// \note min_gain 64 is what the tests below (tails of 100 tokens) were written for; the default (128) is checked in test_min_end_gain
PrefixSlotCache make(fake_engine& e, int mb = -1, int slots = -1, int min_gain = 64) {
    PrefixSlotCache cache;
    cache.configure(mb, slots, min_gain);
    cache.attach(&e, fake_api());
    return cache;
}

void test_slot_limit_and_ownership() {
    std::printf("at most 32 slots; the states live in the cache and go with it\n");
    {
        fake_engine e;
        PrefixSlotCache cache = make(e, 100000, 40);
        for (int i = 0; i < 40; i++) {
            prefill(cache, e, seq(1000 * (i + 1), 10), 0);
        }
        CHECK(cache.slot_count() == 32);
        CHECK(fake_state::alive == 32);               // one state per slot, none kept by the engine
        cache.forget_all();
        CHECK(fake_state::alive == 0);
    }
    CHECK(fake_state::alive == 0);
}

void test_longest_strict_prefix() {
    std::printf("resumes from the longest saved prefix that is strictly shorter than the prompt\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));
    prefill(cache, e, t1, 600);
    const auto t2 = cat(t1, seq(6000, 30));
    CHECK(resume(cache, t2) == t1.size());
    // the same prompt again: the end snapshot is not strictly shorter, the boundary one is
    CHECK(resume(cache, t1) == sys.size());
    // a different conversation with the same system prompt resumes at the boundary
    CHECK(resume(cache, cat(sys, seq(7000, 40))) == sys.size());
    // a different system prompt matches nothing
    CHECK(resume(cache, seq(9000, 700)) == 0);
}

void test_head_restored_after_cold_request() {
    std::printf("an end snapshot gets its head rows from the boundary snapshot once the live cache lost them\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));
    prefill(cache, e, t1, 600);
    // a side request with another system prompt (a title call) empties the live cache
    CHECK(resume(cache, seq(9000, 100)) == 0);
    cache.clear_live();
    e.calls.clear();
    CHECK(resume(cache, cat(t1, seq(6000, 30))) == t1.size());
    CHECK(e.calls.size() == 2 && e.calls[0] == "write_kv" && e.calls[1] == "write");
    // with the live cache intact, no head restore is needed
    e.calls.clear();
    CHECK(resume(cache, cat(t1, seq(6100, 30))) == t1.size());
    CHECK(e.calls.size() == 1);
}

void test_eviction_order() {
    std::printf("eviction: superseded end snapshots, then LRU end snapshots, then boundary snapshots without dependents\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 4);
    const auto sys_a = seq(1, 600);
    const auto a1 = cat(sys_a, seq(5000, 100));
    prefill(cache, e, a1, 600);                       // B_a, E_a1
    const auto a2 = cat(a1, seq(6000, 100));
    CHECK(resume(cache, a2) == a1.size());
    prefill(cache, e, a2, 600);                       // E_a2 supersedes E_a1 (B_a is already saved)
    CHECK(cache.slot_count() == 3);
    const auto sys_b = seq(20000, 600);
    const auto b1 = cat(sys_b, seq(25000, 100));
    prefill(cache, e, b1, 600);                       // B_b fits (4 slots), E_b1 forces one eviction: E_a1 (superseded)
    CHECK(cache.slot_count() == 4);
    CHECK(resume(cache, cat(a2, seq(7000, 10))) == a2.size());
    CHECK(resume(cache, cat(a1, seq(8000, 10))) == sys_a.size());  // E_a1 is gone, B_a serves it
    CHECK(resume(cache, cat(b1, seq(9000, 10))) == b1.size());
    // conversation c needs two slots: the end snapshots go (E_a2, least recently used, then E_b1) before any
    // boundary snapshot, so all three system prompts stay reusable
    const auto sys_c = seq(40000, 600);
    prefill(cache, e, cat(sys_c, seq(45000, 100)), 600);
    CHECK(cache.slot_count() == 4);
    CHECK(resume(cache, cat(a2, seq(7000, 10))) == sys_a.size());
    CHECK(resume(cache, cat(b1, seq(9000, 10))) == sys_b.size());
    CHECK(resume(cache, cat(sys_c, seq(47000, 10))) == sys_c.size());
}

void test_boundary_kept_for_dependents() {
    std::printf("a boundary snapshot is never evicted while an end snapshot needs its head rows\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 2);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));
    prefill(cache, e, t1, 600);                       // B (kv_from 0), E (kv_from 600)
    const auto t2 = cat(sys, seq(6000, 100));
    CHECK(resume(cache, t2) == sys.size());
    prefill(cache, e, t2, 600);                       // E2 needs B: E (not B) is evicted
    cache.clear_live();
    CHECK(resume(cache, cat(t2, seq(7000, 5))) == t2.size());
    CHECK(resume(cache, cat(sys, seq(8000, 5))) == sys.size());
}

void test_shared_history() {
    std::printf("prompts sharing everything before their last user turn resume there from the second one on, and that slot outlives the end slots\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 4);
    // a system prompt too short for a boundary slot, followed by few-shot examples
    const auto history = seq(1, 1500);
    prefill(cache, e, cat(history, seq(5000, 100)), 0, history.size());    // no prompt before it: only E1
    CHECK(cache.slot_count() == 1);
    const auto q2 = cat(history, seq(6000, 100));
    CHECK(resume(cache, q2) == 0);
    prefill(cache, e, q2, 0, history.size());                              // forked from the first: H, E2 (kv_from 1500)
    CHECK(cache.slot_count() == 3);
    for (int i = 0; i < 6; i++) {
        const auto q = cat(history, seq(7000 + 1000 * i, 100));
        CHECK(resume(cache, q) == history.size());
        prefill(cache, e, q, 0, history.size(), history.size());          // resumed at H: only an end slot is added
    }
    // the end slots were evicted, least recently used first, and H, which they need, stayed
    CHECK(cache.slot_count() == 4);
    cache.clear_live();
    CHECK(resume(cache, cat(history, seq(20000, 100))) == history.size());
}

void test_chat_saves_no_history() {
    std::printf("a chat turn, which only continues the turn before, saves no slot before its last user turn\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 8);
    const auto sys = seq(1, 600);
    const auto turn1 = cat(sys, seq(5000, 200));
    prefill(cache, e, turn1, 600);                                  // one user turn: B, E1
    const auto history2 = cat(turn1, seq(6000, 300));               // turn 1 with its reply, and the start of turn 2
    const auto turn2 = cat(history2, seq(7000, 200));
    CHECK(resume(cache, turn2) == turn1.size());
    prefill(cache, e, turn2, 600, history2.size(), turn1.size());  // continues turn 1: E2 only
    CHECK(cache.slot_count() == 3);
    CHECK(resume(cache, cat(history2, seq(8000, 30))) == turn1.size());
    // a prompt which shares the history of turn 2 but asks something else is a fork: it saves the history slot
    const auto other = cat(history2, seq(9000, 200));
    CHECK(resume(cache, other) == turn1.size());
    prefill(cache, e, other, 600, history2.size(), turn1.size());
    CHECK(resume(cache, cat(history2, seq(10000, 30))) == history2.size());
}

void test_find_user_turns() {
    std::printf("user turns are found by all of their marker tokens: the first and the last one, none at position 0\n");
    const std::vector<int> marker = {50000, 50001};
    const auto turn = [&](int from) { return cat(marker, seq(from, 10)); };
    const auto prompt = cat(cat(cat(seq(1, 5), turn(100)), turn(200)), turn(300));   // turns at 5, 17 and 29
    auto t = PrefixSlotCache::find_user_turns(prompt, marker);
    CHECK(t.first == 5 && t.last == 29);
    t = PrefixSlotCache::find_user_turns(cat(turn(100), turn(200)), marker);
    CHECK(t.first == 12 && t.last == 12);
    t = PrefixSlotCache::find_user_turns(cat(seq(1, 5), {50000, 7, 50001}), marker);
    CHECK(t.first == 0 && t.last == 0);
    t = PrefixSlotCache::find_user_turns(prompt, {});
    CHECK(t.first == 0 && t.last == 0);
}

void test_forked() {
    std::printf("a prompt forks from the prompt before when that one holds the history and more, and the prompt does not continue it\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    const auto history = seq(1, 1000);
    const auto p1 = cat(history, seq(5000, 100));
    CHECK(!cache.forked(p1, 1000));                                   // no prompt before it
    cache.set_last_prompt(p1);
    CHECK(cache.forked(cat(history, seq(6000, 100)), 1000));          // another last message
    CHECK(cache.forked(cat(history, seq(6000, 10)), 1000));           // a shorter one
    CHECK(cache.forked(cat(history, seq(5000, 50)), 1000));           // it ends inside the prompt before (an earlier turn again)
    CHECK(!cache.forked(p1, 1000));                                   // the same prompt again (a retry)
    CHECK(!cache.forked(cat(p1, seq(7000, 50)), 1000));               // it continues the prompt before (a chat turn)
    CHECK(!cache.forked(cat(seq(1, 999), seq(6000, 101)), 1000));     // it differs before the history ends
    CHECK(!cache.forked(cat(history, seq(6000, 100)), 1200));         // the history is longer than the prompt
    cache.forget_all();
    CHECK(!cache.forked(cat(history, seq(6000, 100)), 1000));
}

void test_plan_heads() {
    std::printf("plan_heads: which heads a prompt saves in prefill passes of their own\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 8);
    const auto prompt = cat(seq(1, 2000), seq(5000, 100));
    const PrefixSlotCache::user_turns_t turns{600, 2000};
    cache.set_last_prompt(cat(seq(1, 2000), seq(6000, 100)));        // the prompt before forked at 2000
    auto h = cache.plan_heads(prompt, turns, 0, true);
    CHECK(h.boundary == 600 && h.history == 2000 && h.save_boundary && h.save_history);
    // pictures still to read: the prompt is not split
    h = cache.plan_heads(prompt, turns, 0, false);
    CHECK(h.boundary == 600 && h.history == 2000 && !h.save_boundary && !h.save_history);
    // resumed at the boundary: only the history is left to save
    h = cache.plan_heads(prompt, turns, 600, true);
    CHECK(!h.save_boundary && h.save_history);
    // resumed less than --prefix-min-gain (64 here) before the history, or the history that close to the boundary
    h = cache.plan_heads(prompt, turns, 1950, true);
    CHECK(!h.save_boundary && !h.save_history);
    h = cache.plan_heads(prompt, {1950, 2000}, 0, true);
    CHECK(h.save_boundary && !h.save_history);
    // a head before min_boundary (512) is none, and a single user turn has no history
    h = cache.plan_heads(prompt, {500, 2000}, 0, true);
    CHECK(h.boundary == 0 && h.history == 2000 && !h.save_boundary && h.save_history);
    h = cache.plan_heads(prompt, {600, 600}, 0, true);
    CHECK(h.boundary == 600 && h.history == 0 && h.save_boundary && !h.save_history);
    h = cache.plan_heads(prompt, {100, 400}, 0, true);
    CHECK(h.boundary == 0 && h.history == 0 && !h.save_boundary && !h.save_history);
    // a prompt which continues the prompt before (a chat turn) is no fork
    cache.set_last_prompt(seq(1, 1500));
    h = cache.plan_heads(prompt, turns, 0, true);
    CHECK(h.save_boundary && !h.save_history);
}

void test_head_not_kept_not_split() {
    std::printf("a head whose slot would not be kept costs no prefill pass\n");
    const auto prompt = cat(seq(1, 3000), seq(5000, 100));
    const auto before = cat(seq(1, 3000), seq(6000, 100));
    // 1 MiB state + 1 KiB per row: 3 MiB holds a 1000-row slot, not a 3000-row one
    fake_engine e;
    PrefixSlotCache cache = make(e, 3, 8);
    cache.set_last_prompt(before);
    auto h = cache.plan_heads(prompt, {1000, 3000}, 0, true);
    CHECK(h.save_boundary && !h.save_history);
    fake_engine off_e;
    PrefixSlotCache off = make(off_e, -1, 0);
    off.set_last_prompt(before);
    h = off.plan_heads(prompt, {1000, 3000}, 0, true);
    CHECK(!h.save_boundary && !h.save_history);
}

void test_history_with_pictures() {
    std::printf("an end slot keeps the rows of a head with pictures itself, so another picture's rows never go under it\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 8);
    cache.image_token_id = 99999;   // not a token of seq()
    const auto sys = seq(1, 600);
    prefill(cache, e, cat(sys, seq(3000, 100)), 600);                          // a text conversation: B
    // a chat with picture A in its first turn, then an edited second turn: a history slot with picture A
    const auto c1 = cat(cat(sys, std::vector<int>(16, 99999)), seq(2000, 100));
    CHECK(resume(cache, c1, {111}) == sys.size());
    prefill(cache, e, c1, 600, 600, sys.size(), {111}, false);              // E1 from row 600, picture A read
    const auto history = cat(c1, seq(2500, 300));                           // the reply, up to the second user turn
    const auto c2 = cat(history, seq(4000, 100));
    CHECK(resume(cache, c2, {111}) == c1.size());
    prefill(cache, e, c2, 600, history.size(), c1.size(), {111});
    const auto c3 = cat(history, seq(4500, 100));
    CHECK(resume(cache, c3, {111}) == c1.size());
    cache.begin_report();
    prefill(cache, e, c3, 600, history.size(), c1.size(), {111});           // a fork of c2: H with picture A
    auto rep = cache.report();
    CHECK(rep.saved.size() == 2 && rep.saved[0].boundary && rep.saved[0].tokens == static_cast<int>(history.size()));
    CHECK(rep.saved[1].kv_from == static_cast<int>(sys.size()));            // the end slot keeps the picture's rows
    // the same tokens with picture B: H does not serve it, and its end slot keeps the rows from the boundary on
    const auto p4 = cat(history, seq(8000, 100));
    CHECK(resume(cache, p4, {222}) == sys.size());
    cache.begin_report();
    prefill(cache, e, p4, 600, history.size(), sys.size(), {222}, false);
    rep = cache.report();
    CHECK(rep.saved.size() == 1 && rep.saved[0].kv_from == static_cast<int>(sys.size()));
    // once the live context is gone, its head rows come from B, not from H with picture A
    cache.clear_live();
    e.kv_written.clear();
    CHECK(resume(cache, cat(p4, seq(9000, 5)), {222}) == p4.size());
    CHECK(e.kv_written == std::vector<int>{600});
}

void test_small_tail_not_saved() {
    std::printf("a side call a few tokens past the shared head is not saved, so it cannot push out the main conversation\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 2);
    const auto sys = seq(1, 600);
    const auto main1 = cat(sys, seq(5000, 1000));
    prefill(cache, e, main1, 600);                    // B, E_main
    for (int i = 0; i < 3; i++) {
        const auto side = cat(sys, seq(9000 + 100 * i, 17));
        CHECK(resume(cache, side) == sys.size());
        prefill(cache, e, side, 600);                 // 17 tokens past B: not saved
        CHECK(cache.slot_count() == 2);
        CHECK(resume(cache, cat(main1, seq(20000 + 100 * i, 30))) == main1.size());
    }
    // the first turn of a chat (a short user message after the head) is not saved either; the next turn resumes at the head
    const auto chat1 = cat(sys, seq(30000, 20));
    prefill(cache, e, chat1, 600);
    CHECK(resume(cache, cat(chat1, seq(31000, 30))) == sys.size());
}

void test_min_end_gain() {
    std::printf("--prefix-min-gain: an end snapshot must add that many tokens over a covering slot (default 128, 0 = save all)\n");
    const auto base = seq(1, 1000);
    const auto plus100 = cat(base, seq(5000, 100));
    const auto plus200 = cat(base, seq(6000, 200));
    {
        fake_engine e;
        PrefixSlotCache cache;                        // flag not given
        cache.configure(100000, 32, -1);
        cache.attach(&e, fake_api());
        CHECK(cache.min_end_gain == 128);
        prefill(cache, e, base, 0);
        prefill(cache, e, plus100, 0);
        CHECK(cache.slot_count() == 1);               // 100 < 128: not saved, the next request resumes from `base`
        CHECK(resume(cache, cat(plus100, seq(7000, 10))) == base.size());
        prefill(cache, e, plus200, 0);
        CHECK(cache.slot_count() == 2);               // 200 >= 128: saved
    }
    {
        fake_engine e;
        PrefixSlotCache cache = make(e, 100000, 32, 0);  // --prefix-min-gain 0: every end snapshot is saved
        prefill(cache, e, base, 0);
        prefill(cache, e, plus100, 0);
        CHECK(cache.slot_count() == 2);
        CHECK(resume(cache, cat(plus100, seq(7000, 10))) == plus100.size());
    }
    {
        fake_engine e;
        PrefixSlotCache cache = make(e, 100000, 32, 300);  // --prefix-min-gain 300
        CHECK(cache.min_end_gain == 300);
        prefill(cache, e, base, 0);
        prefill(cache, e, plus200, 0);
        CHECK(cache.slot_count() == 1);               // 200 < 300: not saved
        prefill(cache, e, cat(base, seq(8000, 300)), 0);
        CHECK(cache.slot_count() == 2);               // 300 >= 300: saved
    }
}

void test_budget() {
    std::printf("the MiB budget evicts, and a prompt larger than the budget is not saved\n");
    fake_engine e;
    // 1 MiB state + 1 KiB per row; 3 MiB holds two 300-row self-contained snapshots, not three
    PrefixSlotCache cache = make(e, 3, 32);
    prefill(cache, e, seq(1, 300), 0);
    prefill(cache, e, seq(10000, 300), 0);
    prefill(cache, e, seq(20000, 300), 0);
    CHECK(cache.slot_count() == 2);
    CHECK(resume(cache, seq(1, 400)) == 0);           // the oldest went
    CHECK(resume(cache, seq(20000, 400)) == 300);
    // the default budget is 4096 MiB: a 2000 MiB snapshot fits with default flags (1024 MiB would refuse it)
    fake_engine huge;
    huge.state_bytes = 2000ull << 20;
    PrefixSlotCache defaults = make(huge);
    prefill(defaults, huge, seq(1, 10), 0);
    CHECK(defaults.slot_count() == 1);
    // 3000 rows = 1 MiB + 2.9 MiB > 3 MiB: refused
    fake_engine big;
    PrefixSlotCache small = make(big, 3, 32);
    prefill(small, big, seq(1, 3000), 0);
    CHECK(small.slot_count() == 0);
}

void test_images() {
    std::printf("a snapshot with images is only reused for the same pictures\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    cache.image_token_id = 99;
    std::vector<int> prompt = cat(seq(1, 20), std::vector<int>(16, 99));
    prompt = cat(prompt, seq(200, 20));
    e.length = static_cast<int>(prompt.size());
    cache.set_live(prompt);
    cache.save(prompt, {111}, false, 0);
    const auto next = cat(prompt, seq(500, 10));
    CHECK(resume(cache, next, {222}) == 0);
    CHECK(resume(cache, next, {111}) == prompt.size());

    // the same tokens with another picture of the same size replace the slot: the engine holds that one now
    cache.set_live(prompt);
    cache.save(prompt, {222}, false, 0);
    CHECK(cache.slot_count() == 1);
    CHECK(resume(cache, next, {111}) == 0);
    CHECK(resume(cache, next, {222}) == prompt.size());

    // a slot saved without the hash of a picture it holds is never resumed: the picture could be any
    fake_engine e2;
    PrefixSlotCache unhashed = make(e2);
    unhashed.image_token_id = 99;
    e2.length = static_cast<int>(prompt.size());
    unhashed.set_live(prompt);
    unhashed.save(prompt, {}, true, 0);
    CHECK(unhashed.slot_count() == 1);
    CHECK(resume(unhashed, next, {111}) == 0);

    struct image { int grid_h, grid_w; };
    struct payload { std::vector<image> images; std::vector<uint16_t> _data__processed; };
    payload p1{{{2, 2}, {2, 2}}, std::vector<uint16_t>(2 * 2 * 4 * 3 * 2, 7)};
    payload p2 = p1;
    p2._data__processed.back() = 8;                   // second picture differs in one value
    const auto h1 = prefix_image_hashes(p1, 2, 1);
    const auto h2 = prefix_image_hashes(p2, 2, 1);
    CHECK(h1.size() == 2 && h2.size() == 2);
    CHECK(h1[0] == h2[0]);
    CHECK(h1[1] != h2[1]);
}

void test_picture_replaced_keeps_others() {
    std::printf("a slot replaced for another picture takes only the slots that need its rows\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    cache.image_token_id = 99999;   // not a token of seq()
    const auto sys = seq(1, 600);
    const auto x = cat(sys, seq(5000, 100));                                   // a text conversation
    prefill(cache, e, x, 600);                                                 // boundary H, end EX
    const auto y = cat(cat(sys, std::vector<int>(16, 99999)), seq(6000, 100));    // an image conversation
    CHECK(resume(cache, y, {111}) == sys.size());
    e.length = static_cast<int>(y.size());
    cache.set_live(y);
    PrefixSlotCache::heads_t at_sys;
    at_sys.boundary = 600;
    cache.save(y, {111}, false, cache.end_kv_from(y, at_sys));                 // EY
    CHECK(cache.slot_count() == 3);
    cache.save(y, {222}, false, cache.end_kv_from(y, at_sys));                 // the same question about another picture
    CHECK(cache.slot_count() == 3);
    CHECK(resume(cache, cat(y, seq(7000, 5)), {222}) == y.size());
    CHECK(resume(cache, cat(y, seq(7000, 5)), {111}) == sys.size());
    CHECK(resume(cache, cat(x, seq(8000, 5))) == x.size());                    // the text conversation keeps its end slot
}

void test_restore_failure_and_off() {
    std::printf("a failed restore forgets every slot; 0 slots turns the cache off\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    prefill(cache, e, seq(1, 600), 0);
    e.fail_restore = true;
    CHECK(resume(cache, seq(1, 700)) == 0);
    CHECK(cache.slot_count() == 0);
    CHECK(fake_state::alive == 0);

    fake_engine off_e;
    PrefixSlotCache off = make(off_e, -1, 0);
    CHECK(!off.enabled());
    prefill(off, off_e, seq(1, 600), 0);
    CHECK(off.slot_count() == 0);
}

void test_report() {
    std::printf("the report says what a request did: lookup, slots saved and dropped, fill level\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, -1, 3);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));

    // first request: nothing saved yet, then it saves a boundary and an end slot
    cache.begin_report();
    std::vector<int> matched;
    auto r = cache.resume(t1, {}, matched);
    CHECK(!r.resumed && r.miss_reason == "no snapshots saved yet");
    prefill(cache, e, t1, 600);
    auto rep = cache.report();
    CHECK(rep.valid && !rep.hit && rep.miss_reason == "no snapshots saved yet");
    CHECK(rep.saved.size() == 2 && rep.saved[0].boundary && !rep.saved[1].boundary);
    CHECK(rep.saved[0].slot == 1 && rep.saved[1].slot == 2 && rep.saved[1].kv_from == 600);
    CHECK(rep.slots_used == 2 && rep.slots_max == 3 && rep.bytes_used > 0 && rep.bytes_used <= rep.bytes_budget);

    // the next turn resumes from the end slot
    cache.begin_report();
    const auto t2 = cat(t1, seq(6000, 30));
    r = cache.resume(t2, {}, matched);
    rep = cache.report();
    CHECK(r.resumed && rep.hit && rep.slot == 2 && !rep.slot_boundary && rep.matched_tokens == static_cast<int>(t1.size()));
    CHECK(rep.saved.empty() && rep.dropped == 0);

    // an unrelated prompt: a miss that says how far it matched
    cache.begin_report();
    r = cache.resume(seq(9000, 700), {}, matched);
    CHECK(!r.resumed && r.miss_reason.find("no snapshot is a prefix") == 0);
    // the exact same prompt: the slot is not strictly shorter
    cache.begin_report();
    r = cache.resume(t1, {}, matched);
    CHECK(r.resumed && cache.report().slot_boundary);  // the boundary slot serves it
    cache.begin_report();
    r = cache.resume(sys, {}, matched);
    CHECK(!r.resumed && r.miss_reason == "the prompt is not longer than a saved one");

    // ids are never reused: a fourth slot evicts one and gets id 3
    cache.begin_report();
    prefill(cache, e, seq(20000, 700), 0);
    prefill(cache, e, seq(30000, 700), 0);
    rep = cache.report();
    CHECK(rep.dropped >= 1 && rep.saved.size() == 2 && rep.saved[0].slot == 3 && rep.saved[1].slot == 4);
}
} // namespace

int main() {
    test_slot_limit_and_ownership();
    test_longest_strict_prefix();
    test_head_restored_after_cold_request();
    test_eviction_order();
    test_boundary_kept_for_dependents();
    test_shared_history();
    test_chat_saves_no_history();
    test_find_user_turns();
    test_forked();
    test_plan_heads();
    test_head_not_kept_not_split();
    test_history_with_pictures();
    test_small_tail_not_saved();
    test_min_end_gain();
    test_budget();
    test_images();
    test_picture_replaced_keeps_others();
    test_restore_failure_and_off();
    test_report();
    if (failures > 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
