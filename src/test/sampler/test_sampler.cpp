// Sampler (src/common/modules/sampler.cpp) on the real code, no NPU. Built twice: scalar, and with USEAVX2=1.
#include "modules/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;
std::string g_failure_text;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            char msg_[256];                                                       \
            std::snprintf(msg_, sizeof msg_, __VA_ARGS__);                        \
            g_failure_text += std::string("\n    ") + #cond + ": " + msg_;       \
            g_failures++;                                                         \
        }                                                                         \
    } while (0)

using Entries = std::vector<std::pair<int, float>>;

// The buffer owns its memory (the vector constructors of buffer<> only map it).
buffer<bf16> logits_of(int vocab, const Entries& set, float fill = -10.0f) {
    buffer<bf16> b(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; i++) b[i] = bf16(fill);
    for (const auto& [id, x] : set) b[id] = bf16(x);
    return b;
}

sampler_config config_of(int top_k, float temperature, uint64_t seed = 1234) {
    sampler_config c;
    c.top_k = top_k;
    c.top_p = 1.0f;
    c.min_p = 0.0f;
    c.temperature = temperature;
    c.has_rng_seed = true;
    c.rng_seed = seed;
    c.repeat_last_n = 0;
    return c;
}

// Equal maxima in different lanes of an 8-wide scan: the lowest id must win (llama.cpp and vLLM argmax keep the
// first maximum).
void TestGreedyTieTakesTheSmallestId() {
    {
        auto x = logits_of(1024, {{3, 5.0f}, {7, 5.0f}, {9, 5.0f}, {300, 5.0f}});
        auto c = config_of(1, 0.7f);
        Sampler s(1024, c);
        int got = s.sample(x);
        CHECK(got == 3, "top_k=1 on tied logits {3,7,9,300} returned %d, want 3", got);
    }
    {
        auto x = logits_of(1024, {{3, 5.0f}, {7, 5.0f}, {9, 5.0f}, {300, 5.0f}});
        auto c = config_of(40, 0.0f);
        Sampler s(1024, c);
        int got = s.sample(x);
        CHECK(got == 3, "temperature 0 on tied logits returned %d, want 3", got);
    }
    {
        // the tail of the scan (ids past the last full group of 8) takes a different loop
        auto x = logits_of(1027, {{1000, 5.0f}, {1025, 5.0f}});
        auto c = config_of(1, 1.0f);
        Sampler s(1027, c);
        int got = s.sample(x);
        CHECK(got == 1000, "tie between 1000 and 1025 returned %d, want 1000", got);
    }
}

// top_k <= 0 means no top-k (vLLM and llama.cpp); what is set at the sampler must never leave the previous token's
// candidates in place, nor crash.
void TestTopKOfZeroOrBelowIsOffAndLeavesNoStaleCandidates() {
    Entries first, second;
    for (int i = 0; i < 40; i++) first.push_back({i, 8.0f - 0.01f * i});
    for (int i = 100; i < 140; i++) second.push_back({i, 8.0f - 0.01f * i});
    for (int k : {0, -1, -7}) {
        auto x1 = logits_of(1024, first);
        auto x2 = logits_of(1024, second);
        auto c = config_of(40, 1.0f);
        Sampler s(1024, c);
        s.sample(x1);
        s.top_k = k;
        int stale = 0;
        for (int i = 0; i < 200; i++) {
            int t = s.sample(x2);
            if (t < 40) stale++;
        }
        CHECK(stale == 0, "top_k=%d: %d of 200 samples came from the previous step's candidates", k, stale);
    }
}

void TestSamplerBuiltWithAnOutOfRangeTopKDoesNotCrash() {
    for (int k : {0, -1, -3, 1024, 100000}) {
        auto x = logits_of(1024, {{5, 4.0f}, {6, 3.0f}});
        auto c = config_of(k, 1.0f);
        Sampler s(1024, c);
        int got = s.sample(x);
        CHECK(got >= 0 && got < 1024, "top_k=%d returned %d", k, got);
    }
}

// Below 1e-6 the softmax is the argmax anyway; dividing by it overflows to inf and then NaN.
void TestTinyTemperatureIsGreedy() {
    for (float T : {1e-6f, 1e-20f, 1e-30f, 1e-38f, 1e-45f}) {
        auto x = logits_of(1024, {{11, 6.0f}, {12, 5.0f}, {900, 4.0f}});
        auto c = config_of(40, T);
        Sampler s(1024, c);
        for (int i = 0; i < 20; i++) {
            int got = s.sample(x);
            CHECK(got == 11, "temperature %g returned %d, want the argmax 11", T, got);
            if (got != 11) break;
        }
    }
}

void TestSameSeedSameSequence() {
    auto x = logits_of(1024, {{1, 2.0f}, {2, 2.0f}, {3, 2.0f}, {4, 2.0f}, {5, 2.0f}, {6, 2.0f}});
    std::vector<int> a, b;
    for (int run = 0; run < 2; run++) {
        auto c = config_of(40, 1.0f);
        Sampler s(1024, c);
        s.set_seed(777);
        for (int i = 0; i < 50; i++) (run == 0 ? a : b).push_back(s.sample(x));
    }
    CHECK(a == b, "the same seed gave two different sequences");
}

// The token sequence a seed gives (four equal candidates, so the probabilities are exact): one draw per token. The
// expected ids were computed with the same independent mt19937_64 as below.
void TestSeededSequenceIsPortable() {
    auto x = logits_of(64, {{10, 1.0f}, {20, 1.0f}, {30, 1.0f}, {40, 1.0f}});
    auto c = config_of(4, 1.0f);
    Sampler s(64, c);
    s.set_seed(777);
    const std::vector<int> want = {10, 30, 40, 10, 20, 40, 30, 40, 10, 30, 40, 10,
                                   40, 10, 20, 20, 20, 20, 20, 40, 20, 20, 30, 20};
    std::vector<int> got;
    for (size_t i = 0; i < want.size(); i++) got.push_back(s.sample(x));
    CHECK(got == want, "seed 777 gave a different sequence than the portable one (first: %d, want %d)", got[0], want[0]);
}

// The draw itself: u = (mt19937_64() >> 40) * 2^-24, the same on every compiler and standard library (the standard
// fixes mt19937_64, but not std::uniform_real_distribution). Expected integers come from an independent mt19937_64
// in Python, checked against the standard's 10000th output for seed 5489.
// A template, so that the `requires` below is a check and not a compile error where next_uniform() does not exist.
template <class S>
void check_draws(S& s) {
    if constexpr (requires(S& t) { t.next_uniform(); }) {
        const uint32_t want[6] = {1397429, 10729138, 12812495, 363642, 7625712, 13399418};
        for (int i = 0; i < 6; i++) {
            const float got = s.next_uniform();
            CHECK(got == float(want[i]) / 16777216.0f, "draw %d: got %.9g, want %u / 2^24", i, got, want[i]);
        }
    } else {
        CHECK(false, "Sampler has no next_uniform(): the draw is std::uniform_real_distribution<float>, implementation-defined");
    }
}

void TestUniformDrawsArePortable() {
    auto c = config_of(4, 1.0f);
    Sampler s(64, c);
    s.set_seed(777);
    check_draws(s);
}

// Penalties: a positive logit is divided by rep_penalty, a negative one multiplied; the count and presence penalties
// are subtracted (llama.cpp's formulas). Checked through the winner of a greedy pick.
void TestPenaltyFormulas() {
    {
        auto x = logits_of(16, {{1, 3.0f}, {2, 2.9f}});
        sampler_config c = config_of(1, 1.0f);
        c.repeat_last_n = 64;
        c.rep_penalty = 1.2f;  // 3.0 / 1.2 = 2.5 < 2.9
        Sampler s(16, c);
        int first = s.sample(x);
        int second = s.sample(x);
        CHECK(first == 1 && second == 2, "repetition penalty: got %d then %d, want 1 then 2", first, second);
    }
    {
        auto x = logits_of(16, {{1, 3.0f}, {2, 2.9f}});
        sampler_config c = config_of(1, 1.0f);
        c.repeat_last_n = 64;
        c.freq_penalty = 0.2f;  // 3.0 - 0.2 = 2.8 < 2.9
        Sampler s(16, c);
        int first = s.sample(x);
        int second = s.sample(x);
        CHECK(first == 1 && second == 2, "frequency penalty: got %d then %d, want 1 then 2", first, second);
    }
    {
        auto x = logits_of(16, {{1, 3.0f}, {2, 2.9f}});
        sampler_config c = config_of(1, 1.0f);
        c.repeat_last_n = 64;
        c.pre_penalty = 0.2f;  // presence: 3.0 - 0.2 = 2.8 < 2.9
        Sampler s(16, c);
        int first = s.sample(x);
        int second = s.sample(x);
        CHECK(first == 1 && second == 2, "presence penalty: got %d then %d, want 1 then 2", first, second);
    }
    {
        // a negative logit becomes more negative: -0.5 * 2 = -1.0 loses to -0.9
        auto x = logits_of(16, {{1, -0.5f}, {2, -0.9f}}, -30.0f);
        sampler_config c = config_of(1, 1.0f);
        c.repeat_last_n = 64;
        c.rep_penalty = 2.0f;
        Sampler s(16, c);
        int first = s.sample(x);
        int second = s.sample(x);
        CHECK(first == 1 && second == 2, "negative logit under repetition penalty: got %d then %d, want 1 then 2", first, second);
    }
}

double tv_distance(const std::vector<int>& counts, const std::vector<double>& want, int n) {
    double d = 0;
    for (size_t i = 0; i < want.size(); i++) d += std::fabs(counts[i] / double(n) - want[i]);
    return d / 2;
}

// top-p and min-p cut the candidate list, the rest is renormalised: compared with the exact distribution.
void TestTopPAndMinPDistributions() {
    const double p[5] = {0.5, 0.3, 0.1, 0.07, 0.03};
    Entries set;
    for (int i = 0; i < 5; i++) set.push_back({i, float(std::log(p[i]))});
    const int N = 40000;
    {
        auto x = logits_of(16, set);
        sampler_config c = config_of(5, 1.0f, 99);
        c.top_p = 0.85f;  // 0.5, 0.8, 0.9: three tokens stay
        Sampler s(16, c);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < N; i++) counts[s.sample(x)]++;
        double tvd = tv_distance(counts, {0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0, 0}, N);
        CHECK(tvd < 0.02, "top_p=0.85: total variation distance %.4f", tvd);
    }
    {
        auto x = logits_of(16, set);
        sampler_config c = config_of(5, 1.0f, 99);
        c.min_p = 0.25f;  // keep p >= 0.25 * 0.5: tokens 0 and 1
        Sampler s(16, c);
        std::vector<int> counts(5, 0);
        for (int i = 0; i < N; i++) counts[s.sample(x)]++;
        double tvd = tv_distance(counts, {0.5 / 0.8, 0.3 / 0.8, 0, 0, 0}, N);
        CHECK(tvd < 0.02, "min_p=0.25: total variation distance %.4f", tvd);
    }
}

// sample() as it was before top-k off got its own sort: the heap of sampler_topk_apply() over the whole vocabulary,
// then every softmax of the two pipelines. Built from the sampler's own steps, so only sample() itself is compared.
int reference_sample(Sampler& s, buffer<bf16>& x, const std::vector<uint32_t>* allowed) {
    for (int i = 0; i < s.in_features; i++) s.logits[i] = x[i];
    s.sampler_penalty_apply_sparse();
    s.sampler_allowed_apply(allowed);
    s.sampler_topk_apply(s.top_k);
    if (s.use_optimized_sampling) {
        s.softmax_with_topp_minp(s.top_p, s.min_p);
        s.sampler_temp_apply(s.temperature);
        s.softmax_inplace();
    } else {
        s.softmax_inplace();
        s.sampler_topp_apply(s.top_p);
        s.softmax_inplace();
        s.sampler_minp_apply(s.min_p);
        s.sampler_temp_apply(s.temperature);
        s.softmax_inplace();
    }
    int t = s.sample_from_probs();
    s.ring_buffer_update_sparse(t);
    return t;
}

uint32_t bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

// Same candidate list bit for bit (logits and probabilities) and the same history
bool same_state(const Sampler& a, const Sampler& b) {
    if (a.top_k_logits.size() != b.top_k_logits.size()) return false;
    for (size_t i = 0; i < a.top_k_logits.size(); i++) {
        const auto& p = a.top_k_logits[i];
        const auto& q = b.top_k_logits[i];
        if (p.token_id != q.token_id || bits_of(p.logits) != bits_of(q.logits) || bits_of(p.prob) != bits_of(q.prob)) {
            return false;
        }
    }
    return a.token_history == b.token_history;
}

void set_bits(buffer<bf16>& b, int id, uint16_t bits) { std::memcpy(&b[id], &bits, sizeof bits); }

// Logits with many exact ties: bf16 keeps 8 bits of mantissa, and two patterns draw from a handful of values, -0 and
// +0 among them.
buffer<bf16> tied_logits(int vocab, int pattern, std::mt19937& rng) {
    buffer<bf16> b(static_cast<size_t>(vocab));
    std::normal_distribution<float> normal(0.0f, 3.0f);
    const float few[6] = {2.0f, 1.0f, 0.0f, -0.0f, -1.0f, -6.0f};
    for (int i = 0; i < vocab; i++) {
        switch (pattern) {
            case 0: b[i] = bf16(normal(rng)); break;
            case 1: b[i] = bf16(few[rng() % 6]); break;
            case 2: b[i] = bf16(0.0f); break;
            default: b[i] = bf16(normal(rng) * 0.25f - 4.0f); break;  // flat: top-p keeps most of the vocabulary
        }
    }
    if (pattern == 2) {
        for (int i = 0; i < vocab; i += 3) set_bits(b, i, 0x8000);  // -0, tied with the +0 around it
    }
    return b;
}

struct SamplingCase {
    float top_p, min_p, temperature;
    bool optimized, penalties;
};

// A top-k-off spelling against the old heap path: same seed, same tokens, same candidate list, step by step.
int compare_with_reference(int vocab, int steps, int pattern, const SamplingCase& sc, int top_k, bool masked,
                           uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    sampler_config c = config_of(top_k, sc.temperature, seed);
    c.top_p = sc.top_p;
    c.min_p = sc.min_p;
    c.use_optimized_sampling = sc.optimized;
    if (sc.penalties) {
        c.repeat_last_n = 16;
        c.rep_penalty = 1.3f;
        c.freq_penalty = 0.1f;
        c.pre_penalty = 0.2f;
    }
    Sampler fast(vocab, c);
    sampler_config ref_c = c;
    ref_c.top_k = vocab;  // what top-k off stands for
    Sampler ref(vocab, ref_c);
    for (int step = 0; step < steps; step++) {
        auto x = tied_logits(vocab, pattern, rng);
        std::vector<uint32_t> allowed((vocab + 31) / 32, 0xFFFFFFFFu);
        if (masked) {
            for (auto& w : allowed) w = static_cast<uint32_t>(rng());
        }
        const auto* mask = masked ? &allowed : nullptr;
        const int got = fast.sample(x, mask);
        const int want = reference_sample(ref, x, mask);
        if (got != want || !same_state(fast, ref)) {
            CHECK(false, "vocab %d pattern %d top_k %d top_p %g min_p %g T %g %s%s%s step %d: token %d, reference %d",
                  vocab, pattern, top_k, sc.top_p, sc.min_p, sc.temperature, sc.optimized ? "optimized" : "legacy",
                  sc.penalties ? " penalties" : "", masked ? " masked" : "", step, got, want);
            return 1;
        }
    }
    return 0;
}

// top_k <= 0 or >= the vocabulary sorts the whole vocabulary without the heap, and skips the softmaxes nothing
// reads: none of it may change a token, a candidate or a probability.
void TestTopKOffMatchesTheHeapPath() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<SamplingCase> cases;
    for (bool optimized : {true, false}) {
        for (float top_p : {1.0f, 0.95f, 0.5f, 0.0f, nan}) {
            for (float min_p : {0.0f, 0.05f, 0.5f, 1.0f, 2.0f, nan}) {
                for (float T : {0.0f, 0.7f, 1.3f}) {
                    cases.push_back({top_p, min_p, T, optimized, (cases.size() % 3) == 0});
                }
            }
        }
    }
    int failures = 0;
    uint64_t seed = 1;
    for (const auto& sc : cases) {
        for (int pattern = 0; pattern < 4; pattern++) {
            for (int vocab : {1000, 37}) {
                for (int top_k : {0, -1, vocab, vocab + 5}) {
                    failures += compare_with_reference(vocab, 4, pattern, sc, top_k, (seed % 4) == 1, seed);
                    seed++;
                    if (failures > 5) return;
                }
            }
        }
    }
    // a real vocabulary size, fewer combinations: this is where the heap sort cost
    const SamplingCase large[] = {{1.0f, 0.0f, 0.7f, true, false},  {0.9f, 0.0f, 0.7f, true, true},
                                  {1.0f, 0.05f, 0.7f, true, false}, {0.95f, 0.05f, 1.0f, true, false},
                                  {1.0f, 0.0f, 1.0f, false, true},  {0.9f, 0.1f, 0.8f, false, false}};
    for (const auto& sc : large) {
        for (int pattern = 0; pattern < 4; pattern++) {
            failures += compare_with_reference(151936, 2, pattern, sc, 0, pattern == 3, seed++);
        }
    }
}

// -0 and +0 compare equal, so the lower id goes first whichever of them it holds
void TestSignedZerosTieByIdWithTopKOff() {
    for (bool minus_first : {true, false}) {
        auto x = logits_of(1024, {});
        set_bits(x, 3, minus_first ? 0x8000 : 0x0000);
        set_bits(x, 5, minus_first ? 0x0000 : 0x8000);
        auto c = config_of(0, 0.0f);
        Sampler s(1024, c);
        const int got = s.sample(x);
        CHECK(got == 3, "%s at 3: top_k=0, temperature 0 returned %d, want 3", minus_first ? "-0" : "+0", got);
        CHECK(s.top_k_logits.size() >= 2 && s.top_k_logits[1].token_id == 5, "%s at 3: the runner-up is not 5",
              minus_first ? "-0" : "+0");
    }
}

// A NaN logit has no place in the order: top-k off falls back to the heap path, and so still agrees with it.
void TestNaNLogitFallsBackToTheHeapPath() {
    std::mt19937 rng(5);
    auto x = tied_logits(1000, 0, rng);
    set_bits(x, 17, 0x7FC0);
    auto c = config_of(0, 0.8f, 42);
    c.top_p = 0.9f;
    Sampler fast(1000, c);
    sampler_config ref_c = c;
    ref_c.top_k = 1000;
    Sampler ref(1000, ref_c);
    for (int i = 0; i < 8; i++) {
        const int got = fast.sample(x);
        const int want = reference_sample(ref, x, nullptr);
        CHECK(got == want && same_state(fast, ref), "step %d: token %d, reference %d", i, got, want);
    }
}

// sample() on an fp32 row (a verify row of a speculative batch) is sample() on the bf16 buffer it came from: same
// tokens, same candidates, same penalty history, same draws.
void TestFloatRowMatchesTheBf16Buffer() {
    std::mt19937 rng(11);
    uint64_t seed = 1;
    for (int top_k : {1, 20, 0}) {
        for (float T : {0.0f, 0.7f}) {
            for (bool penalties : {false, true}) {
                for (bool masked : {false, true}) {
                    sampler_config c = config_of(top_k, T, seed++);
                    c.top_p = 0.9f;
                    c.min_p = 0.05f;
                    if (penalties) {
                        c.repeat_last_n = 16;
                        c.rep_penalty = 1.1f;
                        c.pre_penalty = 1.5f;
                    }
                    Sampler a(1000, c), b(1000, c);
                    for (int step = 0; step < 8; step++) {
                        auto x = tied_logits(1000, step % 4, rng);
                        std::vector<float> row(1000);
                        for (int i = 0; i < 1000; i++) row[i] = x[i];
                        std::vector<uint32_t> allowed((1000 + 31) / 32, 0xFFFFFFFFu);
                        if (masked) {
                            for (auto& w : allowed) w = static_cast<uint32_t>(rng()) | 1u;
                        }
                        const auto* mask = masked ? &allowed : nullptr;
                        const int want = a.sample(x, mask);
                        const int got = b.sample(row.data(), mask);
                        if (got != want || !same_state(a, b)) {
                            CHECK(false, "top_k %d T %g%s%s step %d: row gave %d, buffer %d", top_k, T,
                                  penalties ? " penalties" : "", masked ? " masked" : "", step, got, want);
                            return;
                        }
                    }
                }
            }
        }
    }
}

// The engine's side of one speculative cycle (qwen3_8mtp_npu::speculate with set_speculation_chooser): the token of
// verify row i comes from the chooser, here the sampler; drafts are kept while they equal it, and the first row that
// differs, or the row past the last draft, is the cycle's free token.
using Target = std::function<std::vector<float>(const std::vector<int>&)>;

std::vector<int> speculate_with(Sampler& s, const Target& target, std::vector<int>& ctx, const std::vector<int>& drafts,
                                int* accepted_drafts = nullptr) {
    std::vector<int> committed;
    for (size_t i = 0; i <= drafts.size(); i++) {
        const std::vector<float> row = target(ctx);
        const int token = s.sample(row.data());
        committed.push_back(token);
        ctx.push_back(token);
        if (i == drafts.size() || token != drafts[i]) break;
        if (accepted_drafts) (*accepted_drafts)++;
    }
    return committed;
}

// One verify row against an argmax draft d (q one-hot): d is kept with probability min(1, p(d)/q(d)) = p(d), and a
// rejection draws from max(0, p - q) renormalised, which is p without d.
void TestSpeculativeAcceptProbabilityAndResidual() {
    const std::vector<double> p = {0.4, 0.3, 0.15, 0.1, 0.05};
    std::vector<float> row(16, -30.0f);
    for (int i = 0; i < 5; i++) row[i] = float(std::log(p[i]));
    const Target target = [&](const std::vector<int>&) { return row; };
    const int N = 60000;
    for (int d : {0, 1, 4}) {
        auto c = config_of(5, 1.0f, 2024 + d);
        Sampler s(16, c);
        int kept = 0;
        std::vector<int> rejected(5, 0);
        for (int n = 0; n < N; n++) {
            std::vector<int> ctx;
            const auto out = speculate_with(s, target, ctx, {d});
            if (out.size() == 2) {
                kept++;
            } else {
                CHECK(out.size() == 1 && out[0] != d && out[0] < 5, "draft %d: a rejection committed %d", d, out[0]);
                rejected[out[0]]++;
            }
        }
        const double q_d = 1.0;
        const double want_keep = std::min(1.0, p[d] / q_d);
        CHECK(std::fabs(kept / double(N) - want_keep) < 0.01, "draft %d kept %.4f of the time, want %.4f", d,
              kept / double(N), want_keep);
        std::vector<double> residual(5);
        double z = 0;
        for (int i = 0; i < 5; i++) z += residual[i] = std::max(0.0, p[i] - (i == d ? q_d : 0.0));
        for (auto& r : residual) r /= z;
        const double tvd = tv_distance(rejected, residual, N - kept);
        CHECK(tvd < 0.02, "draft %d: rejected tokens are %.4f from the residual in total variation", d, tvd);
    }
}

// top_k 1, or a temperature of 0, with no penalties: the batch is the longest prefix of drafts that are each row's
// argmax, then the argmax of the next row, which is what the engine commits without a chooser.
void TestGreedySpeculationKeepsTheArgmaxPrefix() {
    std::mt19937 rng(3);
    for (auto [top_k, T] : {std::pair{1, 0.7f}, std::pair{20, 0.0f}}) {
        auto c = config_of(top_k, T);
        Sampler s(1000, c);
        for (int cycle = 0; cycle < 200; cycle++) {
            std::vector<std::vector<float>> rows(5, std::vector<float>(1000));
            std::vector<int> argmax(5);
            for (int r = 0; r < 5; r++) {
                std::normal_distribution<float> normal(0.0f, 3.0f);
                for (auto& v : rows[r]) v = normal(rng);
                argmax[r] = int(std::max_element(rows[r].begin(), rows[r].end()) - rows[r].begin());
            }
            std::vector<int> drafts = {argmax[0], argmax[1], argmax[2], argmax[3]};
            const int wrong = cycle % 5;  // 4: every draft right
            if (wrong < 4) drafts[wrong] = (argmax[wrong] + 1) % 1000;
            std::vector<int> want(argmax.begin(), argmax.begin() + (wrong < 4 ? wrong + 1 : 5));
            std::vector<int> ctx;
            const Target target = [&](const std::vector<int>& x) { return rows[x.size()]; };
            const auto got = speculate_with(s, target, ctx, drafts);
            if (got != want) {
                CHECK(false, "top_k %d T %g cycle %d: %zu tokens, want %zu", top_k, T, cycle, got.size(), want.size());
                return;
            }
        }
    }
}

// A toy model whose next-token logits depend on the previous token, sampled with temperature, top-k and a presence
// penalty, against a draft that is the argmax of a different table (so it is sometimes wrong). The first three tokens
// speculative decoding commits are distributed exactly as when every token is sampled directly.
void TestSpeculativeSamplingMatchesDirectSampling() {
    const int V = 4;
    const float table[V + 1][V] = {
        {1.30f, 0.70f, 0.20f, -0.40f},   // after the start token
        {0.10f, 1.10f, 0.60f, -0.20f},
        {0.90f, -0.30f, 0.40f, 1.20f},
        {0.50f, 0.80f, -0.60f, 0.15f},
        {1.00f, 0.25f, 0.85f, -0.10f},
    };
    const int draft_of[V + 1] = {0, 2, 3, 1, 0};
    const float T = 0.7f, pre = 0.6f;
    const int top_k = 3;
    auto config = [&](uint64_t seed) {
        sampler_config c = config_of(top_k, T, seed);
        c.repeat_last_n = 64;
        c.pre_penalty = pre;
        return c;
    };
    // ctx holds the start token (V) then the reply
    const Target target = [&](const std::vector<int>& ctx) {
        const int prev = ctx.back();
        return std::vector<float>(table[prev], table[prev] + V);
    };

    // exact: the penalised logits, top-k, temperature and softmax, over every 3-token reply
    auto step_probs = [&](const std::vector<int>& ctx) {
        std::vector<double> l(V);
        for (int t = 0; t < V; t++) {
            l[t] = table[ctx.back()][t];
            if (std::find(ctx.begin() + 1, ctx.end(), t) != ctx.end()) l[t] -= pre;
        }
        std::vector<int> order(V);
        for (int t = 0; t < V; t++) order[t] = t;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return l[a] > l[b]; });
        std::vector<double> pr(V, 0.0);
        double z = 0;
        for (int r = 0; r < top_k; r++) z += pr[order[r]] = std::exp(l[order[r]] / T);
        for (auto& x : pr) x /= z;
        return pr;
    };
    std::vector<double> exact(V * V * V);
    for (int a = 0; a < V; a++) {
        for (int b = 0; b < V; b++) {
            for (int c = 0; c < V; c++) {
                exact[(a * V + b) * V + c] =
                    step_probs({V})[a] * step_probs({V, a})[b] * step_probs({V, a, b})[c];
            }
        }
    }

    const int N = 60000;
    std::vector<int> spec(V * V * V, 0), direct(V * V * V, 0);
    int accepted = 0, cycles = 0;
    sampler_config cs = config(5), cd = config(6);
    Sampler ss(V, cs), sd(V, cd);
    for (int n = 0; n < N; n++) {
        ss.reset_penalties();
        std::vector<int> ctx = {V};
        while (ctx.size() < 4) {
            // the head drafts greedily from its own table, two ahead
            std::vector<int> drafts = {draft_of[ctx.back()]};
            drafts.push_back(draft_of[drafts[0]]);
            speculate_with(ss, target, ctx, drafts, &accepted);
            cycles++;
        }
        spec[(ctx[1] * V + ctx[2]) * V + ctx[3]]++;

        sd.reset_penalties();
        std::vector<int> d = {V};
        while (d.size() < 4) {
            const auto row = target(d);
            d.push_back(sd.sample(row.data()));
        }
        direct[(d[1] * V + d[2]) * V + d[3]]++;
    }
    const double tv_spec = tv_distance(spec, exact, N);
    const double tv_direct = tv_distance(direct, exact, N);
    CHECK(tv_spec < 0.02, "speculative replies are %.4f from the exact distribution in total variation", tv_spec);
    CHECK(tv_direct < 0.02, "direct replies are %.4f from the exact distribution in total variation", tv_direct);
    // neither every draft kept nor every draft rejected, or the comparison says nothing about the residual
    CHECK(accepted > N / 4 && accepted < 2 * cycles - N / 4, "%d drafts kept over %d cycles", accepted, cycles);
}

// A seed gives the same speculative reply twice: the sampler's seeded draw is the only randomness.
void TestSeededSpeculationIsReproducible() {
    std::mt19937 rng(9);
    std::vector<std::vector<float>> rows(64, std::vector<float>(256));
    std::normal_distribution<float> normal(0.0f, 1.5f);
    for (auto& r : rows) for (auto& v : r) v = normal(rng);
    const Target target = [&](const std::vector<int>& ctx) { return rows[ctx.size() % rows.size()]; };
    std::vector<int> replies[2];
    for (auto& reply : replies) {
        sampler_config c = config_of(20, 0.7f, 4242);
        c.repeat_last_n = 64;
        c.pre_penalty = 1.5f;
        Sampler s(256, c);
        while (reply.size() < 40) {
            std::vector<int> drafts;
            for (int i = 0; i < 3; i++) {
                const auto& r = rows[(reply.size() + i) % rows.size()];
                drafts.push_back(int(std::max_element(r.begin(), r.end()) - r.begin()));
            }
            speculate_with(s, target, reply, drafts);
        }
    }
    CHECK(replies[0] == replies[1], "the same seed gave two different speculative replies");
}

}  // namespace

int main(int argc, char** argv) {
    // one test per process (ctest runs them that way), so a crash names its test
    const std::string only = argc > 1 ? argv[1] : "";
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"TestGreedyTieTakesTheSmallestId", TestGreedyTieTakesTheSmallestId},
        {"TestTopKOfZeroOrBelowIsOffAndLeavesNoStaleCandidates", TestTopKOfZeroOrBelowIsOffAndLeavesNoStaleCandidates},
        {"TestSamplerBuiltWithAnOutOfRangeTopKDoesNotCrash", TestSamplerBuiltWithAnOutOfRangeTopKDoesNotCrash},
        {"TestTinyTemperatureIsGreedy", TestTinyTemperatureIsGreedy},
        {"TestSameSeedSameSequence", TestSameSeedSameSequence},
        {"TestSeededSequenceIsPortable", TestSeededSequenceIsPortable},
        {"TestUniformDrawsArePortable", TestUniformDrawsArePortable},
        {"TestPenaltyFormulas", TestPenaltyFormulas},
        {"TestTopPAndMinPDistributions", TestTopPAndMinPDistributions},
        {"TestTopKOffMatchesTheHeapPath", TestTopKOffMatchesTheHeapPath},
        {"TestSignedZerosTieByIdWithTopKOff", TestSignedZerosTieByIdWithTopKOff},
        {"TestNaNLogitFallsBackToTheHeapPath", TestNaNLogitFallsBackToTheHeapPath},
        {"TestFloatRowMatchesTheBf16Buffer", TestFloatRowMatchesTheBf16Buffer},
        {"TestSpeculativeAcceptProbabilityAndResidual", TestSpeculativeAcceptProbabilityAndResidual},
        {"TestGreedySpeculationKeepsTheArgmaxPrefix", TestGreedySpeculationKeepsTheArgmaxPrefix},
        {"TestSpeculativeSamplingMatchesDirectSampling", TestSpeculativeSamplingMatchesDirectSampling},
        {"TestSeededSpeculationIsReproducible", TestSeededSpeculationIsReproducible},
    };
    int failed_tests = 0;
#if USEAVX2
    std::printf("build: AVX2\n");
#else
    std::printf("build: scalar\n");
#endif
    for (const auto& [name, fn] : tests) {
        if (!only.empty() && only != name) continue;
        const int before = g_failures;
        g_failure_text.clear();
        fn();
        if (g_failures == before) {
            std::printf("PASS %s\n", name);
        } else {
            failed_tests++;
            std::printf("FAIL %s%s\n", name, g_failure_text.c_str());
        }
        std::fflush(stdout);
    }
    std::printf("%d of %zu tests failed\n", failed_tests, tests.size());
    return failed_tests == 0 ? 0 : 1;
}
