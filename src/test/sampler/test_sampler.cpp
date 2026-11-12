// Sampler (src/common/modules/sampler.cpp) on the real code, no NPU. Built twice: scalar, and with USEAVX2=1.
#include "modules/sampler.hpp"

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
