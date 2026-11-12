// Sampler (src/common/modules/sampler.cpp) on the real code, no NPU. Built twice: scalar, and with USEAVX2=1.
#include "modules/sampler.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
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
