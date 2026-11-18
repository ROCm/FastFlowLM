// effective_top_k: 0 and below mean "off" (the whole vocabulary); the rest is clamped to [1, vocab].
#include "modules/top_k.hpp"

#include <cstdio>

static int g_failures = 0;

#define CHECK_EQ(actual, expected)                                                          \
    do {                                                                                    \
        const int a_ = (actual);                                                            \
        const int e_ = (expected);                                                          \
        if (a_ != e_) {                                                                     \
            std::printf("FAIL %s:%d: %s = %d, expected %d\n", __FILE__, __LINE__, #actual, a_, e_); \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

int main() {
    constexpr int V = 151936;
    CHECK_EQ(effective_top_k(1, V), 1);          // greedy
    CHECK_EQ(effective_top_k(20, V), 20);
    CHECK_EQ(effective_top_k(V, V), V);
    CHECK_EQ(effective_top_k(V + 1, V), V);      // above the vocabulary: clamped
    CHECK_EQ(effective_top_k(0, V), V);          // off
    CHECK_EQ(effective_top_k(-1, V), V);         // off (vLLM's spelling)
    CHECK_EQ(effective_top_k(-1000, V), V);
    if (g_failures == 0) {
        std::printf("test_top_k: PASS\n");
    }
    return g_failures == 0 ? 0 : 1;
}
