// Unit test for the HRX executable-cache key (hrx_cpp.hpp).
//
// Verifies the properties the exe-cache relies on after keying by a
// once-computed content digest instead of copying/hashing the whole xclbin
// image on every create_run:
//   1. Identical images produce an identical digest and key (real cache hit).
//   2. Distinct images produce distinct digests and keys (no false hit) --
//      including two images that happen to reuse the SAME heap address, which
//      the previous address-keyed registry could have confused.
//   3. The same image with different control codes keys differently.
//
// These functions are pure (no NPU/driver dependency), so this builds and runs
// on a host without hardware. Compile with the FLM include path, e.g.:
//   cl /std:c++17 /EHsc /I src/include /I <hrx-sdk-include> \
//      src/tests/hrx_exe_key_test.cpp
//   c++ -std=c++17 -I src/include -I <hrx-sdk-include> \
//      src/tests/hrx_exe_key_test.cpp -o hrx_exe_key_test

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "hrx_cpp/hrx_cpp.hpp"

namespace {

std::vector<uint8_t> make_image(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<uint8_t>((i * 131u + seed * 7u + 17u) & 0xff);
    return v;
}

int failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);     \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

}  // namespace

int main() {
    const uint32_t cc_a[] = {1, 2, 3, 4};
    const uint32_t cc_b[] = {1, 2, 3, 5};

    // 1. Same content -> same digest and same key.
    auto img1 = make_image(4096, 0xA5);
    auto img1_copy = img1;
    auto d1 = hrx::compute_xclbin_digest(img1);
    auto d1c = hrx::compute_xclbin_digest(img1_copy);
    CHECK(d1 == d1c);
    CHECK(hrx::make_exe_key(img1.size(), d1, cc_a, 4) ==
          hrx::make_exe_key(img1_copy.size(), d1c, cc_a, 4));

    // 2a. Different content, same size -> different digest and key.
    auto img2 = make_image(4096, 0x5A);
    auto d2 = hrx::compute_xclbin_digest(img2);
    CHECK(!(d1 == d2));
    CHECK(hrx::make_exe_key(img1.size(), d1, cc_a, 4) !=
          hrx::make_exe_key(img2.size(), d2, cc_a, 4));

    // 2b. One differing byte flips the digest (and key).
    auto img3 = img1;
    img3[2048] ^= 0x01;
    auto d3 = hrx::compute_xclbin_digest(img3);
    CHECK(!(d1 == d3));
    CHECK(hrx::make_exe_key(img1.size(), d1, cc_a, 4) !=
          hrx::make_exe_key(img3.size(), d3, cc_a, 4));

    // 2c. Two distinct images allocated (sequentially) at the SAME address must
    // still key differently -- the digest is content-derived, not address-
    // derived, so the old registry-reuse hazard cannot produce a false hit.
    hrx::xclbin_digest da, db;
    const void* addr_a = nullptr;
    const void* addr_b = nullptr;
    {
        auto a = std::make_unique<std::vector<uint8_t>>(make_image(2048, 0x11));
        addr_a = static_cast<const void*>(a.get());
        da = hrx::compute_xclbin_digest(*a);
    }  // freed; next allocation may reuse the address
    {
        auto b = std::make_unique<std::vector<uint8_t>>(make_image(2048, 0x22));
        addr_b = static_cast<const void*>(b.get());
        db = hrx::compute_xclbin_digest(*b);
    }
    (void)addr_a;
    (void)addr_b;  // address reuse is irrelevant to the key now
    CHECK(!(da == db));
    CHECK(hrx::make_exe_key(2048, da, cc_a, 4) !=
          hrx::make_exe_key(2048, db, cc_a, 4));

    // 3. Same image, different control codes -> different key.
    CHECK(hrx::make_exe_key(img1.size(), d1, cc_a, 4) !=
          hrx::make_exe_key(img1.size(), d1, cc_b, 4));

    if (failures == 0) {
        std::puts("hrx_exe_key_test: PASS");
        return 0;
    }
    std::fprintf(stderr, "hrx_exe_key_test: %d FAILURE(S)\n", failures);
    return 1;
}
