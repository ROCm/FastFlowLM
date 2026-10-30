// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/worker.hpp"

#include "../phi4_core/test_support.hpp"
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

void SetEnvironment(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}  // namespace

int main() {
    SetEnvironment("FLM_IRON_WORKER_MODULE", "fake_worker");

    flm::iron::Worker llama("/unused", "llama", 32);
    const auto logits = llama.logits(std::vector<int>{4, 10, 26});
    TEST_REQUIRE(logits == std::vector<std::uint16_t>({26, 3}));
    llama.reset();
    TEST_REQUIRE(!llama.poisoned());

    flm::iron::Worker embedding("/unused", "embedding_gemma2");
    const auto result = embedding.embed("hello", "query", 128);
    TEST_REQUIRE(result == std::vector<float>({5.0f, 128.0f}));

    for (const char* model : {"bad_count", "bad_header", "error", "truncated"}) {
        flm::iron::Worker broken("/unused", model);
        RequireThrows([&] { broken.logits({1}); });
        TEST_REQUIRE(broken.poisoned());
        RequireContains(RequireThrows([&] { broken.reset(); }), "reload");
    }

    bool pipe_failed = false;
    try {
        flm::iron::Worker exits("/unused", "dies");
        exits.reset();
    } catch (const std::runtime_error&) {
        pipe_failed = true;
    }
    TEST_REQUIRE(pipe_failed);

    bool failed = false;
    SetEnvironment("FLM_IRON_WORKER_MODULE", "missing_worker");
    try {
        flm::iron::Worker dead("/unused", "llama", 32);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    TEST_REQUIRE(failed);
    return 0;
}
