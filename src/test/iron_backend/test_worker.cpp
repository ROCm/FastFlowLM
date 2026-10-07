// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "iron/worker.hpp"

#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <vector>

int main() {
    setenv("FLM_IRON_WORKER_MODULE", "fake_worker", 1);

    flm::iron::Worker llama("/unused", 32);
    const auto logits = llama.logits(std::vector<int>{4, 7, 9});
    assert(logits == std::vector<std::uint16_t>({9, 3}));
    llama.reset();
    assert(!llama.poisoned());

    flm::iron::Worker embedding("/unused", 512, "embedding");
    const auto result = embedding.embed("hello", "query", 128);
    assert(result == std::vector<float>({5.0f, 128.0f}));

    bool failed = false;
    setenv("FLM_IRON_WORKER_MODULE", "missing_worker", 1);
    try {
        flm::iron::Worker dead("/unused", 32);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
    return 0;
}