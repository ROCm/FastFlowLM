// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// buf_life.cpp <mode> <rounds>: lifecycle test of 2 MB-page device buffers (FLM_HUGEPAGE_BO=<mode>).
// Each round makes owning device buffers (large and small), moves, copies and move-assigns them, writes
// through the mapping, then drops them all. Prints RSS and AnonHugePages after each round: with the
// memory freed both stay flat; the leaking experiment grew by the buffers' size every round.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "buffer.hpp"

static void mem(const char* tag) {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string line;
    long rss = 0, huge = 0;
    while (std::getline(f, line)) {
        if (line.rfind("Rss:", 0) == 0) rss = std::atol(line.c_str() + 4);
        if (line.rfind("AnonHugePages:", 0) == 0) huge = std::atol(line.c_str() + 14);
    }
    std::printf("%-10s rss %7ld MB  anon_huge %7ld MB\n", tag, rss / 1024, huge / 1024);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    setenv("FLM_HUGEPAGE_BO", argc > 1 ? argv[1] : "1", 1);
    int rounds = argc > 2 ? std::atoi(argv[2]) : 10;
    xrt::device dev(0);
    mem("start");
    for (int r = 0; r < rounds; ++r) {
        {
            std::vector<bytes> keep;
            for (size_t sz : {64u << 20, 24u << 20, 5u << 20, 3u << 20, 4096u, 100u << 10}) {
                bytes b(dev, sz);
                std::memset(b.data(), r & 0xff, sz);
                bytes alias(b);                 // non-owning copy
                bytes moved(std::move(b));      // owner moves
                keep.push_back(std::move(moved));
                (void)alias;
            }
            bytes a(dev, 8u << 20), c(dev, 6u << 20);
            a = std::move(c);                   // move-assign frees a's old memory
            buffer<uint16_t> t(dev, 1u << 20);  // typed buffer, 2 MB (free() is for host memory only)
            keep.front().sync_to_device();
            if (r == 0) mem("held");
        }
        char tag[32];
        std::snprintf(tag, sizeof tag, "round %d", r);
        mem(tag);
    }
    return 0;
}
