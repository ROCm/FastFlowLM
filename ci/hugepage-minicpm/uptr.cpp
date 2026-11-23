// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// uptr.cpp: does a user-pointer xrt::ext::bo over 2 MB-aligned anonymous memory work here, and does it get huge pages?
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include "xrt/xrt_device.h"
#include "xrt/experimental/xrt_ext.h"

static long huge_kb() {
    std::ifstream f("/proc/self/smaps_rollup");
    std::string l;
    while (std::getline(f, l))
        if (l.rfind("AnonHugePages:", 0) == 0) return std::atol(l.c_str() + 14);
    return -1;
}

int main() {
    const size_t huge = 2u << 20, n = 64u << 20;
    void* raw = mmap(nullptr, n + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char* p = (char*)(((uintptr_t)raw + huge - 1) / huge * huge);
    std::printf("madvise %d\n", madvise(p, n, MADV_HUGEPAGE));
    std::memset(p, 0, n);
    std::printf("after memset anon_huge %ld kB\n", huge_kb());
    xrt::device dev(0);
    try {
        xrt::ext::bo bo(dev, p, n);
        std::printf("user-ptr bo ok, anon_huge %ld kB\n", huge_kb());
    } catch (const std::exception& e) {
        std::printf("user-ptr bo threw: %s\n", e.what());
    }
    return 0;
}
