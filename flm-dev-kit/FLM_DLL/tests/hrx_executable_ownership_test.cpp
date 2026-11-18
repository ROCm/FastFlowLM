// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// Host-only lifetime tests for FastFlowLM's HRX executable ownership wrappers.

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "hrx_cpp/hrx_cpp.hpp"

struct hrx_executable_s {
    int references = 1;
};

extern "C" void hrx_executable_retain(hrx_executable_t executable) {
    ++executable->references;
}

extern "C" void hrx_executable_release(hrx_executable_t executable) {
    --executable->references;
}

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #condition,          \
                         __LINE__);                                             \
            ++failures;                                                        \
        }                                                                      \
    } while (false)

void test_current_generation_ownership() {
    hrx_executable_s first;
    hrx_executable_s second;

    {
        hrx::executable_owner owner;
        CHECK(owner.acquire(nullptr) == nullptr);

        owner.replace(&first, 7, 13);
        CHECK(owner.matches(7));
        CHECK(!owner.matches(8));
        CHECK(first.references == 1);

        uint32_t ordinal = 0;
        hrx_executable_t acquired = owner.acquire(&ordinal);
        CHECK(acquired == &first);
        CHECK(ordinal == 13);
        CHECK(first.references == 2);
        hrx_executable_release(acquired);
        CHECK(first.references == 1);

        owner.replace(&second, 8, 17);
        CHECK(first.references == 0);
        CHECK(second.references == 1);

        hrx::executable_owner moved(std::move(owner));
        CHECK(owner.acquire(nullptr) == nullptr);
        CHECK(moved.matches(8));

        hrx::executable_owner assigned;
        assigned = std::move(moved);
        CHECK(moved.acquire(nullptr) == nullptr);
        CHECK(assigned.matches(8));
        CHECK(second.references == 1);
    }

    CHECK(second.references == 0);
}

void test_in_flight_run_references() {
    hrx_executable_s executable;

    {
        hrx::run original(&executable, 23);
        CHECK(executable.references == 1);

        {
            hrx::run copied(original);
            CHECK(executable.references == 2);

            hrx::run moved(std::move(copied));
            CHECK(executable.references == 2);

            hrx::run copy_assigned;
            copy_assigned = original;
            CHECK(executable.references == 3);

            hrx::run move_assigned;
            move_assigned = std::move(copy_assigned);
            CHECK(executable.references == 3);
        }

        CHECK(executable.references == 1);
    }

    CHECK(executable.references == 0);
}

void test_xclbin_content_identity() {
    const std::vector<uint8_t> first = {1, 2, 3, 4};
    const std::vector<uint8_t> same = first;
    const std::vector<uint8_t> different = {1, 2, 3, 5};

    const auto first_identity = hrx::compute_xclbin_identity(first);
    const auto same_identity = hrx::compute_xclbin_identity(same);
    const auto different_identity = hrx::compute_xclbin_identity(different);
    CHECK(first_identity.hash_1 == same_identity.hash_1);
    CHECK(first_identity.hash_2 == same_identity.hash_2);
    CHECK(first_identity.data_length == first.size());
    CHECK(first_identity.hash_1 != different_identity.hash_1);
    CHECK(first_identity.hash_2 != different_identity.hash_2);
}

}  // namespace

int main() {
    test_current_generation_ownership();
    test_in_flight_run_references();
    test_xclbin_content_identity();
    if (failures == 0) {
        std::puts("hrx_executable_ownership_test: PASS");
        return 0;
    }
    std::fprintf(stderr, "hrx_executable_ownership_test: %d FAILURE(S)\n",
                 failures);
    return 1;
}
