# Sampler tests (src/common/modules/sampler.cpp): ties, top-k, temperature, seeds and the penalty formulas on the
# real sampler. No NPU and no XRT: the stub runtime headers of test/model_frontend stand in for the device runtime,
# so this builds and runs on Linux CI as well as on Windows.
#   cmake -S src/test/sampler -B build-sampler && cmake --build build-sampler && ctest --test-dir build-sampler
#
# The sampler has two code paths for the greedy scan: scalar, and AVX2 (-DUSEAVX2=1, which the NPU test builds use
# and the release build does not). Both are built and both must pass.
cmake_minimum_required(VERSION 3.22)
project(sampler_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

# Keep in step with the table in main() of test_sampler.cpp; each case runs in its own process.
set(SAMPLER_TEST_CASES
    TestGreedyTieTakesTheSmallestId
    TestTopKOfZeroOrBelowIsOffAndLeavesNoStaleCandidates
    TestSamplerBuiltWithAnOutOfRangeTopKDoesNotCrash
    TestTinyTemperatureIsGreedy
    TestSameSeedSameSequence
    TestSeededSequenceIsPortable
    TestUniformDrawsArePortable
    TestPenaltyFormulas
    TestTopPAndMinPDistributions)

function(add_sampler_test name)
    add_executable(${name} test_sampler.cpp "${FLM_SOURCE_DIR}/common/modules/sampler.cpp")
    # the stubs directory first: it shadows the device runtime header
    target_include_directories(${name} PRIVATE
        "${FLM_SOURCE_DIR}/test/model_frontend/stubs"
        "${FLM_SOURCE_DIR}/include")
    if(WIN32)
        target_compile_definitions(${name} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    foreach(case IN LISTS SAMPLER_TEST_CASES)
        add_test(NAME ${name}.${case} COMMAND ${name} ${case})
    endforeach()
endfunction()

enable_testing()

add_sampler_test(test_sampler_scalar)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_sampler_test(test_sampler_avx2)
    target_compile_definitions(test_sampler_avx2 PRIVATE USEAVX2=1)
    if(MSVC)
        target_compile_options(test_sampler_avx2 PRIVATE /arch:AVX2)
    else()
        target_compile_options(test_sampler_avx2 PRIVATE -mavx2)
    endif()
endif()
