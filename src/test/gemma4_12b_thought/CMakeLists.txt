# Host-only check of when Gemma4 12B trims the empty thought channel.
# No NPU, no XRT, so this builds on the CPU Tests runner.
#   cmake -S src/test/gemma4_12b_thought -B build-gemma4-12b-thought
#   cmake --build build-gemma4-12b-thought && ctest --test-dir build-gemma4-12b-thought
cmake_minimum_required(VERSION 3.22)
project(gemma4_12b_thought_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_gemma4_12b_thought test_gemma4_12b_thought.cpp)
target_include_directories(test_gemma4_12b_thought PRIVATE "${FLM_SOURCE_DIR}/include")
if(MSVC)
    target_compile_options(test_gemma4_12b_thought PRIVATE /utf-8)
endif()

enable_testing()
add_test(NAME test_gemma4_12b_thought COMMAND test_gemma4_12b_thought)
