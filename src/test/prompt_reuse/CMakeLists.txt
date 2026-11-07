# plan_prompt_reuse tests: no NPU, no XRT, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/prompt_reuse -B build-prompt-reuse && cmake --build build-prompt-reuse && ctest --test-dir build-prompt-reuse
cmake_minimum_required(VERSION 3.22)
project(prompt_reuse_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_prompt_reuse test_prompt_reuse.cpp)
target_include_directories(test_prompt_reuse PRIVATE "${FLM_SOURCE_DIR}/include")

enable_testing()
add_test(NAME test_prompt_reuse COMMAND test_prompt_reuse)
