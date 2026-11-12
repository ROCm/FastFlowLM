# effective_top_k: what a request's top_k means (0 and below turn top-k off). Header only, no NPU.
#   cmake -S src/test/top_k -B build-top-k && cmake --build build-top-k && ctest --test-dir build-top-k
cmake_minimum_required(VERSION 3.22)
project(top_k_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_top_k test_top_k.cpp)
target_include_directories(test_top_k PRIVATE "${FLM_SOURCE_DIR}/include")

enable_testing()
add_test(NAME test_top_k COMMAND test_top_k)
