# image_raw (raw RGB8 frames) request parsing. No NPU, no XRT, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/raw_rgb8 -B build-raw && cmake --build build-raw && ctest --test-dir build-raw
cmake_minimum_required(VERSION 3.22)
project(raw_rgb8_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_raw_rgb8 test_raw_rgb8.cpp)
target_include_directories(test_raw_rgb8 PRIVATE "${FLM_SOURCE_DIR}/include")

enable_testing()
add_test(NAME test_raw_rgb8 COMMAND test_raw_rgb8)
