# http_guard tests: no NPU, no XRT, no Boost, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/http_guard -B build-http-guard && cmake --build build-http-guard && ctest --test-dir build-http-guard
cmake_minimum_required(VERSION 3.22)
project(http_guard_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_http_guard test_http_guard.cpp)
target_include_directories(test_http_guard PRIVATE "${FLM_SOURCE_DIR}/include" "${FLM_SOURCE_DIR}/server")
# The route table test reads server.cpp: every engine route must take the NPU queue.
target_compile_definitions(test_http_guard PRIVATE FLM_SERVER_CPP="${FLM_SOURCE_DIR}/server/server.cpp")

enable_testing()
add_test(NAME test_http_guard COMMAND test_http_guard)
