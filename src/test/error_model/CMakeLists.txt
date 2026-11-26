# error_model tests: no NPU, no XRT, no Boost, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/error_model -B build-error-model && cmake --build build-error-model && ctest --test-dir build-error-model
cmake_minimum_required(VERSION 3.22)
project(error_model_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_error_model test_error_model.cpp)
target_include_directories(test_error_model PRIVATE "${FLM_SOURCE_DIR}/include" "${FLM_SOURCE_DIR}/server")

enable_testing()
add_test(NAME test_error_model COMMAND test_error_model)
