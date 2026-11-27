# Inkling-2 image preprocessing (cell grid, padding, normalization, cell order). No NPU, no XRT and no weights, so this
# builds and runs on an AVX-512 BF16 CPU (the engine's own target).
#   cmake -S src/test/inkling2_vision -B build-i2v && cmake --build build-i2v && ctest --test-dir build-i2v
cmake_minimum_required(VERSION 3.22)
project(inkling2_vision_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../flm-dev-kit/FLM_DLL/detail/inkling2_npu")

find_package(OpenMP REQUIRED)
add_executable(test_inkling2_vision test_inkling2_vision.cpp "${ENGINE_DIR}/inkling2_vision.cpp")
target_include_directories(test_inkling2_vision PRIVATE "${ENGINE_DIR}" "${ENGINE_DIR}/../../include")
target_compile_options(test_inkling2_vision PRIVATE -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512bf16 -mfma)
target_link_libraries(test_inkling2_vision PRIVATE OpenMP::OpenMP_CXX)

enable_testing()
add_test(NAME test_inkling2_vision COMMAND test_inkling2_vision)
