# Local model folder tests (flm serve --model-dir): the catalog entry built from a folder and the way
# model_list takes it. No NPU, no XRT runtime library, so this builds and runs on Linux CI as well as Windows.
cmake_minimum_required(VERSION 3.22)
project(local_model_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_local_model
    test_local_model.cpp
    "${FLM_SOURCE_DIR}/common/npu_platform.cpp")
target_include_directories(test_local_model PRIVATE
    "${FLM_SOURCE_DIR}/include")
target_compile_definitions(test_local_model PRIVATE
    FLM_TEST_MODEL_LIST_PATH="${FLM_SOURCE_DIR}/model_list.json")

enable_testing()
add_test(NAME test_local_model COMMAND test_local_model)
