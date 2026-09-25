# Catalog-only tests: no NPU, no XRT runtime library, so this builds and runs on
# Linux CI as well as Windows.
cmake_minimum_required(VERSION 3.22)
project(model_list_platform_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_model_list_platform
    test_model_list_platform.cpp
    "${FLM_SOURCE_DIR}/common/npu_platform.cpp")
target_include_directories(test_model_list_platform PRIVATE
    "${FLM_SOURCE_DIR}/include")
target_compile_definitions(test_model_list_platform PRIVATE
    FLM_TEST_MODEL_LIST_PATH="${FLM_SOURCE_DIR}/model_list.json")

enable_testing()
add_test(NAME test_model_list_platform COMMAND test_model_list_platform)
