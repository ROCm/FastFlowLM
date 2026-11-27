# Local OGA model folder tests (flm serve --model-dir): the catalog entry built from an OGA package folder and
# the way model_list takes it. No GPU, no NPU, no XRT runtime library, so this builds and runs on Linux CI as
# well as Windows.
cmake_minimum_required(VERSION 3.22)
project(local_model_oga_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_local_model_oga
    test_local_model_oga.cpp
    "${FLM_SOURCE_DIR}/common/npu_platform.cpp")
target_include_directories(test_local_model_oga PRIVATE
    "${FLM_SOURCE_DIR}/include")
target_compile_definitions(test_local_model_oga PRIVATE
    FLM_TEST_MODEL_LIST_PATH="${FLM_SOURCE_DIR}/model_list.json")

enable_testing()
add_test(NAME test_local_model_oga COMMAND test_local_model_oga)
