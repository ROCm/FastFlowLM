# Registry-only tests: no NPU, runtime headers, or prebuilt engine library.
cmake_minimum_required(VERSION 3.22)
project(model_backend_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_model_backend
    test_model_backend.cpp
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp")
target_include_directories(test_model_backend PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../model_frontend/stubs"
    "${FLM_SOURCE_DIR}/include")
if(WIN32)
    target_compile_definitions(test_model_backend PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

enable_testing()
add_test(NAME test_model_backend COMMAND test_model_backend)
