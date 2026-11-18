# Hardware-free model extension contract tests.
cmake_minimum_required(VERSION 3.22)
project(model_extension_contract_tests LANGUAGES CXX)

enable_testing()

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../model_frontend"
    model_frontend)
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../model_list_platform"
    model_list_platform)
add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../model_backend"
    model_backend)
