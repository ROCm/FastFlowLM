# Message roles without a model: the probe of what a chat template does with `developer` and with a `system`
# message after the first turn, on the Qwen3 and Qwen3.5 templates the packages ship (templates/), and the
# normalization the server applies before every template. No NPU, no XRT, so this builds and runs on Linux CI
# as well as Windows.
#   cmake -S src/test/message_roles -B build-mr && cmake --build build-mr && ctest --test-dir build-mr
cmake_minimum_required(VERSION 3.22)
project(message_roles_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_message_roles test_message_roles.cpp)
target_include_directories(test_message_roles PRIVATE "${FLM_SOURCE_DIR}/include")
target_compile_definitions(test_message_roles PRIVATE
    MESSAGE_ROLES_TEMPLATE_DIR="${CMAKE_CURRENT_LIST_DIR}/templates")
if(MSVC)
    target_compile_options(test_message_roles PRIVATE /utf-8 /bigobj)
endif()

enable_testing()
add_test(NAME test_message_roles COMMAND test_message_roles)
