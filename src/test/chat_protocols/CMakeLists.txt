# Chat protocols without a model: the events of a chat turn (src/server/chat_core.hpp) through a
# protocol's sink must give the bytes recorded in golden/. No NPU, no XRT, so this builds and runs
# on Linux CI as well as Windows.
#   cmake -S src/test/chat_protocols -B build-cp && cmake --build build-cp && ctest --test-dir build-cp
cmake_minimum_required(VERSION 3.22)
project(chat_protocols_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_chat_protocols
    test_chat_protocols.cpp
    test_anthropic.cpp
    "${FLM_SOURCE_DIR}/common/chat/chat_args.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_continue.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_format.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_msg.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_parser.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_stream.cpp")
target_include_directories(test_chat_protocols PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/server")
target_compile_definitions(test_chat_protocols PRIVATE GOLDEN_DIR="${CMAKE_CURRENT_LIST_DIR}/golden")
if(MSVC)
    target_compile_options(test_chat_protocols PRIVATE /utf-8 /bigobj)
endif()

enable_testing()
add_test(NAME test_chat_protocols COMMAND test_chat_protocols)
