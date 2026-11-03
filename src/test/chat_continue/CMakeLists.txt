# continue_final_message tests: the prompt cut on real template shapes, and the output parser for a continued message.
# No NPU, no XRT, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/chat_continue -B build-cc && cmake --build build-cc && ctest --test-dir build-cc
cmake_minimum_required(VERSION 3.22)
project(chat_continue_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_chat_continue
    test_chat_continue.cpp
    "${FLM_SOURCE_DIR}/common/chat/chat_args.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_continue.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_format.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_msg.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_parser.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_stream.cpp")
target_include_directories(test_chat_continue PRIVATE "${FLM_SOURCE_DIR}/include")
if(MSVC)
    # the DeepSeek markers below are UTF-8 literals
    target_compile_options(test_chat_continue PRIVATE /utf-8 /bigobj)
endif()

enable_testing()
add_test(NAME test_chat_continue COMMAND test_chat_continue)
