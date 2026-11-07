# Gemma4 tool call arguments: quotes inside <|"|> strings (FastFlowLM#900), the plain-quote closers the reader
# still accepts, and streaming. No NPU, no XRT, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/chat_gemma4_args -B build-g4a && cmake --build build-g4a && ctest --test-dir build-g4a
cmake_minimum_required(VERSION 3.22)
project(chat_gemma4_args_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_chat_gemma4_args
    test_chat_gemma4_args.cpp
    "${FLM_SOURCE_DIR}/common/chat/chat_args.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_format.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_msg.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_parser.cpp"
    "${FLM_SOURCE_DIR}/common/chat/chat_stream.cpp")
target_include_directories(test_chat_gemma4_args PRIVATE "${FLM_SOURCE_DIR}/include")
if(MSVC)
    target_compile_options(test_chat_gemma4_args PRIVATE /utf-8 /bigobj)
endif()

enable_testing()
add_test(NAME test_chat_gemma4_args COMMAND test_chat_gemma4_args)
