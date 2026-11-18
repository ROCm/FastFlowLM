# Tool calling without a model: the output parser on what models write, and the decoding grammar
# against the calls the chat templates render. No NPU, no XRT, so this builds and runs on Linux CI
# as well as Windows.
#   cmake -S src/test/chat_tools -B build-ct && cmake --build build-ct && ctest --test-dir build-ct
cmake_minimum_required(VERSION 3.22)
project(chat_tools_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")

add_executable(test_chat_tools test_chat_tools.cpp ${FLM_CHAT_SOURCES})
target_include_directories(test_chat_tools PRIVATE "${FLM_SOURCE_DIR}/include")
target_link_libraries(test_chat_tools PRIVATE flm_xgrammar)
if(MSVC)
    target_compile_options(test_chat_tools PRIVATE /utf-8 /bigobj)
endif()

enable_testing()
add_test(NAME test_chat_tools COMMAND test_chat_tools)
