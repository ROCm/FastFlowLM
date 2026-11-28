# special_token_ids: the BOS id comes from the tokenizer when tokenizer_config.json states another (#1171).
# Builds FastFlowLM's Tokenizer with tokenizers-cpp (needs cargo and the third_party/tokenizers-cpp submodule). No NPU:
# the stubs directory shadows the device runtime, as in src/test/ignore_eos.
#   cmake -S src/test/special_token_ids -B build-special-token-ids && cmake --build build-special-token-ids
#   ctest --test-dir build-special-token-ids
cmake_minimum_required(VERSION 3.22)
project(special_token_ids_tests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
add_subdirectory("${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp" tokenizers EXCLUDE_FROM_ALL)

add_executable(test_special_token_ids
    test_special_token_ids.cpp
    "${FLM_SOURCE_DIR}/common/tokenizer/tokenizer.cpp")
target_include_directories(test_special_token_ids PRIVATE
    "${FLM_SOURCE_DIR}/test/model_frontend/stubs"
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include")
target_link_libraries(test_special_token_ids PRIVATE tokenizers_cpp)

enable_testing()
add_test(NAME test_special_token_ids
         COMMAND test_special_token_ids "${CMAKE_CURRENT_LIST_DIR}/fixtures")
