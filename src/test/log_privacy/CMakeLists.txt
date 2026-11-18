# log_privacy tests: what the server log and the model downloads do by default (WP-15).
# No NPU, no XRT, no Boost, no libcurl to link, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/log_privacy -B build-log-privacy && cmake --build build-log-privacy && ctest --test-dir build-log-privacy
cmake_minimum_required(VERSION 3.22)
project(log_privacy_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(FLM_REPO_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")

# The switches, the request-body redaction and the download TLS settings (headers only; curl.h for its constants).
add_executable(test_log_privacy test_log_privacy.cpp)
target_include_directories(test_log_privacy PRIVATE "${FLM_SOURCE_DIR}/include" "${FLM_SOURCE_DIR}/pull"
                           "${FLM_REPO_DIR}/third_party/curl/include")

# Source checks: every "Model RAW Output" dump goes through one helper, no transfer turns TLS verification off.
add_executable(test_log_privacy_sources test_log_privacy_sources.cpp)
target_compile_definitions(test_log_privacy_sources PRIVATE FLM_SOURCE_DIR="${FLM_SOURCE_DIR}")

enable_testing()
add_test(NAME test_log_privacy COMMAND test_log_privacy)
add_test(NAME test_log_privacy_sources COMMAND test_log_privacy_sources)
