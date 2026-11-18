# Host-only check of the ignore_eos branch in AutoModel::_sample.
# No NPU: the stubs directory shadows the device runtime, same as
# src/test/model_frontend.
#   cmake -S src/test/ignore_eos -B build-ignore-eos
#   cmake --build build-ignore-eos && ctest --test-dir build-ignore-eos
cmake_minimum_required(VERSION 3.22)
project(ignore_eos_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")

add_executable(test_ignore_eos
    test_ignore_eos.cpp
    "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
    "${FLM_SOURCE_DIR}/common/AutoModel/model_extension_backend.cpp"
    "${FLM_SOURCE_DIR}/common/modules/sampler.cpp"
    "${FLM_SOURCE_DIR}/common/prefix_slot_cache.cpp"
    "${FLM_SOURCE_DIR}/common/utils.cpp"
    ${FLM_CHAT_SOURCES})
target_include_directories(test_ignore_eos PRIVATE
    "${FLM_SOURCE_DIR}/test/model_frontend/stubs"
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include")
target_compile_definitions(test_ignore_eos PRIVATE
    CMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/missing-assets"
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/missing-assets")
if(WIN32)
    target_compile_definitions(test_ignore_eos PRIVATE
        WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
if(MSVC)
    target_compile_options(test_ignore_eos PRIVATE /utf-8 /bigobj)
endif()
target_link_libraries(test_ignore_eos PRIVATE flm_xgrammar)

enable_testing()
add_test(NAME test_ignore_eos COMMAND test_ignore_eos)
