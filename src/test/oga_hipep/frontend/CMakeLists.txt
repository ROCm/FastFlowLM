# Copyright (c) 2026 Advanced Micro Devices, Inc.

cmake_minimum_required(VERSION 3.22)
project(oga_hipep_frontend_tests LANGUAGES CXX)

if(NOT WIN32)
    message(FATAL_ERROR "oga-hipep frontend tests require Windows")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_default_flm_source_dir "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(FLM_SOURCE_DIR "${_default_flm_source_dir}" CACHE PATH "FastFlowLM src directory")

find_path(BOOST_INCLUDE_DIR NAMES boost/any.hpp
    HINTS "$ENV{CONDA_PREFIX}/Library/include"
          "$ENV{USERPROFILE}/anaconda3/Library/include" REQUIRED)
find_path(XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
    HINTS "$ENV{XRT_INCLUDE_DIR}"
          "C:/dev/XRT/src/runtime_src/core/include" REQUIRED)
set(XRT_LIB_DIR "${XRT_INCLUDE_DIR}/../lib" CACHE PATH "Directory containing xrt_coreutil")

include("${FLM_SOURCE_DIR}/common/chat/chat_sources.cmake")
include(CTest)

function(add_frontend_test name frontend_source)
    add_executable(${name}
        "${FLM_SOURCE_DIR}/test/oga_hipep/${name}.cpp"
        "${FLM_SOURCE_DIR}/common/prefix_slot_cache.cpp"
        "${FLM_SOURCE_DIR}/common/AutoModel/automodel.cpp"
        "${FLM_SOURCE_DIR}/common/AutoModel/model_backend.cpp"
        "${FLM_SOURCE_DIR}/common/AutoModel/${frontend_source}"
        ${FLM_CHAT_SOURCES})

    target_include_directories(${name} PRIVATE
        "${FLM_SOURCE_DIR}/include"
        "${FLM_SOURCE_DIR}/pull"
        "${FLM_SOURCE_DIR}/runner"
        "${FLM_SOURCE_DIR}/server"
        "${FLM_SOURCE_DIR}/../third_party/tokenizers-cpp/include"
        "${BOOST_INCLUDE_DIR}"
        "${XRT_INCLUDE_DIR}")
    target_compile_definitions(${name} PRIVATE
        DEV_BUILD=1 __WINDOWS__ USEAVX2=1 DISABLE_ABI_CHECK=1 _ENABLE_EXTENDED_ALIGNED_STORAGE
        WIN32_LEAN_AND_MEAN NOMINMAX)
    target_compile_options(${name} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/wd4005 /wd4244>)
    target_link_directories(${name} PRIVATE "${XRT_LIB_DIR}")
    target_link_libraries(${name} PRIVATE xrt_coreutil flm_xgrammar)

    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_frontend_test(test_qwen_frontend modeling_qwen3_5vl.cpp)
add_frontend_test(test_gpt_oss_frontend modeling_gpt_oss.cpp)
