# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Windows (MSVC + XRT) build of the step-1 host programs; Linux uses the Makefile.
# The package is built on Linux (`make package`) and read from precompiled/pkg.
#
#   cmake -S . -B build/msvc
#   cmake --build build/msvc --config Release
cmake_minimum_required(VERSION 3.22)
project(whisper_encoder_launchlib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL/cmake/flm_xrt_win.cmake) # XRT_INCLUDE_DIR, XRT_LIB_DIR
set(FLM_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL) # nlohmann/json

foreach(prog switch_cost make_switch)
  add_executable(${prog} host/${prog}.cpp)
  target_include_directories(${prog} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../launchlib/include
    ${FLM_DLL}/include
    ${XRT_INCLUDE_DIR})
  target_compile_definitions(${prog} PRIVATE DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${prog} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
  # The static CRT (/MT), as in FLM_DLL's XRT builds.
  set_property(TARGET ${prog} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_link_directories(${prog} PRIVATE ${XRT_LIB_DIR})
  target_link_libraries(${prog} PRIVATE xrt_coreutil)
endforeach()
