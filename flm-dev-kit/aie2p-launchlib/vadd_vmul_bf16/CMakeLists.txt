# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Windows (MSVC + XRT) build of the host programs; Linux uses the Makefile.
# The packages are built on Linux (`make precompile`) and read from precompiled/.
#
#   cmake -S . -B build/c8x2 -DCOLS=8 -DCORES=2
#   cmake --build build/c8x2 --config Release
cmake_minimum_required(VERSION 3.22)
project(vadd_vmul_bf16 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EW_N 256 CACHE STRING "bf16 values per block")
set(COLS 1 CACHE STRING "columns")
set(CORES 1 CACHE STRING "cores per column")

set(FLM_XRT_WIN_XRT_ONLY ON) # XRT paths only: no aiebu, no Boost
include(${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL/cmake/flm_xrt_win.cmake) # XRT_INCLUDE_DIR, XRT_LIB_DIR
set(FLM_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL) # nlohmann/json

foreach(prog host stress ovlp make_overlay)
  add_executable(${prog} host/${prog}.cpp)
  target_include_directories(${prog} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../launchlib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${FLM_DLL}/include
    ${XRT_INCLUDE_DIR})
  target_compile_definitions(${prog} PRIVATE
    EW_N=${EW_N} EW_COLS=${COLS} EW_CORES=${CORES}
    DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${prog} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
  # The static CRT (/MT), as in FLM_DLL's XRT builds.
  set_property(TARGET ${prog} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_link_directories(${prog} PRIVATE ${XRT_LIB_DIR})
  target_link_libraries(${prog} PRIVATE xrt_coreutil)
endforeach()
