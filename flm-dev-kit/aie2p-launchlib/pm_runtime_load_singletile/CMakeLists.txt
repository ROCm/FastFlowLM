# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Windows (MSVC + XRT) build of the host programs; Linux uses the Makefile.
# The package is built on Linux (`make precompile COLS=8`) and read from precompiled/c8.
#
#   cmake -S . -B build/c8 -DCOLS=8
#   cmake --build build/c8 --config Release
cmake_minimum_required(VERSION 3.22)
project(pm_runtime_load_singletile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COLS 8 CACHE STRING "columns of 4 cores")

set(FLM_XRT_WIN_XRT_ONLY ON) # XRT paths only: no aiebu, no Boost
include(${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL/cmake/flm_xrt_win.cmake) # XRT_INCLUDE_DIR, XRT_LIB_DIR
set(FLM_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL) # nlohmann/json

add_executable(test host_test/test.cpp)
add_executable(stress host_test/stress.cpp)
add_executable(check_sequence host_test/checks/check_sequence.cpp)
foreach(prog test stress check_sequence)
  target_include_directories(${prog} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../launchlib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/host_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../vadd_vmul_bf16/include # spawn.hpp
    ${FLM_DLL}/include
    ${XRT_INCLUDE_DIR})
  target_compile_definitions(${prog} PRIVATE COLS=${COLS} DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${prog} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
  # The static CRT (/MT), as in FLM_DLL's XRT builds.
  set_property(TARGET ${prog} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_link_directories(${prog} PRIVATE ${XRT_LIB_DIR})
  target_link_libraries(${prog} PRIVATE xrt_coreutil)
endforeach()

# ctx_probe: XRT's internal list of hardware contexts, printed and timed. Needs XRT's
# source headers (core/common) and boost; the XRT DLL it runs against must be
# the same XRT (the XRT under FLM_WIN_DEPS_ROOT is not: pass -DXRT_SRC_DIR=<Xilinx/XRT at the hash
# xrt-smi prints>/src/runtime_src; files the tree generates at build time come
# from XRT_INCLUDE_DIR). The DLL CRT (the query hands back std::string).
set(XRT_SRC_DIR ${FLM_WIN_DEPS_ROOT}/XRT/src/runtime_src CACHE PATH "XRT source tree (core/common)")
set(BOOST_INCLUDE_DIR ${FLM_WIN_DEPS_ROOT}/boost_1_88_0 CACHE PATH "boost headers")
if(EXISTS ${XRT_SRC_DIR}/core/common/device.h AND EXISTS ${BOOST_INCLUDE_DIR}/boost/property_tree)
  add_executable(ctx_probe host_test/checks/ctx_probe.cpp)
  target_include_directories(ctx_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../launchlib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../vadd_vmul_bf16/include # spawn.hpp
    ${XRT_SRC_DIR} ${XRT_SRC_DIR}/core/include ${XRT_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})
  target_compile_definitions(ctx_probe PRIVATE DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS)
  target_compile_options(ctx_probe PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
  target_link_directories(ctx_probe PRIVATE ${XRT_LIB_DIR})
  target_link_libraries(ctx_probe PRIVATE xrt_coreutil)
  # slot_check: NpuWatch's vtable slots (npu_watch.hpp) against XRT's headers.
  add_executable(slot_check host_test/checks/slot_check.cpp)
  target_include_directories(slot_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../launchlib/include
    ${XRT_SRC_DIR} ${XRT_SRC_DIR}/core/include ${XRT_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})
  target_compile_definitions(slot_check PRIVATE DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS)
  target_compile_options(slot_check PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
  target_link_directories(slot_check PRIVATE ${XRT_LIB_DIR})
  target_link_libraries(slot_check PRIVATE xrt_coreutil)
else()
  message(STATUS "ctx_probe: not built (no ${XRT_SRC_DIR}/core/common/device.h or boost in ${BOOST_INCLUDE_DIR})")
endif()

# list_probe: the same list with public XRT headers only, as NpuWatch
# reads it (the query key found by RTTI, the element size by the length of the
# list). list_probe_mt: with the static CRT, as test and stress.
foreach(prog list_probe list_probe_mt)
  add_executable(${prog} host_test/checks/list_probe.cpp)
  target_include_directories(${prog} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../launchlib/include ${XRT_INCLUDE_DIR})
  target_compile_definitions(${prog} PRIVATE DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${prog} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus>)
  target_link_directories(${prog} PRIVATE ${XRT_LIB_DIR})
  target_link_libraries(${prog} PRIVATE xrt_coreutil)
endforeach()
set_property(TARGET list_probe_mt PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# The Session's context list on Windows (npu_watch.hpp, LAUNCH_NPU_WATCH_XRT=1):
# public XRT headers only; checked at run time (XRT 2.19 or later, the query and
# its layout recognized), else blind. Without it, test and stress are blind
# (every warm and rerun runs as cold unless --exclusive).
option(NPU_WATCH "test and stress see other contexts through XRT (LAUNCH_NPU_WATCH_XRT=1)" OFF)
if(NPU_WATCH)
  foreach(prog test stress)
    target_compile_definitions(${prog} PRIVATE LAUNCH_NPU_WATCH_XRT=1)
  endforeach()
endif()
