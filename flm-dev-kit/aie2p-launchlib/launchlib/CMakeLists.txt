# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Windows (MSVC) build of the unit tests (no NPU, no XRT); Linux uses the Makefile.
#
#   cmake -S . -B build/msvc
#   cmake --build build/msvc --config Release
#   build/msvc/Release/test_txn <pkg>/vadd/init.txn ...   (as `make test`)
#   build/msvc/Release/test_elf <pkg> vadd vmul ...       (as `make test_elf`)
#   build/msvc/Release/test_ctrlpkt <pkg> vadd vmul ...   (as `make test`)
cmake_minimum_required(VERSION 3.22)
project(launchlib_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_DLL ${CMAKE_CURRENT_SOURCE_DIR}/../../FLM_DLL) # nlohmann/json, aiebu headers and aiebu_static.lib

foreach(test test_txn test_rt test_build test_errors test_ctrlpkt)
  add_executable(${test} tests/${test}.cpp)
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${FLM_DLL}/include)
  target_compile_definitions(${test} PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(${test} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O1 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
endforeach()

# test_elf: txn::to_elf against aiebu's ELF. aiebu_static is built with the
# static CRT (/MT), as in FLM_DLL's XRT builds.
add_executable(test_elf tests/test_elf.cpp)
target_include_directories(test_elf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${FLM_DLL}/include)
target_compile_definitions(test_elf PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
target_compile_options(test_elf PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2 /EHsc /utf-8 /Zc:__cplusplus /bigobj>)
set_property(TARGET test_elf PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_directories(test_elf PRIVATE ${FLM_DLL}/external_libs)
target_link_libraries(test_elf PRIVATE aiebu_static)
