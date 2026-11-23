# Windows (MSVC + XRT) build of the DMA page-translation probe; Linux builds with g++ directly.
cmake_minimum_required(VERSION 3.22)
project(dma_pages LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_XRT_WIN_XRT_ONLY ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../flm-dev-kit/FLM_DLL/cmake/flm_xrt_win.cmake)

add_executable(probe probe.cpp)
target_include_directories(probe PRIVATE ${XRT_INCLUDE_DIR})
target_compile_definitions(probe PRIVATE DISABLE_ABI_CHECK=1 _CRT_SECURE_NO_WARNINGS NOMINMAX)
target_compile_options(probe PRIVATE /O2 /EHsc /utf-8 /Zc:__cplusplus)
set_property(TARGET probe PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
target_link_directories(probe PRIVATE ${XRT_LIB_DIR})
target_link_libraries(probe PRIVATE xrt_coreutil advapi32)
