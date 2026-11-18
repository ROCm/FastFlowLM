# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.22)
project(qwen3_8mtp_npu VERSION 1.0.0 LANGUAGES CXX)

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and enable OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling multi-threading support")
else()
    message(WARNING "OpenMP not found - optimized functions will run single-threaded")
endif()

# Force x64 architecture on Windows
if(WIN32)
    set(CMAKE_GENERATOR_PLATFORM x64)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/qwen3_8mtp_npu/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/qwen3_8mtp_npu/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/qwen3_8mtp_npu/lib)

# Force output directories to be absolute
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# ———————————————————————————————————————————————
# User-tweakable XRT settings (override on cmake command-line)
# ———————————————————————————————————————————————
option(FLM_USE_HRX "Build FLM engine libs against the HRX amdxdna runtime instead of XRT" OFF)
if(FLM_USE_HRX)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/flm_hrx.cmake)
    # Stale CMake cache can leak XRT headers into an HRX reconfigure.
    set(XRT_INCLUDE_DIR "" CACHE PATH "Where XRT headers live" FORCE)
    set(XRT_LIB_DIR "" CACHE PATH "Where XRT libs live" FORCE)
    set(FLM_NPU_CORE_LIB ${HRX_LIBRARIES})
    set(FLM_NPU_AIEBU_LIB "")
else()
    set(XRT_INCLUDE_DIR C:/dev/XRT/src/runtime_src/core/include CACHE PATH   "Where XRT headers live")
    set(XRT_LIB_DIR     C:/dev/xrtNPUfromDLL     CACHE PATH   "Where XRT libs live")
    set(HRX_INCLUDE_DIRS "")
    set(FLM_NPU_CORE_LIB xrt_coreutil)
    set(FLM_NPU_AIEBU_LIB aiebu_static)
endif()

# ———————————————————————————————————————————————
# Gather your sources
# ———————————————————————————————————————————————
# ../vision_common is not globbed (unused, and a duplicate symbol source);
# tools/ is excluded -- the milestone drivers have a main().
file(GLOB SOURCES
    "*.cpp"
    "common/*.cpp"
    "common/npu/*.cpp"
    "prefill/*.cpp"
    "prefill/npu/*.cpp"
    "speculate/*.cpp"
    "speculate/npu/*.cpp"
    "verify/*.cpp"
    "verify/npu/*.cpp"
    "image/*.cpp"
    "image/npu/*.cpp"
)
file(GLOB HEADERS
    "../../include/*.hpp"
    "*.hpp"
    "common/*.hpp"
    "common/npu/*.hpp"
    "prefill/*.hpp"
    "prefill/npu/*.hpp"
    "speculate/*.hpp"
    "speculate/npu/*.hpp"
    "verify/*.hpp"
    "verify/npu/*.hpp"
    "image/*.hpp"
    "image/npu/*.hpp"
)


# ———————————————————————————————————————————————
# Define the library
# ———————————————————————————————————————————————
add_library(qwen3_8mtp_npu SHARED ${SOURCES} ${HEADERS})



# HRX uses the dynamic CRT, the same one as flm and hrx. std::codecvt::id
# then stays in msvcp140.dll, so WINDOWS_EXPORT_ALL_SYMBOLS does not publish it.
# The XRT build links aiebu_static (/MT) and keeps the static CRT.
if(MSVC)
    if(FLM_USE_HRX)
        set_property(TARGET qwen3_8mtp_npu PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        target_compile_options(qwen3_8mtp_npu PRIVATE $<$<CONFIG:Debug>:/MDd>$<$<NOT:$<CONFIG:Debug>>:/MD>)
    else()
        set_property(TARGET qwen3_8mtp_npu PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(qwen3_8mtp_npu PRIVATE /MT$<$<CONFIG:Debug>:d>)
    endif()
endif()


set_target_properties(qwen3_8mtp_npu PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/qwen3_8mtp_npu/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/qwen3_8mtp_npu/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/qwen3_8mtp_npu/lib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(qwen3_8mtp_npu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../../include
    ${XRT_INCLUDE_DIR} ${HRX_INCLUDE_DIRS}
    C:/dev/boost_1_88_0
    C:/dev/vcpkg/installed/x64-windows/include
)

# Add AVX2 specific definitions and flags
target_compile_definitions(qwen3_8mtp_npu PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    USEAVX512=1
    _WIN32    
    __WINDOWS__
    _ENABLE_EXTENDED_ALIGNED_STORAGE
)

# Enhanced compiler options for AVX-512
target_compile_options(qwen3_8mtp_npu PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512 /O2 /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -march=native -O3>
)

# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(qwen3_8mtp_npu PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/openmp>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fopenmp>
    )
endif()

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(qwen3_8mtp_npu PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add AVX2 specific link flags for MSVC
if(MSVC)
    set_target_properties(qwen3_8mtp_npu PROPERTIES
        LINK_FLAGS "/MACHINE:X64"
    )
    # Add C++ standard library for RTTI and other symbols
    target_link_libraries(qwen3_8mtp_npu PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:msvcrt.lib>
    )
endif()

target_link_directories(qwen3_8mtp_npu PUBLIC
    ${XRT_LIB_DIR}
    C:/dev/boost_1_88_0/stage/lib
    ${CMAKE_SOURCE_DIR}/../../external_libs
    C:/dev/vcpkg/installed/x64-windows/lib
)

# Check if all required libraries exist
set(Q4NPU_EXPRESS_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/q4_npu_eXpress.lib)
set(DEQUANT_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/dequant.lib)
set(GEMM_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/gemm.lib)
set(MHA_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/mha.lib)

set(ALL_LIBS_EXIST TRUE)
if(NOT EXISTS ${Q4NPU_EXPRESS_LIB_PATH})
    set(ALL_LIBS_EXIST FALSE)
    message(WARNING "q4_npu_eXpress.lib not found at ${Q4NPU_EXPRESS_LIB_PATH}. Build q4nx first for full functionality.")
endif()
if(NOT EXISTS ${DEQUANT_LIB_PATH})
    set(ALL_LIBS_EXIST FALSE)
    message(WARNING "dequant.lib not found at ${DEQUANT_LIB_PATH}. Build dequant first for full functionality.")
endif()
if(NOT EXISTS ${GEMM_LIB_PATH})
    set(ALL_LIBS_EXIST FALSE)
    message(WARNING "gemm.lib not found at ${GEMM_LIB_PATH}. Build gemm first for full functionality.")
endif()
if(NOT EXISTS ${MHA_LIB_PATH})
    set(ALL_LIBS_EXIST FALSE)
    message(WARNING "mha.lib not found at ${MHA_LIB_PATH}. Build mha first for full functionality.")
endif()

if(${ALL_LIBS_EXIST})
    target_link_libraries(qwen3_8mtp_npu PUBLIC
        ${FLM_NPU_CORE_LIB}
        ${FLM_NPU_AIEBU_LIB}
        ${Q4NPU_EXPRESS_LIB_PATH}
        ${DEQUANT_LIB_PATH}
        ${GEMM_LIB_PATH}
        ${MHA_LIB_PATH}
    )
else()
    target_link_libraries(qwen3_8mtp_npu PUBLIC
        ${FLM_NPU_CORE_LIB}
    )
endif()

# Copy the built library to build/lib directory
add_custom_command(TARGET qwen3_8mtp_npu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../../build/lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:qwen3_8mtp_npu>
        ${CMAKE_SOURCE_DIR}/../../build/lib/qwen3_8mtp_npu.dll
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_IMPORT_FILE:qwen3_8mtp_npu>
        ${CMAKE_SOURCE_DIR}/../../build/lib/qwen3_8mtp_npu.lib
)
