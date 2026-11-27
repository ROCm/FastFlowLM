# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

cmake_minimum_required(VERSION 3.22)
project(inkling2_npu VERSION 1.0.0 LANGUAGES CXX)

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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/inkling2_npu/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/inkling2_npu/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/inkling2_npu/lib)

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
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/flm_xrt_win.cmake)
    set(HRX_INCLUDE_DIRS "")
    set(FLM_NPU_CORE_LIB xrt_coreutil)
    set(FLM_NPU_AIEBU_LIB aiebu_static)
endif()

# ———————————————————————————————————————————————
# Gather your sources
# ———————————————————————————————————————————————
file(GLOB SOURCES
    "*.cpp"
)
file(GLOB HEADERS "../../include/*.hpp" "*.hpp")


# ———————————————————————————————————————————————
# Define the library
# ———————————————————————————————————————————————
add_library(inkling2_npu SHARED ${SOURCES} ${HEADERS})



# HRX uses the dynamic CRT, the same one as flm and hrx. std::codecvt::id
# then stays in msvcp140.dll, so WINDOWS_EXPORT_ALL_SYMBOLS does not publish it.
# The XRT build links aiebu_static (/MT) and keeps the static CRT.
if(MSVC)
    if(FLM_USE_HRX)
        set_property(TARGET inkling2_npu PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        target_compile_options(inkling2_npu PRIVATE $<$<CONFIG:Debug>:/MDd>$<$<NOT:$<CONFIG:Debug>>:/MD>)
    else()
        set_property(TARGET inkling2_npu PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(inkling2_npu PRIVATE /MT$<$<CONFIG:Debug>:d>)
    endif()
endif()


set_target_properties(inkling2_npu PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/inkling2_npu/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/inkling2_npu/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/inkling2_npu/lib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(inkling2_npu PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/../../../aie2p-launchlib/launchlib/include
    ${XRT_INCLUDE_DIR} ${HRX_INCLUDE_DIRS} ${FLM_BOOST_INCLUDE_DIR}
)

# Add AVX2 specific definitions and flags
target_compile_definitions(inkling2_npu PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    USEAVX512=1
    _WIN32    
    __WINDOWS__
    _ENABLE_EXTENDED_ALIGNED_STORAGE
)

# Enhanced compiler options for AVX-512
# the loop controller uses AVX-512 BF16 intrinsics (see the Makefile). MSVC needs no flag for them.
target_compile_options(inkling2_npu PUBLIC
    $<$<BOOL:${MSVC}>:/arch:AVX512 /O2 /fp:fast>
    $<$<NOT:$<BOOL:${MSVC}>>:-mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -mavx512bf16 -O3>
)

# Add OpenMP compiler flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(inkling2_npu PUBLIC
        $<$<BOOL:${MSVC}>:/openmp>
        $<$<NOT:$<BOOL:${MSVC}>>:-fopenmp>
    )
endif()

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(inkling2_npu PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add AVX2 specific link flags for MSVC
if(MSVC)
    set_target_properties(inkling2_npu PROPERTIES
        LINK_FLAGS "/MACHINE:X64"
    )
    # Add C++ standard library for RTTI and other symbols
    target_link_libraries(inkling2_npu PUBLIC
        $<$<BOOL:${MSVC}>:msvcrt.lib>
    )
endif()

target_link_directories(inkling2_npu PUBLIC
    ${XRT_LIB_DIR}
    ${FLM_AIEBU_LIB_DIR}
)

# No host compute modules (gemm / mha / dequant). q4_npu_eXpress is still needed:
# SafeTensors::load_weights lives in q4nx/safe_tensors.cpp, and a DLL must resolve
# it at link time (the Linux .so picks it up from libq4_npu_eXpress.so at load time)
set(Q4NPU_EXPRESS_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/q4_npu_eXpress.lib)
target_link_libraries(inkling2_npu PUBLIC
    ${FLM_NPU_CORE_LIB}
    ${FLM_NPU_AIEBU_LIB}
    ${Q4NPU_EXPRESS_LIB_PATH}
)

# Copy the built library to build/lib directory
add_custom_command(TARGET inkling2_npu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../../build/lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:inkling2_npu>
        ${CMAKE_SOURCE_DIR}/../../build/lib/inkling2_npu.dll
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_IMPORT_FILE:inkling2_npu>
        ${CMAKE_SOURCE_DIR}/../../build/lib/inkling2_npu.lib
)
