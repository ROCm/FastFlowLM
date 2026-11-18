# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.22)
project(lm_head VERSION 1.0.0 LANGUAGES CXX)

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force x64 architecture on Windows
if(WIN32)
    set(CMAKE_GENERATOR_PLATFORM x64)
    set(CMAKE_VS_PLATFORM_NAME "x64")
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/lm_head/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/lm_head/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/lm_head/lib)

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
file(GLOB SOURCES
    "*.cpp"
)
file(GLOB HEADERS "../../include/*.hpp" "*.hpp")



# ———————————————————————————————————————————————
# Define the library
# ———————————————————————————————————————————————
add_library(lm_head SHARED ${SOURCES} ${HEADERS})
set_target_properties(lm_head PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/lm_head/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/lm_head/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/lm_head/lib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(lm_head PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${XRT_INCLUDE_DIR} ${HRX_INCLUDE_DIRS}
    C:/dev/boost_1_88_0
)

# Add AVX2 specific definitions and flags
target_compile_definitions(lm_head PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    _ENABLE_EXTENDED_ALIGNED_STORAGE
    __WINDOWS__
)

# Enhanced compiler options for AVX2
target_compile_options(lm_head PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /O2 /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma -march=native -O2>
)

# Add AVX2 specific link flags for MSVC
if(MSVC)
    set_target_properties(lm_head PROPERTIES
        LINK_FLAGS "/MACHINE:X64"
    )
    # Add C++ standard library for RTTI and other symbols
    target_link_libraries(lm_head PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:msvcrt.lib>
    )
endif()

# HRX uses the dynamic CRT, the same one as flm and hrx. std::codecvt::id
# then stays in msvcp140.dll, so WINDOWS_EXPORT_ALL_SYMBOLS does not publish it.
# The XRT build links aiebu_static (/MT) and keeps the static CRT.
if(MSVC)
    if(FLM_USE_HRX)
        set_property(TARGET lm_head PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        target_compile_options(lm_head PRIVATE $<$<CONFIG:Debug>:/MDd>$<$<NOT:$<CONFIG:Debug>>:/MD>)
    else()
        set_property(TARGET lm_head PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(lm_head PRIVATE /MT$<$<CONFIG:Debug>:d>)
    endif()
endif()


target_link_directories(lm_head PUBLIC
    ${XRT_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/../../external_libs
    C:/Program\ Files/boost/boost_1_88_0/lib64-msvc-14.3
)

# Check if npu_utils library exists
set(Q4NPU_EXPRESS_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/q4_npu_eXpress.lib)
target_link_libraries(lm_head PUBLIC
    ${FLM_NPU_CORE_LIB}
    ${FLM_NPU_AIEBU_LIB}
    ${Q4NPU_EXPRESS_LIB_PATH}
)

# Copy the built library to build/lib directory
add_custom_command(TARGET lm_head POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../../build/lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:lm_head>
        ${CMAKE_SOURCE_DIR}/../../build/lib/lm_head.dll
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_IMPORT_FILE:lm_head>
        ${CMAKE_SOURCE_DIR}/../../build/lib/lm_head.lib
)
