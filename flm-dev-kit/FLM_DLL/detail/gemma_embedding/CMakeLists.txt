# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.22)
project(gemma_embedding VERSION 1.0.0 LANGUAGES CXX)

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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/gemma_embedding/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/gemma_embedding/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/gemma_embedding/lib)

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
add_library(gemma_embedding SHARED ${SOURCES} ${HEADERS})


# HRX uses the dynamic CRT, the same one as flm and hrx. std::codecvt::id
# then stays in msvcp140.dll, so WINDOWS_EXPORT_ALL_SYMBOLS does not publish it.
# The XRT build links aiebu_static (/MT) and keeps the static CRT.
if(MSVC)
    if(FLM_USE_HRX)
        set_property(TARGET gemma_embedding PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        target_compile_options(gemma_embedding PRIVATE $<$<CONFIG:Debug>:/MDd>$<$<NOT:$<CONFIG:Debug>>:/MD>)
    else()
        set_property(TARGET gemma_embedding PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        target_compile_options(gemma_embedding PRIVATE /MT$<$<CONFIG:Debug>:d>)
    endif()
endif()

set_target_properties(gemma_embedding PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/gemma_embedding/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/gemma_embedding/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/gemma_embedding/lib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(gemma_embedding PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${XRT_INCLUDE_DIR} ${HRX_INCLUDE_DIRS}
    C:/dev/boost_1_88_0
)

# Add AVX2 specific definitions and flags
target_compile_definitions(gemma_embedding PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    USEAVX512=1
    _WIN32
    __WINDOWS__
    _ENABLE_EXTENDED_ALIGNED_STORAGE
)

# Enhanced compiler options for AVX512 and best optimization (O3)
target_compile_options(gemma_embedding PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512 /O2 /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx512 -mfma -march=native -O3>
)

# Add AVX2 specific link flags for MSVC
if(MSVC)
    set_target_properties(gemma_embedding PROPERTIES
        LINK_FLAGS "/MACHINE:X64"
    )
    # Add C++ standard library for RTTI and other symbols
    target_link_libraries(gemma_embedding PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:msvcrt.lib>
    )
endif()

target_link_directories(gemma_embedding PUBLIC
    ${XRT_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/../../external_libs
    C:/dev/boost_1_88_0/stage/lib
)

# Check if all required libraries exist
set(Q4NPU_EXPRESS_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/q4_npu_eXpress.lib)

set(ALL_LIBS_EXIST TRUE)
if(NOT EXISTS ${Q4NPU_EXPRESS_LIB_PATH})
    set(ALL_LIBS_EXIST FALSE)
    message(WARNING "q4_npu_eXpress.lib not found at ${Q4NPU_EXPRESS_LIB_PATH}. Build q4nx first for full functionality.")
endif()


if(${ALL_LIBS_EXIST})
    target_link_libraries(gemma_embedding PUBLIC
        ${FLM_NPU_CORE_LIB}
        ${FLM_NPU_AIEBU_LIB}
        ${Q4NPU_EXPRESS_LIB_PATH}
    )
else()
    target_link_libraries(gemma_embedding PUBLIC
        ${FLM_NPU_CORE_LIB}
    )
endif()

# Copy the built library to build/lib directory
add_custom_command(TARGET gemma_embedding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../../build/lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:gemma_embedding>
        ${CMAKE_SOURCE_DIR}/../../build/lib/gemma_embedding.dll
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_IMPORT_FILE:gemma_embedding>
        ${CMAKE_SOURCE_DIR}/../../build/lib/gemma_embedding.lib
)
