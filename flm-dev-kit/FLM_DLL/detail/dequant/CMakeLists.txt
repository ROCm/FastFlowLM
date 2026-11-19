# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.22)
project(dequant VERSION 1.0.0 LANGUAGES CXX)

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
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/dequant/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/dequant/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/dequant/lib)

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
file(GLOB HEADERS "../../include/*.hpp" "../../include/*/*.hpp" "*.hpp")



# ———————————————————————————————————————————————
# Define the library
# ———————————————————————————————————————————————
add_library(dequant SHARED ${SOURCES} ${HEADERS})
set_target_properties(dequant PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/dequant/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/dequant/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/dequant/lib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(dequant PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${XRT_INCLUDE_DIR} ${HRX_INCLUDE_DIRS} ${FLM_BOOST_INCLUDE_DIR}
)

# Add AVX2 specific definitions and flags
target_compile_definitions(dequant PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    _ENABLE_EXTENDED_ALIGNED_STORAGE
    __WINDOWS__
)

# Enhanced compiler options for AVX2
target_compile_options(dequant PUBLIC
    $<$<BOOL:${MSVC}>:/arch:AVX512 /O2 /fp:fast>
    $<$<NOT:$<BOOL:${MSVC}>>:-mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -O2>
)

# Add AVX2 specific link flags for MSVC
if(MSVC)
    set_target_properties(dequant PROPERTIES
        LINK_FLAGS "/MACHINE:X64"
    )
    # Add C++ standard library for RTTI and other symbols
    target_link_libraries(dequant PUBLIC
        $<$<BOOL:${MSVC}>:msvcrt.lib>
    )
endif()

target_link_directories(dequant PUBLIC
    ${XRT_LIB_DIR}
)

# Check if npu_utils library exists
set(NPU_UTILS_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/npu_utils.lib)
if(EXISTS ${NPU_UTILS_LIB_PATH})
    target_link_libraries(dequant PUBLIC
        ${FLM_NPU_CORE_LIB}
        ${NPU_UTILS_LIB_PATH}
    )
else()
    target_link_libraries(dequant PUBLIC
        ${FLM_NPU_CORE_LIB}
    )
    message(WARNING "npu_utils.lib not found at ${NPU_UTILS_LIB_PATH}. Build npu_utils first for full functionality.")
endif()

# Copy the built library to build/lib directory
add_custom_command(TARGET dequant POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../../build/lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:dequant>
        ${CMAKE_SOURCE_DIR}/../../build/lib/dequant.dll
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_IMPORT_FILE:dequant>
        ${CMAKE_SOURCE_DIR}/../../build/lib/dequant.lib
)
