# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.22)
project(q4nx VERSION 1.0.0 LANGUAGES CXX)

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib)

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
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "../../include/*.hpp" "*.hpp" "*.h")

# ———————————————————————————————————————————————
# Define the library
# ———————————————————————————————————————————————
add_library(q4_npu_eXpress SHARED ${SOURCES} ${HEADERS})
set_target_properties(q4_npu_eXpress PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/q4nx/lib"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/q4nx/lib"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/../../build/q4nx/lib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(q4_npu_eXpress PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}
    ${XRT_INCLUDE_DIR} ${HRX_INCLUDE_DIRS} ${FLM_BOOST_INCLUDE_DIR}
)

target_compile_definitions(q4_npu_eXpress PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
)

target_compile_options(q4_npu_eXpress PRIVATE
    $<$<BOOL:${MSVC}>:/arch:AVX512 /O2 /fp:fast>
    $<$<NOT:$<BOOL:${MSVC}>>:-mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -O2>
)

target_link_directories(q4_npu_eXpress PUBLIC
    ${XRT_LIB_DIR}
)

# Check if npu_utils library exists
set(NPU_UTILS_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib/npu_utils.lib)
if(EXISTS ${NPU_UTILS_LIB_PATH})
    target_link_libraries(q4_npu_eXpress PUBLIC
        ${FLM_NPU_CORE_LIB}
        ${NPU_UTILS_LIB_PATH}
    )
else()
    target_link_libraries(q4_npu_eXpress PUBLIC
        ${FLM_NPU_CORE_LIB}
    )
    message(WARNING "npu_utils.lib not found at ${NPU_UTILS_LIB_PATH}. Build npu_utils first for full functionality.")
endif()

# Copy the built library to build/lib directory
add_custom_command(TARGET q4_npu_eXpress POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/../../build/lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:q4_npu_eXpress>
        ${CMAKE_SOURCE_DIR}/../../build/lib/q4_npu_eXpress.dll
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_IMPORT_FILE:q4_npu_eXpress>
        ${CMAKE_SOURCE_DIR}/../../build/lib/q4_npu_eXpress.lib
)
