# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Set defaults if not already set
cmake_minimum_required(VERSION 3.28)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(XRT REQUIRED)



set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Boost_USE_DEBUG_LIBS        ON)  # ignore debug libs and
else()
    set(Boost_USE_DEBUG_LIBS        OFF)  # ignore debug libs and
endif()
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)


set(HOST_SRCDIR "${HOST_DIR}")
set(HOST_O_DIR "${CMAKE_CURRENT_BINARY_DIR}")
# Use host-specific target name to avoid conflicts when building from root
# HOST_NAME should be set by the caller
if(NOT DEFINED HOST_NAME)
    message(FATAL_ERROR "HOST_NAME must be defined by the caller")
endif()
set(HOST_C_TARGET "${HOST_NAME}_run.exe")

# Host source files
file(GLOB HOST_HEADERS "${HOST_SRCDIR}/*.hpp")


# Set compiler and flags
set(CXX_COMPILER "g++-13")
set(VERBOSE 0)


set(COMMON_CXX_FLAGS
    -std=c++23  -fPIC -Wall -DUSEAVX512=1
    -O3 -march=native -march=native  
    -DVERBOSE=${VERBOSE}
)
# if debug, set -ggdb
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_CXX_FLAGS ${COMMON_CXX_FLAGS} -ggdb)
endif()


if(DEVICE STREQUAL "npu1" OR DEVICE STREQUAL "npu2")
    set(CMAKE_CXX_COMPILER ${CXX_COMPILER})
    add_compile_options(${COMMON_CXX_FLAGS})
    link_directories(/opt/xilinx/xrt/lib)
else()
    message(FATAL_ERROR "Unsupported DEVICE=${DEVICE}")
endif()



# === Step 2: Define executable ===
file(GLOB HOST_SRCS "${HOST_SRCDIR}/*.cpp")
# 3. Add all sources to the executable. CMake will compile and link them.
add_executable(${HOST_C_TARGET} ${HOST_SRCS} )

target_compile_features(${HOST_C_TARGET} PRIVATE cxx_std_23)

# 4. Add the 'common' directory to the include paths, just like the Makefile's -I flag.
target_include_directories(${HOST_C_TARGET} PRIVATE
    ${HOME_DIR}/host
)

target_link_libraries(${HOST_C_TARGET}
    npu_common
    -lm
    XRT::xrt_coreutil
    Boost::program_options
    Boost::filesystem
    safetensors_cpp
    # ggml
    # gemma3_common  
    #TODO: somethin pass from outside
)

set_target_properties(${HOST_C_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${HOST_O_DIR}"
)
message(STATUS "Host target: ${HOST_C_TARGET}")
message(STATUS "Sources: ${HOST_SRCS}")
