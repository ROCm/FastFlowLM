# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(iron_backend_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
find_path(BOOST_PROCESS_INCLUDE_DIR NAMES boost/process.hpp
    HINTS "$ENV{BOOST_ROOT}" "C:/dev/boost_1_88_0" REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(test_iron_worker
    test_worker.cpp
    "${FLM_SOURCE_DIR}/common/iron/worker.cpp")
target_include_directories(test_iron_worker PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${BOOST_PROCESS_INCLUDE_DIR}")
if(WIN32)
    target_link_libraries(test_iron_worker PRIVATE shell32)
endif()

enable_testing()
add_test(NAME test_iron_worker COMMAND test_iron_worker)
set_tests_properties(test_iron_worker PROPERTIES
    TIMEOUT 30
    ENVIRONMENT
        "FLM_IRON_PYTHON=${Python3_EXECUTABLE};PYTHONPATH=${CMAKE_CURRENT_LIST_DIR}")
