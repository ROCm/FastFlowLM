# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(iron_backend_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_iron_worker
    test_worker.cpp
    "${FLM_SOURCE_DIR}/common/iron/worker.cpp")
target_include_directories(test_iron_worker PRIVATE "${FLM_SOURCE_DIR}/include")

enable_testing()
add_test(NAME test_iron_worker COMMAND test_iron_worker)
set_tests_properties(test_iron_worker PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_LIST_DIR}")