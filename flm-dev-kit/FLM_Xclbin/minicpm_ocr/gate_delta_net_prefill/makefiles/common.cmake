# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# common.cmake
# Licensed under the Apache License v2.0 with LLVM Exceptions
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Created by Alfred (Converted to CMake)

# User can override AIETOOLS_DIR and AIEOPT_DIR
set(AIETOOLS_DIR "$ENV{AIETOOLS_ROOT}" CACHE PATH "Path to AIETOOLS installation (Chess only)")
set(AIEOPT_DIR "" CACHE PATH "Path to AIEOPT installation")

# Chess only (ENABLE_CHESSCC): find xchesscc if AIETOOLS_DIR is not set
if(ENABLE_CHESSCC AND NOT AIETOOLS_DIR)
  find_program(XCHESSCC_PATH xchesscc)
  if(NOT XCHESSCC_PATH)
    message(FATAL_ERROR "ENABLE_CHESSCC needs Vitis AIE Essentials: xchesscc not found in PATH and AIETOOLS_DIR / AIETOOLS_ROOT not set")
  endif()
  get_filename_component(AIETOOLS_BIN_DIR "${XCHESSCC_PATH}" DIRECTORY)
  get_filename_component(AIETOOLS_DIR "${AIETOOLS_BIN_DIR}/.." ABSOLUTE)
endif()

# Find aie-opt if AIEOPT_DIR not set
if(NOT AIEOPT_DIR)
  find_program(AIE_OPT_PATH aie-opt)
  if(NOT AIE_OPT_PATH)
    message(FATAL_ERROR "aie-opt not found in PATH and AIEOPT_DIR not set")
  endif()
  get_filename_component(AIEOPT_BIN_DIR "${AIE_OPT_PATH}" DIRECTORY)
  get_filename_component(AIEOPT_DIR "${AIEOPT_BIN_DIR}/.." ABSOLUTE)
endif()

# Find include directories
find_path(AIE_INCLUDE_DIR aie_api.h
  HINTS "${AIETOOLS_DIR}/data/versal_prod/lib" "${AIETOOLS_DIR}/data/aie_ml/lib"
)
if(NOT AIE_INCLUDE_DIR)
  set(AIE_INCLUDE_DIR "${AIETOOLS_DIR}/data/versal_prod/lib")
endif()
set(AIE2_INCLUDE_DIR "${AIETOOLS_DIR}/data/aie_ml/lib")

# Compiler flags
set(WARNING_FLAGS -Wno-parentheses -Wno-attributes -Wno-macro-redefined)

set(CHESSCCWRAP2_FLAGS
    aie2
    -I ${AIETOOLS_DIR}/include
)
set(CHESSCCWRAP2P_FLAGS
    aie2p
    -I ${AIETOOLS_DIR}/include
    -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
)
set(PEANOWRAP2_FLAGS
    -O2
    -v
    -std=c++20
    --target=aie2-none-unknown-elf
    ${WARNING_FLAGS}
    -DNDEBUG
    -I ${AIEOPT_DIR}/include
)
set(PEANOWRAP2P_FLAGS
    -O2
    -v
    -std=c++20
    --target=aie2p-none-unknown-elf
    ${WARNING_FLAGS}
    -DNDEBUG
    -I ${AIEOPT_DIR}/include
    -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
)

# AIECC_FLAGS
set(AIECC_FLAGS --aie-generate-cdo --no-compile-host)
if(ENABLE_CHESSCC)
    set(AIECC_FLAGS ${AIECC_FLAGS} --xchesscc --xbridge)
else()
    set(AIECC_FLAGS ${AIECC_FLAGS} --no-xchesscc --no-xbridge)
endif()

# Export variables
set(AIE_INCLUDE_DIR "${AIE_INCLUDE_DIR}")
set(AIE2_INCLUDE_DIR "${AIE2_INCLUDE_DIR}")
set(AIEOPT_DIR "${AIEOPT_DIR}")
set(CHESSCCWRAP2_FLAGS "${CHESSCCWRAP2_FLAGS}")
set(CHESSCCWRAP2P_FLAGS "${CHESSCCWRAP2P_FLAGS}")
set(PEANOWRAP2_FLAGS "${PEANOWRAP2_FLAGS}")
set(PEANOWRAP2P_FLAGS "${PEANOWRAP2P_FLAGS}")