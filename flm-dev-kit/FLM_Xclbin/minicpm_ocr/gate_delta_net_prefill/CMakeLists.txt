# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

cmake_minimum_required(VERSION 3.13)
project(HOST2_Project)

get_filename_component(HOST_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# ========== Host-Specific Configuration ==========
set(DEVICE "npu2" CACHE STRING "Device target: npu1 or npu2")
set(TRACE_SIZE "0" CACHE STRING "Trace size parameter")
set(ENABLE_CHESSCC OFF CACHE BOOL "Enable CHESSCC compiler (slow)")
set(HOME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Project home directory")
set(HOST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# ========== Derived Configurations ==========
set(IRON_ARGS ${DEVICE} ${TRACE_SIZE})

# ========== Load Modules ==========
# NOTE: kernel.cmake is handled by root CMakeLists.txt for shared kernel
include(${HOME_DIR}/makefiles/common.cmake)
# include(${HOME_DIR}/makefiles/kernel.cmake)  # Commented out - using shared kernel
include(${HOME_DIR}/makefiles/bitstream.cmake)
include(${HOME_DIR}/makefiles/mlir_bitstream.cmake)

# Set HOST_NAME for host.cmake
include(${HOME_DIR}/makefiles/host.cmake)

# ========== ${HOST_NAME}-Specific Targets ==========

# ${HOST_NAME} bitstream (unique to this host) - uses targets from bitstream.cmake
add_custom_target(${HOST_NAME}_bitstream DEPENDS 
  ${HOST_NAME}_generate_mlirs 
  ${HOST_NAME}_generate_xclbins 
  ${HOST_NAME}_generate_insts)

# ${HOST_NAME} instructions (unique to this host)
add_custom_target(${HOST_NAME}_instructions DEPENDS ${HOST_NAME}_generate_insts)

# ${HOST_NAME} executable (unique to this host)
add_custom_target(${HOST_NAME}_host)
add_dependencies(${HOST_NAME}_host npu_common ${HOST_C_TARGET} ${HOST_NAME}_all_python_bitstreams ${HOST_NAME}_all_mlir_bitstreams)

# Main ${HOST_NAME} target - depends on ${HOST_NAME}-specific builds
# Note: shared_kernel dependency is handled at root level
add_custom_target(${HOST_NAME}_all)
add_dependencies(${HOST_NAME}_all ${HOST_NAME}_bitstream ${HOST_NAME}_instructions ${HOST_NAME}_host)

# ========== Clean Target Configuration ==========
# Find all files starting with 'trace' in the build directory
file(GLOB_RECURSE TRACE_FILES "trace*")
file(GLOB_RECURSE LOG_FILES "*.log")
file(GLOB_RECURSE EXE_FILES "*.exe")

# Combine the lists into one
set(EXTRA_CLEAN_FILES
  ${TRACE_FILES}
  ${LOG_FILES}
  ${EXE_FILES}
)
set_directory_properties(PROPERTIES
  ADDITIONAL_MAKE_CLEAN_FILES "${EXTRA_CLEAN_FILES}"
)


# ========== ${HOST_NAME} route target ==========
# Assume route toolchain exists: aie-opt + aie-translate + visualize.py
if(${HOST_NAME}_GENERATED_MLIRS)
    list(GET ${HOST_NAME}_GENERATED_MLIRS 0 IRON_BOTH_MLIR_TARGET)
    add_custom_target(${HOST_NAME}_route
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/routes
      COMMAND aie-opt --aie-create-pathfinder-flows --aie-find-flows ${IRON_BOTH_MLIR_TARGET} |
              aie-translate --aie-flows-to-json > ${CMAKE_BINARY_DIR}/routes/route.json
      COMMAND python3 ${HOME_DIR}/visualize.py -j ${CMAKE_BINARY_DIR}/routes/route.json
      DEPENDS ${IRON_BOTH_MLIR_TARGET}
      COMMENT "Generating routing graph for ${HOST_NAME}"
    )
endif()



# ========== ${HOST_NAME} run target ==========
add_custom_target(${HOST_NAME}_run
  COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} ./${HOST_C_TARGET} | tee out.log
  DEPENDS ${HOST_NAME}_route
)
add_dependencies(${HOST_NAME}_run ${HOST_NAME}_all)

