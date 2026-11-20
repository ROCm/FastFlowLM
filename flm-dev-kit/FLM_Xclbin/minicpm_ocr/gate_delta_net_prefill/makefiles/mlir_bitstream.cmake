# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# mlir_bitstream.cmake
# Must be included after: HOME_DIR, HOST_DIR, DEVICE, ENABLE_CHESSCC, KERNEL_OBJS

set(BITSTREAM_O_DIR "${CMAKE_CURRENT_BINARY_DIR}/build/bitstream")
set(XCLBIN_O_DIR "${CMAKE_CURRENT_BINARY_DIR}/build/xclbins")
set(INSTS_O_DIR "${CMAKE_CURRENT_BINARY_DIR}/build/insts")

# Collect all MLIR input files
file(GLOB MLIR_BOTH_SRC "${HOST_DIR}/mlir/*.mlir")



# Accumulators
set(GENERATED_XCLBINS "")
set(GENERATED_INSTS "")

foreach(MLIR_FILE ${MLIR_BOTH_SRC})
    get_filename_component(BASE_NAME ${MLIR_FILE} NAME_WE)
    set(XCLBIN_PATH "${BITSTREAM_O_DIR}/${BASE_NAME}.xclbin")
    set(INSTS_PATH "${BITSTREAM_O_DIR}/${BASE_NAME}.txt")

    # Generate xclbin
    add_custom_command(
        OUTPUT ${XCLBIN_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BITSTREAM_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${KERNEL_OBJS} ${BITSTREAM_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${BITSTREAM_O_DIR}
                aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin
                --xclbin-name=${BASE_NAME}.xclbin ${MLIR_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${XCLBIN_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${XCLBIN_PATH} ${XCLBIN_O_DIR}/
        DEPENDS ${MLIR_FILE} ${KERNEL_OBJS}
        COMMENT "Generating XCLBIN from MLIR: ${XCLBIN_PATH}"
    )
    list(APPEND GENERATED_XCLBINS ${XCLBIN_PATH})

    # Generate instructions
    add_custom_command(
        OUTPUT ${INSTS_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BITSTREAM_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${BITSTREAM_O_DIR}
                aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts
                --npu-insts-name=${BASE_NAME}.txt ${MLIR_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTS_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${INSTS_PATH} ${INSTS_O_DIR}/
        DEPENDS ${MLIR_FILE}
        COMMENT "Generating NPU instructions: ${INSTS_PATH}"
    )
    list(APPEND GENERATED_INSTS ${INSTS_PATH})
endforeach()

# Get host-specific prefix for target names
get_filename_component(HOST_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add top-level targets with host-specific names
add_custom_target(${HOST_PREFIX}_generate_xclbins_from_mlir DEPENDS ${KERNEL_OBJS} ${GENERATED_XCLBINS})
add_custom_target(${HOST_PREFIX}_generate_insts_from_mlir DEPENDS ${GENERATED_INSTS})

add_custom_target(${HOST_PREFIX}_all_mlir_bitstreams
    DEPENDS ${HOST_PREFIX}_generate_xclbins_from_mlir ${HOST_PREFIX}_generate_insts_from_mlir
)

# Export variables for use by host CMakeLists.txt
set(${HOST_PREFIX}_MLIR_GENERATED_XCLBINS ${GENERATED_XCLBINS})
set(${HOST_PREFIX}_MLIR_GENERATED_INSTS ${GENERATED_INSTS})
