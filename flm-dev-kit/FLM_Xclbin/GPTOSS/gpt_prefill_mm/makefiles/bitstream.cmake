# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Assume: HOME_DIR and HOST_DIR are already defined
# For example:
# set(HOME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
# set(HOST_DIR ${HOME_DIR}/host)

file(GLOB IRON_BOTH_SRC "${HOST_DIR}/iron/*_aie.py")
file(GLOB OTHER_IRON_SRC "${HOST_DIR}/iron/*.py")

set(MLIR_O_DIR ${CMAKE_CURRENT_BINARY_DIR}/build/mlir)
set(BITSTREAM_O_DIR ${CMAKE_CURRENT_BINARY_DIR}/build/bitstream)
set(XCLBIN_O_DIR ${CMAKE_CURRENT_BINARY_DIR}/build/xclbins)
set(INSTS_O_DIR ${CMAKE_CURRENT_BINARY_DIR}/build/insts)

set(GENERATED_MLIRS "")
set(GENERATED_XCLBINS "")
set(GENERATED_INSTS "")
message(STATUS "IRON_ARGS: ${IRON_ARGS}")
message(STATUS "IRON_BOTH_SRC: ${IRON_BOTH_SRC}")
message(STATUS "KERNEL_OBJS: ${KERNEL_OBJS}")
foreach(SRC ${IRON_BOTH_SRC})
    get_filename_component(NAME_WE ${SRC} NAME_WE)
    string(REPLACE "_aie" "" BASE_NAME ${NAME_WE})

    set(MLIR_TARGET "${MLIR_O_DIR}/${BASE_NAME}.mlir")
    set(XCLBIN_TARGET "${BITSTREAM_O_DIR}/${BASE_NAME}.xclbin")
    set(INSTS_TARGET "${BITSTREAM_O_DIR}/${BASE_NAME}.txt")

    # MLIR generation
    add_custom_command(
        OUTPUT ${MLIR_TARGET}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MLIR_O_DIR}
        COMMAND python3 ${SRC} ${IRON_ARGS} > ${MLIR_TARGET}
        DEPENDS ${SRC} ${OTHER_IRON_SRC}
        COMMENT "Generating MLIR: ${MLIR_TARGET}"
    )
    list(APPEND GENERATED_MLIRS ${MLIR_TARGET})

    # xclbin generation
    add_custom_command(
        OUTPUT ${XCLBIN_TARGET}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BITSTREAM_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${KERNEL_OBJS} ${BITSTREAM_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${BITSTREAM_O_DIR}
                aiecc.py ${AIECC_FLAGS} --aie-generate-xclbin --xclbin-name=${BASE_NAME}.xclbin ${MLIR_TARGET}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${XCLBIN_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${XCLBIN_TARGET} ${XCLBIN_O_DIR}/
        DEPENDS ${MLIR_TARGET} ${KERNEL_OBJS}
        COMMENT "Generating XCLBIN: ${XCLBIN_TARGET}"
    )
    list(APPEND GENERATED_XCLBINS ${XCLBIN_TARGET})

    # insts generation
    add_custom_command(
        OUTPUT ${INSTS_TARGET}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BITSTREAM_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E chdir ${BITSTREAM_O_DIR}
                aiecc.py -n ${AIECC_FLAGS} --aie-generate-npu-insts --npu-insts-name=${BASE_NAME}.txt
                ${MLIR_O_DIR}/${BASE_NAME}.mlir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${INSTS_O_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy ${INSTS_TARGET} ${INSTS_O_DIR}/
        DEPENDS ${MLIR_TARGET}
        COMMENT "Generating NPU INSTRUCTIONS: ${INSTS_TARGET}"
    )
    list(APPEND GENERATED_INSTS ${INSTS_TARGET})
endforeach()

message(STATUS "MLIR_TARGET: ${MLIR_TARGET}")
message(STATUS "GENERATED_MLIRS: ${GENERATED_MLIRS}")
message(STATUS "GENERATED_XCLBINS: ${GENERATED_XCLBINS}")
message(STATUS "GENERATED_INSTS: ${GENERATED_INSTS}")

# Get host-specific prefix for target names
get_filename_component(HOST_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add master targets with host-specific names
add_custom_target(${HOST_PREFIX}_generate_mlirs DEPENDS ${KERNEL_OBJS} ${GENERATED_MLIRS})
add_custom_target(${HOST_PREFIX}_generate_xclbins DEPENDS ${HOST_PREFIX}_generate_mlirs  ${GENERATED_XCLBINS})
add_custom_target(${HOST_PREFIX}_generate_insts DEPENDS  ${HOST_PREFIX}_generate_xclbins ${GENERATED_INSTS})

add_custom_target(${HOST_PREFIX}_all_python_bitstreams
    DEPENDS ${HOST_PREFIX}_generate_mlirs ${HOST_PREFIX}_generate_xclbins ${HOST_PREFIX}_generate_insts
)

# Export variables for use by host CMakeLists.txt
set(${HOST_PREFIX}_GENERATED_MLIRS ${GENERATED_MLIRS})
set(${HOST_PREFIX}_GENERATED_XCLBINS ${GENERATED_XCLBINS})
set(${HOST_PREFIX}_GENERATED_INSTS ${GENERATED_INSTS})
