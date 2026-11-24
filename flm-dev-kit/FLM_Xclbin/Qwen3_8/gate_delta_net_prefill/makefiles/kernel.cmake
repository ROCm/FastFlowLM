# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# kernel.cmake
# Must include after defining: HOME_DIR, DEVICE, ENABLE_CHESSCC, PEANO_INSTALL_DIR, KERNEL_DIR, KERNEL_O_DIR, KERNEL_SRCS, KERNEL_HEADERS

# Verify required variables are defined
if(NOT DEFINED KERNEL_DIR)
    message(FATAL_ERROR "KERNEL_DIR must be defined before including kernel.cmake")
endif()
if(NOT DEFINED KERNEL_O_DIR)
    message(FATAL_ERROR "KERNEL_O_DIR must be defined before including kernel.cmake")
endif()

# set(KERNEL_DIR "${HOME_DIR}/kernel")
message(STATUS "DEBUGGING KERNEL_DIR: ${KERNEL_DIR}")
# set(KERNEL_O_DIR "${CMAKE_BINARY_DIR}/kernel")
message(STATUS "DEBUGGING KERNEL_O_DIR: ${KERNEL_O_DIR}")
file(MAKE_DIRECTORY ${KERNEL_O_DIR})


# Gather source and headers
# file(GLOB KERNEL_SRCS "${KERNEL_DIR}/*.cc")
# file(GLOB KERNEL_HEADERS "${KERNEL_DIR}/*.h")
# Verify kernel sources and headers are defined
if(NOT DEFINED KERNEL_SRCS)
    message(FATAL_ERROR "KERNEL_SRCS must be defined before including kernel.cmake")
endif()
if(NOT DEFINED KERNEL_HEADERS)
    message(FATAL_ERROR "KERNEL_HEADERS must be defined before including kernel.cmake")
endif()

# Create object targets manually (for ASM + OBJDUMP)
set(KERNEL_OBJECTS "")
foreach(SRC ${KERNEL_SRCS})
    get_filename_component(FILENAME_WE ${SRC} NAME_WE)
    set(OBJ "${KERNEL_O_DIR}/${FILENAME_WE}.o")
    set(ASM "${KERNEL_O_DIR}/${FILENAME_WE}.asm")
    set(OBJDUMP "${KERNEL_O_DIR}/${FILENAME_WE}.s")

    # Determine compile command
    if(DEVICE STREQUAL "npu1")
        if(ENABLE_CHESSCC)
            add_custom_command(
                OUTPUT ${OBJ}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_O_DIR}
                COMMAND ${CMAKE_COMMAND} -E chdir ${KERNEL_O_DIR}
                        xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c ${SRC} -o ${FILENAME_WE}.o
                DEPENDS ${SRC} ${KERNEL_HEADERS}
                COMMENT "Compiling kernel ${SRC} with xchesscc (NPU1)"
            )
        else()
            add_custom_command(
                OUTPUT ${OBJ} ${ASM} ${OBJDUMP}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_O_DIR}
                COMMAND $ENV{PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${SRC} -o ${OBJ}
                COMMAND $ENV{PEANO_INSTALL_DIR}/bin/llvm-objdump -d -g -S -s -t -T -x --full-contents ${OBJ} > ${OBJDUMP}
                COMMAND $ENV{PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -S -fverbose-asm -o ${ASM} ${SRC}
                DEPENDS ${SRC} ${KERNEL_HEADERS}
                COMMENT "Compiling kernel ${SRC} with Peano Clang++ (NPU1)"
            )
        endif()
    elseif(DEVICE STREQUAL "npu2")
        if(ENABLE_CHESSCC)
            add_custom_command(
                OUTPUT ${OBJ}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_O_DIR}
                COMMAND ${CMAKE_COMMAND} -E chdir ${KERNEL_O_DIR}
                        xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -DNPU2 -c ${SRC} -o ${FILENAME_WE}.o
                DEPENDS ${SRC} ${KERNEL_HEADERS}
                COMMENT "Compiling kernel ${SRC} with xchesscc (NPU2)"
            )
        else()
            add_custom_command(
                OUTPUT ${OBJ} ${ASM} ${OBJDUMP}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${KERNEL_O_DIR}
                COMMAND $ENV{PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c ${SRC} -o ${OBJ}
                COMMAND $ENV{PEANO_INSTALL_DIR}/bin/llvm-objdump -d -g -S -s -t -T -x --full-contents ${OBJ} > ${OBJDUMP}
                COMMAND $ENV{PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -S -fverbose-asm -o ${ASM} ${SRC}
                DEPENDS ${SRC} ${KERNEL_HEADERS}
                COMMENT "Compiling kernel ${SRC} with Peano Clang++ (NPU2)"
            )
        endif()
    else()
        message(FATAL_ERROR "Unsupported DEVICE: ${DEVICE}")
    endif()

    list(APPEND KERNEL_OBJECTS ${OBJ})
endforeach()

# Export for use in other modules
set(KERNEL_OBJS ${KERNEL_OBJECTS})

# # Optionally, define a phony target to trigger kernel build
# add_custom_target(build_kernels ALL DEPENDS ${KERNEL_OBJECTS})
