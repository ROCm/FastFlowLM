# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Standalone lm_head test (host/test_lm_head.cpp), built like the gemma_npu engine:
# it compiles FLM_DLL's gemma_npu_sequence.cpp and the q4nx reader directly.

FLM_DLL_DIR := $(realpath ${HOME_DIR}/../../../FLM_DLL)
FLM_SRC_LIB_DIR := $(realpath ${HOME_DIR}/../../../../src/lib/xrt)
include ${FLM_DLL_DIR}/flags.mk

TEST_O_DIR := build/test
TEST_TARGET := ${TEST_O_DIR}/test_lm_head
TEST_CXX := g++-13
TEST_FLAGS := -std=c++23 -O2 -fopenmp $(FLM_MARCH) -mavx512bf16 -DVERBOSE=0
TEST_FLAGS += -I${FLM_DLL_DIR}/include -I${FLM_DLL_DIR}/detail/q4nx -I${FLM_DLL_DIR}/detail/gemma_npu
TEST_FLAGS += -I$(XRT_ROOT)/include
TEST_SRCS := ${HOME_DIR}/host/test_lm_head.cpp ${FLM_DLL_DIR}/detail/gemma_npu/gemma_npu_sequence.cpp
TEST_SRCS += $(wildcard ${FLM_DLL_DIR}/detail/q4nx/*.cpp)
TEST_LDFLAGS := -L$(XRT_ROOT)/lib -lxrt_coreutil ${FLM_SRC_LIB_DIR}/libaiebu.a -fopenmp

${TEST_TARGET}: ${TEST_SRCS}
	mkdir -p ${@D}
	${TEST_CXX} ${TEST_FLAGS} -o $@ ${TEST_SRCS} ${TEST_LDFLAGS}

test_lm_head: ${TEST_TARGET}

# Layer timing (host/test_layer_time.cpp)
LAYER_TIME_TARGET := ${TEST_O_DIR}/test_layer_time
LAYER_TIME_SRCS := ${HOME_DIR}/host/test_layer_time.cpp ${FLM_DLL_DIR}/detail/gemma_npu/gemma_npu_sequence.cpp
LAYER_TIME_SRCS += $(wildcard ${FLM_DLL_DIR}/detail/q4nx/*.cpp)

${LAYER_TIME_TARGET}: ${LAYER_TIME_SRCS} ${FLM_DLL_DIR}/detail/gemma_npu/gemma_bfp16_pack.hpp
	mkdir -p ${@D}
	${TEST_CXX} ${TEST_FLAGS} -o $@ ${LAYER_TIME_SRCS} ${TEST_LDFLAGS}

test_layer_time: ${LAYER_TIME_TARGET}
