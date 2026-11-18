# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# Qwen NPU Library Makefile
# =============================================================================
#
# Created: 2025-06-30
#
# This Makefile is used to build the Qwen NPU library components.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================


# Linux build environment
# Use g++-13 directly without CMake

BUILD_DIR := build
COMMON_DIR := ../../libraries/common

HOST_C_TARGET = run.exe

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall
CXX_FLAGS += -DUSE_AVX2=1
CXX_FLAGS += -mavx2 -march=native -mfma -ffast-math
CXX_FLAGS += -I../../libraries/include
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP -DLINUX	

# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L../../libraries/build/lib
TEST_FLAGS += -L/opt/xilinx/xrt/lib
TEST_FLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib

LDFLAGS += -lxrt_coreutil
LDFLAGS += -lboost_program_options -lboost_filesystem
LDFLAGS += -L../../libraries/build/tokenizer/lib
LDFLAGS += -ltokenizers_cpp -ltokenizers_c -l sentencepiece
LDFLAGS += -lq4_npu_eXpress -lqwen_npu -lllama_npu -lgemma_text_npu -lnpu_utils

TEST_SOURCES += host/host.cpp
TEST_SOURCES += $(COMMON_DIR)/npu_utils/npu_utils.cpp
TEST_SOURCES += $(COMMON_DIR)/npu_utils/npu_instr_utils.cpp
TEST_SOURCES += $(COMMON_DIR)/modules/sampler.cpp
TEST_SOURCES += $(COMMON_DIR)/tokenizer/tokenizer.cpp
TEST_HEADERS += $(wildcard host/*.hpp)

DEPS = $(TEST_SOURCES:.cpp=.d)
-include $(DEPS)

$(HOST_C_TARGET): $(BUILD_DIR)/$(HOST_C_TARGET)
	cp $(BUILD_DIR)/$(HOST_C_TARGET) ./


$(BUILD_DIR)/$(HOST_C_TARGET): $(TEST_SOURCES) $(TEST_HEADERS)
	$(CXX) $(TEST_FLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: all clean test directories
