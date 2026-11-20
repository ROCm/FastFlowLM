# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# Llama NPU Library Makefile
# =============================================================================
#
# Created: 2024
#
# This Makefile is used to build the Llama NPU library components.
#
# Usage:
#   make        - Build all targets
#   make clean  - Remove all built files
#   make help   - Show this help message
#
# =============================================================================

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

ifeq ($(WSL), 0)
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
CXX_FLAGS += -MMD -MP

# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L../../libraries/build/lib
TEST_FLAGS += -Wl,-rpath,../../libraries/build/lib
TEST_FLAGS += -L/opt/xilinx/xrt/lib
TEST_FLAGS += -Wl,-rpath,/opt/xilinx/xrt/lib

LDFLAGS += -lxrt_coreutil -lq4_npu_eXpress -lnpu_utils -lgpt_oss_npu
LDFLAGS += -lgemm -ldequant -llm_head
LDFLAGS += -lboost_program_options -lboost_filesystem
# LDFLAGS += -L../../libraries/build/tokenizer/lib
# LDFLAGS += -ltokenizers_cpp -ltokenizers_c -l sentencepiece

SOURCES = $(wildcard host/*.cpp)
TEST_SOURCES += $(SOURCES)

$(HOST_C_TARGET): $(BUILD_DIR)/$(HOST_C_TARGET)
	cp $(BUILD_DIR)/$(HOST_C_TARGET) ./


$(BUILD_DIR)/$(HOST_C_TARGET): $(TEST_SOURCES)
	$(CXX) $(TEST_FLAGS) -o $@ $^ $(LDFLAGS)



.PHONY: all clean test directories


else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

BUILD_DIR := ../build/llama_npu

TARGET = $(BUILD_DIR)/llama_npu.so

endif 