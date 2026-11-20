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

# Linux build environment
# Use g++-13 directly without CMake

HOST_C_TARGET := run.exe
HOST_COMMON_DIR := ../../host_common

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall
CXX_FLAGS += -DUSE_AVX2=1
CXX_FLAGS += -mavx2 -march=native -mfma -ffast-math
CXX_FLAGS += -Ihost
CXX_FLAGS += -I$(HOST_COMMON_DIR)
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP

LDFLAGS += -L/opt/xilinx/xrt/lib
LDFLAGS += -L$(HOST_COMMON_DIR)
LDFLAGS += -lxrt_coreutil $(HOST_COMMON_DIR)/../../../src/lib/xrt/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem

# Find all source files in host/ directory (including subdirectories)
HOST_SOURCES := $(shell find host -name '*.cpp')
HOST_OBJECTS := $(HOST_SOURCES:%.cpp=build/%.o)
HOST_DEPENDS := $(HOST_OBJECTS:.o=.d)

# Default target
$(HOST_C_TARGET): build/$(HOST_C_TARGET)
	cp build/$(HOST_C_TARGET) ./

# Link the executable
build/$(HOST_C_TARGET): $(HOST_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

# Compile source files and generate dependencies
build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Include dependency files
-include $(HOST_DEPENDS)