# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# Host decode speed test for MiniCPM-OCR (Qwen3.5-0.8B text shape) (see host/README.md)
# =============================================================================
#
# Built against FLM_DLL's headers and q4nx reader (not ../../host_common, which
# has no Q4_K support), with the same compile flags as the FLM_DLL engine.
#
# =============================================================================

HOST_C_TARGET := run.exe
FLM_DLL_DIR := ../../../FLM_DLL

VERBOSE ?= 0
DEBUG_LEVEL ?= 0

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -Ihost
CXX_FLAGS += -I$(FLM_DLL_DIR)/include
CXX_FLAGS += -I$(FLM_DLL_DIR)/detail/q4nx
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)

LDFLAGS += -fopenmp
LDFLAGS += -L/opt/xilinx/xrt/lib -Wl,-rpath,/opt/xilinx/xrt/lib
LDFLAGS += -lxrt_coreutil $(FLM_DLL_DIR)/../../src/lib/xrt/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem

HOST_SOURCES := $(shell find host -name '*.cpp')
HOST_SOURCES += $(FLM_DLL_DIR)/detail/q4nx/q4_npu_eXpress.cpp
HOST_SOURCES += $(FLM_DLL_DIR)/detail/q4nx/safe_tensors.cpp
HOST_OBJECTS := $(patsubst %.cpp,build/host_obj/%.o,$(notdir $(HOST_SOURCES)))
HOST_DEPENDS := $(HOST_OBJECTS:.o=.d)
vpath %.cpp $(sort $(dir $(HOST_SOURCES)))

$(HOST_C_TARGET): build/$(HOST_C_TARGET)
	cp build/$(HOST_C_TARGET) ./

build/$(HOST_C_TARGET): $(HOST_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

build/host_obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

-include $(HOST_DEPENDS)
