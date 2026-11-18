# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Host-side scoring harness for libminicpm_v_4_7_npu.so. See README.md.
#
#   make            build run_engine and run_vision_e2e against ../../../build/lib/libminicpm_v_4_7_npu.so
#   make clean
#
# Lives in its own directory because the engine Makefile compiles every *.cpp in
# detail/minicpm_v_4_7_npu/ into the .so.

CXX := g++-13
LIB := ../../../build/lib
BUILD := ../../../build/minicpm_v_4_7_npu/host_ref
DEBUG_LEVEL ?= 1

CXX_FLAGS := -std=c++20 -O2 -fopenmp -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native
CXX_FLAGS += -I../../../include -I../../q4nx -I/opt/xilinx/xrt/include
CXX_FLAGS += -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=0
LDFLAGS := -L$(LIB) -Wl,-rpath,$(abspath $(LIB)) -lminicpm_v_4_7_npu -lq4_npu_eXpress
LDFLAGS += -L/opt/xilinx/xrt/lib -lxrt_coreutil -fopenmp

all: $(BUILD)/run_engine $(BUILD)/run_vision_e2e

$(BUILD)/run_engine: run_engine.cpp
	mkdir -p $(BUILD)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(LDFLAGS)

# Reads vision_ref.safetensors through the tower's own header-only reader.
$(BUILD)/run_vision_e2e: run_vision_e2e.cpp
	mkdir -p $(BUILD)
	$(CXX) $(CXX_FLAGS) -I.. -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
