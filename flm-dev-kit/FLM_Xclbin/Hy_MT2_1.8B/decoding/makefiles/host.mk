# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Host for this folder: host/bench_decode.cpp, built against the FLM_DLL engine
# code (FLM_DLL/detail/hunyuan_npu) so it runs the same sequences as production.

HOST_C_TARGET := bench_decode.exe
FLM_DLL_DIR := ../../../FLM_DLL
ENGINE_DIR := $(FLM_DLL_DIR)/detail/hunyuan_npu

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -I$(FLM_DLL_DIR)/include -I$(FLM_DLL_DIR)/detail/q4nx -I$(ENGINE_DIR)
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=0 -DVERBOSE=0

LDFLAGS := -L/opt/xilinx/xrt/lib -L$(FLM_DLL_DIR)/build/lib
LDFLAGS += -Wl,-rpath,$(realpath $(FLM_DLL_DIR)/build/lib) -Wl,-rpath,/opt/xilinx/xrt/lib
LDFLAGS += -lq4_npu_eXpress -lxrt_coreutil $(FLM_DLL_DIR)/external_libs/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem -fopenmp

HOST_OBJECTS := build/host/bench_decode.o build/host/hunyuan_npu_sequence.o

$(HOST_C_TARGET): $(HOST_OBJECTS)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

build/host/bench_decode.o: host/bench_decode.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

build/host/hunyuan_npu_sequence.o: $(ENGINE_DIR)/hunyuan_npu_sequence.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

-include $(HOST_OBJECTS:.o=.d)
