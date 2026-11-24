# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

HOST_C_TARGET := run.exe
HOST_COMMON_DIR := ../../host_common
CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O2 -march=native -fopenmp
CXX_FLAGS += -Ihost -I$(HOST_COMMON_DIR) -I/opt/xilinx/xrt/include -MMD -MP

LDFLAGS += -L/opt/xilinx/xrt/lib
LDFLAGS += -lxrt_coreutil $(HOST_COMMON_DIR)/../../../src/lib/xrt/libaiebu.a
LDFLAGS += -lboost_program_options -lboost_filesystem -fopenmp

HOST_SOURCES := $(shell find host -name '*.cpp')
HOST_OBJECTS := $(HOST_SOURCES:%.cpp=build/%.o)
HOST_DEPENDS := $(HOST_OBJECTS:.o=.d)

$(HOST_C_TARGET): build/$(HOST_C_TARGET)
	cp build/$(HOST_C_TARGET) ./

build/$(HOST_C_TARGET): $(HOST_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

-include $(HOST_DEPENDS)
