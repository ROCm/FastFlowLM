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

BUILD_DIR := ../../build/gpt_oss_npu
FINAL_LIB_DIR := ../../build/lib
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
VERBOSE ?= 0
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

TARGET = $(LIB_DIR)/libgpt_oss_npu.so

CXX := g++-13
include ../../flags.mk
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += $(FLM_MARCH) -ffast-math
# gpt_mlp.hpp and gpt_router.hpp call AVX-512 BF16 intrinsics without a guard.
# FLM_MARCH (the product's AVX-512 set) does not enable them; every Ryzen AI
# host (Zen 4 or later) has them.
CXX_FLAGS += -mavx512bf16
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I$(XRT_ROOT)/include
endif
CXX_FLAGS += -MMD -MP
CXX_FLAGS += -DVERBOSE=$(VERBOSE)

# Flags for shared library
LIB_FLAGS := -shared
LIB_FLAGS += $(CXX_FLAGS)
# Each engine .so compiles its own copy of the shared runtime (npu_sequence,
# npu_app_manager, weight_desc_t, bytes/buffer, gen_mha_main, simd_*, ...) and
# exports it with default visibility. flm links every engine at once, so ELF
# interposition makes all references -- including calls from inside this .so --
# bind to whichever engine comes first in DT_NEEDED, silently running another
# model's implementation. -Bsymbolic-functions binds our internal function calls
# to our own definitions at link time. Functions only: typeinfo/vtables stay
# interposable so dynamic_cast across the .so boundary keeps working.
LIB_FLAGS += -Wl,-Bsymbolic-functions

# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L$(LIB_DIR)

ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_RPATH)
else
LDFLAGS += -L$(XRT_ROOT)/lib
endif

LDFLAGS += -L../../../../src/lib/xrt
LDFLAGS += -L../../build/lib
ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_LIBS)
else
# Link the aiebu that matches ../../include/aiebu by path. With -laiebu, an
# XRT installed in $(XRT_ROOT)/lib (searched first) supplies its own
# libaiebu.a, whose assembler prints a "Header version ..." banner to stdout
# on every _setup_kernel().
LDFLAGS += -lxrt_coreutil ../../../../src/lib/xrt/libaiebu.a
endif
LDFLAGS += -fopenmp


SOURCES = $(wildcard *.cpp)

HEADERS = $(wildcard *.hpp)
HEADERS += $(wildcard ../include/*.hpp)

DEPS := $(SOURCES:.cpp=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): $(SOURCES)
	$(CXX) $(LIB_FLAGS) -o $@ $^ $(LDFLAGS)


clean:
	rm -rf $(BUILD_DIR)


-include $(DEPS)
.PHONY: all clean test directories

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe
HRX_CMAKE_ARGS :=
ifeq ($(FLM_USE_HRX),1)
HRX_CMAKE_ARGS := -DFLM_USE_HRX=ON
endif

BUILD_DIR := ../../build/gpt_oss_npu

TARGET = $(BUILD_DIR)/gpt_oss_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/gpt_oss_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target gpt_oss_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
