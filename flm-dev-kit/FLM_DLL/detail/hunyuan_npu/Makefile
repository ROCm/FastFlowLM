# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# Hunyuan NPU Library Makefile
# =============================================================================
#
# Created: 2024
#
# This Makefile is used to build the Hunyuan NPU library components.
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

BUILD_DIR := ../../build/hunyuan_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libhunyuan_npu.so

CXX := g++-13
include ../../flags.mk
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += $(FLM_MARCH) -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I$(XRT_ROOT)/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)
# Feed the prompt through the decode overlay one token at a time instead of the
# host-side batched matmul. Comment out to go back to the mm prefill.
CXX_FLAGS += -DMVPREFILL=0
# make PREFILL_PROFILE=1 -- prints a per-call breakdown of the mm prefill path
# (embedding / layers / lm_head / total + tok/s). Off by default: the macros
# compile to nothing, so there is no cost in a normal build.
PREFILL_PROFILE ?= 0
ifneq ($(PREFILL_PROFILE), 0)
CXX_FLAGS += -DHUNYUAN_PREFILL_PROFILE
endif
# make RUN_TRACE=1 BUILD_DIR=../../build/hunyuan_npu_trace -- start / finish of every NPU
# run into a memory-mapped ring file (HY_RUN_TRACE); see "Run trace" in README.md
RUN_TRACE ?= 0
ifneq ($(RUN_TRACE), 0)
CXX_FLAGS += -DHUNYUAN_RUN_TRACE
endif
# make HOST_ATTN=1 BUILD_DIR=../../build/hunyuan_npu_ref -- prefill and decode
# attention on the host in fp32: the reference for long-context checks.
HOST_ATTN ?= 0
ifneq ($(HOST_ATTN), 0)
CXX_FLAGS += -DHUNYUAN_ATTN_ON_NPU=0 -DHUNYUAN_DECODE_ON_NPU=0
endif
# make HOST_MM=1 ... -- the prefill matmuls on the host too (with HOST_ATTN=1: the
# whole graph on the host except the lm_head)
HOST_MM ?= 0
ifneq ($(HOST_MM), 0)
CXX_FLAGS += -DHUNYUAN_MM_ON_NPU=0
endif

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

LDFLAGS += -fopenmp
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


# Each source compiles to a flattened object in LIB_DIR so dependency (.d)
# files are emitted next to the objects and -include picks up header changes.
LOCAL_SOURCES = $(wildcard *.cpp)
GEMM_SOURCES = $(wildcard ../gemm/*.cpp)
LMHEAD_SOURCES = $(wildcard ../lm_head/*.cpp)

LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))
GEMM_OBJS = $(patsubst %.cpp,$(LIB_DIR)/gemm_%.o,$(notdir $(GEMM_SOURCES)))
LMHEAD_OBJS = $(patsubst %.cpp,$(LIB_DIR)/lm_head_%.o,$(notdir $(LMHEAD_SOURCES)))

# No dequant objects: this engine uses the fused dequant_mm (on the host for
# now, on the mm overlay later) and never calls the standalone Dequant module.
OBJS = $(LOCAL_OBJS) $(GEMM_OBJS) $(LMHEAD_OBJS)

DEPS := $(OBJS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(LIB_DIR)/%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/gemm_%.o: ../gemm/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/lm_head_%.o: ../lm_head/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
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

BUILD_DIR := ../../build/hunyuan_npu

TARGET = $(BUILD_DIR)/hunyuan_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/hunyuan_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target hunyuan_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
