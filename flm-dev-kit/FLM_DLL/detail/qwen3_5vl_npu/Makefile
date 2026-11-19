# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# Qwen NPU Library Makefile
# =============================================================================
#
# Created: 2024
#
# This Makefile is used to build the Qwen NPU library components.
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

BUILD_DIR := ../../build/qwen3_5vl_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libqwen3_5vl_npu.so

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
# CXX_FLAGS += -DMVPREFILL

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
VISION_SOURCES = $(wildcard ../vision_common/*.cpp)

LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))
GEMM_OBJS = $(patsubst %.cpp,$(LIB_DIR)/gemm_%.o,$(notdir $(GEMM_SOURCES)))
LMHEAD_OBJS = $(patsubst %.cpp,$(LIB_DIR)/lm_head_%.o,$(notdir $(LMHEAD_SOURCES)))
VISION_OBJS = $(patsubst %.cpp,$(LIB_DIR)/vision_%.o,$(notdir $(VISION_SOURCES)))

# Compile dequant with hidden visibility to avoid runtime symbol collision
# with other .so files (e.g. libqwen2_npu.so) that export the same Dequant:: symbols.
DEQUANT_SOURCES = $(wildcard ../dequant/*.cpp)
DEQUANT_OBJS = $(patsubst %.cpp,$(LIB_DIR)/dequant_%.o,$(notdir $(DEQUANT_SOURCES)))

OBJS = $(LOCAL_OBJS) $(GEMM_OBJS) $(LMHEAD_OBJS) $(VISION_OBJS)

DEPS := $(OBJS:.o=.d) $(DEQUANT_OBJS:.o=.d)

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

$(LIB_DIR)/vision_%.o: ../vision_common/%.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(LIB_DIR)/dequant_%.o: ../dequant/%.cpp
	$(CXX) $(CXX_FLAGS) -fvisibility=hidden -c -o $@ $<

$(TARGET): $(OBJS) $(DEQUANT_OBJS)
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

BUILD_DIR := ../../build/qwen3_5vl_npu

TARGET = $(BUILD_DIR)/qwen3_5vl_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/qwen3_5vl_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target qwen3_5vl_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
