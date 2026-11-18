# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

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

BUILD_DIR := ../../build/minicpm_v_4_7_moe_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

DEBUG_LEVEL ?= 0

TARGET = $(LIB_DIR)/libminicpm_v_4_7_moe_npu.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
CXX_FLAGS += -I.
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I/opt/xilinx/xrt/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL)
# CXX_FLAGS += -DMVPREFILL
VISION_PROFILE ?= 0
ifeq ($(VISION_PROFILE), 1)
CXX_FLAGS += -DMINICPM_V_MOE_VISION_PROFILE
endif

# Flags for shared library
LIB_FLAGS := -shared
LIB_FLAGS += $(CXX_FLAGS)
# Every engine .so compiles its own copy of the shared runtime (npu_sequence,
# npu_app_manager, weight_desc_t, gen_mha_main, simd_*, ...) and exports it with
# default visibility. flm links all of them, so ELF interposition makes every
# reference -- including calls from inside this .so -- bind to whichever engine
# appears first in DT_NEEDED. This .so is 9th, so it silently ran other models'
# stale implementations. -Bsymbolic-functions binds our internal function calls
# to our own definitions at link time. Functions only: typeinfo/vtables stay
# interposable so dynamic_cast across the .so boundary keeps working.
LIB_FLAGS += -Wl,-Bsymbolic-functions

# Flags for test executable
TEST_FLAGS := $(CXX_FLAGS)
TEST_FLAGS += -L$(LIB_DIR)

ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_RPATH)
else
LDFLAGS += -L/opt/xilinx/xrt/lib
endif

LDFLAGS += -fopenmp
LDFLAGS += -L../../external_libs
LDFLAGS += -L../../build/lib
ifeq ($(FLM_USE_HRX),1)
LDFLAGS += $(HRX_LIBS)
else
# Link the aiebu that matches ../../include/aiebu by path. With -laiebu, an
# XRT installed in /opt/xilinx/xrt/lib (searched first) supplies its own
# libaiebu.a, whose assembler prints a "Header version ..." banner to stdout
# on every _setup_kernel().
LDFLAGS += -lxrt_coreutil ../../external_libs/libaiebu.a
endif
ifeq ($(FLM_USE_HRX),1)
# HRX path links no Boost (no engine source uses Boost); Linux engine .so stay Boost-free.
else
LDFLAGS += -lboost_program_options -lboost_filesystem
endif


LOCAL_SOURCES = $(wildcard *.cpp) $(wildcard image/*.cpp) $(wildcard image/npu/*.cpp)

# Each source compiles to a flattened object in LIB_DIR so dependency (.d)
# files are emitted next to the objects and -include picks up header changes.
LOCAL_OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))

# Compile dequant with hidden visibility to avoid runtime symbol collision
# with other .so files (e.g. libqwen2_npu.so) that export the same Dequant:: symbols.

OBJS = $(LOCAL_OBJS)

DEPS := $(OBJS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(LIB_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
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

BUILD_DIR := ../../build/minicpm_v_4_7_moe_npu

TARGET = $(BUILD_DIR)/minicpm_v_4_7_moe_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/minicpm_v_4_7_moe_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target minicpm_v_4_7_moe_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
