# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# Hy-Memory-30B-A3B NPU Library Makefile
# =============================================================================
#
# Usage:
#   make        - Build libhy_mem_npu.so (copied to ../../build/lib)
#   make clean  - Remove all built files
#
# The engine has no host compute modules to link (no gemm / mha / dequant /
# lm_head objects): every layer and the lm_head run on their own xclbins.
# =============================================================================

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

BUILD_DIR := ../../build/hy_mem_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libhy_mem_npu.so

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I/opt/xilinx/xrt/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)
# make PREFILL_WITH_MV=1: prefill through the decode overlays one token at a time,
# instead of the batched prefill (hy_mem_prefill.hpp)
PREFILL_WITH_MV ?= 0
ifneq ($(PREFILL_WITH_MV), 0)
CXX_FLAGS += -DPREFILL_WITH_MV
endif
# where the batched prefill's heavy stages run: 1 = prefill_overlay.xclbin, 0 = host
# (HY_MEM_ATTN_HOST=1 / HY_MEM_MM_HOST=1 at run time also put them on the host)
HY_MEM_ATTN_ON_NPU ?= 1
CXX_FLAGS += -DHY_MEM_ATTN_ON_NPU=$(HY_MEM_ATTN_ON_NPU)
HY_MEM_MM_ON_NPU ?= 1
CXX_FLAGS += -DHY_MEM_MM_ON_NPU=$(HY_MEM_MM_ON_NPU)

# Flags for shared library
LIB_FLAGS := -shared
LIB_FLAGS += $(CXX_FLAGS)
# Each engine .so compiles its own copy of the shared runtime (npu_sequence,
# npu_app_manager, weight_desc_t, bytes/buffer, ...) and exports it with default
# visibility. flm links every engine at once, so ELF interposition makes all
# references -- including calls from inside this .so -- bind to whichever engine
# comes first in DT_NEEDED, silently running another model's implementation.
# -Bsymbolic-functions binds our internal function calls to our own definitions
# at link time. Functions only: typeinfo/vtables stay interposable so
# dynamic_cast across the .so boundary keeps working.
LIB_FLAGS += -Wl,-Bsymbolic-functions

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
LDFLAGS += -lxrt_coreutil -laiebu
endif
ifeq ($(FLM_USE_HRX),1)
# HRX path links no Boost (no engine source uses Boost); Linux engine .so stay Boost-free.
else
LDFLAGS += -lboost_program_options -lboost_filesystem
endif

# Each source compiles to a flattened object in LIB_DIR so dependency (.d)
# files are emitted next to the objects and -include picks up header changes.
LOCAL_SOURCES = $(wildcard *.cpp)
OBJS = $(patsubst %.cpp,$(LIB_DIR)/%.o,$(LOCAL_SOURCES))
DEPS := $(OBJS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(LIB_DIR)/%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(TARGET): $(OBJS)
	$(CXX) $(LIB_FLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
.PHONY: all clean directories

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe
HRX_CMAKE_ARGS :=
ifeq ($(FLM_USE_HRX),1)
HRX_CMAKE_ARGS := -DFLM_USE_HRX=ON
endif

TARGET = $(BUILD_DIR)/hy_mem_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/hy_mem_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target hy_mem_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean directories

endif
