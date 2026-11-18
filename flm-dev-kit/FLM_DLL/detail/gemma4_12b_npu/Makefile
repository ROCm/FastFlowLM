# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# Gemma4-12B NPU Library Makefile
#
#   make        - Build libgemma4_12b_npu.so
#   make clean  - Remove all built files
# =============================================================================

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

BUILD_DIR := ../../build/gemma4_12b_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

COMMON_LIB_DIR := ../../build/common/lib

TARGET = $(LIB_DIR)/libgemma4_12b_npu.so
VERBOSE ?= 0
DEBUG_LEVEL ?= 0

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
CXX_FLAGS += -MMD -MP
CXX_FLAGS += -DVERBOSE=$(VERBOSE) -DDEBUG_LEVEL=$(DEBUG_LEVEL)
MVPREFILL ?= 0
ifeq ($(MVPREFILL), 1)
CXX_FLAGS += -DMVPREFILL
endif

# Backend selection for the prefill: PREFILL_MM picks where the batched prefill
# matmuls run, PREFILL_ATTN picks where the prefill attention runs. PREFILL_ATTN=npu
# moves the global (full attention) layers onto attn_global.xclbin and the sliding
# window layers onto attn_sliding.xclbin. PREFILL_ATTN=cpu puts both kinds back on the
# host kernels.
#
# The image/video block rows are a third case -- neither of those two overlays can mask
# a bidirectional block -- and PREFILL_IMAGE_ATTN picks where their fix-up runs. npu
# sends the block rows of the *global* layers to attn_global_image.xclbin, installed as
# xclbins/<model>/attn_global_image.xclbin and loaded only once a prompt actually carries
# an image; the sliding layers keep the host kernel regardless, and so does everything
# when that bitstream is missing. PREFILL_IMAGE_ATTN=cpu pins all of it to the host.
PREFILL_MM ?= npu
ifeq ($(PREFILL_MM), cpu)
CXX_FLAGS += -DFLM_PREFILL_MM_ON_CPU
endif
PREFILL_ATTN ?= npu
ifeq ($(PREFILL_ATTN), cpu)
CXX_FLAGS += -DFLM_PREFILL_ATTN_ON_CPU
endif
PREFILL_IMAGE_ATTN ?= npu
ifeq ($(PREFILL_IMAGE_ATTN), cpu)
CXX_FLAGS += -DFLM_PREFILL_IMAGE_ATTN_ON_CPU
endif

# Backend for the three matmuls of the multi modal projector (image patches / audio
# frames -> language embeddings). MM_PROJ=npu runs them on the audio_image_mm overlay,
# which has to be installed as xclbins/<model>/mm_projector.xclbin. The norms and the
# position embedding stay on the host either way. Like PREFILL_MM this is an opt *out*:
# npu is the default and MM_PROJ=cpu is the fallback for a device-less machine.
MM_PROJ ?= npu
ifeq ($(MM_PROJ), cpu)
CXX_FLAGS += -DFLM_MM_PROJ_ON_CPU
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


# Sources are grouped per directory so each group gets its own object tree under
# $(OBJ_DIR). Compiling to objects (instead of feeding every .cpp to a single
# compile-and-link command) is what makes header tracking work: -MMD writes one
# .d next to each .o, and those .d files are what pull a rebuild when a header
# changes. In the old single-step build gcc emitted a single .d named after the
# -o argument, overwritten by each translation unit, so header edits were
# silently ignored.
OBJ_DIR := $(BUILD_DIR)/obj

SRC_LOCAL   := $(wildcard *.cpp)
SRC_DEQUANT := $(wildcard ../dequant/*.cpp)
SRC_GEMM    := $(wildcard ../gemm/*.cpp)
SRC_LM_HEAD := $(wildcard ../lm_head/*.cpp)
SRC_VISION  := ../vision_common/norm.cpp

SOURCES := $(SRC_LOCAL) $(SRC_DEQUANT) $(SRC_GEMM) $(SRC_LM_HEAD) $(SRC_VISION)

OBJECTS := $(patsubst %.cpp,$(OBJ_DIR)/gemma4_12b_npu/%.o,$(SRC_LOCAL))
OBJECTS += $(patsubst ../dequant/%.cpp,$(OBJ_DIR)/dequant/%.o,$(SRC_DEQUANT))
OBJECTS += $(patsubst ../gemm/%.cpp,$(OBJ_DIR)/gemm/%.o,$(SRC_GEMM))
OBJECTS += $(patsubst ../lm_head/%.cpp,$(OBJ_DIR)/lm_head/%.o,$(SRC_LM_HEAD))
OBJECTS += $(patsubst ../vision_common/%.cpp,$(OBJ_DIR)/vision_common/%.o,$(SRC_VISION))

DEPS := $(OBJECTS:.o=.d)

all: directories $(TARGET)
	cp $(TARGET) $(FINAL_LIB_DIR)

directories:
	mkdir -p $(FINAL_LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(OBJ_DIR)/gemma4_12b_npu/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/dequant/%.o: ../dequant/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/gemm/%.o: ../gemm/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/lm_head/%.o: ../lm_head/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/vision_common/%.o: ../vision_common/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
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

BUILD_DIR := ../../build/gemma4_12b_npu

TARGET = $(BUILD_DIR)/gemma4_12b_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/gemma4_12b_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target gemma4_12b_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)


.PHONY: all clean directories

endif 
