# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# =============================================================================
# Inkling-2-Mini NPU Library Makefile
# =============================================================================
#
# Usage:
#   make        - Build libinkling2_npu.so (copied to ../../build/lib)
#   make clean  - Remove all built files
#
# The engine compiles its own host model (inkling2_host.cpp); no shared host
# compute modules are linked.
# =============================================================================

# Check if it is in Windows Subsystem for Linux (WSL)
ifeq ($(shell uname -a | grep -i WSL),)
	WSL := 0
else
	WSL := 1
endif

BUILD_DIR := ../../build/inkling2_npu
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin
FINAL_LIB_DIR := ../../build/lib
ifeq ($(WSL), 0)
# Linux build environment
# Use g++-13 directly without CMake

DEBUG_LEVEL ?= 0
VERBOSE ?= 0

TARGET = $(LIB_DIR)/libinkling2_npu.so

CXX := g++-13
include ../../flags.mk
CXX_FLAGS := -std=c++20 -fPIC -Wall -O3 -fopenmp
CXX_FLAGS += $(FLM_MARCH) -ffast-math
# the loop controller uses AVX-512 BF16 intrinsics (vdpbf16ps). FLM_MARCH (the product AVX-512 set) does not
# enable them; every Ryzen AI host (Zen 4 or later) has them.
CXX_FLAGS += -mavx512bf16
CXX_FLAGS += -fmax-errors=1
CXX_FLAGS += -I../../include
CXX_FLAGS += -I../q4nx
# the instruction-stream ELF writer (launch::txn::to_elf), header-only
CXX_FLAGS += -I../../../aie2p-launchlib/launchlib/include
ifeq ($(FLM_USE_HRX),1)
include ../../hrx.mk
CXX_FLAGS += $(HRX_INC)
else
CXX_FLAGS += -I$(XRT_ROOT)/include
endif
CXX_FLAGS += -MMD -MP -DDEBUG_LEVEL=$(DEBUG_LEVEL) -DVERBOSE=$(VERBOSE)

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

# The engine-level harness (tools/inkling2_engine_run.cpp): reference prompts through the
# causal_lm interface, plus the saved-state checks. tools/inkling2_host_run.cpp drives the
# host model alone.
harness: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/inkling2_engine_run tools/inkling2_engine_run.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -linkling2_npu -lq4_npu_eXpress $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/inkling2_host_run tools/inkling2_host_run.cpp inkling2_host.cpp

# tools/inkling2_fused_probe.cpp: NPU decode (inkling2_fused) step by step against the host decode trace
fprobe: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/i2fprobe tools/inkling2_fused_probe.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -linkling2_npu $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

# tools/inkling2_lm_head_probe.cpp: the NPU lm_head vs the host's on real weights
lmprobe: all
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/i2lmprobe tools/inkling2_lm_head_probe.cpp \
		-L$(LIB_DIR) -L$(FINAL_LIB_DIR) -linkling2_npu $(LDFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,'$$ORIGIN/../../lib'

# tools/inkling2_vision_run.cpp: the image tower against the HF goldens (host path)
vprobe: directories
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/inkling2_vision_run tools/inkling2_vision_run.cpp inkling2_vision.cpp inkling2_vision_npu.cpp $(LDFLAGS)

# tools/inkling2_mm_probe.cpp: the prefill matmul layout on real weights vs an fp32 reference
probe: directories
	$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/inkling2_mm_probe tools/inkling2_mm_probe.cpp inkling2_host.cpp $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
.PHONY: all clean directories harness probe fprobe lmprobe vprobe

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe
HRX_CMAKE_ARGS :=
ifeq ($(FLM_USE_HRX),1)
HRX_CMAKE_ARGS := -DFLM_USE_HRX=ON
endif

TARGET = $(BUILD_DIR)/inkling2_npu.dll

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(LIB_DIR)

$(TARGET): directories
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../detail/inkling2_npu $(HRX_CMAKE_ARGS)"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target inkling2_npu --parallel 4"

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean directories

endif
