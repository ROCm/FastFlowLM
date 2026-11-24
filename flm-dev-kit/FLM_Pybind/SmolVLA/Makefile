# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# SMOLVLA UI Host Makefile
# =============================================================================
#
# Usage:
#   make          - Build run.exe
#   make python   - Build the smolvla_npu python extension module
#   make clean    - Remove all built files
#
# =============================================================================

HOST_C_TARGET := run.exe
HOST_COMMON_DIR := ../../FLM_Xclbin/host_common

VERBOSE ?= 0
PROFILE ?= 0
SKIP ?= 0

# DIST=1 builds the python module for redistribution: the binary goes to people
# who do not get the source, so it should not carry more of it than it has to.
#   -fvisibility=hidden   hide every symbol but the module init (pybind11 marks
#                         its own exports); drops ~500 internal names from the
#                         dynamic symbol table.
#   -DNDEBUG              compile out assert(), which otherwise embeds the
#                         source path and the assertion text of every one.
#   -s                    strip at link time.
# It is off by default because NDEBUG also removes the shape checks that make a
# local mistake fail loudly instead of silently corrupting output.
DIST ?= 0

# Parallel compile by default, so a plain `make` is not serial. Override with
# `make JOBS=8`; an explicit -j on the command line still wins over this.
JOBS ?= 4
MAKEFLAGS += -j$(JOBS)

CXX := g++-13
CXX_FLAGS := -std=c++20 -fPIC -Wall
# Without an explicit -O the build defaults to -O0, which costs ~8x on the CPU
# kernels (the bf16 micro-kernel in cpu_functions.hpp needs its accumulator tile
# kept in registers, which only happens once the optimiser runs). -O2 measured
# slightly ahead of -O3 on the connector matmul.
CXX_FLAGS += -O2
CXX_FLAGS += -DUSE_AVX2=1
CXX_FLAGS += -mavx2 -march=native -mfma -ffast-math
CXX_FLAGS += -fopenmp
CXX_FLAGS += -Iinclude
CXX_FLAGS += -Isrc
CXX_FLAGS += -Isrc/SmolVLA
CXX_FLAGS += -Isrc/SmolVLA/vlm
CXX_FLAGS += -I$(HOST_COMMON_DIR)
CXX_FLAGS += -I/opt/xilinx/xrt/include
CXX_FLAGS += -MMD -MP
CXX_FLAGS += -DVERBOSE=$(VERBOSE)
ifeq ($(PROFILE), 1)
CXX_FLAGS += -DFLM_PROFILE
endif
ifeq ($(DIST), 1)
CXX_FLAGS += -fvisibility=hidden -fvisibility-inlines-hidden -DNDEBUG
PY_LDFLAGS += -s
endif

LDFLAGS += -fopenmp
LDFLAGS += -L/opt/xilinx/xrt/lib
LDFLAGS += -L$(HOST_COMMON_DIR)
LDFLAGS += -lxrt_coreutil $(HOST_COMMON_DIR)/../../../src/lib/xrt/libaiebu.a
LDFLAGS += -lboost_program_options

# Find all source files in src/ directory (including subdirectories). The two
# targets share every object except their entry point: run.exe takes main.cpp,
# the python module takes src/python/bindings.cpp. Everything is already built
# -fPIC, so the same build/ tree feeds both and neither forces a rebuild of the
# other.
HOST_SOURCES := $(shell find src -name '*.cpp')
EXE_SOURCES  := $(filter-out src/python/%,$(HOST_SOURCES))
LIB_SOURCES  := $(filter-out src/main.cpp,$(HOST_SOURCES))
HOST_OBJECTS := $(EXE_SOURCES:%.cpp=build/%.o)
PY_OBJECTS   := $(LIB_SOURCES:%.cpp=build/%.o)
HOST_DEPENDS := $(sort $(HOST_OBJECTS:.o=.d) $(PY_OBJECTS:.o=.d))

# Default target
$(HOST_C_TARGET): build/$(HOST_C_TARGET)
	cp build/$(HOST_C_TARGET) ./

# Link the executable
build/$(HOST_C_TARGET): $(HOST_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LDFLAGS)

# Compile source files and generate dependencies
build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# --------------------------------------------------------------------------
# Python extension module
# --------------------------------------------------------------------------
# Point PYTHON at the interpreter the module will be imported from: the
# extension is built against that interpreter's headers and named with its ABI
# tag, so it will not load under a different one.
#
# NOTE: objects are keyed by source path only, so switching PROFILE between a
# `make` and a `make python` does not trigger a rebuild -- run `make clean`
# first when changing it. With PROFILE=1 (the default) every forward() prints
# the per-stage timing table; build `make python PROFILE=0` for a quiet module.
PYTHON ?= ./.venv/bin/python

PY_INC     := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYBIND_INC := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_TARGET  := smolvla_npu$(EXT_SUFFIX)

# bindings.cpp is the only file that needs the python/pybind11 headers.
build/src/python/%.o: CXX_FLAGS += -I$(PY_INC) -I$(PYBIND_INC) -fvisibility=hidden

# No -lboost_program_options: only main.cpp uses it. No -lpython either --
# extension modules resolve interpreter symbols at load time.
python: $(PY_TARGET)

$(PY_TARGET): build/$(PY_TARGET)
	cp build/$(PY_TARGET) ./

build/$(PY_TARGET): $(PY_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -shared $^ -o $@ $(PY_LDFLAGS) \
		-fopenmp -L/opt/xilinx/xrt/lib -L$(HOST_COMMON_DIR) -lxrt_coreutil $(HOST_COMMON_DIR)/../../../src/lib/xrt/libaiebu.a

# Run the executable
run: $(HOST_C_TARGET)
	./$(HOST_C_TARGET)

# Clean built files
clean:
	rm -rf build $(HOST_C_TARGET) smolvla_npu*.so

.PHONY: run clean python

# Include dependency files
-include $(HOST_DEPENDS)