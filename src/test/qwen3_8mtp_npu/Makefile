# =============================================================================
# Qwen3.8-27B (MTP) NPU Test Makefile
# =============================================================================
#
# This Makefile builds the standalone test harness for Qwen3.8-27B.
#
# Usage:
#   make        - Build the test executable
#   make clean  - Remove all built files
#   make test   - Build and run with a short prompt
#
# =============================================================================
-include ../common.mk


SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/modeling_qwen3_8mtp.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/qwen3_8mtp/qwen3_8mtp_npu.hpp
HEADERS += ../../include/AutoModel/modeling_qwen3_8mtp.hpp


ifeq ($(WSL), 0)

LDFLAGS += -lqwen3_8mtp_npu

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_qwen3_8mtp_npu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_qwen3_8mtp_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

# Phase 1 is CPU-only: ~1-2 s/token, so -n stays small. Raise it once the
# NPU offload lands, not before.
test: $(BUILD_DIR)/test_qwen3_8mtp_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib/xrt && ./test_qwen3_8mtp_npu --model qwen3-8mtp:27b -s 1 -n 16

-include $(DEPS)
.PHONY: all clean test

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories test

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_qwen3_8mtp_npu.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/qwen3_8mtp_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_qwen3_8mtp_npu"

clean:
	rm -rf $(BUILD_DIR)


test: directories $(BUILD_DIR)/test_qwen3_8mtp_npu.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_qwen3_8mtp_npu.exe -m qwen3-8mtp:27b -s 1 -p 0 -n 16"

.PHONY: all clean test directories

endif
