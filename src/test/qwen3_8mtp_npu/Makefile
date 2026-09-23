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
SOURCES += ../../common/AutoModel/modeling_qwen3_8mtp_image.cpp
SOURCES += ../../common/image_process_utils/imageproc.cpp
SOURCES += ../../common/image_process_utils/imageprocAVX512.cpp
SOURCES += ../../common/image/image_reader.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/qwen3_8mtp/qwen3_8mtp_npu.hpp
HEADERS += ../../include/AutoModel/modeling_qwen3_8mtp.hpp


ifeq ($(WSL), 0)

LDFLAGS += -lqwen3_8mtp_npu
# image_reader.cpp decodes through ffmpeg, same as the Qwen3_5VL test.
LDFLAGS += -lavformat -lavcodec -lavutil -lswscale -lswresample

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

# Scores the HOST image preprocessing against a captured pixel_values. Separate
# from the test binary because it loads no model and touches no device -- see
# the header comment in verify_preprocess.cpp for why it is not covered by
# tools/verify_n14_vision.cpp.
VERIFY_SOURCES := $(filter-out test.cpp,$(CPP_SOURCES)) verify_preprocess.cpp
VERIFY_OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(VERIFY_SOURCES:.cpp=.o)))
DEPS += $(VERIFY_OBJECTS:.o=.d)

verify_preprocess: $(BUILD_DIR)/verify_preprocess

$(BUILD_DIR)/verify_preprocess: $(VERIFY_OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

REF_IMAGE ?= /scratch/michyu/Projects/FastFlowLM_IRON/models_simple/test_input/observation_images_image.png
REF_DUMP  ?= /scratch/michyu/models/Qwen3.8-27B-NPU2/vision_ref.safetensors

verify: $(BUILD_DIR)/verify_preprocess
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib/xrt \
	    && ./verify_preprocess $(REF_IMAGE) $(REF_DUMP)

clean:
	rm -rf $(BUILD_DIR)

# Decode is ~1-2 s/token here, so -n stays small.
#
# The draft head now runs on hardware by default. MTP_CPU=1 used to be forced
# because MTP_layer.xclbin timed out its second dispatch (npu_mtp_layer:
# dispatch state 8 at cache row 1); that is repaired, so the default is 0 and
# the variable is only kept as an A/B knob. Speculation is output-neutral --
# speculate() verifies every draft against the real stack -- so either setting
# produces the same text, and only throughput moves.
#
#   MTP_CPU=0  draft head on the NPU: MTP_layer.xclbin for the six one-token
#              steps, attention_prefill.xclbin for step 0's length-L window
#   MTP_CPU=1  draft head on the CPU (attention stack stays on the NPU)
#
# MTP_PREFILL=0 keeps the head on the NPU but forces step 0 back onto the
# per-row MTP_layer.xclbin path -- the A/B for the step-0 split alone.
# PROMPT=0 reads ../../../../prompt.txt (346 tokens) instead of the 8-token
# smoke prompt; use it when measuring, not when just checking it runs.
MTP_CPU     ?= 0
MTP_PREFILL ?= 1
PROMPT      ?= 1
NTOK        ?= 512

test: $(BUILD_DIR)/test_qwen3_8mtp_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && export LD_LIBRARY_PATH=../../../lib/xrt \
	    && export FLM_Q38_MTP_CPU=$(MTP_CPU) \
	    && export FLM_Q38_MTP_PREFILL=$(MTP_PREFILL) \
	    && ./test_qwen3_8mtp_npu --model qwen3-8mtp:27b -s $(PROMPT) -n $(NTOK)

-include $(DEPS)
.PHONY: all clean test verify verify_preprocess

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
