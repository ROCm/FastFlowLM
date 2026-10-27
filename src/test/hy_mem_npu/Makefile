# =============================================================================
# Hy-Memory-30B-A3B NPU test Makefile
# =============================================================================
#
# Usage:
#   make          - Build test_hy_mem (messages -> JSON) and check_hy_mem (numeric check)
#   make test     - Run test_hy_mem on MESSAGES (default: the model card demo)
#   make chain    - Run hy_memory's EXTRACT -> RECONCILE chain
#   make check    - Run check_hy_mem against REFERENCE (default: the Q4_K one)
#   make test-long- The same task at PROMPT_TOKENS prompt lengths (default 128,512,2048)
#   make clean    - Remove all built files
#
# Run `source activate.sh` first: it copies the engine, its header and the
# xclbins from FLM_DLL / FLM_Xclbin and sets FLM_MODEL_PATH.
# =============================================================================
-include ../common.mk

MESSAGES ?= ../../../test/hy_mem_npu/messages/model_card_demo.json
LENGTH   ?= 2000
# make test-long: prompt lengths for the same extraction task. The ~100-token
# messages file is too short to measure prefill with; every stage runs on the NPU
# at every length, and the curve shows how much of a short run is padding.
PROMPT_TOKENS ?= 128,512,2048
# runs per length; prefill speed moves ~20% run to run, so raise this to compare
REPEAT ?= 1
TAG      ?= hy-memory:30b-a3b
# models_simple/ref_hy_mem_e2e_q4k.py (same Q4_K weights as the NPU); the bf16
# one, hy_mem_e2e_reference.safetensors, says how far Q4_K is from the original
REFERENCE ?= /scratch/$(USER)/Projects/FastFlowLM_IRON/models_simple/hy_mem_e2e_reference_q4k.safetensors
# weights at $(FLM_MODEL_PATH)/models/Hy-Memory-30B-A3B-NPU2 (as in activate.sh)
FLM_MODEL_PATH ?= /scratch/$(USER)
# test / chain: write each request's messages and reply here (empty = off)
SAVE ?=
RUN_ENV := FLM_MODEL_PATH=$(FLM_MODEL_PATH)

SOURCES += test.cpp
SOURCES += check_engine.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
# automodel.cpp's prefix-slot resume/save path
SOURCES += ../../common/prefix_slot_cache.cpp
SOURCES += $(CHAT_SOURCES)
SOURCES += ../../common/AutoModel/model_backend.cpp
SOURCES += ../../common/AutoModel/modeling_hy_memory.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

HEADERS += ../../include/models/hy_mem/flm/aie2p/hy_mem_npu.hpp
HEADERS += ../../include/AutoModel/modeling_hy_memory.hpp

ifeq ($(WSL), 0)

LDFLAGS += -lhy_mem_npu

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
ALL_OBJS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
# test.cpp and check_engine.cpp each carry a main(), so they link into separate
# executables over the same shared objects
SHARED_OBJS := $(filter-out $(BUILD_DIR)/test.o $(BUILD_DIR)/check_engine.o,$(ALL_OBJS))
DEPS := $(ALL_OBJS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_hy_mem $(BUILD_DIR)/check_hy_mem

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_hy_mem: $(BUILD_DIR)/test.o $(SHARED_OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

$(BUILD_DIR)/check_hy_mem: $(BUILD_DIR)/check_engine.o $(SHARED_OBJS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_hy_mem
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && $(RUN_ENV) ./test_hy_mem --model $(TAG) --messages $(MESSAGES) --Length $(LENGTH) --save "$(SAVE)"

test-long: $(BUILD_DIR)/test_hy_mem
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && $(RUN_ENV) ./test_hy_mem --model $(TAG) --prompt-tokens $(PROMPT_TOKENS) --repeat $(REPEAT) --Length $(LENGTH) --save "$(SAVE)"

chain: $(BUILD_DIR)/test_hy_mem
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && $(RUN_ENV) ./test_hy_mem --model $(TAG) --chain --save "$(SAVE)"

check: $(BUILD_DIR)/check_hy_mem
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && $(RUN_ENV) ./check_hy_mem --model $(TAG) --reference $(REFERENCE)

-include $(DEPS)
.PHONY: all clean test test-long chain check

else

# WSL build environment
# Use CMake to invoke the Visual Studio
PWSH := powershell.exe

all: directories $(BUILD_DIR)/test_hy_mem.exe

directories:
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_hy_mem.exe: $(SOURCES)
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake ../../../test/hy_mem_npu"
	cd $(BUILD_DIR) && $(PWSH) -Command "cmake --build . --config Release --target test_hy_mem"

clean:
	rm -rf $(BUILD_DIR)

test: directories $(BUILD_DIR)/test_hy_mem.exe
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ${PWSH} -Command "\$$env:PATH = '..\..\..\lib;' + \$$env:PATH; .\test_hy_mem.exe -m $(TAG)"

.PHONY: all clean test directories

endif
