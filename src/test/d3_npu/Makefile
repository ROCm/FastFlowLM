# =============================================================================
# d3 NPU harness: System One requests through AutoDecisionModel, no HTTP
#   make && make test   (needs the d3:lite package in the FLM model dir)
# =============================================================================
-include ../common.mk

SOURCES += test.cpp
SOURCES += ../../common/AutoDecisionModel/auto_decision_model.cpp
SOURCES += ../../common/decision/d3_contract.cpp
SOURCES += ../../common/decision/py_json.cpp
SOURCES += ../../common/video/video_sampling.cpp
SOURCES += ../../common/video/video_preprocess.cpp
SOURCES += ../../common/video/video_decode.cpp
SOURCES += ../../common/video/video_input.cpp
SOURCES += ../../common/video/ffmpeg_dl.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp

ifeq ($(WSL), 0)

LDFLAGS += -ld3_npu -ldl
CXX_FLAGS += -DFLM_HAS_D3_NPU=1

CPP_SOURCES := $(filter %.cpp,$(SOURCES))
OBJECTS := $(addprefix $(BUILD_DIR)/,$(notdir $(CPP_SOURCES:.cpp=.o)))
DEPS := $(OBJECTS:.o=.d)
VPATH := $(sort $(dir $(CPP_SOURCES)))

all: $(BUILD_DIR)/test_d3_npu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/test_d3_npu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LDFLAGS) $(DEPENDENCY_LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/test_d3_npu
	cp ../../model_list.json $(BUILD_DIR)/model_list.json
	cd $(BUILD_DIR) && ./test_d3_npu -m d3:lite -i $(abspath ../d3_contract/fixtures/goldens.jsonl) -o replies.jsonl

-include $(DEPS)
.PHONY: all clean test

endif
