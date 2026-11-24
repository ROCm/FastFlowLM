# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Standalone build of the CPU reference decoder + golden test (no NPU / XRT dependency).
#   make && ./test_decoder_goldens <model.q4nx> <goldens dir> <clip>...
include ../../../flags.mk
CXX      ?= g++-13
CXXFLAGS ?= -O3 $(FLM_MARCH) -mavx512bf16 -mavx512vnni -mavx512vbmi -fopenmp -std=c++17 -Wall -Wextra -Wno-unused-parameter
INC      := -I../../../include

test_decoder_goldens: test_decoder_goldens.cpp qwen3_asr_cpu_decoder.cpp qwen3_asr_cpu_decoder.hpp bf16_util.hpp mapped_safetensors.hpp npy_io.hpp
	$(CXX) $(CXXFLAGS) $(INC) test_decoder_goldens.cpp qwen3_asr_cpu_decoder.cpp -o $@

clean:
	rm -f test_decoder_goldens
