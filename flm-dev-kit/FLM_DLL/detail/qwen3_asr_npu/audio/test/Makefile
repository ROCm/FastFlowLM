# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Standalone host test of the Qwen3-ASR audio encoder (no NPU / XRT needed).  Needs g++-13 and an AVX-512 CPU.
#   make && make run
CXX      = g++-13
INC      := -I../../../../include
CXXFLAGS := -O3 -std=c++20 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16 -fopenmp -Wall -Wno-unused-function $(INC)

WEIGHTS  ?= /path/to/Confucius4-R2T2-NPU2/audio_weight.q4nx
GOLDENS  ?= /path/to/goldens
CLIPS    ?= syn_short1s syn_silence3s repo_test ls_4077-13751-0006 zh_0006 zh_0015 syn_quiet_onset syn_long_concat

audio_encoder_test: audio_encoder_test.cpp ../qwen3_asr_audio_encoder.cpp ../qwen3_asr_audio_encoder.hpp
	$(CXX) $(CXXFLAGS) -o $@ audio_encoder_test.cpp ../qwen3_asr_audio_encoder.cpp

# streaming: cached conv front end == uncached over a growing recording (AddressSanitizer build)
conv_cache_test: conv_cache_test.cpp ../qwen3_asr_audio_encoder.cpp ../qwen3_asr_audio_encoder.hpp
	$(CXX) $(CXXFLAGS) -O1 -g -fsanitize=address -o $@ conv_cache_test.cpp ../qwen3_asr_audio_encoder.cpp

conv_cache: conv_cache_test
	./conv_cache_test $(WEIGHTS) $(GOLDENS)/zh_0015/mel.npy 16
	./conv_cache_test $(WEIGHTS) $(GOLDENS)/syn_long_concat/mel.npy 200

run: audio_encoder_test
	./audio_encoder_test $(WEIGHTS) $(GOLDENS) $(CLIPS)

clean:
	rm -f audio_encoder_test conv_cache_test

.PHONY: run conv_cache clean
