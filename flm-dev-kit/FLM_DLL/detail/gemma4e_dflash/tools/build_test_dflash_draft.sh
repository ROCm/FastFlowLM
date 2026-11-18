#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

# Build the host drafter test rig:  tools/build_test_dflash_draft.sh  ->  ../../../build/gemma4e_dflash/bin/test_dflash_draft
set -e
cd "$(dirname "$0")/.."
mkdir -p ../../build/gemma4e_dflash/bin
g++-13 -std=c++20 -O3 -fopenmp -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -march=native -ffast-math -Wall \
    -I../../include ${EXTRA_FLAGS} dflash_draft.cpp tools/test_dflash_draft.cpp -o ../../build/gemma4e_dflash/bin/test_dflash_draft
echo built ../../build/gemma4e_dflash/bin/test_dflash_draft
