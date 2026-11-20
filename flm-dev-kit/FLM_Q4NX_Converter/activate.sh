#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# =============================================================================
# File        : activate.sh
# Created     : 2025-03-27
# Description : This is a script to activate the Quark project.
# =============================================================================

# deactivate conda environment
# conda deactivate

# activate the Quark project
source venv/bin/activate

# set the HF_HOME to the cache directory
mkdir -p ./hf_cache
export HF_HOME=./hf_cache

export PATH=/usr/local/cuda/bin:$PATH
