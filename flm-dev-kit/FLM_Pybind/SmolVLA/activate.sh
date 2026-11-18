# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

# Source this file to activate the venv: source activate.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

export HF_HOME="/scratch/$USER/.hf_cache"
