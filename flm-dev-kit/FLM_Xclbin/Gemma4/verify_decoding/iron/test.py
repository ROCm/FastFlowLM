# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import sys
import os

# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.models import qwen3_0_6b_config
from iron.model_config import get_model_config

model_config = get_model_config("QWEN3_0_6B")
print(model_config.D)