# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""attn8attn: RoPE + q park + global/SWA attention columns, stand-alone (plan S6.5-S6.10, debug stage).  See iron/attn8_aie.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn8_aie import flow_arch

if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B", stage = "attn")
