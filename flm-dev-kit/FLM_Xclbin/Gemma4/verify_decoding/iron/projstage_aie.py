# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""projstage: projonly with the y de-permute stage (iron/proj_tile.py y_depermute_mt) between the engine and the shim."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projonly_aie import flow_arch

if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B", stage = True)
