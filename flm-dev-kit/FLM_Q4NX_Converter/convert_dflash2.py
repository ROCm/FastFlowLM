#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""z-lab/Qwen3.8-27B-DFlash2 -> <out>/dflash_weight.q4nx (+ config.json copied).

usage: convert_dflash2.py <hf_dir> <out_dir> [--ggml-lib libggml-base.so]
"""
import argparse
import os
import shutil

from q4nx.models.qwen38_dflash2 import Qwen38DFlash2

ap = argparse.ArgumentParser()
ap.add_argument("hf_dir")
ap.add_argument("out_dir")
ap.add_argument("--ggml-lib", default=None)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
Qwen38DFlash2(a.hf_dir, a.ggml_lib).convert(a.out_dir)
shutil.copy(os.path.join(a.hf_dir, "config.json"), os.path.join(a.out_dir, "config.json"))
