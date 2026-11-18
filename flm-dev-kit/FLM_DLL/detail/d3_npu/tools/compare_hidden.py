#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Compare two d3_engine_run out.bin files (bf16 [n, D] rows): number of bit-identical rows, cosine min / median / mean.
  compare_hidden.py A.bin B.bin D [MAP]
With MAP (one line per row of A: the row of B it corresponds to, make_state_once_prompts.py's <out>_row.map) B may hold more rows
than A (A: the per-row run of the first K questions of every group, B: the state-once run of all of them); only the first
len(A) map entries that B holds are compared.
"""
import sys
import numpy as np

a_path, b_path, D = sys.argv[1], sys.argv[2], int(sys.argv[3])
def load(p):
    u = np.fromfile(p, dtype=np.uint16).reshape(-1, D)
    return u, (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
ua, a = load(a_path); ub, b = load(b_path)
if len(sys.argv) > 4:
    m = [int(x) for x in open(sys.argv[4]).read().split()]
    m = [i for i in m[:len(a)] if i < len(b)]
    a, ua = a[:len(m)], ua[:len(m)]
    b, ub = b[m], ub[m]
if a.shape != b.shape:
    sys.exit(f"shape mismatch {a.shape} vs {b.shape}")
same = int((ua == ub).all(axis=1).sum())
cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
print(f"rows {len(a)}  bit-identical {same}  cosine min {cos.min():.6f} median {np.median(cos):.6f} mean {cos.mean():.6f}  finite {bool(np.isfinite(a).all() and np.isfinite(b).all())}")
