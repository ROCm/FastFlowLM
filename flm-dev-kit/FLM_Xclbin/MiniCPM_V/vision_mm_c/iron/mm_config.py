# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tile geometry of the MiniCPM-V vision-tower matmul (design C).

Kept in step with kernel/typedef.hpp by hand - the kernel selects the same
constants off -DMODEL_TYPE, and a mismatch between the two shows up as garbage
rather than as a build error, so check() below asserts the invariants both sides
rely on.

The array is 4 rows x 8 cols of compute tiles. COLUMNS split M, ROWS split N, so
one array pass covers (total_cols * m) x (total_rows * n):

  C  m=64  k=384 n=96   -> 512 x 384 per array pass
     every vision-tower matmul whose N is a multiple of 384: attn.qkv (3456),
     attn.proj (1152), mlp.fc1 (4608), mlp.fc2 (1152), vit_merger.fc1 (17280),
     merger.fc1 (4608), patch_embed (1152). N = 1024 (merger.fc2) stays on the
     old design B.

The compute tile has 64 KB of local memory: fp32 accumulator 24 KB, w tiles
2 x 12 KB, x chunks 2 x 2 KB, and the converted bfp16 tiles (1.1 KB x, 6.9 KB w).
"""

import numpy as np
from ml_dtypes import bfloat16

TOTAL_ROWS = 4
TOTAL_COLS = 8

CONFIGS = {
    "VISION_MM_C": dict(m=64, k=384, n=96),
}

# Slice sizes. n_slice tracks n: the core consumes a
# whole n-column strip at once, so w_ty is (n, k_slice).
M_SLICE = 16
K_SLICE = 64
Y_SLICE = 64

CORE_MEM_BYTES = 0x10000   # npu2 compute tile local memory
MEMTILE_BYTES = 0x80000    # npu2 memtile


class mm_config:
    def __init__(self, name):
        if name not in CONFIGS:
            raise ValueError(
                f"unknown design {name!r}; expected one of {sorted(CONFIGS)}")
        self.name = name
        c = CONFIGS[name]
        self.m = c["m"]
        self.k = c["k"]
        self.n = c["n"]
        self.m_slice = M_SLICE
        self.k_slice = K_SLICE
        self.n_slice = c["n"]
        self.y_slice = Y_SLICE
        self.total_rows = TOTAL_ROWS
        self.total_cols = TOTAL_COLS

        self.x_dtype = np.dtype[bfloat16]
        self.w_dtype = np.dtype[bfloat16]
        self.y_dtype = np.dtype[bfloat16]
        self.y_acc_dtype = np.dtype[np.float32]

        self.check()

    # Array-level tile covered by one pass.
    @property
    def array_m(self):
        return self.total_cols * self.m

    @property
    def array_n(self):
        return self.total_rows * self.n

    def core_bytes(self, stack=3 * 1024):
        """Local memory a single compute tile needs, matching mm_cttile()."""
        y_acc = self.m * self.n * 4
        w = 2 * self.n_slice * self.k_slice * 2
        x = 2 * self.m_slice * self.k_slice * 2
        y = 2 * self.y_slice * 2
        b = self.n * 4
        xc = (self.k_slice // 8) * 2 * 72
        wc = (self.n_slice // 8) * (self.k_slice // 8) * 72
        return y_acc + w + x + y + b + xc + wc + stack

    def memtile_bytes(self):
        """Local memory a single memtile needs, matching mm_memtile()."""
        x = 2 * self.m * self.k * 2
        w = 2 * self.n * self.k * 2
        y = 2 * self.m * (self.total_rows * self.n) * 2
        return x + w + y

    def check(self):
        assert self.m % 8 == 0, "m must be a multiple of the mmul row block"
        assert self.n % 8 == 0, "n must be a multiple of the mmul col block"
        assert self.k % self.k_slice == 0, "k must divide into k_slice chunks"
        assert self.m % self.m_slice == 0, "m must divide into m_slice chunks"
        assert self.k_slice % 8 == 0
        # mm_aie() steps colB by 2 and rowA by 2.
        assert (self.n_slice // 8) % 2 == 0, "n_slice/8 must be even"
        assert (self.m_slice // 8) % 2 == 0, "m_slice/8 must be even"
        assert (self.m * self.n) % self.y_slice == 0
        assert self.core_bytes() <= CORE_MEM_BYTES, (
            f"{self.name}: core needs {self.core_bytes()} B > "
            f"{CORE_MEM_BYTES} B of tile memory")
        assert self.memtile_bytes() <= MEMTILE_BYTES, (
            f"{self.name}: memtile needs {self.memtile_bytes()} B > "
            f"{MEMTILE_BYTES} B")
