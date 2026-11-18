# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Tile geometry for the two Qwen3.8-27B vision-tower MM designs.

Kept in step with kernel/typedef.hpp by hand - the kernel selects the same
constants off -DMODEL_TYPE, and a mismatch between the two shows up as garbage
rather than as a build error, so check_config() below asserts the invariants
both sides rely on.

The array is 4 rows x 8 cols of compute tiles. Rows split M, cols split N, so
one array pass covers (total_rows * m) x (total_cols * n).

  A  m=64  k=384 n=48   -> 256 x 384 per array pass
     attn.qkv (N=3456), attn.proj (N=1152), mlp.fc1 (N=4304->4608),
     mlp.fc2 (K=4304->4608), merger.fc1 (4608x4608)

  B  m=32  k=384 n=128  -> 128 x 1024 per array pass
     merger.fc2 (4608x5120), 5120 = 5 * 1024

m is 32 in design B, not 64, because the compute tile has only 64 KB of local
memory: at m=64,n=128 the fp32 accumulator is 32 KB and the double-buffered w
tile another 32 KB, which overflows before x and the stack are counted. Keeping
k_slice at 64 in both designs leaves the mmul inner-loop trip count unchanged,
so kernel/mm.cc's chess_loop_range(7,7) hint stays honest.
"""

import numpy as np
from ml_dtypes import bfloat16

TOTAL_ROWS = 4
TOTAL_COLS = 8

CONFIGS = {
    "VISION_MM_A": dict(m=64, k=384, n=48),
    "VISION_MM_B": dict(m=32, k=384, n=128),
}

# Slice sizes are shared by both designs. n_slice tracks n: the core consumes a
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
        return self.total_rows * self.m

    @property
    def array_n(self):
        return self.total_cols * self.n

    def core_bytes(self, stack=3 * 1024):
        """Local memory a single compute tile needs, matching mm_cttile()."""
        y_acc = self.m * self.n * 4
        w = 2 * self.n_slice * self.k_slice * 2
        x = 2 * self.m_slice * self.k_slice * 2
        y = 2 * self.y_slice * 2
        b = self.n * 4
        return y_acc + w + x + y + b + stack

    def memtile_bytes(self):
        """Local memory a single memtile needs, matching mm_memtile()."""
        x = 2 * self.m * self.k * 2
        w = 2 * self.n * self.k * 2
        y = 2 * (self.total_rows * self.m) * self.n * 2
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
