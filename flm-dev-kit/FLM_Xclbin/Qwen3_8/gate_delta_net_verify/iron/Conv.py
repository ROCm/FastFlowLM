# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.dialects.ext.arith import constant

from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP

def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

class Conv(AIE_CT_IP):
    """Causal depthwise conv1d + SiLU over CONV_WIDTH channels, L_TOKENS at a time.

    One output: MM2S ch0 (Y_out) emits each token repacked as 48 per-v-head
    (k | v | q) triplets, 384 bf16 each, for the delta-rule cores.

    Note what is *not* here: alpha and beta.  The L=1 design had the a/b tile
    hand them to this tile so the kernel could interleave them into the packet,
    which forced a 160-word padded packet and a CT[0][5] / CT[1][5] adjacency.
    At L=8 the triplets have to be reordered head-major anyway (the delta cores
    keep S resident across all 8 tokens), so they land in a DDR arena first --
    and once DDR is in the path the a/b tile can simply scatter its own dump
    into the two-word holes the arena leaves.  Hence 384 = 3 x 128, no padding.

    x and w arrive on *separate* S2MM channels (ch0 and ch1) rather than
    interleaved in one buffer.  See the docstring of linear_aux_memtile for why:
    an L=8 sliding window over an interleaved x/w layout is not an affine
    address pattern.  Each is its own (CONV_SIZE, DH) ping/pong, 1 KB a piece.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, alpha_beta_buffer, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "conv.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing Conv at ({self.ct_row}, {self.ct_col})")

        self.model_config = model_config
        self.DQ = model_config.linear_DQ
        self.DK = model_config.linear_DK
        self.DV = model_config.linear_DV
        self.DH = model_config.linear_v_head_dim
        self.conv_kernel_size = model_config.linear_conv_kernel_size
        self.alpha_beta_buffer = alpha_beta_buffer
        # One (k | v | q) triplet for one v head.  The alpha / beta words that
        # sit between them in the DDR arena are not this tile's business, so the
        # on-core packet is dense.
        self.triplet_size = 3 * self.DH
        self.y_ty = np.ndarray[(self.triplet_size,), self.dtype]

        self.alpha_beta_size = model_config.linear_num_heads_padded * 2
        self.alpha_beta_w_ty = np.ndarray[(self.alpha_beta_size * 2,), self.dtype]

        # One 128-channel slice of the window: CONV_SIZE taps of DH channels.
        self.window_ty = np.ndarray[(self.conv_kernel_size, self.DH), self.dtype]
        self.conv_result_ty = np.ndarray[(self.DQ + self.DK + self.DV, ), self.dtype]
        self.y_buffer_0 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_buffer_0_{self.ct_row}_{self.ct_col}")
        self.y_buffer_1 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_buffer_1_{self.ct_row}_{self.ct_col}")

        self.conv_buffer = aie.buffer(tile = self.ct, datatype = self.conv_result_ty, name = f"conv_y_buffer_{self.ct_row}_{self.ct_col}")

        self.x_buffer_0 = aie.buffer(tile = self.ct, datatype = self.window_ty, name = f"conv_x_buffer_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = aie.buffer(tile = self.ct, datatype = self.window_ty, name = f"conv_x_buffer_1_{self.ct_row}_{self.ct_col}")
        self.w_buffer_0 = aie.buffer(tile = self.ct, datatype = self.window_ty, name = f"conv_w_buffer_0_{self.ct_row}_{self.ct_col}")
        self.w_buffer_1 = aie.buffer(tile = self.ct, datatype = self.window_ty, name = f"conv_w_buffer_1_{self.ct_row}_{self.ct_col}")

        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.y_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.conv = ctx.func("conv", [self.window_ty, self.window_ty, self.window_ty, self.window_ty, self.conv_result_ty, self.y_ty, self.y_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.conv(self.x_buffer_0, self.x_buffer_1, self.w_buffer_0, self.w_buffer_1, self.conv_buffer, self.y_buffer_0, self.y_buffer_1)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            # ---- S2MM ch0: the sliding x window ------------------------------
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])

            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            # ---- S2MM ch1: the fixed w taps ----------------------------------
            with block[3]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[8])

            with block[4]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[4])

            # ---- MM2S ch0: the packed (k | v | q) triplets, one v head each --
            # Dense 384 words out; the DDR-side BD does the head-major scatter.
            with block[8]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[9], chain = block[11])

            with block[9]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = self.triplet_size)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[10])

            with block[10]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = self.triplet_size)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[9])

            with block[11]:
                aie.EndOp()

    @property
    def X_in(self):
        return self.S2MM[0]

    @property
    def W_in(self):
        return self.S2MM[1]

    @property
    def Y_out(self):
        return self.MM2S[0]

