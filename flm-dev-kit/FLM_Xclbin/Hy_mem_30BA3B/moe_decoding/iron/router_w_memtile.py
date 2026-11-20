# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
import sys
import os
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie

from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_MT_IP
from iron.utils import comment_print


class router_w_memtile(AIE_MT_IP):
    """Holds the whole router weight (num_experts x D, bf16 = 512 KB) and feeds it
    to the router CT and the router helper at the same time.

        S2MM 0 (IT5 channel 0): experts 0 .. E/2-1 into half 0, then E/2 .. E-1 into half 1
        MM2S 0: half 0 -> router CT       MM2S 1: half 1 -> router helper

    The host sends the weight early in the call, so IT5 fills both halves while
    attention and the shared expert run; then the two CTs each stream only half,
    in parallel (the router weight used to come in over one MT -> CT stream).
    The two halves use the whole 512 KB of the mem tile.
    """
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing router_w_memtile at ({self.mt_row}, {self.mt_col})")
        self.router_weight_row_block: int = ctx.parameters["router_weight_row_block"]
        assert(model_config.num_experts % (2 * self.router_weight_row_block) == 0)

        self.half_ty = np.ndarray[(model_config.num_experts // 2 * model_config.hidden_size, ), self.dtype]
        # per half: [buffer, producer lock, consumer lock]
        self.halves = []
        for h in range(2):
            buf = aie.buffer(tile=self.mt, datatype=self.half_ty, name=f"router_w_half_{h}_{self.mt_row}_{self.mt_col}")
            prod = aie.lock(self.mt, lock_id=2 * h, init=1)
            cons = aie.lock(self.mt, lock_id=2 * h + 1, init=0)
            self.halves.append((buf, prod, cons))

        self.__call__()

    def __call__(self):
        (buf0, prod0, cons0), (buf1, prod1, cons1) = self.halves

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # in: half 0 then half 1, each waits until its previous contents went out
            with block[0]:
                aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=0, dest=block[1], chain=block[3])
            with block[1]:
                lock_acquire(prod0, value=1)
                aie.dma_bd(buf0)
                lock_release(cons0, value=1)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(prod1, value=1)
                aie.dma_bd(buf1)
                lock_release(cons1, value=1)
                aie.next_bd(block[1])

            # out: half 0 -> router CT, half 1 -> router helper
            with block[3]:
                aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=0, dest=block[4], chain=block[5])
            with block[4]:
                lock_acquire(cons0, value=1)
                aie.dma_bd(buf0)
                lock_release(prod0, value=1)
                aie.next_bd(block[4])
            with block[5]:
                aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=1, dest=block[6], chain=block[7])
            with block[6]:
                lock_acquire(cons1, value=1)
                aie.dma_bd(buf1)
                lock_release(prod1, value=1)
                aie.next_bd(block[6])
            with block[7]:
                aie.EndOp()

    @property
    def W_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def W_out(self) -> dma_port:
        return self.MM2S[0]

    @property
    def W2_out(self) -> dma_port:
        return self.MM2S[1]
