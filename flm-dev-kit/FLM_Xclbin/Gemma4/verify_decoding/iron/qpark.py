# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""q park of the 8-token verify design (plan S6.5, S6.9): a memtile that holds the rotated q of all L_TOKENS rows of a layer and hands it to
the attention qk tile one token at a time.

Why: the global qk tile cannot hold q for 8 tokens (8 heads x 512 x 2 B x 8 = 64 KB), and the RoPE tile emits q one HEAD at a time for
all tokens ([head][token][DH], head-major, one [L][DH] chunk per head).  The qk tile wants, per token, all 8 heads of that token
([head][DH] = 4096 elements) which it reorders itself with its own S2MM BD (iron/attn.py q_in_order).

  S2MM : the whole layer's q stream  [NQ][L][DH]  into one buffer, one BD (releases `cons` when complete)
  MM2S : L_TOKENS BDs, BD t reads token t's rows of every head: offset t*DH, dims ((NQ, L*DH), (DH, 1))  (runs of one head = 1 KB, no short runs)
         counting locks: the write BD acquires `prod` L_TOKENS units and releases `cons` L_TOKENS units; every read BD acquires one `cons` unit
         and releases one `prod` unit; the chain is cyclic: one pass per layer dispatch.
Two independent parks (global DH 512 on channel 0, SWA DH 256 on channel 1): a layer uses one kind, the other chain waits idle at its first lock.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dtypes import bfloat16
import aie.dialects.aie as aie

from constants import L_TOKENS
from utils import AIE_MT_IP, tile_with_location, mlir_context, comment_print


class q_park_memtile(AIE_MT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, nq: int, dh_global: int, dh_swa: int):
        super().__init__(tile)
        assert L_TOKENS > 1
        self.nq, self.dh_g, self.dh_s = nq, dh_global, dh_swa
        comment_print(f"Initializing q_park_memtile at MT({self.mt_row}, {self.mt_col}): global {nq}x{L_TOKENS}x{dh_global}, swa {nq}x{L_TOKENS}x{dh_swa}")
        self.g_ty = np.ndarray[(nq * L_TOKENS * dh_global,), np.dtype[bfloat16]]
        self.s_ty = np.ndarray[(nq * L_TOKENS * dh_swa,), np.dtype[bfloat16]]
        self.g_buf = aie.buffer(self.mt, self.g_ty, name = f"q_park_global_{self.mt_row}_{self.mt_col}")
        self.s_buf = aie.buffer(self.mt, self.s_ty, name = f"q_park_swa_{self.mt_row}_{self.mt_col}")
        # counting locks (every BD needs an acquire and a release): the write BD takes/gives L_TOKENS units, each token's read BD one
        self.g_prod = aie.lock(self.mt, lock_id = 0, init = L_TOKENS)
        self.g_cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.s_prod = aie.lock(self.mt, lock_id = 2, init = L_TOKENS)
        self.s_cons = aie.lock(self.mt, lock_id = 3, init = 0)
        self.__call__()

    def _chain(self, blocks, first, buf, dh, prod, cons, chain_after):
        """L_TOKENS cyclic read BDs starting at block index `first`"""
        nq = self.nq
        for t in range(L_TOKENS):
            with blocks[first + t]:
                aie.use_lock(cons, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(buf, offset = t * dh, len = nq * dh, dimensions = [(nq, L_TOKENS * dh), (dh, 1)])
                aie.use_lock(prod, aie.LockAction.Release, value = 1)
                aie.next_bd(blocks[first + (t + 1) % L_TOKENS])

    def __call__(self):
        nq = self.nq
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[2])
            with block[1]:
                aie.use_lock(self.g_prod, aie.LockAction.AcquireGreaterEqual, value = L_TOKENS)
                aie.dma_bd(self.g_buf, offset = 0, len = nq * L_TOKENS * self.dh_g)
                aie.use_lock(self.g_cons, aie.LockAction.Release, value = L_TOKENS)
                aie.next_bd(block[1])

            with block[2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[10], chain = block[20])
            self._chain(block, 10, self.g_buf, self.dh_g, self.g_prod, self.g_cons, None)

            with block[20]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[21], chain = block[22])
            with block[21]:
                aie.use_lock(self.s_prod, aie.LockAction.AcquireGreaterEqual, value = L_TOKENS)
                aie.dma_bd(self.s_buf, offset = 0, len = nq * L_TOKENS * self.dh_s)
                aie.use_lock(self.s_cons, aie.LockAction.Release, value = L_TOKENS)
                aie.next_bd(block[21])

            with block[22]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[30], chain = block[40])
            self._chain(block, 30, self.s_buf, self.dh_s, self.s_prod, self.s_cons, None)

            with block[40]:
                aie.EndOp()

    @property
    def Q_in(self):
        return self.S2MM[0]

    @property
    def Q_out(self):
        return self.MM2S[0]

    @property
    def SWA_Q_in(self):
        return self.S2MM[1]

    @property
    def SWA_Q_out(self):
        return self.MM2S[1]
