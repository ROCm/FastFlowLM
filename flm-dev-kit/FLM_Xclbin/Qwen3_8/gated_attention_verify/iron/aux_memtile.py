# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import connect_port
from utils import AIE_MT_IP

from constants import *
from iron.utils import comment_print


class aux_memtile(AIE_MT_IP):
    """Token-major staging for the attention output gate z, L_TOKENS at a time.

    `attn * sigmoid(z)` is elementwise, but its two operands reach the gate tile
    from opposite ends of the layer.  Only z comes through here:

      * z, from the projection engine, is **row-block permuted**:
        `raw[(rb * L + t) * m + r]`.  It is produced early, right after the qkv
        mvm, so it has to be parked for the whole attention latency -- q/k norm,
        RoPE, the KV cache write and the softmax column -- and at L * DQ = 96 KB
        that is far past a 64 KB compute tile.
      * the attention output arrives straight from the attention column's own
        memtile, already token-major and already late, so it needs no staging
        and does not enter this tile at all.  (The delta-net copy of this class
        stages both, because its o comes off 48 per-head recurrent states and no
        whole token exists until the scan ends.)

    Token-major rather than row-block-major staging because the de-permute is
    then **3-dimensional**:

        z : [(DQ // m, m), (L, DQ), (m, 1)]    rb,t,r -> rb*32 + t*6144 + r

    and 4-dim memtile BDs are unverified in this version of IRON.  It also keeps
    `gate_tile` at its L=1 shape (one DQ row per invocation, looped L_TOKENS
    times by the self-chained BD on its side).

    One producer and one consumer, so one lock pair:

        producer (S2MM)          acq free(L)      rel ready(L)    one BD
        consumer (-> tile)       acq ready(1)     rel free(1)     L chained BDs

    Every count is <= L_TOKENS, well inside the memtile's 6-bit lock counter.

    Channel map:
        S2MM ch1  z, row-block permuted, from the projection engine
        MM2S ch1  z token-major, one row per BD, to gate_tile

    ch1 on both sides, not ch0: the channel indices are inherited from the
    delta-net two-operand version so that the two projects' port numbering stays
    comparable, and nothing here is contended.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.model_config = model_config
        self.DQ  = model_config.DQ                      # 6144, one token of z
        self.L   = L_TOKENS
        self.m   = model_config.q4nx_row_block_size     # 32
        self.stage_size = self.L * self.DQ

        self.stage_ty = np.ndarray[(self.stage_size,), self.dtype]
        comment_print(f"aux_memtile z stage buffer: {self.stage_size} bf16 "
                      f"({self.stage_size * 2} B)")

        self.z_buffer = aie.buffer(tile = self.mt, datatype = self.stage_ty, name = f"aux_z_buffer_{self.mt_row}_{self.mt_col}")

        self.z_free_lock  = aie.lock(self.mt, lock_id = 2, init = self.L)
        self.z_ready_lock = aie.lock(self.mt, lock_id = 3, init = 0)

        self.__call__()

    def __call__(self):
        # [rb][t][r] -> [t][rb * m + r].  Stream order is rb, t, r.
        z_dims = [(self.DQ // self.m, self.m), (self.L, self.DQ), (self.m, 1)]

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # ---- S2MM ch1: z, row-block permuted, de-permuted on the way in --
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.z_free_lock, self.L)
                aie.dma_bd(self.z_buffer, offset = 0, len = self.stage_size, dimensions = z_dims)
                lock_release(self.z_ready_lock, self.L)
                aie.next_bd(block[1])

            # ---- MM2S ch1: one z row per BD, to the gate tile ----------------
            # One self-chained BD cannot walk the offset, so this is L chained
            # BDs that differ only in where they start.
            with block[2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[30], chain = block[40])

            for t in range(self.L):
                with block[30 + t]:
                    lock_acquire(self.z_ready_lock, 1)
                    aie.dma_bd(self.z_buffer, offset = t * self.DQ, len = self.DQ)
                    lock_release(self.z_free_lock, 1)
                    aie.next_bd(block[30 + (t + 1) % self.L])

            with block[40]:
                aie.EndOp()

    @property
    def Z_in(self) -> dma_port:
        return self.S2MM[1]

    @property
    def Gate_out(self) -> dma_port:
        return self.MM2S[1]
