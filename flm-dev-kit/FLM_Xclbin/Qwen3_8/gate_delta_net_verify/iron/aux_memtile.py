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
    """Token-major staging for column 3, L_TOKENS at a time.

    Three activations pass through here, and all three arrive permuted and leave
    token-major. Two of them are the gated norm's operands (stage 7b); the third
    is ssm_out_proj's result on its way to the residual (stage 7c).

    `norm(o) * silu(z)` is elementwise, but its two operands reach this tile in
    different permutations and at very different times:

      * o, from the delta column's MT[4] MM2S ch5, is **head-major**: 48 back to
        back BDs, each one head's `[t][dv]`, i.e. `[h][t][128]`.  It is the last
        thing the layer produces -- every core holds one head's recurrent state
        for all L_TOKENS steps, so no whole token exists until the scan ends.
      * z, from the projection engine, is **row-block permuted**:
        `raw[(rb * L + t) * m + r]`.  It is produced early, right after the qkv
        mvm, so it has to be parked for the whole delta-rule latency --
        L * Linear_DV = 96 KB, far past a 64 KB compute tile.

    So both land here, both get de-permuted to token-major in their inbound BD,
    and both leave one token at a time.  That keeps `DeltaNet_Norm` and
    `gate_tile` at their L=1 shapes (one Linear_DV row per invocation, looped
    L_TOKENS times by the self-chained BDs on their side) and it means the
    gated result reaches DDR token-major, so the host needs no un-permute at
    all.

    The third, `op_buffer`, is ssm_out_proj's y -- the same projection engine,
    the same row-block permute as z, only D wide instead of Linear_DV:

        op : [(D // m, m), (L, D), (m, 1)]     rb,t,r -> rb*32 + t*5120 + r

    The un-permute is a memtile's job and this tile sits directly under the rms
    tile, so o never reaches DDR: it lands here, is de-permuted by the same
    3-dimensional BD shape stage 5 proved, and goes one hop north as L_TOKENS
    whole D-wide rows.  down's y takes the identical route afterwards.

    It has to be the whole L x D in one inbound BD: token 0's row is not
    complete until the last row block has arrived, so nothing can leave before
    everything has landed.

    Token-major rather than head-major staging because both de-permutes are
    then **3-dimensional**:

        o : [(NV, DVH), (L, DV), (DVH, 1)]     h,t,d -> h*128 + t*6144 + d
        z : [(DV // m, m), (L, DV), (m, 1)]    rb,t,r -> rb*32 + t*6144 + r

    The head-major alternative -- feeding both tiles one head x L tokens --
    needs a 4-dimensional BD to get z head-major, and 4-dim memtile BDs are
    unverified in this version of IRON.  The z pattern above is the same shape
    as linear_aux_memtile's QKV_in BD, which stage 5 already proved.

    Each buffer has one producer and one consumer, so it needs one lock pair:

        producer (S2MM)          acq free(L)      rel ready(L)    one BD
        consumer (-> tile)       acq ready(1)     rel free(1)     L chained BDs

    Every count is <= L_TOKENS, well inside the memtile's 6-bit lock counter.

    A DDR tap on one of these buffers has to go back as a *chained* reader
    (acq done(L), rel free(L)) behind the compute tile, not as a second racer on
    `ready`: the two would share a lock, and the tile-side BD is self-chained and
    greedy enough to drain the tap's token too.

    Channel map:
        S2MM ch0  o, head-major, from the delta column
        S2MM ch1  z, row-block permuted, from the projection engine
        S2MM ch2  ssm_out_proj's y, row-block permuted, from the same engine
        MM2S ch0  o token-major, one row per BD, to DeltaNet_Norm
        MM2S ch1  z token-major, one row per BD, to gate_tile
        MM2S ch2  ssm_out_proj token-major, one row per BD, to the rms tile

    Even and odd channels draw BD ids from different halves of the memtile's 48,
    so the split matters: the even channels here want 1 + 1 + 8 + 8 = 18 of the
    24 below 24, the odd ones 1 + 8 = 9 of the rest.

    Buffers are 96 + 96 + 80 = 272 KiB of the memtile's 512.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.model_config = model_config
        self.DV  = model_config.linear_DV               # 6144, one token of o / z
        self.DVH = model_config.linear_v_head_dim       # 128
        self.NV  = model_config.linear_num_heads        # 48
        self.D   = model_config.D                       # 5120, one token of ssm_out_proj
        self.L   = L_TOKENS
        self.m   = model_config.q4nx_row_block_size     # 32
        self.stage_size = self.L * self.DV
        self.op_size    = self.L * self.D

        assert self.DV % self.m == 0 and self.D % self.m == 0, \
            "both de-permutes walk whole row blocks"

        self.stage_ty = np.ndarray[(self.stage_size,), self.dtype]
        self.op_ty    = np.ndarray[(self.op_size,), self.dtype]
        self.int_ty    = np.ndarray[(self.L, model_config.intermediate_size), self.dtype]
        comment_print(f"aux_memtile stage buffer: {self.stage_size} bf16 "
                      f"({self.stage_size * 2} B) x 2, plus ssm_out_proj "
                      f"{self.op_size} bf16 ({self.op_size * 2} B)")

        self.o_buffer = aie.buffer(tile = self.mt, datatype = self.stage_ty, name = f"aux_o_buffer_{self.mt_row}_{self.mt_col}")
        self.z_buffer = aie.buffer(tile = self.mt, datatype = self.stage_ty, name = f"aux_z_buffer_{self.mt_row}_{self.mt_col}")
        self.op_buffer = aie.buffer(tile = self.mt, datatype = self.op_ty, name = f"aux_op_buffer_{self.mt_row}_{self.mt_col}")

        self.o_free_lock  = aie.lock(self.mt, lock_id = 0, init = self.L)
        self.o_ready_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        self.z_free_lock  = aie.lock(self.mt, lock_id = 2, init = self.L)
        self.z_ready_lock = aie.lock(self.mt, lock_id = 3, init = 0)
        self.op_free_lock  = aie.lock(self.mt, lock_id = 4, init = self.L)
        self.op_ready_lock = aie.lock(self.mt, lock_id = 5, init = 0)

        self.__call__()

    def __call__(self):
        # [h][t][dv] -> [t][h * DVH + dv].  Stream order is h, t, d.
        o_dims = [(self.NV, self.DVH), (self.L, self.DV), (self.DVH, 1)]
        # [rb][t][r] -> [t][rb * m + r].  Stream order is rb, t, r.
        z_dims = [(self.DV // self.m, self.m), (self.L, self.DV), (self.m, 1)]
        # The same permute, D wide: ssm_out_proj's y off the same engine.
        op_dims = [(self.D // self.m, self.m), (self.L, self.D), (self.m, 1)]

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # ---- S2MM ch0: o, head-major, de-permuted on the way in ---------
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.o_free_lock, self.L)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.stage_size, dimensions = o_dims)
                lock_release(self.o_ready_lock, self.L)
                aie.next_bd(block[1])

            # ---- S2MM ch1: z, row-block permuted, de-permuted on the way in --
            with block[2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[3], chain = block[4])

            with block[3]:
                lock_acquire(self.z_free_lock, self.L)
                aie.dma_bd(self.z_buffer, offset = 0, len = self.stage_size, dimensions = z_dims)
                lock_release(self.z_ready_lock, self.L)
                aie.next_bd(block[3])

            # ---- S2MM ch2: ssm_out_proj's y, de-permuted on the way in -------
            # Whole-buffer, like the two above: the row-block permute means no
            # token is complete until the last block lands, so there is nothing
            # to hand over early.
            with block[4]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest = block[5], chain = block[6])

            with block[5]:
                lock_acquire(self.op_free_lock, self.L)
                aie.dma_bd(self.op_buffer, offset = 0, len = self.op_size, dimensions = op_dims)
                lock_release(self.op_ready_lock, self.L)
                aie.next_bd(block[5])

            # ---- MM2S ch0: one o row per BD, to the norm tile ----------------
            # One self-chained BD cannot walk the offset, so this is L chained
            # BDs that differ only in where they start.
            with block[6]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[10], chain = block[20])

            for t in range(self.L):
                with block[10 + t]:
                    lock_acquire(self.o_ready_lock, 1)
                    aie.dma_bd(self.o_buffer, offset = t * self.DV, len = self.DV)
                    lock_release(self.o_free_lock, 1)
                    aie.next_bd(block[10 + (t + 1) % self.L])

            # ---- MM2S ch1: one z row per BD, to the gate tile ----------------
            with block[20]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[30], chain = block[40])

            for t in range(self.L):
                with block[30 + t]:
                    lock_acquire(self.z_ready_lock, 1)
                    aie.dma_bd(self.z_buffer, offset = t * self.DV, len = self.DV)
                    lock_release(self.z_free_lock, 1)
                    aie.next_bd(block[30 + (t + 1) % self.L])

            # ---- MM2S ch2: one ssm_out_proj row per BD, to the rms tile ------
            # A packet flow, unlike the two above: the rms tile's x port is a
            # merge -- the shim feeds it the layer input in phase 1, this feeds
            # it the residual's second operand in phase 2 -- so the port is
            # packet-switched and every source has to stamp an id.
            with block[40]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[50], chain = block[60])

            for t in range(self.L):
                with block[50 + t]:
                    lock_acquire(self.op_ready_lock, 1)
                    aie.dma_bd_packet(0, aux_to_rms_pkt_id.to_residual)
                    aie.dma_bd(self.op_buffer, offset = t * self.D, len = self.D)
                    lock_release(self.op_free_lock, 1)
                    aie.next_bd(block[50 + (t + 1) % self.L])

            with block[60]:
                aie.EndOp()

    @property
    def O_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def Z_in(self) -> dma_port:
        return self.S2MM[1]

    @property
    def Norm_out(self) -> dma_port:
        return self.MM2S[0]

    @property
    def Gate_out(self) -> dma_port:
        return self.MM2S[1]

    @property
    def OProj_in(self) -> dma_port:
        return self.S2MM[2]

    @property
    def Residual_out(self) -> dma_port:
        return self.MM2S[2]

