# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig

import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

from utils import comment_print, mlir_context, tile_with_location, lock_acquire, lock_release, AIE_CT_IP
from constants import L_TOKENS


# npu2 memory geometry, per AIETargetModel.h: a compute tile's memory module is
# 64 KiB split into four 16 KiB banks, and AIEAssignBuffers round-robins one
# allocation at a time over those banks, each of which must hold the whole
# allocation. So a D-sized bf16 row (10 KiB at D = 5120) gets a bank to itself
# and a tile holds four of them.
CT_BANK_SIZE = 16 * 1024
CT_NUM_BANKS = 4


def _reachable_from(core: tile_with_location, mem: tile_with_location) -> bool:
    """AIE2TargetModel::isLegalMemAffinity, for a compute tile's core.

    A core loads and stores across four 64 KiB windows of its own address map:
    its own memory module at 0x70000 ("east"), plus west at 0x50000, north at
    0x60000 and south at 0x40000. South is only legal if the tile below is
    another compute tile -- row 1 is the memtile row and is excluded.
    """
    if mem.col == core.col and mem.row == core.row + 1:
        return True                                   # north
    if mem.col == core.col and mem.row == core.row - 1 and mem.row >= 2:
        return True                                   # south, and not a memtile
    if mem.row == core.row and mem.col == core.col - 1:
        return True                                   # west
    return False


class rms_residual_tile(AIE_CT_IP):
    """Both of the layer's norms and both of its residuals, L_TOKENS rows each.

    Phase 1 is input_layernorm: L_TOKENS rows in on X_in, out on Y_out (to the
    projection engine) and AB_Yout (to the a/b tile). Each row is also saved
    into the resident hidden state, below. Phase 2 is the delta-net residual add
    plus post_attention_layernorm: L_TOKENS rows in on the same X_in --
    ssm_out_proj's output only -- added into the saved row in place and normed,
    out on Y_out only. Phase 3 is the MLP residual, down_proj's output on the
    same X_in again, added in place and sent out unnormed as the layer's h2.
    See kernels/rms_residual.cc.

    Three phases and no new hardware. The x chain below is a ping/pong that does
    not care how many rows run through it, 3 * L_TOKENS is even so its parity
    still lines up across invocations, and the y BD is self-chained and fires
    once per row whatever the phase -- the phases differ only in the packet id
    the kernel stamps, so they are told apart at the switchbox and not here. The
    one thing that is per-phase is the weight BD, and phase 3 has no norm, so it
    does not touch it.

    The compute stays one row wide and loops; the four local buffers are a
    ping/pong pair, the norm weight and the packet-framed output, one per 16 KiB
    bank. What does not fit locally is the residual: L_TOKENS x D bf16 is 80 KiB
    against a 64 KiB tile.

    So it lives next door. A core reaches its west, north and south neighbours'
    memory modules as well as its own (see _reachable_from), and this tile's are
    idle -- CT[1][3] carries no IP and column 2 has no compute tiles at all -- so
    the hidden state is split across two of them, four rows each, and the core
    loads and stores it directly. No DMA is involved: the neighbours have no
    aie.core and no aie.mem, they are just memory, and having no core they also
    reserve no stack, so all four of their banks are free.

    That is what lets phase 2 take L_TOKENS rows instead of 2 * L_TOKENS: the
    layer input never leaves the array. The saved row is overwritten by the sum,
    so the same 80 KiB is also h1 for the MLP's residual and then h2 for the next
    layer.
    """
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, residual_tiles: list[tile_with_location], stack_size: int = 1024, link_with: str = "rms_residual.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing rms_residual_tile at ({self.ct_row}, {self.ct_col})")

        self.model_config = model_config
        self.x_ty = np.ndarray[(model_config.D,), self.dtype]
        self.w_ty = np.ndarray[(model_config.D,), self.dtype]
        self.y_pkt_ty = np.ndarray[(model_config.D + 16,), self.dtype]

        self.D = model_config.D
        comment_print(f"rms_residual_tile D: {self.D}")

        # Four D-sized buffers, one per 16 KiB bank. Declaration order is the
        # placement order, so this must stay at four.
        self.x_ping = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_ping_{self.ct_row}_{self.ct_col}")
        self.x_pong = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_pong_{self.ct_row}_{self.ct_col}")
        self.w_buffer = ctx.buffer(tile = self.located_tile, dtype = self.w_ty, name = f"w_buffer_{self.ct_row}_{self.ct_col}")
        self.y = ctx.buffer(tile = self.located_tile, dtype = self.y_pkt_ty, name = f"y_{self.ct_row}_{self.ct_col}")

        # The resident hidden state, on the idle neighbours. One D-sized row per
        # bank, filled tile by tile, so row t lives on residual_tiles[t // rows_per_tile].
        rows_per_tile = CT_NUM_BANKS * (CT_BANK_SIZE // (self.D * np.dtype(bfloat16).itemsize))
        if rows_per_tile < 1:
            raise ValueError(f"a D = {self.D} bf16 row does not fit a {CT_BANK_SIZE} B bank")
        capacity = rows_per_tile * len(residual_tiles)
        if L_TOKENS > capacity:
            raise ValueError(
                f"L_TOKENS = {L_TOKENS} needs {L_TOKENS} resident rows but "
                f"{len(residual_tiles)} neighbour tile(s) hold {capacity} "
                f"({rows_per_tile} each). Give rms_residual_tile another reachable tile."
            )
        for mem in residual_tiles:
            if not _reachable_from(self.located_tile, mem):
                raise ValueError(
                    f"tile ({mem.row}, {mem.col}) is not reachable from the rms core at "
                    f"({self.ct_row}, {self.ct_col}); only west, north and south are"
                )
        self.residual = []
        for t in range(L_TOKENS):
            mem = residual_tiles[t // rows_per_tile]
            comment_print(f"rms_residual_tile hidden state row {t} on tile ({mem.row}, {mem.col})")
            self.residual.append(ctx.buffer(tile = mem, dtype = self.x_ty, name = f"h_{t}_{mem.row}_{mem.col}"))
        self.w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 2, init = 0)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.x_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)
        # init 0, not 1. The stage-4 kernel body releases both consumer locks and
        # then waits on both producer locks *after* writing the row, so the
        # producer lock has to start empty -- an initial token would let row t+1's
        # rms_norm overwrite `y` while row t's AB DMA was still draining, which
        # corrupts every row but the first and the last. The full-layer body below
        # acquires before the write instead, and that pairing does want init = 1.
        self.ab_out_prod_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.ab_out_cons_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        # (y, x_ping, x_pong, w) then the L_TOKENS neighbour-resident rows, in
        # token order. The kernel gathers them back into an array, so this
        # ordering is the ABI -- kernels/rms_residual.cc static_asserts on the
        # count rather than trusting it.
        self.core_func = ctx.func("rms_residual", [self.y_pkt_ty, self.x_ty, self.x_ty, self.w_ty] + [self.x_ty] * L_TOKENS)

        # create mlir code
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.core_func(self.y, self.x_ping, self.x_pong, self.w_buffer, *self.residual)

        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])
            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_ping, offset = 0, len = self.D)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_pong, offset = 0, len = self.D)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[3], chain=block[5])
            
            # One weight vector per *norm*: the kernel acquires w_cons once
            # outside each of its first two row loops and releases w_prod at the
            # end of each, so this self-chained BD fires twice per invocation --
            # not three times, phase 3 is a bare residual add and takes no
            # weight. The host sends 2 * D -- input_layernorm, then
            # post_attention_layernorm overwriting it. That reload is the only
            # reason a four-bank tile can run both norms.
            with block[3]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer, offset = 0, len = self.D)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[3])


            with block[5]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[6], chain = block[7])

            # Fires once per row: 1 word of packet header at y[14:16] then the
            # D-element result at y[16:]. The header is stripped by the
            # packetflow, so DDR sees L_TOKENS * D contiguous bf16.
            with block[6]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y, offset = 14, len = self.D + 2)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[6])

            # Stage 4: the same normalised row, a second time, to the AB tile.
            #
            # The designed hand-off copies the row into its own y_out buffer
            # first, but the tile has no bank left for a fifth D-sized buffer
            # (see the class docstring). Reading `self.y` again costs nothing:
            # the two channels take different slices of one buffer -- ch0 sends
            # the packet header at y[14:16] plus the data, ch1 sends the bare
            # data at y[16:] on a circuit-switched flow, which is what the AB
            # tile's x_buffer wants. The kernel holds the row until *both*
            # channels have drained, so there is no overwrite race.
            with block[7]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[8], chain = block[10])

            with block[8]:
                lock_acquire(self.ab_out_cons_lock, 1)
                aie.dma_bd(self.y, offset = 16, len = self.D)
                lock_release(self.ab_out_prod_lock, 1)
                aie.next_bd(block[8])

            with block[10]:
                aie.EndOp()

    @property
    def Y_out(self):
        return self.MM2S[0]
    
    @property
    def AB_Yout(self):
        return self.MM2S[1]


    @property
    def X_in(self):
        return self.S2MM[0]
    
    @property
    def W_in(self):
        return self.S2MM[1]
