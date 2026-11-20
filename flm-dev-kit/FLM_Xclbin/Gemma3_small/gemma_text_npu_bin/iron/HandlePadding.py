# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import mlir_context, tile_with_location, lock_acquire, lock_release


def HandlePadding(tile: tile_with_location, ctx: mlir_context):
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    x_ty = ctx.data_tys["x_ty"]
    x_padded_ty = ctx.data_tys["x_padded_ty"]

    x_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    x_cons_lock = aie.lock(ct, lock_id = 1, init = 0)

    

  
    x_padded_buffer = aie.buffer(ct, x_padded_ty, name = f"x_padded_buffer_{ct_row}_{ct_col}")
    D = ctx.parameters["D"]
    PADDED_D = ctx.parameters["PADDED_D"]

    @aie.mem(ct)
    def mem_dma_body(block):
        s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_padded_buffer)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])

        with block[2]:
            s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[4])

        with block[3]:
            lock_acquire(x_cons_lock, 1)
            aie.dma_bd_packet(0, 8)
            aie.dma_bd(x_padded_buffer, offset = 0, len = D)
            lock_release(x_prod_lock, 1)
            aie.next_bd(block[3])

        with block[4]:
            aie.EndOp()