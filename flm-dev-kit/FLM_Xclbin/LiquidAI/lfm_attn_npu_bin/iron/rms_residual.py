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

def sum_tile(tile: tile_with_location, ctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    sum = ctx.funcs["sum"]
    D = ctx.parameters["D"]
    x_ty = ctx.data_tys["x_ty"]
    w_ty = ctx.data_tys["w_ty"]
    y_ty = ctx.data_tys["y_ty"]
    # q_proj tile and mem 
    # double buffer
    x_0 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_0_{ct_row}_{ct_col}")
    x_1 = aie.buffer(tile = ct, datatype = x_ty, name = f"x_1_{ct_row}_{ct_col}")
    y = aie.buffer(tile = ct, datatype = y_ty, name = f"y_{ct_row}_{ct_col}")

    y_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    y_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    x_prod_lock = aie.lock(ct, lock_id = 2, init = 2)
    x_cons_lock = aie.lock(ct, lock_id = 3, init = 0)


    @aie.core(ct, link_with = "sum.o")
    def core_body():
        for _ in range_(0xFFFFFFFF):
            sum(y, x_0, x_1)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        with block[1]:
            aie.use_lock(x_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(x_0, offset = 0, len = D)
            aie.use_lock(x_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[2])
        
        with block[2]:  
            aie.use_lock(x_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(x_1, offset = 0, len = D)
            aie.use_lock(x_cons_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[1])

        with block[3]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[5])
        
        with block[4]:
            aie.use_lock(y_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd_packet(0, 0)
            aie.dma_bd(y, offset = 0, len = D)
            aie.use_lock(y_prod_lock, aie.LockAction.Release, value = 1)
            aie.next_bd(block[4])

        with block[5]:
            aie.EndOp()


def rms_residual_tile(tile: tile_with_location, ctx: mlir_context):

    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    rms_residual = ctx.funcs["rms_residual"]
    D = ctx.parameters["D"]
    x_ty = ctx.data_tys["x_ty"]
    w_ty = ctx.data_tys["w_ty"]
    y_pkt_ty = ctx.data_tys["y_pkt_ty"]
    # q_proj tile and mem 
    # double buffer
    x = aie.buffer(tile = ct, datatype = x_ty, name = f"x_{ct_row}_{ct_col}")
    w = aie.buffer(tile = ct, datatype = w_ty, name = f"w_{ct_row}_{ct_col}")
    y = aie.buffer(tile = ct, datatype = y_pkt_ty, name = f"y_{ct_row}_{ct_col}")

    w_prod_lock = aie.lock(ct, lock_id = 0, init = 1)
    w_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 2, init = 1)
    y_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    x_prod_lock = aie.lock(ct, lock_id = 4, init = 1)
    x_cons_lock = aie.lock(ct, lock_id = 5, init = 0)


    # aie.packetflow(pkt_id = 0, source = tile_ref, source_port = 0, source_channel = 0, dest = tile_ref, dest_port = 0, dest_channel = 0, keep_pkt_header = False)
    @aie.core(ct, link_with = "rms_residual.o")
    def core_body():
        for _ in range_(0xFFFFFFFF):
            rms_residual(y, x, w)
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])
        with block[1]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x, offset = 0, len = D)
            lock_release(x_cons_lock, 1)
            aie.next_bd(block[1])

        with block[2]:
            s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[3], chain=block[4])
        
        with block[3]:
            lock_acquire(w_prod_lock, 1)
            aie.dma_bd(w, offset = 0, len = D)
            lock_release(w_cons_lock, 1)
            aie.next_bd(block[3])
        
        with block[4]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[5], chain = block[6])
            
        with block[5]:
            lock_acquire(y_cons_lock, 1)
            aie.dma_bd(y, offset = 14, len = D + 2)
            lock_release(y_prod_lock, 1)
            aie.next_bd(block[5])


        with block[6]:
            aie.EndOp()
