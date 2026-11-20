# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from platform import release
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

from utils import DMAChannelDir, mlir_context, next_bd, tile_with_location, connect_dma, lock_acquire, lock_release
from typing import List


def proj_assemble(tile: tile_with_location, ctx: mlir_context):

    mt = tile.tile
    row = tile.row
    col = tile.col

    m = ctx.parameters["m"]
    MVM_INTERLEAVE_SIZE = ctx.parameters["MVM_INTERLEAVE_SIZE"]
    x_chunk_size = ctx.parameters["x_chunk_size"]
    x_chunk_ty = ctx.data_tys["x_chunk_ty"]
    m_full_ty = ctx.data_tys["m_full_ty"]
    assert x_chunk_size == 4*MVM_INTERLEAVE_SIZE

    y_buffer_0 = aie.buffer(mt, m_full_ty, name = f"y_buffer_0_{row}_{col}")
    y_buffer_1 = aie.buffer(mt, m_full_ty, name = f"y_buffer_1_{row}_{col}")
    x_buffer_0 = aie.buffer(mt, x_chunk_ty, name = f"x_buffer_0_{row}_{col}")
    x_buffer_1 = aie.buffer(mt, x_chunk_ty, name = f"x_buffer_1_{row}_{col}")
    x_buffer_2 = aie.buffer(mt, x_chunk_ty, name = f"x_buffer_2_{row}_{col}")
    x_buffer_3 = aie.buffer(mt, x_chunk_ty, name = f"x_buffer_3_{row}_{col}")



    m_prod_lock_0 = aie.lock(mt, lock_id = 0, init = 2)
    m_prod_lock_1 = aie.lock(mt, lock_id = 1, init = 0)
    m_prod_lock_2 = aie.lock(mt, lock_id = 2, init = 0)
    m_prod_lock_3 = aie.lock(mt, lock_id = 3, init = 0)
    m_cons_lock = aie.lock(mt, lock_id = 4, init = 0)

    x_prod_lock = aie.lock(mt, lock_id = 5, init = 4)
    
    x_cons_lock_0 = aie.lock(mt, lock_id=6, init=0)
    x_cons_lock_1 = aie.lock(mt, lock_id=7, init=0)
    x_cons_lock_2 = aie.lock(mt, lock_id=8, init=0)
    x_cons_lock_3 = aie.lock(mt, lock_id=9, init=0)
    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
        with block[1]:
            lock_acquire(m_prod_lock_0, 1)
            aie.dma_bd(y_buffer_0, offset = 0, len = 4 * m + 2)
            lock_release(m_prod_lock_1, 1)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(m_prod_lock_0, 1)
            aie.dma_bd(y_buffer_1, offset = 0, len = 4 * m + 2)
            lock_release(m_prod_lock_1, 1)
            aie.next_bd(block[1])

        with block[3]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])

        with block[4]:
            lock_acquire(m_prod_lock_1, 1)
            aie.dma_bd(y_buffer_0, offset = 4 * m + 2, len = 4 * m)
            lock_release(m_prod_lock_2, 1)
            aie.next_bd(block[5])

        with block[5]:
            lock_acquire(m_prod_lock_1, 1)
            aie.dma_bd(y_buffer_1, offset = 4 * m + 2, len = 4 * m)
            lock_release(m_prod_lock_2, 1)
            aie.next_bd(block[4])


        with block[6]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])

        with block[7]:
            lock_acquire(m_prod_lock_2, 1)
            aie.dma_bd(y_buffer_0, offset = 8 * m + 2, len = 4 * m)
            lock_release(m_prod_lock_3, 1)
            aie.next_bd(block[8])

        with block[8]:
            lock_acquire(m_prod_lock_2, 1)
            aie.dma_bd(y_buffer_1, offset = 8 * m + 2, len = 4 * m)
            lock_release(m_prod_lock_3, 1)
            aie.next_bd(block[7])


        with block[9]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

        with block[10]:
            lock_acquire(m_prod_lock_3, 1)
            aie.dma_bd(y_buffer_0, offset = 12 * m + 2, len = 4 * m)
            lock_release(m_cons_lock, 1)
            aie.next_bd(block[11])

        with block[11]:
            lock_acquire(m_prod_lock_3, 1)
            aie.dma_bd(y_buffer_1, offset = 12 * m + 2, len = 4 * m)
            lock_release(m_cons_lock, 1)
            aie.next_bd(block[10])


        with block[12]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[13], chain = block[15])

        with block[13]:
            lock_acquire(m_cons_lock, 1)
            aie.dma_bd(y_buffer_0)
            lock_release(m_prod_lock_0, 1)
            aie.next_bd(block[14])

        with block[14]:
            lock_acquire(m_cons_lock, 1)
            aie.dma_bd(y_buffer_1)
            lock_release(m_prod_lock_0, 1)
            aie.next_bd(block[13])

        with block[15]:
            aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[16], chain = block[20])

        with block[16]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_0)
            lock_release(x_cons_lock_0, 1)
            aie.next_bd(block[17])

        with block[17]:
            lock_acquire(x_prod_lock, 1)
            aie.dma_bd(x_buffer_1)
            lock_release(x_cons_lock_0, 1)
            aie.next_bd(block[18])

        with block[18]:
            aie.use_lock(x_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(x_buffer_2)
            aie.use_lock(x_cons_lock_0, aie.LockAction.Release, value = 1)
            aie.next_bd(block[19])

        with block[19]:
            aie.use_lock(x_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(x_buffer_3)
            aie.use_lock(x_cons_lock_0, aie.LockAction.Release, value = 1)
            aie.next_bd(block[16])

        with block[20]:
            aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[21], chain = block[25])

        with block[21]:
            lock_acquire(x_cons_lock_0, 1)
            aie.dma_bd(x_buffer_0, offset=0, len = MVM_INTERLEAVE_SIZE )
            lock_release(x_cons_lock_1, 1)
            aie.next_bd(block[22])

        with block[22]:
            lock_acquire(x_cons_lock_0, 1)
            aie.dma_bd(x_buffer_1, offset=0, len = MVM_INTERLEAVE_SIZE )
            lock_release(x_cons_lock_1, 1)
            aie.next_bd(block[23])

        with block[23]:
            aie.use_lock(x_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(x_buffer_2, offset=0, len = MVM_INTERLEAVE_SIZE )
            aie.use_lock(x_cons_lock_1, aie.LockAction.Release, value = 1)
            aie.next_bd(block[24])

        with block[24]:
            aie.use_lock(x_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
            aie.dma_bd(x_buffer_3, offset=0, len = MVM_INTERLEAVE_SIZE )
            aie.use_lock(x_cons_lock_1, aie.LockAction.Release, value = 1)
            aie.next_bd(block[21])

        with block[25]:
            aie.dma_start(DMAChannelDir.MM2S, 2, dest=block[26], chain=block[30])
        with block[26]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_0, offset=MVM_INTERLEAVE_SIZE, len = MVM_INTERLEAVE_SIZE )
            lock_release(x_cons_lock_2, 1)
            next_bd(block[27])
        with block[27]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_1, offset=MVM_INTERLEAVE_SIZE, len = MVM_INTERLEAVE_SIZE )
            lock_release(x_cons_lock_2, 1)
            next_bd(block[28])
        with block[28]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_2, offset=MVM_INTERLEAVE_SIZE, len = MVM_INTERLEAVE_SIZE )
            lock_release(x_cons_lock_2, 1)
            next_bd(block[29])
        with block[29]:
            lock_acquire(x_cons_lock_1, 1)
            aie.dma_bd(x_buffer_3, offset=MVM_INTERLEAVE_SIZE, len = MVM_INTERLEAVE_SIZE )
            lock_release(x_cons_lock_2, 1)
            next_bd(block[26])
        with block[30]:
            aie.dma_start(DMAChannelDir.MM2S, 3, dest=block[31], chain=block[35])
        with block[31]:
            lock_acquire(x_cons_lock_2, 1)
            aie.dma_bd(x_buffer_0, offset=2*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_cons_lock_3, 1)
            next_bd(block[32])
        with block[32]:
            lock_acquire(x_cons_lock_2, 1)
            aie.dma_bd(x_buffer_1, offset=2*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_cons_lock_3, 1)
            next_bd(block[33])
        with block[33]:
            lock_acquire(x_cons_lock_2, 1)
            aie.dma_bd(x_buffer_2, offset=2*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_cons_lock_3, 1)
            next_bd(block[34])
        with block[34]:
            lock_acquire(x_cons_lock_2, 1)
            aie.dma_bd(x_buffer_3, offset=2*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_cons_lock_3, 1)
            next_bd(block[31])
        with block[35]:
            aie.dma_start(DMAChannelDir.MM2S, 4, dest=block[36], chain=block[40])
        with block[36]:
            lock_acquire(x_cons_lock_3,1)
            aie.dma_bd(x_buffer_0, offset=3*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_prod_lock, 1)
            next_bd(block[37])
        with block[37]:
            lock_acquire(x_cons_lock_3,1)
            aie.dma_bd(x_buffer_1, offset=3*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_prod_lock, 1)
            next_bd(block[38])
        with block[38]:
            lock_acquire(x_cons_lock_3,1)
            aie.dma_bd(x_buffer_2, offset=3*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_prod_lock, 1)
            next_bd(block[39])
        with block[39]:
            lock_acquire(x_cons_lock_3,1)
            aie.dma_bd(x_buffer_3, offset=3*MVM_INTERLEAVE_SIZE, len=MVM_INTERLEAVE_SIZE)
            lock_release(x_prod_lock, 1)
            next_bd(block[36])
        with block[40]:
            aie.EndOp()