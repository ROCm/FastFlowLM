# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
from aie.ir import IntegerAttr, IntegerType

from aie.dialects.aie import *
from aie.dialects.aiex import *
try:  # mlir-aie as pinned by CI
    from aie.helpers.dialects.ext.scf import _for as range_
except ImportError:  # newer mlir-aie moved it
    from aie.helpers.dialects.scf import _for as range_
from utils import core_link_kwargs, dma_bd_compat, mlir_context, tile_with_location, lock_acquire, lock_release

# Core memory layout (both cores of a pair; the kernel's statics follow the RTP). Keep in sync with
# kernel/gemm.h (lock ids) and host/seq_gen.hpp (RTP address).
ACC_ADDR = 2048          # 32 KB fp32 accumulators, bf16 y written in place
X_ADDR = (34816, 37120)  # 2304 B each
W_ADDR = (39424, 48768)  # 9344 B each
RTP_ADDR = 58112         # 16 x int32

# core locks
W_PROD = (0, 2)          # 2 credits: both cores of the pair read every weight buffer
W_CONS = (1, 3)
X_PROD, X_CONS = 4, 5
Y_PROD, Y_CONS = 6, 7

# memtile layout and locks; the host sequence programs the weight path on these
MT_Y_ADDR = 0
MT_PANEL_ADDR = (65536, 294912)
MT_Y_PROD = (0, 1, 2, 3)        # per core row, 4 strip credits
MT_Y_CONS = (4, 5, 6, 7)
MT_PANEL_LOCKS = (8, 9, 10, 11)  # panel 0 / 1 producer, panel 0 / 1 consumer (host-initialized)


def _core(ct: tile_with_location, mctx: mlir_context):
    tile = ct.tile
    i32 = IntegerType.get_signless(32)

    def pinned(ty, name, address):
        return aie.buffer(tile=tile, datatype=ty, name=f"{name}_{ct.row}_{ct.col}", address=address)
    acc = pinned(mctx.acc_ty, "acc", ACC_ADDR)
    x = [pinned(mctx.x_ty, f"x{i}", X_ADDR[i]) for i in range(2)]
    w = [pinned(mctx.w_ty, f"w{i}", W_ADDR[i]) for i in range(2)]
    rtp = mctx.buffers[f"rtp_{ct.row}_{ct.col}"]
    rtp.owner.attributes["address"] = IntegerAttr.get(i32, RTP_ADDR)

    w_prod = [aie.lock(tile, lock_id=W_PROD[i], init=2) for i in range(2)]
    w_cons = [aie.lock(tile, lock_id=W_CONS[i], init=0) for i in range(2)]
    x_prod = aie.lock(tile, lock_id=X_PROD, init=2)
    x_cons = aie.lock(tile, lock_id=X_CONS, init=0)
    y_prod = aie.lock(tile, lock_id=Y_PROD, init=4)
    y_cons = aie.lock(tile, lock_id=Y_CONS, init=0)
    return acc, x, w, rtp, (w_prod, w_cons, x_prod, x_cons, y_prod, y_cons)


def _core_dma(ct: tile_with_location, acc, x, w, locks):
    w_prod, w_cons, x_prod, x_cons, y_prod, y_cons = locks

    @aie.mem(ct.tile)
    def mem_body(block):
        aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=0, dest=block[1], chain=block[3])
        for i in range(2):
            with block[1 + i]:
                lock_acquire(x_prod, 1)
                dma_bd_compat(x[i])
                lock_release(x_cons, 1)
                aie.next_bd(block[2 - i])

        with block[3]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=1, dest=block[4], chain=block[6])
        for i in range(2):
            with block[4 + i]:
                lock_acquire(w_prod[i], 2)
                dma_bd_compat(w[i])
                lock_release(w_cons[i], 2)
                aie.next_bd(block[5 - i])

        # y: one 16-row strip per descriptor, the kernel leaves it as eight 16 x 16 bf16 blocks at
        # 512 B steps from the strip start; sent as 16 rows of 128 columns
        with block[6]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=0, dest=block[7], chain=block[11])
        for s in range(4):
            with block[7 + s]:
                lock_acquire(y_cons, 1)
                dma_bd_compat(acc, offset=2048 * s, len=1024, dimensions=[(16, 8), (8, 128), (8, 1)])
                lock_release(y_prod, 1)
                aie.next_bd(block[7 + (s + 1) % 4])

        with block[11]:
            aie.EndOp()


def gemm_pair(tile_e: tile_with_location, tile_o: tile_with_location, mctx: mlir_context):
    """A vertical pair: the even core holds the lo weight halves, the odd core the hi halves."""
    acc_e, x_e, w_e, rtp_e, locks_e = _core(tile_e, mctx)
    acc_o, x_o, w_o, rtp_o, locks_o = _core(tile_o, mctx)

    @aie.core(tile_e.tile, stack_size=2048, **core_link_kwargs("gemm_e.o"))
    def core_e():
        for _ in range_(0xFFFFFFFF):
            mctx.gemm_e(acc_e, x_e[0], x_e[1], w_e[0], w_e[1], w_o[0], w_o[1], rtp_e)

    @aie.core(tile_o.tile, stack_size=2048, **core_link_kwargs("gemm_o.o"))
    def core_o():
        for _ in range_(0xFFFFFFFF):
            mctx.gemm_o(acc_o, x_o[0], x_o[1], w_e[0], w_e[1], w_o[0], w_o[1], rtp_o)

    _core_dma(tile_e, acc_e, x_e, w_e, locks_e)
    _core_dma(tile_o, acc_o, x_o, w_o, locks_o)


def gemm_memtile(tile: tile_with_location, mctx: mlir_context):
    mt = tile.tile
    y = aie.buffer(tile=mt, datatype=mctx.y_mem_ty, name=f"y_{tile.row}_{tile.col}", address=MT_Y_ADDR)
    for i in range(2):
        aie.buffer(tile=mt, datatype=mctx.panel_ty, name=f"p{i}_{tile.row}_{tile.col}", address=MT_PANEL_ADDR[i])
    y_prod = [aie.lock(mt, lock_id=MT_Y_PROD[r], init=4) for r in range(4)]
    y_cons = [aie.lock(mt, lock_id=MT_Y_CONS[r], init=0) for r in range(4)]
    for lid in MT_PANEL_LOCKS:
        aie.lock(mt, lock_id=lid, init=0)

    @aie.memtile_dma(mt)
    def memtile_dma_body(block):
        # each channel's dma_start sits in the chain block of the previous one
        def start(chain_from, direction, ch, first):
            if chain_from is None:
                aie.dma_start(channel_dir=direction, channel_index=ch, dest=block[first], chain=block[first + 4])
            else:
                with block[chain_from]:
                    aie.dma_start(channel_dir=direction, channel_index=ch, dest=block[first], chain=block[first + 4])
            return first + 4

        # y in: core row r on S2MM 1 + r, one 4 KB strip per descriptor into its 16 KB row block
        chain, first = None, 1
        for r in range(4):
            chain = start(chain, aie.DMAChannelDir.S2MM, 1 + r, first)
            for s in range(4):
                with block[first + s]:
                    lock_acquire(y_prod[r], 1)
                    dma_bd_compat(y, offset=8192 * r + 2048 * s, len=2048)
                    lock_release(y_cons[r], 1)
                    aie.next_bd(block[first + (s + 1) % 4])
            first = chain + 1
        # y out: the 4 row blocks of a round in row order, 64 rows x 128 columns each
        chain = start(chain, aie.DMAChannelDir.MM2S, 2, first)
        for r in range(4):
            with block[first + r]:
                lock_acquire(y_cons[r], 4)
                dma_bd_compat(y, offset=8192 * r, len=8192)
                lock_release(y_prod[r], 4)
                aie.next_bd(block[first + (r + 1) % 4])
        with block[chain]:
            aie.EndOp()
