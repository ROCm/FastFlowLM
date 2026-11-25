# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# Inkling-2-Mini decode overlay. Core rows 2 and 3 of every column are stream cores: shim MM2S channel r of column
# j feeds core (j, 2 + r) directly, one 4736-byte unit at a time (a Q4_K block; the stream is a sequence of units,
# each task starting with a header unit, see kernel/decode_core.cc), and the core's y units go back on shim S2MM
# channel r. 16 shim channels carry weights, the most the array has.

import numpy as np

from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
try:  # mlir-aie as pinned by CI
    from aie.helpers.dialects.ext.scf import _for as range_
except ImportError:  # newer mlir-aie moved it
    from aie.helpers.dialects.scf import _for as range_

from utils import connect_dma, core_link_kwargs, dma_bd_compat, generate_tile_map, lock_acquire, lock_release, NEW_LINK_API

COLS = 8
import os
STREAM_ROWS = int(os.environ.get("DEC_ROWS", "2"))   # 1: debug build, shim S2MM 1 of column 0 takes the hub trace
UNIT = 4736          # bytes: one Q4_K block of 32 x 256
RING = 4             # units in flight per core
Y_UNIT = 96          # floats: one row group (3 row blocks)
X_MAX = 3072         # bf16: the widest input (o proj)
STACK = 8192         # bytes: decode_core frame is ~3.6 KB


def stream_core(ct, unit_ty, y_ty, x_ty, acc_ty, xn_ty, xs_ty, xr_ty, ss_ty, fn):
    w = [aie.buffer(ct.tile, unit_ty, name=f"w{i}_{ct.row}_{ct.col}") for i in range(RING)]
    y = [aie.buffer(ct.tile, y_ty, name=f"y{i}_{ct.row}_{ct.col}") for i in range(2)]
    x = aie.buffer(ct.tile, x_ty, name=f"x_{ct.row}_{ct.col}")
    acc = aie.buffer(ct.tile, acc_ty, name=f"acc_{ct.row}_{ct.col}")
    xn = aie.buffer(ct.tile, xn_ty, name=f"xn_{ct.row}_{ct.col}")
    xs = aie.buffer(ct.tile, xs_ty, name=f"xs_{ct.row}_{ct.col}")
    xr = aie.buffer(ct.tile, xr_ty, name=f"xr_{ct.row}_{ct.col}")
    ss = aie.buffer(ct.tile, ss_ty, name=f"ss_{ct.row}_{ct.col}")
    w_prod = aie.lock(ct.tile, lock_id=0, init=RING)
    w_cons = aie.lock(ct.tile, lock_id=1, init=0)
    y_prod = aie.lock(ct.tile, lock_id=2, init=2)
    y_cons = aie.lock(ct.tile, lock_id=3, init=0)

    @aie.core(ct.tile, stack_size=STACK, **core_link_kwargs("decode_core.o"))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            fn(*w, *y, x, acc, xn, xs, xr, ss)

    @aie.mem(ct.tile)
    def mem_body(block):
        aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=1, dest=block[1], chain=block[1 + RING])
        for i in range(RING):
            with block[1 + i]:
                lock_acquire(w_prod, 1)
                dma_bd_compat(w[i])
                lock_release(w_cons, 1)
                aie.next_bd(block[1 + (i + 1) % RING])
        with block[1 + RING]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=0, dest=block[2 + RING], chain=block[4 + RING])
        for i in range(2):
            with block[2 + RING + i]:
                lock_acquire(y_cons, 1)
                dma_bd_compat(y[i])
                lock_release(y_prod, 1)
                aie.next_bd(block[2 + RING + (i + 1) % 2])
        with block[4 + RING]:
            aie.EndOp()


HUB_COL, HUB_ROW = 0, 2     # CT index: core (0, 4)
HUB_OUT, HUB_IN, HUB_ARGS, HUB_TR = 0x8000, 0x8100, 0x8200, 0x8300   # fixed addresses: the host writes args
HUB_STACK = 4096
# Control packet ids, hub -> shim j, and the read responses (kernel/hub.cc has the same table). Found by search:
# aiecc merges the rules of flows that leave a switch the same way into one mask/match pair, and the switch takes
# the first matching rule, so most id sets send some shims' packets the wrong way (check_routes.py catches it).
CTRL_IDS = [int(v) for v in os.environ.get("DEC_CTRL_IDS", "5,8,17,31,22,30,21,23").split(",")]
RESP_ID = int(os.environ.get("DEC_RESP_ID", "9"))


def hub_core(ct):
    u32 = np.dtype[np.uint32]
    out = aie.buffer(ct.tile, np.ndarray[(8,), u32], name="hub_out", address=HUB_OUT)
    inp = aie.buffer(ct.tile, np.ndarray[(4,), u32], name="hub_in", address=HUB_IN)
    args = aie.buffer(ct.tile, np.ndarray[(16,), np.dtype[np.int32]], name="hub_args", address=HUB_ARGS)
    out_prod = aie.lock(ct.tile, lock_id=6, init=1)
    out_cons = aie.lock(ct.tile, lock_id=7, init=0)
    in_prod = aie.lock(ct.tile, lock_id=8, init=1)
    in_cons = aie.lock(ct.tile, lock_id=9, init=0)
    aie.lock(ct.tile, lock_id=10, init=0)   # start, set by the host
    tr = aie.buffer(ct.tile, np.ndarray[(4,), u32], name="hub_tr", address=HUB_TR)
    tr_prod = aie.lock(ct.tile, lock_id=11, init=1)
    tr_cons = aie.lock(ct.tile, lock_id=12, init=0)
    ty = [np.ndarray[(8,), u32], np.ndarray[(4,), u32], np.ndarray[(16,), np.dtype[np.int32]], np.ndarray[(4,), u32]]
    if NEW_LINK_API:
        fn = aie.external_func("hub", ty, link_with="hub.o")
    else:
        fn = aie.external_func("hub", ty)

    @aie.core(ct.tile, stack_size=HUB_STACK, **core_link_kwargs("hub.o"))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            fn(out, inp, args, tr)

    @aie.mem(ct.tile)
    def mem_body(block):
        aie.dma_start(channel_dir=aie.DMAChannelDir.S2MM, channel_index=1, dest=block[1], chain=block[2])
        with block[1]:
            lock_acquire(in_prod, 1)
            aie.dma_bd(inp, offset=0, len=1, bd_id=11)
            lock_release(in_cons, 1)
            aie.next_bd(block[1])
        with block[2]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=1, dest=block[3], chain=block[4])
        with block[3]:
            lock_acquire(out_cons, 1)
            aie.dma_bd(out, offset=0, len=1, bd_id=10)   # length set per packet by the kernel
            lock_release(out_prod, 1)
            aie.next_bd(block[3])
        with block[4]:
            aie.dma_start(channel_dir=aie.DMAChannelDir.MM2S, channel_index=0, dest=block[5], chain=block[6])
        with block[5]:
            lock_acquire(tr_cons, 1)
            aie.dma_bd(tr, offset=0, len=1)
            lock_release(tr_prod, 1)
            aie.next_bd(block[5])
        with block[6]:
            aie.EndOp()


def design():
    with mlir_mod_ctx() as ctx:
        @aie.device(aie.AIEDevice.npu2)
        def device_body():
            IT, MT, CT = generate_tile_map(4 + 2, COLS)
            unit_ty = np.ndarray[(UNIT,), np.dtype[np.uint8]]
            y_ty = np.ndarray[(Y_UNIT,), np.dtype[np.float32]]
            x_ty = np.ndarray[(X_MAX,), np.dtype[np.uint16]]
            acc_ty = np.ndarray[(12 * 64,), np.dtype[np.float32]]
            # one column block's prepared x: xnr 8 x 64 fp32, xs 8 x 8 bf16, xr 32 x 64 bf16; ss 16 x 64 bf16
            xn_ty = np.ndarray[(512,), np.dtype[np.float32]]
            xs_ty = np.ndarray[(64,), np.dtype[np.uint16]]
            xr_ty = np.ndarray[(2048,), np.dtype[np.uint16]]
            ss_ty = np.ndarray[(1024,), np.dtype[np.uint16]]
            args = [unit_ty] * RING + [y_ty] * 2 + [x_ty, acc_ty, xn_ty, xs_ty, xr_ty, ss_ty]
            if NEW_LINK_API:
                fn = aie.external_func("decode_core", args, link_with="decode_core.o")
            else:
                fn = aie.external_func("decode_core", args)
            for j in range(COLS):
                for r in range(STREAM_ROWS):
                    stream_core(CT[r][j], unit_ty, y_ty, x_ty, acc_ty, xn_ty, xs_ty, xr_ty, ss_ty, fn)
                    connect_dma(IT[j], CT[r][j], r, 1)
                    connect_dma(CT[r][j], IT[j], 0, r)
            hub = CT[HUB_ROW][HUB_COL]
            hub_core(hub)
            for j in range(COLS):
                aie.packetflow(pkt_id=CTRL_IDS[j], source=hub.tile, source_port=aie.WireBundle.DMA, source_channel=1,
                               dest=IT[j].tile, dest_port=aie.WireBundle.TileControl, dest_channel=0)
            aie.packetflow(pkt_id=RESP_ID, source=IT[0].tile, source_port=aie.WireBundle.TileControl, source_channel=0,
                           dest=hub.tile, dest_port=aie.WireBundle.DMA, dest_channel=1)
            if STREAM_ROWS == 1:
                connect_dma(hub, IT[0], 0, 1)
        print(ctx.module)


if __name__ == "__main__":
    design()
