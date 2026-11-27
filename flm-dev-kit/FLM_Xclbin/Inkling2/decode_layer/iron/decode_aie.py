# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# Inkling-2-Mini decode overlay. Core rows 3 and 4 of every column are stream cores: shim MM2S channel r of column
# j feeds core (j, 3 + r) directly, one 4736-byte unit at a time (a Q4_K block; the stream is a sequence of units,
# each task starting with a header unit, see kernel/decode_core.cc), and the core's y units go back on shim S2MM
# channel r. 16 shim channels carry weights, the most the array has.
# Expert slot e is column e: its two cores share h through a buffer of the row 3 tile (locks 4-7 there), and core
# (0, 4) hands the router logits to the hub above (buffer tk, locks 13-15 of the hub tile), which routes and pushes
# the chosen experts' weights.
# Glue cores (kernel/glue.hpp) sit beside the stream cores: row 2 under row 3 (every column), row 5 over row 4
# (columns 1-7; (0, 5) is the hub). A glue core runs its partner's glue tasks on the partner's ring and y buffers
# (buffer rl and locks 8, 9 of the partner tile hand over).

import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
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
GLUE_A, GLUE_B = 9216, 3072   # floats: the scratch buffers of a glue_up core (OP_ATTN's layout in kernel/glue.hpp)
GLUE_DN_A, GLUE_DN_B = 12352, 32   # floats: of a glue_dn core (OP_MIX: 5 states of 2304, a 768 chunk, scalars)
GU = 1152            # floats: the gate / up rows of an expert half
H_PAD = 1280         # bf16: h, zero padded to the down matrix's columns
TK = 1152            # words of the hub's top-k buffer (as gu: one type for every core's tk argument)


def stream_core(ct, unit_ty, y_ty, x_ty, acc_ty, xn_ty, xs_ty, xr_ty, ss_ty, gu_ty, fn, hb, tk, row):
    w = [aie.buffer(ct.tile, unit_ty, name=f"w{i}_{ct.row}_{ct.col}") for i in range(RING)]
    y = [aie.buffer(ct.tile, y_ty, name=f"y{i}_{ct.row}_{ct.col}") for i in range(2)]
    x = aie.buffer(ct.tile, x_ty, name=f"x_{ct.row}_{ct.col}")
    acc = aie.buffer(ct.tile, acc_ty, name=f"acc_{ct.row}_{ct.col}")
    xn = aie.buffer(ct.tile, xn_ty, name=f"xn_{ct.row}_{ct.col}")
    xs = aie.buffer(ct.tile, xs_ty, name=f"xs_{ct.row}_{ct.col}")
    xr = aie.buffer(ct.tile, xr_ty, name=f"xr_{ct.row}_{ct.col}")
    ss = aie.buffer(ct.tile, ss_ty, name=f"ss_{ct.row}_{ct.col}")
    gu = aie.buffer(ct.tile, gu_ty, name=f"gu_{ct.row}_{ct.col}")
    rl = aie.buffer(ct.tile, np.ndarray[(4,), np.dtype[np.int32]], name=f"rl_{ct.row}_{ct.col}")
    aie.lock(ct.tile, lock_id=8, init=0)   # GL_GO
    aie.lock(ct.tile, lock_id=9, init=0)   # GL_DONE
    w_prod = aie.lock(ct.tile, lock_id=0, init=RING)
    w_cons = aie.lock(ct.tile, lock_id=1, init=0)
    y_prod = aie.lock(ct.tile, lock_id=2, init=2)
    y_cons = aie.lock(ct.tile, lock_id=3, init=0)

    @aie.core(ct.tile, stack_size=STACK, **core_link_kwargs("decode_core.o"))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            fn(*w, *y, x, acc, xn, xs, xr, ss, gu, hb, gu if tk is None else tk, row, rl)

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
    return w, y, rl


def glue_core(ct, partner, fn, obj, na, nb):
    w, y, rl = partner
    a = aie.buffer(ct.tile, np.ndarray[(na,), np.dtype[np.float32]], name=f"ga_{ct.row}_{ct.col}")
    b = aie.buffer(ct.tile, np.ndarray[(nb,), np.dtype[np.float32]], name=f"gb_{ct.row}_{ct.col}")

    @aie.core(ct.tile, stack_size=STACK, **core_link_kwargs(obj))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            fn(*w, *y, rl, a, b)


HUB_COL, HUB_ROW = 0, 3     # CT index: core (0, 5)
HUB_OUT, HUB_IN, HUB_ARGS, HUB_TR = 0x8000, 0x8100, 0x8200, 0x8300   # fixed addresses: the host writes args
HUB_STACK = 4096
# Control packet ids, hub -> shim j, and the read responses (kernel/hub.cc has the same table). Found by search:
# aiecc merges the rules of flows that leave a switch the same way into one mask/match pair, and the switch takes
# the first matching rule, so most id sets send some shims' packets the wrong way (check_routes.py catches it).
CTRL_IDS = [int(v) for v in os.environ.get("DEC_CTRL_IDS", "16,17,0,8,12,14,6,7").split(",")]
RESP_ID = int(os.environ.get("DEC_RESP_ID", "28"))


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
    # top-k with core (0, 3) below: tk = [logits 128][bias 128][ids 8][weights 8]
    tk = aie.buffer(ct.tile, np.ndarray[(TK,), np.dtype[np.float32]], name="hub_tk")
    aie.lock(ct.tile, lock_id=13, init=1)   # TK_PROD
    aie.lock(ct.tile, lock_id=14, init=0)   # TK_CONS
    aie.lock(ct.tile, lock_id=15, init=0)   # TK_DONE
    ty = [np.ndarray[(8,), u32], np.ndarray[(4,), u32], np.ndarray[(16,), np.dtype[np.int32]], np.ndarray[(4,), u32],
          np.ndarray[(TK,), np.dtype[np.float32]]]
    if NEW_LINK_API:
        fn = aie.external_func("hub", ty, link_with="hub.o")
    else:
        fn = aie.external_func("hub", ty)

    @aie.core(ct.tile, stack_size=HUB_STACK, **core_link_kwargs("hub.o"))
    def core_body():
        for _ in range_(0xFFFFFFFF):
            fn(out, inp, args, tr, tk)

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
    return tk


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
            gu_ty = np.ndarray[(GU,), np.dtype[np.float32]]
            hb_ty = np.ndarray[(H_PAD,), np.dtype[np.uint16]]
            rl_ty = np.ndarray[(4,), np.dtype[np.int32]]
            args = [unit_ty] * RING + [y_ty] * 2 + [x_ty, acc_ty, xn_ty, xs_ty, xr_ty, ss_ty, gu_ty, hb_ty, gu_ty, T.i32(), rl_ty]
            def glue_args(na, nb):
                return [unit_ty] * RING + [y_ty] * 2 + [rl_ty, np.ndarray[(na,), np.dtype[np.float32]],
                                                         np.ndarray[(nb,), np.dtype[np.float32]]]

            def ext(name, ty):
                return aie.external_func(name, ty, link_with=f"{name}.o") if NEW_LINK_API else aie.external_func(name, ty)
            fn = ext("decode_core", args)
            glue_fn = {"glue_up": ext("glue_up", glue_args(GLUE_A, GLUE_B)), "glue_dn": ext("glue_dn", glue_args(GLUE_DN_A, GLUE_DN_B))}
            hub = CT[HUB_ROW][HUB_COL]
            hub_tk = hub_core(hub)
            for j in range(COLS):
                # h of expert slot j: in the row 3 tile, shared with row 4 (locks 4-7 there)
                hb = aie.buffer(CT[1][j].tile, hb_ty, name=f"hb_{j}")
                aie.lock(CT[1][j].tile, lock_id=4, init=0)   # HB_H2: row 3 wrote its half
                aie.lock(CT[1][j].tile, lock_id=5, init=0)   # HB_H3: row 4 wrote its half
                aie.lock(CT[1][j].tile, lock_id=6, init=1)   # HB_C2: row 3 took the whole h
                aie.lock(CT[1][j].tile, lock_id=7, init=1)   # HB_C3: row 4 took the whole h
                for r in range(STREAM_ROWS):
                    tk = hub_tk if (j == HUB_COL and r == 1) else None
                    st = CT[1 + r][j]
                    partner = stream_core(st, unit_ty, y_ty, x_ty, acc_ty, xn_ty, xs_ty, xr_ty, ss_ty, gu_ty, fn, hb, tk, j + COLS * r)
                    connect_dma(IT[j], st, r, 1)
                    connect_dma(st, IT[j], 0, r)
                    if r == 0:
                        glue_core(CT[0][j], partner, glue_fn["glue_up"], "glue_up.o", GLUE_A, GLUE_B)
                    elif j != HUB_COL:
                        glue_core(CT[3][j], partner, glue_fn["glue_dn"], "glue_dn.o", GLUE_DN_A, GLUE_DN_B)
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
