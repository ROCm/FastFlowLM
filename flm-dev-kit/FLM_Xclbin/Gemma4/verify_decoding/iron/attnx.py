# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""attnx: 8-token verify attention that reads every K/V row once per verify (stream A, notes/attn_longctx.md).

Tiles (see kernels/attnx_common.h for the data layouts):
  attnx_score_tile  q.K^T + masked online softmax for 4 tokens x 8 heads; P + row corrections to the PV tiles of its tokens
  attnx_pv_tile     P.V: global 2 tokens x 8 heads per tile (4 tiles), SWA 4 tokens x 8 heads (tiles 0 and 2; 1 and 3 drain V)
  attnx_qpark       memtile: q of the 8 tokens (RoPE output) re-read per score tile in the layout its Q BDs expect
  attnx_oasm        memtile: lands each PV tile's o rows on a circuit of its own (global: 4 x 8192, SWA: 2 x 8192 elements), token-major,
                    and sends 8 token rows per kind (packet ids 17 / 18 = the o landings of the fused design)

Both layer kinds share these tiles; the kind is an RTP.  Everything the tile DMAs see is kind-independent (K/V sub-chunks of 4096 elements,
Q as 4 BDs of 4096, P messages of MSG elements) except the o, which each PV kernel sends on the MM2S channel of its kind.
No packet flow has two sources with paths that diverge: mlir-aie's router multicasts same-id flows and widens packet-rule masks over the ids of
other groups, which duplicated / misrouted the o packets and deadlocked the first version (notes/attn_longctx.md).
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

from constants import L_TOKENS, to_proj_x_in_pkt_id
from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print, AIE_CT_IP, AIE_MT_IP

SUB = 4096                 # K/V sub-chunk (one kv head): global 8 keys x 512, SWA 16 keys x 256
QBD = 4096                 # one Q BD (8 global rows / 16 SWA rows)
MSG = 320                  # P message, bf16 slots (kernels/attnx_common.h pmsg)
TILE_DIMS = [(8, 8), (64, 64), (8, 1)]   # every 4096-element sub-chunk / Q BD lands as 8x8 blocks [kd][8][8]
PKT_Q = (29, 30)           # q park -> score tile s
PKT_O = (int(to_proj_x_in_pkt_id.from_attn), int(to_proj_x_in_pkt_id.from_swa_attn))   # o rows out of the assembly memtile (17, 18)

assert L_TOKENS == 8


class attnx_score_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, stack_size: int = 1024 * 8, link_with: str = "attnx_score.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        r, c = self.ct_row, self.ct_col
        comment_print(f"Initializing attnx_score_tile at ({r}, {c})")
        self.q_ty = np.ndarray[(4 * QBD,), self.dtype]
        self.k_ty = np.ndarray[(2 * SUB,), self.dtype]
        self.p_ty = np.ndarray[(2 * MSG,), self.dtype]
        self.q = ctx.buffer(tile = self.located_tile, dtype = self.q_ty, name = f"attnx_q_{r}_{c}")
        self.k = ctx.buffer(tile = self.located_tile, dtype = self.k_ty, name = f"attnx_k_{r}_{c}")
        self.pa = ctx.buffer(tile = self.located_tile, dtype = self.p_ty, name = f"attnx_pa_{r}_{c}")
        self.pb = ctx.buffer(tile = self.located_tile, dtype = self.p_ty, name = f"attnx_pb_{r}_{c}")
        self.rtp = ctx.buffer(f"RTP_attnx_score_{r}_{c}", self.located_tile, ctx.data_tys["RTP_ty"], use_write_rtp = True)
        self.q_prod = aie.lock(self.ct, lock_id = 0, init = 1)
        self.q_cons = aie.lock(self.ct, lock_id = 1, init = 0)
        self.k_prod = aie.lock(self.ct, lock_id = 2, init = 2)
        self.k_cons = aie.lock(self.ct, lock_id = 3, init = 0)
        self.pa_prod = aie.lock(self.ct, lock_id = 4, init = 2)
        self.pa_cons = aie.lock(self.ct, lock_id = 5, init = 0)
        self.pb_prod = aie.lock(self.ct, lock_id = 6, init = 2)
        self.pb_cons = aie.lock(self.ct, lock_id = 7, init = 0)
        self.q_chain = aie.lock(self.ct, lock_id = 8, init = 0)
        self.func = ctx.func("attnx_score", [self.q_ty, self.k_ty, self.p_ty, self.p_ty, ctx.data_tys["RTP_ty"]])
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.func(self.q, self.k, self.pa, self.pb, self.rtp)

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[10])
            for i in range(4):
                with block[1 + i]:
                    lock_acquire(self.q_prod if i == 0 else self.q_chain, 1)
                    aie.dma_bd(self.q, offset = i * QBD, len = QBD, dimensions = TILE_DIMS)
                    lock_release(self.q_cons if i == 3 else self.q_chain, 1)
                    aie.next_bd(block[1 + (i + 1) % 4])
            with block[10]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[11], chain = block[20])
            for i in range(2):
                with block[11 + i]:
                    lock_acquire(self.k_prod, 1)
                    aie.dma_bd(self.k, offset = i * SUB, len = SUB)   # linear: a 16-byte-run S2MM pattern ran at ~1/4 of the stream rate
                    lock_release(self.k_cons, 1)
                    aie.next_bd(block[11 + (i + 1) % 2])
            for ch, buf, prod, cons, base in ((0, self.pa, self.pa_prod, self.pa_cons, 20), (1, self.pb, self.pb_prod, self.pb_cons, 30)):
                with block[base]:
                    aie.dma_start(aie.DMAChannelDir.MM2S, ch, dest = block[base + 1], chain = block[base + 10])
                for i in range(2):
                    with block[base + 1 + i]:
                        lock_acquire(cons, 1)
                        aie.dma_bd(buf, offset = i * MSG, len = MSG)
                        lock_release(prod, 1)
                        aie.next_bd(block[base + 1 + (i + 1) % 2])
            with block[40]:
                aie.EndOp()

    @property
    def Q_in(self):
        return self.S2MM[0]

    @property
    def K_in(self):
        return self.S2MM[1]

    @property
    def P_out(self):
        return self.MM2S


class attnx_pv_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, index: int, stack_size: int = 1024 * 4, link_with: str = "attnx_pv.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        r, c = self.ct_row, self.ct_col
        comment_print(f"Initializing attnx_pv_tile {index} at ({r}, {c})")
        self.index = index
        self.v_ty = np.ndarray[(2 * SUB,), self.dtype]
        self.p_ty = np.ndarray[(2 * MSG,), self.dtype]
        self.y_ty = np.ndarray[(4 * SUB,), self.dtype]     # 32 KB: f32 y [head][kd][8][8] (global), bf16 o in place at the end
        self.v = ctx.buffer(tile = self.located_tile, dtype = self.v_ty, name = f"attnx_v_{r}_{c}")
        self.p = ctx.buffer(tile = self.located_tile, dtype = self.p_ty, name = f"attnx_p_{r}_{c}")
        self.y = ctx.buffer(tile = self.located_tile, dtype = self.y_ty, name = f"attnx_y_{r}_{c}")
        self.rtp = ctx.buffer(f"RTP_attnx_pv_{r}_{c}", self.located_tile, ctx.data_tys["RTP_ty"], use_write_rtp = True)
        self.v_prod = aie.lock(self.ct, lock_id = 0, init = 2)
        self.v_cons = aie.lock(self.ct, lock_id = 1, init = 0)
        self.p_prod = aie.lock(self.ct, lock_id = 2, init = 2)
        self.p_cons = aie.lock(self.ct, lock_id = 3, init = 0)
        self.og_cons = aie.lock(self.ct, lock_id = 4, init = 0)
        self.os_cons = aie.lock(self.ct, lock_id = 5, init = 0)
        self.o_sent = aie.lock(self.ct, lock_id = 6, init = 0)
        self.og_chain = aie.lock(self.ct, lock_id = 7, init = 0)
        self.os_chain = aie.lock(self.ct, lock_id = 8, init = 0)
        self.func = ctx.func("attnx_pv", [self.v_ty, self.p_ty, self.y_ty, ctx.data_tys["RTP_ty"]])
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.func(self.v, self.p, self.y, self.rtp)

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[10])
            for i in range(2):
                with block[1 + i]:
                    lock_acquire(self.v_prod, 1)
                    aie.dma_bd(self.v, offset = i * SUB, len = SUB)
                    lock_release(self.v_cons, 1)
                    aie.next_bd(block[1 + (i + 1) % 2])
            with block[10]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[11], chain = block[20])
            for i in range(2):
                with block[11 + i]:
                    lock_acquire(self.p_prod, 1)
                    aie.dma_bd(self.p, offset = i * MSG, len = MSG)
                    lock_release(self.p_cons, 1)
                    aie.next_bd(block[11 + (i + 1) % 2])
            # o (circuits to the assembly memtile): global MM2S0 (tl2, h) x 2048: (hql 4, stride 16), (kd 64, stride 128), (d8 8, 1);
            # SWA MM2S1 (tiles 0, 2) (tl, h) x 1024: (hql 4, stride 16), (kd 32, stride 128), (d8 8, 1), offset (2h + tl/2)*32*128 + (tl%2)*8
            g = [(tl2 * 8 + h * 64 * 128, [(4, 16), (64, 128), (8, 1)], 2048) for tl2 in range(2) for h in range(2)]
            sw = [((2 * h + tl // 2) * 32 * 128 + (tl % 2) * 8, [(4, 16), (32, 128), (8, 1)], 1024) for tl in range(4) for h in range(2)]
            chains = [(0, self.og_cons, self.og_chain, g, 20)]
            if self.index % 2 == 0:
                chains.append((1, self.os_cons, self.os_chain, sw, 30))
            for ch, cons, chain, bds, base in chains:
                with block[base]:
                    aie.dma_start(aie.DMAChannelDir.MM2S, ch, dest = block[base + 1], chain = block[base + 10 if ch == 1 or len(chains) == 2 else 40])
                for i, (off, dims, n) in enumerate(bds):
                    with block[base + 1 + i]:
                        lock_acquire(cons if i == 0 else chain, 1)
                        aie.dma_bd(self.y, offset = off, len = n, dimensions = dims)
                        lock_release(self.o_sent if i == len(bds) - 1 else chain, 1)
                        aie.next_bd(block[base + 1 + (i + 1) % len(bds)])
            with block[40]:
                aie.EndOp()

    @property
    def V_in(self):
        return self.S2MM[0]

    @property
    def P_in(self):
        return self.S2MM[1]

    @property
    def O_out(self):
        return self.MM2S[0]

    @property
    def SWA_O_out(self):
        assert self.index % 2 == 0, "SWA o comes from PV tiles 0 and 2"
        return self.MM2S[1]


class attnx_qpark(AIE_MT_IP):
    """q of the 8 tokens, as the RoPE tiles emit it ([head][t][DH] per kind), re-read per score tile s:
         global BD (s, rb): rows (hq 2rb, 2rb+1) x (t 4s..4s+3), [(2, 8 DH), (4, DH), (DH, 1)]          -> 4 x 4096 per score tile
         SWA    BD (s, b) : rows (hq 4b..4b+3) x (t 4s..4s+3), [(4, 8 DH), (4, DH), (DH, 1)], then 2 filler BDs  -> 4 x 4096
       packet PKT_Q[s].  Counting locks: the write BD takes / gives 8 units, every read BD one."""

    def __init__(self, tile: tile_with_location, ctx: mlir_context, nq: int, dh_global: int, dh_swa: int):
        super().__init__(tile)
        self.nq, self.dg, self.ds = nq, dh_global, dh_swa
        comment_print(f"Initializing attnx_qpark at MT({self.mt_row}, {self.mt_col})")
        self.g_buf = aie.buffer(self.mt, np.ndarray[(nq * L_TOKENS * dh_global,), np.dtype[bfloat16]], name = f"attnx_qpark_g_{self.mt_row}_{self.mt_col}")
        self.s_buf = aie.buffer(self.mt, np.ndarray[(nq * L_TOKENS * dh_swa,), np.dtype[bfloat16]], name = f"attnx_qpark_s_{self.mt_row}_{self.mt_col}")
        self.g_prod = aie.lock(self.mt, lock_id = 0, init = 8)
        self.g_cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.s_prod = aie.lock(self.mt, lock_id = 2, init = 8)
        self.s_cons = aie.lock(self.mt, lock_id = 3, init = 0)
        self.__call__()

    def __call__(self):
        nq, dg, ds = self.nq, self.dg, self.ds
        g_reads, s_reads = [], []
        for s in range(2):
            for rb in range(4):
                g_reads.append((PKT_Q[s], (2 * rb) * 8 * dg + 4 * s * dg, [(2, 8 * dg), (4, dg), (dg, 1)]))
        for s in range(2):
            for b in range(2):
                s_reads.append((PKT_Q[s], (4 * b) * 8 * ds + 4 * s * ds, [(4, 8 * ds), (4, ds), (ds, 1)]))
            for f in range(2):
                s_reads.append((PKT_Q[s], 0, None))   # filler

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # global: S2MM0 / MM2S0, SWA: S2MM1 / MM2S1
            for ch, buf, prod, cons, reads, n, base in ((0, self.g_buf, self.g_prod, self.g_cons, g_reads, nq * L_TOKENS * dg, 0),
                                                        (1, self.s_buf, self.s_prod, self.s_cons, s_reads, nq * L_TOKENS * ds, 20)):
                with block[base]:
                    aie.dma_start(aie.DMAChannelDir.S2MM, ch, dest = block[base + 1], chain = block[base + 2])
                with block[base + 1]:
                    lock_acquire(prod, 8)
                    aie.dma_bd(buf, offset = 0, len = n)
                    lock_release(cons, 8)
                    aie.next_bd(block[base + 1])
                with block[base + 2]:
                    aie.dma_start(aie.DMAChannelDir.MM2S, ch, dest = block[base + 3], chain = block[base + 20])
                for i, (pkt, off, dims) in enumerate(reads):
                    with block[base + 3 + i]:
                        lock_acquire(cons, 1)
                        aie.dma_bd_packet(0, pkt)
                        if dims is None:
                            aie.dma_bd(buf, offset = 0, len = QBD)
                        else:
                            aie.dma_bd(buf, offset = off, len = QBD, dimensions = dims)
                        lock_release(prod, 1)
                        aie.next_bd(block[base + 3 + (i + 1) % len(reads)])
            with block[40]:
                aie.EndOp()

    @property
    def Q_in(self):
        return self.S2MM[0]

    @property
    def SWA_Q_in(self):
        return self.S2MM[1]

    @property
    def Q_out(self):
        return self.MM2S[0]

    @property
    def SWA_Q_out(self):
        return self.MM2S[1]


class attnx_oasm(AIE_MT_IP):
    """o assembly: global pieces (PV tile j: tokens 2j, 2j+1 x 4096) on S2MM j, SWA pieces (PV tile 2s: tokens 4s..4s+3 x 2048) on
    S2MM 4 + s, landed token-major; MM2S0 sends 8 global rows (4096) as packets PKT_O[0], MM2S1 8 SWA rows (2048) as PKT_O[1].
    Counting locks: every landing BD takes one unit of its kind's prod lock and gives one cons unit; the first row BD takes all of them."""

    def __init__(self, tile: tile_with_location, ctx: mlir_context, dq: int, swa_dq: int):
        super().__init__(tile)
        self.dq, self.sdq = dq, swa_dq
        comment_print(f"Initializing attnx_oasm at MT({self.mt_row}, {self.mt_col})")
        self.g_buf = aie.buffer(self.mt, np.ndarray[(L_TOKENS * dq,), np.dtype[bfloat16]], name = f"attnx_o_g_{self.mt_row}_{self.mt_col}")
        self.s_buf = aie.buffer(self.mt, np.ndarray[(L_TOKENS * swa_dq,), np.dtype[bfloat16]], name = f"attnx_o_s_{self.mt_row}_{self.mt_col}")
        self.g_prod = aie.lock(self.mt, lock_id = 0, init = 4)
        self.g_cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.g_chain = aie.lock(self.mt, lock_id = 2, init = 0)
        self.s_prod = aie.lock(self.mt, lock_id = 3, init = 2)
        self.s_cons = aie.lock(self.mt, lock_id = 4, init = 0)
        self.s_chain = aie.lock(self.mt, lock_id = 5, init = 0)
        self.__call__()

    def __call__(self):
        dq, sdq, L = self.dq, self.sdq, L_TOKENS
        lands = [(j, self.g_buf, self.g_prod, self.g_cons, j * 2 * dq, 2 * dq) for j in range(4)] + \
                [(4 + s, self.s_buf, self.s_prod, self.s_cons, s * 4 * sdq, 4 * sdq) for s in range(2)]

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            n = 0
            for i, (ch, buf, prod, cons, off, ln) in enumerate(lands):
                with block[n]:
                    aie.dma_start(aie.DMAChannelDir.S2MM, ch, dest = block[n + 1], chain = block[n + 2])
                with block[n + 1]:
                    lock_acquire(prod, 1)
                    aie.dma_bd(buf, offset = off, len = ln)
                    lock_release(cons, 1)
                    aie.next_bd(block[n + 1])
                n += 2
            for ch, buf, prod, cons, chain, w, k, pkt in ((0, self.g_buf, self.g_prod, self.g_cons, self.g_chain, dq, 4, PKT_O[0]),
                                                         (1, self.s_buf, self.s_prod, self.s_cons, self.s_chain, sdq, 2, PKT_O[1])):
                with block[n]:
                    aie.dma_start(aie.DMAChannelDir.MM2S, ch, dest = block[n + 1], chain = block[n + 1 + L])
                for t in range(L):
                    with block[n + 1 + t]:
                        lock_acquire(cons if t == 0 else chain, k if t == 0 else 1)
                        aie.dma_bd_packet(0, pkt)
                        aie.dma_bd(buf, offset = t * w, len = w)
                        lock_release(prod if t == L - 1 else chain, k if t == L - 1 else 1)
                        aie.next_bd(block[n + 1 + (t + 1) % L])
                n += 1 + L
            with block[n]:
                aie.EndOp()

    def G_in(self, j):
        return self.S2MM[j]

    def S_in(self, s):
        return self.S2MM[4 + s]

    @property
    def G_out(self):
        return self.MM2S[0]

    @property
    def S_out(self):
        return self.MM2S[1]


def attnx_q_park_stage(hub, nq: int, dh_g: int, dh_s: int, g: int, s: int):
    """attnx_qpark as chains of a fused8 memtile hub (iron/fused8_mt.py): global S2MM / MM2S `g`, SWA `s`.  Returns an object with the
    ports Q_in / SWA_Q_in / Q_out / SWA_Q_out (the read chains send packets PKT_Q[s] to score tile s)."""
    from fused8_mt import _bd, S2MM, MM2S
    L = L_TOKENS

    class _park:
        pass

    p = _park()
    for ch, dh, kind in ((g, dh_g, "global"), (s, dh_s, "swa")):
        buf = hub.buffer(nq * L * dh, f"attnx_q_park_{kind}")
        prod, cons = hub.lock(8), hub.lock(0)
        hub.chain(S2MM, ch, [_bd(buf, 0, nq * L * dh, prod, 8, cons, 8)])
        reads = []
        for sc in range(2):
            if kind == "global":
                for rb in range(4):
                    reads.append(_bd(buf, (2 * rb) * L * dh + 4 * sc * dh, QBD, cons, 1, prod, 1, pkt = PKT_Q[sc],
                                     dims = [(2, L * dh), (4, dh), (dh, 1)]))
            else:
                for b in range(2):
                    reads.append(_bd(buf, (4 * b) * L * dh + 4 * sc * dh, QBD, cons, 1, prod, 1, pkt = PKT_Q[sc],
                                     dims = [(4, L * dh), (4, dh), (dh, 1)]))
                for f in range(2):
                    reads.append(_bd(buf, 0, QBD, cons, 1, prod, 1, pkt = PKT_Q[sc]))   # filler
        hub.chain(MM2S, ch, reads)
    p.Q_in, p.Q_out = hub.S2MM_port(g), hub.MM2S_port(g)
    p.SWA_Q_in, p.SWA_Q_out = hub.S2MM_port(s), hub.MM2S_port(s)
    return p
