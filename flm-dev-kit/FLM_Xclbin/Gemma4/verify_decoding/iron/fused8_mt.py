# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Memtile pieces of the fused 8-row verify layer (DESIGN=fused8, plan S6.11).

A memtile has ONE `aie.memtile_dma` body.  The stage designs (rmsglu, ple8, attn8, projstage) each owned a whole memtile; the fused layer
shares MT4 and MT5 between several stages, so every stage is a set of DMA *chains* (one `dma_start` + a cycle of BDs on one channel) that a
`memtile_hub` links into one body.  Channel and lock ids are handed out by the hub.

  hub A (MT4): q park (global, swa), y stage global ([t][512] = one head per token), y stage SWA ([head][t][256]), GLU stage ([half][t][256])
  hub B (MT5): rows stage (5 o/down rounds -> 8 token rows), ple stage (80 up packets -> token-major rows), ple feed (pli, constants, h2 rows)
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dtypes import bfloat16
import aie.dialects.aie as aie

from constants import *
from utils import AIE_MT_IP, tile_with_location, mlir_context, comment_print, lock_acquire, lock_release, dma_port
from proj_tile import KL
from rmsglu_tiles import proj_x_replay_mt_glu


class _bd:
    """one BD of a chain: buffer slice + lock actions (+ packet id, + address dims)"""
    def __init__(self, buf, offset, length, acq, acq_n, rel, rel_n, pkt = None, dims = None):
        self.buf, self.offset, self.length = buf, offset, length
        self.acq, self.acq_n, self.rel, self.rel_n = acq, acq_n, rel, rel_n
        self.pkt, self.dims = pkt, dims


class memtile_hub(AIE_MT_IP):
    """A memtile that hosts several stages: every stage registers chains, `finish()` emits the single DMA body."""
    def __init__(self, tile: tile_with_location, ctx: mlir_context, name: str):
        super().__init__(tile)
        self.ctx = ctx
        self.name = name
        self.chains = []                 # (direction, channel, [bd])
        self.next_lock = 0
        self.used = {aie.DMAChannelDir.S2MM: set(), aie.DMAChannelDir.MM2S: set()}

    # ---- resources ----
    def buffer(self, n_elems: int, name: str):
        return aie.buffer(tile = self.mt, datatype = np.ndarray[(n_elems,), np.dtype[bfloat16]], name = f"{name}_{self.mt_row}_{self.mt_col}")

    def lock(self, init: int):
        assert self.next_lock < 64, "out of memtile locks"
        lk = aie.lock(self.mt, lock_id = self.next_lock, init = init)
        self.next_lock += 1
        return lk

    def chain(self, direction, channel: int, bds):
        assert channel not in self.used[direction], f"{self.name}: channel {channel} ({direction}) used twice"
        assert channel < 6
        self.used[direction].add(channel)
        self.chains.append((direction, channel, bds))

    def S2MM_port(self, ch: int) -> dma_port:
        return dma_port(self.located_tile, ch)

    MM2S_port = S2MM_port

    def __call__(self):
        pass   # emitted by finish()

    def finish(self):
        plan, n = [], 0
        for direction, ch, bds in self.chains:
            plan.append((direction, ch, bds, n))
            n += 1 + len(bds)
        end = n

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            for i, (direction, ch, bds, start) in enumerate(plan):
                nxt = plan[i + 1][3] if i + 1 < len(plan) else end
                with block[start]:
                    aie.dma_start(direction, ch, dest = block[start + 1], chain = block[nxt])
                for k, bd in enumerate(bds):
                    with block[start + 1 + k]:
                        lock_acquire(bd.acq, bd.acq_n)
                        if bd.pkt is not None:
                            aie.dma_bd_packet(0, bd.pkt)
                        if bd.dims is None:
                            aie.dma_bd(bd.buf, offset = bd.offset, len = bd.length)
                        else:
                            aie.dma_bd(bd.buf, offset = bd.offset, len = bd.length, dimensions = bd.dims)
                        lock_release(bd.rel, bd.rel_n)
                        aie.next_bd(block[start + 1 + (k + 1) % len(bds)])
            with block[end]:
                aie.EndOp()


S2MM, MM2S = aie.DMAChannelDir.S2MM, aie.DMAChannelDir.MM2S


class pingpong_stage:
    """land a round in a ping/pong buffer, read it back with one address pattern: y stages and the GLU stage.
    kind: 'tokmajor'  raw [b = 16][t][32] -> [t][b][32]              (global y stage: one head per token)
          'half'      raw [b = 16][t][32] -> [half = 2][t][cb = 8][32]   (SWA y stage: [head][t][256]; GLU stage: [half][t][256])"""
    def __init__(self, hub: memtile_hub, name: str, s2mm: int, mm2s: int, kind: str):
        n = KL * 512
        self.n = n
        self.hub, self.s2mm, self.mm2s = hub, s2mm, mm2s
        buf = [hub.buffer(n, f"{name}_{i}") for i in range(2)]
        prod, cons = hub.lock(2), hub.lock(0)
        if kind == "tokmajor":
            dims = [(KL, 32), (16, 32 * KL), (32, 1)]
        elif kind == "half":
            dims = [(2, 8 * 16 * 16), (KL, 32), (8, 32 * KL), (32, 1)]
        else:
            raise ValueError(kind)
        hub.chain(S2MM, s2mm, [_bd(buf[i], 0, n, prod, 1, cons, 1) for i in range(2)])
        hub.chain(MM2S, mm2s, [_bd(buf[i], 0, n, cons, 1, prod, 1, dims = dims) for i in range(2)])

    @property
    def y_in(self): return self.hub.S2MM_port(self.s2mm)

    @property
    def y_out(self): return self.hub.MM2S_port(self.mm2s)


class q_park_stage:
    """q park (iron/qpark.py) as chains on hub channels: global S2MM/MM2S `g`, SWA `s`"""
    def __init__(self, hub: memtile_hub, nq: int, dh_g: int, dh_s: int, g: int, s: int):
        self.hub, self.g, self.s = hub, g, s
        for ch, dh, tag in ((g, dh_g, "global"), (s, dh_s, "swa")):
            buf = hub.buffer(nq * KL * dh, f"q_park_{tag}")
            prod, cons = hub.lock(KL), hub.lock(0)
            hub.chain(S2MM, ch, [_bd(buf, 0, nq * KL * dh, prod, KL, cons, KL)])
            hub.chain(MM2S, ch, [_bd(buf, t * dh, nq * dh, cons, 1, prod, 1, dims = [(nq, KL * dh), (dh, 1)]) for t in range(KL)])

    @property
    def Q_in(self): return self.hub.S2MM_port(self.g)
    @property
    def Q_out(self): return self.hub.MM2S_port(self.g)
    @property
    def SWA_Q_in(self): return self.hub.S2MM_port(self.s)
    @property
    def SWA_Q_out(self): return self.hub.MM2S_port(self.s)


class rows_stage:
    """the ROUNDS rounds of an o / down projection (raw [b][t][32] packets) -> KL token rows [t][ROUNDS * 512] as one packet (iron/rmsglu_tiles.py y_rows_mt)"""
    def __init__(self, hub: memtile_hub, rounds: int, out_pkt_id: int, s2mm: int, mm2s: int):
        self.hub, self.s2mm, self.mm2s = hub, s2mm, mm2s
        rnd = KL * 512
        buf = hub.buffer(rounds * rnd, "rows")
        prod, cons = hub.lock(rounds), hub.lock(0)
        hub.chain(S2MM, s2mm, [_bd(buf, r * rnd, rnd, prod, 1, cons, 1) for r in range(rounds)])
        hub.chain(MM2S, mm2s, [_bd(buf, 0, rounds * rnd, cons, rounds, prod, rounds, pkt = out_pkt_id,
                                   dims = [(KL, 32), (rounds, rnd), (16, 32 * KL), (32, 1)])])

    @property
    def y_in(self): return self.hub.S2MM_port(self.s2mm)
    @property
    def y_out(self): return self.hub.MM2S_port(self.mm2s)


class ple_stage:
    """80 up packets y[8][32] -> [rb][t][32], read back token-major [t][rb][32] (iron/ple8_tiles.py ple8_stage_mt)"""
    def __init__(self, hub: memtile_hub, D: int, s2mm: int, mm2s: int):
        self.hub, self.s2mm, self.mm2s = hub, s2mm, mm2s
        rbs = D // 32
        n = rbs * KL * 32
        buf = hub.buffer(n, "ple_stage")
        prod, cons = hub.lock(1), hub.lock(0)
        hub.chain(S2MM, s2mm, [_bd(buf, 0, n, prod, 1, cons, 1)])
        hub.chain(MM2S, mm2s, [_bd(buf, 0, n, cons, 1, prod, 1, dims = [(KL, 32), (rbs, KL * 32), (32, 1)])])

    @property
    def y_in(self): return self.hub.S2MM_port(self.s2mm)
    @property
    def up_out(self): return self.hub.MM2S_port(self.mm2s)


class ple_feed:
    """The ordered sources of the PLE tiles.
    G.S2MM0 wants  pli [8][256], then 8 row blocks x 10 k blocks of h slices [t][256]   (shim BD sizes {8,10,8,256} strides {0,256,2560,1} in the stage design)
    N.S2MM1 wants  [w_post_pli(2560) | layer_scalar + 31 pad] (2592 el), then the h rows [8][2560]
    pli / constants come from the shim (packets), h2 comes from the norm tile (8 packets of one row, id 24)."""
    def __init__(self, hub: memtile_hub, D: int, pli_d: int, s2mm_pli: int, s2mm_h: int, s2mm_const: int, mm2s_g: int, mm2s_n: int):
        self.hub = hub
        self.ch = dict(pli = s2mm_pli, h = s2mm_h, const = s2mm_const, g = mm2s_g, n = mm2s_n)
        rb_g = pli_d // 32                       # 8 row blocks of the gate matrix
        kb = D // 256                            # 10 k blocks
        pli_n, const_n, h_n = KL * pli_d, D + 32, KL * D
        pli = hub.buffer(pli_n, "ple_pli")
        const = hub.buffer(const_n, "ple_const")
        h = hub.buffer(h_n, "ple_h")
        pli_p, pli_c = hub.lock(1), hub.lock(0)
        cst_p, cst_c = hub.lock(1), hub.lock(0)
        h_p, h_c, h_mid = hub.lock(KL), hub.lock(0), hub.lock(0)
        hub.chain(S2MM, s2mm_pli, [_bd(pli, 0, pli_n, pli_p, 1, pli_c, 1)])
        hub.chain(S2MM, s2mm_const, [_bd(const, 0, const_n, cst_p, 1, cst_c, 1)])
        hub.chain(S2MM, s2mm_h, [_bd(h, r * D, D, h_p, 1, h_c, 1) for r in range(KL)])
        # -> G: pli, then the h slices; the h read is the first of two readers: it hands the buffer on to the N reader through h_mid
        # (a memtile BD cannot repeat with stride 0: the 8 row blocks of the gate matrix are 8 chained BDs over the same slices)
        rep = hub.lock(0)
        sl = kb * KL * 256
        hdims = [(kb, 256), (KL, D), (256, 1)]
        g_bds = [_bd(pli, 0, pli_n, pli_c, 1, pli_p, 1)]
        for i in range(rb_g):
            g_bds.append(_bd(h, 0, sl, h_c if i == 0 else rep, KL if i == 0 else 1, h_mid if i == rb_g - 1 else rep, KL if i == rb_g - 1 else 1, dims = hdims))
        hub.chain(MM2S, mm2s_g, g_bds)
        # -> N: constants, then the h rows; the buffer is free again after this read
        hub.chain(MM2S, mm2s_n, [
            _bd(const, 0, const_n, cst_c, 1, cst_p, 1),
            _bd(h, 0, h_n, h_mid, KL, h_p, KL),
        ])

    @property
    def pli_in(self): return self.hub.S2MM_port(self.ch["pli"])
    @property
    def const_in(self): return self.hub.S2MM_port(self.ch["const"])
    @property
    def h_in(self): return self.hub.S2MM_port(self.ch["h"])
    @property
    def to_G(self): return self.hub.MM2S_port(self.ch["g"])
    @property
    def to_N(self): return self.hub.MM2S_port(self.ch["n"])


class proj_x_replay_mt_f8(proj_x_replay_mt_glu):
    """proj_x_replay_mt_glu + phases that land `land_rounds` packets (one per token) back to back, [t][x_dim] layout, normal replay read:
    the attention's o rows arrive one packet per token (o_global / o_swa)."""

    def __call__(self):
        plan, n = [], 0
        def n_bds(p):
            return getattr(p, "glu_rounds", 0) or getattr(p, "land_rounds", 0) or 1
        for st in self.stages.values():
            plan.append(("land", aie.DMAChannelDir.S2MM, st["s2mm_ch"], st, n))
            n += 1 + sum(n_bds(p) for p in st["phases"])
        for g in self.replays.values():
            plan.append(("replay", aie.DMAChannelDir.MM2S, g["mm2s_ch"], g, n))
            n += 1 + g["fanout"]
        end = n
        n_locks = len(self.stages) + sum(g["fanout"] for g in self.replays.values())
        chain_lock = aie.lock(self.mt, lock_id = n_locks, init = 0)
        assert n_locks + 1 <= 64

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            for i, (kind, dir, ch, arg, start) in enumerate(plan):
                nxt = plan[i + 1][4] if i + 1 < len(plan) else end
                with block[start]:
                    aie.dma_start(dir, ch, dest = block[start + 1], chain = block[nxt])
                if kind == "land":
                    ph = arg["phases"]
                    flat = []
                    for k, p in enumerate(ph):
                        c = n_bds(p)
                        flat += [(k, j, c) for j in range(c)]
                    for idx, (k, j, c) in enumerate(flat):
                        p = ph[k]
                        with block[start + 1 + idx]:
                            lock_acquire(arg["prod_lock"] if j == 0 else chain_lock, ph[k - 1].credits if j == 0 else 1)
                            if n_bds(p) > 1:
                                chunk = KL * p.x_dim // n_bds(p)
                                aie.dma_bd(arg["buffer"], offset = j * chunk, len = chunk)
                            else:
                                aie.dma_bd(arg["buffer"], offset = 0, len = KL * p.x_dim)
                            if j == c - 1:
                                lock_release(p.chan["cons_locks"][0], p.credits)
                            else:
                                lock_release(chain_lock, 1)
                            aie.next_bd(block[start + 1 + (idx + 1) % len(flat)])
                else:
                    f = arg["fanout"]
                    glu_fed = any(getattr(p, "glu_rounds", 0) for p in arg["phases"])
                    for k in range(f):
                        with block[start + 1 + k]:
                            lock_acquire(arg["cons_locks"][k], 1)
                            aie.dma_bd_packet(0, self.x_pkt_id)
                            if glu_fed:
                                aie.dma_bd(arg["stage"]["buffer"], offset = 0, len = KL * arg["x_dim"],
                                           dimensions = [(arg["x_dim"] // self.x_slice_size, KL * self.x_slice_size), (self.x_slice_size // 8, 8),
                                                         (KL, self.x_slice_size), (8, 1)])
                            else:
                                aie.dma_bd(arg["stage"]["buffer"], offset = 0, len = KL * arg["x_dim"],
                                           dimensions = [(arg["x_dim"] // self.x_slice_size, self.x_slice_size), (self.x_slice_size // 8, 8),
                                                         (KL, arg["x_dim"]), (8, 1)])
                            if k + 1 < f:
                                lock_release(arg["cons_locks"][k + 1], 1)
                            else:
                                lock_release(arg["stage"]["prod_lock"], 1)
                            aie.next_bd(block[start + 1 + (k + 1) % f])
            with block[end]:
                aie.EndOp()
