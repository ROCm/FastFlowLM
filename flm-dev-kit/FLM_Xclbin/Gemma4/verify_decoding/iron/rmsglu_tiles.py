# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tiles of the 8-row rms/GLU stage design (plan S6.6 / S6.7, D17).

  rms_residual_l8_tile   norm tile (3,2) with the residual h[8][D] lent on the coreless neighbour (3,3)
  glu_l8_tile            GLU on a free tile (the old (3,3) slot is the lender now)
  glu_stage_mt           memtile: y rounds of the up|gate projection -> [half][t][256] order for the GLU (and the down x layout)
  y_rows_mt              memtile: the 5 rounds of an o / down projection -> 8 token rows [t][D] for the norm tile
  proj_x_replay_mt_glu   proj_x_replay_mt whose `down` phase is fed round by round by the GLU (landing layout [slice][t][256])
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

from utils import comment_print, mlir_context, tile_with_location, lock_acquire, lock_release, dma_port, connect_port
from utils import AIE_CT_IP, AIE_MT_IP
from constants import *
import proj_tile
from proj_tile import proj_x_replay_mt, KL

# packet ids of this design (kernels/rms_residual_l8.cc PKT_* must match)
PKT_RMS_TO_MAIN = int(proj_x_in_pkt_id.to_main)   # 16: normed rows -> replay tile main buffer (qkv, up|gate)
PKT_RMS_OUT = 24                                  # phase 3 rows -> shim (layer output)
PKT_ROWS_TO_RMS = 25                              # y_rows_mt -> norm tile
PKT_SHIM_TO_RMS = 26                              # the layer input rows, shim -> norm tile


# core -> neighbour memory windows, see Qwen gated_attention_verify/iron/rms_residual.py
def _reachable_from(core: tile_with_location, mem: tile_with_location) -> bool:
    if mem.col == core.col and mem.row == core.row + 1:
        return True
    if mem.col == core.col and mem.row == core.row - 1 and mem.row >= 2:
        return True
    if mem.row == core.row and mem.col == core.col - 1:
        return True
    return False


class rms_residual_l8_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, lender: tile_with_location,
                 stack_size: int = 4096, link_with: str = "rms_residual_l8.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        assert KL > 1, "rms_residual_l8 is the 8-row kernel"
        comment_print(f"Initializing rms_residual_l8_tile at ({self.ct_row}, {self.ct_col}), residual lent on ({lender.row}, {lender.col})")
        if not _reachable_from(self.located_tile, lender):
            raise ValueError(f"lender ({lender.row},{lender.col}) is not reachable from the rms core at ({self.ct_row},{self.ct_col})")
        D = model_config.D
        self.D = D
        self.x_ty = np.ndarray[(D,), self.dtype]
        self.w_ty = np.ndarray[(4, D), self.dtype]
        self.y_pkt_ty = np.ndarray[(D + 16,), self.dtype]
        self.x_ping = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_ping_{self.ct_row}_{self.ct_col}")
        self.x_pong = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_pong_{self.ct_row}_{self.ct_col}")
        self.w_buffer = ctx.buffer(tile = self.located_tile, dtype = self.w_ty, name = f"w_buffer_{self.ct_row}_{self.ct_col}")
        self.y = ctx.buffer(tile = self.located_tile, dtype = self.y_pkt_ty, name = f"y_{self.ct_row}_{self.ct_col}")
        # the resident hidden state: one D-wide row per token on the lender (8 x 5 KiB = 40 KiB of its 64)
        self.residual = [ctx.buffer(tile = lender, dtype = self.x_ty, name = f"h_{t}_{lender.row}_{lender.col}") for t in range(KL)]
        self.w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 2, init = 0)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.x_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.core_func = ctx.func("rms_residual_l8", [self.y_pkt_ty, self.x_ty, self.x_ty, self.w_ty] + [self.x_ty] * KL)
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.core_func(self.y, self.x_ping, self.x_pong, self.w_buffer, *self.residual)

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[2])
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
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[3], chain = block[5])
            with block[3]:      # the four norm weight vectors, once per dispatch
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer, offset = 0, len = 4 * self.D)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[3])
            with block[5]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[6], chain = block[10])
            with block[6]:      # one packet per row: 1 header word at y[14:16], then D elements
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y, offset = 14, len = self.D + 2)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[6])
            with block[10]:
                aie.EndOp()

    @property
    def Y_out(self) -> dma_port:
        return self.MM2S[0]

    @property
    def X_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def W_in(self) -> dma_port:
        return self.S2MM[1]


class glu_l8_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 4096, link_with: str = "glu_l8.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)
        assert KL > 1
        self.round = KL * 16 * model_config.q4nx_row_block_size       # 4096 elements per round
        self.x_ty = np.ndarray[(2 * self.round,), self.dtype]         # [up round | gate round]
        self.y_ty = np.ndarray[(self.round,), self.dtype]
        comment_print(f"glu_l8_tile at ({self.ct_row}, {self.ct_col}): round {self.round}")
        self.x_buffer_0 = aie.buffer(tile = self.ct, datatype = self.x_ty, name = f"glu_x_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = aie.buffer(tile = self.ct, datatype = self.x_ty, name = f"glu_x_1_{self.ct_row}_{self.ct_col}")
        self.y_buffer_0 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"glu_y_0_{self.ct_row}_{self.ct_col}")
        self.y_buffer_1 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"glu_y_1_{self.ct_row}_{self.ct_col}")
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.glu = aie.external_func("glu_l8", [self.y_ty, self.y_ty, self.x_ty, self.x_ty])
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.glu(self.y_buffer_0, self.y_buffer_1, self.x_buffer_0, self.x_buffer_1)

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])
            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])
            with block[4]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd_packet(0, int(proj_x_in_pkt_id.to_down))
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[5])
            with block[5]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd_packet(0, int(proj_x_in_pkt_id.to_down))
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[4])
            with block[6]:
                aie.EndOp()

    @property
    def X_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def Y_out(self) -> dma_port:
        return self.MM2S[0]


class glu_stage_mt(AIE_MT_IP):
    """up|gate y rounds (raw engine order [b = 16][t][32], one packet of KL * 512 elements each) -> the GLU tile in the order [half][t][256]:
    element (half, t, c) of a round is raw (b = 8 half + c // 32, t, c % 32).  One 4-dim read BD (64 B runs, read side)."""
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, tap_pkt_id: int = None):
        super().__init__(tile)
        self.n = KL * 16 * model_config.q4nx_row_block_size
        self.tap_pkt_id = tap_pkt_id
        self.round_ty = np.ndarray[(self.n,), np.dtype[bfloat16]]
        self.buf_0 = ctx.buffer(tile = self.located_tile, dtype = self.round_ty, name = f"glu_stage_0_{self.mt_row}_{self.mt_col}")
        self.buf_1 = ctx.buffer(tile = self.located_tile, dtype = self.round_ty, name = f"glu_stage_1_{self.mt_row}_{self.mt_col}")
        self.prod = aie.lock(self.mt, lock_id = 0, init = 2)
        self.cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.mid = aie.lock(self.mt, lock_id = 2, init = 0)     # debug tap: the raw round leaves after the GLU read, then the buffer is free
        self.__call__()

    @property
    def y_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def y_out(self) -> dma_port:
        return self.MM2S[0]

    @property
    def tap_out(self) -> dma_port:
        return self.MM2S[1]

    def __call__(self):
        n = self.n
        tap = self.tap_pkt_id is not None
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.prod, 1)
                aie.dma_bd(self.buf_0, offset = 0, len = n)
                lock_release(self.cons, 1)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(self.prod, 1)
                aie.dma_bd(self.buf_1, offset = 0, len = n)
                lock_release(self.cons, 1)
                aie.next_bd(block[1])
            with block[3]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[4], chain = block[6] if tap else block[8])
            for k, buf in ((4, self.buf_0), (5, self.buf_1)):
                with block[k]:
                    lock_acquire(self.cons, 1)
                    # raw [b][t][32] -> [half][t][cb = 8][32]:  address = half * 2048 + t * 32 + cb * 256 + c32   (b = 8 half + cb)
                    aie.dma_bd(buf, offset = 0, len = n, dimensions = [(2, 8 * 16 * 16), (KL, 32), (8, 32 * KL), (32, 1)])
                    lock_release(self.mid if tap else self.prod, 1)
                    aie.next_bd(block[4 + (k - 4 + 1) % 2])
            with block[6]:
                if tap:
                    aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[7], chain = block[8])
                else:
                    aie.EndOp()
            if tap:
                with block[7]:
                    lock_acquire(self.mid, 1)
                    aie.dma_bd_packet(0, self.tap_pkt_id)
                    aie.dma_bd(self.buf_0, offset = 0, len = n)
                    lock_release(self.prod, 1)
                    aie.next_bd(block[9])
                with block[9]:
                    lock_acquire(self.mid, 1)
                    aie.dma_bd_packet(0, self.tap_pkt_id)
                    aie.dma_bd(self.buf_1, offset = 0, len = n)
                    lock_release(self.prod, 1)
                    aie.next_bd(block[7])
            with block[8]:
                aie.EndOp()


class y_rows_mt(AIE_MT_IP):
    """The ROUNDS rounds of an o / down projection (raw [b][t][32] packets, KL * 512 elements each) -> KL token rows [t][ROUNDS * 512] as one packet
    to the norm tile.  Landing: ROUNDS BDs in a cycle at offsets r * round; read: one 4-dim BD  address = t * 32 + r * round + b * 32 * KL + c32."""
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, rounds: int, out_pkt_id: int, tap_pkt_id: int = None):
        super().__init__(tile)
        self.rounds = rounds
        self.out_pkt_id = out_pkt_id
        self.tap_pkt_id = tap_pkt_id
        self.round = KL * 16 * model_config.q4nx_row_block_size
        self.buf_ty = np.ndarray[(rounds * self.round,), np.dtype[bfloat16]]
        self.buf = ctx.buffer(tile = self.located_tile, dtype = self.buf_ty, name = f"y_rows_{self.mt_row}_{self.mt_col}")
        # one credit per round slot: the landing BD of round r takes a producer credit and gives a consumer credit, the read waits for all R
        self.prod = aie.lock(self.mt, lock_id = 0, init = rounds)
        self.cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.mid = aie.lock(self.mt, lock_id = 2, init = 0)     # debug tap: the raw rounds leave after the norm-tile read, then the buffer is free
        self.__call__()

    @property
    def y_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def y_out(self) -> dma_port:
        return self.MM2S[0]

    @property
    def tap_out(self) -> dma_port:
        return self.MM2S[1]

    def __call__(self):
        R, rnd = self.rounds, self.round
        tap = self.tap_pkt_id is not None
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[R + 1])
            for r in range(R):
                with block[1 + r]:
                    lock_acquire(self.prod, 1)
                    aie.dma_bd(self.buf, offset = r * rnd, len = rnd)
                    lock_release(self.cons, 1)
                    aie.next_bd(block[1 + (r + 1) % R])
            with block[R + 1]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[R + 2], chain = block[R + 3] if tap else block[R + 6])
            with block[R + 2]:
                lock_acquire(self.cons, R)
                aie.dma_bd_packet(0, self.out_pkt_id)
                aie.dma_bd(self.buf, offset = 0, len = R * rnd, dimensions = [(KL, 32), (R, rnd), (16, 32 * KL), (32, 1)])
                lock_release(self.mid if tap else self.prod, R if not tap else 1)
                aie.next_bd(block[R + 2])
            with block[R + 3]:
                if tap:
                    aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[R + 4], chain = block[R + 6])
                else:
                    aie.EndOp()
            if tap:
                with block[R + 4]:
                    lock_acquire(self.mid, 1)
                    aie.dma_bd_packet(0, self.tap_pkt_id)
                    aie.dma_bd(self.buf, offset = 0, len = R * rnd)
                    lock_release(self.prod, R)
                    aie.next_bd(block[R + 4])
            with block[R + 6]:
                aie.EndOp()


class proj_x_replay_mt_glu(proj_x_replay_mt):
    """proj_x_replay_mt whose phases may be fed in rounds: a phase with `glu_rounds = n` lands n packets of `chunk` elements back to back
    (BD r at offset r * chunk, linear) instead of one packet of KL * x_dim, and its replay BD reads the landed buffer as
    [slice j][t][256] (the GLU hands over [half][t][256] chunks) rather than [t][x_dim]."""

    def __call__(self):
        plan, n = [], 0
        for st in self.stages.values():
            plan.append(("land", aie.DMAChannelDir.S2MM, st["s2mm_ch"], st, n))
            fed = [getattr(p, "glu_rounds", 0) for p in st["phases"]]
            n += 1 + sum(f if f else 1 for f in fed)
        for g in self.replays.values():
            plan.append(("replay", aie.DMAChannelDir.MM2S, g["mm2s_ch"], g, n))
            n += 1 + g["fanout"]
        end = n
        # every BD needs both an acquire and a release: the BDs of a multi-BD landing phase pass one token along a private lock
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
                    # flat list of (phase index, BD number inside the phase, BD count of the phase)
                    flat = []
                    for k, p in enumerate(ph):
                        c = getattr(p, "glu_rounds", 0) or 1
                        flat += [(k, j, c) for j in range(c)]
                    for idx, (k, j, c) in enumerate(flat):
                        p = ph[k]
                        with block[start + 1 + idx]:
                            lock_acquire(arg["prod_lock"] if j == 0 else chain_lock, ph[k - 1].credits if j == 0 else 1)
                            if getattr(p, "glu_rounds", 0):
                                chunk = KL * p.x_dim // p.glu_rounds
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
                                # landed as [slice j][t][256]: per 256-column slice [kt = 32][t][k8 = 8]
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
