# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Tiles of the 8-token PLE stage (plan S6.8).

Per layer the per-layer-input epilogue is, for 8 tokens t:   gate = gelu(W_gate . h_t) * pli_t ;  up = W_up . gate ;
out_t = (h_t + rmsnorm(up) * w_post_pli) * layer_scalar.   pli (= per layer input, depends only on token ids) is computed on the host.

  gate tile G (4,2): streams W_gate, 3-buffer x ring [8][256], emits u = gelu(gate) * pli  [8][256]
  up   tile U (5,2): streams W_up, emits 80 packets y[8][32]
  stage MT5        : lands the 80 packets as [rb][t][32] (one 40 KB BD) and reads them back token-major (one 3-dim BD)
  norm tile N (5,3): rms + residual + layer scalar per token, out rows [2560]

Everything is circuit-switched flows with hand-written BD chains, as in the decode design.  The kernels are kernels/ple8_*.cc.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *

from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_CT_IP, AIE_MT_IP, comment_print

T = 8            # token rows of the stage
BF = np.dtype[bfloat16]


def _ty(n):
    return np.ndarray[(n,), BF]


class ple8_gate_tile(AIE_CT_IP):
    def __init__(self, ct: tile_with_location, ctx: mlir_context, D: int, pli_d: int, stack_size: int = 4096, link_with: str = "ple8_gate.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)
        self.D, self.pli_d = D, pli_d
        x_ty = _ty(T * 256)
        w_ty = _ty(32 * 256)
        u_ty = _ty(T * pli_d)
        n = f"{self.ct_row}_{self.ct_col}"
        self.x = [aie.buffer(tile = self.ct, datatype = x_ty, name = f"ple8_x{i}_{n}") for i in range(3)]
        self.w = [aie.buffer(tile = self.ct, datatype = w_ty, name = f"ple8_w{i}_{n}") for i in range(2)]
        self.pli = aie.buffer(tile = self.ct, datatype = u_ty, name = f"ple8_pli_{n}")
        self.u = aie.buffer(tile = self.ct, datatype = u_ty, name = f"ple8_u_{n}")
        self.x_prod = aie.lock(self.ct, lock_id = 0, init = 3)
        self.x_cons = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod = aie.lock(self.ct, lock_id = 2, init = 2)
        self.w_cons = aie.lock(self.ct, lock_id = 3, init = 0)
        self.u_prod = aie.lock(self.ct, lock_id = 4, init = 1)
        self.u_cons = aie.lock(self.ct, lock_id = 5, init = 0)
        self.func = ctx.func("ple8_gate", [x_ty, x_ty, x_ty, w_ty, w_ty, u_ty, u_ty])
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.func(self.x[0], self.x[1], self.x[2], self.w[0], self.w[1], self.pli, self.u)

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[4])
            for i in range(3):
                with block[1 + i]:
                    lock_acquire(self.x_prod, 1)
                    aie.dma_bd(self.x[i])
                    lock_release(self.x_cons, 1)
                    aie.next_bd(block[1 + (i + 1) % 3])
            with block[4]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[5], chain = block[7])
            for i in range(2):
                with block[5 + i]:
                    lock_acquire(self.w_prod, 1)
                    aie.dma_bd(self.w[i])
                    lock_release(self.w_cons, 1)
                    aie.next_bd(block[5 + (i + 1) % 2])
            with block[7]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[8], chain = block[9])
            with block[8]:
                lock_acquire(self.u_cons, 1)
                aie.dma_bd(self.u)
                lock_release(self.u_prod, 1)
                aie.next_bd(block[8])
            with block[9]:
                aie.EndOp()

    @property
    def X_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def W_in(self) -> dma_port:
        return self.S2MM[1]

    @property
    def U_out(self) -> dma_port:
        return self.MM2S[0]


class ple8_up_tile(AIE_CT_IP):
    def __init__(self, ct: tile_with_location, ctx: mlir_context, pli_d: int, stack_size: int = 4096, link_with: str = "ple8_up.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)
        u_ty = _ty(T * pli_d)
        w_ty = _ty(32 * 256)
        y_ty = _ty(T * 32)
        n = f"{self.ct_row}_{self.ct_col}"
        self.u = aie.buffer(tile = self.ct, datatype = u_ty, name = f"ple8_u_{n}")
        self.w = [aie.buffer(tile = self.ct, datatype = w_ty, name = f"ple8_w{i}_{n}") for i in range(2)]
        self.y = [aie.buffer(tile = self.ct, datatype = y_ty, name = f"ple8_y{i}_{n}") for i in range(2)]
        self.u_prod = aie.lock(self.ct, lock_id = 0, init = 1)
        self.u_cons = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod = aie.lock(self.ct, lock_id = 2, init = 2)
        self.w_cons = aie.lock(self.ct, lock_id = 3, init = 0)
        self.y_prod = aie.lock(self.ct, lock_id = 4, init = 2)
        self.y_cons = aie.lock(self.ct, lock_id = 5, init = 0)
        self.func = ctx.func("ple8_up", [u_ty, w_ty, w_ty, y_ty, y_ty])
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.func(self.u, self.w[0], self.w[1], self.y[0], self.y[1])

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[2])
            with block[1]:
                lock_acquire(self.u_prod, 1)
                aie.dma_bd(self.u)
                lock_release(self.u_cons, 1)
                aie.next_bd(block[1])
            with block[2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[3], chain = block[5])
            for i in range(2):
                with block[3 + i]:
                    lock_acquire(self.w_prod, 1)
                    aie.dma_bd(self.w[i])
                    lock_release(self.w_cons, 1)
                    aie.next_bd(block[3 + (i + 1) % 2])
            with block[5]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[6], chain = block[8])
            for i in range(2):
                with block[6 + i]:
                    lock_acquire(self.y_cons, 1)
                    aie.dma_bd(self.y[i])
                    lock_release(self.y_prod, 1)
                    aie.next_bd(block[6 + (i + 1) % 2])
            with block[8]:
                aie.EndOp()

    @property
    def U_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def W_in(self) -> dma_port:
        return self.S2MM[1]

    @property
    def Y_out(self) -> dma_port:
        return self.MM2S[0]


class ple8_norm_tile(AIE_CT_IP):
    def __init__(self, ct: tile_with_location, ctx: mlir_context, D: int, stack_size: int = 4096, link_with: str = "ple8_norm.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)
        self.D = D
        row_ty = _ty(D)
        wc_ty = _ty(D + 32)
        n = f"{self.ct_row}_{self.ct_col}"
        self.wc = aie.buffer(tile = self.ct, datatype = wc_ty, name = f"ple8_wc_{n}")
        self.up = [aie.buffer(tile = self.ct, datatype = row_ty, name = f"ple8_up{i}_{n}") for i in range(2)]
        self.h = [aie.buffer(tile = self.ct, datatype = row_ty, name = f"ple8_h{i}_{n}") for i in range(2)]
        self.o = [aie.buffer(tile = self.ct, datatype = row_ty, name = f"ple8_o{i}_{n}") for i in range(2)]
        self.w_prod = aie.lock(self.ct, lock_id = 0, init = 1)
        self.w_cons = aie.lock(self.ct, lock_id = 1, init = 0)
        self.up_prod = aie.lock(self.ct, lock_id = 2, init = 2)
        self.up_cons = aie.lock(self.ct, lock_id = 3, init = 0)
        self.h_prod = aie.lock(self.ct, lock_id = 4, init = 2)
        self.h_cons = aie.lock(self.ct, lock_id = 5, init = 0)
        self.o_prod = aie.lock(self.ct, lock_id = 6, init = 2)
        self.o_cons = aie.lock(self.ct, lock_id = 7, init = 0)
        self.func = ctx.func("ple8_norm", [wc_ty, row_ty, row_ty, row_ty, row_ty, row_ty, row_ty])
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.func(self.wc, self.up[0], self.up[1], self.h[0], self.h[1], self.o[0], self.o[1])

        @aie.mem(self.ct)
        def mem_body(block):
            # up rows from the memtile stage
            aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])
            for i in range(2):
                with block[1 + i]:
                    lock_acquire(self.up_prod, 1)
                    aie.dma_bd(self.up[i])
                    lock_release(self.up_cons, 1)
                    aie.next_bd(block[1 + (i + 1) % 2])
            # [w_post_pli | layer_scalar] once, then the 8 h rows: a 9 BD loop that restarts at every dispatch
            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[4], chain = block[13])
            with block[4]:
                lock_acquire(self.w_prod, 1)
                aie.dma_bd(self.wc)
                lock_release(self.w_cons, 1)
                aie.next_bd(block[5])
            for t in range(T):
                with block[5 + t]:
                    lock_acquire(self.h_prod, 1)
                    aie.dma_bd(self.h[t % 2])
                    lock_release(self.h_cons, 1)
                    aie.next_bd(block[5 + t + 1] if t < T - 1 else block[4])
            with block[13]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[14], chain = block[16])
            for i in range(2):
                with block[14 + i]:
                    lock_acquire(self.o_cons, 1)
                    aie.dma_bd(self.o[i])
                    lock_release(self.o_prod, 1)
                    aie.next_bd(block[14 + (i + 1) % 2])
            with block[16]:
                aie.EndOp()

    @property
    def Up_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def HW_in(self) -> dma_port:
        return self.S2MM[1]

    @property
    def O_out(self) -> dma_port:
        return self.MM2S[0]


class ple8_stage_mt(AIE_MT_IP):
    """Collects the 80 up packets y[8][32] as [rb][t][32] and reads them back token-major [t][rb][32] (read-side permute: memtile
    S2MM permutes hang, MM2S ones work, see iron/proj_tile.py proj_x_replay_mt)."""
    def __init__(self, tile: tile_with_location, ctx: mlir_context, D: int):
        super().__init__(tile)
        self.D = D
        self.rbs = D // 32
        self.n = self.rbs * T * 32
        self.buf = aie.buffer(tile = self.mt, datatype = _ty(self.n), name = f"ple8_stage_{self.mt_row}_{self.mt_col}")
        self.prod = aie.lock(self.mt, lock_id = 0, init = 1)
        self.cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.__call__()

    @property
    def y_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def up_out(self) -> dma_port:
        return self.MM2S[0]

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])
            with block[1]:
                lock_acquire(self.prod, 1)
                aie.dma_bd(self.buf, offset = 0, len = self.n)
                lock_release(self.cons, 1)
                aie.next_bd(block[1])
            with block[2]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[5])
            with block[3]:
                lock_acquire(self.cons, 1)
                # [rb][t][32] -> [t][rb][32]
                aie.dma_bd(self.buf, offset = 0, len = self.n, dimensions = [(T, 32), (self.rbs, T * 32), (32, 1)])
                lock_release(self.prod, 1)
                aie.next_bd(block[3])
            with block[5]:
                aie.EndOp()
