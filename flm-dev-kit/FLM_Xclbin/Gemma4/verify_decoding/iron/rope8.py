# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""8-token RoPE tiles of the verify design (plan S6.5).

One tile per head width: `RoPE8` (global layers, DH 512, CT[2][3]) and `SWA_RoPE8` (sliding layers, DH 256, CT[3][3]).  Same tile
roles and kernel names as the decode RoPE/SWA_RoPE (iron/RoPE.py, kernels/rope.cc), rewritten for L_TOKENS rows:

  S2MM0  qkv rows, one head per BD, [L_TOKENS][DH] token-major (q heads, then k heads, then v heads)
  S2MM1  rope table  [ (cos(DH/2) | sin(DH/2)) x L_TOKENS | q_norm(DH) | k_norm(DH) ]   (one BD per layer dispatch)
  MM2S0  q, one head per buffer fill, [L_TOKENS][DH]  (-> q park, heads head-major)
  MM2S1  k0, k1, v0, v1, each [L_TOKENS][DH], through one reusable buffer (-> the KV append on the shim); nothing on skip layers

L1 (global): qkv 2 x 8 KB + q 8 KB + kv 8 KB + table 10 KB = 42 KB + stack.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

from constants import L_TOKENS
from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP


class _RoPE8(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    rtp_name = "RTP_RoPE_SKIP_KV"

    def __init__(self, tile: tile_with_location, ctx: mlir_context, DH: int, DQ: int, DK: int, DV: int,
                 stack_size: int, link_with: str, func_name: str, rtp_name: str):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        assert L_TOKENS > 1, "RoPE8 is the verify design's tile; the decode RoPE is iron/RoPE.py"
        comment_print(f"Initializing {func_name} (8-token) at ({self.ct_row}, {self.ct_col}), DH {DH}, L_TOKENS {L_TOKENS}")
        self.DH, self.DQ, self.DK, self.DV = DH, DQ, DK, DV
        self.rows_ty = np.ndarray[(L_TOKENS * DH,), self.dtype]
        self.table_ty = np.ndarray[(L_TOKENS * DH + 2 * DH,), self.dtype]

        r, c = self.ct_row, self.ct_col
        self.qkv_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.rows_ty, name = f"qkv_buffer_0_{r}_{c}")
        self.qkv_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.rows_ty, name = f"qkv_buffer_1_{r}_{c}")
        self.q_buffer = ctx.buffer(tile = self.located_tile, dtype = self.rows_ty, name = f"q_buffer_{r}_{c}")
        self.kv_buffer = ctx.buffer(tile = self.located_tile, dtype = self.rows_ty, name = f"kv_buffer_{r}_{c}")
        self.rope_buffer = ctx.buffer(tile = self.located_tile, dtype = self.table_ty, name = f"rope_buffer_{r}_{c}")
        self.qkv_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.qkv_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.q_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.q_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.kv_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.kv_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.rope_prod_lock = aie.lock(self.ct, lock_id = 8, init = 1)
        self.rope_cons_lock = aie.lock(self.ct, lock_id = 9, init = 0)

        self.SKIP_KV = ctx.buffer(f"{rtp_name}_{r}_{c}", self.located_tile, ctx.data_tys["RTP_ty"], use_write_rtp = True)
        self.rope = ctx.func(func_name, [self.rows_ty, self.rows_ty, self.rows_ty, self.rows_ty, self.table_ty, ctx.data_tys["RTP_ty"]])
        self.func_name = func_name
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.rope(self.q_buffer, self.kv_buffer, self.qkv_buffer_0, self.qkv_buffer_1, self.rope_buffer, self.SKIP_KV)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.qkv_prod_lock, 1)
                aie.dma_bd(self.qkv_buffer_0)
                lock_release(self.qkv_cons_lock, 1)
                aie.next_bd(block[100])

            with block[100]:
                lock_acquire(self.qkv_prod_lock, 1)
                aie.dma_bd(self.qkv_buffer_1)
                lock_release(self.qkv_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[4])

            with block[3]:
                lock_acquire(self.q_cons_lock, 1)
                aie.dma_bd(self.q_buffer)
                lock_release(self.q_prod_lock, 1)
                aie.next_bd(block[3])

            with block[4]:
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[5], chain = block[9])

            with block[5]:
                lock_acquire(self.kv_cons_lock, 1)
                aie.dma_bd(self.kv_buffer)
                lock_release(self.kv_prod_lock, 1)
                aie.next_bd(block[5])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[10], chain = block[11])

            with block[10]:
                lock_acquire(self.rope_prod_lock, 1)
                aie.dma_bd(self.rope_buffer)
                lock_release(self.rope_cons_lock, 1)
                aie.next_bd(block[10])

            with block[11]:
                aie.EndOp()

    @property
    def QKV_in(self):
        return self.S2MM[0]

    @property
    def ROPE_in(self):
        return self.S2MM[1]

    @property
    def Q_out(self):
        return self.MM2S[0]

    @property
    def KV_out(self):
        return self.MM2S[1]


class RoPE8(_RoPE8):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024 * 4, link_with: str = "rope.o"):
        super().__init__(tile, ctx, model_config.DH, model_config.DQ, model_config.DK, model_config.DV, stack_size, link_with,
                         "rope", "RTP_RoPE_SKIP_KV")


class SWA_RoPE8(_RoPE8):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024 * 4, link_with: str = "swa_rope.o"):
        super().__init__(tile, ctx, model_config.SWA_DH, model_config.SWA_DQ, model_config.SWA_DK, model_config.SWA_DV, stack_size, link_with,
                         "swa_rope", "RTP_SWA_RoPE_SKIP_KV")
