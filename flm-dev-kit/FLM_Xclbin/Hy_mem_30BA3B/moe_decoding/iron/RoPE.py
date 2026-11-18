# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
import sys
import os
from typing import Any
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP
from constants import ROPE_Q_PKT_ID_BASE

class RoPE(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "rope.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing RoPE at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.DQ = model_config.DQ
        self.DK = model_config.DK
        self.DV = model_config.DV
        self.num_attn_cus = model_config.num_attn_cus
        q_heads_per_group = model_config.num_heads // model_config.num_kv_heads
        self.DQ_PADDED = (q_heads_per_group + model_config.q_heads_padding - 1) // model_config.q_heads_padding * model_config.q_heads_padding * model_config.num_kv_heads * model_config.head_dim
        self.q_in_ty = np.ndarray[(self.DQ,), self.dtype]
        self.kv_in_ty = np.ndarray[(self.DK + self.DV,), self.dtype]
        comment_print(f"DQ: {self.DQ}, DK: {self.DK}, DV: {self.DV}, DQ_PADDED: {self.DQ_PADDED}")
        self.q_ty = np.ndarray[(self.DQ_PADDED,), self.dtype]
        self.k_ty = np.ndarray[(self.DK,), self.dtype]
        self.v_ty = np.ndarray[(self.DV,), self.dtype]
        rope_size = model_config.ROPE_DIM # 64, half cos, half sin
        
        if model_config.has_qk_norm:
            rope_size += model_config.DH * 2
        if model_config.has_attn_bias:
            rope_size += (model_config.DQ + model_config.DK + model_config.DV)

        self.rope_ty = np.ndarray[(rope_size,), self.dtype]

        comment_print(f"\tRoPE DQ: {self.DQ}, DK: {self.DK}, DV: {self.DV}, DQ_PADDED: {self.DQ_PADDED}, rope_size: {rope_size}")

        self.q_in_buffer = ctx.buffer(tile = self.located_tile, dtype = self.q_in_ty, name = f"q_in_buffer_{self.ct_row}_{self.ct_col}")
        self.kv_in_buffer = ctx.buffer(tile = self.located_tile, dtype = self.kv_in_ty, name = f"kv_in_buffer_{self.ct_row}_{self.ct_col}")
        self.q_buffer = ctx.buffer(tile = self.located_tile, dtype = self.q_ty, name = f"q_buffer_{self.ct_row}_{self.ct_col}")
        self.k_buffer = ctx.buffer(tile = self.located_tile, dtype = self.k_ty, name = f"k_buffer_{self.ct_row}_{self.ct_col}")
        self.v_buffer = ctx.buffer(tile = self.located_tile, dtype = self.v_ty, name = f"v_buffer_{self.ct_row}_{self.ct_col}")
        self.rope_buffer = ctx.buffer(tile = self.located_tile, dtype = self.rope_ty,  name = f"rope_buffer_{self.ct_row}_{self.ct_col}")
        self.qkv_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.qkv_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.q_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.q_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.k_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.k_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.v_prod_lock = aie.lock(self.ct, lock_id = 6, init = 1)
        self.v_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.rope_prod_lock = aie.lock(self.ct, lock_id = 8, init = 1)
        self.rope_cons_lock = aie.lock(self.ct, lock_id = 9, init = 0)
        self.q_pass_lock = aie.lock(self.ct, lock_id = 10, init = 0)

        self.rope = ctx.func(link_with[:-2], [self.q_ty, self.k_ty, self.v_ty, self.q_in_ty, self.kv_in_ty, self.rope_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.rope(self.q_buffer, self.k_buffer, self.v_buffer, self.q_in_buffer, self.kv_in_buffer, self.rope_buffer)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.qkv_prod_lock, 1)
                aie.dma_bd(self.kv_in_buffer)
                lock_release(self.qkv_cons_lock, 1)
                aie.next_bd(block[100])

            with block[100]:
                lock_acquire(self.qkv_prod_lock, 1)
                aie.dma_bd(self.q_in_buffer)
                lock_release(self.qkv_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[20], chain = block[4])

            # Q goes out in num_attn_cus equal parts, part i with packet id
            # ROPE_Q_PKT_ID_BASE + i (see constants.py)
            # (one KV-head group per attention CU). The parts run as one BD chain;
            # q_pass_lock hands the turn from one part to the next, the first part
            # waits for the kernel and the last one gives the buffer back.
            n = self.num_attn_cus
            q_part = self.DQ_PADDED // n
            for i in range(n):
                acq_lock = self.q_cons_lock if i == 0 else self.q_pass_lock
                rel_lock = self.q_prod_lock if i == n - 1 else self.q_pass_lock
                with block[20 + i]:
                    lock_acquire(acq_lock, 1)
                    aie.dma_bd_packet(0, ROPE_Q_PKT_ID_BASE + i)
                    aie.dma_bd(self.q_buffer, offset=i * q_part, len=q_part)
                    lock_release(rel_lock, 1)
                    aie.next_bd(block[20 + (i + 1) % n])

            with block[4]:
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[5], chain = block[9])

            with block[5]:
                lock_acquire(self.k_cons_lock, 1)
                aie.dma_bd(self.k_buffer)
                lock_release(self.k_prod_lock, 1)
                aie.next_bd(block[6])

            with block[6]:
                lock_acquire(self.v_cons_lock, 1)
                aie.dma_bd(self.v_buffer)
                lock_release(self.v_prod_lock, 1)
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