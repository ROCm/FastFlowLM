# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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
from constants import kv_out_pkt_id

from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP

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
        # Stages the multi-packet k|v sends so their legs leave in column order
        # and the buffer is only released after the last one.
        self.k_pass_lock = aie.lock(self.ct, lock_id = 11, init = 0)
        self.v_pass_lock = aie.lock(self.ct, lock_id = 12, init = 0)

        # **q leaves whole and padded, in one BD.** It used to leave as one
        # packet per CU with the GQA padding stripped, because it was going
        # straight to DDR to be scored against a DQ-wide reference and the host
        # would otherwise have had to de-stride it. The attention CUs exist now
        # and want the opposite: the 8x8x8 mmul indexes every group with one
        # stride, so each CU needs its group's *padded* 2048 elements. The split
        # moved to MT[3] (iron/attn_hub.py), which takes this whole row and
        # hands each CU a contiguous slice -- the padding is interior to each
        # group, so that is four offsets and no striding.
        self.num_attn_cus_total = model_config.num_kv_heads // model_config.kv_heads_per_cu
        self.q_slice = model_config.q_heads_per_group_padded * model_config.DH
        assert self.q_slice * self.num_attn_cus_total == self.DQ_PADDED, \
            f"q slices {self.q_slice} x {self.num_attn_cus_total} do not cover DQ_PADDED {self.DQ_PADDED}"
        comment_print(f"\tq: whole {self.DQ_PADDED} row to MT[3], which splits it "
                      f"{self.num_attn_cus_total} x {self.q_slice}")

        # k and v are split here, but by *column* rather than by CU: each
        # column's shim appends the heads that column owns, so the append and
        # the replay share an IT and npu_dma_wait can order them. An append
        # landing on IT3 could not be waited on before the replay reads from
        # IT4/IT5, and the CUs would attend to a cache row that had not been
        # written yet. (memory: npu-dma-wait-is-per-shim)
        #
        # The halves are contiguous: the cache row is head-major, so column c
        # owns [c * kv_col_width, + kv_col_width).
        self.num_attn_columns = model_config.num_attn_columns
        self.kv_col_width = model_config.kv_heads_per_column * model_config.DH
        assert self.kv_col_width * self.num_attn_columns == self.DK, \
            f"kv column widths {self.kv_col_width} x {self.num_attn_columns} do not cover DK {self.DK}"
        # k and v carry different ids, so they land on different S2MM channels
        # of the column's shim. See KV_out and iron/constants.py.
        self.k_pkt_base = int(kv_out_pkt_id.k_to_col_0)
        self.v_pkt_base = int(kv_out_pkt_id.v_to_col_0)
        comment_print(f"\tk|v split: {self.num_attn_columns} columns x {self.kv_col_width} = {self.DK}, "
                      f"k pkt ids {self.k_pkt_base}..{self.k_pkt_base + self.num_attn_columns - 1}, "
                      f"v pkt ids {self.v_pkt_base}..{self.v_pkt_base + self.num_attn_columns - 1}")

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
                s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[4])

            # The whole padded row, one BD, circuit-switched to MT[3]. The split
            # into the four CUs' groups happens there -- see the note on q_slice
            # above and iron/attn_hub.py.
            with block[3]:
                lock_acquire(self.q_cons_lock, 1)
                aie.dma_bd(self.q_buffer)
                lock_release(self.q_prod_lock, 1)
                aie.next_bd(block[3])

            with block[4]:
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[5], chain = block[9])

            # k then v, alternating -- but each split by kv head across the two
            # attention columns' shims, and now by *plane* as well: k carries
            # k_pkt_base + col and v carries v_pkt_base + col, so the two land
            # on different S2MM channels of the same shim.
            #
            # **Two splits, for two different reasons.**
            #
            # By column (which shim): a synchronisation fix. See the note on
            # kv_col_width above -- the append has to land on the same shim the
            # replay reads from, or npu_dma_wait cannot order the two and the
            # CUs attend to rows that were never written.
            #
            # By plane (which channel): a layout fix, and what lifts the context
            # ceiling. The stream order here is k then v, so one landing channel
            # forced the host to describe both planes in ONE descriptor -- the
            # column's heads on one axis, "k or v" on another whose stride is
            # the whole k plane, MAX_L * DK. A shim BD's stride field is 20 bits
            # and bf16 strides are halved on lowering, which stopped encoding
            # past MAX_L = 2048; npu_instr_utils only prints and masks, so every
            # v row landed at a wrapped address and only the next pass saw it.
            # On two channels each plane gets its own BD, whose only stride is a
            # row (DK or DV) and whose plane base is a full-width byte offset.
            # Nothing in the append scales with MAX_L any more.
            #
            # The ids sit above the q range rather than restarting at 0. All
            # four flows leave this tile heading east and share switchboxes on
            # the way to columns 4 and 5, and a switchbox routes on the id alone
            # -- it does not know which master port a packet came from.
            with block[5]:
                lock_acquire(self.k_cons_lock, 1)
                aie.dma_bd_packet(0, self.k_pkt_base + 0)
                aie.dma_bd(self.k_buffer, offset = 0, len = self.kv_col_width)
                lock_release(self.k_pass_lock, 1)
                aie.next_bd(block[6])

            with block[6]:
                lock_acquire(self.k_pass_lock, 1)
                aie.dma_bd_packet(0, self.k_pkt_base + 1)
                aie.dma_bd(self.k_buffer, offset = self.kv_col_width, len = self.kv_col_width)
                lock_release(self.k_prod_lock, 1)
                aie.next_bd(block[7])

            with block[7]:
                lock_acquire(self.v_cons_lock, 1)
                aie.dma_bd_packet(0, self.v_pkt_base + 0)
                aie.dma_bd(self.v_buffer, offset = 0, len = self.kv_col_width)
                lock_release(self.v_pass_lock, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.v_pass_lock, 1)
                aie.dma_bd_packet(0, self.v_pkt_base + 1)
                aie.dma_bd(self.v_buffer, offset = self.kv_col_width, len = self.kv_col_width)
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