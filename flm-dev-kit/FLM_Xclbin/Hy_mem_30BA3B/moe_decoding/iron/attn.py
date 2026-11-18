# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig

from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

from utils import connect_port, mlir_context, tile_with_location
from utils import AIE_CT_IP, AIE_MT_IP, comment_print, ShimTile

class qk_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, shared_s_buffers: List[Any], s_ty: Any,  model_config: ModelConfig, stack_size: int = 1024, link_with: str = "attn_qk.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing attn_qk_tile at ({self.ct_row}, {self.ct_col})")
        self.Q_HEAD_PER_ROUND = model_config.Q_HEAD_PER_ROUND_PADDED
        self.KV_HEAD_PER_ROUND = model_config.kv_head_per_cu
        self.DH = model_config.DH
        self.lc = model_config.lq

        self.attn_q_ty = np.ndarray[(self.Q_HEAD_PER_ROUND * self.DH,), self.dtype]
        self.attn_k_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_ROUND * self.DH), self.dtype]
        comment_print(f"attn_q_ty shape: ({self.Q_HEAD_PER_ROUND * self.DH},), attn_k_ty shape: ({self.lc}, {self.KV_HEAD_PER_ROUND * self.DH})")
        self.attn_s_ty = s_ty
        self.L_ty = ctx.data_tys["L_ty"]

        self.attn_qk = ctx.func("attn_qk", [self.attn_q_ty, self.attn_q_ty, self.attn_k_ty, self.attn_k_ty, self.attn_s_ty, self.attn_s_ty, self.L_ty])

        self.k_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_k_ty, name = f"k_0_{self.ct_row}_{self.ct_col}")
        self.k_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_k_ty, name = f"k_1_{self.ct_row}_{self.ct_col}")

        self.q_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_q_ty, name = f"q_0_{self.ct_row}_{self.ct_col}")
        self.q_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_q_ty, name = f"q_1_{self.ct_row}_{self.ct_col}")
        self.q_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.q_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.k_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.k_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.s_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.s_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.l_prod_lock = aie.lock(self.ct, lock_id = 6, init = 2)
        self.l_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.s_buffer_0 = shared_s_buffers[0]
        self.s_buffer_1 = shared_s_buffers[1]

        self.L_buffer = ctx.buffer(f"RTP_L_attn_qk_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.L_ty, use_write_rtp = True)
        self.q_heads_len = self.Q_HEAD_PER_ROUND * self.DH
        self.q_in_order = [(self.Q_HEAD_PER_ROUND, 8), (self.DH // 8, self.Q_HEAD_PER_ROUND * 8), (8, 1)]

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.attn_qk(self.q_buffer_0, self.q_buffer_1, self.k_buffer_0, self.k_buffer_1, self.s_buffer_0, self.s_buffer_1, self.L_buffer)

            
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

            with block[1]:
                aie.use_lock(self.q_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.q_buffer_0, offset = 0, len = self.q_heads_len, dimensions = self.q_in_order)
                aie.use_lock(self.q_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[2])

            with block[2]:
                aie.use_lock(self.q_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.q_buffer_1, offset = 0, len = self.q_heads_len, dimensions = self.q_in_order)
                aie.use_lock(self.q_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
            
            with block[4]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.k_buffer_0)
                aie.use_lock(self.k_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[5])

            with block[5]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.k_buffer_1)
                aie.use_lock(self.k_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[4])

            with block[6]:
                aie.EndOp()
    
    @property
    def Q_in(self):
        return self.S2MM[0]
    
    @property
    def K_in(self):
        return self.S2MM[1]

class kv_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, s_ty: Any, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "attn_kv.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing attn_kv_tile at ({self.ct_row}, {self.ct_col})")
        self.NQ_PER_CU = model_config.Q_HEAD_PER_ROUND
        self.DH = model_config.DH
        self.KV_HEAD_PER_ROUND = model_config.kv_head_per_cu
        self.lc = model_config.lk

        self.attn_v_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_ROUND * self.DH), self.dtype]
        self.attn_o_ty = np.ndarray[(self.DH * self.NQ_PER_CU, ), self.dtype]
        comment_print(f"attn_v_ty shape: ({self.lc}, {self.KV_HEAD_PER_ROUND * self.DH}), attn_o_ty shape: {(self.DH * self.NQ_PER_CU, )}")
        self.attn_s_ty = s_ty
        self.L_ty = ctx.data_tys["L_ty"]
        self.s_ty = s_ty
        self.attn_kv = ctx.func("attn_kv", [self.attn_s_ty, self.attn_s_ty, self.attn_v_ty, self.attn_v_ty, self.attn_o_ty, self.attn_o_ty, self.L_ty])

        self.L_buffer = ctx.buffer(f"RTP_L_attn_kv_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.L_ty, use_write_rtp = True)
        self.s_buffer_0 = ctx.buffer(f"s_shared_buffer_0_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty)
        self.s_buffer_1 = ctx.buffer(f"s_shared_buffer_1_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty)
        self.v_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_0_{self.ct_row}_{self.ct_col}")
        self.v_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_1_{self.ct_row}_{self.ct_col}")

        self.o_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_o_ty, name = f"o_0_{self.ct_row}_{self.ct_col}")
        self.o_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_o_ty, name = f"o_1_{self.ct_row}_{self.ct_col}")

        self.o_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.o_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.v_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.v_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.s_proj_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.s_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        self.__call__()
    
    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.attn_kv(self.s_buffer_0, self.s_buffer_1, self.v_buffer_0, self.v_buffer_1, self.o_buffer_0, self.o_buffer_1, self.L_buffer)

        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[1], chain=block[3])
            
            with block[1]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_0)
                aie.use_lock(self.v_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[2])
                
            with block[2]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_1)
                aie.use_lock(self.v_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

            with block[4]:
                aie.use_lock(self.o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.o_buffer_0, offset = 0, len = self.NQ_PER_CU * self.DH, dimensions = [(self.NQ_PER_CU, 8), (self.DH // 8, self.NQ_PER_CU * 8), (8, 1)])
                aie.use_lock(self.o_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[5])

            with block[5]:
                aie.use_lock(self.o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.o_buffer_1, offset = 0, len = self.NQ_PER_CU * self.DH, dimensions = [(self.NQ_PER_CU, 8), (self.DH // 8, self.NQ_PER_CU * 8), (8, 1)])
                aie.use_lock(self.o_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[4])
                
            with block[6]:
                aie.EndOp()
    
    @property
    def V_in(self):
        return self.S2MM[1]
    
    @property
    def O_out(self):
        return self.MM2S[0]
    
    @property
    def get_s_buffers(self):
        return [self.s_buffer_0, self.s_buffer_1]

class attn_cu:
    def __init__(self, 
        attn_qk_tile: tile_with_location, 
        attn_kv_tile: tile_with_location, 
        model_config: ModelConfig,
        ctx: mlir_context, 
        qk_stack_size: int, 
        kv_stack_size: int, 
        qk_link_with: str, 
        kv_link_with: str
    ):  
        comment_print(f"Initializing attn_cu at QK({attn_qk_tile.row}, {attn_qk_tile.col}) KV({attn_kv_tile.row}, {attn_kv_tile.col})")
        self.NQ_PER_CU = model_config.Q_HEAD_PER_ROUND_PADDED
        self.s_ty = np.ndarray[(self.NQ_PER_CU * 16 + 16, 1), np.dtype[bfloat16]] # for attention score and correction
        comment_print(f"\tattn_s_ty shape: {(self.NQ_PER_CU * 16 + 16, 1)}")
        self.kv_core = kv_tile(attn_kv_tile, ctx, self.s_ty, model_config, stack_size = kv_stack_size, link_with = kv_link_with)
        self.qk_core = qk_tile(attn_qk_tile, ctx, self.kv_core.get_s_buffers, self.s_ty, model_config, stack_size = qk_stack_size, link_with = qk_link_with)
    
    @property
    def Q_in(self):
        return self.qk_core.Q_in
    
    @property
    def K_in(self):
        return self.qk_core.K_in
    
    @property
    def V_in(self):
        return self.kv_core.V_in
    
    @property
    def O_out(self):
        return self.kv_core.O_out

class attn_memtile(AIE_MT_IP):
    """Mem tile of one attention column.

    K/V: takes the K and V cache of this column's KV heads from the shim, in
    chunks of lc rows, and splits each chunk between the column's 2 CUs (one
    head each).

    O: joins o_parts inputs, in order, into one buffer and sends the result out
    o_sends times (packet id o_pkt_id, or a plain flow if None). With tap=True
    it is also sent once more on MM2S 5, after those, for checking on the host.

    Channels:  S2MM 0 K, 1 V, 2.. O parts      MM2S 0/1 K, 2/3 V, 4 O, 5 tap
    """

    def __init__(self, tile, mctx: mlir_context, model_config: ModelConfig, kv_heads: int,
                 o_parts: List[int], o_sends: int, o_pkt_id: int = None, tap: bool = False):
        super().__init__(tile)
        self.num_kv_heads = kv_heads
        self.num_kv_heads_per_round = model_config.kv_head_per_cu
        self.DH = model_config.DH
        self.lc = model_config.lk
        assert 2 <= len(o_parts) <= 4, "O parts come in on S2MM 2..5"
        self.o_parts = o_parts
        self.o_size = sum(o_parts)
        self.o_sends = o_sends
        self.o_pkt_id = o_pkt_id
        self.tap = tap

        self.k_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.v_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.o_ty     = np.ndarray[(self.o_size, ), np.dtype[bfloat16]]
        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}) with lc: {self.lc}, d: { self.num_kv_heads * self.DH }, o_parts: {o_parts}")

        self.k_buffer_0 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.k_buffer_1 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.v_buffer_0 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.v_buffer_1 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.o_buffer = aie.buffer(self.mt, self.o_ty, name = f"o_buffer_{self.mt_row}_{self.mt_col}")

        self.k_prod_lock = aie.lock(self.mt, lock_id = 0, init = 2)
        self.k_cons_lock_0 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.k_cons_lock_1 = aie.lock(self.mt, lock_id = 2, init = 0)
        self.v_prod_lock = aie.lock(self.mt, lock_id = 3, init = 2)
        self.v_cons_lock_0 = aie.lock(self.mt, lock_id = 4, init = 0)
        self.v_cons_lock_1 = aie.lock(self.mt, lock_id = 5, init = 0)

        # O locks, a chain: part i waits on o_part_locks[i] and passes the turn
        # to part i + 1; the last part hands the full buffer to the senders.
        # Trick: the first lock starts at the number of sends (not 1). Filling
        # takes all counts and each send gives one back, so O is only refilled
        # after its last send has left.
        # With the tap, the o_sends main sends hand their counts to tap_lock
        # instead, and the tap (one send, after all of them) gives all of them
        # back at once: the tap always gets exactly one copy, the main output
        # exactly o_sends, and they never compete for the same count.
        self.o_part_locks = [aie.lock(self.mt, lock_id = 6 + i, init = self.o_sends if i == 0 else 0)
                             for i in range(len(o_parts))]
        self.o_cons_lock = aie.lock(self.mt, lock_id = 6 + len(o_parts), init = 0)
        if self.tap:
            self.tap_lock = aie.lock(self.mt, lock_id = 7 + len(o_parts), init = 0)

        self.k_mem_row_size = self.num_kv_heads * self.DH
        self.k_mem_cu_row_size = self.num_kv_heads_per_round * self.DH
        self.k_order = [(self.k_mem_cu_row_size // 8, 8), (16, self.k_mem_row_size), (8, 1)]
        self.k_offset = self.k_mem_cu_row_size
        comment_print(f"\tattn_memtile k_order: {self.k_order}")
        self.k_len_per_cu = self.k_mem_cu_row_size * self.lc

        self.v_mem_row_size = self.num_kv_heads * self.DH
        self.v_mem_cu_row_size = self.num_kv_heads_per_round * self.DH
        self.v_order = [(self.lc // 8, self.lc // 2 * self.v_mem_row_size), (self.v_mem_cu_row_size // 8, 8), (8, self.v_mem_row_size), (8, 1)]
        comment_print(f"\tattn_memtile v_order: {self.v_order}")
        self.v_len_per_cu = self.v_mem_cu_row_size * self.lc
        self.v_offset = self.v_mem_cu_row_size
        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

            with block[1]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0)
                aie.use_lock(self.k_cons_lock_0, aie.LockAction.Release, value=1)
                aie.next_bd(block[2])

            with block[2]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1)
                aie.use_lock(self.k_cons_lock_0, aie.LockAction.Release, value=1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

            with block[4]:
                aie.use_lock(self.k_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0, offset = 0, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_cons_lock_1, aie.LockAction.Release, value=1)
                aie.next_bd(block[5])

            with block[5]:
                aie.use_lock(self.k_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1, offset = 0, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_cons_lock_1, aie.LockAction.Release, value=1)
                aie.next_bd(block[4])

            with block[6]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[9])

            with block[7]:
                aie.use_lock(self.k_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0, offset = self.k_offset, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[8])

            with block[8]:
                aie.use_lock(self.k_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1, offset = self.k_offset, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[10], chain=block[12])

            with block[10]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.v_buffer_0)
                aie.use_lock(self.v_cons_lock_0, aie.LockAction.Release, value=1)
                aie.next_bd(block[11])

            with block[11]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.v_buffer_1)
                aie.use_lock(self.v_cons_lock_0, aie.LockAction.Release, value=1)
                aie.next_bd(block[10])

            with block[12]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])

            with block[13]:
                aie.use_lock(self.v_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_0, offset = 0, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_cons_lock_1, aie.LockAction.Release, value = 1)
                aie.next_bd(block[14])

            with block[14]:
                aie.use_lock(self.v_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_1, offset = 0, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_cons_lock_1, aie.LockAction.Release, value = 1)
                aie.next_bd(block[13])

            with block[15]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])

            with block[16]:
                aie.use_lock(self.v_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_0, offset = self.v_offset, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[17])

            with block[17]:
                aie.use_lock(self.v_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_1, offset = self.v_offset, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[16])

            # O in: part i on S2MM 2 + i, written to its place in o_buffer
            with block[18]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[30], chain=block[19])
            n_parts = len(self.o_parts)
            offset = 0
            for i, part_len in enumerate(self.o_parts):
                is_last = i == n_parts - 1
                acq_value = self.o_sends if i == 0 else 1
                rel_lock = self.o_cons_lock if is_last else self.o_part_locks[i + 1]
                rel_value = self.o_sends if is_last else 1
                with block[30 + i]:
                    aie.use_lock(self.o_part_locks[i], aie.LockAction.AcquireGreaterEqual, value = acq_value)
                    aie.dma_bd(self.o_buffer, offset = offset, len = part_len)
                    aie.use_lock(rel_lock, aie.LockAction.Release, value = rel_value)
                    aie.next_bd(block[30 + i])
                if not is_last:
                    with block[19 + i]:
                        aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3 + i, dest=block[31 + i], chain=block[20 + i])
                offset += part_len
            chain_after_parts = 19 + n_parts - 1

            # O out, o_sends times
            with block[chain_after_parts]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[40], chain=block[41])
            with block[40]:
                aie.use_lock(self.o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                if self.o_pkt_id is not None:
                    aie.dma_bd_packet(0, self.o_pkt_id)
                aie.dma_bd(self.o_buffer)
                aie.use_lock(self.tap_lock if self.tap else self.o_part_locks[0], aie.LockAction.Release, value = 1)
                aie.next_bd(block[40])

            # optional tap: one more copy of O, for the host
            if self.tap:
                with block[41]:
                    aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[42], chain=block[43])
                with block[42]:
                    aie.use_lock(self.tap_lock, aie.LockAction.AcquireGreaterEqual, value = self.o_sends)
                    aie.dma_bd(self.o_buffer)
                    aie.use_lock(self.o_part_locks[0], aie.LockAction.Release, value = self.o_sends)
                    aie.next_bd(block[42])
                end_block = 43
            else:
                end_block = 41

            with block[end_block]:
                aie.EndOp()

    @property
    def K_in(self):
        return self.S2MM[0]

    @property
    def V_in(self):
        return self.S2MM[1]

    @property
    def K_out_0(self):
        return self.MM2S[0]

    @property
    def K_out_1(self):
        return self.MM2S[1]

    @property
    def V_out_0(self):
        return self.MM2S[2]

    @property
    def V_out_1(self):
        return self.MM2S[3]

    def O_in(self, part: int):
        return self.S2MM[2 + part]

    @property
    def O_out(self):
        return self.MM2S[4]

    @property
    def TAP_out(self):
        assert self.tap
        return self.MM2S[5]


class attn_column:
    """One column: 2 attention CUs (one KV head each), its mem tile and shim.

    The shim feeds this column's slice of the KV cache to the mem tile, which
    splits it between the 2 CUs. The CUs' outputs come back to the mem tile as
    O parts 0 and 1; extra_o_parts lists the sizes of more O inputs (from
    another column) that join after them.
    """
    def __init__(self,
        attn_qk_tiles: List[tile_with_location],
        attn_kv_tiles: List[tile_with_location],
        memtile: tile_with_location,
        shim_tile: ShimTile,
        model_config: ModelConfig,
        ctx: mlir_context,
        qk_stack_size: int,
        kv_stack_size: int,
        qk_link_with: str,
        kv_link_with: str,
        extra_o_parts: List[int],
        o_sends: int,
        o_pkt_id: int = None,
        tap: bool = False,
    ):
        comment_print(f"Initializing attn_column at MT({memtile.row}, {memtile.col})")
        assert len(attn_qk_tiles) == 2 and len(attn_kv_tiles) == 2
        self.attn_cus = []
        for i in range(len(attn_qk_tiles)):
            self.attn_cus.append(attn_cu(
                attn_qk_tile = attn_qk_tiles[i],
                attn_kv_tile = attn_kv_tiles[i],
                model_config = model_config,
                ctx = ctx,
                qk_stack_size = qk_stack_size,
                kv_stack_size = kv_stack_size,
                qk_link_with = qk_link_with,
                kv_link_with = kv_link_with
            ))
        cu_o_size = model_config.Q_HEAD_PER_ROUND * model_config.DH
        self.o_size = cu_o_size * len(self.attn_cus)
        self.memtile = attn_memtile(memtile, ctx, model_config,
                                    kv_heads = model_config.kv_head_per_cu * len(self.attn_cus),
                                    o_parts = [cu_o_size] * len(self.attn_cus) + extra_o_parts,
                                    o_sends = o_sends, o_pkt_id = o_pkt_id, tap = tap)
        self.IT = shim_tile

    def route(self):
        connect_port(self.IT.MM2S[0], self.memtile.K_in)
        connect_port(self.IT.MM2S[1], self.memtile.V_in)
        connect_port(self.memtile.K_out_0, self.attn_cus[0].K_in)
        connect_port(self.memtile.V_out_0, self.attn_cus[0].V_in)
        connect_port(self.memtile.K_out_1, self.attn_cus[1].K_in)
        connect_port(self.memtile.V_out_1, self.attn_cus[1].V_in)
        connect_port(self.attn_cus[0].O_out, self.memtile.O_in(0))
        connect_port(self.attn_cus[1].O_out, self.memtile.O_in(1))

    @property
    def Q_in(self):
        return [cu.Q_in for cu in self.attn_cus]

    @property
    def O_out(self):
        return self.memtile.O_out


class attn_block:
    """Attention over 2 columns x 2 CUs, one KV head per CU (KV heads 0..3).

    The main column (KV heads 0, 1) owns the output. The second column (KV
    heads 2, 3) joins its 2 CU outputs in its own mem tile and sends that half
    to the main mem tile, which joins it after its own 2 parts, so O is in head
    order. The main mem tile then sends O o_sends times (to o_proj), and once
    more to the tap if asked.
    """
    def __init__(self, main_column: dict, second_column: dict, model_config: ModelConfig, ctx: mlir_context,
                 qk_stack_size: int, kv_stack_size: int, qk_link_with: str, kv_link_with: str,
                 o_sends: int, o_pkt_id: int, tap: bool = False):
        assert model_config.num_attn_cus == 4 and model_config.kv_head_per_cu == 1
        common = dict(model_config = model_config, ctx = ctx,
                      qk_stack_size = qk_stack_size, kv_stack_size = kv_stack_size,
                      qk_link_with = qk_link_with, kv_link_with = kv_link_with)
        self.second = attn_column(**second_column, **common, extra_o_parts = [], o_sends = 1)
        self.main = attn_column(**main_column, **common, extra_o_parts = [self.second.o_size],
                                o_sends = o_sends, o_pkt_id = o_pkt_id, tap = tap)

    def route(self):
        self.main.route()
        self.second.route()
        connect_port(self.second.O_out, self.main.memtile.O_in(2))

    @property
    def Q_in(self):
        # CU i gets Q packet i: KV head i and its query heads
        return self.main.Q_in + self.second.Q_in

    @property
    def O_out(self):
        return self.main.O_out

    @property
    def TAP_out(self):
        return self.main.memtile.TAP_out
