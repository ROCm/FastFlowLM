# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import connect_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_CT_IP, AIE_MT_IP, comment_print, ShimTile

class qk_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, shared_s_buffers: List[Any], s_ty: Any,  model_config: ModelConfig, stack_size: int = 1024, link_with: str = "attn_qk.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing attn_qk_tile at ({self.ct_row}, {self.ct_col})")
        self.Q_HEAD_PER_ROUND = model_config.Q_HEAD_PER_ROUND_PADDED
        self.KV_HEAD_PER_ROUND = model_config.kv_head_per_round
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
        self.KV_HEAD_PER_ROUND = model_config.kv_head_per_round
        self.lc = model_config.lk
        self.enable_packet_id = model_config.num_attn_cus == 1

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
                if self.enable_packet_id:
                    aie.dma_bd_packet(0, 0)
                aie.dma_bd(self.o_buffer_0, offset = 0, len = self.NQ_PER_CU * self.DH, dimensions = [(self.NQ_PER_CU, 8), (self.DH // 8, self.NQ_PER_CU * 8), (8, 1)])
                aie.use_lock(self.o_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[5])
            
            with block[5]:
                aie.use_lock(self.o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                if self.enable_packet_id:
                    aie.dma_bd_packet(0, 0)
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

    def __init__(self, tile, mctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.num_kv_heads = model_config.num_kv_heads
        self.num_kv_heads_per_round = model_config.kv_head_per_round
        self.DH = model_config.DH
        self.lc = model_config.lk
        self.DQ = model_config.DQ

        self.k_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.v_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.o_ty     = np.ndarray[(self.DQ, ), np.dtype[bfloat16]]
        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}) with lc: {self.lc}, d: { self.num_kv_heads * self.DH }")

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
        self.o_prod_lock_0 = aie.lock(self.mt, lock_id = 6, init = 1)
        self.o_prod_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.o_cons_lock = aie.lock(self.mt, lock_id = 8, init = 0)

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

            
            with block[18]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[19], chain=block[20])

            with block[19]:
                aie.use_lock(self.o_prod_lock_0, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.DQ // 2)
                aie.use_lock(self.o_prod_lock_1, aie.LockAction.Release, value = 1)
                aie.next_bd(block[19])

            with block[20]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[21], chain=block[22])

            with block[21]:
                aie.use_lock(self.o_prod_lock_1, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.o_buffer, offset = self.DQ // 2, len = self.DQ // 2)
                aie.use_lock(self.o_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[21])

            with block[22]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[23], chain=block[24])
            
            with block[23]:
                aie.use_lock(self.o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd_packet(0, 0)
                aie.dma_bd(self.o_buffer)
                aie.use_lock(self.o_prod_lock_0, aie.LockAction.Release, value = 1)
                aie.next_bd(block[23])

            
            with block[24]:
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
    
    @property
    def O_in_0(self):
        return self.S2MM[2]
    
    @property
    def O_in_1(self):
        return self.S2MM[3]
    
    @property
    def O_out(self):
        return self.MM2S[4]
    

class attn_memtile_single_cu(AIE_MT_IP):

    def __init__(self, tile, mctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.num_kv_heads = model_config.num_kv_heads
        self.num_kv_heads_per_round = model_config.kv_head_per_round
        self.DH = model_config.DH
        self.lc = model_config.lk
        self.DQ = model_config.DQ

        self.k_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.v_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.o_ty     = np.ndarray[(self.DQ, ), np.dtype[bfloat16]]
        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}) with lc: {self.lc}, d: { self.num_kv_heads * self.DH }")

        self.k_buffer_0 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.k_buffer_1 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.v_buffer_0 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.v_buffer_1 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_1_{self.mt_row}_{self.mt_col}")

        self.k_prod_lock = aie.lock(self.mt, lock_id = 0, init = 2)
        self.k_cons_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        self.v_prod_lock = aie.lock(self.mt, lock_id = 3, init = 2)
        self.v_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)
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
                aie.use_lock(self.k_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[2])

            with block[2]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1)
                aie.use_lock(self.k_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[9])

            with block[4]:
                aie.use_lock(self.k_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0, offset = 0, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[5])

            with block[5]:
                aie.use_lock(self.k_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1, offset = 0, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[4])

            with block[9]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[10], chain=block[12])

            with block[10]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.v_buffer_0)
                aie.use_lock(self.v_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[11])

            with block[11]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.v_buffer_1)
                aie.use_lock(self.v_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[10])

            with block[12]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[24])

            with block[13]:
                aie.use_lock(self.v_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_0, offset = 0, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[14])

            with block[14]:
                aie.use_lock(self.v_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_1, offset = 0, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[13])

        
            with block[24]:
                aie.EndOp()

    @property
    def K_in(self):
        return self.S2MM[0]
    
    @property
    def V_in(self):
        return self.S2MM[1]
    
    @property
    def K_out(self):
        return self.MM2S[0]
    
    @property
    def V_out(self):
        return self.MM2S[2]
    
    
class attn_column:
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
        kv_link_with: str
    ):  
        comment_print(f"Initializing attn_column at MT({memtile.row}, {memtile.col})")
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
        self.memtile = attn_memtile(memtile, ctx, model_config)
        self.IT = shim_tile

    def route(self):
        connect_port(self.IT.MM2S[0], self.memtile.K_in)
        connect_port(self.IT.MM2S[1], self.memtile.V_in)
        connect_port(self.memtile.K_out_0, self.attn_cus[0].K_in)
        connect_port(self.memtile.V_out_0, self.attn_cus[0].V_in)
        connect_port(self.memtile.K_out_1, self.attn_cus[1].K_in)
        connect_port(self.memtile.V_out_1, self.attn_cus[1].V_in)
        connect_port(self.attn_cus[0].O_out, self.memtile.O_in_0)
        connect_port(self.attn_cus[1].O_out, self.memtile.O_in_1)

    
    @property
    def Q_in(self):
        return [cu.Q_in for cu in self.attn_cus]
    
    @property
    def O_out(self):
        return self.memtile.O_out
    

class attn_column_single_cu:
    def __init__(self, 
        attn_qk_tile: tile_with_location, 
        attn_kv_tile: tile_with_location, 
        memtile: tile_with_location,
        shim_tile: ShimTile,
        model_config: ModelConfig,
        ctx: mlir_context, 
        qk_stack_size: int, 
        kv_stack_size: int, 
        qk_link_with: str, 
        kv_link_with: str
    ):  
        comment_print(f"Initializing attn_column at MT({memtile.row}, {memtile.col})")
        self.attn_cus = []
        self.attn_cus.append(attn_cu(
            attn_qk_tile = attn_qk_tile,
            attn_kv_tile = attn_kv_tile,
            model_config = model_config,
            ctx = ctx,
            qk_stack_size = qk_stack_size,
            kv_stack_size = kv_stack_size,
            qk_link_with = qk_link_with,
            kv_link_with = kv_link_with
        ))
        self.memtile = attn_memtile_single_cu(memtile, ctx, model_config)
        self.IT = shim_tile

    def route(self):
        connect_port(self.IT.MM2S[0], self.memtile.K_in)
        connect_port(self.IT.MM2S[1], self.memtile.V_in)
        connect_port(self.memtile.K_out, self.attn_cus[0].K_in)
        connect_port(self.memtile.V_out, self.attn_cus[0].V_in)

    @property
    def Q_in(self):
        return self.attn_cus[0].Q_in
    
    @property
    def O_out(self):
        return self.attn_cus[0].O_out
    