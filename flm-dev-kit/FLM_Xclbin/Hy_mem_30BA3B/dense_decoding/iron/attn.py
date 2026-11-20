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
        self.DQ_PADDED = model_config.PADDED_DQ
        self.DQ = model_config.DQ
        self.DK = model_config.DK
        self.lc = model_config.lq

        self.attn_q_ty = np.ndarray[(self.DQ_PADDED // 4,), self.dtype]
        self.attn_k_ty = np.ndarray[(self.lc, self.DK // 4), self.dtype]
        comment_print(f"attn_q_ty shape: ({self.DQ_PADDED // 4},), attn_k_ty shape: ({self.lc}, {self.DK // 4})")
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
                aie.dma_bd(self.q_buffer_0)
                aie.use_lock(self.q_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[2])

            with block[2]:
                aie.use_lock(self.q_prod_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.q_buffer_1)
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
        self.NQ_PER_CU = model_config.num_heads // 4
        self.DH = model_config.DH
        self.DV = model_config.DV
        self.lc = model_config.lk

        self.attn_v_ty = np.ndarray[(self.lc, self.DV // 4), self.dtype]
        self.attn_o_ty = np.ndarray[(self.DH * self.NQ_PER_CU, ), self.dtype]
        comment_print(f"attn_v_ty shape: ({self.lc}, {self.DV // 4}), attn_o_ty shape: {(self.DH * self.NQ_PER_CU, )}")
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
        self.NQ_PER_CU = model_config.NUM_Q_HEADS_PADDED // 4
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
        self.KV_HEAD_PER_COLUMN = model_config.num_kv_heads // 2
        self.DH = model_config.DH
        self.lc = model_config.lk

        self.k_mem_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_COLUMN * self.DH), np.dtype[bfloat16]]
        self.v_mem_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_COLUMN * self.DH), np.dtype[bfloat16]]
        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}) with lc: {self.lc}, d: { self.KV_HEAD_PER_COLUMN * self.DH }")

        self.k_buffer_0 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.k_buffer_1 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.v_buffer_0 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.v_buffer_1 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_1_{self.mt_row}_{self.mt_col}")

        self.k_prod_lock    = aie.lock(self.mt, lock_id = 0, init = 2)
        self.k_p0_cons_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        self.k_p1_cons_lock = aie.lock(self.mt, lock_id = 2, init = 0)
        self.v_prod_lock    = aie.lock(self.mt, lock_id = 3, init = 2)
        self.v_p0_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)
        self.v_p1_cons_lock = aie.lock(self.mt, lock_id = 5, init = 0)

        self.k_mem_row_size = self.KV_HEAD_PER_COLUMN * self.DH
        self.k_cu_offset = self.k_mem_row_size // 2
        comment_print(f"\tattn_memtile k_cu_offset: {self.k_cu_offset}")
        self.k_order = [(self.k_cu_offset // 8, 8), (16, self.k_mem_row_size), (8, 1)]
        comment_print(f"\tattn_memtile k_order: {self.k_order}")
        self.k_len_per_cu = self.k_cu_offset * self.lc

        self.v_mem_row_size = self.KV_HEAD_PER_COLUMN * self.DH
        self.v_cu_offset = self.v_mem_row_size // 2
        comment_print(f"\tattn_memtile v_cu_offset: {self.v_cu_offset}")
        self.v_order = [(2, self.lc // 2 * self.v_mem_row_size), (self.v_cu_offset // 8, 8), (8, self.v_mem_row_size), (8, 1)]
        comment_print(f"\tattn_memtile v_order: {self.v_order}")
        self.v_len_per_cu = self.v_cu_offset * self.lc
        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

            with block[1]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0)
                aie.use_lock(self.k_p0_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[2])

            with block[2]:
                aie.use_lock(self.k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1)
                aie.use_lock(self.k_p0_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

            with block[4]:
                aie.use_lock(self.k_p0_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0, offset = 0, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_p1_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[5])

            with block[5]:
                aie.use_lock(self.k_p0_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1, offset = 0, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_p1_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[4])

            with block[6]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[9])

            with block[7]:
                aie.use_lock(self.k_p1_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_0, offset = self.k_cu_offset, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[8])

            with block[8]:
                aie.use_lock(self.k_p1_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.k_buffer_1, offset = self.k_cu_offset, len = self.k_len_per_cu, dimensions = self.k_order)
                aie.use_lock(self.k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[10], chain=block[12])

            with block[10]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.v_buffer_0)
                aie.use_lock(self.v_p0_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[11])

            with block[11]:
                aie.use_lock(self.v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.v_buffer_1)
                aie.use_lock(self.v_p0_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[10])

            with block[12]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])

            with block[13]:
                aie.use_lock(self.v_p0_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_0, offset = 0, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_p1_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[14])

            with block[14]:
                aie.use_lock(self.v_p0_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_1, offset = 0, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_p1_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[13])

            with block[15]:
                s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])
            
            with block[16]:
                aie.use_lock(self.v_p1_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_0, offset = self.v_cu_offset, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[17])

            with block[17]:
                aie.use_lock(self.v_p1_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.v_buffer_1, offset = self.v_cu_offset, len = self.v_len_per_cu, dimensions = self.v_order)
                aie.use_lock(self.v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[16])
            
            with block[18]:
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
    

class attn_column:
    def __init__(self, 
        attn_qk_tile: List[tile_with_location], 
        attn_kv_tile: List[tile_with_location], 
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

        for i in range(2):
            comment_print(f"Initializing attn_cu {i} with QK({attn_qk_tile[i].row}, {attn_qk_tile[i].col}) KV({attn_kv_tile[i].row}, {attn_kv_tile[i].col})")
            self.attn_cus.append(attn_cu(
                attn_qk_tile = attn_qk_tile[i],
                attn_kv_tile = attn_kv_tile[i],
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
        connect_port(self.memtile.K_out_1, self.attn_cus[1].K_in)
        connect_port(self.memtile.V_out_0, self.attn_cus[0].V_in)
        connect_port(self.memtile.V_out_1, self.attn_cus[1].V_in)
    
    @property
    def Q_in_0(self):
        return self.attn_cus[0].Q_in
    
    @property
    def Q_in_1(self):
        return self.attn_cus[1].Q_in
    
    @property
    def O_out_0(self):
        return self.attn_cus[0].O_out
    
    @property
    def O_out_1(self):
        return self.attn_cus[1].O_out