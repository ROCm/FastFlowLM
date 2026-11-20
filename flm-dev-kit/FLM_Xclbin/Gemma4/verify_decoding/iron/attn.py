# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from iron.utils import ENABLE_CHESSCC

# The host writes the decode context length into attn_kv's RTP buffer at a
# HARDCODED address -- FLM_DLL/detail/gemma4e_npu/gemma4e_npu_sequence.hpp,
# .l_kv_address = 14976 for BOTH e2b and e4b.  Peano needs this tile repacked
# (its ldscript only grants a .data/.bss tail above the highest buffer, and the
# default layout runs to the top of memory), but repacking moves that RTP: the
# host then writes L somewhere attn_kv never reads, the kernel spins on a
# garbage trip count and the run dies with ERT_CMD_STATE_TIMEOUT on the second
# decode token.  So on peano we place this tile's buffers by hand: same address
# for the RTP as chess, everything else packed low enough to leave a tail.
HOST_L_KV_ADDRESS = 14976
ATTN_KV_STACK_BYTES = 6144   # core stack occupies [0, 6144)

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref
from constants import *
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
        if model_config.name == "gemma4_e2b":
            self.attn_k_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_ROUND * self.DH), self.dtype]
        else:
            self.attn_k_ty = np.ndarray[(self.lc, self.DH), self.dtype]

        comment_print(f"attn_q_ty shape: ({self.Q_HEAD_PER_ROUND * self.DH},), attn_k_ty shape: ({self.lc}, {self.KV_HEAD_PER_ROUND * self.DH})")
        self.attn_s_ty = s_ty
        self.RTP_ty = ctx.data_tys["RTP_ty"]

        self.attn_qk = ctx.func("attn_qk", [self.attn_q_ty, self.attn_q_ty, self.attn_k_ty, self.attn_k_ty, self.attn_s_ty, self.attn_s_ty, self.RTP_ty])

        self.k_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_k_ty, name = f"k_0_{self.ct_row}_{self.ct_col}")
        self.k_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_k_ty, name = f"k_1_{self.ct_row}_{self.ct_col}")

        self.q_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_q_ty, name = f"q_0_{self.ct_row}_{self.ct_col}")
        self.q_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_q_ty, name = f"q_1_{self.ct_row}_{self.ct_col}")
        if model_config.name == "gemma4_e2b":
            self.q_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
            self.q_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
            self.k_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
            self.k_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
            self.s_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
            self.s_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
            self.l_cons_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        else: # for gemma4_e4b, we have 2x kv head per round, so we need only 1 lock for k and 1 lock for s
            self.q_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
            self.q_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
            self.k_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
            self.k_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
            self.s_prod_lock_0 = aie.lock(self.ct, lock_id = 4, init = 1)
            self.s_cons_lock_0 = aie.lock(self.ct, lock_id = 5, init = 0)
            self.s_prod_lock_1 = aie.lock(self.ct, lock_id = 6, init = 1)
            self.s_cons_lock_1 = aie.lock(self.ct, lock_id = 7, init = 0)
            self.l_cons_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        self.s_buffer_0 = shared_s_buffers[0]
        self.s_buffer_1 = shared_s_buffers[1]

        self.L_buffer = ctx.buffer(f"RTP_L_attn_qk_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.RTP_ty, use_write_rtp = True)
        self.q_heads_len = self.Q_HEAD_PER_ROUND * self.DH
        self.q_in_order = [(self.Q_HEAD_PER_ROUND, 8), (self.DH // 8, self.Q_HEAD_PER_ROUND * 8), (8, 1)]

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size * 4)
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
        
        self.O_REPEATS = model_config.D // (model_config.q4nx_row_block_size * 16) 
        if L_TOKENS > 1:
            self.O_REPEATS = 1   # verify design: o leaves once per token, the o x-replay happens downstream (o park)

        if model_config.name == "gemma4_e2b":
            self.attn_v_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_ROUND * self.DH), self.dtype]
        else:
            self.attn_v_ty = np.ndarray[(self.lc, self.DH), self.dtype]

        self.attn_o_ty = np.ndarray[(self.DH * self.NQ_PER_CU, ), self.dtype]
        self.attn_y_ty = np.ndarray[(self.DH * self.NQ_PER_CU, ), np.dtype[np.float32]]
        comment_print(f"attn_v_ty shape: ({self.lc}, {self.KV_HEAD_PER_ROUND * self.DH}), attn_o_ty shape: {(self.DH * self.NQ_PER_CU, )}")
        self.attn_s_ty = s_ty
        self.RTP_ty = ctx.data_tys["RTP_ty"]
        self.s_ty = s_ty
        self.attn_kv = ctx.func("attn_kv", [self.attn_s_ty, self.attn_s_ty, self.attn_v_ty, self.attn_v_ty, self.attn_y_ty, self.attn_o_ty, self.RTP_ty])

        # The host writes the decode context length here via a HARDCODED address
        # (FLM_DLL/detail/gemma4e_npu/gemma4e_npu_sequence.hpp, .l_kv_address = 14976,
        # for both E2B and E4B).  The chess allocator happens to place it there; the
        # peano path puts this tile on allocation_scheme="basic-sequential" (see
        # utils.py) which repacks it to 64128, so the host writes L to an address the
        # kernel never reads, attn_kv spins on a garbage trip count and the run dies
        # with ERT_CMD_STATE_TIMEOUT on the second decode token.  Pin it so both
        # toolchains present the same address to the runtime.  chess is left on the
        # allocator's own choice so its artifacts stay byte-identical to main.
        def _nbytes(ty):
            shape, dt = ty.__args__[0], ty.__args__[1]
            # dt is either a numpy scalar type or np.dtype[<scalar>]
            dt = np.dtype(getattr(dt, "__args__", (dt,))[0])
            n = 1
            for d in (shape if isinstance(shape, tuple) else (shape,)):
                n *= d
            return n * dt.itemsize

        def _addr(prev_end, ty):
            # 64-byte granularity, matching what the allocators emit
            return None if ENABLE_CHESSCC else ((prev_end + 63) // 64) * 64

        if ENABLE_CHESSCC:
            a_o = a_s0 = a_s1 = a_rtp = a_v0 = a_v1 = a_y = None
        else:
            a_o  = ATTN_KV_STACK_BYTES
            a_s0 = _addr(a_o  + _nbytes(self.attn_o_ty), self.s_ty)
            a_s1 = _addr(a_s0 + _nbytes(self.s_ty),      self.s_ty)
            a_rtp= _addr(a_s1 + _nbytes(self.s_ty),      self.RTP_ty)
            # This is the whole point of the hand layout -- if a future size
            # change moves it, fail loudly here instead of hanging on hardware.
            assert a_rtp == HOST_L_KV_ADDRESS, (
                f"attn_kv RTP landed at {a_rtp}, but FLM_DLL hardcodes "
                f"{HOST_L_KV_ADDRESS} (gemma4e_npu_sequence.hpp .l_kv_address)")
            a_v0 = _addr(a_rtp + _nbytes(self.RTP_ty),   self.attn_v_ty)
            a_v1 = _addr(a_v0  + _nbytes(self.attn_v_ty), self.attn_v_ty)
            a_y  = _addr(a_v1  + _nbytes(self.attn_v_ty), self.attn_y_ty)

        self.L_buffer = ctx.buffer(f"RTP_L_attn_kv_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.RTP_ty, use_write_rtp = True, use_address = a_rtp)
        self.s_buffer_0 = ctx.buffer(f"s_shared_buffer_0_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty, use_address = a_s0)
        self.s_buffer_1 = ctx.buffer(f"s_shared_buffer_1_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty, use_address = a_s1)
        self.v_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_0_{self.ct_row}_{self.ct_col}", use_address = a_v0)
        self.v_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_1_{self.ct_row}_{self.ct_col}", use_address = a_v1)
        self.y_buffer = ctx.buffer(tile = self.located_tile, dtype = self.attn_y_ty, name = f"y_{self.ct_row}_{self.ct_col}", use_address = a_y)
        self.o_buffer = ctx.buffer(tile = self.located_tile, dtype = self.attn_o_ty, name = f"o_{self.ct_row}_{self.ct_col}", use_address = a_o)


        self.o_prod_lock = aie.lock(self.ct, lock_id = 0, init = self.O_REPEATS)
        self.o_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.v_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.v_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.__call__()
    
    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size * 6)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.attn_kv(self.s_buffer_0, self.s_buffer_1, self.v_buffer_0, self.v_buffer_1, self.y_buffer, self.o_buffer, self.L_buffer)

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
                aie.dma_bd_packet(0, to_proj_x_in_pkt_id.from_attn)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.NQ_PER_CU * self.DH, dimensions = [(self.NQ_PER_CU, 8), (self.DH // 8, self.NQ_PER_CU * 8), (8, 1)])
                # aie.dma_bd(self.o_buffer)
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
        self.s_ty = np.ndarray[(self.NQ_PER_CU * 16 + 32, 1), np.dtype[bfloat16]] # for attention score and correction
        comment_print(f"\tattn_s_ty shape: {(self.NQ_PER_CU * 16 + 32, 1)}")
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

class swa_qk_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, shared_s_buffers: List[Any], s_ty: Any,  model_config: ModelConfig, stack_size: int = 1024, link_with: str = "swa_attn_qk.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing swa_attn_qk_tile at ({self.ct_row}, {self.ct_col})")
        self.Q_HEAD_PER_ROUND = model_config.Q_HEAD_PER_ROUND_PADDED
        self.KV_HEAD_PER_ROUND = model_config.kv_head_per_round
        self.DH = model_config.SWA_DH
        self.lc = model_config.lq

        self.attn_q_ty = np.ndarray[(self.Q_HEAD_PER_ROUND * self.DH,), self.dtype]
        self.attn_k_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_ROUND * self.DH), self.dtype]
        comment_print(f"swa_attn_q_ty shape: ({self.Q_HEAD_PER_ROUND * self.DH},), swa_attn_k_ty shape: ({self.lc}, {self.KV_HEAD_PER_ROUND * self.DH})")
        self.attn_s_ty = s_ty
        self.RTP_ty = ctx.data_tys["RTP_ty"]

        self.swa_attn_qk = ctx.func("swa_attn_qk", [self.attn_q_ty, self.attn_q_ty, self.attn_k_ty, self.attn_k_ty, self.attn_s_ty, self.attn_s_ty, self.RTP_ty])

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
        self.l_cons_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        self.s_buffer_0 = shared_s_buffers[0]
        self.s_buffer_1 = shared_s_buffers[1]

        self.L_buffer = ctx.buffer(f"RTP_L_swa_attn_qk_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.RTP_ty, use_write_rtp = True)
        self.q_heads_len = self.Q_HEAD_PER_ROUND * self.DH
        self.q_in_order = [(self.Q_HEAD_PER_ROUND, 8), (self.DH // 8, self.Q_HEAD_PER_ROUND * 8), (8, 1)]

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.swa_attn_qk(self.q_buffer_0, self.q_buffer_1, self.k_buffer_0, self.k_buffer_1, self.s_buffer_0, self.s_buffer_1, self.L_buffer)

            
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

class swa_kv_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, s_ty: Any, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "swa_attn_kv.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing swa_attn_kv_tile at ({self.ct_row}, {self.ct_col})")
        self.NQ_PER_CU = model_config.Q_HEAD_PER_ROUND
        self.DH = model_config.SWA_DH
        self.KV_HEAD_PER_ROUND = model_config.kv_head_per_round
        self.lc = model_config.lk
        
        self.O_REPEATS = model_config.D // (model_config.q4nx_row_block_size * 16) 
        if L_TOKENS > 1:
            self.O_REPEATS = 1   # verify design: o leaves once per token, the o x-replay happens downstream (o park)

        self.attn_v_ty = np.ndarray[(self.lc, self.KV_HEAD_PER_ROUND * self.DH), self.dtype]
        self.attn_o_ty = np.ndarray[(self.DH * self.NQ_PER_CU, ), self.dtype]
        self.attn_y_ty = np.ndarray[(self.DH * self.NQ_PER_CU, ), np.dtype[np.float32]]
        comment_print(f"swa_attn_v_ty shape: ({self.lc}, {self.KV_HEAD_PER_ROUND * self.DH}), swa_attn_o_ty shape: {(self.DH * self.NQ_PER_CU, )}")
        self.attn_s_ty = s_ty
        self.RTP_ty = ctx.data_tys["RTP_ty"]
        self.s_ty = s_ty
        self.attn_kv = ctx.func("swa_attn_kv", [self.attn_s_ty, self.attn_s_ty, self.attn_v_ty, self.attn_v_ty, self.attn_y_ty, self.attn_o_ty, self.RTP_ty])

        self.L_buffer = ctx.buffer(f"RTP_L_swa_attn_kv_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.RTP_ty, use_write_rtp = True)
        self.s_buffer_0 = ctx.buffer(f"s_shared_buffer_0_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty)
        self.s_buffer_1 = ctx.buffer(f"s_shared_buffer_1_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty)
        self.v_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_0_{self.ct_row}_{self.ct_col}")
        self.v_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_1_{self.ct_row}_{self.ct_col}")

        self.y_buffer = ctx.buffer(tile = self.located_tile, dtype = self.attn_y_ty, name = f"y_{self.ct_row}_{self.ct_col}")

        self.o_buffer = ctx.buffer(tile = self.located_tile, dtype = self.attn_o_ty, name = f"o_{self.ct_row}_{self.ct_col}")

        self.o_prod_lock = aie.lock(self.ct, lock_id = 0, init = self.O_REPEATS)
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
                self.attn_kv(self.s_buffer_0, self.s_buffer_1, self.v_buffer_0, self.v_buffer_1, self.y_buffer, self.o_buffer, self.L_buffer)

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
                aie.dma_bd_packet(0, to_proj_x_in_pkt_id.from_swa_attn)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.NQ_PER_CU * self.DH, dimensions = [(self.NQ_PER_CU, 8), (self.DH // 8, self.NQ_PER_CU * 8), (8, 1)])
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

class swa_attn_cu:
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
        comment_print(f"Initializing swa_attn_cu at QK({attn_qk_tile.row}, {attn_qk_tile.col}) KV({attn_kv_tile.row}, {attn_kv_tile.col})")
        self.NQ_PER_CU = model_config.Q_HEAD_PER_ROUND_PADDED
        self.s_ty = np.ndarray[(self.NQ_PER_CU * 16 + 32, 1), np.dtype[bfloat16]] # for attention score and correction
        comment_print(f"\tswa_attn_s_ty shape: {(self.NQ_PER_CU * 16 + 32, 1)}")
        self.kv_core = swa_kv_tile(attn_kv_tile, ctx, self.s_ty, model_config, stack_size = kv_stack_size, link_with = kv_link_with)
        self.qk_core = swa_qk_tile(attn_qk_tile, ctx, self.kv_core.get_s_buffers, self.s_ty, model_config, stack_size = qk_stack_size, link_with = qk_link_with)
    
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
        self.SWA_DH = model_config.SWA_DH
        self.lc = model_config.lk

        self.k_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.v_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.DH), np.dtype[bfloat16]]
        self.swa_k_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.SWA_DH), np.dtype[bfloat16]]
        self.swa_v_mem_ty = np.ndarray[(self.lc, self.num_kv_heads * self.SWA_DH), np.dtype[bfloat16]]
        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}) with lc: {self.lc}, d: { self.num_kv_heads * self.DH }")

        self.k_buffer_0 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.k_buffer_1 = aie.buffer(self.mt, self.k_mem_ty, name = f"k_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.v_buffer_0 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.v_buffer_1 = aie.buffer(self.mt, self.v_mem_ty, name = f"v_mem_buffer_1_{self.mt_row}_{self.mt_col}")


        self.swa_k_buffer_0 = aie.buffer(self.mt, self.swa_k_mem_ty, name = f"swa_k_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.swa_k_buffer_1 = aie.buffer(self.mt, self.swa_k_mem_ty, name = f"swa_k_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.swa_v_buffer_0 = aie.buffer(self.mt, self.swa_v_mem_ty, name = f"swa_v_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.swa_v_buffer_1 = aie.buffer(self.mt, self.swa_v_mem_ty, name = f"swa_v_mem_buffer_1_{self.mt_row}_{self.mt_col}")

        self.k_prod_lock = aie.lock(self.mt, lock_id = 0, init = 2)
        self.k_cons_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        self.v_prod_lock = aie.lock(self.mt, lock_id = 3, init = 2)
        self.v_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)

        self.swa_k_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.swa_k_cons_lock = aie.lock(self.mt, lock_id = 6, init = 0)
        self.swa_v_prod_lock = aie.lock(self.mt, lock_id = 7, init = 2)
        self.swa_v_cons_lock = aie.lock(self.mt, lock_id = 8, init = 0)

        if model_config.name == "gemma4_e2b":
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
        else:
            self.k_mem_row_size = self.num_kv_heads * self.DH
            self.k_mem_cu_row_size = self.num_kv_heads_per_round * self.DH
            self.k_order = [(self.k_mem_cu_row_size // 8, 8), (16, self.k_mem_row_size), (8, 1)]
            self.k_offset = self.k_mem_cu_row_size
            comment_print(f"\tattn_memtile k_order: {self.k_order}")
            self.k_len_per_cu = self.k_mem_cu_row_size * self.lc

            self.v_mem_row_size = self.num_kv_heads * self.DH
            self.v_mem_cu_row_size = self.num_kv_heads_per_round * self.DH
            self.v_order = [(self.v_mem_cu_row_size // 8, 8), (self.lc, self.v_mem_row_size), (8, 1)]
            comment_print(f"\tattn_memtile v_order: {self.v_order}")
            self.v_len_per_cu = self.v_mem_cu_row_size * self.lc
            self.v_offset = self.v_mem_cu_row_size

        self.swa_k_mem_row_size = self.num_kv_heads * self.SWA_DH
        self.swa_k_mem_cu_row_size = self.num_kv_heads_per_round * self.SWA_DH
        self.swa_k_order = [(self.swa_k_mem_cu_row_size // 8, 8), (16, self.swa_k_mem_row_size), (8, 1)]
        self.swa_k_offset = self.swa_k_mem_cu_row_size
        comment_print(f"\tattn_memtile swa_k_order: {self.swa_k_order}")
        self.swa_k_len_per_cu = self.swa_k_mem_cu_row_size * self.lc

        self.swa_v_mem_row_size = self.num_kv_heads * self.SWA_DH
        self.swa_v_mem_cu_row_size = self.num_kv_heads_per_round * self.SWA_DH
        self.swa_v_order = [(self.lc // 8, self.lc // 2 * self.swa_v_mem_row_size), (self.swa_v_mem_cu_row_size // 8, 8), (8, self.swa_v_mem_row_size), (8, 1)]
        comment_print(f"\tattn_memtile swa_v_order: {self.swa_v_order}")
        self.swa_v_len_per_cu = self.swa_v_mem_cu_row_size * self.lc
        self.swa_v_offset = self.swa_v_mem_cu_row_size
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
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[13], chain=block[24])

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
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[25], chain=block[27])

            with block[25]:
                aie.use_lock(self.swa_k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_k_buffer_0)
                aie.use_lock(self.swa_k_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[26])

            with block[26]:
                aie.use_lock(self.swa_k_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_k_buffer_1)
                aie.use_lock(self.swa_k_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[25])

            with block[27]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[28], chain=block[30])

            with block[28]:
                aie.use_lock(self.swa_k_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_k_buffer_0, offset = 0, len = self.swa_k_len_per_cu, dimensions = self.swa_k_order)
                aie.use_lock(self.swa_k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[29])

            with block[29]:
                aie.use_lock(self.swa_k_cons_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_k_buffer_1, offset = 0, len = self.swa_k_len_per_cu, dimensions = self.swa_k_order)
                aie.use_lock(self.swa_k_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[28])

            with block[30]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[31], chain=block[33])

            with block[31]:
                aie.use_lock(self.swa_v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_v_buffer_0)
                aie.use_lock(self.swa_v_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[32])

            with block[32]:
                aie.use_lock(self.swa_v_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_v_buffer_1)
                aie.use_lock(self.swa_v_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[31])

            with block[33]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[34], chain=block[36])

            with block[34]:
                aie.use_lock(self.swa_v_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.swa_v_buffer_0, offset = 0, len = self.swa_v_len_per_cu, dimensions = self.swa_v_order)
                aie.use_lock(self.swa_v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[35])

            with block[35]:
                aie.use_lock(self.swa_v_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.swa_v_buffer_1, offset = 0, len = self.swa_v_len_per_cu, dimensions = self.swa_v_order)
                aie.use_lock(self.swa_v_prod_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[34])

            with block[36]:
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
        return self.MM2S[1]
    
    @property
    def SWA_K_in(self):
        return self.S2MM[2]
    
    @property
    def SWA_V_in(self):
        return self.S2MM[3]
    
    @property
    def SWA_K_out(self):
        return self.MM2S[2]
    
    @property
    def SWA_V_out(self):
        return self.MM2S[3]
    

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
        kv_link_with: str,
        swa_qk_link_with: str, 
        swa_kv_link_with: str
    ):  
        comment_print(f"Initializing attn_column at MT({memtile.row}, {memtile.col})")
        self.global_attn_cu = attn_cu(
            attn_qk_tile = attn_qk_tiles[0],
            attn_kv_tile = attn_kv_tiles[0],
            model_config = model_config,
            ctx = ctx,
            qk_stack_size = qk_stack_size,
            kv_stack_size = kv_stack_size,
            qk_link_with = qk_link_with,
            kv_link_with = kv_link_with
        )

        self.sliding_swa_attn_cu = swa_attn_cu(
            attn_qk_tile = attn_qk_tiles[1],
            attn_kv_tile = attn_kv_tiles[1],
            model_config = model_config,
            ctx = ctx,
            qk_stack_size = qk_stack_size,
            kv_stack_size = kv_stack_size,
            qk_link_with = swa_qk_link_with,
            kv_link_with = swa_kv_link_with
        )
        self.memtile = attn_memtile(memtile, ctx, model_config)
        self.IT = shim_tile

    def route(self):
        connect_port(self.IT.MM2S[0], self.memtile.K_in, pkt_id = 12, keep_pkt_header = False)
        connect_port(self.IT.MM2S[0], self.memtile.SWA_K_in, pkt_id = 13, keep_pkt_header = False)
        connect_port(self.IT.MM2S[1], self.memtile.V_in, pkt_id = 12, keep_pkt_header = False)
        connect_port(self.IT.MM2S[1], self.memtile.SWA_V_in, pkt_id = 13, keep_pkt_header = False)
        connect_port(self.memtile.K_out, self.global_attn_cu.K_in)
        connect_port(self.memtile.V_out, self.global_attn_cu.V_in)
        connect_port(self.memtile.SWA_K_out, self.sliding_swa_attn_cu.K_in)
        connect_port(self.memtile.SWA_V_out, self.sliding_swa_attn_cu.V_in)

    @property
    def Q_in(self):
        return self.global_attn_cu.Q_in
    
    @property
    def O_out(self):
        return self.global_attn_cu.O_out
    
    @property
    def SWA_Q_in(self):
        return self.sliding_swa_attn_cu.Q_in
    
    @property
    def SWA_O_out(self):
        return self.sliding_swa_attn_cu.O_out
    