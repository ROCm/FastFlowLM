# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from aie.extras.dialects.ext.arith import constant, index_cast
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

import threading
from utils import mlir_context, tile_with_location, connect_dma, dma_port, connect_port, lock_acquire, lock_release, comment_print, ShimTile
from utils import AIE_CT_IP, AIE_MT_IP

from constants import proj_pkt_id, proj_x_pkt_id, proj_x_in_pkt_id, KERNEL_L
from math import gcd
KL = KERNEL_L   # tokens the projection core handles per call (1 = shipping decode core)

from typing import Any, List
# phase 1: qkv proj, D -> (D + D // 2) to RoPE

# phase 2: o_proj, D -> D to rms_residual

# phase 3: up/gate proj, D -> D * 8 to MT_GLU

# phase 4: down proj, D * 4 -> D, to rms_residual

# minimal output unit: (m, 1)

class proj_tile_main(AIE_CT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "proj_main.o",
                 send_x_out:bool=True, main_y_buffer_0:Any|None= None, main_y_buffer_1:Any|None=None
                 ):
        
        """
        send_x_out: boolean determine of this CT sends the output x to mt
        
        For context: 2CT are group into one mvm unit produce length of 2*m output, but only CT with send_x_out=True responsible sending the output to MT
        
        IF `send_x_out`==False, this CT writes the output to `main_y_buffer_0` and `main_y_buffer_1`, the output buffer are buffer allocate in its neighbor CT that has `send_x_out`=True
        """
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing proj_tile at ({tile.row}, {tile.col})")

        self.send_x_out = send_x_out
        
        self.m = model_config.q4nx_row_block_size * KL      # elements per tile per row block (KL tokens x 32 rows)
        self.x_slice_size = model_config.x_slice_size * KL  # elements per x slice (KL tokens x 256 columns)
        
        self.x_slice_ty = np.ndarray[(self.x_slice_size,), np.dtype[bfloat16]]
        self.linear_w_ty = np.ndarray[(model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        self.m_pkt_ty = np.ndarray[(2 * self.m + 16,), np.dtype[bfloat16]]

        self.IS_SWA_BUFFER = ctx.buffer(f"RTP_PROJ_IS_SWA_BUFFER_{self.ct_row}_{self.ct_col}", tile, ctx.data_tys["RTP_ty"], use_write_rtp = True)
        self.SKIP_KV_BUFFER = ctx.buffer(f"RTP_PROJ_SKIP_KV_BUFFER_{self.ct_row}_{self.ct_col}", tile, ctx.data_tys["RTP_ty"], use_write_rtp = True)
        self.x_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_0_{self.ct_row}_{self.ct_col}")
        self.w_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.linear_w_ty, name = f"w_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_1_{self.ct_row}_{self.ct_col}")
        self.w_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.linear_w_ty, name = f"w_1_{self.ct_row}_{self.ct_col}")
        
        if send_x_out:
            self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_0_{self.ct_row}_{self.ct_col}")
            self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_1_{self.ct_row}_{self.ct_col}")
        else:
            assert main_y_buffer_0 is not  None
            self.y_buffer_0 = main_y_buffer_0
            assert main_y_buffer_1 is not None
            self.y_buffer_1 = main_y_buffer_1
            
            # allocate fake buffer for the sake of same rtp address
            # or use a different addressing mode
            self._y_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_0_{self.ct_row}_{self.ct_col}")
            self._y_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_1_{self.ct_row}_{self.ct_col}")
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.y_prod_ping_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.y_prod_pong_lock = aie.lock(self.ct, lock_id = 5, init = 2)
        self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)
        self.y_cons_ping_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.y_cons_pong_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        
        self.proj = ctx.func("proj_main", [self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, ctx.RTP_ty, ctx.RTP_ty, np.int32])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                if self.send_x_out:
                    self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, self.IS_SWA_BUFFER, self.SKIP_KV_BUFFER, constant(1))
                else:
                    self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, self.IS_SWA_BUFFER, self.SKIP_KV_BUFFER, constant(0))
        
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
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
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
            
            with block[4]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[5])
            
            with block[5]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[4])
            if self.send_x_out:
                with block[6]:
                    s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[7], chain = block[9])
                    
                with block[7]: # x, to mt
                    lock_acquire(self.y_cons_ping_lock, 2)
                    aie.dma_bd(self.y_buffer_0, offset = 14, len = 2 * self.m + 2)
                    lock_release(self.y_prod_ping_lock, 2)
                    aie.next_bd(block[8])


                with block[8]: # x, to mt
                    lock_acquire(self.y_cons_pong_lock, 2)
                    aie.dma_bd(self.y_buffer_1, offset = 14, len = 2 * self.m + 2)
                    lock_release(self.y_prod_pong_lock, 2)
                    aie.next_bd(block[7])

                with block[9]:
                    aie.EndOp()
            else:
                with block[6]:
                    aie.EndOp()

    @property
    def y_buffer_ping(self):
        return self.y_buffer_0
    
    @property
    def y_buffer_pong(self):
        return self.y_buffer_1
    
    @property
    def w_in(self):
        return self.S2MM[1]
    
    @property
    def x_in(self):
        return self.S2MM[0]
    
    @property
    def y_out(self):
        return self.MM2S[0]

# class proj_tile(AIE_CT_IP):
#     def __init__(self, tile: tile_with_location, ctx: mlir_context, y_buffer_ping: Any, y_buffer_pong: Any, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "proj.o"):
#         super().__init__(tile, stack_size = stack_size, link_with = link_with)
#         comment_print(f"Initializing proj_tile at ({tile.row}, {tile.col})")

#         self.m = model_config.q4nx_row_block_size
#         self.x_slice_size = model_config.x_slice_size
        
#         self.x_slice_ty = np.ndarray[(self.x_slice_size,), np.dtype[bfloat16]]
#         self.linear_w_ty = np.ndarray[(model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
#         self.m_pkt_ty = np.ndarray[(2 * self.m + 16,), np.dtype[bfloat16]]

#         self.L_buffer = ctx.buffer(f"RTP_PROJ_IS_ATTN_BUFFER_{self.ct_row}_{self.ct_col}", tile, ctx.data_tys["RTP_ty"], use_write_rtp = True)
#         self.x_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_0_{self.ct_row}_{self.ct_col}")
#         self.w_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.linear_w_ty, name = f"w_0_{self.ct_row}_{self.ct_col}")
#         self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_0_{self.ct_row}_{self.ct_col}")
#         self.x_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_1_{self.ct_row}_{self.ct_col}")
#         self.w_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.linear_w_ty, name = f"w_1_{self.ct_row}_{self.ct_col}")
#         self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_1_{self.ct_row}_{self.ct_col}")
#         self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
#         self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
#         self.w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
#         self.w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
#         self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)

#         self.y_buffer_ping = y_buffer_ping
#         self.y_buffer_pong = y_buffer_pong
         
#         self.proj = ctx.func("proj", [self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, ctx.RTP_ty])

#         self.__call__()

#     def __call__(self):
#         @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
#         def core_body():
#             for _ in range_(0xFFFFFFFF):
#                 self.proj(self.y_buffer_ping, self.w_buffer_0, self.x_buffer_0, self.y_buffer_pong, self.w_buffer_1, self.x_buffer_1, self.L_buffer)
        
#         @aie.mem(self.ct)
#         def mem_body(block):
#             s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
#             with block[1]:
#                 lock_acquire(self.x_prod_lock, 1)
#                 aie.dma_bd(self.x_buffer_0)
#                 lock_release(self.x_cons_lock, 1)
#                 aie.next_bd(block[2])

#             with block[2]:
#                 lock_acquire(self.x_prod_lock, 1)
#                 aie.dma_bd(self.x_buffer_1)
#                 lock_release(self.x_cons_lock, 1)
#                 aie.next_bd(block[1])
            
#             with block[3]:
#                 s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])
            
#             with block[4]:
#                 lock_acquire(self.w_prod_lock, 1)
#                 aie.dma_bd(self.w_buffer_0)
#                 lock_release(self.w_cons_lock, 1)
#                 aie.next_bd(block[5])
            
#             with block[5]:
#                 lock_acquire(self.w_prod_lock, 1)
#                 aie.dma_bd(self.w_buffer_1)
#                 lock_release(self.w_cons_lock, 1)
#                 aie.next_bd(block[4])

#             with block[6]:
#                 aie.EndOp()

#     @property
#     def w_in(self):
#         return self.S2MM[1]
    
#     @property
#     def x_in(self):
#         return self.S2MM[0]

class proj_combo:
    def __init__(self, cts: List[tile_with_location], ctx, model_config: ModelConfig, stack_size: int = 1024, link_with: List[str] = ["proj_main.o"]):
        self.proj_main = proj_tile_main(tile=cts[0],ctx= ctx, model_config=model_config,stack_size= stack_size,link_with= link_with[0], send_x_out=True)
        
        
        self.proj = proj_tile_main(tile=cts[1], ctx=ctx, model_config=model_config, stack_size=stack_size, 
                                   link_with=link_with[0], send_x_out=False, main_y_buffer_0=self.proj_main.y_buffer_ping, main_y_buffer_1=self.proj_main.y_buffer_pong
                                   
                                   )

    @property
    def w_in_0(self):
        return self.proj_main.w_in
    @property
    def x_in_0(self):
        return self.proj_main.x_in
    @property
    def w_in_1(self):
        return self.proj.w_in
    @property
    def x_in_1(self):
        return self.proj.x_in
    @property
    def y_out(self):
        return self.proj_main.y_out

class proj_assemble_col(AIE_MT_IP):

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_col at ({tile.row}, {tile.col})")

        self.m = model_config.q4nx_row_block_size * KL
        self.w_block_size = model_config.W_BLOCK_SIZE
        self.m_col_ty = np.ndarray[(8 * self.m + 2,), np.dtype[bfloat16]]
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_col m: {self.m}")
        comment_print(f"proj_assemble_col w_block_size: {self.w_block_size}")

        self.y_buffer_0 = aie.buffer(self.mt, self.m_col_ty, name = f"y_buffer_0_{self.mt_row}_{self.mt_col}")
        self.y_buffer_1 = aie.buffer(self.mt, self.m_col_ty, name = f"y_buffer_1_{self.mt_row}_{self.mt_col}")

        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.y_prod_lock_0 = aie.lock(self.mt, lock_id = 0, init = 2)
        self.y_prod_lock_1 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.y_prod_lock_2 = aie.lock(self.mt, lock_id = 2, init = 0)
        self.y_prod_lock_3 = aie.lock(self.mt, lock_id = 3, init = 0)
        self.y_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 8, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 9, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        self.__call__()

    @property
    def y_in(self):
        return [self.S2MM[i] for i in range(4)]
    
    @property
    def y_out(self):
        return self.MM2S[4]
    
    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]
    
    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 2 * self.m + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 2 * self.m + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[1])
                
            with block[3]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])
            
            with block[4]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 2 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[5])
                
            with block[5]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 2 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[4])
                
            with block[6]:
                s2 = aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])
            
            with block[7]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_0, offset = 4 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_1, offset = 4 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_0, offset = 6 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_1, offset = 6 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_cons_lock, 1)
                aie.next_bd(block[10])
                
            with block[12]:
                s4 = aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[13], chain = block[15])

            with block[13]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[14])
                
            with block[14]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[13])

            with block[15]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[16], chain = block[18])

            with block[16]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[17])

            with block[17]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[16])

            with block[18]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[19], chain = block[21])

            with block[19]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[20])
                
            with block[20]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[19])

            with block[21]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[22], chain = block[24])

            with block[22]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[23])

            with block[23]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[22])
                
            with block[24]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[25], chain = block[27])

            with block[25]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[26])

            with block[26]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[25])

            with block[27]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 4, dest = block[28], chain = block[30])

            with block[28]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.y_prod_lock_0, 1)
                aie.next_bd(block[29])

            with block[29]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.y_prod_lock_0, 1)
                aie.next_bd(block[28])

            with block[30]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[31], chain = block[33])
            
            with block[31]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[32])
                
            with block[32]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[31])

            with block[33]:
                aie.EndOp()
    
class proj_assemble_weight_only(AIE_MT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_col at ({tile.row}, {tile.col})")

        self.w_block_size = model_config.W_BLOCK_SIZE
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_col w_block_size: {self.w_block_size}")


        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 8, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 9, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        
        self.__call__()

    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]
    
    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]
    


    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[13], chain = block[15])

            with block[13]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[14])
                
            with block[14]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[13])

            with block[15]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[16], chain = block[18])

            with block[16]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[17])

            with block[17]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[16])

            with block[18]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[19], chain = block[21])

            with block[19]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[20])
                
            with block[20]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[19])

            with block[21]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[22], chain = block[24])

            with block[22]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[23])

            with block[23]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[22])
                
            with block[24]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[25], chain = block[30])

            with block[25]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[26])

            with block[26]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[25])

            with block[30]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[31], chain = block[33])
            
            with block[31]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[32])
                
            with block[32]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[31])


            with block[33]:
                aie.EndOp()

class proj_assemble_weight_x(AIE_MT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, replay_x: bool = False):
        """replay_x: x arrives from proj_x_replay_mt (one slice of KL tokens per FIFO half, merged under one packet id)
        instead of being produced chunk by chunk by the rms/attention/glu producers (two slices per half)."""
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_weight_x at ({tile.row}, {tile.col}) replay_x={replay_x}")

        self.w_block_size = model_config.W_BLOCK_SIZE
        self.m = model_config.q4nx_row_block_size * KL
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_col m: {self.m}")
        comment_print(f"proj_assemble_col w_block_size: {self.w_block_size}")

        self.x_chunk_ty = np.ndarray[(model_config.x_slice_size * KL * (1 if replay_x else 2),), np.dtype[bfloat16]]
        self.m_full_ty = np.ndarray[(16 * self.m + 2,), np.dtype[bfloat16]]
        self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_0_{self.mt_row}_{self.mt_col}")
        self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_1_{self.mt_row}_{self.mt_col}")
        self.x_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_chunk_ty, name = f"x_buffer_0_{self.mt_row}_{self.mt_col}")
        self.x_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_chunk_ty, name = f"x_buffer_1_{self.mt_row}_{self.mt_col}")


        self.m_prod_lock_0 = aie.lock(self.mt, lock_id = 0, init = 2)
        self.m_prod_lock_1 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.m_cons_lock = aie.lock(self.mt, lock_id = 2, init = 0)
        self.x_prod_lock = aie.lock(self.mt, lock_id = 3, init = 2)
        self.x_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)

        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 8, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 9, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        
        self.__call__()

    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]
    
    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]
    
    @property
    def x_out(self):
        return self.MM2S[4]
    
    @property
    def x_in(self):
        return self.S2MM[3]
    

    @property
    def y_out(self):
        return self.MM2S[5]
    
    @property
    def y_in(self):
        return [self.S2MM[i] for i in range(2)]

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            chain_id = 0
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[chain_id + 1], chain = block[chain_id + 3])
            with block[chain_id + 1]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 8 * self.m + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 8 * self.m + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 8 * self.m + 2, len = 8 * self.m)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 8 * self.m + 2, len = 8 * self.m)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 2])
                
            with block[chain_id + 2]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[chain_id + 1], chain = block[chain_id + 3])
            
            with block[chain_id + 1]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 2])
                
            with block[chain_id + 2]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 5, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.m_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.m_prod_lock_0, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.m_prod_lock_0, 1)
                aie.next_bd(block[chain_id + 1])


            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 4, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[chain_id + 1])


            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])
                
            with block[chain_id + 2]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 1])
                
            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.EndOp()
    
class x_replay_phase:
    """One activation the projection engine reduces over and how many times it is re-read (replayed to all 16 cores).

    `replays` is the number of broadcasts of the whole KL x `x_dim` activation: the total of the rounds (M / (16 * 32)) the
    cores run for that projection, drained rounds included (rounds a layer variant does not need are consumed without weights).
    Phases are time-multiplexed on one shared staging buffer unless `dedicated`: a phase that arrives while an earlier one is still
    replaying (down: its input is produced by the up|gate mvms) or that only some layer variants run (o_global / o_swa) gets a buffer
    and a landing channel of its own.
    """
    SHARED = "main"

    def __init__(self, x_dim: int, replays: int, name: str, dedicated: bool = False):
        self.x_dim = x_dim
        self.replays = replays
        self.name = name
        self.dedicated = dedicated
        self.buffer = name if dedicated else self.SHARED


class proj_x_replay_mt(AIE_MT_IP):
    """Holds every x the 16 projection cores reduce over and replays it, one phase at a time (D17, plan S6.4).

    Landing: x arrives token-major `[t][x_dim]` as one packet per phase and is stored as is. A replay reads it back slice by slice in the
    8-token core's operand layout `[x_dim / 256][kt = 32][t][k8 = 8]` (kernels/q4_l8.h) with one 4-dim read BD.

    Replay: a cycle of `fanout` identical linear BDs per distinct (buffer, width), driven only by lock counts: BD k waits for its
    consumer lock, sends the whole buffer as one packet under `x_pkt_id`, and the last BD of the cycle credits the producer lock. The
    landing BD of a phase takes the previous phase's credits (the buffer hand-over: it cannot complete before the previous phase's last
    replay has left), and releases its own `replays / fanout` credits to the replay channel. All lock counts stay below 64.

    The replay channels packet-merge under one id into main_mt's x FIFO (the broadcast tile): that FIFO is the only source of x for the
    16 cores (a circuit flow, one multicast tree). The FIFO is on a neighbouring tile because the router cannot do a zero-hop packet flow.
    """
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, phases: List[x_replay_phase],
                 x_pkt_id: int = proj_x_pkt_id.from_replay):
        super().__init__(tile)
        comment_print(f"Initializing proj_x_replay_mt at ({tile.row}, {tile.col}) KL={KL}")
        self.x_slice_size = model_config.x_slice_size
        self.phases = phases
        self.x_pkt_id = x_pkt_id
        assert len({p.name for p in phases}) == len(phases)
        self.stages = {}
        for p in phases:
            assert p.x_dim % self.x_slice_size == 0, f"{p.name}: x_dim {p.x_dim} is not a whole number of x slices"
            st = self.stages.setdefault(p.buffer, {"phases": [], "elems": 0})
            st["phases"].append(p)
            st["elems"] = max(st["elems"], KL * p.x_dim)
        self.replays = {}
        for p in phases:
            g = self.replays.setdefault((p.buffer, p.x_dim), {"phases": [], "x_dim": p.x_dim})
            g["phases"].append(p)
            p.chan = g
        for (name, x_dim), g in self.replays.items():
            g["fanout"] = self._pick_fanout([q.replays for q in g["phases"]], f"{name}/{x_dim}")
            g["stage"] = self.stages[name]
            for p in g["phases"]:
                p.credits = p.replays // g["fanout"]
        s2mm_ch, mm2s_ch, lock_id = 0, 0, 0
        for name, st in self.stages.items():
            st["buffer"] = ctx.buffer(tile = self.located_tile, dtype = np.ndarray[(st["elems"],), np.dtype[bfloat16]],
                                      name = f"x_buffer_{name}_{self.mt_row}_{self.mt_col}")
            # owes the last phase's credits: the first phase's landing BD acquires them, and the loop is closed
            st["prod_lock"] = aie.lock(self.mt, lock_id = lock_id, init = st["phases"][-1].credits)
            st["s2mm_ch"] = s2mm_ch
            lock_id += 1
            s2mm_ch += 1
            comment_print(f"  buffer {name}: {st['elems'] * 2 // 1024} KiB, phases {[p.name for p in st['phases']]}")
        for g in self.replays.values():
            g["mm2s_ch"] = mm2s_ch
            g["cons_locks"] = [aie.lock(self.mt, lock_id = lock_id + i, init = 0) for i in range(g["fanout"])]
            lock_id += g["fanout"]
            mm2s_ch += 1
            for p in g["phases"]:
                comment_print(f"  {p.name}: x_dim {p.x_dim}, {p.replays} replays = {g['fanout']} x {p.credits}")
        assert s2mm_ch <= 6 and mm2s_ch <= 6, "out of memtile DMA channels"
        assert lock_id <= 64, "out of memtile locks"
        self.__call__()

    @staticmethod
    def _pick_fanout(counts: List[int], where: str) -> int:
        g = 0
        for c in counts:
            g = gcd(g, c)
        for f in range(1, g + 1):
            if g % f == 0 and max(counts) // f <= 63:
                return f
        raise ValueError(f"{where}: replays {counts} do not split into a cycle short enough for a 63-count lock")

    def x_in(self, name: str) -> dma_port:
        """The landing port of the phase called `name` (phases sharing a buffer share the port, told apart by arrival order)."""
        for p in self.phases:
            if p.name == name:
                return self.S2MM[self.stages[p.buffer]["s2mm_ch"]]
        raise KeyError(name)

    def route_to(self, fifo_in: dma_port):
        for g in self.replays.values():
            connect_port(self.MM2S[g["mm2s_ch"]], fifo_in, pkt_id = self.x_pkt_id, keep_pkt_header = False)

    def __call__(self):
        plan, n = [], 0
        for st in self.stages.values():
            plan.append(("land", aie.DMAChannelDir.S2MM, st["s2mm_ch"], st, n))
            n += 1 + len(st["phases"])
        for g in self.replays.values():
            plan.append(("replay", aie.DMAChannelDir.MM2S, g["mm2s_ch"], g, n))
            n += 1 + g["fanout"]
        end = n

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            for i, (kind, dir, ch, arg, start) in enumerate(plan):
                nxt = plan[i + 1][4] if i + 1 < len(plan) else end
                with block[start]:
                    aie.dma_start(dir, ch, dest = block[start + 1], chain = block[nxt])
                if kind == "land":
                    ph = arg["phases"]
                    for k, p in enumerate(ph):
                        with block[start + 1 + k]:
                            lock_acquire(arg["prod_lock"], ph[k - 1].credits)
                            # x lands token-major [t][x_dim], exactly as the producer sends it
                            aie.dma_bd(arg["buffer"], offset = 0, len = KL * p.x_dim)
                            lock_release(p.chan["cons_locks"][0], p.credits)
                            aie.next_bd(block[start + 1 + (k + 1) % len(ph)])
                else:
                    f = arg["fanout"]
                    for k in range(f):
                        with block[start + 1 + k]:
                            lock_acquire(arg["cons_locks"][k], 1)
                            aie.dma_bd_packet(0, self.x_pkt_id)
                            if KL == 1:
                                aie.dma_bd(arg["stage"]["buffer"], offset = 0, len = arg["x_dim"])
                            else:
                                # [t][x_dim] -> per 256-column slice j, [kt = 32][t][k8 = 8]: the 8-token core's mmul A-tile layout, so that
                                # the core's x S2MM BD is linear. The permute has to be on this read side: the same address pattern as a
                                # write (a 4-dim memtile S2MM on the landing BD, or a 3-dim tile S2MM on the core) hangs the DMA on
                                # hardware (measured 2026-10-29), reads with 16-byte runs are fine.
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


class y_depermute_mt(AIE_MT_IP):
    """The y of the projection engine, token-major (plan S6.5, first half).

    The engine emits one 512-row round at a time as a packet laid out `[b = 16 row blocks][t][32 rows]` (each tile writes its
    `[t][32]` block, see kernels/q4_l8.h). Consumers want per-token rows, e.g. the RoPE tile one head (512 / 256 rows) of one token at a
    time. This tile lands a round (S2MM 0, header stripped, ping/pong) and reads it back token-major `[t][b * 32 + r]` with one 3-dim
    read BD: a round of KL tokens x 512 rows, i.e. one global head (DH = 512) or two SWA heads (DH = 256) per token.
    Read-side permutes work on the memtile DMA; write-side ones hang it (iron/proj_tile.py proj_x_replay_mt).
    """
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing y_depermute_mt at ({tile.row}, {tile.col}) KL={KL}")
        self.rows = 16 * model_config.q4nx_row_block_size            # 512 rows per round
        self.round_ty = np.ndarray[(KL * self.rows,), np.dtype[bfloat16]]
        self.buf_0 = ctx.buffer(tile = self.located_tile, dtype = self.round_ty, name = f"y_stage_0_{self.mt_row}_{self.mt_col}")
        self.buf_1 = ctx.buffer(tile = self.located_tile, dtype = self.round_ty, name = f"y_stage_1_{self.mt_row}_{self.mt_col}")
        self.prod = aie.lock(self.mt, lock_id = 0, init = 2)
        self.cons = aie.lock(self.mt, lock_id = 1, init = 0)
        self.__call__()

    @property
    def y_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def y_out(self) -> dma_port:
        return self.MM2S[0]

    def __call__(self):
        n = KL * self.rows
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
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[4], chain = block[6])
            for k, buf in ((4, self.buf_0), (5, self.buf_1)):
                with block[k]:
                    lock_acquire(self.cons, 1)
                    if KL == 1:
                        aie.dma_bd(buf, offset = 0, len = n)
                    else:
                        # [b][t][32] -> [t][b][32]
                        aie.dma_bd(buf, offset = 0, len = n, dimensions = [(KL, 32), (16, 32 * KL), (32, 1)])
                    lock_release(self.prod, 1)
                    aie.next_bd(block[4 + (k - 4 + 1) % 2])
            with block[6]:
                aie.EndOp()


class q4_proj_engine:
    def __init__(self, 
        CTs: List[List[tile_with_location]], 
        MTs: List[tile_with_location], 
        ITs: List[ShimTile], 
        proj_core_stack_size: int,
        proj_core_link_with: str,
        ctx: mlir_context, 
        model_config: ModelConfig,
        replay_x: bool = False
    ):
        self.CTs = CTs
        self.MTs = MTs
        self.ITs = ITs
        self.proj_cores = []
        for col in range(len(CTs) // 2):
            cores_this_group = []
            for row in range(len(CTs[0]) // 2):
                comment_print(f"Initializing proj_core ({row}, {col})")
                cores_this_group.append(proj_combo(CTs[2 * col][row * 2 : row * 2 + 2], ctx, model_config, proj_core_stack_size, proj_core_link_with))
            
            for row in range(len(CTs[0]) // 2):
                comment_print(f"Initializing proj_core ({row}, {col})")
                cores_this_group.append(proj_combo(CTs[2 * col + 1][row * 2 : row * 2 + 2], ctx, model_config, proj_core_stack_size, proj_core_link_with))
            self.proj_cores.append(cores_this_group)

        self.pure_assemble_mts = []
        self.pure_assemble_mts.append(
            proj_assemble_col(MTs[0], ctx, model_config)
        )
        self.pure_assemble_mts.append(
            proj_assemble_col(MTs[2], ctx, model_config)
        )
        self.pure_weight_mt = proj_assemble_weight_only(MTs[3], ctx, model_config)

        self.main_mt = proj_assemble_weight_x(MTs[1], ctx, model_config, replay_x = replay_x)
        
        
    def route(self):
        # connect IT to MT
        for i in range(len(self.MTs)):
            connect_port(self.ITs[i].MM2S[0], self.MTs[i].S2MM[4])
            connect_port(self.ITs[i].MM2S[1], self.MTs[i].S2MM[5])

        # connect MT weight to all CTs
        for i in range(len(self.MTs)):
            for j in range(len(self.CTs[i])):
                connect_port(self.MTs[i].MM2S[j], self.CTs[i][j].S2MM[1])

        # connect x broadcast to all CTs
        for i in range(len(self.CTs)):
            for j in range(len(self.CTs[i])):
                connect_port(self.main_mt.x_out, self.CTs[i][j].S2MM[0])

        # collect y from two columns to each pure assemble mt
        for i in range(len(self.proj_cores)):
            for j in range(len(self.proj_cores[i])):
                if j == 0:
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = True)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = True)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_rms, keep_pkt_header = True)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = True)
                else:
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = False)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

        # collect y from pure assemble mts to main mt
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_rms, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = False)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

         # connect weight from pure weight mt to main mt
    @property
    def X_in(self):
        return self.main_mt.x_in
    
    @property
    def Y_out(self):
        return self.main_mt.y_out