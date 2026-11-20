# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig

import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

from utils import mlir_context, tile_with_location, connect_dma, dma_port, connect_port, lock_acquire, lock_release, comment_print, ShimTile
from utils import AIE_CT_IP, AIE_MT_IP

from constants import proj_pkt_id

from typing import List
# phase 1: qkv proj, D -> (D + D // 2) to RoPE

# phase 2: o_proj, D -> D to rms_residual

# phase 3: up/gate proj, D -> D * 8 to MT_GLU

# phase 4: down proj, D * 4 -> D, to rms_residual

# minimal output unit: (m, 1)

class proj_tile(AIE_CT_IP):


    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "proj.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing proj_tile at ({tile.row}, {tile.col})")

        self.m = model_config.q4nx_row_block_size
        self.x_slice_size = model_config.x_slice_size
        
        self.x_slice_ty = np.ndarray[(self.x_slice_size,), np.dtype[bfloat16]]
        self.linear_w_ty = np.ndarray[(model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        self.m_pkt_ty = np.ndarray[(self.m + 16,), np.dtype[bfloat16]]

        self.x_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_0_{self.ct_row}_{self.ct_col}")
        self.w_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.linear_w_ty, name = f"w_0_{self.ct_row}_{self.ct_col}")
        self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_1_{self.ct_row}_{self.ct_col}")
        self.w_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.linear_w_ty, name = f"w_1_{self.ct_row}_{self.ct_col}")
        self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_pkt_ty, name = f"y_1_{self.ct_row}_{self.ct_col}")
        self.L_buffer = ctx.buffer(f"RTP_PROJ_IS_ATTN_BUFFER_{self.ct_row}_{self.ct_col}", tile, ctx.data_tys["L_ty"], use_write_rtp = True)
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)
         
        self.proj = ctx.func("proj", [self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, ctx.L_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, self.L_buffer)
        
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

            with block[6]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[7], chain = block[9])
                
            with block[7]: # x, to mt
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0, offset = 14, len = self.m + 2)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[8])

            with block[8]: # w, to mt
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1, offset = 14, len = self.m + 2)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[7])

            with block[9]:
                aie.EndOp()

class proj_assemble_col(AIE_MT_IP):

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_col at ({tile.row}, {tile.col})")

        self.m = model_config.q4nx_row_block_size
        self.w_block_size = model_config.W_BLOCK_SIZE
        self.m_col_ty = np.ndarray[(4 * self.m + 2,), np.dtype[bfloat16]]
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


    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = self.m + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = self.m + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[1])
                
            with block[3]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])
            
            with block[4]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = self.m + 2, len = self.m)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[5])
                
            with block[5]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = self.m + 2, len = self.m)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[4])
                
            with block[6]:
                s2 = aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])
            
            with block[7]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_0, offset = 2 * self.m + 2, len = self.m)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_1, offset = 2 * self.m + 2, len = self.m)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_0, offset = 3 * self.m + 2, len = self.m)
                lock_release(self.y_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_1, offset = 3 * self.m + 2, len = self.m)
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
    @property
    def Y_out(self):
        return self.MM2S[4]
    

    
class proj_assemble(AIE_MT_IP):

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble mt at ({tile.row}, {tile.col})")

        self.m = model_config.q4nx_row_block_size
        comment_print(f"proj_assemble m: {self.m}")
        self.x_chunk_ty = np.ndarray[(model_config.x_slice_size * 2,), np.dtype[bfloat16]]
        self.m_full_ty = np.ndarray[(16 * self.m + 2,), np.dtype[bfloat16]]
        self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_0_{self.mt_row}_{self.mt_col}")
        self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_1_{self.mt_row}_{self.mt_col}")
        self.x_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_chunk_ty, name = f"x_buffer_0_{self.mt_row}_{self.mt_col}")
        self.x_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_chunk_ty, name = f"x_buffer_1_{self.mt_row}_{self.mt_col}")


        self.m_prod_lock_0 = aie.lock(self.mt, lock_id = 0, init = 2)
        self.m_prod_lock_1 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.m_prod_lock_2 = aie.lock(self.mt, lock_id = 2, init = 0)
        self.m_prod_lock_3 = aie.lock(self.mt, lock_id = 3, init = 0)
        self.m_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)
        self.x_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.x_cons_lock = aie.lock(self.mt, lock_id = 6, init = 0)
        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 4 * self.m + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 4 * self.m + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[1])

            with block[3]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])

            with block[4]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 4 * self.m + 2, len = 4 * self.m)
                lock_release(self.m_prod_lock_2, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 4 * self.m + 2, len = 4 * self.m)
                lock_release(self.m_prod_lock_2, 1)
                aie.next_bd(block[4])


            with block[6]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])

            with block[7]:
                lock_acquire(self.m_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_0, offset = 8 * self.m + 2, len = 4 * self.m)
                lock_release(self.m_prod_lock_3, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.m_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_1, offset = 8 * self.m + 2, len = 4 * self.m)
                lock_release(self.m_prod_lock_3, 1)
                aie.next_bd(block[7])


            with block[9]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.m_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_0, offset = 12 * self.m + 2, len = 4 * self.m)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.m_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_1, offset = 12 * self.m + 2, len = 4 * self.m)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[10])


            with block[12]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[13], chain = block[15])

            with block[13]:
                lock_acquire(self.m_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.m_prod_lock_0, 1)
                aie.next_bd(block[14])

            with block[14]:
                lock_acquire(self.m_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.m_prod_lock_0, 1)
                aie.next_bd(block[13])

            with block[15]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[16], chain = block[20])

            with block[16]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[17])

            with block[17]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[16])

            with block[20]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[21], chain = block[25])

            with block[21]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[22])

            with block[22]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[21])

            with block[25]:
                aie.EndOp()
    @property
    def X_in(self):
        return dma_port(self.located_tile, 4)
    
    @property
    def X_out(self):
        return dma_port(self.located_tile, 1)
    
    @property
    def Y_out(self):
        return self.MM2S[0]

class proj_column:
    def __init__(self, 
        CTs: List[tile_with_location], 
        MT: tile_with_location, 
        IT: ShimTile, 
        In: dma_port, 
        proj_core_stack_size: int,
        proj_core_link_with: str,
        ctx: mlir_context, 
        model_config: ModelConfig
    ):
        self.proj_cores = []
        for row in range(len(CTs)):
            self.proj_cores.append(proj_tile(CTs[row], ctx, model_config, proj_core_stack_size, proj_core_link_with))
        self.proj_assemble_mt = proj_assemble_col(MT, ctx, model_config)

        self.IT = IT
        self.In = In

    def route(self):
        # build connections within this column

        connect_port(self.IT.MM2S[0], self.proj_assemble_mt.S2MM[4])
        connect_port(self.IT.MM2S[1], self.proj_assemble_mt.S2MM[5])

        for row in range(len(self.proj_cores)):
            connect_port(self.In, self.proj_cores[row].S2MM[0])
            connect_port(self.proj_assemble_mt.MM2S[row], self.proj_cores[row].S2MM[1])

            if row == 0:
                keep_pkt_header = True
            else:
                keep_pkt_header = False
            connect_port(self.proj_cores[row].MM2S[0], self.proj_assemble_mt.S2MM[row], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = keep_pkt_header)
            connect_port(self.proj_cores[row].MM2S[0], self.proj_assemble_mt.S2MM[row], pkt_id = proj_pkt_id.to_rms_norm, keep_pkt_header = keep_pkt_header)
            connect_port(self.proj_cores[row].MM2S[0], self.proj_assemble_mt.S2MM[row], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = keep_pkt_header)

    @property
    def Y_out(self):
        return self.proj_assemble_mt.Y_out
    
class q4_proj_engine:
    def __init__(self, 
        CTs: List[List[tile_with_location]], 
        MTs: List[tile_with_location], 
        ITs: List[ShimTile], 
        assemble_mt: tile_with_location, 
        proj_core_stack_size: int,
        proj_core_link_with: str,
        ctx: mlir_context, 
        model_config: ModelConfig
    ):
        self.proj_assemble_mt = proj_assemble(assemble_mt, ctx, model_config)
        self.proj_columns = []
        for col in range(len(CTs)):
            comment_print(f"Initializing proj_column {col}")
            self.proj_columns.append(proj_column(CTs[col], MTs[col], ITs[col], self.proj_assemble_mt.X_out, proj_core_stack_size, proj_core_link_with, ctx, model_config))
        
    def route(self):
        # build connections between columns
        for col in range(len(self.proj_columns)):
            self.proj_columns[col].route()

        for col in range(len(self.proj_columns)):
            if (col == 0):
                keep_pkt_header = True
            else:
                keep_pkt_header = False
            connect_port(self.proj_columns[col].Y_out, self.proj_assemble_mt.S2MM[col], pkt_id = proj_pkt_id.to_rope    , keep_pkt_header = keep_pkt_header)
            connect_port(self.proj_columns[col].Y_out, self.proj_assemble_mt.S2MM[col], pkt_id = proj_pkt_id.to_rms_norm, keep_pkt_header = keep_pkt_header)
            connect_port(self.proj_columns[col].Y_out, self.proj_assemble_mt.S2MM[col], pkt_id = proj_pkt_id.to_glu     , keep_pkt_header = keep_pkt_header)
            

    @property
    def X_in(self):
        return self.proj_assemble_mt.X_in
    
    @property
    def Y_out(self):
        return self.proj_assemble_mt.Y_out