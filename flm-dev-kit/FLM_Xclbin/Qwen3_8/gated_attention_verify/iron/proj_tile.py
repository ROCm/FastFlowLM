# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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
from math import gcd
from utils import mlir_context, tile_with_location, connect_dma, dma_port, connect_port, lock_acquire, lock_release, comment_print, ShimTile
from utils import AIE_CT_IP, AIE_MT_IP

from utils import ENABLE_CHESSCC
from constants import proj_pkt_id, proj_x_pkt_id, L_TOKENS

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
        
        self.m = model_config.q4nx_row_block_size
        self.x_slice_size = model_config.x_slice_size
        # Payload one CT contributes per row block: the same 32 rows for all
        # L_TOKENS queries. Every y offset and length below is in these units.
        self.m_chunk = self.m * L_TOKENS

        # One column block of x for all L_TOKENS rows, [token][col].
        self.x_slice_ty = np.ndarray[(self.x_slice_size * L_TOKENS,), np.dtype[bfloat16]]
        self.linear_w_ty = np.ndarray[(model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        self.m_pkt_ty = np.ndarray[(2 * self.m_chunk + 16,), np.dtype[bfloat16]]

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
        
        proj_args = [self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, np.int32]
        self.extra_args = []
        if not ENABLE_CHESSCC:
            # Peano only: the column-sum cache of proj_main.cc, sized as there,
            # as two buffers of this tile. Peano links the kernel's statics into
            # one region, which has no room for it.
            col_block = model_config.q4nx_col_block_size
            max_k = max(model_config.hidden_size, model_config.intermediate_size,
                        getattr(model_config, "linear_DV", 0), getattr(model_config, "DQ", 0))
            limbs = 2 * L_TOKENS * col_block // model_config.q4nx_group_size
            blocks = max_k // col_block
            lo_ty = np.ndarray[((blocks // 2) * limbs,), np.dtype[bfloat16]]
            hi_ty = np.ndarray[((blocks - blocks // 2) * limbs,), np.dtype[bfloat16]]
            self.extra_args = [ctx.buffer(tile = self.located_tile, dtype = lo_ty, name = f"bcol_lo_{self.ct_row}_{self.ct_col}"),
                               ctx.buffer(tile = self.located_tile, dtype = hi_ty, name = f"bcol_hi_{self.ct_row}_{self.ct_col}")]
            proj_args += [lo_ty, hi_ty]
        self.proj = ctx.func("proj_main", proj_args)

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                if self.send_x_out:
                    self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, constant(1), *self.extra_args)
                else:
                    self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, constant(0), *self.extra_args)
        
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
                    aie.dma_bd(self.y_buffer_0, offset = 14, len = 2 * self.m_chunk + 2)
                    lock_release(self.y_prod_ping_lock, 2)
                    aie.next_bd(block[8])


                with block[8]: # x, to mt
                    lock_acquire(self.y_cons_pong_lock, 2)
                    aie.dma_bd(self.y_buffer_1, offset = 14, len = 2 * self.m_chunk + 2)
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

        self.m = model_config.q4nx_row_block_size
        self.m_chunk = self.m * L_TOKENS
        self.w_block_size = model_config.W_BLOCK_SIZE
        # 4 combos x 2 CTs, each contributing m_chunk, plus one packet header
        self.m_col_ty = np.ndarray[(8 * self.m_chunk + 2,), np.dtype[bfloat16]]
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
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 2 * self.m_chunk + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 2 * self.m_chunk + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[1])
                
            with block[3]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])
            
            with block[4]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 2 * self.m_chunk + 2, len = 2 * self.m_chunk)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[5])
                
            with block[5]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 2 * self.m_chunk + 2, len = 2 * self.m_chunk)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[4])
                
            with block[6]:
                s2 = aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])
            
            with block[7]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_0, offset = 4 * self.m_chunk + 2, len = 2 * self.m_chunk)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_1, offset = 4 * self.m_chunk + 2, len = 2 * self.m_chunk)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_0, offset = 6 * self.m_chunk + 2, len = 2 * self.m_chunk)
                lock_release(self.y_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_1, offset = 6 * self.m_chunk + 2, len = 2 * self.m_chunk)
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
    """The top of the y tree, this column's share of the weights, and the FIFO
    that x is broadcast out of.

    Same weight path as proj_assemble_col; the difference is the y side, which
    takes the two sub-trees rather than four cores and emits the assembled
    packet to DDR.

    It used to hold the whole activation -- L_TOKENS * D = 80 KiB, written once
    by the rms tile and replayed to the 16 cores -- but every x the engine
    reduces over lives on proj_x_replay_mt now, and it replays them. What is
    left here is 4 KiB of rate matching: proj_x_replay_mt's replay channels
    packet-merge into `x_in`, and `x_out` broadcasts what lands, circuit-
    switched, as the design's only source of x.

    The FIFO is here rather than on the replay tile because the pathfinder
    cannot route a packet flow whose source and destination are the same
    switchbox -- a zero-hop route leaves the destination's incoming port unset,
    and the rule comes out on South : 0, which nothing drives. (Except when the
    source and destination channel indices happen to be equal, where the
    fallback coincides with the right answer and hides it.) So the merge needs a
    neighbouring tile, and this is the neighbour: it was already broadcasting x
    to these 16 cores, so the outbound leg is unchanged.
    """
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_weight_x at ({tile.row}, {tile.col})")

        self.w_block_size = model_config.W_BLOCK_SIZE
        self.m = model_config.q4nx_row_block_size
        self.m_chunk = self.m * L_TOKENS
        self.D = model_config.D
        self.x_slice_size = model_config.x_slice_size
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_weight_x m: {self.m}")
        comment_print(f"proj_assemble_weight_x w_block_size: {self.w_block_size}")

        self.m_full_ty = np.ndarray[(16 * self.m_chunk + 2,), np.dtype[bfloat16]]
        self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_0_{self.mt_row}_{self.mt_col}")
        self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_1_{self.mt_row}_{self.mt_col}")

        self.m_prod_lock_0 = aie.lock(self.mt, lock_id = 0, init = 2)
        self.m_prod_lock_1 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.m_cons_lock = aie.lock(self.mt, lock_id = 2, init = 0)

        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 3, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 4, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 5, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 6, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 8, init = 0)

        # Exactly one CT x buffer per half, so a slice can be filling while the
        # previous one drains, and the halves stay in step whatever the phase --
        # every replay length is a whole number of slices. Nothing to gain from
        # more: this is a rate matcher between one sender and one broadcast, not
        # a reorder, and the cores chop the stream up with their own locks.
        self.x_slice_ty = np.ndarray[(L_TOKENS * self.x_slice_size,), np.dtype[bfloat16]]
        self.x_fifo_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_fifo_0_{self.mt_row}_{self.mt_col}")
        self.x_fifo_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_slice_ty, name = f"x_fifo_1_{self.mt_row}_{self.mt_col}")
        self.x_prod_lock = aie.lock(self.mt, lock_id = 9, init = 2)
        self.x_cons_lock = aie.lock(self.mt, lock_id = 10, init = 0)

        self.__call__()

    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]

    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]

    @property
    def x_in(self):
        """Where proj_x_replay_mt's replay channels merge, under one id."""
        return self.S2MM[3]

    @property
    def x_out(self):
        """The one port in the design that sends x."""
        return self.MM2S[4]

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
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 8 * self.m_chunk + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 8 * self.m_chunk + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 8 * self.m_chunk + 2, len = 8 * self.m_chunk)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 8 * self.m_chunk + 2, len = 8 * self.m_chunk)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[chain_id + 1])

            # The x FIFO. Both sides walk the same two slices in the same order,
            # so a replay's packet -- which spans many of them, and whose length
            # differs per phase -- comes out as one contiguous stream.
            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_fifo_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_fifo_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 4, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_fifo_0)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_fifo_1)
                lock_release(self.x_prod_lock, 1)
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
    """One activation the projection engine reduces over, and how many times it
    re-reads it.

    `replays` is the number of broadcasts of the whole L_TOKENS x `x_dim`
    activation -- the total of the M / m loop counts proj_main() runs for that
    projection.

    Phases are time-multiplexed by default: they share one staging buffer sized
    to the widest of them and take it over in list order, which is what the
    layer's data dependencies already enforce. Set `dedicated` only when a phase
    is *filling* while an earlier one is still replaying -- down is, because the
    SwiGLU result arrives during up|gate -- and it gets a buffer and an inbound
    channel to itself.

    `in_block` is the landing un-permute, and it is off by default because most
    producers here are already token-major: the rms tile emits one whole D-wide
    row per packet, the gate tile likewise. The GLU is not -- it inherits the
    projection engine's row-block permute, `raw[(rb * L + t) * m + r]`, because
    both its operands carry it and the kernel is elementwise -- so down sets
    `in_block = m` and the inbound BD un-permutes on the way in, the same
    3-dimensional shape aux_memtile runs for o and z. The replay side must see
    token-major [token][col]; that is what its own slicing assumes.

    Deliberately no packet id and no channel: the tile derives both. See
    proj_x_pkt_id in constants.py for why per-phase ids do not route.
    """
    # The shared lifetime every non-dedicated phase lands in. A dedicated phase
    # names its buffer after itself, so the two can never collide.
    SHARED = "main"

    def __init__(self, x_dim: int, replays: int, name: str = "x", dedicated: bool = False,
                 in_block: int = 0):
        self.x_dim = x_dim
        self.replays = replays
        self.name = name
        self.dedicated = dedicated
        self.in_block = in_block
        self.buffer = name if dedicated else self.SHARED
        assert in_block == 0 or x_dim % in_block == 0, \
            f"{name}: the landing un-permute walks whole row blocks, {x_dim} % {in_block} != 0"


#: The largest size a memtile BD dimension can hold, in bf16 elements. The field
#: is 10 bits, but a bf16 BD is lowered to 32-bit words with the sizes doubled,
#: so aie-opt rejects anything past 512 -- 513 already fails its "Size may not
#: exceed 1023" check.
BD_MAX_SIZE = 512


def _split_bd_size(n: int, where: str) -> tuple:
    """Factor `n` into (hi, lo) with lo <= BD_MAX_SIZE, for a count that has to
    be walked by two BD dimensions instead of one. hi = 1 means it fits already.
    Prefers the smallest hi, so the split is as shallow as the divisors allow."""
    if n <= BD_MAX_SIZE:
        return 1, n
    for hi in range(2, BD_MAX_SIZE + 1):
        if n % hi == 0 and n // hi <= BD_MAX_SIZE:
            return hi, n // hi
    raise ValueError(f"{where}: {n} does not factor into two BD dimensions of "
                     f"at most {BD_MAX_SIZE}")


class proj_x_replay_mt(AIE_MT_IP):
    """The one tile that holds x, and the only thing in the design that sends it.

    Every projection reduces over a different activation -- input_layernorm for
    qkv|z, the gated delta-net output for o, post_attention_layernorm for
    up|gate, the SwiGLU result for down -- and none of them fits a 64 KiB
    compute tile at L_TOKENS = 8, nor is any of them emitted more than once by
    the core that makes it. So they are all caught here, in a column the
    delta-net layer otherwise leaves idle, and replayed to the 16 mvm cores one
    phase at a time.

    **One staging buffer per lifetime, not one per phase.** The phases are
    strictly ordered by the layer's data dependencies, so by default they all
    share one buffer, sized to the widest of them, and hand it over through a
    single producer lock: the inbound BD of phase k+1 acquires exactly the
    credits phase k's replays produce, which cannot happen until the last of
    them has left. o cannot start landing until qkv|z's 32nd replay is out of
    the buffer. That is why the live phases cost L_TOKENS * max(D, linear_DV) =
    96 KiB rather than the sum, and why the lock state at the end of a layer is
    bit-for-bit the state at the start -- the whole tile is one closed loop.

    Only a phase that arrives *while an earlier one is still replaying* breaks
    that, and only down does: the SwiGLU result is produced by the up|gate mvms
    themselves, so the buffer is being written and read at once. It is the one
    phase marked `dedicated`, which buys it a buffer and an inbound channel of
    its own. Everything else is time-multiplexed and free.

    **Why the replay channels feed a FIFO instead of the cores.** A BD fires
    once per traversal of its chain and there is no per-BD repeat: dma_bd has no
    repeat field, dma_start's repeat_count repeats the whole chain, and a
    zero-stride dimension -- the other way to make one send repeat -- is
    rejected by DMABDOp::verify(). A replay channel is therefore a *cycle* of
    identical BDs driven purely by lock counts, and a cycle can never hand over
    to a differently shaped BD. So there is one replay channel per distinct
    activation width, and a layer's 32 + 10 + 68 + 10 = 120 sends could not come
    out of one port in any case: a memtile channel can only reach 24 of the 48
    BDs, since even channels take bd_id < 24 and odd ones the rest.

    Several ports broadcasting into S2MM[0] on 16 cores means several multicast
    trees, which is exactly what the arbiter allocator gets wrong (the failure
    is written out in constants.py's proj_x_pkt_id). So the replay channels do
    not reach the cores at all: they packet-merge under one id into the FIFO on
    proj_assemble_weight_x, and that tile's MM2S is the single port that feeds
    the cores -- circuit-switched, no id, no mask/match rule, one tree. The
    phases are indistinguishable on that last hop because the core side chops
    the stream into L_TOKENS x x_slice_size buffers against its own ping/pong
    locks and never sees a BD boundary; every replay length is a whole number of
    those.

    **Why `fanout`.** The inbound BD releases, in one lock op, as many ready
    tokens as the phase has replays, and the cycle walks them back one at a
    time. A memtile lock saturates at 63, and up|gate's 68 does not fit. A cycle
    of f BDs relays instead: BD i hands its token to BD i+1 and only the last
    credits the producer, so one trip round the cycle costs one token and emits
    f identical sends. f = 2 puts up|gate at 34. The BDs read the same buffer
    with the same shape, so the cores cannot tell which one they got, and as
    long as every phase on the channel is a multiple of f the cycle ends where
    it started. (The memtile analogue of the chunked release / acquire in
    Gemma4_12B_QAT/decoding/kernels/rms_residual.cc, which at L = 1 does the
    same thing in the rms kernel.)
    """
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig,
                 phases: List[x_replay_phase],
                 x_pkt_id: int = proj_x_pkt_id.from_replay):
        super().__init__(tile)
        comment_print(f"Initializing proj_x_replay_mt at ({tile.row}, {tile.col})")

        self.x_slice_size = model_config.x_slice_size
        self.phases = phases
        self.x_pkt_id = x_pkt_id

        # A staging buffer per lifetime -- the shared one plus one per dedicated
        # phase -- sized by its widest phase, with one inbound channel walking
        # that buffer's phases in layer order.
        assert len({p.name for p in phases}) == len(phases), \
            "phase names must be unique: a dedicated phase's buffer is named after it"
        self.stages = {}
        for p in phases:
            assert p.x_dim % self.x_slice_size == 0, \
                f"{p.name}: x_dim {p.x_dim} is not a whole number of x slices"
            st = self.stages.setdefault(p.buffer, {"phases": [], "elems": 0})
            st["phases"].append(p)
            st["elems"] = max(st["elems"], L_TOKENS * p.x_dim)

        # A replay channel per distinct width in a buffer: same buffer, same BD,
        # so qkv|z and up|gate share one.
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
            st["buffer"] = ctx.buffer(
                tile = self.located_tile,
                dtype = np.ndarray[(st["elems"],), np.dtype[bfloat16]],
                name = f"x_buffer_{name}_{self.mt_row}_{self.mt_col}")
            # Starts out owing the last phase's credits, because that is what
            # the first phase's BD acquires -- the loop is cyclic, and at t = 0
            # the buffer is as free as it is between layers.
            st["prod_lock"] = aie.lock(self.mt, lock_id = lock_id,
                                       init = st["phases"][-1].credits)
            st["s2mm_ch"] = s2mm_ch
            lock_id += 1
            s2mm_ch += 1
            comment_print(f"  buffer {name}: {st['elems'] * 2 // 1024} KiB, "
                          f"phases {[p.name for p in st['phases']]}")

        for g in self.replays.values():
            g["mm2s_ch"] = mm2s_ch
            g["cons_locks"] = [aie.lock(self.mt, lock_id = lock_id + i, init = 0)
                               for i in range(g["fanout"])]
            lock_id += g["fanout"]
            mm2s_ch += 1
            for p in g["phases"]:
                comment_print(f"  {p.name}: x_dim {p.x_dim}, {p.replays} replays "
                              f"= {g['fanout']} x {p.credits}")

        assert s2mm_ch <= 6 and mm2s_ch <= 6, "out of memtile DMA channels"
        assert lock_id <= 64, "out of memtile locks"

        self.__call__()

    @staticmethod
    def _pick_fanout(counts: List[int], where: str) -> int:
        """The shortest replay cycle that keeps every lock op inside the 6-bit
        memtile counter. 63 rather than 64 so the count is still representable
        after the +1 a release makes momentarily.

        Every phase on the channel has to be a whole number of trips round the
        cycle, or it would leave the cycle mid-way and the next phase would
        start from the wrong BD."""
        g = 0
        for c in counts:
            g = gcd(g, c)
        for f in range(1, g + 1):
            if g % f == 0 and max(counts) // f <= 63:
                return f
        raise ValueError(f"{where}: replays {counts} do not split into a cycle "
                         f"short enough for a 63-count lock")

    def x_in(self, i: int = 0) -> dma_port:
        """The inbound port for phase `i`: the S2MM of the buffer it lands in.
        Phases sharing a buffer share the port and are told apart by arrival
        order, not by id."""
        return self.S2MM[self.stages[self.phases[i].buffer]["s2mm_ch"]]

    def route_to(self, fifo_in: dma_port):
        """The replay channels' one hop, into the broadcast tile's x FIFO.

        One id for all of them, so the router merges the channels into a single
        flow and the FIFO sees one stream to chop into slices. Per-channel ids
        would be per-channel trees, which is the failure in proj_x_pkt_id."""
        for g in self.replays.values():
            connect_port(self.MM2S[g["mm2s_ch"]], fifo_in,
                         pkt_id = self.x_pkt_id, keep_pkt_header = False)

    def __call__(self):
        # Lay the channels out first: each one is a dma_start block followed by
        # its BD blocks, and every start has to name the next start as its
        # chain, so the indices have to be known before anything is emitted.
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
                            # x lands as L_TOKENS packets of x_dim, one per
                            # token, as the producer makes them. Taking the
                            # previous phase's credits is the buffer hand-over:
                            # it cannot complete until that phase's last replay
                            # has been sent.
                            lock_acquire(arg["prod_lock"], ph[k - 1].credits)
                            if p.in_block:
                                # ...unless the producer is row-block permuted,
                                # in which case nothing lands until everything
                                # has: token 0's row is not complete until the
                                # last block arrives. Stream order is rb, t, r
                                # and the destination is [t][rb * in_block + r].
                                #
                                # rb is split over two dimensions when it does
                                # not fit one. The BD size field is 10 bits, but
                                # these sizes are in bf16 and the hardware
                                # counts 32-bit words, so the verifier's real
                                # ceiling is 512 elements, not 1023 -- down's
                                # 544 row blocks miss it by 32. hi is the outer
                                # factor, so it strides a whole lo-block run.
                                blocks = p.x_dim // p.in_block
                                hi, lo = _split_bd_size(blocks, f"{p.name}: row blocks")
                                dims = [(blocks, p.in_block)] if hi == 1 else \
                                       [(hi, lo * p.in_block), (lo, p.in_block)]
                                aie.dma_bd(arg["buffer"], offset = 0,
                                           len = L_TOKENS * p.x_dim,
                                           dimensions = dims + [(L_TOKENS, p.x_dim),
                                                                (p.in_block, 1)])
                            else:
                                aie.dma_bd(arg["buffer"], offset = 0, len = L_TOKENS * p.x_dim)
                            lock_release(p.chan["cons_locks"][0], p.credits)
                            aie.next_bd(block[start + 1 + (k + 1) % len(ph)])

                else:
                    f = arg["fanout"]
                    for k in range(f):
                        with block[start + 1 + k]:
                            lock_acquire(arg["cons_locks"][k], 1)
                            aie.dma_bd_packet(0, self.x_pkt_id)
                            # [token][col] out as column blocks: for each of the
                            # x_dim / x_slice_size blocks, all L_TOKENS rows of
                            # that block back to back, which is one CT x buffer.
                            aie.dma_bd(arg["stage"]["buffer"], offset = 0,
                                       len = L_TOKENS * arg["x_dim"],
                                       dimensions = [(arg["x_dim"] // self.x_slice_size, self.x_slice_size),
                                                     (L_TOKENS, arg["x_dim"]),
                                                     (self.x_slice_size, 1)])
                            # Only the last BD of the cycle credits the
                            # producer; the rest relay to the next one, so a
                            # trip costs one token and emits `fanout` sends.
                            if k + 1 < f:
                                lock_release(arg["cons_locks"][k + 1], 1)
                            else:
                                lock_release(arg["stage"]["prod_lock"], 1)
                            aie.next_bd(block[start + 1 + (k + 1) % f])

            with block[end]:
                aie.EndOp()


class q4_proj_engine:
    def __init__(self, 
        CTs: List[List[tile_with_location]], 
        MTs: List[tile_with_location], 
        ITs: List[ShimTile], 
        proj_core_stack_size: int,
        proj_core_link_with: str,
        ctx: mlir_context, 
        model_config: ModelConfig
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
        self.main_mt = proj_assemble_weight_x(MTs[1], ctx, model_config)

        # Packet ids the y tree must route, in the order proj_main() emits them.
        # Every id costs one mask/match rule on each slave port in the tree.
        #
        # The ids are one-hot, and they have to stay that way. The router widens
        # a rule's mask to cover the ids leaving a port in the same direction
        # without checking it against the other ids arriving there, so a shared
        # bit lets one id swallow another's packets -- it did, when to_glu was 15
        # and o's y matched to_glu's rule out of shim(1,0). Where no widening can
        # separate them the router refuses outright ("can lead to false packet id
        # match"), which is what MT[3]'s west port did with to_rms = 4 next to
        # to_aux = 5.
        #
        # So: ids are shared by destination, not by projection, and moving one
        # means re-reading the packet_rules on every port it traverses in
        # input_physical.mlir (`aie-opt --aie-create-pathfinder-flows`).
        # Every mvm the layer runs. Three of the four ids land on MT[3] -- qkv in
        # the staging buffer, z in the parking buffer, o *and down* in the
        # de-permute buffer -- which is the cheap case for the tree: no port they
        # traverse needs a new direction, so no rule has to be widened.
        #
        # to_glu is the one that leaves the column, and it is 16 precisely so
        # that it is the only id in the tree with bit 4 set: a widened mask that
        # admits it cannot admit any of the others. It was 15 once, and o's y
        # matched its rule out of shim(1,0) and went east instead.
        self.pkt_ids = [proj_pkt_id.to_rope,   # qkv     -> MT[3], staging
                        proj_pkt_id.to_aux,    # z       -> MT[3], parking
                        proj_pkt_id.to_rms,    # o, down -> MT[3], de-permuted
                        proj_pkt_id.to_glu]    # up|gate -> the GLU tile

    def route_x(self):
        """Broadcast x to S2MM[0] of all 16 proj cores.

        Circuit-switched: every activation the engine reduces over funnels
        through main_mt's one FIFO port, so there is nothing for a packet id to
        disambiguate at the cores and the 16 flows collapse into a single
        multicast tree with no mask/match rule at any hop. It was packet-based
        while two tiles sent x, and the arbiter allocator could not place the
        two trees -- see proj_x_pkt_id.
        """
        for i in range(len(self.CTs)):
            for j in range(len(self.CTs[i])):
                connect_port(self.main_mt.x_out, self.CTs[i][j].S2MM[0])

    def route(self):
        # connect IT to MT
        for i in range(len(self.MTs)):
            connect_port(self.ITs[i].MM2S[0], self.MTs[i].S2MM[4])
            connect_port(self.ITs[i].MM2S[1], self.MTs[i].S2MM[5])

        # connect MT weight to all CTs
        for i in range(len(self.MTs)):
            for j in range(len(self.CTs[i])):
                connect_port(self.MTs[i].MM2S[j], self.CTs[i][j].S2MM[1])

        # connect the x broadcast to all CTs. What feeds main_mt's FIFO is
        # proj_x_replay_mt, which the engine does not own; layer_aie.py wires
        # that leg up against X_in.
        self.route_x()

        # collect y from two columns to each pure assemble mt.
        #
        # Every leg carries every id, not just the one its final hop cares
        # about: proj_main() writes the header into the buffer at y + 14 rather
        # than letting a BD insert it, so the same word travels every hop and a
        # hop with no rule for it drops the packet.
        #
        # keep_pkt_header on the first contributor only, so its header lands at
        # the head of the memtile buffer and is re-emitted as the assembled
        # packet's.
        for pkt_id in self.pkt_ids:
            for i in range(len(self.proj_cores)):
                for j in range(len(self.proj_cores[i])):
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = pkt_id, keep_pkt_header = (j == 0))

            # collect y from pure assemble mts to main mt
            connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = pkt_id, keep_pkt_header = True)
            connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = pkt_id, keep_pkt_header = False)

         # connect weight from pure weight mt to main mt
    @property
    def X_in(self):
        """The x FIFO's inbound port -- where proj_x_replay_mt's replays land."""
        return self.main_mt.x_in

    @property
    def Y_out(self):
        return self.main_mt.y_out