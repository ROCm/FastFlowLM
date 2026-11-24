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

from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import connect_port
from utils import AIE_MT_IP

from constants import *
from iron.utils import comment_print

class aux_memtile(AIE_MT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.DH = model_config.head_dim
        # num repeats for o_projection and down_projection, the output dim is always hidden_size
        self.D = model_config.hidden_size
        self.DQ = model_config.DQ
        self.repeats = self.D // (model_config.q4nx_row_block_size * 16) # 16 is a constant of 16 CTs in projection engine

        self.DQ_PADDED = model_config.PADDED_DQ

        self.q_ty = np.ndarray[(self.DQ_PADDED,), self.dtype]
        self.out_ty = np.ndarray[(self.DQ,), self.dtype]

        self.intermediate_size = model_config.intermediate_size
        self.down_ty = np.ndarray[(self.intermediate_size,), self.dtype]
        comment_print(f"Initializing aux_memtile at ({self.mt_row}, {self.mt_col}) with D: {self.D}, DQ_PADDED: {self.DQ_PADDED}, intermediate_size: {self.intermediate_size}, repeats: {self.repeats}")

        self.down_prod_lock = aie.lock(self.mt, lock_id = 0, init = self.repeats)
        self.down_cons_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        self.q_prod_lock    = aie.lock(self.mt, lock_id = 2, init = 1)
        self.q_cons_lock_0  = aie.lock(self.mt, lock_id = 3, init = 0)
        self.q_cons_lock_1  = aie.lock(self.mt, lock_id = 4, init = 0)
      
        self.o_prod_lock_0  = aie.lock(self.mt, lock_id = 7, init = self.repeats)
        self.o_prod_lock_1  = aie.lock(self.mt, lock_id = 8, init = 0)
      
        self.o_cons_lock = aie.lock(self.mt, lock_id = 11, init = 0)

        self.q_buffer = aie.buffer(tile = self.mt, datatype = self.q_ty, name = f"q_buffer_{self.mt_row}_{self.mt_col}")
        self.o_buffer = aie.buffer(tile = self.mt, datatype = self.out_ty, name = f"o_buffer_{self.mt_row}_{self.mt_col}")

        self.down_buffer = aie.buffer(tile = self.mt, datatype = self.down_ty, name = f"down_buffer_{self.mt_row}_{self.mt_col}")


        # q is in shape (NQ PADDED, DH)
        self.Q_HEADS_PADDED = model_config.NUM_Q_HEADS_PADDED
        self.q_heads_len = self.DQ_PADDED // 2
        self.q_heads_per_cu = self.Q_HEADS_PADDED // 2  # assuming 2 CUs
        self.q_offset = self.q_heads_per_cu * self.DH
        self.q_out_order = [(self.DH // 8, 8), (self.q_heads_per_cu, self.DH), (8, 1)]
        comment_print(f"\taux_memtile q_offset: {self.q_offset}")   
        comment_print(f"\taux_memtile q_out_order: {self.q_out_order}")

        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])

            with block[1]:
                lock_acquire(self.down_prod_lock, self.repeats)
                aie.dma_bd(self.down_buffer)
                lock_release(self.down_cons_lock, self.repeats)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.down_prod_lock, self.repeats)
                aie.dma_bd(self.down_buffer)
                lock_release(self.down_cons_lock, self.repeats)
                aie.next_bd(block[1])


            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])

            with block[4]:
                lock_acquire(self.down_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_down_proj)
                aie.dma_bd(self.down_buffer)
                lock_release(self.down_prod_lock, 1)
                aie.next_bd(block[4])


            with block[6]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[7], chain = block[8])

            with block[7]:
                lock_acquire(self.q_prod_lock, 1)
                aie.dma_bd(self.q_buffer)
                lock_release(self.q_cons_lock_0, 1)
                aie.next_bd(block[7])

            with block[8]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[9], chain = block[10])

            with block[9]:
                lock_acquire(self.q_cons_lock_0, 1)
                aie.dma_bd(self.q_buffer, offset = 0, len = self.q_heads_len, dimensions = self.q_out_order)
                lock_release(self.q_cons_lock_1, 1)
                aie.next_bd(block[9])

            with block[10]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[11], chain = block[16])

            with block[11]:
                lock_acquire(self.q_cons_lock_1, 1)
                aie.dma_bd(self.q_buffer, offset = self.q_offset, len = self.q_heads_len, dimensions = self.q_out_order)
                lock_release(self.q_prod_lock, 1)
                aie.next_bd(block[11])

            with block[16]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest = block[17], chain = block[18])

            with block[17]:
                lock_acquire(self.o_prod_lock_0, self.repeats)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.DQ // 2)
                lock_release(self.o_prod_lock_1, 1)
                aie.next_bd(block[17])

            with block[18]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest = block[19], chain = block[24])

            with block[19]:
                lock_acquire(self.o_prod_lock_1, 1)
                aie.dma_bd(self.o_buffer, offset = self.DQ // 2, len = self.DQ // 2)
                lock_release(self.o_cons_lock, self.repeats)
                aie.next_bd(block[19])

            with block[24]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest = block[25], chain = block[26])

            with block[25]:
                lock_acquire(self.o_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_out_proj)
                aie.dma_bd(self.o_buffer)
                lock_release(self.o_prod_lock_0, 1)
                aie.next_bd(block[25])

            with block[26]:
                aie.EndOp()

    @property
    def O_out(self) -> dma_port:
        return self.MM2S[5]
    
    @property
    def down_out(self) -> dma_port:
        return self.MM2S[0]
    
    @property
    def Q_in(self) -> dma_port:
        return self.S2MM[1]
    
    @property
    def down_in(self) -> dma_port:
        return self.S2MM[0]
    
    @property
    def Q_out(self) -> List[dma_port]:
        return [self.MM2S[i] for i in range(1, 3)]
    
    @property
    def O_in(self) -> List[dma_port]:
        return [self.S2MM[i] for i in range(2, 4)]
            
            
            
            
            
            
            
            
