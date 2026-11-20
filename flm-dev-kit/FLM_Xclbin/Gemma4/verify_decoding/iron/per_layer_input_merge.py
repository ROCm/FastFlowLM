# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *

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
from typing import List
from utils import dma_port, mlir_context, tile_with_location, connect_dma, lock_acquire, lock_release
from utils import AIE_CT_IP, comment_print


class proj_layer_merge(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "proj_layer_embedding.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.enable_tb = False
        
        self.D = model_config.D
        self.layer_D = model_config.per_layer_input_dim

        self.dual_y_ty = np.ndarray[(2 * (self.layer_D + self.D) + 32,), self.dtype]
       
       
        self.y_buffer = aie.buffer(tile = self.ct, datatype = self.dual_y_ty, name = f"y_{self.ct_row}_{self.ct_col}")

        self.norm_w_i_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)  
        self.residual_gate_prod_lock = aie.lock(self.ct, lock_id = 1, init = 0)  
        self.cons_lock = aie.lock(self.ct, lock_id = 2, init = 0)

        self.__call__()

    def __call__(self):
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[6])
            with block[1]:
                lock_acquire(self.norm_w_i_prod_lock, 1)
                aie.dma_bd(self.y_buffer, offset = 0, len = self.D + self.layer_D + 32)
                lock_release(self.residual_gate_prod_lock, 1)
                aie.next_bd(block[1])

            with block[6]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[7], chain = block[9])

            with block[7]:
                lock_acquire(self.residual_gate_prod_lock, 1)
                aie.dma_bd(self.y_buffer, offset = self.D + self.layer_D + 32, len = self.D + self.layer_D)
                lock_release(self.cons_lock, 1)
                aie.next_bd(block[7])
            
            with block[9]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.cons_lock, 1)
                aie.dma_bd(self.y_buffer)
                lock_release(self.norm_w_i_prod_lock, 1)
                aie.next_bd(block[10])

            with block[12]:
                aie.EndOp()

    @property
    def NORM_I_IN(self) -> dma_port:
        return self.S2MM[0]

    @property
    def RES_GATE_IN(self) -> dma_port:
        return self.S2MM[1]
    
    @property
    def Y_Out(self) -> dma_port:
        return self.MM2S[0]


class proj_layer_up(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "per_layer_up.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.enable_tb = False
        
        self.D = model_config.D
        self.layer_D = model_config.per_layer_input_dim 
        self.M = model_config.bf16_w_row_block_size
        self.N = model_config.bf16_w_col_block_size
        self.x_slice_size = model_config.bf16_x_slice_size
        self.w_block_size = model_config.bf16_w_row_block_size * model_config.bf16_w_col_block_size
        self.w_block_ty = np.ndarray[(self.w_block_size,), self.dtype]

        self.dual_x_ty = np.ndarray[(2 * (self.layer_D + self.D) + 32,), self.dtype]
        self.y_ty = np.ndarray[(self.D,), self.dtype]
       
       
        self.x_buffer = aie.buffer(tile = self.ct, datatype = self.dual_x_ty, name = f"x_{self.ct_row}_{self.ct_col}")
        self.y_buffer = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_{self.ct_row}_{self.ct_col}")
        self.proj_w_buffer_0 = aie.buffer(tile = self.ct, datatype = self.w_block_ty, name = f"proj_w_0_{self.ct_row}_{self.ct_col}")
        self.proj_w_buffer_1 = aie.buffer(tile = self.ct, datatype = self.w_block_ty, name = f"proj_w_1_{self.ct_row}_{self.ct_col}")

        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)  
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)  
        self.proj_w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.proj_w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)


        self.proj = ctx.func("per_layer_up", [self.dual_x_ty, self.w_block_ty, self.w_block_ty, self.y_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.proj(self.x_buffer, self.proj_w_buffer_0, self.proj_w_buffer_1, self.y_buffer)

        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])
            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[3], chain=block[9])

            with block[3]:
                lock_acquire(self.proj_w_prod_lock, 1)
                aie.dma_bd(self.proj_w_buffer_0)
                lock_release(self.proj_w_cons_lock, 1)
                aie.next_bd(block[4])

            with block[4]:
                lock_acquire(self.proj_w_prod_lock, 1)
                aie.dma_bd(self.proj_w_buffer_1)
                lock_release(self.proj_w_cons_lock, 1)
                aie.next_bd(block[3])
            
            with block[9]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[10])

            with block[12]:
                aie.EndOp()

    @property
    def X_IN(self) -> dma_port:
        return self.S2MM[0]
    @property
    def W_IN(self) -> dma_port:
        return self.S2MM[1]


    @property
    def Y_Out(self) -> dma_port:
        return self.MM2S[0]
    