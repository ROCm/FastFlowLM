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


class proj_layer_embedding(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "proj_layer_embedding.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.enable_tb = False
        
        self.D = model_config.D
        self.layer_D = model_config.per_layer_input_dim
        self.M = model_config.bf16_w_row_block_size
        self.N = model_config.bf16_w_col_block_size
        self.x_slice_size = model_config.bf16_x_slice_size
        self.w_block_size = model_config.bf16_w_row_block_size * model_config.bf16_w_col_block_size
        self.w_block_ty = np.ndarray[(self.w_block_size,), self.dtype]
        
        self.norm_w_ty = np.ndarray[(self.layer_D + self.D + 32,), self.dtype]
        # x0 stands for always from layer 0
        self.x0_per_layer_ty = np.ndarray[(self.layer_D,), self.dtype]
        self.x0_ty = np.ndarray[(self.D,), self.dtype]
        self.y_ty = np.ndarray[(self.layer_D + self.D + 32,), self.dtype]

        self.norm_w_ty_buffer = aie.buffer(tile = self.ct, datatype = self.norm_w_ty, name = f"norm_w_{self.ct_row}_{self.ct_col}")
        self.x0_per_layer_buffer = aie.buffer(tile = self.ct, datatype = self.x0_per_layer_ty, name = f"x0_per_layer_{self.ct_row}_{self.ct_col}")
        self.x0_buffer = aie.buffer(tile = self.ct, datatype = self.x0_ty, name = f"x0_{self.ct_row}_{self.ct_col}")

        self.x_proj_buffer = aie.buffer(tile = self.ct, datatype = self.x0_per_layer_ty, name = f"x_proj_{self.ct_row}_{self.ct_col}")
        self.y_buffer = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_{self.ct_row}_{self.ct_col}")
        self.proj_w_buffer_0 = aie.buffer(tile = self.ct, datatype = self.w_block_ty, name = f"proj_w_0_{self.ct_row}_{self.ct_col}")
        self.proj_w_buffer_1 = aie.buffer(tile = self.ct, datatype = self.w_block_ty, name = f"proj_w_1_{self.ct_row}_{self.ct_col}")

        self.norm_w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 0)  
        self.x0_per_layer_prod_lock = aie.lock(self.ct, lock_id = 1, init = 1)  
        self.x0_prod_lock = aie.lock(self.ct, lock_id = 2, init = 0)
        self.xw_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.proj_w_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.proj_w_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        self.y_prod_lock = aie.lock(self.ct, lock_id = 6, init = 1)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)

        self.norm_w_p2_prod_lock = aie.lock(self.ct, lock_id = 8, init = 0)

        self.proj = ctx.func("proj_layer_embedding", [self.norm_w_ty, self.x0_per_layer_ty, self.x0_ty, self.x0_per_layer_ty, self.y_ty, self.w_block_ty, self.w_block_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.proj(self.norm_w_ty_buffer, self.x0_per_layer_buffer, self.x0_buffer, self.x_proj_buffer, self.y_buffer, self.proj_w_buffer_0, self.proj_w_buffer_1)
        
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[6])
            with block[1]:
                lock_acquire(self.x0_per_layer_prod_lock, 1)
                aie.dma_bd(self.x0_per_layer_buffer)
                lock_release(self.norm_w_prod_lock, 1)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(self.norm_w_prod_lock, 1)
                aie.dma_bd(self.norm_w_ty_buffer, offset = 0, len = self.layer_D)
                lock_release(self.norm_w_p2_prod_lock, 1)
                aie.next_bd(block[3])

            with block[3]:
                lock_acquire(self.norm_w_p2_prod_lock, 1)
                aie.dma_bd(self.norm_w_ty_buffer, offset = self.layer_D, len = self.D + 32)
                lock_release(self.x0_prod_lock, 1)
                aie.next_bd(block[4])
            
            with block[4]:
                lock_acquire(self.x0_prod_lock, 1)
                aie.dma_bd(self.x0_buffer)
                lock_release(self.xw_cons_lock, 1)
                aie.next_bd(block[1])

            with block[6]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[7], chain = block[9])

            with block[7]:
                lock_acquire(self.proj_w_prod_lock, 1)
                aie.dma_bd(self.proj_w_buffer_0)
                lock_release(self.proj_w_cons_lock, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.proj_w_prod_lock, 1)
                aie.dma_bd(self.proj_w_buffer_1)
                lock_release(self.proj_w_cons_lock, 1)
                aie.next_bd(block[7])
            
            with block[9]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[10])

            with block[12]:
                aie.EndOp()

    @property
    def XXW_In(self) -> dma_port:
        return self.S2MM[0]

    @property
    def PROJ_W_In(self) -> dma_port:
        return self.S2MM[1]
    
    @property
    def Y_Out(self) -> dma_port:
        return self.MM2S[1]
    