# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig

import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

from utils import comment_print, mlir_context, tile_with_location, lock_acquire, lock_release, AIE_CT_IP



class rms_residual_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "rms_residual.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing rms_residual_tile at ({self.ct_row}, {self.ct_col})")
        
        self.model_config = model_config
        self.x_ty = np.ndarray[(model_config.D,), self.dtype]
        # [input_layernorm | post_attention_layernorm]
        self.w_ty = np.ndarray[(model_config.D*2,), self.dtype]
        self.y_pkt_ty = np.ndarray[(model_config.D + 16,), self.dtype]

        self.D = model_config.D
        comment_print(f"rms_residual_tile D: {self.D}")

        self.x_ping = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_ping_{self.ct_row}_{self.ct_col}")
        self.x_pong = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_pong_{self.ct_row}_{self.ct_col}")
        self.x_buf = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"x_buf_{self.ct_row}_{self.ct_col}")
        self.w_buffer = ctx.buffer(tile = self.located_tile, dtype = self.w_ty, name = f"w_buffer_{self.ct_row}_{self.ct_col}")
        self.y = ctx.buffer(tile = self.located_tile, dtype = self.y_pkt_ty, name = f"y_{self.ct_row}_{self.ct_col}")
        self.w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 2, init = 0)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.x_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.core_func = ctx.func("rms_residual", [self.y_pkt_ty, self.x_ty, self.x_ty, self.w_ty, self.x_ty])

        # create mlir code
        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.core_func(self.y, self.x_ping, self.x_pong, self.w_buffer, self.x_buf)
        
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])
            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_ping, offset = 0, len = self.D)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_pong, offset = 0, len = self.D)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[3], chain=block[5])
            
            with block[3]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer, offset = 0, len = self.D*2)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[3])


            with block[5]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[6], chain = block[10])

            # y + 14 holds the packet id the kernel picked, so start the send there
            with block[6]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y, offset = 14, len = self.D + 2)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[6])

            with block[10]:
                aie.EndOp()

    @property
    def Y_out(self):
        return self.MM2S[0]

    @property
    def X_in(self):
        return self.S2MM[0]
    
    @property
    def W_in(self):
        return self.S2MM[1]
