# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from constants import kv_out_pkt_id

from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP

class DeltaNet_Norm(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "delta_norm.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing DeltaNet_Norm at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.DV = model_config.linear_DV
        self.NORM_DIM = model_config.linear_v_head_dim
        self.V_ty = np.ndarray[(self.DV,), self.dtype]
        self.W_ty = np.ndarray[(self.NORM_DIM,), self.dtype]
    
        self.v_buffer = aie.buffer(tile = self.ct, datatype = self.V_ty, name = f"V_buffer_{self.ct_row}_{self.ct_col}")
        self.y_buffer = aie.buffer(tile = self.ct, datatype = self.V_ty, name = f"Y_buffer_{self.ct_row}_{self.ct_col}")
        self.w_buffer = aie.buffer(tile = self.ct, datatype = self.W_ty, name = f"W_buffer_{self.ct_row}_{self.ct_col}")
        self.v_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)
        self.v_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        self.delta_norm = ctx.func("delta_net_norm", [self.V_ty, self.V_ty, self.W_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.delta_norm(self.y_buffer, self.v_buffer, self.w_buffer)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[3], chain = block[9])

            with block[3]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd_packet(0, 7)
                aie.dma_bd(self.y_buffer)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[3])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[10], chain = block[11])

            with block[10]:
                lock_acquire(self.v_prod_lock, 1)
                aie.dma_bd(self.v_buffer)
                lock_release(self.v_cons_lock, 1)
                aie.next_bd(block[10])

            with block[11]:
                aie.EndOp()

    @property
    def DH_in(self):
        return self.S2MM[1]
    
    @property
    def W_in(self):
        return self.S2MM[0]
    
    @property
    def DH_out(self):
        return self.MM2S[0]