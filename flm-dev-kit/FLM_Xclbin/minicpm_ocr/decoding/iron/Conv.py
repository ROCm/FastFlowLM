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
from aie.extras.dialects.ext.arith import constant
from constants import kv_out_pkt_id

from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP

def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

class Conv(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    
    def __init__(self, tile: tile_with_location, ctx: mlir_context, alpha_beta_buffer, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "conv.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing Conv at ({self.ct_row}, {self.ct_col})")

        self.enable_tb = False
        self.model_config = model_config
        self.DQ = model_config.linear_DQ
        self.DK = model_config.linear_DK
        self.DV = model_config.linear_DV
        self.DH = model_config.linear_v_head_dim
        self.conv_kernel_size = model_config.linear_conv_kernel_size
        self.alpha_beta_buffer = alpha_beta_buffer
        self.actual_valid_qkv_data_size = self.DH + 2
        self.qkv_packet_size = round_up(self.actual_valid_qkv_data_size, 32)
        self.y_ty = np.ndarray[(3 * self.qkv_packet_size,), self.dtype]

        self.alpha_beta_size = model_config.linear_num_heads_padded * 2
        self.alpha_beta_w_ty = np.ndarray[(self.alpha_beta_size * 2,), self.dtype]
        
        self.wx_ty = np.ndarray[(self.conv_kernel_size * 2, self.DH), self.dtype]
        self.conv_result_ty = np.ndarray[(self.DQ + self.DK + self.DV, ), self.dtype]
        self.y_buffer_0 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_buffer_0_{self.ct_row}_{self.ct_col}")
        self.y_buffer_1 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_buffer_1_{self.ct_row}_{self.ct_col}")

        self.conv_buffer = aie.buffer(tile = self.ct, datatype = self.conv_result_ty, name = f"conv_y_buffer_{self.ct_row}_{self.ct_col}")
  
        self.wx_buffer_0 = aie.buffer(tile = self.ct, datatype = self.wx_ty, name = f"conv_wx_buffer_0_{self.ct_row}_{self.ct_col}")
        self.wx_buffer_1 = aie.buffer(tile = self.ct, datatype = self.wx_ty, name = f"conv_wx_buffer_1_{self.ct_row}_{self.ct_col}")
        self.wx_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.wx_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.alpha_beta_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.alpha_beta_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.tb_prod_lock = aie.lock(self.ct, lock_id = 7, init = 1)
        self.tb_cons_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        self.tb_flag: int = 1 if self.enable_tb else 0

        self.conv = ctx.func("conv", [self.wx_ty, self.wx_ty, self.y_ty, self.y_ty, self.conv_result_ty, self.alpha_beta_w_ty, np.int32])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.conv(self.wx_buffer_0, self.wx_buffer_1, self.y_buffer_0, self.y_buffer_1, self.conv_buffer, self.alpha_beta_buffer, self.tb_flag)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[1], chain = block[3])

            with block[1]:
                lock_acquire(self.wx_prod_lock, 1)
                aie.dma_bd(self.wx_buffer_0)
                lock_release(self.wx_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.wx_prod_lock, 1)
                aie.dma_bd(self.wx_buffer_1)
                lock_release(self.wx_cons_lock, 1)
                aie.next_bd(block[1])
            
            with block[3]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[4], chain = block[6 if self.enable_tb else 8])

            with block[4]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = self.actual_valid_qkv_data_size * 3, dimensions=[(3, self.qkv_packet_size), (self.actual_valid_qkv_data_size, 1)])  
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[5])
            
            with block[5]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = self.actual_valid_qkv_data_size * 3, dimensions=[(3, self.qkv_packet_size), (self.actual_valid_qkv_data_size, 1)])
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[4])
            
            with block[6]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[7], chain = block[8])

            with block[7]:
                lock_acquire(self.tb_cons_lock, 1)
                aie.dma_bd(self.conv_buffer, offset = 0, len = (64))
                lock_release(self.tb_prod_lock, 1)
                aie.next_bd(block[7])

            with block[8]:
                aie.EndOp()
    
    @property
    def W_in(self):
        return self.S2MM[1]
    
    @property
    def Y_out(self):
        return self.MM2S[0]

    
    @property
    def TB_OUT(self):
        return self.MM2S[1]
    
    