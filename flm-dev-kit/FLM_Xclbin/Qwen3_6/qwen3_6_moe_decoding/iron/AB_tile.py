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

class AB_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, 
                 
                 
                hidden_state_in_ty,
                hidden_state_in_buffer,
                
                top_k_expert_score_ty,
                top_k_expert_score_buffer,
                
                router_weight_ping_pong_ty,
                router_weight_ping_pong_buffer,
                
                
                router_weight_ty,
                router_weight_0,
                router_weight_1,
                stack_size: int = 1024, link_with: str = "ab_proj.o"
                 
                 ):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing AB_tile at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.D = model_config.hidden_size
        self.linear_heads = model_config.linear_num_heads
        self.alpha_beta_size = model_config.linear_num_heads_padded * 2
        
        self.ab_proj_weight_row_block = ctx.parameters["ab_proj_weight_row_block"]
        self.ab_proj_weight_col_block = ctx.parameters["ab_proj_weight_col_block"]
        
        
        
            
        self.hidden_state_in_ty=hidden_state_in_ty
        self.hidden_state_in_buffer =hidden_state_in_buffer
        
        self.top_k_expert_score_ty=top_k_expert_score_ty
        self.top_k_expert_score_buffer=top_k_expert_score_buffer
        
        self.router_weight_ping_pong_ty=router_weight_ping_pong_ty
        self.router_weight_ping_pong_buffer=router_weight_ping_pong_buffer
        
        
        self.router_weight_ty=router_weight_ty
        self.router_weight_0=router_weight_0
        self.router_weight_1= router_weight_1

        
        
        
        
        
        self.w_block_ty = np.ndarray[( self.ab_proj_weight_row_block* self.ab_proj_weight_col_block, ), self.dtype]
        
        self.alpha_beta_w_ty = np.ndarray[(self.alpha_beta_size * 2,), self.dtype]
        self.y_ty = np.ndarray[(self.alpha_beta_size * 2,), self.dtype]
        self.x_ty = np.ndarray[(self.D,), self.dtype]
        
        self.w_buffer_0 = aie.buffer(tile = self.ct, datatype = self.w_block_ty, name = f"ab_proj_w_buffer_0_{self.ct_row}_{self.ct_col}")
        self.w_buffer_1 = aie.buffer(tile = self.ct, datatype = self.w_block_ty, name = f"ab_proj_w_buffer_1_{self.ct_row}_{self.ct_col}")

        self.x_buffer = aie.buffer(tile = self.ct, datatype = self.x_ty, name = f"ab_proj_x_buffer_{self.ct_row}_{self.ct_col}")
        self.alpha_beta_w_buffer = aie.buffer(tile = self.ct, datatype = self.alpha_beta_w_ty, name = f"ab_proj_alpha_beta_w_buffer_{self.ct_row}_{self.ct_col}")
        self.y_buffer = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"ab_proj_y_buffer_{self.ct_row}_{self.ct_col}")

        self.w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)

        self.x_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.alpha_beta_w_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.alpha_beta_w_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        self.y_prod_lock = aie.lock(self.ct, lock_id = 6, init = 1)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)

        self.rtp_available_lock = aie.lock(self.ct, lock_id = 14, init=0)
        self.ab_proj = ctx.func("ab_proj", [self.y_ty, self.w_block_ty, self.w_block_ty, self.x_ty, self.alpha_beta_w_ty, 
                                            self.hidden_state_in_ty,
                                            self.router_weight_ty, self.router_weight_ty,
                                            self.top_k_expert_score_ty, self.router_weight_ping_pong_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.ab_proj(self.y_buffer, self.w_buffer_0, self.w_buffer_1, self.x_buffer, self.alpha_beta_w_buffer, 
                             self.hidden_state_in_buffer,
                             self.router_weight_0, self.router_weight_1,
                             self.top_k_expert_score_buffer, self.router_weight_ping_pong_buffer)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])

            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.alpha_beta_w_prod_lock, 1)
                aie.dma_bd(self.alpha_beta_w_buffer)
                lock_release(self.alpha_beta_w_cons_lock, 1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])
            
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
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[7], chain = block[8])

            with block[7]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[7])

            with block[8]:
                aie.EndOp()

    @property
    def X_in(self):
        return self.S2MM[0]
    
    @property
    def W_in(self):
        return self.S2MM[1]
    
    @property
    def Y_out(self):
        return self.MM2S[1]
