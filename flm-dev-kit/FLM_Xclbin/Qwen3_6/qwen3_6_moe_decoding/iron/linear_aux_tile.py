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

class linear_aux_memtile(AIE_MT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        # for all linear layer weight transitions
        # PW_in -> ab_tile -> W_in, S2MM[0] -> MM2S[0]

        # Conv weights: W_in -> conv_core, S2MM[1] -> MM2S[1]
        # Delta_net_norm -> delta_net_norm -> S2MM[1] -> MM2S[2]
        # A and dt_bias for alpha_beta, -> send to conv tile X_in, append to the X_in

        # Weight order
        # | 8192 * 4 conv weights | 128 delta_net_norm weights | alpha_beta weights |

        # for pipelines, all DMA are seperated into 4 partials (ping-ponged, 4 * 4 = 16 bds, plenty)
        super().__init__(tile)
        self.model_config = model_config
        self.CONV_WIDTH = model_config.linear_DQ + model_config.linear_DK + model_config.linear_DV
        comment_print(f"Linear DQ: {model_config.linear_DQ}")
        comment_print(f"Linear DK: {model_config.linear_DK}")
        comment_print(f"Linear DV: {model_config.linear_DV}")
        self.DH = model_config.linear_v_head_dim
        self.CONV_SIZE  = model_config.linear_conv_kernel_size
        self.delta_net_norm_size = model_config.linear_v_head_dim
        self.alpha_beta_size = model_config.linear_num_heads_padded * 2 * 2 # first x2 is alpha and beta, second x2 is for using bf16 to store fp32
        self.hiddens_states_size = model_config.hidden_size
        
        self.linear_heads = model_config.linear_num_heads
  

        self.conv_wx_ty = np.ndarray[(2 * self.CONV_SIZE, self.CONV_WIDTH), self.dtype]
        comment_print(f"Conv weight ty: {self.conv_wx_ty}")
        self.delta_norm_w_ty = np.ndarray[(self.delta_net_norm_size,), self.dtype]
        comment_print(f"delta_norm_ty: {self.delta_norm_w_ty}")
        self.alpha_beta_ty = np.ndarray[(self.alpha_beta_size,), self.dtype]
        self.hiddens_states_ty = np.ndarray[(self.hiddens_states_size,), self.dtype]


        
        #TODO: better naming
        # router_weight_row_block because original router weight is (num_expert x D) which is too large for MT, need to split  to chunks
        self.router_weight_row_block:int = ctx.parameters["router_weight_row_block"]
        # self.router_weight_col_block:int = ctx.parameters["router_weight_col_block"]
        self.ab_proj_weight_row_block = ctx.parameters["ab_proj_weight_row_block"]
        self.ab_proj_weight_col_block = ctx.parameters["ab_proj_weight_col_block"]
        self.ab_proj_w_ty = np.ndarray[(self.ab_proj_weight_row_block*self.ab_proj_weight_col_block,), self.dtype]        
        assert(model_config.num_experts %  self.router_weight_row_block == 0)
        # assert(model_config.hidden_size % self.router_weight_col_block == 0)
        
        
        self.router_w_ty= np.ndarray[(model_config.hidden_size* self.router_weight_row_block, ), self.dtype  ]
        
        self.conv_wx_buffer = aie.buffer(tile = self.mt, datatype = self.conv_wx_ty, name = f"mt_conv_wx_buffer_{self.mt_row}_{self.mt_col}")
        self.delta_norm_buffer = aie.buffer(tile = self.mt, datatype = self.delta_norm_w_ty, name = f"delta_norm_buffer_{self.mt_row}_{self.mt_col}")
        self.alpha_beta_buffer = aie.buffer(tile = self.mt, datatype = self.alpha_beta_ty, name = f"alpha_beta_buffer_{self.mt_row}_{self.mt_col}")
        self.hiddens_states_buffer = aie.buffer(tile = self.mt, datatype = self.hiddens_states_ty, name = f"hiddens_states_buffer_{self.mt_row}_{self.mt_col}")
        self.ab_proj_w_buffer_0 = aie.buffer(tile = self.mt, datatype = self.ab_proj_w_ty, name = f"ab_proj_w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.ab_proj_w_buffer_1 = aie.buffer(tile = self.mt, datatype = self.ab_proj_w_ty, name = f"ab_proj_w_buffer_1_{self.mt_row}_{self.mt_col}")
        
        self.router_w_buffer_0 = aie.buffer(tile=self.mt, datatype=self.router_w_ty,
                                            name=f"router_w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.router_w_buffer_1 = aie.buffer(tile=self.mt, datatype=self.router_w_ty,
                                            name=f"router_w_buffer_1_{self.mt_row}_{self.mt_col}")    


        self.conv_w_prod_lock = aie.lock(self.mt, lock_id = 0, init = 1)
        self.conv_w_cons_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        self.delta_norm_prod_lock = aie.lock(self.mt, lock_id = 2, init = 1)
        self.delta_norm_cons_lock = aie.lock(self.mt, lock_id = 3, init = 0)
        self.alpha_beta_buffer_prod_lock = aie.lock(self.mt, lock_id = 4, init = 1)
        self.alpha_beta_buffer_cons_lock = aie.lock(self.mt, lock_id = 5, init = 0)

        self.ab_proj_prod_lock = aie.lock(self.mt, lock_id = 6, init = 2)
        self.ab_proj_cons_lock = aie.lock(self.mt, lock_id = 7, init = 0)

        self.hidden_states_prod_lock = aie.lock(self.mt, lock_id = 8, init = 1)
        self.hidden_states_cons_lock = aie.lock(self.mt, lock_id = 9, init = 0)

        self.conv_old_qkv_prod_lock = aie.lock(self.mt, lock_id = 10, init = 0)
        self.conv_new_qkv_prod_lock = aie.lock(self.mt, lock_id = 11, init = 0)
        self.conv_all_qkvw_cons_lock = aie.lock(self.mt, lock_id = 12, init = 0)

        self.conv_new_cache_out_cons_lock = aie.lock(self.mt, lock_id = 13, init = 0)
        
        
        self.router_w_prod_lock = aie.lock(self.mt, lock_id=14, init=2)
        self.router_w_con_lock = aie.lock(self.mt, lock_id=15, init=0)

        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])

            with block[1]:
                lock_acquire(self.ab_proj_prod_lock, 1)
                aie.dma_bd(self.ab_proj_w_buffer_0)
                lock_release(self.ab_proj_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.ab_proj_prod_lock, 1)
                aie.dma_bd(self.ab_proj_w_buffer_1)
                lock_release(self.ab_proj_cons_lock, 1)
                aie.next_bd(block[1])

            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])

            with block[4]:
                lock_acquire(self.ab_proj_cons_lock, 1)
                aie.dma_bd(self.ab_proj_w_buffer_0)
                lock_release(self.ab_proj_prod_lock, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.ab_proj_cons_lock, 1)
                aie.dma_bd(self.ab_proj_w_buffer_1)
                lock_release(self.ab_proj_prod_lock, 1)
                aie.next_bd(block[4])

            with block[6]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[7], chain = block[10])

            with block[7]:
                lock_acquire(self.conv_w_prod_lock, 1)
                aie.dma_bd(self.conv_wx_buffer, offset = self.CONV_WIDTH, len = self.CONV_WIDTH * self.CONV_SIZE, dimensions=[(self.CONV_SIZE, self.CONV_WIDTH * 2), (16, self.CONV_WIDTH // 16), (self.CONV_WIDTH // 16, 1)])
                lock_release(self.conv_old_qkv_prod_lock, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.delta_norm_prod_lock, 1)
                aie.dma_bd(self.delta_norm_buffer)
                lock_release(self.delta_norm_cons_lock, 1)
                aie.next_bd(block[9])

            with block[9]:
                lock_acquire(self.alpha_beta_buffer_prod_lock, 1)
                aie.dma_bd(self.alpha_beta_buffer)
                lock_release(self.alpha_beta_buffer_cons_lock, 1)
                aie.next_bd(block[100])
            
            with block[100]:
                lock_acquire(self.conv_old_qkv_prod_lock, 1)
                aie.dma_bd(self.conv_wx_buffer, offset = 0, len = (self.CONV_SIZE - 1) * self.CONV_WIDTH, dimensions=[(self.CONV_SIZE - 1, self.CONV_WIDTH * 2), (16, self.CONV_WIDTH // 16), (self.CONV_WIDTH // 16, 1)])
                lock_release(self.conv_new_qkv_prod_lock, 1)
                aie.next_bd(block[7])

            with block[10]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[11], chain = block[12])

            with block[11]:
                lock_acquire(self.conv_all_qkvw_cons_lock, 1)
                aie.dma_bd(self.conv_wx_buffer, offset = 0, len = 2 * self.CONV_WIDTH * self.CONV_SIZE, dimensions = [(self.CONV_WIDTH // self.DH, self.DH), (2 * self.CONV_SIZE, self.CONV_WIDTH),(self.DH, 1)])
                lock_release(self.conv_new_cache_out_cons_lock, 1)
                aie.next_bd(block[11])

            with block[12]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[13], chain = block[14])

            with block[13]:
                lock_acquire(self.delta_norm_cons_lock, 1)
                aie.dma_bd(self.delta_norm_buffer)
                lock_release(self.delta_norm_prod_lock, 1)
                aie.next_bd(block[13])

            with block[14]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest = block[15], chain = block[16])

            with block[15]:
                lock_acquire(self.hidden_states_prod_lock, 1)
                aie.dma_bd(self.hiddens_states_buffer)
                lock_release(self.hidden_states_cons_lock, 1)
                aie.next_bd(block[15])

            
            with block[16]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest = block[17], chain = block[19])

            with block[17]:
                lock_acquire(self.hidden_states_cons_lock, 1)
                aie.dma_bd(self.hiddens_states_buffer)
                lock_release(self.hidden_states_prod_lock, 1)
                aie.next_bd(block[18])

            with block[18]:
                lock_acquire(self.alpha_beta_buffer_cons_lock, 1)
                aie.dma_bd(self.alpha_beta_buffer)
                lock_release(self.alpha_beta_buffer_prod_lock, 1)
                aie.next_bd(block[17])
            
            with block[19]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest = block[20], chain = block[21])

            with block[20]:
                lock_acquire(self.conv_new_qkv_prod_lock, 1)
                aie.dma_bd(self.conv_wx_buffer, offset = (2 * (self.CONV_SIZE - 1) * self.CONV_WIDTH), len = self.CONV_WIDTH)
                lock_release(self.conv_all_qkvw_cons_lock, 1)
                aie.next_bd(block[20])

            with block[21]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest = block[22], chain = block[200])

            with block[22]:
                lock_acquire(self.conv_new_cache_out_cons_lock, 1)
                aie.dma_bd(self.conv_wx_buffer, offset = 2 * self.CONV_WIDTH, len = (self.CONV_SIZE - 1) * self.CONV_WIDTH, dimensions = [(3, self.CONV_WIDTH * 2), (16, self.CONV_WIDTH // 16),(self.CONV_WIDTH // 16, 1)])
                lock_release(self.conv_w_prod_lock, 1)
                aie.next_bd(block[22])

            with block[200]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index=5, dest=block[201], chain=block[203])
            with block[201]:
                lock_acquire(self.router_w_prod_lock, value=1)
                aie.dma_bd(self.router_w_buffer_0)
                lock_release(self.router_w_con_lock, value=1)
                aie.next_bd(block[202])
            with block[202]:
                lock_acquire(self.router_w_prod_lock, value=1)
                aie.dma_bd(self.router_w_buffer_1)
                lock_release(self.router_w_con_lock, value=1)                
                aie.next_bd(block[201])
            with block[203]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index=5, dest=block[204], chain=block[400])
            with block[204]:
                lock_acquire(self.router_w_con_lock, value=1)
                aie.dma_bd(self.router_w_buffer_0)
                lock_release(self.router_w_prod_lock, value=1)
                aie.next_bd(block[205])
            with block[205]:
                lock_acquire(self.router_w_con_lock, value=1)
                aie.dma_bd(self.router_w_buffer_1)                
                lock_release(self.router_w_prod_lock, value=1)                
                aie.next_bd(block[204])
            with block[400]:
                aie.EndOp()

    @property
    def AB_PROJ_W_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def W_in(self) -> dma_port:
        return self.S2MM[1]
    
    @property
    def X_in(self) -> dma_port:
        return self.S2MM[2]

    @property
    def QKV_in(self) -> dma_port:
        return self.S2MM[3]
    
    @property
    def router_w_in(self)->dma_port:
        return self.S2MM[5]
    
    @property
    def AB_PROJ_W_out(self) -> dma_port:
        return self.MM2S[0]
    
    @property
    def CONV_W_out(self) -> dma_port:
        return self.MM2S[1]  
        
    @property
    def DELTA_NORM_W_out(self) -> dma_port:
        return self.MM2S[2]  

    @property
    def HIDDEN_STATES_AB_BIAS_out(self) -> dma_port:
        return self.MM2S[3]
    

    @property
    def NEW_CACHE_out(self) -> dma_port:
        return self.MM2S[4]
    
    @property
    def router_w_out(self)->dma_port:
        return self.MM2S[5]
    