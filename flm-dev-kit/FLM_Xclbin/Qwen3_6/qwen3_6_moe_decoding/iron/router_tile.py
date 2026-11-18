# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from matplotlib.cbook import ls_mapper
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

from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP
from aie.extras.dialects.ext.arith import constant, index_cast



class Router_tile(AIE_CT_IP):
    
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, 
                up_neighbor_hidden_state_in_buffer,
                up_neighbor_share_expert_output_buffer,
                up_neighbor_top_k_expert_score_buffer,
                up_neighbor_top_k_expert_index_buffer,
                
                up_neighbor_hidden_state_in_buffer_ty, 
                up_neighbor_share_expert_output_buffer_ty,
                up_neighbor_top_k_expert_score_buffer_ty,
                up_neighbor_top_k_expert_index_buffer_ty,

                stack_size: int = 3*1024, link_with: str = "router.o"
                ):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing AB_tile at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.D = model_config.hidden_size
  
        self.num_active_experts = model_config.num_active_experts
        self.num_experts = model_config.num_experts
        
        
        
        
        
        self.router_weight_row_block:int = ctx.parameters["router_weight_row_block"]
        self.router_weight_col_block:int = ctx.parameters["router_weight_col_block"]        
        
        self.up_neighbor_hidden_state_in_buffer=up_neighbor_hidden_state_in_buffer
        self.up_neighbor_share_expert_output_buffer=up_neighbor_share_expert_output_buffer
        self.up_neighbor_top_k_expert_score_buffer= up_neighbor_top_k_expert_score_buffer
        self.up_neighbor_top_k_expert_index_buffer = up_neighbor_top_k_expert_index_buffer

   
        self.router_control_packet_ty = np.ndarray[(
            (16,), np.dtype[np.uint32]
        )]
        
        self.ping_pong_ty = np.ndarray[(
            (16,),np.dtype[np.int32]
        )]
        self.up_neighbor_hidden_state_in_buffer_ty=up_neighbor_hidden_state_in_buffer_ty
        self.up_neighbor_top_k_expert_score_buffer_ty=up_neighbor_top_k_expert_score_buffer_ty
        self.ct_hidden_state_in_buffer = aie.buffer(
            self.ct, 
            datatype = up_neighbor_hidden_state_in_buffer_ty
        )   
        self.ct_top_k_expert_score_local = aie.buffer(
            self.ct,
            datatype=up_neighbor_top_k_expert_score_buffer_ty
        )
        
        self.router_weight_ping_pong_flag = aie.buffer(
            self.ct,
            datatype  = self.ping_pong_ty
        )
    
        self.weight_ty = np.ndarray[(
             self.router_weight_col_block* self.router_weight_row_block,),  np.dtype[bfloat16]]
        
    
        self.router_weight_buffer_0 = aie.buffer(tile=self.ct,
                                            datatype=self.weight_ty,
                                            name=f"router_weight_0_{self.ct_row}_{self.ct_col}")
        
        self.router_weight_buffer_1 = aie.buffer(tile=self.ct,
                                            datatype=self.weight_ty,
                                            name=f"router_weight_1_{self.ct_row}_{self.ct_col}")
        

        self.router_weight_prod_lock = aie.lock(self.ct, lock_id=2, init=2)
        self.router_weight_con_lock = aie.lock(self.ct, lock_id=3, init=0)
        
        self.CT_to_tile_control_buffer = aie.buffer(self.ct, datatype=self.router_control_packet_ty)
        self.control_packet_out_prod_lock = aie.lock(self.ct, lock_id=6, init=1)
        self.control_packet_out_con_lock = aie.lock(self.ct, lock_id=7, init=0)

        self.CT_from_tile_control_buffer = aie.buffer(self.ct, datatype=self.router_control_packet_ty)
        self.control_packet_in_prod_lock = aie.lock(self.ct, lock_id=8, init=1)
        self.control_packet_in_con_lock = aie.lock(self.ct, lock_id=9, init=0)
        
        
        
        self.shimtile_configure_ready_lock = aie.lock(self.ct, lock_id=10, init=0)
        
        
        self.router_helper_prod_lock = aie.lock(self.ct, lock_id=11, init=1)
        self.router_helper_con_lock  = aie.lock(self.ct, lock_id=13, init=0)
        self.router_helper_finished_lock = aie.lock(self.ct, lock_id=14, init=0)
        
        self.SHIM_BD_1_VIRTUAL_buffer = aie.buffer(
            self.ct, datatype=self.router_control_packet_ty
        )
        
        self.CT_to_shimtile_control_bd_id = 10
        self.CT_from_shimtile_control_bd_id = 11

        self.CT_from_shimtile_control_bd_port = ctx.parameters["CT_from_shimtile_control_bd_port"]
        self.CT_to_shimtile_control_bd_port = ctx.parameters["CT_to_shimtile_control_bd_port"]
        
        self.zero_func =ctx.func(
            "zero_int32_1",[
                self.ping_pong_ty
            ]
        )
        self.router_func = ctx.func("router", [
           up_neighbor_hidden_state_in_buffer_ty, 
           up_neighbor_hidden_state_in_buffer_ty,
           self.weight_ty, self.weight_ty,
           up_neighbor_top_k_expert_score_buffer_ty, up_neighbor_top_k_expert_score_buffer_ty, 
           up_neighbor_top_k_expert_index_buffer_ty,
           up_neighbor_share_expert_output_buffer_ty,
           
            self.router_control_packet_ty,  # for virtual buffer
            self.router_control_packet_ty, self.router_control_packet_ty,  # for ct contorl packer          
            np.int32, np.int32,
            np.int32, np.int32,
            
            np.int32, np.int32,
            np.int32, np.int32,
            self.ping_pong_ty
        ])

        self.__call__()
    
    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            self.zero_func(self.router_weight_ping_pong_flag)
            for _ in range_(0xFFFFFFFF):
                self.router_func(
                    self.up_neighbor_hidden_state_in_buffer, 
                    self.ct_hidden_state_in_buffer,
                    self.router_weight_buffer_0, self.router_weight_buffer_1, 
                    self.up_neighbor_top_k_expert_score_buffer, self.ct_top_k_expert_score_local, 
                    self.up_neighbor_top_k_expert_index_buffer,
                    self.up_neighbor_share_expert_output_buffer,
                    self.SHIM_BD_1_VIRTUAL_buffer,
                    self.CT_to_tile_control_buffer, self.CT_from_tile_control_buffer,
                    constant(48+6), constant(48+7),
                    constant(48+8), constant(48+9),
                    constant(self.CT_to_shimtile_control_bd_id), constant(self.CT_from_shimtile_control_bd_id),
                    constant(self.CT_to_shimtile_control_bd_port), constant(self.CT_from_shimtile_control_bd_port),
                    self.router_weight_ping_pong_flag              
                )
        
        
        @aie.mem(self.ct)
        def mem_dma_body(block):


            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, channel_index=self.CT_from_shimtile_control_bd_port, dest=block[1], chain=block[2])
            with block[1]:
                lock_acquire(self.control_packet_in_prod_lock, value=1)
                aie.dma_bd(self.CT_from_tile_control_buffer, offset=0, len=1,  # len=1 since we expect a single value per read to shimtile
                        bd_id=self.CT_from_shimtile_control_bd_id)
                lock_release(self.control_packet_in_con_lock,  value=1)
                aie.next_bd(block[1])
            with block[2]:
                s1 = aie.dma_start(aie.DMAChannelDir.MM2S, channel_index=self.CT_to_shimtile_control_bd_port, dest=block[3], chain=block[4])
            with block[3]:
                lock_acquire(self.control_packet_out_con_lock,  value=1)
                #NOTE: the len is something changes dynamicaly
                aie.dma_bd(self.CT_to_tile_control_buffer, offset=0, len=1, #NOTE: for now
                        bd_id=self.CT_to_shimtile_control_bd_id) 
                lock_release(self.control_packet_out_prod_lock, value=1)
                aie.next_bd(block[3])

            with block[4]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[7], chain=block[9])
            with block[7]:
                lock_acquire(self.router_weight_prod_lock, value=1)
                aie.dma_bd(self.router_weight_buffer_0)
                lock_release(self.router_weight_con_lock, value=1)
                aie.next_bd(block[8])      
            with block[8]:
                lock_acquire(self.router_weight_prod_lock, value=1)
                aie.dma_bd(self.router_weight_buffer_1)
                lock_release(self.router_weight_con_lock, value=1)
                aie.next_bd(block[7])      
            with block[9]:          
                aie.EndOp()
    

    # @property
    # def router_x_in(self)->dma_port:
    #     return self.S2MM[0]
    
    @property
    def router_w_in(self):
        return self.S2MM[0]
    
    @property
    def router_out_debug(self)->dma_port:
        return self.MM2S[0]