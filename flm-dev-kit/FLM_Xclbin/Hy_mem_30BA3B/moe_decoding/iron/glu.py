# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_CT_IP, comment_print
from constants import *


class glu_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "glu.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)
        self.D = model_config.D
        
        assert(model_config.hidden_size ==   model_config.D)
        self.intermediate_size = model_config.intermediate_size
        self.glu_slice = model_config.glu_slice
        self.up_gate_ty = np.ndarray[(self.glu_slice,), self.dtype]
        self.hid_ty = np.ndarray[(self.intermediate_size,), self.dtype] # plust 16 for packet_id and alignment issue
        self.D_out_ty = np.ndarray[(self.D,), self.dtype] # + 4 for packed_id, + 12 for alignment
        
        self.hidden_state_ty = np.ndarray[(self.D,), self.dtype]
        
        self.shared_expert_output_ty = np.ndarray[(self.D,),  np.dtype[np.float32]]
        
        
        self.top_k_expert_score_ty = np.ndarray[(model_config.num_experts,), np.dtype[np.float32]]
        self.top_k_expert_index_ty = np.ndarray[(model_config.num_experts,), np.dtype[np.int32]]
        
        
        # +14 for alignment, +2 for packet_id
        self.to_proj_x_slice_with_pkt_buf_size = 2+ model_config.x_slice_size
        self.to_proj_x_slice_with_pkt_ty = np.ndarray[(model_config.x_slice_size+ 16,), self.dtype]
        
        
        assert(self.D % model_config.glu_slice == 0)

        comment_print(f"GLU tile at ({self.ct_row}, {self.ct_col}) with intermediate_size: {self.intermediate_size}, glu_slice: {self.glu_slice}")
        # q_proj tile and mem 
        # double buffer
        self.glu_slice_in_buffer_0 = aie.buffer(tile = self.ct, datatype = self.up_gate_ty, name = f"glu_x_slice_buffer_0{self.ct_row}_{self.ct_col}")
        self.glu_slice_in_buffer_1 = aie.buffer(tile = self.ct, datatype = self.up_gate_ty, name = f"glu_slice_buffer_1{self.ct_row}_{self.ct_col}")

        self.intermediate_buffer = aie.buffer(tile = self.ct, datatype = self.hid_ty, name = f"intermediate_out_buffer_{self.ct_row}_{self.ct_col}")
        self.D_mlp_final_to_rms = aie.buffer(tile=self.ct, datatype = self.D_out_ty, name= f"D_out_buffer{self.ct_row}_{self.ct_col}")
        
        self.D_in_buffer = aie.buffer(self.ct, datatype = self.hidden_state_ty, name= f"D_in_buffer_{self.ct_row}_{self.ct_col}")
        # despite the name: fp32 sum of the routed experts (see glu.cc)
        self.shared_expert_output_buffer = aie.buffer(self.ct, datatype = self.shared_expert_output_ty, name= f"shared_Expert_buffer_{self.ct_row}_{self.ct_col}")

        self.top_k_expert_score_buffer = aie.buffer(
            tile=self.ct, datatype=self.top_k_expert_score_ty, name=f"top_k_expert_score_{self.ct_row}_{self.ct_col}")
        self.top_k_expert_index_buffer = aie.buffer(
            tile=self.ct, datatype=self.top_k_expert_index_ty, name=f"top_k_epxert_index_{self.ct_row}_{self.ct_col}")
        
        
        
        self.to_proj_x_slice_with_pkt_buffer_0 = aie.buffer(
            tile=self.ct, datatype=self.to_proj_x_slice_with_pkt_ty, name=f"to_proj_x_slice_with_pkt_buffer__0_{self.ct_row}_{self.ct_col}"
        )
        
        self.to_proj_x_slice_with_pkt_buffer_1 = aie.buffer(
            tile=self.ct, datatype=self.to_proj_x_slice_with_pkt_ty, name=f"to_proj_x_slice_with_pkt_buffer__1_{self.ct_row}_{self.ct_col}"
        )        
        
        
        
        self.glu_slice_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.glu_slice_con_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        
        
        self.to_proj_x_slice_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.to_proj_x_slice_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        
        self.D_out_prod_lock = aie.lock(self.ct, lock_id=4, init=0)
        self.D_out_cons_lock = aie.lock(self.ct, lock_id=5, init=0)
        
        self.D_in_prod_lock = aie.lock(self.ct, lock_id=6, init=2) #NOTE 2 not for ping-pong, but for 2 CT
        self.D_in_con_lock = aie.lock(self.ct, lock_id=7, init =0)
        

    
    
        self.glu_request_router_send_lock = aie.lock(self.ct, lock_id=12, init=0)

    
        self.glu = aie.external_func("glu", [
            self.top_k_expert_score_ty, self.top_k_expert_index_ty,
            self.hid_ty,  self.D_out_ty, self.hidden_state_ty, self.shared_expert_output_ty,
                                             self.up_gate_ty, self.up_gate_ty, self.to_proj_x_slice_with_pkt_ty, self.to_proj_x_slice_with_pkt_ty
                                             ])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.glu(self.top_k_expert_score_buffer, self.top_k_expert_index_buffer,
                        self.intermediate_buffer, self.D_mlp_final_to_rms, self.D_in_buffer, self.shared_expert_output_buffer,
                        self.glu_slice_in_buffer_0, self.glu_slice_in_buffer_1,
                        self.to_proj_x_slice_with_pkt_buffer_0, self.to_proj_x_slice_with_pkt_buffer_1
                        )
        
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
            with block[1]:
                lock_acquire(self.glu_slice_prod_lock, 1)
                aie.dma_bd(self.glu_slice_in_buffer_0)
                lock_release(self.glu_slice_con_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.glu_slice_prod_lock, 1)
                aie.dma_bd(self.glu_slice_in_buffer_1)
                lock_release(self.glu_slice_con_lock, 1)
                aie.next_bd(block[1])

            with block[3]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])
                
            with block[4]: # x, to mt
                lock_acquire(self.to_proj_x_slice_cons_lock, 1)
                aie.dma_bd(self.to_proj_x_slice_with_pkt_buffer_0, offset=14, len=self.to_proj_x_slice_with_pkt_buf_size)
                lock_release(self.to_proj_x_slice_prod_lock, 1)
                aie.next_bd(block[5])

            with block[5]: # x, to mt
                lock_acquire(self.to_proj_x_slice_cons_lock, 1)
                aie.dma_bd(self.to_proj_x_slice_with_pkt_buffer_1,offset=14, len=self.to_proj_x_slice_with_pkt_buf_size)
                lock_release(self.to_proj_x_slice_prod_lock, 1)
                aie.next_bd(block[4])

            with block[6]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[8])
            with block[7]:
                lock_acquire(self.D_out_cons_lock, 1)
                aie.dma_bd_packet(0, rms_pkt_id.from_glu)
                aie.dma_bd(self.D_mlp_final_to_rms)
                lock_release(self.D_out_prod_lock, 1)
                aie.next_bd(block[7])
                
            with block[8]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index =1, dest=block[9], chain=block[10])
            with block[9]:
                lock_acquire(self.D_in_prod_lock, value=2)
                aie.dma_bd(self.D_in_buffer)
                lock_release(self.D_in_con_lock, value=2)
                aie.next_bd(block[9])
            with block[10]:
                aie.EndOp()

    @property
    def Up_Gate_in(self) -> dma_port:
        return self.S2MM[0]
    
    @property
    def D_in(self):
        return self.S2MM[1]
    
    @property
    def Intermediate_out(self) -> dma_port:
        return self.MM2S[0]

    
    @property
    def D_out(self):
        return self.MM2S[1]