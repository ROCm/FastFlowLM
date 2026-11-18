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


class gate_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "gate.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.enable_tb = False
        
        self.DQ = model_config.DQ
        self.D = model_config.D
        self.repeats = self.D // (model_config.q4nx_row_block_size * 16) # 16 is a constant of 16 CTs in projection engine
        # q_proj tile and mem 
        # double buffer
        self.gate_ty = np.ndarray[(self.DQ,), self.dtype]

        self.gate_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"gate_{self.ct_row}_{self.ct_col}")
        self.attn_out_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"attn_out_{self.ct_row}_{self.ct_col}")
        self.gated_attn_out_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"gated_attn_out_{self.ct_row}_{self.ct_col}")

        self.gate_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)  
        self.gate_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.attn_out_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.attn_out_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.gated_attn_out_prod_lock = aie.lock(self.ct, lock_id = 4, init = self.repeats)
        self.gated_attn_out_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)
        self.tb_lock = aie.lock(self.ct, lock_id = 7, init = 0)

        # STATELESS_X (Makefile: MODEL_TYPE=<model>_SL): the host writes the kind of layer into an RTP at the start of every run (kernels/gate.cc)
        self.stateless = os.environ.get("STATELESS_X") == "1"
        if self.stateless:
            self.rtp_ty = np.ndarray[(8,), np.dtype[np.int32]]
            # the allocator adds 32: the RTP lands at 36864, where FLM_DLL's sequence writes it (the build checks the address)
            self.rtp_buffer = ctx.buffer(f"RTP_gate_mode_{self.ct_row}_{self.ct_col}", self.located_tile, self.rtp_ty, use_write_rtp = True, use_address = 36832)
            self.gate = aie.external_func("gate", [self.gate_ty, self.gate_ty, self.gate_ty, self.rtp_ty])
        else:
            self.gate = aie.external_func("gate", [self.gate_ty, self.gate_ty, self.gate_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                if self.stateless:
                    self.gate(self.gated_attn_out_buffer, self.gate_buffer, self.attn_out_buffer, self.rtp_buffer)
                else:
                    self.gate(self.gated_attn_out_buffer, self.gate_buffer, self.attn_out_buffer)
        
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])
            with block[1]:
                lock_acquire(self.gate_prod_lock, 1)
                aie.dma_bd(self.gate_buffer)
                lock_release(self.gate_cons_lock, 1)
                aie.next_bd(block[1])


            with block[2]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[3], chain = block[4])

            with block[3]:
                lock_acquire(self.attn_out_prod_lock, 1)
                aie.dma_bd(self.attn_out_buffer)
                lock_release(self.attn_out_cons_lock, 1)
                aie.next_bd(block[3])

            with block[4]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[5], chain = block[6 if self.enable_tb else 8])
                
            with block[5]: # x, to mt
                lock_acquire(self.gated_attn_out_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_out_proj)
                aie.dma_bd(self.gated_attn_out_buffer)
                if (self.enable_tb):
                    lock_release(self.tb_lock, 1)
                else:
                    lock_release(self.gated_attn_out_prod_lock, 1)
                aie.next_bd(block[5])
                
            with block[6]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[7], chain = block[8])
                
            with block[7]: # x, to mt
                lock_acquire(self.tb_lock, self.repeats)
                aie.dma_bd(self.gated_attn_out_buffer)
                lock_release(self.gated_attn_out_prod_lock, self.repeats)
                aie.next_bd(block[7])
            with block[8]:
                aie.EndOp()

    @property
    def Gate_In(self) -> dma_port:
        return self.S2MM[0]

    @property
    def Attn_In(self) -> dma_port:
        return self.S2MM[1]
    
    @property
    def Attn_Out(self) -> dma_port:
        return self.MM2S[0]
    
    @property
    def TB_Out(self) -> dma_port:
        return self.MM2S[1]
