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

        # The delta-net's gated RMSNorm: one token of the linear-attention core
        # output, Linear_DV wide. Matches apply_gate<Linear_DV> in gate.cc.
        self.DV = model_config.linear_DV
        self.D = model_config.D
        # Stage 7b: the gated row goes to DDR once, not once per row-block group
        # of the out projection. Must stay in step with `repeats` in gate.cc.
        self.repeats = 1
        # q_proj tile and mem
        # double buffer
        self.gate_ty = np.ndarray[(self.DV,), self.dtype]

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

        self.gate = aie.external_func("gate", [self.gate_ty, self.gate_ty, self.gate_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
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
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[5], chain = block[6])
                
            with block[5]: # x, to mt
                lock_acquire(self.gated_attn_out_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_out_proj)
                aie.dma_bd(self.gated_attn_out_buffer)
                lock_release(self.gated_attn_out_prod_lock, 1)
                aie.next_bd(block[5])

            with block[6]:
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
    
