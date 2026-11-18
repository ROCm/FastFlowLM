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
from typing import List
from utils import dma_port, mlir_context, tile_with_location, connect_dma, lock_acquire, lock_release
from utils import AIE_CT_IP, comment_print
from constants import *


class glu_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "glu.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)
        
        self.intermediate_size = model_config.intermediate_size
        self.glu_slice = model_config.glu_slice
        self.up_gate_ty = np.ndarray[(self.glu_slice,), self.dtype]
        if model_config.enable_double_wide_mlp:
            self.hid_ty = np.ndarray[(self.intermediate_size * 2,), self.dtype]
        else:
            self.hid_ty = np.ndarray[(self.intermediate_size,), self.dtype]
        self.y_ty = np.ndarray[(self.glu_slice // 2,), self.dtype]
        self.D = model_config.D

        self.repeats = self.D // (model_config.q4nx_row_block_size * 16) # 16 is a constant of 16 CTs in projection engine
        comment_print(f"GLU tile at ({self.ct_row}, {self.ct_col}) with intermediate_size: {self.intermediate_size}, glu_slice: {self.glu_slice}")
        # q_proj tile and mem 
        # double buffer
        self.x_buffer_0 = aie.buffer(tile = self.ct, datatype = self.up_gate_ty, name = f"x_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = aie.buffer(tile = self.ct, datatype = self.up_gate_ty, name = f"x_1_{self.ct_row}_{self.ct_col}")

        self.y_buffer_0 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_0_{self.ct_row}_{self.ct_col}")
        self.y_buffer_1 = aie.buffer(tile = self.ct, datatype = self.y_ty, name = f"y_1_{self.ct_row}_{self.ct_col}")

        self.hid_buffer = aie.buffer(tile = self.ct, datatype = self.hid_ty, name = f"hid_0_{self.ct_row}_{self.ct_col}")
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.hid_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.hid_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.rtp_lock = aie.lock(self.ct, lock_id = 6, init = 0)

        self.glu = aie.external_func("glu", [self.hid_ty, self.up_gate_ty, self.up_gate_ty, self.y_ty, self.y_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.glu(self.hid_buffer, self.x_buffer_0, self.x_buffer_1, self.y_buffer_0, self.y_buffer_1)

        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            with block[3]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])
                
            with block[4]: # x, to mt
                lock_acquire(self.hid_cons_lock, 1)
                aie.dma_bd_packet(0, to_proj_x_in_pkt_id.from_glu)
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.hid_prod_lock, 1)
                aie.next_bd(block[5])
            with block[5]: # x, to mt
                lock_acquire(self.hid_cons_lock, 1)
                aie.dma_bd_packet(0, to_proj_x_in_pkt_id.from_glu)
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.hid_prod_lock, 1)
                aie.next_bd(block[4])

            with block[6]:
                aie.EndOp()

    @property
    def Up_Gate_in(self) -> dma_port:
        return self.S2MM[0]
    
    @property
    def Hid_out(self) -> dma_port:
        return self.MM2S[0]

