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
    def __init__(self, tile, mctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.Q_HEAD_PER_ROUND = model_config.Q_HEAD_PER_ROUND_PADDED
        self.SWA_Q_HEAD_PER_ROUND = model_config.SWA_Q_HEAD_PER_ROUND_PADDED
        self.num_heads = model_config.num_heads
        self.DH = model_config.DH
        self.SWA_DH = model_config.SWA_DH

        self.q_mem_ty = np.ndarray[(1, self.num_heads * self.DH), np.dtype[bfloat16]]
        self.o_mem_ty = np.ndarray[(1, self.num_heads * self.DH), np.dtype[bfloat16]]
        self.swa_q_mem_ty = np.ndarray[(1, self.num_heads * self.SWA_DH), np.dtype[bfloat16]]
        self.swa_o_mem_ty = np.ndarray[(1, self.num_heads * self.SWA_DH), np.dtype[bfloat16]]
        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}) with, d: { self.num_heads * self.DH }")

        self.q_buffer = aie.buffer(self.mt, self.q_mem_ty, name = f"q_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.o_buffer = aie.buffer(self.mt, self.o_mem_ty, name = f"o_mem_buffer_0_{self.mt_row}_{self.mt_col}")


        self.swa_q_buffer = aie.buffer(self.mt, self.swa_q_mem_ty, name = f"swa_q_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.swa_o_buffer = aie.buffer(self.mt, self.swa_o_mem_ty, name = f"swa_o_mem_buffer_0_{self.mt_row}_{self.mt_col}")

        self.q_prod_lock = aie.lock(self.mt, lock_id = 0, init = 1)
        self.q_cons_lock_0 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.q_cons_lock_1 = aie.lock(self.mt, lock_id = 2, init = 0)
        self.o_prod_lock_0 = aie.lock(self.mt, lock_id = 3, init = 1)
        self.o_prod_lock_1 = aie.lock(self.mt, lock_id = 4, init = 0)
        self.o_cons_lock = aie.lock(self.mt, lock_id = 5, init = 0)

        self.swa_q_prod_lock = aie.lock(self.mt, lock_id = 6, init = 1)
        self.swa_q_cons_lock_0 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.swa_q_cons_lock_1 = aie.lock(self.mt, lock_id = 8, init = 0)
        self.swa_o_prod_lock_0 = aie.lock(self.mt, lock_id = 9, init = 1)
        self.swa_o_prod_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        self.swa_o_cons_lock = aie.lock(self.mt, lock_id = 11, init = 0)
        
        self.q_mem_row_size = self.num_heads * self.DH
        self.q_mem_cu_row_size = self.Q_HEAD_PER_ROUND * self.DH
        self.q_offset = self.q_mem_cu_row_size

        self.o_mem_row_size = self.num_heads * self.DH
        self.o_mem_cu_row_size = self.Q_HEAD_PER_ROUND * self.DH
        self.o_offset = self.o_mem_cu_row_size

        self.swa_q_mem_row_size = self.num_heads * self.SWA_DH
        self.swa_q_mem_cu_row_size = self.SWA_Q_HEAD_PER_ROUND * self.SWA_DH
        self.swa_q_offset = self.swa_q_mem_cu_row_size

        self.swa_o_mem_row_size = self.num_heads * self.SWA_DH
        self.swa_o_mem_cu_row_size = self.SWA_Q_HEAD_PER_ROUND * self.SWA_DH
        self.swa_o_offset = self.swa_o_mem_cu_row_size
        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

            with block[1]:
                aie.use_lock(self.q_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.q_buffer)
                aie.use_lock(self.q_cons_lock_0, aie.LockAction.Release, value=1)
                aie.next_bd(block[1])

            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

            with block[4]:
                aie.use_lock(self.q_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.q_buffer, offset = 0, len = self.q_mem_cu_row_size)
                aie.use_lock(self.q_cons_lock_1, aie.LockAction.Release, value=1)
                aie.next_bd(block[4])

            with block[6]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[9])

            with block[7]:
                aie.use_lock(self.q_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.q_buffer, offset = self.q_offset, len = self.q_mem_cu_row_size)
                aie.use_lock(self.q_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[10], chain=block[12])

            with block[10]:
                aie.use_lock(self.o_prod_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.o_mem_cu_row_size)
                aie.use_lock(self.o_prod_lock_1, aie.LockAction.Release, value=1)
                aie.next_bd(block[10])

            with block[12]:
                s4 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[13], chain=block[15])

            with block[13]:
                aie.use_lock(self.o_prod_lock_1, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd(self.o_buffer, offset = self.o_offset, len = self.o_mem_cu_row_size)
                aie.use_lock(self.o_cons_lock, aie.LockAction.Release, value = 1)
                aie.next_bd(block[13])

            with block[15]:
                s5 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[16], chain=block[18])

            with block[16]:
                aie.use_lock(self.o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd_packet(0, to_proj_x_in_pkt_id.from_attn)
                aie.dma_bd(self.o_buffer)
                aie.use_lock(self.o_prod_lock_0, aie.LockAction.Release, value = 1)
                aie.next_bd(block[16])

            with block[18]:
                s6 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[25], chain=block[27])

            with block[25]:
                aie.use_lock(self.swa_q_prod_lock, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_q_buffer)
                aie.use_lock(self.swa_q_cons_lock_0, aie.LockAction.Release, value=1)
                aie.next_bd(block[25])

            with block[27]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[28], chain=block[30])

            with block[28]:
                aie.use_lock(self.swa_q_cons_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_q_buffer, offset = 0, len = self.swa_q_mem_cu_row_size)
                aie.use_lock(self.swa_q_cons_lock_1, aie.LockAction.Release, value=1)
                aie.next_bd(block[28])

            with block[30]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[31], chain=block[33])

            with block[31]:
                aie.use_lock(self.swa_q_cons_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_q_buffer, offset = self.swa_q_offset, len = self.swa_q_mem_cu_row_size)
                aie.use_lock(self.swa_q_prod_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[31])

            with block[33]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 4, dest=block[34], chain=block[36])

            with block[34]:
                aie.use_lock(self.swa_o_prod_lock_0, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_o_buffer, offset = 0, len = self.swa_o_mem_cu_row_size)
                aie.use_lock(self.swa_o_prod_lock_1, aie.LockAction.Release, value=1)
                aie.next_bd(block[34])

            with block[36]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 5, dest=block[37], chain=block[39])

            with block[37]:
                aie.use_lock(self.swa_o_prod_lock_1, aie.LockAction.AcquireGreaterEqual, value=1)
                aie.dma_bd(self.swa_o_buffer, offset = self.swa_o_offset, len = self.swa_o_mem_cu_row_size)
                aie.use_lock(self.swa_o_cons_lock, aie.LockAction.Release, value=1)
                aie.next_bd(block[37])

            with block[39]:
                s7 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest=block[40], chain=block[42])

            with block[40]:
                aie.use_lock(self.swa_o_cons_lock, aie.LockAction.AcquireGreaterEqual, value = 1)
                aie.dma_bd_packet(0, to_proj_x_in_pkt_id.from_swa_attn)
                aie.dma_bd(self.swa_o_buffer)
                aie.use_lock(self.swa_o_prod_lock_0, aie.LockAction.Release, value = 1)
                aie.next_bd(block[40])

            with block[42]:
                aie.EndOp()

    @property
    def q_in(self):
        return self.S2MM[0]
    
    @property
    def q_out0(self):
        return self.MM2S[0]
    
    @property
    def q_out1(self):
        return self.MM2S[1]
    
    @property
    def o_in0(self):
        return self.S2MM[1]
    
    @property
    def o_in1(self):
        return self.S2MM[2]
    
    @property
    def o_out(self):
        return self.MM2S[2]
    
    @property
    def SWA_q_in(self):
        return self.S2MM[3]
    
    @property
    def SWA_q_out0(self):
        return self.MM2S[3]
    
    @property
    def SWA_q_out1(self):
        return self.MM2S[4]
    
    @property
    def SWA_o_in0(self):
        return self.S2MM[4]
    
    @property
    def SWA_o_in1(self):
        return self.S2MM[5]
    
    @property
    def SWA_o_out(self):
        return self.MM2S[5]