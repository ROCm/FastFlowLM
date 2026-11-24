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

class aux_memtile(AIE_MT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        # aux mt sorts the proj engine outputs by their fixed order in a layer
        # (X_in, pkt to_aux):  out_projection  -> to RMS, MM2S 1
        #                      down_projection -> to RMS, MM2S 1
        # (GLU_IN, pkt to_glu): up_gate_projection -> to GLU, MM2S 2

        # for pipelines, all DMA are seperated into 4 partials (ping-ponged, 4 * 4 = 16 bds, plenty)
        super().__init__(tile)
        self.model_config = model_config
        self.D = model_config.D
        self.intermediate_size = model_config.intermediate_size
        self.out_down_proj_ty = np.ndarray[(self.D // 4,), self.dtype]
        # A BD hands its piece on only when full, so the piece must divide every
        # amount that flows to the GLU in a call: shared up/gate 2 * I, routed
        # up/gate 4 * 2 * I and routed down 4 * D per round. Half a GLU slice (256)
        # does; the old 2 * I / 4 (384 at I = 768) left routed down's last piece
        # unsent and the GLU waited for it forever.
        self.glu_slice_ty = np.ndarray[(model_config.glu_slice // 2,), self.dtype]
        for total in (2 * self.intermediate_size, 8 * self.intermediate_size, 4 * self.D):
            assert total % (model_config.glu_slice // 2) == 0

        self.out_prod_lock  = aie.lock(self.mt, lock_id = 2, init = 2)
        self.out_cons_lock  = aie.lock(self.mt, lock_id = 3, init = 0)
        self.glu_slice_prod_lock  = aie.lock(self.mt, lock_id = 4, init = 2)
        self.glu_slice_cons_lock  = aie.lock(self.mt, lock_id = 5, init = 0)
        self.down_prod_lock  = aie.lock(self.mt, lock_id = 6, init = 2)
        self.down_cons_lock  = aie.lock(self.mt, lock_id = 7, init = 0)

        self.out_down_buffer_0 = aie.buffer(tile = self.mt, datatype = self.out_down_proj_ty, name = f"out_down_buffer_0_{self.mt_row}_{self.mt_col}")
        self.out_down_buffer_1 = aie.buffer(tile = self.mt, datatype = self.out_down_proj_ty, name = f"out_down_buffer_1_{self.mt_row}_{self.mt_col}")
        self.down_buffer_0 = aie.buffer(tile = self.mt, datatype = self.out_down_proj_ty, name = f"down_buffer_0_{self.mt_row}_{self.mt_col}")
        self.down_buffer_1 = aie.buffer(tile = self.mt, datatype = self.out_down_proj_ty, name = f"down_buffer_1_{self.mt_row}_{self.mt_col}")        
        self.glu_slice_buffer_0 = aie.buffer(tile = self.mt, datatype = self.glu_slice_ty, name = f"glu_slice_buffer_0_{self.mt_row}_{self.mt_col}")
        self.glu_slice_buffer_1 = aie.buffer(tile = self.mt, datatype = self.glu_slice_ty, name = f"glu_slice_buffer_1_{self.mt_row}_{self.mt_col}")

        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[5], chain = block[50])

            with block[5]:
                lock_acquire(self.out_prod_lock, 1)
                aie.dma_bd(self.out_down_buffer_0)
                lock_release(self.out_cons_lock, 1)
                aie.next_bd(block[6])

            with block[6]:
                lock_acquire(self.out_prod_lock, 1)
                aie.dma_bd(self.out_down_buffer_1)
                lock_release(self.out_cons_lock, 1)
                aie.next_bd(block[7])
            
            with block[7]:
                lock_acquire(self.out_prod_lock, 1)
                aie.dma_bd(self.out_down_buffer_0)
                lock_release(self.out_cons_lock, 1)
                aie.next_bd(block[8])
            
            with block[8]:
                lock_acquire(self.out_prod_lock, 1)
                aie.dma_bd(self.out_down_buffer_1)
                lock_release(self.out_cons_lock, 1)
                aie.next_bd(block[13])

            with block[13]:
                lock_acquire(self.down_prod_lock, 1)
                aie.dma_bd(self.down_buffer_0)
                lock_release(self.down_cons_lock, 1)
                aie.next_bd(block[14])

            with block[14]:
                lock_acquire(self.down_prod_lock, 1)
                aie.dma_bd(self.down_buffer_1)
                lock_release(self.down_cons_lock, 1)
                aie.next_bd(block[15])
            
            with block[15]:
                lock_acquire(self.down_prod_lock, 1)
                aie.dma_bd(self.down_buffer_0)
                lock_release(self.down_cons_lock, 1)
                aie.next_bd(block[16])

            with block[16]:
                lock_acquire(self.down_prod_lock, 1)
                aie.dma_bd(self.down_buffer_1)
                lock_release(self.down_cons_lock, 1)
                aie.next_bd(block[5])


            with block[50]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[51], chain=block[100] )
                
            with block[51]:
                lock_acquire(self.glu_slice_prod_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_0)
                lock_release(self.glu_slice_cons_lock, 1)
                aie.next_bd(block[52])
            
            with block[52]:
                lock_acquire(self.glu_slice_prod_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_1)
                lock_release(self.glu_slice_cons_lock, 1)
                aie.next_bd(block[53])

            with block[53]:
                lock_acquire(self.glu_slice_prod_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_0)
                lock_release(self.glu_slice_cons_lock, 1)
                aie.next_bd(block[54])

            with block[54]:
                lock_acquire(self.glu_slice_prod_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_1)
                lock_release(self.glu_slice_cons_lock, 1)
                aie.next_bd(block[51])

            with block[100]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[201], chain = block[300])

            with block[201]:
                lock_acquire(self.out_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.out_down_buffer_0)
                lock_release(self.out_prod_lock, 1)
                aie.next_bd(block[202])

            with block[202]:
                lock_acquire(self.out_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.out_down_buffer_1)
                lock_release(self.out_prod_lock, 1)
                aie.next_bd(block[203])
            
            with block[203]:
                lock_acquire(self.out_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.out_down_buffer_0)
                lock_release(self.out_prod_lock, 1)
                aie.next_bd(block[204])

            with block[204]:
                lock_acquire(self.out_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.out_down_buffer_1)
                lock_release(self.out_prod_lock, 1)
                aie.next_bd(block[205])

            with block[205]:
                lock_acquire(self.down_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.down_buffer_0)
                lock_release(self.down_prod_lock, 1)
                aie.next_bd(block[206])

            with block[206]:
                lock_acquire(self.down_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.down_buffer_1)
                lock_release(self.down_prod_lock, 1)
                aie.next_bd(block[207])

            with block[207]:
                lock_acquire(self.down_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.down_buffer_0)
                lock_release(self.down_prod_lock, 1)
                aie.next_bd(block[208])

            with block[208]:
                lock_acquire(self.down_cons_lock, 1)
                aie.dma_bd_packet(0, it1_pkt_id.to_rms)
                aie.dma_bd(self.down_buffer_1)
                lock_release(self.down_prod_lock, 1)
                aie.next_bd(block[201])

            with block[300]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[301], chain = block[400])

            with block[301]:
                lock_acquire(self.glu_slice_cons_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_0)
                lock_release(self.glu_slice_prod_lock, 1)
                aie.next_bd(block[302])

            with block[302]:
                lock_acquire(self.glu_slice_cons_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_1)
                lock_release(self.glu_slice_prod_lock, 1)
                aie.next_bd(block[303])

            with block[303]:
                lock_acquire(self.glu_slice_cons_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_0)
                lock_release(self.glu_slice_prod_lock, 1)
                aie.next_bd(block[304])
            
            with block[304]:
                lock_acquire(self.glu_slice_cons_lock, 1)
                aie.dma_bd(self.glu_slice_buffer_1)
                lock_release(self.glu_slice_prod_lock, 1)
                aie.next_bd(block[301])

            with block[400]:
                aie.EndOp()

    @property
    def X_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def OUT(self) -> dma_port:
        return self.MM2S[1]  
    
    
    @property
    def GLU_IN(self)->dma_port:
        return self.S2MM[1]
    
    @property
    def GLU_out(self) -> dma_port:
        return self.MM2S[2]
