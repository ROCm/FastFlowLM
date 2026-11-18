# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

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

from utils import mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP

class AB_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "ab_proj.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing AB_tile at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.D = model_config.hidden_size
        self.alpha_beta_size = model_config.linear_num_heads_padded * 2
        self.w_block_size = model_config.AB_W_BLOCK_SIZE
        
        self.w_block_ty = np.ndarray[(self.w_block_size,), self.dtype]
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

        # Lock ids 4 and 5 used to gate a dedicated inbound BD for the activation
        # bias. The bias rides in as the first block of every token's weight
        # stream now (see mem_dma_body), so nothing DMAs into
        # alpha_beta_w_buffer and these are free. Left declared because
        # ab_proj.cc names the y locks by their literal ids -- renumbering here
        # would silently repoint them.
        self.alpha_beta_w_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.alpha_beta_w_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        self.y_prod_lock = aie.lock(self.ct, lock_id = 6, init = 1)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        
        self.ab_proj = ctx.func("ab_proj", [self.y_ty, self.w_block_ty, self.w_block_ty, self.x_ty, self.alpha_beta_w_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.ab_proj(self.y_buffer, self.w_buffer_0, self.w_buffer_1, self.x_buffer, self.alpha_beta_w_buffer)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            # Stage 4 feeds this tile from two shim channels instead of MT[5]:
            # ch0 is x straight from the rms tile, ch1 is the q8_0 weight stream
            # off IT5 MM2S ch0, one full a/b weight set per token, each set
            # preceded by the block carrying the activation bias. The designed
            # wiring puts the memtile in between and alternates x with the bias
            # on ch0, but MT[5]'s S2MM chain also fills the conv and delta-norm
            # weights, which do not exist yet.
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[3], chain = block[5])

            # A uniform ping/pong ring, and it has to stay uniform. This chain
            # used to open with a one-shot prologue BD that landed the 512 B
            # activation bias into alpha_beta_w_buffer -- nothing chained back to
            # it, so it fired once per *xclbin load*, not once per pass. That
            # made the design single-dispatch: pass two's bias was taken by the
            # w_buffer_0 BD below, every block after it was shifted by 512 B, the
            # last one never completed and the host parked on its DMA wait until
            # XRT's 60 s timeout. The engine dispatches this bitstream 48 times
            # per verify pass, so the prologue had to go.
            #
            # The bias rides in as the first AB_W_BLOCK_SIZE block of every
            # token's weight stream instead, and ab_proj() copies it out of the
            # w buffer before it starts the projection. One block per token is
            # 8.5 KB against the 696 KB of a/b weights that follow it.
            with block[3]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[4])

            with block[4]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[3])

            with block[5]:
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[6], chain = block[7])

            with block[6]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[6])

            with block[7]:
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
