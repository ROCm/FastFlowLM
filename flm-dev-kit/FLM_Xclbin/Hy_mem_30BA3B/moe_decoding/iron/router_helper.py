# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import sys
import os
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_

from utils import mlir_context, tile_with_location, comment_print, lock_acquire, lock_release
from utils import AIE_CT_IP


class router_helper(AIE_CT_IP):
    """Router logits of experts num_experts/2 .. num_experts-1.

    The router weight comes in over 2 streams (IT5 channel 0 -> router CT,
    IT5 channel 1 -> this tile), so each tile streams half of it. This tile has
    its own weight blocks; it reads x from the router's memory and writes its
    logits there (it sits right below the router CT).
    """
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, router,
                 stack_size: int = 1024, link_with: str = "router_helper.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing router_helper at ({self.ct_row}, {self.ct_col})")
        self.router = router

        # same block shape as the router's: 32 experts x 256 inputs, double buffered
        self.w_buffer_0 = aie.buffer(tile=self.ct, datatype=router.weight_ty, name=f"router_helper_w_0_{self.ct_row}_{self.ct_col}")
        self.w_buffer_1 = aie.buffer(tile=self.ct, datatype=router.weight_ty, name=f"router_helper_w_1_{self.ct_row}_{self.ct_col}")
        self.w_prod_lock = aie.lock(self.ct, lock_id=0, init=2)
        self.w_cons_lock = aie.lock(self.ct, lock_id=1, init=0)

        self.router_helper = ctx.func("router_helper", [
            router.up_neighbor_hidden_state_in_buffer_ty,
            router.weight_ty, router.weight_ty,
            router.up_neighbor_top_k_expert_score_buffer_ty,
        ])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.router_helper(
                    self.router.ct_hidden_state_in_buffer,
                    self.w_buffer_0, self.w_buffer_1,
                    self.router.ct_top_k_expert_score_local,
                )

        @aie.mem(self.ct)
        def mem_body(block):
            aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
            with block[1]:
                lock_acquire(self.w_prod_lock, value=1)
                aie.dma_bd(self.w_buffer_0)
                lock_release(self.w_cons_lock, value=1)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(self.w_prod_lock, value=1)
                aie.dma_bd(self.w_buffer_1)
                lock_release(self.w_cons_lock, value=1)
                aie.next_bd(block[1])
            with block[3]:
                aie.EndOp()

    @property
    def w_in(self):
        return self.S2MM[0]
