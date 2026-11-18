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
    """SwiGLU on the fused up | gate projection: silu(gate) * up.

    Two rounds in, accumulate, one whole row out. The projection engine
    assembles its y one *round* at a time -- 16 cores x q4nx_row_block_size rows
    = m_per_round output columns -- and the host weaves the up and gate weight
    rows in blocks of exactly m_per_round, so a round is either all up or all
    gate and the two alternate. glu_slice is two rounds wide, so one x buffer
    holds an (up, gate) pair and the kernel's split at glu_slice/2 lands on the
    boundary by construction.

    At L = 1 the whole hidden result is intermediate_size bf16 = 34 KiB, which
    fits this tile, so hid is accumulated across all the pairs and leaves once.
    That is what `replays` encodes -- and it is the L = 1 shape specifically;
    ../gated_attention_verify runs L = 8, where 278 KiB cannot be held at all
    and the tile emits a round at a time instead.
    """
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "glu.o", replays: int = 1):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.intermediate_size = model_config.intermediate_size
        self.glu_slice = model_config.glu_slice
        self.up_gate_ty = np.ndarray[(self.glu_slice,), self.dtype]
        self.hid_ty = np.ndarray[(self.intermediate_size,), self.dtype]
        self.D = model_config.D

        # 16 is a constant of 16 CTs in the projection engine; mirrors
        # M_PER_ROUND in kernels/aie_array_layout.h.
        self.m_per_round = model_config.q4nx_row_block_size * 16
        assert self.glu_slice == 2 * self.m_per_round, \
            "a glu_slice must be exactly one up round plus one gate round"
        assert self.intermediate_size % self.m_per_round == 0, \
            "m_per_round must divide intermediate_size: a round has to be all up or all gate"

        # How many times the assembled hid row is sent.
        #
        # down_proj is a streaming mvm(): it re-reads the whole hidden row once
        # per row block, and it owns D // m_per_round = 10 of them. So this
        # replay *is* down's M/m loop, exactly as gate_tile.repeats is o_proj's.
        # Must stay equal to GLU_OUT_REPLAYS in kernels/aie_array_layout.h: the
        # core takes that many credits before writing hid and releases that many
        # after, and the outbound BD takes one per send.
        #
        # 10 is well under the lock's 63 ceiling. up|gate needed 68 and had to
        # go resident instead; this one does not.
        self.replays = replays
        assert self.replays == self.D // self.m_per_round, \
            "the hid replay count is down_proj's row-block count"
        comment_print(f"GLU tile at ({self.ct_row}, {self.ct_col}) with intermediate_size: {self.intermediate_size}, glu_slice: {self.glu_slice}, replays: {self.replays}")
        # q_proj tile and mem
        # double buffer
        self.x_buffer_0 = aie.buffer(tile = self.ct, datatype = self.up_gate_ty, name = f"x_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = aie.buffer(tile = self.ct, datatype = self.up_gate_ty, name = f"x_1_{self.ct_row}_{self.ct_col}")

        self.hid_buffer = aie.buffer(tile = self.ct, datatype = self.hid_ty, name = f"hid_0_{self.ct_row}_{self.ct_col}")
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.hid_prod_lock = aie.lock(self.ct, lock_id = 2, init = self.replays)
        self.hid_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.glu = aie.external_func("glu", [self.hid_ty, self.up_gate_ty, self.up_gate_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.glu(self.hid_buffer, self.x_buffer_0, self.x_buffer_1)
        
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
                
            # The assembled hid row out, into the projection engine's x FIFO as
            # down_proj's activation. One credit per send and the core released
            # `replays` of them, so this self-chained BD fires that many times
            # per invocation -- which is down's M/m loop.
            #
            # **dma_bd_packet, and with an id of its own.** The engine's x FIFO
            # is packet-switched and the gate tile already enters it under
            # to_out_proj; two producers wanting one id on one slave port is
            # what the arbiter allocator cannot express. Note the contrast with
            # the rms and passthrough tiles, which stamp their id in band at
            # y+14 because one buffer there carries different ids on different
            # phases -- this row only ever goes one place.
            with block[4]: # hid, to down_proj's x
                lock_acquire(self.hid_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_down_proj)
                aie.dma_bd(self.hid_buffer)
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
