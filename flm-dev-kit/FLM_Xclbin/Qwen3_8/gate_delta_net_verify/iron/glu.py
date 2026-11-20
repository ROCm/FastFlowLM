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
    """SwiGLU on the fused up | gate projection: activation(gate) * up.

    Round in, round out. The projection engine assembles its y one *round* at a
    time -- 16 cores x q4nx_row_block_size rows = m_per_round output columns for
    all L_TOKENS token rows -- and load_weights() weaves the up and gate weight
    rows in blocks of exactly m_per_round, so a round is either all up or all
    gate and the two alternate. This tile takes two rounds and emits one, and
    since both operands arrive under the identical [core][token][row]
    permutation the kernel is a flat elementwise pass; the un-permute is left to
    the shim BD that writes the result to DDR.

    Buffers are two rounds in (ping/pong) and two rounds out (ping/pong), i.e.
    4 * L_TOKENS * m_per_round bf16 -- 32 KB at L_TOKENS = 8, half the tile.

    This is not the L = 1 tile it replaces. There, hid was accumulated whole
    (intermediate_size bf16) and replayed D / (32 * 16) times straight into the
    down projection, which is what hid_prod_lock's `repeats` init encoded. At
    L_TOKENS = 8 the whole hidden result is 278 KB, four times the tile, so it
    cannot be held at all -- it leaves a round at a time. Feeding the down
    projection from here again means giving MT[2] (or another memtile) a third
    replay spec, not restoring the old lock init.
    """
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "glu.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.intermediate_size = model_config.intermediate_size
        # 16 is a constant of 16 CTs in the projection engine; mirrors
        # M_PER_ROUND in kernels/aie_array_layout.h.
        self.m_per_round = model_config.q4nx_row_block_size * 16
        self.round_size = L_TOKENS * self.m_per_round
        self.round_ty = np.ndarray[(self.round_size,), self.dtype]
        self.D = model_config.D

        assert self.intermediate_size % self.m_per_round == 0, \
            "m_per_round must divide intermediate_size: a round has to be all up or all gate"
        self.pairs = self.intermediate_size // self.m_per_round
        comment_print(f"GLU tile at ({self.ct_row}, {self.ct_col}) with intermediate_size: {self.intermediate_size}, round: {self.round_size}, pairs: {self.pairs}")

        # double buffer, both sides
        self.x_buffer_0 = aie.buffer(tile = self.ct, datatype = self.round_ty, name = f"x_0_{self.ct_row}_{self.ct_col}")
        self.x_buffer_1 = aie.buffer(tile = self.ct, datatype = self.round_ty, name = f"x_1_{self.ct_row}_{self.ct_col}")

        self.hid_buffer_0 = aie.buffer(tile = self.ct, datatype = self.round_ty, name = f"hid_0_{self.ct_row}_{self.ct_col}")
        self.hid_buffer_1 = aie.buffer(tile = self.ct, datatype = self.round_ty, name = f"hid_1_{self.ct_row}_{self.ct_col}")

        # x_prod init 2: the kernel holds *both* input buffers at once (up and
        # gate are consumed together), so the DMA has to be allowed to fill both
        # before the core takes either.
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.hid_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.hid_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)

        self.glu = aie.external_func("glu", [self.round_ty, self.round_ty, self.round_ty, self.round_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.glu(self.hid_buffer_0, self.hid_buffer_1, self.x_buffer_0, self.x_buffer_1)

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

            # One round out per pair, circuit switched to the shim: a single
            # sink, so no packet header. 2 * pairs rounds in and pairs rounds out
            # are both even, so the two chains come back to their first BD
            # together at the end of every invocation.
            with block[4]:
                lock_acquire(self.hid_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_down_proj)
                aie.dma_bd(self.hid_buffer_0)
                lock_release(self.hid_prod_lock, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.hid_cons_lock, 1)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_down_proj)
                aie.dma_bd(self.hid_buffer_1)
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
