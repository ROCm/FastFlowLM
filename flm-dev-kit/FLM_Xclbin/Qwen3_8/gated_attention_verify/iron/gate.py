# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

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
from aie.extras.dialects.ext.arith import constant
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
from aie.dialects import memref
from typing import List
from utils import dma_port, mlir_context, tile_with_location, connect_dma, lock_acquire, lock_release
from utils import AIE_CT_IP, comment_print


def _is_north_of(mem: tile_with_location, core: tile_with_location) -> bool:
    """Is `mem` directly north of `core`?

    The kernel reaches the shared row through its north window (_up_lock_*), so
    the helper must sit directly south of the emitter. Wrong placement hangs
    rather than failing to build -- the same trap as the qk/kv pairing.
    """
    return mem.col == core.col and mem.row == core.row + 1


class gate_tile(AIE_CT_IP):
    """Half a row of `attn * sigmoid(z)`, for one attention column.

    Two of these, one per attention column. Column 4 computes q heads 0..11 --
    dims [0, DQ/2) of the row -- and column 5 computes heads 12..23, so each
    gate tile owns the half its column produced.

    **Why the gate is split rather than the attention merged.** Merging the two
    halves first would need three inbound streams on a tile that has two S2MM
    channels. `attn * sigmoid(z)` is elementwise, so a half-row of attention only
    ever meets the matching half of z: splitting the gate gives each tile exactly
    two inbound streams and the halves run concurrently.

    Free on the z side too -- DQ/2 is a whole number of the engine's 32-row
    blocks, so "which half" is only where the staging BD starts.

    **The row is rejoined through neighbour memory.** Downstream (o_proj) wants
    the whole DQ row. Two flows into one port on the replay tile cannot express
    that -- two producers on one S2MM channel must be one merged packet flow, and
    which half lands first would be a timing race. So the tiles sit in the same
    column, adjacent; the output buffer is a whole DQ row allocated once on the
    emitter, and the helper writes its half into it through the north window, no
    DMA. One MM2S sends the assembled row.

    `partner = None` makes a tile the **emitter** (owns the row buffer and the
    outbound channel); otherwise it is the **helper**, which has none.

    Channel map, emitter:

        S2MM ch0   z, this half, token-major, from MT[3]
        S2MM ch1   the attention half, from this tile's attention column
        MM2S ch0   the assembled DQ row -> MT[2], as o_proj's x

    The helper is the same minus the MM2S channel: both its S2MM channels are
    spoken for, which is the constraint the arrangement is built around.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig,
                 half: int, stack_size: int = 1024, link_with: str = "gate.o",
                 partner: 'gate_tile' = None):
        """
        :param half: which half this tile computes -- 0 is attention column 4
            (q heads 0..11), 1 is column 5. Also its offset into the shared row,
            so it must agree with the column feeding `Attn_In`.
        :param partner: the emitter, if this tile is the helper. None makes this
            tile the emitter.
        """
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        self.DQ = model_config.DQ
        self.D = model_config.D
        self.half = half
        self.is_emitter = partner is None
        assert half in (0, 1)

        # Half a row: this tile's attention column's share. Must equal
        # GATE_WIDTH in kernels/gate.cc -- the kernel derives it from DQ the
        # same way, but nothing cross-checks the two files.
        self.width = self.DQ // 2
        assert self.width % 32 == 0, "apply_gate steps 32 lanes at a time"

        # Sends per token. 1 because nothing replays this in-tile; the L = 1
        # whole-layer design set it to the round count. Matches kernels/gate.cc.
        self.repeats = 1

        self.gate_ty = np.ndarray[(self.width,), self.dtype]
        self.row_ty = np.ndarray[(self.DQ,), self.dtype]
        comment_print(f"Initializing gate_tile at ({self.ct_row}, {self.ct_col}): "
                      f"half {half}, {self.width} bf16, "
                      f"{'emitter' if self.is_emitter else 'helper'}")

        self.gate_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"gate_{self.ct_row}_{self.ct_col}")
        self.attn_out_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"attn_out_{self.ct_row}_{self.ct_col}")

        if self.is_emitter:
            # The whole row, allocated once. The helper writes its half into it
            # directly, so this is the only copy of the assembled activation.
            self.gated_attn_out_buffer = aie.buffer(
                tile = self.ct, datatype = self.row_ty,
                name = f"gated_attn_out_{self.ct_row}_{self.ct_col}")
        else:
            if not _is_north_of(partner.located_tile, self.located_tile):
                raise ValueError(
                    f"the gate emitter at ({partner.ct_row}, {partner.ct_col}) must sit "
                    f"directly north of the helper at ({self.ct_row}, {self.ct_col}): "
                    f"the kernel reaches the shared row through its north window "
                    f"(_up_lock_acquire). Any other placement hangs.")
            self.gated_attn_out_buffer = partner.gated_attn_out_buffer

        # Lock ids are the kernel's constants; see the block at the top of
        # kernels/gate.cc.
        self.gate_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)
        self.gate_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.attn_out_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.attn_out_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        # Both tiles declare this pair but only the emitter's is used: the helper
        # takes the emitter's across the tile boundary (_up_lock_*), since the
        # buffer lives there. Declaring on both keeps one compiled layout.
        #
        # **Counted in halves, not rows.** Each core takes and releases one
        # credit per token, so the row costs 2 of each: init 2, and the outbound
        # BD acquires 2. Acquiring 1 sends a half-written row *and* drifts --
        # that made tokens 0..4 right and everything from 5 on garbage.
        self.halves = 2
        self.gated_attn_out_prod_lock = aie.lock(self.ct, lock_id = 4,
                                                 init = self.repeats * self.halves)
        self.gated_attn_out_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        # ctx.func, not aie.external_func: the declaration is module-scope, so
        # the second tile would be a redefinition error at parse time.
        self.gate = ctx.func("gate", [self.row_ty, self.gate_ty, self.gate_ty,
                                      np.int32, np.int32])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                # `half` is where this core writes into the shared row;
                # `is_emitter` is which lock window guards it. Separate on
                # purpose -- see the kernel's header comment.
                self.gate(self.gated_attn_out_buffer, self.gate_buffer,
                          self.attn_out_buffer,
                          constant(self.half), constant(1 if self.is_emitter else 0))

        @aie.mem(self.ct)
        def mem_body(block):
            # ---- S2MM ch0: z, this tile's half, from the staging memtile -----
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[2])
            with block[1]:
                lock_acquire(self.gate_prod_lock, 1)
                aie.dma_bd(self.gate_buffer)
                lock_release(self.gate_cons_lock, 1)
                aie.next_bd(block[1])

            # ---- S2MM ch1: the attention half, from this column's memtile ----
            with block[2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1,
                              dest = block[3], chain = block[4])

            with block[3]:
                lock_acquire(self.attn_out_prod_lock, 1)
                aie.dma_bd(self.attn_out_buffer)
                lock_release(self.attn_out_cons_lock, 1)
                aie.next_bd(block[3])

            if not self.is_emitter:
                # The helper has no outbound channel: its result is already in
                # the emitter's buffer, written through the tile boundary.
                with block[4]:
                    aie.EndOp()
                return

            # ---- MM2S ch0: the assembled row -> o_proj's replay buffer -------
            # **Both counts are `halves`, not 1.** Each of the two cores
            # releases one cons credit per token, so a row is worth 2; acquiring
            # 1 sends a half-written row and drifts a credit per token, which
            # reads as "the first few tokens are right and then it hangs".
            #
            # A packet flow because the replay tile's landing channel is already
            # a packet destination for the rms tile's rows, and a switchbox will
            # not drive one DMA channel from both a circuit and a packet flow.
            with block[4]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0,
                              dest = block[5], chain = block[6])

            with block[5]:
                lock_acquire(self.gated_attn_out_cons_lock, self.halves)
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_out_proj)
                aie.dma_bd(self.gated_attn_out_buffer)
                lock_release(self.gated_attn_out_prod_lock, self.halves)
                aie.next_bd(block[5])

            with block[6]:
                aie.EndOp()

    @property
    def Gate_In(self) -> dma_port:
        """z, this tile's half."""
        return self.S2MM[0]

    @property
    def Attn_In(self) -> dma_port:
        """The attention half, from this tile's own attention column."""
        return self.S2MM[1]

    @property
    def Attn_Out_replay(self) -> dma_port:
        """The assembled DQ row -> proj_x_replay_mt, as o_proj's x. Emitter only."""
        assert self.is_emitter, "only the emitter has an outbound channel"
        return self.MM2S[0]
