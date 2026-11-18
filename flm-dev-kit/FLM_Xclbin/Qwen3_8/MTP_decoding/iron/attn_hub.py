# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie

from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_MT_IP, comment_print


class attn_hub_memtile(AIE_MT_IP):
    """MT[3]: q fan-out to the four attention CUs, and the final o collect.

    Two independent jobs on one otherwise-idle memtile, in opposite directions.

      * **q distribute.** The RoPE tile emits one DQ_PADDED-wide row -- four
        GQA groups of 8 heads, the two padding heads lying *inside* each group.
        This tile takes it whole on one S2MM and hands each CU its own group as
        a contiguous 2048-element slice on its own MM2S. The split is four
        offsets and no striding, because the padding is interior; the
        head-major -> dim-block-major transpose the mmul needs is done by the
        receiving qk tile's inbound BD, not here.

        One MM2S per CU rather than one packet-switched port with four ids: a
        memtile has six of each and this only needs four, so a circuit flow per
        destination is both available and simpler to route. The four fire in CU
        order off a lock chain so the buffer is released only after the last.

      * **o collect.** Each attention column assembles its own two CUs into a
        half-row (2 CUs x 6 real heads x DH = DQ/2) and sends it here. This tile
        lays the two halves down side by side and emits one dense DQ-wide row,
        which is what the shim writes to DDR with a single linear BD.

    The two jobs share no buffers and no locks. They are on the same tile
    because it is free and central: columns 4 and 5 are east of it and the RoPE
    tile is west, so both flows are short.

    Channel budget: S2MM 0 (q in), 2 and 3 (o in) of 6; MM2S 0..3 (q out) and 4
    (o out) of 6.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.DQ = model_config.DQ
        self.DH = model_config.DH
        self.num_attn_columns = model_config.num_attn_columns           # 2
        self.cus_per_column = model_config.cus_per_column               # 2
        self.num_cus = self.num_attn_columns * self.cus_per_column      # 4
        self.NQ_PER_CU = model_config.q_heads_per_group                 # 6, real
        self.NQ_PER_CU_PADDED = model_config.q_heads_per_group_padded   # 8, what arrives

        # What the RoPE tile sends: every group padded to 8 heads.
        self.q_slice = self.NQ_PER_CU_PADDED * self.DH                  # 2048
        self.DQ_PADDED = self.q_slice * self.num_cus                    # 8192
        # What comes back: only the real heads, so the row is dense and DQ wide.
        self.o_half = self.cus_per_column * self.NQ_PER_CU * self.DH    # 3072
        assert self.o_half * self.num_attn_columns == self.DQ, \
            f"o halves {self.o_half} x {self.num_attn_columns} do not cover DQ {self.DQ}"

        comment_print(f"Initializing attn_hub_memtile at ({self.mt_row}, {self.mt_col}): "
                      f"q {self.DQ_PADDED} -> {self.num_cus} x {self.q_slice}, "
                      f"o {self.num_attn_columns} x {self.o_half} -> {self.DQ}")

        self.q_ty = np.ndarray[(self.DQ_PADDED,), self.dtype]
        self.o_ty = np.ndarray[(self.DQ,), self.dtype]

        self.q_buffer = aie.buffer(self.mt, self.q_ty, name = f"q_hub_buffer_{self.mt_row}_{self.mt_col}")
        self.o_buffer = aie.buffer(self.mt, self.o_ty, name = f"o_hub_buffer_{self.mt_row}_{self.mt_col}")

        # q: filled once, then read by each CU in turn. The credit only comes
        # back to the inbound BD after the last CU has taken its slice, so the
        # next row cannot overwrite a group still in flight.
        # (here the chain does the
        # counting instead of an acquire of N, since the four reads are ordered.)
        self.q_prod_lock = aie.lock(self.mt, lock_id = 0, init = 1)
        self.q_cons_locks = [
            aie.lock(self.mt, lock_id = 1 + i, init = 0)
            for i in range(self.num_cus)
        ]

        # o: column 0's half lands, then column 1's, then the row leaves.
        self.o_prod_lock_0 = aie.lock(self.mt, lock_id = 5, init = 1)
        self.o_prod_lock_1 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.o_cons_lock   = aie.lock(self.mt, lock_id = 7, init = 0)

        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # ---- S2MM ch0: the whole padded q row from the RoPE tile ---------
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[2])

            with block[1]:
                lock_acquire(self.q_prod_lock, 1)
                aie.dma_bd(self.q_buffer)
                # Only the first CU is unblocked here; the rest are released by
                # the CU before them, so the four reads are serialised.
                lock_release(self.q_cons_locks[0], 1)
                aie.next_bd(block[1])

            # ---- MM2S ch0..ch3: one CU's padded GQA group each ---------------
            # Group c is [c * q_slice, + q_slice) -- contiguous, because
            # pseduo_rope_q() lays the two padding heads inside each group
            # rather than after all of them.
            for cu in range(self.num_cus):
                last = (cu == self.num_cus - 1)
                start = block[2 + 2 * cu]
                body  = block[3 + 2 * cu]
                nxt   = block[2 + 2 * (cu + 1)]
                with start:
                    aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = cu,
                                  dest = body, chain = nxt)
                with body:
                    lock_acquire(self.q_cons_locks[cu], 1)
                    aie.dma_bd(self.q_buffer, offset = cu * self.q_slice, len = self.q_slice)
                    lock_release(self.q_prod_lock if last else self.q_cons_locks[cu + 1], 1)
                    aie.next_bd(body)

            # blocks 2..9 are the four q MM2S pairs; the o chain starts at 10.
            o_base = 2 + 2 * self.num_cus

            # ---- S2MM ch2/ch3: the two columns' half-rows --------------------
            # One producer per channel: a column is one source, and merging both
            # onto a single channel would be a masterset error with no ordering
            # guarantee either way.
            with block[o_base]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2,
                              dest = block[o_base + 1], chain = block[o_base + 2])

            with block[o_base + 1]:
                lock_acquire(self.o_prod_lock_0, 1)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.o_half)
                lock_release(self.o_prod_lock_1, 1)
                aie.next_bd(block[o_base + 1])

            with block[o_base + 2]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3,
                              dest = block[o_base + 3], chain = block[o_base + 4])

            with block[o_base + 3]:
                lock_acquire(self.o_prod_lock_1, 1)
                aie.dma_bd(self.o_buffer, offset = self.o_half, len = self.o_half)
                lock_release(self.o_cons_lock, 1)
                aie.next_bd(block[o_base + 3])

            # ---- MM2S ch4: the assembled DQ-wide row, to the shim ------------
            with block[o_base + 4]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4,
                              dest = block[o_base + 5], chain = block[o_base + 6])

            with block[o_base + 5]:
                lock_acquire(self.o_cons_lock, 1)
                aie.dma_bd(self.o_buffer)
                lock_release(self.o_prod_lock_0, 1)
                aie.next_bd(block[o_base + 5])

            with block[o_base + 6]:
                aie.EndOp()

    @property
    def Q_in(self) -> dma_port:
        """The whole DQ_PADDED row from the RoPE tile."""
        return self.S2MM[0]

    def Q_out(self, cu: int) -> dma_port:
        """CU `cu`'s padded GQA group. CUs are numbered across columns:
        0 and 1 are column 0's, 2 and 3 are column 1's."""
        assert 0 <= cu < self.num_cus
        return self.MM2S[cu]

    def O_in(self, column: int) -> dma_port:
        """Column `column`'s assembled half-row."""
        assert 0 <= column < self.num_attn_columns
        return self.S2MM[2 + column]

    @property
    def O_out(self) -> dma_port:
        """The dense DQ-wide attention output, to DDR."""
        return self.MM2S[4]
