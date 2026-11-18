# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig

import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

from utils import comment_print, mlir_context, tile_with_location, lock_acquire, lock_release, AIE_CT_IP


class passthrough_tile(AIE_CT_IP):
    """Normalises the MTP fusion's two input halves and stages them for proj.

    The fc reduces over ``cat[rms(e, w_emb), rms(h, w_hid)]`` -- 2 * D = 10240
    bf16 = 20 KiB. Nothing on the array holds that, and nothing needs to: the
    projection engine eats x in ``x_slice_size`` slices and never looks at more
    than one at a time. So this tile forwards the two halves as two separate
    D-wide packets and the concat becomes their *arrival order*.

    The two halves take *different* gains -- ``pre_fc_norm_embedding`` for the
    embedding half, ``pre_fc_norm_hidden`` for the hidden half -- which is why
    the weights are two named buffers rather than one ping/pong pair. A toggle
    would work only as long as its parity stayed even, and landing on the wrong
    phase swaps the two gains: still plausible output, still wrong. Here the
    core names the buffer it wants, so there is no parity to desync.

    The tile is a core rather than a memtile BD for four reasons:

      1. the rms itself -- a BD cannot reduce,
      2. the packet id is stamped into the payload by the producing core,
      3. the two halves come from different DDR provenance, and
      4. it gives the fc input a DDR tap that can be scored on its own -- an
         end-to-end cosine would charge fc for this tile's error by a gain we
         cannot predict.

    Ping/pong parity on the x side is static: two halves per step, so the toggle
    in kernels/passthrough.cc lands back where it started and never desyncs from
    these BDs.
    """

    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig,
                 stack_size: int = 1024, link_with: str = "passthrough.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing passthrough_tile at ({self.ct_row}, {self.ct_col})")

        self.model_config = model_config
        self.D = model_config.D
        self.x_ty = np.ndarray[(self.D,), self.dtype]
        self.w_ty = np.ndarray[(self.D,), self.dtype]
        # 16 bf16 of packet header, payload at +16; the outbound BD starts at
        # 14 so the id word lands immediately in front of the data.
        self.y_pkt_ty = np.ndarray[(self.D + 16,), self.dtype]
        comment_print(f"passthrough_tile D: {self.D}")

        self.x_ping = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"pass_x_ping_{self.ct_row}_{self.ct_col}")
        self.x_pong = ctx.buffer(tile = self.located_tile, dtype = self.x_ty, name = f"pass_x_pong_{self.ct_row}_{self.ct_col}")
        # Named, not ping/pong: which gain goes on which half is fixed.
        self.w_embed = ctx.buffer(tile = self.located_tile, dtype = self.w_ty, name = f"pass_w_embed_{self.ct_row}_{self.ct_col}")
        self.w_hidden = ctx.buffer(tile = self.located_tile, dtype = self.w_ty, name = f"pass_w_hidden_{self.ct_row}_{self.ct_col}")
        self.y = ctx.buffer(tile = self.located_tile, dtype = self.y_pkt_ty, name = f"pass_y_{self.ct_row}_{self.ct_col}")

        # Lock ids are the ABI with kernels/passthrough.cc -- keep them in step.
        # w takes 0/1 to match rms_residual's numbering, so the same role reads
        # as the same id in both tiles.
        self.w_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.w_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.y_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.y_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.x_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        self.core_func = ctx.func("passthrough", [self.y_pkt_ty, self.x_ty, self.x_ty, self.w_ty, self.w_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.core_func(self.y, self.x_ping, self.x_pong, self.w_embed, self.w_hidden)

        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_ping, offset = 0, len = self.D)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_pong, offset = 0, len = self.D)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[1])

            # The two gains, embedding first. Two D-wide BDs rather than one
            # 2*D BD for the same reason the x side is split: each fills its own
            # buffer and releases once, and the core acquires once per half.
            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[4], chain = block[6])

            with block[4]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_embed, offset = 0, len = self.D)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.w_prod_lock, 1)
                aie.dma_bd(self.w_hidden, offset = 0, len = self.D)
                lock_release(self.w_cons_lock, 1)
                aie.next_bd(block[4])

            with block[6]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[7], chain = block[8])

            with block[7]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y, offset = 14, len = self.D + 2)
                lock_release(self.y_prod_lock, 1)
                aie.next_bd(block[7])

            with block[8]:
                aie.EndOp()

    @property
    def X_in(self):
        """The two raw halves land here, embedding first."""
        return self.S2MM[0]

    @property
    def W_in(self):
        """The two gains, embedding first. Point-to-point, no packet id."""
        return self.S2MM[1]

    @property
    def Y_out(self):
        """To the projection engine's x FIFO, and to the DDR tap."""
        return self.MM2S[0]
