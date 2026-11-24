# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

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
    """Conv1d staging for the delta-net layer, L_TOKENS at a time.

    Carries the conv1d path plus the ssm_norm relay that feeds DeltaNet_Norm.
    The AB tile is fed straight from the shim, so nothing for a/b passes through
    here.

    x and w are held in *separate* buffers, which is the one structural change
    from the L=1 design.  That design interleaved them in a single
    (2 * CONV_SIZE, CONV_WIDTH) buffer -- even rows x, odd rows w -- and shipped
    one 8-row slice per 128-channel group.  That cannot express an L=8 sliding
    window in one BD: with x in rows 0..L+CONV_SIZE-2 and w appended after it,
    token t's tap sequence is x(t+0), w(0), x(t+1), w(1), ... whose intra-pair
    delta depends on t, and DMA strides are affine.  Replicating w per token
    would cost L_TOKENS * CONV_SIZE * CONV_WIDTH * 2 = 655 KB, past the memtile.

    So the conv core takes two input channels instead of one: S2MM ch0 gets the
    4 sliding x taps, S2MM ch1 the 4 fixed w taps.

    Layout:
      conv_x_buffer (L_TOKENS + CONV_SIZE - 1, CONV_WIDTH)   rows 0..CONV_SIZE-2
                                                             = incoming conv cache,
                                                             the rest = the L new tokens
      conv_w_buffer (CONV_SIZE, CONV_WIDTH)                  tap-major, as the
                                                             .q4nx already stores it
    """
    dtype = np.dtype[bfloat16]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        # for all linear layer weight transitions
        # PW_in -> ab_tile -> W_in, S2MM[0] -> MM2S[0]

        # Conv weights: W_in -> conv_core, S2MM[1] -> MM2S[1]
        # Delta_net_norm -> delta_net_norm -> S2MM[1] -> MM2S[2]
        # A and dt_bias for alpha_beta, -> send to conv tile X_in, append to the X_in

        # Weight order
        # | 8192 * 4 conv weights | 128 delta_net_norm weights | alpha_beta weights |

        # for pipelines, all DMA are seperated into 4 partials (ping-ponged, 4 * 4 = 16 bds, plenty)
        super().__init__(tile)
        self.model_config = model_config
        self.CONV_WIDTH = model_config.linear_DQ + model_config.linear_DK + model_config.linear_DV
        comment_print(f"Linear DQ: {model_config.linear_DQ}")
        comment_print(f"Linear DK: {model_config.linear_DK}")
        comment_print(f"Linear DV: {model_config.linear_DV}")
        self.DH = model_config.linear_v_head_dim
        self.CONV_SIZE  = model_config.linear_conv_kernel_size
        self.L = L_TOKENS
        # rows of conv_x_buffer: (CONV_SIZE - 1) cache rows + L new token rows
        self.X_ROWS = self.L + self.CONV_SIZE - 1
        self.m = model_config.q4nx_row_block_size
        self.delta_net_norm_size = model_config.linear_v_head_dim

        self.conv_x_ty = np.ndarray[(self.X_ROWS, self.CONV_WIDTH), self.dtype]
        self.conv_w_ty = np.ndarray[(self.CONV_SIZE, self.CONV_WIDTH), self.dtype]
        comment_print(f"Conv x ty: {self.conv_x_ty} ({self.X_ROWS * self.CONV_WIDTH * 2} B)")
        comment_print(f"Conv w ty: {self.conv_w_ty} ({self.CONV_SIZE * self.CONV_WIDTH * 2} B)")
        self.delta_norm_w_ty = np.ndarray[(self.delta_net_norm_size,), self.dtype]
        comment_print(f"delta_norm_ty: {self.delta_norm_w_ty}")

        self.conv_x_buffer = aie.buffer(tile = self.mt, datatype = self.conv_x_ty, name = f"mt_conv_x_buffer_{self.mt_row}_{self.mt_col}")
        self.conv_w_buffer = aie.buffer(tile = self.mt, datatype = self.conv_w_ty, name = f"mt_conv_w_buffer_{self.mt_row}_{self.mt_col}")
        # Stage 7b: ssm_norm, the gated norm's per-channel weight.  It sits
        # right after conv_w in rope_rms_buffer, so the same IT5 MM2S ch1
        # transfer that primes the taps fills this too -- see the S2MM ch1
        # chain below and _send_linear_conv_weights() on the host.
        self.delta_norm_buffer = aie.buffer(tile = self.mt, datatype = self.delta_norm_w_ty, name = f"delta_norm_buffer_{self.mt_row}_{self.mt_col}")

        # Lock budget, one invocation = L_TOKENS tokens.  Every count is <= L+1,
        # well inside the 6-bit memtile lock counter.
        #
        #   w_fill    : acq w_free(L)     rel w_ready(L)      chains to norm_fill
        #   norm_fill : acq norm_free(L)  rel norm_ready(L)   chains to x_cache_fill
        #   x_cache   : acq x_free(L+1)   rel cache_done(1)   chains back to w_fill
        #   qkv_in    : acq cache_done(1) rel x_ready(L+1)    self-chained
        #   w_out     : acq w_ready(1)    rel w_free(1)       self-chained, fires L times
        #   norm_out  : acq norm_ready(1) rel norm_free(1)    self-chained, fires L times
        #   x_out[t]  : acq x_ready(1)    rel x_free(1)       L chained BDs,
        #                                 and the last also rel cache_out_ready(1)
        #   cache_out : acq cache_out_ready(1) rel x_free(1)  self-chained, once
        #
        # cache_done is what keeps the qkv stream from racing the cache rows: the
        # cache write and the token writes land in the same buffer, and only the
        # lock orders them.
        #
        # The writeback gets its *own* ready lock rather than a share of x_ready,
        # and that is the whole trick.  An earlier version released x_ready L + 1
        # times and had cache_out acquire one of them.  The ledger balanced, but
        # a lock is a counting semaphore, not an assignment: the ch5 window chain
        # is circular, so once it has taken its L counts it loops straight back
        # to block[10] and races ch4 for the (L + 1)th.  ch5 usually wins, ch4
        # blocks forever, and the host's IT5 S2MM ch0 wait never returns -- a
        # 60 s XRT timeout and -nan out, not a wrong number.
        #
        # Chaining removes the race outright: the last window BD hands
        # cache_out_ready to the writeback, so cache_out cannot run before the
        # windows are done and nothing else can consume its count.  x_free still
        # counts L + 1, so the next invocation's cache fill in block[2] blocks
        # until the writeback has released -- the tail rows cannot be
        # overwritten while the readback is pending.  (Same shape as Qwen3.5's
        # conv_new_cache_out_cons_lock; see Qwen3_5/qwen3_5_decoding.)
        self.conv_w_free_lock = aie.lock(self.mt, lock_id = 0, init = self.L)
        self.conv_w_ready_lock = aie.lock(self.mt, lock_id = 1, init = 0)
        # L, not L + 1: the last window BD hands its count to NEW_CACHE_out
        # rather than releasing one of its own.  See the comment on block[2].
        self.conv_x_free_lock = aie.lock(self.mt, lock_id = 2, init = self.L)
        self.conv_x_ready_lock = aie.lock(self.mt, lock_id = 3, init = 0)
        self.conv_cache_done_lock = aie.lock(self.mt, lock_id = 4, init = 0)
        self.conv_cache_out_ready_lock = aie.lock(self.mt, lock_id = 7, init = 0)

        # Stage 7b.  The delta-norm kernel re-acquires its weight lock every
        # invocation, so the 128 words are replayed once per token rather than
        # held resident -- exactly what CONV_W_out does with the taps.
        self.delta_norm_free_lock = aie.lock(self.mt, lock_id = 5, init = self.L)
        self.delta_norm_ready_lock = aie.lock(self.mt, lock_id = 6, init = 0)

        self.__call__()

    def __call__(self):
        # One 128-channel slice of the window, four taps deep, as the conv core
        # wants it: slice outer, tap middle, channel innermost.  Used unchanged
        # for both the w buffer and every per-token x window.
        window_dims = [(self.CONV_WIDTH // self.DH, self.DH), (self.CONV_SIZE, self.CONV_WIDTH), (self.DH, 1)]
        window_len = self.CONV_SIZE * self.CONV_WIDTH

        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # ---- S2MM ch1: host weights, then the incoming conv cache --------
            # The three BDs cycle, so per invocation the host must send exactly
            # conv_w (CONV_SIZE * CONV_WIDTH), ssm_norm (linear_v_head_dim) and
            # then the cache ((CONV_SIZE - 1) * CONV_WIDTH) on this channel, in
            # that order.  conv_w and ssm_norm are adjacent in rope_rms_buffer,
            # so the first two are one shim transfer.
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[1], chain = block[3])

            with block[1]:
                lock_acquire(self.conv_w_free_lock, self.L)
                aie.dma_bd(self.conv_w_buffer, offset = 0, len = window_len)
                lock_release(self.conv_w_ready_lock, self.L)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.delta_norm_free_lock, self.L)
                aie.dma_bd(self.delta_norm_buffer)
                lock_release(self.delta_norm_ready_lock, self.L)
                aie.next_bd(block[2])

            # Acquires L: L - 1 counts come back from the sliding windows and
            # the last from NEW_CACHE_out, so the next invocation's cache write
            # cannot start until this one's window has been read back.  Without
            # that ordering the tail rows would be overwritten while the
            # writeback was still pending -- silently wrong, not a hang.
            #
            # It used to be L + 1, with every window BD releasing a count and
            # the writeback releasing the extra one.  That put *two* releases on
            # the last window BD -- conv_x_free and conv_cache_out_ready -- and
            # an AIE2 BD descriptor has a single lock-release field, so one of
            # the two is silently dropped when the BDs are lowered to register
            # writes.  aie-translate accepts the pair without a diagnostic and
            # it survives into input_with_addresses.mlir, so nothing upstream
            # flags it.  The writeback ran, which means conv_cache_out_ready is
            # the one that survived and conv_x_free came back one short: this
            # lock ended every pass at L rather than L + 1, block[2] could never
            # arm again, and the design was single-dispatch.  Pass one was
            # unaffected because block[2] had already armed off the init value,
            # so the whole layer scored correctly and only a second dispatch
            # hung -- at 60 s, in the host's DMA wait on the cache send.
            #
            # Handing the count over instead of releasing both keeps every BD in
            # this tile at one acquire and one release, which is what the
            # hardware can actually express.
            with block[2]:
                lock_acquire(self.conv_x_free_lock, self.L)
                aie.dma_bd(self.conv_x_buffer, offset = 0, len = (self.CONV_SIZE - 1) * self.CONV_WIDTH)
                lock_release(self.conv_cache_done_lock, 1)
                aie.next_bd(block[1])

            # ---- S2MM ch3: the L new tokens from the projection engine -------
            # The proj engine emits raw[(rb * L + t) * m + r]; token t channel
            # (rb * m + r) belongs at row (CONV_SIZE - 1 + t).  The de-permute
            # is done here in the BD, so the conv core sees natural
            # [token][channel] order and the host needs no unpermute on readback.
            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest = block[4], chain = block[5])

            with block[4]:
                lock_acquire(self.conv_cache_done_lock, 1)
                aie.dma_bd(self.conv_x_buffer,
                           offset = (self.CONV_SIZE - 1) * self.CONV_WIDTH,
                           len = self.L * self.CONV_WIDTH,
                           dimensions = [(self.CONV_WIDTH // self.m, self.m), (self.L, self.CONV_WIDTH), (self.m, 1)])
                # L, not L + 1: the writeback waits on cache_out_ready, which the
                # last window BD releases, so it is sequenced after these rather
                # than competing with them for a count.
                lock_release(self.conv_x_ready_lock, self.L)
                aie.next_bd(block[4])

            # ---- MM2S ch1: the 4 weight taps, replayed once per token --------
            with block[5]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[6], chain = block[7])

            with block[6]:
                lock_acquire(self.conv_w_ready_lock, 1)
                aie.dma_bd(self.conv_w_buffer, offset = 0, len = window_len, dimensions = window_dims)
                lock_release(self.conv_w_free_lock, 1)
                aie.next_bd(block[6])

            # ---- MM2S ch5: the sliding x window, one BD per token ------------
            # One BD for all L tokens would be L * CONV_SIZE * CONV_WIDTH * 2 =
            # 655 KB, past the memtile BD length limit (17-bit word count), so
            # the L windows are L chained BDs that differ only in their offset.
            with block[7]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5, dest = block[10], chain = block[9])

            # The last window BD arms the writeback *instead of* releasing its
            # own conv_x_free count -- one release per BD, because that is all a
            # BD descriptor holds.  Token L - 1's window is the last read that
            # touches the tail rows, so once it has been taken the outgoing
            # cache is final and ch4 is free to send it; ch4 then passes the
            # count on to block[2], which is the ordering block[2] wants anyway.
            for t in range(self.L):
                with block[10 + t]:
                    lock_acquire(self.conv_x_ready_lock, 1)
                    aie.dma_bd(self.conv_x_buffer, offset = t * self.CONV_WIDTH, len = window_len, dimensions = window_dims)
                    if t == self.L - 1:
                        lock_release(self.conv_cache_out_ready_lock, 1)
                    else:
                        lock_release(self.conv_x_free_lock, 1)
                    aie.next_bd(block[10 + (t + 1) % self.L])

            # ---- MM2S ch2: ssm_norm, replayed once per token -----------------
            # Column 5 to CT[3][2] is the one cross-column hop stage 7b adds.
            with block[9]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest = block[18], chain = block[19])

            with block[18]:
                lock_acquire(self.delta_norm_ready_lock, 1)
                aie.dma_bd(self.delta_norm_buffer)
                lock_release(self.delta_norm_free_lock, 1)
                aie.next_bd(block[18])

            # ---- MM2S ch4: every accept-length's conv window ------------------
            # Speculative decode does not know at dispatch time how many of the
            # L drafted tokens will be accepted, so one window is not enough: if
            # j of them survive, the next pass has to convolve against the last
            # CONV_SIZE - 1 pre-conv rows *as of token j*, not as of token L.
            #
            # Every one of those windows is already in this buffer.  It holds
            # (CONV_SIZE - 1) cache rows then the L new token rows, so the
            # accept-j window is rows j .. j + CONV_SIZE - 2 -- and as j runs
            # 1..L those are exactly rows 1 .. L + CONV_SIZE - 2.  So the whole
            # tail, everything but row 0, goes out as one contiguous run and the
            # host slices the window it needs once the verify result is known.
            # Row 0 is the only row no accept length can want: it falls out of
            # the window at j = 1 already.
            #
            # L + CONV_SIZE - 2 rows is 102400 words here, inside the memtile
            # BD's 17-bit length field -- a per-j BD chain would cost L BDs and
            # buy nothing, since the runs overlap and the host can index.
            #
            # Contiguous, hence no `dimensions`: block[4] already un-permuted the
            # engine's row-block order on the way in, so the rows are in natural
            # [token][channel] layout and window j starts at row j - 1 of it.
            with block[19]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest = block[20], chain = block[400])

            with block[20]:
                lock_acquire(self.conv_cache_out_ready_lock, 1)
                aie.dma_bd(self.conv_x_buffer,
                           offset = self.CONV_WIDTH,
                           len = (self.L + self.CONV_SIZE - 2) * self.CONV_WIDTH)
                lock_release(self.conv_x_free_lock, 1)
                aie.next_bd(block[20])

            with block[400]:
                aie.EndOp()

    # S2MM ch0/ch2 and MM2S ch0/ch3 carried the a/b weight relay and a hidden
    # state hop in the L=1 design. The a/b tile is fed straight from IT[5] now
    # and nothing here has a buffer or a BD for them, so the ports are gone
    # rather than left as names that route nothing.

    @property
    def W_in(self) -> dma_port:
        return self.S2MM[1]

    @property
    def QKV_in(self) -> dma_port:
        return self.S2MM[3]

    @property
    def CONV_W_out(self) -> dma_port:
        return self.MM2S[1]

    @property
    def DELTA_NORM_W_out(self) -> dma_port:
        return self.MM2S[2]


    @property
    def NEW_CACHE_out(self) -> dma_port:
        """Every accept-length's conv window: pre-conv rows 1 .. L+CONV_SIZE-2,
        contiguous.  The accept-j window is the CONV_SIZE - 1 rows starting at
        row j - 1 of this run, in the same layout W_in takes the cache in."""
        return self.MM2S[4]

    @property
    def CONV_X_out(self) -> dma_port:
        return self.MM2S[5]