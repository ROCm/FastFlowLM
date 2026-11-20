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

from utils import dma_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_MT_IP, comment_print

from constants import L_TOKENS, aux_to_rms_pkt_id


class qkv_stage_memtile(AIE_MT_IP):
    """Token-major staging between the projection engine and the RoPE tile.

    The engine's y is row-block permuted and the RoPE tile wants whole tokens,
    so something has to transpose. This is the tile that does it, for the same
    reason aux_memtile stages z: the permutation only resolves once *every* row
    block has landed -- token 0's row is not complete until the last block
    arrives -- so the whole L x (DQ + DK + DV) activation has to be held
    somewhere, and at 128 KiB that is past a compute tile and comfortable on a
    memtile.

    The engine emits

        raw[(rb * L_TOKENS + t) * m + r]        m = 32, rb = 0 .. QKV/m - 1

    in rb order, and the fused matrix is laid out q | k | v, so row blocks
    0 .. DQ/m are q and the rest are k | v. The RoPE tile, though, takes **kv
    first and q second** -- its S2MM chain alternates kv_in_buffer then
    q_in_buffer, and the kernel rotates k before it rotates q, because k and v
    have to be on their way to the cache while q is still being normed.

    So the staging layout is rotated rather than natural:

        stage[t] = | k | v  (DK + DV) | q (DQ) |

    which is what makes the outbound side one linear BD per token instead of
    two. The rotation costs nothing: it is expressed as *where the two inbound
    BDs start*, not as extra work.

      inbound   BD 1  the q row blocks   -> offset kv_dim, 3-dim de-permute
                BD 2  the k | v blocks   -> offset 0,      same shape
      outbound  L BDs one whole rotated token each, to the RoPE tile

    Two inbound BDs and not one because the destination offset differs; they
    chain, and the split lands exactly on a packet boundary, since the engine
    emits one packet per round of 16 row blocks and both q and k | v are a whole
    number of rounds. The stream is consumed in arrival order either way, so the
    boundary is bookkeeping, not synchronisation.

    Locks. The buffer fills completely before any of it is readable, so the
    hand-over is one credit per *token* and the two inbound BDs are staged
    against each other:

        q  BD   acq free(L)   rel mid(1)
        kv BD   acq mid(1)    rel ready(L)
        out BD  acq ready(1)  rel free(1)      x L, cyclic

    Every count is <= L_TOKENS, well inside the memtile's 6-bit lock counter,
    and the cycle returns the buffer to free = L after the last token leaves.

    **This tile also parks z**, the output gate, in a second buffer.

    z is the projection engine's second mvm, so it is produced immediately after
    qkv -- and its consumer, the gate, cannot run until the attention output
    exists, which is q/k norm, RoPE, the cache append and the whole softmax
    column later. Something has to hold L_TOKENS x DQ = 96 KiB across that
    latency, which is past a compute tile and comfortable here. It arrives with
    the same row-block permutation as qkv and is de-permuted on the way in by
    the same 3-dimensional BD shape.

    It leaves as **two half-rows, one per attention column**, because there are
    two gate tiles: column 4 computes q heads 0..11 and column 5 heads 12..23,
    and `attn * sigmoid(z)` is elementwise, so each gate tile only needs the half
    matching its own column. The split is free -- DQ/2 is a whole number of the
    engine's 32-row blocks, so it is only where the outbound BD starts and how
    long it is. See iron/gate.py for why the gate is split rather than the
    attention merged.

    **And it stages o_proj's y**, the third thing off the same engine with the
    same row-block permute -- D wide rather than DQ -- on its way to the rms
    tile. That un-permute is a memtile's job for the same reason as the other
    two, and the tile is directly south of the rms tile, so it is one hop.

    Channel map:
        S2MM ch0  qkv from the projection engine, row-block permuted
        MM2S ch0  one rotated token per BD, to the RoPE tile
        S2MM ch1  z from the projection engine, row-block permuted
        MM2S ch1  z's low half,  one token per BD, to gate tile 0
        MM2S ch2  z's high half, one token per BD, to gate tile 1
        S2MM ch2  o_proj's y, row-block permuted
        MM2S ch5  o token-major, one row per BD, north to the rms tile
        MM2S ch3  o token-major, the whole buffer in one BD, to DDR
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing qkv_stage_memtile at ({tile.row}, {tile.col})")

        self.DQ = model_config.DQ
        self.D = model_config.D
        self.kv_dim = model_config.DK + model_config.DV
        self.qkv_dim = self.DQ + self.kv_dim
        self.L = L_TOKENS
        self.m = model_config.q4nx_row_block_size          # 32
        self.stage_size = self.L * self.qkv_dim

        self.q_blocks = self.DQ // self.m
        self.kv_blocks = self.kv_dim // self.m

        # The BD size field is 10 bits, but a bf16 BD is lowered to 32-bit words
        # with the sizes doubled, so the verifier's real ceiling is 512.
        for n, what in ((self.q_blocks, "q row blocks"),
                        (self.kv_blocks, "k|v row blocks"),
                        (self.L, "tokens"), (self.m, "rows per block")):
            assert n <= 512, f"{what} = {n} does not fit one memtile BD dimension"

        comment_print(f"  stage buffer: {self.stage_size} bf16 "
                      f"({self.stage_size * 2 // 1024} KiB), layout | kv {self.kv_dim} | q {self.DQ} |")
        comment_print(f"  row blocks: q {self.q_blocks}, kv {self.kv_blocks}")

        self.stage_ty = np.ndarray[(self.stage_size,), self.dtype]
        self.qkv_buffer = aie.buffer(self.mt, self.stage_ty,
                                     name = f"qkv_stage_buffer_{self.mt_row}_{self.mt_col}")

        self.free_lock  = aie.lock(self.mt, lock_id = 0, init = self.L)
        self.mid_lock   = aie.lock(self.mt, lock_id = 1, init = 0)
        self.ready_lock = aie.lock(self.mt, lock_id = 2, init = 0)

        # ---- z, the output gate, parked for the whole attention latency ------
        # Same permutation as qkv and the same de-permute, but its own buffer
        # and its own lock pair: it is filled while qkv is still draining to the
        # RoPE tile and is not read until long after qkv is gone, so sharing
        # either would serialise two stages that have no dependency.
        self.z_blocks = self.DQ // self.m
        self.z_size = self.L * self.DQ
        # Halved between the two gate tiles, one per attention column.
        self.z_half = self.DQ // 2
        assert self.z_half % self.m == 0, \
            f"the gate halves must land on row-block boundaries: {self.z_half} % {self.m} != 0"
        assert self.z_blocks <= 512, "a memtile BD dimension is 10 bits, halved for bf16"

        comment_print(f"  z buffer: {self.z_size} bf16 ({self.z_size * 2 // 1024} KiB), "
                      f"{self.z_blocks} row blocks, split 2 x {self.z_half} to the gate tiles")
        comment_print(f"  total staged: {(self.stage_size + self.z_size) * 2 // 1024} KiB")

        self.z_ty = np.ndarray[(self.z_size,), self.dtype]
        self.z_buffer = aie.buffer(self.mt, self.z_ty,
                                   name = f"z_stage_buffer_{self.mt_row}_{self.mt_col}")

        # One producer, two consumers -- the two halves of each token leave on
        # different channels, so a token's credit is only returned once both
        # gate tiles have taken their half: low -> high -> free.
        self.z_free_lock = aie.lock(self.mt, lock_id = 3, init = self.L)
        self.z_low_lock  = aie.lock(self.mt, lock_id = 4, init = 0)
        self.z_high_lock = aie.lock(self.mt, lock_id = 5, init = 0)

        # ---- o_proj's y, on its way back to the rms tile ---------------------
        # The third thing this tile de-permutes, and the same 3-dimensional BD
        # as the other two -- only D wide rather than DQ, because o_proj's
        # output is the model dimension. It comes off the same engine with the
        # same row-block permute, and the rms tile wants whole D-wide tokens,
        # which a compute tile's BD cannot produce: it walks one buffer and the
        # rms tile has no bank to spare for a second.
        #
        # It replaces a DDR round trip -- o out one shim, straight back in on
        # another, un-permuted by the shim BD -- which is 160 KB per layer and a
        # shim channel.
        #
        # **The buffer serves down_proj as well as o_proj**, under the same id,
        # route, BD and locks: both are D-wide y off the same engine headed for
        # the same residual, and they cannot collide because o's eight rows have
        # drained before the post-attention norm exists, which is itself four
        # dependency steps upstream of down. The rms tile tells them apart by
        # arrival order, exactly as it tells its phases apart. So the MLP costs
        # this tile nothing.
        self.o_size = self.L * self.D
        self.o_blocks = self.D // self.m
        assert self.D % self.m == 0, "the o de-permute walks whole row blocks"
        assert self.o_blocks <= 512, "a memtile BD dimension is 10 bits, halved for bf16"

        comment_print(f"  o buffer: {self.o_size} bf16 ({self.o_size * 2 // 1024} KiB), "
                      f"{self.o_blocks} row blocks")
        comment_print(f"  total staged: "
                      f"{(self.stage_size + self.z_size + self.o_size) * 2 // 1024} KiB")

        self.o_ty = np.ndarray[(self.o_size,), self.dtype]
        self.o_buffer = aie.buffer(self.mt, self.o_ty,
                                   name = f"o_stage_buffer_{self.mt_row}_{self.mt_col}")

        # One producer, two consumers: the rms tile takes each row north for the
        # residual, and the DDR tap copies the buffer for scoring. The chain is
        # ready -> ddr -> free, so the rms tile goes first -- it is the critical
        # path -- and the slot only returns once both have read it.
        self.o_free_lock  = aie.lock(self.mt, lock_id = 6, init = self.L)
        self.o_ready_lock = aie.lock(self.mt, lock_id = 7, init = 0)

        self.__call__()

    def _dims(self, blocks: int):
        """[rb][t][r] -> [t][rb * m + r], for a run of `blocks` row blocks.
        Stream order is rb, t, r -- the outer dimension is the one the engine
        walks slowest."""
        return [(blocks, self.m), (self.L, self.qkv_dim), (self.m, 1)]

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # ---- S2MM ch0: qkv in, de-permuted and rotated on the way --------
            with block[0]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0,
                              dest = block[1], chain = block[3])

            # q arrives first -- it is the head of the fused matrix -- but lands
            # second, after the k | v half.
            with block[1]:
                lock_acquire(self.free_lock, self.L)
                aie.dma_bd(self.qkv_buffer, offset = self.kv_dim,
                           len = self.L * self.DQ, dimensions = self._dims(self.q_blocks))
                lock_release(self.mid_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.mid_lock, 1)
                aie.dma_bd(self.qkv_buffer, offset = 0,
                           len = self.L * self.kv_dim, dimensions = self._dims(self.kv_blocks))
                lock_release(self.ready_lock, self.L)
                aie.next_bd(block[1])

            # ---- MM2S ch0: one whole rotated token per BD, to the RoPE tile ---
            # One self-chained BD cannot walk the offset, so this is L chained
            # BDs that differ only in where they start. The RoPE tile's own S2MM
            # chain splits each one back into kv_in_buffer then q_in_buffer; a
            # DMA stream does not have to agree with the far side about BD
            # boundaries, only about byte order.
            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0,
                              dest = block[10], chain = block[10 + self.L])

            for t in range(self.L):
                with block[10 + t]:
                    lock_acquire(self.ready_lock, 1)
                    aie.dma_bd(self.qkv_buffer, offset = t * self.qkv_dim, len = self.qkv_dim)
                    lock_release(self.free_lock, 1)
                    aie.next_bd(block[10 + (t + 1) % self.L])

            # ---- S2MM ch1: z in, de-permuted the same way ---------------------
            # Its own channel and its own buffer: z lands while qkv is still
            # draining to the RoPE tile, so it cannot share either without
            # serialising two stages that do not depend on each other.
            with block[10 + self.L]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1,
                              dest = block[50], chain = block[52])

            with block[50]:
                lock_acquire(self.z_free_lock, self.L)
                aie.dma_bd(self.z_buffer, offset = 0, len = self.z_size,
                           dimensions = self._z_dims())
                lock_release(self.z_low_lock, self.L)
                aie.next_bd(block[50])

            # ---- MM2S ch1/ch2: z's two halves, one per gate tile -------------
            # Token-major by now, so a half is one contiguous run: BD t of the
            # low channel starts at t * DQ and the high channel at
            # t * DQ + z_half. `len` here is a buffer length, not a 10-bit
            # dimension size, so 3072 needs no splitting.
            #
            # The credit chain is low -> high -> free: both tiles take their
            # half of a token before the slot is returned. They still pipeline,
            # because the inbound BD releases all L low credits at once.
            with block[52]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1,
                              dest = block[60], chain = block[60 + self.L])

            for t in range(self.L):
                with block[60 + t]:
                    lock_acquire(self.z_low_lock, 1)
                    aie.dma_bd(self.z_buffer, offset = t * self.DQ, len = self.z_half)
                    lock_release(self.z_high_lock, 1)
                    aie.next_bd(block[60 + (t + 1) % self.L])

            with block[60 + self.L]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2,
                              dest = block[80], chain = block[80 + self.L])

            for t in range(self.L):
                with block[80 + t]:
                    lock_acquire(self.z_high_lock, 1)
                    aie.dma_bd(self.z_buffer, offset = t * self.DQ + self.z_half,
                               len = self.z_half)
                    lock_release(self.z_free_lock, 1)
                    aie.next_bd(block[80 + (t + 1) % self.L])

            # ---- S2MM ch2: o_proj's y, de-permuted the same way --------------
            # Whole-buffer in one BD, like z: the row-block permute means no
            # token is complete until the last block lands, so there is nothing
            # to hand over early.
            with block[80 + self.L]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2,
                              dest = block[100], chain = block[101])

            with block[100]:
                lock_acquire(self.o_free_lock, self.L)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.o_size,
                           dimensions = self._o_dims())
                lock_release(self.o_ready_lock, self.L)
                aie.next_bd(block[100])

            with block[101]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 5,
                              dest = block[110], chain = block[110 + self.L])

            for t in range(self.L):
                with block[110 + t]:
                    lock_acquire(self.o_ready_lock, 1)
                    aie.dma_bd_packet(0, aux_to_rms_pkt_id.to_residual)
                    aie.dma_bd(self.o_buffer, offset = t * self.D, len = self.D)
                    lock_release(self.o_free_lock, 1)
                    aie.next_bd(block[110 + (t + 1) % self.L])

            with block[110 + self.L]:
                aie.EndOp()

    def _z_dims(self):
        """[rb][t][r] -> [t][rb * m + r] over the whole DQ-wide z row."""
        return [(self.z_blocks, self.m), (self.L, self.DQ), (self.m, 1)]

    def _o_dims(self):
        """The same permute, D wide: o_proj's y off the same engine."""
        return [(self.o_blocks, self.m), (self.L, self.D), (self.m, 1)]

    @property
    def QKV_in(self) -> dma_port:
        return self.S2MM[0]

    @property
    def QKV_out(self) -> dma_port:
        return self.MM2S[0]

    @property
    def Z_in(self) -> dma_port:
        """z from the projection engine, row-block permuted."""
        return self.S2MM[1]

    @property
    def O_in(self) -> dma_port:
        """o_proj's y from the projection engine, row-block permuted."""
        return self.S2MM[2]

    @property
    def O_out(self) -> dma_port:
        """o token-major, the whole L x D buffer in one BD, to DDR."""
        return self.MM2S[3]

    @property
    def Residual_out(self) -> dma_port:
        """One o row per BD, north into the rms tile's x port, for the residual.

        On an odd channel because MT[3]'s even BD ids are nearly spent; see the
        chain's comment.
        """
        return self.MM2S[5]

    def Z_out(self, half: int) -> dma_port:
        """z's `half` (0 = q heads 0..11, 1 = heads 12..23), to that half's gate
        tile. The index is the attention column index, and the two have to agree
        -- gate tile `half` is fed by attention column `half`."""
        assert half in (0, 1)
        return self.MM2S[1 + half]
