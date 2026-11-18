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
from constants import L_TOKENS

class RoPE(AIE_CT_IP):
    """q/k RMSNorm and partial rotary, L_TOKENS tokens per invocation.

    Qwen3.8 norms each 256-dim head before rotating -- the reverse of HunYuan --
    and rotates only the first ROPE_DIM = 64 dims, rotate_half over the halves
    0..31 and 32..63. v passes through untouched; it is here only because it
    shares the fused row.

    **The tile stays one token wide and loops.** All that is resident for the
    whole pass is the cos/sin table, because each token sits at a different
    position and the rotation is the one thing that cannot be shared between
    them. Everything else is a single row: the staging memtile upstream
    (qkv_stage_memtile) holds the L x (DQ + DK + DV) activation and feeds this
    tile one rotated token at a time, so the six buffers here total 38 KiB
    against the tile's 64 KiB whatever L_TOKENS is.

    Input order is **kv then q**, which is why the staging tile rotates each
    token to | k | v | q |. k and v have to be on their way to the cache while q
    is still being normed; taking q first would hold the cache write behind six
    q heads of work per token.

    The cos/sin table is L_TOKENS x ROPE_DIM -- half cos, half sin per token --
    followed by the two norm weights:

        rope_w[t * ROPE_DIM              .. + ROPE_DIM/2)   cos for token t
        rope_w[t * ROPE_DIM + ROPE_DIM/2 .. + ROPE_DIM/2)   sin for token t
        rope_w[L_TOKENS * ROPE_DIM       .. + DH)           q_norm.weight
        rope_w[L_TOKENS * ROPE_DIM + DH  .. + DH)           k_norm.weight

    The L = 1 design put the norm weights at a fixed +ROPE_DIM; they move with
    the table, and kernels/rope.cc reads the same two expressions. It arrives
    once per invocation on a self-chained BD, acquired once outside the token
    loop -- the same shape as the norm weight on the rms tile.

    Q leaves as one packet per attention CU -- each CU owns one kv head and
    takes that head's padded GQA group. See Q_out.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "rope.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing RoPE at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.DQ = model_config.DQ
        self.DK = model_config.DK
        self.DV = model_config.DV
        self.num_attn_cus = model_config.num_attn_cus
        q_heads_per_group = model_config.num_heads // model_config.num_kv_heads
        self.DQ_PADDED = (q_heads_per_group + model_config.q_heads_padding - 1) // model_config.q_heads_padding * model_config.q_heads_padding * model_config.num_kv_heads * model_config.head_dim
        self.q_in_ty = np.ndarray[(self.DQ,), self.dtype]
        self.kv_in_ty = np.ndarray[(self.DK + self.DV,), self.dtype]
        comment_print(f"DQ: {self.DQ}, DK: {self.DK}, DV: {self.DV}, DQ_PADDED: {self.DQ_PADDED}")
        self.q_ty = np.ndarray[(self.DQ_PADDED,), self.dtype]
        self.k_ty = np.ndarray[(self.DK,), self.dtype]
        self.v_ty = np.ndarray[(self.DV,), self.dtype]

        # L_TOKENS cos|sin pairs, then the two per-head norm weights. The L = 1
        # design sized this ROPE_DIM and put the weights straight after; both the
        # host and kernels/rope.cc index off L_TOKENS * ROPE_DIM now.
        rope_size = L_TOKENS * model_config.ROPE_DIM

        if model_config.has_qk_norm:
            rope_size += model_config.DH * 2
        if model_config.has_attn_bias:
            rope_size += (model_config.DQ + model_config.DK + model_config.DV)

        self.rope_ty = np.ndarray[(rope_size,), self.dtype]

        comment_print(f"\tRoPE DQ: {self.DQ}, DK: {self.DK}, DV: {self.DV}, DQ_PADDED: {self.DQ_PADDED}, rope_size: {rope_size}")

        self.q_in_buffer = ctx.buffer(tile = self.located_tile, dtype = self.q_in_ty, name = f"q_in_buffer_{self.ct_row}_{self.ct_col}")
        self.kv_in_buffer = ctx.buffer(tile = self.located_tile, dtype = self.kv_in_ty, name = f"kv_in_buffer_{self.ct_row}_{self.ct_col}")
        self.q_buffer = ctx.buffer(tile = self.located_tile, dtype = self.q_ty, name = f"q_buffer_{self.ct_row}_{self.ct_col}")
        self.k_buffer = ctx.buffer(tile = self.located_tile, dtype = self.k_ty, name = f"k_buffer_{self.ct_row}_{self.ct_col}")
        self.v_buffer = ctx.buffer(tile = self.located_tile, dtype = self.v_ty, name = f"v_buffer_{self.ct_row}_{self.ct_col}")
        self.rope_buffer = ctx.buffer(tile = self.located_tile, dtype = self.rope_ty,  name = f"rope_buffer_{self.ct_row}_{self.ct_col}")
        self.qkv_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.qkv_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.q_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.q_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.k_prod_lock = aie.lock(self.ct, lock_id = 4, init = 1)
        self.k_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.v_prod_lock = aie.lock(self.ct, lock_id = 6, init = 1)
        self.v_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.rope_prod_lock = aie.lock(self.ct, lock_id = 8, init = 1)
        self.rope_cons_lock = aie.lock(self.ct, lock_id = 9, init = 0)
        # Stage the multi-packet sends so their legs leave in column order and
        # the buffer is only released after the last one.
        self.q_pass_lock = aie.lock(self.ct, lock_id = 10, init = 0)
        self.k_pass_lock = aie.lock(self.ct, lock_id = 11, init = 0)
        self.v_pass_lock = aie.lock(self.ct, lock_id = 12, init = 0)

        # q leaves in one packet per attention CU. A CU owns one kv head, so its
        # slice is that head's padded GQA group, contiguous because the padding
        # sits inside the group.
        self.num_attn_cus_total = model_config.num_kv_heads // model_config.kv_heads_per_cu
        self.q_slice = model_config.q_heads_per_group_padded * model_config.DH
        assert self.q_slice * self.num_attn_cus_total == self.DQ_PADDED, \
            f"q slices {self.q_slice} x {self.num_attn_cus_total} do not cover DQ_PADDED {self.DQ_PADDED}"
        comment_print(f"\tq split: {self.num_attn_cus_total} CUs x {self.q_slice} = {self.DQ_PADDED}")

        # k and v are split the same way, but by *column* rather than by CU:
        # each column's shim appends the heads that column owns, so the append
        # and the replay share an IT and npu_dma_wait can order them.
        self.num_attn_columns = model_config.num_attn_columns
        self.kv_col_width = model_config.kv_heads_per_column * model_config.DH
        assert self.kv_col_width * self.num_attn_columns == self.DK
        # **k and v get their own id per column, not a shared one.** Four ids,
        # not two. They are the same wire out of this tile and the same shim,
        # but a distinct id lets the switchbox drop k on that shim's S2MM ch0
        # and v on ch1 -- two landing channels, so the host writes each plane
        # with its own BD instead of one BD that crosses between them.
        #
        # That crossing was the cache's context ceiling. A single BD had to hop
        # `plane_rows * DK` from a token's k to its v, which is a 20-bit shim
        # stride field (halved for bf16), so the cache topped out around 2048
        # positions and overflowed by *masking* rather than faulting. Split
        # across two channels, neither BD has an MAX_L-dependent stride at all:
        # k advances by DK, v by DV, and the plane base is a byte offset the
        # host patches in at full width. See decoding_layer::_receive_kv_cache.
        #
        # Disjoint from the q ids: the flows share switchboxes east of this
        # tile and a switchbox routes on the id alone.
        self.k_pkt_base = self.num_attn_cus_total
        self.v_pkt_base = self.k_pkt_base + self.num_attn_columns
        comment_print(f"\tk|v split: {self.num_attn_columns} columns x {self.kv_col_width} = {self.DK}, "
                      f"k pkt ids {self.k_pkt_base}..{self.k_pkt_base + self.num_attn_columns - 1}, "
                      f"v pkt ids {self.v_pkt_base}..{self.v_pkt_base + self.num_attn_columns - 1}")

        self.rope = ctx.func(link_with[:-2], [self.q_ty, self.k_ty, self.v_ty, self.q_in_ty, self.kv_in_ty, self.rope_ty]) # sometimes rope_phi4.o

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.rope(self.q_buffer, self.k_buffer, self.v_buffer, self.q_in_buffer, self.kv_in_buffer, self.rope_buffer)

        @aie.mem(self.ct)
        def mem_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[2])

            # kv first, then q: one staged token, split back out of the single
            # contiguous BD the staging memtile sends. Both halves fire
            # L_TOKENS times per invocation.
            with block[1]:
                lock_acquire(self.qkv_prod_lock, 1)
                aie.dma_bd(self.kv_in_buffer)
                lock_release(self.qkv_cons_lock, 1)
                aie.next_bd(block[100])

            with block[100]:
                lock_acquire(self.qkv_prod_lock, 1)
                aie.dma_bd(self.q_in_buffer)
                lock_release(self.qkv_cons_lock, 1)
                aie.next_bd(block[1])

            with block[2]:
                s1 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[20], chain = block[4])

            # One packet per attention CU, per token. Each CU owns one kv head,
            # so it wants exactly that head's GQA group -- Q_HEADS_PER_GROUP
            # padded to 8, i.e. q_slice = 2048 contiguous elements, because
            # pseduo_rope_q() lays the padding *inside* each group. So the split
            # is four offsets and no striding.
            #
            # The packet id selects the CU. This is a genuine one-to-many: one
            # master port on this tile feeds four destinations across two
            # columns, which a circuit flow cannot express.
            #
            # q no longer goes to DDR. It was read back while the CUs did not
            # exist, to check the layout the CUs would index against; they exist
            # now and consume that layout directly, so the readback is gone and
            # what is scored is the attention output instead.
            #
            # The chain is staged on q_pass_lock so the four packets leave in CU
            # order and the buffer is only released after the last one.
            for cu in range(self.num_attn_cus_total):
                first = (cu == 0)
                last = (cu == self.num_attn_cus_total - 1)
                with block[20 + cu]:
                    lock_acquire(self.q_cons_lock if first else self.q_pass_lock, 1)
                    aie.dma_bd_packet(0, cu)
                    aie.dma_bd(self.q_buffer, offset = cu * self.q_slice, len = self.q_slice)
                    lock_release(self.q_prod_lock if last else self.q_pass_lock, 1)
                    aie.next_bd(block[20 + (cu + 1) % self.num_attn_cus_total])

            with block[4]:
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[5], chain = block[9])

            # k then v, alternating, one pair per token -- but each split by kv
            # head across the two attention columns' shims, and now by *plane*
            # as well: k carries k_pkt_base + col and v carries v_pkt_base +
            # col, so the two land on different S2MM channels of the same shim.
            #
            # **Two reasons for the split, and they are different splits.**
            #
            # By column (which shim): a synchronisation fix. npu_dma_wait only
            # orders transfers on the *same* shim tile, so an append landing on
            # IT3 cannot be waited on before the replay reads from IT4 and IT5
            # -- the replay would start immediately and the CUs would attend to
            # a cache that had not been written yet, which is exactly the
            # all-zero output this produced. Sending each column's half to that
            # column's own shim puts the append and the replay on one IT, where
            # the wait means something.
            #
            # By plane (which channel): a layout fix, and what lifts the context
            # ceiling. The stream order here is k(t) v(t) k(t+1) v(t+1)..., so a
            # single landing channel forced the host to describe both planes in
            # ONE descriptor -- k on one axis, "k or v" on another whose stride
            # is the whole k plane. That stride is a 20-bit field, halved for
            # bf16, which capped the cache near 2048 positions and, worse,
            # overflowed silently: npu_instr_utils prints and masks. On two
            # channels each plane gets its own BD, whose only stride is a row
            # (DK or DV) and whose plane base is a full-width byte offset, so
            # nothing in the append scales with MAX_L any more.
            #
            # The halves are contiguous: the cache row is head-major, so column
            # c owns [c * kv_col_width, + kv_col_width).
            # The ids continue the q range rather than restarting at 0. All
            # these flows leave this tile heading east and share switchboxes on
            # the way to columns 4 and 5, and a switchbox routes on the id alone
            # -- it does not know which master port a packet came from. Reusing
            # 0 and 1 here made id 0 ambiguous at column 4 between a qk core and
            # the shim, and the q that never arrived stalled every CU.
            with block[5]:
                lock_acquire(self.k_cons_lock, 1)
                aie.dma_bd_packet(0, self.k_pkt_base + 0)
                aie.dma_bd(self.k_buffer, offset = 0, len = self.kv_col_width)
                lock_release(self.k_pass_lock, 1)
                aie.next_bd(block[6])

            with block[6]:
                lock_acquire(self.k_pass_lock, 1)
                aie.dma_bd_packet(0, self.k_pkt_base + 1)
                aie.dma_bd(self.k_buffer, offset = self.kv_col_width, len = self.kv_col_width)
                lock_release(self.k_prod_lock, 1)
                aie.next_bd(block[7])

            with block[7]:
                lock_acquire(self.v_cons_lock, 1)
                aie.dma_bd_packet(0, self.v_pkt_base + 0)
                aie.dma_bd(self.v_buffer, offset = 0, len = self.kv_col_width)
                lock_release(self.v_pass_lock, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.v_pass_lock, 1)
                aie.dma_bd_packet(0, self.v_pkt_base + 1)
                aie.dma_bd(self.v_buffer, offset = self.kv_col_width, len = self.kv_col_width)
                lock_release(self.v_prod_lock, 1)
                aie.next_bd(block[5])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[10], chain = block[11])

            # The cos/sin table and the two norm weights, once per invocation --
            # the kernel acquires this outside its token loop, so one firing
            # covers all L_TOKENS rows.
            with block[10]:
                lock_acquire(self.rope_prod_lock, 1)
                aie.dma_bd(self.rope_buffer)
                lock_release(self.rope_cons_lock, 1)
                aie.next_bd(block[10])

            with block[11]:
                aie.EndOp()

    @property
    def QKV_in(self):
        return self.S2MM[0]

    @property
    def ROPE_in(self):
        return self.S2MM[1]

    @property
    def Q_out(self):
        return self.MM2S[0]

    @property
    def KV_out(self):
        return self.MM2S[1]
