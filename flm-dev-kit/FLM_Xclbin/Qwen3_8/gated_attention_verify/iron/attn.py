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
from aie.dialects import memref

from utils import connect_port, mlir_context, tile_with_location, lock_acquire, lock_release
from utils import AIE_CT_IP, AIE_MT_IP, comment_print, ShimTile

from constants import L_TOKENS

# The context is streamed to the CUs in chunks of this many keys, and both
# attention kernels count their inner loop down by it. It is not a free
# parameter: attn_qk builds its causal mask from an aie::vector<int, 16>, so 16
# is baked into the kernel and this constant only names it.
L_CHUNK = 16


class qk_tile(AIE_CT_IP):
    """q . k^T for one kv head, one query token at a time, chunk by chunk.

    Holds q for the token it is working on and double-buffers k, so the next
    chunk of the context lands while the current one is being scored. The
    running max and the softmax correction live in the kernel's statics; the
    scores themselves go straight into the *kv tile's* memory (see attn_cu), so
    this tile never sends them anywhere -- the neighbour reads them in place.

    Key counts, from one RTP word. Token t attends to ctx + t + 1 keys, which
    the kernel turns into its causal mask; the host sends only the last token's
    count, L_max = ctx + L_TOKENS, and the kernel's own token counter recovers
    the rest as L_max - (L_TOKENS - 1) + t. The *chunk* count is one per pass,
    ceil(L_max / L_CHUNK), because every token is sent the whole context and so
    every token has to consume it. See attn_memtile.

    **q is buffered L_TOKENS deep, and that is a deadlock fix, not a
    throughput one.** The new k and v have to reach DDR and be appended to the
    cache before the host can replay the context back in, but the RoPE tile
    emits k | v and q interleaved per token and q comes straight here. Double
    buffered, the RoPE tile would stall on the third token's q -- which this
    core cannot accept until it has k, which the host cannot send until the
    append it is still waiting on completes. Holding every token's q lets the
    RoPE tile drain in one go and breaks the cycle.

    Eight separate 4 KiB buffers rather than one 32 KiB array: AIEAssignBuffers
    places one allocation per bank and a bank is 16 KiB, so a single buffer that
    size would not place. Same reason rms_residual.py splits h into one row per
    bank.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, shared_s_buffers: List[Any], s_ty: Any, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "attn_qk.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing attn_qk_tile at ({self.ct_row}, {self.ct_col})")
        # One kv head per CU, so a CU's q is exactly one GQA group -- 6 real
        # heads padded to 8. The padding is what the 1x8x1 mmul needs and it is
        # already what leaves the RoPE tile.
        self.Q_HEADS_PER_CU_PADDED = model_config.q_heads_per_group_padded
        self.KV_HEADS_PER_CU = model_config.kv_heads_per_cu
        self.DH = model_config.DH
        self.lc = L_CHUNK

        self.attn_q_ty = np.ndarray[(self.Q_HEADS_PER_CU_PADDED * self.DH,), self.dtype]
        self.attn_k_ty = np.ndarray[(self.lc, self.KV_HEADS_PER_CU * self.DH), self.dtype]
        comment_print(f"\tattn_q_ty shape: ({self.Q_HEADS_PER_CU_PADDED * self.DH},), attn_k_ty shape: ({self.lc}, {self.KV_HEADS_PER_CU * self.DH})")
        self.attn_s_ty = s_ty
        self.L_ty = ctx.data_tys["L_ty"]

        # The kernel takes the q buffers as L_TOKENS separate arguments and
        # indexes them by the token counter; k stays double buffered.
        self.attn_qk = ctx.func("attn_qk",
            [self.attn_q_ty] * L_TOKENS + [self.attn_k_ty, self.attn_k_ty,
             self.attn_s_ty, self.attn_s_ty, self.L_ty])

        self.k_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_k_ty, name = f"k_0_{self.ct_row}_{self.ct_col}")
        self.k_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_k_ty, name = f"k_1_{self.ct_row}_{self.ct_col}")

        q_bytes = self.Q_HEADS_PER_CU_PADDED * self.DH * 2
        assert q_bytes <= 16 * 1024, f"a q buffer is {q_bytes} B, past one 16 KiB bank"
        self.q_buffers = [
            ctx.buffer(tile = self.located_tile, dtype = self.attn_q_ty,
                       name = f"q_{t}_{self.ct_row}_{self.ct_col}")
            for t in range(L_TOKENS)
        ]
        comment_print(f"\tq buffered {L_TOKENS} deep ({q_bytes} B each, "
                      f"{L_TOKENS * q_bytes // 1024} KiB) so the RoPE tile can drain "
                      f"before the KV replay starts")

        # **q is transposed on the way in.** It arrives head-major -- the RoPE
        # tile emits head 0's 256 dims, then head 1's -- but _attn_qk consumes it
        # as the A operand of an 8x8x8 mmul, which wants each 8-dim block of all
        # 8 heads contiguous: [dim_block][head][8 dims]. A linear BD leaves q in
        # the wrong shape and every score is garbage, in a way that looks like
        # noise rather than a permutation.
        #
        #   buffer[j * 64 + i * 8 + k] = q[head i][dim j * 8 + k]
        self.q_heads_len = self.Q_HEADS_PER_CU_PADDED * self.DH
        self.q_in_order = [(self.Q_HEADS_PER_CU_PADDED, 8),
                           (self.DH // 8, self.Q_HEADS_PER_CU_PADDED * 8),
                           (8, 1)]
        comment_print(f"\tq_in_order: {self.q_in_order}, len {self.q_heads_len}")

        self.q_prod_lock = aie.lock(self.ct, lock_id = 0, init = L_TOKENS)
        self.q_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.k_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.k_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.s_prod_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.s_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.l_prod_lock = aie.lock(self.ct, lock_id = 6, init = 2)
        self.l_cons_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.s_buffer_0 = shared_s_buffers[0]
        self.s_buffer_1 = shared_s_buffers[1]

        # The pass's largest key count, one int. Written by the host before the
        # run; see decoding_layer::_write_attn_L.
        self.L_buffer = ctx.buffer(f"RTP_L_attn_qk_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.L_ty, use_write_rtp = True)

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.attn_qk(*self.q_buffers, self.k_buffer_0, self.k_buffer_1, self.s_buffer_0, self.s_buffer_1, self.L_buffer)

        @aie.mem(self.ct)
        def mem_body(block):
            # ---- S2MM ch0: q, one token's GQA group per firing ---------------
            # L_TOKENS BDs in a cycle, one per token, so the whole pass lands
            # without this core having to consume any of it. Linear: the RoPE
            # tile already emits each CU's q heads contiguously and padded.
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[10], chain=block[3])

            for t in range(L_TOKENS):
                with block[10 + t]:
                    lock_acquire(self.q_prod_lock, 1)
                    aie.dma_bd(self.q_buffers[t], offset = 0,
                               len = self.q_heads_len, dimensions = self.q_in_order)
                    lock_release(self.q_cons_lock, 1)
                    aie.next_bd(block[10 + (t + 1) % L_TOKENS])

            # ---- S2MM ch1: k, one L_CHUNK of context per firing --------------
            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[4], chain=block[6])

            with block[4]:
                lock_acquire(self.k_prod_lock, 1)
                aie.dma_bd(self.k_buffer_0)
                lock_release(self.k_cons_lock, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.k_prod_lock, 1)
                aie.dma_bd(self.k_buffer_1)
                lock_release(self.k_cons_lock, 1)
                aie.next_bd(block[4])

            with block[6]:
                aie.EndOp()

    @property
    def Q_in(self):
        return self.S2MM[0]

    @property
    def K_in(self):
        return self.S2MM[1]


class kv_tile(AIE_CT_IP):
    """softmax(s) . v for one kv head, accumulating over the context chunks.

    Owns the two score buffers its qk neighbour writes into -- they are declared
    here and passed north, which is the whole reason the two tiles have to be
    vertically adjacent. Accumulates y in fp32 across chunks, then rescales by
    the softmax denominator once the last chunk of a token has landed.

    o is sized to the *padded* head count because the kernel writes
    Q_HEADS_PER_GROUP_PADDED * DH of it unconditionally (scale_div_aie<8 * DH>);
    the outbound BD then takes only the real heads, so the padding never leaves
    the tile.
    """
    dtype = np.dtype[bfloat16]

    def __init__(self, tile: tile_with_location, ctx: mlir_context, s_ty: Any, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "attn_kv.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing attn_kv_tile at ({self.ct_row}, {self.ct_col})")
        self.NQ_PER_CU = model_config.q_heads_per_group          # 6, what is real
        self.NQ_PER_CU_PADDED = model_config.q_heads_per_group_padded  # 8, what is written
        self.KV_HEADS_PER_CU = model_config.kv_heads_per_cu
        self.DH = model_config.DH
        self.lc = L_CHUNK

        self.attn_v_ty = np.ndarray[(self.lc, self.KV_HEADS_PER_CU * self.DH), self.dtype]
        # Padded: attn_kv.cc stores Q_HEADS_PER_GROUP_PADDED * DH into this.
        self.attn_o_ty = np.ndarray[(self.DH * self.NQ_PER_CU_PADDED, ), self.dtype]
        comment_print(f"\tattn_v_ty shape: ({self.lc}, {self.KV_HEADS_PER_CU * self.DH}), attn_o_ty shape: {(self.DH * self.NQ_PER_CU_PADDED, )} (padded; {self.NQ_PER_CU} heads leave)")
        self.attn_s_ty = s_ty
        self.L_ty = ctx.data_tys["L_ty"]
        self.s_ty = s_ty
        self.attn_kv = ctx.func("attn_kv", [self.attn_s_ty, self.attn_s_ty, self.attn_v_ty, self.attn_v_ty, self.attn_o_ty, self.attn_o_ty, self.L_ty])

        self.L_buffer = ctx.buffer(f"RTP_L_attn_kv_core_{self.ct_row}_{self.ct_col}", self.located_tile, self.L_ty, use_write_rtp = True)
        # Declared here, consumed by the qk tile to the north. The name carries
        # that tile's coordinates so the two agree on which buffer is which.
        self.s_buffer_0 = ctx.buffer(f"s_shared_buffer_0_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty)
        self.s_buffer_1 = ctx.buffer(f"s_shared_buffer_1_{self.ct_row + 1}_{self.ct_col}", tile = self.located_tile, dtype = self.s_ty)
        self.v_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_0_{self.ct_row}_{self.ct_col}")
        self.v_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_v_ty, name = f"v_1_{self.ct_row}_{self.ct_col}")

        self.o_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.attn_o_ty, name = f"o_0_{self.ct_row}_{self.ct_col}")
        self.o_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.attn_o_ty, name = f"o_1_{self.ct_row}_{self.ct_col}")

        self.o_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.o_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.v_prod_lock = aie.lock(self.ct, lock_id = 2, init = 2)
        self.v_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.s_proj_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.s_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)

        # o is dense but dim-block major, so it needs transposing on the way out.
        #
        # attn_fv accumulates y as mmul C tiles, where lane 8*r + c of a block is
        # head r, dim 8*d + c. scale_div_aie drops the padding -- it extracts
        # only NQ_PER_CU of the 8 heads per 64-wide block -- but it stores those
        # blocks in place, so o is [dim_block][head][8 dims]. Dense, and still
        # dim-block major. This BD walks it back into head-major order.
        #
        # The stride is NQ_PER_CU * 8, the *real* head count, because that is
        # the block scale_div_aie actually emits. Using NQ_PER_CU_PADDED here
        # indexes past the end of the real data and scattered the output: it
        # left 1570 of 6144 elements zero, 2/8 of the row, which reads exactly
        # like the padded head fraction and is easy to misdiagnose as the
        # transpose itself being spurious. Dropping the transpose instead leaves
        # every value correct but dim-block ordered -- a period-8 permutation.
        self.o_out_len = self.NQ_PER_CU * self.DH
        self.o_out_order = [(self.NQ_PER_CU, 8),
                            (self.DH // 8, self.NQ_PER_CU * 8),
                            (8, 1)]
        comment_print(f"\to_out: len {self.o_out_len}, order {self.o_out_order} "
                      f"([dim_block][head][8] -> head-major; {self.NQ_PER_CU} real heads)")

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.attn_kv(self.s_buffer_0, self.s_buffer_1, self.v_buffer_0, self.v_buffer_1, self.o_buffer_0, self.o_buffer_1, self.L_buffer)

        @aie.mem(self.ct)
        def mem_body(block):
            # ---- S2MM ch1: v, one L_CHUNK of context per firing --------------
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[1], chain=block[3])

            with block[1]:
                lock_acquire(self.v_prod_lock, 1)
                aie.dma_bd(self.v_buffer_0)
                lock_release(self.v_cons_lock, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.v_prod_lock, 1)
                aie.dma_bd(self.v_buffer_1)
                lock_release(self.v_cons_lock, 1)
                aie.next_bd(block[1])

            # ---- MM2S ch0: o, one token's real q heads per firing ------------
            # The padded tail stays behind: len is the real heads only.
            with block[3]:
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

            with block[4]:
                lock_acquire(self.o_cons_lock, 1)
                aie.dma_bd(self.o_buffer_0, offset = 0, len = self.o_out_len,
                           dimensions = self.o_out_order)
                lock_release(self.o_prod_lock, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.o_cons_lock, 1)
                aie.dma_bd(self.o_buffer_1, offset = 0, len = self.o_out_len,
                           dimensions = self.o_out_order)
                lock_release(self.o_prod_lock, 1)
                aie.next_bd(block[4])

            with block[6]:
                aie.EndOp()

    @property
    def V_in(self):
        return self.S2MM[1]

    @property
    def O_out(self):
        return self.MM2S[0]

    @property
    def get_s_buffers(self):
        return [self.s_buffer_0, self.s_buffer_1]


class attn_cu:
    """One kv head's worth of attention: a qk tile with its kv tile to the north.

    The pair has to be vertically adjacent, and the order is fixed by the
    kernels, not by preference. The score buffers are declared on the kv tile
    and the qk core writes into them as its *north* neighbour's memory; in the
    other direction `attn_kv.cc` reaches the qk core's locks with
    `_down_lock_acquire`, which only names the tile to the south. Flip the pair
    and each core signals a neighbour that is not listening: the run completes,
    because nothing deadlocks at the DMA level, and the output stays zero.
    """
    def __init__(self,
        attn_qk_tile: tile_with_location,
        attn_kv_tile: tile_with_location,
        model_config: ModelConfig,
        ctx: mlir_context,
        qk_stack_size: int,
        kv_stack_size: int,
        qk_link_with: str,
        kv_link_with: str
    ):
        comment_print(f"Initializing attn_cu at QK({attn_qk_tile.row}, {attn_qk_tile.col}) KV({attn_kv_tile.row}, {attn_kv_tile.col})")
        assert attn_qk_tile.col == attn_kv_tile.col and attn_kv_tile.row == attn_qk_tile.row + 1, \
            "the kv tile must sit directly north of its qk tile -- the qk core writes " \
            "the scores into the kv tile's memory, and attn_kv.cc reaches back with " \
            "_down_lock_acquire"
        self.NQ_PER_CU = model_config.q_heads_per_group_padded
        # Scores for the padded head count, plus a 16-element tail the two
        # kernels use to hand the softmax correction across.
        self.s_ty = np.ndarray[(self.NQ_PER_CU * 16 + 16, 1), np.dtype[bfloat16]]
        comment_print(f"\tattn_s_ty shape: {(self.NQ_PER_CU * 16 + 16, 1)}")
        self.kv_core = kv_tile(attn_kv_tile, ctx, self.s_ty, model_config, stack_size = kv_stack_size, link_with = kv_link_with)
        self.qk_core = qk_tile(attn_qk_tile, ctx, self.kv_core.get_s_buffers, self.s_ty, model_config, stack_size = qk_stack_size, link_with = qk_link_with)

    @property
    def Q_in(self):
        return self.qk_core.Q_in

    @property
    def K_in(self):
        return self.qk_core.K_in

    @property
    def V_in(self):
        return self.kv_core.V_in

    @property
    def O_out(self):
        return self.kv_core.O_out


class attn_memtile(AIE_MT_IP):
    """The KV staging and output assembly for one attention column.

    Two jobs, and they are independent:

      * **K/V replay.** The whole context arrives from DDR as | k | v | for the
        two kv heads this column owns, and has to be handed to the two CUs one
        L_CHUNK at a time, each CU seeing only its own head. The split is an
        offset, and the chunking is the BD length -- a self-chained BD fires
        once per chunk, so the host's transfer length decides how many chunks
        arrive and the kernels decide how many they consume. Those two counts
        have to agree, and the way they are made to agree is that **both sides
        derive the count from the longest token of the pass**,
        ceil(L[L_TOKENS - 1] / L_CHUNK) -- the kernels do not stop at their own
        L[t], they mask past it. A kernel that stopped early would leave the
        surplus chunks queued here for the next token to consume in place of the
        ones it wanted. See the assert in _replay_kv_cache and the comment in
        kernels/attn_qk.cc.

      * **O assembly.** Each CU emits its real q heads per token; this tile
        concatenates the column's two CUs into one contiguous run so the shim
        writes DDR with a single linear BD.

    Double-buffered on the input side so the next chunk lands while the current
    one is in the CUs.
    """

    def __init__(self, tile, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        self.KV_HEADS_PER_COLUMN = model_config.kv_heads_per_column     # 2
        self.KV_HEADS_PER_CU = model_config.kv_heads_per_cu             # 1
        self.DH = model_config.DH
        self.lc = L_CHUNK
        self.NQ_PER_CU = model_config.q_heads_per_group                 # 6, real
        self.cus_per_column = model_config.cus_per_column               # 2

        self.kv_row = self.KV_HEADS_PER_COLUMN * self.DH                # 512
        self.cu_row = self.KV_HEADS_PER_CU * self.DH                    # 256
        self.chunk_elems = self.lc * self.kv_row                        # one chunk, both heads

        comment_print(f"Initializing attn_memtile at ({self.mt_row}, {self.mt_col}): "
                      f"chunk {self.lc} x {self.kv_row} = {self.chunk_elems} bf16 "
                      f"({self.chunk_elems * 2 // 1024} KiB), {self.cus_per_column} CUs")

        self.kv_chunk_ty = np.ndarray[(self.lc, self.kv_row), np.dtype[bfloat16]]
        # One token of assembled output for the whole column.
        self.o_ty = np.ndarray[(self.cus_per_column * self.NQ_PER_CU * self.DH, ), np.dtype[bfloat16]]

        self.k_buffer_0 = aie.buffer(self.mt, self.kv_chunk_ty, name = f"k_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.k_buffer_1 = aie.buffer(self.mt, self.kv_chunk_ty, name = f"k_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.v_buffer_0 = aie.buffer(self.mt, self.kv_chunk_ty, name = f"v_mem_buffer_0_{self.mt_row}_{self.mt_col}")
        self.v_buffer_1 = aie.buffer(self.mt, self.kv_chunk_ty, name = f"v_mem_buffer_1_{self.mt_row}_{self.mt_col}")
        self.o_buffer   = aie.buffer(self.mt, self.o_ty,        name = f"o_buffer_{self.mt_row}_{self.mt_col}")

        # Each chunk is consumed by both CUs, so a producer credit comes back
        # only after the second one has taken it: cons_0 -> cons_1 -> prod.
        self.k_prod_lock   = aie.lock(self.mt, lock_id = 0, init = 2)
        self.k_cons_lock_0 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.k_cons_lock_1 = aie.lock(self.mt, lock_id = 2, init = 0)
        self.v_prod_lock   = aie.lock(self.mt, lock_id = 3, init = 2)
        self.v_cons_lock_0 = aie.lock(self.mt, lock_id = 4, init = 0)
        self.v_cons_lock_1 = aie.lock(self.mt, lock_id = 5, init = 0)
        # o fills CU 0's half then CU 1's, then leaves once, to this column's
        # gate tile: prod_0 -> prod_1 -> cons -> prod_0.
        self.o_prod_lock_0 = aie.lock(self.mt, lock_id = 6, init = 1)
        self.o_prod_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.o_cons_lock   = aie.lock(self.mt, lock_id = 8, init = 0)

        # **k and v leave transposed, and differently from each other.** The two
        # kernels consume them as mmul B operands, not as rows: attn_qk wants k
        # dim-major so its 8x8 transpose lands right, and attn_fv walks v in
        # half-chunk planes. A plain [t][d] slice of this CU's head -- which is
        # what the data looks like in the buffer -- gives the first few elements
        # of a head correctly and scrambles the rest, because only the first
        # 8-element group lines up. That was worth about 0.02 cosine against the
        # golden, with token 0 (which attends to one key, so its output is just
        # v) matching on its opening elements and diverging after.
        #
        # These are the reference's orders from generic_decoding_layer, with the
        # geometry substituted: a chunk is lc x kv_row here and a CU takes
        # cu_row of every row.
        #
        #   k: [d/8][t][d%8]      -> i * 8 + t * kv_row + j
        #   v: [half][d/8][t][d%8] with the half plane stride lc/2 * kv_row
        self.k_order = [(self.cu_row // 8, 8), (self.lc, self.kv_row), (8, 1)]
        self.v_order = [(self.lc // 8, self.lc // 2 * self.kv_row),
                        (self.cu_row // 8, 8), (8, self.kv_row), (8, 1)]
        self.kv_len_per_cu = self.lc * self.cu_row
        self.o_half = self.NQ_PER_CU * self.DH
        comment_print(f"\tk_order: {self.k_order}")
        comment_print(f"\tv_order: {self.v_order}")
        comment_print(f"\tlen per cu: {self.kv_len_per_cu}, o half: {self.o_half}")

        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            # ---- S2MM ch0: k chunks from the shim ----------------------------
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])

            with block[1]:
                lock_acquire(self.k_prod_lock, 1)
                aie.dma_bd(self.k_buffer_0)
                lock_release(self.k_cons_lock_0, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.k_prod_lock, 1)
                aie.dma_bd(self.k_buffer_1)
                lock_release(self.k_cons_lock_0, 1)
                aie.next_bd(block[1])

            # ---- MM2S ch0: k to CU 0, its head only --------------------------
            with block[3]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest=block[4], chain=block[6])

            with block[4]:
                lock_acquire(self.k_cons_lock_0, 1)
                aie.dma_bd(self.k_buffer_0, offset = 0, len = self.kv_len_per_cu, dimensions = self.k_order)
                lock_release(self.k_cons_lock_1, 1)
                aie.next_bd(block[5])

            with block[5]:
                lock_acquire(self.k_cons_lock_0, 1)
                aie.dma_bd(self.k_buffer_1, offset = 0, len = self.kv_len_per_cu, dimensions = self.k_order)
                lock_release(self.k_cons_lock_1, 1)
                aie.next_bd(block[4])

            # ---- MM2S ch1: k to CU 1, the column's second head ---------------
            with block[6]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest=block[7], chain=block[9])

            with block[7]:
                lock_acquire(self.k_cons_lock_1, 1)
                aie.dma_bd(self.k_buffer_0, offset = self.cu_row, len = self.kv_len_per_cu, dimensions = self.k_order)
                lock_release(self.k_prod_lock, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.k_cons_lock_1, 1)
                aie.dma_bd(self.k_buffer_1, offset = self.cu_row, len = self.kv_len_per_cu, dimensions = self.k_order)
                lock_release(self.k_prod_lock, 1)
                aie.next_bd(block[7])

            # ---- S2MM ch1: v chunks from the shim ----------------------------
            with block[9]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[10], chain=block[12])

            with block[10]:
                lock_acquire(self.v_prod_lock, 1)
                aie.dma_bd(self.v_buffer_0)
                lock_release(self.v_cons_lock_0, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.v_prod_lock, 1)
                aie.dma_bd(self.v_buffer_1)
                lock_release(self.v_cons_lock_0, 1)
                aie.next_bd(block[10])

            # ---- MM2S ch2: v to CU 0 -----------------------------------------
            with block[12]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 2, dest=block[13], chain=block[15])

            with block[13]:
                lock_acquire(self.v_cons_lock_0, 1)
                aie.dma_bd(self.v_buffer_0, offset = 0, len = self.kv_len_per_cu, dimensions = self.v_order)
                lock_release(self.v_cons_lock_1, 1)
                aie.next_bd(block[14])

            with block[14]:
                lock_acquire(self.v_cons_lock_0, 1)
                aie.dma_bd(self.v_buffer_1, offset = 0, len = self.kv_len_per_cu, dimensions = self.v_order)
                lock_release(self.v_cons_lock_1, 1)
                aie.next_bd(block[13])

            # ---- MM2S ch3: v to CU 1 -----------------------------------------
            with block[15]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 3, dest=block[16], chain=block[18])

            with block[16]:
                lock_acquire(self.v_cons_lock_1, 1)
                aie.dma_bd(self.v_buffer_0, offset = self.cu_row, len = self.kv_len_per_cu, dimensions = self.v_order)
                lock_release(self.v_prod_lock, 1)
                aie.next_bd(block[17])

            with block[17]:
                lock_acquire(self.v_cons_lock_1, 1)
                aie.dma_bd(self.v_buffer_1, offset = self.cu_row, len = self.kv_len_per_cu, dimensions = self.v_order)
                lock_release(self.v_prod_lock, 1)
                aie.next_bd(block[16])

            # ---- S2MM ch2/ch3: o from the two CUs, into the two halves -------
            # Serialised by the lock chain so the buffer is only released once
            # both halves of a token are in.
            with block[18]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 2, dest=block[19], chain=block[20])

            with block[19]:
                lock_acquire(self.o_prod_lock_0, 1)
                aie.dma_bd(self.o_buffer, offset = 0, len = self.o_half)
                lock_release(self.o_prod_lock_1, 1)
                aie.next_bd(block[19])

            with block[20]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 3, dest=block[21], chain=block[22])

            with block[21]:
                lock_acquire(self.o_prod_lock_1, 1)
                aie.dma_bd(self.o_buffer, offset = self.o_half, len = self.o_half)
                lock_release(self.o_cons_lock, 1)
                aie.next_bd(block[21])

            # ---- MM2S ch4: the column's assembled token, to its gate tile ----
            # A half-row, not a whole one: the gate is split per column -- see
            # iron/gate.py.
            with block[22]:
                aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 4, dest=block[23], chain=block[24])

            with block[23]:
                lock_acquire(self.o_cons_lock, 1)
                aie.dma_bd(self.o_buffer)
                lock_release(self.o_prod_lock_0, 1)
                aie.next_bd(block[23])

            with block[24]:
                aie.EndOp()

    @property
    def K_in(self):
        return self.S2MM[0]

    @property
    def V_in(self):
        return self.S2MM[1]

    @property
    def K_out_0(self):
        return self.MM2S[0]

    @property
    def K_out_1(self):
        return self.MM2S[1]

    @property
    def V_out_0(self):
        return self.MM2S[2]

    @property
    def V_out_1(self):
        return self.MM2S[3]

    @property
    def O_in_0(self):
        return self.S2MM[2]

    @property
    def O_in_1(self):
        return self.S2MM[3]

    @property
    def O_out(self):
        """The assembled half-row, to this column's gate tile."""
        return self.MM2S[4]


class attn_column:
    """One column of the attention: two CUs, their KV memtile and its shim.

    Four compute tiles in rows 0..3, paired north-over-south into two CUs, so a
    column holds two kv heads. Two columns cover all four.
    """
    def __init__(self,
        attn_qk_tiles: List[tile_with_location],
        attn_kv_tiles: List[tile_with_location],
        memtile: tile_with_location,
        shim_tile: ShimTile,
        model_config: ModelConfig,
        ctx: mlir_context,
        qk_stack_size: int,
        kv_stack_size: int,
        qk_link_with: str,
        kv_link_with: str
    ):
        comment_print(f"Initializing attn_column at MT({memtile.row}, {memtile.col})")
        assert len(attn_qk_tiles) == len(attn_kv_tiles) == model_config.cus_per_column
        self.attn_cus = []
        for i in range(len(attn_qk_tiles)):
            self.attn_cus.append(attn_cu(
                attn_qk_tile = attn_qk_tiles[i],
                attn_kv_tile = attn_kv_tiles[i],
                model_config = model_config,
                ctx = ctx,
                qk_stack_size = qk_stack_size,
                kv_stack_size = kv_stack_size,
                qk_link_with = qk_link_with,
                kv_link_with = kv_link_with
            ))
        self.memtile = attn_memtile(memtile, ctx, model_config)
        self.IT = shim_tile

    def route(self):
        """The column's internal flows. All circuit-switched: every one is a
        single source to a single destination."""
        connect_port(self.IT.MM2S[0], self.memtile.K_in)
        connect_port(self.IT.MM2S[1], self.memtile.V_in)
        connect_port(self.memtile.K_out_0, self.attn_cus[0].K_in)
        connect_port(self.memtile.V_out_0, self.attn_cus[0].V_in)
        connect_port(self.memtile.K_out_1, self.attn_cus[1].K_in)
        connect_port(self.memtile.V_out_1, self.attn_cus[1].V_in)
        connect_port(self.attn_cus[0].O_out, self.memtile.O_in_0)
        connect_port(self.attn_cus[1].O_out, self.memtile.O_in_1)
        # The assembled token goes to the gate tile and is wired at top level,
        # since the column does not own that tile.

    @property
    def Q_in(self):
        return [cu.Q_in for cu in self.attn_cus]

    @property
    def O_out(self):
        """To the gate tile. Wired by the caller -- see iron/layer_aie.py."""
        return self.memtile.O_out

    @property
    def qk_tiles(self):
        return [cu.qk_core for cu in self.attn_cus]

    @property
    def kv_tiles(self):
        return [cu.kv_core for cu in self.attn_cus]
