# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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


class gate_tile(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    def __init__(self, ct: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "gate.o"):
        super().__init__(ct, stack_size = stack_size, link_with = link_with)

        # The gated row's DDR tap:
        #   MM2S[0] (Attn_Out) replays it into o_proj's x FIFO, `repeats` times,
        #   MM2S[1] (TB_Out)   would send it once to DDR as the scored gate tap.
        #
        # **Off, and it must stay off until TB_Out is actually routed.** With it
        # on, block[5] hands its credit to tb_lock instead of to
        # gated_attn_out_prod_lock, and only block[7] -- the DDR send -- puts
        # that credit back. MTP_layer_aie.py never calls connect_port on
        # TB_Out, and an unrouted MM2S chain still fires: block[7] takes all
        # `repeats` tb credits, stalls on a channel that reaches nothing, and
        # never releases the producer lock. The gate core then runs pass 1
        # entirely on the lock's initial credit and blocks forever on pass 2,
        # which read as "the first dispatch completes and every one after it
        # times out" -- and, because o_proj's x never arrives, the timeout
        # landed on the o weight round rather than anywhere near here.
        #
        # To bring the tap back: wire gate_core.TB_Out to a shim in
        # MTP_layer_aie.py AND post a matching receive in the host sequence
        # (decoding_layer::_gen_sequence) and in the DLL port's
        # gen_mtp_layer_seq(). Routing it without receiving it hangs the same
        # way. gated_attention_verify has no tb chain at all, which is why that
        # design re-dispatches.
        self.enable_tb = False

        self.DQ = model_config.DQ
        self.D = model_config.D
        # How many times the gated row is sent into the projection engine.
        #
        # o_proj is a streaming mvm(): it re-reads the whole activation once per
        # row block, and it owns D / (m * 16) = 10 of them. This is the same
        # number rms_residual.cc replays xn QKV_REPEATS times for.
        #
        # Must stay equal to `repeats` in kernels/gate.cc: the core releases the
        # consumer lock that many times and block[5] takes one credit per send.
        self.repeats = model_config.D // (model_config.q4nx_row_block_size * 16)
        # q_proj tile and mem 
        # double buffer
        self.gate_ty = np.ndarray[(self.DQ,), self.dtype]

        self.gate_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"gate_{self.ct_row}_{self.ct_col}")
        self.attn_out_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"attn_out_{self.ct_row}_{self.ct_col}")
        self.gated_attn_out_buffer = aie.buffer(tile = self.ct, datatype = self.gate_ty, name = f"gated_attn_out_{self.ct_row}_{self.ct_col}")

        self.gate_prod_lock = aie.lock(self.ct, lock_id = 0, init = 1)  
        self.gate_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.attn_out_prod_lock = aie.lock(self.ct, lock_id = 2, init = 1)
        self.attn_out_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.gated_attn_out_prod_lock = aie.lock(self.ct, lock_id = 4, init = self.repeats)
        self.gated_attn_out_cons_lock = aie.lock(self.ct, lock_id = 5, init = 0)
        self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)
        self.tb_lock = aie.lock(self.ct, lock_id = 7, init = 0)

        self.gate = aie.external_func("gate", [self.gate_ty, self.gate_ty, self.gate_ty])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.gate(self.gated_attn_out_buffer, self.gate_buffer, self.attn_out_buffer)
        
        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[2])
            with block[1]:
                lock_acquire(self.gate_prod_lock, 1)
                aie.dma_bd(self.gate_buffer)
                lock_release(self.gate_cons_lock, 1)
                aie.next_bd(block[1])


            with block[2]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest = block[3], chain = block[4])

            with block[3]:
                lock_acquire(self.attn_out_prod_lock, 1)
                aie.dma_bd(self.attn_out_buffer)
                lock_release(self.attn_out_cons_lock, 1)
                aie.next_bd(block[3])

            with block[4]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[5], chain = block[6 if self.enable_tb else 8])
                
            with block[5]: # the gated row, replayed into o_proj's x FIFO
                lock_acquire(self.gated_attn_out_cons_lock, 1)
                # One credit per send, and the core released `repeats` of them,
                # so this BD fires that many times off a single core iteration.
                # That replay *is* o_proj's M/m loop: the mvm re-reads the whole
                # DQ-wide row once per row block.
                #
                # dma_bd_packet rather than the in-band header the rms and
                # passthrough tiles write at y+14: those stamp the id from the
                # kernel because the same buffer carries different ids on
                # different phases, whereas this row only ever goes one place.
                # The RoPE tile's k|v sends take the same route.
                aie.dma_bd_packet(0, aux_to_proj_x_in_pkt_id.to_out_proj)
                aie.dma_bd(self.gated_attn_out_buffer)
                if (self.enable_tb):
                    lock_release(self.tb_lock, 1)
                else:
                    lock_release(self.gated_attn_out_prod_lock, 1)
                aie.next_bd(block[5])

            # Emitted only when the tap is on. Leaving these in the module while
            # block[4] chains past them would leave a dma_start on MM2S ch1 for
            # the lowering to reach on its own; the chain is the only thing that
            # is supposed to decide, so do not hand it a second opinion.
            if not self.enable_tb:
                with block[8]:
                    aie.EndOp()
                return

            with block[6]:
                s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 1, dest = block[7], chain = block[8])

            with block[7]: # the scored tap, to DDR via IT[2]
                # Takes all `repeats` credits at once, so it sends the row once
                # and only after every replay above has left. That ordering is
                # what makes the buffer safe to recycle: the producer lock is
                # released here, not in block[5].
                #
                # No dma_bd_packet: this is a circuit flow to a shim and the
                # header would land in the DDR buffer as leading garbage.
                lock_acquire(self.tb_lock, self.repeats)
                aie.dma_bd(self.gated_attn_out_buffer)
                lock_release(self.gated_attn_out_prod_lock, self.repeats)
                aie.next_bd(block[7])
            with block[8]:
                aie.EndOp()

    @property
    def Gate_In(self) -> dma_port:
        return self.S2MM[0]

    @property
    def Attn_In(self) -> dma_port:
        return self.S2MM[1]
    
    @property
    def Attn_Out(self) -> dma_port:
        return self.MM2S[0]
    
    @property
    def TB_Out(self) -> dma_port:
        return self.MM2S[1]
