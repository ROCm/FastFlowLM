# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from aie.extras.dialects.ext.arith import constant, index_cast
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from ml_dtypes import bfloat16

import threading
from utils import mlir_context, tile_with_location, connect_dma, dma_port, connect_port, lock_acquire, lock_release, comment_print, ShimTile
from utils import AIE_CT_IP, AIE_MT_IP

from constants import proj_pkt_id

from typing import Any, List
from aie.ir import IntegerAttr, IntegerType

# Mirrors the makefiles' ENABLE_CHESSCC switch (1 = chess, 0 = Peano).
ENABLE_CHESSCC = os.environ.get("ENABLE_CHESSCC", "0") == "1"
BANK = 16 * 1024


def peano_proj_bank_plan(stack_size: int, w_bytes: int, y_bytes: int, x_bytes: int) -> list:
    """Peano layout of a proj core: (buffer, bank, bytes), in placement order.

    Peano's linker puts .data / .bss only after the highest buffer, so the int8
    x caches of proj_main.cc (20 KB) need one free block there. w, x and y go
    into banks 0 and 1, and a pad pushes bank 1's last buffer to the bank's
    end, which leaves banks 2 and 3 to the caches (the inner loop reads them
    together with w). Only the bank is pinned: this mlir-aie's bank-aware
    allocator counts a buffer pinned by address twice. See README.md, "Peano".
    """
    pad = BANK - w_bytes - y_bytes - x_bytes
    assert stack_size + w_bytes + x_bytes + y_bytes <= BANK, "proj bank 0 overflow"
    assert pad >= 0 and (BANK - x_bytes) % 64 == 0
    return [("w_0", 0, w_bytes), ("x_0", 0, x_bytes), ("y_0", 0, y_bytes),
            ("w_1", 1, w_bytes), ("pad", 1, pad), ("y_1", 1, y_bytes), ("x_1", 1, x_bytes)]

# phase 1: qkv proj, D -> (D + D // 2) to RoPE

# phase 2: o_proj, D -> D to rms_residual

# phase 3: up/gate proj, D -> D * 8 to MT_GLU

# phase 4: down proj, D * 4 -> D, to rms_residual

# minimal output unit: (m, 1)

class proj_tile_main(AIE_CT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "proj_main.o",
                 send_x_out:bool=True, main_y_buffer_0:Any|None= None, main_y_buffer_1:Any|None=None,
                 col_idx: int = 0
                 ):
        
        """
        send_x_out: boolean determine of this CT sends the output x to mt
        
        For context: 2CT are group into one mvm unit produce length of 2*m output, but only CT with send_x_out=True responsible sending the output to MT
        
        IF `send_x_out`==False, this CT writes the output to `main_y_buffer_0` and `main_y_buffer_1`, the output buffer are buffer allocate in its neighbor CT that has `send_x_out`=True
        """
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing proj_tile at ({tile.row}, {tile.col})")

        self.send_x_out = send_x_out
        # which of the 4 proj columns this core is in (0..3): in the routed-expert
        # down step every core receives all 4 experts' x and keeps only its own
        self.col_idx = col_idx
        
        self.m = model_config.q4nx_row_block_size
        self.x_slice_size = model_config.x_slice_size
        
        self.x_slice_ty = np.ndarray[(self.x_slice_size,), np.dtype[bfloat16]]
        # 2 weight blocks per buffer, 2 buffers: a 4-block ring (kernels/proj_main.cc W_RING_BLOCKS)
        self.w_block = model_config.W_BLOCK_SIZE
        self.linear_w_ty = np.ndarray[(2 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        self.m_pkt_ty = np.ndarray[(2 * self.m + 16,), np.dtype[bfloat16]]

        types = {"x_0": self.x_slice_ty, "x_1": self.x_slice_ty, "w_0": self.linear_w_ty,
                 "w_1": self.linear_w_ty, "y_0": self.m_pkt_ty, "y_1": self.m_pkt_ty}
        bufs = {}
        if ENABLE_CHESSCC:
            # the original declaration order; the allocator's layout depends on it
            for kind in ["x_0", "w_0", "x_1", "w_1", "y_0", "y_1"]:
                bufs[kind] = ctx.buffer(tile = self.located_tile, dtype = types[kind],
                                        name = f"{kind}_{self.ct_row}_{self.ct_col}")
        else:
            i32 = IntegerType.get_signless(32)
            plan = peano_proj_bank_plan(self.stack_size, 2 * 2 * self.w_block, 2 * (2 * self.m + 16), 2 * self.x_slice_size)
            for kind, bank, nbytes in plan:
                ty = types.get(kind, np.ndarray[(nbytes,), np.dtype[np.int8]])
                b = ctx.buffer(tile = self.located_tile, dtype = ty, name = f"{kind}_{self.ct_row}_{self.ct_col}")
                b.owner.attributes["mem_bank"] = IntegerAttr.get(i32, bank)
                bufs[kind] = b

        self.x_buffer_0 = bufs["x_0"]
        self.w_buffer_0 = bufs["w_0"]
        self.x_buffer_1 = bufs["x_1"]
        self.w_buffer_1 = bufs["w_1"]

        if send_x_out:
            self.y_buffer_0 = bufs["y_0"]
            self.y_buffer_1 = bufs["y_1"]
        else:
            assert main_y_buffer_0 is not  None
            self.y_buffer_0 = main_y_buffer_0
            assert main_y_buffer_1 is not None
            self.y_buffer_1 = main_y_buffer_1

            # allocate fake buffer for the sake of same rtp address
            # or use a different addressing mode
            self._y_0 = bufs["y_0"]
            self._y_1 = bufs["y_1"]
        self.x_prod_lock = aie.lock(self.ct, lock_id = 0, init = 2)
        self.x_cons_lock = aie.lock(self.ct, lock_id = 1, init = 0)
        self.w_prod_lock = aie.lock(self.ct, lock_id = 2, init = 4)   # 4 one-block slots
        self.w_cons_lock = aie.lock(self.ct, lock_id = 3, init = 0)
        self.y_prod_ping_lock = aie.lock(self.ct, lock_id = 4, init = 2)
        self.y_prod_pong_lock = aie.lock(self.ct, lock_id = 5, init = 2)
        self.rtp_available_lock = aie.lock(self.ct, lock_id = 6, init = 0)
        self.y_cons_ping_lock = aie.lock(self.ct, lock_id = 7, init = 0)
        self.y_cons_pong_lock = aie.lock(self.ct, lock_id = 8, init = 0)
        
        self.proj = ctx.func("proj_main", [self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, self.m_pkt_ty, self.linear_w_ty, self.x_slice_ty, np.int32, np.int32])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                if self.send_x_out:
                    self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, constant(1), constant(self.col_idx))
                else:
                    self.proj(self.y_buffer_0, self.w_buffer_0, self.x_buffer_0, self.y_buffer_1, self.w_buffer_1, self.x_buffer_1, constant(0), constant(self.col_idx))
        
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
                s1 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 1, dest=block[20], chain=block[6])

            # weight ring: one block per BD, w_buffer_0 [0], [1], w_buffer_1 [0], [1]
            slots = [(self.w_buffer_0, 0), (self.w_buffer_0, 1), (self.w_buffer_1, 0), (self.w_buffer_1, 1)]
            for k, (buf, half) in enumerate(slots):
                with block[20 + k]:
                    lock_acquire(self.w_prod_lock, 1)
                    aie.dma_bd(buf, offset = half * self.w_block, len = self.w_block)
                    lock_release(self.w_cons_lock, 1)
                    aie.next_bd(block[20 + (k + 1) % len(slots)])
            if self.send_x_out:
                with block[6]:
                    s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[7], chain = block[9])
                    
                with block[7]: # x, to mt
                    lock_acquire(self.y_cons_ping_lock, 2)
                    aie.dma_bd(self.y_buffer_0, offset = 14, len = 2 * self.m + 2)
                    lock_release(self.y_prod_ping_lock, 2)
                    aie.next_bd(block[8])


                with block[8]: # x, to mt
                    lock_acquire(self.y_cons_pong_lock, 2)
                    aie.dma_bd(self.y_buffer_1, offset = 14, len = 2 * self.m + 2)
                    lock_release(self.y_prod_pong_lock, 2)
                    aie.next_bd(block[7])

                with block[9]:
                    aie.EndOp()
            else:
                with block[6]:
                    aie.EndOp()

    @property
    def y_buffer_ping(self):
        return self.y_buffer_0
    
    @property
    def y_buffer_pong(self):
        return self.y_buffer_1
    
    @property
    def w_in(self):
        return self.S2MM[1]
    
    @property
    def x_in(self):
        return self.S2MM[0]
    
    @property
    def y_out(self):
        return self.MM2S[0]

#         return self.S2MM[0]

class proj_combo:
    def __init__(self, cts: List[tile_with_location], ctx, model_config: ModelConfig, stack_size: int = 1024, link_with: List[str] = ["proj_main.o"],
                 col_idx: int = 0):
        self.proj_main = proj_tile_main(tile=cts[0],ctx= ctx, model_config=model_config,stack_size= stack_size,link_with= link_with[0], send_x_out=True,
                                        col_idx=col_idx)
        
        
        self.proj = proj_tile_main(tile=cts[1], ctx=ctx, model_config=model_config, stack_size=stack_size, 
                                   link_with=link_with[0], send_x_out=False, main_y_buffer_0=self.proj_main.y_buffer_ping, main_y_buffer_1=self.proj_main.y_buffer_pong,
                                   col_idx=col_idx
                                   
                                   )

    @property
    def w_in_0(self):
        return self.proj_main.w_in
    @property
    def x_in_0(self):
        return self.proj_main.x_in
    @property
    def w_in_1(self):
        return self.proj.w_in
    @property
    def x_in_1(self):
        return self.proj.x_in
    @property
    def y_out(self):
        return self.proj_main.y_out

class proj_assemble_col(AIE_MT_IP):

    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_col at ({tile.row}, {tile.col})")

        self.m = model_config.q4nx_row_block_size
        self.w_block_size = model_config.W_BLOCK_SIZE
        self.m_col_ty = np.ndarray[(8 * self.m + 2,), np.dtype[bfloat16]]
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_col m: {self.m}")
        comment_print(f"proj_assemble_col w_block_size: {self.w_block_size}")

        self.y_buffer_0 = aie.buffer(self.mt, self.m_col_ty, name = f"y_buffer_0_{self.mt_row}_{self.mt_col}")
        self.y_buffer_1 = aie.buffer(self.mt, self.m_col_ty, name = f"y_buffer_1_{self.mt_row}_{self.mt_col}")

        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.y_prod_lock_0 = aie.lock(self.mt, lock_id = 0, init = 2)
        self.y_prod_lock_1 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.y_prod_lock_2 = aie.lock(self.mt, lock_id = 2, init = 0)
        self.y_prod_lock_3 = aie.lock(self.mt, lock_id = 3, init = 0)
        self.y_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 8, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 9, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        self.__call__()

    @property
    def y_in(self):
        return [self.S2MM[i] for i in range(4)]
    
    @property
    def y_out(self):
        return self.MM2S[4]
    
    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]
    
    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 2 * self.m + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[2])

            with block[2]:
                lock_acquire(self.y_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 2 * self.m + 2)
                lock_release(self.y_prod_lock_1, 1)
                aie.next_bd(block[1])
                
            with block[3]:
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[4], chain = block[6])
            
            with block[4]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 2 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[5])
                
            with block[5]:
                lock_acquire(self.y_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 2 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_2, 1)
                aie.next_bd(block[4])
                
            with block[6]:
                s2 = aie.dma_start(aie.DMAChannelDir.S2MM, 2, dest = block[7], chain = block[9])
            
            with block[7]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_0, offset = 4 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[8])

            with block[8]:
                lock_acquire(self.y_prod_lock_2, 1)
                aie.dma_bd(self.y_buffer_1, offset = 4 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_prod_lock_3, 1)
                aie.next_bd(block[7])

            with block[9]:
                s3 = aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[10], chain = block[12])

            with block[10]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_0, offset = 6 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_cons_lock, 1)
                aie.next_bd(block[11])

            with block[11]:
                lock_acquire(self.y_prod_lock_3, 1)
                aie.dma_bd(self.y_buffer_1, offset = 6 * self.m + 2, len = 2 * self.m)
                lock_release(self.y_cons_lock, 1)
                aie.next_bd(block[10])
                
            with block[12]:
                s4 = aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[13], chain = block[15])

            with block[13]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[14])
                
            with block[14]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[13])

            with block[15]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[16], chain = block[18])

            with block[16]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[17])

            with block[17]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[16])

            with block[18]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[19], chain = block[21])

            with block[19]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[20])
                
            with block[20]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[19])

            with block[21]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[22], chain = block[24])

            with block[22]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[23])

            with block[23]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[22])
                
            with block[24]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[25], chain = block[27])

            with block[25]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[26])

            with block[26]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[25])

            with block[27]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 4, dest = block[28], chain = block[30])

            with block[28]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.y_prod_lock_0, 1)
                aie.next_bd(block[29])

            with block[29]:
                lock_acquire(self.y_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.y_prod_lock_0, 1)
                aie.next_bd(block[28])

            with block[30]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[31], chain = block[33])
            
            with block[31]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[32])
                
            with block[32]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[31])

            with block[33]:
                aie.EndOp()
    
class proj_assemble_weight_only(AIE_MT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_col at ({tile.row}, {tile.col})")

        self.w_block_size = model_config.W_BLOCK_SIZE
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_col w_block_size: {self.w_block_size}")


        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 8, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 9, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        
        self.__call__()

    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]
    
    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]
    


    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[13], chain = block[15])

            with block[13]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[14])
                
            with block[14]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[13])

            with block[15]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[16], chain = block[18])

            with block[16]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[17])

            with block[17]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[16])

            with block[18]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[19], chain = block[21])

            with block[19]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[20])
                
            with block[20]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[19])

            with block[21]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[22], chain = block[24])

            with block[22]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[23])

            with block[23]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[22])
                
            with block[24]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[25], chain = block[30])

            with block[25]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[26])

            with block[26]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[25])

            with block[30]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[31], chain = block[33])
            
            with block[31]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[32])
                
            with block[32]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[31])


            with block[33]:
                aie.EndOp()

class proj_assemble_weight_x(AIE_MT_IP):
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):
        super().__init__(tile)
        comment_print(f"Initializing proj_assemble_col at ({tile.row}, {tile.col})")

        self.w_block_size = model_config.W_BLOCK_SIZE
        self.m = model_config.q4nx_row_block_size
        self.linear_4w_ty = np.ndarray[(4 * model_config.W_BLOCK_SIZE,), np.dtype[bfloat16]]
        comment_print(f"proj_assemble_col m: {self.m}")
        comment_print(f"proj_assemble_col w_block_size: {self.w_block_size}")

        self.x_chunk_ty = np.ndarray[(model_config.x_slice_size * 2,), np.dtype[bfloat16]]
        self.m_full_ty = np.ndarray[(16 * self.m + 2,), np.dtype[bfloat16]]
        self.y_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_0_{self.mt_row}_{self.mt_col}")
        self.y_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.m_full_ty, name = f"y_buffer_1_{self.mt_row}_{self.mt_col}")
        self.x_buffer_0 = ctx.buffer(tile = self.located_tile, dtype = self.x_chunk_ty, name = f"x_buffer_0_{self.mt_row}_{self.mt_col}")
        self.x_buffer_1 = ctx.buffer(tile = self.located_tile, dtype = self.x_chunk_ty, name = f"x_buffer_1_{self.mt_row}_{self.mt_col}")


        self.m_prod_lock_0 = aie.lock(self.mt, lock_id = 0, init = 2)
        self.m_prod_lock_1 = aie.lock(self.mt, lock_id = 1, init = 0)
        self.m_cons_lock = aie.lock(self.mt, lock_id = 2, init = 0)
        self.x_prod_lock = aie.lock(self.mt, lock_id = 3, init = 2)
        self.x_cons_lock = aie.lock(self.mt, lock_id = 4, init = 0)

        self.w_buffer_0 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_0_{self.mt_row}_{self.mt_col}")
        self.w_buffer_1 = aie.buffer(self.mt, self.linear_4w_ty, name = f"w_buffer_1_{self.mt_row}_{self.mt_col}")
        self.w_p0_prod_lock = aie.lock(self.mt, lock_id = 5, init = 2)
        self.w_p0_cons_lock_0 = aie.lock(self.mt, lock_id = 6, init = 0)
        self.w_p0_cons_lock_1 = aie.lock(self.mt, lock_id = 7, init = 0)
        self.w_p1_prod_lock = aie.lock(self.mt, lock_id = 8, init = 2)
        self.w_p1_cons_lock_0 = aie.lock(self.mt, lock_id = 9, init = 0)
        self.w_p1_cons_lock_1 = aie.lock(self.mt, lock_id = 10, init = 0)
        
        self.__call__()

    @property
    def w_in(self):
        return [self.S2MM[i] for i in range(4, 6)]
    
    @property
    def w_out(self):
        return [self.MM2S[i] for i in range(4)]
    
    @property
    def x_out(self):
        return self.MM2S[4]
    
    @property
    def x_in(self):
        return self.S2MM[3]
    

    @property
    def y_out(self):
        return self.MM2S[5]
    
    @property
    def y_in(self):
        return [self.S2MM[i] for i in range(2)]

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            chain_id = 0
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[chain_id + 1], chain = block[chain_id + 3])
            with block[chain_id + 1]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_0, offset = 0, len = 8 * self.m + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_prod_lock_0, 1)
                aie.dma_bd(self.y_buffer_1, offset = 0, len = 8 * self.m + 2)
                lock_release(self.m_prod_lock_1, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_0, offset = 8 * self.m + 2, len = 8 * self.m)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_prod_lock_1, 1)
                aie.dma_bd(self.y_buffer_1, offset = 8 * self.m + 2, len = 8 * self.m)
                lock_release(self.m_cons_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 3, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.x_prod_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_cons_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 4, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 2])
                
            with block[chain_id + 2]:
                lock_acquire(self.w_p0_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = 2 * self.w_block_size)
                lock_release(self.w_p0_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.S2MM, 5, dest = block[chain_id + 1], chain = block[chain_id + 3])
            
            with block[chain_id + 1]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 2])
                
            with block[chain_id + 2]:
                lock_acquire(self.w_p1_prod_lock, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = 2 * self.w_block_size)
                lock_release(self.w_p1_cons_lock_0, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 5, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.m_cons_lock, 1)
                aie.dma_bd(self.y_buffer_0)
                lock_release(self.m_prod_lock_0, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.m_cons_lock, 1)
                aie.dma_bd(self.y_buffer_1)
                lock_release(self.m_prod_lock_0, 1)
                aie.next_bd(block[chain_id + 1])


            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 4, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_buffer_0)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.x_cons_lock, 1)
                aie.dma_bd(self.x_buffer_1)
                lock_release(self.x_prod_lock, 1)
                aie.next_bd(block[chain_id + 1])


            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.w_p0_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 0, len = self.w_block_size)
                lock_release(self.w_p0_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 1, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])
                
            with block[chain_id + 2]:
                lock_acquire(self.w_p0_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p0_prod_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 2, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_0, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.w_p1_cons_lock_0, 1)
                aie.dma_bd(self.w_buffer_1, offset = 2 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_cons_lock_1, 1)
                aie.next_bd(block[chain_id + 1])
                
            chain_id += 3
            with block[chain_id]:
                aie.dma_start(aie.DMAChannelDir.MM2S, 3, dest = block[chain_id + 1], chain = block[chain_id + 3])

            with block[chain_id + 1]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_0, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[chain_id + 2])

            with block[chain_id + 2]:
                lock_acquire(self.w_p1_cons_lock_1, 1)
                aie.dma_bd(self.w_buffer_1, offset = 3 * self.w_block_size, len = self.w_block_size)
                lock_release(self.w_p1_prod_lock, 1)
                aie.next_bd(block[chain_id + 1])

            chain_id += 3
            with block[chain_id]:
                aie.EndOp()
    
class q4_proj_engine:
    def __init__(self, 
        CTs: List[List[tile_with_location]], 
        MTs: List[tile_with_location], 
        ITs: List[ShimTile], 
        proj_core_stack_size: int,
        proj_core_link_with: str,
        ctx: mlir_context, 
        model_config: ModelConfig
    ):
        self.CTs = CTs
        self.MTs = MTs
        self.ITs = ITs
        self.proj_cores = []
        for col in range(len(CTs) // 2):
            cores_this_group = []
            for row in range(len(CTs[0]) // 2):
                comment_print(f"Initializing proj_core ({row}, {col})")
                cores_this_group.append(proj_combo(CTs[2 * col][row * 2 : row * 2 + 2], ctx, model_config, proj_core_stack_size, proj_core_link_with,
                                                   col_idx = 2 * col))
            
            for row in range(len(CTs[0]) // 2):
                comment_print(f"Initializing proj_core ({row}, {col})")
                cores_this_group.append(proj_combo(CTs[2 * col + 1][row * 2 : row * 2 + 2], ctx, model_config, proj_core_stack_size, proj_core_link_with,
                                                   col_idx = 2 * col + 1))
            self.proj_cores.append(cores_this_group)

        self.pure_assemble_mts = []
        self.pure_assemble_mts.append(
            proj_assemble_col(MTs[0], ctx, model_config)
        )
        self.pure_assemble_mts.append(
            proj_assemble_col(MTs[2], ctx, model_config)
        )
        self.pure_weight_mt = proj_assemble_weight_only(MTs[3], ctx, model_config)

        self.main_mt = proj_assemble_weight_x(MTs[1], ctx, model_config)
        
        
    def route(self):
        # connect IT to MT
        for i in range(len(self.MTs)):
            connect_port(self.ITs[i].MM2S[0], self.MTs[i].S2MM[4])
            connect_port(self.ITs[i].MM2S[1], self.MTs[i].S2MM[5])

        # connect MT weight to all CTs
        for i in range(len(self.MTs)):
            for j in range(len(self.CTs[i])):
                connect_port(self.MTs[i].MM2S[j], self.CTs[i][j].S2MM[1])

        # connect x broadcast to all CTs: one circuit, every core gets every
        # x chunk (as in Gemma4 and Qwen3.5 decoding; no packet ids, no arbiter)
        for i in range(len(self.CTs)):
            for j in range(len(self.CTs[i])):
                connect_port(self.main_mt.x_out, self.CTs[i][j].S2MM[0])

        # collect y from two columns to each pure assemble mt
        for i in range(len(self.proj_cores)):
            for j in range(len(self.proj_cores[i])):
                if j == 0:
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = True)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_aux, keep_pkt_header = True)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = True)
                else:
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)
                    connect_port(self.proj_cores[i][j].y_out, self.pure_assemble_mts[i].y_in[j], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

        # collect y from pure assemble mts to main mt
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_aux, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[0].y_out, self.main_mt.y_in[0], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = True)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)
        connect_port(self.pure_assemble_mts[1].y_out, self.main_mt.y_in[1], pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

         # connect weight from pure weight mt to main mt
    @property
    def X_in(self):
        return self.main_mt.x_in
    
    @property
    def Y_out(self):
        return self.main_mt.y_out
