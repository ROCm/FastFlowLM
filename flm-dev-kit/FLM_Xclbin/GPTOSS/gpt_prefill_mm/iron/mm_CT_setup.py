# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from re import L
from threading import stack_size
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.dialects.ext.arith import constant, index_cast
from sympy import use


from aie.ir import *
from aie.ir import MemRefType, IndexType
from aie.dialects import arith, memref
from aie.dialects.memref import AllocaScopeOp

from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.dialects.memref import alloc, store, alloca
from aie.extras import types as T

from aie.dialects.aiex import control_packet
from ctypes import addressof, c_size_t
from aie.extras.dialects.ext.memref import (
    MemRef,
    store as memref_store,
    load as memref_load,
    global_ as memref_global
)
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent

import numpy as np
import numpy.typing as npt


def define_CT_for_MM(stack_size:int, col:int, row:int,
                     TOTAL_CT_ROW:int, TOTAL_CT_COL:int,
                     CT_bias_ty:npt.NDArray,
                     CT_a_ty:npt.NDArray, CT_b_ty:npt.NDArray,
                     CT_out_ty:npt.NDArray, CT_acc_ty:npt.NDArray,ping_pong_ty:npt.NDArray, rtp_ty:npt.NDArray,
                     CT_a_ty_len:int, CT_b_ty_len:int, CT_out_ty_len:int,
                     zero_int_1,mm_func_optimized, mm_func_b_col_optimized,
                     CT_link_o:str, 
                     IS_B_ROW_MAJOR :bool
                     ):
    
    assert row >= 2

   
    CT= tile(col, row,allocation_scheme="basic-sequential")
    rtp_buffer_name = f"CT_{row}_{col}_K_div_k"
    rtp_buffer = buffer(CT, datatype=rtp_ty, use_write_rtp=True, name=rtp_buffer_name, address=stack_size)
    rtp_sync_lock = lock(CT, lock_id=10, init=0)
    CT_bias_buffer = buffer(
        CT, datatype=CT_bias_ty, name=f"bias_buffer_{row}_{col}"
    )
    CT_b_buffer = [
        buffer(CT, datatype=CT_b_ty, name=f"in_b_0_{row}_{col}"),
        buffer(CT, datatype=CT_b_ty, name=f"in_b_1_{row}_{col}"),
    ]
    CT_b_prod_lock  = lock(CT, lock_id=2, init=2, sym_name=f"CT_0_2_b_prod_lock_{row}_{col}") # No ping pong for
    CT_b_con_lock = lock(CT, lock_id=3, init=0,   sym_name=f"CT_0_2_b_con_lock_{row}_{col}")
    
    CT_out_acc_buffer = [
        buffer(CT, datatype=CT_acc_ty, name=f"c_acc_{row}_{col}")
    ]
    
    CT_out_prod_lock  = lock(CT, lock_id=4, init=2, sym_name=f"CT_0_2_out_prod_lock_{row}_{col}")
    CT_out_con_lock = lock(CT, lock_id=5, init=0, sym_name=f"CT_0_2_out_con_lock_{row}_{col}")
    CT_out_buffer = [
        buffer(tile=CT, datatype=CT_out_ty, name=f"out_0_{row}_{col}"),  # ping pong
        buffer(tile=CT, datatype=CT_out_ty, name=f"out_1_{row}_{col}")
    ]
    # data allocations for CT_0_2
    CT_a_buffer = [
        buffer(CT, datatype=CT_a_ty, name=f"in_a_0_{row}_{col}"),
        buffer(CT, datatype=CT_a_ty, name=f"in_a_1_{row}_{col}"),
    ]
    CT_a_prod_lock = lock(CT, lock_id=0, init=2, sym_name=f"CT_0_2_a_prod_lock_{row}_{col}")
    CT_a_con_lock = lock(CT, lock_id=1, init=0, sym_name=f"CT_0_2_a_con_lock_{row}_{col}")
    
    ping_pong_buf = [
        buffer(CT, datatype=ping_pong_ty),
        buffer(CT, datatype=ping_pong_ty),] #NOTE: place of definition is important for memory alignment of instruction load/store


    #CT_logic

    @core(CT, CT_link_o, stack_size=stack_size )     
    def core_body():
        zero_int_1(ping_pong_buf[0])
        zero_int_1(ping_pong_buf[1])       

        for _ in range_(0xFFFFFFFF):
            # for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
            use_lock(rtp_sync_lock, LockAction.AcquireGreaterEqual, value=1)
            if IS_B_ROW_MAJOR:
                mm_func_optimized(
                    CT_a_buffer[0], CT_a_buffer[1],
                    CT_b_buffer[0], CT_b_buffer[1],
                    CT_bias_buffer,
                    CT_out_acc_buffer[0],
                    CT_out_buffer[0], CT_out_buffer[1],
                    ping_pong_buf[0],ping_pong_buf[1],
                    constant(row-2),constant(col), constant(TOTAL_CT_ROW), constant(TOTAL_CT_COL), # give CT's row and column index(all start from 0)
                    rtp_buffer,
                    constant(48+0), constant(48+1),
                    constant(48+2), constant(48+3), 
                    constant(48+4), constant(48+5)
                )
            else:
                mm_func_b_col_optimized(
                    CT_a_buffer[0], CT_a_buffer[1],
                    CT_b_buffer[0], CT_b_buffer[1], 
                    CT_bias_buffer,
                    CT_out_acc_buffer[0],
                    CT_out_buffer[0], CT_out_buffer[1],
                    ping_pong_buf[0], ping_pong_buf[1],
                    constant(row-2),constant(col), constant(TOTAL_CT_ROW), constant(TOTAL_CT_COL),
                    rtp_buffer,             
                    constant(48+0), constant(48+1),
                    constant(48+2), constant(48+3), 
                    constant(48+4), constant(48+5)
                )
            


    #CT Memtile logic: For now, don't do any data reordering on CT yet
    @mem(CT)
    def CT_0_2_dma_body(block):
        s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        with block[1]:  # receive from MT
            use_lock(CT_a_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_a_buffer[0], offset=0, len=CT_a_ty_len)
            use_lock(CT_a_con_lock, LockAction.Release, value=1)
            next_bd(block[2])
        with block[2]:
            use_lock(CT_a_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_a_buffer[1], offset=0, len=CT_a_ty_len)
            use_lock(CT_a_con_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[3]:
            s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
        with block[4]:
            use_lock(CT_b_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_b_buffer[0], offset=0, len=CT_b_ty_len)
            use_lock(CT_b_con_lock, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            use_lock(CT_b_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_b_buffer[1], offset=0, len=CT_b_ty_len)
            use_lock(CT_b_con_lock, LockAction.Release, value=1)
            next_bd(block[4])
        with block[6]:
            s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[9])
        with block[7]: #For now, do not do any reorder at CT yet
            use_lock(CT_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_out_buffer[0], offset=0, len=CT_out_ty_len)
            use_lock(CT_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            use_lock(CT_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_out_buffer[1], offset=0, len=CT_out_ty_len)
            use_lock(CT_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[7])                
        with block[9]:
            EndOp()
    

    
    return CT, rtp_buffer_name, rtp_sync_lock