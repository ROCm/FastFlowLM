# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import ValuesView
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





from utils import lock_acquire, lock_release
import numpy.typing as npt
from aie.dialects.aiex import NpuDmaMemcpyNd



def setup_dequant_CT(
    row:int, col:int,
    quant_data_ty: npt.NDArray,
    dequant_out_ty:npt.NDArray,
    rtp_ty:npt.NDArray,
    ping_pong_buf_ty:npt.NDArray,
    dequant_kernel,
    zero_int32_func,
    link_o_name:str,
    stack_size:int,
    Q4NX_ROW_SIZE,
    Q4NX_COL_SIZE
):
    
    CT_x = tile(col=col, row=row)
    
    # data allocations for CT_x
    CT_scalar_data_in_buffer_buffer = [
        buffer(CT_x, datatype=quant_data_ty, name=f"scalar_in_0_{row}_{col}"),
        buffer(CT_x, datatype=quant_data_ty, name=f"scalar_in_1_{row}_{col}")
    ]
    CT_scalar_data_in_prod_lock = lock(CT_x, lock_id=0, init=2, sym_name=f"CT_scalar_data_in_prod_lock_{row}_{col}")
    CT_scalar_data_in_con_lock = lock(CT_x, lock_id=1, init=0, sym_name=f"CT_scalar_data_in_con_lock_{row}_{col}")


    CT_out_buffer = [
        buffer(CT_x, datatype=dequant_out_ty, name=f"CT_out_0_{row}_{col}"),
        buffer(CT_x, datatype=dequant_out_ty, name=f"CT_out_1_{row}_{col}"),
    ]
    CT_out_prod_lock = lock(CT_x, lock_id=4, init=2, sym_name=f"CT_out_prod_lock_{row}_{col}")
    CT_out_con_lock = lock(CT_x, lock_id=5, init=0, sym_name=f"CT_out_con_lock_{row}_{col}")
    
    CT_rtp_lock = lock(CT_x, lock_id=6, init=0, sym_name=f"CT_rtp_lock_{row}_{col}")

    ping_pong_buf = buffer(CT_x, datatype=ping_pong_buf_ty)
    
    rtp_buffer_name: str = f"CT_{row}_{col}_rtp"
    # first one is num_of data block [Q4nx_Row x Q4nx_Column] to dequant
    # second one is the dequant mode
    rtp_buffer = buffer(tile=CT_x, datatype=rtp_ty, use_write_rtp=True, name=rtp_buffer_name)   
    
    
    @core(CT_x, link_o_name,  stack_size=stack_size )     
    def core_body():

        # assume we compute the result in a row base order
        # this need be in sync with runtimesequence on how it sends data to us
        zero_int32_func(ping_pong_buf)
        for _ in range_(0xFFFFFFFF):
    
            

            use_lock(CT_scalar_data_in_con_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(CT_out_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dequant_kernel(CT_scalar_data_in_buffer_buffer[0], CT_out_buffer[0], rtp_buffer)
            use_lock(CT_scalar_data_in_prod_lock, LockAction.Release, value=1)
            use_lock(CT_out_con_lock, LockAction.Release, value=1)                
    

            use_lock(CT_scalar_data_in_con_lock, LockAction.AcquireGreaterEqual, value=1)
            use_lock(CT_out_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dequant_kernel(CT_scalar_data_in_buffer_buffer[1], CT_out_buffer[1], rtp_buffer )
            use_lock(CT_scalar_data_in_prod_lock, LockAction.Release, value=1)
            use_lock(CT_out_con_lock, LockAction.Release, value=1)   
    #CT Memtile logic: For now, don't do any data reordering on CT yet
    
            # Although the output blocks is Q4NX_ROW_SIZEx Q4NX_COL_SIZE
            # but it is actually in column major order with Q4NX_COL_STRIDE
            # we want to reorder it back to pure column major order instead
    # output_reorder = [  (Q4NX_COL_SIZE, Q4NX_COL_STRIDE),
    #                   ( Q4NX_ROW_SIZE//Q4NX_COL_STRIDE, Q4NX_COL_SIZE*Q4NX_COL_STRIDE),
    #                   ( Q4NX_COL_STRIDE,1 )]
    assert Q4NX_ROW_SIZE == 32
    assert Q4NX_COL_SIZE == 128
    @mem(CT_x)
    def CT_x_dma_body(block):
        s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        with block[1]:  # receive from MT
            use_lock(CT_scalar_data_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_scalar_data_in_buffer_buffer[0])
            use_lock(CT_scalar_data_in_con_lock, LockAction.Release, value=1)
            next_bd(block[2])
        with block[2]:
            use_lock(CT_scalar_data_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_scalar_data_in_buffer_buffer[1])
            use_lock(CT_scalar_data_in_con_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[3]:
            s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
        with block[4]: 
            use_lock(CT_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buffer=CT_out_buffer[0])
            use_lock(CT_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            use_lock(CT_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(buffer=CT_out_buffer[1])
            use_lock(CT_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[4])    
            
        with block[6]:
            EndOp()
        
    return CT_x
                