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


import utils



from aie.dialects.aiex import NpuDmaMemcpyNd

from dequant_column import setup_dequant_column

def MXFP4_dequant(arch="npu2", trace_size:int=64*1024 ):
    
    


    Q4NX_ROW_SIZE = 32
    Q4NX_COL_SIZE = 128
    Q4NX_COL_STRIDE = 16
    

    dtype_scalar_data_in = np.dtype[np.uint8]
     
    dtype_out = np.dtype[bfloat16]
    
    scalar_in_size = Q4NX_ROW_SIZE * (Q4NX_COL_SIZE//32) *4 # times 4 due to padding
    data_in_size = (Q4NX_ROW_SIZE* Q4NX_COL_SIZE) // 2
    
    dequant_block_in_size_byte =scalar_in_size + data_in_size 
    data_out_size = (Q4NX_ROW_SIZE* Q4NX_COL_SIZE)
    @device(AIEDevice.npu2)
    def device_body():

    
        
        quant_data_ty = np.ndarray[ (dequant_block_in_size_byte ,), dtype_scalar_data_in]
        dequant_out_ty = np.ndarray[ (data_out_size, ), dtype_out]
        rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
        ping_pong_buf_ty=np.ndarray[(16,), np.dtype[np.int32]]
        
        link_o_name = "dequant_MXFP4.o"
        dequant_mxfp4_block_func = external_func(
            f"dequant_Q4NX_block_32_128_16", inputs=[
                quant_data_ty, dequant_out_ty,rtp_ty
            ]   
        )
        zero_int32_func = external_func(
            name="zero_int32_1", inputs=[ping_pong_buf_ty]
        ) 

        #times 4 because of 4 row per column
        MT_quant_data_size = 4*dequant_block_in_size_byte
        MT_quant_data_ty = np.ndarray[ (MT_quant_data_size,), dtype_scalar_data_in]
        MT_dequant_data_size = 4*data_out_size
        MT_dequant_data_ty = np.ndarray[ (MT_dequant_data_size,), dtype_out]

        for col in range(8):
            setup_dequant_column(
                col=col, link_o_name=link_o_name,
                dequant_mxfp4_block_func=dequant_mxfp4_block_func, 
                zero_int32_func=zero_int32_func,
                quant_data_ty=quant_data_ty, dequant_out_ty=dequant_out_ty,
                ping_pong_buf_ty=ping_pong_buf_ty, rtp_ty=rtp_ty,
                MT_quant_data_size=MT_quant_data_size, MT_dequant_data_size=MT_dequant_data_size,
                MT_quant_data_ty=MT_quant_data_ty, MT_dequant_data_ty=MT_dequant_data_ty,
                Q4NX_ROW_SIZE = Q4NX_ROW_SIZE,
                Q4NX_COL_SIZE = Q4NX_COL_SIZE,
                Q4NX_COL_STRIDE = Q4NX_COL_STRIDE
                    
            )


with mlir_mod_ctx() as ctx:            
    MXFP4_dequant()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)