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

import numpy.typing as npt

from dequant_CT import setup_dequant_CT
from dequant_MT import MXFP4_dequant_MT



def setup_dequant_column(
    col:int,
    link_o_name:str, dequant_mxfp4_block_func, zero_int32_func,
    
    quant_data_ty:npt.NDArray, dequant_out_ty:npt.NDArray,
    rtp_ty:npt.NDArray,
    ping_pong_buf_ty:npt.NDArray,    
    MT_quant_data_size:int, MT_dequant_data_size:int, 
    MT_quant_data_ty:npt.NDArray, MT_dequant_data_ty:npt.NDArray,
    
    Q4NX_ROW_SIZE = 32,
    Q4NX_COL_SIZE = 128,
    Q4NX_COL_STRIDE = 16
    
):
    
    
    stack_size =1024*8
    CT_cols = []

    for i in range(4):
        CT_cols.append(setup_dequant_CT(
            col=col, row = 2+i,
            quant_data_ty=quant_data_ty, dequant_out_ty=dequant_out_ty,
            rtp_ty=rtp_ty, ping_pong_buf_ty=ping_pong_buf_ty,
            dequant_kernel=dequant_mxfp4_block_func,zero_int32_func=zero_int32_func, 
            link_o_name=link_o_name, stack_size=stack_size,
            Q4NX_ROW_SIZE=Q4NX_ROW_SIZE, Q4NX_COL_SIZE=Q4NX_COL_SIZE
        ))
    MT_0 =MXFP4_dequant_MT(
        col=col,
        MT_quant_data_size=MT_quant_data_size, MT_quant_data_ty=MT_quant_data_ty,
        MT_dequant_data_size=MT_dequant_data_size, MT_dequant_data_ty=MT_dequant_data_ty,
        Q4NX_ROW_SIZE=Q4NX_ROW_SIZE, Q4NX_COL_SIZE=Q4NX_COL_SIZE
    )         
    
    SHM_0 = tile(col,0)
    
    
    flow(SHM_0, WireBundle.DMA, 0,MT_0, WireBundle.DMA, 4)
    flow(SHM_0, WireBundle.DMA, 1,MT_0, WireBundle.DMA, 5)    
    flow(MT_0, WireBundle.DMA, 4,SHM_0, WireBundle.DMA, 0)
    flow(MT_0, WireBundle.DMA, 5,SHM_0, WireBundle.DMA, 1)        
    for idx, ct in enumerate(CT_cols):
        flow(MT_0, WireBundle.DMA, idx,
            ct, WireBundle.DMA, 0 
            )
        flow( ct, WireBundle.DMA, 0,
            MT_0, WireBundle.DMA, idx
                )

            