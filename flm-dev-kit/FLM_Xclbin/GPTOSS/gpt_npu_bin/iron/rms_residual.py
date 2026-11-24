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




from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref
from typing import Dict, Any, List, Optional
from aie.extras import types as T


from utils import mlir_context, tile_with_location, lock_acquire, lock_release


def rms_residual_tile(tile: tile_with_location, ctx: mlir_context):
    
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    rms_residual = ctx.funcs["rms_residual"]
    zero_int32_1 = ctx.funcs["zero_int32_1"]
    D= ctx.parameters["D"]
    D_PADDED_Q4NX_COL = ctx.parameters["D_PADDED_Q4NX_COL"]
    D_ATTN_O_PADDED_MVM_SIZE= ctx.parameters["D_ATTN_O_PADDED_MVM_SIZE"]
    PKT_ID_RMS_CORE_TO_MT = ctx.parameters["PKT_ID_RMS_CORE_TO_MT"]
    PKT_ID_RMS_CORE_TO_ROUTER = ctx.parameters["PKT_ID_RMS_CORE_TO_ROUTER"]
    
    x_ty= ctx.data_tys["x_ty"]
    rms_w_ty = ctx.data_tys["rms_w_ty"]
    rms_residual_out_padded_ty  = ctx.data_tys["rms_residual_out_padded_ty"]
    ping_pong_ty = ctx.data_tys["ping_pong_ty"]
    proj_rtp_ty = ctx.data_tys['proj_rtp_ty']
    
    #TODO: def need to be in sync with the functions!!!
    w_prod_lock = aie.lock(ct, lock_id = 0, init = 2)
    w_cons_lock = aie.lock(ct, lock_id = 1, init = 0)
    y_prod_lock = aie.lock(ct, lock_id = 2, init = 0)  # y output is 1x_hidden_size_padded
    y_cons_lock = aie.lock(ct, lock_id = 3, init = 0)
    x_prod_lock = aie.lock(ct, lock_id = 4, init = 1)
    x_cons_lock = aie.lock(ct, lock_id = 5, init = 0)
    x_final_prod_lock = aie.lock(ct, lock_id=6, init=2)
    x_final_con_lock = aie.lock(ct, lock_id=7, init=0)
    
    # hidden_size (1xD)
    x = aie.buffer(tile=ct,datatype=x_ty, name= f"x_{ct_row}_{ct_col}")
    
    w_buffers = [
        aie.buffer(ct, datatype=rms_w_ty, name=f"rms_w_ping_{ct_row}_{ct_col}"),
        aie.buffer(ct, datatype=rms_w_ty, name=f"rms_w_pong_{ct_row}_{ct_col}"),
    ]
    rms_out_buffer =aie.buffer(
        tile=ct, datatype=rms_residual_out_padded_ty, name=f"rms_out_{ct_row}_{ct_col}"
    )
    x_out_0 =aie.buffer(ct, datatype=x_ty, name=f"x_out_0__{ct_row}_{ct_col}")
    x_out_1 =aie.buffer(ct, datatype=x_ty, name=f"x_out_1__{ct_row}_{ct_col}")
    
    x_buf_0 = buffer(tile=ct, datatype=x_ty, name= f"x_buf{ct_row}_{ct_col}") #hold result for residual
    x_buf_temp =  buffer(tile=ct, datatype=x_ty, name= f"x_temp{ct_row}_{ct_col}") #hold temp result during calculation
    x_buf_temp2 =  buffer(tile=ct, datatype=x_ty, name= f"x_temp2{ct_row}_{ct_col}") #hold temp result during calculation
    
    ping_pong_buf = buffer(tile=ct, datatype=ping_pong_ty, name=f"ping_pong_{ct_row}_{ct_col}")
    x_out_ping_pong = buffer(tile=ct,datatype=ping_pong_ty, name=f"x_out_ping_pong_{ct_row}_{ct_col}")
    
    rms_rtp_lock = aie.lock(ct, lock_id=8, init=0)
    rms_rtp_buffer = aie.buffer(ct,proj_rtp_ty, f"rms_rtp_{ct_row}_{ct_col}",  use_write_rtp = True)
    
    @aie.core(ct, link_with="rms_residual.o", stack_size=4*1024)
    def core_body():
        zero_int32_1(ping_pong_buf)
        zero_int32_1(x_out_ping_pong)
        for _ in range_(0xFFFFFFFF):
            rms_residual(
                rms_out_buffer, x_out_0, x_out_1, x,
                w_buffers[0], w_buffers[1], x_buf_0, x_buf_temp, x_buf_temp2,
                constant(PKT_ID_RMS_CORE_TO_MT), constant(PKT_ID_RMS_CORE_TO_ROUTER),
                ping_pong_buf, x_out_ping_pong, rms_rtp_buffer
            )
    
    @aie.mem(ct)
    def mem_body(block):
        s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2] )
        with block[1]:
            use_lock(x_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(x, offset=0, len=D_ATTN_O_PADDED_MVM_SIZE)
            use_lock(x_cons_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            s1= dma_start(DMAChannelDir.S2MM, 1, dest=block[3], chain=block[5])
        with block[3]:
            use_lock(w_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(w_buffers[0], offset=0, len=D)
            use_lock(w_cons_lock, LockAction.Release, value=1)
            next_bd(block[4])
        with block[4]:
            use_lock(w_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(w_buffers[1], offset=0, len=D)
            use_lock(w_cons_lock, LockAction.Release, value=1)
            next_bd(block[3])
        with block[5]:
            s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[6], chain=block[7])
        with block[6]:
            use_lock(y_cons_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(rms_out_buffer, offset=14, len = 2+D_PADDED_Q4NX_COL ) # 2 byte for the packet_header
            use_lock(y_prod_lock, LockAction.Release, value=1)
            next_bd(block[6])
        with block[7]:
            s3 = dma_start(DMAChannelDir.MM2S, 1, dest=block[8], chain=block[10])
        with block[8]:
            use_lock(x_final_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(x_out_0, offset=0, len=D_ATTN_O_PADDED_MVM_SIZE) #no packet flow for this NOTE: change shape later, for now
            use_lock(x_final_prod_lock, LockAction.Release, value=1)
            next_bd(block[9])
        with block[9]:
            use_lock(x_final_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(x_out_1, offset=0, len=D_ATTN_O_PADDED_MVM_SIZE) #no packet flow for this NOTE: change shape later, for now
            use_lock(x_final_prod_lock, LockAction.Release, value=1)
            next_bd(block[8])
        with block[10]:
            EndOp()
        