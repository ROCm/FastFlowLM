# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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




def MT_OPTIMIZED_logic_gen(block, start_block_id,
    m:int, k:int, n:int, r:int, s:int, t:int,
    k_div_CT_k_max:int,
    MT_a_buffer:list[buffer], MT_b_buffer:list[buffer],
    MT_a_ty_len:int, MT_b_ty_len:int,
    MT_a_prod_lock, MT_a_con_lock, MT_b_prod_lock, MT_b_con_lock,
    IS_B_ROW_MAJOR:bool, A_broadcast_pkt_id:int|None=None
                           ):
    
    assert MT_a_ty_len % k_div_CT_k_max == 0, "MT_a_ty_len must be divisible by k_div_CT_k_max"
    assert MT_b_ty_len % k_div_CT_k_max == 0, "MT_b_ty_len must be divisible by k_div_CT_k_max"
    assert k % k_div_CT_k_max == 0, "k must be divisible by k_div_CT_k_max"
    MT_to_CT_A_dimension_div_k = [(m//r, r*k ), ( (k//k_div_CT_k_max)//s, s), (r, k), (s,1)  ]  # because CT maximum size is k=128, so send left/right of A matrix
    MT_to_CT_b_row_major_dimension = [ (k//s, s*n), (n//t, t), (s,n), (t,1) ]
    MT_to_CT_b_col_major_dimension_div_k = [ (n//t, t*k), ((k//k_div_CT_k_max)//s, s), (t,k), (s,1) ]

    
    with block[start_block_id]:
        dma_start(DMAChannelDir.MM2S, 0, dest=block[start_block_id+1], chain=block[start_block_id+1+ 2*k_div_CT_k_max]) # times 2 becuse of ping pong
        start_block_id+=1
    # A-ping
    A_start_block_id = start_block_id
    for idx, block_id in enumerate(range(start_block_id, start_block_id + k_div_CT_k_max)):
        with block[block_id]:
            use_lock(MT_a_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_a_buffer[0], offset= (k//k_div_CT_k_max)*idx, len=MT_a_ty_len//k_div_CT_k_max,
                   dimensions=MT_to_CT_A_dimension_div_k, packet=(0, A_broadcast_pkt_id) if A_broadcast_pkt_id is not None else None)
            use_lock(MT_a_prod_lock, LockAction.Release, value=1)
            next_bd(block[block_id+1])
    start_block_id+= k_div_CT_k_max
    # A-pong
    for idx, block_id in enumerate(range(start_block_id, start_block_id + k_div_CT_k_max)):
        with block[block_id]:
            use_lock(MT_a_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_a_buffer[1], offset= (k//k_div_CT_k_max)*idx, len=MT_a_ty_len//k_div_CT_k_max,
                   dimensions=MT_to_CT_A_dimension_div_k, packet=(0, A_broadcast_pkt_id) if A_broadcast_pkt_id is not None else None)
            use_lock(MT_a_prod_lock, LockAction.Release, value=1)
            if(block_id +1  ==start_block_id +k_div_CT_k_max ):
                next_bd(block[A_start_block_id])
            else:
                next_bd(block[block_id+1])
    start_block_id+= k_div_CT_k_max
    
    # for sending B

    if IS_B_ROW_MAJOR:
        with block[start_block_id]:
            dma_start(DMAChannelDir.MM2S, 1, dest=block[start_block_id+1], chain=block[start_block_id+3]) # times 2 becuse of ping pong
            start_block_id+=1
        with block[start_block_id]:
            use_lock(MT_b_con_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max) # because lock value is init to 2* this value at MT
            dma_bd(MT_b_buffer[0], offset=0, len=MT_b_ty_len, dimensions=MT_to_CT_b_row_major_dimension)
            use_lock(MT_b_prod_lock, LockAction.Release, value=k_div_CT_k_max)
            next_bd(block[start_block_id+1])
        start_block_id+=1
        with block[start_block_id]:
            use_lock(MT_b_con_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max) # because lock value is init to 2* this value at MT
            dma_bd(MT_b_buffer[1],offset=0, len=MT_b_ty_len, dimensions=MT_to_CT_b_row_major_dimension)
            use_lock(MT_b_prod_lock, LockAction.Release, value=k_div_CT_k_max)
            next_bd(block[start_block_id-1]) # go back to previous
            
        start_block_id+=1
        with block[start_block_id]:
            EndOp()
    else:
        with block[start_block_id]:
            dma_start(DMAChannelDir.MM2S, 1, dest=block[start_block_id+1], chain=block[start_block_id+1+ 2*k_div_CT_k_max]) # times 2 becuse of ping pong
            start_block_id+=1
        B_start_block_id =start_block_id
        # B-ping
        for idx, block_idx in enumerate(range(start_block_id, start_block_id+ k_div_CT_k_max )):
            with block[block_idx]:
                use_lock(MT_b_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_b_buffer[0], offset=(k//(k_div_CT_k_max) ) * idx  , len=MT_b_ty_len//k_div_CT_k_max,
                                dimensions=MT_to_CT_b_col_major_dimension_div_k)
                use_lock(MT_b_prod_lock, LockAction.Release, value=1)
                next_bd(block[block_idx+1])
        start_block_id+= k_div_CT_k_max
        
        # B_poing
        for idx, block_idx in enumerate(range(start_block_id, start_block_id+ k_div_CT_k_max )):
            with block[block_idx]:
                use_lock(MT_b_con_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_b_buffer[1], offset=(k//(k_div_CT_k_max) ) * idx  , len=MT_b_ty_len//k_div_CT_k_max,
                                dimensions=MT_to_CT_b_col_major_dimension_div_k)
                use_lock(MT_b_prod_lock, LockAction.Release, value=1)
                if (block_idx+1 == start_block_id + k_div_CT_k_max):
                    next_bd(block[B_start_block_id])
                else:
                    next_bd(block[block_idx+1])
        
        start_block_id+= k_div_CT_k_max
        with block[start_block_id]:
            EndOp()
    
    
    