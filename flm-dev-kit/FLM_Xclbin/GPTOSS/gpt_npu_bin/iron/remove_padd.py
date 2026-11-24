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


def remove_zero_pad_tile(tile:tile_with_location, ctx:mlir_context):
    
    ct = tile.tile
    ct_row = tile.row
    ct_col = tile.col

    # the function

    D= ctx.parameters["D"]
    D_ATTN_O_PADDED_MVM_SIZE = ctx.parameters["D_ATTN_O_PADDED_MVM_SIZE"]


    
    x_ATTN_O_padded_ty = ctx.data_tys["x_ATTN_O_padded_ty"]
    MVM_interleave_ty = ctx.data_tys["MVM_interleave_ty"]
    x_ty= ctx.data_tys["x_ty"]
    
    padded_x_in_prod_lock = aie.lock(ct, lock_id = 0,init= 1)
    padded_x_in_con_lock = aie.lock(ct, lock_id = 1, init=0)
    
    x_padded_in = aie.buffer(tile=ct, datatype=x_ATTN_O_padded_ty, name=f"padded_in_{ct_row}_{ct_col}")

    
    # buffer for repeat interleave for MVM operation
    mvm_x_in_buf= [
        aie.buffer(ct, datatype=MVM_interleave_ty),
        aie.buffer(ct, datatype=MVM_interleave_ty),
        aie.buffer(ct, datatype=MVM_interleave_ty),        
    ]
    mvm_x_in_prod_lock_0 = aie.lock(tile=ct, lock_id=4, init=4)
    mvm_x_in_con_lock_0 = aie.lock(tile=ct, lock_id=5, init=0)
    mvm_x_in_prod_lock_1 = aie.lock(tile=ct, lock_id=6, init=4)
    mvm_x_in_con_lock_1 = aie.lock(tile=ct, lock_id=7, init=0)
    mvm_x_in_prod_lock_2 = aie.lock(tile=ct, lock_id=8, init=4)
    mvm_x_in_con_lock_2 = aie.lock(tile=ct, lock_id=9, init=0)


    @aie.mem(ct)
    def mem_body(block):

        s2 = dma_start(DMAChannelDir.S2MM, 1,dest=block[1], chain=block[4] )
        with block[1]:
            use_lock(mvm_x_in_prod_lock_0, LockAction.AcquireGreaterEqual, value=4)
            dma_bd(mvm_x_in_buf[0])
            use_lock(mvm_x_in_con_lock_0, LockAction.Release, value=4)
            next_bd(block[2])
        with block[2]:
            use_lock(mvm_x_in_prod_lock_1, LockAction.AcquireGreaterEqual, value=4)
            dma_bd(mvm_x_in_buf[1])
            use_lock(mvm_x_in_con_lock_1, LockAction.Release, value=4)
            next_bd(block[3])
        with block[3]:
            use_lock(mvm_x_in_prod_lock_2, LockAction.AcquireGreaterEqual, value=4)
            dma_bd(mvm_x_in_buf[2])
            use_lock(mvm_x_in_con_lock_2, LockAction.Release, value=4)
            next_bd(block[1])
            
        with block[4]:
            s3 = dma_start(DMAChannelDir.MM2S, 1, dest=block[5], chain=block[17])
        with block[5]:
            use_lock(mvm_x_in_con_lock_0, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[0], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_0, LockAction.Release, value=1)
            next_bd(block[6])
        with block[6]:
            use_lock(mvm_x_in_con_lock_0, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[0], packet=(0,0))
            use_lock(mvm_x_in_prod_lock_0, LockAction.Release, value=1)
            next_bd(block[7])
        with block[7]:
            use_lock(mvm_x_in_con_lock_0, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[0], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_0, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            use_lock(mvm_x_in_con_lock_0, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[0], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_0, LockAction.Release, value=1)
            next_bd(block[9])
        with block[9]:
            use_lock(mvm_x_in_con_lock_1, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[1], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_1, LockAction.Release, value=1)
            next_bd(block[10])  
        with block[10]:
            use_lock(mvm_x_in_con_lock_1, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[1], packet=(0,0))           
            use_lock(mvm_x_in_prod_lock_1, LockAction.Release, value=1)
            next_bd(block[11])  
        with block[11]:
            use_lock(mvm_x_in_con_lock_1, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[1], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_1, LockAction.Release, value=1)
            next_bd(block[12])      
        with block[12]:
            use_lock(mvm_x_in_con_lock_1, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[1], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_1, LockAction.Release, value=1)
            next_bd(block[13])   
              
        with block[13]:
            use_lock(mvm_x_in_con_lock_2, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[2], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_2, LockAction.Release, value=1)
            next_bd(block[14])  
        with block[14]:
            use_lock(mvm_x_in_con_lock_2, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[2], packet=(0,0))           
            use_lock(mvm_x_in_prod_lock_2, LockAction.Release, value=1)
            next_bd(block[15])  
        with block[15]:
            use_lock(mvm_x_in_con_lock_2, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[2], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_2, LockAction.Release, value=1)
            next_bd(block[16])      
        with block[16]:
            use_lock(mvm_x_in_con_lock_2, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mvm_x_in_buf[2], packet=(0,0))            
            use_lock(mvm_x_in_prod_lock_2, LockAction.Release, value=1)
            next_bd(block[5])     
            
            
        with block[17]:
            EndOp()
    