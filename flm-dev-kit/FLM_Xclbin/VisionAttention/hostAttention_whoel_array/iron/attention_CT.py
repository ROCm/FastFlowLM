# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from re import L
from threading import stack_size
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from pyparsing import col
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






def setup_CT_configure(
    row:int,col:int,
    stack_size:int,
    
    CT_lq_internal_len:int,
    CT_lk_lv_internal_len:int,
    CT_O_len:int,
    CT_S_pij_len:int,
    CT_lq_ty:MemRefType,
    CT_lk_lv_ty:MemRefType,
    CT_S_pij_ty:MemRefType,
    CT_O_ty:MemRefType,
    rtp_config_ty:MemRefType,
    ping_pong_ty:MemRefType,
    link_o_name:str,
    
    zero_int_array_func,
    attention_bf16_func
    
):
    
    assert row >=2 
    
    cur_CT = tile(col, row, allocation_scheme="basic-sequential")
        
    CT_lq_buffer= buffer(cur_CT, CT_lq_ty, name=f"CT_lq_buffer_{row}_{col}")
    CT_lk_lv_buffer = [
        buffer(cur_CT, CT_lk_lv_ty, name=f"CT_lk_lv_buffer_{row}_{col}_0"),
        buffer(cur_CT, CT_lk_lv_ty, name=f"CT_lk_lv_buffer_{row}_{col}_1"),
    ]
    
    CT_S_pij_buffer = buffer(cur_CT, CT_S_pij_ty, name=f"CT_S_pij_buffer_{row}_{col}")
    CT_O_buffer = buffer(cur_CT, CT_O_ty, name=f"CT_O_buffer_{row}_{col}")
    
    
    rtp_config_buffer_str = f"rtp_config_buffer_{row}_{col}"
    rtp_config_buffer = buffer(cur_CT, rtp_config_ty, name=rtp_config_buffer_str, use_write_rtp=True, address=stack_size)
    ping_pong_buffer = buffer(cur_CT, ping_pong_ty, name=f"ping_pong_buffer_{row}_{col}")
    
    
    CT_lq_buffer_prod_lock = lock(cur_CT, lock_id= 0, init=1, sym_name=f"CT_lq_buffer_prod_lock_{row}_{col}")
    CT_lq_buffer_cons_lock = lock(cur_CT, lock_id= 1, init=0, sym_name=f"CT_lq_buffer_cons_lock_{row}_{col}")
    
    CT_lk_lv_buffer_prod_lock = lock(cur_CT, lock_id= 2, init=2, sym_name=f"CT_lk_lv_buffer_prod_lock_{row}_{col}")
    CT_lk_lv_buffer_cons_lock = lock(cur_CT, lock_id= 3, init=0, sym_name=f"CT_lk_lv_buffer_cons_lock_{row}_{col}")

    CT_O_buffer_prod_lock = lock(cur_CT, lock_id= 4, init=1, sym_name=f"CT_O_buffer_prod_lock_{row}_{col}")
    CT_O_buffer_cons_lock = lock(cur_CT, lock_id= 5, init=0, sym_name=f"CT_O_buffer_cons_lock_{row}_{col}")
    
    rtp_sync_lock = lock(cur_CT, lock_id=10, init=0, sym_name=f"CT_rtp_sync_lock_{row}_{col}")
    CT_debug_s_buffer_prod_lock = lock(cur_CT, lock_id=11, init=1, sym_name=f"CT_debug_s_buffer_prod_lock_{row}_{col}")
    CT_debug_s_buffer_con_lock = lock(cur_CT, lock_id=12, init=0, sym_name=f"CT_debug_s_buffer_con_lock_{row}_{col}")
    
    @core(cur_CT, link_o_name, stack_size=stack_size)
    def core_body():
        zero_int_array_func(ping_pong_buffer, constant(32))
        for _ in range_(0xFFFFFFFF):
            use_lock(rtp_sync_lock, LockAction.AcquireGreaterEqual, value=1)
            attention_bf16_func(
                CT_lq_buffer,
                CT_lk_lv_buffer[0],CT_lk_lv_buffer[1],
                CT_S_pij_buffer,
                CT_O_buffer,
                rtp_config_buffer, ping_pong_buffer
            )
        
    @mem(cur_CT)
    def CT_0_2_dma_body(block):
        s0= dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[2])
        with block[1]: # receive Q block from MT
            use_lock(CT_lq_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_lq_buffer, offset=0, len=CT_lq_internal_len)
            use_lock(CT_lq_buffer_cons_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[3], chain=block[5])
        with block[3]: # receive K/V block from MT
            use_lock(CT_lk_lv_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_lk_lv_buffer[0], offset=0, len=CT_lk_lv_internal_len)
            use_lock(CT_lk_lv_buffer_cons_lock, LockAction.Release, value=1)    
            next_bd(block[4])
        with block[4]:
            use_lock(CT_lk_lv_buffer_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_lk_lv_buffer[1], offset=0, len=CT_lk_lv_internal_len)
            use_lock(CT_lk_lv_buffer_cons_lock, LockAction.Release, value=1)    
            next_bd(block[3])
        with block[5]:
            s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[6], chain=block[7] )
        with block[6]:
            use_lock(CT_O_buffer_cons_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_O_buffer, offset=0, len=CT_O_len//2) # NOTE: because CT_O_buffer is defined as float, but output is bfloat
            use_lock(CT_O_buffer_prod_lock, LockAction.Release, value=1)
            next_bd(block[6])
        with block[7]:
            # debug purpose for now
            s3 = dma_start(DMAChannelDir.MM2S, 1, dest=block[8], chain=block[9])
        with block[8]:
            use_lock(CT_debug_s_buffer_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_S_pij_buffer, offset=0, len=CT_S_pij_len)
            use_lock(CT_debug_s_buffer_prod_lock, LockAction.Release, value=1)
            next_bd(block[8])
        with block[9]:
            EndOp()
            
    return cur_CT, rtp_sync_lock, rtp_config_buffer_str
        