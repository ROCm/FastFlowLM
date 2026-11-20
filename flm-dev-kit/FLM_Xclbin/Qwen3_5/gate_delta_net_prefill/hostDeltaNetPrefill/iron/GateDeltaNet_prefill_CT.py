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


import numpy.typing as npt

def setup_CT(
    row_idx:int, col_idx:int,
    stack_size:int,
    delta_net_prefill_function, zero_int_func,
    link_o_name:str,
    
    CT_Q_K_V_in_ty: npt.NDArray,
    CT_S_in_ty:npt.NDArray,
    CT_output_bf16_ty:npt.NDArray,
    ping_pong_ty:npt.NDArray,
    rtp_config_ty:npt.NDArray,
    ping_pong_rtp_len:int,  
    CT_L_size:int,
    actual_valid_qkv_data_size_per_L:int,
    padded_qkv_data_size_per_L:int
    
):
    
    
    assert row_idx >=2
    CT_core = tile(col_idx, row_idx)
            
    rtp_config_buffer_str = f"rtp_config_{row_idx}_{col_idx}"
    rtp_config_buffer = buffer(CT_core, datatype=rtp_config_ty, name=rtp_config_buffer_str, use_write_rtp=True)  
    rtp_sync_lock = lock(CT_core, lock_id=12, init=0)
    
    
    
    CT_S_IN_prod_lock_id = 0
    CT_S_IN_con_lock_id = 1
    CT_Q_prod_lock_id = 2
    CT_Q_con_lock_id = 3
    CT_K_prod_lock_id = 4
    CT_K_con_lock_id = 5
    CT_V_prod_lock_id = 6
    CT_V_con_lock_id = 7
    CT_output_prod_lock_id = 8
    CT_output_con_lock_id = 9
    
    CT_S_IN_prod_lock = lock(CT_core, lock_id=CT_S_IN_prod_lock_id, init=1)
    CT_S_IN_con_lock = lock(CT_core, lock_id=CT_S_IN_con_lock_id, init=0)
    
    CT_Q_prod_lock = lock(CT_core, lock_id=CT_Q_prod_lock_id, init=2)
    CT_Q_con_lock = lock(CT_core, lock_id=CT_Q_con_lock_id, init=0)
    
    CT_K_prod_lock = lock(CT_core, lock_id=CT_K_prod_lock_id, init=2)
    CT_K_con_lock = lock(CT_core, lock_id=CT_K_con_lock_id, init=0)
    
    CT_V_prod_lock = lock(CT_core, lock_id=CT_V_prod_lock_id, init=2)
    CT_V_con_lock = lock(CT_core, lock_id=CT_V_con_lock_id, init=0)
    
    CT_output_prod_lock = lock(CT_core, lock_id=CT_output_prod_lock_id, init=2)
    CT_output_con_lock = lock(CT_core, lock_id=CT_output_con_lock_id, init=0)
         

    CT_S_chunk_in_buffer = buffer(CT_core, CT_S_in_ty)

    CT_Q_buffer = [
        buffer(CT_core,CT_Q_K_V_in_ty),
        buffer(CT_core,CT_Q_K_V_in_ty)
    ]    
    CT_K_buffer = [
        buffer(CT_core, CT_Q_K_V_in_ty),
        buffer(CT_core, CT_Q_K_V_in_ty),
    ]
    CT_V_buffer = [
        buffer(CT_core, CT_Q_K_V_in_ty),
        buffer(CT_core, CT_Q_K_V_in_ty),
    ]
    
    CT_O_chunk_buffer = [
        buffer(CT_core, CT_output_bf16_ty),
        buffer(CT_core, CT_output_bf16_ty)
    ]

    ping_pong_buf = [
        buffer(CT_core, datatype=ping_pong_ty), # for input ping_pong
    ]

    
    
    
    @core(CT_core, link_o_name, stack_size=stack_size)
    def core_body():
        zero_int_func(ping_pong_buf[0], constant(ping_pong_rtp_len))
        for _ in range_(0xFFFFFFFF):
            
            #acquire the rtp lock
            use_lock(rtp_sync_lock, LockAction.AcquireGreaterEqual, value=1)
            
            delta_net_prefill_function(
                CT_S_chunk_in_buffer,
                CT_Q_buffer[0],CT_Q_buffer[1],
                CT_K_buffer[0], CT_K_buffer[1],
                CT_V_buffer[0], CT_V_buffer[1],
                CT_O_chunk_buffer[0], CT_O_chunk_buffer[1],
                constant(padded_qkv_data_size_per_L),
                constant( row_idx - 2), ping_pong_buf[0],
                rtp_config_buffer,
                constant(48+CT_S_IN_prod_lock_id), constant(48+CT_S_IN_con_lock_id),
                constant(48+CT_Q_prod_lock_id), constant(48+CT_Q_con_lock_id),
                constant(48+CT_K_prod_lock_id), constant(48+CT_K_con_lock_id),
                constant(48+CT_V_prod_lock_id), constant(48+CT_V_con_lock_id),
                constant(48+CT_output_prod_lock_id), constant(48+CT_output_con_lock_id)
            )

    
    assert padded_qkv_data_size_per_L >= actual_valid_qkv_data_size_per_L
    MT_to_CT_QKV_reorder = [
        (CT_L_size,padded_qkv_data_size_per_L ),
        (actual_valid_qkv_data_size_per_L, 1)
        
    ]
    @mem(CT_core)
    def CT_0_2_dma_body(block):
        # receive data_block buffer
        s0 = dma_start(DMAChannelDir.S2MM, 0,  dest=block[1], chain=block[3])
        with block[1]:
            use_lock(CT_S_IN_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_S_chunk_in_buffer)
            use_lock(CT_S_IN_con_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[3]:
            # receve vec_in
            s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[10])
        with block[4]:
            use_lock(CT_K_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_K_buffer[0], offset=0, len=CT_L_size*actual_valid_qkv_data_size_per_L,dimensions=MT_to_CT_QKV_reorder)
            use_lock(CT_K_con_lock, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            use_lock(CT_V_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_V_buffer[0], offset=0, len=CT_L_size*actual_valid_qkv_data_size_per_L,dimensions=MT_to_CT_QKV_reorder)
            use_lock(CT_V_con_lock, LockAction.Release, value=1)
            next_bd(block[6])
        with block[6]:
            use_lock(CT_Q_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_Q_buffer[0], offset=0, len=CT_L_size*actual_valid_qkv_data_size_per_L,dimensions=MT_to_CT_QKV_reorder)
            use_lock(CT_Q_con_lock, LockAction.Release, value=1)
            next_bd(block[7])
        with block[7]:
            use_lock(CT_K_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_K_buffer[1], offset=0, len=CT_L_size*actual_valid_qkv_data_size_per_L,dimensions=MT_to_CT_QKV_reorder)
            use_lock(CT_K_con_lock, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            use_lock(CT_V_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_V_buffer[1], offset=0, len=CT_L_size*actual_valid_qkv_data_size_per_L,dimensions=MT_to_CT_QKV_reorder)
            use_lock(CT_V_con_lock, LockAction.Release, value=1)
            next_bd(block[9])
        with block[9]:
            use_lock(CT_Q_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_Q_buffer[1], offset=0, len=CT_L_size*actual_valid_qkv_data_size_per_L,dimensions=MT_to_CT_QKV_reorder)
            use_lock(CT_Q_con_lock, LockAction.Release, value=1)
            next_bd(block[4])
        with block[10]:
            s2 = dma_start(DMAChannelDir.MM2S, 0, dest=block[11], chain=block[13])
        with block[11]:
            use_lock(CT_output_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_O_chunk_buffer[0])
            use_lock(CT_output_prod_lock, LockAction.Release, value=1)
            next_bd(block[12])
        with block[12]:
            use_lock(CT_output_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_O_chunk_buffer[1])
            use_lock(CT_output_prod_lock, LockAction.Release, value=1)
            next_bd(block[11])
        with block[13]:
            EndOp()       
            
    return CT_core, rtp_sync_lock, rtp_config_buffer_str