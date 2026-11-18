# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from re import L
from struct import pack
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


from helper_func import MT_OPTIMIZED_logic_gen
def define_MT_for_MM(col:int,
                    MT_a_ty:npt.NDArray, MT_b_ty:npt.NDArray, out_MT_ty:npt.NDArray,
                    MT_a_ty_len:int, MT_b_ty_len:int, out_MT_size:int,
                    k_div_CT_k_max:int,
                    m_times_n :int,
                    r:int, s:int, t:int,
                    m:int, k:int, n:int,
                    IS_B_ROW_MAJOR:bool,
                    A_broadcast_pkt_id:int|None = None,
                    IS_BFLOAT16_OUT_TY:bool|None = None
                     ):
    assert IS_BFLOAT16_OUT_TY is not None
    assert out_MT_size == m_times_n*4 == 4*m*n
    # row should always be 1
    if k_div_CT_k_max >2:
        MT = tile(col=col, row = 1,allocation_scheme="basic-sequential")
    else:
        MT = tile(col=col, row = 1)
    row = 1
    # define the buffer at MT for 
    MT_a_buffer = [
        buffer(MT, datatype=MT_a_ty),
        buffer(MT, datatype=MT_a_ty),
    ]

    CT_max_k_size_m_128_n_64 = k//k_div_CT_k_max
    
    MT_a_prod_lock = lock(MT, lock_id=0, init=2*k_div_CT_k_max)
    MT_a_con_lock = lock(MT, lock_id=1, init=0)
    
    MT_b_buffer = [
        buffer(MT, datatype=MT_b_ty), 
        buffer(MT, datatype=MT_b_ty),
    ]

    # No ping pong for MT_b_buffer in optimized mode    
    MT_b_prod_lock = lock(MT, lock_id=2, init=2*k_div_CT_k_max)
    MT_b_con_lock = lock(MT, lock_id=3, init=0)
    
    if IS_BFLOAT16_OUT_TY:
        MT_out_buffer = [
            buffer(MT, datatype=out_MT_ty), 
            buffer(MT, datatype=out_MT_ty),
        ]

                                
        MT_prod_ping = lock(MT, lock_id=12, init=4) # ping pong for each CT
        MT_con_ping = lock(MT, lock_id=13, init=0)
        MT_prod_pong = lock(MT, lock_id=14, init=4)
        MT_con_pong = lock(MT, lock_id=15, init=0)
    else:
        # no ping pont for float type
        MT_out_buffer = [
            buffer(MT, datatype=out_MT_ty), 
        ]
           
        MT_prod_ping = lock(MT, lock_id=12, init=4) # ping pong for each CT
        MT_con_ping = lock(MT, lock_id=13, init=0)
        MT_prod_pong = None
        MT_con_pong = None
                                          
    CT_to_MT_dimension = [ (m//r, r*n),(n//t, t),(r,n),(t,1)]
    #Memtile logic
    SHM_to_MT_A_row_major_to_r_s_block = [(m//r, r*k), (r, s), ( k//s, r*s), (s, 1)]
    SHM_to_MT_B_row_major_to_s_t_block = [(k//s, s*n),(s, t),(n//t, s*t),(t,1)]
    SHM_to_MT_B_col_major_to_s_t_block = [(n//t, k*t),(t,s),(k//s, s*t),(s,1)]
    
    MT_to_A_row_major_K_CT_max = [            
                                    (k_div_CT_k_max, r*CT_max_k_size_m_128_n_64),
                                    (m//r, r*k),
                                    ( r*CT_max_k_size_m_128_n_64, 1)]
    MT_to_B_col_major_K_CT_max = [
            (k_div_CT_k_max, t*CT_max_k_size_m_128_n_64 ),
            (n//t,  k*t),
            (t*CT_max_k_size_m_128_n_64, 1)
    ]
    @memtile_dma(MT)    
    def Memtile_0_1_dma_body(block):
        s4 = dma_start(DMAChannelDir.S2MM, 2, dest=block[1], chain=block[3])
        with block[1]:
            use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_out_buffer[0], offset=0, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
            use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[2])
        with block[2]:
            
            if IS_BFLOAT16_OUT_TY:
                use_lock(MT_prod_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[1], offset=0, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
                use_lock(MT_con_pong, LockAction.Release, value=1)
            else:
                use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[0], offset=0, len=m_times_n,
                        dimensions=CT_to_MT_dimension)                
                use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[1])
        with block[3]:
            s5 = dma_start(DMAChannelDir.S2MM, 3, dest=block[4], chain=block[6])
        with block[4]:
            use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_out_buffer[0], offset=m_times_n, len=m_times_n,
                    dimensions=CT_to_MT_dimension)
            use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            
            if IS_BFLOAT16_OUT_TY:
                use_lock(MT_prod_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[1], offset=m_times_n, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
                use_lock(MT_con_pong, LockAction.Release, value=1)
            else:
                use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[0], offset=m_times_n, len=m_times_n,
                        dimensions=CT_to_MT_dimension)  
                use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[4])
        with block[6]:
            s6 = dma_start(DMAChannelDir.S2MM, 4, dest=block[7], chain=block[9])
        with block[7]:
            use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_out_buffer[0], offset=m_times_n*2, len=m_times_n,
                    dimensions=CT_to_MT_dimension)
            use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            
            if IS_BFLOAT16_OUT_TY:
                use_lock(MT_prod_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[1], offset=m_times_n*2, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
                use_lock(MT_con_pong, LockAction.Release, value=1)
            else:
                use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[0], offset=m_times_n*2, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
                use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[7])
        with block[9]:
            s7 = dma_start(DMAChannelDir.S2MM, 5, dest=block[10], chain=block[12])
        with block[10]:
            use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_out_buffer[0], offset=m_times_n*3, len=m_times_n,
                    dimensions=CT_to_MT_dimension)
            use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[11])
        with block[11]:
            
            
            if IS_BFLOAT16_OUT_TY:
                use_lock(MT_prod_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[1], offset=m_times_n*3, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
                use_lock(MT_con_pong, LockAction.Release, value=1)
            else:
                use_lock(MT_prod_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_out_buffer[0], offset=m_times_n*3, len=m_times_n,
                        dimensions=CT_to_MT_dimension)
                use_lock(MT_con_ping, LockAction.Release, value=1)
            next_bd(block[10])
        with block[12]:
            s8 = dma_start(DMAChannelDir.MM2S, 2, dest=block[13], chain=block[15])
        with block[13]:
            use_lock(MT_con_ping, LockAction.AcquireGreaterEqual, value=4)
            dma_bd(MT_out_buffer[0], offset=0, len=out_MT_size,
                    dimensions=[ (4*m,n), (n,1) ])
            use_lock(MT_prod_ping, LockAction.Release, value=4)
            next_bd(block[14])
        with block[14]:
                  
            if IS_BFLOAT16_OUT_TY:      
                use_lock(MT_con_pong, LockAction.AcquireGreaterEqual, value=4)
                dma_bd(MT_out_buffer[1], offset=0, len=out_MT_size,
                        dimensions=[  (4*m,n), (n,1)])
                use_lock(MT_prod_pong, LockAction.Release, value=4)
            else:
                use_lock(MT_con_ping, LockAction.AcquireGreaterEqual, value=4)
                dma_bd(MT_out_buffer[0], offset=0, len=out_MT_size,
                        dimensions=[  (4*m,n), (n,1)])                
                use_lock(MT_prod_ping, LockAction.Release, value=4)   
            next_bd(block[13])   
        with block[15]:
            s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[16], chain=block[18])
        # simply receive data from Shimtile
        with block[16]:
            use_lock(MT_a_prod_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            dma_bd(MT_a_buffer[0], offset=0, len=MT_a_ty_len, dimensions=SHM_to_MT_A_row_major_to_r_s_block)
            use_lock(MT_a_con_lock, LockAction.Release, value=k_div_CT_k_max)
            next_bd(block[17])
        with block[17]:
            use_lock(MT_a_prod_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            dma_bd(MT_a_buffer[1], offset=0, len=MT_a_ty_len, dimensions=SHM_to_MT_A_row_major_to_r_s_block)
            use_lock(MT_a_con_lock, LockAction.Release, value=k_div_CT_k_max)                
            next_bd(block[16])
        with block[18]:
            s2 = dma_start(DMAChannelDir.S2MM, 1, dest=block[19], chain=block[21])
        with block[19]:
            use_lock(MT_b_prod_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            if IS_B_ROW_MAJOR:
                dma_bd(MT_b_buffer[0], offset=0, len=MT_b_ty_len, dimensions=SHM_to_MT_B_row_major_to_s_t_block)
            else:
                dma_bd(MT_b_buffer[0], offset=0, len=MT_b_ty_len, dimensions=SHM_to_MT_B_col_major_to_s_t_block)
            use_lock(MT_b_con_lock, LockAction.Release, value=k_div_CT_k_max)                
            next_bd(block[20])
        with block[20]:
            use_lock(MT_b_prod_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            if IS_B_ROW_MAJOR:
                dma_bd(MT_b_buffer[1], offset=0, len=MT_b_ty_len, dimensions=SHM_to_MT_B_row_major_to_s_t_block)
            else:
                dma_bd(MT_b_buffer[1], offset=0, len=MT_b_ty_len, dimensions=SHM_to_MT_B_col_major_to_s_t_block)  
            use_lock(MT_b_con_lock, LockAction.Release, value=k_div_CT_k_max)                
            next_bd(block[19])            
        with block[21]:
            dma_start(DMAChannelDir.MM2S, 0, dest=block[22], chain=block[24])
        with block[22]:
            # Send A ping
            use_lock(MT_a_con_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            dma_bd(MT_a_buffer[0], offset=0, len=MT_a_ty_len, dimensions=MT_to_A_row_major_K_CT_max)
            use_lock(MT_a_prod_lock, LockAction.Release, value=k_div_CT_k_max)
            next_bd(block[23])
            
        with block[23]:
            use_lock(MT_a_con_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            dma_bd(MT_a_buffer[1], offset=0, len=MT_a_ty_len, dimensions=MT_to_A_row_major_K_CT_max )
            use_lock(MT_a_prod_lock, LockAction.Release, value=k_div_CT_k_max)
            next_bd(block[22])
        with block[24]:
            dma_start(DMAChannelDir.MM2S, 1, dest=block[25], chain=block[27])
        with block[25]:
            use_lock(MT_b_con_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            if IS_B_ROW_MAJOR:
                dma_bd(MT_b_buffer[0], offset=0, len=MT_b_ty_len)
            else:
                dma_bd(MT_b_buffer[0], offset=0, len=MT_b_ty_len, dimensions=MT_to_B_col_major_K_CT_max)
            use_lock(MT_b_prod_lock, LockAction.Release, value=k_div_CT_k_max)           
            next_bd(block[26])
        with block[26]:
            use_lock(MT_b_con_lock, LockAction.AcquireGreaterEqual, value=k_div_CT_k_max)
            if IS_B_ROW_MAJOR:
                dma_bd(MT_b_buffer[1], offset=0, len=MT_b_ty_len)
            else:
                dma_bd(MT_b_buffer[1], offset=0, len=MT_b_ty_len, dimensions=MT_to_B_col_major_K_CT_max)
            use_lock(MT_b_prod_lock, LockAction.Release, value=k_div_CT_k_max)
            next_bd(block[25])
        with block[27]:
            EndOp()               
    return MT