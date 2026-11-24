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






def setup_MT_configure(
    col:int,
    NUM_OF_CT:int,
    
    MT_q_ty:MemRefType,
    MT_k_v_ty:MemRefType,
    MT_o_ty:MemRefType,
    
    MT_q_len:int,
    MT_k_len:int,
    MT_v_len:int,
    MT_o_len:int,
    
    r:int,
    s:int,
    t:int,
    DH:int,
    LQ:int,
    LK:int,


    MT_o_len_per_CT:int,
    MT_q_len_per_CT:int,
    MT_k_v_len_per_CT:int
):

    assert NUM_OF_CT >=1
    assert MT_o_len_per_CT * NUM_OF_CT == MT_o_len
    assert MT_q_len_per_CT * NUM_OF_CT == MT_q_len
    assert MT_k_v_len_per_CT * NUM_OF_CT == MT_k_len + MT_v_len
    
    MT_k_len_per_CT = MT_k_v_len_per_CT//2
    MT_v_len_per_CT = MT_k_len_per_CT 

    
    row = 1  #fixed for MT
    
    MT_core = tile(col,row, allocation_scheme="basic-sequential")
        
    MT_q_buffer = [
        buffer(MT_core, datatype=MT_q_ty),
        buffer(MT_core, datatype=MT_q_ty)
    ]
    MT_k_v_buffer = [
        buffer(MT_core, datatype=MT_k_v_ty),
        buffer(MT_core, datatype=MT_k_v_ty)
    ]
    
    MT_o_buffer = [
        buffer(MT_core, datatype=MT_o_ty),
        buffer(MT_core, datatype=MT_o_ty)
    ]
    
    #NOTE: this is for single CT
    MT_q_prod_lock_ping = lock(MT_core, lock_id=0, init = NUM_OF_CT)
    MT_q_con_lock_ping = lock(MT_core, lock_id=1, init=0)
    MT_q_prod_lock_pong = lock(MT_core, lock_id=2, init=NUM_OF_CT)
    MT_q_con_lock_pong = lock(MT_core, lock_id=3, init=0)
    

    
    #NOTE: this is for single CT
    MT_o_prod_lock_ping = lock(MT_core, lock_id=6, init=NUM_OF_CT) # number of active CT
    MT_o_con_lock_ping = lock(MT_core, lock_id=7, init=0)
    MT_o_prod_lock_pong = lock(MT_core, lock_id=8, init=NUM_OF_CT)
    MT_o_con_lock_pong = lock(MT_core, lock_id=9, init=0)
    
    MT_k_v_prod_lock_ping = lock(MT_core, lock_id=10, init=NUM_OF_CT)
    MT_k_v_con_lock_ping = lock(MT_core, lock_id=11, init=0)
    
    MT_k_v_prod_lock_pong = lock(MT_core, lock_id=12, init=NUM_OF_CT)
    MT_k_v_con_lock_pong = lock(MT_core, lock_id=13, init=0)
            
    
    shm_to_mt_q_reorder = [
        (LQ//r, r*DH), 
        (r, s), 
        (DH//s, r*s), 
        (s,1)
    ]
    shm_to_mt_k_v_reorder = [
        ((LK*2)//s  ,s*DH),  # times two because of k, v buffer
        (s,t),
        (DH//t, s*t),
        (t,1)
    ]
    CT_to_mt_O_reorder = [
        (LQ//r, r*DH),
        (DH//t, t),
        (r, DH),
        (t,1)
    ]
    assert s==t # because k, v share same buffer, so
    
    # because the MT_k_v_buffers holds value of 
    
    #[K_CT_2, K_CT_3, K_CT_4, K_CT_5, V_CT_2, V_CT_3, V_CT_4, V_CT_5]
    # so send one block of K, send skip the other blocks, to send the corresponding V
    assert MT_k_len_per_CT%512 == 0
    mt_k_v_to_per_CT = [
        (2, NUM_OF_CT*MT_k_len_per_CT),
        (MT_k_len_per_CT//512, 512),
        (512, 1)   
    ]
    # receive: DMA 0: Q
    # receive: DMA 1: K, V
    # receive: DMA 2: O of CT_0_2
    # receive: DMA 3: O of CT_0_3
    # receive: DMA 4: O of CT_0_4
    # receive: DMA 5: O of CT_0_5
    
    # send: DMA 0: All O
    # send: DMA 2: Q, K, V of CT_0_2
    # send: DMA 3: Q, K, V of CT_0_3
    # send: DMA 4: Q, K, V of CT_0_4
    # send: DMA 5: Q, K, V of CT_0_5
    
    assert NUM_OF_CT == 4
    
    @memtile_dma(MT_core)
    def memtile_0_1_dma_body(block):
        s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        with block[1]: # receive the q block from shimtile for all CTs in the column
            use_lock(MT_q_prod_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_q_buffer[0], offset=0, len= MT_q_len, dimensions=shm_to_mt_q_reorder)
            use_lock(MT_q_con_lock_ping, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[2])
        with block[2]:
            use_lock(MT_q_prod_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_q_buffer[1], offset=0, len= MT_q_len, dimensions=shm_to_mt_q_reorder)
            use_lock(MT_q_con_lock_pong, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[1])
        with block[3]:
            s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[6])
        with block[4]:
            use_lock(MT_k_v_prod_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_k_v_buffer[0], offset=0, len=MT_k_len+MT_v_len, dimensions=shm_to_mt_k_v_reorder)
            use_lock(MT_k_v_con_lock_ping, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[5])
        with block[5]:
            use_lock(MT_k_v_prod_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_k_v_buffer[1], offset=0, len=MT_k_len+MT_v_len, dimensions=shm_to_mt_k_v_reorder)
            use_lock(MT_k_v_con_lock_pong, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[4])    
        with block[6]:
            # receive the o buffers from CT_0_2
            s2 = dma_start(DMAChannelDir.S2MM, 2, dest=block[7], chain=block[9])
        with block[7]:
            use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[0], offset=0, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[1], offset=0, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
            next_bd(block[7])
        with block[9]:
            # receive o buffer from CT_0_3
            s2 =dma_start(DMAChannelDir.S2MM, 3, dest=block[10], chain=block[12])
        with block[10]:
            use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[0], offset=MT_o_len_per_CT, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
            next_bd(block[11])
        with block[11]:
            use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[1], offset=MT_o_len_per_CT, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
            next_bd(block[10])
        with block[12]:
            # receive O buffer form cT_0_4
            s2 =dma_start(DMAChannelDir.S2MM, 4, dest=block[13], chain=block[15])
        with block[13]:
            use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[0], offset=MT_o_len_per_CT*2, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
            next_bd(block[14])
        with block[14]:
            use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[1], offset=MT_o_len_per_CT*2, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
            next_bd(block[13])
        with block[15]:
            # receive O buffer from CT_0_5            
            s2 =dma_start(DMAChannelDir.S2MM, 5, dest=block[16], chain=block[18])
        with block[16]:
            use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[0], offset=MT_o_len_per_CT*3, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
            next_bd(block[17])
        with block[17]:            
            use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_o_buffer[1], offset=MT_o_len_per_CT*3, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
            use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
            next_bd(block[16])                    
        with block[18]:
            # send O to shimtile as final output
            s3 = dma_start(DMAChannelDir.MM2S, channel_index=0, dest=block[19], chain=block[21])
        with block[19]:
            use_lock(MT_o_con_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_o_buffer[0], offset=0, len=MT_o_len)
            use_lock(MT_o_prod_lock_ping, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[20])
        with block[20]:
            use_lock(MT_o_con_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_o_buffer[1], offset=0, len=MT_o_len)
            use_lock(MT_o_prod_lock_pong, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[19])
        with block[21]:
            # Send Q, K, V to CT_0_2
            s4 = dma_start(DMAChannelDir.MM2S, channel_index=2, dest=block[22], chain=block[26])
        with block[22]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=0, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[23])
        with block[23]:
            use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[0], offset=0, len=MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_ping, LockAction.Release,value=1)
            next_bd(block[24])
        with block[24]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=0, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[25])
        with block[25]:
            use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[1], offset=0, len= MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[22])
        with block[26]:
            s4 = dma_start(DMAChannelDir.MM2S, channel_index=3, dest=block[27], chain=block[31])
        with block[27]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[28])
        with block[28]:
            use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[0], offset=MT_k_len_per_CT, len=MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_ping, LockAction.Release,value=1)
            next_bd(block[29])
        with block[29]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[30])
        with block[30]:
            use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[1], offset=MT_k_len_per_CT, len= MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[27])            
        with block[31]:
            s4 = dma_start(DMAChannelDir.MM2S, channel_index=4, dest=block[32], chain=block[36])
        with block[32]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=2*MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[33])
        with block[33]:
            use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[0], offset=2*MT_k_len_per_CT, len=MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_ping, LockAction.Release,value=1)
            next_bd(block[34])
        with block[34]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=2*MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[35])
        with block[35]:
            use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[1], offset=2*MT_k_len_per_CT, len= MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[32])                        
        with block[36]:
            s4 = dma_start(DMAChannelDir.MM2S, channel_index=5,dest=block[37], chain=block[41])
        with block[37]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=3*MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[38])
        with block[38]:
            use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[0], offset=3*MT_k_len_per_CT, len=MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_ping, LockAction.Release,value=1)
            next_bd(block[39])
        with block[39]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=3*MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[40])
        with block[40]:
            use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_k_v_buffer[1], offset=3*MT_k_len_per_CT, len= MT_k_v_len_per_CT, dimensions=mt_k_v_to_per_CT)
            use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[37])
        with block[41]:
            EndOp()                             

        
    return MT_core
