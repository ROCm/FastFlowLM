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
    MT_KV_repeat:int,
    

    MT_o_len_per_CT:int,
    MT_q_len_per_CT:int
):

    assert NUM_OF_CT  == 4
    assert MT_o_len_per_CT * NUM_OF_CT == MT_o_len
    assert MT_q_len_per_CT * NUM_OF_CT == MT_q_len

    row = 1  #fixed for MT
    
    MT_core = tile(col,row, allocation_scheme="basic-sequential")
        
        
        
        
    
    #NOTE:
    # we might run out of space, if that is the case, try not doing ping-pong for O buffer instead
    output_ping_pong = True
    if( ( MT_o_len *2 + MT_q_len *2 + (MT_k_len + MT_v_len)*2 ) * 2  >  512*1024 ):  # whole size times two because of bfloat16
        output_ping_pong = False            
        
    elif MT_KV_repeat == 8:
        # run out of bd chain 
        output_ping_pong = False
    MT_q_buffer = [
        buffer(MT_core, datatype=MT_q_ty),
        buffer(MT_core, datatype=MT_q_ty)
    ]
    MT_k_v_buffer = [
        buffer(MT_core, datatype=MT_k_v_ty),
        buffer(MT_core, datatype=MT_k_v_ty)
    ]
    if output_ping_pong:
        MT_o_buffer = [
            buffer(MT_core, datatype=MT_o_ty),
            buffer(MT_core, datatype=MT_o_ty)
        ]
    else:
        MT_o_buffer = [
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
    if output_ping_pong:
        MT_o_prod_lock_pong = lock(MT_core, lock_id=8, init=NUM_OF_CT)
        MT_o_con_lock_pong = lock(MT_core, lock_id=9, init=0)
    else:
        MT_o_prod_lock_pong = None
        MT_o_con_lock_pong = None
    
    MT_k_v_prod_lock_ping = lock(MT_core, lock_id=10, init=MT_KV_repeat)
    MT_k_v_con_lock_ping = lock(MT_core, lock_id=11, init=0)
    
    MT_k_v_prod_lock_pong = lock(MT_core, lock_id=12, init=MT_KV_repeat)
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
    @memtile_dma(MT_core)
    def memtile_0_1_dma_body(block):
        s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        with block[1]: # receive the q block from shimtile for all CTs in the column
            use_lock(MT_q_prod_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_q_buffer[0], offset=0, len= MT_q_len, dimensions=shm_to_mt_q_reorder, bd_id= 0)
            use_lock(MT_q_con_lock_ping, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[2])

        with block[2]:
            use_lock(MT_q_prod_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
            dma_bd(MT_q_buffer[1], offset=0, len= MT_q_len, dimensions=shm_to_mt_q_reorder, bd_id= 1)
            use_lock(MT_q_con_lock_pong, LockAction.Release, value=NUM_OF_CT)
            next_bd(block[1])

        with block[3]:
            # send the q to CT_0_2
            s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
        with block[4]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=0, len=MT_q_len_per_CT,bd_id=2)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=0, len=MT_q_len_per_CT,bd_id=3)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[4])
            
        with block[6]:
            s2 = dma_start(DMAChannelDir.MM2S, 1, dest=block[7], chain=block[9])
        with block[7]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=MT_q_len_per_CT, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[7])
        
        with block[9]:
            s2 = dma_start(DMAChannelDir.MM2S, 2, dest=block[10], chain=block[12])
        with block[10]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=MT_q_len_per_CT*2, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[11])
        with block[11]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=MT_q_len_per_CT*2, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)
            next_bd(block[10])
            
        with block[12]:
            s2 = dma_start(DMAChannelDir.MM2S, channel_index=3, dest=block[13], chain=block[15])
        with block[13]:
            use_lock(MT_q_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[0], offset=MT_q_len_per_CT*3, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_ping, LockAction.Release, value=1)
            next_bd(block[14])
        with block[14]:
            use_lock(MT_q_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_q_buffer[1], offset=MT_q_len_per_CT*3, len=MT_q_len_per_CT)
            use_lock(MT_q_prod_lock_pong, LockAction.Release, value=1)            
            next_bd(block[13])

        with block[15]:
            #TODO: for now, assert no k/v repeat internall

            # receive k&V from the shimtile
            s4 = dma_start(DMAChannelDir.S2MM, channel_index=5, dest=block[16], chain=block[18] )
        with block[16]:
            use_lock(MT_k_v_prod_lock_ping, LockAction.AcquireGreaterEqual, value=MT_KV_repeat)
            dma_bd(MT_k_v_buffer[0], offset=0, len=MT_v_len+MT_k_len, dimensions=shm_to_mt_k_v_reorder)
            use_lock(MT_k_v_con_lock_ping, LockAction.Release, value=MT_KV_repeat)
            next_bd(block[17])
        with block[17]:
            use_lock(MT_k_v_prod_lock_pong, LockAction.AcquireGreaterEqual, value=MT_KV_repeat)
            dma_bd(MT_k_v_buffer[1], offset=0, len=MT_v_len+MT_k_len, dimensions=shm_to_mt_k_v_reorder )
            use_lock(MT_k_v_con_lock_pong, LockAction.Release, value=MT_KV_repeat)
            next_bd(block[16]) 
        
        if output_ping_pong:
            with block[18]:
                # receive O from CT_0_2
                s2 = dma_start(DMAChannelDir.S2MM, channel_index=1,dest=block[19], chain=block[21])
            with block[19]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = 0, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
                next_bd(block[20])
            with block[20]:
                use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[1], offset = 0, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
                next_bd(block[19])
                
            with block[21]:
                s2 = dma_start(DMAChannelDir.S2MM, 2,dest=block[22], chain=block[24] )
            with block[22]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = MT_o_len_per_CT, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
                next_bd(block[23])
            with block[23]:
                use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[1], offset = MT_o_len_per_CT, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
                next_bd(block[22])
                
            with block[24]:
                s2 = dma_start(DMAChannelDir.S2MM, channel_index=3, dest=block[25], chain=block[27])
            with block[25]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = MT_o_len_per_CT*2, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)            
                next_bd(block[26])
            with block[26]:
                use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[1], offset = MT_o_len_per_CT*2, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
                next_bd(block[25])
                
            with block[27]:
                s2 = dma_start(DMAChannelDir.S2MM, channel_index=4, dest=block[28], chain=block[30])
            with block[28]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = MT_o_len_per_CT*3, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)     
                next_bd(block[29])
            with block[29]:
                use_lock(MT_o_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[1], offset = MT_o_len_per_CT*3, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_pong, LockAction.Release, value=1)
                next_bd(block[28])
            with block[30]:
                # send out all the O to shimtile
                s3 = dma_start(DMAChannelDir.MM2S, channel_index=4, dest=block[31], chain=block[33])
            with block[31]:
                use_lock(MT_o_con_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
                dma_bd(MT_o_buffer[0], offset=0, len=MT_o_len)
                use_lock(MT_o_prod_lock_ping, LockAction.Release, value=NUM_OF_CT)
                next_bd(block[32])
            with block[32]:
                use_lock(MT_o_con_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
                dma_bd(MT_o_buffer[1], offset=0, len=MT_o_len)
                use_lock(MT_o_prod_lock_pong, LockAction.Release, value=NUM_OF_CT)
                next_bd(block[31])
            start_block_idx = 33
        else:
            with block[18]:
                # receive O from CT_0_2
                s2 = dma_start(DMAChannelDir.S2MM, channel_index=1,dest=block[19], chain=block[20])
            with block[19]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = 0, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
                next_bd(block[19])
            with block[20]:
                s2 = dma_start(DMAChannelDir.S2MM, 2,dest=block[21], chain=block[22] )
            with block[21]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = MT_o_len_per_CT, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)
                next_bd(block[21])
            with block[22]:
                s2 = dma_start(DMAChannelDir.S2MM, channel_index=3, dest=block[23], chain=block[24])
            with block[23]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = MT_o_len_per_CT*2, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)            
                next_bd(block[23])
                
            with block[24]:
                s2 = dma_start(DMAChannelDir.S2MM, channel_index=4, dest=block[25], chain=block[26])
            with block[25]:
                use_lock(MT_o_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_o_buffer[0], offset = MT_o_len_per_CT*3, len=MT_o_len_per_CT, dimensions=CT_to_mt_O_reorder)
                use_lock(MT_o_con_lock_ping, LockAction.Release, value=1)     
                next_bd(block[25])

            with block[26]:
                # send out all the O to shimtile
                s3 = dma_start(DMAChannelDir.MM2S, channel_index=4, dest=block[27], chain=block[28])
            with block[27]:
                use_lock(MT_o_con_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_OF_CT)
                dma_bd(MT_o_buffer[0], offset=0, len=MT_o_len)
                use_lock(MT_o_prod_lock_ping, LockAction.Release, value=NUM_OF_CT)
                next_bd(block[27])
   
            start_block_idx = 28
        

        end_block_idx = start_block_idx + MT_KV_repeat * 2 + 1  # multiple 2 for pingpong
        
        start_data_block_idx =start_block_idx +1
        
        
        with block[start_block_idx]:
            s5 = dma_start(DMAChannelDir.MM2S, channel_index=5, dest=block[start_data_block_idx], chain=block[end_block_idx])
        for i in range(start_data_block_idx, end_block_idx) :
            with block[i]:
                if (i - start_data_block_idx) < MT_KV_repeat:
                    use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(MT_k_v_buffer[0], offset=0, len=MT_v_len+MT_k_len)
                    use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
                    next_bd(block[i+1])
                else:
                    use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(MT_k_v_buffer[1], offset=0, len=MT_v_len+MT_k_len)
                    use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
                    
                    if (i +1) == end_block_idx:
                        next_bd(block[start_block_idx+1])
                    else:
                        next_bd(block[i+1])
        with block[end_block_idx]:
            EndOp()
                    
                    
         
        # if MT_KV_repeat == 1:     
        #     with block[15]:
        #         # broadcast kv to all CT
        #         s5 = dma_start(DMAChannelDir.MM2S, channel_index=5, dest=block[16], chain=block[18])
        #     with block[16]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[17])
        #     with block[17]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[16])
        #     with block[18]:
        #         EndOp()
        # elif MT_KV_repeat == 2:
        #     with block[15]:
        #         # broadcast kv to all CT
        #         s5 = dma_start(DMAChannelDir.MM2S, channel_index=5, dest=block[16], chain=block[20])
        #     with block[16]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[17])
        #     with block[17]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[18])
        #     with block[18]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[19])
        #     with block[19]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[16])

        #     with block[20]:
        #         EndOp()
        # elif MT_KV_repeat == 4:
        #     with block[15]:
        #         # broadcast kv to all CT
        #         s5 = dma_start(DMAChannelDir.MM2S, channel_index=5, dest=block[16], chain=block[24])
        #     with block[16]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[17])
        #     with block[17]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[18])
        #     with block[18]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[19])
        #     with block[19]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[20])                    
        #     with block[20]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[21])
        #     with block[21]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[22])
        #     with block[22]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[23])
        #     with block[23]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[16])
        #     with block[24]:
        #         EndOp()

        # elif MT_KV_repeat == 8:
        #     with block[15]:
        #         # broadcast kv to all CT
        #         s5 = dma_start(DMAChannelDir.MM2S, channel_index=5, dest=block[16], chain=block[32])
        #     with block[16]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[17])
        #     with block[17]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[18])
        #     with block[18]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[19])
        #     with block[19]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[20])
        #     with block[20]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[21])
        #     with block[21]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[22])
        #     with block[22]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[23])
        #     with block[23]:
        #         use_lock(MT_k_v_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[0])
        #         use_lock(MT_k_v_prod_lock_ping, LockAction.Release, value=1)
        #         next_bd(block[24])
        #     with block[24]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[25])
        #     with block[25]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[26])
        #     with block[26]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[27])
        #     with block[27]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[28])
        #     with block[28]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[29])
        #     with block[29]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[30])
        #     with block[30]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[31])
        #     with block[31]:
        #         use_lock(MT_k_v_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
        #         dma_bd(MT_k_v_buffer[1])
        #         use_lock(MT_k_v_prod_lock_pong, LockAction.Release, value=1)
        #         next_bd(block[16])
        #     with block[32]:
        #         EndOp()                    

        # else:
        #     raise ValueError("TODO support, customize")
        
    return MT_core
