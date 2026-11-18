# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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

import numpy.typing as npt
def setup_MT(
    col_idx:int,
    
    Q_K_V_size_bf16:int, 
    MT_L_size:int, 
    CT_L_size:int,
    MT_S_in_size_fp32:int,  # in float element count
    MT_Output_size_bf16:int, # in bf16 element count, should be equal to DV*MT_L_size, or MT_S_size*4//2, because S is float32 and O is bf16
    MT_qkv_alpha_beta_in_size_bf16:int, 
    CT_S_chunk_len_fp32:int, 
    
    MT_S_in_block_ty, # MT_S_size, in float
    MT_Output_ty,  # MT_S_Size in bfloat16 / MT_L_size*DV in bfloat16, use for output O / S to shimtile
    MT_qkv_alpha_beta_in_ty, # in bfloat16, use for receive q, k, v. alpha & beta from host
    NUM_CT_PER_COLUMN:int,
    DQ:int, DK:int, DV:int, 
    DV_chunk_per_CT:int 
):
    assert Q_K_V_size_bf16 == DQ+2 == DK+2 == DV+2
    assert MT_qkv_alpha_beta_in_size_bf16 == 3*MT_L_size * Q_K_V_size_bf16
    
    assert DV_chunk_per_CT * NUM_CT_PER_COLUMN == DV
    assert  MT_L_size * DV *2   == MT_Output_size_bf16*2
    MT_core = tile(col_idx, 1)
    

         
    # define the buffer 
    MT_S_IN_buffer = [
        buffer(MT_core, datatype=MT_S_in_block_ty, name=f"MT_S_IN_buffer_ping_col{col_idx}"),
        buffer(MT_core, datatype=MT_S_in_block_ty, name=f"MT_S_IN_buffer_pong_col{col_idx}")
    ]
    MT_S_IN_prod_lock_ping = lock(MT_core, lock_id=0, init = NUM_CT_PER_COLUMN, sym_name=f"MT_S_IN_prod_lock_ping_col{col_idx}")
    MT_S_IN_con_lock_ping = lock(MT_core, lock_id=1, init=0,sym_name=f"MT_S_IN_con_lock_ping_col{col_idx}")
    MT_S_IN_prod_lock_pong = lock(MT_core, lock_id=2, init=NUM_CT_PER_COLUMN, sym_name=f"MT_S_IN_prod_lock_pong_col{col_idx}")
    MT_S_IN_con_lock_pong = lock(MT_core, lock_id=3, init=0, sym_name=f"MT_S_IN_con_lock_pong_col{col_idx}")
    

    MT_Out_buffer = [
        buffer(MT_core, datatype=MT_Output_ty, name=f"MT_S_OUT_buffer_ping_col{col_idx}"),
        buffer(MT_core, datatype=MT_Output_ty, name=f"MT_S_OUT_buffer_pong_col{col_idx}")
    ]
    MT_S_Out_prod_lock_ping = lock(MT_core, lock_id=4, init=NUM_CT_PER_COLUMN, sym_name=f"MT_S_OUT_prod_lock_ping_col{col_idx}")
    MT_S_Out_con_lock_ping = lock(MT_core, lock_id=5, init=0, sym_name=f"MT_S_OUT_con_lock_ping_col{col_idx}")
    MT_S_Out_prod_lock_pong = lock(MT_core, lock_id=6, init=NUM_CT_PER_COLUMN, sym_name=f"MT_S_OUT_prod_lock_pong_col{col_idx}")
    MT_S_Out_con_lock_pong = lock(MT_core, lock_id=7, init=0, sym_name=f"MT_S_OUT_con_lock_pong_col{col_idx}")
    
    
    #NOTE: the order is k_alpha, v_beta, q_with padding
    MT_qkv_alpha_beta_in_buffer = [
        buffer(MT_core, datatype = MT_qkv_alpha_beta_in_ty, name=f"MT_qkv_alpha_beta_in_buffer_col_ping{col_idx}"),
        buffer(MT_core, datatype = MT_qkv_alpha_beta_in_ty, name=f"MT_qkv_alpha_beta_in_buffer_col_pong{col_idx}"),
    ]
    MT_alpha_prod_lock = lock(MT_core, lock_id=8, init=2, sym_name=f"MT_alpha_prod_lock_col{col_idx}")
    MT_beta_prod_lock = lock(MT_core, lock_id=9, init=0, sym_name=f"MT_beta_prod_lock_col{col_idx}")
    MT_k_prod_lock = lock(MT_core, lock_id=10, init=0, sym_name=f"MT_k_prod_lock_col{col_idx}")
    MT_v_prod_lock = lock(MT_core, lock_id=11, init=0, sym_name=f"MT_v_prod_lock_col{col_idx}")
    MT_q_prod_lock = lock(MT_core, lock_id=12, init=0, sym_name=f"MT_q_prod_lock_col{col_idx}")    
    MT_a_b_qkv_con_lock = lock(MT_core, lock_id=13, init=0, sym_name=f"MT_a_b_qkv_con_lock_col{col_idx}")    
    
    
    #NOTE: assume the S input from Shimtile is already in column major order
    assert MT_S_in_size_fp32 == DV*DK
    S_MT_TO_CT = [
        (DK, DV),
        (DV_chunk_per_CT, 1)
    ]
    # view the MT_qkv_alpha_beta_in_buffer as 3x (MT_L_sizex Q_K_V_size_bf16) where (MT_L_sizex Q_K_V_size_bf16) is in row-major order
    # and the order is K, V, Q
    alpha_beta_SHM_to_MT = [
        (MT_L_size, Q_K_V_size_bf16),
        (2, 1) # because of float==2 bfloat16
    ]
    
    q_SHM_to_MT = [
        (MT_L_size, Q_K_V_size_bf16),
        (DQ,1),
    ]    
    
    k_SHM_to_MT = [
        (MT_L_size, Q_K_V_size_bf16),
        (DK,1),
    ]    
    
    v_SHM_to_MT = [
        (MT_L_size, Q_K_V_size_bf16),
        (DV,1),
    ]    
    MT_to_CT_MT_qkv_alpha_beta_in_buffer = [
        
        (MT_L_size//CT_L_size,   CT_L_size*Q_K_V_size_bf16),
        (3, MT_L_size*Q_K_V_size_bf16),
        (CT_L_size,Q_K_V_size_bf16 ),
        (Q_K_V_size_bf16, 1)        
        # (CT_L_size*Q_K_V_size_bf16, 1)
        
    ]
    S_IN_port = 0
    QKV_alpha_beta_in_port = 5
    
    MT_to_CT_S_port_idx = [1,2,3,4]
    CT_to_MT_port_idx = [1,2,3,4]
    
    OUT_Port = 0
    QKV_alpha_beta_out_port = 5
    
    assert Q_K_V_size_bf16 == DQ+2 == DK+2 == DV+2
    
    @memtile_dma(MT_core)
    def memtile_dma_body(block):
        s0  = dma_start(DMAChannelDir.S2MM, S_IN_port, dest=block[1], chain=block[3])
        with block[1]: # receive S datablock
            use_lock(MT_S_IN_prod_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_CT_PER_COLUMN)
            dma_bd(MT_S_IN_buffer[0])
            use_lock(MT_S_IN_con_lock_ping, LockAction.Release, value=NUM_CT_PER_COLUMN)
            next_bd(block[2])
        with block[2]:
            use_lock(MT_S_IN_prod_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_CT_PER_COLUMN)
            dma_bd(MT_S_IN_buffer[1])
            use_lock(MT_S_IN_con_lock_pong, LockAction.Release, value=NUM_CT_PER_COLUMN)
            next_bd(block[1])
        with block[3]: # receive qkv alpha beta data block
            s1 = dma_start(DMAChannelDir.S2MM, QKV_alpha_beta_in_port, dest=block[4], chain=block[14])
        with block[4]:
            use_lock(MT_alpha_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[0], offset=DK, len= MT_L_size *2, dimensions=alpha_beta_SHM_to_MT)
            use_lock(MT_beta_prod_lock, LockAction.Release, value=1)
            next_bd(block[5])
        with block[5]:
            use_lock(MT_beta_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[0], offset = MT_L_size*Q_K_V_size_bf16 + DV, len=MT_L_size*2, dimensions=alpha_beta_SHM_to_MT  )
            use_lock(MT_k_prod_lock, LockAction.Release, value=1)
            next_bd(block[6])
        with block[6]:
            use_lock(MT_k_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[0], offset=MT_L_size*Q_K_V_size_bf16*2, len=MT_L_size*DQ, dimensions=q_SHM_to_MT)            

            use_lock(MT_v_prod_lock, LockAction.Release, value=1)
            next_bd(block[7])
        with block[7]:
            use_lock(MT_v_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[0], offset=0, len=MT_L_size* DK, dimensions=k_SHM_to_MT)            

            use_lock(MT_q_prod_lock, LockAction.Release, value=1)
            next_bd(block[8])
        with block[8]:
            use_lock(MT_q_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[0], offset=MT_L_size* Q_K_V_size_bf16, len=MT_L_size*DV, dimensions=v_SHM_to_MT)
            use_lock(MT_a_b_qkv_con_lock, LockAction.Release, value=1)
            next_bd(block[9])
        with block[9]:
            use_lock(MT_alpha_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[1], offset=DK, len= MT_L_size *2, dimensions=alpha_beta_SHM_to_MT)
            use_lock(MT_beta_prod_lock, LockAction.Release, value=1)
            next_bd(block[10])
        with block[10]:
            use_lock(MT_beta_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[1], offset = MT_L_size*Q_K_V_size_bf16 + DV, len=MT_L_size*2, dimensions=alpha_beta_SHM_to_MT)
            use_lock(MT_k_prod_lock, LockAction.Release, value=1)
            next_bd(block[11])
        with block[11]:
            use_lock(MT_k_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[1], offset=MT_L_size*Q_K_V_size_bf16*2, len=MT_L_size*DQ, dimensions=q_SHM_to_MT)            
            use_lock(MT_v_prod_lock, LockAction.Release, value=1)
            next_bd(block[12])
        with block[12]:
            use_lock(MT_v_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[1], offset=0, len=MT_L_size* DK, dimensions=k_SHM_to_MT)            
            use_lock(MT_q_prod_lock, LockAction.Release, value=1)
            next_bd(block[13])
        with block[13]:
            use_lock(MT_q_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[1], offset=MT_L_size* Q_K_V_size_bf16, len=MT_L_size*DV, dimensions=v_SHM_to_MT)
            use_lock(MT_a_b_qkv_con_lock, LockAction.Release, value=1)
            next_bd(block[4])
        
        
        # now, send qkv alpha beta to each CT through broadcast
        with block[14]:
            s1 = dma_start(DMAChannelDir.MM2S, QKV_alpha_beta_out_port, dest=block[15], chain=block[17])
        with block[15]:
            use_lock(MT_a_b_qkv_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[0],offset=0, len=MT_qkv_alpha_beta_in_size_bf16, dimensions=MT_to_CT_MT_qkv_alpha_beta_in_buffer)
            use_lock(MT_alpha_prod_lock, LockAction.Release, value=1)
            next_bd(block[16])
        with block[16]:
            use_lock(MT_a_b_qkv_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(MT_qkv_alpha_beta_in_buffer[1],offset=0, len=MT_qkv_alpha_beta_in_size_bf16, dimensions=MT_to_CT_MT_qkv_alpha_beta_in_buffer)
            use_lock(MT_alpha_prod_lock, LockAction.Release, value=1)
            next_bd(block[15])
            
        cur_BD = 17
        for row in range(NUM_CT_PER_COLUMN):
            with block[cur_BD]:
                cur_BD += 1
                s1 = dma_start(DMAChannelDir.MM2S, MT_to_CT_S_port_idx[row],
                               dest=block[cur_BD], chain=block[cur_BD+2]
                               )
            with block[cur_BD]:
                cur_BD = cur_BD+1
                use_lock(MT_S_IN_con_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_S_IN_buffer[0], offset=row*DV_chunk_per_CT, len=CT_S_chunk_len_fp32, dimensions=S_MT_TO_CT)
                use_lock(MT_S_IN_prod_lock_ping, LockAction.Release, value=1)
                next_bd(block[cur_BD])
            with block[cur_BD]:
                cur_BD+= 1
                use_lock(MT_S_IN_con_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_S_IN_buffer[1], offset=row*DV_chunk_per_CT, len=CT_S_chunk_len_fp32, dimensions=S_MT_TO_CT)
                use_lock(MT_S_IN_prod_lock_pong, LockAction.Release, value=1)
                next_bd(block[cur_BD-2])


        # now, for gather output from each CT

        # SHimtile will do the reorder, since the output could be MT_L_size*DV for O, of S_out
        assert MT_Output_size_bf16 % NUM_CT_PER_COLUMN == 0
        MT_output_size_per_CT_bf16 = MT_Output_size_bf16 // NUM_CT_PER_COLUMN
        for row in range(NUM_CT_PER_COLUMN):
            with block[cur_BD]:
                cur_BD += 1
                s1 = dma_start(DMAChannelDir.S2MM, channel_index=CT_to_MT_port_idx[row],
                               dest=block[cur_BD], chain=block[cur_BD+2]
                               )
            with block[cur_BD]:
                cur_BD += 1
                use_lock(MT_S_Out_prod_lock_ping, LockAction.AcquireGreaterEqual, value=1)
                dma_bd( MT_Out_buffer[0], offset = row*MT_output_size_per_CT_bf16, len=MT_output_size_per_CT_bf16)
                use_lock(MT_S_Out_con_lock_ping, LockAction.Release, value=1)
                next_bd(block[cur_BD])
            with block[cur_BD]:
                cur_BD +=1
                use_lock(MT_S_Out_prod_lock_pong, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(MT_Out_buffer[1], offset=row*MT_output_size_per_CT_bf16, len=MT_output_size_per_CT_bf16)
                use_lock(MT_S_Out_con_lock_pong, LockAction.Release, value=1)
                next_bd(block[cur_BD -2])

        with block[cur_BD]:
            cur_BD+= 1
            s3 = dma_start(DMAChannelDir.MM2S,OUT_Port, dest=block[cur_BD], chain=block[cur_BD+2] )
        with block[cur_BD]:
            cur_BD+= 1
            use_lock(MT_S_Out_con_lock_ping, LockAction.AcquireGreaterEqual, value=NUM_CT_PER_COLUMN)
            dma_bd(MT_Out_buffer[0])
            use_lock(MT_S_Out_prod_lock_ping, LockAction.Release, value=NUM_CT_PER_COLUMN)
            next_bd(block[cur_BD])
        with block[cur_BD]:
            cur_BD+= 1
            use_lock(MT_S_Out_con_lock_pong, LockAction.AcquireGreaterEqual, value=NUM_CT_PER_COLUMN)
            dma_bd(MT_Out_buffer[1])
            use_lock(MT_S_Out_prod_lock_pong, LockAction.Release, value=NUM_CT_PER_COLUMN)
            next_bd(block[cur_BD-2])
        with block[cur_BD]:
            EndOp()
    
    return MT_core