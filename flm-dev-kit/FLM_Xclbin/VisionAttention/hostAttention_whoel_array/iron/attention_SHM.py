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


from attention_CT import setup_CT_configure
from attention_MT import setup_MT_configure

def get_shimtile_port_str(shim_index:int):
    


    cur_shim_Q_str =f"Q_SHM_{shim_index}" 
    cur_shim_KV_str = f"KV_SHM_{shim_index}"
    cur_shim_O_str = f"O_SHM_{shim_index}"
    cur_shim_S_str = f"S_SHM_{shim_index}"

    
    return cur_shim_Q_str, cur_shim_KV_str, cur_shim_O_str, cur_shim_S_str




# ALL SHIMTILE agree on common bds for Q, K, V O that it is using
# for Q, used bd 0, 8
# for k, used bd 1, 9
# for V, used bd 2, 10
# for O, used bd 3, 11
def get_Q_bd_id( ping_pong:int):
    return 0 if ping_pong ==0 else 8

def update_ping_pong( ping_pong:int):
    if ping_pong ==0:
        return 1
    else:
        return 0


def get_K_bd_id( ping_pong:int):
    return 1 if ping_pong ==0 else 9
def get_V_bd_id( ping_pong:int):
    return 2 if ping_pong ==0 else 10
def get_O_bd_id( ping_pong:int):
    return 3 if ping_pong ==0 else 11


def wait_DMA_queue_if_full( 
    cur_shim_idx:int,
    SHIM_queue_counter:list[int],
    cur_shim_str:str,
    
    max_queue_size = 2
):
    
    # NOTE: it will not wait if max_queue_size is not achieve
    
    if SHIM_queue_counter[cur_shim_idx] > max_queue_size:
        raise RuntimeError("SHIM DMA queue overflow")
    
    if SHIM_queue_counter[cur_shim_idx] == max_queue_size:
        npu_dma_wait(
            cur_shim_str
        )
        SHIM_queue_counter[cur_shim_idx] -= 1
        

def force_wait_DMA_queue(
    cur_shim_idx:int,
    SHIM_queue_counter:list[int],
    cur_shim_str:str
):
    if SHIM_queue_counter[cur_shim_idx] > 0:
        npu_dma_wait(
            cur_shim_str
        )
        SHIM_queue_counter[cur_shim_idx] -= 1

def increment_DMA_queue(
    cur_shim_idx:int,
    SHIM_queue_counter:list[int],
    max_queue_size =2
):
    if SHIM_queue_counter[cur_shim_idx] < max_queue_size:
        SHIM_queue_counter[cur_shim_idx] += 1
    else:
        raise RuntimeError("SHIM DMA queue overflow")


def process_SHM_Q_chunk_no_Q_repeat(
    NUM_OF_CT_per_column:int,
    MT_KV_repeat:int,
    DH:int,  # attention head dim

    NUM_OF_DH:int,  #number of attention _head
    DH_index_start:int,
    SHIM_LQ_start_row:int,
    SHIM_S_SEQ: int, # unpadded S_seq of k, v seq len
    SHIM_S_SEQ_PADDED:int,  # total size  of k, v seq len

    
    CU_column:list[list[int]],
    LQ_CU_chunk_size:int,
    LK_CU_chunk_size:int, 
    
    LQ_per_CT:int,
    LK_per_CT:int,
    
    
    SHIM_Q_queue_counter:list[int],
    SHIM_K_queue_counter:list[int],
    SHIM_V_queue_counter:list[int],
    SHIM_O_queue_counter:list[int],
    
    SHIM_Q_ping_pong:list[int],
    SHIM_K_ping_pong:list[int],
    SHIM_V_ping_pong:list[int],
    SHIM_O_ping_pong:list[int],
    
    Q, K, V, O
   
):
    
    
    assert SHIM_S_SEQ_PADDED%LK_CU_chunk_size ==0
    assert MT_KV_repeat == 1
    assert LK_CU_chunk_size == LK_per_CT


    
    
    
    
    LQ_per_column = LQ_CU_chunk_size//(len(CU_column[0]))  # should be same len in every CU_column
    assert NUM_OF_CT_per_column*LQ_per_CT == LQ_per_column
        
    
    for cu_idx in range(len(CU_column)):
    
        cur_CU_shimtile_cols = CU_column[cu_idx]


        DH_index = DH_index_start + cu_idx
        if DH_index >= NUM_OF_DH:
            break  # done 
        
        Shim_Q_offset =     SHIM_LQ_start_row *  (DH*NUM_OF_DH) + DH_index * DH
        SHIM_O_offset = Shim_Q_offset

        
        


        for idx, shim_col_idx in enumerate(cur_CU_shimtile_cols):
            cur_shim_Q_str, cur_shim_KV_str, cur_shim_O_str, cur_shim_S_str = get_shimtile_port_str(shim_col_idx)
            
            
            
            wait_DMA_queue_if_full(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_Q_queue_counter,
                cur_shim_str=cur_shim_Q_str
            )
            
            npu_dma_memcpy_nd(
                metadata=cur_shim_Q_str,
                bd_id= get_Q_bd_id( SHIM_Q_ping_pong[shim_col_idx]),
                mem=Q,
                offsets=[0,0,0,  Shim_Q_offset + LQ_per_column*idx*(DH*NUM_OF_DH)] , 
                sizes =[1, 1, LQ_per_column            , DH],
                strides=[0,0, DH*NUM_OF_DH, 1],
                issue_token=True
            )

            SHIM_Q_ping_pong[shim_col_idx]= update_ping_pong( SHIM_Q_ping_pong[shim_col_idx])
            increment_DMA_queue(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_Q_queue_counter
            )
            
            # now, receive the O 
            wait_DMA_queue_if_full(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_O_queue_counter,
                cur_shim_str=cur_shim_O_str
            )
            npu_dma_memcpy_nd(
                metadata=cur_shim_O_str,
                bd_id= get_O_bd_id(SHIM_O_ping_pong[shim_col_idx]),
                mem=O,
                offsets=[0,0,0, SHIM_O_offset + LQ_per_column*idx*(DH*NUM_OF_DH)],
                sizes =[1, 1, LQ_per_column            , DH],
                strides=[0,0, DH*NUM_OF_DH, 1],
                issue_token=True
            )
            SHIM_O_ping_pong[shim_col_idx] = update_ping_pong(SHIM_O_ping_pong[shim_col_idx])
            increment_DMA_queue(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_O_queue_counter
            )
        
        
    # only the first shimtile column in cur_CU_shimtile_cols will be used to send KV data

            
    for kv_chunk_idx in range( SHIM_S_SEQ_PADDED//LK_CU_chunk_size):

        for cu_idx in range(len(CU_column)):

            cur_CU_shimtile_cols = CU_column[cu_idx]
            
            DH_index = DH_index_start + cu_idx
            if DH_index >= NUM_OF_DH:
                break  # done 
                
                    
            shim_col_idx = cur_CU_shimtile_cols[0]
            _, cur_shim_KV_str, _, _ = get_shimtile_port_str(shim_col_idx)            
            
            
            SHIM_K_offset = DH_index * DH
            SHIM_V_offset = DH_index * DH            
            
            wait_DMA_queue_if_full(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_V_queue_counter,
                cur_shim_str=cur_shim_KV_str
            )                
                
            npu_dma_memcpy_nd(
                metadata=cur_shim_KV_str,
                bd_id=get_K_bd_id(SHIM_K_ping_pong[shim_col_idx]),
                mem=K,
                offsets=[0,0,0, SHIM_K_offset + kv_chunk_idx*LK_CU_chunk_size*(DH*NUM_OF_DH) ],  #TODO: assume now no need to jump stride!!!! 
                sizes = [1, 1, LK_CU_chunk_size,          DH],
                strides=[0,0,  DH*NUM_OF_DH, 1],
                issue_token=False
            )
            SHIM_K_ping_pong[shim_col_idx] = update_ping_pong(SHIM_K_ping_pong[shim_col_idx])

            
            npu_dma_memcpy_nd(
                metadata=cur_shim_KV_str,
                bd_id=get_V_bd_id(SHIM_V_ping_pong[shim_col_idx]),
                mem=V,
                offsets=[0,0,0, SHIM_V_offset + kv_chunk_idx*LK_CU_chunk_size*(DH*NUM_OF_DH) ],
                sizes = [1, 1, LK_CU_chunk_size,           DH],
                strides=[0,0,  DH*NUM_OF_DH,  1],                    
                issue_token=True
            )
            SHIM_V_ping_pong[shim_col_idx] = update_ping_pong(SHIM_V_ping_pong[shim_col_idx])
            increment_DMA_queue(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_V_queue_counter
            )        
    
    

def process_SHM_Q_chunk_Q_repeat(
    NUM_OF_CT_per_column:int,
    MT_KV_repeat:int,
    DH:int,  # attention head dim

    NUM_OF_DH:int,  #number of attention _head
    DH_index_start:int,
    SHIM_LQ_start_row:int,
    SHIM_S_SEQ: int, # unpadded S_seq of k, v seq len
    SHIM_S_SEQ_PADDED:int,  # total size  of k, v seq len

    
    CU_column:list[list[int]],
    LQ_CU_chunk_size:int,
    LK_CU_chunk_size:int, 
    
    LQ_per_CT:int,
    LK_per_CT:int,
    
    
    SHIM_Q_queue_counter:list[int],
    SHIM_K_queue_counter:list[int],
    SHIM_V_queue_counter:list[int],
    SHIM_O_queue_counter:list[int],
    
    SHIM_Q_ping_pong:list[int],
    SHIM_K_ping_pong:list[int],
    SHIM_V_ping_pong:list[int],
    SHIM_O_ping_pong:list[int],
    
    Q, K, V, O,
):
    
    
    assert SHIM_S_SEQ_PADDED%LK_CU_chunk_size ==0
    assert MT_KV_repeat >1
    assert LK_CU_chunk_size == LK_per_CT


    
    LQ_per_column = LQ_CU_chunk_size//(len(CU_column[0]))  # should be same len in every CU_column
    assert NUM_OF_CT_per_column*LQ_per_CT == LQ_per_column
    

    for cu_idx in range(len(CU_column)):
        cur_CU_shimtile_cols = CU_column[cu_idx]
        DH_index = DH_index_start + cu_idx
        if DH_index >= NUM_OF_DH:
            break  # done
        
        
        for idx, shim_col_idx in enumerate(cur_CU_shimtile_cols):
            cur_shim_Q_str, cur_shim_KV_str, cur_shim_O_str, cur_shim_S_str = get_shimtile_port_str(shim_col_idx)
            
            
            SHIM_O_offset =     SHIM_LQ_start_row *  (DH*NUM_OF_DH) + DH_index * DH

            
            # now, receive the O 
            wait_DMA_queue_if_full(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_O_queue_counter,
                cur_shim_str=cur_shim_O_str
            )
            npu_dma_memcpy_nd(
                metadata=cur_shim_O_str,
                bd_id= get_O_bd_id(SHIM_O_ping_pong[shim_col_idx]*(DH*NUM_OF_DH)),
                mem=O,
                offsets=[0,0,0, SHIM_O_offset + LQ_per_column*idx*(DH*NUM_OF_DH)],
                sizes =[1, 1, LQ_per_column            , DH],
                strides=[0,0, DH*NUM_OF_DH, 1],
                issue_token=True
            )
            SHIM_O_ping_pong[shim_col_idx] = update_ping_pong(SHIM_O_ping_pong[shim_col_idx])
            increment_DMA_queue(
                cur_shim_idx=shim_col_idx,
                SHIM_queue_counter=SHIM_O_queue_counter
            )
            
    
    for kv_chunk_idx in range( SHIM_S_SEQ_PADDED//LK_CU_chunk_size):
        
        for cu_idx in range(len(CU_column)):

            cur_CU_shimtile_cols = CU_column[cu_idx]
            
            DH_index = DH_index_start + cu_idx
            if DH_index >= NUM_OF_DH:
                break  # done
        
            Shim_Q_offset =     SHIM_LQ_start_row *  (DH*NUM_OF_DH) + DH_index * DH
            SHIM_K_offset = DH_index * DH
            SHIM_V_offset = DH_index * DH        
            for idx, shim_col_idx in enumerate(cur_CU_shimtile_cols):
                cur_shim_Q_str, cur_shim_KV_str, cur_shim_O_str, cur_shim_S_str = get_shimtile_port_str(shim_col_idx)
                            
                wait_DMA_queue_if_full(
                    cur_shim_idx=shim_col_idx,
                    SHIM_queue_counter=SHIM_Q_queue_counter,
                    cur_shim_str=cur_shim_Q_str
                )
                
                npu_dma_memcpy_nd(
                    metadata=cur_shim_Q_str,
                    bd_id= get_Q_bd_id( SHIM_Q_ping_pong[shim_col_idx]),
                    mem=Q,
                    offsets=[0,0,0,  Shim_Q_offset + LQ_per_column*idx*(DH*NUM_OF_DH)] , 
                    # sizes= [1,1,1, LQ_per_column * DH],
                    # strides=[0,0,0,1],
                    sizes =[1, 1, LQ_per_column            , DH],
                    strides=[0,0, DH*NUM_OF_DH, 1],
                    issue_token=True
                )

                SHIM_Q_ping_pong[shim_col_idx]= update_ping_pong( SHIM_Q_ping_pong[shim_col_idx])
                increment_DMA_queue(
                    cur_shim_idx=shim_col_idx,
                    SHIM_queue_counter=SHIM_Q_queue_counter
                )
                
            # only the first shimtile column in cur_CU_shimtile_cols will be used to send KV data

        
            shim_kv_col_idx = cur_CU_shimtile_cols[0]           
            _, shim_KV_str, _, _ = get_shimtile_port_str(shim_kv_col_idx)                       
    
            
            wait_DMA_queue_if_full(
                cur_shim_idx=shim_kv_col_idx,
                SHIM_queue_counter=SHIM_V_queue_counter,
                cur_shim_str=shim_KV_str
            )                
                
            npu_dma_memcpy_nd(
                metadata=shim_KV_str,
                bd_id=get_K_bd_id(SHIM_K_ping_pong[shim_kv_col_idx]),
                mem=K,
                offsets=[0,0,0, SHIM_K_offset + kv_chunk_idx*LK_CU_chunk_size*(DH*NUM_OF_DH) ],  #TODO: assume now no need to jump stride!!!! 

                sizes = [1, 1, LK_CU_chunk_size,          DH],
                strides=[0,0,  DH*NUM_OF_DH, 1],
                issue_token=False
            )
            SHIM_K_ping_pong[shim_kv_col_idx] = update_ping_pong(SHIM_K_ping_pong[shim_kv_col_idx])

            
            npu_dma_memcpy_nd(
                metadata=shim_KV_str,
                bd_id=get_V_bd_id(SHIM_V_ping_pong[shim_kv_col_idx]),
                mem=V,
                offsets=[0,0,0, SHIM_V_offset + kv_chunk_idx*LK_CU_chunk_size*(DH*NUM_OF_DH) ],
                sizes = [1, 1, LK_CU_chunk_size,           DH],
                strides=[0,0,  DH*NUM_OF_DH,  1],                    
                issue_token=True
            )
            SHIM_V_ping_pong[shim_kv_col_idx] = update_ping_pong(SHIM_V_ping_pong[shim_kv_col_idx])
            increment_DMA_queue(
                cur_shim_idx=shim_kv_col_idx,
                SHIM_queue_counter=SHIM_V_queue_counter
            )


def setup_SHM_configuration(
    DH:int, 
    NUM_OF_DH:int,
    
    
    # PER CT values
    LQ_per_CT:int, # per CT
    LK_per_CT:int,
    
    # whole thing
    SHIM_L_SEQ:int,
    SHIM_S_SEQ:int,
    SHIM_S_SEQ_PADDED:int, 
    
    # specs:
    NUM_OF_COLUMN:int, 
    NUM_OF_CT_per_column:int,
    MT_KV_repeat:int,
    CU_column:list[list[int]],
    
    Q, K, V, O, #input argument
    CT_rtp_lock_lists:list[list]
):



    LQ_per_CU = len(CU_column[0]) * (LQ_per_CT*NUM_OF_CT_per_column)
    assert  SHIM_L_SEQ%(LQ_per_CU) == 0    


    SHIM_Q_queue_counter = [ 0 for _ in range(NUM_OF_COLUMN) ]
    SHIM_K_queue_counter = [ 0 for _ in range(NUM_OF_COLUMN) ]
    SHIM_V_queue_counter = [ 0 for _ in range(NUM_OF_COLUMN) ]
    SHIM_O_queue_counter = [ 0 for _ in range(NUM_OF_COLUMN) ]
    
    SHIM_Q_ping_pong = [ 0 for _ in range(NUM_OF_COLUMN) ]
    SHIM_K_ping_pong = [ 0 for _ in range(NUM_OF_COLUMN) ]
    SHIM_V_ping_pong = [ 0 for _ in range(NUM_OF_COLUMN) ]
    SHIM_O_ping_pong = [ 0 for _ in range(NUM_OF_COLUMN) ]
    
    
    
    
    
    
    cur_active_CU_idx = 0 
    # precalculat how mnany SHIM_L_chunk each ct in the column will calculate
    chunk_per_CT_in_column = [0 for _ in range(NUM_OF_COLUMN)]
        
    for head_idx in range(NUM_OF_DH):
        for SHIM_L_chunk_idx in range( SHIM_L_SEQ//LQ_per_CU ):
            
            for col_idx in CU_column[cur_active_CU_idx]:
                chunk_per_CT_in_column[col_idx] += 1
            
            ##TODO: for now
            cur_active_CU_idx += 1
            if cur_active_CU_idx >= len(CU_column):
                cur_active_CU_idx = 0
                
                
                
    #TODO: move it outside so only write it once?
    for cur_CU_shimtile_cols in CU_column:
        for shim_col_idx in cur_CU_shimtile_cols:
            for row_idx in range(NUM_OF_CT_per_column):
                rtp_buf_str = f"rtp_config_buffer_{row_idx+2}_{shim_col_idx}"
                npu_rtp_write(rtp_buf_str, 0, SHIM_S_SEQ)
                npu_rtp_write(rtp_buf_str, 1, chunk_per_CT_in_column[shim_col_idx]*LQ_per_CT)
                set_lock_value(CT_rtp_lock_lists[shim_col_idx][row_idx], value=1)
        
    

    
    
    NUM_OF_CU_UNITS = len(CU_column)
    for head_idx_start in range(0, NUM_OF_DH, NUM_OF_CU_UNITS): 

        for SHIM_L_chunk_idx in range( SHIM_L_SEQ//LQ_per_CU ):
            if MT_KV_repeat == 1:
                process_SHM_Q_chunk_no_Q_repeat(
                    NUM_OF_CT_per_column=NUM_OF_CT_per_column,
                    MT_KV_repeat=MT_KV_repeat, 
                    
                    DH=DH, NUM_OF_DH=NUM_OF_DH, DH_index_start=head_idx_start,
                    
                    SHIM_LQ_start_row=   SHIM_L_chunk_idx*LQ_per_CU,
                    SHIM_S_SEQ= SHIM_S_SEQ,
                    SHIM_S_SEQ_PADDED = SHIM_S_SEQ_PADDED,
                    
                    CU_column=CU_column,
                    LQ_CU_chunk_size=LQ_per_CU,
                    LK_CU_chunk_size=LK_per_CT,
                    
                    LQ_per_CT = LQ_per_CT,
                    LK_per_CT=LK_per_CT,
                    
                    
                    SHIM_Q_queue_counter=SHIM_Q_queue_counter,
                    SHIM_K_queue_counter=SHIM_K_queue_counter,
                    SHIM_V_queue_counter = SHIM_V_queue_counter,
                    SHIM_O_queue_counter = SHIM_O_queue_counter,
                    SHIM_Q_ping_pong=SHIM_Q_ping_pong,
                    SHIM_K_ping_pong = SHIM_K_ping_pong,
                    SHIM_V_ping_pong=SHIM_V_ping_pong,
                    SHIM_O_ping_pong = SHIM_O_ping_pong,
                    Q=Q, K=K, V=V, O=O
                )
            else:
                process_SHM_Q_chunk_Q_repeat(
                    NUM_OF_CT_per_column=NUM_OF_CT_per_column,
                    MT_KV_repeat=MT_KV_repeat, 
                    
                    DH=DH, NUM_OF_DH=NUM_OF_DH, DH_index_start=head_idx_start,
                    
                    SHIM_LQ_start_row=   SHIM_L_chunk_idx*LQ_per_CU,
                    SHIM_S_SEQ= SHIM_S_SEQ,
                    SHIM_S_SEQ_PADDED = SHIM_S_SEQ_PADDED,
                    
                    CU_column = CU_column,
                    LQ_CU_chunk_size=LQ_per_CU,
                    LK_CU_chunk_size=LK_per_CT,
                    
                    LQ_per_CT = LQ_per_CT,
                    LK_per_CT=LK_per_CT,
                    
                    
                    SHIM_Q_queue_counter=SHIM_Q_queue_counter,
                    SHIM_K_queue_counter=SHIM_K_queue_counter,
                    SHIM_V_queue_counter = SHIM_V_queue_counter,
                    SHIM_O_queue_counter = SHIM_O_queue_counter,
                    SHIM_Q_ping_pong=SHIM_Q_ping_pong,
                    SHIM_K_ping_pong = SHIM_K_ping_pong,
                    SHIM_V_ping_pong=SHIM_V_ping_pong,
                    SHIM_O_ping_pong = SHIM_O_ping_pong,
                    Q=Q, K=K, V=V, O=O, 
                )
                
     
    
    # clear and wait for all remaining 
    
    is_all_flushed = False
    
    while not is_all_flushed:
        is_all_flushed = True  # init to True, 
        
        
        
        # start with q
        for idx in range(len(SHIM_Q_queue_counter)):
            cur_shim_Q_str, _, _,_ = get_shimtile_port_str(idx)
            if SHIM_Q_queue_counter[idx] > 0:
                is_all_flushed= False
                force_wait_DMA_queue(
                    cur_shim_idx= idx,
                    SHIM_queue_counter=SHIM_Q_queue_counter,
                    cur_shim_str= cur_shim_Q_str
                )
        
        # THEN K, V 
        for idx in range(len(SHIM_K_queue_counter)):
            _, cur_shim_KV_str, _, _ = get_shimtile_port_str(idx)
            
            # assert SHIM_K_queue_counter[idx] == SHIM_V_queue_counter[idx] 
            assert SHIM_K_queue_counter[idx] == 0
            if SHIM_V_queue_counter[idx] > 0:
                is_all_flushed = False
                
                
                # force_wait_DMA_queue(
                #     cur_shim_idx= idx,
                #     SHIM_queue_counter=SHIM_K_queue_counter,
                #     cur_shim_str= cur_shim_KV_str
                # )
                force_wait_DMA_queue(
                    cur_shim_idx= idx,
                    SHIM_queue_counter=SHIM_V_queue_counter,
                    cur_shim_str= cur_shim_KV_str
                )
    
    is_all_flushed = False
    while not is_all_flushed:
        is_all_flushed = True  # init to True, 
        
        for idx in range(len(SHIM_O_queue_counter)):
            _, _, cur_shim_O_str, _ = get_shimtile_port_str(idx)
            assert SHIM_O_queue_counter[idx] <= 2
            if SHIM_O_queue_counter[idx] > 0:
                is_all_flushed = False
                force_wait_DMA_queue(
                    cur_shim_idx=idx,
                    SHIM_queue_counter = SHIM_O_queue_counter,
                    cur_shim_str=cur_shim_O_str 
                )

        
    
