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
from attention_SHIM import get_SHIM_str
import argparse


def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def matmul(
    NUM_OF_CT_PER_COLUMN:int, NUM_OF_COLUMNS:int,
    r:int, s:int, t:int,
    DH:int, LQ:int, LK:int, LOW_PRECISION:int,USE_CASUAL_MASK:int,  NUM_HEAD:int,
    SHIM_L_SEQ:int, SHIM_S_SEQ:int,    
    arch="npu2", trace_size:int=0           
    ):

 
    assert(LQ==LK)
    assert( SHIM_L_SEQ == SHIM_S_SEQ)

    assert(   SHIM_L_SEQ %(NUM_OF_CT_PER_COLUMN * LQ) == 0)
    

    

    


    # sanity checks

    assert DH%s == 0, "DH must be multiple of s"

    
    # assert LQ%(2*r) == 0, "LQ must be multiple of 2 r"
    # assert LK%(2*t) == 0, "LK must be multiple of 2 t"
    # multiple of 2 for better pipeline
    

    
    dtype_in = np.dtype[bfloat16]
    dtype_s = np.dtype[bfloat16]
    dtype_out = np.dtype[bfloat16]
    
    
    
    @device(AIEDevice.npu2)
    def device_body():
        stack_size = 4096

        
        
        #TODO: need ping-pong on both k, v or not?
        CT_q_len = LQ*DH
        CT_k_len = LK*DH
        CT_v_len = LK*DH
        

        CT_O_len = CT_q_len
        
        CT_S_pij_len =  LQ*LK
        
        CT_lq_ty = np.ndarray[ (CT_q_len,), dtype_in ]
        CT_lk_ty = np.ndarray[ (CT_k_len,), dtype_in]
        CT_lv_ty = np.ndarray[(CT_v_len,), dtype_in]
        
        CT_S_pij_ty = np.ndarray[ (CT_S_pij_len  ,), dtype_s ]
        CT_O_ty = np.ndarray[ (CT_O_len,), dtype_out]
   
        ping_pong_ty = np.ndarray[(32,), np.dtype[np.int32]]

        if LOW_PRECISION== 1 and USE_CASUAL_MASK== 0:
            link_o_name = f"window_attention_{LQ}_{LK}_mm_{r}_{s}_{t}_DH_{DH}_LOW_PRECISION.o"            
        elif LOW_PRECISION== 0 and USE_CASUAL_MASK== 0:
            link_o_name = f"window_attention_{LQ}_{LK}_mm_{r}_{s}_{t}_DH_{DH}.o"
        elif LOW_PRECISION== 1 and USE_CASUAL_MASK== 1:
            link_o_name = f"window_attention_{LQ}_{LK}_mm_{r}_{s}_{t}_DH_{DH}_LOW_PRECISION_casual_mask.o"
        else:
            link_o_name = f"window_attention_{LQ}_{LK}_mm_{r}_{s}_{t}_DH_{DH}_casual_mask.o"
        zero_int_array_func = external_func("zero_int32_array", inputs=[
            ping_pong_ty, np.int32
        ])
        attention_bf16_func = external_func("attention_bf16",inputs=[
            CT_lq_ty, CT_lq_ty,
            CT_lk_ty,
            CT_lv_ty,
            CT_S_pij_ty,
            CT_O_ty,
            ping_pong_ty
        ])
        
        
        
        CT_array = [  [] for _ in range(NUM_OF_COLUMNS)]
        for col_idx in range(NUM_OF_COLUMNS):
            
            for row_idx in range(NUM_OF_CT_PER_COLUMN):
                CT_core= setup_CT_configure(
                    row=2+row_idx, col = col_idx,
                    stack_size=stack_size,
                    CT_lq_len=CT_q_len, CT_lk_len=CT_k_len, CT_lv_len=CT_v_len,
                    CT_O_len=CT_O_len,
                    CT_S_pij_ty=CT_S_pij_ty,
                    CT_O_ty=CT_O_ty,
                    CT_lq_ty=CT_lq_ty, CT_lk_ty=CT_lk_ty, CT_lv_ty=CT_lv_ty,
                    ping_pong_ty=ping_pong_ty,
                    link_o_name=link_o_name,
                    zero_int_array_func=zero_int_array_func,
                    attention_bf16_func=attention_bf16_func
                )
                CT_array[col_idx].append(CT_core)                

        MT_q_len_per_CT = LQ*DH
        MT_k_len_per_CT = LK*DH
        MT_v_len_per_CT = MT_k_len_per_CT
        MT_o_len_per_CT = LQ*DH 
        
        
        MT_q_len = MT_q_len_per_CT * NUM_OF_CT_PER_COLUMN
        MT_k_len = MT_k_len_per_CT * NUM_OF_CT_PER_COLUMN
        MT_v_len = MT_v_len_per_CT * NUM_OF_CT_PER_COLUMN
        MT_o_len = MT_o_len_per_CT * NUM_OF_CT_PER_COLUMN
        
        MT_q_ty = np.ndarray[(MT_q_len,), dtype_in]
        MT_k_v_ty = np.ndarray[(MT_k_len + MT_v_len,), dtype_in]

        MT_o_ty = np.ndarray[(MT_o_len,), dtype_out]


        MT_array = []
        
        
        for col_idx in range(NUM_OF_COLUMNS):
                
            MT_core = setup_MT_configure(
                col=col_idx,
                NUM_OF_CT=NUM_OF_CT_PER_COLUMN,
                LQ=LQ,
                LK=LK,
                DH=DH,
                r=r,
                s=s,
                t=t,
                MT_q_ty=MT_q_ty,
                MT_k_v_ty=MT_k_v_ty,
                MT_o_ty=MT_o_ty,
                MT_q_len=MT_q_len,
                MT_k_len=MT_k_len,
                MT_v_len=MT_v_len,
                MT_o_len=MT_o_len,
                
                MT_o_len_per_CT= MT_o_len_per_CT,
                MT_q_len_per_CT=MT_q_len_per_CT,
                MT_k_v_len_per_CT= MT_k_len_per_CT + MT_v_len_per_CT
            )
            MT_array.append(MT_core)
        
        

    

        SHIM_array = []
        for col_idx in range(NUM_OF_COLUMNS):
            Shimtile_core =tile(col_idx,0)
            SHIM_array.append(Shimtile_core)
            Q_str, K_V_str, O_str = get_SHIM_str(col_idx=col_idx)
        
            # shimtile, configure it for now
            memref.global_(Q_str, T.memref(SHIM_L_SEQ*DH, T.bf16()), sym_visibility="public")
            memref.global_(K_V_str, T.memref(SHIM_S_SEQ*DH*2, T.bf16()), sym_visibility="public")
            memref.global_(O_str, T.memref(SHIM_S_SEQ*SHIM_L_SEQ, T.bf16()), sym_visibility="public")

            
            shim_dma_allocation(Q_str, DMAChannelDir.MM2S, 0,col_idx)
            shim_dma_allocation(K_V_str, DMAChannelDir.MM2S, 1, col_idx)
            shim_dma_allocation(O_str, DMAChannelDir.S2MM, 0, col_idx)
            
            
            
            
        for col_idx in range(NUM_OF_COLUMNS):
            shim_core = SHIM_array[col_idx]
            MT_core = MT_array[col_idx]
            

            flow(shim_core, WireBundle.DMA, 0, MT_core, WireBundle.DMA, 0)
            flow(shim_core, WireBundle.DMA, 1,MT_core, WireBundle.DMA, 1)
            flow(MT_core, WireBundle.DMA, 0, shim_core, WireBundle.DMA, 0)
            
            
            for row_idx in range(NUM_OF_CT_PER_COLUMN):
                    
                # send q,k,v
                CT_core = CT_array[col_idx][row_idx]
                flow(MT_core, WireBundle.DMA, 2+row_idx, CT_core, WireBundle.DMA, 0)
                flow(CT_core, WireBundle.DMA, 0, MT_core, WireBundle.DMA, 2+row_idx)


        @runtime_sequence(
            np.ndarray[(SHIM_L_SEQ*DH,), dtype_in],
            np.ndarray[(SHIM_S_SEQ*DH,), dtype_in],
            np.ndarray[(SHIM_S_SEQ*DH,), dtype_in],
            np.ndarray[(SHIM_S_SEQ*SHIM_L_SEQ,), dtype_out]
        )
        def sequence(Q, K,V, O):
            # for now, do a simple passthrough kernel?
            
            


            assert SHIM_L_SEQ == SHIM_S_SEQ

            assert SHIM_S_SEQ%(NUM_OF_CT_PER_COLUMN * LK) == 0
            

            
            
            O_queue_list = [0 for _ in range(NUM_OF_COLUMNS)]
            
            
            # for q: 0/8 k:1/9 v:2/10 o: 3/11
            bd_list = [0 for _ in range(NUM_OF_COLUMNS)] # 0 or 8

            
            # given each column is suppose to process one head at a time
            
            
         

            num_of_transaction = NUM_HEAD//(NUM_OF_COLUMNS)
            if(NUM_HEAD % NUM_OF_COLUMNS != 0):
                num_of_transaction += 1

            for transaction_idx in range(num_of_transaction):
                
                for blocks_idx in range( (SHIM_L_SEQ//(NUM_OF_CT_PER_COLUMN*LQ))  ):
                    
                    for cur_col_idx in range(NUM_OF_COLUMNS):
                        
                        head_idx = transaction_idx*NUM_OF_COLUMNS + cur_col_idx
                        if head_idx >= NUM_HEAD:
                            # all heads are processed
                            break;
                        
                        cur_Q_str, cur_K_V_Str, cur_O_str = get_SHIM_str(cur_col_idx)   
                        q_offset = head_idx * DH
                        k_offset = head_idx *DH
                        v_offset = head_idx* DH
                        o_offset = head_idx *DH
                

                    
                        if O_queue_list[cur_col_idx] == 2:
                            npu_dma_wait(cur_O_str)       
                            O_queue_list[cur_col_idx]-=1 
                        
                        if bd_list[cur_col_idx] == 0:
                            q_bd_id = 0
                            k_bd_id = 1
                            v_bd_id = 2
                            o_bd_id = 3
                            bd_list[cur_col_idx] = 1
                        else:
                            q_bd_id = 0+8
                            k_bd_id = 1+8
                            v_bd_id = 2+8
                            o_bd_id = 3+8
                            bd_list[cur_col_idx] = 0
                        
                        

                        npu_dma_memcpy_nd(
                            metadata=cur_Q_str,
                            bd_id=q_bd_id,
                            mem=Q,
                            offsets=[0,0,0, q_offset+  blocks_idx* (NUM_OF_CT_PER_COLUMN*LQ)*(DH * NUM_HEAD) ],
                            sizes = [1, 1, (NUM_OF_CT_PER_COLUMN*LQ), DH],
                            strides = [0,0, (DH*NUM_HEAD), 1]
                        )
                        
                        npu_dma_memcpy_nd(
                            metadata=cur_K_V_Str,
                            bd_id=k_bd_id,
                            mem=K,
                            offsets=[0,0,0, k_offset + blocks_idx*(NUM_OF_CT_PER_COLUMN*LK) *(DH* NUM_HEAD)],
                            sizes = [1, 1, (NUM_OF_CT_PER_COLUMN*LK), DH],
                            strides = [0,0, (DH*NUM_HEAD), 1]                        
                        )
                        npu_dma_memcpy_nd(
                            metadata=cur_K_V_Str,
                            bd_id=v_bd_id,
                            mem=V,
                            offsets=[0,0,0, v_offset + blocks_idx*(NUM_OF_CT_PER_COLUMN*LK) *(DH* NUM_HEAD)],
                            sizes = [1, 1, (NUM_OF_CT_PER_COLUMN*LK), DH],
                            strides = [0,0, (DH*NUM_HEAD), 1]       
                        )    
                        
                        
                        # receive in terminal
                        npu_dma_memcpy_nd(
                            metadata=cur_O_str,
                            bd_id=o_bd_id,
                            mem=O,
                            offsets=[0,0,0, o_offset + blocks_idx*((NUM_OF_CT_PER_COLUMN*LQ)*(DH*NUM_HEAD))],
                            sizes = [1, 1, (NUM_OF_CT_PER_COLUMN*LQ), DH],
                            strides = [0,0, (DH*NUM_HEAD), 1],                        
                            issue_token=True
                        )
                        O_queue_list[cur_col_idx] += 1
    
                    

                    
  

            
            max_remaining:int = -1
            for col_idx in range(NUM_OF_COLUMNS):
                if O_queue_list[col_idx] > max_remaining:
                    max_remaining = O_queue_list[col_idx]

            
            for _ in range(max_remaining):
                for col_idx in range(NUM_OF_COLUMNS):
                    _, _, O_shim = get_SHIM_str(col_idx=col_idx)
                    if O_queue_list[col_idx] > 0:
                        npu_dma_wait(O_shim)
                        O_queue_list[col_idx] -= 1
                              

with mlir_mod_ctx() as ctx:     
    
    parser = argparse.ArgumentParser(description="Run matmul with command line arguments.")
    parser.add_argument("arch", type=str)
    parser.add_argument("trace_size", type=int)
    parser.add_argument("NUM_OF_COLUMNS", type=int)
    parser.add_argument("NUM_OF_CT_PER_COLUMN", type=int)
    parser.add_argument("r", type=int)
    parser.add_argument("s", type=int)
    parser.add_argument("t", type=int)
    parser.add_argument("DH", type=int)
    parser.add_argument("LQ", type=int)
    parser.add_argument("LK", type=int)
    parser.add_argument("LOW_PRECISION", type=int)
    parser.add_argument("USE_CASUAL_MASK", type=int)
    parser.add_argument("NUM_HEAD", type=int)
    parser.add_argument("SHIM_L_SEQ", type=int)
    parser.add_argument("SHIM_S_SEQ", type=int)

    args = parser.parse_args()

    matmul(
        NUM_OF_CT_PER_COLUMN=args.NUM_OF_CT_PER_COLUMN,
        NUM_OF_COLUMNS=args.NUM_OF_COLUMNS,
        r=args.r,
        s=args.s,
        t=args.t,
        DH=args.DH,
        LQ=args.LQ,
        LK=args.LK,
        LOW_PRECISION=args.LOW_PRECISION,
        USE_CASUAL_MASK=args.USE_CASUAL_MASK,
        NUM_HEAD=args.NUM_HEAD,
        SHIM_L_SEQ=args.SHIM_L_SEQ,
        SHIM_S_SEQ=args.SHIM_S_SEQ,
        arch=args.arch,
        trace_size=args.trace_size
    )
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)