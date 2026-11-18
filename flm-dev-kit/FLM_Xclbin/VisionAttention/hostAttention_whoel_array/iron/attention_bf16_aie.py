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

from attention_SHM import get_shimtile_port_str, setup_SHM_configuration
from attention_CT import setup_CT_configure
from attention_MT import setup_MT_configure


import argparse

def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

def matmul(
    NUM_OF_COLUMNS:int, NUM_OF_CT_per_column:int,
    r:int, s:int, t:int,
    lq_internal:int, lk_lv_internal:int,
    
    DH:int, LQ_per_CT:int, LK_per_CT:int,
    
    NUM_OF_DH:int,  CU_mode:int,
    SHIM_L_SEQ:int, SHIM_S_SEQ:int,
    LOW_PRECISION:bool
 ):
 

    SHIM_S_SEQ_PADDED =  round_up(SHIM_S_SEQ, LK_per_CT)  #TODO: for now
        
    assert NUM_OF_CT_per_column == 4 

    if CU_mode == 0:
        # 1-1 
        CU_column = []
        for col in range(NUM_OF_COLUMNS):
            CU_column.append( [col] )
        # CU_column = [ 
        #             [0],[1],[2],[3],[4],[5],[6],[7]
        #             ]
    elif CU_mode == 1:
        #1-8
        assert NUM_OF_COLUMNS == 8
        CU_column = [ [0,1,2,3,4,5,6,7]]
    elif CU_mode == 2:
        #1-4
        assert NUM_OF_COLUMNS == 8  #TODO: for now        
        CU_column = [ [0,1,2,3], [4,5,6,7]]
    elif CU_mode == 3:
        #1-2
        assert NUM_OF_COLUMNS == 8  #TODO: for now        
        CU_column = [ [0,1],[2,3], [4,5],[6,7]]
    else:
        raise ValueError("unknown CU_mode parameter")
        
        

    
    # sanity checks

    assert LQ_per_CT%lq_internal == 0 and LQ_per_CT>=lq_internal, "LQ must be multiple of lq_internal and >= lq_internal"
    assert LK_per_CT%lk_lv_internal == 0 and LK_per_CT>=lk_lv_internal, "LK must be multiple of lk_v_internal and >= lk_v_internal"
    
    assert DH%s == 0, "DH must be multiple of s"

    
    assert lq_internal%(2*r) == 0, "lq_internal must be multiple of 2 r"
    assert lk_lv_internal%(2*t) == 0, "lk_v_internal must be multiple of 2 t"
    # multiple of 2 for better pipeline
    
    MT_KV_repeat:int = LQ_per_CT//lq_internal
    

    
    dtype_in = np.dtype[bfloat16]
    dtype_s = np.dtype[bfloat16]
    dtype_out = np.dtype[bfloat16]
    
    dtype_acc = np.dtype[np.float32]
    
    
    @device(AIEDevice.npu2)
    def device_body():
        stack_size = 4096

 

        
        CT_lq_internal_len = lq_internal*DH
        CT_lk_lv_internal_len = lk_lv_internal*DH
        
        CT_O_len = LQ_per_CT*DH
        CT_S_pij_len = lq_internal*LK_per_CT
        
        CT_lq_ty = np.ndarray[ (CT_lq_internal_len,), dtype_in ]
        CT_lk_lv_ty = np.ndarray[ (CT_lk_lv_internal_len,), dtype_in ]
        CT_S_pij_ty = np.ndarray[ (CT_S_pij_len,), dtype_s ]
        CT_O_ty = np.ndarray[ (CT_O_len,), dtype_acc ]
        
        rtp_config_ty = np.ndarray[(32,), np.dtype[np.int32]]
        ping_pong_ty = np.ndarray[(32,), np.dtype[np.int32]]

        if LOW_PRECISION == False:
            link_o_name = f"attention_{LQ_per_CT}_{LK_per_CT}_{lq_internal}_{lk_lv_internal}_mm_{r}_{s}_{t}_DH_{DH}.o"
        else:
            link_o_name = f"attention_{LQ_per_CT}_{LK_per_CT}_{lq_internal}_{lk_lv_internal}_mm_{r}_{s}_{t}_DH_{DH}_LOW_PRECISION.o"
            
        
        zero_int_array_func = external_func("zero_int32_array", inputs=[
            ping_pong_ty, np.int32
        ])
        attention_bf16_func = external_func("attention_bf16",inputs=[
            CT_lq_ty, 
            CT_lk_lv_ty,CT_lk_lv_ty,
            CT_S_pij_ty,
            CT_O_ty,
            rtp_config_ty,
            ping_pong_ty
        ])
        


        
        CT_lists :list[list] = [ ]
        CT_rtp_lock_lists:list[list] = []
        CT_rtp_config_buffer_lists:list[list] = []
        
        for col_idx in range(NUM_OF_COLUMNS):
            
            CT_per_column = []
            CT_rtp_lock_per_column = []
            CT_rtp_config_buffer_per_column = []
            for row_idx in range(NUM_OF_CT_per_column):
                
                
        
                CT, rtp_sync_lock, rtp_config_buffer = setup_CT_configure(
                    row=row_idx+2, col = col_idx,
                    stack_size=stack_size,
                    CT_lq_internal_len=CT_lq_internal_len,
                    CT_lk_lv_internal_len=CT_lk_lv_internal_len,
                    CT_O_len=CT_O_len,
                    CT_S_pij_len=CT_S_pij_len,
                    CT_lq_ty=CT_lq_ty,
                    CT_lk_lv_ty=CT_lk_lv_ty,
                    CT_S_pij_ty=CT_S_pij_ty,
                    CT_O_ty=CT_O_ty,
                    rtp_config_ty=rtp_config_ty,
                    ping_pong_ty=ping_pong_ty,
                    link_o_name=link_o_name,
                    zero_int_array_func=zero_int_array_func,
                    attention_bf16_func=attention_bf16_func
                )
                CT_per_column.append(CT)
                CT_rtp_lock_per_column.append(rtp_sync_lock)
                CT_rtp_config_buffer_per_column.append(rtp_config_buffer)
            CT_lists.append(CT_per_column)
            CT_rtp_lock_lists.append(CT_rtp_lock_per_column)
            CT_rtp_config_buffer_lists.append(CT_rtp_config_buffer_per_column)


        MT_q_len_per_CT = LQ_per_CT*DH
        MT_q_len = MT_q_len_per_CT * NUM_OF_CT_per_column
        
        
        MT_k_len = LK_per_CT*DH
        MT_v_len = MT_k_len
        MT_o_len_per_CT = LQ_per_CT*DH 
        MT_o_len = MT_o_len_per_CT * NUM_OF_CT_per_column
        
        MT_q_ty = np.ndarray[(MT_q_len,), dtype_in]
        MT_k_v_ty = np.ndarray[ (MT_k_len + MT_v_len,), dtype_in]
        MT_o_ty = np.ndarray[(MT_o_len,), dtype_out]


        MT_lists = []
        
        for col_idx in range(NUM_OF_COLUMNS):
            
            MT_core = setup_MT_configure(
                col=col_idx,
                NUM_OF_CT=NUM_OF_CT_per_column,
                LQ=LQ_per_CT,
                LK=LK_per_CT,
                DH=DH,
                r=r,
                s=s,
                t=t,
                MT_KV_repeat=MT_KV_repeat,
                MT_q_len=MT_q_len,
                MT_k_len=MT_k_len,
                MT_v_len=MT_v_len,
                MT_o_len=MT_o_len,
                MT_o_len_per_CT=MT_o_len_per_CT,
                MT_q_len_per_CT=MT_q_len_per_CT,
                MT_q_ty=MT_q_ty,
                MT_k_v_ty=MT_k_v_ty,
                MT_o_ty=MT_o_ty,
            )
            MT_lists.append(MT_core)

        
        

        
        SHIM_lists = []

        for i in range(NUM_OF_COLUMNS):
            SHIM_lists.append(tile(i, 0))
            

            cur_shim_Q_str, cur_shim_KV_str, cur_shim_O_str, cur_shim_S_str = get_shimtile_port_str(i)
            # shimtile, configure it for now
            memref.global_(cur_shim_Q_str, T.memref(SHIM_L_SEQ*DH, T.bf16()), sym_visibility="public")
            memref.global_(cur_shim_KV_str, T.memref(SHIM_S_SEQ_PADDED*DH*2, T.bf16()), sym_visibility="public")
            memref.global_(cur_shim_O_str, T.memref(SHIM_S_SEQ_PADDED*SHIM_L_SEQ, T.bf16()), sym_visibility="public")
            memref.global_(cur_shim_S_str, T.memref(SHIM_S_SEQ_PADDED*SHIM_L_SEQ, T.bf16()), sym_visibility="public")
            
            shim_dma_allocation(cur_shim_Q_str, DMAChannelDir.MM2S, 0,i)
            shim_dma_allocation(cur_shim_KV_str, DMAChannelDir.MM2S, 1, i)
            shim_dma_allocation(cur_shim_O_str, DMAChannelDir.S2MM, 0, i)
            shim_dma_allocation(cur_shim_S_str, DMAChannelDir.S2MM, 1, i)
            
  
            
        # mapping the shimtile kv route
        for cu_idxes in CU_column :
            
            
            # only the first column shimtile connect to memtile
            shim_kv_column_idx  =  cu_idxes[0]
            cur_kv_col_shimtile = SHIM_lists[shim_kv_column_idx]  

            for col_idx in cu_idxes:
                cur_col_memtile = MT_lists[col_idx]            
                flow(cur_kv_col_shimtile, WireBundle.DMA, 1,cur_col_memtile, WireBundle.DMA, 5)

        for col_idx in range(NUM_OF_COLUMNS):
            
            cur_col_shimtile = SHIM_lists[col_idx]
            cur_col_memtile = MT_lists[col_idx]
            
            assert NUM_OF_CT_per_column == 4
            
            cur_col_2_CT = CT_lists[col_idx][0]
            cur_col_3_CT = CT_lists[col_idx][1]
            cur_col_4_CT = CT_lists[col_idx][2]
            cur_col_5_CT = CT_lists[col_idx][3]                                    
            
            
            flow(cur_col_shimtile, WireBundle.DMA, 0, cur_col_memtile, WireBundle.DMA, 0)
            

            flow(cur_col_memtile, WireBundle.DMA, 4, cur_col_shimtile, WireBundle.DMA, 0)
            
            # send q
            flow(cur_col_memtile, WireBundle.DMA, 0, cur_col_2_CT, WireBundle.DMA, 0)
            flow(cur_col_memtile, WireBundle.DMA, 1, cur_col_3_CT, WireBundle.DMA, 0)
            flow(cur_col_memtile, WireBundle.DMA, 2, cur_col_4_CT, WireBundle.DMA, 0)
            flow(cur_col_memtile, WireBundle.DMA, 3, cur_col_5_CT, WireBundle.DMA, 0)            
            
            # send kv
            flow(cur_col_memtile, WireBundle.DMA, 5, cur_col_2_CT, WireBundle.DMA, 1)
            flow(cur_col_memtile, WireBundle.DMA, 5, cur_col_3_CT, WireBundle.DMA, 1)
            flow(cur_col_memtile, WireBundle.DMA, 5, cur_col_4_CT, WireBundle.DMA, 1)
            flow(cur_col_memtile, WireBundle.DMA, 5, cur_col_5_CT, WireBundle.DMA, 1)
            
            # receive 0                                    
            
            flow(cur_col_2_CT, WireBundle.DMA, 0, cur_col_memtile, WireBundle.DMA, 1)
            flow(cur_col_3_CT, WireBundle.DMA, 0, cur_col_memtile, WireBundle.DMA, 2)
            flow(cur_col_4_CT, WireBundle.DMA, 0, cur_col_memtile, WireBundle.DMA, 3)
            flow(cur_col_5_CT, WireBundle.DMA, 0, cur_col_memtile, WireBundle.DMA, 4)                                    
            
            

        # @runtime_sequence(
        #     np.ndarray[(SHIM_L_SEQ*DH,), dtype_in],
        #     np.ndarray[(SHIM_S_SEQ_PADDED*DH,), dtype_in],
        #     np.ndarray[(SHIM_S_SEQ_PADDED*DH,), dtype_in],            
        #     np.ndarray[(SHIM_S_SEQ_PADDED*SHIM_L_SEQ,), dtype_out]
        # )
        # def sequence(Q, K,V, O):
        #     # for now, do a simple passthrough kernel?
        #     setup_SHM_configuration(
        #         DH=DH,
        #         NUM_OF_DH=NUM_OF_DH,
        #         LQ_per_CT = LQ_per_CT,
        #         LK_per_CT  = LK_per_CT,
        #         SHIM_L_SEQ = SHIM_L_SEQ,
        #         SHIM_S_SEQ= SHIM_S_SEQ,
        #         SHIM_S_SEQ_PADDED=SHIM_S_SEQ_PADDED,
        #         NUM_OF_CT_per_column=NUM_OF_CT_per_column,
        #         NUM_OF_COLUMN=NUM_OF_COLUMNS,
        #         MT_KV_repeat=MT_KV_repeat,
        #         CU_column=CU_column,
        #         Q=Q, K=K, V=V, O=O,
        #         CT_rtp_lock_lists=CT_rtp_lock_lists
        #     )
                           
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run matmul with command line arguments.")
    parser.add_argument("arch",type=str)
    parser.add_argument("r", type=int)
    parser.add_argument("s", type=int)
    parser.add_argument("t", type=int)
    parser.add_argument("DH", type=int)
    parser.add_argument("LQ_per_CT", type=int)
    parser.add_argument("LK_per_CT", type=int)
    parser.add_argument("lq_internal", type=int)
    parser.add_argument("lk_lv_internal", type=int)
    parser.add_argument("NUM_OF_DH", type=int)
    parser.add_argument("SHIM_L_SEQ", type=int)
    parser.add_argument("SHIM_S_SEQ", type=int)
    parser.add_argument("NUM_OF_COLUMNS", type=int)
    parser.add_argument("NUM_OF_CT_per_column", type=int)
    parser.add_argument("CU_mode", type=int)
    parser.add_argument("LOW_PRECISION", type=int)
    parser.add_argument("RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION", type=int)
    
    args = parser.parse_args()

    with mlir_mod_ctx() as ctx:            
        matmul(
            NUM_OF_COLUMNS=args.NUM_OF_COLUMNS,
            NUM_OF_CT_per_column=args.NUM_OF_CT_per_column,
            r=args.r,
            s=args.s,
            t=args.t,
            lq_internal=args.lq_internal,
            lk_lv_internal=args.lk_lv_internal,
            DH=args.DH,
            LQ_per_CT=args.LQ_per_CT,
            LK_per_CT=args.LK_per_CT,
            NUM_OF_DH=args.NUM_OF_DH,
            CU_mode=args.CU_mode,
            SHIM_L_SEQ=args.SHIM_L_SEQ,
            SHIM_S_SEQ=args.SHIM_S_SEQ,
            LOW_PRECISION=  True if args.LOW_PRECISION == 1 else False
        )
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)