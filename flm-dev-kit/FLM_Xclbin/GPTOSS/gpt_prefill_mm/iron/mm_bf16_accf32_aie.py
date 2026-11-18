# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.dialects.ext.arith import constant, index_cast
from pyparsing import col
from sympy import use
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.utils.trace import MemTilePortEvent

from aie.ir import *
from aie.ir import MemRefType, IndexType
from aie.dialects import arith, memref
from aie.dialects.memref import AllocaScopeOp
from aie.extras.dialects.ext.arith import Scalar

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

from mm_CT_setup import define_CT_for_MM
from mm_MT_setup import define_MT_for_MM
from mm_shm_setup import get_shim_A_B_C_str, generate_runtime_sequence
import argparse
def matmul(arch="npu2", trace_size:int=0, M_size:int=0, K_size:int=0, N_size:int=0,
           m=0,k=0,n=0, IS_B_ROW_MAJOR=None, total_row:int=0, total_col:int=0,
           r=0, s=0, t=0, ENABLE_AXI4=None, IS_B_COL_MAJOR_S_T_ROW_MAJOR=None, IS_BFLOAT16_OUT_TY=None,
           ADD_BIAS=None, OUTPUT_MODE=None
           ):
    assert ENABLE_AXI4 is not None
    assert IS_B_ROW_MAJOR is not None
    #HARDCODe for now
    # assert(arch=="npu2")
    # assert(M>0 and K>0 and N>0 and m >0 and k >0 and n >0)
    # dev = AIEDevice.npu2
    # total_row = 4 #TODO
    # total_col = 8
    assert IS_BFLOAT16_OUT_TY is not None
    assert IS_B_ROW_MAJOR is not None
    # #TODO:
    # trace_size = 16384 # in byte
    # IS_B_ROW_MAJOR = False
    # M_size = 256
    # K_size = 64
    # N_size = 512
    # m = 64
    # k = 64
    # n =64
    
    
    assert M_size%(m*total_row) == 0
    assert K_size%k == 0
    assert N_size%(n) == 0  # Only ensure N_size is divisible by n
    assert m %r == 0
    assert k %s == 0
    assert n %t == 0

    
    # #tiles is number of mxn this CT needs to produce for
    # tiles = M_div_m * N_div_n
    
    dtype_in = np.dtype[bfloat16]   
    if IS_BFLOAT16_OUT_TY:
        dtype_out = np.dtype[bfloat16]
    else:
        dtype_out = np.dtype[np.float32]
    dtype_acc = np.dtype[np.float32]
    
    m_times_k = m*k
    m_times_n = m*n
    k_times_n = k*n
    
    A_sz = M_size * K_size
    B_sz = K_size * N_size
    C_sz = M_size * N_size
        

    
    @device(AIEDevice.npu2)
    def device_body():
        
        k_div_CT_k_max =0
        stack_size = 4096
        if n== 64:
            CT_max_k_size_m_128_n_64 = min(64, k)            
        elif n==128:
            CT_max_k_size_m_128_n_64 = min(32, k)
        elif n==256:
            CT_max_k_size_m_128_n_64 = min(16, k)
        elif n==32:
            CT_max_k_size_m_128_n_64 = 32#min(16, k)
        elif n==16:
            CT_max_k_size_m_128_n_64 = 16
        else:
            CT_max_k_size_m_128_n_64 = 0
            assert 1==2
        k_div_CT_k_max = k//CT_max_k_size_m_128_n_64
        assert k%CT_max_k_size_m_128_n_64 == 0
        
        CT_a_ty_len = (2*r)*CT_max_k_size_m_128_n_64  # left, then right
        CT_b_ty_len = CT_max_k_size_m_128_n_64*n  # top, then bottom
        CT_out_ty_len = 512 
        
        
        CT_a_ty = np.ndarray[ (CT_a_ty_len,), dtype_in]
        CT_b_ty = np.ndarray[ (CT_b_ty_len, ), dtype_in]
        CT_out_ty = np.ndarray[ (CT_out_ty_len, ), dtype_out]
        CT_acc_ty = np.ndarray[ (m_times_n, ), dtype_acc]
        CT_bias_ty = np.ndarray[   (r*n,), dtype_in]
        
        ping_pong_ty = np.ndarray[(1,), np.dtype[np.int32]]
        rtp_ty = np.ndarray[(32,), np.dtype[np.int32]]  # 32 for alignment requirement, since need 32 byte aligned address. so 

        zero_int_1 = external_func("zero_int32_1", inputs=[
            ping_pong_ty
        ])

        out_ty_str = "bfloat16_out" if IS_BFLOAT16_OUT_TY else "float_out"  
        mm_func_optimized = external_func( f"matmul_{m}_{k}_{n}_{r}_{s}_{t}_ping_pong_float_acc_{out_ty_str}_mega_down", inputs=[
            CT_a_ty, CT_a_ty,
            CT_b_ty,CT_b_ty,
            CT_bias_ty,            
            CT_acc_ty,
            CT_out_ty, CT_out_ty,
            ping_pong_ty, ping_pong_ty,
            np.int32, np.int32, np.int32, np.int32, 
            rtp_ty,
            np.int32, np.int32, 
            np.int32, np.int32,
            np.int32, np.int32
        ])
        if IS_B_COL_MAJOR_S_T_ROW_MAJOR == False:
            mm_func_b_col_optimized = external_func(f"matmul_{m}_{k}_{n}_{r}_{s}_{t}_ping_pong_float_acc_col_major_{out_ty_str}_mega_down", inputs=[
                CT_a_ty, CT_a_ty,
                CT_b_ty, CT_b_ty,
                CT_bias_ty,                
                CT_acc_ty,
                CT_out_ty, CT_out_ty,
                ping_pong_ty, ping_pong_ty,
                np.int32, np.int32, np.int32, np.int32, 
                rtp_ty,             
                np.int32, np.int32, 
                np.int32, np.int32,
                np.int32, np.int32
            ])      
        else:
            mm_func_b_col_optimized = external_func(f"matmul_{m}_{k}_{n}_{r}_{s}_{t}_ping_pong_float_acc_col_major_s_t_row_major_{out_ty_str}_mega_down", inputs=[
                CT_a_ty, CT_a_ty,
                CT_b_ty, CT_b_ty,
                CT_bias_ty,                
                CT_acc_ty,
                CT_out_ty, CT_out_ty,
                ping_pong_ty, ping_pong_ty,
                np.int32, np.int32, np.int32, np.int32, 
                rtp_ty,    
                np.int32, np.int32, 
                np.int32, np.int32,
                np.int32, np.int32
            ])     
        

        if IS_B_ROW_MAJOR:
            CT_link_o = f"mm_{m}_{k}_{n}_{r}_{s}_{t}bfloat16_mega_down.o"
        else:
            CT_link_o = f"mm_{m}_{k}_{n}_{r}_{s}_{t}bfloat16_col_major_mega_down.o"

        CT_list =[[None for _ in range(total_col)] for _ in range(total_row)]
        CT_rtp_name_list: List[List[Any]] =[[None for _ in range(total_col)] for _ in range(total_row)]        
        CT_rtp_lock_list =[[None for _ in range(total_col)] for _ in range(total_row)]        
        for row_idx in range(total_row):
            for col_idx in range(total_col):
                CT_list[row_idx][col_idx], CT_rtp_name_list[row_idx][col_idx], CT_rtp_lock_list[row_idx][col_idx] =define_CT_for_MM(stack_size=stack_size, 
                        col=col_idx, row=2+row_idx, TOTAL_CT_ROW=total_row, TOTAL_CT_COL=total_col,
                        CT_bias_ty=CT_bias_ty,
                         CT_a_ty=CT_a_ty, CT_b_ty=CT_b_ty, CT_out_ty=CT_out_ty, CT_acc_ty=CT_acc_ty, ping_pong_ty=ping_pong_ty, rtp_ty=rtp_ty,
                         CT_a_ty_len=CT_a_ty_len, CT_b_ty_len=CT_b_ty_len, CT_out_ty_len=CT_out_ty_len,
                         zero_int_1=zero_int_1, mm_func_optimized=mm_func_optimized, mm_func_b_col_optimized=mm_func_b_col_optimized,
                         CT_link_o = CT_link_o, IS_B_ROW_MAJOR=IS_B_ROW_MAJOR
                         ) 
                
                
        MT_list = [None for _  in range(max(total_row,total_col))]
            
        MT_a_ty_len = m_times_k
        MT_b_ty_len = k_times_n

        MT_a_ty =  np.ndarray[ (MT_a_ty_len,), dtype_in]
        MT_b_ty = np.ndarray[ (MT_b_ty_len,), dtype_in]


        out_MT_size = m_times_n*4
        out_MT_ty = np.ndarray[ (out_MT_size, ), dtype_out]  # gather 4 CT's result at a time   
        
        # A_pkt_broadcast_id = 0   # 0 to total_row number            
        for col_idx in range(max(total_row,total_col)):

            MT_list[col_idx] = define_MT_for_MM(col=col_idx, 
                                    k_div_CT_k_max=k_div_CT_k_max,
                                    MT_a_ty=MT_a_ty, MT_b_ty=MT_b_ty, out_MT_ty=out_MT_ty,
                                    out_MT_size=out_MT_size, 
                                    MT_a_ty_len=MT_a_ty_len, MT_b_ty_len=MT_b_ty_len, m_times_n=m_times_n,
                                    r=r, s=s, t= t, m=m, n=n, k=k,
                                    IS_B_ROW_MAJOR=IS_B_ROW_MAJOR, A_broadcast_pkt_id= None, IS_BFLOAT16_OUT_TY=IS_BFLOAT16_OUT_TY
                                    )
            # A_pkt_broadcast_id+= 1

        Shim_list = [None for _ in range(max(total_col, total_row))]
        for idx in range(max(total_col, total_row)):
            Shim_list[idx]=tile(col=idx, row=0)
        
        for i in range(len(Shim_list)):
            A_str, B_str, C_str = get_shim_A_B_C_str(i)
            memref.global_(A_str, T.memref( A_sz, T.bf16() ), sym_visibility="public")            
            memref.global_(B_str, T.memref( B_sz, T.bf16()), sym_visibility="public" ) # result outTODO:
            memref.global_(C_str, T.memref(C_sz, T.bf16()), sym_visibility="public")
         
            shim_dma_allocation(A_str, DMAChannelDir.MM2S, 0, i)
            shim_dma_allocation(B_str, DMAChannelDir.MM2S, 1,i)
            shim_dma_allocation(C_str, DMAChannelDir.S2MM, 0, i)
        

        # mt-> shimtile                  
        #SHM to MT mapping, direct mapping
        assert(len(Shim_list)  ==len(MT_list))
        for i in range(len(MT_list)):
            shm_temp = Shim_list[i]
            MT_temp = MT_list[i]
            flow(shm_temp, WireBundle.DMA, 0, MT_temp, WireBundle.DMA, 0)
            flow(shm_temp, WireBundle.DMA, 1, MT_temp, WireBundle.DMA, 1)
            
            flow(MT_temp, WireBundle.DMA, 2, shm_temp, WireBundle.DMA, 0)
            
        # MT to CT mapping
        # A: broadcast to row
        valid_A_MT_shimtile_index = [0, 2, 4 ,6]
        
        for i in range(total_row):
            # A is broadcast across row from MT to all rows in CT
            MT_temp = MT_list[  valid_A_MT_shimtile_index[i] ]
            row_CTS = CT_list[i]
            for CT_temp in row_CTS:
                flow( MT_temp, WireBundle.DMA, 0, CT_temp, WireBundle.DMA, 0)
        # B: broadcast to all column
        for i in range(total_col):
            MT_temp = MT_list[i]
            for row_id in range(total_row):
                CT_temp =CT_list[row_id][i]
                flow(MT_temp, WireBundle.DMA, 1, CT_temp, WireBundle.DMA, 1 )
        # Result:
        # EACH MT gather Each column instead
        #MT0: CT0_2-> CT_3_2
        #MT1: CT_
        for col_idx in range(total_col):
            
            MT_temp = MT_list[col_idx]
            for row_idx in range(total_row):
                flow(
                    CT_list[row_idx][col_idx], WireBundle.DMA, 0,
                    MT_temp, WireBundle.DMA, 2+row_idx
                    
                )
        # @runtime_sequence(
        #     np.ndarray[(A_sz, ), dtype_in],
        #     np.ndarray[(B_sz, ), dtype_in],
        #     np.ndarray[(C_sz, ), dtype_out]
        # )
        # def sequence(A,B,C):
            
        #     #M_div_m_row = M_size//(m*total_row)
        #     K_div_k: int = K_size//k
        #     #N_div_n_col = N_size//(n*total_col)
            
        #     for row_idx in range(total_row):
        #         for col_idx in range(total_col):
        #             npu_rtp_write(  CT_rtp_name_list[row_idx][col_idx], 0, K_div_k)
        #             npu_rtp_write(  CT_rtp_name_list[row_idx][col_idx], 1, M_size)
        #             npu_rtp_write(  CT_rtp_name_list[row_idx][col_idx], 2, N_size)     
        #             npu_rtp_write(  CT_rtp_name_list[row_idx][col_idx], 3, ADD_BIAS) #NOTE: for now
        #             npu_rtp_write(  CT_rtp_name_list[row_idx][col_idx], 4, OUTPUT_MODE)                                                           
        #             set_lock( CT_rtp_lock_list[row_idx][col_idx], value=1)
                    
        #     list_C_shim_queue = [  0 for _ in range(len(Shim_list))]
        #     list_A_shim_queue = [0 for _ in range(total_col)]
        #     list_B_shim_queue = [0 for _ in range(total_col)]
        #     generate_runtime_sequence(
        #         A=A,B=B,C=C, M_size=M_size, N_size=N_size,
        #         K_size=K_size, m=m,n=n,k=k, total_npu_row=total_row, total_npu_col = total_col,
        #         list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
        #         list_C_shim_queue=list_C_shim_queue,
        #         IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
        #         valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
        #         ADD_BIAS=ADD_BIAS,
        #         ENABLE_AXI4=ENABLE_AXI4
        #     )
            
        #     #now, wait for every C
        #     max_C_remain = 0
        #     for li in list_C_shim_queue:
        #         max_C_remain =max( li, max_C_remain)


        #     for _ in range(max_C_remain):
                
        #         for shim_index in range(total_col):
        #             A_str,B_str, C_str = get_shim_A_B_C_str(shim_index)
        #             if list_A_shim_queue[shim_index] > 0:
        #                 npu_dma_wait(A_str)
        #                 list_A_shim_queue[shim_index] -= 1
        #             if list_B_shim_queue[shim_index] > 0:
        #                 npu_dma_wait(B_str)
        #                 list_B_shim_queue[shim_index] -=1
        #             if list_C_shim_queue[shim_index] > 0:
        #                 npu_dma_wait(C_str)
        #                 list_C_shim_queue[shim_index] -=1
          
            
          


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run matmul with command line arguments.")

    parser.add_argument("arch",type=str)
    parser.add_argument("trace_size", type=int)
    parser.add_argument("M_size", type=int)
    parser.add_argument("K_size", type=int)
    parser.add_argument("N_size", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("IS_B_ROW_MAJOR", type=int)
    parser.add_argument("total_row", type=int)
    parser.add_argument("total_col", type=int)
    parser.add_argument("r", type=int)
    parser.add_argument("s", type=int)
    parser.add_argument("t", type=int)
    parser.add_argument("ENABLE_AXI4", type=int)
    parser.add_argument("IS_B_COL_MAJOR_S_T_ROW_MAJOR", type=int)
    parser.add_argument("IS_BFLOAT16_OUT_TY", type=int)
    parser.add_argument("ADD_BIAS", type=int)
    parser.add_argument("OUTPUT_MODE", type=int)
    parser.add_argument('OUTPUT_CLAMP', type=int)

    args = parser.parse_args()
    
  
    with mlir_mod_ctx() as ctx:
        matmul(
            arch=args.arch,
            trace_size=args.trace_size,
            M_size=args.M_size,
            K_size=args.K_size,
            N_size=args.N_size,
            m=args.m,
            k=args.k,
            n=args.n,
            IS_B_ROW_MAJOR=args.IS_B_ROW_MAJOR==1, # 1 if row major order
            total_row=args.total_row,
            total_col=args.total_col,
            r=args.r,
            s=args.s,
            t=args.t,
            ENABLE_AXI4=args.ENABLE_AXI4==1,
            IS_B_COL_MAJOR_S_T_ROW_MAJOR=args.IS_B_COL_MAJOR_S_T_ROW_MAJOR == 1, # 1 if B is col major, but s_t block is row major
                                                                                #NOTE: no affect if B is col major
            IS_BFLOAT16_OUT_TY=args.IS_BFLOAT16_OUT_TY==1,
            ADD_BIAS=args.ADD_BIAS,
            OUTPUT_MODE=args.OUTPUT_MODE
        )
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)
