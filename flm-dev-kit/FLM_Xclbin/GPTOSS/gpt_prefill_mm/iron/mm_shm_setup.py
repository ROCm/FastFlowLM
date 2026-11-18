# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from re import L
from termios import N_SLIP
from threading import stack_size
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.dialects.ext.arith import constant, index_cast
from sympy import true, use


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



def get_shim_A_B_C_str(col:int):
    return f"A_SHM_{col}",f"B_SHM_{col}",f"C_SHM_{col}", 









def generate_shimtile_sequence_per_k_block(shim_index:int, 
                               total_npu_row:int, total_npu_col:int,
                               mega_block_row_idx:int, mega_block_col_idx:int,
                               M_size:int, K_size:int, N_size:int,
                               m:int, k:int, n:int,
                               A:npt.NDArray, B:npt.NDArray, C:npt.NDArray,
                               list_A_shim_queue:list[int], list_B_shim_queue:list[int],
                               list_C_shim_queue:list[int],
                               list_A_bd_pingpong_flag:list[int],
                               list_B_bd_pingpong_flag:list[int],
                               list_C_bd_pingpong_flag:list[int],
                               IS_B_ROW_MAJOR:bool,
                               valid_A_MT_shimtile_index:list[int],
                               SEND_BIAS:int, 
                               ADD_BIAS:int,
                               ENABLE_AXI4:bool, VALID_COLUMN:bool=True
                               ):

    K_div_k = K_size//k
    A_str, B_str, C_str = get_shim_A_B_C_str(shim_index)
    
    #EACH A,B can use 6 id. Ping-pong but every 3 ids
    AXI_FLAG = 0b1110<<24 # write through read allocate
    DMA_BD_x_5 = 0x000001D014



    



    
    if shim_index in valid_A_MT_shimtile_index and  valid_A_MT_shimtile_index.index(shim_index)  < total_npu_row:
        if list_A_shim_queue[shim_index] == 2:
            npu_dma_wait(A_str)
            list_A_shim_queue[shim_index] -= 1        
        A_offset = mega_block_row_idx *(total_npu_row*m) * K_size
        A_offset += valid_A_MT_shimtile_index.index(shim_index) *(m*K_size)        
        # generate A sequence 
        if list_A_bd_pingpong_flag[shim_index] == 0:
            bd_id = 0
            list_A_bd_pingpong_flag[shim_index] = 1
        else:
            bd_id = 1
            list_A_bd_pingpong_flag[shim_index] = 0
        npu_dma_memcpy_nd(
            metadata=A_str,
            bd_id = bd_id,
            mem=A,
            offsets= [0,0,0,A_offset],
            sizes = [1,K_div_k, m,     k],
            strides=[0,k      , K_size,1],
            issue_token=True
        )
        list_A_shim_queue[shim_index] += 1
        if ENABLE_AXI4:
            # do a mask write to configure thte AXI property

            npu_maskwrite32(
                address= DMA_BD_x_5 + 32*bd_id, value=AXI_FLAG, mask=0xF000000, row=0, column=shim_index
            )
        else:
            npu_maskwrite32(
                address= DMA_BD_x_5 + 32*bd_id, value=0x0, mask=0xF000000, row=0, column=shim_index
            )
        
    if shim_index <  total_npu_col and VALID_COLUMN:
        
        if SEND_BIAS == 1:

            if list_B_shim_queue[shim_index] == 2:
                npu_dma_wait(B_str)
                list_B_shim_queue[shim_index] -= 1
            # now send the bias to it
            # assume BIAS is store in first N element of B matrix
            _BIAS_DATA_OFFSET = mega_block_col_idx * (total_npu_col*n) +  shim_index *n

            npu_dma_memcpy_nd(
                metadata=B_str,
                bd_id= 6,# reserved for bias
                mem=B,
                offsets=[0,0,0,_BIAS_DATA_OFFSET],
                sizes = [1,1, 1, k*n],
                strides=[0,0,0,1],
                issue_token=True
            )
            list_B_shim_queue[shim_index] += 1 # +1 for bias bd transaction

        if ADD_BIAS == 1:
            BIAS_OFFSET = N_size            
        else:
            BIAS_OFFSET = 0    
        
        
        if list_B_shim_queue[shim_index] == 2:
            npu_dma_wait(B_str)
            list_B_shim_queue[shim_index] -= 1
            
        if list_B_bd_pingpong_flag[shim_index] == 0:
            b_bd_id = 2
            list_B_bd_pingpong_flag[shim_index] = 1
        else:
            b_bd_id = 3
            list_B_bd_pingpong_flag[shim_index] = 0
            
            
        if IS_B_ROW_MAJOR:
            B_offset = mega_block_col_idx * (total_npu_col*n)
            B_offset += shim_index *n

            npu_dma_memcpy_nd(
                metadata=B_str,
                bd_id= b_bd_id,
                mem=B,
                offsets=[0,0,0,B_offset + BIAS_OFFSET],
                sizes = [1,K_div_k,   k     ,n],
                strides=[0,k* N_size, N_size,1],
                issue_token=True
            )
        else:
            B_offset = mega_block_col_idx * (total_npu_col*n) * K_size
            B_offset += shim_index *n*K_size

            npu_dma_memcpy_nd(
                metadata=B_str,
                bd_id= b_bd_id,
                mem=B,
                offsets=[0,0,0,B_offset + BIAS_OFFSET],
                sizes = [   1,K_div_k,       n,     k],
                strides=[   0,k,             K_size,1],
                issue_token=True
            )
            
        list_B_shim_queue[shim_index] += 1
        # if ENABLE_AXI4:
        #     # do a mask write to configure thte AXI property
        #     npu_maskwrite32(
        #         address= DMA_BD_x_5 + 32*b_bd_id, value=AXI_FLAG, mask=0xF000000, row=0, column=shim_index
        #     )
        # else:
        #     npu_maskwrite32(
        #         address= DMA_BD_x_5 + 32*b_bd_id, value=0x0, mask=0xF000000, row=0, column=shim_index
        #     )            
    C_offset  = mega_block_col_idx*(n*total_npu_col)
    C_offset += mega_block_row_idx*(m*total_npu_row)* N_size
    C_offset += shim_index*(n)
    valid_C=False
    if shim_index < total_npu_col:
        valid_C = True
    
    if valid_C and VALID_COLUMN:
        if list_C_shim_queue[shim_index]  == 2:
            npu_dma_wait(C_str)
            list_C_shim_queue[shim_index] -=1
        
            
        if list_C_bd_pingpong_flag[shim_index] == 0:
            list_C_bd_pingpong_flag[shim_index] = 1
            c_bd_id = 14
        else:
            list_C_bd_pingpong_flag[shim_index] = 0
            c_bd_id = 15

        npu_dma_memcpy_nd(
            metadata=C_str,
            bd_id=c_bd_id,
            mem=C,
            offsets=[0,0,0,C_offset],
            sizes = [1,1, 4*m,     n],
            strides=[0,0, N_size ,1],
            issue_token=True
        )
        # if ENABLE_AXI4:
        #     npu_maskwrite32(
        #         address= DMA_BD_x_5 + 32*c_bd_id, value=AXI_FLAG, mask=0xF000000, row=0, column=shim_index
        #     )     
        # else:
        #     npu_maskwrite32(
        #         address= DMA_BD_x_5 + 32*c_bd_id, value=0x0, mask=0xF000000, row=0, column=shim_index
        #     )             
        list_C_shim_queue[shim_index]+=1
    


# def generate_runtime_sequence(
#     A:npt.NDArray,
#     B:npt.NDArray,
#     C:npt.NDArray,
#     M_size:int, N_size:int, K_size:int,
#     m:int, n:int, k:int,
#     total_npu_row:int, total_npu_col:int,
#     list_A_shim_queue:list[int],
#     list_B_shim_queue:list[int],
#     list_C_shim_queue:List[int],
#     IS_B_ROW_MAJOR:bool,
#     valid_A_MT_shimtile_index:list[int],
#     ENABLE_AXI4:bool
    
# ):
#     M_div_num_row_m = M_size//(m*total_npu_row)
#     N_div_num_col_n = N_size//(n*total_npu_col)

#     N_div_num_col_n_remainder_blocks =  (N_size % (n*total_npu_col))// n


#     list_A_BD_pingpong_flag = [0]*max(total_npu_row, total_npu_col)
#     list_B_BD_pingpong_flag = [0]*max(total_npu_row, total_npu_col)
#     list_C_BD_pingpong_flag = [0]*max(total_npu_row, total_npu_col)
    
#     """C in row_major
#     [ num_row*mxnum_col*n_0,   num_row*mxnum_col*n_1, ....... repeat for N//(num_col*n) ]
#     .... repeat for M//(num_row*n)
#     """
    
#     """A in row_major
#     [ num_rowxmxk_0,  num_rowxmxk_1 ,.... repeat for K/k],
#     [ num_rowxmxk_k,  num_rowxmxk_k+1,.... repeat for K/k],
#     .. repeat for M/(num_row*m)
#     """
    

#     for mega_block_row_idx in range(M_div_num_row_m):
#         col_block_range = N_div_num_col_n
#         if N_div_num_col_n_remainder_blocks!= 0:
#             col_block_range += 1
#         for mega_block_col_idx in range(col_block_range):
#             for shim_index in range(max(total_npu_col, total_npu_row)):
#                 if N_div_num_col_n_remainder_blocks!= 0 and mega_block_col_idx == N_div_num_col_n:
#                     if shim_index < N_div_num_col_n_remainder_blocks:
#                         generate_shimtile_sequence_per_k_block(
#                             shim_index=shim_index, 
#                             total_npu_col=total_npu_col, total_npu_row=total_npu_row,
#                             mega_block_col_idx=mega_block_col_idx, mega_block_row_idx=mega_block_row_idx,
#                             M_size=M_size, K_size=K_size, N_size=N_size,
#                             m=m,k=k,n=n, 
#                             A=A,B=B, C=C, 
#                             list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
#                             list_C_shim_queue=list_C_shim_queue,
#                             list_A_bd_pingpong_flag=list_A_BD_pingpong_flag,
#                             list_B_bd_pingpong_flag=list_B_BD_pingpong_flag,
#                             list_C_bd_pingpong_flag=list_C_BD_pingpong_flag,
#                             IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
#                             valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
#                             ENABLE_AXI4=ENABLE_AXI4
#                         )                        
#                     else:
#                         generate_shimtile_sequence_per_k_block(
#                             shim_index=shim_index, 
#                             total_npu_col=total_npu_col, total_npu_row=total_npu_row,
#                             mega_block_col_idx=mega_block_col_idx, mega_block_row_idx=mega_block_row_idx,
#                             M_size=M_size, K_size=K_size, N_size=N_size,
#                             m=m,k=k,n=n, 
#                             A=A,B=B, C=C, 
#                             list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
#                             list_C_shim_queue=list_C_shim_queue,
#                             list_A_bd_pingpong_flag=list_A_BD_pingpong_flag,
#                             list_B_bd_pingpong_flag=list_B_BD_pingpong_flag,
#                             list_C_bd_pingpong_flag=list_C_BD_pingpong_flag,                            
#                             IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
#                             valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
#                             ENABLE_AXI4=ENABLE_AXI4, VALID_COLUMN=False
#                         )
#                 else:
#                     generate_shimtile_sequence_per_k_block(
#                         shim_index=shim_index, 
#                         total_npu_col=total_npu_col, total_npu_row=total_npu_row,
#                         mega_block_col_idx=mega_block_col_idx, mega_block_row_idx=mega_block_row_idx,
#                         M_size=M_size, K_size=K_size, N_size=N_size,
#                         m=m,k=k,n=n, 
#                         A=A,B=B, C=C, 
#                         list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
#                         list_C_shim_queue=list_C_shim_queue,
#                         list_A_bd_pingpong_flag=list_A_BD_pingpong_flag,
#                         list_B_bd_pingpong_flag=list_B_BD_pingpong_flag,
#                         list_C_bd_pingpong_flag=list_C_BD_pingpong_flag,
#                         IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
#                         valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
#                         ENABLE_AXI4=ENABLE_AXI4
#                     )
                    

def generate_runtime_sequence(
    A:npt.NDArray,
    B:npt.NDArray,
    C:npt.NDArray,
    M_size:int, N_size:int, K_size:int,
    m:int, n:int, k:int,
    total_npu_row:int, total_npu_col:int,
    list_A_shim_queue:list[int],
    list_B_shim_queue:list[int],
    list_C_shim_queue:List[int],
    IS_B_ROW_MAJOR:bool,
    valid_A_MT_shimtile_index:list[int],
    ADD_BIAS:int,
    ENABLE_AXI4:bool
    
):
    M_div_num_row_m = M_size//(m*total_npu_row)
    N_div_num_col_n = N_size//(n*total_npu_col)

    N_div_num_col_n_remainder_blocks =  (N_size % (n*total_npu_col))// n


    list_A_BD_pingpong_flag = [0]*max(total_npu_row, total_npu_col)
    list_B_BD_pingpong_flag = [0]*max(total_npu_row, total_npu_col)
    list_C_BD_pingpong_flag = [0]*max(total_npu_row, total_npu_col)
    
    """C in row_major
    [ num_row*mxnum_col*n_0,   num_row*mxnum_col*n_1, ....... repeat for N//(num_col*n) ]
    .... repeat for M//(num_row*n)
    """
    
    """A in row_major
    [ num_rowxmxk_0,  num_rowxmxk_1 ,.... repeat for K/k],
    [ num_rowxmxk_k,  num_rowxmxk_k+1,.... repeat for K/k],
    .. repeat for M/(num_row*m)
    """
    
    # --- Start of changes ---

    # 1. Calculate col_block_range *once* before the loops
    col_block_range = N_div_num_col_n
    if N_div_num_col_n_remainder_blocks != 0:
        col_block_range += 1

    # 2. Swapped loop order: mega_block_col_idx is now the outer loop
    for mega_block_col_idx in range(col_block_range):
        for mega_block_row_idx in range(M_div_num_row_m):
            for shim_index in range(max(total_npu_col, total_npu_row)):
                
                # 3. The inner logic is unchanged, as it only depends on 
                #    mega_block_col_idx and shim_index
                
                # only send bias block if mega_block_row_idx == 0
                if mega_block_row_idx ==0 and ADD_BIAS == 1:
                    SEND_ADD_BIAS = 1
                else:
                    SEND_ADD_BIAS = 0
                if N_div_num_col_n_remainder_blocks != 0 and mega_block_col_idx == N_div_num_col_n:
                    if shim_index < N_div_num_col_n_remainder_blocks:
                        generate_shimtile_sequence_per_k_block(
                            shim_index=shim_index, 
                            total_npu_col=total_npu_col, total_npu_row=total_npu_row,
                            mega_block_col_idx=mega_block_col_idx, mega_block_row_idx=mega_block_row_idx,
                            M_size=M_size, K_size=K_size, N_size=N_size,
                            m=m,k=k,n=n, 
                            A=A,B=B, C=C, 
                            list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
                            list_C_shim_queue=list_C_shim_queue,
                            list_A_bd_pingpong_flag=list_A_BD_pingpong_flag,
                            list_B_bd_pingpong_flag=list_B_BD_pingpong_flag,
                            list_C_bd_pingpong_flag=list_C_BD_pingpong_flag,
                            IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
                            valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
                            SEND_BIAS=SEND_ADD_BIAS,
                            ADD_BIAS=ADD_BIAS,
                            ENABLE_AXI4=ENABLE_AXI4
                        )                        
                    else:
                        generate_shimtile_sequence_per_k_block(
                            shim_index=shim_index, 
                            total_npu_col=total_npu_col, total_npu_row=total_npu_row,
                            mega_block_col_idx=mega_block_col_idx, mega_block_row_idx=mega_block_row_idx,
                            M_size=M_size, K_size=K_size, N_size=N_size,
                            m=m,k=k,n=n, 
                            A=A,B=B, C=C, 
                            list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
                            list_C_shim_queue=list_C_shim_queue,
                            list_A_bd_pingpong_flag=list_A_BD_pingpong_flag,
                            list_B_bd_pingpong_flag=list_B_BD_pingpong_flag,
                            list_C_bd_pingpong_flag=list_C_BD_pingpong_flag,                            
                            IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
                            valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
                            SEND_BIAS=SEND_ADD_BIAS,
                            ADD_BIAS=ADD_BIAS,
                            ENABLE_AXI4=ENABLE_AXI4, VALID_COLUMN=False
                        )
                else:
                    generate_shimtile_sequence_per_k_block(
                        shim_index=shim_index, 
                        total_npu_col=total_npu_col, total_npu_row=total_npu_row,
                        mega_block_col_idx=mega_block_col_idx, mega_block_row_idx=mega_block_row_idx,
                        M_size=M_size, K_size=K_size, N_size=N_size,
                        m=m,k=k,n=n, 
                        A=A,B=B, C=C, 
                        list_A_shim_queue=list_A_shim_queue, list_B_shim_queue=list_B_shim_queue,
                        list_C_shim_queue=list_C_shim_queue,
                        list_A_bd_pingpong_flag=list_A_BD_pingpong_flag,
                        list_B_bd_pingpong_flag=list_B_BD_pingpong_flag,
                        list_C_bd_pingpong_flag=list_C_BD_pingpong_flag,
                        IS_B_ROW_MAJOR=IS_B_ROW_MAJOR,
                        valid_A_MT_shimtile_index=valid_A_MT_shimtile_index,
                        SEND_BIAS=SEND_ADD_BIAS,
                        ADD_BIAS=ADD_BIAS,
                        ENABLE_AXI4=ENABLE_AXI4
                    )
    # --- End of changes ---