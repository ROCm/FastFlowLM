# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from re import L
from threading import stack_size
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.extras.context import mlir_mod_ctx
from pyparsing import C, col


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

from GateDeltaNet_prefill_CT import setup_CT
from GateDeltaNet_prefill_MT import setup_MT
from GateDeltaNet_prefill_SHM import (
    get_S_IN_bd_id, get_alpha_bd_id, get_beta_bd_id, get_q_k_bd_id, get_k_bd_id, get_v_bd_id, get_output_o_s_bd_id,
    wait_DMA_queue_if_full, increment_DMA_queue, get_SHM_buffer_str_list,force_wait_DMA_queue,update_ping_pong
    
)

import argparse



def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def gateDeltaNetPrefill(
    NUM_HEAD:int,QK_HEAD_GROUP:int, 
    L:int, MT_L_size:int, CT_L_size:int,
    DQ:int, DK:int, DV:int, 
    NUM_CT_PER_COLUMN:int, NUM_OF_COLUMN:int,     
    QKV_bffer_L_head_row_major:int,  
    arch="npu2", trace_size:int=0,             
    ):
 
    trace_size = 2048*1024 # in byte
    
    
            
    QKV_buffer_row_major = True if QKV_bffer_L_head_row_major==1 else False
    
        
 
    
         
    

    

    
    
    
    assert DK == DQ
    
    

    
    
    DV_chunk_per_CT = DV//NUM_CT_PER_COLUMN
    assert DV_chunk_per_CT%NUM_CT_PER_COLUMN == 0    
    
    assert DV==DK
    Q_K_V_size_bf16 = DQ + 2  # Plus two because each Q,K, V(padd) follow a float scalar(alpha, beta)     
    
    dtype_S_chunk  = np.dtype[np.float32] # partial S in each CT
    dtype_Q_K_V = np.dtype[bfloat16]
    dtype_O = dtype_Q_K_V

    
    

    
    assert MT_L_size % CT_L_size == 0
    
    
    
    
    
    @device(AIEDevice.npu2)
    def device_body():
        stack_size = 8192


        Q_K_V_size_bf16_align_32 = round_up(Q_K_V_size_bf16,32 )

        assert DV_chunk_per_CT*NUM_CT_PER_COLUMN == DV
        CT_S_chunk_len_fp32 = DV_chunk_per_CT * DK
        CT_per_Q_K_V_len =CT_L_size* Q_K_V_size_bf16_align_32
                
        CT_per_S_out_len_in_bf16 = CT_S_chunk_len_fp32 *2 # times 2 because S is float32 and O is bf16 
        CT_per_O_out_len_in_bf16 = CT_L_size*DV_chunk_per_CT # for simplicity, we use the same buffer for S and O (ping-pong), so the size should be the same (and we take the max of the two possible sizes,
        # because we want to use same buffer for S and O, and O chunk size can be different from S chunk size, we take the max.)
        
        assert CT_per_S_out_len_in_bf16 % CT_per_O_out_len_in_bf16 == 0
        
        
        
        CT_Q_K_V_in_ty = np.ndarray[ (CT_per_Q_K_V_len,), dtype_Q_K_V]        
        
        CT_S_in_ty = np.ndarray[  (CT_S_chunk_len_fp32,), dtype_S_chunk]
        CT_output_bf16_ty = np.ndarray[ (CT_per_O_out_len_in_bf16,), dtype_O]
        
        
        ping_pong_rtp_len = 32
        ping_pong_ty = np.ndarray[(ping_pong_rtp_len,), np.dtype[np.int32]]
        rtp_config_ty = np.ndarray[(ping_pong_rtp_len,), np.dtype[np.int32]]
        
        link_o_name = f"gate_delta_net_prefill_{DQ}_{DK}_{DV}_{DV_chunk_per_CT}_{CT_L_size}_{MT_L_size}.o"
        gate_delta_net_prefill_function_name = f"gate_delta_net_prefill_{DQ}_{DK}_{DV}_{DV_chunk_per_CT}_{CT_L_size}_{MT_L_size}"  
        
        
        delta_net_prefill_function = external_func(
            name=gate_delta_net_prefill_function_name,
            inputs = [
                CT_S_in_ty,   
                CT_Q_K_V_in_ty, CT_Q_K_V_in_ty,  # Q ping-pong
                CT_Q_K_V_in_ty, CT_Q_K_V_in_ty, # K ping-pong
                CT_Q_K_V_in_ty, CT_Q_K_V_in_ty, # V ping-pong
                CT_output_bf16_ty, CT_output_bf16_ty, # output/S_chunk ping-pong
                np.int32, # Q_K_V_size_bf16_align_32
                np.int32, ping_pong_ty,  # DV_chunk_idx, ping-pong buffer for synchronization,
                rtp_config_ty, # rtp config buffer
                
                np.int32, np.int32, # S_in prod, con lock
                np.int32, np.int32, # Q prod, con lock
                np.int32, np.int32, # K prod, con lock
                np.int32, np.int32, # V prod, con lock
                np.int32, np.int32 # output/S_out_chunk prod, con lock
            ]
            
        )
        zero_int_func = external_func("zero_int32_array", inputs=[
            ping_pong_ty, np.int32
        ])

        
        CT_columns = []
        CT_rtp_lock_columns = []
        CT_rtp_buffer_str_columns = []
        MT_columns = []
        SHM_columns = []
            
            
        for col_idx in range(NUM_OF_COLUMN):

            CT_per_column = []
            CT_rtp_buffer_str_per_column = []
            CT_rtp_lock_per_column = []
                        
            for row in range(NUM_CT_PER_COLUMN):
                CT_core, CT_lock, CT_rtp_str = setup_CT(
                    row_idx= 2+row, col_idx =col_idx ,stack_size=stack_size,
                    delta_net_prefill_function=delta_net_prefill_function, zero_int_func=zero_int_func,
                    link_o_name=link_o_name,
                    CT_Q_K_V_in_ty = CT_Q_K_V_in_ty, CT_S_in_ty=CT_S_in_ty,
                    CT_output_bf16_ty=CT_output_bf16_ty,
                    ping_pong_ty=ping_pong_ty, rtp_config_ty=rtp_config_ty,
                    ping_pong_rtp_len=ping_pong_rtp_len, CT_L_size=CT_L_size,
                    actual_valid_qkv_data_size_per_L=Q_K_V_size_bf16,
                    padded_qkv_data_size_per_L=Q_K_V_size_bf16_align_32
                    
                )
                
                CT_per_column.append(CT_core)
                CT_rtp_lock_per_column.append(CT_lock)
                CT_rtp_buffer_str_per_column.append(CT_rtp_str)

            
            CT_columns.append(CT_per_column)
            CT_rtp_lock_columns.append(CT_rtp_lock_per_column)
            CT_rtp_buffer_str_columns.append(CT_rtp_buffer_str_per_column)
                
                
        
        

        MT_S_in_size_fp32 = DK* DV
        #MT need to define the q, k, v with consider of alpha & beta      
        
         
        MT_qkv_alpha_beta_in_size_bf16 =MT_L_size* (3*Q_K_V_size_bf16)  # times 3 for q, k, v
        MT_Output_size_bf16 = DV * MT_L_size        


        NUM_MT_O_CHUNK_FOR_S_output =  (MT_S_in_size_fp32* 4)// (MT_Output_size_bf16 * 2) 
        print(f"NUM_MT_O_CHUNK_FOR_S_output: {NUM_MT_O_CHUNK_FOR_S_output}", file=sys.stderr)
        assert   (MT_S_in_size_fp32* 4) % (MT_Output_size_bf16 * 2) == 0 # assert S_output is multiple of MT_OUTPUT buffer
        
        
        MT_S_in_block_ty = np.ndarray[ ( MT_S_in_size_fp32,), dtype_S_chunk]
        MT_Output_ty = np.ndarray[(MT_Output_size_bf16,), dtype_O ]
        MT_qkv_alpha_beta_in_ty = np.ndarray[  (MT_qkv_alpha_beta_in_size_bf16,), dtype_Q_K_V]

        for col_idx in range(NUM_OF_COLUMN):
            MT_core = setup_MT(
                col_idx=col_idx,
                Q_K_V_size_bf16=Q_K_V_size_bf16,
                MT_L_size=MT_L_size,
                CT_L_size=CT_L_size,
                MT_S_in_size_fp32=MT_S_in_size_fp32,
                MT_Output_size_bf16=MT_Output_size_bf16,
                MT_qkv_alpha_beta_in_size_bf16=MT_qkv_alpha_beta_in_size_bf16,
                CT_S_chunk_len_fp32=CT_S_chunk_len_fp32,
                MT_S_in_block_ty=MT_S_in_block_ty,
                MT_Output_ty=MT_Output_ty,
                MT_qkv_alpha_beta_in_ty=MT_qkv_alpha_beta_in_ty,
                NUM_CT_PER_COLUMN=NUM_CT_PER_COLUMN,
                DQ=DQ, DK=DK, DV=DV,
                DV_chunk_per_CT=DV_chunk_per_CT
            )
            MT_columns.append(MT_core)
        
        
    
        for SHIM_COL_IDX in range(NUM_OF_COLUMN):
            shimtile_core = tile(col=SHIM_COL_IDX,row= 0) 
            SHM_columns.append(shimtile_core)
            
            S_IN_SHM_Str, QKV_ALPHA_BETA_IN_SHM_Str, O_S_OUTPUT_SHM_Str = get_SHM_buffer_str_list(SHIM_COL_IDX)
            
            
            
                
            memref.global_(S_IN_SHM_Str, T.memref(DK*DV, T.i32()), sym_visibility="public")
            memref.global_(QKV_ALPHA_BETA_IN_SHM_Str, T.memref(Q_K_V_size_bf16*3, T.bf16()  ), sym_visibility="public")
            memref.global_(O_S_OUTPUT_SHM_Str, T.memref(Q_K_V_size_bf16, T.bf16()  ), sym_visibility="public")

            shim_dma_allocation(S_IN_SHM_Str, DMAChannelDir.MM2S, 0, SHIM_COL_IDX)
            shim_dma_allocation(QKV_ALPHA_BETA_IN_SHM_Str, DMAChannelDir.MM2S, 1, SHIM_COL_IDX)
            shim_dma_allocation(O_S_OUTPUT_SHM_Str, DMAChannelDir.S2MM, 0, SHIM_COL_IDX)
            
            
            
        for i in range(NUM_OF_COLUMN):
            # send the flow
            flow(SHM_columns[i], WireBundle.DMA, 0, MT_columns[i], WireBundle.DMA, 0)
            flow(SHM_columns[i], WireBundle.DMA, 1, MT_columns[i], WireBundle.DMA, 5) # qkv alpha beta
            flow(MT_columns[i], WireBundle.DMA, 0, SHM_columns[i], WireBundle.DMA, 0) # S, O output
            
        for col_idx in range(NUM_OF_COLUMN):
            MT_core = MT_columns[col_idx]
            for row_idx in range(NUM_CT_PER_COLUMN):
                # broacast alpha beta to all CT in the column
                flow(MT_core, WireBundle.DMA, 5, CT_columns[col_idx][row_idx], WireBundle.DMA, 1) # qkv alpha beta

                # send S to all CT in the column in through each individual port flow
                flow(MT_core, WireBundle.DMA, row_idx+1, CT_columns[col_idx][row_idx], WireBundle.DMA, 0) # S to CT, offset by 1 because 0 is used for alpha beta
                
                # receive S/O from CT through each individual port flow
                flow(CT_columns[col_idx][row_idx], WireBundle.DMA, 0, MT_core, WireBundle.DMA, row_idx+1) # S/O from CT, offset
                

        assert NUM_HEAD % QK_HEAD_GROUP == 0    
        NUM_Q_K_HEAD = NUM_HEAD // QK_HEAD_GROUP


        # NOTE: 
        # assume S is shape of  NUM_HEAD x (DVx DK), where each DVxDK is in column major order
        
        #if QKV_buffer_row_major: 
        # assume QKV_in is shape of L x  (NUM_Q_K_HEAD x DQ+ NUM_Q_K_HEAD x DK + NUM_HEADx DV), and the entire buffer is in row major oder
        # else
        # QKV_in is shape of   NUM_Q_K_HEAD * (LxDQ) + NUM_Q_K_HEAD * (LxDK) + NUM_HEAD * (LxDV) in row major order, where each LxDQ, LxDK, LxDV is in column major order.
        
        
        
        # assume alpha_beta is in shape of [2 , LxNUM_HEAD] in row-major order that is appened to the tail of QKV buffer1111

        # where first L x NUM_HEAD is alpha in row major order and second L x NUM_HEAD is beta in row major order
        L_PADDED = round_up(L, MT_L_size)
        qkv_buffer_row_stride = NUM_Q_K_HEAD* DQ + NUM_Q_K_HEAD* DK + NUM_HEAD*DV
        alpha_beta_buffer_row_stride = NUM_HEAD        
        o_buffer_row_stride = NUM_HEAD *DV
        #NOTE: be careful the reorder of receivce O and S as output
        # @runtime_sequence(
        #     np.ndarray[(NUM_HEAD*MT_S_in_size_fp32,), dtype_S_chunk],
        #     np.ndarray[( L_PADDED * (qkv_buffer_row_stride),), dtype_Q_K_V],
        #     np.ndarray[( (L_PADDED* o_buffer_row_stride),), dtype_O]
        # )
        # def sequence(S_in_out, QKV_in, O_out):
            
        #     # for now, simply test the single head 
            
            
        #     s_in_BD_ping_pong =  [0 for _ in range(NUM_OF_COLUMN)]
        #     alpha_beta_qkv_BD_ping_pong = [0 for _ in range(NUM_OF_COLUMN)]
        #     output_BD_ping_pong = [0 for _ in range(NUM_OF_COLUMN)]

        #     s_in_bd_queue_counter = [0 for _ in range(NUM_OF_COLUMN)]
        #     alpha_beta_qkv_queue_counter = [0 for _ in range(NUM_OF_COLUMN)]
        #     output_bd_queue_counter = [0 for _ in range(NUM_OF_COLUMN)]
            
            
            
        #     # # first, write the rtp value 
        #     # # for now, assume simplicity 
            
        #     #NOTE: potentially work for non-divisiable 
        #     assert NUM_HEAD % NUM_OF_COLUMN == 0
        #     HEAD_PER_COLUMN = NUM_HEAD // NUM_OF_COLUMN
            
            
            
            
        #     for col_idx in range(NUM_OF_COLUMN):
        #         for row_idx in range(NUM_CT_PER_COLUMN):
        #             npu_rtp_write( CT_rtp_buffer_str_columns[col_idx][row_idx], 0, HEAD_PER_COLUMN )
        #             npu_rtp_write( CT_rtp_buffer_str_columns[col_idx][row_idx],1, L)
        #             set_lock(CT_rtp_lock_columns[col_idx][row_idx], 1)        
        #     num_bf16_for_float = 2
            

       
            
        #     head_idx_per_transaction = []
            
        #     cur_transation_head_idx_list:list[tuple[int, int]] = []
        #     for _h_idx in range(0, NUM_HEAD, QK_HEAD_GROUP):
        #         for group_idx in range(QK_HEAD_GROUP):
        #             g_head_idx = _h_idx//QK_HEAD_GROUP 
        #             head_idx = _h_idx + group_idx

        #             if len(cur_transation_head_idx_list) == NUM_OF_COLUMN:
        #                 head_idx_per_transaction.append(cur_transation_head_idx_list)
        #                 cur_transation_head_idx_list = []
        #             cur_transation_head_idx_list.append((g_head_idx, head_idx))
        #     if len(cur_transation_head_idx_list) > 0:
        #         head_idx_per_transaction.append(cur_transation_head_idx_list) # append the last
                    


        #     for transation_head_idx_list in head_idx_per_transaction:
        #         for cur_col_idx, (g_head_idx, head_idx) in enumerate(iterable=transation_head_idx_list):

        #             S_offset = head_idx * MT_S_in_size_fp32   
                    
        #             col_S_in_Str, col_qkv_a_b_in_Str, col_o_str= get_SHM_buffer_str_list(cur_col_idx)
                    
        #             # send the S buffer
        #             # NOTE: assume S is already in DVxDK column major order
                    
        #             wait_DMA_queue_if_full(
        #                 cur_shim_idx=cur_col_idx,
        #                 SHIM_queue_counter=s_in_bd_queue_counter,
        #                 cur_shim_str=col_S_in_Str
        #             )
        #             npu_dma_memcpy_nd(
        #                 metadata=col_S_in_Str,
        #                 bd_id= get_S_IN_bd_id(s_in_BD_ping_pong[cur_col_idx]),
        #                 mem=S_in_out,
        #                 offsets = [0,0,0,S_offset],
        #                 sizes = [1,1,1, MT_S_in_size_fp32],    # assume S is already column major order
        #                 strides=[0,0,0,1],
        #                 issue_token=True
        #             )
        #             s_in_BD_ping_pong[cur_col_idx] = update_ping_pong(s_in_BD_ping_pong[cur_col_idx])
        #             increment_DMA_queue(
        #                 cur_shim_idx=cur_col_idx,
        #                 SHIM_queue_counter = s_in_bd_queue_counter
        #             )                
                    
    
        #             # now, receive O from the shimtile
        #             wait_DMA_queue_if_full(
        #                 cur_shim_idx=cur_col_idx,
        #                 SHIM_queue_counter=output_bd_queue_counter,
        #                 cur_shim_str=col_o_str
        #             )                        
        #             o_offset = head_idx * DV 
        #             npu_dma_memcpy_nd(
        #                 metadata=col_o_str,
        #                 bd_id=get_output_o_s_bd_id(output_BD_ping_pong[cur_col_idx]),
        #                 mem=O_out,
        #                 offsets=[0,0,0,o_offset],
        #                 sizes=[L_PADDED//MT_L_size,                               NUM_CT_PER_COLUMN,     MT_L_size,             DV_chunk_per_CT],
        #                 strides=[MT_L_size*o_buffer_row_stride , DV_chunk_per_CT,   o_buffer_row_stride,                 1],
        #                 issue_token=True
        #             )
        #             output_BD_ping_pong[cur_col_idx] = update_ping_pong(output_BD_ping_pong[cur_col_idx])
        #             increment_DMA_queue(
        #                 cur_shim_idx=cur_col_idx,
        #                 SHIM_queue_counter = output_bd_queue_counter
        #             )
                    
                    
        #             # receive S
        #             MT_output_size_per_CT_bf16 = MT_Output_size_bf16// NUM_CT_PER_COLUMN        
        #             DK_per_S_chunk_per_CT =   (MT_output_size_per_CT_bf16//2 ) //DV_chunk_per_CT  # div 2 because S is float32
        #             wait_DMA_queue_if_full(
        #                 cur_shim_idx=cur_col_idx,
        #                 SHIM_queue_counter=output_bd_queue_counter,
        #                 cur_shim_str=col_o_str
        #             )
                    
                    
        #             npu_dma_memcpy_nd(
        #                 metadata=col_o_str,
        #                 bd_id=get_output_o_s_bd_id(output_BD_ping_pong[cur_col_idx]),
        #                 mem=S_in_out,
        #                 offsets=[0,0,0,S_offset],
        #                 sizes = [NUM_MT_O_CHUNK_FOR_S_output,                         NUM_CT_PER_COLUMN,DK_per_S_chunk_per_CT, DV_chunk_per_CT],
        #                 strides=[DK_per_S_chunk_per_CT*DV,  DV_chunk_per_CT, DV, 1],
        #                 issue_token=True
        #             )
        #             output_BD_ping_pong[cur_col_idx] = update_ping_pong(output_BD_ping_pong[cur_col_idx])
        #             increment_DMA_queue(
        #                 cur_shim_idx=cur_col_idx,
        #                 SHIM_queue_counter = output_bd_queue_counter
        #             )                                        
        #         beta_buffer_offset_bf16 =  num_bf16_for_float* L_PADDED* alpha_beta_buffer_row_stride # times 2 because two bfloat16 for 1 float
        #         # Interleave QKV transfers across columns (L-chunk outer, column inner)
        #         # so all columns stay fed in parallel instead of serializing per column.
        #         for cur_L in range(0,L_PADDED, MT_L_size):
        #             for cur_col_idx, (g_head_idx, head_idx) in enumerate(transation_head_idx_list):
        #                 col_S_in_Str, col_qkv_a_b_in_Str, col_o_str= get_SHM_buffer_str_list(cur_col_idx)
                        
        #                 if QKV_buffer_row_major:
        #                     q_offset = g_head_idx * DQ + cur_L* qkv_buffer_row_stride
        #                     k_offset = g_head_idx * DK + (NUM_Q_K_HEAD* DQ) + cur_L* qkv_buffer_row_stride
        #                     v_offset = head_idx * DV + (NUM_Q_K_HEAD* DQ) + (NUM_Q_K_HEAD* DK) + cur_L* qkv_buffer_row_stride
                            
        #                 else:
        #                     q_offset = g_head_idx *(L_PADDED * DQ) +  cur_L * DQ
        #                     k_offset = g_head_idx *(L_PADDED * DK) +  cur_L * DK +  ( NUM_Q_K_HEAD* L_PADDED * DQ)
        #                     v_offset = head_idx * (L_PADDED * DV) + cur_L * DV + ( NUM_Q_K_HEAD* L_PADDED * DQ) + (NUM_Q_K_HEAD* L_PADDED * DK)
                            
                            
                            
                            
        #                 a_offset_bf16 = L_PADDED*(DQ*NUM_Q_K_HEAD + DV*NUM_Q_K_HEAD+ DV*NUM_HEAD)+ \
        #                     num_bf16_for_float*(cur_L* alpha_beta_buffer_row_stride + head_idx) # times 2 because two bfloat16 for 1 float
        #                 b_offset_bf16 = beta_buffer_offset_bf16 + a_offset_bf16
                        

                        
                        
        #                 alpha_bd_id = get_alpha_bd_id(alpha_beta_qkv_BD_ping_pong[cur_col_idx])
        #                 q_bd_id = get_q_k_bd_id(alpha_beta_qkv_BD_ping_pong[cur_col_idx])
        #                 v_bd_id = get_v_bd_id(alpha_beta_qkv_BD_ping_pong[cur_col_idx])
        #                 wait_DMA_queue_if_full(
        #                     cur_shim_idx=cur_col_idx,
        #                     SHIM_queue_counter=alpha_beta_qkv_queue_counter,
        #                     cur_shim_str=col_qkv_a_b_in_Str
        #                 )
                        

        #                 npu_dma_memcpy_nd(
        #                     metadata=col_qkv_a_b_in_Str,
        #                     bd_id=alpha_bd_id,
        #                     mem=QKV_in,
        #                     offsets = [0,0,0,a_offset_bf16],
        #                     sizes = [  1,2,             MT_L_size,                                   1*num_bf16_for_float],
        #                     strides = [0,beta_buffer_offset_bf16,alpha_beta_buffer_row_stride*num_bf16_for_float,1],
        #                     issue_token=False
        #                 )

        #                 if QKV_buffer_row_major:
                
        #                     # send Q-K
        #                     assert DQ== DK
        #                     npu_dma_memcpy_nd(
        #                         metadata=col_qkv_a_b_in_Str,
        #                         bd_id=q_bd_id,
        #                         mem=QKV_in,
        #                         offsets = [0,0,0,q_offset],
        #                         sizes = [1,2,                 MT_L_size, DQ],
        #                         strides=[0,(NUM_Q_K_HEAD* DQ),qkv_buffer_row_stride,1],
        #                         issue_token=False
        #                     )
                            

        #                     # send V 
        #                     npu_dma_memcpy_nd(
        #                         metadata=col_qkv_a_b_in_Str,
        #                         bd_id=v_bd_id,
        #                         mem=QKV_in,
        #                         offsets=[0,0,0,v_offset],
        #                         sizes = [1,1, MT_L_size, DV],
        #                         strides=[0,0,qkv_buffer_row_stride,1],
        #                         issue_token=True
        #                     )
        #                 else:
        #                     assert DQ== DK
        #                     npu_dma_memcpy_nd(
        #                         metadata=col_qkv_a_b_in_Str,
        #                         bd_id=q_bd_id,
        #                         mem=QKV_in,
        #                         offsets = [0,0,0,q_offset],
        #                         sizes = [1,  2,                       MT_L_size,        DQ],
        #                         strides=[0,  NUM_Q_K_HEAD*L_PADDED*DQ,  DQ,             1],
        #                         issue_token=False
        #                     )
        #                     npu_dma_memcpy_nd(
        #                         metadata=col_qkv_a_b_in_Str,
        #                         bd_id=v_bd_id,
        #                         mem=QKV_in,
        #                         offsets=[0,0,0,v_offset],
        #                         sizes = [1,1,1,            DV*MT_L_size],
        #                         strides=[0,0,0,1],
        #                         issue_token=True
        #                     )
        #                 alpha_beta_qkv_BD_ping_pong[cur_col_idx] = update_ping_pong(alpha_beta_qkv_BD_ping_pong[cur_col_idx])
        #                 increment_DMA_queue(
        #                     cur_shim_idx=cur_col_idx,
        #                     SHIM_queue_counter=alpha_beta_qkv_queue_counter
        #                 )
    
            
        #     # now, force to wait for all the queue remainig 
        #     max_remain_queue_number = -1
        #     for col_idx in range(NUM_OF_COLUMN):
        #         max_remain_queue_number = max(max_remain_queue_number, s_in_bd_queue_counter[col_idx])
        #         max_remain_queue_number = max(max_remain_queue_number, alpha_beta_qkv_queue_counter[col_idx])

        #         max_remain_queue_number = max(max_remain_queue_number, output_bd_queue_counter[col_idx])
            
            
            
        #     for _ in range(max_remain_queue_number):
        #         for col_idx in range(NUM_OF_COLUMN):
        #             col_S_in_Str, col_qkv_a_b_in_Str, col_o_str= get_SHM_buffer_str_list(col_idx)
        #             force_wait_DMA_queue(
        #                 cur_shim_idx=col_idx,
        #                 SHIM_queue_counter=s_in_bd_queue_counter,
        #                 cur_shim_str=col_S_in_Str
        #             )
        #             force_wait_DMA_queue(
        #                 cur_shim_idx=col_idx,
        #                 SHIM_queue_counter=alpha_beta_qkv_queue_counter,
        #                 cur_shim_str=col_qkv_a_b_in_Str
        #             )                 
        #             force_wait_DMA_queue(
        #                 cur_shim_idx=col_idx,
        #                 SHIM_queue_counter=output_bd_queue_counter,
        #                 cur_shim_str=col_o_str
        #             )
                        
                    
            
            
                


with mlir_mod_ctx() as ctx:          
    
    parser = argparse.ArgumentParser(description="Run matmul with command line arguments.")

    parser.add_argument("arch",type=str)
    parser.add_argument("trace_size", type=int)    
    parser.add_argument("L", type=int)
    parser.add_argument("DQ", type=int)
    parser.add_argument("DK", type=int)
    parser.add_argument("DV", type=int)
    parser.add_argument("NUM_OF_ROW", type=int)
    parser.add_argument("NUM_OF_COLUMN", type=int)
    parser.add_argument("MT_L_chunk_size", type=int)
    parser.add_argument("CT_L_chunk_size", type=int)
    parser.add_argument("NUM_HEAD", type=int)
    parser.add_argument("QK_HEAD_GROUP", type=int)    
    parser.add_argument("QKV_buffer_L_head_row_major", type=int)
    args = parser.parse_args()
    gateDeltaNetPrefill(
        arch=args.arch,
        trace_size=args.trace_size,
        L=args.L,
        DQ=args.DQ,
        DK=args.DK,
        DV=args.DV,
        NUM_CT_PER_COLUMN=args.NUM_OF_ROW,
        NUM_OF_COLUMN=args.NUM_OF_COLUMN,
        MT_L_size=args.MT_L_chunk_size,
        CT_L_size=args.CT_L_chunk_size,
        NUM_HEAD=args.NUM_HEAD,
        QK_HEAD_GROUP=args.QK_HEAD_GROUP,
        QKV_bffer_L_head_row_major=args.QKV_buffer_L_head_row_major
    )
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)