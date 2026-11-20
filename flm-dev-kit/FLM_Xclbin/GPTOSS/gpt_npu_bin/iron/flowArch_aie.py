# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import ValuesView
from matplotlib.pylab import uint
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
import struct
from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref
from typing import Dict, Any, List, Optional
from aie.extras import types as T

from utils import dma_mm2s_ch, dma_s2mm_ch, flow_mm2s_ch, flow_s2mm_ch, connect_dma, generate_tile_map, mlir_context

from utils import dma_port, connect_port
from utils import ENABLE_CHESSCC

from rms_residual import rms_residual_tile
from proj_assemble import proj_assemble
from proj_tile import mvm_column
from RoPE import RoPE
from remove_padd import remove_zero_pad_tile
from router import moe_router_setup
from aux_memtile import aux_memtile
from attn import attn_tile, attn_memtile
import numpy as np
def flow_arch(model_type: str = "GPTOSS_20B"):
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    pass
    if model_type == "GPTOSS_20B":
        #TODO: maybe share a common config file
        D = 2880 #HIDDEN_SIZE
        INTERMEDIATE_SIZE=2880

        Q4NX_ROW = 32
        Q4NX_COL = 128
        Q4NX_ROW_TILE = 16
        Q4NX_BLOCK_SIZE =  Q4NX_ROW*Q4NX_COL
        #NOTE: both MXFP4 and Q4_1 block are being padded to same size
        #in Q4_1, 2 byte of scale, 1 byte of zero for every 32 value
        _byte_size = (Q4NX_ROW * (Q4NX_COL//32 ) )*4 + (Q4NX_BLOCK_SIZE//2) # div2 because each value is 4bit
        Q4NX_BLOCK_SIZE_bf16 =  _byte_size//2 # div 2 because bf16 is 2 byte
        
        MVM_CT_CORE = 16
        D_PADDED_Q4NX_COL =   (D +(Q4NX_COL-1) )//(Q4NX_COL)  * Q4NX_COL
        INTERMEDIATE_PADDED_Q4NX_COL = (INTERMEDIATE_SIZE +(Q4NX_COL-1) )//(Q4NX_COL)  * Q4NX_COL
        
        D_PADDED_FOUR_Q4NX_ROW = (((( D) + (4*Q4NX_ROW) - 1)// (4*Q4NX_ROW)) * (4*Q4NX_ROW))
        DQ = 4096
        DK = 512
        DV = DK
        DH = 64
        num_kv_heads = 8
        num_attn_heads = DQ//DH
        kv_groups = num_attn_heads // num_kv_heads
        # Router stuff
        NUM_EXPERT = 32
        TOPK = 4
        EXPERT_BLOCK_ROWS = 32
        EXPERT_BLOCK_COLS = 64
        EXPERT_BLOCK_TILE_ROWS = 16
        EXPERT_BLOCK_SIZE = EXPERT_BLOCK_ROWS * EXPERT_BLOCK_COLS
        assert D%EXPERT_BLOCK_COLS == 0
        assert INTERMEDIATE_SIZE%EXPERT_BLOCK_COLS == 0
        expert_bias_size  = ((NUM_EXPERT + EXPERT_BLOCK_SIZE -1)//EXPERT_BLOCK_SIZE ) * EXPERT_BLOCK_SIZE    # bias size but padded into ExpertBlock for transfer
        num_expert_ceil_16 = (NUM_EXPERT + 15)//16 *16
        D_ATTN_O_PADDED_MVM_SIZE = ((((D) + (MVM_CT_CORE * Q4NX_ROW) - 1) // (MVM_CT_CORE * Q4NX_ROW)) * (MVM_CT_CORE * Q4NX_ROW));  

        VOCAB_SIZE = 201088
        VOCAB_SIZE_PADDED = ((((VOCAB_SIZE) + (D_ATTN_O_PADDED_MVM_SIZE) - 1) // (D_ATTN_O_PADDED_MVM_SIZE)) * (D_ATTN_O_PADDED_MVM_SIZE));
    else:
        raise ValueError(f"Invalid model type: {model_type}")


    dtype = np.dtype[bfloat16]
    
    
    m = Q4NX_ROW
    k = Q4NX_COL
    w_block_size = Q4NX_BLOCK_SIZE_bf16
    GLU_SLICE= (32*4*4)*2 # Each colum produce 32*4, *4 for 4 column *2 for gate and up interactivelty 
    
    proj_col = 2
    PKT_ID_RMS_CORE_TO_MT = 0
    PKT_ID_RMS_CORE_TO_ROUTER=17
    ASSEMBLE_MT_X_OUT_PORT = 1
    

    
    MVM_INTERLEAVE_SIZE = 128
    x_chunk_size = MVM_INTERLEAVE_SIZE*4 # partial of hidden_size
    # Router specific
    CT_from_shimtile_control_bd_port = 1  
    CT_to_shimtile_control_bd_port = 1    
    IT_CONTROL_packet_list: List[int] = [-1, -1, 28, 27, 29, 31, -1, -1 ]
    IT_CONTRO_response_list = [-1,-1, -1, -1, -1, 30, -1, -1]
    l = 16
    
    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
        

            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols, IT_CONTROL_packet_list)
            ctx = mlir_context()
            
            ctx.data_ty('ping_pong_ty', (16, ), np.dtype[np.int32])
            ctx.data_ty('proj_rtp_ty', (16,), np.dtype[np.int32])
            ctx.data_ty('mxfp4_lut_ty', (512,), np.dtype[np.int32])
            ctx.data_ty('x_ty', (D_ATTN_O_PADDED_MVM_SIZE,), dtype)
            ctx.data_ty('rms_w_ty', (D,), dtype)
            assert D_ATTN_O_PADDED_MVM_SIZE >=D_PADDED_Q4NX_COL+16
            ctx.data_ty('rms_residual_out_padded_ty', (D_ATTN_O_PADDED_MVM_SIZE,), dtype=dtype) # +16 for vector access alignment (balue 14, 15 is packet_head for sending the result out)
            ctx.data_ty('x_chunk_ty', (x_chunk_size,),dtype=dtype)
            ctx.data_ty("x_ATTN_O_padded_ty", (D_ATTN_O_PADDED_MVM_SIZE, ), dtype=dtype)
            #ctx.data_ty('mvm_res_full_ty', (MVM_CT_CORE*Q4NX_ROW + 2, ), dtype=dtype) #+2 for packet header, the result of all CT MVM being sendout from MT

            ctx.data_ty('MVM_interleave_ty', (MVM_INTERLEAVE_SIZE,), dtype=dtype)
        

            ctx.data_ty('project_acc_ty', (Q4NX_ROW,), dtype=np.dtype[np.float32])
            ctx.data_ty('x_slice_ty', (Q4NX_COL,), dtype) 
            ctx.data_ty('linear_w_ty', (w_block_size,), dtype)
            ctx.data_ty('linear_4w_ty', (4 * w_block_size,), dtype)
            ctx.data_ty('m_pkt_ty', (m + 16,), dtype)
            ctx.data_ty('m_ty', (m,), dtype)
            ctx.data_ty('m4_ty', (4 * m,), dtype)
            ctx.data_ty('m_col_ty', (4 * m + 2,), dtype)
            ctx.data_ty('m_full_ty', (16 * m + 2,), dtype)     
            
            #ROPE & Attention stuff
            ctx.data_ty('qkv_ty', (DQ + DK + DV,), dtype)
            ctx.data_ty('q_ty', (DQ+DH,), dtype)
            ctx.data_ty('k_ty', (DK,), dtype)
            ctx.data_ty('v_ty', (DV,), dtype)
            ctx.data_ty('out_ty', (DQ,), dtype)            
            # First DQ, Dk, DV is the bias & DH(attention sink value) then DH is the cos&sine rope
            ctx.data_ty('rope_ty', (DH+ DQ +DK+DV +DH , 1), dtype)
            
            
            # ROUTER STUFF'
            assert EXPERT_BLOCK_SIZE == expert_bias_size #NOTE: for now
            ctx.data_ty('router_hidden_in_ty', (D_PADDED_Q4NX_COL,), dtype=dtype)
            ctx.data_ty("router_weight_ty", (EXPERT_BLOCK_SIZE,), dtype=dtype)
            ctx.data_ty('router_bias_ty', (expert_bias_size,), dtype=dtype)
            ctx.data_ty('router_score_ty', (num_expert_ceil_16,), dtype=dtype )
            ctx.data_ty('router_score_acc_ty', (num_expert_ceil_16, ), dtype=np.dtype[np.float32])
            ctx.data_ty('router_top_k_value_ty', (num_expert_ceil_16,),dtype=dtype)
            ctx.data_ty('router_top_k_index_ty', (num_expert_ceil_16,), dtype=np.dtype[np.int32])
            ctx.data_ty('router_control_packet_ty', (16,), np.dtype[np.uint32])
            
            # GELU stuff
            ctx.data_ty("ug_ty", (GLU_SLICE,), dtype=dtype)
            ctx.data_ty("hid_ty", (GLU_SLICE//2,), dtype=dtype) # result of gelyu
            # DOWN projection
            ctx.data_ty("down_ty", (TOPK*INTERMEDIATE_PADDED_Q4NX_COL, ), dtype=dtype)
            ctx.data_ty("down_res_ty", (TOPK*D_PADDED_FOUR_Q4NX_ROW, ), dtype=dtype)
            attn_q_ty = ctx.data_ty('attn_q_ty', (16, DH +1 ), dtype) #+1 for sink value
            
            assert (64*(DH+1)) == DQ + DH # sanity check to be same with q_ty in rope 
            attn_q_mem_ty = ctx.data_ty('attn_q_mem_ty', (64, DH+1), dtype) # +1 for sink value
            attn_k_ty = ctx.data_ty('attn_k_ty', (16, DH * 2), dtype)
            attn_k_mem_ty = ctx.data_ty('attn_k_mem_ty', (16, DH * 4), dtype)
            attn_s_ty = ctx.data_ty('attn_s_ty', (20, 16), dtype)
            ctx.data_ty('attn_v_ty', (16, DH * 2), dtype)
            ctx.data_ty('attn_v_mem_ty', (16, DH * 4), dtype)
            ctx.data_ty('attn_y_ty', (16, DH), np.dtype[np.float32])
            ctx.data_ty('attn_o_ty', (16, DH), dtype)
            ctx.data_ty('attn_o_mem_ty', (64, DH), dtype)
            ctx.parameter("DH", DH)
            L_ty = ctx.data_ty('L_ty', (16,), np.dtype[np.int32])
            sink_ty = ctx.data_ty('sink_ty', (24, 16), np.dtype[np.uint16])

            
            #PARAMETERS
            ctx.parameter("Q4NX_ROW",Q4NX_ROW)
            ctx.parameter("D", D)
            ctx.parameter("D_PADDED_Q4NX_COL", D_PADDED_Q4NX_COL)
            ctx.parameter("PKT_ID_RMS_CORE_TO_MT", PKT_ID_RMS_CORE_TO_MT)
            ctx.parameter("PKT_ID_RMS_CORE_TO_ROUTER", PKT_ID_RMS_CORE_TO_ROUTER)
            ctx.parameter("NUM_EXPERT", NUM_EXPERT)
            ctx.parameter("TOPK", TOPK)
            ctx.parameter("EXPERT_BLOCK_ROWS", EXPERT_BLOCK_ROWS)
            ctx.parameter("EXPERT_BLOCK_COLS", EXPERT_BLOCK_COLS)
            ctx.parameter("EXPERT_BLOCK_TILE_ROWS", EXPERT_BLOCK_TILE_ROWS)
            ctx.parameter("EXPERT_BLOCK_SIZE", EXPERT_BLOCK_SIZE)
            ctx.parameter("expert_bias_size", expert_bias_size)
            ctx.parameter("m", m)
            ctx.parameter("k", k)
            ctx.parameter("w_block_size", w_block_size)     
            ctx.parameter("DQ", DQ)
            ctx.parameter("DV", DV)
            ctx.parameter("num_attn_heads", num_attn_heads)
            ctx.parameter("kv_groups", kv_groups)
            ctx.parameter("CT_from_shimtile_control_bd_port", CT_from_shimtile_control_bd_port)
            ctx.parameter("CT_to_shimtile_control_bd_port", CT_to_shimtile_control_bd_port)
            ctx.parameter("MVM_INTERLEAVE_SIZE", MVM_INTERLEAVE_SIZE)
            ctx.parameter("x_chunk_size",x_chunk_size)
            ctx.parameter("D_PADDED_FOUR_Q4NX_ROW", D_PADDED_FOUR_Q4NX_ROW)
            ctx.parameter("D_ATTN_O_PADDED_MVM_SIZE", D_ATTN_O_PADDED_MVM_SIZE)            
            ctx.parameter("l", l)

              
            #AIE CORE FUNCTION declaration
            ctx.func("rms_residual", [ctx.rms_residual_out_padded_ty, 
                                      ctx.x_ty, ctx.x_ty, ctx.x_ty,
                                      ctx.rms_w_ty, ctx.rms_w_ty,
                                      ctx.x_ty, ctx.x_ty, ctx.x_ty,
                                      np.int32, np.int32,
                                      ctx.ping_pong_ty, ctx.ping_pong_ty, ctx.proj_rtp_ty
                                      ])
            ctx.func("zero_int32_1", [ctx.ping_pong_ty])
            ctx.func("proj", [ctx.m_pkt_ty, ctx.linear_w_ty, ctx.x_slice_ty, ctx.m_pkt_ty, ctx.linear_w_ty, ctx.x_slice_ty, 
                              ctx.project_acc_ty, ctx.ping_pong_ty, ctx.ping_pong_ty, ctx.proj_rtp_ty]
                              # Peano: the MXFP4 LUTs are buffers, see proj_tile.py
                              + ([] if ENABLE_CHESSCC else [ctx.mxfp4_lut_ty, ctx.mxfp4_lut_ty]))
            ctx.func("rope", [ctx.q_ty, ctx.k_ty, ctx.v_ty, ctx.qkv_ty, ctx.rope_ty])
            ctx.func("remove_hidden_state_padding", [ctx.x_ty, ctx.rms_residual_out_padded_ty])
            
            #router functions
            ctx.func('MVM_bf16_block_32_64_16_with_bias_master',
                     [ 
                        ctx.router_weight_ty, ctx.router_weight_ty,
                        ctx.router_hidden_in_ty,
                        ctx.router_score_ty,
                        ctx.router_score_acc_ty, ctx.router_bias_ty,
                        ctx.ping_pong_ty,
                        
 
                     ])
            
            ctx.func("MVM_bf16_block_32_64_16_with_bias_slave",[
                ctx.router_weight_ty, ctx.router_weight_ty,
                ctx.router_hidden_in_ty, 
                ctx.router_score_acc_ty,
                ctx.ping_pong_ty
                
            ])
            ctx.func("select_top_k_with_index", [
                ctx.router_top_k_value_ty,
                ctx.router_score_ty,
                ctx.router_top_k_index_ty,

                ctx.ping_pong_ty,  # SHIM_BD_1_VIRTUAL_buffer
                ctx.router_control_packet_ty, ctx.router_control_packet_ty,
                np.int32, np.int32, 
                np.int32, np.int32,
                
                np.int32, np.int32,
                np.int32, np.int32,
                np.int32,
                ctx.down_res_ty, ctx.rms_residual_out_padded_ty
            ])
            
            ctx.func( "gelu", [
                ctx.hid_ty, ctx.hid_ty,
                ctx.ug_ty, ctx.ug_ty,
                ctx.ping_pong_ty, ctx.ping_pong_ty,
                np.int32
            ])
            ctx.func("attn_qk", [ctx.attn_q_ty, ctx.attn_q_ty, ctx.attn_k_ty, ctx.attn_k_ty, ctx.attn_s_ty, ctx.attn_s_ty, sink_ty, L_ty])
            ctx.func("attn_kv", [ctx.attn_s_ty, ctx.attn_s_ty, ctx.attn_v_ty, ctx.attn_v_ty, ctx.attn_o_ty, ctx.attn_o_ty, ctx.attn_y_ty, sink_ty, L_ty])

            def floats_to_upper16_hex(float_list):
                """
                Convert a list of float32 numbers to the upper 16 bits of their IEEE 754
                representation, returned as a list of hex values (suitable for uint16 storage).
                """
                upper16_hex = []
                for f in float_list:
                    bits = struct.unpack('<I', struct.pack('<f', f))[0]  # float32 -> 32-bit int
                    upper16 = (bits >> 16) & 0xFFFF                      # take upper 16 bits
                    upper16_hex.append(upper16)
                return upper16_hex

            
            # connections
            attn_qk_cores = [CT[0][0], CT[2][0], CT[0][7], CT[2][7]]
            attn_kv_cores = [CT[1][0], CT[3][0], CT[1][7], CT[3][7]]
            L_buffers = []
            for i in range(len(attn_qk_cores)):
                L_buffers.append(ctx.buffer(f"L_attn_qk_core_{attn_qk_cores[i].row}_{attn_qk_cores[i].col}", attn_qk_cores[i], L_ty, use_write_rtp = True))
                L_buffers.append(ctx.buffer(f"L_attn_kv_core_{attn_kv_cores[i].row}_{attn_kv_cores[i].col}", attn_kv_cores[i], L_ty, use_write_rtp = True))
                
            attn_memtile(MT[0], ctx)
            attn_memtile(MT[7], ctx)
            attn_tile(attn_qk_cores[0], attn_kv_cores[0], ctx)
            attn_tile(attn_qk_cores[1], attn_kv_cores[1], ctx)
            attn_tile(attn_qk_cores[2], attn_kv_cores[2], ctx)
            attn_tile(attn_qk_cores[3], attn_kv_cores[3], ctx)    
            rms_core = CT[0][1]
            rms_residual_tile(rms_core, ctx)
            
            assemble_mt = MT[1]
            proj_assemble(assemble_mt, ctx)
    
            rop_core = CT[2][1]
            RoPE(rop_core, ctx)
            
            remove_padding_core = CT[1][1]
            remove_zero_pad_tile(remove_padding_core, ctx)
            
            router_master = CT[2][6]
            router_slave = CT[1][6]
            gelu_core = CT[0][6]
            moe_router_setup(router_master, router_slave, gelu_core,ctx)
            aux_mt = MT[6]
            aux_memtile(aux_mt, ctx)
            
            connect_dma(remove_padding_core,assemble_mt, 1, 4, 0 )
            
            connect_dma(IT[1], rms_core, 0, 0, 0) # send initial x
            connect_dma(IT[1], rms_core, 1, 1, 0) # send RMS weight
            connect_dma(IT[1], rop_core, 0, 1, 1) # after send initial x, send the q,k,v bias and rope
            connect_dma(rms_core, remove_padding_core, 0,1, PKT_ID_RMS_CORE_TO_MT )

            
            proj_cores_col_0 = [CT[i][proj_col] for i in range(4)]
            proj_cores_col_1 = [CT[i][proj_col + 1] for i in range(4)]
            proj_cores_col_2 = [CT[i][proj_col + 2] for i in range(4)]
            proj_cores_col_3 = [CT[i][proj_col + 3] for i in range(4)]
            
            
            # for i in range(len(proj_cores_col_0)):
            #     ctx.buffer(f"proj_core_{proj_cores_col_0[i].row}_{proj_cores_col_0[i].col}", proj_cores_col_0[i], ctx.proj_rtp_ty, use_write_rtp = True)
            #     ctx.buffer(f"proj_core_{proj_cores_col_1[i].row}_{proj_cores_col_0[i].col}", proj_cores_col_0[i], ctx.proj_rtp_ty, use_write_rtp = True)
            #     ctx.buffer(f"proj_core_{proj_cores_col_2[i].row}_{proj_cores_col_0[i].col}", proj_cores_col_0[i], ctx.proj_rtp_ty, use_write_rtp = True)
            #     ctx.buffer(f"proj_core_{proj_cores_col_3[i].row}_{proj_cores_col_0[i].col}", proj_cores_col_0[i], ctx.proj_rtp_ty, use_write_rtp = True)                 
            
            #@ Projections
            mvm_column(proj_cores_col_0, MT[proj_col],     IT[proj_col],     assemble_mt, 1, ctx)
            mvm_column(proj_cores_col_1, MT[proj_col + 1], IT[proj_col + 1], assemble_mt, 2, ctx)
            mvm_column(proj_cores_col_2, MT[proj_col + 2], IT[proj_col + 2], assemble_mt, 3, ctx)
            mvm_column(proj_cores_col_3, MT[proj_col + 3], IT[proj_col + 3], assemble_mt, 4, ctx)

            # # fanout:
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 1, True)
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 4, True)
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 8, True)
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 10, True)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 1, False)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 4, False)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 8, False)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 10, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 1, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 4, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 8, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 10, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 1, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 4, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 8, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 10, False)


            connect_dma(assemble_mt, rop_core, 0, 0, 1, False)
            connect_dma(rop_core, aux_mt, 0, 1)
            connect_dma(rop_core, IT[0], 1, 0, 14) #K
            connect_dma(rop_core, IT[7], 1, 0, 15) #V
            connect_dma(aux_mt, attn_qk_cores[0], 1, 0)
            connect_dma(aux_mt, attn_qk_cores[1], 2, 0)
            connect_dma(aux_mt, attn_qk_cores[2], 3, 0)
            connect_dma(aux_mt, attn_qk_cores[3], 4, 0)
            connect_dma(attn_kv_cores[0], aux_mt, 0, 2)
            connect_dma(attn_kv_cores[1], aux_mt, 0, 3)
            connect_dma(attn_kv_cores[2], aux_mt, 0, 4)
            connect_dma(attn_kv_cores[3], aux_mt, 0, 5)
            #DEBUG
            connect_dma(aux_mt, remove_padding_core, 5, 1, 2)  # simulate here#connect_dma(aux_mt, assemble_mt, 5, 4, 2)# send the attention hidden_state out  
            # then assemble_mt will automaticall send out to MVM engined doing the o-projection
            
            # The result of o-projection will then send back to assemble_mt in packet_i4, which is then rounted to rms_core
            # #NOTE: for now, send back to shimtile
            # connect_dma(assemble_mt, remove_padding_core, 0, 0, 4, False)
            # connect_dma(remove_padding_core, rms_core, 0, 0, 4, False)
            
            connect_dma(assemble_mt, rms_core, 0, 0, 4, False)

            
            # send hidden_size_padded to router_master
            connect_dma(rms_core, router_master, 0, 1, PKT_ID_RMS_CORE_TO_ROUTER, False )
            connect_dma(IT[6], router_master, 0, 0 )  #Send router bias& weight
            

            # setup control packet for router to control
            for idx, val in enumerate(IT_CONTROL_packet_list):
                if val != -1:
                    packetflow(
                        val, 
                        router_slave.tile, WireBundle.DMA, CT_to_shimtile_control_bd_port,
                        IT[idx].tile, WireBundle.TileControl, 0
                    )
            for idx, val in enumerate(IT_CONTRO_response_list):
                if val != -1:
                    packetflow(
                        val, 
                        IT[idx].tile, WireBundle.TileControl, 0,
                        router_slave.tile, WireBundle.DMA, CT_from_shimtile_control_bd_port
                    )
     
     
            connect_dma(assemble_mt, gelu_core, 0, 0, 8, False)
            
            
           #DEBUG: send out the gate, up projection
            connect_dma(gelu_core, aux_mt, 0, 0)        
            connect_dma(aux_mt, assemble_mt, 0, 4, 0)                
            # connect_dma( assemble_mt, IT[0],0, 1, 10) # send out for debug

            connect_dma( assemble_mt, router_slave,0, 0, 10) # send out for debug
            connect_dma(router_slave, rms_core, 0,0 , 7)
            connect_dma(rms_core, IT[1], 1, 1)            
            # receive the final x
            # connect_dma(rms_core, IT[1], 1,1) #DEBUG
            
            connect_dma(IT[0], MT[0], 0, 0)
            connect_dma(IT[0], MT[0], 1, 1)
            connect_dma(IT[7], MT[7], 0, 0)
            connect_dma(IT[7], MT[7], 1, 1)

            connect_dma(MT[0], attn_qk_cores[0], 0, 1)
            connect_dma(MT[0], attn_qk_cores[1], 1, 1)
            connect_dma(MT[0], attn_kv_cores[0], 2, 1)
            connect_dma(MT[0], attn_kv_cores[1], 3, 1)
        
            connect_dma(MT[7], attn_qk_cores[2], 0, 1)
            connect_dma(MT[7], attn_qk_cores[3], 1, 1)
            connect_dma(MT[7], attn_kv_cores[2], 2, 1)
            connect_dma(MT[7], attn_kv_cores[3], 3, 1)
            # @aiex.runtime_sequence()
            # def runtime_sequence_body():
            #     pass

        print(mctx.module)


if __name__ == "__main__":
    model_type = "GPTOSS_20B"
    # if len(sys.argv) != 2:
    #     print("Usage: python flowArch_aie.py <model_type>, Options: GPTOSS_20B")
    #     print("Use default model type: GPTOSS_20B")
    # else:
    #     model_type = sys.argv[1]
    flow_arch(model_type)
