# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from typing import ValuesView
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

from utils import mlir_context, tile_with_location, lock_acquire, lock_release

def moe_router_setup(
    MASTER_CT_in:tile_with_location, SLAVE_CT_in:tile_with_location, GELU_CT_in:tile_with_location,  ctx:mlir_context
    
):
    MASTER_CT = MASTER_CT_in.tile
    SLAVE_CT = SLAVE_CT_in.tile
    GELU_CT = GELU_CT_in.tile
    
    assert SLAVE_CT_in.col == 6
    assert SLAVE_CT_in.row == 3
    NUM_EXPERT = ctx.parameters["NUM_EXPERT"]
    D_ATTN_O_PADDED_MVM_SIZE = ctx.parameters["D_ATTN_O_PADDED_MVM_SIZE"]
    D = ctx.parameters["D"]
    TOPK = ctx.parameters["TOPK"]
    EXPERT_BLOCK_SIZE = ctx.parameters["EXPERT_BLOCK_SIZE"]
    expert_bias_size = ctx.parameters["expert_bias_size"]
    CT_from_shimtile_control_bd_port = ctx.parameters["CT_from_shimtile_control_bd_port"]
    CT_to_shimtile_control_bd_port = ctx.parameters["CT_to_shimtile_control_bd_port"]
    
    router_hidden_in_ty  = ctx.data_tys["router_hidden_in_ty"]
    ping_pong_ty = ctx.data_tys["ping_pong_ty"]
    router_weight_ty = ctx.data_tys["router_weight_ty"]
    router_bias_ty = ctx.data_tys["router_bias_ty"]
    router_score_ty = ctx.data_tys["router_score_ty"]
    router_score_acc_ty = ctx.data_tys["router_score_acc_ty"]
    router_top_k_value_ty = ctx.data_tys["router_top_k_value_ty"]
    router_top_k_index_ty = ctx.data_tys["router_top_k_index_ty"]
    router_control_packet_ty = ctx.data_tys["router_control_packet_ty"]
    down_res_ty = ctx.data_tys["down_res_ty"]
    rms_residual_out_padded_ty = ctx.data_tys["rms_residual_out_padded_ty"]
    #GELU stuff
    ug_ty = ctx.data_tys["ug_ty"]
    hid_ty = ctx.data_tys["hid_ty"]

    zero_int_1 =ctx.funcs["zero_int32_1"]
    

    
    mvm_bf16_block_func_master = ctx.funcs["MVM_bf16_block_32_64_16_with_bias_master"]
    mvm_bf16_block_func_slave = ctx.funcs["MVM_bf16_block_32_64_16_with_bias_slave"]
    select_top_k_func =ctx.funcs["select_top_k_with_index"]
    gelu_func = ctx.funcs["gelu"]
    

    M_WEIGHT_PROD_LOCK = 2
    M_WEIGHT_CON_LOCK = 3
    M_D_PROD_LOCK = 0
    M_D_CON_LOCK = 1
    M_SCORE_PROD_LOCK = 4
    M_SCORE_CON_LOCK = 5
    M_Prod_S_Con_LOCK = 10
    M_Con_S_Prod_LOCK = 11    
    
    
    #MASTER
    # MASTER_CT = tile(COL_INDEX,row=4)
    Master_hidden_state_in_buffer = [
        aie.buffer(MASTER_CT, datatype=router_hidden_in_ty, name="Master_hidden_state_in_buffer"),
    ]
    Master_hidden_state_in_prod_lock = lock(MASTER_CT, lock_id=M_D_PROD_LOCK, init=1, sym_name="Master_hidden_state_in_prod_lock")
    Master_hidden_state_in_con_lock = lock(MASTER_CT, lock_id=M_D_CON_LOCK, init=0, sym_name="Master_hidden_state_in_con_lock")
    
    Master_weight_in_buffer = [
        buffer(MASTER_CT, datatype=router_weight_ty, name="mat_block_in_0"),
        buffer(MASTER_CT, datatype=router_weight_ty, name="mat_block_in_1")
    ]
    Master_weight_in_prod_lock = lock(MASTER_CT, lock_id=M_WEIGHT_PROD_LOCK, init=2, sym_name="Master_weight_in_prod_lock")
    Master_weight_in_con_lock = lock(MASTER_CT, lock_id=M_WEIGHT_CON_LOCK, init=0, sym_name="Master_weight_in_con_lock")
    
    
    Master_router_score_buffer = [
        buffer(MASTER_CT, datatype=router_score_ty),
    ]
    Master_router_score_buffer_prod_lock = lock(MASTER_CT, lock_id=M_SCORE_PROD_LOCK, init=1, sym_name="Master_router_score_buffer_prod_lock")
    Master_router_score_buffer_con_lock = lock(MASTER_CT, lock_id=M_SCORE_CON_LOCK, init=0, sym_name="Master_router_score_buffer_con_lock")
    
    Master_router_score_acc = [
        buffer(MASTER_CT, datatype=router_score_acc_ty, name="Master_router_score_acc"),
    ]
    Master_weight_bias_buffer = [
        buffer(MASTER_CT, datatype=router_bias_ty, name="Master_weight_bias_buffer"),
    ]
    ping_pong_buffer = [
        buffer(MASTER_CT, datatype=ping_pong_ty)
    ]
    
    #SLAVE

    # SLAVE_CT = tile(COL_INDEX,row=3)
    CT_to_shimtile_control_bd_id = 10

    CT_from_shimtile_control_bd_id = 11

    
    CT_master_produce_slave_consume_lock = lock(SLAVE_CT, lock_id=M_Prod_S_Con_LOCK, init=0)
    CT_master_consume_slave_produce_lock = lock(SLAVE_CT, lock_id=M_Con_S_Prod_LOCK, init=0)
    
    Slave_top_k_value_buffer = [
        buffer(SLAVE_CT, datatype=router_top_k_value_ty, name="top_k_value"),
    ]  
    Slave_top_k_index_buffer = [
        buffer(SLAVE_CT, datatype=router_top_k_index_ty, name="top_k_index"),
    ]

    CT_to_tile_control_buffer = buffer(SLAVE_CT, datatype=router_control_packet_ty)
    control_packet_out_prod_lock = lock(SLAVE_CT, lock_id=6, init=1)
    control_packet_out_con_lock = lock(SLAVE_CT, lock_id=7, init=0)
    
    CT_from_tile_control_buffer = buffer(SLAVE_CT, datatype=router_control_packet_ty)
    control_packet_in_prod_lock = lock(SLAVE_CT, lock_id=8, init=1)
    control_packet_in_con_lock = lock(SLAVE_CT, lock_id=9, init=0)
    
    down_proj_res_in_buffer = buffer(SLAVE_CT, datatype=down_res_ty)
    down_proj_res_prod_lock  = lock(SLAVE_CT, lock_id =0, init=1)
    down_proj_res_con_lock = lock(SLAVE_CT, lock_id=1, init=0)
    
    mlp_out_buffer = buffer(SLAVE_CT,rms_residual_out_padded_ty  )
    mlp_out_prod_lock = lock(SLAVE_CT, lock_id=2, init=1)
    mlp_out_con_lock = lock(SLAVE_CT, lock_id=3, init=0)
    SLAVE_STACK_SIZE = 8*1024
    SHIM_BD_1_VIRTUAL_buffer = buffer(SLAVE_CT, ping_pong_ty,name="SHIM_BD_1_VIRTUAL_buffer_SLAVE", use_write_rtp=True ) # right after the stack
    #NOTEL need to sync with c++ config
    enable_core_acces_lock = lock(SLAVE_CT, lock_id=15, init=0) # set by runtime sequence
    
    #GELU
    up_gate_buffer =[
        buffer(GELU_CT, datatype=ug_ty),
        buffer(GELU_CT, datatype=ug_ty)
    ]
    up_gate_prod_lock  = lock(GELU_CT, lock_id=0, init=2)
    up_gate_con_lock = lock(GELU_CT, lock_id=1, init=0)
    
    gel_res_buffer = [
        buffer(GELU_CT, datatype=hid_ty),
        buffer(GELU_CT, datatype=hid_ty)
    ]
    gel_res_prod_lock = lock(GELU_CT, lock_id=2, init=2)
    gel_res_con_lock = lock(GELU_CT, lock_id = 3, init=0)
    gelu_y_ping_pong = aie.buffer(GELU_CT, ping_pong_ty)
    gelu_x_ping_pong = aie.buffer(GELU_CT, ping_pong_ty)
    gelu_request_down_lock = lock(GELU_CT, lock_id = 4, init=0)
        
    @core(SLAVE_CT, "gptoss_moe_router_slave.o",  stack_size=SLAVE_STACK_SIZE )
    def core_body_slave():
        for _ in range_(start=0xFFFFFFFF):
            mvm_bf16_block_func_slave(
                Master_weight_in_buffer[0], Master_weight_in_buffer[1],
                Master_hidden_state_in_buffer[0], Master_router_score_acc[0],
                ping_pong_buffer[0]
            )
            # top router
            select_top_k_func(
                Slave_top_k_value_buffer[0],
                Master_router_score_buffer[0], 
                
                Slave_top_k_index_buffer[0],
             
                SHIM_BD_1_VIRTUAL_buffer,
                
                # control packet related 
                CT_to_tile_control_buffer, CT_from_tile_control_buffer,
                constant(48+6), constant(48+7),
                constant(48+8), constant(48+9),
                constant(CT_to_shimtile_control_bd_id), constant(CT_from_shimtile_control_bd_id),
                constant(CT_to_shimtile_control_bd_port), constant(CT_from_shimtile_control_bd_port),
                constant(4), #gelu_request_down_lock,
                down_proj_res_in_buffer, mlp_out_buffer
            )
    
    # # dma send it out
    @mem(SLAVE_CT)
    def SLAVE_CT_dma_body(block):
        s0 = dma_start(DMAChannelDir.S2MM, channel_index=CT_from_shimtile_control_bd_port, dest=block[1], chain=block[2])
        with block[1]:
            use_lock(control_packet_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(CT_from_tile_control_buffer, offset=0, len=1,  # len=1 since we expect a single value per read to shimtile
                    bd_id=CT_from_shimtile_control_bd_id)
            use_lock(control_packet_in_con_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[2]:
            s1 = dma_start(DMAChannelDir.MM2S, channel_index=CT_to_shimtile_control_bd_port, dest=block[3], chain=block[4])
        with block[3]:
            use_lock(control_packet_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            #NOTE: the len is something changes dynamicaly
            dma_bd(CT_to_tile_control_buffer, offset=0, len=1, #NOTE: for now
                    bd_id=CT_to_shimtile_control_bd_id) 
            use_lock(control_packet_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[3])
        with block[4]:
            assert(CT_from_shimtile_control_bd_port == 1)
            s2 = dma_start(DMAChannelDir.S2MM, channel_index=0, dest=block[5], chain=block[6])
        with block[5]:
            use_lock(down_proj_res_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd( down_proj_res_in_buffer )
            use_lock(down_proj_res_con_lock, LockAction.Release, value=1)
            next_bd(block[5])
        with block[6]:
            s3 = dma_start(DMAChannelDir.MM2S, 0, dest=block[7], chain=block[8])
        with block[7]:
            use_lock(mlp_out_con_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(mlp_out_buffer, offset=0, len=D_ATTN_O_PADDED_MVM_SIZE, packet=(0,7)) # 2 for contorl pacekt, only send length D!!!
            use_lock(mlp_out_prod_lock, LockAction.Release, value=1)
            next_bd(block[7])
        with block[8]:
            EndOp()
    # @mem(SLAVE_CT)
    # def CT_dma_body_slave(block):
    #     s2 =dma_start(DMAChannelDir.MM2S, 0, dest=block[1], chain=block[2])
    #     with block[1]:
    #         use_lock(Slave_top_k_value_con_lock, LockAction.AcquireGreaterEqual, value=1)
    #         dma_bd(Slave_top_k_value_buffer[0])
    #         use_lock(Slave_top_k_value_prod_lock, LockAction.Release, value=1)
    #         next_bd(block[1])
    #     with block[2]:
    #         EndOp()
    
    
    @core(MASTER_CT, "gptoss_moe_router_master.o",  stack_size=8*1024 )     
    def core_body():

        # assume we compute the result in a row base order
        # this need be in sync with runtimesequence on how it sends data to us
        zero_int_1(ping_pong_buffer[0])
        for _ in range_(start=0xFFFFFFFF):
            mvm_bf16_block_func_master(
                Master_weight_in_buffer[0], Master_weight_in_buffer[1],
                Master_hidden_state_in_buffer[0],
                Master_router_score_buffer[0], 
                Master_router_score_acc[0], Master_weight_bias_buffer[0],
                ping_pong_buffer[0],
            )
            
            
    
    
    #CT Memtile logic: For now, don't do any data reordering on CT yet
    @mem(MASTER_CT)
    def CT_dma_body(block):
        # Receive matrix_in_block
        s0 = dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
        with block[1]:
            use_lock(Master_weight_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(Master_weight_in_buffer[0])
            use_lock(Master_weight_in_con_lock, LockAction.Release, value=1)
            next_bd(block[2])
        with block[2]:
            use_lock(Master_weight_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(Master_weight_in_buffer[1])
            use_lock(Master_weight_in_con_lock, LockAction.Release, value=1)
            next_bd(block[1])
        with block[3]:
            # receive vector_in
            s1 = dma_start(DMAChannelDir.S2MM, 1, dest=block[4], chain=block[5])
        with block[4]:
            use_lock(Master_hidden_state_in_prod_lock, LockAction.AcquireGreaterEqual, value=1)
            dma_bd(Master_hidden_state_in_buffer[0])
            use_lock(Master_hidden_state_in_con_lock, LockAction.Release, value=1)
            next_bd(block[4])
        with block[5]:
            EndOp()
    
    
    
    
    @core(GELU_CT, "gelu.o", stack_size=1024*8)
    def core_body():
        zero_int_1(gelu_y_ping_pong)
        zero_int_1(gelu_x_ping_pong)
        for _ in range_(start=0xFFFFFFFF):
            gelu_func(gel_res_buffer[0], gel_res_buffer[1],
                      up_gate_buffer[0], up_gate_buffer[1],
                      gelu_y_ping_pong, gelu_x_ping_pong,
                      constant(4) #gelu_request_down_lock
                      )
    
    @mem(GELU_CT)
    def GELU_DMA_BODY(block):
        s0 = aie.dma_start(channel_dir = aie.DMAChannelDir.S2MM, channel_index = 0, dest=block[1], chain=block[3])
        with block[1]:
            lock_acquire(up_gate_prod_lock, 1)
            aie.dma_bd(up_gate_buffer[0])
            lock_release(up_gate_con_lock, 1)
            aie.next_bd(block[2])

        with block[2]:
            lock_acquire(up_gate_prod_lock, 1)
            aie.dma_bd(up_gate_buffer[1])
            lock_release(up_gate_con_lock, 1)
            aie.next_bd(block[1])

        with block[3]:
            s2 = aie.dma_start(channel_dir = aie.DMAChannelDir.MM2S, channel_index = 0, dest = block[4], chain = block[6])
            
        with block[4]: # x, to mt
            lock_acquire(gel_res_con_lock, 1)
            aie.dma_bd(gel_res_buffer[0])
            lock_release(gel_res_prod_lock, 1)
            aie.next_bd(block[5])

        with block[5]: # w, to mt
            lock_acquire(gel_res_con_lock, 1)
            aie.dma_bd(gel_res_buffer[1])
            lock_release(gel_res_prod_lock, 1)
            aie.next_bd(block[4])

        with block[6]:
            aie.EndOp()
    
    
    return MASTER_CT, SLAVE_CT