# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os
from typing import Any, List
# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import ModelConfig
from ml_dtypes import bfloat16
import aie.dialects.aie as aie
from aie.helpers.dialects.ext.scf import _for as range_
from aie.extras.dialects.ext.arith import constant
from constants import kv_out_pkt_id

from utils import AIE_MT_IP, dma_port, mlir_context, tile_with_location, lock_acquire, lock_release, comment_print
from utils import AIE_CT_IP, ShimTile, connect_port
def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple
class delta_core(AIE_CT_IP):
    dtype = np.dtype[bfloat16]
    dtype_S  = np.dtype[np.float32]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig, stack_size: int = 1024, link_with: str = "gate_delta_net_decode_128_128_128_32.o"):
        super().__init__(tile, stack_size = stack_size, link_with = link_with)
        comment_print(f"Initializing Conv at ({self.ct_row}, {self.ct_col})")
        self.model_config = model_config
        self.dh = model_config.linear_v_head_dim
        self.dv = self.dh // model_config.num_delta_net_cts
        self.dk = self.dh
        self.actual_valid_qkv_data_size = self.dh + 2
        self.qkv_packet_size = round_up(self.actual_valid_qkv_data_size, 32)
        ping_pong_buffer_len = 32
        self.ping_pong_ty = np.ndarray[(ping_pong_buffer_len,), np.dtype[np.int32]]
        self.CT_S_ty = np.ndarray[(self.dv, self.dk), self.dtype_S]
        self.CT_Q_K_V_in_ty = np.ndarray[(self.qkv_packet_size,), self.dtype]
        self.CT_O_ty = np.ndarray[(self.dv,), self.dtype]

        self.CT_S_IN_prod_lock = aie.lock(self.ct, lock_id=0, init=2)
        self.CT_S_IN_con_lock = aie.lock(self.ct, lock_id=1, init=0)
        
        self.CT_Q_prod_lock = aie.lock(self.ct, lock_id=2, init=2)
        self.CT_Q_con_lock = aie.lock(self.ct, lock_id=3, init=0)
        
        self.CT_K_prod_lock = aie.lock(self.ct, lock_id=4, init=2)
        self.CT_K_con_lock = aie.lock(self.ct, lock_id=5, init=0)
        
        self.CT_V_prod_lock = aie.lock(self.ct, lock_id=6, init=2)
        self.CT_V_con_lock = aie.lock(self.ct, lock_id=7, init=0)
        
        self.CT_S_O_out_prod_lock = aie.lock(self.ct, lock_id=8, init=1)
        self.CT_S_Finsh_O_start_lock = aie.lock(self.ct, lock_id=9, init=0)
        self.CT_S_O_out_con_lock = aie.lock(self.ct, lock_id=11, init=0)
            

        self.CT_S_chunk_in_buffer = [aie.buffer(self.ct, self.CT_S_ty),
                                     aie.buffer(self.ct, self.CT_S_ty)]
        self.CT_s_chunk_out_buffer = aie.buffer(self.ct, self.CT_S_ty)
        self.CT_Q_buffer = [
            aie.buffer(self.ct, self.CT_Q_K_V_in_ty),
            aie.buffer(self.ct, self.CT_Q_K_V_in_ty)
        ]    
        self.CT_K_buffer = [
            aie.buffer(self.ct, self.CT_Q_K_V_in_ty),
            aie.buffer(self.ct, self.CT_Q_K_V_in_ty)
        ]
        self.CT_V_buffer = [
            aie.buffer(self.ct, self.CT_Q_K_V_in_ty),
            aie.buffer(self.ct, self.CT_Q_K_V_in_ty)
        ]
        self.CT_O_chunk_buffer = [
            aie.buffer(self.ct, self.CT_O_ty)
        ]


        self.ping_pong_buf = [
            aie.buffer(self.ct, datatype=self.ping_pong_ty), # for input ping_pong
        ]
        self.delta_net = ctx.func("gate_delta_net_decode_128_128_128_32", [
            self.CT_S_ty, self.CT_S_ty, self.CT_S_ty,
            self.CT_Q_K_V_in_ty, self.CT_Q_K_V_in_ty,
            self.CT_Q_K_V_in_ty, self.CT_Q_K_V_in_ty,
            self.CT_Q_K_V_in_ty,self.CT_Q_K_V_in_ty,
            self.CT_O_ty,
            np.int32, self.ping_pong_ty
        ])

        self.__call__()

    def __call__(self):
        @aie.core(self.ct, link_with = self.link_with, stack_size = self.stack_size)
        def core_body():
            for _ in range_(0xFFFFFFFF):
                self.delta_net(
                    self.CT_S_chunk_in_buffer[0],
                    self.CT_S_chunk_in_buffer[1],
                    self.CT_s_chunk_out_buffer,
                    self.CT_Q_buffer[0], self.CT_Q_buffer[1],
                    self.CT_K_buffer[0], self.CT_K_buffer[1],
                    self.CT_V_buffer[0], self.CT_V_buffer[1],
                    self.CT_O_chunk_buffer[0], 
                    constant( self.ct_row - 2), self.ping_pong_buf[0] 
                )

        @aie.mem(self.ct)
        def mem_body(block):
            s0 = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest = block[1], chain = block[3])
            with block[1]:
                lock_acquire(self.CT_S_IN_prod_lock, 1)
                aie.dma_bd(self.CT_S_chunk_in_buffer[0])
                lock_release(self.CT_S_IN_con_lock, 1)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(self.CT_S_IN_prod_lock, 1)
                aie.dma_bd(self.CT_S_chunk_in_buffer[1])
                lock_release(self.CT_S_IN_con_lock, 1)
                aie.next_bd(block[1])                
            with block[3]:
                # receve vec_in
                s1 = aie.dma_start(aie.DMAChannelDir.S2MM, 1, dest=block[4], chain=block[10])
            with block[4]:
                lock_acquire(self.CT_K_prod_lock, 1)
                aie.dma_bd(self.CT_K_buffer[0], offset=0, len=self.actual_valid_qkv_data_size)
                lock_release(self.CT_K_con_lock, 1)
                aie.next_bd(block[5])
            with block[5]:
                lock_acquire(self.CT_V_prod_lock, 1)
                aie.dma_bd(self.CT_V_buffer[0], offset=0, len=self.actual_valid_qkv_data_size)
                lock_release(self.CT_V_con_lock, 1)
                aie.next_bd(block[6])
            with block[6]:
                lock_acquire(self.CT_Q_prod_lock, 1)
                aie.dma_bd(self.CT_Q_buffer[0], offset=0, len=self.actual_valid_qkv_data_size)
                lock_release(self.CT_Q_con_lock, 1)
                aie.next_bd(block[7])
            with block[7]:
                lock_acquire(self.CT_K_prod_lock, 1)
                aie.dma_bd(self.CT_K_buffer[1], offset=0, len=self.actual_valid_qkv_data_size)
                lock_release(self.CT_K_con_lock, 1)
                aie.next_bd(block[8])
            with block[8]:
                lock_acquire(self.CT_V_prod_lock, 1)
                aie.dma_bd(self.CT_V_buffer[1], offset=0, len=self.actual_valid_qkv_data_size)
                lock_release(self.CT_V_con_lock, 1)
                aie.next_bd(block[9])
            with block[9]:
                lock_acquire(self.CT_Q_prod_lock, 1)
                aie.dma_bd(self.CT_Q_buffer[1], offset=0, len=self.actual_valid_qkv_data_size)
                lock_release(self.CT_Q_con_lock, 1)
                aie.next_bd(block[4])
            with block[10]:
                s2 = aie.dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[11], chain=block[13])
            with block[11]:
                lock_acquire(self.CT_S_O_out_con_lock, 1)
                aie.dma_bd(self.CT_s_chunk_out_buffer)
                lock_release(self.CT_S_Finsh_O_start_lock, 1)
                aie.next_bd(block[12])
            with block[12]:
                lock_acquire(self.CT_S_Finsh_O_start_lock, 1)
                aie.dma_bd(self.CT_O_chunk_buffer[0])
                lock_release(self.CT_S_O_out_prod_lock, 1)
                aie.next_bd(block[11])
            with block[13]:
                aie.EndOp()       

class delta_net_mt(AIE_MT_IP):
    dtype = np.dtype[bfloat16]
    dtype_S  = np.dtype[np.float32]
    def __init__(self, tile: tile_with_location, ctx: mlir_context, model_config: ModelConfig):

        # for pipelines, all DMA are seperated into 4 partials (ping-ponged, 4 * 4 = 16 bds, plenty)
        super().__init__(tile)
        self.model_config = model_config
        self.DV = model_config.linear_v_head_dim
        self.DK = model_config.linear_qk_head_dim
        self.DV_chunk = self.DV // model_config.num_delta_net_cts
        self.MT_S_size = self.DK * self.DV
        self.MT_O_size = self.DV
        self.NUM_CT_PER_COLUMN = model_config.num_delta_net_cts
        self.CT_S_chunk_size = self.DK * self.DV_chunk
        self.CT_O_chunk_size = self.DV_chunk

        self.MT_S_block_ty = np.ndarray[ ( self.MT_S_size,), self.dtype_S]
        self.MT_O_ty = np.ndarray[ ( self.MT_O_size,), self.dtype]
        # define the buffer 
        self.MT_S_IN_buffer = [
            aie.buffer(self.mt, datatype=self.MT_S_block_ty, name=f"MT_S_IN_buffer_0_col{self.mt_col}"),
            aie.buffer(self.mt, datatype=self.MT_S_block_ty, name=f"MT_S_IN_buffer_1_col{self.mt_col}"),
            aie.buffer(self.mt, datatype=self.MT_S_block_ty, name=f"MT_S_IN_buffer_2_col{self.mt_col}"),
            aie.buffer(self.mt, datatype=self.MT_S_block_ty, name=f"MT_S_IN_buffer_3_col{self.mt_col}")
        ]
        self.MT_S_IN_prod_lock_ping = aie.lock(self.mt, lock_id=0, init = self.NUM_CT_PER_COLUMN, sym_name=f"MT_S_IN_prod_lock_ping_col{self.mt_col}")
        self.MT_S_IN_con_lock_ping  = aie.lock(self.mt, lock_id=1, init = 0, sym_name=f"MT_S_IN_con_lock_ping_col{self.mt_col}")
        self.MT_S_IN_prod_lock_pong = aie.lock(self.mt, lock_id=2, init = self.NUM_CT_PER_COLUMN, sym_name=f"MT_S_IN_prod_lock_pong_col{self.mt_col}")
        self.MT_S_IN_con_lock_pong  = aie.lock(self.mt, lock_id=3, init = 0, sym_name=f"MT_S_IN_con_lock_pong_col{self.mt_col}")
        
        
        self.MT_S_IN_prod_lock_3 = aie.lock(self.mt, lock_id=4, init = self.NUM_CT_PER_COLUMN, sym_name=f"MT_S_IN_prod_lock_3_col{self.mt_col}")
        self.MT_S_IN_con_lock_3  = aie.lock(self.mt, lock_id=5, init = 0, sym_name=f"MT_S_IN_con_lock_3_col{self.mt_col}")
        
        self.MT_S_IN_prod_lock_4 = aie.lock(self.mt, lock_id=6, init = self.NUM_CT_PER_COLUMN, sym_name=f"MT_S_IN_prod_lock_4_col{self.mt_col}")
        self.MT_S_IN_con_lock_4  = aie.lock(self.mt, lock_id=7, init = 0, sym_name=f"MT_S_IN_con_lock_4_col{self.mt_col}")        
        
        
        # NEED to use different style of lock for VEC_out due to too small? #NOTE: WARNING: TODO
        self.MT_vec_out_buffer = [
            aie.buffer(self.mt, datatype = self.MT_O_ty),
            aie.buffer(self.mt, datatype = self.MT_O_ty)
        ]
        self.MT_vec_out_0_prod = aie.lock(self.mt, lock_id=12, init=2)
        self.MT_vec_out_1_prod = aie.lock(self.mt, lock_id=13, init=0)
        self.MT_vec_out_2_prod = aie.lock(self.mt, lock_id=14, init=0)
        self.MT_vec_out_3_prod = aie.lock(self.mt, lock_id=15, init=0)
        self.MT_vec_out_con = aie.lock(self.mt, lock_id=16, init=0)
        
        self.MT_S_OUT_buffer =[
            aie.buffer(self.mt, datatype=self.MT_S_block_ty, name=f"MT_S_OUT_buffer_0_col{self.mt_col}"),
            aie.buffer(self.mt, datatype=self.MT_S_block_ty, name=f"MT_S_OUT_buffer_1_col{self.mt_col}")
        ]
        self.MT_S_OUT_prod_lock_ping = aie.lock(self.mt, lock_id=8, init=self.NUM_CT_PER_COLUMN, sym_name=f"MT_S_OUT_prod_lock_ping_col{self.mt_col}")
        self.MT_S_OUT_con_lock_ping = aie.lock(self.mt, lock_id=9, init=0, sym_name=f"MT_S_OUT_con_lock_ping_col{self.mt_col}")
        self.MT_S_OUT_prod_lock_pong = aie.lock(self.mt, lock_id=10, init=self.NUM_CT_PER_COLUMN, sym_name=f"MT_S_OUT_prod_lock_pong_col{self.mt_col}")
        self.MT_S_OUT_con_lock_pong = aie.lock(self.mt, lock_id=11, init=0, sym_name=f"MT_S_OUT_con_lock_pong_col{self.mt_col}")
        
        
        
        
        #NOTE: assume the S input from Shimtile is already in column major order
        
        self.S_MT_TO_CT = [
            (self.DK, self.DV),
            (self.DV_chunk, 1)
        ]
        
        self.S_CT_TO_MT = [
            (self.DK, self.DV),
            (self.DV_chunk, 1)
        ]
        
        
        
        self.MT_to_CT_S_port_idx = [1,2,3,4]
        self.CT_to_MT_port_idx = [1,2,3,4]
        self.__call__()

    def __call__(self):
        @aie.memtile_dma(self.mt)
        def memtile_dma_body(block):
            s0  = aie.dma_start(aie.DMAChannelDir.S2MM, 0, dest=block[1], chain=block[5])
            with block[1]: # receive data_block
                lock_acquire(self.MT_S_IN_prod_lock_ping, self.NUM_CT_PER_COLUMN)
                aie.dma_bd(self.MT_S_IN_buffer[0])
                lock_release(self.MT_S_IN_con_lock_ping, self.NUM_CT_PER_COLUMN)
                aie.next_bd(block[2])
            with block[2]:
                lock_acquire(self.MT_S_IN_prod_lock_pong, self.NUM_CT_PER_COLUMN)
                aie.dma_bd(self.MT_S_IN_buffer[1])
                lock_release(self.MT_S_IN_con_lock_pong, self.NUM_CT_PER_COLUMN)
                aie.next_bd(block[3])
            with block[3]: # receive data_block
                lock_acquire(self.MT_S_IN_prod_lock_3, self.NUM_CT_PER_COLUMN)
                aie.dma_bd(self.MT_S_IN_buffer[2])
                lock_release(self.MT_S_IN_con_lock_3, self.NUM_CT_PER_COLUMN)
                aie.next_bd(block[4])
            with block[4]:
                lock_acquire(self.MT_S_IN_prod_lock_4, self.NUM_CT_PER_COLUMN)
                aie.dma_bd(self.MT_S_IN_buffer[3])
                lock_release(self.MT_S_IN_con_lock_4, self.NUM_CT_PER_COLUMN)
                aie.next_bd(block[1])
                
            cur_BD = 5
            for row in range(self.NUM_CT_PER_COLUMN):
                with block[cur_BD]:
                    cur_BD += 1
                    s1 = aie.dma_start(aie.DMAChannelDir.MM2S, self.MT_to_CT_S_port_idx[row],
                                dest=block[cur_BD], chain=block[cur_BD+4]
                                )
                with block[cur_BD]:
                    cur_BD = cur_BD+1
                    lock_acquire(self.MT_S_IN_con_lock_ping, 1)
                    aie.dma_bd(self.MT_S_IN_buffer[0], offset=row*self.DV_chunk, len=self.CT_S_chunk_size, dimensions=self.S_MT_TO_CT)
                    lock_release(self.MT_S_IN_prod_lock_ping, 1)
                    aie.next_bd(block[cur_BD])
                with block[cur_BD]:
                    cur_BD+= 1
                    lock_acquire(self.MT_S_IN_con_lock_pong, 1)
                    aie.dma_bd(self.MT_S_IN_buffer[1], offset=row*self.DV_chunk, len=self.CT_S_chunk_size, dimensions=self.S_MT_TO_CT)
                    lock_release(self.MT_S_IN_prod_lock_pong, 1)
                    aie.next_bd(block[cur_BD])
                with block[cur_BD]:
                    cur_BD+= 1
                    lock_acquire(self.MT_S_IN_con_lock_3, 1)
                    aie.dma_bd(self.MT_S_IN_buffer[2], offset=row*self.DV_chunk, len=self.CT_S_chunk_size, dimensions=self.S_MT_TO_CT)
                    lock_release(self.MT_S_IN_prod_lock_3, 1)   
                    aie.next_bd(block[cur_BD])                                     
                with block[cur_BD]:
                    cur_BD+= 1
                    lock_acquire(self.MT_S_IN_con_lock_4, 1)
                    aie.dma_bd(self.MT_S_IN_buffer[3], offset=row*self.DV_chunk, len=self.CT_S_chunk_size, dimensions=self.S_MT_TO_CT)
                    lock_release(self.MT_S_IN_prod_lock_4, 1)                        
                    aie.next_bd(block[cur_BD-4])

            list_of_vect_out_lock_order = [self.MT_vec_out_0_prod, self.MT_vec_out_1_prod, self.MT_vec_out_2_prod, self.MT_vec_out_3_prod, self.MT_vec_out_con]
            for row in range(self.NUM_CT_PER_COLUMN):
                with block[cur_BD]:
                    cur_BD += 1
                    s1 = aie.dma_start(aie.DMAChannelDir.S2MM, channel_index=self.CT_to_MT_port_idx[row],
                                dest=block[cur_BD], chain=block[cur_BD+4]
                                )
                with block[cur_BD]:
                    cur_BD += 1
                    lock_acquire(self.MT_S_OUT_prod_lock_ping, 1)
                    aie.dma_bd(self.MT_S_OUT_buffer[0], offset=row*self.DV_chunk, len=self.CT_S_chunk_size, dimensions=self.S_CT_TO_MT)
                    lock_release(self.MT_S_OUT_con_lock_ping, 1)
                    aie.next_bd(block[cur_BD])
                with block[cur_BD]:
                    cur_BD +=1
                    lock_acquire(list_of_vect_out_lock_order[row], 1)
                    aie.dma_bd(self.MT_vec_out_buffer[0], offset=row*self.CT_O_chunk_size, len=self.CT_O_chunk_size)
                    lock_release(list_of_vect_out_lock_order[row+1], 1)
                    aie.next_bd(block[cur_BD])
                with block[cur_BD]:
                    cur_BD += 1
                    lock_acquire(self.MT_S_OUT_prod_lock_pong, 1)
                    aie.dma_bd(self.MT_S_OUT_buffer[1], offset=row*self.DV_chunk, len=self.CT_S_chunk_size, dimensions=self.S_CT_TO_MT)
                    lock_release(self.MT_S_OUT_con_lock_pong, 1)
                    aie.next_bd(block[cur_BD])
                with block[cur_BD]:
                    cur_BD +=1
                    lock_acquire(list_of_vect_out_lock_order[row], 1)
                    aie.dma_bd(self.MT_vec_out_buffer[1], offset=row*self.CT_O_chunk_size, len=self.CT_O_chunk_size )
                    lock_release(list_of_vect_out_lock_order[row+1], 1)
                    aie.next_bd(block[cur_BD - 4])
            
            with block[cur_BD]:
                cur_BD+= 1
                s3 = aie.dma_start(aie.DMAChannelDir.MM2S,0, dest=block[cur_BD], chain=block[cur_BD+2] )
            with block[cur_BD]:
                cur_BD+= 1
                lock_acquire(self.MT_S_OUT_con_lock_ping, self.NUM_CT_PER_COLUMN)
                aie.dma_bd(self.MT_S_OUT_buffer[0])
                lock_release(self.MT_S_OUT_prod_lock_ping, self.NUM_CT_PER_COLUMN)
                aie.next_bd(block[cur_BD])
            with block[cur_BD]:
                cur_BD+= 1
                lock_acquire(self.MT_S_OUT_con_lock_pong, self.NUM_CT_PER_COLUMN)            
                aie.dma_bd(self.MT_S_OUT_buffer[1])
                lock_release(self.MT_S_OUT_prod_lock_pong, self.NUM_CT_PER_COLUMN)
                aie.next_bd(block[cur_BD - 2])

            with block[cur_BD]:
                cur_BD += 1
                s4 = aie.dma_start(aie.DMAChannelDir.MM2S,5, dest=block[cur_BD], chain=block[cur_BD+2] )

            with block[cur_BD]:
                cur_BD+= 1
                lock_acquire(list_of_vect_out_lock_order[-1], 1)
                aie.dma_bd(self.MT_vec_out_buffer[0])
                lock_release(list_of_vect_out_lock_order[0], 1)
                aie.next_bd(block[cur_BD])
            with block[cur_BD]:
                cur_BD +=1
                lock_acquire(list_of_vect_out_lock_order[-1], 1)
                aie.dma_bd(self.MT_vec_out_buffer[1])
                lock_release(list_of_vect_out_lock_order[0], 1)
                aie.next_bd(block[cur_BD - 2])
            with block[cur_BD]:
                aie.EndOp()

class delta_net_column:
    def __init__(self, 
        compute_tiles: List[tile_with_location], 
        memtile: tile_with_location,
        shim_tile: ShimTile,
        model_config: ModelConfig,
        ctx: mlir_context, 
        ct_stack_size: int,
        ct_link_with: str
    ):  
        comment_print(f"Initializing attn_column at MT({memtile.row}, {memtile.col})")
        self.cts = []
        for ct in compute_tiles:
            self.cts.append(delta_core(ct, ctx, model_config, stack_size=ct_stack_size, link_with=ct_link_with))
        self.mt  = delta_net_mt(memtile, ctx, model_config)
        self.IT = shim_tile

    def route(self, qkv_in_port: dma_port):
        connect_port(self.IT.MM2S[0], self.mt.S2MM[0])
        connect_port(self.mt.MM2S[0], self.IT.S2MM[0])
        for i, ct in enumerate(self.cts):
            connect_port(qkv_in_port, ct.S2MM[1])
            connect_port(self.mt.MM2S[i + 1], ct.S2MM[0])
            connect_port(ct.MM2S[0], self.mt.S2MM[i + 1])

    
    @property
    def O_out(self):
        return self.mt.MM2S[5]
