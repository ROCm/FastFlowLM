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



from utils import lock_acquire, lock_release
import numpy.typing as npt
from aie.dialects.aiex import NpuDmaMemcpyNd




def MXFP4_dequant_MT(
    MT_quant_data_ty:npt.NDArray,
    MT_dequant_data_ty:npt.NDArray,
    col:int,
    MT_quant_data_size:int,
    MT_dequant_data_size:int,
    Q4NX_ROW_SIZE = 32,
    Q4NX_COL_SIZE = 128    
):

    MT_X = tile(col=col, row=1)
    
    quant_block_buffer = [
        buffer(MT_X, MT_quant_data_ty, name=f"MT_quant_0_{col}"),
        buffer(MT_X, MT_quant_data_ty, name=f"MT_quant_1_{col}")        
    ]
    dequant_block_buffer = [
        buffer(MT_X, MT_dequant_data_ty, name=f"MT_dequant_0_{col}"),
        buffer(MT_X, MT_dequant_data_ty, name=f"MT_dequant_1_{col}")        
    ]
    
    MT_x_quant_in_prod_lock_0 = aie.lock( tile=MT_X, lock_id=1, init=2) 
    MT_x_quant_in_prod_lock_1 = aie.lock( tile=MT_X, lock_id=11, init=2)     
    MT_x_quant_in_con_0_lock = aie.lock(MT_X, lock_id=2, init=0) 
    MT_x_quant_in_con_1_lock = aie.lock(MT_X, lock_id=3, init=0)
    MT_x_quant_in_con_2_lock = aie.lock(MT_X, lock_id=4, init=0)
    MT_x_quant_in_con_3_lock = aie.lock(MT_X, lock_id=5, init=0)
    
    
    MT_x_dequant_out_prod_0_lock = aie.lock(MT_X,lock_id=6, init=2)
    MT_x_dequant_out_prod_1_lock = aie.lock(MT_X, lock_id=7, init=0)
    MT_x_dequant_out_prod_2_lock = aie.lock(MT_X, lock_id=8, init=0)
    MT_x_dequant_out_prod_3_lock = aie.lock(MT_X, lock_id=9, init=0)
    
    MT_x_dequant_con_0_lock = aie.lock(MT_X, lock_id=10, init=0) # port 4, send 4 32x356 block = 64z256
    
    assert MT_quant_data_size%4 == 0
    MT_quant_size_div_4 = MT_quant_data_size//4
    
    assert MT_dequant_data_size %4 == 0
    MT_dequant_size_div_4 = MT_dequant_data_size //4
    
    weight_receive_reorder = [(4, 1024), (16, 8), (8, 128), (8, 1)]
    @aie.memtile_dma(MT_X)
    def memtile_dma_body(block):
        s0 = dma_start(aie.DMAChannelDir.S2MM, 4, dest=block[1], chain=block[3])

        with block[1]:
            lock_acquire(MT_x_quant_in_prod_lock_0,1)
            dma_bd(quant_block_buffer[0], offset=0, len=MT_quant_data_size//2)
            lock_release(MT_x_quant_in_con_0_lock, 1)
            next_bd(block[2])
        with block[2]:
            lock_acquire(MT_x_quant_in_prod_lock_0,1)
            dma_bd(quant_block_buffer[1], offset=0, len=MT_quant_data_size//2)
            lock_release(MT_x_quant_in_con_0_lock, 1)
            next_bd(block[1])        
        with block[3]:
            s1 = dma_start(aie.DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
        with block[4]:
            lock_acquire(MT_x_quant_in_con_0_lock,1 )    
            dma_bd(quant_block_buffer[0], offset=0, len=MT_quant_size_div_4)
            lock_release(MT_x_quant_in_con_1_lock,1)
            next_bd(block[5])
        with block[5]:
            lock_acquire(MT_x_quant_in_con_0_lock,1 )    
            dma_bd(quant_block_buffer[1], offset=0, len=MT_quant_size_div_4)
            lock_release(MT_x_quant_in_con_1_lock,1)
            next_bd(block[4])
        with block[6]:
            s2 = dma_start(aie.DMAChannelDir.MM2S, 1, dest=block[7], chain=block[9])
        with block[7]:
            lock_acquire(MT_x_quant_in_con_1_lock, 1)
            dma_bd(quant_block_buffer[0], offset=MT_quant_size_div_4, len=MT_quant_size_div_4)
            lock_release(MT_x_quant_in_prod_lock_0, 1)
            next_bd(block[8])
        with block[8]:
            lock_acquire(MT_x_quant_in_con_1_lock, 1)
            dma_bd(quant_block_buffer[1], offset=MT_quant_size_div_4, len=MT_quant_size_div_4)
            lock_release(MT_x_quant_in_prod_lock_0, 1)
            next_bd(block[7])
        with block[9]:
            s3 = dma_start(aie.DMAChannelDir.MM2S, 2, dest=block[10], chain=block[12])
        with block[10]:
            lock_acquire(MT_x_quant_in_con_2_lock,1)
            dma_bd(quant_block_buffer[0], offset=MT_quant_size_div_4*2, len=MT_quant_size_div_4)
            lock_release(MT_x_quant_in_con_3_lock, 1)
            next_bd(block[11])
        with block[11]:
            lock_acquire(MT_x_quant_in_con_2_lock,1)
            dma_bd(quant_block_buffer[1], offset=MT_quant_size_div_4*2, len=MT_quant_size_div_4)
            lock_release(MT_x_quant_in_con_3_lock, 1)
            next_bd(block[10])         
        with block[12]:
            s4 = dma_start(aie.DMAChannelDir.MM2S, 3, dest=block[13], chain=block[15])
        with block[13]:
            lock_acquire(MT_x_quant_in_con_3_lock,1)
            dma_bd(quant_block_buffer[0], offset=MT_quant_size_div_4*3, len=MT_quant_size_div_4)            
            lock_release(MT_x_quant_in_prod_lock_1,1)   
            next_bd(block[14])
        with block[14]:
            lock_acquire(MT_x_quant_in_con_3_lock,1)
            dma_bd(quant_block_buffer[1], offset=MT_quant_size_div_4*3, len=MT_quant_size_div_4)            
            lock_release(MT_x_quant_in_prod_lock_1,1) 
            next_bd(block[13])
        with block[15]:
            s6 = dma_start(DMAChannelDir.S2MM, 0, dest=block[16], chain=block[18])
        with block[16]:
            lock_acquire(MT_x_dequant_out_prod_0_lock,1)
            dma_bd(dequant_block_buffer[0],   offset = 0, len =MT_dequant_size_div_4,dimensions=weight_receive_reorder) 
            lock_release(MT_x_dequant_out_prod_1_lock, 1)
            next_bd(block[17])
        with block[17]:
            lock_acquire(MT_x_dequant_out_prod_0_lock,1)
            dma_bd(dequant_block_buffer[1],   offset = 0, len =MT_dequant_size_div_4,dimensions=weight_receive_reorder)
            lock_release(MT_x_dequant_out_prod_1_lock, 1)
            next_bd(block[16])
        with block[18]:
            s7 = dma_start(DMAChannelDir.S2MM, 1, dest=block[19], chain=block[21])
        with block[19]:
            lock_acquire(MT_x_dequant_out_prod_1_lock, 1)
            dma_bd(dequant_block_buffer[0],   offset=MT_dequant_size_div_4, len =MT_dequant_size_div_4,dimensions=weight_receive_reorder)
            lock_release(MT_x_dequant_out_prod_2_lock, 1)
            next_bd(block[20])
        with block[20]:
            lock_acquire(MT_x_dequant_out_prod_1_lock, 1)
            dma_bd(dequant_block_buffer[1],   offset=MT_dequant_size_div_4, len =MT_dequant_size_div_4 ,dimensions=weight_receive_reorder)
            lock_release(MT_x_dequant_out_prod_2_lock, 1)
            next_bd(block[19])
        with block[21]:
            s8 = dma_start(DMAChannelDir.S2MM, 2, dest=block[22], chain=block[24])
        with block[22]:
            lock_acquire(MT_x_dequant_out_prod_2_lock,1)
            dma_bd(dequant_block_buffer[0], offset=2*MT_dequant_size_div_4, len =MT_dequant_size_div_4 ,dimensions=weight_receive_reorder )
            lock_release(MT_x_dequant_out_prod_3_lock, 1)
            next_bd(block[23])
        with block[23]:
            lock_acquire(MT_x_dequant_out_prod_2_lock,1)
            dma_bd(dequant_block_buffer[1], offset=2*MT_dequant_size_div_4, len =MT_dequant_size_div_4 ,dimensions=weight_receive_reorder )
            lock_release(MT_x_dequant_out_prod_3_lock, 1)
            next_bd(block[22])
        with block[24]:
            s9 = dma_start(DMAChannelDir.S2MM, 3, dest=block[25], chain=block[27])
        with block[25]:
            lock_acquire(MT_x_dequant_out_prod_3_lock,1)
            dma_bd(dequant_block_buffer[0], offset=3*MT_dequant_size_div_4, len=MT_dequant_size_div_4,dimensions=weight_receive_reorder)
            lock_release(MT_x_dequant_con_0_lock, 1)
            next_bd(block[26])
        with block[26]:
            lock_acquire(MT_x_dequant_out_prod_3_lock,1)
            dma_bd(dequant_block_buffer[1], offset=3*MT_dequant_size_div_4, len=MT_dequant_size_div_4,dimensions=weight_receive_reorder)
            lock_release(MT_x_dequant_con_0_lock, 1)
            next_bd(block[25])
        with block[27]:
            s10 = dma_start(DMAChannelDir.MM2S,4, dest=block[28], chain=block[30] )
        with block[28]:
            lock_acquire(MT_x_dequant_con_0_lock,1 )
            dma_bd(dequant_block_buffer[0], offset=0, len=4*MT_dequant_size_div_4)
            lock_release(MT_x_dequant_out_prod_0_lock, 1)
            next_bd(block[29])
        with block[29]:
            lock_acquire(MT_x_dequant_con_0_lock,1 )
            dma_bd(dequant_block_buffer[1], offset=0, len=4*MT_dequant_size_div_4)
            lock_release(MT_x_dequant_out_prod_0_lock, 1)
            next_bd(block[28])
        with block[30]:
            s0 = dma_start(aie.DMAChannelDir.S2MM, 5, dest=block[31], chain=block[33])            
        with block[31]:
            lock_acquire(MT_x_quant_in_prod_lock_1,1)
            dma_bd(quant_block_buffer[0], offset=MT_quant_data_size//2, len=MT_quant_data_size//2)
            lock_release(MT_x_quant_in_con_2_lock, 1)
            next_bd(block[32])
        with block[32]:
            lock_acquire(MT_x_quant_in_prod_lock_1,1)
            dma_bd(quant_block_buffer[1], offset=MT_quant_data_size//2, len=MT_quant_data_size//2)
            lock_release(MT_x_quant_in_con_2_lock, 1)
            next_bd(block[31])    
        with block[33]:
            EndOp()
    return MT_X