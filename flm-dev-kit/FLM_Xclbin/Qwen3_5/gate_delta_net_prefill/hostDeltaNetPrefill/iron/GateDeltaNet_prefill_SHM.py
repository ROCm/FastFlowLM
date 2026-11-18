# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

from re import L
import numpy as np

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



def get_SHM_buffer_str_list(
    col_idx:int
)->tuple[str, str, str]:
    S_IN_SHM_Str = f"S_IN_SHIM_{col_idx}"
    QKV_ALPHA_BETA_IN_SHM_Str = f"QKV_ALPHA_BETA_IN_SHIM_{col_idx}"
    O_S_OUTPUT_SHM_Str = f"O_S_OUTPUT_SHM_{col_idx}"
    
    return S_IN_SHM_Str, QKV_ALPHA_BETA_IN_SHM_Str, O_S_OUTPUT_SHM_Str


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
    
    
def get_S_IN_bd_id(ping_pong:int):
    return 0 if ping_pong == 0 else 8

def get_alpha_bd_id(ping_pong:int):
    return 1 if ping_pong == 0 else 9

def get_beta_bd_id(ping_pong:int):
    return 2 if ping_pong == 0 else 10
def get_q_k_bd_id(ping_pong:int):
    return 3 if ping_pong == 0 else 11
def get_k_bd_id(ping_pong:int):
    return 4 if ping_pong == 0 else 12
def get_v_bd_id(ping_pong:int):
    return 5 if ping_pong == 0 else 13
def get_output_o_s_bd_id(ping_pong:int):
    return 6 if ping_pong == 0 else 14

    
def update_ping_pong( ping_pong:int):
    if ping_pong ==0:
        return 1
    else:
        return 0

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