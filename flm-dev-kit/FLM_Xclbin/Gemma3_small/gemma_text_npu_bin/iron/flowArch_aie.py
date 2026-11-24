# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys

from ml_dtypes import bfloat16
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
import aie.dialects.aie as aie
import aie.dialects.aiex as aiex
from aie.helpers.dialects.ext.scf import _for as range_
import aie.utils.trace as trace_utils
from aie.utils.trace import MemTilePortEvent, MemTileEvent, PortEvent
from aie.utils.trace_events_enum import CoreEvent, MemEvent, ShimTileEvent, MemTileEvent
from aie.dialects import memref

from utils import dma_mm2s_ch, dma_s2mm_ch, flow_mm2s_ch, flow_s2mm_ch, connect_dma, generate_tile_map, mlir_context
from proj_tile import proj_tile, mvm_column
from rms_residual import rms_residual_tile
from aux_memtile import aux_memtile
from gelu import gelu_tile
from RoPE import RoPE
from attn import attn, attn_memtile
from proj_assemble import proj_assemble
from utils import dma_port, connect_port
from HandlePadding import HandlePadding

def flow_arch(model_type: str = "GEMMA3_1B"):
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    if model_type == "GEMMA3_1B":
        D = 1152
        num_attn_heads = 4
        num_kv_heads = 1
        HIDDEN_SIZE = 6912
        DQ = 1024
    elif model_type == "GEMMA3_270M":
        D = 640
        num_attn_heads = 4
        num_kv_heads = 1
        HIDDEN_SIZE = 2048
        DQ = 1024
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    DH = DQ // num_attn_heads
    DV = num_kv_heads * DH
    DK = num_kv_heads * DH
    kv_groups = num_attn_heads // num_kv_heads
    PADDED_D = (D + 255) // 256 * 256
    GLU_SLICE = 256

    m = 16
    k = 128
    dtype = np.dtype[bfloat16]
    w_block_size = m * k * 5 // 8 // 2 


    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)
            ctx = mlir_context()
            ctx.data_ty('x_ty', (D,), dtype)
            ctx.data_ty('x_padded_ty', (PADDED_D,), dtype)
            ctx.data_ty('x_slice_ty', (128,), dtype)
            ctx.data_ty('x_chunk_ty', (128,), dtype)
            ctx.data_ty('w_ty', (D,), dtype)
            ctx.data_ty('y_ty', (D,), dtype)
            ctx.data_ty('y_pkt_ty', (D + 16,), dtype)
            ctx.data_ty('qkv_ty', (DQ + DK + DV,), dtype)
            ctx.data_ty('q_ty', (DQ,), dtype)
            ctx.data_ty('k_ty', (DK,), dtype)
            ctx.data_ty('v_ty', (DV,), dtype)
            ctx.data_ty('out_ty', (DQ,), dtype)
            ctx.data_ty('linear_w_ty', (w_block_size,), dtype)
            ctx.data_ty('linear_4w_ty', (4 * w_block_size,), dtype)
            ctx.data_ty('m_pkt_ty', (m + 16,), dtype)
            ctx.data_ty('m_ty', (m,), dtype)
            ctx.data_ty('m_col_ty', (4 * m + 2,), dtype)
            ctx.data_ty('m_full_ty', (16 * m + 2,), dtype)
            ctx.data_ty('hid_ty', (GLU_SLICE // 2,), dtype)
            ctx.data_ty('ug_ty', (GLU_SLICE,), dtype)
            ctx.data_ty('down_ty', (HIDDEN_SIZE,), dtype)
            ctx.data_ty('rope_ty', (DH * 3, 1), dtype)

            attn_q_ty = ctx.data_ty('attn_q_ty', (kv_groups, DH), dtype)
            attn_k_ty = ctx.data_ty('attn_k_ty', (16, DH), dtype)
            attn_s_ty = ctx.data_ty('attn_s_ty', (kv_groups + 1, 16), dtype) # (G, l) + (1, 16)c
            attn_v_ty = ctx.data_ty('attn_v_ty', (16, DH), dtype)
            attn_o_ty = ctx.data_ty('attn_o_ty', (kv_groups, DH), dtype)

            mem_k_ty = ctx.data_ty('mem_k_ty', (16, DH), dtype)
            mem_v_ty = ctx.data_ty('mem_v_ty', (16, DH), dtype)

            L_ty = ctx.data_ty('L_ty', (8,), np.dtype[np.int32])
            ctx.parameter("D", D)
            ctx.parameter("PADDED_D", PADDED_D)
            ctx.parameter("DQ", DQ)
            ctx.parameter("DH", DH)
            ctx.parameter("DV", DV)
            ctx.parameter("m", m)
            ctx.parameter("k", k)
            ctx.parameter("HIDDEN_SIZE", HIDDEN_SIZE)
            ctx.parameter("w_block_size", w_block_size)
            ctx.parameter("kv_groups", kv_groups)


            # AIE Core Function declarations
            ctx.func("proj", [ctx.m_pkt_ty, ctx.linear_w_ty, ctx.x_slice_ty, ctx.m_pkt_ty, ctx.linear_w_ty, ctx.x_slice_ty])
            ctx.func("rms_residual", [ctx.y_pkt_ty, ctx.x_ty, ctx.w_ty, ctx.w_ty])
            ctx.func("gelu", [ctx.hid_ty, ctx.hid_ty, ctx.ug_ty, ctx.ug_ty])
            ctx.func("rope", [ctx.q_ty, ctx.k_ty, ctx.v_ty, ctx.qkv_ty, ctx.rope_ty])
            ctx.func("attn_qk", [attn_q_ty, attn_q_ty, attn_k_ty, attn_k_ty, attn_s_ty, attn_s_ty, L_ty])
            ctx.func("attn_kv", [attn_s_ty, attn_s_ty, attn_v_ty, attn_v_ty, attn_o_ty, attn_o_ty, L_ty])
            # rms_residual_core = rms_core

            gelu_core = CT[0][6]
            rms_core = CT[0][1]
            rop_core = CT[2][1]
            padding_core = CT[1][1]
            attn_qk_cores = CT[0][0]
            attn_kv_cores = CT[1][0]
            proj_col = 2
            assemble_mt = MT[1]
            aux_mt = MT[6]

            L_buffers = []
            L_buffers.append(ctx.buffer(f"L_attn_qk_core_{attn_qk_cores.row}_{attn_qk_cores.col}", attn_qk_cores, L_ty, use_write_rtp = True))
            L_buffers.append(ctx.buffer(f"L_attn_kv_core_{attn_kv_cores.row}_{attn_kv_cores.col}", attn_kv_cores, L_ty, use_write_rtp = True))
            proj_cores_col_0 = [CT[i][proj_col] for i in range(4)]
            proj_cores_col_1 = [CT[i][proj_col + 1] for i in range(4)]
            proj_cores_col_2 = [CT[i][proj_col + 2] for i in range(4)]
            proj_cores_col_3 = [CT[i][proj_col + 3] for i in range(4)]
            # Set up compute tiles
            # dataflows
            RoPE(rop_core, ctx)
            rms_residual_tile(rms_core, ctx)
            attn(attn_qk_cores, attn_kv_cores, ctx)
            attn_memtile(MT[0], ctx)
            HandlePadding(padding_core, ctx)

            proj_assemble(assemble_mt, ctx)
            aux_memtile(aux_mt, ctx)

            connect_dma(IT[1], rms_core, 0, 0, 0)
            connect_dma(IT[1], rms_core, 1, 1, 0)
            connect_dma(IT[1], rop_core, 0, 1, 1)
            connect_dma(rms_core, assemble_mt, 0, 4, 0)
            

            #@ Projections
            mvm_column(proj_cores_col_0, MT[proj_col],     IT[proj_col],     assemble_mt, 1, ctx)
            mvm_column(proj_cores_col_1, MT[proj_col + 1], IT[proj_col + 1], assemble_mt, 1, ctx)
            mvm_column(proj_cores_col_2, MT[proj_col + 2], IT[proj_col + 2], assemble_mt, 1, ctx)
            mvm_column(proj_cores_col_3, MT[proj_col + 3], IT[proj_col + 3], assemble_mt, 1, ctx)
            gelu_tile(gelu_core, ctx)
            # fanout:
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 1, True)
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 4, True)
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 8, True)
            connect_dma(MT[proj_col],     assemble_mt, 4, 0, 5, True)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 1, False)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 4, False)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 8, False)
            connect_dma(MT[proj_col + 1], assemble_mt, 4, 1, 5, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 1, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 4, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 8, False)
            connect_dma(MT[proj_col + 2], assemble_mt, 4, 2, 5, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 1, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 4, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 8, False)
            connect_dma(MT[proj_col + 3], assemble_mt, 4, 3, 5, False)

            connect_dma(assemble_mt, rop_core, 0, 0, 1, False)
            connect_dma(rop_core, aux_mt, 0, 1)
            connect_dma(rop_core, IT[0], 1, 0, 14, False)
            connect_dma(rop_core, IT[0], 1, 1, 15, False)
            connect_dma(aux_mt, attn_qk_cores, 1, 0)
            connect_dma(attn_kv_cores, aux_mt, 0, 2)
            connect_dma(aux_mt, assemble_mt, 2, 4, 2)

            connect_dma(assemble_mt, padding_core, 0, 0, 4, False)
            connect_dma(padding_core, rms_core, 0, 0, 8, False)

            connect_dma(assemble_mt, gelu_core, 0, 0, 8, False)

            connect_dma(gelu_core, aux_mt, 0, 0)
            connect_dma(aux_mt, assemble_mt, 0, 4, 0)
            connect_dma(assemble_mt, padding_core, 0, 0, 5, False)
            connect_dma(padding_core, rms_core, 0, 0, 8, False)
            connect_dma(rms_core, IT[1], 0, 1, 1)
            
            connect_dma(IT[0], MT[0], 0, 0)
            connect_dma(IT[0], MT[0], 1, 1)

            connect_dma(MT[0], attn_qk_cores, 0, 1)
            connect_dma(MT[0], attn_kv_cores, 1, 1)
            
            @aiex.runtime_sequence()
            def runtime_sequence_body():
                for buf in L_buffers:
                    buf[0] = 8

        print(mctx.module)


if __name__ == "__main__":
    model_type = "GEMMA3_1B"
    if len(sys.argv) != 2:
        print("Usage: python flowArch_aie.py <model_type>, Options: GEMMA3_1B, GEMMA3_270M")
        print("Use default model type: GEMMA3_1B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)
