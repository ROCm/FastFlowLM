# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import get_model_config
from iron.models import *

from constants import *

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

from utils import generate_tile_map, mlir_context, ENABLE_CHESSCC
from proj_tile import q4_proj_engine
from rms_residual import rms_residual_tile
from aux_memtile import aux_memtile
from linear_aux_tile import linear_aux_memtile
from glu import glu_tile
from gate import gate_tile
from RoPE import RoPE
from attn import attn_column, attn_column_single_cu
from Conv import Conv
from DeltaNetNorm import DeltaNet_Norm
from AB_tile import AB_tile
from Gated_Delta import delta_net_column
from utils import dma_port, connect_port

def flow_arch(model_type: str = "QWEN3_5_4B"):
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            ctx = mlir_context()
            ctx.data_ty('L_ty', (8,), np.dtype[np.int32]) # for all RTP requests
            
            # Instantiate IPs with single CT/MT
            rms_core = rms_residual_tile(CT[0][2], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rms_residual_link_with)
            rope_core = RoPE(CT[1][2], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with) # phi4 is special
            ab_core = AB_tile(CT[1][5], ctx, model_config, stack_size = 1024 * 4, link_with = "ab_proj.o") # phi4 is special
            conv_core = Conv(CT[0][5], ctx, ab_core.y_buffer, model_config, stack_size = 1024 * 8, link_with = "conv.o") # phi4 is special
            delta_net_norm_core = DeltaNet_Norm(CT[3][2], ctx, model_config, stack_size = 1024 * 4, link_with = "delta_net_norm.o") # phi4 is special
            gate_core = gate_tile(CT[2][2], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.gate_link_with) # phi4 is special
            glu = glu_tile(CT[3][5], ctx, model_config, stack_size = 1 * 1024, link_with = model_config.glu_link_with)

            aux_mt = aux_memtile(MT[2], ctx, model_config)
            linear_aux_mt = linear_aux_memtile(MT[5], ctx, model_config)

            # Instantiate IPs with multiple CTs/MTs
            proj_cores = []
            proj_assemble_mts = []
            proj_shim_tiles = []
            for col in range(0, 2):
                proj_cores_col = []
                for row in range(0, 4):
                    proj_cores_col.append(CT[row][col])
                proj_cores.append(proj_cores_col)
                proj_assemble_mts.append(MT[col])
                proj_shim_tiles.append(IT[col])
            
            for col in range(6, 8):
                proj_cores_col = []
                for row in range(0, 4):
                    proj_cores_col.append(CT[row][col])
                proj_cores.append(proj_cores_col)
                proj_assemble_mts.append(MT[col])
                proj_shim_tiles.append(IT[col])
            proj_engine = q4_proj_engine(proj_cores, proj_assemble_mts, proj_shim_tiles, 10 * 1024, ["proj_main.o", "proj.o"], ctx, model_config)

            if model_config.num_attn_cus == 1:
                _attn_column = attn_column_single_cu(
                    attn_qk_tile = CT[0][3],
                    attn_kv_tile = CT[1][3],
                    memtile = MT[3],
                    shim_tile = IT[3],
                    model_config = model_config,
                    ctx = ctx,
                    qk_stack_size = 1024 if ENABLE_CHESSCC else 2048,
                    kv_stack_size = 1024 if ENABLE_CHESSCC else 2048,
                    qk_link_with = model_config.attn_qk_link_with,
                    kv_link_with = model_config.attn_kv_link_with
                )
            else:
                _attn_column = attn_column(
                    attn_qk_tiles = [CT[0][3], CT[2][3]],
                    attn_kv_tiles = [CT[1][3], CT[3][3]],
                    memtile = MT[3],
                    shim_tile = IT[3],
                    model_config = model_config,
                    ctx = ctx,
                    qk_stack_size = 1024 if ENABLE_CHESSCC else 2048,
                    kv_stack_size = 1024 if ENABLE_CHESSCC else 2048,
                    qk_link_with = model_config.attn_qk_link_with,
                    kv_link_with = model_config.attn_kv_link_with
                )

            _delta_net_column = delta_net_column(
                compute_tiles = [CT[0][4], CT[1][4], CT[2][4], CT[3][4]],
                memtile = MT[4],
                shim_tile = IT[4],
                model_config = model_config,
                ctx = ctx,
                ct_stack_size = 1024 * 8,
                ct_link_with = "gate_delta_net_decode_128_128_128_32.o"
            )

            # Connect all DMAs, the order of connection matters.
            # Long connection has higher priority in routing.
            connect_port(IT[2].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[0], rope_core.ROPE_in, pkt_id = it1_pkt_id.to_rope, keep_pkt_header = False)

            connect_port(rms_core.Y_out, proj_engine.X_in, pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)

            proj_engine.route()

            connect_port(proj_engine.Y_out, aux_mt.X_in, pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)

            if (model_config.num_attn_cus == 1):
                connect_port(rope_core.Q_out, _attn_column.Q_in)
            else:
                connect_port(rope_core.Q_out, _attn_column.Q_in[0], pkt_id=0, keep_pkt_header=False)
                connect_port(rope_core.Q_out, _attn_column.Q_in[1], pkt_id=1, keep_pkt_header=False)

            connect_port(rope_core.KV_out, IT[3].S2MM[0])
            
            connect_port(aux_mt.Gate_out, gate_core.Gate_In)
            connect_port(proj_engine.Y_out, aux_mt.GLU_IN, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)            
            connect_port(aux_mt.GLU_out, glu.Up_Gate_in)

            connect_port(aux_mt.OutDown_out, rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(_attn_column.O_out, gate_core.Attn_In, pkt_id=0, keep_pkt_header=False)
            
            connect_port(proj_engine.Y_out, linear_aux_mt.QKV_in, pkt_id = proj_pkt_id.to_conv, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, rope_core.QKV_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, IT[1].S2MM[0], pkt_id = proj_pkt_id.to_out, keep_pkt_header = False)

            connect_port(_delta_net_column.O_out, delta_net_norm_core.DH_in)
            connect_port(delta_net_norm_core.DH_out, gate_core.Attn_In, pkt_id=0, keep_pkt_header=False)

            _delta_net_column.route(conv_core.Y_out)
            connect_port(gate_core.Attn_Out, proj_engine.X_in, pkt_id = aux_to_proj_x_in_pkt_id.to_out_proj, keep_pkt_header = False)
            connect_port(glu.Hid_out, proj_engine.X_in, pkt_id = aux_to_proj_x_in_pkt_id.to_down_proj, keep_pkt_header = False)

            connect_port(rms_core.Y_out, IT[2].S2MM[1], pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)
            connect_port(rms_core.AB_Yout, linear_aux_mt.X_in)

            connect_port(IT[5].MM2S[0], linear_aux_mt.AB_PROJ_W_in)
            connect_port(linear_aux_mt.AB_PROJ_W_out, ab_core.W_in)
            connect_port(IT[5].MM2S[1], linear_aux_mt.W_in)
            connect_port(linear_aux_mt.CONV_W_out, conv_core.W_in)
            connect_port(linear_aux_mt.DELTA_NORM_W_out, delta_net_norm_core.W_in)
            connect_port(linear_aux_mt.HIDDEN_STATES_AB_BIAS_out, ab_core.X_in)
            connect_port(linear_aux_mt.NEW_CACHE_out, IT[5].S2MM[1])

            _attn_column.route()

            # TB connection for testing
            TB_PORT = IT[5].S2MM[0]
            # connect_port(_delta_net_column.O_out, TB_PORT)
            # connect_port(gate_core.TB_Out, TB_PORT)
            # connect_port(conv_core.TB_OUT, TB_PORT)
            
            # It is required to disable the original path
            # connect_port(aux_mt.OutDown_out, TB_PORT, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)

            
        print(mctx.module)

if __name__ == "__main__":
    model_type = "LLAMA_3_2_1B"
    if len(sys.argv) != 2:
        print("Usage: python flowArch_aie.py <model_type>, Options: LLAMA_3_2_1B, LLAMA_3_2_3B, LLAMA_3_1_8B")
        print("Use default model type: LLAMA_3_2_1B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)
