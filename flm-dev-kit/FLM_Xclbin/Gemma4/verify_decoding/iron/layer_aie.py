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

from utils import generate_tile_map, mlir_context
from proj_tile import q4_proj_engine
from rms_residual import rms_residual_tile
from linear_aux_tile import linear_aux_memtile
from glu import glu_tile
from proj_layer_embedding import proj_layer_embedding
from gate_layer_embedding import gate_layer_embedding
from per_layer_input_merge import proj_layer_merge, proj_layer_up
from RoPE import RoPE, SWA_RoPE
from attn import attn_column
# from Gemma4.decoding.iron.bf16_proj_engine import AB_tile
from utils import dma_port, connect_port

def flow_arch(model_type: str = "QWEN3_5_4B"):
    if L_TOKENS != 1:
        raise SystemExit("iron/layer_aie.py: the fused 8-row layer is built stage by stage (plan S6.3-S6.10); only L_TOKENS=1 (the decode design) "
                         "exists so far. Use DESIGN=projonly or DESIGN=projstage for the 8-row projection engine.")
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
            ctx.data_ty('RTP_ty', (16,), np.dtype[np.int32]) # for all RTP requests
            
            # Instantiate IPs with single CT/MT
            rms_core = rms_residual_tile(CT[0][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rms_residual_link_with)
            rope_core = RoPE(CT[2][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with)
            swa_rope_core = SWA_RoPE(CT[3][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.swa_rope_link_with)

            proj_layer_embd = proj_layer_embedding(CT[1][4], ctx, model_config, stack_size = 1024 * 4)
            gate_layer_embd = gate_layer_embedding(CT[0][4], rms_core.y_out_buffer, ctx, model_config, stack_size = 1024 * 4)
            proj_layer_merge_core = proj_layer_merge(CT[1][5], ctx, model_config, stack_size = 1024 * 4)
            proj_layer_up_core = proj_layer_up(CT[0][5], ctx, model_config, stack_size = 1024 * 4)
            glu = glu_tile(CT[1][3], ctx, model_config, stack_size = 4 * 1024, link_with = model_config.glu_link_with)

            # aux_mt = aux_memtile(MT[3], ctx, model_config)

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

            _attn_column = attn_column(
                attn_qk_tiles = [CT[0][2], CT[2][2]],
                attn_kv_tiles = [CT[1][2], CT[3][2]],
                memtile = MT[2],
                shim_tile = IT[2],
                model_config = model_config,
                ctx = ctx,
                qk_stack_size = 1024,
                kv_stack_size = 1024,
                qk_link_with = model_config.attn_qk_link_with,
                kv_link_with = model_config.attn_kv_link_with,
                swa_qk_link_with = model_config.swa_attn_qk_link_with,
                swa_kv_link_with = model_config.swa_attn_kv_link_with
            )


            # TB connection for testing
            
            TB_PORT = IT[3].S2MM[0]
            # connect_port(glu.Hid_out, TB_PORT, pkt_id = to_proj_x_in_pkt_id.from_glu, keep_pkt_header = False)

            # connect_port(rms_core.Y_out, TB_PORT, pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)
            # Connect all DMAs, the order of connection matters.
            # Long connection has higher priority in routing.
            connect_port(IT[3].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[3].MM2S[0], rope_core.ROPE_in, pkt_id = it1_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], swa_rope_core.ROPE_in, pkt_id = it1_pkt_id.to_rope, keep_pkt_header = False)

            connect_port(rms_core.Y_out, proj_engine.X_in, pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)

            proj_engine.route()

            connect_port(proj_engine.Y_out, glu.Up_Gate_in, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, rms_core.X_in, pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, rope_core.QKV_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, swa_rope_core.QKV_in, pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = False)

            connect_port(rope_core.Q_out, _attn_column.Q_in)
            connect_port(swa_rope_core.Q_out, _attn_column.SWA_Q_in)

            connect_port(rope_core.KV_out, IT[2].S2MM[0])
            connect_port(swa_rope_core.KV_out, IT[2].S2MM[1])

            connect_port(_attn_column.O_out, proj_engine.X_in, pkt_id = to_proj_x_in_pkt_id.from_attn, keep_pkt_header = False)
            connect_port(_attn_column.SWA_O_out, proj_engine.X_in, pkt_id = to_proj_x_in_pkt_id.from_swa_attn, keep_pkt_header = False)

            connect_port(glu.Hid_out, proj_engine.X_in, pkt_id = to_proj_x_in_pkt_id.from_glu, keep_pkt_header = False)

            _attn_column.route()

            connect_port(IT[4].MM2S[0], proj_layer_embd.XXW_In)
            connect_port(IT[4].MM2S[1], proj_layer_embd.PROJ_W_In)
            connect_port(proj_layer_embd.Y_Out, proj_layer_merge_core.NORM_I_IN)
            connect_port(IT[5].MM2S[0], gate_layer_embd.PROJ_W_In)
            connect_port(gate_layer_embd.Y_Out, proj_layer_merge_core.RES_GATE_IN)
            connect_port(proj_layer_merge_core.Y_Out, proj_layer_up_core.X_IN)
            connect_port(IT[5].MM2S[1], proj_layer_up_core.W_IN)
            connect_port(proj_layer_up_core.Y_Out, IT[3].S2MM[0])

            # connect_port(proj_engine.Y_out, TB_PORT, pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
            # connect_port(rope_core.Q_out, TB_PORT)
            # connect_port(proj_layer_embd.Y_Out, TB_PORT)
            # connect_port(rms_core.Y_out, TB_PORT, pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)
            # connect_port(aux_mt.OutDown_out, TB_PORT, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)

            # connect_port(_attn_column.O_out, TB_PORT, pkt_id = 14, keep_pkt_header = False)
            # connect_port(_attn_column.SWA_O_out, TB_PORT, pkt_id = 15, keep_pkt_header = False)
            
            # connect_port(rope_core.Q_out, IT[5].S2MM[0])
            # connect_port(swa_rope_core.Q_out, IT[5].S2MM[1])
            # connect_port(proj_engine.Y_out, IT[5].S2MM[0], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            # connect_port(proj_engine.Y_out, IT[5].S2MM[1], pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = False)
            # connect_port(rms_core.Y_out, TB_PORT, pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)
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
