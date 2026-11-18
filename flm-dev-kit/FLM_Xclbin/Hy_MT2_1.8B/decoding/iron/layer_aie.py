# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import get_model_config
from iron.models import *

from constants import *
from iron.utils import ENABLE_CHESSCC

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
from aux_memtile import aux_memtile
from glu import glu_tile
from RoPE import RoPE
from attn import attn_column
from utils import dma_port, connect_port

def flow_arch(model_type: str = "LLAMA_3B"):
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
            glu = glu_tile(CT[0][5], ctx, model_config, stack_size = 1 * 1024, link_with = model_config.glu_link_with)
            attn_aux_mt = aux_memtile(MT[5], ctx, model_config)

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

            # Peano's attn_kv frames are much larger than chess's, so its stack
            # is bigger. The AIE stack grows upward into the next buffer, and
            # makefiles/check_peano_stack.sh fails the build if a frame can
            # overrun it; raise the size here if that check fires.
            #
            # The stack sits below the RTP buffers, so resizing it can move them,
            # and FLM_DLL hardcodes RTP addresses per model. After changing either
            # size, diff xclbins/rtp_addresses.txt against a chess build for every
            # MODEL_TYPE.
            ATTN_QK_STACK = 1024
            ATTN_KV_STACK = 1024 if ENABLE_CHESSCC else 4096

            attn_column_left = attn_column(
                attn_qk_tile = [CT[0][3], CT[2][3]],
                attn_kv_tile = [CT[1][3], CT[3][3]],
                memtile = MT[3],
                shim_tile = IT[3],
                tail_port = IT[5].MM2S[0],
                model_config = model_config,
                ctx = ctx,
                qk_stack_size = ATTN_QK_STACK,
                kv_stack_size = ATTN_KV_STACK,
                qk_link_with = model_config.attn_qk_link_with,
                kv_link_with = model_config.attn_kv_link_with
            )

            attn_column_right = attn_column(
                attn_qk_tile = [CT[0][4], CT[2][4]],
                attn_kv_tile = [CT[1][4], CT[3][4]],
                memtile = MT[4],
                shim_tile = IT[4],
                tail_port = IT[5].MM2S[1],
                model_config = model_config,
                ctx = ctx,
                qk_stack_size = ATTN_QK_STACK,
                kv_stack_size = ATTN_KV_STACK,
                qk_link_with = "attn_qk.o",
                kv_link_with = "attn_kv.o"
            )


            # Connect all DMAs, the order of connection matters.
            # Long connection has higher priority in routing.
            connect_port(IT[2].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[0], rope_core.ROPE_in, pkt_id = it1_pkt_id.to_rope, keep_pkt_header = False)

            connect_port(rms_core.Y_out, proj_engine.X_in, pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)

            proj_engine.route()

            connect_port(proj_engine.Y_out, rope_core.QKV_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(rope_core.Q_out, attn_aux_mt.Q_in)
            connect_port(rope_core.KV_out, IT[3].S2MM[0], pkt_id = kv_out_pkt_id.to_it0, keep_pkt_header = False)
            connect_port(rope_core.KV_out, IT[4].S2MM[0], pkt_id = kv_out_pkt_id.to_it7, keep_pkt_header = False)
            
            connect_port(attn_aux_mt.Q_out[0], attn_column_left.Q_in_0)
            connect_port(attn_aux_mt.Q_out[1], attn_column_left.Q_in_1)
            connect_port(attn_aux_mt.Q_out[2], attn_column_right.Q_in_0)
            connect_port(attn_aux_mt.Q_out[3], attn_column_right.Q_in_1)

            connect_port(attn_column_left.O_out_0, attn_aux_mt.O_in[0])
            connect_port(attn_column_left.O_out_1, attn_aux_mt.O_in[1])
            connect_port(attn_column_right.O_out_0, attn_aux_mt.O_in[2])
            connect_port(attn_column_right.O_out_1, attn_aux_mt.O_in[3])
            
            connect_port(attn_aux_mt.O_out, proj_engine.X_in, pkt_id = aux_to_proj_x_in_pkt_id.to_out_proj, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, rms_core.X_in   , pkt_id = proj_pkt_id.to_rms_norm            , keep_pkt_header = False)
            connect_port(proj_engine.Y_out, glu.Up_Gate_in  , pkt_id = proj_pkt_id.to_glu                 , keep_pkt_header = False)
            
            connect_port(glu.Hid_out, attn_aux_mt.down_in)

            connect_port(attn_aux_mt.down_out, proj_engine.X_in, pkt_id = aux_to_proj_x_in_pkt_id.to_down_proj, keep_pkt_header = False)


            connect_port(rms_core.Y_out, IT[2].S2MM[1], pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)

            connect_port(rms_core.LM_Head_out, IT[3].S2MM[1])

            attn_column_left.route()
            attn_column_right.route()
            
        print(mctx.module)

if __name__ == "__main__":
    model_type = "LLAMA_3_2_1B"
    if len(sys.argv) != 2:
        print("Usage: python flowArch_aie.py <model_type>, Options: LLAMA_3_2_1B, LLAMA_3_2_3B, LLAMA_3_1_8B")
        print("Use default model type: LLAMA_3_2_1B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)
