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
from passthrough import passthrough_tile
from rms_residual import rms_residual_tile
from aux_memtile import aux_memtile
from glu import glu_tile
from gate import gate_tile
from RoPE import RoPE
from attn import attn_column
from attn_hub import attn_hub_memtile
from utils import dma_port, connect_port

def flow_arch(model_type: str = "QWEN3_8_27B"):
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
            pass_core = passthrough_tile(CT[3][3], ctx, model_config, stack_size = 1024 * 2) 
            gate_core = gate_tile(CT[2][2], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.gate_link_with)
            rope_core = RoPE(CT[1][2], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with)
            # replays = down_proj's row-block count: the SwiGLU row is the x that
            # streaming mvm re-reads once per row block.
            glu = glu_tile(CT[3][2], ctx, model_config, stack_size = 1 * 1024, link_with = model_config.glu_link_with,
                           replays = model_config.D // (model_config.q4nx_row_block_size * 16))

            aux_mt = aux_memtile(MT[2], ctx, model_config)

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

            # Two attention columns, two CUs each, one kv head per CU -- 4 heads.
            # Within a column the qk tile must sit directly north of its kv tile:
            # it writes the scores through the south window with no DMA.
            attn_columns = [
                attn_column(
                    attn_qk_tiles = [CT[0][col], CT[2][col]],
                    attn_kv_tiles = [CT[1][col], CT[3][col]],
                    memtile = MT[col],
                    shim_tile = IT[col],
                    model_config = model_config,
                    ctx = ctx,
                    qk_stack_size = 1024,
                    kv_stack_size = 1024,
                    qk_link_with = model_config.attn_qk_link_with,
                    kv_link_with = model_config.attn_kv_link_with,
                )
                for col in (4, 5)
            ]

            # MT[3] is otherwise unused and sits between the RoPE tile and the
            # attention columns, so it does both halves of the cross-column
            # plumbing: q out to the four CUs, o back in from the two columns.
            attn_hub = attn_hub_memtile(MT[3], ctx, model_config)

            # Connect all DMAs, the order of connection matters.
            # Long connection has higher priority in routing.
            connect_port(IT[2].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[0], rope_core.ROPE_in, pkt_id = it1_pkt_id.to_rope, keep_pkt_header = False)

            #fc in
            connect_port(IT[3].MM2S[0], pass_core.X_in)
            connect_port(IT[3].MM2S[1], pass_core.W_in)

            connect_port(pass_core.Y_out, proj_engine.X_in, pkt_id = pass_pkt_id.to_proj, keep_pkt_header = False)

            # fc_out north to the rms tile -- and down_proj's row too, at the
            # end of the pass. Both leave the engine under to_rms and land on
            # this one channel; the tile's ping/pong chain orders them by
            # arrival, as it does o arriving from MT[2] in between. One flow,
            # three phases. (see iron/constants.py for why down cannot have an
            # id of its own here)
            connect_port(proj_engine.Y_out, rms_core.X_in, pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)

            # Both norms leave west under one id: phase 1's xn feeds qkv and z,
            # phase 2's post-attention norm feeds up | gate. 
            connect_port(rms_core.Y_out, proj_engine.X_in, pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)

            connect_port(proj_engine.Y_out, rope_core.QKV_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)

            proj_engine.route()

            connect_port(proj_engine.Y_out, aux_mt.X_in, pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)

            # k and v to each column's own shim, on *separate* S2MM channels.
            #
            # One channel per plane is what removes the cache's context ceiling:
            # the host then walks each plane with its own BD, whose only stride
            # is one row, instead of one descriptor crossing k -> v with a
            # MAX_L * DK stride that a 20-bit BD field masks past MAX_L = 2048.
            # See iron/RoPE.py's KV_out chain and
            # host/decoding_layer.cpp::_receive_kv_cache.
            #
            # S2MM[1] was free because the attention output leaves north out of
            # MT[3] to the gate core rather than south to these shims -- the
            # column shim is write-only for the cache and read-only for the
            # replay (MM2S[0] k, MM2S[1] v, see attn.py::attn_column.route).
            for col, column in enumerate(attn_columns):
                connect_port(rope_core.KV_out, column.IT.S2MM[0],
                             pkt_id = rope_core.k_pkt_base + col, keep_pkt_header = False)
                connect_port(rope_core.KV_out, column.IT.S2MM[1],
                             pkt_id = rope_core.v_pkt_base + col, keep_pkt_header = False)

            # q, whole and padded, to MT[3] -- which fans it out to the four CUs
            # as one contiguous GQA group each.
            connect_port(rope_core.Q_out, attn_hub.Q_in)
            for cu in range(model_config.num_attn_columns * model_config.cus_per_column):
                column, cu_in_column = divmod(cu, model_config.cus_per_column)
                connect_port(attn_hub.Q_out(cu), attn_columns[column].Q_in[cu_in_column])

            # Each column's internal flows: shim -> memtile -> CUs, and o back.
            for column in attn_columns:
                column.route()

            # The two columns' half-rows into MT[3], and the assembled DQ-wide
            # attention output on to the gate tile.
            for col, column in enumerate(attn_columns):
                connect_port(column.O_out, attn_hub.O_in(col))

            # o *= sigmoid(z). 
            connect_port(attn_hub.O_out, gate_core.Attn_In)
            connect_port(aux_mt.Gate_out, gate_core.Gate_In)

            # o_proj's x.
            connect_port(gate_core.Attn_Out, proj_engine.X_in, pkt_id = aux_to_proj_x_in_pkt_id.to_out_proj, keep_pkt_header = False)

            # h = fc_out + W_o @ o, then post_attention_layernorm.
            connect_port(aux_mt.OutDown_out, rms_core.X_in, pkt_id = aux_to_rms_pkt_id.to_residual, keep_pkt_header = False)

            # up | gate, a round at a time, straight to the SwiGLU tile.
            connect_port(proj_engine.Y_out, glu.Up_Gate_in, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

            # The SwiGLU result becomes down_proj's x
            connect_port(glu.Hid_out, proj_engine.X_in, pkt_id = aux_to_proj_x_in_pkt_id.to_down_proj, keep_pkt_header = False)

            connect_port(rms_core.Y_out, IT[2].S2MM[1], pkt_id = rms_pkt_id.to_out, keep_pkt_header = False)

        print(mctx.module)

if __name__ == "__main__":
    model_type = "LLAMA_3_2_1B"
    if len(sys.argv) != 2:
        print("Usage: python flowArch_aie.py <model_type>, Options: LLAMA_3_2_1B, LLAMA_3_2_3B, LLAMA_3_1_8B")
        print("Use default model type: LLAMA_3_2_1B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)
