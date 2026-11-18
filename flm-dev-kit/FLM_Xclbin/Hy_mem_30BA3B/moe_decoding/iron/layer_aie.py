# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import iron as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import get_model_config
from iron.models import *

from constants import *

import aie.dialects.aie as aie
from aie.dialects.aie import WireBundle, packetflow
from aie.extras.context import mlir_mod_ctx

from utils import generate_tile_map, mlir_context, connect_port
from proj_tile import q4_proj_engine, ENABLE_CHESSCC
from rms_residual import rms_residual_tile
from aux_memtile import aux_memtile
from router_w_memtile import router_w_memtile
from glu import glu_tile
from RoPE import RoPE
from attn import attn_block
from router_tile import Router_tile
from router_helper import router_helper

# One attention + MoE decoder layer for one token.
#
#        col 0  col 1  col 2         col 3          col 4          col 5          col 6  col 7
# row 5  proj   proj   -             attn kv (CU1)  attn kv (CU3)  GLU            proj   proj
# row 4  proj   proj   -             attn qk (CU1)  attn qk (CU3)  router         proj   proj
# row 3  proj   proj   RoPE+qk-norm  attn kv (CU0)  attn kv (CU2)  router helper  proj   proj
# row 2  proj   proj   rms+residual  attn qk (CU0)  attn qk (CU2)  -              proj   proj
# MT     gather proj   aux           KV, O join     KV, O half     router w       gather proj w
# IT     w      w      x/norms/rope  KV (heads 0,1) KV (heads 2,3) router w       w      w
#                                    O tap (opt.)

def flow_arch(model_type: str, attn_tap: bool = False, debug_tap: str = ""):
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4
    # control packet ids the router uses to patch the expert weight BDs of
    # shim 0, 1, 6, 7 (one per proj column), and to read shim 6 back
    IT_CONTROL_packet_list = [29, 25,-1, -1, -1, -1, 22, 31 ]
    IT_CONTRO_response_list = [-1,-1, -1, -1, -1, -1, 30, -1]
    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)

            ctx = mlir_context()
            ctx.data_ty('L_ty', (16,), np.dtype[np.int32]) # for all RTP requests
            ctx.parameter('router_weight_row_block', 32)
            ctx.parameter("router_weight_col_block", value=256)

            CT_from_shimtile_control_bd_port = 1
            CT_to_shimtile_control_bd_port = 1

            ctx.parameter("CT_from_shimtile_control_bd_port", CT_from_shimtile_control_bd_port)
            ctx.parameter("CT_to_shimtile_control_bd_port", CT_to_shimtile_control_bd_port)
            # Instantiate IPs with single CT/MT
            rms_core = rms_residual_tile(CT[0][2], ctx, model_config, stack_size = 1024 * 8, link_with = model_config.rms_residual_link_with)
            rope_core = RoPE(CT[1][2], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with)
            glu = glu_tile(CT[3][5], ctx, model_config, stack_size = 20 * 1024, link_with = model_config.glu_link_with)

            router = Router_tile(
                CT[2][5],
                ctx=ctx, model_config=model_config, stack_size=12*1024,
                up_neighbor_top_k_expert_score_buffer_ty=glu.top_k_expert_score_ty,
                up_neighbor_top_k_expert_index_buffer_ty=glu.top_k_expert_index_ty,
                up_neighbor_top_k_expert_score_buffer=glu.top_k_expert_score_buffer,
                up_neighbor_top_k_expert_index_buffer=glu.top_k_expert_index_buffer,
                up_neighbor_hidden_state_in_buffer=glu.D_in_buffer,
                up_neighbor_hidden_state_in_buffer_ty=glu.hidden_state_ty,
                up_neighbor_share_expert_output_buffer = glu.shared_expert_output_buffer,
                up_neighbor_share_expert_output_buffer_ty = glu.shared_expert_output_ty,
                link_with="router.o"
            )
            # works in the router's memory, so it must sit right below the router
            router_helper_core = router_helper(CT[1][5], ctx, model_config, router, stack_size = 1024 * 14, link_with = "router_helper.o")
            aux_mt = aux_memtile(MT[2], ctx, model_config)
            router_w_mt = router_w_memtile(MT[5], ctx, model_config)

            # Instantiate IPs with multiple CTs/MTs
            proj_cores = []
            proj_assemble_mts = []
            proj_shim_tiles = []
            for col in [0, 1, 6, 7]:
                proj_cores_col = []
                for row in range(0, 4):
                    proj_cores_col.append(CT[row][col])
                proj_cores.append(proj_cores_col)
                proj_assemble_mts.append(MT[col])
                proj_shim_tiles.append(IT[col])
            # proj core stack 4 KB: every proj function has a 0-byte stack frame (core_0_2.elf.map),
            # and the 4 KB went to data memory (4-block weight ring + int8 x cache)
            # Peano: 6 KB, its frames are real (deepest chain 4480 B); see README.md, "Peano"
            proj_stack = 4 * 1024 if ENABLE_CHESSCC else 6 * 1024
            proj_engine = q4_proj_engine(proj_cores, proj_assemble_mts, proj_shim_tiles, proj_stack, ["proj_main.o", "proj.o"], ctx, model_config)

            # attention: 2 columns x 2 CUs, one KV head per CU. Column 3 holds
            # KV heads 0, 1 and gathers O; column 4 holds KV heads 2, 3.
            # o_proj needs O once per row-block round of a proj core.
            _attn = attn_block(
                main_column = dict(attn_qk_tiles = [CT[0][3], CT[2][3]], attn_kv_tiles = [CT[1][3], CT[3][3]],
                                   memtile = MT[3], shim_tile = IT[3]),
                second_column = dict(attn_qk_tiles = [CT[0][4], CT[2][4]], attn_kv_tiles = [CT[1][4], CT[3][4]],
                                     memtile = MT[4], shim_tile = IT[4]),
                model_config = model_config,
                ctx = ctx,
                qk_stack_size = 1024,
                kv_stack_size = 1024 if ENABLE_CHESSCC else 4 * 1024,   # Peano: attn_fv's frame is 2 KB
                qk_link_with = model_config.attn_qk_link_with,
                kv_link_with = model_config.attn_kv_link_with,
                o_sends = model_config.D // (model_config.q4nx_row_block_size * 16),
                o_pkt_id = proj_x_pkt_id.from_attn,
                tap = attn_tap,
            )

            # Connect all DMAs, the order of connection matters.
            # Long connection has higher priority in routing.
            connect_port(IT[2].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[0], rope_core.ROPE_in, pkt_id = it1_pkt_id.to_rope, keep_pkt_header = False)

            # x for the proj engine: senders merge by packet id into MT1's x input,
            # and MT1 broadcasts it to all 16 proj cores over one circuit
            # (as in Gemma4 and Qwen3.5 decoding)
            connect_port(rms_core.Y_out, proj_engine.X_in, pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)
            # GLU -> proj, one id: shared expert h and routed up/gate x (all cores
            # use it), and the 4 routed experts' h (each column keeps its own slices)
            connect_port(glu.Intermediate_out, proj_engine.X_in, pkt_id = proj_x_pkt_id.from_glu, keep_pkt_header = False)

            # control packets: router -> shim tile control ports, and the read-back
            for idx, val in enumerate(IT_CONTROL_packet_list):
                if val != -1:
                    packetflow(
                        val,
                        router.ct, WireBundle.DMA, CT_to_shimtile_control_bd_port,
                        IT[idx].st, WireBundle.TileControl, 0
                    )
            for idx, val in enumerate(IT_CONTRO_response_list):
                if val != -1:
                    packetflow(
                        val,
                        IT[idx].st, WireBundle.TileControl, 0,
                        router.ct, WireBundle.DMA, CT_from_shimtile_control_bd_port
                    )
            proj_engine.route()

            connect_port(proj_engine.Y_out, aux_mt.X_in, pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)

            # Q packet ROPE_Q_PKT_ID_BASE + i -> attention CU i
            for i, q_in in enumerate(_attn.Q_in):
                connect_port(rope_core.Q_out, q_in, pkt_id=ROPE_Q_PKT_ID_BASE + i, keep_pkt_header=False)

            connect_port(rope_core.KV_out, IT[3].S2MM[0])

            connect_port(proj_engine.Y_out, aux_mt.GLU_IN, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)
            connect_port(aux_mt.GLU_out, glu.Up_Gate_in)

            connect_port(glu.D_out, rms_core.X_in, pkt_id=rms_pkt_id.from_glu, keep_pkt_header=False)

            connect_port(proj_engine.Y_out, rope_core.QKV_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)

            # attention output -> o_proj
            connect_port(_attn.O_out, proj_engine.X_in, pkt_id = proj_x_pkt_id.from_attn, keep_pkt_header = False)

            # layer output back to the host
            connect_port(rms_core.Y_out, IT[2].S2MM[1], pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)

            connect_port(IT[5].MM2S[0], router_w_mt.W_in, pkt_id = router_w_pkt_id.to_router_w, keep_pkt_header=False)

            # proj results (o_proj, shared down) -> rms. It can land on the same arbiter
            # as x packets into MT1 (a wait loop: every call hung). The router's choice
            # changes from run to run, so no order here guarantees anything; the build
            # reruns aiecc until check_routes.py passes (makefiles/bitstream.mk).
            connect_port(aux_mt.OUT, rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)

            _attn.route()

            connect_port(rms_core.Y_out, glu.D_in, pkt_id=rms_pkt_id.to_glu, keep_pkt_header=False)

            # DEBUG ONLY (debug_tap == "x"): copy rms's x (to proj) to IT4 S2MM 0,
            # to see whether rms sends it
            if debug_tap == "x":
                connect_port(rms_core.Y_out, IT[4].S2MM[0], pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)
            connect_port(router_w_mt.W_out, router.router_w_in)
            # experts 64..127 -> router helper, in parallel with the router's half
            connect_port(router_w_mt.W2_out, router_helper_core.w_in)

            # attention tap (ATTN_TAP=1 build): O copy -> IT3 S2MM 1
            if attn_tap:
                connect_port(_attn.TAP_out, IT[3].S2MM[1])

        print(mctx.module)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3, 4):
        print("Usage: python layer_aie.py <model_type> [attn_tap 0|1] [debug_tap], Options: HY_MEMORY_30B_A3B")
        sys.exit(1)
    flow_arch(sys.argv[1], attn_tap = len(sys.argv) >= 3 and sys.argv[2] == "1",
              debug_tap = sys.argv[3] if len(sys.argv) == 4 else "")
