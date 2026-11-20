# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""attnx: stand-alone stage design of the 8-token attention that reads K/V once per verify (stream A, notes/attn_longctx.md).

  shim IT3 (global) / IT4 (SWA): qkv rows + RoPE table -> RoPE (3,4) / (3,5) -> q -> q park MT4 -> score tiles (4,3) / (4,5)
                                                                              -> k/v append -> IT2.S2MM0 / S2MM1
  IT2.MM2S0: K head-major, once (multicast to both score tiles); IT2.MM2S1: V (multicast to the 4 PV tiles (2,2)..(2,5))
  score tile s -> P -> PV tiles 2s, 2s+1 (SWA: PV 2s only);  PV tiles -> o (circuits) -> MT2 o assembly -> 8 token rows (packet 17 global
  -> IT3.S2MM0, 18 SWA -> IT4.S2MM0)

The column-2 tiles are the decode design's attention tiles; the score tiles use two of the tiles fused8 leaves free.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iron.model_config import get_model_config
from iron.models import *
from constants import *

from aie.extras.context import mlir_mod_ctx
import aie.dialects.aie as aie

from utils import generate_tile_map, mlir_context, connect_port
from rope8 import RoPE8, SWA_RoPE8
from attnx import attnx_score_tile, attnx_pv_tile, attnx_qpark, attnx_oasm, PKT_Q, PKT_O

PKT_QKV = 1
PKT_TABLE = 2


def flow_arch(model_type: str = "GEMMA4_E4B"):
    assert L_TOKENS == 8
    dev = aie.AIEDevice.npu2
    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(6, 8)
            ctx = mlir_context()
            ctx.data_ty('RTP_ty', (16,), np.dtype[np.int32])

            rope = RoPE8(CT[2][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with)
            swa_rope = SWA_RoPE8(CT[3][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.swa_rope_link_with)
            qpark = attnx_qpark(MT[4], ctx, model_config.num_heads, model_config.DH, model_config.SWA_DH)
            score = [attnx_score_tile(CT[1][4], ctx), attnx_score_tile(CT[3][4], ctx)]
            pv = [attnx_pv_tile(CT[j][2], ctx, j) for j in range(4)]

            # RoPE feeds (as the attn8 stage)
            connect_port(IT[3].MM2S[0], rope.QKV_in, pkt_id = PKT_QKV, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], rope.ROPE_in, pkt_id = PKT_TABLE, keep_pkt_header = False)
            connect_port(IT[4].MM2S[0], swa_rope.QKV_in, pkt_id = PKT_QKV, keep_pkt_header = False)
            connect_port(IT[4].MM2S[1], swa_rope.ROPE_in, pkt_id = PKT_TABLE, keep_pkt_header = False)
            connect_port(rope.KV_out, IT[2].S2MM[0])
            connect_port(swa_rope.KV_out, IT[2].S2MM[1])
            # q -> park -> score tiles
            connect_port(rope.Q_out, qpark.Q_in)
            connect_port(swa_rope.Q_out, qpark.SWA_Q_in)
            for s in range(2):
                connect_port(qpark.Q_out, score[s].Q_in, pkt_id = PKT_Q[s], keep_pkt_header = False)
                connect_port(qpark.SWA_Q_out, score[s].Q_in, pkt_id = PKT_Q[s], keep_pkt_header = False)
            # K / V once, multicast
            for s in range(2):
                connect_port(IT[2].MM2S[0], score[s].K_in)
            for j in range(4):
                connect_port(IT[2].MM2S[1], pv[j].V_in)
            # P
            for s in range(2):
                for jj in range(2):
                    connect_port(score[s].P_out[jj], pv[2 * s + jj].P_in)
            # o: PV tiles -> assembly memtile MT2 (circuits) -> 8 token rows per kind -> shim (global IT3.S2MM0, SWA IT4.S2MM0)
            oasm = attnx_oasm(MT[2], ctx, model_config.DQ, model_config.SWA_DQ)
            for j in range(4):
                connect_port(pv[j].O_out, oasm.G_in(j))
            for s in range(2):
                connect_port(pv[2 * s].SWA_O_out, oasm.S_in(s))
            connect_port(oasm.G_out, IT[3].S2MM[0], pkt_id = PKT_O[0], keep_pkt_header = False)
            connect_port(oasm.S_out, IT[4].S2MM[0], pkt_id = PKT_O[1], keep_pkt_header = False)

        print(mctx.module)


if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B")
