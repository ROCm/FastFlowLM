# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""ple8: the per layer input epilogue for 8 token rows on its own (plan S6.8, debug/stage design).

  host -> IT4.MM2S0 -> G.S2MM0   pli rows, then 8 x 10 h slices           (G = gate tile, 4,2)
  host -> IT5.MM2S0 -> G.S2MM1   W_gate (80 blocks)
  G.MM2S0 -> U.S2MM0             u = gelu(gate) * pli                      (U = up tile, 5,2)
  host -> IT5.MM2S1 -> U.S2MM1   W_up (80 blocks)
  U.MM2S0 -> MT5 -> N.S2MM0      80 packets [rb][t][32], read back token-major
  host -> IT4.MM2S1 -> N.S2MM1   [w_post_pli | layer_scalar], then 8 h rows (N = norm tile, 5,3)
  N.MM2S0 -> IT5.S2MM0 -> host   out rows [8][2560]

The fused design's PLE tiles (4,3) pli and (5,3) merge are gone (pli comes from the host); G, U, N take their places.
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
from ple8_tiles import ple8_gate_tile, ple8_up_tile, ple8_norm_tile, ple8_stage_mt


def flow_arch(model_type: str = "GEMMA4_E4B"):
    stage = os.environ.get("PLE8_STAGE", "full")      # gate: u -> shim | up: token-major up rows -> shim | full: the whole stage
    assert stage in ("gate", "up", "full")
    dev = aie.AIEDevice.npu2
    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    D, pli_d = model_config.D, model_config.per_layer_input_dim
    assert pli_d == 256 and D % 256 == 0

    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(6, 8)
            ctx = mlir_context()

            G = ple8_gate_tile(CT[0][4], ctx, D, pli_d)
            connect_port(IT[4].MM2S[0], G.X_in)
            connect_port(IT[5].MM2S[0], G.W_in)
            if stage == "gate":
                connect_port(G.U_out, IT[5].S2MM[0])
                return
            U = ple8_up_tile(CT[0][5], ctx, pli_d)
            S = ple8_stage_mt(MT[5], ctx, D)
            connect_port(G.U_out, U.U_in)
            connect_port(IT[5].MM2S[1], U.W_in)
            connect_port(U.Y_out, S.y_in)
            if stage == "up":
                connect_port(S.up_out, IT[5].S2MM[0])
                return
            N = ple8_norm_tile(CT[1][5], ctx, D)
            connect_port(S.up_out, N.Up_in)
            connect_port(IT[4].MM2S[1], N.HW_in)
            connect_port(N.O_out, IT[5].S2MM[0])
        print(mctx.module)


if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B")
