# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""rmsglu: stage design of the 8-row verify layer (plan S6.6 / S6.7, D17): the projection engine + the norm tile with the resident residual +
the GLU, fed and drained by the shim where the attention / RoPE / PLE stages will sit.

  shim x[8][D] --pkt PKT_SHIM_TO_RMS--> norm tile (3,2)      phase 1: h = x (lent on (3,3)), y = n_in(h)
  norm tile --pkt PKT_RMS_TO_MAIN--> MT3 (replay) --> 16 Q4 cores           qkv projection (y -> shim, stands in for RoPE/attention)
  shim o_x[8][DQ] --pkt to_o_global / to_o_swa--> MT3 --> cores             o projection (stands in for the attention output)
  cores y (id to_rms) --> MT5 y_rows_mt --> norm tile                       phase 2: h += n_pa(o), y = n_pff(h) --> MT3 --> cores (up|gate)
  cores y (id to_glu) --> MT4 glu_stage_mt --> GLU (4,4) --pkt to_down--> MT3 (down buffer, rounds)   --> cores (down)
  cores y (id to_rms) --> MT5 --> norm tile                                 phase 3: h += n_pf(down); rows --pkt PKT_RMS_OUT--> shim

The up/gate/down weights, the o/qkv weights and the norm weights are the real ones: the host drives the same weight streams as the decode layer.
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
from proj_tile import q4_proj_engine, x_replay_phase
PKT_TAP_ROWS = 28
PKT_TAP_UG = 29
from rmsglu_tiles import (rms_residual_l8_tile, glu_l8_tile, glu_stage_mt, y_rows_mt, proj_x_replay_mt_glu,
                          PKT_RMS_TO_MAIN, PKT_RMS_OUT, PKT_ROWS_TO_RMS, PKT_SHIM_TO_RMS)


def flow_arch(model_type: str = "GEMMA4_E4B"):
    assert L_TOKENS == 8, "the rmsglu stage design is the 8-row design"
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    print(f"// L_TOKENS={L_TOKENS} KERNEL_L={KERNEL_L}")
    D = model_config.D
    DQ, DK, DV = model_config.DQ, model_config.DK, model_config.DV
    SWA_DQ = model_config.SWA_DQ
    I = model_config.intermediate_size
    cores, m = 16, model_config.q4nx_row_block_size
    qkv_replays = (DQ + DK + DV) // cores // m
    o_replays = D // cores // m
    up_replays = 2 * I // cores // m
    down_replays = D // cores // m
    glu_rounds = I // (cores * m)

    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)
            ctx = mlir_context()
            ctx.data_ty('RTP_ty', (16,), np.dtype[np.int32])

            proj_cores, proj_assemble_mts, proj_shim_tiles = [], [], []
            for col in list(range(0, 2)) + list(range(6, 8)):
                proj_cores.append([CT[row][col] for row in range(0, 4)])
                proj_assemble_mts.append(MT[col])
                proj_shim_tiles.append(IT[col])
            proj_engine = q4_proj_engine(proj_cores, proj_assemble_mts, proj_shim_tiles, 10 * 1024, ["proj_main.o", "proj.o"], ctx,
                                         model_config, replay_x = True)

            down = x_replay_phase(I, down_replays, "down", dedicated = True)
            down.glu_rounds = glu_rounds
            replay = proj_x_replay_mt_glu(MT[3], ctx, model_config, phases = [
                x_replay_phase(D, qkv_replays, "qkv"),
                x_replay_phase(DQ, o_replays, "o_global", dedicated = True),
                x_replay_phase(SWA_DQ, o_replays, "o_swa", dedicated = True),
                x_replay_phase(D, up_replays, "up_gate"),
                down,
            ])

            # norm tile CT[0][3] = (col 3, row 2); the residual rows live on CT[1][3] = (3, 3), its north neighbour (no core there any more)
            rms = rms_residual_l8_tile(CT[0][3], ctx, model_config, lender = CT[1][3])
            # GLU on a free tile
            glu = glu_l8_tile(CT[2][4], ctx, model_config)
            taps = os.environ.get("RMSGLU_TAPS", "0") == "1"
            glu_stage = glu_stage_mt(MT[4], ctx, model_config, tap_pkt_id = PKT_TAP_UG if taps else None)
            rows_stage = y_rows_mt(MT[5], ctx, model_config, rounds = D // (cores * m), out_pkt_id = PKT_ROWS_TO_RMS,
                                   tap_pkt_id = PKT_TAP_ROWS if taps else None)

            # ---- inputs from the shim ----
            connect_port(IT[3].MM2S[0], rms.X_in, pkt_id = PKT_SHIM_TO_RMS, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], rms.W_in)
            connect_port(IT[3].MM2S[0], replay.x_in("o_global"), pkt_id = proj_x_in_pkt_id.to_o_global, keep_pkt_header = False)
            connect_port(IT[3].MM2S[0], replay.x_in("o_swa"), pkt_id = proj_x_in_pkt_id.to_o_swa, keep_pkt_header = False)
            # ---- norm tile output ----
            connect_port(rms.Y_out, replay.x_in("qkv"), pkt_id = PKT_RMS_TO_MAIN, keep_pkt_header = False)
            connect_port(rms.Y_out, IT[3].S2MM[1], pkt_id = PKT_RMS_OUT, keep_pkt_header = False)
            # ---- projection engine ----
            replay.route_to(proj_engine.X_in)
            proj_engine.route()
            connect_port(proj_engine.Y_out, IT[3].S2MM[0], pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, IT[3].S2MM[0], pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, rows_stage.y_in, pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, glu_stage.y_in, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)
            # ---- stages ----
            connect_port(rows_stage.y_out, rms.X_in, pkt_id = PKT_ROWS_TO_RMS, keep_pkt_header = False)
            connect_port(glu_stage.y_out, glu.X_in)
            connect_port(glu.Y_out, replay.x_in("down"), pkt_id = proj_x_in_pkt_id.to_down, keep_pkt_header = False)
            if taps:
                # debug taps (RMSGLU_TAPS=1): the raw o / down projection rounds and the raw up|gate rounds, copied by the memtile stages
                connect_port(rows_stage.tap_out, IT[4].S2MM[0], pkt_id = PKT_TAP_ROWS, keep_pkt_header = False)
                connect_port(glu_stage.tap_out, IT[4].S2MM[1], pkt_id = PKT_TAP_UG, keep_pkt_header = False)

        print(mctx.module)


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B"
    flow_arch(model_type)
