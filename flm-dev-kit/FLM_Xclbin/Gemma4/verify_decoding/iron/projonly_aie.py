# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""projonly: the projection engine of the verify layer on its own (plan S6.3/S6.4, debug design).

16 Q4 projection cores + their assembly memtiles + the x replay tile (MT3), fed and drained by the shim: the host sends each phase's
x token-major as one packet, reads the y packets of every round back.  Nothing else of the layer (rms, rope, attention, glu, PLE) is
in the design, so a projection can be checked against a numpy reference with the real weights and timed in isolation.

  host -> IT3.MM2S0 --pkt--> MT3 (landing + replay) --pkt--> MT1 x FIFO --circuit--> 16 cores
  16 cores --y packets (id per phase)--> MT0/MT6 --> MT1 --> IT3.S2MM0 -> host

Phases of one layer (same order and shapes as the fused layer): qkv (D, 12 replays) | o (4096 or 2048 wide, 5) | up|gate (D, 40) | down (10240, 5).
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
from proj_tile import q4_proj_engine, proj_x_replay_mt, x_replay_phase, y_depermute_mt


def flow_arch(model_type: str = "GEMMA4_E4B", stage: bool = False):
    """stage=False: y packets straight to the shim (DDR-side de-permute); stage=True: y -> MT4 token-major stage -> shim IT4."""
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    print(f"// L_TOKENS={L_TOKENS} KERNEL_L={KERNEL_L}")
    D = model_config.D
    DQ, DK, DV = model_config.DQ, model_config.DK, model_config.DV
    SWA_DQ, SWA_DK, SWA_DV = model_config.SWA_DQ, model_config.SWA_DK, model_config.SWA_DV
    I = model_config.intermediate_size
    cores, m = 16, model_config.q4nx_row_block_size
    # replays of x per phase = rounds of 16 x 32 rows; qkv is fixed at the global layer's count (other variants drain the surplus)
    qkv_replays = (DQ + DK + DV) // cores // m
    o_replays = D // cores // m
    up_replays = 2 * I // cores // m
    down_replays = D // cores // m

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

            replay = proj_x_replay_mt(MT[3], ctx, model_config, phases = [
                x_replay_phase(D, qkv_replays, "qkv"),
                x_replay_phase(DQ, o_replays, "o_global", dedicated = True),
                x_replay_phase(SWA_DQ, o_replays, "o_swa", dedicated = True),
                x_replay_phase(D, up_replays, "up_gate"),
                x_replay_phase(I, down_replays, "down", dedicated = True),
            ])

            # host -> landing channels, one packet id per landing buffer
            connect_port(IT[3].MM2S[0], replay.x_in("qkv"), pkt_id = proj_x_in_pkt_id.to_main, keep_pkt_header = False)
            connect_port(IT[3].MM2S[0], replay.x_in("o_global"), pkt_id = proj_x_in_pkt_id.to_o_global, keep_pkt_header = False)
            connect_port(IT[3].MM2S[0], replay.x_in("o_swa"), pkt_id = proj_x_in_pkt_id.to_o_swa, keep_pkt_header = False)
            connect_port(IT[3].MM2S[0], replay.x_in("down"), pkt_id = proj_x_in_pkt_id.to_down, keep_pkt_header = False)
            # replay channels -> the x FIFO on main_mt -> cores
            replay.route_to(proj_engine.X_in)
            proj_engine.route()
            # y packets of every phase -> host
            if stage:
                ystage = y_depermute_mt(MT[4], ctx, model_config)
                for pid in (proj_pkt_id.to_rope, proj_pkt_id.to_swa_rope, proj_pkt_id.to_rms, proj_pkt_id.to_glu):
                    connect_port(proj_engine.Y_out, ystage.y_in, pkt_id = pid, keep_pkt_header = False)
                connect_port(ystage.y_out, IT[4].S2MM[0])
            else:
                for pid in (proj_pkt_id.to_rope, proj_pkt_id.to_swa_rope, proj_pkt_id.to_rms, proj_pkt_id.to_glu):
                    connect_port(proj_engine.Y_out, IT[3].S2MM[0], pkt_id = pid, keep_pkt_header = False)

        print(mctx.module)


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B"
    flow_arch(model_type)
