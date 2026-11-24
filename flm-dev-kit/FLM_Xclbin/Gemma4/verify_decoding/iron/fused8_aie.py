# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""fused8: the whole 8-row verify layer (plan S6.11, D17), merging the stage designs rmsglu / ple8 / attn8 / projstage.

  shim x[8][D] --26--> norm tile (3,2)  phase 1: h = x (lender (3,3)), y = n_in(h)     --16--> MT3 replay --> 16 Q4 cores: qkv
  cores y --1/2--> MT4 y stage (global [t][512] / SWA [head][t][256]) --> RoPE (3,4)/(3,5) --> q --> MT4 q park --> qk tile; k/v --> IT2 (append)
  attention (col 2)  o per token --17/18--> MT3 o landing (= o park) --> cores: o
  cores y --4--> MT5 rows stage --25--> norm tile  phase 2: h += n_pa(o), y = n_pff(h) --16--> MT3 --> cores: up|gate
  cores y --8--> MT4 GLU stage --> GLU (4,4) --19--> MT3 down buffer --> cores: down
  cores y --4--> MT5 rows --25--> norm tile  phase 3: h += n_pf(down), rows --24--> MT5 ple feed
  ple feed (pli, constants from the shim) --> G (4,2) / N (5,3);  W_gate, W_up from the shim;  G -> U (5,2) -> MT5 ple stage -> N -> IT5.S2MM0 (layer output)

FUSED8_STAGE (env): `attn` = up to the attention output (o rows leave on IT3.S2MM0), `nople` = whole layer without PLE (h2 rows leave on IT3.S2MM0),
`full` (default) = the whole layer.  See notes/fused8_allocation.md for the allocation table.
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
from rmsglu_tiles import rms_residual_l8_tile, glu_l8_tile, PKT_RMS_TO_MAIN, PKT_RMS_OUT, PKT_ROWS_TO_RMS, PKT_SHIM_TO_RMS
from ple8_tiles import ple8_gate_tile, ple8_up_tile, ple8_norm_tile
from rope8 import RoPE8, SWA_RoPE8
from attn import attn_column
from fused8_mt import (memtile_hub, pingpong_stage, q_park_stage, rows_stage, ple_stage, ple_feed, proj_x_replay_mt_f8)
from attnx import attnx_score_tile, attnx_pv_tile, attnx_oasm, attnx_q_park_stage, PKT_Q, PKT_O

PKT_ROPE_TABLE = 9        # IT3.MM2S0 -> global RoPE table
PKT_SWA_ROPE_TABLE = 10   # IT3.MM2S1 -> SWA RoPE table
PKT_RMS_WEIGHTS = 11      # IT3.MM2S1 -> norm tile weights
PKT_PLI = 27              # IT4.MM2S0 -> MT5 ple feed
PKT_PLE_CONST = 28        # IT4.MM2S1 -> MT5 ple feed


def flow_arch(model_type: str = "GEMMA4_E4B"):
    assert L_TOKENS == 8, "fused8 is the 8-row design"
    stage = os.environ.get("FUSED8_STAGE", "full")
    assert stage in ("attn", "nople", "full")
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    print(f"// L_TOKENS={L_TOKENS} KERNEL_L={KERNEL_L} stage={stage}")
    D = model_config.D
    DQ, DK, DV = model_config.DQ, model_config.DK, model_config.DV
    SWA_DQ = model_config.SWA_DQ
    I = model_config.intermediate_size
    pli_d = model_config.per_layer_input_dim
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

            # ---------------- projection engine + x replay (MT3) ----------------
            proj_cores, proj_assemble_mts, proj_shim_tiles = [], [], []
            for col in list(range(0, 2)) + list(range(6, 8)):
                proj_cores.append([CT[row][col] for row in range(0, 4)])
                proj_assemble_mts.append(MT[col])
                proj_shim_tiles.append(IT[col])
            proj_engine = q4_proj_engine(proj_cores, proj_assemble_mts, proj_shim_tiles, 10 * 1024, ["proj_main.o", "proj.o"], ctx,
                                         model_config, replay_x = True)

            full_ffn = stage != "attn"
            # Memtile BD ids are split by channel parity (even channels: ids < 24, odd: >= 24, S2MM and MM2S share them).  MT3: even = main 2 + down 20
            # + replay MM2S0/2 = 24, odd = o_global 8 + o_swa 8 + replay MM2S1/3 = 18, hence this phase order (landing channels follow it).
            qkv = x_replay_phase(D, qkv_replays, "qkv")
            o_g = x_replay_phase(DQ, o_replays, "o_global", dedicated = True)
            o_s = x_replay_phase(SWA_DQ, o_replays, "o_swa", dedicated = True)
            o_g.land_rounds = L_TOKENS
            o_s.land_rounds = L_TOKENS
            if full_ffn:
                down = x_replay_phase(I, down_replays, "down", dedicated = True)
                down.glu_rounds = glu_rounds
                phases = [qkv, o_g, down, o_s, x_replay_phase(D, up_replays, "up_gate")]
            else:
                phases = [qkv, o_g, o_s]
            replay = proj_x_replay_mt_f8(MT[3], ctx, model_config, phases = phases)

            # ---------------- norm tile + lender, RoPE, attention ----------------
            rms = rms_residual_l8_tile(CT[0][3], ctx, model_config, lender = CT[1][3])
            rope_core = RoPE8(CT[2][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with)
            swa_rope_core = SWA_RoPE8(CT[3][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.swa_rope_link_with)
            # attention: attnx (default; K/V once per verify: score tiles (4,3)/(4,5), PV tiles col 2, o assembly MT2) or the attn8 column
            attn_kind = os.environ.get("FUSED8_ATTN", "x")   # "x" = attnx (shipped), "8" = per-token K/V replay (A/B only)
            assert attn_kind in ("x", "8")
            if attn_kind == "8":
                attn = attn_column(
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
            else:
                score = [attnx_score_tile(CT[1][4], ctx), attnx_score_tile(CT[3][4], ctx)]
                pv = [attnx_pv_tile(CT[j][2], ctx, j) for j in range(4)]
                oasm = attnx_oasm(MT[2], ctx, DQ, SWA_DQ)

            # ---------------- memtile hubs ----------------
            hubA = memtile_hub(MT[4], ctx, "hubA")
            if attn_kind == "8":
                qpark = q_park_stage(hubA, model_config.num_heads, model_config.DH, model_config.SWA_DH, g = 0, s = 1)
            else:
                qpark = attnx_q_park_stage(hubA, model_config.num_heads, model_config.DH, model_config.SWA_DH, g = 0, s = 1)
            y_g = pingpong_stage(hubA, "y_stage_g", s2mm = 2, mm2s = 2, kind = "tokmajor")
            y_s = pingpong_stage(hubA, "y_stage_s", s2mm = 3, mm2s = 3, kind = "half")
            if full_ffn:
                glu_stage = pingpong_stage(hubA, "glu_stage", s2mm = 4, mm2s = 4, kind = "half")
                hubB = memtile_hub(MT[5], ctx, "hubB")
                rows = rows_stage(hubB, rounds = D // (cores * m), out_pkt_id = PKT_ROWS_TO_RMS, s2mm = 0, mm2s = 0)
                if stage == "full":
                    ple_st = ple_stage(hubB, D, s2mm = 1, mm2s = 1)
                    feed = ple_feed(hubB, D, pli_d, s2mm_pli = 2, s2mm_h = 3, s2mm_const = 4, mm2s_g = 2, mm2s_n = 3)
                glu = glu_l8_tile(CT[2][4], ctx, model_config)
                if stage == "full":
                    G = ple8_gate_tile(CT[0][4], ctx, D, pli_d)
                    U = ple8_up_tile(CT[0][5], ctx, pli_d)
                    N = ple8_norm_tile(CT[1][5], ctx, D)
            hubA.finish()
            if full_ffn:
                hubB.finish()

            # ---------------- inputs from the shim ----------------
            connect_port(IT[3].MM2S[0], rms.X_in, pkt_id = PKT_SHIM_TO_RMS, keep_pkt_header = False)
            connect_port(IT[3].MM2S[0], rope_core.ROPE_in, pkt_id = PKT_ROPE_TABLE, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], rms.W_in, pkt_id = PKT_RMS_WEIGHTS, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], swa_rope_core.ROPE_in, pkt_id = PKT_SWA_ROPE_TABLE, keep_pkt_header = False)
            # ---------------- norm tile -> projection engine ----------------
            connect_port(rms.Y_out, replay.x_in("qkv"), pkt_id = PKT_RMS_TO_MAIN, keep_pkt_header = False)
            replay.route_to(proj_engine.X_in)
            proj_engine.route()
            # ---------------- qkv: y stage -> RoPE -> q park / KV append -> attention -> o landing ----------------
            connect_port(proj_engine.Y_out, y_g.y_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(proj_engine.Y_out, y_s.y_in, pkt_id = proj_pkt_id.to_swa_rope, keep_pkt_header = False)
            connect_port(y_g.y_out, rope_core.QKV_in)
            connect_port(y_s.y_out, swa_rope_core.QKV_in)
            connect_port(rope_core.Q_out, qpark.Q_in)
            connect_port(swa_rope_core.Q_out, qpark.SWA_Q_in)
            connect_port(rope_core.KV_out, IT[2].S2MM[0])
            connect_port(swa_rope_core.KV_out, IT[2].S2MM[1])
            if attn_kind == "8":
                connect_port(qpark.Q_out, attn.Q_in)
                connect_port(qpark.SWA_Q_out, attn.SWA_Q_in)
                o_g, o_s = attn.O_out, attn.SWA_O_out
            else:
                for sc in range(2):
                    connect_port(qpark.Q_out, score[sc].Q_in, pkt_id = PKT_Q[sc], keep_pkt_header = False)
                    connect_port(qpark.SWA_Q_out, score[sc].Q_in, pkt_id = PKT_Q[sc], keep_pkt_header = False)
                    connect_port(IT[2].MM2S[0], score[sc].K_in)            # K once, multicast
                    for jj in range(2):
                        connect_port(score[sc].P_out[jj], pv[2 * sc + jj].P_in)
                for j in range(4):
                    connect_port(IT[2].MM2S[1], pv[j].V_in)                # V once, multicast
                    connect_port(pv[j].O_out, oasm.G_in(j))
                for sc in range(2):
                    connect_port(pv[2 * sc].SWA_O_out, oasm.S_in(sc))
                o_g, o_s = oasm.G_out, oasm.S_out
            if full_ffn:
                connect_port(o_g, replay.x_in("o_global"), pkt_id = to_proj_x_in_pkt_id.from_attn, keep_pkt_header = False)
                connect_port(o_s, replay.x_in("o_swa"), pkt_id = to_proj_x_in_pkt_id.from_swa_attn, keep_pkt_header = False)
            else:
                # stage `attn`: the o rows leave on the shim (no o projection)
                connect_port(o_g, IT[3].S2MM[0], pkt_id = to_proj_x_in_pkt_id.from_attn, keep_pkt_header = False)
                connect_port(o_s, IT[3].S2MM[0], pkt_id = to_proj_x_in_pkt_id.from_swa_attn, keep_pkt_header = False)
            if attn_kind == "8":
                attn.route()
            if full_ffn:
                # ---------------- o / down rows -> norm tile; up|gate -> GLU -> down ----------------
                connect_port(proj_engine.Y_out, rows.y_in, pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
                connect_port(rows.y_out, rms.X_in, pkt_id = PKT_ROWS_TO_RMS, keep_pkt_header = False)
                connect_port(proj_engine.Y_out, glu_stage.y_in, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)
                connect_port(glu_stage.y_out, glu.X_in)
                connect_port(glu.Y_out, replay.x_in("down"), pkt_id = proj_x_in_pkt_id.to_down, keep_pkt_header = False)
                if stage == "nople":
                    connect_port(rms.Y_out, IT[3].S2MM[0], pkt_id = PKT_RMS_OUT, keep_pkt_header = False)
                else:
                    connect_port(rms.Y_out, feed.h_in, pkt_id = PKT_RMS_OUT, keep_pkt_header = False)
                    # ---------------- PLE ----------------
                    connect_port(IT[4].MM2S[0], feed.pli_in, pkt_id = PKT_PLI, keep_pkt_header = False)
                    connect_port(IT[4].MM2S[1], feed.const_in, pkt_id = PKT_PLE_CONST, keep_pkt_header = False)
                    connect_port(feed.to_G, G.X_in)
                    connect_port(feed.to_N, N.HW_in)
                    connect_port(IT[5].MM2S[0], G.W_in)
                    connect_port(G.U_out, U.U_in)
                    connect_port(IT[5].MM2S[1], U.W_in)
                    connect_port(U.Y_out, ple_st.y_in)
                    connect_port(ple_st.up_out, N.Up_in)
                    connect_port(N.O_out, IT[5].S2MM[0])

        print(mctx.module)


if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B")
