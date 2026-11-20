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
from proj_tile import q4_proj_engine, proj_x_replay_mt, x_replay_phase
from rms_residual import rms_residual_tile
from aux_memtile import aux_memtile
from linear_aux_tile import linear_aux_memtile
from glu import glu_tile
from gate import gate_tile
from Conv import Conv
from DeltaNetNorm import DeltaNet_Norm
from AB_tile import AB_tile
from Gated_Delta import delta_net_column
from utils import dma_port, connect_port

# One gated delta-net layer of Qwen3.8-27B, whole. Design notes -- tile map,
# activation schedule, routing constraints, the history behind each stage --
# live in ../ARCHITECTURE.md; the section numbers below refer to it.

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

            # ---- single-tile IPs. ARCHITECTURE.md sec 2 (tile map) ----------
            # rms keeps h resident in CT[1][3] and CT[0][2], neither of which
            # gets a core or a mem, so nothing else here has to know.
            rms_core = rms_residual_tile(CT[0][3], ctx, model_config, residual_tiles = [CT[1][3], CT[0][2]], stack_size = 1024 * 4, link_with = model_config.rms_residual_link_with)
            ab_core = AB_tile(CT[1][5], ctx, model_config, stack_size = 1024 * 4, link_with = "ab_proj.o") # phi4 is special
            conv_core = Conv(CT[0][5], ctx, ab_core.y_buffer, model_config, stack_size = 1024 * 8, link_with = "conv.o") # phi4 is special
            delta_net_norm_core = DeltaNet_Norm(CT[3][3], ctx, model_config, stack_size = 1024 * 4, link_with = "delta_net_norm.o") # phi4 is special
            gate_core = gate_tile(CT[2][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.gate_link_with) # phi4 is special
            glu = glu_tile(CT[3][5], ctx, model_config, stack_size = 1 * 1024, link_with = model_config.glu_link_with)

            aux_mt = aux_memtile(MT[3], ctx, model_config)
            linear_aux_mt = linear_aux_memtile(MT[5], ctx, model_config)

            # ---- the 16-core projection engine, columns 0, 1, 6, 7 ----------
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
            proj_engine = q4_proj_engine(proj_cores, proj_assemble_mts, proj_shim_tiles, 10 * 1024, ["proj_main.o"], ctx, model_config)

            # ---- every x the engine reduces over. ARCHITECTURE.md sec 3 -----
            # Phase order is the layer's dependency order. The first three share
            # one 96 KiB buffer; qkvz and up_gate also share the D-wide inbound
            # channel. `replays` must match proj_main()'s M / m loops.
            x_replay_mt = proj_x_replay_mt(MT[2], ctx, model_config, phases = [
                x_replay_phase(
                    name = "qkvz",
                    x_dim = model_config.D,
                    replays = (model_config.linear_DQ + model_config.linear_DK
                               + model_config.linear_DV     # qkv, -> conv memtile
                               + model_config.linear_DV)    # z,   -> the gated norm
                              // 16 // model_config.q4nx_row_block_size,
                ),
                x_replay_phase(
                    name = "o",
                    x_dim = model_config.linear_DV,
                    replays = model_config.D // (model_config.q4nx_row_block_size * 16),
                ),
                x_replay_phase(
                    name = "up_gate",
                    x_dim = model_config.D,
                    replays = 2 * model_config.intermediate_size
                              // (model_config.q4nx_row_block_size * 16),
                ),
                x_replay_phase(
                    name = "down",
                    x_dim = model_config.intermediate_size,
                    replays = model_config.D // (model_config.q4nx_row_block_size * 16),
                    dedicated = True,   # fills while up|gate is still replaying
                    in_block = model_config.q4nx_row_block_size,
                ),
            ])

            _delta_net_column = delta_net_column(
                compute_tiles = [CT[0][4], CT[1][4], CT[2][4], CT[3][4]],
                memtile = MT[4],
                shim_tile = IT[4],
                model_config = model_config,
                ctx = ctx,
                ct_stack_size = 1024 * 8,
                ct_link_with = "gate_delta_net_decode_128_128_128_32.o"
            )

            # =================== wiring ======================================
            # Connection order matters: long connections route first. Two DDR
            # sinks -- h2 and the recurrent state -- plus the arena round trip.
            # ARCHITECTURE.md sec 4 and sec 8.

            # ---- stage 1: input_layernorm. ARCHITECTURE.md sec 5.1 ----------
            # One id for both: the shim channels are told apart by the CT-side
            # S2MM channel they land on.
            connect_port(IT[3].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[3].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)

            # ---- stages 2/3: the qkv|z projection. ARCHITECTURE.md sec 5.2 --
            # route() also brings the q4_k weights in on IT[0,1,6,7].
            connect_port(rms_core.Y_out, x_replay_mt.x_in(0), pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)

            proj_engine.route()

            # ---- stage 5: the causal conv1d. ARCHITECTURE.md sec 5.4 --------
            # x and w take separate channels: an L=8 sliding window over an
            # interleaved x/w layout is not an affine address pattern.
            connect_port(proj_engine.Y_out, linear_aux_mt.QKV_in, pkt_id = proj_pkt_id.to_conv, keep_pkt_header = False)
            connect_port(IT[5].MM2S[1], linear_aux_mt.W_in)
            connect_port(linear_aux_mt.CONV_X_out, conv_core.X_in)
            connect_port(linear_aux_mt.CONV_W_out, conv_core.W_in)
            # The conv state out, the pass's third real output. It is the same
            # CONV_SIZE - 1 pre-conv rows the cache came in as, straight off
            # MT[5] to its own column's shim -- a fresh master port, so it adds
            # no destination set to one that already has two (sec 6). IT[5]
            # S2MM ch0 was free; ch1 is the a/b scatter into the arena.
            connect_port(linear_aux_mt.NEW_CACHE_out, IT[5].S2MM[0])

            # z (stage 6) has no wire in: it is the engine's second mvm off the
            # same resident x, and its y goes to MT[3] with 7b's wiring below.

            # ---- stage 4: the a / b projections. ARCHITECTURE.md sec 5.3 ----
            # All circuit-switched, one producer to one consumer. IT[5].S2MM[1]
            # is not a dump -- the BD scatters alpha|beta into the arena holes,
            # and the delta kernel reads them from there. The tile must also
            # keep draining AB_Yout or rms stalls at row 0.
            connect_port(rms_core.AB_Yout, ab_core.X_in)
            connect_port(IT[5].MM2S[0], ab_core.W_in)
            connect_port(ab_core.Y_out, IT[5].S2MM[1])

            # ---- stage 7a: the delta rule. ARCHITECTURE.md sec 5.6 ----------
            # DDR is the token-major -> head-major reorder buffer; no tile on
            # this path can transpose. IT[6]'s S2MM is otherwise free.
            connect_port(conv_core.Y_out, IT[6].S2MM[0])
            _delta_net_column.route(IT[4].MM2S[1])

            # ---- stage 7b: the gated norm. ARCHITECTURE.md sec 5.7 ----------
            # MT[3] stages o and z, de-permutes both to token-major in the
            # inbound BD and hands out one row at a time. Everything but the
            # ssm_norm hop stays inside column 3.
            connect_port(proj_engine.Y_out, aux_mt.Z_in, pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)
            connect_port(_delta_net_column.O_out, aux_mt.O_in)
            connect_port(aux_mt.Norm_out, delta_net_norm_core.DH_in)
            connect_port(aux_mt.Gate_out, gate_core.Gate_In)
            connect_port(linear_aux_mt.DELTA_NORM_W_out, delta_net_norm_core.W_in)
            connect_port(delta_net_norm_core.DH_out, gate_core.Attn_In, pkt_id = 0, keep_pkt_header = False)

            # ---- stage 7c: ssm_out_proj, residual, post-norm. sec 5.8 -------
            # The gated rows into MT[2] as o's x. Same inbound port as qkv|z's,
            # told apart by arrival order; the id is only for the rms leg that
            # shares the port. Neither residual operand touches DDR.
            connect_port(gate_core.Attn_Out, x_replay_mt.x_in(1), pkt_id = aux_to_proj_x_in_pkt_id.to_out_proj, keep_pkt_header = False)

            # The replay channels' one hop west into MT[1]'s x FIFO -- the
            # single circuit-switched port that broadcasts x, so the 16 cores
            # need no mask/match rule. route() made the broadcast half.
            x_replay_mt.route_to(proj_engine.X_in)

            # o's y east out of MT[1], off at MT[3], un-permuted, one hop north
            # into rms X_in -- a merge with the shim's phase-1 rows, ordered by
            # the layer's dependencies.
            connect_port(proj_engine.Y_out, aux_mt.OProj_in, pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(aux_mt.Residual_out, rms_core.X_in, pkt_id = aux_to_rms_pkt_id.to_residual, keep_pkt_header = False)

            # ---- stages 7d/7e: up|gate and the SwiGLU. sec 5.9 --------------
            # The post-attention norm, up|gate's x. Its own connect_port even
            # though x_in(2) is a port to_proj already reaches: rms_residual.cc
            # stamps the id per phase, so this row travels under to_st.
            connect_port(rms_core.Y_out, x_replay_mt.x_in(2), pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)

            # to_glu = 16 is one-hot on purpose -- a widened mask that admits it
            # cannot admit another id in the y tree. ARCHITECTURE.md sec 6.
            connect_port(proj_engine.Y_out, glu.Up_Gate_in, pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

            # ---- stage 7f: down_proj and h2. ARCHITECTURE.md sec 5.10 -------
            # hid into MT[2]'s dedicated phase; glu.py stamps to_down_proj to
            # match. h2 leaves the rms tile's third phase for DDR.
            connect_port(glu.Hid_out, x_replay_mt.x_in(3), pkt_id = aux_to_proj_x_in_pkt_id.to_down_proj, keep_pkt_header = False)
            connect_port(rms_core.Y_out, IT[3].S2MM[1], pkt_id = rms_pkt_id.to_out, keep_pkt_header = False)

        print(mctx.module)

if __name__ == "__main__":
    model_type = "QWEN3_8_27B"
    if len(sys.argv) != 2:
        print("Usage: python layer_aie.py <model_type>, Options: QWEN3_8_27B")
        print("Use default model type: QWEN3_8_27B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)
