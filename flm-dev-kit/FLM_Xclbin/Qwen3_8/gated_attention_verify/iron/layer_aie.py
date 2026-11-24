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
from constants import L_TOKENS

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
from qkv_stage import qkv_stage_memtile
from aux_memtile import aux_memtile
from glu import glu_tile
from gate import gate_tile
from RoPE import RoPE
from attn import attn_column
from utils import dma_port, connect_port, comment_print

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
            
            # One int for the whole attention pass: the key count of its last
            # token, context_offset + L_TOKENS. Everything else the attention
            # cores need is an offset from it -- token t attends to
            # L_max - (L_TOKENS - 1) + t keys, and the chunk count is
            # ceil(L_max / L_CHUNK) for every token alike. See attn.py.
            ctx.data_ty('L_ty', (1,), np.dtype[np.int32])

            # ===============================================================
            # SCOPE: input_layernorm, the fused qkv projection, q/k norm and
            # partial RoPE. This is the first half of the full-attention layer
            # of Qwen3.8-27B (layer i, (i + 1) % 4 == 0); the other three in four
            # are gated delta net and are a separate xclbin,
            # ../../gate_delta_net_verify.
            #
            #   IT[2].MM2S[0]  L_TOKENS x MODEL_DIM hidden states -> the norm
            #   IT[2].MM2S[1]  input_layernorm.weight
            #   rms Y_out      each normed row -> MT[2], the x replay tile
            #   MT[2]          replays the whole L x D activation 16 times,
            #                  column-block major, into MT[1]'s 4 KiB FIFO
            #   MT[1] MM2S[4]  the one port that broadcasts x, circuit-switched,
            #                  to S2MM[0] of all 16 mvm cores
            #   proj Y_out     the assembled q | k | v rows -> MT[3]
            #   MT[3]          de-permutes out of row-block order and rotates
            #                  each token to | k | v | q | -> the RoPE tile
            #   IT[3].MM2S[0]  the cos/sin table and the two qk-norm weights
            #   RoPE KV_out    k, v per token -> the DDR KV cache, appended;
            #                  k on IT[4/5].S2MM[0] and v on S2MM[1], a channel
            #                  per plane so neither BD strides by MAX_L
            #   RoPE Q_out     the padded, rotated q -> the four attention CUs,
            #                  one packet each
            #   IT[4/5].MM2S   the whole context replayed out of the cache, in
            #                  chunks of 16 keys, k on ch0 and v on ch1
            #   proj Y_out     z, the output gate -> MT[3]'s parking buffer
            #   MT[3]          z de-permuted, then split in halves, one per
            #                  attention column, to the two gate tiles
            #   gate CT[3][2], CT[3][3]  attn * sigmoid(z), half a row each
            #   IT[2/3].S2MM[0] the two gated halves -> DDR, for scoring
            #
            # Downstream -- o_proj, the residual and post_attention_layernorm,
            # the MLP -- is still out of this build, and comes back as wiring
            # rather than re-layout. The mvms that feed those stages are
            # commented out in kernels/proj_main.cc and their y-tree ids are out
            # of q4_proj_engine.pkt_ids; the two have to move together.
            #
            # Column occupancy:
            #
            #   0, 1, 6, 7   the 16-core projection engine. MT[1] also carries
            #                the column's weights and the x FIFO.
            #   2            MT[2] x replay, gate CT[3][2]. CT[0][2] is *memory
            #                only* -- half the rms tile's resident hidden state.
            #   3            rms CT[0][3], RoPE CT[2][3], gate CT[3][3], and
            #                MT[3], which stages qkv *and* parks z.
            #                CT[1][3] is memory only, the other half of h.
            #   4, 5         the attention: two CUs per column (a qk core north
            #                of a kv core), the column's KV staging memtile and
            #                its shim. The CUs have to be vertically adjacent --
            #                the qk core reads its scores out of the kv core's
            #                memory module -- which is what makes this whole
            #                columns rather than four tiles placed anywhere.
            #
            # Why the rms tile sits at CT[0][3] with its west and north
            # neighbours empty: an 8-row bf16 activation is 80 KiB against a
            # 64 KiB compute tile, so the residual branch cannot be resident on
            # the tile that computes it. A core can address its west, north and
            # south neighbours' memory modules directly, so h lives on CT[0][2]
            # and CT[1][3] -- four D-wide rows each, one per 16 KiB bank -- and
            # the core loads and stores it with ordinary pointers. Those two
            # tiles get no aie.core and no aie.mem, so they reserve no stack and
            # all four banks are free. See iron/rms_residual.py.
            # ===============================================================

            rms_core = rms_residual_tile(
                CT[0][3], ctx, model_config,
                residual_tiles = [CT[0][2], CT[1][3]],   # west, then north
                stack_size = 1024 * 4,
                link_with = model_config.rms_residual_link_with)

            rope_core = RoPE(CT[2][3], ctx, model_config, stack_size = 1024 * 4,
                             link_with = model_config.rope_link_with)

            qkv_stage = qkv_stage_memtile(MT[3], ctx, model_config)

            # ---- the output gate, two tiles in one column -------------------
            # attn * sigmoid(z), split by attention column: gate tile i takes
            # column 4 + i's half-row and the matching half of z. Two tiles
            # because a compute tile has two S2MM channels and a merged gate
            # would need three -- see iron/gate.py.
            #
            # **Vertically adjacent, and that is load-bearing**, like the qk/kv
            # pairing: the emitter owns a full-row buffer and the helper writes
            # its half into it through the north window, so the row is assembled
            # with no DMA and leaves on one port.
            #
            # CT[3][2] is north of CT[2][2], so the emitter goes there and is
            # constructed first -- the helper needs its buffer.
            gate_emitter = gate_tile(CT[3][2], ctx, model_config, half = 1,
                                     stack_size = 1024,
                                     link_with = model_config.gate_link_with)
            gate_helper = gate_tile(CT[2][2], ctx, model_config, half = 0,
                                    stack_size = 1024,
                                    link_with = model_config.gate_link_with,
                                    partner = gate_emitter)
            # Indexed by attention column, so gate_cores[i].half must be i.
            gate_cores = [gate_helper, gate_emitter]

            # ---- the MLP's SwiGLU ------------------------------------------
            # Two engine rounds in, one out: round 2p is up and 2p+1 is gate,
            # because the host weaves the two weight matrices M_PER_ROUND rows
            # at a time. Both operands carry the engine's row-block permute and
            # the kernel is elementwise, so nothing is reshuffled here -- the
            # un-permute happens once, in the replay tile's landing BD.
            #
            # Only CT[1][2] and CT[3][3] are actually free -- CT[0][2] and
            # CT[1][3] look idle but are the rms core's h lenders and must stay
            # coreless. CT[1][2] is the one in the replay tile's own column, so
            # hid's hop back is short. The hidden result never materialises: at
            # L = 8 it is 278 KiB, four times this tile, so it leaves a round at
            # a time.
            glu = glu_tile(CT[1][2], ctx, model_config, stack_size = 1 * 1024,
                           link_with = model_config.glu_link_with)

            # ---- attention, columns 4 and 5 --------------------------------
            # Four CUs, one kv head each. A CU is a qk core stacked directly
            # north of its kv core, because the score buffers live in the kv
            # core's memory module and a compute core can only reach its
            # immediate neighbours -- that adjacency is what reserved whole
            # columns for this rather than letting the CUs be placed anywhere.
            #
            # Each column owns two kv heads, stages their context through its
            # own memtile in chunks of 16 keys, and assembles its two CUs'
            # output into one contiguous run before the shim writes DDR.
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
            proj_engine = q4_proj_engine(proj_cores, proj_assemble_mts, proj_shim_tiles, 10 * 1024, ["proj_main.o"], ctx, model_config)

            # ---- every x the engine reduces over ---------------------------
            # qkv and z share a phase, because they reduce over the *same* x --
            # the input norm -- so z costs a longer replay rather than an entry
            # of its own. o does not: it reduces over the gated rows, which is a
            # different activation and therefore a second phase.
            #
            # `replays` must equal the total of the M / m loop counts
            # proj_main() runs -- QKV_REPEATS and O_DOWN_REPEATS in
            # kernels/aie_array_layout.h -- and nothing cross-checks the two
            # files: get it wrong and the engine parks on the x acquire.
            #
            #   qkv   (DQ + DK + DV) / 16 / 32 = 8192 / 512 = 16 replays
            #   z      DQ            / 16 / 32 = 6144 / 512 = 12 replays
            #                                                 -- 28 total
            #   o      D             / 16 / 32 = 5120 / 512 = 10 replays
            #
            # The two phases **time-multiplex one 96 KiB buffer**, handed over
            # by the prod_lock credit chain: o's inbound BD acquires exactly the
            # credits qkv|z's replays produce, so it cannot complete until the
            # last of them has left. That is safe because o's x cannot exist
            # until the whole attention column has run, which is long after
            # qkv|z's last replay -- the dependency the design already has, made
            # into the buffer's hand-over.
            #
            # The MLP's up_gate and down come next, and down is the one phase
            # that is `dedicated` -- it fills while up_gate is still replaying.
            x_replay_mt = proj_x_replay_mt(MT[2], ctx, model_config, phases = [
                x_replay_phase(
                    name = "qkv_z",
                    x_dim = model_config.D,
                    replays = (model_config.DQ + model_config.DK + model_config.DV
                               + model_config.DQ)
                              // 16 // model_config.q4nx_row_block_size,
                ),
                x_replay_phase(
                    name = "o",
                    x_dim = model_config.DQ,
                    replays = model_config.D // 16 // model_config.q4nx_row_block_size,
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
                    # The one phase that cannot share the buffer: the SwiGLU
                    # result is produced *by* the up|gate mvms, a round at a
                    # time, so it is written while up_gate is still replaying.
                    dedicated = True,
                    # ...and the only one whose producer is row-block permuted.
                    # The GLU inherits the engine's permute because both its
                    # operands carry it and the kernel is elementwise, so the
                    # landing BD un-permutes. 32, not M_PER_ROUND: a round is
                    # [core][token][row] and the token index turns over every
                    # 32 words.
                    in_block = model_config.q4nx_row_block_size,
                ),
            ])

            # =================== wiring ======================================
            # Connection order matters: long connections route first.

            # ---- input_layernorm -------------------------------------------
            # One id for both: the shim channels are told apart by the CT-side
            # S2MM channel they land on, not by the packet id.
            connect_port(IT[2].MM2S[0], rms_core.X_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)
            connect_port(IT[2].MM2S[1], rms_core.W_in, pkt_id = it1_pkt_id.to_rms, keep_pkt_header = False)

            # ---- the qkv projection ----------------------------------------
            # The normed rows into the replay tile, one whole D-wide row per
            # packet -- so no landing un-permute, unlike the SwiGLU's hid.
            connect_port(rms_core.Y_out, x_replay_mt.x_in(0), pkt_id = rms_pkt_id.to_proj, keep_pkt_header = False)

            # route() brings the q4_k weights in on IT[0, 1, 6, 7] and makes the
            # broadcast half of the x path: MT[1]'s FIFO out to the 16 cores,
            # circuit-switched.
            proj_engine.route()

            # The replay channels' one hop west into that FIFO. This is the
            # half route() cannot make, because the engine does not own the
            # replay tile. It is also why the FIFO is on MT[1] and not MT[2]:
            # the pathfinder cannot route a packet flow whose source and
            # destination are the same switchbox.
            x_replay_mt.route_to(proj_engine.X_in)

            # ---- q | k | v out of the engine, into staging -------------------
            # The engine's Y_out is row-block permuted -- raw[(rb * L + t) * 32
            # + r] -- and the RoPE tile needs whole tokens, so MT[3] holds the
            # whole L x (DQ + DK + DV) activation and un-permutes it on the way
            # in. It also rotates each token to | k | v | q |, which is the order
            # the RoPE tile consumes: k and v go to the cache while q is still
            # being normed.
            #
            # This is where the qkv readback used to drop out to DDR (IT[1]
            # S2MM ch1). It is not tapped any more -- the qkv projection was
            # verified at cosine 0.999986 and what is scored now is what comes
            # out the far end of RoPE.
            connect_port(proj_engine.Y_out, qkv_stage.QKV_in, pkt_id = proj_pkt_id.to_rope, keep_pkt_header = False)
            connect_port(qkv_stage.QKV_out, rope_core.QKV_in)

            # ---- z, the output gate, into the same tile's parking buffer ----
            # The engine's second mvm, emitted immediately after qkv and not
            # consumed until the gate runs -- which is q/k norm, RoPE, the cache
            # append and the whole softmax column later. 96 KiB has to sit
            # somewhere across that gap, and a memtile is the only thing big
            # enough. It shares MT[3] with the qkv staging buffer: the same
            # destination for both ids is the cheap case for the y tree, since
            # no port in it needs a new direction.
            connect_port(proj_engine.Y_out, qkv_stage.Z_in, pkt_id = proj_pkt_id.to_aux, keep_pkt_header = False)

            # ---- q/k norm and partial RoPE ----------------------------------
            # The cos/sin table (one pair per token) plus q_norm.weight and
            # k_norm.weight, once per invocation. IT[3] and not IT[2]: the norm
            # column's shim already drives two flows and a third direction out of
            # one master port over-subscribes an arbiter.
            connect_port(IT[3].MM2S[0], rope_core.ROPE_in)

            # k | v alternate, one pair per token, out to the DDR cache -- but
            # to the *attention columns' own shims*, split by kv head, and
            # split again by plane: k on that shim's S2MM ch0, v on ch1.
            #
            # Not IT[3], which is where this used to go and is the natural home
            # for it. npu_dma_wait only orders transfers on one shim tile, so an
            # append on IT[3] cannot be waited on before the replay reads from
            # IT[4] and IT[5]: the replay would start against a cache that had
            # not been written yet. Routing each column's heads to that column's
            # shim puts both halves of the round trip on one IT, which is what
            # makes the wait in the host sequence mean anything.
            #
            # **Two channels, one per plane, is what lifts the context
            # ceiling.** Both used to share ch0, so the host had to catch an
            # interleaved k(t) v(t) k(t+1)... stream in one descriptor and cross
            # between the planes with a stride of the whole k plane -- a 20-bit
            # field that capped MAX_L near 2048 and masked instead of faulting
            # above it. A channel each gives the host one BD per plane, neither
            # of which has a stride that grows with MAX_L. ch1 on these shims
            # was free: it used to carry the attention-output diagnostic, which
            # went away when O_out started feeding the gate tiles directly.
            #
            # Ids continue past the q range -- see k_pkt_base / v_pkt_base in
            # iron/RoPE.py. Four now where there were two.
            for col, column in enumerate(attn_columns):
                connect_port(rope_core.KV_out, column.IT.S2MM[0],
                             pkt_id = rope_core.k_pkt_base + col)
                connect_port(rope_core.KV_out, column.IT.S2MM[1],
                             pkt_id = rope_core.v_pkt_base + col)

            # q fans out to the four CUs, one packet each -- a CU owns one kv
            # head and takes that head's padded GQA group. The id is the CU
            # index, in the order the RoPE tile emits them, so CUs 0 and 1 are
            # column 4 and CUs 2 and 3 are column 5. That order is also the
            # order the output comes back in, which is what lets the host treat
            # the two columns' readback as two halves of one row.
            #
            # q does not go to DDR any more. It was read back while the CUs did
            # not exist; they consume it directly now, and the attention output
            # is what gets scored.
            for cu_id, (column, cu_in_column) in enumerate(
                    [(c, i) for c in range(len(attn_columns)) for i in range(model_config.cus_per_column)]):
                connect_port(rope_core.Q_out, attn_columns[column].Q_in[cu_in_column], pkt_id = cu_id)

            # ---- the attention columns -------------------------------------
            # route() makes each column's internals: its shim's two MM2S into
            # the memtile, the memtile's per-head splits out to the two CUs,
            # both CUs' output back in, and the assembled half-row out to DDR
            # as the upstream diagnostic.
            for column in attn_columns:
                column.route()

            # ---- the output gate -------------------------------------------
            # Gate tile i takes attention column 4 + i's half-row and the
            # matching half of z. The two indices are the same index: column 4
            # computes q heads 0..11, which is z dims [0, DQ/2), and column 5
            # computes heads 12..23. Cross them and nothing hangs -- the shapes
            # are identical -- it just gates each half by the wrong gate, which
            # is why the mapping is spelled out in both Z_out() and here.
            # GATE_SWAP_HALVES=1 is the negative control for exactly this
            # mapping: it feeds each gate tile the *other* half of z. Nothing
            # hangs -- the shapes are identical -- so if the test still passes
            # with this set, the test is not measuring the gate at all.
            swap = os.environ.get("GATE_SWAP_HALVES", "0") == "1"
            if swap:
                comment_print("!! GATE_SWAP_HALVES: z halves crossed (negative control)")
            for i, g in enumerate(gate_cores):
                connect_port(attn_columns[i].O_out, g.Attn_In)
                connect_port(qkv_stage.Z_out(1 - i if swap else i), g.Gate_In)

            # ---- the assembled gated row, o_proj's x ------------------------
            # One producer, one flow: the emitter sends the whole DQ-wide row,
            # the helper having written its half into that buffer directly. So
            # there is nothing to merge and no arrival order to race.
            #
            # It lands on the *same* inbound port as the rms tile's rows and is
            # told apart by arrival order, which the replay tile's own credit
            # chain enforces. The id exists only because that port is already a
            # packet destination -- a switchbox will not drive one DMA channel
            # from both a circuit and a packet flow.
            connect_port(gate_emitter.Attn_Out_replay, x_replay_mt.x_in(1),
                         pkt_id = aux_to_proj_x_in_pkt_id.to_out_proj,
                         keep_pkt_header = False)

            # ---- o_proj ----------------------------------------------------
            # The third mvm's y, back into MT[3] to be un-permuted into whole
            # D-wide tokens. Same tile as qkv and z, and a destination the y
            # tree already reaches, so to_rms costs the tree a mask/match rule
            # and no new direction on any port.
            connect_port(proj_engine.Y_out, qkv_stage.O_in,
                         pkt_id = proj_pkt_id.to_rms, keep_pkt_header = False)

            # o out to DDR for scoring, on column 3's shim. Kept now that the
            # rms tile consumes o directly: it is the only thing that still
            # makes o_proj separately observable, and without it a bad o and a
            # bad residual would look alike.
            connect_port(qkv_stage.O_out, IT[3].S2MM[1])

            # ---- the attention residual and post_attention_layernorm --------
            # o's rows one hop north into the rms tile's x port -- a merge with
            # the shim's phase-1 rows, which is why it is a packet flow. The two
            # phases are told apart by arrival order, not by the id: the tile
            # does not distinguish them and does not need to, since o cannot
            # exist until the whole attention column has run.
            #
            # Neither residual operand touches DDR. h has been resident on the
            # rms tile's two coreless neighbours since phase 1.
            connect_port(qkv_stage.Residual_out, rms_core.X_in,
                         pkt_id = aux_to_rms_pkt_id.to_residual, keep_pkt_header = False)

            # ---- the MLP ----------------------------------------------------
            # The post-attention norm is up|gate's x. Its own connect_port even
            # though x_in(2) is a port to_proj already reaches: the kernel
            # stamps the id per phase, so this row travels under to_st and the
            # replay tile tells the two phases apart by arrival order.
            #
            # **This is what used to be the post-norm DDR tap.** rms_core.Y_out
            # carries at most two destination sets -- AIECreatePacketFlows
            # merges to_proj and to_st into one group because both go west to
            # MT[2], leaving one for h2 going south. The tap was a third and had
            # to go, not be joined.
            connect_port(rms_core.Y_out, x_replay_mt.x_in(2),
                         pkt_id = rms_pkt_id.to_st, keep_pkt_header = False)

            # up|gate's y, a round at a time, to the SwiGLU. to_glu = 16 is the
            # only id in the y tree with bit 4 set, so a widened mask admitting
            # it cannot admit any of the others.
            connect_port(proj_engine.Y_out, glu.Up_Gate_in,
                         pkt_id = proj_pkt_id.to_glu, keep_pkt_header = False)

            # The SwiGLU result into the replay tile's dedicated phase, still
            # row-block permuted -- the landing BD un-permutes it. Its own id:
            # the gate tile reaches that tile's east slave port under
            # to_out_proj, and one id wanting two different local DMAs off one
            # slave is what the arbiter allocator cannot express.
            connect_port(glu.Hid_out, x_replay_mt.x_in(3),
                         pkt_id = aux_to_proj_x_in_pkt_id.to_down_proj,
                         keep_pkt_header = False)

            # down's y needs no wire of its own: it rejoins o's flow above --
            # same id, same route, same buffer and BD on MT[3]. o's eight rows
            # drained before the post-attention norm existed, and that norm is
            # four dependency steps upstream of down, so the buffer is free by
            # the time down arrives. The rms tile tells phases 2 and 3 apart by
            # arrival order, exactly as it tells 1 and 2 apart.

            # h2 = h1 + down, the layer's output and the next layer's input.
            # The one phase whose y goes south rather than west, and the second
            # of this master port's two destination sets.
            connect_port(rms_core.Y_out, IT[2].S2MM[1],
                         pkt_id = rms_pkt_id.to_out, keep_pkt_header = False)

        print(mctx.module)

if __name__ == "__main__":
    model_type = "QWEN3_8_27B"
    if len(sys.argv) != 2:
        print("Usage: python layer_aie.py <model_type>, Options: QWEN3_8_27B")
        print("Use default model type: QWEN3_8_27B")
    else:
        model_type = sys.argv[1]
    flow_arch(model_type)
