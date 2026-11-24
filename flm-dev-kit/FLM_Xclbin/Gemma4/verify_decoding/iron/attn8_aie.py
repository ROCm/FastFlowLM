# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""attn8: the attention path of the 8-token verify layer as stand-alone debug stages (plan S6.5 second half, S6.9, S6.10).

  stage "rope"  : RoPE tiles only.  The shim feeds each rope tile the qkv rows of one layer (one head per [L][DH] chunk: q heads, k
                  heads, v heads) and its cos/sin/norm table, and captures q (one [L][DH] chunk per head) and the k/v rows back.
  stage "attn"  : + q park (MT4) + the global and SWA attention columns + KV replay; o rows leave on the shim.  (added in later steps)

Tile roles are the decode design's: global RoPE CT[2][3], SWA RoPE CT[3][3], global attention CT[0][2]/CT[1][2], SWA attention
CT[2][2]/CT[3][2] (IRON row/col index; AIE (col,row) = (c, r+2)), KV staging memtile MT[2], KV shim IT[2].
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
from qpark import q_park_memtile
from attn import attn_column

# packet ids of the stage feeds (any distinct ids: they only select a route)
PKT_QKV = 1
PKT_TABLE = 2


def flow_arch(model_type: str = "GEMMA4_E4B", stage: str = "rope"):
    assert L_TOKENS > 1, "attn8 stages are the 8-row design"
    dev = aie.AIEDevice.npu2
    total_cols = 8
    total_rows = 4

    model_config = get_model_config(model_type)
    print(f"// model_config: {model_config}")
    print(f"// L_TOKENS={L_TOKENS} stage={stage}")
    with mlir_mod_ctx() as mctx:
        @aie.device(dev)
        def device_body():
            IT, MT, CT = generate_tile_map(total_rows + 2, total_cols)
            ctx = mlir_context()
            ctx.data_ty('RTP_ty', (16,), np.dtype[np.int32])

            rope_core = RoPE8(CT[2][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.rope_link_with)
            swa_rope_core = SWA_RoPE8(CT[3][3], ctx, model_config, stack_size = 1024 * 4, link_with = model_config.swa_rope_link_with)

            if stage == "rope":
                # global: IT3 feeds qkv rows (ch0) and the table (ch1), q comes back on IT3.S2MM0, k/v on IT2.S2MM0
                connect_port(IT[3].MM2S[0], rope_core.QKV_in, pkt_id = PKT_QKV, keep_pkt_header = False)
                connect_port(IT[3].MM2S[1], rope_core.ROPE_in, pkt_id = PKT_TABLE, keep_pkt_header = False)
                connect_port(rope_core.Q_out, IT[3].S2MM[0])
                connect_port(rope_core.KV_out, IT[2].S2MM[0])
                # swa: IT4 feeds, q comes back on IT4.S2MM0, k/v on IT2.S2MM1
                connect_port(IT[4].MM2S[0], swa_rope_core.QKV_in, pkt_id = PKT_QKV, keep_pkt_header = False)
                connect_port(IT[4].MM2S[1], swa_rope_core.ROPE_in, pkt_id = PKT_TABLE, keep_pkt_header = False)
                connect_port(swa_rope_core.Q_out, IT[4].S2MM[0])
                connect_port(swa_rope_core.KV_out, IT[2].S2MM[1])
            elif stage == "attn":
                qpark = q_park_memtile(MT[4], ctx, model_config.num_heads, model_config.DH, model_config.SWA_DH)
                _attn_column = attn_column(
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
                # global: IT3 feeds rope (qkv rows ch0, table ch1), q -> park -> qk tile, k/v -> IT2.S2MM0 (KV append), o -> IT3.S2MM0
                connect_port(IT[3].MM2S[0], rope_core.QKV_in, pkt_id = PKT_QKV, keep_pkt_header = False)
                connect_port(IT[3].MM2S[1], rope_core.ROPE_in, pkt_id = PKT_TABLE, keep_pkt_header = False)
                connect_port(rope_core.Q_out, qpark.Q_in)
                connect_port(qpark.Q_out, _attn_column.Q_in)
                connect_port(rope_core.KV_out, IT[2].S2MM[0])
                connect_port(_attn_column.O_out, IT[3].S2MM[0], pkt_id = to_proj_x_in_pkt_id.from_attn, keep_pkt_header = False)
                # swa: IT4 feeds, q -> park -> swa qk tile, k/v -> IT2.S2MM1, o -> IT4.S2MM0
                connect_port(IT[4].MM2S[0], swa_rope_core.QKV_in, pkt_id = PKT_QKV, keep_pkt_header = False)
                connect_port(IT[4].MM2S[1], swa_rope_core.ROPE_in, pkt_id = PKT_TABLE, keep_pkt_header = False)
                connect_port(swa_rope_core.Q_out, qpark.SWA_Q_in)
                connect_port(qpark.SWA_Q_out, _attn_column.SWA_Q_in)
                connect_port(swa_rope_core.KV_out, IT[2].S2MM[1])
                connect_port(_attn_column.SWA_O_out, IT[4].S2MM[0], pkt_id = to_proj_x_in_pkt_id.from_swa_attn, keep_pkt_header = False)
                _attn_column.route()
            else:
                raise NotImplementedError(stage)

        print(mctx.module)


if __name__ == "__main__":
    flow_arch(sys.argv[1] if len(sys.argv) > 1 else "GEMMA4_E4B", stage = os.environ.get("ATTN8_STAGE", "rope"))
