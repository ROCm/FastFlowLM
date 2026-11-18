# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Fixtures for tools/test_positions_cpu.cpp: transformers' Qwen3.5 get_rope_index on the real d3-turbo prompts of the
M3 / M4 golden sets (box_image, assets, video, box_video, track), kept compact.

    source /path/to/npu/d3-turbo/work/env_turbo.sh
    python tools/gen_position_fixtures.py --goldens /path/to/npu/d3-turbo/goldens --out tools/fixtures/positions_goldens.json

Per golden row (a [state | question] prompt of the upstream composed path, or state + block of the state-once route):
  runs    visual runs in prompt order: [first index, pad id, grid_h, grid_w, expected (t, h, w) of the first token,
          expected (t, h, w) of the last token, i.e. the planner's positions inside the run]
  s       state token count (the state's pad runs all end before it; question images are after it)
  n, next row length and the position after the last token (get_rope_index's max + 1)
  text_at expected position of the token at index 0, s - 1, s, n - 1 (text tokens count up; checks the resume points)
mrope_video_agent.json (the FLM video agent's fixtures, full positions of 5 synthetic prompts) is copied next to it.
"""
import argparse
import json
import types
from pathlib import Path

import torch

IMAGE_PAD, VIDEO_PAD = 248056, 248057


def rope(ids, image_grid, video_grid):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cfg = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=2))
    fake = types.SimpleNamespace(config=cfg)
    fake.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, fake)
    t = torch.as_tensor(ids)[None]
    types_ = torch.zeros_like(t)
    types_[t == IMAGE_PAD] = 1
    types_[t == VIDEO_PAD] = 2
    ig = torch.tensor(image_grid) if image_grid else None
    vg = torch.tensor(video_grid) if video_grid else None
    pos, _ = Qwen3_5Model.get_rope_index(fake, t, types_, ig, vg.clone() if vg is not None else None)
    return pos[:, 0]   # [3][n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    G = Path(args.goldens)
    out = []
    for name in ("box_image", "assets", "video", "box_video", "track"):
        rows = [json.loads(l) for l in open(G / name / "rows.jsonl")]
        toks = [json.loads(l) for l in open(G / name / "tokens.jsonl")]
        vis = {}
        for l in open(G / name / "vision.jsonl"):
            v = json.loads(l)
            vis[v["key"]] = v
        seen = set()
        for r, tk in zip(rows, toks):
            ids = tk["ids"]
            ig = r.get("image_grid_thw")
            vg = r.get("video_grid_thw")
            if ig is None and vg is None:   # state-once route rows: grids from the vision calls
                ig = [g for k in r["vision"] if vis[k]["kind"] == "image" for g in vis[k]["grid_thw"]]
                vg = [g for k in r["vision"] if vis[k]["kind"] == "video" for g in vis[k]["grid_thw"]]
            sig = (name, tuple(ids[:r["state"]]), tuple(ids[r["state"]:]))
            if sig in seen:
                continue
            seen.add(sig)
            pos = rope(ids, ig or None, vg or None)
            # visual runs in prompt order; video grids split into t groups
            groups_img = [(g[1], g[2]) for g in (ig or [])]
            groups_vid = [(g[1], g[2]) for g in (vg or []) for _ in range(g[0])]
            runs, i, ii, vi = [], 0, 0, 0
            while i < len(ids):
                if ids[i] in (IMAGE_PAD, VIDEO_PAD):
                    gh, gw = groups_img[ii] if ids[i] == IMAGE_PAD else groups_vid[vi]
                    ii += ids[i] == IMAGE_PAD
                    vi += ids[i] == VIDEO_PAD
                    n = gh * gw // 4
                    assert all(x == ids[i] for x in ids[i:i + n]) and (i + n == len(ids) or ids[i + n] != ids[i]), (name, r["id"], i)
                    runs.append([i, ids[i], gh, gw, pos[:, i].tolist(), pos[:, i + n - 1].tolist()])
                    i += n
                else:
                    i += 1
            assert ii == len(groups_img) and vi == len(groups_vid), (name, r["id"])
            s, n = r["state"], len(ids)
            out.append({"set": name, "id": r["id"], "key": r["key"], "moment": r["moment"], "s": s, "n": n, "next": int(pos.max()) + 1,
                        "runs": runs, "text_at": [pos[0, j].item() for j in (0, s - 1, s, n - 1)],
                        # compact token ids: the text (non-visual) tokens are not needed; the planner only sees pad runs, so
                        # a row is [run lengths and the text gaps between them]
                        "gaps": [b - (a[0] + a[2] * a[3] // 4) for a, b in zip([[0, 0, 0, 0]] + runs, [x[0] for x in runs] + [n])]})
    json.dump({"rows": out, "image_id": IMAGE_PAD, "video_id": VIDEO_PAD}, open(args.out, "w"), separators=(",", ":"))
    print(len(out), "rows")


if __name__ == "__main__":
    main()
