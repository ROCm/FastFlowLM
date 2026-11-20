# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Check a Confucius4-R2T2 / Qwen3-ASR q4nx package against the BF16 HF checkpoint.

    python check_qwen3_asr.py --hf <dir with model.safetensors> --pkg <dir with model.q4nx and audio_weight.q4nx>

language (model.q4nx): the tensor names match the HF names one to one (+ lm_head.weight), the embedding and the norms
are the HF values bit for bit, and the Q8_0 matrices (a sample of layers, every projection, and lm_head) equal an
independent quantization of the HF weights: ggml Q8_0 (fp32 scale = absmax/127, round half away from zero, fp16 scale
stored) packed with the converter's own _pack_q8nx. This pins down the weight type, the row/col block layout and
that nothing went through a lossy detour.
audio (audio_weight.q4nx): every tensor is the HF value, bf16; matrices are re-ordered by audio_mm_weight_rearrange
only, so they must equal that function applied to the HF weight.
"""
import argparse
import os
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4nx.model_converter import __Q4NX_Converter as Conv   # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def q8_0_parts(w):
    """ggml Q8_0 of w [rows, cols]: (fp16 scales [rows, cols/32], int8 quants [rows, cols])."""
    b = w.float().reshape(w.shape[0], -1, 32)
    d = b.abs().amax(-1, keepdim=True) / 127
    inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
    x = b * inv
    q = torch.sign(x) * torch.floor(x.abs() + 0.5)          # round half away from zero, as ggml's roundf
    return d.half().squeeze(-1), q.to(torch.int8).reshape(w.shape)          # the packer wants fp16 scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--pkg", required=True)
    a = ap.parse_args()

    hf = safe_open(os.path.join(a.hf, "model.safetensors"), "pt")
    keys = list(hf.keys())

    # shim with just the attributes the borrowed packers read
    shim = types.SimpleNamespace(row_block_size=32, col_block_size=256, parallel_size=32, keep_block_in_2D=True,
                                 audio_MM_K=512, audio_MM_N=64)
    shim._multi_modal_mm_weight_rearrange = types.MethodType(Conv._multi_modal_mm_weight_rearrange, shim)
    pack_q8 = types.MethodType(Conv._pack_q8nx, shim)
    audio_rearrange = types.MethodType(Conv.audio_mm_weight_rearrange, shim)

    # ---------------- language ----------------
    lm = load_file(os.path.join(a.pkg, "model.q4nx"))
    want = {}
    for k in keys:
        if k.startswith("thinker.model."):
            want["model." + k[len("thinker.model."):]] = k
    want["lm_head.weight"] = "thinker.model.embed_tokens.weight"
    check(set(lm) == set(want), f"language tensor names match ({len(lm)} tensors, expected {len(want)})")
    missing, extra = sorted(set(want) - set(lm))[:3], sorted(set(lm) - set(want))[:3]
    if missing or extra:
        print("   missing:", missing, "extra:", extra)

    emb = hf.get_tensor("thinker.model.embed_tokens.weight")
    check(torch.equal(lm["model.embed_tokens.weight"], emb), "embedding is the HF BF16 table, bit for bit")
    bad = [n for n in lm if n.endswith(("norm.weight", "layernorm.weight")) and not torch.equal(lm[n], hf.get_tensor(want[n]))]
    check(not bad, f"all {sum(1 for n in lm if n.endswith(('norm.weight', 'layernorm.weight')))} norm tensors equal HF (bad: {bad[:3]})")

    sample = []
    for layer in (0, 13, 27):
        for p in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            sample.append(f"model.layers.{layer}.{p}.weight")
    sample.append("lm_head.weight")
    for n in sample:
        w = hf.get_tensor(want[n])
        d, q = q8_0_parts(w)
        ref = pack_q8(d, None, q)
        got = lm[n]
        same = got.shape == ref.shape and got.dtype == ref.dtype and torch.equal(got, ref)
        check(same, f"{n}: Q8_0 q4nx == independent quantization of the HF weight (shape {tuple(got.shape)}, {got.dtype})")

    # ---------------- audio ----------------
    au = load_file(os.path.join(a.pkg, "audio_weight.q4nx"))
    hf_audio = {k: k for k in keys if k.startswith("thinker.audio_tower.")}
    want_au = {}
    for k in hf_audio:
        r = k[len("thinker.audio_tower."):]
        # the omni name map files the projector under model.audio.tower.*, everything else under model.audio.*
        want_au["model.audio.tower." + r if r.startswith("proj") else "model.audio." + r] = k
    check(set(au) == set(want_au), f"audio tensor names match ({len(au)} tensors, expected {len(want_au)})")
    mats = ("fc1.weight", "fc2.weight", "k_proj.weight", "out_proj.weight", "q_proj.weight", "v_proj.weight",
            "conv_out.weight", "proj1.weight", "proj2.weight")
    n_ok = n_mat = 0
    for n, k in want_au.items():
        if n not in au:
            continue
        w = hf.get_tensor(k)
        ref = audio_rearrange(w).contiguous() if n.endswith(mats) else w
        n_mat += n.endswith(mats)
        n_ok += au[n].dtype == torch.bfloat16 and au[n].shape == ref.shape and torch.equal(au[n], ref)
    check(n_ok == len(want_au), f"{n_ok}/{len(want_au)} audio tensors equal HF (bf16; {n_mat} matrices re-ordered by audio_mm_weight_rearrange)")

    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
