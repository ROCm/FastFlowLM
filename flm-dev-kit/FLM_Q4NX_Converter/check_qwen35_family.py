# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""HF-referenced check of a Qwen3.5-family model.q4nx (Qwen3.5 0.8B/2B/4B/9B/27B, Qwen3.6, Qwen3.8).

The only inputs are the original HF checkpoint (config.json + safetensors) and the converted model.q4nx; no
GGUF is consulted, so the check is independent of how llama.cpp (or any other exporter) lays the tensors out.
Everything the engine reads is compared against the HF tensor in HF order, after decoding the q4nx blocks:

  * layer pattern / head geometry come from the HF text_config (`layer_types`, linear_num_{key,value}_heads, ...),
    so irregular layer layouts, ratio 1 (0.8B/2B), 2 (4B/9B) and 3 (27B / 3.8) are all handled by one code path;
  * delta-net v heads must be in HF (grouped) order: the un-permuted GGUF (tiled) order is off by ~1.4 relative L2;
  * full-attention q_proj rows are [all q heads | all gate heads];
  * norm weights follow the llama.cpp convention (RMSNorm weight + 1) except the gated delta-net norm;
  * alpha/beta rows are padded (zeros) or duplicated up to a multiple of 32, never anything else;
  * `.bf16` twins, when present, are bit-exact copies of in_proj_a / in_proj_b;
  * no stray `model.layers.<n>` with n >= num_hidden_layers (an MTP block must not leak in as a decoder layer);
  * the MTP block, when kept, lives under `model.mtp.*`.

    python check_qwen35_family.py --hf <hf dir> --q4nx model.q4nx [--layers 0 1 2 3 | all] [--mtp expect|absent|any]

Exit status 0 only if every check passes.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_minicpm_v_q4nx import dequant_q4nx, q4nx_format  # noqa: E402

FAILS = []
QUANT_TOL = {"q4k": 0.25, "q41": 0.25, "q40": 0.30, "q80": 0.03}   # relative L2 vs the HF original


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


class HF:
    """HF checkpoint reader that hides the `model.language_model.` / `language_model.` prefix variants."""

    def __init__(self, path):
        self.path = path
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg = cfg.get("text_config", cfg)
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            wm = json.load(open(index))["weight_map"]
        else:
            with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
                wm = {k: "model.safetensors" for k in f.keys()}
        self.wm = wm
        self.handles = {}

    def _resolve(self, name):
        for pre in ("model.language_model.", "language_model.model.", "language_model.", "model.", ""):
            if pre + name in self.wm:
                return pre + name
        return None

    def has(self, name):
        return self._resolve(name) is not None

    def get(self, name):
        full = self._resolve(name)
        fn = self.wm[full]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.path, fn), "pt")
        return self.handles[fn].get_tensor(full)

    def mtp(self, name):
        full = "mtp." + name
        if full not in self.wm:
            return None
        fn = self.wm[full]
        if fn not in self.handles:
            self.handles[fn] = safe_open(os.path.join(self.path, fn), "pt")
        return self.handles[fn].get_tensor(full)


class Q4NX:
    def __init__(self, path):
        self.f = safe_open(path, "pt")
        self.names = set(self.f.keys())

    def has(self, name):
        return name in self.names

    def raw(self, name):
        return self.f.get_tensor(name)

    def quant(self, name):
        """(format, decoded float32 array)"""
        t = self.f.get_tensor(name)
        assert t.dtype == torch.int8, f"{name} is {t.dtype}, expected a packed int8 tensor"
        return q4nx_format(t.numpy()), dequant_q4nx(t.numpy())


def check_quant(q, name, ref, label=None, rows=None, tol_scale=1.0):
    """Decoded q4nx tensor vs HF tensor (HF order).  Columns/rows beyond the HF shape must be padding."""
    label = label or name
    if not q.has(name):
        check(False, f"{label}: missing {name}")
        return None
    fmt, d = q.quant(name)
    ref = ref.float().numpy()
    r, c = ref.shape
    got = d[:r, :c]
    err = rel(got, ref)
    tol = QUANT_TOL[fmt] * tol_scale
    check(err < tol, f"{label}: {fmt} rel-L2 vs HF {err:.3f} (tol {tol:.2f}), shape {tuple(d.shape)}")
    extra_cols = d[:, c:]
    if extra_cols.size:
        check(not extra_cols.any(), f"{label}: padding columns beyond {c} are zero")
    return d


def check_exact(q, name, ref, label=None, transpose=False, plus_one=False, bf16=False):
    label = label or name
    if not q.has(name):
        check(False, f"{label}: missing {name}")
        return
    got = q.raw(name).float()
    want = ref.float()
    if plus_one:
        want = want + 1.0
    if transpose:
        want = want.T
    loose = plus_one or bf16   # bf16-stored norms: rounding error up to 2^-8 relative
    ok = got.shape == want.shape and torch.allclose(got, want, rtol=1e-2 if loose else 1e-6, atol=1e-2 if plus_one else 0)
    detail = "" if ok else f" got {tuple(got.shape)} vs {tuple(want.shape)}, max|d|={float((got - want).abs().max()) if got.shape == want.shape else float('nan'):.3g}"
    check(ok, f"{label}: {'HF+1' if plus_one else 'HF'}{' (transposed)' if transpose else ''}{detail}")


def check_moe_mlp(q, hf, hp, qp, layer_prefix, c):
    """Routed experts are stacked along the rows (expert-major); router is transposed; shared expert is dense."""
    n_exp = c["num_experts"]
    fused_gu = hf.has(hp + "experts.gate_up_proj")
    if fused_gu:      # fused (E, 2I, H) / (E, H, I) checkpoint layout
        gu, dn = hf.get(hp + "experts.gate_up_proj"), hf.get(hp + "experts.down_proj")
        inter = gu.shape[1] // 2
        gate, up, down = gu[:, :inter], gu[:, inter:], dn
    else:             # per-expert checkpoint layout
        gate = torch.stack([hf.get(hp + f"experts.{e}.gate_proj.weight") for e in range(n_exp)])
        up = torch.stack([hf.get(hp + f"experts.{e}.up_proj.weight") for e in range(n_exp)])
        down = torch.stack([hf.get(hp + f"experts.{e}.down_proj.weight") for e in range(n_exp)])
    for nm, w in (("gate_exps_proj", gate), ("up_exps_proj", up), ("down_exps_proj", down)):
        check_quant(q, qp + f"{nm}.weight", w.reshape(-1, w.shape[-1]), f"routed experts {nm}")
    for nm, src in (("share_gate_exps_proj", "gate_proj"), ("share_up_exps_proj", "up_proj"), ("share_down_exps_proj", "down_proj")):
        check_quant(q, qp + f"{nm}.weight", hf.get(hp + f"shared_expert.{src}.weight"), f"shared expert {nm}")
    router = hf.get(hp + "gate.weight")
    check_exact(q, layer_prefix + "moe_router.weight", router, "moe_router (transposed)", transpose=True, bf16=True)
    check_exact(q, layer_prefix + "shared_expert_gate.weight", hf.get(hp + "shared_expert_gate.weight").reshape(-1), "shared_expert_gate", bf16=True)


def check_rows_padded(d, nrows, label):
    """alpha / beta: rows beyond the HF head count must be zero padding or a duplicate of the real rows."""
    if d is None:
        return
    extra = d[nrows:]
    ok = d.shape[0] % 32 == 0 and d.shape[0] >= nrows
    if extra.size:
        zero = not extra.any()
        dup = extra.shape[0] % nrows == 0 and all(np.array_equal(extra[i:i + nrows], d[:nrows]) for i in range(0, extra.shape[0], nrows))
        ok = ok and (zero or dup)
        kind = "zero pad" if zero else ("duplicate" if dup else "NEITHER zero nor duplicate")
    else:
        kind = "no extra rows"
    check(ok, f"{label}: {d.shape[0]} rows for {nrows} heads, {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--q4nx", required=True)
    ap.add_argument("--layers", nargs="+", default=["sample"], help="layer ids, 'all' or 'sample' (first 4, last 4)")
    ap.add_argument("--mtp", choices=["expect", "absent", "any"], default="any",
                    help="whether the q4nx must contain the MTP block under model.mtp.* (default: either way)")
    ap.add_argument("--norm-plus-one", choices=["auto", "yes", "no"], default="auto",
                    help="whether q4nx RMSNorm weights are HF+1 (llama.cpp) or HF; auto = whichever layer 0 matches")
    a = ap.parse_args()

    hf, q = HF(a.hf), Q4NX(a.q4nx)
    c = hf.cfg
    n_layers = c["num_hidden_layers"]
    types = c.get("layer_types") or [
        "full_attention" if (i + 1) % c["full_attention_interval"] == 0 else "linear_attention" for i in range(n_layers)]
    assert len(types) == n_layers
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    is_moe = c.get("num_experts", 0) > 0
    LP = "model.layer." if any(n.startswith("model.layer.") for n in q.names) else "model.layers."   # MoE configs say "layer"
    print(f"{n_layers} layers ({types.count('full_attention')} full), delta-net {nk} qk heads / {nv} v heads "
          f"(ratio {nv // nk}), q4nx tensors {len(q.names)}" + (f", MoE {c['num_experts']} experts" if is_moe else ""))

    if a.layers == ["all"]:
        layers = list(range(n_layers))
    elif a.layers == ["sample"]:
        layers = sorted(set(range(min(4, n_layers))) | set(range(max(0, n_layers - 4), n_layers)))
    else:
        layers = [int(x) for x in a.layers]
    # a sample must contain both kinds when the model has both
    for kind in ("linear_attention", "full_attention"):
        if kind in types and not any(types[i] == kind for i in layers):
            layers.append(types.index(kind))

    # ---- norm convention
    ref_norm = hf.get("layers.0.input_layernorm.weight").float()
    got_norm = q.raw(LP + "0.input_layernorm.weight").float()
    plus_one = {"yes": True, "no": False}.get(a.norm_plus_one)
    if plus_one is None:
        plus_one = bool(torch.allclose(got_norm, ref_norm + 1, rtol=1e-2, atol=1e-2))
    print(f"norm convention: q4nx = HF{' + 1' if plus_one else ''}")

    # ---- global tensors
    tied = c.get("tie_word_embeddings", False) or not hf.has("lm_head.weight")
    embed = hf.get("embed_tokens.weight")
    if tied:
        check_quant(q, "lm_head.weight", embed, "lm_head (tied to embed_tokens)")
        check("model.embed_tokens.weight" not in q.names, "tied: no duplicate bf16 model.embed_tokens.weight")
    else:
        check_quant(q, "lm_head.weight", hf.get("lm_head.weight"), "lm_head")
        check(q.has("model.embed_tokens.weight"), "untied: bf16 model.embed_tokens.weight present")
        if q.has("model.embed_tokens.weight"):
            if q.raw("model.embed_tokens.weight").dtype == torch.int8:      # packed Q8_0 (Qwen3.8 config)
                check_quant(q, "model.embed_tokens.weight", embed, "embed_tokens (packed)")
            else:
                e = rel(q.raw("model.embed_tokens.weight").float().numpy(), embed.float().numpy())
                check(e < 0.03, f"embed_tokens (bf16, from the GGUF Q8_0/BF16 embedding) rel-L2 vs HF {e:.4f}")
    check_exact(q, "model.norm.weight", hf.get("norm.weight"), "final norm", plus_one=plus_one)

    # ---- per-layer
    for L in layers:
        kind = types[L]
        p, qp = f"layers.{L}.", f"{LP}{L}."
        print(f"--- layer {L} ({kind})")
        check_exact(q, qp + "input_layernorm.weight", hf.get(p + "input_layernorm.weight"), "input_layernorm", plus_one=plus_one)
        check_exact(q, qp + "post_attention_layernorm.weight", hf.get(p + "post_attention_layernorm.weight"),
                    "post_attention_layernorm", plus_one=plus_one)
        if is_moe:
            check_moe_mlp(q, hf, p + "mlp.", qp + "mlp.", qp, c)
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                check_quant(q, qp + f"mlp.{proj}.weight", hf.get(p + f"mlp.{proj}.weight"), f"mlp.{proj}")

        if kind == "linear_attention":
            la, qa = p + "linear_attn.", qp + "linear_attn."
            check_quant(q, qa + "qkv_proj.weight", hf.get(la + "in_proj_qkv.weight"), "qkv_proj")
            check_quant(q, qp + "self_attn.gate_proj.weight", hf.get(la + "in_proj_z.weight"), "gate_proj (in_proj_z)")
            check_quant(q, qa + "ssm_out_proj.weight", hf.get(la + "out_proj.weight"), "ssm_out_proj")
            for nm, src in (("ssm_alpha_proj", "in_proj_a"), ("ssm_beta_proj", "in_proj_b")):
                ref = hf.get(la + src + ".weight")
                d = check_quant(q, qa + nm + ".weight", ref, nm)
                if d is not None:
                    check_rows_padded(d, ref.shape[0], nm)
                if q.has(qa + nm + ".bf16.weight"):
                    tw = q.raw(qa + nm + ".bf16.weight")
                    if tw.shape == ref.shape and torch.equal(tw, ref):
                        check(True, f"{nm}.bf16 bit-exact with {src}")
                    else:   # derived from the dequantized Q8_0 copy (MoE converter): equal up to Q8 rounding, HF head order
                        e = rel(tw.float().numpy(), ref.float().numpy()) if tw.shape == ref.shape else float("inf")
                        check(e < 0.03, f"{nm}.bf16 equals {src} up to Q8_0 rounding (rel-L2 {e:.4f}), shape {tuple(tw.shape)}")
            conv = hf.get(la + "conv1d.weight").float().reshape(-1, c["linear_conv_kernel_dim"])
            check_exact(q, qa + "ssm_conv1d.weight", conv, "ssm_conv1d", transpose=True)
            check_exact(q, qa + "ssm_a", -torch.exp(hf.get(la + "A_log").float()), "ssm_a (-exp(A_log))")
            check_exact(q, qa + "ssm_dt.bias", hf.get(la + "dt_bias"), "ssm_dt.bias")
            check_exact(q, qa + "ssm_norm.weight", hf.get(la + "norm.weight"), "ssm_norm (gated norm, plain weight)", bf16=True)
        else:
            sa, qa = p + "self_attn.", qp + "self_attn."
            nh, dh = c["num_attention_heads"], c["head_dim"]
            hq = hf.get(sa + "q_proj.weight")
            want = hq.float().reshape(nh, 2, dh, -1).permute(1, 0, 2, 3).reshape(2 * nh * dh, -1)
            d = check_quant(q, qa + "q_proj.weight", want, "q_proj [q heads | gate heads]")
            if d is not None:
                wrong = rel(d[:want.shape[0], :want.shape[1]], hq.float().numpy())
                check(wrong > 0.5, f"q_proj is not left in the interleaved HF order (rel vs interleaved {wrong:.2f})")
            for proj in ("k_proj", "v_proj", "o_proj"):
                check_quant(q, qa + f"{proj}.weight", hf.get(sa + f"{proj}.weight"), proj)
            check_exact(q, qa + "q_norm.weight", hf.get(sa + "q_norm.weight"), "q_norm", plus_one=plus_one)
            check_exact(q, qa + "k_norm.weight", hf.get(sa + "k_norm.weight"), "k_norm", plus_one=plus_one)

    # ---- structure: no leaked decoder layers, MTP placement
    ids = {int(m.group(1)) for n in q.names if (m := re.match(r"model\.layers?\.(\d+)\.", n))}
    check(ids == set(range(n_layers)), f"decoder layer ids are exactly 0..{n_layers - 1}"
          + ("" if ids == set(range(n_layers)) else f" (extra {sorted(ids - set(range(n_layers)))}, missing {sorted(set(range(n_layers)) - ids)})"))
    mtp = sorted(n for n in q.names if n.startswith("model.mtp."))
    if a.mtp == "expect":
        check(bool(mtp), f"MTP block present under model.mtp.* ({len(mtp)} tensors)")
    elif a.mtp == "absent":
        check(not mtp, "no model.mtp.* tensors")
    else:
        print(f"INFO MTP tensors in q4nx: {len(mtp)}")
    if mtp:
        want = {"model.mtp.fc.weight", "model.mtp.norm.weight", "model.mtp.pre_fc_norm_embedding.weight",
                "model.mtp.pre_fc_norm_hidden.weight"}
        check(want <= set(mtp), f"MTP head tensors present ({sorted(want - set(mtp))} missing)")
        mq = [n for n in mtp if n.startswith("model.mtp.layers.0.")]
        check(bool(mq), f"MTP decoder block under model.mtp.layers.0.* ({len(mq)} tensors)")
        if hf.mtp("fc.weight") is not None:
            check_quant(q, "model.mtp.fc.weight", hf.mtp("fc.weight"), "mtp.fc")
            check_quant(q, "model.mtp.layers.0.self_attn.q_proj.weight",
                        hf.mtp("layers.0.self_attn.q_proj.weight").float().reshape(c["num_attention_heads"], 2, c["head_dim"], -1)
                        .permute(1, 0, 2, 3).reshape(-1, c["hidden_size"]), "mtp q_proj [q | gate]")

    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
