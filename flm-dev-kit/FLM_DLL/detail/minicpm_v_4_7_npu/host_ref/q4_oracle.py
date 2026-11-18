# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""S2's fp32 reference, re-run on the weights the engine actually holds.

    source /path/to/FastFlowLM_IRON/FLM_Q4NX_Converter/venv/bin/activate
    python FLM_DLL/detail/minicpm_v_4_7_npu/host_ref/q4_oracle.py

S2's golden dumps (models_simple/dumps/minicpm_v/) come from the HuggingFace bf16 checkpoint.
The engine runs the Q4_K / Q8_0 weights in model.q4nx, so even a perfect engine cannot match
those dumps exactly. This script separates the two error sources:

  * it loads SimpleMiniCPMVText (S2's reference, unchanged) and overwrites every weight with
    S1's dequant_q4nx() of the tensor the engine loads, undoing the converter's reshapes and
    the GGUF's folded `1 + w` norm weights;
  * it runs the same prompt (S2's input_ids.npy) and writes the manifest's dumps to
    models_simple/dumps/minicpm_v_q4/;
  * it prints the per-layer cosine of those dumps against S2's bf16 dumps, which is the
    quantization floor: the best per-layer cosine any engine reading model.q4nx can reach.

The engine is then scored against both (compare_engine.py). Against this oracle, the only
remaining differences are the engine's own (bf16 activations, host kernel numerics).
"""
import argparse
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models_simple"))
sys.path.insert(0, os.path.join(REPO, "FLM_Q4NX_Converter"))

from transformers import AutoConfig  # noqa: E402
from simple.minicpm_v_simple import SimpleMiniCPMVText  # noqa: E402
from verify_minicpm_v_q4nx import load_q4nx_tensor  # noqa: E402

HF_PATH = "/path/to/models/MiniCPM-V-4.7-1B-hf"
Q4NX = "/path/to/models/MiniCPM-V-4.7-1B-NPU2/model.q4nx"
REF_DIR = os.path.join(REPO, "models_simple", "dumps", "minicpm_v")
OUT_DIR = os.path.join(REPO, "models_simple", "dumps", "minicpm_v_q4")


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4nx", default=Q4NX)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(HF_PATH, trust_remote_code=True).text_config
    model = SimpleMiniCPMVText(cfg).eval()
    f = safe_open(args.q4nx, "pt")

    def w(name):
        return torch.from_numpy(np.ascontiguousarray(load_q4nx_tensor(f, name)[1]))

    def set_(param, value):
        assert param.shape == value.shape, (param.shape, value.shape)
        param.data = value.to(torch.float32).clone()

    DH, H = cfg.head_dim, cfg.num_attention_heads
    with torch.no_grad():
        set_(model.embed_tokens.weight, w("model.embed_tokens.weight"))
        # GGUF folds Qwen3.5's zero-centred norms to 1 + w; SimpleRMSNorm adds the 1 itself.
        set_(model.norm.weight, w("model.norm.weight") - 1)
        set_(model.lm_head.weight, w("lm_head.weight")[: cfg.vocab_size])
        for i, layer in enumerate(model.layers):
            p = f"model.layers.{i}."
            set_(layer.input_layernorm.weight, w(p + "input_layernorm.weight") - 1)
            set_(layer.post_attention_layernorm.weight, w(p + "post_attention_layernorm.weight") - 1)
            set_(layer.mlp.up_proj.weight, w(p + "mlp.up_proj.weight"))
            set_(layer.mlp.gate_proj.weight, w(p + "mlp.gate_proj.weight"))
            set_(layer.mlp.down_proj.weight, w(p + "mlp.down_proj.weight"))
            if layer.layer_type == "full_attention":
                a = layer.self_attn
                # q4nx holds [q heads | gate heads]; the reference wants [q_h | gate_h] per head.
                qg = w(p + "self_attn.q_proj.weight").view(2, H, DH, -1).transpose(0, 1).reshape(2 * H * DH, -1)
                set_(a.q_proj.weight, qg)
                set_(a.k_proj.weight, w(p + "self_attn.k_proj.weight"))
                set_(a.v_proj.weight, w(p + "self_attn.v_proj.weight"))
                set_(a.o_proj.weight, w(p + "self_attn.o_proj.weight"))
                set_(a.query_norm.weight, w(p + "self_attn.q_norm.weight") - 1)
                set_(a.key_norm.weight, w(p + "self_attn.k_norm.weight") - 1)
            else:
                g = layer.linear_attn
                set_(g.in_proj_qkv.weight, w(p + "linear_attn.qkv_proj.weight"))
                set_(g.in_proj_z.weight, w(p + "self_attn.gate_proj.weight"))
                # the engine's host path reads the bf16 twins, not the Q8_0 copies
                set_(g.in_proj_a.weight, w(p + "linear_attn.ssm_alpha_proj.bf16.weight"))
                set_(g.in_proj_b.weight, w(p + "linear_attn.ssm_beta_proj.bf16.weight"))
                set_(g.conv1d.weight, w(p + "linear_attn.ssm_conv1d.weight").T.unsqueeze(1))
                set_(g.dt_bias, w(p + "linear_attn.ssm_dt.bias"))
                set_(g.A_log, torch.log(-w(p + "linear_attn.ssm_a")))  # GGUF holds -exp(A_log)
                set_(g.norm.weight, w(p + "linear_attn.ssm_norm.weight"))  # gated norm: no +1
                set_(g.out_proj.weight, w(p + "linear_attn.ssm_out_proj.weight"))

    ids = np.load(os.path.join(REF_DIR, "input_ids.npy"))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        logits, dumps = model(torch.from_numpy(ids.astype(np.int64)).view(1, -1))

    def to_np(t):
        t = t.detach().to(torch.float32)
        if t.dim() >= 2 and t.shape[0] == 1:
            t = t[0]
        return np.ascontiguousarray(t.numpy())

    files = {"embed_tokens.npy": dumps["embed_tokens"], "final_norm.npy": dumps["final_norm"],
             "logits.npy": logits}
    for i, ld in enumerate(dumps["layers"]):
        for k, v in ld.items():
            files[f"layers/{i:02d}/{k}.npy"] = v
    for rel, t in files.items():
        path = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, to_np(t))
    np.save(os.path.join(args.out, "input_ids.npy"), ids)

    print("quantization floor: q4nx-weight reference vs S2's bf16 reference")
    worst = 1.0
    for i in range(cfg.num_hidden_layers):
        c = cosine(np.load(os.path.join(args.out, f"layers/{i:02d}/residual_out.npy")),
                   np.load(os.path.join(REF_DIR, f"layers/{i:02d}/residual_out.npy")))
        worst = min(worst, c)
        print(f"  layers/{i:02d}/residual_out  cos {c:.6f}")
    for name in ("final_norm", "logits"):
        c = cosine(np.load(os.path.join(args.out, f"{name}.npy")), np.load(os.path.join(REF_DIR, f"{name}.npy")))
        print(f"  {name:<24} cos {c:.6f}")
    q4_last = np.load(os.path.join(args.out, "logits.npy"))[-1]
    ref_last = np.load(os.path.join(REF_DIR, "logits.npy"))[-1]
    print(f"  argmax logits[-1]: q4 {int(q4_last.argmax())}  bf16 {int(ref_last.argmax())}")
    print(f"  worst residual_out cosine {worst:.6f}")
    print(f"dumps: {args.out}")


if __name__ == "__main__":
    main()
