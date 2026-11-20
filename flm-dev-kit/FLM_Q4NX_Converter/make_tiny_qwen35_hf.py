# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Random-weight, small HF checkpoint with Qwen3.5-family geometry, for converter tests.

Real Qwen3.5-27B / Qwen3.6-27B / Qwen3.8-27B are tens of GB, but what the converters get wrong is the
*geometry*: delta-net value/key head ratio (1, 2 or 3), untied lm_head, an MTP block, the layer pattern.
This writes a checkpoint with that geometry in a few hundred MB, in the same tensor naming as the
released checkpoints (`model.language_model.*`, `mtp.*`), so it goes through llama.cpp's own
convert_hf_to_gguf.py -> llama-quantize -> convert.py -> check_qwen35_family.py unchanged.

    python make_tiny_qwen35_hf.py --out tiny-r3 --tokenizer-from <real Qwen3.5 HF dir> --ratio 3 --untied --mtp
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-from", required=True, help="a real Qwen3.5 HF dir (tokenizer + chat template are copied)")
    ap.add_argument("--config-from", default="", help="HF dir whose config.json is the base (default: --tokenizer-from); "
                    "use the Qwen3.6-35B-A3B config with --moe")
    ap.add_argument("--moe", action="store_true", help="Qwen3.5/3.6-MoE (A3B) structure: routed experts + shared expert")
    ap.add_argument("--ratio", type=int, default=3, help="delta-net v heads per qk group")
    ap.add_argument("--qk-heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--layer-types", default="", help="comma list of l/f, overrides the every-4th-is-full pattern")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--mtp", action="store_true", help="add an MTP block (mtp.*) like the released checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    base = json.load(open(os.path.join(a.config_from or a.tokenizer_from, "config.json")))
    t, v = base["text_config"], base["vision_config"]
    if a.layer_types:
        types = ["full_attention" if c == "f" else "linear_attention" for c in a.layer_types.split(",")]
    else:
        types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(a.layers)]
    t.update(
        hidden_size=a.hidden, intermediate_size=2 * a.hidden, num_hidden_layers=len(types), layer_types=types,
        linear_num_key_heads=a.qk_heads, linear_num_value_heads=a.qk_heads * a.ratio,
        linear_key_head_dim=128, linear_value_head_dim=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=256, tie_word_embeddings=not a.untied, mtp_num_hidden_layers=1 if a.mtp else 0)
    if a.moe:
        t.update(num_experts=16, num_experts_per_tok=4, moe_intermediate_size=256, shared_expert_intermediate_size=256)
        t.pop("intermediate_size", None)
    v.update(depth=2, hidden_size=128, num_heads=4, intermediate_size=256, out_hidden_size=a.hidden)
    base["tie_word_embeddings"] = not a.untied
    if a.moe:
        model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(**base)).to(torch.bfloat16)
    else:
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**base)).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p.float()) * 15 + 1).to(p.dtype))
            elif name.endswith("dt_bias"):
                p.copy_(torch.rand_like(p.float()).to(p.dtype))
            elif "norm" in name and name.endswith("weight") and p.dim() == 1:
                p.copy_((torch.randn_like(p.float()) * 0.1).to(p.dtype))
            else:
                p.copy_((torch.randn_like(p.float()) * 0.05).to(p.dtype))
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="100GB")

    path = os.path.join(a.out, "model.safetensors")
    from safetensors.torch import load_file
    sd = load_file(path)
    if a.untied and not any(k.endswith("lm_head.weight") for k in sd):
        raise SystemExit("untied requested but the saved checkpoint has no lm_head.weight")

    if a.mtp:
        # MTP block as in the released checkpoints: one full-attention decoder layer + fc + three norms
        first_full = types.index("full_attention")
        pre = f"model.language_model.layers.{first_full}."
        for k in [k for k in sd if k.startswith(pre)]:
            if "linear_attn" not in k:
                sd["mtp.layers.0." + k[len(pre):]] = sd[k].clone().normal_(0, 0.05)
        h = a.hidden
        sd["mtp.fc.weight"] = (torch.randn(h, 2 * h) * 0.05).to(torch.bfloat16)
        for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            sd["mtp." + n + ".weight"] = (torch.randn(h) * 0.1).to(torch.bfloat16)
        os.remove(path)
        save_file(sd, path, metadata={"format": "pt"})

    for fn in os.listdir(a.tokenizer_from):
        if fn.endswith((".json", ".jinja", ".txt")) and not fn.startswith("model.safetensors") and fn != "config.json":
            shutil.copy(os.path.join(a.tokenizer_from, fn), os.path.join(a.out, fn))
    print(f"wrote {a.out}: {len(types)} layers {''.join('f' if x == 'full_attention' else 'l' for x in types)}, "
          f"ratio {a.ratio}, {'untied' if a.untied else 'tied'}, mtp={a.mtp}, {sum(x.numel() for x in sd.values()) / 1e6:.0f}M params")


if __name__ == "__main__":
    main()
