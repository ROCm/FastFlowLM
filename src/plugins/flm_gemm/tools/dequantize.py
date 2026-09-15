"""Write a dequantized weight sidecar next to model.q4nx.

Usage:
  python3 dequantize.py <model_dir> [--mode bf16] [--only SUBSTR] [--out NAME]

Selects the projection tensors whose names contain SUBSTR (default: up_proj),
dequantizes each on the CPU, and writes them in the dequant app's buffer layout.
Layer 0's up projection is checked against a captured dequant output first, so a
regression in the dequantizer stops the run before anything is written.
"""

import argparse
import json
import os
import sys

import numpy as np

import sidecar

# Which logical dimension is the contraction (the tensor's in_features). The
# dequant layout is indexed by it, so it cannot be inferred from the block count
# alone.
IN_FEATURES = {
    "self_attn.q_proj": "D",
    "self_attn.k_proj": "D",
    "self_attn.v_proj": "D",
    "self_attn.o_proj": "DQ",
    "mlp.gate_proj": "D",
    "mlp.up_proj": "D",
    "mlp.down_proj": "I",
}

DUMP = os.environ.get("FLM_DUMP", "dump")
CHECK_TENSOR = "model.layers.0.mlp.up_proj.weight"
CHECK_COLS = 1536
# Captured from the running model: the dequant app's output buffer, and the
# packed copy the validated host repack produced from it.
CHECK_DUMP = {
    "bf16": os.path.join(DUMP, "B_dequant.bin"),
    "bfp": os.path.join(DUMP, "B_packed.bin"),
}


def layer_dims(cfg, layer_idx):
    d = cfg["hidden_size"]
    non_skip = cfg["num_hidden_layers"] - cfg["num_kv_shared_layers"]
    inter = cfg["intermediate_size"]
    if layer_idx >= non_skip and cfg.get("use_double_wide_mlp"):
        inter *= 2
    head = (
        cfg["global_head_dim"]
        if cfg["layer_types"][layer_idx] == "full_attention"
        else cfg["head_dim"]
    )
    return {"D": d, "I": inter, "DQ": head * cfg["num_attention_heads"]}


def self_check(model_dir, header, data_start, mode):
    reference = CHECK_DUMP[mode]
    if not os.path.exists(reference):
        print(f"  SKIPPED: no captured reference at {reference}")
        return True
    raw, _ = sidecar.read_tensor(
        os.path.join(model_dir, "model.q4nx"), header, data_start, CHECK_TENSOR
    )
    got, _ = sidecar.build(raw, CHECK_COLS, mode)
    want = np.fromfile(reference, dtype=np.uint8)
    if got.size != want.size:
        print(f"  {CHECK_TENSOR}: size {got.size:,} != reference {want.size:,}")
        return False
    bad = np.count_nonzero(got != want)
    print(f"  {CHECK_TENSOR}: {bad:,} bytes differ of {got.size:,}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--mode", default="bf16", choices=sorted(sidecar.SUFFIX))
    ap.add_argument("--only", default="mlp.up_proj")
    ap.add_argument("--layers", default=None, help="comma separated layer indices")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    layers = None if args.layers is None else {int(x) for x in args.layers.split(",")}

    q4nx = os.path.join(args.model_dir, "model.q4nx")
    header, data_start = sidecar.read_header(q4nx)
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))

    print(f"Gate: CPU {args.mode} output against the captured reference")
    if not self_check(args.model_dir, header, data_start, args.mode):
        print("  FAILED -- refusing to write a sidecar")
        return 1

    plan = []
    for layer_idx in range(cfg["num_hidden_layers"]):
        if layers is not None and layer_idx not in layers:
            continue
        dims = layer_dims(cfg, layer_idx)
        for suffix, which in IN_FEATURES.items():
            if args.only not in suffix:
                continue
            name = f"model.layers.{layer_idx}.{suffix}.weight"
            if name not in header:
                continue
            cols = dims[which]
            meta = header[name]
            raw_bytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            n_values = raw_bytes // sidecar.BLOCK_BYTES * sidecar.WEIGHTS_PER_BLOCK
            n_bytes = n_values * 2 if args.mode == "bf16" else n_values // 8 * 9
            out_name = name + sidecar.SUFFIX[args.mode]

            def make(name=name, cols=cols):
                raw, _ = sidecar.read_tensor(q4nx, header, data_start, name)
                payload, _ = sidecar.build(raw, cols, args.mode)
                return payload

            plan.append((out_name, n_bytes, sidecar.DTYPE[args.mode], make))

    if not plan:
        print(f"No tensors matched --only {args.only!r}")
        return 1

    total = sum(n for _, n, _, _ in plan)
    out = args.out or f"model.dq_{args.mode}"
    out_path = os.path.join(args.model_dir, out)
    print(f"\nWriting {len(plan)} tensors, {total / 1024**3:.3f} GiB -> {out_path}")

    note = {
        "producer": "make_sidecar.py",
        "mode": args.mode,
        "layout": "dequant.xclbin output order; see sidecar.dequant_offset",
        "source": "model.q4nx",
    }
    for i, name in enumerate(sidecar.write_container(out_path, plan, note)):
        if i % 10 == 0 or i == len(plan) - 1:
            print(f"  [{i + 1:3d}/{len(plan)}] {name}")
    print(f"done: {os.path.getsize(out_path) / 1024**3:.3f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
