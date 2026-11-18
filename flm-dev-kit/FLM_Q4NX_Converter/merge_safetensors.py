#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Merge multiple .safetensors files into one and emit model.safetensors.index.json."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def merge(inputs, out_dir, out_name="model.safetensors"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    metadata = {}
    weight_map = {}
    total_size = 0

    for path in inputs:
        path = Path(path)
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
            if meta:
                metadata.update(meta)
            for key in f.keys():
                if key in tensors:
                    raise ValueError(f"Duplicate tensor key '{key}' (from {path})")
                t = f.get_tensor(key)
                tensors[key] = t
                weight_map[key] = out_name
                total_size += t.numel() * t.element_size()

    save_file(tensors, out_dir / out_name, metadata=metadata or None)

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"Merged {len(inputs)} files -> {out_dir / out_name}")
    print(f"Tensors: {len(weight_map)}  Total size: {total_size} bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="Input .safetensors files")
    ap.add_argument("-o", "--out-dir", default=".", help="Output directory")
    ap.add_argument("-n", "--name", default="model.safetensors",
                    help="Output safetensors filename")
    args = ap.parse_args()
    merge(args.inputs, args.out_dir, args.name)


if __name__ == "__main__":
    #python merge_safetensors.py shards/*.safetensors -o ./merged
    #python3 merge_safetensors.py /path/to/huggingfaceCache/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/*.safetensors -o ./merged_safetensors
    main()