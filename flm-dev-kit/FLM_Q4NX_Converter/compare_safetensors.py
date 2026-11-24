# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""
Compare two safetensor files tensor-by-tensor using cosine similarity.
Prints a sorted report (worst first) to help identify badly dequantized tensors.
"""

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from safetensors import safe_open

CONVERTED  = "/path/to/FastFlowLM/FLM_Q4NX_Converter/merged_safetensors/model.safetensors"
REFERENCE  = "/path/to/FastFlowLM/FLM_Q4NX_Converter/merged_safetensors/model_reference.safetensors"

COSINE_WARN_THRESHOLD = 0.99
NUM_WORKERS = 16


def cosine_sim_pair(name, f_conv, f_ref):
    try:
        a = f_conv.get_tensor(name).reshape(-1).to(torch.float32)
        b = f_ref.get_tensor(name).reshape(-1).to(torch.float32)
    except Exception as e:
        return name, float("nan"), None, None, f"ERROR:{e}"

    if a.shape != b.shape:
        return name, float("nan"), None, None, "SHAPE_MISMATCH"

    dot  = torch.dot(a, b)
    norm = a.norm() * b.norm()
    sim  = (dot / norm).item() if norm != 0 else float("nan")
    flag = "!!BAD!!" if (sim != sim or sim < COSINE_WARN_THRESHOLD) else ""
    return name, sim, tuple(a.shape), tuple(b.shape), flag


def main():
    print(f"Loading index for converted : {CONVERTED}")
    f_conv = safe_open(CONVERTED, framework="pt", device="cpu")
    print(f"Loading index for reference : {REFERENCE}")
    f_ref  = safe_open(REFERENCE,  framework="pt", device="cpu")

    conv_keys = set(f_conv.keys())
    ref_keys  = set(f_ref.keys())

    only_in_conv = conv_keys - ref_keys
    only_in_ref  = ref_keys  - conv_keys
    common       = sorted(conv_keys & ref_keys)

    if only_in_conv:
        print(f"\n[WARN] {len(only_in_conv)} keys only in converted:")
        for k in sorted(only_in_conv):
            print(f"  {k}")
    if only_in_ref:
        print(f"\n[WARN] {len(only_in_ref)} keys only in reference:")
        for k in sorted(only_in_ref):
            print(f"  {k}")

    print(f"\nComparing {len(common)} common tensors with {NUM_WORKERS} workers...\n")

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(cosine_sim_pair, name, f_conv, f_ref): name for name in common}
        for fut in as_completed(futures):
            name, sim, shape_a, shape_b, flag = fut.result()
            results.append((sim, name, shape_a, shape_b, flag))
            done += 1
            print(f"\r  {done}/{len(common)}", end="", flush=True)

    print()

    results.sort(key=lambda x: x[0] if x[0] == x[0] else float("-inf"))

    print(f"\n{'cosine_sim':>12}  {'flag':<14}  {'shape':<30}  name")
    print("-" * 110)
    for sim, name, shape_a, shape_b, flag in results:
        if flag == "SHAPE_MISMATCH":
            print(f"{'SHAPE_MISMATCH':>12}  {flag:<14}  conv={shape_a} ref={shape_b}  {name}")
        else:
            sim_str = f"{sim:.6f}"
            print(f"{sim_str:>12}  {flag:<14}  {str(shape_a):<30}  {name}")

    bad = [r for r in results if r[4]]
    print(f"\n{'='*110}")
    print(f"Summary: {len(bad)} tensors with cosine similarity < {COSINE_WARN_THRESHOLD} (or shape mismatch / nan / error)")


if __name__ == "__main__":
    main()
