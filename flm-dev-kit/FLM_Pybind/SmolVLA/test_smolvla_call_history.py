#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Call-history check for the smolvla_npu extension module.

The prefix length (image count, prompt length) changes the attention sequences,
and the kernels' tile parameters are only re-sent when the generator thinks they
changed. This checks that an output does not depend on what ran before it: every
(images, tokens) configuration is run once in a fresh process (the reference),
then all of them are run back to back in one process in a mixed order, and each
result must match its reference (cos >= 0.999).

Run from FLM_Pybind/SmolVLA after `make python` (needs the NPU, no ref file):

    python test_smolvla_call_history.py [model_path]
"""

import os
import subprocess
import sys
import tempfile

import torch

CONFIGS = [(1, 48), (2, 48), (3, 48), (3, 40), (1, 30)]
ORDER = [(3, 48), (1, 48), (3, 40), (2, 48), (3, 48), (1, 30), (1, 48), (2, 48), (3, 40)]
MIN_COS = 0.999  # a stale-state bug gives 0.5-0.8; a prompt-length change alone still costs ~1e-4 (not understood)


def inputs(n_images, n_tokens):
    g = torch.Generator().manual_seed(1234)
    imgs = [(torch.rand(3, 512, 512, generator=g) * 2 - 1).reshape(-1).contiguous() for _ in range(3)]
    tokens = torch.randint(1, 30000, (48,), generator=g, dtype=torch.int64)
    state = torch.randn(32, generator=g)
    noise = torch.normal(0.0, 1.0, size=(50 * 32,), generator=g)
    return imgs[:n_images], tokens[:n_tokens].contiguous(), state, noise


def run(model, calls):
    import smolvla_npu
    m = smolvla_npu.SmolVLA(model)
    return [torch.as_tensor(m.forward(*inputs(*c))).clone().float().flatten() for c in calls]


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "--child":
        out = run(sys.argv[2], [(int(sys.argv[3]), int(sys.argv[4]))])[0]
        torch.save(out, sys.argv[5])
        return 0
    model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMOLVLA_MODEL", ".")
    refs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for c in CONFIGS:
            path = os.path.join(tmp, f"ref_{c[0]}_{c[1]}.pt")
            subprocess.run([sys.executable, __file__, "--child", model, str(c[0]), str(c[1]), path], check=True)
            refs[c] = torch.load(path)
    fails = 0
    for i, (c, out) in enumerate(zip(ORDER, run(model, ORDER))):
        cos = torch.nn.functional.cosine_similarity(out, refs[c], dim=0).item()
        ok = cos >= MIN_COS
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] call {i} images={c[0]} tokens={c[1]} cos vs fresh process = {cos:.6f}")
    print("PASS" if not fails else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
