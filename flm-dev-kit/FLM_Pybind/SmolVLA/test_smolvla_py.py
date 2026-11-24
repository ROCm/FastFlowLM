#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Verification script for the smolvla_npu extension module.

Drives the module with the captured reference tensors in
<model_path>/ref.safetensors and checks the pieces the C++ CLI cannot: that the
extension imports, that the single-instance rule holds (including releasing the
NPU on delete), that forward() accepts torch tensors, and that repeated calls
with the same explicit noise are bit-identical.

Run from FLM_Pybind/SmolVLA after `make python`:

    python test_smolvla_py.py   # any env with torch + the smolvla_npu module built by `make python`
"""

import argparse
import gc
import os
import sys

import torch
from safetensors.torch import load_file

import smolvla_npu

FAILURES = []


def check(name, ok, detail=""):
    """Record and print one pass/fail line."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def load_reference(model_path):
    """Read the captured observation + noise + expected action from ref.safetensors."""
    ref = load_file(os.path.join(model_path, "ref.safetensors"))
    images = [
        ref["vision.camera1.pixel_values"].float().reshape(-1).contiguous(),
        ref["vision.camera2.pixel_values"].float().reshape(-1).contiguous(),
    ]
    tokens = ref["input.processed.observation.language.tokens"].reshape(-1).contiguous()
    state = ref["input.processed.observation.state"].float().reshape(-1).contiguous()
    noise = ref["input.noise"].float().reshape(-1).contiguous()
    pred = ref["result.pred_action"].float().reshape(-1)
    return images, tokens, state, noise, pred


def flm_models_dir():
    """Where flm keeps its models: $FLM_MODEL_PATH/models, else ~/.config/flm/models."""
    root = os.environ.get("FLM_MODEL_PATH")
    if root:
        return os.path.join(root, "models")
    return os.path.join(os.path.expanduser("~"), ".config", "flm", "models")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path",
                    default=os.path.join(flm_models_dir(), "SmolVLA"),
                    help="model directory (config.json, model.safetensors, XCLBINS/)")
    args = ap.parse_args()

    print(f"model path: {args.model_path}")
    print(f"module:     {smolvla_npu.__file__}")

    # --- 1. construct -----------------------------------------------------
    print("\n=== 1. import + construct ===")
    m = smolvla_npu.SmolVLA(args.model_path)
    print(f"  {m!r}")
    check("chunk_size == 50", m.chunk_size == 50, f"got {m.chunk_size}")
    check("action_dim == 32", m.action_dim == 32, f"got {m.action_dim}")
    check("state_dim > 0", m.state_dim > 0, f"got {m.state_dim}")

    # --- 2. singleton -----------------------------------------------------
    # A second instance must be refused: the XRT device and the NPU hardware
    # context it owns cannot be shared.
    print("\n=== 2. single-instance rule ===")
    try:
        smolvla_npu.SmolVLA(args.model_path)
        check("second instance raises", False, "construction unexpectedly succeeded")
    except RuntimeError as e:
        check("second instance raises RuntimeError", True, str(e))

    # A bad path must raise rather than exit(1) out of the interpreter. The
    # singleton check fires first while `m` is alive, so this is covered in
    # test_smolvla_badpath.py, which runs in a fresh process.

    # --- 3. forward -------------------------------------------------------
    print("\n=== 3. forward ===")
    images, tokens, state, noise, ref_pred = load_reference(args.model_path)
    print(f"  images: {len(images)} x {images[0].numel()} | tokens: {tokens.numel()}"
          f" | state: {state.numel()} | noise: {noise.numel()}")

    out = m.forward(images, tokens, state, noise)
    check("output dtype float32", out.dtype == torch.float32, str(out.dtype))
    check(f"output shape [{m.chunk_size}, {m.action_dim}]",
          tuple(out.shape) == (m.chunk_size, m.action_dim), str(tuple(out.shape)))
    check("output all finite", bool(torch.isfinite(out).all()))

    # --- 4. determinism ---------------------------------------------------
    # Same explicit noise, same observation: no RNG in the path, so the two runs
    # must agree bit for bit.
    print("\n=== 4. determinism (explicit noise) ===")
    out2 = m.forward(images, tokens, state, noise)
    check("two forwards bitwise identical", bool(torch.equal(out, out2)),
          f"max abs diff {float((out - out2).abs().max()):.3e}")

    # --- 5. reference comparison (reporting only) -------------------------
    # Not a pass/fail gate. run.exe only matches result.pred_action after
    # overwriting the kv cache with reference values (main.cpp:187-196), to work
    # around a known upstream attention-mask discrepancy. Python always runs the
    # real prefill (skip_prefill is always false), so a larger error here is
    # expected -- cross-check against `./run.exe --skip false` instead.
    print("\n=== 5. vs reference pred_action (reporting only) ===")
    out_dim = ref_pred.numel()
    got = out[0, :out_dim]
    print(f"  got: [{', '.join(f'{v:.5f}' for v in got.tolist())}]")
    print(f"  ref: [{', '.join(f'{v:.5f}' for v in ref_pred.tolist())}]")
    print(f"  max abs error: {float((got - ref_pred).abs().max()):.5f}")
    print("  (informational: the python path never skips the prefill)")

    # --- 6. argument validation -------------------------------------------
    print("\n=== 6. argument validation ===")
    bad_cases = [
        ("wrong image dtype",
         lambda: m.forward([images[0].to(torch.float64), images[1]], tokens, state, noise)),
        ("wrong image size",
         lambda: m.forward([images[0][:100], images[1]], tokens, state, noise)),
        ("too many images",
         lambda: m.forward(images * 3, tokens, state, noise)),
        ("empty image list",
         lambda: m.forward([], tokens, state, noise)),
        ("wrong noise size",
         lambda: m.forward(images, tokens, state, noise[:10])),
        ("non-contiguous image",
         lambda: m.forward([images[0].reshape(2, -1).t()[:, 0], images[1]],
                           tokens, state, noise)),
    ]
    for name, fn in bad_cases:
        try:
            fn()
            check(name + " rejected", False, "call unexpectedly succeeded")
        except (ValueError, RuntimeError, TypeError) as e:
            check(name + " rejected", True, type(e).__name__)

    # noise=None must sample instead of failing.
    smolvla_npu.seed(0)
    sampled = m.forward(images, tokens, state)
    check("noise=None samples", bool(torch.isfinite(sampled).all())
          and tuple(sampled.shape) == (m.chunk_size, m.action_dim))

    # --- 7. teardown ------------------------------------------------------
    # Rebuilding after a forward genuinely does not work: the fresh instance
    # loads fine but its first NPU run dies with ERT_CMD_STATE_TIMEOUT. This is
    # a property of the C++/XRT stack, not the binding -- a pure C++ program
    # doing construct/forward/destroy/construct/forward fails the same way, even
    # with the xrt::device shared across rounds. The binding detects it up front
    # so the caller gets an explanation instead of a timeout inside forward().
    print("\n=== 7. teardown after forward ===")
    del m
    gc.collect()
    try:
        smolvla_npu.SmolVLA(args.model_path)
        check("rebuild after forward refused", False,
              "construction unexpectedly succeeded (would time out on next forward)")
    except RuntimeError as e:
        check("rebuild after forward refused", True, str(e))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
