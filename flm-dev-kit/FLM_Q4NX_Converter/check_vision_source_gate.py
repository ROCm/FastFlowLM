# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Vision source check: a tower that is not BF16 must not be converted silently.

    python check_vision_source_gate.py                      # synthetic tensors, no model files needed
    python check_vision_source_gate.py bf16=mmproj-bf16.gguf f16=mmproj-f16.gguf -f qwen35-2B
    python check_vision_source_gate.py ok=a.gguf refuse=b.gguf -f minicpm-v-1b

The synthetic part exercises `_check_vision_source_is_bf16` and the metadata writer: BF16 passes and
leaves the file byte-identical to what the converter always wrote, F16/quantised sources are refused
(policy "error"), only reported (policy "warn"), escalated by --require-bf16-vision, accepted with
--allow-f16-vision (the rounded count lands in the file metadata), and F32 tensors that bf16 cannot
represent are refused only in strict mode.  With GGUF arguments (`<expect>=<path>`, expect = ok|refuse)
the same check runs on real mmproj files for the converter chosen with -f.

Exit status 0 only if every check behaves as expected.
"""
import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGMLQuantizationType
from safetensors import safe_open
from safetensors.torch import save_file

from q4nx.models.qwen35 import Qwen35
from q4nx.models.qwen3vl import Qwen3VL

fails = 0


def report(ok: bool, what: str):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + what)


def tensor(name, kind, values):
    """Stand-in for a GGUFTensor: only .name, .tensor_type and .data are read by the check."""
    dtype = {GGMLQuantizationType.F16: np.float16, GGMLQuantizationType.F32: np.float32}.get(kind, np.float32)
    data = np.asarray(values, dtype=dtype)
    if kind == GGMLQuantizationType.BF16:   # GGUFReader hands BF16 over as raw 16-bit words
        data = (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return SimpleNamespace(name=name, tensor_type=kind, data=data)


def converter(cls, tensors, policy=None, allow=False, require=False):
    c = object.__new__(cls)             # the base constructor refuses to be called; the check needs no more
    c.gguf_tensors = {t.name: t for t in tensors}
    c.forward_name_map = {}
    c.gguf_reader = SimpleNamespace(data=None)
    c.allow_f16_vision, c.require_bf16_vision = allow, require
    if policy:
        c.vision_source_policy = policy
    return c


def refused(c, **kw):
    try:
        c._check_vision_source_is_bf16(**kw)
    except ValueError as e:
        return "BF16" in str(e)
    return False


BF16, F16, F32 = GGMLQuantizationType.BF16, GGMLQuantizationType.F16, GGMLQuantizationType.F32
good = [tensor("a.weight", BF16, [1.0, 2.0, 3.0]), tensor("b.bias", F32, [0.5, 0.25])]
f16 = [tensor("a.weight", F16, [1.0, 2.0, 3.0001])]
inexact_f32 = [tensor("a.weight", BF16, [1.0]), tensor("n.weight", F32, [1.0 + 2 ** -20])]


def run_synthetic():
    c = converter(Qwen35, good)
    c._check_vision_source_is_bf16()
    report(getattr(c, "q4nx_metadata", None) is None, "BF16 source passes and records nothing (output unchanged)")
    c = converter(Qwen35, good)
    c._check_vision_source_is_bf16(strict_f32=True, record_provenance=True)
    report(c.q4nx_metadata.get("source_dtype") == "BF16:1, F32:1", "provenance records the source dtypes")

    report(Qwen35.vision_source_policy == "error" and refused(converter(Qwen35, f16)), "F16 source is refused (policy error)")
    c = converter(Qwen3VL, f16)
    c._check_vision_source_is_bf16()
    report(Qwen3VL.vision_source_policy == "warn", "F16 source only warns for a converter whose default is warn")
    report(refused(converter(Qwen3VL, f16, require=True)), "--require-bf16-vision escalates warn to error")
    c = converter(Qwen35, f16, allow=True)
    c._check_vision_source_is_bf16()
    report(c.q4nx_metadata.get("rounded_to_bf16") == "1" and c.q4nx_metadata["source_dtype"] == "F16:1",
           "--allow-f16-vision converts and records the rounded count and the real dtype")

    c = converter(Qwen35, inexact_f32)
    c._check_vision_source_is_bf16()
    report(True, "inexact F32 passes when not strict")
    report(refused(converter(Qwen35, inexact_f32), strict_f32=True), "inexact F32 is refused when strict")

    # The writer: without metadata the file is byte-identical to a plain save_file.
    with tempfile.TemporaryDirectory() as d:
        t = {"w": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
        save_file(t, os.path.join(d, "plain.q4nx"))
        c = converter(Qwen35, good)
        c.q4nx_tensors = t
        c._export_q4nx_tensors(d, "written.q4nx")
        same = open(os.path.join(d, "plain.q4nx"), "rb").read() == open(os.path.join(d, "written.q4nx"), "rb").read()
        report(same, "no metadata: written file is byte-identical to a plain save_file")
        c._record_vision_provenance({"BF16": 1})
        c._export_q4nx_tensors(d, "meta.q4nx")
        with safe_open(os.path.join(d, "meta.q4nx"), "pt") as f:
            ok = f.metadata().get("source_dtype") == "BF16:1" and torch.equal(f.get_tensor("w"), t["w"])
        report(ok, "metadata is stored and the tensors are unchanged")


def run_real(specs, force):
    from gguf import GGUFReader
    from q4nx import create_converter
    for spec in specs:
        expect, path = spec.split("=", 1)
        c = create_converter(path, force)
        c.vision_source_policy = "error"
        try:
            c._check_vision_source_is_bf16(strict_f32=getattr(c, "vision_hf_verifier", None) is not None)
            got = "ok"
        except ValueError:
            got = "refuse"
        report(got == expect, f"{path}: {got} (expected {expect})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("real", nargs="*", help="<ok|refuse>=<mmproj.gguf> to check on real files")
    ap.add_argument("-f", "--force", default="", help="model type for the real files (as convert.py -f)")
    args = ap.parse_args()
    run_synthetic()
    if args.real:
        run_real(args.real, args.force)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
