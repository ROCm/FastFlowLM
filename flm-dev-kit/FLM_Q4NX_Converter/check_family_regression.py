# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Regression check: a converter family must produce the same model.q4nx before and after a change.

Converts one GGUF with the converter of a git ref (exported to a temp dir) and with this working tree, then compares
every tensor (names, dtype, shape, bytes).  Used when adding a family (e.g. QWEN35_27B) to show the existing families
(Qwen3.5-0.8B / 2B / 4B / 9B) are byte-identical.

    python check_family_regression.py --base-ref HEAD~1 --gguf Qwen3.5-4B-Q4_1.gguf -f qwen3.5-4B [--python python]
Exit status 0 only if the outputs are identical.
--base-ref must be a commit that already has flm-dev-kit/FLM_Q4NX_Converter (the archive path is taken from this file's place in the repo).
"""
import argparse, os, subprocess, sys, tempfile
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))


def convert(tree, gguf, family, out, python, weights_type):
    os.makedirs(out, exist_ok=True)
    subprocess.run([python, "convert.py", "-i", gguf, "-o", out, "-t", weights_type, "-f", family],
                   cwd=tree, check=True, stdout=subprocess.DEVNULL)
    return os.path.join(out, "model.q4nx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ref", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("-f", "--family", required=True)
    ap.add_argument("-t", "--type", default="language")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
    rel = os.path.relpath(HERE, repo)
    with tempfile.TemporaryDirectory() as tmp:
        arch = subprocess.Popen(["git", "archive", a.base_ref, rel], cwd=repo, stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", tmp], stdin=arch.stdout, check=True)
        arch.wait()
        base = convert(os.path.join(tmp, rel), a.gguf, a.family, os.path.join(tmp, "out-base"), a.python, a.type)
        new = convert(HERE, a.gguf, a.family, os.path.join(tmp, "out-new"), a.python, a.type)
        bad = []
        with safe_open(base, "pt") as fb, safe_open(new, "pt") as fn:
            if set(fb.keys()) != set(fn.keys()):
                bad.append(f"tensor names differ: {sorted(set(fb.keys()) ^ set(fn.keys()))[:5]}")
            for k in sorted(set(fb.keys()) & set(fn.keys())):
                tb, tn = fb.get_tensor(k), fn.get_tensor(k)
                if tb.dtype != tn.dtype or tb.shape != tn.shape or tb.contiguous().view(torch.uint8).numpy().tobytes() != tn.contiguous().view(torch.uint8).numpy().tobytes():
                    bad.append(k)
            n = len(list(fb.keys()))
        print(f"{a.family}: {n} tensors compared against {a.base_ref}: " + ("IDENTICAL" if not bad else f"{len(bad)} DIFFER, e.g. {bad[:3]}"))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
