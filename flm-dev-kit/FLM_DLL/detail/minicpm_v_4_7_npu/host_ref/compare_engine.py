# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Score the NPU build of the engine against its all-host build, tensor by tensor.

    python compare_engine.py --npu DIR --cpu DIR [--split K]

Both directories come from run_engine (FLM_DUMP_DIR set, DEBUG_LEVEL >= 1 engine) on the same
prompt and the same model.q4nx; run_gate.sh makes them. The gate:

  * residual_out of every layer:            cosine > --min-cos-layer
  * last-row logits, decode rows (--split), multi-turn rows:
                                            cosine > --min-cos-logits, and the same argmax
                                            unless the host's top two logits are within --tie

Every other tensor both builds dumped is printed, not gated. Needs only numpy.
"""
import argparse
import os
import sys

import numpy as np


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def rel_l2(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-300))


def top2_margin(row):
    top = np.sort(np.asarray(row, dtype=np.float64))[-2:]
    return float(top[1] - top[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", required=True, help="run_engine output of the NPU build")
    ap.add_argument("--cpu", required=True, help="run_engine output of the all-host build")
    ap.add_argument("--split", type=int, default=12, help="the K run_engine was given with --split")
    ap.add_argument("--min-cos-layer", type=float, default=0.995)
    ap.add_argument("--min-cos-logits", type=float, default=0.92)
    ap.add_argument("--tie", type=float, default=0.25, help="argmax may differ when the host's top-2 gap is below this")
    args = ap.parse_args()

    failures = []

    def load_pair(rel):
        a, b = os.path.join(args.npu, rel), os.path.join(args.cpu, rel)
        if not (os.path.exists(a) and os.path.exists(b)):
            return None, None
        return np.load(a), np.load(b)

    print(f"{'tensor':<44} {'cosine':>10} {'relL2':>9}")
    layers = sorted(os.listdir(os.path.join(args.npu, "layers")))
    for nn in layers:
        for f in sorted(os.listdir(os.path.join(args.npu, "layers", nn))):
            rel = f"layers/{nn}/{f}"
            got, want = load_pair(rel)
            if got is None:
                continue
            c = cosine(got, want)
            gated = f == "residual_out.npy"
            ok = c > args.min_cos_layer
            if gated and not ok:
                failures.append(f"{rel}: cosine {c:.6f} <= {args.min_cos_layer}")
            print(f"{rel:<44} {c:>10.6f} {rel_l2(got, want):>9.2e}  {('ok' if ok else 'FAIL') if gated else ''}")

    def score_logit_rows(name, npu_rows, cpu_rows, row_ids):
        print(f"\n{name}")
        print(f"{'row':<8} {'cosine':>10} {'argmax npu/cpu':>16} {'cpu top-2 gap':>14}")
        for j, i in enumerate(row_ids):
            c = cosine(npu_rows[j], cpu_rows[j])
            a_n, a_c = int(np.argmax(npu_rows[j])), int(np.argmax(cpu_rows[j]))
            gap = top2_margin(cpu_rows[j])
            ok = c > args.min_cos_logits and (a_n == a_c or gap < args.tie)
            if not ok:
                failures.append(f"{name} row {i}: cosine {c:.6f}, argmax {a_n} vs {a_c} (gap {gap:.3f})")
            print(f"{i:<8} {c:>10.6f} {a_n:>8}/{a_c:<7} {gap:>14.3f}  {'' if ok else 'FAIL'}")

    got, want = load_pair("logits_last.npy")
    score_logit_rows("last-row logits of the full prefill", got, want, ["last"])

    got, want = load_pair("logits_split.npy")
    if got is not None:
        K = args.split
        rows = [K - 1 + j for j in range(got.shape[0])]
        score_logit_rows(f"decode path: prefill {K} tokens, then forward() one token at a time", got, want, rows)

    got, want = load_pair("logits_chunks.npy")
    if got is not None:
        score_logit_rows("multi-turn prefill: three chunks", got, want, ["chunk 2", "chunk 3"])

    got, want = load_pair("gen_ids.npy")
    if got is not None:
        same = 0
        while same < min(len(got), len(want)) and got[same] == want[same]:
            same += 1
        print(f"\ngreedy continuation: the first {same} of {len(want)} ids agree (not gated: a near-tie "
              f"can send the two builds down different but equally valid paths)")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
