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

A gated file missing from either dump fails, as does a layer only one build dumped or no layers at
all; logits_split / logits_chunks may be absent, but not from one side only. Every other tensor is
printed, not gated. Needs only numpy.
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

    def missing_from(rel):
        return [side for side, root in (("npu", args.npu), ("cpu", args.cpu))
                if not os.path.exists(os.path.join(root, rel))]

    def load_pair(rel):
        return np.load(os.path.join(args.npu, rel)), np.load(os.path.join(args.cpu, rel))

    def list_dir(root, rel):
        path = os.path.join(root, rel)
        return set(os.listdir(path)) if os.path.isdir(path) else set()

    print(f"{'tensor':<44} {'cosine':>10} {'relL2':>9}")
    npu_layers = list_dir(args.npu, "layers")
    cpu_layers = list_dir(args.cpu, "layers")
    if not npu_layers or not cpu_layers:
        failures.append(f"layers/: no layer dumps (npu {len(npu_layers)}, cpu {len(cpu_layers)})")
    for nn in sorted(npu_layers ^ cpu_layers):
        failures.append(f"layers/{nn}: dumped by one build only")
    one_sided = set()
    for nn in sorted(npu_layers & cpu_layers):
        files = list_dir(args.npu, f"layers/{nn}") | list_dir(args.cpu, f"layers/{nn}") | {"residual_out.npy"}
        for f in sorted(files):
            rel = f"layers/{nn}/{f}"
            gated = f == "residual_out.npy"
            missing = missing_from(rel)
            if missing and gated:
                failures.append(f"{rel}: missing from {' and '.join(missing)}")
                print(f"{rel:<44} missing from {' and '.join(missing)}  FAIL")
                continue
            if missing:
                one_sided.add(f)
                continue
            got, want = load_pair(rel)
            c = cosine(got, want)
            ok = c > args.min_cos_layer
            if gated and not ok:
                failures.append(f"{rel}: cosine {c:.6f} <= {args.min_cos_layer}")
            print(f"{rel:<44} {c:>10.6f} {rel_l2(got, want):>9.2e}  {('ok' if ok else 'FAIL') if gated else ''}")

    if one_sided:
        print(f"not compared, dumped by one build only: {', '.join(sorted(one_sided))}")

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

    def gated_pair(rel, required):
        missing = missing_from(rel)
        if not missing:
            return load_pair(rel)
        if required or len(missing) == 1:
            failures.append(f"{rel}: missing from {' and '.join(missing)}")
            print(f"\n{rel}: missing from {' and '.join(missing)}  FAIL")
        return None, None

    got, want = gated_pair("logits_last.npy", required=True)
    if got is not None:
        score_logit_rows("last-row logits of the full prefill", got, want, ["last"])

    got, want = gated_pair("logits_split.npy", required=False)
    if got is not None:
        K = args.split
        rows = [K - 1 + j for j in range(got.shape[0])]
        score_logit_rows(f"decode path: prefill {K} tokens, then forward() one token at a time", got, want, rows)

    got, want = gated_pair("logits_chunks.npy", required=False)
    if got is not None:
        score_logit_rows("multi-turn prefill: three chunks", got, want, ["chunk 2", "chunk 3"])

    if missing_from("gen_ids.npy"):
        print(f"\ngen_ids.npy: missing from {' and '.join(missing_from('gen_ids.npy'))} (not gated)")
    else:
        got, want = load_pair("gen_ids.npy")
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
