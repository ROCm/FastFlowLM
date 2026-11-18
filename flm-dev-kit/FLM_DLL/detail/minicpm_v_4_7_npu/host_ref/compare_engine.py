# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Score the engine's dumps against S2's reference and against the q4-weight oracle.

    source /path/to/FastFlowLM_IRON/FLM_Q4NX_Converter/venv/bin/activate
    python FLM_DLL/detail/minicpm_v_4_7_npu/host_ref/compare_engine.py [--engine DIR] [--split K]

Every tensor the engine dumped (run_engine with FLM_DUMP_DIR, DEBUG_LEVEL >= 1) is scored
twice:

  bf16  vs models_simple/dumps/minicpm_v/     S2's reference on HF's bf16 weights
  q4    vs models_simple/dumps/minicpm_v_q4/  the same reference on model.q4nx's weights
                                              (q4_oracle.py)

The q4 column is the engine's own error; the bf16 column is that plus Q4_K quantization,
whose floor q4_oracle.py prints. The gate is the q4 column: residual_out cosine > --min-cos
at every layer, plus the last-row logits and the decode path (--split) above the same bar,
and the same argmax as the reference.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DUMPS = os.path.join(REPO, "models_simple", "dumps")


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def rel_l2(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(DUMPS, "minicpm_v_engine"))
    ap.add_argument("--ref", default=os.path.join(DUMPS, "minicpm_v"))
    ap.add_argument("--q4", default=os.path.join(DUMPS, "minicpm_v_q4"))
    ap.add_argument("--split", type=int, default=12, help="the K run_engine was given with --split")
    ap.add_argument("--min-cos", type=float, default=0.999)
    args = ap.parse_args()

    failures = []
    rows = []

    def score(rel, got, ref_rel=None, ref_slice=None, gated=False):
        ref_rel = ref_rel or rel
        want_bf, want_q4 = (np.load(os.path.join(d, ref_rel)) for d in (args.ref, args.q4))
        if ref_slice is not None:
            want_bf, want_q4 = want_bf[ref_slice], want_q4[ref_slice]
        want_bf = want_bf.reshape(got.shape)
        want_q4 = want_q4.reshape(got.shape)
        c_bf, c_q4 = cosine(got, want_bf), cosine(got, want_q4)
        floor = cosine(want_q4, want_bf)
        ok = c_q4 > args.min_cos
        if gated and not ok:
            failures.append(f"{rel}: cosine vs q4 oracle {c_q4:.6f} <= {args.min_cos}")
        rows.append((rel, c_q4, rel_l2(got, want_q4), c_bf, floor, ("ok" if ok else "FAIL") if gated else ""))

    for rel in ["embed_tokens.npy"]:
        p = os.path.join(args.engine, rel)
        if os.path.exists(p):
            score(rel, np.load(p), gated=True)

    layers = sorted(os.listdir(os.path.join(args.engine, "layers")))
    for nn in layers:
        d = os.path.join(args.engine, "layers", nn)
        for f in sorted(os.listdir(d)):
            score(f"layers/{nn}/{f}", np.load(os.path.join(d, f)), gated=(f == "residual_out.npy"))

    p = os.path.join(args.engine, "final_norm_last.npy")
    if os.path.exists(p):
        score("final_norm_last.npy", np.load(p), ref_rel="final_norm.npy", ref_slice=slice(-1, None))
    got = np.load(os.path.join(args.engine, "logits_last.npy"))
    score("logits_last.npy", got, ref_rel="logits.npy", ref_slice=slice(-1, None), gated=True)

    print(f"{'tensor':<44} {'cos vs q4':>10} {'relL2 q4':>9} {'cos vs bf16':>11} {'q4 floor':>9}")
    for rel, c_q4, r_q4, c_bf, floor, st in rows:
        print(f"{rel:<44} {c_q4:>10.6f} {r_q4:>9.2e} {c_bf:>11.6f} {floor:>9.6f}  {st}")

    ref_logits = np.load(os.path.join(args.ref, "logits.npy"))
    q4_logits = np.load(os.path.join(args.q4, "logits.npy"))
    am_eng, am_ref, am_q4 = int(got[-1].argmax()), int(ref_logits[-1].argmax()), int(q4_logits[-1].argmax())
    print(f"\nargmax logits[-1]: engine {am_eng}  bf16 reference {am_ref}  q4 oracle {am_q4}")
    if am_eng != am_ref:
        failures.append(f"argmax logits[-1]: engine {am_eng} != reference {am_ref}")

    # decode path: prefill ids[:K], then forward() one token at a time
    p = os.path.join(args.engine, "logits_split.npy")
    if os.path.exists(p):
        split = np.load(p)
        K = args.split
        print(f"\ndecode path: prefill {K} tokens, then forward() for tokens {K}..{K + split.shape[0] - 2}")
        print(f"{'row':<8} {'cos vs q4':>10} {'cos vs bf16':>11} {'argmax eng/ref':>16}")
        for j in range(split.shape[0]):
            i = K - 1 + j
            c_q4, c_bf = cosine(split[j], q4_logits[i]), cosine(split[j], ref_logits[i])
            a_e, a_r = int(split[j].argmax()), int(ref_logits[i].argmax())
            ok = c_q4 > args.min_cos and a_e == a_r
            if not ok:
                failures.append(f"decode row {i}: cosine vs q4 oracle {c_q4:.6f}, argmax {a_e} vs {a_r}")
            print(f"{i:<8} {c_q4:>10.6f} {c_bf:>11.6f} {a_e:>8}/{a_r:<7} {'' if ok else 'FAIL'}")

    # multi-turn prefill: [0, K), [K, K+2), [K+2, L)
    p = os.path.join(args.engine, "logits_chunks.npy")
    if os.path.exists(p):
        chunks = np.load(p)
        K = args.split
        print(f"\nmulti-turn prefill: chunks [0,{K}) [{K},{K + 2}) [{K + 2},{ref_logits.shape[0]})")
        for j, i in enumerate((K + 1, ref_logits.shape[0] - 1)):
            c_q4, c_bf = cosine(chunks[j], q4_logits[i]), cosine(chunks[j], ref_logits[i])
            a_e, a_r = int(chunks[j].argmax()), int(ref_logits[i].argmax())
            ok = c_q4 > args.min_cos and a_e == a_r
            if not ok:
                failures.append(f"multi-turn row {i}: cosine vs q4 oracle {c_q4:.6f}, argmax {a_e} vs {a_r}")
            print(f"{i:<8} {c_q4:>10.6f} {c_bf:>11.6f} {a_e:>8}/{a_r:<7} {'' if ok else 'FAIL'}")

    p = os.path.join(args.engine, "gen_ids.npy")
    if os.path.exists(p):
        gen = np.load(p).astype(np.int64).tolist()
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("/path/to/models/MiniCPM-V-4.7-1B-NPU2")
            print(f"\ngreedy continuation: {tok.decode(gen)!r}")
        except Exception as e:  # tokenizer is only for display
            print(f"\ngreedy ids: {gen} ({e})")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
