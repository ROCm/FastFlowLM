#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Run the DocVQA accuracy bench against an already-running `flm serve` model.

Ported from FastFlowLM_IRON's .github/scripts/vlm-acc-bench.

Streams the eval set built by docvqa_fetch.py through /v1/chat/completions
(image + question), asks for a single word or phrase, and writes one
prediction record per question. Greedy decoding, so results are deterministic
and any score difference between models/quantizations is attributable to the
weights.

Request handling, failure classification, the per-question record fields and
resume behaviour are shared with gsm8k_run.py -- see acc_common.py. The run only exits non-zero when the
server is unreachable or looks hung (timeouts in a row); whether the result
is usable is score_anls.py's call.

Resumable (re-run to fill in missing or failed questions).

Usage:
  python run_vqa.py --base-url http://127.0.0.1:52625 \
      --data data/docvqa/eval.jsonl --out predictions.jsonl
"""
import argparse
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as ac  # noqa: E402

# Images make prefill slow on the big models; leave plenty of room beyond decode.
TIMEOUT_OVERHEAD_S = 600.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="Model tag served by flm (recorded, and sent in the request).")
    ap.add_argument("--base-url", default="http://127.0.0.1:52625",
                    help="Base URL of the already-running flm server (default: http://127.0.0.1:52625)")
    ap.add_argument("--data", required=True, help="eval.jsonl from docvqa_fetch.py")
    ap.add_argument("--out", required=True, help="output predictions jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=None,
                    help="Fixed per-request timeout in seconds. Default: scale with --max-tokens, then "
                    "adapt to the model's measured decode rate (see acc_common.Asker).")
    ap.add_argument("--max-consecutive-timeouts", type=int, default=ac.MAX_CONSECUTIVE_TIMEOUTS,
                    help="Skip the remaining questions after this many timeouts in a row (0 = never).")
    ap.add_argument("--time-budget-s", type=float, default=0.0,
                    help="Ask no new question once this many seconds have passed (0 = no limit); "
                    "the rest are left out of the run (config.asked < config.n).")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    asker = ac.Asker(args.base_url, args.max_tokens, TIMEOUT_OVERHEAD_S, args.timeout,
                     args.max_consecutive_timeouts, args.time_budget_s)
    print(f"[config] max-tokens={args.max_tokens} timeout={asker.timeout:.0f}s"
          f"{' (adaptive)' if asker.adaptive else ''}", flush=True)

    img_dir = os.path.join(os.path.dirname(os.path.abspath(args.data)), "images")
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    server = ac.server_info(args.base_url)
    done = ac.load_resumable(args.out, "question_id")
    todo = [r for r in rows if str(r["question_id"]) not in done]
    print(f"[data] {len(rows)} total, {len(done)} already done, {len(todo)} to run", flush=True)
    config = {"n": len(rows), "max_tokens": args.max_tokens, "temperature": 0.0, "top_k": 1}
    asked = len(done)
    ac.write_meta(args.out, bench="docvqa", model=args.model, server=server, config={**config, **asker.stats()})
    if not todo:
        print("[data] nothing to do; run score_anls.py on the output.")
        return

    t_start = time.time()
    with open(args.out, "a", encoding="utf-8") as out_f:
        for i, r in enumerate(todo, 1):
            if not asker.time_left():
                ac.annotate("warning", f"DocVQA {args.model}: the {args.time_budget_s:.0f}s time budget is "
                                       f"spent; {len(todo) - i + 1} question(s) not asked")
                break
            asked += 1
            with open(os.path.join(img_dir, os.path.basename(str(r["image"]))), "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            payload = {
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": r["prompt"]},
                ]}],
                "temperature": 0.0, "top_k": 1,
                "max_tokens": args.max_tokens, "stream": False,
            }
            if args.model:
                payload["model"] = args.model
            rep = asker.aborted_reply() if asker.aborted else asker.ask(payload, r["question_id"])
            pred = None if rep.get("error") else (ac.strip_think(rep.get("content")) or None)
            rec = {**r, **ac.reply_fields(rep), "pred": pred, "reasoning_only": ac.reasoning_only(rep, pred)}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            ac.progress(i, len(todo), t_start, r["question_id"], rep, pred, r["answers"][0] if r["answers"] else "")

    ac.write_meta(args.out, bench="docvqa", model=args.model, server=server,
                  config={**config, "asked": asked, **asker.stats()})
    print(f"[done] {(time.time() - t_start) / 60:.1f} min -> {args.out}\n"
          f"       score it:  python score_anls.py {args.out}", flush=True)
    if asker.aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
