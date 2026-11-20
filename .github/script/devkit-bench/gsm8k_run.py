#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
"""Run the GSM8K accuracy bench against an already-running `flm serve` model.

Streams the vendored GSM8K test questions (data/gsm8k_100.jsonl, the first
100 of the GSM8K test set; see THIRD_PARTY_NOTICES.txt) through
/v1/chat/completions, asks for step-by-step work ending in a fixed
`#### <number>` line, and writes one prediction record per question.
Greedy decoding, so results are deterministic and any score difference
between models/quantizations is attributable to the weights.

Request handling, failure classification, the per-question record fields and
resume behaviour are shared with run_vqa.py -- see acc_common.py. The run only exits non-zero when the
server is unreachable or looks hung (timeouts in a row); whether the result
is usable is gsm8k_score.py's call.

Resumable (re-run to fill in missing or failed questions).

Usage:
  python gsm8k_run.py --model llama3.2:1b --base-url http://127.0.0.1:52625 \
      --out results/predictions.jsonl [--n 50] [--time-budget-s 3600]
"""
import argparse
import json
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import acc_common as ac  # noqa: E402

PROMPT_TMPL = (
    "Solve the following grade-school math problem. Show your work, then end "
    "your response with a final line in exactly this format:\n"
    "#### <answer>\n"
    "where <answer> is a single number with no commas, units, or extra text.\n\n"
    "Question: {question}"
)

FINAL_LINE = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")
ANY_NUMBER = re.compile(r"-?[\d,]*\.?\d+")

TIMEOUT_OVERHEAD_S = 180.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model tag passed to `flm serve`.")
    p.add_argument("--base-url", default="http://127.0.0.1:52625",
                   help="Base URL of the already-running flm server.")
    p.add_argument("--data", default=os.path.join(SCRIPT_DIR, "data", "gsm8k_100.jsonl"))
    p.add_argument("--out", required=True, help="Output predictions jsonl.")
    p.add_argument("--n", type=int, default=50, help="Only test the first N questions (0 = all 100).")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Fixed per-request timeout in seconds. Default: scale with --max-tokens, then "
        "adapt to the model's measured decode rate (see acc_common.Asker).",
    )
    p.add_argument("--max-consecutive-timeouts", type=int, default=ac.MAX_CONSECUTIVE_TIMEOUTS,
                   help="Skip the remaining questions after this many timeouts in a row (0 = never).")
    p.add_argument("--time-budget-s", type=float, default=0.0,
                   help="Ask no new question once this many seconds have passed (0 = no limit); "
                   "the rest are left out of the run (config.asked < config.n).")
    return p.parse_args(argv)


def extract_answer(reply):
    """The `#### <n>` answer, else the last number in the reply; None if it has no number."""
    text = ac.strip_think(reply)
    m = FINAL_LINE.search(text)
    if not m:
        matches = ANY_NUMBER.findall(text)
        if not matches:
            return None
        m_text = matches[-1]
    else:
        m_text = m.group(1)
    m_text = m_text.replace(",", "")
    try:
        return float(m_text) if "." in m_text else int(m_text)
    except ValueError:
        return None


def numbers_equal(a, b):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-4


def main(argv=None):
    args = parse_args(argv)
    asker = ac.Asker(args.base_url, args.max_tokens, TIMEOUT_OVERHEAD_S, args.timeout,
                     args.max_consecutive_timeouts, args.time_budget_s)
    print(f"[config] max-tokens={args.max_tokens} timeout={asker.timeout:.0f}s"
          f"{' (adaptive)' if asker.adaptive else ''}", flush=True)

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.n:
        rows = rows[: args.n]

    server = ac.server_info(args.base_url)
    done = ac.load_resumable(args.out, "id")
    todo = [r for r in rows if str(r["id"]) not in done]
    print(f"[data] {len(rows)} total, {len(done)} already done, {len(todo)} to run", flush=True)
    config = {"n": len(rows), "max_tokens": args.max_tokens, "temperature": 0.0, "think": False}
    asked = len(done)
    ac.write_meta(args.out, bench="gsm8k", model=args.model, server=server, config={**config, **asker.stats()})
    if not todo:
        print("[data] nothing to do; run gsm8k_score.py on the output.")
        return

    t_start = time.time()
    with open(args.out, "a", encoding="utf-8") as out_f:
        for i, r in enumerate(todo, 1):
            if not asker.time_left():
                ac.annotate("warning", f"GSM8K {args.model}: the {args.time_budget_s:.0f}s time budget is "
                                       f"spent; {len(todo) - i + 1} question(s) not asked")
                break
            asked += 1
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": PROMPT_TMPL.format(question=r["question"])}],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "think": False,
            }
            rep = asker.aborted_reply() if asker.aborted else asker.ask(payload, r["id"])
            pred = None if rep.get("error") else extract_answer(rep.get("content"))
            rec = {
                "id": r["id"],
                "question": r["question"],
                "gt": r["answer"],
                **ac.reply_fields(rep),
                "pred": pred,
                "reasoning_only": ac.reasoning_only(rep, pred),
                "correct": numbers_equal(pred, r["answer"]),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            ac.progress(i, len(todo), t_start, r["id"], rep, pred, r["answer"])

    ac.write_meta(args.out, bench="gsm8k", model=args.model, server=server,
                  config={**config, "asked": asked, **asker.stats()})
    print(f"[done] {(time.time() - t_start) / 60:.1f} min -> {args.out}\n"
          f"       score it:  python gsm8k_score.py {args.out}", flush=True)
    if asker.aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
