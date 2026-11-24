#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Reload-between-runs regression test for the MiniCPM-V decode design (needs a test build: `make FAULT_RELOAD=1`, see the Makefile).

A decode token is a list of 25 runs on one hw context. Another process's command between two of them makes the firmware reload the design and wipes whatever
the design keeps on the array. A test build can force that without a second process: with FLM_MINICPM_FAULT_RELOAD=<kinds> it issues the runs one by one and
dispatches an empty sequence of another xclbin before the chosen ones (R = the context-length run, F = first linear, L = other linear, A = attention,
H = the lm_head run, `all`, `rand:P` percent of the runs, `at:2,5` run indices). The answers must equal the exclusive one for a design that keeps no state
between runs (layer_lm_sl.xclbin).

  fault_reload_test.py --flm /path/to/flm [--modes A,L,F,all,rand:30,rand:70] [--max-tokens 64] [--expect-diff]

Exit code 0 iff every mode gives the exclusive answer (--expect-diff: iff some mode differs, to check that the test bites on the stateful layer_lm.xclbin
with FLM_MINICPM_LM_STATELESS=0). Same environment as `flm serve` (FLM_MODEL_PATH, LD_LIBRARY_PATH pointing at the test build's lib).
"""
import argparse, json, os, subprocess, sys, time, urllib.request

PROMPTS = [[{"role": "user", "content": "Write one sentence about the sea. " * 30 + "\nWhat did I ask? Then count from one to ten."}],
           [{"role": "user", "content": "Name three prime numbers greater than 100 and explain why they are prime."}]]

def run(flm, model, port, mode, max_tokens):
    env = dict(os.environ)
    if mode != "-": env["FLM_MINICPM_FAULT_RELOAD"] = mode
    srv = subprocess.Popen([flm, "serve", model, "-p", str(port), "--prefix-slots", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(300):
            try: urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2); break
            except Exception: time.sleep(1)
        out = []
        for p in PROMPTS * 2:   # the second round is warm
            body = {"model": model, "messages": p, "temperature": 0, "max_tokens": max_tokens, "stream": False}
            req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            out.append(json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"])
        return out[2:]
    finally:
        srv.terminate()
        try: srv.wait(timeout=60)
        except Exception: srv.kill()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flm", required=True); ap.add_argument("--model", default="minicpm-v-4.7:1b")
    ap.add_argument("--modes", default="A,L,F,all,rand:30,rand:70"); ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--port", type=int, default=12790); ap.add_argument("--expect-diff", action="store_true")
    a = ap.parse_args()
    ref = run(a.flm, a.model, a.port, "-", a.max_tokens)
    bad = 0
    for m in a.modes.split(","):
        r = run(a.flm, a.model, a.port, m, a.max_tokens)
        same = r == ref
        print(f"{m:10s} {'same' if same else 'DIFF'}")
        bad += 0 if same else 1
    sys.exit((0 if bad else 1) if a.expect_diff else (1 if bad else 0))

if __name__ == "__main__":
    main()
