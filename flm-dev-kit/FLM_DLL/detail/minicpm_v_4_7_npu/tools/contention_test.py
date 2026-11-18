#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""NPU-sharing regression test for the MiniCPM-V engine: another process on the NPU must never turn an answer into garbage.

Starts `flm serve`, takes greedy reference answers while the NPU is exclusive, starts a FOREIGN process that keeps using the NPU, asks the same
prompts again and classifies every answer as same (equal to the reference), error (an HTTP error: acceptable, the engine refused) or diff
(a different answer: the failure this test exists for). After the foreign process stops, the answers must equal the reference again.
Exit code 0 iff there is no diff and the recovery check passes. Needs the same environment as `flm serve` (FLM_MODEL_PATH, LD_LIBRARY_PATH).

  contention_test.py --flm /path/to/flm --foreign 'python3 my_npu_loop.py' [--prompts short|1k|4k] [--n 25] [--max-tokens 24]

--foreign is any process that keeps dispatching on the NPU (e.g. an XRT loop alternating two xclbins, or any other NPU workload). Docs: the note
notes/tricks/2026-11-06-decode-residual-stays-on-the-array-between-runs.md
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.error, urllib.request

PROMPTS = {
    "short": [[{"role": "user", "content": "Name three prime numbers greater than 100."}],
              [{"role": "user", "content": "Hi, my name is Sam."}, {"role": "assistant", "content": "Hello Sam! How can I help?"},
               {"role": "user", "content": "What is my name and what is 2+2?"}]],
    "1k": [[{"role": "user", "content": "Write one sentence about the sea. " * 130 + "\nWhat did I ask?"}]],
    "4k": [[{"role": "user", "content": "Write one sentence about the sea. " * 500 + "\nWhat did I ask?"}]],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flm", required=True)
    ap.add_argument("--model", default="minicpm-v-4.7:1b")
    ap.add_argument("--foreign", required=True)
    ap.add_argument("--prompts", default="short", choices=PROMPTS)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--port", default="12650")
    ap.add_argument("--warm", type=float, default=15.0, help="seconds the foreign process runs before the first request")
    a = ap.parse_args()

    def post(msgs):
        body = {"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": a.max_tokens, "stream": False}
        req = urllib.request.Request(f"http://localhost:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            return 200, json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            return e.code, ""

    srv = subprocess.Popen([a.flm, "serve", a.model, "-p", a.port, "--prefix-slots", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    foreign = None
    try:
        for _ in range(300):
            try: urllib.request.urlopen(f"http://localhost:{a.port}/v1/models", timeout=2); break
            except Exception: time.sleep(1)
        ps = PROMPTS[a.prompts]
        ref = [post(p) for p in ps]
        ref = [post(p) for p in ps]            # the second pass is warm
        if any(c != 200 for c, _ in ref): sys.exit("reference run failed: the NPU was not exclusive")
        foreign = subprocess.Popen(["bash", "-c", "exec " + a.foreign], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(a.warm)
        n = {"same": 0, "error": 0, "diff": 0}
        for _ in range(a.n):
            for k, p in enumerate(ps):
                c, t = post(p)
                n["error" if c != 200 else ("same" if t == ref[k][1] else "diff")] += 1
        os.killpg(foreign.pid, signal.SIGTERM); foreign.wait(timeout=30); foreign = None
        time.sleep(3)
        recovered = all(post(p) == ref[k] for k, p in enumerate(ps))
        print(json.dumps({"under_contention": n, "recovered_after_foreign_stopped": recovered}))
        return 0 if n["diff"] == 0 and recovered else 1
    finally:
        if foreign:
            try: os.killpg(foreign.pid, signal.SIGKILL)
            except Exception: pass
        srv.terminate()
        try: srv.wait(timeout=60)
        except Exception: srv.kill()

if __name__ == "__main__":
    sys.exit(main())
