#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""fault_runstate_test.py INSTALL_DIR KIND:K [RECOVER=0|1] [N=8] [PROMPT=short|1k]

Needs an engine built with `make FAULT_RUNSTATE=1`. Starts `flm serve`, sends N identical greedy requests and makes the K-th prefill ("p:K": the logits run
that ends a prefill) or the K-th decode token ("d:K": the lm_head round) fail: the engine throws between start() and wait() of the persistent lm_head
run, which leaves that run started but never waited for. Expected with recovery (default): exactly that request returns an explicit error (HTTP 500)
and every other answer equals the first one. With FLM_MINICPM_RECOVER=0 every later request fails ("bad command state, can't launch") until the server
restarts. Needs LD_LIBRARY_PATH to reach the server's dependencies. Exit code 0 when the expectation of the chosen mode holds."""
import json, os, subprocess, sys, time, urllib.request, urllib.error

inst, spec = sys.argv[1], sys.argv[2]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
recover = opt.get("RECOVER", "1") == "1"; n = int(opt.get("N", 8)); prompt = opt.get("PROMPT", "short")
k = int(spec.split(":")[1]); port = str(12900 + (os.getpid() % 90))
text = ("Name three prime numbers greater than 100." if prompt == "short" else "Write one sentence about the sea. " * 130 + "\nWhat did I ask?")
env = dict(os.environ, FLM_MODEL_PATH=os.environ["FLM_MODEL_PATH"], FLM_MINICPM_FAULT_RUNSTATE=spec,
           FLM_MINICPM_RECOVER="1" if recover else "0",
           LD_LIBRARY_PATH=f"{inst}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))

def post():
    b = {"model": "minicpm-v-4.7:1b", "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 12, "stream": False}
    r = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(r, timeout=300))["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")[:120]

while subprocess.run(["pgrep", "-x", "flm"], capture_output=True).returncode == 0:
    time.sleep(5)    # another server holds NPU contexts
srv = subprocess.Popen([f"{inst}/bin/flm", "serve", "minicpm-v-4.7:1b", "-p", port, "--prefix-slots", "0"], env=env,
                       stdout=open(f"fault_runstate_{spec.replace(':', '_')}_r{int(recover)}.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try: urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2); break
        except Exception: time.sleep(1)
    res = [post() for _ in range(n)]
finally:
    srv.terminate()
    try: srv.wait(timeout=30)
    except Exception: srv.kill()
codes = [c for c, _ in res]; good = [t for c, t in res if c == 200]
print("codes", codes)
first_ok = good[0] if good else None
same = all(t == first_ok for t in good)
n_err = codes.count(500)
if recover:
    ok = n_err == 1 and same and len(good) == n - 1
else:
    ok = n_err >= 2 and all(c == 500 for c in codes[codes.index(500):]) if 500 in codes else False
print("PASS" if ok else "FAIL", "recovery" if recover else "no recovery (expects sticky errors)")
sys.exit(0 if ok else 1)
