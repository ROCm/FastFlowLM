# FLM#1001 suite driver, Linux and Windows: for each server config start `flm serve`, run long_tests.py
# cases against it, stop it. Also the load matrix (does each --ctx-len come up, what does it warn).
# usage: ci_suite.py <flm executable> <out dir> <plan name> [model]
import json, os, subprocess, sys, time, urllib.request

FLM, OUT, PLAN = sys.argv[1], sys.argv[2], sys.argv[3]
MODEL = sys.argv[4] if len(sys.argv) > 4 else "qwen3.8-mtp:27b"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
TAG = MODEL.replace(":", "_").replace(".", "_")

def up(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=2); return True
    except Exception:
        return False

def serve(name, ctx, env, port, extra):
    log = open(os.path.join(OUT, f"{TAG}_{name}.log"), "w")
    e = dict(os.environ); e.update(env)
    p = subprocess.Popen([FLM, "serve", MODEL, "--port", str(port), "--ctx-len", str(ctx)] + extra,
                         stdout=log, stderr=subprocess.STDOUT, env=e)
    for _ in range(300):
        time.sleep(2)
        if p.poll() is not None or up(port): break
    return p, log

def stop(p, log):
    p.kill(); p.wait(); log.close(); time.sleep(5)

def grep(path, words):
    with open(path, errors="replace") as f:
        return [l.strip()[:200] for l in f if any(w in l for w in words)]

# (name, --ctx-len, env, cases); cases "" = load check only
PLANS = {
  "lin": [
    ("load8192", 8192, {}, ""), ("load8193", 8193, {}, ""), ("load12288", 12288, {}, ""),
    ("load16384", 16384, {}, ""), ("load32768", 32768, {}, ""),
    ("spec16k", 16384, {}, "calibrate,short,needle9k,needle12k,needle16k,fill16377,fill16377_json,fill16377_tools,"
                           "tail16377_40,tail16377_5,tail16377_-30,over16377,grow16377,reporter"),
    ("nospec16k", 16384, {"FLM_DISABLE_SPECULATION": "1"}, "calibrate,fill16377,tail16377_5,grow16377"),
    ("ctx10500", 10500, {}, "calibrate,reporter,fill10500"),
    ("ctx8192", 8192, {}, "calibrate,short,needle6k,fill8192"),
  ],
  "win": [
    ("load8192", 8192, {}, ""), ("load12288", 12288, {}, ""), ("load16384", 16384, {}, ""), ("load32768", 32768, {}, ""),
    ("spec16k", 16384, {}, "calibrate,short,needle12k,needle16k,fill16377,fill16377_json,tail16377_5,tail16377_-30,"
                           "over16377,grow16377"),
    ("nospec16k", 16384, {"FLM_DISABLE_SPECULATION": "1"}, "calibrate,fill16377,tail16377_5"),
    ("ctx10500", 10500, {}, "calibrate,reporter"),
  ],
  "local2": [
    ("nospec16k", 16384, {"FLM_DISABLE_SPECULATION": "1"}, "calibrate,fill16377,tail16377_5"),
    ("ctx10500", 10500, {}, "calibrate,reporter,fill10500"),
    ("ctx8192", 8192, {}, "calibrate,short,needle6k,fill8192"),
  ],
  "main8192": [
    ("ctx8192", 8192, {}, "calibrate,short,needle6k,fill8192"),
  ],
  "dflash": [
    ("load16384", 16384, {}, ""),
    ("spec16k", 16384, {}, "calibrate,short,needle12k,fill16377,fill16377_json,tail16377_5,over16377"),
    ("ctx10500", 10500, {}, "calibrate,reporter"),
  ],
}

summary = []
port = 53100
for name, ctx, env, cases in PLANS[PLAN]:
    port += 1
    args_env = {k: v for k, v in env.items() if not k.startswith("__")}
    p, log = serve(name, ctx, args_env, port, env.get("__args", "").split())
    path = os.path.join(OUT, f"{TAG}_{name}.log")
    alive = p.poll() is None and up(port)
    line = f"[{name}] ctx={ctx} up={alive} :: " + " | ".join(grep(path, ["WARNING", "rror", "chunks"])[:4])
    print(line, flush=True); summary.append(line)
    if alive and cases:
        r = subprocess.run([sys.executable, os.path.join(HERE, "long_tests.py"), str(port),
                            os.path.join(OUT, f"{TAG}_{name}.json"), cases, MODEL], capture_output=True, text=True)
        print(r.stdout, r.stderr[-2000:], flush=True)
        summary += [f"  {l}" for l in r.stdout.splitlines()]
        still = p.poll() is None and up(port)
        summary.append(f"  server after: {'alive' if still else 'DEAD'}")
    stop(p, log)

open(os.path.join(OUT, f"{TAG}_summary.txt"), "w").write("\n".join(summary) + "\n")
bad = [l for l in summary if "FAIL" in l or "up=False" in l or "DEAD" in l]
print("SUITE_PASS" if not bad else "SUITE_FAIL:\n" + "\n".join(bad), flush=True)
sys.exit(0)
