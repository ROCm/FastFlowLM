#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# leak.sh <out file> [VAR=value ...]: flm serve with the given FLM_HUGEPAGE_BO* settings, then requests that
# switch the model back and forth (minicpm-v-4.7:1b <-> llama3.2:1b, CYCLES times; each switch unloads the
# old engine and loads the new one). Prints the server's Rss / Pss / AnonHugePages after each request:
# memory freed on unload keeps the minicpm rows flat; a leak grows them every cycle.
W=$(cd "$(dirname "$0")" && pwd)
F=${FLM_DIR:-$W/pkg/flm}
PORT=${PORT:-52638}
CYCLES=${CYCLES:-4}
out=$1; shift
env "$@" $F/flm serve minicpm-v-4.7:1b --port $PORT > $out.serve.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:$PORT/v1/models > /dev/null && break; sleep 2; done
sleep 5
server=$(pgrep -f "flm-real serve minicpm-v-4.7:1b" -u "$(id -u)" | head -1)
echo "env $*  server pid ${server:-none}" > "$out"
python3 - "$PORT" "${server:-0}" "$CYCLES" >> "$out" <<'EOF'
import json, sys, time, urllib.request
port, pid, cycles = sys.argv[1], sys.argv[2], int(sys.argv[3])

def mem():
    v = {}
    try:
        for line in open(f"/proc/{pid}/smaps_rollup"):
            k = line.split(":")[0]
            if k in ("Rss", "Pss", "AnonHugePages"):
                v[k] = int(line.split()[1]) // 1024
    except OSError:
        return "server gone"
    return f"rss {v.get('Rss', 0):6d} MB  pss {v.get('Pss', 0):6d} MB  anon_huge {v.get('AnonHugePages', 0):6d} MB"

print(f"{'start':>28}  {mem()}", flush=True)
for c in range(cycles):
    for model in ("minicpm-v-4.7:1b", "llama3.2:1b"):
        body = {"model": model, "messages": [{"role": "user", "content": "Name three colors."}],
                "temperature": 0, "max_tokens": 16, "stream": False}
        req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=600))
            ok = "ok" if r["choices"][0]["message"]["content"] else "empty"
        except Exception as e:
            ok = f"error {e}"
        time.sleep(3)
        print(f"cycle {c} {model:>16} {ok:5}  {mem()}", flush=True)
EOF
kill $pid; wait $pid 2>/dev/null
cat "$out"
exit 0
