#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.
# check_out.sh <out file> [VAR=value ...]: greedy answers of minicpm-v-4.7:1b to a short and a ~2k-token
# prompt through flm serve with the given FLM_HUGEPAGE_BO* settings, to compare the switch off and on.
W=$(cd "$(dirname "$0")" && pwd)
F=${FLM_DIR:-$W/pkg/flm}
PORT=${PORT:-52637}
out=$1; shift
env "$@" $F/flm serve minicpm-v-4.7:1b --port $PORT > $out.serve.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:$PORT/v1/models > /dev/null && break; sleep 2; done
sleep 5
python3 - "$PORT" "$out" <<'EOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
para = ("The history of the printing press begins in the fifteenth century, when movable type changed how "
        "books were made, sold and read across Europe, and with them how ideas travelled. ")
prompts = ["What is the capital of France? Answer in one sentence.",
           para * 60 + "\nSummarize the text above in two sentences."]
res = []
for p in prompts:
    body = {"model": "minicpm-v-4.7:1b", "messages": [{"role": "user", "content": p}],
            "temperature": 0, "top_k": 1, "seed": 1, "max_tokens": 128, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    res.append({"prompt_tokens": r["usage"]["prompt_tokens"], "text": r["choices"][0]["message"]["content"]})
json.dump(res, open(out, "w"), indent=1)
EOF
kill $pid; wait $pid 2>/dev/null
exit 0
