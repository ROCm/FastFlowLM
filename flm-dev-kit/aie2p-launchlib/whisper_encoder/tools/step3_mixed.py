#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# step3_mixed.py VARIANT ROUNDS [CHAT_MODEL] -- step 3 under load: one
# `flm serve CHAT_MODEL -a 1` of a variant (stock, tree, switch, as
# tools/step3_asr.sh), a thread sending chat requests without pause, and ROUNDS
# passes over the English clips of the FLEURS set (en-short, en-long,
# en-longform). The server runs one request at a time, so the chat model's
# hw_contexts run between Whisper's requests, not inside one.
# Each transcript is compared with build/step3/stock's (tools/step3_asr.sh
# stock). Writes build/step3/mixed-VARIANT.json (or mixed-$STEP3_NAME.json). Run from whisper_encoder; needs
# FLM_TEST_DIR and FLM_MODEL_PATH. CHAT_MODEL none: Whisper alone in the server,
# no chat thread; STEP3_CLIPS=N: only the first N clips.

import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ["FLM_TEST_DIR"])
from npu_test.server import FlmServer  # noqa: E402

variant, rounds = sys.argv[1], int(sys.argv[2])
chat_model = sys.argv[3] if len(sys.argv) > 3 else "qwen3:0.6b"
rt = HERE / "build/rt" / ("stock" if variant == "stock" else "tree")
env = {"FLM_WHISPER_SWITCH_DIR": str(HERE / "build/switch")} if variant == "switch" else {}
out_dir = HERE / "build/step3"
name = os.environ.get("STEP3_NAME", variant)
ref = json.loads((out_dir / "stock/asr-whisper-v3_turbo.json").read_text())["items"]
clips_dir = Path(os.environ["FLM_TEST_DIR"]) / "npu_test/assets/asr_eval"
clips = [k for k in ref if ref[k]["set"].startswith("en-")][:int(os.environ.get("STEP3_CLIPS", "1000"))]

server = FlmServer(str(rt / "flm"), name=f"mixed-{name}", log_dir=out_dir,
                   args=([] if chat_model == "none" else [chat_model]) + ["-a", "1"], env=env)
server.start()
stop = threading.Event()
chats = {"done": 0, "failed": 0}


def chat_loop():
    while not stop.is_set():
        try:
            server.client.post_json("/v1/chat/completions", {
                "model": chat_model, "max_tokens": 48, "stream": False,
                "messages": [{"role": "user", "content": f"Count from {chats['done']} to {chats['done'] + 20}."}]})
            chats["done"] += 1
        except Exception:  # noqa: BLE001 (counted; the transcripts decide)
            chats["failed"] += 1
            time.sleep(1)


thread = threading.Thread(target=chat_loop, daemon=True)
if chat_model != "none":
    thread.start()
results, same, total, errors = [], 0, 0, 0
try:
    for r in range(rounds):
        for k in clips:
            path = clips_dir / k
            started = time.monotonic()
            try:
                _, raw = server.client.post_multipart("/v1/audio/transcriptions", {"model": "whisper-v3:turbo"},
                                                      {"file": (k, path.read_bytes(), "audio/mpeg")})
                text = json.loads(raw.decode("utf-8", "replace")).get("text", "")
            except Exception as exc:  # noqa: BLE001 (an error counts as a different transcript)
                text = f"<error: {str(exc)[:200]}>"
                errors += 1
            seconds = time.monotonic() - started
            ok = text == ref[k]["text"]
            same += ok
            total += 1
            results.append({"round": r + 1, "clip": k, "seconds": round(seconds, 3), "identical": ok, "text": text})
            print(f"  {k}: {seconds:.1f} s {'identical' if ok else 'DIFFERENT'}", flush=True)
            if not ok:
                print(f"round {r + 1} {k}: DIFFERENT\n  ref: {ref[k]['text'][:160]}\n  got: {text[:160]}", flush=True)
        print(f"{variant} round {r + 1}: {same}/{total} identical so far, chats {chats['done']} done "
              f"{chats['failed']} failed", flush=True)
finally:
    stop.set()
    if thread.is_alive():
        thread.join(timeout=120)
    server.stop()
(out_dir / f"mixed-{name}.json").write_text(json.dumps(
    {"identical": same, "total": total, "errors": errors, "chats": chats, "results": results}, indent=1, ensure_ascii=False))
print(f"{variant}: {same}/{total} transcripts identical to stock, {errors} errors; chats {chats['done']} done, {chats['failed']} failed")
