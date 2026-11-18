#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# step3_timing.py VARIANT ROUNDS AUDIO... -- step 3: one `flm serve -a 1` of a
# variant (as tools/step3_asr.sh: stock, tree, switch), then each AUDIO file
# transcribed ROUNDS times in turn. Prints seconds per request and writes
# build/step3/timing-VARIANT.json (per file: the seconds and the text of each
# round). Run from whisper_encoder; needs FLM_TEST_DIR and FLM_MODEL_PATH (the
# server and client of FastFlowLM's npu_test are used as they are).

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ["FLM_TEST_DIR"])
from npu_test.server import FlmServer  # noqa: E402

variant, rounds, audio = sys.argv[1], int(sys.argv[2]), [Path(a) for a in sys.argv[3:]]
rt = HERE / "build/rt" / ("stock" if variant == "stock" else "tree")
env = {"FLM_WHISPER_SWITCH_DIR": str(HERE / "build/switch")} if variant == "switch" else {}
out_dir = HERE / "build/step3"
server = FlmServer(str(rt / "flm"), name=f"timing-{variant}", log_dir=out_dir, args=["-a", "1"], env=env)
results = {str(a): {"seconds": [], "text": []} for a in audio}
server.start()
try:
    for r in range(rounds):
        for a in audio:
            started = time.monotonic()
            _, raw = server.client.post_multipart("/v1/audio/transcriptions", {"model": "whisper-v3:turbo"},
                                                  {"file": (a.name, a.read_bytes(), "audio/mpeg")})
            seconds = time.monotonic() - started
            body = json.loads(raw.decode("utf-8", "replace"))
            if "error" in body:
                raise RuntimeError(f"{a.name}: {body}")
            results[str(a)]["seconds"].append(round(seconds, 3))
            results[str(a)]["text"].append(body.get("text", ""))
            print(f"{variant} round {r + 1} {a.name}: {seconds:.2f} s, {len(body.get('text', ''))} chars", flush=True)
finally:
    server.stop()
(out_dir / f"timing-{variant}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
