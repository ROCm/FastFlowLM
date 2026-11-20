#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# hwctx_survey.py FLM OUT_DIR PROMPT_FILE IMAGE MODEL... -- for each model: one
# `FLM serve MODEL` with hwctx_trace.so preloaded (build/hwctx_trace.so next to
# this folder), then four requests, one after the other:
#
#   warmup   a short chat, 8 tokens (loads, first-run work)
#   prefill  PROMPT_FILE as the prompt, 1 token
#   decode   a short prompt, up to 128 tokens
#   image    (a MODEL written MODEL+image) IMAGE and a short prompt, 1 token
#
# Writes OUT_DIR/MODEL.trace (hwctx_trace.c) and OUT_DIR/MODEL.json (each
# request: CLOCK_MONOTONIC start and end in ns, usage). hwctx_switches.py reads
# both. Needs FLM_TEST_DIR (the server class of FastFlowLM's npu_test) and
# FLM_MODEL_PATH.

import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ["FLM_TEST_DIR"])
from npu_test.server import FlmServer  # noqa: E402

flm, out_dir, prompt_file, image = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
models = sys.argv[5:]
tracer = Path(__file__).resolve().parent.parent / "build/hwctx_trace.so"
out_dir.mkdir(parents=True, exist_ok=True)
long_prompt = "Summarize this text in one sentence.\n\n" + prompt_file.read_text()
image_url = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()

for arg in models:
    model, with_image = arg.removesuffix("+image"), arg.endswith("+image")
    name = model.replace(":", "_")
    trace = out_dir / f"{name}.trace"
    trace.unlink(missing_ok=True)
    server = FlmServer(flm, name=f"survey-{name}", log_dir=out_dir, args=[model],
                       env={"LD_PRELOAD": str(tracer), "HWCTX_TRACE": str(trace)})
    record = {"model": model, "requests": []}
    try:
        server.start()
        requests = [("warmup", "Say hello.", 8, None), ("prefill", long_prompt, 1, None),
                    ("decode", "Write a long story about a lighthouse keeper.", 128, None)]
        if with_image:
            requests.append(("image", "Describe this image in one word.", 1, image_url))
        for kind, text, max_tokens, img in requests:
            content = text if img is None else [{"type": "text", "text": text},
                                                {"type": "image_url", "image_url": {"url": img}}]
            time.sleep(1)
            start = time.monotonic_ns()
            try:
                body = server.client.post_json("/v1/chat/completions", {
                    "model": model, "max_tokens": max_tokens, "stream": False, "temperature": 0,
                    "messages": [{"role": "user", "content": content}]})
                usage, error = body.get("usage", {}), None
            except Exception as exc:  # noqa: BLE001 (recorded)
                usage, error = {}, str(exc)[:300]
            end = time.monotonic_ns()
            record["requests"].append({"kind": kind, "start_ns": start, "end_ns": end, "usage": usage, "error": error})
            print(f"{model} {kind}: {(end - start) / 1e6:.0f} ms, usage {usage}{', ERROR ' + error if error else ''}",
                  flush=True)
    except Exception as exc:  # noqa: BLE001 (a model that does not load is recorded)
        record["error"] = str(exc)[:300]
        print(f"{model}: {exc}", flush=True)
    finally:
        server.stop()
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1))
    time.sleep(5)
