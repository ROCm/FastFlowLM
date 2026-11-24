#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# Portions of this file consist of AI-generated content.
#
# chat_load.py SECONDS [CHAT_MODEL] -- another process on the NPU: a stock
# `flm serve CHAT_MODEL` (build/rt/stock, no ASR) answering chat requests
# without pause for SECONDS. Prints how many it answered. Run from
# whisper_encoder; needs FLM_TEST_DIR and FLM_MODEL_PATH.

import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ["FLM_TEST_DIR"])
from npu_test.server import FlmServer  # noqa: E402

# Started in the background by a script, this process ignores SIGINT, and so
# would the flm it starts (an ignored signal stays ignored across exec): a
# handler here gives flm the default again, so the server's stop() works.
# SIGTERM stops the load as Ctrl+C does.
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(signal.SIGTERM, signal.default_int_handler)

seconds = float(sys.argv[1])
chat_model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:0.6b"
server = FlmServer(str(HERE / "build/rt/stock/flm"), name="chat-load", log_dir=HERE / "build/step3",
                   args=[chat_model])
server.start()
done = failed = 0
end = time.monotonic() + seconds
try:
    while time.monotonic() < end:
        try:
            server.client.post_json("/v1/chat/completions", {
                "model": chat_model, "max_tokens": 48, "stream": False,
                "messages": [{"role": "user", "content": f"Count from {done} to {done + 20}."}]})
            done += 1
        except Exception:  # noqa: BLE001 (counted)
            failed += 1
            time.sleep(1)
except KeyboardInterrupt:  # the caller stops the load early
    pass
finally:
    server.stop()
print(f"chat load: {done} answered, {failed} failed in {seconds:.0f} s")
