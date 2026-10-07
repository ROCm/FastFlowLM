# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import aie.utils as aie_utils
from aie.iron.device import from_name
from ml_dtypes import bfloat16


def read_exact(size):
    data = bytearray()
    while len(data) < size:
        chunk = sys.stdin.buffer.read(size - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


def receive():
    size = struct.unpack("=I", read_exact(4))[0]
    return json.loads(read_exact(size))


def send(header, payload=b""):
    data = json.dumps(header, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack("=I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def load_llama(command):
    from iron.lm.llama3.model import LLAMA_3_2_1B, Runner

    root = Path(command["model_path"])
    weights = root / "model.safetensors"
    tokenizer = root / "tokenizer.model"
    config = replace(LLAMA_3_2_1B, max_seq_len=command["context_length"])
    return Runner(weights, tokenizer, config).npu()


def load_embedding(command):
    from iron.lm.embeddinggemma2.encoder import Encoder

    return Encoder(command["model_path"], max_tokens=command["context_length"])


def main():
    aie_utils.set_current_device(from_name("npu2", n_cols=8))
    model = None
    while True:
        command = receive()
        try:
            name = command["command"]
            if name == "load_llama":
                model = load_llama(command)
                send({"ok": True})
            elif name == "load_embedding":
                model = load_embedding(command)
                send({"ok": True})
            elif name == "logits":
                tokens = np.frombuffer(read_exact(command["count"] * 4), np.int32)
                logits = np.asarray(model.logits(tokens), dtype=bfloat16)
                words = logits.view(np.uint16).astype("<u2", copy=False)
                send({"ok": True, "count": words.size}, words.tobytes())
            elif name == "reset":
                model._seen = np.empty(0, dtype=np.int64)
                send({"ok": True})
            elif name == "embed":
                text = read_exact(command["size"]).decode()
                value = np.asarray(
                    model(text, command["task"], command["dimensions"]),
                    dtype="<f4",
                )
                send({"ok": True, "count": value.size}, value.tobytes())
            elif name == "close":
                send({"ok": True})
                return
            else:
                raise ValueError(f"unknown command: {name}")
        except Exception as error:
            send({"ok": False, "error": str(error)})


if __name__ == "__main__":
    main()
