# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import struct
import sys


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
    data = json.dumps(header).encode()
    sys.stdout.buffer.write(struct.pack("=I", len(data)) + data + payload)
    sys.stdout.buffer.flush()


def main():
    while True:
        command = receive()
        name = command["command"]
        if name.startswith("load_") or name == "reset":
            send({"ok": True})
        elif name == "logits":
            tokens = struct.unpack(f"={command['count']}i", read_exact(command["count"] * 4))
            send({"ok": True, "count": 2}, struct.pack("=HH", tokens[-1], len(tokens)))
        elif name == "embed":
            text = read_exact(command["size"]).decode()
            send({"ok": True, "count": 2}, struct.pack("=ff", len(text), command["dimensions"]))
        elif name == "exit":
            return
        elif name == "close":
            send({"ok": True})
            return


if __name__ == "__main__":
    main()