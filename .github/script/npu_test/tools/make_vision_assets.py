#!/usr/bin/env python3
"""Write the synthetic images in assets/vision/ that the chat suite asks about.

A photo only supports "the model said something"; a large solid square on a
white background has one right answer, so a vision encoder that hands the
language model garbage fails the colour check instead of passing it. Two
colours, so a model that always answers "red" still fails.

The images are committed; rerun this only to change them:

    python .github/script/npu_test/tools/make_vision_assets.py

Standard library only, and deterministic: the same bytes on every run.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "vision"

# 448 is a whole number of patches for the Qwen-VL (28 px) and Gemma (14 px)
# encoders, so no resize blurs the square's edges.
SIZE = 448
MARGIN = 80
WHITE = (255, 255, 255)

SQUARES = {
    "red_square.png": (220, 20, 20),
    "blue_square.png": (20, 40, 220),
}


def png(width: int, height: int, pixel) -> bytes:
    """An 8-bit RGB PNG; pixel(x, y) returns an (r, g, b) tuple."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            rows.extend(pixel(x, y))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def square(color):
    inside = range(MARGIN, SIZE - MARGIN)
    return lambda x, y: color if x in inside and y in inside else WHITE


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, color in SQUARES.items():
        path = OUT_DIR / name
        path.write_bytes(png(SIZE, SIZE, square(color)))
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
