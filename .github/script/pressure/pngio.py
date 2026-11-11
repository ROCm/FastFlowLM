"""Reads and writes 8-bit, non-interlaced RGB/RGBA PNGs with the standard library only.

The Windows runners have a plain Python without Pillow, and PNG is lossless, so the
pixels a request sends as `image_raw` and as a PNG `image_url` are the same bytes.
"""

import struct
import zlib

_SIG = b"\x89PNG\r\n\x1a\n"


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def read_rgb(path):
    """Returns (width, height, rgb bytes); alpha is dropped."""
    data = open(path, "rb").read()
    if data[:8] != _SIG:
        raise ValueError(f"{path}: not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + n
    w, h, depth, ctype, _, _, interlace = ihdr
    if depth != 8 or ctype not in (2, 6) or interlace:
        raise ValueError(f"{path}: only 8-bit non-interlaced RGB/RGBA PNGs are supported")
    bpp = 3 if ctype == 2 else 4
    raw = zlib.decompress(b"".join(idat))
    stride = w * bpp
    prev = bytearray(stride)
    out = bytearray(w * h * 3)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if f == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif f == 2:
            line = bytearray((a + b) & 255 for a, b in zip(line, prev))
        elif f == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up_left = prev[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + _paeth(left, prev[i], up_left)) & 255
        elif f != 0:
            raise ValueError(f"{path}: bad filter {f}")
        o = y * w * 3
        if bpp == 3:
            out[o:o + stride] = line
        else:
            row = out[o:o + w * 3]
            row[0::3], row[1::3], row[2::3] = line[0::4], line[1::4], line[2::4]
            out[o:o + w * 3] = row
        prev = line
    return w, h, bytes(out)


def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def write_rgb(w, h, rgb):
    """PNG bytes for one RGB frame (filter 0 on every row)."""
    stride = w * 3
    rows = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(h))
    return (_SIG + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(rows, 6)) + _chunk(b"IEND", b""))
