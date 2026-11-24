#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Inspect safetensors / q4nx / gguf files: list metadata + tensors, or dump one tensor.

Usage:
    python show_safetensors.py -f model.q4nx
    python show_safetensors.py -f model.safetensors -t lm_head.weight
    python show_safetensors.py -f model.gguf
    python show_safetensors.py -f model.gguf -t token_embd.weight
"""
import argparse
import json
import struct
import sys

import numpy as np

# safetensors dtype string -> (numpy dtype, bytes per element)
# For dtypes numpy can't represent natively (bf16, low-bit quant) we read raw
# bytes and decode/visualize as best we can.
_NP_DTYPES = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("<i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("<u1"),
    "BOOL": np.dtype("?"),
}

# Bit widths for dtypes we don't map to a numpy dtype directly.
_BITS = {
    "BF16": 16,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F64": 64, "F32": 32, "F16": 16,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8,
    "U64": 64, "U32": 32, "U16": 16, "U8": 8,
    "BOOL": 8,
}


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------
def read_safetensors(path):
    """Return (metadata, tensors) where each tensor entry has
    dtype / shape / offset (absolute) / nbytes."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("file too small to be a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        header_bytes = f.read(n)
        if len(header_bytes) < n:
            raise ValueError("truncated header")
        header = json.loads(header_bytes)
    metadata = header.pop("__metadata__", None)
    base = 8 + n
    tensors = {}
    for name, e in header.items():
        start, end = e["data_offsets"]
        tensors[name] = {
            "dtype": e["dtype"],
            "shape": e["shape"],
            "offset": base + start,
            "nbytes": end - start,
        }
    return metadata, tensors


# ---------------------------------------------------------------------------
# gguf
# ---------------------------------------------------------------------------
GGUF_MAGIC = b"GGUF"

# gguf metadata value types
(GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32,
 GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64) = range(13)

# ggml tensor type -> (name, block_size, type_size_bytes)
_GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


class _Cursor:
    """Little sequential reader over a file handle."""

    def __init__(self, f, ver):
        self.f = f
        self.ver = ver

    def _u(self, fmt, size):
        return struct.unpack("<" + fmt, self.f.read(size))[0]

    def u32(self):
        return self._u("I", 4)

    def u64(self):
        return self._u("Q", 8)

    def count(self):
        # v1 used u32 for counts / lengths; v2+ uses u64.
        return self.u32() if self.ver == 1 else self.u64()

    def string(self):
        n = self.count()
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, vtype):
        f = self.f
        if vtype == GGUF_U8:
            return self._u("B", 1)
        if vtype == GGUF_I8:
            return self._u("b", 1)
        if vtype == GGUF_U16:
            return self._u("H", 2)
        if vtype == GGUF_I16:
            return self._u("h", 2)
        if vtype == GGUF_U32:
            return self._u("I", 4)
        if vtype == GGUF_I32:
            return self._u("i", 4)
        if vtype == GGUF_F32:
            return self._u("f", 4)
        if vtype == GGUF_BOOL:
            return bool(self._u("B", 1))
        if vtype == GGUF_U64:
            return self._u("Q", 8)
        if vtype == GGUF_I64:
            return self._u("q", 8)
        if vtype == GGUF_F64:
            return self._u("d", 8)
        if vtype == GGUF_STR:
            return self.string()
        if vtype == GGUF_ARR:
            elem_type = self.u32()
            n = self.count()
            # Big arrays (e.g. tokenizer vocab) are summarized, not materialized.
            if n > 16:
                # still consume the bytes so the cursor stays aligned
                for _ in range(n):
                    self.value(elem_type)
                tname = _GGUF_VALUE_NAMES.get(elem_type, str(elem_type))
                return f"<array {tname} x{n}>"
            return [self.value(elem_type) for _ in range(n)]
        raise ValueError(f"unknown gguf metadata value type: {vtype}")


_GGUF_VALUE_NAMES = {
    GGUF_U8: "u8", GGUF_I8: "i8", GGUF_U16: "u16", GGUF_I16: "i16",
    GGUF_U32: "u32", GGUF_I32: "i32", GGUF_F32: "f32", GGUF_BOOL: "bool",
    GGUF_STR: "str", GGUF_ARR: "array", GGUF_U64: "u64", GGUF_I64: "i64",
    GGUF_F64: "f64",
}


def _align_up(n, a):
    return (n + a - 1) // a * a


def read_gguf(path):
    """Return (metadata, tensors) where each tensor entry has
    dtype / shape / offset (absolute) / nbytes / ggml_type."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError("not a gguf file (bad magic)")
        ver = struct.unpack("<I", f.read(4))[0]
        cur = _Cursor(f, ver)
        tensor_count = cur.count()
        kv_count = cur.count()

        metadata = {}
        for _ in range(kv_count):
            key = cur.string()
            vtype = cur.u32()
            metadata[key] = cur.value(vtype)

        infos = []
        for _ in range(tensor_count):
            name = cur.string()
            n_dims = cur.u32()
            dims = [cur.count() for _ in range(n_dims)]
            ggml_type = cur.u32()
            offset = cur.u64()
            infos.append((name, dims, ggml_type, offset))

        data_start = _align_up(f.tell(), int(metadata.get("general.alignment", 32)))

    tensors = {}
    for name, dims, ggml_type, offset in infos:
        tname, blck, tsize = _GGML_TYPES.get(ggml_type, (f"TYPE_{ggml_type}", 1, 0))
        # gguf stores ne[] fastest-first; reverse to the row-major (torch) shape.
        shape = list(reversed(dims))
        nelem = 1
        for d in dims:
            nelem *= d
        nbytes = (nelem // blck) * tsize if blck else 0
        tensors[name] = {
            "dtype": tname,
            "shape": shape,
            "offset": data_start + offset,
            "nbytes": nbytes,
            "ggml_type": ggml_type,
            "nelem": nelem,
        }
    return metadata, tensors


# ---------------------------------------------------------------------------
# tensor byte reading + decoding
# ---------------------------------------------------------------------------
def read_tensor_bytes(path, entry):
    with open(path, "rb") as f:
        f.seek(entry["offset"])
        return f.read(entry["nbytes"])


def bf16_to_f32(buf):
    """Decode big-block of bfloat16 (little-endian) bytes to float32 array."""
    u16 = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
    # bf16 is just the top 16 bits of an IEEE-754 float32.
    return (u16 << 16).view(np.float32)


def f8_e4m3_to_f32(buf):
    b = np.frombuffer(buf, dtype=np.uint8).astype(np.uint32)
    sign = (b >> 7) & 0x1
    exp = (b >> 3) & 0xF
    man = b & 0x7
    out = np.zeros(b.shape, dtype=np.float32)
    # normal: exp != 0
    norm = exp != 0
    out[norm] = ((1 + man[norm] / 8.0) * 2.0 ** (exp[norm].astype(np.float32) - 7))
    # subnormal: exp == 0, man != 0
    sub = (exp == 0) & (man != 0)
    out[sub] = (man[sub] / 8.0) * 2.0 ** (1 - 7)
    out = out * np.where(sign == 1, -1.0, 1.0).astype(np.float32)
    return out.astype(np.float32)


def f8_e5m2_to_f32(buf):
    b = np.frombuffer(buf, dtype=np.uint8).astype(np.uint32)
    sign = (b >> 7) & 0x1
    exp = (b >> 2) & 0x1F
    man = b & 0x3
    out = np.zeros(b.shape, dtype=np.float32)
    norm = exp != 0
    out[norm] = ((1 + man[norm] / 4.0) * 2.0 ** (exp[norm].astype(np.float32) - 15))
    sub = (exp == 0) & (man != 0)
    out[sub] = (man[sub] / 4.0) * 2.0 ** (1 - 15)
    out = out * np.where(sign == 1, -1.0, 1.0).astype(np.float32)
    return out.astype(np.float32)


def _blocks(buf, tsize):
    """Reshape raw bytes into a contiguous (nblocks, tsize) uint8 matrix."""
    b = np.frombuffer(buf, dtype=np.uint8)
    nb = b.size // tsize
    return b[: nb * tsize].reshape(nb, tsize)


def _f16_col(mat, i):
    """Read a little-endian f16 scalar at byte column [i:i+2] of each block."""
    cols = np.ascontiguousarray(mat[:, i:i + 2])
    return np.frombuffer(cols.tobytes(), dtype="<f2").astype(np.float32)


def dequant_q8_0(buf):
    mat = _blocks(buf, 34)
    d = _f16_col(mat, 0).reshape(-1, 1)
    qs = np.ascontiguousarray(mat[:, 2:34]).view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)


def dequant_q4_0(buf):
    mat = _blocks(buf, 18)
    d = _f16_col(mat, 0).reshape(-1, 1)
    q = mat[:, 2:18].astype(np.int32)
    lo = (q & 0x0F) - 8
    hi = (q >> 4) - 8
    out = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (d * out).reshape(-1)


def dequant_q4_1(buf):
    mat = _blocks(buf, 20)
    d = _f16_col(mat, 0).reshape(-1, 1)
    m = _f16_col(mat, 2).reshape(-1, 1)
    q = mat[:, 4:20].astype(np.int32)
    lo = q & 0x0F
    hi = q >> 4
    out = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (d * out + m).reshape(-1)


def dequant_q5_0(buf):
    mat = _blocks(buf, 22)
    d = _f16_col(mat, 0).reshape(-1, 1)
    qh = np.ascontiguousarray(mat[:, 2:6]).view("<u4").reshape(-1, 1)
    q = mat[:, 6:22].astype(np.int32)
    i = np.arange(16)
    xh_lo = ((qh >> i) << 4) & 0x10
    xh_hi = (qh >> (i + 12)) & 0x10
    lo = ((q & 0x0F) | xh_lo) - 16
    hi = ((q >> 4) | xh_hi) - 16
    out = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (d * out).reshape(-1)


def dequant_q5_1(buf):
    mat = _blocks(buf, 24)
    d = _f16_col(mat, 0).reshape(-1, 1)
    m = _f16_col(mat, 2).reshape(-1, 1)
    qh = np.ascontiguousarray(mat[:, 4:8]).view("<u4").reshape(-1, 1)
    q = mat[:, 8:24].astype(np.int32)
    i = np.arange(16)
    xh_lo = ((qh >> i) << 4) & 0x10
    xh_hi = (qh >> (i + 12)) & 0x10
    lo = (q & 0x0F) | xh_lo
    hi = (q >> 4) | xh_hi
    out = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (d * out + m).reshape(-1)


_GGML_DEQUANT = {
    "Q8_0": dequant_q8_0,
    "Q4_0": dequant_q4_0,
    "Q4_1": dequant_q4_1,
    "Q5_0": dequant_q5_0,
    "Q5_1": dequant_q5_1,
}


def decode_tensor(buf, dtype, shape):
    """Return (numpy_array_or_None, note). None array means undecodable -> raw view."""
    if dtype in _NP_DTYPES:
        arr = np.frombuffer(buf, dtype=_NP_DTYPES[dtype])
        return (arr.reshape(shape) if shape else arr), None
    if dtype == "BF16":
        arr = bf16_to_f32(buf)
        return (arr.reshape(shape) if shape else arr), "decoded bf16 -> float32"
    if dtype == "F8_E4M3":
        arr = f8_e4m3_to_f32(buf)
        return (arr.reshape(shape) if shape else arr), "decoded f8_e4m3 -> float32"
    if dtype == "F8_E5M2":
        arr = f8_e5m2_to_f32(buf)
        return (arr.reshape(shape) if shape else arr), "decoded f8_e5m2 -> float32"
    if dtype in _GGML_DEQUANT:
        arr = _GGML_DEQUANT[dtype](buf)
        if shape:
            try:
                arr = arr.reshape(shape)
            except ValueError:
                pass  # block padding may leave extra elements; show flat
        return arr, f"dequantized {dtype} -> float32"
    # Unknown / quantized dtype: expose raw bytes as uint8.
    return None, f"cannot dequantize {dtype}: showing raw bytes as uint8"


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024.0


def nelems(shape):
    out = 1
    for d in shape:
        out *= d
    return out


def list_file(path, metadata, tensors):
    print(f"File: {path}")
    print(f"Tensors: {len(tensors)}")
    print()
    print("=== Metadata ===")
    if metadata:
        for k, v in metadata.items():
            vs = str(v)
            if len(vs) > 200:
                vs = vs[:200] + "…"
            print(f"  {k}: {vs}")
    else:
        print("  (none)")
    print()

    # dtype summary
    from collections import Counter
    dt = Counter(v["dtype"] for v in tensors.values())
    print("=== dtype summary ===")
    for k, c in sorted(dt.items()):
        print(f"  {k}: {c}")
    print()

    total_bytes = 0
    total_params = 0
    rows = []
    for name, e in tensors.items():
        nb = e["nbytes"]
        total_bytes += nb
        total_params += nelems(e["shape"])
        rows.append((name, e["dtype"], e["shape"], nb))

    name_w = min(max((len(r[0]) for r in rows), default=4), 80)
    dt_w = max((len(r[1]) for r in rows), default=5)
    print("=== Tensors ===")
    hdr = f"  {'NAME'.ljust(name_w)}  {'DTYPE'.ljust(dt_w)}  {'SHAPE':<22}  {'SIZE':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, dtype, shape, nb in rows:
        nm = name if len(name) <= name_w else name[: name_w - 1] + "…"
        print(f"  {nm.ljust(name_w)}  {dtype.ljust(dt_w)}  {str(shape):<22}  {human_bytes(nb):>12}")
    print()
    print(f"Total params: {total_params:,}")
    print(f"Total tensor bytes: {human_bytes(total_bytes)} ({total_bytes:,} bytes)")


def show_tensor(path, tensors, name, edgeitems):
    if name not in tensors:
        print(f"Tensor '{name}' not found.", file=sys.stderr)
        # suggest close matches
        matches = [k for k in tensors if name in k]
        if matches:
            print("Did you mean one of:", file=sys.stderr)
            for m in matches[:20]:
                print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    entry = tensors[name]
    dtype = entry["dtype"]
    shape = entry["shape"]
    buf = read_tensor_bytes(path, entry)

    print(f"Name : {name}")
    print(f"Dtype: {dtype}")
    print(f"Shape: {shape}")
    print(f"Bytes: {human_bytes(len(buf))} ({len(buf):,} bytes)")

    arr, note = decode_tensor(buf, dtype, shape)
    if note:
        print(f"Note : {note}")
    print()

    if arr is None:
        arr = np.frombuffer(buf, dtype=np.uint8)

    # stats (only meaningful for numeric)
    try:
        if arr.size and np.issubdtype(arr.dtype, np.number):
            finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            if finite.size:
                print(
                    f"Stats: min={finite.min():.6g} max={finite.max():.6g} "
                    f"mean={finite.astype(np.float64).mean():.6g} "
                    f"std={finite.astype(np.float64).std():.6g}"
                )
                print()
    except Exception:
        pass

    with np.printoptions(edgeitems=edgeitems, threshold=edgeitems * 2, precision=6, suppress=False):
        print(arr)


def load_file(path):
    """Detect format by magic and return (metadata, tensors)."""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == GGUF_MAGIC:
        return read_gguf(path)
    return read_safetensors(path)


def main():
    ap = argparse.ArgumentParser(description="Inspect safetensors / q4nx / gguf files.")
    ap.add_argument("-f", "--file", required=True, help="path to .safetensors / .q4nx / .gguf file")
    ap.add_argument("-t", "--tensor", help="name of a specific tensor to print")
    ap.add_argument("-e", "--edgeitems", type=int, default=5, help="edge items per side when printing (default 5)")
    args = ap.parse_args()

    try:
        metadata, tensors = load_file(args.file)
    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.tensor:
        show_tensor(args.file, tensors, args.tensor, args.edgeitems)
    else:
        list_file(args.file, metadata, tensors)


if __name__ == "__main__":
    main()
