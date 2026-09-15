"""Build a dequantized weight sidecar for FastFlowLM.

Set IRON_PATH to an IRON checkout for --mode bfp, which calls the GEMM
operator's own packer so the layout cannot drift from the kernel's. --mode bf16
needs only numpy.

The sidecar uses the same container format as model.q4nx (safetensors: an 8-byte
little-endian header length, a JSON header, then blobs), so it can later be
merged into the main file by concatenating tensor entries.

Two output modes are supported. ``bf16`` writes what dequant.xclbin would have
written, in the same buffer layout, so the engine can load it in place of
running the dequant app. ``bfp`` writes the bfp16ebs8 form FLMGEMM consumes.

Correctness rests on three facts established against a captured dequant output;
see gate_dequant.py. The value is ``M + S*q`` in f32, the conversion to bf16
rounds toward negative infinity, and logical (k, n) lands at dequant_offset.
"""

import json
import os
import struct
import sys

import numpy as np

IRON = os.environ.get("IRON_PATH", "")

ROW_CHUNK, COL_CHUNK, GROUP, PARALLEL = 32, 256, 32, 16
BLOCK_BYTES = ROW_CHUNK * COL_CHUNK * 5 // 8
WEIGHTS_PER_BLOCK = ROW_CHUNK * COL_CHUNK

SUFFIX = {"bf16": ".dq_bf16", "bfp": ".dq_bfp"}
DTYPE = {"bf16": "BF16", "bfp": "U8"}

# The operator's tiling, from iron/operators/flm/gemm/design.py. Taken locally so
# a weight conversion does not need the AIE toolchain installed; design.py pulls
# in aie.helpers, packing.py does not.
K_TILE, S, T = 512, 8, 8
CT_MAX_K_FOR_N = {16: 16, 32: 32, 64: 128, 128: 32}


def read_header(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len)), 8 + header_len


def read_tensor(path, header, data_start, name):
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        return np.frombuffer(f.read(end - start), dtype=np.uint8), meta


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_floor(x):
    """Convert f32 to bf16 rounding toward negative infinity.

    This is the dequant kernel's accumulator conversion. Truncating the bit
    pattern instead rounds toward zero, which differs on every inexact negative.
    """
    u = x.view(np.uint32)
    inexact = (u & 0xFFFF) != 0
    negative = (u >> 31) != 0
    return ((u >> 16) + (inexact & negative)).astype(np.uint16)


def dequantize(raw, cols):
    """Dequantize a q4nx tensor to f32, shaped (out_features, cols)."""
    blocks_per_row = cols // COL_CHUNK
    n_blocks = raw.size // BLOCK_BYTES
    rows = n_blocks // blocks_per_row * ROW_CHUNK
    b = raw.reshape(n_blocks, BLOCK_BYTES)

    n_groups = COL_CHUNK // GROUP
    sm_bytes = n_groups * ROW_CHUNK * 2
    scales = bf16_to_f32(b[:, :sm_bytes].view(np.uint16).reshape(n_blocks, n_groups, ROW_CHUNK))
    mins = bf16_to_f32(
        b[:, sm_bytes : 2 * sm_bytes].view(np.uint16).reshape(n_blocks, n_groups, ROW_CHUNK)
    )
    qs = b[:, 2 * sm_bytes :].reshape(n_blocks, ROW_CHUNK // PARALLEL, COL_CHUNK, PARALLEL // 2)

    q = np.empty((n_blocks, ROW_CHUNK // PARALLEL, COL_CHUNK, PARALLEL), dtype=np.float32)
    q[..., 0::2] = (qs & 0xF).astype(np.float32)
    q[..., 1::2] = (qs >> 4).astype(np.float32)
    q = q.transpose(0, 1, 3, 2).reshape(n_blocks, ROW_CHUNK, COL_CHUNK)

    grp = np.arange(COL_CHUNK) // GROUP
    s = scales[:, grp, :].transpose(0, 2, 1)
    m = mins[:, grp, :].transpose(0, 2, 1)
    vals = m + s * q

    out = np.empty((rows, cols), dtype=np.float32)
    for i in range(n_blocks):
        r0 = (i // blocks_per_row) * ROW_CHUNK
        c0 = (i % blocks_per_row) * COL_CHUNK
        out[r0 : r0 + ROW_CHUNK, c0 : c0 + COL_CHUNK] = vals[i]
    return out


def dequant_offset(k, n, k_dim):
    """Position of logical (k, n) in the dequant app's output buffer."""
    return (
        (n // 128) * 128 * k_dim
        + (k // 512) * 128 * 512
        + ((n % 128) // 8) * 8 * 512
        + (k % 8) * 512
        + ((k % 512) // 8) * 8
        + (n % 8)
    )


def to_dequant_layout(w):
    """Scatter an (N, K) f32 matrix into the dequant app's output buffer order."""
    n_dim, k_dim = w.shape
    kk, nn = np.meshgrid(np.arange(k_dim), np.arange(n_dim), indexing="ij")
    out = np.empty(k_dim * n_dim, dtype=np.uint16)
    out[dequant_offset(kk, nn, k_dim)] = f32_to_bf16_floor(np.ascontiguousarray(w.T))
    return out


def through_bf16(w):
    """Round f32 to bf16 and widen back, as the dequant app's output would be."""
    return (f32_to_bf16_floor(w).astype(np.uint32) << 16).view(np.float32)


def to_bfp16ebs8(w):
    """Pack an (N, K) matrix into the order and format FLMGEMM's B operand takes.

    Calls the operator's own packer, so the layout cannot drift from the
    kernel's. The values pass through bf16 first, because the weights FLMGEMM
    would otherwise receive have been through the dequant app's bf16 output.
    """
    if IRON and IRON not in sys.path:
        sys.path.insert(0, IRON)
    import torch

    torch.set_num_threads(2)
    from iron.operators.flm.packing import pack_b

    k_dim = w.shape[1]
    # How the operator resolves tile_n on aie2p. Every E2B shape has K > K_TILE
    # and so lands on 64.
    tile_n = 128 if k_dim == K_TILE else 64
    b = torch.from_numpy(np.ascontiguousarray(through_bf16(w).T))  # (K, N)
    packed = pack_b(
        b,
        k_tile=K_TILE,
        n_tile=tile_n,
        s=S,
        t=T,
        ct_k=CT_MAX_K_FOR_N[tile_n],
        bfp16=True,
        round_conv_even=False,  # the cores round toward negative infinity
    )
    return packed.numpy().view(np.uint8)


def build(raw, cols, mode):
    """Produce the sidecar payload for one tensor."""
    w = dequantize(raw, cols)
    if mode == "bf16":
        return to_dequant_layout(w).view(np.uint8), DTYPE[mode]
    if mode == "bfp":
        return to_bfp16ebs8(w), DTYPE[mode]
    raise ValueError(f"unknown mode {mode!r}")


def write_container(path, plan, note):
    """Write tensors in safetensors layout, one at a time.

    ``plan`` holds (name, nbytes, dtype, make_payload) so the header can be
    written before any payload is built; only one tensor is held at a time.
    """
    header, offset = {}, 0
    for name, nbytes, dtype, _ in plan:
        header[name] = {
            "dtype": dtype,
            "shape": [nbytes // (2 if dtype == "BF16" else 1)],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header["__metadata__"] = note

    blob = json.dumps(header).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name, nbytes, _, make_payload in plan:
            payload = make_payload()
            assert payload.nbytes == nbytes, f"{name}: {payload.nbytes} != {nbytes}"
            f.write(payload.tobytes())
            yield name
