# DMA-only page-translation probe (FastFlowLM#1080).
# Every shim channel pair moves bytes host -> mem tile -> host, no core code.
# One side can be scattered: rows of R bytes, one row per 4 KB page (stride 4 KB + R),
# P distinct rows (pages) per channel, repeated until each channel moved `total` bytes.
#
#   python3 pages.py MODE R P [--cols C] [--fifos F] [--total BYTES]
#   MODE: stream | gather (read side scattered) | scatter (write side scattered)
import argparse
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

PAGE = 4096
LINE = 4096  # mem tile buffer per object-fifo element


def contig(n):
    rows = n // 2048
    r1 = min(rows, 512)
    return [1, rows // r1, r1, 512], [0, r1 * 512, 512, 1]


def scattered(R, P):
    """P pages, each written / read fully, one R-byte row at a time, rows of consecutive
    elements on consecutive pages. P <= 512: one transfer [slot in page, page, row].
    P > 512: one transfer per slot [page group, page, row] (the shim's 4th dimension does
    not repeat), the caller steps the offset."""
    w = R // 4
    q = PAGE // R
    if P <= 512:
        return [1, q, P, w], [0, w, PAGE // 4, 1], 1
    g = P // 512
    assert g * 512 == P
    return [1, g, 512, w], [0, 512 * PAGE // 4, PAGE // 4, 1], q


def design(mode, R, P, cols, fifos, total):
    nch = cols * fifos
    # every round moves the same P pages on both sides; stream = in address order
    span = P * PAGE
    passes = max(1, total // span)
    ssz, sst, slots = scattered(R, P)
    if mode == "stream":
        slots = 1
    rounds = passes * slots
    step = span // slots  # bytes per round on the contiguous side
    sstep = span // slots if mode == "stream" else (R if slots > 1 else 0)
    csz, cst = contig(step)
    in_span = out_span = span
    cstep = step if slots > 1 else 0
    if mode == "gather":
        isz, ist, istep, osz, ost, ostep = ssz, sst, sstep, csz, cst, cstep
    elif mode == "scatter":
        isz, ist, istep, osz, ost, ostep = csz, cst, cstep, ssz, sst, sstep
    else:
        isz, ist, istep, osz, ost, ostep = csz, cst, 0, csz, cst, 0
    in_words = nch * in_span // 4
    out_words = nch * out_span // 4
    line_ty = np.ndarray[(LINE // 4,), np.dtype[np.int32]]
    in_ty = np.ndarray[(in_words,), np.dtype[np.int32]]
    out_ty = np.ndarray[(out_words,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            ins, outs = [], []
            for c in range(cols):
                shim = tile(c, 0)
                mem = tile(c, 1)
                for f in range(fifos):
                    a = object_fifo(f"in{c}_{f}", shim, mem, 2, line_ty)
                    b = object_fifo(f"out{c}_{f}", mem, shim, 2, line_ty)
                    object_fifo_link(a, b)
                    ins.append(a)
                    outs.append(b)

            @runtime_sequence(in_ty, out_ty)
            def sequence(A, C):
                for r in range(rounds):
                    s = r % slots
                    for k in range(nch):
                        npu_dma_memcpy_nd(metadata=outs[k], bd_id=0, mem=C,
                                          offsets=[0, 0, 0, (k * out_span + s * ostep) // 4],
                                          sizes=osz, strides=ost, issue_token=True)
                        npu_dma_memcpy_nd(metadata=ins[k], bd_id=1, mem=A,
                                          offsets=[0, 0, 0, (k * in_span + s * istep) // 4],
                                          sizes=isz, strides=ist)
                    dma_wait(*outs)

    return str(ctx.module), in_words * 4, out_words * 4, passes


def chan_base(N, R, pitch):
    """Byte distance between channel regions on the scattered side: wide pitches interleave the
    channels one page apart inside one region, narrow ones get their own region."""
    return PAGE if pitch >= 16 * PAGE else N * pitch


def design_x(mode, R, N, pitch, cols, fifos, total):
    """xscatter / xgather: N rows of R bytes per channel, `pitch` bytes apart (the issue's
    layout is pitch = 64 KB + R), each pass moves every row once. Chunks of <= 512 rows."""
    nch = cols * fifos
    w = R // 4
    n1 = min(N, 512)
    chunks = N // n1
    assert chunks * n1 == N
    passes = max(1, min(total // (N * R), 128 // chunks))  # <= 128 rounds keeps aiecc fast
    rounds = passes * chunks
    cb = chan_base(N, R, pitch)
    sc_bytes = (nch - 1) * cb + N * pitch
    co_bytes = nch * N * R
    ssz, sst = [1, 1, n1, w], [0, 0, pitch // 4, 1]
    csz, cst = contig(n1 * R) if n1 * R >= 2048 else ([1, 1, 1, n1 * w], [0, 0, 0, 1])
    sc = lambda k, c: (k * cb + c * n1 * pitch) // 4
    co = lambda k, c: (k * N * R + c * n1 * R) // 4
    if mode == "xgather":
        in_b, out_b, isz, ist, ioff, osz, ost, ooff = sc_bytes, co_bytes, ssz, sst, sc, csz, cst, co
    else:
        in_b, out_b, isz, ist, ioff, osz, ost, ooff = co_bytes, sc_bytes, csz, cst, co, ssz, sst, sc
    line_ty = np.ndarray[(LINE // 4,), np.dtype[np.int32]]
    in_ty = np.ndarray[(in_b // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[(out_b // 4,), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu2)
        def device_body():
            ins, outs = [], []
            for c in range(cols):
                shim = tile(c, 0)
                mem = tile(c, 1)
                for f in range(fifos):
                    a = object_fifo(f"in{c}_{f}", shim, mem, 2, line_ty)
                    b = object_fifo(f"out{c}_{f}", mem, shim, 2, line_ty)
                    object_fifo_link(a, b)
                    ins.append(a)
                    outs.append(b)

            @runtime_sequence(in_ty, out_ty)
            def sequence(A, C):
                for r in range(rounds):
                    c = r % chunks
                    for k in range(nch):
                        npu_dma_memcpy_nd(metadata=outs[k], bd_id=0, mem=C, offsets=[0, 0, 0, ooff(k, c)],
                                          sizes=osz, strides=ost, issue_token=True)
                        npu_dma_memcpy_nd(metadata=ins[k], bd_id=1, mem=A, offsets=[0, 0, 0, ioff(k, c)],
                                          sizes=isz, strides=ist)
                    dma_wait(*outs)

    return str(ctx.module), in_b, out_b, passes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stream", "gather", "scatter"])
    p.add_argument("R", type=int)
    p.add_argument("P", type=int)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--fifos", type=int, default=2)
    p.add_argument("--total", type=int, default=1 << 20)
    a = p.parse_args()
    mod, ib, ob, passes = design(a.mode, a.R, a.P, a.cols, a.fifos, a.total)
    print(mod)
