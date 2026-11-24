# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Q4_1 for the mixed Q4_1 / Q8_0 NPU decode (generic_decoding_layer, QUANT_TYPE __Q41_Q80_MIX__).

The kernel reads q41_block_t (kernels/model_spec.h) packed plain column major over the whole 32-row block (PARALLEL_ROWS 32):

    bf16 scales[256 / 32][32]; bf16 mins[256 / 32][32]; uint4 qs[256 columns][32 rows]   (5120 bytes per 32x256 block)

with w ~= d q + m per 32-column group, q in 0..15. The scale and the min are quantized to bf16 *before* q is chosen (so q is the
best code for the values the kernel will actually use), then refined by least squares: with q fixed, the d and m that minimise
the group's squared error, rounded to bf16, and q again - kept only where it lowers that group's error. On the Qwen3-ASR
projections this lowers the RMS error of plain min/max Q4_1 by about 7 %.
"""
import torch

GROUP = 32
ROWS = 32
COLS = 256


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float32)


def quantize_q41(w: torch.Tensor, refine: int = 3):
    """w [N, K] -> d, m [N, K / 32] (bf16-valued float32), q [N, K] uint8 in 0..15"""
    N, K = w.shape
    assert K % GROUP == 0
    x = w.float().reshape(N, K // GROUP, GROUP)
    lo, hi = x.amin(-1), x.amax(-1)
    d = _bf16((hi - lo) / 15.0)
    m = _bf16(lo)

    def codes(d, m):
        inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
        return torch.clamp(torch.floor((x - m[..., None]) * inv[..., None] + 0.5), 0, 15)

    q = codes(d, m)
    err = ((q * d[..., None] + m[..., None] - x) ** 2).sum(-1)
    for _ in range(refine):
        qm, xm = q.mean(-1, keepdim=True), x.mean(-1, keepdim=True)
        var = ((q - qm) ** 2).sum(-1)
        cov = ((q - qm) * (x - xm)).sum(-1)
        d2 = _bf16(torch.where(var > 0, cov / torch.clamp(var, min=1e-30), d))
        m2 = _bf16(xm[..., 0] - d2 * qm[..., 0])
        q2 = codes(d2, m2)
        e2 = ((q2 * d2[..., None] + m2[..., None] - x) ** 2).sum(-1)
        better = e2 < err
        d, m = torch.where(better, d2, d), torch.where(better, m2, m)
        q = torch.where(better[..., None], q2, q)
        err = torch.where(better, e2, err)
    return d.reshape(N, -1), m.reshape(N, -1), q.reshape(N, K).to(torch.uint8)


def pack_q41(d: torch.Tensor, m: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """-> int8 [N / 32, K / 256, 5120] q41_block_t blocks; nibble byte = row 2r (low) | row 2r + 1 (high), columns in order."""
    N, K = q.shape
    assert N % ROWS == 0 and K % COLS == 0
    P, Q, gpb = N // ROWS, K // COLS, COLS // GROUP
    ds = d.reshape(P, ROWS, Q, gpb).permute(0, 2, 3, 1).contiguous().to(torch.bfloat16).view(torch.uint8).reshape(P, Q, -1)
    ms = m.reshape(P, ROWS, Q, gpb).permute(0, 2, 3, 1).contiguous().to(torch.bfloat16).view(torch.uint8).reshape(P, Q, -1)
    qq = q.reshape(P, ROWS, Q, COLS).permute(0, 2, 3, 1).contiguous().reshape(P, Q, COLS, ROWS // 2, 2)
    qb = ((qq[..., 0] & 0x0F) | ((qq[..., 1] & 0x0F) << 4)).reshape(P, Q, -1)
    return torch.cat([ds, ms, qb], dim=-1).contiguous().view(torch.int8)
