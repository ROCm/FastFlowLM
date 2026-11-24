# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Torch fp32 reference of the DFlash draft (Appendix A of dflash-bolt.md), plus a teacher-forced replay that measures
the acceptance length (AL) against a known greedy continuation.

Weights come from the llama.cpp GGUF; the target embedding table E (raw, [V,2560], shared with the draft) from the HF BF16 target.
Weight formats for the S2.1 sweep: 'bf16' | 'int8g32' | 'q41' ; `bf16_keep` = tensor-name substrings kept bf16 in 'q41'/'int8g32'.
"""
import os, sys, math, json, time
import numpy as np, torch
from gguf import GGUFReader, quantize, dequantize, GGMLQuantizationType as QT

GGUF = os.environ.get('DFLASH_GGUF', 'Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf')   # from nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash
NL, D, H, HKV, DH, FFN, BLK, MASK_ID, WIN = 5, 2560, 32, 8, 128, 5120, 8, 4, 1024
EPS, THETA, ESCALE = 1e-6, 1e6, 50.59644317626953

def _bf16(x): return x.to(torch.bfloat16).to(torch.float32)

def fake_quant(w: torch.Tensor, mode: str) -> torch.Tensor:
    """w [out,in] fp32 -> fp32 values after the storage format round trip."""
    if mode == 'bf16': return _bf16(w)
    if mode == 'int8g32':
        g = w.reshape(w.shape[0], -1, 32)
        s = g.abs().amax(-1, keepdim=True) / 127.0 + 1e-30
        return (torch.round(g / s).clamp(-127, 127) * s).reshape(w.shape)
    if mode == 'q41':
        # exactly the converter path: ggml Q4_1 quantize from the bf16 weights; scales/mins stored bf16
        a = _bf16(w).numpy()
        deq = dequantize(quantize(a, QT.Q4_1), QT.Q4_1)
        return torch.from_numpy(np.ascontiguousarray(deq)).reshape(w.shape)
    raise ValueError(mode)

def rms(x, w=None, eps=EPS):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w

def rope_neox(x, pos):  # x [n, heads, 128], pos [n]
    half = DH // 2
    inv = THETA ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * inv[None]
    c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

class DFlashRef:
    def __init__(self, E: torch.Tensor, mode='bf16', bf16_keep=(), cache_dir=None):
        self.E = E            # [V, D] fp32 raw target embedding (shared draft embedding + lm_head)
        self.mode, self.keep = mode, tuple(bf16_keep)
        cp = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cp = f"{cache_dir}/{mode}_{'+'.join(sorted(b.replace('.', '_') for b in self.keep)) or 'none'}.pt"
        if cp and os.path.exists(cp):
            self.W = torch.load(cp, weights_only=True)
        else:
            r = GGUFReader(GGUF); self.W = {}
            for t in r.tensors:
                raw = t.data.view(np.uint16).astype(np.uint32) << 16 if t.tensor_type == QT.BF16 else None
                a = (raw.view(np.float32) if raw is not None else np.array(t.data, np.float32)).reshape(tuple(t.shape.tolist()[::-1]))
                a = torch.from_numpy(np.ascontiguousarray(a))
                if a.ndim == 2:
                    m = 'bf16' if any(k in t.name for k in self.keep) else mode
                    a = fake_quant(a, m)
                else:
                    a = _bf16(a)      # norms are stored bf16
                self.W[t.name] = a
            if cp: torch.save(self.W, cp)
        self.reset()

    def reset(self):
        self.Kc = [torch.zeros(0, HKV, DH) for _ in range(NL)]
        self.Vc = [torch.zeros(0, HKV, DH) for _ in range(NL)]
        self.cpos = torch.zeros(0, dtype=torch.long)

    # ---- context injection: taps [n,5,D] (outputs of HF layers 5,11,17,23,35), positions [n]
    def inject(self, taps: torch.Tensor, pos: torch.Tensor):
        W = self.W
        c = rms(taps.reshape(taps.shape[0], -1) @ W['fc.weight'].T) * W['enc.output_norm.weight']
        for l in range(NL):
            k = (c @ W[f'blk.{l}.attn_k.weight'].T).reshape(-1, HKV, DH)
            v = (c @ W[f'blk.{l}.attn_v.weight'].T).reshape(-1, HKV, DH)
            k = rope_neox(rms(k) * W[f'blk.{l}.attn_k_norm.weight'], pos)
            self.Kc[l] = torch.cat([self.Kc[l], k]); self.Vc[l] = torch.cat([self.Vc[l], v])
        self.cpos = torch.cat([self.cpos, pos])

    def trim(self, n_keep_from_pos):  # drop ctx entries with position >= n_keep_from_pos (rollback)
        keep = self.cpos < n_keep_from_pos
        self.Kc = [k[keep] for k in self.Kc]; self.Vc = [v[keep] for v in self.Vc]; self.cpos = self.cpos[keep]

    # ---- one block: returns drafts d1..d7 (ids) and the 8 final hidden rows
    def draft(self, id_last: int, pos0: int, return_logits=False):
        W = self.W
        blk = torch.tensor([id_last] + [MASK_ID] * (BLK - 1))
        pos = pos0 + torch.arange(BLK)
        x = self.E[blk] * ESCALE
        kpos = torch.cat([self.cpos, pos])
        mask = (pos[:, None] - kpos[None, :]) < WIN                      # no causal term inside the block
        bias = torch.where(mask, 0.0, float('-inf'))
        for l in range(NL):
            h = rms(x) * W[f'blk.{l}.attn_norm.weight']
            q = (h @ W[f'blk.{l}.attn_q.weight'].T).reshape(BLK, H, DH)
            k = (h @ W[f'blk.{l}.attn_k.weight'].T).reshape(BLK, HKV, DH)
            v = (h @ W[f'blk.{l}.attn_v.weight'].T).reshape(BLK, HKV, DH)
            q = rope_neox(rms(q) * W[f'blk.{l}.attn_q_norm.weight'], pos)
            k = rope_neox(rms(k) * W[f'blk.{l}.attn_k_norm.weight'], pos)
            K = torch.cat([self.Kc[l], k]).repeat_interleave(H // HKV, 1)   # [n,32,128]
            V = torch.cat([self.Vc[l], v]).repeat_interleave(H // HKV, 1)
            s = torch.einsum('ihd,jhd->hij', q, K) / math.sqrt(DH) + bias[None]
            a = torch.einsum('hij,jhd->ihd', torch.softmax(s, -1), V).reshape(BLK, H * DH)
            x = x + a @ W[f'blk.{l}.attn_output.weight'].T
            h2 = rms(x) * W[f'blk.{l}.ffn_norm.weight']
            x = x + (torch.nn.functional.silu(h2 @ W[f'blk.{l}.ffn_gate.weight'].T) * (h2 @ W[f'blk.{l}.ffn_up.weight'].T)) @ W[f'blk.{l}.ffn_down.weight'].T
        out = rms(x) * W['output_norm.weight']
        logits = out[1:] @ self.E.T                                        # rows 1..7, no softcap for argmax
        d = logits.argmax(-1)
        return (d, logits) if return_logits else d

def replay_al(ref: DFlashRef, tokens, taps, prompt_len, max_cycles=10**9, trace=None):
    """Teacher-forced replay: tokens = prompt + greedy continuation (len T), taps [T,5,D] from the target on that sequence.
    Returns list of accepted counts k per cycle (emitted per cycle = k+1)."""
    T = len(tokens); P = prompt_len
    ref.reset()
    ref.inject(taps[:P], torch.arange(P))                    # prefill injection (prompt rows)
    pos0, ks = P, []                                          # id_last = tokens[P] (first generated token)
    while pos0 + BLK < T and len(ks) < max_cycles:
        d = ref.draft(int(tokens[pos0]), pos0).tolist()
        k = 0
        while k < BLK - 1 and d[k] == int(tokens[pos0 + 1 + k]): k += 1
        if trace is not None: trace.append(dict(pos0=pos0, id_last=int(tokens[pos0]), d=d, k=k))
        ks.append(k)
        ref.inject(taps[pos0:pos0 + k + 1], torch.arange(pos0, pos0 + k + 1))
        pos0 += k + 1
    return ks
