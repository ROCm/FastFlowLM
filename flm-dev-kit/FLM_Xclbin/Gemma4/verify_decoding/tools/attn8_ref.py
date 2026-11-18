#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Reference and scorer for the 8-token attention stages (plan S6.5 second half / S6.9 / S6.10; host/test_attn8.hpp, `run.exe -t attn8`).

  attn8_ref.py gen   --stage rope|attn --layer N --ctx C --dir D [--rows 8] [--seed S]
  attn8_ref.py check --stage rope|attn --layer N --ctx C --dir D

Inputs are deterministic and realistic: q/k/v rows N(0, 2) (the norms rescale them anyway), the real q_norm/k_norm weights and the real
cos/sin of the decode host (host/decoding_layer.cpp inv_freq tables, positions C..C+7) for layer N of the shipped Bolt model.q4nx.
Everything is bf16 on the device side; the reference is fp32 on the same bf16 inputs (the kernels round to bf16 between stages).

stage rope : files (bf16 raw)  qkv_in.bin   [q heads x [t][DH]] [k heads] [v heads]  (k/v absent on skip layers)
                               table.bin    [t][cos(DH/2)|sin(DH/2)] x rows | q_norm(DH) | k_norm(DH)
             written by host   q_out.bin    [q head][t][DH]
                               kv_out.bin   [k head 0][t][DH] [k head 1] [v head 0] [v head 1]   (absent on skip layers)
"""
import argparse, os, re, sys
import numpy as np
import torch
from safetensors import safe_open

MODEL = os.environ.get('BOLT_Q4NX', 'q4nx_files/Gemma4-E4B-IT-NPU2/model.q4nx')   # the nemotron3.5-bolt:e4b package
HOSTCPP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'host', 'decoding_layer.cpp')
NH, NKV, DH_G, DH_S, NL, SHARED, PERIOD = 8, 2, 512, 256, 42, 18, 6
EPS = 1e-6


def layer_info(layer):
    swa = (layer % PERIOD) < (PERIOD - 1)
    skip = layer >= NL - SHARED
    return dict(swa=swa, skip=skip, dh=DH_S if swa else DH_G)


def bf16val(a):
    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16).float().numpy()


def bf16bits(a):
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def from_bf16bits(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def inv_freqs():
    s = open(HOSTCPP).read()
    out = {}
    for name in ('inv_freq_global', 'inv_freq_swa'):
        m = re.search(name + r'\[\d+\]\s*=\s*\{(.*?)\};', s, re.S)
        out[name] = np.array([float(x) for x in m.group(1).replace('\n', ' ').split(',') if x.strip()], np.float64)
    return out


def norm_weights(layer):
    f = safe_open(MODEL, 'pt')
    q = f.get_tensor(f'model.layers.{layer}.self_attn.q_norm.weight')
    k = f.get_tensor(f'model.layers.{layer}.self_attn.k_norm.weight')
    cv = lambda t: np.asarray(t.view(torch.int16).numpy().view(np.uint16)).copy() if t.dtype == torch.bfloat16 else bf16bits(t.float().numpy())
    return from_bf16bits(cv(q)), from_bf16bits(cv(k))


def rope_table(info, ctx, rows):
    dh = info['dh']
    fr = inv_freqs()['inv_freq_swa' if info['swa'] else 'inv_freq_global']
    t = np.zeros((rows, dh), np.float32)
    for r in range(rows):
        ang = fr[:dh // 2] * (ctx + r)
        t[r, :dh // 2] = np.cos(ang)
        t[r, dh // 2:] = np.sin(ang)
    return bf16val(t)


def rope_table_pos(info, positions):
    dh = info['dh']
    fr = inv_freqs()['inv_freq_swa' if info['swa'] else 'inv_freq_global']
    ang = np.asarray(positions, np.float64)[:, None] * fr[None, :dh // 2]
    return bf16val(np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32))


def rms_rows(x, w=None):
    d = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS)
    y = x * d
    return y if w is None else (x * w) * d


def rotate(x, tab):
    dh = x.shape[-1]
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    c, s = tab[..., :dh // 2], tab[..., dh // 2:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def make_inputs(layer, ctx, rows, seed):
    info = layer_info(layer)
    dh = info['dh']
    rng = np.random.RandomState(seed + 1000 * layer + ctx)
    q = bf16val(rng.randn(NH, rows, dh) * 2.0)
    k = bf16val(rng.randn(NKV, rows, dh) * 2.0)
    v = bf16val(rng.randn(NKV, rows, dh) * 2.0)
    qn, kn = norm_weights(layer)
    tab = rope_table(info, ctx, rows)
    return info, q, k, v, qn, kn, tab


def rope_reference(q, k, v, qn, kn, tab):
    qr = np.stack([rotate(rms_rows(q[h], qn), tab) for h in range(q.shape[0])])
    kr = np.stack([rotate(rms_rows(k[h], kn), tab) for h in range(k.shape[0])])
    vr = np.stack([rms_rows(v[h]) for h in range(v.shape[0])])
    return qr, kr, vr


ATTN_SCALE = {False: 1.0, True: 1.0}   # the E4B kernels apply no score scale (Gemma4 q/k norms make it 1)
WINDOW = 512


def make_context(info, layer, ctx, seed):
    """K/V rows of the positions < ctx as the cache holds them: k = rope(k_norm(k)), v = v_norm(v), bf16, [ctx][2][DH]"""
    dh = info['dh']
    rng = np.random.RandomState(seed + 7000 * layer + ctx)
    kpre = bf16val(rng.randn(ctx, NKV, dh) * 2.0)
    vpre = bf16val(rng.randn(ctx, NKV, dh) * 2.0)
    qn, kn = norm_weights(layer)
    if ctx == 0:
        return np.zeros((0, NKV, dh), np.float32), np.zeros((0, NKV, dh), np.float32)
    tab = rope_table_pos(info, np.arange(ctx))
    k = np.stack([bf16val(rotate(rms_rows(kpre[:, h], kn), tab)) for h in range(NKV)], 1)
    v = np.stack([bf16val(rms_rows(vpre[:, h])) for h in range(NKV)], 1)
    return k.astype(np.float32), v.astype(np.float32)


def attention_reference(info, qr, k_all, v_all, ctx, rows):
    """qr [NH][rows][dh] (bf16 values), k_all/v_all [ctx+rows][NKV][dh] (bf16 values); token t sees keys [lo_t, ctx+t]"""
    dh = info['dh']
    scale = ATTN_SCALE[info['swa']]
    o = np.zeros((rows, NH * dh), np.float64)
    for t in range(rows):
        hi = ctx + t
        lo = max(0, hi + 1 - WINDOW) if info['swa'] else 0
        for h in range(NH):
            g = h // (NH // NKV)
            sc = (k_all[lo:hi + 1, g].astype(np.float64) @ qr[h, t].astype(np.float64)) * scale
            sc -= sc.max()
            p = np.exp(sc); p /= p.sum()
            o[t, h * dh:(h + 1) * dh] = p @ v_all[lo:hi + 1, g].astype(np.float64)
    return o


def cmp(name, got, ref):
    got = got.reshape(-1, got.shape[-1]).astype(np.float64)
    ref = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
    rel = np.linalg.norm(got - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-30)
    print(f'{name:10s} rows {len(cos):4d}  cos min {cos.min():.6f} mean {cos.mean():.6f}   relL2 max {rel.max():.5f} mean {rel.mean():.5f}  '
          f'nan {int(np.isnan(got).sum())}')
    return cos.min(), rel.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gen', 'check'])
    ap.add_argument('--stage', default='rope')
    ap.add_argument('--layer', type=int, default=5)
    ap.add_argument('--ctx', type=int, default=0)
    ap.add_argument('--rows', type=int, default=8)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cos-tol', type=float, default=None, help='default: 0.997 (swa, or global ctx < 300), 0.975 for long near-uniform global attention, 0.995 when --qn-scale > 1')
    ap.add_argument('--qn-scale', type=float, default=1.0, help='scales the q_norm weights (softer/harder attention logits)')
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    info, q, k, v, qn, kn, tab = make_inputs(a.layer, a.ctx, a.rows, a.seed)
    qn = bf16val(qn * a.qn_scale)
    dh = info['dh']
    if a.cos_tol is None:
        a.cos_tol = 0.995 if a.qn_scale > 1 else (0.975 if (not info['swa'] and a.ctx >= 300) else 0.997)
    if a.stage == 'rope':
        if a.mode == 'gen':
            parts = [q.reshape(-1)] + ([] if info['skip'] else [k.reshape(-1), v.reshape(-1)])
            bf16bits(np.concatenate(parts)).tofile(os.path.join(a.dir, 'qkv_in.bin'))
            bf16bits(np.concatenate([tab.reshape(-1), qn, kn])).tofile(os.path.join(a.dir, 'table.bin'))
            qr, kr, vr = rope_reference(q, k, v, qn, kn, tab)
            np.savez(os.path.join(a.dir, 'ref_rope.npz'), q=qr, k=kr, v=vr)
            print(f'gen rope: layer {a.layer} {"swa" if info["swa"] else "global"} {"skip" if info["skip"] else "non-skip"} dh {dh} ctx {a.ctx}')
        else:
            ref = np.load(os.path.join(a.dir, 'ref_rope.npz'))
            qo = from_bf16bits(np.fromfile(os.path.join(a.dir, 'q_out.bin'), np.uint16)).reshape(NH, a.rows, dh)
            worst = cmp('q', qo, ref['q'])
            if not info['skip']:
                kv = from_bf16bits(np.fromfile(os.path.join(a.dir, 'kv_out.bin'), np.uint16)).reshape(2 * NKV, a.rows, dh)
                worst = tuple(map(min, zip(worst, cmp('k', kv[:NKV], ref['k'])))) if False else worst
                c1 = cmp('k', kv[:NKV], ref['k'])
                c2 = cmp('v', kv[NKV:], ref['v'])
                worst = (min(worst[0], c1[0], c2[0]), max(worst[1], c1[1], c2[1]))
            ok = worst[0] >= a.cos_tol
            print('RESULT', 'PASS' if ok else 'FAIL', f'min cos {worst[0]:.6f}')
            sys.exit(0 if ok else 1)
    elif a.stage == 'attn':
        kc, vc = make_context(info, a.layer, a.ctx, a.seed)
        qr, kr, vr = rope_reference(q, k, v, qn, kn, tab)
        qb, krb, vrb = bf16val(qr), bf16val(kr), bf16val(vr)
        if a.mode == 'gen':
            parts = [q.reshape(-1)] + ([] if info['skip'] else [k.reshape(-1), v.reshape(-1)])
            bf16bits(np.concatenate(parts)).tofile(os.path.join(a.dir, 'qkv_in.bin'))
            bf16bits(np.concatenate([tab.reshape(-1), qn, kn])).tofile(os.path.join(a.dir, 'table.bin'))
            # the cache rows < ctx (host places them); skip layers also hold the 8 new rows (the owner layer wrote them in the same chain)
            knew = krb.transpose(1, 0, 2); vnew = vrb.transpose(1, 0, 2)          # [rows][NKV][dh]
            k_all = np.concatenate([kc, knew], 0); v_all = np.concatenate([vc, vnew], 0)
            n_ctx_rows = a.ctx + (a.rows if info['skip'] else 0)
            bf16bits(k_all[:n_ctx_rows].reshape(n_ctx_rows, NKV * dh)).tofile(os.path.join(a.dir, 'kctx.bin'))
            bf16bits(v_all[:n_ctx_rows].reshape(n_ctx_rows, NKV * dh)).tofile(os.path.join(a.dir, 'vctx.bin'))
            o = attention_reference(info, qb, k_all, v_all, a.ctx, a.rows)
            np.savez(os.path.join(a.dir, 'ref_attn.npz'), o=o, k=krb, v=vrb, q=qb)
            print(f'gen attn: layer {a.layer} {"swa" if info["swa"] else "global"} {"skip" if info["skip"] else "non-skip"} dh {dh} ctx {a.ctx}')
        else:
            ref = np.load(os.path.join(a.dir, 'ref_attn.npz'))
            od = from_bf16bits(np.fromfile(os.path.join(a.dir, 'o_out.bin'), np.uint16)).reshape(a.rows, NH * dh)
            c0 = cmp('o', od, ref['o'])
            worst = c0
            if not info['skip']:
                kvr = from_bf16bits(np.fromfile(os.path.join(a.dir, 'kv_new.bin'), np.uint16)).reshape(2, a.rows, NKV, dh)  # [k|v][t][head][dh]
                c1 = cmp('k_new', kvr[0].transpose(1, 0, 2), ref['k'])
                c2 = cmp('v_new', kvr[1].transpose(1, 0, 2), ref['v'])
                worst = (min(c0[0], c1[0], c2[0]), max(c0[1], c1[1], c2[1]))
            ok = c0[0] >= a.cos_tol and worst[0] >= a.cos_tol
            print('RESULT', 'PASS' if ok else 'FAIL', f'o min cos {c0[0]:.6f}')
            sys.exit(0 if ok else 1)
    else:
        raise SystemExit('stage ' + a.stage + ' not implemented yet')


if __name__ == '__main__':
    main()
