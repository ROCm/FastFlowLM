#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Reference, input generator and scorer for the 8-token PLE stage (host/test_ple8.hpp, `run.exe -t ple8`).

  ple8_ref.py gen   --layer N --dir D [--seed S] [--hscale R]   writes D/x_in.bin (bf16 [h 8x2560 | pli 8x256 | w_post 2560 | scalar, pad 32]) and D/ref_*.npy
  ple8_ref.py check --layer N --dir D [--cos 0.9995] [--rel 0.03]  scores D/y.bin written by the host

The stage computes, per token t (Gemma-4 per layer input epilogue of one decoder layer):
    gate = gelu_tanh(W_gate . h_t) ; u = gate * pli_t ; up = W_up . u ; out_t = (h_t + rmsnorm(up) * w_post_pli) * layer_scalar
with the REAL Bolt weights (model.q4nx): W_gate = layers.N.inp_gate.weight (256 x 2560), W_up = layers.N.per_layer_projection.weight (2560 x 256),
both stored [row block][k][32 rows]; w_post_pli = layers.N.post_layernorm.weight; layer_scalar = layers.N.layer_output_scale.weight.

HOST pli CONTRACT (what the engine must hand the verify xclbin per layer and token; `pli_row` below is the reference implementation):
    x0      = bf16 token embedding row (model.embed_tokens: int8 row * f32 scale per 32 group, rounded to bf16)  [2560]
    x0_pl   = bf16 per-layer embedding row slice  (model.per_layer_token_embd row, columns N*256 .. N*256+255, same int8/scale format)  [256]
    t       = bf16( W_plmp . x0 )          W_plmp = model.per_layer_model_proj.weight_layerN (256 x 2560, [row block][k][32] layout), fp32 accumulate
    t       = bf16( t * 0.01976423537605237 )                 (PER_LAYER_INPUT_SCALE = 1/sqrt(2560))
    t       = bf16( t * rsqrt(mean(t^2) + 1e-6) * w_pn )      w_pn = model.per_layer_proj_norm.weight [256]
    t       = bf16( t + x0_pl )
    pli     = bf16( t * 0.7071067811865476 )                  (PER_LAYER_MODEL_PROJECTION_SCALE = 1/sqrt(2))
(the same operations, in the same order and with the same bf16 roundings, as kernels/proj_layer_embedding.cc of the decode design.)
h rows of the test: scaled embedding rows plus a few outlier channels (a mid-network residual stream has large per channel outliers).
"""
import argparse, json, os, sys
import numpy as np
import torch
from safetensors import safe_open

MODEL = os.environ.get('BOLT_Q4NX', 'q4nx_files/Gemma4-E4B-IT-NPU2/model.q4nx')   # the nemotron3.5-bolt:e4b package
D, P, T, GS = 2560, 256, 8, 32
S1, S2 = 0.01976423537605237, 0.7071067811865476


def bf(x):
    return torch.as_tensor(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()


def bf_bits(x):
    return torch.as_tensor(np.ascontiguousarray(x, np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def from_bits(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def deq_row(f, name, idx, lo=0, hi=None):
    q = f.get_slice(name)[idx:idx + 1].numpy().astype(np.float32)[0]
    s = f.get_slice(name + '.scale')[idx:idx + 1].numpy()[0]
    full = (q.reshape(-1, GS) * s[:, None]).reshape(-1)
    return bf(full[lo:hi])


def unblock(a, rows, cols):
    """[row block][k][32] -> [rows][cols] (kernels/bf16_proj.h layout)"""
    a = a.float().numpy().reshape(rows // 32, cols, 32)
    return np.ascontiguousarray(a.transpose(0, 2, 1).reshape(rows, cols))


def weights(layer):
    with safe_open(MODEL, 'pt') as f:
        Wg = unblock(f.get_tensor(f'model.layers.{layer}.inp_gate.weight'), P, D)
        Wu = unblock(f.get_tensor(f'model.layers.{layer}.per_layer_projection.weight'), D, P)
        wpost = f.get_tensor(f'model.layers.{layer}.post_layernorm.weight').float().numpy()
        scalar = float(f.get_tensor(f'model.layers.{layer}.layer_output_scale.weight').float().numpy().reshape(-1)[0])
        Wp = unblock(f.get_tensor(f'model.per_layer_model_proj.weight_layer{layer}'), P, D)
        wpn = f.get_tensor('model.per_layer_proj_norm.weight').float().numpy()
    return Wg, Wu, wpost, scalar, Wp, wpn


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def pli_row(layer, tok, f, Wp, wpn):
    x0 = deq_row(f, 'model.embed_tokens.weight', tok)
    x0pl = deq_row(f, 'model.per_layer_token_embd.weight', tok, layer * P, (layer + 1) * P)
    t = bf(Wp @ x0)
    t = bf(t * S1)
    t = bf(t * (1.0 / np.sqrt(np.mean(t * t) + 1e-6)) * wpn)
    t = bf(t + x0pl)
    return bf(t * S2), x0


def stage_ref(h, pli, Wg, Wu, wpost, scalar):
    g = gelu_tanh(h @ Wg.T)
    u = g * pli
    up = u @ Wu.T
    n = up / np.sqrt(np.mean(up * up, axis=1, keepdims=True) + 1e-6) * wpost
    return (h + n) * scalar, g, up


def gen(a):
    os.makedirs(a.dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    toks = rng.integers(2000, 200000, T)
    Wg, Wu, wpost, scalar, Wp, wpn = weights(a.layer)
    pli = np.zeros((T, P), np.float32); h = np.zeros((T, D), np.float32)
    with safe_open(MODEL, 'pt') as f:
        for t, tok in enumerate(toks):
            pli[t], x0 = pli_row(a.layer, int(tok), f, Wp, wpn)
            hh = x0 * (a.hscale / max(float(np.sqrt(np.mean(x0 * x0))), 1e-6))
            for c, v in ((7, 40.0), (301, -30.0), (1999, 55.0)):
                hh[c] = v * (1 if t % 2 == 0 else -1)
            h[t] = bf(hh)
    ref, g, up = stage_ref(h, pli, Wg, Wu, bf(wpost), bf(scalar))
    wc = np.zeros(D + 32, np.float32); wc[:D] = wpost; wc[D] = scalar
    x_in = np.concatenate([h.reshape(-1), pli.reshape(-1), wc])
    bf_bits(x_in).tofile(f'{a.dir}/x_in.bin')
    np.save(f'{a.dir}/ref_out.npy', ref); np.save(f'{a.dir}/ref_gate.npy', g); np.save(f'{a.dir}/ref_up.npy', up); np.save(f'{a.dir}/ref_u.npy', g * pli)
    json.dump(dict(layer=a.layer, tokens=[int(t) for t in toks], seed=a.seed, hscale=a.hscale, scalar=scalar,
                   rms_h=float(np.sqrt(np.mean(h * h))), rms_ref=float(np.sqrt(np.mean(ref * ref)))), open(f'{a.dir}/meta.json', 'w'))
    print(f'gen: layer {a.layer} tokens {list(map(int, toks))} rms(h) {np.sqrt(np.mean(h*h)):.2f} rms(out) {np.sqrt(np.mean(ref*ref)):.2f} scalar {scalar:.4f}')


def check(a):
    cols = {'gate': P, 'up': D, 'full': D}[a.stage]
    ref = np.load(f'{a.dir}/ref_{ {"gate": "u", "up": "up", "full": "out"}[a.stage] }.npy')
    y = from_bits(np.fromfile(f'{a.dir}/y_{a.stage}.bin', np.uint16, T * cols)).reshape(T, cols)
    ok = True
    worst_cos, worst_rel = 1.0, 0.0
    for t in range(T):
        r, d = ref[t], y[t]
        cos = float(np.dot(r, d) / (np.linalg.norm(r) * np.linalg.norm(d) + 1e-30))
        rel = float(np.linalg.norm(d - r) / (np.linalg.norm(r) + 1e-30))
        mx = float(np.max(np.abs(d - r)))
        worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
        bad = not (cos >= a.cos and rel <= a.rel) or not np.isfinite(d).all()
        ok &= not bad
        print(f'  token {t}: cos {cos:.6f} relL2 {rel:.5f} maxabs {mx:.4f} {"FAIL" if bad else "ok"}')
    print(f'ple8 {a.stage} layer {a.layer}: worst cos {worst_cos:.6f} worst relL2 {worst_rel:.5f} -> {"PASS" if ok else "FAIL"} (gate cos >= {a.cos}, relL2 <= {a.rel})')
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['gen', 'check'])
    ap.add_argument('--layer', type=int, default=0); ap.add_argument('--dir', default='ple8_data'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--stage', default='full', choices=['gate', 'up', 'full']); ap.add_argument('--hscale', type=float, default=8.0); ap.add_argument('--cos', type=float, default=0.9995); ap.add_argument('--rel', type=float, default=0.03)
    a = ap.parse_args()
    sys.exit(gen(a) or 0 if a.cmd == 'gen' else check(a))
