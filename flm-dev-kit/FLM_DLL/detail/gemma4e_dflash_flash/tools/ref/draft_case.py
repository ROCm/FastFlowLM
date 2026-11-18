#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Host-drafter test cases (S5.2): dump inputs + reference outputs, then check the C++ output.

  draft_case.py dump  <pkg_dir> <case_dir> [ctx ...]     writes case_<ctx>.bin (inputs) and case_<ctx>.ref.npz (references)
  draft_case.py check <pkg_dir> <case_dir>                    compares case_<ctx>.out.bf16 (written by test_dflash_draft) with the references

References: (q) fp32 DFlashRef on the SAME Q4NX weights as the C++ code (dequantized with the converter's unpack),
            (b) fp32 DFlashRef on the BF16 GGUF weights (what the model was trained/shipped with).
Inputs: real target taps (taps/*.pt next to REF, concatenated to reach the requested ctx), target embedding rows from E_tgt.pt.
Hidden rows compared are the rows BEFORE the final output_norm (rows 1..7 of the block); tokens go through output_norm and E_tgt^T.
"""
import glob, json, os, struct, sys
import numpy as np, torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash_ref as R

REF = os.path.dirname(os.path.abspath(__file__))
RB, CB, GS, PS = 32, 256, 32, 16


def unpack_q4nx(blob, rows, cols):                     # same as FLM_Q4NX_Converter/check_dflash_q4nx.py
    n_chunks = (rows // RB) * (cols // CB)
    b = blob.view(torch.uint8).numpy().reshape(n_chunks, -1)
    ng = CB // GS; dsz = ng * RB * 2
    d = np.ascontiguousarray(b[:, :dsz]).view(np.uint16).reshape(n_chunks, ng, RB)
    m = np.ascontiguousarray(b[:, dsz:2 * dsz]).view(np.uint16).reshape(n_chunks, ng, RB)
    f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
    d, m = f(d), f(m)
    qs = b[:, 2 * dsz:].reshape(n_chunks, RB // PS, CB, PS // 2)
    q = np.empty((n_chunks, RB // PS, CB, PS // 2, 2), np.uint8)
    q[..., 0] = qs & 0xF; q[..., 1] = qs >> 4
    q = q.reshape(n_chunks, RB // PS, CB, PS).transpose(0, 1, 3, 2).reshape(n_chunks, RB, CB)
    w = q.astype(np.float32).reshape(n_chunks, RB, ng, GS) * d.transpose(0, 2, 1)[..., None] + m.transpose(0, 2, 1)[..., None]
    w = w.reshape(n_chunks, RB, CB)
    return w.reshape(rows // RB, cols // CB, RB, CB).transpose(0, 2, 1, 3).reshape(rows, cols)


def q4nx_weights(path):
    t = load_file(path)
    W = {}
    def mat(name, rows, cols):
        x = t[name]
        if x.dtype == torch.bfloat16:
            return torch.from_numpy(np.ascontiguousarray(x.float().permute(0, 2, 1).reshape(-1, x.shape[1]).numpy()))
        return torch.from_numpy(np.ascontiguousarray(unpack_q4nx(x, rows, cols)))
    vec = lambda n: t[n].float()
    W['fc.weight'] = mat('model.fc.weight', 2560, 12800)
    W['enc.output_norm.weight'] = vec('model.hidden_norm.weight')
    W['output_norm.weight'] = vec('model.norm.weight')
    for l in range(5):
        p = f'model.layers.{l}.'
        for g, q, r, c in [('attn_q', 'self_attn.q_proj', 4096, 2560), ('attn_k', 'self_attn.k_proj', 1024, 2560),
                           ('attn_v', 'self_attn.v_proj', 1024, 2560), ('attn_output', 'self_attn.o_proj', 2560, 4096),
                           ('ffn_gate', 'mlp.gate_proj', 5120, 2560), ('ffn_up', 'mlp.up_proj', 5120, 2560),
                           ('ffn_down', 'mlp.down_proj', 2560, 5120)]:
            W[f'blk.{l}.{g}.weight'] = mat(p + q + '.weight', r, c)
        W[f'blk.{l}.attn_norm.weight'] = vec(p + 'input_layernorm.weight')
        W[f'blk.{l}.ffn_norm.weight'] = vec(p + 'post_attention_layernorm.weight')
        W[f'blk.{l}.attn_q_norm.weight'] = vec(p + 'self_attn.q_norm.weight')
        W[f'blk.{l}.attn_k_norm.weight'] = vec(p + 'self_attn.k_norm.weight')
    return W


def ref_from_W(E, W):
    r = R.DFlashRef.__new__(R.DFlashRef)
    r.E, r.mode, r.keep, r.W = E, 'q4nx', (), W
    r.reset()
    return r


def ref_draft(r, id_last, pos0, emb_bf16):
    """DFlashRef.draft with the embedding rows given (bf16-rounded x scale) and the pre-norm hidden returned."""
    W = r.W; BLK, NL, H, HKV, DH = R.BLK, R.NL, R.H, R.HKV, R.DH
    pos = pos0 + torch.arange(BLK)
    x = emb_bf16.float()
    kpos = torch.cat([r.cpos, pos])
    bias = torch.where((pos[:, None] - kpos[None, :]) < R.WIN, 0.0, float('-inf'))
    for l in range(NL):
        h = R.rms(x) * W[f'blk.{l}.attn_norm.weight']
        q = (h @ W[f'blk.{l}.attn_q.weight'].T).reshape(BLK, H, DH)
        k = (h @ W[f'blk.{l}.attn_k.weight'].T).reshape(BLK, HKV, DH)
        v = (h @ W[f'blk.{l}.attn_v.weight'].T).reshape(BLK, HKV, DH)
        q = R.rope_neox(R.rms(q) * W[f'blk.{l}.attn_q_norm.weight'], pos)
        k = R.rope_neox(R.rms(k) * W[f'blk.{l}.attn_k_norm.weight'], pos)
        K = torch.cat([r.Kc[l], k]).repeat_interleave(H // HKV, 1)
        V = torch.cat([r.Vc[l], v]).repeat_interleave(H // HKV, 1)
        s = torch.einsum('ihd,jhd->hij', q, K) / R.math.sqrt(DH) + bias[None]
        a = torch.einsum('hij,jhd->ihd', torch.softmax(s, -1), V).reshape(BLK, H * DH)
        x = x + a @ W[f'blk.{l}.attn_output.weight'].T
        h2 = R.rms(x) * W[f'blk.{l}.ffn_norm.weight']
        x = x + (torch.nn.functional.silu(h2 @ W[f'blk.{l}.ffn_gate.weight'].T) * (h2 @ W[f'blk.{l}.ffn_up.weight'].T)) @ W[f'blk.{l}.ffn_down.weight'].T
    return x[1:]                                          # pre-norm hidden rows 1..7


def tokens_of(E, W, hid):
    out = R.rms(hid.float()) * W['output_norm.weight']
    return (out @ E.T).argmax(-1)


def load_taps(n):
    rows = []
    for f in sorted(glob.glob(f'{REF}/taps/*.pt'), key=lambda s: int(os.path.basename(s).split('.')[0])):
        t = torch.load(f, weights_only=True)['taps']
        rows.append(t)
        if sum(x.shape[0] for x in rows) >= n: break
    t = torch.cat(rows)[:n] if rows else torch.zeros(0, 5, 2560)
    assert t.shape[0] >= n, f'only {t.shape[0]} tap rows available'
    return t.to(torch.bfloat16)


def chunks_for(ctx):
    out, left, i = [], ctx, 0
    pat = [8, 3, 1, 5, 4, 7, 2, 6]
    while left > 0:
        c = pat[i % 8] if ctx < 400 or i < 24 else 100     # long contexts: a few small chunks, then big ones
        c = min(c, left); out.append(c); left -= c; i += 1
    return out


def dump(pkg, cdir, ctxs):
    os.makedirs(cdir, exist_ok=True)
    E = torch.load(f'{REF}/E_tgt.pt', weights_only=True).to(torch.float32)
    Wq = q4nx_weights(f'{pkg}/dflash_weight.q4nx')
    rq = ref_from_W(E, Wq)
    rb = R.DFlashRef(E, 'bf16', cache_dir=f'{REF}/wcache')
    gen = torch.Generator().manual_seed(7)
    for ctx in ctxs:
        taps = load_taps(ctx) if ctx else torch.zeros(0, 5, 2560, dtype=torch.bfloat16)
        id_last = int(torch.randint(1000, 200000, (1,), generator=gen))
        blk = torch.tensor([id_last] + [R.MASK_ID] * 7)
        emb = (E[blk] * R.ESCALE).to(torch.bfloat16)
        ch = chunks_for(ctx)
        refs = {}
        for name, r in (('q', rq), ('b', rb)):
            r.reset(); p = 0
            for c in ch:
                r.inject(taps[p:p + c].float(), torch.arange(p, p + c)); p += c
            hid = ref_draft(r, id_last, ctx, emb)
            refs[name + '_hidden'] = hid.numpy(); refs[name + '_tokens'] = tokens_of(E, r.W, hid).numpy()
        np.savez(f'{cdir}/case_{ctx}.ref.npz', **refs)
        with open(f'{cdir}/case_{ctx}.bin', 'wb') as f:
            f.write(struct.pack('<iii', ctx, id_last, len(ch)) + struct.pack(f'<{len(ch)}i', *ch))
            f.write(taps.contiguous().view(torch.int16).numpy().tobytes())
            f.write(emb.contiguous().view(torch.int16).numpy().tobytes())
        print(f'case ctx={ctx} chunks={len(ch)} q-tokens={refs["q_tokens"].tolist()} b-tokens={refs["b_tokens"].tolist()}', flush=True)


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


def check(pkg, cdir):
    E = torch.load(f'{REF}/E_tgt.pt', weights_only=True).to(torch.float32)
    w = load_file(f'{pkg}/dflash_weight.q4nx')['model.norm.weight'].float()
    bad = 0
    files = glob.glob(f'{cdir}/case_*.out.bf16')
    for f in sorted(files, key=lambda s: int(os.path.basename(s).split('_')[1].split('.')[0])):
        ctx = int(os.path.basename(f).split('_')[1].split('.')[0])
        ref = np.load(f'{cdir}/case_{ctx}.ref.npz')
        out = torch.from_numpy(np.fromfile(f, np.int16).reshape(7, 2560)).view(torch.bfloat16).float()
        cq = cos(out.numpy(), ref['q_hidden']); cb = cos(out.numpy(), ref['b_hidden'])
        tok = ((R.rms(out) * w) @ E.T).argmax(-1).numpy()
        tq = float((tok == ref['q_tokens']).mean()); tb = float((tok == ref['b_tokens']).mean())
        ok = cq.min() >= 0.999
        bad += (not ok)
        print(f'ctx={ctx:5d} cos(C++ vs fp32 same-Q4) min={cq.min():.6f}  cos(C++ vs BF16 gguf) min={cb.min():.4f}  '
              f'tokens==q {tq:.2f}  tokens==bf16 {tb:.2f}  {"OK" if ok else "FAIL"}')
    sys.exit(1 if bad or not files else 0)


if __name__ == '__main__':
    if sys.argv[1] == 'dump':
        dump(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]] or [0, 5, 300, 1500, 2300])
    else:
        check(sys.argv[2], sys.argv[3])
