#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Reference for the engine's DFlash2 drafter, from the z-lab/dflash model.py (DFlash2DraftModel), fp32 on CPU.

Replays an engine dump (FLM_DFLASH_DUMP=<dir>, files c%04d_*) cycle by cycle:
  ctx rows (bf16 taps), positions, token indices, noise embeddings  ->  final pre-norm hidden [8,5120]
and compares with the engine's h_pre (cosine, relative L2). With --sel it also reruns the candidate selector on the
engine's own candidate ids/values and compares the drafted path.

  dflash2_ref.py <dump_dir> [--weights hf|q4k] [--cycles N]
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from safetensors import safe_open

HF = os.environ.get("DFLASH2_HF", "model.safetensors")            # the z-lab checkpoint
PKG = os.environ.get("DFLASH2_PKG", "dflash_weight.q4nx")        # the converted package (for --weights q4k)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../FLM_Q4NX_Converter"))
D, H, KVH, HD, INTER, NL, WIN, EPS, THETA = 5120, 32, 8, 128, 17408, 5, 2048, 1e-6, 1e7
T = 5
G = 16


CACHE = os.environ.get("DFLASH2_Q4K_CACHE", "q4k_dequant.pt")   # the dequantized weights, saved once (3.5 GB)


def load(kind):
    f = safe_open(HF, "pt")
    W = {k: f.get_tensor(k).float() for k in f.keys() if "codebook" not in k}
    cb = {k: f.get_tensor(k) for k in f.keys() if "codebook" in k}
    if kind == "q4k" and os.path.exists(CACHE):
        W = {k: v.float() for k, v in torch.load(CACHE, weights_only=True).items()}
    elif kind == "q4k":
        import check_dflash2_q4nx as check_q4k
        pkg = safe_open(PKG, "pt")
        for k in list(W):
            if W[k].dim() == 2 and (k == "fc.weight" or k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                                                     "o_proj.weight", "gate_proj.weight", "up_proj.weight",
                                                                     "down_proj.weight"))):
                r, c = W[k].shape
                W[k] = torch.from_numpy(check_q4k.dequant_q4k(pkg.get_tensor("model." + k), r, c)).float()
        torch.save({k: v.to(torch.bfloat16) if v.dim() == 2 and v.shape[0] > 1000 else v for k, v in W.items()}, CACHE)
    return W, cb


def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):  # x [n, heads, 128]; pos [n]
    inv = THETA ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = pos.double()[:, None] * inv[None]
    c, s = ang.cos().float(), ang.sin().float()
    c = torch.cat([c, c], -1)[:, None]
    s = torch.cat([s, s], -1)[:, None]
    h = HD // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * c + rot * s


def conv(x, dyn, base):  # x [L,D], dyn [L,2,G'] (the K=2 taps of one half), base [2,D]
    L = x.shape[0]
    out = torch.zeros_like(x)
    xb = x.view(L, D // G, G)
    for o in range(2):
        sh = xb if o == 0 else torch.cat([torch.zeros_like(xb[:1]), xb[:-1]], 0)
        out = out + ((base[o].view(1, D // G, G) + dyn[:, o, :, None]) * sh).view(L, D)
    return out


class Draft:
    def __init__(self, W):
        self.W = W

    def prep(self, ctx_feats):
        W = self.W
        return rms(ctx_feats @ W["fc.weight"].T, W["hidden_norm.weight"]) if len(ctx_feats) else ctx_feats.new_zeros(0, D)

    def layer(self, l, h, th, ctx_pos, ctx_idx, anchor_idx, anchor_pos):
        W = self.W
        n = th.shape[0]
        pos_all = torch.cat([ctx_pos, anchor_pos + torch.arange(8)])
        p = f"layers.{l}."
        res = h
        x = rms(h, W[p + "input_layernorm.weight"])
        dyn = (x @ W[p + "attention_conv.kernel_projection.weight"].T).view(8, 2, 2, D // G)
        xc = conv(x, dyn[:, 0], W[p + "attention_conv.base_kernel"][0])
        q = (xc @ W[p + "self_attn.q_proj.weight"].T).view(8, H, HD)
        q = rms(q, W[p + "self_attn.q_norm.weight"])
        kc = th @ W[p + "self_attn.k_proj.weight"].T
        vc = th @ W[p + "self_attn.v_proj.weight"].T
        kn = xc @ W[p + "self_attn.k_proj.weight"].T
        vn = xc @ W[p + "self_attn.v_proj.weight"].T
        k = rms(torch.cat([kc, kn], 0).view(n + 8, KVH, HD), W[p + "self_attn.k_norm.weight"])
        v = torch.cat([vc, vn], 0).view(n + 8, KVH, HD)
        q = rope(q, pos_all[n:])
        k = rope(k, pos_all)
        qi = anchor_idx + torch.arange(8)
        vis = torch.ones(8, n + 8, dtype=torch.bool)
        if n:
            vis[:, :n] = ctx_idx[None, :] > (qi[:, None] - WIN)
        kk = k.repeat_interleave(H // KVH, dim=1)
        vv = v.repeat_interleave(H // KVH, dim=1)
        sc = torch.einsum("thd,shd->hts", q, kk) / math.sqrt(HD)
        sc = sc.masked_fill(~vis[None], float("-inf"))
        a = torch.softmax(sc, -1)
        o = torch.einsum("hts,shd->thd", a, vv).reshape(8, H * HD)
        o = o @ W[p + "self_attn.o_proj.weight"].T
        o = conv(o, dyn[:, 1], W[p + "attention_conv.base_kernel"][1])
        h = res + o
        self.h_mid = h
        res = h
        x = rms(h, W[p + "post_attention_layernorm.weight"])
        dyn = (x @ W[p + "mlp_conv.kernel_projection.weight"].T).view(8, 2, 2, D // G)
        xc = conv(x, dyn[:, 0], W[p + "mlp_conv.base_kernel"][0])
        g = torch.nn.functional.silu(xc @ W[p + "mlp.gate_proj.weight"].T) * (xc @ W[p + "mlp.up_proj.weight"].T)
        m = g @ W[p + "mlp.down_proj.weight"].T
        self.m_raw = m
        m = conv(m, dyn[:, 1], W[p + "mlp_conv.base_kernel"][1])
        return res + m

    def forward(self, ctx_feats, ctx_pos, ctx_idx, noise, anchor_idx, anchor_pos):
        th = self.prep(ctx_feats)
        self.th = th
        self.hs = []
        h = noise.clone()
        for l in range(NL):
            h = self.layer(l, h, th, ctx_pos, ctx_idx, anchor_idx, anchor_pos)
            self.hs.append(h)
        return h


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--weights", default="hf")
    ap.add_argument("--cycles", type=int, default=1000)
    ap.add_argument("--sel", action="store_true")
    a = ap.parse_args()
    W, cb = load(a.weights)
    dr = Draft(W)
    ctx_f = torch.zeros(0, T * D)
    ctx_pos = torch.zeros(0, dtype=torch.long)
    ctx_idx = torch.zeros(0, dtype=torch.long)
    metas = sorted(glob.glob(os.path.join(a.dump, "c*_meta.i32")))
    same = tot = 0
    for mp in metas[:a.cycles]:
        c = os.path.basename(mp)[:5]
        anchor, anchor_idx, anchor_pos, k = np.fromfile(mp, np.int32)
        cp = os.path.join(a.dump, f"{c}_ctx.u16")
        if os.path.exists(cp):
            u = np.fromfile(cp, np.uint16).astype(np.uint32) << 16
            f = torch.from_numpy(u.view(np.float32)).view(-1, T * D)
            ctx_f = torch.cat([ctx_f, f])
            ctx_pos = torch.cat([ctx_pos, torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_ctx_pos.i32"), np.int32)).long()])
            ctx_idx = torch.cat([ctx_idx, torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_ctx_idx.i32"), np.int32)).long()])
        keep = ctx_idx > anchor_idx - WIN
        ctx_f, ctx_pos, ctx_idx = ctx_f[keep], ctx_pos[keep], ctx_idx[keep]
        noise = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_noise_emb.f32"), np.float32)).view(8, D)
        eng = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_h_pre.f32"), np.float32)).view(8, D)
        ref = dr.forward(ctx_f, ctx_pos, ctx_idx, noise, int(anchor_idx), int(anchor_pos))
        if os.environ.get("LAYERS"):
            for l in range(NL):
                p = os.path.join(a.dump, f"{c}_h_l{l}.f32")
                if os.path.exists(p):
                    e = torch.from_numpy(np.fromfile(p, np.float32)).view(8, D)
                    print(f"   layer {l}: cos", " ".join(f"{cos(dr.hs[l][t], e[t]):.3f}" for t in range(8)),
                          " relL2", " ".join(f"{float((dr.hs[l][t]-e[t]).norm()/dr.hs[l][t].norm()):.2f}" for t in range(8)))
            p = os.path.join(a.dump, f"{c}_ctx_feat.f32")
            if os.path.exists(p):
                e = torch.from_numpy(np.fromfile(p, np.float32)).view(-1, D)
                r = dr.th[:len(e)]
                print("   ctx_feat cos", " ".join(f"{cos(r[t], e[t]):.4f}" for t in range(min(8, len(e)))), "relL2",
                      f"{float((r-e).norm()/r.norm()):.3f}")
        if os.environ.get("ISO"):
            for l in range(1, NL):
                ein = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_h_l{l-1}.f32"), np.float32)).view(8, D)
                eout = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_h_l{l}.f32"), np.float32)).view(8, D)
                th = dr.th
                r = dr.layer(l, ein, th, ctx_pos, ctx_idx, int(anchor_idx), int(anchor_pos))
                print(f"   ISO layer {l} (engine input): relL2", " ".join(f"{float((r[t]-eout[t]).norm()/r[t].norm()):.3f}" for t in range(8)),
                      " |h_in| max", f"{float(ein.abs().max()):.1f}", "|res_out| max", f"{float(r.abs().max()):.1f}",
                      " |m_raw| max", f"{float(dr.m_raw.abs().max()):.1f}")
        cs = [cos(ref[t], eng[t]) for t in range(8)]
        rl = [float((ref[t] - eng[t]).norm() / ref[t].norm()) for t in range(8)]
        if os.environ.get("ROWS"):
            print("   row cos", " ".join(f"{x:.3f}" for x in cs), " relL2", " ".join(f"{x:.2f}" for x in rl))
        line = f"{c} ctx={len(ctx_f):5d} cos min {min(cs):.4f} mean {sum(cs)/8:.4f}  relL2 mean {sum(rl)/8:.3f}"
        if a.sel:
            hn = rms(ref, W["norm.weight"])
            proj = hn @ W["candidate_selector.hidden_projection.weight"].T
            pred = cb["candidate_selector.predecessor_codebook"].float()
            succ = cb["candidate_selector.successor_codebook"].float()
            prev = int(anchor)
            path = []
            for j in range(1, int(k) + 1):
                ids = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_cand{j}.i32"), np.int32)).long()
                vals = torch.from_numpy(np.fromfile(os.path.join(a.dump, f"{c}_candv{j}.f32"), np.float32))
                sc = vals + (succ[ids] @ (pred[prev] * proj[j]))
                prev = int(ids[sc.argmax()])
                path.append(prev)
            drafts = np.fromfile(os.path.join(a.dump, f"{c}_drafts.i32"), np.int32).tolist()
            same += sum(1 for x, y in zip(path, drafts) if x == y); tot += len(drafts)
            line += f"  path match {path == drafts}"
        print(line, flush=True)
    if tot:
        print(f"selector token agreement {same}/{tot}")


if __name__ == "__main__":
    main()
