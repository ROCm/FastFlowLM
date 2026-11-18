# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""CPU check of the Qwen3.5-27B GDN head reorders (any ratio Hv/Hqk), before any NPU run.

Proves, for the chosen delta-net layers, that
  1. llama.cpp's BF16 GGUF stores the value heads TILED, i.e. equals the HF tensor under the
     permutation `tiled = grouped[idx]` (exact),
  2. the converter's model.q4nx (decoded back to float) equals the GGUF's Q4_K/Q8_0 values in HF
     (grouped) order, so q4nx_value_head h == HF value head h: qkv v rows, gate rows, ssm_out columns,
     alpha/beta rows (+ zero padding to a multiple of 32), conv1d, ssm_a, ssm_dt;
     the permuted comparison is tight, the un-permuted (wrong) one is not,
  3. the bf16 alpha/beta copies equal in_proj_a / in_proj_b bit for bit.

    python check_qwen35_27b_perm.py --hf <hf dir with unrenamed language_model.* shards> \
        --bf16-gguf bf16.gguf --q4k-gguf q4k.gguf --q4nx model.q4nx --layers 0 1 2
Exit status 0 only if everything passes.
"""
import argparse, json, os, sys
import numpy as np
import torch
from gguf import GGUFReader, dequantize
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_minicpm_v_q4nx import dequant_q4nx  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def tiled_index(num_k, per_k, head_dim):
    n = num_k * per_k * head_dim
    return torch.arange(n).reshape(num_k, per_k, head_dim).permute(1, 0, 2).contiguous().reshape(n)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--bf16-gguf", required=True)
    ap.add_argument("--q4k-gguf", required=True)
    ap.add_argument("--q4nx", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--full-layers", type=int, nargs="*", default=[3, 63])
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.hf, "config.json")))["text_config"]
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    hk, hv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    R = nv // nk
    qk_rows = 2 * nk * hk
    print(f"Hqk {nk}, Hv {nv}, ratio {R}, head dim {hv}, qk rows {qk_rows}")
    idx = tiled_index(nk, R, hv)          # tiled = grouped[idx]
    inv = torch.argsort(idx)              # grouped = tiled[inv]
    idxh = tiled_index(nk, R, 1)
    invh = torch.argsort(idxh)

    wm = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def hf(name):
        fn = wm[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(a.hf, fn), "pt")
        return handles[fn].get_tensor(name)

    gb = {t.name: t for t in GGUFReader(a.bf16_gguf).tensors}
    gq = {t.name: t for t in GGUFReader(a.q4k_gguf).tensors}

    def gget(tab, name):
        t = tab[name]
        return torch.from_numpy(np.asarray(dequantize(t.data, t.tensor_type)).astype(np.float32).reshape(t.shape[::-1]))

    qf = safe_open(a.q4nx, "pt")

    def qdeq(name):
        t = qf.get_tensor(name)
        assert t.dtype == torch.int8, (name, t.dtype)
        return torch.from_numpy(dequant_q4nx(t.numpy()))

    for L in a.layers:
        pre = f"language_model.layers.{L}.linear_attn."
        qp = f"model.layers.{L}.linear_attn."
        print(f"--- layer {L}")
        if f"language_model.layers.{L}.linear_attn.in_proj_qkv.weight" not in wm:
            check(False, f"layer {L} is not a delta-net layer"); continue
        # ---- 1. llama.cpp layout (BF16 GGUF vs HF, exact)
        h_qkv, h_z, h_o = hf(pre + "in_proj_qkv.weight").float(), hf(pre + "in_proj_z.weight").float(), hf(pre + "out_proj.weight").float()
        h_a, h_b = hf(pre + "in_proj_a.weight"), hf(pre + "in_proj_b.weight")
        g = gget(gb, f"blk.{L}.attn_qkv.weight")
        check(torch.equal(g[:qk_rows], h_qkv[:qk_rows]) and torch.equal(g[qk_rows:], h_qkv[qk_rows:][idx]), "gguf attn_qkv: q,k rows untouched, v rows tiled == HF[idx]")
        check(torch.equal(gget(gb, f"blk.{L}.attn_gate.weight"), h_z[idx]), "gguf attn_gate rows tiled == HF[idx]")
        check(torch.equal(gget(gb, f"blk.{L}.ssm_out.weight"), h_o[:, idx]), "gguf ssm_out columns tiled == HF[:, idx]")
        check(torch.equal(gget(gb, f"blk.{L}.ssm_alpha.weight"), h_a.float()[idxh]) and torch.equal(gget(gb, f"blk.{L}.ssm_beta.weight"), h_b.float()[idxh]), "gguf ssm_alpha/beta rows tiled == HF[idxh]")
        conv = hf(pre + "conv1d.weight").float().reshape(-1, cfg["linear_conv_kernel_dim"])
        gc = gget(gb, f"blk.{L}.ssm_conv1d.weight")
        check(torch.equal(gc[:qk_rows], conv[:qk_rows]) and torch.equal(gc[qk_rows:], conv[qk_rows:][idx]), "gguf conv1d: v channels tiled == HF[idx]")
        a_hf = -torch.exp(hf(pre + "A_log").float())
        check(torch.allclose(gget(gb, f"blk.{L}.ssm_a"), a_hf[idxh], rtol=1e-6), "gguf ssm_a == -exp(A_log)[idxh]")
        dt = hf(pre + "dt_bias").float()
        check(torch.equal(gget(gb, f"blk.{L}.ssm_dt.bias"), dt[idxh]), "gguf ssm_dt == dt_bias[idxh]")

        # ---- 2. q4nx (decoded) in HF order; Q-ref = the quantized GGUF values, permuted back
        def cmp(name, got, q_ref_hf, q_ref_tiled, hf_orig, tol=1e-2, extra=""):
            e_ok, e_bad, e_hf = rel(got.numpy(), q_ref_hf.numpy()), rel(got.numpy(), q_ref_tiled.numpy()), rel(got.numpy(), hf_orig.numpy())
            check(e_ok < tol and e_bad > 50 * e_ok, f"q4nx {name}: vs Q-ref(HF order) {e_ok:.2e}, vs un-permuted {e_bad:.2e}, vs HF original {e_hf:.3f}{extra}")

        q = gget(gq, f"blk.{L}.attn_qkv.weight")
        qq = torch.cat([q[:qk_rows], q[qk_rows:][inv]])
        cmp("qkv_proj", qdeq(qp + "qkv_proj.weight"), qq, q, h_qkv)
        z = gget(gq, f"blk.{L}.attn_gate.weight")
        cmp("gate_proj", qdeq(f"model.layers.{L}.self_attn.gate_proj.weight"), z[inv], z, h_z)
        o = gget(gq, f"blk.{L}.ssm_out.weight")
        cmp("ssm_out_proj", qdeq(qp + "ssm_out_proj.weight"), o[:, inv], o, h_o)
        for nm, gname, hfw in (("ssm_alpha_proj", "ssm_alpha", h_a), ("ssm_beta_proj", "ssm_beta", h_b)):
            d = qdeq(qp + nm + ".weight")
            padded_ok = d.shape[0] % 32 == 0 and d.shape[0] >= nv and bool((d[nv:] == 0).all())
            gq_ = gget(gq, f"blk.{L}.{gname}.weight")
            cmp(nm, d[:nv], gq_[invh], gq_, hfw.float(), tol=1e-2, extra=f", rows {d.shape[0]}")
            check(padded_ok, f"q4nx {nm}: {d.shape[0]} rows (multiple of 32, >= {nv}), pad rows zero")
            bf = qf.get_tensor(qp + nm + ".bf16.weight")
            check(bf.dtype == torch.bfloat16 and bf.shape == hfw.shape and torch.equal(bf, hfw), f"q4nx {nm}.bf16 == HF in_proj bit-exact, shape {tuple(bf.shape)}")
        cq = qf.get_tensor(qp + "ssm_conv1d.weight").float()
        check(torch.equal(cq, conv.T.contiguous()), f"q4nx conv1d == HF conv1d (transposed {tuple(cq.shape)}), bit-exact (HF order)")
        check(torch.allclose(qf.get_tensor(qp + "ssm_a"), a_hf, rtol=1e-6), "q4nx ssm_a == -exp(A_log) (HF order)")
        check(torch.equal(qf.get_tensor(qp + "ssm_dt.bias"), dt), "q4nx ssm_dt == dt_bias (HF order)")

    # full-attention layers: q_proj rows are [head][q|gate] interleaved in HF and GGUF, the q4nx wants all q heads then all gates
    nh, dh = cfg["num_attention_heads"], cfg["head_dim"]
    for L in a.full_layers:
        print(f"--- full-attention layer {L}")
        hq = hf(f"language_model.layers.{L}.self_attn.q_proj.weight").float()
        want = hq.reshape(nh, 2, dh, -1).permute(1, 0, 2, 3).reshape(2 * nh * dh, -1)
        got = qdeq(f"model.layers.{L}.self_attn.q_proj.weight")
        e = rel(got.numpy(), want.numpy())
        check(e < 0.1 and e < 0.5 * rel(got.numpy(), hq.numpy()), f"q4nx q_proj == HF q_proj with [q heads | gate heads] rows: {e:.3f} (Q4_K level), un-permuted {rel(got.numpy(), hq.numpy()):.3f}")

    names = set(qf.keys())
    check("lm_head.weight" in names and "model.embed_tokens.weight" not in names, "tied handling: lm_head.weight present, no model.embed_tokens.weight")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
