// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_fused_probe.cpp
/// \brief NPU decode (inkling2_fused) step by step against the host model's decode trace on real weights
/// \note usage: i2fprobe model.q4nx prompt.ids mm_rtm2_glu.xclbin decode.xclbin first_app last_app stages split
///       The host runs the prompt, then the next token with its trace on; the NPU takes the host state from before
///       that token and runs applications first..last (steps `stages`, a bit mask of inkling2_fused::ST_*), one run
///       per application or, with split = 1, one run per step. Each output is compared with the host (rel rms).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../inkling2_fused.hpp"
#include "../inkling2_host.hpp"
#include "../inkling2_prefill.hpp"
#include "npu_utils/npu_utils.hpp"

using namespace inkling2_host;

namespace {
std::map<std::pair<int, std::string>, std::vector<float>> trace;

float bf(u16 v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

const std::vector<float>* ref(int slot, const char* tag) {
    auto it = trace.find({slot, tag});
    return it == trace.end() ? nullptr : &it->second;
}

template <class G> double rel(const float* r, G got, int n) {
    double e = 0, s = 0;
    for (int i = 0; i < n; i++) { const double d = (double)got(i) - r[i]; e += d * d; s += (double)r[i] * r[i]; }
    return std::sqrt(e / std::max(s, 1e-30));
}
}   // namespace

int main(int argc, char** argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s model.q4nx prompt.ids mm_rtm2_glu.xclbin decode.xclbin first_app last_app stages split\n", argv[0]);
        return 2;
    }
    const int a0 = atoi(argv[5]), a1 = atoi(argv[6]), stages = atoi(argv[7]), split = atoi(argv[8]), ngen = argc > 9 ? atoi(argv[9]) : 0,
              every = argc > 10 ? atoi(argv[10]) : 0;
    std::vector<int> ids;
    {
        std::ifstream f(argv[2]);
        for (int t; f >> t;) ids.push_back(t);
    }
    model m;
    m.load(argv[1]);
    const int ctx = getenv("INKLING2_PROBE_CTX") ? atoi(getenv("INKLING2_PROBE_CTX")) : 2048;   // the context the buffers are for
    m.set_max_length(ctx);
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    inkling2_prefill pf(&mgr, argv[3]);
    pf.load(m.w_mut());
    std::vector<float> logits(VOCAB_PAD);
    m.run(ids.data(), (int)ids.size(), logits.data());
    int tok = (int)(std::max_element(logits.begin(), logits.begin() + VOCAB) - logits.begin());
    printf("prompt %zu tokens, next %d\n", ids.size(), tok);

    std::vector<inkling2_fused::layer_src> src;
    for (int il = 0; il < NL; il++) {
        const auto d = pf.device_layer(il);
        src.push_back({d.w, d.o, d.gate_up, d.down});
    }
    inkling2_fused F(&mgr, argv[4], m.w(), src, ctx);
    // INKLING2_PROBE_LMH=<inkling2_decode.switches>: the lm_head on the NPU inside the token's run (checked against the
    // host's lm_head of the same x)
    if (const char* sw = getenv("INKLING2_PROBE_LMH")) {
        if (!F.enable_lm_head(&mgr, sw)) { printf("lm_head switches did not load\n"); return 2; }
        printf("lm_head on the NPU (rounds per descriptor %d)\n", F.lmh->rpb);
    }
    F.load_state(m);
    if (ngen > 0) {
        // NPU greedy decode from `tok`, the host fed the same tokens: top-1 agreement, the host's margin where they differ
        int agree = 0;
        double t_npu = 0, t_lm = 0;
        F.sync_all = getenv("INKLING2_SYNC_ALL") && atoi(getenv("INKLING2_SYNC_ALL"));
        printf("chain %d, sync_all %d\n", F.chain, (int)F.sync_all);
        std::vector<float> x(D), nl(VOCAB_PAD), hl(VOCAB_PAD), dl(VOCAB_PAD);
        double lm_worst = 0, t_fin = 0, t_begin = 0;
        int lm_agree = 0, lm_top8 = 0;
        for (int k = 0; k < ngen; k++) {
            if (every > 0 && k > 0 && k % every == 0) {
                // round trip: the host takes the NPU's state (vs its own, teacher-forced), the NPU reloads it
                std::vector<u16> hk(KVD), nk(KVD);
                std::vector<float> hc((size_t)(NCONV - 1) * CONV_CH), nc(hc.size());
                std::vector<std::vector<u16>> rows;
                std::vector<std::vector<float>> convs;
                for (int s = 0; s < NSLOT; s++) {
                    m.slot_conv(s, hc.data());
                    convs.push_back(hc);
                    for (int q = F.position() - 1; q >= F.position() - every; q--)
                        for (int v = 0; v < 2; v++) { m.slot_row(s, q, v, hk.data()); rows.push_back(hk); }
                }
                F.store_state(m);
                double er = 0, sr = 0, ec = 0, sc = 0;
                size_t i = 0;
                for (int s = 0; s < NSLOT; s++) {
                    m.slot_conv(s, nc.data());
                    for (size_t c = 0; c < nc.size(); c++) { ec += (double)(nc[c] - convs[s][c]) * (nc[c] - convs[s][c]); sc += (double)convs[s][c] * convs[s][c]; }
                    for (int q = F.position() - 1; q >= F.position() - every; q--)
                        for (int v = 0; v < 2; v++, i++) {
                            m.slot_row(s, q, v, nk.data());
                            for (int c = 0; c < KVD; c++) { const double a = bf(nk[c]), b = bf(rows[i][c]); er += (a - b) * (a - b); sr += b * b; }
                        }
                }
                printf("round trip at %d: rows rel %.2e conv rel %.2e", F.position(), std::sqrt(er / sr), std::sqrt(ec / sc));
                {
                    const int at[5] = {CONV_K, CONV_V, CONV_ATTN, CONV_FFN, CONV_CH};
                    const char* nm[4] = {"k", "v", "attn", "ffn"};
                    for (int t = 0; t < 4; t++) {
                        double e = 0, z = 0, emax = 0; int smax = -1;
                        for (int s2 = 0; s2 < NSLOT; s2++) {
                            m.slot_conv(s2, nc.data());
                            double es = 0, zs = 0;
                            for (int k = 0; k < NCONV - 1; k++)
                                for (int c = at[t]; c < at[t + 1]; c++) { const double d = nc[(size_t)k * CONV_CH + c] - convs[s2][(size_t)k * CONV_CH + c]; es += d * d; zs += (double)convs[s2][(size_t)k * CONV_CH + c] * convs[s2][(size_t)k * CONV_CH + c]; }
                            e += es; z += zs;
                            if (std::sqrt(es / zs) > emax) { emax = std::sqrt(es / zs); smax = s2; }
                        }
                        printf(" | %s %.2e (worst slot %d %.2e)", nm[t], std::sqrt(e / z), smax, emax);
                    }
                    printf("\n");
                }
                F.load_state(m);
            }
            const auto tb = std::chrono::steady_clock::now();
            F.begin_token(tok);
            const auto t0 = std::chrono::steady_clock::now();
            t_begin += std::chrono::duration<double, std::milli>(t0 - tb).count();
            if (!F.run_token()) { printf("token %d RUN FAILED\n", k); return 3; }
            t_npu += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            for (int c = 0; c < D; c++) x[c] = bf(((const u16*)(F.io_data() + F.XZ))[c]);
            const auto tl = std::chrono::steady_clock::now();
            lm_head(m.w(), x.data(), nl.data());
            t_lm += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count();
            if (F.lm_on_npu()) {
                const auto tf = std::chrono::steady_clock::now();
                F.logits(dl.data());
                t_fin += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
                double e = 0, s = 0;
                for (int i = 0; i < VOCAB; i++) { e += (double)(dl[i] - nl[i]) * (dl[i] - nl[i]); s += (double)nl[i] * nl[i]; }
                lm_worst = std::max(lm_worst, std::sqrt(e / s));
                std::vector<int> a(VOCAB), b(VOCAB);
                for (int i = 0; i < VOCAB; i++) a[i] = b[i] = i;
                std::partial_sort(a.begin(), a.begin() + 8, a.end(), [&](int u, int v) { return dl[u] > dl[v]; });
                std::partial_sort(b.begin(), b.begin() + 8, b.end(), [&](int u, int v) { return nl[u] > nl[v]; });
                lm_agree += a[0] == b[0];
                for (int i = 0; i < 8; i++) lm_top8 += std::count(b.begin(), b.begin() + 8, a[i]) > 0;
            }
            F.end_token();
            m.run(&tok, 1, hl.data());
            const int tn = (int)(std::max_element(nl.begin(), nl.begin() + VOCAB) - nl.begin());
            const int th = (int)(std::max_element(hl.begin(), hl.begin() + VOCAB) - hl.begin());
            double e = 0, s = 0;
            for (int i = 0; i < VOCAB; i++) { e += (double)(nl[i] - hl[i]) * (nl[i] - hl[i]); s += (double)hl[i] * hl[i]; }
            agree += tn == th;
            printf("token %3d: npu %6d host %6d%s logits rel %.2e%s\n", k, tn, th, tn == th ? "" : " DIFF", std::sqrt(e / s),
                   tn == th ? "" : (" host margin " + std::to_string(hl[th] - hl[tn])).c_str());
            fflush(stdout);
            tok = tn;
        }
        if (F.ck_n) printf("ctrl check: %d queries, %ld elements differ, worst %.2e\n", F.ck_n, F.ck_off, F.ck_worst);
        if (F.dk_n) printf("device ctrl check: %d steps, down %.2e (of max |g|), up %.2e, out %.2e (%ld of %ld bf16 differ, %ld by > 1 ulp)\n", F.dk_n, F.dk_down, F.dk_up, F.dk_out, F.dk_off, F.dk_el, F.dk_far);
        printf("agree %d/%d, NPU layers %.1f ms/token, host lm_head %.1f ms/token, begin_token (embedding row, headers) %.2f ms/token\n",
               agree, ngen, t_npu / ngen, t_lm / ngen, t_begin / ngen);
        if (F.lm_on_npu())
            printf("NPU lm_head vs host lm_head (same x): argmax %d/%d, top-8 %d/%d, worst logits rel %.2e, finish (y sync + soft cap) %.2f ms/token\n",
                   lm_agree, ngen, lm_top8, 8 * ngen, lm_worst, t_fin / ngen);
        printf("per token: %.1f runs, seq %.2f ms, run %.2f ms (create_run %.2f), sync %.2f ms, host ctrl %.2f ms, next built during the run %.2f ms\n", (double)F.n_run / ngen,
               F.t_seq / ngen, F.t_run / ngen, F.t_create / ngen, F.t_sync / ngen, F.t_ctrl / ngen, F.t_pre / ngen);
        for (int b = 0; b < 128; b++)
            if (F.n_by[b]) printf("  steps %3d: %5.1f runs/token, %.3f ms/run\n", b, (double)F.n_by[b] / ngen, F.t_by[b] / F.n_by[b]);
        return 0;
    }
    decode_trace = [](int slot, const char* tag, const float* x, int n) { trace[{slot, tag}].assign(x, x + n); };
    m.run(&tok, 1, logits.data());
    decode_trace = nullptr;
    F.begin_token(tok);

    const int order[6] = {inkling2_fused::ST_A, inkling2_fused::ST_PROJ, inkling2_fused::ST_ATTN, inkling2_fused::ST_OPROJ,
                          inkling2_fused::ST_F, inkling2_fused::ST_MOE};
    const char* name[6] = {"A", "proj", "attn", "oproj", "F", "moe"};
    bool ok = true;
    for (int j = a0; j <= a1; j++) {
        const auto I = inkling2_fused::info(j);
        printf("app %2d (layer %2d pass %d slot %2d %s%s):", j, I.il, I.pass, I.slot, I.glob ? "global" : "swa", I.starts ? ", starts" : "");
        const auto t0 = std::chrono::steady_clock::now();
        if (split) {
            for (int k = 0; k < 6; k++)
                if ((stages & order[k]) && !F.run_app(j, order[k])) { printf(" %s RUN FAILED\n", name[k]); return 3; }
        } else if (!F.run_app(j, stages)) { printf(" RUN FAILED\n"); return 3; }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        const uint8_t* io = F.io_data();
        auto Fp = [&](size_t at) { return (const float*)(io + at); };
        auto Bp = [&](size_t at) { return (const u16*)(io + at); };
        auto report = [&](const char* what, const std::vector<float>* r, int n, auto got, double lim) {
            if (!r) { printf(" %s:no-ref", what); return; }
            const double e = rel(r->data(), got, std::min(n, (int)r->size()));
            printf(" %s %.2e%s", what, e, e < lim ? "" : "!");
            ok &= e < lim;
        };
        const size_t H = I.three ? F.H3 : F.H1;
        if (stages & inkling2_fused::ST_A) {
            const auto* r = ref(I.slot, "nrm");
            report("nrm", r, r ? (int)r->size() : 0, [&](int i) { return bf(Bp(H + 128)[i]); }, 2e-2);
            if (j > 0) {
                const size_t cur_at = I.starts ? F.ST + (size_t)I.S * D * 4 : F.BF[j] + inkling2_fused::CB;
                report("in", ref(I.slot, "in"), D, [&](int i) { return Fp(cur_at)[i]; }, 3e-2);
            }
        }
        if ((stages & inkling2_fused::ST_A) && I.pass > 0 && I.il == CORE0) {
            // the pass completion (o17 vs the host's) and the messages vs a mix of the NPU's own o11, o17
            const auto Jp = inkling2_fused::info(j - 1);
            report("o17", ref(Jp.slot, "out"), D, [&](int i) { return Fp(F.BANK)[i]; }, 3e-2);
            const float* st[2] = {Fp(F.ST + (size_t)2 * D * 4), Fp(F.BANK)};
            double inv[2];
            for (int k = 0; k < 2; k++) {
                double ss = 0;
                for (int c = 0; c < D; c++) ss += (double)st[k][c] * st[k][c];
                inv[k] = 1.0 / std::sqrt(ss / D + EPS);
            }
            std::vector<float> mref((size_t)4 * (CORE1 - CORE0) * D);
            for (int q = 0; q < 4 * (CORE1 - CORE0); q++) {
                const float* qv = m.w().layer[CORE0 + q / 4].msg_query + (size_t)(q % 4) * D;
                double wv[2], mx = -1e300, sum = 0;
                for (int k = 0; k < 2; k++) {
                    double d = 0;
                    for (int c = 0; c < D; c++) d += (double)qv[c] * st[k][c];
                    wv[k] = d * inv[k] / std::sqrt((double)D);
                    mx = std::max(mx, wv[k]);
                }
                for (int k = 0; k < 2; k++) sum += (wv[k] = std::exp(wv[k] - mx));
                for (int c = 0; c < D; c++) mref[(size_t)q * D + c] = (float)(wv[0] * inv[0] / sum * st[0][c] + wv[1] * inv[1] / sum * st[1][c]);
            }
            const std::vector<float> mv = mref;
            report("msg", &mv, (int)mv.size(), [&](int i) { return Fp(F.MSG + (size_t)(i / D) * F.CQ)[i % D]; }, 1e-2);
        }
        if (stages & inkling2_fused::ST_PROJ) {
            report("q", ref(I.slot, "q"), QD, [&](int i) { return Fp(F.PY + 29 * 384)[i]; }, 3e-2);
            report("k", ref(I.slot, "k"), KVD, [&](int i) { return Fp(F.PY)[i]; }, 3e-2);
        }
        if (stages & inkling2_fused::ST_ATTN)
            report("o", ref(I.slot, "o"), QD, [&](int n) {
                const int g = n / 384;
                return bf(Bp(F.OH + (size_t)(g / 4) * 4736 + 128 + (g % 4) * 768)[n % 384]);
            }, 5e-2);
        if (stages & inkling2_fused::ST_OPROJ)
            report("y", ref(I.slot, "y"), D, [&](int i) { return Fp(F.OY)[i] + Fp(F.OY)[D + i]; }, 5e-2);
        if (stages & inkling2_fused::ST_F) {
            const size_t cur_at = j + 1 < inkling2_fused::NAPP ? F.BA[j + 1] + inkling2_fused::CB : F.FIN;
            report("res", ref(I.slot, "res"), D, [&](int i) { return Fp(cur_at)[i]; }, 3e-2);
            report("xf", ref(I.slot, "xf"), D, [&](int i) { return bf(Bp(F.E_HDR + 128)[i]) + bf(Bp(F.E_HDR + 128)[D + i]); }, 3e-2);
        }
        if (stages & inkling2_fused::ST_MOE) {
            const auto* s = ref(I.slot, "sel");
            const int* ti = (const int*)(io + F.TY);
            const float* tw = Fp(F.TY) + TOPK;
            if (s) {
                int same = 0;
                for (int a = 0; a < TOPK; a++)
                    for (int b = 0; b < TOPK; b++) same += ti[a] == (int)(*s)[b];
                printf(" sel %d/8", same);
                if (same < TOPK) {
                    // a near tie of the router: the swapped experts' weights on either side
                    for (int a = 0; a < TOPK; a++)
                        if (std::find(s->begin(), s->begin() + TOPK, (float)ti[a]) == s->begin() + TOPK) printf(" [npu %d w %.4f]", ti[a], tw[a]);
                    for (int b = 0; b < TOPK; b++)
                        if (std::find(ti, ti + TOPK, (int)(*s)[b]) == ti + TOPK) printf(" [host %d w %.4f]", (int)(*s)[b], (*s)[TOPK + b]);
                }
                ok &= same >= 7;
            }
            if (s && std::count_if(ti, ti + TOPK, [&](int e) { return std::find(s->begin(), s->begin() + TOPK, (float)e) != s->begin() + TOPK; }) < TOPK)
                printf(" (moe vs host not comparable)");
            else
            report("moe", ref(I.slot, "moe"), D, [&](int i) {
                const int r = i / 1152, k = i % 1152;
                double f = 0;
                for (int e = 0; e < TOPK; e++) f += (double)tw[e] * Fp(F.EY)[(size_t)(e + 8 * r) * 1152 + k];
                return (float)f;
            }, 5e-2);
        }
        if ((stages & inkling2_fused::ST_Z) && j == inkling2_fused::NAPP - 1) {
            report("fin", ref(I.slot, "out"), D, [&](int i) { return Fp(F.XF)[i]; }, 5e-2);
            report("x", ref(NSLOT, "final"), D, [&](int i) { return bf(Bp(F.XZ)[i]); }, 5e-2);
        }
        printf(" | %.0f us\n", us);
        fflush(stdout);
    }
    for (int b = 0; b < 128; b++)
        if (F.n_by[b]) printf("  steps %3d: %3d runs, %.3f ms/run\n", b, F.n_by[b], F.t_by[b] / F.n_by[b]);
    if (split) {
        // one token's profile: the bytes each step reads from DRAM (weights, K / V records; activations < 1 MB not
        // counted) over its runs' time less a per-run dispatch overhead. The overhead: (sum of the split runs - the
        // fused token run, INKLING2_PROBE_FUSED_MS) / runs, i.e. the split is scaled to add up to the fused run.
        constexpr double UNITB = 4736, Q4K = 4736.0 / 8192;
        const int pos = F.position();
        double by[128] = {};
        for (int j = a0; j <= a1; j++) {
            const auto I = inkling2_fused::info(j);
            const auto d = pf.device_layer(I.il);
            const bool loop = I.pass > 0;
            by[inkling2_fused::ST_PROJ] += (double)d.o + 96 * UNITB;   // attention rows; dense rows (sinks, conv)
            const int rows = I.glob ? pos + 1 : std::min(pos + 1, SWA);
            by[inkling2_fused::ST_ATTN] += NKV * (10 * UNITB + (rows + 7) / 8 * UNITB);
            by[inkling2_fused::ST_OPROJ] += (double)(d.gate_up - d.o);
            by[inkling2_fused::ST_MOE] += 256 * UNITB + TOPK * ((double)(d.down - d.gate_up) / NE + (double)D * FF_PAD * Q4K);
            if (loop) {   // the controllers' weights (bf16): attention down + up after A, ffn after F
                by[inkling2_fused::ST_A] += (384 + 432) * UNITB;
                by[inkling2_fused::ST_F] += (128 + 144) * UNITB;
            }
        }
        double sum = 0;
        int runs = 0;
        for (int b = 0; b < 128; b++) { sum += F.t_by[b]; runs += F.n_by[b]; }
        const double fused = getenv("INKLING2_PROBE_FUSED_MS") ? atof(getenv("INKLING2_PROBE_FUSED_MS")) : 0;
        const double ovh = fused > 0 ? std::max(0.0, (sum - fused) / runs) : 0;
        printf("\none token at position %d (applications %d-%d): split runs %.2f ms in %d runs; fused run %.2f ms -> dispatch "
               "overhead %.3f ms/run taken off\n", pos, a0, a1, sum, runs, fused, ovh);
        printf("| NPU engine | runs | split ms | est. device ms | %% of NPU | MB/token | effective GB/s |\n|---|---|---|---|---|---|---|\n");
        const int order2[6] = {inkling2_fused::ST_A, inkling2_fused::ST_PROJ, inkling2_fused::ST_ATTN, inkling2_fused::ST_OPROJ,
                               inkling2_fused::ST_F, inkling2_fused::ST_MOE};
        const char* nm2[6] = {"A mix (+ loop attn controllers)", "projections (k v q gate rel)", "attention (conv, clip, KV, sinks, gate)",
                              "o proj", "F mix (+ loop ffn controllers)", "router + top-8 experts (gate_up, situ, down)"};
        double dev_all = 0, mb_all = 0;
        for (int k = 0; k < 6; k++) dev_all += F.t_by[order2[k]] - ovh * F.n_by[order2[k]];
        for (int k = 0; k < 6; k++) {
            const int b = order2[k];
            const double dev = F.t_by[b] - ovh * F.n_by[b];
            mb_all += by[b] / 1e6;
            printf("| %s | %d | %.2f | %.2f | %.1f | %.1f | %.1f |\n", nm2[k], F.n_by[b], F.t_by[b], dev, 100 * dev / dev_all, by[b] / 1e6,
                   by[b] / 1e6 / dev);
        }
        printf("| total | %d | %.2f | %.2f | 100 | %.1f | %.1f |\n", runs, sum, dev_all, mb_all, mb_all / dev_all);
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
