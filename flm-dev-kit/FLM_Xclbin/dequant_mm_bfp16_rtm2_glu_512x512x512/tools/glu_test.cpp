// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Fused GLU dispatch (rtp_act 3) test.
//   glu_test <xclbin> [reps]
// Random Q4_K up/gate weights of the MiniCPM-V 1B projection (1024 -> 7168, slabs of 512 up | 512 gate rows).
//   unfused: plain dispatch (act 0) on the model's layout -> y[L][7168]; host GLU hid_ref = bf16(up * silu(gate))
//            from those bf16 values, as cpu_func::_exp_glu does
//   fused:   act 3 dispatch on glu_repack()ed weights -> hid[L][3584]
// Reports exact matches and the bf16 ulp distribution of fused vs host GLU; checks rows past `covered` stay untouched
// and that plain dispatches stay bit-identical when interleaved with fused ones (the memtile descriptors are
// rewritten and restored around every fused dispatch).
#include <bits/stdc++.h>
#include <stdfloat>
#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "lm_config.hpp"
#include "seq_gen.hpp"
#include "tensor_2d.hpp"
#include "cpu_q4_k.hpp"
#include "util.hpp"
#include "weights_desc.hpp"
using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b){ return std::chrono::duration<double, std::micro>(b - a).count(); }
static float bf(bf16 v){ return (float)v; }
static uint16_t bits(bf16 v){ uint16_t u; memcpy(&u, &v, 2); return u; }
static int ulp_dist(bf16 a, bf16 b){
    // distance in bf16 steps on the ordered number line
    auto ord = [](uint16_t u){ return (u & 0x8000) ? -(int)(u & 0x7FFF) : (int)(u & 0x7FFF); };
    return std::abs(ord(bits(a)) - ord(bits(b)));
}

int main(int argc, char** argv){
    if (argc < 2) { fprintf(stderr, "usage: glu_test <xclbin> [reps]\n"); return 2; }
    const std::string xcl = argv[1];
    const int reps = argc > 2 ? atoi(argv[2]) : 2;
    const int Din = 1024, Dout = 7168, IS = Dout / 2, LMAX = 2048;
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mm = mgr.register_xclbin(xcl);
    std::mt19937 rng(777);
    const size_t wbytes = q4_k_size((size_t)Din * Dout);
    buffer<bf16> w((size_t)Dout * Din), wd((size_t)Dout * Din);
    std::normal_distribution<float> dist(0.f, 0.03f);
    for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
    buffer<u8> wp(wbytes);
    q4_k_quantize_pack(wp.data(), wd.data(), w.data(), Dout, Din);
    std::vector<bf16> xh((size_t)LMAX * Din);
    std::normal_distribution<float> xd(0.f, 1.f);
    for (auto& v : xh) v = bf16(xd(rng));

    npu_app base = mm->create_app();
    buffer<u8> qw = base.create_bo_buffer<u8>(wbytes);
    reorder_cpy(qw.data(), wp, Din, 2);
    qw.sync_to_device();
    npu_app app_g0 = mm->create_app();   // owns the glu weights
    buffer<u8> qwg = app_g0.create_bo_buffer<u8>(wbytes);
    glu_repack(qwg.data(), qw.data(), Din, Dout);
    qwg.sync_to_device();

    buffer<bf16> x = base.create_bo_buffer<bf16>((size_t)LMAX * Din);
    memcpy(x.data(), xh.data(), xh.size() * sizeof(bf16)); x.sync_to_device();
    buffer<bf16> y = base.create_bo_buffer<bf16>((size_t)LMAX * Dout);
    buffer<bf16> hid = base.create_bo_buffer<bf16>((size_t)LMAX * IS);

    std::vector<int> rowsets = {1, 8, 24, 72, 264, 520, 1032, 2048};
    if (const char* rs = getenv("GLU_ROWS")) { rowsets.clear(); std::stringstream ss(rs); std::string t; while (std::getline(ss, t, ',')) rowsets.push_back(atoi(t.c_str())); }
    // both kinds of dispatch for every row count
    std::vector<std::unique_ptr<npu_app>> plain, fused;
    for (int rows : rowsets) {
        plain.push_back(std::make_unique<npu_app>(mm->create_app()));
        gen_dequant_mm_rtm(plain.back()->seq(), rows, Din, Dout, 0, 0, 0);
        fused.push_back(std::make_unique<npu_app>(mm->create_app()));
        gen_dequant_mm_rtm(fused.back()->seq(), rows, Din, Dout, 0, 0, 3);
    }
    int bad = 0;
    // a row's fused result must not depend on how many rows the dispatch covers (m of the reduced-row geometry, blocks):
    // the first dispatch that reaches LMAX rows is the reference for every other row count's overlapping rows
    std::vector<bf16> fused_full;
    // reference per row count (plain dispatch) taken once, in order, then compared in shuffled interleaved reps
    std::vector<std::vector<bf16>> plain_ref(rowsets.size());
    std::vector<int> order(rowsets.size()); std::iota(order.begin(), order.end(), 0);
    const bf16 sentinel = bf16(12345.f);
    for (int r = 0; r < reps; r++) {
        if (r > 0) std::shuffle(order.begin(), order.end(), rng);
        for (int oi : order) {
            const int rows = rowsets[oi];
            const rtm_geometry g = rtm_pick_geometry(rows);
            // --- plain dispatch
            for (size_t i = 0; i < y.size(); i++) y[i] = sentinel;
            y.sync_to_device();
            auto pa = clk::now();
            (*plain[oi])(y, x, qw);
            auto pb = clk::now();
            y.sync_from_device();
            if (plain_ref[oi].empty()) plain_ref[oi].assign(y.data(), y.data() + (size_t)rows * Dout);
            else if (memcmp(plain_ref[oi].data(), y.data(), (size_t)rows * Dout * sizeof(bf16))) { printf("BAD plain rows=%d rep=%d: plain output changed after fused dispatches\n", rows, r); bad++; }
            // host GLU of the plain output
            std::vector<bf16> ref((size_t)rows * IS);
            for (int l = 0; l < rows; l++)
                for (int s = 0; s < Dout / 1024; s++)
                    for (int i = 0; i < 512; i++) {
                        const float up = bf(y[(size_t)l * Dout + s * 1024 + i]);
                        const float gt = bf(y[(size_t)l * Dout + s * 1024 + 512 + i]);
                        ref[(size_t)l * IS + s * 512 + i] = bf16(up * (gt / (1.0f + expf(-gt))));
                    }
            // --- fused dispatch
            for (size_t i = 0; i < hid.size(); i++) hid[i] = sentinel;
            hid.sync_to_device();
            auto a = clk::now();
            (*fused[oi])(hid, x, qwg);
            auto b = clk::now();
            hid.sync_from_device();
            size_t exact = 0, tot = (size_t)rows * IS, nover = 0; int maxu = 0; double sumu = 0; size_t hist[5] = {0};
            double se = 0, sr = 0; size_t nanbad = 0;
            for (int l = 0; l < rows; l++)
                for (int i = 0; i < IS; i++) {
                    const bf16 got = hid[(size_t)l * IS + i], want = ref[(size_t)l * IS + i];
                    if (!std::isfinite(bf(got))) { nanbad++; continue; }
                    const int d = ulp_dist(got, want);
                    exact += d == 0; maxu = std::max(maxu, d); sumu += d; hist[std::min(d, 4)]++;
                    const double e = (double)bf(got) - (double)bf(want); se += e * e; sr += (double)bf(want) * bf(want);
                }
            if (r == 0) {
                // error against an fp64 reference from the dequantized weights (first rows only): the plain dispatch +
                // host GLU path and the fused one
                const int nt = std::min(rows, 48);
                double e_f = 0, e_h = 0, r2 = 0; size_t cnt = 0;
                std::vector<double> up_t(512), gt_t(512);
                for (int l = 0; l < nt; l++)
                    for (int s2 = 0; s2 < Dout / 1024; s2++) {
                        for (int i = 0; i < 512; i++) {
                            double au = 0, ag = 0;
                            const bf16* wu = &wd[((size_t)s2 * 1024 + i) * Din];
                            const bf16* wg = &wd[((size_t)s2 * 1024 + 512 + i) * Din];
                            for (int k = 0; k < Din; k++) { const double xv = bf(xh[(size_t)l * Din + k]); au += xv * bf(wu[k]); ag += xv * bf(wg[k]); }
                            const double tru = au * ag / (1.0 + exp(-ag));
                            const double fv = bf(hid[(size_t)l * IS + s2 * 512 + i]), hv = bf(ref[(size_t)l * IS + s2 * 512 + i]);
                            e_f += (fv - tru) * (fv - tru); e_h += (hv - tru) * (hv - tru); r2 += tru * tru; cnt++;
                        }
                    }
                printf("  vs fp64 truth (%zu values): host GLU of plain output rel-rms %.3e   fused rel-rms %.3e\n", cnt, sqrt(e_h / r2), sqrt(e_f / r2));
            }
            if (getenv("GLU_DIAG") && r == 0) {
                // error by gate magnitude (relative to the reference), and samples
                const double edges[] = {0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9};
                double eb[7] = {0}, rb[7] = {0}; size_t cb[7] = {0};
                for (int l = 0; l < rows; l++)
                    for (int s2 = 0; s2 < Dout / 1024; s2++)
                        for (int i = 0; i < 512; i++) {
                            const float gt = bf(y[(size_t)l * Dout + s2 * 1024 + 512 + i]);
                            const double want = bf(ref[(size_t)l * IS + s2 * 512 + i]), got = bf(hid[(size_t)l * IS + s2 * 512 + i]);
                            double ag = fabs(gt); int b = 0; while (b < 6 && ag >= edges[b + 1]) b++;
                            eb[b] += (got - want) * (got - want); rb[b] += want * want; cb[b]++;
                        }
                for (int b = 0; b < 7; b++) printf("  |gate| in [%.2f,%.2f): n=%zu rel-rms %.3e  (err energy %.3e)\n", edges[b], edges[b + 1], cb[b], rb[b] > 0 ? sqrt(eb[b] / rb[b]) : 0.0, eb[b]);
                for (int i = 0; i < 10; i++) {
                    const int idx = 37 * i + 5;
                    const float up = bf(y[(size_t)3 % rows * Dout + idx]), gt = bf(y[(size_t)(3 % rows) * Dout + 512 + idx]);
                    printf("  sample up %+.4f gate %+.4f  ref %+.5f got %+.5f\n", up, gt, bf(ref[(size_t)(3 % rows) * IS + idx]), bf(hid[(size_t)(3 % rows) * IS + idx]));
                }
            }
            for (size_t row = g.covered; row < (size_t)LMAX; row++) {
                const bf16* p = &hid[row * IS];
                for (int n = 0; n < IS; n++) if (!(p[n] == sentinel)) { nover++; break; }
            }
            printf("rows=%-5d rep=%d  plain %7.0f us fused %7.0f us  exact %.4f%%  ulp mean %.3f max %d  [0:%zu 1:%zu 2:%zu 3:%zu >=4:%zu]  rel-rms %.2e  nonfinite %zu overrun-rows %zu\n",
                   rows, r, us(pa, pb), us(a, b), 100.0 * exact / tot, sumu / tot, maxu, hist[0], hist[1], hist[2], hist[3], hist[4], sqrt(se / sr), nanbad, nover);
            if (rows >= LMAX && fused_full.empty()) fused_full.assign(hid.data(), hid.data() + (size_t)LMAX * IS);
            if (!fused_full.empty() && rows < LMAX) {
                size_t nd = 0;
                for (int l = 0; l < rows; l++) if (memcmp(&hid[(size_t)l * IS], &fused_full[(size_t)l * IS], IS * sizeof(bf16))) nd++;
                if (nd) { printf("BAD fused rows=%d: %zu rows differ from the same rows of the %d-row dispatch\n", rows, nd, LMAX); bad++; }
                else printf("  fused rows=%d identical to the same rows of the %d-row dispatch\n", rows, LMAX);
            }
            if (nover || nanbad) bad++;
            fflush(stdout);
        }
    }
    printf(bad ? "RESULT FAIL\n" : "RESULT PASS\n");
    return bad != 0;
}
