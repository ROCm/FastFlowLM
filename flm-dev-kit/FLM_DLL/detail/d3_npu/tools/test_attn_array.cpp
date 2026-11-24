// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Array attention test / microbench of the d3 engine: d3_attn_block::_array_attention (attn.xclbin, one
// run for an unsplit prompt, two runs for prefix_bidir) against the host kernel (d3_cpu::attention, one
// call, or the state x state + question x all pair) on random q / k / v, for GQA 4 (d3-mini: 16 heads,
// 4 kv) and GQA 6 (d3-turbo: 24 heads, 4 kv), several (S, L) including L > 1,536 and > 4,096, a state
// that is not a multiple of 128, one-token questions, and segments packed behind another one (staged
// q / o path).
//
//   d3_test_attn_array <attn.xclbin> [--quick] [--reps N]
//
// Pass criteria (printed per case): cosine >= 0.999 and relative RMS error <= 0.05 against the host
// result, separately for the state rows and the question rows. The array keeps scores, the max and the
// exponent in bf16 (the host in fp32), so bit-exactness against the host is not expected; two things
// are bit-exact and checked: the state rows of a prefix_bidir prompt equal a standalone array run of the
// state alone (run A is exactly that), and two runs of the same prompt are identical.
// Exit code 0 = all passed. Needs the NPU (ps -C flm first; nothing else may use it).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "d3_prefill.hpp"
#include "models/d3/d3_npu.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct Stat { double cos, rel; };

static Stat compare(const bf16* a, const bf16* ref, size_t n) {
    double ab = 0, aa = 0, bb = 0, dd = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = (float)a[i], y = (float)ref[i];
        ab += x * y; aa += x * x; bb += y * y; dd += (x - y) * (x - y);
    }
    return {ab / (std::sqrt(aa) * std::sqrt(bb) + 1e-30), std::sqrt(dd / (bb + 1e-30))};
}

static int g_fail = 0;

struct Fixture {
    npu_xclbin_manager* npu;
    d3_desc desc;
    d3_npu_sequence seq_gen;
    std::unique_ptr<d3_attn_block> blk;
    Fixture(npu_xclbin_manager* n, const std::string& attn_xclbin, int H, int NKV) : npu(n) {
        desc.num_heads = H; desc.num_kv_heads = NKV; desc.DH = 256;
        desc.DQ = 256 * H; desc.DK = desc.DV = 256 * NKV;
        seq_gen = d3_npu_sequence(desc);
        blk = std::make_unique<d3_attn_block>(&desc, &seq_gen, nullptr, n, attn_xclbin);
        if (!blk->_array_ready()) throw std::runtime_error("cannot register " + attn_xclbin);
    }
};

// one batch: segments (S, L) packed in order, array attention for each, host reference, compare
static void run_case(Fixture& f, const std::vector<std::pair<int, int>>& segs, std::mt19937& rng, int reps, const char* tag) {
    std::vector<int> lens, states;
    for (auto& s : segs) { states.push_back(s.first); lens.push_back(s.second); }
    d3_batch b;
    b.build(lens, states);
    const size_t DQ = f.desc.DQ, DK = f.desc.DK;
    d3_scratch sc;
    sc.R_padded = b.R_padded;
    const size_t R = b.R_padded;
    sc.q = d3_cpu::scratch_alloc<bf16>(f.npu, R * DQ);
    sc.k = d3_cpu::scratch_alloc<bf16>(f.npu, R * DK);
    sc.v = d3_cpu::scratch_alloc<bf16>(f.npu, R * DK);
    sc.after_mix = d3_cpu::scratch_alloc<bf16>(f.npu, R * DQ);
    // k, v ~ N(0, 1). q = 0.6 * (a key the row is allowed to see) + 0.4 * N(0, 1): softmax is peaked on that key
    // (score ~ 9.6 against ~N(0, 0.4) for the rest), so outputs are O(1) like a trained model's. Uniform random
    // q / k give near-flat softmax over L keys, outputs ~ 1 / sqrt(L) and a meaningless relative error.
    // Picking only from the visible keys makes a leak across the state / question mask show up.
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto* buf : {&sc.k, &sc.v})
        for (size_t i = 0; i < buf->size(); i++) buf->data()[i] = bf16(nd(rng));
    const int NH = f.desc.num_heads, NKVh = f.desc.num_kv_heads, group = NH / NKVh;
    for (const auto& s : b.segs) {
        for (int r = 0; r < s.len; r++) {
            const int visible = (s.state > 0 && r < s.state) ? s.state : s.len;
            for (int h = 0; h < NH; h++) {
                const int j = (int)(rng() % (unsigned)visible), hk = h / group;
                for (int d = 0; d < 256; d++)
                    sc.q.data()[((size_t)(s.begin + r) * NH + h) * 256 + d] =
                        bf16(0.6f * (float)sc.k.data()[((size_t)(s.begin + j) * NKVh + hk) * 256 + d] + 0.4f * nd(rng));
            }
        }
    }
    memset(sc.after_mix.data(), 0, R * DQ * sizeof(bf16));
    for (auto* buf : {&sc.q, &sc.k, &sc.v, &sc.after_mix}) d3_cpu::scratch_push(*buf);
    f.blk->sc = &sc;

    // host reference, as the engine's host path computes it
    std::vector<bf16> ref(R * DQ, bf16(0.0f));
    std::vector<float> kt;
    const float scale = 1.0f / std::sqrt((float)f.desc.DH);
    double host_ms = 0;
    for (const auto& s : b.segs) {
        const bf16 *qp = sc.q.data() + s.begin * DQ, *kp = sc.k.data() + s.begin * DK, *vp = sc.v.data() + s.begin * DK;
        bf16* o = ref.data() + s.begin * DQ;
        auto t0 = std::chrono::steady_clock::now();
        if (s.state > 0) {
            d3_cpu::attention(o, qp, kp, vp, s.state, s.state, f.desc.num_heads, f.desc.num_kv_heads, 256, 0, 0, DK, DK, scale, false, kt);
            d3_cpu::attention(o + (size_t)s.state * DQ, qp + (size_t)s.state * DQ, kp, vp, s.len - s.state, s.len, f.desc.num_heads,
                              f.desc.num_kv_heads, 256, 0, 0, DK, DK, scale, false, kt);
        } else {
            d3_cpu::attention(o, qp, kp, vp, s.len, s.len, f.desc.num_heads, f.desc.num_kv_heads, 256, 0, 0, DK, DK, scale, false, kt);
        }
        host_ms += ms_since(t0);
    }

    std::vector<bf16> first;
    double arr_ms = 1e30;
    bool repeat_ok = true;
    for (int rep = 0; rep < reps; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (const auto& s : b.segs) f.blk->_array_attention(s, b);
        const double ms = ms_since(t0);
        if (rep > 0) arr_ms = std::min(arr_ms, ms);   // rep 0 builds the sequences
        std::vector<bf16> cur(sc.after_mix.data(), sc.after_mix.data() + (size_t)b.R * DQ);
        if (rep == 0) first = cur;
        else if (memcmp(first.data(), cur.data(), cur.size() * sizeof(bf16)) != 0) repeat_ok = false;
        if (reps == 1) arr_ms = ms;
    }
    f.blk->attn.drop_above(-1);
    f.blk->attn_state.drop_above(-1);

    for (size_t i = 0; i < b.segs.size(); i++) {
        const auto& s = b.segs[i];
        const int S = s.state;
        const bf16* a = first.data() + (size_t)s.begin * DQ;
        const bf16* r = ref.data() + (size_t)s.begin * DQ;
        Stat all = compare(a, r, (size_t)s.len * DQ);
        Stat st = S ? compare(a, r, (size_t)S * DQ) : all;
        Stat qs = S ? compare(a + (size_t)S * DQ, r + (size_t)S * DQ, (size_t)(s.len - S) * DQ) : all;
        const bool ok = st.cos >= 0.999 && st.rel <= 0.05 && qs.cos >= 0.999 && qs.rel <= 0.05;
        std::printf("%-5s GQA %d seg %zu/%zu S=%5d L=%5d  state cos %.6f rel %.4f | question cos %.6f rel %.4f  %s", tag,
                    f.desc.num_heads / f.desc.num_kv_heads, i + 1, b.segs.size(), S, s.len, st.cos, st.rel, qs.cos, qs.rel, ok ? "ok" : "FAIL");
        if (i == 0) std::printf("   [array %.2f ms host %.2f ms per layer, whole batch]", arr_ms, host_ms);
        std::printf("\n");
        if (!ok) g_fail++;
    }
    if (!repeat_ok) { std::printf("FAIL: repeated array runs differ\n"); g_fail++; }

    // the state rows equal a standalone array run of the state alone (bit-exact): a one-segment batch of S tokens
    for (size_t i = 0; i < b.segs.size(); i++) {
        const auto& s = b.segs[i];
        if (s.state <= 0 || i != 0) continue;
        d3_batch b2;
        b2.build({s.state});
        d3_scratch sc2;
        const size_t R2 = b2.R_padded;
        sc2.R_padded = b2.R_padded;
        sc2.q = d3_cpu::scratch_alloc<bf16>(f.npu, R2 * DQ);
        sc2.k = d3_cpu::scratch_alloc<bf16>(f.npu, R2 * DK);
        sc2.v = d3_cpu::scratch_alloc<bf16>(f.npu, R2 * DK);
        sc2.after_mix = d3_cpu::scratch_alloc<bf16>(f.npu, R2 * DQ);
        memset(sc2.q.data(), 0, R2 * DQ * 2); memset(sc2.k.data(), 0, R2 * DK * 2); memset(sc2.v.data(), 0, R2 * DK * 2);
        memset(sc2.after_mix.data(), 0, R2 * DQ * 2);
        memcpy(sc2.q.data(), sc.q.data() + s.begin * DQ, (size_t)s.state * DQ * 2);
        memcpy(sc2.k.data(), sc.k.data() + s.begin * DK, (size_t)s.state * DK * 2);
        memcpy(sc2.v.data(), sc.v.data() + s.begin * DK, (size_t)s.state * DK * 2);
        for (auto* buf : {&sc2.q, &sc2.k, &sc2.v, &sc2.after_mix}) d3_cpu::scratch_push(*buf);
        f.blk->sc = &sc2;
        f.blk->_array_attention(b2.segs[0], b2);
        f.blk->attn.drop_above(-1);
        const bool same = memcmp(sc2.after_mix.data(), first.data() + (size_t)s.begin * DQ, (size_t)s.state * DQ * 2) == 0;
        std::printf("      state rows == standalone array run of S=%d tokens: %s\n", s.state, same ? "bit-identical" : "DIFFERENT");
        if (!same) g_fail++;
        f.blk->sc = &sc;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s attn.xclbin [--quick] [--reps N]\n", argv[0]); return 2; }
    const std::string xclbin = argv[1];
    bool quick = false;
    int reps = 3;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = true;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    std::mt19937 rng(2024);

    struct Shape { int H, NKV; };
    for (const Shape sh : {Shape{16, 4}, Shape{24, 4}}) {
        Fixture f(&npu, xclbin, sh.H, sh.NKV);
        std::printf("--- %d heads, %d kv heads (GQA %d)\n", sh.H, sh.NKV, sh.H / sh.NKV);
        run_case(f, {{0, 300}}, rng, reps, "full");                          // unsplit, short
        run_case(f, {{100, 300}}, rng, reps, "pfx");
        run_case(f, {{129, 1600}}, rng, reps, "pfx");                        // odd state, past 1,536
        run_case(f, {{255, 256}}, rng, reps, "pfx");                         // one-token question
        run_case(f, {{1, 129}}, rng, reps, "pfx");
        run_case(f, {{128, 640}, {0, 200}, {77, 333}}, rng, reps, "pack");   // packed: segments behind segment 0 are staged
        if (quick) continue;
        run_case(f, {{1000, 1700}}, rng, reps, "pfx");
        run_case(f, {{0, 2000}}, rng, reps, "full");
        run_case(f, {{1537, 2100}}, rng, reps, "pfx");
        run_case(f, {{3000, 4300}}, rng, reps, "pfx");                       // > 4,096
        run_case(f, {{4200, 4300}}, rng, reps, "pfx");
        run_case(f, {{0, 4500}}, rng, reps, "full");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASSED");
    return g_fail ? 1 : 0;
}
