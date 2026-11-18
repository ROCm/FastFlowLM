// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n4_prefill_attn.cpp
/// \brief N4 -- the prefill attention core on the NPU, and the hybrid prefill
///        layer built around it.
///
/// Build (one line; a backslash at the end of a // comment splices the next
/// line into it, so the continuations are dropped here):
///
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20 -ffast-math
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n4_prefill_attn.cpp prefill/npu/prefill_attn_sequence.cpp
///       verify/npu/attn_sequence.cpp ../q4nx/q4_npu_eXpress.cpp
///       ../q4nx/safe_tensors.cpp
///       -L../../../../src/lib/xrt -L../../build/lib -L/opt/xilinx/xrt/lib
///       -lxrt_coreutil -laiebu -luuid -lpthread -o /tmp/verify_n4_prefill_attn
///
/// ---------------------------------------------------------------------------
/// What is under test
/// ---------------------------------------------------------------------------
/// `attn.xclbin` (FLM_Xclbin/Qwen3_8/gated_attention_prefill) computes ONLY
///
///     o = softmax(q K^T / sqrt(d) + causal) V
///
/// -- no projections, no norms, no RoPE, no gate, no MLP. So the suite is in
/// two parts, and they fail for different reasons:
///
///   Part A, checks 1-7  the core alone, on synthetic q/k/v. No model, no
///                       weights, no second bitstream. This is where addressing
///                       bugs live: the head <-> CU <-> kv-stream mapping, the
///                       q BO's origin at L_begin rather than 0, the v plane's
///                       base at plane_rows(), and the top-of-buffer guard.
///   Part B, checks 8-9  the hybrid prefill layer -- cpu_ref's own
///                       project_qkv_rope() and gate_out_mlp() around one core
///                       dispatch -- against the unsplit cpu_ref::forward(),
///                       and then a verify-bitstream pass on the cache the
///                       prefill path just wrote.
///
/// Part A's inputs are NOT N(0,1). Independent Gaussian q and k put ~99.98% of
/// the softmax mass on a single key once 1/sqrt(256) has spread the logits,
/// which makes the replayed context worth approximately nothing and the test
/// blind to almost every arithmetic bug. Each token instead gets a low-rank
/// "topic" code shared between q and k, so keys within a topic genuinely
/// compete. The generator is host.cpp's, unchanged, and the effective-key
/// count is printed and gated -- below ~4 the comparison means nothing and the
/// suite says so rather than passing.
///
/// Matching our own reference is also not enough on its own: a consistent
/// permutation of the head mapping agrees with itself. Check 3 scores a
/// reference computed with a deliberately rotated kv-head assignment and
/// requires the NPU to agree with it MUCH less well.
#include "prefill/npu/prefill_attn_op.hpp"
#include "verify/npu/attn_layer.hpp"
#include "verify/npu/attn_weights.hpp"
#include "common/cpu_ref_attention_layer.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/rope.hpp"

#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kLayer = 3;      ///< the first full attention layer: (3+1)%4 == 0
constexpr int kMaxL  = 1024;   ///< the layout's MAX_L for this suite

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-42s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

double cosine(const float* a, const float* b, size_t n) {
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; i++) {
        num += (double)a[i] * b[i];
        da  += (double)a[i] * a[i];
        db  += (double)b[i] * b[i];
    }
    if (da == 0 || db == 0) return 0.0;
    return num / (std::sqrt(da) * std::sqrt(db));
}

double report_cos(const char* tag, double cos, double gate, const char* note = "") {
    char buf[200];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f) %s", cos, gate, note);
    report(tag, cos >= gate, buf);
    return cos;
}

// ==========================================================================
// the CPU reference for the core alone
// ==========================================================================
/// `o[t] = softmax(q[t] K[0 .. L_begin+t]^T * scale) V`, fp32, GQA 1:6.
///
/// Deliberately a local reimplementation rather than a call into
/// cpu_ref::attention_layer: at this point there is no layer, no cache object
/// and no weights, only three arrays. It is the same six lines the reference
/// layer's attn_core() runs, and Part B scores that one directly.
///
/// \param eff_keys  out, mean exp(entropy) of the softmax rows. 1.0 means the
///        distribution collapsed onto one key and the comparison is vacuous.
void gqa_ref(const float* q, const float* k, const float* v, float* o,
             int L_begin, int L, double* eff_keys) {
    using namespace cpu_ref::q38;
    const int d = HEAD_DIM, H = NUM_ATTN_HEADS, KVH = NUM_KV_HEADS;
    const int DQ_ = d * H, DKV_ = d * KVH, group = H / KVH;
    double ent_sum = 0.0;
    long   ent_n   = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(+:ent_sum,ent_n)
    for (int t = 0; t < L; t++) {
        for (int h = 0; h < H; h++) {
            const int kvh  = h / group;
            const int nkey = L_begin + t + 1;
            const float* qv = q + (size_t)t * DQ_ + (size_t)h * d;
            std::vector<float> s(nkey);
            float smax = -INFINITY;
            for (int j = 0; j < nkey; j++) {
                const float* kv = k + (size_t)j * DKV_ + (size_t)kvh * d;
                float acc = 0.0f;
                for (int i = 0; i < d; i++) acc += qv[i] * kv[i];
                s[j] = acc * ATTN_SCALE;
                smax = std::max(smax, s[j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < nkey; j++) { s[j] = std::exp(s[j] - smax); sum += s[j]; }
            const float inv = 1.0f / sum;
            float* out = o + (size_t)t * DQ_ + (size_t)h * d;
            for (int i = 0; i < d; i++) out[i] = 0.0f;
            double ent = 0.0;
            for (int j = 0; j < nkey; j++) {
                const float p = s[j] * inv;
                if (p > 0.0f) ent -= (double)p * std::log((double)p);
                if (p == 0.0f) continue;
                const float* vv = v + (size_t)j * DKV_ + (size_t)kvh * d;
                for (int i = 0; i < d; i++) out[i] += p * vv[i];
            }
            ent_sum += std::exp(ent);
            ent_n++;
        }
    }
    if (eff_keys) *eff_keys = ent_n ? ent_sum / (double)ent_n : 0.0;
}

// ==========================================================================
/// host.cpp's low-rank topic-code generator, unchanged in substance.
struct topic_inputs {
    std::vector<float> q, k, v;   // [rows][DQ], [rows][DKV], [rows][DKV]
};

topic_inputs make_topic_inputs(int rows, unsigned seed) {
    using namespace cpu_ref::q38;
    const int d = HEAD_DIM, H = NUM_ATTN_HEADS, KVH = NUM_KV_HEADS;
    const int DQ_ = d * H, DKV_ = d * KVH;
    const int RANK = 8;
    const float sig_amp = 0.06f;    // keeps logits O(1) after the 1/sqrt(d) scale
    const float noise_amp = 0.15f;

    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);

    std::vector<float> topic((size_t)rows * RANK);
    for (auto& x : topic) x = g(rng);
    std::vector<float> qb((size_t)H * RANK * d), kb((size_t)KVH * RANK * d);
    for (auto& x : qb) x = g(rng) * 0.25f;
    for (auto& x : kb) x = g(rng) * 0.25f;

    topic_inputs in;
    in.q.assign((size_t)rows * DQ_, 0.0f);
    in.k.assign((size_t)rows * DKV_, 0.0f);
    in.v.assign((size_t)rows * DKV_, 0.0f);

    for (int t = 0; t < rows; t++) {
        const float* tc = topic.data() + (size_t)t * RANK;
        for (int h = 0; h < H; h++) {
            float* dst = in.q.data() + (size_t)t * DQ_ + (size_t)h * d;
            const float* b = qb.data() + (size_t)h * RANK * d;
            for (int i = 0; i < d; i++) {
                float acc = 0.0f;
                for (int r = 0; r < RANK; r++) acc += tc[r] * b[(size_t)r * d + i];
                dst[i] = cpu_ref::round_bf16(sig_amp * acc + noise_amp * g(rng));
            }
        }
        for (int h = 0; h < KVH; h++) {
            float* kd = in.k.data() + (size_t)t * DKV_ + (size_t)h * d;
            float* vd = in.v.data() + (size_t)t * DKV_ + (size_t)h * d;
            const float* b = kb.data() + (size_t)h * RANK * d;
            for (int i = 0; i < d; i++) {
                float acc = 0.0f;
                for (int r = 0; r < RANK; r++) acc += tc[r] * b[(size_t)r * d + i];
                kd[i] = cpu_ref::round_bf16(sig_amp * acc + noise_amp * g(rng));
                vd[i] = cpu_ref::round_bf16(0.5f * g(rng));
            }
        }
    }
    return in;
}

/// Fill a cache BO's two planes from fp32 k/v rows, through the layout's own
/// accessors -- so a layout change moves the seeder with every descriptor.
void seed_cache(buffer<bf16>& cache, const q38npu::attn_layout& t,
                const std::vector<float>& k, const std::vector<float>& v, int rows) {
    const size_t DKV_ = (size_t)t.DK;
    cache.memset((bf16)0);
    for (int r = 0; r < rows; r++) {
        bf16* kp = cache.data() + t.k_at(r);
        bf16* vp = cache.data() + t.v_at(r);
        for (size_t i = 0; i < DKV_; i++) kp[i] = (bf16)k[(size_t)r * DKV_ + i];
        for (size_t i = 0; i < DKV_; i++) vp[i] = (bf16)v[(size_t)r * DKV_ + i];
    }
    cache.sync_to_device();
}

/// The kv rows a pass at (L_begin, L_pad) actually streams, from the sequence
/// generator's own arithmetic. Check 7 reads this rather than inferring it.
int kv_reach(int L_begin, int L_pad) {
    const int C = q38npu::prefill_attn_tiles::Q_CHUNK;
    const int Lq_last = L_begin + L_pad - C;
    return (Lq_last + 2 * C - 1) / C * C;
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string prefill_xclbin =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_prefill/build/xclbins/attn.xclbin";
    const std::string verify_xclbin =
        argc > 3 ? argv[3]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_verify/build/QWEN3_8_27B/xclbins/layer.xclbin";

    using namespace cpu_ref::q38;
    const int d = HEAD_DIM, H = NUM_ATTN_HEADS, KVH = NUM_KV_HEADS;
    const int DQ_ = d * H, DKV_ = d * KVH, D = MODEL_DIM;
    const int C = q38npu::prefill_attn_op::CHUNK;

    std::printf("== N4: the prefill attention core on the NPU ==\n");
    std::printf("   model    %s\n   prefill  %s\n   verify   %s\n\n",
                model_dir.c_str(), prefill_xclbin.c_str(), verify_xclbin.c_str());

    const q38npu::attn_layout t = q38npu::attn_layout::build(kMaxL, /*L=*/8);
    std::printf("   D=%d DQ=%d DKV=%d  MAX_L=%d = %zu plane rows (no apron), "
                "KV BO %.1f MB\n\n",
                t.D, t.DQ, t.DK, t.MAX_L, t.plane_rows(),
                t.kv_cache_size * sizeof(bf16) / 1048576.0);

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);

    // ======================================================================
    // 1. the RTP addresses
    // ======================================================================
    // Per-build, and a stale one does not fault: it writes L_begin / Q_chunks
    // into some other buffer, the cores never see them, and the dispatch hangs
    // until ERT_CMD_STATE_TIMEOUT. If this suite times out below, diff
    // attn.mlir.prj/input_with_addresses.mlir against these three.
    const q38npu::prefill_attn_rtp_addresses rtp;
    {
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "L_begin=%u L_chunks=%u window=%u (from "
                      "input_with_addresses.mlir)",
                      rtp.l_begin, rtp.l_chunks, rtp.window_size);
        report("rtp addresses", rtp.l_begin == 61568 && rtp.l_chunks == 11904 &&
                                rtp.window_size == 61600, buf);
    }

    npu_app_manager* pre_mgr = mgr.register_xclbin(prefill_xclbin);
    q38npu::prefill_attn_op op(pre_mgr->create_app(), t, rtp);

    // A cache BO from an app that is NOT the prefill op's, on purpose: this is
    // exactly what the production path does -- npu_attn_layer allocates the
    // cache out of the verify xclbin's app and hands it here. BOs are
    // device-scoped, so it is legal; if it were not, every dispatch below
    // would hang rather than fault, which is why it is worth stating.
    buffer<bf16> cache = pre_mgr->create_app().create_bo_buffer<bf16>(t.kv_cache_size);

    // ======================================================================
    // 2. the attention core against its own reference
    // ======================================================================
    std::printf("\n-- the core alone: synthetic topic-code q/k/v --\n");

    const int rows = t.MAX_L;
    const topic_inputs in = make_topic_inputs(rows, /*seed=*/32);

    double base_cos = 0.0;
    std::vector<float> o_npu, o_ref;
    {
        const int L_begin = 0, L = 256;
        seed_cache(cache, t, in.k, in.v, L_begin + L);
        o_npu.assign((size_t)L * DQ_, 0.0f);
        o_ref.assign((size_t)L * DQ_, 0.0f);
        op.run(in.q.data() + (size_t)L_begin * DQ_, L, L_begin, cache, o_npu.data());

        double eff = 0.0;
        gqa_ref(in.q.data() + (size_t)L_begin * DQ_, in.k.data(), in.v.data(),
                o_ref.data(), L_begin, L, &eff);

        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "mean exp(entropy) = %.2f keys  (1.0 == degenerate)", eff);
        report("softmax is not collapsed", eff >= 4.0, buf);

        // The online softmax accumulates in bf16, so this is a loose bound on
        // purpose; the structural checks below are what establish correctness.
        base_cos = report_cos("core vs CPU, L_begin=0 L=256",
                              cosine(o_npu.data(), o_ref.data(), o_npu.size()), 0.95);
    }

    // ======================================================================
    // 3. per-head cosine, and the rotated-kv cross-head control
    // ======================================================================
    {
        const int L = 256;
        double worst = 1.0; int worst_h = -1;
        std::printf("        per-head cosine:");
        for (int h = 0; h < H; h++) {
            double dot = 0, na = 0, nb = 0;
            for (int tt = 0; tt < L; tt++) {
                const size_t b = (size_t)tt * DQ_ + (size_t)h * d;
                for (int i = 0; i < d; i++) {
                    const double a = o_npu[b + i], r = o_ref[b + i];
                    dot += a * r; na += a * a; nb += r * r;
                }
            }
            const double c = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
            if (h % 8 == 0) std::printf("\n         ");
            std::printf(" h%02d=%.4f", h, c);
            if (c < worst) { worst = c; worst_h = h; }
        }
        std::printf("\n");
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "worst h%d at %.5f, aggregate %.5f -- an outlier means the "
                      "head/kv-stream mapping, a uniform sag means accumulation",
                      worst_h, worst, base_cos);
        report("no head is an outlier", worst >= base_cos - 0.05, buf);

        // The control. Recompute the reference with the kv-head assignment
        // rotated by one; the device must agree with it MUCH less well. A
        // permutation bug in the mapping would agree with the rotated
        // reference just as well as with the real one.
        std::vector<float> k_rot(in.k.size()), v_rot(in.v.size());
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < KVH; h++) {
                const int src = (h + 1) % KVH;
                std::memcpy(k_rot.data() + (size_t)r * DKV_ + (size_t)h * d,
                            in.k.data() + (size_t)r * DKV_ + (size_t)src * d,
                            d * sizeof(float));
                std::memcpy(v_rot.data() + (size_t)r * DKV_ + (size_t)h * d,
                            in.v.data() + (size_t)r * DKV_ + (size_t)src * d,
                            d * sizeof(float));
            }
        std::vector<float> o_ctl((size_t)L * DQ_);
        gqa_ref(in.q.data(), k_rot.data(), v_rot.data(), o_ctl.data(), 0, L, nullptr);
        const double ctl = cosine(o_npu.data(), o_ctl.data(), o_npu.size());
        std::snprintf(buf, sizeof buf,
                      "real %.5f vs rotated-kv %.5f -- the control must be MUCH worse",
                      base_cos, ctl);
        report("cross-head control separates", base_cos > ctl + 0.05, buf);
    }

    // ======================================================================
    // 4. per-segment cosine by absolute token
    // ======================================================================
    // Accuracy falls smoothly with causal depth because the online softmax
    // accumulates in bf16. Printing it by segment is what separates "deep rows
    // are legitimately noisier" from "one block of rows is addressed wrong".
    {
        const int L = 256;
        double worst = 1.0; int worst_s = -1;
        std::printf("        per-segment cosine (absolute token):");
        for (int s0 = 0; s0 < L; s0 += C) {
            const int s1 = std::min(s0 + C, L);
            const double c = cosine(o_npu.data() + (size_t)s0 * DQ_,
                                    o_ref.data() + (size_t)s0 * DQ_,
                                    (size_t)(s1 - s0) * DQ_);
            std::printf("  [%d,%d)=%.5f", s0, s1, c);
            if (c < worst) { worst = c; worst_s = s0; }
        }
        std::printf("\n");
        char buf[160];
        std::snprintf(buf, sizeof buf, "worst segment [%d,%d) at %.5f",
                      worst_s, worst_s + C, worst);
        report("every segment clears the gate", worst >= 0.95, buf);
    }

    // ======================================================================
    // 5. the L_begin sweep
    // ======================================================================
    // 0 replays nothing -- the whole cache the pass reads is what it was
    // handed, and a design that ignored L_begin entirely would score the same.
    // 22 is ragged on purpose: attn.cc masks on ABSOLUTE position, so an
    // unaligned cursor is legal, and a pass that works at 0 and fails at 22 is
    // addressing and not math. 250 is ragged and deep.
    std::printf("\n-- L_begin sweep --\n");
    for (int L_begin : {0, 22, 128, 250}) {
        const int L = 256;
        if (L_begin + L > t.MAX_L) continue;
        seed_cache(cache, t, in.k, in.v, L_begin + L);
        std::vector<float> a((size_t)L * DQ_), r((size_t)L * DQ_);
        op.run(in.q.data() + (size_t)L_begin * DQ_, L, L_begin, cache, a.data());
        gqa_ref(in.q.data() + (size_t)L_begin * DQ_, in.k.data(), in.v.data(),
                r.data(), L_begin, L, nullptr);
        char tag[64];
        std::snprintf(tag, sizeof tag, "L_begin = %d", L_begin);
        char note[96];
        std::snprintf(note, sizeof note, "-- %d keys at the deepest row",
                      L_begin + L);
        report_cos(tag, cosine(a.data(), r.data(), a.size()), 0.95, note);
    }

    // ======================================================================
    // 6. padding
    // ======================================================================
    // run() rounds the window up to a chunk by repeating the last real row. Two
    // things have to hold: the short call must equal the hand-built padded one
    // exactly, and the pad rows must not be able to move the rows below them.
    // The second is the one that matters -- it is the causal mask, tested.
    std::printf("\n-- padding --\n");
    {
        const int L_begin = 64, L = 100;
        seed_cache(cache, t, in.k, in.v, L_begin + L);
        const float* qsrc = in.q.data() + (size_t)L_begin * DQ_;

        std::vector<float> a((size_t)L * DQ_);
        op.run(qsrc, L, L_begin, cache, a.data());

        // hand-built: rows [L, C) repeat row L-1, exactly what run() does
        std::vector<float> q_rep((size_t)C * DQ_);
        std::memcpy(q_rep.data(), qsrc, (size_t)L * DQ_ * sizeof(float));
        for (int r = L; r < C; r++)
            std::memcpy(q_rep.data() + (size_t)r * DQ_,
                        qsrc + (size_t)(L - 1) * DQ_, (size_t)DQ_ * sizeof(float));
        std::vector<float> b((size_t)C * DQ_);
        op.run(q_rep.data(), C, L_begin, cache, b.data());
        report_cos("L=100 is the hand-padded L=128",
                   cosine(a.data(), b.data(), (size_t)L * DQ_), 1.0 - 1e-9,
                   "-- must be 1.0 exactly");

        // and with zeros in the tail instead: rows below must not move.
        std::vector<float> q_zero((size_t)C * DQ_, 0.0f);
        std::memcpy(q_zero.data(), qsrc, (size_t)L * DQ_ * sizeof(float));
        std::vector<float> c2((size_t)C * DQ_);
        op.run(q_zero.data(), C, L_begin, cache, c2.data());
        report_cos("pad rows cannot reach rows below them",
                   cosine(a.data(), c2.data(), (size_t)L * DQ_), 1.0 - 1e-9,
                   "-- must be 1.0 exactly");
    }

    // ======================================================================
    // 7. the top of the buffer
    // ======================================================================
    // **This is where the apron used to be.** The kv broadcast walks whole
    // 128-row blocks and overshoots the query chunk by one, so a pass reads
    // through `round_up(L_begin + round_up(L, 128), 128)` -- past the cursor,
    // and with no apron above MAX_L, potentially past the BO. That read lands
    // in the next layer's cache, silently: a DDR read past a BO does not fault
    // here. It cannot be clamped, either, because attn.cc derives the same
    // block count itself and a short BD would strand blocks for the next
    // dispatch. So the overrun is made UNREACHABLE instead: run() guards the
    // PADDED window against MAX_L, and MAX_L is a whole number of chunks so
    // the round-up is a fixed point at the top.
    //
    // Two halves, and both are needed. The scan says every pass the guard
    // admits stays inside the plane; the boundary pair says the guard is on
    // the right row and not merely conservative.
    std::printf("\n-- the top of the buffer --\n");
    {
        int worst = 0, worst_begin = 0;
        for (int L_begin = 0; L_begin < t.MAX_L; L_begin++) {
            for (int L : {1, 7, 128, 129, 256}) {
                const int L_pad = (L + C - 1) / C * C;
                if (L_begin + L_pad > t.MAX_L) continue;   // run() refuses it
                const int reach = kv_reach(L_begin, L_pad);
                if (reach > worst) { worst = reach; worst_begin = L_begin; }
            }
        }
        char buf[190];
        std::snprintf(buf, sizeof buf,
                      "deepest reach %d rows (at L_begin %d), plane holds %zu "
                      "= MAX_L %d, no apron",
                      worst, worst_begin, t.plane_rows(), t.MAX_L);
        report("every pass the guard admits stays inside the plane",
               (size_t)worst <= t.plane_rows(), buf);

        // And run the deepest one, so the arithmetic is not the only witness.
        const int L = C, L_begin = t.MAX_L - L;
        seed_cache(cache, t, in.k, in.v, t.MAX_L);
        std::vector<float> a((size_t)L * DQ_), r((size_t)L * DQ_);
        op.run(in.q.data() + (size_t)L_begin * DQ_, L, L_begin, cache, a.data());
        gqa_ref(in.q.data() + (size_t)L_begin * DQ_, in.k.data(), in.v.data(),
                r.data(), L_begin, L, nullptr);
        char note[96];
        std::snprintf(note, sizeof note, "-- L_begin %d, reach %d", L_begin,
                      kv_reach(L_begin, L));
        report_cos("the deepest legal pass runs",
                   cosine(a.data(), r.data(), a.size()), 0.95, note);

        // The boundary, to the row. One query at MAX_L - 128 pads to a whole
        // chunk ending exactly at MAX_L and must run; the same query one row
        // higher pads PAST it and must be refused rather than absorbed. A
        // guard written against the unpadded L would accept both, and the
        // second would read 127 rows of the next layer's cache.
        std::vector<float> one(DQ_), one_o(DQ_);
        std::memcpy(one.data(), in.q.data() + (size_t)(t.MAX_L - C) * DQ_,
                    (size_t)DQ_ * sizeof(float));
        bool last_ok = true, past_threw = false;
        try { op.run(one.data(), 1, t.MAX_L - C, cache, one_o.data()); }
        catch (const std::exception&) { last_ok = false; }
        try { op.run(one.data(), 1, t.MAX_L - C + 1, cache, one_o.data()); }
        catch (const std::exception&) { past_threw = true; }
        std::snprintf(buf, sizeof buf,
                      "L_begin %d ran %s, L_begin %d %s",
                      t.MAX_L - C, last_ok ? "ok" : "THREW", t.MAX_L - C + 1,
                      past_threw ? "threw" : "RAN -- it pads past the plane");
        report("the padded-window guard is on the right row",
               last_ok && past_threw, buf);
    }

    // ======================================================================
    // Part B: the hybrid prefill layer
    // ======================================================================
    std::printf("\n-- the hybrid layer: cpu_ref's halves around one dispatch --\n");

    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");

    char key[64];
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", kLayer - 1);
    std::vector<float> x_all;
    try {
        x_all = ref.f32(key);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ref.safetensors: %s\n", e.what());
        return 1;
    }
    const int LP = (int)(x_all.size() / D);          // 221
    // Round to what the array will receive: letting the reference start from
    // extra precision charges the device for a narrowing it cannot avoid.
    for (float& f : x_all) f = cpu_ref::round_bf16(f);

    std::vector<float> cos_all, sin_all;
    cpu_ref::text_rope_tables(0, LP + 8, cos_all, sin_all);

    npu_app_manager* ver_mgr = mgr.register_xclbin(verify_xclbin);
    q38npu::attn_engine eng(ver_mgr->create_app(), t, q38npu::attn_rtp_addresses{});
    q38npu::npu_attn_layer dev_layer(&eng, q4nx, kLayer);

    cpu_ref::attention_layer cpu(q4nx, kLayer, t.MAX_L);
    // max_len = 0: the weights, none of the cache. Only the two const halves
    // are ever called on it; forward() would throw, which is the intent.
    cpu_ref::attention_layer half(q4nx, kLayer, /*max_len=*/0);

    // ---- 8. the whole layer ------------------------------------------------
    std::vector<float> y_ref((size_t)LP * D), y_hyb((size_t)LP * D);
    cpu.forward(x_all.data(), LP, cos_all.data(), sin_all.data(), y_ref.data());

    {
        std::vector<float> qg, kk, vv, qh;
        half.project_qkv_rope(x_all.data(), LP, cos_all.data(), sin_all.data(),
                              qg, kk, vv, qh);
        dev_layer.append_rows(kk.data(), vv.data(), LP);
        std::vector<float> attn((size_t)LP * DQ_);
        op.run(qh.data(), LP, /*L_begin=*/0, dev_layer.cache(), attn.data());
        half.gate_out_mlp(qg.data(), attn.data(), x_all.data(), LP, y_hyb.data());

        char note[96];
        std::snprintf(note, sizeof note, "-- L = %d in ONE dispatch (tiled: %d)",
                      LP, (LP + 7) / 8);
        report_cos("hybrid layer vs cpu_ref::forward",
                   cosine(y_hyb.data(), y_ref.data(), y_hyb.size()), 0.999, note);

        // The k/v that landed in the BO, against the CPU's own cache. Scored
        // separately because the layer output cannot see a v-plane base error
        // clearly -- each query's own row dominates its softmax, so an
        // addressing error in the cache is diluted by the time it reaches y.
        const std::vector<float> k_dev = dev_layer.read_k(0, LP);
        const std::vector<float> v_dev = dev_layer.read_v(0, LP);
        report_cos("k rows in the BO match the CPU cache",
                   cosine(k_dev.data(), cpu.cache_k().data(), k_dev.size()),
                   0.9999, "-- the v plane's base is MAX_L * DK");
        report_cos("v rows in the BO match the CPU cache",
                   cosine(v_dev.data(), cpu.cache_v().data(), v_dev.size()),
                   0.9999, "");
    }

    // ---- 9. cache continuity ----------------------------------------------
    // The integration gate. The prefill path wrote CPU-computed rows through
    // append_rows; a verify pass now reads them through the OTHER bitstream's
    // replay, on the same BO and the same cursor. If the two disagree about
    // the layout -- the plane base, the apron, the cursor -- this is where it
    // shows, and nowhere earlier.
    std::printf("\n-- cache continuity: prefill wrote, verify reads --\n");
    {
        char buf[160];
        std::snprintf(buf, sizeof buf, "device %d, cpu %d",
                      dev_layer.tokens_seen(), cpu.cache_len());
        report("cursors agree after the prefill window",
               dev_layer.tokens_seen() == cpu.cache_len() &&
               dev_layer.tokens_seen() == LP, buf);

        std::snprintf(key, sizeof key, "verify.layer_%02d_out", kLayer - 1);
        std::vector<float> xv;
        try {
            xv = ref.f32(key);
        } catch (const std::exception&) {
            xv.assign(x_all.end() - (size_t)8 * D, x_all.end());
        }
        for (float& f : xv) f = cpu_ref::round_bf16(f);

        const float* cv = cos_all.data() + (size_t)LP * ROPE_DIM;
        const float* sv = sin_all.data() + (size_t)LP * ROPE_DIM;

        std::vector<float> yv_ref((size_t)8 * D), yv_dev((size_t)8 * D);
        cpu.forward(xv.data(), 8, cv, sv, yv_ref.data());
        dev_layer.forward(xv.data(), 8, cv, sv, yv_dev.data());

        report_cos("an L=8 verify pass on the prefilled cache",
                   cosine(yv_dev.data(), yv_ref.data(), yv_dev.size()), 0.99,
                   "-- the other bitstream, the same BO");
        std::snprintf(buf, sizeof buf, "device %d, cpu %d (want %d)",
                      dev_layer.tokens_seen(), cpu.cache_len(), LP + 8);
        report("cursors agree after it", dev_layer.tokens_seen() == LP + 8 &&
                                         cpu.cache_len() == LP + 8, buf);
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
