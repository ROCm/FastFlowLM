// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lfm3_npu_detail.hpp"
#include <algorithm>
#include <chrono>

using namespace lfm3_cpu;

/// LFM3_PROFILE=1: per-stage wall time of the NPU prefill, printed after every prefill call
namespace {
struct stage_timer {
    bool on = std::getenv("LFM3_PROFILE") != nullptr;
    std::vector<std::pair<std::string, double>> acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void tick(const char* name) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - t0).count();
        t0 = now;
        for (auto& a : acc) if (a.first == name) { a.second += ms; return; }
        acc.emplace_back(name, ms);
    }
    void print(int tokens) {
        if (!on) return;
        double tot = 0;
        for (auto& a : acc) tot += a.second;
        std::fprintf(stderr, "prefill profile (%d tokens, %.0f ms accounted):", tokens, tot);
        for (auto& a : acc) std::fprintf(stderr, "  %s %.0f", a.first.c_str(), a.second);
        std::fprintf(stderr, "\n");
    }
};
}

/// Prefill of `ids` on the NPU stages (see lfm3_prefill.hpp), in passes of at most PASS_TOKENS tokens.
buffer<bf16> lfm3_npu::Impl::prefill_npu(std::vector<int>& ids) {
    buffer<bf16> logits;
    const int total = (int)ids.size();
    for (int begin = 0; begin < total; begin += lfm3_prefill_ctx::PASS_TOKENS) {
        const int n = std::min(lfm3_prefill_ctx::PASS_TOKENS, total - begin);
        const bool last = begin + n == total;
        buffer<bf16> r = prefill_npu_pass(ids.data() + begin, n, last);
        if (last) logits = std::move(r);   // move: copy-assigning a buffer makes a non-owning view (XRT build)
    }
    return logits;
}

buffer<bf16> lfm3_npu::Impl::prefill_npu_pass(const int* ids, int n, bool want_logits) {
    if (!weights_loaded) throw std::runtime_error("lfm3_npu: weights are not loaded");
    if ((uint32_t)(ctx + n) > MAX_L) throw std::runtime_error("lfm3_npu: context would exceed MAX_L (" + std::to_string(MAX_L) + ")");
    lfm3_prefill_ctx& P = *pctx;
    static stage_timer T;
    T.t0 = std::chrono::steady_clock::now();
    const int D = desc.D, DQ = desc.DQ, DK = desc.DK, nh = desc.num_heads, I = desc.intermediate_size;
    const lfm3_prefill_shape s = P.shape(n, ctx);
    P.setup(s);
    const int r0 = s.L_offset;                                   // first row of the new tokens
    auto rows = [&](buffer<bf16>& b, int width) { return u16p(b.data()) + (size_t)r0 * width; };

    // rows outside the new tokens are padding: zero them (a previous, longer pass may have left values there)
    auto zero_padding = [&](buffer<bf16>& b, int width) {
        std::memset(b.data(), 0, (size_t)r0 * width * sizeof(bf16));
        std::memset(b.data() + (size_t)(r0 + n) * width, 0, (size_t)(s.L_padded - r0 - n) * width * sizeof(bf16));
    };
    zero_padding(P.x, D);
    zero_padding(P.mixer_out, DQ);
    zero_padding(P.o, D);
    P.o.sync_to_device();
    std::fill(P.residual.begin(), P.residual.end(), 0);

    for (int t = 0; t < n; t++) {
        if (ids[t] < 0 || (uint32_t)ids[t] >= desc.vocab_size) throw std::runtime_error("lfm3_npu: token id out of range");
        std::memcpy(&P.residual[(size_t)(r0 + t) * D], u16p(embedding.w.data()) + (size_t)ids[t] * D, (size_t)D * sizeof(uint16_t));
    }
    uint16_t* res = &P.residual[(size_t)r0 * D];                 // residual rows of the new tokens

    // LFM3_VERIFY=1: compare every NPU gemm with the host matmul on the same input rows (first layers only)
    static const bool verify = std::getenv("LFM3_VERIFY") != nullptr;
    auto check = [&](const char* what, int li, const lfm3_cpu::qmat& W, const uint16_t* in, int ldin, const uint16_t* npu_out, int ldout) {
        if (!verify || li > 3) return;
        std::vector<uint16_t> ref((size_t)n * W.rows);
        matmul(ref.data(), W.rows, W, in, ldin, n);
        double dot = 0, na = 0, nb = 0;
        for (int t = 0; t < n; t++)
            for (int c = 0; c < W.rows; c++) {
                const double a = bf2f(ref[(size_t)t * W.rows + c]), b = bf2f(npu_out[(size_t)t * ldout + c]);
                dot += a * b; na += a * a; nb += b * b;
            }
        std::fprintf(stderr, "verify layer %2d %-10s cos=%.6f\n", li, what, dot / std::sqrt(na * nb + 1e-30));
    };

    for (uint32_t li = 0; li < desc.num_hidden_layers; li++) {
        lfm3_layer_weights& w = layers[li];
        const bool attn = w.is_attn();

        // 1. dequantize this layer's weights while the host computes the operator norm
        T.tick("other");
        P.dequant_start(w.proj, attn);
        uint16_t* xr = rows(P.x, D);
        for (int t = 0; t < n; t++) rms_norm_vec(&xr[(size_t)t * D], &res[(size_t)t * D], u16p(w.operator_norm.data()), D, desc.eps);
        P.x.sync_to_device();
        T.tick("norm1(host)");
        P.dequant_wait();
        T.tick("dequant_wait");

        // 2. the mixer
        if (attn) {
            P.q_proj(P.q, P.x, P.w_slot0);
            P.k_proj(P.k, P.x, P.w_slot0);
            P.v_proj(P.v, P.x, P.w_slot0);
            P.q.sync_from_device();
            P.k.sync_from_device();
            P.v.sync_from_device();
            T.tick("qkv_gemm");
            check("q_proj", li, w.q_proj, xr, D, rows(P.q, DQ), DQ);
            check("k_proj", li, w.k_proj, xr, D, rows(P.k, DK), DK);
            check("v_proj", li, w.v_proj, xr, D, rows(P.v, DK), DK);
            // gate = bf16(sigmoid(bf16(h . Wg))), from the same normed input
            std::vector<uint16_t> g((size_t)n * nh);
            matmul_dense(g.data(), nh, u16p(w.gate_proj.data()), nh, D, xr, D, n);
            for (auto& v : g) v = f2bf(sigmoidf(bf2f(v)));
            uint16_t* qr = rows(P.q, DQ);
            attn_norm_rope_cache(li, w, qr, rows(P.k, DK), rows(P.v, DK), n, s.L_begin);
            P.q.sync_to_device();                               // q and k were modified in place: flush the dirty lines now
            P.k.sync_to_device();
            T.tick("gate+rope(host)");
            const bool on_npu = w.kind == lfm3_block_kind::sliding_attention ? stage_attn_swa : stage_attn_full;
            uint16_t* mr = rows(P.mixer_out, DQ);
            if (on_npu) {
                P.q.sync_to_device();
                kv_cache[li].sync_to_device();
                const int window = w.kind == lfm3_block_kind::sliding_attention ? (int)desc.sliding_window : 0;
                npu_app& app = P.attention_app(s, window);
                app(P.mixer_out, P.q, kv_cache[li]);
                P.mixer_out.sync_from_device();
                for (int t = 0; t < n; t++)                         // per-head output gate
                    for (int hd = 0; hd < nh; hd++) {
                        const float gv = bf2f(g[(size_t)t * nh + hd]);
                        uint16_t* p = &mr[(size_t)t * DQ + (size_t)hd * desc.DH];
                        for (int d = 0; d < (int)desc.DH; d++) p[d] = f2bf(bf2f(p[d]) * gv);
                    }
                if (verify && li <= 22) {   // the kernel's attention (after the gate) against the host attention on the same q / cache
                    std::vector<uint16_t> ref((size_t)n * DQ);
                    attn_host_core(li, w, qr, g.data(), ref.data(), n, s.L_begin);
                    double dot = 0, na = 0, nb = 0, worst = 0;
                    for (size_t i = 0; i < ref.size(); i++) {
                        const double a = bf2f(ref[i]), b = bf2f(mr[i]);
                        dot += a * b; na += a * a; nb += b * b; worst = std::max(worst, std::fabs(a - b));
                    }
                    std::fprintf(stderr, "verify layer %2d attn(%s) cos=%.6f maxabs=%.4f\n", li,
                                 window ? "swa " : "full", dot / std::sqrt(na * nb + 1e-30), worst);
                }
            } else {
                attn_host_core(li, w, qr, g.data(), mr, n, s.L_begin);
            }
            P.mixer_out.sync_to_device();
            T.tick("attention");
            P.o_proj(P.o, P.mixer_out, P.w_o);
        } else {
            P.conv_in_proj(P.bcx, P.x, P.w_slot0);
            P.bcx.sync_from_device();
            T.tick("conv_in_gemm");
            check("conv_in", li, w.conv_in, xr, D, rows(P.bcx, 3 * D), 3 * D);
            conv_core(li, w, rows(P.bcx, 3 * D), rows(P.mixer_out, DQ), n);
            P.mixer_out.sync_to_device();
            T.tick("conv(host)");
            P.o_proj(P.o, P.mixer_out, P.w_o);
        }
        P.o.sync_from_device();
        T.tick("o_gemm");
        check("out/o_proj", li, attn ? w.o_proj : w.conv_out, rows(P.mixer_out, DQ), DQ, rows(P.o, D), D);
        if (verify && li >= 1 && li <= 4) {   // which layer's weights did the kernel really use?
            for (int cand = 0; cand <= li; cand++) {
                const lfm3_cpu::qmat& W = layers[cand].is_attn() ? layers[cand].o_proj : layers[cand].conv_out;
                char name[32]; std::snprintf(name, sizeof name, "o_w[L%d]", cand);
                check(name, 0, W, rows(P.mixer_out, DQ), DQ, rows(P.o, D), D);
            }
        }
        // Host work on a device-written buffer goes to a scratch copy, never in place: a dirty host cache line left in P.o would be
        // written back over the next kernel's output at the following sync_from_device().
        std::vector<uint16_t> post((size_t)n * D);
        const uint16_t* orow = rows(P.o, D);
        for (int t = 0; t < n; t++) rms_norm_vec(&post[(size_t)t * D], &orow[(size_t)t * D], u16p(w.post_operator_norm.data()), D, desc.eps);
        residual_add(res, post.data(), (size_t)n * D);
        T.tick("post1+res(host)");

        // 3. FFN
        for (int t = 0; t < n; t++) rms_norm_vec(&xr[(size_t)t * D], &res[(size_t)t * D], u16p(w.ffn_norm.data()), D, desc.eps);
        P.x.sync_to_device();
        P.w_up.sync_to_device();
        P.up_proj(P.up, P.x, P.w_up);
        P.up.sync_from_device();
        T.tick("up_gemm");
        check("up", li, w.w3, xr, D, rows(P.up, I), I);
        P.w_gate.sync_to_device();
        P.gate_proj(P.gate, P.x, P.w_gate);          // SiLU is applied in the kernel
        P.gate.sync_from_device();
        P.hid.sync_from_device();
        T.tick("gate_gemm");
        if (verify && li <= 3) {   // gate: SiLU(host gate matmul) against the kernel output
            std::vector<uint16_t> ref((size_t)n * I);
            matmul(ref.data(), I, w.w1, xr, D, n);
            double dot = 0, na = 0, nb = 0;
            for (int t = 0; t < n; t++) for (int c = 0; c < I; c++) {
                const double a = siluf(bf2f(ref[(size_t)t * I + c])), b = bf2f(rows(P.gate, I)[(size_t)t * I + c]);
                dot += a * b; na += a * a; nb += b * b;
            }
            std::fprintf(stderr, "verify layer %2d %-10s cos=%.6f\n", li, "gate+silu", dot / std::sqrt(na * nb + 1e-30));
        }
        {   // hid = bf16(gate * up) over the whole padded buffer, like the LFM2 engine
            const size_t cnt = (size_t)s.L_padded * I;
            uint16_t* h = u16p(P.hid.data());
            const uint16_t* gt = u16p(P.gate.data());
            const uint16_t* up = u16p(P.up.data());
            #pragma omp parallel for num_threads(host_threads()) schedule(static)
            for (long i = 0; i < (long)cnt; i++) h[i] = f2bf(bf2f(gt[i]) * bf2f(up[i]));
        }
        P.hid.sync_to_device();
        T.tick("gate*up(host)");
        P.w_down.sync_to_device();
        // these round trips look redundant, but dropping them corrupts what the gemm reads (carried over from the LFM2 engine)
        P.o.sync_from_device();
        P.o.sync_to_device();
        P.down_proj(P.o, P.hid, P.w_down);
        P.o.sync_from_device();
        T.tick("down_gemm");
        check("down", li, w.w2, rows(P.hid, I), I, rows(P.o, D), D);
        orow = rows(P.o, D);
        for (int t = 0; t < n; t++) rms_norm_vec(&post[(size_t)t * D], &orow[(size_t)t * D], u16p(w.post_ffn_norm.data()), D, desc.eps);
        residual_add(res, post.data(), (size_t)n * D);
        dump_layer(li, res, n);
        T.tick("post2+res(host)");
    }
    this->ctx += n;
    note_context();
    if (!want_logits) { T.print(n); return buffer<bf16>(); }

    std::vector<uint16_t> last(D);
    rms_norm_vec(last.data(), &res[(size_t)(n - 1) * D], u16p(final_norm.data()), D, desc.eps);
    buffer<bf16> logits(desc.vocab_size);
    matmul(u16p(logits.data()), desc.vocab_size, lm_head, last.data(), D, 1);
    T.tick("lm_head(host)");
    T.print(n);
    return logits;
}
