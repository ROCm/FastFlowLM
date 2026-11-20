// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_prefill.cpp
/// \brief Batched prefill of Hy-Memory-30B-A3B, see hy_mem_prefill.hpp
#include <chrono>
#include <iostream>
#include "hy_mem_prefill.hpp"

using namespace hy_mem_cpu;

namespace {
const bool time_host = std::getenv("HY_TIME_HOST") != nullptr;
// HY_MEM_ATTN_HOST / HY_MEM_MM_HOST = 1: that stage on the host instead (for comparisons)
const bool attn_forced_host = std::getenv("HY_MEM_ATTN_HOST") != nullptr;
const bool mm_forced_host = std::getenv("HY_MEM_MM_HOST") != nullptr;
// HY_MM_CHECK=1: every NPU dequant_mm also on the host, cosine printed
const int mm_check = std::getenv("HY_MM_CHECK") ? std::atoi(std::getenv("HY_MM_CHECK")) : 0;
// Shortest prompt chunk whose routed experts go to the overlay (HY_MEM_MM_NPU_MIN).
// 0 = always, which is the default: the prefill is meant to run on the NPU, and a
// size-dependent fallback would also make the path depend on how a prompt was
// chunked. Set it to e.g. 1024 to put short chunks' experts back on the host, which
// is faster today (see the note in forward()).
const int mm_npu_min = std::getenv("HY_MEM_MM_NPU_MIN") ? std::atoi(std::getenv("HY_MEM_MM_NPU_MIN")) : 0;
using host_clock = std::chrono::steady_clock;

/// \brief adds the time since `t` to `acc` and moves `t` on
void lap(double& acc, host_clock::time_point& t) {
    const host_clock::time_point now = host_clock::now();
    acc += std::chrono::duration<double, std::micro>(now - t).count();
    t = now;
}

/// \brief bytes of one routed expert in the proj buffer: its up and gate (768 rows
///        each, D columns) and its down (D rows, 768 columns)
size_t expert_gate_up_bytes(const hy_mem_desc& d) { return 2 * (size_t)(d.expert_width / 32) * (d.D / 256) * Q4K_BLOCK; }
size_t expert_down_bytes(const hy_mem_desc& d) { return (size_t)(d.D / 32) * (d.expert_width / 256) * Q4K_BLOCK; }
}  // namespace

// =============================================================================
// weight views
// =============================================================================
q4k_matrix hy_mem_prefill_context::kvq(int layer, const buffer<u8>& proj) const {
    const bool moe = desc.layer_kind(layer) == hy_mem_layer_kind::moe;
    q4k_matrix m;
    m.base = proj.data() + (moe ? desc.moe.attn_k.offset : desc.dense.attn_k.offset);
    m.K = desc.D;
    m.N = desc.DK + desc.DV + desc.DQ;   // k, v, q back to back, each reorder_cpy(.., 2) on its own
    m.relaid = moe;
    return m;
}

q4k_matrix hy_mem_prefill_context::o(int layer, const buffer<u8>& proj) const {
    const bool moe = desc.layer_kind(layer) == hy_mem_layer_kind::moe;
    q4k_matrix m;
    m.base = proj.data() + (moe ? desc.moe.attn_output.offset : desc.dense.attn_output.offset);
    m.K = desc.DQ;
    m.N = desc.D;
    m.relaid = moe;
    return m;
}

q4k_matrix hy_mem_prefill_context::up_gate(int layer, const buffer<u8>& proj) const {
    const bool moe = desc.layer_kind(layer) == hy_mem_layer_kind::moe;
    q4k_matrix m;
    m.base = proj.data() + (moe ? desc.moe.ffn_up_shexp.offset : desc.dense.ffn_up.offset);
    m.K = desc.D;
    m.N = 2 * (moe ? desc.expert_width : desc.dense_width);   // [up 256 | gate 256] per round
    m.relaid = moe;
    return m;
}

q4k_matrix hy_mem_prefill_context::down(int layer, const buffer<u8>& proj) const {
    const bool moe = desc.layer_kind(layer) == hy_mem_layer_kind::moe;
    q4k_matrix m;
    m.base = proj.data() + (moe ? desc.moe.ffn_down_shexp.offset : desc.dense.ffn_down.offset);
    m.K = moe ? desc.expert_width : desc.dense_width;
    m.N = desc.D;
    m.relaid = moe;
    return m;
}

q4k_matrix hy_mem_prefill_context::expert_up_gate(int expert, const buffer<u8>& proj) const {
    const size_t expert_bytes = expert_gate_up_bytes(desc) + expert_down_bytes(desc);
    q4k_matrix m;
    m.base = proj.data() + desc.moe.ffn_up_exps.offset + expert * expert_bytes;
    m.K = desc.D;
    m.N = 2 * desc.expert_width;                    // [up 64 | gate 64] per round
    m.kind = q4k_matrix::expert_channels;
    m.relaid = true;
    m.channel_bytes = expert_bytes / 2;
    m.part_offset = 0;                              // each stream starts with its gate/up blocks
    return m;
}

q4k_matrix hy_mem_prefill_context::expert_down(int expert, const buffer<u8>& proj) const {
    q4k_matrix m = expert_up_gate(expert, proj);
    m.K = desc.expert_width;
    m.N = desc.D;
    m.part_offset = expert_gate_up_bytes(desc) / 2;  // ... then its down blocks
    return m;
}

// =============================================================================
// forward
// =============================================================================
void hy_mem_prefill_context::_reserve(int L) {
    if (L <= capacity) return;
    capacity = L;
    const size_t D = desc.D;
    const size_t widest = std::max(desc.dense_width, desc.expert_width);
    xn.resize((size_t)L * D);
    kvq_out.resize((size_t)L * (desc.DK + desc.DV + desc.DQ));
    attn_out.resize((size_t)L * desc.DQ);
    proj_out.resize((size_t)L * D);
    fused.resize((size_t)L * 2 * widest);
    hid.resize((size_t)L * widest);
    logits.resize((size_t)L * desc.num_experts);
    moe_sum.resize((size_t)L * D);
}

void hy_mem_prefill_context::forward(bf16* x, int L, int pos0, std::vector<hy_mem_layer_buffers>& layers, uint32_t MAX_L) {
    _reserve(L);
    std::fill(std::begin(stage_us), std::end(stage_us), 0.0);
    const host_clock::time_point t0 = host_clock::now();

    // the attention run is the same for every layer of this chunk (each on its own cache)
    attn_on_npu = HY_MEM_ATTN_ON_NPU && overlay && !attn_forced_host;
    mm_on_npu = HY_MEM_MM_ON_NPU && overlay && !mm_forced_host;
    // The routed experts run on the overlay like everything else (mm_npu_min 0).
    //
    // What it costs today, and why the knob exists: a run is a whole number of
    // 256-token rounds, and an expert only sees about L * top_k / num_experts =
    // L / 16 of a chunk's tokens. A 128-token chunk gives an expert ~8 rows and the
    // run still computes 256, so ~97% of the work is padding; measured, the host is
    // then faster (128 tokens: 47.7 vs 32.7 prefill tok/s, 256: 83.7 vs 56.3). The
    // crossover is somewhere in 384..1024 -- it cannot be pinned tighter because the
    // run-to-run spread here is about +-20%. From ~2K tokens up the overlay wins
    // outright (2473-token prompt: this stage 11.1 s on the host, 5.14 s on the NPU).
    // The real fix is to stop padding -- a shorter round, or one run per layer
    // covering every expert -- not to send the work back to the host.
    //
    // If the knob is set, the test is the CHUNK, not the expert's own token count:
    // routing is skewed, so a hot expert can clear a per-expert threshold in one
    // prefill and miss it when the same prompt arrives in two halves, and the same
    // tokens would then take different code paths (a resumed prefill stopped
    // matching a whole one, cos 0.999 instead of 1).
    experts_on_npu = mm_on_npu && L >= mm_npu_min;
    if (attn_on_npu) overlay->prepare_attention(L, pos0, MAX_L);

    for (uint32_t layer = 0; layer < desc.num_hidden_layers; layer++) {
        _attention_block(layer, x, L, pos0, layers[layer], MAX_L);
        if (desc.layer_kind(layer) == hy_mem_layer_kind::dense) {
            _dense_mlp(layer, x, L, layers[layer]);
        } else {
            _moe(layer, x, L, layers[layer]);
        }
    }

    if (time_host) {
        static const char* names[S_COUNT] = {"rms norm", "dequant_mm k|v|q, o", "q/k norm + rope + kv store",
                                             "attention", "dequant_mm dense + shared", "router",
                                             "dequant_mm routed experts", "glu + residual + gather", "kv cache sync"};
        const double total = std::chrono::duration<double, std::micro>(host_clock::now() - t0).count();
        std::cout << "[hy_mem prefill] " << L << " tokens at " << pos0 << ": " << total / 1000.0 << " ms ("
                  << total / L / 1000.0 << " ms/token), attention on " << (attn_on_npu ? "NPU" : "host")
                  << ", dequant_mm on " << (mm_on_npu ? "NPU" : "host")
                  << ", routed experts on " << (experts_on_npu ? "NPU" : "host")
                  << " (chunk " << L << (L >= mm_npu_min ? " >= " : " < ") << mm_npu_min << ")" << std::endl;
        for (int s = 0; s < S_COUNT; s++) {
            std::cout << "    " << names[s] << ": " << stage_us[s] / 1000.0 << " ms" << std::endl;
        }
    }
}

/// A routed expert's blocks alternate between two channel streams; the run walks
/// them from the same start, so only the stride goes with it (channel_bytes).
/// part_offset picks the expert's up|gate or down part inside each stream.
size_t hy_mem_prefill_context::_w_offset(const q4k_matrix& w, const hy_mem_layer_buffers& buf) const {
    const bool experts = w.kind == q4k_matrix::expert_channels;
    return (size_t)(w.base - buf.proj->data()) + (experts ? w.part_offset : 0);
}

void hy_mem_prefill_context::_mm(bf16* y, const q4k_matrix& w, const bf16* x, int L, hy_mem_layer_buffers& buf) {
    const bool experts = w.kind == q4k_matrix::expert_channels;
    if (!mm_on_npu || (experts && !experts_on_npu) || !hy_mem_prefill_overlay::fits(w.K, w.N)) {
        dequant_mm(y, w, x, L);
        return;
    }
    overlay->dequant_mm(y, x, L, *buf.proj_prefill, _w_offset(w, buf), w.K, w.N,
                        experts ? w.channel_bytes : 0);

    // HY_MM_CHECK=1: the same product on the host, reported per shape
    if (mm_check) {
        std::vector<bf16> ref((size_t)L * w.N);
        dequant_mm(ref.data(), w, x, L);
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const double a = (double)y[i], b = (double)ref[i];
            dot += a * b; na += a * a; nb += b * b;
        }
        const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
        // a biased rounding shows up as a non-zero mean difference; plain noise does not
        double sum_d = 0, sum_abs = 0, sum_ref_abs = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const double d = (double)y[i] - (double)ref[i];
            sum_d += d; sum_abs += std::fabs(d); sum_ref_abs += std::fabs((double)ref[i]);
        }
        const double n = (double)ref.size();
        std::cout << "[hy_mem mm] L " << L << ", " << w.K << " -> " << w.N << ", w_offset " << _w_offset(w, buf)
                  << ": cos " << cos << " mean_diff " << sum_d / n
                  << " mean_|diff| " << sum_abs / n << " mean_|ref| " << sum_ref_abs / n
                  << (cos > 0.99 ? "  OK" : "  BAD") << std::endl;
        // HY_MM_CHECK=2: run the NPU but carry the host's numbers forward, so a
        // failure here means the run disturbs something else, not its own answer
        if (mm_check > 1) memcpy(y, ref.data(), ref.size() * sizeof(bf16));
    }
}

void hy_mem_prefill_context::_attention_block(int layer, bf16* x, int L, int pos0, hy_mem_layer_buffers& buf, uint32_t MAX_L) {
    const bool moe = desc.layer_kind(layer) == hy_mem_layer_kind::moe;
    const int D = desc.D;
    host_clock::time_point t = host_clock::now();

    const size_t in_norm_offset = moe ? desc.moe.input_layer_norm.offset : desc.dense.input_layer_norm.offset;
    const bf16* in_norm = reinterpret_cast<const bf16*>((const u8*)buf.rms->data() + in_norm_offset);
    rms_norm(xn.data(), x, in_norm, L, D, desc.rms_norm_eps);
    lap(stage_us[S_NORM], t);

    _mm(kvq_out.data(), kvq(layer, *buf.proj), xn.data(), L, buf);
    lap(stage_us[S_MM_ATTN], t);

    // rope buffer: [cos | sin | q_norm | k_norm]
    const bf16* q_norm = buf.rope_rms->data() + desc.RoPE_DIM;
    const bf16* k_norm = q_norm + desc.DH;
    qk_norm_rope(kvq_out.data(), L, desc, q_norm, k_norm, pos0);
    lap(stage_us[S_ROPE_KV], t);

    kv_cache_view kv;
    kv.base = buf.kv_cache->data();
    kv.MAX_L = MAX_L;
    kv.DH = desc.DH;
    kv.heads_per_col = desc.num_kv_heads / hy_mem_desc::ATTN_COLS;
    // rows before pos0 may have come from the NPU: see them before writing ours
    if (pos0 > 0) buf.kv_cache->sync_from_device();
    lap(stage_us[S_SYNC], t);
    store_kv(kv, kvq_out.data(), L, desc, pos0);
    // Nothing to clean up past our rows: the engine keeps everything at or past the
    // context length zero (hy_mem_npu::Impl::set_context_length), so the round the
    // attention reads beyond this chunk is already zeros.
    lap(stage_us[S_ROPE_KV], t);
    buf.kv_cache->sync_to_device();    // the decode overlay reads them next
    lap(stage_us[S_SYNC], t);

    // q is the last DQ of each k|v|q row
    const int kvq_stride = desc.DK + desc.DV + desc.DQ;
    if (attn_on_npu) {
        overlay->attention(attn_out.data(), kvq_out.data() + desc.DK + desc.DV, kvq_stride, *buf.kv_cache);
    } else {
        attention(attn_out.data(), kvq_out.data() + desc.DK + desc.DV, kvq_stride, kv, L, pos0, desc.num_heads, desc.num_kv_heads);
    }
    lap(stage_us[S_ATTN], t);

    _mm(proj_out.data(), o(layer, *buf.proj), attn_out.data(), L, buf);
    lap(stage_us[S_MM_ATTN], t);
    add_rows(x, proj_out.data(), (size_t)L * D);
    lap(stage_us[S_GLU_ADD], t);
}

void hy_mem_prefill_context::_dense_mlp(int layer, bf16* x, int L, hy_mem_layer_buffers& buf) {
    const int D = desc.D;
    host_clock::time_point t = host_clock::now();
    const bf16* post_norm = reinterpret_cast<const bf16*>((const u8*)buf.rms->data() + desc.dense.post_attntion_norm.offset);
    rms_norm(xn.data(), x, post_norm, L, D, desc.rms_norm_eps);
    lap(stage_us[S_NORM], t);

    _mm(fused.data(), up_gate(layer, *buf.proj), xn.data(), L, buf);
    lap(stage_us[S_MM_MLP], t);
    silu_glu(hid.data(), fused.data(), L, desc.dense_width, hy_mem_desc::DENSE_ROWS_PER_ROUND);
    lap(stage_us[S_GLU_ADD], t);
    _mm(proj_out.data(), down(layer, *buf.proj), hid.data(), L, buf);
    lap(stage_us[S_MM_MLP], t);
    add_rows(x, proj_out.data(), (size_t)L * D);
    lap(stage_us[S_GLU_ADD], t);
}

void hy_mem_prefill_context::_moe(int layer, bf16* x, int L, hy_mem_layer_buffers& buf) {
    const int D = desc.D, W = desc.expert_width;
    host_clock::time_point t = host_clock::now();
    const u8* rms = (const u8*)buf.rms->data();
    rms_norm(xn.data(), x, reinterpret_cast<const bf16*>(rms + desc.moe.post_attntion_norm.offset), L, D, desc.rms_norm_eps);
    lap(stage_us[S_NORM], t);

    // router on the host: fp32 logits, sigmoid, top 8 of score + bias
    router_logits(logits.data(), xn.data(), reinterpret_cast<const bf16*>(rms + desc.moe.ffn_gate_inp.offset),
                  L, D, desc.num_experts);
    std::vector<expert_task> tasks = route(logits.data(), *buf.router_bias, L, desc.num_experts, desc.top_k, desc.router_scaling);
    lap(stage_us[S_ROUTER], t);

    // shared expert: every token, weight 1; starts the fp32 sum
    _mm(fused.data(), up_gate(layer, *buf.proj), xn.data(), L, buf);
    lap(stage_us[S_MM_MLP], t);
    silu_glu(hid.data(), fused.data(), L, W, hy_mem_desc::SHARED_ROWS_PER_ROUND);
    lap(stage_us[S_GLU_ADD], t);
    _mm(proj_out.data(), down(layer, *buf.proj), hid.data(), L, buf);
    lap(stage_us[S_MM_MLP], t);
    for (size_t i = 0; i < (size_t)L * D; i++) moe_sum[i] = (float)proj_out[i];
    lap(stage_us[S_GLU_ADD], t);

    // routed experts: each runs on its own tokens only, gathered into a dense block
    // (the shape a dequant_mm overlay run takes), then scattered back with its weight
    //
    // One run per expert, not one per layer: batching every expert's rounds into a
    // single run was tried and reverted (2026-10-27). It cannot help, because a
    // round is 256 tokens of ONE expert, so the padding -- which is what the stage
    // actually spends its time on -- stays. Measured at 2473 tokens: the NPU stage
    // did not move (5.8 -> 6.4 s), while padding the host's gather / GLU / scatter
    // buffers to match cost 2 s more. Shrinking the round (kernel/dq/common.hpp
    // NPU_M_TILE) is the change that would pay.
    std::vector<bf16> xe, fe, he, ye;
    for (const expert_task& task : tasks) {
        const int n = (int)task.tokens.size();
        xe.resize((size_t)n * D);
        fe.resize((size_t)n * 2 * W);
        he.resize((size_t)n * W);
        ye.resize((size_t)n * D);
        for (int i = 0; i < n; i++) {
            memcpy(xe.data() + (size_t)i * D, xn.data() + (size_t)task.tokens[i] * D, D * sizeof(bf16));
        }
        lap(stage_us[S_GLU_ADD], t);
        _mm(fe.data(), expert_up_gate(task.expert, *buf.proj), xe.data(), n, buf);
        lap(stage_us[S_MM_EXPERT], t);
        silu_glu(he.data(), fe.data(), n, W, hy_mem_desc::EXPERT_ROWS_PER_ROUND);
        lap(stage_us[S_GLU_ADD], t);
        _mm(ye.data(), expert_down(task.expert, *buf.proj), he.data(), n, buf);
        lap(stage_us[S_MM_EXPERT], t);
        // a token appears once per expert, so rows never collide inside one task
        #pragma omp parallel for num_threads(prefill_threads()) schedule(static)
        for (int i = 0; i < n; i++) {
            float* acc = moe_sum.data() + (size_t)task.tokens[i] * D;
            const bf16* y = ye.data() + (size_t)i * D;
            for (int d = 0; d < D; d++) acc[d] += task.weights[i] * (float)y[d];
        }
        lap(stage_us[S_GLU_ADD], t);
    }
    add_rows_f32(x, moe_sum.data(), (size_t)L * D);
    lap(stage_us[S_GLU_ADD], t);
}

// =============================================================================
// layout self-check (HY_CHECK_LAYOUT=1)
// =============================================================================
namespace {
/// \brief one source tensor and where its row blocks land in a view's logical rows
struct source_rows {
    const buffer<u8>* blob;   // the tensor as in model.q4nx: [row block][column block] plain blocks
    int first_rb;             // first row block of this tensor (stacked experts: e * rows)
};

/// \brief compares every block of `view` with the q4nx blocks it came from
/// \param map logical row block -> (source tensor, its row block)
template <typename Map>
size_t compare_view(const q4k_matrix& view, Map map) {
    size_t bad = 0;
    alignas(64) u8 plain[Q4K_BLOCK];
    for (int rb = 0; rb < view.row_blocks(); rb++) {
        const auto [src, src_rb] = map(rb);
        for (int cb = 0; cb < view.blocks_per_row(); cb++) {
            const u8* got = view.block(rb, cb);
            if (view.relaid) {
                q4k_unrelayout(plain, got);
                got = plain;
            }
            const u8* want = src->data() + ((size_t)src_rb * view.blocks_per_row() + cb) * Q4K_BLOCK;
            bad += memcmp(got, want, Q4K_BLOCK) != 0;
        }
    }
    return bad;
}

/// \brief row block rb of an [up slab | gate slab]-interleaved matrix -> (up or gate, row block)
std::pair<const buffer<u8>*, int> interleaved(int rb, int slab_rows, const buffer<u8>* up, const buffer<u8>* gate, int first_rb) {
    const int per = slab_rows / Q4K_ROWS;            // row blocks per slab
    const int piece = rb / (2 * per), within = rb % (2 * per);
    return within < per ? std::make_pair(up, first_rb + piece * per + within)
                        : std::make_pair(gate, first_rb + piece * per + within - per);
}
}  // namespace

size_t hy_mem_prefill_context::check_layout(int layer, Q4NX& q4nx, const buffer<u8>& proj) const {
    const bool moe = desc.layer_kind(layer) == hy_mem_layer_kind::moe;
    auto load = [&](weight_desc_t& w) {
        buffer<u8> b;
        q4nx.load_weights(b, w.format_name(layer));
        return b;
    };
    size_t bad = 0;

    // k | v | q
    weight_desc_t& wk = moe ? desc.moe.attn_k : desc.dense.attn_k;
    weight_desc_t& wv = moe ? desc.moe.attn_v : desc.dense.attn_v;
    weight_desc_t& wq = moe ? desc.moe.attn_q : desc.dense.attn_q;
    buffer<u8> k = load(wk), v = load(wv), q = load(wq);
    const int k_rbs = desc.DK / Q4K_ROWS, v_rbs = desc.DV / Q4K_ROWS;
    bad += compare_view(kvq(layer, proj), [&](int rb) {
        if (rb < k_rbs) return std::make_pair(&k, rb);
        if (rb < k_rbs + v_rbs) return std::make_pair(&v, rb - k_rbs);
        return std::make_pair(&q, rb - k_rbs - v_rbs);
    });
    buffer<u8> wo = load(moe ? desc.moe.attn_output : desc.dense.attn_output);
    bad += compare_view(o(layer, proj), [&](int rb) { return std::make_pair(&wo, rb); });

    // dense MLP or shared expert
    buffer<u8> up = load(moe ? desc.moe.ffn_up_shexp : desc.dense.ffn_up);
    buffer<u8> gate = load(moe ? desc.moe.ffn_gate_shexp : desc.dense.ffn_gate);
    buffer<u8> dn = load(moe ? desc.moe.ffn_down_shexp : desc.dense.ffn_down);
    const int slab = moe ? hy_mem_desc::SHARED_ROWS_PER_ROUND : hy_mem_desc::DENSE_ROWS_PER_ROUND;
    bad += compare_view(up_gate(layer, proj), [&](int rb) { return interleaved(rb, slab, &up, &gate, 0); });
    bad += compare_view(down(layer, proj), [&](int rb) { return std::make_pair(&dn, rb); });

    if (moe) {
        // routed experts: stacked in the q4nx tensors, expert e = row blocks e * (rows / 32) ..
        buffer<u8> eu = load(desc.moe.ffn_up_exps), eg = load(desc.moe.ffn_gate_exps), ed = load(desc.moe.ffn_down_exps);
        const int ug_rbs = desc.expert_width / Q4K_ROWS, d_rbs = desc.D / Q4K_ROWS;
        for (int e = 0; e < (int)desc.num_experts; e++) {
            bad += compare_view(expert_up_gate(e, proj), [&](int rb) {
                return interleaved(rb, hy_mem_desc::EXPERT_ROWS_PER_ROUND, &eu, &eg, e * ug_rbs);
            });
            bad += compare_view(expert_down(e, proj), [&](int rb) { return std::make_pair(&ed, e * d_rbs + rb); });
        }
    }
    return bad;
}
