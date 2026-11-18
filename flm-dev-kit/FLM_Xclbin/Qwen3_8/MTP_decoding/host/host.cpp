// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "typedef.hpp"
#include "buffer.hpp"
#include "decoding_layer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "models.h"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"

#include "cpu_ref_bridge.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct cpu_fc {
    std::vector<float> embed;        // [D], raw, bf16-rounded
    std::vector<float> hidden;       // [D], raw, bf16-rounded
    std::vector<float> norm_embed;   // [D]
    std::vector<float> norm_hidden;  // [D]
    std::vector<float> fc_out;       // [D]
    std::vector<float> xn;           // [D],   rms(fc_out, input_layernorm)
    std::vector<float> q;            // [DQ],  after q_norm and the rotation
    std::vector<float> k;            // [DK],  after k_norm and the rotation
    std::vector<float> v;            // [DV],  untouched
    std::vector<float> z;            // [DQ],  the output gate, **pre**-sigmoid
};

/// Partial rotary, in place: dims [0, rope_dim) only, rotate_half over the
/// halves [0, 32) and [32, 64). Dims [64, 256) pass through.
///
/// The same arithmetic as cpu_ref_mtp_head.cpp's rope() and kernels/rope.cc's
/// apply_rope(). cos and sin are rope_dim wide with the 32 frequency slots
/// duplicated across both halves, which is what makes this a single pass.
void rope_inplace(float* v, const float* cos, const float* sin, int rope_dim) {
    const int H = rope_dim / 2;
    std::vector<float> r((size_t)rope_dim);
    for (int j = 0; j < H; j++)        r[j] = v[j] * cos[j] - v[j + H] * sin[j];
    for (int j = H; j < rope_dim; j++) r[j] = v[j] * cos[j] + v[j - H] * sin[j];
    std::memcpy(v, r.data(), (size_t)rope_dim * sizeof(float));
}

/// The cos/sin pair for one position, bf16-rounded.
///
/// Must match decoding_layer::set_rope_position element for element: the array
/// is scored against this, so any difference reads out as datapath error the
/// hardware never committed. Rounded for that reason -- transformers builds the
/// table in fp32 and casts, and the reference consumed the cast one.
void rope_pair(int position, int rope_dim, std::vector<float>& cos,
               std::vector<float>& sin) {
    constexpr double ROPE_THETA = 1e7;
    const int H = rope_dim / 2;
    cos.assign((size_t)rope_dim, 0.0f);
    sin.assign((size_t)rope_dim, 0.0f);
    for (int j = 0; j < H; j++) {
        const double inv = std::pow(ROPE_THETA, -(double)(2 * j) / (double)rope_dim);
        const double ang = (double)position * inv;
        const float c = cpu_ref::round_bf16((float)std::cos(ang));
        const float s = cpu_ref::round_bf16((float)std::sin(ang));
        cos[j] = cos[j + H] = c;
        sin[j] = sin[j + H] = s;
    }
}

cpu_fc compute_fc(const std::string& q4nx_path, int32_t token_id,
                  const std::vector<float>& prev_hidden, int D,
                  int DQ, int DK, int DV, int DH, int ROPE_DIM, int position) {
    using namespace cpu_ref;
    const int FC_IN = 2 * D;

    safetensors q4nx(q4nx_path);

    q4nx_tensor w_fc    = get_q4nx(q4nx, "model.mtp.fc.weight");
    q4nx_tensor w_embed = get_q4nx(q4nx, "model.embed_tokens.weight");
    std::vector<float> g_embed  = q4nx.f32("model.mtp.pre_fc_norm_embedding.weight");
    std::vector<float> g_hidden = q4nx.f32("model.mtp.pre_fc_norm_hidden.weight");

    const std::string lp = "model.mtp.layers.0.";
    q4nx_tensor w_q = get_q4nx(q4nx, lp + "self_attn.q_proj.weight");
    q4nx_tensor w_k = get_q4nx(q4nx, lp + "self_attn.k_proj.weight");
    q4nx_tensor w_v = get_q4nx(q4nx, lp + "self_attn.v_proj.weight");
    std::vector<float> g_in_ln = q4nx.f32(lp + "input_layernorm.weight");

    if (w_fc.rows != D || w_fc.cols != FC_IN)
        throw std::runtime_error("fc geometry mismatch in " + q4nx_path);
    // q_proj holds the queries and the output gate; only the queries are wired.
    if (w_q.rows != 2 * DQ || w_q.cols != D)
        throw std::runtime_error("q_proj geometry mismatch in " + q4nx_path);
    if (w_k.rows != DK || w_v.rows != DV)
        throw std::runtime_error("k/v_proj geometry mismatch in " + q4nx_path);

    // embed_tokens row lookup. The table is Q8_0 and stored in 32x256 blocks
    // whose tiles come back column-major, so a row is one sweep across the
    // block columns picking out this token's lane.
    cpu_fc r;
    std::vector<float> e((size_t)D, 0.0f);
    {
        const int rb = token_id / Q4NX_ROW_BLOCK;
        const int rl = token_id % Q4NX_ROW_BLOCK;
        std::vector<float> tile((size_t)Q4NX_COL_BLOCK * Q4NX_ROW_BLOCK);
        for (int cb = 0; cb < w_embed.ncb; cb++) {
            w_embed.dequant_block(rb, cb, tile.data());
            float* dst = e.data() + (size_t)cb * Q4NX_COL_BLOCK;
            for (int c = 0; c < Q4NX_COL_BLOCK; c++)
                dst[c] = tile[(size_t)c * Q4NX_ROW_BLOCK + rl];
        }
    }

    // The array norms what the DMA delivered, which is bf16. Round both halves
    // here so the CPU reduces over the same values: skip this and the rounding
    // enters the sum of squares on one side only, and reads out as datapath
    // error the hardware never committed.
    r.embed.assign((size_t)D, 0.0f);
    r.hidden.assign((size_t)D, 0.0f);
    for (int i = 0; i < D; i++) {
        r.embed[i]  = round_bf16(e[i]);
        r.hidden[i] = round_bf16(prev_hidden[i]);
    }

    r.norm_embed.assign((size_t)D, 0.0f);
    r.norm_hidden.assign((size_t)D, 0.0f);
    rms_norm(r.embed.data(),  g_embed.data(),  D, r.norm_embed.data());
    rms_norm(r.hidden.data(), g_hidden.data(), D, r.norm_hidden.data());

    // cat[norm_embed, norm_hidden] -- embedding half first, same order the
    // array reconstructs from arrival order.
    std::vector<float> fc_in((size_t)FC_IN);
    std::memcpy(fc_in.data(),     r.norm_embed.data(),  (size_t)D * sizeof(float));
    std::memcpy(fc_in.data() + D, r.norm_hidden.data(), (size_t)D * sizeof(float));

    r.fc_out.assign((size_t)D, 0.0f);
    q4nx_matmul(w_fc, fc_in.data(), 1, r.fc_out.data());

    // The array norms what the rms tile received, which is the bf16 fc_out the
    // projection cores emitted. Round before reducing for the same reason the
    // two input halves are rounded above: otherwise the rounding enters the sum
    // of squares on one side only and reads out as datapath error.
    std::vector<float> fc_out_bf16((size_t)D);
    for (int i = 0; i < D; i++) fc_out_bf16[i] = round_bf16(r.fc_out[i]);

    r.xn.assign((size_t)D, 0.0f);
    rms_norm(fc_out_bf16.data(), g_in_ln.data(), D, r.xn.data());

    // q_proj is [2 * DQ][D] -- queries then output gate. The q4nx converter has
    // de-interleaved HF's per-head (q, gate) pairs, so the split is a clean
    // halfway cut with no per-head stride to undo.
    // (memory: mtp-q4nx-layout-and-error-floor)
    std::vector<float> qg((size_t)2 * DQ, 0.0f);
    q4nx_matmul(w_q, r.xn.data(), 1, qg.data());
    r.q.assign(qg.begin(), qg.begin() + DQ);
    // Stored raw. The sigmoid is applied where the gate is applied, so that
    // this stays comparable against a z tap if one is ever added.
    r.z.assign(qg.begin() + DQ, qg.end());

    r.k.assign((size_t)DK, 0.0f);
    r.v.assign((size_t)DV, 0.0f);
    q4nx_matmul(w_k, r.xn.data(), 1, r.k.data());
    q4nx_matmul(w_v, r.xn.data(), 1, r.v.data());

    // ---- per-head q/k RMSNorm, then the partial rotary --------------------
    // The norm runs BEFORE the rotation and on the 256-wide head dim -- the
    // same order as the other Qwen ports, and the reverse of HunYuan. The
    // gains are stored raw: the GGUF -> q4nx path already folded the +1 that
    // transformers writes as `x_norm * (1 + w)`.
    //
    // v is untouched. It is projected here only because it shares the fused
    // row with q and k; the array likewise just forwards it.
    std::vector<float> g_q_norm = q4nx.f32(lp + "self_attn.q_norm.weight");
    std::vector<float> g_k_norm = q4nx.f32(lp + "self_attn.k_norm.weight");
    if ((int)g_q_norm.size() != DH || (int)g_k_norm.size() != DH)
        throw std::runtime_error("q/k_norm geometry mismatch in " + q4nx_path);

    std::vector<float> cos, sin;
    rope_pair(position, ROPE_DIM, cos, sin);

    std::vector<float> tmp((size_t)DH);
    for (int h = 0; h < DQ / DH; h++) {
        float* dst = r.q.data() + (size_t)h * DH;
        rms_norm(dst, g_q_norm.data(), DH, tmp.data());
        std::memcpy(dst, tmp.data(), (size_t)DH * sizeof(float));
        rope_inplace(dst, cos.data(), sin.data(), ROPE_DIM);
    }
    for (int h = 0; h < DK / DH; h++) {
        float* dst = r.k.data() + (size_t)h * DH;
        rms_norm(dst, g_k_norm.data(), DH, tmp.data());
        std::memcpy(dst, tmp.data(), (size_t)DH * sizeof(float));
        rope_inplace(dst, cos.data(), sin.data(), ROPE_DIM);
    }
    return r;
}

/// How many rows of context the cache is seeded with before the pass. The
/// drafted token appends its own k|v at row CTX, so attention runs over
/// CTX + 1 = 1121 keys -- deliberately not a multiple of L_CHUNK = 16, so the
/// final chunk is partial and the kernel's mask has to drop 15 of its 16 lanes.
/// A multiple would leave that edge untested.
constexpr int CTX = 1120;

/// softmax(q K^T / sqrt(DH)) V for one query row over `keys` cached rows.
///
/// GQA: q head h reads kv head h / (NH / NKV). No mask -- a T = 1 step attends
/// to the whole cache, so every key is visible and the mask is all-ones.
///
/// In fp64 and with the max subtracted, which is not what the array does: it
/// runs an online softmax in bf16 with a running max and a per-chunk correction
/// factor. This is the reference those two are being scored against, so it
/// should be the accurate one; the gap between them is the thing under test.
std::vector<float> cpu_attention(const std::vector<float>& q,
                                 const std::vector<float>& k_cache,
                                 const std::vector<float>& v_cache,
                                 int keys, int DH, int NH, int NKV) {
    const int gqa = NH / NKV;
    const int DK = NKV * DH;
    const double scale = 1.0 / std::sqrt((double)DH);
    std::vector<float> o((size_t)NH * DH, 0.0f);

    for (int h = 0; h < NH; h++) {
        const int kvh = h / gqa;
        const float* qh = q.data() + (size_t)h * DH;

        std::vector<double> s((size_t)keys);
        double m = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < keys; t++) {
            const float* kt = k_cache.data() + (size_t)t * DK + (size_t)kvh * DH;
            double dot = 0.0;
            for (int j = 0; j < DH; j++) dot += (double)qh[j] * (double)kt[j];
            s[t] = dot * scale;
            if (s[t] > m) m = s[t];
        }
        double denom = 0.0;
        for (int t = 0; t < keys; t++) { s[t] = std::exp(s[t] - m); denom += s[t]; }

        std::vector<double> acc((size_t)DH, 0.0);
        for (int t = 0; t < keys; t++) {
            const float* vt = v_cache.data() + (size_t)t * DK + (size_t)kvh * DH;
            const double w = s[t];
            for (int j = 0; j < DH; j++) acc[j] += w * (double)vt[j];
        }
        float* oh = o.data() + (size_t)h * DH;
        for (int j = 0; j < DH; j++) oh[j] = (float)(acc[j] / denom);
    }
    return o;
}

void run_fc_test(model_list::model_t model, LM_Config& lm_config, Q4NX& q4nx,
                 npu_app_manager* app_mgr, const std::string& model_path,
                 const std::string& ref_path) {
    const int D  = lm_config.hidden_size;
    const int DH = lm_config.head_dim;
    const int DQ = DH * lm_config.num_attention_heads;
    const int DK = DH * lm_config.num_key_value_heads;
    const int DV = DK;
    const int ROPE_DIM = DH / 4;   // partial_rotary_factor 0.25
    const int step = 1;

    header_print("info", "=== MTP fc + input_layernorm + qkv + q/k norm + RoPE "
                         "+ attention + output gate + o_proj + residual "
                         "+ post_attention_layernorm + up|gate + SwiGLU "
                         "+ down_proj + MLP residual -- the whole decoder "
                         "layer, decode shape ===");

    // The dump supplies this step's *inputs* only: the token id it embeds and
    // the prev_hidden it fuses with. Reproducing those would mean running the
    // whole 27B main model, which is not this harness's job.
    cpu_ref::safetensors ref(ref_path);
    const std::string tag = "mtp.step" + std::to_string(step) + ".";
    if (!ref.has(tag + "input_ids") || !ref.has(tag + "prev_hidden")) {
        header_print("error", "step " + std::to_string(step) + " is absent from the dump");
        return;
    }

    std::vector<int32_t> ids = ref.i32(tag + "input_ids");
    std::vector<float> prev_hidden = ref.f32(tag + "prev_hidden");
    if (ids.size() != 1 || (int)prev_hidden.size() != D) {
        header_print("error", "step " + std::to_string(step) + " is not T = 1 (got " +
                                  std::to_string(ids.size()) + " ids, " +
                                  std::to_string(prev_hidden.size()) + " hidden)");
        return;
    }

    // The position the token rotates at. All three M-RoPE channels carry the
    // same value for a text draft, which is what collapses interleaved M-RoPE
    // to plain 1-D RoPE -- so channel 0 is the position.
    std::vector<int32_t> pos_thw = ref.i32(tag + "position_ids");
    if (pos_thw.size() != 3) {
        header_print("error", "expected 3 position channels, got " +
                                  std::to_string(pos_thw.size()));
        return;
    }
    if (pos_thw[0] != pos_thw[1] || pos_thw[1] != pos_thw[2]) {
        header_print("error", "the three M-RoPE channels differ -- this is not a "
                              "text-only draft and the 1-D collapse does not hold");
        return;
    }
    const int position = pos_thw[0];
    header_print("info", "step " + std::to_string(step) + " drafts token " +
                             std::to_string(ids[0]) + " at position " +
                             std::to_string(position));

    const std::string q4nx_path = cpu_ref::q4nx_path_of(model_path);
    header_print("info", "computing the reference from " + q4nx_path + " ...");
    cpu_fc cpu = compute_fc(q4nx_path, ids[0], prev_hidden, D, DQ, DK, DV,
                            DH, ROPE_DIM, position);

    // The *raw* halves, embedding first. The passthrough tile owns the two rms
    // norms now, so sending pre-normed input would norm twice.
    std::vector<float> fc_in;
    fc_in.reserve(2 * (size_t)D);
    fc_in.insert(fc_in.end(), cpu.embed.begin(),  cpu.embed.end());
    fc_in.insert(fc_in.end(), cpu.hidden.begin(), cpu.hidden.end());
    assert(fc_in.size() == 2 * (size_t)D && "fc takes both D-wide halves");
    buffer<bf16> in_buffer = cpu_ref::to_bf16(fc_in);

    // The cache holds CTX seeded rows plus the one this step appends. The draft
    // head's cache is its own -- MTP slot j is main-model slot j+1 -- so nothing
    // else may alias it.
    //
    // **The rotation position and the cache slot are decoupled here.** They
    // coincide in a real chained decode, but this bench seeds CTX rows of noise
    // rather than replaying real history, so it appends at row CTX while still
    // rotating q and k at the dump's position -- which is what the CPU q/k above
    // were computed with. Passing `position` to set_context_length would rotate
    // correctly but attend over the wrong number of keys.
    const int slot = CTX;
    const int keys = CTX + 1;
    // Rounded up to a whole number of L_CHUNKs. The replay always sends whole
    // chunks -- the kernels stop at L[0] and ignore the tail -- so the cache has
    // to own the rows that tail reads, or the descriptor walks past the end of
    // the buffer. 1121 keys means 1136 allocated rows here.
    constexpr int L_CHUNK = 16;
    const int cache_rows = (keys + L_CHUNK - 1) / L_CHUNK * L_CHUNK;
    decoding_layer layer(model, lm_config, app_mgr, cache_rows);
    layer.load_weights(q4nx);

    // ---- the cache the pass attends over: CTX rows, head-major, DK wide -----
    //
    // **v is pure noise, but k is not, and that is deliberate.** With k drawn
    // from N(0, 1) the cached keys are near-orthogonal to q, so their scores sit
    // around 0 while the appended row -- whose k came from the same token as q
    // -- scores |q|^2 / 16 ~ 16. exp() turns that gap into 99.98 % of the
    // softmax mass on one row, and the other 1120 contribute nothing: the
    // replay path would be scored at about one part in 5000, and a completely
    // broken chunk stream would still read as cosine 0.999. Measured, not
    // assumed -- dropping the appended row from the reference moved the cosine
    // from 0.9999 to 0.002, which is the signature of a single dominant key.
    //
    // So each cached key is given a deliberate projection onto its kv head's
    // query, scaled so its *mean* score matches the appended row's, plus noise
    // for spread. Every key then carries comparable weight, the sum over 1121
    // rows is a real reduction, and a mis-ordered or mis-strided chunk moves the
    // answer. v stays independent noise -- it is what the weights mix, so
    // correlating it would hide exactly the errors this is meant to catch.
    std::mt19937 rng(20260914);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> k_cache((size_t)keys * DK), v_cache((size_t)keys * DV);

    const int NKV = lm_config.num_key_value_heads;
    const int gqa = lm_config.num_attention_heads / NKV;
    for (int kvh = 0; kvh < NKV; kvh++) {
        // The group's first query head. The other five are only partly aligned
        // with it, which is fine -- it spreads their weights further, and the
        // point is that no single row runs away with the softmax.
        const float* qh = cpu.q.data() + (size_t)(kvh * gqa) * DH;
        double qn2 = 0.0;
        for (int j = 0; j < DH; j++) qn2 += (double)qh[j] * qh[j];

        // The appended row's own score, which is the level to match.
        const float* kn = cpu.k.data() + (size_t)kvh * DH;
        double self = 0.0;
        for (int j = 0; j < DH; j++) self += (double)qh[j] * kn[j];
        // alpha * |q|^2 == self  =>  alpha = self / |q|^2. The 1/sqrt(DH) scale
        // divides both sides, so it does not enter.
        const double alpha = (qn2 > 0.0) ? self / qn2 : 0.0;

        for (int t = 0; t < CTX; t++) {
            float* kt = k_cache.data() + (size_t)t * DK + (size_t)kvh * DH;
            float* vt = v_cache.data() + (size_t)t * DV + (size_t)kvh * DH;
            for (int j = 0; j < DH; j++) {
                // bf16-rounded: the cache is a bf16 buffer on both sides, so
                // rounding here keeps the reference reducing over exactly the
                // values the array was given.
                kt[j] = cpu_ref::round_bf16((float)(alpha * qh[j]) + gauss(rng));
                vt[j] = cpu_ref::round_bf16(gauss(rng));
            }
        }
    }
    layer.seed_kv_cache(k_cache, v_cache, CTX);

    layer.set_context_length(slot);
    layer.set_rope_position(position);

    // The one and only invocation. Nothing below this line may touch the array.
    buffer<bf16> got_layer_out = layer(in_buffer);

    // FLM_MTP_BENCH: the *other* multi-dispatch path -- decoding_layer::bench()
    // enqueues num_hidden_layers runs of ONE sequence in a single runlist, with
    // no regeneration and no kernel re-setup between them. If this completes
    // while FLM_MTP_REPEAT times out, the array is re-entrant and the fault is
    // in the per-dispatch host path; if it times out too, the design is not.
    if (getenv("FLM_MTP_BENCH")) {
        header_print("info", "runlist bench: many runs of one sequence");
        try {
            const float tps = layer.bench(in_buffer);
            header_print("info", "  runlist completed, " + std::to_string(tps) +
                                     " layer/s");
        } catch (const std::exception& e) {
            header_print("error", std::string("runlist threw: ") + e.what());
            exit(1);
        }
        return;
    }

    // TEMPORARY probe: does one app survive a second dispatch? FLM_MTP_REPEAT=n
    // dispatches n extra times at advancing cursors, the way the DLL port does.
    if (const char* rep = getenv("FLM_MTP_REPEAT")) {
        // FLM_MTP_BASE pins the repeat cursors low, where the DLL port fails;
        // FLM_MTP_ROT advances the rotary position per dispatch, as it does.
        const char* base_s = getenv("FLM_MTP_BASE");
        const int base = base_s ? atoi(base_s) : slot + 1;
        const bool rot = getenv("FLM_MTP_ROT") != nullptr;
        // FLM_MTP_TRUNC2=n truncates only the REPEAT dispatches, leaving the
        // scored first pass whole -- so the array starts each probe from a
        // known-good at-rest state and the phase index that first times out is
        // the phase the second pass cannot get past.
        if (const char* t2 = getenv("FLM_MTP_TRUNC2")) {
            setenv("FLM_MTP_TRUNC", t2, 1);
            header_print("info", std::string("repeat dispatches truncated after phase ") + t2);
        }
        for (int i = 0; i < atoi(rep); i++) {
            layer.set_context_length(base + i);
            if (rot) layer.set_rope_position(position + 1 + i);
            header_print("info", "repeat dispatch " + std::to_string(i + 1) +
                                     " at cursor " + std::to_string(base + i) +
                                     (rot ? ", rotating" : ", same angle"));
            buffer<bf16> again = layer(in_buffer);
            header_print("info", "  ok, out[0] = " + std::to_string((float)again[0]));
        }
    }
    assert((int)got_layer_out.size() == D &&
           "the layer output is one model-dimension-wide row");

    // The appended row, read back out of the cache. Scored on its own first:
    // attention reduces over it along with 1120 others, so an error here is
    // worth about 1/1121 of the output and would never show up in o's cosine.
    buffer<bf16> got_k = layer.get_k_cache(slot);
    buffer<bf16> got_v = layer.get_v_cache(slot);

    struct { const char* name; const std::vector<float>* ref; buffer<bf16>* got; } taps[] = {
        {"k (normed + rotated)", &cpu.k, &got_k},
        {"v (untouched)",        &cpu.v, &got_v},
    };
    for (const auto& t : taps) {
        buffer<bf16> cpu_part = cpu_ref::to_bf16(*t.ref);
        assert(t.got->size() == cpu_part.size());
        header_print("info", std::string("=== ") + t.name + ", the appended row ===");
        print_error_metrics(get_error_metrics(*t.got, cpu_part));
    }

    // The reference attends over the same cache the array did: the CPU's own
    // k and v at the appended row, and the seeded noise everywhere else. Using
    // the NPU's k|v here instead would charge the attention for the projection
    // and rotation in front of it. (memory: score-a-projection-against-its-own-input)
    std::memcpy(k_cache.data() + (size_t)slot * DK, cpu.k.data(), (size_t)DK * sizeof(float));
    std::memcpy(v_cache.data() + (size_t)slot * DV, cpu.v.data(), (size_t)DV * sizeof(float));

    // How concentrated the softmax actually is, for head 0. If one row carries
    // almost all the mass then o is essentially that row's v and this test
    // proves nothing about the replay -- so report it rather than trusting the
    // cache construction above to have worked.
    {
        const float* qh = cpu.q.data();
        double m = -std::numeric_limits<double>::infinity();
        std::vector<double> s((size_t)keys);
        for (int t = 0; t < keys; t++) {
            const float* kt = k_cache.data() + (size_t)t * DK;
            double d = 0.0;
            for (int j = 0; j < DH; j++) d += (double)qh[j] * (double)kt[j];
            s[t] = d / std::sqrt((double)DH);
            if (s[t] > m) m = s[t];
        }
        double tot = 0.0, top = 0.0;
        for (int t = 0; t < keys; t++) { s[t] = std::exp(s[t] - m); tot += s[t];
                                         if (s[t] > top) top = s[t]; }
        double H = 0.0;
        for (int t = 0; t < keys; t++) { const double p = s[t] / tot;
                                         if (p > 1e-300) H -= p * std::log(p); }
        header_print("info", "head 0 softmax: largest weight " +
                                 std::to_string(top / tot) + ", effective keys " +
                                 std::to_string(std::exp(H)) + " of " +
                                 std::to_string(keys));
    }

    header_print("info", "computing the attention reference over " +
                             std::to_string(keys) + " keys ...");
    std::vector<float> cpu_o = cpu_attention(cpu.q, k_cache, v_cache, keys, DH,
                                             lm_config.num_attention_heads,
                                             lm_config.num_key_value_heads);

    // ---- the output gate: y = o * sigmoid(z) ------------------------------
    //
    // sigmoid via tanh, matching kernels/lut_based_ops.h's
    // getSigmoidBf16_nonLUT -- sigmoid(x) = (tanh(x/2) + 1) / 2. The identity is
    // exact, so this is about evaluating the same expression, not about copying
    // an approximation.
    //
    // z is rounded to bf16 first: the gate core reads what the DMA delivered,
    // so reducing over the unrounded value here would read out as datapath
    // error the hardware never committed.
    auto sigmoid = [](float x) {
        return 0.5f * (std::tanh(0.5f * x) + 1.0f);
    };

    std::vector<float> cpu_y((size_t)DQ);
    for (int i = 0; i < DQ; i++)
        cpu_y[i] = cpu_o[i] * sigmoid(cpu_ref::round_bf16(cpu.z[i]));
    buffer<bf16> cpu_y_bf16 = cpu_ref::to_bf16(cpu_y);

    // ---- o_proj, the residual, the norm, the SwiGLU, down, and h2 ----------
    //
    //     h    = fc_out + W_o @ (o * sigmoid(z))
    //     yn   = rms(h, post_attention_layernorm)
    //     hid  = silu(gate(yn)) * up(yn)
    //     down = W_down @ hid
    //     h2   = h + down                     <-- what comes back now
    //
    // h2 closes the decoder layer. Everything between o and it stays resident:
    // o goes to the rms tile as the attention residual, yn is up|gate's x, hid
    // is down's x, and down goes back to the rms tile as the MLP residual's
    // second operand.
    //
    // **Scored twice, and the second number is the one to read.** h2 carries h
    // straight through an identity term, so its raw cosine is flattered by a
    // gain that has nothing to do with the MLP -- subtracting the CPU's own h
    // recovers down, which is what actually moves when the MLP is wrong.
    // (memory: score-a-projection-against-its-own-input)
    {
        const std::string q4nx_path = cpu_ref::q4nx_path_of(model_path);
        cpu_ref::safetensors q4nx(q4nx_path);
        cpu_ref::q4nx_tensor w_o =
            cpu_ref::get_q4nx(q4nx, "model.mtp.layers.0.self_attn.o_proj.weight");
        if (w_o.rows != D || w_o.cols != DQ)
            throw std::runtime_error("o_proj geometry mismatch in " + q4nx_path);
        std::vector<float> g_post_ln =
            q4nx.f32("model.mtp.layers.0.post_attention_layernorm.weight");
        if ((int)g_post_ln.size() != D)
            throw std::runtime_error("post_attention_layernorm geometry mismatch");

        std::vector<float> o_proj((size_t)D, 0.0f);
        cpu_ref::q4nx_matmul(w_o, cpu_y.data(), 1, o_proj.data());

        // The residual, then the norm. Both sides rounded to bf16 before the
        // sum and again before the reduction: the array adds what the DMA
        // delivered and reduces over what the adder wrote, so leaving these in
        // fp32 would read out as datapath error the hardware never committed.
        // (memory: aie-to-vector-bf16-floors)
        std::vector<float> h((size_t)D);
        for (int i = 0; i < D; i++)
            h[i] = cpu_ref::round_bf16(cpu_ref::round_bf16(cpu.fc_out[i]) +
                                       cpu_ref::round_bf16(o_proj[i]));

        std::vector<float> ref_norm((size_t)D, 0.0f);
        cpu_ref::rms_norm(h.data(), g_post_ln.data(), D, ref_norm.data());

        // ---- the MLP's SwiGLU: hid = silu(gate(yn)) * up(yn) ---------------
        //
        // yn is rounded to bf16 first. The array's up|gate mvm reduces over what
        // the DMA delivered to the projection cores, so reducing over the
        // unrounded norm here would read out as datapath error the hardware
        // never committed. (memory: aie-to-vector-bf16-floors)
        const int I = lm_config.intermediate_size;
        cpu_ref::q4nx_tensor w_up =
            cpu_ref::get_q4nx(q4nx, "model.mtp.layers.0.mlp.up_proj.weight");
        cpu_ref::q4nx_tensor w_gate =
            cpu_ref::get_q4nx(q4nx, "model.mtp.layers.0.mlp.gate_proj.weight");
        if (w_up.rows != I || w_up.cols != D || w_gate.rows != I || w_gate.cols != D)
            throw std::runtime_error("up/gate geometry mismatch in " + q4nx_path);

        // No isolated score for this stage. It would need the NPU's own
        // post-attention norm as the input, and that row is no longer
        // observable: it became up|gate's x and the rms tile's y port had no
        // room for a second destination. The swapped control below is what
        // stands in for it. (memory: score-a-projection-against-its-own-input)
        std::vector<float> yn_r((size_t)D);
        for (int i = 0; i < D; i++) yn_r[i] = cpu_ref::round_bf16(ref_norm[i]);

        std::vector<float> up((size_t)I, 0.0f), gate((size_t)I, 0.0f);
        cpu_ref::q4nx_matmul(w_up,   yn_r.data(), 1, up.data());
        cpu_ref::q4nx_matmul(w_gate, yn_r.data(), 1, gate.data());

        // The activation goes on **gate**, and the product is with up. Both
        // operands are rounded to bf16 first: the GLU tile reads what the y tree
        // delivered, so reducing over the unrounded values here would read out
        // as datapath error the hardware never committed.
        std::vector<float> hid((size_t)I), hid_swapped((size_t)I);
        for (int i = 0; i < I; i++) {
            const float g = cpu_ref::round_bf16(gate[i]);
            const float u = cpu_ref::round_bf16(up[i]);
            hid[i]         = cpu_ref::siluf_(g) * u;
            hid_swapped[i] = cpu_ref::siluf_(u) * g;
        }

        // ---- down_proj: down = W_down @ hid --------------------------------
        //
        // hid is rounded to bf16 first, for the same reason yn was: the array's
        // down mvm reduces over what the GLU tile's MM2S delivered to the
        // projection cores. (memory: aie-to-vector-bf16-floors)
        cpu_ref::q4nx_tensor w_down =
            cpu_ref::get_q4nx(q4nx, "model.mtp.layers.0.mlp.down_proj.weight");
        if (w_down.rows != D || w_down.cols != I)
            throw std::runtime_error("down_proj geometry mismatch in " + q4nx_path);

        auto down_of = [&](const std::vector<float>& h_in) {
            std::vector<float> h_r((size_t)I);
            for (int i = 0; i < I; i++) h_r[i] = cpu_ref::round_bf16(h_in[i]);
            std::vector<float> out((size_t)D, 0.0f);
            cpu_ref::q4nx_matmul(w_down, h_r.data(), 1, out.data());
            return out;
        };

        std::vector<float> down = down_of(hid);

        // ---- the MLP residual: h2 = h + down -------------------------------
        //
        // Both operands rounded to bf16 before the sum, for the same reason the
        // attention residual's were: the rms tile adds what the DMA delivered to
        // what residual_add left in x_buf. (memory: aie-to-vector-bf16-floors)
        std::vector<float> h2((size_t)D);
        for (int i = 0; i < D; i++)
            h2[i] = cpu_ref::round_bf16(cpu_ref::round_bf16(h[i]) +
                                        cpu_ref::round_bf16(down[i]));

        // End to end: the whole chain from the fusion input to here, in one
        // number. It carries every upstream stage's error -- the fc, the qkv
        // projection, the rotation, the softmax, the gate, o_proj, the two
        // norms, the SwiGLU and down.
        buffer<bf16> ref_h2 = cpu_ref::to_bf16(h2);
        header_print("info", "=== h2 = h + down -- the whole layer, end to end ===");
        print_error_metrics(get_error_metrics(got_layer_out, ref_h2));
    }
}

} // namespace

int main(int argc, char** argv) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()
        ("model,m", arg_utils::po::value<std::string>()->default_value("QWEN3_8_27B"),
            "Model type, e.g., QWEN3_8_27B")
        ("ref", arg_utils::po::value<std::string>()->default_value(""),
            "Reference dump, relative to the model path. Supplies this step's "
            "input_ids and prev_hidden; the scored reference is computed on the "
            "CPU from model.q4nx. Empty picks the model's own file from models.h.");

    arg_utils::parse_options(argc, argv, desc, vm);

    const std::string model_type = vm["model"].as<std::string>();
    const std::string ref_arg    = vm["ref"].as<std::string>();

    header_print("info", "Testing model: " + model_type);
    model_list::model_t model = model_list::get_model_type(model_type);
    if (model == model_list::UNKNOWN_MODEL) {
        std::cerr << "Unknown model type: " << model_type << std::endl;
        return 1;
    }

    const std::string model_path = model_list::get_model_path(model);
    header_print("info", "Model path: " + model_path);

    LM_Config lm_config;
    lm_config.from_pretrained(model_path);
    lm_config.layer_xclbin_name = model_list::get_layer_xclbin_path(model);
    header_print("info", "Layer xclbin path: " + lm_config.layer_xclbin_name);

    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    header_print("info", "Initializing NPU manager...");
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* app_mgr = xclbin_mgr.register_xclbin(lm_config.layer_xclbin_name);

    const std::string ref_path = ref_arg.empty() ? model_list::get_reference_path(model)
                                                 : model_path + "/" + ref_arg;

    run_fc_test(model, lm_config, *q4nx_model, app_mgr, model_path, ref_path);

    return 0;
}
