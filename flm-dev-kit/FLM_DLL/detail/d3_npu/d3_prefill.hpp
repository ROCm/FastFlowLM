// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __D3_PREFILL_HPP__
#define __D3_PREFILL_HPP__
/// \file d3_prefill.hpp
/// \brief The d3 decision encoder: the Qwen3.5-0.8B text tower run over a *packed batch* of
///        independent prompts, with bidirectional full attention.
///
/// Derived from minicpm_v_4_7_npu's prefill graph (same text tower, same weights layout, same
/// host kernels). What is different, and why:
///
///  * **Packed segments.** `d3_batch` concatenates the rows of several prompts. Every row-wise
///    stage (all projections, RMS norms, q/k norm + RoPE, gates, GLU, residual adds) runs once
///    over all rows; the three stages that mix rows -- causal conv1d, the gated delta rule and
///    attention -- run per segment, each from a zero state. The array GEMM pads rows to 512
///    (gen_dequant_mm_512), so packing is what fills that padding.
///  * **Fresh state, no caches.** Every segment starts from zero conv / recurrent state and
///    attention reads its own k/v rows directly, so there is no KV cache, no context length
///    and nothing to clear between calls.
///  * **Bidirectional full attention** (decision_config.json `noncausal_full_attention`):
///    every query of a segment sees every key of the same segment; the gated delta-net layers
///    stay causal. Upstream: d3_runtime.PromptEncoder.
///  * **No bitstream switches.** Only mm.xclbin is registered; conv1d, the delta rule and
///    attention run on the host (notes/tricks/2026-10-05-minicpm-v-conv1d-prefill-belongs-on-the-host.md: on this array a token-serial stage
///    does not amortize its bitstream switch). `make PREFILL_MM=cpu` runs everything on the host.
///  * **No lm_head, no decode.** The caller gets the final-norm hidden state of each segment's
///    last row; the 255-way readout lives in the application.
#include <climits>
#include <filesystem>
#include "d3_npu_sequence.hpp"
#include "modules/gemm.hpp"
#include "d3_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "d3_prefill_cpu.hpp"

// ---------------------------------------------------------------------------
// Prefill profiling: -DD3_PREFILL_PROFILE (zero cost otherwise). Same macros as the
// qwen3_5vl / minicpm engines, so one log parser reads all three.
// ---------------------------------------------------------------------------
#ifdef D3_PREFILL_PROFILE
#define PREFILL_PROF_INIT() time_utils::time_point __prof_t = time_utils::now()
#define PREFILL_PROF_TICK(name)                                                \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::duration_us(__prof_t, __prof_now));\
        header_print("prefill-prof", name << ": " << __prof_d.first << " "     \
                                          << __prof_d.second);                 \
        __prof_t = __prof_now;                                                 \
    } while (0)
#else
#define PREFILL_PROF_INIT() do {} while (0)
#define PREFILL_PROF_TICK(name) do {} while (0)
#endif

// ---------------------------------------------------------------------------
// Rig dumps, named like the checkpoints models_simple/verify_d3_lite.py saves. -DD3_RIG compiles them in; D3_RIG_DUMP=<dir>
// turns them on. Only segment 0 of a batch is dumped, rows [0, len).
// ---------------------------------------------------------------------------
#ifdef D3_RIG
#define D3_RIG_DUMP(name, ptr, rows, cols, ld) d3_cpu::rig_dump((name), (ptr), (rows), (cols), (ld))
#else
#define D3_RIG_DUMP(name, ptr, rows, cols, ld) do {} while (0)
#endif

/// @brief round a GEMM input to the array's bfp16 grid (d3_cpu::bfp16_round_rows) when the
///        array runs the GEMMs; D3_BFP_ROUND=0 turns it off (rig comparisons)
inline void d3_round_gemm_input(bf16* p, int ld, int rows, int cols) {
#if D3_MM_ON_NPU
    static const bool on = [] {
        const char* e = std::getenv("D3_BFP_ROUND");
        return !(e && e[0] == '0');
    }();
    if (on) d3_cpu::bfp16_round_rows(p, ld, rows, cols);
#else
    (void)p; (void)ld; (void)rows; (void)cols;
#endif
}

/// @brief whether one of the gated-input projections runs on the host in an NPU build:
///        D3_HOST_GEMMS=gdn_o,attn_o,down (any subset). Those three read an elementwise product
///        (gated norm, sigmoid gate x attention, up x silu(gate)), which the array's bfp16 MAC
///        handles worst (notes/tricks/2026-10-02-prefill-mm-overlay-loses-accuracy-on-gated-inputs.md).
inline bool d3_host_gemm(const char* which) {
#if D3_MM_ON_NPU
    static const std::string list = [] {
        const char* e = std::getenv("D3_HOST_GEMMS");
        return std::string(",") + (e ? e : "") + ",";
    }();
    return list.find(std::string(",") + which + ",") != std::string::npos;
#else
    (void)which;
    return true;
#endif
}

/// @brief prompts whose attention work reaches that of a prompt of this many tokens run it on the
///        array (attn.xclbin), smaller ones on the host. Work is the number of query-key pairs
///        (d3_attn_work): len * len for a full bidirectional prompt, so the knob stays "tokens" for
///        the lite / nano / mini prompts. D3_ATTN_ARRAY_MIN=<tokens> overrides it, 0 = never (rig).
///        The default is the measured crossover (Strix Halo, one question, median of 3 x 5):
///        array / host end to end 1.10 at 512 tokens, 0.99 at 1,024, 1.06 at 1,280, 0.97 at 1,536,
///        0.84 at 2,048, 0.76 at 4,096, 0.40 at 16,384. Below it the array's fixed costs per
///        attention layer (the switch to attn.xclbin, staging K/V) outweigh its speed. Not re-swept
///        for prefix_bidir prompts or GQA 6.
static constexpr int D3_ATTN_ARRAY_MIN_DEFAULT = 1536;
inline int d3_attn_array_min() {
#if D3_MM_ON_NPU
    static const int n = [] {
        const char* e = std::getenv("D3_ATTN_ARRAY_MIN");
        if (!e) return D3_ATTN_ARRAY_MIN_DEFAULT;
        const int v = std::atoi(e);
        return v > 0 ? v : INT_MAX;
    }();
    return n;
#else
    return INT_MAX;
#endif
}

/// @brief query-key pairs one prompt's full-attention layer scores: len * len without a state
///        split, state * state + (len - state) * len with one (state in 1..len-1)
inline int64_t d3_attn_work(int len, int state) {
    if (state <= 0 || state >= len) return (int64_t)len * len;
    return (int64_t)state * state + (int64_t)(len - state) * len;
}

/// @brief rows [begin, begin + len) of a packed batch hold one prompt
struct d3_segment {
    int begin;
    int len;
    int state = 0;   //!< prefix_bidir: leading rows forming the state block, 0 < state < len; 0 = no split
    // state-once (encode_state_once): the segment is a state block whose per-layer results are saved
    // (save), or a question block continuing from the saved state (dependent). Both false: a prompt.
    bool save = false;
    bool dependent = false;
};

/// @brief one layer's share of a state-once cache: what a question block needs of the state block.
///        Plain host heap (not device buffers): the question attention is host code, and the cache
///        would count against the device buffer budget for nothing.
struct d3_layer_cache {
    std::vector<bf16> k, v;    //!< full-attention layer: the state's keys (after k_norm + RoPE) and values, [S][DK] each
    std::vector<bf16> tail;    //!< linear layer: the last KS - 1 rows of the state's in_proj_qkv output (conv input), [KS - 1][DQKV]
    std::vector<float> S;      //!< linear layer: the recurrent state after the state's last row, [Hv][DK][DV]
};
struct d3_state_cache {
    int state_len = 0;
    std::vector<d3_layer_cache> layers;   //!< per decoder layer
};

/// @brief the shape of one packed batch
struct d3_batch {
    std::vector<d3_segment> segs;
    int R = 0;            //!< real rows, the sum of the segment lengths
    int R_padded = 0;     //!< R rounded up to the GEMM row tile (512)
    std::vector<int> input_idx;   //!< 3 RoPE positions per row (t, h, w); text: all = pos in segment
    d3_state_cache* cache = nullptr;   //!< state-once: written by the `save` segment, read by the `dependent` ones

    /// @param states per prompt state length (prefix_bidir), empty = none; values outside 1..len-1 mean no split
    void build(const std::vector<int>& lens, const std::vector<int>& states = {}) {
        segs.clear();
        cache = nullptr;
        R = 0;
        for (size_t i = 0; i < lens.size(); i++) {
            const int n = lens[i];
            const int st = (i < states.size() && states[i] > 0 && states[i] < n) ? states[i] : 0;
            segs.push_back({R, n, st});
            R += n;
        }
        R_padded = (R + d3_npu_sequence::MM_ROWS - 1) / d3_npu_sequence::MM_ROWS * d3_npu_sequence::MM_ROWS;
        input_idx.assign((size_t)R * 3, 0);
        for (const auto& s : segs) {
            for (int l = 0; l < s.len; l++) {
                int* p = input_idx.data() + (size_t)(s.begin + l) * 3;
                p[0] = p[1] = p[2] = l;
            }
        }
    }

    /// @brief a state-once pass: with `with_state` segment 0 is the state block (unsplit bidirectional
    ///        attention over its rows, results saved into `c`), every other segment is a question block
    ///        continuing from `c` with RoPE positions pos0 + l on the three axes. Without `with_state` all
    ///        segments are question blocks of a state already in `c`. The state's own positions are text
    ///        positions here; the caller overwrites them for images.
    void build_state_once(const std::vector<int>& lens, bool with_state, int pos0, d3_state_cache* c) {
        build(lens);
        cache = c;
        for (size_t i = 0; i < segs.size(); i++) {
            if (with_state && i == 0) { segs[i].save = true; continue; }
            segs[i].dependent = true;
            for (int l = 0; l < segs[i].len; l++) {
                int* p = input_idx.data() + (size_t)(segs[i].begin + l) * 3;
                p[0] = p[1] = p[2] = pos0 + l;
            }
        }
    }
};

/// @brief every prefill scratch buffer of one row bucket (R_padded rows): device BOs for what the
///        array reads or writes, host memory for the rest. d3_prefill_context keeps a few buckets.
struct d3_scratch {
    int R_padded = 0;
    size_t bytes = 0;
    uint64_t last_use = 0;
    buffer<bf16> x, gate;     //!< block input (normed residual); the GDN output gate
    buffer<bf16> after_mix;   //!< the mixer's output: [R, Linear_DV] (delta rule) or [R, DQ] (attention)
    buffer<bf16> qkv;         //!< [R, DQKV] in_proj_qkv output
    buffer<bf16> conv_qkv;    //!< [R, DQKV] conv + silu, q/k L2-normalised (host)
    buffer<float> alpha_beta; //!< (host)
    buffer<bf16> q, k, v, g;
    buffer<bf16> g_sigmoid;   //!< (host)
    buffer<bf16> upgate;      //!< array: interleaved 512-row up/gate slabs; host: [ up | silu(gate) ]
    buffer<bf16> upgate_hi;   //!< array, over d3_upgate_bo_limit(): the slabs past the first part (see d3_mlp_block)
    buffer<bf16> hid;
};

/// @brief the largest device buffer one array run may write. The XRT shim mmaps a BO with a signed 32-bit
///        length: d3-turbo's up/gate result for a 32,768-row pass, [32768, 2 * 17408] bf16 = 2.28 GB, fails to
///        allocate ("mmap_range(len=-2013265920) failed"), while 2.139 GB (30,720 rows) works. Passes whose
///        up/gate result would pass the limit run it as two column parts (d3_mlp_block).
///        D3_BO_LIMIT_MB=<n> lowers it (tests: forces the two-part GEMM on small models; the result is bit-identical).
inline size_t d3_bo_limit() {
    static const size_t v = [] {
        size_t limit = ((size_t)1 << 31) - ((size_t)1 << 24);
        if (const char* e = std::getenv("D3_BO_LIMIT_MB")) {
            const long mb = std::atol(e);
            if (mb > 0 && (size_t)mb << 20 < limit) limit = (size_t)mb << 20;
        }
        return limit;
    }();
    return v;
}

/// @brief the up/gate GEMM as column parts: {slice pairs of part 0, of part 1}; the second is 0 when
///        one buffer holds the result. A pair is a 512-wide up slice and its gate slice (1,024 columns).
inline std::pair<int, int> d3_upgate_pairs(size_t R_padded, size_t intermediate_size) {
    const int pairs = (int)(intermediate_size / 512);
    if (R_padded * intermediate_size * 2 * sizeof(bf16) <= d3_bo_limit()) return {pairs, 0};
    const int p0 = (pairs + 1) / 2;
    if (R_padded * (size_t)p0 * 1024 * sizeof(bf16) > d3_bo_limit())
        throw std::length_error("d3_npu: a pass of " + std::to_string(R_padded) + " rows does not fit the device buffer limit");
    return {p0, pairs - p0};
}

// ---------------------------------------------------------------------------
// The projections. A d3_mm is one array GEMM. Its instruction sequence depends on the batch's
// row bucket (R_padded), and turning a sequence into a runnable kernel (aiebu ELF, xrt module,
// kernel) is expensive, so every bucket up to the cached rows (d3_prefill_context::cache_rows)
// keeps its own app for the life of the engine: at 4,096 rows eight buckets, 5 MB of instructions
// for all ten projections. Instructions grow with the rows (8 MB at 32,768), so a longer prompt's
// bucket gets apps for its call only; kept, every bucket up to 32,768 rows would be 0.26 GB.
// ---------------------------------------------------------------------------
struct d3_mm {
    npu_app_manager* mgr = nullptr;
    std::map<int64_t, npu_app> apps;   //!< per R_padded (attention: per sequence key)
    npu_app* cur = nullptr;

    void build(npu_app_manager* m) { mgr = m; }

    /// @brief make R_padded's app current; true when it is new and its sequence must be generated
    bool select(int64_t R_padded) {
        auto it = apps.find(R_padded);
        const bool fresh = (it == apps.end());
        if (fresh) it = apps.emplace(R_padded, mgr->create_app()).first;
        cur = &it->second;
        return fresh;
    }

    /// @brief drop the apps of buckets over `rows`
    void drop_above(int64_t rows) {
        for (auto it = apps.upper_bound(rows); it != apps.end();) {
            if (cur == &it->second) cur = nullptr;
            it = apps.erase(it);
        }
    }
    npu_sequence* seq() { return cur->seq(); }
    template <typename... A> auto operator()(A&... a) { return (*cur)(a...); }
    template <typename... A> auto create_run(A&... a) { return cur->create_run(a...); }
};

struct d3_linear_block{
    d3_desc* desc;
    d3_npu_sequence* seq_gen;
    d3_scratch* sc = nullptr;
    int R_padded_old = -1;
    int layer_idx = -1;

    d3_mm qkv_proj, g_proj, o_proj;

    std::vector<bf16> conv_in;   //!< one segment's conv input: KS - 1 zero state rows, then its rows
    std::vector<float> S;        //!< one segment's recurrent state [Hv][DK][DV], zeroed (or copied from a state-once cache) per segment

    d3_linear_block(d3_desc* desc, d3_npu_sequence* seq_gen, npu_app_manager* gemm)
        : desc(desc), seq_gen(seq_gen) {
        qkv_proj.build(gemm);
        g_proj.build(gemm);
        o_proj.build(gemm);
        S.resize((size_t)desc->linear_num_v_heads * desc->linear_qk_head_dim * desc->linear_v_head_dim);
    }

    d3_linear_layer& weights() {
        return static_cast<d3_linear_layer&>(*desc->weight_desc[desc->lin_idx]);
    }

    void reserve(int R_padded) {
        if (R_padded_old == R_padded) return;
#if D3_MM_ON_NPU
        if (qkv_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(qkv_proj.seq(), R_padded, desc->D, desc->Linear_DQKV, desc->get_linear_qkv_offset(), 0, 0);
        // rtp_act = 0: the array's fused silu (rtp_act = 1) is off by up to ~0.015 absolute for
        // negative inputs (-0.055 for -0.042), and the gated norm amplifies that per channel.
        // _in_proj applies the silu on the host instead.
        if (g_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(g_proj.seq(), R_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 0);
        if (o_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(o_proj.seq(), R_padded, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0);
#endif
        R_padded_old = R_padded;
    }

    void drop_above(int rows) {
        for (d3_mm* m : {&qkv_proj, &g_proj, &o_proj}) m->drop_above(rows);
        if (R_padded_old > rows) R_padded_old = -1;
        // conv_in only grows: give back a longer prompt's (0.4 GB at 32,768 rows)
        if (conv_in.capacity() > (size_t)rows * desc->Linear_DQKV + 64 * desc->Linear_DQKV) std::vector<bf16>().swap(conv_in);
    }

    void _in_proj(const d3_batch& b, buffer<u8>& proj_weights) {
        buffer<bf16>& qkv = sc->qkv;
#if D3_MM_ON_NPU
        qkv_proj(qkv, sc->x, proj_weights);
        g_proj(sc->gate, sc->x, proj_weights);
        qkv.sync_from_device();
        sc->gate.sync_from_device();
        D3_RIG_DUMP(_n("gdn.z"), sc->gate.data(), b.segs[0].len, desc->Linear_DV, desc->Linear_DV);
        d3_cpu::silu_inplace(sc->gate.data(), (size_t)b.R * desc->Linear_DV);
        d3_cpu::scratch_push(sc->gate);   // keep the BO coherent (scratch discipline)
#else
        const int DQKV = desc->Linear_DQKV;
        d3_cpu::dequant_mm(qkv.data(), DQKV, proj_weights.data() + desc->get_linear_qkv_offset(),
                           sc->x.data(), desc->D, b.R, desc->D, DQKV, d3_cpu::BODY_LAYOUT);
        d3_cpu::dequant_mm(sc->gate.data(), desc->Linear_DV, proj_weights.data() + desc->get_linear_g_offset(),
                           sc->x.data(), desc->D, b.R, desc->D, desc->Linear_DV, d3_cpu::BODY_LAYOUT);
        D3_RIG_DUMP(_n("gdn.z"), sc->gate.data(), b.segs[0].len, desc->Linear_DV, desc->Linear_DV);
        d3_cpu::silu_inplace(sc->gate.data(), (size_t)b.R * desc->Linear_DV);
#endif
    }

    std::string _n(const char* s) const { return "l" + std::to_string(layer_idx) + "." + s; }

    buffer<bf16> forward(const d3_batch& b, buffer<u8>& proj_weights, buffer<bf16>& ssm_weights,
                         buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
        PREFILL_PROF_INIT();
        buffer<bf16>& qkv = sc->qkv;
        buffer<bf16>& conv_qkv = sc->conv_qkv;
        buffer<float>& alpha_beta = sc->alpha_beta;
        const int DQKV = desc->Linear_DQKV;
        const int KS = desc->conv_kernel_size;
        const int Hv = desc->linear_num_v_heads;
        reserve(b.R_padded);

        _in_proj(b, proj_weights);
        D3_RIG_DUMP(_n("gdn.qkv"), qkv.data(), b.segs[0].len, DQKV, DQKV);
        D3_RIG_DUMP(_n("gdn.z_silu"), sc->gate.data(), b.segs[0].len, desc->Linear_DV, desc->Linear_DV);
        PREFILL_PROF_TICK("linear:qkv+g_proj");

        // causal conv1d + silu per segment, from a zero conv state (state-once: a question block continues
        // from the state's conv tail, the state saves it)
        const bf16* conv_w = ssm_weights.data() + weights().ssm_conv1d.offset / sizeof(bf16);
        d3_layer_cache* lc = b.cache ? &b.cache->layers[layer_idx] : nullptr;
        for (const auto& s : b.segs) {
            if (s.save) lc->tail.assign((size_t)(KS - 1) * DQKV, bf16(0.0f));
            d3_cpu::conv1d_silu_carry(conv_qkv.data() + (size_t)s.begin * DQKV, qkv.data() + (size_t)s.begin * DQKV,
                                      s.dependent ? lc->tail.data() : nullptr, conv_w, s.len, DQKV, KS, conv_in,
                                      s.save ? lc->tail.data() : nullptr);
        }
        D3_RIG_DUMP(_n("gdn.conv"), conv_qkv.data(), b.segs[0].len, DQKV, DQKV);
        d3_cpu::l2norm_heads(conv_qkv.data(), DQKV, b.R,
                             (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                             desc->linear_qk_head_dim);
        PREFILL_PROF_TICK("linear:conv1d");

        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        cpu_func::_cpu_mm<bf16>(alpha_ptr, alpha_proj_cpu.data(), sc->x.data(), b.R, desc->D, Hv);
        cpu_func::_cpu_mm<bf16>(beta_ptr,  beta_proj_cpu.data(),  sc->x.data(), b.R, desc->D, Hv);
        D3_RIG_DUMP(_n("gdn.alpha"), alpha_ptr, b.segs[0].len, Hv, Hv);
        D3_RIG_DUMP(_n("gdn.beta"), beta_ptr, b.segs[0].len, Hv, Hv);
        cpu_func::_activate_a(alpha_ptr,
                              (float*)(ssm_weights.data() + desc->get_linear_A_offset() / sizeof(bf16)),
                              (float*)(ssm_weights.data() + desc->get_linear_dt_offset() / sizeof(bf16)),
                              b.R, Hv);
        cpu_func::_activate_b(beta_ptr, b.R, Hv);
        PREFILL_PROF_TICK("linear:alpha_beta");

        // the gated delta rule per segment, from a zero recurrent state (state-once: a question block continues
        // from the state's, which the state segment saves)
        for (const auto& s : b.segs) {
            d3_cpu::gated_delta_net_carry(
                sc->after_mix.data() + (size_t)s.begin * desc->Linear_DV, S.data(), s.dependent ? lc->S.data() : nullptr,
                conv_qkv.data() + (size_t)s.begin * DQKV, DQKV,
                0, desc->Linear_DQ, desc->Linear_DQ + desc->Linear_DK,
                alpha_ptr + (size_t)s.begin * Hv, beta_ptr + (size_t)s.begin * Hv, s.len,
                desc->linear_num_qk_heads, Hv, desc->linear_qk_head_dim, desc->linear_v_head_dim);
            if (s.save) lc->S.assign(S.begin(), S.end());
        }
        D3_RIG_DUMP(_n("gdn.core"), sc->after_mix.data(), b.segs[0].len, desc->Linear_DV, desc->Linear_DV);
        PREFILL_PROF_TICK("linear:deltanet");

        bf16* gate_rms_norm_weight_ptr = ssm_weights.data() + desc->get_linear_ssm_norm_offset() / sizeof(bf16);
        const int dv = desc->Linear_DV / Hv;
        d3_for(b.R, [&](int ll) {
            for (int h = 0; h < Hv; h++) {
                const size_t off = (size_t)ll * desc->Linear_DV + (size_t)h * dv;
                cpu_func::_gated_norm(sc->after_mix.data() + off, sc->gate.data() + off,
                                      gate_rms_norm_weight_ptr, desc->linear_v_head_dim, 1);
            }
        });
        D3_RIG_DUMP(_n("gdn.norm"), sc->after_mix.data(), b.segs[0].len, desc->Linear_DV, desc->Linear_DV);
        PREFILL_PROF_TICK("linear:gated_norm");

#if D3_MM_ON_NPU
        if (!d3_host_gemm("gdn_o")) {
            d3_round_gemm_input(sc->after_mix.data(), desc->Linear_DV, b.R, desc->Linear_DV);
            sc->after_mix.sync_to_device();
            o_proj(sc->x, sc->after_mix, proj_weights);
            sc->x.sync_from_device();
        } else
#endif
        {
            d3_cpu::dequant_mm(sc->x.data(), desc->D, proj_weights.data() + desc->get_linear_o_offset(),
                               sc->after_mix.data(), desc->Linear_DV, b.R, desc->Linear_DV, desc->D, d3_cpu::BODY_LAYOUT);
            d3_cpu::scratch_push(sc->after_mix);   // the gated norm above wrote it on the host
            d3_cpu::scratch_push(sc->x);
        }
        D3_RIG_DUMP(_n("gdn.out"), sc->x.data(), b.segs[0].len, desc->D, desc->D);
        PREFILL_PROF_TICK("linear:o_proj");
        return buffer<bf16>(sc->x.data(), (size_t)b.R * desc->D);
    }
};

struct d3_attn_block{
    d3_desc* desc;
    d3_npu_sequence* seq_gen;
    d3_scratch* sc = nullptr;
    int R_padded_old = -1;
    int layer_idx = -1;
    bool causal = false;   //!< false: decision_config noncausal_full_attention (d3); true only for the rig's causal baseline

    d3_mm q_proj, k_proj, v_proj, g_proj, o_proj;
    std::vector<float> kt_buf;   //!< d3_cpu::attention's K^T scratch, kept across calls
    std::vector<float> st_kt;    //!< state-once: the cached state's K^T for the pass's question blocks (freed by drop_above)

    // Array attention (attn.xclbin) for prompts of at least d3_attn_array_min() tokens. The
    // bitstream is registered on the first such prompt, so a server that only sees short prompts
    // keeps one hardware context; if it cannot be registered (missing file, the device's
    // hardware contexts all taken) attention stays on the host, with one log line.
    npu_xclbin_manager* npu = nullptr;
    std::string attn_xclbin;
    npu_app_manager* attn_mgr = nullptr;
    bool attn_unavailable = false;
    d3_mm attn;                        //!< an app per prompt length (len, or state << 32 | len for prefix_bidir's question run) in the current call (one per attention layer)
    d3_mm attn_state;                  //!< prefix_bidir's state run, an app per state length
    buffer<bf16> st_kv, st_q, st_o;    //!< staging: K then V always; q / o for a segment not at row 0
    buffer<bf16> st_kv_state;          //!< prefix_bidir: the state's own K then V

    d3_attn_block(d3_desc* desc, d3_npu_sequence* seq_gen, npu_app_manager* gemm, npu_xclbin_manager* npu,
                  std::string attn_xclbin)
        : desc(desc), seq_gen(seq_gen), npu(npu), attn_xclbin(std::move(attn_xclbin)) {
        q_proj.build(gemm); k_proj.build(gemm); v_proj.build(gemm); g_proj.build(gemm); o_proj.build(gemm);
    }

    std::string _n(const char* s) const { return "l" + std::to_string(layer_idx) + "." + s; }

    void reserve(int R_padded) {
        if (R_padded_old == R_padded) return;
#if D3_MM_ON_NPU
        if (q_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(q_proj.seq(), R_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0);
        if (k_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(k_proj.seq(), R_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0);
        if (v_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(v_proj.seq(), R_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0);
        if (g_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(g_proj.seq(), R_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0);
        if (o_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(o_proj.seq(), R_padded, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0);
#endif
        R_padded_old = R_padded;
    }

    void drop_above(int rows) {
        for (d3_mm* m : {&q_proj, &k_proj, &v_proj, &g_proj, &o_proj}) m->drop_above(rows);
        if (R_padded_old > rows) R_padded_old = -1;
        // kt_buf only grows: give back a longer prompt's (0.1 GB at 32,768 rows)
        const size_t kt_rows = (size_t)desc->num_kv_heads * desc->DH + (size_t)d3_cpu::prefill_threads() * d3_cpu::ATT_NV;
        if (kt_buf.capacity() > kt_rows * (size_t)(rows + 16)) std::vector<float>().swap(kt_buf);
        std::vector<float>().swap(st_kt);
        // array attention staging and its sequence (instructions grow with the rows)
        if (st_kv.size() > (size_t)2 * rows * desc->DK) st_kv = buffer<bf16>();
        if (st_kv_state.size() > (size_t)2 * rows * desc->DK) st_kv_state = buffer<bf16>();
        // prefix_bidir stages up to S + ceil128(len - S) rows of q / o
        if (st_q.size() > (size_t)(rows + 128) * desc->DQ) st_q = buffer<bf16>();
        if (st_o.size() > (size_t)(rows + 128) * desc->DQ) st_o = buffer<bf16>();
        // per prompt length, so worth keeping only within a call
        attn.drop_above(-1);
        attn_state.drop_above(-1);
    }

    /// @brief attn.xclbin registered (now, on first use); false: attention stays on the host
    bool _array_ready() {
        if (attn_mgr) return true;
        if (attn_unavailable) return false;
        try {
            if (!std::filesystem::exists(attn_xclbin)) throw std::runtime_error(attn_xclbin + " not found");
            attn_mgr = npu->register_xclbin(attn_xclbin);
            attn.build(attn_mgr);
            attn_state.build(attn_mgr);
            return true;
        } catch (const std::exception& e) {
            header_print("d3", "attention stays on the host for long prompts: cannot load attn.xclbin (" << e.what() << ")");
            attn_unavailable = true;
            return false;
        }
    }

    /// @brief bidirectional attention of segment s on the array into sc->after_mix. Writes
    ///        ceil128(len) rows of output; past the segment end that is the next segment's rows
    ///        (rewritten by its own attention afterwards) or padding.
    ///
    ///        prefix_bidir (s.state = S > 0): two runs of the same kernel. Run A is the state over
    ///        itself (S query rows, S keys) with its own K/V staging, so it is exactly a standalone
    ///        S-token prompt (and its sequence depends on S only). Run B is the question rows over
    ///        all keys (q_base = o_base = S rows, len - S query rows, len keys). A comes first: it
    ///        writes whole 128-row chunks, i.e. rows [S, ceil128(S)) of the output belong to B and
    ///        are overwritten by it; B writes whole chunks too, up to S + ceil128(len - S) rows.
    void _array_attention(const d3_segment& s, const d3_batch& b) {
        const size_t R = d3_npu_sequence::ATTN_ROWS;
        auto up = [R](size_t n) { return (n + R - 1) / R * R; };
        const size_t L = s.len, S = s.state;   // S = 0: no split
        const size_t Lc = up(L);
        const size_t DQ = desc->DQ, DK = desc->DK;
        // rows of q / o the runs touch (B reaches past A: S + ceil128(L - S) >= S + 128 > ceil128(S))
        const size_t rows = S ? S + up(L - S) : Lc;
        // q / o in place when the segment starts at row 0 (its chunks stay inside the bucket),
        // else staged
        const bool direct = s.begin == 0 && rows <= (size_t)b.R_padded;
        if (st_kv.size() < 2 * Lc * DK) st_kv = npu->create_bo_buffer<bf16>(2 * Lc * DK);
        if (!direct && st_q.size() < rows * DQ) st_q = npu->create_bo_buffer<bf16>(rows * DQ);
        if (!direct && st_o.size() < rows * DQ) st_o = npu->create_bo_buffer<bf16>(rows * DQ);
        const size_t kv_half = Lc * DK;   // V right after this prompt's K: the sequence depends on len only
        // K then V, rows past the segment zero. Padding V enters P * V as 0 * V, and the array's
        // bfp16 blocks share an exponent along the keys, so it must be finite and small.
        bf16* kd = st_kv.data();
        bf16* vd = st_kv.data() + kv_half;
        memcpy(kd, sc->k.data() + (size_t)s.begin * DK, L * DK * sizeof(bf16));
        memcpy(vd, sc->v.data() + (size_t)s.begin * DK, L * DK * sizeof(bf16));
        memset(kd + L * DK, 0, (Lc - L) * DK * sizeof(bf16));
        memset(vd + L * DK, 0, (Lc - L) * DK * sizeof(bf16));
        st_kv.sync_to_device();
        buffer<bf16>& qb = direct ? sc->q : st_q;
        buffer<bf16>& ob = direct ? sc->after_mix : st_o;
        if (!direct) {
            memcpy(st_q.data(), sc->q.data() + (size_t)s.begin * DQ, L * DQ * sizeof(bf16));
            memset(st_q.data() + L * DQ, 0, (rows - L) * DQ * sizeof(bf16));
            st_q.sync_to_device();
        }
        // the host wrote after_mix last (the previous layer's mixer): flush it before the array
        // writes it, or its dirty lines land over the array's output at the sync below
        d3_cpu::scratch_push(ob);
        if (!S) {
            if (attn.select((int64_t)L)) seq_gen->gen_bidirectional_attention(attn.seq(), (uint32_t)L, 0, 0, kv_half);
            attn(ob, qb, st_kv);
        } else {
            // run A: the state over itself, K / V of its own (rows past S zero, as for a prompt of S tokens)
            const size_t Sc = up(S);
            if (st_kv_state.size() < 2 * Sc * DK) st_kv_state = npu->create_bo_buffer<bf16>(2 * Sc * DK);
            bf16* ka = st_kv_state.data();
            bf16* va = st_kv_state.data() + Sc * DK;
            memcpy(ka, kd, S * DK * sizeof(bf16));
            memcpy(va, vd, S * DK * sizeof(bf16));
            memset(ka + S * DK, 0, (Sc - S) * DK * sizeof(bf16));
            memset(va + S * DK, 0, (Sc - S) * DK * sizeof(bf16));
            st_kv_state.sync_to_device();
            if (attn_state.select((int64_t)S)) seq_gen->gen_bidirectional_attention(attn_state.seq(), (uint32_t)S, 0, 0, Sc * DK);
            attn_state(ob, qb, st_kv_state);
            // run B: the question rows over every key
            if (attn.select(((int64_t)S << 32) | (int64_t)L))
                seq_gen->gen_bidirectional_attention(attn.seq(), (uint32_t)L, S * DQ, S * DQ, kv_half, (uint32_t)(L - S));
            attn(ob, qb, st_kv);
        }
        ob.sync_from_device();
        if (!direct) memcpy(sc->after_mix.data() + (size_t)s.begin * DQ, st_o.data(), L * DQ * sizeof(bf16));
    }

    /// @brief q / gate / k / v projections, q_norm + k_norm + RoPE on q and k, sigmoid of the
    ///        gate. On the array each host stage overlaps the next projection run.
    void _qkvg(d3_batch& b, buffer<u8>& proj_weights, bf16* q_norm_weight, bf16* k_norm_weight) {
        buffer<bf16> &q = sc->q, &k = sc->k, &v = sc->v, &g = sc->g, &g_sigmoid = sc->g_sigmoid;
#if D3_MM_ON_NPU
        q_proj(q, sc->x, proj_weights);
        q.sync_from_device();
        auto g_run = g_proj.create_run(g, sc->x, proj_weights);
        g_run.start();
        cpu_func::_rope_rms_batch(q, desc->DQ, b.input_idx, b.R, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        flm::npu::wait_for_completion(g_run, "g_run");
        g.sync_from_device();

        auto k_run = k_proj.create_run(k, sc->x, proj_weights);
        k_run.start();
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), b.R * desc->DQ);
        flm::npu::wait_for_completion(k_run, "k_run");
        k.sync_from_device();

        auto v_run = v_proj.create_run(v, sc->x, proj_weights);
        v_run.start();
        cpu_func::_rope_rms_batch(k, desc->DK, b.input_idx, b.R, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        flm::npu::wait_for_completion(v_run, "v_run");
        v.sync_from_device();
        // q and k were rewritten in place on the host. Flush them now: left dirty, those cache
        // lines are written back over the next attention layer's projections at its
        // sync_from_device() (every attention layer after the first came back with k cos ~0).
        q.sync_to_device();
        k.sync_to_device();
#else
        const u8* w = proj_weights.data();
        const bf16* x = sc->x.data();
        const int D = desc->D;
        d3_cpu::dequant_mm(q.data(), desc->DQ, w + desc->get_attn_q_offset(), x, D, b.R, D, desc->DQ, d3_cpu::BODY_LAYOUT);
        d3_cpu::dequant_mm(g.data(), desc->DQ, w + desc->get_attn_g_offset(), x, D, b.R, D, desc->DQ, d3_cpu::BODY_LAYOUT);
        d3_cpu::dequant_mm(k.data(), desc->DK, w + desc->get_attn_k_offset(), x, D, b.R, D, desc->DK, d3_cpu::BODY_LAYOUT);
        d3_cpu::dequant_mm(v.data(), desc->DV, w + desc->get_attn_v_offset(), x, D, b.R, D, desc->DV, d3_cpu::BODY_LAYOUT);
        cpu_func::_rope_rms_batch(q, desc->DQ, b.input_idx, b.R, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        cpu_func::_rope_rms_batch(k, desc->DK, b.input_idx, b.R, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), b.R * desc->DQ);
#endif
    }

    buffer<bf16> forward(d3_batch& b, buffer<u8>& proj_weights, buffer<bf16>& rope_rms_weights) {
        PREFILL_PROF_INIT();
        reserve(b.R_padded);
        buffer<bf16> &q = sc->q, &k = sc->k, &v = sc->v, &g = sc->g, &g_sigmoid = sc->g_sigmoid;
        bf16* q_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM;
        bf16* k_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM + desc->DH;

        _qkvg(b, proj_weights, q_norm_weight, k_norm_weight);
        const int L0 = b.segs[0].len;
        D3_RIG_DUMP(_n("attn.q"), q.data(), L0, desc->DQ, desc->DQ);
        D3_RIG_DUMP(_n("attn.k"), k.data(), L0, desc->DK, desc->DK);
        D3_RIG_DUMP(_n("attn.v"), v.data(), L0, desc->DV, desc->DV);
        D3_RIG_DUMP(_n("attn.g"), g.data(), L0, desc->DQ, desc->DQ);
        PREFILL_PROF_TICK("attn:qkvg_proj");

        // state-once: the state segment saves its keys and values (post k_norm + RoPE, raw V) for the
        // question blocks that follow it, in this pass and in later ones
        d3_layer_cache* lc = b.cache ? &b.cache->layers[layer_idx] : nullptr;
        bool any_dependent = false;
        for (const auto& s : b.segs) {
            if (s.save) {
                lc->k.assign(k.data() + (size_t)s.begin * desc->DK, k.data() + (size_t)(s.begin + s.len) * desc->DK);
                lc->v.assign(v.data() + (size_t)s.begin * desc->DV, v.data() + (size_t)(s.begin + s.len) * desc->DV);
            }
            any_dependent |= s.dependent;
        }
        const int cached = lc ? (int)(lc->k.size() / desc->DK) : 0;
        if (any_dependent) {
            if (cached != b.cache->state_len) throw std::logic_error("d3_npu: state-once cache of layer " + std::to_string(layer_idx) + " is missing");
            st_kt.resize(d3_cpu::keys_t_floats(cached, desc->num_kv_heads, desc->DH));
            d3_cpu::transpose_keys(st_kt.data(), lc->k.data(), cached, desc->DK, desc->num_kv_heads, desc->DH);
        }

        // attention per segment over its own rows; padding rows are never read. Long segments
        // on the array first, in row order: each writes whole 128-row chunks, and any rows it
        // writes past its end belong to a later segment, whose attention then rewrites them.
        std::vector<char> on_array(b.segs.size(), 0);
#if D3_MM_ON_NPU
        const int64_t array_min = d3_attn_array_min();
        for (size_t i = 0; i < b.segs.size(); i++) {
            const int st = causal ? 0 : b.segs[i].state;
            if (b.segs[i].dependent) continue;   // question blocks of a cached state run on the host
            if (causal || d3_attn_work(b.segs[i].len, st) < array_min * array_min || !_array_ready()) continue;
            _array_attention(b.segs[i], b);
            on_array[i] = 1;
        }
#endif
        const float scale = 1.0f / sqrtf((float)desc->DH);
        for (size_t i = 0; i < b.segs.size(); i++) {
            const auto& s = b.segs[i];
            if (on_array[i]) continue;
            bf16* o = sc->after_mix.data() + (size_t)s.begin * desc->DQ;
            const bf16* qp = q.data() + (size_t)s.begin * desc->DQ;
            const bf16* kp = k.data() + (size_t)s.begin * desc->DK;
            const bf16* vp = v.data() + (size_t)s.begin * desc->DV;
            if (s.dependent) {
                // state-once: the block over [cached state K,V | its own], bit-identical to the per-row path
                d3_cpu::attention_prefix(o, qp, kp, vp, s.len, st_kt.data(), lc->v.data(), cached, desc->num_heads,
                                         desc->num_kv_heads, desc->DH, desc->DK, desc->DV, desc->DK, scale, kt_buf);
            } else if (!causal && s.state > 0 && s.state < s.len) {
                // prefix_bidir: the state rows over the state, then the question rows over every row
                const int S = s.state;
                d3_cpu::attention(o, qp, kp, vp, S, S, desc->num_heads, desc->num_kv_heads, desc->DH,
                                  0, 0, desc->DK, desc->DV, scale, false, kt_buf);
                d3_cpu::attention(o + (size_t)S * desc->DQ, qp + (size_t)S * desc->DQ, kp, vp,
                                  s.len - S, s.len, desc->num_heads, desc->num_kv_heads, desc->DH,
                                  0, 0, desc->DK, desc->DV, scale, false, kt_buf);
            } else {
                d3_cpu::attention(o, qp, kp, vp, s.len, s.len, desc->num_heads, desc->num_kv_heads, desc->DH,
                                  0, 0, desc->DK, desc->DV, scale, causal, kt_buf);
            }
        }
        D3_RIG_DUMP(_n("attn.core"), sc->after_mix.data(), L0, desc->DQ, desc->DQ);
        PREFILL_PROF_TICK("attn:mha");

        cpu_func::_elementwise_mul(g.data(), g_sigmoid.data(), sc->after_mix.data(), b.R * desc->DQ);
        D3_RIG_DUMP(_n("attn.gated"), g.data(), L0, desc->DQ, desc->DQ);
#if D3_MM_ON_NPU
        if (!d3_host_gemm("attn_o")) {
            d3_round_gemm_input(g.data(), desc->DQ, b.R, desc->DQ);
            g.sync_to_device();
            o_proj(sc->x, g, proj_weights);
            sc->x.sync_from_device();
        } else
#endif
        {
            d3_cpu::dequant_mm(sc->x.data(), desc->D, proj_weights.data() + desc->get_attn_o_offset(),
                               g.data(), desc->DQ, b.R, desc->DQ, desc->D, d3_cpu::BODY_LAYOUT);
            d3_cpu::scratch_push(g);   // the gating multiply above wrote it; the array writes g next pass
            d3_cpu::scratch_push(sc->x);
        }
        D3_RIG_DUMP(_n("attn.out"), sc->x.data(), L0, desc->D, desc->D);
        PREFILL_PROF_TICK("attn:o_proj");
        return buffer<bf16>(sc->x.data(), (size_t)b.R * desc->D);
    }
};

struct d3_mlp_block{
    d3_desc* desc;
    d3_npu_sequence* seq_gen;
    d3_scratch* sc = nullptr;
    int R_padded_old = -1;
    int layer_idx = -1;

    d3_mm upgate_proj, down_proj;
    d3_mm upgate_proj_hi;   //!< the second column part of passes whose up/gate result passes d3_bo_limit()

    d3_mlp_block(d3_desc* desc, d3_npu_sequence* seq_gen, npu_app_manager* gemm)
        : desc(desc), seq_gen(seq_gen) {
        upgate_proj.build(gemm); down_proj.build(gemm); upgate_proj_hi.build(gemm);
    }

    std::string _n(const char* s) const { return "l" + std::to_string(layer_idx) + "." + s; }

    void reserve(int R_padded) {
        if (R_padded_old == R_padded) return;
#if D3_MM_ON_NPU
        const auto parts = d3_upgate_pairs(R_padded, desc->intermediate_size);
        if (upgate_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(upgate_proj.seq(), R_padded, desc->D, parts.first * 1024, desc->get_upgate_offset(), 0, 0);
        if (parts.second && upgate_proj_hi.select(R_padded))
            seq_gen->gen_dequant_mm_512(upgate_proj_hi.seq(), R_padded, desc->D, parts.second * 1024,
                                        desc->get_upgate_offset() + get_quantization_byte_size((size_t)parts.first * 1024 * desc->D, flm_q4k), 0, 0);
        if (down_proj.select(R_padded)) seq_gen->gen_dequant_mm_512(down_proj.seq(), R_padded, desc->intermediate_size, desc->D, desc->get_down_offset(), 0, 0);
#endif
        R_padded_old = R_padded;
    }

    void drop_above(int rows) {
        for (d3_mm* m : {&upgate_proj, &down_proj, &upgate_proj_hi}) m->drop_above(rows);
        if (R_padded_old > rows) R_padded_old = -1;
    }

    buffer<bf16> forward(const d3_batch& b, buffer<u8>& proj_weights) {
        reserve(b.R_padded);
        buffer<bf16> &upgate = sc->upgate, &hid = sc->hid;
        const int IS = desc->intermediate_size;
#if D3_MM_ON_NPU
        const auto parts = d3_upgate_pairs(b.R_padded, IS);
        if (!parts.second) {
            upgate_proj(upgate, sc->x, proj_weights);
            upgate.sync_from_device();
            cpu_func::_exp_glu(hid.data(), upgate.data(), b.R, IS, 512);
        } else {
            // the result would not fit one buffer: two column parts, the second GEMM overlapping the first's GLU
            upgate_proj(upgate, sc->x, proj_weights);
            upgate.sync_from_device();
            auto hi_run = upgate_proj_hi.create_run(sc->upgate_hi, sc->x, proj_weights);
            hi_run.start();
            cpu_func::_exp_glu_part(hid.data(), IS, upgate.data(), b.R, parts.first * 512, 512);
            flm::npu::wait_for_completion(hi_run, "d3 upgate hi");
            sc->upgate_hi.sync_from_device();
            cpu_func::_exp_glu_part(hid.data() + (size_t)parts.first * 512, IS, sc->upgate_hi.data(), b.R, parts.second * 512, 512);
        }
        if (!d3_host_gemm("down")) {
            d3_round_gemm_input(hid.data(), IS, b.R, IS);
            hid.sync_to_device();
            D3_RIG_DUMP(_n("mlp.act"), hid.data(), b.segs[0].len, IS, IS);
            down_proj(sc->x, hid, proj_weights);
            sc->x.sync_from_device();
        } else {
            D3_RIG_DUMP(_n("mlp.act"), hid.data(), b.segs[0].len, IS, IS);
            d3_cpu::dequant_mm(sc->x.data(), desc->D, proj_weights.data() + desc->get_down_offset(),
                               hid.data(), IS, b.R, IS, desc->D, d3_cpu::BODY_LAYOUT);
            d3_cpu::scratch_push(hid);   // _exp_glu above wrote it on the host
            d3_cpu::scratch_push(sc->x);
        }
#else
        d3_cpu::dequant_mm_upgate(upgate.data(), proj_weights.data() + desc->get_upgate_offset(),
                                  sc->x.data(), b.R, desc->D, IS, 512);
        d3_cpu::glu_mul(hid.data(), upgate.data(), b.R, IS);
        D3_RIG_DUMP(_n("mlp.act"), hid.data(), b.segs[0].len, IS, IS);
        d3_cpu::dequant_mm(sc->x.data(), desc->D, proj_weights.data() + desc->get_down_offset(),
                           hid.data(), IS, b.R, IS, desc->D, d3_cpu::BODY_LAYOUT);
#endif
        D3_RIG_DUMP(_n("mlp.out"), sc->x.data(), b.segs[0].len, desc->D, desc->D);
        return buffer<bf16>(sc->x.data(), (size_t)b.R * desc->D);
    }
};

/// @brief owns the per-kind blocks and the scratch shared between them
struct d3_prefill_context{
    d3_desc* desc;
    d3_npu_sequence* seq_gen;
    npu_xclbin_manager* npu;
    // Scratch per row bucket, least recently used out first, within a byte budget of one pass of
    // `cache_rows`: a server whose requests alternate between buckets (one question, then sixteen)
    // stops reallocating and re-flushing every BO on each switch. A longer prompt's bucket is
    // allocated on its own (everything cached is dropped first) and dropped after its call, with
    // its GEMM apps (trim()), so peak memory is what that single pass needs and idle memory stays
    // what the cached buckets need: 0.29 GB of scratch at 4,096 rows, 2.3 GB at 32,768.
    std::map<int, std::unique_ptr<d3_scratch>> scratch;   //!< per R_padded; held by pointer, never copied
    d3_scratch* sc = nullptr;                             //!< the current bucket's
    int cache_rows = 0;                                   //!< set_scratch_rows(), rounded to the row tile
    size_t scratch_budget = 0;                            //!< bytes of one cache_rows pass
    uint64_t tick = 0;

    npu_app_manager* gemm_app_manager = nullptr;
    std::unique_ptr<d3_attn_block> attn_block;
    std::unique_ptr<d3_linear_block> linear_block;
    std::unique_ptr<d3_mlp_block> mlp;

    d3_prefill_context(npu_xclbin_manager* npu, d3_desc* model_desc, d3_npu_sequence* seq_gen, LM_Config& config)
        : desc(model_desc), seq_gen(seq_gen), npu(npu) {
#if D3_MM_ON_NPU
        gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
#else
        (void)config;
#endif
        linear_block = std::make_unique<d3_linear_block>(desc, seq_gen, gemm_app_manager);
        attn_block = std::make_unique<d3_attn_block>(desc, seq_gen, gemm_app_manager, npu,
            utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
        mlp = std::make_unique<d3_mlp_block>(desc, seq_gen, gemm_app_manager);
    }

    /// @brief after_mix holds either mixer's output
    size_t mix_cols() const { return std::max<size_t>(desc->Linear_DV, desc->DQ); }

    /// @brief bytes of one bucket's scratch
    size_t scratch_bytes(int R_padded) const {
        const size_t per_row = sizeof(bf16) * (desc->D + desc->Linear_DV + mix_cols() + 2 * desc->Linear_DQKV
                                               + 3 * desc->DQ + desc->DK + desc->DV + 3 * desc->intermediate_size)
                             + sizeof(float) * 2 * desc->linear_num_v_heads;
        return per_row * (size_t)R_padded;
    }

    /// @brief cache buckets up to `rows` rows, within the scratch of one `rows`-row pass
    void set_scratch_rows(int rows) {
        const int tile = d3_npu_sequence::MM_ROWS;
        cache_rows = (rows + tile - 1) / tile * tile;
        scratch_budget = scratch_bytes(cache_rows);
        trim();
    }

    /// @brief drop what is not cached between calls: buckets over cache_rows (scratch and GEMM
    ///        apps), then least recently used scratch past the budget
    void trim() {
        for (auto it = scratch.upper_bound(cache_rows); it != scratch.end();) {
            if (sc == it->second.get()) sc = nullptr;
            it = scratch.erase(it);
        }
        size_t used = 0;
        for (const auto& e : scratch) used += e.second->bytes;
        while (used > scratch_budget) {
            auto lru = scratch.begin();
            for (auto e = scratch.begin(); e != scratch.end(); ++e)
                if (e->second->last_use < lru->second->last_use) lru = e;
            if (sc == lru->second.get()) sc = nullptr;
            used -= lru->second->bytes;
            scratch.erase(lru);
        }
        if (!sc) linear_block->sc = attn_block->sc = mlp->sc = nullptr;
        linear_block->drop_above(cache_rows);
        attn_block->drop_above(cache_rows);
        mlp->drop_above(cache_rows);
    }

    std::unique_ptr<d3_scratch> _alloc(int R_padded) {
        auto s = std::make_unique<d3_scratch>();
        const size_t R = (size_t)R_padded;
        s->R_padded = R_padded;
        s->bytes = scratch_bytes(R_padded);
        s->x = d3_cpu::scratch_alloc<bf16>(npu, R * desc->D);
        s->gate = d3_cpu::scratch_alloc<bf16>(npu, R * desc->Linear_DV);
        s->after_mix = d3_cpu::scratch_alloc<bf16>(npu, R * mix_cols());
        s->qkv = d3_cpu::scratch_alloc<bf16>(npu, R * desc->Linear_DQKV);
        s->conv_qkv = buffer<bf16>(R * desc->Linear_DQKV);
        s->alpha_beta = buffer<float>(R * desc->linear_num_v_heads * 2);
        s->q = d3_cpu::scratch_alloc<bf16>(npu, R * desc->DQ);
        s->k = d3_cpu::scratch_alloc<bf16>(npu, R * desc->DK);
        s->v = d3_cpu::scratch_alloc<bf16>(npu, R * desc->DV);
        s->g = d3_cpu::scratch_alloc<bf16>(npu, R * desc->DQ);
        s->g_sigmoid = buffer<bf16>(R * desc->DQ);
#if D3_MM_ON_NPU
        const auto parts = d3_upgate_pairs(R_padded, desc->intermediate_size);
        s->upgate = d3_cpu::scratch_alloc<bf16>(npu, R * parts.first * 1024);
        if (parts.second) s->upgate_hi = d3_cpu::scratch_alloc<bf16>(npu, R * parts.second * 1024);
#else
        s->upgate = d3_cpu::scratch_alloc<bf16>(npu, R * desc->intermediate_size * 2);
#endif
        s->hid = d3_cpu::scratch_alloc<bf16>(npu, R * desc->intermediate_size);
        // padding rows of every GEMM input stay zero (finite) for the life of the buffers
        for (buffer<bf16>* b : {&s->x, &s->after_mix, &s->gate, &s->hid}) {
            memset(b->data(), 0, b->size() * sizeof(bf16));
            // Flush the memset before the array writes these BOs. Otherwise its dirty cache lines
            // are written back over the array's output by the next sync_from_device(): `gate`
            // (written only by the g projection) came back with random 64-byte runs of zeros.
            d3_cpu::scratch_push(*b);
        }
        return s;
    }

    /// @brief make R_padded's scratch current; allocate it, evicting least recently used buckets
    ///        past the budget, when it is not cached
    void reserve(int R_padded) {
        if (!(sc && sc->R_padded == R_padded)) {
            auto it = scratch.find(R_padded);
            if (it == scratch.end()) {
                const size_t need = scratch_bytes(R_padded);
                size_t used = 0;
                for (const auto& e : scratch) used += e.second->bytes;
                while (!scratch.empty() && used + need > scratch_budget) {
                    auto lru = scratch.begin();
                    for (auto e = scratch.begin(); e != scratch.end(); ++e)
                        if (e->second->last_use < lru->second->last_use) lru = e;
                    used -= lru->second->bytes;
                    scratch.erase(lru);
                }
                it = scratch.emplace(R_padded, _alloc(R_padded)).first;
            }
            sc = it->second.get();
            linear_block->sc = attn_block->sc = mlp->sc = sc;
        }
        sc->last_use = ++tick;
    }

    /// @brief one decoder layer over the packed batch; residual [R, D] is updated in place
    void forward(int layer_idx, d3_batch& b, buffer<bf16>& residual, buffer<u8>& proj_weights,
                 buffer<bf16>& rms_weights, buffer<bf16>& rope_weights,
                 buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
        PREFILL_PROF_INIT();
        reserve(b.R_padded);
        linear_block->layer_idx = attn_block->layer_idx = mlp->layer_idx = layer_idx;
        const std::string ln = "l" + std::to_string(layer_idx) + ".";

        cpu_func::_rms_norm(sc->x, residual, rms_weights.data(), desc->D, b.R, 0, 0);
        d3_round_gemm_input(sc->x.data(), desc->D, b.R, desc->D);
        d3_cpu::scratch_push(sc->x);
        D3_RIG_DUMP(ln + "in_norm", sc->x.data(), b.segs[0].len, desc->D, desc->D);
        PREFILL_PROF_TICK("layer:input_rms_norm");

        buffer<bf16> block_output;
        if (desc->is_attn_layer(layer_idx)) {
            block_output = attn_block->forward(b, proj_weights, rope_weights);
            PREFILL_PROF_TICK("layer:attn_block_total");
        } else {
            block_output = linear_block->forward(b, proj_weights, rope_weights, alpha_proj_cpu, beta_proj_cpu);
            PREFILL_PROF_TICK("layer:linear_block_total");
        }
        cpu_func::_residual_add(residual, block_output, residual, (size_t)b.R * desc->D);

        cpu_func::_rms_norm(sc->x, residual, rms_weights.data() + desc->D, desc->D, b.R, 0, 0);
        d3_round_gemm_input(sc->x.data(), desc->D, b.R, desc->D);
        d3_cpu::scratch_push(sc->x);
        D3_RIG_DUMP(ln + "post_norm", sc->x.data(), b.segs[0].len, desc->D, desc->D);
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        auto mlp_out = mlp->forward(b, proj_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");
        cpu_func::_residual_add(residual, mlp_out, residual, (size_t)b.R * desc->D);
        D3_RIG_DUMP(ln + "out", residual.data(), b.segs[0].len, desc->D, desc->D);
    }
};

#endif
