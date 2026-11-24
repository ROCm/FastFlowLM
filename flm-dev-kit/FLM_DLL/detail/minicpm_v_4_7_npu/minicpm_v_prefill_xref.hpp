// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __MINICPM_V_PREFILL_XREF_HPP__
#define __MINICPM_V_PREFILL_XREF_HPP__

/// \file minicpm_v_prefill_xref.hpp
/// \brief Cross reference scoring for the MiniCPM-V-4.7-1B prefill: every stage the array
///        runs is replayed on the host, over the same inputs, and the two are compared.
///
/// `minicpm_v_prefill.hpp` already carries both implementations of every array stage --
/// `_<stage>_npu()` on the overlay and `_<stage>_host()` in `minicpm_v_cpu` -- and which
/// one a build uses is that stage's `MINICPM_V_*_ON_NPU` macro. That makes the host kernel
/// a reference that is always available, on the real prompt, at every layer, with no file
/// next to the model. The replay here feeds the host kernel the *array's own inputs*, so a
/// disagreement names the overlay rather than accumulating through the whole tower the way
/// `host_ref/compare_engine.py` does.
///
/// Stages covered, one per overlay:
///
/// | overlay                      | scored tensors                                  |
/// |------------------------------|-------------------------------------------------|
/// | `mm.xclbin`                  | `linear_qkv`, `linear_z`, `linear_o`,           |
/// |                              | `attn_q`, `attn_k`, `attn_v`, `attn_gate`,      |
/// |                              | `attn_o`, `mlp_upgate`, `mlp_glu`, `mlp_down`   |
/// | `conv.xclbin`                | `conv_qkv`                                      |
/// | `GateDeltaNet_prefill.xclbin`| `gdn_out`, `gdn_state`                          |
/// | `attn.xclbin`                | `attention_out`                                 |
///
/// \note What this can and cannot catch. It proves the two backends agree; it cannot prove
///       either is right. Neither can `host_ref/run_gate.sh`, which scores the NPU build
///       against the all-host build; no independent reference ships with the dev kit. Only
///       stages with two implementations are covered. The rms norms, the q/k norm + RoPE,
///       the gated norm, the alpha/beta projections, the attention gate sigmoid and
///       multiply, `fill_kv_cache` and the conv-state carry are the same `cpu_func` call
///       in both builds, so a bug in one of those is invisible here.
///
/// \note Everything is ENABLE_DEBUG only. With DEBUG_LEVEL=0 this header expands to
///       nothing, `minicpm_v_prefill.hpp` holds no xref member and issues no host
///       recompute, and the prefill path is byte for byte the release one.
///
/// \note Scoring is off by default even at DEBUG_LEVEL=1 unless FLM_XREF_LAYERS is set --
///       except for the three default layers, which turn on at DEBUG_LEVEL >= 1. The host
///       replay is seconds per prefill, not milliseconds.

#include "utils/debug_utils.hpp"

#ifdef ENABLE_DEBUG

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "typedef.hpp"
#include "utils/error_measure.hpp"

/// \brief Which layers get scored, how many rows of them, and the report line.
///
/// The host replay is orders of magnitude slower than the array, so scoring every stage of
/// every layer would dominate a debug prefill. Both bounds are therefore on by default and
/// widened from the environment, the same way `minicpm_v_cpu::prefill_threads()` reads
/// FLM_PREFILL_THREADS.
struct minicpm_v_prefill_xref {
    //!< rows scored per tensor, counted from the start of the prompt. Matmul rows are
    //!< independent, so a prefix is a partial check rather than a weakened one. The
    //!< recurrent stages (conv, gated delta rule, attention) are *not* row independent,
    //!< so they are replayed over the whole prompt and only reported on `rows` of it.
    static constexpr int DEFAULT_ROWS = 256;

    //!< cosine above which a tensor is called a match and printed green
    static constexpr float PASS_COS = 0.999f;

    //!< DEBUG_LEVEL at which the default layer set turns on. Below it, only an explicit
    //!< FLM_XREF_LAYERS scores anything.
    static constexpr int DEFAULT_ON_AT_LEVEL = 1;

    bool             all_layers = false; //!< FLM_XREF_LAYERS=all
    std::vector<int> layers;             //!< empty and !all_layers == scoring off
    int              row_cap = DEFAULT_ROWS;

    //!< worst cosine seen and where, so a long run ends with one line that says pass/fail
    float       worst_cos   = 2.0f;
    std::string worst_what;
    int         checked     = 0;
    int         failed      = 0;

    /// \brief Append a layer to the default set, ignoring out of range and duplicates.
    /// \note  The three defaults can collide -- `full_attention_interval` is 4, so on a
    ///        four layer model the first attention layer is also the last layer -- and a
    ///        layer listed twice would be replayed twice.
    void _add_default(int layer_idx) {
        if (layer_idx < 0) { return; }
        for (int l : layers) { if (l == layer_idx) { return; } }
        this->layers.push_back(layer_idx);
    }

    /// \param attn_layer_idx first full-attention layer, i.e. `minicpm_v_desc::attn_idx`
    ///        (3 here). Layer 0 is a linear (gated delta net) layer, so the two together
    ///        cover both layer kinds, which is all the kinds there are.
    /// \param last_layer_idx `num_hidden_layers - 1`. The first two layers see an input the
    ///        two backends agree on by construction, so they score their stages in
    ///        isolation; the last one is fed by 23 layers of accumulated divergence and is
    ///        where a difference small enough to pass at layer 0 shows up.
    minicpm_v_prefill_xref(int attn_layer_idx, int last_layer_idx) {
        const char* rows_env = std::getenv("FLM_XREF_ROWS");
        if (rows_env != nullptr) {
            const int v    = std::atoi(rows_env);
            this->row_cap  = (v > 0) ? v : 0; //!< 0 == every row
        }

        const char* layers_env = std::getenv("FLM_XREF_LAYERS");
        if (layers_env == nullptr) {
            if (DEBUG_LEVEL >= DEFAULT_ON_AT_LEVEL) {
                this->_add_default(0);
                this->_add_default(attn_layer_idx);
                this->_add_default(last_layer_idx);
            }
        } else if (strcmp(layers_env, "all") == 0) {
            this->all_layers = true;
        } else {
            //!< comma list; an empty value leaves `layers` empty, which turns scoring off
            std::stringstream ss(layers_env);
            std::string       tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) { this->layers.push_back(std::atoi(tok.c_str())); }
            }
        }

        std::ostringstream what;
        if (all_layers) {
            what << "all layers";
        } else if (layers.empty()) {
            header_print("xref", "cpu/npu cross reference off, set FLM_XREF_LAYERS "
                                 "(or build with DEBUG_LEVEL="
                                 << DEFAULT_ON_AT_LEVEL << ") to turn it on");
            return;
        } else {
            what << "layer";
            for (size_t i = 0; i < layers.size(); i++) { what << (i ? "," : " ") << layers[i]; }
        }
        what << ", " << (row_cap ? std::to_string(row_cap) : std::string("all")) << " rows";
        header_print("xref", "scoring the npu stages against the host kernels on "
                             << what.str() << " (FLM_XREF_LAYERS / FLM_XREF_ROWS)");
    }

    /// \brief Whether this layer is scored at all.
    /// \note Checked *before* any replay is issued, and before any extra `sync_from_device`
    ///       the replay would need, so a layer that is off costs nothing. That matters most
    ///       for attention and the gated delta rule, whose references are an L x DQ tensor
    ///       and a 16 x 128 x 128 fp32 state -- pulling those every layer costs more than
    ///       the stage being checked.
    bool enabled(int layer_idx) const {
        if (all_layers) { return true; }
        for (int l : layers) { if (l == layer_idx) { return true; } }
        return false;
    }

    /// \brief Rows to score out of a prompt of `L`.
    int rows(int L) const { return (row_cap == 0 || row_cap > L) ? L : row_cap; }

    /// \brief FNV-1a over a strided region of a tensor.
    /// \note Only ever compared against itself, so the hash needs no quality beyond
    ///       noticing that some element changed.
    template <typename T>
    u64 fingerprint(const T* p, int D, int stride, int rows) const {
        u64 h = 1469598103934665603ull;
        for (int r = 0; r < rows; r++) {
            const u8* row = reinterpret_cast<const u8*>(p + (size_t)r * stride);
            for (size_t i = 0; i < (size_t)D * sizeof(T); i++) {
                h = (h ^ row[i]) * 1099511628211ull;
            }
        }
        return h;
    }

    /// \brief Proves a scored tensor is byte identical before and after the xref block.
    ///
    /// The cross reference is an observer: the npu result has to be what reaches the rest
    /// of the model, and the host recompute has to stay in the scratch it was written to.
    /// Nothing in the type system says so -- every reference here is a local std::vector
    /// purely by convention -- so this checks it instead of trusting it. Scope one over the
    /// live buffer around each block of check() calls and a future edit that writes into
    /// the tensor being scored says so at once, on the layer it happened.
    ///
    /// \note The gated delta rule is the reason this is not paranoia: its host kernel
    ///       updates the recurrent state in place, so replaying it against the array's own
    ///       cache slot would silently double-apply the scan.
    template <typename T>
    struct guard_t {
        const minicpm_v_prefill_xref* xref;
        const char*                   what;
        int                           layer_idx;
        const T*                      p;
        int                           D, stride, rows;
        u64                           before;

        guard_t(const minicpm_v_prefill_xref* xref, const char* what, int layer_idx,
                const T* p, int D, int stride, int rows)
            : xref(xref), what(what), layer_idx(layer_idx), p(p), D(D), stride(stride),
              rows(rows), before(xref->fingerprint(p, D, stride, rows)) {}

        ~guard_t() {
            if (xref->fingerprint(p, D, stride, rows) == before) { return; }
            std::ostringstream line;
            line << "L" << layer_idx << " " << what
                 << ": the cross reference MODIFIED the npu result, so what reached the "
                    "rest of the model is not what the array produced";
            header_print_r("xref", line.str());
            assert(false && "xref must not write to the tensor it scores");
        }

        guard_t(const guard_t&)            = delete;
        guard_t& operator=(const guard_t&) = delete;
    };

    using guard = guard_t<bf16>;

    /// \brief Score `rows` rows of `got` against `want` and report one line.
    /// \param name       what the tensor is called in the report
    /// \param got        the result the compiled in backend produced
    /// \param want       the same tensor from the other implementation, packed [rows][D]
    /// \param D          columns compared
    /// \param got_stride distance between two rows of `got`; larger than D when the tensor
    ///                   is one half of an interleaved pair, as the mlp's up/gate slabs are
    ///
    /// \note The two sides are NOT symmetric. `get_cosine_similarity()` and friends index
    ///       their first argument with the padded stride and their second one packed, so
    ///       `want` has to be compacted to [rows][D] by the caller. Passing a strided
    ///       reference reads every row after the first from the wrong offset, which scores
    ///       as a stable, plausible looking low cosine rather than as an obvious failure.
    template <typename T>
    void check(const char* name, int layer_idx, const T* got, const T* want,
               int D, int got_stride, int rows) {
        if (rows <= 0) { return; }

        const float cos = get_cosine_similarity(got, want, 1, rows, D, rows, got_stride);
        const float l2  = get_relativeL2      (got, want, 1, rows, D, rows, got_stride);
        const float mx  = get_max_abs_error   (got, want, 1, rows, D, rows, got_stride);

        std::ostringstream line;
        line << "L" << std::left << std::setw(3) << layer_idx
             << std::setw(16) << name << std::right
             << " rows " << std::setw(5) << rows
             << "  cos " << std::fixed << std::setprecision(6) << cos
             << "  rel_l2 " << std::setprecision(6) << l2
             << "  max_abs " << std::scientific << std::setprecision(2) << mx;

        this->checked++;
        if (cos < worst_cos) {
            this->worst_cos  = cos;
            std::ostringstream w;
            w << "L" << layer_idx << " " << name;
            this->worst_what = w.str();
        }
        if (cos > PASS_COS) { header_print_g("xref", line.str()); }
        else                { this->failed++; header_print_r("xref", line.str()); }
    }

    /// \brief One line saying whether anything fell below PASS_COS over the whole run.
    /// \note Printed from the prefill driver after the last layer, because the per tensor
    ///       lines are hundreds long at FLM_XREF_LAYERS=all and the question being asked
    ///       is a single pass/fail.
    void summary() const {
        if (checked == 0) { return; }
        std::ostringstream line;
        line << checked << " tensors scored, " << failed << " below "
             << std::fixed << std::setprecision(3) << PASS_COS
             << "; worst " << std::setprecision(6) << worst_cos << " at " << worst_what;
        if (failed == 0) { header_print_g("xref", line.str()); }
        else             { header_print_r("xref", line.str()); }
    }
};

#endif // ENABLE_DEBUG

#endif
