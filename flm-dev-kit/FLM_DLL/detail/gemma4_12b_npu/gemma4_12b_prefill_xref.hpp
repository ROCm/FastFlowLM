// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __GEMMA4_12B_PREFILL_XREF_HPP__
#define __GEMMA4_12B_PREFILL_XREF_HPP__

/// \file gemma4_12b_prefill_xref.hpp
/// \brief Cross reference scoring for the batched prefill: every dequantizing matmul the
///        NPU runs is replayed on the host and the two are compared.
///
/// gemma4_12b_prefill.hpp carries both implementations of the matmul -- the array via
/// dequant_mm.xclbin, and gemma4_12b_cpu::dequant_mm() -- and which one a build uses is
/// GEMMA4_12B_MM_ON_NPU. That makes the host kernel a reference that is always available,
/// on the real prompt, at every layer, with no file next to the model. This replaces the
/// torch activation dump that used to fill that role, which could only score layer 0 of one
/// fixed 1989 token prompt and had to replace the prompt and exit(0) to do it.
///
/// \note What this can and cannot catch. It proves the two backends agree; it cannot prove
///       either is right. Only the stages that have two implementations are covered:
///       q/k/v/o_proj, the fused up/gate pair, down_proj, and -- once an attention xclbin
///       exists -- attention_out. The rms norms, rope, and the glu multiply are the same
///       cpu_func call in both builds, so a bug in one of those is invisible here.
///
/// \note Everything is ENABLE_DEBUG only. With DEBUG_LEVEL=0 this header expands to
///       nothing, gemma4_12b_prefill.hpp holds no xref member and issues no host
///       recompute, and the prefill path is byte for byte the release one.
///
/// \note Scoring is off by default at DEBUG_LEVEL=1, which is what a plain `make` builds.
///       The host replay is seconds per prefill, not milliseconds, so it is opt in: it
///       turns on at DEBUG_LEVEL > 1, or at any debug level via FLM_XREF_LAYERS.

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
/// The host replay is orders of magnitude slower than the array, so scoring every matmul of
/// every layer would dominate a debug prefill. Both bounds are therefore on by default and
/// widened from the environment, the same way gemma4_12b_cpu::prefill_threads() reads
/// FLM_PREFILL_THREADS.
struct gemma4_12b_prefill_xref {
    //!< rows scored per tensor, counted from the start of the prompt. Matmul rows are
    //!< independent, so a prefix is a partial check rather than a weakened one.
    static constexpr int DEFAULT_ROWS = 256;

    //!< cosine above which a tensor is called a match and printed green
    static constexpr float PASS_COS = 0.999f;

    //!< DEBUG_LEVEL at which the default layer set turns on. Below it, only an explicit
    //!< FLM_XREF_LAYERS scores anything.
    static constexpr int DEFAULT_ON_AT_LEVEL = 1;

    bool             all_layers = false; //!< FLM_XREF_LAYERS=all
    std::vector<int> layers;             //!< empty and !all_layers == scoring off
    int              row_cap = DEFAULT_ROWS;

    /// \brief Append a layer to the default set, ignoring out of range and duplicates.
    /// \note  The three defaults collide on a small model -- a 6 layer one has its first
    ///        global layer at 5, which is also its last layer -- and a layer listed twice
    ///        would be scored twice.
    void _add_default(int layer_idx) {
        if (layer_idx < 0) { return; }
        for (int l : layers) { if (l == layer_idx) { return; } }
        this->layers.push_back(layer_idx);
    }

    /// \param global_layer_idx first global (attention_k_eq_v) layer, i.e.
    ///        sliding_window_pattern - 1. Layer 0 is the first SWA layer, so the two
    ///        together cover both layer kinds, which is all the kinds there are.
    /// \param last_layer_idx  num_hidden_layers - 1. The first two layers see an input the
    ///        two backends agree on by construction, so they score the matmuls in isolation;
    ///        the last one is fed by 47 layers of accumulated divergence and is where a
    ///        difference small enough to pass at layer 0 shows up.
    gemma4_12b_prefill_xref(int global_layer_idx, int last_layer_idx) {
        const char* rows_env = std::getenv("FLM_XREF_ROWS");
        if (rows_env != nullptr) {
            const int v = std::atoi(rows_env);
            this->row_cap = (v > 0) ? v : 0; //!< 0 == every row
        }

        const char* layers_env = std::getenv("FLM_XREF_LAYERS");
        if (layers_env == nullptr) {
            if (DEBUG_LEVEL >= DEFAULT_ON_AT_LEVEL) {
                this->_add_default(0);
                this->_add_default(global_layer_idx);
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
        header_print("xref", "scoring the npu matmuls against the host kernels on "
                             << what.str() << " (FLM_XREF_LAYERS / FLM_XREF_ROWS)");
    }

    /// \brief Whether this layer is scored at all.
    /// \note Checked before the host replay is issued, so a layer that is off costs nothing.
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
    u64 fingerprint(const bf16* p, int D, int stride, int rows) const {
        u64 h = 1469598103934665603ull;
        for (int r = 0; r < rows; r++) {
            const u16* row = reinterpret_cast<const u16*>(p + (size_t)r * stride);
            for (int i = 0; i < D; i++) { h = (h ^ row[i]) * 1099511628211ull; }
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
    struct guard {
        const gemma4_12b_prefill_xref* xref;
        const char*                    what;
        int                            layer_idx;
        const bf16*                    p;
        int                            D, stride, rows;
        u64                            before;

        guard(const gemma4_12b_prefill_xref* xref, const char* what, int layer_idx,
              const bf16* p, int D, int stride, int rows)
            : xref(xref), what(what), layer_idx(layer_idx), p(p), D(D), stride(stride),
              rows(rows), before(xref->fingerprint(p, D, stride, rows)) {}

        ~guard() {
            if (xref->fingerprint(p, D, stride, rows) == before) { return; }
            std::ostringstream line;
            line << "L" << layer_idx << " " << what
                 << ": the cross reference MODIFIED the npu result, so what reached the "
                    "rest of the model is not what the array produced";
            header_print_r("xref", line.str());
            assert(false && "xref must not write to the tensor it scores");
        }

        guard(const guard&)            = delete;
        guard& operator=(const guard&) = delete;
    };

    /// \brief Score `rows` rows of `got` against `want` and report one line.
    /// \param name       what the tensor is called in the report
    /// \param got        the result the compiled in backend produced
    /// \param want       the same tensor from the other implementation, packed [rows][D]
    /// \param D          columns compared
    /// \param got_stride distance between two rows of `got`; larger than D when the tensor
    ///                   is one half of an interleaved pair, as up/gate are
    ///
    /// \note The two sides are NOT symmetric. get_cosine_similarity() and friends index
    ///       their first argument with the padded stride and their second one packed, so
    ///       `want` has to be compacted to [rows][D] by the caller. Passing a strided
    ///       reference reads every row after the first from the wrong offset, which scores
    ///       as a stable low cosine rather than as an obvious failure.
    void check(const char* name, int layer_idx, const bf16* got, const bf16* want,
               int D, int got_stride, int rows) const {
        if (rows <= 0) { return; }

        const float cos = get_cosine_similarity(got, want, 1, rows, D, rows, got_stride);
        const float l2  = get_relativeL2      (got, want, 1, rows, D, rows, got_stride);
        const float mx  = get_max_abs_error   (got, want, 1, rows, D, rows, got_stride);

        std::ostringstream line;
        line << "L" << std::left << std::setw(3) << layer_idx
             << std::setw(18) << name << std::right
             << " rows " << std::setw(5) << rows
             << "  cos " << std::fixed << std::setprecision(6) << cos
             << "  rel_l2 " << std::setprecision(6) << l2
             << "  max_abs " << std::scientific << std::setprecision(2) << mx;
        if (cos > PASS_COS) { header_print_g("xref", line.str()); }
        else                { header_print_r("xref", line.str()); }
    }
};

#endif // ENABLE_DEBUG

#endif
