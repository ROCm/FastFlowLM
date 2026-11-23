// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_prefill.hpp
/// \brief the prompt matmuls on the NPU (mm_rtm2_glu.xclbin): attention projections and routed experts
/// \note Owns the Q4_K weights of every layer in one device buffer each, in the kernel's grouped layout
///       (inkling2_npu_mm.hpp), and points the host model's matrices at them: decode reads the same copy.
///       Per layer: [k | v | q | gate | rel] (width 2304), o (3072), gate_up (2304), down (1280), tail.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "inkling2_host.hpp"
#include "npu_utils/npu_utils.hpp"
#include "inkling2_launch.hpp"
#include "typedef.hpp"

class inkling2_prefill final : public inkling2_host::mm_offload {
public:
    /// \param xclbin path of mm_rtm2_glu.xclbin
    inkling2_prefill(npu_xclbin_manager* npu, const std::string& xclbin);
    ~inkling2_prefill();

    /// \brief copy the projections of every layer into device buffers and point w's matrices at them
    void load(inkling2_host::weights& w);

    /// \brief prompt steps shorter than this stay on the host
    int min_T = 64;

    /// \brief allocate the device buffers of a T-token prompt step now (load) rather than in the first prompt:
    ///        ~80 ms at 2k tokens, ~1 GB at 4k
    void prewarm(int T);

    bool attn_in(int il, const float* xq, const float* xk, const float* xv, int T, float* Q, float* K, float* V, float* R,
                 float* G, inkling2_host::bf16_rows* view) override;
    bool attn_out(int il, const float* O, int T, float* y, inkling2_host::bf16_rows* view) override;
    bool experts(int il, const float* x, int T, const int* token_of, const int* count, const int* start,
                 const int* slot_of, const float* wsel, float* out) override;
    bool attention(int il, int p0, int T, const float* Q, const float* R, const float* sinks, const float* rel_bias,
                   const u16* Kc, const u16* Vc, int ring, int window, const float* K, const float* V, const float* G,
                   const inkling2_host::bf16_rows* Gb, float* O, const std::function<void()>& packed) override;

    /// \brief the attention xclbin (prefill_attn.xclbin); without it attention stays on the host. With the
    ///        control-packet switches (switches: inkling2.switches, see inkling2_launch.hpp) attention runs in the
    ///        matmuls' hardware context instead of its own.
    void set_attention(const std::string& xclbin, const std::string& switches);

    /// \brief attention shares the matmuls' hardware context: a prompt holds inkling2_array_lock
    bool switched() const { return this->ll.ready; }

    /// \brief attn_in, attention and attn_out of the calling thread use host buffers set `lane` (0 or 1): two threads
    ///        on different lanes pack and unpack at the same time and take turns on the NPU
    void set_lane(int lane) override;

private:
    struct layer_buf {
        buffer<uint8_t> w;
        size_t o = 0, gate_up = 0, down = 0;   // byte offsets of o, gate_up, down (attention at 0)
    };

    // INKLING2_PROF: seconds per phase (attn_in, attn_out, expert pack, gate_up submit, situ + down submit, down wait, unpack)
    bool prof = false;
    bool unchain = false;   // INKLING2_EXP_CHECK: rerun the experts on the host-activated path
    double t[19] = {};
    double npu_t[5] = {};   // DIAG: NPU start..wait per kind (in 1 input, in 3 inputs, out, attention swa, global)
    long npu_n[5] = {};
    // DIAG: host submit (create_run + start; attention: gen_seq) per kind, + expert gate_up / down submits and runs
    double npu_s[7] = {};
    long npu_sn[7] = {};
    long calls = 0, runs = 0;
    void report() const;
    npu_xclbin_manager* npu;
    npu_app_manager* mgr;
    std::vector<layer_buf> layers;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, npu_app> apps;
    std::vector<npu_app> app_gu, app_dn;   // the experts' dispatches per group (INKLING2_EXP_GROUPS), rebuilt per call
    // activations, grown on demand, per lane: attention x (3 inputs) and y; experts x, gate_up out, act, down out
    struct lane_buf {
        buffer<bf16> ax[3], ay;
        buffer<bf16> att_o, att_q, att_kv, att_tab;   // attention output, query packs, K/V frame, bias table
    } lanes[2];
    lane_buf& lane();
    std::mutex npu_mu;     // the NPU runs, the apps and their sequences
    std::mutex alloc_mu;   // device buffer allocation
    buffer<bf16> ex, egu, ea, ey;
    // attention: its xclbin (or the mm context with the switches), one app per frame: a new sequence costs its
    // ELF and kernel (~0.5 ms each call); the layers and passes of a prompt repeat a few frames
    npu_app_manager* att_mgr = nullptr;   // the context attention runs in (mgr with the switches)
    std::map<std::tuple<int, int, int, int>, npu_app> att_apps;
    inkling2_launch ll;

    /// \brief the dispatch of rows x Din (weights of width Kw) -> Npad, by its geometry
    npu_app& app(uint32_t rows, uint32_t Din, uint32_t Kw, uint32_t Npad);
    /// \brief the attention dispatch of query rows [Lb, Le), keys 0 .. Le - 1, window (0 global), K/V buffer of kv u16
    npu_app& att_app(int Lb, int Le, int window, int kv);
    /// \brief b holds at least n elements
    void reserve(buffer<bf16>& b, size_t n);
    /// \brief rows of the expert activation buffers for any routing of T tokens
    static size_t expert_rows(int T);
};
