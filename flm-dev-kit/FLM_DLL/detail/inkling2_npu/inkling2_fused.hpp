// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_fused.hpp
/// \brief decode on the NPU (decode.xclbin, the decode_layer stream overlay): a token's 48 layer applications
/// \note Application j (weight layer il, loop pass, cache slot) is up to six steps, passing data through io
///       (argument 0) only: A mix (the previous application's experts, ffn conv, residual; the attention inputs'
///       block mix and norms) -> projections -> attention (K / V conv, clips, cache, sinks, gate) -> o proj ->
///       F mix (attention conv, residual, the FFN input's block mix and norm) -> router, top 8, experts.
///       The Q4_K weights stay in inkling2_prefill's buffers, read at their device addresses; the sequence state
///       (conv rings in the mix blocks, the attention regions with the K / V records) lives on the device.
///       Loop passes 2-3 (applications 18-41): a pass starts with the previous pass's output (completion mix) and
///       its 48 messages (one mix over [o11, o17]); the rank-64 controllers and the following norms run after the A /
///       F mixes (raw fp32 outputs): down and up on the stream cores, the output and norm on glue_dn.
///       Argument 1: per layer the dense rows, router and top-k header, then the sliding-window slots' regions;
///       arguments 2, 3: the global slots' regions (4 each); argument 4: the loop controllers' weights (bf16 units:
///       per looped layer attention down 384, ffn down 128, attention up 432, ffn up 144).
#pragma once
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "inkling2_host.hpp"
#include "inkling2_lm_head.hpp"
#include "npu_utils/npu_utils.hpp"
#if !defined(FLM_USE_HRX)
#include "inkling2_launch.hpp"
#endif
#include "typedef.hpp"

// The decode and lm_head programs load control-packet switches as ELF modules (xrt::elf / xrt::module / xrt::ext::kernel), which the
// HRX shim does not provide (as d3's switches): on HRX decode and lm_head stay on the host.
#if !defined(FLM_USE_HRX)
class inkling2_fused {
public:
    /// \brief a layer's weight buffer (inkling2_prefill::device_layer): attention rows at 0, then o, gate_up, down
    struct layer_src {
        buffer<uint8_t>* w;
        size_t o, gate_up, down;
    };
    /// \brief application j: weight layer, loop pass, cache slot; global attention; starts a block; block states the
    ///        attention mix sees
    struct app_info {
        int il, pass, slot;
        bool glob, starts;
        bool three;   // three attention inputs (global layers; loop passes 2-3: one per controller)
        int S;
    };
    static app_info info(int j);
    static constexpr int NAPP = inkling2_host::NSLOT;
    /// \brief step bits of run_app
    /// ST_Z (the last application only): the token's output, the final mix and norm -> x (bf16 at XZ)
    enum { ST_PROJ = 1, ST_ATTN = 2, ST_OPROJ = 4, ST_F = 8, ST_MOE = 16, ST_A = 32, ST_ALL = 63, ST_Z = 64 };

    inkling2_fused(npu_xclbin_manager* npu, const std::string& xclbin, const inkling2_host::weights& w,
                   const std::vector<layer_src>& src, int max_len);

    /// \brief take the host model's sequence state at its position (convolution inputs, K / V cache)
    void load_state(const inkling2_host::model& m);
    /// \brief give the host model the state of the tokens run here since load_state: their K / V rows, the
    ///        convolution inputs, the position
    void store_state(inkling2_host::model& m);
    /// \brief start token `id` at the current position: h0, the per-application headers
    void begin_token(int id);
    /// \brief run steps `stages` of application j; false if the run did not complete (the device is suspect then)
    bool run_app(int j, int stages);
    /// \brief run all of the token's applications (ST_ALL | ST_Z), `chain` applications per run at most
    bool run_token();
    int chain = NAPP;       // INKLING2_FUSED_CHAIN
    bool sync_all = true;   // false: after a run only the ranges the host reads (XC, MSG, XF, XZ) come back
    // the loop controllers run on the device, inside the runs (a token is one run); INKLING2_HOST_CTRL=1 or
    // INKLING2_CTRL_CHECK: on the host between runs
    bool dev_ctrl = true;
    // one run per token (device controllers, chain 48): the next position's run is built and created while this
    // one runs; INKLING2_FUSED_PIPE=0: built after the id is known
    bool pipe = true;
    /// \brief the lm_head on the NPU in the decode's hardware context: the token's run is a window of control-packet
    ///        switches (cold_dec, the decode, dec_then_lmh, the lm_head; `switches`: inkling2_decode.switches, built
    ///        with the decode.xclbin in use, FLM_Xclbin/Inkling2/decode_switches). One run per token only (the pipe);
    ///        false if they do not load (the host's lm_head then)
    bool enable_lm_head(npu_xclbin_manager* npu, const std::string& switches);
    bool lm_on_npu() const { return this->lmh != nullptr; }
    /// \brief with the lm_head on the NPU, after run_token: the token's logits[VOCAB]
    void logits(float* out) { this->lmh->finish(out); }
    std::unique_ptr<inkling2_lm_head> lmh;
    /// \brief the token's applications ran: the next position
    void end_token() { this->pos++; }
    int position() const { return this->pos; }

    // ---- the io layout (byte offsets) and its host copy after the last run, for tests
    size_t H1 = 0, H3 = 0, PY = 0, AH = 0, OH = 0, OY = 0, ST = 0, R_HDR = 0, E_HDR = 0, RY = 0, TY = 0, EY = 0, MH = 0,
           MHX = 0, MZ = 0, FIN = 0, XC = 0, XZ = 0, XF = 0, MSG = 0, BANK = 0, BM = 0;
    size_t BP[inkling2_host::NPASS - 1] = {};
    std::vector<size_t> BA, BF;
    static constexpr size_t CB = (size_t)24 * 8 * 96 * 4, RES = (size_t)inkling2_host::D * 4;
    // a loop pass's mix output per query (out mode 3): fp32 x, then bf16 x / rms(x)
    static constexpr size_t CQ = (size_t)inkling2_host::D * 6;
    // the loop controllers on the device: down / up / out headers (attention, ffn), the down partials (16 y units),
    // per (looped layer, controller) tanh f / a (48 y units) then the norm g (fp32) and padding to whole units; CK:
    // the out step's output in check mode
    size_t H_DN = 0, H_UP = 0, H_CT = 0, GD = 0, FA = 0, CK = 0;
    static constexpr size_t FA_LEN = 28800;
    static constexpr int CW_UNITS = 1088;
    const uint8_t* io_data() const { return this->io.data(); }
    /// \brief the last token's final-normed x (bf16, after ST_Z)
    const uint16_t* x_out() const { return (const uint16_t*)(this->io.data() + this->XZ); }
    /// \brief host-side time split (ms) of the runs so far: sequence build, run, io sync, host controllers; runs
    double t_seq = 0, t_run = 0, t_sync = 0, t_ctrl = 0;
    int n_run = 0;
    double t_create = 0;   // of t_run: create_run (the sequence's module setup)
    double t_pre = 0;      // the next position's sequence build and create_run, while a run is on the device
    double t_by[128] = {};   // run time by step mask (& 127)
    int n_by[128] = {};

private:
    struct mix_cfg {
        int pre, fl, S, nq, om, nn;
    };
    static mix_cfg cfg_A(int j);
    static mix_cfg cfg_F(int j);
    static mix_cfg cfg_C();   // a pass's output: the last application's experts, ffn conv, residual
    static mix_cfg cfg_M();   // a pass's messages: 12 layers x 4 queries over [o11, o17]
    static mix_cfg cfg_P(int j);   // a later pass's first attention mix: [h0] and the prelude output o5
    static mix_cfg cfg_Z();   // the token's output: the last experts, ffn conv, residual; the final query and norm
    static int mix_nout(const mix_cfg& C);
    void add_mix(const mix_cfg& C, size_t MH_, size_t S0, size_t P0, size_t B0, size_t cur_at, size_t out_at);
    bool run_steps(int j, int stages);
    void add_steps(int j, int stages);
    bool submit(int stages);
    std::chrono::steady_clock::time_point t_build;
    std::vector<std::pair<size_t, size_t>> want;   // io ranges (offset, bytes) the host reads after the run
    void host_attn_ctrl(int j);
    void host_ffn_ctrl(int j);
    void ctrl_check(const uint8_t* q);
    /// \brief a loop application's controllers on the device after its A (attn) or F mix: down, up (upto >= 2), out
    ///        (upto 3, to ct_at if not 0)
    void add_ctrl(int j, bool attn, int upto, size_t ct_at);
    void dev_check(int j, bool attn, int level);
    // controller k's query (fp32 x then bf16 x / rms): h 0 its x (the mix output), h 1 its message
    size_t ctrl_q(int j, bool attn, int k, int h) const;

public:
    int ck_n = 0;                 // INKLING2_CTRL_CHECK: queries checked, worst |difference| (x / rms has rms 1)
    float ck_worst = 0;
    long ck_off = 0;              // elements that differ
    // INKLING2_CTRL_CHECK >= 2: the device controllers vs the host's (down: worst error / max |g|; up: worst tanh
    // error; out: worst error, elements that differ)
    int dk_n = 0;
    float dk_down = 0, dk_up = 0, dk_out = 0;
    long dk_off = 0, dk_far = 0, dk_el = 0;

private:
    void sync_range(size_t at, size_t n);
    static size_t mix_pb(const mix_cfg& C);
    static size_t mix_bb(const mix_cfg& C);
    // the projections' dispatch of 1 (sliding window) or 3 (global) inputs
    struct proj_plan {
        int nx = 1;
        int g0[16] = {}, ng[16] = {};
        std::vector<std::pair<int, int>> runs[16];   // (y slot, groups)
        struct dense {
            int c, n, x, u;
        };
        std::vector<dense> dns;
        int nd_of[16] = {};
        size_t hdr = 0;   // header bytes (whole units)
    };
    void plan_proj(proj_plan& P, size_t at);
    // where slot s's attention region is: argument, byte offset, bytes per head
    void region(int slot, int& arg, size_t& off, size_t& rs) const;

    const inkling2_host::weights& w;
    std::vector<layer_src> src;
    int max_len;
    int pos = 0;
    int pos0 = 0;   // the position of load_state / store_state: the host model has the state up to here
    npu_app_manager* mgr;
    // two apps: the next position's run is created in the other one while a run is on the device
    npu_app apps[2];
    int cur = 0;
    npu_app& app() { return this->apps[this->cur]; }
    std::optional<xrt::run> next_run;
    int next_pos = -1;
    bool run_token_pipe();
#if !defined(FLM_USE_HRX)
    inkling2_launch ll;
    bool lmh_state = false;   // the array was left as the lm_head's (the next decode run needs cold_dec first)
    struct window {
        inkling2_array_lock lock{true};
        xrt::runlist rl;
        xrt::run guest;
        bool lm = false;
    };
    std::unique_ptr<window> start_window(xrt::run& run, bool lm);
    bool wait_window(std::unique_ptr<window> w);
#endif
    buffer<uint8_t> io, wt, gl[2], cw;
    proj_plan pp1, pp3;
    size_t RSS = 0, RSG = 0, SW0 = 0;
    int gidx[inkling2_host::NSLOT], sidx[inkling2_host::NSLOT];
};
#else
class inkling2_fused {};   // never made on HRX (inkling2_npu::Impl::make_fused)
#endif   // !FLM_USE_HRX
