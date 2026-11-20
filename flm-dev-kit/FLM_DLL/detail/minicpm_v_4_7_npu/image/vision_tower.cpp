// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/vision_tower.cpp
#include "image/vision_tower.hpp"

#include "image/vision_glue.hpp"
#include "image/vision_par.hpp"
#include "image/vision_profile.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <stdexcept>

namespace mcv {

using vision::activation;

namespace {
std::string block_name(int idx) {
    return std::string("block_") + (idx < 10 ? "0" : "") + std::to_string(idx);
}
}  // namespace

vision_tower::vision_tower(minicpm_v_vision_desc& desc,
                           std::unique_ptr<vision::op_factory> factory)
    : desc_(desc), factory_(std::move(factory)) {
    if (!desc_.present)
        throw std::runtime_error("vision_tower built for a checkpoint with no vision_model_weight");

    // Opening the checkpoint corrects desc_.D_ff / vm_ff from the shipped
    // tensors, so it must happen before any op is bound to a width.
    w_ = std::make_unique<vision::weights>(desc_);

    const int D  = (int)desc_.D;
    const int F  = (int)desc_.D_ff;
    const int PD = (int)desc_.patch_dim();
    const int MI = (int)desc_.merger_in();

    // K = 588 is not a multiple of vision_mm's 384; the device op pads it.
    patch_mm_ = factory_->make_mm("patch_embed");
    patch_mm_->bind(w_->patch_w(), w_->patch_b(), PD, D, activation::none);

    block_ops_.resize(desc_.depth);
    for (uint32_t i = 0; i < desc_.depth; i++) {
        const vision::block_weights& bw = w_->block((int)i);
        block_ops& o = block_ops_[i];
        const std::string tag = "block" + std::to_string(i);
        o.qkv  = factory_->make_mm((tag + ".qkv").c_str());
        o.proj = factory_->make_mm((tag + ".proj").c_str());
        o.fc1  = factory_->make_mm((tag + ".fc1").c_str());
        o.fc2  = factory_->make_mm((tag + ".fc2").c_str());
        o.qkv ->bind(bw.qkv.w,  bw.qkv.b,  D, 3 * D, activation::none);
        o.proj->bind(bw.proj.w, bw.proj.b, D, D,     activation::none);
        // gelu_pytorch_tanh, fused: what vision_mm's rtp_act 2 computes.
        o.fc1 ->bind(bw.fc1.w,  bw.fc1.b,  D, F,     activation::gelu);
        o.fc2 ->bind(bw.fc2.w,  bw.fc2.b,  F, D,     activation::none);
    }

    if (desc_.use_vit_merger()) {
        const vision::vit_merger_weights& vw = w_->vit_merger();
        const int VI = (int)desc_.vm_in();
        const int VF = (int)desc_.vm_ff;
        vm_ops_.qkv  = factory_->make_mm("vit_merger.qkv");
        vm_ops_.proj = factory_->make_mm("vit_merger.proj");
        vm_ops_.fc1  = factory_->make_mm("vit_merger.fc1");
        vm_ops_.fc2  = factory_->make_mm("vit_merger.fc2");
        vm_ops_.qkv ->bind(vw.qkv.w,  vw.qkv.b,  D,  3 * D, activation::none);
        vm_ops_.proj->bind(vw.proj.w, vw.proj.b, D,  D,     activation::none);
        // tanh, NOT erf: the vit_merger hardcodes gelu_pytorch_tanh
        // (modeling_minicpmv4_7.py:327). Only the projector below is erf.
        vm_ops_.fc1 ->bind(vw.fc1.w,  vw.fc1.b,  VI, VF,    activation::gelu);
        vm_ops_.fc2 ->bind(vw.fc2.w,  vw.fc2.b,  VF, D,     activation::none);
    }

    merger_fc1_ = factory_->make_mm("merger.fc1");
    merger_fc2_ = factory_->make_mm("merger.fc2");
    // gelu_erf: MiniCPMV4_7DownsampleMLP is nn.GELU(). vision_mm has no erf, so
    // the device op binds `none` and keeps this epilogue on the host.
    merger_fc1_->bind(w_->merger_fc1().w, w_->merger_fc1().b, MI, MI, activation::gelu_erf);
    merger_fc2_->bind(w_->merger_fc2().w, w_->merger_fc2().b, MI,
                      (int)desc_.out_hidden_size, activation::none);

    attn_ = factory_->make_attn("vision_attn");

    backend_ = std::string("mm=") + patch_mm_->backend() + " attn=" + attn_->backend() +
               (vision::cpu_ops_emulate_bf16() ? " bf16" : " fp32");
}

void vision_tower::_ensure_scratch(int L) {
    if (L <= scratch_L_) return;
    const size_t D  = desc_.D;
    const size_t F  = std::max<size_t>(desc_.D_ff, (desc_.vm_ff + 3) / 4);
    const size_t MI = std::max(desc_.merger_in(), desc_.vm_in());
    const size_t Lq = ((size_t)L + 3) / 4;

    x_.assign((size_t)L * D, 0.0f);
    h_.assign((size_t)L * D, 0.0f);
    qkv_.assign((size_t)L * 3 * D, 0.0f);
    attn_o_.assign((size_t)L * D, 0.0f);
    proj_.assign((size_t)L * D, 0.0f);
    ff_.assign((size_t)L * F, 0.0f);
    merged_.assign(Lq * MI, 0.0f);
    merged_n_.assign(Lq * MI, 0.0f);
    mean_.assign(Lq * D, 0.0f);
    scratch_L_ = L;
    qkv16_.clear(); qkv16_.shrink_to_fit(); attn_o16_.clear(); attn_o16_.shrink_to_fit();
}

void vision_tower::_prepare(const std::vector<view_grid>& views, int L) {
    // Every op's M for this call, then all the device sequences at once. The
    // ops are independent (one app each) and the per-op cost is CPU-side ELF
    // assembly plus a module/kernel build, so this is a plain parallel loop.
    // Cheap when nothing changed: prepare() is a compare per op.
    struct job { vision::mm_op* op; int M; };
    std::vector<job> jobs;
    const int Lq = desc_.use_vit_merger() ? L / 4 : L;
    jobs.push_back({patch_mm_.get(), L});
    // The streamed path runs the four block ops in chunks, so their sequences are
    // built for the chunk's rows, not the image's; the staging slots are sized
    // here too, serially, before any dispatch is in flight.
    for (uint32_t i = 0; i < desc_.depth; i++) {
        const int rows = ((int32_t)i <= desc_.insert_layer_id) ? L : Lq;
        vision::mm_op* ops[4] = {block_ops_[i].qkv.get(), block_ops_[i].proj.get(),
                                 block_ops_[i].fc1.get(), block_ops_[i].fc2.get()};
        if (streamed_) {
            _stream_prepare(rows, ops, 4);
            for (vision::mm_op* o : ops) jobs.push_back({o, _chunk_rows(rows)});
        } else {
            for (vision::mm_op* o : ops) jobs.push_back({o, rows});
        }
    }
    if (desc_.use_vit_merger()) {
        if (streamed_) {
            vision::mm_op* full_ops[2] = {vm_ops_.qkv.get(), vm_ops_.proj.get()};
            vision::mm_op* mlp_ops[2]  = {vm_ops_.fc1.get(), vm_ops_.fc2.get()};
            _stream_prepare(L, full_ops, 2);
            _stream_prepare(Lq, mlp_ops, 2);
            for (vision::mm_op* o : full_ops) jobs.push_back({o, _chunk_rows(L)});
            for (vision::mm_op* o : mlp_ops) jobs.push_back({o, _chunk_rows(Lq)});
        } else {
            jobs.push_back({vm_ops_.qkv.get(), L});
            jobs.push_back({vm_ops_.proj.get(), L});
            jobs.push_back({vm_ops_.fc1.get(), L / 4});
            jobs.push_back({vm_ops_.fc2.get(), L / 4});
        }
    }
    jobs.push_back({merger_fc1_.get(), Lq / 4});
    jobs.push_back({merger_fc2_.get(), Lq / 4});

    std::vector<int> full, quarter;
    for (const view_grid& v : views) {
        full.push_back(v.h * v.w);
        quarter.push_back(v.h * v.w / 4);
    }
    attn_->prepare(full);
    if (desc_.use_vit_merger()) attn_->prepare(quarter);

#pragma omp parallel for schedule(dynamic, 1) num_threads(vision::par_threads())
    for (int j = 0; j < (int)jobs.size(); j++) jobs[(size_t)j].op->prepare(jobs[(size_t)j].M);
}

void vision_tower::_block(int idx, const std::vector<view_grid>& views, int L) {
    const int D = (int)desc_.D;
    const vision::block_weights& bw = w_->block(idx);
    block_ops& op = block_ops_[(size_t)idx];
    const std::string pre = block_name(idx);

    // ---- attention sublayer ------------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, bw.norm1_w.data(), bw.norm1_b.data(),
                         desc_.ln_eps, h_.data()); }
    _emit((pre + ".norm1").c_str(), h_.data(), L, D);

    op.qkv->run(h_.data(), L, qkv_.data());

    // Block-diagonal over VIEWS: each view attends only within itself. q, k and
    // v are read in place out of the fused [L][3D] projection.
    std::vector<int> lengths;
    for (const view_grid& v : views) lengths.push_back(v.h * v.w);
    attn_->run_views(qkv_.data(), qkv_.data() + D, qkv_.data() + 2 * D, 3 * D, lengths,
                     attn_o_.data());

    op.proj->run(attn_o_.data(), L, proj_.data());
    // After out_proj and before the residual: where MiniCPMV4_7VisionAttention
    // returns, so where the reference's `block_NN.attn_proj` is hooked.
    _emit((pre + ".attn_proj").c_str(), proj_.data(), L, D);
    { VPROF(resid_us); vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }

    // ---- MLP sublayer ------------------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, bw.norm2_w.data(), bw.norm2_b.data(),
                         desc_.ln_eps, h_.data()); }
    op.fc1->run(h_.data(), L, ff_.data());          // gelu_tanh fused
    op.fc2->run(ff_.data(), L, proj_.data());
    { VPROF(resid_us); vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }
    _emit((pre + ".out").c_str(), x_.data(), L, D);
}

void vision_tower::_vit_merger(std::vector<view_grid>& views, int L) {
    const int D  = (int)desc_.D;
    const int H  = (int)desc_.num_heads;
    const int VI = (int)desc_.vm_in();
    const int VF = (int)desc_.vm_ff;
    const int Lq = L / 4;
    const vision::vit_merger_weights& vw = w_->vit_merger();

    // ---- windowed attention + residual #1 ----------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, vw.norm1_w.data(), vw.norm1_b.data(),
                         desc_.ln_eps, h_.data()); }
    _emit("vm.norm1", h_.data(), L, D);

    vm_ops_.qkv->run(h_.data(), L, qkv_.data());
    {
        VPROF(wattn_us);
        const float* qkv = qkv_.data();
        size_t off = 0;
        for (const view_grid& v : views) {
            vision::window_attention(qkv + off * 3 * D, qkv + off * 3 * D + D,
                                     qkv + off * 3 * D + 2 * D, 3 * D, v.h, v.w, H,
                                     (int)desc_.DH, desc_.attn_scale, attn_o_.data() + off * D);
            off += (size_t)v.h * v.w;
        }
    }
    vm_ops_.proj->run(attn_o_.data(), L, proj_.data());
    _emit("vm.attn_proj", proj_.data(), L, D);
    { VPROF(resid_us); vision::add_inplace(x_.data(), proj_.data(), (size_t)L * D); }
    _emit("vm.attn_residual", x_.data(), L, D);

    // ---- 2x2 merge, MLP, + mean-of-4 (residual #2) -------------------------
    {
        VPROF(gather_us);
        size_t off = 0, moff = 0;
        for (const view_grid& v : views) {
            vision::regroup_2x2(x_.data() + off * D, v.h, v.w, D,
                                merged_.data() + moff * VI, mean_.data() + moff * D);
            off  += (size_t)v.h * v.w;
            moff += (size_t)(v.h / 2) * (v.w / 2);
        }
    }
    _emit("vm.merged", merged_.data(), Lq, VI);
    _emit("vm.mean", mean_.data(), Lq, D);
    { VPROF(norm_us);
      vision::layer_norm(merged_.data(), Lq, VI, vw.pre_norm_w.data(), vw.pre_norm_b.data(),
                         desc_.ln_eps, merged_n_.data()); }
    _emit("vm.pre_norm", merged_n_.data(), Lq, VI);

    vm_ops_.fc1->run(merged_n_.data(), Lq, ff_.data());     // gelu_tanh fused
    _emit("vm.fc1", ff_.data(), Lq, VF);
    vm_ops_.fc2->run(ff_.data(), Lq, x_.data());
    { VPROF(resid_us); vision::add_inplace(x_.data(), mean_.data(), (size_t)Lq * D); }
    _emit("vm.out", x_.data(), Lq, D);

    for (view_grid& v : views) { v.h /= 2; v.w /= 2; }
}

// ===========================================================================
// the streamed encoder
// ===========================================================================

namespace {
/// Chunks in flight at once: enough that the device always has a queued job while
/// the host works on a finished one, few enough that the staging stays small.
constexpr int kStreamSlots = 3;

int stream_target_rows() {
    static const int v = [] {
        // A tuning knob, not a correctness one: chunking never changes a result.
        if (const char* e = std::getenv("FLM_MINICPM_V_VISION_CHUNK")) {
            const int n = std::atoi(e);
            if (n > 0) return n;
        }
        // 120-frame tower: 1024 -> 8.50 s, 3072 / 4096 -> 7.99 s, 6144 -> 8.01 s.
        return 4096;
    }();
    return v;
}
}  // namespace

bool vision_tower::_can_stream() const {
    // FLM_MINICPM_V_VISION_PIPE=0 is the A/B arm (the sequential path), not a production knob.
    if (const char* e = std::getenv("FLM_MINICPM_V_VISION_PIPE"))
        if (e[0] == '0') return false;
    if (tap_) return false;   // the verify drivers tap every intermediate, which the chunks do not materialise
    if (!attn_->supports_bf16()) return false;
    for (const block_ops& o : block_ops_)
        if (!o.qkv->streamable() || !o.proj->streamable() || !o.fc1->streamable() ||
            !o.fc2->streamable())
            return false;
    if (desc_.use_vit_merger())
        for (const vision::mm_op* o : {vm_ops_.qkv.get(), vm_ops_.proj.get(), vm_ops_.fc1.get(), vm_ops_.fc2.get()})
            if (!o->streamable()) return false;
    return true;
}

/// One chunk size for a stretch of `rows` rows, a multiple of the dispatch tile
/// and the same for every chunk, so each op sees one M and builds one sequence.
/// The last chunk is the short one: its tail rows are zeroed, like any padding.
int vision_tower::_chunk_rows(int rows) const {
    const int tile = std::max(1, block_ops_[0].qkv->stream_tile_m_for(rows));
    const int T    = stream_target_rows();
    const int nch  = std::max(1, (rows + T / 2) / T);
    const int per  = (rows + nch - 1) / nch;
    return (per + tile - 1) / tile * tile;
}

void vision_tower::_stream_prepare(int rows, vision::mm_op* const* ops, int n_ops) {
    const int chunk = _chunk_rows(rows);
    const int nch   = (rows + chunk - 1) / chunk;
    const int slots = std::min(kStreamSlots, nch);
    for (int i = 0; i < n_ops; i++) ops[i]->stream_reserve(chunk, slots);
}

void vision_tower::_run_stream(const std::vector<stream_step>& steps, int rows,
                               const std::function<void(int)>& done) {
    VTRACE("h_stream");
    const int S     = (int)steps.size();
    const int chunk = _chunk_rows(rows);
    const int nch   = (rows + chunk - 1) / chunk;
    const int fly   = std::min(kStreamSlots, nch);

    struct chunk_state {
        int r0 = 0, n = 0, step = 0, slot = 0;
        std::future<void> fut;
    };
    std::vector<chunk_state> cs((size_t)nch);
    for (int c = 0; c < nch; c++) {
        cs[(size_t)c].r0 = c * chunk;
        cs[(size_t)c].n  = std::min(chunk, rows - c * chunk);
    }
    // Jobs run on the device lane in submission order, so the front of this
    // queue is always the next to finish.
    std::deque<chunk_state*> q;
    auto submit = [&](chunk_state& c) {
        c.fut = steps[(size_t)c.step].op->stream_submit(c.slot, c.n);
        q.push_back(&c);
    };
    auto begin = [&](chunk_state& c, int slot) {
        c.slot = slot;
        c.step = 0;
        steps[0].load(slot, c.r0, c.n);
        submit(c);
    };

    int next = 0;
    try {
        for (; next < fly; next++) begin(cs[(size_t)next], next);
        while (!q.empty()) {
            chunk_state* c = q.front();
            q.pop_front();
            {
                VPROF(wait_us);
                c->fut.get();
            }
            steps[(size_t)c->step].store(c->slot, c->r0, c->n);
            // chunks leave the last step in order, so rows [0, r0 + n) are final
            if (c->step + 1 == S && done) done(c->r0 + c->n);
            if (++c->step < S) {
                steps[(size_t)c->step].load(c->slot, c->r0, c->n);
                submit(*c);
            } else if (next < nch) {
                begin(cs[(size_t)next++], c->slot);
            }
        }
    } catch (...) {
        // Nothing queued may outlive the buffers it points into.
        for (chunk_state* c : q) {
            try { c->fut.wait(); } catch (...) {}
        }
        throw;
    }
}

vision_tower::stream_step vision_tower::_head_step(int block) {
    const int D = (int)desc_.D;
    const vision::block_weights& bw = w_->block(block);
    block_ops& op = block_ops_[(size_t)block];
    stream_step s;
    s.op = op.qkv.get();
    s.load = [this, &bw, &op, D](int slot, int r0, int n) {
        { VPROF(norm_us);
          vision::layer_norm(x_.data() + (size_t)r0 * D, n, D, bw.norm1_w.data(), bw.norm1_b.data(),
                             desc_.ln_eps, h_.data() + (size_t)r0 * D); }
        op.qkv->stream_load_f32(slot, h_.data() + (size_t)r0 * D, n);
    };
    s.store = [this, &op, D](int slot, int r0, int n) {
        op.qkv->stream_store_bf16(slot, qkv16_.data() + (size_t)r0 * 3 * D, (size_t)3 * D, n);
    };
    return s;
}

void vision_tower::_tail_steps(int block, std::vector<stream_step>& out) {
    const int D = (int)desc_.D;
    const vision::block_weights& bw = w_->block(block);
    block_ops& op = block_ops_[(size_t)block];

    stream_step proj;
    proj.op = op.proj.get();
    proj.load = [this, &op, D](int slot, int r0, int n) {
        op.proj->stream_load_bf16(slot, attn_o16_.data() + (size_t)r0 * D, n);
    };
    proj.store = [this, &op, D](int slot, int r0, int n) {
        VPROF(resid_us);
        op.proj->stream_store_add(slot, x_.data() + (size_t)r0 * D, n);
    };
    stream_step fc1;
    fc1.op = op.fc1.get();
    fc1.load = [this, &bw, &op, D](int slot, int r0, int n) {
        { VPROF(norm_us);
          vision::layer_norm(x_.data() + (size_t)r0 * D, n, D, bw.norm2_w.data(), bw.norm2_b.data(),
                             desc_.ln_eps, h_.data() + (size_t)r0 * D); }
        op.fc1->stream_load_f32(slot, h_.data() + (size_t)r0 * D, n);
    };
    fc1.store = [&op](int slot, int, int n) { op.fc1->stream_store_gelu_to(slot, *op.fc2, n); };
    stream_step fc2;
    fc2.op = op.fc2.get();
    fc2.load = [](int, int, int) {};   // the fc1 store already filled this slot's x
    fc2.store = [this, &op, D](int slot, int r0, int n) {
        VPROF(resid_us);
        op.fc2->stream_store_add(slot, x_.data() + (size_t)r0 * D, n);
    };
    out.push_back(std::move(proj));
    out.push_back(std::move(fc1));
    out.push_back(std::move(fc2));
}

/// A _run_stream `done` hook that stages the next attention's first dispatch (attn_op::stage_bf16)
/// once the qkv16_ rows it reads are final, so its copy overlaps the stretch's remaining chunks.
std::function<void(int)> vision_tower::_attn_stager(const std::vector<view_grid>& views) {
    std::vector<int> lengths;
    for (const view_grid& v : views) lengths.push_back(v.h * v.w);
    const int need = attn_->stage_rows(lengths);
    if (need <= 0) return {};
    const int D = (int)desc_.D;
    return [this, D, need, lengths, staged = false](int rows) mutable {
        if (staged || rows < need) return;
        VTRACE("h_stage");
        attn_->stage_bf16(qkv16_.data(), 3 * D, lengths);
        staged = true;
    };
}

/// A _run_stream `done` hook that calls fn(view, its first row) for each view, in order, once all its rows are final.
std::function<void(int)> vision_tower::_per_view(const std::vector<view_grid>& views,
                                                 std::function<void(const view_grid&, size_t)> fn) {
    return [views, fn = std::move(fn), next = (size_t)0, off = (size_t)0](int rows) mutable {
        while (next < views.size() && off + (size_t)views[next].h * views[next].w <= (size_t)rows) {
            fn(views[next], off);
            off += (size_t)views[next].h * views[next].w;
            next++;
        }
    };
}

/// The vit_merger's window attention, per view as the stretch ending in _vm_qkv_step finishes its rows.
std::function<void(int)> vision_tower::_vm_wattn_hook(const std::vector<view_grid>& views) {
    const int D = (int)desc_.D;
    return _per_view(views, [this, D](const view_grid& v, size_t off) {
        VPROF(wattn_us);
        const float* qkv = qkv_.data() + off * 3 * D;
        vision::window_attention(qkv, qkv + D, qkv + 2 * D, 3 * D, v.h, v.w, (int)desc_.num_heads,
                                 (int)desc_.DH, desc_.attn_scale, attn_o_.data() + off * D);
    });
}

vision_tower::stream_step vision_tower::_vm_qkv_step() {
    const int D = (int)desc_.D;
    const vision::vit_merger_weights& vw = w_->vit_merger();
    stream_step s;
    s.op = vm_ops_.qkv.get();
    s.load = [this, &vw, D](int slot, int r0, int n) {
        { VPROF(norm_us);
          vision::layer_norm(x_.data() + (size_t)r0 * D, n, D, vw.norm1_w.data(), vw.norm1_b.data(),
                             desc_.ln_eps, h_.data() + (size_t)r0 * D); }
        vm_ops_.qkv->stream_load_f32(slot, h_.data() + (size_t)r0 * D, n);
    };
    // fp32, as run() writes it: the window attention reads it on the host
    s.store = [this, D](int slot, int r0, int n) {
        vm_ops_.qkv->stream_store_f32(slot, qkv_.data() + (size_t)r0 * 3 * D, n);
    };
    return s;
}

/// _vit_merger with its matmuls streamed. Its qkv and window attention already ran in the stretch
/// before (_vm_wattn_hook); the 2x2 regroup runs per view inside the proj's stretch, so the host work
/// overlaps the device. The residual #2 is seeded with the mean
/// and fc2's output added to it, the same sum in the other order. `next` > 0: that block's norm1 +
/// qkv rides in the MLP's stretch.
void vision_tower::_vit_merger_streamed(std::vector<view_grid>& views, int L, int next) {
    const int D  = (int)desc_.D;
    const int VI = (int)desc_.vm_in();
    const int Lq = L / 4;
    const vision::vit_merger_weights& vw = w_->vit_merger();

    stream_step proj;
    proj.op = vm_ops_.proj.get();
    proj.load = [this, D](int slot, int r0, int n) {
        vm_ops_.proj->stream_load_f32(slot, attn_o_.data() + (size_t)r0 * D, n);
    };
    proj.store = [this, D](int slot, int r0, int n) {
        VPROF(resid_us);
        vm_ops_.proj->stream_store_add(slot, x_.data() + (size_t)r0 * D, n);
    };
    // the 2x2 regroup of each view as soon as its rows are final
    _run_stream({proj}, L, _per_view(views, [this, D, VI](const view_grid& v, size_t off) {
        VPROF(gather_us);
        vision::regroup_2x2(x_.data() + off * D, v.h, v.w, D,
                            merged_.data() + off / 4 * VI, mean_.data() + off / 4 * D);
    }));

    std::vector<stream_step> steps(2);
    steps[0].op = vm_ops_.fc1.get();
    steps[0].load = [this, &vw, D, VI](int slot, int r0, int n) {
        // residual #2's seed; x_'s old rows below Lq were all regrouped already
        std::memcpy(x_.data() + (size_t)r0 * D, mean_.data() + (size_t)r0 * D, (size_t)n * D * sizeof(float));
        { VPROF(norm_us);
          vision::layer_norm(merged_.data() + (size_t)r0 * VI, n, VI, vw.pre_norm_w.data(),
                             vw.pre_norm_b.data(), desc_.ln_eps, merged_n_.data() + (size_t)r0 * VI); }
        vm_ops_.fc1->stream_load_f32(slot, merged_n_.data() + (size_t)r0 * VI, n);
    };
    steps[0].store = [this](int slot, int, int n) {
        vm_ops_.fc1->stream_store_gelu_to(slot, *vm_ops_.fc2, n);
    };
    steps[1].op = vm_ops_.fc2.get();
    steps[1].load = [](int, int, int) {};   // the fc1 store already filled this slot's x
    steps[1].store = [this, D](int slot, int r0, int n) {
        VPROF(resid_us);
        vm_ops_.fc2->stream_store_add(slot, x_.data() + (size_t)r0 * D, n);
    };
    (void)Lq;
    for (view_grid& v : views) { v.h /= 2; v.w /= 2; }
    if (next > 0) steps.push_back(_head_step(next));
    _run_stream(steps, Lq, next > 0 ? _attn_stager(views) : std::function<void(int)>());
}

void vision_tower::_encode_blocks_streamed(std::vector<view_grid>& views, int& L) {
    const int D = (int)desc_.D;
    qkv16_.resize((size_t)L * 3 * D);
    attn_o16_.resize((size_t)L * D);

    // embed = patch + pos, per chunk ahead of block 0's norm (encode left the pos rows in h_)
    stream_step head0 = _head_step(0);
    head0.load = [this, D, inner = head0.load](int slot, int r0, int n) {
        { VPROF(resid_us);
          vision::add_inplace(x_.data() + (size_t)r0 * D, h_.data() + (size_t)r0 * D, (size_t)n * D); }
        inner(slot, r0, n);
    };
    _run_stream({head0}, L, _attn_stager(views));
    for (int i = 0; i < (int)desc_.depth; i++) {
        std::vector<int> lengths;
        for (const view_grid& v : views) lengths.push_back(v.h * v.w);
        { VTRACE("h_attn"); attn_->run_views_bf16(qkv16_.data(), 3 * D, lengths, attn_o16_.data()); }

        std::vector<stream_step> steps;
        _tail_steps(i, steps);
        const bool last  = i + 1 >= (int)desc_.depth;
        const bool merge = i == desc_.insert_layer_id;
        // The next op that reads these rows rides in the same stretch: the next
        // block's norm + qkv, or the vit_merger's.
        if (merge)      steps.push_back(_vm_qkv_step());
        else if (!last) steps.push_back(_head_step(i + 1));
        _run_stream(steps, L, merge ? _vm_wattn_hook(views)
                              : !last ? _attn_stager(views) : std::function<void(int)>());

        if (merge) {
            _vit_merger_streamed(views, L, last ? -1 : i + 1);
            L /= 4;
        }
    }
}

void vision_tower::encode(const float* pixel_values, const std::vector<view_grid>& views_in,
                          float* out) {
    const int D  = (int)desc_.D;
    const int MI = (int)desc_.merger_in();
    const int gm = desc_.grid_multiple();
    if (views_in.empty()) throw std::runtime_error("vision_tower::encode got no views");

    int L = 0, max_view = 0;
    for (const view_grid& v : views_in) {
        if (v.h <= 0 || v.w <= 0 || v.h % gm || v.w % gm)
            throw std::runtime_error("vision_tower::encode: view " + std::to_string(v.h) + "x" +
                                     std::to_string(v.w) + " is not a positive multiple of " +
                                     std::to_string(gm) + " patches");
        L += v.h * v.w;
        max_view = std::max(max_view, v.h * v.w);
    }
    _ensure_scratch(L);

    // Sized for the largest view seen so far: reconfiguring per call would
    // rebuild device state per image.
    if (max_view > attn_configured_L_) {
        attn_->configure(max_view, (int)desc_.num_heads, (int)desc_.DH, desc_.attn_scale);
        attn_configured_L_ = max_view;
    }

    streamed_ = _can_stream();
    _prepare(views_in, L);

    // ---- patch embedding + position table -----------------------------------
    // The position rows are looked up on a helper thread while the patch matmul runs.
    auto pos = std::async(std::launch::async, [&] {
        VPROF(pos_us);
        size_t off = 0;
        for (const view_grid& v : views_in) {
            vision::pos_embed_lookup(w_->pos_embed(), (int)desc_.num_patches_per_side, D,
                                     v.h, v.w, h_.data() + off * D);
            off += (size_t)v.h * v.w;
        }
    });
    try {
        patch_mm_->run(pixel_values, L, x_.data());
    } catch (...) {
        pos.wait();
        throw;
    }
    pos.get();
    _emit("patch_embed", x_.data(), L, D);
    _emit("pos_embed", h_.data(), L, D);
    // the streamed encoder adds them chunk by chunk (_encode_blocks_streamed)
    if (!streamed_) { VPROF(resid_us); vision::add_inplace(x_.data(), h_.data(), (size_t)L * D); }
    _emit("embed", x_.data(), L, D);

    // ---- encoder, with the vit_merger after block insert_layer_id ------------
    std::vector<view_grid> views = views_in;
    if (streamed_) {
        _encode_blocks_streamed(views, L);
    } else {
        for (uint32_t i = 0; i < desc_.depth; i++) {
            _block((int)i, views, L);
            if ((int32_t)i == desc_.insert_layer_id) {
                _vit_merger(views, L);
                L /= 4;
            }
        }
    }

    // ---- post_layernorm + projector -------------------------------------------
    { VPROF(norm_us);
      vision::layer_norm(x_.data(), L, D, w_->post_ln_w().data(), w_->post_ln_b().data(),
                         desc_.ln_eps, h_.data()); }
    _emit("post_ln", h_.data(), L, D);

    const int rows = L / 4;
    {
        VPROF(gather_us);
        size_t off = 0, moff = 0;
        for (const view_grid& v : views) {
            vision::regroup_2x2(h_.data() + off * D, v.h, v.w, D, merged_.data() + moff * MI,
                                nullptr);
            off  += (size_t)v.h * v.w;
            moff += (size_t)(v.h / 2) * (v.w / 2);
        }
    }
    _emit("merger.merged", merged_.data(), rows, MI);
    // MiniCPMV4_7DownsampleMLP builds this LayerNorm with eps=1e-6 explicitly.
    { VPROF(norm_us);
      vision::layer_norm(merged_.data(), rows, MI, w_->pre_norm_w().data(),
                         w_->pre_norm_b().data(), 1e-6f, merged_n_.data()); }
    _emit("merger.pre_norm", merged_n_.data(), rows, MI);
    merger_fc1_->run(merged_n_.data(), rows, merged_.data());   // erf gelu
    _emit("merger_fc1", merged_.data(), rows, MI);
    merger_fc2_->run(merged_.data(), rows, out);
    _emit("image_embeds", out, rows, (int)desc_.out_hidden_size);
}

}  // namespace mcv
