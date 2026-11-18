// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/npu/proj_op.hpp
/// \brief One projection on `dequant_mm.xclbin`: fused Q4_K dequant + GEMM,
///        over a weight BO some other object owns.
///
/// Same shape as `prefill_attn_op_t` and for the same reasons
/// (prefill/npu/prefill_attn_op.hpp). This is deliberately NOT a layer: the
/// bitstream computes `y = x . W^T` and an optional activation, nothing else,
/// so the natural object is an operator the caller invokes between two CPU
/// halves. It holds no weights -- the caller hands the BO in, and BOs are
/// device-scoped (`npu_app::create_bo_buffer` builds `buffer<T>(*device,
/// size)`), so `npu_mtp_layer`'s 245 MB `proj_` BO is legal here even though it
/// was allocated from the MTP_layer app.
///
/// What it does own is the x and y staging BOs. They are created from this app
/// for tidiness, not out of necessity: `npu_app::create_bo_buffer` is
/// `buffer<T>(*this->device, size)` (npu_utils.hpp), so a BO binds to the
/// device and not to the hardware context, and any app over any xclbin may
/// dispatch against it. `qwen3_5vl_npu` relies on exactly that -- its `qkv` BO
/// is allocated from a `mm.xclbin` app and handed to `conv.xclbin`'s
/// (qwen3_5vl_prefill.hpp:122, :161).
///
/// That matters because it is what the activation hand-off needs: one bf16
/// buffer can be the y of one projection and the x of the next, across apps,
/// without the fp32 round trip `run()` pays. `stage()` / `run_staged()` are
/// that entry point -- `run()` and `run_tile()` are the fp32 one, kept because
/// the host half of a layer genuinely is fp32.
///
/// **The shape is pinned for the life of the object.** `Dout`, `Din`, the
/// weight offset and the pass list are constructor arguments and the sequence
/// is emitted once, because all four are baked into every descriptor. The MTP
/// head therefore builds five of these, one per projection, out of five apps
/// over one xclbin.
///
/// `L` is pinned too, at `PASS_ROWS` (512), and a longer window is tiled here
/// rather than dispatched wide. That is not a limitation being worked around:
/// the engine's DDR traffic is linear in L in every term, so a wide dispatch
/// moves exactly as many bytes as the tiles do while needing an L-sized
/// staging BO and an L-sized instruction sequence. See the cost model in
/// proj_sequence.hpp.
#ifndef __QWEN3_8MTP_NPU_PROJ_OP_HPP__
#define __QWEN3_8MTP_NPU_PROJ_OP_HPP__

#include "speculate/npu/proj_sequence.hpp"
#include "verify/npu/q4nx.hpp"      // q38npu::q4k_bytes, for the BO size guard
#include "common/cpu_simd.hpp"      // the vectorised fp32 <-> bf16 marshalling
#include "common/host_par.hpp"      // q38npu::host_par_min

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"
#include "utils/utils.hpp"          // time_utils, for the PREFILL_PROFILE counters

#include <cstdlib>          // getenv, for the host-threading gate
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q38npu {

/// `host_par_min()` used to live here; it is `common/host_par.hpp` now, because
/// the per-row stages in `cpu_ref::attention_layer` and
/// `cpu_ref::delta_net_layer` want the same decision and sit underneath this
/// header. It is still `q38npu::host_par_min()` and every call site below is
/// unchanged; the measurements that justify the gate moved with it.

#ifdef QWEN3_8MTP_PREFILL_PROFILE
/// Where a `dequant_mm` tile's wall clock goes, accumulated across every
/// projection in the process. Built only under `make PREFILL_PROFILE=1`; the
/// counters exist so the per-stage ticks in prefill_cpu.hpp can be split into
/// the part the device spends and the part the host spends staging for it.
/// Not thread safe, and it does not need to be -- the prefill path is serial.
struct dequant_mm_prof {
    static inline double stage_us    = 0;  ///< fp32 -> bf16 + sync_to_device
    static inline double dispatch_us = 0;  ///< the blocking NPU call
    static inline double read_us     = 0;  ///< sync_from_device
    static inline double widen_us    = 0;  ///< bf16 -> fp32 in run()
    /// The caller-side halves, filled in by prefill_proj / delta_proj: sizing
    /// an output vector, and the scatter or un-weave that re-lays the device's
    /// fused output out.
    ///
    /// The sizing used to be `assign(n, 0.0f)` and measured 7.0% of an L = 1024
    /// window -- a memset of every output buffer in module 1, each one fully
    /// overwritten by the very next statement. They are `resize(n)` now, so in
    /// steady state this counter reads ~0 and only a window that grows pays
    /// anything. **The invariant that makes that legal is per call site: every
    /// one of the five writes all `L` rows and all columns of what it sized**
    /// -- `dequant_mm_op::run` covers the whole `[L][Dout]`, the scatter loops
    /// cover `r in [0, L)` and the un-weave covers all `DI / 512` bands. A
    /// future partial writer has to put the zeroing back, and nothing here will
    /// warn it: the symptom would be stale rows from the previous layer, which
    /// is plausible-looking garbage rather than a fault.
    static inline double zerofill_us = 0;
    static inline double scatter_us  = 0;
    /// The other two prefill bitstreams' blocking calls, so the per-stage tick
    /// can charge the device for the attention core and the conv + fold too.
    static inline double other_dispatch_us = 0;
    /// Open brackets for the two above. Plain statics rather than locals so the
    /// open and the close can sit in different statements of the same function
    /// without each pair needing its own variable name.
    static inline time_utils::time_point mark_z{};
    static inline time_utils::time_point mark_s{};
    static inline long   tiles       = 0;
    static void reset() {
        stage_us = dispatch_us = read_us = widen_us = 0;
        zerofill_us = scatter_us = other_dispatch_us = 0;
        tiles = 0;
    }
};
#endif

/// Bracket a caller-side zero-fill (`PROF8_Z0/Z1`) or scatter (`PROF8_S0/S1`)
/// so its cost lands in `dequant_mm_prof`. Nested Z inside S is not supported
/// and does not occur. No-ops without the profile flag.
#ifdef QWEN3_8MTP_PREFILL_PROFILE
#define PROF8_Z0() q38npu::dequant_mm_prof::mark_z = time_utils::now()
#define PROF8_Z1() q38npu::dequant_mm_prof::zerofill_us +=                     \
                       time_utils::duration_us(q38npu::dequant_mm_prof::mark_z,\
                                               time_utils::now()).first
#define PROF8_S0() q38npu::dequant_mm_prof::mark_s = time_utils::now()
/// Wrap a non-dequant_mm blocking dispatch so it counts as device time.
#define PROF8_DEV(expr)                                                        \
    [&]{ const auto __d0 = time_utils::now();                                  \
         auto __r = (expr);                                                    \
         q38npu::dequant_mm_prof::other_dispatch_us +=                         \
             time_utils::duration_us(__d0, time_utils::now()).first;           \
         return __r; }()
#define PROF8_S1() q38npu::dequant_mm_prof::scatter_us +=                      \
                       time_utils::duration_us(q38npu::dequant_mm_prof::mark_s,\
                                               time_utils::now()).first
#else
#define PROF8_Z0() do {} while (0)
#define PROF8_Z1() do {} while (0)
#define PROF8_S0() do {} while (0)
#define PROF8_S1() do {} while (0)
#define PROF8_DEV(expr) (expr)
#endif

class dequant_mm_op {
public:
    /// Rows one dispatch covers. A window is tiled at this grain.
    static constexpr int ROWS = dequant_mm_tiles::PASS_ROWS;   // 512

    /// \param a    an app over dequant_mm.xclbin. One app per projection: the
    ///             sequence is per-shape and an app owns exactly one.
    /// \param Dout output width, a multiple of 512
    /// \param Din  reduction width, a multiple of 512
    /// \param w_byte_offset  where this matrix starts in the weight BO handed
    ///             to `run()`. **Bytes**, matching `mtp_layout`'s `w_*_offset`
    ///             fields -- never the `w_*_words()` accessors, which exist for
    ///             the other bitstream's bf16-element descriptors.
    /// \param passes  the output-column blocks and their activations. One
    ///             `{0, 1, act}` pass is the common case; the woven `up|gate`
    ///             takes two so silu lands on its gate bands alone. They must
    ///             partition the blocks -- `gen_dequant_mm_seq` checks it.
    /// \param rtp  the per-build core RTP addresses
    dequant_mm_op(npu_app a, int Dout, int Din, size_t w_byte_offset,
                  std::vector<dequant_mm_pass> passes,
                  const dequant_mm_rtp_addresses& rtp = dequant_mm_rtp_addresses{})
        : app_(std::move(a)), Dout_((size_t)Dout), Din_((size_t)Din),
          w_off_(w_byte_offset), passes_(std::move(passes)) {
        if (Dout <= 0 || Din <= 0)
            throw std::runtime_error("dequant_mm_op: non-positive shape");

        // Emitted once and never again: ROWS, Dout, Din, the offset and the
        // passes are all fixed, and they are the whole of the sequence's
        // input. `npu_app::operator()` notices the version and sets the kernel
        // up on the first dispatch.
        gen_dequant_mm_seq(app_.seq(), (uint32_t)ROWS, (uint32_t)Dout,
                           (uint32_t)Din, w_byte_offset, passes_, rtp);

        // Fixed size, so no grow-only staging: unlike the attention op, whose
        // BOs scale with the window, these are the tile's and the tile never
        // changes. Zeroed once here and only the live rows are rewritten, so
        // the tail-tile pad rows below stay zero without a per-call memset of
        // the whole buffer.
        _build_slot(0);
    }

    /// One pass over every column block at a single activation -- what every
    /// projection but the woven `up|gate` wants.
    dequant_mm_op(npu_app a, int Dout, int Din, size_t w_byte_offset,
                  int act = dequant_mm_act_none,
                  const dequant_mm_rtp_addresses& rtp = dequant_mm_rtp_addresses{})
        : dequant_mm_op(std::move(a), Dout, Din, w_byte_offset,
                        std::vector<dequant_mm_pass>{{0, 1, act}}, rtp) {}

    int    Dout() const { return (int)Dout_; }
    int    Din()  const { return (int)Din_; }
    size_t weight_offset() const { return w_off_; }
    const std::vector<dequant_mm_pass>& passes() const { return passes_; }

    /// `y[L][Dout] = x[L][Din] . W^T`, tiling `L` at `ROWS`.
    ///
    /// \param x  [L][Din] fp32, row major. Rounded to bf16 on the way in --
    ///           the device reduces bf16, so a reference meant to isolate this
    ///           kernel's error must round its own x the same way
    /// \param L  any positive row count; the last tile is zero-padded
    /// \param w  the weight BO. Must hold this op's matrix at
    ///           `weight_offset()`; only that region is read
    /// \param y  [L][Dout] fp32 out
    /// \param resid  optional [L][Dout] fp32 added to `y` during the widen. The
    ///           residual adds on both block paths are a read-modify-write over
    ///           a buffer this loop is already writing, so folding them here
    ///           costs nothing and saves a pass. `y` and `resid` may not alias.
    ///           The sum is fp32 either way, so it is bit-identical to a
    ///           separate `y[i] += resid[i]` afterwards.
    /// \param resid  see above
    ///
    /// Above one tile this runs as a **two-stage software pipeline**: tile
    /// `t+1`'s fp32 -> bf16 staging happens while tile `t` is on the device,
    /// and tile `t`'s read-back and widen happen while tile `t+1` is. The
    /// device order, and so the arithmetic, is exactly the serial order's --
    /// only host work moves, and it moves into wall clock the device was
    /// already spending. Output is bit-identical (checked when this was written).
    ///
    /// At `L <= ROWS` there is one tile and nothing to overlap, so the serial
    /// loop runs and the second slot is never allocated. That is the whole of
    /// the decode path and all of MTP step 0.
    void run(const float* x, int L, buffer<u8>& w, float* y,
             const float* resid = nullptr) {
        run_tiles(x, L, w, [&](const bf16* out, int rows, int t) {
            _widen(out, rows, y + (size_t)t * Dout_,
                   resid ? resid + (size_t)t * Dout_ : nullptr);
        });
    }

    /// Tile `L` rows through this op, handing each tile's bf16 output to
    /// `consume(y, rows, t)` -- `y` being `[ROWS][Dout]` with `rows` live, and
    /// `t` the tile's first row.
    ///
    /// This is the pipeline, and `run()` is one caller of it. Every consumer on
    /// this path re-lays the output out anyway -- a scatter into three fp32
    /// vectors, an un-weave and multiply, a plain widen -- so the loop is
    /// "dispatch a tile, then walk the previous one's rows", and the walk is
    /// exactly the host work there is to hide.
    ///
    /// Above one tile it runs as a **two-stage software pipeline**: tile
    /// `t+1`'s fp32 -> bf16 staging happens while tile `t` is on the device,
    /// and tile `t`'s read-back and `consume` happen while tile `t+1` is. The
    /// device order, and so the arithmetic, is exactly the serial order's --
    /// only host work moves, and it moves into wall clock the device was
    /// already spending. **Output is bit-identical**, and the whole-log diff of
    /// `verify_qwen3_8mtp` is what says so.
    ///
    /// At `L <= ROWS` there is one tile and nothing to overlap, so the serial
    /// loop runs and the second slot is never allocated. That is the whole of
    /// the decode path and all of MTP step 0 at live widths.
    ///
    /// `consume` must not dispatch this op.
    template <class Consume>
    void run_tiles(const float* x, int L, buffer<u8>& w, Consume&& consume) {
        if (L <= 0) return;
        if (L <= ROWS || !pipeline_enabled()) {
            for (int t = 0; t < L; t += ROWS) {
                const int rows = (L - t) < ROWS ? (L - t) : ROWS;
                consume(run_tile(x + (size_t)t * Din_, rows, w), rows, t);
            }
            return;
        }

        _ensure_pipelined();
        const int first = ROWS < L ? ROWS : L;
        _stage_rows(x, first);
        dispatch_staged(first, w);

        for (int t = 0; t < L; t += ROWS) {
            const int rows = (L - t) < ROWS ? (L - t) : ROWS;
            const int nt   = t + ROWS;
            const int next = nt < L ? ((L - nt) < ROWS ? (L - nt) : ROWS) : 0;

            // Host work for the NEXT tile, while THIS one is on the device.
            if (next) _stage_rows(x + (size_t)nt * Din_, next);

            const bf16* out = finish();            // tile t lands
            if (next) dispatch_staged(next, w);    // device busy again at once

            // ...and host work for THIS tile, while the next one is running.
            consume(out, rows, t);
        }
    }

    /// One tile, leaving the result in the y BO and handing back the bf16
    /// rows.
    ///
    /// For callers that have to re-lay-out the output anyway -- the woven
    /// up|gate -- so the scatter folds into the bf16 -> fp32 conversion
    /// instead of paying a second pass over 2 x 17408 x L floats. The returned
    /// pointer is valid until the next call on this object.
    ///
    /// \param x     [rows][Din] fp32
    /// \param rows  1 .. ROWS
    /// \param w     the weight BO
    /// \return      [ROWS][Dout] bf16; rows >= `rows` are the zero pad
    const bf16* run_tile(const float* x, int rows, buffer<u8>& w) {
        _stage_rows(x, rows);
        return run_staged(rows, w);
    }

    /// `stage(rows)` plus the fp32 -> bf16 fill of it, counted as staging.
    ///
    /// Split out of `run_tile` because the pipeline in `run()` needs to do this
    /// for the next tile without also dispatching it.
    void _stage_rows(const float* x, int rows) {
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        const auto __s0 = time_utils::now();
#endif
        bf16* xd = stage(rows);
        const size_t n = (size_t)rows * Din_;
        // Vectorised first, threaded second: `q38simd::f32_to_bf16` is ~4x one
        // core, and the threads are worth another ~2x on top of it -- but only
        // where the barriers sleep, which is what the gate is for.
#pragma omp parallel for schedule(static) if (n >= host_par_min())
        for (int r = 0; r < rows; r++)
            q38simd::f32_to_bf16(xd + (size_t)r * Din_, x + (size_t)r * Din_, Din_);
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        dequant_mm_prof::stage_us +=
            time_utils::duration_us(__s0, time_utils::now()).first;
#endif
    }

    /// The x staging BO itself, for a producer that can write bf16 directly.
    ///
    /// `run_tile` exists because the host half of this module speaks fp32. Where
    /// the producer of x is *itself* a device output -- the woven `up|gate` y
    /// feeding `down` is the case that matters, at `L x 17408` the widest buffer
    /// on the path -- the fp32 is a pure round trip: the value arrives as bf16,
    /// is widened, is written to a `std::vector<float>`, is read back, and is
    /// narrowed to the same bf16 it started as. This hands out the destination
    /// so the producer writes it once.
    ///
    /// **The caller must write all `rows * Din()` elements before calling
    /// `run_staged(rows, w)`, and must not dispatch this op in between.**
    /// Nothing checks the coverage, and a partial write is silent -- it reads
    /// as stale rows from the previous tile, which look like plausible
    /// activations.
    ///
    /// Rows above `rows` are the zero pad and are maintained here. They are
    /// ZEROED, not replicated as the attention op's are: a zero row through a
    /// matmul is exactly a zero output row, so the pad cannot contaminate a
    /// real row and cannot produce a NaN for one to be confused with. The
    /// attention op replicates instead because a zero query there would make
    /// every pad row's softmax uniform over the cache, which is harmless but
    /// indistinguishable from a real row that went wrong.
    ///
    /// \return [ROWS][Din] bf16, row major, valid until the next call on this
    ///         object
    bf16* stage(int rows) {
        _check_rows(rows, "stage");
        slot& s = s_[w_slot_];
        bf16* xd = s.x.data();
        // Only the live rows are written by the caller. The BO was zeroed at
        // construction and rows >= `rows` are only ever touched by a longer
        // tile, so the stale tail of a previous longer call is cleared here.
        // Per slot, not per op: the two x buffers carry independent tails.
        if (rows < s.live_rows)
            std::memset(xd + (size_t)rows * Din_, 0,
                        (size_t)(s.live_rows - rows) * Din_ * sizeof(bf16));
        s.live_rows = rows;
        return xd;
    }

    /// Dispatch on whatever `stage(rows)` handed out, with the same `rows`.
    ///
    /// \return [ROWS][Dout] bf16; rows >= `rows` are the zero pad
    const bf16* run_staged(int rows, buffer<u8>& w) {
        dispatch_staged(rows, w);
        return finish();
    }

    /// Start a dispatch on whatever `stage(rows)` last handed out and return
    /// WITHOUT waiting for it. Pairs one-to-one with `finish()`.
    ///
    /// **At most one dispatch per op may be in flight**, and this throws rather
    /// than allowing two. That is not a conservative choice, it is what the
    /// runtime supports: `hrx::run::wait()` is `hrx_stream_wait(rt().stream)`
    /// on the one global stream (hrx_cpp.hpp), so it waits for everything
    /// queued and not for a particular run. With one in flight that is exact;
    /// with two, "wait for tile t" would silently also wait for tile t+1 and
    /// the pipeline would be a serial loop with extra buffers.
    ///
    /// The same fact is why this overlaps the HOST and not two dispatches: one
    /// stream means the device still runs the tiles in order, which is the
    /// reason the result is bit-identical.
    void dispatch_staged(int rows, buffer<u8>& w) {
        _check_rows(rows, "dispatch_staged");
        slot& s = s_[w_slot_];
        if (rows != s.live_rows)
            throw std::runtime_error(
                "dequant_mm_op::dispatch_staged: " + std::to_string(rows) +
                " rows, but stage() was last called with " +
                std::to_string(s.live_rows));
        if (inflight_ >= 0)
            throw std::runtime_error(
                "dequant_mm_op::dispatch_staged: a dispatch is already in "
                "flight -- finish() it first; hrx::run::wait() waits the whole "
                "stream, so this op keeps at most one");
        if (w.size() < w_off_ + q4k_bytes(Dout_ * Din_))
            throw std::runtime_error(
                "dequant_mm_op::dispatch_staged: the weight BO is " +
                std::to_string(w.size()) + " B, too small for a " +
                std::to_string(Dout_) + " x " + std::to_string(Din_) +
                " matrix at byte offset " + std::to_string(w_off_));

        // The sync counts as staging, not as dispatch: it is the cost of
        // getting x to the device, and it is the one part of that cost both
        // entry points pay.
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        const auto __s0 = time_utils::now();
#endif
        s.x.sync_to_device();
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        dequant_mm_prof::stage_us +=
            time_utils::duration_us(__s0, time_utils::now()).first;
        mark_d_ = time_utils::now();
#endif
        s.run = app_.create_run(s.y, s.x, w);
        s.run.start();
        inflight_ = w_slot_;
        w_slot_  ^= pipelined_ ? 1 : 0;
    }

    /// Wait for the dispatch `dispatch_staged` started, read y back, hand it
    /// over.
    ///
    /// \return [ROWS][Dout] bf16; rows >= the dispatched `rows` are the zero
    ///         pad. Valid until this op's slot is dispatched again -- two
    ///         tiles, not one, when the pipeline is on
    const bf16* finish() {
        if (inflight_ < 0)
            throw std::runtime_error(
                "dequant_mm_op::finish: nothing in flight");
        const int i = inflight_;
        inflight_ = -1;
        slot& s = s_[i];

        const ert_cmd_state st = s.run.wait();
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        const auto __s2 = time_utils::now();
        dequant_mm_prof::dispatch_us += time_utils::duration_us(mark_d_, __s2).first;
        dequant_mm_prof::tiles++;
#endif
        if (st != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(
                "dequant_mm_op::finish: dispatch state " +
                std::to_string((int)st) + " for a " + std::to_string(Dout_) +
                " x " + std::to_string(Din_) + " projection at byte offset " +
                std::to_string(w_off_) +
                " -- a timeout is a sequence bug or a stale RTP address (diff "
                "dequant_mm.mlir.prj/input_with_addresses.mlir against "
                "dequant_mm_rtp_addresses)");

        s.y.sync_from_device();
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        dequant_mm_prof::read_us +=
            time_utils::duration_us(__s2, time_utils::now()).first;
#endif
        return s.y.data();
    }

    /// Whether `run()` may pipeline. `FLM_Q38_PIPELINE=0` turns it off, and
    /// the off arm is the serial loop this replaced rather than a host path.
    static bool pipeline_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("FLM_Q38_PIPELINE");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    /// Allocate the second slot, once. Called only from a path that has more
    /// than one tile to run, so an op that only ever sees `L <= ROWS` -- every
    /// op on the decode path, and all five of MTP step 0's at live widths --
    /// never pays the second x and y BO.
    ///
    /// The pair costs `ROWS * (Din + Dout) * 2` bytes: 41 MB for the widest
    /// (`up|gate`, Dout = 34816), ~193 MB across module 1's eight prefill ops.
    /// That is against a 4 GB *per BO* addressing limit with no aggregate one
    /// and ~1% of the resident model.
    void _ensure_pipelined() {
        if (pipelined_) return;
        _build_slot(1);
        pipelined_ = true;
    }

    /// `y[rows][Dout]` fp32 (+ `resid`) out of a tile's bf16, counted as widen.
    ///
    /// Public because a caller that drives `stage`/`run_staged` itself still
    /// wants the same loop, and the profile counter should see it.
    void _widen(const bf16* out, int rows, float* y, const float* resid) const {
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        const auto __w0 = time_utils::now();
#endif
        const size_t n = (size_t)rows * Dout_;
#pragma omp parallel for schedule(static) if (n >= host_par_min())
        for (int r = 0; r < rows; r++) {
            const size_t o = (size_t)r * Dout_;
            if (resid) q38simd::bf16_to_f32_add(y + o, out + o, resid + o, Dout_);
            else       q38simd::bf16_to_f32(y + o, out + o, Dout_);
        }
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        dequant_mm_prof::widen_us +=
            time_utils::duration_us(__w0, time_utils::now()).first;
#endif
    }

private:
    void _check_rows(int rows, const char* who) const {
        if (rows <= 0 || rows > ROWS)
            throw std::runtime_error(
                std::string("dequant_mm_op::") + who + ": " +
                std::to_string(rows) + " rows, the tile is " +
                std::to_string(ROWS));
    }

    void _build_slot(int i) {
        s_[i].x = app_.create_bo_buffer<bf16>((size_t)ROWS * Din_);
        s_[i].y = app_.create_bo_buffer<bf16>((size_t)ROWS * Dout_);
        s_[i].x.memset((bf16)0.0f);
        s_[i].y.memset((bf16)0.0f);
        s_[i].x.sync_to_device();
        s_[i].y.sync_to_device();
        s_[i].live_rows = 0;
    }

    /// Whatever `npu_app::create_run` hands back. Spelled as a `decltype`
    /// because there are two `npu_app`s in npu_utils.hpp behind a build
    /// switch -- the HRX one returns `hrx::run`, the XRT one `xrt::run` -- and
    /// this header is compiled against both (the `.so` takes HRX, the
    /// tools that link the sequence generators take XRT). Both
    /// offer the default ctor, `start()` and an `ert_cmd_state wait()`, which
    /// is all of the type used below.
    using run_t = decltype(std::declval<npu_app&>().create_run(
        std::declval<buffer<bf16>&>(), std::declval<buffer<bf16>&>(),
        std::declval<buffer<u8>&>()));

    /// One pipeline stage: an x BO to fill, a y BO to read, and the run that
    /// ties them. Two of these is what lets host work for one tile sit inside
    /// the device time of another.
    struct slot {
        buffer<bf16> x, y;
        run_t        run;
        /// How many x rows the last tile wrote INTO THIS SLOT, so a shorter one
        /// knows how much of the buffer is stale rather than zero.
        int          live_rows = 0;
    };

    npu_app      app_;
    size_t       Dout_ = 0, Din_ = 0;
    size_t       w_off_ = 0;
    std::vector<dequant_mm_pass> passes_;

    slot         s_[2];
    /// The slot `stage()` writes. Pinned to 0 until `_ensure_pipelined()`.
    int          w_slot_   = 0;
    /// The slot a started-but-unfinished dispatch is writing, or -1.
    int          inflight_ = -1;
    bool         pipelined_ = false;
#ifdef QWEN3_8MTP_PREFILL_PROFILE
    /// Open bracket for the dispatch counter: `dispatch_staged` starts the
    /// clock and `finish` stops it, so the counter measures the device leg and
    /// not the host work the pipeline now hides inside it. It is therefore no
    /// longer the same quantity as before the split -- with the pipeline on,
    /// stage + widen + read no longer sum with it to the wall clock, they
    /// overlap it. That is the whole point, but it means the profile table's
    /// rows stop adding up and the window line is the number to read.
    time_utils::time_point mark_d_{};
#endif
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_PROJ_OP_HPP__
