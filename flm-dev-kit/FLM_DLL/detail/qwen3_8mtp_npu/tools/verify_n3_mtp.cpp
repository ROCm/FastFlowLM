// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n3_mtp.cpp
/// \brief N3 -- the MTP draft head's fused layer on the NPU, scored against the
///        CPU reference, cold and with a real seeded cache.
///
/// Builds a q38npu::npu_mtp_layer out of the DLL's OWN code -- not the IRON
/// tree's decoding_layer -- so what it exercises is the port, not the thing
/// that was ported.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n3_mtp.cpp speculate/npu/mtp_sequence.cpp
///       speculate/mtp_head.cpp
///       -L/opt/xilinx/xrt/lib -lxrt_coreutil -laiebu -luuid -lpthread
///       -o /tmp/verify_n3_mtp
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// ---------------------------------------------------------------------------
/// What is being compared, and why that reference
/// ---------------------------------------------------------------------------
/// The reference is cpu_ref::mtp_head, not the transformers dump. Both sides
/// start from the same model.q4nx, so a disagreement is NPU-side and nothing
/// else; scoring against the dump would fold Q4_K's error in and set the bar
/// where no NPU bug could clear it. The dump supplies only the inputs -- the
/// step-0 window's token ids, the hidden states that predicted them, and their
/// positions.
///
/// The scored tap is `layer_out`, the fused layer's output. That is exactly
/// where MTP_layer.xclbin stops: `rms(layer_out, mtp.norm)` and the shared
/// lm_head stay on the host, so scoring norm_out here would be scoring host
/// code that is common to both sides.
///
/// Three things are checked, because they fail independently:
///
///   h2   the layer output. What speculate_npu norms and projects.
///   k    the rotated, normed key the pass appended. h2 can look right while
///        this is subtly wrong: the token's own key dominates its softmax, so
///        an error in the rotation is diluted by the time it reaches h2.
///   v    the appended value. The only check on the v half of the append BD;
///        k and v ride different channels to different offsets inside the row.
///
/// ---------------------------------------------------------------------------
/// The two controls that turn a high cosine into evidence
/// ---------------------------------------------------------------------------
/// The FUSION control. fc's input is cat(norm(e), norm(h)) -- two D-wide
/// hidden states, of the same shape and roughly the same scale. Feeding them
/// in the wrong order is the single most plausible bug this port can have, and
/// it cannot be ruled out by a number that only says "close to the reference".
/// So the halves are deliberately swapped and the score is required to
/// collapse. (memory: score-swiglu-by-swapping-up-and-gate)
///
/// The BLANKING control. A cold pass at cache length 0 replays a chunk of rows
/// the append is about to overwrite, so essentially none of the replay
/// addressing is under test -- a layer that never read the cache would score
/// the same. So one chunk of a REAL seeded context is zeroed, BOTH sides are
/// run on that same perturbed cache, and the NPU is required to track the
/// perturbed reference rather than the clean one. And the seed must be real:
/// a cache of N(0,1) puts ~99.98% of the softmax mass on the appended row, and
/// any addressing bug in the replay scores perfectly.
/// (memory: random-kv-cache-cannot-test-attention)
#include "speculate/npu/mtp_layer.hpp"
#include "speculate/npu/mtp_weights.hpp"
#include "speculate/mtp_head.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/embedding.hpp"
#include "common/rope.hpp"

#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kWindow = 48;   ///< rows of the step-0 window that are driven
constexpr int kMaxL   = 1024; ///< small, so the grow check has somewhere to go

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-40s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

double cosine(const float* a, const float* b, size_t n) {
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; i++) {
        num += (double)a[i] * b[i];
        da  += (double)a[i] * a[i];
        db  += (double)b[i] * b[i];
    }
    if (da == 0 || db == 0) return 0.0;
    return num / (std::sqrt(da) * std::sqrt(db));
}

double report_cos(const char* tag, double cos, double gate) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f)", cos, gate);
    report(tag, cos >= gate, buf);
    return cos;
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string xclbin_dir =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM_IRON/FLM_Xclbin/Qwen3_8/"
                   "MTP_decoding/build/QWEN3_8_27B/xclbins";
    const std::string xclbin = xclbin_dir + "/MTP_layer.xclbin";

    std::printf("== N3: the MTP draft head's fused layer on the NPU ==\n");
    std::printf("   model  %s\n   xclbin %s\n\n", model_dir.c_str(), xclbin.c_str());

    using namespace cpu_ref::q38;
    const int D    = MODEL_DIM;
    const int DKV_ = DKV;

    const q38npu::mtp_layout t = q38npu::mtp_layout::build(kMaxL);
    std::printf("   D=%d DQ=%d DKV=%d I=%d  proj BO %.1f MB  KV BO %.1f MB\n\n",
                t.D, t.DQ, t.DK, t.intermediate_size,
                t.proj_bytes / 1048576.0, t.kv_cache_size * 2 / 1048576.0);

    // ======================================================================
    // 0. the RTP addresses
    // ======================================================================
    // Hard-coded, and per-build -- and NOT the gated_attention_verify design's,
    // though the kv one happens to coincide. A stale address does not fault: it
    // writes the key count into some other buffer, the cores never see their L,
    // and the run hangs in the wait. If this suite times out, diff the build's
    // rtp_addresses.txt against these. (memory: rtp-addresses-are-per-build)
    const q38npu::mtp_rtp_addresses rtp;
    {
        char buf[80];
        std::snprintf(buf, sizeof buf, "qk = %u, kv = %u", rtp.qk, rtp.kv);
        report("rtp addresses", rtp.qk != 0 && rtp.kv != 0, buf);
    }

    // ======================================================================
    // 1. inputs
    // ======================================================================
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");
    std::vector<int32_t> ids_all, pos_all;
    std::vector<float>   hid_all;
    try {
        ids_all = ref.i32("mtp.step0.input_ids");
        // [3][P], channel-major. NOT all equal -- this dump's row 46 is
        // (46, 46, 47). Take the triple with a stride of P, never contiguously.
        pos_all = ref.i32("mtp.step0.position_ids");
        hid_all = ref.f32("mtp.step0.prev_hidden");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ref.safetensors: %s\n", e.what());
        return 1;
    }
    const int P = (int)ids_all.size();
    if (P < kWindow || (int)(hid_all.size() / D) < kWindow ||
        (int)pos_all.size() < 3 * P) {
        std::fprintf(stderr, "dump's step-0 window is %d rows, need %d\n", P, kWindow);
        return 1;
    }
    // Round to what the array will actually receive. Letting the reference start
    // from extra precision charges the NPU for a narrowing it cannot avoid, and
    // moves the gate for no reason.
    hid_all.resize((size_t)kWindow * D);
    for (float& f : hid_all) f = cpu_ref::round_bf16(f);

    // The head is shifted one position: its rows start at 1, not 0. Take the
    // dump's own positions rather than deriving them -- the off-by-one is the
    // thing most worth not re-deriving. (README section 5.3)
    std::vector<int32_t> pos_thw(3 * (size_t)kWindow);
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < kWindow; r++)
            pos_thw[(size_t)c * kWindow + r] = pos_all[(size_t)c * P + r];
    std::printf("   window %d rows, positions %d..%d\n\n",
                kWindow, pos_thw[0], pos_thw[kWindow - 1]);

    // One row's (t,h,w) triple, packed for a T=1 step().
    //
    // pos_thw is [3][kWindow], so the triple for row r is strided by kWindow,
    // NOT contiguous. Handing step() `pos_thw.data() + r` -- which reads the
    // t-channel positions of rows r, r+1, r+2 -- rotates the h and w sections
    // of q and k by the wrong angle while leaving v untouched, because v is the
    // one projection m-rope never touches. The symptom is k drifting to ~0.997
    // against a reference that is itself the thing that is wrong.
    auto pos_of = [&](int r) {
        return std::array<int32_t, 3>{pos_thw[r], pos_thw[kWindow + r],
                                      pos_thw[2 * kWindow + r]};
    };

    // npu_mtp_layer::forward takes a single scalar and expands it through
    // cpu_ref::text_rope_tables -- (p, p, p). It is not an oversight that it
    // cannot take a triple: the draft interface is text-only on BOTH sides.
    // speculate_npu.hpp counts `pos + r` off pos0 and speculate_cpu.hpp calls
    // cpu_ref::text_position_ids, so neither speculator ever reads the
    // channels that qwen3_8mtp_npu::get_input_idx assembles for the base
    // stack. The scored NPU-vs-CPU-head comparison is therefore apples to
    // apples whatever this dump holds.
    //
    // What still has to hold is that the *dump's* channels stay close enough
    // that driving its rows through a scalar interface is the computation the
    // reference performs. This window's tail is a vision seam: t stalls at 46
    // while w keeps counting, so rows 45-47 carry a spread of up to two. That
    // is scored below, against the window's own error distribution, rather
    // than gated on the spread itself -- how much a spread of two costs is
    // exactly the kind of number not to guess.
    int rope_spread = 0, rope_spread_row = 0;
    for (int r = 0; r < kWindow; r++) {
        const int s = std::max(std::abs(pos_thw[kWindow + r] - pos_thw[r]),
                               std::abs(pos_thw[2 * kWindow + r] - pos_thw[r]));
        if (s > rope_spread) { rope_spread = s; rope_spread_row = r; }
    }
    std::printf("   widest m-rope spread: row %d is (t,h,w) = (%d,%d,%d)\n\n",
                rope_spread_row, pos_thw[rope_spread_row],
                pos_thw[kWindow + rope_spread_row],
                pos_thw[2 * kWindow + rope_spread_row]);

    // ======================================================================
    // 2. both sides
    // ======================================================================
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::embedding_table embed(q4nx);
    cpu_ref::mtp_head cpu(q4nx, &embed, kMaxL);

    std::printf("-- staging weights --\n");
    std::vector<uint8_t>  proj_host(t.proj_bytes);
    std::vector<uint16_t> rms_host(t.rms_words);
    std::vector<uint16_t> qk_host(2 * (size_t)t.DH);
    q38npu::stage_mtp_proj_weights(proj_host.data(), q4nx, t);
    q38npu::stage_mtp_rms_weights(rms_host.data(), q4nx, t);
    q38npu::stage_mtp_qk_norm(qk_host.data(), q4nx, t);

    // A structural check: that the regions tile the BO exactly and that each
    // one's first block is the first block of the tensor it claims. That
    // catches a swapped or misaligned region, which is the failure this staging
    // is most likely to have -- and the k|v|q order here is INVERTED relative
    // to the attention layer's q|k|v, which is exactly the kind of thing a
    // structural check is for. It does NOT vouch for reorder_cpy itself; the
    // cosines below are what do that. (memory: rope-tile-consumes-kv-before-q)
    {
        const std::string p = "model.mtp.layers.0.";
        struct { const char* name; size_t off; std::string tensor; size_t src_off; } r[] = {
            {"fc",   t.w_fc_offset,     "model.mtp.fc.weight",         0},
            {"k",    t.w_qkv_offset,    p + "self_attn.k_proj.weight", 0},
            {"v",    t.w_qkv_offset + q38npu::q4k_bytes((size_t)t.DK * t.D),
                                        p + "self_attn.v_proj.weight", 0},
            {"q",    t.w_qkv_offset + q38npu::q4k_bytes((size_t)(t.DK + t.DV) * t.D),
                                        p + "self_attn.q_proj.weight", 0},
            {"z",    t.w_gate_offset,   p + "self_attn.q_proj.weight",
                                        q38npu::q4k_bytes((size_t)t.DQ * t.D)},
            {"o",    t.w_o_offset,      p + "self_attn.o_proj.weight", 0},
            {"up",   t.w_upgate_offset, p + "mlp.up_proj.weight",      0},
            {"down", t.w_down_offset,   p + "mlp.down_proj.weight",    0},
        };
        bool ok = t.proj_bytes ==
            t.w_down_offset + q38npu::q4k_bytes((size_t)t.D * t.intermediate_size);
        std::string bad;
        for (const auto& e : r) {
            // reorder_cpy emits source block 0 first, so the staged region's
            // first block must be the tensor's first block verbatim.
            if (std::memcmp(proj_host.data() + e.off,
                            q4nx.raw(e.tensor) + e.src_off,
                            q38npu::W_BLOCK_BYTES) != 0) {
                ok = false;
                bad += std::string(bad.empty() ? "" : ", ") + e.name;
            }
        }
        report("proj BO regions tile and align",
               ok, ok ? (std::to_string(t.proj_bytes) + " B, 8 regions")
                      : ("first block differs at: " + bad));
    }

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* app_mgr = mgr.register_xclbin(xclbin);

    // The constructor loads from q4nx; stage() then overwrites with the blobs
    // checked above, so the test runs the bytes it inspected.
    q38npu::npu_mtp_layer npu(app_mgr->create_app(), t, q4nx, rtp);
    npu.stage(proj_host, rms_host, qk_host);
    report("layer built", true, "sequence generated, cache zeroed");


    // Drive `n` rows of the window through the NPU, one dispatch each, starting
    // from whatever cursor the layer is on. Returns [n][D] of h2.
    std::vector<float> e_row((size_t)D);
    auto npu_run = [&](int base, int n, bool swap_halves) {
        std::vector<float> out((size_t)n * D);
        for (int r = 0; r < n; r++) {
            const int32_t id = ids_all[(size_t)base + r];
            const float*  h  = hid_all.data() + (size_t)(base + r) * D;
            embed.embed_row(id, e_row.data());
            if (swap_halves) npu.forward(h, e_row.data(), pos_thw[base + r],
                                         out.data() + (size_t)r * D);
            else             npu.forward(e_row.data(), h, pos_thw[base + r],
                                         out.data() + (size_t)r * D);
        }
        return out;
    };

    // The repeated-dispatch probe, off by default and returning before the
    // graded sections. Kept because it is how the second-dispatch timeout was
    // cornered (speculate/npu/SECOND_DISPATCH_TIMEOUT.md, now fixed) and it is
    // still the cheapest way to re-open that class of failure.
    //
    // The word-for-word sequence diff against the scaffold that used to sit
    // above this -- FLM_N3_DUMPCTX, paired with the engine's FLM_MTP_SEQDUMP --
    // is gone with its other half: the port matched, the design was repaired,
    // and the dump had no business shipping inside the .so.
    //
    //   FLM_N3_PROBE=n    dispatch n times from FLM_N3_BASE, advancing the
    //                     cursor unless FLM_N3_HOLD is set. FLM_N3_SAME pins
    //                     the input row too, so HOLD+SAME repeats a literally
    //                     identical dispatch; FLM_N3_NOISE seeds the cache
    //                     from an LCG instead of zeros.
    if (const char* p = getenv("FLM_N3_PROBE")) {
        const char* b = getenv("FLM_N3_BASE");
        const int base = b ? std::atoi(b) : 0;
        const bool hold = getenv("FLM_N3_HOLD") != nullptr;
        if (base > 0) {
            // set_max_length rounds to PREFILL_CHUNK itself; rounding here too
            // just keeps the probe's intent (room for base + the append row)
            // legible.
            npu.set_max_length((base + 1 + q38npu::mtp_layout::PREFILL_CHUNK - 1) /
                               q38npu::mtp_layout::PREFILL_CHUNK *
                               q38npu::mtp_layout::PREFILL_CHUNK);
            std::vector<float> z((size_t)base * DKV_, 0.0f);
            if (getenv("FLM_N3_NOISE")) {
                uint32_t s = 12345u;
                for (auto& f : z) {
                    s = s * 1664525u + 1013904223u;
                    f = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
                }
            }
            npu.mirror_from_cpu(z, z, base);
        }
        const bool same = getenv("FLM_N3_SAME") != nullptr;
        std::printf("   probing at MAX_L %d, x %zu w, proj %zu B, rms %zu w, "
                    "rope %zu w, cache %zu w\n",
                    npu.layout().MAX_L, npu.layout().x_words,
                    npu.layout().proj_bytes, npu.layout().rms_words,
                    npu.layout().rope_words, npu.layout().kv_cache_size);
        for (int i = 0; i < std::atoi(p); i++) {
            std::printf("   probe dispatch %d at cursor %d%s ... ",
                        i + 1, npu.cache_len(), hold ? ", held" : "");
            std::fflush(stdout);
            const std::vector<float> y = npu_run(same ? 0 : i % kWindow, 1, false);
            std::printf("ok, y[0] = %f\n", y[0]);
            if (hold) npu.crop(base);
        }
        return 0;
    }

    // ======================================================================
    // 3. the cold window
    // ======================================================================
    // The wide CPU step and kWindow single-row dispatches are supposed to be
    // the same computation: each dispatch appends its own k|v before the next
    // one replays, so row r attends to rows 0..r just as the block-causal mask
    // would have. That equivalence is the whole reason a T=1 bitstream can
    // serve the width-T priming step, so it is scored per row -- an aggregate
    // cosine would buy a pass by averaging 47 good rows over one bad one.
    std::printf("\n-- the step-0 window, cold (%d rows) --\n", kWindow);
    cpu_ref::mtp_taps tp;
    cpu.reset();
    cpu.step(ids_all.data(), pos_thw.data(), hid_all.data(), kWindow, tp);

    npu.reset();
    const std::vector<float> y_npu = npu_run(0, kWindow, false);

    report("cursors agree", npu.cache_len() == cpu.cache_len(),
           "npu " + std::to_string(npu.cache_len()) + " vs cpu " +
           std::to_string(cpu.cache_len()));

    std::vector<double> cos_r((size_t)kWindow);
    for (int r = 0; r < kWindow; r++)
        cos_r[r] = cosine(y_npu.data() + (size_t)r * D,
                          tp.layer_out.data() + (size_t)r * D, D);
    // A single bad row in 48 is either a real defect at one cursor or a row
    // whose own magnitude makes cosine fragile, and the two look identical
    // from the minimum alone. Print the shape and the reference norm so the
    // distinction is visible without a rebuild.
    if (getenv("FLM_N3_ROWS")) {
        std::vector<double> s = cos_r;
        std::sort(s.begin(), s.end());
        std::printf("   per-row cos: min %.6f  p25 %.6f  med %.6f  max %.6f\n",
                    s.front(), s[(size_t)kWindow / 4], s[(size_t)kWindow / 2],
                    s.back());
        for (int r = 0; r < kWindow; r++) {
            double n2 = 0.0, d2 = 0.0;
            const float* a = y_npu.data() + (size_t)r * D;
            const float* b = tp.layer_out.data() + (size_t)r * D;
            for (int i = 0; i < D; i++) {
                n2 += (double)b[i] * b[i];
                d2 += (double)(a[i] - b[i]) * (a[i] - b[i]);
            }
            std::printf("   row %2d  cos %.6f  |ref| %9.4f  |err| %9.4f  "
                        "rel %.5f  pos %d\n", r, cos_r[r], std::sqrt(n2),
                        std::sqrt(d2), std::sqrt(d2 / (n2 + 1e-30)),
                        pos_thw[r]);
        }
    }
    // Scored on RELATIVE ERROR, not on the per-row cosine.
    //
    // The error this layer leaves is additive and flat: |h2_npu - h2_cpu| sits
    // at 2.5-7 on every row of the window regardless of that row's magnitude,
    // because it comes from bf16 accumulation against an fp32 reference on the
    // same Q4_K bytes. Cosine divides that by |ref|, so it reports the row's
    // own norm as if it were accuracy -- the three worst-scoring rows here
    // (19, 22, 26) are precisely the three smallest-norm rows, while row 41
    // carries the single largest error in the window and scores 0.99984 on a
    // norm of 435. A 0.999 per-row cosine gate therefore fails correct rows
    // and would pass a genuinely wrong one that happened to be large.
    //
    // What the per-row scoring is actually for is catching ONE cursor computed
    // wrong while the other 47 carry the aggregate. Such a row is wrong by
    // order its own size -- rel ~ 1 -- so it is separated from the floor by
    // more than a decimal order, and the imposter below measures that distance
    // rather than assuming it. FLM_N3_ROWS=1 prints the whole distribution.
    std::vector<double> rel((size_t)kWindow);
    {
        double worst_rel = 0.0; int worst_rel_r = -1;
        for (int r = 0; r < kWindow; r++) {
            double n2 = 0.0, d2 = 0.0;
            const float* a = y_npu.data() + (size_t)r * D;
            const float* b = tp.layer_out.data() + (size_t)r * D;
            for (int i = 0; i < D; i++) {
                n2 += (double)b[i] * b[i];
                d2 += (double)(a[i] - b[i]) * (a[i] - b[i]);
            }
            rel[r] = std::sqrt(d2 / (n2 + 1e-30));
            if (rel[r] > worst_rel) { worst_rel = rel[r]; worst_rel_r = r; }
        }
        // The best score a wrong-but-plausible row could post: each NPU row
        // against its NEIGHBOUR's reference. Nothing real may come near it.
        double imposter = 1e30;
        for (int r = 0; r + 1 < kWindow; r++) {
            double n2 = 0.0, d2 = 0.0;
            const float* a = y_npu.data() + (size_t)r * D;
            const float* b = tp.layer_out.data() + (size_t)(r + 1) * D;
            for (int i = 0; i < D; i++) {
                n2 += (double)b[i] * b[i];
                d2 += (double)(a[i] - b[i]) * (a[i] - b[i]);
            }
            imposter = std::min(imposter, std::sqrt(d2 / (n2 + 1e-30)));
        }
        std::vector<double> s = rel;
        std::sort(s.begin(), s.end());
        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "worst rel err is row %d at %.4f, median %.4f; the best "
                      "misaligned row would post %.4f (gate 0.15)",
                      worst_rel_r, worst_rel, s[(size_t)kWindow / 2], imposter);
        report("h2 per row", worst_rel < 0.15 && imposter > 4.0 * worst_rel, buf);
    }

    // And now the m-rope seam, scored rather than assumed. forward() flattens
    // (t,h,w) to (t,t,t), so rows 45-47 are the only rows where the port is
    // solving a slightly different problem than the reference. If that cost
    // anything, they would be the window's worst rows. They are not -- the
    // worst row is a text row whose norm is simply small -- which is what
    // licenses driving this dump through a scalar interface.
    //
    // A real image span spreads the channels by hundreds, not by two, and
    // would land here as these rows dominating. Neither speculator can express
    // that today (speculate_npu.hpp counts pos + r; speculate_cpu.hpp calls
    // text_position_ids), so this test is the tripwire for the day one of them
    // learns to.
    {
        double div_worst = 0.0, txt_worst = 0.0; int div_r = -1, n_div = 0;
        for (int r = 0; r < kWindow; r++) {
            const bool diverges = pos_thw[kWindow + r] != pos_thw[r] ||
                                  pos_thw[2 * kWindow + r] != pos_thw[r];
            if (diverges) {
                n_div++;
                if (rel[r] > div_worst) { div_worst = rel[r]; div_r = r; }
            } else if (rel[r] > txt_worst) txt_worst = rel[r];
        }
        char buf[220];
        std::snprintf(buf, sizeof buf,
                      "%d of %d rows diverge (spread %d); worst of them is row "
                      "%d at rel %.4f, against %.4f for the worst text row",
                      n_div, kWindow, rope_spread, div_r, div_worst, txt_worst);
        report("the m-rope seam costs nothing measurable",
               n_div == 0 || div_worst <= txt_worst, buf);
    }
    const double cos_h2 = cosine(y_npu.data(), tp.layer_out.data(), y_npu.size());
    report_cos("h2 over the window", cos_h2, 0.999);

    // ---- the appended k and v ---------------------------------------------
    // Row 9's pair is kept for the ceiling section: k and v are functions of the
    // token, the position and the weights alone, so the same row driven at a
    // depth of 16352 must reproduce these *exactly*. Comparing the deep rows
    // against these rather than against the CPU takes Q4_K out of the
    // comparison -- both sides are then the same arithmetic on the same bytes,
    // and any gap left is addressing.
    // (memory: score-dequant-mm-against-its-own-quantized-bytes)
    std::vector<float> k_shallow_9, v_shallow_9;
    {
        double wk = 2.0, wv = 2.0;
        for (int r = 0; r < kWindow; r++) {
            const std::vector<float> kn = npu.read_k(r), vn = npu.read_v(r);
            if (r == 9) { k_shallow_9 = kn; v_shallow_9 = vn; }
            wk = std::min(wk, cosine(kn.data(), cpu.cache_k().data() + (size_t)r * DKV_,
                                     (size_t)DKV_));
            wv = std::min(wv, cosine(vn.data(), cpu.cache_v().data() + (size_t)r * DKV_,
                                     (size_t)DKV_));
        }
        char buf[120];
        std::snprintf(buf, sizeof buf, "worst row cos = %.8f (gate 0.9990)", wk);
        report("appended k", wk >= 0.999, buf);
        std::snprintf(buf, sizeof buf, "worst row cos = %.8f (gate 0.9990)", wv);
        report("appended v", wv >= 0.999, buf);
    }

    // ---- the rotation control ----------------------------------------------
    // Consecutive rows of a hidden state are correlated, so a high cosine is
    // worth less than it looks until a deliberately wrong alignment is shown to
    // score visibly worse.
    {
        std::vector<float> rot(y_npu.size());
        for (int r = 0; r < kWindow; r++)
            std::memcpy(rot.data() + (size_t)r * D,
                        tp.layer_out.data() + (size_t)((r + 1) % kWindow) * D,
                        (size_t)D * sizeof(float));
        const double c = cosine(y_npu.data(), rot.data(), rot.size());
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "aligned %.6f vs rotated %.6f -- separation %.6f",
                      cos_h2, c, cos_h2 - c);
        report("h2 rotation control", cos_h2 - c > 0.05, buf);
    }

    // ---- the fusion control -------------------------------------------------
    // The two halves of fc's input are both D-wide hidden states. Swapping them
    // is the most plausible wrong answer this port has, and nothing above can
    // exclude it -- so run it and require the score to collapse.
    {
        npu.reset();
        const std::vector<float> y_swap = npu_run(0, kWindow, true);
        const double c = cosine(y_swap.data(), tp.layer_out.data(), y_swap.size());
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "e|h %.6f vs h|e %.6f -- separation %.6f; the passthrough "
                      "tile's two norms are not interchangeable",
                      cos_h2, c, cos_h2 - c);
        report("fusion half-order control", cos_h2 - c > 0.05, buf);
    }

    // ======================================================================
    // 4. the seeded pass and the blanking control
    // ======================================================================
    // At cache length 0 the replay moves one chunk of rows the append is about
    // to overwrite, so none of the replay addressing was under test above.
    // Everything interesting lives here.
    std::printf("\n-- seeded context: is the replay actually read? --\n");
    {
        // CTX is deliberately ragged: the append writes at the exact offset
        // while the replay starts at row 0 and rounds up to a whole L_CHUNK, so
        // an unaligned context is the normal case.
        const int ctx = 37;
        const int row = ctx;                       // the pass under test

        cpu.reset();
        cpu_ref::mtp_taps warm;
        cpu.step(ids_all.data(), pos_thw.data(), hid_all.data(), ctx, warm);
        const std::vector<float> k_clean = cpu.cache_k();
        const std::vector<float> v_clean = cpu.cache_v();

        auto run_both = [&](const std::vector<float>& k, const std::vector<float>& v,
                            std::vector<float>& yc, std::vector<float>& yn) {
            cpu_ref::mtp_taps one;
            cpu.reset();
            cpu.seed_cache(k, v, ctx);
            const std::array<int32_t, 3> pr = pos_of(row);
            cpu.step(ids_all.data() + row, pr.data(),
                     hid_all.data() + (size_t)row * D, 1, one);
            yc = one.layer_out;

            npu.reset();
            npu.mirror_from_cpu(k, v, ctx);
            yn = npu_run(row, 1, false);
        };

        std::vector<float> y_cpu_c, y_npu_c;
        run_both(k_clean, v_clean, y_cpu_c, y_npu_c);
        report_cos("seeded pass: h2",
                   cosine(y_npu_c.data(), y_cpu_c.data(), (size_t)D), 0.999);

        // One chunk at a time. Blanking, not poisoning with a large constant: a
        // big key scores +-|q||k| and lands on whichever side of the softmax the
        // sign picks, so roughly half the heads suppress it entirely and the
        // probe measures nothing. A zeroed key scores exactly 0 -- a middling
        // weight, never suppressed -- and its zeroed value then removes a known
        // share of the attention output.
        //
        // Both sides get the same blanked cache, and the assertion is that the
        // NPU tracks the BLANKED reference. That is decisive exactly when the
        // chunk is worth more than the noise floor: an NPU that ignored the
        // chunk would return its clean answer, which differs from the blanked
        // truth by that amount. When it is worth less, the comparison proves
        // nothing either way, and the honest report is that rather than a pass
        // dressed up as evidence.
        constexpr int CH = q38npu::mtp_layout::L_CHUNK;
        for (int c0 = 0; c0 * CH < ctx; c0++) {
            const int lo = c0 * CH, hi = std::min(ctx, lo + CH);
            std::vector<float> k_bad = k_clean, v_bad = v_clean;
            std::fill(k_bad.begin() + (size_t)lo * DKV_,
                      k_bad.begin() + (size_t)hi * DKV_, 0.0f);
            std::fill(v_bad.begin() + (size_t)lo * DKV_,
                      v_bad.begin() + (size_t)hi * DKV_, 0.0f);

            std::vector<float> y_cpu_b, y_npu_b;
            run_both(k_bad, v_bad, y_cpu_b, y_npu_b);

            const double worth = 1.0 - cosine(y_cpu_b.data(), y_cpu_c.data(), (size_t)D);
            const double agree = cosine(y_npu_b.data(), y_cpu_b.data(), (size_t)D);

            char tag[80], buf[220];
            std::snprintf(tag, sizeof tag, "chunk %d is replayed", c0);
            if (worth < 1e-4) {
                std::printf("  [ -- ] %-40s rows %2d-%2d carry %.2e of the answer"
                            " -- under the noise floor, not a test\n",
                            tag, lo, hi - 1, worth);
                continue;
            }
            std::snprintf(buf, sizeof buf,
                          "rows %2d-%2d worth %.2e; npu tracks blanked cpu at %.6f "
                          "(clean-answer imposter would score %.6f)",
                          lo, hi - 1, worth, agree, 1.0 - worth);
            report(tag, agree > 0.999 && agree > 1.0 - worth, buf);
        }
    }

    // ======================================================================
    // 5. the rotary BO is consumed
    // ======================================================================
    // Nothing above can separate "RoPE applied correctly" from "RoPE applied
    // with the wrong angle", because both sides were handed the same position.
    // RoPE is relative, so shifting EVERY position by a constant changes
    // nothing -- the probe has to shift one row against a cache built at the
    // real positions. (README section 5.3 is the same trap, one level up.)
    std::printf("\n-- the rotary pair reaches the RoPE tile --\n");
    {
        const int ctx = 16, row = ctx;
        cpu.reset();
        cpu_ref::mtp_taps warm;
        cpu.step(ids_all.data(), pos_thw.data(), hid_all.data(), ctx, warm);

        npu.reset();
        npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);
        const std::vector<float> y_right = npu_run(row, 1, false);

        npu.reset();
        npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);
        embed.embed_row(ids_all[(size_t)row], e_row.data());
        std::vector<float> y_wrong((size_t)D);
        npu.forward(e_row.data(), hid_all.data() + (size_t)row * D,
                    pos_thw[row] + 64, y_wrong.data());   // a real, wrong angle

        const double c = cosine(y_right.data(), y_wrong.data(), (size_t)D);

        // How much *should* a 64-position shift move h2? Less than instinct
        // says: ROPE_DIM is 64 of DH 256, so only a quarter of each head
        // rotates, and the shift then has to survive a softmax over ctx + 1
        // keys and the whole MLP. Gating on a guessed constant marked a
        // correct run as a failure at cos 0.9994. The reference answers it.
        cpu.crop(ctx);
        cpu_ref::mtp_taps t_right;
        const std::array<int32_t, 3> pr = pos_of(row);
        cpu.step(ids_all.data() + row, pr.data(),
                 hid_all.data() + (size_t)row * D, 1, t_right);
        int32_t pos_wrong[3] = {pr[0] + 64, pr[1] + 64, pr[2] + 64};
        cpu.crop(ctx);
        cpu_ref::mtp_taps t_wrong;
        cpu.step(ids_all.data() + row, pos_wrong,
                 hid_all.data() + (size_t)row * D, 1, t_wrong);
        const double c_cpu = cosine(t_right.layer_out.data(),
                                    t_wrong.layer_out.data(), (size_t)D);

        // If the pair never arrived the two NPU passes are the SAME dispatch
        // twice and cos is 1.0 to the bit. The real test is that the NPU moves
        // by what the reference moves by.
        const bool ok = c < 0.9999 && c < c_cpu + 5e-4;
        char buf[240];
        std::snprintf(buf, sizeof buf,
                      "same row at position %d vs %d: npu cos = %.6f, cpu cos = "
                      "%.6f -- the npu must move by what the reference moves by",
                      pos_thw[row], pos_thw[row] + 64, c, c_cpu);
        report("a wrong position changes h2", ok, buf);
    }

    // ======================================================================
    // 6. the cursor, and growing the cache
    // ======================================================================
    // crop() is how a rejected draft is undone, and the contract is that it is
    // a cursor move and nothing else: rows above it are streamed by the replay
    // but must never be read. If they were, run B would differ from run A --
    // identical input from an identical cursor, but B's cache holds A's
    // leftovers above the cursor where A's held the warm-up's.
    std::printf("\n-- crop is a cursor; grow keeps the cache --\n");
    {
        const int ctx = 22, row = ctx;
        cpu.reset();
        cpu_ref::mtp_taps warm;
        cpu.step(ids_all.data(), pos_thw.data(), hid_all.data(), ctx, warm);
        npu.reset();
        npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);

        const std::vector<float> y_a = npu_run(row, 1, false);
        report("cursor advanced", npu.cache_len() == ctx + 1,
               std::to_string(npu.cache_len()) + " after one dispatch");

        npu.crop(ctx);
        report("cursor rolled back", npu.cache_len() == ctx,
               std::to_string(npu.cache_len()) + " after crop");

        const std::vector<float> y_b = npu_run(row, 1, false);
        const double c = cosine(y_a.data(), y_b.data(), (size_t)D);
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "re-run after crop cos = %.8f -- must be 1.0 exactly", c);
        report("crop discards the appended row", c > 0.99999999, buf);

        // A reallocation and a copy, not a resize. The check is that the pass
        // it interrupts still reproduces its own answer: if the kept rows
        // landed at the wrong offset, the replayed context would change and h2
        // with it.
        //
        // A sharper check under two planes than it was under the interleave: a
        // grow moves the v plane's BASE, so a carry-over that copied the old BO
        // flat would leave every v row inside the new k plane. It grows to 4096,
        // the engine's MAX_L floor and past the 2048 a single-channel append
        // used to top out at. Watch stderr: an overflowing stride does not
        // throw, it prints "stride_2 out of range" and masks.
        npu.crop(ctx);
        const size_t before = npu.layout().kv_cache_size;
        npu.set_max_length(4096);
        const q38npu::mtp_layout& grown = npu.layout();
        // Two planes of pitch MAX_L, and nothing else: the cache has no apron,
        // so the BO is exactly `2 * MAX_L * DK` -- the scaffold's shape, and
        // qwen3_5vl_npu's `get_kv_cache_size(MAX_L) = MAX_L * (DK + DV)`.
        const size_t want = 2 * (size_t)grown.MAX_L * (size_t)grown.DK;
        report("cache is exactly two planes of MAX_L", grown.kv_cache_size == want,
               std::to_string(before * 2 / 1048576) + " MB -> " +
               std::to_string(grown.kv_cache_size * 2 / 1048576) + " MB (two "
               "planes of " + std::to_string(grown.MAX_L) + " rows, no apron)");

        const std::vector<float> y_c = npu_run(row, 1, false);
        const double cg = cosine(y_a.data(), y_c.data(), (size_t)D);
        std::snprintf(buf, sizeof buf,
                      "same pass across the grow cos = %.8f -- must be 1.0 exactly", cg);
        report("grow preserves the cache", cg > 0.99999999, buf);
    }

    // ======================================================================
    // 7. the ceiling
    // ======================================================================
    // Everything above runs at a cursor of a few dozen. The descriptor fields
    // that the two-plane split was for only get exercised at depth: the replay
    // is {1, chunks, L_CHUNK, width} and `chunks` is a 10-bit dimension, so
    // MAX_L_LIMIT rows is exactly 1024 chunks -- one short of a field that does
    // not fault when it overflows, it masks. A run at a shallow cursor inside a
    // deep layout would not notice, because it posts a small chunk count; the
    // cursor has to actually be up there.
    //
    // The CPU head grows with it, so h2 is scored against a real reference and
    // not just against itself. The context is the warm window tiled, with a
    // few-percent per-row scale so the rows are distinguishable -- real keys,
    // correlated with the query that reads them.
    // (memory: random-kv-cache-cannot-test-attention, planar-kv-cache-masks-stride-past-2048)
    const int kCeil = q38npu::mtp_layout::MAX_L_LIMIT;
    std::printf("\n-- the cache at its ceiling (%d rows) --\n", kCeil);
    {
        // The rows to tile, taken from a warm window before the CPU cache is
        // reseeded out from under it.
        cpu.reset();
        cpu_ref::mtp_taps warm;
        cpu.step(ids_all.data(), pos_thw.data(), hid_all.data(), kWindow, warm);
        const std::vector<float> src_k = cpu.cache_k(), src_v = cpu.cache_v();

        // Two whole chunks stay above the cursor: there is no apron, so the
        // append row and the replay's round-up have to fit inside MAX_L.
        const int ctx = kCeil - q38npu::mtp_layout::L_CHUNK * 2;
        std::vector<float> dk((size_t)ctx * DKV_), dv((size_t)ctx * DKV_);
        for (int r = 0; r < ctx; r++) {
            const size_t s = (size_t)(r % kWindow) * DKV_, d = (size_t)r * DKV_;
            const float g = 1.0f + 0.02f * (float)((r % 37) - 18) / 18.0f;
            for (int i = 0; i < DKV_; i++) dk[d + i] = src_k[s + i] * g;
            for (int i = 0; i < DKV_; i++) dv[d + i] = src_v[s + i] * g;
        }

        npu.crop(0);
        npu.set_max_length(kCeil);
        const q38npu::mtp_layout& deep = npu.layout();
        report("grew to the ceiling",
               deep.MAX_L == kCeil &&
               deep.kv_cache_size == 2 * (size_t)deep.MAX_L * (size_t)deep.DK,
               std::to_string(deep.MAX_L) + " rows, " +
               std::to_string(deep.kv_cache_size * 2 / 1048576) + " MB, " +
               std::to_string(deep.MAX_L / q38npu::mtp_layout::L_CHUNK) +
               " replay chunks");

        cpu.set_max_length(kCeil);
        cpu.seed_cache(dk, dv, ctx);
        npu.mirror_from_cpu(dk, dv, ctx);
        report("seeded at depth", npu.cache_len() == ctx && cpu.cache_len() == ctx,
               "cursor " + std::to_string(npu.cache_len()) + " of " +
               std::to_string(kCeil));

        const int row = 9;
        cpu_ref::mtp_taps deep_tp;
        const std::array<int32_t, 3> pr = pos_of(row);
        cpu.step(ids_all.data() + row, pr.data(),
                 hid_all.data() + (size_t)row * D, 1, deep_tp);
        const std::vector<float> y_d = npu_run(row, 1, false);

        report_cos("h2 at the ceiling",
                   cosine(y_d.data(), deep_tp.layer_out.data(), (size_t)D), 0.999);

        // k and v depend on the token, its position and the weights -- never on
        // the context -- so this is an exact reference for the append ADDRESS
        // at a row the shallow tests never reach.
        const std::vector<float> k_deep = npu.read_k(ctx), v_deep = npu.read_v(ctx);
        report_cos("appended k at the ceiling",
                   cosine(k_deep.data(),
                          cpu.cache_k().data() + (size_t)ctx * DKV_, (size_t)DKV_),
                   0.999);
        report_cos("appended v at the ceiling",
                   cosine(v_deep.data(),
                          cpu.cache_v().data() + (size_t)ctx * DKV_, (size_t)DKV_),
                   0.999);

        // Same token, same position, same weights as row 9 of the cold window,
        // so these are the *same* NPU computation landing at a different
        // address. Anything short of equality is the depth leaking into the
        // value; a Q4_K floor cannot show up here, because both sides carry it.
        {
            double mk = 0.0, mv = 0.0;
            for (int i = 0; i < DKV_; i++) {
                mk = std::max(mk, (double)std::fabs(k_deep[i] - k_shallow_9[i]));
                mv = std::max(mv, (double)std::fabs(v_deep[i] - v_shallow_9[i]));
            }
            char buf[160];
            std::snprintf(buf, sizeof buf,
                          "max|dk| = %.3e, max|dv| = %.3e vs the same row at "
                          "cursor 9", mk, mv);
            report("the deep pair reproduces the shallow one", mk == 0.0 && mv == 0.0,
                   buf);
        }

        // And the next legal buffer size past the cap must THROW rather than
        // mask: a descriptor built from more than 1024 chunks is accepted by
        // npu_instr_utils with a printed warning and a wrapped field, which is
        // a wrong answer, not a failure. One PREFILL_CHUNK up, not one L_CHUNK:
        // the cache has no apron and build() requires the coarser alignment,
        // so a +16 would throw for the alignment and never reach the cap.
        const int kPast = kCeil + q38npu::mtp_layout::PREFILL_CHUNK;
        bool threw = false;
        try { (void)q38npu::mtp_layout::build(kPast); }
        catch (const std::exception&) { threw = true; }
        report("past the cap throws", threw,
               std::to_string(kPast) + " rows is " +
               std::to_string(kPast / q38npu::mtp_layout::L_CHUNK) +
               " chunks in a 10-bit field");

        // **The alignment is the whole apron-free argument, so assert it.** A
        // ragged MAX_L would leave the replay's round-up to L_CHUNK and the
        // prefill broadcast's to PREFILL_CHUNK landing past the top of the
        // buffer, with nothing under them -- and a read past a BO does not
        // fault here.
        bool ragged_threw = false;
        std::string ragged_why;
        try { (void)q38npu::mtp_layout::build(4096 + 1); }
        catch (const std::exception& e) { ragged_threw = true; ragged_why = e.what(); }
        report("a ragged max_len throws", ragged_threw,
               ragged_threw ? "build(4097) threw -- the buffer must be a whole "
                              "PREFILL_CHUNK"
                            : "build(4097) returned -- the round-ups would read "
                              "past the end of the BO, silently");
        if (ragged_threw) std::printf("        %s\n", ragged_why.c_str());
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
