// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_n2_attn.cpp
/// \brief N2 -- one gated full-attention layer on the NPU, scored against the
///        CPU reference, cold and with a real seeded cache.
///
/// Builds a q38npu::attn_engine + npu_attn_layer out of the DLL's OWN code --
/// not the IRON tree's decoding_layer -- so what it exercises is the port, not
/// the thing that was ported.
///
/// Build:
///   source ~/setup.sh
///   g++-13 -O2 -march=native -fopenmp -std=c++20
///       -I. -I../../include -I../q4nx -I/opt/xilinx/xrt/include
///       tools/verify_n2_attn.cpp qwen3_8mtp_npu_attn_sequence.cpp
///       -L/opt/xilinx/xrt/lib -lxrt_coreutil -laiebu
///       -o /tmp/verify_n2_attn
///   (one line; the continuations are dropped here because a backslash at the
///    end of a // comment splices the next line into it)
///
/// ---------------------------------------------------------------------------
/// What is being compared, and why that reference
/// ---------------------------------------------------------------------------
/// The reference is cpu_ref::attention_layer, not the transformers dump. Both
/// sides start from the same model.q4nx, so a disagreement is NPU-side and
/// nothing else; scoring against the dump would fold Q4_K's error in and set
/// the bar where no NPU bug could clear it. The dump supplies only the input
/// rows -- layer 2's output, which is what layer 3 actually consumes.
///
/// Three outputs are scored, because they fail independently:
///
///   h2   the layer output, post-MLP-residual. What the next layer consumes.
///   k    the rotated, normed keys the pass appended. h2 can look right while
///        these are subtly wrong: each query's own row dominates its softmax,
///        so an error in the rotation is diluted by the time it reaches h2.
///   v    the appended values. The only check on the v half of the append BD;
///        k and v ride different channels to different plane offsets.
///
/// ---------------------------------------------------------------------------
/// Why the cold pass is not enough
/// ---------------------------------------------------------------------------
/// At context_offset = 0 the replay descriptor moves one chunk of rows that the
/// append is about to overwrite, so essentially none of the replay addressing is
/// under test -- a layer that never read the cache at all would score the same.
/// Everything interesting lives in the seeded passes.
///
/// And the seed has to be REAL. A cache filled with N(0,1) cannot test
/// attention: random keys are near-orthogonal to the query, the current token's
/// own key is not, and ~99.98% of the softmax mass lands on the appended row.
/// The replayed context is then multiplied by approximately zero and any
/// addressing bug in it scores perfectly. So the seed here is produced by
/// driving the CPU reference over the preceding rows of the same real prompt
/// and mirroring its cache across -- correlated keys, realistic attention mass.
///
/// The BLANKING control is what turns that from an assertion into evidence.
/// One chunk of the seed is zeroed, BOTH implementations are run on that same
/// perturbed cache, and the NPU is required to track the perturbed CPU answer
/// rather than the clean one -- which it can only do by having read the chunk.
/// Chunks past the first are what matters: chunk 0 is reached even by a replay
/// whose chunk axis is broken.
///
/// Attention is local, so the oldest chunk of a long context is worth less than
/// this port's noise floor and no control can say anything about it. Those are
/// reported as untested rather than scored -- see the control itself.
#include "verify/npu/attn_layer.hpp"
#include "verify/npu/attn_weights.hpp"
#include "common/cpu_ref_attention_layer.hpp"
#include "common/cpu_ref_common.hpp"
#include "common/rope.hpp"

#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kLayer = 3;   ///< the first full attention layer: (3+1) % 4 == 0
constexpr int kL     = 8;   ///< L_TOKENS, baked into the bitstream

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-38s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
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

/// Rows [base, base+kL) of the CPU layer's k or v plane, as fp32.
std::vector<float> cpu_rows(const std::vector<float>& plane, int base, int width) {
    const float* p = plane.data() + (size_t)base * width;
    return std::vector<float>(p, p + (size_t)kL * width);
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";
    const std::string xclbin_dir =
        argc > 2 ? argv[2]
                 : "/path/to/FastFlowLM/flm-dev-kit/FLM_Xclbin/Qwen3_8/"
                   "gated_attention_verify/build/QWEN3_8_27B/xclbins";
    const std::string xclbin = xclbin_dir + "/layer.xclbin";

    std::printf("== N2: gated attention layer %d on the NPU, L = %d ==\n", kLayer, kL);
    std::printf("   model  %s\n   xclbin %s\n\n", model_dir.c_str(), xclbin.c_str());

    using namespace cpu_ref::q38;
    const int D    = MODEL_DIM;
    const int DKV_ = DKV;

    // A layer that is not full attention has no q_proj of 2 * DQ rows, and the
    // staging would throw somewhere less legible than here.
    if ((kLayer + 1) % FULL_ATTN_INTERVAL != 0) {
        std::fprintf(stderr, "layer %d is delta-net, not full attention\n", kLayer);
        return 1;
    }

    // 1024, so the grow check below has somewhere to grow to -- it grows to
    // attn_layout::MAX_L_LIMIT, the largest context the two-plane cache can
    // address.
    const q38npu::attn_layout t = q38npu::attn_layout::build(/*max_len=*/1024, kL);
    std::printf("   D=%d DQ=%d DKV=%d I=%d  proj BO %.1f MB  KV BO %.1f MB\n\n",
                t.D, t.DQ, t.DK, t.intermediate_size,
                t.proj_bytes / 1048576.0, t.kv_cache_size * 2 / 1048576.0);

    // ======================================================================
    // 0. the RTP addresses
    // ======================================================================
    // Hard-coded, and per-build. A stale address does not fault: it writes the
    // per-token key counts into some other buffer, the cores never see their L,
    // and the run hangs in the wait -- 60 s, then ERT_CMD_STATE_TIMEOUT. If
    // this suite times out, diff the build's rtp_addresses.txt against these.
    const q38npu::attn_rtp_addresses rtp;
    {
        char buf[80];
        std::snprintf(buf, sizeof buf, "qk = %u, kv = %u", rtp.qk, rtp.kv);
        report("rtp addresses", rtp.qk != 0 && rtp.kv != 0, buf);
    }

    // ======================================================================
    // 1. inputs
    // ======================================================================
    // LC rows of real activations: rows [0, CTX) become the seeded context and
    // rows [CTX, CTX + 8) are the pass under test. CTX is deliberately ragged --
    // the append writes at the exact offset while the replay starts at row 0 and
    // rounds up to a whole L_CHUNK, so an unaligned context is the normal case.
    const int LC_max = 56;
    cpu_ref::safetensors ref(model_dir + "/ref.safetensors");
    char key[64];
    std::snprintf(key, sizeof key, "prefill.layer_%02d_out", kLayer - 1);
    std::vector<float> x_all;
    try {
        x_all = ref.f32(key);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ref.safetensors: %s\n", e.what());
        return 1;
    }
    if ((int)(x_all.size() / D) < LC_max) {
        std::fprintf(stderr, "dump's %s has %zu rows, need %d\n",
                     key, x_all.size() / D, LC_max);
        return 1;
    }
    x_all.resize((size_t)LC_max * D);
    // Round to what the array will actually receive. Letting the CPU reference
    // start from extra precision charges the NPU for a narrowing it cannot
    // avoid, and moves the gate for no reason.
    for (float& f : x_all) f = cpu_ref::round_bf16(f);

    std::vector<float> cos_all, sin_all;
    cpu_ref::text_rope_tables(0, LC_max, cos_all, sin_all);

    // ======================================================================
    // 2. both sides
    // ======================================================================
    cpu_ref::safetensors q4nx(model_dir + "/model.q4nx");
    cpu_ref::attention_layer cpu(q4nx, kLayer, t.MAX_L);

    std::printf("\n-- staging weights --\n");
    std::vector<uint8_t>  proj_host(t.proj_bytes);
    std::vector<uint16_t> rms_host(t.rms_words);
    std::vector<uint16_t> qk_host(2 * (size_t)t.DH);
    q38npu::stage_attn_proj_weights(proj_host.data(), q4nx, t, kLayer);
    q38npu::stage_attn_rms_weights(rms_host.data(), q4nx, t, kLayer);
    q38npu::stage_attn_qk_norm(qk_host.data(), q4nx, t, kLayer);

    // A structural check, not the byte compare against the IRON harness: that
    // harness builds its buffer in process and dumps nothing, so there is no
    // file to diff. What this does confirm is that the five regions tile the BO
    // exactly and that each one's first block is the first block of the tensor
    // it claims -- which catches a swapped or misaligned region, the failure
    // this staging is most likely to have. It does NOT vouch for reorder_cpy
    // itself; the cosine below is what does that.
    {
        const std::string p = "model.layers." + std::to_string(kLayer) + ".";
        struct { const char* name; size_t off; std::string tensor; size_t src_off; } r[] = {
            {"q|k|v", 0,                p + "self_attn.q_proj.weight", 0},
            {"z",     t.w_z_offset,     p + "self_attn.q_proj.weight",
                                        q38npu::q4k_bytes((size_t)t.DQ * t.D)},
            {"o",     t.w_o_offset,     p + "self_attn.o_proj.weight", 0},
            {"up",    t.w_upgate_offset, p + "mlp.up_proj.weight",     0},
            {"down",  t.w_down_offset,  p + "mlp.down_proj.weight",    0},
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
               ok, ok ? (std::to_string(t.proj_bytes) + " B, 5 regions")
                      : ("first block differs at: " + bad));
    }

    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* app_mgr = mgr.register_xclbin(xclbin);
    q38npu::attn_engine eng(app_mgr->create_app(), t, rtp);

    // The constructor loads this layer from q4nx; stage() then overwrites it
    // with the blobs checked above, so the test runs the bytes it inspected.
    q38npu::npu_attn_layer npu(&eng, q4nx, kLayer);
    npu.stage(proj_host, rms_host, qk_host);
    report("engine built", true, "sequence generated, cache zeroed");

    // ======================================================================
    // 3. the passes
    // ======================================================================
    std::vector<float> y_cpu((size_t)kL * D), y_npu((size_t)kL * D);
    std::vector<float> scratch;

    // CTX = 0 exercises no replay at all; 22 is one ragged chunk past the
    // first; 48 walks four. The last is the one that actually strides the
    // replay's chunk axis.
    const int ctxs[] = {0, 22, 48};

    for (int ctx : ctxs) {
        std::printf("\n-- context offset %d (%d rows in cache, %d chunks replayed) --\n",
                    ctx, ctx, (ctx + kL + 15) / 16);

        // Rebuild the seed from scratch each time: drive the CPU reference over
        // rows [0, ctx) of the real prompt, then mirror its cache across. Real
        // keys, correlated with the queries that will read them -- which is the
        // whole point, since random ones would put all the softmax mass on the
        // appended row and make the replay untestable.
        cpu.reset();
        if (ctx > 0) {
            scratch.assign((size_t)ctx * D, 0.0f);
            cpu.forward(x_all.data(), ctx, cos_all.data(), sin_all.data(),
                        scratch.data());
        }
        npu.reset();
        if (ctx > 0) npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);

        const float* xin  = x_all.data()   + (size_t)ctx * D;
        const float* cin  = cos_all.data() + (size_t)ctx * ROPE_DIM;
        const float* sin_ = sin_all.data() + (size_t)ctx * ROPE_DIM;

        cpu.forward(xin, kL, cin, sin_, y_cpu.data());
        npu.forward(xin, kL, cin, sin_, y_npu.data());

        if (npu.tokens_seen() != cpu.cache_len())
            report("cursors agree", false,
                   "npu " + std::to_string(npu.tokens_seen()) + " vs cpu " +
                   std::to_string(cpu.cache_len()));

        char tag[80];
        std::snprintf(tag, sizeof tag, "ctx %2d: h2", ctx);
        const double cos_h2 =
            report_cos(tag, cosine(y_npu.data(), y_cpu.data(), y_npu.size()), 0.999);

        const std::vector<float> k_npu = npu.read_k(ctx);
        const std::vector<float> k_cpu = cpu_rows(cpu.cache_k(), ctx, DKV_);
        std::snprintf(tag, sizeof tag, "ctx %2d: appended k", ctx);
        report_cos(tag, cosine(k_npu.data(), k_cpu.data(), k_npu.size()), 0.999);

        const std::vector<float> v_npu = npu.read_v(ctx);
        const std::vector<float> v_cpu = cpu_rows(cpu.cache_v(), ctx, DKV_);
        std::snprintf(tag, sizeof tag, "ctx %2d: appended v", ctx);
        report_cos(tag, cosine(v_npu.data(), v_cpu.data(), v_npu.size()), 0.999);

        // The seed as it stood for this pass. Taken now because the blanking
        // control below drives the CPU layer itself and overwrites its cache;
        // rows [0, ctx) are the seed either way, the pass only appended above
        // them.
        const std::vector<float> k_clean = cpu.cache_k();
        const std::vector<float> v_clean = cpu.cache_v();

        // ---- the rotation control ------------------------------------------
        // Eight consecutive rows of a hidden state are correlated, so a high
        // cosine on h2 is worth less than it looks until a deliberately wrong
        // alignment is shown to score visibly worse. Rotating the reference by
        // one token is the cheapest such wrong alignment.
        {
            std::vector<float> rot(y_cpu.size());
            for (int r = 0; r < kL; r++)
                std::memcpy(rot.data() + (size_t)r * D,
                            y_cpu.data() + (size_t)((r + 1) % kL) * D,
                            (size_t)D * sizeof(float));
            const double cos_rot = cosine(y_npu.data(), rot.data(), rot.size());
            char buf[160];
            std::snprintf(buf, sizeof buf,
                          "aligned %.6f vs rotated %.6f -- separation %.6f",
                          cos_h2, cos_rot, cos_h2 - cos_rot);
            std::snprintf(tag, sizeof tag, "ctx %2d: h2 rotation control", ctx);
            report(tag, cos_h2 - cos_rot > 0.05, buf);
        }

        // ---- the blanking control, one replay chunk at a time ----------------
        // The high score above says the answer is right. It does not say the
        // replay was READ: the CUs use the k and v they computed this pass
        // directly, so a replay that fetched nothing would still get most of h2
        // right, and a replay whose chunk axis was broken would still get
        // chunk 0.
        //
        // So blank one chunk of the seeded context at a time and watch what
        // happens. Blanking, not poisoning with a large constant: a big key
        // gives a score of +-|q| |k| and lands on whichever side of the softmax
        // the sign picks, so roughly half the heads suppress it entirely and
        // the probe measures nothing. A zeroed key scores exactly 0 -- a
        // middling weight, never suppressed -- and its zeroed value then removes
        // a known share of the attention output.
        //
        // What is asked of the result matters as much as the perturbation.
        // Requiring only that the NPU's score MOVE is a one-sided test, and it
        // fails for an honest reason: attention is local, so blanking the
        // oldest chunk of a 48-row context perturbs the true answer by less
        // than this port's own 5e-6 noise floor. There is nothing to detect,
        // and no gate can tell that apart from a chunk that was skipped.
        //
        // So seed BOTH sides with the same blanked cache and ask two things:
        //
        //   how much the chunk is worth   1 - cos(cpu_blanked, cpu_clean),
        //                                 measured, not assumed
        //   whether the NPU followed      cos(npu_blanked, cpu_blanked)
        //
        // The second is the assertion. It is decisive exactly when the first is
        // large: an NPU that ignored this chunk would return its CLEAN answer,
        // which differs from cpu_blanked by that first number -- so scoring
        // 0.999 against cpu_blanked is only possible if the chunk was read.
        // When the chunk is worth less than the noise floor the comparison
        // proves nothing either way, and the honest report is that, not a pass
        // dressed up as evidence.
        for (int c0 = 0; c0 * 16 < ctx; c0++) {
            const int lo = c0 * 16, hi = std::min(ctx, lo + 16);
            std::vector<float> k_bad = k_clean, v_bad = v_clean;
            std::fill(k_bad.begin() + (size_t)lo * DKV_,
                      k_bad.begin() + (size_t)hi * DKV_, 0.0f);
            std::fill(v_bad.begin() + (size_t)lo * DKV_,
                      v_bad.begin() + (size_t)hi * DKV_, 0.0f);

            std::vector<float> y_cpu_bad((size_t)kL * D), y_npu_bad((size_t)kL * D);
            cpu.reset();
            cpu.seed_cache(k_bad, v_bad, ctx);
            cpu.forward(xin, kL, cin, sin_, y_cpu_bad.data());

            npu.reset();
            npu.mirror_from_cpu(k_bad, v_bad, ctx);
            npu.forward(xin, kL, cin, sin_, y_npu_bad.data());

            // How far blanking moved the true answer, and how far the NPU is
            // from the blanked truth.
            const double worth = 1.0 - cosine(y_cpu_bad.data(), y_cpu.data(),
                                              y_cpu.size());
            const double agree = cosine(y_npu_bad.data(), y_cpu_bad.data(),
                                        y_cpu_bad.size());

            char buf[200];
            std::snprintf(tag, sizeof tag, "ctx %2d: chunk %d is replayed", ctx, c0);
            if (worth < 1e-4) {
                // Below the floor: report it and move on. Counting this as a
                // pass would be claiming evidence that was not collected.
                std::printf("  [ -- ] %-38s rows %2d-%2d carry %.2e of the answer"
                            " -- under the noise floor, not a test\n",
                            tag, lo, hi - 1, worth);
                continue;
            }
            std::snprintf(buf, sizeof buf,
                          "rows %2d-%2d worth %.2e; npu tracks blanked cpu at "
                          "%.6f (clean-answer imposter would score %.6f)",
                          lo, hi - 1, worth, agree, 1.0 - worth);
            report(tag, agree > 0.999 && agree > 1.0 - worth, buf);
        }
    }

    // ======================================================================
    // 4. the state slots
    // ======================================================================
    // The rollback contract, which nothing above touches: a rejected draft is
    // undone by moving the cursor back, and the rows above it are streamed by
    // the replay but must never be read. If they were, run B would differ from
    // run A -- they see identical input from an identical cursor, but B's cache
    // holds A's leftovers above the cursor and A's held zeros.
    std::printf("\n-- state slots: rollback is a cursor --\n");
    {
        const int ctx = 22;
        cpu.reset();
        scratch.assign((size_t)ctx * D, 0.0f);
        cpu.forward(x_all.data(), ctx, cos_all.data(), sin_all.data(), scratch.data());
        npu.reset();
        npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);

        const float* xin  = x_all.data()   + (size_t)ctx * D;
        const float* cin  = cos_all.data() + (size_t)ctx * ROPE_DIM;
        const float* sin_ = sin_all.data() + (size_t)ctx * ROPE_DIM;

        npu.save_state(qwen38::state_slot::speculative);
        std::vector<float> y_a((size_t)kL * D), y_b((size_t)kL * D);
        npu.forward(xin, kL, cin, sin_, y_a.data());
        report("cursor advanced", npu.tokens_seen() == ctx + kL,
               std::to_string(npu.tokens_seen()) + " after one pass");

        npu.restore_state(qwen38::state_slot::speculative);
        report("cursor rolled back", npu.tokens_seen() == ctx,
               std::to_string(npu.tokens_seen()) + " after restore");

        npu.forward(xin, kL, cin, sin_, y_b.data());
        const double cos_ab = cosine(y_a.data(), y_b.data(), y_a.size());
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "re-run after rollback cos = %.8f -- must be 1.0 exactly",
                      cos_ab);
        report("rollback discards the appended rows", cos_ab > 0.99999999, buf);

        // And restoring a slot that was never saved must throw, not silently
        // load a zeroed shadow.
        bool threw = false;
        try { npu.restore_state(qwen38::state_slot::turn); }
        catch (const std::exception&) { threw = true; }
        report("restore before save throws", threw, threw ? "threw" : "returned");
    }

    // ======================================================================
    // 5. growing the cache
    // ======================================================================
    // A reallocation and a copy, not a resize. The check is that the pass it
    // interrupts still reproduces its own answer: if the kept rows landed at
    // the wrong offset, the replayed context would change and h2 with it.
    //
    // With two planes that is a sharper test than it was when the cache was
    // interleaved: growing moves the plane pitch, so the v plane's base moves
    // and a flat prefix copy -- which was correct for the interleaved layout --
    // now lands the whole of v on top of the new k plane's tail. The cosine
    // below is what catches that.
    //
    // It grows to MAX_L_LIMIT deliberately: the largest buffer build() accepts,
    // 16384 here, so the BO is two planes of 16 K rows -- 67 MB for this one
    // layer. Deliberate: the point is to run a real pass with the descriptors
    // that depth produces, and the deepest offsets are the ones a 10-bit dim or
    // a 20-bit stride would silently mask rather than fault on.
    const int kGrowTo = q38npu::attn_layout::MAX_L_LIMIT;
    std::printf("\n-- set_max_length: growing to the replay's cap, %d (%.0f MB) --\n",
                kGrowTo, 2.0 * kGrowTo * DKV_ * sizeof(uint16_t) / 1048576.0);
    {
        const int ctx = 22;
        cpu.reset();
        scratch.assign((size_t)ctx * D, 0.0f);
        cpu.forward(x_all.data(), ctx, cos_all.data(), sin_all.data(), scratch.data());
        npu.reset();
        npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);

        const float* xin  = x_all.data()   + (size_t)ctx * D;
        const float* cin  = cos_all.data() + (size_t)ctx * ROPE_DIM;
        const float* sin_ = sin_all.data() + (size_t)ctx * ROPE_DIM;

        std::vector<float> y_small((size_t)kL * D), y_big((size_t)kL * D);
        npu.save_state(qwen38::state_slot::speculative);
        npu.forward(xin, kL, cin, sin_, y_small.data());
        npu.restore_state(qwen38::state_slot::speculative);   // back to ctx

        const size_t before = eng.t.kv_cache_size;
        npu.set_max_length(kGrowTo);
        const q38npu::attn_layout& grown = eng.t;
        const size_t after = grown.kv_cache_size;
        // Two planes of pitch MAX_L, and nothing else: the cache has no apron,
        // so the BO is exactly `2 * MAX_L * DK` -- qwen3_5vl_npu's
        // `get_kv_cache_size(MAX_L) = MAX_L * (DK + DV)` written for a layout
        // that names its two planes separately. This assertion IS the house
        // rule; if it ever needs a second term again, something grew an apron.
        const size_t want = 2 * (size_t)grown.MAX_L * (size_t)grown.DK;
        report("cache is exactly two planes of MAX_L", after == want,
               std::to_string(before * 2 / 1048576) + " MB -> " +
               std::to_string(after * 2 / 1048576) + " MB (two planes of " +
               std::to_string(grown.MAX_L) + " rows, no apron)");

        npu.forward(xin, kL, cin, sin_, y_big.data());
        const double cos_g = cosine(y_small.data(), y_big.data(), y_big.size());
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "same pass across the grow cos = %.8f -- must be 1.0 exactly",
                      cos_g);
        report("grow preserves both planes", cos_g > 0.99999999, buf);

        // And one row past the cap must THROW, not mask. What is being guarded
        // has moved: the append no longer carries a MAX_L-scaled stride at all
        // (k and v go to two S2MM channels, each walking its own plane at a row
        // stride of DK or DV), so the 20-bit field is out of the picture. What
        // is left is the replay's dim1, a 10-bit chunk count, and an oversized
        // size field masks exactly as quietly as an oversized stride did: the
        // pass would replay `chunks & 0x3FF` chunks of context and score against
        // a truncated history. Hence the throw.
        //
        // The engine's own MAX_L floor is 4096, comfortably under this cap now,
        // so a default-configured engine no longer trips it at load.
        //
        // The boundary is MAX_L_LIMIT itself, and it is a single number now:
        // with no apron and the guards refusing a pass that reaches past the
        // top, the deepest replay ends exactly AT MAX_L, so the cap is simply
        // MAX_L / L_CHUNK <= CHUNKS_LIMIT. The next legal buffer size above it
        // is one PREFILL_CHUNK up, which is what is tested.
        const int kCap  = q38npu::attn_layout::MAX_L_LIMIT;
        const int kPast = kCap + q38npu::attn_cache::PREFILL_CHUNK;
        bool threw = false;
        std::string why;
        try { (void)q38npu::attn_layout::build(kPast, kL); }
        catch (const std::exception& e) { threw = true; why = e.what(); }
        report("max_len past the cap throws", threw,
               threw ? "build(" + std::to_string(kPast) + ") threw"
                     : "build(" + std::to_string(kPast) + ") returned -- "
                       "an overflowing chunk count would be masked, not caught");
        if (threw) std::printf("        %s\n", why.c_str());

        // And the cap itself does not. Together these pin the bound to a
        // buffer size rather than to a neighbourhood.
        bool cap_ok = true;
        try { (void)q38npu::attn_layout::build(kCap, kL); }
        catch (const std::exception&) { cap_ok = false; }
        report("max_len at the cap builds", cap_ok,
               "build(" + std::to_string(kCap) + ") " +
               (cap_ok ? "returned" : "threw -- the cap is one chunk too tight"));

        // **The alignment is the whole apron-free argument, so assert it.** A
        // ragged max_len would leave the replay's round-up to L_CHUNK and the
        // prefill broadcast's to PREFILL_CHUNK landing past the top of the
        // buffer, with nothing under them. build() must refuse rather than
        // build a layout whose descriptors read the next layer's cache.
        bool ragged_threw = false;
        std::string ragged_why;
        try { (void)q38npu::attn_layout::build(4096 + 1, kL); }
        catch (const std::exception& e) { ragged_threw = true; ragged_why = e.what(); }
        report("a ragged max_len throws", ragged_threw,
               ragged_threw ? "build(4097) threw -- the buffer must be a whole "
                              "PREFILL_CHUNK"
                            : "build(4097) returned -- the round-ups would read "
                              "past the end of the BO, silently");
        if (ragged_threw) std::printf("        %s\n", ragged_why.c_str());

        // And the pass at the very top of the buffer is refused, not absorbed.
        // This is the guard that replaced the apron: the append writes
        // L_TOKENS rows from the cursor however few the caller asked for, so a
        // pass starting at MAX_L - 1 would write through MAX_L + L_TOKENS - 2.
        const int kLast = grown.MAX_L - grown.L_TOKENS;
        bool top_ok = true, past_threw = false;
        try { eng.set_ctx(kLast); }        // the deepest legal pass
        catch (const std::exception&) { top_ok = false; }
        try { eng.set_ctx(kLast + 1); }    // one row too deep
        catch (const std::exception&) { past_threw = true; }
        report("the top-of-buffer pass boundary is exact", top_ok && past_threw,
               "set_ctx(" + std::to_string(kLast) + ") " +
               (top_ok ? "ok" : "THREW") + ", set_ctx(" +
               std::to_string(kLast + 1) + ") " +
               (past_threw ? "threw" : "RETURNED -- an 8-row append would run "
                                       "off the end of the plane"));
        eng.set_ctx(ctx);                  // the sequence the rest of §5 wants
    }

    // ======================================================================
    // 6. L != 8: padding and tiling
    // ======================================================================
    // The bitstream dispatches eight query rows and nothing else, but the real
    // engine asks for one (decode) and for a chunk (prefill). forward() serves
    // both by tiling into 8-row dispatches and padding the last one, which is
    // only sound if the padding rows cannot reach the rows that are kept.
    //
    // The decisive test of that needs no reference and no test hook, because a
    // padded pass is just an L = 8 pass whose tail rows the caller ignores: run
    // TWO L = 8 passes that agree on rows [0, n) and disagree completely above
    // them, and require rows [0, n) of the answers to be identical. If the mask
    // leaked, they could not be. This is the property the padding rests on, and
    // it is tested directly rather than inferred.
    //
    // Then the consequences: an L = n pass must equal rows [0, n) of the real
    // L = 8 batch, and eight L = 1 passes -- what decoding actually does -- must
    // reproduce the batch as a whole.
    std::printf("\n-- L != 8: one bitstream, every pass length --\n");
    {
        const int ctx = 22;
        auto seed = [&]() {
            cpu.reset();
            scratch.assign((size_t)ctx * D, 0.0f);
            cpu.forward(x_all.data(), ctx, cos_all.data(), sin_all.data(),
                        scratch.data());
            npu.reset();
            npu.mirror_from_cpu(cpu.cache_k(), cpu.cache_v(), ctx);
        };
        const float* xin  = x_all.data()   + (size_t)ctx * D;
        const float* cin  = cos_all.data() + (size_t)ctx * ROPE_DIM;
        const float* sin_ = sin_all.data() + (size_t)ctx * ROPE_DIM;

        seed();
        std::vector<float> y8((size_t)kL * D);
        npu.forward(xin, kL, cin, sin_, y8.data());

        // ---- the mask control: rows above n cannot reach rows below ----------
        for (int n : {1, 3, 7}) {
            std::vector<float> xa((size_t)kL * D), xb((size_t)kL * D);
            std::memcpy(xa.data(), xin, (size_t)n * D * sizeof(float));
            std::memcpy(xb.data(), xin, (size_t)n * D * sizeof(float));
            // Above n: one copies the last real row, as forward() pads; the
            // other is left zero. Two paddings as unlike as the input allows.
            for (int r = n; r < kL; r++)
                std::memcpy(xa.data() + (size_t)r * D,
                            xa.data() + (size_t)(n - 1) * D, (size_t)D * sizeof(float));

            std::vector<float> ya((size_t)kL * D), yb((size_t)kL * D);
            seed(); npu.forward(xa.data(), kL, cin, sin_, ya.data());
            seed(); npu.forward(xb.data(), kL, cin, sin_, yb.data());

            const double c = cosine(ya.data(), yb.data(), (size_t)n * D);
            char tag[80], buf[200];
            std::snprintf(tag, sizeof tag, "n = %d: rows above n cannot reach it", n);
            std::snprintf(buf, sizeof buf,
                          "two L=8 passes, same rows [0,%d), %d tail rows replaced "
                          "by zeros: cos = %.8f -- must be 1.0 exactly", n, kL - n, c);
            report(tag, c > 0.99999999, buf);
        }

        // ---- forward()'s tiling is exactly that L = 8 pass -------------------
        // The control above varies x above row n. This one varies nothing: it
        // hand-builds the L = 8 pass that forward() runs internally for L = n --
        // x tail and rotary tail both replicating row n - 1 -- and requires the
        // two to agree bit for bit. A difference here would be in the tiling
        // itself, not in the mask.
        for (int n : {1, 3, 7}) {
            std::vector<float> xp((size_t)kL * D);
            std::vector<float> cp((size_t)kL * ROPE_DIM), sp(cp.size());
            std::memcpy(xp.data(), xin, (size_t)n * D * sizeof(float));
            std::memcpy(cp.data(), cin, (size_t)n * ROPE_DIM * sizeof(float));
            std::memcpy(sp.data(), sin_, (size_t)n * ROPE_DIM * sizeof(float));
            for (int r = n; r < kL; r++) {
                std::memcpy(xp.data() + (size_t)r * D,
                            xp.data() + (size_t)(n - 1) * D, (size_t)D * sizeof(float));
                std::memcpy(cp.data() + (size_t)r * ROPE_DIM,
                            cp.data() + (size_t)(n - 1) * ROPE_DIM,
                            (size_t)ROPE_DIM * sizeof(float));
                std::memcpy(sp.data() + (size_t)r * ROPE_DIM,
                            sp.data() + (size_t)(n - 1) * ROPE_DIM,
                            (size_t)ROPE_DIM * sizeof(float));
            }
            std::vector<float> y_hand((size_t)kL * D), y_n((size_t)n * D);
            seed(); npu.forward(xp.data(), kL, cp.data(), sp.data(), y_hand.data());
            seed(); npu.forward(xin, n, cin, sin_, y_n.data());

            const double c = cosine(y_n.data(), y_hand.data(), (size_t)n * D);
            char tag[80], buf[200];
            std::snprintf(tag, sizeof tag, "L = %d: tiling is the padded L=8 pass", n);
            std::snprintf(buf, sizeof buf,
                          "forward(L=%d) vs the hand-padded L=8 it runs: cos = %.8f"
                          " -- must be 1.0 exactly", n, c);
            report(tag, c > 0.99999999, buf);
        }

        // ---- a short pass is the prefix of a full one ------------------------
        // The one comparison here that is NOT bit-exact, and the two controls
        // above localise why: rows above n do not reach rows below them, and the
        // tiling reproduces its own L = 8 pass exactly, so what is left is the
        // rotary tail. A padded pass hands the RoPE tile row n - 1's angle eight
        // times where the real batch hands it eight different ones. It costs
        // 4e-8 of cosine -- four orders of magnitude under the bf16 boundary
        // this layer already pays against the CPU reference.
        for (int n : {1, 3, 7}) {
            seed();
            std::vector<float> y_n((size_t)n * D);
            npu.forward(xin, n, cin, sin_, y_n.data());

            char tag[80], buf[200];
            std::snprintf(tag, sizeof tag, "L = %d: a short pass is a prefix", n);
            const double c = cosine(y_n.data(), y8.data(), y_n.size());
            std::snprintf(buf, sizeof buf,
                          "vs rows [0,%d) of an L=8 pass cos = %.8f -- one dispatch, "
                          "%d padding rows; cursor %d", n, c, kL - n,
                          npu.tokens_seen());
            report(tag, c > 0.9999999 && npu.tokens_seen() == ctx + n, buf);
        }

        // ---- decode reproduces the verify batch ------------------------------
        // Eight passes of one token. Each sees the same context the matching row
        // of the L = 8 pass saw, so every row must come back the same -- the
        // difference being that here the earlier rows are read back out of the
        // cache rather than reached inside the block.
        {
            seed();
            std::vector<float> y_step((size_t)kL * D);
            for (int r = 0; r < kL; r++)
                npu.forward(xin + (size_t)r * D, 1, cin + (size_t)r * ROPE_DIM,
                            sin_ + (size_t)r * ROPE_DIM,
                            y_step.data() + (size_t)r * D);
            // Per row, not aggregated: this is the only check that puts the
            // cache path and the in-block path side by side, so an aggregate
            // cosine here buys a pass by averaging seven good rows over one
            // bad one. Row r reaches r rows, so a leak shows as a trend in r.
            double worst = 2.0;
            int worst_r = -1;
            char rows[400]; int off = 0;
            for (int r = 0; r < kL; r++) {
                const double cr = cosine(y_step.data() + (size_t)r * D,
                                         y8.data() + (size_t)r * D, D);
                if (cr < worst) { worst = cr; worst_r = r; }
                if (off >= (int)sizeof rows) continue;
                off += std::snprintf(rows + off, sizeof rows - off, "%s%d:%.7f",
                                     r ? " " : "", r, cr);
            }
            char buf[600];
            std::snprintf(buf, sizeof buf,
                          "8 x L=1 vs 1 x L=8, per row [%s]; worst row %d at "
                          "%.8f; cursor %d (want %d)",
                          rows, worst_r, worst, npu.tokens_seen(), ctx + kL);
            report("decode reproduces the verify batch",
                   worst > 0.9999999 && npu.tokens_seen() == ctx + kL, buf);
        }

        // ---- a pass longer than the batch tiles ------------------------------
        // 20 rows is 8 + 8 + 4, so it exercises both a whole tile that is not
        // the first and a ragged one, and it moves the cursor across a chunk
        // boundary mid-call.
        {
            const int LP = 20;
            seed();
            std::vector<float> y_cpu_p((size_t)LP * D), y_npu_p((size_t)LP * D);
            cpu.forward(xin, LP, cin, sin_, y_cpu_p.data());
            npu.forward(xin, LP, cin, sin_, y_npu_p.data());

            char buf[200];
            const double c = cosine(y_npu_p.data(), y_cpu_p.data(), y_npu_p.size());
            std::snprintf(buf, sizeof buf,
                          "L=20 (8+8+4) vs the reference cos = %.8f; cursor %d "
                          "(want %d)", c, npu.tokens_seen(), ctx + LP);
            report("a long pass tiles", c > 0.999 && npu.tokens_seen() == ctx + LP,
                   buf);
        }
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
