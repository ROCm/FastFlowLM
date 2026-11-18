// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file verify_qwen3_8mtp.cpp
/// \brief Milestone driver for the Qwen3.8-27B CPU engine.
///
/// Drives qwen3_8mtp_npu through the phases in order and scores each against
/// ref.safetensors -- the transformers dump of one real forward pass. Lives in
/// tools/ so the Makefile's `*.cpp` glob misses it: it has a main() and would
/// otherwise be linked into the shared library.
///
/// Build:
///   g++-13 -O3 -march=native -fopenmp -std=c++20 \
///       -I../../../include -I../../q4nx -I/opt/xilinx/xrt/include \
///       tools/verify_qwen3_8mtp.cpp \
///       -L../../../build/lib -lqwen3_8mtp_npu -lxrt_coreutil \
///       -Wl,-rpath,../../../build/lib -o verify_qwen3_8mtp
/// Run:
///   ./verify_qwen3_8mtp [model_dir] [exec_path]
///
/// The 16 attention layers run on the NPU; the 48 delta-net layers are the CPU
/// reference. The gates do not move: M9 is still an exact 8/8 argmax, so the
/// offload either reproduces the reference decision or it does not.
///
/// What each milestone actually proves, and what it does not:
///
///   M1  embedding      the Q8_0 table and the row indexing
///   M2  m-rope         the table, against the bf16 one the model consumed
///   M7  prefill chain  the 64-layer sweep. Its cosine DECAYS with depth --
///                      64 chained Q4_K layers end near 0.975 and that is the
///                      quantisation floor, not a bug. Do not gate on it.
///   M9  verify argmax  the real gate. 8 positions, exact integer compare,
///                      against a continuation that resumes at position 165
///                      on caches already holding the 221 prefill tokens.
///   M11 next token     prefill's own argmax vs prefill.next_token
///
/// M9 is the one that catches things, because argmax is exact where a cosine
/// is a judgement call.
#include "models/qwen3_8mtp/qwen3_8mtp_npu.hpp"
#include "lm_config.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --------------------------------------------------------------------------
// a minimal safetensors reader, so the driver depends on nothing but the .so
// --------------------------------------------------------------------------
namespace {

inline float bf16_to_f32_(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}

struct ref_tensor {
    std::string          dtype;
    std::vector<int64_t> shape;
    size_t               off0 = 0, off1 = 0;
};

class ref_file {
public:
    explicit ref_file(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        ::fstat(fd_, &st);
        size_ = (size_t)st.st_size;
        base_ = (const uint8_t*)::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap " + path);
        uint64_t hlen; std::memcpy(&hlen, base_, 8);
        payload_ = base_ + 8 + hlen;
        parse((const char*)(base_ + 8), hlen);
    }
    ~ref_file() {
        if (base_ && base_ != MAP_FAILED) ::munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
    }

    bool has(const std::string& n) const { return t_.count(n) != 0; }

    const ref_tensor& info(const std::string& n) const {
        auto it = t_.find(n);
        if (it == t_.end()) throw std::runtime_error("no tensor '" + n + "'");
        return it->second;
    }

    std::vector<float> f32(const std::string& n) const {
        const ref_tensor& t = info(n);
        size_t c = 1; for (int64_t s : t.shape) c *= (size_t)s;
        std::vector<float> out(c);
        const uint8_t* p = payload_ + t.off0;
        if (t.dtype == "F32") std::memcpy(out.data(), p, c * 4);
        else if (t.dtype == "BF16") {
            const uint16_t* h = (const uint16_t*)p;
            for (size_t i = 0; i < c; i++) out[i] = bf16_to_f32_(h[i]);
        }
        else throw std::runtime_error("f32(): dtype " + t.dtype + " for " + n);
        return out;
    }

    std::vector<int32_t> i32(const std::string& n) const {
        const ref_tensor& t = info(n);
        size_t c = 1; for (int64_t s : t.shape) c *= (size_t)s;
        std::vector<int32_t> out(c);
        std::memcpy(out.data(), payload_ + t.off0, c * 4);
        return out;
    }

private:
    void parse(const char* p, size_t n) {
        size_t i = 0;
        auto ws = [&] { while (i < n && (p[i]==' '||p[i]=='\n'||p[i]=='\t'||p[i]=='\r')) i++; };
        auto str = [&]() -> std::string {
            ws(); i++;  // opening quote
            std::string s;
            while (i < n && p[i] != '"') {
                if (p[i] == '\\' && i + 1 < n) { s.push_back(p[i+1]); i += 2; }
                else s.push_back(p[i++]);
            }
            i++;
            return s;
        };
        auto num = [&]() -> int64_t {
            ws(); char* e = nullptr;
            long long v = std::strtoll(p + i, &e, 10);
            i = (size_t)(e - p);
            return v;
        };
        auto skip = [&]() {
            ws();
            if (p[i] == '"') { str(); return; }
            if (p[i] == '{' || p[i] == '[') {
                int d = 0; bool q = false;
                for (; i < n; i++) {
                    char c = p[i];
                    if (q) { if (c == '\\') i++; else if (c == '"') q = false; continue; }
                    if (c == '"') q = true;
                    else if (c == '{' || c == '[') d++;
                    else if (c == '}' || c == ']') { d--; if (!d) { i++; return; } }
                }
                return;
            }
            while (i < n && p[i] != ',' && p[i] != '}' && p[i] != ']') i++;
        };

        ws(); i++;  // '{'
        while (true) {
            ws();
            if (i >= n || p[i] == '}') break;
            std::string key = str();
            ws(); if (p[i] == ':') i++;
            if (key == "__metadata__") { skip(); }
            else {
                ws(); i++;  // '{'
                ref_tensor t;
                while (true) {
                    ws();
                    if (i >= n || p[i] == '}') { i++; break; }
                    std::string f = str();
                    ws(); if (p[i] == ':') i++;
                    if (f == "dtype") t.dtype = str();
                    else if (f == "shape") {
                        ws(); i++;
                        while (true) {
                            ws();
                            if (p[i] == ']') { i++; break; }
                            t.shape.push_back(num());
                            ws(); if (p[i] == ',') i++;
                        }
                    }
                    else if (f == "data_offsets") {
                        ws(); i++;
                        t.off0 = (size_t)num(); ws(); if (p[i] == ',') i++;
                        t.off1 = (size_t)num(); ws(); if (p[i] == ']') i++;
                    }
                    else skip();
                    ws(); if (p[i] == ',') i++;
                }
                t_.emplace(std::move(key), std::move(t));
            }
            ws(); if (i < n && p[i] == ',') i++;
        }
    }

    int                              fd_ = -1;
    size_t                           size_ = 0;
    const uint8_t*                   base_ = nullptr;
    const uint8_t*                   payload_ = nullptr;
    std::map<std::string, ref_tensor> t_;
};

// --------------------------------------------------------------------------
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

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-22s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

void report_cos(const char* tag, double cos, double gate) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "cos = %.8f  (gate %.4f)", cos, gate);
    report(tag, cos >= gate, buf);
}

} // namespace

// ==========================================================================
int main(int argc, char** argv) {
    const std::string model_dir =
        argc > 1 ? argv[1] : "/path/to/models/Qwen3.8-27B-NPU2";

    std::printf("== Qwen3.8-27B CPU engine, milestone driver ==\n");
    std::printf("   model %s\n\n", model_dir.c_str());

    // ---- config ------------------------------------------------------------
    // from_pretrained() lives in the runtime, not in FLM_DLL, so the driver
    // loads config.json itself. Same fields the runtime would populate.
    LM_Config config;
    config.model_path = model_dir;
    config.model_name = "Qwen3.8-27B-NPU2";
    config.exec_path  = ".";
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
        f >> config._json_config;
    }

    ref_file ref(model_dir + "/ref.safetensors");

    // ---- the NPU attention layers ------------------------------------------
    // Not optional: load_weights registers attention_layer.xclbin and builds
    // the 16 attention layers on the device. Running the whole suite over them
    // -- M7, M9, M11, M13 -- is the only check that the offload is right *in
    // the engine*; tools/verify_n2_attn.cpp only shows it is right in isolation.
    //
    // exec_path is the root holding xclbins/<model_name>/, relative to wherever
    // the driver is run from.
    if (argc > 2) config.exec_path = argv[2];
    xrt::device        npu_dev(0);
    npu_xclbin_manager npu_mgr(npu_device::device_npu2, &npu_dev, false);
    std::printf("   xclbins under %s\n", config.exec_path.c_str());

    // ---- M0: construction + weights ---------------------------------------
    std::printf("-- M0: construct and load --\n");
    qwen3_8mtp_npu engine(config, &npu_mgr, 4096);
    {
        // Q4NX takes the model directory, not the .q4nx file -- it appends the
        // filename itself. load_weights mmaps model.q4nx directly anyway; the
        // Q4NX exists to satisfy causal_lm's signature.
        Q4NX q4nx(model_dir);
        engine.load_weights(q4nx);
    }
    report("weights loaded", true, "model.q4nx mapped, 64 layers built");
    report("mtp head", engine.has_mtp_head(),
           engine.has_mtp_head() ? "model.mtp.* present" : "absent");

    // ---- the dumped window -------------------------------------------------
    // 221 tokens, and NOT 221 positions: the window is multimodal, so 64 image
    // slots advance position by only 8 and the run ends at position 165. The
    // ids alone cannot reproduce it -- the image rows come from the vision
    // encoder, which phase 1 does not have -- so the dump's inputs_embeds and
    // position_ids are fed through the payload seam.
    const int P = (int)ref.info("prefill.input_ids").shape[1];
    std::vector<int32_t> pre_ids32 = ref.i32("prefill.input_ids");
    std::vector<int>     pre_ids(pre_ids32.begin(), pre_ids32.end());
    std::vector<int32_t> pre_pos   = ref.i32("prefill.position_ids");
    std::vector<float>   pre_emb   = ref.f32("prefill.inputs_embeds");

    std::printf("\n-- M1: the window is multimodal --\n");
    {
        // text_embeds is the pure table lookup; inputs_embeds is that with the
        // 64 image rows spliced in. They must differ -- if they did not, this
        // prompt would be reproducible from ids alone and the payload seam
        // below would be pointless. Scoring a text-only embedding against
        // inputs_embeds reports ~0.017 and looks like a catastrophic bug when
        // nothing is wrong, so establish which tensor is which up front.
        const std::vector<float> text_ref = ref.f32("prefill.text_embeds");
        const double cos_text_vs_inputs =
            cosine(text_ref.data(), pre_emb.data(), text_ref.size());
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "text_embeds vs inputs_embeds cos = %.6f -- image rows differ",
                      cos_text_vs_inputs);
        report("multimodal window", cos_text_vs_inputs < 0.99, buf);

        // The embedding table itself is scored transitively by M9 below, which
        // feeds ids with no `embeds` override: its 8 tokens are pure text, so
        // every one of them goes through embed_rows.
    }

    // ---- M7 / M11: the 64-layer prefill ------------------------------------
    std::printf("\n-- M7/M11: prefill %d tokens --\n", P);
    qwen3_8mtp_payload_t payload;
    payload.position_ids = pre_pos.data();
    payload.embeds       = pre_emb.data();

    buffer<bf16> logits = engine.prefill(pre_ids, &payload);

    {
        int L = 0, D = 0;
        const float* hid = engine.last_hidden_states(L, D);
        const std::vector<float> ref63 = ref.f32("prefill.layer_63_out");
        const double cos63 = cosine(hid, ref63.data(), (size_t)L * D);
        // Informational, not a gate: 64 chained Q4_K layers land near 0.975
        // and that is the quantisation floor. M11 below is the real check.
        std::printf("  [info] layer_63_out            cos = %.8f "
                    "(decays with depth; argmax is the gate)\n", cos63);

        // The two taps are genuinely different tensors, and the MTP head wants
        // the normed one. Scoring both keeps the distinction honest: the dump
        // ties final_norm_out to mtp.step0.prev_hidden exactly, so a mix-up
        // here would silently degrade every draft.
        const float* nrm = engine.last_normed_states(L, D);
        const std::vector<float> refn = ref.f32("prefill.final_norm_out");
        std::printf("  [info] final_norm_out          cos = %.8f\n",
                    cosine(nrm, refn.data(), (size_t)L * D));
    }

    {
        const int ref_next = ref.i32("prefill.next_token")[0];
        int best = 0;
        for (uint32_t i = 1; i < logits.size(); i++)
            if ((float)logits[i] > (float)logits[best]) best = (int)i;
        char buf[128];
        std::snprintf(buf, sizeof buf, "argmax = %d, ref = %d", best, ref_next);
        report("M11 next_token", best == ref_next, buf);
    }

    {
        char buf[128];
        std::snprintf(buf, sizeof buf, "ctx = %d (expected %d)",
                      engine.get_current_context_length(), P);
        report("context length", engine.get_current_context_length() == P, buf);
    }

    // ---- M9: the verify continuation ---------------------------------------
    // This resumes at position 165 against caches that already hold the 221
    // prefill tokens -- it is a continuation, not a fresh window. Running it
    // on an empty stack scores 3/8 and looks like a layer bug; it is not.
    //
    // It goes through prefill_verify() rather than prefill(), because that is
    // the only entry point that projects all k+1 rows. Acceptance needs every
    // position: position i predicts the token at i+1, and the last row alone
    // says nothing about where a draft first diverged.
    std::printf("\n-- M9: verify continuation (8 tokens) --\n");
    {
        const std::vector<int32_t> vids32 = ref.i32("verify.input_ids");
        std::vector<int>           vids(vids32.begin(), vids32.end());
        const std::vector<int32_t> vpos  = ref.i32("verify.position_ids");
        const std::vector<int32_t> varg  = ref.i32("verify.argmax");

        std::printf("  resuming at position %d, cache holds %d tokens\n",
                    vpos[0], engine.get_current_context_length());

        // verify_ids leads with the token prefill produced, then the 7 drafts.
        std::vector<int> drafts(vids.begin() + 1, vids.end());
        std::vector<int> got;
        const int accepted = engine.prefill_verify(vids, drafts, got);

        int match = 0;
        for (size_t t = 0; t < got.size() && t < varg.size(); t++)
            if (got[t] == varg[t]) match++;

        const int ref_acc = ref.has("verify.accepted_len")
                          ? ref.i32("verify.accepted_len")[0] : -1;
        char buf[192];
        std::snprintf(buf, sizeof buf,
                      "%d/%zu positions exact, accepted = %d (ref %d)",
                      match, varg.size(), accepted, ref_acc);
        report("M9 verify argmax", match == (int)varg.size(), buf);
    }

    // ---- M13: the MTP draft head -------------------------------------------
    // Step 0 primes the head's cache with the whole P-token window and emits
    // one draft; steps 1..6 are chained T=1. The reference's own chained run
    // matches 6 of 7 -- the 7th diverges through accumulated Q4_K error, not
    // a bug -- so this gates on the first draft, which is the one every
    // speculative cycle actually depends on.
    if (engine.has_mtp_head()) {
        std::printf("\n-- M13: MTP draft head --\n");

        // Rebuild the prefill state the head's step 0 was captured against.
        engine.clear_context();
        engine.prefill(pre_ids, &payload);

        int L = 0, D = 0;
        const float* prev = engine.last_normed_states(L, D);

        // Step 0's window is [input_ids[1:], next_token]: the head is shifted
        // one position, so it sees the prompt minus its first token, with the
        // token prefill just produced appended.
        const std::vector<int32_t> s0 = ref.i32("mtp.step0.input_ids");
        const std::vector<int>     ids0(s0.begin(), s0.end());
        const std::vector<int32_t> ref_drafts = ref.i32("mtp.draft_tokens");

        const std::vector<int> got = engine.mtp_draft(ids0, prev, (int)ids0.size(), 7);

        int match = 0;
        std::string line;
        for (size_t i = 0; i < got.size() && i < ref_drafts.size(); i++) {
            if (got[i] == ref_drafts[i]) match++;
            line += std::to_string(got[i]) + "/" + std::to_string(ref_drafts[i]) + " ";
        }
        std::printf("  got/ref: %s\n", line.c_str());

        char buf[160];
        std::snprintf(buf, sizeof buf, "%d/%zu drafts match (first draft is the gate)",
                      match, ref_drafts.size());
        report("M13 mtp draft", !got.empty() && got[0] == ref_drafts[0], buf);
    }

    // ---- M19: multi-cycle speculation --------------------------------------
    // M13 only ever exercises cycle 1, against an EMPTY head cache -- and that
    // is precisely why it cannot see a position-base bug. RoPE is relative, so
    // shifting every position in an all-new window by a constant leaves every
    // q.k dot product unchanged; M13 scores 7/7 with the base off by one.
    // (Measured: reverting mtp_pos_base to cache_len() changes nothing there.)
    //
    // A wrong base only bites once NEW rows must be consistent with EXISTING
    // cached rows, which first happens in cycle 2. So this drives several real
    // cycles and asserts the two cache invariants after each one.
    //
    // The other thing only a second cycle reaches: after a FULL accept the
    // head is two rows behind the base stack, not one, because it drafted k
    // while verify committed k+1. Code that assumes a steady state of one
    // token is correct until the first perfect cycle and misaligned forever
    // after -- with no symptom except acceptance decaying.
    if (engine.has_mtp_head()) {
        std::printf("\n-- M19: multi-cycle speculation --\n");

        engine.clear_context();
        buffer<bf16> y = engine.prefill(pre_ids, &payload);
        int last = 0;
        for (uint32_t i = 1; i < y.size(); i++)
            if ((float)y[i] > (float)y[last]) last = (int)i;

        const int CYCLES = 4;
        bool ok = true;
        int  produced = 0;
        std::string detail;

        for (int c = 0; c < CYCLES && ok; c++) {
            const int ctx_before = engine.get_current_context_length();
            const std::vector<int> acc = engine.speculate(last, 7);
            if (acc.empty()) {
                detail += "cycle " + std::to_string(c) + " declined; ";
                ok = false;
                break;
            }
            produced += (int)acc.size();
            const int ctx_after = engine.get_current_context_length();

            // The base cache trails the committed text by exactly one token,
            // so it must have grown by exactly the number of tokens returned.
            if (ctx_after != ctx_before + (int)acc.size()) {
                detail += "cycle " + std::to_string(c) + ": ctx " +
                          std::to_string(ctx_before) + " -> " +
                          std::to_string(ctx_after) + " for " +
                          std::to_string(acc.size()) + " tokens; ";
                ok = false;
                break;
            }
            // One logits row per returned token, each the row its token is
            // the argmax of: the runtime reports logprobs from these, and an
            // off-by-one would pair every token with its neighbour's numbers.
            int rows = 0, vocab = 0;
            const float* lg = engine.last_speculation_logits(rows, vocab);
            if (!lg || rows != (int)acc.size()) {
                detail += "cycle " + std::to_string(c) + ": " +
                          std::to_string(rows) + " logits rows for " +
                          std::to_string(acc.size()) + " tokens; ";
                ok = false;
                break;
            }
            for (int i = 0; i < rows && ok; i++) {
                const float* row = lg + (size_t)i * vocab;
                for (int v = 0; v < vocab; v++) {
                    if (row[v] > row[acc[i]]) {
                        detail += "cycle " + std::to_string(c) + ": token " +
                                  std::to_string(i) + " is not its row's argmax; ";
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) break;
            detail += std::to_string(acc.size());
            detail += (c + 1 < CYCLES) ? "+" : "";
            last = acc.back();
        }
        // A declined cycle returns no tokens, so it must expose no rows either,
        // not the previous cycle's (one more decline in the counters below)
        if (ok) {
            int rows = -1, vocab = -1;
            (void)engine.speculate(last, 0);  // declines: max_draft <= 0
            if (engine.last_speculation_logits(rows, vocab) != nullptr || rows != 0) {
                detail += "declined cycle still exposes " + std::to_string(rows) + " rows; ";
                ok = false;
            }
        }

        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "%d cycles, %d tokens (%s), mean accepted = %.2f",
                      CYCLES, produced, detail.c_str(),
                      engine.mean_accepted_length());
        report("M19 multi-cycle", ok, buf);

        // The hit rate itself. Printed rather than asserted: it is a quality
        // number, not a correctness one, and it moves with the prompt. A low
        // rate is a performance bug that M17 will still call output-neutral.
        engine.report_speculation_stats();

        // Cross-check the two counters against tokens observed here, so the
        // rate cannot quietly drift from what the loop above actually saw.
        // produced counts committed tokens; each cycle commits accepted + 1.
        const double acc_from_loop = (double)(produced - CYCLES);
        const bool   agrees = engine.speculation_cycles() >= (uint64_t)CYCLES &&
                              std::fabs(engine.mean_accepted_length() * CYCLES -
                                        acc_from_loop) < 1e-6;
        std::snprintf(buf, sizeof buf,
                      "counter %.2f accepted vs %.0f seen by the driver",
                      engine.mean_accepted_length() * CYCLES, acc_from_loop);
        report("M19 stats agree", !ok || agrees, buf);
    }

    // ---- M17: speculation is output-neutral --------------------------------
    // THE gate. Speculation exists to make greedy decoding faster, not
    // different: every rejected draft is replaced by the base model's own
    // argmax, so the token stream must be identical to plain forward() +
    // argmax, token for token.
    //
    // This is the only invariant that actually matters, and it is the one a
    // human review cannot check -- a speculative decoder that drifts still
    // emits fluent text, and the drift is only visible against the
    // non-speculative run of the SAME prompt. A mean-accepted counter cannot
    // see it either: a wrong-but-confident draft is accepted happily.
    //
    // With one exception, which the tie check below states in full: when the
    // top two logits are the same bf16 value, "identical token streams" is not
    // a property the engine can have.
    if (engine.has_mtp_head()) {
        std::printf("\n-- M17: speculation is output-neutral --\n");

        const int N = 16;   // tokens to compare

        // Baseline: no speculation at all, plain greedy decode.
        engine.clear_context();
        buffer<bf16> y0 = engine.prefill(pre_ids, &payload);
        auto argmax_of = [](buffer<bf16>& b) {
            int best = 0;
            for (uint32_t i = 1; i < b.size(); i++)
                if ((float)b[i] > (float)b[best]) best = (int)i;
            return best;
        };
        // Keep the logits each plain token was chosen from. They are the only
        // way to tell a drift from a tie at the first mismatch, and by then the
        // producing buffer is long gone. ~150k bf16 x 16 steps, ~10 MB as
        // fp32; the alternative is re-running the whole decode.
        auto snapshot_of = [](buffer<bf16>& b) {
            std::vector<float> v(b.size());
            for (uint32_t i = 0; i < b.size(); i++) v[i] = (float)b[i];
            return v;
        };
        std::vector<int>                plain;
        std::vector<std::vector<float>> plain_logits;
        int tok = argmax_of(y0);
        std::vector<float> lg = snapshot_of(y0);
        while ((int)plain.size() < N) {
            plain.push_back(tok);
            plain_logits.push_back(std::move(lg));
            buffer<bf16> y = engine.forward(tok);
            lg  = snapshot_of(y);
            tok = argmax_of(y);
        }

        // Speculative: same prompt, same greedy rule, drafts in between.
        engine.clear_context();
        buffer<bf16> y1 = engine.prefill(pre_ids, &payload);
        std::vector<int> spec;
        int last2 = argmax_of(y1);
        while ((int)spec.size() < N) {
            spec.push_back(last2);
            const std::vector<int> acc = engine.speculate(last2, 7);
            if (acc.empty()) {                    // declined: fall back
                buffer<bf16> y = engine.forward(last2);
                last2 = argmax_of(y);
                continue;
            }
            // acc[0..n-2] are further committed tokens; acc.back() becomes the
            // next "last". Mirrors exactly what _shared_generate does.
            for (size_t i = 0; i + 1 < acc.size() && (int)spec.size() < N; i++)
                spec.push_back(acc[i]);
            last2 = acc.back();
        }
        plain.resize(N);
        spec.resize(N);

        int first_diff = -1;
        for (int i = 0; i < N; i++)
            if (plain[i] != spec[i]) { first_diff = i; break; }

        std::string pl, sp;
        for (int i = 0; i < N; i++) {
            pl += std::to_string(plain[i]) + " ";
            sp += std::to_string(spec[i]) + " ";
        }
        std::printf("  plain: %s\n  spec : %s\n", pl.c_str(), sp.c_str());

        // An exact tie is not a divergence. The lm_head writes bf16, 8 mantissa
        // bits, so two tokens can land on the same representable logit -- and
        // then "the argmax" is not a fact about the model, it is a fact about
        // which reduction ran first. Plain decode scans an L=1 row and keeps
        // the lower index; the verify batch reaches the same number down a
        // different path. Neither is wrong, and no amount of fixing the engine
        // will make them agree.
        //
        // So the mismatch is charged against the PLAIN logits: if the two
        // candidates are the same bf16 value there, the test has found a tie
        // and not a drift. The comparison necessarily stops at that point --
        // every later token is conditioned on a different prefix, so a
        // disagreement past it means nothing either way.
        //
        // This is a real weakening, and it is worth being exact about what
        // survives: a drift of even one bf16 ulp is still caught, because a
        // one-ulp gap is not a zero gap. What is no longer caught is a
        // divergence that happens to occur at a step where the top two are bit
        // identical, which is the one case where there was nothing to catch.
        bool  tie    = false;
        float v_plain = 0.0f, v_spec = 0.0f;
        if (first_diff >= 0) {
            const std::vector<float>& lgd = plain_logits[(size_t)first_diff];
            v_plain = lgd[(size_t)plain[first_diff]];
            v_spec  = lgd[(size_t)spec[first_diff]];
            tie     = (v_plain == v_spec);
        }

        char buf[256];
        if (first_diff < 0)
            std::snprintf(buf, sizeof buf, "%d/%d tokens identical", N, N);
        else if (tie)
            std::snprintf(buf, sizeof buf,
                          "%d/%d identical, then an exact bf16 tie at %d: "
                          "%d and %d both %.5f",
                          first_diff, N, first_diff,
                          plain[first_diff], spec[first_diff], (double)v_plain);
        else
            std::snprintf(buf, sizeof buf,
                          "DIVERGED at %d: plain %d (%.5f) vs spec %d (%.5f), "
                          "gap %.6f",
                          first_diff, plain[first_diff], (double)v_plain,
                          spec[first_diff], (double)v_spec,
                          (double)(v_plain - v_spec));
        report("M17 output-neutral", first_diff < 0 || tie, buf);
    }

    // ---- summary -----------------------------------------------------------
    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
