// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_prefill.cpp
/// \brief inkling2_prefill: the prompt matmuls on the NPU
#include "inkling2_prefill.hpp"

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <set>
#include <stdexcept>

#include "inkling2_npu_attn.hpp"
#include "inkling2_npu_mm.hpp"

using namespace inkling2_host;
namespace mm = inkling2_npu_mm;
namespace bfp = inkling2_bfp16;

namespace {

constexpr uint32_t DIN = D;           // 2304
constexpr uint32_t DIN_O = QD;        // 3072
constexpr uint32_t DIN_DOWN = FF_PAD;   // 1280
constexpr uint32_t NPAD_D = 2304;     // D outputs, a multiple of mm::PASS_ROWS
constexpr int GLU_BLOCKS = FF / QROWS;   // gate (and up) row blocks per expert: 36

uint32_t pad768(uint32_t n) { return (n + mm::PASS_ROWS - 1) / mm::PASS_ROWS * mm::PASS_ROWS; }
/// \brief rows of storage of a matrix of n rows: whole block groups, so the next matrix starts on a group
constexpr int group_rows(int n) { return (n + NPU_GROUP * QROWS - 1) / (NPU_GROUP * QROWS) * NPU_GROUP * QROWS; }
// the attention rows k v q gate rel in storage: k and v (32 blocks each) padded to whole groups
constexpr int ATT_K = 0, ATT_V = group_rows(KVD), ATT_Q = 2 * group_rows(KVD), ATT_G = ATT_Q + group_rows(QD),
              ATT_R = ATT_G + group_rows(QD);
constexpr int ATT_N = ATT_R + group_rows(NH * NREL);   // 8640

/// \brief fp32 -> bf16, round to nearest even
void to_bf16(u16* dst, const float* src, int n) {
    int i = 0;
    const __m512i one = _mm512_set1_epi32(1), bias = _mm512_set1_epi32(0x7FFF);
    for (; i + 16 <= n; i += 16) {
        __m512i v = _mm512_castps_si512(_mm512_loadu_ps(src + i));
        v = _mm512_add_epi32(v, _mm512_add_epi32(_mm512_and_si512(_mm512_srli_epi32(v, 16), one), bias));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm512_cvtepi32_epi16(_mm512_srli_epi32(v, 16)));
    }
    for (; i < n; i++) {
        uint32_t u;
        memcpy(&u, src + i, 4);
        dst[i] = (u16)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
}

void from_bf16(float* dst, const u16* src, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512i v = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i))), 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(v));
    }
    for (; i < n; i++) {
        const uint32_t u = (uint32_t)src[i] << 16;
        memcpy(dst + i, &u, 4);
    }
}

using clk = std::chrono::steady_clock;
double since(clk::time_point& t0) {
    const clk::time_point t1 = clk::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    t0 = t1;
    return s;
}

u16* raw(buffer<bf16>& b) { return reinterpret_cast<u16*>(b.data()); }
u8* bytes_of(buffer<bf16>& b) { return reinterpret_cast<u8*>(b.data()); }

/// \brief rows [0, T) of src[T][K] as the bfp16 x of a dispatch of covered rows at m rows per core row (Din columns,
///        columns [K, Din) and rows [T, covered) zero)
void pack(u8* dst, uint32_t Din, const float* src, int K, int T, const mm::geometry& g) {
    std::vector<const float*> rows(g.covered, nullptr);
    for (int t = 0; t < T; t++) rows[t] = src + (size_t)t * K;
    #pragma omp parallel for schedule(static)
    for (int R = 0; R < (int)g.covered; R += 16)
        bfp::pack_group(dst + bfp::bytes((size_t)R / g.m * g.m, Din), g.m, (R % g.m) / 16, rows.data() + R, K, Din);
}

/// \brief matrix w (row-block-major, in the mapped file) into the grouped layout at dst, an incomplete last group
///        zero-filled; returns the bytes written. glu > 0: stacked gate / up rows, stored as qmat::glu describes
size_t group_copy(u8* dst, const qmat& w, int glu = 0) {
    const int nc = w.col_blocks(), rb = w.row_blocks();
    const int groups = (rb + NPU_GROUP - 1) / NPU_GROUP;
    auto logical = [glu](int rb) {
        if (!glu) return rb;
        const int e = rb / (2 * glu), j = rb % (2 * glu);
        return e * 2 * glu + (j % 2 ? glu : 0) + j / 2;
    };
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < groups; p++)
        for (int cb = 0; cb < nc; cb++)
            for (int i = 0; i < NPU_GROUP; i++) {
                u8* d = dst + (((size_t)p * nc + cb) * NPU_GROUP + i) * Q4K_BYTES;
                if (NPU_GROUP * p + i < rb) q4k_to_npu(d, w.block(logical(NPU_GROUP * p + i), cb));
                else memset(d, 0, Q4K_BYTES);
            }
    return (size_t)groups * nc * NPU_GROUP * Q4K_BYTES;
}

}  // namespace

inkling2_prefill::inkling2_prefill(npu_xclbin_manager* npu, const std::string& xclbin)
    : prof(std::getenv("INKLING2_PROF") != nullptr), npu(npu), mgr(npu->register_xclbin(xclbin)),
      app_gu(), app_dn() {
    const char* g = std::getenv("INKLING2_EXP_GROUPS");
    const int G = std::clamp(g ? std::atoi(g) : 5, 1, 16);
    for (int i = 0; i < G; i++) {
        this->app_gu.push_back(this->mgr->create_app());
        this->app_dn.push_back(this->mgr->create_app());
    }
}

inkling2_prefill::~inkling2_prefill() { this->report(); }

void inkling2_prefill::report() const {
    if (!this->prof) return;
    const char* names[19] = {"attn in", "attn out", "expert pack", "gate_up start", "situ+down start", "gate_up wait", "down wait", "unpack", "down build wait", "experts total", "attention pack", "attention run", "attention unpack", "proj pack", "proj run", "proj unpack", "gu npu (serial)", "dn npu (serial)", "idle work in experts"};
    for (int i = 0; i < 18; i++) std::fprintf(stderr, "inkling2 npu prefill %-16s %8.3f s\n", names[i], this->t[i]);
    std::fprintf(stderr, "inkling2 npu prefill %ld expert calls, %ld runs\n", this->calls, this->runs);
    if (this->dec)
        std::fprintf(stderr, "inkling2 npu decode experts: %ld dispatches, start %.3f s, wait %.3f s\n", this->dec->calls,
                     this->dec->t_start, this->dec->t_wait);
    const char* kinds[5] = {"attn_in 1x", "attn_in 3x", "attn_out", "attention swa", "attention global"};
    for (int i = 0; i < 5; i++) std::fprintf(stderr, "inkling2 npu kind %-16s %8.3f s %ld calls\n", kinds[i], this->npu_t[i], this->npu_n[i]);
    const char* skinds[7] = {"attn_in 1x", "attn_in 3x", "attn_out", "attention swa", "attention global", "expert gate_up", "expert down"};
    for (int i = 0; i < 7; i++) std::fprintf(stderr, "inkling2 npu submit %-16s %8.4f s %ld runs\n", skinds[i], this->npu_s[i], this->npu_sn[i]);
}

void inkling2_prefill::set_decode(const std::string& xclbin, int n) {
    if (n <= 0 || !std::filesystem::exists(xclbin)) return;
    this->dec = std::make_unique<inkling2_decode>(this->npu, xclbin);
    this->dec_n = std::min(n, inkling2_decode::MAX_N);
}

void inkling2_prefill::dec_start(int il, int kind, const int* ex, int n, const float* in) {
    layer_buf& B = this->layers[il];
    this->dec->start(kind, B.w, kind == 0 ? B.gate_up : B.down, ex, n, in);
}

size_t inkling2_prefill::expert_rows(int T) {
    // sized for any routing of T tokens, so the buffers are allocated once per prompt length, not each time
    // a layer's padded rows set a new maximum (a reallocation zero-fills tens of MB): pick_geometry pads an
    // expert by less than 128 rows, or 32 per block of up to 256 rows
    return (size_t)T * TOPK + (size_t)T * TOPK / 8 + (size_t)NE * 192;
}

void inkling2_prefill::prewarm(int T) {
    namespace at = inkling2_attn;
    // a step of T tokens at position 0, either lane taking all of them: projections (one input or three), o,
    // attention (keys 0 .. T - 1), and the experts. Later global steps still grow the attention K/V buffer.
    const uint32_t cov = mm::pick_geometry(T, DIN).covered, cov_o = mm::pick_geometry(T, DIN_O).covered;
    const uint32_t ycols = std::max(pad768(ATT_N), 2 * pad768(KVD) + pad768(ATT_N - ATT_Q));
    const size_t Le = (size_t)(T + at::ROUND - 1) / at::ROUND * at::ROUND;
    for (lane_buf& Ln : this->lanes) {
        this->reserve(Ln.ay, std::max((size_t)cov * ycols, (size_t)cov_o * NPAD_D));
        this->reserve(Ln.ax[0], std::max(bfp::bytes(cov, DIN), bfp::bytes(cov_o, DIN_O)) / 2);
        for (int i = 1; i < 3; i++) this->reserve(Ln.ax[i], bfp::bytes(cov, DIN) / 2);
        this->reserve(Ln.att_o, Le * QD);
        this->reserve(Ln.att_q, at::q_pack_size((int)(Le / at::ROUND)));
        this->reserve(Ln.att_kv, 2 * Le * NKV * at::KV_ROW);
        this->reserve(Ln.att_tab, at::TAB_SIZE);
    }
    const size_t cap = expert_rows(T);
    this->reserve(this->ex, bfp::bytes(cap, DIN) / 2);
    this->reserve(this->ea, bfp::bytes(cap, DIN_DOWN) / 2);
    this->reserve(this->ey, cap * NPAD_D);
    this->reserve(this->egu, cap * NPAD_D);
}

void inkling2_prefill::reserve(buffer<bf16>& b, size_t n) {
    if (b.size() >= n) return;
    std::lock_guard<std::mutex> lock(this->alloc_mu);
    b = this->npu->create_bo_buffer<bf16>(n);
}

namespace {
thread_local int lane_of_thread = 0;
}

void inkling2_prefill::set_lane(int lane) { lane_of_thread = lane; }

/// DIAG: INKLING2_HALF_TRACE=<layer>: an NPU call of that layer (asked, got the NPU, done; steady clock ns)
static void trace_npu(int il, const char* kind, clk::time_point req, clk::time_point got) {
    static const int v = [] { const char* e = std::getenv("INKLING2_HALF_TRACE"); return e ? std::atoi(e) : -1; }();
    if (il != v && v != -2) return;   // -2: every layer
    auto ns = [](clk::time_point t) { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count(); };
    std::fprintf(stderr, "NT %d %s %d %lld %lld %lld\n", il, kind, lane_of_thread, ns(req), ns(got), ns(clk::now()));
}



inkling2_prefill::lane_buf& inkling2_prefill::lane() { return this->lanes[lane_of_thread]; }

npu_app& inkling2_prefill::app(uint32_t rows, uint32_t Din, uint32_t Kw, uint32_t Npad) {
    const mm::geometry g = mm::pick_geometry(rows, Din);
    const auto key = std::make_tuple(g.blocks, g.m, Din, Kw, Npad);
    auto it = this->apps.find(key);
    if (it == this->apps.end()) {
        it = this->apps.emplace(key, this->mgr->create_app()).first;
        mm::gen(it->second.seq(), rows, Din, Kw, Npad);
    }
    return it->second;
}

npu_app& inkling2_prefill::att_app(int Lb, int Le, int window, int kv) {
    const auto key = std::make_tuple(Lb, Le, window, kv);
    auto it = this->att_apps.find(key);
    if (it == this->att_apps.end()) {
        if (this->att_apps.size() >= 64) this->att_apps.clear();   // prompts of many lengths (a server)
        it = this->att_apps.emplace(key, this->att_mgr->create_app()).first;
        inkling2_attn::gen_seq(it->second.seq(), Lb, Le, Le, window, kv);
    }
    return it->second;
}

void inkling2_prefill::load(weights& w) {
    // INKLING2_SHARED=0: decode keeps reading the host copy (more memory)
    const char* sh = std::getenv("INKLING2_SHARED");
    const bool shared = !(sh && sh[0] == '0');
    this->layers.resize(NL);
    for (int il = 0; il < NL; il++) {
        layer_weights& L = w.layer[il];
        for (const qmat* m : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.rel, &L.gate_up, &L.down})
            if (m->fmt != qfmt::q4k || m->grouped) throw std::runtime_error("inkling2: NPU prefill needs Q4_K projections");
        layer_buf& B = this->layers[il];
        const size_t att = mm::q4k_bytes((size_t)ATT_N * D);
        B.o = att;
        B.gate_up = B.o + mm::q4k_bytes((size_t)D * QD);
        B.down = B.gate_up + mm::q4k_bytes((size_t)NE * 2 * FF * D);
        const size_t end = B.down + mm::q4k_bytes((size_t)NE * D * FF_PAD);
        B.w = this->npu->create_bo_buffer<u8>(end + mm::tail_bytes(D, NPAD_D, DIN_DOWN, FF_PAD));
        u8* base = B.w.data();
        memset(base + end, 0, B.w.size() - end);
        size_t at = 0;
        // the attention rows in dispatch order k v q gate rel; each matrix keeps its own (grouped) view
        for (qmat* m : {&L.k, &L.v, &L.q, &L.gate, &L.rel}) {
            const size_t n = group_copy(base + at, *m);
            if (shared) {
                m->data = base + at;
                m->grouped = true;
            }
            at += n;
        }
        assert(at == att);
        group_copy(base + B.o, L.o);
        // gate block j beside up block j: a core's pair is the gate and up rows of 32 channels, activated in place
        group_copy(base + B.gate_up, L.gate_up, GLU_BLOCKS);
        group_copy(base + B.down, L.down);
        if (shared) {
            L.o.data = base + B.o;
            L.gate_up.data = base + B.gate_up;
            L.down.data = base + B.down;
            L.o.grouped = L.gate_up.grouped = L.down.grouped = true;
            L.gate_up.glu = GLU_BLOCKS;
        }
        B.w.sync_to_device();
    }
}

bool inkling2_prefill::attn_in(int il, const float* xq, const float* xk, const float* xv, int T, float* Q, float* K, float* V,
                               float* R, float* G, bf16_rows* view) {
    if (T < this->min_T) return false;
    clk::time_point t0 = clk::now();
    layer_buf& B = this->layers[il];
    const uint32_t cov = mm::pick_geometry(T, DIN).covered;
    const bool one = xk == xq && xv == xq;
    // one input: all 8640 storage rows in one dispatch; three: k, v, then q gate rel
    struct part {
        const float* x;
        uint32_t row0, n;
    };
    std::vector<part> parts;
    if (one) parts = {{xq, 0, (uint32_t)ATT_N}};
    else parts = {{xk, ATT_K, (uint32_t)KVD}, {xv, ATT_V, (uint32_t)KVD}, {xq, ATT_Q, (uint32_t)(ATT_N - ATT_Q)}};
    uint32_t ycols = 0;
    for (auto& p : parts) ycols += pad768(p.n);
    lane_buf& Ln = this->lane();
    this->reserve(Ln.ay, (size_t)cov * ycols);
    clk::time_point ts = clk::now();
    for (size_t i = 0; i < parts.size(); i++) {
        this->reserve(Ln.ax[i], bfp::bytes(cov, DIN) / 2);
        pack(bytes_of(Ln.ax[i]), DIN, parts[i].x, D, T, mm::pick_geometry(T, DIN));
        Ln.ax[i].sync_range_to_device_exact(0, bfp::bytes(cov, DIN));
    }
    this->t[13] += since(ts);
    {
        const clk::time_point treq = clk::now();   // DIAG
        std::lock_guard<std::mutex> lock(this->npu_mu);
        clk::time_point tn = clk::now();
        struct ntrace {   // DIAG: INKLING2_HALF_TRACE
            int il; const char* kind; clk::time_point a, b;
            ~ntrace() { trace_npu(il, kind, a, b); }
        } ntr{il, "in", treq, tn};
        // the dispatches queue back to back (no host round trip between them)
        uint32_t y0 = 0;
        std::vector<flm_rt::run> runs;
        for (size_t i = 0; i < parts.size(); i++) {
            const uint32_t np = pad768(parts[i].n);
            clk::time_point tsub = clk::now();
            npu_app& a = this->app(T, DIN, D, np);
            runs.push_back(a.create_run(bound_arg{&Ln.ay, (size_t)cov * y0 * sizeof(u16)}, Ln.ax[i],
                                        bound_arg{&B.w, mm::q4k_bytes((size_t)parts[i].row0 * D)}));
            runs.back().start();
            this->npu_s[one ? 0 : 1] += since(tsub);
            this->npu_sn[one ? 0 : 1]++;
            y0 += np;
        }
        for (flm_rt::run& r : runs) r.wait();
        this->npu_t[one ? 0 : 1] += since(tn);
        this->npu_n[one ? 0 : 1]++;
    }
    Ln.ay.sync_range_from_device_exact(0, (size_t)cov * ycols * sizeof(u16));
    this->t[14] += since(ts);
    // y of part i: [cov][pad768(n_i)] at cov * (columns before it)
    struct dst {
        int row0;   // attention row
        float* out;
        int n;
    };
    const dst outs[5] = {{ATT_K, K, KVD}, {ATT_V, V, KVD}, {ATT_Q, Q, QD}, {ATT_G, G, QD}, {ATT_R, R, NH * NREL}};
    if (view) {
        // Q K V R G where the NPU wrote them
        const int order[5] = {2, 0, 1, 4, 3};
        for (int i = 0; i < 5; i++) {
            const dst& o = outs[order[i]];
            uint32_t base = 0;
            for (const auto& p : parts) {
                const uint32_t np = pad768(p.n);
                if (o.row0 >= (int)p.row0 && o.row0 < (int)(p.row0 + p.n))
                    view[i] = {raw(Ln.ay) + (size_t)cov * base + (o.row0 - p.row0), np};
                base += np;
            }
        }
        this->t[15] += since(ts);
        this->t[0] += since(t0);
        return true;
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        uint32_t base = 0;
        for (const auto& p : parts) {
            const uint32_t np = pad768(p.n);
            const u16* yrow = raw(Ln.ay) + (size_t)cov * base + (size_t)t * np;
            for (const auto& o : outs)
                if (o.row0 >= (int)p.row0 && o.row0 < (int)(p.row0 + p.n))
                    from_bf16(o.out + (size_t)t * o.n, yrow + (o.row0 - p.row0), o.n);
            base += np;
        }
    }
    this->t[15] += since(ts);
    this->t[0] += since(t0);
    return true;
}

bool inkling2_prefill::attn_out(int il, const float* O, int T, float* y, bf16_rows* view) {
    if (T < this->min_T) return false;
    clk::time_point t0 = clk::now();
    layer_buf& B = this->layers[il];
    const uint32_t cov = mm::pick_geometry(T, DIN_O).covered;
    lane_buf& Ln = this->lane();
    this->reserve(Ln.ax[0], bfp::bytes(cov, DIN_O) / 2);
    this->reserve(Ln.ay, (size_t)cov * NPAD_D);
    clk::time_point ts = clk::now();
    pack(bytes_of(Ln.ax[0]), DIN_O, O, QD, T, mm::pick_geometry(T, DIN_O));
    Ln.ax[0].sync_range_to_device_exact(0, bfp::bytes(cov, DIN_O));
    this->t[13] += since(ts);
    {
        const clk::time_point treq = clk::now();   // DIAG
        std::lock_guard<std::mutex> lock(this->npu_mu);
        clk::time_point tn = clk::now();
        struct ntrace {   // DIAG: INKLING2_HALF_TRACE
            int il; const char* kind; clk::time_point a, b;
            ~ntrace() { trace_npu(il, kind, a, b); }
        } ntr{il, "out", treq, tn};
        npu_app& a = this->app(T, DIN_O, QD, NPAD_D);
        flm_rt::run r = a.create_run(Ln.ay, Ln.ax[0], bound_arg{&B.w, B.o});
        r.start();
        this->npu_s[2] += since(tn);
        this->npu_sn[2]++;
        r.wait();
        this->npu_t[2] += since(tn);
        this->npu_n[2]++;
    }
    Ln.ay.sync_range_from_device_exact(0, (size_t)cov * NPAD_D * sizeof(u16));
    this->t[14] += since(ts);
    if (view) *view = {raw(Ln.ay), NPAD_D};
    else {
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) from_bf16(y + (size_t)t * D, raw(Ln.ay) + (size_t)t * NPAD_D, D);
    }
    this->t[15] += since(ts);
    this->t[1] += since(t0);
    return true;
}

bool inkling2_prefill::experts(int il, const float* x, int T, const int* token_of, const int* count, const int* start,
                                const int* slot_of, const float* wsel, float* out) {
    if (T < this->min_T) return false;
    clk::time_point t0 = clk::now();
    this->calls++;
    layer_buf& B = this->layers[il];
    // each expert's rows at row0[e] of the activation buffers, padded to its dispatch's covered rows (the same
    // geometry for gate_up and down). The experts are laid out in order ord, by rows per core row (m) from the
    // largest: a group then holds few m values, and each change of m inside a dispatch costs a memtile rearm and
    // a drain (~50-60 us)
    assert(mm::max_m(DIN) == mm::max_m(DIN_DOWN));
    std::vector<uint32_t> row0(NE + 1, 0), cov(NE, 0), mof(NE, 0);
    for (int e = 0; e < NE; e++)
        if (count[e]) {
            const mm::geometry ge = mm::pick_geometry(count[e], DIN);
            cov[e] = ge.covered;
            mof[e] = ge.m;
        }
    std::vector<int> ord(NE);
    for (int e = 0; e < NE; e++) ord[e] = e;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return mof[a] > mof[b]; });
    size_t rows = 0;
    for (int e : ord) {
        row0[e] = (uint32_t)rows;
        rows += cov[e];
    }
    row0[NE] = (uint32_t)rows;
    const size_t cap = expert_rows(T);
    assert(rows <= cap);
    this->reserve(this->ex, bfp::bytes(cap, DIN) / 2);
    this->reserve(this->ea, bfp::bytes(cap, DIN_DOWN) / 2);
    this->reserve(this->ey, cap * NPAD_D);
    // G groups of consecutive experts with about equal rows, pipelined: the NPU runs one group's gate_up while
    // the host packs, activates and unpacks its neighbours
    const int G = (int)this->app_gu.size();
    // DIAG trace
    static const int trace_il = [] { const char* e = std::getenv("INKLING2_EXP_TRACE"); return e ? std::atoi(e) : -1; }();
    std::vector<std::pair<std::string, double>> tr;
    auto mark = [&](std::string n) { if ((il == trace_il || trace_il == -2)) tr.push_back({n, std::chrono::duration<double, std::milli>(clk::now() - t0).count()}); };
    static const bool serial = std::getenv("INKLING2_EXP_SERIAL") != nullptr;
    // INKLING2_EXP_CHAIN=1: gate_up writes situ(gate, up) as the down's bfp16 x on the NPU and the down follows it in
    // the same queue. Off by default: the kernel epilogue adds ~10% to gate_up and the host loses its overlap
    // (pf_2k 502 vs 517 tok/s); by default the host activates gate_up's bf16 output while the next group runs
    // not in the current kernel: the setting is ignored
    const bool chain = false;
    if (!chain) this->reserve(this->egu, cap * NPAD_D);
    // group g = experts ord[e0[g]] .. ord[e0[g + 1] - 1], rows [r0g[g], r0g[g + 1]). The first and last groups
    // take a fraction edge of the rows each (the NPU waits for the first one's pack and sequence, and the host
    // sums the last one's down output after the NPU is done), the others equal parts of the rest
    static const double edge = [] { const char* c = std::getenv("INKLING2_EXP_EDGE"); return c ? std::atof(c) : 0.08; }();
    std::vector<double> cut(G + 1, 1.0);
    cut[0] = 0.0;
    const double ed = G > 2 ? edge : 0.0, mid = G > 2 ? (1.0 - 2 * ed) / (G - 2) : 1.0 / G;
    for (int g = 1; g < G; g++) cut[g] = ed > 0 ? ed + (g - 1) * mid : (double)g / G;
    std::vector<int> e0(G + 1, NE);
    std::vector<size_t> r0g(G + 1, rows);
    e0[0] = 0;
    r0g[0] = 0;
    for (int g = 1; g < G; g++) {
        int i = e0[g - 1];
        while (i < NE && row0[ord[i]] < rows * cut[g]) i++;
        e0[g] = i;
        r0g[g] = i < NE ? row0[ord[i]] : rows;
    }
    const size_t gu_bytes = mm::q4k_bytes((size_t)2 * FF * D), dn_bytes = mm::q4k_bytes((size_t)D * FF_PAD);
    auto range = [&](int g) { return std::make_pair(r0g[g], r0g[g + 1]); };
    std::vector<flm_rt::run> gu(G), dn(G);
    std::vector<char> live(G, 0);
    std::vector<int> group_of(NE, 0);
    for (int g = 0; g < G; g++)
        for (int i = e0[g]; i < e0[g + 1]; i++) {
            live[g] |= count[ord[i]] > 0;
            group_of[ord[i]] = g;
        }
    // every dispatch's sequence, ELF and kernel depend only on the counts: built on worker threads while the host
    // packs (about 1 ms each, most of it the ELF and the kernel object). The first gate_up's alone, so the NPU
    // starts as soon as its x is packed; the others once it runs (they would compete with it for the cores)
    std::vector<std::future<void>> built_gu(G), built_dn(G);
    int first = 0;
    while (first < G && !live[first]) first++;
    auto launch = [&](int g, bool down) {
        auto build = [&, g](bool down) {
            std::vector<mm::job> jobs;
            for (int i = e0[g]; i < e0[g + 1]; i++)
                if (const int e = ord[i]; count[e])
                    jobs.push_back({(uint32_t)count[e], (uint32_t)bfp::bytes(row0[e], down ? DIN_DOWN : DIN),
                                    !down && chain ? (uint32_t)bfp::bytes(row0[e], DIN_DOWN) : row0[e] * NPAD_D,
                                    (down ? B.down + e * dn_bytes : B.gate_up + e * gu_bytes)});
            npu_app& a = down ? this->app_dn[g] : this->app_gu[g];
            mm::gen_batch(a.seq(), down ? DIN_DOWN : DIN, down ? FF_PAD : D, NPAD_D, jobs);
            a.update_ctrl_seq();
        };
        (down ? built_dn : built_gu)[g] = std::async(std::launch::async, build, down);
    };
    if (first < G) launch(first, false);
    auto launch_rest = [&] {
        for (int g = first; g < G; g++) {
            if (!live[g]) continue;
            if (g != first) launch(g, false);
            launch(g, true);
        }
    };
    // the 16-row groups of group g: (expert, first row)
    auto units = [&](int g) {
        std::vector<std::pair<int, uint32_t>> u;
        for (int i = e0[g]; i < e0[g + 1]; i++)
            for (uint32_t r = 0; r < cov[ord[i]]; r += 16) u.push_back({ord[i], r});
        return u;
    };
    // the bfp16 x of expert e's 16 rows from r into buf (DIN_X columns), rows past count[e] zero
    auto pack_unit = [&](u8* buf, uint32_t din, int e, uint32_t r, const float* const* src, uint32_t K) {
        const mm::geometry ge = mm::pick_geometry(count[e], din);
        bfp::pack_group(buf + bfp::bytes(row0[e] + r / ge.m * ge.m, din), ge.m, (r % ge.m) / 16, src, K, din);
    };

    auto start_gu = [&](int g) {
        clk::time_point t = clk::now();
        const auto [r0, r1] = range(g);
        const auto U = units(g);
        #pragma omp parallel for schedule(dynamic, 4)
        for (int64_t i = 0; i < (int64_t)U.size(); i++) {
            const auto [e, r] = U[i];
            const float* src[16];
            for (uint32_t j = 0; j < 16; j++)
                src[j] = (int)(r + j) < count[e] ? x + (size_t)token_of[start[e] + r + j] * D : nullptr;
            pack_unit(bytes_of(this->ex), DIN, e, r, src, D);
        }
        this->ex.sync_range_to_device_exact(bfp::bytes(r0, DIN), bfp::bytes(r1 - r0, DIN));
        this->t[2] += since(t);
        mark("packed gu" + std::to_string(g));
        built_gu[g].get();
        clk::time_point tsub = clk::now();
        gu[g] = this->app_gu[g].create_run(chain ? this->ea : this->egu, this->ex, B.w);
        gu[g].start();
        this->npu_s[5] += since(tsub);
        this->npu_sn[5]++;
        mark("started gu" + std::to_string(g));
        if (g == first) launch_rest();
        this->runs++;
        this->t[3] += since(t);
        if (serial) {
            gu[g].wait();
            mark("ran gu" + std::to_string(g));
            this->t[16] += since(t);
        }
        if (chain) {
            // the down reads what the gate_up run writes: queued behind it, no host step
            t = clk::now();
            built_dn[g].get();
            dn[g] = this->app_dn[g].create_run(this->ey, this->ea, B.w);
            dn[g].start();
            this->runs++;
            this->t[4] += since(t);
        }
    };
    auto prep_dn = [&](int g) {
        clk::time_point t = clk::now();
        built_dn[g].get();
        clk::time_point tsub = clk::now();
        dn[g] = this->app_dn[g].create_run(this->ey, this->ea, B.w);
        this->npu_s[6] += since(tsub);
        this->t[8] += since(t);
    };
    auto start_dn = [&](int g) {
        clk::time_point t = clk::now();
        mark("wait gu" + std::to_string(g));
        gu[g].wait();
        mark("done gu" + std::to_string(g));
        this->t[5] += since(t);
        const auto [r0, r1] = range(g);
        this->egu.sync_range_from_device_exact(r0 * NPAD_D * sizeof(u16), (r1 - r0) * NPAD_D * sizeof(u16));
        const auto U = units(g);
        #pragma omp parallel for schedule(dynamic, 4)
        for (int64_t i = 0; i < (int64_t)U.size(); i++) {
            const auto [e, r] = U[i];
            const u16* gp = raw(this->egu) + (size_t)(row0[e] + r) * NPAD_D;
            std::vector<float> af((size_t)16 * FF);
            const float* src[16];
            for (uint32_t j = 0; j < 16; j++) {
                src[j] = nullptr;
                if ((int)(r + j) >= count[e]) continue;
                const u16* g = gp + (size_t)j * NPAD_D;
                float* a = af.data() + (size_t)j * FF;
                // gate channels c in row c / 32 * 64 + c % 32, up 32 rows after (qmat::glu)
                for (int c = 0; c < FF; c += 16) {
                    const int gi = c / 32 * 64 + c % 32;
                    _mm512_storeu_ps(a + c, situ16(load_bf16x16(g + gi), load_bf16x16(g + gi + 32)));
                }
                src[j] = a;
            }
            pack_unit(bytes_of(this->ea), DIN_DOWN, e, r, src, FF);
        }
        this->ea.sync_range_to_device_exact(bfp::bytes(r0, DIN_DOWN), bfp::bytes(r1 - r0, DIN_DOWN));
        clk::time_point tsub = clk::now();
        dn[g].start();
        this->npu_s[6] += since(tsub);
        this->npu_sn[6]++;
        mark("started dn" + std::to_string(g));
        this->runs++;
        this->t[4] += since(t);
        if (serial) {
            dn[g].wait();
            mark("ran dn" + std::to_string(g));
            this->t[17] += since(t);
        }
    };
    // out[t] = the weighted sum of its TOPK down rows, read as bf16 where the NPU wrote them, one group at a time
    std::vector<uint32_t> dev_row((size_t)T * TOPK);
    std::vector<uint8_t> slot_group((size_t)T * TOPK);
    for (int e = 0; e < NE; e++)
        for (int r = 0; r < count[e]; r++) {
            dev_row[start[e] + r] = row0[e] + r;
            slot_group[start[e] + r] = (uint8_t)group_of[e];
        }
    const u16* ey = raw(this->ey);
    bool first_sum = true;
    auto accumulate = [&](int g) {
        const bool init = first_sum;
        first_sum = false;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            float* o = out + (size_t)t * D;
            const u16* y[TOPK];
            __m512 w[TOPK];
            int n = 0;
            for (int k = 0; k < TOPK; k++) {
                const int sl = slot_of[(size_t)t * TOPK + k];
                if (slot_group[sl] != g) continue;
                y[n] = ey + (size_t)dev_row[sl] * NPAD_D;
                w[n++] = _mm512_set1_ps(wsel[(size_t)t * TOPK + k]);
            }
            if (n == 0) {
                if (init) std::fill(o, o + D, 0.0f);
                continue;
            }
            for (int c = 0; c < D; c += 16) {
                __m512 acc = init ? _mm512_setzero_ps() : _mm512_loadu_ps(o + c);
                for (int k = 0; k < n; k++) acc = _mm512_fmadd_ps(w[k], load_bf16x16(y[k] + c), acc);
                _mm512_storeu_ps(o + c, acc);
            }
        }
    };
    auto finish = [&](int g) {
        clk::time_point t = clk::now();
        mark("wait dn" + std::to_string(g));
        dn[g].wait();
        mark("done dn" + std::to_string(g));
        this->t[6] += since(t);
        const auto [r0, r1] = range(g);
        this->ey.sync_range_from_device_exact(r0 * NPAD_D * sizeof(u16), (r1 - r0) * NPAD_D * sizeof(u16));
        accumulate(g);
        this->t[7] += since(t);
    };

    if (!serial) { clk::time_point tp = t0; this->t[16] += since(tp); }
    // NPU order: gu 0, gu 1, dn 0, gu 2, dn 1, ...; chained: gu 0, dn 0, gu 1, dn 1, ... (the host packs group
    // g + 1 while the NPU runs group g)
    int prev = -1;
    for (int g = 0; chain && g < G; g++)
        if (live[g]) {
            start_gu(g);
            if (prev >= 0) finish(prev);
            prev = g;
        }
    for (int g = 0; !chain && g < G; g++)
        if (live[g]) {
            if (prev < 0) start_gu(g);
            int next = g + 1;
            while (next < G && !live[next]) next++;
            if (next < G) start_gu(next);
            if (this->idle) {
                auto job = std::move(this->idle);
                this->idle = nullptr;
                clk::time_point ti = clk::now();
                job();
                mark("idle");
                this->t[18] += since(ti);
            }
            prep_dn(g);
            start_dn(g);
            if (prev >= 0) finish(prev);
            prev = g;
        }
    if (prev >= 0) finish(prev);
    if (first_sum) std::fill(out, out + (size_t)T * D, 0.0f);
    this->t[9] += since(t0);
    mark("combined");
    if ((il == trace_il || trace_il == -2)) {
        std::fprintf(stderr, "inkling2 trace layer %d T %d:", il, T);
        for (int g = 0; g < G; g++) {
            int n = 0, mx = 0;
            std::set<uint32_t> ms;
            for (int i = e0[g]; i < e0[g + 1]; i++) {
                const int e = ord[i];
                n += count[e], mx = std::max(mx, count[e]);
                if (count[e]) ms.insert(mm::pick_geometry(count[e], DIN).m);
            }
            std::fprintf(stderr, " [g%d experts %d rows %d covered %zu max %d m-values %zu]", g, e0[g + 1] - e0[g], n,
                         range(g).second - range(g).first, mx, ms.size());
        }
        for (auto& [n, v] : tr) std::fprintf(stderr, " %s@%.2f", n.c_str(), v);
        std::fprintf(stderr, "\n");
    }
    if (this->prof && this->calls % (NL + 24) == 0) this->report();
    // INKLING2_EXP_CHECK: the chained result against the host-activated one on the same input
    static const bool check = std::getenv("INKLING2_EXP_CHECK") != nullptr;
    if (check && chain) {
        std::vector<float> ref((size_t)T * D);
        // the activations as the NPU wrote them, then as the host packs them: error by channel position
        const size_t abytes = bfp::bytes(row0[NE], DIN_DOWN);
        this->ea.sync_range_from_device_exact(0, abytes);
        std::vector<u8> dev(bytes_of(this->ea), bytes_of(this->ea) + abytes);
        this->unchain = true;
        this->experts(il, x, T, token_of, count, start, slot_of, wsel, ref.data());
        this->unchain = false;
        double d2 = 0, r2 = 0, dmax = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const double d = (double)out[i] - ref[i];
            d2 += d * d;
            r2 += (double)ref[i] * ref[i];
            dmax = std::max(dmax, std::fabs(d));
        }
        {
            const u8* hst = bytes_of(this->ea);
            auto val = [&](const u8* b, int e, uint32_t R, int c) {
                const mm::geometry ge = mm::pick_geometry(count[e], DIN_DOWN);
                const size_t at = bfp::bytes(row0[e] + R / ge.m * ge.m, DIN_DOWN) + (size_t)(c / 256) * (ge.m / 16) * 4608 +
                                  (R % ge.m) / 16 * 4608 + (c % 256) / 8 * 144 + (R % 16) / 8 * 72 + (R % 8) * 9;
                return (float)(int8_t)b[at + 1 + c % 8] * std::ldexp(1.0f, (int)b[at] - 133);
            };
            // buckets: channel % 32 (core column slot), channel / 256 (pass), row % 16
            double eb[3][32] = {}, rb[3][32] = {};
            for (int e = 0; e < NE; e++)
                for (int R = 0; R < count[e]; R++)
                    for (int c = 0; c < FF; c++) {
                        const double a = val(dev.data(), e, R, c), b = val(hst, e, R, c);
                        const int k[3] = {c % 32, c / 256, R % 16};
                        for (int i = 0; i < 3; i++) {
                            eb[i][k[i]] += (a - b) * (a - b);
                            rb[i][k[i]] += b * b;
                        }
                    }
            for (int i = 0; i < 3 && il == 0; i++) {
                std::fprintf(stderr, "inkling2 exp check act by %s:", i == 0 ? "ch%32" : i == 1 ? "pass" : "row%16");
                for (int j = 0; j < (i == 1 ? 5 : i == 0 ? 32 : 16); j++) std::fprintf(stderr, " %.3f", std::sqrt(eb[i][j] / std::max(rb[i][j], 1e-30)));
                std::fprintf(stderr, "\n");
            }
        }
        std::fprintf(stderr, "inkling2 exp check layer %2d T %d: rel %.3e max abs %.3e (rms ref %.3e)\n", il, T,
                     std::sqrt(d2 / std::max(r2, 1e-30)), dmax, std::sqrt(r2 / ref.size()));
    }
    return true;
}

void inkling2_prefill::set_attention(const std::string& xclbin, const std::string& switches) {
#if !defined(FLM_USE_HRX)
    if (std::filesystem::exists(switches)) this->ll.file = switches;
    if (this->ll.possible()) {
        this->ll.tried = true;
        const char* dir = std::getenv("INKLING2_SWITCH_DIR");
        const bool use_dir = dir && dir[0];
        try {
            if (use_dir) this->ll.load(dir, this->mgr->get_device(), this->mgr->get_context(), this->mgr->get_kernel_name());
            else this->ll.load_file(this->ll.file, this->mgr->get_device(), this->mgr->get_context(), this->mgr->get_kernel_name());
            header_print("inkling2", "attention shares the matmul hardware context (control-packet switches from "
                                         << (use_dir ? dir : this->ll.file.c_str()) << ")");
        } catch (const std::exception& e) {
            this->ll.ready = false;
            this->ll.sw.clear();
            header_print("inkling2", "control-packet switches off: " << e.what());
        }
    }
    if (this->ll.ready) {
        this->att_mgr = this->mgr;
        return;
    }
#else
    (void)switches;
#endif
    this->att_mgr = this->npu->register_xclbin(xclbin);
}

bool inkling2_prefill::attention(int il, int p0, int T, const float* Q, const float* R, const float* sinks,
                                 const float* rel_bias, const u16* Kc, const u16* Vc, int ring, int window,
                                 const float* K, const float* V, const float* G, const bf16_rows* Gb, float* O,
                                 const std::function<void()>& packed) {
    namespace at = inkling2_attn;
    if (!this->att_mgr || T < this->min_T) return false;
    clk::time_point t0 = clk::now();
    // the kernel's frame: key row r is position F + r; a SWA query needs no key before p0 - window + 1
    const int F = window ? std::max(0, p0 - window) : 0;
    const int q0 = p0 - F;
    const int Lb = q0 / at::ROUND * at::ROUND, Le = (q0 + T + at::ROUND - 1) / at::ROUND * at::ROUND;
    const int rounds = (Le - Lb) / at::ROUND;
    const size_t half = (size_t)Le * NKV * at::KV_ROW;   // u16 units of the K (or V) tiles
    lane_buf& Ln = this->lane();
    this->reserve(Ln.att_o, (size_t)(Le - Lb) * QD);
    this->reserve(Ln.att_q, at::q_pack_size(rounds));
    this->reserve(Ln.att_kv, 2 * half);
    this->reserve(Ln.att_tab, at::TAB_SIZE);
    uint8_t* kv = (uint8_t*)raw(Ln.att_kv);
    // head-major bfp16 tiles of 16 keys, so the shim reads each kv window as one contiguous run
    #pragma omp parallel for schedule(static)   // flat: MSVC's OpenMP 2.0 ignores collapse
    for (int it = 0; it < Le / 16 * NKV; it++) {
        const int t = it / NKV * 16, h = it % NKV;
        alignas(64) float k[16 * DH], v[16 * DH];
        for (int i = 0; i < 16; i++) {
            const int r = t + i;
            float* kr = k + i * DH;
            float* vr = v + i * DH;
            if (r < q0) {
                const int p = F + r;
                const size_t row = (size_t)(ring ? p % ring : p) * KVD + h * DH;
                from_bf16(kr, Kc + row, DH);
                from_bf16(vr, Vc + row, DH);
            } else if (r < q0 + T) {
                memcpy(kr, K + (size_t)(r - q0) * KVD + h * DH, DH * sizeof(float));
                memcpy(vr, V + (size_t)(r - q0) * KVD + h * DH, DH * sizeof(float));
            } else {
                std::fill(kr, kr + DH, 0.0f);
                std::fill(vr, vr + DH, 0.0f);
            }
        }
        uint8_t* kt = kv + ((size_t)h * Le + t) * at::KV_ROW * sizeof(u16);
        at::pack_kv_tile(kt, kt + half * sizeof(u16), k, v, DH, bfp::pack2);
    }
    // query packs; padding rows are zero queries with a zero sink
    static const float zero[DH] = {};
    #pragma omp parallel for schedule(static)   // flat, as above
    for (int ir = 0; ir < (Le - Lb) * NH; ir++) {
        const int r = Lb + ir / NH, h = ir % NH;
        const int t = r - q0;
        const bool in = t >= 0 && t < T;
        at::pack_q(Ln.att_q.data(), rounds, h, r - Lb, in ? Q + (size_t)t * QD + h * DH : zero,
                   in ? R + ((size_t)t * NH + h) * NREL : zero, in ? sinks[(size_t)t * NH + h] : 0.0f);
    }
    at::pack_table(Ln.att_tab.data(), rel_bias);
    Ln.att_kv.sync_range_to_device_exact(0, 2 * half * sizeof(u16));
    Ln.att_q.sync_range_to_device_exact(0, at::q_pack_size(rounds) * sizeof(u16));
    Ln.att_tab.sync_range_to_device_exact(0, at::TAB_SIZE * sizeof(u16));
    if (packed) packed();   // the cache is free: the next half's attention packs while this one runs
    this->t[10] += since(t0);
    {
        const clk::time_point treq = clk::now();   // DIAG
        std::lock_guard<std::mutex> lock(this->npu_mu);
        clk::time_point tg = clk::now();
        struct ntrace {   // DIAG: INKLING2_HALF_TRACE
            int il; const char* kind; clk::time_point a, b;
            ~ntrace() { trace_npu(il, kind, a, b); }
        } ntr{il, "att", treq, tg};
        npu_app& a = this->att_app(Lb, Le, window, (int)(2 * half));
        clk::time_point tn = clk::now();
        this->npu_s[window ? 3 : 4] += std::chrono::duration<double>(tn - tg).count();
        this->npu_sn[window ? 3 : 4]++;
#if !defined(FLM_USE_HRX)
        if (this->ll.ready) {
            // one submission: cold_mm, the switch to attention, attention, the switch back
            std::vector<flm_rt::run> guest;
            guest.push_back(a.create_run(Ln.att_o, Ln.att_q, Ln.att_kv, Ln.att_tab));
            this->ll.window(this->mgr, "mm_then_attn", "attn_then_mm", guest);
        } else
#endif
        a(Ln.att_o, Ln.att_q, Ln.att_kv, Ln.att_tab);
        this->npu_t[window ? 3 : 4] += since(tn);
        this->npu_n[window ? 3 : 4]++;
    }
    this->t[11] += since(t0);
    Ln.att_o.sync_range_from_device_exact(0, (size_t)(Le - Lb) * QD * sizeof(u16));
    // O = sigmoid(G) * attention
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const u16* a = raw(Ln.att_o) + (size_t)(q0 - Lb + t) * QD;
        const float* g = Gb ? nullptr : G + (size_t)t * QD;
        const u16* gb = Gb ? Gb->row(t) : nullptr;
        float* o = O + (size_t)t * QD;
        for (int c = 0; c < QD; c += 16) {
            const __m512 gv = gb ? load_bf16x16(gb + c) : _mm512_loadu_ps(g + c);
            const __m512 s = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), exp16(_mm512_sub_ps(_mm512_setzero_ps(), gv))));
            _mm512_storeu_ps(o + c, _mm512_mul_ps(s, load_bf16x16(a + c)));
        }
    }
    this->t[12] += since(t0);
    return true;
}
