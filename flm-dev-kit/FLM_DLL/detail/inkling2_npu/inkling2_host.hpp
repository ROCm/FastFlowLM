// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_host.hpp
/// \brief Inkling-2 (inkling2_loop) on the host: the whole model, read in place
///        from a memory-mapped model.q4nx
/// \note This is the reference the NPU stages are checked against, and the path
///       a stage takes when it is not on the NPU. fp32 activations, the K/V cache
///       in bf16.
///
///       Model, per token:
///         h0 = rmsnorm(embed[id]) * embed_norm
///         layers 0-5, then layers 6-17 three times (the loop), then layers 18-23:
///         48 layer applications, each with its own cache slot (KV + conv state).
///         Layers 5, 11, 17, 23 attend to the whole context, the others to the
///         last 1024 tokens. No RoPE: a learned bias per distance (< 1024) and
///         head, plus a per-query sink logit.
///         Each layer's attention and FFN inputs are softmax mixes over the block
///         outputs so far (one block = 6 layers); loop passes 2 and 3 add a gated
///         message from the previous pass.
#pragma once
#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace inkling2_host {

using u8 = uint8_t;
using u16 = uint16_t;

// ---- dimensions (config.json text_config; checked against it at load)
constexpr int D = 2304;              // hidden size
constexpr int NH = 24;               // query heads
constexpr int NKV = 8;               // KV heads
constexpr int DH = 128;              // head dim
constexpr int QD = NH * DH;          // 3072
constexpr int KVD = NKV * DH;        // 1024
constexpr int NREL = 16;             // relative-position features per head
constexpr int REL_EXT = 1024;        // distances with a learned bias
constexpr int SWA = 1024;            // sliding window: keys with p_q - p_k < SWA
constexpr int NL = 24;               // weight layers
constexpr int CORE0 = 6, CORE1 = 18; // the looped layers [CORE0, CORE1)
constexpr int NPASS = 3;
constexpr int NSLOT = NL + (NPASS - 1) * (CORE1 - CORE0);   // 48 cache slots
constexpr int BLOCK = 6;             // layers per residual block
constexpr int NE = 128, TOPK = 8;    // routed experts
constexpr int FF = 1152;             // expert intermediate size
constexpr int FF_PAD = 1280;         // down_proj columns in model.q4nx (zero padded)
constexpr int RANK = 64;             // loop controller rank
constexpr int VOCAB_PAD = 201024;    // embedding / lm_head rows
constexpr int VOCAB = 200058;        // real vocabulary (logits are sliced to this)
constexpr int NCONV = 4;             // short convolution taps (index 3 = this token)
constexpr float EPS = 1e-6f;
constexpr float QK_CLIP = 22.627416997969522f;
constexpr float REL_CLIP = 8.0f;
constexpr float SOFTCAP = 30.0f;
constexpr float SITU_BETA = 4.0f, SITU_LINEAR_BETA = 25.0f;
constexpr float ROUTE_SCALE = 8.0f;
constexpr float LOG_SCALE_FLOOR = 8192.0f, LOG_SCALE_ALPHA = 0.1f;
// conv state channels: K, V, attention out, FFN out
constexpr int CONV_K = 0, CONV_V = KVD, CONV_ATTN = 2 * KVD, CONV_FFN = 2 * KVD + D;
constexpr int CONV_CH = 2 * KVD + 2 * D;   // 6656
// prompt tokens per step (INKLING2_CHUNK overrides); a step's SWA layers write and attend in sub-steps of
// at most SUB tokens, so a SWA slot keeps SWA - 1 + SUB keys in its ring
constexpr int CHUNK = 4096;
constexpr int SUB = 1024;
constexpr int SWA_RING = SWA + SUB;

inline bool is_global(int layer) { return layer % BLOCK == BLOCK - 1; }
/// \brief the weight layer of cache slot s
inline int slot_layer(int s) { return s < NL ? s : CORE0 + (s - NL) % (CORE1 - CORE0); }
/// \brief the cache slot of weight layer `layer` in loop pass `pass`
inline int layer_slot(int layer, int pass) {
    return (pass == 0 || layer < CORE0 || layer >= CORE1) ? layer : NL + (pass - 1) * (CORE1 - CORE0) + layer - CORE0;
}

// ---- quantized matrices: q4nx blocks of 32 rows x 256 columns, [row block][column block]
enum class qfmt { q4k, q41, q80 };
constexpr int QROWS = 32, QCOLS = 256;
constexpr size_t Q4K_BYTES = 4736, Q41_BYTES = 5120, Q80_BYTES = 8704;

/// \brief one Q4_K block (scales u8 [8 g][32 row], mins likewise, qs [256 col][16 B] with row 2i in the low nibble of
///        byte i, S bf16 [32], M bf16 [32]) into the NPU kernel's order. The two cores of a row pair take rows
///        16 H .. 16 H + 15 each, as 8 even and 8 odd rows (par): row = 16 H + 2 r + par. Same 4736 bytes:
///        scales u8 [H][par][g][r], mins likewise, qs [H][par][kt 32][r 8][k 8] nibbles (low first, col = 8 kt + k),
///        S bf16 [H][par][r], M likewise. Each core then reads its scales of all 8 groups as one vector and its
///        weights as n major 8 x 8 tiles, with no permutes.
inline void q4k_to_npu(u8* dst, const u8* src) {
    for (int H = 0; H < 2; H++)
        for (int par = 0; par < 2; par++) {
            const int hp = 2 * H + par;
            for (int r = 0; r < 8; r++) {
                const int row = 16 * H + 2 * r + par;
                for (int g = 0; g < 8; g++) {
                    dst[hp * 64 + g * 8 + r] = src[g * 32 + row];
                    dst[256 + hp * 64 + g * 8 + r] = src[256 + g * 32 + row];
                }
                memcpy(dst + 4608 + hp * 16 + 2 * r, src + 4608 + 2 * row, 2);
                memcpy(dst + 4672 + hp * 16 + 2 * r, src + 4672 + 2 * row, 2);
            }
        }
    for (int H = 0; H < 2; H++)
        for (int kt = 0; kt < 32; kt++) {
            u8* d0 = dst + 512 + 2048 * H + 32 * kt;   // par 0
            u8* d1 = d0 + 1024;                        // par 1
            for (int r = 0; r < 8; r++)
                for (int k = 0; k < 8; k += 2) {
                    // byte 8 H + r of a column: row 16 H + 2 r (low nibble), 16 H + 2 r + 1 (high)
                    const u8 a = src[512 + (8 * kt + k) * 16 + 8 * H + r], b = src[512 + (8 * kt + k + 1) * 16 + 8 * H + r];
                    d0[(r * 8 + k) / 2] = (u8)((a & 0x0F) | (b << 4));
                    d1[(r * 8 + k) / 2] = (u8)((a >> 4) | (b & 0xF0));
                }
        }
}

/// \brief the inverse of q4k_to_npu
inline void q4k_from_npu(u8* dst, const u8* src) {
    for (int H = 0; H < 2; H++)
        for (int par = 0; par < 2; par++) {
            const int hp = 2 * H + par;
            for (int r = 0; r < 8; r++) {
                const int row = 16 * H + 2 * r + par;
                for (int g = 0; g < 8; g++) {
                    dst[g * 32 + row] = src[hp * 64 + g * 8 + r];
                    dst[256 + g * 32 + row] = src[256 + hp * 64 + g * 8 + r];
                }
                memcpy(dst + 4608 + 2 * row, src + 4608 + hp * 16 + 2 * r, 2);
                memcpy(dst + 4672 + 2 * row, src + 4672 + hp * 16 + 2 * r, 2);
            }
        }
    for (int H = 0; H < 2; H++)
        for (int kt = 0; kt < 32; kt++) {
            const u8* s0 = src + 512 + 2048 * H + 32 * kt;
            const u8* s1 = s0 + 1024;
            for (int r = 0; r < 8; r++)
                for (int k = 0; k < 8; k++) {
                    const int i = r * 8 + k, sh = 4 * (i & 1);
                    const u8 lo = (s0[i / 2] >> sh) & 0x0F, hi = (s1[i / 2] >> sh) & 0x0F;
                    dst[512 + (8 * kt + k) * 16 + 8 * H + r] = (u8)(lo | (hi << 4));
                }
        }
}

constexpr int NPU_GROUP = 3;   // row blocks per group of the NPU layout (qmat::grouped)

struct qmat {
    qfmt fmt = qfmt::q4k;
    int N = 0, K = 0;        // rows (outputs), columns (inputs, padded to 256)
    const u8* data = nullptr;
    /// \brief the NPU's layout: row blocks in triples, the three interleaved along the column blocks
    ///        ([3t, 0] [3t + 1, 0] [3t + 2, 0] [3t, 1] ...), each block in q4k_to_npu order; see inkling2_npu_mm.hpp
    bool grouped = false;
    /// \brief > 0: stacked gate / up matrices (glu gate row blocks, then glu up row blocks, per expert) stored with
    ///        gate block j and up block j side by side (physical row blocks 2 j, 2 j + 1), so a pair holds the gate
    ///        and up rows of the same 32 channels; row block numbers stay logical
    int glu = 0;

    size_t block_bytes() const { return fmt == qfmt::q4k ? Q4K_BYTES : fmt == qfmt::q41 ? Q41_BYTES : Q80_BYTES; }
    int col_blocks() const { return K / QCOLS; }
    int row_blocks() const { return N / QROWS; }
    const u8* block(int rb, int cb) const {
        if (glu) {
            const int e = rb / (2 * glu), j = rb % (2 * glu);
            rb = e * 2 * glu + (j < glu ? 2 * j : 2 * (j - glu) + 1);
        }
        const size_t i = grouped ? ((size_t)(rb / NPU_GROUP) * col_blocks() + cb) * NPU_GROUP + rb % NPU_GROUP
                                 : (size_t)rb * col_blocks() + cb;
        return data + i * block_bytes();
    }
    /// \brief rows [32 rb0, 32 rb0 + n): one expert of a stacked expert tensor (rb0 a multiple of NPU_GROUP when
    ///        grouped, a whole expert when glu)
    qmat rows(int rb0, int n) const { return {fmt, n, K, block(rb0, 0), grouped, glu}; }
};

// ---- weights of one layer (pointers into the mapped file)
struct loop_ctrl {
    const float* down = nullptr;     // [RANK][2 D]
    const float* forget = nullptr;   // [D][RANK]
    const float* gain = nullptr;     // [D][RANK]
};

struct layer_weights {
    qmat q, k, v, o, gate, rel;      // [QD|KVD|KVD|D|QD|NH*NREL] x D (o: D x QD)
    const u16* sink_proj = nullptr;  // bf16 [NH][D]
    const u16* k_conv_proj = nullptr;  // bf16 [NKV*NCONV][D], global layers
    const u16* v_conv_proj = nullptr;
    const float* q_clip = nullptr;   // [NH][DH]
    const float* rel_clip = nullptr; // [NH][NREL]
    const float* rel_bias = nullptr; // [REL_EXT][NREL]
    const float* k_conv = nullptr;   // [KVD][NCONV]
    const float* v_conv = nullptr;
    const float* attn_conv = nullptr;  // [D][NCONV]
    const float* ffn_conv = nullptr;
    const float* attn_norm = nullptr;  // SWA layers
    const float* qkv_norm[3] = {};     // global layers
    const float* attn_query = nullptr; // [1 or 3][D], none on layer 0
    const float* ffn_query = nullptr;  // [D]
    const float* ffn_norm = nullptr;
    const float* router = nullptr;     // [NE][D]
    const float* expert_bias = nullptr;
    qmat gate_up;                      // [NE * 2 FF] x D: per expert FF gate rows, then FF up rows
    qmat down;                         // [NE * D] x FF_PAD
    const float* msg_query = nullptr;  // [4][D], looped layers
    loop_ctrl ctrl[4];                 // q, k, v, ffn
};

struct weights {
    const u16* embed = nullptr;        // bf16 [VOCAB_PAD][D]
    const float* embed_norm = nullptr;
    const float* final_query = nullptr;
    const float* norm = nullptr;
    qmat lm_head;                      // Q8_0 [VOCAB_PAD] x D
    layer_weights layer[NL];
};

/// \brief the experts' activation: situ(g, u) = b tanh(g / b) sigmoid(g) * lb tanh(u / lb)
inline float situ(float g, float u) {
    return SITU_BETA * std::tanh(g / SITU_BETA) / (1.0f + std::exp(-g)) * SITU_LINEAR_BETA * std::tanh(u / SITU_LINEAR_BETA);
}

/// \brief 16 bf16 values as fp32
inline __m512 load_bf16x16(const u16* p) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}

/// \brief n bf16 values as fp32 (n a multiple of 16)
inline void bf16_to_f32(float* dst, const u16* src, int n) {
    for (int i = 0; i < n; i += 16) _mm512_storeu_ps(dst + i, load_bf16x16(src + i));
}

/// \brief rows of bf16 values left where a device wrote them: row t at p + t ld
struct bf16_rows {
    const u16* p = nullptr;
    size_t ld = 0;
    const u16* row(int t) const { return p + (size_t)t * ld; }
};

/// \brief e^x, 16 lanes (Cephes polynomial, rel. error ~2e-7; x clamped to [-87, 88])
inline __m512 exp16(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.0f)), _mm512_set1_ps(88.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(p, n);
}

/// \brief tanh, 16 lanes: sign(x) (1 - 2 / (e^(2|x|) + 1))
inline __m512 tanh16(__m512 x) {
    const __m512 ax = _mm512_abs_ps(x);
    const __m512 e = exp16(_mm512_add_ps(ax, ax));
    const __m512 t = _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, _mm512_set1_ps(1.0f))));
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(t),
                                               _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32((int)0x80000000))));
}

/// \brief situ of 16 lanes
inline __m512 situ16(__m512 g, __m512 u) {
    const __m512 sg = _mm512_div_ps(_mm512_set1_ps(SITU_BETA * SITU_LINEAR_BETA), _mm512_add_ps(_mm512_set1_ps(1.0f), exp16(_mm512_sub_ps(_mm512_setzero_ps(), g))));
    return _mm512_mul_ps(_mm512_mul_ps(sg, tanh16(_mm512_mul_ps(g, _mm512_set1_ps(1.0f / SITU_BETA)))),
                         tanh16(_mm512_mul_ps(u, _mm512_set1_ps(1.0f / SITU_LINEAR_BETA))));
}


/// \brief the matmuls of a prompt step another device takes over (inkling2_npu); a call that returns
///        false leaves the work to the host. il is the weight layer.
class mm_offload {
public:
    virtual ~mm_offload() = default;
    /// \brief host work that does not need the next call's result: experts() runs it while the NPU computes (once,
    ///        then clears it); whoever set it runs it if it is still set afterwards
    std::function<void()> idle;
    /// \brief Q = xq q^T, K = xk k^T, V = xv v^T, R = xq rel^T, G = xq gate^T for T tokens ([T][D] inputs;
    ///        xk and xv may be xq); with view, Q K V R G stay bf16 as view[0 .. 4] instead, valid until this
    ///        thread's next attn_out
    virtual bool attn_in(int il, const float* xq, const float* xk, const float* xv, int T, float* Q, float* K, float* V,
                         float* R, float* G, bf16_rows* view = nullptr) = 0;
    /// \brief y[T][D] = O[T][QD] o^T; with view, y stays bf16 as *view instead, valid until this thread's next attn_in
    virtual bool attn_out(int il, const float* O, int T, float* y, bf16_rows* view = nullptr) = 0;
    /// \brief attn_in, attention and attn_out may run on two threads at once, one per lane (0, 1), set by each thread
    virtual void set_lane(int lane) {}
    /// \brief the routed experts: row r of the expert-grouped batch (experts e in order, count[e] rows from
    ///        start[e]) is token token_of[r] of x[T][D]; Z[r] = down(situ(gate_up(x[token_of[r]]))), and
    ///        out[t] = sum over k < TOPK of wsel[t TOPK + k] Z[slot_of[t TOPK + k]], [T][D]
    virtual bool experts(int il, const float* x, int T, const int* token_of, const int* count, const int* start,
                         const int* slot_of, const float* wsel, float* out) = 0;
    /// \brief O[T][QD] = sigmoid(G) * attention of the T queries at p0 .. (Q, R, sinks as attention() takes them) over
    ///        the cached keys before p0 (Kc, Vc rows, ring 0 = row p) and this step's K, V [T][KVD]; G [T][QD] the output
    ///        gate (or Gb, its bf16 rows, when set); the cache is not written. packed (when set) runs once the cache
    ///        has been read, before the device runs (the caller may write the cache then); it runs iff this returns true
    virtual bool attention(int il, int p0, int T, const float* Q, const float* R, const float* sinks, const float* rel_bias,
                           const u16* Kc, const u16* Vc, int ring, int window, const float* K, const float* V,
                           const float* G, const bf16_rows* Gb, float* O, const std::function<void()>& packed) {
        return false;
    }
    /// \brief decode (one token): how many of a token's TOPK experts the device takes (0: none); the host runs the rest
    ///        meanwhile
    virtual int dec_experts() const { return 0; }
    /// \brief start experts ex[0 .. n) of layer il on the device: kind 0 gate_up of in = x [D], kind 1 down of
    ///        in = [n][FF_PAD] activations
    virtual void dec_start(int il, int kind, const int* ex, int n, const float* in) {}
    /// \brief wait for dec_start: kind 0 out [n][2 FF] (gate, then up), kind 1 out [n][D]
    virtual void dec_wait(float* out) {}
};

/// \brief the model.q4nx tensors, located in place (safetensors container)
class q4nx_file;

/// \brief DIAG: set, it sees the values of every one-token layer application (cache slot, tag, n floats):
///        in, nrm, q, k, o, y, res, xf, sel (ids, then weights), moe, out
extern std::function<void(int slot, const char* tag, const float* x, int n)> decode_trace;

/// \brief one token of a loop pass input: rms_norm(control(x, message m)) * g (the NPU decode's controller step)
void control_norm(const float* x, const float* m, const loop_ctrl& c, const float* g, float* out);

/// \brief logits[VOCAB] = softcap(lm_head . x) of one final-normed token
void lm_head(const weights& w, const float* x, float* logits);

/// \brief the whole model on the host, one sequence
class model {
public:
    model();
    ~model();
    model(const model&) = delete;
    model& operator=(const model&) = delete;

    /// \brief map model.q4nx and locate every tensor; throws on a missing or misshapen one
    void load(const std::string& q4nx_path);
    /// \brief allocate the caches for up to max_len tokens (global KV grows linearly with it); keeps the
    ///        sequence when it still fits
    void set_max_length(int max_len);
    int max_length() const;
    /// \brief forget the sequence
    void reset();
    /// \brief tokens so far
    int position() const;
    /// \brief move the position without touching the caches (after load_fixed / write_rows)
    void set_position(int pos);

    // ---- saved sequence state
    // The state of a sequence at length L is the "fixed" part -- every slot's convolution
    // inputs and the K/V rows of the last SWA - 1 positions of every sliding-window slot --
    // plus the K/V rows of the global slots, one KVD row each per position and slot.
    static constexpr int NGLOBAL = 8;   // global cache slots
    /// \brief bytes of the fixed part
    static size_t fixed_bytes();
    /// \brief bytes of one position's rows in the global slots (K and V)
    static size_t row_bytes() { return (size_t)NGLOBAL * 2 * KVD * sizeof(u16); }
    /// \brief copy the fixed part at the current position into dst (fixed_bytes())
    void save_fixed(u8* dst) const;
    /// \brief put back a fixed part saved at length `pos` and move there
    void load_fixed(const u8* src, int pos);
    /// \brief copy rows [from, from + n) of global slot j (0 .. NGLOBAL - 1) out of / into the cache
    void read_rows(int j, int from, int n, u16* k, u16* v) const;
    void write_rows(int j, int from, int n, const u16* k, const u16* v);
    /// \brief K or V of one position in one cache slot (global positions any; SWA the last 2048)
    void slot_row(int slot, int pos, bool v, u16* dst) const;
    void set_slot_row(int slot, int pos, bool v, const u16* src);
    /// \brief the convolution state of one cache slot: [NCONV - 1][CONV_CH], oldest first
    void slot_conv(int slot, float* dst) const;
    void set_slot_conv(int slot, const float* src);

    /// \brief run n tokens (any n; split into steps of at most CHUNK)
    /// \param logits if not null, [VOCAB] logits of the last token
    /// \param image_rows bf16 [image_count][D]: the rows that stand in for the tokens equal to image_id, in order, as
    ///        they are (no RMSNorm); null: those tokens are embedded like any other. Throws if the counts differ.
    void run(const int* ids, int n, float* logits, const u16* image_rows = nullptr, int image_count = 0, int image_id = -1);
    /// \brief the most tokens run() hands one step: CHUNK, or INKLING2_CHUNK
    static int chunk();

    /// \brief the weights (for the NPU stages and tests)
    const weights& w() const;
    /// \brief the weights, to move matrices into device buffers (qmat::data / grouped)
    weights& w_mut();
    /// \brief route the prompt matmuls through `o` (not owned; null: all on the host)
    void set_offload(mm_offload* o);

    struct state;   // caches + the sequence position
private:
    void step(const int* ids, int T, float* logits, const u16* image_rows = nullptr, int image_id = -1);
    std::unique_ptr<q4nx_file> file_;
    std::unique_ptr<weights> w_;
    std::unique_ptr<state> s_;
};

}  // namespace inkling2_host
