// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref_common.hpp
/// \brief Shared plumbing for the Qwen3.8-27B CPU reference layers.
///
/// Deliberately self-contained: only libstdc++ + POSIX mmap, so the two
/// cpu_ref_*.cpp files build and run without the XRT / npu_utils tree.
///
/// What lives here
///   * a minimal SafeTensors reader (works for both ref.safetensors and
///     model.q4nx -- q4nx *is* a safetensors file with extra dtypes),
///   * the FLM Q4_K block layout and a fused dequantise+matmul,
///   * the handful of activations / norms the two layers share,
///   * error_matrix(), mirroring models_simple/verify_qwen3_8.py.
///
/// Everything is fp32. This is a golden reference for the *math*, not a
/// bit-accurate model of the AIE datapath, so the expected agreement with
/// ref.safetensors is "Q4_K's own error floor", not 1e-6.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace cpu_ref {

// --------------------------------------------------------------------------
// Qwen3.8-27B geometry (config.json of Qwen/Qwen3.8-27B, revision 1d4bf0f)
// --------------------------------------------------------------------------
namespace q38 {
constexpr int MODEL_DIM         = 5120;
constexpr int NUM_HIDDEN_LAYERS = 64;
constexpr int INTERMEDIATE_SIZE = 17408;
// Which layers this xclbin is for. Layer i is full attention iff (i+1)%4 == 0,
// and those are ../../gated_attention_verify's; the other three in four are
// gated delta net and are what everything below describes.
constexpr int FULL_ATTN_INTERVAL = 4;

// gated delta net
constexpr int L_QK_HEADS   = 16;
constexpr int L_V_HEADS    = 48;
constexpr int L_EXPAND     = L_V_HEADS / L_QK_HEADS;        // 3
constexpr int L_K_HEAD_DIM = 128;
constexpr int L_V_HEAD_DIM = 128;
constexpr int L_DQ         = L_QK_HEADS * L_K_HEAD_DIM;     // 2048
constexpr int L_DK         = L_DQ;                          // 2048
constexpr int L_DV         = L_V_HEADS  * L_V_HEAD_DIM;     // 6144
constexpr int CONV_DIM     = L_DQ + L_DK + L_DV;            // 10240
constexpr int CONV_K       = 4;
const     float L_ATTN_SCALE = 1.0f / 11.313708498984761f;  // 1/sqrt(128)

constexpr float RMS_EPS  = 1e-6f;
constexpr float L2N_EPS  = 1e-6f;

inline bool is_full_attention(int layer) {
    return (layer + 1) % FULL_ATTN_INTERVAL == 0;
}
} // namespace q38

// --------------------------------------------------------------------------
// bf16
// --------------------------------------------------------------------------
inline float bf16_to_f32(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    // round-to-nearest-even, matching torch.Tensor.bfloat16()
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// --------------------------------------------------------------------------
// Minimal SafeTensors reader
//
//   | u64 header_len | header_len bytes of JSON | payload |
//
// The JSON is always a flat object of {name: {dtype, shape, data_offsets}},
// so a hand-rolled scanner is enough and keeps this header dependency-free.
// --------------------------------------------------------------------------
struct tensor_info {
    std::string          dtype;
    std::vector<int64_t> shape;
    size_t               off0 = 0, off1 = 0;   // relative to payload start

    int64_t numel() const {
        int64_t n = 1; for (int64_t s : shape) n *= s; return n;
    }
};

class safetensors {
public:
    explicit safetensors(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("fstat " + path);
        size_ = static_cast<size_t>(st.st_size);
        base_ = static_cast<const uint8_t*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (base_ == MAP_FAILED) throw std::runtime_error("mmap " + path);
        uint64_t hlen; std::memcpy(&hlen, base_, 8);
        payload_ = base_ + 8 + hlen;
        parse_header(reinterpret_cast<const char*>(base_ + 8), hlen);
    }

    ~safetensors() {
        if (base_ && base_ != MAP_FAILED) ::munmap(const_cast<uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    safetensors(const safetensors&) = delete;
    safetensors& operator=(const safetensors&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }

    const tensor_info& info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("no tensor '" + name + "' in " + path_);
        return it->second;
    }

    const uint8_t* raw(const std::string& name) const {
        return payload_ + info(name).off0;
    }

    size_t nbytes(const std::string& name) const {
        const tensor_info& t = info(name);
        return t.off1 - t.off0;
    }

    /// Read a BF16 / F32 / F16 tensor out as fp32.
    std::vector<float> f32(const std::string& name) const {
        const tensor_info& t = info(name);
        const size_t n = static_cast<size_t>(t.numel());
        std::vector<float> out(n);
        const uint8_t* p = payload_ + t.off0;
        if (t.dtype == "F32") {
            std::memcpy(out.data(), p, n * 4);
        } else if (t.dtype == "BF16") {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
            for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(h[i]);
        } else {
            throw std::runtime_error("f32(): unsupported dtype " + t.dtype +
                                     " for '" + name + "'");
        }
        return out;
    }

    std::vector<int32_t> i32(const std::string& name) const {
        const tensor_info& t = info(name);
        const size_t n = static_cast<size_t>(t.numel());
        std::vector<int32_t> out(n);
        std::memcpy(out.data(), payload_ + t.off0, n * 4);
        return out;
    }

    const std::map<std::string, tensor_info>& tensors() const { return tensors_; }

private:
    void parse_header(const char* p, size_t n) {
        size_t i = 0;
        auto ws = [&] { while (i < n && (p[i]==' '||p[i]=='\n'||p[i]=='\t'||p[i]=='\r')) i++; };
        auto str = [&]() -> std::string {
            ws();
            if (p[i] != '"') throw std::runtime_error("safetensors header: expected string");
            i++;
            std::string s;
            while (i < n && p[i] != '"') {
                if (p[i] == '\\' && i + 1 < n) { s.push_back(p[i+1]); i += 2; }
                else s.push_back(p[i++]);
            }
            i++;  // closing quote
            return s;
        };
        auto skip_value = [&]() {
            ws();
            if (p[i] == '"') { str(); return; }
            if (p[i] == '{' || p[i] == '[') {
                int depth = 0; bool in_str = false;
                for (; i < n; i++) {
                    char c = p[i];
                    if (in_str) { if (c == '\\') i++; else if (c == '"') in_str = false; continue; }
                    if (c == '"') in_str = true;
                    else if (c == '{' || c == '[') depth++;
                    else if (c == '}' || c == ']') { depth--; if (depth == 0) { i++; return; } }
                }
                return;
            }
            while (i < n && p[i] != ',' && p[i] != '}' && p[i] != ']') i++;
        };
        auto num = [&]() -> int64_t {
            ws();
            char* end = nullptr;
            long long v = std::strtoll(p + i, &end, 10);
            i = static_cast<size_t>(end - p);
            return v;
        };

        ws();
        if (p[i] != '{') throw std::runtime_error("safetensors header: expected object");
        i++;
        while (true) {
            ws();
            if (i >= n || p[i] == '}') break;
            std::string key = str();
            ws(); if (p[i] == ':') i++;
            if (key == "__metadata__") {
                skip_value();
            } else {
                ws();
                if (p[i] != '{') throw std::runtime_error("safetensors header: bad entry " + key);
                i++;
                tensor_info t;
                while (true) {
                    ws();
                    if (i >= n || p[i] == '}') { i++; break; }
                    std::string f = str();
                    ws(); if (p[i] == ':') i++;
                    if (f == "dtype") {
                        t.dtype = str();
                    } else if (f == "shape") {
                        ws(); i++;  // '['
                        while (true) {
                            ws();
                            if (p[i] == ']') { i++; break; }
                            t.shape.push_back(num());
                            ws(); if (p[i] == ',') i++;
                        }
                    } else if (f == "data_offsets") {
                        ws(); i++;  // '['
                        t.off0 = static_cast<size_t>(num()); ws(); if (p[i] == ',') i++;
                        t.off1 = static_cast<size_t>(num()); ws(); if (p[i] == ']') i++;
                    } else {
                        skip_value();
                    }
                    ws(); if (p[i] == ',') i++;
                }
                tensors_.emplace(std::move(key), std::move(t));
            }
            ws(); if (i < n && p[i] == ',') i++;
        }
    }

    std::string                        path_;
    int                                fd_     = -1;
    size_t                             size_   = 0;
    const uint8_t*                     base_   = nullptr;
    const uint8_t*                     payload_ = nullptr;
    std::map<std::string, tensor_info> tensors_;
};

// --------------------------------------------------------------------------
// FLM q4nx weight blocks
//
// Every quantised tensor in a .q4nx is a [row_blocks][col_blocks][block_bytes]
// array of fixed-size structs, one struct per 32-row x 256-column chunk (the
// converter's row_block_size x col_block_size with keep_block_in_2D). Which
// struct depends on the GGUF type it was packed from, and each type has a
// distinct byte size at this chunk shape, so the loader dispatches on shape[2]
// alone. g = (column within chunk)/32 is the group index; the uint4 planes pack
// a row pair per byte with the even row in the low nibble, and all three layouts
// store the quants column-major over the chunk, which is why a single
// dequant_block() shape (`tile[256][32]`) serves all of them.
//
//   Q4_K  4736 B   uint8 s[8][32] | uint8 m[8][32] | uint4 q[256][32]
//                  | bf16 S[32] | bf16 M[32]           (M is stored negated)
//                  w[r][c] = S[r]*s[g][r]*q + M[r]*m[g][r]
//   Q4_1  5120 B   bf16 d[8][32] | bf16 m[8][32] | uint4 q[256][32]
//                  w[r][c] = d[g][r]*q + m[g][r]
//   Q8_0  8704 B   bf16 d[8][32] | int8 q[256][32]
//                  w[r][c] = d[g][r]*q
//
// Mirrors q4k_block_t in the kernels' model_spec.h and _pack_q4k / _pack_q4nx /
// _pack_q8nx in FLM_Q4NX_Converter. Which projection uses which type is a
// per-model choice in FLM_Q4NX_Converter/configs/*.json -- Qwen3.5-9B defaults
// to Q4_1 while Qwen3.8-27B defaults to Q4_K -- hence the runtime dispatch.
// --------------------------------------------------------------------------
constexpr int Q4NX_ROW_BLOCK = 32;
constexpr int Q4NX_COL_BLOCK = 256;
constexpr int Q4NX_GROUP     = 32;
constexpr int Q4NX_GROUPS    = Q4NX_COL_BLOCK / Q4NX_GROUP;   // 8

constexpr int Q4K_BLOCK_BYTES = 4736;
constexpr int Q41_BLOCK_BYTES = 5120;
constexpr int Q80_BLOCK_BYTES = 8704;

enum class q4nx_format { q4_k, q4_1, q8_0 };

inline const char* format_name(q4nx_format f) {
    switch (f) {
        case q4nx_format::q4_k: return "Q4_K";
        case q4nx_format::q4_1: return "Q4_1";
        default:                return "Q8_0";
    }
}

struct q4nx_tensor {
    const uint8_t* blocks = nullptr;
    q4nx_format    fmt    = q4nx_format::q4_k;
    int rows = 0, cols = 0, nrb = 0, ncb = 0, block_bytes = 0;

    /// Dequantise one chunk into col-major `tile[256][32]` (the qs order).
    void dequant_block(int rb, int cb, float* tile) const {
        const uint8_t* b =
            blocks + (static_cast<size_t>(rb) * ncb + cb) * static_cast<size_t>(block_bytes);
        float sc[Q4NX_GROUPS][Q4NX_ROW_BLOCK], mn[Q4NX_GROUPS][Q4NX_ROW_BLOCK];
        const uint8_t* qs = nullptr;

        switch (fmt) {
            case q4nx_format::q4_k: {
                const uint8_t*  s8 = b;
                const uint8_t*  m8 = b + 256;
                const uint16_t* S  = reinterpret_cast<const uint16_t*>(b + 512 + 4096);
                const uint16_t* M  = reinterpret_cast<const uint16_t*>(b + 512 + 4096 + 64);
                qs = b + 512;
                for (int g = 0; g < Q4NX_GROUPS; g++)
                    for (int r = 0; r < Q4NX_ROW_BLOCK; r++) {
                        sc[g][r] = bf16_to_f32(S[r]) * static_cast<float>(s8[g * 32 + r]);
                        mn[g][r] = bf16_to_f32(M[r]) * static_cast<float>(m8[g * 32 + r]);
                    }
                break;
            }
            case q4nx_format::q4_1: {
                const uint16_t* d = reinterpret_cast<const uint16_t*>(b);
                const uint16_t* m = reinterpret_cast<const uint16_t*>(b + 512);
                qs = b + 1024;
                for (int g = 0; g < Q4NX_GROUPS; g++)
                    for (int r = 0; r < Q4NX_ROW_BLOCK; r++) {
                        sc[g][r] = bf16_to_f32(d[g * 32 + r]);
                        mn[g][r] = bf16_to_f32(m[g * 32 + r]);
                    }
                break;
            }
            case q4nx_format::q8_0: {
                const uint16_t* d = reinterpret_cast<const uint16_t*>(b);
                qs = b + 512;
                for (int g = 0; g < Q4NX_GROUPS; g++)
                    for (int r = 0; r < Q4NX_ROW_BLOCK; r++) {
                        sc[g][r] = bf16_to_f32(d[g * 32 + r]);
                        mn[g][r] = 0.0f;
                    }
                break;
            }
        }

        if (fmt == q4nx_format::q8_0) {
            const int8_t* q8 = reinterpret_cast<const int8_t*>(qs);
            for (int c = 0; c < Q4NX_COL_BLOCK; c++) {
                const int g = c >> 5;
                const int8_t* qc = q8 + c * Q4NX_ROW_BLOCK;
                float* dst = tile + c * Q4NX_ROW_BLOCK;
                for (int r = 0; r < Q4NX_ROW_BLOCK; r++)
                    dst[r] = sc[g][r] * static_cast<float>(qc[r]);
            }
        } else {
            for (int c = 0; c < Q4NX_COL_BLOCK; c++) {
                const int g = c >> 5;
                const uint8_t* qc = qs + c * (Q4NX_ROW_BLOCK / 2);
                float* dst = tile + c * Q4NX_ROW_BLOCK;
                for (int rp = 0; rp < Q4NX_ROW_BLOCK / 2; rp++) {
                    const uint8_t byte = qc[rp];
                    const int r0 = 2 * rp, r1 = r0 + 1;
                    dst[r0] = sc[g][r0] * static_cast<float>(byte & 0x0F) + mn[g][r0];
                    dst[r1] = sc[g][r1] * static_cast<float>(byte >> 4)   + mn[g][r1];
                }
            }
        }
    }
};

/// Look a quantised tensor up in a .q4nx file. Shape is [nrb, ncb, block_bytes];
/// the block size picks the format.
inline q4nx_tensor get_q4nx(const safetensors& f, const std::string& name) {
    const tensor_info& t = f.info(name);
    if (t.shape.size() != 3)
        throw std::runtime_error("'" + name + "' is not a [nrb,ncb,bytes] q4nx tensor");
    q4nx_tensor q;
    q.block_bytes = static_cast<int>(t.shape[2]);
    switch (q.block_bytes) {
        case Q4K_BLOCK_BYTES: q.fmt = q4nx_format::q4_k; break;
        case Q41_BLOCK_BYTES: q.fmt = q4nx_format::q4_1; break;
        case Q80_BLOCK_BYTES: q.fmt = q4nx_format::q8_0; break;
        default:
            throw std::runtime_error("'" + name + "': unhandled q4nx block size " +
                                     std::to_string(q.block_bytes));
    }
    q.blocks = f.raw(name);
    q.nrb    = static_cast<int>(t.shape[0]);
    q.ncb    = static_cast<int>(t.shape[1]);
    q.rows   = q.nrb * Q4NX_ROW_BLOCK;
    q.cols   = q.ncb * Q4NX_COL_BLOCK;
    return q;
}

/// out[l][m] = sum_c W[m][c] * x[l][c].   x: [L][W.cols], out: [L][W.rows].
///
/// Streams the weights: one 32x256 chunk is dequantised into 32 KB at a time and
/// consumed for all L tokens, so the 17 GB checkpoint never gets expanded in
/// memory. Parallel over row blocks, which is also how the AIE array partitions
/// the work (parallel_size = 32 rows).
inline void q4nx_matmul(const q4nx_tensor& W, const float* x, int L, float* out) {
    const int M = W.rows;
#pragma omp parallel for schedule(static)
    for (int rb = 0; rb < W.nrb; rb++) {
        std::vector<float> acc(static_cast<size_t>(L) * Q4NX_ROW_BLOCK, 0.0f);
        std::vector<float> tile(Q4NX_COL_BLOCK * Q4NX_ROW_BLOCK);
        for (int cb = 0; cb < W.ncb; cb++) {
            W.dequant_block(rb, cb, tile.data());
            for (int l = 0; l < L; l++) {
                const float* xl = x + static_cast<size_t>(l) * W.cols + cb * Q4NX_COL_BLOCK;
                float* al = acc.data() + static_cast<size_t>(l) * Q4NX_ROW_BLOCK;
                for (int c = 0; c < Q4NX_COL_BLOCK; c++) {
                    const float xv = xl[c];
                    if (xv == 0.0f) continue;
                    const float* tc = tile.data() + c * Q4NX_ROW_BLOCK;
                    for (int r = 0; r < Q4NX_ROW_BLOCK; r++) al[r] += tc[r] * xv;
                }
            }
        }
        for (int l = 0; l < L; l++)
            std::memcpy(out + static_cast<size_t>(l) * M + rb * Q4NX_ROW_BLOCK,
                        acc.data() + static_cast<size_t>(l) * Q4NX_ROW_BLOCK,
                        Q4NX_ROW_BLOCK * sizeof(float));
    }
}

/// Dense fp32 equivalent, for the small bf16 tensors (alpha/beta projections).
inline void dense_matmul(const float* W, int rows, int cols,
                         const float* x, int L, float* out) {
    for (int l = 0; l < L; l++) {
        const float* xl = x + static_cast<size_t>(l) * cols;
        float* ol = out + static_cast<size_t>(l) * rows;
        for (int m = 0; m < rows; m++) {
            const float* wm = W + static_cast<size_t>(m) * cols;
            float acc = 0.0f;
            for (int c = 0; c < cols; c++) acc += wm[c] * xl[c];
            ol[m] = acc;
        }
    }
}

// --------------------------------------------------------------------------
// activations / norms
// --------------------------------------------------------------------------
inline float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float siluf_(float x)    { return x * sigmoidf_(x); }

/// softplus, matching torch.nn.functional.softplus (threshold 20).
inline float softplusf_(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

/// Qwen3_5RMSNorm. NOTE: transformers computes `x*rsqrt(var+eps) * (1+w)`; the
/// GGUF -> q4nx path already folds the +1, so `w` here is the final multiplier.
inline void rms_norm(const float* x, const float* w, int n, float* y,
                     float eps = q38::RMS_EPS) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += x[i] * x[i];
    const float r = 1.0f / std::sqrt(acc / static_cast<float>(n) + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}

/// l2norm() of modeling_qwen3_5.py: x * rsqrt(sum(x^2) + eps).
inline void l2_norm(const float* x, int n, float* y, float extra_scale = 1.0f,
                    float eps = q38::L2N_EPS) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += x[i] * x[i];
    const float r = extra_scale / std::sqrt(acc + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r;
}

// --------------------------------------------------------------------------
// error_matrix(), same four numbers verify_qwen3_8.py prints
// --------------------------------------------------------------------------
struct err_stats { double rel_l1, rel_l2, cosine, rmse; };

inline err_stats error_matrix(const float* a, const float* b, size_t n) {
    double d1 = 0, r1 = 0, d2 = 0, r2 = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = a[i], y = b[i], d = x - y;
        d1 += std::fabs(d); r1 += std::fabs(y);
        d2 += d * d;        r2 += y * y;
        dot += x * y; na += x * x; nb += y * y;
    }
    err_stats s;
    s.rel_l1 = r1 > 0 ? d1 / r1 : 0.0;
    s.rel_l2 = r2 > 0 ? std::sqrt(d2 / r2) : 0.0;
    s.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    s.rmse   = std::sqrt(d2 / static_cast<double>(n));
    return s;
}

inline void print_err(const char* tag, const err_stats& s) {
    std::printf("  %-24s relL1 %.5f   relL2 %.5f   cos %.6f   rmse %.6f\n",
                tag, s.rel_l1, s.rel_l2, s.cosine, s.rmse);
}

/// Compare row by row as well as in aggregate -- a mask or ordering bug shows
/// up as one bad row long before it moves the aggregate cosine.
inline void compare_rows(const char* tag, const float* got, const float* ref,
                         int L, int D) {
    std::printf("%s\n", tag);
    for (int t = 0; t < L; t++) {
        err_stats s = error_matrix(got + static_cast<size_t>(t) * D,
                                   ref + static_cast<size_t>(t) * D, D);
        char buf[64]; std::snprintf(buf, sizeof buf, "token %d", t);
        print_err(buf, s);
    }
    print_err("ALL", error_matrix(got, ref, static_cast<size_t>(L) * D));
}

/// Both residual streams pass `x` straight through, so comparing y against the
/// reference y flatters the result: the identity path is exact by construction
/// and dilutes whatever the block actually computed. Subtracting the input
/// leaves mixer+MLP alone, which is the number that catches a wrong head order,
/// a stale conv window or an off-by-one mask.
inline void compare_residual_free(const char* tag, const float* got,
                                  const float* ref, const float* in,
                                  int L, int D) {
    const size_t n = static_cast<size_t>(L) * D;
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; i++) { a[i] = got[i] - in[i]; b[i] = ref[i] - in[i]; }
    compare_rows(tag, a.data(), b.data(), L, D);
}

} // namespace cpu_ref
