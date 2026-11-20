// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file cpu_ref.cpp
/// \brief fp32 CPU reference for one Qwen3.5 decode step; see README.md
#include "cpu_ref.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace cpu_ref {

namespace {

constexpr int ROW_BLOCK = 32;
constexpr int COL_BLOCK = 256;
constexpr int GROUP = 32;
constexpr int Q4K_CHUNK_BYTES = 4736;
constexpr int Q80_CHUNK_BYTES = 8704;
constexpr float EPS = 1e-6f;

std::string layer_name(int layer, const char* suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

float bf16_at(const uint8_t* p, int index) {
    bf16 v;
    memcpy(&v, p + index * sizeof(bf16), sizeof(bf16));
    return (float)v;
}

void dequant_q4k_chunk(const uint8_t* chunk, float* dst, int row0, int col0, int cols) {
    const uint8_t* scales = chunk;
    const uint8_t* mins = chunk + 256;
    const uint8_t* qs = chunk + 512;
    const uint8_t* super_scale = chunk + 4608;
    const uint8_t* super_min = chunk + 4672;
    for (int r = 0; r < ROW_BLOCK; r++) {
        float S = bf16_at(super_scale, r);
        float M = bf16_at(super_min, r);
        for (int c = 0; c < COL_BLOCK; c++) {
            int g = c / GROUP;
            uint8_t byte = qs[c * (ROW_BLOCK / 2) + r / 2];
            int q = (r % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float w = S * scales[g * ROW_BLOCK + r] * q + M * mins[g * ROW_BLOCK + r];
            dst[(size_t)(row0 + r) * cols + col0 + c] = w;
        }
    }
}

void dequant_q80_chunk(const uint8_t* chunk, float* dst, int row0, int col0, int cols, int rows_kept) {
    const uint8_t* scales = chunk;
    const int8_t* data = reinterpret_cast<const int8_t*>(chunk + 512);
    for (int r = 0; r < ROW_BLOCK && row0 + r < rows_kept; r++) {
        for (int c = 0; c < COL_BLOCK; c++) {
            float scale = bf16_at(scales, (c / GROUP) * ROW_BLOCK + r);
            dst[(size_t)(row0 + r) * cols + col0 + c] = scale * data[c * ROW_BLOCK + r];
        }
    }
}

matrix_t load_q4k(Q4NX& q4nx, const std::string& name, int rows, int cols) {
    buffer<u8> raw;
    q4nx.load_weights(raw, name);
    matrix_t m{rows, cols, vec((size_t)rows * cols)};
    int col_blocks = cols / COL_BLOCK;
    for (int p = 0; p < rows / ROW_BLOCK; p++) {
        for (int q = 0; q < col_blocks; q++) {
            const uint8_t* chunk = raw.data() + ((size_t)p * col_blocks + q) * Q4K_CHUNK_BYTES;
            dequant_q4k_chunk(chunk, m.w.data(), p * ROW_BLOCK, q * COL_BLOCK, cols);
        }
    }
    return m;
}

matrix_t load_q80(Q4NX& q4nx, const std::string& name, int rows_kept, int cols) {
    buffer<u8> raw;
    q4nx.load_weights(raw, name);
    matrix_t m{rows_kept, cols, vec((size_t)rows_kept * cols)};
    int col_blocks = cols / COL_BLOCK;
    int row_blocks = (rows_kept + ROW_BLOCK - 1) / ROW_BLOCK;
    for (int p = 0; p < row_blocks; p++) {
        for (int q = 0; q < col_blocks; q++) {
            const uint8_t* chunk = raw.data() + ((size_t)p * col_blocks + q) * Q80_CHUNK_BYTES;
            dequant_q80_chunk(chunk, m.w.data(), p * ROW_BLOCK, q * COL_BLOCK, cols, rows_kept);
        }
    }
    return m;
}

vec load_bf16(Q4NX& q4nx, const std::string& name) {
    buffer<bf16> raw;
    q4nx.load_weights(raw, name);
    vec v(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        v[i] = (float)raw[i];
    }
    return v;
}

vec load_f32(Q4NX& q4nx, const std::string& name) {
    buffer<float> raw;
    q4nx.load_weights(raw, name);
    return vec(raw.data(), raw.data() + raw.size());
}

vec matvec(const matrix_t& m, const vec& x) {
    vec y(m.rows);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < m.rows; r++) {
        const float* row = m.w.data() + (size_t)r * m.cols;
        double acc = 0;
        for (int c = 0; c < m.cols; c++) {
            acc += (double)row[c] * x[c];
        }
        y[r] = (float)acc;
    }
    return y;
}

void rms_norm_inplace(float* x, int n, const float* w) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i] * x[i];
    }
    float inv = 1.0f / std::sqrt((float)(sum / n) + EPS);
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * inv * w[i];
    }
}

vec rms_norm(const vec& x, const vec& w) {
    vec y = x;
    rms_norm_inplace(y.data(), (int)y.size(), w.data());
    return y;
}

void l2_norm_inplace(float* x, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i] * x[i];
    }
    float inv = 1.0f / std::sqrt((float)sum + EPS);
    for (int i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }
float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }
float softplus(float v) { return v > 20.0f ? v : std::log1p(std::exp(v)); }

vec add(const vec& a, const vec& b) {
    vec y(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        y[i] = a[i] + b[i];
    }
    return y;
}

vec mlp_block(const vec& x, const vec& rms_post, const matrix_t& up, const matrix_t& gate, const matrix_t& down) {
    vec h = rms_norm(x, rms_post);
    vec u = matvec(up, h);
    vec g = matvec(gate, h);
    for (size_t i = 0; i < u.size(); i++) {
        u[i] = silu(g[i]) * u[i];
    }
    return add(x, matvec(down, u));
}

void apply_rope(float* head, int rope_dim, int position) {
    int half = rope_dim / 2;
    for (int j = 0; j < half; j++) {
        float angle = qwen3_5vl_desc::inv_freq[j] * position;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float x1 = head[j];
        float x2 = head[half + j];
        head[j] = x1 * c - x2 * s;
        head[half + j] = x2 * c + x1 * s;
    }
}

vec conv_step(const qwen3_5vl_desc& desc, const gdn_weights_t& w, gdn_state_t& state, const vec& qkv) {
    int C = desc.Linear_DQKV;
    int K = desc.conv_kernel_size;
    vec y(C);
    for (int c = 0; c < C; c++) {
        float acc = w.conv[(size_t)(K - 1) * C + c] * qkv[c];
        for (int k = 0; k < K - 1; k++) {
            acc += w.conv[(size_t)k * C + c] * state.conv[(size_t)k * C + c];
        }
        y[c] = silu(acc);
    }
    for (int k = 0; k < K - 2; k++) {
        memcpy(&state.conv[(size_t)k * C], &state.conv[(size_t)(k + 1) * C], C * sizeof(float));
    }
    memcpy(&state.conv[(size_t)(K - 2) * C], qkv.data(), C * sizeof(float));
    return y;
}

void delta_rule_head(float* S, const float* q, const float* k, const float* v, float decay, float beta,
                     int dk, int dv, float* o) {
    std::vector<float> kv_mem(dv, 0.0f);
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] *= decay;
            kv_mem[j] += S[i * dv + j] * k[i];
        }
    }
    for (int j = 0; j < dv; j++) {
        kv_mem[j] = (v[j] - kv_mem[j]) * beta;
        o[j] = 0.0f;
    }
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] += k[i] * kv_mem[j];
            o[j] += S[i * dv + j] * q[i];
        }
    }
}

float round_to_e4m3(float v) {
    float a = std::fabs(v);
    if (a == 0.0f) {
        return 0.0f;
    }
    a = std::min(a, 448.0f);
    int e = (int)std::floor(std::log2(a));
    float step = (e < -6) ? std::ldexp(1.0f, -9) : std::ldexp(1.0f, e - 3);
    float r = std::min(std::nearbyint(a / step) * step, 448.0f);
    return v < 0 ? -r : r;
}

void quantize_head(float* x, int n, kv_quant_t q) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        max_abs = std::max(max_abs, std::fabs(x[i]));
    }
    if (max_abs == 0.0f) {
        return;
    }
    float qmax = (q == kv_quant_t::int8) ? 127.0f : 448.0f;
    float scale = max_abs / qmax;
    for (int i = 0; i < n; i++) {
        float v = x[i] / scale;
        v = (q == kv_quant_t::int8) ? std::clamp(std::nearbyint(v), -127.0f, 127.0f) : round_to_e4m3(v);
        x[i] = v * scale;
    }
}

}  // namespace

const char* kv_quant_name(kv_quant_t q) {
    return q == kv_quant_t::int8 ? "int8" : (q == kv_quant_t::fp8 ? "fp8" : "bf16");
}

void quantize_rows(float* data, size_t rows, int heads, int head_dim, kv_quant_t q) {
    if (q == kv_quant_t::none) {
        return;
    }
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++) {
        for (int h = 0; h < heads; h++) {
            quantize_head(data + (r * heads + h) * head_dim, head_dim, q);
        }
    }
}

vec embed_token(const buffer<u8>& lm_head_q80, int token, int D) {
    vec row(D);
    int col_blocks = D / COL_BLOCK;
    int p = token / ROW_BLOCK;
    int r = token % ROW_BLOCK;
    for (int q = 0; q < col_blocks; q++) {
        const uint8_t* chunk = lm_head_q80.data() + ((size_t)p * col_blocks + q) * Q80_CHUNK_BYTES;
        const int8_t* data = reinterpret_cast<const int8_t*>(chunk + 512);
        for (int c = 0; c < COL_BLOCK; c++) {
            float scale = bf16_at(chunk, (c / GROUP) * ROW_BLOCK + r);
            row[q * COL_BLOCK + c] = scale * data[c * ROW_BLOCK + r];
        }
    }
    return row;
}

vec lm_head_logits(const buffer<u8>& lm_head_q80, const vec& x, const vec& norm, int rows) {
    const int D = (int)x.size();
    const int col_blocks = D / COL_BLOCK;
    const int stored_row_blocks = (int)(lm_head_q80.size() / Q80_CHUNK_BYTES / col_blocks);
    const int row_blocks = std::min(rows / ROW_BLOCK, stored_row_blocks);
    const vec xn = rms_norm(x, norm);
    vec logits(rows, 0.0f);
#pragma omp parallel for
    for (int p = 0; p < row_blocks; p++) {
        for (int q = 0; q < col_blocks; q++) {
            const uint8_t* chunk = lm_head_q80.data() + ((size_t)p * col_blocks + q) * Q80_CHUNK_BYTES;
            const int8_t* data = reinterpret_cast<const int8_t*>(chunk + 512);
            const float* x_chunk = xn.data() + q * COL_BLOCK;
            for (int r = 0; r < ROW_BLOCK; r++) {
                float sum = 0;
                for (int g = 0; g < COL_BLOCK / GROUP; g++) {
                    float group_sum = 0;
                    for (int t = 0; t < GROUP; t++) {
                        int c = g * GROUP + t;
                        group_sum += data[c * ROW_BLOCK + r] * x_chunk[c];
                    }
                    sum += bf16_at(chunk, g * ROW_BLOCK + r) * group_sum;
                }
                logits[p * ROW_BLOCK + r] += sum;
            }
        }
    }
    return logits;
}

gdn_weights_t load_gdn_weights(Q4NX& q4nx, const qwen3_5vl_desc& desc, int layer) {
    int D = desc.D;
    gdn_weights_t w;
    w.rms_pre = load_bf16(q4nx, layer_name(layer, "input_layernorm.weight"));
    w.rms_post = load_bf16(q4nx, layer_name(layer, "post_attention_layernorm.weight"));
    w.conv = load_bf16(q4nx, layer_name(layer, "linear_attn.ssm_conv1d.weight"));
    w.norm = load_bf16(q4nx, layer_name(layer, "linear_attn.ssm_norm.weight"));
    w.a = load_f32(q4nx, layer_name(layer, "linear_attn.ssm_a"));
    w.dt = load_f32(q4nx, layer_name(layer, "linear_attn.ssm_dt.bias"));
    w.qkv = load_q4k(q4nx, layer_name(layer, "linear_attn.qkv_proj.weight"), desc.Linear_DQKV, D);
    w.z = load_q4k(q4nx, layer_name(layer, "self_attn.gate_proj.weight"), desc.Linear_DV, D);
    w.out = load_q4k(q4nx, layer_name(layer, "linear_attn.ssm_out_proj.weight"), D, desc.Linear_DV);
    w.alpha = load_q80(q4nx, layer_name(layer, "linear_attn.ssm_alpha_proj.weight"), desc.linear_num_v_heads, D);
    w.beta = load_q80(q4nx, layer_name(layer, "linear_attn.ssm_beta_proj.weight"), desc.linear_num_v_heads, D);
    w.up = load_q4k(q4nx, layer_name(layer, "mlp.up_proj.weight"), desc.intermediate_size, D);
    w.gate = load_q4k(q4nx, layer_name(layer, "mlp.gate_proj.weight"), desc.intermediate_size, D);
    w.down = load_q4k(q4nx, layer_name(layer, "mlp.down_proj.weight"), D, desc.intermediate_size);
    return w;
}

attn_weights_t load_attn_weights(Q4NX& q4nx, const qwen3_5vl_desc& desc, int layer) {
    int D = desc.D;
    attn_weights_t w;
    w.rms_pre = load_bf16(q4nx, layer_name(layer, "input_layernorm.weight"));
    w.rms_post = load_bf16(q4nx, layer_name(layer, "post_attention_layernorm.weight"));
    w.q_norm = load_bf16(q4nx, layer_name(layer, "self_attn.q_norm.weight"));
    w.k_norm = load_bf16(q4nx, layer_name(layer, "self_attn.k_norm.weight"));
    w.q = load_q4k(q4nx, layer_name(layer, "self_attn.q_proj.weight"), 2 * desc.DQ, D);
    w.k = load_q4k(q4nx, layer_name(layer, "self_attn.k_proj.weight"), desc.DK, D);
    w.v = load_q4k(q4nx, layer_name(layer, "self_attn.v_proj.weight"), desc.DV, D);
    w.o = load_q4k(q4nx, layer_name(layer, "self_attn.o_proj.weight"), D, desc.DQ);
    w.up = load_q4k(q4nx, layer_name(layer, "mlp.up_proj.weight"), desc.intermediate_size, D);
    w.gate = load_q4k(q4nx, layer_name(layer, "mlp.gate_proj.weight"), desc.intermediate_size, D);
    w.down = load_q4k(q4nx, layer_name(layer, "mlp.down_proj.weight"), D, desc.intermediate_size);
    return w;
}

vec gdn_layer(const qwen3_5vl_desc& desc, const gdn_weights_t& w, gdn_state_t& state, const vec& x) {
    int H = desc.linear_num_v_heads;
    int dk = desc.linear_qk_head_dim;
    int dv = desc.linear_v_head_dim;
    vec h = rms_norm(x, w.rms_pre);
    vec qkv = matvec(w.qkv, h);
    vec z = matvec(w.z, h);
    vec a = matvec(w.alpha, h);
    vec b = matvec(w.beta, h);
    vec mixed = conv_step(desc, w, state, qkv);
    float* q = mixed.data();
    float* k = mixed.data() + desc.Linear_DQ;
    float* v = mixed.data() + desc.Linear_DQ + desc.Linear_DK;
    vec o((size_t)H * dv);
    float q_scale = 1.0f / std::sqrt((float)dk);
    for (int head = 0; head < H; head++) {
        float* qh = q + head * dk;
        float* kh = k + head * dk;
        l2_norm_inplace(qh, dk);
        l2_norm_inplace(kh, dk);
        for (int i = 0; i < dk; i++) {
            qh[i] *= q_scale;
        }
        float beta = sigmoid(b[head]);
        float decay = std::exp(w.a[head] * softplus(a[head] + w.dt[head]));
        float* oh = o.data() + head * dv;
        delta_rule_head(state.S.data() + (size_t)head * dk * dv, qh, kh, v + head * dv, decay, beta, dk, dv, oh);
        rms_norm_inplace(oh, dv, w.norm.data());
        for (int j = 0; j < dv; j++) {
            oh[j] *= silu(z[head * dv + j]);
        }
    }
    vec x1 = add(x, matvec(w.out, o));
    return mlp_block(x1, w.rms_post, w.up, w.gate, w.down);
}

vec attn_layer(const qwen3_5vl_desc& desc, const attn_weights_t& w, const float* k_cache, const float* v_cache,
               int context_len, int position, const vec& x, attn_step_t& step, kv_quant_t quant) {
    int NH = desc.num_heads;
    int NKV = desc.num_kv_heads;
    int DH = desc.DH;
    int group = NH / NKV;
    vec h = rms_norm(x, w.rms_pre);
    vec qg = matvec(w.q, h);
    vec q(qg.begin(), qg.begin() + desc.DQ);
    vec gate(qg.begin() + desc.DQ, qg.end());
    step.k_new = matvec(w.k, h);
    step.v_new = matvec(w.v, h);
    for (int head = 0; head < NH; head++) {
        rms_norm_inplace(q.data() + head * DH, DH, w.q_norm.data());
        apply_rope(q.data() + head * DH, desc.RoPE_DIM, position);
    }
    for (int head = 0; head < NKV; head++) {
        rms_norm_inplace(step.k_new.data() + head * DH, DH, w.k_norm.data());
        apply_rope(step.k_new.data() + head * DH, desc.RoPE_DIM, position);
    }
    vec k_new = step.k_new;
    vec v_new = step.v_new;
    quantize_rows(k_new.data(), 1, NKV, DH, quant);
    quantize_rows(v_new.data(), 1, NKV, DH, quant);
    int total = context_len + 1;
    float scale = 1.0f / std::sqrt((float)DH);
    vec attn((size_t)NH * DH, 0.0f);
    #pragma omp parallel for schedule(static)
    for (int head = 0; head < NH; head++) {
        int kv = head / group;
        const float* qh = q.data() + head * DH;
        std::vector<double> scores(total);
        double max_score = -1e30;
        for (int t = 0; t < total; t++) {
            double s = 0;
            for (int d = 0; d < DH; d++) {
                float kd = (t < context_len) ? k_cache[(size_t)t * desc.DK + (size_t)kv * DH + d] : k_new[kv * DH + d];
                s += (double)qh[d] * kd;
            }
            scores[t] = s * scale;
            max_score = std::max(max_score, scores[t]);
        }
        double sum = 0;
        for (int t = 0; t < total; t++) {
            scores[t] = std::exp(scores[t] - max_score);
            sum += scores[t];
        }
        float* out = attn.data() + head * DH;
        for (int t = 0; t < total; t++) {
            float p = (float)(scores[t] / sum);
            for (int d = 0; d < DH; d++) {
                float vd = (t < context_len) ? v_cache[(size_t)t * desc.DV + (size_t)kv * DH + d] : v_new[kv * DH + d];
                out[d] += p * vd;
            }
        }
    }
    step.core_out = attn;
    for (size_t i = 0; i < attn.size(); i++) {
        attn[i] *= sigmoid(gate[i]);
    }
    vec x1 = add(x, matvec(w.o, attn));
    return mlp_block(x1, w.rms_post, w.up, w.gate, w.down);
}

}  // namespace cpu_ref
