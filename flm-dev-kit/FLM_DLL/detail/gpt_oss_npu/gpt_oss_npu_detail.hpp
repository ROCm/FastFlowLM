// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include <algorithm>
#include "models/gpt_oss/gpt_oss_npu.hpp"
#include "utils/kv_rows_state.hpp"
#include "gpt_oss_params.hpp"
#include "gpt_dequant_q4_1.hpp"
#include "gpt_dequant_mxfp4.hpp"
#include "gpt_oss_gemm.hpp"
#include "gpt_expert_mm.hpp"
#include "gpt_router.hpp"
#include "gpt_mlp.hpp"
#include <thread>
#include <future>
#include <cmath>
#include <omp.h>
#include <unordered_map>

#if 1
inline __m256 _fflm_fast_tanh_ps(__m256 x) {
    // Clamp to improve stability for large magnitudes
    const __m256 clamp_hi = _mm256_set1_ps(5.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-5.0f);
    x = _mm256_min_ps(clamp_hi, _mm256_max_ps(clamp_lo, x));

    // tanh approximation: x * (27 + x^2) / (27 + 9*x^2)
    const __m256 c27 = _mm256_set1_ps(27.0f);
    const __m256 c9 = _mm256_set1_ps(9.0f);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 num = _mm256_mul_ps(x, _mm256_add_ps(c27, x2));
    __m256 den = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));
    return _mm256_div_ps(num, den);
}
#endif

inline void GeLU(buffer<bf16>& data, int length) {
    int len = std::min<int>(length, static_cast<int>(data.size()));
    if (len <= 0) return;

#if 1
    bf16* ptr = data.data();
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c0 = _mm256_set1_ps(0.7978845608028654f); // sqrt(2/pi)
    const __m256 c1 = _mm256_set1_ps(0.044715f);

    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m128i bf16_vals0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
        __m128i bf16_vals1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i + 8));

        __m256 x0 = bf16o_fp32(bf16_vals0);
        __m256 x1 = bf16o_fp32(bf16_vals1);

        __m256 x0_2 = _mm256_mul_ps(x0, x0);
        __m256 x1_2 = _mm256_mul_ps(x1, x1);
        __m256 x0_3 = _mm256_mul_ps(x0_2, x0);
        __m256 x1_3 = _mm256_mul_ps(x1_2, x1);

        __m256 inner0 = _mm256_add_ps(x0, _mm256_mul_ps(c1, x0_3));
        __m256 inner1 = _mm256_add_ps(x1, _mm256_mul_ps(c1, x1_3));

        __m256 t0 = _mm256_mul_ps(c0, inner0);
        __m256 t1 = _mm256_mul_ps(c0, inner1);

        __m256 th0 = _fflm_fast_tanh_ps(t0);
        __m256 th1 = _fflm_fast_tanh_ps(t1);

        __m256 y0 = _mm256_mul_ps(half, _mm256_mul_ps(x0, _mm256_add_ps(one, th0)));
        __m256 y1 = _mm256_mul_ps(half, _mm256_mul_ps(x1, _mm256_add_ps(one, th1)));

        __m128i out_bf16_0 = f32o_bf16(y0);
        __m128i out_bf16_1 = f32o_bf16(y1);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16_0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i + 8), out_bf16_1);
    }

    for (; i + 8 <= len; i += 8) {
        __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
        __m256 x = bf16o_fp32(bf16_vals);

        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 inner = _mm256_add_ps(x, _mm256_mul_ps(c1, x3));
        __m256 t = _mm256_mul_ps(c0, inner);
        __m256 th = _fflm_fast_tanh_ps(t);
        __m256 y = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, th)));

        __m128i out_bf16 = f32o_bf16(y);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16);
    }

    for (; i < len; ++i) {
        float x = static_cast<float>(ptr[i]);
        float y = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        ptr[i] = bf16(y);
    }
#else
    for (int i = 0; i < len; ++i) {
        float x = static_cast<float>(data[i]);
        float y = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        data[i] = bf16(y);
    }
#endif
}

inline void Clamp(buffer<bf16>& data, int length, float min_val, float max_val) {
    int len = std::min<int>(length, static_cast<int>(data.size()));
    if (len <= 0) return;

#if 1
    bf16* ptr = data.data();
    const __m256 minv = _mm256_set1_ps(min_val);
    const __m256 maxv = _mm256_set1_ps(max_val);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
        __m256 x = bf16o_fp32(bf16_vals);
        x = _mm256_max_ps(minv, _mm256_min_ps(maxv, x));
        __m128i out_bf16 = f32o_bf16(x);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16);
    }

    for (; i < len; ++i) {
        float x = static_cast<float>(ptr[i]);
        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;
        ptr[i] = bf16(x);
    }
#else
    for (int i = 0; i < len; ++i) {
        float x = static_cast<float>(data[i]);
        if (x < min_val) x = min_val;
        if (x > max_val) x = max_val;
        data[i] = bf16(x);
    }
#endif
}

inline void Inc(buffer<bf16>& data, int length) {
    int len = std::min<int>(length, static_cast<int>(data.size()));
    if (len <= 0) return;

#if 1
    bf16* ptr = data.data();
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
        __m256 x = bf16o_fp32(bf16_vals);
        x = _mm256_add_ps(x, one);
        __m128i out_bf16 = f32o_bf16(x);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16);
    }

    for (; i < len; ++i) {
        float x = static_cast<float>(ptr[i]);
        ptr[i] = bf16(x + 1.0f);
    }
#else
    for (int i = 0; i < len; ++i) {
        float x = static_cast<float>(data[i]);
        data[i] = bf16(x + 1.0f);
    }
#endif
}
// #define VERBOSE 1
struct gpt_oss_npu::Impl{
    /// @brief constexprs
    static constexpr int xy_group_id = 3;
    static constexpr int w_proj_group_id = 4;
    static constexpr int w_rms_group_id = 5;
    static constexpr int w_rope_group_id = 6;
    static constexpr int kv_cache_group_id = 7;

    static constexpr int mha_y_group_id = 3;
    static constexpr int mha_q_group_id = 4;
    static constexpr int mha_kv_group_id = 5;

    static constexpr float inv_freq[32] = {
        1.0000000000e+00, 6.8904429674e-01, 4.7478204966e-01, 3.2714587450e-01, 2.2541800141e-01, 1.5532298386e-01, 1.0702442378e-01, 7.3744565248e-02,
        5.0813272595e-02, 3.1705696136e-02, 1.9334999844e-02, 1.1592049152e-02, 6.7949593067e-03, 3.8603590801e-03, 2.0937926602e-03, 1.0526021942e-03,
        4.5648391824e-04, 1.2931869423e-04, 3.8308811781e-05, 2.6396468456e-05, 1.8188336981e-05, 1.2532569599e-05, 8.6354957602e-06, 5.9502394834e-06,
        4.0999784687e-06, 2.8250667583e-06, 1.9465962851e-06, 1.3412909539e-06, 9.2420896181e-07, 6.3682091422e-07, 4.3879785494e-07, 3.0235113968e-07
    };

    static constexpr int LC = 32;
    /// @brief  initialize the gpt_oss_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    std::string model_path;
    /// @brief forward the gpt_oss_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> forward(buffer<bf16>& x);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    buffer<bf16> _prefill_with_mv(std::vector<int>& ids);
    buffer<bf16> _prefill_with_mm(std::vector<int>& ids);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();
    int checkpoint();
    int restore();
    void _allocate_checkpoint_buffers();

    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// @brief one flag per layer: global attention (keeps K/V rows) or sliding window (keeps a ring)
    std::vector<bool> attn_mask;
    /// @brief highest context length reached since the global rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    /// @brief sliding_temp_cache holds what an earlier prefill left there (it is a scratch rebuilt from the rings, but
    ///        the rows the attention mask lets through at its edges are not rewritten)
    bool temp_dirty = false;
    void _zero_temp_cache();
    bool is_checkpoint_valid;
    int checkpoint_context_length;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;

    size_t sliding_K03_offset;
    size_t sliding_K47_offset;
    size_t sliding_V03_offset;
    size_t sliding_V47_offset;

    typedef enum: int {
        use_cpu_mm = 0,
        use_npu_short_mm,
        use_npu_large_mm
    } expert_mm_mode_t;

    LM_Config config;
    npu_xclbin_manager *npu;

    void _set_rope_weights(int idx);
    inline void _rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight, int length = 0);
    inline void _rope(buffer<bf16>& x, int idx);
    inline void _residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual, int length = 0);
    inline bool is_sliding_window(int layer_idx) {return layer_idx % 2 == 0;}
    inline void _cpu_expert_task(
        int layer_idx,
        int expert_id,
        int L_this_expert,
        std::vector<std::pair<int, bf16>>& expert_to_index_map,
        buffer<bf16>& expert_buffer_x,
        tensor_2d<bf16>& hidden_states_tensor,
        buffer<bf16>& expert_buffer_gate_result,
        buffer<bf16>& expert_buffer_up_result
    );
    inline void _npu_expert_task(
        int layer_idx,
        int expert_id,
        int L_this_expert,
        std::vector<std::pair<int, bf16>>& expert_to_index_map,
        buffer<bf16>& expert_buffer_x,
        tensor_2d<bf16>& hidden_states_tensor,
        buffer<bf16>& expert_buffer_gate_result,
        buffer<bf16>& expert_buffer_up_result
    );
    inline void _npu_short_expert_task(
        int layer_idx,
        int expert_id,
        int L_this_expert,
        std::vector<std::pair<int, bf16>>& expert_to_index_map,
        buffer<bf16>& expert_buffer_x,
        tensor_2d<bf16>& hidden_states_tensor,
        buffer<bf16>& expert_buffer_gate_result,
        buffer<bf16>& expert_buffer_up_result
    );
    inline void _npu_large_expert_task(
        int layer_idx,
        int expert_id,
        int L_this_expert,
        std::vector<std::pair<int, bf16>>& expert_to_index_map,
        buffer<bf16>& expert_buffer_x,
        tensor_2d<bf16>& hidden_states_tensor,
        buffer<bf16>& expert_buffer_gate_result,
        buffer<bf16>& expert_buffer_up_result
    );
    inline void _clamp_gelu(buffer<bf16>& data, float min_val, float max_val, int rows, int length, int row_size);
    inline void _clamp_inc(buffer<bf16>& data, float min_val, float max_val, int rows, int length, int row_size);
    inline void _expert_mac(buffer<float>& acc, bf16* x, bf16 score, int length);
    inline void _elementwise_mul(bf16* dest, bf16* src1, bf16* src2, int length);
    inline void _export_to_bf16(buffer<bf16>& dest, buffer<float>& src, int length = 0);
    void _cpu_mm(bf16* y_out, const bf16* x, const bf16* w, const bf16* b, int L, int Din, int Dout, int y_row_size, int x_row_size, int w_row_size);
    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx);
    void fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int sliding_l_end, int layer_idx);
    inline int linear2ring(int l) {return l % 128;} // & 0x7F is the same as % 128

    npu_app_manager *layer_app_manager;
    npu_app_manager *gemm_app_manager;
    npu_app_manager *short_seq_gemm_app_manager;
    npu_app_manager *dequant_attn_app_manager;
    npu_app_manager *dequant_mlp_app_manager;
    npu_app_manager *mha_app_manager;
    npu_app_manager *expert_mm_app_manager;

    npu_app layer;
    npu_app sliding_layer;
    npu_app dequant_attn_qkv;
    npu_app dequant_attn_o;
    npu_app dequant_mlp;
    npu_app mha_engine;
    npu_app expert_mm;
    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;

    npu_app up_proj;
    npu_app gate_proj;
    npu_app down_proj;

    npu_app short_seq_up_proj;
    npu_app short_seq_gate_proj;
    npu_app short_seq_down_proj;

    npu_app lm_head;

    // Regeneration-skip (builds on the Shared-Executable Collapse). Because the
    // per-expert weight/bias base is now a runtime binding offset rather than
    // baked into the TXN, each expert-MM app's control code is a pure function of
    // its fixed (proj, N, M, mode) and the padded length Lp -- independent of
    // expert_id and layer_idx. So if the next dispatch on the same app has the
    // same Lp, generate_seq would reproduce byte-identical control code; we skip
    // it and create_run reuses the already-built executable with no host-side
    // regeneration. Safe under the single-app ping-pong because each run captures
    // its own executable at create_run time, so an in-flight run is never bound to
    // the mutable host seq buffer.
    std::unordered_map<const npu_app*, int> _expert_last_lp;
    template <typename GemmT, typename... Args>
    inline void _gen_expert_seq_if_needed(GemmT* g, npu_app& app, int Lp, Args&&... args) {
        auto it = _expert_last_lp.find(&app);
        if (it != _expert_last_lp.end() && it->second == Lp) return;
        g->generate_seq(app.seq(), Lp, std::forward<Args>(args)...);
        _expert_last_lp[&app] = Lp;
    }

    std::unique_ptr<gpt_oss_npu_sequence> sequence;
    std::unique_ptr<GPT_OSS_Gemm> gemm;
    std::unique_ptr<GPT_OSS_Gemm> short_seq_gemm;

    buffer<bf16> sliding_temp_cache;

    uint32_t MAX_L;
    uint32_t num_hidden_layers;
    // size
    std::vector<buffer<bf16>> rms_o_bias_weights;
    buffer<bf16> rope_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;

    std::vector<buffer<bf16>> host_sink;
    std::vector<buffer<bf16>> host_router_weight;
    std::vector<buffer<bf16>> host_router_bias;
    std::vector<buffer<bf16>> up_bias;
    std::vector<buffer<bf16>> gate_bias;
    std::vector<buffer<bf16>> down_bias;
    std::vector<buffer<bf16>> host_up_bias;
    std::vector<buffer<bf16>> host_gate_bias;
    std::vector<buffer<bf16>> host_down_bias;

    buffer<bf16> w_qkv;
    buffer<bf16> w_o;
    buffer<bf16> w_up_gate_down;
    buffer<u8> w_lm_head;
    buffer<bf16> logits;

    buffer<bf16> router_weight_buffer;

    buffer<bf16> x;
    std::unique_ptr<Embedding> embedding;

    buffer<bf16> final_rms_weight;
    
    flm_rt::runlist layers_run;
    flm_rt::runlist dequant_all;
    flm_rt::run lm_head_run;
    int current_context_length;
    buffer<bf16> logits_temp;

    // define some parameters
    int L_pad_required_for_large_MM;
    int L_pad_required_for_small_MM;

    int q_proj_dequant_M;
    int q_proj_dequant_N;
    int q_proj_dequant_M_mem_pad;
    int q_proj_dequant_N_mem_pad;

    int k_proj_dequant_M;
    int k_proj_dequant_N;
    int k_proj_dequant_M_mem_pad;
    int k_proj_dequant_N_mem_pad;

    int v_proj_dequant_M;
    int v_proj_dequant_N;
    int v_proj_dequant_M_mem_pad;
    int v_proj_dequant_N_mem_pad;

    int o_proj_dequant_M;
    int o_proj_dequant_N;
    int o_proj_dequant_M_mem_pad;
    int o_proj_dequant_N_mem_pad;

    int up_proj_dequant_M;
    int up_proj_dequant_N;
    int up_proj_dequant_M_mem_pad;
    int up_proj_dequant_N_mem_pad;

    int gate_proj_dequant_M;
    int gate_proj_dequant_N;
    int gate_proj_dequant_M_mem_pad;
    int gate_proj_dequant_N_mem_pad;

    int down_proj_dequant_M;
    int down_proj_dequant_N;
    int down_proj_dequant_M_mem_pad;
    int down_proj_dequant_N_mem_pad;

    int dequant_gate_up_mat_size_padded;
    int dequant_down_mat_size_padded;

    int router_D_padded;
    inline int round_up_to_multiple (int x, int multiple) {
            return ((x + multiple - 1) / multiple) * multiple;
    };
};

