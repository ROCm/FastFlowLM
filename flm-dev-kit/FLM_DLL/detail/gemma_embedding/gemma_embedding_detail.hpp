// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include <algorithm>
#include "models/gemma_embedding/gemma_embedding.hpp"
#include "gemma_embedding_params.hpp"
#include "gemma_embedding_sequences.hpp"
#include <thread>
#include <future>
#define VERBOSE 1
struct gemma_embedding::Impl{
private:
    /// @brief constexprs

    static constexpr int xy_group_id = 3;
    static constexpr int w_proj_group_id = 4;
    static constexpr int w_rms_group_id = 5;
    static constexpr int w_rope_group_id = 6;
    static constexpr int kv_cache_group_id = 7;

    static constexpr int mha_y_group_id = 3;
    static constexpr int mha_q_group_id = 4;
    static constexpr int mha_kv_group_id = 5;

    size_t total_proj_weights =
          (num_attention_heads * head_dim) * hidden_size
        + (num_key_value_heads * head_dim) * hidden_size * 2
        + hidden_size * hidden_size
        + intermediate_size * hidden_size * 3;

    size_t kv_cache_size = MAX_L * (DK + DV);

    static constexpr f32 inv_freq[128] = {
        1.0000e+00, 8.9769e-01, 8.0584e-01, 7.2339e-01, 6.4938e-01, 5.8294e-01, 5.2330e-01, 4.6976e-01, 4.2170e-01, 3.7855e-01, 3.3982e-01, 3.0505e-01,
        2.7384e-01, 2.4582e-01, 2.2067e-01, 1.9810e-01, 1.7783e-01, 1.5963e-01, 1.4330e-01, 1.2864e-01, 1.1548e-01, 1.0366e-01, 9.3057e-02, 8.3536e-02,
        7.4989e-02, 6.7317e-02, 6.0430e-02, 5.4247e-02, 4.8697e-02, 4.3714e-02, 3.9242e-02, 3.5227e-02, 3.1623e-02, 2.8387e-02, 2.5483e-02, 2.2876e-02,
        2.0535e-02, 1.8434e-02, 1.6548e-02, 1.4855e-02, 1.3335e-02, 1.1971e-02, 1.0746e-02, 9.6466e-03, 8.6596e-03, 7.7737e-03, 6.9783e-03, 6.2643e-03,
        5.6234e-03, 5.0481e-03, 4.5316e-03, 4.0679e-03, 3.6517e-03, 3.2781e-03, 2.9427e-03, 2.6416e-03, 2.3714e-03, 2.1288e-03, 1.9110e-03, 1.7154e-03,
        1.5399e-03, 1.3824e-03, 1.2409e-03, 1.1140e-03, 1.0000e-03, 8.9769e-04, 8.0584e-04, 7.2339e-04, 6.4938e-04, 5.8294e-04, 5.2330e-04, 4.6976e-04,
        4.2170e-04, 3.7855e-04, 3.3982e-04, 3.0505e-04, 2.7384e-04, 2.4582e-04, 2.2067e-04, 1.9810e-04, 1.7783e-04, 1.5963e-04, 1.4330e-04, 1.2864e-04,
        1.1548e-04, 1.0366e-04, 9.3057e-05, 8.3536e-05, 7.4989e-05, 6.7317e-05, 6.0430e-05, 5.4247e-05, 4.8697e-05, 4.3714e-05, 3.9242e-05, 3.5227e-05,
        3.1623e-05, 2.8387e-05, 2.5483e-05, 2.2876e-05, 2.0535e-05, 1.8434e-05, 1.6548e-05, 1.4855e-05, 1.3335e-05, 1.1971e-05, 1.0746e-05, 9.6466e-06,
        8.6596e-06, 7.7737e-06, 6.9783e-06, 6.2643e-06, 5.6234e-06, 5.0481e-06, 4.5316e-06, 4.0679e-06, 3.6517e-06, 3.2781e-06, 2.9427e-06, 2.6416e-06,
        2.3714e-06, 2.1288e-06, 1.9110e-06, 1.7154e-06, 1.5399e-06, 1.3824e-06, 1.2409e-06, 1.1140e-06
    };

    static constexpr f32 inv_freq_local[128] = {
        1.0000e+00, 9.3057e-01, 8.6596e-01, 8.0584e-01, 7.4989e-01, 6.9783e-01, 6.4938e-01, 6.0430e-01, 5.6234e-01, 5.2330e-01, 4.8697e-01, 4.5316e-01, 4.2170e-01, 3.9242e-01, 3.6517e-01, 3.3982e-01,
        3.1623e-01, 2.9427e-01, 2.7384e-01, 2.5483e-01, 2.3714e-01, 2.2067e-01, 2.0535e-01, 1.9110e-01, 1.7783e-01, 1.6548e-01, 1.5399e-01, 1.4330e-01, 1.3335e-01, 1.2409e-01, 1.1548e-01, 1.0746e-01,
        1.0000e-01, 9.3057e-02, 8.6596e-02, 8.0584e-02, 7.4989e-02, 6.9783e-02, 6.4938e-02, 6.0430e-02, 5.6234e-02, 5.2330e-02, 4.8697e-02, 4.5316e-02, 4.2170e-02, 3.9242e-02, 3.6517e-02, 3.3982e-02,
        3.1623e-02, 2.9427e-02, 2.7384e-02, 2.5483e-02, 2.3714e-02, 2.2067e-02, 2.0535e-02, 1.9110e-02, 1.7783e-02, 1.6548e-02, 1.5399e-02, 1.4330e-02, 1.3335e-02, 1.2409e-02, 1.1548e-02, 1.0746e-02,
        1.0000e-02, 9.3057e-03, 8.6596e-03, 8.0584e-03, 7.4989e-03, 6.9783e-03, 6.4938e-03, 6.0430e-03, 5.6234e-03, 5.2330e-03, 4.8697e-03, 4.5316e-03, 4.2170e-03, 3.9242e-03, 3.6517e-03, 3.3982e-03,
        3.1623e-03, 2.9427e-03, 2.7384e-03, 2.5483e-03, 2.3714e-03, 2.2067e-03, 2.0535e-03, 1.9110e-03, 1.7783e-03, 1.6548e-03, 1.5399e-03, 1.4330e-03, 1.3335e-03, 1.2409e-03, 1.1548e-03, 1.0746e-03,
        1.0000e-03, 9.3057e-04, 8.6596e-04, 8.0584e-04, 7.4989e-04, 6.9783e-04, 6.4938e-04, 6.0430e-04, 5.6234e-04, 5.2330e-04, 4.8697e-04, 4.5316e-04, 4.2170e-04, 3.9242e-04, 3.6517e-04, 3.3982e-04,
        3.1623e-04, 2.9427e-04, 2.7384e-04, 2.5483e-04, 2.3714e-04, 2.2067e-04, 2.0535e-04, 1.9110e-04, 1.7783e-04, 1.6548e-04, 1.5399e-04, 1.4330e-04, 1.3335e-04, 1.2409e-04, 1.1548e-04, 1.0746e-04
    };

    void load_all_proj_weights(buffer<bf16>& w, Q4NX& q4nx, int layer_idx){
        bf16* w_ptr = w.data();
        bf16* w_ptr_end = w.end();
        {
            LOG_VERBOSE(1, "Loading self attn qkv weights");
            assert(w_ptr < w_ptr_end);
            buffer<bf16> w_q;
            buffer<bf16> w_k;
            buffer<bf16> w_v;
            q4nx.load_weights(w_q, "layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
            q4nx.load_weights(w_k, "layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
            q4nx.load_weights(w_v, "layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");
            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            memcpy(w_ptr, w_q.data(), DQ * D * 2);
            w_ptr += DQ * D;
            memcpy(w_ptr, w_k.data(), DK * D * 2);
            w_ptr += DK * D;
            memcpy(w_ptr, w_v.data(), DV * D * 2);
            w_ptr += DV * D;
        }
        // read o weights
        {
            LOG_VERBOSE(1, "Loading self attn o weights");
            assert(w_ptr < w_ptr_end);
            buffer<bf16> w_o;
            q4nx.load_weights(w_o, "layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight");
            memcpy(w_ptr, w_o.data(), D * D * 2);
            w_ptr += D * D;
        }
        // read up gate weights
        {
            LOG_VERBOSE(1, "Loading up gate weights");
            assert(w_ptr < w_ptr_end);
            buffer<bf16> w_up;
            buffer<bf16> w_gate;
            q4nx.load_weights(w_up, "layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight");
            q4nx.load_weights(w_gate, "layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight");
            LOG_VERBOSE(1, "Copying up and gate weights to buffer");
            memcpy(w_ptr, w_up.data(), INTERMEDIATE_SIZE * D * 2);
            w_ptr += INTERMEDIATE_SIZE * D;
            memcpy(w_ptr, w_gate.data(), INTERMEDIATE_SIZE * D * 2);
            w_ptr += INTERMEDIATE_SIZE * D;
        }
        // read down weights
        {
            LOG_VERBOSE(1, "Loading down weights");
            assert(w_ptr < w_ptr_end);
            buffer<bf16> w_down;
            q4nx.load_weights(w_down, "layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");
            memcpy(w_ptr, w_down.data(), INTERMEDIATE_SIZE * D * 2);
            w_ptr += INTERMEDIATE_SIZE * D;
        }

        // read rms weights
        {
            LOG_VERBOSE(1, "Loading rms weights");
            buffer<bf16> w_input_rms;
            buffer<bf16> w_post_attn_rms;
            buffer<bf16> w_pre_mlp_rms;
            buffer<bf16> w_post_mlp_rms;
            q4nx.load_weights(w_input_rms, "layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
            q4nx.load_weights(w_post_attn_rms, "layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
            q4nx.load_weights(w_pre_mlp_rms, "layers." + std::to_string(layer_idx) + ".pre_feedforward_layernorm.weight");
            q4nx.load_weights(w_post_mlp_rms, "layers." + std::to_string(layer_idx) + ".post_feedforward_layernorm.weight");
        }
    }

    void migrate_kv_cache(int layer_idx, int l_to_migrate){
        this->kv_caches[layer_idx].sync_from_device();

        bf16* k_ptr = this->kv_caches[layer_idx].data();
        bf16* v_ptr = this->kv_caches[layer_idx].data() + this->kv_cache_size / 2;

        for (int l = 0; l < l_to_migrate; l++){
            memcpy(k_ptr, this->tensor_k[l].data(), DK * sizeof(bf16));
            memcpy(v_ptr, this->tensor_v[l].data(), DV * sizeof(bf16));

            k_ptr += DK;
            v_ptr += DV;
        }
        // Both attention kernels read keys and values past the prompt: up to the next 16 rows (full) or
        // 128 rows (sliding). They mask those probabilities to 0, but only after taking the block's max
        // over every row, so whatever an earlier request left there shifted the softmax and the result
        // depended on the request history (and Inf/NaN left there would poison it). Zero rows only raise
        // the max to 0, which the softmax does not depend on.
        const int l_pad = std::min(MAX_L, (l_to_migrate + 127) / 128 * 128);
        memset(k_ptr, 0, (size_t)(l_pad - l_to_migrate) * DK * sizeof(bf16));
        memset(v_ptr, 0, (size_t)(l_pad - l_to_migrate) * DV * sizeof(bf16));
        this->kv_caches[layer_idx].sync_to_device();
    }

    void apply_rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight){
        int n = rms_weight.size();
        const int simd_width = 8;

        __m256 sum_xx_vec = _mm256_setzero_ps();

        // Step 1: Compute sum of squares using AVX
        int i = 0;
        for (; i <= n - simd_width; i += simd_width) {
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);

            // Convert BF16 to FP32
            __m256 fp32_vals = bf16o_fp32(bf16_vals);
            sum_xx_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
        }

        // Horizontal sum of the AVX register
        f32 sum_xx_arr[simd_width];
        _mm256_storeu_ps(sum_xx_arr, sum_xx_vec);
        f32 sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3] +
                        sum_xx_arr[4] + sum_xx_arr[5] + sum_xx_arr[6] + sum_xx_arr[7];

        sum_xx /= (f32)n;
        f32 rms_x = sqrtf(sum_xx + 1e-6f);

        // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
        __m256 rms_x_vec = _mm256_set1_ps(rms_x);
        for (i = 0; i <= n - simd_width; i += simd_width) {
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
            __m256 fp32_vals = bf16o_fp32(bf16_vals);
            __m128i bf16_vals_w = _mm_loadu_si128((__m128i*)&rms_weight[i]);
            __m256 fp32_vals_w = bf16o_fp32(bf16_vals_w);
            __m256 multiplied = _mm256_mul_ps(fp32_vals_w, fp32_vals);
            __m256 y_vec = _mm256_div_ps(_mm256_add_ps(multiplied, fp32_vals), rms_x_vec);
            __m128i result_bf16 = f32o_bf16(y_vec);
            _mm_storeu_si128((__m128i*)&y[i], result_bf16);
        }
    }

    void rope_rms(buffer<bf16>& x, int idx, buffer<bf16>& rms_weight, int layer_idx){
        int DH = 256;
        int heads = x.size() / DH;
        buffer<f32> cos_buf(DH / 2);
        buffer<f32> sin_buf(DH / 2);
        for (uint32_t i = 0; i < DH / 2; i++){
            // use avx2 to calculate cos and sin
            if (is_sliding_window(layer_idx)){
                cos_buf[i] = std::cos(inv_freq_local[i] * idx);
                sin_buf[i] = std::sin(inv_freq_local[i] * idx);
            }
            else{
                cos_buf[i] = std::cos(inv_freq[i] * idx);
                sin_buf[i] = std::sin(inv_freq[i] * idx);
            }
        }
        for (int h = 0; h < heads; h++){
            buffer<bf16> x_head(x.data() + h * DH, DH);
            apply_rms_norm(x_head, x_head, rms_weight);
            buffer<bf16> x_left(x_head.data(), DH / 2);
            buffer<bf16> x_right(x_head.data() + DH / 2, DH / 2);
            int i = 0, simd = 8;
            // vectorized main loop
            for (; i + simd <= DH / 2; i += simd) {
                // load 8 BF16 → float32
                __m128i left_bf16  = _mm_loadu_si128((__m128i*)(x_left.data()  + i));
                __m128i right_bf16 = _mm_loadu_si128((__m128i*)(x_right.data() + i));
                __m256  L = bf16o_fp32(left_bf16);
                __m256  R = bf16o_fp32(right_bf16);

                // load sin/cos
                __m256 C = _mm256_loadu_ps(cos_buf.data() + i);
                __m256 S = _mm256_loadu_ps(sin_buf.data() + i);

                // new_left  = L*C - R*S
                __m256 newL = _mm256_sub_ps(_mm256_mul_ps(L, C),
                                            _mm256_mul_ps(R, S));
                // new_right = L*S + R*C
                __m256 newR = _mm256_add_ps(_mm256_mul_ps(L, S),
                                            _mm256_mul_ps(R, C));
                // store back
                __m128i newL_bf16 = f32o_bf16(newL);
                __m128i newR_bf16 = f32o_bf16(newR);
                _mm_storeu_si128((__m128i*)(x_left.data()  + i), newL_bf16);
                _mm_storeu_si128((__m128i*)(x_right.data() + i), newR_bf16);
            }
        }
    }

    // ---- GELU (approximate="tanh") implemented in bf16 ----
    inline bf16 gelu_tanh_bf16(bf16 x_bf16) {
        float x = float(x_bf16);

        // tanh‑based approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        float result = 0.5f * x * (1.0f + tanhf(inner));

        return bf16(result);
    }

    void element_wise(buffer<bf16>& dest, buffer<bf16>& up_gate, int L){
        for (int i = 0; i < L; i++) {
            const bf16* row = up_gate.data() + (i * intermediate_size * 2);
            const bf16* up   = row;
            const bf16* gate = row + intermediate_size;
            bf16* out = dest.data() + (i * intermediate_size);

            for (int j = 0; j < intermediate_size; j++) {
                bf16 fused = gelu_tanh_bf16(gate[j]);
                out[j] = fused * up[j];
            }
        }
    }

    void residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual){
        int n = src.size();
        const int simd_width = 8;

        for (int i = 0; i < n; i += simd_width){
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[i]);
            __m128i bf16_vals_r = _mm_loadu_si128((__m128i*)&residual[i]);
            __m256 fp32_vals = bf16o_fp32(bf16_vals);
            __m256 fp32_vals_r = bf16o_fp32(bf16_vals_r);
            __m256 result_vec = _mm256_add_ps(fp32_vals, fp32_vals_r);
            __m128i result_bf16 = f32o_bf16(result_vec);
            _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
        }
    }

    void pooling_mean(buffer<bf16>& dest, buffer<bf16>& src, int L, int D){
        // doing mean in L dimension,
        // Input is L x D
        // Output is D
        const int simd_width = 8;

        // Process D elements in chunks of 8
        for (int i = 0; i < D; i += simd_width){
            __m256 sum_vec = _mm256_setzero_ps();

            // Sum across all L rows for the current 8 columns
            for (int j = 0; j < L; j++){
                __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[j * D + i]);
                __m256 fp32_vals = bf16o_fp32(bf16_vals);
                sum_vec = _mm256_add_ps(sum_vec, fp32_vals);
            }

            // Divide by L to get mean
            __m256 result_vec = _mm256_div_ps(sum_vec, _mm256_set1_ps((f32)L));
            __m128i result_bf16 = f32o_bf16(result_vec);
            _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
        }

        // Handle remaining elements if D is not divisible by 8
        for (int i = (D / simd_width) * simd_width; i < D; i++){
            float sum = 0.0f;
            for (int j = 0; j < L; j++){
                sum += (float)src[j * D + i];
            }
            dest[i] = (bf16)(sum / L);
        }
    }

    void l2_norm(buffer<bf16>& dest, buffer<bf16>& src){
        // x[i] = x[i] / sqrt(sum(x[i] * x[i]))
        int n = src.size();
        const int simd_width = 8;

        // calculate sum of squares
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;

        // Process full SIMD chunks
        for (; i <= n - simd_width; i += simd_width){
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[i]);
            __m256 fp32_vals = bf16o_fp32(bf16_vals);
            sum_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals, sum_vec);
        }

        // Handle remaining elements
        float sum = 0.0f;
        for (; i < n; i++){
            float val = (float)src[i];
            sum += val * val;
        }

        // Extract sum from SIMD register
        float sum_arr[simd_width];
        _mm256_storeu_ps(sum_arr, sum_vec);
        for (int j = 0; j < simd_width; j++){
            sum += sum_arr[j];
        }

        // Calculate L2 norm (no division by n)
        float l2_norm = sqrtf(sum);

        // Avoid division by zero
        if (l2_norm == 0.0f) {
            l2_norm = 1.0f;
        }

        // normalize by dividing by L2 norm
        i = 0;
        for (; i <= n - simd_width; i += simd_width){
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&src[i]);
            __m256 fp32_vals = bf16o_fp32(bf16_vals);
            __m256 result_vec = _mm256_div_ps(fp32_vals, _mm256_set1_ps(l2_norm));
            __m128i result_bf16 = f32o_bf16(result_vec);
            _mm_storeu_si128((__m128i*)&dest[i], result_bf16);
        }

        // Handle remaining elements
        for (; i < n; i++){
            dest[i] = (bf16)((float)src[i] / l2_norm);
        }
    }

    LM_Config config;
    npu_xclbin_manager *npu;

    std::unique_ptr<Embedding> embedding;

    npu_app_manager *gemm_app_manager;
    npu_app_manager *full_attn_app_manager;
    npu_app_manager *window_attn_app_manager;
    npu_app_manager *mv_app_manager;

    npu_app full_attn;
    npu_app window_attn;
    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app o_proj;
    npu_app up_gate_proj;
    npu_app down_proj;

    npu_app dense_1;
    npu_app dense_2;
    buffer<bf16> final_tokens_buffer;
    // buffers
    buffer<bf16> residual_buffer;
    buffer<bf16> hidden_states_buffer;
    buffer<bf16> transformed_buffer;
    buffer<bf16> buf_q;
    buffer<bf16> buf_k;
    buffer<bf16> buf_v;
    buffer<bf16> buf_up_gate;
    buffer<bf16> buf_down;
    buffer<bf16> buf_x;
    buffer<bf16> buf_attn_out;
    buffer<bf16> buf_pooling_out;

    tensor_2d<bf16> tensor_residual;
    tensor_2d<bf16> tensor_hidden_states;
    tensor_2d<bf16> tensor_q;
    tensor_2d<bf16> tensor_k;
    tensor_2d<bf16> tensor_v;
    tensor_2d<bf16> tensor_up_gate;
    tensor_2d<bf16> tensor_down;
    tensor_2d<bf16> tensor_x;
    tensor_2d<bf16> tensor_attn_out;

    std::string model_path;

    buffer<bf16> dense_1_weights;
    buffer<bf16> dense_2_weights;
    buffer<bf16> buf_dense_1_out;
    // each layer parameters
    std::vector<buffer<bf16>> input_layernorm_weights;
    std::vector<buffer<bf16>> post_attention_layernorm_weights;
    std::vector<buffer<bf16>> pre_feedforward_layernorm_weights;
    std::vector<buffer<bf16>> post_feedforward_layernorm_weights;
    std::vector<buffer<bf16>> q_norm_weights;
    std::vector<buffer<bf16>> k_norm_weights;
    std::vector<buffer<bf16>> proj_weights;
    std::vector<buffer<bf16>> kv_caches; // full attention, window attention

    // final norm
    buffer<bf16> norm_weights;

    flm_rt::runlist layers_run;
    inline bool is_sliding_window(int layer_idx) {return (layer_idx + 1) % 6 != 0;}
public:

    /// @brief  initialize the gemma_embedding
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance);
    ~Impl();

    void load_weights(Q4NX& q4nx);
    buffer<bf16> embed(std::vector<int>& tokens);
};
