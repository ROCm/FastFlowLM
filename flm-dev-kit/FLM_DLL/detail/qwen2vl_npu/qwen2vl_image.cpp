// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "qwen2vl_image.hpp"

#include <thread>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif

#include "utils/error_measure.hpp"
#include "mmRuntimeSequence.hpp"
#include "vision/conv3dPatchEmbedding.hpp"
#include "npu_sequences/window_attention_sequence.hpp"
#include "image_attention_sequence.hpp"
// #define DEBUG_PRINT_ENCODE_ERROR_METRICS 1
// #define DEBUG_PRINT_ENCODE_TIME_DETAIL 1

Qwen2VLImageEncoder::~Qwen2VLImageEncoder()
{
}

void Qwen2VLImageEncoder::simd_add(
    bf16* input1,
    bf16* input2,
    bf16* output,
    size_t size
){
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4; // Process 64 elements per iteration
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements

    // Use OpenMP only if size is large enough to benefit from parallelization
    // Threshold: 8 chunks (512 elements) per thread minimum to avoid overhead
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    constexpr int max_threads = 4;
    const bool use_parallel = size >= (min_elements_per_thread * 2);

    // Calculate number of full chunks for parallel processing
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    // Process full chunks with OpenMP (chunked distribution for cache locality)
    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + i + 64), _MM_HINT_T0);

        // Load all inputs first (better for out-of-order execution)
        __m256i a_bh_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i));
        __m256i b_bh_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        __m256i a_bh_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 16));
        __m256i b_bh_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 16));

        __m256i a_bh_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 32));
        __m256i b_bh_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 32));

        __m256i a_bh_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 48));
        __m256i b_bh_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 48));

        // Convert to 32-bit and shift (interleaved for better pipeline usage)
        __m512i a_shifted0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec0), 16);
        __m512i b_shifted0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec0), 16);

        __m512i a_shifted1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec1), 16);
        __m512i b_shifted1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec1), 16);

        __m512i a_shifted2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec2), 16);
        __m512i b_shifted2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec2), 16);

        __m512i a_shifted3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec3), 16);
        __m512i b_shifted3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec3), 16);

        // Add as floats
        __m512 sum0 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted0), _mm512_castsi512_ps(b_shifted0));
        __m512 sum1 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted1), _mm512_castsi512_ps(b_shifted1));
        __m512 sum2 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted2), _mm512_castsi512_ps(b_shifted2));
        __m512 sum3 = _mm512_add_ps(_mm512_castsi512_ps(a_shifted3), _mm512_castsi512_ps(b_shifted3));

        // Convert back to bfloat16 and store
        __m256i bf16_0 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum0), 16));
        __m256i bf16_1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum1), 16));
        __m256i bf16_2 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum2), 16));
        __m256i bf16_3 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum3), 16));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), bf16_0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 16), bf16_1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 32), bf16_2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 48), bf16_3);
    }

    // Process remaining elements (remainder after full chunks)
    size_t i = num_chunks * CHUNK_SIZE;

    // Process remaining 16-element chunks
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m256i a_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i));
        __m256i b_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        __m512i a_shifted = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec), 16);
        __m512i b_shifted = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec), 16);

        __m512 sum = _mm512_add_ps(_mm512_castsi512_ps(a_shifted), _mm512_castsi512_ps(b_shifted));

        __m256i bf16_data = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(sum), 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), bf16_data);
    }

    // Scalar tail
    for (; i < size; ++i) {
        output[i] = static_cast<bf16>(static_cast<float>(input1[i]) + static_cast<float>(input2[i]));
    }
}

void Qwen2VLImageEncoder::simd_mul(
    bf16* input1,
    bf16* input2,
    bf16* output,
    size_t size
){
    constexpr size_t SIMD_WIDTH = 16;
    constexpr size_t UNROLL_FACTOR = 4; // Process 64 elements per iteration
    constexpr size_t CHUNK_SIZE = SIMD_WIDTH * UNROLL_FACTOR; // 64 elements

    // Use OpenMP only if size is large enough to benefit from parallelization
    // Threshold: 8 chunks (512 elements) per thread minimum to avoid overhead
    constexpr size_t min_elements_per_thread = CHUNK_SIZE * 8;
    constexpr int max_threads = 4;
    const bool use_parallel = size >= (min_elements_per_thread * 2);

    // Calculate number of full chunks for parallel processing
    const size_t num_chunks = size / CHUNK_SIZE;
    const int signed_num_chunks = static_cast<int>(num_chunks);

    // Process full chunks with OpenMP (chunked distribution for cache locality)
    #pragma omp parallel for num_threads(max_threads) if(use_parallel) schedule(static)
    for (int chunk_idx = 0; chunk_idx < signed_num_chunks; ++chunk_idx) {
        size_t i = static_cast<size_t>(chunk_idx) * CHUNK_SIZE;

        // Prefetch next cache lines
        _mm_prefetch(reinterpret_cast<const char*>(input1 + i + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input2 + i + 64), _MM_HINT_T0);

        // Load all inputs first (better for out-of-order execution)
        __m256i a_bh_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i));
        __m256i b_bh_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        __m256i a_bh_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 16));
        __m256i b_bh_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 16));

        __m256i a_bh_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 32));
        __m256i b_bh_vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 32));

        __m256i a_bh_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i + 48));
        __m256i b_bh_vec3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i + 48));

        // Convert to 32-bit and shift (interleaved for better pipeline usage)
        __m512i a_shifted0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec0), 16);
        __m512i b_shifted0 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec0), 16);

        __m512i a_shifted1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec1), 16);
        __m512i b_shifted1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec1), 16);

        __m512i a_shifted2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec2), 16);
        __m512i b_shifted2 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec2), 16);

        __m512i a_shifted3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec3), 16);
        __m512i b_shifted3 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec3), 16);

        // Mul as floats
        __m512 mul0 = _mm512_mul_ps(_mm512_castsi512_ps(a_shifted0), _mm512_castsi512_ps(b_shifted0));
        __m512 mul1 = _mm512_mul_ps(_mm512_castsi512_ps(a_shifted1), _mm512_castsi512_ps(b_shifted1));
        __m512 mul2 = _mm512_mul_ps(_mm512_castsi512_ps(a_shifted2), _mm512_castsi512_ps(b_shifted2));
        __m512 mul3 = _mm512_mul_ps(_mm512_castsi512_ps(a_shifted3), _mm512_castsi512_ps(b_shifted3));

        // Convert back to bfloat16 and store
        __m256i bf16_0 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(mul0), 16));
        __m256i bf16_1 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(mul1), 16));
        __m256i bf16_2 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(mul2), 16));
        __m256i bf16_3 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(mul3), 16));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), bf16_0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 16), bf16_1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 32), bf16_2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i + 48), bf16_3);
    }

    // Process remaining elements (remainder after full chunks)
    size_t i = num_chunks * CHUNK_SIZE;

    // Process remaining 16-element chunks
    for (; i + SIMD_WIDTH <= size; i += SIMD_WIDTH) {
        __m256i a_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input1 + i));
        __m256i b_bh_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input2 + i));

        __m512i a_shifted = _mm512_slli_epi32(_mm512_cvtepu16_epi32(a_bh_vec), 16);
        __m512i b_shifted = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b_bh_vec), 16);

        __m512 mul = _mm512_mul_ps(_mm512_castsi512_ps(a_shifted), _mm512_castsi512_ps(b_shifted));

        __m256i bf16_data = _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(mul), 16));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), bf16_data);
    }

    // Scalar tail
    for (; i < size; ++i) {
        output[i] = static_cast<bf16>(static_cast<float>(input1[i]) * static_cast<float>(input2[i]));
    }
}

void Qwen2VLImageEncoder::_rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight){
    int total_elements = x.size();
    int hidden_size = rms_weight.size();

    if (hidden_size == 0 || total_elements % hidden_size != 0) {
        // Simple error handling for mismatch
        printf("Error: RMS Norm input size %d not divisible by hidden size %d\n", total_elements, hidden_size);
        return;
    }

    int num_tokens = total_elements / hidden_size;
    const int simd_width = 16;

    for (int t = 0; t < num_tokens; ++t) {
        int offset = t * hidden_size;
        bf16* x_ptr = &x[offset];
        bf16* y_ptr = &y[offset];

        __m512 sum_xx_vec = _mm512_setzero_ps();

        // Step 1: Compute sum of squares using AVX-512
        int i = 0;
        for (; i <= hidden_size - simd_width; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&x_ptr[i]);

            // Convert BF16 to FP32
            __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
            sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
        }

        // Horizontal sum of the AVX-512 register
        f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec);

        sum_xx /= (f32)hidden_size;
        f32 rms_x = sqrtf(sum_xx + QWEN2_VISION_RMS_NORM_EPSILON);

        // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
        __m512 rms_x_vec = _mm512_set1_ps(rms_x);
        for (i = 0; i <= hidden_size - simd_width; i += simd_width) {
            __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&x_ptr[i]);
            __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
            __m256i bf16_vals_w = _mm256_loadu_si256((__m256i*)&rms_weight[i]);
            __m512 fp32_vals_w = bf16o_fp32_512(bf16_vals_w);
            __m512 y_vec = _mm512_div_ps(_mm512_mul_ps(fp32_vals_w, fp32_vals), rms_x_vec);
            __m256i result_bf16 = f32o_bf16_512(y_vec);
            _mm256_storeu_si256((__m256i*)&y_ptr[i], result_bf16);
        }
    }
}
std::vector<int> Qwen2VLImageEncoder::RotaryEmbedGeneration(
    std::vector<std::vector<int32_t>> image_grid_thw,
    int seq_len,
    int seq_len_padded,
    std::vector<float>& cos_emb,
    std::vector<float>& sin_emb)
{
    // ROTARY EMBEDDING GENERATION

    std::vector<int> window_index;
    window_index.reserve(seq_len);

    // Qwen2-VL constants
    const int spatial_merge_size = QWEN2_SPATIAL_MERGE_SIZE; // QWEN2_SPATIAL_MERGE_SIZE
    const int patch_size = QWEN2_PATCH_SIZE;
    const int window_size = QWEN2_WINDOW_ATTENTION_PIXEL_SIZE; // Standard Qwen2-VL window size in pixels
    const int head_dim = QWEN2_VISION_HEAD_DIM; // 80
    const float rope_theta = 10000.0f; // Standard RoPE theta

    std::vector<std::pair<int, int>> pos_ids;
    pos_ids.reserve(seq_len);

    int vit_merger_window_size = window_size / spatial_merge_size / patch_size; // typically 4
    int current_seq_offset = 0;

    for (const auto& thw : image_grid_thw) {
        int t = thw[0];
        int h = thw[1];
        int w = thw[2];

        int h_blocks = h / spatial_merge_size;
        int w_blocks = w / spatial_merge_size;

        // 1. Generate local pos_ids (h, w) for this image
        for(int ti=0; ti<t; ++ti) {
           for(int hb=0; hb<h_blocks; ++hb) {
               for(int wb=0; wb<w_blocks; ++wb) {
                   for(int hi=0; hi<spatial_merge_size; ++hi) {
                       for(int wi=0; wi<spatial_merge_size; ++wi) {
                           int h_val = hb * spatial_merge_size + hi;
                           int w_val = wb * spatial_merge_size + wi;
                           pos_ids.push_back({h_val, w_val});
                       }
                   }
               }
           }
        }

        // 2. Generate Window Permutation Indices
        int llm_grid_h = h_blocks;
        int llm_grid_w = w_blocks;

        int pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
        int pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;

        int num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        int num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

        for (int ti = 0; ti < t; ++ti) {
            for (int wh = 0; wh < num_windows_h; ++wh) {
                for (int ww = 0; ww < num_windows_w; ++ww) {
                    for (int i = 0; i < vit_merger_window_size; ++i) {
                        for (int j = 0; j < vit_merger_window_size; ++j) {
                            int global_h = wh * vit_merger_window_size + i;
                            int global_w = ww * vit_merger_window_size + j;

                            if (global_h < llm_grid_h && global_w < llm_grid_w) {
                                int relative_block_idx = ti * (llm_grid_h * llm_grid_w) +
                                                         global_h * llm_grid_w +
                                                         global_w;

                                int base_token_idx = current_seq_offset +
                                                     relative_block_idx * spatial_merge_size * spatial_merge_size;

                                for(int k=0; k<spatial_merge_size*spatial_merge_size; ++k) {
                                    window_index.push_back(base_token_idx + k);
                                }
                            }
                        }
                    }
                }
            }
        }
        current_seq_offset += t * h * w;
    }

    // 3. Compute RoPE Cache (Inv Freq)
    int dim = head_dim / 2; // 40
    int half_dim = dim / 2; // 20
    std::vector<float> inv_freq(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(rope_theta, (float)(2 * i) / dim);
    }

    // 4. Generate Final Cos/Sin Embeddings
    cos_emb.resize(seq_len_padded * head_dim, 0.0f);
    sin_emb.resize(seq_len_padded * head_dim, 0.0f);

    for (size_t i = 0; i < seq_len; ++i) {
        int perm_idx = window_index[i];
        int h = pos_ids[perm_idx].first;
        int w = pos_ids[perm_idx].second;

        float* c_ptr = &cos_emb[i * head_dim];
        float* s_ptr = &sin_emb[i * head_dim];

        // Fill H part (0..19 and 40..59)
        for (int k = 0; k < half_dim; ++k) {
            float ang = h * inv_freq[k];
            float c = std::cos(ang);
            float s = std::sin(ang);
            c_ptr[k] = c;
            s_ptr[k] = s;
            c_ptr[40 + k] = c;
            s_ptr[40 + k] = s;
        }

        // Fill W part (20..39 and 60..79)
        for (int k = 0; k < half_dim; ++k) {
            float ang = w * inv_freq[k];
            float c = std::cos(ang);
            float s = std::sin(ang);
            c_ptr[20 + k] = c;
            s_ptr[20 + k] = s;
            c_ptr[60 + k] = c;
            s_ptr[60 + k] = s;
        }
    }

    return window_index;
}
Qwen2VLImageEncoder::Qwen2VLImageEncoder(LM_Config config,
    npu_xclbin_manager *npu_instance
):config(config), npu(npu_instance), model_path(config.model_path){

    this->vision_flash_attention = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_attention.xclbin"));
    this->vision_proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));
    this->vision_window_attention = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_window_attention.xclbin"));

    this->vision_flash_attention_app = this->vision_flash_attention->create_app();
    this->vision_window_attention_app = this->vision_window_attention->create_app();

    this->vision_q_projection_app = this->vision_proj->create_app();
    this->vision_k_projection_app = this->vision_proj->create_app();
    this->vision_v_projection_app = this->vision_proj->create_app();
    this->vision_o_projection_app = this->vision_proj->create_app();

    this->vision_mlp_gate_proj_app = this->vision_proj->create_app();
    this->vision_mlp_up_proj_app = this->vision_proj->create_app();
    this->vision_mlp_down_proj_app = this->vision_proj->create_app();

    this->merger_mlp_fc1_app = this->vision_proj->create_app();
    this->merger_mlp_fc2_app = this->vision_proj->create_app();

    // resize the vectors to hold all layers weight
    this->norm_1_weight.resize(QWEN2_NUM_LAYERS);
    this->norm_2_weight.resize(QWEN2_NUM_LAYERS);

    this->attn_q_weight.resize(QWEN2_NUM_LAYERS);
    this->attn_k_weight.resize(QWEN2_NUM_LAYERS);
    this->attn_v_weight.resize(QWEN2_NUM_LAYERS);
    this->attn_o_weight.resize(QWEN2_NUM_LAYERS);

    this->mlp_gate_proj_weight.resize(QWEN2_NUM_LAYERS);
    this->mlp_up_proj_weight.resize(QWEN2_NUM_LAYERS);
    this->mlp_down_proj_weight.resize(QWEN2_NUM_LAYERS);
}

void Qwen2VLImageEncoder::init_weights(SafeTensors &q4nx){

    q4nx.load_weights(this->patch_embed_proj_weight, "model.visual.patch_embed.proj.weight");

    q4nx.load_weights(this->post_norm_weight, "model.visual.merger.ln_q.weight");

    for(int layer_idx = 0; layer_idx< QWEN2_NUM_LAYERS; layer_idx++){
        q4nx.load_weights(
            this->norm_1_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".rmsnorm1.weight"
        );
        q4nx.load_weights(
            this->norm_2_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".rmsnorm2.weight"
        );

        // attention
        this->attn_q_weight[layer_idx] = vision_q_projection_app.create_bo_buffer<bf16>(
            QWEN2_VISION_HIDDEN_SIZE*QWEN2_VISION_HIDDEN_SIZE +  QWEN2_VISION_HIDDEN_SIZE); // weight + bias
        q4nx.load_weights(
            this->attn_q_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.q_proj.bias",0
        );
        q4nx.load_weights(
            this->attn_q_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.q_proj.weight", sizeof(bf16)* QWEN2_VISION_HIDDEN_SIZE
        );

        this->attn_k_weight[layer_idx] = vision_k_projection_app.create_bo_buffer<bf16>(
            QWEN2_VISION_HIDDEN_SIZE*QWEN2_VISION_HIDDEN_SIZE +  QWEN2_VISION_HIDDEN_SIZE); // weight + bias
        q4nx.load_weights(
            this->attn_k_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.k_proj.bias",0
        );
        q4nx.load_weights(
            this->attn_k_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.k_proj.weight", sizeof(bf16)* QWEN2_VISION_HIDDEN_SIZE
        );

        this->attn_v_weight[layer_idx] = vision_v_projection_app.create_bo_buffer<bf16>(
            QWEN2_VISION_HIDDEN_SIZE*QWEN2_VISION_HIDDEN_SIZE +  QWEN2_VISION_HIDDEN_SIZE); // weight + bias
        q4nx.load_weights(
            this->attn_v_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.v_proj.bias",0
        );
        q4nx.load_weights(
            this->attn_v_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.v_proj.weight", sizeof(bf16)* QWEN2_VISION_HIDDEN_SIZE
        );
        this->attn_o_weight[layer_idx] = vision_o_projection_app.create_bo_buffer<bf16>(
            QWEN2_VISION_HIDDEN_SIZE*QWEN2_VISION_HIDDEN_SIZE +  QWEN2_VISION_HIDDEN_SIZE); // weight + bias
        q4nx.load_weights(
            this->attn_o_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.o_proj.bias",0
        );
        q4nx.load_weights(
            this->attn_o_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".attn.o_proj.weight", sizeof(bf16)* QWEN2_VISION_HIDDEN_SIZE
        );

        this->mlp_gate_proj_weight[layer_idx] = vision_mlp_gate_proj_app.create_bo_buffer<bf16>(
            QWEN2_VISION_HIDDEN_SIZE*vision_intermediate_size_padded + vision_intermediate_size_padded); // weight + bias  bias is not padded
        q4nx.load_weights(
            this->mlp_gate_proj_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".mlp.gate_proj.bias",0
        );
        q4nx.load_weights(
            this->mlp_gate_proj_weight[layer_idx],
            "model.visual." +std::to_string(layer_idx) + ".mlp.gate_proj.weight", sizeof(bf16)* vision_intermediate_size_padded
        );

        this->mlp_up_proj_weight[layer_idx] = vision_mlp_up_proj_app.create_bo_buffer<bf16>(
            QWEN2_VISION_HIDDEN_SIZE*vision_intermediate_size_padded + vision_intermediate_size_padded); // weight + bias
        q4nx.load_weights(
            this->mlp_up_proj_weight[layer_idx],
            "model.visual." + std::to_string(layer_idx) + ".mlp.up_proj.bias",0
        );
        q4nx.load_weights(
            this->mlp_up_proj_weight[layer_idx],
            "model.visual." + std::to_string(layer_idx) + ".mlp.up_proj.weight", sizeof(bf16)* vision_intermediate_size_padded
        );

        this->mlp_down_proj_weight[layer_idx] = vision_mlp_down_proj_app.create_bo_buffer<bf16>(
            vision_intermediate_size_padded*QWEN2_VISION_HIDDEN_SIZE + QWEN2_VISION_HIDDEN_SIZE); // weight + bias
        q4nx.load_weights(
            this->mlp_down_proj_weight[layer_idx],
            "model.visual." + std::to_string(layer_idx) + ".mlp.down_proj.bias",0
        );
        q4nx.load_weights(
            this->mlp_down_proj_weight[layer_idx],
            "model.visual." + std::to_string(layer_idx) + ".mlp.down_proj.weight", sizeof(bf16)* QWEN2_VISION_HIDDEN_SIZE
        );
    }

    //LOAD merger weights and bias
    this->merger_mlp_fc1_weight = merger_mlp_fc1_app.create_bo_buffer<bf16>(
        QWEN2_MERGER_HIDDEN_SIZE*QWEN2_MERGER_MLP_INTERMEDIATE_SIZE + QWEN2_MERGER_MLP_INTERMEDIATE_SIZE); // weight + bias
    q4nx.load_weights(
        this->merger_mlp_fc1_weight,
        "model.visual.merger.mlp.0.bias",0
    );
    q4nx.load_weights(
        this->merger_mlp_fc1_weight,
        "model.visual.merger.mlp.0.weight", sizeof(bf16)* QWEN2_MERGER_MLP_INTERMEDIATE_SIZE
    );

    this->merger_mlp_fc2_weight = merger_mlp_fc2_app.create_bo_buffer<bf16>(
        QWEN2_MERGER_MLP_INTERMEDIATE_SIZE*QWEN2_VISION_OUT_HIDDEN_SIZE + QWEN2_VISION_OUT_HIDDEN_SIZE); // weight + bias
    q4nx.load_weights(
        this->merger_mlp_fc2_weight,
        "model.visual.merger.mlp.2.bias",0
    );
    q4nx.load_weights(
        this->merger_mlp_fc2_weight,
        "model.visual.merger.mlp.2.weight", sizeof(bf16)* QWEN2_VISION_OUT_HIDDEN_SIZE
    );
}

static void apply_rotary_pos_emb_vision_avx512(
    bf16* q,
    bf16* k,
    const float* cos_emb,
    const float* sin_emb,
    int seq_len,
    int hidden_size,
    int head_dim
) {
    int num_heads = hidden_size / head_dim;
    int half_dim = head_dim / 2;

    for (int i = 0; i < seq_len; ++i) {
        const float* c_ptr = cos_emb + i * head_dim;
        const float* s_ptr = sin_emb + i * head_dim;

        bf16* q_token = q + i * hidden_size;
        bf16* k_token = k + i * hidden_size;

        for (int h = 0; h < num_heads; ++h) {
            bf16* q_head = q_token + h * head_dim;
            bf16* k_head = k_token + h * head_dim;

            // Process blocks of 16
            for (int j = 0; j < half_dim; j += 16) {
                int remain = half_dim - j;
                // We assume remain is either >= 16 or exactly 8 based on Qwen2VL head_dim=80

                // Load cos/sin (fp32)
                __m512 vc, vs;
                if (remain >= 16) {
                    vc = _mm512_loadu_ps(c_ptr + j);
                    vs = _mm512_loadu_ps(s_ptr + j);
                } else {
                    __mmask16 mask = 0xFF; // load 8 elements
                    vc = _mm512_maskz_loadu_ps(mask, c_ptr + j);
                    vs = _mm512_maskz_loadu_ps(mask, s_ptr + j);
                }

                auto process_projection = [&](bf16* buf) {
                    __m512 v1, v2;
                    // Load v1 (first half)
                    if (remain >= 16) {
                        __m256i tmp = _mm256_loadu_si256((__m256i*)(buf + j));
                        __m512i tmp_32 = _mm512_cvtepu16_epi32(tmp);
                        v1 = _mm512_castsi512_ps(_mm512_slli_epi32(tmp_32, 16));
                    } else {
                        // Load 8 elements
                        __m128i tmp128 = _mm_loadu_si128((__m128i*)(buf + j));
                        __m256i tmp256 = _mm256_cvtepu16_epi32(tmp128);
                        __m512i tmp512 = _mm512_castsi256_si512(tmp256);
                        v1 = _mm512_castsi512_ps(_mm512_slli_epi32(tmp512, 16));
                    }

                    // Load v2 (second half)
                    bf16* buf2 = buf + j + half_dim;
                    if (remain >= 16) {
                        __m256i tmp = _mm256_loadu_si256((__m256i*)(buf2));
                        __m512i tmp_32 = _mm512_cvtepu16_epi32(tmp);
                        v2 = _mm512_castsi512_ps(_mm512_slli_epi32(tmp_32, 16));
                    } else {
                         __m128i tmp128 = _mm_loadu_si128((__m128i*)(buf2));
                        __m256i tmp256 = _mm256_cvtepu16_epi32(tmp128);
                        __m512i tmp512 = _mm512_castsi256_si512(tmp256);
                        v2 = _mm512_castsi512_ps(_mm512_slli_epi32(tmp512, 16));
                    }

                    // RoPE rotation
                    // q_new = q * cos - rotate_half(q) * sin
                    // v1_new = v1 * c - v2 * s
                    // v2_new = v2 * c + v1 * s
                    __m512 v1_out = _mm512_sub_ps(_mm512_mul_ps(v1, vc), _mm512_mul_ps(v2, vs));
                    __m512 v2_out = _mm512_add_ps(_mm512_mul_ps(v2, vc), _mm512_mul_ps(v1, vs));

                    // Store v1
                    __m512i v1_i = _mm512_castps_si512(v1_out);
                    v1_i = _mm512_srli_epi32(v1_i, 16);
                    __m256i v1_res = _mm512_cvtepi32_epi16(v1_i);
                     if (remain >= 16) {
                        _mm256_storeu_si256((__m256i*)(buf + j), v1_res);
                    } else {
                        __m128i res128 = _mm256_castsi256_si128(v1_res);
                        _mm_storeu_si128((__m128i*)(buf + j), res128);
                    }

                    // Store v2
                    __m512i v2_i = _mm512_castps_si512(v2_out);
                    v2_i = _mm512_srli_epi32(v2_i, 16);
                    __m256i v2_res = _mm512_cvtepi32_epi16(v2_i);
                     if (remain >= 16) {
                        _mm256_storeu_si256((__m256i*)(buf2), v2_res);
                    } else {
                        __m128i res128 = _mm256_castsi256_si128(v2_res);
                        _mm_storeu_si128((__m128i*)(buf2), res128);
                    }
                };

                process_projection(q_head);
                process_projection(k_head);
            }
        }
    }
}

std::vector<bf16> Qwen2VLImageEncoder::encode(void *image_payload_ptr){

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        SafeTensors reference_tensor(this->model_path + "/qwen2_5_vl_debug.safetensors");
    #endif
    auto encoder_start_time = std::chrono::high_resolution_clock::now();

    auto round_up_to_multiple_lambda = [](int x, int multiple) -> int {
            return ((x + multiple - 1) / multiple) * multiple;
        };

    qwen2vl_image_payload_t *image_payload = (qwen2vl_image_payload_t *)image_payload_ptr;
    std::vector<std::vector<bf16>> final_res;

    std::vector<std::vector<int32_t>> image_grid_thw;
    std::vector<float> cos_emb;
    std::vector<float> sin_emb;

    int seq_len = 0;
    int seq_len_of_last_image = 0;
    for(auto img: image_payload->images ){
        int t = img.grid_h * img.grid_w;
        seq_len += t;
        seq_len_of_last_image = t;
        image_grid_thw.push_back({1, img.grid_h, img.grid_w});
    }

    int attention_padded_size = round_up_to_multiple_lambda(seq_len_of_last_image, vision_L_padded_requirement_for_attention);
    attention_padded_size = round_up_to_multiple_lambda(attention_padded_size, window_L_padded_requirement_for_attention);

    int seq_len_padded = attention_padded_size - seq_len_of_last_image;
    seq_len_padded += seq_len;
    seq_len_padded = round_up_to_multiple_lambda(seq_len_padded, seq_len_pad_requirement_for_MM);

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "seq_len before padding: " << seq_len << std::endl;
        std::cout << "seq_len after padding: " << seq_len_padded << std::endl;
    #endif

    if(  seq_len_padded %(QWEN2_SPATIAL_MERGE_SIZE*QWEN2_SPATIAL_MERGE_SIZE) != 0){
        std::cerr << "seq_len_padded must be divisible by QWEN2_SPATIAL_MERGE_SIZE*QWEN2_SPATIAL_MERGE_SIZE" << std::endl;
        exit(1);
    }
    int merge_seq_len = seq_len / (QWEN2_SPATIAL_MERGE_SIZE*QWEN2_SPATIAL_MERGE_SIZE);
    int merge_seq_len_padded = (seq_len_padded ) / (QWEN2_SPATIAL_MERGE_SIZE*QWEN2_SPATIAL_MERGE_SIZE);
    merge_seq_len_padded = round_up_to_multiple_lambda(merge_seq_len_padded, seq_len_pad_requirement_for_MM);
    merge_seq_len_padded = std::max(merge_seq_len_padded, 1024);

    hidden_state.resize(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
    memset(hidden_state.data(), 0, sizeof(bf16)*hidden_state.size());
    {
        bool ADD_BIAS = true;
        generate_mm_sequence<bf16, bf16>(
            *this->vision_q_projection_app.seq(),
            seq_len_padded, QWEN2_VISION_HIDDEN_SIZE, QWEN2_VISION_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, QWEN2_VISION_HEAD_DIM
        );
        generate_mm_sequence<bf16, bf16>(
            *this->vision_k_projection_app.seq(),
            seq_len_padded, QWEN2_VISION_HIDDEN_SIZE, QWEN2_VISION_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, QWEN2_VISION_HEAD_DIM
        );
        generate_mm_sequence<bf16, bf16>(
            *this->vision_v_projection_app.seq(),
            seq_len_padded, QWEN2_VISION_HIDDEN_SIZE, QWEN2_VISION_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, QWEN2_VISION_HEAD_DIM
        );
        generate_mm_sequence<bf16, bf16>(
            *this->vision_o_projection_app.seq(),
            seq_len_padded, QWEN2_VISION_HIDDEN_SIZE, QWEN2_VISION_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, QWEN2_VISION_HEAD_DIM
        );

        generate_mm_sequence<bf16, bf16>(
            *this->vision_mlp_gate_proj_app.seq(),
            seq_len_padded, QWEN2_VISION_HIDDEN_SIZE, vision_intermediate_size_padded,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 2, // silu
            ENABLE_QKV_REORDER, -1
        );
        generate_mm_sequence<bf16, bf16>(
            *this->vision_mlp_up_proj_app.seq(),
            seq_len_padded, QWEN2_VISION_HIDDEN_SIZE, vision_intermediate_size_padded,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, -1
        );
        generate_mm_sequence<bf16, bf16>(
            *this->vision_mlp_down_proj_app.seq(),
            seq_len_padded, vision_intermediate_size_padded, QWEN2_VISION_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, -1
        );

        // assert

        generate_mm_sequence<bf16, bf16>(
            *this->merger_mlp_fc1_app.seq(),
            merge_seq_len_padded, QWEN2_MERGER_HIDDEN_SIZE, QWEN2_MERGER_MLP_INTERMEDIATE_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 1, // gelu
            ENABLE_QKV_REORDER, -1
        );

        generate_mm_sequence<bf16, bf16>(
            *this->merger_mlp_fc2_app.seq(),
            merge_seq_len_padded, QWEN2_MERGER_MLP_INTERMEDIATE_SIZE, QWEN2_VISION_OUT_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address,rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0,
            ENABLE_QKV_REORDER, -1

        );

        window_attention_runtimeSequence<bf16, bf16>(
            *this->vision_window_attention_app.seq(),
            1, window_LQ, window_LK, QWEN2_VISION_HEAD_DIM, QWEN2_VISION_NUM_HEADS,
            vision_num_of_columns, vision_num_of_rows,
            {seq_len_padded}, {seq_len_padded},
            0,0,0,0,
            0,1,2,3
        );

        {
            if(vision_S_padded_requirement_for_attention >vision_L_padded_requirement_for_attention ){
                std::cerr << "vision S padded requirement greater than L padded requirement"<< std::endl;
                exit(1);
            }

            // support of multiple batch mha
            std::vector<int> L_seq_list;
            std::vector<int> S_seq_list;
            std::vector<int> S_seq_padded_list;

            std::vector<int> Q_batch_offset_list;
            std::vector<int> K_batch_offset_list;
            std::vector<int> V_batch_offset_list;
            std::vector<int> O_batch_offset_list;

            uint32_t total_seq_len = 0;
            for(int b= 0; b < image_grid_thw.size(); b++){
                int end_seq_len = 1;
                for(int v: image_grid_thw[b]){
                    end_seq_len *= v;
                }
                total_seq_len += end_seq_len;
            }
            //total_seq_len is now batch*L_Seq_per_batch

            uint32_t Q_total_seq_len_per_head = round_up_to_multiple_lambda( total_seq_len,vision_L_padded_requirement_for_attention );
            uint32_t K_total_seq_len_per_head = round_up_to_multiple_lambda( total_seq_len,vision_L_padded_requirement_for_attention );
            uint32_t V_total_seq_len_per_head = K_total_seq_len_per_head;

            int cur_seq_len = 0;
            for(int b = 0; b < image_grid_thw.size(); b++){

                int start_seq_len = cur_seq_len;
                int end_seq_len = 1;
                for(int v: image_grid_thw[b]){
                    end_seq_len *= v;
                }

                uint32_t l_seq_padded = round_up_to_multiple_lambda(end_seq_len, vision_L_padded_requirement_for_attention);// same for L, and S
                uint32_t s_seq_padded = round_up_to_multiple_lambda(end_seq_len, vision_S_padded_requirement_for_attention);
                L_seq_list.push_back(l_seq_padded);
                S_seq_list.push_back(end_seq_len);
                S_seq_padded_list.push_back(s_seq_padded);

                // but since

                Q_batch_offset_list.push_back(start_seq_len * (QWEN2_VISION_HIDDEN_SIZE) );
                K_batch_offset_list.push_back(start_seq_len * (QWEN2_VISION_HIDDEN_SIZE)  );
                V_batch_offset_list.push_back(start_seq_len * (QWEN2_VISION_HIDDEN_SIZE) );
                O_batch_offset_list.push_back(start_seq_len * QWEN2_VISION_HIDDEN_SIZE );

                cur_seq_len += end_seq_len;
            }
            if(total_seq_len != cur_seq_len){
                std::cerr << "total_seq_len != cur_seq_len" << std::endl;
                exit(-1);
            }

            // because q, k, v buffer all in on buffer after MM
            constexpr int ARG_Q = 0;
            constexpr int ARG_K = 1;
            constexpr int ARG_V = 2;
            constexpr int ARG_O = 3;

            AttentionConfig config = {
                (uint32_t)QWEN2_VISION_HEAD_DIM, (uint32_t)QWEN2_VISION_NUM_HEADS,
                vision_LQ_per_CT, vision_LK_per_CT,
                vision_LQ_internal, vision_LK_internal,
                vision_num_of_columns, vision_num_of_rows,
                vision_CU_mode,
                (uint32_t)image_grid_thw.size(),

                ARG_Q, ARG_K, ARG_V, ARG_O,

                (uint32_t)(QWEN2_VISION_HIDDEN_SIZE), (uint32_t)(QWEN2_VISION_HIDDEN_SIZE), (uint32_t)(QWEN2_VISION_HIDDEN_SIZE),
                (uint32_t)QWEN2_VISION_HIDDEN_SIZE,

                ENABLE_QKV_REORDER,
                Q_total_seq_len_per_head, K_total_seq_len_per_head, V_total_seq_len_per_head
            };

            BatchMetadata batch_data = {
                L_seq_list, S_seq_list, S_seq_padded_list,
                Q_batch_offset_list, K_batch_offset_list, V_batch_offset_list, O_batch_offset_list
            };

            setup_SHM_configuration<bf16, bf16>(
                *this->vision_flash_attention_app.seq(),
                config,
                batch_data,
                1.0/std::sqrt(QWEN2_VISION_HEAD_DIM)
            );
        }
    }
    {

        this->mm_qkv_input = vision_q_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        this->mm_q_output = vision_q_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        this->mm_k_output = vision_k_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        this->mm_v_output = vision_v_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        this->window_attention_o_output = vision_window_attention_app.create_bo_buffer<bf16>(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        this->mm_o_output_fc1_input = vision_o_projection_app.create_bo_buffer<bf16>(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        this->mm_gate_output = vision_mlp_gate_proj_app.create_bo_buffer<bf16>((size_t)seq_len_padded * vision_intermediate_size_padded);
        this->mm_up_output = vision_mlp_up_proj_app.create_bo_buffer<bf16>((size_t)seq_len_padded * vision_intermediate_size_padded);
        this->merger_mlp_fc1_input = merger_mlp_fc1_app.create_bo_buffer<bf16>(merge_seq_len_padded * QWEN2_MERGER_MLP_INTERMEDIATE_SIZE);
        this->merger_mlp_fc1_out_fc2_in = merger_mlp_fc1_app.create_bo_buffer<bf16>(merge_seq_len_padded * QWEN2_MERGER_MLP_INTERMEDIATE_SIZE);
        this->merger_mlp_fc2_out = merger_mlp_fc2_app.create_bo_buffer<bf16>(merge_seq_len_padded * QWEN2_VISION_OUT_HIDDEN_SIZE);
    }
    auto time_start_patch_embed = std::chrono::high_resolution_clock::now();
    std::vector<bf16> patch_embed= conv3d_patch_embed_avx512_max_performance(
        image_payload->_data__processed.data(),
        patch_embed_proj_weight.data(),
        nullptr,  // Qwen2-VL's patch_embed conv has no bias
        seq_len,
        3,
        QWEN2_VISION_HIDDEN_SIZE,
        QWEN2_TEMPORAL_PATCH_SIZE,
        QWEN2_PATCH_SIZE
    );
    auto time_end_patch_embed = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto time_patch_embed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_patch_embed - time_start_patch_embed).count();
        std::cout << "Time spent on patch embedding: " << time_patch_embed << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error metrics for patch embedding: " << std::endl;
        buffer<bf16> after_conv3d;
        reference_tensor.load_weights(after_conv3d, "after_conv3d");
        print_error_metrics<bf16, bf16>(
            patch_embed.data(), after_conv3d.data(),
            1,
            seq_len, QWEN2_VISION_HIDDEN_SIZE,
            seq_len, QWEN2_VISION_HIDDEN_SIZE
        );
    #endif

    // memcpy(hidden_state.data(), patch_embed.data(), sizeof(bf16)*seq_len*QWEN2_VISION_HIDDEN_SIZE);

    // COMPARE THE ERRORS
    std::vector<int> window_index = RotaryEmbedGeneration(
        image_grid_thw,
        seq_len,
        seq_len_padded,
        cos_emb,
        sin_emb
    );
    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error metrics for position embeddings: " << std::endl;
        buffer<float> reference_cos_emb(seq_len * QWEN2_VISION_HEAD_DIM);
        buffer<float> reference_sin_emb(seq_len * QWEN2_VISION_HEAD_DIM);
        reference_tensor.load_weights(reference_cos_emb, "emb_cos");
        reference_tensor.load_weights(reference_sin_emb, "emb_sin");
        print_error_metrics<float, float>(
            cos_emb.data(), reference_cos_emb.data(),
            1,
            seq_len, QWEN2_VISION_HEAD_DIM,
            seq_len, QWEN2_VISION_HEAD_DIM
        );
        print_error_metrics<float, float>(
            sin_emb.data(), reference_sin_emb.data(),
            1,
            seq_len, QWEN2_VISION_HEAD_DIM,
            seq_len, QWEN2_VISION_HEAD_DIM
        );
    #endif

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "hidden size before layer"<<std::endl;
        buffer<bf16> hidden_state_before_layer(seq_len * QWEN2_VISION_HIDDEN_SIZE);
        reference_tensor.load_weights(hidden_state_before_layer, "hidden_state_before_layer_row_major");
        print_error_metrics<bf16, bf16>(
            patch_embed.data(), hidden_state_before_layer.data(),
            1,
            seq_len, QWEN2_VISION_HIDDEN_SIZE,
            seq_len, QWEN2_VISION_HIDDEN_SIZE
        );

    #endif
    // The reference python code does reorder on hidden states,
    // Thus, we should also do it here, since it will affect that rmsnorm, attention, mlp results..
    {
        // std::vector<bf16> hidden_state_permuted(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
        #pragma omp parallel for
        for (int i = 0; i < seq_len; ++i) {
            int src_idx = window_index[i];
            const bf16* src_ptr = patch_embed.data() + src_idx * QWEN2_VISION_HIDDEN_SIZE;
            bf16* dst_ptr = hidden_state.data() + i * QWEN2_VISION_HIDDEN_SIZE;
            memcpy(dst_ptr, src_ptr, QWEN2_VISION_HIDDEN_SIZE * sizeof(bf16));
        }
        // memcpy(hidden_state.data(), hidden_state_permuted.data(), sizeof(bf16)*seq_len*QWEN2_VISION_HIDDEN_SIZE);
    }

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "hidden size before layer_reordered"<<std::endl;
        buffer<bf16> hidden_state_before_layer_reorde(seq_len * QWEN2_VISION_HIDDEN_SIZE);
        reference_tensor.load_weights(hidden_state_before_layer_reorde, "hidden_state_before_layer");
        print_error_metrics<bf16, bf16>(
            hidden_state.data(), hidden_state_before_layer_reorde.data(),
            1,
            seq_len, QWEN2_VISION_HIDDEN_SIZE,
            seq_len, QWEN2_VISION_HIDDEN_SIZE
        );
    #endif

    constexpr bool DISABLE_LAYER_PASS_THROUGH = true;

    buffer<bf16> residual(seq_len_padded * QWEN2_VISION_HIDDEN_SIZE);
    for(int layer_idx = 0; layer_idx < QWEN2_NUM_LAYERS; layer_idx++){  // for now, DEBUG
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS

            if (DISABLE_LAYER_PASS_THROUGH){
                // copy
                buffer<bf16> residual1_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
                reference_tensor.load_weights(residual1_reference, "residual_1_" + std::to_string(layer_idx));
                memcpy(hidden_state.data(), residual1_reference.data(), sizeof(bf16)*seq_len*QWEN2_VISION_HIDDEN_SIZE);
            }
        #endif

        //memcpy(residual.data(), hidden_state.data(), sizeof(bf16)*seq_len_padded*QWEN2_VISION_HIDDEN_SIZE);
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - residual1 "<<std::endl;
            buffer<bf16> residual1_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(residual1_reference, "residual_1_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                hidden_state.data(), residual1_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );

        #endif

        _rms_norm(mm_qkv_input, hidden_state, norm_1_weight[layer_idx]);
        mm_qkv_input.sync_to_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - hidden state after RMSNorm 1"<<std::endl;
            buffer<bf16> rms_1_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(rms_1_out_reference, "norm_1_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_qkv_input.data(), rms_1_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );

        #endif

        auto time_start_q = std::chrono::high_resolution_clock::now();
        attn_q_weight[layer_idx].sync_to_device();
        vision_q_projection_app(
            mm_qkv_input,
            attn_q_weight[layer_idx],
            mm_q_output
        );
        mm_q_output.sync_from_device();

        auto time_end_q = std::chrono::high_resolution_clock::now();

        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto time_q_proj = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_q - time_start_q).count();
            std::cout << "Time spent on Q projection of layer " << layer_idx << ": " << time_q_proj << " ms" << std::endl;
        #endif
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - Q projection output "<<std::endl;
            buffer<bf16> q_proj_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(q_proj_out_reference, "q_vision_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_q_output.data(), q_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );

        #endif

        auto time_start_k = std::chrono::high_resolution_clock::now();
        attn_k_weight[layer_idx].sync_to_device();
        mm_qkv_input.sync_to_device();
        vision_k_projection_app(
            mm_qkv_input,
            attn_k_weight[layer_idx],
            mm_k_output
        );
        mm_k_output.sync_from_device();
        auto time_end_k = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto time_k_proj = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_k - time_start_k).count();
            std::cout << "Time spent on K projection of layer " << layer_idx << ": " << time_k_proj << " ms" << std::endl;
        #endif
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - K projection output "<<std::endl;
            buffer<bf16> k_proj_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(k_proj_out_reference, "k_vision_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_k_output.data(), k_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif

        auto run_v = vision_v_projection_app.create_run(  mm_qkv_input,
            attn_v_weight[layer_idx],
            mm_v_output);
        auto time_start_v = std::chrono::high_resolution_clock::now();
        attn_v_weight[layer_idx].sync_to_device();
        mm_qkv_input.sync_to_device();
        // vision_v_projection_app(
        //     mm_qkv_input,
        //     attn_v_weight[layer_idx],
        //     mm_v_output
        // );

        run_v.start();
        apply_rotary_pos_emb_vision_avx512(
            mm_q_output.data(),
            mm_k_output.data(),
            cos_emb.data(),
            sin_emb.data(),
            seq_len,
            QWEN2_VISION_HIDDEN_SIZE,
            QWEN2_VISION_HEAD_DIM
        );
        mm_q_output.sync_to_device();
        mm_k_output.sync_to_device();
        flm::npu::wait_for_completion(run_v, "run_v");

        mm_v_output.sync_from_device();
        auto time_end_v = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto time_v_proj = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_v - time_start_v).count();
            std::cout << "Time spent on V projection of layer " << layer_idx << ": " << time_v_proj << " ms" << std::endl;
        #endif
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - V projection output "<<std::endl;
            buffer<bf16> v_proj_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(v_proj_out_reference, "v_vision_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_v_output.data(), v_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - Q after RoPE "<<std::endl;
            buffer<bf16> q_after_rope_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(q_after_rope_reference, "q_vision_rotary_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_q_output.data(), q_after_rope_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );

            std::cout << "Layer " << layer_idx << " - K after RoPE "<<std::endl;
            buffer<bf16> k_after_rope_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(k_after_rope_reference, "k_vision_rotary_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_k_output.data(), k_after_rope_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif

        if(layer_idx == 7 || layer_idx == 15 || layer_idx == 23 ||layer_idx == 31){
            this->mm_v_output.sync_to_device();
            auto time_start_attention = std::chrono::high_resolution_clock::now();
            vision_flash_attention_app(
                mm_q_output,
                mm_k_output,
                mm_v_output,
                window_attention_o_output
            );
            auto time_end_attention = std::chrono::high_resolution_clock::now();
            #if DEBUG_PRINT_ENCODE_TIME_DETAIL
                auto time_attention = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_attention - time_start_attention).count();
                std::cout << "Time spent on FlashAttention of layer " << layer_idx << ": " << time_attention << " ms" << std::endl;
            #endif
            window_attention_o_output.sync_from_device();

            #if DEBUG_PRINT_ENCODE_ERROR_METRICS
                std::cout << "Layer " << layer_idx << " - Attention output "<<std::endl;
                buffer<bf16> attn_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
                reference_tensor.load_weights(attn_out_reference, "attn_output_vision_" + std::to_string(layer_idx));
                print_error_metrics<bf16, bf16>(
                    window_attention_o_output.data(), attn_out_reference.data(),
                    1,
                    seq_len, QWEN2_VISION_HIDDEN_SIZE,
                    seq_len, QWEN2_VISION_HIDDEN_SIZE
                );
            #endif
        }else{
            this->mm_v_output.sync_to_device();
            auto time_start_attention = std::chrono::high_resolution_clock::now();
            vision_window_attention_app(
                mm_q_output,
                mm_k_output,
                mm_v_output,
                window_attention_o_output
            );
            window_attention_o_output.sync_from_device();
            auto time_end_attention = std::chrono::high_resolution_clock::now();
            #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
                auto time_attention = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_attention - time_start_attention).count();
                std::cout << "Time spent on Attention of layer " << layer_idx << ": " << time_attention << " ms" << std::endl;
            #endif

             #if DEBUG_PRINT_ENCODE_ERROR_METRICS
                std::cout << "Layer " << layer_idx << " - Attention output "<<std::endl;
                buffer<bf16> attn_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
                reference_tensor.load_weights(attn_out_reference, "attn_output_vision_" + std::to_string(layer_idx));
                print_error_metrics<bf16, bf16>(
                    window_attention_o_output.data(), attn_out_reference.data(),
                    1,
                    seq_len, QWEN2_VISION_HIDDEN_SIZE,
                    seq_len, QWEN2_VISION_HIDDEN_SIZE
                );
            #endif
        }

        this->window_attention_o_output.sync_to_device();
        attn_o_weight[layer_idx].sync_to_device();
        vision_o_projection_app(
            window_attention_o_output,
            attn_o_weight[layer_idx],
            mm_o_output_fc1_input
        );

        this->mm_o_output_fc1_input.sync_from_device();

            #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - Attention output after o_proj "<<std::endl;
            buffer<bf16> attn_o_proj_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(attn_o_proj_out_reference, "attn_output_proj_vision_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_o_output_fc1_input.data(), attn_o_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
            #endif

        // update mm_o_output_fc1_input with residual
        // Note: the first half hidden_state is untouched, we can directly add to mm_o_output_fc1_input
        simd_add(mm_o_output_fc1_input.data(), hidden_state.data(), mm_o_output_fc1_input.data(), seq_len * QWEN2_VISION_HIDDEN_SIZE);

        mm_o_output_fc1_input.sync_to_device();
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - hidden state after adding residual "<<std::endl;
            buffer<bf16> after_residual_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(after_residual_reference, "hidden_states_after_attn_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_o_output_fc1_input.data(), after_residual_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif

        // okay, now the residual 2 and norm 2
        memcpy(residual.data(), mm_o_output_fc1_input.data(), sizeof(bf16)*seq_len_padded*QWEN2_VISION_HIDDEN_SIZE);

        mm_o_output_fc1_input.sync_to_device();
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - residual2 after after attn output "<<std::endl;
            buffer<bf16> residual2_before_attn_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(residual2_before_attn_reference, "residual_2_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                residual.data(), residual2_before_attn_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif
        this->_rms_norm(mm_o_output_fc1_input, residual, norm_2_weight[layer_idx]);

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - hidden state after RMSNorm 2 "<<std::endl;
            buffer<bf16> rms_2_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(rms_2_out_reference, "norm_2_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_o_output_fc1_input.data(), rms_2_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif

        // now mlp
        mm_o_output_fc1_input.sync_to_device();
        mlp_gate_proj_weight[layer_idx].sync_to_device();
        auto time_start_mlp_gate = std::chrono::high_resolution_clock::now();
        vision_mlp_gate_proj_app(
            mm_o_output_fc1_input,
            mlp_gate_proj_weight[layer_idx],
            mm_gate_output
        );
        auto time_end_mlp_gate = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto time_mlp_gate = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_gate - time_start_mlp_gate).count();
            std::cout << "Time spent on MLP gate projection of layer " << layer_idx << ": " << time_mlp_gate << " ms" << std::endl;
        #endif
        mm_gate_output.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - MLP gate projection output "<<std::endl;
            buffer<bf16> mlp_gate_proj_out_reference(seq_len * QWEN2_VISION_MLP_INTERMEDIATE_SIZE);
            reference_tensor.load_weights(mlp_gate_proj_out_reference, "mlp_gate_nonlinear_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_gate_output.data(), mlp_gate_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_MLP_INTERMEDIATE_SIZE,
                seq_len, vision_intermediate_size_padded
            );
        #endif

        mm_o_output_fc1_input.sync_to_device();
        mlp_up_proj_weight[layer_idx].sync_to_device();
        auto time_start_mlp_up = std::chrono::high_resolution_clock::now();
        vision_mlp_up_proj_app(
            mm_o_output_fc1_input,
            mlp_up_proj_weight[layer_idx],
            mm_up_output
        );
        auto time_end_mlp_up = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto time_mlp_up = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_up - time_start_mlp_up).count();
            std::cout << "Time spent on MLP up projection of layer " << layer_idx << ": " << time_mlp_up << " ms" << std::endl;
        #endif
        mm_up_output.sync_from_device();
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - MLP up projection output "<<std::endl;
            buffer<bf16> mlp_up_proj_out_reference(seq_len * QWEN2_VISION_MLP_INTERMEDIATE_SIZE);
            reference_tensor.load_weights(mlp_up_proj_out_reference, "mlp_up_proj_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_up_output.data(), mlp_up_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_MLP_INTERMEDIATE_SIZE,
                seq_len, vision_intermediate_size_padded
            );
        #endif

        // now do element wise multiply of gate and up
        simd_mul(mm_gate_output.data(), mm_up_output.data(), mm_gate_output.data(), (size_t)seq_len * vision_intermediate_size_padded);
        // now, result is in mm_gate_output

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - MLP after element wise multiply "<<std::endl;
            buffer<bf16> mlp_after_mul_reference(seq_len * QWEN2_VISION_MLP_INTERMEDIATE_SIZE);
            reference_tensor.load_weights(mlp_after_mul_reference, "mlp_non_linear_mul_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_gate_output.data(), mlp_after_mul_reference.data(),
                1,
                seq_len, QWEN2_VISION_MLP_INTERMEDIATE_SIZE,
                seq_len, vision_intermediate_size_padded
            );
        #endif

        mm_gate_output.sync_to_device();
        mlp_down_proj_weight[layer_idx].sync_to_device();
        auto time_start_mlp_down = std::chrono::high_resolution_clock::now();
        vision_mlp_down_proj_app(
            mm_gate_output,
            mlp_down_proj_weight[layer_idx],
            mm_o_output_fc1_input
        );
        auto time_end_mlp_down = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto time_mlp_down = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_down - time_start_mlp_down).count();
            std::cout << "Time spent on MLP down projection of layer " << layer_idx << ": " << time_mlp_down << " ms" << std::endl;
        #endif
        mm_o_output_fc1_input.sync_from_device();
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - MLP down projection output "<<std::endl;
            buffer<bf16> mlp_down_proj_out_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(mlp_down_proj_out_reference, "mlp_down_proj_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                mm_o_output_fc1_input.data(), mlp_down_proj_out_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif

        // then hidden_States = resudual + mm_o_output_fc1_input
        simd_add(mm_o_output_fc1_input.data(), residual.data(), hidden_state.data(), seq_len * QWEN2_VISION_HIDDEN_SIZE);
        mm_o_output_fc1_input.sync_to_device();
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "Layer " << layer_idx << " - hidden state after MLP and adding residual "<<std::endl;
            buffer<bf16> after_mlp_residual_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
            reference_tensor.load_weights(after_mlp_residual_reference, "hidden_states_after_mlp_" + std::to_string(layer_idx));
            print_error_metrics<bf16, bf16>(
                hidden_state.data(), after_mlp_residual_reference.data(),
                1,
                seq_len, QWEN2_VISION_HIDDEN_SIZE,
                seq_len, QWEN2_VISION_HIDDEN_SIZE
            );
        #endif
    }

    // // DEBUG FOR NOW

    // post norm
    this->_rms_norm(this->merger_mlp_fc1_input, hidden_state, post_norm_weight);
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "After post norm "<<std::endl;
        buffer<bf16> after_post_norm_reference(seq_len * QWEN2_VISION_HIDDEN_SIZE);
        reference_tensor.load_weights(after_post_norm_reference, "patch_merger_ln_q");
        print_error_metrics<bf16, bf16>(
            merger_mlp_fc1_input.data(), after_post_norm_reference.data(),
            1,
            seq_len, QWEN2_VISION_HIDDEN_SIZE,
            seq_len, QWEN2_VISION_HIDDEN_SIZE
        );
    #endif

    auto time_start_merger_mlp = std::chrono::high_resolution_clock::now();
    this->merger_mlp_fc1_input.sync_to_device();
    this->merger_mlp_fc1_weight.sync_to_device();
    merger_mlp_fc1_app(
        this->merger_mlp_fc1_input,
        this->merger_mlp_fc1_weight,
        this->merger_mlp_fc1_out_fc2_in
    );
    auto time_end_merger_mlp =  std::chrono::high_resolution_clock::now();

    this->merger_mlp_fc1_out_fc2_in.sync_from_device(); //TODO: no necessary?

    // validate merger_mlp_fc1_out_fc2_in
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "After merger MLP fc1 "<<std::endl;
        buffer<bf16> after_merger_mlp_fc1_reference(merge_seq_len * QWEN2_MERGER_MLP_INTERMEDIATE_SIZE);
        reference_tensor.load_weights(after_merger_mlp_fc1_reference, "patch_merger_mlp_1");
        print_error_metrics<bf16, bf16>(
            merger_mlp_fc1_out_fc2_in.data(), after_merger_mlp_fc1_reference.data(),
            1,
            merge_seq_len, QWEN2_MERGER_MLP_INTERMEDIATE_SIZE,
            merge_seq_len, QWEN2_MERGER_MLP_INTERMEDIATE_SIZE
        );

    #endif

    auto time_start_merger_mlp_fc2 =  std::chrono::high_resolution_clock::now();
    this->merger_mlp_fc1_out_fc2_in.sync_to_device();
    this->merger_mlp_fc2_weight.sync_to_device();
    merger_mlp_fc2_app(
        this->merger_mlp_fc1_out_fc2_in,
        this->merger_mlp_fc2_weight,
        this->merger_mlp_fc2_out // reuse buffer
    );
    this->merger_mlp_fc2_out.sync_from_device();
    auto time_end_merger_mlp_fc2 =  std::chrono::high_resolution_clock::now();

    // compare error
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "After merger MLP "<<std::endl;
        buffer<bf16> after_merger_mlp_reference(merge_seq_len * QWEN2_VISION_OUT_HIDDEN_SIZE);
        reference_tensor.load_weights(after_merger_mlp_reference, "patch_merger_mlp");
        print_error_metrics<bf16, bf16>(
            merger_mlp_fc2_out.data(), after_merger_mlp_reference.data(),
            1,
            merge_seq_len, QWEN2_VISION_OUT_HIDDEN_SIZE,
            merge_seq_len, QWEN2_VISION_OUT_HIDDEN_SIZE
        );

    #endif

    // now, do the
    /*
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]
    */

    std::vector<int> reverse_indices(merge_seq_len);
    std::iota(reverse_indices.begin(), reverse_indices.end(), 0);
    // Sort indices based on the window_index (downsampled)
    // We compare based on the original raster index associated with each merged block
    int stride = QWEN2_SPATIAL_MERGE_SIZE * QWEN2_SPATIAL_MERGE_SIZE;
    std::sort(reverse_indices.begin(), reverse_indices.end(), [&](int a, int b) {
        return window_index[(size_t)a * stride] < window_index[(size_t)b * stride];
    });

    std::vector<bf16> final_vec( merge_seq_len* QWEN2_VISION_OUT_HIDDEN_SIZE);

    for (int i = 0; i < merge_seq_len; ++i) {
        int src_idx = reverse_indices[i];
        bf16* dst_ptr = final_vec.data() + i * QWEN2_VISION_OUT_HIDDEN_SIZE;
        bf16* src_ptr = merger_mlp_fc2_out.data() + src_idx * QWEN2_VISION_OUT_HIDDEN_SIZE;
        memcpy(dst_ptr, src_ptr, QWEN2_VISION_OUT_HIDDEN_SIZE * sizeof(bf16));
    }

    // compare the error
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "After reverse reorder for merger output "<<std::endl;
        buffer<bf16> after_reverse_reorder_reference(merge_seq_len * QWEN2_VISION_OUT_HIDDEN_SIZE);
        reference_tensor.load_weights(after_reverse_reorder_reference, "hidden_states_Final");
        print_error_metrics<bf16, bf16>(
            final_vec.data(), after_reverse_reorder_reference.data(),
            1,
            merge_seq_len, QWEN2_VISION_OUT_HIDDEN_SIZE,
            merge_seq_len, QWEN2_VISION_OUT_HIDDEN_SIZE
        );
    #endif

    // Pack into result format (vector of vectors usually implies batch or just returning the flat sequence wrapped)
    return final_vec;
}
