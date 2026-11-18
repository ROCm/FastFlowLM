// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "gamma_image.hpp"
#include "gemma_image_attention_sequence.hpp"
#include "gemma_image_sequence.hpp"
#include "gemma_image_util.hpp"
#include <thread>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif
// #define GEMMA_DEBUGGING 1
GemmaImageEncoder::~GemmaImageEncoder()
{
}

void GemmaImageEncoder::processConv2D_with_Bias(
    buffer< biovault::bfloat16_t> &conv2dOutput,
    const  biovault::bfloat16_t *image_ptr, int num_of_image, int C_in, int H_in, int W_in,
    int conv2d_stride,
    bool verify_value,
    bool print_time_details,
    SafeTensors *reference
){

    int C_out  = CONV2D_COUT;

    int K_H = CONV2D_KERNEL_SIZE;
    int K_W = CONV2D_KERNEL_SIZE; // Assume kernel width is available
    int K = K_H;                       // Use square kernel size

    // Correctly calculate both output height and width
    int H_out = (H_in - K) / conv2d_stride + 1;
    int W_out = (W_in - K) / conv2d_stride + 1;
    if (!(H_out == CONV2D_IMAGE_OUT_HEIGHT && W_out == CONV2D_IMAGE_OUT_WIDTH)) {
        std::cerr << "Error: H_out (" << H_out << ") != CONV2D_IMAGE_OUT_HEIGHT (" << CONV2D_IMAGE_OUT_HEIGHT
                  << ") or W_out (" << W_out << ") != CONV2D_IMAGE_OUT_WIDTH (" << CONV2D_IMAGE_OUT_WIDTH << ")" << std::endl;
        std::abort();
    }

    // Use H_out * W_out for size and offset calculations
    int single_output_feature_map_size = C_out * H_out * W_out;
    int conv2d_output_size = num_of_image * single_output_feature_map_size;
    #ifdef GEMMA_DEBUGGING
    auto start_conv2d = std::chrono::high_resolution_clock::now();
    #endif
    // first, do the convolution2D with bias
    for (int i = 0; i < num_of_image; i++)
    {
        if (!(conv2dOutput.size() >= (size_t)num_of_image * C_out * H_out * W_out)) {
            std::cerr << "Error: conv2dOutput size (" << conv2dOutput.size()
                      << ") is less than required (" << num_of_image * C_out * H_out * W_out << ")" << std::endl;
            std::abort();
        }
        //NOTE: The data output becomes [num_image, H_out*W_out, C_out]
        // For perfomrance reasnos
        conv2d_avx512_bf16_ptr_bias_hwc_out(
            image_ptr + i * (C_in * H_in * W_in),
            conv2d_kernels_weights.data(), conv2d_bias.data(),
            reinterpret_cast<biovault::bfloat16_t*>( conv2dOutput.data()) + i * (C_out * H_out * W_out), C_in, H_in, W_in, C_out, K, conv2d_stride);
    }
    #ifdef GEMMA_DEBUGGING
    auto end_conv2d = std::chrono::high_resolution_clock::now();
    if(print_time_details)
    {
        // Print the time taken for the convolution operation
        std::cout << "Time taken for Conv2D: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_conv2d - start_conv2d).count()
                << " ms" << std::endl;
    }
    if(verify_value && reference!=nullptr){

        // showe the errors

        std::cout << "Error matrix after conv2d" << std::endl;
        buffer<float> ref_conv2d;
        reference->load_weights(ref_conv2d, "after_conv2d_with_reordered");
        //NOTE: at this point, nothing is being padded yet
        print_error_metrics(ref_conv2d.data(), conv2dOutput.data(), num_of_image, H_out*W_out, C_out, H_out*W_out, C_out);
    }
    #endif
}

void GemmaImageEncoder::processEmbedding(
     biovault::bfloat16_t* __restrict data, //[batch_size, seq_len, hidden_size_padded_to_512]

    const int batch_size,
    const int sequence_length,
    const int hidden_size,
    const int hidden_size_padded,
    bool verify_value,
    bool print_time_details,
    SafeTensors *reference
)
{
    #ifdef GEMMA_DEBUGGING
    auto add_embedding_start = std::chrono::high_resolution_clock::now();
    #endif
    add_embedding(
        data, (const biovault::bfloat16_t *)image_embedding_weight.data(), batch_size, sequence_length, hidden_size, hidden_size_padded
    );
    #ifdef GEMMA_DEBUGGING
    auto end_add_embedding = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for add_embedding: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_add_embedding - add_embedding_start).count()
                << " ms" << std::endl;
    }

    if(verify_value && reference != nullptr){

        // showe the errors
        std::cout << "Error matrix after embedding" << std::endl;
        buffer<float> ref;
        reference->load_weights(ref, "after_embeddings");
        //NOTE: sequence_length is not padded, but hidden size is padded for this version
        print_error_metrics(data, ref.data(),  batch_size,sequence_length, hidden_size,  sequence_length, hidden_size_padded);
    }
    #endif
}

void GemmaImageEncoder::multi_head_attention(
    biovault::bfloat16_t* hidden_states,
    size_t layer_id,
    const size_t num_heads,
    const size_t head_dim,
    const size_t batch_size,
    const size_t sequence_len,
    const size_t sequence_len_padded,
    const size_t hidden_size_padded,
    bool print_details,
    SafeTensors *reference
)
{
   const size_t hidden_size = num_heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t AVX_FLOAT_COUNT = 16;

    int hidden_in_padded_size = batch_size * sequence_len_padded* hidden_size_padded;
    int q_weight_in_size = hidden_size_padded * hidden_size_padded;
    int q_projected_padded_size  = batch_size * sequence_len_padded * hidden_size_padded;

    //note: runtime sequence is pregenerate beforehand

    q_hidden_in_bo.sync_to_device();
    this->attn_q_weights[layer_id].sync_to_device();
    this->attn_k_weights[layer_id].sync_to_device();
    this->attn_v_weights[layer_id].sync_to_device();

    auto run_q = q_projection.create_run(q_hidden_in_bo, this->attn_q_weights[layer_id], q_projected_padded);
    auto run_k = k_projection.create_run(q_hidden_in_bo, this->attn_k_weights[layer_id], k_projected_padded);
    auto run_v = v_projection.create_run(q_hidden_in_bo, this->attn_v_weights[layer_id], v_projected_padded);
    run_q.start();
    flm::npu::wait_for_completion(run_q, "run_q");
    run_k.start();
    flm::npu::wait_for_completion(run_k, "run_k");
    run_v.start();
    flm::npu::wait_for_completion(run_v, "run_v");

    #ifdef GEMMA_DEBUGGING
    auto flash_attention_start = std::chrono::high_resolution_clock::now();
    #endif

    flash_attention(flash_attent_concatenated_out, q_projected_padded, k_projected_padded, v_projected_padded);
    // flash_attent_concatenated_out.sync_from_device();
    #ifdef GEMMA_DEBUGGING
    auto flash_attention_end = std::chrono::high_resolution_clock::now();
    if(print_details){
        std::cout << "Time taken for Flash Attention: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(flash_attention_end - flash_attention_start).count()
                << " ms" << std::endl;
    }

    #endif

    #ifdef GEMMA_DEBUGGING
        auto start_output_projection = std::chrono::high_resolution_clock::now();
    #endif
    this->attn_output_weights[layer_id].sync_to_device();
    output_projection(flash_attent_concatenated_out, this->attn_output_weights[layer_id], projected_output);
    projected_output.sync_from_device();
    #ifdef GEMMA_DEBUGGING
    auto end_output_projection = std::chrono::high_resolution_clock::now();
    if(print_details){
        std::cout << "Time taken for Output Projection: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_output_projection - start_output_projection).count()
                << " ms" << std::endl;
    }
    #endif
}

void GemmaImageEncoder::mlp_layer_avx512(
    biovault::bfloat16_t* hidden_space,             // [batch_size * seq_len, padded_hidden_size]
    size_t layer_id,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    int padded_hidden_size,
    int padded_intermediate_size,
    bool print_time_details,
    SafeTensors *reference
) {

    int fc1_hidden_size = batch_size*seq_len*padded_hidden_size;
    int fc1_weight_size = padded_hidden_size*padded_intermediate_size;
    int fc2_weight_size = fc1_weight_size; // same, but reverse the dimension

    int fc2_hidden_size = batch_size*seq_len* padded_intermediate_size;

    //NOTEL runtime sequence is pregenerate beforehand
    #ifdef GEMMA_DEBUGGING
    auto start_fc1 = std::chrono::high_resolution_clock::now();
    #endif
    fc1_hidden.sync_to_device();

    fc1_weights[layer_id].sync_to_device();
    fc1(  fc1_hidden, fc1_weights[layer_id], fc1_res);

    #ifdef GEMMA_DEBUGGING
    auto end_fc1 = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for FC1 with gelu and bias "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_fc1 - start_fc1).count()
                  << " ms" << std::endl;
    }
    auto start_fc2 = std::chrono::high_resolution_clock::now();
    #endif

    fc2_weights[layer_id].sync_to_device();
    fc2(fc1_res, fc2_weights[layer_id], fc2_res);
    fc2_res.sync_from_device();
    #ifdef GEMMA_DEBUGGING
    auto end_fc2 = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for FC2 "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_fc2 - start_fc2).count()
                  << " ms" << std::endl;
    }
    #endif
}

void GemmaImageEncoder::processEncoderLayer(
     biovault::bfloat16_t * __restrict data, //[batch_size, seq_len_padded, hidden_size_padded]
    const int batch_size, const int sequence_length, const int hidden_size,
    const int sequence_length_padded, // this is the padded sequence length
    const int hidden_size_padded, // this is the padded hidden size
    const int layer_id,
    const size_t num_heads,
    const size_t head_dim, const size_t intermediate_size,
    const size_t padded_intermediate_size,

    const float layernorm_eps,
    bool verify_value,
    bool print_time_details,
    SafeTensors *reference
){

    if(num_heads != 16 || head_dim!=72){
        std::cerr << "Error: num_heads and head_dim must be 16 and 72 respectively for this implementation." << std::endl;
        std::abort();
    }

    // residual is preallocated already
    std::memcpy(residual_1.data(), data, (size_t)batch_size*sequence_length_padded*hidden_size_padded*sizeof(biovault::bfloat16_t));
    // norm 1
    #ifdef GEMMA_DEBUGGING
    auto start_norm1 = std::chrono::high_resolution_clock::now();
    #endif
    layernorm_avx512_bf16_single_core(
        data, q_hidden_in_bo.data(), reinterpret_cast<const bf16*>( encoder_layer_norm_1_weights[layer_id].data()),
        reinterpret_cast<const bf16*>( encoder_layer_norm_1_bias[layer_id].data()),
        batch_size, sequence_length, sequence_length_padded, hidden_size, hidden_size_padded, layernorm_eps
    );
    #ifdef GEMMA_DEBUGGING
    auto end_norm1 = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for norm1: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_norm1 - start_norm1).count()
                  << " ms" << std::endl;
    }
    if(verify_value && reference != nullptr){

        std::cout << "Error matrix after norm1 at layer:" << std::to_string(layer_id) << std::endl;
        buffer<float> ref;
        reference->load_weights(ref,"layer_" + std::to_string(layer_id)+"_after_norm1" );
        print_error_metrics(data, ref.data(), batch_size, sequence_length, hidden_size, sequence_length_padded, hidden_size_padded);
    }

    auto attention_start = std::chrono::high_resolution_clock::now();
    #endif
    multi_head_attention(
        data,
        layer_id,
        num_heads, head_dim, batch_size, sequence_length, sequence_length_padded, hidden_size_padded,
        print_time_details,
        reference
    );
    #ifdef GEMMA_DEBUGGING
    auto attention_end = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for attention: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(attention_end - attention_start).count()
                  << " ms" << std::endl;
    }
    if(verify_value){
        buffer<float> ref;
        std::cout << "Error matrix after attention at layer:" << std::to_string(layer_id) << std::endl;
        reference->load_weights(ref,"layer_" + std::to_string(layer_id)+"_after_attention");
        print_error_metrics(projected_output.data(), ref.data(),
            batch_size, sequence_length, hidden_size, sequence_length_padded, hidden_size_padded
        );
    }
    #endif

    // now apply the residual back to the data
    vector_add_bfloat16(data, projected_output.data(), residual_1.data(),(size_t)batch_size*sequence_length_padded*hidden_size_padded);

    //residual 2 is preallocated
    std::memcpy(residual_2.data(), data, (size_t)batch_size*sequence_length_padded*hidden_size_padded*sizeof(biovault::bfloat16_t));

    //now apply layernorm2
    #ifdef GEMMA_DEBUGGING
    auto start_norm2 = std::chrono::high_resolution_clock::now();
    #endif
    layernorm_avx512_bf16_single_core(
        data, fc1_hidden.data(), reinterpret_cast<const bf16*>( encoder_layer_norm_2_weights[layer_id].data()),
        reinterpret_cast<const bf16*>( encoder_layer_norm_2_bias[layer_id].data()),
        batch_size, sequence_length, sequence_length_padded, hidden_size, hidden_size_padded, layernorm_eps
    );
    #ifdef GEMMA_DEBUGGING
    auto end_norm2 = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for norm2: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_norm2 - start_norm2).count()
                  << " ms" << std::endl;
    }

    if(verify_value && reference != nullptr){

        std::cout << "Error matrix after norm2 at layer:" << std::to_string(layer_id) << std::endl;
        buffer<float> ref;
        reference->load_weights(ref,"layer_" + std::to_string(layer_id)+"_after_norm2" );
        print_error_metrics(fc1_hidden.data(), ref.data(), batch_size, sequence_length, hidden_size, sequence_length_padded, hidden_size_padded);
    }

    auto mlp_start = std::chrono::high_resolution_clock::now();
    #endif
    mlp_layer_avx512(
        data, layer_id,
        batch_size,
        sequence_length, hidden_size, intermediate_size,
        hidden_size_padded, padded_intermediate_size,
        print_time_details,
        reference
    );
    #ifdef GEMMA_DEBUGGING
    auto mlp_end = std::chrono::high_resolution_clock::now();
    if(print_time_details){
        std::cout << "Time taken for MLP: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(mlp_end - mlp_start).count()
                  << " ms" << std::endl;
    }
    if(verify_value && reference != nullptr){

        std::cout << "Error matrix after MLP at layer:" << std::to_string(layer_id) << std::endl;
        buffer<float> ref;
        reference->load_weights(ref,"layer_" + std::to_string(layer_id)+"_after_mlp" );
        print_error_metrics(fc2_res.data(), ref.data(), batch_size, sequence_length, hidden_size, sequence_length_padded, hidden_size_padded);
    }

   // Add residual connection back
    auto add_residual_start = std::chrono::high_resolution_clock::now();
    #endif
    vector_add_bfloat16(data, fc2_res.data(), residual_2.data(),(size_t)batch_size*sequence_length_padded*hidden_size_padded);
    #ifdef GEMMA_DEBUGGING
    auto end_add_residual = std::chrono::high_resolution_clock::now();
    if(print_time_details)
    {
        std::cout << "Time taken for add residual: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_add_residual - add_residual_start).count()
                << " us" << std::endl;
    }
    if(verify_value){
        buffer<float> ref;
        reference->load_weights(ref,"layer_" + std::to_string(layer_id)+"_final");
        std::cout << "Error matrix this layer:" << std::to_string(layer_id) << std::endl;

        print_error_metrics(data, ref.data(), batch_size, sequence_length, hidden_size, sequence_length_padded, hidden_size_padded);
    }

    #endif
}

template <bool print_time_detail, bool verify_value>
buffer< bf16> GemmaImageEncoder::encode(buffer<bf16> pixel_values, SafeTensors *reference)
{

    int num_of_image = pixel_values.size() / (CONV2D_CIN * CONV2D_IMAGE_IN_HEIGHT * CONV2D_IMAGE_IN_WIDTH);
    multi_model_proj_seq_len_padded  = round_up_to_multiple(num_of_image* IMAGE_SEQ_LEN_AVERAGE_POOLING, mm_m*MM_ROW_SIZE);
    {
        generate_mm_sequence<bf16,bf16>(
            *this->multi_model_proj.seq(),
            multi_model_proj_seq_len_padded, IMAGE_HIDDEN_SIZE_PADDED, MULTI_MODAL_HIDDEN_SIZE,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,  //TODO: for now
            false, 0 // No bias

        );

        generate_mm_sequence<bf16,bf16>(
            *this->fc1.seq(),
            num_of_image * sequence_length_per_image, IMAGE_HIDDEN_SIZE_PADDED, IMAGE_INTERMEDIATE_SIZE_PADDED,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,  //TODO: for now
            ADD_BIAS, 1

        );

        generate_mm_sequence< bf16, bf16>(
            *this->fc2.seq(),
            num_of_image * sequence_length_per_image, IMAGE_INTERMEDIATE_SIZE_PADDED, IMAGE_HIDDEN_SIZE_PADDED,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,  //TODO: for now
            ADD_BIAS, 0

        );

        // the instructions for q, k, v, output projection
        generate_mm_sequence< bf16, bf16>(
        *this->q_projection.seq(),
            sequence_length_per_image * num_of_image, IMAGE_HIDDEN_SIZE_PADDED, IMAGE_HIDDEN_SIZE_PADDED,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,  //TODO: for now
            ADD_BIAS, 0

        );
        generate_mm_sequence< bf16, bf16>(
            *this->k_projection.seq(),
            sequence_length_per_image * num_of_image, IMAGE_HIDDEN_SIZE_PADDED, IMAGE_HIDDEN_SIZE_PADDED,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0

        );
        generate_mm_sequence< bf16, bf16>(
            *this->v_projection.seq(),
            sequence_length_per_image * num_of_image, IMAGE_HIDDEN_SIZE_PADDED, IMAGE_HIDDEN_SIZE_PADDED,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0

        );
        generate_mm_sequence< bf16, bf16>(
            *this->output_projection.seq(),
            sequence_length_per_image * num_of_image, IMAGE_HIDDEN_SIZE_PADDED, IMAGE_HIDDEN_SIZE_PADDED,
            mm_m, mm_k, mm_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE, MM_COL_SIZE,
            0, 0, 0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            ADD_BIAS, 0
        );
        // // generate the flasth attention runtime sequence
        // // NOTEL for now, have flash attention generate at runtime
        // gen_mha_engine_seq(
        //     this->flash_attention.seq(),    // with emulation
        //     sequence_length_per_image,              // Same as padded for this case, so
        //     IMAGE_HIDDEN_SIZE_PADDED, IMAGE_HIDDEN_SIZE_PADDED, // The hiddensize for q, k, v is all being padded, same with the projections
        //     num_of_image, IMAGE_HIDDEN_SIZE_PADDED);

        // std::cout <<"IMAGE_HIDDEN_SIZE_PADDED : " << IMAGE_HIDDEN_SIZE_PADDED << std::endl;

        // setup_SHM_configuration<bf16, bf16>(
        //     *this->flash_attention.seq(),
        //     IMAGE_DH, IMAGE_NUM_HEADS,
        //     LQ_per_CT, LK_per_CT,
        //     LQ_internal, LK_internal,
        //     SHIM_L_SEQ_list, SHIM_S_SEQ_list, SHIM_S_SEQ_PADDED_list,
        //     NUM_OF_COLUMN, NUM_OF_CT_PER_COLUMN, CU_MODE,
        //     IMAGE_HIDDEN_SIZE_PADDED,IMAGE_HIDDEN_SIZE_PADDED,IMAGE_HIDDEN_SIZE_PADDED,IMAGE_HIDDEN_SIZE_PADDED,
        //     num_of_image
        // );

        // because q, k, v buffer all in on buffer after MM
        constexpr int ARG_Q = 1;
        constexpr int ARG_K = 2;
        constexpr int ARG_V = 3;
        constexpr int ARG_O = 0;

        AttentionConfig config = {
            (uint32_t)IMAGE_DH, (uint32_t)IMAGE_NUM_HEADS,
            LQ_per_CT, LK_per_CT,
            LQ_internal, LK_internal,
            NUM_OF_COLUMN, NUM_OF_CT_PER_COLUMN,
            CU_MODE,
            (uint32_t)num_of_image,

            ARG_Q, ARG_K, ARG_V, ARG_O,

            (uint32_t)IMAGE_HIDDEN_SIZE_PADDED, (uint32_t)IMAGE_HIDDEN_SIZE_PADDED, (uint32_t)IMAGE_HIDDEN_SIZE_PADDED,
            (uint32_t)IMAGE_HIDDEN_SIZE_PADDED,

            false,
            0,0,0
        };

        std::vector<int> SHIM_L_SEQ_list(num_of_image, sequence_length_per_image);
        std::vector<int> SHIM_S_SEQ_list(num_of_image, sequence_length_per_image);
        std::vector<int> SHIM_S_SEQ_PADDED_list(num_of_image, sequence_length_per_image);

        std::vector<int> Q_batch_offset_list(num_of_image);
        std::vector<int> K_batch_offset_list(num_of_image);
        std::vector<int> V_batch_offset_list(num_of_image);
        std::vector<int> O_batch_offset_list(num_of_image);

        for(int l = 0; l< num_of_image; l++){
            Q_batch_offset_list[l] = l* sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
            K_batch_offset_list[l] = l* sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
            V_batch_offset_list[l] = l* sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
            O_batch_offset_list[l] = l* sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
        }

        BatchMetadata batch_data = {
            SHIM_L_SEQ_list, SHIM_S_SEQ_list, SHIM_S_SEQ_PADDED_list,
            Q_batch_offset_list, K_batch_offset_list, V_batch_offset_list, O_batch_offset_list
        };

        setup_SHM_configuration<bf16, bf16>(
            *this->flash_attention.seq(),
            config,
            batch_data,
            1.0/std::sqrt(IMAGE_DH)
        );
    }

    if (num_of_image <= 0) {
        std::cerr << "Error: num_of_image must be greater than 0" << std::endl;
        std::abort();
    }
    if (pixel_values.size() % (CONV2D_CIN * CONV2D_IMAGE_IN_HEIGHT * CONV2D_IMAGE_IN_WIDTH) != 0) {
        std::cerr << "Error: pixel_values size is not a multiple of input image dimensions" << std::endl;
        std::abort();
    }

    buffer< biovault::bfloat16_t> conv2dOutput(num_of_image * CONV2D_COUT * CONV2D_IMAGE_OUT_HEIGHT * CONV2D_IMAGE_OUT_WIDTH);

    //conv2D
    processConv2D_with_Bias(
        conv2dOutput, reinterpret_cast<biovault::bfloat16_t*>(pixel_values.data()), num_of_image, CONV2D_CIN, CONV2D_IMAGE_IN_HEIGHT, CONV2D_IMAGE_IN_WIDTH,
        CONV2D_STRIDE, verify_value, print_time_detail, reference
    );

    // At this point, the output of convolution is [num_image, H_out*W_out, C_out] which is same as [num_image, hidden_size, sequence_length]
    // But for computation later, we need to reorder and padded into
    //  [num_image, 4096(sequence_length), hidden_size_multiple_of_512()]
    int padded_conv2d_output_size = num_of_image * sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
    buffer< biovault::bfloat16_t> image_hidden_space(padded_conv2d_output_size);
    #pragma omp parallel for schedule(static) num_threads(TOTAL_OPENMP_THREAD)
    for (int i = 0; i < num_of_image; ++i) {
        for (int j = 0; j < sequence_length_per_image; ++j) {
            // Source pointer for the current token
             biovault::bfloat16_t* src_ptr = conv2dOutput.data() + i * (sequence_length_per_image * IMAGE_HIDDEN_SIZE) + j * IMAGE_HIDDEN_SIZE;
            // Destination pointer for the current token in the padded buffer
             biovault::bfloat16_t* dest_ptr = image_hidden_space.data() + i * (sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED) + j * IMAGE_HIDDEN_SIZE_PADDED;

            // Copy the existing data
            std::memcpy(dest_ptr, src_ptr, IMAGE_HIDDEN_SIZE * sizeof( biovault::bfloat16_t));

            // Zero-pad the rest of the hidden dimension
            if (IMAGE_HIDDEN_SIZE_PADDED > IMAGE_HIDDEN_SIZE) {
                std::memset(dest_ptr + IMAGE_HIDDEN_SIZE, 0, (IMAGE_HIDDEN_SIZE_PADDED - IMAGE_HIDDEN_SIZE) * sizeof( biovault::bfloat16_t));
            }
        }
    }

    processEmbedding(
      image_hidden_space.data(), num_of_image, sequence_length_per_image,IMAGE_HIDDEN_SIZE, IMAGE_HIDDEN_SIZE_PADDED, verify_value,print_time_detail , reference
    );

    {
        int hidden_in_padded_size = num_of_image * sequence_length_per_image* IMAGE_HIDDEN_SIZE_PADDED;
        int q_weight_in_size = IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED;
        int q_projected_padded_size  = num_of_image * sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
        // preallocat buffers to avoid allocation at runtime for attention and mlp laters
        this->q_hidden_in_bo = q_projection.create_bo_buffer< biovault::bfloat16_t>(hidden_in_padded_size);
        this->q_projected_padded = q_projection.create_bo_buffer<bf16>(q_projected_padded_size);

        this->k_projected_padded =  k_projection.create_bo_buffer<bf16>(q_projected_padded_size);

        this->v_projected_padded = v_projection.create_bo_buffer<bf16>(q_projected_padded_size);

        int size_in_one_batch = sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED;
        // flash attention
        this->flash_attent_concatenated_out = flash_attention.create_bo_buffer< biovault::bfloat16_t>((size_t)num_of_image * size_in_one_batch);

        // o projection
        this->projected_output = output_projection.create_bo_buffer<bf16>(q_projected_padded_size);

        // mlp pre allocation
        int fc1_hidden_size = num_of_image*sequence_length_per_image*IMAGE_HIDDEN_SIZE_PADDED;
        int fc1_weight_size = IMAGE_HIDDEN_SIZE_PADDED*IMAGE_INTERMEDIATE_SIZE_PADDED;
        int fc2_weight_size = fc1_weight_size; // same, but reverse the dimension

        int fc2_hidden_size = num_of_image*sequence_length_per_image* IMAGE_INTERMEDIATE_SIZE_PADDED;

        this->fc1_hidden = fc1.create_bo_buffer<biovault::bfloat16_t>(fc1_hidden_size);
        this->fc1_res = fc1.create_bo_buffer<bf16>( fc2_hidden_size);
        this->fc2_res = fc2.create_bo_buffer<bf16>(  fc1_hidden_size);

        residual_1.resize(hidden_in_padded_size);
        residual_2.resize(hidden_in_padded_size);
        this->multi_model_proj_hidden = this->multi_model_proj.create_bo_buffer< biovault::bfloat16_t>((size_t)multi_model_proj_seq_len_padded *IMAGE_HIDDEN_SIZE_PADDED );
        this->multi_model_proj_res = multi_model_proj.create_bo_buffer<bf16>(multi_model_proj_seq_len_padded * MULTI_MODAL_HIDDEN_SIZE);
    }

    // // now, process the encoder layers
    for(int i = 0; i < VISION_LAYER; i++){//NOTE: for gemma npu, only 1 layer is used for image encoder

        processEncoderLayer(image_hidden_space.data(),
            num_of_image, sequence_length_per_image, IMAGE_HIDDEN_SIZE,
            sequence_length_per_image, IMAGE_HIDDEN_SIZE_PADDED, // this is the padded sequence length and hidden size. Note: sequence_length is not padded, but hidden size is padded for this version
            i,
            IMAGE_NUM_HEADS, IMAGE_DH,IMAGE_INTERMEDIATE_SIZE, IMAGE_INTERMEDIATE_SIZE_PADDED,
            VISION_LAYER_NORM_EPS,
            verify_value ,print_time_detail, reference
        );
    }
    // //DEBUG: remove later

    //     print_error_metrics(image_hidden_space.data(), ref.data(), num_of_image, sequence_length_per_image, IMAGE_HIDDEN_SIZE, sequence_length_per_image, IMAGE_HIDDEN_SIZE_PADDED );

    //     for(int b = 0; b< num_of_image; b++){
    //         for(int s =0; s< sequence_length_per_image; s++){
    //             // memcpy(
    //             //     image_hidden_space.data() + b* sequence_length_per_image * IMAGE_HIDDEN_SIZE_PADDED + s* IMAGE_HIDDEN_SIZE_PADDED,
    //             //     ref.data() + b* sequence_length_per_image * IMAGE_HIDDEN_SIZE + s* IMAGE_HIDDEN_SIZE,
    //             //     IMAGE_HIDDEN_SIZE * sizeof(bf16)
    //             // );

    // now apply the post layer norm

    auto start_post_layernorm = std::chrono::high_resolution_clock::now();
    layernorm_avx512_bf16_single_core(
        image_hidden_space.data(),
        image_hidden_space.data(),
        static_cast<const bf16*>(this->post_layer_norm_weight.data()), static_cast<const bf16*>(this->post_layer_norm_bias.data()),
        num_of_image, sequence_length_per_image, sequence_length_per_image,
        IMAGE_HIDDEN_SIZE, IMAGE_HIDDEN_SIZE_PADDED, VISION_LAYER_NORM_EPS  //NOTE: assume  post layer norm eps is same with encoder
    );
    auto end_post_layernorm = std::chrono::high_resolution_clock::now();
    if(print_time_detail){
        std::cout << "Time taken for post layer norm: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_post_layernorm - start_post_layernorm).count()
                  << " ms" << std::endl;
    }

    if(verify_value && reference != nullptr){
        buffer<float> ref;
        reference->load_weights(ref,"after_post_layernorm");
        std::cout << "Error matrix after post layer norm" << std::endl;

        print_error_metrics(image_hidden_space.data(), ref.data(), num_of_image, sequence_length_per_image, IMAGE_HIDDEN_SIZE, sequence_length_per_image, IMAGE_HIDDEN_SIZE_PADDED );
    }

    // average pooling for it
    if(AVERAGE_POOLING_KERNEL_SIZE != 4  || AVERAGE_POOLING_STRIDE!= 4 ||  AVERAGE_POOLING_PADDING!=0){
        std::cerr << "Error: AVERAGE_POOLING_KERNEL_SIZE, AVERAGE_POOLING_STRIDE, and AVERAGE_POOLING_PADDING must be 4, 4, and 0 respectively for this implementation." << std::endl;
        std::abort();
    }
    auto start_average_pooling = std::chrono::high_resolution_clock::now();
    int sequence_length_after_average_pooling = average_pooling_avx512(
        image_hidden_space.data(),

        num_of_image, sequence_length_per_image, sequence_length_per_image,
        IMAGE_HIDDEN_SIZE, IMAGE_HIDDEN_SIZE_PADDED, AVERAGE_POOLING_KERNEL_SIZE, AVERAGE_POOLING_STRIDE, AVERAGE_POOLING_PADDING
    );
    if(sequence_length_after_average_pooling !=IMAGE_SEQ_LEN_AVERAGE_POOLING ){
        std::cerr << "Error: sequence_length_after_average_pooling does not match expected value." << std::endl;
        std::abort();
    }
    auto end_average_pooling = std::chrono::high_resolution_clock::now();

    if(print_time_detail){
        std::cout << "Time taken for average pooling: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_average_pooling - start_average_pooling).count()
            << " ms" << std::endl;
    }
    if(verify_value && reference != nullptr ){
        std::cout << "error after average pooling" << std::endl;
        buffer<float> ref;
        reference->load_weights(ref,"after_average_pooling");
        print_error_metrics(image_hidden_space.data(), ref.data(), num_of_image, sequence_length_after_average_pooling, IMAGE_HIDDEN_SIZE, sequence_length_after_average_pooling, IMAGE_HIDDEN_SIZE_PADDED);
    }

    // now gemma rms norm
    //NOTE: Gemma RMS norm requires adding 1.0 to weights
    // This is Done beforehand
    auto start = std::chrono::high_resolution_clock::now();
    rmsnorm_avx512_bf16_single_core(
        this->multi_model_proj_hidden.data(), image_hidden_space.data(),
        this->gemma_rms_norm_weight.data(),
        num_of_image, sequence_length_after_average_pooling, sequence_length_after_average_pooling,
        IMAGE_HIDDEN_SIZE, IMAGE_HIDDEN_SIZE_PADDED, VISION_RMS_NORM_EPS
    );
    auto end = std::chrono::high_resolution_clock::now();
    if(print_time_detail){
        std::cout << "Time taken for GemmaRMSNorm: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms" << std::endl;
    }
    if(verify_value && reference != nullptr){
        buffer<float> ref;
        reference->load_weights(ref,"after_mm_soft_emb_norm");
        //NOTE: sequence_length is not padded, but hidden size is padded for this version
        print_error_metrics(image_hidden_space.data(), ref.data(),  num_of_image,sequence_length_after_average_pooling, IMAGE_HIDDEN_SIZE,
          sequence_length_after_average_pooling, IMAGE_HIDDEN_SIZE_PADDED);
    }

    // // final multimodal projection

    buffer<bf16> projection_res((size_t)num_of_image* sequence_length_after_average_pooling * MULTI_MODAL_HIDDEN_SIZE);

    auto start_matmul = std::chrono::high_resolution_clock::now();

    this->multi_model_proj_hidden.sync_to_device();
    this->multi_model_projection_weight.sync_to_device();
    this->multi_model_proj(multi_model_proj_hidden,multi_model_projection_weight, multi_model_proj_res
    );
    this->multi_model_proj_res.sync_from_device();

    auto end_matmul = std::chrono::high_resolution_clock::now();
    if(print_time_detail){
        std::cout << "Time taken for MM_DOWN_PROJECTION: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_matmul - start_matmul).count()
                << " ms" << std::endl;
    }

    if(verify_value && reference != nullptr){
        buffer<float> ref;
        reference->load_weights(ref,"after_mm_input_projection");
        std::cout << "Error matrix after MM_PROJECTION" << std::endl;
        //NOTE: sequence_length is not padded, but down_projection_size is the output dimension
        print_error_metrics(multi_model_proj_res.data(), ref.data(),
        num_of_image,
        IMAGE_SEQ_LEN_AVERAGE_POOLING, MULTI_MODAL_HIDDEN_SIZE,
        IMAGE_SEQ_LEN_AVERAGE_POOLING, MULTI_MODAL_HIDDEN_SIZE);
    }

    // memcpy(projection_res.data(), multi_model_proj_res.data(),
    //     projection_res.size() * sizeof(bf16)
    // );

    return multi_model_proj_res;
}

// Explicit template instantiations to provide definitions for common combinations
template buffer< bf16> GemmaImageEncoder::encode<false, false>(buffer<bf16> pixel_values, SafeTensors *reference);
template buffer< bf16> GemmaImageEncoder::encode<true,  false>(buffer<bf16> pixel_values, SafeTensors *reference);
template buffer< bf16> GemmaImageEncoder::encode<false, true >(buffer<bf16> pixel_values, SafeTensors *reference);
template buffer< bf16> GemmaImageEncoder::encode<true,  true >(buffer<bf16> pixel_values, SafeTensors *reference);

GemmaImageEncoder::GemmaImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance)
    : config(config), npu(npu_instance)
{
    this->model_path = config.model_path;

    IMAGE_HIDDEN_SIZE_PADDED = round_up_to_multiple(IMAGE_HIDDEN_SIZE, mm_k);
    IMAGE_INTERMEDIATE_SIZE_PADDED = round_up_to_multiple(IMAGE_INTERMEDIATE_SIZE, mm_k);
    IMAGE_SEQ_LEN_AVERAGE_POOLING_PADDED = round_up_to_multiple(IMAGE_SEQ_LEN_AVERAGE_POOLING, mm_m * MM_ROW_SIZE);
    #ifdef GEMMA_DEBUGGING
        // print out all the parameters above
        std::cout << "=== GemmaImageEncoder Parameters ===" << std::endl;
        std::cout << "IMAGE_DH: " << IMAGE_DH << std::endl;
        std::cout << "IMAGE_DQ: " << IMAGE_DQ << std::endl;
        std::cout << "IMAGE_DK: " << IMAGE_DK << std::endl;
        std::cout << "IMAGE_DV: " << IMAGE_DV << std::endl;
        std::cout << "IMAGE_NUM_HEADS: " << IMAGE_NUM_HEADS << std::endl;
        std::cout << "IMAGE_HIDDEN_SIZE: " << IMAGE_HIDDEN_SIZE << std::endl;
        std::cout << "IMAGE_INTERMEDIATE_SIZE: " << IMAGE_INTERMEDIATE_SIZE << std::endl;
        std::cout << "MULTI_MODAL_HIDDEN_SIZE: " << MULTI_MODAL_HIDDEN_SIZE << std::endl;
        std::cout << "CONV2D_CIN: " << CONV2D_CIN << std::endl;
        std::cout << "CONV2D_COUT: " << CONV2D_COUT << std::endl;
        std::cout << "CONV2D_KERNEL_SIZE: " << CONV2D_KERNEL_SIZE << std::endl;
        std::cout << "CONV2D_STRIDE: " << CONV2D_STRIDE << std::endl;
        std::cout << "CONV2D_PADDING: " << CONV2D_PADDING << std::endl;
        std::cout << "CONV2D_IMAGE_IN_HEIGHT: " << CONV2D_IMAGE_IN_HEIGHT << std::endl;
        std::cout << "CONV2D_IMAGE_IN_WIDTH: " << CONV2D_IMAGE_IN_WIDTH << std::endl;
        std::cout << "CONV2D_IMAGE_OUT_HEIGHT: " << CONV2D_IMAGE_OUT_HEIGHT << std::endl;
        std::cout << "CONV2D_IMAGE_OUT_WIDTH: " << CONV2D_IMAGE_OUT_WIDTH << std::endl;
        std::cout << "AVERAGE_POOLING_KERNEL_SIZE: " << AVERAGE_POOLING_KERNEL_SIZE << std::endl;
        std::cout << "AVERAGE_POOLING_PADDING: " << AVERAGE_POOLING_PADDING << std::endl;
        std::cout << "AVERAGE_POOLING_STRIDE: " << AVERAGE_POOLING_STRIDE << std::endl;
        std::cout << "AVERAGE_POOLING_IMAGE_IN_HEIGHT: " << AVERAGE_POOLING_IMAGE_IN_HEIGHT << std::endl;
        std::cout << "AVERAGE_POOLING_IMAGE_IN_WIDTH: " << AVERAGE_POOLING_IMAGE_IN_WIDTH << std::endl;
        std::cout << "AVERAGE_POOLING_IMAGE_OUT_HEIGHT: " << AVERAGE_POOLING_IMAGE_OUT_HEIGHT << std::endl;
        std::cout << "AVERAGE_POOLING_IMAGE_OUT_WIDTH: " << AVERAGE_POOLING_IMAGE_OUT_WIDTH << std::endl;
        std::cout << "VISION_LAYER: " << VISION_LAYER << std::endl;
        std::cout << "VISION_LAYER_NORM_EPS: " << VISION_LAYER_NORM_EPS << std::endl;
        std::cout << "VISION_RMS_NORM_EPS: " << VISION_RMS_NORM_EPS << std::endl;
        std::cout << "IMAGE_HIDDEN_SIZE_PADDED: " << IMAGE_HIDDEN_SIZE_PADDED << std::endl;
        std::cout << "IMAGE_INTERMEDIATE_SIZE_PADDED: " << IMAGE_INTERMEDIATE_SIZE_PADDED << std::endl;
        std::cout << "====================================" << std::endl;
    #endif

    this->sequence_length_per_image = CONV2D_IMAGE_OUT_HEIGHT * CONV2D_IMAGE_OUT_WIDTH;
    if (CONV2D_COUT != IMAGE_HIDDEN_SIZE) {
        std::cerr << "Error: CONV2D_COUT (" << CONV2D_COUT << ") != IMAGE_HIDDEN_SIZE (" << IMAGE_HIDDEN_SIZE << ")" << std::endl;
        std::abort();
    }

    // now, create the app for it
    this->fla = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_attn.xclbin"));
    this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));

    this->fc1 = this->proj->create_app();
    this->fc2 = this->proj->create_app();
    this->multi_model_proj = this->proj->create_app();
    this->q_projection = this->proj->create_app();
    this->k_projection = this->proj->create_app();
    this->v_projection = this->proj->create_app();
    this->output_projection = this->proj->create_app();
    this->flash_attention = this->fla->create_app();

    if (!this->npu->is_preemption_enabled()){
        this->qkv_projs = this->proj->create_runlist();
    }

    // first resize the vectors

    this->encoder_layer_norm_1_weights.resize(VISION_LAYER);
    this->encoder_layer_norm_1_bias.resize(VISION_LAYER);
    this->encoder_layer_norm_2_weights.resize(VISION_LAYER);
    this->encoder_layer_norm_2_bias.resize(VISION_LAYER);

    this->attn_q_weights.resize(VISION_LAYER);
    this->attn_k_weights.resize(VISION_LAYER);
    this->attn_v_weights.resize(VISION_LAYER);
    this->attn_output_weights.resize(VISION_LAYER);

    this->fc1_weights.resize(VISION_LAYER);
    this->fc2_weights.resize(VISION_LAYER);
}

void GemmaImageEncoder::init_weights(SafeTensors &q4nx)
{
    // loading the weights for the vision models
    //[hidden_size, C_in, Conv2d_kernelxConv2d_kernel]

    q4nx.load_weights(this->conv2d_kernels_weights, "vision_tower.vision_model.embeddings.patch_embedding.weight");
    if (this->conv2d_kernels_weights.size() !=  IMAGE_HIDDEN_SIZE * CONV2D_CIN * CONV2D_KERNEL_SIZE * CONV2D_KERNEL_SIZE) {
        std::cout << "Error: conv2d_kernels_weights size mismatch" << std::endl;
        std::cout << "Expected size: " <<  IMAGE_HIDDEN_SIZE * CONV2D_CIN * CONV2D_KERNEL_SIZE * CONV2D_KERNEL_SIZE << std::endl;
        std::cout << "  Actual size: " << this->conv2d_kernels_weights.size() << std::endl;
        std::abort();
    }

    //[hidden_size]
    q4nx.load_weights(this->conv2d_bias, "vision_tower.vision_model.embeddings.patch_embedding.bias");
    if (this->conv2d_bias.size() != IMAGE_HIDDEN_SIZE) {
        std::cout << "Error: conv2d_bias size mismatch" << std::endl;
        std::abort();
    }

    //[image_sequence_length, hidden_size]
    q4nx.load_weights(this->image_embedding_weight, "vision_tower.vision_model.embeddings.position_embedding.weight");
    if (this->image_embedding_weight.size() != this->sequence_length_per_image * IMAGE_HIDDEN_SIZE) {
        std::cout << "Error: image_embedding_weight size mismatch" << std::endl;
        std::abort();
    }

    for (int i = 0; i < VISION_LAYER; i++)
    {
        // the layernorm 1
        q4nx.load_weights(this->encoder_layer_norm_1_weights[i], "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".layer_norm1.weight");
        q4nx.load_weights(this->encoder_layer_norm_1_bias[i], "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".layer_norm1.bias");

        // layernorm 2
        q4nx.load_weights(this->encoder_layer_norm_2_weights[i], "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".layer_norm2.weight");
        q4nx.load_weights(this->encoder_layer_norm_2_bias[i], "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".layer_norm2.bias");

        // attention

        // bias and the weights
        this->attn_q_weights[i] = this->q_projection.create_bo_buffer< biovault::bfloat16_t>(IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED + IMAGE_HIDDEN_SIZE_PADDED );
        this->attn_k_weights[i] = this->k_projection.create_bo_buffer< biovault::bfloat16_t>(IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED + IMAGE_HIDDEN_SIZE_PADDED);
        this->attn_v_weights[i] = this->v_projection.create_bo_buffer< biovault::bfloat16_t>(IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED + IMAGE_HIDDEN_SIZE_PADDED);
        this->attn_output_weights[i] = this->output_projection.create_bo_buffer< biovault::bfloat16_t>(IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED+ IMAGE_HIDDEN_SIZE_PADDED);

        {

            buffer<bf16> temp_q_weight;
            buffer<bf16> temp_k_weight;
            buffer<bf16> temp_v_weight;
            buffer<bf16> temp_output_weight;
            buffer<bf16> temp_q_bias;
            buffer<bf16> temp_k_bias;
            buffer<bf16> temp_v_bias;
            buffer<bf16> temp_output_bias;
            q4nx.load_weights(temp_q_weight,  "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.q_proj.weight");
            q4nx.load_weights(temp_k_weight, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.k_proj.weight");
            q4nx.load_weights(temp_v_weight, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.v_proj.weight");
            q4nx.load_weights(temp_output_weight, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.out_proj.weight");

            q4nx.load_weights(temp_q_bias, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.q_proj.bias");
            q4nx.load_weights(temp_k_bias, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.k_proj.bias");
            q4nx.load_weights(temp_v_bias, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.v_proj.bias");
            q4nx.load_weights(temp_output_bias, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".self_attn.out_proj.bias");

            if(temp_q_weight.size() != (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED){
                std::cout << "Error: attn_q_weights size mismatch" << std::endl;
                std::abort();
            }
            if(temp_k_weight.size() != (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED){
                std::cout << "Error: attn_k_weights size mismatch" << std::endl;
                std::abort();
            }
            if(temp_v_weight.size() != (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED){
                std::cout << "Error: attn_v_weights size mismatch" << std::endl;
                std::abort();
            }
            if(temp_output_weight.size() != (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED){
                std::cout << "Error: attn_output_weights size mismatch" << std::endl;
                std::abort();
            }

            memcpy(this->attn_q_weights[i].data() +IMAGE_HIDDEN_SIZE_PADDED , temp_q_weight.data(), (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED * sizeof(bf16));
            memcpy(this->attn_k_weights[i].data() +IMAGE_HIDDEN_SIZE_PADDED , temp_k_weight.data(), (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED * sizeof(bf16));
            memcpy(this->attn_v_weights[i].data() +IMAGE_HIDDEN_SIZE_PADDED , temp_v_weight.data(), (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED * sizeof(bf16));
            memcpy(this->attn_output_weights[i].data() +IMAGE_HIDDEN_SIZE_PADDED , temp_output_weight.data(), (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED * sizeof(bf16));
            memcpy(this->attn_q_weights[i].data(), temp_q_bias.data(), temp_q_bias.size() * sizeof(bf16));
            memcpy(this->attn_k_weights[i].data(), temp_k_bias.data(), temp_k_bias.size() * sizeof(bf16));
            memcpy(this->attn_v_weights[i].data(), temp_v_bias.data(), temp_v_bias.size() * sizeof(bf16));
            memcpy(this->attn_output_weights[i].data(), temp_output_bias.data(), temp_output_bias.size() * sizeof(bf16));
        }

        this->fc1_weights[i] = this->fc1.create_bo_buffer< biovault::bfloat16_t>(IMAGE_HIDDEN_SIZE_PADDED * IMAGE_INTERMEDIATE_SIZE_PADDED+ IMAGE_INTERMEDIATE_SIZE_PADDED);
        this->fc2_weights[i] = this->fc2.create_bo_buffer< biovault::bfloat16_t>(IMAGE_HIDDEN_SIZE_PADDED * IMAGE_INTERMEDIATE_SIZE_PADDED + IMAGE_HIDDEN_SIZE_PADDED);
        {

            buffer<bf16> temp_fc1_weight;
            buffer<bf16> temp_fc2_weight;
            buffer<bf16> temp_fc1_bias;
            buffer<bf16> temp_fc2_bias;
            q4nx.load_weights( temp_fc1_weight, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc1.weight");
            q4nx.load_weights( temp_fc2_weight, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc2.weight");

            q4nx.load_weights( temp_fc1_bias, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc1.bias");
            q4nx.load_weights( temp_fc2_bias, "vision_tower.vision_model.encoder.layers." + std::to_string(i) + ".mlp.fc2.bias");

            if(temp_fc1_weight.size() != (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_INTERMEDIATE_SIZE_PADDED){
                std::cout << "Error: fc1_weights size mismatch" << std::endl;
                std::abort();
            }
            if(temp_fc2_weight.size() != (size_t)IMAGE_INTERMEDIATE_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED){
                std::cout << "Error: fc2_weights size mismatch" << std::endl;
                std::abort();
            }

            memcpy(this->fc1_weights[i].data() + IMAGE_INTERMEDIATE_SIZE_PADDED, temp_fc1_weight.data(), (size_t)IMAGE_HIDDEN_SIZE_PADDED * IMAGE_INTERMEDIATE_SIZE_PADDED * sizeof(bf16));
            memcpy(this->fc2_weights[i].data() + IMAGE_HIDDEN_SIZE_PADDED, temp_fc2_weight.data(), (size_t)IMAGE_INTERMEDIATE_SIZE_PADDED * IMAGE_HIDDEN_SIZE_PADDED * sizeof(bf16));
            memcpy(this->fc1_weights[i].data(), temp_fc1_bias.data(), temp_fc1_bias.size() * sizeof(bf16));
            memcpy(this->fc2_weights[i].data(), temp_fc2_bias.data(), temp_fc2_bias.size() * sizeof(bf16));
        }
    }
    // post layernorm
    q4nx.load_weights(this->post_layer_norm_weight, "vision_tower.vision_model.post_layernorm.weight");
    q4nx.load_weights(this->post_layer_norm_bias, "vision_tower.vision_model.post_layernorm.bias");

    // gemma rms norm (already +1 during weights preprocessing)
    q4nx.load_weights(this->gemma_rms_norm_weight, "multi_modal_projector.mm_soft_emb_norm.weight");
    {
        buffer<bf16> multimodal_weight;
        q4nx.load_weights(multimodal_weight, "multi_modal_projector.mm_input_projection_weight");
        if (multimodal_weight.size() !=  MULTI_MODAL_HIDDEN_SIZE * IMAGE_HIDDEN_SIZE_PADDED) {
            std::cout << "Error: multi_model_projection_weight size mismatch" << std::endl;
            std::abort();
        }

        this->multi_model_projection_weight = this->multi_model_proj.create_bo_buffer<bf16>(MULTI_MODAL_HIDDEN_SIZE * IMAGE_HIDDEN_SIZE_PADDED);
        memcpy(this->multi_model_projection_weight.data(), multimodal_weight.data(), MULTI_MODAL_HIDDEN_SIZE * IMAGE_HIDDEN_SIZE_PADDED * sizeof(bf16));
    }

    #ifdef GEMMA_DEBUGGING
        std::cout << "GemmaImageEncoder weights loaded successfully." << std::endl;
    #endif
}
