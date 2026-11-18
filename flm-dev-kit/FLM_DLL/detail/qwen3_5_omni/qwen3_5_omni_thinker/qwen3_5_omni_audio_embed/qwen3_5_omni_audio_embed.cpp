// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file qwen3_5_omni_audio_embed.cpp
/// \brief qwen3_5_omni audio embedder implementation — simple stub.
/// \author FastFlowLM Team
#include "qwen3_5_omni_audio_embed.hpp"
#include "utils/error_measure.hpp"
#include "conv2d.hpp"
#include <chrono>
#include "npu_sequences/mmRuntimeSequence.hpp"
#include "vision/norm.hpp"
#include "vision/vision_prefill_helper.hpp"
#define DEBUG_PRINT_ENCODE_TIME_DETAIL (DEBUG_LEVEL >= 1)
#define DEBUG_PRINT_ENCODE_ERROR_METRICS (DEBUG_LEVEL >= 1)

// #define DEBUG_PRINT_ENCODE_TIME_DETAIL 1
// #define DEBUG_PRINT_ENCODE_ERROR_METRICS 1

qwen3_5_omni_audio_embed::qwen3_5_omni_audio_embed(LM_Config config, npu_xclbin_manager *npu_instance)
    : config(config), npu(npu_instance)
{
    desc.build(config);
    pos_emb.build(desc.max_source_positions, desc.d_model);

    padded_audio_conv_out_k_size = round_up_to_multiple(
        desc.conv_out_k, desc.audio_mm_tile_k
    );
    padded_audio_mlp_intermediate_size = round_up_to_multiple(
        desc.encoder_ffn_dim, desc.audio_mm_tile_k
    );
    padded_audio_hidden_size = round_up_to_multiple(
        desc.d_model, desc.audio_mm_tile_k
    );
    padded_audio_out_hidden_size = round_up_to_multiple(
        desc.output_dim,  desc.audio_mm_tile_k
    );
    seq_len_pad_requirement_for_MM = MM_ROW_SIZE* desc.audio_mm_tile_m;

    this->window_fla = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "window_attention.xclbin"));
    this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));

    this->proj_high= npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm_high.xclbin"));

    this->flash_window_attention_app = this->window_fla->create_app();
    this->conv_out_proj = this->proj_high->create_app();

    this->q_proj_app= this->proj->create_app();
    this->k_proj_app= this->proj->create_app();
    this->o_proj_app= this->proj->create_app();
    this->v_proj_app= this->proj->create_app();
    this->mlp_fc1_app= this->proj->create_app();
    this->mlp_fc2_app = this->proj->create_app();
    this->merger_fc1_app = this->proj->create_app();
    this->merger_fc2_app = this->proj->create_app();

    norm_1_bias.resize(desc.num_hidden_layers);
    norm_1_weight.resize(desc.num_hidden_layers);
    norm_2_bias.resize(desc.num_hidden_layers);
    norm_2_weight.resize(desc.num_hidden_layers);

    q_weight.resize(desc.num_hidden_layers);
    k_weight.resize(desc.num_hidden_layers);
    v_weight.resize(desc.num_hidden_layers);
    o_weight.resize(desc.num_hidden_layers);

    mlp_fc1_weight.resize(desc.num_hidden_layers);
    mlp_fc2_weight.resize(desc.num_hidden_layers);
}

qwen3_5_omni_audio_embed::~qwen3_5_omni_audio_embed() = default;

buffer<bf16> qwen3_5_omni_audio_embed::forward(qwen3_5_omni_audio_payload_t& payload)
{

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        SafeTensors reference_tensor(utils::path_join(config.model_path, "audio_prefill_data_debug.safetensors"));
        std::cout << "debugging qwen3.5 audio code audio" << std::endl;
    #endif

    // compare the input
    // given the mel_spectrograms are in [num_audio, frame, bin ] row major

    // which can be view as [ summation( frame_each_audio), bin] row major
    // chunk = n_window*2
    // which can be view as [  summation( frame_each_audio) /(chunk), chunk, bin   ] row major

    std::vector<bf16> total_input_chunked;
    int total_data_size = 0;

    int total_audio_frame = 0;
    for(int a_idx = 0; a_idx< payload.num_audios; a_idx++ ){
        total_data_size+= payload.mel_spectrograms[a_idx].size();
        total_audio_frame  += payload.mel_spectrogram_frames_per_audio[a_idx];
    }

    int bin = payload.mel_spectrogram_bins_per_audio[0];  // same for all audios
    assert(bin == desc.num_mel_bins);
    int chunk_size = desc.n_window * 2;
    total_input_chunked.resize(total_data_size);

    // now, do a transpose to have it be [ (num_audio*frame_each_audio)/chunk, bin, chunk] row major

    int gf = 0;  // running global frame index
    for (int a_idx = 0; a_idx < payload.num_audios; a_idx++) {
        const auto& mel = payload.mel_spectrograms[a_idx];   // [frames_a, bin] row-major
        int frames_a = mel.size() / bin;
        for (int f = 0; f < frames_a; f++, gf++) {
            int b = gf / chunk_size;         // which block
            int c = gf % chunk_size;         // position within chunk
            const bf16* src_row = mel.data() + (size_t)f * bin;
            bf16* dst_base = total_input_chunked.data()
                        + (size_t)b * chunk_size * bin   // block offset
                        + c;                             // chunk column
            for (int n = 0; n < bin; n++)
                dst_base[(size_t)n * chunk_size] = src_row[n];  // dst[b, n, c]
        }
    }

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "initial padded feature compare to reference error" <<std::endl;
        buffer<bf16>initial_padded_features;
        reference_tensor.load_weights(initial_padded_features, "initial_padded_features");
        assert(total_input_chunked.size() == initial_padded_features.size());
        print_error_metrics<bf16, bf16>(
            total_input_chunked.data(), initial_padded_features.data(),
            1,
            1, total_input_chunked.size(),
            1, total_input_chunked.size()

        );

        // //TODO: remove later
        // // to avoid accumulateion error, overwrite with reference
        // memcpy(total_input_chunked.data(), initial_padded_features.data(), initial_padded_features.size()*sizeof(bf16));

    #endif

    // total_input_chunked layout: [num_blocks, C_in=1, bin, chunk_size]
    // matches Python padded_feature [num_blocks, 1, num_mel_bins, chunk_size]
    int num_blocks = total_audio_frame / chunk_size;
    assert(total_audio_frame % chunk_size == 0);

    // const int H1 = (desc.num_mel_bins + 1) / 2, W1 = (chunk_size + 1) / 2;
    // const int H2 = (H1           + 1) / 2, W2 = (W1         + 1) / 2;
    // const int H3 = (H2           + 1) / 2, W3 = (W2         + 1) / 2;
    // const int H4 = (H3           + 1) / 2, W4 = (W3         + 1) / 2;
    // buffer<bf16> conv2dout4 = this->conv_out_proj.create_bo_buffer<bf16>(
    //     num_blocks*  (desc.downsample_hidden_size*H4 * W4 )); // TODO: for now
    std::vector<bf16> conv2dout4;
    int conv2dout4_b;
    int conv2dout4_t;

    {
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto t0 = std::chrono::high_resolution_clock::now();
        #endif
        audio_conv_downsample(
            total_input_chunked.data(),
            num_blocks,
            bin,
            chunk_size,
            conv_wt,

            conv2dout4,
            desc.downsample_hidden_size,
            conv2dout4_b,
            conv2dout4_t

        );
        #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            std::cout << "num_blocks" <<std::endl;
            std::cout << "b*t" <<conv2dout4_b* conv2dout4_t <<std::endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            std::cout << "[timing] audio_conv_downsample: " << ms << " ms"
                        << "  (" << num_blocks << " blocks)\n";
        #endif
    }

    int window_attention_len =  conv2dout4_t *(desc.n_window_infer/ (desc.n_window*2));

    auto seq_len =conv2dout4_b*conv2dout4_t;
    assert((size_t)seq_len* desc.conv_out_k == conv2dout4.size());

    auto seq_len_pad_requirement_for_window_attention = window_attention_len*vision_num_of_rows;

    auto seq_len_padded_for_window_attention = round_up_to_multiple( seq_len,seq_len_pad_requirement_for_window_attention );
    auto seq_len_padded = round_up_to_multiple( seq_len_padded_for_window_attention,
                                 seq_len_pad_requirement_for_MM );

    buffer<bf16> conv_out_input = this->conv_out_proj.create_bo_buffer<bf16>(
         (size_t)seq_len_padded* padded_audio_conv_out_k_size
    );
    hidden_states = this->conv_out_proj.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_hidden_size
    );

    buffer<bf16> q_proj_out = this->q_proj_app.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_hidden_size
    );
    buffer<bf16> k_proj_out = this->k_proj_app.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_hidden_size
    );
    buffer<bf16> v_proj_out = this->v_proj_app.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_hidden_size
    );
    buffer<bf16>window_attention_out_o_proj_in = this->flash_window_attention_app.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_hidden_size
    );
    buffer<bf16> mlp_intermediate_buffer = this->mlp_fc1_app.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_mlp_intermediate_size
    );

    // memset to 0
    memset(conv_out_input.data(), 0, conv_out_input.size()*sizeof(bf16));
    for(int s = 0; s< seq_len; s++ ){
        size_t input_offset = (size_t)s* desc.conv_out_k;
        size_t output_offset = (size_t)s* padded_audio_conv_out_k_size;
        memcpy(
            conv_out_input.data() + output_offset,
            conv2dout4.data() +input_offset,
            desc.conv_out_k*sizeof(bf16)
        );
    }

    {
        generate_mm_sequence<bf16, bf16>(*this->conv_out_proj.seq(),
            seq_len_padded, padded_audio_conv_out_k_size,    padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            false, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );

        generate_mm_sequence<bf16, bf16>(*this->q_proj_app.seq(),
            seq_len_padded, padded_audio_hidden_size,    padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );
        generate_mm_sequence<bf16, bf16>(*this->k_proj_app.seq(),
            seq_len_padded, padded_audio_hidden_size,    padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );
        generate_mm_sequence<bf16, bf16>(*this->v_proj_app.seq(),
            seq_len_padded, padded_audio_hidden_size,    padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );
        generate_mm_sequence<bf16, bf16>(*this->o_proj_app.seq(),
            seq_len_padded, padded_audio_hidden_size,    padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );

        generate_mm_sequence<bf16, bf16>(*this->mlp_fc1_app.seq(),
            seq_len_padded, padded_audio_hidden_size,    padded_audio_mlp_intermediate_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 1,
            ENABLE_QKV_REORDER, desc.head_dim
        );
       generate_mm_sequence<bf16, bf16>(*this->mlp_fc2_app.seq(),
            seq_len_padded, padded_audio_mlp_intermediate_size, padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );

        generate_mm_sequence<bf16, bf16>(*this->merger_fc1_app.seq(),
            seq_len_padded, padded_audio_hidden_size,    padded_audio_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 1, //gelu
            ENABLE_QKV_REORDER, desc.head_dim
        );
       generate_mm_sequence<bf16, bf16>(*this->merger_fc2_app.seq(),
            seq_len_padded, padded_audio_hidden_size, padded_audio_out_hidden_size,
            desc.audio_mm_tile_m,desc.audio_mm_tile_k,desc.audio_mm_tile_n,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
            true, 0,
            ENABLE_QKV_REORDER, desc.head_dim
        );

        //NOTE: the window attention runtimesequence did not consider hiddensize is padded, thus
        assert(padded_audio_hidden_size == desc.d_model);
        window_attention_runtimeSequence<bf16, bf16>(
            *this->flash_window_attention_app.seq(),
            1, window_attention_len,window_attention_len,desc.head_dim, desc.encoder_attention_heads,
            vision_num_of_columns, vision_num_of_rows,
            {seq_len_padded_for_window_attention}, {seq_len_padded_for_window_attention},
            0,0,0,0,
            0,1,2,3
        );
    }

    // compare the error with the reference

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            // conv2dout4 is now in [b, t, c*f] layout (padded_embed_permuted),
            // so compare directly against the Python reference without any permutation.
            std::cout << "comparing conv2d4 output (padded_embed_permuted) errors" << std::endl;
            buffer<bf16> conv4Res;
            reference_tensor.load_weights(conv4Res, "padded_embed_permuted");
            print_error_metrics<bf16, bf16>(
                conv2dout4.data(), conv4Res.data(),
                1,
                1, conv2dout4.size(),
                1, conv2dout4.size()
            );
    #endif

    conv_out_input.sync_to_device();
    audio_conv_out_weight.sync_to_device();
    conv_out_proj(
        conv_out_input, audio_conv_out_weight, hidden_states
    );
    hidden_states.sync_from_device();

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS

        std::cout << "comparing padded_embed_after_conv_out" <<std::endl;
        buffer<bf16> ref_padded_embed_after_conv_out;
        reference_tensor.load_weights(ref_padded_embed_after_conv_out, "padded_embed_after_conv_out");

        print_error_metrics<bf16, bf16>(
            hidden_states.data(), ref_padded_embed_after_conv_out.data(),
            1,
            seq_len, desc.d_model,
            seq_len_padded, padded_audio_hidden_size
        );
    #endif

    pos_emb.apply(hidden_states.data(), conv2dout4_t, conv2dout4_b);

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS

        std::cout << "comparing padded_embed_plus_positional_embedding" <<std::endl;
        buffer<bf16> ref_padded_embed_plus_pe;
        reference_tensor.load_weights(ref_padded_embed_plus_pe, "padded_embed_plus_positional_embedding");

        print_error_metrics<bf16, bf16>(
            hidden_states.data(), ref_padded_embed_plus_pe.data(),
            1,
            seq_len, desc.d_model,
            seq_len_padded, padded_audio_hidden_size
        );
    #endif

    this->residual_buffer.resize(hidden_states.size());

    for(int layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){

        memcpy(residual_buffer.data(), hidden_states.data(), sizeof(bf16)* hidden_states.size());

        layernorm_parallel(
            seq_len,
            desc.d_model,
            padded_audio_hidden_size,
            hidden_states.data(),
            norm_1_weight[layer_idx].data(),
            norm_1_bias[layer_idx].data(),
            desc.layer_norm_eps,
            hidden_states.data()
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS

            std::cout << "_atten_layer_norm" << std::endl;
            buffer<bf16> ref_atten_layer_norm;
            reference_tensor.load_weights(
                ref_atten_layer_norm,
                "layer_" + std::to_string(layer_idx)+"_atten_layer_norm"
            );

            print_error_metrics<bf16, bf16>(
                hidden_states.data(), ref_atten_layer_norm.data(),
                1,
                seq_len, desc.d_model,
                seq_len_padded, padded_audio_hidden_size
            );

        #endif

        // do q,k,v projection
        hidden_states.sync_to_device();
        q_weight[layer_idx].sync_to_device();
        q_proj_app(hidden_states, q_weight[layer_idx], q_proj_out);
        q_proj_out.sync_from_device();

        k_weight[layer_idx].sync_to_device();
        k_proj_app(hidden_states, k_weight[layer_idx], k_proj_out);
        k_proj_out.sync_from_device();

        v_weight[layer_idx].sync_to_device();
        v_proj_app(hidden_states, v_weight[layer_idx], v_proj_out);
        v_proj_out.sync_from_device();

        // compare the errors
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout << "comparing qkv proj" << std::endl;

            buffer<bf16> ref_q;
            buffer<bf16> ref_k;
            buffer<bf16> ref_v;

            reference_tensor.load_weights(
                ref_q,
                "layer_" + std::to_string(layer_idx)+"_q"
            );
            reference_tensor.load_weights(
                ref_k,
                "layer_" + std::to_string(layer_idx)+"_k"
            );
            reference_tensor.load_weights(
                ref_v,
                "layer_" + std::to_string(layer_idx)+"_v"
            );
            print_error_metrics<bf16, bf16>(
                q_proj_out.data(), ref_q.data(),
                1,
                seq_len, desc.d_model,
                seq_len_padded, padded_audio_hidden_size
            );

            print_error_metrics<bf16, bf16>(
                k_proj_out.data(), ref_k.data(),
                1,
                seq_len, desc.d_model,
                seq_len_padded, padded_audio_hidden_size
            );
            print_error_metrics<bf16, bf16>(
                v_proj_out.data(), ref_v.data(),
                1,
                seq_len, desc.d_model,
                seq_len_padded, padded_audio_hidden_size
            );

        #endif
        q_proj_out.sync_to_device();
        k_proj_out.sync_to_device();
        v_proj_out.sync_to_device();

        flash_window_attention_app( q_proj_out, k_proj_out, v_proj_out, window_attention_out_o_proj_in );
        window_attention_out_o_proj_in.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS

            std::cout << "comparing window attention" << std::endl;

            buffer<bf16> ref_attention_o;
            reference_tensor.load_weights(
                ref_attention_o,
                "layer_" + std::to_string(layer_idx)+"attention_out"
            );
            print_error_metrics<bf16, bf16>(
                window_attention_out_o_proj_in.data(), ref_attention_o.data(),
                1,
                seq_len, desc.d_model,
                seq_len_padded, padded_audio_hidden_size
            );

        #endif

        window_attention_out_o_proj_in.sync_to_device();
        this->o_weight[layer_idx].sync_to_device();
        o_proj_app(window_attention_out_o_proj_in,this->o_weight[layer_idx], hidden_states);
        hidden_states.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            std::cout  << "Attention out project" << std::endl;
            buffer<bf16> ref_attention_o_proj;
            reference_tensor.load_weights(
                ref_attention_o_proj,
                "layer_" + std::to_string(layer_idx)+"_after_attention"
            );
          print_error_metrics<bf16, bf16>(
                hidden_states.data(), ref_attention_o_proj.data(),
                1,
                seq_len, desc.d_model,
                seq_len_padded, padded_audio_hidden_size
            );

        #endif

        // perform the residual
        simd_add(
            hidden_states.data(),
            residual_buffer.data(),
            hidden_states.data(),
            (size_t)seq_len* padded_audio_hidden_size
        );

        memcpy(
            residual_buffer.data(),
            hidden_states.data(),
            (size_t)seq_len* padded_audio_hidden_size*sizeof(bf16)
        );

        layernorm_parallel(
            seq_len,
            desc.d_model,
            padded_audio_hidden_size,
            hidden_states.data(),
            norm_2_weight[layer_idx].data(),
            norm_2_bias[layer_idx].data(),
            desc.layer_norm_eps,
            hidden_states.data()
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
                std::cout << "_final_layer_norm" << std::endl;
                buffer<bf16> ref_final_layer_norm;
                reference_tensor.load_weights(
                    ref_final_layer_norm,
                    "layer_" + std::to_string(layer_idx)+"_final_layer_norm"
                );

                print_error_metrics<bf16, bf16>(
                    hidden_states.data(), ref_final_layer_norm.data(),
                    1,
                    seq_len, desc.d_model,
                    seq_len_padded, padded_audio_hidden_size
                );
        #endif

        this->mlp_fc1_weight[layer_idx].sync_to_device();
        this->hidden_states.sync_to_device();
        this->mlp_fc2_weight[layer_idx].sync_to_device();
        this->mlp_fc1_app(hidden_states, mlp_fc1_weight[layer_idx], mlp_intermediate_buffer);
        //mlp_intermediate_buffer.sync_from_device();

        this->mlp_fc2_app(mlp_intermediate_buffer, mlp_fc2_weight[layer_idx], hidden_states);
        hidden_states.sync_from_device();

        simd_add(
            hidden_states.data(),
            residual_buffer.data(),
            hidden_states.data(),
            (size_t)seq_len* padded_audio_hidden_size
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
                std::cout << "_final" << std::endl;
                buffer<bf16> ref__final;
                reference_tensor.load_weights(
                    ref__final,
                    "layer_" + std::to_string(layer_idx)+"_final"
                );

                print_error_metrics<bf16, bf16>(
                    hidden_states.data(), ref__final.data(),
                    1,
                    seq_len, desc.d_model,
                    seq_len_padded, padded_audio_hidden_size
                );
        #endif
    }

    // do the final norm

    layernorm_parallel(
        seq_len,
        desc.d_model,
        padded_audio_hidden_size,
        hidden_states.data(),
        this->merger_norm_weight.data(),
        this->merger_norm_bias.data(),
        desc.layer_norm_eps,
        hidden_states.data()
    );
    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "hidden_state_after_ln_post" << std::endl;
        buffer<bf16> ref_hidden_state_after_ln_post;
        reference_tensor.load_weights(
            ref_hidden_state_after_ln_post,
            "hidden_state_after_ln_post"
        );

        print_error_metrics<bf16, bf16>(
            hidden_states.data(), ref_hidden_state_after_ln_post.data(),
            1,
            seq_len, desc.d_model,
            seq_len_padded, padded_audio_hidden_size
        );
    #endif

    buffer<bf16> final_res = this->merger_fc2_app.create_bo_buffer<bf16>(
        (size_t)seq_len_padded* padded_audio_out_hidden_size
    );

    hidden_states.sync_to_device();
    this->merger_fc1_weight.sync_to_device();
    this->merger_fc2_weight.sync_to_device();
    merger_fc1_app(hidden_states, merger_fc1_weight, mlp_intermediate_buffer);
    merger_fc2_app(mlp_intermediate_buffer, merger_fc2_weight, final_res);
    final_res.sync_from_device();

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "hidden_states_after_proj2" << std::endl;
        buffer<bf16> ref_hidden_states_after_proj2;
        reference_tensor.load_weights(
            ref_hidden_states_after_proj2,
            "hidden_states_after_proj2"
        );

        print_error_metrics<bf16, bf16>(
            final_res.data(), ref_hidden_states_after_proj2.data(),
            1,
            seq_len, desc.output_dim,
            seq_len_padded, padded_audio_out_hidden_size
        );
    #endif

    return final_res;
}

void qwen3_5_omni_audio_embed::load_weights(SafeTensors &q4nx) {
    q4nx.load_weights(conv2d_1_weight, "model.audio.conv2d1.weight");
    q4nx.load_weights(conv2d_1_bias,   "model.audio.conv2d1.bias");
    q4nx.load_weights(conv2d_2_weight, "model.audio.conv2d2.weight");
    q4nx.load_weights(conv2d_2_bias,   "model.audio.conv2d2.bias");
    q4nx.load_weights(conv2d_3_weight, "model.audio.conv2d3.weight");
    q4nx.load_weights(conv2d_3_bias,   "model.audio.conv2d3.bias");
    q4nx.load_weights(conv2d_4_weight, "model.audio.conv2d4.weight");
    q4nx.load_weights(conv2d_4_bias,   "model.audio.conv2d4.bias");

    // Pre-transpose weights into [C_in, 9, C_out] layout for cache-friendly AXPY.
    conv_wt.prepare(
        conv2d_1_weight.data(), conv2d_1_bias.data(),
        conv2d_2_weight.data(), conv2d_2_bias.data(),
        conv2d_3_weight.data(), conv2d_3_bias.data(),
        conv2d_4_weight.data(), conv2d_4_bias.data(),
        desc.downsample_hidden_size
    );

    audio_conv_out_weight = this->conv_out_proj.create_bo_buffer<bf16>(
        (size_t)padded_audio_hidden_size* padded_audio_conv_out_k_size
    );
    q4nx.load_weights(audio_conv_out_weight, "model.audio.conv_out.weight", 0);

    q4nx.load_weights(
        merger_norm_weight,
        "model.audio.ln_post.weight"
    );
    q4nx.load_weights(
        merger_norm_bias,
        "model.audio.ln_post.bias"
    );

    this->merger_fc1_weight = this->merger_fc1_app.create_bo_buffer<bf16>(
        padded_audio_hidden_size*padded_audio_hidden_size+padded_audio_hidden_size
    );
    this->merger_fc2_weight = this->merger_fc2_app.create_bo_buffer<bf16>(
        padded_audio_hidden_size* padded_audio_out_hidden_size +padded_audio_out_hidden_size
    );

    q4nx.load_weights(
        this->merger_fc1_weight,
        "model.audio.tower.proj1.bias",0
    );
    q4nx.load_weights(
        this->merger_fc1_weight,
        "model.audio.tower.proj1.weight", sizeof(bf16)* padded_audio_hidden_size
    );
    q4nx.load_weights(
        this->merger_fc2_weight,
        "model.audio.tower.proj2.bias", 0
    );
    q4nx.load_weights(
        this->merger_fc2_weight,
        "model.audio.tower.proj2.weight", sizeof(bf16)* padded_audio_out_hidden_size
    );

    for(int layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){

        q4nx.load_weights(
            this->norm_1_weight[layer_idx],
            "model.audio.layers." + std::to_string(layer_idx) + ".self_attn_layer_norm.weight"
        );
        q4nx.load_weights(
            this->norm_1_bias[layer_idx],
            "model.audio.layers." + std::to_string(layer_idx) + ".self_attn_layer_norm.bias"
        );
        q4nx.load_weights(
            this->norm_2_weight[layer_idx],
            "model.audio.layers." + std::to_string(layer_idx) + ".final_layer_norm.weight"
        );
        q4nx.load_weights(
            this->norm_2_bias[layer_idx],
            "model.audio.layers." + std::to_string(layer_idx) + ".final_layer_norm.bias"
        );

        this->q_weight[layer_idx] = q_proj_app.create_bo_buffer<bf16>(padded_audio_hidden_size* padded_audio_hidden_size+padded_audio_hidden_size);
        this->k_weight[layer_idx] = k_proj_app.create_bo_buffer<bf16>(padded_audio_hidden_size* padded_audio_hidden_size+padded_audio_hidden_size);
        this->v_weight[layer_idx] = v_proj_app.create_bo_buffer<bf16>(padded_audio_hidden_size* padded_audio_hidden_size+padded_audio_hidden_size);
        this->o_weight[layer_idx] = o_proj_app.create_bo_buffer<bf16>(padded_audio_hidden_size* padded_audio_hidden_size+padded_audio_hidden_size);
        this->mlp_fc1_weight[layer_idx] = this->mlp_fc1_app.create_bo_buffer<bf16>(
            padded_audio_hidden_size* padded_audio_mlp_intermediate_size + padded_audio_mlp_intermediate_size
        );
        this->mlp_fc2_weight[layer_idx] = this->mlp_fc2_app.create_bo_buffer<bf16>(
            padded_audio_hidden_size* padded_audio_mlp_intermediate_size + padded_audio_hidden_size
        );

        q4nx.load_weights(this->q_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.q_proj.bias",
            0
        );
        q4nx.load_weights(this->k_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.k_proj.bias",
            0
        );
        q4nx.load_weights(this->v_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.v_proj.bias",
            0
        );
        q4nx.load_weights(this->o_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.out_proj.bias",
            0
        );
        q4nx.load_weights(this->mlp_fc1_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".fc1.bias",
            0
        );
        q4nx.load_weights(this->mlp_fc2_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".fc2.bias",
            0
        );

        q4nx.load_weights(this->q_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.q_proj.weight",
            padded_audio_hidden_size*sizeof(bf16)
        );
        q4nx.load_weights(this->k_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.k_proj.weight",
            padded_audio_hidden_size*sizeof(bf16)
        );
        q4nx.load_weights(this->v_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.v_proj.weight",
            padded_audio_hidden_size*sizeof(bf16)
        );
        q4nx.load_weights(this->o_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".self_attn.out_proj.weight",
            padded_audio_hidden_size*sizeof(bf16)
        );
        q4nx.load_weights(
            this->mlp_fc1_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".fc1.weight",
            padded_audio_mlp_intermediate_size*sizeof(bf16)
        );
        q4nx.load_weights(
            this->mlp_fc2_weight[layer_idx],
            "model.audio.layers."+std::to_string(layer_idx)+".fc2.weight",
            padded_audio_hidden_size*sizeof(bf16)
        );
    }
}
