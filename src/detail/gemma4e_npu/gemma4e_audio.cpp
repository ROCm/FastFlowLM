#include "flm_override.hpp"
#include "gemma4e_audio.hpp"

#include <thread>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif
#include "utils/debug_utils.hpp"
#include "utils/error_measure.hpp"

#include "gemma4e_vision_prefill_helper.hpp"

#include "vision/norm.hpp"
#include "mmRuntimeSequence.hpp"
#include "rot_pos_emb.hpp"

#include "gemma4e_audio_attention.hpp"
#include "conv1d_prefill.hpp"
#include "utils/utils.hpp"
#include <numbers>
// #define DEBUG_PRINT_ENCODE_ERROR_METRICS 1

Gemma4e_AudioEncoder::~Gemma4e_AudioEncoder() {}

Gemma4e_AudioEncoder::Gemma4e_AudioEncoder(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_npu* parent_npu_ptr)
    : config(config), npu(npu_instance), model_path(config.model_path), parent_npu_ptr(parent_npu_ptr)
{

    // load parameters from json file

    {
        MM_tile_M = config.sub("audio_config").value("Audio_MM_TILE_M", -1);
        MM_tile_K = config.sub("audio_config").value("Audio_MM_TILE_K", -1);
        MM_tile_N = config.sub("audio_config").value("Audio_MM_TILE_N", -1);

        seq_len_pad_requirement_for_MM = MM_ROW_SIZE*MM_tile_M;
        assert( MM_tile_K % MM_tile_N == 0);

        Gemma4E_Audio_residual_weight = config.sub("audio_config").value("Gemma4E_Audio_residual_weight", 0.0);
        assert(this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE % this->parent_npu_ptr->Gemma4E_Audio_num_attention_heads == 0);
        Gemma4E_Audio_attention_head_dim = this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE / this->parent_npu_ptr->Gemma4E_Audio_num_attention_heads;
    }

    Gemma4E_Audio_q_scale =    (1/std::sqrt(Gemma4E_Audio_attention_head_dim))  /  std::log(2);
    Gemma4E_Audio_k_scale = std::log(1 + std::numbers::e) / std::log(2);

    Gemma4E_Audio_padded_requirement_for_conv1d = this->parent_npu_ptr->Gemma4E_Audio_conv1d_kernel_size\
         - this->parent_npu_ptr->Gemma4E_Audio_conv1d_stride;
    DEBUG_BLOCK(1,
    std::cout << "Audio q scale: " << Gemma4E_Audio_q_scale << ", k_scale: " << Gemma4E_Audio_k_scale << std::endl;
    std::cout << "Gemma4E_Audio_padded_requirement_for_conv1d : " << Gemma4E_Audio_padded_requirement_for_conv1d << std::endl;
    )

    Padded_GEMMA4E_Audio_HIDDEN_SIZE = round_up_to_multiple(this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE, MM_tile_K);
    Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE = round_up_to_multiple(this->parent_npu_ptr->Gemma4E_Audio_INTERMEDIATE_SIZE, MM_tile_K);
    Padded_Gemma4E_Audio_Multimodal_Output_SIZE = round_up_to_multiple(this->parent_npu_ptr->Gemma4E_Audio_Multimodal_Output_SIZE, MM_tile_K);
    Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE = round_up_to_multiple(
        this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE*2, MM_tile_K
    );
    assert(Padded_GEMMA4E_Audio_HIDDEN_SIZE %Gemma4E_Audio_attention_head_dim == 0);
    Padded_Gemma4E_Audio_num_attention_heads = Padded_GEMMA4E_Audio_HIDDEN_SIZE / Gemma4E_Audio_attention_head_dim;
    assert(Padded_Gemma4E_Audio_num_attention_heads >= this->parent_npu_ptr->Gemma4E_Audio_num_attention_heads);

    this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));
    this->proj_high_precision = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm_high_precision.xclbin"));
    this->conv1d = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "audio_conv1d.xclbin"));

    this->sub_sampleConvProjection_app = this->proj_high_precision->create_app();
    this->q_proj_app = this->proj->create_app();
    this->k_proj_app = this->proj->create_app();
    this->k_relative_proj_app = this->proj->create_app();
    this->v_proj_app = this->proj->create_app();
    this->o_proj_app = this->proj->create_app();
    this->ffn_down_proj_app = this->proj->create_app();
    this->ffn_up_proj_app = this->proj->create_app();
    this->conv1d_start_proj_app = this->proj->create_app();
    this->conv1d_end_proj_app = this->proj->create_app();
    this->conv1d_app = this->conv1d->create_app();
    audio_pre_encode_proj_app = this->proj->create_app();
    audio_to_language_proj_app = this->proj->create_app();

    this->audio_attn_k_rel_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_attn_k_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_attn_v_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_attn_q_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_attn_o_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_conv1d_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_conv_pw_1_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_conv_pw_2_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_ffn_down_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_ffn_up_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_ffn_down_1_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->audio_ffn_up_1_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->attn_post_norm_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->attn_pre_norm_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->conv_norm_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->norm_conv_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->ffn_norm_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->ffn_norm_1_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->ffn_post_norm_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->ffn_post_norm_1_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->norm2_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
    this->per_dim_scale_with_softplus_weight.resize(this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers);
}

void Gemma4e_AudioEncoder::conv1d_layer(

    int layer_idx,
    int seq_len, int seq_len_padded,
    std::vector<int> &seq_len_per_audio, std::vector<int> &start_seq_len_index_per_audio,
    buffer<bf16> &conv1d_start_proj_input, buffer<bf16> &conv1d_start_proj_output,
    buffer<bf16> &conv1d_input, buffer<bf16> &conv1d_output,
    buffer<bf16> &conv1d_end_proj_input, buffer<bf16> &conv1d_end_proj_output,

    bf16 conv1d_start_input_min, bf16 conv1d_start_input_max,
    bf16 conv1d_start_output_min, bf16 conv1d_start_output_max,

    bf16 conv1d_end_input_min, bf16 conv1d_end_input_max,
    bf16 conv1d_end_output_min, bf16 conv1d_end_output_max,

    gemma4e_audio_payload_t* audio_payload,
    SafeTensors *reference_safetensor
){

    memcpy(residual.data(), hidden_state.data(), seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));

    // compare the norm weigths
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "Comparing conv1d_layer " << layer_idx << " norm weights..." << std::endl;
        buffer<bf16> ref_per_layer_norm_weight;
        reference_safetensor->load_weights(
            ref_per_layer_norm_weight,
            "Gemma4AudioLightConv1d_"+std::to_string(layer_idx)+ "_pre_layer_norm_weight"
        );
        print_error_metrics<bf16, bf16>(
            this->conv_norm_weight[layer_idx].data(), ref_per_layer_norm_weight.data(),
            1,
            ref_per_layer_norm_weight.size(), 1,
            ref_per_layer_norm_weight.size(), 1
        );

        std::cout << "Comparing conv1d_layer " << layer_idx << " conv norm weights..." << std::endl;
        buffer<bf16> ref_conv_norm_weight;
        reference_safetensor->load_weights(
            ref_conv_norm_weight,
            "Gemma4AudioLightConv1d_"+std::to_string(layer_idx)+ "_conv_norm_weight"
        );
        print_error_metrics<bf16, bf16>(
            this->norm_conv_weight[layer_idx].data(), ref_conv_norm_weight.data(),
            1,
            ref_conv_norm_weight.size(), 1,
            ref_conv_norm_weight.size(), 1
        );
    }
    #endif

    simd_rms_norm(
        hidden_state.data(), this->conv_norm_weight[layer_idx].data(), conv1d_start_proj_input.data(),
        seq_len, this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE, seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, 1e-6f
    );

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "Comparing conv1d_layer " << layer_idx << " pre-attention norm output..." << std::endl;
        buffer<bf16> ref_Gemma4AudioLightConv1d_layer_idx_hidden_states_after_pre_layer_norm;
        reference_safetensor->load_weights(
            ref_Gemma4AudioLightConv1d_layer_idx_hidden_states_after_pre_layer_norm,
            "Gemma4AudioLightConv1d_"+std::to_string(layer_idx)+ "_hidden_states_after_pre_layer_norm"
        );
        size_t ref_offset_per_audio = ref_Gemma4AudioLightConv1d_layer_idx_hidden_states_after_pre_layer_norm.size() / audio_payload->num_audios;
        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                conv1d_start_proj_input.data() + start_seq_len_index_per_audio[i]*Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                ref_Gemma4AudioLightConv1d_layer_idx_hidden_states_after_pre_layer_norm.data() + i*ref_offset_per_audio,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
        }
    }
    #endif

    simd_clamp(
        conv1d_start_proj_input.data(), conv1d_start_proj_input.data(),
        conv1d_start_input_min, conv1d_start_input_max,
        seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    {

        generate_mm_sequence<bf16, bf16>(
            *this->conv1d_start_proj_app.seq(),
            seq_len_padded, this->Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            false, 0, //no bias, no activation
            1,  conv1d_start_output_min, conv1d_start_output_max, // enable clamp in output
            ENABLE_QKV_REORDER, 0// since we don't need it anymore

        );
    }
    DEBUG_BLOCK(1,
    std::cout << "DEBUG: Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE: " << Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE << std::endl;
    )
    conv1d_start_proj_input.sync_to_device();
    audio_conv_pw_1_weight[layer_idx].sync_to_device();
    FLM_OVERRIDE(audio_conv1d_start_proj, conv1d_start_proj_app(conv1d_start_proj_input, audio_conv_pw_1_weight[layer_idx], conv1d_start_proj_output), layer_idx);
    conv1d_start_proj_output.sync_from_device();
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "Comparing conv1d_layer " << layer_idx << " conv1d linear output..." << std::endl;
        buffer<bf16> ref_conv1d_start_proj_output;
        reference_safetensor->load_weights(
            ref_conv1d_start_proj_output,
            "Gemma4AudioLightConv1d_" + std::to_string(layer_idx) +"_hidden_states_after_linear_start"
        );
        size_t ref_offset_per_audio = ref_conv1d_start_proj_output.size() / audio_payload->num_audios;
        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                conv1d_start_proj_output.data() + start_seq_len_index_per_audio[i]*Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE,
                ref_conv1d_start_proj_output.data() + i*ref_offset_per_audio,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE * 2,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE
            );
        }
    }
    #endif

    assert(Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE == (Padded_GEMMA4E_Audio_HIDDEN_SIZE*2) );

    for(int i = 0, seq_len_offset = 0; i < audio_payload->num_audios; i++){

        seq_len_offset += Gemma4E_Audio_padded_requirement_for_conv1d;
        simd_glu(
            conv1d_start_proj_output.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE,
            conv1d_input.data() + (seq_len_offset * Padded_GEMMA4E_Audio_HIDDEN_SIZE),
            seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );

        seq_len_offset += (seq_len_per_audio[i] );
    }
    conv1d_start_proj_output.sync_to_device();

    // compare with error metrics

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "Comparing conv1d_layer " << layer_idx << " conv1d GLU output..." << std::endl;
        buffer<bf16> ref_conv1d_glu_output;
        reference_safetensor->load_weights(
            ref_conv1d_glu_output,
            "Gemma4AudioLightConv1d_" + std::to_string(layer_idx) +"_hidden_states_after_glu"
        );

        size_t ref_offset_per_audio = ref_conv1d_glu_output.size() / audio_payload->num_audios;

        for(int i = 0, seq_len_offset = 0; i < audio_payload->num_audios; i++){
           seq_len_offset += Gemma4E_Audio_padded_requirement_for_conv1d;

            print_error_metrics<bf16, bf16>(
                conv1d_input.data() + seq_len_offset*Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                ref_conv1d_glu_output.data() + i*ref_offset_per_audio,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );

            seq_len_offset += seq_len_per_audio[i];
        }
    }
    #endif

    // conv1d
    for(int i = 0, seq_len_offset = 0; i < audio_payload->num_audios; i++){

        conv1d_prefill(
            this->conv1d_app.seq(),
            seq_len_per_audio[i],
            -1.18e30f, 3.38e30f, // no clamping,
            seq_len_offset * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            start_seq_len_index_per_audio[i] *Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );

        // scalar_conv1d(
        //     this->parent_npu_ptr->Gemma4E_Audio_conv1d_kernel_size,
        //     this->parent_npu_ptr->Gemma4E_Audio_conv1d_stride,
        //     conv1d_input.data() + seq_len_offset * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        //     audio_conv1d_weight[layer_idx].data(),
        //     conv1d_output.data() + start_seq_len_index_per_audio[i] *Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        //     seq_len_per_audio[i],
        //     Padded_GEMMA4E_Audio_HIDDEN_SIZE
        // );

        seq_len_offset += (Gemma4E_Audio_padded_requirement_for_conv1d + seq_len_per_audio[i]);

        conv1d_input.sync_to_device();
        audio_conv1d_weight[layer_idx].sync_to_device();
        FLM_OVERRIDE(audio_conv1d, conv1d_app(conv1d_output, conv1d_input,audio_conv1d_weight[layer_idx]   ), layer_idx);
        conv1d_output.sync_from_device();
    }

    // utils::print_matrix<bf16>(
    //     audio_conv1d_weight[layer_idx], Padded_GEMMA4E_Audio_HIDDEN_SIZE

    // );

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "Comparing conv1d_layer " << layer_idx << " conv1d output..." << std::endl;
        buffer<bf16> ref_conv1d_output;
        reference_safetensor->load_weights(
            ref_conv1d_output,
            "Gemma4AudioLightConv1d_" + std::to_string(layer_idx) +"_hidden_states_after_depthwise_conv1d"
        );

        size_t ref_offset_per_audio = ref_conv1d_output.size() / audio_payload->num_audios;
        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                conv1d_output.data() + start_seq_len_index_per_audio[i]*Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                ref_conv1d_output.data() + i*ref_offset_per_audio,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
        }
    }
    #endif

    simd_rms_norm(
        conv1d_output.data(), this->norm_conv_weight[layer_idx].data(), conv1d_output.data(),
        seq_len, this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE, seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, 1e-6f
    );

    simd_silu(
        conv1d_output.data(), conv1d_end_proj_input.data(),
        seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    conv1d_output.sync_to_device();
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> ref_hidden_states_after_conv_act;
        reference_safetensor->load_weights(
            ref_hidden_states_after_conv_act,
            "Gemma4AudioLightConv1d_" + std::to_string(layer_idx) +"_hidden_states_after_conv_act"
        );

        size_t ref_offset_per_audio = ref_hidden_states_after_conv_act.size() / audio_payload->num_audios;

        for(int i = 0; i < audio_payload->num_audios; i++){
            std::cout << "Comparing conv1d_layer " << layer_idx << " conv1d activation output for audio " << i << std::endl;
            print_error_metrics<bf16, bf16>(
                conv1d_end_proj_input.data() + start_seq_len_index_per_audio[i]*Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                ref_hidden_states_after_conv_act.data() + i*ref_offset_per_audio,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
        }
    }
    #endif

    simd_clamp(
        conv1d_end_proj_input.data(), conv1d_end_proj_input.data(),
        conv1d_end_input_min, conv1d_end_input_max,
        seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );

    {
        generate_mm_sequence<bf16, bf16>(
            *this->conv1d_end_proj_app.seq(),
            seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            false, 0, //no bias, no activation
            1,  conv1d_end_output_min,conv1d_end_output_max, // enable clamp in output
            ENABLE_QKV_REORDER, 0// since we don't need it anymore

        );
    }

    conv1d_end_proj_input.sync_to_device();
    audio_conv_pw_2_weight[layer_idx].sync_to_device();
    FLM_OVERRIDE(audio_conv1d_end_proj, conv1d_end_proj_app(conv1d_end_proj_input, audio_conv_pw_2_weight[layer_idx], conv1d_end_proj_output), layer_idx);
    conv1d_end_proj_output.sync_from_device();

    simd_add(conv1d_end_proj_output.data(), residual.data(), hidden_state.data(),
    seq_len*Padded_GEMMA4E_Audio_HIDDEN_SIZE);

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> ref_conv1d_final_output;
        reference_safetensor->load_weights(
            ref_conv1d_final_output,
            "Gemma4AudioLightConv1d_" + std::to_string(layer_idx) +"_hidden_states_after_residual"
        );

        size_t ref_offset_per_audio = ref_conv1d_final_output.size() / audio_payload->num_audios;
        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                hidden_state.data() + start_seq_len_index_per_audio[i]*Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                ref_conv1d_final_output.data() + i*ref_offset_per_audio,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
        }
    }
    #endif
}

void Gemma4e_AudioEncoder::ffn_layer(

    buffer<bf16> &ffn_up_proj_input,
    buffer<bf16> &ffn_up_proj_output_down_input,
    buffer<bf16> &ffn_down_proj_output,

    buffer<bf16> &cur_ffn_norm_weight,  // ffn_norm or ffn_norm_1
    buffer<bf16> &cur_ffn_post_norm_weight,// ffn_post_norm_weight or ffn_post_norm_1_weight
    buffer<bf16> &cur_ffn_up_weight, // audio_ffn_up_weight or audio_ffn_up_1_weight
    buffer<bf16> &cur_ffn_down_weight, // audio_ffn_down_weight
    int seq_len, int seq_len_padded,

    bf16 cur_audio_ffn_up_input_min, bf16 cur_audio_ffn_up_input_max,
    bf16 cur_audio_ffn_up_output_min, bf16 cur_audio_ffn_up_output_max,
    bf16 cur_audio_ffn_down_input_min, bf16 cur_audio_ffn_down_input_max,
    bf16 cur_audio_ffn_down_output_min, bf16 cur_audio_ffn_down_output_max

){

    memcpy(residual.data(), hidden_state.data(), seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));

    generate_mm_sequence<bf16, bf16>(
        *this->ffn_up_proj_app.seq(),
        seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE,
        MM_tile_M, MM_tile_K, MM_tile_N,
        8,8,8,
        rtp_address, rtp_sync_lock_id,
        MM_ROW_SIZE,MM_COL_SIZE,
        0,0,0,
        IS_B_ROW_MAJOR, ENABLE_AXI4, true,
        false, 2, //no bias, silu activation
        1, cur_audio_ffn_up_output_min,cur_audio_ffn_up_output_max, // with clamp
        ENABLE_QKV_REORDER, 0// since we don't need it anymore

    );

    simd_rms_norm(
        hidden_state.data(),
        cur_ffn_norm_weight.data(),
        ffn_up_proj_input.data(),
        seq_len,
        this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
        seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        1e-6f
    );
    simd_clamp(ffn_up_proj_input.data(),
        ffn_up_proj_input.data(),
        cur_audio_ffn_up_input_min, cur_audio_ffn_up_input_max,
        seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );

    //ffn_up_proj_app(ffn_up_proj_input, cur_ffn_up_weight, ffn_up_proj_output_down_input);
    auto ffn_up_proj_run = FLM_OVERRIDE(audio_ffn_up_proj, ffn_up_proj_app.create_run(
        ffn_up_proj_input,cur_ffn_up_weight, ffn_up_proj_output_down_input
    ));
    ffn_up_proj_input.sync_to_device();
    cur_ffn_up_weight.sync_to_device();
    ffn_up_proj_run.start();

        generate_mm_sequence<bf16, bf16>(
            *this->ffn_down_proj_app.seq(),
            seq_len_padded, Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            false, 0, //no bias, no activation
            1, cur_audio_ffn_down_output_min,cur_audio_ffn_down_output_max, // with clamp
            ENABLE_QKV_REORDER, 0// since we don't need it anymore

        );

    ffn_up_proj_run.wait();
    ffn_up_proj_output_down_input.sync_from_device();

    simd_clamp(
        ffn_up_proj_output_down_input.data(),
        ffn_up_proj_output_down_input.data(),
        cur_audio_ffn_down_input_min, cur_audio_ffn_down_input_max,
        seq_len * Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE
    );

    ffn_up_proj_output_down_input.sync_to_device();
    cur_ffn_down_weight.sync_to_device();
    FLM_OVERRIDE(audio_ffn_down_proj, ffn_down_proj_app(ffn_up_proj_output_down_input,cur_ffn_down_weight, ffn_down_proj_output ));
    ffn_down_proj_output.sync_from_device();

    // perform a post_layer_nrom
    simd_rms_norm(
        ffn_down_proj_output.data(),
        cur_ffn_post_norm_weight.data(),
        ffn_down_proj_output.data(),
        seq_len,
        this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
        seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        1e-6f
    );

    simd_add(
        ffn_down_proj_output.data(), residual.data(), hidden_state.data(),
        seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    ffn_down_proj_output.sync_to_device();
}

void Gemma4e_AudioEncoder::init_weights(SafeTensors &q4nx){
    DEBUG_BLOCK(1,
    std::cout << "Initializing Gemma4e_AudioEncoder weights from model path: " << model_path << std::endl;
    )

    q4nx.load_weights(this->audio_subsample_conv2d_weight_0,"model.audio.subsample.conv_layer0.weight");
    q4nx.load_weights(this->audio_subsample_conv2d_norm_weight_0,"model.audio.subsample.conv_layer0.norm.weight");
    q4nx.load_weights(this->audio_subsample_conv2d_weight_1,"model.audio.subsample.conv_layer1.weight");
    q4nx.load_weights(this->audio_subsample_conv2d_norm_weight_1,"model.audio.subsample.conv_layer1.norm.weight");
    {
        buffer<bf16> temp_buffer;
        q4nx.load_weights(temp_buffer, "model.audio.encode_input_projection.weight");
        this->audio_embedding_projection_weight = this->sub_sampleConvProjection_app.create_bo_buffer<bf16>(temp_buffer.size());
        memcpy(
            this->audio_embedding_projection_weight.data(),
            temp_buffer.data(),
            temp_buffer.size() * sizeof(bf16)
        );
    }

    audio_pre_encode_weight = this->audio_pre_encode_proj_app.create_bo_buffer<bf16>(
        Padded_GEMMA4E_Audio_HIDDEN_SIZE * Padded_Gemma4E_Audio_Multimodal_Output_SIZE + Padded_Gemma4E_Audio_Multimodal_Output_SIZE

    );
    DEBUG_BLOCK(1,
    std::cout <<"Padded_Gemma4E_Audio_Multimodal_Output_SIZE: " << Padded_Gemma4E_Audio_Multimodal_Output_SIZE << std::endl;
    )
    q4nx.load_weights(
        this->audio_pre_encode_weight,
       "model.audio.pre_encoder.bias", 0 // the bias
    );
    q4nx.load_weights(
        this->audio_pre_encode_weight,
       "model.audio.pre_encoder.weight",
        sizeof(bf16) * Padded_Gemma4E_Audio_Multimodal_Output_SIZE // the weight
    );

    audio_to_language_projection_weight = this->audio_to_language_proj_app.create_bo_buffer<bf16>(
        Padded_Gemma4E_Audio_Multimodal_Output_SIZE * parent_npu_ptr->Gemma4E_Audio_language_projection_output_size
    );
    assert(
        parent_npu_ptr->Gemma4E_Audio_language_projection_output_size% MM_tile_N == 0
    );
    q4nx.load_weights(
        audio_to_language_projection_weight,
        "model.audio.embedding_projection.weight"
    );

    for(int layer_id =0; layer_id < this->parent_npu_ptr->Gemma4E_Audio_num_attention_layers; layer_id ++){

        this->audio_attn_k_weight[layer_id] = this->k_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE*Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_attn_k_weight[layer_id],
            "model.audio." + std::to_string(layer_id)+ ".attn_k_proj.weight"
        );

        buffer<bf16> k_input_max;
        q4nx.load_weights(k_input_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_k_proj.input_max");
        assert(k_input_max.size() == 1);
        this->audio_k_input_max.push_back(k_input_max[0]);

        buffer<bf16> k_input_min;
        q4nx.load_weights(k_input_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_k_proj.input_min");
        assert(k_input_min.size() == 1);
        this->audio_k_input_min.push_back(k_input_min[0]);

        buffer<bf16> k_output_max;
        q4nx.load_weights(k_output_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_k_proj.output_max");
        assert(k_output_max.size() == 1);
        this->audio_k_output_max.push_back(k_output_max[0]);

        buffer<bf16> k_output_min;
        q4nx.load_weights(k_output_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_k_proj.output_min");
        assert(k_output_min.size() == 1);
        this->audio_k_output_min.push_back(k_output_min[0]);

        this->audio_attn_k_rel_weight[layer_id] = this->k_relative_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE*Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_attn_k_rel_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".attn_k_proj.rel_weight"
        );

        this->audio_attn_o_weight[layer_id] = this->o_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE*Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_attn_o_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".attn_out_proj.weight"
        );

        buffer<bf16> o_input_max;
        q4nx.load_weights(o_input_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_out_proj.input_max");
        assert(o_input_max.size() == 1);

        buffer<bf16> o_input_min;
        q4nx.load_weights(o_input_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_out_proj.input_min");
        assert(o_input_min.size() == 1);
        this->audio_o_input_min.push_back(o_input_min[0]);

        this->audio_o_input_max.push_back(o_input_max[0]);
        buffer<bf16> o_output_max;
        q4nx.load_weights(o_output_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_out_proj.output_max");
        assert(o_output_max.size() == 1);

        this->audio_o_output_max.push_back(o_output_max[0]);
        buffer<bf16> o_output_min;
        q4nx.load_weights(o_output_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_out_proj.output_min");
        assert(o_output_min.size() == 1);
        this->audio_o_output_min.push_back(o_output_min[0]);

        q4nx.load_weights(
            this->attn_post_norm_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".attn_post_norm.weight"
        );
        q4nx.load_weights(
            this->attn_pre_norm_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".attn_pre_norm.weight"
        );

        this->audio_attn_q_weight[layer_id] = this->q_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE*Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_attn_q_weight[layer_id],
            "model.audio."+ std::to_string(layer_id)+".attn_q_proj.weight"
        );
        buffer<bf16> q_input_max;
        q4nx.load_weights(q_input_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_q_proj.input_max");
        assert(q_input_max.size() == 1);
        this->audio_q_input_max.push_back(q_input_max[0]);

        buffer<bf16> q_input_min;
        q4nx.load_weights(q_input_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_q_proj.input_min");
        assert(q_input_min.size() == 1);
        this->audio_q_input_min.push_back(q_input_min[0]);

        buffer<bf16> q_output_max;
        q4nx.load_weights(q_output_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_q_proj.output_max");
        assert(q_output_max.size() == 1);
        this->audio_q_output_max.push_back(q_output_max[0]);

        buffer<bf16> q_output_min;
        q4nx.load_weights(q_output_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_q_proj.output_min");
        assert(q_output_min.size() == 1);
        this->audio_q_output_min.push_back(q_output_min[0]);

        this->audio_attn_v_weight[layer_id] = this->v_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE*Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_attn_v_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".attn_v_proj.weight");
        buffer<bf16> v_input_max;
        q4nx.load_weights(v_input_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_v_proj.input_max");
        assert(v_input_max.size() == 1);
        this->audio_v_input_max.push_back(v_input_max[0]);

        buffer<bf16> v_input_min;
        q4nx.load_weights(v_input_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_v_proj.input_min");
        assert(v_input_min.size() == 1);
        this->audio_v_input_min.push_back(v_input_min[0]);

        buffer<bf16> v_output_max;
        q4nx.load_weights(v_output_max,
            "model.audio."+std::to_string(layer_id)+ ".attn_v_proj.output_max");
        assert(v_output_max.size() == 1);
        this->audio_v_output_max.push_back(v_output_max[0]);

        buffer<bf16> v_output_min;
        q4nx.load_weights(v_output_min,
            "model.audio."+std::to_string(layer_id)+ ".attn_v_proj.output_min");
        assert(v_output_min.size() == 1);
        this->audio_v_output_min.push_back(v_output_min[0]);

        this->audio_conv1d_weight[layer_id] = this->conv1d_app.create_bo_buffer<bf16>(
            this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE * this->parent_npu_ptr->Gemma4E_Audio_conv1d_kernel_size
        );
        q4nx.load_weights(
            this->audio_conv1d_weight[layer_id],
            "model.audio." +std::to_string(layer_id) + ".conv_dw.weight"
        );

        q4nx.load_weights(
            this->conv_norm_weight[layer_id],
            "model.audio."+std::to_string(layer_id) +".conv_norm.weight"
        );
        q4nx.load_weights(
            this->norm_conv_weight[layer_id],
            "model.audio."+std::to_string(layer_id) +".norm_conv.weight"
        );

        //
        this->audio_conv_pw_1_weight[layer_id] = this->conv1d_start_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE * Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE
        );
        q4nx.load_weights(
            this->audio_conv_pw_1_weight[layer_id],
            "model.audio." +std::to_string(layer_id) + ".conv_pw_1.weight"
        );
        buffer<bf16> conv_pw_1_input_max;
        q4nx.load_weights(conv_pw_1_input_max,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_1.input_max");
        assert(conv_pw_1_input_max.size() == 1);
        this->audio_conv_pw1_input_max.push_back(conv_pw_1_input_max[0]);

        buffer<bf16> conv_pw_1_input_min;
        q4nx.load_weights(conv_pw_1_input_min,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_1.input_min");
        assert(conv_pw_1_input_min.size() == 1);
        this->audio_conv_pw1_input_min.push_back(conv_pw_1_input_min[0]);

        buffer<bf16> conv_pw_1_output_max;
        q4nx.load_weights(conv_pw_1_output_max,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_1.output_max");
        assert(conv_pw_1_output_max.size() == 1);
        this->audio_conv_pw1_output_max.push_back(conv_pw_1_output_max[0]);

        buffer<bf16> conv_pw_1_output_min;
        q4nx.load_weights(conv_pw_1_output_min,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_1.output_min");
        assert(conv_pw_1_output_min.size() == 1);
        this->audio_conv_pw1_output_min.push_back(conv_pw_1_output_min[0]);

        this->audio_conv_pw_2_weight[layer_id] = this->conv1d_end_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE * Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_conv_pw_2_weight[layer_id],
            "model.audio." +std::to_string(layer_id) + ".conv_pw_2.weight"
        );
        buffer<bf16> conv_pw_2_input_max;
        q4nx.load_weights(conv_pw_2_input_max,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_2.input_max");
        assert(conv_pw_2_input_max.size() == 1);
        this->audio_conv_pw2_input_max.push_back(conv_pw_2_input_max[0]);

        buffer<bf16> conv_pw_2_input_min;
        q4nx.load_weights(conv_pw_2_input_min,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_2.input_min");
        assert(conv_pw_2_input_min.size() == 1);
        this->audio_conv_pw2_input_min.push_back(conv_pw_2_input_min[0]);

        buffer<bf16> conv_pw_2_output_max;
        q4nx.load_weights(conv_pw_2_output_max,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_2.output_max");
        assert(conv_pw_2_output_max.size() == 1);
        this->audio_conv_pw2_output_max.push_back(conv_pw_2_output_max[0]);

        buffer<bf16> conv_pw_2_output_min;
        q4nx.load_weights(conv_pw_2_output_min,
            "model.audio."+std::to_string(layer_id)+ ".conv_pw_2.output_min");
        assert(conv_pw_2_output_min.size() == 1);
        this->audio_conv_pw2_output_min.push_back(conv_pw_2_output_min[0]);

        this->audio_ffn_down_weight[layer_id] = this->ffn_down_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE * Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE
        );
        q4nx.load_weights(
            this->audio_ffn_down_weight[layer_id],
            "model.audio." +std::to_string(layer_id) + ".ffn.down_proj.weight"
        );
        buffer<bf16> ffn_down_input_max;
        q4nx.load_weights(ffn_down_input_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj.input_max");
        assert(ffn_down_input_max.size() == 1);
        this->audio_ffn_down_input_max.push_back(ffn_down_input_max[0]);

        buffer<bf16> ffn_down_input_min;
        q4nx.load_weights(ffn_down_input_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj.input_min");
        assert(ffn_down_input_min.size() == 1);
        this->audio_ffn_down_input_min.push_back(ffn_down_input_min[0]);

        buffer<bf16> ffn_down_output_max;
        q4nx.load_weights(ffn_down_output_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj.output_max");
        assert(ffn_down_output_max.size() == 1);
        this->audio_ffn_down_output_max.push_back(ffn_down_output_max[0]);

        buffer<bf16> ffn_down_output_min;
        q4nx.load_weights(ffn_down_output_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj.output_min");
        assert(ffn_down_output_min.size() == 1);
        this->audio_ffn_down_output_min.push_back(ffn_down_output_min[0]);

        this->audio_ffn_down_1_weight[layer_id] = this->ffn_down_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_HIDDEN_SIZE * Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE
        );
        q4nx.load_weights(
            this->audio_ffn_down_1_weight[layer_id],
            "model.audio." +std::to_string(layer_id) + ".ffn.down_proj_1.weight"
        );
        buffer<bf16> ffn_down_1_input_max;
        q4nx.load_weights(ffn_down_1_input_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj_1.input_max");
        assert(ffn_down_1_input_max.size() == 1);
        this->audio_ffn_down_1_input_max.push_back(ffn_down_1_input_max[0]);

        buffer<bf16> ffn_down_1_input_min;
        q4nx.load_weights(ffn_down_1_input_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj_1.input_min");
        assert(ffn_down_1_input_min.size() == 1);
        this->audio_ffn_down_1_input_min.push_back(ffn_down_1_input_min[0]);

        buffer<bf16> ffn_down_1_output_max;
        q4nx.load_weights(ffn_down_1_output_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj_1.output_max");
        assert(ffn_down_1_output_max.size() == 1);
        this->audio_ffn_down_1_output_max.push_back(ffn_down_1_output_max[0]);

        buffer<bf16> ffn_down_1_output_min;
        q4nx.load_weights(ffn_down_1_output_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.down_proj_1.output_min");
        assert(ffn_down_1_output_min.size() == 1);
        this->audio_ffn_down_1_output_min.push_back(ffn_down_1_output_min[0]);

        q4nx.load_weights(
            this->ffn_norm_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ffn_norm.weight"
        );
        q4nx.load_weights(
            this->ffn_norm_1_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ffn_norm_1.weight"
        );
        q4nx.load_weights(
            this->ffn_post_norm_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ffn_post_norm.weight"
        );
        //NOTE: Optimization in multiple ffn_post_norm_weight with this.Gemma4E_Audio_residual_weight
        for(int i = 0; i < this->ffn_post_norm_weight[layer_id].size(); i++){
            this->ffn_post_norm_weight[layer_id][i]  = this->ffn_post_norm_weight[layer_id][i] *  this->Gemma4E_Audio_residual_weight;
        }

        q4nx.load_weights(
            this->ffn_post_norm_1_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ffn_post_norm_1.weight"
        );
        //NOTE: Optimization in multiple ffn_post_norm_1_weight with this.Gemma4E_Audio_residual_weight
        for(int i = 0; i < this->ffn_post_norm_1_weight[layer_id].size(); i++){
            this->ffn_post_norm_1_weight[layer_id][i]  = this->ffn_post_norm_1_weight[layer_id][i] *  this->Gemma4E_Audio_residual_weight;
        }

        this->audio_ffn_up_weight[layer_id] = this->ffn_up_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE * Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_ffn_up_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ffn.up_proj.weight"
        );

        buffer<bf16> ffn_up_input_max;
        q4nx.load_weights(ffn_up_input_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj.input_max");
        assert(ffn_up_input_max.size() == 1);
        this->audio_ffn_up_input_max.push_back(ffn_up_input_max[0]);

        buffer<bf16> ffn_up_input_min;
        q4nx.load_weights(ffn_up_input_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj.input_min");
        assert(ffn_up_input_min.size() == 1);
        this->audio_ffn_up_input_min.push_back(ffn_up_input_min[0]);

        buffer<bf16> ffn_up_output_max;
        q4nx.load_weights(ffn_up_output_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj.output_max");
        assert(ffn_up_output_max.size() == 1);
        this->audio_ffn_up_output_max.push_back(ffn_up_output_max[0]);

        buffer<bf16> ffn_up_output_min;
        q4nx.load_weights(ffn_up_output_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj.output_min");
        assert(ffn_up_output_min.size() == 1);
        this->audio_ffn_up_output_min.push_back(ffn_up_output_min[0]);

        this->audio_ffn_up_1_weight[layer_id] = this->ffn_up_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE * Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        q4nx.load_weights(
            this->audio_ffn_up_1_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ffn.up_proj_1.weight"
        );
        buffer<bf16> ffn_up_1_input_max;
        q4nx.load_weights(ffn_up_1_input_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj_1.input_max");
        assert(ffn_up_1_input_max.size() == 1);
        this->audio_ffn_up_1_input_max.push_back(ffn_up_1_input_max[0]);

        buffer<bf16> ffn_up_1_input_min;
        q4nx.load_weights(ffn_up_1_input_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj_1.input_min");
        assert(ffn_up_1_input_min.size() == 1);
        this->audio_ffn_up_1_input_min.push_back(ffn_up_1_input_min[0]);

        buffer<bf16> ffn_up_1_output_max;
        q4nx.load_weights(ffn_up_1_output_max,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj_1.output_max");
        assert(ffn_up_1_output_max.size() == 1);
        this->audio_ffn_up_1_output_max.push_back(ffn_up_1_output_max[0]);

        buffer<bf16> ffn_up_1_output_min;
        q4nx.load_weights(ffn_up_1_output_min,
            "model.audio."+std::to_string(layer_id)+ ".ffn.up_proj_1.output_min");
        assert(ffn_up_1_output_min.size() == 1);
        this->audio_ffn_up_1_output_min.push_back(ffn_up_1_output_min[0]);

        q4nx.load_weights(
            this->norm2_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".ln2.weight"
        );

        q4nx.load_weights(
            this->per_dim_scale_with_softplus_weight[layer_id],
            "model.audio."+std::to_string(layer_id)+".pre_dim_scale.weight"
        );
        // Any optimization, multiple per_dim_scale with self.q_scale at load weights
        for(int i = 0; i < this->per_dim_scale_with_softplus_weight[layer_id].size(); i++){
            this->per_dim_scale_with_softplus_weight[layer_id][i] = this->per_dim_scale_with_softplus_weight[layer_id][i] * Gemma4E_Audio_q_scale;
        }
    }
}

std::vector<bf16> Gemma4e_AudioEncoder::encode(void* audio_payload_ptr){

    DEBUG_BLOCK(1,
    std::cout << "Gemma4e_AudioEncoder::encode called with audio_payload_ptr: " << audio_payload_ptr << std::endl;
    )

    //DEBUG

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    SafeTensors reference_tensors(
        this->model_path + "/audio_reference_data.safetensors"
    );

    #endif

    gemma4e_audio_payload_t* audio_payload = static_cast<gemma4e_audio_payload_t*>(audio_payload_ptr);
    assert(audio_payload != nullptr);

    /*
    // compare the weights
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "DEBUG: Start computing error metrics for audio encoder weights" << std::endl;
        buffer<bf16> reference_conv_weight_0;
        buffer<bf16> reference_conv_weight_1;
        reference_tensors.load_weights(reference_conv_weight_0,"2D_convolution_weights_0");
        reference_tensors.load_weights(reference_conv_weight_1,"2D_convolution_weights_1");

        assert(audio_subsample_conv2d_weight_0.size() == reference_conv_weight_0.size());
        assert(audio_subsample_conv2d_weight_1.size() == reference_conv_weight_1.size());
        print_error_metrics<bf16, bf16>(
            this->audio_subsample_conv2d_weight_0.data(), reference_conv_weight_0.data(),
            1,
            this->audio_subsample_conv2d_weight_0.size(),1,
            this->audio_subsample_conv2d_weight_0.size(),1
        );
        print_error_metrics<bf16, bf16>(
            this->audio_subsample_conv2d_weight_1.data(), reference_conv_weight_1.data(),
            1,
            this->audio_subsample_conv2d_weight_1.size(), 1,
            this->audio_subsample_conv2d_weight_1.size(), 1
        );
    }
    #endif
    */

    // now, compare with Gemma4AudioSubSampleConvProjection_hidden_states_before_conv

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "DEBUG: Start computing error metrics for audio encoder input" << std::endl;
        buffer<float> reference_projection_input;
        reference_tensors.load_weights(reference_projection_input,"Gemma4AudioSubSampleConvProjection_hidden_states_before_conv");
        size_t reference_projection_input_num_elements = reference_projection_input.size() / audio_payload->num_audios ;

        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, float>(
                audio_payload->mel_spectrograms[i].data(),
                reference_projection_input.data() + i* reference_projection_input_num_elements ,
                1,
                audio_payload->mel_spectrogram_frames_per_audio[i], audio_payload->mel_spectrogram_bins_per_audio[i],
                audio_payload->mel_spectrogram_frames_per_audio[i], audio_payload->mel_spectrogram_bins_per_audio[i]
            );
        }

        //TODO: FIXME: remove it later
        for(int i = 0; i <  audio_payload->num_audios; i++){

            for(int l = 0; l< audio_payload->mel_spectrogram_frames_per_audio[i]*audio_payload->mel_spectrogram_bins_per_audio[i]; l++ ){
                audio_payload->mel_spectrograms[i][l] = (bf16)(reference_projection_input[i* reference_projection_input_num_elements + l]);
            }
        }
    }
    #endif

    // sanity checks, ensure to be the same
    int initial_audio_bins =  audio_payload->mel_spectrogram_bins_per_audio[0];
    for(int i = 1; i < audio_payload->num_audios; i++){
        assert(initial_audio_bins == audio_payload->mel_spectrogram_bins_per_audio[i]);
    }

    // recall the equation for calculation conv2d output as the following
    //H_out = (H_in + 2*padding - K) / stride + 1
    //W_out = (W_in + 2*padding - K) / stride + 1
    auto calc_conv2d_out = [](int h_in, int padding, int k, int stride) -> int {
            return (h_in + 2 * padding - k) / stride + 1;
        };

    std::vector<std::vector<bf16>> subSample_conv_layer_0_res( audio_payload->num_audios);
    //does the first SubSample convlution operation
    int audio_bin_after_conv = calc_conv2d_out(
        initial_audio_bins, this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding,
        this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size, this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride
    );
    std::vector<int> audio_frames_after_conv2d_0(audio_payload->num_audios);

    {

        for(int i = 0; i <audio_payload->num_audios; i++){
            audio_frames_after_conv2d_0[i] = calc_conv2d_out(
                audio_payload->mel_spectrogram_frames_per_audio[i], this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding,
                this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size, this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride
            );
            std::vector<bf16> conv_out(
                this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0 * audio_frames_after_conv2d_0[i] * audio_bin_after_conv
            );
            simd_conv2d(
                audio_payload->mel_spectrograms[i].data(),
                this->audio_subsample_conv2d_weight_0.data(),
                conv_out.data(),
                1,
                audio_payload->mel_spectrogram_frames_per_audio[i], audio_payload->mel_spectrogram_bins_per_audio[i],
                this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0,
                this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size,
                this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride,
                this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding

            );
            // scalar_conv2d(
            //     audio_payload->mel_spectrograms[i].data(),
            //     this->audio_subsample_conv2d_weight_0.data(),
            //     conv_out.data(),
            //     1,
            //     audio_payload->mel_spectrogram_frames_per_audio[i], audio_payload->mel_spectrogram_bins_per_audio[i],
            //     this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0,
            //     this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size,
            //     this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride,
            //     this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding
            // );
            subSample_conv_layer_0_res[i] = std::move(conv_out);
        }

        /*
        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // {

        //     size_t reference_projection_input_num_elements_per_chanel = reference_projection_input_num_elements / this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0;

        //         print_error_metrics<bf16, bf16>(
        //             subSample_conv_layer_0_res[i].data() +  c* audio_frames_after_conv2d_0[i]* audio_bin_after_conv,
        //             Gemma4AudioSubSampleConvProjectionLayer_0_hidden_states_after_conv.data() + i* reference_projection_input_num_elements  + c* reference_projection_input_num_elements_per_chanel,
        //             1,
        //             audio_frames_after_conv2d_0[i], audio_bin_after_conv,
        //             audio_frames_after_conv2d_0[i], audio_bin_after_conv
        //         );

        //         }

        //     }

        //     // //TODO: FIXME:
        //     // for(int i = 0; i < audio_payload->num_audios; i++){
        //     //         for(int c = 0; c <this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0; c++ ){
        //     //             memcpy(
        //     //                 subSample_conv_layer_0_res[i].data() +  c* audio_frames_after_conv2d_0[i]* audio_bin_after_conv,
        //     //                 Gemma4AudioSubSampleConvProjectionLayer_0_hidden_states_after_conv.data() + i* reference_projection_input_num_elements  + c* reference_projection_input_num_elements_per_chanel,
        //     //                 audio_frames_after_conv2d_0[i]* audio_bin_after_conv * sizeof(bf16)
        //     //             );
        //     //         }

        //     //     }

        // }

        // #endif

        */

    // now, perform the reorder with layernorm
    // Python reference does: act(norm(hidden_states.permute(0,2,3,1)).permute(0,3,1,2))
    // Instead of permuting NCHW→NHWC, norming, permuting back, we apply LayerNorm
    // directly over the channel dimension (stride = H*W) in NCHW layout, then ReLU.

        for(int i = 0; i < audio_payload->num_audios; i++){
            int H = audio_frames_after_conv2d_0[i];
            int W = audio_bin_after_conv;
            int HW = H * W;
            bf16* data = subSample_conv_layer_0_res[i].data();

            // LayerNorm over C channels at each (h,w), then ReLU, all in NCHW layout
            layernorm_relu_nchw(data, this->audio_subsample_conv2d_norm_weight_0.data(),
            this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0,HW, 1e-6f);
        }

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // {

        //     size_t reference_projection_input_num_elements_per_chanel = reference_projection_input_num_elements / this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0;

        //         print_error_metrics<bf16, bf16>(
        //             subSample_conv_layer_0_res[i].data() +  c* audio_frames_after_conv2d_0[i]* audio_bin_after_conv,
        //             Gemma4AudioSubSampleConvProjectionLayer_0_hidden_states_after_act.data() + i* reference_projection_input_num_elements  + c* reference_projection_input_num_elements_per_chanel,
        //             1,
        //             audio_frames_after_conv2d_0[i], audio_bin_after_conv,
        //             audio_frames_after_conv2d_0[i], audio_bin_after_conv
        //         );

        //         }

        //     }

        // }
        // #endif
    }

    //subSample_conv_layer_0_res is [num_audio, this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0,
    //    audio_frames_after_conv2d_0[i], audio_bin_after_conv ]

    // now, the second subSample_conv_layer_1
    int audio_bin_after_conv_1 = calc_conv2d_out(
        audio_bin_after_conv, this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding,
        this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size, this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride
    );
    std::vector<int> audio_frames_after_conv2d_1( audio_payload->num_audios);
    std::vector<std::vector<bf16>> subSample_conv_layer_1_res( audio_payload->num_audios);
    {

        for(int i = 0; i < audio_payload->num_audios; i++){
            audio_frames_after_conv2d_1[i] = calc_conv2d_out(
                audio_frames_after_conv2d_0[i], this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding,
                this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size, this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride
            );

            std::vector<bf16> conv_out(
                this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1 * audio_frames_after_conv2d_1[i] * audio_bin_after_conv_1
            );
            simd_conv2d(
                subSample_conv_layer_0_res[i].data(),
                this->audio_subsample_conv2d_weight_1.data(),
                conv_out.data(),
                this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0,
                audio_frames_after_conv2d_0[i], audio_bin_after_conv,
                this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1,
                this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size,
                this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride,
                this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding
            );

            // scalar_conv2d(
            //     subSample_conv_layer_0_res[i].data(),
            //     this->audio_subsample_conv2d_weight_1.data(),
            //     conv_out.data(),
            //     this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0,
            //     audio_frames_after_conv2d_0[i], audio_bin_after_conv,
            //     this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1,
            //     this->parent_npu_ptr->Gemma4E_Audio_conv2d_kernel_size,
            //     this->parent_npu_ptr->Gemma4E_Audio_conv2d_Stride,
            //     this->parent_npu_ptr->Gemma4e_Audio_conv2d_Padding
            // );
            subSample_conv_layer_1_res[i] = std::move(conv_out);
        }

        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // {
        //     buffer<bf16> Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_conv;
        //     reference_tensors.load_weights(Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_conv,"Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_conv");
        //     size_t reference_projection_input_num_elements = Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_conv.size() / audio_payload->num_audios ;
        //     std::cout << "DEBUG: Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_conv" << std::endl;

        //     size_t reference_projection_input_num_elements_per_chanel = reference_projection_input_num_elements / this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1;
        //     for(int i = 0; i < audio_payload->num_audios; i++){
        //         for(int c = 0; c <this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1; c++ ){
        //             print_error_metrics<bf16, bf16>(
        //                 subSample_conv_layer_1_res[i].data() +  c* audio_frames_after_conv2d_1[i]* audio_bin_after_conv_1,
        //                 Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_conv.data() + i* reference_projection_input_num_elements  + c* reference_projection_input_num_elements_per_chanel,
        //                 1,
        //                 audio_frames_after_conv2d_1[i], audio_bin_after_conv_1,
        //                 audio_frames_after_conv2d_1[i], audio_bin_after_conv_1
        //             );
        //         }
        //     }

        // }
        // #endif

        for(int i = 0; i < audio_payload->num_audios; i++){
            int H = audio_frames_after_conv2d_1[i];
            int W = audio_bin_after_conv_1;
            int HW = H * W;
            bf16* data = subSample_conv_layer_1_res[i].data();

            // LayerNorm over C channels at each (h,w), then ReLU, all in NCHW layout
            layernorm_relu_nchw(data, this->audio_subsample_conv2d_norm_weight_1.data(),
            this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1,HW, 1e-6f);
        }
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<bf16> Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_act;
            reference_tensors.load_weights(Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_act,"Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_act");
            size_t reference_projection_input_num_elements = Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_act.size() / audio_payload->num_audios ;
            std::cout << "DEBUG: Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_act" << std::endl;

            size_t reference_projection_input_num_elements_per_chanel = reference_projection_input_num_elements / this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1;

            for(int i = 0; i < audio_payload->num_audios; i++){
                for(int c = 0; c <this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1; c++ ){
                    print_error_metrics<bf16, bf16>(
                        subSample_conv_layer_1_res[i].data() +  c* audio_frames_after_conv2d_1[i]* audio_bin_after_conv_1,
                        Gemma4AudioSubSampleConvProjectionLayer_1_hidden_states_after_act.data() + i* reference_projection_input_num_elements  + c* reference_projection_input_num_elements_per_chanel,
                        1,
                        audio_frames_after_conv2d_1[i], audio_bin_after_conv_1,
                        audio_frames_after_conv2d_1[i], audio_bin_after_conv_1
                    );
                }
            }
        }
        #endif
    }

    // now, subSample_conv_layer_1_res is shape of [ num_audio, this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1, audio_frames_after_conv2d_1[i], audio_bin_after_conv_1 ]

    // reorder subSample_conv_layer_1_res.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
    // aka seq_len = audio_frames_after_conv2d_1[i]
    // the hidden_Size is   audio_bin_after_conv_1 *this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1
    // the reorder output to audio_embedding_projection_input, which is [num_audio, seq_len_per_audio, Padded_Gemma4E_Audio_Multimodal_Output_SIZE  ]
    // NOTE: Padded_Gemma4E_Audio_Multimodal_Output_SIZE>= audio_bin_after_conv_1*this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1 == Gemma4E_Audio_HIDDEN_SIZE

    std::vector<int> seq_len_per_audio(audio_payload->num_audios);
    std::vector<int> start_seq_len_index_per_audio(audio_payload->num_audios);
    int seq_len = 0;
    int seq_len_of_last_audio = 0;

    for(int i = 0; i < audio_payload->num_audios; i++){
        seq_len_per_audio[i] = audio_frames_after_conv2d_1[i] ;

        seq_len += seq_len_per_audio[i];
        seq_len_of_last_audio = seq_len_per_audio[i];

        start_seq_len_index_per_audio[i] = seq_len - seq_len_per_audio[i];
    }
    int seq_len_padded = 0;

    //TODO: FIXME: padding for conv1d

    seq_len_padded = round_up_to_multiple(seq_len, seq_len_pad_requirement_for_MM);

    assert(this->parent_npu_ptr->Gemma4E_Audio_Multimodal_Output_SIZE % MM_tile_K == 0);
    assert(this->parent_npu_ptr->Gemma4E_Audio_Multimodal_Output_SIZE % MM_tile_N == 0);
    assert(this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE  == audio_bin_after_conv_1*this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1);
    assert(this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_0/4 * this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1 ==
    this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE);

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        for(int i = 0; i < audio_payload->num_audios; i++) {
            std::cout << "DEBUG: seq_len_per_audio[" << i << "]: " << seq_len_per_audio[i] << std::endl;
            std::cout << "DEBUG: start_seq_len_index_per_audio[" << i << "]: " << start_seq_len_index_per_audio[i] << std::endl;
        }
        std::cout << "DEBUG: seq_len: " << seq_len << std::endl;
        std::cout << "DEBUG: seq_len_padded: " << seq_len_padded << std::endl;
    }

    #endif

    buffer<bf16> audio_embedding_projection_input = this->sub_sampleConvProjection_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(audio_embedding_projection_input.data() + seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    audio_embedding_projection_input.sync_to_device();

    buffer<bf16> audio_embedding_projection_output = this->sub_sampleConvProjection_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(audio_embedding_projection_output.data() + seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    audio_embedding_projection_output.sync_to_device();

    buffer<bf16> ffn_up_proj_input = this->ffn_up_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(ffn_up_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    ffn_up_proj_input.sync_to_device();

    buffer<bf16> ffn_up_proj_output_down_input = this->ffn_up_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE
    );
    memset(ffn_up_proj_output_down_input.data() + seq_len * Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_MLP_INTERMEDIATE_SIZE * sizeof(bf16)
    );
    ffn_up_proj_output_down_input.sync_to_device();

    buffer<bf16> ffn_down_proj_output = this->ffn_down_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(ffn_down_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    ffn_down_proj_output.sync_to_device();

    buffer<bf16> q_proj_input = this->q_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(q_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    q_proj_input.sync_to_device();

    buffer<bf16> q_proj_output = this->q_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(q_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    q_proj_output.sync_to_device();

    buffer<bf16> k_proj_input = this->k_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(k_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    k_proj_input.sync_to_device();

    buffer<bf16> k_proj_output = this->k_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(k_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    k_proj_output.sync_to_device();

    buffer<bf16> v_proj_input = this->v_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(v_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    v_proj_input.sync_to_device();

    buffer<bf16> v_proj_output = this->v_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(v_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    v_proj_output.sync_to_device();

    buffer<bf16> o_output_proj_input = this->o_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(o_output_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    o_output_proj_input.sync_to_device();

    buffer<bf16> o_output_proj_output = this->o_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(o_output_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    o_output_proj_output.sync_to_device();

    buffer<bf16> conv1d_start_proj_input = this->conv1d_start_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(conv1d_start_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    conv1d_start_proj_input.sync_to_device();

    buffer<bf16> conv1d_start_proj_output = this->conv1d_start_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE
    );
    memset(conv1d_start_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_Conv1d_Linear_OUTPUT_SIZE * sizeof(bf16)
    );
    conv1d_start_proj_output.sync_to_device();

    buffer<bf16> audio_conv1d_input = this->conv1d_app.create_bo_buffer<bf16>(
        (seq_len_padded + audio_payload->num_audios* this->Gemma4E_Audio_padded_requirement_for_conv1d) * Padded_GEMMA4E_Audio_HIDDEN_SIZE

    );
    memset(audio_conv1d_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len + audio_payload->num_audios* this->Gemma4E_Audio_padded_requirement_for_conv1d) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    audio_conv1d_input.sync_to_device();

    buffer<bf16> audio_conv1d_output = this->conv1d_app.create_bo_buffer<bf16>(
        seq_len_padded  * Padded_GEMMA4E_Audio_HIDDEN_SIZE

    );
    memset(audio_conv1d_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    audio_conv1d_output.sync_to_device();

    buffer<bf16> conv1d_end_proj_input = this->conv1d_end_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(conv1d_end_proj_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    conv1d_end_proj_input.sync_to_device();

    buffer<bf16> conv1d_end_proj_output = this->conv1d_end_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(conv1d_end_proj_output.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    conv1d_end_proj_output.sync_to_device();

    buffer<bf16> audio_pre_encode_input = this->audio_pre_encode_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_GEMMA4E_Audio_HIDDEN_SIZE
    );
    memset(audio_pre_encode_input.data() + seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        0, (seq_len_padded - seq_len) * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    );
    audio_pre_encode_input.sync_to_device();

    buffer<bf16> audio_pre_encode_output = this->audio_pre_encode_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_Gemma4E_Audio_Multimodal_Output_SIZE
    );
    memset(audio_pre_encode_output.data() + seq_len * Padded_Gemma4E_Audio_Multimodal_Output_SIZE,
        0, (seq_len_padded - seq_len) * Padded_Gemma4E_Audio_Multimodal_Output_SIZE * sizeof(bf16)
    );
    audio_pre_encode_output.sync_to_device();

    buffer<bf16> audio_to_language_project_input = this->audio_to_language_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * Padded_Gemma4E_Audio_Multimodal_Output_SIZE
    );
    memset(audio_to_language_project_input.data() + seq_len * Padded_Gemma4E_Audio_Multimodal_Output_SIZE,
        0, (seq_len_padded - seq_len) * Padded_Gemma4E_Audio_Multimodal_Output_SIZE * sizeof(bf16)
    );
    audio_to_language_project_input.sync_to_device();

    buffer<bf16> audio_to_language_project_output = this->audio_to_language_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * parent_npu_ptr->Gemma4E_Audio_language_projection_output_size
    );
    memset(audio_to_language_project_output.data() + seq_len * parent_npu_ptr->Gemma4E_Audio_language_projection_output_size,
        0, (seq_len_padded - seq_len) * parent_npu_ptr->Gemma4E_Audio_language_projection_output_size * sizeof(bf16)
    );
    audio_to_language_project_output.sync_to_device();
    assert(parent_npu_ptr->Gemma4E_Audio_language_projection_output_size % MM_tile_N == 0);

    {
        generate_mm_sequence<bf16, bf16>(
            *this->sub_sampleConvProjection_app.seq(),
            seq_len_padded, this->Padded_GEMMA4E_Audio_HIDDEN_SIZE, this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            false, 0, //no bias, no activation
            0,  -10000.0, 1000000.0, // do not clamp on output
            ENABLE_QKV_REORDER, 0// since we don't need it anymore

        );

        generate_mm_sequence<bf16, bf16>(
            *this->audio_pre_encode_proj_app.seq(),
            seq_len_padded, this->Padded_GEMMA4E_Audio_HIDDEN_SIZE, this->Padded_Gemma4E_Audio_Multimodal_Output_SIZE,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            true, 0, //no bias, no activation
            0,  -10000.0, 1000000.0, // do not clamp on output
            ENABLE_QKV_REORDER, 0// since we don't need it anymore

        );
        generate_mm_sequence<bf16, bf16>(
            *this->audio_to_language_proj_app.seq(),
            seq_len_padded, Padded_Gemma4E_Audio_Multimodal_Output_SIZE, parent_npu_ptr->Gemma4E_Audio_language_projection_output_size,
            MM_tile_M, MM_tile_K, MM_tile_N,
            8,8,8,
            rtp_address, rtp_sync_lock_id,
            MM_ROW_SIZE,MM_COL_SIZE,
            0,0,0,
            IS_B_ROW_MAJOR, ENABLE_AXI4, true,
            false, 0, //no bias, no activation
            0,  -10000.0, 1000000.0, // do not clamp on output
            ENABLE_QKV_REORDER, 0// since we don't need it anymore

        );
    }

    memset(audio_embedding_projection_input.data(), 0, audio_embedding_projection_input.size() * sizeof(bf16));

    // reorder of subSample_conv_layer_1_res.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
    for(int i = 0; i < audio_payload->num_audios; i++){

        for(int  h = 0; h < audio_frames_after_conv2d_1[i]; h++){
            for(int w = 0; w < audio_bin_after_conv_1; w++){
                for(int c = 0; c < this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1; c++){
                    size_t input_index = c* audio_frames_after_conv2d_1[i]* audio_bin_after_conv_1 + h* audio_bin_after_conv_1 + w;

                    size_t output_index =  start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE +  h* this->Padded_GEMMA4E_Audio_HIDDEN_SIZE\
                        + w* this->parent_npu_ptr->Gemma4E_Audio_subsampling_conv_channels_1 + c;
                    audio_embedding_projection_input[output_index] = subSample_conv_layer_1_res[i][input_index];
                }
            }
        }
    }
    // subSample_conv_layer_1_res is now shape of [ seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE ]

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> Gemma4AudioSubSampleConvProjection_hidden_states_before_linear;
        reference_tensors.load_weights(Gemma4AudioSubSampleConvProjection_hidden_states_before_linear,"Gemma4AudioSubSampleConvProjection_hidden_states_before_linear");
        size_t reference_projection_input_num_elements = Gemma4AudioSubSampleConvProjection_hidden_states_before_linear.size() / audio_payload->num_audios ;
        std::cout << "DEBUG: Gemma4AudioSubSampleConvProjection_hidden_states_before_linear" << std::endl;

        // for(int i = 0; i < audio_payload->num_audios; i++){
        //     print_error_metrics<bf16, bf16>(
        //         audio_embedding_projection_input.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        //         Gemma4AudioSubSampleConvProjection_hidden_states_before_linear.data() + i* reference_projection_input_num_elements ,
        //         1,
        //         seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
        //         seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
        //     );
        // }
        // //TODO:FIXME: remove it later

        //     memcpy(
        //         seq_len_per_audio[i]* Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
    }
    #endif

    // debugt
    assert(audio_embedding_projection_weight.size() == this->Padded_GEMMA4E_Audio_HIDDEN_SIZE* this->Padded_GEMMA4E_Audio_HIDDEN_SIZE);
    DEBUG_BLOCK(1,
    std::cout << "Padded_GEMMA4E_Audio_HIDDEN_SIZE : " << this->Padded_GEMMA4E_Audio_HIDDEN_SIZE << std::endl;
    )

    audio_embedding_projection_input.sync_to_device();
    this->audio_embedding_projection_weight.sync_to_device();
    FLM_OVERRIDE(audio_sub_sample_proj, sub_sampleConvProjection_app( audio_embedding_projection_input, audio_embedding_projection_weight, audio_embedding_projection_output));
    audio_embedding_projection_output.sync_from_device();

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> Gemma4AudioSubSampleConvProjection_hidden_states_after_linear;
        reference_tensors.load_weights(Gemma4AudioSubSampleConvProjection_hidden_states_after_linear,"Gemma4AudioSubSampleConvProjection_hidden_states_after_linear");
        size_t reference_projection_input_num_elements = Gemma4AudioSubSampleConvProjection_hidden_states_after_linear.size() / audio_payload->num_audios ;
        std::cout << "DEBUG: Gemma4AudioSubSampleConvProjection_hidden_states_after_linear" << std::endl;

        // for(int i = 0; i < audio_payload->num_audios; i++){
        //     print_error_metrics<bf16, bf16>(
        //         audio_embedding_projection_output.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
        //         Gemma4AudioSubSampleConvProjection_hidden_states_after_linear.data() + i* reference_projection_input_num_elements ,
        //         1,
        //         seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
        //         seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
        //     );
        // }
    }
    #endif

    std::vector<bf16> position_embedding(13 *this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE );
    generate_gemma4_audio_rotary_pos_emb(
        this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
        this->parent_npu_ptr->Gemma4E_Audio_attention_chunk_size,
        this->parent_npu_ptr->Gemma4E_Audio_attention_context_left,
        this->parent_npu_ptr->Gemma4E_Audio_attention_context_right,
        position_embedding
    );

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> ref_position_embeddings;
        reference_tensors.load_weights(ref_position_embeddings,"position_embeddings");

        std::cout << "DEBUG: position_embeddings" << std::endl;

        print_error_metrics<bf16, bf16>(
            position_embedding.data(), ref_position_embeddings.data(),
            1,
            position_embedding.size(), 1,
            position_embedding.size(), 1
        );
    }
    #endif

    std::vector<std::vector<int>> audio_sliding_window_attention_mask(audio_payload->num_audios);
    for(int i = 0; i < audio_payload->num_audios; i++){
        create_sliding_window_attention_mask(
            seq_len_per_audio[i],
            this->parent_npu_ptr->Gemma4E_Audio_attention_context_left-1,
            this->parent_npu_ptr->Gemma4E_Audio_attention_context_right,
            audio_sliding_window_attention_mask[i]

        );
    }

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<int> attention_mask_before_block_5d;
        reference_tensors.load_weights(attention_mask_before_block_5d,"attention_mask_before_block_5d");

        size_t reference_attention_mask_num_elements = attention_mask_before_block_5d.size() / audio_payload->num_audios ;
        size_t reference_attention_mask_length =  std::sqrt(reference_attention_mask_num_elements);

        std::cout << "DEBUG: attention_mask_before_block_5d" << std::endl;

        //NOTE: because the reference attention mask is >= each seq_len_per_audio[i]
        // For ease of comparison, we create same mask size and load audio_sliding_windo_attention_mask to it
        std::vector<int> expanded_attention_mask(attention_mask_before_block_5d.size(), 0);

        for(int i = 0; i < audio_payload->num_audios; i++){
            int* mask_start_ptr = expanded_attention_mask.data() + i* reference_attention_mask_num_elements;
            // Fill valid rows (0 to seq_len_per_audio[i]-1) from the C++ sliding window mask
            for(int l = 0; l < seq_len_per_audio[i]; l++){
                memcpy(
                    mask_start_ptr + l* reference_attention_mask_length,
                    audio_sliding_window_attention_mask[i].data() + l* seq_len_per_audio[i],
                    seq_len_per_audio[i] * sizeof(int)
                );
            }
            // Fill padding rows (seq_len_per_audio[i] to reference_attention_mask_length-1)
            // Python's create_bidirectional_mask doesn't mask queries, only keys.
            // So padding rows still have 1s for valid columns within the sliding window.
            int sliding_window_left = this->parent_npu_ptr->Gemma4E_Audio_attention_context_left - 1;
            int sliding_window_right = this->parent_npu_ptr->Gemma4E_Audio_attention_context_right;
            for(int l = seq_len_per_audio[i]; l < (int)reference_attention_mask_length; l++){
                for(int c = 0; c < seq_len_per_audio[i]; c++){
                    int dist = l - c;
                    bool left_mask = (dist >= 0) && (dist < sliding_window_left);
                    bool right_mask = (dist < 0) && (-dist < sliding_window_right);
                    if(left_mask || right_mask){
                        mask_start_ptr[l * reference_attention_mask_length + c] = 1;
                    }
                }
            }
        }
        print_error_metrics<int, int>(
            expanded_attention_mask.data(), attention_mask_before_block_5d.data(),
            1,
            expanded_attention_mask.size(), 1,
            expanded_attention_mask.size(), 1
        );
    }
    #endif

    // now, convert to block attention mask
    // [batch_Size, num_blocks, chunk_size, context_size]
    std::vector<std::vector<int>> block_attention_mask_per_audio(audio_payload->num_audios);
    std::vector<int> num_blocks_per_audio(audio_payload->num_audios);
    std::vector<int> context_size_per_audio(audio_payload->num_audios);
    for(int i = 0; i < audio_payload->num_audios; i++){
        convert_mask_to_blocked(
            audio_sliding_window_attention_mask[i],
            this->parent_npu_ptr->Gemma4E_Audio_attention_chunk_size,
            this->parent_npu_ptr->Gemma4E_Audio_attention_context_left,
            this->parent_npu_ptr->Gemma4E_Audio_attention_context_right,
            block_attention_mask_per_audio[i],
            num_blocks_per_audio[i], context_size_per_audio[i]
        );
    }

    hidden_state.resize(seq_len_padded * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE, 0.0f);
    residual.resize(seq_len_padded * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE, 0.0f);

    memcpy(hidden_state.data(), audio_embedding_projection_output.data(), seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));
    for(int layer_id = 0; layer_id <parent_npu_ptr->Gemma4E_Audio_num_attention_layers; layer_id++){

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            //Gemma4AudioLayer_{layer_idx}_hidden_states_before_ffw1
            buffer<bf16> Gemma4AudioLayer_hidden_State_before_ffw1;
            reference_tensors.load_weights(Gemma4AudioLayer_hidden_State_before_ffw1,"Gemma4AudioLayer_"+std::to_string(layer_id)+"_hidden_states_before_ffw1");
            size_t reference_ffn_input_num_elements = Gemma4AudioLayer_hidden_State_before_ffw1.size() / audio_payload->num_audios ;
            std::cout << "DEBUG: Gemma4AudioLayer_hidden_State_before_ffw1" << std::endl;

            for(int i = 0; i < audio_payload->num_audios; i++){
                print_error_metrics<bf16, bf16>(
                    hidden_state.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioLayer_hidden_State_before_ffw1.data() + i* reference_ffn_input_num_elements ,
                    1,
                    seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }

            // // TODO: FIXME: remove later

            // for(int i = 0; i < audio_payload->num_audios; i++){
            //     memcpy(
            //         hidden_state.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            //         Gemma4AudioLayer_hidden_State_before_ffw1.data() + i* reference_ffn_input_num_elements ,
            //         seq_len_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16)
            //     );
            // }
            // memset(hidden_state.data() + seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE, 0,
            // (seq_len_padded - seq_len)* this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));
        }
        #endif

        ffn_layer(
            ffn_up_proj_input, ffn_up_proj_output_down_input, ffn_down_proj_output,
            ffn_norm_weight[layer_id],
            ffn_post_norm_weight[layer_id],
            audio_ffn_up_weight[layer_id], audio_ffn_down_weight[layer_id],
            seq_len, seq_len_padded,

            audio_ffn_up_input_min[layer_id], audio_ffn_up_input_max[layer_id],
            audio_ffn_up_output_min[layer_id], audio_ffn_up_output_max[layer_id],
            audio_ffn_down_input_min[layer_id], audio_ffn_down_input_max[layer_id],
            audio_ffn_down_output_min[layer_id], audio_ffn_down_output_max[layer_id]

        );

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<bf16> Gemma4AudioLayer_hidden_states_after_ffn1;
            reference_tensors.load_weights(Gemma4AudioLayer_hidden_states_after_ffn1,
                "Gemma4AudioLayer_"+std::to_string(layer_id)+"_hidden_states_after_ffw1");
            size_t reference_attn_output_num_elements =
                Gemma4AudioLayer_hidden_states_after_ffn1.size() / audio_payload->num_audios;
            std::cout << "DEBUG: Gemma4AudioLayer " +std::to_string(layer_id)+ "hidden_states_after_ffw1" << std::endl;

            for (int i = 0; i < audio_payload->num_audios; i++) {
                print_error_metrics<bf16,bf16>(
                    hidden_state.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioLayer_hidden_states_after_ffn1.data() + i * reference_attn_output_num_elements,
                    1,
                    seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif

        memcpy(residual.data(), hidden_state.data(), seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));

        simd_rms_norm(
            hidden_state.data(),
            this->attn_pre_norm_weight[layer_id].data(),
            hidden_state.data(),
            seq_len,
            this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
            seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );

        memcpy(q_proj_input.data(), hidden_state.data(), seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));
        simd_clamp(
            q_proj_input.data(),
            q_proj_input.data(),
            audio_q_input_min[layer_id], audio_q_input_max[layer_id],
            seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );

        {
            generate_mm_sequence<bf16, bf16>(
                *this->q_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                MM_tile_M, MM_tile_K, MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0, //no bias, no activation
                1,  audio_q_output_min[layer_id], audio_q_output_max[layer_id], // do not clamp on output
                ENABLE_QKV_REORDER, 0// since we don't need it anymore

            );
        }

        auto q_proj_run = FLM_OVERRIDE(audio_q_proj, q_proj_app.create_run(
            q_proj_input, this->audio_attn_q_weight[layer_id], q_proj_output
        ), layer_id);

        q_proj_input.sync_to_device();
        this->audio_attn_q_weight[layer_id].sync_to_device();
        q_proj_run.start();

            // setup for K
            generate_mm_sequence<bf16, bf16>(
                *this->k_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                MM_tile_M, MM_tile_K, MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0, //no bias, no activation
                1,  audio_k_output_min[layer_id], audio_k_output_max[layer_id], // do not clamp on output
                ENABLE_QKV_REORDER, 0// since we don't need it anymore

            );
            memcpy(k_proj_input.data(), hidden_state.data(), seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));
            simd_clamp(
                k_proj_input.data(),
                k_proj_input.data(),
                audio_k_input_min[layer_id], audio_k_input_max[layer_id],
                seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
            k_proj_input.sync_to_device();

            auto k_proj_run = FLM_OVERRIDE(audio_k_proj, k_proj_app.create_run(
                k_proj_input, this->audio_attn_k_weight[layer_id], k_proj_output
            ), layer_id);

        q_proj_run.wait();
        q_proj_output.sync_from_device();

        k_proj_input.sync_to_device();
        this->audio_attn_k_weight[layer_id].sync_to_device();
        k_proj_run.start();

            generate_mm_sequence<bf16, bf16>(
                *this->v_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                MM_tile_M, MM_tile_K, MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0, //no bias, no activation
                1,  audio_v_output_min[layer_id], audio_v_output_max[layer_id], // do not clamp on output
                ENABLE_QKV_REORDER, 0// since we don't need it anymore

            );
            memcpy(v_proj_input.data(), hidden_state.data(), seq_len * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE * sizeof(bf16));
            simd_clamp(
                v_proj_input.data(),
                v_proj_input.data(),
                audio_v_input_min[layer_id], audio_v_input_max[layer_id],
                seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
            v_proj_input.sync_to_device();
            auto v_proj_run = FLM_OVERRIDE(audio_v_proj, v_proj_app.create_run(
                v_proj_input, this->audio_attn_v_weight[layer_id], v_proj_output
            ), layer_id);
        k_proj_run.wait();
        k_proj_output.sync_from_device();

        v_proj_input.sync_to_device();
        this->audio_attn_v_weight[layer_id].sync_to_device();
        v_proj_run.start();
            // At this point, q, k proj_output is shape of [num_audio, seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE]
            // But can also be viewed as [num_audio, seq_len_per_audio[i], Padded_Gemma4E_Audio_num_attention_heads, Gemma4E_Audio_attention_head_dim]
            for(int b = 0; b < audio_payload->num_audios; b++){

                for(int s= 0; s< seq_len_per_audio[b]; s++){

                    for(int h_idx = 0; h_idx < parent_npu_ptr->Gemma4E_Audio_num_attention_heads; h_idx++){

                        size_t offset =  (start_seq_len_index_per_audio[b]+s) * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE \
                             + h_idx * Gemma4E_Audio_attention_head_dim;
                        simd_mul(
                            q_proj_output.data() + offset,
                            per_dim_scale_with_softplus_weight[layer_id].data(),
                            q_proj_output.data() + offset,
                            Gemma4E_Audio_attention_head_dim
                        );
                    }
                }
            }
            q_proj_output.sync_to_device();
            simd_mul(
                k_proj_output.data(),
                this->Gemma4E_Audio_k_scale,
                k_proj_output.data(),
                seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
            k_proj_output.sync_to_device();

            // setup o_projection
            generate_mm_sequence<bf16, bf16>(
                *this->o_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                MM_tile_M, MM_tile_K, MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0, //no bias, no activation
                1,  audio_o_output_min[layer_id], audio_o_output_max[layer_id], // do not clamp on output
                ENABLE_QKV_REORDER, 0// since we don't need it anymore

            );
        v_proj_run.wait();
        v_proj_output.sync_from_device();

        // now, compare q, k, v with
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<float> Gemma4AudioAttention_query_states_after_scaling;
            buffer<float> Gemma4AudioAttention_key_states_after_scaling;
            buffer<float> Gemma4AudioAttention_value_states_after_scaling;

            reference_tensors.load_weights(Gemma4AudioAttention_query_states_after_scaling,"Gemma4AudioAttention_"+std::to_string(layer_id)+"_query_states_after_scaling");
            reference_tensors.load_weights(Gemma4AudioAttention_key_states_after_scaling,"Gemma4AudioAttention_"+std::to_string(layer_id)+"_key_states_after_scaling");
            reference_tensors.load_weights(Gemma4AudioAttention_value_states_after_scaling,"Gemma4AudioAttention_"+std::to_string(layer_id)+"_value_states_after_scaling");

            size_t reference_qkv_num_elements = Gemma4AudioAttention_query_states_after_scaling.size() / audio_payload->num_audios ;
            std::cout << "DEBUG: Gemma4AudioAttention_query_states_after_scaling" << std::endl;

            for(int i = 0; i < audio_payload->num_audios; i++){
                print_error_metrics<bf16, float>(
                    q_proj_output.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioAttention_query_states_after_scaling.data() + i* reference_qkv_num_elements ,
                    1,
                    seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
                print_error_metrics<bf16, float>(
                    k_proj_output.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioAttention_key_states_after_scaling.data() + i* reference_qkv_num_elements ,
                    1,
                    seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
                print_error_metrics<bf16, float>(
                    v_proj_output.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioAttention_value_states_after_scaling.data() + i* reference_qkv_num_elements ,
                    1,
                    seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif

        int hidden_size = this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE;
        int num_positions = (int)(position_embedding.size() / hidden_size);

        memset(o_output_proj_input.data(), 0, o_output_proj_input.size() * sizeof(bf16));

        compute_audio_self_attention(
            q_proj_output.data(),
            k_proj_output.data(),
            v_proj_output.data(),
            this->audio_attn_k_rel_weight[layer_id].data(),
            position_embedding.data(),
            block_attention_mask_per_audio,
            num_blocks_per_audio,
            context_size_per_audio,
            seq_len_per_audio,
            start_seq_len_index_per_audio,
            o_output_proj_input.data(),
            audio_payload->num_audios,
            this->parent_npu_ptr->Gemma4E_Audio_attention_chunk_size,
            this->parent_npu_ptr->Gemma4E_Audio_attention_context_left,
            this->parent_npu_ptr->Gemma4E_Audio_attention_context_right,
            this->parent_npu_ptr->Gemma4E_Audio_num_attention_heads,
            Gemma4E_Audio_attention_head_dim,
            hidden_size,
            Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            num_positions,
            parent_npu_ptr->Gemma4E_Audio_attention_softcap,
            -1e9f   // invalid_logits_value
        );

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<float> Gemma4AudioAttention_attn_output_before_post;
            reference_tensors.load_weights(Gemma4AudioAttention_attn_output_before_post,
                "Gemma4AudioAttention_"+std::to_string(layer_id)+"_attn_output_before_post");
            size_t reference_attn_output_num_elements =
                Gemma4AudioAttention_attn_output_before_post.size() / audio_payload->num_audios;
            std::cout << "DEBUG: Gemma4AudioAttention_attn_output_before_post" << std::endl;

            for (int i = 0; i < audio_payload->num_audios; i++) {
                print_error_metrics<bf16,float>(
                    o_output_proj_input.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioAttention_attn_output_before_post.data() + i * reference_attn_output_num_elements,
                    1,
                    seq_len_per_audio[i], hidden_size,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif

        // o_proj
        simd_clamp(
            o_output_proj_input.data(),
            o_output_proj_input.data(),
            audio_o_input_min[layer_id], audio_o_input_max[layer_id],
            seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );
        o_output_proj_input.sync_to_device();
        this->audio_attn_o_weight[layer_id].sync_to_device();
        FLM_OVERRIDE(audio_o_proj, o_proj_app(o_output_proj_input, this->audio_attn_o_weight[layer_id], o_output_proj_output), layer_id);
        o_output_proj_output.sync_from_device();

        simd_rms_norm(
            o_output_proj_output.data(),
            this->attn_post_norm_weight[layer_id].data(),
            hidden_state.data(),
            seq_len,
            this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
            seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            1e-6f
        );

        simd_add(hidden_state.data(), residual.data(), hidden_state.data(),
            seq_len * Padded_GEMMA4E_Audio_HIDDEN_SIZE
        );

        // now compare it
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<bf16> Gemma4AudioAttention_hidden_states_before_conv1d;
            reference_tensors.load_weights(Gemma4AudioAttention_hidden_states_before_conv1d,
                "Gemma4AudioLayer_"+std::to_string(layer_id)+"_hidden_states_before_conv1d");
            size_t reference_attn_output_num_elements =
                Gemma4AudioAttention_hidden_states_before_conv1d.size() / audio_payload->num_audios;
            std::cout << "DEBUG: Gemma4AudioLayer__hidden_states_before_conv1d" << std::endl;

            for (int i = 0; i < audio_payload->num_audios; i++) {
                print_error_metrics<bf16,bf16>(
                    hidden_state.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioAttention_hidden_states_before_conv1d.data() + i * reference_attn_output_num_elements,
                    1,
                    seq_len_per_audio[i], hidden_size,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif
        conv1d_layer(
            layer_id,
            seq_len, seq_len_padded,
            seq_len_per_audio, start_seq_len_index_per_audio,
            conv1d_start_proj_input, conv1d_start_proj_output,
            audio_conv1d_input, audio_conv1d_output,
            conv1d_end_proj_input, conv1d_end_proj_output,
            this->audio_conv_pw1_input_min[layer_id], this->audio_conv_pw1_input_max[layer_id],
            this->audio_conv_pw1_output_min[layer_id], this->audio_conv_pw1_output_max[layer_id],
            this->audio_conv_pw2_input_min[layer_id], this->audio_conv_pw2_input_max[layer_id],
            this->audio_conv_pw2_output_min[layer_id], this->audio_conv_pw2_output_max[layer_id],
            audio_payload,
            #if DEBUG_PRINT_ENCODE_ERROR_METRICS
            &reference_tensors
            #else
            nullptr
            #endif
        );

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<bf16> Gemma4AudioLayer_hidden_states_after_conv1d;
            reference_tensors.load_weights(Gemma4AudioLayer_hidden_states_after_conv1d,
                "Gemma4AudioLayer_"+std::to_string(layer_id)+"_hidden_states_after_conv1d");
            size_t reference_attn_output_num_elements =
                Gemma4AudioLayer_hidden_states_after_conv1d.size() / audio_payload->num_audios;
            std::cout << "DEBUG: Gemma4AudioLayer_hidden_states_after_conv1d" << std::endl;

            for (int i = 0; i < audio_payload->num_audios; i++) {
                print_error_metrics<bf16,bf16>(
                    hidden_state.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioLayer_hidden_states_after_conv1d.data() + i * reference_attn_output_num_elements,
                    1,
                    seq_len_per_audio[i], hidden_size,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif

        ffn_layer(

            ffn_up_proj_input, ffn_up_proj_output_down_input, ffn_down_proj_output,
            ffn_norm_1_weight[layer_id],
            ffn_post_norm_1_weight[layer_id],
            audio_ffn_up_1_weight[layer_id], audio_ffn_down_1_weight[layer_id],
            seq_len, seq_len_padded,

            audio_ffn_up_1_input_min[layer_id], audio_ffn_up_1_input_max[layer_id],
            audio_ffn_up_1_output_min[layer_id], audio_ffn_up_1_output_max[layer_id],
            audio_ffn_down_1_input_min[layer_id], audio_ffn_down_1_input_max[layer_id],
            audio_ffn_down_1_output_min[layer_id], audio_ffn_down_1_output_max[layer_id]

        );
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<bf16> Gemma4AudioLayer_hidden_states_after_ffn1;
            reference_tensors.load_weights(Gemma4AudioLayer_hidden_states_after_ffn1,
                "Gemma4AudioLayer_"+std::to_string(layer_id)+"_hidden_states_after_ffw2");
            size_t reference_attn_output_num_elements =
                Gemma4AudioLayer_hidden_states_after_ffn1.size() / audio_payload->num_audios;
            std::cout << "DEBUG: Gemma4AudioLayer_hidden_states_after_ffw2" << std::endl;

            for (int i = 0; i < audio_payload->num_audios; i++) {
                print_error_metrics<bf16,bf16>(
                    hidden_state.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioLayer_hidden_states_after_ffn1.data() + i * reference_attn_output_num_elements,
                    1,
                    seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif

        simd_rms_norm(
            hidden_state.data(), norm2_weight[layer_id].data(), hidden_state.data(),
            seq_len, this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE, seq_len_padded, Padded_GEMMA4E_Audio_HIDDEN_SIZE,
            1e-6f
        );
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            buffer<bf16> Gemma4AudioLayer_hidden_states_after_norm_out;
            reference_tensors.load_weights(Gemma4AudioLayer_hidden_states_after_norm_out,
                "Gemma4AudioLayer_"+std::to_string(layer_id)+"_hidden_states_after_norm_out");
            size_t reference_attn_output_num_elements =
                Gemma4AudioLayer_hidden_states_after_norm_out.size() / audio_payload->num_audios;
            std::cout << "DEBUG: Gemma4AudioLayer" + std::to_string(layer_id)+ "_hidden_states_after_norm_out" << std::endl;

            for (int i = 0; i < audio_payload->num_audios; i++) {
                print_error_metrics<bf16,bf16>(
                    hidden_state.data() + start_seq_len_index_per_audio[i] * Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                    Gemma4AudioLayer_hidden_states_after_norm_out.data() + i * reference_attn_output_num_elements,
                    1,
                    seq_len_per_audio[i], hidden_size,
                    seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
                );
            }
        }
        #endif
    }

    memcpy(audio_pre_encode_input.data(), hidden_state.data(),
        seq_len* this->Padded_GEMMA4E_Audio_HIDDEN_SIZE* sizeof(bf16)
    );

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> Gemma4AudioPreEncodeProjection_hidden_states_before_linear;
        reference_tensors.load_weights(Gemma4AudioPreEncodeProjection_hidden_states_before_linear,"hidden_states_before_output_proj");
        size_t reference_projection_input_num_elements = Gemma4AudioPreEncodeProjection_hidden_states_before_linear.size() / audio_payload->num_audios ;
        std::cout << "DEBUG: hidden_states_before_output_proj" << std::endl;

        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                audio_pre_encode_input.data() + start_seq_len_index_per_audio[i] * this->Padded_GEMMA4E_Audio_HIDDEN_SIZE,
                Gemma4AudioPreEncodeProjection_hidden_states_before_linear.data() + i* reference_projection_input_num_elements ,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_HIDDEN_SIZE,
                seq_len_per_audio[i], Padded_GEMMA4E_Audio_HIDDEN_SIZE
            );
        }
    }
    #endif

    audio_pre_encode_input.sync_to_device();
    this->audio_pre_encode_weight.sync_to_device();
    FLM_OVERRIDE(audio_pre_encode_proj, audio_pre_encode_proj_app( audio_pre_encode_input, audio_pre_encode_weight, audio_pre_encode_output));
    audio_pre_encode_output.sync_from_device();

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> Gemma4AudioPreEncodeProjection_output;
        reference_tensors.load_weights(Gemma4AudioPreEncodeProjection_output,"audio_model_output_proj_result");
        size_t reference_projection_input_num_elements = Gemma4AudioPreEncodeProjection_output.size() / audio_payload->num_audios ;
        std::cout << "DEBUG: audio_model_output_proj_result" << std::endl;

        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                audio_pre_encode_output.data() + start_seq_len_index_per_audio[i] * this->Padded_Gemma4E_Audio_Multimodal_Output_SIZE,
                Gemma4AudioPreEncodeProjection_output.data() + i* reference_projection_input_num_elements ,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_Multimodal_Output_SIZE,
                seq_len_per_audio[i], Padded_Gemma4E_Audio_Multimodal_Output_SIZE
            );
        }
    }
    #endif

    simd_rms_norm(
        audio_pre_encode_output.data(),
        audio_to_language_project_input.data(),
        seq_len, this->parent_npu_ptr->Gemma4E_Audio_Multimodal_Output_SIZE, seq_len_padded, Padded_Gemma4E_Audio_Multimodal_Output_SIZE,
        1e-6f
    );
    audio_pre_encode_output.sync_to_device();

    audio_to_language_project_input.sync_to_device();
    this->audio_to_language_projection_weight.sync_to_device();
    FLM_OVERRIDE(audio_to_language_proj, audio_to_language_proj_app(audio_to_language_project_input, this->audio_to_language_projection_weight, audio_to_language_project_output));
    audio_to_language_project_output.sync_from_device();

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        buffer<bf16> audio_to_language_projection_output_ref;
        reference_tensors.load_weights(audio_to_language_projection_output_ref,"audio_final_embs_after_embed_audio");
        size_t reference_projection_input_num_elements = audio_to_language_projection_output_ref.size() / audio_payload->num_audios ;
        std::cout << "DEBUG: audio_final_embs_after_embed_audio" << std::endl;

        for(int i = 0; i < audio_payload->num_audios; i++){
            print_error_metrics<bf16, bf16>(
                audio_to_language_project_output.data() + start_seq_len_index_per_audio[i] * this->parent_npu_ptr->Gemma4E_Audio_language_projection_output_size,
                audio_to_language_projection_output_ref.data() + i* reference_projection_input_num_elements ,
                1,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_language_projection_output_size,
                seq_len_per_audio[i], this->parent_npu_ptr->Gemma4E_Audio_language_projection_output_size
            );
        }
    }
    #endif

    std::vector<bf16> dummy_output(audio_to_language_project_output.size()); // return half of the hidden size as dummy output
    memcpy(dummy_output.data(), audio_to_language_project_output.data(), audio_to_language_project_output.size() * sizeof(bf16));
    return dummy_output;
}

