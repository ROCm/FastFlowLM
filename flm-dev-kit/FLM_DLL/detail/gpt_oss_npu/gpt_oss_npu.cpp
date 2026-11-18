// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "gpt_oss_npu_detail.hpp"
#include <algorithm>
#include <exception>
#include <mutex>
// #define DEBUG_TIME_PREFILL 1
#include "utils/error_measure.hpp"
#include "utils/kv_rows_state.hpp"
// #define DEBUG_PREFILL 1
gpt_oss_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance){
    num_hidden_layers = config.get("num_hidden_layers");
    current_context_length = 0;
    is_checkpoint_valid = false;
    this->model_path = config.model_path;
    // Set OpenMP to use up to 4 threads
    omp_set_num_threads(PREFILL_THREADS_BUDGET);

    this->MAX_L = std::max(MAX_L, 4096);
    MAX_L = this->MAX_L;

    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->gemm = std::make_unique<GPT_OSS_Gemm>();
    this->gemm->m = 64;
    this->gemm->k = 128;
    this->gemm->B_in_K_N_block_col_major_order=true; //Note: although set to false, that on means how the dequant data reordeing, but in the actual kernel, it does missing a transpose on sxt blocks

    this->short_seq_gemm = std::make_unique<GPT_OSS_Gemm>();
    this->short_seq_gemm->m = 32;
    this->short_seq_gemm->k = 128;
    this->short_seq_gemm->B_in_K_N_block_col_major_order=true;

    this->sequence = std::make_unique<gpt_oss_npu_sequence>(config, MAX_L);

    this->layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));

    this->layer = this->layer_app_manager->create_app();
    this->sliding_layer = this->layer_app_manager->create_app();
    this->lm_head = this->layer_app_manager->create_app();

    this->layers_run = this->layer_app_manager->create_runlist();
    this->dequant_mlp_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "dequant.xclbin"));
    this->dequant_attn_qkv = this->dequant_mlp_app_manager->create_app();
    this->dequant_attn_o = this->dequant_mlp_app_manager->create_app();

    this->dequant_mlp = this->dequant_mlp_app_manager->create_app();

    this->mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
    this->mha_engine = this->mha_app_manager->create_app();

    this->gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
    this->short_seq_gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "short_seq_mm.xclbin"));
    this->q_proj = this->gemm_app_manager->create_app();
    this->k_proj = this->gemm_app_manager->create_app();
    this->v_proj = this->gemm_app_manager->create_app();
    this->o_proj = this->gemm_app_manager->create_app();
    this->up_proj = this->gemm_app_manager->create_app();
    this->gate_proj = this->gemm_app_manager->create_app();
    this->down_proj = this->gemm_app_manager->create_app();

    this->short_seq_up_proj = this->short_seq_gemm_app_manager->create_app();
    this->short_seq_gate_proj = this->short_seq_gemm_app_manager->create_app();
    this->short_seq_down_proj = this->short_seq_gemm_app_manager->create_app();

    this->expert_mm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "expert.xclbin"));
    this->expert_mm = this->expert_mm_app_manager->create_app();

    int real_num_hidden_layers = 24;
    rms_o_bias_weights.resize(real_num_hidden_layers);
    proj_weights.resize(real_num_hidden_layers);
    kv_caches.resize(real_num_hidden_layers);
    attn_mask.resize(real_num_hidden_layers);
    for (int i = 0; i < real_num_hidden_layers; i++) attn_mask[i] = !is_sliding_window(i);
    host_sink.resize(real_num_hidden_layers);
    host_router_weight.resize(real_num_hidden_layers);
    host_router_bias.resize(real_num_hidden_layers);
    up_bias.resize(real_num_hidden_layers);
    gate_bias.resize(real_num_hidden_layers);
    down_bias.resize(real_num_hidden_layers);
    host_up_bias.resize(real_num_hidden_layers);
    host_gate_bias.resize(real_num_hidden_layers);
    host_down_bias.resize(real_num_hidden_layers);

    size_t total_proj_weights_size_in_byte = (GATE_UP_DOWN_WEIGHT_BYTE);
        // 2 rms weight + o bias + q, k, v o quantization weight
    size_t total_rms_o_bias_weights_size_in_byte = 3 * GPTOSS_HIDDEN_SIZE * 2 + Q_WEIGHT_BYTE + K_WEIGHT_BYTE + V_WEIGHT_BYTE + O_WEIGHT_BYTE;
    // moe router weights and bias
    total_rms_o_bias_weights_size_in_byte = total_rms_o_bias_weights_size_in_byte + EXPERT_WEIGHT_SIZE_IN_BYTE + EXPERT_BIAS_SIZE_IN_BYTE;
    // q, k, v, o bias weights and sink weights
    total_rms_o_bias_weights_size_in_byte = total_rms_o_bias_weights_size_in_byte + (GPTOSS_DQ + GPTOSS_DK + GPTOSS_DV +NUM_ATTN_HEADS) * 2;

    size_t weight_buffer_padding = 512 * 1024 * 1024; // 512 MB
    size_t total_proj_weights_size_padded = (total_proj_weights_size_in_byte + weight_buffer_padding - 1) / weight_buffer_padding * weight_buffer_padding;
    size_t kv_cache_size = MAX_L * GPTOSS_DV * 2;
    size_t kv_cache_size_sliding = 128 * GPTOSS_DV * 2;

    for (size_t i = 0; i < real_num_hidden_layers; i++){
        proj_weights[i] = this->layer.create_bo_buffer<u8>(total_proj_weights_size_padded);
    }

    for (size_t i = 0; i < real_num_hidden_layers; i++){
        rms_o_bias_weights[i] = this->layer.create_bo_buffer<bf16>(total_rms_o_bias_weights_size_in_byte / 2);
    }
    this->final_rms_weight = this->layer.create_bo_buffer<bf16>(GPTOSS_HIDDEN_SIZE);
    this->w_lm_head = this->layer.create_bo_buffer<u8>(LM_HEAD_LM_HEAD_WEIGHT_BUFFER_SIZE);
    this->logits = this->layer.create_bo_buffer<bf16>(VOCAB_SIZE_PADDED);

    for (size_t i = 0; i < real_num_hidden_layers; i++){
        kv_caches[i] = this->layer.create_bo_buffer<bf16>(is_sliding_window(i) ? kv_cache_size_sliding : kv_cache_size);
        kv_caches[i].memset((bf16)0.0f);
        this->kv_caches[i].sync_to_device();
    }
    this->sliding_temp_cache = this->layer.create_bo_buffer<bf16>(kv_cache_size);
    this->sliding_temp_cache.memset((bf16)0.0f);
    this->sliding_temp_cache.sync_to_device();

    rope_weights = this->layer.create_bo_buffer<bf16>(GPTOSS_HEAD_DIM);
    x = this->layer.create_bo_buffer<bf16>(D_ATTEN_O_PADDED_MVM_SIZE);

    this->K03_offset = 0;
    this->K47_offset = 1 * kv_cache_size / 4;
    this->V03_offset = 2 * kv_cache_size / 4;
    this->V47_offset = 3 * kv_cache_size / 4;

    this->sliding_K03_offset = 0 * kv_cache_size_sliding / 4;
    this->sliding_K47_offset = 1 * kv_cache_size_sliding / 4;
    this->sliding_V03_offset = 2 * kv_cache_size_sliding / 4;
    this->sliding_V47_offset = 3 * kv_cache_size_sliding / 4;
    this->set_context_length(0);
    this->embedding = std::make_unique<Embedding>(config.get("vocab_size"), config.get("hidden_size"));

    {
        this->L_pad_required_for_large_MM = this->gemm->m * this->gemm->total_rows;
        this->L_pad_required_for_small_MM = this->short_seq_gemm->m * this->short_seq_gemm->total_rows;

        this->q_proj_dequant_M = DQ;
        this->q_proj_dequant_M_mem_pad = DQ;
        this->q_proj_dequant_N = D_PADDED_Q4NX_COL;
        this->q_proj_dequant_N_mem_pad = round_up_to_multiple(D_PADDED_Q4NX_COL, this->gemm->k);

        this->k_proj_dequant_M = DK;
        this->k_proj_dequant_M_mem_pad = DK;
        this->k_proj_dequant_N = D_PADDED_Q4NX_COL;
        this->k_proj_dequant_N_mem_pad = round_up_to_multiple(D_PADDED_Q4NX_COL, this->gemm->k);

        this->v_proj_dequant_M = DV;
        this->v_proj_dequant_M_mem_pad = DV;
        this->v_proj_dequant_N = D_PADDED_Q4NX_COL;
        this->v_proj_dequant_N_mem_pad = round_up_to_multiple(D_PADDED_Q4NX_COL, this->gemm->k);

        this->o_proj_dequant_M = D_PADDED_128;
        this->o_proj_dequant_M_mem_pad = D_PADDED_128;
        this->o_proj_dequant_N = DQ;
        this->o_proj_dequant_N_mem_pad = round_up_to_multiple(DQ, this->gemm->k);

        this->up_proj_dequant_M = INTERMEDIATE_PADDED_FOUR_Q4NX_ROW;
        this->up_proj_dequant_M_mem_pad = INTERMEDIATE_PADDED_FOUR_Q4NX_ROW;
        this->up_proj_dequant_N = D_PADDED_Q4NX_COL;
        this->up_proj_dequant_N_mem_pad = round_up_to_multiple(D_PADDED_Q4NX_COL, this->gemm->k);

        this->gate_proj_dequant_M = this->up_proj_dequant_M;
        this->gate_proj_dequant_M_mem_pad = this->up_proj_dequant_M_mem_pad ;
        this->gate_proj_dequant_N = this->up_proj_dequant_N;
        this->gate_proj_dequant_N_mem_pad =this->up_proj_dequant_N_mem_pad ;

        this->down_proj_dequant_M = D_PADDED_FOUR_Q4NX_ROW;
        this->down_proj_dequant_M_mem_pad = D_PADDED_FOUR_Q4NX_ROW;
        this->down_proj_dequant_N = INTERMEDIATE_PADDED_Q4NX_COL;
        this->down_proj_dequant_N_mem_pad = round_up_to_multiple(INTERMEDIATE_PADDED_Q4NX_COL, this->gemm->k);

        this->dequant_gate_up_mat_size_padded = this->up_proj_dequant_M_mem_pad * this->up_proj_dequant_N_mem_pad;
        this->dequant_down_mat_size_padded = this->down_proj_dequant_M_mem_pad * this->down_proj_dequant_N_mem_pad;

        router_D_padded = round_up_to_multiple(D, this->gemm->k);
    }

    if (!this->npu->is_preemption_enabled()){
        this->layers_run = layer_app_manager->create_runlist();
    }
    this->logits_temp = buffer<bf16>(this->config.get("vocab_size"));

    // allocate host side buffers
    for (size_t i = 0; i < real_num_hidden_layers; i++){

        host_sink[i] = buffer<bf16>(NUM_ATTN_HEADS);
        host_router_weight[i] = buffer<bf16>(NUM_OF_EXPERT * router_D_padded);
        host_router_bias[i] = buffer<bf16>(NUM_OF_EXPERT);
    }
    for(int i = 0; i < real_num_hidden_layers; i++){
        up_bias[i] =  this->o_proj.create_bo_buffer<bf16>(NUM_OF_EXPERT * up_proj_dequant_M);
        gate_bias[i] =  this->o_proj.create_bo_buffer<bf16>(NUM_OF_EXPERT * gate_proj_dequant_M);
        down_bias[i] =  this->o_proj.create_bo_buffer<bf16>(NUM_OF_EXPERT * down_proj_dequant_M);
        host_up_bias[i] = buffer<bf16>(NUM_OF_EXPERT * up_proj_dequant_M);
        host_gate_bias[i] = buffer<bf16>(NUM_OF_EXPERT * gate_proj_dequant_M);
        host_down_bias[i] = buffer<bf16>(NUM_OF_EXPERT * down_proj_dequant_M);
    }

    // setup some buffers that allow pre allocation
    this->w_qkv = this->dequant_attn_qkv.create_bo_buffer<bf16>(
        (size_t)this->q_proj_dequant_N_mem_pad *
        ( this->q_proj_dequant_M_mem_pad + this->k_proj_dequant_M_mem_pad + this->v_proj_dequant_M_mem_pad ));
    memset(this->w_qkv.data(), 0, this->w_qkv.size()*sizeof(bf16));

    this->w_o   = this->dequant_attn_o.create_bo_buffer<bf16>((size_t)this->o_proj_dequant_M_mem_pad * this->o_proj_dequant_N_mem_pad);
    memset(this->w_o.data(), 0, this->w_o.size()*sizeof(bf16));

    generate_dequant_q4_1_seq(this->dequant_attn_qkv.seq(),
     this->q_proj_dequant_N,
    ( this->q_proj_dequant_M + this->k_proj_dequant_M + this->v_proj_dequant_M),
      qkv_weight_offset_in_byte
    );

    generate_dequant_q4_1_seq(this->dequant_attn_o.seq(),
     this->o_proj_dequant_N,
     this->o_proj_dequant_M,
     qkv_weight_offset_in_byte + Q_WEIGHT_BYTE + K_WEIGHT_BYTE + V_WEIGHT_BYTE
    );

    router_weight_buffer = this->expert_mm.create_bo_buffer<bf16>(NUM_OF_EXPERT * router_D_padded);

    this->w_up_gate_down = this->dequant_mlp.create_bo_buffer<bf16>(
    (this->dequant_gate_up_mat_size_padded * 2 + this->dequant_down_mat_size_padded) * NUM_OF_EXPERT); // dequantize up, gate, down weights
    memset(this->w_up_gate_down.data(), 0, this->w_up_gate_down.size()*sizeof(bf16));

    // launch the dequant app (quant_in, dequant_out)
    generate_mxfp4_gpotss_dequant_sequence
    <Q4NX_ROW, Q4NX_COL, Q4NX_BLOCK_BYTE_SIZE,
    INTERMEDIATE_PADDED_FOUR_Q4NX_ROW, D_PADDED_Q4NX_COL,
    D_PADDED_FOUR_Q4NX_ROW, INTERMEDIATE_PADDED_Q4NX_COL, NUM_OF_EXPERT
    >
    (
        *this->dequant_mlp.seq(),
        this->dequant_gate_up_mat_size_padded, this->dequant_down_mat_size_padded // Each Gate, Up, Down buffer is 2944 x 3072
    );

    this->sequence->generate_lm_head_seq(this->lm_head.seq());
    this->lm_head_run = this->lm_head.create_run(this->logits, this->w_lm_head, this->x, this->final_rms_weight);
}

void gpt_oss_npu::Impl::load_weights(Q4NX& q4nx){
    // read qkv weights
    LOG_VERBOSE(1, "Loading embedding weights");
    this->embedding->init_weights(&q4nx, "model.embed_tokens");

    for (uint32_t layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        u8* w_ptr = this->proj_weights[layer_idx].data();
        u8* w_ptr_end = this->proj_weights[layer_idx].end();
        int layer2load = layer_idx;
         // read up/gate/down weights
        {
            assert(w_ptr + GATE_UP_DOWN_WEIGHT_BYTE < w_ptr_end);
            LOG_VERBOSE(1, "Loading up/gate/down weights");
            buffer<u8> w_up_gate_down;
            q4nx.load_weights(w_up_gate_down, "model.layers." + std::to_string(layer2load) + ".ffn_gate_up_down_exps.weight");
            // For every D rows, half is up, half is gate

            memcpy(w_ptr, w_up_gate_down.data(), w_up_gate_down.size());
            w_ptr += GATE_UP_DOWN_WEIGHT_BYTE;
        }
        if (w_ptr > w_ptr_end){
            std::cout << "Warning: w_ptr != w_ptr_end" << std::endl;
            exit(1);
        }
        w_ptr = (uint8_t*)(this->rms_o_bias_weights[layer_idx].data());
        w_ptr_end = (uint8_t*)(this->rms_o_bias_weights[layer_idx].end());
        {

            LOG_VERBOSE(1, "Loading qkv, router bias weights");
            buffer<bf16> q_bias;
            buffer<bf16> k_bias;
            buffer<bf16> v_bias;
            buffer<bf16> sink;
            buffer<bf16> router_bias;
            buffer<bf16> router_weight;
            q4nx.load_weights(q_bias, "model.layers." + std::to_string(layer2load) + ".self_attn.q_proj.bias");
            q4nx.load_weights(k_bias, "model.layers." + std::to_string(layer2load) + ".self_attn.k_proj.bias");
            q4nx.load_weights(v_bias, "model.layers." + std::to_string(layer2load) + ".self_attn.v_proj.bias");
            q4nx.load_weights(sink,  "model.layers." + std::to_string(layer2load) + ".self_attn.sinks.weight");
            q4nx.load_weights(router_bias, "model.layers." + std::to_string(layer2load) + ".mlp.router.bias");
            q4nx.load_weights(router_weight, "model.layers." + std::to_string(layer2load) + ".mlp.router.weight");

            memcpy(w_ptr, q_bias.data(), GPTOSS_DQ * sizeof(bf16));
            w_ptr += GPTOSS_DQ * sizeof(bf16);

            memcpy(w_ptr, k_bias.data(), GPTOSS_DK * sizeof(bf16));
            w_ptr += GPTOSS_DK * sizeof(bf16);

            memcpy(w_ptr, v_bias.data(), GPTOSS_DV * sizeof(bf16));
            w_ptr += GPTOSS_DV * sizeof(bf16);

            memcpy(w_ptr, sink.data(), NUM_ATTN_HEADS * sizeof(bf16));
            memcpy(this->host_sink[layer_idx].data(), sink.data(), NUM_ATTN_HEADS * sizeof(bf16));
            w_ptr += NUM_ATTN_HEADS * sizeof(bf16);

            const size_t router_bias_bytes =
                static_cast<size_t>(router_bias.size()) * sizeof(bf16);
            if (router_bias_bytes > EXPERT_BIAS_SIZE_IN_BYTE) {
                throw std::runtime_error(
                    "gpt-oss router bias is larger than the device slot");
            }
            if (router_bias_bytes != 0) {
                memcpy(w_ptr, router_bias.data(), router_bias_bytes);
            }
            if (router_bias_bytes < EXPERT_BIAS_SIZE_IN_BYTE) {
                memset(w_ptr + router_bias_bytes, 0,
                       EXPERT_BIAS_SIZE_IN_BYTE - router_bias_bytes);
            }
            const size_t host_bias_bytes = std::min(
                router_bias_bytes, static_cast<size_t>(NUM_OF_EXPERT) * sizeof(bf16));
            memcpy(this->host_router_bias[layer_idx].data(), router_bias.data(),
                   host_bias_bytes);
            w_ptr += EXPERT_BIAS_SIZE_IN_BYTE;

            memcpy(w_ptr, router_weight.data(), EXPERT_WEIGHT_SIZE_IN_BYTE);
            w_ptr += EXPERT_WEIGHT_SIZE_IN_BYTE;

            LOG_VERBOSE(1, "Loading rms weights");
            buffer<bf16> w_rms_pre;
            buffer<bf16> w_o_bias;
            buffer<bf16> w_rms_post;

            // bf16* w_rms_ptr = this->rms_o_bias_weights[layer_idx].data();
            q4nx.load_weights(w_rms_pre, "model.layers." + std::to_string(layer2load) + ".input_layernorm.weight");
            q4nx.load_weights(w_o_bias, "model.layers." + std::to_string(layer2load) + ".self_attn.o_proj.bias");
            q4nx.load_weights(w_rms_post, "model.layers." + std::to_string(layer2load) + ".post_attention_layernorm.weight");

            memcpy((unsigned char*)w_ptr, w_rms_pre.data(), D * sizeof(bf16));
            w_ptr += D * sizeof(bf16);
            memcpy((unsigned char*)w_ptr, w_o_bias.data(), D * sizeof(bf16));
            w_ptr += D * sizeof(bf16);
            memcpy((unsigned char*)w_ptr, w_rms_post.data(), D * sizeof(bf16));
            w_ptr += D * sizeof(bf16);
        }
        {
            LOG_VERBOSE(1, "Loading qkv weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_q;
            buffer<u8> w_k;
            buffer<u8> w_v;
            q4nx.load_weights(w_q, "model.layers." + std::to_string(layer2load) + ".self_attn.q_proj.weight");
            q4nx.load_weights(w_k, "model.layers." + std::to_string(layer2load) + ".self_attn.k_proj.weight");
            q4nx.load_weights(w_v, "model.layers." + std::to_string(layer2load) + ".self_attn.v_proj.weight");

            LOG_VERBOSE(1, "Copying qkv weights to buffer");
            memcpy(w_ptr, w_q.data(), Q_WEIGHT_BYTE);
            w_ptr += Q_WEIGHT_BYTE;
            memcpy(w_ptr, w_k.data(), K_WEIGHT_BYTE);
            w_ptr += K_WEIGHT_BYTE;
            memcpy(w_ptr, w_v.data(), V_WEIGHT_BYTE);
            w_ptr += V_WEIGHT_BYTE;

            LOG_VERBOSE(1, "Loading o weights");
            assert(w_ptr < w_ptr_end);
            buffer<u8> w_o;
            q4nx.load_weights(w_o, "model.layers." + std::to_string(layer2load) + ".self_attn.o_proj.weight");
            memcpy(w_ptr, w_o.data(), w_o.size());
            w_ptr += O_WEIGHT_BYTE;
        }

        for (uint32_t i = 0; i < DH; i++){
            this->rope_weights[i] = i < DH / 2 ? (bf16)1.0f : (bf16)0.0f; // for idx 0.
        }
        buffer<bf16> router_weight;
        q4nx.load_weights(router_weight, "model.layers." + std::to_string(layer2load) + ".mlp.router.weight_prefill");
        //this->host_router_weight[layer_idx].copy_from(router_weight);
        for(int i = 0; i < NUM_OF_EXPERT; i++){
            memcpy(this->host_router_weight[layer_idx].data() + i* router_D_padded,
                router_weight.data() + i *D_PADDED_128, D_PADDED_128 * sizeof(bf16)
            );
            memset(this->host_router_weight[layer_idx].data() + i* router_D_padded + D_PADDED_128,
                0, (router_D_padded - D_PADDED_128) * sizeof(bf16)
            );
        }

        buffer<bf16> _up_bias;
        buffer<bf16> _gate_bias;
        buffer<bf16> _down_bias;
        q4nx.load_weights(_up_bias, "model.layers." + std::to_string(layer2load) + ".mlp.experts.up_proj_bias");
        q4nx.load_weights(_gate_bias, "model.layers." + std::to_string(layer2load) + ".mlp.experts.gate_proj_bias");
        q4nx.load_weights(_down_bias, "model.layers." + std::to_string(layer2load) + ".mlp.experts.down_proj_bias");

        // memcpy(this->host_down_bias[layer_idx].data(), down_bias.data(), 32 * D * sizeof(bf16));
        for(int exp_id = 0; exp_id < NUM_OF_EXPERT; exp_id++){

            memcpy(up_bias[layer_idx].data() + exp_id * up_proj_dequant_M, _up_bias.data() + exp_id * D, D * sizeof(bf16));
            memset(up_bias[layer_idx].data() + exp_id * up_proj_dequant_M + D, 0, (up_proj_dequant_M - D) * sizeof(bf16));

            memcpy(gate_bias[layer_idx].data() + exp_id * gate_proj_dequant_M, _gate_bias.data() + exp_id * D, D * sizeof(bf16));
            memset(gate_bias[layer_idx].data() + exp_id * gate_proj_dequant_M + D, 0, (gate_proj_dequant_M - D) * sizeof(bf16));

            memcpy(down_bias[layer_idx].data() + exp_id * down_proj_dequant_M, _down_bias.data() + exp_id * D, D * sizeof(bf16));
            memset(down_bias[layer_idx].data() + exp_id * down_proj_dequant_M + D, 0, (down_proj_dequant_M - D) * sizeof(bf16));

            memcpy(host_up_bias[layer_idx].data() + exp_id * up_proj_dequant_M, _up_bias.data() + exp_id * D, D * sizeof(bf16));
            memset(host_up_bias[layer_idx].data() + exp_id * up_proj_dequant_M + D, 0, (up_proj_dequant_M - D) * sizeof(bf16));

            memcpy(host_gate_bias[layer_idx].data() + exp_id * gate_proj_dequant_M, _gate_bias.data() + exp_id * D, D * sizeof(bf16));
            memset(host_gate_bias[layer_idx].data() + exp_id * gate_proj_dequant_M + D, 0, (gate_proj_dequant_M - D) * sizeof(bf16));
            memcpy(host_down_bias[layer_idx].data() + exp_id * down_proj_dequant_M, _down_bias.data() + exp_id * D, D * sizeof(bf16));
            memset(host_down_bias[layer_idx].data() + exp_id * down_proj_dequant_M + D, 0, (down_proj_dequant_M - D) * sizeof(bf16));
        }
        this->proj_weights[layer_idx].sync_to_device();
        this->rms_o_bias_weights[layer_idx].sync_to_device();
        this->rope_weights.sync_to_device();
    }
    LOG_VERBOSE(1, "Loading lm head weights");
    {
        buffer<u8> w_lm_head_w;
        q4nx.load_weights(w_lm_head_w, "lm_head.weight");

        memcpy(this->w_lm_head.data(),w_lm_head_w.data(), sizeof(uint8_t)*w_lm_head_w.size());
        this->w_lm_head.sync_to_device();

        buffer<bf16> w_lm_head_norm_weight_buf;
        // load the rms weight
        q4nx.load_weights(w_lm_head_norm_weight_buf, "model.norm.weight");
        memcpy(this->final_rms_weight.data(),  w_lm_head_norm_weight_buf.data(), sizeof(bf16)*w_lm_head_norm_weight_buf.size());
        this->final_rms_weight.sync_to_device();
    }

    LOG_VERBOSE(1, "Weights loaded, clearing context");
    this->clear_context();
}

void gpt_oss_npu::Impl::_set_rope_weights(int idx){
    //TODO: avx512 optimization
    bf16* w_rope_ptr = this->rope_weights.data();
    for (uint32_t i = 0; i < DH / 2; i++){
        w_rope_ptr[i] = bf16(std::cos(inv_freq[i] * idx) * GPTOSS_ROPE_SCALING);
        w_rope_ptr[i + DH / 2] = bf16(std::sin(inv_freq[i] * idx) * GPTOSS_ROPE_SCALING);
    }
    this->rope_weights.sync_to_device();
}

void gpt_oss_npu::Impl::set_context_length(int L){
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1, false);
    this->sequence->gen_layer_seq(this->sliding_layer.seq(), L + 1, true);

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.reset();
        for (size_t i = 0; i < num_hidden_layers; i++){
            if (is_sliding_window(i)){
                this->layers_run.add(this->sliding_layer.create_run(this->proj_weights[i], this->x, this->rms_o_bias_weights[i], this->rope_weights, this->kv_caches[i]));
            }
            else{
                this->layers_run.add(this->layer.create_run(this->proj_weights[i], this->x, this->rms_o_bias_weights[i], this->rope_weights, this->kv_caches[i]));
            }
        }
    }

    this->_set_rope_weights(L);
}

buffer<bf16> gpt_oss_npu::Impl::forward(buffer<bf16>& x){

    int L = this->current_context_length;
    std::cout << "Index: " << L << std::endl;

    for (int i = 0; i < num_hidden_layers; i++){
        this->x.copy_from(x);

        this->x.sync_to_device();
        if (is_sliding_window(i)){
            this->x.sync_to_device();
            this->proj_weights[i].sync_to_device();
            this->rms_o_bias_weights[i].sync_to_device();
            this->rope_weights.sync_to_device();
            this->kv_caches[i].sync_to_device();
            this->sliding_layer(this->proj_weights[i], this->x, this->rms_o_bias_weights[i], this->rope_weights, this->kv_caches[i]);
            this->x.sync_from_device();
            this->kv_caches[i].sync_from_device();
        }
        else{
            this->x.sync_to_device();
            this->proj_weights[i].sync_to_device();
            this->rms_o_bias_weights[i].sync_to_device();
            this->rope_weights.sync_to_device();
            this->kv_caches[i].sync_to_device();
            this->layer(this->proj_weights[i], this->x, this->rms_o_bias_weights[i], this->rope_weights, this->kv_caches[i]);
            this->x.sync_from_device();
            this->kv_caches[i].sync_from_device();
        }
        bool contains_nan = false;
        for (int j = 0; j < this->x.size(); j++){
            if (std::isnan((float)this->x[j])){
                header_print("error", "x contains nan at layer " << i << " at index " << j);
                contains_nan = true;
                break;
            }
        }
    }

    this->x.sync_from_device();

    this->set_context_length(this->current_context_length + 1);

    return this->x;
}

buffer<bf16> gpt_oss_npu::Impl::forward(int ids){
    this->embedding->forward(ids, this->x);

    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled()){
        this->layers_run.execute();
        flm::npu::wait_for_completion(this->layers_run, "layers_run");
    }
    else{
        for (int i = 0; i < num_hidden_layers; i++){
            if (is_sliding_window(i)){
                this->sliding_layer(this->proj_weights[i], this->x, this->rms_o_bias_weights[i], this->rope_weights, this->kv_caches[i]);
            }
            else{
                this->layer(this->proj_weights[i], this->x, this->rms_o_bias_weights[i], this->rope_weights, this->kv_caches[i]);
            }
        }
    }

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits;
}

buffer<bf16> gpt_oss_npu::Impl::prefill(std::vector<int>& ids, void* payload){
#ifdef MVPREFILL
    header_print("warning", "Prefilling with Decoder, Slow!");
    return this->_prefill_with_mv(ids);
#else
    if (ids.size() < 32){
        return this->_prefill_with_mv(ids);
    }
    else{
        return this->_prefill_with_mm(ids);
    }
#endif
}

buffer<bf16> gpt_oss_npu::Impl::_prefill_with_mv(std::vector<int>& ids){
    for (int i = 0; i < ids.size(); i++){
        this->embedding->forward(ids[i], this->x);
        this->x.sync_to_device();
        if (!this->npu->is_preemption_enabled()){
            this->layers_run.execute();
            flm::npu::wait_for_completion(this->layers_run, "layers_run");
        }
        else{
            for (int j = 0; j < num_hidden_layers; j++){
                if (is_sliding_window(j)){
                    this->sliding_layer(this->proj_weights[j], this->x, this->rms_o_bias_weights[j], this->rope_weights, this->kv_caches[j]);
                }
                else{
                    this->layer(this->proj_weights[j], this->x, this->rms_o_bias_weights[j], this->rope_weights, this->kv_caches[j]);
                }
            }
        }
        this->set_context_length(this->current_context_length + 1);
    }
    this->final_rms_weight.sync_to_device();
    this->w_lm_head.sync_to_device();
    this->x.sync_from_device();
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();

    return this->logits;
}

buffer<bf16> gpt_oss_npu::Impl::_prefill_with_mm(std::vector<int>& ids){

    // debug
    #ifdef DEBUG_PREFILL
        int debug_ids[135] = {200006,  17360, 200008,   3575,    553,  17554, 162016,     11,    261,
            4410,   6439,   2359,  22203,    656,   7788,  17527,    558,  87447,
            100594,     25,    220,   1323,     19,     12,   3218,    198,   6576,
            3521,     25,    220,   1323,     20,     12,   3114,     12,   2040,
                279,  30377,    289,     25,  14093,    279,      2,  13888,  18403,
                25,   8450,     11,  49159,     11,   1721,     13,  21030,   2804,
                413,   7360,    395,   1753,   3176,     13, 200007, 200006,  77944,
            200008,      2,  68406,    279,   3575,    553,    261,  10297,  29186,
                364, 200007, 200006,   1428, 200008,   4827,    382,    290,   9029,
                328,  10128,     30,   3253,   1991,   1665,   4561,    306,  10128,
                30,   3253,   4150,    382,    480,    316,   9741,     30, 200007,
            200006, 173781, 200005,  17196, 200008,    976,   9029,    328,  10128,
                382,  12650,     13,    623,   2086,    328,   1665,   6405,    306,
            10128,    382,    220,   5462,   5749,     13,    623,   9324,   2870,
            12650,    326,   9741,    382,    220,  21401,  10753,     13, 200002};

        // assign debug ids to ids
        ids.clear();
        ids.resize(135);
        for (int i = 0; i < 135; i++){
            ids[i] = debug_ids[i];
        }
    #endif

    int D_PADDED = round_up_to_multiple(D, this->gemm->k); // 294
    constexpr int DQ = GPTOSS_DQ;
    constexpr int DK = GPTOSS_DK;
    constexpr int DV = GPTOSS_DV;
    const int L_MIN = 256;
    const int L_CHUNK_SIZE = 128;
    int L_in = ids.size();
    int L_begin = this->current_context_length;
    int L_end = L_begin + L_in;
    // get the L in product of 16
    int L_begin_chunked = (L_begin / L_CHUNK_SIZE) * L_CHUNK_SIZE;
    int L_offset = L_begin - L_begin_chunked;
    int L_end_chunked = L_begin_chunked + (L_in + L_offset + L_MIN - 1) / L_MIN * L_MIN;
    int L_padded = L_end_chunked - L_begin_chunked;

    int L_effective = L_end - L_begin;
    int sliding_l_begin = (L_begin_chunked - SLIDING_WINDOW) > 0 ? (L_begin_chunked - SLIDING_WINDOW) : 0;
    int sliding_l_length = (L_end_chunked - sliding_l_begin);

    #ifdef DEBUG_PREFILL
    std::cout << "L_begin: " << L_begin << " L_end: " << L_end << " L_effective: " << L_effective << std::endl;
    std::cout << "L_begin_chunked: " << L_begin_chunked << " L_end_chunked: " << L_end_chunked << " L_padded: " << L_padded << std::endl;
    std::cout << "L_offset: " << L_offset << std::endl;
    std::cout << "sliding_l_begin: " << sliding_l_begin << " sliding_l_length: " << sliding_l_length << std::endl;
    #endif
    //allocate temporary buffers

    // Test code, use layer 0 only
    buffer<bf16> hidden_states = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * D_PADDED);
    buffer<bf16> buffer_o = this->o_proj.create_bo_buffer<bf16>((size_t)L_padded * this->o_proj_dequant_M_mem_pad);
    memset(hidden_states.data(), 0, hidden_states.size() * sizeof(bf16));
    tensor_2d<bf16> hidden_states_tensor(hidden_states, D_PADDED, L_offset);
    tensor_2d<bf16> buffer_o_tensor(buffer_o, this->o_proj_dequant_M_mem_pad, L_offset);

    buffer<bf16> buffer_q = this->q_proj.create_bo_buffer<bf16>((size_t)L_padded * this->q_proj_dequant_M_mem_pad);
    tensor_2d<bf16> buffer_q_tensor(buffer_q, this->q_proj_dequant_M_mem_pad, L_offset);

    buffer<bf16> buffer_k = this->k_proj.create_bo_buffer<bf16>((size_t)L_padded * this->k_proj_dequant_M_mem_pad );
    tensor_2d<bf16> buffer_k_tensor(buffer_k, this->k_proj_dequant_M_mem_pad , L_offset);

    buffer<bf16> buffer_v = this->v_proj.create_bo_buffer<bf16>((size_t)L_padded * this->v_proj_dequant_M_mem_pad);
    tensor_2d<bf16> buffer_v_tensor(buffer_v, this->v_proj_dequant_M_mem_pad, L_offset);

    buffer<bf16> residual = buffer<bf16>((size_t)L_padded * D_PADDED);
    tensor_2d<bf16> residual_tensor(residual, D_PADDED, L_offset);

    buffer<bf16> router_score_buffer = this->expert_mm.create_bo_buffer<bf16>(L_padded * NUM_OF_EXPERT);
    tensor_2d<bf16> router_score_tensor(router_score_buffer, NUM_OF_EXPERT, L_offset);

    buffer<bf16> router_score_buffer_cpu = buffer<bf16>(L_padded * NUM_OF_EXPERT);
    tensor_2d<bf16> router_score_tensor_cpu(router_score_buffer_cpu, NUM_OF_EXPERT, L_offset);

    buffer<float> mlp_acc = buffer<float>((size_t)L_effective * D_PADDED);
    tensor_2d<float> mlp_acc_tensor(mlp_acc, D_PADDED, 0); // no offset
    // the CPU's experts accumulate on their own and are added to mlp_acc after both workers finish,
    // so the order of the float adds does not depend on which worker finishes first
    // (one element where no expert goes to the CPU: a buffer of zero bytes throws "Invalid size for bytes allocation", which made every
    // prefill fail on Linux, where NPU_MIN_LENGTH is 0)
    buffer<float> mlp_acc_cpu = buffer<float>(NPU_MIN_LENGTH > 0 ? L_effective * D_PADDED : 1);
    tensor_2d<float> mlp_acc_cpu_tensor(mlp_acc_cpu, D_PADDED, 0); // no offset

    int initial_expert_l_size = 64;
    //NOTE: on windows, buffer is shared across different xclbin, so don't distinguish buffer by different xclbin
    buffer<bf16> expert_buffer_x = this->up_proj.create_bo_buffer<bf16>(initial_expert_l_size * D_PADDED); // L x 2944
    buffer<bf16> expert_buffer_gate_result = this->gate_proj.create_bo_buffer<bf16>(initial_expert_l_size * gate_proj_dequant_M_mem_pad); // L x 2944
    buffer<bf16> expert_buffer_up_result = this->up_proj.create_bo_buffer<bf16>(initial_expert_l_size * up_proj_dequant_M_mem_pad); // L x 2944
    // Second ping-pong expert buffer set. While the NPU runs expert i on
    // one set, the host builds expert i+1's runs into the other set.
    buffer<bf16> expert_buffer_x2 = this->up_proj.create_bo_buffer<bf16>(initial_expert_l_size * D_PADDED);
    buffer<bf16> expert_buffer_gate_result2 = this->gate_proj.create_bo_buffer<bf16>(initial_expert_l_size * gate_proj_dequant_M_mem_pad);
    buffer<bf16> expert_buffer_up_result2 = this->up_proj.create_bo_buffer<bf16>(initial_expert_l_size * up_proj_dequant_M_mem_pad);
    buffer<bf16> expert_buffer_x_cpu = buffer<bf16>(initial_expert_l_size * D_PADDED); // L x 2944
    buffer<bf16> expert_buffer_gate_result_cpu = buffer<bf16>(initial_expert_l_size * gate_proj_dequant_M_mem_pad); // L x 2944
    buffer<bf16> expert_buffer_up_result_cpu = buffer<bf16>(initial_expert_l_size * up_proj_dequant_M_mem_pad); // L x 2944

    buffer<bf16> buffer_attn_out = this->mha_engine.create_bo_buffer<bf16>(L_padded * DQ);
    tensor_2d<bf16> buffer_attn_out_tensor(buffer_attn_out, DQ, L_offset);

    this->gemm->generate_seq(this->o_proj.seq(),
        L_padded, this->o_proj_dequant_N_mem_pad, this->o_proj_dequant_M_mem_pad, 0, true, 0,
        (rms_o_bias_offset_in_byte ) / sizeof(bf16) +  GPTOSS_HIDDEN_SIZE

    );  //DEBUG:
    this->gemm->generate_seq(this->q_proj.seq(),
    L_padded, q_proj_dequant_N_mem_pad, q_proj_dequant_M_mem_pad,
    0, true, 0, 0);

    this->gemm->generate_seq(this->k_proj.seq(),
    L_padded, k_proj_dequant_N_mem_pad, k_proj_dequant_M_mem_pad,
    q_proj_dequant_N_mem_pad * q_proj_dequant_M_mem_pad, true, 0, DQ);

    this->gemm->generate_seq(this->v_proj.seq(),
    L_padded, this->v_proj_dequant_N_mem_pad, this->v_proj_dequant_M_mem_pad,
    q_proj_dequant_N_mem_pad * q_proj_dequant_M_mem_pad + k_proj_dequant_N_mem_pad*k_proj_dequant_M_mem_pad, true, 0, DQ + DK);

    generate_expert_mm_sequence(this->expert_mm.seq(), L_padded, D_PADDED, NUM_OF_EXPERT);
    // The scratch window the sliding layers attend over is rebuilt from the rings on every call, but what an earlier call left in the
    // rows outside what it writes is not harmless to the kernel: start every prefill from zeros, so a prefill continued live and one
    // that follows a restored state see the same scratch.
    this->_zero_temp_cache();
    // do embedding
    #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
    for (int i = 0; i < L_effective; i++){
        this->embedding->forward(ids[i], residual_tensor[i]);
    }

    #ifdef DEBUG_PREFILL
        SafeTensors reference_tensor(this->model_path + "/prefill_data_debug.safetensors");
        std::cout << "L_EFFECTIVE: " << L_effective << std::endl;
        int useful_L = 0;
        int L_computed = 0;
        float gen_seq_time = 0;
        time_utils::time_with_unit mlp_time = {0, "us"};
        time_utils::time_point mlp_start = time_utils::now();
        time_utils::time_point mlp_end = time_utils::now();

    #endif
    for (int layer_idx = 0; layer_idx < num_hidden_layers; layer_idx++){
        #ifdef DEBUG_PREFILL
            if(layer_idx > 0){
                break;//only run layer 0
            }
        #endif
        memset(mlp_acc.data(), 0, (size_t)L_effective * D_PADDED * sizeof(float)); // this is accumulator, must be cleard to 0.

        buffer<bf16> pre_attn_rms_weight = buffer<bf16>(this->rms_o_bias_weights[layer_idx].data() + rms_o_bias_offset_in_byte / 2, D);
        buffer<bf16> post_attn_rms_weight = buffer<bf16>(this->rms_o_bias_weights[layer_idx].data() + rms_o_bias_offset_in_byte / 2 + D + D, D);

        // Launch both attention weight dequants on the NPU and let them run while the
        // host builds the MHA sequence and computes the pre-attn RMS norm below (neither
        // depends on w_qkv / w_o). w_qkv and w_o are produced and consumed on-device, so
        // the earlier host round-trip syncs were pure DMA overhead and are removed.
        this->rms_o_bias_weights[layer_idx].sync_to_device();
        flm_rt::run dequant_qkv_run = this->dequant_attn_qkv.create_run(this->w_qkv, this->rms_o_bias_weights[layer_idx]);
        flm_rt::run dequant_o_run   = this->dequant_attn_o.create_run(this->w_o,   this->rms_o_bias_weights[layer_idx]);
        dequant_qkv_run.start();
        dequant_o_run.start();

        this->sequence->gen_mha_engine_seq(this->mha_engine.seq(), L_begin_chunked, L_end_chunked, this->host_sink[layer_idx].data(), is_sliding_window(layer_idx));
        #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
        for (int i = 0; i < L_effective; i++) {
            _rms_norm(hidden_states_tensor[i], residual_tensor[i], pre_attn_rms_weight);
        }
        flm::npu::wait_for_completion(dequant_qkv_run, "gpt-oss layer " + std::to_string(layer_idx) + " dequant_attn_qkv");
        flm::npu::wait_for_completion(dequant_o_run, "gpt-oss layer " + std::to_string(layer_idx) + " dequant_attn_o");
        hidden_states.sync_to_device();
        auto q_proj_run = this->q_proj.create_run(buffer_q, hidden_states, w_qkv,   this->rms_o_bias_weights[layer_idx]);
        auto k_proj_run = this->k_proj.create_run(buffer_k, hidden_states, w_qkv,   this->rms_o_bias_weights[layer_idx]);
        auto v_proj_run = this->v_proj.create_run(buffer_v, hidden_states, w_qkv,   this->rms_o_bias_weights[layer_idx]);
        q_proj_run.start();
        flm::npu::wait_for_completion(q_proj_run, "q_proj_run");
        k_proj_run.start();
        buffer_q.sync_from_device();
        // do q_rope
        #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
        for (int ll = 0; ll < L_effective; ll++) {
            _rope(buffer_q_tensor[ll], ll + L_begin);
        }
        buffer_q.sync_to_device();
        #ifdef DEBUG_PREFILL
            std::cout << "Error after Q rope at layer " << layer_idx << ": ";
            buffer<bf16> after_q_rope;
            reference_tensor.load_weights(
                after_q_rope, "layer_" + std::to_string(layer_idx) + "_q_rope"
            );
            print_error_metrics<bf16, bf16>(
                buffer_q.data(), after_q_rope.data(),
                1,
                L_effective,  DQ,
                L_effective, DQ

            );
        #endif

        flm::npu::wait_for_completion(k_proj_run, "k_proj_run");
        buffer_k.sync_from_device();
        v_proj_run.start();

        #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
        for (int ll = 0; ll < L_effective; ll++) {
            _rope(buffer_k_tensor[ll], ll + L_begin);
        }
        buffer_k.sync_to_device();
        flm::npu::wait_for_completion(v_proj_run, "v_proj_run");
        buffer_v.sync_from_device();

        #ifdef DEBUG_PREFILL
            std::cout << "Error after K proj at layer " << layer_idx << ": ";
            buffer<bf16> after_k_proj;
            reference_tensor.load_weights(
                after_k_proj, "layer_" + std::to_string(layer_idx) + "_k_rope"
            );
            print_error_metrics<bf16, bf16>(
                buffer_k.data(), after_k_proj.data(),
                1,
                L_effective,  DK,
                L_effective, DK
            );
            std::cout << "Error after V proj at layer " << layer_idx << ": ";
            buffer<bf16> after_v_proj;
            reference_tensor.load_weights(
                after_v_proj, "layer_" + std::to_string(layer_idx) + "_v_proj"
            );
            print_error_metrics<bf16, bf16>(
                buffer_v.data(), after_v_proj.data(),
                1,
                L_effective,  DV,
                L_effective, DV
            );
        #endif

        if (is_sliding_window(layer_idx)){
            fill_kv_cache_sliding(this->sliding_temp_cache, buffer_k, buffer_v, L_offset, L_begin, L_end, L_effective, sliding_l_begin, L_end_chunked, layer_idx);
            this->sliding_temp_cache.sync_to_device();
            this->temp_dirty = true;
        }
        else{
            fill_kv_cache(buffer_k, buffer_v, L_offset, L_begin, L_effective, layer_idx);
            this->kv_caches[layer_idx].sync_to_device();
        }

        // start = time_utils::now();
        // Step 5: apply mha
        buffer_q.sync_to_device();
        buffer_attn_out.sync_from_device();
        if (is_sliding_window(layer_idx)){
            this->mha_engine(buffer_attn_out, buffer_q, this->sliding_temp_cache);
        }
        else{
            this->mha_engine(buffer_attn_out, buffer_q, this->kv_caches[layer_idx]);
        }
        buffer_attn_out.sync_from_device();

        // o_projection
        this->w_o.sync_to_device();
        this->rms_o_bias_weights[layer_idx].sync_to_device();
        this->o_proj(buffer_o, buffer_attn_out, this->w_o, this->rms_o_bias_weights[layer_idx]);
        buffer_o.sync_from_device();
        #ifdef DEBUG_PREFILL
            std::cout << "Error after O proj at layer " << layer_idx << ": ";
            buffer<bf16> after_o_proj;
            reference_tensor.load_weights(
                after_o_proj, "layer_" + std::to_string(layer_idx) + "_attn_out"
            );
            print_error_metrics<bf16, bf16>(
                buffer_o.data(),  after_o_proj.data(),
                1,
                L_effective,  D,
                L_effective, this->o_proj_dequant_M_mem_pad
            );
        #endif

        #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
        for (int i = 0; i < L_effective; i++) {
            _residual_add(residual_tensor[i], residual_tensor[i], buffer_o_tensor[i], D);
            _rms_norm(hidden_states_tensor[i], residual_tensor[i], post_attn_rms_weight, D);
        }

        #ifdef DEBUG_PREFILL
            std::cout << "Error after Post Attn RMS at layer " << layer_idx << ": ";
            buffer<bf16> after_post_rms;
            reference_tensor.load_weights(
                after_post_rms, "layer_" + std::to_string(layer_idx) + "_after_post_layernorm"
            );
            print_error_metrics<bf16, bf16>(
                hidden_states.data(), after_post_rms.data(),
                1,
                L_effective,  D,
                L_effective, D_PADDED
            );
        #endif

        hidden_states.sync_to_device();
        // determine experts
        this->router_weight_buffer.copy_from(this->host_router_weight[layer_idx]);
        this->router_weight_buffer.sync_to_device();
        this->expert_mm(router_score_buffer, hidden_states, router_weight_buffer);
        router_score_buffer.sync_from_device();
        // The MLP weight dequant depends only on the quantized per-layer weights, not the
        // router result, so launch it on the NPU now and let it run while the host does
        // the router bias-add + top-k + expert sizing below. Waited just before the
        // expert-MM phase that consumes w_up_gate_down.
        flm_rt::run dequant_mlp_run = this->dequant_mlp.create_run(this->proj_weights[layer_idx], this->w_up_gate_down);
        dequant_mlp_run.start();
        #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
        for (int i = 0; i < L_effective; i++) {
            _residual_add(router_score_tensor_cpu[i], router_score_tensor[i], this->host_router_bias[layer_idx], 32);
        }
        router_score_buffer.sync_to_device();
        std::vector<std::vector<std::pair<int, bf16>>> expert_to_index_map; // The time index list for each expert
        // std::cout << "finding experts" << std::endl;
        std::vector<std::pair<int, int>> expert_id_length;
        expert_id_length.reserve(32);
        try {
            moeRouterPrefill<4, 32>(router_score_tensor_cpu[0].data(), expert_to_index_map, L_effective);
            for (int i = 0; i < 32; i++) {
                expert_id_length.push_back(std::make_pair(i, expert_to_index_map[i].size()));
            }
            std::sort(expert_id_length.begin(), expert_id_length.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second > b.second;
            });
        }
        catch (const std::exception& e) {
            std::cout << "Error in finding experts: " << e.what() << std::endl;
            exit(1);
        }
        int max_length = expert_id_length[0].second;
        int max_length_padded;
        if(max_length <= L_pad_required_for_large_MM ){
            max_length_padded = round_up_to_multiple(max_length, this->L_pad_required_for_small_MM);
        }
        else{
            max_length_padded = round_up_to_multiple(max_length, this->L_pad_required_for_large_MM);
        }

        //NOTE: on windows, buffer is shared across different xclbin, so don't distinguish buffer by different xclbin
        if(expert_buffer_x.size() < (size_t)max_length_padded * D_PADDED){
            expert_buffer_x = this->up_proj.create_bo_buffer<bf16>((size_t)max_length_padded * D_PADDED); // L x 2944
            expert_buffer_gate_result = this->gate_proj.create_bo_buffer<bf16>((size_t)max_length_padded * gate_proj_dequant_M_mem_pad); // L x 2944
            expert_buffer_up_result = this->up_proj.create_bo_buffer<bf16>((size_t)max_length_padded * up_proj_dequant_M_mem_pad); // L x 2944
            expert_buffer_x_cpu = buffer<bf16>((size_t)max_length_padded * D_PADDED); // L x 2944
            expert_buffer_gate_result_cpu = buffer<bf16>((size_t)max_length_padded * gate_proj_dequant_M_mem_pad); // L x 2944
            expert_buffer_up_result_cpu = buffer<bf16>((size_t)max_length_padded * up_proj_dequant_M_mem_pad); // L x 2944
        }

        // Size the second ping-pong buffer set ("slot 1") with a memory budget.
        // expert_id_length is sorted descending by routed-token count, and the
        // ping-pong loop below walks the NPU tasks in that same sorted order,
        // alternating slots (task 0 -> slot 0, task 1 -> slot 1, task 2 -> slot 0,
        // ...) without ever reordering tasks. slot 0 is sized above to the largest
        // routed expert, so it can hold any expert in the layer. slot 1 only needs
        // to hold experts that actually get pipelined through it, so it is sized to
        // the second-largest routed expert here rather than to the largest -- when
        // routing is skewed toward a single expert the second-largest is much
        // smaller, and mirroring slot 0's size would permanently reserve far more
        // device memory than slot 1 ever uses.
        //
        // To stay safe when routing is skewed toward two or more similarly large
        // ("co-dominant") experts -- where the second-largest is itself large, so
        // two max-sized buffer sets resident at once can exhaust device memory --
        // slot 1 is additionally capped at GPTOSS_PINGPONG_SLOT1_MAX_LEN tokens.
        // Experts too large for the capped slot 1 are not pipelined; the loop below
        // runs them serially on the slot-0 buffers instead. This bounds the peak
        // expert-buffer footprint to one max-sized set (slot 0) plus one capped set
        // (slot 1), regardless of how skewed routing is.
        int second_max_length = (expert_id_length.size() > 1) ? expert_id_length[1].second : 0;
        int second_max_length_padded;
        if (second_max_length <= L_pad_required_for_large_MM) {
            second_max_length_padded = round_up_to_multiple(second_max_length, this->L_pad_required_for_small_MM);
        }
        else {
            second_max_length_padded = round_up_to_multiple(second_max_length, this->L_pad_required_for_large_MM);
        }
        // Cap slot 1 at GPTOSS_PINGPONG_SLOT1_MAX_LEN tokens, padded with the same
        // rule as any expert MM so the capped buffer still holds a whole number of
        // padded expert MMs.
        int slot1_cap_padded;
        if ((int)GPTOSS_PINGPONG_SLOT1_MAX_LEN <= L_pad_required_for_large_MM) {
            slot1_cap_padded = round_up_to_multiple((int)GPTOSS_PINGPONG_SLOT1_MAX_LEN, this->L_pad_required_for_small_MM);
        }
        else {
            slot1_cap_padded = round_up_to_multiple((int)GPTOSS_PINGPONG_SLOT1_MAX_LEN, this->L_pad_required_for_large_MM);
        }
        if (second_max_length_padded > slot1_cap_padded) {
            second_max_length_padded = slot1_cap_padded;
        }
        // Floor to the same default size used when these buffers were first created
        // (see initial_expert_l_size above), so a layer with 0 or 1 experts routed to
        // the NPU -- where slot 1 is never actually written by the ping-pong loop --
        // still gets a nonzero-size device buffer instead of a zero-length allocation.
        if (second_max_length_padded < initial_expert_l_size) {
            second_max_length_padded = initial_expert_l_size;
        }
        if (expert_buffer_x2.size() < (size_t)second_max_length_padded * D_PADDED) {
            expert_buffer_x2 = this->up_proj.create_bo_buffer<bf16>((size_t)second_max_length_padded * D_PADDED);
            expert_buffer_gate_result2 = this->gate_proj.create_bo_buffer<bf16>((size_t)second_max_length_padded * gate_proj_dequant_M_mem_pad);
            expert_buffer_up_result2 = this->up_proj.create_bo_buffer<bf16>((size_t)second_max_length_padded * up_proj_dequant_M_mem_pad);
        }

        #ifdef DEBUG_PREFILL
            std::cout << "Generating expert MM sequence" << std::endl;
            mlp_start = time_utils::now();
        #endif
        //TODO: also a set of down buffer?
        // Wait for the MLP weight dequant launched above (overlapped with the host router
        // bias-add + top-k + expert sizing) before the expert-MM phase consumes it.
        flm::npu::wait_for_completion(dequant_mlp_run, "gpt-oss layer " + std::to_string(layer_idx) + " dequant_mlp");

        #ifdef DEBUG_PREFILL
        {
            int cpu_mm_count = 0;
            int npu_mm_count = 0;
            for (int expert_counter = 0; expert_counter < expert_id_length.size(); expert_counter++) {
                int expert_id = expert_id_length[expert_counter].first;
                int L_this_expert = expert_to_index_map[expert_id].size();
                if (L_this_expert < 32){
                    cpu_mm_count++;
                }
                if (L_this_expert > 8){
                    npu_mm_count++;
                }
            }
            header_print("info", "Layer " << layer_idx << " Expert MM distribution: CPU MM (<32): " << cpu_mm_count << ", NPU MM (>8): " << npu_mm_count);
        }
        #endif
        // doing expert with CPU/NPU overlap. The split is fixed by length before either worker starts:
        // the CPU and the NPU round differently, so an expert that could land on either one would make
        // the prefill, and every token sampled after it, differ from run to run even with a fixed seed.
        struct ExpertTask {
            int expert_id;
            int length;
        };
        std::vector<ExpertTask> npu_tasks;
        std::vector<ExpertTask> cpu_tasks;
        npu_tasks.reserve(expert_id_length.size());
        cpu_tasks.reserve(expert_id_length.size());

        for (int expert_counter = 0; expert_counter < expert_id_length.size(); expert_counter++) {
            int expert_id = expert_id_length[expert_counter].first;
            int L_this_expert = expert_to_index_map[expert_id].size();
            if (L_this_expert == 0) {
                continue;
            }
            if (L_this_expert > NPU_MIN_LENGTH) {
                npu_tasks.push_back({expert_id, L_this_expert});
            }
            else {
                cpu_tasks.push_back({expert_id, L_this_expert});
            }
        }
        if (!cpu_tasks.empty()) {
            memset(mlp_acc_cpu.data(), 0, (size_t)L_effective * D_PADDED * sizeof(float));
        }

        auto accumulate_expert = [&](tensor_2d<float>& acc_tensor, int expert_id, int L_this_expert, buffer<bf16>* expert_buffer_up_result_current){
            #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
            for (int i = 0; i < L_this_expert; i++) {
                int time_index = expert_to_index_map[expert_id][i].first;
                bf16 score = expert_to_index_map[expert_id][i].second;
                _expert_mac(acc_tensor[time_index], expert_buffer_up_result_current->data() + (i*up_proj_dequant_M_mem_pad), score, D);
            }
        };

        auto padded_length = [&](int L_this_expert) {
            if (L_this_expert <= this->L_pad_required_for_large_MM) {
                return round_up_to_multiple(L_this_expert, this->L_pad_required_for_small_MM);
            }
            return round_up_to_multiple(L_this_expert, this->L_pad_required_for_large_MM);
        };

        #ifdef DEBUG_PREFILL
        std::mutex debug_mutex;
        #endif
        auto cpu_worker = [&](){
            for (const ExpertTask& task : cpu_tasks) {
                _cpu_expert_task(
                    layer_idx,
                    task.expert_id,
                    task.length,
                    expert_to_index_map[task.expert_id],
                    expert_buffer_x_cpu,
                    hidden_states_tensor,
                    expert_buffer_gate_result_cpu,
                    expert_buffer_up_result_cpu
                );
                #ifdef DEBUG_PREFILL
                header_print("layer " + std::to_string(layer_idx), "CPU processed expert " << task.expert_id << " with length " << task.length);
                #endif
                accumulate_expert(mlp_acc_cpu_tensor, task.expert_id, task.length, &expert_buffer_up_result_cpu);
                #ifdef DEBUG_PREFILL
                {
                    std::lock_guard<std::mutex> guard(debug_mutex);
                    useful_L += task.length;
                    L_computed += padded_length(task.length);
                }
                #endif
            }
        };

        std::exception_ptr npu_error;
        auto npu_worker = [&](){
            try {
                for (const ExpertTask& task : npu_tasks) {
                    try {
                        _npu_expert_task(
                            layer_idx,
                            task.expert_id,
                            task.length,
                            expert_to_index_map[task.expert_id],
                            expert_buffer_x,
                            hidden_states_tensor,
                            expert_buffer_gate_result,
                            expert_buffer_up_result
                        );
                    } catch (const std::exception& e) {
                        throw std::runtime_error(
                            "gpt-oss layer " + std::to_string(layer_idx) +
                            " expert " + std::to_string(task.expert_id) + ": " + e.what());
                    }
                    accumulate_expert(mlp_acc_tensor, task.expert_id, task.length, &expert_buffer_up_result);
                    #ifdef DEBUG_PREFILL
                    {
                        std::lock_guard<std::mutex> guard(debug_mutex);
                        useful_L += task.length;
                        L_computed += padded_length(task.length);
                    }
                    header_print("layer " + std::to_string(layer_idx), "NPU processed expert " << task.expert_id << " with length " << task.length);
                    #endif
                }
            } catch (...) {
                npu_error = std::current_exception();
            }
        };

#if CPU_MAX_LENGTH == 0
        // ---- single-threaded 2-deep ping-pong expert dispatch pipeline ----
        // All experts run on the NPU. Each expert's per-MM host overhead (generate_seq
        // + gather + create_run + record) dominates prefill host time and runs with the
        // NPU idle under a synchronous start->wait path. Overlap the host prep of expert
        // i+1 with the NPU execution of expert i's gate+up MMs, using a ping-pong of two
        // expert buffer sets. At most 2 dispatches are in flight; the in-order single
        // stream and unchanged h2d coherence keep results identical to the serial path.
        (void)npu_worker; (void)cpu_worker;
        {
            buffer<bf16>* Xs[2] = { &expert_buffer_x,           &expert_buffer_x2 };
            buffer<bf16>* Gs[2] = { &expert_buffer_gate_result, &expert_buffer_gate_result2 };
            buffer<bf16>* Us[2] = { &expert_buffer_up_result,   &expert_buffer_up_result2 };

            // gate+up are batched into one runlist (record both -> one flush -> one
            // drain) instead of two separate start()/wait() round-trips. down stays a
            // single run.
            struct PipeExpert { flm_rt::runlist rgu; flm_rt::run rd; int expert_id=0; int L=0; int Lp=0; int slot=0; bool is_short=false; bool valid=false; };

            // Split-phase prep. prepA builds gate+up runs (gather X, sync bias),
            // overlapped with the NPU gate+up of the *previous* expert. prepB builds the
            // down run, overlapped with the NPU down of the *current* expert. Each run
            // captures its own executable/TXN at create_run time, so regenerating the
            // shared seq objects for the next expert never disturbs an in-flight run.
            auto prepA = [&](int ti, int slot) -> PipeExpert {
                PipeExpert pe; pe.valid=true; pe.slot=slot;
                const ExpertTask& t = npu_tasks[ti];
                pe.expert_id = t.expert_id; pe.L = t.length;
                buffer<bf16>& X = *Xs[slot]; buffer<bf16>& G = *Gs[slot]; buffer<bf16>& U = *Us[slot];
                pe.is_short = (t.length <= this->L_pad_required_for_small_MM);
                pe.Lp = pe.is_short ? round_up_to_multiple(t.length, this->L_pad_required_for_small_MM)
                                    : round_up_to_multiple(t.length, this->L_pad_required_for_large_MM);
                // Bind the gate+up runlist to the context of whichever GEMM app owns the runs.
                pe.rgu = pe.is_short ? this->short_seq_gemm_app_manager->create_runlist()
                                     : this->gemm_app_manager->create_runlist();
                // Shared-Executable Collapse: the per-expert weight/bias base is
                // supplied at dispatch as a buffer-binding byte offset, so the TXN
                // (generate_seq) is expert-independent and all experts at a given Lp
                // share one cached executable. Offsets are bf16-element strides * 2B.
                const size_t estride = (size_t)(2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded);
                const size_t w_gate_off = (size_t)t.expert_id * estride * sizeof(bf16);
                const size_t w_up_off   = ((size_t)t.expert_id * estride + dequant_gate_up_mat_size_padded) * sizeof(bf16);
                const size_t b_gate_off = (size_t)t.expert_id * gate_proj_dequant_M * sizeof(bf16);
                const size_t b_up_off   = (size_t)t.expert_id * up_proj_dequant_M * sizeof(bf16);
                if (pe.is_short) {
                    _gen_expert_seq_if_needed(this->short_seq_gemm.get(), short_seq_gate_proj, pe.Lp, gate_proj_dequant_N_mem_pad, gate_proj_dequant_M_mem_pad, 0, true, 1, 0);
                    _gen_expert_seq_if_needed(this->short_seq_gemm.get(), short_seq_up_proj,   pe.Lp, up_proj_dequant_N_mem_pad,   up_proj_dequant_M_mem_pad,   0, true, 2, 0);
                } else {
                    _gen_expert_seq_if_needed(this->gemm.get(), this->gate_proj, pe.Lp, gate_proj_dequant_N_mem_pad, gate_proj_dequant_M_mem_pad, 0, true, 1, 0);
                    _gen_expert_seq_if_needed(this->gemm.get(), this->up_proj,   pe.Lp, up_proj_dequant_N_mem_pad,   up_proj_dequant_M_mem_pad,   0, true, 2, 0);
                }
                for (int i = 0; i < t.length; i++) {
                    int time_index = expert_to_index_map[t.expert_id][i].first;
                    memcpy(X.data() + (i*D_PADDED_128), hidden_states_tensor[time_index].data(), D * sizeof(bf16));
                }
                X.sync_to_device();
                this->gate_bias[layer_idx].sync_to_device();
                this->up_bias[layer_idx].sync_to_device();
                this->down_bias[layer_idx].sync_to_device();
                if (pe.is_short) {
                    pe.rgu.add(short_seq_gate_proj.create_run(G, X, bound_arg{&this->w_up_gate_down, w_gate_off}, bound_arg{&this->gate_bias[layer_idx], b_gate_off}));
                    pe.rgu.add(short_seq_up_proj.create_run(U, X, bound_arg{&this->w_up_gate_down, w_up_off}, bound_arg{&this->up_bias[layer_idx], b_up_off}));
                } else {
                    pe.rgu.add(this->gate_proj.create_run(G, X, bound_arg{&this->w_up_gate_down, w_gate_off}, bound_arg{&this->gate_bias[layer_idx], b_gate_off}));
                    pe.rgu.add(this->up_proj.create_run(U, X, bound_arg{&this->w_up_gate_down, w_up_off}, bound_arg{&this->up_bias[layer_idx], b_up_off}));
                }
                return pe;
            };
            auto prepB = [&](PipeExpert& pe) {
                buffer<bf16>& X = *Xs[pe.slot]; buffer<bf16>& U = *Us[pe.slot];
                const size_t estride = (size_t)(2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded);
                const size_t w_down_off = ((size_t)pe.expert_id * estride + (size_t)dequant_gate_up_mat_size_padded * 2) * sizeof(bf16);
                const size_t b_down_off = (size_t)pe.expert_id * down_proj_dequant_M * sizeof(bf16);
                if (pe.is_short) {
                    _gen_expert_seq_if_needed(this->short_seq_gemm.get(), short_seq_down_proj, pe.Lp, down_proj_dequant_N_mem_pad, down_proj_dequant_M_mem_pad, 0, true, 0, 0);
                    pe.rd = short_seq_down_proj.create_run(U, X, bound_arg{&this->w_up_gate_down, w_down_off}, bound_arg{&this->down_bias[layer_idx], b_down_off});
                } else {
                    _gen_expert_seq_if_needed(this->gemm.get(), this->down_proj, pe.Lp, down_proj_dequant_N_mem_pad, down_proj_dequant_M_mem_pad, 0, true, 0, 0);
                    pe.rd = this->down_proj.create_run(U, X, bound_arg{&this->w_up_gate_down, w_down_off}, bound_arg{&this->down_bias[layer_idx], b_down_off});
                }
            };

            const int E = (int)npu_tasks.size();

            // Peel off the leading experts that are too large for the capped slot 1
            // and run them serially on the slot-0 buffers (which are sized to the
            // largest expert, so they can hold any expert). npu_tasks is sorted by
            // descending routed length, so the oversized experts are always a prefix.
            // The remaining experts all fit slot 1 and are pipelined below. Expert
            // contributions are still accumulated in descending task order, so the
            // result is identical to pipelining every expert.
            const int slot1_rows = (int)(expert_buffer_x2.size() / D_PADDED);
            int serial_count = 0;
            while (serial_count + 1 < E &&
                   padded_length(npu_tasks[serial_count + 1].length) > slot1_rows) {
                serial_count++;
            }
            for (int i = 0; i < serial_count; i++) {
                const ExpertTask& t = npu_tasks[i];
                _npu_expert_task(
                    layer_idx,
                    t.expert_id,
                    t.length,
                    expert_to_index_map[t.expert_id],
                    expert_buffer_x,
                    hidden_states_tensor,
                    expert_buffer_gate_result,
                    expert_buffer_up_result
                );
                accumulate_expert(mlp_acc_tensor, t.expert_id, t.length, &expert_buffer_up_result);
            }

            if (serial_count < E) {
                PipeExpert cur = prepA(serial_count, 0);
                prepB(cur);
                for (int i = serial_count; i < E; i++) {
                    buffer<bf16>& X = *Xs[cur.slot]; buffer<bf16>& G = *Gs[cur.slot]; buffer<bf16>& U = *Us[cur.slot];
                    // launch gate+up(i) as one runlist: both recorded, then a single
                    // flush -> NPU runs gate then up back-to-back with no inter-op host
                    // handshake, while the host preps expert i+1.
                    cur.rgu.execute();
                    // overlap: build expert i+1's gate+up runs (dominant host cost) while
                    // the NPU executes gate+up(i).
                    PipeExpert nxt;
                    if (i + 1 < E) nxt = prepA(i + 1, cur.slot ^ 1);
                    flm::npu::wait_for_completion(cur.rgu, "gpt-oss layer " + std::to_string(layer_idx) + " expert " + std::to_string(cur.expert_id) + " upgate");  // ONE drain for gate+up; marks both runs' bindings
                    // GLU on host, then down(i). Invalidate only the cur.L rows read.
                    G.sync_from_device((size_t)cur.L * gate_proj_dequant_M_mem_pad);
                    U.sync_from_device((size_t)cur.L * up_proj_dequant_M_mem_pad);
                    #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
                    for (int k = 0; k < cur.L; k++) {
                        _elementwise_mul(X.data() + (k*D_PADDED_128), U.data() + (k*up_proj_dequant_M_mem_pad), G.data() + (k*gate_proj_dequant_M_mem_pad), D);
                    }
                    X.sync_to_device();
                    cur.rd.start();
                    // Overlap: build expert i+1's down run while the NPU executes down(i).
                    if (i + 1 < E) prepB(nxt);
                    flm::npu::wait_for_completion(cur.rd, "gpt-oss layer " + std::to_string(layer_idx) + " expert " + std::to_string(cur.expert_id) + " down");
                    U.sync_from_device((size_t)cur.L * up_proj_dequant_M_mem_pad);
                    accumulate_expert(mlp_acc_tensor, cur.expert_id, cur.L, &U);
                    cur = nxt;
                }
            }
        }
#else
        std::thread npu_thread(npu_worker);
        // Stage the dequantized MLP weights to host for the CPU expert path, overlapped
        // with the NPU expert MMs already in flight.
        this->w_up_gate_down.sync_from_device();
        std::thread cpu_thread(cpu_worker);
        npu_thread.join();
        cpu_thread.join();
        if (npu_error) {
            std::rethrow_exception(npu_error);
        }
        if (!cpu_tasks.empty()) {
            #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
            for (int i = 0; i < L_effective; i++){
                float* acc = mlp_acc_tensor[i].data();
                const float* acc_cpu = mlp_acc_cpu_tensor[i].data();
                for (uint32_t j = 0; j < D; j++){
                    acc[j] += acc_cpu[j];
                }
            }
        }
#endif
        #ifdef DEBUG_PREFILL
            mlp_end = time_utils::now();
            auto mlp_this_layer = time_utils::duration_us(mlp_start, mlp_end);
            mlp_time.first += mlp_this_layer.first;
            mlp_time.second = mlp_this_layer.second;
        #endif
        // export to bf16, this is the end.
        #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
        for (int i = 0; i < L_effective; i++){
            _export_to_bf16(hidden_states_tensor[i], mlp_acc_tensor[i], D);
            _residual_add(residual_tensor[i], hidden_states_tensor[i], residual_tensor[i], D);
        }
    } // end for layer_idx

    buffer<bf16> last_output = residual_tensor[L_effective - 1];

    this->set_context_length(this->current_context_length + L_effective);

    // _rms_norm(this->x_lm_head_exposed, last_output, this->final_rms_weight);
    memcpy(this->x.data(), last_output.data(), D * sizeof(bf16));
    this->x.sync_to_device();
    this->final_rms_weight.sync_to_device();
    this->w_lm_head.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits;
    #ifdef DEBUG_PREFILL
    float ratio = (float)useful_L / (float)L_computed;
    float gen_seq_time_per_s = gen_seq_time / 1000000.0f;
    std::cout << std::endl;
    header_print("info", "Expert MM efficiency: " << useful_L << " / " << L_computed << " = " << ratio);
    header_print("info", "Gen seq time per second: " << gen_seq_time_per_s << " s");
    header_print("info", "MLP time: " << mlp_time.first / 1e6 << " s");
    #endif
    return logits;
}

inline void gpt_oss_npu::Impl::_cpu_expert_task(
    int layer_idx,
    int expert_id,
    int L_this_expert,
    std::vector<std::pair<int, bf16>>& expert_to_index_map,
    buffer<bf16>& expert_buffer_x,
    tensor_2d<bf16>& hidden_states_tensor,
    buffer<bf16>& expert_buffer_gate_result,
    buffer<bf16>& expert_buffer_up_result
){
    for (int i = 0; i < L_this_expert; i++) {
        int time_index = expert_to_index_map[i].first;
        memcpy(expert_buffer_x.data() + (i * D_PADDED_128), hidden_states_tensor[time_index].data(), D * sizeof(bf16)); // only copy valid D (2880)
    }

    // first does the gate projection, then up projection
    _cpu_mm(
        expert_buffer_gate_result.data(),
        expert_buffer_x.data(),
        this->w_up_gate_down.data() + expert_id * (2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded),
        this->host_gate_bias[layer_idx].data() + expert_id * up_proj_dequant_M,
        L_this_expert,
        GPTOSS_HIDDEN_SIZE,
        GPTOSS_HIDDEN_SIZE,
        up_proj_dequant_M_mem_pad,
        D_PADDED_128,
        up_proj_dequant_N_mem_pad
    );
    _clamp_gelu(expert_buffer_gate_result, -std::numeric_limits<float>::infinity(), 7.0f, L_this_expert, GPTOSS_HIDDEN_SIZE, up_proj_dequant_M_mem_pad);

    _cpu_mm(
        expert_buffer_up_result.data(),
        expert_buffer_x.data(),
        this->w_up_gate_down.data() + expert_id * (2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded) + dequant_gate_up_mat_size_padded,
        this->host_up_bias[layer_idx].data() + expert_id * up_proj_dequant_M,
        L_this_expert,
        GPTOSS_HIDDEN_SIZE,
        GPTOSS_HIDDEN_SIZE,
        up_proj_dequant_M_mem_pad,
        D_PADDED_128,
        up_proj_dequant_N_mem_pad
    );
    _clamp_inc(expert_buffer_up_result, -7.0f, 7.0f, L_this_expert, GPTOSS_HIDDEN_SIZE, up_proj_dequant_M_mem_pad);

    // do a simple element-wise multiplication for GLU

    #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
    for (int i = 0; i < L_this_expert; i++) {
        _elementwise_mul(expert_buffer_x.data() + (i*D_PADDED_128), expert_buffer_up_result.data() + (i*up_proj_dequant_M_mem_pad), expert_buffer_gate_result.data() + (i*gate_proj_dequant_M_mem_pad), D);
    }

    _cpu_mm(
        expert_buffer_up_result.data(),
        expert_buffer_x.data(),
        this->w_up_gate_down.data() + expert_id * (2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded) + dequant_gate_up_mat_size_padded * 2,
        this->host_down_bias[layer_idx].data() + expert_id * up_proj_dequant_M,
        L_this_expert,
        GPTOSS_HIDDEN_SIZE,
        GPTOSS_HIDDEN_SIZE,
        up_proj_dequant_M_mem_pad,
        D_PADDED_128,
        up_proj_dequant_N_mem_pad
    );
}
inline void gpt_oss_npu::Impl::_npu_expert_task(
    int layer_idx,
    int expert_id,
    int L_this_expert,
    std::vector<std::pair<int, bf16>>& expert_to_index_map,
    buffer<bf16>& expert_buffer_x,
    tensor_2d<bf16>& hidden_states_tensor,
    buffer<bf16>& expert_buffer_gate_result,
    buffer<bf16>& expert_buffer_up_result
){
    if (L_this_expert <= this->L_pad_required_for_small_MM){
        _npu_short_expert_task(
            layer_idx,
            expert_id,
            L_this_expert,
            expert_to_index_map,
            expert_buffer_x,
            hidden_states_tensor,
            expert_buffer_gate_result,
            expert_buffer_up_result
        );
    }
    else{
        _npu_large_expert_task(
            layer_idx,
            expert_id,
            L_this_expert,
            expert_to_index_map,
            expert_buffer_x,
            hidden_states_tensor,
            expert_buffer_gate_result,
            expert_buffer_up_result
        );
    }
}

inline void gpt_oss_npu::Impl::_npu_short_expert_task(
    int layer_idx,
    int expert_id,
    int L_this_expert,
    std::vector<std::pair<int, bf16>>& expert_to_index_map,
    buffer<bf16>& expert_buffer_x,
    tensor_2d<bf16>& hidden_states_tensor,
    buffer<bf16>& expert_buffer_gate_result,
    buffer<bf16>& expert_buffer_up_result
){

    int L_this_expert_padded = round_up_to_multiple(L_this_expert, this->L_pad_required_for_small_MM);
    // Shared-Executable Collapse: zero the baked per-expert weight/bias offsets;
    // they are supplied at dispatch via bound_arg byte offsets below.
    const size_t estride = (size_t)(2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded);
    const size_t w_gate_off = (size_t)expert_id * estride * sizeof(bf16);
    const size_t w_up_off   = ((size_t)expert_id * estride + dequant_gate_up_mat_size_padded) * sizeof(bf16);
    const size_t w_down_off = ((size_t)expert_id * estride + (size_t)dequant_gate_up_mat_size_padded * 2) * sizeof(bf16);
    const size_t b_gate_off = (size_t)expert_id * gate_proj_dequant_M * sizeof(bf16);
    const size_t b_up_off   = (size_t)expert_id * up_proj_dequant_M * sizeof(bf16);
    const size_t b_down_off = (size_t)expert_id * down_proj_dequant_M * sizeof(bf16);
    _gen_expert_seq_if_needed(this->short_seq_gemm.get(), short_seq_gate_proj, L_this_expert_padded, gate_proj_dequant_N_mem_pad, gate_proj_dequant_M_mem_pad, 0, true, 1, 0);
    _gen_expert_seq_if_needed(this->short_seq_gemm.get(), short_seq_up_proj,   L_this_expert_padded, up_proj_dequant_N_mem_pad, up_proj_dequant_M_mem_pad, 0, true, 2, 0);
    _gen_expert_seq_if_needed(this->short_seq_gemm.get(), short_seq_down_proj, L_this_expert_padded, down_proj_dequant_N_mem_pad, down_proj_dequant_M_mem_pad, 0, true, 0, 0);
    for (int i = 0; i < L_this_expert; i++) {
        int time_index = expert_to_index_map[i].first;
        memcpy(expert_buffer_x.data() + (i*D_PADDED_128), hidden_states_tensor[time_index].data(), D * sizeof(bf16)); // only copy valid D (2880)
    }
    expert_buffer_x.sync_to_device();
    // dequant end
    // first does the gate projection, then up projection
    this->gate_bias[layer_idx].sync_to_device();

    this->short_seq_gate_proj(expert_buffer_gate_result,  expert_buffer_x,  bound_arg{&this->w_up_gate_down, w_gate_off}, bound_arg{&this->gate_bias[layer_idx], b_gate_off});
    // Invalidate only the L_this_expert_padded rows this expert writes/reads,
    // not the whole max-expert-sized allocation.
    expert_buffer_gate_result.sync_from_device((size_t)L_this_expert_padded * gate_proj_dequant_M_mem_pad);

    this->up_bias[layer_idx].sync_to_device();
    this->short_seq_up_proj(expert_buffer_up_result,  expert_buffer_x,  bound_arg{&this->w_up_gate_down, w_up_off}, bound_arg{&this->up_bias[layer_idx], b_up_off});
    expert_buffer_up_result.sync_from_device((size_t)L_this_expert_padded * up_proj_dequant_M_mem_pad);

    // do a simple element-wise multiplication for GLU

    #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
    for (int i = 0; i < L_this_expert; i++) {
        _elementwise_mul(expert_buffer_x.data() + (i*D_PADDED_128), expert_buffer_up_result.data() + (i*up_proj_dequant_M_mem_pad), expert_buffer_gate_result.data() + (i*gate_proj_dequant_M_mem_pad), D);
    }

    expert_buffer_x.sync_to_device();

    this->down_bias[layer_idx].sync_to_device();

    this->short_seq_down_proj(expert_buffer_up_result, expert_buffer_x, bound_arg{&this->w_up_gate_down, w_down_off}, bound_arg{&this->down_bias[layer_idx], b_down_off});
    expert_buffer_up_result.sync_from_device((size_t)L_this_expert_padded * up_proj_dequant_M_mem_pad);
}

inline void gpt_oss_npu::Impl::_npu_large_expert_task(
    int layer_idx,
    int expert_id,
    int L_this_expert,
    std::vector<std::pair<int, bf16>>& expert_to_index_map,
    buffer<bf16>& expert_buffer_x,
    tensor_2d<bf16>& hidden_states_tensor,
    buffer<bf16>& expert_buffer_gate_result,
    buffer<bf16>& expert_buffer_up_result
){

    int L_this_expert_padded = round_up_to_multiple(L_this_expert, this->L_pad_required_for_large_MM);
    // Shared-Executable Collapse: zero the baked per-expert weight/bias offsets;
    // they are supplied at dispatch via bound_arg byte offsets below.
    const size_t estride = (size_t)(2*dequant_gate_up_mat_size_padded + dequant_down_mat_size_padded);
    const size_t w_gate_off = (size_t)expert_id * estride * sizeof(bf16);
    const size_t w_up_off   = ((size_t)expert_id * estride + dequant_gate_up_mat_size_padded) * sizeof(bf16);
    const size_t w_down_off = ((size_t)expert_id * estride + (size_t)dequant_gate_up_mat_size_padded * 2) * sizeof(bf16);
    const size_t b_gate_off = (size_t)expert_id * gate_proj_dequant_M * sizeof(bf16);
    const size_t b_up_off   = (size_t)expert_id * up_proj_dequant_M * sizeof(bf16);
    const size_t b_down_off = (size_t)expert_id * down_proj_dequant_M * sizeof(bf16);
    _gen_expert_seq_if_needed(this->gemm.get(), this->gate_proj, L_this_expert_padded, gate_proj_dequant_N_mem_pad, gate_proj_dequant_M_mem_pad, 0, true, 1, 0);
    _gen_expert_seq_if_needed(this->gemm.get(), this->up_proj,   L_this_expert_padded, up_proj_dequant_N_mem_pad, up_proj_dequant_M_mem_pad, 0, true, 2, 0);
    _gen_expert_seq_if_needed(this->gemm.get(), this->down_proj, L_this_expert_padded, down_proj_dequant_N_mem_pad, down_proj_dequant_M_mem_pad, 0, true, 0, 0);
    for (int i = 0; i < L_this_expert; i++) {
        int time_index = expert_to_index_map[i].first;
        memcpy(expert_buffer_x.data() + (i*D_PADDED_128), hidden_states_tensor[time_index].data(), D * sizeof(bf16)); // only copy valid D (2880)
    }
    expert_buffer_x.sync_to_device();

    // first does the gate projection, then up projection
    this->gate_bias[layer_idx].sync_to_device();

    this->gate_proj(expert_buffer_gate_result,  expert_buffer_x,  bound_arg{&this->w_up_gate_down, w_gate_off}, bound_arg{&this->gate_bias[layer_idx], b_gate_off});
    // Invalidate only the L_this_expert_padded rows this expert writes/reads.
    expert_buffer_gate_result.sync_from_device((size_t)L_this_expert_padded * gate_proj_dequant_M_mem_pad);

    this->up_bias[layer_idx].sync_to_device();
    this->up_proj(expert_buffer_up_result,  expert_buffer_x,  bound_arg{&this->w_up_gate_down, w_up_off}, bound_arg{&this->up_bias[layer_idx], b_up_off});
    expert_buffer_up_result.sync_from_device((size_t)L_this_expert_padded * up_proj_dequant_M_mem_pad);

    // do a simple element-wise multiplication for GLU

    #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
    for (int i = 0; i < L_this_expert; i++) {
        _elementwise_mul(expert_buffer_x.data() + (i*D_PADDED_128), expert_buffer_up_result.data() + (i*up_proj_dequant_M_mem_pad), expert_buffer_gate_result.data() + (i*gate_proj_dequant_M_mem_pad), D);
    }

    expert_buffer_x.sync_to_device();

    this->down_bias[layer_idx].sync_to_device();

    this->down_proj(expert_buffer_up_result, expert_buffer_x, bound_arg{&this->w_up_gate_down, w_down_off}, bound_arg{&this->down_bias[layer_idx], b_down_off});
    expert_buffer_up_result.sync_from_device((size_t)L_this_expert_padded * up_proj_dequant_M_mem_pad);
}

buffer<bf16> gpt_oss_npu::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(GPTOSS_DK);
    bf16* k_cache_ptr = this->kv_caches[layer_idx].data();
    idx = is_sliding_window(layer_idx) ? idx % 128 : idx;
    uint32_t offset = idx * DH * 4;
    int k47_offset_selected = is_sliding_window(layer_idx) ? sliding_K47_offset : K47_offset;
    memcpy(static_cast<bf16*>(k_cache.data()), static_cast<bf16*>(k_cache_ptr + offset), GPTOSS_DK / 2 * sizeof(bf16) );
    memcpy(static_cast<bf16*>(k_cache.data() + GPTOSS_DK / 2), static_cast<bf16*>(k_cache_ptr + offset + k47_offset_selected), GPTOSS_DK / 2 * sizeof(bf16) );

    return k_cache;
}

buffer<bf16> gpt_oss_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(GPTOSS_DV);
    bf16* v_cache_ptr = this->kv_caches[layer_idx].data();
    idx = is_sliding_window(layer_idx) ? idx % 128 : idx;
    uint32_t offset = idx * DH * 4;
    int v03_offset_selected = is_sliding_window(layer_idx) ? sliding_V03_offset : V03_offset;
    int v47_offset_selected = is_sliding_window(layer_idx) ? sliding_V47_offset : V47_offset;
    memcpy(static_cast<bf16*>(v_cache.data()), static_cast<bf16*>(v_cache_ptr + offset + v03_offset_selected), GPTOSS_DV / 2 * sizeof(bf16));
    memcpy(static_cast<bf16*>(v_cache.data() + GPTOSS_DV / 2), static_cast<bf16*>(v_cache_ptr + offset + v47_offset_selected), GPTOSS_DV / 2 * sizeof(bf16));
    return v_cache;
}

buffer<bf16> gpt_oss_npu::Impl::get_logits(buffer<bf16>& x){
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return this->logits;
}

void gpt_oss_npu::Impl::fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_effective, int layer_idx){
    tensor_2d<bf16> k_cache(buffer_k, GPTOSS_DK / 2, L_offset * 2);
    tensor_2d<bf16> v_cache(buffer_v, GPTOSS_DV / 2, L_offset * 2);

    buffer<bf16> kv_cache = this->kv_caches[layer_idx];
    kv_cache.sync_from_device();

    bf16* k_03_ptr = kv_cache.data() + this->K03_offset + L_begin * GPTOSS_DK / 2;
    bf16* k_47_ptr = kv_cache.data() + this->K47_offset + L_begin * GPTOSS_DK / 2;
    bf16* v_03_ptr = kv_cache.data() + this->V03_offset + L_begin * GPTOSS_DV / 2;
    bf16* v_47_ptr = kv_cache.data() + this->V47_offset + L_begin * GPTOSS_DV / 2;

    for (int l = 0; l < L_effective; l++){
        memcpy(k_03_ptr, k_cache[2 * l].data(), GPTOSS_DK / 2 * sizeof(bf16));
        memcpy(k_47_ptr, k_cache[2 * l + 1].data(), GPTOSS_DK / 2 * sizeof(bf16));
        memcpy(v_03_ptr, v_cache[2 * l].data(), GPTOSS_DV / 2 * sizeof(bf16));
        memcpy(v_47_ptr, v_cache[2 * l + 1].data(), GPTOSS_DV / 2 * sizeof(bf16) );

        k_03_ptr += GPTOSS_DK / 2;
        k_47_ptr += GPTOSS_DK / 2;
        v_03_ptr += GPTOSS_DV / 2;
        v_47_ptr += GPTOSS_DV / 2;
    }
}

///@brief move new kv to sliding window kv cache
void gpt_oss_npu::Impl::fill_kv_cache_sliding(buffer<bf16>& local_kv_cache, buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, int L_offset, int L_begin, int L_end, int L_effective, int sliding_l_begin, int sliding_l_end, int layer_idx){
    int in_memory = (L_begin - SLIDING_WINDOW) > 0 ? L_begin - SLIDING_WINDOW : 0;
    int l_idx = in_memory % SLIDING_WINDOW;
    int local_offset = in_memory - sliding_l_begin;

    tensor_2d<bf16> k_cache(buffer_k, DK / 2, L_offset * 2);
    tensor_2d<bf16> v_cache(buffer_v, DV / 2, L_offset * 2);
    int local_length = sliding_l_end - sliding_l_begin;

    this->kv_caches[layer_idx].sync_from_device();
    local_kv_cache.sync_from_device();
    bf16* k_03_ptr = local_kv_cache.data() + this->K03_offset + in_memory * DK / 2;
    bf16* k_47_ptr = local_kv_cache.data() + this->K47_offset + in_memory * DK / 2;
    bf16* v_03_ptr = local_kv_cache.data() + this->V03_offset + in_memory * DV / 2;
    bf16* v_47_ptr = local_kv_cache.data() + this->V47_offset + in_memory * DV / 2;

    buffer<bf16> system_k03 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_K03_offset, SLIDING_WINDOW * DK / 2);
    buffer<bf16> system_k47 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_K47_offset, SLIDING_WINDOW * DK / 2);
    buffer<bf16> system_v03 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_V03_offset, SLIDING_WINDOW * DV / 2);
    buffer<bf16> system_v47 = buffer<bf16>(this->kv_caches[layer_idx].data() + this->sliding_V47_offset, SLIDING_WINDOW * DV / 2);

    tensor_2d<bf16> tensor_system_k03(system_k03, DK / 2, 0);
    tensor_2d<bf16> tensor_system_k47(system_k47, DK / 2, 0);
    tensor_2d<bf16> tensor_system_v03(system_v03, DV / 2, 0);
    tensor_2d<bf16> tensor_system_v47(system_v47, DV / 2, 0);
    // | sliding_l_begin -> in_memory | in_memory -> L_begin | L_begin -> L_end | L_end -> sliding_l_end |
    // | Part 1                       | Part 2               | Part 3           | Part 4                 |

    // part 1: padded, no actual data exists

    // part 2: copy from system kv to local kv
    for (int i = 0; i < (L_begin - in_memory); i++){
        int absolute_idx = in_memory + i;
        int ring_idx = linear2ring(absolute_idx);

        memcpy(k_03_ptr, tensor_system_k03[ring_idx].data(), DK / 2 * sizeof(bf16));
        memcpy(k_47_ptr, tensor_system_k47[ring_idx].data(), DK / 2 * sizeof(bf16));
        memcpy(v_03_ptr, tensor_system_v03[ring_idx].data(), DV / 2 * sizeof(bf16));
        memcpy(v_47_ptr, tensor_system_v47[ring_idx].data(), DV / 2 * sizeof(bf16));

        k_03_ptr += DK / 2;
        k_47_ptr += DK / 2;
        v_03_ptr += DV / 2;
        v_47_ptr += DV / 2;
    }
    // part 3: copy new kv to local kv and system kv
    for (int i = 0; i < (L_end - L_begin); i++){
        int absolute_idx = L_begin + i;
        int ring_idx = linear2ring(absolute_idx);

        memcpy(k_03_ptr, k_cache[2 * i].data(), DK / 2 * sizeof(bf16));
        memcpy(k_47_ptr, k_cache[2 * i + 1].data(), DK / 2 * sizeof(bf16));
        memcpy(v_03_ptr, v_cache[2 * i].data(), DV / 2 * sizeof(bf16));
        memcpy(v_47_ptr, v_cache[2 * i + 1].data(), DV / 2 * sizeof(bf16));

        memcpy(tensor_system_k03[ring_idx].data(), k_cache[2 * i].data(), DK / 2 * sizeof(bf16));
        memcpy(tensor_system_k47[ring_idx].data(), k_cache[2 * i + 1].data(), DK / 2 * sizeof(bf16));
        memcpy(tensor_system_v03[ring_idx].data(), v_cache[2 * i].data(), DV / 2 * sizeof(bf16));
        memcpy(tensor_system_v47[ring_idx].data(), v_cache[2 * i + 1].data(), DV / 2 * sizeof(bf16));

        k_03_ptr += DK / 2;
        k_47_ptr += DK / 2;
        v_03_ptr += DV / 2;
        v_47_ptr += DV / 2;
    }
}
// AVX512 optimized memset for float buffers (set to zero)

void gpt_oss_npu::Impl::_rms_norm(buffer<bf16>& y, buffer<bf16>& x, buffer<bf16>& rms_weight, int length){
    int n;
    if (length == 0){
        n = rms_weight.size();
    }
    else{
        n = length;
    }

    // Hoist raw pointers so codegen is independent of buffer<T>'s wrapped
    // device-buffer type; see gemma_npu _rms_norm fix.
    const bf16* xp = x.data();
    const bf16* wp = rms_weight.data();
    bf16* yp = y.data();

    constexpr int simd_width = 16;
    __m512 sum_xx_vec = _mm512_setzero_ps();

    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xp + i));
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
    }

    int remaining = n - i;
    if (remaining > 0) {
        __mmask16 mask = remaining == simd_width ? static_cast<__mmask16>(0xFFFF)
                                                 : static_cast<__mmask16>((1u << remaining) - 1u);
        __m256i bf16_vals = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(xp + i));
        __m512 fp32_vals = bf16o_fp32_512(bf16_vals);
        sum_xx_vec = _mm512_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);
    }

    f32 sum_xx = _mm512_reduce_add_ps(sum_xx_vec) / static_cast<f32>(n);
    f32 rms_x = sqrtf(sum_xx + 1e-5f);
    __m512 rms_x_vec = _mm512_set1_ps(rms_x);

    i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m256i x_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xp + i));
        __m256i w_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + i));
        __m512 x_fp32 = bf16o_fp32_512(x_bf16);
        __m512 w_fp32 = bf16o_fp32_512(w_bf16);
        __m512 y_vec = _mm512_div_ps(_mm512_mul_ps(w_fp32, x_fp32), rms_x_vec);
        __m256i result_bf16 = f32o_bf16_512(y_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(yp + i), result_bf16);
    }

    remaining = n - i;
    if (remaining > 0) {
        __mmask16 mask = remaining == simd_width ? static_cast<__mmask16>(0xFFFF)
                                                 : static_cast<__mmask16>((1u << remaining) - 1u);
        __m256i x_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(xp + i));
        __m256i w_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(wp + i));
        __m512 x_fp32 = bf16o_fp32_512(x_bf16);
        __m512 w_fp32 = bf16o_fp32_512(w_bf16);
        __m512 y_vec = _mm512_div_ps(_mm512_mul_ps(w_fp32, x_fp32), rms_x_vec);
        __m256i result_bf16 = f32o_bf16_512(y_vec);
        _mm256_mask_storeu_epi16(reinterpret_cast<void*>(yp + i), mask, result_bf16);
    }
}

void gpt_oss_npu::Impl::_rope(buffer<bf16>& x, int idx){
    int heads = x.size() / DH;
    std::vector<f32> cos_buf(DH / 2);
    std::vector<f32> sin_buf(DH / 2);
    for (uint32_t i = 0; i < DH / 2; i++){
        // use avx2 to calculate cos and sin
        cos_buf[i] = std::cos(this->inv_freq[i] * idx) * GPTOSS_ROPE_SCALING;
        sin_buf[i] = std::sin(this->inv_freq[i] * idx) * GPTOSS_ROPE_SCALING;
    }

    for (int h = 0; h < heads; h++){
        buffer<bf16> x_left(x.data() + h * DH, DH / 2);
        buffer<bf16> x_right(x.data() + h * DH + DH / 2, DH / 2);
        int i = 0;
        constexpr int simd = 16;

        for (; i <= (DH / 2) - simd; i += simd) {
            __m256i left_bf16  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_left.data()  + i));
            __m256i right_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_right.data() + i));
            __m512  L = bf16o_fp32_512(left_bf16);
            __m512  R = bf16o_fp32_512(right_bf16);

            __m512 C = _mm512_loadu_ps(cos_buf.data() + i);
            __m512 S = _mm512_loadu_ps(sin_buf.data() + i);

            __m512 newL = _mm512_sub_ps(_mm512_mul_ps(L, C), _mm512_mul_ps(R, S));
            __m512 newR = _mm512_add_ps(_mm512_mul_ps(L, S), _mm512_mul_ps(R, C));

            __m256i newL_bf16 = f32o_bf16_512(newL);
            __m256i newR_bf16 = f32o_bf16_512(newR);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(x_left.data()  + i), newL_bf16);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(x_right.data() + i), newR_bf16);
        }

        int remaining = (DH / 2) - i;
        if (remaining > 0) {
            __mmask16 mask = remaining == simd ? static_cast<__mmask16>(0xFFFF)
                                               : static_cast<__mmask16>((1u << remaining) - 1u);
            __m256i left_bf16  = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(x_left.data()  + i));
            __m256i right_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(x_right.data() + i));
            __m512  L = bf16o_fp32_512(left_bf16);
            __m512  R = bf16o_fp32_512(right_bf16);

            __m512 C = _mm512_maskz_loadu_ps(mask, cos_buf.data() + i);
            __m512 S = _mm512_maskz_loadu_ps(mask, sin_buf.data() + i);

            __m512 newL = _mm512_sub_ps(_mm512_mul_ps(L, C), _mm512_mul_ps(R, S));
            __m512 newR = _mm512_add_ps(_mm512_mul_ps(L, S), _mm512_mul_ps(R, C));

            __m256i newL_bf16 = f32o_bf16_512(newL);
            __m256i newR_bf16 = f32o_bf16_512(newR);
            _mm256_mask_storeu_epi16(reinterpret_cast<void*>(x_left.data()  + i), mask, newL_bf16);
            _mm256_mask_storeu_epi16(reinterpret_cast<void*>(x_right.data() + i), mask, newR_bf16);
        }
    }
}

void gpt_oss_npu::Impl::_clamp_gelu(buffer<bf16>& data, float min_val, float max_val, int rows, int length, int row_size){
    if (rows <= 0 || length <= 0 || row_size <= 0) return;

    int total = static_cast<int>(data.size());
    if (total <= 0) return;

    int max_rows = (total + row_size - 1) / row_size;
    int effective_rows = std::min(rows, max_rows);

    bf16* base = data.data();
    const __m256 minv = _mm256_set1_ps(min_val);
    const __m256 maxv = _mm256_set1_ps(max_val);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c0 = _mm256_set1_ps(0.7978845608028654f); // sqrt(2/pi)
    const __m256 c1 = _mm256_set1_ps(0.044715f);

    auto apply_row = [&](bf16* ptr, int row_len) {
        int i = 0;
        for (; i + 16 <= row_len; i += 16) {
            __m128i bf16_vals0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
            __m128i bf16_vals1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i + 8));

            __m256 x0 = bf16o_fp32(bf16_vals0);
            __m256 x1 = bf16o_fp32(bf16_vals1);

            x0 = _mm256_max_ps(minv, _mm256_min_ps(maxv, x0));
            x1 = _mm256_max_ps(minv, _mm256_min_ps(maxv, x1));

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

        for (; i + 8 <= row_len; i += 8) {
            __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
            __m256 x = bf16o_fp32(bf16_vals);
            x = _mm256_max_ps(minv, _mm256_min_ps(maxv, x));

            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);
            __m256 inner = _mm256_add_ps(x, _mm256_mul_ps(c1, x3));
            __m256 t = _mm256_mul_ps(c0, inner);
            __m256 th = _fflm_fast_tanh_ps(t);
            __m256 y = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, th)));

            __m128i out_bf16 = f32o_bf16(y);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16);
        }

        for (; i < row_len; ++i) {
            float x = static_cast<float>(ptr[i]);
            if (x < min_val) x = min_val;
            if (x > max_val) x = max_val;
            float y = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            ptr[i] = bf16(y);
        }
    };

    const bool full_rows = (length <= row_size) && (rows * row_size <= total);
    if (full_rows) {
        for (int r = 0; r < rows; ++r) {
            apply_row(base + r * row_size, length);
        }
        return;
    }

    for (int r = 0; r < effective_rows; ++r) {
        int row_offset = r * row_size;
        if (row_offset >= total) break;

        int row_len = std::min(length, row_size);
        row_len = std::min(row_len, total - row_offset);

        apply_row(base + row_offset, row_len);
    }
}

void gpt_oss_npu::Impl::_clamp_inc(buffer<bf16>& data, float min_val, float max_val, int rows, int length, int row_size){
    if (rows <= 0 || length <= 0 || row_size <= 0) return;

    int total = static_cast<int>(data.size());
    if (total <= 0) return;

    int max_rows = (total + row_size - 1) / row_size;
    int effective_rows = std::min(rows, max_rows);

    bf16* base = data.data();
    const __m256 minv = _mm256_set1_ps(min_val);
    const __m256 maxv = _mm256_set1_ps(max_val);
    const __m256 one = _mm256_set1_ps(1.0f);

    auto apply_row = [&](bf16* ptr, int row_len) {
        int i = 0;
        for (; i + 16 <= row_len; i += 16) {
            __m128i bf16_vals0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
            __m128i bf16_vals1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i + 8));

            __m256 x0 = bf16o_fp32(bf16_vals0);
            __m256 x1 = bf16o_fp32(bf16_vals1);

            x0 = _mm256_max_ps(minv, _mm256_min_ps(maxv, x0));
            x1 = _mm256_max_ps(minv, _mm256_min_ps(maxv, x1));

            x0 = _mm256_add_ps(x0, one);
            x1 = _mm256_add_ps(x1, one);

            __m128i out_bf16_0 = f32o_bf16(x0);
            __m128i out_bf16_1 = f32o_bf16(x1);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16_0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i + 8), out_bf16_1);
        }

        for (; i + 8 <= row_len; i += 8) {
            __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + i));
            __m256 x = bf16o_fp32(bf16_vals);
            x = _mm256_max_ps(minv, _mm256_min_ps(maxv, x));
            x = _mm256_add_ps(x, one);

            __m128i out_bf16 = f32o_bf16(x);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + i), out_bf16);
        }

        for (; i < row_len; ++i) {
            float x = static_cast<float>(ptr[i]);
            if (x < min_val) x = min_val;
            if (x > max_val) x = max_val;
            ptr[i] = bf16(x + 1.0f);
        }
    };

    const bool full_rows = (length <= row_size) && (rows * row_size <= total);
    if (full_rows) {
        for (int r = 0; r < rows; ++r) {
            apply_row(base + r * row_size, length);
        }
        return;
    }

    for (int r = 0; r < effective_rows; ++r) {
        int row_offset = r * row_size;
        if (row_offset >= total) break;

        int row_len = std::min(length, row_size);
        row_len = std::min(row_len, total - row_offset);

        apply_row(base + row_offset, row_len);
    }
}

void gpt_oss_npu::Impl::_elementwise_mul(bf16* dest, bf16* src1, bf16* src2, int length){
    constexpr int simd_width = 16;
    int i = 0;

    // Process 16 elements at a time with AVX512
    for (; i <= length - simd_width; i += simd_width) {
        __m256i src1_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src1[i]));
        __m256i src2_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src2[i]));

        __m512 src1_fp32 = bf16o_fp32_512(src1_bf16);
        __m512 src2_fp32 = bf16o_fp32_512(src2_bf16);

        __m512 result_fp32 = _mm512_mul_ps(src1_fp32, src2_fp32);

        __m256i result_bf16 = f32o_bf16_512(result_fp32);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dest[i]), result_bf16);
    }

    // Handle remaining elements with masked operations
    int remaining = length - i;
    if (remaining > 0) {
        __mmask16 mask = remaining == simd_width ? static_cast<__mmask16>(0xFFFF)
                                                 : static_cast<__mmask16>((1u << remaining) - 1u);

        __m256i src1_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(&src1[i]));
        __m256i src2_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(&src2[i]));

        __m512 src1_fp32 = bf16o_fp32_512(src1_bf16);
        __m512 src2_fp32 = bf16o_fp32_512(src2_bf16);

        __m512 result_fp32 = _mm512_mul_ps(src1_fp32, src2_fp32);

        __m256i result_bf16 = f32o_bf16_512(result_fp32);
        _mm256_mask_storeu_epi16(reinterpret_cast<void*>(&dest[i]), mask, result_bf16);
    }
}

void gpt_oss_npu::Impl::_cpu_mm(bf16* y_out, const bf16* x, const bf16* w, const bf16* b, int L, int Din, int Dout, int y_row_size, int x_row_size, int w_row_size){
    constexpr int tile = 128;
    const int tiles_per_row = (w_row_size + tile - 1) / tile; // w_row_size is Din padded in memory
    const int tile_cols = (Din + tile - 1) / tile;
    // w is stored as 128x128 tiles (row-major within tile), tiles laid out row-major
    constexpr int simd_width = 16; // bf16 elements per 256-bit load
    #pragma omp parallel for num_threads(PREFILL_THREADS_BUDGET)
    for (int l = 0; l < L; ++l){
        const bf16* x_row = x + l * x_row_size;
        bf16* y_row = y_out + l * y_row_size;
        const int out_end = std::min(Dout, y_row_size);
        for (int o = 0; o < out_end; ++o) {

            float acc = (b != nullptr) ? (float)b[o] : 0.0f;
            const int tile_row = o / tile;
            const int row_in_tile = o % tile;

            __m512 sum_vec = _mm512_setzero_ps();

            for (int tile_col = 0; tile_col < tile_cols; ++tile_col) {
                const int col_start = tile_col * tile;
                const int col_end = std::min(col_start + tile, Din);
                const int len = col_end - col_start;

                const int block_index = tile_row * tiles_per_row + tile_col;
                const bf16* w_block = w + block_index * tile * tile;
                const bf16* w_row_in_block = w_block + row_in_tile * tile;

                int kk = 0;
                const int vec_end = len - (len % simd_width);
                for (; kk < vec_end; kk += simd_width) {
                    __m256i x_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_row + col_start + kk));
                    __m256i w_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_row_in_block + kk));
                    __m512 x_fp32 = bf16o_fp32_512(x_bf16);
                    __m512 w_fp32 = bf16o_fp32_512(w_bf16);
                    sum_vec = _mm512_fmadd_ps(x_fp32, w_fp32, sum_vec);
                }
                const int remaining = len - kk;
                if (remaining > 0) {
                    __mmask16 mask = remaining == simd_width ? static_cast<__mmask16>(0xFFFF)
                                                             : static_cast<__mmask16>((1u << remaining) - 1u);
                    __m256i x_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(x_row + col_start + kk));
                    __m256i w_bf16 = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(w_row_in_block + kk));
                    __m512 x_fp32 = bf16o_fp32_512(x_bf16);
                    __m512 w_fp32 = bf16o_fp32_512(w_bf16);
                    sum_vec = _mm512_fmadd_ps(x_fp32, w_fp32, sum_vec);
                }
            }

            acc += _mm512_reduce_add_ps(sum_vec);
            y_row[o] = (bf16)acc;
        }
        if (y_row_size > out_end) {
            std::fill(y_row + out_end, y_row + y_row_size, (bf16)0.0f);
        }
    }
}
void gpt_oss_npu::Impl::_expert_mac(buffer<float>& acc, bf16* x, bf16 score, int length){
    // acc[i] += x[i] * score;
    int n = length;

    constexpr int simd_width = 16;
    float score_fp32 = (float)score;
    __m512 score_vec = _mm512_set1_ps(score_fp32);

    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m256i bf16_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        __m512 x_fp32 = bf16o_fp32_512(bf16_vals);
        __m512 acc_vec = _mm512_loadu_ps(&acc[i]);
        __m512 result_vec = _mm512_fmadd_ps(x_fp32, score_vec, acc_vec);
        _mm512_storeu_ps(&acc[i], result_vec);
    }

    int remaining = n - i;
    if (remaining > 0) {
        __mmask16 mask = remaining == simd_width ? static_cast<__mmask16>(0xFFFF)
                                                 : static_cast<__mmask16>((1u << remaining) - 1u);
        __m256i bf16_vals = _mm256_maskz_loadu_epi16(mask, reinterpret_cast<const void*>(x + i));
        __m512 x_fp32 = bf16o_fp32_512(bf16_vals);
        __m512 acc_vec = _mm512_maskz_loadu_ps(mask, &acc[i]);
        __m512 result_vec = _mm512_fmadd_ps(x_fp32, score_vec, acc_vec);
        _mm512_mask_storeu_ps(&acc[i], mask, result_vec);
    }
}

void gpt_oss_npu::Impl::_export_to_bf16(buffer<bf16>& dest, buffer<float>& src, int length){
    int n;
    if (length == 0){
        n = src.size();
    }
    else{
        n = length;
    }

    int i = 0;
    constexpr int simd_width_512 = 16;
    for (; i <= n - simd_width_512; i += simd_width_512) {
        __m512 fp32_vals = _mm512_loadu_ps(&src[i]);
        __m256i bf16_vals = f32o_bf16_512(fp32_vals);
        _mm256_storeu_si256((__m256i*)&dest[i], bf16_vals);
    }
    for (; i < n; i++) {
        dest[i] = bf16(src[i]);
    }
}

void gpt_oss_npu::Impl::_residual_add(buffer<bf16>& dest, buffer<bf16>& src, buffer<bf16>& residual, int length){
    int n;
    if (length == 0){
        n = src.size();
    }
    else{
        n = length;
    }

    int i = 0;
    constexpr int simd_width_512 = 16;
    for (; i <= n - simd_width_512; i += simd_width_512){
        __m256i bf16_vals = _mm256_loadu_si256((__m256i*)&src[i]);
        __m512i expanded = _mm512_cvtepu16_epi32(bf16_vals);
        __m512 fp32_vals = _mm512_castsi512_ps(_mm512_slli_epi32(expanded, 16));

        __m256i bf16_vals_r = _mm256_loadu_si256((__m256i*)&residual[i]);
        __m512i expanded_r = _mm512_cvtepu16_epi32(bf16_vals_r);
        __m512 fp32_vals_r = _mm512_castsi512_ps(_mm512_slli_epi32(expanded_r, 16));

        __m512 result_vec = _mm512_add_ps(fp32_vals, fp32_vals_r);
        __m256i result_bf16 = f32o_bf16_512(result_vec);
        _mm256_storeu_si256((__m256i*)&dest[i], result_bf16);
    }

    for (; i < n; i++) {
        dest[i] = bf16((float)src[i] + (float)residual[i]);
    }
}

void gpt_oss_npu::Impl::clear_context(){
    this->set_context_length(0);
    for (int i = 0; i < num_hidden_layers; i++){
        this->kv_caches[i].memset((bf16)0.0f);
        this->kv_caches[i].sync_to_device();
    }
    this->dirty_rows = 0;
}

void gpt_oss_npu::Impl::_zero_temp_cache(){
    if (!this->temp_dirty) return;
    this->sliding_temp_cache.memset((bf16)0.0f);
    this->sliding_temp_cache.sync_to_device();
    this->temp_dirty = false;
}

kv_row_layout gpt_oss_npu::Impl::kv_layout() const{
    kv_row_layout l;
    l.DK = GPTOSS_DK; l.DV = GPTOSS_DV; l.striped = true;
    l.k47 = K47_offset; l.v03 = V03_offset; l.v47 = V47_offset;
    return l;
}

std::unique_ptr<hybrid_engine_state> gpt_oss_npu::Impl::new_state(){
    size_t ring_bytes = 0, global_layers = 0;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) global_layers++;
        else ring_bytes += kv_caches[i].size() * sizeof(bf16);
    }
    return kv_rows_new_state(global_layers, kv_layout(), ring_bytes);
}

int gpt_oss_npu::Impl::read_state(hybrid_engine_state& state, int kv_from){
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    const int length = kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from, -1, &attn_mask);
    if (length < 0) return -1;
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        state.state[i] = buffer<bf16>(this->kv_caches[i].size());
        memcpy(static_cast<void*>(state.state[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
    }
    return length;
}

int gpt_oss_npu::Impl::write_state_kv(const hybrid_engine_state& state){
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask);
}

int gpt_oss_npu::Impl::write_state(const hybrid_engine_state& state){
    // check the rings before anything is written
    bool ok = state.state.size() == this->kv_caches.size();
    for (size_t i = 0; ok && i < kv_caches.size(); i++){
        if (!attn_mask[i] && state.state[i].size() != this->kv_caches[i].size()) ok = false;
    }
    const int length = ok ? kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L, &attn_mask) : -1;
    if (length < 0){
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    for (size_t i = 0; i < kv_caches.size(); i++){
        if (attn_mask[i]) continue;
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<const void*>(state.state[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(this->kv_caches, kv_layout(), static_cast<size_t>(length), this->dirty_rows, &attn_mask);
    this->dirty_rows = static_cast<size_t>(length);
    this->set_context_length(length);
    return length;
}

int gpt_oss_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

void gpt_oss_npu::Impl::_allocate_checkpoint_buffers(){
    this->kv_checkpoint.clear();
    this->kv_checkpoint.resize(num_hidden_layers);
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_checkpoint[i] = this->layer.create_bo_buffer<bf16>(this->kv_caches[i].size());
        }
    }
}

int gpt_oss_npu::Impl::checkpoint(){
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    if (!is_checkpoint_valid) {
        _allocate_checkpoint_buffers();
    } // lazy allocation of checkpoint buffers, only allocate when checkpoint is called for the first time
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(static_cast<void*>(this->kv_checkpoint[i].data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return checkpoint_context_length;
}

int gpt_oss_npu::Impl::restore(){
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    for (int i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_caches[i].sync_from_device();
            memcpy(static_cast<void*>(this->kv_caches[i].data()), static_cast<void*>(this->kv_checkpoint[i].data()), this->kv_caches[i].size() * sizeof(bf16));
            this->kv_caches[i].sync_to_device();
        }
    }
    this->set_context_length(checkpoint_context_length); // regenerate sequence
    return checkpoint_context_length;
}

void gpt_oss_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L){
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
	}
    this->MAX_L = std::max((int)MAX_L, 4096);
    if (this->MAX_L > 131072){
        this->MAX_L = 131072;
    }
    MAX_L = this->MAX_L;

    // make MAX_L a power of 2
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L)){
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L
    size_t kv_cache_size = MAX_L * DK * 2;
    this->K03_offset = 0;
    this->K47_offset = 1 * kv_cache_size / 4;
    this->V03_offset = 2 * kv_cache_size / 4;
    this->V47_offset = 3 * kv_cache_size / 4;
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    this->sequence->set_max_length(MAX_L);
    for (uint32_t i = 0; i < num_hidden_layers; i++){
        if (is_sliding_window(i)){
            this->kv_caches[i].memset((bf16)0.0f);
            this->kv_caches[i].sync_to_device();
        }
        else{
            this->kv_caches[i] = this->layer.create_bo_buffer<bf16>(kv_cache_size);
            this->kv_caches[i].memset((bf16)0.0f);
            this->kv_caches[i].sync_to_device();
        }
    }
    this->sliding_temp_cache = this->layer.create_bo_buffer<bf16>(kv_cache_size);
    this->sliding_temp_cache.memset((bf16)0.0f);
    this->sliding_temp_cache.sync_to_device();
    this->temp_dirty = false;
    this->dirty_rows = 0; // everything was zeroed
    this->set_context_length(0); // regenerate sequence
    this->is_checkpoint_valid = false; // invalidate checkpoint since the underlying buffer has changed
}
///@brief destructor of gpt_oss_npu
gpt_oss_npu::Impl::~Impl() = default;
/*
gpt_oss_npu::Impl::~Impl(){
    for (int i = 0; i < num_hidden_layers; i++){
        this->kv_caches[i].free();
        this->proj_weights[i].free();
        this->rms_weights[i].free();
    }
    this->kv_caches.clear();
    this->rms_weights.clear();
    this->proj_weights.clear();
    this->rope_weights.release();
    this->weights_p0.release();
    this->weights_p1.release();
    this->x.release();
    this->lm_head.reset();
}
*/
// ===============================================
// Externals for gpt_oss_npu
// ===============================================
gpt_oss_npu::gpt_oss_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> gpt_oss_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> gpt_oss_npu::forward(buffer<bf16>& x){
    return this->_impl->forward(x);
}

buffer<bf16> gpt_oss_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void gpt_oss_npu::set_context_length(int L){
    this->_impl->set_context_length(L);
}

void gpt_oss_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void gpt_oss_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void gpt_oss_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> gpt_oss_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> gpt_oss_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int gpt_oss_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int gpt_oss_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int gpt_oss_npu::restore(){
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> gpt_oss_npu::new_state(){
    return this->_impl->new_state();
}

int gpt_oss_npu::read_state(hybrid_engine_state& state, int kv_from){
    return this->_impl->read_state(state, kv_from);
}

int gpt_oss_npu::write_state(const hybrid_engine_state& state){
    return this->_impl->write_state(state);
}

int gpt_oss_npu::write_state_kv(const hybrid_engine_state& state){
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* gpt_oss_npu_state_api(void* engine){
    return hybrid_state_api<gpt_oss_npu>(engine);
}

gpt_oss_npu::~gpt_oss_npu(){
    delete this->_impl;
}
