#include "qwen3_5vl_image.hpp"

#include <thread>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif
#include "utils/debug_utils.hpp"
#include "utils/error_measure.hpp"
#include "vision/pos_embed_interpolate.hpp"
#include "vision/vision_prefill_helper.hpp"
#include "vision/rot_pos_emb.hpp"
#include "vision/norm.hpp"
#include "vision_timer.hpp"
#include "vision_overlay.hpp"
#include <fstream>
#include "npu_sequences/mmRuntimeSequence.hpp"
#include "seq_gen.hpp"

// #define DEBUG_PRINT_ENCODE_TIME_DETAIL 1
// #define DEBUG_PRINT_ENCODE_ERROR_METRICS 1

Qwen3_5VLImageEncoder::~Qwen3_5VLImageEncoder()
{
}

Qwen3_5VLImageEncoder::Qwen3_5VLImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance, qwen3_5vl_npu* parent_npu_ptr)
    : config(config), npu(npu_instance), model_path(config.model_path), parent_npu_ptr(parent_npu_ptr)
{
    // load parameters from json file

    {
        QWEN3_5_VISION_EMBED_DIM = config.sub("vision_config").value("QWEN3_5_VISION_EMBED_DIM", -1);
        QWEN3_5_VISION_NUM_HEADS = config.sub("vision_config").value("QWEN3_5_VISION_NUM_HEADS", -1);
        QWEN3_5_VISION_HEAD_DIM =config.sub("vision_config").value("QWEN3_5_VISION_HEAD_DIM", -1);
        _QWEN3_5_VISION_HIDDEN_SIZE = QWEN3_5_VISION_EMBED_DIM;
        _QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE = config.sub("vision_config").value("QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE", -1);
        QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS = config.sub("vision_config").value("QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS", -1);
        QWEN3_5_VISION_NUM_LAYERS = config.sub("vision_config").value("QWEN3_5_VISION_NUM_LAYERS", -1);
        QWEN3_5_VISION_LAYER_NORM_EPSILON = config.sub("vision_config").value("QWEN3_5_VISION_LAYER_NORM_EPSILON", -1.0f);
        _QWEN3_5_MERGER_HIDDEN_SIZE = _QWEN3_5_VISION_HIDDEN_SIZE * (this->parent_npu_ptr->QWEN3_5_SPATIAL_MERGE_SIZE*this->parent_npu_ptr->QWEN3_5_SPATIAL_MERGE_SIZE);
        _QWEN3_5_VISION_OUT_HIDDEN_SIZE = config.sub("vision_config").value("QWEN3_5_VISION_OUT_HIDDEN_SIZE", -1);

        MM_tile_M = config.sub("vision_config").value("VISION_MM_TILE_M", -1);
        MM_tile_K = config.sub("vision_config").value("VISION_MM_TILE_K", -1);
        MM_tile_N = config.sub("vision_config").value("VISION_MM_TILE_N", -1);

        seq_len_pad_requirement_for_MM = MM_ROW_SIZE*MM_tile_M;
        assert( MM_tile_K % MM_tile_N == 0); // we need this for the way we generate the sequence, because k and N is interchangable at MLP (gate-down)
    }

    Padded_QWEN3_5_VISION_HIDDEN_SIZE = round_up_to_multiple(_QWEN3_5_VISION_HIDDEN_SIZE, MM_tile_K);
    Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE = round_up_to_multiple(_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE, MM_tile_K);
    Padded_QWEN3_5_MERGER_HIDDEN_SIZE = round_up_to_multiple(_QWEN3_5_MERGER_HIDDEN_SIZE, MM_tile_K);
    Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE = round_up_to_multiple(_QWEN3_5_VISION_OUT_HIDDEN_SIZE, MM_tile_K);

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // print all the parameters for debug
        std::cout << "QWEN3_5_VISION_EMBED_DIM: " << QWEN3_5_VISION_EMBED_DIM << std::endl;
        std::cout << "QWEN3_5_VISION_NUM_HEADS: " << QWEN3_5_VISION_NUM_HEADS << std::endl;
        std::cout << "QWEN3_5_VISION_HEAD_DIM: " <<  QWEN3_5_VISION_HEAD_DIM << std::endl;
        std::cout << "QWEN3_5_VISION_HIDDEN_SIZE: " << _QWEN3_5_VISION_HIDDEN_SIZE << std::endl;
        std::cout << "QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE: " << _QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE << std::endl;
        std::cout << "QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS: " << QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS << std::endl;
        std::cout << "QWEN3_5_VISION_NUM_LAYERS: " << QWEN3_5_VISION_NUM_LAYERS << std::endl;
        std::cout << "QWEN3_5_VISION_LAYER_NORM_EPSILON: " << QWEN3_5_VISION_LAYER_NORM_EPSILON << std::endl;
        std::cout << "QWEN3_5_MERGER_HIDDEN_SIZE: " << _QWEN3_5_MERGER_HIDDEN_SIZE << std::endl;
        std::cout << "QWEN3_5_VISION_OUT_HIDDEN_SIZE: " << _QWEN3_5_VISION_OUT_HIDDEN_SIZE << std::endl;

        // print the padded parameters for debug
        std::cout << "Padded_QWEN3_5_VISION_HIDDEN_SIZE: " << Padded_QWEN3_5_VISION_HIDDEN_SIZE << std::endl;
        std::cout << "Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE: " << Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE << std::endl;
        std::cout << "Padded_QWEN3_5_MERGER_HIDDEN_SIZE: " << Padded_QWEN3_5_MERGER_HIDDEN_SIZE << std::endl;
        std::cout << "Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE: " << Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE << std::endl;

    #endif

    const char* overlay_xclbin = std::getenv("VISION_OVERLAY");
    this->use_overlay = overlay_xclbin != nullptr && overlay_xclbin[0] != '\0';
    if (this->use_overlay) {
        this->proj = npu->register_xclbin(overlay_xclbin);
        this->fla = this->proj;
    } else {
        this->fla = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_attn.xclbin"));
        this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));
    }

    this->flash_attention_app = this->fla->create_app();
    this->qkv_projection_app = this->proj->create_app();
    this->qkv_first_projection_app = this->proj->create_app();
    this->o_projection_app = this->proj->create_app();
    this->mlp_fc1_app = this->proj->create_app();
    this->mlp_fc2_app = this->proj->create_app();
    this->merger_fc1_app = this->proj->create_app();
    this->merger_fc2_app = this->proj->create_app();

    // attempting to read the info that vision model needs from config.sub("vision_config")

    // resize the vectors to hold all layers weights
    norm_1_weight.resize(QWEN3_5_VISION_NUM_LAYERS);
    norm_1_bias.resize(QWEN3_5_VISION_NUM_LAYERS);
    norm_2_weight.resize(QWEN3_5_VISION_NUM_LAYERS);
    norm_2_bias.resize(QWEN3_5_VISION_NUM_LAYERS);
    attn_qkv_weight.resize(QWEN3_5_VISION_NUM_LAYERS);
    o_proj_weight.resize(QWEN3_5_VISION_NUM_LAYERS);
    mlp_fc1_weight.resize(QWEN3_5_VISION_NUM_LAYERS);
    mlp_fc2_weight.resize(QWEN3_5_VISION_NUM_LAYERS);
}
void Qwen3_5VLImageEncoder::init_weights(SafeTensors &q4nx){
    // std::cout << "Loading Qwen3VL Image Encoder weights from " << this->config.get<std::string>("vision_model_weight", "") << " ..." <<std::endl;
    q4nx.load_weights(patch_embed_proj_weight, "model.visual.patch_embed.proj.weight");
    q4nx.load_weights(patch_embed_proj_bias,"model.visual.patch_embed.proj.bias");
    q4nx.load_weights(pos_embed_weight,  "model.visual.pos_embed.weight");

    //load each layer weights
    for(int layer_idx = 0; layer_idx < QWEN3_5_VISION_NUM_LAYERS; layer_idx++){
        q4nx.load_weights(
            this->norm_1_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm1.weight"
        );
        q4nx.load_weights(
            this->norm_1_bias[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm1.bias"
        );
        q4nx.load_weights(
            this->norm_2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm2.weight"
        );
        q4nx.load_weights(
            this->norm_2_bias[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".norm2.bias"
        );

        this->attn_qkv_weight[layer_idx] = qkv_projection_app.create_bo_buffer<bf16>(Padded_QWEN3_5_VISION_HIDDEN_SIZE*(3*Padded_QWEN3_5_VISION_HIDDEN_SIZE) + 3*Padded_QWEN3_5_VISION_HIDDEN_SIZE);
        this->o_proj_weight[layer_idx] = o_projection_app.create_bo_buffer<bf16>(Padded_QWEN3_5_VISION_HIDDEN_SIZE*(Padded_QWEN3_5_VISION_HIDDEN_SIZE) + Padded_QWEN3_5_VISION_HIDDEN_SIZE );

        q4nx.load_weights(this->attn_qkv_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.qkv.bias",
            0
        );
        q4nx.load_weights(this->o_proj_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.proj.bias",
            0
        );

        q4nx.load_weights(this->attn_qkv_weight[layer_idx],
             "model.visual.blocks." + std::to_string(layer_idx) + ".attn.qkv.weight",
             sizeof(bf16) * 3*Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->o_proj_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".attn.proj.weight",
            sizeof(bf16) * Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );
        this->attn_qkv_weight[layer_idx].sync_to_device();
        this->o_proj_weight[layer_idx].sync_to_device();

        // mlo
        this->mlp_fc1_weight[layer_idx] = mlp_fc1_app.create_bo_buffer<bf16>(Padded_QWEN3_5_VISION_HIDDEN_SIZE*Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE
             + Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE);
        this->mlp_fc2_weight[layer_idx] = mlp_fc2_app.create_bo_buffer<bf16>(Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE*Padded_QWEN3_5_VISION_HIDDEN_SIZE
            + Padded_QWEN3_5_VISION_HIDDEN_SIZE);

        q4nx.load_weights(this->mlp_fc1_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc1.bias",
            0
        );
        q4nx.load_weights(this->mlp_fc2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc2.bias",
            0
        );

        q4nx.load_weights(this->mlp_fc1_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc1.weight",
            sizeof(bf16) * Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE
        );
        q4nx.load_weights(this->mlp_fc2_weight[layer_idx],
            "model.visual.blocks." + std::to_string(layer_idx) + ".mlp.linear_fc2.weight",
            sizeof(bf16) * Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );
        this->mlp_fc1_weight[layer_idx].sync_to_device();
        this->mlp_fc2_weight[layer_idx].sync_to_device();
    }
    q4nx.load_weights(
        this->merger_norm_weight,
        "model.visual.merger.norm.weight"
    );
    q4nx.load_weights(
        this->merger_norm_bias,
        "model.visual.merger.norm.bias"
    );

    this->merger_fc1_weight = merger_fc1_app.create_bo_buffer<bf16>(
        Padded_QWEN3_5_MERGER_HIDDEN_SIZE*Padded_QWEN3_5_MERGER_HIDDEN_SIZE+Padded_QWEN3_5_MERGER_HIDDEN_SIZE

    );
    this->merger_fc2_weight = merger_fc2_app.create_bo_buffer<bf16>(
        Padded_QWEN3_5_MERGER_HIDDEN_SIZE* Padded_QWEN3_5_MERGER_HIDDEN_SIZE + Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE
    );
    q4nx.load_weights(this->merger_fc1_weight,
        "model.visual.merger.linear_fc1.bias",
        0
    );
    q4nx.load_weights(this->merger_fc2_weight,
        "model.visual.merger.linear_fc2.bias",
        0
    );
    q4nx.load_weights(this->merger_fc1_weight,
        "model.visual.merger.linear_fc1.weight",
        sizeof(bf16) * Padded_QWEN3_5_MERGER_HIDDEN_SIZE
    );
    q4nx.load_weights(this->merger_fc2_weight,
        "model.visual.merger.linear_fc2.weight",
        sizeof(bf16) * Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE
    );
}


static void dump_stage(const char* tag, int layer_idx, const bf16* data, size_t count) {
    const char* prefix = std::getenv("VISION_DUMP_STAGES");
    if (prefix == nullptr) {
        return;
    }
    const char* all_layers = std::getenv("VISION_DUMP_ALL_LAYERS");
    const char* one_layer = std::getenv("VISION_DUMP_LAYER");
    const bool chosen = one_layer != nullptr && std::atoi(one_layer) == layer_idx;
    if (layer_idx > 1 && !chosen && !(all_layers != nullptr && std::string(tag) == "fc2")) {
        return;
    }
    std::ofstream out(std::string(prefix) + "_" + tag + "_" + std::to_string(layer_idx) + ".bin", std::ios::binary);
    out.write((const char*)data, count * sizeof(bf16));
}

std::vector<std::vector<bf16>> Qwen3_5VLImageEncoder::encode( void* image_payload_ptr)
{
    //
    //DEBUG
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
    SafeTensors reference_tensor(this->model_path + "/vision_data_prefill_9b.safetensors");
    std::cout << "debug Qwen3_5VLImageEncoder::encode called with image_payload_ptr: " << image_payload_ptr << std::endl;
    #endif
    auto encoder_start_time = std::chrono::high_resolution_clock::now();
    vision_timer_t vt;
    vt.start();
    const bool vt_probe = vision_timer_t::probe_enabled();

    qwen3_5vl_image_payload_t* image_payload = (qwen3_5vl_image_payload_t*)image_payload_ptr;
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

    int seq_len_padded = round_up_to_multiple(seq_len_of_last_image, vision_L_padded_requirement_for_attention) - seq_len_of_last_image;
    seq_len_padded += seq_len;
    seq_len_padded = round_up_to_multiple(seq_len_padded, seq_len_pad_requirement_for_MM);

    int merger_seq_len = (seq_len*_QWEN3_5_VISION_HIDDEN_SIZE) / (_QWEN3_5_MERGER_HIDDEN_SIZE); //NOTE: use the unpadded version for calculation
    int merger_seq_len_padded = round_up_to_multiple(merger_seq_len, seq_len_pad_requirement_for_MM);

    #if DEBUG_PRINT_ENCODE_TIME_DETAIL
        std::cout <<  "Total seq len: " << seq_len << std::endl;
        std::cout <<  "Total seq len padded: " << seq_len_padded << std::endl;
    #endif
    hidden_state.resize(seq_len_padded * Padded_QWEN3_5_VISION_HIDDEN_SIZE);
    memset(hidden_state.data(), 0, hidden_state.size() * sizeof(bf16)); // clear the buffer to avoid reading uninitialized data in case of partial tiles

    {
        bool ADD_BIAS= true;

        // pre-generate the mm runtime sequence
        generate_mm_sequence<bf16, bf16>(*this->qkv_projection_app.seq(),
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE,    3*Padded_QWEN3_5_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                ENABLE_QKV_REORDER, QWEN3_5_VISION_HEAD_DIM
        );

        generate_mm_sequence<bf16, bf16>(*this->o_projection_app.seq(),
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE,  Padded_QWEN3_5_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );
        generate_mm_sequence<bf16, bf16>(*this->mlp_fc1_app.seq(),
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE,  Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 1,
                false,0
        );

        generate_mm_sequence<bf16, bf16>(*this->mlp_fc2_app.seq(),
                seq_len_padded,  Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE,  Padded_QWEN3_5_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );

        generate_mm_sequence<bf16, bf16>(*this->merger_fc1_app.seq(),
                merger_seq_len_padded, Padded_QWEN3_5_MERGER_HIDDEN_SIZE,  Padded_QWEN3_5_MERGER_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 1, // gelu
                false,0
        );
        generate_mm_sequence<bf16, bf16>(*this->merger_fc2_app.seq(),
                merger_seq_len_padded, Padded_QWEN3_5_MERGER_HIDDEN_SIZE,  Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true, // set to false for all time
                ADD_BIAS, 0,
                false,0
        );

        gen_mha_vision_attention(
            this->flash_attention_app.seq(),
            image_grid_thw,
            vision_L_padded_requirement_for_attention,
            vision_S_padded_requirement_for_attention,
            vision_num_of_columns,
            vision_num_of_rows,
            vision_CU_mode,
            vision_LQ_per_CT,
            vision_LK_per_CT,
            vision_LQ_internal,
            vision_LK_internal,
            ENABLE_QKV_REORDER,

            this
        );

        if (this->use_overlay) {
            using namespace vision_overlay;
            npu_sequence prefix(device_npu2);
            auto [qkv_words, qkv_len] = this->qkv_projection_app.seq()->dump();
            this->qkv_first_projection_app.seq()->from_vector(std::vector<uint32_t>(qkv_words, qkv_words + qkv_len));
            gen_prefix(prefix, OP_MM, OP_NONE);
            prepend(*this->qkv_first_projection_app.seq(), prefix);
            gen_prefix(prefix, OP_ATTN, OP_MM);
            prepend(*this->flash_attention_app.seq(), prefix);
            gen_prefix(prefix, OP_MM, OP_ATTN);
            prepend(*this->o_projection_app.seq(), prefix);
        }
    }

    {
        // initialize the remainig buffer
        mm_qkv_input = qkv_projection_app.create_bo_buffer<bf16>(seq_len_padded * Padded_QWEN3_5_VISION_HIDDEN_SIZE);
        mm_qkv_output_mha_input = qkv_projection_app.create_bo_buffer<bf16>(seq_len_padded*Padded_QWEN3_5_VISION_HIDDEN_SIZE*3);
        mha_qkv_output_mm_o_input = o_projection_app.create_bo_buffer<bf16>(seq_len_padded * Padded_QWEN3_5_VISION_HIDDEN_SIZE);
        mm_o_output_fc1_proj_input = o_projection_app.create_bo_buffer<bf16>(seq_len_padded * Padded_QWEN3_5_VISION_HIDDEN_SIZE);
        mm_fc1_output_fc2_input = mlp_fc1_app.create_bo_buffer<bf16>(seq_len_padded * Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE);
        mm_fc2_output = mlp_fc2_app.create_bo_buffer<bf16>(seq_len_padded * Padded_QWEN3_5_VISION_HIDDEN_SIZE);

        merger_fc1_input = merger_fc1_app.create_bo_buffer<bf16>(merger_seq_len_padded*Padded_QWEN3_5_MERGER_HIDDEN_SIZE);
        merger_fc1_output_fc2_input = merger_fc1_app.create_bo_buffer<bf16>( merger_seq_len_padded*Padded_QWEN3_5_MERGER_HIDDEN_SIZE);
        merger_fc2_output = merger_fc2_app.create_bo_buffer<bf16>( merger_seq_len_padded*Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE);

        memset(mm_qkv_input.data(), 0, mm_qkv_input.size() * sizeof(bf16));
        mm_qkv_input.sync_to_device();
        memset(mm_o_output_fc1_proj_input.data(), 0, mm_o_output_fc1_proj_input.size() * sizeof(bf16));
        mm_o_output_fc1_proj_input.sync_to_device();
        memset(mm_fc1_output_fc2_input.data(), 0, mm_fc1_output_fc2_input.size() * sizeof(bf16));
        mm_fc1_output_fc2_input.sync_to_device();
        memset(mm_fc2_output.data(), 0, mm_fc2_output.size() * sizeof(bf16));
        mm_fc2_output.sync_to_device();

        memset(mm_qkv_output_mha_input.data(), 0, mm_qkv_output_mha_input.size() * sizeof(bf16));
        mm_qkv_output_mha_input.sync_to_device();
        memset(mha_qkv_output_mm_o_input.data(), 0, mha_qkv_output_mm_o_input.size() * sizeof(bf16));
        mha_qkv_output_mm_o_input.sync_to_device();
    }

    DEBUG_BLOCK(
        1,
        std::cout << "seq_len: " << seq_len << std::endl;
        std::cout << "seq_len_padded: " << seq_len_padded << std::endl;
        std::cout << "merger_seq_len: " << merger_seq_len << std::endl;
        std::cout << "merger_seq_len_padded: " << merger_seq_len_padded << std::endl;
    )

    auto time_start_patch_embed = std::chrono::high_resolution_clock::now();
    std::vector<bf16> patch_embed = conv3d_patch_embed_avx512_max_performance(
        image_payload->_data__processed.data(),
        patch_embed_proj_weight.data(),
        patch_embed_proj_bias.data(),
        seq_len,
        3, // r, g, b
        QWEN3_5_VISION_EMBED_DIM,
        this->parent_npu_ptr->QWEN3_5_TEMPORAL_PATCH_SIZE,
        this->parent_npu_ptr->QWEN3_5_PATCH_SIZE
    );
    auto time_end_patch_embed = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_patch_embed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_patch_embed - time_start_patch_embed).count();
        std::cout << "Patch embedding time: " << duration_patch_embed << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error after patch embedding comparison:" << std::endl;
        buffer<bf16> after_patch_embed;
        reference_tensor.load_weights(
            after_patch_embed,
            "hidden_states_after_patch_embed"
        );
        print_error_metrics<bf16, bf16>(
            patch_embed.data(), after_patch_embed.data(),
                1,
                seq_len, QWEN3_5_VISION_EMBED_DIM,
                seq_len, QWEN3_5_VISION_EMBED_DIM
        );

    #endif

    auto time_start_pos_emb = std::chrono::high_resolution_clock::now();
    std::vector<bf16>  pos_embeds =  fast_pos_embed_interpolate(
        image_grid_thw,
        pos_embed_weight.data(),
        std::sqrt(QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS),
        QWEN3_5_VISION_EMBED_DIM,
        this->parent_npu_ptr->QWEN3_5_SPATIAL_MERGE_SIZE
    );
    auto time_end_pos_emb = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_pos_emb = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_pos_emb - time_start_pos_emb).count();
        std::cout << "Position embedding time: " << duration_pos_emb << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        std::cout << "Error after pos embed comparison:" << std::endl;
        buffer<bf16> after_pos_embed;
        reference_tensor.load_weights(
            after_pos_embed,
            "pos_embeds"
        );
        print_error_metrics<bf16, bf16>(
            pos_embeds.data(), after_pos_embed.data(),
            1,
            seq_len, QWEN3_5_VISION_EMBED_DIM,
            seq_len, QWEN3_5_VISION_EMBED_DIM
        );
    #endif

    auto time_start_pose_embed_plus_patch_embed = std::chrono::high_resolution_clock::now();
    // simd_add(
    //     patch_embed.data(), pos_embeds.data(),
    //     hidden_state.data(), seq_len *QWEN3_5_VISION_HIDDEN_SIZE
    // );
    //note: because patch_embed is seq_len * _QWEN3_5_VISION_EMBED_DIM, and hidden_state is seq_len * Padded_QWEN3_5_VISION_HIDDEN_SIZE
    assert(QWEN3_5_VISION_EMBED_DIM  == _QWEN3_5_VISION_HIDDEN_SIZE);
    for(int ll = 0; ll < seq_len; ll++){
        simd_add(
            patch_embed.data() + ll * QWEN3_5_VISION_EMBED_DIM,
            pos_embeds.data() + ll * QWEN3_5_VISION_EMBED_DIM,
            hidden_state.data() + ll * Padded_QWEN3_5_VISION_HIDDEN_SIZE,
            QWEN3_5_VISION_EMBED_DIM
        );
    }

    auto time_end_pose_embed_plus_patch_embed = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_pos_embed_plus_patch_embed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_pose_embed_plus_patch_embed - time_start_pose_embed_plus_patch_embed).count();
        std::cout << "Position embedding plus patch embedding time: " << duration_pos_embed_plus_patch_embed << " ms" << std::endl;
    #endif

    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        std::cout << "Error after hidden_states_plus_pos_embeds:" << std::endl;
        buffer<bf16> after_pos_embed_plus_patch_embed;
        reference_tensor.load_weights(
            after_pos_embed_plus_patch_embed,
            "hidden_states_after_adding_pos_emb"
        );
        print_error_metrics<bf16, bf16>(
            hidden_state.data(), after_pos_embed_plus_patch_embed.data(),
            1,
            seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
            seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );
    #endif

    auto time_Start_rotary_pos_emb = std::chrono::high_resolution_clock::now();
    std::vector<float> rotary_pos_emb = rot_pos_emb_float(
        image_grid_thw,
        this->parent_npu_ptr->QWEN3_5_SPATIAL_MERGE_SIZE,           // spatial_merge_size = 2
        QWEN3_5_VISION_HEAD_DIM / 2           // rot_dim = 64 / 2 = 32
    );
    auto time_end_rotary_pos_emb = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_rotary_pos_emb = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_rotary_pos_emb - time_Start_rotary_pos_emb).count();
        std::cout << "Rotary position embedding time: " << duration_rotary_pos_emb << " ms" << std::endl;
    #endif

    auto time_start_cos_sine_pose_emb = std::chrono::high_resolution_clock::now();
    cos_sine_pose_emb(rotary_pos_emb, cos_emb, sin_emb, QWEN3_5_VISION_HEAD_DIM);
    auto time_end_cos_sine_pose_emb = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto duration_cos_sine_pose_emb = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_cos_sine_pose_emb - time_start_cos_sine_pose_emb).count();
        std::cout << "Cosine and Sine embedding computation time: " << duration_cos_sine_pose_emb << " ms" << std::endl;
    #endif
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        buffer<float> position_embeddings_cos_ptr;
        reference_tensor.load_weights(
            position_embeddings_cos_ptr,
            "position_embeddings_cos"
        );
        buffer<float> position_embeddings_sin_ptr;
        reference_tensor.load_weights(
            position_embeddings_sin_ptr,
            "position_embeddings_sin"
        );
        std::cout << "Error after position_embeddings_cos comparison:" << std::endl;
        print_error_metrics<float, float>(
            cos_emb.data(), position_embeddings_cos_ptr.data(),
            1,
            seq_len, QWEN3_5_VISION_HEAD_DIM,
            seq_len, QWEN3_5_VISION_HEAD_DIM
        );

        std::cout << "Error after position_embeddings_sin comparison:" << std::endl;
        print_error_metrics<float, float>(
            sin_emb.data(), position_embeddings_sin_ptr.data(),
            1,
            seq_len, QWEN3_5_VISION_HEAD_DIM,
            seq_len, QWEN3_5_VISION_HEAD_DIM
        );

    #endif
    residual_buffer.resize(hidden_state.size());
    // set residual buffer to 0
    memset(residual_buffer.data(), 0, sizeof(bf16)* residual_buffer.size());
    DEBUG_BLOCK(1,
    header_print_r("DEBUG", "Before all layers, hidden states");
    utils::print_matrix(hidden_state, Padded_QWEN3_5_VISION_HIDDEN_SIZE, 8, 10);
    )

    constexpr bool DEBUG_NO_REGRESSION_ERROR = false;
    vt.mark("pre_layers");

    for(int layer_idx = 0; layer_idx < QWEN3_5_VISION_NUM_LAYERS; layer_idx++){

    #if DEBUG_PRINT_ENCODE_TIME_DETAIL
            if  (DEBUG_NO_REGRESSION_ERROR && layer_idx!=0){
                // each layer, copy from the reference

                buffer<bf16> hidden_state_reference;
                reference_tensor.load_weights(
                    hidden_state_reference,
                    "residual_1_" + std::to_string(layer_idx) //
                );
                // memcpy(hidden_state.data(), hidden_state_reference.data(), sizeof(bf16)* seq_len* QWEN3_5_VISION_HIDDEN_SIZE);
                for(int ll = 0; ll < seq_len; ll++){
                    memcpy(
                        hidden_state.data() + ll * Padded_QWEN3_5_VISION_HIDDEN_SIZE,
                        hidden_state_reference.data() + ll * _QWEN3_5_VISION_HIDDEN_SIZE,
                        sizeof(bf16)* _QWEN3_5_VISION_HIDDEN_SIZE
                    );
                }
            }

    #endif

        memcpy(residual_buffer.data(), hidden_state.data(), sizeof(bf16)* seq_len* Padded_QWEN3_5_VISION_HIDDEN_SIZE);

        // first, do the norm_1 output
        auto time_start_layernorm1 = std::chrono::high_resolution_clock::now();
        layernorm_parallel(
            seq_len,
            _QWEN3_5_VISION_HIDDEN_SIZE,
            Padded_QWEN3_5_VISION_HIDDEN_SIZE,
            hidden_state.data(),
            norm_1_weight[layer_idx].data(),
            norm_1_bias[layer_idx].data(),
            QWEN3_5_VISION_LAYER_NORM_EPSILON,
            mm_qkv_input.data()  // Use separate output buffer!
        );
        auto time_end_layernorm1 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_layernorm1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_layernorm1 - time_start_layernorm1).count();
            std::cout << "Layer norm 1 time for layer " << layer_idx << ": " << duration_layernorm1 << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            std::cout << "Error after layernorm 1 comparison:" << std::endl;
            buffer<bf16> after_layernorm1;
            reference_tensor.load_weights(
                after_layernorm1,
                  "norm_1_" + std::to_string(layer_idx)
            );
            std::cout << "Error after norm1 layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16,bf16>(
                mm_qkv_input.data(), after_layernorm1.data(),  // Compare mm_qkv buffer which has norm output
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );

        #endif

        vt.mark("ln1");
        auto time_start_qkv_projection = std::chrono::high_resolution_clock::now();
        mm_qkv_input.sync_to_device();
        vt.mark("qkv_sync_in");
        attn_qkv_weight[layer_idx].sync_to_device();
        vt.mark("qkv_w_sync");
        if (this->use_overlay && layer_idx == 0) {
            qkv_first_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
        } else {
            qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
        }
        vt.mark("qkv_run");
        if (vt_probe && !this->use_overlay) {
            qkv_projection_app( mm_qkv_input,  attn_qkv_weight[layer_idx], mm_qkv_output_mha_input   );
            vt.mark("qkv_run_again");
        }
        mm_qkv_output_mha_input.sync_from_device();
        vt.mark("qkv_sync_out");
        dump_stage("qkv", layer_idx, mm_qkv_output_mha_input.data(), mm_qkv_output_mha_input.size());
        DEBUG_BLOCK(1,
        if (layer_idx == 0) {
            header_print_r("DEBUG", "After QKV projection, before adding bias, hidden states");
            utils::print_matrix(mm_qkv_output_mha_input, 3*Padded_QWEN3_5_VISION_HIDDEN_SIZE, 8, 10);
        }
        )
        auto time_end_qkv_projection = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_qkv_projection = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_qkv_projection - time_start_qkv_projection).count();
            std::cout << "QKV projection time for layer " << layer_idx << ": " << duration_qkv_projection << " ms" << std::endl;
        #endif
        auto time_start_add_qkv_bias = std::chrono::high_resolution_clock::now();
        // perform the add bias of q,k,v bias - ONLY process seq_len, not seq_len_padded!
        // for(int l = 0;  l < seq_len; l++){
        //     simd_add(
        //         mm_qkv_output_mha_input.data() + l * (3*QWEN3_5_VISION_HIDDEN_SIZE),
        //         attn_qkv_weight[layer_idx].data(),
        //         mm_qkv_output_mha_input.data() + l * (3*QWEN3_5_VISION_HIDDEN_SIZE),
        //         3*QWEN3_5_VISION_HIDDEN_SIZE
        //     );
        // }
        auto time_end_add_qkv_bias = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_add_qkv_bias = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_add_qkv_bias - time_start_add_qkv_bias).count();
            std::cout << "Add QKV bias time for layer " << layer_idx << ": " << duration_add_qkv_bias << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference

            buffer<bf16> after_qkv_projection;
            reference_tensor.load_weights(
                after_qkv_projection,
                "vision_attention_qkv_mm_with_bias_" + std::to_string(layer_idx)
            );
            std::cout << "Error after qkv mm layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_qkv_output_mha_input.data(), after_qkv_projection.data(),
                1,
                seq_len, 3*_QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, 3*Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );

        #endif

        // now apply rope to q, k
        auto time_start_rotary_pos_emb_application = std::chrono::high_resolution_clock::now();
        // if(ENABLE_QKV_REORDER){
        //     apply_rotary_pos_emb_vision_inplace_reordered(
        //         mm_qkv_output_mha_input.data(),
        //         cos_emb.data(),
        //         sin_emb.data(),
        //         seq_len,
        //         seq_len_padded,
        //         _QWEN3_5_VISION_HIDDEN_SIZE,
        //         Padded_QWEN3_5_VISION_HIDDEN_SIZE,
        //         QWEN3_5_VISION_NUM_HEADS
        //     );
        // }else{
            apply_rotary_pos_emb_vision_inplace(
                mm_qkv_output_mha_input.data(),
                cos_emb.data(),
                sin_emb.data(),
                seq_len,
                seq_len_padded,
                _QWEN3_5_VISION_HIDDEN_SIZE,
                Padded_QWEN3_5_VISION_HIDDEN_SIZE * 3,
                QWEN3_5_VISION_NUM_HEADS
            );
        // }

        auto time_end_rotary_pos_emb_application = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_rotary_pos_emb_application = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_rotary_pos_emb_application - time_start_rotary_pos_emb_application).count();
            std::cout << "Rotary position embedding application time for layer " << layer_idx << ": " << duration_rotary_pos_emb_application << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_q_rope;
            reference_tensor.load_weights(
                after_q_rope,
                "vision_attention_q_rotary_" + std::to_string(layer_idx)
            );
            std::cout << "Error after rotary position embedding application on qkv for layer " << layer_idx << " comparison:" << std::endl;

            print_error_metrics<bf16, bf16>(
                mm_qkv_output_mha_input.data(), after_q_rope.data(),
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, 3*Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );

            buffer<bf16> after_k_rope;
            reference_tensor.load_weights(
                after_k_rope,
                "vision_attention_k_rotary_" + std::to_string(layer_idx)
            );
            std::cout << "Error after rotary position embedding application on k for layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_qkv_output_mha_input.data() + _QWEN3_5_VISION_HIDDEN_SIZE, after_k_rope.data(),
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, 3*Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );

        #endif

        // now mha
        vt.mark("bias_rope");
        auto time_start_mha = std::chrono::high_resolution_clock::now();
        mm_qkv_output_mha_input.sync_to_device();
        mha_qkv_output_mm_o_input.sync_to_device();
        vt.mark("attn_sync_in");
        flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
        vt.mark("attn_run");
        if (vt_probe && !this->use_overlay) {
            flash_attention_app(mha_qkv_output_mm_o_input, mm_qkv_output_mha_input);
            vt.mark("attn_run_again");
        }
        mha_qkv_output_mm_o_input.sync_from_device();
        vt.mark("attn_sync_out");
        dump_stage("attn", layer_idx, mha_qkv_output_mm_o_input.data(), mha_qkv_output_mm_o_input.size());

        DEBUG_BLOCK(1,
        if (layer_idx == 0) {
            header_print_r("DEBUG", "After MHA");
            buffer<bf16> temp = buffer<bf16>(mha_qkv_output_mm_o_input.data() + (seq_len - 2) * Padded_QWEN3_5_VISION_HIDDEN_SIZE, (seq_len_padded - (seq_len - 2)) * Padded_QWEN3_5_VISION_HIDDEN_SIZE); // only print the q part which is the same as mha output
            utils::print_matrix(temp, Padded_QWEN3_5_VISION_HIDDEN_SIZE, 8, 10);
        }
        )

        auto time_end_mha = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mha = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mha - time_start_mha).count();
            std::cout << "MHA time for layer " << layer_idx << ": " << duration_mha << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_attention_output_ptr;
            reference_tensor.load_weights(
                after_attention_output_ptr,
                "vision_attention_output" + std::to_string(layer_idx)
            );
            std::cout << "Error after attention output layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mha_qkv_output_mm_o_input.data(), after_attention_output_ptr.data(),
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );
        #endif

        // now the o projection
        vt.mark("host_after_attn");
        auto time_start_o_projection = std::chrono::high_resolution_clock::now();
        mha_qkv_output_mm_o_input.sync_to_device();
        vt.mark("o_sync_in");
        o_proj_weight[layer_idx].sync_to_device();
        vt.mark("o_w_sync");
        o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
        vt.mark("o_run");
        if (vt_probe && !this->use_overlay) {
            o_projection_app( mha_qkv_output_mm_o_input,  o_proj_weight[layer_idx], mm_o_output_fc1_proj_input   );
            vt.mark("o_run_again");
        }
        mm_o_output_fc1_proj_input.sync_from_device();
        vt.mark("o_sync_out");
        dump_stage("o", layer_idx, mm_o_output_fc1_proj_input.data(), mm_o_output_fc1_proj_input.size());

        DEBUG_BLOCK(1,
        if (layer_idx == 0) {
            header_print_r("DEBUG", "After O projection:");
            utils::print_matrix(mm_o_output_fc1_proj_input, Padded_QWEN3_5_VISION_HIDDEN_SIZE, 8, 10);
        }
        )

        auto time_end_o_projection = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_o_projection = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_o_projection - time_start_o_projection).count();
            std::cout << "O projection time for layer " << layer_idx << ": " << duration_o_projection << " ms" << std::endl;
        #endif
        // add the bias
        auto time_start_add_o_proj_bias = std::chrono::high_resolution_clock::now();
        // for(int l = 0;  l < seq_len; l++){
        //     simd_add(
        //         mm_o_output_fc1_proj_input.data() + l * (QWEN3_5_VISION_HIDDEN_SIZE),
        //         o_proj_weight[layer_idx].data(),
        //         mm_o_output_fc1_proj_input.data() + l * (QWEN3_5_VISION_HIDDEN_SIZE),
        //         QWEN3_5_VISION_HIDDEN_SIZE
        //     );
        // }
        auto time_end_add_o_proj_bias = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_add_o_proj_bias = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_add_o_proj_bias - time_start_add_o_proj_bias).count();
            std::cout << "Add o projection bias time for layer " << layer_idx << ": " << duration_add_o_proj_bias << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_o_project_ptr;
            reference_tensor.load_weights(
                after_o_project_ptr,
                "vision_attention_output_proj_" + std::to_string(layer_idx)
            );
            std::cout << "Error after o projection layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_o_output_fc1_proj_input.data(), after_o_project_ptr.data(),
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );
        #endif

        auto time_start_residual_add_1 = std::chrono::high_resolution_clock::now();
        simd_add(
            mm_o_output_fc1_proj_input.data(),
            residual_buffer.data(),
            hidden_state.data(),
            seq_len * Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );
        auto time_end_residual_add_1 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_residual_add_1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_residual_add_1 - time_start_residual_add_1).count();
            std::cout << "Residual add 1 time for layer " << layer_idx << ": " << duration_residual_add_1 << " ms" << std::endl;
        #endif
        memcpy(
            residual_buffer.data(),
            hidden_state.data(),
            sizeof(bf16) * seq_len * Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );

        // norm2
        auto time_start_layernorm2 = std::chrono::high_resolution_clock::now();
        layernorm_parallel(
            seq_len,
            _QWEN3_5_VISION_HIDDEN_SIZE,
            Padded_QWEN3_5_VISION_HIDDEN_SIZE,
            hidden_state.data(),
            norm_2_weight[layer_idx].data(),
            norm_2_bias[layer_idx].data(),
            QWEN3_5_VISION_LAYER_NORM_EPSILON,
            mm_o_output_fc1_proj_input.data()
        );
        auto time_end_layernorm2 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_layernorm2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_layernorm2 - time_start_layernorm2).count();
            std::cout << "Layer norm 2 time for layer " << layer_idx << ": " << duration_layernorm2 << " ms" << std::endl;
        #endif

        // now to fc1
        vt.mark("bias_res_ln2");
        auto time_start_fc1 = std::chrono::high_resolution_clock::now();
        mm_o_output_fc1_proj_input.sync_to_device();
        vt.mark("fc1_sync_in");
        mlp_fc1_weight[layer_idx].sync_to_device();
        mlp_fc2_weight[layer_idx].sync_to_device();
        vt.mark("fc_w_sync");
        mlp_fc1_app( mm_o_output_fc1_proj_input,  mlp_fc1_weight[layer_idx],   mm_fc1_output_fc2_input );
        vt.mark("fc1_run");
        // mm_fc1_output_fc2_input.sync_from_device();
        auto time_end_fc1 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_fc1 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_fc1 - time_start_fc1).count();
            std::cout << "MLP FC1 time for layer " << layer_idx << ": " << duration_fc1 << " ms" << std::endl;
        #endif

        // Combined bias addition and GELU activation using optimized AVX-512
        auto time_start_mlp_fc1_gelu = std::chrono::high_resolution_clock::now();
        // simd_bias_add_gelu(
        //     mm_fc1_output_fc2_input.data(),
        //     mlp_fc1_weight[layer_idx].data(),
        //     mm_fc1_output_fc2_input.data(),
        //     seq_len * QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE,
        //     QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE
        // );
        auto time_end_mlp_fc1_gelu = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc1_gelu = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc1_gelu - time_start_mlp_fc1_gelu).count();
            std::cout << "MLP FC1 GELU time for layer " << layer_idx << ": " << duration_mlp_fc1_gelu << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare with the reference
            buffer<bf16> after_mlp_fc1_gelu_ptr;
            reference_tensor.load_weights(
                after_mlp_fc1_gelu_ptr,
                "vision_mlp_activated_output_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp fc1 gelu layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_fc1_output_fc2_input.data(), after_mlp_fc1_gelu_ptr.data(),
                1,
                seq_len, _QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE,
                seq_len_padded, Padded_QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE
            );
        #endif

        // now do the fc2
        vt.mark("host_after_fc1");
        auto time_start_mlp_fc2 = std::chrono::high_resolution_clock::now();
        mlp_fc2_app( mm_fc1_output_fc2_input,  mlp_fc2_weight[layer_idx],   mm_fc2_output );
        vt.mark("fc2_run");
        mm_fc2_output.sync_from_device();
        vt.mark("fc2_sync_out");
        dump_stage("fc2", layer_idx, mm_fc2_output.data(), mm_fc2_output.size());

        DEBUG_BLOCK(1,
        if (layer_idx == 0) {
            header_print_r("DEBUG", "After FC2 projection:");
            utils::print_matrix(mm_fc2_output, Padded_QWEN3_5_VISION_HIDDEN_SIZE, 8, 10);
        }
        )

        auto time_end_mlp_fc2 = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc2 = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc2 - time_start_mlp_fc2).count();
            std::cout << "MLP FC2 time for layer " << layer_idx << ": " << duration_mlp_fc2 << " ms" << std::endl;
        #endif

        // add bias
        auto time_start_mlp_fc2_bias_add = std::chrono::high_resolution_clock::now();
        auto time_end_mlp_fc2_bias_add = std::chrono::high_resolution_clock::now();
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_fc2_bias_add = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_fc2_bias_add - time_start_mlp_fc2_bias_add).count();
            std::cout << "MLP FC2 bias add time for layer " << layer_idx << ": " << duration_mlp_fc2_bias_add << " ms" << std::endl;
        #endif

        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            //compare with the reference
            buffer<bf16> after_mlp_fc2_ptr;
            reference_tensor.load_weights(
                after_mlp_fc2_ptr,
                "vision_mlp_final_output_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp fc2 layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                mm_fc2_output.data(), after_mlp_fc2_ptr.data(),
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );
        #endif

        auto time_start_mlp_residual_add = std::chrono::high_resolution_clock::now();
        simd_add(
            mm_fc2_output.data(),
            residual_buffer.data(),
            hidden_state.data(),
            seq_len * Padded_QWEN3_5_VISION_HIDDEN_SIZE
        );
        auto time_end_mlp_residual_add = std::chrono::high_resolution_clock::now();
        vt.mark("residual2");
        #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
            auto duration_mlp_residual_add = std::chrono::duration_cast<std::chrono::milliseconds>(time_end_mlp_residual_add - time_start_mlp_residual_add).count();
            std::cout << "MLP residual add time for layer " << layer_idx << ": " << duration_mlp_residual_add << " ms" << std::endl;
        #endif
        #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
            // compare the final
            buffer<bf16> after_mlp_residual_add_ptr;
            reference_tensor.load_weights(
                after_mlp_residual_add_ptr,
                "vision_block_post_mlp_" + std::to_string(layer_idx)
            );
            std::cout << "Error after mlp residual add layer " << layer_idx << " comparison:" << std::endl;
            print_error_metrics<bf16, bf16>(
                hidden_state.data(), after_mlp_residual_add_ptr.data(),
                1,
                seq_len, _QWEN3_5_VISION_HIDDEN_SIZE,
                seq_len_padded, Padded_QWEN3_5_VISION_HIDDEN_SIZE
            );

        #endif
        DEBUG_BLOCK(1,
        header_print_r("DEBUG", "After layer " + std::to_string(layer_idx) + ", hidden states");
        utils::print_matrix(hidden_state, Padded_QWEN3_5_VISION_HIDDEN_SIZE, 8, 10);
        )
    }

    // // do it for last layer deepstack

    std::vector<bf16> final_deepstack_hidden_vector(merger_seq_len_padded * Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE);
    int deep_stack_indx = final_res.size();

    // Apply layernorm per token AND rearrange into merged layout.
    // Python equivalent: x = self.norm(x).view(-1, self.hidden_size)
    // where self.hidden_size = spatial_merge_size^2 * embed_dim = 4 * 1152 = 4608
    // This packs every spatial_merge_size^2 consecutive tokens into one merged row.
    int spatial_merge_size_sq = this->parent_npu_ptr->QWEN3_5_SPATIAL_MERGE_SIZE * this->parent_npu_ptr->QWEN3_5_SPATIAL_MERGE_SIZE;
    for(int s = 0; s < seq_len; s++){
        int merged_row = s / spatial_merge_size_sq;
        int sub_idx = s % spatial_merge_size_sq;
        layernorm_high_precision(
            _QWEN3_5_VISION_HIDDEN_SIZE,
            hidden_state.data() + s * Padded_QWEN3_5_VISION_HIDDEN_SIZE,
            merger_norm_weight.data(),
            merger_norm_bias.data(),
            QWEN3_5_VISION_LAYER_NORM_EPSILON,
            merger_fc1_input.data() + merged_row * Padded_QWEN3_5_MERGER_HIDDEN_SIZE + sub_idx * _QWEN3_5_VISION_HIDDEN_SIZE
        );
    }

    vt.mark("merger_ln");
    merger_fc1_input.sync_to_device();
    merger_fc1_weight.sync_to_device();
    // merger_fc1_output_fc2_input.sync_to_device();
    merger_fc2_weight.sync_to_device();
    merger_fc1_app( merger_fc1_input,  merger_fc1_weight,   merger_fc1_output_fc2_input );

    //fc2

    merger_fc2_app( merger_fc1_output_fc2_input,  merger_fc2_weight,   merger_fc2_output );
    merger_fc2_output.sync_from_device();
    vt.mark("merger_mm");

    memcpy(final_deepstack_hidden_vector.data(), merger_fc2_output.data(), sizeof(bf16) *merger_seq_len * Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE);
    final_res.push_back(final_deepstack_hidden_vector);
    #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // compare with the reference
        buffer<bf16> deepstack_output_ptr_ref;
        reference_tensor.load_weights(
            deepstack_output_ptr_ref,
            "final_hidden_states_after_merger"
        );
        std::cout << "Error after final patch merger comparison:" << std::endl;
        print_error_metrics<bf16, bf16>(
            final_deepstack_hidden_vector.data(), deepstack_output_ptr_ref.data(),
            1,
            merger_seq_len, _QWEN3_5_VISION_OUT_HIDDEN_SIZE,
            merger_seq_len_padded, Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE
        );

    #endif

    auto encode_time_end = std::chrono::high_resolution_clock::now();
    #ifdef DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto total_encode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encode_time_end - encoder_start_time).count();
        std::cout << "Total encoding time: " << total_encode_duration << " ms" << std::endl;
    #endif

    vt.mark("tail");
    vt.print(seq_len, seq_len_padded);
    if (const char* dump_prefix = std::getenv("VISION_DUMP")) {
        std::ofstream out(std::string(dump_prefix) + "_L" + std::to_string(seq_len) + ".bin", std::ios::binary);
        for (const auto& v : final_res) {
            out.write((const char*)v.data(), v.size() * sizeof(bf16));
        }
    }
    return final_res;
}

