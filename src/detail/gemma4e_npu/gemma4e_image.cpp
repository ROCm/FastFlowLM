#include "gemma4e_image.hpp"

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
#include "seq_gen.hpp"

#define DEBUG_PRINT_ENCODE_TIME_DETAIL (DEBUG_LEVEL >= 1)
#define DEBUG_PRINT_ENCODE_ERROR_METRICS (DEBUG_LEVEL >= 1)

Gemma4e_ImageEncoder::~Gemma4e_ImageEncoder()
{
}

Gemma4e_ImageEncoder::Gemma4e_ImageEncoder(LM_Config config, npu_xclbin_manager *npu_instance, gemma4e_npu* parent_npu_ptr)
    : config(config), npu(npu_instance), model_path(config.model_path), parent_npu_ptr(parent_npu_ptr)
{

    //TODO: FIXME:hi

    // load parameters from json file

    {

        MM_tile_M = config.sub("vision_config").value("VISION_MM_TILE_M", -1);
        MM_tile_K = config.sub("vision_config").value("VISION_MM_TILE_K", -1);
        MM_tile_N = config.sub("vision_config").value("VISION_MM_TILE_N", -1);

        seq_len_pad_requirement_for_MM = MM_ROW_SIZE*MM_tile_M;
        assert( MM_tile_K % MM_tile_N == 0); // we need this for the way we generate the sequence, because k and N is interchangable at MLP (gate-down)
        DEBUG_BLOCK(1,
        std::cout << "MM_tile_M: " << MM_tile_M << ", MM_tile_K: " << MM_tile_K << ", MM_tile_N: " << MM_tile_N << std::endl;
        )
    }

    Padded_GEMMA4E_VISION_HIDDEN_SIZE = round_up_to_multiple(this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE, MM_tile_K);
    Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE = round_up_to_multiple(this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE, MM_tile_K);
    Padded_GEMMA4E_VISION_OUT_HIDDEN_SIZE = round_up_to_multiple(this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE, MM_tile_K);

    // #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    //     // print all the parameters for debug
    //     std::cout << "QWEN3_5_VISION_EMBED_DIM: " << QWEN3_5_VISION_EMBED_DIM << std::endl;
    //     std::cout << "QWEN3_5_VISION_NUM_HEADS: " << QWEN3_5_VISION_NUM_HEADS << std::endl;
    //     std::cout << "QWEN3_5_VISION_HEAD_DIM: " <<  QWEN3_5_VISION_HEAD_DIM << std::endl;
    //     std::cout << "QWEN3_5_VISION_HIDDEN_SIZE: " << _QWEN3_5_VISION_HIDDEN_SIZE << std::endl;
    //     std::cout << "QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE: " << _QWEN3_5_VISION_MLP_INTERMEDIATE_SIZE << std::endl;
    //     std::cout << "QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS: " << QWEN3_5_VISION_NUM_POSITION_EMBEDDINGS << std::endl;
    //     std::cout << "QWEN3_5_VISION_NUM_LAYERS: " << QWEN3_5_VISION_NUM_LAYERS << std::endl;
    //     std::cout << "QWEN3_5_VISION_LAYER_NORM_EPSILON: " << QWEN3_5_VISION_LAYER_NORM_EPSILON << std::endl;
    //     std::cout << "QWEN3_5_MERGER_HIDDEN_SIZE: " << _QWEN3_5_MERGER_HIDDEN_SIZE << std::endl;
    //     std::cout << "QWEN3_5_VISION_OUT_HIDDEN_SIZE: " << _QWEN3_5_VISION_OUT_HIDDEN_SIZE << std::endl;

    //     // print the padded parameters for debug

    // #endif

    this->fla = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_attn.xclbin"));
    this->proj = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm.xclbin"));
    this->proj_high_precision = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "vision_mm_high_precision.xclbin"));

    this->flash_attention_app = this->fla->create_app();
    this->patch_embedder_posisiton_embedding_dim_0_app = this->proj->create_app();//this->proj->create_app(); //TODO: this might not needeD?
    this->patch_embedder_posisiton_embedding_dim_1_app = this->proj->create_app();//this->proj->create_app();
    this->patch_embedding_app                          = this->proj_high_precision->create_app();//this->proj->create_app();
    this->q_proj_app = this->proj->create_app();
    this->k_proj_app = this->proj->create_app();
    this->v_proj_app = this->proj->create_app();
    this->o_proj_app = this->proj->create_app();
    this->gate_proj_app = this->proj->create_app();
    this->up_proj_app = this->proj->create_app();
    this->down_proj_app = this->proj->create_app();

    this->vision_to_language_input_projection_app = this->proj->create_app();

    // // attempting to read the info that vision model needs from config.sub("vision_config")

    q_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    k_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    v_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    o_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    gate_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    up_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    down_proj_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    q_norm_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    k_norm_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    post_o_norm_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    post_ffn_norm_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    layer_norm_1_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
    layer_norm_2_weight.resize(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS);
}
void Gemma4e_ImageEncoder::init_weights(SafeTensors &q4nx){
    DEBUG_BLOCK(1,
    std::cout << "HIT: init_weights of Gemma4e_ImageEncoder is called. Loading weights from " << this->model_path << std::endl;
    );

    {

        buffer<bf16> temp_buffer;

        q4nx.load_weights(temp_buffer,"model.vision.patch_embedder.position_embedding_table" );
        this->patch_embedder_position_embedding_table = this->patch_embedder_posisiton_embedding_dim_0_app.create_bo_buffer<bf16>(
            temp_buffer.size()
        );
        memcpy(this->patch_embedder_position_embedding_table.data(), temp_buffer.data(), temp_buffer.size()*sizeof(bf16));
    }
    {
        buffer<bf16> temp_buffer;
        q4nx.load_weights(temp_buffer,"model.vision.patch_embd.weight" );
        this->patch_embd_weight = this->patch_embedding_app.create_bo_buffer<bf16>(
            temp_buffer.size()
        );
        memcpy(this->patch_embd_weight.data(), temp_buffer.data(), temp_buffer.size()*sizeof(bf16));
    }
    {
        buffer<bf16> temp_buffer;
        q4nx.load_weights(temp_buffer,"model.vision.embedding_projection.weight" );
        this->vision_to_language_input_projection_weight = this->vision_to_language_input_projection_app.create_bo_buffer<bf16>(
            temp_buffer.size()
        );
        memcpy(this->vision_to_language_input_projection_weight.data(), temp_buffer.data(), temp_buffer.size()*sizeof(bf16));
    }

    for(int layer_id=0; layer_id < this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS; layer_id++){

        this->q_proj_weight[layer_id] = this->q_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_HIDDEN_SIZE*Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->q_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".vision_attn.q_proj.weight"
        );

        this->k_proj_weight[layer_id] = this->k_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_HIDDEN_SIZE*Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->k_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".vision_attn.k_proj.weight"
        );
        this->v_proj_weight[layer_id] = this->v_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_HIDDEN_SIZE*Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->v_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".vision_attn.v_proj.weight"
        );
        this->o_proj_weight[layer_id] = this->o_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_HIDDEN_SIZE*Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->o_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".vision_attn.out_proj.weight"
        );
        this->gate_proj_weight[layer_id] = this->gate_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_HIDDEN_SIZE*Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE
        );
        q4nx.load_weights(this->gate_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".ffn.gate_proj.weight"
        );
        this->up_proj_weight[layer_id] = this->up_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_HIDDEN_SIZE*Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE
        );
        q4nx.load_weights(this->up_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".ffn.up_proj.weight"
        );
        this->down_proj_weight[layer_id] = this->down_proj_app.create_bo_buffer<bf16>(
            Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE*Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        q4nx.load_weights(this->down_proj_weight[layer_id],
          "model.vision."+std::to_string(layer_id)+".ffn.down_proj.weight"
        );

        // load all norm weights
        q4nx.load_weights(
            this->q_norm_weight[layer_id],
            "model.vision."+std::to_string(layer_id)+".vision_attn.q_norm.weight"
        );
        q4nx.load_weights(
            this->k_norm_weight[layer_id],
            "model.vision."+std::to_string(layer_id)+".vision_attn.k_norm.weight"
        );
        q4nx.load_weights(
            this->post_o_norm_weight[layer_id],
            "model.vision."+std::to_string(layer_id)+".post_attn_norm.weight"
        );
        q4nx.load_weights(
            this->post_ffn_norm_weight[layer_id],
            "model.vision."+std::to_string(layer_id)+".ffn_post_norm.weight"
        );
        q4nx.load_weights(
            this->layer_norm_1_weight[layer_id],
            "model.vision."+std::to_string(layer_id)+".norm1.weight"
        );
        q4nx.load_weights(
            this->layer_norm_2_weight[layer_id],
            "model.vision."+std::to_string(layer_id)+".norm2.weight"
        );

        // now, we load the min max scalar
        buffer<bf16> q_input_min;
        q4nx.load_weights(q_input_min,
            "model.vision."+std::to_string(layer_id)+".vision_attn.q_input_min"
        );
        assert(q_input_min.size() == 1);
        this->input_q_min.push_back(q_input_min[0]);

        buffer<bf16> q_input_max;
        q4nx.load_weights(q_input_max,
            "model.vision."+std::to_string(layer_id)+".vision_attn.q_input_max"
        );
        assert(q_input_max.size() == 1);
        this->input_q_max.push_back(q_input_max[0]);

        buffer<bf16> q_output_min;
        q4nx.load_weights(q_output_min,
            "model.vision."+std::to_string(layer_id)+".vision_attn.q_output_min"
        );
        assert(q_output_min.size() == 1);
        this->output_q_min.push_back(q_output_min[0]);

        buffer<bf16> q_output_max;
        q4nx.load_weights(q_output_max,
            "model.vision."+std::to_string(layer_id)+".vision_attn.q_output_max"
        );
        assert(q_output_max.size() == 1);
        this->output_q_max.push_back(q_output_max[0]);

        // k min/max
        buffer<bf16> k_input_min;
        q4nx.load_weights(k_input_min,
            "model.vision."+std::to_string(layer_id)+".vision_attn.k_input_min"
        );
        assert(k_input_min.size() == 1);
        this->input_k_min.push_back(k_input_min[0]);

        buffer<bf16> k_input_max;
        q4nx.load_weights(k_input_max,
            "model.vision."+std::to_string(layer_id)+".vision_attn.k_input_max"
        );
        assert(k_input_max.size() == 1);
        this->input_k_max.push_back(k_input_max[0]);

        buffer<bf16> k_output_min;
        q4nx.load_weights(k_output_min,
            "model.vision."+std::to_string(layer_id)+".vision_attn.k_output_min"
        );
        assert(k_output_min.size() == 1);
        this->output_k_min.push_back(k_output_min[0]);

        buffer<bf16> k_output_max;
        q4nx.load_weights(k_output_max,
            "model.vision."+std::to_string(layer_id)+".vision_attn.k_output_max"
        );
        assert(k_output_max.size() == 1);
        this->output_k_max.push_back(k_output_max[0]);

        // v min/max
        buffer<bf16> v_input_min;
        q4nx.load_weights(v_input_min,
            "model.vision."+std::to_string(layer_id)+".vision_attn.v_input_min"
        );
        assert(v_input_min.size() == 1);
        this->input_v_min.push_back(v_input_min[0]);

        buffer<bf16> v_input_max;
        q4nx.load_weights(v_input_max,
            "model.vision."+std::to_string(layer_id)+".vision_attn.v_input_max"
        );
        assert(v_input_max.size() == 1);
        this->input_v_max.push_back(v_input_max[0]);

        buffer<bf16> v_output_min;
        q4nx.load_weights(v_output_min,
            "model.vision."+std::to_string(layer_id)+".vision_attn.v_output_min"
        );
        assert(v_output_min.size() == 1);
        this->output_v_min.push_back(v_output_min[0]);

        buffer<bf16> v_output_max;
        q4nx.load_weights(v_output_max,
            "model.vision."+std::to_string(layer_id)+".vision_attn.v_output_max"
        );
        assert(v_output_max.size() == 1);
        this->output_v_max.push_back(v_output_max[0]);

        // o (attn_out) min/max
        buffer<bf16> o_input_min;
        q4nx.load_weights(o_input_min,
            "model.vision."+std::to_string(layer_id)+".attn_out_input_min"
        );
        assert(o_input_min.size() == 1);
        this->input_o_min.push_back(o_input_min[0]);

        buffer<bf16> o_input_max;
        q4nx.load_weights(o_input_max,
            "model.vision."+std::to_string(layer_id)+".attn_out_input_max"
        );
        assert(o_input_max.size() == 1);
        this->input_o_max.push_back(o_input_max[0]);

        buffer<bf16> o_output_min;
        q4nx.load_weights(o_output_min,
            "model.vision."+std::to_string(layer_id)+".attn_out_output_min"
        );
        assert(o_output_min.size() == 1);
        this->output_o_min.push_back(o_output_min[0]);

        buffer<bf16> o_output_max;
        q4nx.load_weights(o_output_max,
            "model.vision."+std::to_string(layer_id)+".attn_out_output_max"
        );
        assert(o_output_max.size() == 1);
        this->output_o_max.push_back(o_output_max[0]);

        // gate min/max
        buffer<bf16> gate_input_min;
        q4nx.load_weights(gate_input_min,
            "model.vision."+std::to_string(layer_id)+".ffn.gate_input_min"
        );
        assert(gate_input_min.size() == 1);
        this->input_gate_min.push_back(gate_input_min[0]);

        buffer<bf16> gate_input_max;
        q4nx.load_weights(gate_input_max,
            "model.vision."+std::to_string(layer_id)+".ffn.gate_input_max"
        );
        assert(gate_input_max.size() == 1);
        this->input_gate_max.push_back(gate_input_max[0]);

        buffer<bf16> gate_output_min;
        q4nx.load_weights(gate_output_min,
            "model.vision."+std::to_string(layer_id)+".ffn.gate_output_min"
        );
        assert(gate_output_min.size() == 1);
        this->output_gate_min.push_back(gate_output_min[0]);

        buffer<bf16> gate_output_max;
        q4nx.load_weights(gate_output_max,
            "model.vision."+std::to_string(layer_id)+".ffn.gate_output_max"
        );
        assert(gate_output_max.size() == 1);
        this->output_gate_max.push_back(gate_output_max[0]);

        // up min/max
        buffer<bf16> up_input_min;
        q4nx.load_weights(up_input_min,
            "model.vision."+std::to_string(layer_id)+".ffn.up_input_min"
        );
        assert(up_input_min.size() == 1);
        this->input_up_min.push_back(up_input_min[0]);

        buffer<bf16> up_input_max;
        q4nx.load_weights(up_input_max,
            "model.vision."+std::to_string(layer_id)+".ffn.up_input_max"
        );
        assert(up_input_max.size() == 1);
        this->input_up_max.push_back(up_input_max[0]);

        buffer<bf16> up_output_min;
        q4nx.load_weights(up_output_min,
            "model.vision."+std::to_string(layer_id)+".ffn.up_output_min"
        );
        assert(up_output_min.size() == 1);
        this->output_up_min.push_back(up_output_min[0]);

        buffer<bf16> up_output_max;
        q4nx.load_weights(up_output_max,
            "model.vision."+std::to_string(layer_id)+".ffn.up_output_max"
        );
        assert(up_output_max.size() == 1);
        this->output_up_max.push_back(up_output_max[0]);

        // down min/max
        buffer<bf16> down_input_min;
        q4nx.load_weights(down_input_min,
            "model.vision."+std::to_string(layer_id)+".ffn.down_input_min"
        );
        assert(down_input_min.size() == 1);
        this->input_down_min.push_back(down_input_min[0]);

        buffer<bf16> down_input_max;
        q4nx.load_weights(down_input_max,
            "model.vision."+std::to_string(layer_id)+".ffn.down_input_max"
        );
        assert(down_input_max.size() == 1);
        this->input_down_max.push_back(down_input_max[0]);

        buffer<bf16> down_output_min;
        q4nx.load_weights(down_output_min,
            "model.vision."+std::to_string(layer_id)+".ffn.down_output_min"
        );
        assert(down_output_min.size() == 1);
        this->output_down_min.push_back(down_output_min[0]);

        buffer<bf16> down_output_max;
        q4nx.load_weights(down_output_max,
            "model.vision."+std::to_string(layer_id)+".ffn.down_output_max"
        );
        assert(down_output_max.size() == 1);
        this->output_down_max.push_back(down_output_max[0]);
    }

    DEBUG_BLOCK(1,
    std::cout << "[DBG] Gemma4e_ImageEncoder::init_weights finished" << std::endl;
    );
}

std::vector<bf16> Gemma4e_ImageEncoder::encode( void* image_payload_ptr)
{

    DEBUG_BLOCK(1,
    std::cout << "HIT: Gemma4e_ImageEncoder::encode is called with image_payload_ptr: " << image_payload_ptr << std::endl;
    );
    // //
    //DEBUG
    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    SafeTensors reference_tensor(this->model_path + "/vision_reference_data.safetensors");
    std::cout << "reference_Tensor path is " << this->model_path + "/vision_reference_data.safetensors" << std::endl;
    std::cout << "debug Gemma4e_ImageEncoder::encode called with image_payload_ptr: " << image_payload_ptr << std::endl;
    #endif
    // auto encoder_start_time = std::chrono::high_resolution_clock::now();

    gemma4e_image_payload_t* image_payload = (gemma4e_image_payload_t*)image_payload_ptr;

    // first, compare the pixel_values with pre_Gemma4VisionPatchEmbedder_pixel_values

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error after patch embedding comparison:" << std::endl;
        buffer<float> pre_Gemma4VisionPatchEmbedder_pixel_values;
        reference_tensor.load_weights(
            pre_Gemma4VisionPatchEmbedder_pixel_values,
            "pre_Gemma4VisionPatchEmbedder_pixel_values"
        );
        std::cout << "Comparing pixel values with reference..." << std::endl;
        uint32_t offset = 0;
        for(int i = 0; i < image_payload->num_images; i++){
            uint32_t cur_size = image_payload->image_patch__element_per_patch[i].first*image_payload->image_patch__element_per_patch[i].second;
            print_error_metrics<bf16, float>(
                image_payload->pixel_values[i].data(),
                pre_Gemma4VisionPatchEmbedder_pixel_values.data() + offset,
                1,
                cur_size, 1,
                cur_size, 1
            );
            offset += cur_size;
        }

    #endif
    // std::cout << "Finished comparing pixel values with reference." << std::endl;

    // TODO: lets use avx512 for it later

    // for each (image_payload->pixel_values)       pixel_values = 2 * (pixel_values - 0.5)

    // sanity checkst
    for(int i = 0; i < image_payload->num_images; i++){
        assert( image_payload->image_patch__element_per_patch[i].second  == this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE);
    }

    std::vector<int> seq_len_per_image; // unpadded seq_len per image
    std::vector<int> start_seq_len_index_per_image; // the start index in the sequence for each image1

    int seq_len = 0;
    int seq_len_of_last_image = 0;
    for(auto image_valid_patch: image_payload->valid_patch_size_per_image ){
        seq_len += image_valid_patch;
        seq_len_of_last_image = image_valid_patch;
        seq_len_per_image.push_back(image_valid_patch);

        start_seq_len_index_per_image.push_back(seq_len - image_valid_patch); // cumulative offset before this image
    }

    int seq_len_padded = round_up_to_multiple(seq_len_of_last_image , vision_L_padded_requirement_for_attention) - seq_len_of_last_image;
    seq_len_padded  += seq_len;
    seq_len_padded = round_up_to_multiple(seq_len_padded, seq_len_pad_requirement_for_MM);

    assert(this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE % MM_tile_K == 0);
    assert(this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE % MM_tile_N == 0);

    DEBUG_BLOCK(1,
    std::cout <<"seq_len_pad_requirement_for_MM is " << seq_len_pad_requirement_for_MM << std::endl;
    )
    buffer<bf16> patch_emb_input = patch_embedding_app.create_bo_buffer<bf16>(
        seq_len_padded *this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> patch_emb_output = patch_embedding_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    buffer<bf16> q_projection_input= q_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> k_projection_input = k_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> v_projection_input = v_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    buffer<bf16> q_projection_output= q_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> k_projection_output = k_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> v_projection_output = v_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    buffer<bf16> attention_output = this->flash_attention_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> o_projection_output = o_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    buffer<bf16> gate_input = gate_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    buffer<bf16> up_input = up_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    buffer<bf16> gate_output = gate_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE
    );
    buffer<bf16> up_output = up_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE
    );
    buffer<bf16> down_output = down_proj_app.create_bo_buffer<bf16>(
        seq_len_padded * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    //[2, seq_len_padded, GEMMA4E_POSITION_EMBEDDING_SIZE]
    buffer<bf16> one_shot_buffer = patch_embedder_posisiton_embedding_dim_0_app.create_bo_buffer<bf16>(
        seq_len_padded* 2* this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE
    );
    //[2, seq_len_padded, PADDED_GEMMA4E_VISION_HIDDEN_SIZE]
    buffer<bf16> position_embedding_table_output = patch_embedder_posisiton_embedding_dim_0_app.create_bo_buffer<bf16>(
        seq_len_padded* 2* Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );

    //     // memset the buffer to zero

    {

        uint32_t ADD_BIAS = false;// no bias at all for all the mm
        generate_mm_sequence<bf16, bf16>(*this->patch_embedding_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,
                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                ADD_BIAS, 0,
                0,  -10000.0, 1000000.0, // do not clamp on output
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
        );

        generate_mm_sequence<bf16, bf16>(*this->patch_embedder_posisiton_embedding_dim_0_app.seq(),
                    seq_len_padded, this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE   ,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                    MM_tile_M,MM_tile_K,MM_tile_N,
                    8,8,8,
                    rtp_address, rtp_sync_lock_id,
                    MM_ROW_SIZE,MM_COL_SIZE,
                    0,0,0,
                    IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                    ADD_BIAS, 0,
                    0,  -10000.0, 1000000.0, // do not clamp on output
                    ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
        );

        generate_mm_sequence<bf16, bf16>(*this->patch_embedder_posisiton_embedding_dim_1_app.seq(),
                    seq_len_padded, this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE   ,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                    MM_tile_M,MM_tile_K,MM_tile_N,
                    8,8,8,
                    rtp_address, rtp_sync_lock_id,
                    MM_ROW_SIZE,MM_COL_SIZE,

                    seq_len_padded* this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE,
                    this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE* Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                    seq_len_padded* Padded_GEMMA4E_VISION_HIDDEN_SIZE,

                    IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                    ADD_BIAS, 0,
                    0,  -10000.0, 1000000.0, // do not clamp on output
                    ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
        );

        gen_mha_vision_attention(

            this->flash_attention_app.seq(),
            seq_len_per_image,
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
            this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
            this->Padded_GEMMA4E_VISION_HIDDEN_SIZE,
            this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM,
            this->parent_npu_ptr->GEMMA4E_VISION_NUM_ATTENTION_HEADS

        );
    }

    // now, apply the    pixel_values = 2 * (pixel_values - 0.5) to all valye in patch_emb_input
    std::vector<bf16> temp_pixel_values(patch_emb_input.size());
    memset(temp_pixel_values.data(), 0, temp_pixel_values.size() * sizeof(bf16));
    for(int i = 0, seq_len_offset = 0; i < image_payload->num_images; i++){

        bf16* raw_pixel_values_ptr = image_payload->pixel_values[i].data();
        bf16* patch_emb_input_ptr = (bf16*)temp_pixel_values.data() + (seq_len_offset* Padded_GEMMA4E_VISION_HIDDEN_SIZE);

        for(int l = 0; l < seq_len_per_image[i]; l++){
            for(int d = 0; d < this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; d++){
                float pixel_value = (float)raw_pixel_values_ptr[l*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE + d];
                pixel_value = 2.0f * (pixel_value - 0.5f);
                patch_emb_input_ptr[l*Padded_GEMMA4E_VISION_HIDDEN_SIZE + d] = (bf16)pixel_value;
            }
        }
        seq_len_offset += seq_len_per_image[i];
    }
    memcpy(patch_emb_input.data(), temp_pixel_values.data(), temp_pixel_values.size()*sizeof(bf16));

    patch_emb_input.sync_to_device();

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        std::cout << "Error after scaled vision hidden_size :" << std::endl;
        buffer<float> scaled_Gemma4VisionPatchEmbedder_pixel_values;
        reference_tensor.load_weights(
            scaled_Gemma4VisionPatchEmbedder_pixel_values,
            "scaled_Gemma4VisionPatchEmbedder_pixel_values"
        );

        for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
            print_error_metrics<bf16, float>(
                patch_emb_input.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                scaled_Gemma4VisionPatchEmbedder_pixel_values.data() + ref_offset,
                1,
                seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                image_payload->image_patch__element_per_patch[i].first, Padded_GEMMA4E_VISION_HIDDEN_SIZE

            );
            ref_offset+= image_payload->image_patch__element_per_patch[i].first*image_payload->image_patch__element_per_patch[i].second;
        }

        // //TODO: FIXME:

        // for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
        //     // memcpy(
        //     //     patch_emb_input.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
        //     //     scaled_Gemma4VisionPatchEmbedder_pixel_values.data() + ref_offset,

        //     //     seq_len_per_image[i]* this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16)

        //     }

    #endif
    DEBUG_BLOCK(1,
    std::cout << "model.vision.patch_embd.weight size: " << patch_embd_weight.size() << std::endl;
    )

    #if DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto patch_emb_start_time = std::chrono::high_resolution_clock::now();
    #endif
    patch_emb_input.sync_to_device();
    patch_embd_weight.sync_to_device();
    patch_embedding_app(patch_emb_input,patch_embd_weight, patch_emb_output  );
    patch_emb_output.sync_from_device();
    #if DEBUG_PRINT_ENCODE_TIME_DETAIL
        auto patch_emb_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> patch_emb_duration = patch_emb_end_time - patch_emb_start_time;
        std::cout << "Time taken for patch embedding: " << patch_emb_duration.count() << " ms" << std::endl;
    #endif
    // #if DEBUG_PRINT_ENCODE_ERROR_METRICS

    //     std::cout << "Error  for Gemma4VisionPatchEmbedder_hidden_states:" << std::endl;
    //     buffer<bf16> Gemma4VisionPatchEmbedder_hidden_states;
    //     reference_tensor.load_weights(
    //         Gemma4VisionPatchEmbedder_hidden_states,
    //         "Gemma4VisionPatchEmbedder_hidden_states"
    //     );

    //     for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
    //         print_error_metrics<bf16, bf16>(
    //             patch_emb_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
    //             Gemma4VisionPatchEmbedder_hidden_states.data() + ref_offset,
    //             1,
    //             seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
    //             image_payload->image_patch__element_per_patch[i].first, Padded_GEMMA4E_VISION_HIDDEN_SIZE

    //     // //TODO: FIXME:
    //     // for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
    //     //     memcpy(
    //     //         patch_emb_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
    //     //         Gemma4VisionPatchEmbedder_hidden_states.data() + ref_offset,

    //     //         seq_len_per_image[i] *  this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16)

    // #endif

    {
        // now the _position_embeddings

        ///torch.Size([3, 2, 2520, 10240])
        // first, zero out all one_shot_buffer  [2, seq_len_padded, GEMMA4E_POSITION_EMBEDDING_SIZE]
        memset(one_shot_buffer.data(), 0, one_shot_buffer.size()*sizeof(bf16));

        bf16* one_shot_buffer_x_base = (bf16*)one_shot_buffer.data();
        bf16* one_shot_buffer_y_base = one_shot_buffer_x_base + seq_len_padded* this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE;

        for(int i = 0; i < image_payload->num_images; i++){
            bf16* x_ptr = one_shot_buffer_x_base + start_seq_len_index_per_image[i] * this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE;
            bf16* y_ptr = one_shot_buffer_y_base + start_seq_len_index_per_image[i] * this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE;

            for(int s = 0; s< seq_len_per_image[i]; s++){
                auto x_val = image_payload->image_grid_pairs_per_image[i][s*2];
                auto y_val = image_payload->image_grid_pairs_per_image[i][s*2 + 1];

                x_ptr[s * this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE + x_val] = 1.0f;
                y_ptr[s * this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE + y_val] = 1.0f;
            }
        }
    }
    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {

            bf16* one_shot_buffer_x_ptr = (bf16*)one_shot_buffer.data();
            bf16* one_shot_buffer_y_ptr = one_shot_buffer_x_ptr + seq_len_padded* this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE;

            std::cout << "Error  for Gemma4VisionPatchEmbedder_one_hot_positions:" << std::endl;
            buffer<bf16> Gemma4VisionPatchEmbedder_one_hot_positions;
            reference_tensor.load_weights(
                Gemma4VisionPatchEmbedder_one_hot_positions,  // shape of [num_image, 2, seq_len_per_image, GEMMA4E_POSITION_EMBEDDING_SIZE]
                "Gemma4VisionPatchEmbedder_one_hot_positions"
            );

            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                print_error_metrics<bf16, bf16>(
                    one_shot_buffer_x_ptr + start_seq_len_index_per_image[i]*this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE,
                    Gemma4VisionPatchEmbedder_one_hot_positions.data() + ref_offset,
                    1,
                    seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE,
                    image_payload->image_patch__element_per_patch[i].first, this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE

                );

                print_error_metrics<bf16, bf16>(
                    one_shot_buffer_y_ptr + start_seq_len_index_per_image[i]*this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE,
                    Gemma4VisionPatchEmbedder_one_hot_positions.data() + ref_offset + image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE ,
                    1,
                    seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE,
                    image_payload->image_patch__element_per_patch[i].first, this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE

                );

                ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_POSITION_EMBEDDING_SIZE*2; // times 2 is for x and y;
            }
        }

    #endif

    // now, we do the
    one_shot_buffer.sync_to_device();
    patch_embedder_position_embedding_table.sync_to_device();
    patch_embedder_posisiton_embedding_dim_0_app(one_shot_buffer, patch_embedder_position_embedding_table, position_embedding_table_output);
    position_embedding_table_output.sync_from_device();

    // dim2
    one_shot_buffer.sync_to_device();
    patch_embedder_position_embedding_table.sync_to_device();
    patch_embedder_posisiton_embedding_dim_1_app(one_shot_buffer, patch_embedder_position_embedding_table, position_embedding_table_output);
    position_embedding_table_output.sync_from_device();

    // now, compare with the python reference
    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {

        bf16* position_embedding_table_output_x_ptr = (bf16*)position_embedding_table_output.data();
        bf16* position_embedding_table_output_y_ptr = position_embedding_table_output_x_ptr + seq_len_padded* Padded_GEMMA4E_VISION_HIDDEN_SIZE;

        std::cout << "Error Gemma4VisionPatchEmbedder_position_embeddings_before_sum:" << std::endl;
        buffer<bf16> Gemma4VisionPatchEmbedder_position_embeddings_before_sum;
        reference_tensor.load_weights(
            Gemma4VisionPatchEmbedder_position_embeddings_before_sum,  // shape of [num_image, 2, seq_len_per_image, GEMMA4E_VISION_HIDDEN_SIZE] (unpadded)
            "Gemma4VisionPatchEmbedder_position_embeddings_before_sum"
        );

        for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
            const int patches = image_payload->image_patch__element_per_patch[i].first;
            const int ref_hidden = this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // Python ref is unpadded [seq_len, 768]

            print_error_metrics<bf16, bf16>(
                position_embedding_table_output_x_ptr + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                Gemma4VisionPatchEmbedder_position_embeddings_before_sum.data() + ref_offset,
                1,
                seq_len_per_image[i], ref_hidden,          // cols to compare = ref row stride (unpadded)
                patches, Padded_GEMMA4E_VISION_HIDDEN_SIZE // cpp row stride (padded)
            );

            print_error_metrics<bf16, bf16>(
                position_embedding_table_output_y_ptr + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                Gemma4VisionPatchEmbedder_position_embeddings_before_sum.data() + ref_offset + patches * ref_hidden,
                1,
                seq_len_per_image[i], ref_hidden,          // cols to compare = ref row stride (unpadded)
                patches, Padded_GEMMA4E_VISION_HIDDEN_SIZE // cpp row stride (padded)
            );

            ref_offset += patches * ref_hidden * 2; // times 2 for x and y, use unpadded ref stride
        }
    }
    #endif

    hidden_state.resize(seq_len_padded * Padded_GEMMA4E_VISION_HIDDEN_SIZE);
    memset(hidden_state.data(), 0, hidden_state.size() * sizeof(bf16));

    // now, we sum the two dim
    //TODO: use avx512 for it later

    {

        bf16* position_embedding_table_output_x_ptr = (bf16*)position_embedding_table_output.data();
        bf16* position_embedding_table_output_y_ptr = position_embedding_table_output_x_ptr + seq_len_padded* Padded_GEMMA4E_VISION_HIDDEN_SIZE;
        for(int i = 0; i < seq_len* Padded_GEMMA4E_VISION_HIDDEN_SIZE; i++){
            hidden_state[i] = position_embedding_table_output_x_ptr[i] + position_embedding_table_output_y_ptr[i] + patch_emb_output[i];
        }
    }

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "Error for vision_inputs_embeds_after_patch_embedder:" << std::endl;
        buffer<bf16> vision_inputs_embeds_after_patch_embedder;
        reference_tensor.load_weights(
            vision_inputs_embeds_after_patch_embedder,  // shape of [seq_len_padded, GEMMA4E_VISION_HIDDEN_SIZE]
            "vision_inputs_embeds_after_patch_embedder"
        );

        auto ref_hidden_state = this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // Python ref is unpadded [seq_len, 768]
        for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
            print_error_metrics<bf16, bf16>(
                hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                vision_inputs_embeds_after_patch_embedder.data() + ref_offset,
                1,
                seq_len_per_image[i],ref_hidden_state,
                image_payload->image_patch__element_per_patch[i].first, Padded_GEMMA4E_VISION_HIDDEN_SIZE

            );
            ref_offset+= image_payload->image_patch__element_per_patch[i].first*ref_hidden_state; // use unpadded ref stride
        }

        // //TODO: fixme
        //         //TODO: FIXME:
        // for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
        //     memcpy(
        //         hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
        //         vision_inputs_embeds_after_patch_embedder.data() + ref_offset,

        //         seq_len_per_image[i] * Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16)

        //     );
        //     ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
        // }
    }

    #endif

    // generate the rope,
    std::vector<bf16> cos_emb;
    std::vector<bf16> sin_emb;

    generate_gemma4_vision_rotary_pos_emb(
        image_payload->image_grid_pairs_per_image,
        seq_len_per_image,
        start_seq_len_index_per_image,
        seq_len_padded,
        (int)this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM,
        this->parent_npu_ptr->GEMMA4E_ROPE_THETA,
        1.0f,  // attention_scaling = 1.0
        cos_emb,
        sin_emb
    );

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        const int head_dim = (int)this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM;

        buffer<bf16> ref_cos, ref_sin;
        reference_tensor.load_weights(ref_cos, "Gemma4VisionEncoder_position_embeddings_cos");
        reference_tensor.load_weights(ref_sin, "Gemma4VisionEncoder_position_embeddings_sin");

        // ref shape is [num_images, ref_seq_per_image, head_dim] — derive per-image stride from total size
        const int ref_seq_per_image = (int)ref_cos.size() / ((int)image_payload->num_images * head_dim);
        std::cout << "ref_seq_per_image" << ref_seq_per_image << std::endl;

        std::cout << "Error for Gemma4VisionEncoder_position_embeddings_cos:" << std::endl;
        for (int i = 0; i < (int)image_payload->num_images; i++) {
            print_error_metrics<bf16, bf16>(
                cos_emb.data() + start_seq_len_index_per_image[i] * head_dim,
                ref_cos.data()  + i * ref_seq_per_image * head_dim,
                1,
                seq_len_per_image[i], head_dim,  // rows/cols to compare; ref row stride = head_dim
                seq_len_per_image[i], head_dim   // cpp cos_emb row stride = head_dim (no padding)
            );
        }

        std::cout << "Error for Gemma4VisionEncoder_position_embeddings_sin:" << std::endl;
        for (int i = 0; i < (int)image_payload->num_images; i++) {
            print_error_metrics<bf16, bf16>(
                sin_emb.data() + start_seq_len_index_per_image[i] * head_dim,
                ref_sin.data()  + i * ref_seq_per_image * head_dim,
                1,
                seq_len_per_image[i], head_dim,
                seq_len_per_image[i], head_dim
            );
        }
    }
    #endif

    residual_buffer.resize(seq_len_padded * Padded_GEMMA4E_VISION_HIDDEN_SIZE);
    memset(residual_buffer.data(), 0, residual_buffer.size() * sizeof(bf16));
    //TODO: FIXME:
    for(int layer_idx = 0; layer_idx < this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS; layer_idx++){
        memcpy(residual_buffer.data(), hidden_state.data(), seq_len* Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16));

        // first, comparew with f"Gemma4VisionEncoderLayer_{layer_idx}_initial_hidden_states"]
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" << layer_idx << "_initial_hidden_states:" << std::endl;
            buffer<bf16> ref_initial_hidden_states;
            reference_tensor.load_weights(
                ref_initial_hidden_states,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_initial_hidden_states"
            );

        //     for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
        //         print_error_metrics<bf16, bf16>(
        //             hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
        //             ref_initial_hidden_states.data() + ref_offset,
        //             1,
        //             seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
        //             seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

        //         );
        //         ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
        //     }
        }
        #endif

        // do the first RMS norm on hidden_states

        simd_rms_norm(
            hidden_state.data(),
            this->layer_norm_1_weight[layer_idx].data(),
            hidden_state.data(),
            seq_len, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
            seq_len_padded, this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        // compare with "Gemma4VisionEncoderLayer_{layer_idx}_initial_hidden_states" in the reference for the result
        // #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        // {
        //     std::cout << "Error for Gemma4VisionEncoderLayer_" << layer_idx << "_post_input_layernorm_hidden_states:" << std::endl;
        //     buffer<bf16> ref_post_input_layernorm_hidden_states;
        //     reference_tensor.load_weights(
        //         ref_post_input_layernorm_hidden_states,
        //         "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_post_input_layernorm_hidden_states"
        //     );
        // }
        // #endif
        // #ifdef DEBUG_PRINT_ENCODE_ERROR_METRICS
        // {
        //     std::cout << "Error for Gemma4VisionEncoderLayer_" << layer_idx << "_post_input_layernorm_hidden_states:" << std::endl;
        //     buffer<bf16> ref_post_input_layernorm_hidden_states;
        //     reference_tensor.load_weights(
        //         ref_post_input_layernorm_hidden_states,
        //         "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_post_input_layernorm_hidden_states"
        //     );

        //     for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
        //         print_error_metrics<bf16, bf16>(
        //             hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
        //             ref_post_input_layernorm_hidden_states.data() + ref_offset,
        //             1,
        //             seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
        //             seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

        //         );
        //         ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
        //     }
        // }
        // #endif

            generate_mm_sequence<bf16, bf16>(*this->q_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0,
                1,  (float)this->output_q_min[layer_idx],(float) this->output_q_max[layer_idx], // clamp output to quantization range for q_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );

            simd_clamp(
                hidden_state.data(),
                q_projection_input.data(),
                this->input_q_min[layer_idx], this->input_q_max[layer_idx],
                seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
            );
            memset(q_projection_input.data() + seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE, 0, (seq_len_padded - seq_len)* Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16));
            q_projection_input.sync_to_device();

        auto q_proj_run = this->q_proj_app.create_run(
            q_projection_input, q_proj_weight[layer_idx], q_projection_output
        );

        q_projection_input.sync_to_device();
        this->q_proj_weight[layer_idx].sync_to_device();
        q_proj_run.start();

            generate_mm_sequence<bf16, bf16>(*this->k_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0,
                1,  (float)this->output_k_min[layer_idx],(float) this->output_k_max[layer_idx], // clamp output to quantization range for k_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );
            simd_clamp(
                hidden_state.data(),
                k_projection_input.data(),
                this->input_k_min[layer_idx], this->input_k_max[layer_idx],
                seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
            );
            memset(k_projection_input.data() + seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE, 0, (seq_len_padded - seq_len)* Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16));
            k_projection_input.sync_to_device();

        q_proj_run.wait();
        q_projection_output.sync_from_device();

        k_projection_input.sync_to_device();
        this->k_proj_weight[layer_idx].sync_to_device();
        auto k_proj_run = this->k_proj_app.create_run(
            k_projection_input, k_proj_weight[layer_idx], k_projection_output
        );
        k_proj_run.start();

            generate_mm_sequence<bf16, bf16>(*this->v_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0,
                1,  (float)this->output_v_min[layer_idx],(float) this->output_v_max[layer_idx], // clamp output to quantization range for v_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );
            simd_clamp(
                hidden_state.data(),
                v_projection_input.data(),
                this->input_v_min[layer_idx], this->input_v_max[layer_idx],
                seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
            );
            memset(v_projection_input.data() + seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE, 0, (seq_len_padded - seq_len)* Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16));
            v_projection_input.sync_to_device();

        k_proj_run.wait();
        k_projection_output.sync_from_device();

        v_projection_input.sync_to_device();
        this->v_proj_weight[layer_idx].sync_to_device();
        auto v_proj_run = this->v_proj_app.create_run(
            v_projection_input, v_proj_weight[layer_idx], v_projection_output
        );
        v_proj_run.start();
            // apply norm for q, and k
            simd_rms_norm(
                q_projection_output.data(),
                this->q_norm_weight[layer_idx].data(),
                q_projection_output.data(),
                seq_len    * (this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE / this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM),this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM,
                seq_len    * (this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE / this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM),this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );
            apply_multidimensional_rope(
                q_projection_output.data(), cos_emb.data(), sin_emb.data(),
                seq_len, this->parent_npu_ptr->GEMMA4E_VISION_NUM_ATTENTION_HEADS, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM);
            q_projection_output.sync_to_device();
            simd_rms_norm(
                k_projection_output.data(),
                this->k_norm_weight[layer_idx].data(),
                k_projection_output.data(),
                seq_len    * (this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE / this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM),this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM,
                seq_len    * (this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE / this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM),this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );

            apply_multidimensional_rope(
                k_projection_output.data(), cos_emb.data(), sin_emb.data(),
                seq_len, this->parent_npu_ptr->GEMMA4E_VISION_NUM_ATTENTION_HEADS, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM);
            k_projection_output.sync_to_device();

        v_proj_run.wait();
        v_projection_output.sync_from_device();

        // apply norm for v
        simd_rms_norm(
            v_projection_output.data(),
            v_projection_output.data(),
            seq_len    * (this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE / this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM),this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM,
            seq_len    * (this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE / this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM),this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
        );
        v_projection_output.sync_to_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionAttention_" + std::to_string(layer_idx)+ "_query_states_after_rope:" << std::endl;
            buffer<bf16> ref_q_projection_output;
            reference_tensor.load_weights(
                ref_q_projection_output,
                "Gemma4VisionAttention_" + std::to_string(layer_idx) + "_query_states_after_rope"
            );

            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                print_error_metrics<bf16, bf16>(
                    q_projection_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                    ref_q_projection_output.data() + ref_offset,
                    1,
                    seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                    seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                );
                ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
            std::cout << "Error for Gemma4VisionAttention_" + std::to_string(layer_idx)+ "_key_states_after_rope:" << std::endl;
            buffer<bf16> ref_k_projection_output;
            reference_tensor.load_weights(
                ref_k_projection_output,
                "Gemma4VisionAttention_" + std::to_string(layer_idx) + "_key_states_after_rope"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        k_projection_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref_k_projection_output.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }

            std::cout << "Error for Gemma4VisionAttention_" + std::to_string(layer_idx)+ "_value_states_after_norm:" << std::endl;
            buffer<bf16> ref_v_projection_output;
            reference_tensor.load_weights(
                ref_v_projection_output,
                "Gemma4VisionAttention_" + std::to_string(layer_idx) + "_value_states_after_norm"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        v_projection_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref_v_projection_output.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif

        auto attention_run = this->flash_attention_app.create_run(
            attention_output, q_projection_output, k_projection_output, v_projection_output
        );
        q_projection_output.sync_to_device();
        k_projection_output.sync_to_device();
        v_projection_output.sync_to_device();
        attention_run.start();

            generate_mm_sequence<bf16, bf16>(*this->o_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0,
                1,  (float)this->output_o_min[layer_idx],(float) this->output_o_max[layer_idx], // clamp output to quantization range for v_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );

        attention_run.wait();
        attention_output.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionAttention_" + std::to_string(layer_idx)+ "_attention_output:" << std::endl;
            buffer<bf16> ref_attention_output;
            reference_tensor.load_weights(
                ref_attention_output,
                "Gemma4VisionAttention_" +  std::to_string(layer_idx)  +"_attn_output_before_o_proj"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        attention_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref_attention_output.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }

            // //TOOD: FIXME: remove later
            // for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
            //     memcpy(
            //         attention_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
            //         ref_attention_output.data() + ref_offset,
            //         seq_len_per_image[i]* this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16)

            //     );
            //     ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            // }
        }

        #endif

        simd_clamp(
            attention_output.data(),
            attention_output.data(),
            this->input_o_min[layer_idx], this->input_o_max[layer_idx],
            seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );

        attention_output.sync_to_device();
        this->o_proj_weight[layer_idx].sync_to_device();
        o_proj_app(attention_output, this->o_proj_weight[layer_idx], o_projection_output );
        o_projection_output.sync_from_device();

        simd_rms_norm(
            o_projection_output.data(),
            this->post_o_norm_weight[layer_idx].data(),
            o_projection_output.data(),
            seq_len, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
            seq_len_padded, this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_post_attention_layernorm_hidden_states:" << std::endl;
            buffer<bf16> ref__post_attention_layernorm_hidden_states;
            reference_tensor.load_weights(
                ref__post_attention_layernorm_hidden_states,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_post_attention_layernorm_hidden_states"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        o_projection_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref__post_attention_layernorm_hidden_states.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif

        o_projection_output.sync_to_device();

        simd_add(
            o_projection_output.data(),
            residual_buffer.data(),
            hidden_state.data(),
            seq_len * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );

        // copy to residual buffer
        memcpy(residual_buffer.data(), hidden_state.data(), seq_len* Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16));

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_hidden_states_after_attention_residual:" << std::endl;
            buffer<bf16> ref__hidden_states_after_attention_residual;
            reference_tensor.load_weights(
                ref__hidden_states_after_attention_residual,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_hidden_states_after_attention_residual"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref__hidden_states_after_attention_residual.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for pre mlp norm weight" << std::endl;
            //Gemma4VisionEncoderLayer_{layer_idx}_pre_feedforward_layernorm_weights
            buffer<bf16> ref_pre_feedforward_layernorm_weights;
            reference_tensor.load_weights(
                ref_pre_feedforward_layernorm_weights,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_pre_feedforward_layernorm_weights"
            );
            print_error_metrics<bf16, bf16>(
                this->layer_norm_2_weight[layer_idx].data(),
                ref_pre_feedforward_layernorm_weights.data(),
                1,
                1, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                1, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE
            );
        }
        #endif

        simd_rms_norm(
            hidden_state.data(),
            this->layer_norm_2_weight[layer_idx].data(),
            hidden_state.data(),
            seq_len, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
            seq_len_padded, this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_post_pre_feedforward_layernorm_hidden_states:" << std::endl;
            buffer<bf16> ref_post_pre_feedforward_layernorm_hidden_states;
            reference_tensor.load_weights(
                ref_post_pre_feedforward_layernorm_hidden_states,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_post_pre_feedforward_layernorm_hidden_states"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref_post_pre_feedforward_layernorm_hidden_states.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif

            simd_clamp(
                hidden_state.data(),
                gate_input.data(),
                this->input_gate_min[layer_idx], this->input_gate_max[layer_idx],
                seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
            );
            gate_input.sync_to_device();

            generate_mm_sequence<bf16, bf16>(*this->gate_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 1, // gelu for gate
                1,  (float)this->output_gate_min[layer_idx],(float) this->output_gate_max[layer_idx], // clamp output to quantization range for v_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );

        gate_input.sync_to_device();
        this->gate_proj_weight[layer_idx].sync_to_device();
        auto gate_proj_run = this->gate_proj_app.create_run(
            gate_input, gate_proj_weight[layer_idx], gate_output
        );
        gate_proj_run.start();

            simd_clamp(
                hidden_state.data(),
                up_input.data(),
                this->input_up_min[layer_idx], this->input_up_max[layer_idx],
                seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
            );
            up_input.sync_to_device();
            generate_mm_sequence<bf16, bf16>(*this->up_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false,  0,
                1,  (float)this->output_up_min[layer_idx],(float) this->output_up_max[layer_idx], // clamp output to quantization range for v_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );

        gate_proj_run.wait();
        gate_output.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_gate_proj_act:" << std::endl;
            buffer<bf16> ref_gate_proj_output;
            reference_tensor.load_weights(
                ref_gate_proj_output,
                "Gemma4VisionMLP_layer_" + std::to_string(layer_idx) + "_gate_proj_act"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        gate_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE,
                        ref_gate_proj_output.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE; // use unpadded ref stride
            }
        }
        #endif

        up_input.sync_to_device();
        this->up_proj_weight[layer_idx].sync_to_device();
        auto up_proj_run = this->up_proj_app.create_run(
            up_input, up_proj_weight[layer_idx], up_output
        );
        up_proj_run.start();

            generate_mm_sequence<bf16, bf16>(*this->down_proj_app.seq(),
                seq_len_padded, Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE, Padded_GEMMA4E_VISION_HIDDEN_SIZE  ,
                MM_tile_M,MM_tile_K,MM_tile_N,
                8,8,8,
                rtp_address, rtp_sync_lock_id,
                MM_ROW_SIZE,MM_COL_SIZE,
                0,0,0,

                IS_B_ROW_MAJOR, ENABLE_AXI4, true,
                false, 0,
                1,  (float)this->output_down_min[layer_idx],(float) this->output_down_max[layer_idx], // clamp output to quantization range for v_proj
                ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
            );

        up_proj_run.wait();
        up_output.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_up_proj_output:" << std::endl;
            buffer<bf16> ref_up_proj_output;
            reference_tensor.load_weights(
                ref_up_proj_output,
                "Gemma4VisionMLP_layer_" + std::to_string(layer_idx) + "_up_proj"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        up_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE,
                        ref_up_proj_output.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE; // use unpadded ref stride
            }
        }
        #endif

        simd_mul(
            gate_output.data(),
            up_output.data(),
            gate_output.data(), // write back to gate_output buffer to save memory
            seq_len * Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_after_act:" << std::endl;
            buffer<bf16> ref_post_gate_mul_hidden_states;
            reference_tensor.load_weights(
                ref_post_gate_mul_hidden_states,
                "Gemma4VisionMLP_layer_" + std::to_string(layer_idx) + "_after_act"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        gate_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE,
                        ref_post_gate_mul_hidden_states.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_MLP_INTERMEDIATE_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_INTERMEDIATE_SIZE; // use unpadded ref stride
            }
        }
        #endif

        simd_clamp(
            gate_output.data(),
            gate_output.data(),
            this->input_down_min[layer_idx], this->input_down_max[layer_idx],
            seq_len * Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );

        gate_output.sync_to_device();
        this->down_proj_weight[layer_idx].sync_to_device();
        this->down_proj_app(gate_output, this->down_proj_weight[layer_idx], down_output);
        down_output.sync_from_device();

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_post_mlp_hidden_states:" << std::endl;
            buffer<bf16> ref_post_mlp_hidden_states;
            reference_tensor.load_weights(
                ref_post_mlp_hidden_states,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_post_mlp_hidden_states"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        down_output.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref_post_mlp_hidden_states.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif

        simd_rms_norm(
            down_output.data(),
            this->post_ffn_norm_weight[layer_idx].data(),
            hidden_state.data(),
            seq_len, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
            seq_len_padded, this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );
        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_post_feedforward_layernorm_hidden_states:" << std::endl;
            buffer<bf16> ref_post_ffn_layernorm_hidden_states;
            reference_tensor.load_weights(
                ref_post_ffn_layernorm_hidden_states,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_post_feedforward_layernorm_hidden_states"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref_post_ffn_layernorm_hidden_states.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif

        simd_add(
            hidden_state.data(),
            residual_buffer.data(),
            hidden_state.data(),
            seq_len * this->Padded_GEMMA4E_VISION_HIDDEN_SIZE
        );

        #if DEBUG_PRINT_ENCODE_ERROR_METRICS
        {
            std::cout << "Error for Gemma4VisionEncoderLayer_" + std::to_string(layer_idx)+ "_final_hidden_states:" << std::endl;
            buffer<bf16> ref__hidden_states_after_ffn_residual;
            reference_tensor.load_weights(
                ref__hidden_states_after_ffn_residual,
                "Gemma4VisionEncoderLayer_" + std::to_string(layer_idx) + "_final_hidden_states"
            );
            for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
                    print_error_metrics<bf16, bf16>(
                        hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
                        ref__hidden_states_after_ffn_residual.data() + ref_offset,
                        1,
                        seq_len_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                        seq_len_per_image[i], Padded_GEMMA4E_VISION_HIDDEN_SIZE

                    );
                    ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
            }
        }
        #endif
    }

    // //TODO: FIXME: DEBUG
    // {

    //     buffer<bf16> ref__hidden_states_after_ffn_residual;
    //     reference_tensor.load_weights(
    //         ref__hidden_states_after_ffn_residual,
    //         "Gemma4VisionEncoderLayer_" + std::to_string(this->parent_npu_ptr->GEMMA4E_VISION_NUM_HIDDEN_LAYERS-1) + "_final_hidden_states"
    //     );
    //     for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
    //         memcpy(
    //             hidden_state.data() + start_seq_len_index_per_image[i]*Padded_GEMMA4E_VISION_HIDDEN_SIZE,
    //             ref__hidden_states_after_ffn_residual.data() + ref_offset,
    //             seq_len_per_image[i] * this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16)
    //         );
    //         ref_offset+= image_payload->image_patch__element_per_patch[i].first*this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE; // use unpadded ref stride
    //     }

    // }

    // // debug, print all the content in seq_len_per_image

    // the vision pooler stage
    std::vector<int> k_per_image;
    std::vector<int> k_squared_per_image;
    for(int i = 0; i < image_payload->num_images; i++){

        int k = std::sqrt(   seq_len_per_image[i]/  image_payload->num_soft_tokens_per_image[i]   );
        k_per_image.push_back(k);
        k_squared_per_image.push_back(k*k);
    }

    std::vector<int> max_x;

    for(int i = 0; i < image_payload->num_images; i++){
        int cur_max_x = -1;
        for(int j = 0; j < seq_len_per_image[i]; j++){

            auto cur_x = image_payload->image_grid_pairs_per_image[i][2*j];
            if(cur_x > cur_max_x){
                cur_max_x = cur_x;
            }
        }
        max_x.push_back(cur_max_x + 1);
    }

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {

        std::cout << "compare max x" <<std::endl;
        buffer<int64_t> Gemma4VisionPooler_max_x;
        reference_tensor.load_weights(
            Gemma4VisionPooler_max_x,
            "Gemma4VisionPooler_max_x"
        );
        for(int i = 0; i < image_payload->num_images; i++){
            if(max_x[i] != Gemma4VisionPooler_max_x.data()[i]){
                std::cout << "max_x mismatch for image " << i << ": " << max_x[i] << " vs ref " << Gemma4VisionPooler_max_x.data()[i] << std::endl;
            }
        }
    }
    #endif

    // now, we divide every image's image_grid_pairs_per_image by k (floor division), matching Python: kernel_idxs = floor(pos / k)
    std::vector<std::vector<int>> kernel_idx(image_payload->num_images);
    for(int i = 0; i < image_payload->num_images; i++){

        for(int j = 0; j < seq_len_per_image[i]; j++){
            image_payload->image_grid_pairs_per_image[i][2*j] /= (float)k_per_image[i]; //NOTE: store back to int, same as round_down in floor mode
            image_payload->image_grid_pairs_per_image[i][2*j + 1] /= (float)k_per_image[i];

            kernel_idx.at(i).push_back(
                image_payload->image_grid_pairs_per_image[i][2*j]  + (max_x[i] /k_per_image[i] ) *  image_payload->image_grid_pairs_per_image[i][2*j+1]
            );
        }
    }

    // now, compare the error of kernel_idx with "Gemma4VisionPooler_kernel_idxs"

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "compare kernel idx" <<std::endl;
        buffer<int64_t> Gemma4VisionPooler_kernel_idxs;
        reference_tensor.load_weights(
            Gemma4VisionPooler_kernel_idxs,
            "Gemma4VisionPooler_kernel_idxs"
        );

        int ref_per_image = Gemma4VisionPooler_kernel_idxs.size() / image_payload->num_images;
        for(int i = 0, ref_offset=0; i < image_payload->num_images; i++){
            for(int j = 0; j < seq_len_per_image[i]; j++){
                if(kernel_idx[i][j] != Gemma4VisionPooler_kernel_idxs.data()[ref_offset + j]){
                    std::cout << "kernel_idx mismatch for image " << i << " token " << j << ": " << kernel_idx[i][j] << " vs ref " << Gemma4VisionPooler_kernel_idxs.data()[ref_offset + j] << std::endl;
                }
            }
            ref_offset+=ref_per_image;
        }
    }
    #endif

    //NOTE: num_soft_token_per_image is in image_payload->num_soft_tokens_per_image[i];
    // L_per_image is in seq_len_per_image[i]
    std::vector<bf16> pooling_output;
    //hidden_state is a row-major buffer
    // at this point, the hidden_state is in shape of [num_image,seq_len_per_image[], Padded_GEMMA4E_VISION_HIDDEN_SIZE], we will do pooling for each image separately, and the pooling weight will be generated based on kernel_idx and k_squared_per_image (which is the number of tokens in each pooling region), and the pooling weight will be shape of [num_image, seq_len_per_image, num_soft_token_per_image]
    //NOTE: seq_len_per_image[] means this varies from image to image, and num_soft_token_per_image is in image_payload->num_soft_tokens_per_image[i]

    //NOTE: the python code generates a matrix that is just too sparse, we use scatter-add instead
    // output = weights.transpose(1, 2) @ hidden_states.float()
    // equivalently: output[t, h] = (1/k²) * Σ hidden_states[l, h]  for all l where kernel_idx[l] == t

    float root_hidden_size = std::sqrt((float)this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE);
    for(int i = 0; i < image_payload->num_images; i++){
        const int cur_seq_len = seq_len_per_image[i];
        const int cur_soft_tokens = image_payload->num_soft_tokens_per_image[i];
        const int HIDDEN_SIZE = this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE;
        const float scale = 1.0f / (float)k_squared_per_image[i];

        // accumulate in float for precision
        std::vector<float> accum(cur_soft_tokens * HIDDEN_SIZE, 0.0f);

        bf16* hidden_base = hidden_state.data() + start_seq_len_index_per_image[i] * Padded_GEMMA4E_VISION_HIDDEN_SIZE;
        for(int l = 0; l < cur_seq_len; l++){
            const int t = kernel_idx[i][l];
            bf16* src = hidden_base + l * Padded_GEMMA4E_VISION_HIDDEN_SIZE;
            float* dst = accum.data() + t * HIDDEN_SIZE;
            for(int h = 0; h < HIDDEN_SIZE; h++){
                dst[h] += (float)src[h] * scale;
            }
        }

        // convert back to bf16
        int cur_pooling_output_size = pooling_output.size();
        pooling_output.resize(cur_pooling_output_size + cur_soft_tokens * Padded_GEMMA4E_VISION_HIDDEN_SIZE, bf16(0));
        for(int t = 0; t < cur_soft_tokens; t++){
            for(int h = 0; h < HIDDEN_SIZE; h++){
                // pooling_output[i][t * Padded_GEMMA4E_VISION_HIDDEN_SIZE + h] = bf16(accum[t * HIDDEN_SIZE + h]);
                pooling_output[cur_pooling_output_size + t * Padded_GEMMA4E_VISION_HIDDEN_SIZE + h] = bf16(accum[t * HIDDEN_SIZE + h] * root_hidden_size ); // add a scaling factor to prevent overflow, matching the implementation in python code
            }
        }
    }

    size_t vision_output_token_size = pooling_output.size() / this->Padded_GEMMA4E_VISION_HIDDEN_SIZE;
    size_t padded_vision_output_token_size = round_up_to_multiple(vision_output_token_size, seq_len_pad_requirement_for_MM);
    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "compare Gemma4VisionPooler_hidden_states_after_scaling" << std::endl;
        buffer<bf16> ref_pooling_output;
        reference_tensor.load_weights(
            ref_pooling_output,
            "Gemma4VisionPooler_hidden_states_after_scaling"
        );

        int ref_per_image = ref_pooling_output.size() / image_payload->num_images;
        for(int i = 0,pool_offset=0, ref_offset=0; i < image_payload->num_images; i++){
            print_error_metrics<bf16, bf16>(
                pooling_output.data() + pool_offset,
                ref_pooling_output.data() + ref_offset,
                1,
                image_payload->num_soft_tokens_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
                image_payload->num_soft_tokens_per_image[i], this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE
            );
            ref_offset += ref_per_image;
            pool_offset +=  image_payload->num_soft_tokens_per_image[i] * Padded_GEMMA4E_VISION_HIDDEN_SIZE;
        }
    }
    #endif

    buffer<bf16> language_embed_input = this->vision_to_language_input_projection_app.create_bo_buffer<bf16>(
        padded_vision_output_token_size* Padded_GEMMA4E_VISION_HIDDEN_SIZE
    );
    memset(language_embed_input.data(), 0, padded_vision_output_token_size* Padded_GEMMA4E_VISION_HIDDEN_SIZE * sizeof(bf16)); // zero padding for padded tokens

    buffer<bf16> language_embed_output = this->vision_to_language_input_projection_app.create_bo_buffer<bf16>(
        padded_vision_output_token_size * this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE
    );

    simd_rms_norm(
        pooling_output.data(),
        language_embed_input.data(),
        vision_output_token_size, this->parent_npu_ptr->GEMMA4E_VISION_HIDDEN_SIZE,
        padded_vision_output_token_size, this->Padded_GEMMA4E_VISION_HIDDEN_SIZE

    );

    generate_mm_sequence<bf16, bf16>(*this->vision_to_language_input_projection_app.seq(),
        padded_vision_output_token_size, Padded_GEMMA4E_VISION_HIDDEN_SIZE,this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE,  //TODO: FIXME:
        MM_tile_M,MM_tile_K,MM_tile_N,
        8,8,8,
        rtp_address, rtp_sync_lock_id,
        MM_ROW_SIZE,MM_COL_SIZE,
        0,0,0,
        IS_B_ROW_MAJOR, ENABLE_AXI4, true,
        false, 0, /// no biase
        0,  -10000.0, 1000000.0, // do not clamp on output
        ENABLE_QKV_REORDER, this->parent_npu_ptr->GEMMA4E_VISION_HEAD_DIM
    );

    language_embed_input.sync_to_device();
    this->vision_to_language_input_projection_weight.sync_to_device();
    this->vision_to_language_input_projection_app(language_embed_input, this->vision_to_language_input_projection_weight, language_embed_output);
    language_embed_output.sync_from_device();

    #if DEBUG_PRINT_ENCODE_ERROR_METRICS
    {
        std::cout << "compare vision to language projection output" << std::endl;
        buffer<bf16> ref_vision_to_language_projection_output;
        reference_tensor.load_weights(
            ref_vision_to_language_projection_output,
            "vision_final_embs_after_embed_vision"
        );
        std::cout << "finished laoding ref vision to language projection output" << std::endl;
        print_error_metrics<bf16, bf16>(
            language_embed_output.data(),
            ref_vision_to_language_projection_output.data(),
            1,
            vision_output_token_size, this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE,
            padded_vision_output_token_size, this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE

        );
    }
    #endif

    ///TODO: consider change the output type interface
    std::vector<bf16> final_res(padded_vision_output_token_size *this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE );
    memcpy(
        final_res.data(),
        language_embed_output.data(),
        padded_vision_output_token_size * this->parent_npu_ptr->GEMMA4E_VISION_IMAGE_OUTPUT_SIZE * sizeof(bf16)
    );

    return final_res;
}
