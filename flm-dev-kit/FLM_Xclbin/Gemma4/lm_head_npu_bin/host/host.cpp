// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "typedef.hpp"
#include "buffer.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/debug_utils.hpp"
#include "utils/vm_args.hpp"
#include "utils/utils.hpp"
#include "metrices.hpp"
#include "tensor_2d.hpp"
#include "lm_config.hpp"

void generate_seq(npu_sequence* seq, int VOCAB_SIZE, int D, float final_scale);
void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks);

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("LLAMA_3_2_1B"), "Model type, e.g., LLAMA_3_2_1B");

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();

    header_print("info", "Testing model: " + model_type);

    std::string model_path     = "./q4nx_files/Gemma4-E2B-NPU2";
    header_print("info", "Model path: " + model_path);
    LM_Config lm_config;
    lm_config.from_pretrained(model_path);

    size_t D          = lm_config.hidden_size;
    size_t VOCAB_SIZE = lm_config.vocab_size;
    size_t M_PER_ROUND = 32 * 32;
    size_t VOCAB_SIZE_PADDED = (VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;
    // overide the layer xclbin

    lm_config.lm_head_xclbin_name = "./build/GEMMA4_E2B/xclbins/lm_head.xclbin";
    header_print("info", "Layer xclbin path: " + lm_config.lm_head_xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(D));
    header_print("info", "Vocab size: " + std::to_string(VOCAB_SIZE));


    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(model_path);

    buffer<bf16> embed;

    q4nx_model->load_weights(embed, "model.embed_tokens.weight");

    header_print("info", "Embedding weights:");
    utils::print_matrix(embed, D, 10);
    header_print("info", "Testing decoding layer...");
    header_print("info", "Initializing NPU manager...");
    
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* lm_head_app_mgr = xclbin_mgr.register_xclbin(lm_config.lm_head_xclbin_name);

    npu_app lm_head = lm_head_app_mgr->create_app();
    header_print("info", "Creating NPU sequence...");

    srand(time(NULL));
    int test_size = 29;
    int* input_ids = new int[test_size];
    srand(0);
    for (int i = 0; i < test_size; i++){
        input_ids[i] = rand() % VOCAB_SIZE;
    }

    buffer<bf16> x(D * test_size);

    for (int i = 0; i < test_size; i++){
        memcpy(x.data() + i * D, embed.data() + input_ids[i] * D, D * sizeof(bf16));
    }

    header_print("info", "Input embeds:");
    utils::print_matrix(x, D);

    size_t lm_head_weight_size = D * VOCAB_SIZE_PADDED * 5 / 8;

    buffer<u8>  lm_head_weights = lm_head.create_bo_buffer<u8>(lm_head_weight_size);
    buffer<bf16> lm_head_output = lm_head.create_bo_buffer<bf16>(VOCAB_SIZE_PADDED);
    buffer<bf16> lm_head_input  = lm_head.create_bo_buffer<bf16>(D * 2);
    buffer<bf16> lm_head_output_valid = buffer<bf16>(lm_head_output.data(), VOCAB_SIZE);

    lm_head_weights.memset(0);
    lm_head_output.memset((bf16)0.0f);
    lm_head_output.sync_to_device();

    {
        buffer<u8> tmp_buffer;
        q4nx_model->load_weights(tmp_buffer, "lm_head.weight");
        header_print("info", "Vocab got: " + std::to_string(tmp_buffer.size() * 8 / 5 / D));
        header_print("info", "Vocab reserved: " + std::to_string(lm_head_weights.size() * 8 / 5 / D));
        reorder_cpy(lm_head_weights.data(), tmp_buffer, D, 4);
    }
    buffer<bf16> final_rms_weight;

    q4nx_model->load_weights(final_rms_weight, "model.norm.weight");

    memcpy(lm_head_input.data() + D, final_rms_weight.data(), D * sizeof(bf16));
    lm_head_input.sync_to_device();

    header_print("info", "Weights loaded...");
    utils::print_matrix(final_rms_weight, D);
    generate_seq(lm_head.seq(), VOCAB_SIZE_PADDED, D, 30.0f);

    header_print("info", "Running NPU sequence...");


    lm_head_weights.sync_to_device();
    time_utils::time_with_unit total_ms;
    total_ms.first = 0;
    lm_head_app_mgr->get_ready();
    lm_head_app_mgr->ready();
    
    for (int i = 0; i < test_size; i++) {
        memcpy(lm_head_input.data(), x.data() + i * D, D * sizeof(bf16));
        lm_head_input.sync_to_device();
        time_utils::time_point start = time_utils::now();
        lm_head(lm_head_output, lm_head_weights, lm_head_input);
        time_utils::time_point end = time_utils::now();

        time_utils::time_with_unit duration = time_utils::duration_ms(start, end);

        total_ms.first += duration.first;
        lm_head_output.sync_from_device();

        buffer<float> y_float(lm_head_output_valid.size());
        for (int j = 0; j < y_float.size(); j++){
            y_float[j] = (float)lm_head_output_valid[j];
        }
        float max = y_float[1];
        int max_index = 1;
        for (int j = 2; j < y_float.size(); j++){
            if (y_float[j] > max){
                max = y_float[j];
                max_index = j;
            }
        }
        std::cout << "Iter: " << i << "; Max index: " << max_index << " Input ID: " << input_ids[i] << std::endl;
    }

    lm_head_output.sync_from_device();


    header_print("info", "Time spent: " + std::to_string(total_ms.first / test_size) + total_ms.second);

    utils::print_matrix(lm_head_output_valid, 1024);
    // buffer<float> y_float(y.size());
    // for (int i = 0; i < y_float.size(); i++){
    //     y_float[i] = y[i].as_float();
    // }
    // float max = y_float[0];
    // int max_index = 0;
    // for (int i = 1; i < y_float.size(); i++){
    //     if (y_float[i] > max){
    //         max = y_float[i];
    //         max_index = i;
    //     }
    // }
    // std::cout << "Max: " << max << std::endl;
    // std::cout << "Max index: " << max_index << std::endl;
    return 0;
}

void generate_seq(npu_sequence* seq, int VOCAB_SIZE, int D, float final_scale){
    static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 32 * 32;
    static constexpr int M    = 32;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static constexpr uint32_t final_scale_address = 49152;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    assert(VOCAB_SIZE % M_PER_ROUND == 0);
    int rounds = VOCAB_SIZE / M_PER_ROUND;
    size_t TOTAL_W_SIZE = size_t(D) * size_t(VOCAB_SIZE) * 5 / 8 / 2;
    size_t WEIGHTS_PER_IT = TOTAL_W_SIZE / std::size(ITs);
    size_t W_PER_COL = (M_PER_COL * D * 5 / 8 / 2);
    size_t W_PER_ROUND = W_PER_COL * COLS;
    
    size_t w_offset = WEIGHTS_PER_IT;
    size_t y_offset = VOCAB_SIZE / std::size(ITs);

    header_print("info", "w_offset: " + std::to_string(w_offset));
    header_print("info", "y_offset: " + std::to_string(y_offset));
    uint32_t scale_int_view = *((uint32_t*)(&final_scale));
    seq->clear_cmds();
    for (int row = 0; row < 4; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row + 2, col);
            seq->rtp_write(tile, final_scale_address, scale_int_view);
        }
    }
    seq->npu_dma_memcpy_nd(
        sizeof(bf16),
        x_arg_id,
        MM2S,
        ITs[0],
        npu_bd_id(bd_0),
        it_channel_0,
        {0, 0, 0, 0},
        {1, 1, 1, (uint32_t)D * 2},
        {0, 0, 0, 1},
        -1, 0, false, aggressive_cache
    );

    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % 2) * 8;
        npu_bd_id bd_y = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_w = npu_bd_id(bd_2 + bd_offset);
        for (size_t col = 0; col < std::size(ITs); col++){
            size_t w_col_offset = r * W_PER_ROUND + col * W_PER_COL;
            uint32_t y_offset = r * M_PER_ROUND + col * M_PER_COL;
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                y_arg_id,
                S2MM,
                ITs[col],
                bd_y,
                it_channel_0,
                {0, 0, 0, (uint32_t)(y_offset)},
                {1, 1, 1, (uint32_t)M_PER_COL},
                {0, 0, 0, 1},
                -1, 0, true, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_w,
                it_channel_1,
                {0, 0, 0, (uint32_t)(w_col_offset)},
                {1, 1, 1, (uint32_t)W_PER_COL},
                {0, 0, 0, 1},
                -1, 0, false, aggressive_cache
            );
            if (r > 0){
                seq->npu_dma_wait(
                    ITs[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (size_t col = 0; col < std::size(ITs); col++){
        seq->npu_dma_wait(
            ITs[col],
            S2MM,
            it_channel_0
        );
    }
    seq->cmds2seq();
}

void reorder_cpy(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
    // memcpy(dst, src.data(), src.size());
    // return;
    const int a_block_size = 32 * 256 * 5 / 8;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    u8 *dst_ptr = dst;
    std::vector<u8 *> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++)
    {
        src_ptr[i] = src.data() + i * a_block_size * blocks_per_row;
    }
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int c = 0; c < blocks_per_row; c++)
        {
            for (int i = 0; i < vertical_blocks; i++)
            {
                memcpy(dst_ptr, src_ptr[i], a_block_size);
                dst_ptr += a_block_size;
                src_ptr[i] += a_block_size;
            }
        }
        for (int i = 0; i < vertical_blocks; i++)
        {
            src_ptr[i] += (vertical_blocks - 1) * a_block_size * blocks_per_row;
            if (src_ptr[i] + a_block_size * blocks_per_row > src.end())
            {
                src_ptr[i] = src.data(); // useless padding
            }
        }
    }
}
