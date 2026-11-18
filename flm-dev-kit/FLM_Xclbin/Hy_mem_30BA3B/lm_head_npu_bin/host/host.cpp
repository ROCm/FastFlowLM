// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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
#include "embedding_q8_0.hpp"
#include <map>
#include <chrono>
#include <cmath>
#include <fstream>

void generate_seq(npu_sequence* seq, int VOCAB_SIZE, int D);
void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks);

// Where a model lives and how its files look.
struct model_entry {
    std::string config_dir;   // folder with config.json
    std::string q4nx_dir;     // folder with model.q4nx
    bool hf_config;           // config.json is the plain HF one (LM_Config::from_pretrained asserts FLM fields)
    bool tied_embedding;      // no model.embed_tokens.weight: rebuild it from lm_head
};

static std::map<std::string, model_entry> get_model_table() {
    const char* user = std::getenv("USER");
    const std::string qwen_base = "/path/to/models/";
    const std::string hy_dir    = "/scratch/" + std::string(user ? user : "") + "/models/Hy-Memory-30B-A3B";
    return {
        {"QWEN3_5_0_8B",      {qwen_base + "Qwen3.5-0.8B-NPU2", qwen_base + "Qwen3.5-0.8B-NPU2", false, true}},
        {"QWEN3_5_2B",        {qwen_base + "Qwen3.5-2B-NPU2",   qwen_base + "Qwen3.5-2B-NPU2",   false, true}},
        {"QWEN3_5_4B",        {qwen_base + "Qwen3.5-4B-NPU2",   qwen_base + "Qwen3.5-4B-NPU2",   false, true}},
        {"QWEN3_5_9B",        {qwen_base + "Qwen3.5-9B-NPU2",   qwen_base + "Qwen3.5-9B-NPU2",   false, true}},
        // Hy: HF config.json at the top, model.q4nx in q4nx/, and a separate bf16
        // embedding (tie_word_embeddings = false)
        {"HY_MEMORY_30B_A3B", {hy_dir, hy_dir + "/q4nx", true, false}},
    };
}

// Only the three numbers this test needs, read straight from an HF config.json.
static void read_hf_config(const std::string& dir, size_t& D, size_t& vocab, float& eps) {
    std::ifstream file(dir + "/config.json");
    if (!file.is_open()) {
        header_print("error", "Cannot open " << dir << "/config.json");
        exit(1);
    }
    nlohmann::json cfg = nlohmann::json::parse(file);
    D     = cfg.at("hidden_size").get<size_t>();
    vocab = cfg.at("vocab_size").get<size_t>();
    eps   = cfg.at("rms_norm_eps").get<float>();
}

/// CPU reference of the whole NPU call: final RMS norm, then logits = W_q8 x.
/// \param packed lm_head.weight as it is in the q4nx file (before reorder_cpy_q80):
///        [vocab/32][D/256] blocks of [scales bf16 (8 groups x 32 rows) | q int8 (256 cols x 32 rows)]
static void ref_lm_head(const u8* packed, const bf16* x, const bf16* norm_w,
                        size_t D, size_t vocab, float eps, std::vector<float>& logits) {
    constexpr int ROWS = 32, COLS = 256, GROUP = 32;
    constexpr size_t SCALE_BYTES = (COLS / GROUP) * ROWS * sizeof(bf16);   // 512
    constexpr size_t BLOCK_BYTES = SCALE_BYTES + ROWS * COLS;              // 8704

    // final norm, done like the kernel: (x * w) * (1 / rms), rounded to bf16
    double sum_sq = 0.0;
    for (size_t i = 0; i < D; i++) {
        double v = (float)x[i];
        sum_sq += v * v;
    }
    float inv_rms = 1.0f / std::sqrt((float)(sum_sq / D) + eps);
    std::vector<float> xn(D);
    for (size_t i = 0; i < D; i++) {
        float wx = (float)x[i] * (float)norm_w[i];
        xn[i] = (float)(bf16)(wx * inv_rms);
    }

    const size_t col_blocks = D / COLS;
    logits.assign(vocab, 0.0f);
    for (size_t br = 0; br < vocab / ROWS; br++) {
        float row_acc[ROWS] = {0};
        for (size_t bc = 0; bc < col_blocks; bc++) {
            const u8* blk = packed + (br * col_blocks + bc) * BLOCK_BYTES;
            const bf16* scales = reinterpret_cast<const bf16*>(blk);
            const int8_t* q = reinterpret_cast<const int8_t*>(blk + SCALE_BYTES);
            for (int g = 0; g < COLS / GROUP; g++) {
                // one 32-wide group: sum q * x for all 32 rows, then apply each row's scale
                float group_acc[ROWS] = {0};
                for (int t = 0; t < GROUP; t++) {
                    const int ci = g * GROUP + t;
                    const float xv = xn[bc * COLS + ci];
                    const int8_t* q_col = q + ci * ROWS;   // a column's 32 rows are next to each other
                    for (int r = 0; r < ROWS; r++)
                        group_acc[r] += (float)q_col[r] * xv;
                }
                for (int r = 0; r < ROWS; r++)
                    row_acc[r] += group_acc[r] * (float)scales[g * ROWS + r];
            }
        }
        for (int r = 0; r < ROWS; r++)
            logits[br * ROWS + r] = row_acc[r];
    }
}

static size_t argmax(const std::vector<float>& v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * b[i];
        aa  += (double)a[i] * a[i];
        bb  += (double)b[i] * b[i];
    }
    return (float)(dot / (std::sqrt(aa) * std::sqrt(bb)));
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("HY_MEMORY_30B_A3B"), "Model type, e.g., HY_MEMORY_30B_A3B");
    desc.add_options()("tokens,n", arg_utils::po::value<int>()->default_value(32), "Number of random tokens to test");

    arg_utils::parse_options(argc, argv, desc, vm);

    std::string model_type = vm["model"].as<std::string>();
    int test_size = vm["tokens"].as<int>();

    header_print("info", "Testing model: " + model_type);

    // -m picks both the checkpoint and the matching xclbin; the builds differ in
    // MODEL_DIM and VOCAB_SIZE, which are baked into the kernel, so they are not
    // interchangeable.
    const auto model_table = get_model_table();
    auto entry_it = model_table.find(model_type);
    if (entry_it == model_table.end()){
        header_print("error", "Unknown model type: " + model_type);
        return 1;
    }
    const model_entry& entry = entry_it->second;
    header_print("info", "Model path: " + entry.q4nx_dir);

    size_t D = 0, VOCAB_SIZE = 0;
    float rms_eps = 0.0f;
    if (entry.hf_config) {
        read_hf_config(entry.config_dir, D, VOCAB_SIZE, rms_eps);
    } else {
        LM_Config lm_config;
        lm_config.from_pretrained(entry.config_dir);
        D          = lm_config.hidden_size;
        VOCAB_SIZE = lm_config.vocab_size;
        rms_eps    = lm_config.rms_norm_eps > 0.0f ? lm_config.rms_norm_eps : 1e-6f;
    }
    size_t M_PER_ROUND = 32 * 32;
    size_t VOCAB_SIZE_PADDED = (VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;

    std::string xclbin_name = "./build/" + model_type + "/xclbins/lm_head.xclbin";
    header_print("info", "LM head xclbin path: " + xclbin_name);
    header_print("info", "Hidden size (D): " + std::to_string(D));
    header_print("info", "Vocab size: " + std::to_string(VOCAB_SIZE) + " (padded " + std::to_string(VOCAB_SIZE_PADDED) + ")");
    header_print("info", "RMS eps: " << rms_eps);

    std::unique_ptr<Q4NX> q4nx_model = std::make_unique<Q4NX>(entry.q4nx_dir);

    buffer<u8> lm_head_packed;
    q4nx_model->load_weights(lm_head_packed, "lm_head.weight");

    // Input rows: random token embeddings. A tied checkpoint has no
    // model.embed_tokens.weight, so its rows are dequantized from lm_head (as the
    // DLL does, FLM_DLL/detail/qwen3_5vl_npu); Hy has its own bf16 table.
    srand(time(NULL));
    std::vector<int> input_ids(test_size);
    for (int i = 0; i < test_size; i++){
        input_ids[i] = rand() % VOCAB_SIZE;
    }
    buffer<bf16> x(D * test_size);
    if (entry.tied_embedding) {
        embedding_q8_0 embed(lm_head_packed, VOCAB_SIZE, D);
        for (int i = 0; i < test_size; i++){
            embed.forward(input_ids[i], x.data() + i * D);
        }
    } else {
        buffer<bf16> embed_table;
        q4nx_model->load_weights(embed_table, "model.embed_tokens.weight");
        for (int i = 0; i < test_size; i++){
            memcpy(x.data() + i * D, embed_table.data() + (size_t)input_ids[i] * D, D * sizeof(bf16));
        }
    }

    header_print("info", "Initializing NPU manager...");
    xrt::device npu_inst = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_inst, false);
    npu_app_manager* lm_head_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);
    npu_app lm_head = lm_head_app_mgr->create_app();

    size_t lm_head_weight_size = D * VOCAB_SIZE_PADDED + D * VOCAB_SIZE_PADDED / 32 * 2;

    buffer<u8>  lm_head_weights = lm_head.create_bo_buffer<u8>(lm_head_weight_size);
    buffer<bf16> lm_head_output = lm_head.create_bo_buffer<bf16>(VOCAB_SIZE_PADDED);
    buffer<bf16> lm_head_input  = lm_head.create_bo_buffer<bf16>(D * 2);
    buffer<bf16> lm_head_output_valid = buffer<bf16>(lm_head_output.data(), VOCAB_SIZE);

    lm_head_weights.memset(0);
    lm_head_output.memset((bf16)0.0f);
    lm_head_output.sync_to_device();

    header_print("info", "Vocab got: " + std::to_string(lm_head_packed.size() * 8 / 8.5 / D));
    header_print("info", "Vocab reserved: " + std::to_string(lm_head_weights.size() * 8 / 8.5 / D));
    reorder_cpy_q80(lm_head_weights.data(), lm_head_packed, D, 4);

    // the kernel reads the final norm weight from the second half of its x buffer
    buffer<bf16> final_rms_weight;
    q4nx_model->load_weights(final_rms_weight, "model.norm.weight");
    memcpy(lm_head_input.data() + D, final_rms_weight.data(), D * sizeof(bf16));
    lm_head_input.sync_to_device();

    generate_seq(lm_head.seq(), VOCAB_SIZE_PADDED, D);

    lm_head_weights.sync_to_device();
    lm_head_app_mgr->get_ready();
    lm_head_app_mgr->ready();

    header_print("info", "Running " + std::to_string(test_size) + " tokens...");
    double total_us = 0.0;
    double sum_cos = 0.0;
    float min_cos = 1.0f;
    int top1_match = 0;
    // A near tie can flip top-1 between two correct answers; count a mismatch as
    // real only when the NPU's pick is clearly below the reference's best logit.
    int top1_bad = 0;
    std::vector<float> y_npu(VOCAB_SIZE), y_ref;
    for (int i = 0; i < test_size; i++) {
        memcpy(lm_head_input.data(), x.data() + i * D, D * sizeof(bf16));
        lm_head_input.sync_to_device();

        auto start = std::chrono::steady_clock::now();
        lm_head(lm_head_output, lm_head_weights, lm_head_input);
        auto end = std::chrono::steady_clock::now();
        total_us += std::chrono::duration<double, std::micro>(end - start).count();

        lm_head_output.sync_from_device();
        for (size_t j = 0; j < VOCAB_SIZE; j++){
            y_npu[j] = (float)lm_head_output_valid[j];
        }

        ref_lm_head(lm_head_packed.data(), x.data() + i * D, final_rms_weight.data(),
                    D, VOCAB_SIZE, rms_eps, y_ref);

        float cos = cosine(y_npu, y_ref);
        size_t top_npu = argmax(y_npu);
        size_t top_ref = argmax(y_ref);
        float gap = y_ref[top_ref] - y_ref[top_npu];   // how much worse the NPU's pick is in the reference
        bool same = top_npu == top_ref;
        bool bad = !same && gap > 0.01f * std::fabs(y_ref[top_ref]);
        top1_match += same;
        top1_bad += bad;
        sum_cos += cos;
        min_cos = std::min(min_cos, cos);

        std::cout << "token " << i << " id " << input_ids[i]
                  << ": cos " << cos
                  << ", top-1 npu " << top_npu << " ref " << top_ref
                  << (same ? "" : (bad ? "  MISMATCH" : "  (near tie)")) << std::endl;
    }

    // simple pass rule: logits nearly identical and no real top-1 mismatch
    const float COS_FLOOR = 0.999f;
    bool pass = min_cos >= COS_FLOOR && top1_bad == 0;

    std::cout << "\n===== RESULT =====" << std::endl;
    std::cout << "logits vs CPU Q8_0 reference: mean cos " << sum_cos / test_size
              << ", min cos " << min_cos << std::endl;
    std::cout << "top-1 match: " << top1_match << "/" << test_size
              << " (real mismatches " << top1_bad << ")" << std::endl;
    std::cout << "time per call: " << total_us / test_size << " us (host + NPU)" << std::endl;
    std::cout << (pass ? "PASS" : "FAIL") << " (min cos >= " << COS_FLOOR << ", no real top-1 mismatch)" << std::endl;
    return pass ? 0 : 1;
}

void generate_seq(npu_sequence* seq, int VOCAB_SIZE, int D){
   static constexpr int y_arg_id = 0;
    static constexpr int w_arg_id = 1;
    static constexpr int x_arg_id = 2;
    static constexpr int M_PER_ROUND = 32 * 32;
    static constexpr int M    = 32;
    static constexpr int M_PER_COL    = M * 4;
    static constexpr int COLS = 8;
    static npu_tiles ITs[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    // Packet ids for the two flows that share IT0's MM2S 0 -- must match X_PKT_ID /
    // W_PKT_ID in iron/q4k_mm.py. Every other shim MM2S port is circuit-switched.
    static constexpr int X_PKT_ID = 0;
    static constexpr int W_PKT_ID = 1;
    static constexpr int NO_PKT   = -1;
    // Rounds of shim BD ping-pong; also how far the y waits lag the y issues.
    static constexpr int LOOKAHEAD = 3;
    assert(VOCAB_SIZE % M_PER_ROUND == 0);
    int rounds = VOCAB_SIZE / M_PER_ROUND;
    size_t W_PER_COL = int(M_PER_COL * D * 8.5 / 8) / sizeof(bf16);
    size_t W_PER_ROUND = W_PER_COL * COLS;
    // Each column's weights are split over two shim MM2S channels, one per pair of
    // cores; reorder_cpy_q80 has already laid the two halves out back to back.
    size_t W_PER_STREAM = W_PER_COL / 2;

    seq->clear_cmds();
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
        X_PKT_ID, 0, false, aggressive_cache
    );

    for (int r = 0; r < rounds; r++) {
        int bd_offset = (r % (LOOKAHEAD + 1)) * 3;
        npu_bd_id bd_y  = npu_bd_id(bd_1 + bd_offset);
        npu_bd_id bd_wa = npu_bd_id(bd_2 + bd_offset);
        npu_bd_id bd_wb = npu_bd_id(bd_3 + bd_offset);
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
            // Group a (cores 0/1) on channel 1, group b (cores 2/3) on channel 0.
            // Channel 0 of IT0 is the one shared with the x broadcast, so only that
            // transfer needs a packet id.
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wa,
                it_channel_1,
                {0, 0, 0, (uint32_t)(w_col_offset)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                NO_PKT, 0, false, aggressive_cache
            );
            seq->npu_dma_memcpy_nd(
                sizeof(bf16),
                w_arg_id,
                MM2S,
                ITs[col],
                bd_wb,
                it_channel_0,
                {0, 0, 0, (uint32_t)(w_col_offset + W_PER_STREAM)},
                {1, 1, 1, (uint32_t)W_PER_STREAM},
                {0, 0, 0, 1},
                col == 0 ? W_PKT_ID : NO_PKT, 0, false, aggressive_cache
            );
            // Four sets of shim BDs are in flight, so a round only has to wait for
            // the y of the round three back rather than the one immediately before.
            // That keeps three rounds of weight transfers queued ahead of the
            // sequencer instead of one.
            if (r > LOOKAHEAD - 1){
                seq->npu_dma_wait(
                    ITs[col],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }
    for (size_t col = 0; col < std::size(ITs); col++){
        for (int i = 0; i < LOOKAHEAD; i++){
            seq->npu_dma_wait(
                ITs[col],
                S2MM,
                it_channel_0
            );
        }
    }

    seq->cmds2seq();
}

void reorder_cpy_q80(u8 *dst, buffer<u8> &src, const int col, const int vertical_blocks)
{
     // return;
    const int a_block_size = 32 * 256 + 256 * 2;
    const int blocks_per_row = col / 256;

    const int rows = src.size() / a_block_size / blocks_per_row;

    // A column's weights arrive on two shim channels, one per pair of cores, so each
    // column-round has to be two contiguous halves in DDR rather than one run with all
    // four cores' blocks interleaved: rows r+0/r+1 of the group for the stream that
    // feeds cores 0 and 1, rows r+2/r+3 for the stream that feeds cores 2 and 3. Within
    // a half the order is still k-major, one block per core, because that is exactly one
    // memtile slot. See lm_head_memtile in iron/q4k_mm.py.
    const int rows_per_stream = vertical_blocks / 2;

    u8 *dst_ptr = dst;
    for (int r = 0; r < rows; r += vertical_blocks)
    {
        for (int half = 0; half < 2; half++)
        {
            for (int c = 0; c < blocks_per_row; c++)
            {
                for (int i = 0; i < rows_per_stream; i++)
                {
                    const int row = r + half * rows_per_stream + i;
                    const u8 *src_ptr = (row < rows)
                        ? src.data() + (size_t)(row * blocks_per_row + c) * a_block_size
                        : src.data(); // useless padding
                    memcpy(dst_ptr, src_ptr, a_block_size);
                    dst_ptr += a_block_size;
                }
            }
        }
    }
}
