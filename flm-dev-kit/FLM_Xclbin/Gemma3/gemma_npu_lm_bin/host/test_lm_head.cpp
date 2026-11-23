// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "lm_config.hpp"
#include "models/gemma/gemma_npu_sequence.hpp"
#include "npu_utils/npu_utils.hpp"
#include "npu_utils/runtime_wait.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

namespace {

constexpr int ROWS_PER_BLOCK = 32;
constexpr int COLS_PER_BLOCK = 256;
constexpr int BLOCK_BYTES = ROWS_PER_BLOCK * COLS_PER_BLOCK * 5 / 8;
constexpr int GROUP_COLS = 32;
constexpr int GROUPS_PER_BLOCK = COLS_PER_BLOCK / GROUP_COLS;
constexpr int ROW_TILE = 16;

struct Options {
    std::string model_dir;
    std::string old_xclbin;
    std::string new_xclbin;
    std::string x_file;
    int random_x = 4;
    int iters = 10;
    int cpu_ref = 2;
    bool stub = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        std::string v = argv[i + 1];
        if (k == "--model") o.model_dir = v;
        else if (k == "--old") o.old_xclbin = v;
        else if (k == "--new") o.new_xclbin = v;
        else if (k == "--x") o.x_file = v;
        else if (k == "--random") o.random_x = std::stoi(v);
        else if (k == "--iters") o.iters = std::stoi(v);
        else if (k == "--cpu-ref") o.cpu_ref = std::stoi(v);
        else if (k == "--stub") o.stub = (v == "1");
        else { std::cerr << "unknown option " << k << std::endl; std::exit(2); }
    }
    if (o.model_dir.empty() || o.new_xclbin.empty() || (!o.stub && o.old_xclbin.empty())) {
        std::cerr << "usage: test_lm_head --model <dir> --new <layer_lm.xclbin> [--old <lm_head.xclbin>]"
                     " [--x <file of bf16 rows>] [--random N] [--iters N] [--cpu-ref N] [--stub 1]" << std::endl;
        std::exit(2);
    }
    return o;
}

// Same as gemma_npu.cpp's reorder_cpy: interleave `vertical_blocks` row blocks per column block.
void reorder_cpy(u8* dst, buffer<u8>& src, int cols, int vertical_blocks) {
    const int blocks_per_row = cols / COLS_PER_BLOCK;
    const int row_blocks = src.size() / BLOCK_BYTES / blocks_per_row;
    u8* dst_ptr = dst;
    std::vector<u8*> src_ptr(vertical_blocks);
    for (int i = 0; i < vertical_blocks; i++) {
        src_ptr[i] = src.data() + (size_t)i * BLOCK_BYTES * blocks_per_row;
    }
    for (int r = 0; r < row_blocks; r += vertical_blocks) {
        for (int c = 0; c < blocks_per_row; c++) {
            for (int i = 0; i < vertical_blocks; i++) {
                memcpy(dst_ptr, src_ptr[i], BLOCK_BYTES);
                dst_ptr += BLOCK_BYTES;
                src_ptr[i] += BLOCK_BYTES;
            }
        }
        for (int i = 0; i < vertical_blocks; i++) {
            src_ptr[i] += (size_t)(vertical_blocks - 1) * BLOCK_BYTES * blocks_per_row;
            if (src_ptr[i] + (size_t)BLOCK_BYTES * blocks_per_row > src.end()) {
                src_ptr[i] = src.data();
            }
        }
    }
}

float bf16_to_float(bf16 v) {
    return static_cast<float>(v);
}

// One weight of a 32 x 256 Q4_1 block, in the order proj.cc's _qmm_q4k_bf16 reads it:
// scales/mins [group][row], codes [row tile][group][column][row in tile], low nibble first.
float block_weight(const u8* block, int r, int c) {
    const bf16* scales = reinterpret_cast<const bf16*>(block);
    const bf16* mins = scales + ROWS_PER_BLOCK * GROUPS_PER_BLOCK;
    const u8* codes = reinterpret_cast<const u8*>(mins + ROWS_PER_BLOCK * GROUPS_PER_BLOCK);
    const int g = c / GROUP_COLS;
    const int col_in_group = c % GROUP_COLS;
    const int tile = r / ROW_TILE;
    const int row_in_tile = r % ROW_TILE;
    const size_t idx = (((size_t)tile * GROUPS_PER_BLOCK + g) * GROUP_COLS + col_in_group) * ROW_TILE + row_in_tile;
    const u8 byte = codes[idx / 2];
    const int q = (idx % 2 == 0) ? (byte & 0xF) : (byte >> 4);
    const float scale = bf16_to_float(scales[g * ROWS_PER_BLOCK + r]);
    const float min = bf16_to_float(mins[g * ROWS_PER_BLOCK + r]);
    return q * scale + min;
}

std::vector<float> cpu_logits(buffer<u8>& w, const bf16* x, int rows, int cols) {
    const int blocks_per_row = cols / COLS_PER_BLOCK;
    std::vector<float> xf(cols);
    for (int i = 0; i < cols; i++) xf[i] = bf16_to_float(x[i]);
    std::vector<float> out(rows);
    #pragma omp parallel for
    for (int row = 0; row < rows; row++) {
        const int rb = row / ROWS_PER_BLOCK;
        const int r = row % ROWS_PER_BLOCK;
        double acc = 0.0;
        for (int cb = 0; cb < blocks_per_row; cb++) {
            const u8* block = w.data() + ((size_t)rb * blocks_per_row + cb) * BLOCK_BYTES;
            for (int c = 0; c < COLS_PER_BLOCK; c++) {
                acc += (double)block_weight(block, r, c) * xf[cb * COLS_PER_BLOCK + c];
            }
        }
        out[row] = (float)acc;
    }
    return out;
}

struct Compare {
    size_t mismatches = 0;
    float max_abs = 0.0f;
    double cos = 0.0;
};

Compare compare(const bf16* a, const bf16* b, size_t n) {
    Compare res;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        uint16_t ua, ub;
        memcpy(&ua, &a[i], 2);
        memcpy(&ub, &b[i], 2);
        if (ua != ub) res.mismatches++;
        const float fa = bf16_to_float(a[i]);
        const float fb = bf16_to_float(b[i]);
        res.max_abs = std::max(res.max_abs, std::fabs(fa - fb));
        dot += (double)fa * fb;
        na += (double)fa * fa;
        nb += (double)fb * fb;
    }
    res.cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    return res;
}

double cos_float(const bf16* a, const std::vector<float>& ref) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double fa = bf16_to_float(a[i]);
        dot += fa * ref[i];
        na += fa * fa;
        nb += (double)ref[i] * ref[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
}

size_t argmax(const bf16* a, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        if (bf16_to_float(a[i]) > bf16_to_float(a[best])) best = i;
    }
    return best;
}

// The sequence of the old 32-core lm_head.xclbin (Gemma3/gemma_lm_head_npu_bin, removed; in git history), the reference
// once the engine stopped using it. Args: 0 logits, 1 weights (reorder_cpy(..., 4)), 2 x.
void gen_old_lm_head_seq(npu_sequence* seq, int D, uint32_t rows) {
    static constexpr npu_tiles tiles[] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr uint32_t columns = 8;
    constexpr int y_arg_idx = 0;
    constexpr int w_arg_idx = 1;
    constexpr int x_arg_idx = 2;
    constexpr int m = 32;
    constexpr int m_per_round = columns * 4 * m;
    constexpr int a_block_size = 32 * 256 * 5 / 8 / 2;
    constexpr int row_blocks_per_round = columns * 4;
    const int blocks_per_row = D / 256;
    const int rounds = rows / m_per_round;

    seq->clear_cmds();
    seq->npu_preemption(0);
    for (int round = 0; round < rounds; round++) {
        uint32_t round_offset = round * m_per_round;
        uint32_t bd_offset = (round % 2) * 8;
        seq->npu_dma_memcpy_nd(sizeof(bf16), x_arg_idx, MM2S, tiles[0], npu_bd_id(bd_0 + bd_offset), it_channel_0,
                               {0, 0, 0, 0}, {1, 1, 1, (uint32_t)D}, {0, 0, 0, 1}, -1, 0, false);
        for (uint32_t col = 0; col < columns; col++) {
            uint32_t y_offset = 4 * m * col + round_offset;
            seq->npu_dma_memcpy_nd(sizeof(bf16), y_arg_idx, S2MM, tiles[col], npu_bd_id(bd_2 + bd_offset), it_channel_0,
                                   {0, 0, 0, y_offset}, {1, 1, 1, 4 * m}, {0, 0, 0, 1}, -1, 0, true);
        }
        for (uint32_t col = 0; col < columns; col++) {
            uint32_t w_offset = (round * row_blocks_per_round + col * 4) * a_block_size * blocks_per_row;
            seq->npu_dma_memcpy_nd(sizeof(bf16), w_arg_idx, MM2S, tiles[col], npu_bd_id(bd_1 + bd_offset), it_channel_1,
                                   {0, 0, 0, w_offset}, {1, 1, 1, (uint32_t)(4 * a_block_size * blocks_per_row)},
                                   {0, 0, 0, 1}, -1, 0, false);
        }
        if (round > 0) {
            for (uint32_t col = 0; col < columns; col++) {
                seq->npu_dma_wait(tiles[col], S2MM, it_channel_0);
            }
        }
    }
    for (uint32_t col = 0; col < columns; col++) {
        seq->npu_dma_wait(tiles[col], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}

std::vector<std::vector<bf16>> load_x(const Options& o, int D, const buffer<bf16>& norm) {
    std::vector<std::vector<bf16>> xs;
    if (!o.x_file.empty()) {
        std::ifstream f(o.x_file, std::ios::binary);
        std::vector<bf16> row(D);
        while (f.read(reinterpret_cast<char*>(row.data()), (std::streamsize)D * sizeof(bf16))) {
            xs.push_back(row);
        }
    }
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int k = 0; k < o.random_x; k++) {
        std::vector<bf16> row(D);
        for (int i = 0; i < D; i++) {
            row[i] = bf16(dist(gen) * (1.0f + bf16_to_float(norm[i])));
        }
        xs.push_back(row);
    }
    return xs;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename Run>
double time_run_ms(Run& run, const char* what) {
    auto t0 = std::chrono::steady_clock::now();
    run.start();
    flm::npu::wait_for_completion(run, what);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);

    LM_Config config;
    {
        std::ifstream cf(o.model_dir + "/config.json");
        config._json_config = nlohmann::json::parse(cf);
    }
    const int D = config.get("hidden_size");
    const int vocab = config.get("vocab_size");
    gemma_npu_sequence sequence(config, 8192);
    const uint32_t new_rows = sequence.get_lm_head_in_layer_rows();
    const uint32_t old_rows = (vocab + 1024 - 1) / 1024 * 1024;
    std::cout << "D " << D << " vocab " << vocab << " old rows " << old_rows << " new rows " << new_rows << std::endl;

    Q4NX q4nx(o.model_dir);
    buffer<u8> w_lm_head;
    q4nx.load_weights(w_lm_head, "lm_head.weight");
    buffer<bf16> final_norm;
    q4nx.load_weights(final_norm, "model.norm.weight");

    flm_rt::device device(0u);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);

    npu_app_manager* new_mgr = npu.register_xclbin(o.new_xclbin);
    npu_app new_app = new_mgr->create_app();
    sequence.gen_lm_head_in_layer_seq(new_app.seq());
    buffer<bf16> new_x = new_app.create_bo_buffer<bf16>(D);
    buffer<u8> new_w = new_app.create_bo_buffer<u8>((size_t)new_rows * D * 5 / 8);
    buffer<bf16> new_norm = new_app.create_bo_buffer<bf16>(D);
    buffer<bf16> new_unused = new_app.create_bo_buffer<bf16>(D);
    buffer<bf16> new_logits = new_app.create_bo_buffer<bf16>(new_rows);
    reorder_cpy(new_w.data(), w_lm_head, D, 2);
    memcpy(new_norm.data(), final_norm.data(), (size_t)D * sizeof(bf16));
    new_w.sync_to_device();
    new_norm.sync_to_device();
    new_unused.sync_to_device();
    auto new_run = new_app.create_run(new_x, new_w, new_norm, new_unused, new_logits);

    npu_app_manager* old_mgr = nullptr;
    npu_app old_app;
    buffer<bf16> old_x, old_logits;
    buffer<u8> old_w;
    xrt::run old_run;
    if (!o.stub) {
        old_mgr = npu.register_xclbin(o.old_xclbin);
        old_app = old_mgr->create_app();
        gen_old_lm_head_seq(old_app.seq(), D, old_rows);
        old_x = old_app.create_bo_buffer<bf16>(D);
        old_w = old_app.create_bo_buffer<u8>((size_t)old_rows * D * 5 / 8);
        old_logits = old_app.create_bo_buffer<bf16>(old_rows);
        reorder_cpy(old_w.data(), w_lm_head, D, 4);
        old_w.sync_to_device();
        old_run = old_app.create_run(old_logits, old_w, old_x);
    }

    const std::vector<std::vector<bf16>> xs = load_x(o, D, final_norm);
    std::cout << "x vectors: " << xs.size() << std::endl;

    bool all_identical = true;
    int cpu_checked = 0;
    double worst_cpu_cos = 1.0;
    for (size_t k = 0; k < xs.size() && !o.stub; k++) {
        memcpy(old_x.data(), xs[k].data(), (size_t)D * sizeof(bf16));
        memcpy(new_x.data(), xs[k].data(), (size_t)D * sizeof(bf16));
        old_x.sync_to_device();
        new_x.sync_to_device();
        old_logits.sync_to_device();
        new_logits.sync_to_device();
        time_run_ms(old_run, "old lm_head");
        time_run_ms(new_run, "lm_head in layer");
        old_logits.sync_from_device();
        new_logits.sync_from_device();

        const Compare c = compare(old_logits.data(), new_logits.data(), vocab);
        const size_t top_old = argmax(old_logits.data(), vocab);
        const size_t top_new = argmax(new_logits.data(), vocab);
        std::cout << "x " << k << ": mismatching logits " << c.mismatches << " / " << vocab
                  << ", max |diff| " << c.max_abs << ", cos " << c.cos
                  << ", top-1 old " << top_old << " new " << top_new;
        if (c.mismatches != 0) all_identical = false;
        if (cpu_checked < o.cpu_ref) {
            const std::vector<float> ref = cpu_logits(w_lm_head, xs[k].data(), vocab, D);
            const double cos_new = cos_float(new_logits.data(), ref);
            const double cos_old = cos_float(old_logits.data(), ref);
            worst_cpu_cos = std::min(worst_cpu_cos, std::min(cos_new, cos_old));
            std::cout << ", cos vs CPU: new " << cos_new << " old " << cos_old;
            cpu_checked++;
        }
        std::cout << std::endl;
    }

    std::vector<double> new_ms, old_ms, alt_new_ms, alt_old_ms;
    time_run_ms(new_run, "warm-up");
    for (int i = 0; i < o.iters; i++) new_ms.push_back(time_run_ms(new_run, "lm_head in layer"));
    if (!o.stub) {
        time_run_ms(old_run, "warm-up");
        for (int i = 0; i < o.iters; i++) old_ms.push_back(time_run_ms(old_run, "old lm_head"));
        for (int i = 0; i < o.iters; i++) {
            alt_old_ms.push_back(time_run_ms(old_run, "old lm_head"));
            alt_new_ms.push_back(time_run_ms(new_run, "lm_head in layer"));
        }
    }
    std::cout << "time ms (median of " << o.iters << "), back to back: new " << median(new_ms);
    if (!o.stub) {
        std::cout << " old " << median(old_ms) << "; alternating (each pays a context switch): new "
                  << median(alt_new_ms) << " old " << median(alt_old_ms);
    }
    std::cout << std::endl;

    if (o.stub) {
        std::cout << "STUB: timing only" << std::endl;
        return 0;
    }
    std::cout << (all_identical ? "PASSED" : "FAILED") << ": logits bit-identical to lm_head.xclbin for "
              << xs.size() << " x vectors; worst cos vs CPU " << worst_cpu_cos << std::endl;
    return all_identical ? 0 : 1;
}
