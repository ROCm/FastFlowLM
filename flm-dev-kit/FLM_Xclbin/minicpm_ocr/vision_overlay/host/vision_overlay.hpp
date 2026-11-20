// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Host side of the MiniCPM-OCR vision overlay (FLM_DLL npu_sequence). See ../README.md.
//
// A run of an op = [switch, only when the array is set up for another op]
//                + [dispatcher words + go lock, every run]
//                + the op's usual sequence (gen_vision_mm_seq / gen_vision_attn_seq, unchanged).
#pragma once
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>
#include "npu_utils/npu_instr_utils.hpp"
#include "op_configs.hpp"
#include "overlay_layout.hpp"

namespace minicpm_ocr_overlay {

using namespace vision_overlay_layout;

constexpr int OP_NONE = 0;     // nothing configured: after a fresh load, or after another xclbin ran
constexpr uint32_t CORE_LOCK_BASE = 0x1F000;
constexpr uint32_t GO_LOCK_REG = CORE_LOCK_BASE + 16 * OP_LOCK;
constexpr uint32_t DMA_CTRL_RESET = 0x2;
constexpr int ROWS = 4, COLS = 8;

inline const op_config& config_of(int op) {
    return op == OP_MM ? op_mm : op_attn;
}

using reg_key = std::tuple<uint32_t, uint32_t, uint32_t>;   // col, row, offset

template <typename Get>
inline std::vector<reg_key> union_of(Get get) {
    std::set<reg_key> all;
    for (const op_config* cfg : {&op_mm, &op_attn}) {
        for (const reg_write& w : get(*cfg)) {
            all.insert({w.col, w.row, w.offset});
        }
    }
    return {all.begin(), all.end()};
}

inline uint32_t value_in(const std::vector<reg_write>& regs, const reg_key& key) {
    for (const reg_write& w : regs) {
        if (w.col == std::get<0>(key) && w.row == std::get<1>(key) && w.offset == std::get<2>(key)) {
            return w.value;
        }
    }
    return 0;
}

inline void write_reg(npu_sequence& seq, uint32_t row, uint32_t col, uint32_t offset, uint32_t value) {
    seq.rtp_write(get_tile(row, col), offset, value);
}

/// @brief configure the array for op `to`
/// @param from the op configured now, or OP_NONE (then every channel is reset and every
///        route written, not only those that differ)
inline void gen_switch(npu_sequence& seq, int to, int from) {
    static const std::vector<reg_key> sw_regs = union_of([](const op_config& c) -> const auto& { return c.sw; });
    static const std::vector<reg_key> lock_regs = union_of([](const op_config& c) -> const auto& { return c.locks; });
    const op_config& cfg = config_of(to);
    const bool full = from != OP_MM && from != OP_ATTN;

    auto reset = [&](uint32_t row, uint32_t col, uint32_t offset) {
        seq.npu_maskwrite(get_tile(row, col), offset, DMA_CTRL_RESET, DMA_CTRL_RESET);
        seq.npu_maskwrite(get_tile(row, col), offset, 0, DMA_CTRL_RESET);
    };
    if (full) {
        for (const reg_write& ch : all_channels) {
            reset(ch.row, ch.col, ch.offset);
        }
    } else {
        for (const chain_start& s : config_of(from).starts) {
            reset(s.row, s.col, s.ctrl_offset);
        }
    }
    for (const reg_key& key : sw_regs) {
        const uint32_t v = value_in(cfg.sw, key);
        if (!full && v == value_in(config_of(from).sw, key)) {
            continue;
        }
        write_reg(seq, std::get<1>(key), std::get<0>(key), std::get<2>(key), v);
    }
    for (const mask_write& m : cfg.mux) {
        seq.npu_maskwrite(get_tile(m.row, m.col), m.offset, m.value, m.mask);
    }
    // the two ops use the same BD numbers, so the target's BDs are written at every switch
    for (const reg_block& b : cfg.bds) {
        for (size_t i = 0; i < b.words.size(); i++) {
            write_reg(seq, b.row, b.col, b.offset + 4 * (uint32_t)i, b.words[i]);
        }
    }
    for (const reg_key& key : lock_regs) {
        if (std::get<1>(key) >= 2 && std::get<2>(key) == GO_LOCK_REG) {
            continue;
        }
        write_reg(seq, std::get<1>(key), std::get<0>(key), std::get<2>(key), value_in(cfg.locks, key));
    }
    for (const chain_start& s : cfg.starts) {
        write_reg(seq, s.row, s.col, s.queue_offset, s.queue_value);
        write_reg(seq, s.row, s.col, s.ctrl_offset, s.ctrl_value);
    }
}

/// @brief the commands that go in front of every run of `to`
/// @param from  the op the array is set up for when this run starts (OP_NONE: nothing / unknown)
/// @param calls how many times each core calls the op's kernel in this run
///              (mm: N / tile_N strips; attention: segments x heads / 4)
inline void gen_prefix(npu_sequence& seq, int to, int from, uint32_t calls) {
    seq.clear_cmds();
    if (from != to) {
        gen_switch(seq, to, from);
    }
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            write_reg(seq, r + 2, c, DISPATCH_ADDR, (uint32_t)to);
            write_reg(seq, r + 2, c, DISPATCH_ADDR + 4, calls);
        }
    }
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            write_reg(seq, r + 2, c, GO_LOCK_REG, 1);
        }
    }
}

/// @brief put `prefix` in front of the already generated `op_seq`
///        (joined as instruction words: 4-word header, then the ops)
inline void prepend(npu_sequence& op_seq, npu_sequence& prefix) {
    auto [pre_words, pre_len] = prefix.dump();
    if (pre_len <= 4) {
        return;
    }
    auto [op_words, op_len] = op_seq.dump();
    std::vector<uint32_t> joined;
    joined.reserve(pre_len + op_len - 4);
    joined.push_back(op_words[0]);
    joined.push_back(op_words[1]);
    joined.push_back(pre_words[2] + op_words[2]);
    joined.push_back((uint32_t)((pre_len + op_len - 4) << 2));
    joined.insert(joined.end(), pre_words + 4, pre_words + pre_len);
    joined.insert(joined.end(), op_words + 4, op_words + op_len);
    op_seq.from_vector(joined);
}

} // namespace minicpm_ocr_overlay

namespace vision_overlay = minicpm_ocr_overlay;
