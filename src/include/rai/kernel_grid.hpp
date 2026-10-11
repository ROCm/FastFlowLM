/// \file kernel_grid.hpp
/// \brief The shapes a corelib stream can actually run
/// \note corelib reports shipped kernels and rounds nothing. A reported
///       M below zero is a test double meaning "every requested M ships
///       exactly"; a real corelib never says that.
#pragma once

#include "rai/corelib_api.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flm::corelib {

struct ShippedShape {
    std::int64_t m = 0;
    std::int64_t k = 0;
    std::int64_t n = 0;
    std::int64_t group = 0;
};

class ShapeGrid {
public:
    bool wildcard = false;
    std::vector<ShippedShape> shapes;

    /// \brief whether this exact (m, k, n) ships
    /// \param group the quantization group, or a negative value to ignore it
    bool Has(std::int64_t m, std::int64_t k, std::int64_t n, std::int64_t group) const {
        if (wildcard) return true;
        for (const auto& shape : shapes) {
            if (shape.m == m && shape.k == k && shape.n == n &&
                (group < 0 || shape.group == group))
                return true;
        }
        return false;
    }

    /// \brief the smallest shipped M at least `requested` for this (k, n)
    /// \return -1 when nothing covers the request. A wildcard grid returns
    ///         `requested` itself.
    std::int64_t SmallestM(std::int64_t requested, std::int64_t k, std::int64_t n,
                           std::int64_t group) const {
        if (wildcard) return requested;
        std::int64_t best = -1;
        for (const auto& shape : shapes) {
            if (shape.k == k && shape.n == n && (group < 0 || shape.group == group) &&
                shape.m >= requested && (best < 0 || shape.m < best))
                best = shape.m;
        }
        return best;
    }

    /// \brief the narrowest N at M=1, K=`k` that is at least `min_n`
    /// \note Qwen3.5's gate projects to a handful of columns and the shipped
    ///       kernel is wider than that. A wildcard grid returns `min_n`.
    std::int64_t SmallestN(std::int64_t m, std::int64_t k, std::int64_t min_n) const {
        if (wildcard) return min_n;
        std::int64_t best = -1;
        for (const auto& shape : shapes) {
            if (shape.m == m && shape.k == k && shape.n >= min_n &&
                (best < 0 || shape.n < best))
                best = shape.n;
        }
        return best;
    }

    static ShapeGrid Matmul(const CorelibApi& api, ryzenai_corelib_stream_ptr stream) {
        ShapeGrid grid;
        api.Check(api.functions().matmul_enum_kernels(stream, &OnMatmul, &grid),
                  "ryzenai_corelib_matmul_bf16_enum_kernels");
        return grid;
    }

    static ShapeGrid MatmulAct(const CorelibApi& api, ryzenai_corelib_stream_ptr stream,
                               ryzenai_corelib_matmul_activation activation) {
        ShapeGrid grid;
        api.Check(api.functions().matmul_act_enum_kernels(stream, activation, &OnMatmul, &grid),
                  "ryzenai_corelib_matmul_act_bf16_enum_kernels");
        return grid;
    }

    /// \param gemma_family keep only the gemma_fusion ELFs, or only the others
    static ShapeGrid SsMlp(const CorelibApi& api, ryzenai_corelib_stream_ptr stream,
                           bool gemma_family) {
        ShapeGrid grid;
        SsMlpCtx ctx{&grid, gemma_family};
        api.Check(api.functions().ssmlp_enum_kernels(stream, &OnSsMlp, &ctx),
                  "ryzenai_corelib_ssmlp_bf16_enum_kernels");
        return grid;
    }

    static ShapeGrid FlatMha(const CorelibApi& api, ryzenai_corelib_stream_ptr stream,
                             const ryzenai_corelib_flat_mha_bf16_desc& desc) {
        ShapeGrid grid;
        FlatCtx ctx{&grid, &desc};
        api.Check(api.functions().flat_mha_enum_kernels(stream, &OnFlatMha, &ctx),
                  "ryzenai_corelib_flat_mha_bf16_enum_kernels");
        // A decode step is seq_len_q == 1, and that ELF has six numeric fields.
        // enum_kernels declines the ambiguous ones, so the callback can miss
        // every token kernel and the smallest reported prefill M is 64.
        // flat_mha still dispatches M=1 and puts the KV bucket in seq_pad
        // (the 64 in ..._8_1_1_64_256_4096_scale1). Record the row, or a
        // one-token step is rounded up to that prefill kernel and refused.
        if (!grid.wildcard && !grid.Has(1, 0, 0, -1)) grid.shapes.push_back({1, 0, 0, 0});
        return grid;
    }

    /// \brief paged attention kernels whose page matches `desc.page_size`
    static ShapeGrid PagedMha(const CorelibApi& api, ryzenai_corelib_stream_ptr stream,
                              const ryzenai_corelib_paged_mha_bf16_desc& desc) {
        ShapeGrid grid;
        PagedCtx ctx{&grid, &desc};
        api.Check(api.functions().paged_mha_enum_kernels(stream, &OnPagedMha, &ctx),
                  "ryzenai_corelib_paged_mha_bf16_enum_kernels");
        // A decode step is seq_len_q == 1, and paged_mha puts the KV bucket in
        // seq_pad: ..._8_1_1_64_256_4096, where 1 is the query rows and 64 is
        // seq_pad. corelib 0.25 reads that six-field name as a token kernel
        // and walks the decode PDI, so Gemma 4's p33 decode kernel is
        // reported. A corelib that does not report M=1 would round a decode
        // up to the smallest prefill kernel (64) and refuse it, so M=1 is
        // still recorded; Has() keeps it from being added twice.
        if (!grid.wildcard && !grid.Has(1, 0, 0, -1)) grid.shapes.push_back({1, 0, 0, 0});
        return grid;
    }

private:
    struct SsMlpCtx {
        ShapeGrid* grid;
        bool gemma_family;
    };
    struct FlatCtx {
        ShapeGrid* grid;
        const ryzenai_corelib_flat_mha_bf16_desc* desc;
    };
    struct PagedCtx {
        ShapeGrid* grid;
        const ryzenai_corelib_paged_mha_bf16_desc* desc;
    };

    static bool OnMatmul(void* ctx, int, std::int64_t m, std::int64_t k, std::int64_t n,
                         std::int64_t group, bool, std::int64_t, std::int64_t) {
        auto* grid = static_cast<ShapeGrid*>(ctx);
        if (m < 0) grid->wildcard = true;
        grid->shapes.push_back({m, k, n, group});
        return true;
    }

    static bool OnSsMlp(void* ctx, int, const char* family, std::int64_t m, std::int64_t k,
                        std::int64_t n, std::int64_t group) {
        auto* pack = static_cast<SsMlpCtx*>(ctx);
        if (m < 0) pack->grid->wildcard = true;
        const bool gemma = family && std::string_view(family).find("gemma") != std::string_view::npos;
        if (m >= 0 && gemma != pack->gemma_family) return true;
        pack->grid->shapes.push_back({m, k, n, group});
        return true;
    }

    static bool OnFlatMha(void* ctx, int, std::int64_t heads, std::int64_t kv_heads,
                          std::int64_t seq_len_q, std::int64_t seq_pad, std::int64_t head_size,
                          std::int64_t max_seq, std::int64_t, std::int64_t window,
                          bool kv_shared, bool scale_one) {
        auto* pack = static_cast<FlatCtx*>(ctx);
        if (seq_len_q < 0) pack->grid->wildcard = true;
        const auto* desc = pack->desc;
        const bool scale_matches = scale_one == (desc->scale == 1.0f);
        // corelib keys a kernel on the window only while the window is shorter
        // than the phase's sequence; a sliding layer at or below its window
        // runs the unwindowed kernel, which reports window 0.
        const std::int64_t phase_seq = seq_pad != 0 ? seq_pad : seq_len_q;
        const std::int64_t keyed_window =
            desc->window > 0 && desc->window < phase_seq ? desc->window : 0;
        if (seq_len_q >= 0 &&
            (heads != desc->num_heads || kv_heads != desc->kv_num_heads ||
             head_size != desc->head_size || max_seq != desc->max_seq ||
             window != keyed_window || kv_shared != (desc->kv_shared != 0) ||
             !scale_matches))
            return true;
        pack->grid->shapes.push_back({seq_len_q, 0, 0, 0});
        return true;
    }

    static bool OnPagedMha(void* ctx, int, std::int64_t heads, std::int64_t kv_heads,
                           std::int64_t seq_len_q, std::int64_t, std::int64_t head_size,
                           std::int64_t page_size, std::int64_t) {
        auto* pack = static_cast<PagedCtx*>(ctx);
        if (seq_len_q < 0) pack->grid->wildcard = true;
        const auto* desc = pack->desc;
        if (seq_len_q >= 0 &&
            (heads != desc->num_heads || kv_heads != desc->kv_num_heads ||
             head_size != desc->head_size || page_size != desc->page_size))
            return true;
        pack->grid->shapes.push_back({seq_len_q, 0, 0, 0});
        return true;
    }
};

/// \brief the row count to allocate and dispatch for one requested bucket
/// \throws when no shipped kernel covers the request
inline std::int64_t CoveringRows(const ShapeGrid& grid, std::int64_t rows, std::int64_t k,
                                 std::int64_t n, std::int64_t group, std::string_view what) {
    const auto covered = grid.SmallestM(rows, k, n, group);
    if (covered < 0) {
        throw std::runtime_error(std::string(what) + " has no kernel at or above " +
                                 std::to_string(rows) + " rows for [" + std::to_string(k) + "," +
                                 std::to_string(n) + "]");
    }
    return covered;
}

}  // namespace flm::corelib
