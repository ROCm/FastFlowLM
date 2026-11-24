// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file mm_bfp16.hpp
/// \brief Host side of mm_bfp16.xclbin (FLM_Xclbin/minicpm_ocr/dequant_mm_prefill): the mm reads its x as bfp16
///        tiles that the host makes while it computes the rows. See README.md "Prefill mm on bfp16 tiles".
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include "minicpm_ocr_model_def.hpp"
#include "minicpm_ocr_prefill_cpu.hpp"
#include "attn_bfp16.hpp"

namespace minicpm_ocr_mm_bfp16 {

constexpr int group_rows = 16;

inline size_t group_bytes(int Din) { return (size_t)Din * 18; }

/// x of one mm input width: [16-row group][Din / 8][2 row blocks][72 B]
struct tiles {
    buffer<u8> buf;
    int Din = 0;
    int rows = 0;
    //!< groups [0, written_groups) may hold non-zero tiles; the rest are zero
    int written_groups = 0;

    void reserve(npu_xclbin_manager* npu, int L_padded, int Din_) {
        if (rows == L_padded && Din == Din_) {
            return;
        }
        Din = Din_;
        rows = L_padded;
        buf = npu->create_bo_buffer<u8>((size_t)L_padded / group_rows * group_bytes(Din));
        memset(buf.data(), 0, buf.size());
        buf.sync_to_device();
        written_groups = 0;
    }

    /// \brief Makes the mm input rows [0, L): produce(row, n, out) writes bf16 rows row .. row + n - 1 to out (row
    ///        stride Din). keep != nullptr: out is keep's own rows (the host reads them again), synced after;
    ///        else a per-thread scratch, and keep is never written. Tile groups a longer call left past L are
    ///        zeroed, so the mm's padding rows read zeros as with bf16 x.
    template <typename Produce>
    void publish(buffer<bf16>* keep, int L, Produce produce) {
        publish_rows(keep, 0, L, L, produce);
    }

    /// \brief publish() for rows [row_begin, row_begin + n) of a call over rows [0, L) (one row chunk; row_begin a
    ///        multiple of group_rows). The rows that end at L also zero the stale groups past L.
    template <typename Produce>
    void publish_rows(buffer<bf16>* keep, int row_begin, int n, int L, Produce produce) {
        if (n <= 0) {
            return;
        }
        const int row_end = row_begin + n;
        const int group_begin = row_begin / group_rows;
        const int group_end = (row_end + group_rows - 1) / group_rows;
        const size_t gbytes = group_bytes(Din);

        #pragma omp parallel for num_threads(cpu_func::host_threads()) schedule(static)
        for (int group = group_begin; group < group_end; group++) {
            static thread_local std::vector<bf16> scratch;
            scratch.resize((size_t)group_rows * Din);
            const int row = group * group_rows;
            const int rows = std::min(row + group_rows, row_end) - row;
            bf16* out = keep != nullptr ? keep->data() + (size_t)row * Din : scratch.data();
            produce(row, rows, out);
            if (rows < group_rows) {
                if (out != scratch.data()) {
                    memcpy(scratch.data(), out, (size_t)rows * Din * sizeof(bf16));
                }
                memset(scratch.data() + (size_t)rows * Din, 0, (size_t)(group_rows - rows) * Din * sizeof(bf16));
                out = scratch.data();
            }
            minicpm_ocr_attn_bfp16::tile_group(buf.data() + (size_t)group * gbytes, out, Din, Din);
        }

        int dirty_end = group_end;
        if (row_end == L) {
            if (written_groups > group_end) {
                memset(buf.data() + (size_t)group_end * gbytes, 0, (size_t)(written_groups - group_end) * gbytes);
                dirty_end = written_groups;
            }
            written_groups = group_end;
        }
        minicpm_ocr_cpu::sync_range_to_device(buf, (size_t)group_begin * gbytes, (size_t)(dirty_end - group_begin) * gbytes);

        if (keep != nullptr) {
            minicpm_ocr_cpu::sync_range_to_device(*keep, (size_t)row_begin * Din, (size_t)n * Din);
        }
    }
};

/// \brief FLM_MINICPM_OCR_MM_BFP16=0 keeps mm.xclbin even when mm_bfp16.xclbin is in the xclbin folder
inline bool enabled_by_env() {
    const char* env = std::getenv("FLM_MINICPM_OCR_MM_BFP16");
    return !(env != nullptr && env[0] == '0');
}

/// \brief FLM_MINICPM_OCR_GLU_NPU=0 keeps the GLU on the host with mm_bfp16_glu.xclbin
inline bool glu_enabled_by_env() {
    const char* env = std::getenv("FLM_MINICPM_OCR_GLU_NPU");
    return !(env != nullptr && env[0] == '0');
}

/// \brief whether the host rows of an mm input must be written as bf16 too: debug builds dump and cross-check them
#ifdef ENABLE_DEBUG
constexpr bool debug_keeps_bf16 = true;
#else
constexpr bool debug_keeps_bf16 = false;
#endif

} // namespace minicpm_ocr_mm_bfp16
