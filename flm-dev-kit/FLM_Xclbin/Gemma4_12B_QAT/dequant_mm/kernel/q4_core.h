// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q4_CORE_H__
#define __Q4_CORE_H__
#include "aie_kernel_utils.h"
#include "common.hpp"

void _Q4_0_bf16(
    bf16* w16_buffer,
    int8* w_buffer,
    bf16 * scale,
    int row
) {
    
    int4* it_quant = (int4*)w_buffer;
    
    bf16* w_ptr_iter_0 = w16_buffer + row * QXNX_COL_BLOCK_SIZE;
    bf16* w_ptr_iter_1 = w16_buffer + (row + 8) * QXNX_COL_BLOCK_SIZE;

    for (int sub_chunk = 0; sub_chunk < QXNX_COL_BLOCK_SIZE / 32; sub_chunk++) {
        aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(scale + sub_chunk * QXNX_ROW_BLOCK_SIZE );

        static constexpr int LANES = 4 * PARALLEL_ROWS;

        auto splat_rows = [](aie::vector<bf16, PARALLEL_ROWS> v) {
            aie::vector<bf16, 2 * PARALLEL_ROWS> v32 = aie::concat(v, v);
            aie::vector<bf16, LANES> v64 = aie::concat(v32, v32);
            return aie::interleave_unzip(v64, v64, 8);
        };
        auto scale_vec = splat_rows(a_scales);
        const aie::vector<bf16, LANES>& scale_lo = scale_vec.first;
        const aie::vector<bf16, LANES>& scale_hi = scale_vec.second;

        auto emit_group = [&](const aie::vector<bf16, PARALLEL_ROWS * 8>& d) {
            auto rows = aie::interleave_unzip(d.extract<LANES>(0), d.extract<LANES>(1), 8);
            aie::store_v(w_ptr_iter_0, aie::mul(rows.first,  scale_lo).to_vector<bf16>());
            aie::store_v(w_ptr_iter_1, aie::mul(rows.second, scale_hi).to_vector<bf16>());
            w_ptr_iter_0 += 64;
            w_ptr_iter_1 += 64;
        };

        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( 2, 2 )
        for (int pair = 0; pair < 2; pair++) {
            aie::vector<int4, PARALLEL_ROWS * 8> qa = aie::load_v<PARALLEL_ROWS * 8>(it_quant);
            it_quant += PARALLEL_ROWS * 4;
            aie::vector<int4, PARALLEL_ROWS * 8> qb = aie::load_v<PARALLEL_ROWS * 8>(it_quant);
            it_quant += PARALLEL_ROWS * 4;
            aie::vector<bf16, PARALLEL_ROWS * 8> da = aie::to_float<bf16>(qa, 0);
            aie::vector<bf16, PARALLEL_ROWS * 8> db = aie::to_float<bf16>(qb, 0);
            emit_group(da);
            emit_group(db);
        }
    } // end of sub_chunk
}

#endif // __Q4_CORE_H__