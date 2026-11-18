// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __Q8_CORE_H__
#define __Q8_CORE_H__
#include "aie_kernel_utils.h"
#include "common.hpp"

template<bool is_even>
void _Q8_x_bf16(
    bf16* w16_buffer,
    bf16* scale_buffer,
    bf16* zp_buffer,
    int8* w_buffer_0,
    int8* w_buffer_1,
    int8* w8_buffer,
    bool& is_w_ping,
    int w_prod_lock_id,
    int w_cons_lock_id,
    int& has_zp
) {
    // first load scales
    constexpr int num_slices_per_row = QXNX_COL_BLOCK_SIZE / Q8_COL_PER_SLICE;
    // has_zp is an RTP value that cannot change inside this call.  Reading it
    // once keeps the compiler from reloading it through the reference on every
    // slice, and folding it into a stride turns the zp cursor step into an
    // unconditional add instead of a branch inside the pipelined slice loop.
    const int has_zp_l = has_zp;
    const int zp_step = (has_zp_l == 1) ? QXNX_ROW_BLOCK_SIZE : 0;
    _lock_acquire(w_cons_lock_id);
    is_w_ping = !is_w_ping;
    if (is_w_ping) {
        _memcpy<W_SLICE_SIZE>((uint8*)scale_buffer, (uint8*)w_buffer_0);
    }
    else {
        _memcpy<W_SLICE_SIZE>((uint8*)scale_buffer, (uint8*)w_buffer_1);
    }
    _lock_release(w_prod_lock_id);

    if (has_zp_l == 1) {
        _lock_acquire(w_cons_lock_id);
        is_w_ping = !is_w_ping;
        if (is_w_ping) {
        _memcpy<W_SLICE_SIZE>((uint8*)zp_buffer, (uint8*)w_buffer_0);
        }
        else {
        _memcpy<W_SLICE_SIZE>((uint8*)zp_buffer, (uint8*)w_buffer_1);
        }
        _lock_release(w_prod_lock_id);
    }
    // With no zero-point the cursor never moves and only the first row block
    // worth of zp entries is ever read; the caller pre-zeroes them once per MM.
    // That makes "no zero-point" an add of zero and deletes the loop-invariant
    // branch that would otherwise sit inside the pipelined body.

    AIE_LOOP_UNROLL_FULL
    for (int row = 0; row < QXNX_ROW_BLOCK_SIZE; row += PARALLEL_ROWS){
        
        bf16* w_ptr_iter_0 = w16_buffer + row * QXNX_COL_BLOCK_SIZE;
        bf16* w_ptr_iter_1 = w16_buffer + (row + 8) * QXNX_COL_BLOCK_SIZE;
        

        bf16* it_scale = scale_buffer + row;
        bf16* it_zp = zp_buffer + row;

        // Only the group loop inside is modulo scheduled: pipelining this one
        // too would put four groups in flight and overflow the register file.
        AIE_NO_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE( num_slices_per_row / 2, num_slices_per_row / 2 )
        for (int slice = 0; slice < num_slices_per_row / 2; slice++) {
            // Two tiles share the same broadcast weight stream: is_even keeps
            // slice 2*s and drops 2*s+1, !is_even does the opposite.  The
            // dropped slice is still drained from the ping-pong buffers so the
            // producer keeps flowing, and the scale / zp / output cursors step
            // over it so the kept slices land at their true positions.
            if constexpr (!is_even) {
                _lock_acquire(w_cons_lock_id);
                is_w_ping = !is_w_ping;
                _lock_release(w_prod_lock_id);
                it_scale += QXNX_ROW_BLOCK_SIZE;
                it_zp += zp_step;
                w_ptr_iter_0 += 4 * 64;
                w_ptr_iter_1 += 4 * 64;
            }
            _lock_acquire(w_cons_lock_id);
            is_w_ping = !is_w_ping;
            _memcpy<W_SLICE_SIZE>((uint8*)w8_buffer, (uint8*)(is_w_ping? w_buffer_0 : w_buffer_1));
            // _memcpy is a synchronous vector copy, so the ping-pong slot is
            // already fully drained here: hand it back now and let the producer
            // DMA refill it while this core scales and interleaves.
            _lock_release(w_prod_lock_id);
            int8* it_q = w8_buffer;

            aie::vector<bf16, PARALLEL_ROWS> a_scales = aie::load_v<PARALLEL_ROWS>(it_scale);
            aie::vector<bf16, PARALLEL_ROWS> a_mins = aie::load_v<PARALLEL_ROWS>(it_zp);

            it_zp += zp_step;
            it_scale += QXNX_ROW_BLOCK_SIZE;

            // A column is only PARALLEL_ROWS==16 bf16 wide, i.e. half of a
            // 512-bit vector register, so anything done one column at a time
            // runs the vector unit at half throughput and needs 32 live
            // quarter-registers (the allocator spilled).  Everything below
            // works on 64-lane vectors instead: full-width ops, and one
            // register per value rather than a quarter of one.
            static constexpr int LANES = 4 * PARALLEL_ROWS;

            // Rows 0-7 of the 8 columns in a group go to w_ptr_iter_0 and rows
            // 8-15 to w_ptr_iter_1.  Splitting the rows before the multiply
            // rather than after means lane i of a half is always row i % 8, so
            // each half just needs its own 8 scales broadcast 8 times -- and it
            // leaves the accumulator feeding the store directly, which lets the
            // final convert fold into the store itself.
            // Feeding the per-row vector through the very same unzip the data
            // takes lands each half's 8 values, repeated 8 times, exactly where
            // that half needs them -- two shuffles, where splatting an 8-element
            // slice out to 64 lanes costs a pile of shifts and selects.
            auto splat_rows = [](aie::vector<bf16, PARALLEL_ROWS> v) {
                aie::vector<bf16, 2 * PARALLEL_ROWS> v32 = aie::concat(v, v);
                aie::vector<bf16, LANES> v64 = aie::concat(v32, v32);
                return aie::interleave_unzip(v64, v64, 8);
            };
            auto scale = splat_rows(a_scales);

            // out = col * scale + min  (min == 0 when no zero-point).  Seeding
            // the accumulator with the zero-point collapses the multiply and
            // the add into one mac and lifts the has_zp branch out of the
            // per-column work.
            auto min = splat_rows(a_mins);
            aie::accum<accfloat, LANES> base_lo;
            aie::accum<accfloat, LANES> base_hi;
            base_lo.from_vector(min.first);
            base_hi.from_vector(min.second);
            const aie::vector<bf16, LANES>& scale_lo = scale.first;
            const aie::vector<bf16, LANES>& scale_hi = scale.second;

            // Each quant load already holds exactly the 8 columns that make up
            // one output store pair, so a group is self contained: nothing from
            // it stays live past its own store and the allocator never spills.
            auto emit_group = [&](const aie::vector<bf16, PARALLEL_ROWS * 8>& d) {
                auto rows = aie::interleave_unzip(d.extract<LANES>(0), d.extract<LANES>(1), 8);
                aie::store_v(w_ptr_iter_0, aie::mac(base_lo, rows.first,  scale_lo).to_vector<bf16>());
                aie::store_v(w_ptr_iter_1, aie::mac(base_hi, rows.second, scale_hi).to_vector<bf16>());
                w_ptr_iter_0 += 64;
                w_ptr_iter_1 += 64;
            };

            // Two groups per iteration, modulo scheduled.  The unpack chain
            // (load -> ups -> round -> convert) is longer than one group's worth
            // of work, so the stalls only disappear if the groups overlap; two
            // per body plus the loop-back edge gives the scheduler that overlap
            // without spilling.
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE( 2, 2 )
            for (int pair = 0; pair < 2; pair++) {
                aie::vector<int8, PARALLEL_ROWS * 8> qa = aie::load_v<PARALLEL_ROWS * 8>(it_q);
                it_q += PARALLEL_ROWS * 8;
                aie::vector<int8, PARALLEL_ROWS * 8> qb = aie::load_v<PARALLEL_ROWS * 8>(it_q);
                it_q += PARALLEL_ROWS * 8;
                // int -> bf16 in a single to_float step (drops the intermediate
                // float vector and accum round-trip).
                aie::vector<bf16, PARALLEL_ROWS * 8> da = aie::to_float<bf16>(qa, 0);
                aie::vector<bf16, PARALLEL_ROWS * 8> db = aie::to_float<bf16>(qb, 0);
                emit_group(da);
                emit_group(db);
            }
            if constexpr (is_even) {
                _lock_acquire(w_cons_lock_id);
                is_w_ping = !is_w_ping;
                _lock_release(w_prod_lock_id);
                it_scale += QXNX_ROW_BLOCK_SIZE;
                it_zp += zp_step;
                w_ptr_iter_0 += 4 * 64;
                w_ptr_iter_1 += 4 * 64;
            }
        }
    }
}
#endif // __Q8_CORE_H__