// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "dequant_mm_bfp16.h"
#include "diag_ts.h"

extern "C" {

    void dequant_mm_o(
        bf16 * y_ping,
        bf16 * y_pong,
        bf16 * x_ping,
        bf16 * x_pong,
        uint8 * qw_ping,
        uint8 * qw_pong,
        uint8 * w_s0,
        uint8 * w_s1,
        float * y_acc,
        int * K_div_k,
        int * rtp_activate
    ) {
        // A row pair takes 48 n per pass, three 16-row halves of the block triple of its column: pair A (core rows
        // 2, 3) both halves of block 0 and half 0 of block 1, pair B (rows 4, 5) half 1 of block 1 and both halves
        // of block 2. Per k slice it receives two blocks (A: 0, 1; B: 1, 2); both cores of the pair dequantise k
        // half KH (this core: 1) of the three halves into the shared regions, rendezvous, then each multiplies its
        // own m x rows (even core: the first m rows of the pair's block, odd core: the next m) against all 48 n.
        // Rows per core of this dispatch, in m_slice (16-row) groups: RTP word 1; the pair (0 = A, 1 = B): RTP
        // word 2. The host sequence writes them while this core is already in its next call, parked on the first
        // weight tile, so they are read only once the weights of the dispatch have arrived (below). Unwritten /
        // out of range = the full block.
        // every conversion of this call (Q4_K scale/min and weight narrowing, the weight and x bfp16 conversion, the
        // y narrowing) rounds to nearest even: the design's default mode, floor, biases each of them downwards.
        // The mode is put back before the call returns.
        const aie::rounding_mode kernel_rnd = aie::swap_rounding(aie::rounding_mode::conv_even);
        TS_DECL
        TSG();
        int m_div_m_slice = NPU_M_TILE / m_slice;
        // ping/pong parity: every call toggles the qw flag an even number of times, so it is `false` at every call
        // start; x takes one buffer per m16 group and k slice (an odd count for an odd K_div_k x groups) and y three
        // per m16 group (odd for an odd group count), so theirs carries over between calls (kept in globals, in
        // registers during the call: the global byte flags cost ~30 cycles per toggle)
        bool is_qw_ping = false, is_x_ping = ::is_x_ping, is_y_ping = ::is_y_ping;
        TSG();
        int k_div_k_slice = NPU_K_TILE / k_slice;
        // regions: 0, 1 = w_s0 halves, 2 = the front of w_s1; pair B's n order is 2, 0, 1
        bool pb = false;
        const uint8 *r0 = w_s0, *r1 = w_s0 + W_REGION_BYTES, *r2 = w_s1;

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                for (int p = 0; p < 2; p++){
                    _down_lock_acquire(qw_cons_lock, 1);
                    TS();
                    if (k == 0 && p == 0 && j == 0) {
                        pb = ((volatile int *)K_div_k)[2] == 1;
                        r0 = pb ? w_s1 : w_s0;
                        r1 = pb ? w_s0 : w_s0 + W_REGION_BYTES;
                        r2 = pb ? w_s0 + W_REGION_BYTES : w_s1;
                    }
                    is_qw_ping = !is_qw_ping;
                    uint8* qw_using = is_qw_ping ? qw_ping : qw_pong;
#ifndef DIAG_NO_DEQ
                    // pair A: block 0 whole, then half 0 of block 1; pair B: half 1 of block 1, then block 2 whole
                    if ((p == 0) != pb) {
                        _Q4_K_bfp16_kh<1>((bfp16ebs8*)w_s0, qw_using, 0);
                        _Q4_K_bfp16_kh<1>((bfp16ebs8*)(w_s0 + W_REGION_BYTES), qw_using, 1);
                    } else {
                        _Q4_K_bfp16_kh<1>((bfp16ebs8*)w_s1, qw_using, pb ? 1 : 0);
                    }
#endif
                    TS();
                    _down_lock_release(qw_prod_lock, 1);
                }

                // Barrier A: both k halves of the three regions are written.
                _down_lock_release(sync_lock, 1);
                _lock_acquire(sync_lock, 1);
                TS();

                {
                    int mr = ((volatile int *)K_div_k)[1];
                    m_div_m_slice = ((unsigned)(mr - 1) < (unsigned)(NPU_M_TILE / m_slice)) ? mr : NPU_M_TILE / m_slice;
                }
                for (int i = 0; i < m_div_m_slice; i++){
                    // one DMA buffer holds the m16 group's whole 256 k slice of bfp16 A tiles, multiplied in place;
                    // the next group streams into the other buffer meanwhile
                    _lock_acquire(x_cons_lock);
                    TS();
                    is_x_ping = !is_x_ping;
                    const uint8 *xc = (const uint8 *)(is_x_ping ? x_ping : x_pong);
#ifndef DIAG_NO_MAC
                    if (k == 0 && j == 0) mm_aie_bfp16<true>(y_acc + i * GROUP_ACC, xc, r0, r1, r2);
                    else mm_aie_bfp16<false>(y_acc + i * GROUP_ACC, xc, r0, r1, r2);
#endif
                    _lock_release(x_prod_lock);
                    TS();
                }
                // Barrier B: neither core reads the regions any more.
                _down_lock_release(sync_lock, 1);
                _lock_acquire(sync_lock, 1);
                TS();
            }
            k += 1;
        } while (k < K_div_k[0]);
        ::is_x_ping = is_x_ping;
        TS();
        int use_activation = rtp_activate[0];
        const int y_num_rt = m_div_m_slice * (GROUP_ACC / y_slice);
        for (int i = 0; i < y_num_rt; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;

            passThrough_aie<float, bf16>(y_acc + i * y_slice, y);
            unpermute_chunk(y);
            if (i == 0 || i == 7 || i == 15 || i == 23 || i == y_num_rt - 1) TS();
            if (i == y_num_rt - 1) { TS_EMIT(y, i); TS_DONE(); }
            MAP_EMIT(y, i);
            if (use_activation == 1){
                silu_aie(y, y, y_slice);
            }
            else if (use_activation == 2){
                gelu_aie(y, y, y_slice);
            }
            _lock_release(y_cons_lock);
        }
        ::is_y_ping = is_y_ping;
        aie::set_rounding(kernel_rnd);
    }
}
