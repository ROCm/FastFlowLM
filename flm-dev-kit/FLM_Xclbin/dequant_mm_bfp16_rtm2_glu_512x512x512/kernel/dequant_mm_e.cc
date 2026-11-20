// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "dequant_mm_bfp16.h"
#include "diag_ts.h"

extern "C" {

    void dequant_mm_e(
        bf16 * y_ping,
        bf16 * y_pong,
        bf16 * x_ping,
        bf16 * x_pong,
        uint8 * qw_ping,
        uint8 * qw_pong,
        uint8 * w_s0,
        uint8 * w_s1,
        uint8 * xc,
        float * y_acc,
        int * K_div_k,
        int * rtp_activate
    ) {
        // Both cores of a row pair dequantise their n pair of BOTH n slices of the
        // k slice into bfp16 tiles (slice 0 in the even tile, slice 1 in the odd
        // tile), rendezvous, then each multiplies its own 128 x rows against the 64 n
        // of both slices: every x chunk is streamed once per k slice, not once per
        // n slice.
        // Rows per core row of this dispatch, in m_slice (16-row) groups: RTP word 1. The host sequence
        // writes it (and rewrites the memtile BD lengths to match) while this core is already in its next
        // call, parked on the first weight tile, so it is read only once the weights of the dispatch have
        // arrived (below). Unwritten / out of range = the full block.
        // every conversion of this call (Q4_K scale/min and weight narrowing, the weight and x bfp16 conversion, the
        // y narrowing) rounds to nearest even: the design's default mode, floor, biases each of them downwards.
        // The mode is put back before the call returns, as the GLU epilogue below already did for itself.
        const aie::rounding_mode kernel_rnd = aie::swap_rounding(aie::rounding_mode::conv_even);
        TS_DECL
        TSG();
        int m_div_m_slice = NPU_M_TILE / m_slice;
        // ping/pong parity: every call toggles each flag an even number of times, so it is `false` at every call
        // start; local flags stay in registers (the global byte flags cost ~30 cycles per toggle)
        bool is_qw_ping = false, is_x_ping = false, is_y_ping = false;
        TSG();
        int k_div_k_slice = NPU_K_TILE / k_slice;

        int k = 0;
        do {
            for (int j = 0; j < k_div_k_slice; j++){
                for (int p = 0; p < 2; p++){
                    _lock_acquire(qw_cons_lock, 1);
                    TS();
                    is_qw_ping = !is_qw_ping;
                    uint8* qw_using = is_qw_ping ? qw_ping : qw_pong;
                    uint8* w_slice = p == 0 ? w_s0 : w_s1;
#ifndef DIAG_NO_DEQ
                    _Q4_K_bfp16((bfp16ebs8*)(w_slice + 0 * W_REGION_BYTES), qw_using, 0);
#endif
                    TS();
                    _lock_release(qw_prod_lock, 1);
                }

                // Barrier A: both halves of both w slices are written.
                _up_lock_release(sync_lock, 1);
                _lock_acquire(sync_lock, 1);
                TS();

                {
                    int mr = ((volatile int *)K_div_k)[1];
                    m_div_m_slice = ((unsigned)(mr - 1) < (unsigned)(NPU_M_TILE / m_slice)) ? mr : NPU_M_TILE / m_slice;
                }
                for (int i = 0; i < m_div_m_slice; i++){
                    // the two 128 k chunks of this m16 group arrive back to back; each DMA buffer is
                    // released as soon as it is converted, so the next chunk streams in during the multiply
                    for (int v = 0; v < 2; v++){
                        _lock_acquire(x_cons_lock);
                        TS();
                        is_x_ping = !is_x_ping;
                        const bf16 *x = is_x_ping ? x_ping : x_pong;
#ifndef DIAG_NO_CONV
                        convert_x_chunk(x, xc + v * XC_BYTES);
#else
                        // timing only: bounds what bfp16 x panels written by the host would save (results are garbage)
                        if (k == 0 && j == 0 && i == 0) convert_x_chunk(x, xc + v * XC_BYTES);
#endif
                        TS();
                        _lock_release(x_prod_lock);
                    }
#ifndef DIAG_NO_MAC
                    if (k == 0 && j == 0) mm_aie_bfp16<true>(y_acc + i * m_slice * NPU_N_TILE, xc, w_s0, w_s1);
                    else mm_aie_bfp16<false>(y_acc + i * m_slice * NPU_N_TILE, xc, w_s0, w_s1);
#endif
                    TS();
                }
                // Barrier B: neither core reads the w slices any more.
                _up_lock_release(sync_lock, 1);
                _lock_acquire(sync_lock, 1);
                TS();
            }
            k += 1;
        } while (k < K_div_k[0]);
        TS();
        int use_activation = rtp_activate[0];
        if (use_activation == 3){
            // one chunk per m8 row group: [up | gate] n8 tiles in, 4 fused n8 tiles out
            const int glu_rt = m_div_m_slice * 2;
            // every narrowing below rounds to nearest even (the design's default, floor, biases a chain of six
            // roundings by about -1%); the mode is put back for the next dispatch's x conversion / y narrowing
            const aie::rounding_mode old_rnd = aie::swap_rounding(aie::rounding_mode::conv_even);
            for (int i = 0; i < glu_rt; i++){
                _lock_acquire(y_prod_lock);
                is_y_ping = !is_y_ping;
                bf16 *y = is_y_ping ? y_ping : y_pong;
                glu_chunk_aie(y_acc + i * (8 * 64), y);
                _lock_release(y_cons_lock);
            }
            aie::set_rounding(old_rnd);
            aie::set_rounding(kernel_rnd);
            return;
        }
        const int y_num_rt = m_div_m_slice * (m_slice * NPU_N_TILE / y_slice);
        for (int i = 0; i < y_num_rt; i++){
            _lock_acquire(y_prod_lock);
            is_y_ping = !is_y_ping;
            bf16 *y = is_y_ping ? y_ping : y_pong;
            
            passThrough_aie<float, bf16>(y_acc + i * y_slice, y);
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
        aie::set_rounding(kernel_rnd);
    }
}