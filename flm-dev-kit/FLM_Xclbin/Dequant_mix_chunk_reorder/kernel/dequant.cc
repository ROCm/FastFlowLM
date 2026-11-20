// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "typedef.hpp"
#include "aie_kernel_utils.h"
#include "q4_core.h"
#include "q8_core.h"

constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_ping_lock = 4;
constexpr int y_prod_pong_lock = 5;
constexpr int y_cons_ping_lock = 7;
constexpr int y_cons_pong_lock = 8;

constexpr int rtp_available_lock = 6;

AIE_PERSIST_ZERO bool is_y_ping = false;
AIE_PERSIST_ZERO bool is_w_ping = false;

template <bool has_zp>
void mvm_Q4(
    int M, int K,
    uint8 * w_ping,
    uint8 * w_pong,
    bf16 * w16_ping,
    bf16 * w16_pong,
    uint8 * w8_buffer,
    bf16* scale_buffer,
    bf16* zp_buffer,
    int& rtp_transpose
) {

    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;

    for (int i = 0; i < M / m; i++) {
        for (int j = 0; j < K / k; j++) {
            is_y_ping = !is_y_ping;
            bf16* y_using = is_y_ping ? w16_ping : w16_pong;
            if (is_y_ping) {
                _lock_acquire(y_prod_ping_lock);
            }
            else {
                _lock_acquire(y_prod_pong_lock);
            }

            _Q4_x_bf16<has_zp>(
                y_using,
                scale_buffer,
                zp_buffer,
                w_ping,
                w_pong,
                w8_buffer,
                is_w_ping,
                w_prod_lock,
                w_cons_lock,
                rtp_transpose
            );

            if (is_y_ping) {
                _lock_release(y_cons_ping_lock);
            }
            else {
                _lock_release(y_cons_pong_lock);
            }
        }
    }
}


template <bool has_zp>
void mvm_Q8(
    int M, int K,
    uint8 * w_ping,
    uint8 * w_pong,
    bf16 * w16_ping,
    bf16 * w16_pong,
    uint8* w8_buffer,
    bf16* scale_buffer,
    bf16* zp_buffer,
    int& rtp_transpose
) {

    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;

    for (int i = 0; i < M / m; i++) {
        for (int j = 0; j < K / k; j++) {

            is_y_ping = !is_y_ping;
            bf16* y_using = is_y_ping ? w16_ping : w16_pong;
            if (is_y_ping) {
                _lock_acquire(y_prod_ping_lock);
            }
            else {
                _lock_acquire(y_prod_pong_lock);
            }

            _Q8_x_bf16<has_zp>(
                y_using,
                scale_buffer,
                zp_buffer,
                (int8 *)w_ping,
                (int8 *)w_pong,
                (int8 *)w8_buffer,
                is_w_ping,
                w_prod_lock,
                w_cons_lock,
                rtp_transpose
            );

            if (is_y_ping) {
                _lock_release(y_cons_ping_lock);
            }
            else {
                _lock_release(y_cons_pong_lock);
            }
        }
    }
}


extern "C" {

void dequant(
    bf16 * y_ping,
    uint8 * w_ping,
    bf16 * y_pong,
    uint8 * w_pong,
    bf16 * scale_buffer,
    bf16 * zp_buffer,
    uint8 * w8_buffer,
    int* Din,
    int* Dout,
    int* is_q4,
    int* is_transpose
)
{
    
    _lock_acquire(rtp_available_lock);
    if (is_q4[0] == 1){
        mvm_Q4<true>(Dout[0], Din[0], w_ping, w_pong, y_ping, y_pong, w8_buffer, scale_buffer, zp_buffer, is_transpose[0]);
    }
    else if (is_q4[0] == 2){
        mvm_Q4<false>(Dout[0], Din[0], w_ping, w_pong, y_ping, y_pong, w8_buffer, scale_buffer, zp_buffer, is_transpose[0]);
    }
    else if (is_q4[0] == 3){
        mvm_Q8<true>(Dout[0], Din[0], w_ping, w_pong, y_ping, y_pong, w8_buffer, scale_buffer, zp_buffer, is_transpose[0]);
    }
    else if (is_q4[0] == 4){
        mvm_Q8<false>(Dout[0], Din[0], w_ping, w_pong, y_ping, y_pong, w8_buffer, scale_buffer, zp_buffer, is_transpose[0]);
    }
}

}