// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "common.h"
#include "q4_k.h"

template<int m>
void zero_y(float * y) {
    zero_vectorized<float, m>(y);
}

///@brief pseduo q projection: y = linear(w, x)
///@param y: output
///@param x: input
///@param w: weight
template<int m, int k>
void pseduo_proj(
    float * y,
    const bf16 * x,
    const bf16 * w,
    const int offset
) {

    for (int i = 0; i < m; i++){
        float acc = y[i];
        for (int j = 0; j < k; j++){
            acc += w[i * k + j] * x[j + offset];
        }
        y[i] = acc;
    }
}

template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}
constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero

float y_acc[32] = {0};
template<int M, int K, int m, int k>
void linear_proj_iD(
    bf16 * x_ping,
    bf16 * x_pong,
    q4k_block * w_ping,
    q4k_block * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id
) {
    // Peano may reorder the pkt_id store past the triggering lock release, which
    // misroutes the packet-switched output -> deadlock. volatile pins the order.
    volatile uint32* pkt_id_ping = reinterpret_cast<volatile uint32*>(y_ping + 14);
    volatile uint32* pkt_id_pong = reinterpret_cast<volatile uint32*>(y_pong + 14);
    for (int i = 0; i < M / m / 2; i++) {
        zero_y<m>(y_acc);
        for (int j = 0; j < K / k / 2; j++)
        {
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            _qmm_q4k_bf16<m, k>(w_ping, x_ping, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);

            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            _qmm_q4k_bf16<m, k>(w_pong, x_pong, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        _lock_acquire(y_prod_lock);
        *pkt_id_ping = pkt_id;
        copy_y<m>(y_ping + 16, y_acc);
        _lock_release(y_cons_lock);

        zero_y<m>(y_acc);
        for (int j = 0; j < K / k / 2; j++)
        {
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            _qmm_q4k_bf16<m, k>(w_ping, x_ping, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);

            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            _qmm_q4k_bf16<m, k>(w_pong, x_pong, y_acc);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        _lock_acquire(y_prod_lock);
        *pkt_id_pong = pkt_id;
        copy_y<m>(y_pong + 16, y_acc);
        _lock_release(y_cons_lock);
    }
}

extern "C" {
constexpr int total_cores = 16;
void proj(
    bf16 * y_ping,
    q4k_block * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    q4k_block * w_pong,
    bf16 * x_pong
)
{
    constexpr int m = 32;
    constexpr int k = 256;
    // phase 1: qkv proj, D -> (D + D / 2) to RoPE
    linear_proj_iD<(D + DK + DV) / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 0);

    // // phase 2: o proj, D -> D to  RMS
    linear_proj_iD<D / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 1);

    // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
    linear_proj_iD<2 * HIDDEN_SIZE / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 2);

    // // phase 4.1 down-> 4D -> D, first D -> D
    linear_proj_iD<D / total_cores, HIDDEN_SIZE, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 3);

}


}