// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"


template<int L>
void pseduo_glu(bf16 * y, const bf16 * x) {
    bf16* gate_ptr = const_cast<bf16*>(x) + (L / 2);
    bf16* hid_ptr = const_cast<bf16*>(x);
    bf16* y_ptr = y;
    for (int i = 0; i < L / 2; i += 16) {
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr + i);
        aie::vector<bf16, 16> hid_vec = aie::load_v<16>(hid_ptr + i);
        gate_vec = getActivationBf16(gate_vec);

        aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, hid_vec);
        aie::store_v(y_ptr + i, y_vec);
    }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
extern "C" {
AIE_PERSIST_ZERO static bool is_x_ping = false;
void glu(
    bf16 * y,
    const bf16 * x_ping,
    const bf16 * x_pong
)
{
    FLM_SET_FLOOR_ROUNDING();
    KERNEL_RNE_SCOPE(5)
    static_assert((2 * INTERMEDIATE_SIZE) % GLU_SLICE == 0, "GLU_SLICE must divide 2 * INTERMEDIATE_SIZE");
    constexpr int chunks = 2 * INTERMEDIATE_SIZE / GLU_SLICE; // number of chunks in one GLU, each chunk is GLU_SLICE in size
    _lock_acquire(y_prod_lock, O_DOWN_REPEATS);

    bf16* y_it = y;

    for (int i = 0; i < chunks; i++) {
        is_x_ping = !is_x_ping;
        const bf16* x_using = is_x_ping ? x_ping : x_pong;
        _lock_acquire(x_cons_lock);
        pseduo_glu<GLU_SLICE>(y_it, x_using);
        y_it += GLU_SLICE / 2;
        _lock_release(x_prod_lock);
    }
    
    _lock_release(y_cons_lock, O_DOWN_REPEATS);
}


}