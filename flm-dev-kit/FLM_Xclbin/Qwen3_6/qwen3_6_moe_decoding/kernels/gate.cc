// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "lut_based_ops.h"

template<int vec_size>
aie::vector<bfloat16, vec_size> getSigmoidBf16_nonLUT(aie::vector<bfloat16, vec_size> x){
  

  const bfloat16 half = 0.5f;
  const bfloat16 one = 1.0f;
  aie::vector<bfloat16, vec_size> v_half = aie::broadcast<bfloat16, vec_size>(half);
  aie::vector<bfloat16, vec_size> v_one = aie::broadcast<bfloat16, vec_size>(one);
  aie::accum<accfloat, vec_size> x_mul_half =aie::mul(x, v_half);
  aie::vector<bfloat16, vec_size> tanh_x_half = aie::tanh<bfloat16>(x_mul_half.template to_vector<float>());

  aie::vector<bfloat16, vec_size> tanh_x_half_plus_one = aie::add(tanh_x_half, v_one);
  aie::vector<bfloat16, vec_size> tanh_x_half_plus_one_div_2 = aie::mul(tanh_x_half_plus_one, v_half);

  return tanh_x_half_plus_one_div_2;


}

template<int L>
void activate_gate(bf16* gate){
    bf16* gate_ptr = gate;
    for (int i = 0; i < L; i += 16) {
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr);
        gate_vec = getSigmoidBf16_nonLUT<16>(gate_vec);

        aie::store_v(gate_ptr, gate_vec);
        gate_ptr += 16;
    }
}


template<int L>
void activate_gate_sliu(bf16* gate){
    bf16* gate_ptr = gate;
    for (int i = 0; i < L; i += 16) {
        aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr);
        gate_vec = getActivationBf16(gate_vec);
        aie::store_v(gate_ptr, gate_vec);
        gate_ptr += 16;
    }
}

template<int L>
void apply_gate(bf16 * y, const bf16 * gate, const bf16 * attn_out){
    bf16* gate_ptr_0 = const_cast<bf16*>(gate);
    bf16* gate_ptr_1 = const_cast<bf16*>(gate) + 16;
    bf16* attn_out_ptr_0 = const_cast<bf16*>(attn_out);
    bf16* attn_out_ptr_1 = const_cast<bf16*>(attn_out) + 16;
    bf16* y_ptr_0 = y;
    bf16* y_ptr_1 = y + 16;
    for (int i = 0; i < L; i += 32) {
        aie::vector<bf16, 16> gate_vec_0 = aie::load_v<16>(gate_ptr_0);
        aie::vector<bf16, 16> gate_vec_1 = aie::load_v<16>(gate_ptr_1);
        aie::vector<bf16, 16> attn_out_vec_0 = aie::load_v<16>(attn_out_ptr_0);
        aie::vector<bf16, 16> attn_out_vec_1 = aie::load_v<16>(attn_out_ptr_1);

        aie::vector<bf16, 16> y_vec_0 = aie::mul(gate_vec_0, attn_out_vec_0);
        aie::vector<bf16, 16> y_vec_1 = aie::mul(gate_vec_1, attn_out_vec_1);

        aie::store_v(y_ptr_0, y_vec_0);
        aie::store_v(y_ptr_1, y_vec_1);
        gate_ptr_0 += 32;
        gate_ptr_1 += 32;
        attn_out_ptr_0 += 32;
        attn_out_ptr_1 += 32;
        y_ptr_0 += 32;
        y_ptr_1 += 32;
    }
}

// Fused: sigmoid(gate) * attn_out in one pass — avoids an extra gate buffer round-trip
template<int L>
void activate_and_apply_gate_sigmoid(bf16 * y, bf16 * gate, const bf16 * attn_out){
    bf16* gate_ptr_0 = gate;
    bf16* gate_ptr_1 = gate + 16;
    bf16* attn_out_ptr_0 = const_cast<bf16*>(attn_out);
    bf16* attn_out_ptr_1 = const_cast<bf16*>(attn_out) + 16;
    bf16* y_ptr_0 = y;
    bf16* y_ptr_1 = y + 16;
    for (int i = 0; i < L; i += 32) {
        aie::vector<bf16, 16> gate_vec_0 = aie::load_v<16>(gate_ptr_0);
        aie::vector<bf16, 16> gate_vec_1 = aie::load_v<16>(gate_ptr_1);
        aie::vector<bf16, 16> sig_vec_0 = getSigmoidBf16_nonLUT<16>(gate_vec_0);
        aie::vector<bf16, 16> sig_vec_1 = getSigmoidBf16_nonLUT<16>(gate_vec_1);
        aie::vector<bf16, 16> attn_out_vec_0 = aie::load_v<16>(attn_out_ptr_0);
        aie::vector<bf16, 16> attn_out_vec_1 = aie::load_v<16>(attn_out_ptr_1);
        aie::vector<bf16, 16> y_vec_0 = aie::mul(sig_vec_0, attn_out_vec_0);
        aie::vector<bf16, 16> y_vec_1 = aie::mul(sig_vec_1, attn_out_vec_1);
        aie::store_v(y_ptr_0, y_vec_0);
        aie::store_v(y_ptr_1, y_vec_1);
        gate_ptr_0 += 32;
        gate_ptr_1 += 32;
        attn_out_ptr_0 += 32;
        attn_out_ptr_1 += 32;
        y_ptr_0 += 32;
        y_ptr_1 += 32;
    }
}

// Fused: silu(gate) * attn_out in one pass
template<int L>
void activate_and_apply_gate_silu(bf16 * y, bf16 * gate, const bf16 * attn_out){
    bf16* gate_ptr_0 = gate;
    bf16* gate_ptr_1 = gate + 16;
    bf16* attn_out_ptr_0 = const_cast<bf16*>(attn_out);
    bf16* attn_out_ptr_1 = const_cast<bf16*>(attn_out) + 16;
    bf16* y_ptr_0 = y;
    bf16* y_ptr_1 = y + 16;
    for (int i = 0; i < L; i += 32) {
        aie::vector<bf16, 16> gate_vec_0 = aie::load_v<16>(gate_ptr_0);
        aie::vector<bf16, 16> gate_vec_1 = aie::load_v<16>(gate_ptr_1);
        aie::vector<bf16, 16> act_vec_0 = getActivationBf16(gate_vec_0);
        aie::vector<bf16, 16> act_vec_1 = getActivationBf16(gate_vec_1);
        aie::vector<bf16, 16> attn_out_vec_0 = aie::load_v<16>(attn_out_ptr_0);
        aie::vector<bf16, 16> attn_out_vec_1 = aie::load_v<16>(attn_out_ptr_1);
        aie::vector<bf16, 16> y_vec_0 = aie::mul(act_vec_0, attn_out_vec_0);
        aie::vector<bf16, 16> y_vec_1 = aie::mul(act_vec_1, attn_out_vec_1);
        aie::store_v(y_ptr_0, y_vec_0);
        aie::store_v(y_ptr_1, y_vec_1);
        gate_ptr_0 += 32;
        gate_ptr_1 += 32;
        attn_out_ptr_0 += 32;
        attn_out_ptr_1 += 32;
        y_ptr_0 += 32;
        y_ptr_1 += 32;
    }
}

constexpr int gate_prod_lock = 0;
constexpr int gate_cons_lock = 1;
constexpr int attn_out_prod_lock = 2;
constexpr int attn_out_cons_lock = 3;
constexpr int gated_attn_out_prod_lock = 4;
constexpr int gated_attn_out_cons_lock = 5;
constexpr int rtp_available_lock = 6;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
extern "C" {

void gate(
    bf16 * y,
    bf16 * gate,
    bf16 * attn_out
)
{
    const int repeats = MODEL_DIM / QXNX_ROW_BLOCK_SIZE / MVM_CORES;
    // _lock_acquire(rtp_available_lock);

    for (int layer = 0; layer < LAYER_PERIOD; layer++)
    if (layer == LAYER_PERIOD - 1){
        _lock_acquire(gate_cons_lock);
        _lock_acquire(attn_out_cons_lock);
        _lock_acquire(gated_attn_out_prod_lock, repeats);

        activate_and_apply_gate_sigmoid<DQ>(y, gate, attn_out);

        _lock_release(gate_prod_lock);
        _lock_release(attn_out_prod_lock);
        _lock_release(gated_attn_out_cons_lock, repeats);
    }
    else {
        _lock_acquire(gate_cons_lock);
        _lock_acquire(attn_out_cons_lock);
        _lock_acquire(gated_attn_out_prod_lock, repeats);

        activate_and_apply_gate_silu<Linear_DV>(y, gate, attn_out);

        _lock_release(gate_prod_lock);
        _lock_release(attn_out_prod_lock);
        _lock_release(gated_attn_out_cons_lock, repeats);
    }
}
}