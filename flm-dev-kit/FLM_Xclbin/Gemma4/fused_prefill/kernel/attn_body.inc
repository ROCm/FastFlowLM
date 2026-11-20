// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

// Shared flash-attention prefill body for the fused Gemma4 array.
//
// This is the merge of the two standalone Gemma4 prefill kernels
// (attention_DH_512_prefill and attention_DH_256_prefill).  They differed in
// four ways, all of which are now parameters:
//   * geometry (lq / lk / dh / lk_mt), from common.h's variant macro block;
//   * the microkernel header (8-wide vs 16-wide), included by the .cc;
//   * DH256's extra reorder_s() before calculate_l(), behind ATTN_HAS_REORDER_S;
//   * the o drain, which DH256 wrote as a 2-level loop.  DH512 has lq == 8, so
//     the outer `o_row < lq / 8` loop degenerates to a single iteration and the
//     2-level form reproduces the DH512 drain exactly -- one code path serves
//     both.
//
// The one genuinely new thing here is that a fused column is a CU on its own:
// four cores in one column rather than 16 (DH512) or 8 (DH256) spread over
// several.  So a q round advances by q_block = 4 * lq tokens, not the 128 both
// standalone kernels hard-coded, and the core's own q slice is picked by row
// alone -- the `col` argument that used to select a half/quarter of the staged
// q is unused.
//
// Requires, before inclusion: ATTN_FUNC_NAME, common.h, a microkernel header.

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int in_prod_lock = 2;
constexpr int in_cons_lock = 3;
constexpr int o_prod_lock = 4;
constexpr int o_cons_lock = 5;

extern "C" {
    void ATTN_FUNC_NAME(
        bf16 *__restrict o_ping,
        bf16 *__restrict o_pong,
        bf16 *__restrict q,
        bf16 *__restrict in_ping,
        bf16 *__restrict in_pong,
        bf16 * s,
        bf16 * m,
        float * y,
        int * L_begin_buffer,
        int * L_end_buffer,
        int * window_size_buffer,
        const int row,
        const int col
    ){
        (void)col;   // a fused column is a whole CU; the row alone picks the q slice

        // Force into .data (file-backed) rather than .bss: under Peano (llvm-aie)
        // the aie-rt CDO loader does not reliably zero a .bss tail, leaving these
        // persistent ping-pong toggles as garbage on load. (chess is fine.)
        AIE_PERSIST_ZERO static bool is_in_ping = false;
        AIE_PERSIST_ZERO static bool is_o_ping = false;

        alignas(aie::vector_decl_align) bf16 prev_m_local[lq];
        alignas(aie::vector_decl_align) bf16 new_m_local[lq];
        alignas(aie::vector_decl_align) float c[lq];
        alignas(aie::vector_decl_align) float l[lq];
        alignas(aie::vector_decl_align) bf16 l_bf16[lq];

        // NOT static: a function-local static is lazily initialized via a guard
        // byte in .bss, which Peano's CDO loader may not zero (leaving the guard
        // non-zero -> init skipped -> garbage vectors). Construct each call; it's
        // a cheap broadcast with no static-init dependency.
        AIE_TRANSIENT const aie::vector<bf16, lq> neg_inf = aie::broadcast<bf16, lq>(-0x1.FEp127f);
        AIE_TRANSIENT const aie::vector<float, lq> one = aie::broadcast<float, lq>(1.0f);

        int i = 0;
        int offset;
        int L_div_l;
        int pointer_block_q = 0;
        int pointer_block_inner_q = 0;
        int pointer_block_k = 0;
        int pointer_block_inner_k = 0;
        int block_location = 0;
        int pointer_block_inner_k_current = 0;
        do{
            aie::store_v(prev_m_local, neg_inf);
            aie::store_v(new_m_local, neg_inf);
            aie::store_v(c, one);
            zero_vectorized<float, lq>(l);
            zero_vectorized<float, lq * dh>(y);

            _lock_acquire(q_cons_lock);
            offset = L_begin_buffer[0] / q_block;
            // determine lq: this core owns q_block tokens' worth of round, and
            // within it the `row`-th run of lq.
            pointer_block_q = L_begin_buffer[0] + i * q_block;
            pointer_block_inner_q = pointer_block_q + row * lq;
            // determine lk
            int window_size = window_size_buffer[0];
            pointer_block_k = pointer_block_q - window_size;
            pointer_block_k = pointer_block_k > 0 ? pointer_block_k : 0;
            // The host streams k/v in whole lk_mt blocks, so the first block this
            // round reads has to start on an lk_mt boundary.  Rounding down only
            // ever pulls in keys that are older than the window; those are masked
            // out by apply_mask_and_get_max just like any other out-of-range key.
            pointer_block_k = (pointer_block_k / lk_mt) * lk_mt;
            // determine llk
            pointer_block_inner_k = pointer_block_inner_q - window_size;
            pointer_block_inner_k = pointer_block_inner_k > -lk ? pointer_block_inner_k : -lk;

            // Blocks of lk_mt keys spanning [pointer_block_k, pointer_block_q +
            // q_block).  With the standalone geometry (q_block == lk_mt == 128,
            // both pointers block-aligned) this is exactly the old
            // (pbq - pbk) / 128 + 1.
            const int n_blocks = (pointer_block_q + q_block - pointer_block_k + lk_mt - 1) / lk_mt;
            for (int block_idx = 0; block_idx < n_blocks; block_idx++){
                for (int j = 0; j < lq; j++){
                    aie::vector<bf16, lk> m_vec = aie::broadcast<bf16, lk>(*(prev_m_local + j));
                    aie::store_v(m + j * lk, m_vec);
                }
                for (int j = 0; j < n_kblk; j++){
                    block_location = block_idx * n_kblk + j;
                    pointer_block_inner_k_current = pointer_block_k + block_location * lk;
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *k = is_in_ping ? in_ping : in_pong;
                    _attn_qk<bf16, bf16, lq/8, dh/8, lk/8, 8, 8, 8>(s + j * lq * lk, q, k);
                    _lock_release(in_prod_lock);
                    apply_mask_and_get_max(s + j * lq * lk, m, pointer_block_inner_k, pointer_block_inner_q, pointer_block_inner_k_current);
                }
                for (int j = 0; j < lq; j++){
                    aie::vector<bf16, lk> m_vec = aie::load_v<lk>(m + j * lk);
                    bf16 vm = aie::reduce_max(m_vec);
                    *(new_m_local + j) = (bf16)vm;
                }
                apply_softmax(s, new_m_local);
                calculate_c(c, prev_m_local, new_m_local);
#ifdef ATTN_HAS_REORDER_S
                reorder_s(s);
#endif
                calculate_l(l, c, s);
                calculate_y(y, c);
                for (int j = 0; j < n_kblk; j++){
                    _lock_acquire(in_cons_lock);
                    is_in_ping = !is_in_ping;
                    bf16 *v = is_in_ping ? in_ping : in_pong;
                    _attn_fv<bf16, float, lq/8, lk/8, dh/8, 8, 8, 8>(y, s + j * lq * lk, v);
                    _lock_release(in_prod_lock);
                }
                aie::vector<bf16, lq> prev_m_local_vec = aie::load_v<lq>(new_m_local);
                aie::store_v(prev_m_local, prev_m_local_vec);
            }
            // q is finished with once the block loop ends; hand the buffer back
            // before the (slow) o drain so the memtile can stage the next round.
            _lock_release(q_prod_lock);
            aie::vector<float, lq> l_vec = aie::load_v<lq>(l);
            l_vec = aie::inv(l_vec);
            aie::accum<accfloat, lq> l_acc;
            l_acc.from_vector(l_vec);
            aie::store_v(l_bf16, l_acc.template to_vector<bf16>());
            for (int o_row = 0; o_row < lq / 8; o_row++){
                for (int o_col = 0; o_col < dh / 8; o_col++){
                    _lock_acquire(o_prod_lock);
                    is_o_ping = !is_o_ping;
                    bf16 *o = is_o_ping ? o_ping : o_pong;
                    scale_div_aie(o, l_bf16 + o_row * 8, y + o_row * 8 * dh + o_col * 64);
                    _lock_release(o_cons_lock);
                }
            }
            i++;
            L_div_l = L_end_buffer[0] / q_block;
            L_div_l -= offset;
        } while (i < L_div_l);
    }
}
