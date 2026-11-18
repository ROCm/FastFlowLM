// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_prefill.hpp
/// \brief prefill graph for the hunyuan-dense engine
/// \author FastFlowLM Team
/// \date 2026-09-01
#ifndef __HUNYUAN_PREFILL_HPP__
#define __HUNYUAN_PREFILL_HPP__
#include <functional>
#include "hunyuan_npu_sequence.hpp"
#include "modules/gemm.hpp"
#include "hunyuan_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "hunyuan_overlay.hpp"
#include "hunyuan_bfp16.hpp"
#include "hunyuan_run_trace.hpp"

// ---------------------------------------------------------------------------
// Backend selection.
//
// Every stage defaults to the array. Setting any of these to 0 falls back to
// the host implementation of that stage alone, which is what the bring-up and
// golden-diff procedure in README.md uses -- nothing else has to change,
// because the host matmuls read the same reordered q4_k bytes the array does
// and every buffer is already a device BO.
//
// Two xclbins back these: layer.xclbin for decode + lm_head, and
// prefill_overlay.xclbin for prefill MM + prefill attention (one bitstream; the
// two take turns on the whole array, ~0.12 ms per switch).
// ---------------------------------------------------------------------------
#ifndef HUNYUAN_MM_ON_NPU
#define HUNYUAN_MM_ON_NPU 1
#endif
#ifndef HUNYUAN_ATTN_ON_NPU
#define HUNYUAN_ATTN_ON_NPU 1
#endif
#ifndef HUNYUAN_LM_HEAD_ON_NPU
#define HUNYUAN_LM_HEAD_ON_NPU 1
#endif
#ifndef HUNYUAN_DECODE_ON_NPU
#define HUNYUAN_DECODE_ON_NPU 1
#endif

// Prefill backend, selected with -DMVPREFILL at build time. Either way a prompt
// of at most HUNYUAN_MV_PREFILL_MAX_TOKENS tokens goes through the decode
// overlay one token at a time (matrix-vector), because the batched prefill's
// L-padded matmuls cost the same for 1 token as for 512. Set, that ceiling goes
// away and every prompt takes the mv path, however long it is.
#ifndef MVPREFILL
#define MVPREFILL 0
#endif

// ---------------------------------------------------------------------------
// Prefill profiling. Enable with -DHUNYUAN_PREFILL_PROFILE; compiles to nothing
// otherwise.
//
//   PREFILL_PROF_INIT();               // once at the start of a function
//   ... work ...
//   PREFILL_PROF_TICK("qkv_proj");     // time since the previous tick
//
// TICK resets the reference point, so a span that encloses several ticks needs
// its own mark:
//
//   PREFILL_PROF_MARK(total);          // an independent reference point
//   ... ticks ...
//   PREFILL_PROF_RATE("total", total, L);  // elapsed since the mark, + tok/s
// ---------------------------------------------------------------------------
#ifdef HUNYUAN_PREFILL_PROFILE
#define PREFILL_PROF_INIT() time_utils::time_point __prof_t = time_utils::now()
#define PREFILL_PROF_TICK(name)                                                \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::duration_us(__prof_t, __prof_now));\
        header_print("prefill-prof", name << ": " << __prof_d.first << " "     \
                                          << __prof_d.second);                 \
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PREFILL_PROF_MARK(var) time_utils::time_point __prof_m_##var = time_utils::now()
#define PREFILL_PROF_RATE(name, var, ntok)                                     \
    do {                                                                       \
        time_utils::time_with_unit __prof_us =                                 \
            time_utils::duration_us(__prof_m_##var, time_utils::now());        \
        time_utils::time_with_unit __prof_d = time_utils::re_unit(__prof_us);  \
        header_print("prefill-prof", name << ": " << __prof_d.first << " "     \
                     << __prof_d.second << " (" << (ntok) << " tokens, "       \
                     << ((ntok) * 1e6f / __prof_us.first) << " tok/s)");       \
    } while (0)
#else
#define PREFILL_PROF_INIT() do {} while (0)
#define PREFILL_PROF_TICK(name) do {} while (0)
#define PREFILL_PROF_MARK(var) do {} while (0)
#define PREFILL_PROF_RATE(name, var, ntok) do {} while (0)
#endif

/// @brief push host writes of elements [first, first + count) of @p b to the device,
///        leaving the rest of the buffer alone (same pattern as gemma4e_dflash_bo.hpp)
inline void hunyuan_flush_range(buffer<bf16>& b, size_t first, size_t count) {
    if (count == 0) return;
    const size_t off_bytes = first * sizeof(bf16);
    const size_t len_bytes = count * sizeof(bf16);
#if defined(FLM_USE_HRX)
    hrx::hrx_report(hrx_buffer_flush_range(b.bo().handle(), off_bytes, len_bytes), "hunyuan flush_range");
#else
    b.bo().sync(XCL_BO_SYNC_BO_TO_DEVICE, len_bytes, off_bytes);
#endif
}

/// @brief make device writes to elements [first, first + count) of @p b visible to the host
inline void hunyuan_invalidate_range(buffer<bf16>& b, size_t first, size_t count) {
    if (count == 0) return;
    const size_t off_bytes = first * sizeof(bf16);
    const size_t len_bytes = count * sizeof(bf16);
#if defined(FLM_USE_HRX)
    hrx::hrx_report(hrx_buffer_invalidate_range(b.bo().handle(), off_bytes, len_bytes), "hunyuan invalidate_range");
#else
    b.bo().sync(XCL_BO_SYNC_BO_FROM_DEVICE, len_bytes, off_bytes);
#endif
}

/// @brief write rows [0, L) of a matmul input as the bfp16 tiles the overlay reads
///        (hunyuan_bfp16::_bfp16_rne_tile_group, [L / 16][Din / 8][2][72 B]) and flush them.
///        cols(row, n, col0, ncols, out) writes columns [col0, col0 + ncols) of the n bf16 rows
///        from `row` on into out (row stride ncols). A piece of work is a 16-row group cut into
///        the fewest column pieces (of a multiple of 256) that give every thread ~4 pieces, made
///        and packed while in cache: a 12-token prompt still uses every thread, and 8 groups do
///        not leave threads idle. Rows of the last group past L are packed as zero, later groups
///        are not touched. A group is Din * 18 bytes, a whole number of 64-byte lines. See "x as
///        bfp16 tiles" in README.md.
template <typename Cols>
inline void hunyuan_write_x_tiles(buffer<u8>& tiles, int L, int Din, Cols cols) {
    const int groups = (L + 15) / 16;
    const int max_pieces = Din / hunyuan_bfp16::X_TILE_PIECE;
    assert(Din % hunyuan_bfp16::X_TILE_PIECE == 0);
    int pieces = 1;
    while (pieces < max_pieces && groups * pieces < 4 * max_prefill_thread) {
        do pieces++; while (max_pieces % pieces != 0);
    }
    const int ncols = Din / pieces;
    #pragma omp parallel for num_threads(max_prefill_thread) schedule(static)
    for (int u = 0; u < groups * pieces; u++) {
        const int g = u / pieces;
        const int c = u % pieces;
        thread_local std::vector<bf16> scratch;
        scratch.resize((size_t)16 * ncols);
        const int n = std::min(16, L - 16 * g);
        cols(16 * g, n, c * ncols, ncols, scratch.data());
        if (n < 16) memset(scratch.data() + (size_t)n * ncols, 0, (size_t)(16 - n) * ncols * sizeof(bf16));
        u8* dst = tiles.data() + hunyuan_bfp16::x_tiles_bytes((size_t)16 * g, Din) + hunyuan_bfp16::x_tiles_bytes(16, ncols) * c;
        hunyuan_bfp16::_bfp16_rne_tile_group(dst, scratch.data(), ncols, ncols);
    }
    hunyuan_bfp16::flush_bytes(tiles, 0, hunyuan_bfp16::x_tiles_bytes((size_t)16 * groups, Din));
}

/// @brief hunyuan_write_x_tiles for the RMSNorm of rows [0, L) of x (row stride D). A piece
///        computes the 1 / rms of its rows itself (again for every column piece of a row:
///        cheaper than another parallel pass)
inline void hunyuan_write_norm_tiles(buffer<u8>& tiles, const bf16* x, const bf16* w, int L, int D, float eps) {
    hunyuan_write_x_tiles(tiles, L, D, [&](int row, int n, int col0, int ncols, bf16* out) {
        for (int r = 0; r < n; r++) {
            const bf16* x_row = x + (size_t)(row + r) * D;
            cpu_func::_rms_norm_cols(out + (size_t)r * ncols, x_row + col0, w + col0, cpu_func::_rms_inv(x_row, D, eps), ncols);
        }
    });
}

/// @brief a zeroed (and flushed) tile buffer for `rows` rows of Din
inline buffer<u8> hunyuan_zero_tiles(npu_xclbin_manager* npu, size_t rows, size_t Din) {
    buffer<u8> tiles = npu->create_bo_buffer<u8>(hunyuan_bfp16::x_tiles_bytes(rows, Din));
    memset(tiles.data(), 0, tiles.size());
    tiles.sync_to_device();
    return tiles;
}

struct hunyuan_common_buffers{
    /// @brief input of the q and kv runs (host matmul build): the host writes rows [0, L),
    ///        rows [L, L_padded) stay zero (nothing else writes this buffer)
    buffer<bf16> x_attn_in;
    /// @brief rows of x_attn_in (or xa_tiles) that may hold a value from an earlier call
    int x_attn_in_rows = 0;
    /// @brief output of the o and down runs (and the up|gate input of the host matmul build)
    buffer<bf16> x;
    buffer<bf16> after_mix;
    /// @brief the overlay's matmul inputs as bfp16 tiles: q and kv (xa, rows past the
    ///        prompt stay zero like x_attn_in's), o (am, packed from after_mix), up|gate (xm)
    buffer<u8> xa_tiles;
    buffer<u8> am_tiles;
    buffer<u8> xm_tiles;

    hunyuan_common_buffers(){}
};

/// @brief one kind of run on the prefill overlay (q, kv, mha, o, up|gate, down)
/// @note  The switch in front of a run depends on what the array was configured
///        for, so there is one app per starting state (unknown / dq / attention),
///        each caching its own sequence: the first q of a prefill call (after the
///        decode layer ran) and the q of layers 1..31 are built once each, not
///        rebuilt every layer. See "Prefill overlay" in README.md.
struct overlay_role{
    npu_app apps[hunyuan_overlay::NUM_OPS];
    std::vector<uint32_t> built_for[hunyuan_overlay::NUM_OPS];

    void init(npu_app_manager* mgr){
        for (npu_app& a : apps) a = mgr->create_app();
    }

    /// @param from what the array is configured for when the run starts
    /// @param key  everything the sequence depends on besides `from`
    /// @param gen  gen(seq, from) writes the sequence
    template <typename Gen>
    npu_app& get(int from, const std::vector<uint32_t>& key, Gen gen){
        const int slot = from < 0 ? 0 : from;
        if (built_for[slot] != key){
            gen(apps[slot].seq(), from);
            built_for[slot] = key;
        }
        return apps[slot];
    }
};

/// @brief the overlay's runs and what the array is configured for right now
struct hunyuan_overlay_state{
    /// @brief -1 after anything else ran on the array (decode layer, lm_head):
    ///        the NPU reloads the overlay then and the next run must switch fully
    int configured = -1;
    /// @brief whose shared BDs the array holds (hunyuan_overlay::bd_owner_after);
    ///        part of every sequence key, since the switch in front depends on it
    int bd_owner = -1;
    overlay_role q, kv, mha, o, upgate, down;

    /// @brief a run of `op` was issued
    void ran(int op){
        bd_owner = hunyuan_overlay::bd_owner_after(op, configured, bd_owner);
        configured = op;
    }

    /// @brief after a run of `op` ended with `st`: nothing to do if it completed. Otherwise the
    ///        driver ended it (timeout) and reset the array, so the overlay state is gone as after
    ///        another xclbin: forget it and run the op again from a full switch (`again()`).
    ///        See "Run failures" in README.md.
    template <typename Again>
    void check(const char* what, int op, ert_cmd_state st, Again again){
        st = HY_TRACE_FAKE_FAIL(st);
        if (st == ERT_CMD_STATE_COMPLETED) return;
        std::cerr << "[hunyuan_npu] prefill run " << what << " ended with state " << (int)st
                  << ", not completed: running it again from a full switch" << std::endl;
        configured = -1;
        bd_owner = -1;
        st = again();
        HY_TRACE(what, 'R', (int)st);
        if (st != ERT_CMD_STATE_COMPLETED) {
            std::string message = std::string("hunyuan_npu: prefill run ") + what + " failed twice";
            if (!flm::npu::ert_failure_detail().empty()) {
                message += ": ";
                message += flm::npu::ert_failure_detail();
                flm::npu::ert_failure_detail().clear();
            }
            throw std::runtime_error(message);
        }
        ran(op);
    }
};

/// @brief one dequant_mm run from a full switch, waited for (the run again of check())
inline ert_cmd_state hunyuan_dq_from_full_switch(overlay_role& role, uint32_t Lm, uint32_t Din, uint32_t Dout,
                                                 size_t w_offset, int act_right,
                                                 buffer<bf16>& y, buffer<u8>& x, buffer<u8>& w){
    npu_app& app = role.get(-1, {Lm, 0u}, [&](npu_sequence* s, int from){
        hunyuan_overlay::gen_dq_run(s, from, Lm, Din, Dout, w_offset, 0, act_right, -1);
    });
    return app.run_for_state(y, x, w);
}

struct hunyuan_attn_block_prefill_context{
    hunyuan_desc* desc;
    hunyuan_npu_sequence* seq_gen;
    hunyuan_common_buffers* common_bufs;
    npu_xclbin_manager* npu;
    hunyuan_overlay_state* ovl;

    /// @brief rows the q/kv allocation was cut for; they are re-cut on any change
    int L_alloc = -1;
    /// @brief rows of q that may be non-zero. A small-row plan writes only Lg rows
    ///        of q, the attention reads L_padded: rows [Lg, L_padded) must be zero, as
    ///        the 128-row plan leaves them (it computes them from the zero rows of its x)
    int q_rows = 0;

    buffer<bf16> q;
    /// @brief k | v per row (k and v weights are adjacent, one run computes both)
    buffer<bf16> kv;

    hunyuan_attn_block_prefill_context(
        hunyuan_desc* desc,
        hunyuan_npu_sequence* seq_gen,
        hunyuan_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        hunyuan_overlay_state* ovl
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu), ovl(ovl){}

    /// @brief drop the cached attention sequences, e.g. after the kv cache was
    ///        resized: they hold the old quadrant offsets and row count
    void invalidate_mha(){
        for (auto& k : ovl->mha.built_for) k.clear();
    }

    /// @param L_begin  number of tokens already committed to the kv cache; the L
    ///                 new tokens occupy absolute positions L_begin .. L_begin+L-1
    /// @param Lg       rows the matmuls compute (pad_prefill_gemm_L)
    /// @param while_attn   host work to run while the attention is on the array
    buffer<bf16> forward(int L, int Lg, buffer<u8>& proj_weights, buffer<bf16>& rope_rms_weights,
                         buffer<bf16>& cache, std::vector<int>& input_idx, int L_begin,
                         const std::function<void()>& while_attn) {
        using namespace hunyuan_overlay;
        PREFILL_PROF_INIT();
        const int L_padded = (int)hunyuan_npu_sequence::pad_prefill_L((uint32_t)L);
        const uint32_t Lp = (uint32_t)L_padded;
        const uint32_t Lm = (uint32_t)Lg;
        [[maybe_unused]] const int dq_op = dq_op_for(Lm);
        const int DKV = desc->DK + desc->DV;

        if (L_alloc != L_padded) {
            q = npu->create_bo_buffer<bf16>((size_t)L_padded * desc->DQ);
            kv = npu->create_bo_buffer<bf16>((size_t)L_padded * DKV);
            L_alloc = L_padded;
            q_rows = L_padded;
        }
        if (q_rows > Lg) {
            const size_t first = (size_t)Lg * desc->DQ;
            const size_t count = (size_t)(q_rows - Lg) * desc->DQ;
            memset(q.data() + first, 0, count * sizeof(bf16));
            hunyuan_flush_range(q, first, count);
        }
        q_rows = L;

        bf16* q_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM;
        bf16* k_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM + desc->DH;

#if HUNYUAN_MM_ON_NPU
        const int owner = ovl->bd_owner;
        npu_app& q_app = ovl->q.get(ovl->configured, {Lm, (uint32_t)(owner + 1)}, [&](npu_sequence* s, int from){
            gen_dq_run(s, from, Lm, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0, owner);
        });
        const int kv_owner = bd_owner_after(dq_op, ovl->configured, owner);
        npu_app& kv_app = ovl->kv.get(dq_op, {Lm, (uint32_t)(kv_owner + 1)}, [&](npu_sequence* s, int from){
            gen_dq_run(s, from, Lm, desc->D, DKV, desc->get_attn_k_offset(), 0, 0, kv_owner);
        });
        PREFILL_PROF_TICK("attn:alloc+seq_get");
        HY_TRACE("q", 'S');
        const ert_cmd_state q_state = q_app.run_for_state(q, common_bufs->xa_tiles, proj_weights);
        HY_TRACE("q", 'D', (int)q_state);
        ovl->check("q", dq_op, q_state, [&](){
            return hunyuan_dq_from_full_switch(ovl->q, Lm, desc->D, desc->DQ, desc->get_attn_q_offset(), 0,
                                               q, common_bufs->xa_tiles, proj_weights);
        });
        ovl->ran(dq_op);
        PREFILL_PROF_TICK("attn:q_run");
        auto kv_run = kv_app.create_run(kv, common_bufs->xa_tiles, proj_weights);
        kv_run.start();
        HY_TRACE("kv", 'S');
        ovl->ran(dq_op);
        PREFILL_PROF_TICK("attn:kv_start");
        hunyuan_invalidate_range(q, 0, (size_t)L * desc->DQ);
        PREFILL_PROF_TICK("attn:q_sync_from");
        cpu_func::_rope_then_qk_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads,
                                          q_norm_weight, desc->DH, desc->inv_freq.data(), desc->rms_norm_eps);
        PREFILL_PROF_TICK("attn:q_rope_norm");
        hunyuan_flush_range(q, 0, (size_t)L * desc->DQ);
        PREFILL_PROF_TICK("attn:q_sync_to");
        const ert_cmd_state kv_state = kv_run.wait();
        HY_TRACE("kv", 'D', (int)kv_state);
        ovl->check("kv", dq_op, kv_state, [&](){
            return hunyuan_dq_from_full_switch(ovl->kv, Lm, desc->D, DKV, desc->get_attn_k_offset(), 0,
                                               kv, common_bufs->xa_tiles, proj_weights);
        });
        PREFILL_PROF_TICK("attn:kv_wait");
        hunyuan_invalidate_range(kv, 0, (size_t)L * DKV);
        PREFILL_PROF_TICK("attn:kv_sync_from");
#else
        cpu_func::_dequant_mm(q.data(), proj_weights.data() + desc->get_attn_q_offset(), common_bufs->x_attn_in.data(), L, desc->D, desc->DQ);
        cpu_func::_dequant_mm(kv.data(), proj_weights.data() + desc->get_attn_k_offset(), common_bufs->x_attn_in.data(), L, desc->D, DKV);
        cpu_func::_rope_then_qk_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads,
                                          q_norm_weight, desc->DH, desc->inv_freq.data(), desc->rms_norm_eps);
        q.sync_to_device();
#endif
        // K is cached *after* rope and after the key norm, V raw. The rope runs on
        // the cache rows, not in kv: the host never writes kv, so kv needs no flush
        // before the next layer's kv run writes it. See "Prefill syncs" in README.md.
        fill_kv_cache(kv.data(), kv.data() + desc->DK, DKV, cache, L_begin, L, input_idx, k_norm_weight);
        PREFILL_PROF_TICK("attn:fill_kv_cache");

#if HUNYUAN_ATTN_ON_NPU
        // With an unaligned L_begin the kv read rounds up past the end of the
        // prefill; the cache has spare rows for that (kv_cache_rows), so this
        // only falls back to the host if those were ever too few.
        if (seq_gen->mha_engine_fits((uint32_t)L_begin, Lp)) {
            const int mha_owner = ovl->bd_owner;
            npu_app& mha_app = ovl->mha.get(ovl->configured, {(uint32_t)L_begin, Lp, (uint32_t)(mha_owner + 1)},
                                            [&](npu_sequence* s, int from){
                gen_attn_run(s, from, (uint32_t)L_begin, Lp, seq_gen->MAX_L, desc->DH, desc->num_heads, desc->num_kv_heads,
                             mha_owner);
            });
            auto mha_run = mha_app.create_run(common_bufs->after_mix, q, cache);
            mha_run.start();
            HY_TRACE("mha", 'S');
            ovl->ran(OP_ATTN);
            if (while_attn) while_attn();
            const ert_cmd_state mha_state = mha_run.wait();
            HY_TRACE("mha", 'D', (int)mha_state);
            ovl->check("mha", OP_ATTN, mha_state, [&](){
                npu_app& app = ovl->mha.get(-1, {(uint32_t)L_begin, Lp, 0u}, [&](npu_sequence* s, int from){
                    gen_attn_run(s, from, (uint32_t)L_begin, Lp, seq_gen->MAX_L, desc->DH, desc->num_heads,
                                 desc->num_kv_heads, -1);
                });
                return app.run_for_state(common_bufs->after_mix, q, cache);
            });
#if !HUNYUAN_MM_ON_NPU
            common_bufs->after_mix.sync_from_device();   // the host o_proj reads it
#endif
        } else {
            std::cout << " Attention could not run on NPU, use CPU!" << std::endl;
            attention_on_host(cache, L, L_begin);
            if (while_attn) while_attn();
        }
#else
        attention_on_host(cache, L, L_begin);
        if (while_attn) while_attn();
#endif
        PREFILL_PROF_TICK("attn:mha");

#if HUNYUAN_MM_ON_NPU
        // the o run reads its x as bfp16 tiles: rows [0, L) of the attention output
        hunyuan_invalidate_range(common_bufs->after_mix, 0, (size_t)L * desc->DQ);
        hunyuan_write_x_tiles(common_bufs->am_tiles, L, desc->DQ, [&](int row, int n, int col0, int ncols, bf16* out) {
            for (int r = 0; r < n; r++)
                memcpy(out + (size_t)r * ncols, common_bufs->after_mix.data() + (size_t)(row + r) * desc->DQ + col0,
                       ncols * sizeof(bf16));
        });
        PREFILL_PROF_TICK("attn:o_x_tiles");
        const int o_owner = ovl->bd_owner;
        npu_app& o_app = ovl->o.get(ovl->configured, {Lm, (uint32_t)(o_owner + 1)}, [&](npu_sequence* s, int from){
            gen_dq_run(s, from, Lm, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0, o_owner);
        });
        HY_TRACE("o", 'S');
        const ert_cmd_state o_state = o_app.run_for_state(common_bufs->x, common_bufs->am_tiles, proj_weights);
        HY_TRACE("o", 'D', (int)o_state);
        ovl->check("o", dq_op, o_state, [&](){
            return hunyuan_dq_from_full_switch(ovl->o, Lm, desc->DQ, desc->D, desc->get_attn_o_offset(), 0,
                                               common_bufs->x, common_bufs->am_tiles, proj_weights);
        });
        ovl->ran(dq_op);
        PREFILL_PROF_TICK("attn:o_run");
        hunyuan_invalidate_range(common_bufs->x, 0, (size_t)L * desc->D);
        PREFILL_PROF_TICK("attn:o_sync_from");
#else
        cpu_func::_dequant_mm(common_bufs->x.data(), proj_weights.data() + desc->get_attn_o_offset(),
                              common_bufs->after_mix.data(), L, desc->DQ, desc->D);
        common_bufs->x.sync_to_device();
        PREFILL_PROF_TICK("attn:o_proj");
#endif

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }

    /// @brief causal GQA over the split cache, on the host
    /// @note  One call per half of the split. Heads [0, num_heads/2) map onto kv
    ///        heads 0..1 and heads [num_heads/2, num_heads) onto kv heads 2..3;
    ///        num_kv_groups is unchanged because both counts halve. q and out
    ///        keep their full DQ row stride, hence the trailing qo_stride.
    ///
    ///        Also the fallback for NPU calls whose kv read would run off the end
    ///        of the cache, so it is compiled in regardless of HUNYUAN_ATTN_ON_NPU.
    ///        Leaves after_mix valid on both sides.
    void attention_on_host(buffer<bf16>& cache, int L, int L_begin){
        cache.sync_from_device();
        const int half_heads = desc->num_heads / 2;
        const int half_kv    = desc->num_kv_heads / 2;
        const int half_DK    = desc->DK / 2;
        const int half_DV    = desc->DV / 2;
        for (int half = 0; half < 2; half++){
            const size_t k_off = half ? seq_gen->get_k47_offset() : seq_gen->get_k03_offset();
            const size_t v_off = half ? seq_gen->get_v47_offset() : seq_gen->get_v03_offset();
            const size_t q_off = (size_t)half * half_heads * desc->DH;
            cpu_func::_attention(
                common_bufs->after_mix.data() + q_off, q.data() + q_off,
                cache.data() + k_off, cache.data() + v_off,
                L, L_begin, half_heads, half_kv, desc->DH,
                half_DK, half_DV, desc->attn_scale, (int)desc->DQ
            );
        }
        common_bufs->after_mix.sync_to_device();
    }

    /// @brief write L new rows into the array's four-way split cache layout
    /// @note  The decode overlay reads the cache as four quadrants -- K for kv
    ///        heads 0..1, K for kv heads 2..3, then the same split for V -- each
    ///        row-major [MAX_L x DK/2]. That layout is dictated by the overlay
    ///        (the rope tile emits K and V on two separate shims), so the host
    ///        prefill writes it too; see hunyuan_npu_sequence::_receive_kv_cache.
    /// @param stride elements between two rows of k_src (and of v_src)
    /// @param pos, k_norm_weight rope position per row and key norm weight, applied
    ///        to the K rows once they are in the cache
    void fill_kv_cache(const bf16* k_src, const bf16* v_src, int stride, buffer<bf16>& cache, int L_begin, int L,
                       const std::vector<int>& pos, const bf16* k_norm_weight){
        const size_t half_k = desc->DK / 2;
        const size_t half_v = desc->DV / 2;

        const size_t k_03_first = seq_gen->get_k03_offset() + (size_t)L_begin * half_k;
        const size_t k_47_first = seq_gen->get_k47_offset() + (size_t)L_begin * half_k;
        const size_t v_03_first = seq_gen->get_v03_offset() + (size_t)L_begin * half_v;
        const size_t v_47_first = seq_gen->get_v47_offset() + (size_t)L_begin * half_v;

        bf16* k_03_ptr = cache.data() + k_03_first;
        bf16* k_47_ptr = cache.data() + k_47_first;
        bf16* v_03_ptr = cache.data() + v_03_first;
        bf16* v_47_ptr = cache.data() + v_47_first;

        for (int l = 0; l < L; l++){
            const bf16* k_row = k_src + (size_t)l * stride;
            const bf16* v_row = v_src + (size_t)l * stride;
            memcpy(k_03_ptr, k_row,           half_k * sizeof(bf16));
            memcpy(k_47_ptr, k_row + half_k,  half_k * sizeof(bf16));
            memcpy(v_03_ptr, v_row,           half_v * sizeof(bf16));
            memcpy(v_47_ptr, v_row + half_v,  half_v * sizeof(bf16));
            k_03_ptr += half_k;
            k_47_ptr += half_k;
            v_03_ptr += half_v;
            v_47_ptr += half_v;
        }
        const int heads_per_quadrant = desc->num_kv_heads / 2;
        for (size_t k_first : {k_03_first, k_47_first}){
            buffer<bf16> k_rows(cache.data() + k_first, (size_t)L * half_k);
            cpu_func::_rope_then_qk_rms_batch(k_rows, (int)half_k, pos, L, heads_per_quadrant,
                                              k_norm_weight, desc->DH, desc->inv_freq.data(), desc->rms_norm_eps);
        }
        // only the rows just written; see "Long context" in README.md
        hunyuan_flush_range(cache, k_03_first, (size_t)L * half_k);
        hunyuan_flush_range(cache, k_47_first, (size_t)L * half_k);
        hunyuan_flush_range(cache, v_03_first, (size_t)L * half_v);
        hunyuan_flush_range(cache, v_47_first, (size_t)L * half_v);
    }
};

struct hunyuan_mlp_prefill_context{
    hunyuan_desc* desc;
    hunyuan_npu_sequence* seq_gen;
    hunyuan_common_buffers* common_bufs;
    npu_xclbin_manager* npu;
    hunyuan_overlay_state* ovl;

    /// @brief rows the upgate/hid allocation was cut for; they are re-cut on any change
    int L_alloc = -1;

    buffer<bf16> upgate;
    /// @brief the glu output: bf16 for the host matmul build, bfp16 tiles (the down run's x) otherwise
    buffer<bf16> hid;
    buffer<u8> hid_tiles;

    hunyuan_mlp_prefill_context(
        hunyuan_desc* desc,
        hunyuan_npu_sequence* seq_gen,
        hunyuan_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        hunyuan_overlay_state* ovl
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu), ovl(ovl){}

    /// @param Lg rows the matmuls compute (pad_prefill_gemm_L)
    buffer<bf16> forward(int L, int Lg, buffer<u8>& proj_weights) {
        using namespace hunyuan_overlay;
        PREFILL_PROF_INIT();
        const int L_padded = (int)hunyuan_npu_sequence::pad_prefill_L((uint32_t)L);
        const uint32_t Lm = (uint32_t)Lg;
        [[maybe_unused]] const int dq_op = dq_op_for(Lm);
        if (L_alloc != L_padded) {
            upgate = npu->create_bo_buffer<bf16>((size_t)L_padded * desc->intermediate_size * 2);
            // the glu only writes L rows, so the padding rows are whatever was
            // last left there -- zero here, stale from a longer prefill later.
            // down_proj then computes garbage into x rows [L, L_padded), which
            // nothing reads: `valid` below is L * D, and the next layer memsets x.
#if HUNYUAN_MM_ON_NPU
            hid_tiles = hunyuan_zero_tiles(npu, L_padded, desc->intermediate_size);
#else
            hid = npu->create_bo_buffer<bf16>((size_t)L_padded * desc->intermediate_size);
            memset(hid.data(), 0, hid.size() * sizeof(bf16));
            hid.sync_to_device();
#endif
            L_alloc = L_padded;
        }
        PREFILL_PROF_TICK("mlp:seq_gen+alloc");

#if HUNYUAN_MM_ON_NPU
        // up and gate are interleaved in 512-row slabs (even = up, odd = gate):
        // the overlay's left half computes the up slabs, the right half the gate
        // slabs with the silu on chip, so one run leaves [up | silu(gate)] slabs
        // behind and the host only multiplies.
        static_assert((uint32_t)hunyuan_desc::UP_GATE_OUTDIM_REORDEER == hunyuan_npu_sequence::PREFILL_UNIT_N,
                      "an up/gate slab must be exactly one 512-wide overlay tile");
        const int upgate_owner = ovl->bd_owner;
        npu_app& upgate_app = ovl->upgate.get(ovl->configured, {Lm, (uint32_t)(upgate_owner + 1)}, [&](npu_sequence* s, int from){
            gen_dq_run(s, from, Lm, desc->D, (uint32_t)desc->intermediate_size * 2, desc->get_upgate_offset(), 0, 1, upgate_owner);
        });
        HY_TRACE("upgate", 'S');
        const ert_cmd_state upgate_state = upgate_app.run_for_state(upgate, common_bufs->xm_tiles, proj_weights);
        HY_TRACE("upgate", 'D', (int)upgate_state);
        ovl->check("upgate", dq_op, upgate_state, [&](){
            return hunyuan_dq_from_full_switch(ovl->upgate, Lm, desc->D, (uint32_t)desc->intermediate_size * 2,
                                               desc->get_upgate_offset(), 1, upgate, common_bufs->xm_tiles, proj_weights);
        });
        ovl->ran(dq_op);
        PREFILL_PROF_TICK("mlp:upgate_run");
        hunyuan_invalidate_range(upgate, 0, (size_t)L * desc->intermediate_size * 2);
        PREFILL_PROF_TICK("mlp:upgate_sync_from");
        // the glu writes the down run's x straight as bfp16 tiles (and flushes them)
        hunyuan_write_x_tiles(hid_tiles, L, desc->intermediate_size, [&](int row, int n, int col0, int ncols, bf16* out) {
            const int I = desc->intermediate_size;
            const int slab = hunyuan_desc::UP_GATE_OUTDIM_REORDEER;
            for (int r = 0; r < n; r++) {
                // _glu_mul_cols works inside one up|gate slab: cut the piece at slab edges
                for (int c = col0; c < col0 + ncols; ) {
                    const int len = std::min(slab - c % slab, col0 + ncols - c);
                    cpu_func::_glu_mul_cols(out + (size_t)r * ncols + (c - col0), upgate.data() + (size_t)(row + r) * 2 * I,
                                            c, len, slab);
                    c += len;
                }
            }
        });
        PREFILL_PROF_TICK("mlp:glu_mul");
#else
        cpu_func::_dequant_mm_upgate(upgate.data(), proj_weights.data() + desc->get_upgate_offset(),
                                     common_bufs->x.data(), L, desc->D, desc->intermediate_size,
                                     hunyuan_desc::UP_GATE_OUTDIM_REORDEER);
        cpu_func::_exp_glu(hid.data(), upgate.data(), L, desc->intermediate_size, hunyuan_desc::UP_GATE_OUTDIM_REORDEER);
        PREFILL_PROF_TICK("mlp:upgate+glu");
#endif

#if HUNYUAN_MM_ON_NPU
        const int down_owner = ovl->bd_owner;
        npu_app& down_app = ovl->down.get(ovl->configured, {Lm, (uint32_t)(down_owner + 1)}, [&](npu_sequence* s, int from){
            gen_dq_run(s, from, Lm, (uint32_t)desc->intermediate_size, desc->D, desc->get_down_offset(), 0, 0, down_owner);
        });
        HY_TRACE("down", 'S');
        const ert_cmd_state down_state = down_app.run_for_state(common_bufs->x, hid_tiles, proj_weights);
        HY_TRACE("down", 'D', (int)down_state);
        ovl->check("down", dq_op, down_state, [&](){
            return hunyuan_dq_from_full_switch(ovl->down, Lm, (uint32_t)desc->intermediate_size, desc->D,
                                               desc->get_down_offset(), 0, common_bufs->x, hid_tiles, proj_weights);
        });
        ovl->ran(dq_op);
        PREFILL_PROF_TICK("mlp:down_run");
        hunyuan_invalidate_range(common_bufs->x, 0, (size_t)L * desc->D);
        PREFILL_PROF_TICK("mlp:down_sync_from");
#else
        cpu_func::_dequant_mm(common_bufs->x.data(), proj_weights.data() + desc->get_down_offset(),
                              hid.data(), L, desc->intermediate_size, desc->D);
        common_bufs->x.sync_to_device();
        PREFILL_PROF_TICK("mlp:down_proj");
#endif

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }
};

struct hunyuan_prefill_context{
    hunyuan_desc* desc;
    hunyuan_npu_sequence* seq_gen;
    hunyuan_common_buffers common_bufs;
    npu_xclbin_manager* npu;
    hunyuan_overlay_state ovl;

    std::unique_ptr<hunyuan_attn_block_prefill_context> attn_block;
    std::unique_ptr<hunyuan_mlp_prefill_context> mlp;

    hunyuan_prefill_context(){}
    hunyuan_prefill_context(
        npu_xclbin_manager* npu,
        hunyuan_desc* model_desc,
        hunyuan_npu_sequence* seq_gen,
        LM_Config& config
    ) : desc(model_desc), seq_gen(seq_gen), npu(npu) {
        // One xclbin, prefill_overlay.xclbin: dequant_mm and attention take turns
        // on the whole array (FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay).
        //
        // Only register when something is actually going to be run: registering
        // opens and validates the file, so doing it unconditionally would make a
        // CPU-only run fail on a machine that has no hunyuan xclbins yet.
#if HUNYUAN_MM_ON_NPU || HUNYUAN_ATTN_ON_NPU
        npu_app_manager* mgr = npu->register_xclbin(
            utils::path_join(config.exec_path, "xclbins", config.model_name, "prefill_overlay.xclbin"));
        for (overlay_role* r : {&ovl.q, &ovl.kv, &ovl.mha, &ovl.o, &ovl.upgate, &ovl.down}) r->init(mgr);
#endif
        (void)config;

        this->attn_block = std::make_unique<hunyuan_attn_block_prefill_context>(
            desc, seq_gen, &common_bufs, npu, &ovl
        );
        this->mlp = std::make_unique<hunyuan_mlp_prefill_context>(
            desc, seq_gen, &common_bufs, npu, &ovl
        );
    }

    /// @brief another xclbin (decode layer, lm_head) ran: the overlay is reloaded
    ///        before its next run, so the next switch has to write everything
    void overlay_lost(){
        ovl.configured = -1;
    }

    /// @brief one decoder layer over L tokens
    /// @param residual  L_padded x D running hidden state, updated in place
    /// @param small_rows let the matmuls use the 32 / 64-row plans (short prompts)
    buffer<bf16> forward(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& all_proj_weights,
                         buffer<bf16>& rms_weights, buffer<bf16>& rope_weights, buffer<bf16>& kv_cache,
                         std::vector<int>& input_idx, int L_begin = 0,
                         const std::function<void()>& while_attn = {}, bool small_rows = false){
        PREFILL_PROF_INIT();
        (void)layer_idx;
        HY_TRACE_WHERE(layer_idx, L_begin, L);
        int L_padded = (int)hunyuan_npu_sequence::pad_prefill_L((uint32_t)L);
        const int Lg = (int)hunyuan_npu_sequence::pad_prefill_gemm_L((uint32_t)L, small_rows);
        size_t x_size = (size_t)L_padded * desc->D;

        if (x_size != common_bufs.x.size()) {
            common_bufs.x = npu->create_bo_buffer<bf16>((size_t)L_padded * desc->D);
            common_bufs.after_mix = npu->create_bo_buffer<bf16>((size_t)L_padded * desc->DQ);
            // the padding rows must stay zero; see "Prefill syncs" in README.md
#if HUNYUAN_MM_ON_NPU
            common_bufs.xa_tiles = hunyuan_zero_tiles(npu, L_padded, desc->D);
            common_bufs.am_tiles = hunyuan_zero_tiles(npu, L_padded, desc->DQ);
            common_bufs.xm_tiles = hunyuan_zero_tiles(npu, L_padded, desc->D);
#else
            common_bufs.x_attn_in = npu->create_bo_buffer<bf16>((size_t)L_padded * desc->D);
            memset(common_bufs.x_attn_in.data(), 0, common_bufs.x_attn_in.size() * sizeof(bf16));
            common_bufs.x_attn_in.sync_to_device();
#endif
            common_bufs.x_attn_in_rows = 0;
        }
#if HUNYUAN_MM_ON_NPU
        // xa's rows are written in 16-row groups (the last one zero past L): zero the
        // later groups an earlier, longer call wrote
        const int xa_rows = (L + 15) / 16 * 16;
        if (xa_rows < common_bufs.x_attn_in_rows) {
            const size_t first = hunyuan_bfp16::x_tiles_bytes(xa_rows, desc->D);
            const size_t count = hunyuan_bfp16::x_tiles_bytes(common_bufs.x_attn_in_rows - xa_rows, desc->D);
            memset(common_bufs.xa_tiles.data() + first, 0, count);
            hunyuan_bfp16::flush_bytes(common_bufs.xa_tiles, first, count);
        }
        common_bufs.x_attn_in_rows = xa_rows;
        PREFILL_PROF_TICK("layer:x_memset");

        // every layer starts by normalizing the residual into the q/kv input
        hunyuan_write_norm_tiles(common_bufs.xa_tiles, residual.data(), rms_weights.data(), L, desc->D, desc->rms_norm_eps);
        PREFILL_PROF_TICK("layer:input_rms_norm");
#else
        if (L < common_bufs.x_attn_in_rows) {
            const size_t first = (size_t)L * desc->D;
            const size_t count = (size_t)(common_bufs.x_attn_in_rows - L) * desc->D;
            memset(common_bufs.x_attn_in.data() + first, 0, count * sizeof(bf16));
            hunyuan_flush_range(common_bufs.x_attn_in, first, count);
        }
        common_bufs.x_attn_in_rows = L;
        PREFILL_PROF_TICK("layer:x_memset");

        // every layer starts by stashing the residual and normalizing into x_attn_in
        cpu_func::_rms_norm(common_bufs.x_attn_in, residual, rms_weights.data(), desc->D, L, 0, 0, desc->rms_norm_eps);
        PREFILL_PROF_TICK("layer:input_rms_norm");
        hunyuan_flush_range(common_bufs.x_attn_in, 0, (size_t)L * desc->D);
        PREFILL_PROF_TICK("layer:x_sync_to");
#endif

        buffer<bf16> block_output = this->attn_block->forward(L, Lg, all_proj_weights, rope_weights, kv_cache, input_idx, L_begin, while_attn);
        PREFILL_PROF_TICK("layer:attn_block_total");

        cpu_func::_residual_add(residual, block_output, residual, (size_t)L * desc->D);
        PREFILL_PROF_TICK("layer:residual_add");

#if HUNYUAN_MM_ON_NPU
        hunyuan_write_norm_tiles(common_bufs.xm_tiles, residual.data(), rms_weights.data() + desc->D, L, desc->D,
                                 desc->rms_norm_eps);
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");
#else
        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data() + desc->D, desc->D, L, 0, 0, desc->rms_norm_eps);
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");
        hunyuan_flush_range(common_bufs.x, 0, (size_t)L * desc->D);
        PREFILL_PROF_TICK("layer:x_sync_to");
#endif

        auto mlp_out = mlp->forward(L, Lg, all_proj_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        cpu_func::_residual_add(residual, mlp_out, residual, (size_t)L * desc->D);
        PREFILL_PROF_TICK("layer:residual_add");

        return residual;
    }
};

#endif // __HUNYUAN_PREFILL_HPP__
