// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __MINICPM_OCR_PREFILL_HPP__
#define __MINICPM_OCR_PREFILL_HPP__
#include "minicpm_ocr_npu_sequence.hpp"
#include "npu_utils/runtime_wait.hpp"
#include "npu_sequences/GateDeltaNetPrefillRuntimeSequence.hpp"
#include "modules/gemm.hpp"
#include "minicpm_ocr_model_def.hpp"
#include "utils/avx512_util.hpp"
#include "cpu_functions.hpp"
#include "minicpm_ocr_prefill_cpu.hpp"
#include "minicpm_ocr_prefill_xref.hpp"
#include "attn_bfp16.hpp"
#include "mm_bfp16.hpp"
#include "prefill_chunks.hpp"
#include <functional>
#include <future>

// ---------------------------------------------------------------------------
// Prefill profiling.
// Enable by compiling with -DMINICPM_OCR_PREFILL_PROFILE (all code below
// compiles to nothing otherwise, so it is zero-cost when disabled).
//
//   PREFILL_PROF_INIT();               // once at the start of a function
//   ... work ...
//   PREFILL_PROF_TICK("dequant");      // prints time since the previous tick
//   ... work ...
//   PREFILL_PROF_TICK("qkv_proj");
// ---------------------------------------------------------------------------
// For accumulating a stage that runs once per loop iteration and reporting the
// summed total once after the loop (avoids one print per iteration):
//
//   PREFILL_PROF_ACC_DECL(upgate);        // before the loop
//   for (...) {
//       PREFILL_PROF_LAP();               // reset cursor at top of iteration
//       ... work ...
//       PREFILL_PROF_ACC(upgate);         // add elapsed to the accumulator
//   }
//   PREFILL_PROF_ACC_PRINT("mlp_loop:upgate", upgate);
// #define MINICPM_OCR_PREFILL_PROFILE
#ifdef MINICPM_OCR_PREFILL_PROFILE
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
#define PREFILL_PROF_LAP() (__prof_t = time_utils::now())
#define PREFILL_PROF_ACC_DECL(var) float __prof_acc_##var = 0.0f
#define PREFILL_PROF_ACC(var)                                                  \
    do {                                                                       \
        time_utils::time_point __prof_now = time_utils::now();                 \
        __prof_acc_##var += time_utils::duration_us(__prof_t, __prof_now).first;\
        __prof_t = __prof_now;                                                 \
    } while (0)
#define PREFILL_PROF_ACC_PRINT(name, var)                                     \
    do {                                                                       \
        time_utils::time_with_unit __prof_d =                                  \
            time_utils::re_unit(time_utils::time_with_unit(__prof_acc_##var, "us"));\
        header_print("prefill-prof", name << " (sum): " << __prof_d.first      \
                                          << " " << __prof_d.second);          \
    } while (0)
#else
#define PREFILL_PROF_INIT() do {} while (0)
#define PREFILL_PROF_TICK(name) do {} while (0)
#define PREFILL_PROF_LAP() do {} while (0)
#define PREFILL_PROF_ACC_DECL(var)
#define PREFILL_PROF_ACC(var) do {} while (0)
#define PREFILL_PROF_ACC_PRINT(name, var) do {} while (0)
#endif

struct minicpm_ocr_common_buffers{
    buffer<bf16> x;
    buffer<bf16> gate;
    buffer<bf16> after_mix;

    //!< mm_bfp16.xclbin (mm_bfp16.hpp): the mm reads bfp16 tiles; xb: the normed x, ab: the o projections' input
    bool mm_bfp16 = false;
    //!< mm_bfp16_glu.xclbin: the up/gate run writes silu(gate) * up itself, into GLU slots the down run reads
    bool glu_npu = false;
    minicpm_ocr_mm_bfp16::tiles xb;
    minicpm_ocr_mm_bfp16::tiles ab;
    //!< this call's row chunks (prefill_chunks.hpp); one chunk unless the mm reads bfp16 tiles
    minicpm_ocr_chunks::layout chunks;

    minicpm_ocr_common_buffers(){}

    /// @brief the buffer an mm run reads its x from
    bytes& mm_in(buffer<bf16>& src, minicpm_ocr_mm_bfp16::tiles& t) {
        return mm_bfp16 ? static_cast<bytes&>(t.buf) : static_cast<bytes&>(src);
    }
};

/// @brief A projection as one run per row chunk (prefill_chunks.hpp): prep(c) makes chunk c's input rows, post(c)
///        handles its output rows. prep(c + 1) and post(c - 1) run on the host while the array runs chunk c; post
///        of the last chunk is left to the caller, to run under the next projection.
template <typename Prep, typename Post>
void run_o_chunks(npu_app* apps, int count, buffer<bf16>& out, bytes& in, buffer<u8>& weights, Prep prep, Post post) {
    prep(0);
    for (int c = 0; c < count; c++) {
        auto run = apps[c].create_run(out, in, weights);
        run.start();
        if (c + 1 < count) {
            prep(c + 1);
        }
        if (c > 0) {
            post(c - 1);
        }
        flm::npu::wait_for_completion(run, std::string("minicpm-ocr o chunk ") + std::to_string(c));
    }
}

// ---------------------------------------------------------------------------
// Each block below splits into stages, one per array overlay. A stage is a pair:
// _<stage>_npu() runs it on the array and returns true, or returns false when its
// MINICPM_OCR_*_ON_NPU macro is 0 (or it declines at run time); the caller then runs
// _<stage>_host(), the minicpm_ocr_cpu kernel over the same buffers. Both leave the
// stage's output valid on host *and* device (minicpm_ocr_cpu::scratch_push/pull), so
// any mix of backends is coherent.
// ---------------------------------------------------------------------------

struct minicpm_ocr_linear_block_prefill_context{
    int L_padded_old = -1;
    //!< the L the GateDeltaNet array sequence was last generated for; only calls that run
    //!< on the array regenerate it (see _gdn_on_npu), so a short call in between costs nothing
    int gdn_seq_L = -1;
    //!< the g run wrote this call's gate (z) rows and the host has not pulled them yet (_pull_gate)
    bool gate_pending = false;
    //!< the L_padded the conv1d array sequence was last generated for. Separate from
    //!< L_padded_old on purpose: two calls can share an L_padded and fall on opposite sides of
    //!< the conv threshold, and the array call must not find a sequence the host call skipped
    int conv_seq_Lp = -1;
#ifdef ENABLE_DEBUG
    //!< set by minicpm_ocr_prefill_context::forward() before every call, so the xref lines
    //!< can name the layer without threading it through every stage signature
    int layer_idx = -1;
    //!< non-null in a debug build; whether it scores anything is up to enabled(layer_idx)
    minicpm_ocr_prefill_xref* xref = nullptr;
#endif
    minicpm_ocr_desc* desc;
    minicpm_ocr_npu_sequence* seq_gen;
    minicpm_ocr_common_buffers* common_bufs;
    npu_xclbin_manager* npu;

#if MINICPM_OCR_MM_ON_NPU
    //!< one run per row chunk (common_bufs->chunks); [0] covers all rows when there is one chunk
    npu_app linear_qkv_proj[minicpm_ocr_chunks::max_chunks];
    npu_app linear_g_proj;
    npu_app linear_o_proj[minicpm_ocr_chunks::max_chunks];
#endif
#if MINICPM_OCR_CONV_ON_NPU
    npu_app conv1d;
#endif
#if MINICPM_OCR_GDN_ON_NPU
    npu_app deltanet;
#endif

    buffer<bf16> qkv;
    buffer<bf16> conv_qkv;
    buffer<float> alpha_beta;

    minicpm_ocr_linear_block_prefill_context(
        minicpm_ocr_desc* desc,
        minicpm_ocr_npu_sequence* seq_gen,
        minicpm_ocr_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* conv1d_app_manager,
        npu_app_manager* gate_delta_net_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu)
    {
#if MINICPM_OCR_MM_ON_NPU
        for (npu_app& app : this->linear_qkv_proj) {
            app = gemm_app_manager->create_app();
        }
        this->linear_g_proj = gemm_app_manager->create_app();
        for (npu_app& app : this->linear_o_proj) {
            app = gemm_app_manager->create_app();
        }
#endif
#if MINICPM_OCR_CONV_ON_NPU
        this->conv1d = conv1d_app_manager->create_app();
#endif
#if MINICPM_OCR_GDN_ON_NPU
        this->deltanet = gate_delta_net_manager->create_app();
#endif
    }

    minicpm_ocr_linear_layer& weights() {
        return static_cast<minicpm_ocr_linear_layer&>(*desc->weight_desc[desc->lin_idx]);
    }

    /// @brief Smallest prefill call the gated delta rule runs on the array for.
    ///
    /// The array is cheaper per token than the host but pays ~125 ms of fixed cost per
    /// prefill call (switches), so it only wins on long calls. 5120 is the crossover measured
    /// after the host GDN got faster (README.md, "GDN and attention placement re-tuned").
    ///
    /// Decided per prefill CALL, because that is where the fixed cost is paid. MiniCPM_OCR
    /// does not split a prompt (max_prefill_len is not read on this path; a 24k prompt is
    /// one call), so in practice the split is between TURNS: a long first prompt runs on the
    /// array, and a cached follow-up -- which prefills only its new tokens -- is usually
    /// short and runs on the host, starting from the state the array left.
    ///
    /// Override without a rebuild with FLM_MINICPM_OCR_GDN_NPU_MIN_TOKENS: 0 always uses the
    /// array, a value above max_prefill_len always uses the host. Read once.
    static int gdn_npu_min_tokens() {
        static const int n = [] {
            int v = 5120;
            if (const char* env = std::getenv("FLM_MINICPM_OCR_GDN_NPU_MIN_TOKENS")) {
                char* end = nullptr;
                const long x = std::strtol(env, &end, 10);
                if (end != env && x >= 0) v = (int)x;
            }
            return v;
        }();
        return n;
    }

    /// @brief Whether this prefill call's gated delta rule goes to the array. A pure
    ///        function of L, which is what keeps the cached array sequence (gdn_seq_L)
    ///        consistent.
    static bool _gdn_on_npu(int L) {
#if MINICPM_OCR_GDN_ON_NPU
        return L >= gdn_npu_min_tokens();
#else
        (void)L;
        return false;
#endif
    }

    /// @brief Smallest prefill call the conv1d runs on the array for; shorter ones run on the
    ///        host. Same shape of decision as gdn_npu_min_tokens(), with its own threshold,
    ///        because conv's costs are different: conv8.xclbin is ~2x faster than the host per
    ///        call, but each array call adds one more context switch to the layer (mm -> conv
    ///        -> GDN instead of mm -> GDN, ~2.8 ms) on top of its launch. See README.md for
    ///        the measured curve the default comes from.
    ///        Override with FLM_MINICPM_OCR_CONV_NPU_MIN_TOKENS (0 = always the array, a huge
    ///        value = always the host). Read once.
    static int conv_npu_min_tokens() {
        static const int n = [] {
            int v = 12288;
            if (const char* env = std::getenv("FLM_MINICPM_OCR_CONV_NPU_MIN_TOKENS")) {
                char* end = nullptr;
                const long x = std::strtol(env, &end, 10);
                if (end != env && x >= 0) v = (int)x;
            }
            return v;
        }();
        return n;
    }

    /// @brief Whether this prefill call's conv1d goes to the array. A pure function of L.
    static bool _conv_on_npu(int L) {
#if MINICPM_OCR_CONV_ON_NPU
        return L >= conv_npu_min_tokens();
#else
        (void)L;
        return false;
#endif
    }

    /// @brief Whether the HOST copy of conv_qkv will be read this call.
    ///
    /// conv_qkv has exactly one consumer, the gated delta rule, so only the side that will run
    /// it needs the data: _gdn_host() reads the host copy, the array GDN reads the device copy.
    /// Copying the other way is pure cost -- ~98 MB a layer at 8k tokens -- so the conv
    /// paths make conv_qkv valid only where it is consumed. This is the one deliberate
    /// exception to "every stage leaves its output valid on host and device" above.
    /// A debug build always keeps the host copy, for the cross reference.
    static bool _conv_qkv_needed_on_host(int L) {
#ifdef ENABLE_DEBUG
        (void)L;
        return true;
#else
        return !_gdn_on_npu(L);
#endif
    }

    /// @brief grow the scratch and regenerate the array sequences for this L
    void reserve(int L, int L_padded) {
        PREFILL_PROF_INIT();
        if (L_padded_old != L_padded){
#if MINICPM_OCR_MM_ON_NPU
            if (common_bufs->mm_bfp16) {
                const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
                for (int c = 0; c < chunks.count; c++) {
                    const size_t out_row = (size_t)(desc->conv_kernel_size - 1) + chunks.begin[c];
                    seq_gen->gen_dequant_mm_bfp16(this->linear_qkv_proj[c].seq(), chunks.padded[c], desc->D, desc->Linear_DQKV, desc->get_linear_qkv_offset(), out_row * desc->Linear_DQKV, 0, false, chunks.begin[c]);
                }
                seq_gen->gen_dequant_mm_bfp16(this->linear_g_proj.seq(), L_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1);
                for (int c = 0; c < chunks.count; c++) {
                    seq_gen->gen_dequant_mm_bfp16(this->linear_o_proj[c].seq(), chunks.padded[c], desc->Linear_DV, desc->D, desc->get_linear_o_offset(), (size_t)chunks.begin[c] * desc->D, 0, false, chunks.begin[c]);
                }
            }
            else {
                seq_gen->gen_dequant_mm_512(this->linear_qkv_proj[0].seq(), L_padded, desc->D, (desc->Linear_DQKV), desc->get_linear_qkv_offset(), (size_t)(desc->conv_kernel_size - 1) * (desc->Linear_DQKV), 0);
                seq_gen->gen_dequant_mm_512(this->linear_g_proj.seq(), L_padded, desc->D, desc->Linear_DV, desc->get_linear_g_offset(), 0, 1);
                seq_gen->gen_dequant_mm_512(this->linear_o_proj[0].seq(), L_padded, desc->Linear_DV, desc->D, desc->get_linear_o_offset(), 0, 0);
            }
#endif
            PREFILL_PROF_TICK("linear:.mm_seq_gen");
            qkv  = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)(L_padded + desc->conv_kernel_size - 1) * desc->Linear_DQKV);
            conv_qkv  = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->Linear_DQKV);
            alpha_beta = minicpm_ocr_cpu::scratch_alloc<float>(npu, (size_t)L_padded * desc->linear_num_v_heads * 2);
            PREFILL_PROF_TICK("linear:.alloc");
            memset(alpha_beta.data(), 0, alpha_beta.size() * sizeof(float));
            minicpm_ocr_cpu::scratch_push(alpha_beta);
            PREFILL_PROF_TICK("linear:.zero_alpha_beta");
        }
#if MINICPM_OCR_CONV_ON_NPU
        // Only for calls that will run on the array (see conv_seq_Lp for why it is tracked
        // separately from L_padded_old). The buffers are run arguments, so a sequence built
        // for this L_padded stays valid across scratch reallocations.
        if (_conv_on_npu(L) && conv_seq_Lp != L_padded) {
            conv_seq_Lp = L_padded;
            seq_gen->gen_seq_conv1d(
                this->conv1d.seq(),
                L_padded + (desc->conv_kernel_size - 1),  desc->Linear_DQKV , L_padded, 0
            );
        }
#endif
#if MINICPM_OCR_GDN_ON_NPU
        // Only for calls that will run on the array; a short call declines in _gdn_npu
        // and must not pay for (or overwrite) the sequence a long one is using.
        if (_gdn_on_npu(L) && gdn_seq_L != L) {
            gdn_seq_L = L;
            generate_gate_delta_net_prefill_sequence(
                *deltanet.seq(),
                8, 4,
                desc->linear_num_v_heads/desc->linear_num_qk_heads, desc->linear_num_v_heads,
                128, //TODO: configute it
                L, (L + 127) / 128 * 128,
                desc->Linear_DQ/desc->linear_num_qk_heads, desc->Linear_DK/desc->linear_num_qk_heads, desc->Linear_DV/desc->linear_num_v_heads,
                0,
                // because kvcache is used to store (conv_kernel_size-1) )* (Linear_DQ + Linear_DK + Linear_DV)
                ((desc->conv_kernel_size - 1) * (desc->Linear_DQKV) ) /2,
                (L_padded * desc->linear_num_v_heads)
            );
            PREFILL_PROF_TICK("linear:.gdn_seq_gen");
        }
#endif
        L_padded_old = L_padded;
    }

    /// @brief in_proj_qkv into qkv rows [KS-1, KS-1+L), in_proj_z into common gate with
    ///        the silu applied (the array fuses it, rtp_act = 1), and -- while those
    ///        array runs are going -- the host work that does not wait on their output.
    ///        With two row chunks (common_bufs->chunks) the qkv projection is two runs:
    ///
    ///            array:  [ qkv rows c0 ][ qkv rows c1 ......... ][ g run ............... ]
    ///            host:   [ alpha/beta ]  [ carry in + conv c0 ]  [ conv c1 + carry out ]
    ///
    ///        A conv row reads only its own qkv row and the KS - 1 rows before it, so chunk 0's
    ///        conv needs only chunk 0's rows. With one chunk: alpha/beta under the qkv run, the
    ///        whole carry + conv under the g run. Same work in the same order per row either
    ///        way, so the output is bit-identical; the host and the array just stop waiting
    ///        for each other.
    /// @note  A conv on the array cannot hide under the g run (one bitstream at a time), so
    ///        then it runs after it.
    bool _in_proj_overlapped_npu(int L, buffer<u8>& proj_weights, buffer<bf16>& ssm_weights, buffer<bf16>& cache,
                                 buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
#if MINICPM_OCR_MM_ON_NPU
        PREFILL_PROF_INIT();
        bytes& x_in = common_bufs->mm_in(common_bufs->x, common_bufs->xb);
        const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
        const bool conv_on_array = _conv_on_npu(L);
        const bool conv_by_chunk = chunks.count > 1 && !conv_on_array;
        const bf16* conv_w = ssm_weights.data() + weights().ssm_conv1d.offset / sizeof(bf16);

        auto qkv_run = linear_qkv_proj[0].create_run(qkv, x_in, proj_weights);
        qkv_run.start();
        _alpha_beta(L, ssm_weights, alpha_proj_cpu, beta_proj_cpu);
        PREFILL_PROF_TICK("linear:.alpha_beta (under qkv)");
        flm::npu::wait_for_completion(qkv_run, "qkv_run");
        PREFILL_PROF_TICK("linear:.qkv_wait");

        int conv_rows_done = 0;
        for (int c = 1; c < chunks.count; c++) {
            auto chunk_run = linear_qkv_proj[c].create_run(qkv, x_in, proj_weights);
            chunk_run.start();
            if (conv_by_chunk) {
                _conv_host_chunk(conv_rows_done, chunks.begin[c], cache, conv_w);
                conv_rows_done = chunks.begin[c];
                PREFILL_PROF_TICK("linear:.conv_chunk (under qkv)");
            }
            flm::npu::wait_for_completion(chunk_run, "chunk_run");
            PREFILL_PROF_TICK("linear:.qkv_wait");
        }

        auto g_run = linear_g_proj.create_run(common_bufs->gate, x_in, proj_weights);
        g_run.start();
        if (conv_by_chunk) {
            // the carry out reads qkv rows [L, L + KS - 1): only after _conv_host_chunk pulled them
            _conv_host_chunk(conv_rows_done, L, cache, conv_w);
            _carry_conv_state_out(L, cache);
            _conv_host_finish(L, conv_w);
            PREFILL_PROF_TICK("linear:.conv_chunk (under g)");
        }
        else {
            // only the rows the run wrote and the carry / conv read: [KS-1, KS-1+L)
            minicpm_ocr_cpu::sync_range_from_device(qkv, (size_t)(desc->conv_kernel_size - 1) * desc->Linear_DQKV, (size_t)L * desc->Linear_DQKV);
            PREFILL_PROF_TICK("linear:.qkv_sync_from (under g)");
            _carry_conv_state(L, cache);
            PREFILL_PROF_TICK("linear:.conv_carry (under g)");
            if (!conv_on_array) {
                _conv_host(L, ssm_weights);
            }
            PREFILL_PROF_TICK("linear:.conv_host (under g)");
        }
        flm::npu::wait_for_completion(g_run, "g_run");
        PREFILL_PROF_TICK("linear:.g_wait");
        // with the GDN on the array the gate rows are pulled under its run (_gdn_npu); nothing reads them before
        gate_pending = true;
        if (!_gdn_on_npu(L)) {
            _pull_gate(L);
        }
        PREFILL_PROF_TICK("linear:.gate_sync_from");

        if (conv_on_array) {
            _conv_npu(L, ssm_weights);
        }
        return true;
#else
        (void)L; (void)proj_weights; (void)ssm_weights; (void)cache;
        (void)alpha_proj_cpu; (void)beta_proj_cpu;
        return false;
#endif
    }

    /// @brief invalidate the gate (z) rows the g run wrote, once per call, before the first host read
    void _pull_gate(int L) {
        if (gate_pending) {
            minicpm_ocr_cpu::sync_range_from_device(common_bufs->gate, 0, (size_t)L * desc->Linear_DV);
            gate_pending = false;
        }
    }

    /// @brief carry the conv state across calls: the last KS-1 rows of the previous call go
    ///        in front of this call's qkv (_in), and this call's last KS-1 rows go into the cache (_out)
    void _carry_conv_state(int L, buffer<bf16>& cache) {
        _carry_conv_state_in(cache);
        _carry_conv_state_out(L, cache);
        // qkv's one reader is the conv, so push it (the carried-in state rows changed) only
        // when the conv runs on the array: ~25 MB a layer at 2k tokens for nothing otherwise
        if (_conv_on_npu(L)) {
            minicpm_ocr_cpu::scratch_push(qkv);
        }
    }

    void _carry_conv_state_in(buffer<bf16>& cache) {
        const int DQKV = desc->Linear_DQKV;
        const int pad = desc->conv_kernel_size - 1;
        cache.sync_from_device();
        memcpy(qkv.data(), cache.data(), (size_t)pad * DQKV * sizeof(bf16));
    }

    void _carry_conv_state_out(int L, buffer<bf16>& cache) {
        const int DQKV = desc->Linear_DQKV;
        const int pad = desc->conv_kernel_size - 1;
        memcpy(cache.data(), qkv.data() + L * DQKV, (size_t)pad * DQKV * sizeof(bf16));
        cache.sync_to_device();
    }

    /// @brief the conv on whichever side this call's length picks
    void _conv(int L, buffer<bf16>& ssm_weights) {
        if (!_conv_npu(L, ssm_weights)) {
            _conv_host(L, ssm_weights);
        }
    }

    /// @brief alpha (the decay) and beta of the gated delta rule, from x, on the host
    void _alpha_beta(int L, buffer<bf16>& ssm_weights, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu) {
        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        cpu_func::_cpu_mm<bf16>(alpha_ptr, alpha_proj_cpu.data(), common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        cpu_func::_cpu_mm<bf16>(beta_ptr,  beta_proj_cpu.data(),  common_bufs->x.data(), L, desc->D, desc->linear_num_v_heads);
        MINICPM_OCR_DUMP("linear_alpha", alpha_ptr, L, desc->linear_num_v_heads);
        MINICPM_OCR_DUMP("linear_beta", beta_ptr, L, desc->linear_num_v_heads);
        // Activate over the full 128-chunk padding the deltanet consumes, not L.
        // The kernel reads ceil(L/128)*128 rows of alpha/beta, so the tail rows
        // sharing the last chunk with real tokens must be activated. Leaving them
        // raw (alpha == 0, an invalid decay) produces NaN in the chunked
        // delta-rule scan at the end of the sequence.
        int L_ssm_padded = (L + 127) / 128 * 128;
        cpu_func::_activate_a(
            alpha_ptr,
            (float*)(ssm_weights.data() + desc->get_linear_A_offset() / sizeof(bf16)),
            (float*)(ssm_weights.data() + desc->get_linear_dt_offset() / sizeof(bf16)),
            L_ssm_padded,
            desc->linear_num_v_heads
        );
        cpu_func::_activate_b(beta_ptr, L_ssm_padded, desc->linear_num_v_heads);
#if DEBUG_LEVEL >= 1
        if (minicpm_ocr_cpu::dumper().on()) {
            // the reference dumps the log-space decay g; the kernels consume exp(g)
            std::vector<float> g((size_t)L * desc->linear_num_v_heads);
            for (size_t i = 0; i < g.size(); i++) g[i] = logf(alpha_ptr[i]);
            MINICPM_OCR_DUMP("linear_g", g.data(), L, desc->linear_num_v_heads);
        }
#endif
        minicpm_ocr_cpu::scratch_push(alpha_beta);
    }

    void _in_proj_host(int L, buffer<u8>& proj_weights) {
        const int DQKV = desc->Linear_DQKV;
        const int pad = desc->conv_kernel_size - 1;
        minicpm_ocr_cpu::dequant_mm(qkv.data() + (size_t)pad * DQKV, DQKV,
                                  proj_weights.data() + desc->get_linear_qkv_offset(),
                                  common_bufs->x.data(), desc->D, L, desc->D, DQKV, minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_cpu::dequant_mm(common_bufs->gate.data(), desc->Linear_DV,
                                  proj_weights.data() + desc->get_linear_g_offset(),
                                  common_bufs->x.data(), desc->D, L, desc->D, desc->Linear_DV, minicpm_ocr_cpu::BODY_LAYOUT);
        MINICPM_OCR_DUMP("linear_z", common_bufs->gate.data(), L, desc->Linear_DV);
        minicpm_ocr_cpu::silu_inplace(common_bufs->gate.data(), (size_t)L * desc->Linear_DV);
        minicpm_ocr_cpu::scratch_push(qkv);
        minicpm_ocr_cpu::scratch_push(common_bufs->gate);
    }

    /// @brief conv_qkv = causal conv1d + silu over [state | qkv], then L2 norm over the q
    ///        and k heads
    /// @note  Declines calls shorter than conv_npu_min_tokens(). Leaves conv_qkv on the
    ///        device, and copies it back only when the host GDN will read it.
    bool _conv_npu(int L, buffer<bf16>& ssm_weights) {
#if MINICPM_OCR_CONV_ON_NPU
        if (!_conv_on_npu(L)) {
            return false;
        }
        conv1d(conv_qkv, qkv, ssm_weights);

        if (_conv_qkv_needed_on_host(L)) {
            conv_qkv.sync_from_device();
        }
        return true;
#else
        (void)L;
        (void)ssm_weights;
        return false;
#endif
    }

    void _conv_host(int L, buffer<bf16>& ssm_weights) {
        const bf16* w = ssm_weights.data() + weights().ssm_conv1d.offset / sizeof(bf16);
        _conv_host_rows(0, L, w);
        _conv_host_finish(L, w);
    }

    /// @brief conv output rows [row, row + n): reads qkv rows [row, row + n + KS - 1)
    void _conv_host_rows(int row, int n, const bf16* w) {
        const int DQKV = desc->Linear_DQKV;
        minicpm_ocr_cpu::conv1d_silu_l2norm(conv_qkv.data() + (size_t)row * DQKV, qkv.data() + (size_t)row * DQKV, w, n,
                                          DQKV, desc->conv_kernel_size,
                                          (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                                          desc->linear_qk_head_dim);
    }

    /// @brief conv output rows [from, to) once the qkv run has written them: pull those qkv rows (and carry the
    ///        state in first, for the chunk that starts at row 0)
    void _conv_host_chunk(int from, int to, buffer<bf16>& cache, const bf16* w) {
        const int DQKV = desc->Linear_DQKV;
        const size_t pad = desc->conv_kernel_size - 1;
        minicpm_ocr_cpu::sync_range_from_device(qkv, (pad + from) * DQKV, (size_t)(to - from) * DQKV);
        if (from == 0) {
            _carry_conv_state_in(cache);
        }
        _conv_host_rows(from, to - from, w);
    }

    /// @brief after the last conv rows: the reference dump, and the push the array GDN needs
    void _conv_host_finish(int L, const bf16* w) {
#if DEBUG_LEVEL >= 1
        if (minicpm_ocr_cpu::dumper().on()) {
            // the reference dumps the conv output before the q/k L2 norm, which the fused pass never stores
            const int DQKV = desc->Linear_DQKV;
            std::vector<bf16> before_norm((size_t)L * DQKV);
            minicpm_ocr_cpu::conv1d_silu(before_norm.data(), qkv.data(), w, L, DQKV, desc->conv_kernel_size);
            MINICPM_OCR_DUMP("linear_conv", before_norm.data(), L, DQKV);
        }
#else
        (void)w;
#endif
        // Push to the device only when the array GDN will read it (_conv_qkv_needed_on_host).
        if (_gdn_on_npu(L)) {
            minicpm_ocr_cpu::scratch_push(conv_qkv);
        }
    }

    /// @brief the gated delta rule over conv_qkv into common after_mix, updating the
    ///        recurrent state that sits behind the conv state in the cache slot
    /// @note  Declines calls shorter than gdn_npu_min_tokens(); the caller then runs
    ///        _gdn_host(). Mixing the two across calls is safe because each leaves the
    ///        state current on both sides: this path syncs `cache` back from the device,
    ///        _gdn_host() syncs it to the device.
    bool _gdn_npu(int L, buffer<bf16>& cache) {
#if MINICPM_OCR_GDN_ON_NPU
        if (!_gdn_on_npu(L)) {
            return false;
        }
        PREFILL_PROF_INIT();
        auto gdn_run = deltanet.create_run(cache, conv_qkv, alpha_beta, common_bufs->after_mix);
        gdn_run.start();
        _pull_gate(L);
        PREFILL_PROF_TICK("linear:.gdn_start+gate_sync_from (under gdn)");
        flm::npu::wait_for_completion(gdn_run, "gdn_run");
        PREFILL_PROF_TICK("linear:.gdn_run");

        minicpm_ocr_cpu::sync_range_from_device(common_bufs->after_mix, 0, (size_t)L * desc->Linear_DV);
        cache.sync_from_device();
        PREFILL_PROF_TICK("linear:.gdn_syncs_from");
        return true;
#else
        (void)L;
        (void)cache;
        return false;
#endif
    }

    void _gdn_host(int L, buffer<bf16>& cache) {
        PREFILL_PROF_INIT();
        float* S = reinterpret_cast<float*>(cache.data() + desc->get_s_offset());
        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        minicpm_ocr_cpu::gated_delta_net(
            common_bufs->after_mix.data(), S, conv_qkv.data(), desc->Linear_DQKV,
            0, desc->Linear_DQ, desc->Linear_DQ + desc->Linear_DK,
            alpha_ptr, beta_ptr, L,
            desc->linear_num_qk_heads, desc->linear_num_v_heads,
            desc->linear_qk_head_dim, desc->linear_v_head_dim
        );
        PREFILL_PROF_TICK("linear:.gdn_host");
        cache.sync_to_device();
        minicpm_ocr_cpu::scratch_push_range(common_bufs->after_mix, 0, (size_t)L * desc->Linear_DV);
        PREFILL_PROF_TICK("linear:.gdn_host_syncs_to");
    }

    /// @brief out_proj of after_mix into common x
    bool _o_proj_npu(buffer<u8>& proj_weights) {
#if MINICPM_OCR_MM_ON_NPU
        // no push of x first: the run only writes it, and the host copy is already clean
        // (pushed after the input norm, untouched since)
        PREFILL_PROF_INIT();
        linear_o_proj[0](common_bufs->x, common_bufs->mm_in(common_bufs->after_mix, common_bufs->ab), proj_weights);
        PREFILL_PROF_TICK("linear:.o_run");
        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("linear:.x_sync_from");
        return true;
#else
        (void)proj_weights;
        return false;
#endif
    }

    void _o_proj_host(int L, buffer<u8>& proj_weights) {
        minicpm_ocr_cpu::dequant_mm(common_bufs->x.data(), desc->D,
                                  proj_weights.data() + desc->get_linear_o_offset(),
                                  common_bufs->after_mix.data(), desc->Linear_DV, L, desc->Linear_DV, desc->D,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_cpu::scratch_push(common_bufs->x);
    }

    // -----------------------------------------------------------------------------------
    // Cross reference. Each block replays the stage above it on the host over the array's
    // own inputs and scores the two. ENABLE_DEBUG only; see minicpm_ocr_prefill_xref.hpp.
    // -----------------------------------------------------------------------------------
#ifdef ENABLE_DEBUG
    /// \brief score the two array projections of the input block
    /// \note  Called after the conv-state carry has overwritten rows [0, pad) of `qkv`,
    ///        which does not matter: only rows [pad, pad + rows) are scored.
    void _xref_in_proj(int L, buffer<u8>& proj_weights) {
#if MINICPM_OCR_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        _pull_gate(L);
        const int   DQKV = desc->Linear_DQKV;
        const int   pad  = desc->conv_kernel_size - 1;
        const bf16* got  = qkv.data() + (size_t)pad * DQKV;

        std::vector<bf16> want((size_t)rows * DQKV);
        minicpm_ocr_cpu::dequant_mm(want.data(), DQKV, proj_weights.data() + desc->get_linear_qkv_offset(),
                                  common_bufs->x.data(), desc->D, rows, desc->D, DQKV,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        {
            minicpm_ocr_prefill_xref::guard g(xref, "linear_qkv", layer_idx, got, DQKV, DQKV, rows);
            xref->check("linear_qkv", layer_idx, got, want.data(), DQKV, DQKV, rows);
        }

        // the array fuses the silu into the g projection (rtp_act = 1); the host does the
        // matmul and then silu_inplace, so the reference has to do both to be comparable
        const int DV = desc->Linear_DV;
        std::vector<bf16> want_g((size_t)rows * DV);
        minicpm_ocr_cpu::dequant_mm(want_g.data(), DV, proj_weights.data() + desc->get_linear_g_offset(),
                                  common_bufs->x.data(), desc->D, rows, desc->D, DV,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_cpu::silu_inplace(want_g.data(), (size_t)rows * DV);
        {
            minicpm_ocr_prefill_xref::guard g(xref, "linear_z", layer_idx, common_bufs->gate.data(), DV, DV, rows);
            xref->check("linear_z", layer_idx, common_bufs->gate.data(), want_g.data(), DV, DV, rows);
        }
#else
        (void)L; (void)proj_weights;
#endif
    }

    /// \brief score the conv overlay: causal conv1d + silu + the q/k L2 norm
    /// \note  Not row independent -- a row reads the conv_kernel_size - 1 rows before it --
    ///        so the replay runs over the whole prompt and only the report is truncated.
    void _xref_conv(int L, buffer<bf16>& ssm_weights) {
#if MINICPM_OCR_CONV_ON_NPU
        // a call that declined ran the host kernel; scoring it against itself means nothing
        if (xref == nullptr || !xref->enabled(layer_idx) || !_conv_on_npu(L)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int   DQKV = desc->Linear_DQKV;
        const bf16* w    = ssm_weights.data() + weights().ssm_conv1d.offset / sizeof(bf16);

        std::vector<bf16> want((size_t)L * DQKV);
        minicpm_ocr_cpu::conv1d_silu(want.data(), qkv.data(), w, L, DQKV, desc->conv_kernel_size);
        minicpm_ocr_cpu::l2norm_heads(want.data(), DQKV, L,
                                    (desc->Linear_DQ + desc->Linear_DK) / desc->linear_qk_head_dim,
                                    desc->linear_qk_head_dim);
        minicpm_ocr_prefill_xref::guard g(xref, "conv_qkv", layer_idx, conv_qkv.data(), DQKV, DQKV, rows);
        xref->check("conv_qkv", layer_idx, conv_qkv.data(), want.data(), DQKV, DQKV, rows);
#else
        (void)L; (void)ssm_weights;
#endif
    }

    /// \brief score the gated delta rule overlay, both its output and the state it leaves
    /// \param S_before snapshot of the recurrent state taken before the array run; the host
    ///        kernel updates the state in place, so replaying it against the array's own
    ///        cache slot would double-apply the scan. This is exactly what xref::guard
    ///        exists to catch.
    void _xref_gdn(int L, buffer<bf16>& cache, std::vector<float>& S_before) {
#if MINICPM_OCR_GDN_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx) || S_before.empty()) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int DV = desc->Linear_DV;

        std::vector<bf16> want((size_t)L * DV);
        float* alpha_ptr = alpha_beta.data();
        float* beta_ptr  = alpha_ptr + alpha_beta.size() / 2;
        minicpm_ocr_cpu::gated_delta_net(
            want.data(), S_before.data(), conv_qkv.data(), desc->Linear_DQKV,
            0, desc->Linear_DQ, desc->Linear_DQ + desc->Linear_DK,
            alpha_ptr, beta_ptr, L,
            desc->linear_num_qk_heads, desc->linear_num_v_heads,
            desc->linear_qk_head_dim, desc->linear_v_head_dim);
        {
            minicpm_ocr_prefill_xref::guard g(xref, "gdn_out", layer_idx, common_bufs->after_mix.data(), DV, DV, rows);
            xref->check("gdn_out", layer_idx, common_bufs->after_mix.data(), want.data(), DV, DV, rows);
        }

        // the state is fp32 [Hv][DK][DV] behind the conv state, and is the half of this
        // stage a wrong answer at token L would not show up in until the next chunk
        const float* S_got = reinterpret_cast<const float*>(cache.data() + desc->get_s_offset());
        const int    S_rows = (int)desc->linear_num_v_heads * desc->linear_qk_head_dim;
        minicpm_ocr_prefill_xref::guard_t<float> g(xref, "gdn_state", layer_idx, S_got,
                                                 desc->linear_v_head_dim, desc->linear_v_head_dim, S_rows);
        xref->check("gdn_state", layer_idx, S_got, S_before.data(),
                    desc->linear_v_head_dim, desc->linear_v_head_dim, S_rows);
#else
        (void)L; (void)cache; (void)S_before;
#endif
    }

    /// \brief score the output projection of the linear block
    /// \param got the block output; null: common x (the row-chunk path passes its copy, x is overwritten by then)
    void _xref_o_proj(int L, buffer<u8>& proj_weights, const bf16* got = nullptr) {
#if MINICPM_OCR_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        if (got == nullptr) { got = common_bufs->x.data(); }
        std::vector<bf16> want((size_t)rows * desc->D);
        minicpm_ocr_cpu::dequant_mm(want.data(), desc->D, proj_weights.data() + desc->get_linear_o_offset(),
                                  common_bufs->after_mix.data(), desc->Linear_DV, rows,
                                  desc->Linear_DV, desc->D, minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_prefill_xref::guard g(xref, "linear_o", layer_idx, got, desc->D, desc->D, rows);
        xref->check("linear_o", layer_idx, got, want.data(), desc->D, desc->D, rows);
#else
        (void)L; (void)proj_weights; (void)got;
#endif
    }
#endif // ENABLE_DEBUG

    /// @param post  null: the gated norm, then one o_proj run. Else the row-chunk path (mm_bfp16.xclbin): the o_proj
    ///              runs per chunk (run_o_chunks), post(c) is the caller's host work on chunk c's output rows, and
    ///              post of the last chunk, the block output's dump and its cross-check are left to the caller.
    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& ssm_weights, buffer<bf16>& cache, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu, const std::function<void(int)>* post = nullptr) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        const int DQKV = desc->Linear_DQKV;
        const int pad = desc->conv_kernel_size - 1;
        this->reserve(L, L_padded);

        PREFILL_PROF_TICK("linear:seq_gen+alloc");

        // in_proj, the conv-state carry, the conv and alpha/beta. On the array the host parts
        // run under the two projections (_in_proj_overlapped_npu), so the profile shows them
        // as one stage; on the host they run one after the other.
        if (_in_proj_overlapped_npu(L, proj_weights, ssm_weights, cache, alpha_proj_cpu, beta_proj_cpu)) {
            PREFILL_PROF_TICK("linear:qkv+g_proj+conv1d+alpha_beta (overlapped)");
        } else {
            _in_proj_host(L, proj_weights);
            PREFILL_PROF_TICK("linear:qkv+g_proj");
            _carry_conv_state(L, cache);
            _conv(L, ssm_weights);
            PREFILL_PROF_TICK("linear:conv1d");
            _alpha_beta(L, ssm_weights, alpha_proj_cpu, beta_proj_cpu);
            PREFILL_PROF_TICK("linear:alpha_beta");
        }
        // rows [pad, pad + L) of qkv are the projection; the carry only touched rows [0, pad)
        MINICPM_OCR_DUMP("linear_qkv", qkv.data() + (size_t)pad * DQKV, L, DQKV);
#ifdef ENABLE_DEBUG
        _xref_in_proj(L, proj_weights);
        _xref_conv(L, ssm_weights);
#endif

#ifdef ENABLE_DEBUG
        // snapshot the recurrent state before the array consumes it, so the host replay has
        // the same starting point (it updates the state in place)
        std::vector<float> S_before;
#if MINICPM_OCR_GDN_ON_NPU
        // Only when the array will actually run this call: a declined one runs the host
        // kernel, and scoring the host against itself would print a meaningless cos 1.0.
        if (xref != nullptr && xref->enabled(layer_idx) && _gdn_on_npu(L)) {
            cache.sync_from_device();
            const float* S = reinterpret_cast<const float*>(cache.data() + desc->get_s_offset());
            S_before.assign(S, S + (size_t)desc->linear_num_v_heads * desc->linear_qk_head_dim * desc->linear_v_head_dim);
        }
#endif
#endif

        if (!_gdn_npu(L, cache)) {
            _gdn_host(L, cache);
        }
#ifdef ENABLE_DEBUG
        _xref_gdn(L, cache, S_before);
#endif
        MINICPM_OCR_DUMP("linear_core", common_bufs->after_mix.data(), L, desc->Linear_DV);
        MINICPM_OCR_DUMP3("linear_state", reinterpret_cast<const float*>(cache.data() + desc->get_s_offset()),
                        desc->linear_num_v_heads, desc->linear_qk_head_dim, desc->linear_v_head_dim);
        PREFILL_PROF_TICK("linear:deltanet");

        _pull_gate(L);
        bf16* gate_rms_norm_weight_ptr = ssm_weights.data() + desc->get_linear_ssm_norm_offset() / sizeof(bf16);

        // the same per-head _gated_norm, on a copy of each row when the bf16 rows are not kept
        const int DV = desc->Linear_DV;
        const int head_stride = DV / desc->linear_num_v_heads;
        buffer<bf16>* keep_after_mix = minicpm_ocr_mm_bfp16::debug_keeps_bf16 ? &common_bufs->after_mix : nullptr;
        auto gated_norm_rows = [&](int row, int n, bf16* out){
            for (int r = 0; r < n; r++) {
                const bf16* in_row = common_bufs->after_mix.data() + (size_t)(row + r) * DV;
                bf16* out_row = out + (size_t)r * DV;
                if (out_row != in_row) {
                    memcpy(out_row, in_row, DV * sizeof(bf16));
                }
                for (int head_idx = 0; head_idx < (int)desc->linear_num_v_heads; head_idx++) {
                    cpu_func::_gated_norm(
                        out_row + head_idx * head_stride,
                        common_bufs->gate.data() + (size_t)(row + r) * DV + head_idx * head_stride,
                        gate_rms_norm_weight_ptr,
                        desc->linear_v_head_dim, 1
                    );
                }
            }
        };

#if MINICPM_OCR_MM_ON_NPU
        if (post != nullptr) {
            const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
            auto prep = [&](int c){
                common_bufs->ab.publish_rows(keep_after_mix, chunks.begin[c], chunks.rows(c, L), L, gated_norm_rows);
            };
            run_o_chunks(linear_o_proj, chunks.count, common_bufs->x, common_bufs->ab.buf, proj_weights, prep, *post);
            MINICPM_OCR_DUMP("linear_norm", common_bufs->after_mix.data(), L, desc->Linear_DV);
            PREFILL_PROF_TICK("linear:gated_norm+o_proj (chunks)");
            return buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        }
#endif

        if (common_bufs->mm_bfp16) {
            common_bufs->ab.publish(keep_after_mix, L, gated_norm_rows);
        }
        else {
            #pragma omp parallel for num_threads(cpu_func::host_threads()) schedule(static)
            for(int ll = 0; ll< L; ll++){
                for(int head_idx = 0; head_idx < (int)desc->linear_num_v_heads; head_idx++){
                    uint32_t offset = (ll*desc->Linear_DV) + head_idx * (desc->Linear_DV/desc->linear_num_v_heads);

                    cpu_func::_gated_norm(
                        common_bufs->after_mix.data() + offset,
                        common_bufs->gate.data() + offset,
                        gate_rms_norm_weight_ptr,
                        desc->linear_v_head_dim, 1
                    );
                }
            }
        }
        MINICPM_OCR_DUMP("linear_norm", common_bufs->after_mix.data(), L, desc->Linear_DV);

        if (!common_bufs->mm_bfp16) {
            minicpm_ocr_cpu::scratch_push(common_bufs->after_mix);
        }
        PREFILL_PROF_TICK("linear:gated_norm");

        if (!_o_proj_npu(proj_weights)) {
            _o_proj_host(L, proj_weights);
        }
        MINICPM_OCR_DUMP("linear_out", common_bufs->x.data(), L, desc->D);
#ifdef ENABLE_DEBUG
        _xref_o_proj(L, proj_weights);
#endif
        PREFILL_PROF_TICK("linear:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }
};

struct minicpm_ocr_attn_block_prefill_context{
    minicpm_ocr_desc* desc;
    minicpm_ocr_npu_sequence* seq_gen;
    minicpm_ocr_common_buffers* common_bufs;
    npu_xclbin_manager* npu;

    int L_padded_old = -1;
    //!< the (L_begin, L_padded) the attention array sequence was last generated for; only calls
    //!< that run on the array regenerate it (see _attn_on_npu)
    int mha_seq_L_begin = -1;
    int mha_seq_Lp = -1;
    //!< set by minicpm_ocr_prefill_context::forward() before every call (also keys the bfp16 KV copy)
    int layer_idx = -1;

    //!< attn_bfp16.xclbin path (attn_bfp16.hpp): q tiles, each layer's bfp16 KV copy, the sequence's (L_begin, L_padded, rows)
    bool use_bfp16 = false;
    buffer<u8> qb;
    bool qb_ready = false;
    minicpm_ocr_attn_bfp16::kv_store kvb;
    int bfp16_seq_L_begin = -1;
    int bfp16_seq_Lp = -1;
    int bfp16_seq_rows = -1;
    //!< elements of one attention layer's bf16 KV cache (set by the engine): lets _reserve_ahead size the bfp16 copies
    size_t kv_cache_elems = 0;
#ifdef ENABLE_DEBUG
    //!< non-null in a debug build; whether it scores anything is up to enabled(layer_idx)
    minicpm_ocr_prefill_xref* xref = nullptr;

    /// \brief score one projection of this block against the host kernel
    /// \note Called from inside _qkvg_proj_npu() for q and k, because both are rotated in
    ///       place immediately after the run and the raw projection is gone by the time
    ///       forward() gets control back.
    void _xref_proj(const char* name, int L, const bf16* got, size_t w_offset,
                    buffer<u8>& proj_weights, int N) {
#if MINICPM_OCR_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        std::vector<bf16> want((size_t)rows * N);
        minicpm_ocr_cpu::dequant_mm(want.data(), N, proj_weights.data() + w_offset,
                                  common_bufs->x.data(), desc->D, rows, desc->D, N,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_prefill_xref::guard gu(xref, name, layer_idx, got, N, N, rows);
        xref->check(name, layer_idx, got, want.data(), N, N, rows);
#else
        (void)name; (void)L; (void)got; (void)w_offset; (void)proj_weights; (void)N;
#endif
    }

    /// \brief score the attention overlay against the host kernel over the same cache
    /// \note Gated on enabled() *before* anything is read, because the reference is an
    ///       L x DQ tensor plus a full pass over the kv cache -- for a 4 k prompt that is
    ///       more work than the attention it is checking, every layer.
    void _xref_attend(int L, buffer<bf16>& cache, int L_begin) {
#if MINICPM_OCR_ATTN_ON_NPU
        // a call that declined ran the host kernel; scoring it against itself means nothing
        if (xref == nullptr || !xref->enabled(layer_idx) || !_attn_on_npu(L, L_begin)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        // reads rows decode may have written on the array; fill_kv_cache no longer pulls
        cache.sync_from_device();
        const bf16* k_cache = cache.data();
        const bf16* v_cache = cache.data() + cache.size() / 2;
        std::vector<bf16> want((size_t)L * desc->DQ);
        minicpm_ocr_cpu::attention(want.data(), q.data(), k_cache, v_cache,
                                 L, L_begin + L, desc->num_heads, desc->num_kv_heads, desc->DH,
                                 L_begin, 0, desc->DK, desc->DV, 1.0f / sqrtf((float)desc->DH));
        minicpm_ocr_prefill_xref::guard gu(xref, "attention_out", layer_idx,
                                         common_bufs->after_mix.data(), desc->DQ, desc->DQ, rows);
        xref->check("attention_out", layer_idx, common_bufs->after_mix.data(), want.data(),
                    desc->DQ, desc->DQ, rows);
#else
        (void)L; (void)cache; (void)L_begin;
#endif
    }

    /// \brief score the output projection of the attention block
    /// \param got the block output; null: common x (the row-chunk path passes its copy, x is overwritten by then)
    void _xref_o_proj(int L, buffer<u8>& proj_weights, const bf16* got = nullptr) {
#if MINICPM_OCR_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx)) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        if (got == nullptr) { got = common_bufs->x.data(); }
        std::vector<bf16> want((size_t)rows * desc->D);
        minicpm_ocr_cpu::dequant_mm(want.data(), desc->D, proj_weights.data() + desc->get_attn_o_offset(),
                                  g.data(), desc->DQ, rows, desc->DQ, desc->D,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_prefill_xref::guard gu(xref, "attn_o", layer_idx, got, desc->D, desc->D, rows);
        xref->check("attn_o", layer_idx, got, want.data(), desc->D, desc->D, rows);
#else
        (void)L; (void)proj_weights; (void)got;
#endif
    }
#endif // ENABLE_DEBUG

#if MINICPM_OCR_MM_ON_NPU
    npu_app q_proj;
    npu_app k_proj;
    npu_app v_proj;
    npu_app g_proj;
    npu_app o_proj[minicpm_ocr_chunks::max_chunks];
#endif
#if MINICPM_OCR_ATTN_ON_NPU
    npu_app mha;
    npu_app mha_bfp16;
#endif

    buffer<bf16> q;
    buffer<bf16> k;
    buffer<bf16> v;
    buffer<bf16> g;
    buffer<bf16> g_sigmoid;

    minicpm_ocr_attn_block_prefill_context(
        minicpm_ocr_desc* desc,
        minicpm_ocr_npu_sequence* seq_gen,
        minicpm_ocr_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        npu_app_manager* gemm_app_manager,
        npu_app_manager* mha_app_manager,
        npu_app_manager* mha_bfp16_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu){
#if MINICPM_OCR_MM_ON_NPU
        this->q_proj = gemm_app_manager->create_app();
        this->k_proj = gemm_app_manager->create_app();
        this->v_proj = gemm_app_manager->create_app();
        this->g_proj = gemm_app_manager->create_app();
        for (npu_app& app : this->o_proj) {
            app = gemm_app_manager->create_app();
        }
#endif
#if MINICPM_OCR_ATTN_ON_NPU
        if (mha_bfp16_app_manager != nullptr) {
            this->use_bfp16 = true;
            this->mha_bfp16 = mha_bfp16_app_manager->create_app();
        }
        else {
            this->mha = mha_app_manager->create_app();
        }
#else
        (void)mha_app_manager;
        (void)mha_bfp16_app_manager;
#endif
    }

    std::vector<int> attn_layer_ids() const {
        std::vector<int> ids;
        for (int i = 0; i < (int)desc->num_hidden_layers; i++) {
            if (desc->is_attn_layer(i)) {
                ids.push_back(i);
            }
        }
        return ids;
    }

    /// @brief whether this call's array attention needs bigger bfp16 KV copies than the ones there (a new BO each)
    bool kv_copies_short(int L, int L_padded, int L_begin) const {
#if MINICPM_OCR_ATTN_ON_NPU
        if (!use_bfp16 || kv_cache_elems == 0 || !_attn_on_npu(L, L_begin)) {
            return false;
        }
        return kvb.short_for(attn_layer_ids(), L_begin, L_padded, (int)(kv_cache_elems / 2 / desc->DK));
#else
        (void)L; (void)L_padded; (void)L_begin;
        return false;
#endif
    }

    /// @brief the bfp16 KV copies of every attention layer sized for this call ahead of its first attention layer
    ///        (_reserve_ahead's thread): the same rows and BOs kv_store::prepare() would make there
    void reserve_kv_copies(int L, int L_padded, int L_begin) {
        if (kv_copies_short(L, L_padded, L_begin)) {
            kvb.reserve(npu, desc, attn_layer_ids(), L_begin, L_padded, (int)(kv_cache_elems / 2 / desc->DK));
        }
    }

    /// @brief the host rewrote the bf16 KV caches from row from_row on (clear, restore, write_state): the bfp16
    ///        copies convert those rows again on their next array call
    void kv_cache_changed(int from_row) {
        kvb.cache_changed(from_row);
    }

    /// @brief FLM_MINICPM_OCR_ATTN_BACKEND: "npu" or "host" forces every prefill attention to
    ///        that side; anything else (or unset) is "auto", the cost estimate below. Read once.
    enum class attn_backend { autopick, npu, host };
    static attn_backend attn_backend_setting() {
        static const attn_backend b = [] {
            const char* env = std::getenv("FLM_MINICPM_OCR_ATTN_BACKEND");
            if (env != nullptr && std::string(env) == "npu") return attn_backend::npu;
            if (env != nullptr && std::string(env) == "host") return attn_backend::host;
            return attn_backend::autopick;
        }();
        return b;
    }

    /// @brief Estimated cost of ONE attention layer of this call, on each side, in us.
    ///
    /// The host kernel does exactly the work asked for: L new rows, each scored against the
    /// L_begin cached rows plus, on average, half of the new ones -- ~33 ns per query-key
    /// pair (8 heads x 256 dims, all cores). The array pays two context switches (into
    /// attn.xclbin and back to mm.xclbin, ~6.5 ms together) and then works on the PADDED
    /// rows against the whole cache, ~0.93 ns per pair. Fitted to the per-stage profile at
    /// 128 / 512 / 1024 / 2048 / 8192 fresh tokens.
    ///
    /// What it predicts, and what was measured (README.md, "Where each stage runs"):
    ///   * a fresh prompt goes to the array from ~700 tokens (measured crossover 750)
    ///   * a short follow-up stays on the host even on a long conversation, because the
    ///     array would still process 512 padded rows against all of it (34 new tokens on
    ///     2k / 4k / 8k cached: the host was ~100 ms faster per turn at every size)
    ///
    /// attn_bfp16.xclbin (use_bfp16): ~6.6 ms fixed, ~0.40 ns per padded pair, and ~0.13 us per
    /// cache row for keeping the bfp16 copy current. Fitted to 13 calls (128 .. 24K fresh, short
    /// follow-ups on 3K / 8K cached), README.md "Prefill attention on bfp16 tiles".
    static double attn_host_us(int L, int L_begin) {
        const double pairs = (double)L * ((double)L_begin + (L + 1) / 2);
        return 0.033 * pairs;
    }
    static constexpr double ATTN_BFP16_FIXED_US = 6575.0;
    static constexpr double ATTN_BFP16_PAIR_US = 0.000399;
    static constexpr double ATTN_BFP16_ROW_US = 0.13;
    double attn_npu_us(int L, int L_begin) const {
        const double L_padded = (double)((L + 511) / 512 * 512);
        if (use_bfp16) {
            return ATTN_BFP16_FIXED_US + ATTN_BFP16_PAIR_US * L_padded * ((double)L_begin + L_padded)
                 + ATTN_BFP16_ROW_US * ((double)L_begin + L_padded);
        }
        return 6500.0 + 0.00093 * L_padded * ((double)L_begin + L_padded);
    }

    /// @brief Whether this call's attention goes to the array: a pure function of L and
    ///        L_begin, which keeps the cached array sequence (mha_seq_*) consistent.
    bool _attn_on_npu(int L, int L_begin) const {
#if MINICPM_OCR_ATTN_ON_NPU
        switch (attn_backend_setting()) {
            case attn_backend::npu:  return true;
            case attn_backend::host: return false;
            default:                 return attn_npu_us(L, L_begin) < attn_host_us(L, L_begin);
        }
#else
        (void)L;
        (void)L_begin;
        return false;
#endif
    }

    /// @brief grow the scratch and regenerate the array sequences for this L / L_begin
    void reserve(int L, int L_padded, int L_begin) {
        PREFILL_PROF_INIT();
        if (L_padded_old != L_padded){
#if MINICPM_OCR_MM_ON_NPU
            if (common_bufs->mm_bfp16) {
                seq_gen->gen_dequant_mm_bfp16(this->q_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0);
                seq_gen->gen_dequant_mm_bfp16(this->k_proj.seq(), L_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0);
                seq_gen->gen_dequant_mm_bfp16(this->v_proj.seq(), L_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0);
                seq_gen->gen_dequant_mm_bfp16(this->g_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0);
                const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
                for (int c = 0; c < chunks.count; c++) {
                    seq_gen->gen_dequant_mm_bfp16(this->o_proj[c].seq(), chunks.padded[c], desc->DQ, desc->D, desc->get_attn_o_offset(), (size_t)chunks.begin[c] * desc->D, 0, false, chunks.begin[c]);
                }
            }
            else {
                seq_gen->gen_dequant_mm_512(this->q_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_q_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->k_proj.seq(), L_padded, desc->D, desc->DK, desc->get_attn_k_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->v_proj.seq(), L_padded, desc->D, desc->DV, desc->get_attn_v_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->g_proj.seq(), L_padded, desc->D, desc->DQ, desc->get_attn_g_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->o_proj[0].seq(), L_padded, desc->DQ, desc->D, desc->get_attn_o_offset(), 0, 0);
            }
#endif
            PREFILL_PROF_TICK("reserve:attn.mm_seq_gen");
            q  = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DQ);
            k  = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DK);
            v  = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DV);
            g  = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->DQ);
            g_sigmoid  = buffer<bf16>((size_t)L_padded * desc->DQ);
#if MINICPM_OCR_ATTN_ON_NPU
            if (use_bfp16) {
                qb = npu->create_bo_buffer<u8>((size_t)L_padded * desc->DQ * 9 / 8);
            }
#endif
            PREFILL_PROF_TICK("reserve:attn.alloc");
        }
#if MINICPM_OCR_ATTN_ON_NPU
        // only for calls that will run on the array; a host call must not pay for (or
        // overwrite) the sequence an array call is using
        if (!use_bfp16 && _attn_on_npu(L, L_begin) && (mha_seq_L_begin != L_begin || mha_seq_Lp != L_padded)) {
            seq_gen->gen_mha_engine_seq(mha.seq(), L_begin, L_padded);
            mha_seq_L_begin = L_begin;
            mha_seq_Lp = L_padded;
        }
#else
        (void)L;
#endif
        L_padded_old = L_padded;
    }

    /// @brief q / gate / k / v projections, q_norm + k_norm + rope on q and k, sigmoid of
    ///        the gate into g_sigmoid. On the array the host stages overlap the next run.
    bool _qkvg_proj_npu(int L, buffer<u8>& proj_weights, bf16* q_norm_weight, bf16* k_norm_weight, std::vector<int>& input_idx, bool make_qb) {
#if MINICPM_OCR_MM_ON_NPU
        PREFILL_PROF_INIT();
        bytes& x_in = common_bufs->mm_in(common_bufs->x, common_bufs->xb);
        q_proj(q, x_in, proj_weights);
        PREFILL_PROF_TICK("attn:.q_run");

        // the q tiles for the array read every padded row; the host steps read rows [0, L)
        if (make_qb) {
            q.sync_from_device();
        }
        else {
            minicpm_ocr_cpu::sync_range_from_device(q, 0, (size_t)L * desc->DQ);
        }
        PREFILL_PROF_TICK("attn:.q_sync_from");
        MINICPM_OCR_DUMP("attn_q_raw", q.data(), L, desc->DQ);
#ifdef ENABLE_DEBUG
        //!< q and k are rotated in place below, so the raw projection has to be scored here
        _xref_proj("attn_q", L, q.data(), desc->get_attn_q_offset(), proj_weights, desc->DQ);
#endif
        auto g_run = g_proj.create_run(g, x_in, proj_weights);
        g_run.start();
        cpu_func::_rope_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        minicpm_ocr_cpu::sync_range_to_device(q, 0, (size_t)L * desc->DQ);
        if (make_qb) {
            minicpm_ocr_attn_bfp16::q_tiles(qb, q.data(), (L + 511) / 512 * 512, desc->num_heads, desc->DH);
            qb_ready = true;
        }
        PREFILL_PROF_TICK("attn:.q_rope+sync_to+tiles (under g)");
        flm::npu::wait_for_completion(g_run, "g_run");
        PREFILL_PROF_TICK("attn:.g_wait");
        minicpm_ocr_cpu::sync_range_from_device(g, 0, (size_t)L * desc->DQ);
        PREFILL_PROF_TICK("attn:.g_sync_from");
        MINICPM_OCR_DUMP("attn_gate_raw", g.data(), L, desc->DQ);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_gate", L, g.data(), desc->get_attn_g_offset(), proj_weights, desc->DQ);
#endif

        auto k_run = k_proj.create_run(k, x_in, proj_weights);
        k_run.start();
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * desc->DQ);
        minicpm_ocr_cpu::sync_range_to_device(g, 0, (size_t)L * desc->DQ);
        PREFILL_PROF_TICK("attn:.sigmoid+g_sync_to (under k)");
        flm::npu::wait_for_completion(k_run, "k_run");
        PREFILL_PROF_TICK("attn:.k_wait");
        minicpm_ocr_cpu::sync_range_from_device(k, 0, (size_t)L * desc->DK);
        PREFILL_PROF_TICK("attn:.k_sync_from");
#ifdef ENABLE_DEBUG
        _xref_proj("attn_k", L, k.data(), desc->get_attn_k_offset(), proj_weights, desc->DK);
#endif

        auto v_run = v_proj.create_run(v, x_in, proj_weights);
        v_run.start();
        cpu_func::_rope_rms_batch(k, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        minicpm_ocr_cpu::sync_range_to_device(k, 0, (size_t)L * desc->DK);
        PREFILL_PROF_TICK("attn:.k_rope+sync_to (under v)");
        flm::npu::wait_for_completion(v_run, "v_run");
        PREFILL_PROF_TICK("attn:.v_wait");
        minicpm_ocr_cpu::sync_range_from_device(v, 0, (size_t)L * desc->DV);
        PREFILL_PROF_TICK("attn:.v_sync_from");
#ifdef ENABLE_DEBUG
        _xref_proj("attn_v", L, v.data(), desc->get_attn_v_offset(), proj_weights, desc->DV);
#endif
        return true;
#else
        (void)L; (void)proj_weights; (void)q_norm_weight; (void)k_norm_weight; (void)input_idx; (void)make_qb;
        return false;
#endif
    }

    void _qkvg_proj_host(int L, buffer<u8>& proj_weights, bf16* q_norm_weight, bf16* k_norm_weight, std::vector<int>& input_idx) {
        const u8*   w = proj_weights.data();
        const bf16* x = common_bufs->x.data();
        const int   D = desc->D;
        minicpm_ocr_cpu::dequant_mm(q.data(), desc->DQ, w + desc->get_attn_q_offset(), x, D, L, D, desc->DQ, minicpm_ocr_cpu::BODY_LAYOUT);
        MINICPM_OCR_DUMP("attn_q_raw", q.data(), L, desc->DQ);
        minicpm_ocr_cpu::dequant_mm(g.data(), desc->DQ, w + desc->get_attn_g_offset(), x, D, L, D, desc->DQ, minicpm_ocr_cpu::BODY_LAYOUT);
        MINICPM_OCR_DUMP("attn_gate_raw", g.data(), L, desc->DQ);
        minicpm_ocr_cpu::dequant_mm(k.data(), desc->DK, w + desc->get_attn_k_offset(), x, D, L, D, desc->DK, minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_cpu::dequant_mm(v.data(), desc->DV, w + desc->get_attn_v_offset(), x, D, L, D, desc->DV, minicpm_ocr_cpu::BODY_LAYOUT);
        cpu_func::_rope_rms_batch(q, desc->DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        cpu_func::_rope_rms_batch(k, desc->DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * desc->DQ);
        minicpm_ocr_cpu::scratch_push(q);
        minicpm_ocr_cpu::scratch_push(k);
        minicpm_ocr_cpu::scratch_push(v);
        minicpm_ocr_cpu::scratch_push(g);
    }

    /// @brief rows [L_begin, L_begin + L) of one cache half (half_offset: 0 = K, cache.size() / 2 = V) from src, as
    ///        fill_kv_cache writes them; rows of `width` elements in both, so one contiguous block, copied by all host
    ///        threads. No flush.
    void _copy_rows_to_cache(buffer<bf16>& cache, size_t half_offset, const bf16* src, int width, int L_begin, int L) {
        const size_t n = (size_t)L * width;
        bf16* dst = cache.data() + half_offset + (size_t)L_begin * width;
        const int threads = minicpm_ocr_cpu::prefill_threads();
        const size_t step = (n + threads - 1) / threads;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int t = 0; t < threads; t++) {
            const size_t begin = (size_t)t * step;
            if (begin < n) {
                memcpy(dst + begin, src + begin, std::min(step, n - begin) * sizeof(bf16));
            }
        }
    }

    /// @brief The whole q / g / k / v + attention stage of an array call on attn_bfp16.xclbin, with every host step
    ///        under an array run (README.md "Attention-layer host steps (WP4d)"):
    ///
    ///        array:  k     v                     q                          g                  mha
    ///        host:         k: pull, norm+RoPE,   K tiles; v: pull, V rows,  q: pull, norm+RoPE, g: pull, sigmoid;
    ///                      K rows                V^T tiles                  q tiles            flushes
    ///
    ///        Each step does the same work as in _qkvg_proj_npu + fill_kv_cache + _attend_npu (same functions, same
    ///        rows), only in another order, so the output is bit-identical. The flushes of q, k, g and the new cache
    ///        rows move under the attention run: nothing reads those device copies before the next layer or decode.
    /// @return false (nothing done) unless the mm and this call's attention are both on the array with bfp16 tiles
    bool _qkvg_attend_bfp16_npu(int L, buffer<u8>& proj_weights, bf16* q_norm_weight, bf16* k_norm_weight,
                                std::vector<int>& input_idx, buffer<bf16>& cache, int L_begin) {
#if MINICPM_OCR_MM_ON_NPU && MINICPM_OCR_ATTN_ON_NPU
        if (!use_bfp16 || !_attn_on_npu(L, L_begin)) {
            return false;
        }
        PREFILL_PROF_INIT();
        const int L_padded = (L + 511) / 512 * 512;
        const int DQ = desc->DQ;
        const int DK = desc->DK;
        const int DV = desc->DV;
        const size_t k_first = (size_t)L_begin * DK;
        const size_t v_first = (size_t)L_begin * DV + cache.size() / 2;
        bytes& x_in = common_bufs->mm_in(common_bufs->x, common_bufs->xb);

        auto k_run = k_proj.create_run(k, x_in, proj_weights);
        k_run.start();
        const minicpm_ocr_attn_bfp16::kv_plan plan = kvb.prepare(npu, desc, layer_idx, cache, L_begin, L, L_padded);
        PREFILL_PROF_TICK("attn:.kv_prepare (under k)");
        flm::npu::wait_for_completion(k_run, "k_run");
        PREFILL_PROF_TICK("attn:.k_wait");

        auto v_run = v_proj.create_run(v, x_in, proj_weights);
        v_run.start();
        minicpm_ocr_cpu::sync_range_from_device(k, 0, (size_t)L * DK);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_k", L, k.data(), desc->get_attn_k_offset(), proj_weights, DK);
#endif
        cpu_func::_rope_rms_batch(k, DK, input_idx, L, desc->num_kv_heads, k_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        _copy_rows_to_cache(cache, 0, k.data(), DK, L_begin, L);
        PREFILL_PROF_TICK("attn:.k_rope+K rows (under v)");
        flm::npu::wait_for_completion(v_run, "v_run");
        PREFILL_PROF_TICK("attn:.v_wait");

        auto q_run = q_proj.create_run(q, x_in, proj_weights);
        q_run.start();
        kvb.convert(plan, desc, cache, minicpm_ocr_attn_bfp16::kv_half::k);
        minicpm_ocr_cpu::sync_range_from_device(v, 0, (size_t)L * DV);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_v", L, v.data(), desc->get_attn_v_offset(), proj_weights, DV);
#endif
        _copy_rows_to_cache(cache, cache.size() / 2, v.data(), DV, L_begin, L);
        kvb.convert(plan, desc, cache, minicpm_ocr_attn_bfp16::kv_half::v);
        minicpm_ocr_attn_bfp16::kv_copy& kv = kvb.finish(plan);
        PREFILL_PROF_TICK("attn:.K tiles+V rows+V tiles (under q)");
        flm::npu::wait_for_completion(q_run, "q_run");
        PREFILL_PROF_TICK("attn:.q_wait");

        auto g_run = g_proj.create_run(g, x_in, proj_weights);
        g_run.start();
        // the q tiles read every padded row
        q.sync_from_device();
        MINICPM_OCR_DUMP("attn_q_raw", q.data(), L, DQ);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_q", L, q.data(), desc->get_attn_q_offset(), proj_weights, DQ);
#endif
        cpu_func::_rope_rms_batch(q, DQ, input_idx, L, desc->num_heads, q_norm_weight, desc->RoPE_DIM, desc->DH, desc->inv_freq);
        minicpm_ocr_attn_bfp16::q_tiles(qb, q.data(), L_padded, desc->num_heads, desc->DH);
        qb_ready = true;
        if (bfp16_seq_L_begin != L_begin || bfp16_seq_Lp != L_padded || bfp16_seq_rows != kvb.rows) {
            seq_gen->gen_mha_bfp16_seq(mha_bfp16.seq(), L_begin, L_padded, kvb.rows);
            bfp16_seq_L_begin = L_begin;
            bfp16_seq_Lp = L_padded;
            bfp16_seq_rows = kvb.rows;
        }
        PREFILL_PROF_TICK("attn:.q_rope+q tiles (under g)");
        flm::npu::wait_for_completion(g_run, "g_run");
        PREFILL_PROF_TICK("attn:.g_wait");

        auto mha_run = mha_bfp16.create_run(common_bufs->after_mix, qb, kv.tiles);
        mha_run.start();
        minicpm_ocr_cpu::sync_range_from_device(g, 0, (size_t)L * DQ);
        MINICPM_OCR_DUMP("attn_gate_raw", g.data(), L, DQ);
#ifdef ENABLE_DEBUG
        _xref_proj("attn_gate", L, g.data(), desc->get_attn_g_offset(), proj_weights, DQ);
#endif
        cpu_func::_sigmoid(g_sigmoid.data(), g.data(), L * DQ);
        minicpm_ocr_cpu::sync_range_to_device(g, 0, (size_t)L * DQ);
        minicpm_ocr_cpu::sync_range_to_device(q, 0, (size_t)L * DQ);
        minicpm_ocr_cpu::sync_range_to_device(k, 0, (size_t)L * DK);
        minicpm_ocr_cpu::sync_range_to_device(cache, k_first, (size_t)L * DK);
        minicpm_ocr_cpu::sync_range_to_device(cache, v_first, (size_t)L * DV);
        PREFILL_PROF_TICK("attn:.sigmoid+flushes (under mha)");
        flm::npu::wait_for_completion(mha_run, "mha_run");
        PREFILL_PROF_TICK("attn:.mha_wait");

        minicpm_ocr_cpu::sync_range_from_device(common_bufs->after_mix, 0, (size_t)L * DQ);
        PREFILL_PROF_TICK("attn:.after_mix_sync_from");
        return true;
#else
        (void)L; (void)proj_weights; (void)q_norm_weight; (void)k_norm_weight; (void)input_idx; (void)cache; (void)L_begin;
        return false;
#endif
    }

    /// @brief causal attention of q against cache rows [0, L_begin + L) into common after_mix
    /// @note  Declines calls the host is estimated to do faster (_attn_on_npu); the caller runs
    ///        _attend_host(). With the GDN also on the host below gdn_npu_min_tokens(), a short
    ///        prompt then never leaves mm.xclbin: no context switch at all.
    bool _attend_npu(int L, buffer<bf16>& cache, int L_begin) {
#if MINICPM_OCR_ATTN_ON_NPU
        if (!_attn_on_npu(L, L_begin)) {
            return false;
        }
        PREFILL_PROF_INIT();
        if (use_bfp16) {
            const int L_padded = (L + 511) / 512 * 512;
            if (!qb_ready) {
                minicpm_ocr_attn_bfp16::q_tiles(qb, q.data(), L_padded, desc->num_heads, desc->DH);
            }
            minicpm_ocr_attn_bfp16::kv_copy& kv = kvb.update(npu, desc, layer_idx, cache, L_begin, L, L_padded);
            PREFILL_PROF_TICK("attn:.kv_bfp16_update");
            if (bfp16_seq_L_begin != L_begin || bfp16_seq_Lp != L_padded || bfp16_seq_rows != kvb.rows) {
                seq_gen->gen_mha_bfp16_seq(mha_bfp16.seq(), L_begin, L_padded, kvb.rows);
                bfp16_seq_L_begin = L_begin;
                bfp16_seq_Lp = L_padded;
                bfp16_seq_rows = kvb.rows;
            }
            mha_bfp16(common_bufs->after_mix, qb, kv.tiles);
        }
        else {
            mha(common_bufs->after_mix, q, cache);
        }
        PREFILL_PROF_TICK("attn:.mha_run");

        minicpm_ocr_cpu::sync_range_from_device(common_bufs->after_mix, 0, (size_t)L * desc->DQ);
        PREFILL_PROF_TICK("attn:.after_mix_sync_from");
        return true;
#else
        (void)L;
        (void)cache;
        (void)L_begin;
        return false;
#endif
    }

    void _attend_host(int L, buffer<bf16>& cache, int L_begin) {
        // The rows before L_begin may have been written by the array (decode), and
        // fill_kv_cache does not pull the cache, so pull it here, the one host reader -- only
        // the rows this call reads, K and V. Rows are 1 KB, whole cache lines.
        PREFILL_PROF_INIT();
        const size_t rows = (size_t)L_begin + L;
        minicpm_ocr_cpu::sync_range_from_device(cache, 0, rows * desc->DK);
        minicpm_ocr_cpu::sync_range_from_device(cache, cache.size() / 2, rows * desc->DV);
        // fill_kv_cache's layout: K rows from 0, V rows from the half-way point
        const bf16* k_cache = cache.data();
        const bf16* v_cache = cache.data() + cache.size() / 2;
        minicpm_ocr_cpu::attention(common_bufs->after_mix.data(), q.data(), k_cache, v_cache,
                                 L, L_begin + L, desc->num_heads, desc->num_kv_heads, desc->DH,
                                 L_begin, 0, desc->DK, desc->DV, 1.0f / sqrtf((float)desc->DH));
        PREFILL_PROF_TICK("attn:.host_attention");
        minicpm_ocr_cpu::scratch_push_range(common_bufs->after_mix, 0, (size_t)L * desc->DQ);
        PREFILL_PROF_TICK("attn:.after_mix_sync_to");
    }

    /// @brief o_proj of the gated attention (g) into common x
    bool _o_proj_npu(buffer<u8>& proj_weights) {
#if MINICPM_OCR_MM_ON_NPU
        PREFILL_PROF_INIT();
        o_proj[0](common_bufs->x, common_bufs->mm_in(g, common_bufs->ab), proj_weights);
        PREFILL_PROF_TICK("attn:.o_run");

        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("attn:.x_sync_from");
        return true;
#else
        (void)proj_weights;
        return false;
#endif
    }

    void _o_proj_host(int L, buffer<u8>& proj_weights) {
        minicpm_ocr_cpu::dequant_mm(common_bufs->x.data(), desc->D,
                                  proj_weights.data() + desc->get_attn_o_offset(),
                                  g.data(), desc->DQ, L, desc->DQ, desc->D, minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_cpu::scratch_push(common_bufs->x);
    }

    /// @param post  as in the linear block's forward(): null runs one o_proj, else the row-chunk path
    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rope_rms_weights, buffer<bf16>& cache, std::vector<int>& input_idx, int L_begin, const std::function<void(int)>* post = nullptr) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        this->reserve(L, L_padded, L_begin);
        PREFILL_PROF_TICK("attn:seq_gen+alloc");

        bf16* q_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM;
        bf16* k_norm_weight = rope_rms_weights.data() + desc->RoPE_DIM + desc->DH;

        qb_ready = false;
        if (_qkvg_attend_bfp16_npu(L, proj_weights, q_norm_weight, k_norm_weight, input_idx, cache, L_begin)) {
            MINICPM_OCR_DUMP("attn_q_rope", q.data(), L, desc->DQ);
            MINICPM_OCR_DUMP("attn_k_rope", k.data(), L, desc->DK);
            MINICPM_OCR_DUMP("attn_v", v.data(), L, desc->DV);
            PREFILL_PROF_TICK("attn:qkvg_proj+kv+mha");
        }
        else {
            if (!_qkvg_proj_npu(L, proj_weights, q_norm_weight, k_norm_weight, input_idx, use_bfp16 && _attn_on_npu(L, L_begin))) {
                _qkvg_proj_host(L, proj_weights, q_norm_weight, k_norm_weight, input_idx);
            }
            MINICPM_OCR_DUMP("attn_q_rope", q.data(), L, desc->DQ);
            MINICPM_OCR_DUMP("attn_k_rope", k.data(), L, desc->DK);
            MINICPM_OCR_DUMP("attn_v", v.data(), L, desc->DV);

            PREFILL_PROF_TICK("attn:qkvg_proj");

            fill_kv_cache(k, v, cache, L_begin, L);
            PREFILL_PROF_TICK("attn:fill_kv_cache");

            if (!_attend_npu(L, cache, L_begin)) {
                _attend_host(L, cache, L_begin);
            }
        }
        MINICPM_OCR_DUMP("attn_out", common_bufs->after_mix.data(), L, desc->DQ);
#ifdef ENABLE_DEBUG
        _xref_attend(L, cache, L_begin);
#endif
        PREFILL_PROF_TICK("attn:mha");

        const int DQ = desc->DQ;
        buffer<bf16>* keep_g = minicpm_ocr_mm_bfp16::debug_keeps_bf16 ? &g : nullptr;
        auto gate_rows = [&](int row, int n, bf16* out){
            cpu_func::_elementwise_mul_serial(out, g_sigmoid.data() + (size_t)row * DQ, common_bufs->after_mix.data() + (size_t)row * DQ, n * DQ);
        };
#if MINICPM_OCR_MM_ON_NPU
        if (post != nullptr) {
            const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
            auto prep = [&](int c){
                common_bufs->ab.publish_rows(keep_g, chunks.begin[c], chunks.rows(c, L), L, gate_rows);
            };
            run_o_chunks(o_proj, chunks.count, common_bufs->x, common_bufs->ab.buf, proj_weights, prep, *post);
            MINICPM_OCR_DUMP("attn_gated", g.data(), L, desc->DQ);
            PREFILL_PROF_TICK("attn:gate+o_proj (chunks)");
            return buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        }
#endif
        if (common_bufs->mm_bfp16) {
            common_bufs->ab.publish(keep_g, L, gate_rows);
        }
        else {
            cpu_func::_elementwise_mul(g.data(), g_sigmoid.data(), common_bufs->after_mix.data(), L * desc->DQ);
            minicpm_ocr_cpu::scratch_push(g);
        }
        MINICPM_OCR_DUMP("attn_gated", g.data(), L, desc->DQ);
        PREFILL_PROF_TICK("attn:.gate_product+tiles");

        if (!_o_proj_npu(proj_weights)) {
            _o_proj_host(L, proj_weights);
        }
        MINICPM_OCR_DUMP("attn_o_proj", common_bufs->x.data(), L, desc->D);
#ifdef ENABLE_DEBUG
        _xref_o_proj(L, proj_weights);
#endif
        PREFILL_PROF_TICK("attn:o_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);
        return valid;
    }

    void fill_kv_cache(buffer<bf16>& buffer_k, buffer<bf16>& buffer_v, buffer<bf16>& cache, int L_begin, int L){
        tensor_2d<bf16> k_cache(buffer_k, desc->DK);
        tensor_2d<bf16> v_cache(buffer_v, desc->DV);

        // No pull first: this only WRITES the new rows, and a row is 1 KB (2 KV heads x 256
        // bf16), so it covers whole cache lines -- a stale host copy of the other rows is
        // never read, and never flushed back. Host attention, the one stage that reads the
        // cache, pulls it itself (_attend_host).
        const size_t k_first = (size_t)L_begin * desc->DK;
        const size_t v_first = (size_t)L_begin * desc->DV + cache.size() / 2;
        bf16* k_ptr = cache.data() + k_first;
        bf16* v_ptr = cache.data() + v_first;

        for (int l = 0; l < L; l++){
            memcpy(k_ptr, k_cache[l].data(), desc->DK * sizeof(bf16));
            memcpy(v_ptr, v_cache[l].data(), desc->DV * sizeof(bf16));
            k_ptr += desc->DK;
            v_ptr += desc->DV;
        }
        // flush only the rows written: a pull + push of the whole cache cost ~1.1 ms per
        // attention layer whatever the prompt length
        minicpm_ocr_cpu::sync_range_to_device(cache, k_first, (size_t)L * desc->DK);
        minicpm_ocr_cpu::sync_range_to_device(cache, v_first, (size_t)L * desc->DV);
    }
};

struct minicpm_ocr_mlp_prefill_context{
    minicpm_ocr_desc* desc;
    minicpm_ocr_npu_sequence* seq_gen;
    minicpm_ocr_common_buffers* common_bufs;
    npu_xclbin_manager* npu;

    int L_old = -1;
#ifdef ENABLE_DEBUG
    //!< set by minicpm_ocr_prefill_context::forward() before every call
    int layer_idx = -1;
    //!< non-null in a debug build; whether it scores anything is up to enabled(layer_idx)
    minicpm_ocr_prefill_xref* xref = nullptr;
#endif

#if MINICPM_OCR_MM_ON_NPU
    npu_app upgate_proj[minicpm_ocr_chunks::max_chunks];
    npu_app down_proj[minicpm_ocr_chunks::max_chunks];
#endif

    /// @brief [L, 2 * IS]: interleaved 512-row up/gate slabs, raw, from the array run; or
    ///        de-interleaved [ up | silu(gate) ] rows from the host (dequant_mm_upgate)
    buffer<bf16> upgate;
    buffer<bf16> hid;
    //!< mm_bfp16.xclbin: hid as the down projection's bfp16 x tiles
    minicpm_ocr_mm_bfp16::tiles hb;
    //!< common_bufs->glu_npu: hid as the up/gate run writes it, the down run's x (no upgate / hid / hb then)
    buffer<u8> hid_slots;

    minicpm_ocr_mlp_prefill_context(
        minicpm_ocr_desc* desc,
        minicpm_ocr_npu_sequence* seq_gen,
        minicpm_ocr_common_buffers* common_bufs,
        npu_xclbin_manager* npu,
        npu_app_manager* gemm_app_manager
    ) : desc(desc), seq_gen(seq_gen), common_bufs(common_bufs), npu(npu){
#if MINICPM_OCR_MM_ON_NPU
        for (int c = 0; c < minicpm_ocr_chunks::max_chunks; c++) {
            upgate_proj[c] = gemm_app_manager->create_app();
            down_proj[c] = gemm_app_manager->create_app();
        }
#endif
    }

    /// @brief grow the scratch and regenerate the array sequences for this L
    void reserve(int L_padded) {
#if MINICPM_OCR_MM_ON_NPU
        PREFILL_PROF_INIT();
        if (L_old != L_padded && common_bufs->glu_npu) {
            const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
            for (int c = 0; c < chunks.count; c++) {
                seq_gen->gen_dequant_mm_bfp16_glu(this->upgate_proj[c].seq(), chunks.padded[c], desc->D, desc->intermediate_size, desc->get_upgate_offset(), chunks.begin[c]);
                seq_gen->gen_dequant_mm_bfp16(this->down_proj[c].seq(), chunks.padded[c], desc->intermediate_size, desc->D, desc->get_down_offset(), (size_t)chunks.begin[c] * desc->D, 0, true, chunks.begin[c]);
            }
            PREFILL_PROF_TICK("reserve:mlp.mm_seq_gen");
            L_old = L_padded;
            hid_slots = npu->create_bo_buffer<u8>(minicpm_ocr_npu_sequence::glu_slots_bytes(L_padded, desc->intermediate_size));
            PREFILL_PROF_TICK("reserve:mlp.alloc");
            return;
        }
#endif
        if (L_old != L_padded){
#if MINICPM_OCR_MM_ON_NPU
            if (common_bufs->mm_bfp16) {
                const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
                for (int c = 0; c < chunks.count; c++) {
                    const size_t row = chunks.begin[c];
                    seq_gen->gen_dequant_mm_bfp16(this->upgate_proj[c].seq(), chunks.padded[c], desc->D, desc->intermediate_size * 2, desc->get_upgate_offset(), row * desc->intermediate_size * 2, 0, false, chunks.begin[c]);
                    seq_gen->gen_dequant_mm_bfp16(this->down_proj[c].seq(), chunks.padded[c], desc->intermediate_size, desc->D, desc->get_down_offset(), row * desc->D, 0, false, chunks.begin[c]);
                }
                hb.reserve(npu, L_padded, desc->intermediate_size);
            }
            else {
                seq_gen->gen_dequant_mm_512(this->upgate_proj[0].seq(), L_padded, desc->D, desc->intermediate_size * 2, desc->get_upgate_offset(), 0, 0);
                seq_gen->gen_dequant_mm_512(this->down_proj[0].seq(), L_padded, desc->intermediate_size, desc->D, desc->get_down_offset(), 0, 0);
            }
#endif
            L_old = L_padded;
            upgate = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->intermediate_size * 2);
            hid = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->intermediate_size);
            memset(hid.data(), 0, hid.size() * sizeof(bf16));
            minicpm_ocr_cpu::scratch_push(hid);
        }
    }

    /// @brief up/gate projection and the glu, into hid
    bool _upgate_npu(int L, buffer<u8>& proj_weights) {
#if MINICPM_OCR_MM_ON_NPU
        if (common_bufs->glu_npu) {
            // no host step: the run writes the down run's x itself
            this->upgate_proj[0](hid_slots, common_bufs->xb.buf, proj_weights);
            return true;
        }
        if (common_bufs->mm_bfp16) {
            const int IS = desc->intermediate_size;
            this->upgate_proj[0](upgate, common_bufs->xb.buf, proj_weights);
            minicpm_ocr_cpu::sync_range_from_device(upgate, 0, (size_t)L * IS * 2);
            hb.publish(minicpm_ocr_mm_bfp16::debug_keeps_bf16 ? &hid : nullptr, L, [&](int row, int n, bf16* out){
                for (int r = 0; r < n; r++) {
                    cpu_func::_exp_glu_row(out + (size_t)r * IS, upgate.data() + (size_t)(row + r) * IS * 2, IS, 512);
                }
            });
            return true;
        }
        this->upgate_proj[0](upgate, this->common_bufs->x, proj_weights);
        upgate.sync_from_device();

        // no pull of hid: the array never writes it (down_proj only reads it), so the host
        // copy is always the current one, and _exp_glu overwrites rows [0, L) anyway
        cpu_func::_exp_glu(hid.data(), upgate.data(), L, desc->intermediate_size, 512);

        hid.sync_to_device();
        return true;
#else
        (void)L; (void)proj_weights;
        return false;
#endif
    }

    void _upgate_host(int L, buffer<u8>& proj_weights) {
        const int IS = desc->intermediate_size;
        minicpm_ocr_cpu::dequant_mm_upgate(upgate.data(), proj_weights.data() + desc->get_upgate_offset(),
                                         common_bufs->x.data(), L, desc->D, IS, 512);
        MINICPM_OCR_DUMP("mlp_up", upgate.data(), L, IS, 2 * IS);
        MINICPM_OCR_DUMP("mlp_gate", upgate.data() + IS, L, IS, 2 * IS);
        minicpm_ocr_cpu::glu_mul(hid.data(), upgate.data(), L, IS);
        minicpm_ocr_cpu::scratch_push(upgate);
        minicpm_ocr_cpu::scratch_push(hid);
    }

    /// @brief down projection of hid into common x
    bool _down_npu(buffer<u8>& proj_weights) {
#if MINICPM_OCR_MM_ON_NPU
        PREFILL_PROF_INIT();
        if (common_bufs->glu_npu) {
            this->down_proj[0](common_bufs->x, hid_slots, proj_weights);
        }
        else {
            this->down_proj[0](common_bufs->x, common_bufs->mm_in(hid, hb), proj_weights);
        }
        PREFILL_PROF_TICK("mlp:.down_run");

        common_bufs->x.sync_from_device();
        PREFILL_PROF_TICK("mlp:.x_sync_from");
        return true;
#else
        (void)proj_weights;
        return false;
#endif
    }

    void _down_host(int L, buffer<u8>& proj_weights) {
        minicpm_ocr_cpu::dequant_mm(common_bufs->x.data(), desc->D,
                                  proj_weights.data() + desc->get_down_offset(),
                                  hid.data(), desc->intermediate_size, L, desc->intermediate_size, desc->D,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_cpu::scratch_push(common_bufs->x);
    }

#ifdef ENABLE_DEBUG
    /// \brief score the up/gate projection and the GLU that follows it
    /// \note The two backends genuinely differ here, and this is the only check that sees
    ///       it: the array leaves `upgate` in the interleaved 512-row slab layout the
    ///       weights are stored in and cpu_func::_exp_glu walks that directly, while the
    ///       host path de-interleaves first (dequant_mm_upgate) and then glu_mul's. So
    ///       `mlp_upgate` scores the matmul alone and `mlp_glu` scores matmul + GLU.
    void _xref_upgate(int L, buffer<u8>& proj_weights) {
#if MINICPM_OCR_MM_ON_NPU
        // with the GLU on the array there is no upgate / hid on the host to score
        if (xref == nullptr || !xref->enabled(layer_idx) || common_bufs->glu_npu) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        const int IS = desc->intermediate_size;

        std::vector<bf16> want((size_t)rows * IS * 2);
        minicpm_ocr_cpu::dequant_mm(want.data(), IS * 2, proj_weights.data() + desc->get_upgate_offset(),
                                  common_bufs->x.data(), desc->D, rows, desc->D, IS * 2,
                                  minicpm_ocr_cpu::BODY_LAYOUT);
        {
            minicpm_ocr_prefill_xref::guard gu(xref, "mlp_upgate", layer_idx, upgate.data(), IS * 2, IS * 2, rows);
            xref->check("mlp_upgate", layer_idx, upgate.data(), want.data(), IS * 2, IS * 2, rows);
        }

        std::vector<bf16> want_ug((size_t)rows * IS * 2), want_hid((size_t)rows * IS);
        minicpm_ocr_cpu::dequant_mm_upgate(want_ug.data(), proj_weights.data() + desc->get_upgate_offset(),
                                         common_bufs->x.data(), rows, desc->D, IS, 512);
        minicpm_ocr_cpu::glu_mul(want_hid.data(), want_ug.data(), rows, IS);
        minicpm_ocr_prefill_xref::guard gu(xref, "mlp_glu", layer_idx, hid.data(), IS, IS, rows);
        xref->check("mlp_glu", layer_idx, hid.data(), want_hid.data(), IS, IS, rows);
#else
        (void)L; (void)proj_weights;
#endif
    }

    /// \brief score the down projection
    /// \param got the down output; null: common x (the row-chunk path passes its copy, x is overwritten by then)
    void _xref_down(int L, buffer<u8>& proj_weights, const bf16* got = nullptr) {
#if MINICPM_OCR_MM_ON_NPU
        if (xref == nullptr || !xref->enabled(layer_idx) || common_bufs->glu_npu) { return; }
        const int rows = xref->rows(L);
        if (rows <= 0) { return; }
        if (got == nullptr) { got = common_bufs->x.data(); }
        std::vector<bf16> want((size_t)rows * desc->D);
        minicpm_ocr_cpu::dequant_mm(want.data(), desc->D, proj_weights.data() + desc->get_down_offset(),
                                  hid.data(), desc->intermediate_size, rows,
                                  desc->intermediate_size, desc->D, minicpm_ocr_cpu::BODY_LAYOUT);
        minicpm_ocr_prefill_xref::guard gu(xref, "mlp_down", layer_idx, got, desc->D, desc->D, rows);
        xref->check("mlp_down", layer_idx, got, want.data(), desc->D, desc->D, rows);
#else
        (void)L; (void)proj_weights; (void)got;
#endif
    }
#endif // ENABLE_DEBUG

    buffer<bf16> forward(int L, buffer<u8>& proj_weights, buffer<bf16>& rms_weights) {
        PREFILL_PROF_INIT();
        int L_padded = (L + 511) / 512 * 512;
        this->reserve(L_padded);
        PREFILL_PROF_TICK("mlp:seq_gen+alloc");

        if (!_upgate_npu(L, proj_weights)) {
            _upgate_host(L, proj_weights);
        }
        PREFILL_PROF_TICK("mlp:upgate_proj+glu");
#ifdef ENABLE_DEBUG
        _xref_upgate(L, proj_weights);
#endif

        if (!_down_npu(proj_weights)) {
            _down_host(L, proj_weights);
        }
        MINICPM_OCR_DUMP("mlp_down", common_bufs->x.data(), L, desc->D);
#ifdef ENABLE_DEBUG
        _xref_down(L, proj_weights);
#endif

        PREFILL_PROF_TICK("mlp:down_proj");

        buffer<bf16> valid = buffer<bf16>(common_bufs->x.data(), (size_t)L * desc->D);

        return valid;
    }

#if MINICPM_OCR_MM_ON_NPU
    /// @brief the host GLU (FLM_MINICPM_OCR_GLU_NPU=0) of chunk c's rows, into the down run's tiles
    void _glu_chunk(int c, int L) {
        const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
        const int IS = desc->intermediate_size;
        const int row_begin = chunks.begin[c];
        const int rows = chunks.rows(c, L);
        minicpm_ocr_cpu::sync_range_from_device(upgate, (size_t)row_begin * IS * 2, (size_t)rows * IS * 2);
        hb.publish_rows(minicpm_ocr_mm_bfp16::debug_keeps_bf16 ? &hid : nullptr, row_begin, rows, L, [&](int row, int n, bf16* out){
            for (int r = 0; r < n; r++) {
                cpu_func::_exp_glu_row(out + (size_t)r * IS, upgate.data() + (size_t)(row + r) * IS * 2, IS, 512);
            }
        });
    }

    /// @brief The MLP as one up/gate and one down run per row chunk (mm_bfp16.xclbin). Host work under the runs:
    ///        post_last() (the caller's post-attention norm of the last chunk) under the first up/gate run; the
    ///        host GLU of chunk c - 1 under the up/gate run of chunk c, the last one under the first down run;
    ///        tail(c - 1) (the caller's residual add and next input norm) under the down run of chunk c. tail of
    ///        the last chunk runs after it. With one chunk everything runs in order.
    void forward_chunked(int L, buffer<u8>& proj_weights, const std::function<void()>& post_last, const std::function<void(int)>& tail) {
        PREFILL_PROF_INIT();
        const int L_padded = (L + 511) / 512 * 512;
        this->reserve(L_padded);
        PREFILL_PROF_TICK("mlp:seq_gen+alloc");

        const minicpm_ocr_chunks::layout& chunks = common_bufs->chunks;
        const int count = chunks.count;
        const bool glu_host = !common_bufs->glu_npu;
        bytes& up_out = glu_host ? static_cast<bytes&>(upgate) : static_cast<bytes&>(hid_slots);
        bytes& down_in = glu_host ? static_cast<bytes&>(hb.buf) : static_cast<bytes&>(hid_slots);

        if (count == 1) {
            post_last();
        }
        for (int c = 0; c < count; c++) {
            auto run = upgate_proj[c].create_run(up_out, common_bufs->xb.buf, proj_weights);
            run.start();
            if (c == 0 && count > 1) {
                post_last();
            }
            if (glu_host && c > 0) {
                _glu_chunk(c - 1, L);
            }
            flm::npu::wait_for_completion(run, std::string("minicpm-ocr upgate chunk ") + std::to_string(c));
        }
        bool glu_done = false;
        if (glu_host && count == 1) {
            _glu_chunk(0, L);
            glu_done = true;
        }
#ifdef ENABLE_DEBUG
        // the cross-check reads every row of hid: make the last chunk's GLU now when it is on
        if (glu_host && !glu_done && xref != nullptr && xref->enabled(layer_idx)) {
            _glu_chunk(count - 1, L);
            glu_done = true;
        }
        _xref_upgate(L, proj_weights);
#endif
        PREFILL_PROF_TICK("mlp:upgate_proj (chunks)");

        for (int c = 0; c < count; c++) {
            auto run = down_proj[c].create_run(common_bufs->x, down_in, proj_weights);
            run.start();
            if (glu_host && !glu_done && c == 0) {
                _glu_chunk(count - 1, L);
                glu_done = true;
            }
            if (c > 0) {
                tail(c - 1);
            }
            flm::npu::wait_for_completion(run, std::string("minicpm-ocr down chunk ") + std::to_string(c));
        }
        tail(count - 1);
        PREFILL_PROF_TICK("mlp:down_proj+tails (chunks)");
    }
#endif
};

struct minicpm_ocr_prefill_context{
    minicpm_ocr_desc* desc;
    minicpm_ocr_npu_sequence* seq_gen;
    npu_xclbin_manager* npu;
    minicpm_ocr_common_buffers common_bufs;

    std::unique_ptr<minicpm_ocr_attn_block_prefill_context> attn_block;
    std::unique_ptr<minicpm_ocr_linear_block_prefill_context> linear_block;
    std::unique_ptr<minicpm_ocr_mlp_prefill_context> mlp;

    // for xclbins; null for every overlay whose stage is on the host
    npu_app_manager* gemm_app_manager = nullptr;
    npu_app_manager* mha_app_manager = nullptr;
    npu_app_manager* mha_bfp16_app_manager = nullptr;
    npu_app_manager * conv1d_app_manager = nullptr;
    npu_app_manager * gate_delta_net_manager = nullptr;

#ifdef ENABLE_DEBUG
    //!< the cpu/npu cross reference. Constructed in every debug build; whether it scores
    //!< anything is FLM_XREF_LAYERS / DEBUG_LEVEL. See minicpm_ocr_prefill_xref.hpp.
    std::unique_ptr<minicpm_ocr_prefill_xref> xref;
    //!< the row-chunk path's copies of x as the array wrote it (block output, down output): the host overwrites x
    //!< chunk by chunk before the dumps and cross-checks, which read every row, get to it
    std::vector<bf16> debug_block_out;
    std::vector<bf16> debug_mlp_out;
#endif
    //!< the layer whose input norm the previous layer's last step already made (row-chunk path), or -1
    int input_normed_layer = -1;
    //!< the MLP's and the attention block's reserve() for a new length, running on a second thread (_reserve_ahead)
    std::future<void> reserved_ahead;

    minicpm_ocr_prefill_context(){}
    minicpm_ocr_prefill_context(
        npu_xclbin_manager* npu,
        minicpm_ocr_desc* model_desc,
        minicpm_ocr_npu_sequence* seq_gen,
        LM_Config& config
    ) : desc(model_desc), seq_gen(seq_gen), npu(npu) {
#if MINICPM_OCR_MM_ON_NPU
        // mm_bfp16_glu.xclbin, else mm_bfp16.xclbin (FLM_Xclbin/minicpm_ocr/dequant_mm_prefill) when it is in the
        // xclbin folder; FLM_MINICPM_OCR_MM_BFP16=0 keeps mm.xclbin. The _glu build also has the GLU epilogue, used
        // unless FLM_MINICPM_OCR_GLU_NPU=0.
        {
            const std::string mm_glu = utils::path_join(config.exec_path, "xclbins", config.model_name, "mm_bfp16_glu.xclbin");
            const std::string mm_bfp16 = utils::path_join(config.exec_path, "xclbins", config.model_name, "mm_bfp16.xclbin");
            if (std::filesystem::exists(mm_glu) && minicpm_ocr_mm_bfp16::enabled_by_env()) {
                this->gemm_app_manager = npu->register_xclbin(mm_glu);
                this->common_bufs.mm_bfp16 = true;
                this->common_bufs.glu_npu = minicpm_ocr_mm_bfp16::glu_enabled_by_env();
            }
            else if (std::filesystem::exists(mm_bfp16) && minicpm_ocr_mm_bfp16::enabled_by_env()) {
                this->gemm_app_manager = npu->register_xclbin(mm_bfp16);
                this->common_bufs.mm_bfp16 = true;
            }
            else {
                this->gemm_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "mm.xclbin"));
            }
        }
#endif
#if MINICPM_OCR_CONV_ON_NPU
        this->conv1d_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "conv8.xclbin"));
#endif
#if MINICPM_OCR_GDN_ON_NPU
        this->gate_delta_net_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "GateDeltaNet_prefill.xclbin"));
#endif
#if MINICPM_OCR_ATTN_ON_NPU
        // attn_bfp16.xclbin (FLM_Xclbin/minicpm_ocr/gated_attention_prefill) when it is in the xclbin folder;
        // FLM_MINICPM_OCR_ATTN_BFP16=0 keeps attn.xclbin
        {
            const std::string attn_bfp16 = utils::path_join(config.exec_path, "xclbins", config.model_name, "attn_bfp16.xclbin");
            const char* env = std::getenv("FLM_MINICPM_OCR_ATTN_BFP16");
            if (std::filesystem::exists(attn_bfp16) && !(env && env[0] == '0')) {
                this->mha_bfp16_app_manager = npu->register_xclbin(attn_bfp16);
            }
            else {
                this->mha_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "attn.xclbin"));
            }
        }
#endif

        // this is to ensure they have same out projection
        assert(desc->Linear_DV == desc->DQ);

        this->linear_block = std::make_unique<minicpm_ocr_linear_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            npu,
            gemm_app_manager,
            conv1d_app_manager,
            gate_delta_net_manager
        );

        this->attn_block = std::make_unique<minicpm_ocr_attn_block_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            npu,
            gemm_app_manager,
            mha_app_manager,
            mha_bfp16_app_manager
        );

        this->mlp = std::make_unique<minicpm_ocr_mlp_prefill_context>(
            desc,
            seq_gen,
            &common_bufs,
            npu,
            gemm_app_manager
        );

#ifdef ENABLE_DEBUG
        // layer 0 is a linear (gated delta net) layer and attn_idx is the first full
        // attention one, so the default set covers both layer kinds plus the deepest layer
        this->xref = std::make_unique<minicpm_ocr_prefill_xref>((int)desc->attn_idx,
                                                             (int)desc->num_hidden_layers - 1);
        this->linear_block->xref = this->xref.get();
        this->attn_block->xref   = this->xref.get();
        this->mlp->xref          = this->xref.get();
#endif
    }

    /// @param next_rms_weights the next layer's norm weights (null for the last layer): the row-chunk path makes the
    ///        next layer's input norm in this layer's last step, chunk by chunk under the down runs
    buffer<bf16> forward(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& all_proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights, buffer<bf16>& kv_cache, std::vector<int>& input_idx, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu, int L_begin = 0, buffer<bf16>* next_rms_weights = nullptr){
        PREFILL_PROF_INIT();
        minicpm_ocr_cpu::dumper().layer = layer_idx;
        this->attn_block->layer_idx   = layer_idx;
#ifdef ENABLE_DEBUG
        this->linear_block->layer_idx = layer_idx;
        this->mlp->layer_idx          = layer_idx;
#endif
        int L_padded = (L + 511) / 512 * 512;
        size_t x_size = (size_t)L_padded * desc->D;

        common_bufs.chunks.setup(L_padded, common_bufs.mm_bfp16 ? minicpm_ocr_chunks::setting() : 1);
        if (x_size != common_bufs.x.size()) {
            common_bufs.x = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->D);
            common_bufs.after_mix = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->Linear_DV);
            common_bufs.gate = minicpm_ocr_cpu::scratch_alloc<bf16>(npu, (size_t)L_padded * desc->Linear_DV);
            PREFILL_PROF_TICK("layer:.common_alloc");
        }
        if (layer_idx == 0) {
            input_normed_layer = -1;
        }
#if MINICPM_OCR_MM_ON_NPU
        if (common_bufs.mm_bfp16) {
            common_bufs.xb.reserve(npu, L_padded, desc->D);
            common_bufs.ab.reserve(npu, L_padded, desc->Linear_DV);
            PREFILL_PROF_TICK("layer:.tiles_reserve");
            return _forward_chunked(layer_idx, residual, L, all_proj_weights, rms_weights, rope_weights, kv_cache, input_idx, alpha_proj_cpu, beta_proj_cpu, L_begin, next_rms_weights);
        }
#else
        (void)next_rms_weights;
#endif
        memset(common_bufs.x.data(), 0, common_bufs.x.size() * sizeof(bf16));

        // for all layers , first step is store a residual and do input layer norm
        // this step do the rms norm and store it to x, the residual is kept
        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data(), desc->D, L, 0, 0);
        minicpm_ocr_cpu::scratch_push(common_bufs.x);
        MINICPM_OCR_DUMP("input_layernorm", common_bufs.x.data(), L, desc->D);
        buffer<bf16> block_output;
        PREFILL_PROF_TICK("layer:input_rms_norm");
        if (desc->is_attn_layer(layer_idx)) {
            block_output = this->attn_block->forward(L, all_proj_weights, rope_weights, kv_cache, input_idx, L_begin);
            PREFILL_PROF_TICK("layer:attn_block_total");
        }
        else {
            block_output = this->linear_block->forward(L, all_proj_weights, rope_weights, kv_cache, alpha_proj_cpu, beta_proj_cpu);
            PREFILL_PROF_TICK("layer:linear_block_total");
        }

        cpu_func::_residual_add(residual, block_output, residual, residual.size());
        PREFILL_PROF_TICK("layer:residual_add_block");

        cpu_func::_rms_norm(common_bufs.x, residual, rms_weights.data() + desc->D, desc->D, L, 0, 0);
        MINICPM_OCR_DUMP("post_attention_layernorm", common_bufs.x.data(), L, desc->D);

        minicpm_ocr_cpu::scratch_push(common_bufs.x);
        PREFILL_PROF_TICK("layer:post_attn_rms_norm");

        auto mlp_out = mlp->forward(L, all_proj_weights, rms_weights);
        PREFILL_PROF_TICK("layer:mlp_block_total");

        cpu_func::_residual_add(residual, mlp_out, residual, residual.size());
        PREFILL_PROF_TICK("layer:residual_add_mlp");
        MINICPM_OCR_DUMP("residual_out", residual.data(), L, desc->D);
        minicpm_ocr_cpu::dumper().layer = -1;

#ifdef ENABLE_DEBUG
        // one pass/fail line after the last layer: at FLM_XREF_LAYERS=all the per tensor
        // report is hundreds of lines and the question being asked is a single bit
        if (layer_idx == (int)desc->num_hidden_layers - 1) { this->xref->summary(); }
#endif

        return residual;
    }

#if MINICPM_OCR_MM_ON_NPU
    /// @brief A call with a new padded length makes new scratch BOs and array sequences, ~0.1 ms per MB of BO
    ///        (8K: ~35 ms for the MLP and the attention block). Their reserve() runs here on a second thread while
    ///        layer 0's block runs (started after that block's own reserve(): two threads allocating at once were
    ///        slower), and _wait_reserved() joins it before either block is used. The two
    ///        reserves touch only their own block's buffers and sequences; the shared sequence generator is
    ///        read-only and each BO allocation is its own XRT call. The thread also sizes the attention layers'
    ///        bfp16 KV copies (only the attention block uses them, after the join).
    void _reserve_ahead(int L, int L_padded, int L_begin) {
        const bool kv_short = attn_block->kv_copies_short(L, L_padded, L_begin);
        if (mlp->L_old == L_padded && attn_block->L_padded_old == L_padded && !kv_short) {
            return;
        }
#if defined(FLM_USE_HRX)
        mlp->reserve(L_padded);
        attn_block->reserve(L, L_padded, L_begin);
        attn_block->reserve_kv_copies(L, L_padded, L_begin);
#else
        reserved_ahead = std::async(std::launch::async, [this, L, L_padded, L_begin]{
            mlp->reserve(L_padded);
            attn_block->reserve(L, L_padded, L_begin);
            attn_block->reserve_kv_copies(L, L_padded, L_begin);
        });
#endif
    }

    void _wait_reserved() {
        if (reserved_ahead.valid()) {
            reserved_ahead.get();
        }
    }

    /// @brief One layer on mm_bfp16.xclbin, as one host/array pipeline over the row chunks (prefill_chunks.hpp):
    ///
    ///        array:  o(c0)    o(c1)      up(c0)    up(c1)   down(c0)  down(c1)  | next layer's in_proj ...
    ///        host:   prep(c1) post(c0)   post(c1)                     tail(c0)  | tail(c1)
    ///
    ///        prep: the gated norm or the gate product (the block's o_proj input); post: residual add + post-attention
    ///        norm; tail: residual add + the NEXT layer's input norm (so the next layer starts with it done). Each
    ///        residual add and the norm after it are one pass per row (_residual_add_rms_norm_row). Same arithmetic
    ///        per row as one layer at a time, so the output is bit-identical. README.md "Host prefill work (WP4)".
    buffer<bf16> _forward_chunked(int layer_idx, buffer<bf16>& residual, int L, buffer<u8>& all_proj_weights, buffer<bf16>& rms_weights, buffer<bf16>& rope_weights, buffer<bf16>& kv_cache, std::vector<int>& input_idx, buffer<bf16>& alpha_proj_cpu, buffer<bf16>& beta_proj_cpu, int L_begin, buffer<bf16>* next_rms_weights) {
        PREFILL_PROF_INIT();
        const int D = desc->D;
        const minicpm_ocr_chunks::layout& chunks = common_bufs.chunks;
        const bool attn = desc->is_attn_layer(layer_idx);
        constexpr bool debug_keeps = minicpm_ocr_mm_bfp16::debug_keeps_bf16;

        if (input_normed_layer != layer_idx) {
            // the linear block's alpha / beta read the normed x on the host
            const bool keep = !attn || debug_keeps;
            const bf16* w = rms_weights.data();
            common_bufs.xb.publish(keep ? &common_bufs.x : nullptr, L, [&](int row, int n, bf16* out){
                for (int r = 0; r < n; r++) {
                    cpu_func::_rms_norm_row(out + (size_t)r * D, residual.data() + (size_t)(row + r) * D, w, D);
                }
            });
        }
        input_normed_layer = -1;
        MINICPM_OCR_DUMP("input_layernorm", common_bufs.x.data(), L, D);
        PREFILL_PROF_TICK("layer:input_rms_norm");

#ifdef ENABLE_DEBUG
        debug_block_out.resize((size_t)L * D);
        debug_mlp_out.resize((size_t)L * D);
#endif
        // chunk c's rows of x as the array just wrote them (block output or down output)
        auto pull_x = [&](int c, bool block_out){
            const size_t row = chunks.begin[c];
            const size_t rows = chunks.rows(c, L);
            minicpm_ocr_cpu::sync_range_from_device(common_bufs.x, row * D, rows * D);
#ifdef ENABLE_DEBUG
            std::vector<bf16>& copy = block_out ? debug_block_out : debug_mlp_out;
            memcpy(copy.data() + row * D, common_bufs.x.data() + row * D, rows * D * sizeof(bf16));
#else
            (void)block_out;
#endif
        };
        // residual += x, then norm(residual) by w into the next up/gate (or in_proj) run's tiles; keep: also as bf16 x
        auto add_and_norm = [&](int c, const bf16* w, bool keep){
            common_bufs.xb.publish_rows(keep ? &common_bufs.x : nullptr, chunks.begin[c], chunks.rows(c, L), L, [&](int row, int n, bf16* out){
                for (int r = 0; r < n; r++) {
                    cpu_func::_residual_add_rms_norm_row(residual.data() + (size_t)(row + r) * D, common_bufs.x.data() + (size_t)(row + r) * D, out + (size_t)r * D, w, D);
                }
            });
        };

        const std::function<void(int)> post = [&](int c){
            pull_x(c, true);
            add_and_norm(c, rms_weights.data() + D, debug_keeps);
        };
        const bool next_is_attn = next_rms_weights != nullptr && desc->is_attn_layer(layer_idx + 1);
        const std::function<void(int)> tail = [&](int c){
            pull_x(c, false);
            if (next_rms_weights != nullptr) {
                add_and_norm(c, next_rms_weights->data(), !next_is_attn || debug_keeps);
                return;
            }
            const int row_begin = chunks.begin[c];
            const int row_end = row_begin + chunks.rows(c, L);
            #pragma omp parallel for num_threads(cpu_func::host_threads()) schedule(static)
            for (int row = row_begin; row < row_end; row++) {
                cpu_func::_residual_add_row(residual.data() + (size_t)row * D, common_bufs.x.data() + (size_t)row * D, D);
            }
        };

        if (layer_idx == 0) {
            // layer 0's own block first, so the second thread's allocations overlap its array runs, not its allocations
            const int L_padded = (L + 511) / 512 * 512;
            if (attn) {
                this->attn_block->reserve(L, L_padded, L_begin);
            }
            else {
                this->linear_block->reserve(L, L_padded);
            }
            PREFILL_PROF_TICK("layer:.layer0_block_reserve");
            _reserve_ahead(L, L_padded, L_begin);
        }
        if (attn) {
            _wait_reserved();
            this->attn_block->forward(L, all_proj_weights, rope_weights, kv_cache, input_idx, L_begin, &post);
            PREFILL_PROF_TICK("layer:attn_block_total");
        }
        else {
            this->linear_block->forward(L, all_proj_weights, rope_weights, kv_cache, alpha_proj_cpu, beta_proj_cpu, &post);
            PREFILL_PROF_TICK("layer:linear_block_total");
        }

        const std::function<void()> post_last = [&]{
            post(chunks.count - 1);
#ifdef ENABLE_DEBUG
            MINICPM_OCR_DUMP(attn ? "attn_o_proj" : "linear_out", debug_block_out.data(), L, D);
            if (attn) {
                this->attn_block->_xref_o_proj(L, all_proj_weights, debug_block_out.data());
            }
            else {
                this->linear_block->_xref_o_proj(L, all_proj_weights, debug_block_out.data());
            }
#endif
            MINICPM_OCR_DUMP("post_attention_layernorm", common_bufs.x.data(), L, D);
        };
        _wait_reserved();
        PREFILL_PROF_TICK("layer:.wait_reserved_ahead");
        mlp->forward_chunked(L, all_proj_weights, post_last, tail);
        PREFILL_PROF_TICK("layer:mlp_block_total");
#ifdef ENABLE_DEBUG
        MINICPM_OCR_DUMP("mlp_down", debug_mlp_out.data(), L, D);
        mlp->_xref_down(L, all_proj_weights, debug_mlp_out.data());
#endif
        MINICPM_OCR_DUMP("residual_out", residual.data(), L, D);
        if (next_rms_weights != nullptr) {
            input_normed_layer = layer_idx + 1;
        }
        minicpm_ocr_cpu::dumper().layer = -1;

#ifdef ENABLE_DEBUG
        if (layer_idx == (int)desc->num_hidden_layers - 1) { this->xref->summary(); }
#endif
        return residual;
    }
#endif // MINICPM_OCR_MM_ON_NPU
};

#endif
