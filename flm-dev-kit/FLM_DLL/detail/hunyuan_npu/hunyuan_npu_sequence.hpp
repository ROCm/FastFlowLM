// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_npu_sequence.hpp
/// \brief npu instruction sequence generation for the hunyuan-dense engine
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \note Two overlays, two bitstreams:
///
///       * layer.xclbin -- decode and lm_head, told apart by the is_attn RTP
///         (1 = decode layer, 0 = final norm + lm_head matvec). Built from
///         FLM_Xclbin/Hy_MT2_1.8B/decoding with MODEL_TYPE=HUNYUAN_DENSE_1_8B.
///       * prefill_overlay.xclbin -- prefill dequant_mm and attention taking
///         turns on the array. Built from FLM_Xclbin/Hy_MT2_1.8B/prefill_overlay;
///         its sequences are in hunyuan_overlay.hpp.
///
///       See hunyuan_prefill.hpp for the HUNYUAN_*_ON_NPU switches.
///
///       The tile maps and RTP addresses below are taken from the built decode
///       overlay's rtp_addresses.txt and are not inherited from another model.
///       Do not copy them to a differently-shaped model.
#ifndef __HUNYUAN_NPU_SEQUENCE_HPP__
#define __HUNYUAN_NPU_SEQUENCE_HPP__

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "hunyuan_model_def.hpp"

struct hunyuan_npu_sequence{
    static constexpr npu_tiles xyr_tile        = IT2;
    static constexpr npu_tiles logits_out_tile = IT3;
    /// @brief the two shims the rope tile splits K (attn_tiles[0]) and V
    ///        (attn_tiles[1]) across; see iron/layer_aie.py rope_core.KV_out.
    static constexpr npu_tiles attn_tiles[2]   = {IT3, IT4};
    /// @brief the newest (bf16) K and V chunk of both columns, on its MM2S 0 / 1
    static constexpr npu_tiles v_tail_tile     = IT5;
    static constexpr npu_tiles mvm_tiles[4]    = {IT0, IT1, IT6, IT7};
    static constexpr npu_tiles attn_qk_tiles[] = {CT03, CT23, CT04, CT24};
    static constexpr npu_tiles attn_kv_tiles[] = {CT13, CT33, CT14, CT34};
    static constexpr npu_tiles proj_tiles[]    = {CT00, CT10, CT20, CT30,
                                                  CT01, CT11, CT21, CT31,
                                                  CT06, CT16, CT26, CT36,
                                                  CT07, CT17, CT27, CT37};
    static constexpr npu_tiles rms_tile        = CT02;

    static constexpr int xy_arg_idx          = 0;
    static constexpr int w_proj_arg_idx      = 1;
    static constexpr int w_rms_arg_idx       = 2;
    static constexpr int w_rope_rms_arg_idx  = 3;
    static constexpr int kv_cache_arg_idx    = 4;

    /// @brief the kv-cache row granularity the attention tiles consume
    static constexpr int L_CHUNK = 16;
    /// @brief q4nx block geometry: a 32 x 256 block of weights per core round
    static constexpr int m = 32;
    static constexpr int k = 256;

    // --- RTP addresses, read off the built overlay ---------------------------
    static constexpr uint32_t rms_addr     = 53248;  // RTP_RMS_IS_ATTN_BUFFER
    static constexpr uint32_t proj_addr    = 33280;  // RTP_PROJ_IS_ATTN_BUFFER_*
    static constexpr uint32_t l_qk_address = 3328;   // RTP_L_attn_qk_core_*: [0] L, [1] bfp16 K chunks
    static constexpr uint32_t l_kv_address = 50176;  // RTP_L_attn_kv_core_*: [0] L, [1] bfp16 V chunks
    /// @brief lock id 6 on both the proj and the rms cores is `rtp_available`,
    ///        initialised to 0 and released by the host once the RTPs are set
    ///        (iron/proj_tile.py, iron/rms_residual.py).
    static constexpr size_t   CT_lock_address_base = 0x000001F000;
    static constexpr int      rtp_available_lock_id = 6;

    // --- prefill overlay geometry -------------------------------------------
    /// @brief L granularity both prefill_overlay ops need. dequant_mm computes
    ///        4 core rows x 32 L rows per round and attention 16 rows x 8 cores
    ///        per round, so both land on 128. Rounding to
    ///        anything coarser only wastes compute on the padding rows.
    static constexpr uint32_t PREFILL_L_GRAIN = 128;

    /// @brief the single place prefill decides how far to pad L
    static uint32_t pad_prefill_L(uint32_t L){
        return (L + PREFILL_L_GRAIN - 1) / PREFILL_L_GRAIN * PREFILL_L_GRAIN;
    }

    /// @brief rows the prefill matmuls compute: 32 or 64 for short prompts (the
    ///        overlay's split plans, see "Small-row plans" in README.md), else
    ///        pad_prefill_L. Attention and every buffer keep pad_prefill_L rows.
    static uint32_t pad_prefill_gemm_L(uint32_t L, bool small_rows){
        if (small_rows && L <= 32) return 32;
        if (small_rows && L <= 64) return 64;
        return pad_prefill_L(L);
    }

    /// @brief Dout columns one dequant_mm half covers per round (desired_n * 4). A
    ///        half shares one activation RTP, so this is also the finest grain at
    ///        which the on-chip silu can be switched on and off.
    static constexpr uint32_t PREFILL_UNIT_N = 512;

    /// @brief shared model-shape definition, owned by hunyuan_npu::Impl
    const hunyuan_desc* desc = nullptr;

    uint32_t MAX_L = 0;
    uint32_t kv_cache_size = 0;

    /// @brief the array's four-way split kv-cache layout: K for kv heads 0..1,
    ///        K for kv heads 2..3, then the same split for V. Each quadrant is
    ///        row-major [MAX_L x DK/2]. The host prefill writes the same layout
    ///        (hunyuan_prefill.hpp::fill_kv_cache).
    size_t K03_offset = 0;
    size_t K47_offset = 0;
    size_t V03_offset = 0;
    size_t V47_offset = 0;
    /// @brief bfp16 copy of the finished 16-key chunks, after the bf16 cache in the same BO:
    ///        K blocks along dh, V as V^T tiles; one block per column, [chunk][kv head (2)][2304 B].
    ///        In bf16 elements. See "bfp16 KV copy" in README.md.
    size_t Kb03_offset = 0;
    size_t Kb47_offset = 0;
    size_t Vb03_offset = 0;
    size_t Vb47_offset = 0;
    /// @brief bf16 elements of the whole per-layer kv BO (bf16 cache + bfp16 K and V)
    size_t kv_buffer_size = 0;

    /// @brief bytes of one bfp16 V^T chunk of one kv head
    uint32_t v_bfp16_head_chunk_bytes() const { return L_CHUNK * desc->DH / 8 * 9; }
    /// @brief finished chunks the decode step at context length L reads as bfp16; the
    ///        chunk holding row L - 1 (the new token) always goes as bf16 (K and V)
    static uint32_t v_bfp16_chunks(uint32_t L){ return (L - 1) / L_CHUNK; }

    hunyuan_npu_sequence(){}
    hunyuan_npu_sequence(hunyuan_desc& desc, LM_Config config, uint32_t MAX_L);
    ~hunyuan_npu_sequence();

    /// @brief the whole decode-side layer (proj + rope + qk norm + mha + mlp)
    void gen_layer_seq(npu_sequence* seq, const uint32_t L);

    /// @brief set required rtps for decoding
    void gen_layer_rtp_seq(npu_sequence* seq, const uint32_t L);

    /// @brief does the prefill attention's kv read (hunyuan_overlay::gen_attn_run)
    ///        stay inside the cache buffer?
    /// @note  Each round reads kv rows [kv_begin, kv_end), where kv_end rounds the
    ///        round's newest query row up to a whole PREFILL_L_GRAIN block -- the
    ///        same expression the generator uses, kept in sync here on purpose.
    ///        When L_begin is block-aligned that round-up is exact and kv_end is
    ///        precisely L_begin + L_wanted, so a prefill that ends flush with the
    ///        cache still fits. Only an unaligned L_begin overshoots, and then by
    ///        at most PREFILL_L_GRAIN - 1 rows; those rows are zero and causally
    ///        masked, so reading them is harmless -- but only while they are still
    ///        inside the buffer.
    bool mha_engine_fits(const uint32_t L_begin, const uint32_t L_wanted) const {
        if (L_wanted == 0) return true;
        // Lq_current of the final round, then gen_attn_run's kv_end
        const size_t last_q = (size_t)L_begin + L_wanted - PREFILL_L_GRAIN;
        const size_t kv_end = (last_q + 2 * PREFILL_L_GRAIN - 1)
                            / PREFILL_L_GRAIN * PREFILL_L_GRAIN;
        return kv_end <= (size_t)MAX_L;
    }

    /// @brief final norm + lm_head matvec, on the same overlay as gen_layer_seq
    void gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D);

    void set_max_length(const uint32_t MAX_L);

    /// @brief V lives in the upper half of the per-layer kv cache buffer
    size_t get_V_offset() const { return this->kv_cache_size / 2; }

    size_t get_k03_offset() const { return this->K03_offset; }
    size_t get_k47_offset() const { return this->K47_offset; }
    size_t get_v03_offset() const { return this->V03_offset; }
    size_t get_v47_offset() const { return this->V47_offset; }

    // --- decode sequence building blocks ------------------------------------
    void _send_x(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight, const size_t chunk_bytes = 0);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
};

#endif // __HUNYUAN_NPU_SEQUENCE_HPP__
