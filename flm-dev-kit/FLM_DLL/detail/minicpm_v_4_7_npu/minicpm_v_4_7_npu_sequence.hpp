// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#pragma once

#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/minicpm_v_4_7/minicpm_v_4_7_npu.hpp"
#include "minicpm_v_model_def.hpp"

struct minicpm_v_4_7_npu_sequence{
    typedef enum: int{
        NORMAL_DEQUANT = 0,
        UP_MATRIX      = 1,
        GATE_MATRIX    = 2
    } dequant_output_mode_t;
    /// @brief constexprs
    static constexpr npu_tiles xyr_tile = IT2;
    static constexpr npu_tiles logits_out_tile = IT3;
    static constexpr npu_tiles attn_tile = IT3;
    static constexpr npu_tiles mvm_tiles[4] = {IT0, IT1, IT6, IT7};
    static constexpr npu_tiles attn_qk_tiles[] = {CT03, CT23};
    static constexpr npu_tiles attn_kv_tiles[] = {CT13, CT33};
    static constexpr npu_tiles proj_tiles[] = {CT00, CT10, CT20, CT30,
                                               CT01, CT11, CT21, CT31,
                                               CT06, CT16, CT26, CT36,
                                               CT07, CT17, CT27, CT37
                                            };
    static constexpr npu_tiles rms_tile = CT02;
    static constexpr npu_tiles gate_tile = CT22;
    /// the gate core (gate_tile above, hw row 4, column 2): in layer_lm_sl.xclbin it reads what kind of layer a run is from an RTP (kernels/gate.cc, built
    /// with STATELESS_X) instead of counting the layers of a period itself
    static constexpr uint32_t gate_mode_addr = 36864;
    static constexpr int gate_mode_linear = 0;
    static constexpr int gate_mode_attn = 1;

    static constexpr int xy_arg_idx = 0;
    static constexpr int w_proj_arg_idx = 1;
    static constexpr int w_rms_arg_idx = 2;
    static constexpr int w_rope_rms_arg_idx = 3;
    static constexpr int kv_cache_arg_idx = 4;
    // lm_head run inside the layer design (mode 2): x, lm_head weights, model.norm, (unused), logits
    static constexpr int logits_arg_idx = 4;
    static constexpr npu_tiles lm_head_logits_tile = IT1;

    static constexpr int L_CHUNK = 16;

    static constexpr uint32_t attn_qk_addr = 53248;
    static constexpr uint32_t attn_kv_addr = 53536;
    // Layer design with the lm_head phase (layer_lm.xclbin, FLM_Xclbin/MiniCPM_V/decoding built for QWEN3_5_08B): every run starts by
    // reading a mode RTP that the host writes and releases with lock 6. The old layer.xclbin has no such RTP, so these writes are only emitted when
    // lm_in_layer is set (the engine sets it iff it loaded layer_lm.xclbin).
    static constexpr uint32_t proj_mode_addr = 33280;
    static constexpr uint32_t ct_lock_base = 0x1F000;
    static constexpr int proj_rtp_lock_id = 6;
    static constexpr int proj_mode_linear = 0;
    static constexpr int proj_mode_attn = 1;
    static constexpr int proj_mode_lm_head = 2;
    static constexpr uint32_t rms_mode_addr = 36864;
    static constexpr int rms_mode_first_linear = 0;
    static constexpr int rms_mode_linear = 1;
    static constexpr int rms_mode_attn = 2;
    static constexpr int rms_mode_lm_head = 3;
    bool lm_in_layer = false;
    /// layer_lm_sl.xclbin (the design built with STATELESS_X, FLM_Xclbin/MiniCPM_V/decoding MODEL_TYPE=QWEN3_5_08B_SL): nothing but DDR
    /// carries state from one run to the next. Every layer run sends x in and receives the new x, and the attention run writes its context-length RTPs
    /// itself, so a firmware design reload between two runs (another process's command) cannot change the result. Implies lm_in_layer.
    bool stateless = false;
    // expert-MM RTP addresses; the CT kernel differs between even and odd rows.
    static constexpr uint32_t CT_rtp_address_even_row = 54784;
    static constexpr uint32_t act_rtp_address_even_row = 54848;
    static constexpr uint32_t CT_rtp_address_odd_row = 61696;
    static constexpr uint32_t act_rtp_address_odd_row = 61760;

    /// @brief shared model-shape definition, owned by minicpm_v_4_7_npu::Impl.
    ///        Dimensions such as D, DK, Linear_DQ, ... are read from here
    ///        instead of being duplicated in this class.
    const minicpm_v_desc* desc = nullptr;

    uint32_t MAX_L;
    uint32_t kv_cache_size;

    size_t linear_kv_cache_size;
    size_t conv_w_size_bf16;

    size_t s_offset;

    static constexpr uint32_t l_begin_mha_address = 61568;
    static constexpr uint32_t l_end_mha_address = 11904;
    static constexpr uint32_t window_size_address = 61600;
    /// RTP addresses of the bfp16 attention kernel (FLM_Xclbin/Qwen3_5/gated_attention_prefill)
    static constexpr uint32_t mha_bfp16_l_begin = 64256;
    static constexpr uint32_t mha_bfp16_l_end = 64288;
    static constexpr uint32_t mha_bfp16_window_size = 64320;

    void _gen_sequence(npu_sequence* seq, int L);
    void _send_hidden_states(npu_sequence* seq);
    void _send_rms_weights(npu_sequence* seq);
    void _send_rope_rms_weights(npu_sequence* seq);
    void _receive_kv_cache(npu_sequence* seq, const int L);
    void _move_kv_cache(npu_sequence* seq, const size_t L);
    void _move_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const weight_desc_t& weight);
    /// @brief decode weight-stream mode (FLM_MCV_WSTREAM, read once): 0 = the original round-by-round BDs, 1 = one strided BD per
    ///        column/channel per matrix, 2 (default) = as 1 plus the next matrix's BDs queued before the previous one's tokens are
    ///        waited for. Same bytes in the same order in every mode, so results are bit-identical.
    static int wstream_mode();
    /// @brief wait for the tokens of the weight matrix still in flight (mode 2), before a sequence ends
    void _flush_mvm(npu_sequence* seq);
    int mvm_pending_ = 0;
    int mvm_set_ = 0;
    void gen_lm_head_seq(npu_sequence* seq, int VOCAB_SIZE, int D);
    void _set_modes(npu_sequence* seq, int proj_mode, int rms_mode);
    /// @brief the lm_head as a run of the layer design: final RMS norm, then the Q8_0 head streamed through the projection cores (8 weight streams)
    void gen_lm_head_in_layer_seq(npu_sequence* seq);

    void _gen_linear_sequence(npu_sequence* seq, bool is_first = true);
    void _send_linear_conv_weights(npu_sequence* seq);
    void _receive_linear_kv_cache(npu_sequence* seq);
    void _move_linear_kv_cache(npu_sequence* seq);
    void _move_s(npu_sequence* seq);
    void _move_alpha_beta_weights(npu_sequence* seq, const size_t Dout, const size_t Din, const size_t w_offset);
    /// \param valid_rows rows of real data (0 = L). Only the runtime-m mm (mm_rtm = true) uses it: the dispatch
    ///        then covers ceil(valid_rows / 64) x 64 rows (spread over blocks of at most 512) instead of L.
    void gen_dequant_mm_512(npu_sequence* seq, const uint32_t L, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const uint32_t row_base = 0, const uint32_t valid_rows = 0);

    // ---- runtime-m mm (FLM_Xclbin/dequant_mm_bfp16_rtm2_glu_512x512x512, shipped as mm_rtm2_glu.xclbin) ----
    bool mm_rtm = false;   //!< the mm xclbin is the runtime-m design
    bool mm_yjoin_par = false;   //!< ... in its v2 form (mm_rtm2.xclbin): the memtile y join is parallel (2 x 4 producer credits, 2 consumer counters)
    bool mm_glu = false;   //!< ... and it is mm_rtm_glu.xclbin (rtp_act 3: fused silu(gate) * up epilogue; needs mm_rtm)
    struct mm_geometry { uint32_t blocks, m, covered; };   //!< `blocks` blocks of 4 x m rows; m a multiple of 16, <= 128
    static mm_geometry mm_pick_geometry(uint32_t rows);
    /// what a cached mm sequence depends on besides L_padded
    uint32_t mm_seq_key(uint32_t L_padded, uint32_t valid_rows) const { return mm_rtm ? L_padded * 256u + mm_pick_geometry(valid_rows).m : L_padded; }
    void gen_dequant_mm_rtm(npu_sequence* seq, const uint32_t rows, const uint32_t Din, const uint32_t Dout, const size_t weight_offset, const size_t out_offset, const int rtp_act, const uint32_t row_base);
    /// y descriptor set of the memtiles: the design's, the fused GLU one (rtp_act 3), or the design's again after one
    enum mm_y_mode : uint32_t { MM_Y_PLAIN = 0, MM_Y_GLU = 1, MM_Y_RESTORE = 2 };
    static void mm_rearm(npu_sequence* seq, const uint32_t m, const bool yjoin_par, const uint32_t ymode = MM_Y_PLAIN);
    static void mm_set_core_rows(npu_sequence* seq, const uint32_t m);
    inline size_t q8_0_size(size_t elems) { return elems + elems / 32 * 2;}

    minicpm_v_4_7_npu_sequence(){}
    minicpm_v_4_7_npu_sequence(minicpm_v_desc& desc, LM_Config config, uint32_t MAX_L);
    ~minicpm_v_4_7_npu_sequence();

    void gen_set_context(npu_sequence* seq, const uint32_t L) {
        for (int i = 0; i < 2; i++){
            seq->rtp_write(attn_qk_tiles[i], attn_qk_addr, L);
            seq->rtp_write(attn_kv_tiles[i], attn_kv_addr, L);
        }
    }
    void gen_layer_seq(npu_sequence* seq, const uint32_t L, bool is_attn_layer, bool is_first = true);
    void gen_mha_engine_seq(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_wanted);
    /// @brief the same attention for the bfp16 kernel: q, K and V^T arrive as bfp16 tiles (rows of 9/8 of DH bytes),
    ///        256 query rows per round (32 per core). q tiles are [head][L_wanted / 16][DH / 8][2][72 B]; K tiles then
    ///        V^T tiles are [kv head][kv_bfp16_rows / 16][DH / 8][2][72 B] in one buffer (K first).
    void gen_mha_engine_seq_bfp16(npu_sequence* seq, const uint32_t L_begin, const uint32_t L_wanted, const uint32_t kv_bfp16_rows);
    void gen_seq_conv1d(
        npu_sequence* seq,
        const uint32_t L_PADDED, // assume padded
        const uint32_t D,
        const uint32_t L_OUT,
        const uint32_t weight_offset
    );
    size_t get_V_offset() { return this->kv_cache_size / 2; }

    void set_max_length(const uint32_t MAX_L);
};
