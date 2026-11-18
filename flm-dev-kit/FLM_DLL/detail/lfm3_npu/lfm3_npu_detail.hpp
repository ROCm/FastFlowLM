// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __LFM3_NPU_DETAIL_HPP__
#define __LFM3_NPU_DETAIL_HPP__
#include "models/lfm3/lfm3_npu.hpp"
#include "lfm3_model_def.hpp"
#include "lfm3_prefill.hpp"
#include "lfm3_npu_sequence.hpp"
#include "utils/kv_rows_state.hpp"

struct lfm3_npu::Impl {
    Impl(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L);
    ~Impl();

    buffer<bf16> forward(int id);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload);
    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();
    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    int get_current_context_length() { return this->ctx; }
    int checkpoint();
    int restore();

    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    /// one flag per layer: attention (keeps K/V rows) or convolution (keeps a fixed blob)
    std::vector<bool> attn_mask;
    /// highest context length reached since the rows past the length were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    void note_context() { dirty_rows = std::max(dirty_rows, static_cast<size_t>(ctx)); }

    buffer<bf16> prefill_npu(std::vector<int>& ids);
    buffer<bf16> prefill_npu_pass(const int* ids, int n, bool want_logits);
    /// LFM3_DUMP_DIR support: residual rows (n x D) after layer `li`, first call only
    void dump_layer(int li, const uint16_t* rows, int n);

    /// \brief Runs `n` tokens through the model starting at position `ctx` and advances the context.
    /// \return logits of the last token when `want_logits`, else an empty buffer
    buffer<bf16> run(const int* ids, int n, bool want_logits);

    // one block of the model for a chunk of `n` tokens
    void conv_block(int layer, const lfm3_layer_weights& w, const uint16_t* h, uint16_t* out, int n);
    void conv_core(int layer, const lfm3_layer_weights& w, const uint16_t* bcx, uint16_t* y, int n);
    void attn_norm_rope_cache(int layer, const lfm3_layer_weights& w, uint16_t* q, uint16_t* k, const uint16_t* v, int n, int pos0);
    void attn_host_core(int layer, const lfm3_layer_weights& w, const uint16_t* q, const uint16_t* gate, uint16_t* o, int n, int pos0);
    void attn_host_core_emu(int layer, const lfm3_layer_weights& w, const uint16_t* q, const uint16_t* gate, uint16_t* o, int n, int pos0, int emu);
    void attention_block(int layer, const lfm3_layer_weights& w, const uint16_t* h, uint16_t* out, int n, int pos0);
    void ffn_block(const lfm3_layer_weights& w, const uint16_t* h, uint16_t* out, int n);

    // ---- NPU prefill stages ----
    std::unique_ptr<lfm3_prefill_ctx> pctx;  ///< null: everything runs on the host
    bool stage_attn_full = false;            ///< full-attention layers on attn.xclbin
    bool stage_attn_swa = false;             ///< sliding-window layers on attn.xclbin
    int npu_prefill_min = 128;               ///< shorter prompts stay on the host (the fixed per-layer dequant cost dominates)

    /// proj buffers are device BOs when an NPU stage is on (prefill dequant or decode layer), so the kernels can read them
    bool proj_on_device() const { return pctx != nullptr || decode_npu; }
    buffer<u8> alloc_proj(size_t bytes) {
        if (pctx) return pctx->dq_mgr->create_bo_buffer<u8>(bytes);
        if (decode_npu) return layer_mgr->create_bo_buffer<u8>(bytes);
        return buffer<u8>(bytes);
    }
    void sync_proj(buffer<u8>& proj) { if (proj_on_device()) proj.sync_to_device(); }

    // ---- NPU decode (layer.xclbin of FLM_Xclbin/LiquidAI/lfm3_npu_bin) ----
    // LFM3_DECODE = host | conv (default when the xclbin is present): which layers run on the NPU. Conv layers first;
    // attention layers join as their kernels land. LFM3_DECODE_VERIFY=1 runs the host layer too and logs the difference.
    bool decode_npu = false;                 ///< conv layers (and the lm_head) decode on the NPU
    bool decode_conv = true;                 ///< conv layers decode on the NPU
    bool decode_attn = false;                ///< attention layers decode on the NPU too (layer.xclbin built for MODEL_TYPE LFM3_1_7B)
    bool decode_verify = false;
    npu_app_manager* layer_mgr = nullptr;
    npu_app conv_app, attn_full_app, attn_swa_app, lm_app;
    std::unique_ptr<lfm3_npu_sequence> dseq;
    std::vector<buffer<bf16>> d_rms;         ///< per layer: operator | post_operator | ffn | post_ffn norms (4 x D)
    std::vector<buffer<bf16>> d_taps;        ///< per conv layer: conv taps [K][D]
    std::vector<buffer<bf16>> d_ring;        ///< per conv layer: history ring [K-1][D], slot = position % (K-1)
    std::vector<float> d_gate_h;             ///< scratch: the normed layer input as fp32 (multiple of 16 long)
    std::vector<std::vector<float>> d_gate_w; ///< per attention layer: the gate projection [heads][D] as fp32 (exact bf16 values)
    std::vector<buffer<bf16>> d_rope;        ///< per attention layer: cos[DH/2] sin[DH/2] q_norm[DH] k_norm[DH] (cos/sin rewritten per token)
    /// where the KV caches are authoritative: the host mapping or the device (after NPU decode steps wrote rows)
    /// per attention layer: who holds the current K/V rows, the host mapping or the device (0 host, 1 device)
    std::vector<char> kv_owner;
    void kv_to_host();                       ///< all layers
    void kv_to_dev();
    void kv_layer_to_host(size_t li);
    void kv_layer_to_dev(size_t li);
    /// LFM3_ATTN_NPU = full | swa | full,swa (default): which attention kinds run on the NPU (the other kind runs on the host)
    bool attn_npu_full = true, attn_npu_swa = true;
    bool attn_on_npu(lfm3_block_kind k) const { return decode_attn && (k == lfm3_block_kind::sliding_attention ? attn_npu_swa : attn_npu_full); }
    buffer<bf16> d_x, d_logits, d_final_norm;
    buffer<u8> d_lm_w;                       ///< lm_head as signed int4 + min', vocab padded, device layout
    /// where the conv history is authoritative: the host rows (oldest first) or the device ring
    enum class conv_where { host, ring } conv_state = conv_where::host;
    void init_decode();
    void build_decode_weights();
    void conv_host_to_ring();                ///< hist (time order) -> ring (slot = position % (K-1)), position = ctx
    void conv_ring_to_host();
    void ensure_conv_host() { if (conv_state == conv_where::ring) conv_ring_to_host(); }
    buffer<bf16> decode_token(int id);
    void gen_decode_seqs(uint32_t L);
    int seq_gen_L = -1;                      ///< context length the current decode sequences were generated for (-1: none)
    flm_rt::run lm_run;                      ///< lm_head launch (arguments are fixed buffers)
    int decode_stats_tokens = 0; double decode_stats_ms_npu = 0, decode_stats_ms_host = 0, decode_stats_ms_head = 0;

    /// one decoder layer (both blocks) on the host for the n rows of x (in place)
    void host_layer(uint32_t li, uint16_t* x, uint16_t* h, uint16_t* mix, int n, int pos0);
    /// KV caches are device BOs too (the attention kernels read them); created through the prefill attention xclbin's
    /// manager or the decode layer's. Slack of 16 tokens per band end: the cache moves are padded to chunks of 16 tokens.
    bool kv_on_device() const { return pctx != nullptr || decode_attn; }
    buffer<bf16> alloc_kv(size_t elems) {
        if (pctx) return pctx->mha_mgr->create_bo_buffer<bf16>(elems);
        if (decode_attn) return layer_mgr->create_bo_buffer<bf16>(elems);
        return buffer<bf16>(elems);
    }
    static constexpr int KV_SLACK_TOKENS = 16;

    LM_Config config;
    npu_xclbin_manager* npu;               ///< unused while every stage runs on the host
    lfm3_desc desc;
    uint32_t MAX_L;
    int ctx = 0;                            ///< tokens in the context
    bool dump_done = false;                 ///< LFM3_DUMP_DIR: only the first call is dumped

    std::vector<lfm3_layer_weights> layers;
    Embedding embedding;
    buffer<bf16> final_norm;                ///< model.embedding_norm
    bytes lm_head_blob;                     ///< tied embedding as a Q4_1 matrix, plain q4nx block order
    lfm3_cpu::qmat lm_head;                 ///< view of lm_head_blob
    bool weights_loaded = false;

    // per-layer state: attention layers hold K and V rows (post-RoPE) in the four-band device layout
    // [K heads 0..h/2-1 | K heads h/2..h-1 | V first half | V second half], each MAX_L rows; conv layers the last K-1 conv inputs
    std::vector<buffer<bf16>> kv_cache;
    std::vector<int> conv_index;            ///< layer -> index into conv_hist, -1 for attention layers
    std::vector<buffer<bf16>> conv_hist;    ///< [K-1][D] per conv layer, oldest row first

    bool ckpt_valid = false;
    int ckpt_ctx = 0;
    std::vector<buffer<bf16>> conv_ckpt;

    static constexpr int CHUNK = 256;       ///< tokens per host pass
};

#endif
