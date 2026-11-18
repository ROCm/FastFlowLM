// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __QWEN3_ASR_NPU_DETAIL_HPP__
#define __QWEN3_ASR_NPU_DETAIL_HPP__
#include "models/qwen3_asr/qwen3_asr_npu.hpp"
#include "qwen3_asr_npu_def.hpp"
#include "decoder/qwen3_asr_cpu_decoder.hpp"
#include "audio/qwen3_asr_audio_encoder.hpp"
#include "utils/kv_rows_state.hpp"
#include <future>
#include <optional>

struct qwen3_asr_npu::Impl{

    static constexpr int LC = 16;
    /// @brief  initialize the qwen3_asr_npu
    /// @param config
    /// @param npu_instance
    Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~Impl();

    /// @brief forward the qwen3_asr_npu
    buffer<bf16> forward(int ids);
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr);
    /// @brief prefill on the CPU (AVX-512 decoder over the same Q8_0 weights); the K/V rows it produces are written
    ///        into the NPU's striped KV cache, so decoding continues on the NPU
    buffer<bf16> _prefill_on_cpu(std::vector<int>& ids, const qwen3_asr_payload_t* payload);
    /// @brief make the CPU decoder's own K/V cache hold exactly the first `rows` rows of the NPU cache
    void _sync_cpu_kv_from_npu(int rows);
    /// @brief make the host's writes to K/V rows [from, to) of every layer visible to the NPU (to_device) or the NPU's to the host;
    ///        only those rows, not the whole 32 MB layer buffer (XRT: byte-range sync of the four planes; HRX: whole buffer)
    void _kv_rows_sync(size_t from, size_t to, bool to_device);
    /// @brief zero K/V rows [from, to) of every layer (like kv_rows_zero, with a row-range sync)
    void _kv_rows_zero(size_t from, size_t to);

    void set_context_length(int L);
    void load_weights(Q4NX& q4nx);
    void update_max_length(uint32_t MAX_L);
    void clear_context();

    buffer<bf16> get_k_cache(int layer_idx, int idx);
    buffer<bf16> get_v_cache(int layer_idx, int idx);
    buffer<bf16> get_logits(buffer<bf16>& x);

    int get_current_context_length();
    int checkpoint();
    int restore();
    void _allocate_checkpoint_buffers();

    /// @brief saved states (see flm_state_api.h)
    kv_row_layout kv_layout() const;
    std::unique_ptr<hybrid_engine_state> new_state();
    int read_state(hybrid_engine_state& state, int kv_from);
    int write_state(const hybrid_engine_state& state);
    int write_state_kv(const hybrid_engine_state& state);
    bool is_checkpoint_valid;
    int checkpoint_context_length;
    std::vector<buffer<bf16>> kv_checkpoint;

    size_t K03_offset;
    size_t K47_offset;
    size_t V03_offset;
    size_t V47_offset;

    LM_Config config;
    npu_xclbin_manager *npu;

    /// @brief model dimensions, weight descriptors and weight loading
    qwen3_asr_desc desc;
    /// @brief CPU prefill (created in load_weights, it reads the same model.q4nx)
    std::unique_ptr<qwen3_asr_cpu::decoder> cpu_decoder;
    /// @brief host audio encoder (created in load_weights when audio_weight.q4nx is in the package)
    std::unique_ptr<qwen3_asr_audio_encoder> audio_encoder;
    /// @brief logits of the last prompt token, in plain host memory. NOT the NPU's `logits` buffer: writing that one from the
    ///        host leaves dirty cache lines that are written back over the NPU's output at the next sync_from_device()
    buffer<bf16> prefill_logits;
    /// @brief streaming caches (see qwen3_asr_payload_t::streaming)
    struct stream_cache_t {
        bool valid = false;
        int windows = 0;                  ///< closed 8 s windows whose encoder output is cached (and whose K/V rows are in both caches)
        std::vector<bf16> rows;           ///< [windows * 104][hidden] encoder output of those windows
        std::vector<int> prefix_ids;      ///< the first head + 104 * windows prompt ids of the last request
        qwen3_asr_audio_encoder::conv_cache conv;   ///< conv front end of the finished 1 s chunks after the closed windows
    } stream_cache;
    /// @brief streaming: the next step's open-window encoding, computed in the background during the NPU decode
    struct prefetch_t {
        int frames = 0, slice_start = 0;  ///< the mel it was computed for: all frames so far, encoded from slice_start
        std::vector<float> mel_slice;     ///< [frames - slice_start][128], to check the next prefill's mel is the same
        std::vector<asr_bf16_bits> bits;  ///< the encoder output
        std::vector<int> ids;             ///< offline: the whole next prompt, prefilled on the CPU (rows [0, ids.size()))
        std::vector<float> logits;        ///< offline: its last-token logits
    };
    std::future<prefetch_t> prefetch;      ///< running background encode (it writes stream_cache.conv: join before touching that)
    std::optional<prefetch_t> prefetch_done;   ///< its result once joined
    void release_host_memory();
    void _join_prefetch() { if (prefetch.valid()) prefetch_done = prefetch.get(); }
    /// @brief start the background job for the next step (see qwen3_asr_payload_t next_mel / next_ids)
    void _start_background(const qwen3_asr_payload_t &payload, bool streaming);
    /// @brief decoder input rows: text tokens from the embedding table, <|audio_pad|> rows from `audio` (from audio row `audio_row`)
    void _embed_rows(const std::vector<int> &ids, int from, int to, int pad_id, const bf16 *audio, int n_audio, int audio_row, float *out);
    /// @brief context length the CPU decoder's K/V cache is valid for (NPU decode steps do not update it)
    int cpu_kv_length = 0;

    void _set_rope_rms_weights(int idx);

    /// @brief decoding path: one xclbin runs a whole layer
    npu_app_manager* layer_app_manager;
    npu_app layer;
    /// @brief writes the per-token RTPs once, ahead of the layer runs
    npu_app set_layer_rtp;
    npu_app lm_head;

    std::unique_ptr<qwen3_asr_npu_sequence> sequence;
    uint32_t MAX_L;

    std::vector<buffer<bf16>> rms_weights;
    /// @brief per layer: the rope cos/sin table, then the q_norm and k_norm rows
    std::vector<buffer<bf16>> rope_rms_weights;
    std::vector<buffer<u8>> proj_weights;
    std::vector<buffer<bf16>> kv_caches;

    buffer<bf16> x;

    buffer<bf16> final_rms_weight;
    buffer<bf16> logits;
    buffer<bf16> logits_valid;
    buffer<u8> lm_head_weights;

    flm_rt::runlist layers_run;
    flm_rt::run lm_head_run;
    int current_context_length;
    /// @brief highest context length reached since the K/V rows past it were last zero (see kv_rows_zero)
    size_t dirty_rows = 0;
    /// @brief streaming: rows [0, npu_kept_rows) of the NPU cache still hold the prompt head and the closed windows of the recording
    ///        after a clear_context (the next step starts there instead of sending them again); 0 otherwise
    size_t npu_kept_rows = 0;
};

#endif
