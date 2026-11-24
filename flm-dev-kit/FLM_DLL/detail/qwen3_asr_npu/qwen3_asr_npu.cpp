// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "qwen3_asr_npu_detail.hpp"
#include <chrono>
#include <cstring>
#include <future>
#include <optional>
#include <filesystem>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

qwen3_asr_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance)
{
    current_context_length = 0;
    is_checkpoint_valid = false;

    this->desc.build(config);

    // the model has a floor on the context it will run with, and the kv cache
    // layout wants a power of two
    this->MAX_L = std::max<uint32_t>(MAX_L, this->desc.min_max_length);
    float log2_MAX_L = std::log2(this->MAX_L);
    if (log2_MAX_L != std::floor(log2_MAX_L))
    {
        this->MAX_L = 1 << ((int)std::floor(log2_MAX_L) + 1);
    }
    MAX_L = this->MAX_L; // synchronize MAX_L

    this->sequence = std::make_unique<qwen3_asr_npu_sequence>(this->desc, config, MAX_L);

    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
    this->layer = layer_app_manager->create_app();
    this->set_layer_rtp = layer_app_manager->create_app();
    if (!this->npu->is_preemption_enabled())
    {
        this->layers_run = layer_app_manager->create_runlist();
    }
    this->lm_head = layer_app_manager->create_app();

    // allocate all buffers
    rms_weights.resize(desc.num_hidden_layers);
    proj_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);

    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);

    for (size_t i = 0; i < desc.num_hidden_layers; i++)
    {
        proj_weights[i] = layer.create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        rms_weights[i] = layer.create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        rope_rms_weights[i] = layer.create_bo_buffer<bf16>(desc.rope_rms_weights_byte_size / sizeof(bf16));
        this->kv_caches[i].sync_to_device();
    }
    x = layer.create_bo_buffer<bf16>(desc.D);
    lm_head_weights = layer.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    final_rms_weight = layer.create_bo_buffer<bf16>(desc.D);
    logits = layer.create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    this->K03_offset = sequence->get_k03_offset();
    this->K47_offset = sequence->get_k47_offset();
    this->V03_offset = sequence->get_v03_offset();
    this->V47_offset = sequence->get_v47_offset();

    this->set_context_length(0);

    this->sequence->gen_lm_head_seq(this->lm_head.seq());
    this->lm_head_run = this->lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
}

void qwen3_asr_npu::Impl::load_weights(Q4NX &q4nx)
{
    std::unique_ptr<Q4NX> npu_q4nx;   // the Q4_1 copies of the mixed build (see qwen3_asr_desc::mixed)
    if (this->desc.mixed)
    {
        npu_q4nx = std::make_unique<Q4NX>(config.model_path);
        npu_q4nx->switch_model(utils::path_join(config.model_path, qwen3_asr_desc::MIX_FILE));
        header_print("FLM", "qwen3_asr_npu: mixed Q4_1 / Q8_0 decode weights");
    }
    this->desc.load_head_weights(q4nx, npu_q4nx.get(), this->lm_head_weights, this->final_rms_weight);
    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++)
    {
        this->desc.load_layer_weights(layer_idx, q4nx, npu_q4nx.get(),
            this->proj_weights[layer_idx],
            this->rms_weights[layer_idx],
            this->rope_rms_weights[layer_idx]);
    }
    this->cpu_decoder = std::make_unique<qwen3_asr_cpu::decoder>(utils::path_join(config.model_path, "model.q4nx"));
    this->cpu_kv_length = 0;
    const std::string audio_path = utils::path_join(config.model_path, "audio_weight.q4nx");
    if (std::filesystem::exists(audio_path))
        this->audio_encoder = std::make_unique<qwen3_asr_audio_encoder>(audio_path);
    this->clear_context();
}

void qwen3_asr_npu::Impl::_set_rope_rms_weights(int idx)
{
    const uint32_t DH = desc.DH;
    buffer<bf16> cos_buf(DH / 2);
    buffer<bf16> sin_buf(DH / 2);
    for (uint32_t j = 0; j < DH / 2; j++)
    {
        cos_buf[j] = (bf16)cos(desc.inv_freq[j] * idx);
        sin_buf[j] = (bf16)sin(desc.inv_freq[j] * idx);
    }

    for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
    {
        bf16 *w_rope_ptr = this->rope_rms_weights[i].data();
        for (uint32_t j = 0; j < DH / 2; j++)
        {
            w_rope_ptr[j] = cos_buf[j];
            w_rope_ptr[j + DH / 2] = sin_buf[j];
        }
        this->rope_rms_weights[i].sync_to_device();
    }
}

void qwen3_asr_npu::Impl::set_context_length(int L)
{
    this->current_context_length = L;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(L));

    this->sequence->gen_layer_rtp_seq(this->set_layer_rtp.seq(), L + 1);
    this->sequence->gen_layer_seq(this->layer.seq(), L + 1);
    if (!this->npu->is_preemption_enabled())
    {
        this->layers_run.reset();
        this->layers_run.add(this->set_layer_rtp.create_run());

        // QWEN3_ASR_NPU_LAYERS=n runs only the first n decoder layers on the NPU (bring-up / bisecting a hang); the output is then
        // meaningless, only "does it complete" is
        uint32_t layers_to_run = desc.num_hidden_layers;
        if (const char *dbg = std::getenv("QWEN3_ASR_NPU_LAYERS"))
            layers_to_run = std::min<uint32_t>(desc.num_hidden_layers, (uint32_t)std::atoi(dbg));
        for (uint32_t i = 0; i < layers_to_run; i++)
        {
            this->layers_run.add(this->layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]));
        }
    }

    this->_set_rope_rms_weights(L);
}

buffer<bf16> qwen3_asr_npu::Impl::forward(int ids)
{
    // QWEN3_ASR_CPU_DECODE=1: decode on the CPU decoder instead of the NPU (accuracy A/B of the NPU decode kernel; the NPU
    // KV cache is not updated in this mode)
    static const bool cpu_decode = std::getenv("QWEN3_ASR_CPU_DECODE") != nullptr;
    if (cpu_decode)
    {
        std::vector<float> e(desc.D), cpu_logits;
        this->cpu_decoder->embed(ids, e.data());
        this->cpu_decoder->forward(e.data(), 1, cpu_logits);
        this->cpu_kv_length = this->cpu_decoder->context_length();
        this->current_context_length += 1;
        if (this->prefill_logits.size() != desc.vocab_size)
            this->prefill_logits = buffer<bf16>(desc.vocab_size);
        for (uint32_t i = 0; i < desc.vocab_size; i++)
            this->prefill_logits.data()[i] = (bf16)cpu_logits[i];
        return this->prefill_logits;
    }
    // the token's row of the bf16 embedding table, which the CPU decoder maps from model.q4nx (no second copy in memory)
    std::memcpy(this->x.data(), this->cpu_decoder->embed_row(ids), desc.D * sizeof(bf16));
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled())
    {
        this->layers_run.execute();
        this->layers_run.wait();
    }
    else
    {
        this->set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
        {
            this->layer(this->x, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i], this->kv_caches[i]);
        }
    }

    this->lm_head_run.start();
    this->set_context_length(this->current_context_length + 1);
    this->lm_head_run.wait();
    this->logits.sync_from_device();
    return this->logits_valid;
}

buffer<bf16> qwen3_asr_npu::Impl::prefill(std::vector<int> &ids, void *payload)
{
    // The prompt is prefilled on the CPU: for the short prompts of speech recognition (about 100-400 tokens: a system
    // message, the audio tokens, the text so far) the CPU is faster than the NPU prefill, and it needs no prefill kernels.
    return this->_prefill_on_cpu(ids, static_cast<const qwen3_asr_payload_t *>(payload));
}

void qwen3_asr_npu::Impl::_sync_cpu_kv_from_npu(int rows)
{
    if (rows == 0)
    {
        this->cpu_decoder->reset();
        this->cpu_kv_length = 0;
        return;
    }
    if (this->cpu_kv_length == rows)
        return; // the CPU cache already holds these rows (the last thing that ran was a CPU prefill of this context)

    const kv_row_layout layout = this->kv_layout();
    std::vector<bf16> k((size_t)rows * desc.DK), v((size_t)rows * desc.DV);
    this->cpu_decoder->reset();
    this->_kv_rows_sync(0, static_cast<size_t>(rows), false);
    for (uint32_t li = 0; li < desc.num_hidden_layers; li++)
    {
        kv_rows_copy(this->kv_caches[li].data(), layout, 0, rows, k.data(), v.data(), true);
        this->cpu_decoder->import_kv((int)li, rows, reinterpret_cast<const uint16_t *>(k.data()), reinterpret_cast<const uint16_t *>(v.data()));
    }
    this->cpu_decoder->set_context_length(rows);
    this->cpu_kv_length = rows;
}

void qwen3_asr_npu::Impl::_embed_rows(const std::vector<int> &ids, int from, int to, int pad_id, const bf16 *audio, int n_audio, int audio_row,
                                      float *out)
{
    const int D = (int)desc.D;
    for (int i = from; i < to; i++)
    {
        float *row = out + (size_t)(i - from) * D;
        if (ids[i] == pad_id)
        {
            if (audio_row >= n_audio)
                throw std::runtime_error("qwen3_asr_npu: more <|audio_pad|> tokens than audio embeddings");
            const bf16 *src = audio + (size_t)audio_row * D;
            for (int d = 0; d < D; d++)
                row[d] = (float)src[d];
            audio_row++;
        }
        else
        {
            this->cpu_decoder->embed(ids[i], row);
        }
    }
}

buffer<bf16> qwen3_asr_npu::Impl::_prefill_on_cpu(std::vector<int> &ids, const qwen3_asr_payload_t *payload)
{
    if (!this->cpu_decoder)
        throw std::runtime_error("qwen3_asr_npu: prefill before load_weights");
    const int n = (int)ids.size();
    const int D = (int)desc.D;
    constexpr int WINDOW_FRAMES = 800, WINDOW_TOKENS = 104, EDGE_FRAMES = 3;   // 8 s of mel, its audio tokens, frames a window needs behind it
    const bool streaming = payload && payload->streaming && payload->mel_frames > 0;
    this->_join_prefetch();   // the background job started by the previous step, if any (it used the CPU decoder and the stream cache)
    std::optional<prefetch_t> prefetched = std::move(this->prefetch_done);
    this->prefetch_done.reset();
    if (streaming && payload->stream_reset)
    {
        this->stream_cache = stream_cache_t();
        prefetched.reset();
    }
    if (!streaming && this->npu_kept_rows > 0)   // a normal request after a recording: back to the zeros of a fresh engine
    {
        this->_kv_rows_zero(0, this->npu_kept_rows);
        this->npu_kept_rows = 0;
        this->stream_cache = stream_cache_t();
    }

    int head = 0;                       // index of the first <|audio_pad|>
    if (payload)
        while (head < n && ids[head] != payload->audio_pad_id) head++;

    // ---- the audio rows ----
    std::vector<bf16> encoded;          // everything the audio part of the prompt needs, in order
    const bf16 *audio_rows = payload ? payload->audio_embeds.data() : nullptr;
    int n_audio = payload ? payload->n_audio : 0;
    int cached_windows = 0;             // streaming: windows taken from the cache
    bool background_used = false;       // the previous step's background job was for exactly this mel (and, offline, this prompt)
    if (payload && payload->mel_frames > 0)
    {
        if (!this->audio_encoder)
            throw std::runtime_error("qwen3_asr_npu: mel given but the package has no audio_weight.q4nx");
        n_audio = qwen3_asr_audio_tokens(payload->mel_frames);
        if (streaming && this->stream_cache.valid)
            cached_windows = this->stream_cache.windows;
        const int slice_start = cached_windows * WINDOW_FRAMES;   // windows are independent: encode from the first one not cached
        const auto t_enc = std::chrono::steady_clock::now();
        std::vector<asr_bf16_bits> bits;
        const int open_frames = payload->mel_frames - slice_start;
        background_used = prefetched && prefetched->frames == payload->mel_frames && prefetched->slice_start == slice_start &&
                          (int)prefetched->mel_slice.size() == open_frames * 128 &&
                          std::memcmp(prefetched->mel_slice.data(), payload->mel.data() + (size_t)slice_start * 128, sizeof(float) * (size_t)open_frames * 128) == 0 &&
                          (streaming || prefetched->ids == ids);
        if (background_used)
            bits = std::move(prefetched->bits);   // encoded in the background during the previous step's decode
        else
            bits = this->audio_encoder->encode(payload->mel.data() + (size_t)slice_start * 128, open_frames,
                                               streaming ? &this->stream_cache.conv : nullptr);
        if (std::getenv("QWEN3_ASR_PROF"))
            std::fprintf(stderr, "[prof] encoder %d frames: %.0f ms%s\n", payload->mel_frames - slice_start,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc).count(), background_used ? " (background)" : "");
        encoded.resize((size_t)cached_windows * WINDOW_TOKENS * D + bits.size());
        if (cached_windows > 0)
            std::memcpy(encoded.data(), this->stream_cache.rows.data(), (size_t)cached_windows * WINDOW_TOKENS * D * sizeof(bf16));
        std::memcpy(encoded.data() + (size_t)cached_windows * WINDOW_TOKENS * D, bits.data(), bits.size() * sizeof(bf16));   // same 16-bit storage
        if ((int)(encoded.size() / D) != n_audio)
            throw std::runtime_error("qwen3_asr_npu: audio token count mismatch");
        audio_rows = encoded.data();
    }

    // ---- how much of the prompt is already in the CPU decoder's cache ----
    int start = this->current_context_length;     // rows already in the caches (normal case: the context so far)
    int first = 0;                                // the first row to compute
    std::vector<float> cpu_logits;
    if (streaming)
    {
        const int reuse = this->stream_cache.valid ? head + cached_windows * WINDOW_TOKENS : 0;
        const bool same_prefix = reuse > 0 && (int)this->stream_cache.prefix_ids.size() >= reuse && reuse <= n &&
                                 std::equal(ids.begin(), ids.begin() + reuse, this->stream_cache.prefix_ids.begin()) &&
                                 this->cpu_kv_length >= reuse;
        start = same_prefix ? reuse : 0;
        first = start;
        if (start == 0)
            this->cpu_decoder->reset();
        else
            this->cpu_decoder->set_context_length(start);   // its rows [0, start) are the head + cached windows
        this->cpu_kv_length = start;
    }
    else if (start == 0 && background_used && !prefetched->logits.empty() && this->cpu_kv_length == n)
    {
        first = n;   // offline piece: the whole prompt was prefilled in the background
        cpu_logits = std::move(prefetched->logits);
    }
    if ((size_t)(start + (n - (streaming ? start : 0))) > (size_t)this->MAX_L)
        throw std::runtime_error("qwen3_asr_npu: context would exceed the maximum length");

    if (first < n)
    {
        const int rows_new = n - first;
        std::vector<float> emb((size_t)rows_new * D);
        this->_embed_rows(ids, first, n, payload ? payload->audio_pad_id : -1, audio_rows, n_audio, payload ? std::max(0, first - head) : 0, emb.data());
        if (!streaming)
            this->_sync_cpu_kv_from_npu(start);
        this->cpu_decoder->forward(emb.data(), rows_new, cpu_logits);
    }
    const int total = n + (streaming ? 0 : start);   // offline: start (the context so far) + the n new ids; streaming: ids hold everything
    this->cpu_kv_length = total;

    // hand the new K/V rows to the NPU's cache. Streaming: the rows of the prompt head and the closed windows are still there when
    // clear_context kept them (npu_kept_rows); otherwise all rows [0, total) go over, from the CPU decoder's cache.
    const kv_row_layout layout = this->kv_layout();
    const int export_from = !streaming ? start : (start <= (int)this->npu_kept_rows ? start : 0);
    const int export_rows = total - export_from;
    std::vector<bf16> k((size_t)export_rows * desc.DK), v((size_t)export_rows * desc.DV);
    for (uint32_t li = 0; li < desc.num_hidden_layers; li++)
    {
        this->cpu_decoder->export_kv((int)li, export_from, export_rows, reinterpret_cast<uint16_t *>(k.data()), reinterpret_cast<uint16_t *>(v.data()));
        kv_rows_copy(this->kv_caches[li].data(), layout, export_from, export_rows, k.data(), v.data(), false);   // whole cache lines: no invalidate needed
    }
    this->_kv_rows_sync(static_cast<size_t>(export_from), static_cast<size_t>(total), true);
    if (streaming)   // rows behind the new end: the previous step's text and tokens unless clear_context zeroed them already
    {
        this->_kv_rows_zero(static_cast<size_t>(total), this->dirty_rows);
        this->dirty_rows = static_cast<size_t>(total);
    }
    this->npu_kept_rows = 0;
    this->dirty_rows = std::max(this->dirty_rows, static_cast<size_t>(total));
    this->set_context_length(total);

    if (streaming)   // what the next step may reuse: the head and the windows closed by now (their encoder rows and K/V rows)
    {
        const int closed = std::max(0, (payload->mel_frames - EDGE_FRAMES) / WINDOW_FRAMES);
        if (closed > this->stream_cache.windows || !this->stream_cache.valid)
        {
            const int keep = std::max(closed, 0);
            const int chunks_per_window = WINDOW_FRAMES / this->audio_encoder->config().chunk_frames();
            this->stream_cache.conv.drop_front((keep - (this->stream_cache.valid ? this->stream_cache.windows : 0)) * chunks_per_window,
                                               this->audio_encoder->config().chunk_tokens() * this->audio_encoder->config().d_model,   // encoder rows, not decoder
                                               this->audio_encoder->config().chunk_frames() * this->audio_encoder->config().num_mel_bins);
            this->stream_cache.rows.assign(audio_rows, audio_rows + (size_t)keep * WINDOW_TOKENS * D);
            this->stream_cache.windows = keep;
        }
        this->stream_cache.prefix_ids.assign(ids.begin(), ids.begin() + std::min(n, head + this->stream_cache.windows * WINDOW_TOKENS));
        this->stream_cache.valid = true;
    }
    if (payload && !std::getenv("QWEN3_ASR_CPU_DECODE"))   // (the CPU decode mode uses the CPU decoder during the decode)
        this->_start_background(*payload, streaming);

    // logits of the last prompt token. They must not go through the NPU's logits buffer (see prefill_logits in the header).
    if (this->prefill_logits.size() != desc.vocab_size)
        this->prefill_logits = buffer<bf16>(desc.vocab_size);
    for (uint32_t i = 0; i < desc.vocab_size; i++)
        this->prefill_logits.data()[i] = (bf16)cpu_logits[i];
    return this->prefill_logits;
}

void qwen3_asr_npu::Impl::_start_background(const qwen3_asr_payload_t &payload, bool streaming)
{
    constexpr int WINDOW_FRAMES = 800;
    const int D = (int)desc.D;
    const int frames = payload.next_mel_frames;
    if (frames <= 0 || (int)payload.next_mel.size() != frames * 128)
        return;
    if (streaming && frames > payload.mel_frames)
    {
        // the next step: this recording plus a chunk - encode its open windows now. (Prefilling their rows too was slower: a second CPU
        // prefill call per step reads all weights again, and the job then outlasts the decode.)
        const int next_start = this->stream_cache.windows * WINDOW_FRAMES;
        std::vector<float> slice(payload.next_mel.begin() + (size_t)next_start * 128, payload.next_mel.end());
        this->prefetch = std::async(std::launch::async, [this, frames, next_start, slice = std::move(slice)]() mutable {
            prefetch_t r;
            r.frames = frames;
            r.slice_start = next_start;
            r.bits = this->audio_encoder->encode(slice.data(), frames - next_start, &this->stream_cache.conv);
            r.mel_slice = std::move(slice);
            return r;
        });
    }
    else if (!streaming && !payload.next_ids.empty())
    {
        // the next offline piece: its whole prompt does not depend on this piece's text
        std::vector<int> ids = payload.next_ids;
        std::vector<float> mel = payload.next_mel;
        const int pad = payload.audio_pad_id;
        this->prefetch = std::async(std::launch::async, [this, frames, pad, D, ids = std::move(ids), mel = std::move(mel)]() mutable {
            prefetch_t r;
            r.frames = frames;
            r.slice_start = 0;
            r.bits = this->audio_encoder->encode(mel.data(), frames, nullptr);
            r.mel_slice = std::move(mel);
            const int n = (int)ids.size(), n_audio = (int)(r.bits.size() / D);
            std::vector<float> emb((size_t)n * D);
            this->_embed_rows(ids, 0, n, pad, reinterpret_cast<const bf16 *>(r.bits.data()), n_audio, 0, emb.data());
            this->cpu_decoder->reset();
            this->cpu_decoder->forward(emb.data(), n, r.logits);
            this->cpu_kv_length = n;
            r.ids = std::move(ids);
            return r;
        });
    }
}

buffer<bf16> qwen3_asr_npu::Impl::get_k_cache(int layer_idx, int idx)
{
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    bf16 *k_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void *>(k_cache.data()), static_cast<void *>(k_cache_ptr + offset), desc.DK / 2 * sizeof(bf16));
    memcpy(static_cast<void *>(k_cache.data() + desc.DK / 2), static_cast<void *>(k_cache_ptr + offset + K47_offset), desc.DK / 2 * sizeof(bf16));

    return k_cache;
}

buffer<bf16> qwen3_asr_npu::Impl::get_v_cache(int layer_idx, int idx)
{
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    bf16 *v_cache_ptr = this->kv_caches[layer_idx].data();
    uint32_t offset = idx * desc.DH * 4;
    memcpy(static_cast<void *>(v_cache.data()), static_cast<void *>(v_cache_ptr + offset + V03_offset), desc.DV / 2 * sizeof(bf16));
    memcpy(static_cast<void *>(v_cache.data() + desc.DV / 2), static_cast<void *>(v_cache_ptr + offset + V47_offset), desc.DV / 2 * sizeof(bf16));
    return v_cache;
}

buffer<bf16> qwen3_asr_npu::Impl::get_logits(buffer<bf16> &predict)
{
    memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    this->lm_head_run.wait();
    this->logits.sync_from_device();
    return this->logits_valid;
}
void qwen3_asr_npu::Impl::clear_context()
{
    this->_join_prefetch();   // it may still be writing the stream cache
    // saved states hold their own copies of the rows; the live ones must not leak into the next conversation. Streaming keeps the
    // rows of the prompt head and the closed windows (the next step of the recording starts there, see _prefill_on_cpu; any other
    // request zeroes them first).
    const size_t keep = this->stream_cache.valid ? std::min(this->stream_cache.prefix_ids.size(), static_cast<size_t>(this->cpu_kv_length)) : 0;
    this->_kv_rows_zero(keep, this->dirty_rows);
    this->dirty_rows = keep;
    this->npu_kept_rows = keep;
    this->set_context_length(0);
}

void qwen3_asr_npu::Impl::release_host_memory()
{
    try
    {
        this->_join_prefetch();
    }
    catch (const std::exception &)
    {
        // a failed background job of a request that is over: nothing uses its result
    }
    this->prefetch_done.reset();
    this->stream_cache = stream_cache_t();   // nothing kept: clear_context zeroes every written NPU row
    this->clear_context();
    this->cpu_kv_length = 0;
    if (!this->cpu_decoder)
        return;
    // Short requests leave little behind, and giving it back costs the next prefill its page faults again (about 4 %): only
    // after a long context (~35 s of audio and more) are the decoder's K/V and the freed heap handed back to the system.
    constexpr int RELEASE_MIN_ROWS = 512;
    if (this->cpu_decoder->peak_context_length() < RELEASE_MIN_ROWS)
    {
        this->cpu_decoder->reset();
        return;
    }
    this->cpu_decoder->release();
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

void qwen3_asr_npu::Impl::_kv_rows_sync(size_t from, size_t to, bool to_device)
{
    if (to <= from)
        return;
    const kv_row_layout l = this->kv_layout();
#if defined(FLM_USE_HRX)
    const bool ranges = false;   // the HRX shim syncs whole buffers (and defers the flush to the next dispatch)
#else
    const bool ranges = l.striped;
#endif
    for (auto &kv : this->kv_caches)
    {
        if (!ranges)
        {
            to_device ? kv.sync_to_device() : kv.sync_from_device();
            continue;
        }
#if !defined(FLM_USE_HRX)
        const size_t bytes = (to - from) * l.stride() * sizeof(bf16);
        for (size_t plane : {size_t(0), l.k47, l.v03, l.v47})   // rows are contiguous within each of the four planes
            kv.bo().sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, bytes, (plane + from * l.stride()) * sizeof(bf16));
#endif
    }
}

void qwen3_asr_npu::Impl::_kv_rows_zero(size_t from, size_t to)
{
    if (to <= from)
        return;
    const kv_row_layout l = this->kv_layout();
    if (!l.striped)
    {
        kv_rows_zero(this->kv_caches, l, from, to);
        return;
    }
    for (auto &kv : this->kv_caches)   // whole cache lines are overwritten: no invalidate needed before
        for (size_t plane : {size_t(0), l.k47, l.v03, l.v47})
            std::memset(kv.data() + plane + from * l.stride(), 0, (to - from) * l.stride() * sizeof(bf16));
    this->_kv_rows_sync(from, to, true);
}

int qwen3_asr_npu::Impl::get_current_context_length()
{
    return this->current_context_length;
}

void qwen3_asr_npu::Impl::_allocate_checkpoint_buffers(){}

int qwen3_asr_npu::Impl::checkpoint()
{
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return checkpoint_context_length;
}

int qwen3_asr_npu::Impl::restore()
{
    if (!is_checkpoint_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(checkpoint_context_length));
    this->set_context_length(checkpoint_context_length);
    return checkpoint_context_length;
}

kv_row_layout qwen3_asr_npu::Impl::kv_layout() const
{
    kv_row_layout l;
    l.DK = desc.DK; l.DV = desc.DV; l.striped = true;
    l.k47 = K47_offset; l.v03 = V03_offset; l.v47 = V47_offset;
    return l;
}

std::unique_ptr<hybrid_engine_state> qwen3_asr_npu::Impl::new_state()
{
    return kv_rows_new_state(desc.num_hidden_layers, kv_layout());
}

int qwen3_asr_npu::Impl::read_state(hybrid_engine_state& state, int kv_from)
{
    header_print_r("FLM", "Reading state at context length " + std::to_string(this->current_context_length) + ", K/V rows from " + std::to_string(kv_from));
    return kv_rows_read(state, this->kv_caches, kv_layout(), this->current_context_length, kv_from);
}

int qwen3_asr_npu::Impl::write_state_kv(const hybrid_engine_state& state)
{
    this->_join_prefetch();
    this->prefetch_done.reset();
    this->npu_kept_rows = 0;
    this->stream_cache = stream_cache_t();
    return kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L);
}

int qwen3_asr_npu::Impl::write_state(const hybrid_engine_state& state)
{
    const int length = kv_rows_write(state, this->kv_caches, kv_layout(), this->MAX_L);
    if (length < 0)
    {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    this->_join_prefetch();
    this->prefetch_done.reset();
    this->npu_kept_rows = 0;
    this->stream_cache = stream_cache_t();
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(this->kv_caches, kv_layout(), static_cast<size_t>(length), this->dirty_rows);
    this->dirty_rows = static_cast<size_t>(length);
    this->set_context_length(length);
    return length;
}

void qwen3_asr_npu::Impl::update_max_length(uint32_t MAX_L)
{
    if (MAX_L <= this->MAX_L)
    {
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return; // no need to update
    }
    this->MAX_L = MAX_L;
    size_t kv_cache_size = desc.get_kv_cache_size(MAX_L);
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    size_t original_k03_offset = this->K03_offset;
    size_t original_k47_offset = this->K47_offset;
    size_t original_v03_offset = this->V03_offset;
    size_t original_v47_offset = this->V47_offset;
    this->sequence->set_max_length(MAX_L);
    this->K03_offset = this->sequence->get_k03_offset();
    this->K47_offset = this->sequence->get_k47_offset();
    this->V03_offset = this->sequence->get_v03_offset();
    this->V47_offset = this->sequence->get_v47_offset();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++)
    {
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void *>(kv_cache_temp.data()), static_cast<void *>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = layer.create_bo_buffer<bf16>(kv_cache_size);
        this->kv_caches[i].sync_to_device();
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->K03_offset), static_cast<void *>(kv_cache_temp.data() + original_k03_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->V03_offset), static_cast<void *>(kv_cache_temp.data() + original_v03_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->K47_offset), static_cast<void *>(kv_cache_temp.data() + original_k47_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
        memcpy(static_cast<void *>(this->kv_caches[i].data() + this->V47_offset), static_cast<void *>(kv_cache_temp.data() + original_v47_offset), (kv_cache_temp.size() / 4) * sizeof(bf16));
    }
    this->set_context_length(this->current_context_length); // regenerate sequence
}
///@brief destructor of qwen3_asr_npu
qwen3_asr_npu::Impl::~Impl() = default;
/*
qwen3_asr_npu::Impl::~Impl(){
    for (int i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].free();
        this->proj_weights[i].free();
        this->rms_weights[i].free();
    }
    this->kv_caches.clear();
    this->rms_weights.clear();
    this->proj_weights.clear();
    this->rope_weights.release();
    this->weights_p0.release();
    this->weights_p1.release();
    this->x.release();
    this->lm_head.reset();
}
*/
// ===============================================
// Externals for qwen3_asr_npu
// ===============================================
qwen3_asr_npu::qwen3_asr_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
{
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> qwen3_asr_npu::forward(int ids)
{
    return this->_impl->forward(ids);
}

buffer<bf16> qwen3_asr_npu::prefill(std::vector<int> &ids, void *payload)
{
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen3_asr_npu::set_context_length(int L)
{
    this->_impl->set_context_length(L);
}

void qwen3_asr_npu::load_weights(Q4NX &q4nx)
{
    this->_impl->load_weights(q4nx);
}

void qwen3_asr_npu::update_max_length(uint32_t MAX_L)
{
    this->_impl->update_max_length(MAX_L);
}

void qwen3_asr_npu::clear_context()
{
    this->_impl->clear_context();
}

void qwen3_asr_npu::release_host_memory()
{
    this->_impl->release_host_memory();
}

buffer<bf16> qwen3_asr_npu::get_k_cache(int layer_idx, int idx)
{
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> qwen3_asr_npu::get_v_cache(int layer_idx, int idx)
{
    return this->_impl->get_v_cache(layer_idx, idx);
}

int qwen3_asr_npu::get_current_context_length()
{
    return this->_impl->get_current_context_length();
}

int qwen3_asr_npu::checkpoint()
{
    return this->_impl->checkpoint();
}

int qwen3_asr_npu::restore()
{
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> qwen3_asr_npu::new_state()
{
    return this->_impl->new_state();
}

int qwen3_asr_npu::read_state(hybrid_engine_state& state, int kv_from)
{
    return this->_impl->read_state(state, kv_from);
}

int qwen3_asr_npu::write_state(const hybrid_engine_state& state)
{
    return this->_impl->write_state(state);
}

int qwen3_asr_npu::write_state_kv(const hybrid_engine_state& state)
{
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* qwen3_asr_npu_state_api(void* engine)
{
    return hybrid_state_api<qwen3_asr_npu>(engine);
}

qwen3_asr_npu::~qwen3_asr_npu()
{
    delete this->_impl;
}
