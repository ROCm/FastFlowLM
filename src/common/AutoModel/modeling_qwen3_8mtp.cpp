/// \file modeling_qwen3_8mtp.cpp
/// \brief Qwen3_8MTP class
/// \author FastFlowLM Team
/// \date 2026-09-16
/// \version 0.9.28
/// \note AutoModel wrapper for Qwen3.8-27B. Multimodal when the checkpoint
///       ships vision_weight.q4nx; image preprocessing lives in
///       modeling_qwen3_8mtp_image.cpp.
/// \note Speculative decode lives here, not in AutoModel: this is the only
///       engine that overrides causal_lm's speculate() hooks, so
///       _speculative_generate() below drives them directly instead of going
///       through AutoModel::_shared_generate()'s plain per-token loop.
/// \note That means _speculative_generate() duplicates _shared_generate()'s
///       seed/stop-rule/teardown structure -- keep the two in sync by hand.

#include "AutoModel/modeling_qwen3_8mtp.hpp"


/************              Qwen3_8MTP family            **************/
Qwen3_8MTP::Qwen3_8MTP(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_8MTP") {}

void Qwen3_8MTP::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->get<std::string>("model_type", "") == qwen3_5
    this->lm_engine = std::make_unique<qwen3_8mtp_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);

    // After setup_tokenizer, which is what fills eos_token_ids -- the engine
    // has no tokenizer and would otherwise run a speculative batch straight
    // past the stop token. Both sides must test the same list or they
    // disagree about where a batch ends: _speculative_generate() stops at the
    // first id that is_eos() accepts, and every token the engine committed
    // after that one is a cache row with nothing in token_history to describe
    // it. Giving the engine the list moves the cut upstream of the caches.
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get()))
        mtp->set_stop_tokens(this->eos_token_ids);

    this->sampler.reset();

    this->enable_tool = true;
    
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);

    // Image-processor geometry: patch/merge/temporal sizes come from
    // config.json's vision_config; the rest fall back to Qwen3.8-27B's
    // upstream preprocessor_config.json values, which the NPU2 checkpoint
    // doesn't ship. A wrong factor here doesn't fail loudly -- it just reads
    // as a slightly confused caption.
    {
        const nlohmann::json& vc = this->lm_config->sub("vision_config");
        this->vision_patch_size          = cfg_get<unsigned int>(vc, "patch_size", 16);
        this->vision_merge_size          = cfg_get<unsigned int>(vc, "spatial_merge_size", 2);
        this->vision_temporal_patch_size = cfg_get<unsigned int>(vc, "temporal_patch_size", 2);
        this->vision_shortest_edge       = cfg_get<unsigned int>(vc, "shortest_edge", 65536);
        this->vision_longest_edge        = cfg_get<unsigned int>(vc, "longest_edge", 16777216);
        this->vision_rescale_factor      = cfg_get<float>(vc, "rescale_factor", 1.0f / 255.0f);
        this->vision_image_mean          = cfg_get<float>(vc, "image_mean", 0.5f);
        this->vision_image_std           = cfg_get<float>(vc, "image_std", 0.5f);
    }
    // if (auto* eng = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
    //     if (eng->has_vision_tower())
    //         header_print("FLM", "vision backend: " << eng->vision_backend());
    // }

    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Qwen3_8MTP::setup_tokenizer(std::string model_path) {
    // tokenizer_config.json lists eos_token_id as [248044, 248046, 248048];
    // _shared_setup_tokenizer registers all of them. config.json alone would
    // only give 248044, which the chat template never emits -- using just
    // that one causes runaway generation.
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_8MTP::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3_8MTP::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // <|image_pad|>, verified against tokenizer.json / config.json's
    // image_token_id. The engine reads the same id, so a mismatch here throws
    // instead of silently scrambling the prompt.
    constexpr int image_soft_token_id = 248056;

    qwen3_8mtp_npu* qwen3_8mtp_engine = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get());

    // No vision_weight.q4nx (or FLM_Q38_VISION=0) means no tower: an image
    // would template into placeholders nothing fills and prefill garbage.
    // Drop it loudly instead.
    const bool vision_ok = qwen3_8mtp_engine && qwen3_8mtp_engine->has_vision_tower();
    if (!vision_ok && !input.images.empty()) {
        header_print("WARNING", "Qwen3.8-27B is loaded without a vision tower; ignoring "
                                << input.images.size() << " image(s)");
        input.images.clear();
    }
    // Audio is not ported: no audio tower, no audio path in this engine.
    if (!input.audios.empty()) {
        header_print("WARNING", "Qwen3.8-27B has no audio path; ignoring "
                                << input.audios.size() << " audio clip(s)");
        input.audios.clear();
    }

    // Decode and preprocess every image BEFORE templating: a failed image
    // must never get a placeholder, or the payload falls out of step and the
    // engine splices image n's rows into image n+1's slots.
    //
    // `pixels` is one contiguous fp32 buffer that grows as images are
    // appended, so engine-facing pointers can't be taken until all images are
    // in. `pixel_offsets` records where each one landed.
    std::vector<qwen3_8mtp_host_image_t> host_images;
    std::vector<size_t> pixel_offsets;
    std::vector<float>  pixels;

    auto stage_image = [&](qwen3_8mtp_host_image_t& image) -> bool {
        if (image.width <= 0 || image.height <= 0) return false;
        const size_t offset = pixels.size();
        this->preprocess_image(image, pixels);
        if (image.grid_h <= 0 || image.grid_w <= 0) {
            pixels.resize(offset);   // undo a partial append
            return false;
        }
        host_images.push_back(std::move(image));
        pixel_offsets.push_back(offset);
        return true;
    };

    if (vision_ok) {
        for (const auto& img_str : input.images) {
            qwen3_8mtp_host_image_t image = this->load_image(img_str);
            if (!stage_image(image))
                header_print("ERROR", "Skipping image that failed to load: " << img_str);
        }
    }

    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        json qwenvl_message = json::array();
        for (const auto& item : input.messages) {
            json entry = item;
            entry.erase("audios");
            if (!vision_ok || !item.contains("images")) {
                entry.erase("images");
                qwenvl_message.push_back(entry);
                continue;
            }

            // Expand into the content-array form the chat template turns into
            // <|vision_start|><|image_pad|><|vision_end|>, one image at a
            // time, in message order.
            json newContent = json::array();
            for (const auto& img : item["images"]) {
                const std::string img_str = img.get<std::string>();
                qwen3_8mtp_host_image_t image = this->load_image_base64(img_str);
                if (!stage_image(image)) {
                    header_print("ERROR", "Skipping invalid base64 image; prefilling "
                                          "language only for this item");
                    continue;
                }
                newContent.push_back({ {"type", "image"}, {"image", img} });
            }
            newContent.push_back({ {"type", "text"}, {"text", item["content"]} });
            qwenvl_message.push_back({ {"role", item["role"]}, {"content", newContent} });
        }
        nlohmann::ordered_json ordered_messages = qwenvl_message;
        templated_text = this->apply_chat_template(ordered_messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        if (host_images.empty()) {
            messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        }
        else {
            nlohmann::ordered_json content;
            content["role"] = "user";
            content["content"] = nlohmann::ordered_json::array();
            for (const auto& img_str : input.images) {
                nlohmann::ordered_json image_obj;
                image_obj["type"]  = "image";
                image_obj["image"] = img_str;
                content["content"].push_back(image_obj);
            }
            nlohmann::ordered_json text_obj;
            text_obj["type"] = "text";
            text_obj["text"] = input.prompt;
            content["content"].push_back(text_obj);
            messages.push_back(content);
        }
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // The template emits ONE <|image_pad|> per image; the model wants one per
    // MERGED patch, which is grid_h * grid_w / merge^2. Expand in place.
    std::vector<int> tokens;
    if (host_images.empty()) {
        tokens = std::move(tokens_init);
    }
    else {
        const int merged = static_cast<int>(this->vision_merge_size *
                                            this->vision_merge_size);
        size_t total_image_tokens = 0;
        for (const auto& im : host_images)
            total_image_tokens += static_cast<size_t>(im.grid_h) * im.grid_w / merged;
        tokens.reserve(tokens_init.size() + total_image_tokens);

        size_t image_counter = 0;
        for (size_t i = 0; i < tokens_init.size(); i++) {
            if (tokens_init[i] == image_soft_token_id &&
                image_counter < host_images.size()) {
                const qwen3_8mtp_host_image_t& im = host_images[image_counter];
                const int n = im.grid_h * im.grid_w / merged;
                tokens.insert(tokens.end(), static_cast<size_t>(n), image_soft_token_id);
                image_counter++;
            } else {
                tokens.push_back(tokens_init[i]);
            }
        }
        if (image_counter != host_images.size()) {
            // Fewer placeholders than staged images: some image's pixels have
            // nowhere to land. Refuse rather than prefill a scrambled caption.
            header_print("ERROR", "templated " << image_counter << " image placeholder(s) "
                                  "for " << host_images.size() << " image(s); refusing to prefill");
            return false;
        }
        header_print("FLM", "Total images: " << image_counter);
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // Prompt-cache aware image alignment. _shared_insert prefix-matches
    // `tokens` against checkpoint_his and erases that prefix itself, so
    // `tokens` must stay untrimmed here. What needs fixing up locally is the
    // image payload: it still holds pixels for cached leading images, so drop
    // those descriptors to keep the survivors aligned with the tokens that
    // survive the erase.
    size_t prefix_skip_count = 0;
    if (!host_images.empty()) {
        const size_t idx = this->checkpoint_his.size();
        for (size_t i = 0; i < idx; i++) {
            if (i < tokens.size() && tokens[i] == this->checkpoint_his[i]) prefix_skip_count++;
            else break;
        }
        // Must match all of checkpoint_his, or _shared_insert clears the
        // context and skips nothing.
        if (prefix_skip_count != idx) prefix_skip_count = 0;

        if (prefix_skip_count > 0) {
            const int merged = static_cast<int>(this->vision_merge_size *
                                                this->vision_merge_size);
            size_t skipped_image_tokens = 0;
            for (size_t i = 0; i < prefix_skip_count; i++)
                if (tokens[i] == image_soft_token_id) skipped_image_tokens++;

            size_t images_to_drop = 0;
            size_t consumed_image_tokens = 0;
            for (const auto& im : host_images) {
                const size_t img_tokens =
                    static_cast<size_t>(im.grid_h) * im.grid_w / merged;
                if (consumed_image_tokens + img_tokens > skipped_image_tokens) break;
                consumed_image_tokens += img_tokens;
                images_to_drop++;
            }

            if (images_to_drop > 0) {
                // pixel_offsets are absolute indices into `pixels`, so
                // dropping the descriptors is enough -- no need to erase the
                // (multi-hundred-MB) buffer itself.
                host_images.erase(host_images.begin(),
                                  host_images.begin() + images_to_drop);
                pixel_offsets.erase(pixel_offsets.begin(),
                                    pixel_offsets.begin() + images_to_drop);
                header_print("FLM", "Prompt-cache hit: dropped " << images_to_drop
                                    << " cached image(s) from payload");
            }
        }
    }

    // Last image token's index, relative to the tokens that survive
    // _shared_insert's prefix erase. Grows chunk 0 (the only chunk that gets
    // the image payload) to cover every image row.
    int last_image_token_index = -1;
    for (int i = static_cast<int>(prefix_skip_count); i < (int)tokens.size(); i++) {
        if (tokens[i] == image_soft_token_id)
            last_image_token_index = i - static_cast<int>(prefix_skip_count);
    }
    last_image_token_index++;   // plus the end-of-image token

    // Engine-facing views. Built here, after every append to `pixels` is done,
    // because the vector reallocates as it grows.
    std::vector<qwen3_8mtp_image_t> image_views(host_images.size());
    for (size_t i = 0; i < host_images.size(); i++) {
        image_views[i].pixel_values = pixels.data() + pixel_offsets[i];
        image_views[i].grid_t       = host_images[i].grid_t;
        image_views[i].grid_h       = host_images[i].grid_h;
        image_views[i].grid_w       = host_images[i].grid_w;
    }
    qwen3_8mtp_image_payload_t image_payload;
    image_payload.images     = image_views.data();
    image_payload.num_images = static_cast<int>(image_views.size());

    qwen3_8mtp_payload_t payload;
    payload.images = &image_payload;
    const bool has_images = image_payload.num_images > 0;

    // hardware
    int restore_idx = -1;

    if (meta_info.restore_allowed) {
        restore_idx = qwen3_8mtp_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    // The chat template's generation prompt ends with the think preamble:
    //   thinking on : "<think>\n"                -> 2 tokens
    //   thinking off: "<think>\n\n</think>\n\n"   -> 4 tokens
    //
    // These are prefilled here as extra rows of the pass that's happening
    // anyway, rather than re-fed one at a time in generate() -- each token
    // there would cost a full ~14 GB weight stream (~3.3 s), so the
    // thinking-off preamble alone used to cost ~13 s with nothing to show for
    // it in any timer. Two side effects: insert()'s sample() now seeds
    // `last_token` from the row that predicts the first ANSWER token (not the
    // preamble), and the MTP head's step-0 catch-up starts primed instead of
    // cold.
    //
    // Thinking on still needs "<think>\n" on screen -- record it for
    // generate() to echo. Thinking off streams nothing: its tokens only close
    // a block that was never opened.
    this->preamble_to_stream.clear();
    if (this->enable_think && tokens.size() >= 2)
        this->preamble_to_stream.assign(tokens.end() - 2, tokens.end());

    bool success = has_images
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = qwen3_8mtp_engine->checkpoint();
    return success;
}

std::string Qwen3_8MTP::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);

    // The think preamble is already in the KV cache and already in
    // token_history: insert() prefilled it with the rest of the prompt
    // instead of trimming it off for this function to re-feed a token at a
    // time. All that is left here is the text, and only when thinking is on.
    //
    // No profiler is touched: there is no model work left to charge, and
    // _speculative_generate() resets DECODING_TIME and TKOEN_DECODE_TIME on entry
    // anyway -- which is what used to silently discard the four forward()
    // calls this block has replaced, so they never appeared in "Decoding
    // time" and the 13 s they cost showed up only as a hole in "Total time".
    for (int id : this->preamble_to_stream) {
        std::string token_str = this->tokenizer->run_time_decoder(id);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }

    // Own loop, not AutoModel::_shared_generate: the draft head commits
    // several tokens per step, which the shared loop can't express.
    result += this->_speculative_generate(meta_info, length_limit, os, is_cancelled);

    // Prints only if a cycle actually ran -- silent for a checkpoint with no
    // MTP head. Worth printing at all because a broken draft path is
    // otherwise invisible: verify overrides every rejected draft with the
    // base model's own argmax, so bad drafting costs only speed, not
    // correctness. The newline guards against running on from the last
    // streamed token when log_raw_output is off.
    //
    // speculation_stats() rather than report_speculation_stats(): same text,
    // but printing it here lets the runtime name the window it covers. The
    // counters now reset per session (clear_context()), while the "Decoding
    // time" in show_profile() resets per turn -- two different windows, on
    // purpose, and the engine cannot label either because it does not know
    // what a session is.
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        if (mtp->speculation_cycles() > 0) {
            if (!this->log_raw_output) std::cout << std::endl;
            header_print("FLM", mtp->speculation_stats() + ".");
        }
    }

    return result;
}

std::string Qwen3_8MTP::_speculative_generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);

    stop_reason_t reason = EOT_DETECTED;

    // Always on. Acceptance is an exact compare against the base model's
    // argmax, so this is only truly correct under greedy sampling -- if the
    // sampler is later changed (via /set or REST params) speculation still
    // runs and silently produces greedy output instead. A checkpoint with no
    // MTP head is unaffected: speculate() just returns {} every step and
    // everything falls through to the ordinary path below.
    const bool spec_enabled = true;
    // Draft depth; also the hit-rate denominator (see report_speculation_stats).
    const int SPEC_MAX_DRAFT = 7;
    if (spec_enabled) {
        header_print("FLM", "Speculative decoding enabled (MTP draft head).");
    }

    int last_sampled_token = this->last_token;
    this->token_history.push_back(this->last_token);
    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(last_sampled_token)){
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
        return result;
    }

    // Commit one token: stream it, record it, test the stop rules. Shared by
    // both branches so a batch can't drift from the single-token path and
    // stream past a stop token.
    auto consume = [&](int token) -> bool {
        this->total_tokens++;
        last_sampled_token = token;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(token)){ // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(token);
        if (this->is_eos(token)){
            meta_info.generated_tokens++;
            // Correct on BOTH paths only because the engine was given the same
            // eos list: a speculative batch is truncated one past its stop
            // token, so an eos out of a batch is always its last element, and
            // the last element is the one token a batch leaves without a KV
            // row. Without that truncation an eos could land mid-batch, where
            // it is already committed, and this forward() would append a
            // second copy of it -- a wasted 64-layer pass and a duplicate row.
            if (this->forward_on_eos) {
                this->lm_engine->forward(token);
            }
            return false;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
            reason = MAX_LENGTH_REACHED;
            return false;
        }
        return this->total_tokens < this->MAX_L;
    };

    while (this->total_tokens < this->MAX_L){
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            // reset stream content
            buffer_.clear();
            current_mode_ = StreamEventType::CONTENT;
            tool_name_.clear();
            is_in_tool_block_ = false;
            break;
        }

        if (spec_enabled) {
            this->profiler_list[DECODING_TIME].start();
            std::vector<int> accepted =
                this->lm_engine->speculate(last_sampled_token, SPEC_MAX_DRAFT);
            // Charge the cycle to however many tokens came out of it, so
            // tok/s stays comparable with the non-speculative path.
            this->profiler_list[DECODING_TIME].stop(
                accepted.empty() ? 1 : (int)accepted.size());

            // Cycle 1 primes the draft head's KV cache over the prompt --
            // prefill-shaped work that ran inside speculate(), so move its
            // time from decode to prefill. 0 on every later cycle; the engine
            // tracks this since only it knows when a context clear starts a
            // new cycle 1.
            if (const uint64_t prime_us =
                    this->lm_engine->last_speculation_prime_us()) {
                this->profiler_list[DECODING_TIME].add_time(-(int64_t)prime_us);
                this->profiler_list[PREFILL_TIME].add_time((int64_t)prime_us);
            }

            if (!accepted.empty()) {
                // Already committed to the engine's caches -- must not be
                // re-fed through forward(). eos mid-batch stops here.
                bool go_on = true;
                for (int tok : accepted) {
                    if (!(go_on = consume(tok))) break;
                }
                if (!go_on) break;
                continue;
            }
            // Empty means the engine declined this step (head not primed, no
            // headroom under MAX_L). Fall through to the ordinary path.
        }

        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        this->_apply_tool_choice_mask(y, meta_info);
        int sampled_token = this->sampler->sample(y);
        this->profiler_list[SAMPLING_TIME].stop(1);

        if (!consume(sampled_token)) break;
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    if (this->log_raw_output) {
        std::cout << std::endl;
        header_print("FLM", "Model RAW Output: \n" + result);
    }
    return result;
}

/// \brief the shared profile, plus the MTP phase breakdown when it applies
/// \note Decoding here has three phases with unrelated costs -- k serial
///       draft steps, one batched verify, and an occasional rollback -- so
///       the engine measures them separately and this appends the result;
///       the runtime profiler only sees one speculate() call per cycle.
/// \note Exception: the "Prime" row (step 0 of the first cycle, the draft
///       head absorbing the prompt) is prefill work, and
///       _speculative_generate() already moved it from Decoding time to
///       Prefill time -- so unlike the other rows, it isn't double-counted
///       against the row above.
std::string Qwen3_8MTP::show_profile() {
    // Base first, so the rows every model shares stay in one place and cannot
    // drift from AutoModel's.
    std::string ss = this->AutoModel::show_profile();

    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        // Empty for a checkpoint with no MTP head -- keeps that session's
        // profile byte-identical to before.
        ss += mtp->speculation_timing();
    }
    return ss;
}

/// \brief the session boundary: base reset, plus the MTP speculation counters
/// \note The engine counts drafts and phase time cumulatively since the .so was
///       loaded. Its own clear_context() resets speculation STATE -- the draft
///       head's KV, mtp_hist, spec_prime_pending -- but deliberately not the
///       STATISTICS, because "a session" is not a concept it has. This is the
///       one place in the runtime that does, which is why the policy is here.
/// \note Reached from every session start, including the automatic one:
///       _shared_insert() calls clear_context() unqualified on a changed system
///       prompt, so that dispatches virtually and lands here too.
void Qwen3_8MTP::clear_context() {
    // Base first. It resets profiler_list and calls the engine's clear_context(),
    // so the spec counters end up zeroed in the same step as every other
    // per-session number rather than in a second place that can drift from it.
    //
    // Ordering is not load-bearing: reset_speculation_stats() deliberately
    // leaves spec_prime_pending alone (it is live state, not a statistic), so
    // it cannot undo the disarm the base just did.
    this->AutoModel::clear_context();

    // Non-virtual on the engine and already exported by the shipped
    // libqwen3_8mtp_npu.so, so this costs no rebuild and adds no vtable slot.
    // The cast also guards the no-MTP-head case -- a checkpoint without the
    // mtp.* tensors is still a qwen3_8mtp_npu, and zeroing counters that never
    // moved is harmless.
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        mtp->reset_speculation_stats();
    }
}

std::string Qwen3_8MTP::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_speculative_generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult Qwen3_8MTP::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    // Qwen3.5-style tool syntax:
    //   <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";
    std::string func_end_tag = "</function>";
    std::string func_open = "<function=";
    std::string param_open = "<parameter=";
    std::string param_close = "</parameter>";

    auto trim_tool_value = [](std::string value) {
        while (!value.empty() && (value.front() == '\n' || value.front() == '\r' || value.front() == ' ' || value.front() == '\t')) {
            value.erase(0, 1);
        }
        while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        return value;
    };

    const std::string think_start_tag = "<think>";
    const std::string think_end_tag = "</think>";
    size_t think_start_pos = response_text.find(think_start_tag);
    size_t think_end_pos = response_text.find(think_end_tag);
    bool is_reasoning = (think_start_pos != std::string::npos && think_end_pos != std::string::npos);
    if (is_reasoning) {
        size_t start = think_start_pos + think_start_tag.length();
        result.reasoning_content = response_text.substr(start, think_end_pos - start);
    }

    size_t search_from = 0;

    while (true) {
        size_t start_pos = response_text.find(start_tag, search_from);
        if (start_pos == std::string::npos) break;

        size_t block_content_start = start_pos + start_tag.length();
        size_t end_pos = response_text.find(end_tag, block_content_start);

        size_t block_end;
        if (end_pos != std::string::npos) {
            block_end = end_pos;
            search_from = end_pos + end_tag.length();
        } else {
            // Unclosed tag — search for </function> fallback
            size_t func_end_pos = response_text.find(func_end_tag, block_content_start);
            if (func_end_pos != std::string::npos) {
                block_end = func_end_pos + func_end_tag.length();
            } else {
                block_end = response_text.length();
            }
            search_from = block_end;
        }

        std::string block = response_text.substr(block_content_start, block_end - block_content_start);

        std::string tool_name;
        size_t func_start = block.find(func_open);
        if (func_start != std::string::npos) {
            func_start += func_open.length();
            size_t func_name_end = block.find(">", func_start);
            if (func_name_end != std::string::npos) {
                tool_name = block.substr(func_start, func_name_end - func_start);
            }
        }

        nlohmann::json args = nlohmann::json::object();
        size_t pos = 0;

        while (true) {
            size_t param_start = block.find(param_open, pos);
            if (param_start == std::string::npos) break;

            param_start += param_open.length();
            size_t param_name_end = block.find(">", param_start);
            if (param_name_end == std::string::npos) break;

            std::string param_name = block.substr(param_start, param_name_end - param_start);
            size_t value_start = param_name_end + 1;
            size_t value_end = block.find(param_close, value_start);

            size_t next_param_pos = block.find(param_open, value_start);
            size_t func_boundary_pos = block.find(func_end_tag, value_start);

            auto use_earlier_boundary = [&value_end](size_t boundary_pos) {
                if (boundary_pos != std::string::npos && (value_end == std::string::npos || boundary_pos < value_end)) {
                    value_end = boundary_pos;
                }
            };

            use_earlier_boundary(next_param_pos);
            use_earlier_boundary(func_boundary_pos);

            if (value_end == std::string::npos) {
                value_end = block.length();
            }

            std::string param_value = trim_tool_value(block.substr(value_start, value_end - value_start));

            try {
                args[param_name] = nlohmann::json::parse(param_value);
            }
            catch (...) {
                args[param_name] = param_value;
            }

            pos = value_end;
            if (block.compare(value_end, param_close.length(), param_close) == 0) {
                pos += param_close.length();
            }
        }

        result.tool_calls_list.emplace_back(tool_name, args.dump());
    }

    if (result.tool_calls_list.empty()) {
        size_t content_start = is_reasoning ? think_end_pos + think_end_tag.length() : 0;
        result.content = response_text.substr(content_start);
    } else {
        // Populate legacy single-tool fields from the first call for backward compatibility
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
        // Extract content before the first <tool_call>
        size_t first_tool = response_text.find(start_tag);
        size_t content_start = is_reasoning ? think_end_pos + think_end_tag.length() : 0;
        if (first_tool != std::string::npos && first_tool > content_start) {
            result.content = trim_tool_value(response_text.substr(content_start, first_tool - content_start));
        }
    }

    return result;
}

// Stream
StreamResult Qwen3_8MTP::parse_stream_content(const std::string content) {
    return parse_stream_content_impl(content, false);
}

StreamResult Qwen3_8MTP::parse_stream_content_final(const std::string content) {
    return parse_stream_content_impl(content, true);
}

StreamResult Qwen3_8MTP::parse_stream_content_impl(const std::string content, bool is_final) {
    const std::string MARKER_THINK_START = "<think>";
    const std::string MARKER_THINK_END = "</think>";
    const std::string MARKER_TOOL_START = "<tool_call>";
    const std::string MARKER_TOOL_END = "</tool_call>";
    const std::string MARKER_FUNC_END = "</function>";


    StreamResult result;
    buffer_ += content;

    while (true) {
        if (!is_in_tool_block_) {
            size_t stray_end_pos = buffer_.find(MARKER_TOOL_END);
            if (stray_end_pos != std::string::npos) {
                buffer_.erase(stray_end_pos, MARKER_TOOL_END.length());
            }
        }

        if (!is_in_tool_block_) {
            size_t tool_start_pos = buffer_.find(MARKER_TOOL_START);
            if (tool_start_pos != std::string::npos) {
                if (tool_start_pos > 0) {
                    result.content = buffer_.substr(0, tool_start_pos);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(tool_start_pos);
                    return result;
                }

                is_in_tool_block_ = true;
                buffer_ = buffer_.substr(MARKER_TOOL_START.length());
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        // tool calling process
        if (is_in_tool_block_) {
            size_t tool_end_pos = buffer_.find(MARKER_TOOL_END);
            size_t func_end_pos = buffer_.find(MARKER_FUNC_END);

            if (tool_end_pos != std::string::npos || func_end_pos != std::string::npos || (is_final && !buffer_.empty())) {
                size_t actual_end_pos = buffer_.size();
                size_t skip_length = 0;

                if (tool_end_pos != std::string::npos) {
                    actual_end_pos = tool_end_pos;
                    skip_length = MARKER_TOOL_END.length();
                }
                else if (func_end_pos != std::string::npos) {
                    actual_end_pos = func_end_pos;
                    skip_length = MARKER_FUNC_END.length();
                }

                std::string block = buffer_.substr(0, actual_end_pos + skip_length);
                buffer_ = buffer_.substr(actual_end_pos + skip_length);
                is_in_tool_block_ = false;

                try {
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));

                    // parse function name
                    std::string func_open = "<function=";
                    size_t func_start = block.find(func_open);
                    if (func_start != std::string::npos) {
                        func_start += func_open.length();
                        size_t func_end = block.find(">", func_start);
                        if (func_end != std::string::npos) {
                            result.tool_name = block.substr(func_start, func_end - func_start);
                        }
                    }

                    // parse parameters
                    nlohmann::json args = nlohmann::json::object();
                    std::string param_open = "<parameter=";
                    std::string param_close = "</parameter>";
                    size_t search_pos = 0;

                    while (true) {
                        size_t p_start = block.find(param_open, search_pos);
                        if (p_start == std::string::npos) break;
                        p_start += param_open.length();
                        size_t p_name_end = block.find(">", p_start);
                        if (p_name_end == std::string::npos) break;
                        std::string param_name = block.substr(p_start, p_name_end - p_start);

                        size_t val_start = p_name_end + 1;
                        if (val_start < block.size() && block[val_start] == '\n') val_start++;

                        size_t param_close_pos = block.find(param_close, val_start);
                        size_t val_end = param_close_pos;

                        size_t next_param_pos = block.find(param_open, val_start);
                        size_t func_boundary_pos = block.find(MARKER_FUNC_END, val_start);
                        size_t tool_boundary_pos = block.find(MARKER_TOOL_END, val_start);

                        auto use_earlier_boundary = [&val_end](size_t boundary_pos) {
                            if (boundary_pos != std::string::npos && (val_end == std::string::npos || boundary_pos < val_end)) {
                                val_end = boundary_pos;
                            }
                        };

                        use_earlier_boundary(next_param_pos);
                        use_earlier_boundary(func_boundary_pos);
                        use_earlier_boundary(tool_boundary_pos);

                        if (val_end == std::string::npos && is_final) {
                            val_end = block.size();
                        }
                        if (val_end == std::string::npos) break;

                        std::string param_value = block.substr(val_start, val_end - val_start);

                        // Enhanced trim: handle multiple newlines or spaces that the model may generate after a parameter
                        while(!param_value.empty() && (param_value.back() == '\n' || param_value.back() == '\r' || param_value.back() == ' ')) {
                            param_value.pop_back();
                        }

                        try {
                            // Try to parse as native JSON type (Integer, Float, Boolean, Array, Object)
                            args[param_name] = nlohmann::json::parse(param_value);
                        }
                        catch (...) {
                            args[param_name] = param_value;
                        }

                        search_pos = param_close_pos != std::string::npos && val_end == param_close_pos
                            ? val_end + param_close.length()
                            : val_end;
                    }
                    result.tool_args_str = args.dump();
                    return result;
                }
                catch (...) {
                    result.type = StreamEventType::CONTENT;
                    result.content = "[Error parsing tool call]";
                    return result;
                }
            }
            else {
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        if (current_mode_ == StreamEventType::CONTENT) {
            size_t think_start_pos = buffer_.find(MARKER_THINK_START);
            if (think_start_pos != std::string::npos) {
                if (think_start_pos > 0) {
                    result.content = buffer_.substr(0, think_start_pos);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(think_start_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        }
        else if (current_mode_ == StreamEventType::REASONING) {
            size_t think_end_pos = buffer_.find(MARKER_THINK_END);
            if (think_end_pos != std::string::npos) {
                if (think_end_pos > 0) {
                    result.content = buffer_.substr(0, think_end_pos);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(think_end_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        if (!buffer_.empty()) {
            size_t last_lt = buffer_.rfind('<');
            // If '<' appears at the end (possibly an incomplete <tool_call> or <think> tag)
            if (last_lt != std::string::npos && (buffer_.length() - last_lt) <= 15) {
                if (last_lt > 0) {
                    // Only output the content before '<'
                    result.content = buffer_.substr(0, last_lt);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(last_lt);
                    return result;
                } else {
                    // If '<' is the first character in the buffer, directly wait for the next chunk
                    result.type = StreamEventType::WAITING;
                    return result;
                }
            }

            result.content = buffer_;
            result.type = current_mode_;
            buffer_.clear();
            return result;
        }

        break;
    }

    result.type = current_mode_;
    return result;
}
