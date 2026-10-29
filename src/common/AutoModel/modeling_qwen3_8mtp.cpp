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

void Qwen3_8MTP::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);

    // After setup_tokenizer, which is what fills eos_token_ids -- the engine
    // has no tokenizer and would otherwise run a speculative batch straight
    // past the stop token. Both sides must test the same list or they
    // disagree about where a batch ends: _speculative_generate() stops at the
    // first id that is_eos() accepts, and every token the engine committed
    // after that one is a cache row with nothing in token_history to describe
    // it. Giving the engine the list moves the cut upstream of the caches.
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine))
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
    // if (auto* eng = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine)) {
    //     if (eng->has_vision_tower())
    //         header_print("FLM", "vision backend: " << eng->vision_backend());
    // }

    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }

    // Prefix slots, phase 1: the shared head (system prompt + tools) of a text conversation. A snapshot is ~157 MB of
    // recurrent state plus 64 KB (K/V) and 20 KB (the head's hidden history) per token, so no end-of-prompt snapshots
    // and smaller default limits than the other models; a conversation that continues keeps using the single checkpoint.
    this->prefix_cache_.boundary_only = true;
    this->prefix_cache_.default_mib   = 1536;
    this->prefix_cache_.default_slots = 4;
    this->_prefix_attach(this->lm_engine, qwen3_8mtp_state_api(this->lm_engine), 248056); // 248056 = <|image_pad|>
}

void Qwen3_8MTP::setup_tokenizer(std::string model_path) {
    // tokenizer_config.json lists eos_token_id as [248044, 248046, 248048];
    // _shared_setup_tokenizer registers all of them. config.json alone would
    // only give 248044, which the chat template never emits -- using just
    // that one causes runaway generation.
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_8MTP::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message);
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

    qwen3_8mtp_npu* qwen3_8mtp_engine = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine);

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
        templated_text = this->apply_chat_template(ordered_messages, input.tools, meta_info.continue_final_message);
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

    // The single checkpoint (the conversation that ran last) covers this prompt when its token history is a prefix of it: the
    // conversation continues, restore() rewinds to the end of the last turn and only the new turn is prefilled.
    // Otherwise (the first turn, or another conversation ran in between) a text prompt resumes from the slot that holds its
    // shared head (system prompt + tools), if there is one, and prefills the rest. The saved head is a self-contained
    // copy, so the engine moving on does not invalidate it.
    const size_t think_trim = this->enable_think ? 2 : 4;
    const bool checkpoint_covers = !checkpoint_his.empty() && checkpoint_his.size() <= tokens.size() &&
                                   std::equal(checkpoint_his.begin(), checkpoint_his.end(), tokens.begin());
    const bool slot_mode = !checkpoint_covers && !has_images &&
                           std::find(tokens.begin(), tokens.end(), image_soft_token_id) == tokens.end() &&
                           this->_prefix_slot_mode(meta_info, tokens.size(), think_trim);
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    if (slot_mode) {
        full_tokens.assign(tokens.begin(), tokens.end() - think_trim);
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, {});
    }
    else if (meta_info.restore_allowed && !checkpoint_his.empty()) {
        restore_idx = this->lm_engine->restore();
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
    //
    // A continued message has no generation prompt: its last tokens are the
    // client's own text, which the output must not repeat.
    this->preamble_to_stream.clear();
    if (this->enable_think && !meta_info.continue_final_message && tokens.size() >= 2)
        this->preamble_to_stream.assign(tokens.end() - 2, tokens.end());

    if (slot_mode) {
        const bool inserted = this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, {}, match_len, "<|im_start|>user",
                                                   nullptr, 0, is_cancelled, input.requested_max_new_tokens);
        if (!inserted) {
            checkpoint_his.clear();   // cancelled half way: the engine holds part of a prompt
            return false;
        }
        // the conversation continues from here: take the single checkpoint as well
        checkpoint_his = token_history;
        this->lm_engine->checkpoint();
        return true;
    }

    bool success = has_images
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = this->lm_engine->checkpoint();
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
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine)) {
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
    // Off under a decoding constraint: speculate() commits the drafts the base
    // model agrees with, so they would never pass the constraint's mask.
    // FLM_DISABLE_SPECULATION=1 turns it off too, for comparing against plain
    // decoding (output, logprobs) on the same build.
    const char* no_spec = std::getenv("FLM_DISABLE_SPECULATION");
    const bool spec_enabled = this->lm_engine->supports_speculation() && !meta_info.constraint &&
        !(no_spec != nullptr && no_spec[0] != '\0' && std::strcmp(no_spec, "0") != 0);
    // Draft depth; also the hit-rate denominator (see report_speculation_stats).
    const int SPEC_MAX_DRAFT = 7;
    if (spec_enabled) {
        header_print("FLM", "Speculative decoding enabled (MTP draft head).");
    }

    int last_sampled_token = this->last_token;
    this->token_history.push_back(this->last_token);
    // the sample insert() took is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
        std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(last_sampled_token)){
        // Sampled from the prefill, so no engine has run it (as in _shared_generate).
        this->token_history.pop_back();
        meta_info.stop_reason = reason;
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();

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
            } else {
                // The engine never ran it, so the next prompt has to carry it.
                this->token_history.pop_back();
            }
            return false;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
            reason = MAX_LENGTH_REACHED;
            return false;
        }
        // No MAX_L test here: a batch's tokens before its last are already in
        // the caches, so stopping mid-batch would leave rows token_history
        // does not describe. The loop tests MAX_L between steps instead.
        return true;
    };

    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Tested here rather than in the loop condition so the stop is
        // recorded as a truncation, as in _shared_generate.
        if (this->total_tokens >= this->MAX_L){
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }

        // A batch must not run past length_limit: speculate() returns up to
        // max_draft + 1 tokens, all but the last already committed, so capping
        // the batch at the tokens left keeps the limit on the last one -- the
        // one token a stop can drop. With one token left, decode it plainly.
        const int tokens_left = (length_limit > 0)
            ? length_limit - meta_info.generated_tokens
            : SPEC_MAX_DRAFT + 1;
        if (spec_enabled && tokens_left > 1) {
            this->profiler_list[DECODING_TIME].start();
            std::vector<int> accepted =
                this->lm_engine->speculate(last_sampled_token,
                                           std::min(SPEC_MAX_DRAFT, tokens_left - 1));
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
                // These never went through _sample(), so their logprobs come
                // from the verify pass that accepted them: row i is the
                // distribution accepted[i] is the argmax of.
                const float* rows = nullptr;
                int row_count = 0;
                int vocab = 0;
                if (meta_info.logprobs_top_n >= 0) {
                    auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine);
                    rows = mtp ? mtp->last_speculation_logits(row_count, vocab) : nullptr;
                    if (!rows || row_count < (int)accepted.size()) {
                        throw std::runtime_error("logprobs: the engine returned " +
                            std::to_string(row_count) + " logits rows for " +
                            std::to_string(accepted.size()) + " speculated tokens");
                    }
                }
                // Already committed to the engine's caches -- must not be
                // re-fed through forward(). eos mid-batch stops here.
                bool go_on = true;
                for (size_t i = 0; i < accepted.size(); i++) {
                    const int tok = accepted[i];
                    if (rows) {
                        // before consume() writes the token: its entry has to
                        // exist by the flush that streams it
                        const float* row = rows + i * (size_t)vocab;
                        sample_logprobs_t raw;
                        raw.top_n = meta_info.logprobs_top_n;
                        raw_logprobs(row, vocab, raw);
                        this->_record_logprob(meta_info, tok, row[tok], raw);
                    }
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
        int sampled_token = this->_sample(y, meta_info);
        this->profiler_list[SAMPLING_TIME].stop(1);

        if (!consume(sampled_token)) break;
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }
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

    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine)) {
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
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine)) {
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
