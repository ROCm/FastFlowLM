/// \file modeling_qwen3_8mtp.cpp
/// \brief Qwen3_8MTP class
/// \author FastFlowLM Team
/// \date 2026-09-16
/// \version 0.9.28
/// \note AutoModel wrapper for Qwen3.8-27B. Multimodal when the checkpoint
///       ships vision_weight.q4nx; the image preprocessing itself lives in
///       modeling_qwen3_8mtp_image.cpp.
/// \note Speculative decode IS wired up: causal_lm carries the
///       supports_speculation()/speculate() hooks and AutoModel::
///       _shared_generate() drives them under a greedy sampler.
/// \note generate() therefore delegates its decode loop to _shared_generate()
///       and keeps only the think-preamble replay. It previously carried a
///       private copy of the loop that called forward() + sample() directly,
///       which left speculation dead on the main chat path -- correct output,
///       no speedup, and a hit rate that never printed because no cycle ran.

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
    this->sampler.reset();

    this->enable_tool = true;

    // Qwen3.5-family recommended decoding defaults; the test harness overrides
    // these with set_topk(1) when it wants a reproducible stream.
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);

    // Image-processor geometry. patch_size / spatial_merge_size /
    // temporal_patch_size are stated by config.json's vision_config and are
    // read from it; the rest live in preprocessor_config.json, which the NPU2
    // checkpoint does not ship, so they fall back to the values that file
    // carries upstream for Qwen3.8-27B. A wrong factor here does not fail --
    // it produces a grid the merger regroups across, which reads as a slightly
    // confused caption.
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
    if (auto* eng = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        if (eng->has_vision_tower())
            header_print("FLM", "vision backend: " << eng->vision_backend());
    }

    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Qwen3_8MTP::setup_tokenizer(std::string model_path) {
    // tokenizer_config.json already lists eos_token_id as
    // [248044 <|endoftext|>, 248046 <|im_end|>, 248048], and
    // _shared_setup_tokenizer registers every entry of that array. config.json
    // alone would only give 248044, which the chat template never emits --
    // that mismatch is the runaway-generation failure mode.
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

    // <|image_pad|>, verified against tokenizer.json and against config.json's
    // image_token_id. The engine reads the same id out of its own config, so a
    // disagreement here surfaces as a span-count mismatch and a throw, not as
    // a scrambled prompt.
    constexpr int image_soft_token_id = 248056;

    qwen3_8mtp_npu* qwen3_8mtp_engine = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get());

    // A checkpoint without vision_weight.q4nx (or with FLM_Q38_VISION=0) has no
    // tower, so an image would be templated into <|vision_start|><|image_pad|>
    // ... placeholders that nothing ever fills -- the prompt would prefill
    // garbage rather than fail. Drop them loudly in that case only.
    const bool vision_ok = qwen3_8mtp_engine && qwen3_8mtp_engine->has_vision_tower();
    if (!vision_ok && !input.images.empty()) {
        header_print("WARNING", "Qwen3.8-27B is loaded without a vision tower; ignoring "
                                << input.images.size() << " image(s)");
        input.images.clear();
    }
    // Audio is not ported at all: the checkpoint carries no audio tower and
    // there is no audio path in this engine.
    if (!input.audios.empty()) {
        header_print("WARNING", "Qwen3.8-27B has no audio path; ignoring "
                                << input.audios.size() << " audio clip(s)");
        input.audios.clear();
    }

    // ----------------------------------------------------------------------
    // Decode and preprocess every image BEFORE templating, so that an image
    // that fails to load never gets a placeholder. If it did, the payload
    // would fall out of step with the placeholders and the engine would
    // splice image n's rows into image n+1's slots -- which every downstream
    // stage accepts.
    //
    // `pixels` is one contiguous fp32 buffer for the whole prompt and it grows
    // as images are appended, so the engine-facing pointers cannot be taken
    // until every image is in. `pixel_offsets` records where each one landed.
    // ----------------------------------------------------------------------
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
            // The template produced fewer placeholders than we staged images
            // for, so some image's pixels have no slots to land in. Refusing is
            // the only safe answer: prefilling would put image n's rows into
            // image n+1's positions and read as a mildly confused caption.
            header_print("ERROR", "templated " << image_counter << " image placeholder(s) "
                                  "for " << host_images.size() << " image(s); refusing to prefill");
            return false;
        }
        header_print("FLM", "Total images: " << host_images.size()
                            << " (" << total_image_tokens << " image tokens)");
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // ----------------------------------------------------------------------
    // Prompt-cache aware image alignment.
    //
    // AutoModel::_shared_insert prefix-matches `tokens` against
    // `checkpoint_his` over the FULL length of checkpoint_his, and erases that
    // prefix before prefilling only if every token of it matches. We must NOT
    // erase `tokens` here -- _shared_insert needs the untrimmed sequence to
    // run that very check. What does need fixing up locally is the payload,
    // which holds pixels for the WHOLE prompt including images already in the
    // cache from earlier turns: drop the fully-cached leading images so the
    // survivors line up with the image tokens that survive the erase.
    // ----------------------------------------------------------------------
    size_t prefix_skip_count = 0;
    if (!host_images.empty()) {
        const size_t idx = this->checkpoint_his.size();
        for (size_t i = 0; i < idx; i++) {
            if (i < tokens.size() && tokens[i] == this->checkpoint_his[i]) prefix_skip_count++;
            else break;
        }
        // Must match the entirety of checkpoint_his, otherwise _shared_insert
        // clears the context and skips nothing.
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
                // The pixels themselves are NOT erased. pixel_offsets are
                // absolute indices into `pixels`, so dropping the descriptors
                // is enough and erasing the front of a 400 MB buffer to save
                // nothing would be the only cost here.
                host_images.erase(host_images.begin(),
                                  host_images.begin() + images_to_drop);
                pixel_offsets.erase(pixel_offsets.begin(),
                                    pixel_offsets.begin() + images_to_drop);
                header_print("FLM", "Prompt-cache hit: dropped " << images_to_drop
                                    << " cached image(s) from payload");
            }
        }
    }

    // The last image token's index, expressed relative to the tokens that will
    // SURVIVE _shared_insert's prefix erase. _chunked_insert hands the payload
    // to chunk 0 only, so this is what grows chunk 0 to cover every image row;
    // without it a second chunk would carry image tokens and no images, and
    // the engine would throw.
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
    //   thinking on : "<think>" "\n"                      -> 2 tokens
    //   thinking off: "<think>" "\n\n" "</think>" "\n\n"   -> 4 tokens
    //
    // These used to be trimmed off here so generate() could re-feed them one
    // at a time, which cost one full 64-layer weight stream per token --
    // forward() is _prefill_with_mm() on a one-row batch, and that batch's
    // cost is the ~14 GB of packed weights, not the row. Measured at ~3.3 s
    // each on this model, so the four-token thinking-off preamble was ~13 s
    // of a 24 s short run, none of it visible in any printed timer.
    //
    // They are constants the tokenizer knew before the run started, so they
    // ride in the prefill batch as 2-4 extra rows of a pass that was
    // happening anyway. Two consequences beyond the time:
    //   - insert()'s own sample() now seeds `last_token` from the row after
    //     the last preamble token, which is the first answer token. It used
    //     to sample the row before the preamble and have generate() throw the
    //     result away.
    //   - `last_window_len` is left at the prompt length instead of 1, so the
    //     MTP head's step-0 catch-up is no longer clamped to a single row and
    //     the head starts the first cycle primed rather than cold.
    //
    // Thinking on still has to put "<think>\n" on the screen -- it opens the
    // reasoning block the user reads -- so record those ids for generate() to
    // echo without forwarding. Thinking off streams nothing: its four tokens
    // exist only to close a block that was never opened, and the old code
    // decoded each one into a `token_str` it then dropped.
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
    // _shared_generate() resets DECODING_TIME and TKOEN_DECODE_TIME on entry
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

    // Hand the decode loop to the base class rather than running a private
    // copy of it. This function used to carry its own while() calling
    // forward() + sample() directly, which meant the whole speculative path --
    // and with it the MTP hit rate -- was dead on the main chat path: only
    // generate_with_prompt() and the milestone driver ever reached
    // speculate(). The preamble above is the only part that is genuinely
    // specific to this model, so it stays; the loop is not.
    //
    // _shared_generate() seeds from this->last_token, emits it, then feeds it
    // to the model. That seed is insert()'s own sample of the prefill's last
    // row -- which, now that the preamble is prefilled too, is the row that
    // predicts the first answer token. Nothing to re-assign here.
    result += this->_shared_generate(meta_info, length_limit, os, is_cancelled);

    // The engine counts every draft/verify cycle, so this prints whenever
    // speculation actually ran, and prints nothing otherwise -- a checkpoint
    // without an MTP head, or a non-greedy sampler, stays silent.
    //
    // Worth printing at all because a broken draft path is invisible in the
    // text: verify overrides every rejected draft with the base model's own
    // argmax, so bad drafting costs only speed.
    //
    // The newline is this function's own: header_print writes to std::cout
    // while tokens stream to `os`, so without it the line would run on from
    // the last token whenever log_raw_output is off (which is what supplies
    // the break today -- not something to depend on from here).
    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        if (mtp->speculation_cycles() > 0) {
            if (!this->log_raw_output) std::cout << std::endl;
            mtp->report_speculation_stats();
        }
    }

    return result;
}

/// \brief the shared profile, plus the MTP phase breakdown when it applies
/// \note "Decoding time" is a single number, but this model decodes in three
///       phases with unrelated cost structures: k serial one-layer draft steps,
///       one batched 64-layer verify over k+1 rows, and -- only on a rejection
///       -- a rollback and re-fold. Lumping them together hides the one thing
///       worth acting on, which is whether drafting is paying for itself.
/// \note The split cannot be made in AutoModel's profiler: from _shared_generate
///       a whole cycle is one speculate() call, so the phase boundary does not
///       exist there. The engine measures it and this appends the result.
/// \note One part of it does cross over. Step 0 of the first cycle after each
///       prefill is the draft head absorbing the prompt, which is prefill by
///       any honest reading, so the engine reports it through
///       last_speculation_prime_us() and _shared_generate moves those
///       microseconds from DECODING_TIME to PREFILL_TIME. The "Prime" row
///       below is therefore already OUT of the "Decoding time" above it --
///       the only row in the breakdown of which that is true.
std::string Qwen3_8MTP::show_profile() {
    // Base first, so the rows every model shares stay in one place and cannot
    // drift from AutoModel's.
    std::string ss = this->AutoModel::show_profile();

    if (auto* mtp = dynamic_cast<qwen3_8mtp_npu*>(this->lm_engine.get())) {
        // Empty when speculation never ran -- a non-greedy sampler, or a
        // checkpoint with no MTP head. Appending nothing then keeps those
        // sessions byte-identical to what they printed before.
        ss += mtp->speculation_timing();
    }
    return ss;
}

std::string Qwen3_8MTP::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
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
