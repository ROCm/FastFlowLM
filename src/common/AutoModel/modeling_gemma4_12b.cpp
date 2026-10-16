/// \file modeling_gemma4_12b.cpp
/// \brief Gemma4_12B class
/// \author FastFlowLM Team
/// \date 2026-08-25
/// \version 0.9.45
/// \note This is a source file for the Gemma4_12B class (text-only interface).

#include "AutoModel/modeling_gemma4_12b.hpp"
#include "metrices.hpp"
#include "error_measure.hpp"
#include <cassert>

/************              Gemma4_12B family            **************/
Gemma4_12B::Gemma4_12B(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma4_12B") {}

void Gemma4_12B::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {

    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);

    // The soft token budget is a preprocessing constant (processor_config.json
    // image_seq_length / image_processor.max_soft_tokens), read by the engine
    // along with the rest of the image front end parameters.
    {
        gemma4_12b_npu* engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine);
        if (engine != nullptr && engine->GEMMA4_12B_vision_max_soft_tokens > 0){
            this->image_softtoken_budget = (int)engine->GEMMA4_12B_vision_max_soft_tokens;
        }
    }
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 64;
    config.top_p = 0.95;
    config.min_p = 0.0;
    config.temperature = 1.0;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Gemma4_12B::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma4_12B::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    // The Gemma4 template emits unquoted keys (argument_needle:<|"|>...<|"|>), which minja's
    // capability probe does not recognize, so it would wrongly polyfill tool calls into a
    // generic JSON blob. The template handles tools/tool_calls/tool_responses natively.
    minja::chat_template_options opt;
    opt.polyfill_tools = false;
    opt.polyfill_tool_calls = false;
    opt.polyfill_tool_responses = false;
    opt.polyfill_object_arguments = false;

    // The template raises if tool_calls[].function.arguments is a string, but OpenAI-style
    // clients send it serialized. Deserialize it here (what polyfill_object_arguments would do).
    nlohmann::ordered_json normalized = messages;
    for (auto& message : normalized) {
        if (!message.is_object() || !message.contains("tool_calls")) continue;
        for (auto& tool_call : message["tool_calls"]) {
            if (!tool_call.is_object() || !tool_call.contains("function")) continue;
            auto& function = tool_call["function"];
            if (!function.is_object() || !function.contains("arguments")) continue;
            auto& arguments = function["arguments"];
            if (!arguments.is_string()) continue;
            auto raw = arguments.get<std::string>();
            if (raw.empty()) {
                arguments = nlohmann::ordered_json::object();
                continue;
            }
            auto parsed = nlohmann::ordered_json::parse(raw, nullptr, /* allow_exceptions= */ false);
            if (parsed.is_object()) arguments = parsed;
        }
    }

    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = normalized;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty())
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs, opt);
}

bool Gemma4_12B::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;

    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }


    gemma4_12b_image_payload_t image_payload;
    gemma4_12b_audio_payload_t audio_payload;
    audio_payload.num_audios = 0;
    image_payload.num_images = 0;

    // [input.audios.size()] -- how many 30 s chunks each input clip was split
    // into, so the chat template emits one <|audio|> placeholder per chunk.
    std::vector<int> audio_chunks_per_input;

    nlohmann::ordered_json gemma4_12b_message = nlohmann::ordered_json::array();
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        int total_images = 0;
        for (auto& message : input.messages) {
            if (!message.contains("images") && !message.contains("audios")) {
                gemma4_12b_message.push_back(message);
                continue;
            }

            nlohmann::ordered_json newContent = nlohmann::ordered_json::array();
            // Process Images
            if (message.contains("images")) {
                for (auto& img : message["images"]) {
                    std::string img_str = img.get<std::string>();

                    gemma4_12b_image_t image = this->load_image_base64(img_str);
                    if (image.width <= 0 || image.height <= 0) {
                        header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                        continue;
                    }
                    std::vector<bf16> pixel_values;
                    std::pair<int, int> patch_element_per_patch;
                    uint32_t valid_patch_size = 0;
                    uint32_t num_soft_tokens = 0;
                    std::vector<int> image_grid_pairs; 

                    preprocess_image(image, patch_element_per_patch, valid_patch_size, pixel_values, image_grid_pairs, num_soft_tokens);

                    image_payload.image_patch__element_per_patch.push_back(patch_element_per_patch);
                    image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
                    image_payload.pixel_values.push_back(pixel_values);
                    image_payload.image_grid_pairs_per_image.push_back(image_grid_pairs);
                    image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                    image_payload.num_images++;
                    total_images++;
                    newContent.push_back({{"type", "image"}, {"image", img}});
                }
        }
            // Process Audios
            if (message.contains("audios")) {
                gemma4_12b_npu *gemma4e_engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine); 
                for (auto& aud : message["audios"]) {
                    std::string audio_str = aud.get<std::string>();
                    audio_data_t audio_data = this->load_audio_base64(audio_str, gemma4e_engine->GEMMA4_12B_audio_sampling_rate, MonoDownmixMode::RMS);
                    if (audio_data.channels > 1) {
                        std::cerr << "only mono audio is supported." << std::endl;
                        exit(-1);
                    }

                    std::vector<std::vector<bf16>> audio_chunks;
                    std::vector<int> audio_chunk_frames;
                    extract_waveform_features_chunked(
                        audio_data,
                        gemma4e_engine->GEMMA4_12B_audio_samples_per_token,
                        gemma4e_engine->GEMMA4_12B_audio_max_soft_tokens,
                        audio_chunks,
                        audio_chunk_frames
                    );

                    if (audio_chunks.size() > 1) {
                        header_print_g("FLM", "Audio in message is split into " << audio_chunks.size() << " chunks for processing.");
                    }

                    for (size_t c = 0; c < audio_chunks.size(); c++) {
                        const int processed_num_frames = audio_chunk_frames[c];
                        audio_payload.num_audios++;
                        audio_payload.num_soft_tokens_per_audio.push_back(
                            processed_num_frames
                        );
                        audio_payload.mel_spectrograms.push_back(
                            std::move(audio_chunks[c])
                        );
                        audio_payload.mel_spectrogram_frames_per_audio.push_back(processed_num_frames);
                        audio_payload.mel_spectrogram_bins_per_audio.push_back(gemma4e_engine->GEMMA4_12B_audio_embed_dim);
                        newContent.push_back({{"type", "audio"}, {"audio", aud}});
                    }
                }
            }

            newContent.push_back({{"type", "text"}, {"text", message.value("content", "")}});

            nlohmann::ordered_json newItem = {
                {"role", message.value("role", "user")},
                {"content", newContent}
            };
            gemma4_12b_message.push_back(newItem);
        }
        header_print("FLM", "Total images: " << total_images);
    }
    else { // a pure text, usually from the cli
        // nlohmann::ordered_json messages;
        if(input.audios.size() > 0){
            gemma4_12b_npu *gemma4e_engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine);  

            for (int i = 0; i < input.audios.size(); i++) {
                std::string audio_str = input.audios[i];
             
                audio_data_t audio_data = this->load_audio(audio_str, gemma4e_engine->GEMMA4_12B_audio_sampling_rate, MonoDownmixMode::RMS); 
                
                if (audio_data.channels > 1) {
                    std::cerr << "only mono audio is supported, but got " << audio_data.original_channels << " channels. Please convert it to mono first." << std::endl;
                    exit(-1);
                }
                // A clip longer than the 750-soft-token (30 s) cap must be split
                // into several chunks; each chunk becomes its own audio entry in
                // the payload and gets its own <|audio> ... <audio|> block below.
                std::vector<std::vector<bf16>> audio_chunks;
                std::vector<int> audio_chunk_frames;
                extract_waveform_features_chunked(
                    audio_data,
                    gemma4e_engine->GEMMA4_12B_audio_samples_per_token,
                    gemma4e_engine->GEMMA4_12B_audio_max_soft_tokens,
                    audio_chunks,
                    audio_chunk_frames
                );

                if (audio_chunks.size() > 1) {
                    header_print("INFO", "Audio " << i << " is longer than "
                        << (gemma4e_engine->GEMMA4_12B_audio_max_soft_tokens
                            * gemma4e_engine->GEMMA4_12B_audio_samples_per_token
                            / (double)gemma4e_engine->GEMMA4_12B_audio_sampling_rate)
                        << " s; split into " << audio_chunks.size() << " chunks");
                }

                audio_chunks_per_input.push_back(static_cast<int>(audio_chunks.size()));

                for (size_t c = 0; c < audio_chunks.size(); c++) {
                    const int processed_num_frames = audio_chunk_frames[c];
                    audio_payload.num_audios++;
                    audio_payload.num_soft_tokens_per_audio.push_back(
                        processed_num_frames
                    );
                    audio_payload.mel_spectrograms.push_back(
                        std::move(audio_chunks[c])
                    );
                    audio_payload.mel_spectrogram_frames_per_audio.push_back(processed_num_frames);
                    audio_payload.mel_spectrogram_bins_per_audio.push_back(gemma4e_engine->GEMMA4_12B_audio_embed_dim);
                }
            }
        }

        if(input.images.size() > 0){
            for (const auto& img_str : input.images) {
                gemma4_12b_image_t image = this->load_image(img_str);

                std::vector<bf16> pixel_values;
                std::pair<int, int> patch_element_per_patch;
                uint32_t valid_patch_size = 0;
                uint32_t num_soft_tokens = 0;
                std::vector<int> image_grid_pairs; 

                preprocess_image(image, patch_element_per_patch, valid_patch_size, pixel_values, image_grid_pairs, num_soft_tokens);

                image_payload.image_patch__element_per_patch.push_back(patch_element_per_patch);
                image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
                image_payload.pixel_values.push_back(pixel_values);
                image_payload.image_grid_pairs_per_image.push_back(image_grid_pairs);
                image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                image_payload.num_images++;
            }
        }
    }
    

    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(gemma4_12b_message, input.tools);
    }
    else{

        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        
        for (int i = 0; i < input.images.size(); i++) {
            content["content"].push_back({{"type", "image"}, {"image", input.images[i]}});
        }
        for (int i = 0; i < input.audios.size(); i++) {
            // One placeholder per 30 s chunk: a clip that was split must produce
            // as many <|audio|> placeholders as there are payload entries, or the
            // expansion loop below leaves the trailing chunks unconsumed.
            const int n_chunks = (i < (int)audio_chunks_per_input.size())
                ? audio_chunks_per_input[i] : 1;
            for (int c = 0; c < n_chunks; c++) {
                content["content"].push_back({{"type", "audio"}, {"audio", input.audios[i]}}); // placeholder
            }
        }
        
        content["content"].push_back({{"type", "text"}, {"text", input.prompt}});
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }

    // update the tokens to include the image tokens
    std::vector<int> tokens;
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    int total_image_tokens = 0;
    for(int i = 0; i < image_payload.num_images; i++){
        total_image_tokens += image_payload.num_soft_tokens_per_image[i];
    }

    int total_audio_tokens = 0;
    for(int i = 0; i < audio_payload.num_audios; i++){
        total_audio_tokens += audio_payload.num_soft_tokens_per_audio[i];
    }

    tokens.reserve(tokens_init.size() + total_image_tokens + total_audio_tokens);

    // Expand each single placeholder into <boi> <soft>*n <eoi> (resp. audio),
    // keeping every other token -- including the interleaving order of images
    // and audios as the chat template emitted them.
    int image_counter = 0;
    int audio_counter = 0;
    for(int i = 0; i < tokens_init.size(); i++){
        if (tokens_init[i] == image_token_id && image_counter < (int)image_payload.num_images) {
            tokens.push_back(boi_token_id); // the first image soft token id, which is reserved for the model to identify the image position, the rest of the soft tokens for this image will be continuous following this id
            for (int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            tokens.push_back(eoi_token_id); // a separator token between images, not necessary but can help the model to better distinguish different images
            image_counter++;
        }
        else if (tokens_init[i] == audio_token_id && audio_counter < (int)audio_payload.num_audios) {
            tokens.push_back(boa_token_id); // the first audio soft token id, which is reserved for the model to identify the audio position, the rest of the soft tokens for this audio will be continuous following this id
            for (int j = 0; j < audio_payload.num_soft_tokens_per_audio[audio_counter]; j++) {
                tokens.push_back(audio_token_id);
            }
            tokens.push_back(eoa_token_id); // a separator token between audios, not necessary but can help the model to better distinguish different audios
            audio_counter++;
        }
        else {
            tokens.push_back(tokens_init[i]);
        }
    }
    assert(image_counter == (int)image_payload.num_images);
    assert(audio_counter == (int)audio_payload.num_audios);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // ----------------------------------------------------------------------
    // Prompt-cache aware multi-modal alignment.
    //
    // AutoModel::_shared_insert prefix-matches `tokens` against `checkpoint_his`
    // over the FULL length of `checkpoint_his`. If every token matches, it
    // erases that prefix before prefilling; otherwise it calls clear_context()
    // and skips nothing. We must NOT erase `tokens` here -- _shared_insert
    // needs the untrimmed sequence to run that very check. What we DO need
    // to fix up locally is the multi-modal payload (which carries
    // pixels/waveforms for the WHOLE prompt, including images/audios from
    // earlier turns that are already cached): drop the fully-cached leading
    // images/audios so the surviving payload aligns with the surviving soft
    // tokens after _shared_insert performs its own erase.
    // ----------------------------------------------------------------------
    size_t prefix_skip_count = 0;
    {
        const size_t idx = this->checkpoint_his.size();
        for (size_t i = 0; i < idx; i++) {
            if (i < tokens.size() && tokens[i] == this->checkpoint_his[i]) {
                prefix_skip_count++;
            } else {
                break;
            }
        }
        // Must match the entirety of checkpoint_his, otherwise _shared_insert
        // will clear the context and not skip anything.
        if (prefix_skip_count != idx) {
            prefix_skip_count = 0;
        }

        if (prefix_skip_count > 0 && (image_payload.num_images > 0 || audio_payload.num_audios > 0)) {
            // Count modal soft-tokens that landed inside the cached prefix.
            int skipped_image_tokens = 0;
            int skipped_audio_tokens = 0;
            for (size_t i = 0; i < prefix_skip_count; i++) {
                if (tokens[i] == image_token_id) skipped_image_tokens++;
                else if (tokens[i] == audio_token_id) skipped_audio_tokens++;
            }

            // Walk images in order; each image's soft-token block is either
            // entirely inside the cached prefix or not at all (chat-template
            // boundaries never split a single image's block).
            size_t images_to_drop = 0;
            {
                int consumed = 0;
                for (unsigned i = 0; i < image_payload.num_images; i++) {
                    const int n = static_cast<int>(image_payload.num_soft_tokens_per_image[i]);
                    if (consumed + n <= skipped_image_tokens) {
                        consumed += n;
                        images_to_drop++;
                    } else {
                        break;
                    }
                }
            }

            // Same for audios.
            size_t audios_to_drop = 0;
            {
                int consumed = 0;
                for (unsigned i = 0; i < audio_payload.num_audios; i++) {
                    const int n = static_cast<int>(audio_payload.num_soft_tokens_per_audio[i]);
                    if (consumed + n <= skipped_audio_tokens) {
                        consumed += n;
                        audios_to_drop++;
                    } else {
                        break;
                    }
                }
            }

            auto drop_front = [](auto& vec, size_t n) {
                if (n == 0) return;
                if (n >= vec.size()) { vec.clear(); return; }
                vec.erase(vec.begin(), vec.begin() + n);
            };

            if (images_to_drop > 0) {
                drop_front(image_payload.image_patch__element_per_patch, images_to_drop);
                drop_front(image_payload.valid_patch_size_per_image,     images_to_drop);
                drop_front(image_payload.pixel_values,                   images_to_drop);
                drop_front(image_payload.image_grid_pairs_per_image,     images_to_drop);
                drop_front(image_payload.num_soft_tokens_per_image,      images_to_drop);
                image_payload.num_images -= static_cast<unsigned>(images_to_drop);
                header_print("FLM",
                    "Prompt-cache hit: dropped " << images_to_drop
                    << " cached image(s) from payload");
            }

            if (audios_to_drop > 0) {
                drop_front(audio_payload.mel_spectrograms,                 audios_to_drop);
                drop_front(audio_payload.mel_spectrogram_frames_per_audio, audios_to_drop);
                drop_front(audio_payload.mel_spectrogram_bins_per_audio,   audios_to_drop);
                drop_front(audio_payload.num_soft_tokens_per_audio,        audios_to_drop);
                audio_payload.num_audios -= static_cast<unsigned>(audios_to_drop);
                header_print("FLM",
                    "Prompt-cache hit: dropped " << audios_to_drop
                    << " cached audio(s) from payload");
            }
        }
    }

    // find the last image token index, expressed relative to the tokens that
    // will SURVIVE _shared_insert's prefix erase (i.e. shifted by -prefix_skip_count).
    int last_image_token_index = -1;
    for (int i = static_cast<int>(prefix_skip_count); i < (int)tokens.size(); i++) {
        if ((tokens[i] == image_token_id || tokens[i] == boi_token_id)) {
            last_image_token_index = i - static_cast<int>(prefix_skip_count);
        }
    }
    last_image_token_index++; // plus the end of image tokens

    // hardware
    int restore_idx = -1;
    gemma4_12b_npu *gemma4_12b_engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine);

    if (meta_info.restore_allowed) {
        restore_idx = gemma4_12b_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    size_t n = tokens.size();
    tokens.resize(n - (this->enable_think ? 0 : 4));

    gemma4_12b_multi_modal_payload_t multi_modal_payload;
    multi_modal_payload.image_payload = image_payload;
    multi_modal_payload.audio_payload = audio_payload;

    const bool has_multimodal = image_payload.num_images > 0 || audio_payload.num_audios > 0;
    bool success = has_multimodal
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &multi_modal_payload, last_image_token_index)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = gemma4_12b_engine->checkpoint();
    return success;
}

std::string Gemma4_12B::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::vector<int> sampled_tokens;
    std::string result;
    if (length_limit > 0){
        sampled_tokens.reserve(length_limit);
    }
    else{
        sampled_tokens.reserve(4096);
    }
    assert(this->last_token != -1);
    stop_reason_t reason = EOT_DETECTED;

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    int last_sampled_token;
    if(!enable_think) {
        this->token_history.push_back(boc_token_id);
        this->profiler_list[DECODING_TIME].start();
        // fed for the KV cache only, it never reaches the output
        this->_skip_insert_sample(meta_info);
        this->lm_engine->forward(boc_token_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(boc_token_id);

        // thought
        this->token_history.push_back(thought_token_id);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(thought_token_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(thought_token_id);
        
        this->token_history.push_back(new_line_token_id);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(new_line_token_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(new_line_token_id);

        this->token_history.push_back(eoc_token_id);
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(eoc_token_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(eoc_token_id);
        sampled_token = this->_sample(y, meta_info);

        this->total_tokens++;
        meta_info.generated_tokens++;
        last_sampled_token = sampled_token;
        token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }
    else {
        last_sampled_token = this->last_token;
        this->token_history.push_back(this->last_token);
        if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
            std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
            result += token_str;
            os << token_str << std::flush;

        }
        if (this->is_eos(last_sampled_token)){
            return result;
        }
    }

    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
        return result;
    }
    while (this->total_tokens < this->MAX_L){
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            break;
        }
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->_sample(y, meta_info);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;

        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)){ // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(sampled_token);
        if (this->is_eos(sampled_token)){
            meta_info.generated_tokens++;
            this->lm_engine->forward(last_sampled_token);
            break;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    return result;

    // if (!this->enable_think) {
    //     gemma4_12b_npu *gemma4_12b_engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine);
    //     int checkpoint_idx = gemma4_12b_engine->checkpoint();
    //     // copy the token history at the checkpoint except the last one token, which is the start
    //     // token for generation and should not be included in the checkpoint history
    //     checkpoint_his = token_history;
    //     if (!checkpoint_his.empty()) {
    //         checkpoint_his.pop_back();
    //     }
    // }

}

std::string Gemma4_12B::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->generate(meta_info, length_limit, os);
}
