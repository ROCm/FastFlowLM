/// \file modeling_gemma4_12b.cpp
/// \brief Gemma4_12B class
/// \author FastFlowLM Team
/// \date 2026-08-25
/// \version 0.9.45
/// \note This is a source file for the Gemma4_12B class (text-only interface).

#include "AutoModel/modeling_gemma4_12b.hpp"
#include "AutoModel/gemma4_12b_thought.hpp"
#include "chat/chat_gemma4_tools.hpp"
#include "metrices.hpp"
#include "error_measure.hpp"
#include "models/gemma4_12b/flm/aie2p/gemma4_12b_npu.hpp"
#if defined(FLM_ENABLE_OGA_HIPEP)
#include "oga/oga_engine.hpp"
#include "oga/oga_image_bytes.hpp"
#endif
#include <algorithm>
#include <cassert>
#include <cctype>
#include <string_view>

namespace {
std::string trim(const std::string& text) {
    const auto first = std::find_if(text.begin(), text.end(), [](unsigned char c) { return !std::isspace(c); });
    const auto last = std::find_if(text.rbegin(), text.rend(), [](unsigned char c) { return !std::isspace(c); }).base();
    return first < last ? std::string(first, last) : std::string();
}
} // namespace

#if defined(FLM_ENABLE_OGA_HIPEP)
using flm::oga::ImageBytesFromBase64;
using flm::oga::ImageBytesFromFile;
#endif

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
    config.freq_penalty = 0.0f;
    config.pre_penalty = 1.0f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    // The OGA engine has no NPU prefix-slot table. NPU prompts keep theirs.
    const flm_state_api* state = this->backend_id() == flm::backend::kOgaHipepBackendId
                                     ? nullptr
                                     : gemma4_12b_npu_state_api(lm_engine);
    _prefix_attach(lm_engine, state);
}

void Gemma4_12B::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma4_12B::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
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
    inputs.messages = normalized;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty())
        inputs.tools = gemma4_template_tools(tools);
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool Gemma4_12B::_continue_verbatim(std::vector<int>& tokens, std::optional<int> requested_max_new_tokens,
                                    std::vector<int>& rendered) {
    const std::vector<int>& ran = this->token_history;
    if (ran.empty() || static_cast<size_t>(this->lm_engine->get_current_context_length()) != ran.size()) {
        return false;
    }
    const std::vector<int> header = this->tokenizer->encode("<|turn>model\n");
    const std::vector<int> turn_end = this->tokenizer->encode("<turn|>");
    if (header.empty() || turn_end.size() != 1) {
        return false;
    }
    // the reply the engine generated follows the last model header it ran
    const auto at = std::find_end(ran.begin(), ran.end(), header.begin(), header.end());
    if (at == ran.end()) {
        return false;
    }
    // After a continued turn the history holds the earlier replies as the engine ran them (with
    // their empty thought channels), where the prompt renders them as continued_prompt_.rendered does.
    const std::vector<int>& held = this->continued_prompt_.ran;
    const std::vector<int>& held_as_rendered = this->continued_prompt_.rendered;
    std::vector<int> spliced;
    if (!held.empty() && !held_as_rendered.empty() && held.size() <= ran.size() && tokens.size() > held_as_rendered.size() &&
        std::equal(held.begin(), held.end(), ran.begin()) &&
        std::equal(held_as_rendered.begin(), held_as_rendered.end(), tokens.begin())) {
        spliced = held;
        spliced.insert(spliced.end(), tokens.begin() + held_as_rendered.size(), tokens.end());
    }
    const std::vector<int>& prompt = spliced.empty() ? tokens : spliced;
    const size_t start = static_cast<size_t>(at - ran.begin()) + header.size();
    if (prompt.size() <= start || !std::equal(ran.begin(), ran.begin() + start, prompt.begin())) {
        return false;
    }
    // A reply opens with its thought channel, which the template leaves out of earlier turns: a reply
    // with a thought, or cut inside the channel, runs as rendered instead.
    const auto reply = ran.begin() + start;
    if (reply != ran.end() && *reply == boc_token_id) {
        const auto close = std::find(reply, ran.end(), eoc_token_id);
        if (close == ran.end()) {
            return false;
        }
        // an empty channel has the channel's name too
        constexpr std::string_view kChannelName = "thought";
        std::string thought = trim(this->tokenizer->decode(std::vector<int>(reply + 1, close)));
        if (thought.rfind(kChannelName, 0) == 0) thought = trim(thought.substr(kChannelName.size()));
        if (!thought.empty()) {
            return false;
        }
    }
    // the prompt's copy of that reply ends at the first <turn|> after its header
    const auto reply_end = std::find(prompt.begin() + start, prompt.end(), turn_end[0]);
    if (reply_end == prompt.end()) {
        return false;
    }
    // an eos that ended the reply was run; a reply cut at the length limit lacks its last token
    const bool ended = ran.back() == turn_end[0];
    // A reply's content follows its thought channel; a reply cut inside that channel has none.
    const auto content = [&](std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end) {
        if (begin != end && *begin == boc_token_id) {
            const auto close = std::find(begin, end, eoc_token_id);
            if (close == end) return std::string();
            begin = close + 1;
        }
        return trim(this->tokenizer->decode(std::vector<int>(begin, end)));
    };
    const std::string generated = content(ran.begin() + start, ended ? ran.end() - 1 : ran.end());
    const std::string sent = content(prompt.begin() + start, reply_end);
    std::vector<int> continued(ran);
    if (generated != sent) {
        // the client did get the token the engine did not run
        if (ended || generated.empty() || sent.size() <= generated.size() || sent.compare(0, generated.size(), generated) != 0) {
            return false;
        }
        const std::vector<int> rest = this->tokenizer->encode(sent.substr(generated.size()));
        continued.insert(continued.end(), rest.begin(), rest.end());
    }
    continued.insert(continued.end(), ended ? reply_end + 1 : reply_end, prompt.end());
    // the history keeps the empty thought channels the rendered prompt leaves out, so it can be the longer one
    try {
        this->_shared_validate_capacity(continued.size(), requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        return false;
    }
    header_print("FLM", "Continuing the " << ran.size() << " tokens the engine ran, last reply included");
    rendered = std::move(tokens);
    tokens = std::move(continued);
    return true;
}

bool Gemma4_12B::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->feed_empty_thought_ = false;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;

    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    // generate() opens the channel from this flag. It depends only on the request, so record it
    // before a later return. The trim below still runs only when the prompt ends with the channel.
    this->feed_empty_thought_ = flm::gemma4_feed_empty_thought(this->enable_think, meta_info.continue_final_message);

#if defined(FLM_ENABLE_OGA_HIPEP)
    const bool oga = this->backend_id() == flm::backend::kOgaHipepBackendId;
    std::vector<std::string> oga_image_bytes;
    if (oga && !input.audios.empty()) {
        throw ModelRequestError(400, false,
            "gemma4-12b on oga-hipep does not accept audio input",
            "invalid_request_error");
    }
#endif

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
#if defined(FLM_ENABLE_OGA_HIPEP)
                    if (oga) {
                        oga_image_bytes.push_back(flm::oga::ImageBytesFromBase64(std::move(img_str)));
                        newContent.push_back({{"type", "image"}, {"image", img}});
                        total_images++;
                        continue;
                    }
#endif
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
            // Gemma 4 does best with images before the text and audio after it (model card)
            newContent.push_back({{"type", "text"}, {"text", message.value("content", "")}});
            // Process Audios
            if (message.contains("audios")) {
#if defined(FLM_ENABLE_OGA_HIPEP)
                if (oga && !message["audios"].empty()) {
                    throw ModelRequestError(400, false,
                        "gemma4-12b on oga-hipep does not accept audio input",
                        "invalid_request_error");
                }
#endif
                gemma4_12b_npu *gemma4e_engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine); 
                if (!gemma4e_engine) {
                    throw std::runtime_error(
                        "Gemma 4 audio requires a matching frontend and "
                        "backend provider");
                }
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
            if (!gemma4e_engine) {
                throw std::runtime_error(
                    "Gemma 4 audio requires a matching frontend and backend "
                    "provider");
            }

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
#if defined(FLM_ENABLE_OGA_HIPEP)
            if (oga) {
                for (const auto& img_str : input.images)
                    oga_image_bytes.push_back(flm::oga::ImageBytesFromFile(img_str));
            } else
#endif
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
        templated_text = this->apply_chat_template(gemma4_12b_message, input.tools, meta_info.continue_final_message);
    }
    else{

        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        
        for (int i = 0; i < input.images.size(); i++) {
            content["content"].push_back({{"type", "image"}, {"image", input.images[i]}});
        }
        content["content"].push_back({{"type", "text"}, {"text", input.prompt}});
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
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }

#if defined(FLM_ENABLE_OGA_HIPEP)
    if (oga && !oga_image_bytes.empty()) {
        // generate() feeds this empty thought channel itself when thinking is off.
        // The processor must not run it, or the four tokens would be in the KV cache twice.
        const std::string thought = "<|channel>thought\n<channel|>";
        if (!this->enable_think && !meta_info.continue_final_message &&
            templated_text.size() >= thought.size() &&
            templated_text.compare(templated_text.size() - thought.size(), thought.size(), thought) == 0) {
            templated_text.resize(templated_text.size() - thought.size());
        }
        auto* engine = dynamic_cast<flm::oga::OgaEngine*>(this->lm_engine);
        if (!engine)
            throw std::runtime_error("oga-hipep engine is not an OgaEngine");
        std::vector<int> image_tokens = engine->StagePromptImages(templated_text, oga_image_bytes);
        this->profiler_list[TKOEN_ENCODE_TIME].stop(image_tokens.size());
        // The processor already ran the whole prompt, so this prefill must not be chunked,
        // nor start after a cached prefix: the staged inputs are prefilled from token 0.
        if (!this->token_history.empty())
            this->clear_context();
        this->continued_prompt_.rendered.clear();
        this->continued_prompt_.ran.clear();
        const int saved_prefill = meta_info.max_prefill_len;
        meta_info.max_prefill_len = 0;
        bool success = false;
        try {
            success = this->_shared_insert(meta_info, image_tokens, is_cancelled, nullptr);
        } catch (...) {
            meta_info.max_prefill_len = saved_prefill;
            engine->DropStagedPrompt();
            throw;
        }
        meta_info.max_prefill_len = saved_prefill;
        // A request that stopped before its prefill (cancelled, or refused for its
        // length) must not leave its stage to the next request.
        engine->DropStagedPrompt();
        this->_checkpoint_history();
        this->lm_engine->checkpoint();
        return success;
    }
#endif

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
    // soft tokens pushed per image / audio chunk, to tell the prompt-cache plan where each item sits
    std::vector<size_t> image_token_counts;
    std::vector<size_t> audio_token_counts;
    for(int i = 0; i < tokens_init.size(); i++){
        if (tokens_init[i] == image_token_id && image_counter < (int)image_payload.num_images) {
            tokens.push_back(boi_token_id); // the first image soft token id, which is reserved for the model to identify the image position, the rest of the soft tokens for this image will be continuous following this id
            for (int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            tokens.push_back(eoi_token_id); // a separator token between images, not necessary but can help the model to better distinguish different images
            image_token_counts.push_back(image_payload.num_soft_tokens_per_image[image_counter]);
            image_counter++;
        }
        else if (tokens_init[i] == audio_token_id && audio_counter < (int)audio_payload.num_audios) {
            tokens.push_back(boa_token_id); // the first audio soft token id, which is reserved for the model to identify the audio position, the rest of the soft tokens for this audio will be continuous following this id
            for (int j = 0; j < audio_payload.num_soft_tokens_per_audio[audio_counter]; j++) {
                tokens.push_back(audio_token_id);
            }
            tokens.push_back(eoa_token_id); // a separator token between audios, not necessary but can help the model to better distinguish different audios
            audio_token_counts.push_back(audio_payload.num_soft_tokens_per_audio[audio_counter]);
            audio_counter++;
        }
        else {
            tokens.push_back(tokens_init[i]);
        }
    }
    assert(image_counter == (int)image_payload.num_images);
    assert(audio_counter == (int)audio_payload.num_audios);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // hardware
    int restore_idx = -1;

    // Prefix slots, text-only prompts. A prompt that carries pictures or audio, or a text turn whose history still holds their
    // soft tokens (the payload is gone by then), takes the in-place path below. With thinking off, generate() feeds the
    // empty thought channel. insert() removes it only when the prompt already ends with it. A tool-result turn does not,
    // and dropping its last four tokens would cut the tool response. generate() still opens the channel after that turn.
    const bool modal_tokens = std::any_of(tokens.begin(), tokens.end(), [this](int t) {
        return t == image_token_id || t == audio_token_id || t == boi_token_id || t == boa_token_id;
    });
    static_assert(boc_token_id == flm::kGemma4ChannelBegin);
    static_assert(eoc_token_id == flm::kGemma4ChannelEnd);
    static_assert(thought_token_id == flm::kGemma4ThoughtToken);
    static_assert(new_line_token_id == flm::kGemma4NewlineToken);
    const size_t slot_trim = flm::gemma4_empty_thought_trim(this->enable_think, meta_info.continue_final_message, tokens);
    if (image_payload.num_images == 0 && audio_payload.num_audios == 0 && !modal_tokens &&
        this->_prefix_slot_mode(meta_info, tokens.size(), slot_trim)) {
        checkpoint_his.clear();
        this->checkpoint_media_ = std::vector<prompt_media_t>{}; // an empty checkpoint holds no media
        return this->_prefix_text_insert(meta_info, tokens, slot_trim, "<|turn>user", is_cancelled);
    }
    this->_prefix_legacy_path();

    // OGA cannot reuse a prompt that leaves the history it ran, so it runs on from that history instead
    std::vector<int> rendered;
    const bool continued = meta_info.restore_allowed && !modal_tokens &&
                           this->backend_id() == flm::backend::kOgaHipepBackendId &&
                           this->_continue_verbatim(tokens, input.requested_max_new_tokens, rendered);

    // The legacy checkpoint keeps only the rings: it relies on the global rows still being the ones it was taken with. A slot request
    // overwrites them, so it drops the checkpoint's token history, and the in-place path below then has nothing to restore.
    if (meta_info.restore_allowed && !checkpoint_his.empty() && !continued) {
        restore_idx = this->lm_engine->restore();
        this->total_tokens = restore_idx;
        this->_restore_history(); // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    size_t n = tokens.size();
    tokens.resize(n - slot_trim);

    // The payload carries every image / audio chunk of the prompt, including those of earlier turns the engine already
    // holds: drop the ones _shared_insert will skip, decided by the same rule it uses, on the same tokens and history.
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(tokens, image_token_id, image_token_counts, prompt_media_t::kImage, media);
    prompt_media_from_counts(tokens, audio_token_id, audio_token_counts, prompt_media_t::kAudio, media);
    // Content hashes of the full payload (before the drop), so a different picture or clip with the same token count is
    // not taken for the one the engine holds. media lists the images first, then the audio chunks, each in payload order.
    {
        auto nonzero = [](uint64_t h) { return h ? h : 1ull; };
        size_t m = 0;
        for (size_t i = 0; i < image_payload.num_images && m < media.size(); i++, m++) {
            const auto& px = image_payload.pixel_values[i];
            const auto& grid = image_payload.image_grid_pairs_per_image[i];
            const auto dims = image_payload.image_patch__element_per_patch[i];
            uint64_t h = 0xcbf29ce484222325ull ^ (static_cast<uint64_t>(static_cast<uint32_t>(dims.first)) << 32 |
                                                 static_cast<uint32_t>(dims.second));
            h ^= static_cast<uint64_t>(image_payload.valid_patch_size_per_image[i]) * 0x9e3779b97f4a7c15ull;
            h = prefix_hash_bytes(grid.data(), grid.size() * sizeof(grid[0]), h);
            h = prefix_hash_bytes(px.data(), px.size() * sizeof(px[0]), h);
            media[m].hash = nonzero(h);
        }
        for (size_t i = 0; i < audio_payload.num_audios && m < media.size(); i++, m++) {
            const auto& mel = audio_payload.mel_spectrograms[i];
            const uint64_t h = 0xcbf29ce484222325ull ^
                (static_cast<uint64_t>(static_cast<uint32_t>(audio_payload.mel_spectrogram_frames_per_audio[i])) << 32 |
                 static_cast<uint32_t>(audio_payload.mel_spectrogram_bins_per_audio[i]));
            media[m].hash = nonzero(prefix_hash_bytes(mel.data(), mel.size() * sizeof(mel[0]), h));
        }
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);
    {
        auto drop_front = [](auto& vec, size_t count) {
            if (count == 0) return;
            if (count >= vec.size()) { vec.clear(); return; }
            vec.erase(vec.begin(), vec.begin() + count);
        };

        const size_t images_to_drop = plan.skipped_media[prompt_media_t::kImage];
        if (images_to_drop > 0) {
            drop_front(image_payload.image_patch__element_per_patch, images_to_drop);
            drop_front(image_payload.valid_patch_size_per_image,     images_to_drop);
            drop_front(image_payload.pixel_values,                   images_to_drop);
            drop_front(image_payload.image_grid_pairs_per_image,     images_to_drop);
            drop_front(image_payload.num_soft_tokens_per_image,      images_to_drop);
            image_payload.num_images -= static_cast<unsigned>(images_to_drop);
        }

        const size_t audios_to_drop = plan.skipped_media[prompt_media_t::kAudio];
        if (audios_to_drop > 0) {
            drop_front(audio_payload.mel_spectrograms,                 audios_to_drop);
            drop_front(audio_payload.mel_spectrogram_frames_per_audio, audios_to_drop);
            drop_front(audio_payload.mel_spectrogram_bins_per_audio,   audios_to_drop);
            drop_front(audio_payload.num_soft_tokens_per_audio,        audios_to_drop);
            audio_payload.num_audios -= static_cast<unsigned>(audios_to_drop);
        }
    }

    // one past the last prefilled image or audio row, relative to the skip: the first chunk is widened to hold
    // them, as their payload travels with it alone. Audio comes after the text, so a long text can push it past
    // a first chunk sized for the images only.
    const int first_len_run = static_cast<int>(plan.media_end);

    gemma4_12b_multi_modal_payload_t multi_modal_payload;
    multi_modal_payload.image_payload = image_payload;
    multi_modal_payload.audio_payload = audio_payload;

    const bool has_multimodal = image_payload.num_images > 0 || audio_payload.num_audios > 0;
    const size_t prompt_tokens = tokens.size();
    bool success = has_multimodal
        ? this->_shared_insert(meta_info, tokens, is_cancelled, &multi_modal_payload, first_len_run)
        : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    this->_checkpoint_history();
    int checkpoint_idx = this->lm_engine->checkpoint();
    // a shorter history is not this prompt, which _shared_insert refused or a cancel cut short
    if (continued && this->token_history.size() >= prompt_tokens) {
        // Kept up to the header of the reply to come, which the next turn renders right after it, without the
        // empty thought channel that generate() feeds when thinking is off.
        const std::vector<int> header = this->tokenizer->encode("<|turn>model\n");
        const auto through_header = [&](const std::vector<int>& prompt) {
            const auto at = std::find_end(prompt.begin(), prompt.end(), header.begin(), header.end());
            return at == prompt.end() ? std::vector<int>() : std::vector<int>(prompt.begin(), at + header.size());
        };
        this->continued_prompt_.rendered = through_header(rendered);
        this->continued_prompt_.ran = through_header(this->token_history);
    } else {
        this->continued_prompt_.rendered.clear();
        this->continued_prompt_.ran.clear();
    }
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
    // A continued message has no empty thought channel to feed: it starts from insert()'s sample.
    // feed_empty_thought_ is what insert() decided, including a tool-result turn whose prompt
    // did not itself end with the channel.
    if (this->feed_empty_thought_) {
        this->feed_empty_thought_ = false;
        // The four forwards below write rows total_tokens..total_tokens+3, so the
        // prompt has to leave room for them: past the engine's KV window they
        // corrupt it for every later request.
        if (this->total_tokens + 4 > this->MAX_L) {
            header_print("WARNING", "Max length reached, stopping generation...");
            meta_info.stop_reason = MAX_LENGTH_REACHED;
            return result;
        }
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

        // all four went through the engine, so all four occupy the context
        this->total_tokens += 4;
        meta_info.generated_tokens++;
        last_sampled_token = sampled_token;
        // the reply's first token: the loop below forwards it, so the history holds it
        this->token_history.push_back(last_sampled_token);
        if (this->is_normal_token(last_sampled_token)) {
            token_str = this->tokenizer->run_time_decoder(last_sampled_token);
            result += token_str;
            os << token_str << std::flush;
        }
        if (this->is_eos(last_sampled_token) && !meta_info.ignore_eos) {
            // the reply ended before it started; no engine has run the eos
            this->token_history.pop_back();
            meta_info.stop_reason = reason;
            return result;
        }
    }
    else {
        last_sampled_token = this->last_token;
        this->token_history.push_back(this->last_token);
        // the sample insert() took is the reply's first token, and counts toward the limit
        meta_info.generated_tokens++;
        if (this->is_normal_token(last_sampled_token) && last_sampled_token != -1){
            std::string token_str = this->tokenizer->run_time_decoder(last_sampled_token);
            result += token_str;
            os << token_str << std::flush;

        }
        if (this->is_eos(last_sampled_token) && !meta_info.ignore_eos){
            // Sampled from the prefill, so no engine has run it.
            this->token_history.pop_back();
            meta_info.stop_reason = reason;
            return result;
        }
    }

    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation: test it here so it is recorded.
        if (this->total_tokens >= this->MAX_L){
            reason = MAX_LENGTH_REACHED;
            break;
        }
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
        if (this->is_eos(sampled_token) && !meta_info.ignore_eos){
            meta_info.generated_tokens++;
            // Its row is past the end of the engine's KV window once the context is
            // full: forwarding it there corrupts the engine for every later request,
            // so the next prompt has to carry it instead.
            if (this->total_tokens < this->MAX_L) {
                this->lm_engine->forward(last_sampled_token);
            } else {
                this->token_history.pop_back();
            }
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
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    std::cout << std::endl;
    this->print_raw_output(result);
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
