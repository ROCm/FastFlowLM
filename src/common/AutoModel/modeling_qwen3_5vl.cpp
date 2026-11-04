/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "metrices.hpp"
#include "oga/oga_vision.hpp"

#include <algorithm>
#include <cctype>
#include <iterator>


/************              Qwen3_5VL family            **************/
Qwen3_5VL::Qwen3_5VL(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Qwen3_5VL") {}

void Qwen3_5VL::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->_load_vision_geometry();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool = (model_info["size"] > 800000000)? true : false;

    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->_attach_prefix_cache();
}

void Qwen3_5VL::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Qwen3_5VL::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message);
}


/************              prefix slots            **************/
namespace {
constexpr int kImageSoftTokenId = 248056;
constexpr int kImEndId = 248046;

/// \brief the images a request carries, in either input form
/// \note The CLI passes images in input.images; the REST API puts them in the
///       "images" arrays of its messages. insert() reads no audio, so audios
///       are not counted here.
std::size_t count_images(const lm_uniform_input_t& input) {
    std::size_t count = input.images.size();
    for (const auto& item : input.messages) {
        if (item.contains("images") && item["images"].is_array()) count += item["images"].size();
    }
    return count;
}

std::string trim_left(const std::string& text) {
    const auto first = std::find_if(text.begin(), text.end(), [](unsigned char c) { return !std::isspace(c); });
    return std::string(first, text.end());
}

std::string trim_right(const std::string& text) {
    const auto last = std::find_if(text.rbegin(), text.rend(), [](unsigned char c) { return !std::isspace(c); });
    return std::string(text.begin(), last.base());
}
} // namespace

void Qwen3_5VL::_attach_prefix_cache() {
    // OGA has no native prefix-slot state API; keep the existing path for other backends.
    const flm_state_api* api = this->backend_id() == flm::backend::kOgaHipepBackendId
        ? nullptr : qwen3_5vl_state_api(this->lm_engine);
    this->_prefix_attach(this->lm_engine, api, kImageSoftTokenId);
}

void Qwen3_5VL::_load_vision_geometry() {
    this->vision_geometry = {};
    if (const auto* npu = dynamic_cast<const qwen3_5vl_npu*>(this->lm_engine)) {
        this->vision_geometry = {npu->QWEN3_5_PATCH_SIZE, npu->QWEN3_5_MERGE_SIZE, npu->QWEN3_5_TEMPORAL_PATCH_SIZE,
                                 npu->QWEN3_5_SHORTEST_EDGE, npu->QWEN3_5_LONGEST_EDGE,
                                 npu->QWEN3_5_VISION_RESCALE_FACTOR, npu->QWEN3_5_VISION_RESCALE_IMAGE_MEAN,
                                 npu->QWEN3_5_VISION_RESCALE_IMAGE_STD};
        return;
    }
    if (this->backend_id() != flm::backend::kOgaHipepBackendId || !this->lm_config) return;
    // An OGA package states its preprocessing as the transforms of OGA's own image processor.
    const nlohmann::json& processor = this->lm_config->sub("processor_config");
    if (processor.empty()) return;
    qwen3_5vl_vision_geometry_t geometry;
    try {
        for (const auto& transform : processor.at("processor").at("transforms")) {
            const auto& operation = transform.at("operation");
            const std::string type = operation.value("type", std::string());
            const nlohmann::json attrs = operation.value("attrs", nlohmann::json::object());
            if (type == "Resize") {
                geometry.min_pixels = attrs.value("min_pixels", 0u);
                geometry.max_pixels = attrs.value("max_pixels", 0u);
            } else if (type == "Rescale") {
                geometry.rescale_factor = attrs.value("rescale_factor", 0.0f);
            } else if (type == "Normalize") {
                const auto mean = attrs.value("mean", std::vector<float>{});
                const auto deviation = attrs.value("std", std::vector<float>{});
                // the normalization takes one mean and one deviation for all three channels
                const auto uniform = [](const std::vector<float>& values) {
                    return !values.empty() &&
                           std::all_of(values.begin(), values.end(), [&](float v) { return v == values.front(); });
                };
                if (!uniform(mean) || !uniform(deviation))
                    throw std::runtime_error("Normalize has a different mean or std per channel");
                geometry.image_mean = mean.front();
                geometry.image_std = deviation.front();
            } else if (type == "PatchImage") {
                geometry.patch_size = attrs.value("patch_size", 0u);
                geometry.temporal_patch_size = attrs.value("temporal_patch_size", 0u);
                geometry.merge_size = attrs.value("merge_size", 0u);
            }
        }
    } catch (const std::exception& e) {
        header_print("WARNING", "processor_config.json: " << e.what() << "; image input is off");
        return;
    }
    if (!geometry.valid() || geometry.rescale_factor == 0.0f || geometry.image_std == 0.0f) {
        header_print("WARNING", "processor_config.json does not state the image preprocessing; image input is off");
        return;
    }
    this->vision_geometry = geometry;
}

bool Qwen3_5VL::_continue_verbatim(std::vector<int>& tokens, std::optional<int> requested_max_new_tokens) {
    const std::vector<int>& ran = this->token_history;
    const std::vector<int>& held = this->checkpoint_his;
    const size_t cp = held.size();
    if (cp == 0 || ran.size() <= cp || tokens.size() <= cp ||
        static_cast<size_t>(this->lm_engine->get_current_context_length()) != ran.size() ||
        !std::equal(held.begin(), held.end(), ran.begin()) || !std::equal(held.begin(), held.end(), tokens.begin())) {
        return false;
    }
    // the prompt's copy of the last reply ends at the first <|im_end|> after the checkpoint
    const auto reply_end = std::find(tokens.begin() + cp, tokens.end(), kImEndId);
    if (reply_end == tokens.end()) return false;
    // A reply's content follows its reasoning block; a reply cut inside that block has none.
    const auto content = [&](std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end) {
        const auto close = std::find(std::make_reverse_iterator(end), std::make_reverse_iterator(begin), this->think_end_id);
        if (close.base() != begin) return this->tokenizer->decode(std::vector<int>(close.base(), end));
        if (begin != end && *begin == this->think_start_id) return std::string();
        return this->tokenizer->decode(std::vector<int>(begin, end));
    };
    const std::string generated = trim_left(content(ran.begin() + cp, ran.end()));
    const std::string sent = trim_right(trim_left(content(tokens.begin() + cp, reply_end)));
    std::vector<int> continued(ran);
    if (trim_right(generated) != sent) {
        // A reply cut at the length limit lacks its last sampled token, which the client did get.
        if (generated.empty() || sent.size() <= generated.size() || sent.compare(0, generated.size(), generated) != 0) {
            return false;
        }
        const std::vector<int> rest = this->tokenizer->encode(sent.substr(generated.size()));
        continued.insert(continued.end(), rest.begin(), rest.end());
    }
    continued.insert(continued.end(), reply_end, tokens.end());
    // The history keeps the reasoning the rendered prompt leaves out, so it can be the longer one.
    try {
        this->_shared_validate_capacity(continued.size(), requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        return false;
    }
    header_print("FLM", "Continuing the " << ran.size() << " tokens the engine ran, last reply included");
    tokens = std::move(continued);
    return true;
}

bool Qwen3_5VL::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // Refuse images before any image work: an engine without image
    // preprocessing runs no vision model.
    const size_t requested_images = count_images(input);
    if (requested_images > 0 && !this->vision_geometry.valid()) {
        // A ModelRequestError, not invalid_argument: the handlers turn only the
        // former into an HTTP 400, and the latter would come back as a 200 body.
        throw ModelRequestError(400, false,
            "the loaded backend '" + this->backend_id() + "' does not accept image input",
            "invalid_request_error");
    }
    const bool oga = this->backend_id() == flm::backend::kOgaHipepBackendId;
    unsigned max_pixels = this->vision_geometry.max_pixels;
    if (oga && requested_images > 0) {
        // OGA runs all the images of a prompt in one vision pass, so they share its patch budget.
        const uint64_t patch_pixels = static_cast<uint64_t>(this->vision_geometry.patch_size) * this->vision_geometry.patch_size;
        const uint64_t share = static_cast<uint64_t>(flm::oga::kMaxImagePatches) / requested_images * patch_pixels;
        max_pixels = static_cast<unsigned>(std::min<uint64_t>(max_pixels, share));
    }

    // preprocess
    constexpr int image_soft_token_id = 248056;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    qwen3_5vl_image_payload_t image_payload;
    image_payload.num_images = 0;
    if (input.images.size() > 0) {


        // header_print("info", "Processing images...");
        
        // time_utils::time_point preprocess_start = time_utils::now();
        for(const auto& img_str : input.images){
            qwen3_5vl_image_t image = this->load_image(img_str);
            if (image.width <= 0 || image.height <= 0) {
                header_print("ERROR", "Skipping image that failed to load: " << img_str);
                continue;
            }

            preprocess_image(image, image_payload._data__processed, max_pixels);
            if (image.grid_h <= 0 || image.grid_w <= 0) {
                header_print("ERROR", "Skipping image that failed to preprocess: " << img_str);
                continue;
            }
            // Push the image AFTER preprocessing so grid_h and grid_w are set
            image_payload.images.push_back(image);
            image_payload.num_images++;
        }
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        json qwenvl_message = json::array();
        int total_images = 0;
        for (const auto& item : input.messages) {
            if (!item.contains("images")) {
                qwenvl_message.push_back(item);
                continue;
            }

            json newContent = json::array();
            for (const auto& img : item["images"]) {
                std::string img_str = img.get<std::string>();

                // Decode/preprocess up front so an invalid image (e.g. bad
                // base64) never gets an image-soft-token placeholder in the
                // template below; otherwise image_payload.images would fall
                // out of sync with the placeholders and the prompt would
                // still try to prefill a nonexistent image.
                qwen3_5vl_image_t image = this->load_image_base64(img_str);
                if (image.width <= 0 || image.height <= 0) {
                    header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                    continue;
                }

                preprocess_image(image, image_payload._data__processed, max_pixels);
                if (image.grid_h <= 0 || image.grid_w <= 0) {
                    header_print("ERROR", "Skipping image that failed to preprocess");
                    continue;
                }

                image_payload.images.push_back(image);
                image_payload.num_images++;
                total_images++;

                newContent.push_back({
                    {"type", "image"},
                    {"image", img}
                });
            }
            newContent.push_back({
                {"type", "text"},
                {"text", item["content"]}
            });

            json newItem = {
                {"role", item["role"]},
                {"content", newContent}
            };

            qwenvl_message.push_back(newItem);
        }
        templated_text = this->apply_chat_template(qwenvl_message, input.tools, meta_info.continue_final_message);
        header_print("FLM", "Total images: " << total_images);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        
        // Add image objects to content array
        for (int i = 0; i < input.images.size(); i++) {
            nlohmann::ordered_json image_obj;
            image_obj["type"] = "image";
            image_obj["image"] = input.images[i];
            content["content"].push_back(image_obj);
        }
        
        // Add text object to content array
        nlohmann::ordered_json text_obj;
        text_obj["type"] = "text";
        text_obj["text"] = input.prompt;
        content["content"].push_back(text_obj);
        
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // update the tokens to include the image tokens
    std::vector<int> tokens;
    int total_image_tokens = 0;
    // Use image_payload.images.size() (not input.images.size()), because on
    // the REST API path images come from `messages` and input.images is empty.
    for (size_t i = 0; i < image_payload.images.size(); i++) {
        total_image_tokens += image_payload.images[i].grid_h * image_payload.images[i].grid_w;
    }
    tokens.reserve(tokens_init.size() + total_image_tokens);
    size_t image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            // the prompt's own text can spell the placeholder too
            if (image_counter == image_payload.images.size()) {
                throw ModelRequestError(400, false, "the prompt has more image placeholders than images",
                                        "invalid_request_error");
            }
            for (int j = 0; j < image_payload.images[image_counter].grid_h * image_payload.images[image_counter].grid_w / 4; j++) {
                tokens.push_back(image_soft_token_id);
            }
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }
    if (oga) {
        // Include the think preamble before it is trimmed from the prefill.
        this->_shared_validate_capacity(tokens.size(), input.requested_max_new_tokens);
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // ----------------------------------------------------------------------
    // Prompt-cache aware image alignment.
    //
    // AutoModel::_shared_insert prefix-matches `tokens` against `checkpoint_his`
    // over the FULL length of `checkpoint_his`. If every token matches, it
    // erases that prefix before prefilling; otherwise it calls clear_context()
    // and skips nothing. We must NOT erase `tokens` here -- _shared_insert
    // needs the untrimmed sequence to run that very check. What we DO need to
    // fix up locally is the image payload (pixels for the WHOLE prompt,
    // including already-cached images from earlier turns): drop the
    // fully-cached leading images so the surviving payload aligns with the
    // surviving image tokens after _shared_insert performs its own erase.
    // ----------------------------------------------------------------------
    // ---- prefix slots: resume from the longest saved prefix of this prompt ----
    // generate() feeds the generation prompt's think tokens itself, so they are left out of the prefill. A continued
    // message keeps all its tokens: they are the client's text, not the generation prompt.
    const size_t trim_tokens = meta_info.continue_final_message ? 0 : (this->enable_think ? 2 : 4);
    qwen3_5vl_npu* slot_eng = dynamic_cast<qwen3_5vl_npu*>(this->lm_engine);
    const bool slot_mode = this->prefix_cache_.enabled() && slot_eng != nullptr && meta_info.restore_allowed && tokens.size() > trim_tokens;
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    std::vector<uint64_t> image_hashes;    // content of each of its images
    if (this->prefix_cache_.enabled() && !slot_mode) {
        // the single-checkpoint path below moves the engine on without telling the cache
        this->prefix_cache_.forget_all();
    }
    if (slot_mode) {
        full_tokens.assign(tokens.begin(), tokens.end() - trim_tokens);
        image_hashes = prefix_image_hashes(image_payload, slot_eng->QWEN3_5_PATCH_SIZE, slot_eng->QWEN3_5_TEMPORAL_PATCH_SIZE);
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    }

    // An engine that would replay the whole history after a restore continues
    // the history it ran instead, when the prompt is that history going on.
    const bool continued = !slot_mode && meta_info.restore_allowed && this->backend_ &&
                           !this->backend_->keeps_state_on_restore() &&
                           this->_continue_verbatim(tokens, input.requested_max_new_tokens);

    size_t prefix_skip_count = 0;
    {
        // what _shared_insert will find in the engine: the history it ran, or the checkpoint restored below
        const std::vector<int>& held = continued ? this->token_history : this->checkpoint_his;
        const size_t idx = held.size();
        for (size_t i = 0; i < idx; i++) {
            if (i < tokens.size() && tokens[i] == held[i]) {
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
        if (slot_mode) {
            prefix_skip_count = match_len;
        }

        if (prefix_skip_count > 0 && !image_payload.images.empty()) {
            // Per-image bf16 footprint depends on runtime patch/temporal
            // config carried by the engine.
            const unsigned patch_size = this->vision_geometry.patch_size;
            const unsigned temporal_patch = this->vision_geometry.temporal_patch_size;

            int skipped_image_tokens = 0;
            for (size_t i = 0; i < prefix_skip_count; i++) {
                if (tokens[i] == image_soft_token_id) skipped_image_tokens++;
            }

            size_t images_to_drop = 0;
            size_t bf16_to_drop = 0;
            int consumed_image_tokens = 0;
            for (const auto& img : image_payload.images) {
                const int img_tokens = (img.grid_h * img.grid_w) / 4;
                const size_t img_bf16 =
                    static_cast<size_t>(img.grid_h) * patch_size *
                    static_cast<size_t>(img.grid_w) * patch_size *
                    3u * temporal_patch;
                if (consumed_image_tokens + img_tokens <= skipped_image_tokens) {
                    consumed_image_tokens += img_tokens;
                    bf16_to_drop += img_bf16;
                    images_to_drop++;
                } else {
                    break;
                }
            }

            if (images_to_drop > 0) {
                image_payload.images.erase(
                    image_payload.images.begin(),
                    image_payload.images.begin() + images_to_drop);
                image_payload.num_images -= static_cast<int>(images_to_drop);
                if (bf16_to_drop >= image_payload._data__processed.size()) {
                    image_payload._data__processed.clear();
                } else {
                    image_payload._data__processed.erase(
                        image_payload._data__processed.begin(),
                        image_payload._data__processed.begin() + bf16_to_drop);
                }
                header_print("FLM",
                    "Prompt-cache hit: dropped " << images_to_drop
                    << " cached image(s) from payload");
            }
        }
    }

    // find the last image token index, expressed relative to the tokens that
    // will SURVIVE _shared_insert's prefix erase (i.e. shifted by -prefix_skip_count).
    int last_image_token_index = -1;
    for (int i = static_cast<int>(prefix_skip_count); i < (int)tokens.size(); i++) {
        if (tokens[i] == image_soft_token_id) {
            last_image_token_index = i - static_cast<int>(prefix_skip_count);
        }
    }
    last_image_token_index++; // plus the end of image tokens


    // hardware
    int restore_idx = -1;
    const bool has_images = image_payload.num_images > 0;

    if (!slot_mode && meta_info.restore_allowed && !continued) {
        restore_idx = this->lm_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    size_t n = tokens.size();
    tokens.resize(n - trim_tokens);

    if (slot_mode) {
        return this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, image_hashes, match_len, "<|im_start|>user",
                                    has_images ? static_cast<void*>(&image_payload) : nullptr, last_image_token_index, is_cancelled);
    }

    // OGA takes the pixels as float16, one block per image
    flm::oga::OgaVisionInput oga_vision;
    if (oga && has_images) {
        const qwen3_5vl_vision_geometry_t& geometry = this->vision_geometry;
        const size_t values = 3u * geometry.temporal_patch_size * geometry.patch_size * geometry.patch_size;
        size_t offset = 0;
        for (const auto& img : image_payload.images) {
            flm::oga::OgaImage image;
            image.grid_h = img.grid_h;
            image.grid_w = img.grid_w;
            const size_t count = static_cast<size_t>(img.grid_h) * img.grid_w * values;
            if (offset + count > image_payload._data__processed.size()) {
                throw std::runtime_error("image payload holds fewer pixels than its images");
            }
            const bf16* from = image_payload._data__processed.data() + offset;
            image.pixels.resize(count);
            std::transform(from, from + count, image.pixels.begin(),
                           [](const bf16& value) { return flm::oga::FloatToHalf(static_cast<float>(value)); });
            offset += count;
            oga_vision.images.push_back(std::move(image));
        }
    }
    void* payload = !has_images ? nullptr : oga ? static_cast<void*>(&oga_vision) : static_cast<void*>(&image_payload);
    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, payload, has_images ? last_image_token_index : 0);

    checkpoint_his = token_history;
    int checkpoint_idx = this->lm_engine->checkpoint();
    return success;
}

std::string Qwen3_5VL::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    // the think preamble below completes the generation prompt insert() left out; a continued
    // message was prefilled whole, so its output starts from the sample insert() made
    if (meta_info.continue_final_message) {
        return this->_shared_generate(meta_info, length_limit, os, std::move(is_cancelled));
    }
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

    // The engine runs ahead of total_tokens: the four preamble tokens below are
    // fed to it but only the sampled one is counted, so the next forward lands
    // on row total_tokens + 3, and the extra forward after an EOS one further.
    // Keep that much room under MAX_L, or those forwards write past the end of
    // the engine's KV window and corrupt it for every later request.
    constexpr uint32_t kEngineLead = 4;
    if (this->total_tokens + kEngineLead >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    if(this->enable_think) {
        this->token_history.push_back(think_start_id);
        this->profiler_list[DECODING_TIME].start();
        this->_force_token(meta_info, think_start_id);
        this->lm_engine->forward(think_start_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_start_id);
        result += token_str;
        os << token_str << std::flush;

        // \n
        this->token_history.push_back(198);
        this->profiler_list[DECODING_TIME].start();
        this->_force_token(meta_info, 198);
        buffer<bf16> y = this->lm_engine->forward(198);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(198);
        result += token_str;
        // First token of the reasoning block.
        sampled_token = this->_sample(y, meta_info);
        os << token_str << std::flush;
    }
    else{
        this->token_history.push_back(think_start_id);
        // fed for the KV cache only, it never reaches the output
        this->_skip_insert_sample(meta_info);
        this->lm_engine->forward(think_start_id);
        token_str = this->tokenizer->run_time_decoder(think_start_id);

        // \n\n
        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        
        this->token_history.push_back(think_end_id);
        this->profiler_list[DECODING_TIME].start();
        this->lm_engine->forward(think_end_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_end_id);

        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        buffer<bf16> y = this->lm_engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        sampled_token = this->_sample(y, meta_info);
    }
    if (this->backend_id() == flm::backend::kOgaHipepBackendId) {
        // The preamble has run; reuse the shared capacity, EOS and history handling.
        this->total_tokens = this->lm_engine->get_current_context_length();
        this->last_token = sampled_token;
        return result + this->_shared_generate(meta_info, length_limit, os, std::move(is_cancelled));
    }
    this->total_tokens++;
    // the first sample after the preamble is the reply's first token, and counts toward the limit
    meta_info.generated_tokens++;
    int last_sampled_token = sampled_token;
    this->token_history.push_back(last_sampled_token);
    token_str = this->tokenizer->run_time_decoder(last_sampled_token);
    result += token_str;
    os << token_str << std::flush;

    if (this->is_eos(last_sampled_token)){
        this->lm_engine->forward(last_sampled_token);
    }
    else if (this->total_tokens + kEngineLead >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation; testing here rather than in
        // the loop condition records it.
        if (this->total_tokens + kEngineLead >= this->MAX_L){
            header_print("WARNING", "Max length reached, stopping generation...");
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
    if (reason != EOT_DETECTED) {
        // Cut short after sampling its last token: no engine has run that one,
        // so the next prompt has to carry it.
        this->token_history.pop_back();
    }

    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    
    return result;
}

std::string Qwen3_5VL::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->backend_id() == flm::backend::kOgaHipepBackendId) {
        return this->generate(meta_info, length_limit, os);
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    int checkpoint_idx = this->lm_engine->checkpoint();
    int restore_idx = this->lm_engine->restore();
    header_print_r("FLM", "Checkpoint before generation: " << checkpoint_idx << ", restore point: " << restore_idx << ", user context length: " << this->token_history.size());
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
}
