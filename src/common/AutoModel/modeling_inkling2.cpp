/// \file modeling_inkling2.cpp
/// \brief Inkling2 class
/// \author FastFlowLM Team
/// \date 2026-11-21
/// \note Portions of this file consist of AI-generated content.
/// \note AutoModel wrapper for the Inkling-2-Mini engine (inkling2_npu).

#include "AutoModel/modeling_inkling2.hpp"
#include "AutoModel/reasoning_effort_levels.hpp"
#include "models/inkling2/flm/aie2p/inkling2_npu.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <cmath>

#include "prefix_slot_cache.hpp"
#include "prompt_reuse.hpp"

namespace {

/// The tool declaration as the reference template writes it: keys sorted, no spaces.
/// minja's tojson has neither sort_keys nor separators, so the package's template
/// prints this string instead of building the list itself.
std::string tool_specs_json(const nlohmann::ordered_json& tools) {
    auto field_or = [](const nlohmann::ordered_json& o, const char* key, nlohmann::json fallback) {
        if (!o.contains(key)) return fallback;
        const auto& v = o[key];
        const bool empty = v.is_null() || (v.is_string() && v.get<std::string>().empty()) || (v.is_object() && v.empty());
        return empty ? fallback : nlohmann::json(v);
    };
    nlohmann::json specs = nlohmann::json::array();
    for (const auto& tool : tools) {
        const nlohmann::ordered_json& fn = tool.contains("function") ? tool["function"] : tool;
        specs.push_back({{"description", field_or(fn, "description", "")},
                         {"name", fn.value("name", nlohmann::ordered_json())},
                         {"parameters", field_or(fn, "parameters", nlohmann::json::object())},
                         {"type", field_or(tool, "type", "function")}});
    }
    return specs.dump();   // nlohmann::json keeps its object keys sorted
}

/// One past call as the model writes it: {"name":..,"args":{..}}, the arguments' top-level
/// keys sorted and their values compact, as given. Arguments may come as a JSON string.
/// \throws std::invalid_argument when the arguments are not an object
std::string call_json(const std::string& name, const nlohmann::ordered_json& arguments) {
    nlohmann::ordered_json args = arguments;
    if (args.is_string()) {
        const std::string text = args.get<std::string>();
        try {
            args = text.empty() ? nlohmann::ordered_json::object() : nlohmann::ordered_json::parse(text);
        } catch (const nlohmann::json::parse_error&) {
            throw std::invalid_argument("tool call arguments of " + name + " are not valid JSON");
        }
    }
    if (args.is_null()) args = nlohmann::ordered_json::object();
    if (!args.is_object()) throw std::invalid_argument("tool call arguments of " + name + " must be an object");
    std::vector<std::string> keys;
    for (const auto& item : args.items()) keys.push_back(item.key());
    std::sort(keys.begin(), keys.end());
    std::string out = "{\"name\":" + nlohmann::ordered_json(name).dump() + ",\"args\":{";
    for (size_t i = 0; i < keys.size(); i++) {
        out += (i ? "," : "") + nlohmann::ordered_json(keys[i]).dump() + ":" + args[keys[i]].dump();
    }
    return out + "}}";
}

}  // namespace

/************              Inkling 2 family            **************/
Inkling2::Inkling2(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Inkling2") {}

void Inkling2::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // generation_config.json: plain sampling at temperature 1.0, no top-k / top-p
    sampler_config config;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 1.0f;
    config.rep_penalty = 1.0f;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    if (const char* cap = std::getenv("FLM_INKLING2_IMAGE_MAX_TOKENS")) {
        this->image_max_tokens = std::max(0, std::atoi(cap));
    }
    if (auto* engine = dynamic_cast<inkling2_npu*>(this->lm_engine)) {
        if (engine->has_vision_tower()) header_print("FLM", "image input: on (vision_weight.q4nx)");
    }
    this->_prefix_attach(this->lm_engine, inkling2_npu_state_api(this->lm_engine), inkling2_npu::image_token_id);
}

bool Inkling2::load_image(const std::string& source, bool base64, image_data_t& out) {
    const bool ok = base64 ? this->image_reader_.load_image_base64(source, out) : this->image_reader_.load_image(source, out);
    if (!ok || out.width <= 0 || out.height <= 0) return false;
    auto* engine = dynamic_cast<inkling2_npu*>(this->lm_engine);
    if (!engine || this->image_max_tokens <= 0) return true;
    int tokens = engine->image_tokens_for(out.height, out.width);
    if (tokens <= this->image_max_tokens) return true;
    // shrink by the area ratio, then by single steps until the cells fit
    const double scale = std::sqrt(static_cast<double>(this->image_max_tokens) / tokens);
    int width = std::max(1, static_cast<int>(out.width * scale));
    int height = std::max(1, static_cast<int>(out.height * scale));
    while (width > 1 && height > 1 && engine->image_tokens_for(height, width) > this->image_max_tokens) {
        width = std::max(1, width - 8);
        height = std::max(1, height - 8);
    }
    image_data_t resized;
    if (!this->image_reader_.resize_image(out, width, height, resized)) return false;
    header_print("FLM", "Image scaled from " << out.width << "x" << out.height << " to " << width << "x" << height
                        << " (at most " << this->image_max_tokens << " image tokens)");
    this->image_reader_.recycle(out);
    out = std::move(resized);
    return true;
}

void Inkling2::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

bool Inkling2::valid_effort(const std::string& effort) {
    return flm::inkling_valid_effort(effort);
}

/// \note The package's chat_template.jinja is the reference template made to run in minja:
///       it prints `inkling_tool_specs` and each call's `inkling_call` (built here) where
///       the reference builds them with tojson(sort_keys=true, separators=...).
std::string Inkling2::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    minja::chat_template_options opt;
    opt.polyfill_tool_responses = false;
    inputs.extra_context = this->extra_context;
    // a number goes in as a number: the template prints it ("Thinking effort level: 0.5")
    double numeric_effort = 0.0;
    if (flm::parse_inkling_numeric_effort(this->reasoning_effort, numeric_effort)) {
        inputs.extra_context["reasoning_effort"] = numeric_effort;
    } else {
        inputs.extra_context["reasoning_effort"] = this->reasoning_effort;
    }
    nlohmann::ordered_json prepared = messages;
    try {
        for (auto& message : prepared) {
            if (!message.contains("tool_calls") || !message["tool_calls"].is_array()) continue;
            for (auto& call : message["tool_calls"]) {
                const nlohmann::ordered_json& fn = call.contains("function") ? call["function"] : call;
                call["inkling_call"] = call_json(fn.value("name", ""), fn.contains("arguments") ? fn["arguments"] : nlohmann::ordered_json());
            }
        }
    } catch (const std::invalid_argument& e) {
        throw ModelRequestError(400, false, e.what(), "invalid_request_error");
    }
    if (tools.is_array() && !tools.empty()) {
        inputs.tools = tools;
        inputs.extra_context["inkling_tool_specs"] = tool_specs_json(tools);
    }
    inputs.messages = prepared;
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool Inkling2::configure_parameter(std::string parameter_name, const std::any& value) {
    if (parameter_name == "reasoning_effort") {
        try {
            const std::string effort = std::any_cast<std::string>(value);
            if (!valid_effort(effort)) {
                header_print("WARNING", "Reasoning effort must be none, minimal, low, medium, high, xhigh, max or a number in [0, 0.99]!");
                return true;
            }
            this->reasoning_effort = effort;
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "enable_think") {
        try {
            const bool on = std::any_cast<bool>(value);
            if (!on) {
                this->reasoning_effort = "none";
            } else if (this->reasoning_effort == "none") {
                this->reasoning_effort = "high";
            }
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }
    if (parameter_name == "toggle_think") {
        this->reasoning_effort = this->reasoning_effort == "none" ? "high" : "none";
        header_print("FLM", "Thinking effort: " << this->reasoning_effort);
        return true;
    }
    return this->AutoModel::configure_parameter(parameter_name, value);
}

bool Inkling2::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    // Without the tower an image would template into placeholders nothing fills: drop it loudly instead.
    auto* engine = dynamic_cast<inkling2_npu*>(this->lm_engine);
    const bool vision_ok = engine && engine->has_vision_tower();
    if (!vision_ok && !input.images.empty()) {
        header_print("WARNING", "Inkling-2-Mini is loaded without a vision tower (vision_weight.q4nx), ignoring "
                                << input.images.size() << " image(s)");
        input.images.clear();
        input.image_payload_types.clear();
    }
    if (!input.audios.empty()) {
        header_print("WARNING", "Inkling-2-Mini does not take audio, ignoring " << input.audios.size() << " audio input(s)");
        input.audios.clear();
        input.audio_payload_types.clear();
    }

    this->profiler_list[TKOEN_ENCODE_TIME].start();

    // Decode every image BEFORE templating: a failed image must never get a placeholder, or the payload falls out
    // of step with the prompt and one image's rows land in the next one's places.
    std::vector<image_data_t> images;
    auto stage = [&](const std::string& source, bool base64) {
        image_data_t image;
        if (!this->load_image(source, base64, image)) return false;
        images.push_back(std::move(image));
        return true;
    };

    std::string templated_text;
    if (!input.messages.empty()) {   // already formatted messages, usually from the REST API
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        for (const auto& item : input.messages) {
            nlohmann::ordered_json entry = item;
            entry.erase("audios");
            entry.erase("images");
            if (vision_ok && item.contains("images") && item["images"].is_array() && !item["images"].empty()) {
                // content parts: the template turns each image part into <|content_image|><|unused_200054|><|end_message|>
                nlohmann::ordered_json content = nlohmann::ordered_json::array();
                for (const auto& img : item["images"]) {
                    if (!img.is_string() || !stage(img.get<std::string>(), true)) {
                        header_print("ERROR", "Skipping an image that failed to load");
                        continue;
                    }
                    content.push_back({ {"type", "image"} });
                }
                const nlohmann::ordered_json& text = item.contains("content") ? item["content"] : nlohmann::ordered_json("");
                if (text.is_array()) {
                    for (const auto& part : text) content.push_back(part);
                } else if (text.is_string() && !text.get<std::string>().empty()) {
                    content.push_back({ {"type", "text"}, {"text", text} });
                }
                entry["content"] = content;
            }
            messages.push_back(entry);
        }
        input.messages = messages;
        templated_text = this->apply_chat_template(input.messages, input.tools, meta_info.continue_final_message);
    }
    else {                           // plain text, usually from the cli
        nlohmann::ordered_json content = nlohmann::ordered_json::array();
        for (const auto& source : input.images) {
            if (stage(source, false)) content.push_back({ {"type", "image"} });
            else header_print("ERROR", "Skipping an image that failed to load: " << source);
        }
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        if (content.empty()) {
            messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        } else {
            content.push_back({ {"type", "text"}, {"text", input.prompt} });
            messages.push_back({ {"role", "user"}, {"content", content} });
        }
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // The template emits ONE placeholder per image; the model wants one per cell. Expand in place.
    constexpr int image_id = inkling2_npu::image_token_id;
    std::vector<int> tokens;
    std::vector<size_t> image_token_counts;   // per staged image, for the prompt-cache plan
    if (images.empty()) {
        tokens = std::move(tokens_init);
    } else {
        size_t total = 0;
        for (const auto& im : images) total += static_cast<size_t>(engine->image_tokens_for(im.height, im.width));
        tokens.reserve(tokens_init.size() + total);
        size_t counter = 0;
        for (int id : tokens_init) {
            if (id == image_id && counter < images.size()) {
                const int n = engine->image_tokens_for(images[counter].height, images[counter].width);
                tokens.insert(tokens.end(), static_cast<size_t>(n), id);
                image_token_counts.push_back(static_cast<size_t>(n));
                counter++;
            } else {
                tokens.push_back(id);
            }
        }
        if (counter != images.size()) {
            header_print("ERROR", "templated " << counter << " image placeholder(s) for " << images.size() << " image(s); refusing to prefill");
            meta_info.error_message = "Prompt has " + std::to_string(counter) + " image placeholder(s) for " + std::to_string(images.size()) + " image(s)";
            meta_info.error_code = 500;
            this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
            return false;
        }
        header_print("FLM", "Total images: " << counter << " (" << total << " image tokens)");
    }
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // Prefix slots. The prompt ends with the user's <|end_message|>, which is also how the next
    // turn renders it, so the whole prompt is saved (trim 0). The tool declaration and the effort
    // line come before the first <|message_user|>: that head gets its own snapshot.
    const bool slot_mode = this->_prefix_slot_mode(meta_info, tokens.size(), 0);
    size_t match_len = 0;
    std::vector<int> full_tokens;          // what this prompt leaves in the engine
    // content of each image (its size and its pixels): the slot cache and the prompt cache tell same-size pictures apart by it
    std::vector<uint64_t> image_hashes;
    for (const auto& im : images) {
        const uint64_t seed = 0xcbf29ce484222325ull ^ (static_cast<uint64_t>(im.height) << 32 | static_cast<uint64_t>(im.width));
        image_hashes.push_back(prefix_hash_bytes(im.pixels.data(), static_cast<size_t>(im.height) * im.width * 3, seed));
    }
    if (slot_mode) {
        full_tokens = tokens;
        match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, image_hashes);
    } else {
        this->_prefix_legacy_path();   // the in-place path below moves the engine on without telling the cache
    }

    // Prompt cache: _shared_insert skips the part of the prompt the engine already holds; plan_prompt_reuse tells
    // both sides the same skip, so the images left in the payload are the ones on the rows that get prefilled.
    std::vector<prompt_media_t> media;
    prompt_media_from_counts(tokens, image_id, image_token_counts, prompt_media_t::kImage, media);
    for (size_t i = 0; i < media.size() && i < image_hashes.size(); i++) {
        media[i].hash = image_hashes[i] != 0 ? image_hashes[i] : 1;   // 0 means "not hashed"
    }
    const prompt_reuse_t plan = this->_plan_reuse(tokens, media);
    const size_t drop = std::min(plan.skipped_media[prompt_media_t::kImage], images.size());
    std::vector<inkling2_image_t> views;
    for (size_t i = drop; i < images.size(); i++) {
        inkling2_image_t view;
        view.rgb = images[i].pixels.data();
        view.height = images[i].height;
        view.width = images[i].width;
        views.push_back(view);
    }
    inkling2_payload_t payload;
    payload.images = views.data();
    payload.num_images = static_cast<int>(views.size());
    void* payload_ptr = views.empty() ? nullptr : static_cast<void*>(&payload);
    // one past the last image row that gets prefilled, relative to the skip (0 when none is)
    const int last_image_token_index = static_cast<int>(plan.media_end);

    bool success;
    if (slot_mode) {
        success = this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, image_hashes, match_len, "<|message_user|>",
                                       payload_ptr, last_image_token_index, is_cancelled);
    } else if (payload_ptr) {
        success = this->_shared_insert(meta_info, tokens, is_cancelled, payload_ptr, last_image_token_index);
    } else {
        success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
    }
    for (auto& im : images) this->image_reader_.recycle(im);
    return success;
}

std::string Inkling2::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Inkling2::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}
