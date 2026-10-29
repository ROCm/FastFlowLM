/// \file automodel.cpp
/// \brief automodel class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the auto_model class

#include "AutoModel/automodel.hpp"
#include "chat/chat_continue.hpp"
#include "utils/npu_platform.hpp"

#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {

/// \brief the context length a model should actually be loaded with
/// \param model_info the catalog entry
/// \param default_context_length the --ctx-len value, or -1 when not given
/// \return the resolved length
/// \note Single-turn models -- dedicated translation models, say -- do not
///       support an arbitrary override, so they always get their own default.
int resolve_context_length(const nlohmann::ordered_json& model_info,
                           int default_context_length) {
    if (default_context_length == -1) {
        return model_info["default_context_length"].get<int>();
    }
    const bool single_turn =
        model_info.contains("label") &&
        std::find(model_info["label"].begin(), model_info["label"].end(),
                  "single-turn") != model_info["label"].end();
    if (single_turn) {
        header_print("FLM", "Single-turn model, 1k max context length allowed only!");
        return model_info["default_context_length"].get<int>();
    }
    return default_context_length;
}

}  // namespace


ModelRequestError::ModelRequestError(
    int http_code, bool session_cleared, std::string message, std::string type)
    : std::runtime_error(std::move(message)),
      http_code_(http_code), session_cleared_(session_cleared), type_(std::move(type)) {}

int ModelRequestError::http_code() const noexcept { return http_code_; }
bool ModelRequestError::session_cleared() const noexcept { return session_cleared_; }
const std::string& ModelRequestError::type() const noexcept { return type_; }

AutoModel::AutoModel(flm_rt::device* npu_device_inst, std::string current_model) {
    this->npu_device_inst = npu_device_inst;
    this->current_model = current_model;
    this->total_tokens = 0;
    this->profiler_list.resize(PROFILER_TYPE_NUM);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i] = profiler();
    }
    this->last_prefill_time = { 0, "us" };
    this->token_history.reserve(MAX_L);
}

AutoModel::~AutoModel() {
    // Members go in reverse declaration order, which would free npu -- the
    // xclbins and their hardware contexts -- before backend_, whose engine
    // holds apps pointing into those contexts. Tear down in the order
    // _shared_load_backend does: the engine first.
    this->lm_engine = nullptr;
    this->backend_.reset();
    this->npu.reset();
}


std::string AutoModel::get_current_model() {
    return this->current_model;
}

/// \brief Setup the tokenizer
/// \note The function will setup the tokenizer
nlohmann::json AutoModel::_shared_setup_tokenizer(std::string model_path) {
    // load tokenizer configurations
    #ifdef _WIN32
    std::string tokenizer_config_path = model_path + "\\tokenizer_config.json";
    #else
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    #endif
    std::ifstream fs_config(tokenizer_config_path, std::ios::in | std::ios::binary);
    if (fs_config.fail()) {
        std::cerr << "Cannot open " << tokenizer_config_path << std::endl;
        exit(1);
    }
    std::string data_config;
    fs_config.seekg(0, std::ios::end);
    size_t size_config = static_cast<size_t>(fs_config.tellg());
    fs_config.seekg(0, std::ios::beg);
    data_config.resize(size_config);
    fs_config.read(data_config.data(), size_config);
    fs_config.close();
    auto tokenizer_config = nlohmann::json::parse(data_config);
    // check if bos_token is null
    if (tokenizer_config["bos_token"].is_null()) {
        this->has_bos_token = false;
    }
    else {
        this->has_bos_token = true;
    }
    // load chat template
    //check if chat_template.jinja exists
    #ifdef _WIN32
    std::string chat_template_path = model_path + "\\chat_template.jinja";
    #else
    std::string chat_template_path = model_path + "/chat_template.jinja";
    #endif
    if (std::filesystem::exists(chat_template_path) == false) {
        // no jinja file exist, see if chat_template in tokenizer_config.json
        if (tokenizer_config.find("chat_template") == tokenizer_config.end()) {
            header_print("ERROR", "No template file found and no chat_template in tokenizer_config.json");
            exit(1);
        }
    }
    else {
        std::ifstream fs_template(chat_template_path, std::ios::in | std::ios::binary);
        if (fs_template.fail()) {
            std::cerr << "Cannot open " << chat_template_path << std::endl;
            exit(1);
        }
        std::string chat_template;
        fs_template.seekg(0, std::ios::end);
        size_t size_template = static_cast<size_t>(fs_template.tellg());
        fs_template.seekg(0, std::ios::beg);
        chat_template.resize(size_template);
        fs_template.read(chat_template.data(), size_template);
        fs_template.close();
        // overwrite chat_template in tokenizer_config.json
        tokenizer_config["chat_template"] = chat_template;
    }
    
    if (tokenizer_config["eos_token"].is_null()) {
        header_print("ERROR", "eos_token is null in tokenizer_config.json");
        exit(1);
    }
    this->chat_tmpl = std::make_unique<minja::chat_template>(
        tokenizer_config["chat_template"],
        this->has_bos_token ? tokenizer_config["bos_token"] : "",
        tokenizer_config["eos_token"]
    );

    if (this->has_bos_token) {
        // Some packages (Gemma 4) name bos_token but state no bos_token_id; a
        // backend that validated its package supplies the id.
        if (const auto forced = this->backend_ ? this->backend_->forced_bos_id()
                                               : std::nullopt) {
            this->bos_token_id = *forced;
        }
        else if (!tokenizer_config["bos_token_id"].is_number_integer()) {
            header_print("ERROR", "bos_token is set in tokenizer_config.json but bos_token_id is missing or not an integer");
            exit(1);
        }
        else {
            this->bos_token_id = tokenizer_config["bos_token_id"].get<int>();
        }
    }
    else {
        this->bos_token_id = -1;
    }
    this->eos_token = tokenizer_config["eos_token"].get<std::string>();
    // A backend that cross-validated its own package knows the stop ids better
    // than tokenizer_config.json does.
    if (const auto forced = this->backend_ ? this->backend_->forced_eos_ids()
                                           : std::nullopt) {
        this->eos_token_ids = *forced;
    } else {
        if (!tokenizer_config["eos_token_id"].is_array()) {
            header_print("ERROR", "eos_token_id is missing or not an array in tokenizer_config.json");
            exit(1);
        }
        for (auto& token : tokenizer_config["eos_token_id"]) {
            if (!token.is_number_integer()) {
                header_print("ERROR", "eos_token_id must be an array of integers in tokenizer_config.json");
                exit(1);
            }
            this->eos_token_ids.push_back(token.get<int>());
        }
    }
    this->user_system_prompt = "";
    this->extra_context["user_system_prompt"] = this->user_system_prompt;
    this->_detect_chat_format(tokenizer_config["chat_template"].get<std::string>());
    return tokenizer_config;
}

void AutoModel::_detect_chat_format(const std::string& template_src) {
    auto is_token = [this](const std::string& text) {
        std::vector<int> ids = this->tokenizer->encode(text);
        if (this->has_bos_token && !ids.empty() && ids.front() == this->bos_token_id) {
            ids.erase(ids.begin());
        }
        return ids.size() == 1;
    };
    this->chat_format = chat_format_detect(template_src, is_token);

    // markers that hand the turn to a tool end generation like eos does
    std::string stops;
    for (const std::string& marker : this->chat_format.extra_stops) {
        std::vector<int> ids = this->tokenizer->encode(marker);
        if (this->has_bos_token && !ids.empty() && ids.front() == this->bos_token_id) {
            ids.erase(ids.begin());
        }
        if (ids.size() != 1) {
            header_print("WARNING", "Stop marker " << marker << " is not a single token, generation will not stop on it");
            continue;
        }
        if (!this->is_eos(ids[0])) {
            this->eos_token_ids.push_back(ids[0]);
        }
        stops += " " + marker;
    }
    header_print("FLM", "Chat format: " << chat_format_name(this->chat_format.kind)
                 << (this->chat_format.has_reasoning() ? " with reasoning" : "")
                 << (stops.empty() ? "" : ", also stops at" + stops));
}


void AutoModel::_shared_load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    if (this->is_model_loaded && this->model_path == model_path) {
        header_print("FLM", "Model already loaded: " << this->model_path);
        return;
    }
    const int context_length =
        resolve_context_length(model_info, default_context_length);
    this->_shared_initialize_model_state(
        std::move(model_path), std::move(model_info), context_length);
    this->_shared_initialize_legacy_npu(enable_preemption);
}

void AutoModel::_shared_initialize_model_state(
    std::string model_path, json, int context_length) {
    this->model_path = std::move(model_path);
    header_print("FLM", "Loading model: " << this->model_path);
    this->lm_config = std::make_unique<LM_Config>();
    this->lm_config->from_pretrained(this->model_path);
    this->MAX_L = context_length;
    this->is_model_loaded = true;
    this->token_history.clear();
    this->token_history.reserve(this->MAX_L);
    this->tokenizer = std::make_unique<Tokenizer>(this->model_path);
    this->last_token = -1;
    this->total_tokens = 0;
}

void AutoModel::_shared_load_backend(std::string model_path, json model_info,
                                     int default_context_length,
                                     bool enable_preemption,
                                     const std::string& requested_backend,
                                     const nlohmann::json* tokenizer_config) {
    if (!model_info.contains("details") ||
        !model_info["details"].contains("family")) {
        throw std::runtime_error("Model entry has no details.family");
    }
    const std::string family = model_info["details"]["family"].get<std::string>();

    auto& registry = flm::backend::BackendRegistry::instance();
    std::string source;
    // The entry carries the kernel flow its artifacts were built for -- the
    // weights differ between flows, so this belongs to the model rather than
    // to the build. model_list wrote it there from the tag the entry was
    // found under; --backend and FLM_BACKEND still win over it.
    const std::string catalog_backend = model_info.value(
        flm::backend::kBackendKey, std::string(flm::backend::kFlmBackendId));
    const std::string id = flm::backend::resolve_backend_id(
        family, catalog_backend, requested_backend, &source);

    // Same model on the same backend is a no-op; a different backend is a real
    // reload even when the path has not changed.
    if (this->is_model_loaded && this->model_path == model_path &&
        this->backend_ && this->backend_->id() == id) {
        header_print("FLM", "Model already loaded: " << this->model_path);
        return;
    }

    const auto traits = registry.traits(family, id);
    if (enable_preemption && !traits.supports_preemption) {
        throw std::invalid_argument(
            "Backend '" + id + "' does not support preemption");
    }
    const int context_length =
        resolve_context_length(model_info, default_context_length);
    if (traits.max_context_length != 0 &&
        (context_length < 1 ||
         static_cast<std::uint32_t>(context_length) > traits.max_context_length)) {
        throw std::out_of_range(
            "Backend '" + id + "' context length must be in 1.." +
            std::to_string(traits.max_context_length));
    }

    // Tear the old model down before building the new one: the engines hold
    // device memory, and a half-loaded model must never look loaded.
    this->backend_.reset();
    this->lm_engine = nullptr;
    this->is_model_loaded = false;

    try {
        this->_shared_initialize_model_state(
            std::move(model_path), model_info, context_length);
        if (traits.needs_npu_xclbin) {
            this->_shared_initialize_legacy_npu(enable_preemption);
        } else {
            this->npu.reset();
            this->enable_preemption = false;
        }

        flm::backend::BackendContext context;
        context.model_path = this->model_path;
        context.model_info = model_info;
        context.config = this->lm_config.get();
        context.npu = this->npu.get();
        context.device = this->npu_device_inst;
        context.context_length = static_cast<std::uint32_t>(context_length);
        context.enable_preemption = this->enable_preemption;
        context.tokenizer_config = tokenizer_config;

        this->backend_ = registry.create(family, id, context);
        this->lm_engine = &this->backend_->engine();
    } catch (...) {
        this->backend_.reset();
        this->lm_engine = nullptr;
        this->tokenizer.reset();
        this->sampler.reset();
        this->lm_config.reset();
        this->is_model_loaded = false;
        throw;
    }
    // Hide the log now.
    // header_print("FLM", "Backend: " << id << " (from " << source << ")");
}

void AutoModel::_shared_after_inference_failure(bool poisoned) {
    this->total_tokens = 0;
    this->last_token = -1;
    this->token_history.clear();
    this->checkpoint_his.clear();
    // A poisoned engine cannot be driven at all, not even to clear itself.
    if (!poisoned && this->lm_engine) {
        try { this->lm_engine->clear_context(); } catch (...) {}
    }
    if (this->sampler) this->sampler->reset_penalties();
}

void AutoModel::_shared_guard_poisoned() const {
    if (this->backend_ && this->backend_->poisoned()) {
        // Same condition as the throw in the frontend's inference-failure path,
        // so it says the same thing: one poisoned model must not be described
        // two ways depending on whether this is the request that broke it.
        throw ModelRequestError(500, true,
            "Backend '" + this->backend_->id() +
            "' failed; unload/reload is required because the model is poisoned");
    }
}

void AutoModel::_shared_initialize_legacy_npu(bool enable_preemption) {
    if (this->npu_device_inst == nullptr) {
        header_print("ERROR", "NPU device instance is nullptr");
        exit(1);
    }
    this->npu = std::make_unique<npu_xclbin_manager>(
        npu_device::device_npu2, this->npu_device_inst, enable_preemption);
    this->enable_preemption = enable_preemption;
}

std::string AutoModel::generate_with_prompt(
    chat_meta_info_t& meta_info,
    lm_uniform_input_t& input,
    int length_limit,
    std::ostream& os,
    std::function<bool()> is_cancelled) {
    if (!insert(meta_info, input, is_cancelled)) return {};
    return generate(meta_info, length_limit, os, std::move(is_cancelled));
}

size_t AutoModel::_prefix_resume(PrefixSlotCache& cache, const std::vector<int>& full_tokens, const std::vector<uint64_t>& image_hashes) {
    std::vector<int> matched;
    cache.begin_report();
    const PrefixSlotCache::resume_t resumed = cache.resume(full_tokens, image_hashes, matched);
    if (!resumed.resumed) {
        this->_prefix_cold_start(cache);
        header_print("FLM", "No saved prefix matches: " << resumed.miss_reason << "; prefilling " << full_tokens.size() << " tokens");
        return 0;
    }
    this->total_tokens = resumed.match_len;
    this->token_history = std::move(matched);
    // a cold prompt starts with an empty penalty history (clear_context); resuming must too,
    // or the tokens of earlier requests change what this one generates
    this->sampler->reset_penalties();
    return resumed.match_len;
}

void AutoModel::_prefix_cold_start(PrefixSlotCache& cache) {
    this->_clear_live_context(); // zeroes the engine and the token history, not the slots
    cache.clear_live();          // the engine's rows hold nothing now
}

void AutoModel::_prefix_attach(void* engine, const flm_state_api* api, int image_token_id) {
    static const bool disabled = std::getenv("FLM_DISABLE_PREFIX_SLOTS") != nullptr;
    this->prefix_cache_.image_token_id = image_token_id;
    this->prefix_cache_.configure(this->prefix_cache_mb_arg_, this->prefix_slots_arg_, this->prefix_min_gain_arg_);
    if (disabled || engine == nullptr || api == nullptr) {
        this->prefix_cache_.attach(nullptr, nullptr);
        return;
    }
    this->prefix_cache_.attach(engine, api);
}

bool AutoModel::_prefix_slot_mode(const chat_meta_info_t& meta_info, size_t n_tokens, size_t trim) const {
    return this->prefix_cache_.enabled() && meta_info.restore_allowed && n_tokens > trim;
}

void AutoModel::_prefix_legacy_path() {
    if (this->prefix_cache_.enabled()) {
        this->prefix_cache_.forget_all();
    }
}

bool AutoModel::_prefix_text_insert(chat_meta_info_t& meta_info, std::vector<int>& tokens, size_t trim, const std::string& boundary_marker,
                                    std::function<bool()> is_cancelled, std::optional<int> requested_max_new_tokens) {
    // _shared_insert erases the part the engine already holds from `tokens`; the cache needs the whole prompt
    const std::vector<int> full_tokens(tokens.begin(), tokens.end() - trim);
    const size_t match_len = this->_prefix_resume(this->prefix_cache_, full_tokens, {});
    tokens.resize(tokens.size() - trim);
    return this->_prefix_insert(this->prefix_cache_, meta_info, tokens, full_tokens, {}, match_len, boundary_marker, nullptr, 0, std::move(is_cancelled),
                                requested_max_new_tokens);
}

bool AutoModel::_prefix_insert(PrefixSlotCache& cache, chat_meta_info_t& meta_info, std::vector<int>& tokens, const std::vector<int>& full_tokens,
                               const std::vector<uint64_t>& image_hashes, size_t match_len, const std::string& boundary_marker,
                               void* payload, int last_image_token_index, std::function<bool()> is_cancelled,
                               std::optional<int> requested_max_new_tokens) {
    bool success = true;
    uint64_t head_prefill_ns = 0;
    // Save the end of system + tools while prefilling, so a prompt that shares only that part
    // (another conversation, a side request) can resume there.
    size_t boundary = 0;
    {
        const std::vector<int> marker = this->tokenizer->encode(boundary_marker);
        for (size_t i = 1; i + marker.size() <= full_tokens.size() && !marker.empty(); i++) {
            if (std::equal(marker.begin(), marker.end(), full_tokens.begin() + i)) {
                boundary = i;
                break;
            }
        }
        if (boundary < cache.min_boundary) boundary = 0;
    }
    if (payload == nullptr && boundary > match_len && boundary < full_tokens.size()) {
        std::vector<int> head(full_tokens.begin(), full_tokens.begin() + boundary);
        chat_meta_info_t head_meta;
        head_meta.max_prefill_len = meta_info.max_prefill_len;
        success = this->_shared_insert(head_meta, head, is_cancelled, nullptr, 0, std::nullopt, false);
        if (success) {
            head_prefill_ns = head_meta.prefill_duration;
            const std::vector<int> head_tokens(full_tokens.begin(), full_tokens.begin() + boundary);
            cache.set_live(head_tokens);
            cache.save(head_tokens, {}, true, 0);
        } else {
            meta_info.stop_reason = head_meta.stop_reason;
        }
    }
    if (success) {
        success = payload != nullptr
            ? this->_shared_insert(meta_info, tokens, is_cancelled, payload, last_image_token_index, requested_max_new_tokens)
            : this->_shared_insert(meta_info, tokens, is_cancelled, nullptr, 0, requested_max_new_tokens);
    }
    if (!success) {
        this->_prefix_cold_start(cache); // cancelled or failed half way: the engine holds part of a prompt; the slots are intact
        return false;
    }
    // the second pass only saw the tokens after the boundary slot as new
    meta_info.cached_prompt_tokens = static_cast<int>(match_len);
    meta_info.prefill_duration += head_prefill_ns;
    cache.set_live(full_tokens);
    // the end of this prompt keeps only what follows the shared head, when a slot holds that head
    if (!cache.boundary_only) {
        cache.save(full_tokens, image_hashes, false, cache.end_kv_from(full_tokens, boundary));
    }
    meta_info.prefix_cache = cache.report();
    return true;
}

void AutoModel::_shared_validate_capacity(std::size_t rendered_tokens,
                                          std::optional<int> requested) const {
    if (!this->backend_ || this->backend_->max_decode_length() == 0) return;
    const std::size_t cap = this->decode_cap();
    const auto normalized = normalize_requested_max_new_tokens(requested);
    if (rendered_tokens >= cap ||
        (normalized && static_cast<std::size_t>(*normalized) > cap - rendered_tokens)) {
        std::ostringstream message;
        message << "Request exceeds the " << cap
                << "-token limit of backend '" << this->backend_->id()
                << "': rendered prompt has " << rendered_tokens << " tokens";
        if (normalized) message << " and requested output has " << *normalized << " tokens";
        throw ModelRequestError(400, false, message.str());
    }
}

bool AutoModel::_shared_insert(chat_meta_info_t& meta_info, std::vector<int>& tokens, std::function<bool()> is_cancelled, void* payload, int first_len_run, std::optional<int> requested_max_new_tokens, bool sample_next) {
    this->_shared_validate_capacity(tokens.size(), requested_max_new_tokens);

    // a prefill-only pass (sample_next false) is the head of a longer prompt: the
    // turn, its constraint and the sampler's penalty history all belong to the
    // pass that finishes the prompt
    if (sample_next) {
        this->_begin_turn(meta_info, tokens);
    }

    // print token history
    // header_print("DEBUG", "Current token history: ");
    // for (size_t i = 0; i < this->token_history.size(); i++) {
    //     std::cout << this->token_history[i] << " ";
    // }
    // std::cout << std::endl;
    // // print tokens to insert
    // header_print("DEBUG", "Tokens to insert: ");
    // for (size_t i = 0; i < tokens.size(); i++) {
    //     std::cout << tokens[i] <<  " ";
    // }
    // std::cout << std::endl;

    // prefix check for tokens and token history to see if we can skip some tokens
    const size_t idx = this->token_history.size();
    size_t skip_count = 0;
    for (size_t i = 0; i < idx; i++) {
        if (i < tokens.size() && tokens[i] == this->token_history[i]) {
            skip_count++;
        }
        else {
            break;
        }
    }
    if (skip_count != idx) {
        if (is_server_mode) {
            header_print("FLM", "Conversation context diverged from cache, clearing context...");
        }
        clear_context();
        skip_count = 0;
    }
    else if (skip_count >= tokens.size() && !tokens.empty()) {
        // The engine holds the whole prompt already (the same request again), but
        // sampling needs the logits of its last token, and the cache cannot step
        // back one: read it in again from the start.
        clear_context();
        skip_count = 0;
    }
    tokens.erase(tokens.begin(), tokens.begin() + skip_count);

    if (this->total_tokens + tokens.size() >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping prefilling...");
        return false;
    }
    for (int token : tokens){
        this->token_history.push_back(token);
    }
    buffer<bf16> y;

    auto prefill_start_time = this->profiler_list[PREFILL_TIME].start();
    
    y = _chunked_insert(meta_info, tokens, is_cancelled, payload, first_len_run);

    auto prefill_end_time = this->profiler_list[PREFILL_TIME].stop(tokens.size());
    meta_info.prefill_duration = (uint64_t)time_utils::duration_ns(prefill_start_time, prefill_end_time).first;
    // `tokens` was trimmed to the uncached suffix above, so add the prefix served
    // from the KV cache back on: usage.prompt_tokens is the whole prompt, and the
    // cached part is reported separately rather than subtracted.
    meta_info.cached_prompt_tokens = static_cast<int>(skip_count);
    meta_info.prompt_tokens = static_cast<int>(skip_count + tokens.size());

    if (meta_info.stop_reason == CANCEL_DETECTED) {
        return false;
    }

    this->total_tokens += tokens.size();
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping prefilling...");
    }
    if (!sample_next) {
        return true;
    }
    this->profiler_list[SAMPLING_TIME].start();
    this->last_token = this->_sample(y, meta_info);
    meta_info.insert_sample_pending = true;
    this->profiler_list[SAMPLING_TIME].stop(1);
    return true;
}

buffer<bf16> AutoModel::_chunked_insert(chat_meta_info_t& meta_info, std::vector<int>& tokens, std::function<bool()> is_cancelled, void* payload, int first_len_run) {
    int max_prefill_len = meta_info.max_prefill_len;
    // log2 of a non-positive length is not a number to shift by; an unset or
    // degenerate limit means "no chunking", which the < 512 branch below is
    if (max_prefill_len > 0) {
        // make max_prefill_len a 2^n
        max_prefill_len = 1 << static_cast<int>(std::ceil(std::log2(max_prefill_len)));
    }
    buffer<bf16> y;
    if (max_prefill_len < 512) {
        if (is_cancelled()) {
            meta_info.stop_reason = CANCEL_DETECTED;
            return y;
        }
        y = this->lm_engine->prefill(tokens, payload);
    }
    else{
        if (first_len_run > 0) {
            int new_max_len = 1 << static_cast<int>(std::ceil(std::log2(first_len_run)));
            if (new_max_len > max_prefill_len) {
                max_prefill_len = new_max_len;
            }
        }
        // A payload's tokens have to stay in the first chunk, which the
        // first_len_run widening above sized for them.
        const bool aligned = payload == nullptr && this->backend_ &&
                             this->backend_->aligns_prefill_chunks();
        const int lead = aligned
            ? this->lm_engine->get_current_context_length() % max_prefill_len
            : 0;
        int chunks = (lead + static_cast<int>(tokens.size()) + max_prefill_len - 1) / max_prefill_len;
        for (int i = 0; i < chunks; i++) {
            int start = std::max(0, i * max_prefill_len - lead);
            int end = std::min(static_cast<int>(tokens.size()), (i + 1) * max_prefill_len - lead);
            std::vector<int> chunk_tokens(tokens.begin() + start, tokens.begin() + end);
            header_print("FLM", "Prefill chunk " + std::to_string(i+1) + "/" + std::to_string(chunks) + " with " + std::to_string(chunk_tokens.size()) + " tokens");
            if (is_cancelled()) {
                meta_info.stop_reason = CANCEL_DETECTED;
                break;
            }
            buffer<bf16> chunk_y = this->lm_engine->prefill(chunk_tokens, (i == 0)? payload : nullptr);
            if (i == chunks - 1) {
                // Moved, not copied: buffer's copy is shallow and does not take
                // ownership, so a copy of logits the engine owns (the core
                // engines return them that way) dangles once chunk_y goes.
                y = std::move(chunk_y);
            }
        }
    }
    return y;
}

std::string AutoModel::_render_chat_template(minja::chat_template_inputs& inputs, bool continue_final_message,
                                             const minja::chat_template_options& options) {
    if (!continue_final_message) {
        inputs.add_generation_prompt = true;
        return this->chat_tmpl->apply(inputs, options);
    }
    if (inputs.messages.empty() || inputs.messages.back().value("role", "") != "assistant") {
        throw ModelRequestError(400, false, "continue_final_message requires the last message to have role \"assistant\"",
                                "invalid_request_error");
    }
    const auto render = [&](const nlohmann::ordered_json& messages, bool add_generation_prompt) {
        minja::chat_template_inputs chat = inputs;
        chat.messages = messages;
        chat.add_generation_prompt = add_generation_prompt;
        return this->chat_tmpl->apply(chat, options);
    };
    try {
        chat_continued continued = chat_continue_prompt(render, inputs.messages);
        // the output parser looks for an open reasoning block in this turn only, not in what the user wrote earlier
        this->continued_turn_ = std::move(continued.turn);
        return continued.prompt;
    } catch (const std::invalid_argument& e) {
        throw ModelRequestError(400, false, e.what(), "invalid_request_error");
    }
}

void AutoModel::_begin_turn(chat_meta_info_t& meta_info, const std::vector<int>& prompt_tokens) {
    // keep the end of the prompt as the model sees it: generation continues from
    // there, so it tells the output parser whether a reasoning block is already open.
    // A continued message may have opened one anywhere in its text, so then it is
    // the whole continued turn.
    constexpr size_t PROMPT_TAIL_TOKENS = 32;
    meta_info.prompt_tail.clear();
    if (meta_info.continue_final_message) {
        meta_info.prompt_tail = this->continued_turn_;
    }
    else {
        for (size_t i = prompt_tokens.size() > PROMPT_TAIL_TOKENS ? prompt_tokens.size() - PROMPT_TAIL_TOKENS : 0;
             i < prompt_tokens.size(); i++) {
            meta_info.prompt_tail += this->token_text(prompt_tokens[i]);
        }
    }
    if (meta_info.constraint) {
        meta_info.constraint->start(chat_format_prefill(this->chat_format, meta_info.prompt_tail,
                                                        meta_info.continue_final_message));
    }
}

std::string AutoModel::token_text(int token) {
    try {
        return this->tokenizer->run_time_decoder(token);
    } catch (const std::runtime_error&) {
        // added tokens outside the byte map, e.g. DeepSeek's <｜Assistant｜>,
        // are stored as plain text; the full decoder handles them
        return this->tokenizer->decode({token});
    }
}

int AutoModel::_sample(buffer<bf16>& logits, chat_meta_info_t& meta_info) {
    decode_constraint* constraint = meta_info.constraint.get();
    const bool masked = constraint != nullptr && constraint->fill_mask(this->constraint_mask_);
    sample_logprobs_t raw;
    raw.top_n = meta_info.logprobs_top_n;
    // The mask goes to the sampler, which applies it to its own copy: the logits
    // live in the engine's device-mapped buffer and are never written here.
    int token = this->sampler->sample(logits, masked ? &this->constraint_mask_ : nullptr,
                                      meta_info.logprobs_top_n >= 0 ? &raw : nullptr);
    auto allowed = [this](size_t i) {
        return i / 32 < this->constraint_mask_.size() && ((this->constraint_mask_[i / 32] >> (i % 32)) & 1u);
    };
    if (masked && (token < 0 || !allowed(static_cast<size_t>(token)))) {
        // Only happens when the logits themselves are broken (NaN everywhere
        // makes the argmax land on 0): stay inside the grammar regardless.
        int first = -1;
        int best = -1;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < logits.size(); i++) {
            if (!allowed(i)) {
                continue;
            }
            const float v = static_cast<float>(logits[i]);
            if (first < 0) first = static_cast<int>(i);
            if (v > best_logit) {  // NaN never compares greater
                best = static_cast<int>(i);
                best_logit = v;
            }
        }
        if (first < 0) {
            // nothing allowed at all: keep the sample, the constraint gives itself up on it
            header_print("WARNING", "The decoding constraint allows no token here, keeping sampled token " << token);
        } else {
            header_print("WARNING", "Sampled token " << token << " is outside the decoding constraint, using "
                         << (best >= 0 ? best : first) << " instead");
            token = best >= 0 ? best : first;
        }
    }
    if (constraint != nullptr) {
        constraint->accept(token);
    }
    // The chosen token's logprob is read here rather than in the sampler: the
    // fallback above may have replaced what the sampler picked.
    meta_info.last_sample_logged = meta_info.logprobs_top_n >= 0 &&
        this->_record_logprob(meta_info, token, static_cast<float>(logits[token]), raw);
    meta_info.insert_sample_pending = false;
    return token;
}

bool AutoModel::_record_logprob(chat_meta_info_t& meta_info, int token, float token_logit,
                                sample_logprobs_t& raw) {
    // Special tokens never reach the output, so they have no place in its logprobs.
    if (!this->is_normal_token(token)) {
        return false;
    }
    meta_info.logprobs.push_back(chat_token_logprob{token, token_logit - raw.log_norm, std::move(raw.top)});
    return true;
}

void AutoModel::_force_token(chat_meta_info_t& meta_info, int token) {
    this->_skip_insert_sample(meta_info);
    if (meta_info.constraint) {
        meta_info.constraint->accept_forced(token);
    }
}

void AutoModel::_skip_insert_sample(chat_meta_info_t& meta_info) {
    if (meta_info.insert_sample_pending) {
        if (meta_info.constraint) {
            meta_info.constraint->rollback(1);
        }
        // the sample never reaches the output, so neither does its logprob
        if (meta_info.last_sample_logged) {
            meta_info.logprobs.pop_back();
            meta_info.last_sample_logged = false;
        }
    }
    meta_info.insert_sample_pending = false;
}

std::shared_ptr<decode_constraint> AutoModel::make_constraint(const chat_constraint_request& request) {
    chat_vocab vocab;
    vocab.size = static_cast<size_t>(this->sampler->in_features);
    vocab.encode = [this](const std::string& text) {
        std::vector<int> ids = this->tokenizer->encode(text);
        if (this->has_bos_token && !ids.empty() && ids.front() == this->bos_token_id) {
            ids.erase(ids.begin());
        }
        return ids;
    };
    vocab.stop_tokens = this->eos_token_ids;
    vocab.tokenizer_json = (std::filesystem::path(this->model_path) / "tokenizer.json").string();

    auto engine = [this, &vocab]() -> chat_grammar_engine& {
        if (!this->grammar_engine_ || this->grammar_engine_path_ != vocab.tokenizer_json) {
            auto start = time_utils::now();
            this->grammar_engine_ = std::make_unique<chat_grammar_engine>(vocab, this->chat_format);
            this->grammar_engine_path_ = vocab.tokenizer_json;
            header_print("FLM", "Grammar engine ready in "
                         << time_utils::duration_ns(start, time_utils::now()).first / 1000000 << " ms");
        }
        return *this->grammar_engine_;
    };
    // thinking turned off on the model, not just in the request: the grammar must
    // not offer a reasoning block either, or a model kept from its first choice of
    // token (Gemma4 before a JSON answer) escapes into one
    chat_constraint_request effective = request;
    if (!this->thinking_enabled()) {
        effective.allow_reasoning = false;
    }
    // a model never shown the tools would make up the arguments of a call it is forced into
    if (!this->tools_enabled()) {
        if (request.tool_choice == chat_tool_choice::REQUIRED || request.tool_choice == chat_tool_choice::FUNCTION) {
            throw std::invalid_argument("this model does not support tool calling; tool_choice must be \"auto\" or \"none\"");
        }
        effective.tools = nullptr;
    }
    return chat_make_constraint(this->chat_format, effective, vocab, engine);
}

std::string AutoModel::_shared_generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
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
        // Sampled from the prefill, so no engine has run it.
        this->token_history.pop_back();
        meta_info.stop_reason = reason;
        return result;
    }
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    // Some backends refuse to decode past a limit of their own, below MAX_L.
    const uint32_t decode_cap = this->decode_cap();
    if (this->total_tokens >= decode_cap){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
    }
    else if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
        reason = MAX_LENGTH_REACHED;
    }
    else while (true){
        // Running out of window is a truncation, not the model choosing to
        // stop, and the caller has to be able to tell those apart: this is
        // what becomes `done_reason` over the API. Testing the cap here rather
        // than in the loop condition is what records it.
        if (this->total_tokens >= decode_cap){
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
            // The FastFlowLM engines want one more forward() to keep their KV
            // cache in step. Two things can say no: a frontend whose cache is
            // discarded anyway (forward_on_eos), and a backend that caps its
            // own decode and would reject the call.
            if (this->forward_on_eos &&
                (!this->backend_ || this->backend_->forwards_past_eos())) {
                this->lm_engine->forward(last_sampled_token);
            } else {
                // The engine never ran it, so the next prompt has to carry it.
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
    if (this->total_tokens >= decode_cap){
        header_print("WARNING", "Max length reached, stopping generation...");
    }
    if (this->log_raw_output) {
        std::cout << std::endl;
        header_print("FLM", "Model RAW Output: \n" + result);
    }
    return result;
}

/// \brief Clear the context
/// \note The function will clear the context
/// \note The function will reset the total tokens
/// \note The function will reset the last token
/// \note The function will clear the context
/// \note The function will reset the total tokens
void AutoModel::clear_context() {
    this->prefix_cache_.clear_live(true); // the saved snapshots belong to other conversations too: keep them
    this->_clear_live_context();
}

void AutoModel::_clear_live_context() {
    this->total_tokens = 0;
    this->last_token = -1;
    this->token_history.clear();
    this->checkpoint_his.clear();
    // A poisoned engine would throw; drop the conversation and leave it alone
    // so the caller can still reload.
    if (this->backend_ && this->backend_->poisoned()) {
        if (this->sampler) this->sampler->reset_penalties();
        return;
    }
    this->lm_engine->clear_context();
    this->total_tokens = 0;
    this->sampler->reset_penalties();
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->last_prefill_time = { 0, "us" };
}


/// \brief Get the current context length
/// \note The function will get the current context length
/// \note The function will return the current context length
int AutoModel::get_current_context_length() {
    return this->total_tokens;
}

/// \brief Set the sampler
/// \param sampler_config the sampler config
/// \note The function will set the sampler
/// \note The function will reset the sampler
/// \note The function will set the sampler config
void AutoModel::set_sampler(sampler_config& sampler_config) {
    if (this->sampler != nullptr) {
        this->sampler.reset();
    }
    this->sampler = std::make_unique<Sampler>(this->lm_config->get("vocab_size"), sampler_config);
}

/// \brief Set the max length
/// \param MAX_L the max length
/// \note The function will set the max length
/// \note The function will update the max length
void AutoModel::set_max_length(unsigned int MAX_L) {
    this->MAX_L = std::max(MAX_L, this->MAX_L);
    if (this->lm_engine != nullptr) {
        this->lm_engine->update_max_length(MAX_L);
    }
}

/// \brief Show the model info
/// \note The function will show the model info
/// \note The function will return the model info
std::string AutoModel::show_model_info() {
    try {
        std::string ss = this->lm_config->_str();
        return ss;
    }
    catch (const std::exception& e) {
        return "Error showing model info: " + std::string(e.what());
    }
}

/// \brief Show the profile
/// \note The function will show the profile
/// \note The function will return the profile
/// \note "Total time" is wall clock around insert() + generate(), while every
///       other row is a narrow window inside it, so the rows do not partition
///       the run. The "Untimed" row at the bottom is the remainder, and it is
///       printed precisely because that gap used to be invisible: Qwen3.8's
///       think preamble spent ~13 s of a 24 s run in four forward() calls
///       that _shared_generate()'s DECODING_TIME.reset() then discarded, and
///       nothing in this block said so.
/// \note The four narrow rows are only comparable with Total on a single-turn
///       run. DECODING_TIME is reset at the top of every _shared_generate()
///       while PREFILL_TIME and TOTAL_TIME accumulate across turns, so a
///       multi-turn session over-reports "Untimed" by the decode time of
///       every turn but the last.
std::string AutoModel::show_profile() {
    std::stringstream ss;
    int total_tokens = this->lm_engine->get_current_context_length();
    time_utils::time_with_unit time = this->profiler_list[TOTAL_TIME].get_total_time();
    ss << "  Statistics:" << std::endl;
    ss << "    Total tokens:        " << this->get_current_context_length() << " (" << total_tokens << ")" << std::endl;
    ss << "    Total time:          " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[DECODING_TIME].get_total_time();
    ss << "    Decoding time:       " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[PREFILL_TIME].get_total_time();
    ss << "    Prefill time:        " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[SAMPLING_TIME].get_total_time();
    ss << "    Sampling time:       " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[TKOEN_ENCODE_TIME].get_total_time();
    ss << "    Token encoding time: " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[TKOEN_DECODE_TIME].get_total_time();
    ss << "    Token decoding time: " << time.first << " " << time.second << std::endl;
    // Same unit for all five before subtracting: get_total_time() re_unit()s
    // each one independently, so their .first fields are not commensurable.
    const float total_us = time_utils::cast_to_us(this->profiler_list[TOTAL_TIME].get_total_time()).first;
    if (total_us > 0.0f) {
        float timed_us = 0.0f;
        for (profiler_type p : {DECODING_TIME, PREFILL_TIME, SAMPLING_TIME,
                                TKOEN_ENCODE_TIME, TKOEN_DECODE_TIME})
            timed_us += time_utils::cast_to_us(this->profiler_list[p].get_total_time()).first;
        // re_unit() only scales upward, so a negative remainder would print as
        // a seven-digit microsecond count. Scale the magnitude and put the
        // sign back: negative is not an error to hide, it is the multi-turn
        // case above announcing itself.
        const float gap_us = total_us - timed_us;
        time = time_utils::re_unit(std::make_pair(std::abs(gap_us), "us"));
        ss << "    Untimed:             " << (gap_us < 0.0f ? -time.first : time.first)
           << " " << time.second << std::endl;
    }
    ss << "    Average decoding speed:       " << this->profiler_list[DECODING_TIME].get_average_speed() << " tokens/s" << std::endl;
    ss << "    Average prefill  speed:       " << this->profiler_list[PREFILL_TIME].get_average_speed() << " tokens/s" << std::endl;
    // ss << "    Average sampling speed:       " << this->profiler_list[SAMPLING_TIME].get_average_speed() << " tokens/s" << std::endl;
    // ss << "    Average token encoding speed: " << this->profiler_list[TKOEN_ENCODE_TIME].get_average_speed() << " tokens/s" << std::endl;
    // ss << "    Average token decoding speed: " << this->profiler_list[TKOEN_DECODE_TIME].get_average_speed() << " tokens/s" << std::endl;
    // ss << "    Average overall speed:        " << this->profiler_list[TOTAL_TIME].get_average_speed() << " tokens/s" << std::endl;

    if (this->backend_) {
        ss << "    Backend:             " << this->backend_->id() << std::endl;
        const std::string detail = this->backend_->detail();
        if (!detail.empty()) {
            ss << "    Backend detail:      " << detail << std::endl;
        }
    }

    return ss.str();
}

/// \brief Get the history
/// \note The function will get the history
/// \note The function will return the history
std::pair<std::string, std::vector<int>> AutoModel::get_history() {
    std::vector<int> history = this->token_history;
    std::string all_context = this->tokenizer->decode(history);
    return std::make_pair(all_context, history);
}

/// \brief Verbose
/// \note The function will verbose
/// \note The function will print the verbose
void AutoModel::verbose() {
    std::cout << std::endl;
    int total_tokens = this->get_current_context_length();
    float prefill_speed = this->profiler_list[PREFILL_TIME].get_average_speed();
    float decoding_speed = this->profiler_list[DECODING_TIME].get_average_speed();
    time_utils::time_with_unit ttft_time = this->profiler_list[TTFT_TIME].get_total_time();
    float context_percentage = (float)total_tokens / (float)this->MAX_L * 100;
    std::cout << "Verbose: " << std::endl;
    std::cout << "  Total tokens:        " << total_tokens << " (" << std::fixed << std::setprecision(2) << context_percentage << "%)" << std::endl;
    std::cout << "  TTFT:                " << ttft_time.first << " " << ttft_time.second << std::endl;
    std::cout << "  Prefill speed:       " << std::fixed << std::setprecision(2) << prefill_speed << " tokens/s" << std::endl;
    std::cout << "  Decoding speed:      " << std::fixed << std::setprecision(2) << decoding_speed << " tokens/s" << std::endl << std::endl;
}

/// \brief Set the top-k
/// \param topk the top-k
/// \note The function will set the top-k
/// \note The function will check if the top-k is valid
void AutoModel::set_topk(int topk) {
    if (topk < 1) {
        header_print("WARNING", "Top-k must be greater than 0");
        return;
    }
    if (topk > this->lm_config->get("vocab_size")) {
        header_print("WARNING", "Top-k is greater than vocab size, set to vocab size: " << this->lm_config->get("vocab_size"));
        topk = this->lm_config->get("vocab_size");
    }
    
    this->sampler->top_k = topk;
}

/// \brief Set the top-p
/// \param topp the top-p
/// \note The function will set the top-p
/// \note The function will check if the top-p is valid
void AutoModel::set_topp(float topp) {
    if (topp < 0.0f || topp > 1.0f) {
        header_print("WARNING", "Top-p must be between 0.0 and 1.0");
        return;
    }
    this->sampler->top_p = topp;
}

/// \brief Set the min-p
/// \param topp the min-p
/// \note The function will set the min-p
/// \note The function will check if the min-p is valid
void AutoModel::set_minp(float minp) {
    if (minp < 0.0f || minp > 1.0f) {
        header_print("WARNING", "Min-p must be between 0.0 and 1.0");
        return;
    }
    this->sampler->min_p = minp;
}

/// \brief Set the temperature
/// \param temperature the temperature
/// \note The function will set the temperature
/// \note The function will check if the temperature is valid
void AutoModel::set_temperature(float temperature) {
    if (temperature < 0.0f) {
        header_print("WARNING", "Temperature must be greater than 0.0");
        return;
    }
    this->sampler->temperature = temperature;
}

/// \brief Set the presence penalty
/// \param presence_penalty the presence penalty
/// \note The function will set the presence penalty
/// \note The function will check if the presence penalty is valid
void AutoModel::set_presence_penalty(float presence_penalty) {
    if (presence_penalty < 0.0f) {
        header_print("WARNING", "Presence penalty must be greater than 0.0");
        return;
    }
    this->sampler->pre_penalty = presence_penalty;
}

/// \brief Set the repetition penalty
/// \param repetition_penalty the repetition penalty
/// \note The function will set the repetition penalty
/// \note The function will check if the repetition penalty is valid
void AutoModel::set_repetition_penalty(float repetition_penalty) {
    if (repetition_penalty < 0.0f) {
        header_print("WARNING", "Repetition penalty must be greater than 0.0");
        return;
    }
    if (repetition_penalty < 1.0f) {
        header_print("WARNING", "If Repetition Penalty < 1.0, it will reward previous generated tokens, which may cause infinite loop!");
    }
    this->sampler->rep_penalty = repetition_penalty;
}

/// \brief Set the frequency penalty
/// \param frequency_
void AutoModel::set_frequency_penalty(float frequency_penalty) {
    if (frequency_penalty < 0.0f) {
        header_print("WARNING", "Frequency penalty must be greater than 0.0");
        return;
    }
    this->sampler->freq_penalty = frequency_penalty;
}

/// \brief Set the frequency penalty window
/// \param frequency_penalty_window the frequency penalty window
/// \note The function will set the frequency penalty window
/// \note The function will check if the frequency penalty window is valid
void AutoModel::set_frequency_penalty_window(int frequency_penalty_window) {
    this->sampler->freq_penalty_window = frequency_penalty_window;
}

/// \brief Set the penalty window
/// \param penalty_window the frequency penalty window
/// \note The function will set the penalty window
/// \note The function will check if the penalty window is valid
void AutoModel::set_penalty_window(int penalty_window) {
    if (penalty_window < -1) {
        header_print("WARNING", "Penalty window must be -1 (the whole reply), 0 (off) or more");
        return;
    }
    this->sampler->set_repeat_last_n(penalty_window == -1
        ? std::numeric_limits<size_t>::max()
        : static_cast<size_t>(penalty_window));
}

/// \brief Restart sampling from a seed
/// \param seed the seed
/// \note The penalty window is emptied too: it can still hold the previous
///       reply, and then the same request and seed would sample differently.
void AutoModel::set_seed(uint64_t seed) {
    this->sampler->set_seed(seed);
    this->sampler->reset_penalties();
}

/// \brief Start the ttft timer
/// \note The function will start the ttft timer
/// \note The function will reset the ttft timer
void AutoModel::start_ttft_timer() {
    this->profiler_list[TTFT_TIME].reset();
    this->profiler_list[TTFT_TIME].start();
}

/// \brief Stop the ttft timer
/// \note The function will stop the ttft timer
/// \note The function will check if the ttft timer is valid
void AutoModel::stop_ttft_timer() {
    this->profiler_list[TTFT_TIME].stop(1);
}

/// \brief Reset the total timer
/// \note The function will reset the total timer
/// \note The function will check if the total timer is valid
void AutoModel::reset_total_timer() {
    this->profiler_list[TOTAL_TIME].reset();
}

/// \brief Start the total timer
/// \note The function will start the total timer
/// \note The function will check if the total timer is valid
void AutoModel::start_total_timer() {
    this->profiler_list[TOTAL_TIME].start();
}

/// \brief Stop the total timer
/// \note The function will stop the total timer
/// \note The function will check if the total timer is valid
void AutoModel::stop_total_timer() {
    this->profiler_list[TOTAL_TIME].stop(this->total_tokens, true);
}
