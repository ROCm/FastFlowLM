/// \file modeling_phi4.cpp
/// \brief Phi-4 frontend
/// \note Every rule that used to be branched on here -- the rai decode cap,
///       poisoning, the no-preemption rule, the package-verified stop ids, the
///       separate decode loop -- now lives behind flm::backend::ModelBackend.
#include "AutoModel/modeling_phi4.hpp"
#include "utils/file_access.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace {

/// \brief read a JSON file, recording the open for the file-access audit
/// \param path the file to read
/// \return the parsed document
nlohmann::json ReadJson(const std::filesystem::path& path) {
    flm::file_access::ObserveOpen(path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.string());
    try {
        return nlohmann::json::parse(input);
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot parse " + path.string() + ": " + error.what());
    }
}

void ConfigureSampler(Phi4& model) {
    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;
    model.set_sampler(config);
}
} // namespace

Phi4::Phi4(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Phi4") {}

void Phi4::load_model(std::string model_path, json model_info,
                      int default_context_length, bool enable_preemption,
                      const std::string& backend) {
    // Read once here and hand the parse to both consumers. The backend needs it
    // to cross-validate its package and the tokenizer setup below needs it for
    // the chat template, but the model directory layout is this frontend's
    // knowledge -- a backend that opens the directory itself duplicates both
    // the read and that knowledge.
    const nlohmann::json tokenizer_config =
        ReadJson(std::filesystem::path(model_path) / "tokenizer_config.json");
    this->_shared_load_backend(model_path, model_info, default_context_length,
                               enable_preemption, backend, &tokenizer_config);
    this->setup_tokenizer(tokenizer_config);
    this->sampler.reset();
    ConfigureSampler(*this);
    for (auto& item : profiler_list) item.reset();
}

void Phi4::setup_tokenizer(const nlohmann::json& config) {
    if (!config.contains("chat_template") || !config["chat_template"].is_string())
        throw std::invalid_argument("Phi-4 tokenizer_config.json requires a string chat_template");

    // Preserve the legacy Phi-4 contract: minja receives no textual BOS/EOS,
    // and there is no automatic BOS token.
    auto chat = std::make_unique<minja::chat_template>(
        config["chat_template"].get<std::string>(), "", "");

    std::vector<int> eos;
    // A backend that cross-validated its own package proved its stop ids from
    // independent sources, so those win over tokenizer_config.json.
    const auto forced = this->backend_ ? this->backend_->forced_eos_ids()
                                       : std::nullopt;
    if (forced) {
        eos = *forced;
    } else {
        if (!config.contains("eos_token_id"))
            throw std::invalid_argument("Phi-4 tokenizer_config.json requires eos_token_id");
        const auto& ids = config["eos_token_id"];
        if (ids.is_number_integer()) eos.push_back(ids.get<int>());
        else if (ids.is_array()) for (const auto& id : ids) eos.push_back(id.get<int>());
        else throw std::invalid_argument("Phi-4 tokenizer_config.json eos_token_id must be integer or array");
    }

    has_bos_token = false;
    bos_token_id = -1;
    eos_token.clear();
    eos_token_ids = std::move(eos);
    chat_tmpl = std::move(chat);
    user_system_prompt.clear();
    extra_context["user_system_prompt"] = user_system_prompt;
}

std::string Phi4::apply_chat_template(nlohmann::ordered_json& messages,
                                      nlohmann::ordered_json) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = extra_context;
    return chat_tmpl->apply(inputs);
}

void Phi4::fail_inference() {
    const bool poisoned = backend_ && backend_->poisoned();
    _shared_after_inference_failure(poisoned);
    throw ModelRequestError(500, true, poisoned
        ? "Inference failed; unload/reload is required because the model is poisoned"
        : "Inference failed; the current conversation was cleared");
}

bool Phi4::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                  std::function<bool()> is_cancelled) {
    _shared_guard_poisoned();

    profiler_list[TKOEN_ENCODE_TIME].start();
    std::string rendered;
    if (input.messages.empty() && input.prompt.empty()) return false;
    if (!input.messages.empty()) rendered = apply_chat_template(input.messages);
    else {
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        messages.push_back({{"role", "user"}, {"content", input.prompt}});
        rendered = apply_chat_template(messages);
    }
    std::vector<int> tokens = tokenizer->encode(rendered);
    profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    if (is_cancelled()) {
        meta_info.stop_reason = CANCEL_DETECTED;
        return false;
    }
    try {
        return _shared_insert(meta_info, tokens, std::move(is_cancelled), nullptr,
                              0, input.requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        throw;
    } catch (...) {
        fail_inference();
    }
}

std::string Phi4::generate(chat_meta_info_t& meta_info, int length_limit,
                           std::ostream& os,
                           std::function<bool()> is_cancelled) {
    _shared_guard_poisoned();
    try {
        return _shared_generate(meta_info, length_limit, os, std::move(is_cancelled));
    } catch (const ModelRequestError&) {
        throw;
    } catch (...) {
        fail_inference();
    }
}

std::string Phi4::generate_with_prompt(chat_meta_info_t& meta_info,
                                       lm_uniform_input_t& input,
                                       int length_limit,
                                       std::ostream& os) {
    if (!insert(meta_info, input)) return {};
    return generate(meta_info, length_limit, os);
}
