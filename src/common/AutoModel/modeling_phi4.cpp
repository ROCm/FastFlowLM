/// \file modeling_phi4.cpp
/// \brief Phi-4 frontend and backend routing
#include "AutoModel/modeling_phi4.hpp"
#include "utils/file_access.hpp"

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include "models/phi4/phi4_corelib_aie4.hpp"
#include "models/phi4/phi4_corelib_gguf.hpp"
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
constexpr std::string_view kAie4Backend = "corelib_aie4_gguf";
constexpr std::string_view kAie4Gguf = "Phi-4-mini-instruct.Q8_0.gguf";
constexpr int kAie4DecodeLimit = 4095;

enum class Phi4Backend { LegacyNpu2, CorelibAie4Gguf };

Phi4Backend ResolveBackend(const json& model_info) {
    const auto details = model_info.find("details");
    if (details == model_info.end() || !details->is_object() ||
        !details->contains("execution_backend")) {
        return Phi4Backend::LegacyNpu2;
    }
    const auto& backend = details->at("execution_backend");
    if (!backend.is_string()) {
        throw std::invalid_argument(
            "Phi-4 details.execution_backend must be a string");
    }
    const std::string value = backend.get<std::string>();
    if (value == kAie4Backend) return Phi4Backend::CorelibAie4Gguf;
    throw std::invalid_argument("Unknown Phi-4 execution backend '" + value + "'");
}

std::uint32_t ResolveContext(const json& model_info, int requested) {
    const std::int64_t value = requested == -1
        ? model_info.at("default_context_length").get<std::int64_t>()
        : requested;
    if (value < 1 || value > 4096) {
        throw std::out_of_range("Phi-4 AIE4 context length must be in 1..4096");
    }
    return static_cast<std::uint32_t>(value);
}

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

#if defined(FLM_CORELIB_TESTING)
Phi4::EngineFactoryForTesting Phi4::engine_factory_for_testing_;
std::function<bool(const causal_lm*)> Phi4::engine_poisoned_for_testing_;
#endif

Phi4::Phi4(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Phi4") {}

void Phi4::load_model(std::string model_path, json model_info,
                      int default_context_length, bool enable_preemption) {
    const Phi4Backend backend = ResolveBackend(model_info);
    if (backend == Phi4Backend::LegacyNpu2) {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        const bool switching_from_aie4 = uses_corelib_aie4_;
        uses_corelib_aie4_ = false;
        aie4_poisoned_ = false;
        corelib_runtime_.reset();
        aie4_corelib_path_.clear();
        if (switching_from_aie4) is_model_loaded = false;
#endif
        _shared_load_model(model_path, model_info, default_context_length, enable_preemption);
        std::unique_ptr<causal_lm> engine;
#if defined(FLM_CORELIB_TESTING)
        if (!engine_factory_for_testing_) throw std::logic_error("test engine factory is not installed");
        engine = engine_factory_for_testing_(false, *lm_config, npu.get(), model_path, MAX_L);
#else
        q4nx = std::make_unique<Q4NX>(this->model_path);
        engine = std::make_unique<phi4_npu>(*lm_config, npu.get(), MAX_L);
        engine->load_weights(*q4nx);
        q4nx.reset();
#endif
        engine->clear_context();
        setup_tokenizer(model_path);
        lm_engine = std::move(engine);
        sampler.reset();
        ConfigureSampler(*this);
    } else {
#if !defined(FLM_ENABLE_CORELIB_AIE4)
        throw std::runtime_error(
            "This binary was built without Phi-4 AIE4 corelib support");
#else
        if (enable_preemption) {
            throw std::invalid_argument("Phi-4 AIE4 does not support preemption");
        }
        const std::uint32_t context_length = ResolveContext(model_info, default_context_length);
        const std::filesystem::path root(model_path);

        // Read and validate every source of truth before runtime acquisition or
        // engine/device creation. There is deliberately no alternate filename.
        const auto config = ReadJson(root / "config.json");
        const auto tokenizer_json = ReadJson(root / "tokenizer.json");
        const auto tokenizer_config = ReadJson(root / "tokenizer_config.json");
        auto package = flm::phi4::Phi4GgufPackage::Open(root / kAie4Gguf);
        package->ValidatePhi4Contract(config, tokenizer_json, tokenizer_config);

        uses_corelib_aie4_ = false;
        aie4_poisoned_ = false;
        try {
            _shared_initialize_model_state(model_path, model_info,
                                           static_cast<int>(context_length));
            npu.reset();
            this->enable_preemption = false;
            setup_tokenizer(model_path, &tokenizer_config);
            sampler.reset();
            ConfigureSampler(*this);

            const auto corelib_path = std::filesystem::absolute(
                flm::corelib::CorelibApi::ResolveLibraryPath(
                    utils::get_executable_directory())).lexically_normal();
            std::unique_ptr<causal_lm> engine;
#if defined(FLM_CORELIB_TESTING)
            if (!engine_factory_for_testing_) throw std::logic_error("test engine factory is not installed");
            engine = engine_factory_for_testing_(true, *lm_config, nullptr,
                                                  root, context_length);
#else
            auto runtime = flm::corelib::CorelibRuntime::GetOrCreate(
                utils::get_executable_directory());
            engine = std::make_unique<flm::phi4::phi4_corelib_aie4>(
                *lm_config, package, runtime, context_length);
            corelib_runtime_ = std::move(runtime);
#endif
            engine->clear_context();
            aie4_corelib_path_ = corelib_path;
            lm_engine = std::move(engine);
            uses_corelib_aie4_ = true;
        } catch (...) {
            lm_engine.reset();
            corelib_runtime_.reset();
            aie4_corelib_path_.clear();
            tokenizer.reset();
            sampler.reset();
            lm_config.reset();
            is_model_loaded = false;
            uses_corelib_aie4_ = false;
            throw;
        }
#endif
    }

    for (auto& item : profiler_list) item.reset();
}

void Phi4::setup_tokenizer(const std::string& model_path,
                           const nlohmann::json* verified_tokenizer_config) {
    nlohmann::json config = verified_tokenizer_config
        ? *verified_tokenizer_config
        : ReadJson(std::filesystem::path(model_path) / "tokenizer_config.json");
    if (!config.contains("chat_template") || !config["chat_template"].is_string())
        throw std::invalid_argument("Phi-4 tokenizer_config.json requires a string chat_template");

    const bool aie4 = verified_tokenizer_config != nullptr;
    // Preserve the legacy Phi-4 contract: minja receives no textual BOS/EOS.
    // AIE4 also disables automatic BOS, with stop IDs supplied only after the
    // cross-source package contract has been validated.
    auto chat = std::make_unique<minja::chat_template>(
        config["chat_template"].get<std::string>(), "", "");
    std::vector<int> eos;
    if (aie4) {
        // ValidatePhi4Contract proved these exact independent sources.
        eos = {200020, 199999};
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

#if defined(FLM_ENABLE_CORELIB_AIE4)
bool Phi4::engine_is_poisoned() const noexcept {
#if defined(FLM_CORELIB_TESTING)
    return engine_poisoned_for_testing_ && lm_engine
        ? engine_poisoned_for_testing_(lm_engine.get())
        : false;
#else
    const auto* engine = dynamic_cast<const flm::phi4::phi4_corelib_aie4*>(lm_engine.get());
    return engine && engine->poisoned();
#endif
}

void Phi4::validate_aie4_capacity(std::size_t rendered_tokens,
                                  std::optional<int> requested) const {
    const std::size_t cap = std::min<std::size_t>(MAX_L, kAie4DecodeLimit);
    const auto normalized = normalize_requested_max_new_tokens(requested);
    if (rendered_tokens >= cap ||
        (normalized && static_cast<std::size_t>(*normalized) > cap - rendered_tokens)) {
        std::ostringstream message;
        message << "Phi-4 AIE4 request exceeds the 4095-token decode limit: rendered prompt has "
                << rendered_tokens << " tokens";
        if (normalized) message << " and requested output has " << *normalized << " tokens";
        throw ModelRequestError(400, false, message.str());
    }
}

void Phi4::clear_after_inference_failure(bool poisoned) {
    aie4_poisoned_ = poisoned;
    total_tokens = 0;
    last_token = -1;
    token_history.clear();
    checkpoint_his.clear();
    if (!poisoned && lm_engine) {
        try { lm_engine->clear_context(); } catch (...) {}
    }
    if (sampler) sampler->reset_penalties();
}

std::string Phi4::generate_aie4(chat_meta_info_t& meta_info,
                                std::ostream& os,
                                std::function<bool()> is_cancelled) {
    std::string result;
    meta_info.stop_reason = EOT_DETECTED;
    int generated = 0;
    while (last_token != -1 && generated < aie4_generation_budget_) {
        if (is_cancelled()) {
            meta_info.stop_reason = CANCEL_DETECTED;
            break;
        }
        const int token = last_token;
        token_history.push_back(token);
        ++total_tokens;
        ++generated;
        ++meta_info.generated_tokens;
        if (is_normal_token(token)) {
            const std::string text = tokenizer->run_time_decoder(token);
            result += text;
            os << text << std::flush;
        }
        if (is_eos(token)) {
            last_token = -1;
            break;
        }
        if (generated >= aie4_generation_budget_ || total_tokens >= std::min<std::uint32_t>(MAX_L, kAie4DecodeLimit)) {
            last_token = -1;
            meta_info.stop_reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            meta_info.stop_reason = CANCEL_DETECTED;
            break;
        }
        auto logits = lm_engine->forward(token);
        last_token = sampler->sample(logits);
    }
    return result;
}
#endif

std::string Phi4::show_profile() {
    std::string profile = AutoModel::show_profile();
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (uses_corelib_aie4_) {
        profile += "    Backend:           corelib_aie4_gguf\n";
        profile += "    Corelib DLL:       " + aie4_corelib_path_.string() + "\n";
    }
#endif
    return profile;
}

void Phi4::clear_context() {
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (uses_corelib_aie4_ && aie4_poisoned_) {
        total_tokens = 0;
        last_token = -1;
        token_history.clear();
        checkpoint_his.clear();
        if (sampler) sampler->reset_penalties();
        return;
    }
#endif
    AutoModel::clear_context();
}

bool Phi4::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                  std::function<bool()> is_cancelled) {
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (uses_corelib_aie4_ && aie4_poisoned_) {
        throw ModelRequestError(500, true,
            "Phi-4 AIE4 model is poisoned; unload/reload is required");
    }
#endif
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

#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (uses_corelib_aie4_) {
        validate_aie4_capacity(tokens.size(), input.requested_max_new_tokens);
        const auto normalized = normalize_requested_max_new_tokens(input.requested_max_new_tokens);
        const int remaining = static_cast<int>(
            std::min<std::uint32_t>(MAX_L, kAie4DecodeLimit) - tokens.size());
        aie4_generation_budget_ = normalized ? *normalized : remaining;
        if (is_cancelled()) {
            meta_info.stop_reason = CANCEL_DETECTED;
            return false;
        }
        try {
            return _shared_insert(meta_info, tokens, std::move(is_cancelled));
        } catch (const ModelRequestError&) {
            throw;
        } catch (...) {
            const bool poisoned = engine_is_poisoned();
            clear_after_inference_failure(poisoned);
            throw ModelRequestError(500, true, poisoned
                ? "AIE4 inference failed; unload/reload is required because the model is poisoned"
                : "AIE4 inference failed; the current conversation was cleared");
        }
    }
#endif
    return _shared_insert(meta_info, tokens, std::move(is_cancelled));
}

std::string Phi4::generate(chat_meta_info_t& meta_info, int length_limit,
                           std::ostream& os,
                           std::function<bool()> is_cancelled) {
#if defined(FLM_ENABLE_CORELIB_AIE4)
    if (uses_corelib_aie4_) {
        if (aie4_poisoned_) throw ModelRequestError(500, true,
            "Phi-4 AIE4 model is poisoned; unload/reload is required");
        try {
            return generate_aie4(meta_info, os, std::move(is_cancelled));
        } catch (const ModelRequestError&) {
            throw;
        } catch (...) {
            const bool poisoned = engine_is_poisoned();
            clear_after_inference_failure(poisoned);
            throw ModelRequestError(500, true, poisoned
                ? "AIE4 inference failed; unload/reload is required because the model is poisoned"
                : "AIE4 inference failed; the current conversation was cleared");
        }
    }
#endif
    return _shared_generate(meta_info, length_limit, os, std::move(is_cancelled));
}

std::string Phi4::generate_with_prompt(chat_meta_info_t& meta_info,
                                       lm_uniform_input_t& input,
                                       int length_limit,
                                       std::ostream& os) {
    if (!insert(meta_info, input)) return {};
    return generate(meta_info, length_limit, os);
}
