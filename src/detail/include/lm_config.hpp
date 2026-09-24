/// \file lm_config.hpp
/// \brief lm_config class
/// \author FastFlowLM Team
/// \date 2025-08-05
/// \version 0.9.10
/// \note This class is used to store the model configuration.
#pragma once

#include "typedef.hpp"
#include "utils/utils.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>

/// \brief read one parameter out of a config json
/// \note Same semantics as JSON_GET (missing OR null falls back to the default),
///       in expression form. Plain nlohmann .value() throws on a null, and real
///       configs do carry nulls (e.g. "sliding_window": null on qwen3).
template <typename T>
inline T cfg_get(const nlohmann::json& jc, const char* key, T default_value){
    if (jc.contains(key) && !jc[key].is_null()){
        return T(jc[key]);
    }
    return default_value;
}

/// \brief read a nested object out of a config json
/// \note Reference-returning counterpart of cfg_get, for walking nested configs
///       (thinker_config.text_config, ...). Returns a shared empty object when
///       the key is missing or null, so the JSON_GET below it still defaults.
inline const nlohmann::json& cfg_sub(const nlohmann::json& jc, const char* key){
    static const nlohmann::json empty = nlohmann::json::object();
    if (jc.contains(key) && !jc[key].is_null()){
        return jc[key];
    }
    return empty;
}

/// \brief LM_Config class
/// \note Model parameters are NOT cached as members. Everything comes from the
///       model's config.json, so every consumer reads what it needs straight out
///       of _json_config with JSON_GET. from_pretrained() only locates the file
///       and normalizes it, so that all readers share one canonical key set.
/// \note This class is passed by value across the FLM_DLL boundary. Its layout
///       must stay identical to FastFlowLM/src/include/lm_config.hpp.
class LM_Config{
    public:
        std::string model_path;
        std::string model_name;
        std::string exec_path;
        std::string flm_version;

        nlohmann::json _json_config;

        /// \brief read one model parameter out of config.json
        /// \note Defaults to u32 because most parameters are dimensions:
        ///       config.get("head_dim"), config.get<f32>("rms_norm_eps", 0.0f),
        ///       config.get<std::string>("vision_model_weight", "").
        template <typename T = u32>
        T get(const char* key, T default_value = T(0)) const {
            return cfg_get<T>(this->_json_config, key, default_value);
        }

        /// \brief read a nested config object (vision_config, audio_config, ...)
        /// \return the sub-object, or an empty object when absent/null
        const nlohmann::json& sub(const char* key) const {
            return cfg_sub(this->_json_config, key);
        }

        /// \brief from pretrained
        /// \param model_name the model name
        void from_pretrained(std::string model_name);
        std::string _str();
        LM_Config(){}

    protected:
        void _resolve_paths(const std::string& model_name);
        void _load_json();
        void _normalize_multi_modal();
        std::string _str_from(const nlohmann::json& jc);
};

class Whisper_Config : public LM_Config{
public:
    /// \brief from pretrained
    /// \param model_name the model name
    void from_pretrained(std::string model_name);
    std::string _str();
    Whisper_Config(){}
};
