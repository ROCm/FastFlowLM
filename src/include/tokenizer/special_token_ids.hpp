/// \file special_token_ids.hpp
/// \brief check the special-token ids tokenizer_config.json states against the tokenizer
/// \note A package's tokenizer_config.json can state a bos_token_id that is not the id of its bos_token: the
///       gpt-oss packages name <|startoftext|> (199998 in o200k) but state 2, which is "#". Every token equal to
///       the BOS id is dropped from the output (AutoModel::is_normal_token), so such a package lost every lone "#".
///       The tokenizer is the authority on which id a token text has.
#pragma once
#include <functional>
#include <string>

#include "nlohmann/json.hpp"

namespace special_token_ids {

/// \brief the text of a special-token entry of tokenizer_config.json: a plain string, or an AddedToken object's "content"
/// \return the text, or "" when the entry is neither
inline std::string text(const nlohmann::json& entry) {
    if (entry.is_string()) return entry.get<std::string>();
    if (entry.is_object()) {
        const auto it = entry.find("content");
        if (it != entry.end() && it->is_string()) return it->get<std::string>();
    }
    return "";
}

/// \brief the BOS id to use: the tokenizer's id of `bos_token` when it has one, else `configured_id`
/// \param bos_token the "bos_token" entry of tokenizer_config.json
/// \param configured_id the "bos_token_id" of tokenizer_config.json
/// \param token_to_id the tokenizer's lookup of a token text (vocabulary and added tokens); a negative result = unknown
/// \param warning set to a message when the configured id is replaced, cleared otherwise (may be null)
inline int checked_bos_id(const nlohmann::json& bos_token, int configured_id,
                          const std::function<int(const std::string&)>& token_to_id, std::string* warning) {
    if (warning) warning->clear();
    const std::string bos = text(bos_token);
    if (bos.empty() || !token_to_id) return configured_id;
    const int id = token_to_id(bos);
    if (id < 0 || id == configured_id) return configured_id;
    if (warning) {
        *warning = "tokenizer_config.json states bos_token_id " + std::to_string(configured_id) + ", but bos_token " + bos +
                   " is token " + std::to_string(id) + " in tokenizer.json; using " + std::to_string(id);
    }
    return id;
}

}  // namespace special_token_ids
