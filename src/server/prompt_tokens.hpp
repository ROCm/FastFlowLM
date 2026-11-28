#pragma once

#include "error_model.hpp"
#include "tokenizer/tokenizer.hpp"

#include <limits>
#include <string_view>

namespace flm {

inline nlohmann::ordered_json prompt_tokens(std::string_view path, const nlohmann::ordered_json& req,
                                             Tokenizer& tokenizer) {
    using json = nlohmann::ordered_json;
    constexpr size_t max_ids = 65536;
    for (const auto& entry : req.items()) {
        const auto& key = entry.key();
        if (key == "model") continue;
        if (path == "/detokenize" ? key == "tokens" :
            key == "content" || key == "with_pieces" || key == "add_special" || key == "parse_special") continue;
        return error_model::make_error(400, "Unsupported tokenizer option", key);
    }
    if (path == "/detokenize") {
        auto it = req.find("tokens");
        if (it == req.end() || !it->is_array() || it->size() > max_ids)
            return error_model::make_error(400, "Expected an array of at most 65536 token IDs", "tokens");
        const auto vocab = tokenizer.vocab_size();
        std::vector<int> ids;
        ids.reserve(it->size());
        for (const auto& token : *it) {
            if (!token.is_number_integer() || token < 0 || token >= vocab ||
                token > std::numeric_limits<int>::max())
                return error_model::make_error(400, "Token ID is outside the vocabulary", "tokens");
            ids.push_back(token.get<int>());
        }
        return json{{"content", ids.empty() ? std::string() : tokenizer.decode(ids)}};
    }
    if (!req.contains("content") || !req.at("content").is_string())
        return error_model::make_error(400, "Expected text in content", "content");
    for (const char* key : {"with_pieces", "add_special", "parse_special"}) {
        if (req.contains(key) && !req.at(key).is_null() && !req.at(key).is_boolean())
            return error_model::make_error(400, "Expected a boolean", key);
    }
    const auto flag = [&](const char* key, bool fallback) {
        const auto it = req.find(key);
        return it == req.end() || it->is_null() ? fallback : it->get<bool>();
    };
    if (flag("add_special", false))
        return error_model::make_error(400, "Adding special tokens is not supported", "add_special");
    if (!flag("parse_special", true))
        return error_model::make_error(400, "Disabling special-token parsing is not supported", "parse_special");
    const auto& text = req.at("content").get_ref<const std::string&>();
    if (text.size() > 1024 * 1024)
        return error_model::make_error(400, "Text exceeds the 1 MiB diagnostic limit", "content");
    const auto ids = tokenizer.encode(text);
    if (ids.size() > max_ids)
        return error_model::make_error(400, "Text exceeds the 65536-token diagnostic limit", "content");
    if (!flag("with_pieces", false)) return json{{"tokens", ids}};
    auto pieces = json::array();
    for (auto id : ids) {
        std::string bytes;
        try {
            bytes = tokenizer.run_time_decoder(id);
        } catch (const std::runtime_error&) {
            bytes = tokenizer.decode({id});
        }
        json piece = bytes;
        try {
            (void)piece.dump(); // a byte-level token can be only part of a UTF-8 character
        } catch (const json::type_error&) {
            piece = std::vector<unsigned char>(bytes.begin(), bytes.end());
        }
        pieces.push_back({{"id", id}, {"piece", std::move(piece)}});
    }
    return json{{"tokens", std::move(pieces)}};
}

} // namespace flm
