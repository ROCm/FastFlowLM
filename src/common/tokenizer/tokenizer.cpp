/// \file tokenizer.cpp
/// \brief Tokenizer implementation for text encoding/decoding
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
#include "tokenizer/tokenizer.hpp"
#include "utils/file_access.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <regex>

/// \brief Constructor
/// \param model_path the model path
Tokenizer::Tokenizer(const std::string& model_path) {
    flm::file_access::ObserveOpen(
        std::filesystem::path(model_path) / "tokenizer.json");
    #ifdef _WIN32
    std::ifstream fs(model_path + "\\tokenizer.json", std::ios::in | std::ios::binary);
    #else
    std::ifstream fs(model_path + "/tokenizer.json", std::ios::in | std::ios::binary);
    #endif
    if (fs.fail()) {
        #ifdef _WIN32
        std::cerr << "Cannot open " << model_path + "\\tokenizer.json" << std::endl;
        #else
        std::cerr << "Cannot open " << model_path + "/tokenizer.json" << std::endl;
        #endif
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    this->tokenizer = tokenizers::Tokenizer::FromBlobJSON(data);
    fs.close();
    std::string decoder_type;
    nlohmann::json data_json = nlohmann::json::parse(data);
    JSON_GET(decoder_type, data_json["decoder"], "type", "ByteLevel", std::string);
    if (data_json.contains("added_tokens") && data_json["added_tokens"].is_array()) {
        for (const auto& t : data_json["added_tokens"]) {
            if (t.contains("id") && t.contains("content")) {
                added_len_[t["id"].get<int>()] = t["content"].get<std::string>().size();
            }
        }
    }
    if (decoder_type == "ByteLevel") {
        this->is_doubled_encoded = true;
    }
    else {
        this->is_doubled_encoded = false;
    }
}

/// \brief Destructor
Tokenizer::~Tokenizer() = default;

/// \brief Make the inverse byte map
/// \return the inverse byte map
std::unordered_map<uint32_t, uint8_t> Tokenizer::make_inverse_byte_map() {
    // 1) Build the "printable" byte list bs exactly as in GPT-2/LLaMA
    std::vector<uint8_t> bs;
    for (int b = 33; b <= 126; b++) bs.push_back((uint8_t)b);
    for (int b = 161; b <= 172; b++) bs.push_back((uint8_t)b);
    for (int b = 174; b <= 255; b++) bs.push_back((uint8_t)b);

    // 2) Build the codepoint list cs (same length, but extended to 256)
    std::vector<uint32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), (uint8_t)b) == bs.end()) {
            bs.push_back((uint8_t)b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // 3) Zip them into an inverse map: cp → original byte
    std::unordered_map<uint32_t, uint8_t> inv;
    for (size_t i = 0; i < bs.size(); i++) {
        inv[ cs[i] ] = bs[i];
    }
    return inv;
}

/// \brief Convert the codepoint to utf8
/// \param input the input
/// \return the utf8 string
std::string Tokenizer::cpt_to_utf8(const std::string& input) {
    static const std::string pattern = "▁";
    static const size_t pattern_size = pattern.size();
    static auto inv_map = this->make_inverse_byte_map();
    std::string output = "";
    output.reserve(input.size());
    size_t i = 0;
    if (!this->is_doubled_encoded) { // simply do pattern substitution of "▁" to " ", temporary solution
        std::string output = "";
        for (size_t i = 0; i < input.size(); i++) {
            if (i <= input.size() - pattern_size && input.substr(i, pattern_size) == pattern) {
                output += " ";
                i += pattern_size - 1; // -1 because the loop will increment i by 1
            }
            else {
                output += input[i];
            }
        }
        return output;
    }
    while (i < input.size()) {
        unsigned char c = input[i];
        uint32_t cp;
        size_t width;

        // --- Decode a UTF-8 codepoint ---
        if (c < 0x80) {
            cp = c;
            width = 1;
        }
        else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= input.size()) throw std::runtime_error("Truncated UTF-8, ll = 2");
            cp  = (uint32_t)(c     & 0x1F) << 6;
            cp |= (uint32_t)(input[i+1] & 0x3F);
            width = 2;
        }
        else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= input.size()) throw std::runtime_error("Truncated UTF-8, ll = 3");
            cp  = (uint32_t)(c     & 0x0F) << 12;
            cp |= (uint32_t)(input[i+1] & 0x3F) << 6;
            cp |= (uint32_t)(input[i+2] & 0x3F);
            width = 3;
        }
        else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= input.size()) throw std::runtime_error("Truncated UTF-8, ll = 4");
            cp  = (uint32_t)(c     & 0x07) << 18;
            cp |= (uint32_t)(input[i+1] & 0x3F) << 12;
            cp |= (uint32_t)(input[i+2] & 0x3F) << 6;
            cp |= (uint32_t)(input[i+3] & 0x3F);
            width = 4;
        }
        else {
            std::cout << std::hex <<"cp: " << cp << std::endl;
            throw std::runtime_error("Invalid UTF-8 byte");
        }

        // --- Look up the original byte ---
        auto it = inv_map.find(cp);
        if (it == inv_map.end()) {
            throw std::runtime_error("Codepoint not in LLaMA byte map");
        }
        output.push_back((char)it->second);

        i += width;
    }

    return output;
}

/// \brief Encode the text
/// \param text the text
/// \return the encoded tokens
std::vector<int> Tokenizer::encode(const std::string& text) {
    return this->tokenizer->Encode(text);
}

/// \brief Decode the tokens
/// \param tokens the tokens
/// \return the decoded text
std::string Tokenizer::decode(const std::vector<int>& tokens) {
    return this->tokenizer->Decode(tokens);
}

/// \brief Run time decoder
/// \param answer_token the answer token
/// \return the decoded text
std::string Tokenizer::run_time_decoder(int answer_token) {
    return this->cpt_to_utf8(this->tokenizer->IdToToken(answer_token));
}

/// \brief Number of UTF-8 bytes of the text one token stands for
size_t Tokenizer::token_byte_length(int id) {
    std::call_once(byte_len_once_, [this]() {
        if (!this->is_doubled_encoded) throw std::runtime_error("token_byte_length needs a byte-level vocabulary");
        const auto inv = this->make_inverse_byte_map();
        size_t n = this->tokenizer->GetVocabSize();
        for (const auto& a : added_len_) n = std::max<size_t>(n, static_cast<size_t>(a.first) + 1);
        std::vector<uint8_t> lens(n, 0);
        for (size_t i = 0; i < n; i++) {
            auto added = added_len_.find(static_cast<int>(i));
            if (added != added_len_.end()) {
                lens[i] = static_cast<uint8_t>(std::min<size_t>(added->second, 255));
                continue;
            }
            // byte-level text: one byte per codepoint of the token string (GPT-2 alphabet); a codepoint
            // outside the alphabet cannot occur in a regular token, count its UTF-8 bytes
            const std::string tok = this->tokenizer->IdToToken(static_cast<int32_t>(i));
            size_t bytes = 0;
            for (size_t k = 0; k < tok.size();) {
                const unsigned char c = tok[k];
                const size_t w = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
                uint32_t cp = w == 1 ? c : (c & (0xFF >> (w + 1)));
                for (size_t j = 1; j < w && k + j < tok.size(); j++) cp = (cp << 6) | (tok[k + j] & 0x3F);
                bytes += inv.count(cp) ? 1 : w;
                k += w;
            }
            lens[i] = static_cast<uint8_t>(std::min<size_t>(bytes, 255));
        }
        this->byte_len_ = std::move(lens);
    });
    if (id < 0 || static_cast<size_t>(id) >= byte_len_.size()) throw std::runtime_error("token id outside the vocabulary");
    return byte_len_[id];
}
