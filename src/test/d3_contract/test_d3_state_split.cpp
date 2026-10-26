/// \file test_d3_state_split.cpp
/// \brief prefix_bidir state block length: FastFlowLM's tokenizer (token ids only, no offsets API) against the
///        upstream rule (tokenizer offset mapping, d3_format.encode_split) on the prompts of state_split.jsonl.
///
///   test_d3_state_split <state_split.jsonl> <package dir with tokenizer.json> [images.jsonl]
///
/// For every prompt: the token ids must equal upstream's (crc) and state_block_length() must give upstream's S.
/// Also exercises the end-to-end path of tokenize() (image expansion, state_len, usage) on the same texts.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "decision/d3_contract.hpp"
#include "decision/py_json.hpp"
#include "tokenizer/tokenizer.hpp"

using flm::d3::PyValue;
using nlohmann::json;

namespace {
uint32_t crc32_ids(const std::vector<int>& ids) {
    uint32_t crc = 0xFFFFFFFFu;
    for (int id : ids) {
        const uint32_t v = static_cast<uint32_t>(id);
        for (int b = 0; b < 4; b++) {
            crc ^= (v >> (8 * b)) & 0xFF;
            for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_d3_state_split <state_split.jsonl> <package dir> [images.jsonl]\n";
        return 2;
    }
    Tokenizer tok(argv[2]);
    const auto encode = [&](const std::string& s) { return tok.encode(s); };
    const auto token_bytes = [&](int id) { return tok.token_byte_length(id); };
    std::ifstream in(argv[1]);
    std::string line;
    int checked = 0, images = 0, ids_differ = 0, upstream_checked = 0, id_mismatch = 0, s_mismatch = 0, straddlers = 0, nfc_shifted = 0;
    while (std::getline(in, line)) {
        const json rec = json::parse(line);
        const std::string text = rec["text"];
        const PyValue state = flm::d3::parse_json(rec["state"].get<std::string>());
        const std::vector<int> ids = encode(text);
        checked++;
        if (rec["images"].get<int>() > 0) images++;
        if (ids.size() != rec["n_ids"].get<size_t>() || crc32_ids(ids) != rec["ids_crc"].get<uint32_t>()) {
            if (id_mismatch++ < 10) std::cerr << "ids differ for state " << rec["state"] << "\n";
            continue;
        }
        const int want = rec["S"];   // from tokenizer.json's own offsets (== upstream's S unless ids_differ)
        if (rec["ids_differ"].get<bool>()) ids_differ++; else upstream_checked++;
        int got = -1;
        try {
            got = flm::d3::state_block_length(text, state, ids, encode, token_bytes, 0, static_cast<int>(ids.size()));
        } catch (const std::exception& e) {
            std::cerr << "state " << rec["state"] << ": " << e.what() << "\n";
        }
        if (got != want) {
            if (s_mismatch++ < 20) std::cerr << "S differs for state " << rec["state"] << ": got " << got << " want " << want << "\n";
        }
        // a token straddling the boundary exists when the next token starts before the end of the state text
        size_t sum = 0;
        for (int i = 0; i < want; i++) sum += token_bytes(ids[i]);
        if (sum != flm::d3::state_end_byte(text, state)) straddlers++;
        size_t total = 0;
        for (int id : ids) total += token_bytes(id);
        if (total != text.size()) nfc_shifted++;
    }
    std::cout << "prompts " << checked << " (images " << images << "), id mismatches " << id_mismatch
              << ", S mismatches " << s_mismatch << ", boundary-straddling prompts " << straddlers
              << ", NFC-changed prompts " << nfc_shifted << "; checked against upstream's own S: "
              << upstream_checked << ", prompts where transformers' Qwen2Tokenizer tokenizes differently from tokenizer.json "
              << "(combining marks): " << ids_differ << "\n";
    // image requests (images.jsonl, optional third argument): tokenize() expands each <|image_pad|> and
    // must place every image token inside the state block, none in a question block
    int bad_images = 0, image_prompts = 0;
    if (argc > 3) {
        std::ifstream f(argv[3]);
        const std::string images_path = argv[3];
        std::ifstream cfg(images_path.substr(0, images_path.find_last_of('/')) + "/decision_config.json");
        const std::vector<std::string> codes = json::parse(cfg)["codes"].get<std::vector<std::string>>();
        std::string l2;
        while (std::getline(f, l2)) {
            const json rec = json::parse(l2);
            if (!rec.contains("ids") || rec["ids"].empty() || rec.value("stage", std::string()) == "decode") continue;
            flm::d3::Prepared p;
            try {
                p = flm::d3::prepare(flm::d3::parse_json(rec["body"].get<std::string>()), codes);
            } catch (const flm::d3::RequestError&) {
                continue;
            }
            flm::d3::tokenize(p, encode, 0, token_bytes);
            const std::vector<int> pad = encode(flm::d3::IMAGE_TOKEN);
            for (const auto& key : p.runnable()) {
                image_prompts++;
                const auto& seq = p.sequences.at(key);
                const int S = p.state_len.at(key);
                bool ok = S > 0 && static_cast<size_t>(S) < seq.size();
                for (size_t i = 0; ok && i < seq.size(); i++) {
                    if (seq[i] == pad[0] && static_cast<int>(i) >= S) ok = false;   // an image token in a question block
                }
                // the question block is the prompt text's own tail: it starts where the unexpanded state ends
                const auto plain = encode(p.texts.at(key));
                const int media = static_cast<int>(seq.size() - plain.size());
                ok = ok && std::vector<int>(seq.begin() + S, seq.end()) == std::vector<int>(plain.begin() + (S - media), plain.end());
                if (!ok && bad_images++ < 10) std::cerr << "image request " << rec["id"] << " question " << key << ": bad state block\n";
            }
        }
        std::cout << "image prompts " << image_prompts << ", bad state blocks " << bad_images << "\n";
    }
    return (id_mismatch || s_mismatch || bad_images || checked == 0) ? 1 : 0;
}
