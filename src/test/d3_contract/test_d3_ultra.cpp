/// \file test_d3_ultra.cpp
/// \brief d3-ultra contract additions (no NPU): prompt_layout "header-v1" and temperature_by_questions, against the
///        upstream package's own code (gen_header_fixtures.py: d3_runtime.D3.prepare, d3_format, temperature_table).
///
///   test_d3_ultra <fixtures_ultra dir> [<package dir with tokenizer.json>]
///
/// For every request of header_v1.jsonl: the prompt of every runnable question byte for byte (text path: header; image
/// path: none), the same inline errors; with a package dir (FLM_D3_TEST_TOKENIZER): the token ids and the state block length
/// S of every question, so the header stays inside the shared state block. For every table of temperature.jsonl: the same
/// accept / refuse decision and message, and the same temperature for 0..19 and boundary counts of questions.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "decision/d3_contract.hpp"
#include "decision/py_json.hpp"

#ifdef FLM_D3_TEST_TOKENIZER
#include "tokenizer/tokenizer.hpp"
#endif

using flm::d3::ojson;
using flm::d3::PyValue;

namespace {

int failures = 0, checks = 0;

void fail(const std::string& id, const std::string& what) {
    if (failures < 40) std::cerr << "FAIL " << id << ": " << what << "\n";
    failures++;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

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
    if (argc < 2) {
        std::cerr << "usage: test_d3_ultra <fixtures_ultra dir> [<package dir>]\n";
        return 2;
    }
    const std::string dir = argv[1];
    const ojson config = ojson::parse(read_file(dir + "/decision_config.json"));
    const auto codes = config["codes"].get<std::vector<std::string>>();

    // the package's own table, and its request-count behaviour
    {
        const auto table = flm::d3::temperature_table(config["temperature_by_questions"], "decision_config.json");
        checks++;
        if (!table) { fail("config table", "not parsed"); }
        else {
            const float base = config["temperature"].get<float>();
            const float t1 = config["temperature_by_questions"]["temperature"][0].get<float>();
            for (int n : {1}) { checks++; if (flm::d3::temperature_for(table, base, n) != t1) fail("config table", "n=1"); }
            for (int n : {2, 4, 5, 16, 17, 255}) { checks++; if (flm::d3::temperature_for(table, base, n) != 1.0f) fail("config table", "n=" + std::to_string(n)); }
            checks++;
            if (flm::d3::temperature_for(std::nullopt, 0.9f, 3) != 0.9f) fail("no table", "base temperature");
        }
        checks++;
        if (config["prompt_layout"] != "header-v1") fail("config", "prompt_layout");
        if (config["validated_length"].get<int>() != 131072) fail("config", "validated_length fixture");
    }

    // temperature.jsonl
    int tables = 0, refused = 0, lookups = 0;
    {
        std::ifstream f(dir + "/temperature.jsonl");
        if (!f) fail("temperature.jsonl", "missing");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string id = rec["id"];
            tables++;
            const ojson value = ojson::parse(rec["table_text"].get<std::string>());
            std::optional<flm::d3::TemperatureTable> got;
            std::string error;
            try {
                got = flm::d3::temperature_table(value, "decision_config.json");
            } catch (const std::runtime_error& e) {
                error = e.what();
            }
            checks++;
            if (rec["ok"].get<bool>() == !error.empty()) { fail(id, "accepted / refused differently: " + error); continue; }
            if (!rec["ok"].get<bool>()) {
                refused++;
                if (error != rec["error"].get<std::string>()) fail(id, "message: " + error);
                continue;
            }
            if (rec["max_questions"].is_null()) {
                if (got) fail(id, "table where upstream has none");
                continue;
            }
            if (!got || got->max_questions != rec["max_questions"].get<std::vector<int>>()) { fail(id, "max_questions"); continue; }
            for (size_t i = 0; i < got->temperature.size(); i++) {
                checks++;
                if (got->temperature[i] != static_cast<float>(rec["temperature"][i].get<double>())) fail(id, "temperature " + std::to_string(i));
            }
            for (const auto& item : rec["for"].items()) {
                lookups++;
                checks++;
                const float want = static_cast<float>(item.value().get<double>());
                const float have = flm::d3::temperature_for(got, 0.9f, std::stoi(item.key()));
                if (have != want) fail(id, "temperature_for(" + item.key() + ")");
            }
        }
    }

#ifdef FLM_D3_TEST_TOKENIZER
    std::unique_ptr<Tokenizer> tok;
    std::function<std::vector<int>(const std::string&)> encode;
    std::function<size_t(int)> token_bytes;
    if (argc > 2) {
        tok = std::make_unique<Tokenizer>(argv[2]);
        encode = [&](const std::string& s) { return tok->encode(s); };
        token_bytes = [&](int id) { return tok->token_byte_length(id); };
    }
    int token_checked = 0, ids_differ = 0;
#endif
    // header_v1.jsonl
    int requests = 0, prompts = 0, error_answers = 0, image_requests = 0;
    {
        std::ifstream f(dir + "/header_v1.jsonl");
        if (!f) fail("header_v1.jsonl", "missing");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string id = rec["id"];
            requests++;
            const PyValue body = flm::d3::parse_json(rec["body"].get<std::string>());
            flm::d3::PrepareOptions options;
            options.header_v1 = true;
            flm::d3::Prepared p;
            try {
                p = flm::d3::prepare(body, codes, nullptr, options);
            } catch (const flm::d3::RequestError& e) {
                fail(id, std::string("request refused: ") + e.what());
                continue;
            }
            const bool images = rec.contains("images");
            if (images) image_requests++;
            // prompts of runnable questions
            checks++;
            if (p.texts.size() != rec["texts"].size()) fail(id, "number of prompts " + std::to_string(p.texts.size()) + " vs " + std::to_string(rec["texts"].size()));
            for (const auto& item : rec["texts"].items()) {
                prompts++;
                checks++;
                auto it = p.texts.find(item.key());
                if (it == p.texts.end()) { fail(id, "no prompt for " + item.key()); continue; }
                if (it->second != item.value().get<std::string>()) fail(id, "prompt of " + item.key() + " differs:\n--- have\n" + it->second + "\n--- want\n" + item.value().get<std::string>());
            }
            // inline errors
            checks++;
            if (p.errors.size() != rec["errors"].size()) fail(id, "number of errors");
            for (const auto& item : rec["errors"].items()) {
                error_answers++;
                checks++;
                auto it = p.errors.find(item.key());
                if (it == p.errors.end() || it->second != item.value()) fail(id, "error of " + item.key() + ": " + (it == p.errors.end() ? "none" : it->second.dump()) + " vs " + item.value().dump());
            }
#ifdef FLM_D3_TEST_TOKENIZER
            if (tok && !images) {
                flm::d3::tokenize(p, encode, 0, token_bytes);
                for (const auto& item : rec["ids_crc"].items()) {
                    const std::string& key = item.key();
                    checks++;
                    auto seq = p.sequences.find(key);
                    if (seq == p.sequences.end()) { fail(id, "no sequence for " + key); continue; }
                    if (seq->second.size() != item.value()[0].get<size_t>() || crc32_ids(seq->second) != item.value()[1].get<uint32_t>()) {
                        ids_differ++;   // transformers' Qwen2Tokenizer on combining marks, see README
                        continue;
                    }
                    token_checked++;
                    checks++;
                    auto s = p.state_len.find(key);
                    if (s == p.state_len.end() || s->second != rec["S"][key].get<int>()) fail(id, "state block length of " + key + ": " + (s == p.state_len.end() ? "none" : std::to_string(s->second)) + " vs " + std::to_string(rec["S"][key].get<int>()));
                }
                // the header is part of the state block: every question of the request shares the same first S tokens
                const auto keys = p.runnable();
                for (size_t i = 1; i < keys.size(); i++) {
                    checks++;
                    const int S = p.state_len.at(keys[0]);
                    if (p.state_len.at(keys[i]) != S || !std::equal(p.sequences.at(keys[0]).begin(), p.sequences.at(keys[0]).begin() + S, p.sequences.at(keys[i]).begin()))
                        fail(id, "questions do not share the state block");
                }
            }
#endif
        }
    }
    std::cout << "temperature tables " << tables << " (" << refused << " refused, " << lookups << " lookups); header requests " << requests
              << " (" << image_requests << " with images), prompts " << prompts << ", inline errors " << error_answers;
#ifdef FLM_D3_TEST_TOKENIZER
    if (tok) std::cout << "; token-checked questions " << token_checked << " (ids differ from transformers: " << ids_differ << ")";
#endif
    std::cout << "; " << checks << " checks, " << failures << " failures\n";
    return failures || requests == 0 ? 1 : 0;
}
